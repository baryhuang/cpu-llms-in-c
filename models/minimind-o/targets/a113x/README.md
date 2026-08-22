# MiniMind-O on Amlogic A113D/A113X

This target is the first native-C MiniMind-O speech-to-speech prototype for
the ThirdReality TRHub-V3. The board reports `amlogic,a113d` and
`amlogic,meson-axg`; its application CPU is four Arm Cortex-A53 cores. The
runtime does not embed or launch Python.

## Resident pipeline

```text
USB microphone, 16 kHz PCM
        |
        v
adaptive RMS VAD + pre-roll + bounded capture queue
        |
        v
frozen SenseVoice encoder --> 768-wide audio embeddings
        |
        v
MiniMind Thinker bridge --> Talker --> 8 Mimi codebooks/frame
                                      |
                                      v
                          stateful Mimi decoder worker
                                      |
                                      v
                         24 kHz PCM WAV --> USB speaker
```

All model stages are resident. Capture runs in a dedicated pthread so ALSA is
drained during inference and playback. The 64 x 512-sample bounded queue drops
the oldest input if inference falls behind and is flushed after a turn; stale
speaker audio is therefore not interpreted as a new command.

## Stateful Mimi streaming

`minimindo_mimi_stream_decode` accepts only newly generated code frames. Its
state contains:

- Transformer key/value cache at the global Mimi position;
- two pending samples per channel from the semantic 2x transposed upsampler;
- `kernel - 1` inputs for every causal convolution;
- `kernel - stride` pending outputs for every causal transposed convolution.

For a transposed convolution, each new input contributes `kernel` output taps.
The first `stride` outputs are emitted and the remaining
`kernel - stride` values are retained for the next call. This makes one-frame
streaming equivalent to whole-sequence decode without recomputing a prefix.

The live runtime starts a Mimi pthread before text/audio generation. While
Thinker and Talker are active, the worker uses one OpenMP thread so it can use
otherwise idle CPU without oversubscribing the four A53 cores. Once the
producer reaches EOS, the worker uses four threads to drain its queue.

Production playback is deliberately buffered. On the measured target, eight
Mimi frames represent 0.64 seconds of audio but still require 1.89 seconds to
decode after optimization. Eager playback would consume an 80 ms frame faster
than the next frame can be produced and would create dropouts. Streaming is
used now to overlap decode with generation and remove final cold work; direct
frame-at-a-time playback remains gated on decoder throughput exceeding
real-time.

## Cortex-A53 kernels

The packed images use row-wise Q8 weights with an f32 scale. The hot kernels
use Armv8.0 NEON, not dot-product or i8mm instructions unavailable on the
Cortex-A53:

- Q8 x f32 GEMV widens int8 values through int16/int32 and performs four f32
  FMAs per vector;
- causal convolution builds each time window once in contiguous im2col order,
  then reuses it across output channels as a long NEON Q8 x f32 dot product;
- transposed convolution updates 8 or 4 adjacent taps per NEON operation;
- OpenMP statically partitions matrix rows, convolution output/time tiles and
  activation ranges across four cores.

The im2col change trades about 11 MiB of temporary memory on the eight-frame
fixture for contiguous vector access and reuse. It reduced Mimi decode from
7.80 to 1.89 seconds on the board.

## Correctness gates

The pinned eight-frame fixture contains all eight codebooks and produces
15,360 samples at 24 kHz.

| Gate | Result |
|---|---:|
| Optimized whole decode vs pre-optimization WAV | 6 differing bytes out of 30,720 PCM bytes |
| Optimized one-frame stream vs optimized whole decode | 9 differing bytes out of 30,720 PCM bytes |
| Whole-decode RMS | 0.0925718303 |
| Streaming RMS | 0.0925718283 |
| Whole-decode peak | 0.710567653 |
| Streaming peak | 0.710567594 |

The byte differences are f32 accumulation-order rounding at roughly one PCM
least-significant bit. Frame count, sample count and audible waveform are
preserved.

## A113D measurements

The raw record is in [results.json](results.json). Important measured values:

| Test | Wall | CPU | Peak RSS | Relative |
|---|---:|---:|---:|---:|
| Full 16-step speech, 1 OpenMP thread | 40.58 s | 98% | 391,448 KiB | 1.00x |
| Full 16-step speech, 4 OpenMP threads | 13.64 s | 307% | 391,080 KiB | 2.98x |
| Mimi 8-frame whole decode, old convolution layout | 7.80 s | 347% | 59,056 KiB | 1.00x |
| Mimi 8-frame whole decode, NEON/im2col | 1.89 s | 347% | 70,300 KiB | 4.13x |
| Mimi 8-frame, one frame per streaming call | 1.93 s | 347% | 48,700 KiB | 4.04x |

The four-thread A/B uses the same input, seed, 16 generation steps, eight
audio frames and identical answer text. Four cores are genuinely used; the
remaining gap from 400% comes from serial sampling/attention sections, short
parallel regions and shared L2/memory bandwidth.

The optimized Mimi profiler for eight frames reports:

| Stage | Time |
|---|---:|
| Codebook projections | 6.783 ms |
| Upsampler + Mimi Transformer | 268.359 ms |
| Initial causal convolution | 27.689 ms |
| Decoder stage 1 | 147.704 ms |
| Decoder stage 2 | 310.648 ms |
| Decoder stage 3 | 549.765 ms |
| Decoder stage 4 | 521.444 ms |
| Final convolution | 44.715 ms |
| Total | 1,877.109 ms |

Mimi is still the largest output-side cost and is not real-time. The next
kernel work should target stages 3 and 4, then the SenseVoice encoder and
Thinker prefill. The current system is substantially faster but does not meet
a 1--3 second conversational latency target.

Two live microphone turns on the optimized resident service both generated 48
steps and 40 Mimi frames. They completed inference in 18.916 and 21.405
seconds; Mimi drain was 8.735 and 10.670 seconds, and both ALSA playbacks
returned zero. The variation follows input duration, model scheduling and
thermal/resource state. These are production-path observations, not a claim
that the system has reached normal conversational latency.

## Service and monitoring

The deployed unit is `threehub-minimindo-native.service`. Follow activity with:

```sh
ssh root@100.123.75.40 \
  'journalctl -fu threehub-minimindo-native.service'
```

Successful native turns include `"stream_mimi":true` and stage fields
`audio_encode_ms`, `prefill_ms`, `generate_ms`, and `mimi_drain_ms`.
