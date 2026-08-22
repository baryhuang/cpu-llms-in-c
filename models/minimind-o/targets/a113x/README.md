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
                          decoded-frame condition queue
                                      |
                                      v
                        raw 24 kHz PCM --> ALSA USB speaker
                                      |
                                      +--> response WAV archive
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
- one previous input sample per channel for every causal transposed
  convolution.

All four Mimi transposed convolutions have `kernel == 2 * stride`. For phase
`r`, output step `t` is the dot product of the current input against kernel
phase `r`, plus the previous input against phase `stride + r`. The loader
repacks those two discontiguous Q8 phase rows into one contiguous row. The hot
path then uses the same NEON Q8 dot kernel as GEMV and retains only the previous
input vector. One-frame streaming remains equivalent to whole-sequence decode
without recomputing a prefix.

The live runtime starts a Mimi pthread before text/audio generation. While
Thinker and Talker are active, the worker uses one OpenMP thread so it can use
otherwise idle CPU without oversubscribing the four A53 cores. Once the
producer reaches EOS, the worker uses four threads to drain its queue.

Production playback is now end-to-end streaming. The decoder publishes each
1,920-sample/80 ms PCM frame under a pthread condition variable. Playback opens
`aplay` in raw 24 kHz mono S16 mode and writes frames directly; the response
WAV is an archive and is no longer on the playback path. `SIGPIPE` is ignored
and ALSA/write failure is reported instead of terminating the resident service.

The A53 still decodes below real time: eight frames represent 0.64 seconds but
require 1.26 seconds. Playback therefore starts after 75% of the response is
decoded (minimum four frames, leaving at least one undecoded frame). This
measured safety buffer prevents speaker underruns while the remaining frames
continue decoding. It is real streaming because `EVENT first_audio` occurs
before Mimi completion and `inference_end`; it is not a claim of low-latency
token-to-audio generation.

## Cortex-A53 kernels

The packed images use row-wise Q8 weights with an f32 scale. The hot kernels
use Armv8.0 NEON, not dot-product or i8mm instructions unavailable on the
Cortex-A53:

- Q8 x f32 GEMV widens int8 values through int16/int32 and performs four f32
  FMAs per vector;
- causal convolution builds each time window once in contiguous im2col order,
  then reuses it across output channels as a long NEON Q8 x f32 dot product;
- transposed convolution packs `(current, previous)` phase weights and uses a
  contiguous NEON Q8 dot instead of scalar/scatter output updates;
- OpenMP statically partitions matrix rows, convolution output/time tiles and
  activation ranges across four cores.

The im2col change trades about 11 MiB of temporary memory on the eight-frame
fixture for contiguous vector access and reuse. It reduced Mimi decode from
7.80 to 1.89 seconds. The phase-packed transposed-convolution kernel then
reduced one-frame streaming decode from 1.93 to 1.26 seconds.

## Correctness gates

The pinned eight-frame fixture contains all eight codebooks and produces
15,360 samples at 24 kHz.

| Gate | Result |
|---|---:|
| Phase-packed whole decode vs prior im2col WAV | 16 differing bytes out of 30,720 PCM bytes |
| Phase-packed one-frame stream vs phase-packed whole decode | 11 differing bytes out of 30,720 PCM bytes |
| Whole-decode RMS | 0.0925718284 |
| Streaming RMS | 0.0925718293 |
| Whole-decode peak | 0.710567594 |
| Streaming peak | 0.710567653 |

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
| Mimi 8-frame, phase-packed one-frame streaming | 1.26 s | 318% | 58,844 KiB | 6.19x |

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

The profile above predates the phase-packed transposed-convolution kernel.
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

The final production microphone turn contained 1,184 ms of captured speech and
generated 40 Mimi frames. Model generation ended at 13,161 ms, the first raw
PCM frame reached ALSA at 17,300 ms, Mimi finished at 19,207 ms, playback ended
at 20,605 ms with result zero, and `inference_end` followed at 20,615 ms. Thus
audio delivery began 1,907 ms before decoder completion. The seven queue waits
(1,061 ms total) mean the unbuffered writer caught up with the decoder; they
are not ALSA underrun reports. No ALSA underrun or capture overrun was logged.

## Service and monitoring

The deployed unit is `threehub-minimindo-native.service`. Follow activity with:

```sh
ssh root@100.123.75.40 \
  'journalctl -fu threehub-minimindo-native.service'
```

Successful native turns include `"stream_mimi":true`,
`"playback_streaming":true`, and stage fields `audio_encode_ms`,
`prefill_ms`, `generate_ms`, `mimi_drain_ms`, `first_audio_ms`, and
`streaming_lead_ms`. The production event order is:

```text
speech_end -> model_end -> first_audio -> playback_end -> inference_end
```
