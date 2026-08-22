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

The live runtime creates the Mimi pthread before generation, but that thread
sleeps until the Talker producer reaches EOS. Thinker/Talker therefore own the
four A53 cores during generation. After EOS, the main OpenMP team sleeps and
Mimi uses a different four-thread team to drain the completed code queue. An
A/B/B/A test rejected one-core Mimi overlap: it created a fifth runnable thread,
made generation about 644 ms slower and saved only about 512 ms in Mimi drain.

Production playback is decoder-to-ALSA streaming, not token-to-speaker
streaming. The decoder publishes each 1,920-sample/80 ms PCM frame under a
pthread condition variable. Playback opens `aplay` in raw 24 kHz mono S16 mode
and writes published frames directly; the response WAV is an archive and is no
longer on the playback path. Inference and model execution are native C11 and
launch no Python. `aplay` remains the small external ALSA transport process.
`SIGPIPE` is ignored and ALSA/write failure is reported instead of terminating
the resident service.

The A53 still decodes below real time: eight frames represent 0.64 seconds but
require 1.22--1.26 seconds. Playback must first know the final response length,
then wait for 75% to be decoded (minimum four frames; responses of four frames
or fewer decode completely). A fixed eight-frame threshold was rejected after
the physical speaker reported a 1.16-second underrun. The dynamic threshold
passed the same 16-frame test with 12 frames/960 ms buffered and no ALSA
underrun. `EVENT first_audio` still occurs before Mimi completion, but only
after model generation has ended.

## Thread ownership and lifetime

The production lifecycle has explicit single-writer ownership:

- the capture pthread exclusively drains ALSA capture into its bounded ring;
- the main thread owns Thinker/Talker state and is the only Mimi-code producer;
- the Mimi pthread exclusively owns the stateful decoder and writes each unique
  PCM range before publishing `decoded_frames` under the mutex;
- the playback pthread reads only published PCM ranges and never touches Mimi
  state;
- the main thread signals producer completion, joins playback, joins Mimi, and
  only then destroys the condition/mutex and frees PCM.

`OMP_WAIT_POLICY=PASSIVE` prevents the inactive team from spinning.
`OMP_PROC_BIND=true` with `OMP_PLACES=cores` keeps the active team on the four
physical cores. Thread sampling observed main + three workers before EOS, then
Mimi + three different workers after EOS; the two compute teams were never
runnable together.

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

Thinker and Talker prefill now run layer-by-layer across the full prompt. This
reuses each Q8 weight row across all prompt positions, computes the Thinker LM
head only at the final prompt token, skips all Talker output heads during
prefill, and computes the fixed pad-codec projection once instead of once per
prompt token. RoPE sine/cosine values are precomputed at model load. For a
37-token prompt, prefill fell from 2.328 seconds to 1.632--1.732 seconds with
identical text and WAV output.

SenseVoice uses a four-position Q8 dot kernel: one weight row feeds four audio
positions before it is evicted. The 20-frame target A/B was 2.31--2.44 seconds
versus 2.52--2.62 seconds for the single-position kernel, with byte-identical
embeddings. Talker top-50 sampling uses a 50-entry min-heap followed by a
50-item sort instead of sorting all 2,112 logits. It removes about 66 ms of
serial work per 24-step fixture and preserves the exact WAV hash.

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

The current 1.18-second input / 24-step / 16-frame production fixture gives a
more useful distribution of the full response path:

| Stage | Wall | Process CPU / wall | Approximate work |
|---|---:|---:|---:|
| SenseVoice + audio projector | 2.275 s | 286% | 4.4B MAC at 20 encoder frames |
| Thinker/Talker batch prefill | 1.769 s | 308% | 2.18B + 1.24B MAC |
| Thinker/Talker generation | 2.147 s | 244% | about 2.2B MAC plus sampling |
| Mimi drain | 2.366 s | 288% | about 166M MAC per output frame |
| Model/decode total | 8.686 s | — | no swap; 427,492 KiB peak RSS |

All four CPUs are active, but no stage sustains 400%. The short transformer
regions have serial attention/sampling gaps, and SenseVoice/Mimi stream large
Q8 weights through the A53 cluster's shared L2. This is why merely increasing
the OpenMP thread count cannot produce 1--3 second latency. With the exact
current graph, the pre-audio floor is already approximately 2.3 s SenseVoice +
1.7 s prefill + 2.1 s generation + 1.8 s to the safe Mimi buffer.

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

Earlier 48-step resident turns took 18.916--21.405 seconds and are retained in
the raw result file as historical measurements. Production now caps the fast
path at 24 steps. On the final physical-speaker fixture, model generation ended
at 6,271 ms, 12 of 16 frames were buffered at 8,044 ms, Mimi completed at
8,659 ms, and playback ended at 9,469 ms. `aplay` returned zero and emitted no
underrun. The four condition waits (614 ms total) happened behind the 960 ms
PCM safety buffer; they did not starve ALSA.

The model is therefore usable as a correctness prototype, but it is not the
1--3 second immediate-dialog path. The product design should keep two paths:
an aggressively bounded tiny safety/intent model for immediate local response,
and this higher-accuracy transcript/reasoning path for notification decisions.
Further exact-graph work should first reduce SenseVoice weight traffic and Mimi
stages 3/4; a separate tiny model is required to change the latency class.

A live 7.52-second utterance demonstrated why the paths must be bounded:
SenseVoice took 63.709 seconds, 142-token prefill took 12.477 seconds, and first
audio arrived at 83.126 seconds. The fast resident service now closes a turn at
approximately three seconds even without detected silence and records
`end=limit`; silence-terminated turns record `end=silence`. Longer recordings
must be handed to the accurate asynchronous path instead of occupying the
interactive process.

## Service and monitoring

The deployed unit is `threehub-minimindo-native.service`. Follow activity with:

```sh
ssh root@100.123.75.40 \
  'journalctl -fu threehub-minimindo-native.service'
```

The unit sets the P10S microphone capture gain to 50% / +8 dB before launch.
At the device's 0 dB setting the endpoint delivered all-zero PCM; +8 dB
produced an idle RMS around 50--70 and speech peaks above 2,000 while leaving
speaker volume unchanged. It also sets `PCM` playback to 5% and explicitly
unmutes it. Although the P10S ALSA descriptor reports raw playback value zero
as 0 dB and switched on, the physical output was silent at that value. A
native 48 kHz stereo endpoint test proved that USB frames were advancing, and
audible playback resumed as soon as the raw control moved to 5%. Keep this
non-zero initialization in the service so USB resets and reboots cannot leave
the endpoint silently accepting samples.

Successful native turns include `"stream_mimi":true`,
`"playback_streaming":true`, and stage fields `audio_encode_ms`,
`prefill_ms`, `generate_ms`, `mimi_drain_ms`, `first_audio_ms`, and
`streaming_lead_ms`. The four `*_cpu_pct` fields expose effective per-stage CPU
parallelism; values above 100% prove multi-core execution. The production event
order is:

```text
speech_end -> model_end -> first_audio -> playback_end -> inference_end
```

The model directory remains `/dev/shm/minimindo-o-native-v1` because the 6.9
GiB root filesystem had only about 138 MiB free while the six runtime artifacts
require about 377 MiB. `/dev/shm` is erased on reboot. The persistent
`/usr/local/bin/run-minimindo-native-a113x.sh` launcher solves that cold-boot
failure by verifying every artifact against a pinned SHA256 and downloading
only missing or corrupt files from the
[`minimindo-native-a113x-v1.0.0` release](https://github.com/baryhuang/llm-in-c/releases/tag/minimindo-native-a113x-v1.0.0).

Downloads use a per-process `.part` file in `/dev/shm`; the launcher verifies
it before an atomic rename. A truncated or incorrect asset is never executed.
Once all six files pass, the launcher `exec`s the native-C binary with the live
production arguments. The unit retries after 30 seconds if the network is
unavailable. A warm service restart performs local SHA checks and downloads
nothing; a board reboot repopulates the volatile directory automatically. The
full empty-cache systemd test downloaded 377 MiB and reached `READY` in about
78 seconds. A subsequent warm restart verified the same files and reached
`READY` in about 6 seconds, including a 539 ms model warm-up.

To verify/download the release without starting inference:

```sh
MINIMINDO_DOWNLOAD_ONLY=1 \
  /usr/local/bin/run-minimindo-native-a113x.sh
```
