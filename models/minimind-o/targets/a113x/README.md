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
drained during inference and playback. The 64 x 512-sample bounded SPSC queue
discards new echo while it is full and is flushed after a turn; stale speaker
audio is therefore not interpreted as a new command.

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

The runtime has no batch/non-streaming mode. It always creates the stateful
Mimi consumer before Talker generation. Talker publishes each complete
eight-codebook frame immediately; Mimi consumes queued frames on its dedicated
CPU3 thread while Thinker/Talker use the caller on CPU0 plus persistent workers
on CPUs1--2. After producer EOS, the caller releases that compute session;
Mimi moves to CPU0 and uses all three workers. The overlap is mandatory
pipeline semantics rather than an optional flag.

Every frame is observable in the production journal. `talker_produce` records
queue insertion, `mimi_decode` records PCM publication, and `alsa_write`
records delivery to the ALSA pipe. The final JSON separately reports
`decode_overlapped_with_generation`, `decode_overlap_frames`,
`decoder_to_alsa_streaming`, and `end_to_end_streaming`; it no longer calls all
of these distinct behaviors `playback_streaming`.

Production is token-to-code-to-PCM-to-ALSA streaming. The decoder release-
publishes each 1,920-sample/80 ms PCM frame and playback begins at a fixed
two-frame/160 ms low watermark; it never waits for producer EOS or final
response length. Playback opens `aplay` in raw 24 kHz mono S16 mode and writes
each published frame directly. The response WAV is an archive and is not on
the playback path. Inference and model execution are native C11 and launch no
Python. `aplay` remains the small external ALSA transport process. `SIGPIPE` is
ignored and ALSA/write failure is reported instead of terminating the service.

This is real end-to-end streaming, but it does not claim real-time codec
throughput: one-core Mimi overlap is about 0.29--0.35 s per 80 ms frame and the
four-core drain is commonly 0.10--0.14 s per frame. `queue_waits`,
`queue_wait_ms`, per-frame RTF and ALSA errors expose underruns instead of
hiding them behind a final-response buffer. Reducing Mimi below 80 ms/frame is
still the main continuous-playback optimization target.

## Thread ownership and lifetime

The production lifecycle has explicit single-writer ownership:

- the capture pthread exclusively enqueues ALSA chunks; VAD exclusively
  dequeues them;
- the main thread owns Thinker/Talker state and is the only Mimi-code producer;
- the Mimi pthread exclusively owns the stateful decoder and writes each unique
  PCM range before release-publishing `decoded_frames`;
- the playback pthread reads only published PCM ranges and never touches Mimi
  state;
- every data queue is SPSC and its payload/index handoff is release/acquire;
- mutexes exist only at an empty-queue dequeue sleep / enqueue notification
  boundary. No mutex covers model state, KV/history, memcpy, compute or I/O;
- the main thread signals producer completion, joins playback, joins Mimi, and
  only then destroys queue wait objects and frees PCM.

OpenMP and `libgomp` are absent. Three pthread workers are created once and
pinned to CPUs1--3. Each worker owns one main-to-worker SPSC mailbox; an atomic
epoch publishes a job and an atomic completion epoch acknowledges it. Workers
futex-sleep only between inference sessions and spin on their mailbox while a
session is active, so a matrix dispatch has no mutex, futex, team creation or
scheduler wakeup. The speech phase handoff guarantees one mailbox producer.

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
- the persistent worker pool statically partitions matrix rows, convolution
  output/time tiles and activation ranges across the owned CPU set;
- streaming Mimi causal/deconvolution windows are activation-quantized to i8,
  then evaluated with NEON W8A8 integer dots and per-window/per-row scales.

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
| Historical full 16-step speech, 1 OpenMP thread | 40.58 s | 98% | 391,448 KiB | 1.00x |
| Historical full 16-step speech, 4 OpenMP threads | 13.64 s | 307% | 391,080 KiB | 2.98x |
| Mimi 8-frame whole decode, old convolution layout | 7.80 s | 347% | 59,056 KiB | 1.00x |
| Mimi 8-frame whole decode, NEON/im2col | 1.89 s | 347% | 70,300 KiB | 4.13x |
| Mimi 8-frame, one frame per streaming call | 1.93 s | 347% | 48,700 KiB | 4.04x |
| Mimi 8-frame, phase-packed one-frame streaming | 1.26 s | 318% | 58,844 KiB | 6.19x |

The historical four-thread A/B uses the same input, seed, 16 generation steps, eight
audio frames and identical answer text. Four cores are genuinely used; the
remaining gap from 400% comes from serial sampling/attention sections, short
parallel regions and shared L2/memory bandwidth.

The current no-OpenMP, queue-only-lock build was measured on a deterministic
Chinese prompt (`seed=20260821`, 50 total steps, 40 audio frames):

| Metric | Result |
|---|---:|
| Full model/decode wall | 8.23 s |
| Aggregate CPU | 358% |
| Peak RSS | 212,456 KiB |
| Prefill | 0.910 s / 293% |
| Generation plus overlapping Mimi | 4.021 s / 373% |
| Four-core Mimi drain | 3.120 s / 367% |
| Frames decoded before Talker EOS | 11 |
| Voluntary context switches | 12 |
| Involuntary scheduler preemptions | 6,911 |
| WAV SHA256 | `b61b0662379fa0bb6b3ec304b72bc53000f8e8457649b81bc611f73c48f7289c` |

The WAV hash is identical to the prior mutex-backed queue build. The 12
voluntary switches cover session wake/sleep and process lifecycle rather than
per-matrix dispatch. Generation reaches 373% process CPU because the
three-thread Thinker/Talker group and the one-thread Mimi consumer occupy all
four cores concurrently.

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
the worker count cannot produce 1--3 second latency. With the exact
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
the raw result file as historical measurements. A 24-step fast-path experiment
reduced latency, but was not production-correct: the same limit bounds both
text generation and the staggered eight-codebook audio stream, so every turn
stopped at 16 Mimi frames (1.28 seconds) even when the decoded text was longer.
The listener therefore heard only the first few words. Production now bounds
text at 64 steps, then gives the audio codebooks a separate 192-step/15.36
second drain budget. It exits that budget early as soon as all eight codebooks
reach EOS. In a fixed-seed Chinese test text reached EOS at step 10, while audio
correctly continued until total step 56 and emitted 44 frames/3.52 seconds. A
forced-limit test capped text at 16 steps; it inserted text EOS at step 17 and
continued to total step 75, producing 61 frames/4.88 seconds with
`text_limit_hit=true` and `audio_drain_complete=true`. The prior 24-step speaker
run remains useful as a transport measurement: model generation ended at
6,271 ms, playback began with 12 buffered frames at 8,044 ms, and `aplay`
finished without an underrun at 9,469 ms.

An earlier OpenMP `3 generation threads + 1 overlapping Mimi thread` A/B was
rejected because repeated team entry and passive wakeups raised generation to
8.161 seconds and total model time to 15.258 seconds. The persistent SPSC pool
removes those hot-path wakeups, so the production scheduler now uses exactly
that 3+1 ownership split and then hands all four cores to Mimi at producer EOS.

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

Successful native turns include `"streaming":true`,
`"decode_overlapped_with_generation":true`, per-frame `STREAM` events, and
stage fields `audio_encode_ms`,
`prefill_ms`, `generate_ms`, `mimi_drain_ms`, `first_audio_ms`, and
`streaming_lead_ms`. The four `*_cpu_pct` fields expose effective per-stage CPU
parallelism; values above 100% prove multi-core execution. The production event
order is:

```text
speech_end -> first_audio -> model_end -> playback_end -> inference_end
```

The model directory remains `/dev/shm/minimindo-o-native-v1` because the 6.9
GiB root filesystem had only about 138 MiB free while the six runtime artifacts
require about 377 MiB. `/dev/shm` is erased on reboot. The persistent
`/usr/local/bin/run-minimindo-native-a113x.sh` launcher solves that cold-boot
failure by verifying every artifact against a pinned SHA256 and downloading
only missing or corrupt files. The executable comes from
[`minimindo-native-a113x-v1.1.0`](https://github.com/baryhuang/llm-in-c/releases/tag/minimindo-native-a113x-v1.1.0);
the unchanged packed model images remain pinned to the v1.0.0 release.

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
