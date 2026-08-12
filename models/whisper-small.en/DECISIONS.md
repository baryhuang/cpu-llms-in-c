# Whisper small.en decision record

This file records the reasoning that selected the deployable model path. It is not a performance claim.

## Product contract

| Item | Decision |
|---|---|
| Input | arbitrary English speech, not a fixed phrase set |
| Output | arbitrary English transcript with ordered timestamps |
| Deployment | continuous local transcription on Amlogic A113X, four Cortex-A53 cores |
| Runtime | dependency-free C; existing runtimes are references only |
| Throughput gate | sustained `RTF <= 1.0` against original audio duration |
| Memory gate | peak RSS below 1 GiB and zero swap |
| Quality gate | no more than 10% relative WER increase against the pinned unmodified `small.en` baseline |

The WER gate is relative. If the unmodified baseline WER is 10.0%, the derived artifact must remain at or below 11.0%. It does not permit a ten-point absolute increase.

## Model selection

| Candidate | Parameters | Encoder / decoder | Width | Role on A113X |
|---|---:|---:|---:|---|
| `tiny.en` | 39M | 4 / 4 | 384 | measured real-time speed floor; lower quality baseline |
| `base.en` | 74M | 6 / 6 | 512 | intermediate quality/speed control |
| `small.en` | 244M | 12 / 12 | 768 | selected quality baseline and deployment challenge |
| `medium.en` | 769M | 24 / 24 | 1024 | possible offline English teacher; not a target artifact |
| `large-v3` | 1.55B | 32 / 32 | 1280 | multilingual quality oracle only; not the required starting graph |

The selection moved away from full `large-v3` for two reasons. The product may be explicitly English-only, so multilingual recognition, automatic language identification and speech translation are not required. More importantly, the full large-v3 encoder projects to roughly 50 RTF on this target before decoder work. The problem is therefore not fitting one large checkpoint; it is retaining small-level English quality while changing the whole model/runtime/CPU stack.

The selected artifact is `small.en`-derived. Structural changes produce a new model identity. It must not be reported as unmodified OpenAI `small.en`.

## Teacher compatibility

| Teacher | Direct token-level distillation into `small.en` | Reason |
|---|---|---|
| `small.en` itself | yes | exact teacher for self-distillation and compression recovery |
| `medium.en` | yes | same 80-bin front end, English tokenizer family and timestamp grammar |
| `large-v3` | no, not without a mapping | 128-bin front end and multilingual tokenizer; token IDs/logit rows are not aligned with `small.en` |

`large-v3` can still create English pseudo-transcripts. It cannot be substituted into a per-token KL loss by comparing equally numbered logit rows. `medium.en` is the cleaner larger English teacher when logit and intermediate-state distillation are required.

## What English-only removes

| Capability | Status |
|---|---|
| Free English transcription | retained |
| Punctuation, no-speech token and timestamps | retained |
| Non-English transcription | removed |
| Automatic spoken-language identification | removed |
| Non-English speech translation into English | removed |
| Speaker diarization | not native to any selected Whisper checkpoint |

Changing `small.en` to `tiny.en` or `base.en` does not remove another interface feature. It primarily reduces recognition accuracy and robustness under noise, far-field speech, accents, unclear speech, overlapping speakers, names, medication terms and numbers.

## Feasibility boundary

The same A113X board measured `tiny.en` Q5_1 greedy at 227.10 seconds for 300 seconds of public looped audio: RTF 0.757, peak RSS 223,000 KiB and zero swap. The encoder used 168.49 seconds. This fixture is a speed reference, not a WER corpus.

At unchanged dimensions, the small encoder's first-order dense-work ratio to tiny is:

```text
(12 / 4) * (768 / 384)^2 = 12
```

The unmodified small target baseline has not been measured. Its expected RTF is an analytical range only. Replacing a framework with C does not by itself supply the required order-of-magnitude gain.

## Speed budget, not a result

The following ranges are research budgets. They are not measured on this repository's `small.en` artifact, and factors do not multiply cleanly.

| Layer | Intended mechanism | Design range |
|---|---|---:|
| Exact runtime and CPU axis | static shapes/arena, packed A53 weights, NEON, fusion and fixed four-core work | 2–3x |
| Decoder structure | 12 to 4 merged/distilled layers | 1.2–1.5x whole-model effect |
| Encoder structure | 12 to 8 layers plus activation-guided low-rank projections | 1.7–2.5x encoder effect |
| Optional temporal structure | compact late encoder frames while retaining original-time spans | 1.3–1.8x affected encoder work |

Only the measured cumulative RTF table decides whether a stage lands. A projected factor is removed once the corresponding target measurement exists.

## Incremental implementation and research plan

| Stage | Axis | Change | Required gate |
|---|---|---|---|
| S0 | reference | unmodified `small.en` F16 and quantized target runs | transcript, WER, timestamps, durations, CPU, RSS and swap recorded |
| S1 | model | exact 80-bin Mel, encoder stem, 12 encoder and 12 cached decoder layers in portable C | boundary comparison with the pinned OpenAI implementation |
| S2 | CPU | fixed arena, packed A53 matrices, NEON kernels, fused residual/normalization/GELU, static four-core schedule | output tolerance and same quality as S1 |
| S3 | model | mixed Q4/Q5/Q6/Q8 selected per layer using calibration data | relative WER increase remains within the stage budget |
| S4 | model | decoder 12 to 4 through layer merging and teacher/student loss | full transcript and timestamp suite; not token-ID agreement |
| S5 | model | encoder 12 to 8 plus per-layer activation-guided low-rank factors | cumulative relative WER increase at most 10% |
| S6 | model | optional late-encoder frame compaction with original-time span metadata | timestamp and short-word gates plus RTF gain on continuous speech |

The planned student keeps the full English tokenizer, output vocabulary and timestamp grammar. It is not a classifier and is not restricted to benchmark phrases.

## Structural training rule

Structured reduction and distillation are combined:

```text
loss = transcript CE
     + teacher/student logit KL
     + encoder intermediate-state loss
     + final encoder-state loss
     + timestamp-token loss
     + measured A113X latency penalty
```

Deleted layers, heads or channels are physically absent from the packed image and generated C schedule. Runtime masks and large matrices containing zeros do not qualify as an optimization result.

Decoder layers are merged before deletion is considered. Encoder layer count and low-rank dimensions are searched with measured target latency rather than FLOPs alone. Frame compaction is optional and follows the exact full-resolution implementation because it changes timestamp behavior.

## Benchmark separation

Correctness verification and timed benchmarks are separate sections in every report. The headline RTF always uses original audio duration, even when VAD or frame compaction reduces internal work. VAD does not establish the continuous-speech gate.

The quality suite must include public or authorized English data covering clean speech, noise, far-field speech, older speakers, television/background speech, silence, numbers, names and care terminology. A looped JFK recording cannot establish the 10% quality gate.

## External method evidence

These sources motivate experiments; none supplies an A113X result for this repository.

| Source | Fact used here |
|---|---|
| [OpenAI Whisper model table](https://github.com/openai/whisper#available-models-and-languages) | model sizes, English-only versus multilingual capability and the stated English advantage of `.en` checkpoints |
| [Distil-Whisper](https://arxiv.org/abs/2311.00430) and its [training code](https://github.com/huggingface/distil-whisper/blob/main/training/run_distillation.py) | pseudo-label filtering and combined transcript cross-entropy plus teacher/student logit KL |
| [LiteASR](https://arxiv.org/abs/2502.20583) | activation-calibrated low-rank encoder projections and low-dimensional attention |
| [BaldWhisper](https://arxiv.org/abs/2510.08599) | layer merging and head shearing as alternatives to deleting layers without recovery |
| [Structured ASR pruning](https://arxiv.org/abs/2305.19549) | learned module masks plus layerwise distillation |
| [Whisper structured sparsity](https://arxiv.org/abs/2510.12666) | ordinary structured sparsity produces moderate FLOP reductions, not the complete A113X speed target |
