# Whisper large-v3

Status: model and deployment contracts are pinned. M1 has started with a scalar C 128-bin log-Mel boundary that passes locally and on A113X. The encoder, decoder and large-v3 A113X artifact are not implemented. No large-v3 throughput is reported by this repository.

This model line compiles local transcription artifacts. Transcript is a primary CareMojo input, not a disposable preprocessing detail. The output is ordered natural language with time boundaries; neither the vocabulary nor the output head may be reduced to a fixed classifier.

## Model pin

| Field | Value |
|---|---|
| Source | OpenAI Whisper `large-v3`, revision `5f86d1d86363843179951550570367b37c5d6f78` |
| Release | 2023-11-06 |
| Parameters | 1,550 M |
| Languages | multilingual; 100 language tokens in the large-v3 checkpoint |
| Input | 16 kHz mono audio; 30-second windows; 128-bin log-Mel spectrogram |
| Encoder | 2 convolution layers, then 32 Transformer blocks; width 1280, 20 heads, audio context 1500 |
| Decoder | 32 autoregressive Transformer blocks with cross-attention; width 1280, 20 heads, text context 448 |
| Vocabulary | 51,866 entries, including language, task and timestamp tokens |
| F16 checkpoint | 3,087,371,615 bytes; SHA-256 in [`pins.json`](pins.json) |
| Converted F16 image | 3,095,033,483 bytes |
| Converted Q5_0 image | 1,081,140,203 bytes |

OpenAI records two architecture changes from large-v2: 128 Mel bins instead of 80 and an added Cantonese token. The upstream model keeps the standard Whisper encoder-decoder graph. Sources: [large-v3 release](https://github.com/openai/whisper/discussions/1762), [reference model](https://github.com/openai/whisper/blob/5f86d1d86363843179951550570367b37c5d6f78/whisper/model.py), and [audio constants](https://github.com/openai/whisper/blob/5f86d1d86363843179951550570367b37c5d6f78/whisper/audio.py).

## Runtime contract

Input and output fields are pinned in [`profile.json`](profile.json).

```text
16 kHz mono audio
        |
        v
ordered transcript segments
{start_ms, end_ms, text, confidence?, speaker?}
```

The following are hard requirements:

| Requirement | Consequence |
|---|---|
| Open natural-language transcript | keep the tokenizer, token embedding and all reachable text rows |
| Event order and time are material | timestamp tokens and segment `start_ms` / `end_ms` remain in the released output |
| Raw audio and transcript are private | target execution is local; public reports use public or synthetic audio only |
| Delayed hazard analysis is allowed | optimize sustained throughput and queue age; real-time first-token latency is not the primary gate |
| Five-minute observed clips | the primary performance unit is a 300-second clip, not only the 11-second JFK sample |

`--no-timestamps` may be used to isolate model compute in a diagnostic run. It is not a valid product configuration or a quality result.

## Separation from the language model

Whisper and Qwen remain separate model artifacts and schedulable stages:

```text
audio -> local Whisper -> ordered timed transcript -> text encoder / Qwen
sensor events -----------------> native temporal encoder ------^
```

The native sensor encoder remains valid. It does not replace the transcript path. On a four-core target, the default policy is sequential: Whisper may use all cores, release them, then the downstream model runs. Concurrent one-core/three-core partitioning is not assumed to preserve throughput.

## Optimization axes

Model-axis work in this file must remain valid on any CPU. CPU/SoC work belongs under `targets/<soc>/`.

### Model-axis incremental stages

Every stage is cumulative. `Increment` is measured against the immediately previous stage with the same model, audio, language, timestamp mode and decoder policy. Estimates are not benchmark results.

| Stage | Type | Cumulative implementation | Expected effect | Required gate | Status |
|---|---|---|---|---|---|
| M0 | reference | pinned OpenAI F16 graph and decoder | numerical and quality oracle | checkpoint hash; reference transcript, logits and times | pinned; not a deployable low-memory baseline |
| M1 | exact contract | C audio front end, tokenizer, 30-second windowing, timestamp grammar, incremental self-KV and per-window cross-KV cache | removes Python and framework state without changing the graph | Mel, encoder, cross-KV, decoder-logit and segment-boundary comparison | in progress — scalar 128-bin log-Mel boundary passes |
| M2 | exact relative to packed weights | static arena; precomputed FFT/window/Mel tables and sinusoidal positions; fused bias/GELU/projection schedules; mmap or direct packed image | bounded memory and fewer dispatches/temporary writes | M1 output/logit tolerances; zero swap | planned |
| M3 | approximate weights | per-tensor and per-layer mixed Q8/Q6/Q5/Q4 search, not one global quantizer | 3.10 GB F16 image toward or below the 1.08 GB Q5_0 reference | WER/CER, language ID, timestamp error and hallucination gates | planned |
| M4 | exact relative to M3 | compile legal-token masks and an exact branch-and-bound tree for top-k output-head search | may avoid scanning all 51,866 rows on easy decoder steps; worst case remains a full scan | top-k token and logit equality to M3 on every validation step | planned experiment |
| M5 | conditional exactness | caller-pinned language/task prefix, cached initial decoder state, and invalid timestamp/token suppression | removes repeated prefix and language-detection work only when the deployment contract fixes it | reject inputs outside the compiled language/task contract | planned |
| M6 | workload approximation | local VAD compacts non-speech before Whisper while retaining original timeline offsets | encoder work scales with detected speech plus padding, not raw clip duration | original-timeline segment recall and timestamp error; report original-audio RTF | planned |
| M7 | decoder approximation | greedy first pass; beam/temperature fallback only for low-confidence or repetition windows | reduces decoder work on confident windows | WER/CER and repetition/hallucination gates by acoustic stratum | planned |
| M8 | architecture approximation | offline teacher-guided structured pruning/distillation of encoder and decoder layers, followed by per-layer precision search | only stage capable of order-of-magnitude compute reduction while retaining large-v3 as teacher | derived artifact gets a new hash and identity; full quality suite rerun | research |

M8 is not labeled “large-v3 unchanged.” It is a derived artifact whose teacher and training/search data are pinned. The full 32+32-layer artifact remains separately identifiable.

## Compile-time search

Compile-time compute is treated as effectively unbounded. The compiler may search independently for every layer:

| Search dimension | Candidate choices |
|---|---|
| Weight format | F16, Q8, Q6, Q5, Q4, mixed outlier rows |
| Quantizer | min/max, MSE, activation-aware, per-channel, group size |
| Layout | row-major, tiled, head-major, interleaved quant blocks |
| Schedule | layer fusion boundary, token block, thread split, prefetch distance |
| Decoder head | full scan, exact bounded tree, coarse index plus exact residual verification |
| Approximate graph | layer retention, head retention, MLP width, teacher loss weights |

The compiler emits a layer manifest. A single label such as “Q5 model” is insufficient because sensitive encoder, cross-attention and decoder tensors may require different formats.

## Why the full model is difficult on A113X

The 1.08 GB Q5_0 image passes only the storage-size test. The model also needs decoder self-KV, cross-KV, working buffers, audio and runtime memory. Fit and zero swap must be measured; file size alone is not a capacity result.

Compute is the larger problem. On the same A113X, the existing `tiny.en` Q5_1 measurement takes about 16.9 seconds for the four-layer, width-384 encoder of one 30-second window. The large-v3 encoder has 32 layers at width 1280. A first-order dense-work ratio is:

```text
(32 / 4) * (1280 / 384)^2 = 88.9
```

That projects about 1,500 seconds of encoder time per 30-second window before decoder work: RTF about 50, or roughly 25 minutes per window. This is an analytical feasibility bound, not a large-v3 benchmark. It establishes that quantization and NEON kernels alone are unlikely to make full large-v3 useful on this target; M6 and especially M8 are required branches.

## Verification and benchmark separation

Verification is excluded from timed benchmark duration.

| Section | Required contents |
|---|---|
| Verification | hashes, Mel/tensor/logit deltas, token equality where exact, WER/CER, language ID, timestamp error, hallucination/repetition cases |
| Benchmark | audio duration, wall duration, RTF, audio/wall speed, encoder and decoder duration, CPU, peak RSS, swap and page faults |
| Review | exact public input identity, human-readable output segments and timestamps; no private CareMojo audio or transcript |

RTF is `wall seconds / original audio seconds`. VAD rows additionally report detected-speech seconds, but their headline RTF still uses original audio duration.

## Records

| Record | Contents |
|---|---|
| [`pins.json`](pins.json) | source revisions, file hashes and model-image hashes |
| [`profile.json`](profile.json) | private local transcript deployment contract and measurement fields |
| [`targets/generic/README.md`](targets/generic/README.md) | portable C implementation boundary |
| [`targets/a113x/README.md`](targets/a113x/README.md) | A113X stage plan and the separate `tiny.en` feasibility measurements |

## Implemented M1 boundary

The first generic C path implements the pinned large-v3 128-bin log-Mel transform with caller-owned memory and no framework dependency. The committed fixture uses synthetic audio and the hashed OpenAI filterbank.

| Run | Values compared | Maximum absolute delta | Result |
|---|---:|---:|---|
| local C vs independent fixture generator | 768 | `1.74045563e-5` | pass |
| A113X C vs the same fixture | 768 | `1.74045563e-5` | pass |

Files: [`whisper_frontend.c`](targets/generic/whisper_frontend.c), [`whisper_frontend.h`](targets/generic/whisper_frontend.h), and [`tests/whisper_log_mel_test.c`](../../tests/whisper_log_mel_test.c). This is a scalar correctness baseline. Its direct DFT is not a performance implementation and no duration is retained as an inference benchmark.
