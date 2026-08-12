# Target: Amlogic A113X — Whisper small.en

Status: the from-scratch C path now accepts an arbitrary mono PCM16 16 kHz WAV, runs the complete `small.en` encoder and cached decoder, and emits English text. The first public end-to-end smoke case is correct after word normalization. It takes 589.056 seconds for 11 seconds of audio, so the unchanged model fails the real-time gate by 53.55x. The `<10%` relative-WER gate remains open because one sample is not a quality suite.

## Target and artifact

| Model | CPU | SoC year | Maximum observed clock | RAM | SIMD |
|---|---|---:|---:|---:|---|
| OpenAI Whisper `small.en` | 4x Arm Cortex-A53 | 2017 | 1.416 GHz | 2,059,239,424 bytes | NEON/ASIMD and FMA; no dotprod, i8mm or SVE |

| Artifact | Format | Quantization | Size | SHA-256 |
|---|---|---|---:|---|
| Full transcription image | `WHSENC01` v5, 483 tensors | 72 encoder matrices Q4; 121 decoder/tied-head matrices Q8; remaining tensors F32 | 214,878,912 bytes | `8f34d629c5ec4b8a3af4c9de03252fd60ab08781538768c3a5171f6a74b82343` |
| Earlier encoder-only image | `WHSENC01` v2, 188 tensors | 72 encoder matrices Q4; remaining tensors F32 | 56,763,776 bytes | `4ac8031884f1cc4d8eae215507713bf750633c793902f6f1ec095b82d91201d5` |

Weights, packed images and target binaries are excluded from Git.

## Current result

| Metric | Gate | Measured | Status |
|---|---:|---:|---|
| Full ASR RTF against original 11-second audio | `<= 1.0` | **53.5505** | fail |
| End-to-end duration | report | **589.055851 s** | measured |
| Decoder text throughput | report | **1.777 text token/s** | measured |
| Overall text throughput including encoder | report | **0.0441 text token/s** | measured |
| Public JFK smoke-case WER after stated normalization | `0` for this case | **0/22 = 0%** | pass for this case only |
| Relative WER increase over an evaluation suite | `<= 10%` | unavailable | open |
| Ordered timestamps | required final output | no-timestamps decoding selected | open |
| Peak RSS | `< 1 GiB` | 327,820 KiB | pass |
| Process swap | `0` | 0 | pass |
| Device swap change | `0` | 0 KiB | pass |

The input is 11 seconds. The current implementation zero-pads it to Whisper's fixed 30-second window. RTF is total duration divided by the original 11-second duration, not the padded duration. The benchmark is one invocation with no warm-up or repeated-sample statistics.

## Benchmark

Benchmark duration excludes verification. Raw data for the current end-to-end case is in [`benchmarks/jfk-11s/`](benchmarks/jfk-11s/); the compact reviewer is [`REVIEW.html`](benchmarks/jfk-11s/REVIEW.html).

### End-to-end public audio

| Input | Front end | Encoder | Decoder core | Output head | Total | CPU | Peak RSS | Swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `jfk.wav`, 11.000 s | 0.216886 s | 574.188459 s | 14.188894 s | 0.439936 s | **589.055851 s** | 386% | 327,820 KiB | 0 |

The encoder consumes 97.48% of total duration. Improving the decoder alone cannot reach real time on this graph.

### Decoder CPU increment

Both rows use the same three-step independent decoder fixture and produce the same next-token IDs `[31340, 685, 50256]`. Duration is cross-attention cache plus three decoder steps plus three tied output-head evaluations.

| Stage | Cumulative implementation | Threads | Duration | Increment | CPU | Peak RSS | Swap |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | Q8 decoder matrices + Cortex-A53 NEON dot | 1 | 0.798079 s | 1.00x | 98% | 155,008 KiB | 0 |
| D1 | D0 + threaded cross-cache, projections, attention heads and output head | 4 | 0.251960 s | **3.17x** | 359% | 154,624 KiB | 0 |

### Encoder CPU increment

All one-second rows use the same Q4 image, deterministic synthetic log-Mel input, 12 encoder layers and one invocation.

| Stage | Cumulative implementation | Duration | Encoder RTF | Increment | Cumulative | CPU | Peak RSS | Swap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | generic scalar Q4 | 32.716950 s | 32.717 | 1.00x | 1.00x | 98% | 54,784 KiB | 0 |
| A1 | G0 + Cortex-A53 NEON Q4 GEMM with frame reuse | 12.479286 s | 12.479 | 2.62x | 2.62x | 98% | 54,912 KiB | 0 |
| A2 | A1 + four-thread convolution, GEMM, normalization and attention schedule | 3.442861 s | 3.443 | 3.62x | 9.50x | 378% | 54,656 KiB | 0 |
| A3 | A2 + Cortex-A53 NEON full-attention dot/context kernels | 3.156888 s | 3.157 | 1.09x | **10.36x** | 379% | 54,656 KiB | 0 |

G0 and A1 were not run at 30 seconds. Full attention is quadratic in frame count, so their full-window values are not extrapolated.

| Stage | 30-second encoder duration | Encoder RTF | Increment | CPU | Peak RSS | Workspace | Swap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 | 1,465.221181 s | 48.841 | 1.00x | 386% | 125,184 KiB | 64,536,000 bytes | 0 |
| A3 | 556.217968 s | 18.541 | **2.63x** | 382% | 125,056 KiB | 64,536,000 bytes | 0 |

At A3 each encoder block takes 44.67–45.87 seconds. The next material change must reduce encoder structure or frame count; another local Q4-dot optimization cannot supply the remaining factor.

## Verification

Verification is separate from benchmark duration. Quantized activation differences are recorded and are not substituted for WER.

| Boundary | Reference | Result |
|---|---|---|
| F32 real stem | independent NumPy graph, pinned weights | max abs `4.77e-7`, pass |
| F32 real layer 0 | same | max abs `5.36e-7`, pass |
| F32 12 layers + final LayerNorm | same | max abs `3.81e-6`, pass |
| Mixed image decoder, 3 steps | independent NumPy full-prefix causal decoder | expected and actual next tokens match 3/3 on one and four threads |
| Mixed image decoder hidden states | F32 fixture | RMSE `0.02724`, `0.05599`, `0.41680`; finite measurement, not a quality gate |
| Four-thread decoder determinism | same fixture and quantized image | identical tokens and hidden-state metrics to one thread |
| JFK transcription | public reference sentence | normalized word edits `0/22`; one-case smoke result only |

## Reproduce

Compile the full graph image outside Git:

```sh
python3 compiler/compile_whisper_small_encoder_image.py \
  --checkpoint model.safetensors --config config.json \
  --mel-filters mel_filters.npz --tokenizer tokenizer.json \
  --graph full --precision mixed-q4-q8 \
  --output small.en-full-mixed.whenc
```

Build and run on A113X:

```sh
make build/whisper-small-transcribe-a113x \
  CC=gcc CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic' \
  OMPFLAGS=-fopenmp

OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  ./build/whisper-small-transcribe-a113x \
  small.en-full-mixed.whenc samples/jfk.wav 64
```

The executable accepts arbitrary mono PCM16 16 kHz WAV content up to 30 seconds. The last argument is the maximum number of decode steps. Current decoding is English transcription, greedy, batch one and no timestamps.

Machine-readable cumulative measurements are in [`results.json`](results.json). Exact input, command, transcript, resource fields, trace and raw-file hashes are in [`benchmarks/jfk-11s/result.json`](benchmarks/jfk-11s/result.json).
