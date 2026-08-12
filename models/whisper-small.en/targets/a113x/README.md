# Target: Amlogic A113X — Whisper small.en

Status: the from-scratch C path accepts an arbitrary mono PCM16 16 kHz WAV, runs the complete `small.en` encoder and cached decoder, and emits English text. Compact-window graph execution and A113X Q4 row/output reuse reduce the 11-second JFK case from 589.056 seconds to a three-run median of 45.047 seconds, a 13.08x cumulative gain. RTF is still 4.095, so real time fails. The `<10%` relative-WER gate remains open because one sample is not a quality suite.

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
| Full ASR RTF against original 11-second audio | `<= 1.0` | **4.0952** median | fail |
| End-to-end duration | report | **45.046989 s** median of 3 | measured |
| Cumulative end-to-end increment | report | **13.08x** vs fixed30 E1 | measured |
| Decoder text throughput | report | **3.834 text token/s** on median run | measured |
| Overall text throughput including encoder | report | **0.555 text token/s** on median run | measured |
| Public JFK smoke-case WER after stated normalization | `0` for this case | **0/22 = 0%** | pass for this case only |
| Relative WER increase over an evaluation suite | `<= 10%` | unavailable | open |
| Ordered timestamps | required final output | no-timestamps decoding selected | open |
| Peak RSS | `< 1 GiB` | 251,276–251,488 KiB over 3 runs | pass |
| Process swap | `0` | 0 | pass |
| Device swap change | `0` | 0 KiB | pass |

The default mode still zero-pads to Whisper's fixed 30-second window. The measured `compact` mode encodes the 1,100 frames present in this 11-second input and produces 550 encoder frames instead of 1,500. RTF is total duration divided by the original 11-second duration. The final headline is the median of three complete process invocations; the coefficient of variation is 1.103%.

## Benchmark

Benchmark duration excludes verification. Raw fixed30 and compact data are in [`benchmarks/jfk-11s/`](benchmarks/jfk-11s/); review artifacts are [`REVIEW.html`](benchmarks/jfk-11s/REVIEW.html) and the [PDF report](../../../../output/pdf/whisper-small-en-a113x-jfk-11s-review.pdf).

### End-to-end public audio

| Mode | Input | Front end | Encoder | Decoder core | Output head | Total | RTF | CPU | Peak RSS | Swap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 fixed30, one run | `jfk.wav`, 11.000 s | 0.216886 s | 574.188459 s | 14.188894 s | 0.439936 s | 589.055851 s | 53.5505 | 386% | 327,820 KiB | 0 |
| E8 compact, run 1 | same | 0.081300 s | 42.223373 s | 6.067983 s | 0.425421 s | 48.808225 s | 4.4371 | 388% | 251,232 KiB | 0 |
| E8 compact, run 2 / median | same | 0.080052 s | 41.625532 s | 6.076246 s | 0.422475 s | **48.214580 s** | **4.3831** | 388% | 251,496 KiB | 0 |
| E8 compact, run 3 | same | 0.080148 s | 41.345683 s | 6.119253 s | 0.428544 s | 47.984083 s | 4.3622 | 388% | 251,448 KiB | 0 |
| E9 row reuse, run 1 | same | 0.084131 s | 39.200258 s | 6.086109 s | 0.429966 s | 45.810786 s | 4.1646 | 380% | 251,276 KiB | 0 |
| E9 row reuse, run 2 | same | 0.081672 s | 38.263763 s | 6.089001 s | 0.427758 s | 44.872162 s | 4.0793 | 388% | 251,488 KiB | 0 |
| E9 row reuse, run 3 / median | same | 0.079188 s | 38.436764 s | 6.094089 s | 0.426832 s | **45.046989 s** | **4.0952** | 388% | 251,396 KiB | 0 |

The encoder consumes 85.33% of the median E9 run. Improving the decoder alone cannot reach real time on this graph.

### End-to-end increment

E1–E3, E8 and E9 correspond to committed feature boundaries. E4–E7 are recorded tile-selection experiments within the Q4 blocking change; their exact intermediate source was not retained as a release commit. E8 and E9 durations are three-run medians; other rows are one run.

| Stage | Cumulative implementation | Duration | RTF | Increment | Cumulative |
|---|---|---:|---:|---:|---:|
| E1 | A3 encoder + D1 decoder, fixed 30-second window | 589.055851 s | 53.5505 | 1.00x | 1.00x |
| E2 | E1 + compact input window | 160.802747 s | 14.6184 | **3.66x** | 3.66x |
| E3 | E2 + 4-column Q4 cache block | 73.774966 s | 6.7068 | **2.18x** | 7.98x |
| E4 | E3 + 8-column block | 61.882272 s | 5.6257 | 1.19x | 9.52x |
| E5 | E4 + 16-column block | 56.771391 s | 5.1610 | 1.09x | 10.38x |
| E6 | E5 + 32-column block | 53.560030 s | 4.8691 | 1.06x | 11.00x |
| E7 | E6 + 64-column block | 52.206838 s | 4.7461 | 1.03x | 11.28x |
| E8 | E7 + fixed 4-output NEON dot; 3-run median | **48.214580 s** | **4.3831** | 1.08x | **12.22x** |
| E9 | E8 + 2-row × 4-output Q4 micro-kernel; 3-run median | **45.046989 s** | **4.0952** | **1.07x** | **13.08x** |

Rejected variants are recorded in [`compact-result.json`](benchmarks/jfk-11s/compact-result.json): shared 8-output accumulators, fixed 6/8-output micro-kernels, GELU lookup tables and a NEON encoder stem all failed the end-to-end duration gate.

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
  small.en-full-mixed.whenc samples/jfk.wav 64 compact
```

The executable accepts arbitrary mono PCM16 16 kHz WAV content up to 30 seconds. The fourth argument is the maximum number of decode steps. The optional fifth argument is `fixed30` or `compact`; omitting it preserves `fixed30`. Current decoding is English transcription, greedy, batch one and no timestamps.

Machine-readable cumulative measurements are in [`results.json`](results.json). Fixed30 fields are in [`result.json`](benchmarks/jfk-11s/result.json); E8 compact fields are in [`compact-result.json`](benchmarks/jfk-11s/compact-result.json); E9 row-reuse statistics, resource audit and raw hashes are in [`row-reuse-result.json`](benchmarks/jfk-11s/row-reuse-result.json).
