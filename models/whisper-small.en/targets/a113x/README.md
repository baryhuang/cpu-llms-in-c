# Target: Amlogic A113X — Whisper small.en

Status: the complete 12-layer `small.en` encoder runs from a compiler-generated image. Q4 Transformer matrices, Cortex-A53 NEON Q4 GEMM, four-thread scheduling and NEON full attention are measured. The decoder and tokenizer are not implemented; these are encoder measurements, not transcription results.

## Target pin

| Item | Value |
|---|---:|
| SoC | Amlogic A113X |
| CPU | 4x Arm Cortex-A53 |
| Maximum observed clock | 1.416 GHz |
| RAM | 2,059,239,424 bytes |
| SIMD | NEON/ASIMD, FMA; no dotprod, i8mm or SVE |
| Chip year | 2017 |
| Q4 encoder image | 56,763,776 bytes; SHA-256 `4ac8031884f1cc4d8eae215507713bf750633c793902f6f1ec095b82d91201d5` |

## Result

The measured complete encoder is not viable for real-time transcription. The final CPU stage takes 556.218 seconds for a 30-second log-Mel window: encoder-only RTF 18.541. Peak RSS and swap pass their gates. Compute does not.

| Metric | Gate | Measured final CPU stage | Status |
|---|---:|---:|---|
| Encoder-only RTF, 30-second window | `<= 1.0` is necessary but not sufficient for full ASR | 18.541 | fail |
| Full ASR RTF | `<= 1.0` | not available; decoder absent | open |
| Relative WER increase | `<= 10%` | not available; decoder absent | open |
| Peak RSS | `< 1 GiB` | 125,056 KiB | pass |
| Process swap | `0` | 0 | pass |
| Device swap change | `0` | `SwapFree` 1,004,972 KiB before and after | pass |

RTF uses original audio-equivalent duration: 3,000 log-Mel frames × 10 ms = 30 seconds. The timed path starts at log-Mel and includes the convolution stem, 12 encoder blocks and final LayerNorm. It excludes waveform-to-Mel, decoder, tokenizer and timestamp generation.

## Incremental benchmark

All rows use the same Q4 image, deterministic synthetic log-Mel values, 12 encoder layers and one timed invocation. Verification is excluded from duration.

### One-second window: CPU increments

| Stage | Cumulative implementation | Duration | Encoder-only RTF | Increment | Cumulative | CPU | Peak RSS | Process swap | Checksum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | generic scalar Q4 | 32.716950 s | 32.717 | 1.00x | 1.00x | 98% | 54,784 KiB | 0 | -5352.40555 |
| A1 | G0 + A53 NEON Q4 GEMM with frame reuse | 12.479286 s | 12.479 | 2.62x | 2.62x | 98% | 54,912 KiB | 0 | -5352.40810 |
| A2 | A1 + static four-thread Conv, GEMM, normalization and attention schedule | 3.442861 s | 3.443 | 3.62x | 9.50x | 378% | 54,656 KiB | 0 | -5352.40810 |
| A3 | A2 + A53 NEON full-attention dot/context kernels | 3.156888 s | 3.157 | 1.09x | 10.36x | 379% | 54,656 KiB | 0 | -5352.40621 |

### Thirty-second window: feasibility measurement

G0 and A1 were not run at 30 seconds. Their cells remain absent rather than being extrapolated from the one-second window; full attention is quadratic in frame count.

| Stage | Duration | Encoder-only RTF | Increment | CPU | Peak RSS | Workspace | Process swap | Device swap change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 | 1,465.221181 s | 48.841 | 1.00x | 386% | 125,184 KiB | 64,536,000 bytes | 0 | 0 KiB |
| A3 | 556.217968 s | 18.541 | **2.63x** | 382% | 125,056 KiB | 64,536,000 bytes | 0 | 0 KiB |

At A3 every block takes 44.67–45.87 seconds. The stem takes 12.81 seconds. Further Q4 dot tuning cannot supply the remaining 18.5x required before decoder work. Frame reduction or a different encoder attention structure is required on the model axis.

## Verification

Verification is not included in either benchmark table.

| Boundary | Reference | Result |
|---|---|---|
| F32 image: real stem | independent NumPy graph, pinned real weights | max abs `4.77e-7`, pass |
| F32 image: real layer 0 | same | max abs `5.36e-7`, pass |
| F32 image: 12 layers + final LN | same | max abs `3.81e-6`, pass |
| Q4 image: real layer 0 | F32 reference activation | RMSE `0.0605037`, finite measurement |
| Q4 image: 12 layers + final LN | F32 reference activation | RMSE `0.2382805`, finite measurement; not a WER gate |
| A3 vs generic Q4 fixture | same real-weight fixture | final RMSE agrees within floating-point reduction order |

The Q4 image quantizes 72 two-dimensional Transformer matrices in signed group-128 Q4 with BF16 scales. Conv, position, bias and LayerNorm tensors remain F32. Q4 activation error is recorded, not declared acceptable; only transcript WER can accept the artifact.

## Build and run

Weights and images are generated outside Git:

```sh
python3 compiler/compile_whisper_small_encoder_image.py \
  --checkpoint model.safetensors --config config.json \
  --mel-filters mel_filters.npz --precision q4 \
  --output small.en-q4.whenc

make build/whisper-small-encoder-bench-a113x \
  CC=gcc CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic' \
  OMPFLAGS=-fopenmp

OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  build/whisper-small-encoder-bench-a113x \
  small.en-q4.whenc 3000 12 1
```

Machine-readable fields and all per-layer durations are in [`results.json`](results.json). Structural reduction belongs to the model axis and remains in [`../../DECISIONS.md`](../../DECISIONS.md).
