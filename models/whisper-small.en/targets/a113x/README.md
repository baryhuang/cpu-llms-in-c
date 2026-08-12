# Target: Amlogic A113X — Whisper small.en

Status: target contract selected; no `small.en` runtime has been measured on this board.

| Item | Pin |
|---|---|
| SoC | Amlogic A113X |
| CPU | 4x Arm Cortex-A53 |
| Maximum observed clock | 1.416 GHz |
| RAM | 2,059,239,424 bytes observed |
| Available SIMD | NEON/ASIMD, FMA |
| Absent | dotprod, i8mm, SVE |
| Chip year | 2017 |

## Target gates

| Metric | Gate |
|---|---:|
| Continuous-speech RTF | `<= 1.0` |
| Relative WER increase vs pinned unmodified `small.en` | `<= 10%` |
| Peak RSS | `< 1 GiB` |
| Swap | `0` |

The existing `tiny.en` Q5_1 measurement belongs to the separate large-v3 feasibility record and is not a small.en benchmark. The first target action after a full generic graph exists is an unmodified `small.en` baseline with frontend, encoder, decoder and wall durations recorded separately.

## CPU work after exact boundaries

| Stage | Mechanism |
|---|---|
| A1 | real FFT and fixed 80-bin filterbank schedule |
| A2 | packed NEON Conv1D stem |
| A3 | packed quantized projection and FFN kernels |
| A4 | fused normalization/residual/GELU schedules |
| A5 | static four-core encoder and head partitions |
| A6 | storage prefetch and bounded static arena |

Structural decoder, encoder, low-rank and frame-compaction work belongs to the model axis and is recorded in [`../../DECISIONS.md`](../../DECISIONS.md).
