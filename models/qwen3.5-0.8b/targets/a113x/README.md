# Target: Amlogic A113X (ThirdReality LinuxBox)

Status: planned. The CPU pin below is provisional until `target_probe` runs on the device; no measurements exist yet.

## Provisional CPU pin

| Field | Value |
|---|---|
| SoC | Amlogic A113X |
| Cores | 4x ARM Cortex-A53 (ARMv8-A, AArch64) |
| SIMD | NEON/ASIMD 128-bit, includes TBL byte table lookup |
| RAM | 1 GB or 2 GB depending on board variant — to be pinned |
| Storage | eMMC 8/32 GB |
| OS | Armbian (Debian bookworm) |

To finalize: cache sizes, measured memory bandwidth, page size, and thermal behavior from an on-device probe. Gates (RSS ceiling, zero swap, latency targets) are set after the first baseline run.

## CPU-axis optimizations (this target only)

In order of execution. Estimates are analytical; none is a benchmark result.

1. **Scalar baseline first.** Cross-compile the unmodified C11 runtime, run the probe and the artifact on the device, and record the first `results.json` here. Every later step is measured against this. The same baseline run also measures off-the-shelf llama.cpp on the device for the rewrite-versus-stack comparison in the top-level [`README.md`](../../../../README.md).
2. **NEON vectorized Q4 GEMV.** Import the proven ggml/llama.cpp kernel technique for the ARMv8.0 baseline: vector nibble unpacking (`vand`/`vshr`) and 16-lane multiply-accumulate. Existing off-the-shelf kernels are a source of CPU-axis techniques, not a fixed advantage of another stack; this step absorbs the one that applies to this core. Expected ~3-5x over the scalar baseline.
3. **NEON TBL lookup kernel.** The T-MAC table-lookup formulation of low-bit GEMV maps onto NEON's TBL instruction (also ARMv8.0): a 16-entry per-input table fits one 128-bit register, eliminating unpacking and multiplication entirely. Measured against step 2 on the device; the faster kernel is kept.
4. **Multi-token lookup prefill.** Vec-LUT-style vectorization: reuse each precomputed table across the batched prompt tokens so prefill saturates bandwidth instead of repeating per-token lookups.
5. **Four-thread static partition.** Fixed row partitions per core with deterministic reductions (versus the current 2-thread OpenMP loop); expected up to ~2x, capped by DRAM bandwidth.
6. **Weight layout for the A53.** Offline reordering of Q4 records for sequential DRAM access in kernel visit order, tile sizes fitted to the small L1/L2, and software prefetch distances tuned on-device.
7. **Experimental: lower-bit LUT.** The lookup kernel's cost scales linearly down with bit width; a Q3/Q2 variant would cut traffic a further 25-50% if task quality survives — measured, not assumed.

## Feasibility notes

- Estimated usable DRAM bandwidth on A53-class parts is 1.5-3 GB/s; at ~290 MB per decode token the analytical ceiling is roughly 5-10 tokens/s, before overheads.
- The ~420 MB Q4 image plus ~18 MB DeltaNet state and small KV fits the 1 GB board with headroom; the 2 GB board would also fit Qwen3.5-2B (~1.05 GB image) at roughly 2.5-3x lower speed.
