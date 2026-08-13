# Qwen3.6-27B on Apple M3 Pro

Status: the real layer-0 MLP and exact-shape one-token DeltaNet recurrent core are implemented and measured. Q4 DeltaNet projections, convolution, full layers, tokenizer and token generation are not implemented. No result here is model tokens/s.

## Target pin

| Property | Value |
|---|---|
| Machine | MacBook Pro `Mac15,6` |
| SoC | Apple M3 Pro, 2023 |
| CPU | 5 performance + 6 efficiency cores |
| GPU | 14 cores, Metal 3 |
| Unified memory | 36 GB / 38,654,705,664 bytes |
| OS used for measurement | macOS 15.7.3, build 24G419 |
| Metal compiler | Apple Metal 32023.864 |

The deployment path has a C API and C data structures. `qwen36_m3.m` is the thin Objective-C system-ABI bridge required to create Metal resources; all compute is in the committed Metal shader. It links no Python, MLX, llama.cpp or C++ runtime.

## Additional specialization over an unmodified MLX invocation

This is an implementation scope, not a performance claim against oMLX.

| Compile-time fact | Generated action | Status |
|---|---|---|
| model revision and affine Q4 group size are pinned | validate exact tensor names, shapes, dtypes, offsets and complete shard bounds in C | implemented |
| all layer shapes are fixed | compile dimensions and dispatch geometry into the Metal functions | implemented for layer-0 MLP |
| quantized nibbles and scale/bias have different access patterns | emit separate continuous streams; convert BF16 metadata to FP16 after an oracle check | implemented |
| gate and up always consume the same activation | compute both projections and SiLU/multiply in one Metal dispatch | implemented |
| CPU and GPU share physical memory | map the compiled image into no-copy `MTLBuffer` views | implemented |
| useful threadgroup count is machine-specific | test 64, 128, 256 and 512 threads at startup and select the measured winner | implemented |
| MLP down is followed by residual addition | fuse residual into the down-projection kernel | implemented |
| layer schedule is exactly `3 GatedDeltaNet + 1 attention`, repeated 16 times | emit a static graph with no dynamic operator dispatch | planned |
| DeltaNet state precision and layout are known | retain state in FP32; one recurrent-core kernel updates all 48 heads | core implemented; projection fusion planned |
| only 16 layers use full attention and only 4 KV heads exist | emit a separate attention kernel and compact KV-cache image | planned |
| prompt prefill has larger matrix shapes than decode | benchmark Metal against Apple CPU Accelerate/BNNS per static shape; choose at compile time | planned |
| a deployment may declare a fixed prompt prefix | compile both KV and DeltaNet-state snapshots | planned, optional |
| MTP dimensions are fixed | generate dedicated proposal and verification kernels | planned, optional |

M3 does not provide the Metal 4 tensor/NAX path. This target therefore uses Metal 3 custom kernels. The CPU remains useful for tokenizer/control work and may win selected prefill shapes, but no CPU/GPU split is accepted without same-machine measurement.

## Measured incremental result

The input is the real layer-0 `gate_proj`, `up_proj` and `down_proj` from the pinned affine-Q4 checkpoint. One invocation uses a deterministic FP16 activation vector of length 5,120:

```text
x[i] = fp16(0.75 * sin(i * 0.013) + 0.20 * cos(i * 0.031))
```

The complete candidate path is:

```text
x[5120]
  -> fused Q4 gate + Q4 up + SiLU/multiply
  -> intermediate[17408]
  -> Q4 down + residual
  -> output[5120]
```

Each number below is the median of the named metric across five independent processes. Each process uses five warmups and 40 measured invocations per path.

| Step | Measured path | Duration | Increment | Cumulative |
|---:|---|---:|---:|---:|
| 0 | internal generic 36-byte interleaved blocks; split gate/up/SiLU | 1.072704 ms | — | 1.000000x |
| 1 | Apple split nibble/metadata streams; split gate/up/SiLU | 1.008423 ms | 1.063744x | 1.063744x |
| 2 | same Apple layout; fused gate/up/SiLU | **0.969930 ms** | 1.039686x | **1.105960x** |
| 3 | fused gate/up/SiLU plus down/residual | **1.367039 ms** | not comparable: adds down projection | **110.022589 GB/s** effective weight traffic |

The full MLP reads 150,405,120 packed weight/metadata bytes. The generated image is 150,409,216 bytes including its page-sized header. It is memory-mapped rather than copied into Metal-owned weight buffers.

Peak process physical footprint was 178,390,656 bytes. This is a benchmark footprint, not a projected release-runtime footprint: the process also keeps 100,648,960 bytes of benchmark-only generic-reference buffers for the incremental comparison.

## DeltaNet recurrent core

Qwen3.6-27B uses 16 key heads and 48 value heads, both with dimension 128. Each key head is repeated for three value heads. One linear-attention layer retains a `48 x 128 x 128` FP32 state: 3,145,728 bytes. All 48 layers retain 150,994,944 bytes (144 MiB), before convolution windows and allocator overhead.

The measured primitive starts after q/k L2 normalization and `g`/`beta` calculation. It applies decay, delta update and output contraction for one token. It excludes Q4 projections, depthwise convolution, gated RMSNorm and output projection.

| Path | Duration | Effective state traffic | Decision |
|---|---:|---:|---|
| scalar thread per value channel | **0.152173 ms** | **62.016185 GB/s** | selected |
| `float2` per thread | 0.155072 ms | 60.860 GB/s | rejected; no gain |

The protocol alternates path order across five samples in each process and resets state before each sample. Each sample contains 100 updates; the table is the median of five process medians. Output error against C is exactly zero for this input; maximum state error is 3.7252903e-9.

## Verification

The C packer computes reference outputs directly from the original checkpoint's U32 quantized weights and BF16 scale/bias tensors before converting metadata to FP16. Those values are stored in the generated image header and compared with the Metal output at runtime.

| Comparison | Maximum absolute error |
|---|---:|
| fused vs split Apple GPU | 0 |
| fused vs generic-layout GPU | 2.98023224e-7 |
| fused vs converted-metadata C scalar, first 8 rows | 6.2584877e-7 |
| fused gate/up vs original-BF16 source oracle, first 8 rows | 6.2584877e-7 |
| complete MLP vs original-BF16 source oracle, first 8 rows | 5.06639481e-7 |

| Row | Source-BF16 complete MLP | Metal complete MLP |
|---:|---:|---:|
| 0 | 0.145588845 | 0.145589232 |
| 1 | 0.340642303 | 0.340641797 |
| 2 | 0.236884966 | 0.236884803 |
| 3 | 0.177922145 | 0.177922025 |
| 4 | 0.211153850 | 0.211153775 |
| 5 | 0.309731871 | 0.309731722 |
| 6 | 0.193791032 | 0.193790987 |
| 7 | 0.203297257 | 0.203297168 |

All five raw process results, artifact hashes and the rejected experiments are in [`results.json`](results.json).

## Rejected experiments

| Candidate | Result | Decision |
|---|---:|---|
| pair gate/up bytes into a second physical representation | 1.0103x median complete-MLP change | rejected; one-percent result did not justify another gate/up copy |
| rewrite `dot(scale*q+bias,x)` as `scale*dot(q,x)+bias*sum(x)` | 1.369155 ms vs 1.366022 ms stable baseline | rejected; no gain and anomalous slow runs |
| cache half a DeltaNet head in 32 KiB threadgroup memory | 0.488036 ms vs 0.328170 ms direct | rejected; explicit copy and synchronization cost more than cached reread |
| process four DeltaNet values per thread | 0.400043 ms vs 0.320628 ms direct | rejected; reduced occupancy dominates scalar reuse |
| process two DeltaNet values per thread | 0.155072 ms vs 0.152173 ms direct | rejected after balanced path-order measurement |

Rejected code is not retained in the execution path.

## oMLX acceptance gate

An external oMLX entry for a 14-GPU-core M3 Pro / 36 GB machine reports Qwen3.6-27B 4-bit at 76.3 prompt tokens/s and 8.8 generated tokens/s for 1K context; 4K reports 77.3 and 8.5. It used oMLX v0.3.5.dev1 and macOS 26.4.1. This is a planning reference, not a same-machine baseline: [oMLX benchmark `3am3b1rj`](https://omlx.ai/benchmarks/3am3b1rj).

| Required before saying "faster than oMLX" | Rule |
|---|---|
| machine | this exact `Mac15,6` |
| model input | revision and hashes in [`../../pins.json`](../../pins.json) |
| quantization and context | same Q4 group-64 input; 1K and 4K profiles |
| output contract | same chat template, sampler and generated-token count |
| execution scope | tokenizer through decoded text; not a primitive |
| quality | logit/output and held-out quality gates pass |
| metrics | prefill, TTFT, decode, peak physical footprint and energy |

The repository has not passed this gate. The current 1.105960x number compares two repository-internal MLP paths; it must not be presented as a speedup over oMLX.

## Build and reproduce

No Python command is used for this target.

```sh
make qwen36-tools
make qwen36-m3-bench
make qwen36-m3-deltanet-bench

build/qwen36-safetensors-inspect SOURCE.safetensors \
  language_model.model.layers.0.mlp.gate_proj.weight \
  language_model.model.layers.0.mlp.gate_proj.scales \
  language_model.model.layers.0.mlp.gate_proj.biases \
  language_model.model.layers.0.mlp.down_proj.weight

build/qwen36-m3-pack SOURCE.safetensors layer0-mlp.q36m3 SOURCE_SHA256

build/qwen36-m3-mlp-bench --image layer0-mlp.q36m3 \
  build/qwen36-m3-q4.metallib 40 5

build/qwen36-m3-deltanet-bench build/qwen36-m3-q4.metallib 100 10
```

`qwen36-m3-pack` refuses to overwrite an existing output file. Checkpoint shards and generated images remain outside Git.

## Next graph work

| Order | Work | Gate |
|---:|---|---|
| 1 | implement RMSNorm and one exact GatedDeltaNet layer with FP32 recurrent state | state and layer-boundary oracle |
| 2 | implement 4-KV-head attention and compact KV cache | attention-boundary and long-context replay |
| 3 | emit the static 64-layer text graph and C tokenizer | greedy logit/token comparison |
| 4 | tune batched prefill across Metal and measured Apple CPU paths | same-machine prompt tokens/s |
| 5 | add fixed-prefix snapshots and dedicated MTP kernels | independent correctness and acceptance-rate gates |
| 6 | run the exact same-machine oMLX comparison | only then evaluate the outperform-oMLX objective |
