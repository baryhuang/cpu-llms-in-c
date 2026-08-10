# cpu-llms-in-c

An offline compiler turns a pinned language model into a packed image, and a small C11 runtime executes it on a pinned CPU/SoC. A target may dispatch compiler-selected graph regions to an on-SoC accelerator. The deployed target needs no Python, PyTorch, llama.cpp, or ONNX Runtime. Task outputs are defined by the prompt at run time — the runtime is not hardwired to one task.

## Organization

Everything is classified along two axes, model first, CPU/SoC target second. The target pins the CPU and, when selected, an on-SoC accelerator. A released artifact is one model x target pair, and results never transfer between pairs. The full contract is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Path | Contents |
|---|---|
| [`tools/`](tools/) | On-target probe: ISA, topology, and measured memory bandwidth for CPU pins |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| `models/<model>/` | Model axis: pins, profile, graph record, reference outputs, model-only optimizations |
| `models/<model>/targets/generic/` | The model's C runtime with model-axis optimizations only, portable to any CPU |
| `models/<model>/targets/<soc>/` | Target axis: CPU/SoC pin, accelerator boundary, specialized kernels, and results measured for that pair |
| [`tests/`](tests/) | Committed correctness tests (`make test`) |

Checkpoints, generated images, binaries, and credentials are never committed.

## Status

| Model | CPU / SoC target | Chip year | Status | Verification | Measured performance | Record |
|---|---|---:|---|---|---|---|
| Gemma 4 E2B | two-vCPU x86-64 dev machine (unpinned) | not pinned | implemented | 12/12 written labels, 10/10 layer-0 boundaries | 0.598 tokens/s scalar, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [inputs/outputs](REVIEW.html) · [raw data](models/gemma-4-e2b/results.json) |
| Qwen3.5-0.8B | Amlogic A113X: 4x Cortex-A53 | 2017 | planned | no target run | nothing measured | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) |
| Qwen3.5-0.8B | Rockchip RK3588S: 4x Cortex-A76 + 4x Cortex-A55, NPU | 2022 | target plan recorded | external baseline only; no target run | nothing measured by this repository | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/rk3588s/README.md) |
| Qwen3.5-0.8B | Rockchip RK3576: 4x Cortex-A72 + 4x Cortex-A53, NPU | 2024 | target plan recorded | external baseline only; no target run | nothing measured by this repository | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/rk3576/README.md) |

Chip year means first public MP release, official launch, or official development-board sale; it is not the board manufacture year. The evidence and exact event are recorded in each target file.

The Gemma artifact predates the prompt-defined output contract and compiles its two labels in — now the restricted special case. The Qwen artifact carries the default contract: runtime tokenizer, full output head, per-call answer sets.

## Memory is the governing factor

Decode must stream every visited weight byte from DRAM for each token, so the hard ceiling is:

```text
tokens/s  ≤  usable memory bandwidth / weight bytes visited per token
```

| Quantity | Gemma 4 E2B on dev x86 (measured) | Qwen3.5-0.8B on A113X (estimate) |
|---|---:|---:|
| Weight bytes visited per decode token | 960 MB | ~350 MB (Q8 DeltaNet projections included) |
| Usable DRAM bandwidth | not the limit yet | ~1.5-3 GB/s (to be probed) |
| Achieved streaming rate | 575 MB/s (scalar-kernel bound) | — |
| Resulting decode rate | 0.598 tokens/s measured | ~4-8 tokens/s ceiling |
| Image + state vs RAM | 966 MB + ~50 MB, fits, zero swap | ~465 MB + ~50 MB vs 1-2 GB, fits |

Two regimes follow, and they map exactly onto the two optimization axes:

| Regime | Binding constraint | Lever | Axis |
|---|---|---|---|
| Compute-bound (current scalar kernel) | 575 MB/s achieved vs multi-GB/s available | faster kernels, more threads | CPU |
| Bandwidth-bound (after SIMD kernels) | DRAM bandwidth itself | fewer bytes per token: smaller model, answer-set scoring, lower bits | model |

Capacity is a gate, not a tunable: the image and state must stay resident with zero swap — exceeding RAM means paging from eMMC at ~100-300 MB/s and an order-of-magnitude collapse.

The Rockchip NPU changes the memory/compute trade. The following numbers are external RKLLM v1.3 reference data with sequence length 128 and 64 generated tokens. They are not measurements by this repository.

| Model | Target | Quantization | TTFT | Decode | Reported memory | 1 GB implication |
|---|---|---|---:|---:|---:|---|
| Qwen3.5-0.8B | RK3588 | W8A8 | 587.74 ms | 27.05 tokens/s | 1039.66 MB | no OS/runtime headroom |
| Qwen3.5-0.8B | RK3576 | W4A16 | 1369.31 ms | 18.79 tokens/s | 689.50 MB | fits on paper; board RSS and zero swap still unverified |

Source: [Rockchip RKLLM benchmark, revision `878f936`](https://github.com/airockchip/rknn-llm/blob/878f9361fd3afa7e167b7079918918f78d2c1c2a/benchmark.md). RK3588S shares the RK3588 compute/NPU block used by this planning baseline, but the exact board must still be measured.

## Optimization roadmap

Each step is an analytical estimate, not a benchmark result; factors do not multiply cleanly and the stack is capped by the target's memory bandwidth. A step lands only after tensor-level correctness tests for the code it touches.

| Step | Axis | Mechanism | Estimated effect |
|---|---|---|---|
| 1. Switch to Qwen3.5-0.8B | model | per-token matrix traffic 960 MB → ~350 MB (Q8 DeltaNet projections — see the model record) | ~2.7x per token |
| 2. Batched prefill | model | read each matrix once per layer per prompt, not per token | up to ~40x on prompt prefill |
| 3. Per-call answer-set scoring | model | skip the 248K-row output head unless generating | ~130 MB saved per decision |
| 4. NEON kernel | CPU | imported llama.cpp-style vectorized GEMV, then T-MAC-style TBL lookup; keep the faster | ~3-5x kernel throughput |
| 5. Four-thread static partition | CPU | all A113X cores, deterministic reductions | up to ~2x, bandwidth-capped |
| 6. Lower-bit LUT (experimental) | CPU | Q3/Q2 with linear LUT cost scaling, only if task quality survives | further 1.3-2x |

Model-axis details: [`models/qwen3.5-0.8b/README.md`](models/qwen3.5-0.8b/README.md). Target details: [A113X](models/qwen3.5-0.8b/targets/a113x/README.md) · [RK3588S](models/qwen3.5-0.8b/targets/rk3588s/README.md) · [RK3576](models/qwen3.5-0.8b/targets/rk3576/README.md).

## Why rewrite instead of using an existing stack

A general inference stack must accept any model and any prompt at load time. This repository instead compiles one pinned model, one task contract, and one CPU/SoC target ahead of time — every advantage below is a consequence of that, not of cleverer kernels. All numbers are analytical estimates for Qwen3.5-0.8B on the 1 GB A113X board, not measurements; the first on-device baseline run measures llama.cpp side by side ([target plan](models/qwen3.5-0.8b/targets/a113x/README.md)).

| Stack | Fits the 1 GB board | Decode traffic per token (est.) | Disqualifier or cost |
|---|---|---:|---|
| This C runtime (plan) | yes | ~350 MB | ~30 MB non-weight memory, zero dependencies, per-call answer sets skip the head |
| llama.cpp (Q4 GGUF) | yes | ~420 MB | pays the full 248,320-row head every token; DeltaNet CPU op is freshly landed, unspecialized, and not tensor-verified |
| PyTorch + Transformers | no | ~1.6 GB | BF16 weights alone exceed RAM — kept only as the numerical oracle |
| ONNX Runtime | no | — | cannot run the architecture: no operators for non-softmax attention as of 2026 |

That leaves llama.cpp as the only stack that runs at all, and kernels are not what separate us from it: its NEON vectorization is imported here as CPU-axis roadmap step 4, after which both stacks face the same DRAM bandwidth wall. The remaining gap is structural — each row below depends on information a generic runtime does not have at load time:

| Structural advantage | Estimated effect on the A113X | Why a generic stack cannot absorb it |
|---|---|---|
| Per-call answer-set scoring | skips ~130 MB of head traffic per decision, ~1.4x on decision latency | needs the task contract in the runtime API; a generic decode loop computes all 248,320 logits every token |
| Compile-time exact rewrites | text-only input contract: MRoPE provably reduces to RoPE, vision tower and MTP head never enter the image | must keep run-time paths for inputs that never arrive |
| Fixed non-weight memory | ~30 MB bounded arena vs ~100-300 MB framework overhead — the headroom that decides fit on 1 GB | allocator, context, and graph machinery sized for generality |
| Dependency-free static binary | ~100 KB, no Python or C++ runtime on the target | frameworks ship their runtime with the model |
| Tensor-verified DeltaNet path | per-boundary comparison against the pinned oracle before any kernel lands | upstream op is freshly landed with no equivalent gate |
| Per-target specialization | roadmap steps 4-6: kernels, layout, and threads tuned to the pinned CPU, results recorded per pair | one build must serve every CPU |

These estimates become results only through the recorded baseline run; llama.cpp's on-device numbers land alongside ours.

## Where the differentiation is

The section above compares stacks on the same board. This one compares boards: the value of this repository falls monotonically with hardware price, because expensive edge hardware already has a solved LLM story. Prices are street prices observed 2026-08, not pinned quotes.

| Tier | Representative hardware | Street price | Existing LLM story | Value of this repository |
|---|---|---:|---|---|
| High-end edge | Jetson Orin Nano Super, 67 TOPS | $249 kit | mature — CUDA/TensorRT/Ollama, sub-1B models at 25-40 tokens/s | ~none |
| Mid SoC | RK3588S / RK3576, 6 TOPS NPU | $75-220 board | RKLLM works (27.05 / 18.79 tokens/s external baseline) | weak — kept as external reference baselines only |
| Low with NPU | RK3562, 1 TOPS NPU | ~$20-40 class | RKLLM supported, but Qwen3.5-0.8B w8a8 needs 1,021 MB — a 1 GB board cannot host it | real — Q4 CPU path is the only one that fits 1 GB |
| Low, no NPU path | A113X, RK3566-class | $10-20 SoC | none: no vendor stack targets these chips for LLMs | all of it |

The battleground is the bottom two rows: low-cost boxes already deployed in the field (smart-home hubs, gateways), 1 GB RAM, A53/A55 cores, hardware no vendor LLM stack serves or plans to serve. A $249 Jetson cannot reach a $20 BOM, and RKLLM requires chips and memory these boxes do not have — a dependency-free Q4 C runtime with ~30 MB overhead is the only path onto them.

Consequences for target priority: A113X stays first. RK3588S and RK3576 remain external reference baselines with no kernel investment. RK3562 is the interesting middle case — the same A53 kernels as the A113X apply unchanged, and one board can host the CPU-versus-NPU comparison directly — queued after the A113X baseline lands.

## Build and test

```sh
make test
```

Model-specific compile and run commands live in each model record, e.g. the [Gemma 4 E2B build](models/gemma-4-e2b/README.md#build).

## Current limits

| Limit | Where it is addressed |
|---|---|
| No runtime tokenizer or free-text input | Qwen artifact plan, model axis |
| No SIMD kernel | CPU-axis roadmap steps 4-6 |
| No full-graph tensor-by-tensor differential test | required before any kernel swap lands |
| No held-out application-quality evaluation | open |
| Cross-model compiler remains a design | the second model record starts generalizing it |

## License

No license has been selected.
