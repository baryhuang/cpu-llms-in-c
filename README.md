# cpu-llms-in-c

An offline compiler turns a pinned language model into a packed Q4 image, and a small C11 runtime executes it on CPU. The deployed target needs no Python, PyTorch, llama.cpp, or ONNX Runtime. Task outputs are defined by the prompt at run time — the runtime is not hardwired to one task.

## Organization

Everything is classified along two axes, model first, CPU second. A released artifact is one model x CPU pair, and results never transfer between pairs. The full contract is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Path | Contents |
|---|---|
| [`tools/`](tools/) | On-target probe: ISA, topology, and measured memory bandwidth for CPU pins |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| `models/<model>/` | Model axis: pins, profile, graph record, reference outputs, model-only optimizations |
| `models/<model>/targets/generic/` | The model's C runtime with model-axis optimizations only, portable to any CPU |
| `models/<model>/targets/<cpu>/` | CPU axis: CPU pin, CPU-specialized kernels, and results measured for that pair |
| [`tests/`](tests/) | Committed correctness tests (`make test`) |

Checkpoints, generated images, binaries, and credentials are never committed.

## Status

| Model | CPU | Status | Verification | Measured performance | Record |
|---|---|---|---|---|---|
| Gemma 4 E2B | two-vCPU x86-64 dev machine (unpinned) | implemented | 12/12 written labels, 10/10 layer-0 boundaries | 0.598 tokens/s scalar, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [inputs/outputs](REVIEW.html) · [raw data](models/gemma-4-e2b/results.json) |
| Qwen3.5-0.8B | Amlogic A113X (4x Cortex-A53, 1-2 GB) | planned | nothing pinned yet | nothing measured yet | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) |

The Gemma artifact predates the prompt-defined output contract and compiles its two labels in — now the restricted special case. The Qwen artifact carries the default contract: runtime tokenizer, full output head, per-call answer sets.

## Memory is the governing factor

Decode must stream every visited weight byte from DRAM for each token, so the hard ceiling is:

```text
tokens/s  ≤  usable memory bandwidth / weight bytes visited per token
```

| Quantity | Gemma 4 E2B on dev x86 (measured) | Qwen3.5-0.8B on A113X (estimate) |
|---|---:|---:|
| Weight bytes visited per decode token | 960 MB | ~290 MB |
| Usable DRAM bandwidth | not the limit yet | ~1.5-3 GB/s (to be probed) |
| Achieved streaming rate | 575 MB/s (scalar-kernel bound) | — |
| Resulting decode rate | 0.598 tokens/s measured | ~5-10 tokens/s ceiling |
| Image + state vs RAM | 966 MB + ~50 MB, fits, zero swap | ~420 MB + ~50 MB vs 1-2 GB, fits |

Two regimes follow, and they map exactly onto the two optimization axes:

| Regime | Binding constraint | Lever | Axis |
|---|---|---|---|
| Compute-bound (current scalar kernel) | 575 MB/s achieved vs multi-GB/s available | faster kernels, more threads | CPU |
| Bandwidth-bound (after SIMD kernels) | DRAM bandwidth itself | fewer bytes per token: smaller model, answer-set scoring, lower bits | model |

Capacity is a gate, not a tunable: the image and state must stay resident with zero swap — exceeding RAM means paging from eMMC at ~100-300 MB/s and an order-of-magnitude collapse.

## Optimization roadmap

Each step is an analytical estimate, not a benchmark result; factors do not multiply cleanly and the stack is capped by the target's memory bandwidth. A step lands only after tensor-level correctness tests for the code it touches.

| Step | Axis | Mechanism | Estimated effect |
|---|---|---|---|
| 1. Switch to Qwen3.5-0.8B | model | per-token matrix traffic 960 MB → ~290 MB | ~3.3x per token |
| 2. Batched prefill | model | read each matrix once per layer per prompt, not per token | up to ~40x on prompt prefill |
| 3. Per-call answer-set scoring | model | skip the 248K-row output head unless generating | ~130 MB saved per decision |
| 4. NEON kernel | CPU | imported llama.cpp-style vectorized GEMV, then T-MAC-style TBL lookup; keep the faster | ~3-5x kernel throughput |
| 5. Four-thread static partition | CPU | all A113X cores, deterministic reductions | up to ~2x, bandwidth-capped |
| 6. Lower-bit LUT (experimental) | CPU | Q3/Q2 with linear LUT cost scaling, only if task quality survives | further 1.3-2x |

Model-axis details: [`models/qwen3.5-0.8b/README.md`](models/qwen3.5-0.8b/README.md). CPU-axis details: [`models/qwen3.5-0.8b/targets/a113x/README.md`](models/qwen3.5-0.8b/targets/a113x/README.md).

## Why rewrite instead of using an existing stack

Theoretical comparison for Qwen3.5-0.8B on the A113X device; no measurements yet.

| Stack | Fits the 1 GB board | Decode traffic per token | Notes |
|---|---|---:|---|
| This C runtime (plan) | yes | ~290 MB | ~30 MB overhead, zero dependencies, per-call answer sets skip the head |
| llama.cpp (Q4 GGUF) | yes | ~420 MB | full 248K-row head every token; DeltaNet CPU op is new and unspecialized |
| PyTorch + Transformers | no | ~1.6 GB | BF16 weights alone exceed RAM; serves as the numerical oracle |
| ONNX Runtime | no | — | no operators for non-softmax attention as of 2026 |

llama.cpp is the only real alternative. Kernel techniques are not what separates the stacks — its NEON vectorization is imported here as CPU-axis step 4, and both then face the same DRAM bandwidth wall. What a generic stack cannot absorb:

| Structural advantage | Theoretical effect |
|---|---|
| Per-call answer-set scoring | skips ~130 MB head traffic per decision, ~1.4x on decision latency |
| Non-weight memory | ~30 MB vs ~100-300 MB, headroom on the 1 GB board |
| Dependency-free static binary | ~100 KB, no Python or C++ runtime |
| Tensor-verified DeltaNet path | per-boundary comparison vs a freshly landed upstream op |
| Per-target kernel and layout specialization | roadmap steps 4-6, tuned per CPU pin |

The on-device baseline measurement includes llama.cpp, and its numbers are recorded alongside ours.

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
