# cpu-llms-in-c

An offline compiler turns a pinned language model into a packed Q4 image, and a small C11 runtime executes it on CPU. The deployed target needs no Python, PyTorch, llama.cpp, or ONNX Runtime. Task outputs are defined by the prompt at run time — the runtime is not hardwired to one task.

## Organization

Everything is classified along two axes, model first, CPU second. A released artifact is one model x CPU pair, and results never transfer between pairs. The full contract is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Path | Contents |
|---|---|
| [`runtime/`](runtime/) | C runtime, headers, layer reference, hardware probe |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| `models/<model>/` | Model axis: pins, profile, graph record, reference outputs, model-only optimizations |
| `models/<model>/targets/<cpu>/` | CPU axis: CPU pin, kernels, and results measured for that pair |
| [`tests/`](tests/) | Committed correctness tests (`make test`) |

Checkpoints, generated images, binaries, and credentials are never committed.

## Status

**Implemented: [Gemma 4 E2B](models/gemma-4-e2b/README.md)** — a complete 35-layer C/Q4 artifact for one compiled two-label profile, verified 12/12 against written labels and 10/10 against layer-0 tensor boundaries. Measured on an unpinned two-vCPU x86-64 dev machine: 0.598 tokens/s, scalar kernel, 926 MiB peak RSS, zero swap. Exact inputs, outputs, and the runtime boundary: [`REVIEW.html`](REVIEW.html). This artifact predates the prompt-defined output contract and compiles its two labels in — now treated as the restricted special case.

**Planned: [Qwen3.5-0.8B](models/qwen3.5-0.8b/README.md)** on an [Amlogic A113X target](models/qwen3.5-0.8b/targets/a113x/README.md) (4x Cortex-A53, 1-2 GB). Hybrid DeltaNet architecture, runtime tokenizer, full output head, per-call answer sets. Nothing is pinned or measured yet.

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

llama.cpp is the only real alternative. Kernel techniques are not what separates the stacks — its NEON vectorization is imported here as CPU-axis step 4, and both then face the same DRAM bandwidth wall. What a generic stack cannot absorb: per-call answer-set scoring (~1.4x per decision), 3-10x smaller non-weight RSS, a ~100 KB dependency-free binary, a tensor-verified DeltaNet path, and per-target kernel/layout specialization. The on-device baseline measurement includes llama.cpp, and its numbers are recorded alongside ours.

## Build and test

```sh
make test
```

Model-specific compile and run commands live in each model record, e.g. the [Gemma 4 E2B build](models/gemma-4-e2b/README.md#build).

## Current limits

- no runtime tokenizer or arbitrary free-text input yet (planned for the Qwen artifact);
- no SIMD kernel yet;
- no full-graph tensor-by-tensor differential test;
- no held-out application-quality evaluation;
- one bounded Gemma 4 profile implemented; the cross-model compiler remains a design.

## License

No license has been selected.
