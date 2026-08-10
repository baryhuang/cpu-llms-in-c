# cpu-llms-in-c

Compile a language model and a bounded workload into a packed model image and a C runtime for CPU-only inference. The target does not require Python, PyTorch, llama.cpp, ONNX Runtime, or another inference framework.

**Review the exact test inputs and outputs in [`REVIEW.html`](REVIEW.html).**

## Benchmark

The first complete prototype specializes Gemma 4 E2B for a two-label hazard smoke workload.

The C/Q4 benchmark used two CPU threads. Verification work is excluded from these durations.

| Benchmark phase | Work | Duration | Throughput |
|---|---:|---:|---:|
| Offline image compilation | 275 Q4 matrices, 112 token rows | 28.246025 s | — |
| Warm classification | 493 prompt tokens, 12 cases | 823.854355 s | 0.598407 tokens/s |
| Warm extra label-token decode | 12 tokens | 19.338979 s | 0.620508 tokens/s |
| Warm process wall time | complete 12-case run | 843.24 s | — |
| Cold classification | case 0, 43 prompt tokens | 81.401387 s | 0.528247 tokens/s |
| Cold extra label-token decode | case 0, 1 token | 1.736948 s | 0.575723 tokens/s |
| Cold process wall time | complete case-0 run | 83.16 s | — |

Warm peak RSS was 948,224 KiB (926 MiB), with zero swap. `classification` covers the prompt forward pass, final normalization, two logits, and the one-bit comparison. The extra label-token decode happens after the decision and is not required to return it.

## Verification

Verification is reported separately from benchmark timing.

| Verification | Result |
|---|---:|
| Q4 decisions against written labels | 12/12 |
| BF16-reference decisions against written labels | 11/12 |
| Q4/BF16 decision agreement | 11/12 |
| Real-weight layer-0 tensor boundaries | 10/10, maximum absolute error 0 |
| BF16-reference execution duration | 24.875661 s compute; 26.67 s process wall time |

The BF16 reference uses batched NumPy execution and its duration is not a C-runtime benchmark. The twelve cases are obvious smoke inputs, not a safety benchmark. The current artifact accepts only inputs compiled from the profile; it is not a general text-generation runtime.

Raw measurements, per-case logits, timings, hashes, and limitations are in [`models/gemma-4-e2b/results.json`](models/gemma-4-e2b/results.json).

## Build and run

Run the committed correctness tests:

```sh
make test
```

Compile the task image on a machine that holds the pinned checkpoint:

```sh
python3 compiler/compile_gemma4_task_image.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output hazard-v1.g4task
```

Build and execute the C runtime:

```sh
make OMPFLAGS=-fopenmp build/gemma4-task
OMP_NUM_THREADS=2 build/gemma4-task hazard-v1.g4task all
```

The model image is intentionally not stored in Git.

## Implementation

```text
checkpoint + tokenizer + task profile
                 |
                 v
        offline Python compiler
                 |
        folded PLE + Q4 matrices
        reachable token rows only
        two output-label rows
                 |
                 v
       G4TASK01 packed image
                 |
                 v
         mmap C11 runtime
                 |
       one-bit decision + diagnostic logits
```

The C runtime implements the full specialized text path: local and global RoPE, 15 physical K/V states, late shared K/V, double-wide late MLPs, PLE, final normalization, and the constrained LM head. The current matrix kernel is scalar Q4 GEMV with optional OpenMP.

## Output contract

The application result is one bit: `0 = safe`, `1 = danger`. The current diagnostic runtime computes two FP32 logits and compares them. Because the final soft cap is monotonic, a decision-only compiler can replace the two output rows with one exact difference row, `W_danger - W_safe`, and test one score against zero.

The measured decode step feeds the selected label token through another 35 layers. That step is for decode measurement; the binary decision is already available after prefill and does not require it.

## Repository

| Path | Contents |
|---|---|
| [`runtime/`](runtime/) | C runtime, headers, layer reference, and hardware probe |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| [`models/`](models/) | One directory per model; per-CPU results live under `models/<model>/targets/<cpu>/` (model first, CPU second) |
| [`tests/`](tests/) | Synthetic and real-weight correctness tests |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Cross-model compiler/runtime contract |
| [`REVIEW.html`](REVIEW.html) | Human-readable test inputs, outputs, commands, and runtime boundary |

Start with the [Gemma 4 E2B record](models/gemma-4-e2b/README.md) for the implemented artifact and the [Qwen3.5-0.8B record](models/qwen3.5-0.8b/README.md) for the planned next one. Large checkpoints, generated model images, generated binaries, and credentials are excluded from the repository.

## Optimization roadmap

Optimizations split along the two repository axes and stack. Model-axis steps hold on any CPU and live in the model record; CPU-axis steps hold for one pinned CPU and live under `targets/<cpu>/`. The factors below are analytical estimates against the measured Gemma baseline (0.598 tokens/s, scalar kernel, two threads) — they are not benchmark results, they do not multiply cleanly, and the stack is capped by the target's memory bandwidth.

| Step | Axis | Mechanism | Estimated effect |
|---|---|---|---|
| 1. Switch to Qwen3.5-0.8B | model | per-token matrix traffic 960 MB → ~290 MB | ~3.3x per token |
| 2. Batched prefill | model | read each matrix once per layer per prompt, not per token | up to ~40x on prompt prefill |
| 3. Per-call answer-set scoring | model | skip the 248K-row output head unless generating | ~130 MB saved per decision |
| 4. NEON kernel | CPU | imported llama.cpp-style vectorized GEMV, then T-MAC-style TBL lookup; the faster is kept | ~3-5x kernel throughput |
| 5. Four-thread static partition | CPU | all A113X cores with deterministic reductions | up to ~2x, bandwidth-capped |
| 6. Lower-bit LUT (experimental) | CPU | Q3/Q2 with linear LUT cost scaling, only if task quality survives | further 1.3-2x |

Details and ordering: model axis in [`models/qwen3.5-0.8b/README.md`](models/qwen3.5-0.8b/README.md), CPU axis in [`models/qwen3.5-0.8b/targets/a113x/README.md`](models/qwen3.5-0.8b/targets/a113x/README.md). Each step lands only after the tensor-level correctness tests for the code it touches.

## Rewrite versus off-the-shelf stacks (theoretical)

The comparison target is Qwen3.5-0.8B on the provisional A113X device (4x Cortex-A53, 1-2 GB RAM). All numbers are analytical, not measurements.

| Stack | Runs on the 1 GB board | Decode traffic per token | Per-decision head cost | Overhead beyond weights | Qwen3.5 hybrid support |
|---|---|---:|---:|---:|---|
| This C runtime (plan) | yes (~420 MB image + ~30 MB) | ~290 MB | a few head rows | ~30 MB, zero dependencies | own DeltaNet implementation, tensor-verified |
| llama.cpp (Q4 GGUF) | yes (~0.5 GB + context/scratch) | ~290 MB + 130 MB full head every token | full 248K-row head | ~100-300 MB, single binary | GATED_DELTA_NET op landed 2026, basic vector CPU path |
| PyTorch + Transformers | no — BF16 weights alone ~1.6 GB | ~1.6 GB | full head | Python + PyTorch, ~1 GB+ | reference implementation (the oracle) |
| ONNX Runtime | no practical path | — | — | — | zero non-softmax attention operators as of 2026 |

What the table implies:

- **PyTorch and ONNX are not candidates on this class of device.** PyTorch does not fit the memory; ONNX cannot express the DeltaNet layers without decomposing into tens of primitive ops per step.
- **The real off-the-shelf comparison is llama.cpp.** Its NEON kernels beat a scalar baseline, but kernel techniques are CPU-axis optimizations, not properties of a stack: the same vectorization is imported as a CPU-axis step here (see the target record), so any kernel gap is transient. Both stacks then face the same DRAM bandwidth wall.
- The rewrite's structural advantages are the parts an upstream generic stack cannot absorb: per-call answer-set scoring skips ~130 MB of head traffic per decision (~1.4x on decision latency, more on short prompts); roughly 3-10x smaller non-weight RSS leaves headroom on the 1 GB board; a dependency-free ~100 KB static binary; a tensor-verified DeltaNet path instead of a freshly landed one; and per-target kernel and layout specialization beyond what a general project ships.
- If off-the-shelf llama.cpp on the device meets the latency and memory gates, that result is recorded too — the baseline measurement includes it.

## Current limits

- no runtime tokenizer or arbitrary free-text input;
- no SIMD kernel yet;
- no full-graph tensor-by-tensor differential test;
- no held-out application-quality evaluation;
- one bounded Gemma 4 profile implemented; the cross-model compiler remains a design.

## License

No license has been selected.
