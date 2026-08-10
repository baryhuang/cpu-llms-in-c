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
| [`models/`](models/) | Model-specific profile, pins, results, and implementation notes |
| [`tests/`](tests/) | Synthetic and real-weight correctness tests |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Cross-model compiler/runtime contract |
| [`REVIEW.html`](REVIEW.html) | Human-readable test inputs, outputs, commands, and runtime boundary |

Start with the [Gemma 4 E2B record](models/gemma-4-e2b/README.md) for the implemented artifact. Large checkpoints, generated model images, generated binaries, and credentials are excluded from the repository.

## Current limits

- no runtime tokenizer or arbitrary free-text input;
- no SIMD kernel yet;
- no full-graph tensor-by-tensor differential test;
- no held-out application-quality evaluation;
- one bounded Gemma 4 profile implemented; the cross-model compiler remains a design.

## License

No license has been selected.
