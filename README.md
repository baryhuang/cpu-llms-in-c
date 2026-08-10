# cpu-llms-in-c

Compile a language model and a bounded workload into a packed model image and a C runtime for CPU-only inference. The target does not require Python, PyTorch, llama.cpp, ONNX Runtime, or another inference framework.

**Review the exact test inputs and outputs in [`REVIEW.html`](REVIEW.html).**

## Measured prototype

The first complete prototype specializes Gemma 4 E2B for a two-label hazard smoke workload.

| Result | Measured value |
|---|---:|
| Executed graph | 35 text layers |
| Packed image | 966,579,776 bytes (921.8 MiB) |
| Peak RSS, warm run | 948,224 KiB (926 MiB) |
| Swap | 0 |
| Warm prefill | 0.598 tokens/s |
| Warm decision step | 0.621 tokens/s |
| Cold prefill | 0.528 tokens/s |
| Cold decision step | 0.576 tokens/s |
| Q4 decisions against written labels | 12/12 |
| BF16-reference decisions against written labels | 11/12 |
| Q4/BF16 decision agreement | 11/12 |

The twelve cases are obvious smoke inputs, not a safety benchmark. The current artifact accepts only inputs compiled from the profile; it is not a general text-generation runtime.

Raw measurements, per-case logits, timings, hashes, and limitations are in [`models/gemma-4-e2b/results.json`](models/gemma-4-e2b/results.json).

## Build and run

Run the committed correctness tests:

```sh
make test
```

Compile the task image on a machine that holds the pinned checkpoint:

```sh
python3 tools/compile_gemma4_task_image.py \
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
      safe/danger logits and timing
```

The C runtime implements the full specialized text path: local and global RoPE, 15 physical K/V states, late shared K/V, double-wide late MLPs, PLE, final normalization, and the constrained LM head. The current matrix kernel is scalar Q4 GEMV with optional OpenMP.

## Repository

| Path | Contents |
|---|---|
| [`include/`](include/) | Public C interfaces |
| [`src/`](src/) | C runtimes |
| [`tools/`](tools/) | Offline compiler, reference evaluator, and hardware probe |
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
