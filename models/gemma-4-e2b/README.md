# Gemma 4 E2B

Implemented status: complete 35-layer text graph for one compiled two-label profile. General text inference is not implemented.

## Benchmark

The C/Q4 benchmark used the pinned checkpoint, generated image, C11 runtime, two CPU threads, and an Ubuntu x86-64 test system. BF16 reference execution and correctness tests are excluded from this section.

| Phase | Work | Duration | Throughput |
|---|---:|---:|---:|
| Offline image compilation | 275 Q4 matrices, 112 token rows | 28.246025 s | — |
| Warm classification | 493 prompt tokens, 12 cases | 823.854355 s | 0.598407 tokens/s |
| Warm extra label-token decode | 12 tokens | 19.338979 s | 0.620508 tokens/s |
| Warm process wall time | complete 12-case run | 843.24 s | — |
| Cold classification | case 0, 43 prompt tokens | 81.401387 s | 0.528247 tokens/s |
| Cold extra label-token decode | case 0, 1 token | 1.736948 s | 0.575723 tokens/s |
| Cold process wall time | complete case-0 run | 83.16 s | — |

Warm peak RSS was 948,224 KiB (926 MiB), with zero swap. Cold peak RSS was 946,176 KiB (924 MiB). The warm aggregate logically visited approximately 575 MB/s of matrix data during classification and 596 MB/s during the extra decode. These are memory-access rates, not storage measurements.

The cold case produced 325 major page faults and 1,886,584 filesystem input blocks. Interpreting Linux input blocks as 512-byte units gives approximately 966 MB, close to the full image size.

`classification duration` ends after final normalization, two logits, and the one-bit comparison. `extra label-token decode duration` measures a subsequent 35-layer token step and is not required to return the binary result.

## Verification

Verification is not included in the benchmark durations above.

| Check | Result |
|---|---:|
| Q4 decisions / written labels | 12/12 |
| BF16-reference decisions / written labels | 11/12 |
| Q4/BF16 agreement | 11/12 |
| Real-weight layer-0 tensor boundaries | 10/10, maximum absolute error 0 |
| BF16 reference compute duration | 24.875661 s |
| BF16 reference process wall time | 26.67 s |

The only Q4/reference disagreement was `safe_empty_zone`: the BF16 reference selected `danger`; Q4 selected the written label `safe`. This boundary crossing does not establish that Q4 is better.

The BF16 reference is an independent batched NumPy execution. Its duration is reported for reproducibility, not as target-runtime performance. The twelve inputs are a smoke set, not a safety evaluation. Open [`../../REVIEW.html`](../../REVIEW.html) to inspect every input and output. Unrounded data are in [`results.json`](results.json); the input contract is in [`profile.json`](profile.json).

## Artifact

| Field | Value |
|---|---|
| Source | `google/gemma-4-E2B-it` |
| Checkpoint revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Checkpoint size | 10,246,621,918 bytes |
| Packed format | `G4TASK01` |
| Weight format | signed Q4, group 128, BF16 scale |
| Packed matrices | 275 |
| Matrix bytes visited per token | 960,638,976 |
| Compiled token rows | 112 |
| Maximum compiled prompt | 44 tokens |
| Output rows | `safe`, `danger` |

The packed image and checkpoint are not committed. Source hashes are recorded in [`pins.json`](pins.json); generated-artifact hashes are recorded in [`results.json`](results.json).

## Implemented graph

| Constant | Value |
|---|---:|
| Text layers | 35 |
| Hidden size | 1,536 |
| Vocabulary | 262,144 |
| Query heads | 8 |
| KV heads | 1 |
| Local/global pattern | four local, one global |
| Local head dimension | 256 |
| Global head dimension | 512 |
| Local window | 512 |
| Physical K/V states | 15 |
| Base/late MLP width | 6,144 / 12,288 |
| PLE dimension per layer | 256 |

The runtime performs, in order:

1. scaled token embedding lookup;
2. local or global attention with Q/K/V normalization and RoPE;
3. late-layer K/V reuse from the last early local or global source;
4. gated GELU MLP;
5. per-layer embedding projection and residual;
6. final normalization and two-row LM-head evaluation.

## Compile-time specialization

The compiler applies only transformations used by this artifact:

- removes vision, audio, and MTP graphs;
- tokenizes the declared profile and retains its reachable embedding rows;
- folds fixed text-token PLE preprocessing into per-token, per-layer rows;
- omits the late K/V matrices absent from the shared-K/V forward path;
- quantizes executed matrices to signed group-128 Q4;
- retains only the two declared output rows;
- emits one immutable, memory-mapped image with offsets and checksums.

The deployed runtime does not tokenize, search kernels, allocate a model graph, or load Python.

## Output semantics

The task has two labels but one bit of application information:

```text
0 = safe
1 = danger
```

The current runtime retains two 1,536-value FP32 output rows, calculates `safe_logit` and `danger_logit`, then returns `danger_logit > safe_logit`. The JSON logits are diagnostic output; they are not two application bits.

For decision-only execution, the compiler can apply this exact rewrite:

```text
delta_weight = W_danger - W_safe
score = delta_weight · normalized_hidden
score > 0  -> danger
score <= 0 -> safe
```

Gemma's final `30 * tanh(logit / 30)` soft cap is monotonic, so comparing the raw projections preserves the selected label. This reduces two head rows to one. It does not reduce the 35-layer body.

After making the decision, the current benchmark path feeds the selected label token through all 35 layers. A deployment that returns only the bit can omit this step.

## Build

Compiler dependencies are Python, NumPy, and `tokenizers`. They are not runtime dependencies.

```sh
python3 compiler/compile_gemma4_task_image.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output hazard-v1.g4task

make OMPFLAGS=-fopenmp build/gemma4-task
OMP_NUM_THREADS=2 build/gemma4-task hazard-v1.g4task all
```

Run the independent BF16-weight reference:

```sh
python3 compiler/evaluate_gemma4_task_reference.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output reference.json
```

## Verification procedure

```sh
make test
```

This runs:

- Python tests for image packing, reference operations, and sparse checkpoint access;
- a miniature C layer fixture covering ten declared tensor boundaries;
- the committed profile/tokenization checks.

A separate ignored 144,857,148-byte fixture was exported from the pinned official weights. Its layer-0 output matched all ten declared boundaries with maximum absolute error 0. Details remain in [`layer0-validation.json`](layer0-validation.json).

## Remaining work

A deployment target CPU has not been selected. The current complete-graph numbers were collected on a two-vCPU x86-64 system; they are not claims about any other hardware. When a CPU target is selected, its pin and measured results move under `targets/<cpu>/` in this directory, following the model-first, CPU-second taxonomy in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).

Not implemented:

- runtime tokenization and arbitrary free-text fields;
- fixed-prefix state snapshots;
- ISA-specific packed GEMV and static worker scheduling;
- full-graph tensor-boundary comparison;
- held-out, adversarial, calibration, or distribution-shift evaluation;
- codebook/LUT compression or conditional MLP trees.

The next useful change is a bounded schema with variable text, followed by tokenizer support and a held-out dataset. SIMD work should follow a tensor-level Q4 correctness test so numerical and kernel changes remain separable.

## Primary references

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 technical report](https://arxiv.org/abs/2607.02770)
- [Transformers Gemma 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
