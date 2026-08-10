# Gemma 4 E2B

Implemented status: complete 35-layer text graph for one compiled two-label profile. General text inference is not implemented.

## Result

The measurement used the pinned official checkpoint, a generated Q4 image, the C11 runtime, and two CPU threads on an Ubuntu x86-64 test system.

| Metric | Result |
|---|---:|
| Image size | 966,579,776 bytes (921.8 MiB) |
| Peak RSS, warm run | 948,224 KiB (926 MiB) |
| Swap | 0 |
| Warm prefill | 0.598 tokens/s |
| Warm one-step decision | 0.621 tokens/s |
| Cold prefill | 0.528 tokens/s |
| Cold one-step decision | 0.576 tokens/s |
| Q4 decisions / written labels | 12/12 |
| BF16-reference decisions / written labels | 11/12 |
| Q4/BF16 agreement | 11/12 |

The only Q4/reference disagreement was `safe_empty_zone`: the BF16 reference selected `danger`; Q4 selected the written label `safe`. This boundary crossing does not establish that Q4 is better.

The twelve inputs are a smoke set, not a safety evaluation. Open [`../../REVIEW.html`](../../REVIEW.html) to inspect every input and output. Unrounded logits, timings, hashes, page-fault counts, and limitations are in [`results.json`](results.json). The input contract is in [`profile.json`](profile.json).

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

## Build

Compiler dependencies are Python, NumPy, and `tokenizers`. They are not runtime dependencies.

```sh
python3 tools/compile_gemma4_task_image.py \
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
python3 tools/evaluate_gemma4_task_reference.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output reference.json
```

## Runtime cost

The warm run processed 493 prompt tokens in 823.854 seconds and 12 decision steps in 19.339 seconds. Peak RSS remained below 1 GiB with no swap.

At the measured aggregate rates, logical matrix access was approximately 575 MB/s during prefill and 596 MB/s during the decision step. These are bytes visited by the runtime, not storage-throughput measurements. The model image must remain resident; paging the image from flash for every token is not viable.

The cold case incurred 325 major page faults and 1,886,584 filesystem input blocks before reaching the same logits as the warm case. On Linux, interpreting those blocks as 512-byte units gives approximately 966 MB, close to the complete image size.

## Validation

```sh
make test
```

This runs:

- Python tests for image packing, reference operations, and sparse checkpoint access;
- a miniature C layer fixture covering ten declared tensor boundaries;
- the committed profile/tokenization checks.

A separate ignored 144,857,148-byte fixture was exported from the pinned official weights. Its layer-0 output matched all ten declared boundaries with maximum absolute error 0. Details remain in [`layer0-validation.json`](layer0-validation.json).

## Target and remaining work

The constrained hardware design target is an Intel Celeron J3455 with SSE4.2, 3.34 GiB visible RAM, a 960 MiB RSS target, a 1,024 MiB hard limit, and no inference swap. The current complete-graph numbers were collected on a different two-vCPU x86-64 system; they are not J3455 benchmark claims.

Not implemented:

- runtime tokenization and arbitrary free-text fields;
- fixed-prefix state snapshots;
- SSE4.2 packed GEMV and static worker scheduling;
- full-graph tensor-boundary comparison;
- held-out, adversarial, calibration, or distribution-shift evaluation;
- codebook/LUT compression or conditional MLP trees.

The next useful change is a bounded schema with variable text, followed by tokenizer support and a held-out dataset. SIMD work should follow a tensor-level Q4 correctness test so numerical and kernel changes remain separable.

## Primary references

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 technical report](https://arxiv.org/abs/2607.02770)
- [Transformers Gemma 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [Intel Celeron J3455 specifications](https://www.intel.com/content/www/us/en/products/sku/95594/intel-celeron-processor-j3455-2m-cache-up-to-2-30-ghz/specifications.html)
