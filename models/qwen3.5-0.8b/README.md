# Qwen3.5-0.8B

Status: model runtime implemented and verified; the A113X target is measured and CPU-specialized. This file lists model-axis optimizations that hold on every target. Each target file pins one CPU/SoC and records only the optimizations and measurements for that combination.

## Target matrix

| Target | Chip year | CPU | Accelerator | Current evidence | Record |
|---|---:|---|---|---|---|
| Amlogic A113X | 2017 | 4x Cortex-A53 | CPU only for this project | 3.6353 prompt tokens/s classification prefill; 2.6005 tokens/s steady greedy decode; zero swap | [target](targets/a113x/README.md) · [raw results](targets/a113x/results.json) |
| Rockchip RK3588S | 2022 | 4x Cortex-A76 + 4x Cortex-A55 | 6 TOPS NPU | external RKLLM baseline; no repository run | [target](targets/rk3588s/README.md) |
| Rockchip RK3576 | 2024 | 4x Cortex-A72 + 4x Cortex-A53 | 6 TOPS RKNN NPU | external RKLLM baseline; no repository run | [target](targets/rk3576/README.md) |

Chip year uses the release-event evidence in the target record, not a board manufacture date. Equal TOPS labels do not make two NPUs interchangeable: supported data paths, memory interfaces, core organization, and software differ.

## Source

| Field | Value |
|---|---|
| Source | `Qwen/Qwen3.5-0.8B` (Apache 2.0) |
| Checkpoint revision | `2fc06364715b967f1860aea9cf38778875588b17` (2026-03-02) |
| Checkpoint size | 1,746,942,600 bytes, single safetensors file |
| Checkpoint sha256 | from LFS metadata, re-verified locally before compiling |
| Reference oracle | `huggingface/transformers` `fd12552d`, `models/qwen3_5/` (pinned in [`pins.json`](pins.json)) |
| Runtime contract | prompt-defined outputs per [`ARCHITECTURE.md`](../../ARCHITECTURE.md): runtime tokenizer, full output head, optional per-call answer sets |

All file hashes are in [`pins.json`](pins.json).

## Architecture (from the published config)

| Constant | Value |
|---|---:|
| Text layers | 24 (repeating 3 linear + 1 full attention) |
| Gated-DeltaNet linear layers | 18 |
| Full-attention layers | 6 (GQA, 8 query / 2 KV heads, head dim 256) |
| Hidden size | 1,024 |
| MLP width | 3,584 (SiLU gated) |
| Linear-attention heads | 16 key/value heads, dim 128, conv kernel 4 |
| Vocabulary | 248,320 (tied embeddings) |
| RoPE | MRoPE sections [11, 11, 10], partial rotary 0.25, theta 1e7 |
| Attention output gate | `attn_output_gate: true` on full-attention layers |
| DeltaNet state dtype | float32 (`mamba_ssm_dtype`) |
| Context | 262,144 (we compile a much smaller bound) |
| Vision tower | 12 layers — removed at compile time |
| MTP head | 1 layer (`mtp_num_hidden_layers`) — removed at compile time |

The config declares 248,320 embedding rows but the tokenizer defines 248,070 entries; the remainder is padding plus special tokens (vision start/end 248053/248054, image 248056, video 248057, EOS 248044).

The Gated-DeltaNet layers keep one fixed-size state matrix per head (about 1 MB per layer, ~18 MB total) instead of a growing KV cache; only the 6 full-attention layers keep KV. Prefill through the linear layers is sequential in the token dimension; their projections still batch.

## Model-axis optimizations (valid on any CPU)

Ordered by expected effect. Estimates are analytical; none is a benchmark result.

1. **Model swap.** Per-token matrix traffic drops from Gemma 4 E2B's 960 MB to roughly 290 MB of Q4 non-embedding weights — about 3.3x less work per token before any kernel change.
2. **Exact graph rewrites.** Remove the vision tower; for text-only input the three MRoPE sections share one position index, so MRoPE reduces exactly to standard RoPE over the first 64 rotary dimensions. Both are provable compile-time rewrites, like the Gemma PLE folding.
3. **Batched prefill.** Process all prompt tokens per weight pass in the MLPs, projections, and full-attention layers, reading each matrix once per layer instead of once per token. For a 40-token prompt this cuts prefill weight traffic by up to ~40x; DeltaNet state updates stay sequential but ride on batched projections.
4. **Per-call answer-set scoring.** When the caller supplies allowed answers, score only those head rows after prefill instead of the full 248,320-row tied head, saving ~130 MB of traffic per decision. Free generation pays the full head as its inherent cost.
5. **Q4 group-128 quantization.** Existing model-independent format and quantizer, reused as-is. Estimated image: ~420 MB (~130 MB embeddings/head + ~290 MB matrices).
6. **Minor.** Precomputed RoPE tables; descriptor index instead of linear name lookup; deterministic thread partitions.

## Verification plan

Same gates as the Gemma record: independent NumPy BF16 reference with per-tensor boundaries — extended with DeltaNet checkpoints (post-conv, post-gate, post-state-update, post-output-gate) — a committed layer fixture, tokenizer round-trip tests against the pinned `tokenizer.json`, and target measurements reported separately from verification.

## Artifact (image v1)

| Field | Value |
|---|---|
| Format | `QW35TSK1` version 2 |
| Image size | 492,962,304 bytes (~470 MiB) |
| Image sha256 | `00bb5a62214f1db4...` (full value in the build manifest) |
| Quantization | Q4 group 128 (97 matrices incl. tied embedding/head); Q8 group 128 for the 54 DeltaNet projections; float32 small tensors |
| Tokenizer tables | vocab byte strings, ranked merges, byte map, and special tokens (~6.7 MB); the runtime tokenizes prompt text itself |
| Estimated decode traffic | ~350 MB per token (quantized matrices minus embedding table) |

**Quantization sensitivity finding.** With every matrix at min/max Q4, smoke decisions flip against the BF16 reference (danger scored below safe on an obvious danger case), and MSE-optimal Q4 scales do not recover them. Class ablation over the NumPy reference isolates the damage: promoting only the DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `out_proj`) to Q8 restores 4/4 smoke decisions with margins >= 1.0; promoting attention instead leaves thin margins (+0.18) and promoting the MLPs still fails a case. The Q4-quantized embedding/head is harmless. This matches the expectation that the recurrent DeltaNet state amplifies weight noise. Four smoke cases are a signal, not an evaluation; the 12-case set and logit deltas run once the C runtime executes the image.

## Verification (12-case smoke evaluation)

Verification is separate from benchmark timing. Full per-case logits: [`results.json`](results.json).

| Check | Result |
|---|---|
| Image (Q4/Q8) decisions / written labels | 10/12 — two false negatives on danger cases, margins -0.18 and -0.05 |
| BF16-reference decisions / written labels | 10/12 — two false positives on safe cases, margins +0.29 and +0.05 |
| Image / reference decision agreement | 8/12 — every disagreement is a case where one side sits within ±0.3 |
| Two-shot prompt mitigation | recovers `danger_robot_entry`, narrows `danger_uncontrolled_pressure` to -0.04, leaves a safe control unchanged — a caller-side fix consistent with prompt-defined outputs |

The 0.8B model is marginal zero-shot on this smoke set (the BF16 reference itself misses two cases); quantization moves borderline cases across the line rather than degrading clear ones. Twelve obvious cases are a smoke signal, not an evaluation.

## Benchmarks

The unpinned x86-64 development run is retained only as implementation evidence:

| Quantity | Value |
|---|---:|
| Total prompt tokens | 488 |
| Total classification duration | 53.29 s |
| Aggregate throughput | 9.16 tokens/s |
| Peak RSS | 377,888 KiB (369 MiB) |

Peak RSS sits below the 470 MiB image because untouched embedding rows are never faulted in — a direct effect of per-call answer-set scoring.

The A113X target runs the same image and 12 cases. This is prompt prefill/classification throughput, not free-generation decode throughput. The rows are cumulative so the CPU-axis increment is visible:

| A113X implementation | Total classification duration | Throughput | Increment vs previous | Peak RSS | Swap |
|---|---:|---:|---:|---:|---:|
| Generic scalar | 592.942 s | 0.8230 token/s | 1.00x | 375,764 KiB | 0 |
| + NEON Q4/Q8 GEMV | 200.846 s | 2.4297 token/s | 2.95x | 375,972 KiB | 0 |
| + contiguous/parallel DeltaNet state | 134.241 s | **3.6353 token/s** | 1.50x | 376,152 KiB | 0 |

The two CPU increments compose to 4.42x over the generic scalar runtime. Full machine fields, wall durations, output logits, and build hashes: [`targets/a113x/results.json`](targets/a113x/results.json).

Free generation is also implemented. One measured greedy run scans the full 248,320-row head per generated token: 6.7701 s time to first token and **2.6005 tokens/s steady decode**, with 499,968 KiB peak RSS and zero swap. Input, output text, token IDs, and per-token durations are in the [A113X target record](targets/a113x/README.md#free-generation-decode-benchmark).

## Verified facts

| Check | Result |
|---|---|
| Source pins | done — revision, all file hashes, and reference oracle in [`pins.json`](pins.json) |
| Answer candidates are single tokens | `safe`=18112, `danger`=30416, `yes`, `no`, `0`, `1`, `A`-`D`, `安全`=96520, `危险`=98693 — all single tokens (pinned tokenizer, `tokenizers` 0.23.1) |
| NumPy reference matches the oracle | [`compiler/qwen35_reference.py`](../../compiler/qwen35_reference.py) matches the pinned transformers `Qwen3_5TextModel` forward to 2e-4 on a tiny random hybrid model ([`tests/test_qwen35_reference.py`](../../tests/test_qwen35_reference.py)) |
| C layer matches the fixture exactly | [`targets/generic/qwen35_layer.c`](targets/generic/qwen35_layer.c) matches all 17 declared boundaries of the committed DeltaNet + full-attention fixture with max_abs 0; the fixture itself is tied back to the NumPy reference ([`tests/test_qwen35_fixture.py`](../../tests/test_qwen35_fixture.py)) |
| C runtime matches the reference on real weights | [`targets/generic/qwen35_task.c`](targets/generic/qwen35_task.c) executing the compiled image reproduces the NumPy-reference answer logits to ~1e-4 on all four smoke cases (4/4 decisions, margins >= 1.0); ~9.5 prefill tokens/s with scalar kernels, 4 threads, on the x86-64 dev machine — an informal dev number, not a target benchmark |
| C tokenizer matches the pinned tokenizer | 20/20 exact id-sequence parity against `tokenizers` on the 12 chat-templated hazard cases plus Chinese, contraction, punctuation, and whitespace stress strings; `--prompt` text produces logits identical to the `--ids` path |
| Chinese zero-shot weakness is the model, not the stack | a Chinese hazard prompt with the runtime answer set `安全/危险` picks the wrong label, and the BF16 reference picks the same wrong label — direction agrees, so the artifact is faithful; 0.8B classification benefits from few-shot examples in the prompt |

## Open items

- Run a larger held-out application-quality evaluation; the current 12 cases remain smoke evidence.
- Measure an equivalent llama.cpp build on the same board before making a measured stack-to-stack performance claim.
- Evaluate TBL lookup and multi-token prefill against the recorded A113X CPU increments.
- The tokenizer pretokenizer approximates `\p{L}\p{N}` with ASCII classes plus treating all non-ASCII bytes as letters; exact on the parity corpus, but non-ASCII punctuation may split differently — extend the corpus before relying on exotic scripts.
