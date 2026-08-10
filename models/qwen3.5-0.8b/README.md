# Qwen3.5-0.8B

Status: planned. Source pins exist; no artifact or target measurement exists yet. This file lists model-axis optimizations that hold on every target. Each target file pins one CPU/SoC and records only the optimizations and measurements for that combination.

## Target matrix

| Target | Chip year | CPU | Accelerator | Current evidence | Record |
|---|---:|---|---|---|---|
| Amlogic A113X | 2017 | 4x Cortex-A53 | CPU only for this project | analytical estimate only | [target](targets/a113x/README.md) |
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

- Formal 12-case evaluation harness writing `results.json` (benchmark/verification separation as in the Gemma record).
- Cross-compile and measure the A113X baseline; llama.cpp measured alongside.
- The tokenizer pretokenizer approximates `\p{L}\p{N}` with ASCII classes plus treating all non-ASCII bytes as letters; exact on the parity corpus, but non-ASCII punctuation may split differently — extend the corpus before relying on exotic scripts.
