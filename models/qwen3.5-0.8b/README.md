# Qwen3.5-0.8B

Status: planned. No artifact, pins, or measurements exist yet. This record and [`targets/a113x/`](targets/a113x/) hold the optimization plan, split by axis: this file lists model-axis optimizations that hold on any CPU; the target file lists CPU-axis optimizations for one selected CPU.

## Source

| Field | Value |
|---|---|
| Source | `Qwen/Qwen3.5-0.8B` (Apache 2.0) |
| Checkpoint pin | not yet pinned |
| Runtime contract | prompt-defined outputs per [`ARCHITECTURE.md`](../../ARCHITECTURE.md): runtime tokenizer, full output head, optional per-call answer sets |

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
| Context | 262,144 (we compile a much smaller bound) |
| Vision tower | 12 layers — removed at compile time |

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

## Open items

- Pin checkpoint/config/tokenizer hashes.
- Verify the runtime BPE tokenizer against the pinned vocabulary.
- Decide the compiled sequence bound for the first artifact.
