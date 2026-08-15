# Qwen3.8-27B

This directory contains the Qwen3.8-27B model adapter and its target implementations. The current artifact compiles the language model for text generation; it does not include the checkpoint's vision tower.

## Model graph

Values below are read from the pinned checkpoint and enforced by the runtime headers and image packers.

| Property | Value |
|---|---:|
| Language-model layers | 64 |
| Layer pattern | 48 gated DeltaNet layers, 16 full-attention layers; full attention every fourth layer |
| Hidden size | 5,120 |
| MLP intermediate size | 17,408 |
| Vocabulary rows | 248,320 |
| Full attention | 24 query heads, 4 KV heads, head dimension 256, partial rotary dimension 64 |
| DeltaNet | 16 key heads, 48 value heads, head dimension 128, convolution width 4 |
| Maximum position setting | 262,144 tokens in the source configuration; target runtime limits are smaller |
| MTP | one separately packed draft layer plus projection and normalization tensors |

The hybrid graph changes the optimization problem. DeltaNet layers carry recurrent and convolution state instead of a growing KV cache; full-attention layers use grouped-query attention and do require KV storage. A useful target backend must optimize both paths and the transitions between them.

## Pinned sources

| Input | Revision / SHA-256 |
|---|---|
| Affine Q4 weights | `mlx-community/Qwen3.8-27B-4bit` at `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| Weight shard 1 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| Weight shard 2 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| Weight shard 3 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |
| MTP draft checkpoint | `mlx-community/Qwen3.8-27B-MTP-4bit` at `b643c01b6d3b094e325edb6ebd832e16c486c575`; SHA-256 `76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6` |
| Tokenizer | checkpoint `tokenizer.json`; SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Numerical reference | `Qwen/Qwen3.8-27B` BF16 checkpoint |

The deployed weights use affine four-bit groups of 64 values. MTP normalization tensors in the quantized checkpoint are already stored as direct multipliers and are copied without adding one during packing.

## Targets

| Target | Status | Entry point |
|---|---|---|
| Apple M3 Pro | End-to-end text generation, batched prefill, adaptive MTP and local serving measured on device | [target documentation](targets/apple-m3-pro/README.md) |

Quality and performance are target-specific. The five-case ARC-Easy adaptation is a smoke test rather than an official benchmark score; its raw outputs are recorded in [`benchmarks/arc-easy-5/results-macos-m3-pro.json`](benchmarks/arc-easy-5/results-macos-m3-pro.json).
