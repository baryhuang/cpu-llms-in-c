# Qwen3.8-27B

Qwen3.8-27B (released 2026-08, Apache-2.0) is architecturally identical
to Qwen3.6-27B. This was verified against the raw Hugging Face files,
not assumed: both checkpoints carry the same 1,199 language-model
tensor names with the same shapes, byte-identical `config.json` model
fields (only `transformers_version` differs), the same
27,781,427,952-parameter count, and the same `qwen3_5` architecture
class. The differences are new training, a new chat template
(`reasoning_effort` levels, `preserve_thinking` defaulting to true), a
deeper recommended MTP draft depth (3), and seven added audio/TTS
special tokens in the previously padded id space (248070-248076); the
base vocabulary and merges are unchanged.

Because the graph is unchanged, this repository runs Qwen3.8-27B
through the same compiled-image format and the same C/Metal runtime as
Qwen3.6-27B — the port is a repack, not a rewrite. See the
[Apple M3 Pro target](targets/apple-m3-pro/README.md).

| Source | Pin |
|---|---|
| Weights | `mlx-community/Qwen3.8-27B-4bit` revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (affine Q4, group 64 — the same layout as the deployed 3.6 images) |
| MTP draft head | `mlx-community/Qwen3.8-27B-MTP-4bit` revision `b643c01b6d3b094e325edb6ebd832e16c486c575` (separate 239 MB checkpoint; same fc + one attention layer + three norms structure as 3.6's `mtp.*` tensors) |
| Official reference | `Qwen/Qwen3.8-27B` (BF16, 18 shards; all 15 `mtp.*` tensors sit in shard 18) |

One conversion fact worth recording: the official BF16 `mtp.*` norm
vectors use the Hugging Face delta convention (effective multiplier
`1 + w`), which the Qwen3.6 MTP packer folds at pack time; the
mlx-community MTP conversion arrives already folded to direct
multipliers, so the Qwen3.8 MTP packer copies them unchanged.
