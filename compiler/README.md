# Offline compiler

`compiler/` contains code that runs before deployment. It may inspect checkpoints, generate fixtures, search formats and emit immutable runtime images. None of it is required by the deployed inference process.

## Qwen3.8-27B / Apple M3 Pro

[`qwen3.8-27b/apple-m3-pro/`](qwen3.8-27b/apple-m3-pro/) contains the complete target image compiler:

| Entry | Output / responsibility |
|---|---|
| `qwen38_compile_runtime_images.sh` | Verifies three checkpoint shards and orchestrates the base runtime-image build |
| `qwen38_m3_global_pack.c` | `global.q38global` |
| `qwen38_m3_pack.c` | DeltaNet-layer `.q38delta` images |
| `qwen38_m3_attention_pack.c` | Full-attention and MTP-layer `.q38att` images |
| `qwen38_tokenizer_pack.c` | `tokenizer.q38tok` |
| `qwen38_mtp_pack.c` | `mtp.q38mtp` |
| `qwen38_safetensors_inspect.c` | Source checkpoint tensor/offset inspection |

The public compile procedure and pinned hashes are in the [target guide](../models/qwen3.8-27b/targets/apple-m3-pro/README.md).

## Shared and earlier model adapters

The files currently at the root of `compiler/` are offline reference, fixture and image-generation utilities for Gemma 4 E2B, Qwen3.5-0.8B and Whisper small.en. Shared safetensors and Q4 helpers also remain here. They are not target commands.
