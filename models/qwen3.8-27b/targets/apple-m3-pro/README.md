# Qwen3.8-27B on Apple M3 Pro

Status: free-text generation runs end to end through the **same
compiled-image format, Metal kernels and runtime binaries as the
Qwen3.6-27B target** — the port changed no graph, kernel or runtime
logic, only source pins. Qwen3.8-27B is architecturally identical to
Qwen3.6-27B (verified tensor-by-tensor; see the
[model page](../../README.md)), so `build/qwen36-m3-chat`,
`tools/qwen36_serve.py` and every optimization landed on the 3.6 target
(half-tile MMA prefill with S64/S32/S16 buckets, FP16 KV cache,
adaptive multi-step MTP speculative decoding, incremental detokenizer)
apply to these images unchanged.

## Target pin

Machine, OS and toolchain: identical to the
[Qwen3.6-27B target pin](../../../qwen3.6-27b/targets/apple-m3-pro/README.md#target-pin)
(MacBook Pro `Mac15,6`, M3 Pro, 36 GB, macOS 15.7.3).

| Source | Pin |
|---|---|
| Weights | `mlx-community/Qwen3.8-27B-4bit` revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, affine Q4 group 64 |
| Shard SHA-256 | `6cc1508e…528c0d` / `83f2a20c…178670` / `31b8c91e…9e560a` (pinned in `qwen36_m3_image.h` as `QWEN38_M3_EXPECTED_SOURCE_SHA256*`) |
| MTP head | `mlx-community/Qwen3.8-27B-MTP-4bit` revision `b643c01b6d3b094e325edb6ebd832e16c486c575`, SHA-256 `76663c10…e4cc6` |
| Tokenizer | the checkpoint's `tokenizer.json`, SHA-256 `06b95093…a0e523`; identical base vocab and merges to 3.6 plus seven added audio/TTS ids in the padded space |

The mlx-community 3.8 conversion is layout-identical to the 3.6 one:
the same 2,180 tensor names and shapes, the same three-shard split with
layers 17 and 42 crossing shard boundaries, so
`tools/qwen38_compile_text_image.sh` is the 3.6 compile script with new
pins. Compiled output: the same 65 image files plus tokenizer,
15,138,643,968 mapped model bytes — byte-count-identical to the 3.6
images. The MTP images (`mtp-layer.q36att` 209,436,672 B, `mtp.q36mtp`
29,556,736 B) pack from the standalone quantized MTP checkpoint:
`build/qwen36-m3-attention-pack … 64` for the draft layer and
`build/qwen38-mtp-pack` for the fc/norm extras. Unlike the official
BF16 `mtp.*` tensors (Hugging Face delta norms, folded by the 3.6
packer), the mlx MTP norms arrive already folded to direct multipliers
and are copied unchanged.

## Build and run

```sh
tools/qwen38_compile_text_image.sh \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  tokenizer.json MODEL_DIR

make qwen38-mtp-pack
build/qwen36-m3-attention-pack MTP.safetensors MODEL_DIR/mtp-layer.q36att \
  76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6 64
build/qwen38-mtp-pack MTP.safetensors MODEL_DIR/mtp.q36mtp \
  76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6

build/qwen36-m3-chat MODEL_DIR build/qwen36-m3-q4.metallib \
  MODEL_DIR/tokenizer.q36tok
```

The serving stack needs no changes either:
`QWEN36_MODEL_DIR=tmp/qwen38-27b-runtime tools/qwen36_chat.sh` starts
the OpenAI-compatible server and the Chatbox client against these
images.

The serving layer implements the full 3.8 template semantics.
`tools/qwen36_serve.py --thinking` (or a per-request OpenAI-style
`reasoning_effort` field: `low` / `medium` / `xhigh`, `none` to
disable) renders the thinking-mode template with the verbatim
reasoning-effort system-message injections, opens the reply at
`<think>\n`, streams the think block as `reasoning_content` deltas and
the answer as `content`, and replays prior turns' reasoning
(`preserve_thinking`). Greedy thinking requests keep lossless
speculative decoding. Verified end to end against these images: an
xhigh request streamed 301 characters of reasoning and the correct
split answer; `reasoning_effort: none` reproduces the pinned no-think
behavior exactly; a low-effort request that exhausted a 400-token
budget mid-think returned the truncated thinking in
`reasoning_content` with empty `content` and `finish_reason: length`
(the OpenAI reasoning-model convention — give thinking mode a larger
token budget). The default remains the pinned no-think template.

## Verification

| Gate | Result |
|---|---|
| Shard and tokenizer sources | SHA-256-pinned at pack time; every image records its source hash |
| Async decode API state machine | 16/16 checks pass against the 3.8 images |
| Prefill-vs-decode state parity | 24/24 checks pass (exact/float-tile/half-tile modes, runs 16-96 tokens, argmax identical on every run, zero NaN) |
| Smoke | "What is 2+2" → `4`; C `max2` function correct; 法国首都 → `巴黎` |
| MTP | auto-enables against the 3.8 draft head; the code smoke accepted 19 drafts over 9 adaptive steps |
| Thinking mode | xhigh request: correct answer with reasoning split into `reasoning_content`; `none` reproduces no-think byte behavior |
| Quality (pinned ARC-Easy-5 smoke) | 3/5 strict at the 32-token budget, 4/5 at 96; every miss states the correct answer's content in prose without the required `Answer: X` line — under the no-think template 3.8 drifts toward explanation where 3.6 stays format-compliant (5/5). [Record](../../benchmarks/arc-easy-5/results-macos-m3-pro.json) |

## Measured throughput

Five-case resident-chat matrix (the same prompts, procedure and greedy
seed as the Qwen3.6 matrix), one process per arm; output
token-identical on 5/5 cases between plain greedy and adaptive MTP.

| Case | Tokens | Decode, MTP off | Decode, adaptive MTP | Speedup | Request wall off/on |
|---|---:|---:|---:|---:|---:|
| C `max2` function | 28 | 8.29 tok/s | 12.16 tok/s | 1.47x | 4.5 / 3.8 s |
| Hash-table prose | 531 | 8.01 | 9.56 | 1.19x | 67.1 / 56.5 s |
| Python `LRUCache` class | 1,155 | 8.10 | 11.54 | 1.42x | 144.6 / 102.3 s |
| Virtual-memory essay | 1,463 | 8.18 | 9.06 | 1.11x | 180.3 / 163.1 s |
| Notes summary, 159-token prompt | 128 | 8.27 | 11.01 | 1.33x | 19.8 / 16.3 s |
| **Aggregate, 3,305 tokens** | | **8.13** | **9.98** | **1.23x** | |

Qwen3.6-27B measured 8.09 → 9.72 tok/s (1.20x) on the same battery
with the same runtime, so the 3.8 weights run at the same base speed
(identical graph) with slightly better draft acceptance. Raw record:
[`results.json`](results.json).
