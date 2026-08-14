# Qwen3.6-27B

Status: the pinned text-only graph runs end to end on the Apple M3 Pro target. The artifact accepts a free-text user prompt, applies the pinned chat template and tokenizer, executes all 64 layers, samples, and decodes text.

This model record contains optimizations that do not depend on a particular CPU or GPU. Target-specific layouts, kernels and measurements belong below `targets/`.

## Pinned graph

| Property | Pinned value |
|---|---:|
| Upstream repository | `Qwen/Qwen3.6-27B` |
| Upstream architecture class | `Qwen3_5ForConditionalGeneration` |
| Hidden size | 5,120 |
| Layers | 64 |
| Layer pattern | three GatedDeltaNet layers, then one full-attention layer, repeated 16 times |
| GatedDeltaNet / attention layers | 48 / 16 |
| MLP intermediate size | 17,408 |
| Attention heads / KV heads / head dimension | 24 / 4 / 256 |
| Vocabulary | 248,320 |
| Maximum declared context | 262,144 tokens |
| MTP depth | one hidden layer, shared embeddings |

The upstream checkpoint is multimodal. This repository's first artifact is text-only: the vision tower and image-input path are excluded from the packed image. This is an exact specialization of the declared input contract, not model pruning. MTP remains optional and must be verified separately from one-token decode.

## Model-axis optimization stack

| Order | Transformation | Status | Correctness gate |
|---:|---|---|---|
| 1 | Remove the unreachable vision graph | implemented; saves 921,460,192 source tensor bytes | text-only tensor inventory and end-to-end output parity |
| 2 | Pack the repeated `3 x GatedDeltaNet + 1 x attention` schedule as a static 16-block program | implemented as 64 fixed layer images and a fixed dispatch loop | complete Delta and attention layer oracles |
| 3 | Import pinned affine Q4 group-64 weights into target layouts | implemented | exact tensor name, shape, dtype, source hash and image checks |
| 4 | Fuse normalization, projection epilogues, gates and residual writes | implemented for one-token execution | independent C boundary oracles |
| 5 | Compile the pinned tokenizer and chat template | implemented for arbitrary single-user UTF-8 prompts | official tokenizer token-ID parity |
| 6 | Batch prompt prefill | open; current graph processes one prompt token at a time | exact prompt/logit comparison |
| 7 | Quantize the attention KV cache | open; current cache is FP32 | long-context replay comparison |
| 8 | Precompute fixed-prefix state when a deployment profile declares one | open, optional | snapshot replay equals ordinary prefill |
| 9 | Optional MTP verification | open, optional | acceptance rate and output-quality measurement |

Free text remains supported. Fixed-prefix compilation is an optional workload specialization; it is not a fixed question, a fixed answer set, or a binary classifier.

## Memory model

The dense model visits nearly every packed weight for every generated token. Q4 group metadata means the executable image is larger than a bare `27B / 2` estimate. The pinned MLX input uses 32 quantized bytes plus one BF16 scale and one BF16 bias per 64 weights: 4.5 bits per weight before tensor alignment and higher-precision exceptions. The Apple image converts those metadata values to FP16 only after a numerical gate.

| State | Planning value | Notes |
|---|---:|---|
| Complete quantized checkpoint tensor bytes | 16,054,262,240 bytes | includes 921,460,192-byte vision tower |
| Text-only tensor bytes | 15,132,802,048 bytes | inventory value before target repacking; excludes vision tower |
| GatedDeltaNet recurrent state | 150,994,944 bytes / 144 MiB | `48 layers x 48 heads x 128 x 128 x FP32` |
| Attention KV cache, current FP32 runtime | 128 KiB/token | 16 attention layers only; 16,777,216 bytes at capacity 128 |
| Attention KV cache, FP16 planning value | 64 KiB/token | not implemented |
| Attention KV cache, Q8 planning value | 32 KiB/token | requires a separate quality gate |
| 16K Q8 KV plus recurrent state | roughly 663 MB | planning value; excludes weights and scratch |

The model fits in 36 GB unified memory at Q4. Decode speed remains governed primarily by bytes visited per token, while prompt prefill has enough arithmetic intensity to benefit from both GPU matrix work and Apple CPU matrix acceleration.

## Current target

| CPU / GPU target | Status | Record |
|---|---|---|
| Apple M3 Pro, 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | free-text end-to-end runtime measured; batched prefill open | [`targets/apple-m3-pro/`](targets/apple-m3-pro/) |
