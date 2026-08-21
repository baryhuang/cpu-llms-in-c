# MOSS-TTS-Nano-100M

Status: native CPU voice cloning is implemented through the pinned `audio.cpp`
runtime. The A113X target adds an exact incremental KV cache and a reproducible
C/C++ build; inference does not use Python, PyTorch, ONNX Runtime, or a remote
service.

## Source

| Field | Value |
|---|---|
| Model | `OpenMOSS-Team/MOSS-TTS-Nano-100M` |
| Runtime | `0xShug0/audio.cpp` release 0.6.1 |
| Runtime commit | `26dcb5c4cf5aa016ae6285096a7b45f2671e5d17` |
| Quantization | Q8_0 GGUF |
| Weight revision | `7e9a59c8aee65253851b91b7b9b8e206544ef1a6` |
| Weight SHA-256 | `3b9c138ce4093514b6493b77abaf3e62f4d2ac58412616d2f7ee89cca28be388` |

Machine-readable pins are in [`source.json`](source.json).

## Target matrix

| Target | Execution path | Evidence |
|---|---|---|
| Amlogic A113X, 4x Cortex-A53, 2 GB | native C++17/GGML, Q8_0 weights, exact incremental global-transformer KV cache | [target record](targets/a113x/README.md) · [raw results](targets/a113x/results.json) |

The voice reference is runtime input. It is deliberately excluded from source
control and release assets.
