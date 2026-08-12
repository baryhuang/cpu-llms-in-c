# Whisper small.en

Status: selected A113X real-time research path. The scalar 80-bin log-Mel front end and complete 12-layer encoder run from compiler-generated F32 and Q4 images. Real-weight F32 boundaries pass; the Q4 encoder and Cortex-A53 kernels are measured on A113X. The decoder and tokenizer are not implemented.

## Pinned architecture

| Item | Value |
|---|---|
| Model | OpenAI Whisper `small.en` |
| Parameters | approximately 244M |
| Language contract | English-only transcription |
| Input | 16 kHz mono audio; arbitrary content |
| Front end | 80-bin log-Mel, 400-sample FFT, 160-sample hop |
| Encoder | two Conv1D layers; 12 Transformer blocks, width 768, 12 heads, FFN 3072 |
| Decoder | 12 cached Transformer blocks, width 768, 12 heads, FFN 3072 |
| Vocabulary | 51,864 English-tokenizer entries including control and timestamp tokens |
| Context | 1500 audio positions after stride-2 convolution; up to 448 text positions |

Exact source, checkpoint and reference-runtime pins are in [`pins.json`](pins.json). The deployment and evaluation contract is in [`profile.json`](profile.json).

## Acceptance gates

| Metric | Gate |
|---|---:|
| Sustained continuous-speech RTF | `<= 1.0` |
| Relative WER increase vs unmodified pinned `small.en` | `<= 10%` |
| Peak RSS | `< 1 GiB` |
| Swap | `0` |
| Required output | arbitrary English transcript and ordered timestamps |

The target is a derived model optimized jointly with the C runtime. It is not required to retain all 12+12 layers unchanged. The decision history, feature boundary, model comparison and stage plan are recorded in [`DECISIONS.md`](DECISIONS.md).

## Current C boundaries

| Boundary | Implementation | Verification |
|---|---|---|
| Log-Mel | centered scalar STFT, pinned `mel_80` filterbank, log clamp and normalization | committed synthetic fixture generated independently in Python |
| Encoder stem | Conv1D 80→768, exact GELU, stride-2 Conv1D 768→768, exact GELU, position addition | miniature independent fixture checks the graph and layout |
| Encoder | stem, position addition, 12 pre-LayerNorm attention/FFN blocks and final LayerNorm | real pinned weights checked at stem, layer 0 and final output against independent NumPy |
| Image compiler | selective safetensors import; exact F32 image or group-128 Q4 Transformer matrices | image and source SHA-256 pins; weights excluded from Git |

The A113X Q4 encoder image is 56,763,776 bytes. Its final measured CPU stage uses 125,056 KiB peak RSS and zero process swap, but takes 556.218 seconds for a 30-second encoder window: encoder-only RTF 18.541. The full-attention topology fails the compute gate before decoder work. Exact increments and per-layer durations are in the [A113X target record](targets/a113x/README.md).

## Build and test

```sh
make test
```

Checkpoints, packed images and generated target binaries are not committed.
