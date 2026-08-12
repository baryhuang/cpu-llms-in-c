# Whisper small.en

Status: selected A113X real-time research path. The scalar C 80-bin log-Mel front end, two-convolution encoder stem and one portable encoder Transformer block are implemented as correctness boundaries. The 12-layer graph, decoder, tokenizer, packed weights and `small.en` target benchmark are not implemented.

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
| Encoder block | pre-LayerNorm, multi-head self-attention, projection/residual, second LayerNorm, exact-GELU FFN/residual | miniature independent fixture checks post-attention and final output |

The scalar DFT and convolution loops are correctness code, not target kernels. A113X FFT, Conv1D, quantized matrix and threading work lands only after the respective output boundary exists.

## Build and test

```sh
make test
```

Checkpoints, packed images and generated target binaries are not committed.
