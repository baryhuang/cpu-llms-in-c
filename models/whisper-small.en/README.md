# Whisper small.en

Status: selected A113X real-time research path. A from-scratch C implementation now covers the exact mixed-radix FFT log-Mel front end, complete 12-layer encoder, cached 12-layer decoder, byte-token table and greedy English transcription. The first public A113X smoke case emits the correct normalized words, but the unchanged graph takes 589.056 seconds for 11 seconds of audio. The real-time and evaluation-suite quality gates remain open.

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
| Decoder | cached self-attention, cross-attention cache, 12 decoder blocks and tied output head | three real-weight causal steps checked against independent NumPy; one- and four-thread tokens agree |
| Token output | compiled byte-token table, suppression policy and greedy no-timestamps decode | public JFK WAV emits readable English text |
| Image compiler | selective safetensors import; F32, encoder Q4 and decoder/tied-head Q8 packing | image and source SHA-256 pins; weights excluded from Git |

The full mixed image is 214,878,912 bytes. On A113X, an 11-second public WAV is zero-padded to the fixed 30-second model window and completes in 589.055851 seconds: RTF 53.5505, 386% CPU, 327,820 KiB peak RSS and zero swap. The encoder consumes 574.188459 seconds. The result establishes a working C path and identifies the unchanged encoder graph as the dominant limit; it does not establish the `<10%` relative-WER gate. Exact increments, input, transcript and raw duration/resource output are in the [A113X target record](targets/a113x/README.md).

## Build and test

```sh
make test
```

Checkpoints, packed images and generated target binaries are not committed.
