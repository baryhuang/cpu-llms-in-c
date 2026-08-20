# Whisper large-v3-turbo

Explicit variant of [Whisper large-v3](../whisper-large-v3/README.md):
same 32-layer encoder, same tokenizer and mel frontend, but the 32-layer
decoder is replaced by a 4-layer distilled decoder. It is tracked as its
own model directory because its performance profile is qualitatively
different — the encoder dominates end-to-end time instead of the decoder —
so target-level optimization decisions diverge from large-v3's.

| Fact | Value |
|---|---|
| Encoder | 32 layers, d=1280, 20 heads (identical to large-v3) |
| Decoder | **4 layers** (large-v3: 32) |
| Mel bins / vocab | 128 / 51,866 |
| Parameters | ~809 M |

## Pinned source

GGML checkpoint from the public `ggml-org/whisper.cpp` Hugging Face
repository, converted from `openai/whisper-large-v3-turbo`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `ggml-large-v3-turbo.bin` | 1,624,555,275 | `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69` |

Weights are not committed; they live on the target device.

## Evaluation set

Same pinned 32-file LibriSpeech `test-clean` subset as large-v3 (251.1 s,
665 normalized reference words, list SHA-256 `19bcb463…`, committed in each
target's `benchmarks/`), WER via `jiwer` + OpenAI `EnglishTextNormalizer`.
Quality gate: corpus WER non-regression plus per-file WER review.

## Targets

| Target | Status |
|---|---|
| [`targets/jetson-orin/`](targets/jetson-orin/README.md) | Certified 7.79 RTFx fp16 (+9.8% over upstream, transcripts byte-identical); 5 quant modes measured; i8 optimization in progress |
