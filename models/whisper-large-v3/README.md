# Whisper large-v3 (and large-v3-turbo)

OpenAI Whisper large-v3 encoder–decoder ASR. Two checkpoints are covered as
variants of one model directory because they share the encoder architecture
and tokenizer; turbo replaces the 32-layer decoder with a 4-layer distilled
decoder.

| Fact | large-v3 | large-v3-turbo |
|---|---|---|
| Encoder | 32 layers, d=1280, 20 heads | same |
| Decoder | 32 layers | 4 layers |
| Mel bins | 128 | 128 |
| Vocab | 51,866 | 51,866 |
| Parameters | ~1.55 B | ~809 M |

## Pinned sources

GGML checkpoints from the public `ggml-org/whisper.cpp` Hugging Face
repository (converted from `openai/whisper-large-v3` and
`openai/whisper-large-v3-turbo`):

| File | Bytes | SHA-256 |
|---|---:|---|
| `ggml-large-v3.bin` | 3,095,033,483 | `64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2` |
| `ggml-large-v3-turbo.bin` | 1,624,555,275 | `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69` |

Weights are not committed; they live on the target device.

## Evaluation set

A pinned 32-file subset of LibriSpeech `test-clean` (speaker 1089), 251.1 s
of audio, 665 reference words after English text normalization. The exact
file list and references are committed under each target's `benchmarks/`
(list SHA-256 `19bcb46351a8317e7faf4d29c6a45e7b638a2ad23e31d38c73c82b51f079dbf6`).
WER is computed with `jiwer` on text normalized by the OpenAI Whisper
`EnglishTextNormalizer` (`whisper-normalizer` package), the same method the
Hugging Face Open ASR Leaderboard uses.

Quality gate for every optimization step: corpus WER must not regress versus
the same model's baseline, and per-file WER deltas are checked — corpus
averages can hide localized regressions.

## Targets

| Target | Status |
|---|---|
| [`targets/jetson-orin/`](targets/jetson-orin/README.md) | Baseline measured; optimization in progress |
