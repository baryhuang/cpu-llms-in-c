# Generic target: Whisper small.en

This directory contains portable C model-axis code. It cannot contain Cortex-A53 instructions, A113X cache sizes, fixed target thread counts or device-specific storage assumptions.

## Implemented

| Stage | Boundary | Status |
|---|---|---|
| G1 | 80-bin OpenAI-compatible log-Mel | exact mixed-radix 400-point FFT implemented; frame loop can use OpenMP |
| G2 | two-convolution audio encoder stem plus positions | scalar correctness path implemented |
| G3 | complete 12-layer encoder and final LayerNorm | F32 and Q4 image execution implemented; real-weight F32 boundaries checked independently |
| G4 | cached 12-layer decoder, tied output head and byte tokenizer | implemented; arbitrary mono PCM16 16 kHz WAV transcription path emits UTF-8 text |
| G5 | packed full-graph image compiler and mmap loader | F32, encoder-Q4 and mixed encoder-Q4/decoder-Q8 images implemented |

The low-level scalar kernels allocate no memory internally. The image-level encoder and decoder states own bounded workspaces and caches. Weights stay read-only in the mapped image. Greedy decoding currently selects English transcription with no timestamps; beam search, timestamp emission and long-audio window scheduling remain open.
