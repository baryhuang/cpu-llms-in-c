# Generic target: Whisper small.en

This directory contains portable C model-axis code. It cannot contain Cortex-A53 instructions, A113X cache sizes, fixed target thread counts or device-specific storage assumptions.

## Implemented

| Stage | Boundary | Status |
|---|---|---|
| G1 | 80-bin OpenAI-compatible log-Mel | scalar correctness path implemented |
| G2 | two-convolution audio encoder stem plus positions | scalar correctness path implemented |
| G3 | complete 12-layer encoder and final LayerNorm | F32 and Q4 image execution implemented; real-weight F32 boundaries checked independently |
| G4 | cached 12-layer decoder and tokenizer | not implemented |
| G5 | packed encoder image compiler and mmap loader | F32 exact image and Q4 Transformer-matrix image implemented |

The low-level scalar kernels allocate no memory internally. The image-level encoder runner owns one bounded workspace per call. Weights stay read-only in the mapped image.
