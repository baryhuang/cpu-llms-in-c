# Generic target: Whisper small.en

This directory contains portable C model-axis code. It cannot contain Cortex-A53 instructions, A113X cache sizes, fixed target thread counts or device-specific storage assumptions.

## Implemented

| Stage | Boundary | Status |
|---|---|---|
| G1 | 80-bin OpenAI-compatible log-Mel | scalar correctness path implemented |
| G2 | two-convolution audio encoder stem plus positions | scalar correctness path implemented |
| G3 | one complete encoder Transformer block | scalar correctness path implemented; 12-layer graph and packed real weights not implemented |
| G4 | cached 12-layer decoder and tokenizer | not implemented |
| G5 | packed model image loader | not implemented |

The current scalar loops allocate no memory internally. Callers provide output and workspace buffers.
