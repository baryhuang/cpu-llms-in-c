# Generic target: Whisper large-v3

Status: M1 front-end implementation started. The scalar 128-bin log-Mel boundary is committed and verified. Encoder and decoder execution remain planned; no transcription performance result is present.

This directory will contain the portable C runtime after model-axis stages M1-M3 are implemented. It may use C11 and a small internal thread interface, but it may not contain A113X-specific NEON, cache sizes, thread counts or storage assumptions.

## Required implementation boundary

| Component | Generic runtime responsibility |
|---|---|
| Audio | signed-16-bit PCM and float32 input; 16 kHz mono normalization |
| Front end | STFT, 128-bin log-Mel, padding/windowing and original timeline mapping |
| Encoder | two convolution layers and 32 residual attention blocks |
| Decoder | multilingual tokenizer, prompt/task tokens, timestamp grammar, self-KV and cross-KV cache |
| Output | ordered `{start_ms, end_ms, text}` segments |
| Memory | compiler-sized arena; immutable packed image; no per-token heap allocation |
| Instrumentation | front-end, encoder, decoder, wall, CPU, RSS, swap and page-fault fields |

## First committed fixture

The first implementation change must add a small deterministic fixture and compare these boundaries against the pinned OpenAI reference:

1. log-Mel values;
2. convolution outputs;
3. encoder block 0 and encoder block 31 outputs;
4. cross-attention K/V for decoder block 0;
5. decoder block 0 and decoder block 31 outputs;
6. selected logits, decoded token IDs and segment boundaries.

The fixture may use reduced random dimensions for primitive coverage. At least one real-weight, public-audio run is required before the model is marked implemented.

## Current boundary

[`whisper_frontend.c`](whisper_frontend.c) implements the pinned centered STFT, 128-bin Mel projection, log clamp and normalization. It has no heap allocation and accepts caller-owned workspace. It deliberately uses a direct DFT so the result is readable and independent of a third-party FFT implementation.

| Target | Values | Maximum absolute delta | Fixture hash | Result |
|---|---:|---:|---|---|
| local development machine | 768 | `1.74045563e-5` | `b2d015b66ecffc30e52a05b756daab5f94102c75b701fc84e015ab9063eca3b1` | pass |
| A113X / GCC 12.2 / Cortex-A53 | 768 | `1.74045563e-5` | same | pass |

This is verification, not a timed benchmark. The next front-end increment replaces the direct DFT with a compiler-selected fixed-size FFT and must reproduce this boundary before its duration is reported.
