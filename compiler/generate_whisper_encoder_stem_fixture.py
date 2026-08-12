#!/usr/bin/env python3
"""Generate a miniature OpenAI Whisper audio-encoder stem fixture."""

import argparse
import math
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8s7I")
KERNEL = 3
STRIDE = 2


def gelu(value: float) -> np.float32:
    return np.float32(0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))))


def build_arrays() -> tuple[np.ndarray, ...]:
    n_mels = 5
    n_state = 8
    frames = 13
    output_frames = (frames + 1) // 2

    audio_index = np.arange(n_mels * frames, dtype=np.float64)
    audio = (0.41 * np.sin(audio_index * 0.17) + 0.09 * np.cos(audio_index * 0.071)).astype("<f4")

    weight1_index = np.arange(n_state * n_mels * KERNEL, dtype=np.float64)
    weight1 = (0.08 * np.sin(weight1_index * 0.113) - 0.025 * np.cos(weight1_index * 0.037)).astype("<f4")
    bias1 = np.linspace(-0.07, 0.06, n_state, dtype=np.float32).astype("<f4")

    weight2_index = np.arange(n_state * n_state * KERNEL, dtype=np.float64)
    weight2 = (0.055 * np.cos(weight2_index * 0.097) + 0.017 * np.sin(weight2_index * 0.043)).astype("<f4")
    bias2 = np.linspace(0.035, -0.045, n_state, dtype=np.float32).astype("<f4")

    position_index = np.arange(output_frames * n_state, dtype=np.float64)
    positions = (0.12 * np.sin(position_index * 0.19)).astype("<f4")

    audio = audio.reshape(n_mels, frames)
    weight1 = weight1.reshape(n_state, n_mels, KERNEL)
    weight2 = weight2.reshape(n_state, n_state, KERNEL)
    positions = positions.reshape(output_frames, n_state)

    hidden = np.empty((n_state, frames), dtype=np.float32)
    for output_channel in range(n_state):
        for frame in range(frames):
            total = float(bias1[output_channel])
            for input_channel in range(n_mels):
                for tap in range(KERNEL):
                    source = frame + tap - 1
                    if 0 <= source < frames:
                        total += float(weight1[output_channel, input_channel, tap]) * float(audio[input_channel, source])
            hidden[output_channel, frame] = gelu(total)

    expected = np.empty((output_frames, n_state), dtype=np.float32)
    for output_frame in range(output_frames):
        for output_channel in range(n_state):
            total = float(bias2[output_channel])
            for input_channel in range(n_state):
                for tap in range(KERNEL):
                    source = output_frame * STRIDE + tap - 1
                    if 0 <= source < frames:
                        total += float(weight2[output_channel, input_channel, tap]) * float(hidden[input_channel, source])
            expected[output_frame, output_channel] = np.float32(gelu(total) + positions[output_frame, output_channel])

    return (
        audio.reshape(-1),
        weight1.reshape(-1),
        bias1,
        weight2.reshape(-1),
        bias2,
        positions.reshape(-1),
        expected.reshape(-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = build_arrays()
    n_mels, n_state, frames = 5, 8, 13
    output_frames = (frames + 1) // 2
    header = HEADER.pack(
        b"WHSTEM01", 1, n_mels, n_state, frames, output_frames, KERNEL, STRIDE
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + b"".join(array.astype("<f4").tobytes() for array in arrays))


if __name__ == "__main__":
    main()
