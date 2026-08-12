#!/usr/bin/env python3
"""Generate a miniature OpenAI Whisper encoder-block fixture."""

import argparse
import math
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8s6I")


def gelu(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for index, value in enumerate(values.reshape(-1)):
        result.reshape(-1)[index] = np.float32(
            0.5 * float(value) * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
        )
    return result


def make_values(count: int, scale: float, phase: float) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    return (scale * np.sin(index * 0.137 + phase) + scale * 0.31 * np.cos(index * 0.071 - phase)).astype("<f4")


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    variance = ((x.astype(np.float64) - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (x.astype(np.float64) - mean) / np.sqrt(variance + 1.0e-5)
    return (normalized * weight.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    result = x.astype(np.float64) @ weight.astype(np.float64).T
    if bias is not None:
        result += bias.astype(np.float64)
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames, n_state, n_heads, n_mlp = 5, 8, 2, 13
    input_values = make_values(frames * n_state, 0.39, 0.2).reshape(frames, n_state)
    attention_norm_weight = (1.0 + make_values(n_state, 0.08, 0.3)).astype("<f4")
    attention_norm_bias = make_values(n_state, 0.04, 0.7)
    query_weight = make_values(n_state * n_state, 0.10, 0.1).reshape(n_state, n_state)
    query_bias = make_values(n_state, 0.035, 0.9)
    key_weight = make_values(n_state * n_state, 0.095, 0.4).reshape(n_state, n_state)
    value_weight = make_values(n_state * n_state, 0.085, 0.8).reshape(n_state, n_state)
    value_bias = make_values(n_state, 0.03, 1.1)
    output_weight = make_values(n_state * n_state, 0.09, 1.3).reshape(n_state, n_state)
    output_bias = make_values(n_state, 0.025, 1.5)
    mlp_norm_weight = (1.0 + make_values(n_state, 0.07, 1.7)).astype("<f4")
    mlp_norm_bias = make_values(n_state, 0.035, 1.9)
    mlp_input_weight = make_values(n_mlp * n_state, 0.08, 2.1).reshape(n_mlp, n_state)
    mlp_input_bias = make_values(n_mlp, 0.03, 2.3)
    mlp_output_weight = make_values(n_state * n_mlp, 0.075, 2.5).reshape(n_state, n_mlp)
    mlp_output_bias = make_values(n_state, 0.025, 2.7)

    normalized = layer_norm(input_values, attention_norm_weight, attention_norm_bias)
    query = linear(normalized, query_weight, query_bias)
    key = linear(normalized, key_weight, None)
    value = linear(normalized, value_weight, value_bias)
    head_width = n_state // n_heads
    context = np.empty_like(query)
    for head in range(n_heads):
        channels = slice(head * head_width, (head + 1) * head_width)
        logits = query[:, channels].astype(np.float64) @ key[:, channels].astype(np.float64).T / math.sqrt(head_width)
        logits -= logits.max(axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=-1, keepdims=True)
        context[:, channels] = (probability @ value[:, channels].astype(np.float64)).astype(np.float32)
    after_attention = (input_values + linear(context, output_weight, output_bias)).astype(np.float32)
    normalized_mlp = layer_norm(after_attention, mlp_norm_weight, mlp_norm_bias)
    hidden = gelu(linear(normalized_mlp, mlp_input_weight, mlp_input_bias))
    expected = (after_attention + linear(hidden, mlp_output_weight, mlp_output_bias)).astype(np.float32)

    arrays = [
        input_values, attention_norm_weight, attention_norm_bias,
        query_weight, query_bias, key_weight, value_weight, value_bias,
        output_weight, output_bias, mlp_norm_weight, mlp_norm_bias,
        mlp_input_weight, mlp_input_bias, mlp_output_weight, mlp_output_bias,
        after_attention, expected,
    ]
    header = HEADER.pack(b"WHENCB01", 1, frames, n_state, n_heads, n_mlp, len(arrays))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + b"".join(array.astype("<f4").tobytes() for array in arrays))


if __name__ == "__main__":
    main()
