#!/usr/bin/env python3

import argparse
import math
import struct
from pathlib import Path


MAGIC = b"G4LYR001"
VERSION = 1


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Generator:
    def __init__(self, seed):
        self.state = seed

    def value(self, scale):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        unit = ((self.state >> 8) & 0xFFFFFF) / float(1 << 24)
        return f32((2.0 * unit - 1.0) * scale)

    def vector(self, count, scale):
        return [self.value(scale) for _ in range(count)]

    def norm(self, count):
        return [f32(1.0 + self.value(0.08)) for _ in range(count)]


def linear(inputs, weights, rows, input_size, output_size):
    outputs = []
    for row in range(rows):
        input_offset = row * input_size
        for output_index in range(output_size):
            weight_offset = output_index * input_size
            total = 0.0
            for input_index in range(input_size):
                total += inputs[input_offset + input_index] * weights[weight_offset + input_index]
            outputs.append(f32(total))
    return outputs


def rms_norm(inputs, scale, rows, width, epsilon):
    outputs = []
    for row in range(rows):
        values = inputs[row * width : (row + 1) * width]
        mean_squared = sum(value * value for value in values) / width + epsilon
        inverse_root = math.pow(mean_squared, -0.5)
        for index, value in enumerate(values):
            result = value * inverse_root
            if scale is not None:
                result *= scale[index]
            outputs.append(f32(result))
    return outputs


def gelu_pytorch_tanh(value):
    coefficient = 0.79788456080286535588
    inner = coefficient * (value + 0.044715 * value * value * value)
    return f32(0.5 * value * (1.0 + math.tanh(inner)))


def apply_rope(states, sequence_length, heads, head_dim, theta):
    result = list(states)
    half = head_dim // 2
    for position in range(sequence_length):
        for head in range(heads):
            offset = (position * heads + head) * head_dim
            original = states[offset : offset + head_dim]
            for index in range(half):
                exponent = (2.0 * index) / head_dim
                angle = position / math.pow(theta, exponent)
                cosine = math.cos(angle)
                sine = math.sin(angle)
                first = original[index]
                second = original[index + half]
                result[offset + index] = f32(first * cosine - second * sine)
                result[offset + index + half] = f32(second * cosine + first * sine)
    return result


def attention_mqa(query, key, value, sequence_length, query_heads, kv_heads, head_dim, sliding_window):
    output = [0.0] * (sequence_length * query_heads * head_dim)
    groups = query_heads // kv_heads
    for query_position in range(sequence_length):
        first_key = max(0, query_position + 1 - sliding_window) if sliding_window else 0
        for query_head in range(query_heads):
            kv_head = query_head // groups
            query_offset = (query_position * query_heads + query_head) * head_dim
            stored_scores = []
            raw_maximum = -math.inf
            for key_position in range(first_key, query_position + 1):
                key_offset = (key_position * kv_heads + kv_head) * head_dim
                raw_score = sum(
                    query[query_offset + index] * key[key_offset + index] for index in range(head_dim)
                )
                stored_scores.append(f32(raw_score))
                raw_maximum = max(raw_maximum, raw_score)

            exponentials = [f32(math.exp(score - raw_maximum)) for score in stored_scores]
            denominator = sum(math.exp(score - raw_maximum) for score in stored_scores)
            for index in range(head_dim):
                total = 0.0
                for score_index, key_position in enumerate(range(first_key, query_position + 1)):
                    value_offset = (key_position * kv_heads + kv_head) * head_dim
                    total += exponentials[score_index] / denominator * value[value_offset + index]
                output[query_offset + index] = f32(total)
    return output


def add(left, right):
    return [f32(a + b) for a, b in zip(left, right)]


def multiply(left, right):
    return [f32(a * b) for a, b in zip(left, right)]


def build_fixture():
    sequence_length = 3
    hidden_size = 8
    query_heads = 2
    kv_heads = 1
    head_dim = 4
    intermediate_size = 12
    ple_size = 4
    sliding_window = 2
    epsilon = f32(1.0e-6)
    theta = f32(10000.0)
    layer_scalar = f32(0.875)
    query_size = query_heads * head_dim
    kv_size = kv_heads * head_dim

    generator = Generator(0x4A17C0DE)
    tensors = {}
    tensors["input"] = generator.vector(sequence_length * hidden_size, 0.7)
    tensors["per_layer_input"] = generator.vector(sequence_length * ple_size, 0.4)
    tensors["input_norm"] = generator.norm(hidden_size)
    tensors["q_proj"] = generator.vector(query_size * hidden_size, 0.25)
    tensors["k_proj"] = generator.vector(kv_size * hidden_size, 0.25)
    tensors["v_proj"] = generator.vector(kv_size * hidden_size, 0.25)
    tensors["q_norm"] = generator.norm(head_dim)
    tensors["k_norm"] = generator.norm(head_dim)
    tensors["o_proj"] = generator.vector(hidden_size * query_size, 0.2)
    tensors["post_attention_norm"] = generator.norm(hidden_size)
    tensors["pre_feedforward_norm"] = generator.norm(hidden_size)
    tensors["gate_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["up_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["down_proj"] = generator.vector(hidden_size * intermediate_size, 0.2)
    tensors["post_feedforward_norm"] = generator.norm(hidden_size)
    tensors["ple_gate"] = generator.vector(ple_size * hidden_size, 0.2)
    tensors["ple_projection"] = generator.vector(hidden_size * ple_size, 0.2)
    tensors["post_ple_norm"] = generator.norm(hidden_size)

    normalized_input = rms_norm(
        tensors["input"], tensors["input_norm"], sequence_length, hidden_size, epsilon
    )
    query = linear(normalized_input, tensors["q_proj"], sequence_length, hidden_size, query_size)
    key = linear(normalized_input, tensors["k_proj"], sequence_length, hidden_size, kv_size)
    value = linear(normalized_input, tensors["v_proj"], sequence_length, hidden_size, kv_size)
    query = rms_norm(query, tensors["q_norm"], sequence_length * query_heads, head_dim, epsilon)
    key = rms_norm(key, tensors["k_norm"], sequence_length * kv_heads, head_dim, epsilon)
    value = rms_norm(value, None, sequence_length * kv_heads, head_dim, epsilon)
    query = apply_rope(query, sequence_length, query_heads, head_dim, theta)
    key = apply_rope(key, sequence_length, kv_heads, head_dim, theta)
    attention_heads = attention_mqa(
        query, key, value, sequence_length, query_heads, kv_heads, head_dim, sliding_window
    )
    attention = linear(attention_heads, tensors["o_proj"], sequence_length, query_size, hidden_size)
    attention_residual = rms_norm(
        attention, tensors["post_attention_norm"], sequence_length, hidden_size, epsilon
    )
    after_attention = add(tensors["input"], attention_residual)

    normalized_mlp = rms_norm(
        after_attention, tensors["pre_feedforward_norm"], sequence_length, hidden_size, epsilon
    )
    gate = linear(normalized_mlp, tensors["gate_proj"], sequence_length, hidden_size, intermediate_size)
    up = linear(normalized_mlp, tensors["up_proj"], sequence_length, hidden_size, intermediate_size)
    activated = multiply([gelu_pytorch_tanh(value) for value in gate], up)
    mlp = linear(activated, tensors["down_proj"], sequence_length, intermediate_size, hidden_size)
    mlp_residual = rms_norm(
        mlp, tensors["post_feedforward_norm"], sequence_length, hidden_size, epsilon
    )
    after_mlp = add(after_attention, mlp_residual)

    ple_gate = linear(after_mlp, tensors["ple_gate"], sequence_length, hidden_size, ple_size)
    ple_activated = multiply([gelu_pytorch_tanh(value) for value in ple_gate], tensors["per_layer_input"])
    ple = linear(ple_activated, tensors["ple_projection"], sequence_length, ple_size, hidden_size)
    ple_residual = rms_norm(ple, tensors["post_ple_norm"], sequence_length, hidden_size, epsilon)
    output = [f32(value * layer_scalar) for value in add(after_mlp, ple_residual)]

    tensors["expected_normalized_input"] = normalized_input
    tensors["expected_query"] = query
    tensors["expected_key"] = key
    tensors["expected_value"] = value
    tensors["expected_attention"] = attention
    tensors["expected_after_attention"] = after_attention
    tensors["expected_mlp"] = mlp
    tensors["expected_after_mlp"] = after_mlp
    tensors["expected_ple"] = ple
    tensors["expected_output"] = output

    order = [
        "input",
        "per_layer_input",
        "input_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "o_proj",
        "post_attention_norm",
        "pre_feedforward_norm",
        "gate_proj",
        "up_proj",
        "down_proj",
        "post_feedforward_norm",
        "ple_gate",
        "ple_projection",
        "post_ple_norm",
        "expected_normalized_input",
        "expected_query",
        "expected_key",
        "expected_value",
        "expected_attention",
        "expected_after_attention",
        "expected_mlp",
        "expected_after_mlp",
        "expected_ple",
        "expected_output",
    ]

    header = struct.pack(
        "<8s10I3f",
        MAGIC,
        VERSION,
        sequence_length,
        hidden_size,
        query_heads,
        kv_heads,
        head_dim,
        intermediate_size,
        ple_size,
        sliding_window,
        len(order),
        epsilon,
        theta,
        layer_scalar,
    )
    payload = b"".join(struct.pack(f"<{len(tensors[name])}f", *tensors[name]) for name in order)
    return header + payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"wrote {len(data)} bytes to {args.output}")


if __name__ == "__main__":
    main()
