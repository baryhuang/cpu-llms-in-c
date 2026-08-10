#!/usr/bin/env python3
"""Generate the committed Qwen3.5 layer fixture.

One Gated-DeltaNet layer followed by one full-attention layer, computed
with double precision and explicit float32 casts at every stored value,
so the C implementation can match each declared boundary exactly. The
mathematics follows compiler/qwen35_reference.py, which is differential-
tested against the pinned transformers oracle.
"""

import argparse
import math
import struct
from pathlib import Path


MAGIC = b"QW35LYR1"
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


def linear(inputs, weights, rows, input_size, output_size):
    outputs = []
    for row in range(rows):
        base = row * input_size
        for out_index in range(output_size):
            weight_base = out_index * input_size
            total = 0.0
            for index in range(input_size):
                total += inputs[base + index] * weights[weight_base + index]
            outputs.append(f32(total))
    return outputs


def rms_norm_one_plus(inputs, weight, rows, width, epsilon):
    outputs = []
    for row in range(rows):
        values = inputs[row * width : (row + 1) * width]
        mean_squared = sum(value * value for value in values) / width + epsilon
        inverse_root = math.pow(mean_squared, -0.5)
        for index, value in enumerate(values):
            outputs.append(f32(value * inverse_root * (1.0 + weight[index])))
    return outputs


def silu(value):
    return value / (1.0 + math.exp(-value))


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def softplus(value):
    return math.log1p(math.exp(value))


def causal_conv_silu(inputs, weight, rows, channels, kernel):
    outputs = []
    for row in range(rows):
        for channel in range(channels):
            total = 0.0
            for tap in range(kernel):
                source = row - (kernel - 1) + tap
                if source >= 0:
                    total += inputs[source * channels + channel] * weight[channel * kernel + tap]
            outputs.append(f32(silu(total)))
    return outputs


def l2norm_scale(states, rows, heads, dim, scale, epsilon=1e-6):
    outputs = []
    for row in range(rows):
        for head in range(heads):
            base = (row * heads + head) * dim
            values = states[base : base + dim]
            inverse = math.pow(sum(v * v for v in values) + epsilon, -0.5)
            for value in values:
                outputs.append(f32(value * inverse * scale))
    return outputs


def apply_partial_rope(states, rows, heads, head_dim, rotary_dim, theta):
    result = list(states)
    half = rotary_dim // 2
    for position in range(rows):
        for head in range(heads):
            base = (position * heads + head) * head_dim
            for index in range(half):
                inv_freq = 1.0 / math.pow(theta, (2.0 * index) / rotary_dim)
                angle = position * inv_freq
                cosine = math.cos(angle)
                sine = math.sin(angle)
                first = states[base + index]
                second = states[base + index + half]
                result[base + index] = f32(first * cosine - second * sine)
                result[base + index + half] = f32(second * cosine + first * sine)
    return result


def add(left, right):
    return [f32(a + b) for a, b in zip(left, right)]


def build_fixture():
    sequence_length = 4
    hidden_size = 8
    linear_k_heads = 2
    linear_v_heads = 2
    linear_k_dim = 4
    linear_v_dim = 4
    conv_kernel = 4
    query_heads = 2
    kv_heads = 1
    head_dim = 8
    rotary_dim = 2
    intermediate_size = 12
    epsilon = f32(1.0e-6)
    theta = f32(1.0e7)

    key_dim = linear_k_heads * linear_k_dim
    value_dim = linear_v_heads * linear_v_dim
    conv_dim = 2 * key_dim + value_dim
    query_size = query_heads * head_dim
    kv_size = kv_heads * head_dim

    generator = Generator(0x35C0FFEE)
    tensors = {}
    tensors["input"] = generator.vector(sequence_length * hidden_size, 0.7)

    tensors["l0_input_norm"] = generator.vector(hidden_size, 0.08)
    tensors["l0_in_proj_qkv"] = generator.vector(conv_dim * hidden_size, 0.25)
    tensors["l0_in_proj_z"] = generator.vector(value_dim * hidden_size, 0.25)
    tensors["l0_in_proj_b"] = generator.vector(linear_v_heads * hidden_size, 0.25)
    tensors["l0_in_proj_a"] = generator.vector(linear_v_heads * hidden_size, 0.25)
    tensors["l0_conv"] = generator.vector(conv_dim * conv_kernel, 0.3)
    tensors["l0_a_log"] = [f32(math.log(0.5 + 3.0 * ((i + 1) / linear_v_heads))) for i in range(linear_v_heads)]
    tensors["l0_dt_bias"] = [f32(1.0 + generator.value(0.2)) for _ in range(linear_v_heads)]
    tensors["l0_gated_norm"] = [f32(1.0 + generator.value(0.08)) for _ in range(linear_v_dim)]
    tensors["l0_out_proj"] = generator.vector(hidden_size * value_dim, 0.2)
    tensors["l0_post_norm"] = generator.vector(hidden_size, 0.08)
    tensors["l0_gate_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["l0_up_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["l0_down_proj"] = generator.vector(hidden_size * intermediate_size, 0.2)

    tensors["l1_input_norm"] = generator.vector(hidden_size, 0.08)
    tensors["l1_q_proj"] = generator.vector(2 * query_size * hidden_size, 0.25)
    tensors["l1_k_proj"] = generator.vector(kv_size * hidden_size, 0.25)
    tensors["l1_v_proj"] = generator.vector(kv_size * hidden_size, 0.25)
    tensors["l1_q_norm"] = generator.vector(head_dim, 0.08)
    tensors["l1_k_norm"] = generator.vector(head_dim, 0.08)
    tensors["l1_o_proj"] = generator.vector(hidden_size * query_size, 0.2)
    tensors["l1_post_norm"] = generator.vector(hidden_size, 0.08)
    tensors["l1_gate_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["l1_up_proj"] = generator.vector(intermediate_size * hidden_size, 0.2)
    tensors["l1_down_proj"] = generator.vector(hidden_size * intermediate_size, 0.2)

    # ---- Layer 0: Gated DeltaNet ----
    l0_normed = rms_norm_one_plus(
        tensors["input"], tensors["l0_input_norm"], sequence_length, hidden_size, epsilon
    )
    mixed_qkv = linear(l0_normed, tensors["l0_in_proj_qkv"], sequence_length, hidden_size, conv_dim)
    z = linear(l0_normed, tensors["l0_in_proj_z"], sequence_length, hidden_size, value_dim)
    b = linear(l0_normed, tensors["l0_in_proj_b"], sequence_length, hidden_size, linear_v_heads)
    a = linear(l0_normed, tensors["l0_in_proj_a"], sequence_length, hidden_size, linear_v_heads)

    post_conv = causal_conv_silu(mixed_qkv, tensors["l0_conv"], sequence_length, conv_dim, conv_kernel)

    query_scale = math.pow(float(linear_k_dim), -0.5)
    queries = []
    keys = []
    values = []
    for row in range(sequence_length):
        base = row * conv_dim
        queries.extend(post_conv[base : base + key_dim])
        keys.extend(post_conv[base + key_dim : base + 2 * key_dim])
        values.extend(post_conv[base + 2 * key_dim : base + 2 * key_dim + value_dim])
    queries = l2norm_scale(queries, sequence_length, linear_k_heads, linear_k_dim, query_scale)
    keys = l2norm_scale(keys, sequence_length, linear_k_heads, linear_k_dim, 1.0)

    gate_values = []
    for row in range(sequence_length):
        for head in range(linear_v_heads):
            a_value = a[row * linear_v_heads + head]
            gate_values.append(
                f32(-math.exp(tensors["l0_a_log"][head]) * softplus(a_value + tensors["l0_dt_bias"][head]))
            )

    state = [
        [[0.0 for _ in range(linear_v_dim)] for _ in range(linear_k_dim)]
        for _ in range(linear_v_heads)
    ]
    core_out = []
    for row in range(sequence_length):
        for head in range(linear_v_heads):
            decay = math.exp(gate_values[row * linear_v_heads + head])
            beta = sigmoid(b[row * linear_v_heads + head])
            k_vec = keys[(row * linear_k_heads + head) * linear_k_dim :][:linear_k_dim]
            q_vec = queries[(row * linear_k_heads + head) * linear_k_dim :][:linear_k_dim]
            v_vec = values[(row * linear_v_heads + head) * linear_v_dim :][:linear_v_dim]
            for ki in range(linear_k_dim):
                for vi in range(linear_v_dim):
                    state[head][ki][vi] *= decay
            for vi in range(linear_v_dim):
                kv_mem = sum(state[head][ki][vi] * k_vec[ki] for ki in range(linear_k_dim))
                delta = (v_vec[vi] - kv_mem) * beta
                for ki in range(linear_k_dim):
                    state[head][ki][vi] += k_vec[ki] * delta
            for vi in range(linear_v_dim):
                core_out.append(
                    f32(sum(state[head][ki][vi] * q_vec[ki] for ki in range(linear_k_dim)))
                )

    state_flat = [
        f32(state[head][ki][vi])
        for head in range(linear_v_heads)
        for ki in range(linear_k_dim)
        for vi in range(linear_v_dim)
    ]

    gated = []
    for row in range(sequence_length):
        for head in range(linear_v_heads):
            base = (row * linear_v_heads + head) * linear_v_dim
            head_values = core_out[base : base + linear_v_dim]
            mean_squared = sum(v * v for v in head_values) / linear_v_dim + epsilon
            inverse_root = math.pow(mean_squared, -0.5)
            for vi in range(linear_v_dim):
                z_value = z[base + vi]
                gated.append(
                    f32(head_values[vi] * inverse_root * tensors["l0_gated_norm"][vi] * silu(z_value))
                )

    l0_mixer = linear(gated, tensors["l0_out_proj"], sequence_length, value_dim, hidden_size)
    l0_after_mixer = add(tensors["input"], l0_mixer)

    l0_mlp_normed = rms_norm_one_plus(
        l0_after_mixer, tensors["l0_post_norm"], sequence_length, hidden_size, epsilon
    )
    l0_gate = linear(l0_mlp_normed, tensors["l0_gate_proj"], sequence_length, hidden_size, intermediate_size)
    l0_up = linear(l0_mlp_normed, tensors["l0_up_proj"], sequence_length, hidden_size, intermediate_size)
    l0_activated = [f32(silu(g) * u) for g, u in zip(l0_gate, l0_up)]
    l0_mlp = linear(l0_activated, tensors["l0_down_proj"], sequence_length, intermediate_size, hidden_size)
    l0_after_mlp = add(l0_after_mixer, l0_mlp)

    # ---- Layer 1: full attention with output gate ----
    l1_normed = rms_norm_one_plus(
        l0_after_mlp, tensors["l1_input_norm"], sequence_length, hidden_size, epsilon
    )
    q_and_gate = linear(l1_normed, tensors["l1_q_proj"], sequence_length, hidden_size, 2 * query_size)
    key = linear(l1_normed, tensors["l1_k_proj"], sequence_length, hidden_size, kv_size)
    value = linear(l1_normed, tensors["l1_v_proj"], sequence_length, hidden_size, kv_size)

    query = []
    attention_gate = []
    for row in range(sequence_length):
        base = row * 2 * query_size
        for head in range(query_heads):
            head_base = base + head * 2 * head_dim
            query.extend(q_and_gate[head_base : head_base + head_dim])
            attention_gate.extend(q_and_gate[head_base + head_dim : head_base + 2 * head_dim])

    query = rms_norm_one_plus(query, tensors["l1_q_norm"], sequence_length * query_heads, head_dim, epsilon)
    key = rms_norm_one_plus(key, tensors["l1_k_norm"], sequence_length * kv_heads, head_dim, epsilon)
    query = apply_partial_rope(query, sequence_length, query_heads, head_dim, rotary_dim, theta)
    key = apply_partial_rope(key, sequence_length, kv_heads, head_dim, rotary_dim, theta)

    groups = query_heads // kv_heads
    scale = math.pow(float(head_dim), -0.5)
    attention_heads = [0.0] * (sequence_length * query_size)
    for position in range(sequence_length):
        for head in range(query_heads):
            kv_head = head // groups
            query_base = (position * query_heads + head) * head_dim
            scores = []
            maximum = -math.inf
            for key_position in range(position + 1):
                key_base = (key_position * kv_heads + kv_head) * head_dim
                raw = sum(
                    query[query_base + index] * key[key_base + index] for index in range(head_dim)
                ) * scale
                scores.append(raw)
                maximum = max(maximum, raw)
            denominator = sum(math.exp(score - maximum) for score in scores)
            for index in range(head_dim):
                total = 0.0
                for key_position in range(position + 1):
                    value_base = (key_position * kv_heads + kv_head) * head_dim
                    weight = math.exp(scores[key_position] - maximum) / denominator
                    total += weight * value[value_base + index]
                attention_heads[query_base + index] = f32(total)

    gated_attention = [
        f32(attention_heads[index] * sigmoid(attention_gate[index]))
        for index in range(sequence_length * query_size)
    ]
    l1_mixer = linear(gated_attention, tensors["l1_o_proj"], sequence_length, query_size, hidden_size)
    l1_after_mixer = add(l0_after_mlp, l1_mixer)

    l1_mlp_normed = rms_norm_one_plus(
        l1_after_mixer, tensors["l1_post_norm"], sequence_length, hidden_size, epsilon
    )
    l1_gate = linear(l1_mlp_normed, tensors["l1_gate_proj"], sequence_length, hidden_size, intermediate_size)
    l1_up = linear(l1_mlp_normed, tensors["l1_up_proj"], sequence_length, hidden_size, intermediate_size)
    l1_activated = [f32(silu(g) * u) for g, u in zip(l1_gate, l1_up)]
    l1_mlp = linear(l1_activated, tensors["l1_down_proj"], sequence_length, intermediate_size, hidden_size)
    output = add(l1_after_mixer, l1_mlp)

    tensors["expected_l0_normed"] = l0_normed
    tensors["expected_l0_post_conv"] = post_conv
    tensors["expected_l0_gate"] = gate_values
    tensors["expected_l0_core_out"] = core_out
    tensors["expected_l0_state"] = state_flat
    tensors["expected_l0_gated"] = gated
    tensors["expected_l0_mixer"] = l0_mixer
    tensors["expected_l0_after_mixer"] = l0_after_mixer
    tensors["expected_l0_after_mlp"] = l0_after_mlp
    tensors["expected_l1_normed"] = l1_normed
    tensors["expected_l1_query"] = query
    tensors["expected_l1_key"] = key
    tensors["expected_l1_attention"] = attention_heads
    tensors["expected_l1_gated_attention"] = gated_attention
    tensors["expected_l1_mixer"] = l1_mixer
    tensors["expected_l1_after_mixer"] = l1_after_mixer
    tensors["expected_output"] = output

    order = [
        "input",
        "l0_input_norm",
        "l0_in_proj_qkv",
        "l0_in_proj_z",
        "l0_in_proj_b",
        "l0_in_proj_a",
        "l0_conv",
        "l0_a_log",
        "l0_dt_bias",
        "l0_gated_norm",
        "l0_out_proj",
        "l0_post_norm",
        "l0_gate_proj",
        "l0_up_proj",
        "l0_down_proj",
        "l1_input_norm",
        "l1_q_proj",
        "l1_k_proj",
        "l1_v_proj",
        "l1_q_norm",
        "l1_k_norm",
        "l1_o_proj",
        "l1_post_norm",
        "l1_gate_proj",
        "l1_up_proj",
        "l1_down_proj",
        "expected_l0_normed",
        "expected_l0_post_conv",
        "expected_l0_gate",
        "expected_l0_core_out",
        "expected_l0_state",
        "expected_l0_gated",
        "expected_l0_mixer",
        "expected_l0_after_mixer",
        "expected_l0_after_mlp",
        "expected_l1_normed",
        "expected_l1_query",
        "expected_l1_key",
        "expected_l1_attention",
        "expected_l1_gated_attention",
        "expected_l1_mixer",
        "expected_l1_after_mixer",
        "expected_output",
    ]

    header = struct.pack(
        "<8s13I2f",
        MAGIC,
        VERSION,
        sequence_length,
        hidden_size,
        linear_k_heads,
        linear_v_heads,
        linear_k_dim,
        linear_v_dim,
        conv_kernel,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        intermediate_size,
        epsilon,
        theta,
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
