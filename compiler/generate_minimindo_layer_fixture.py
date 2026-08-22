#!/usr/bin/env python3
"""Generate a deterministic MiniMind-O transformer-layer oracle fixture."""

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    return weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps))


def rope(x, theta, start_position):
    rows, _, dim = x.shape
    half = dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:half] / dim))
    pos = torch.arange(start_position, start_position + rows, dtype=torch.float32)
    angles = torch.outer(pos, inv)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(1)
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def build():
    torch.manual_seed(0x1130)
    rows, hidden, heads, kv_heads, head_dim, intermediate = 4, 16, 4, 2, 4, 24
    start_position, eps, theta = 3, 1.0e-6, 1.0e6

    def rand(*shape):
        return (torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0) * 0.18

    tensors = {
        "input": rand(rows, hidden),
        "input_norm": 1.0 + rand(hidden) * 0.2,
        "q_proj": rand(hidden, hidden),
        "k_proj": rand(kv_heads * head_dim, hidden),
        "v_proj": rand(kv_heads * head_dim, hidden),
        "q_norm": 1.0 + rand(head_dim) * 0.2,
        "k_norm": 1.0 + rand(head_dim) * 0.2,
        "o_proj": rand(hidden, hidden),
        "post_attention_norm": 1.0 + rand(hidden) * 0.2,
        "gate_proj": rand(intermediate, hidden),
        "up_proj": rand(intermediate, hidden),
        "down_proj": rand(hidden, intermediate),
    }
    normed = rms_norm(tensors["input"], tensors["input_norm"], eps)
    query = rope(rms_norm(F.linear(normed, tensors["q_proj"]).view(rows, heads, head_dim),
                          tensors["q_norm"], eps), theta, start_position)
    key = rope(rms_norm(F.linear(normed, tensors["k_proj"]).view(rows, kv_heads, head_dim),
                        tensors["k_norm"], eps), theta, start_position)
    value = F.linear(normed, tensors["v_proj"]).view(rows, kv_heads, head_dim)
    repeat = heads // kv_heads
    key_attn = key.repeat_interleave(repeat, dim=1).transpose(0, 1)
    value_attn = value.repeat_interleave(repeat, dim=1).transpose(0, 1)
    scores = torch.matmul(query.transpose(0, 1), key_attn.transpose(-2, -1)) / head_dim**0.5
    scores = scores.masked_fill(torch.ones(rows, rows, dtype=torch.bool).triu(1), float("-inf"))
    heads_out = torch.matmul(scores.softmax(dim=-1), value_attn).transpose(0, 1).reshape(rows, hidden)
    attention = F.linear(heads_out, tensors["o_proj"])
    after_attention = tensors["input"] + attention
    normalized_mlp = rms_norm(after_attention, tensors["post_attention_norm"], eps)
    mlp = F.linear(F.silu(F.linear(normalized_mlp, tensors["gate_proj"])) *
                   F.linear(normalized_mlp, tensors["up_proj"]), tensors["down_proj"])
    tensors.update({
        "expected_normed": normed,
        "expected_query": query,
        "expected_key": key,
        "expected_value": value,
        "expected_attention": attention,
        "expected_after_attention": after_attention,
        "expected_normalized_mlp": normalized_mlp,
        "expected_output": after_attention + mlp,
    })
    return (rows, hidden, heads, kv_heads, head_dim, intermediate, start_position), eps, theta, tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config, eps, theta, tensors = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<8sI7I2fI", b"MMOLYR1\0", 1, *config, eps, theta, len(tensors)))
        for name, tensor in tensors.items():
            values = tensor.contiguous().view(-1).tolist()
            encoded = name.encode("ascii")
            stream.write(struct.pack("<II", len(encoded), len(values)))
            stream.write(encoded)
            stream.write(struct.pack(f"<{len(values)}f", *values))
    print(args.output)


if __name__ == "__main__":
    main()
