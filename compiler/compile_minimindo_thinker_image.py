#!/usr/bin/env python3
"""Pack the MiniMind-O Thinker into a mmap-friendly row-Q8 image."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"MMOTSK1\0"
VERSION = 1
HEADER_BYTES = 4096
TYPE_F32 = 1
TYPE_Q8_ROW = 2
EXPECTED_SHA256 = "21530f9bbc540f461e2c0e29292ad359781d4d984d1e0c994510945f9b0edaab"


def align64(value):
    return (value + 63) & ~63


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_payload(tensor):
    values = tensor.detach().float().contiguous().cpu().numpy().astype("<f4", copy=False)
    return values.tobytes(order="C")


def q8_payload(tensor):
    values = tensor.detach().float().contiguous().cpu()
    if values.ndim != 2:
        raise ValueError(f"Q8 tensor must be a matrix, got {tuple(values.shape)}")
    output = bytearray()
    for row in values:
        maximum = float(row.abs().max())
        scale = maximum / 127.0 if maximum else 1.0
        quantized = torch.clamp(torch.round(row / scale), -127, 127).to(torch.int8)
        output.extend(struct.pack("<f", scale))
        output.extend(quantized.numpy().astype(np.int8, copy=False).tobytes())
    return bytes(output)


def write_tensor(stream, tensor, kind):
    rows = tensor.shape[0] if tensor.ndim == 2 else 1
    cols = tensor.shape[1] if tensor.ndim == 2 else tensor.numel()
    payload = f32_payload(tensor) if kind == TYPE_F32 else q8_payload(tensor)
    stream.write(struct.pack("<IIIIQ", kind, rows, cols, 0, len(payload)))
    stream.write(payload)
    padding = align64(stream.tell()) - stream.tell()
    if padding:
        stream.write(b"\0" * padding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()

    actual_sha = file_sha256(args.checkpoint)
    if actual_sha != EXPECTED_SHA256 and not args.allow_unpinned:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {actual_sha}")
    config = json.loads(args.config.read_text())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    tensor_count = 1 + layers * 11 + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w+b") as stream:
        stream.write(b"\0" * HEADER_BYTES)
        write_tensor(stream, state["model.embed_tokens.weight"], TYPE_Q8_ROW)
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            order = [
                ("input_layernorm.weight", TYPE_F32),
                ("self_attn.q_norm.weight", TYPE_F32),
                ("self_attn.k_norm.weight", TYPE_F32),
                ("post_attention_layernorm.weight", TYPE_F32),
                ("self_attn.q_proj.weight", TYPE_Q8_ROW),
                ("self_attn.k_proj.weight", TYPE_Q8_ROW),
                ("self_attn.v_proj.weight", TYPE_Q8_ROW),
                ("self_attn.o_proj.weight", TYPE_Q8_ROW),
                ("mlp.gate_proj.weight", TYPE_Q8_ROW),
                ("mlp.up_proj.weight", TYPE_Q8_ROW),
                ("mlp.down_proj.weight", TYPE_Q8_ROW),
            ]
            for suffix, kind in order:
                write_tensor(stream, state[prefix + suffix], kind)
        write_tensor(stream, state["model.norm.weight"], TYPE_F32)
        file_bytes = stream.tell()
        header = struct.pack(
            "<8s10I2f2Q64s",
            MAGIC, VERSION, HEADER_BYTES, layers, hidden, heads, kv_heads,
            head_dim, intermediate, vocab, tensor_count,
            float(config["rms_norm_eps"]), float(config["rope_theta"]),
            HEADER_BYTES, file_bytes, actual_sha.encode("ascii"),
        )
        stream.seek(0)
        stream.write(header)
        stream.flush()
    image_sha = file_sha256(args.output)
    print(json.dumps({"output": str(args.output), "bytes": file_bytes,
                      "checkpoint_sha256": actual_sha,
                      "image_sha256": image_sha,
                      "tensors": tensor_count}))


if __name__ == "__main__":
    main()
