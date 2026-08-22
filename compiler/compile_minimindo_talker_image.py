#!/usr/bin/env python3
"""Pack the native MiniMind-O Talker into a mmap-friendly row-Q8 image."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"MMOTALK1"
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
    return tensor.detach().float().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()


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
    stream.write(b"\0" * (align64(stream.tell()) - stream.tell()))


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
    layers = int(config["num_talker_hidden_layers"])
    hidden = int(config["talker_hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["audio_vocab_size"])
    adapters, rank = 8, 256
    # base embed + 16 embed adapters + 10 projections + 44 blocks + norm
    # + base head + 16 head adapters + speaker projection + 2 scales.
    tensor_count = 1 + 16 + 10 + layers * 11 + 1 + 1 + 16 + 1 + 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w+b") as stream:
        stream.write(b"\0" * HEADER_BYTES)
        write_tensor(stream, state["talker.embed_tokens.base.weight"], TYPE_Q8_ROW)
        for index in range(adapters):
            write_tensor(stream, state[f"talker.embed_tokens.adapters.{index}.0.weight"], TYPE_Q8_ROW)
            write_tensor(stream, state[f"talker.embed_tokens.adapters.{index}.2.weight"], TYPE_Q8_ROW)
        for prefix in ("talker.codec_proj", "talker.embed_proj"):
            for suffix, kind in (("0.weight", TYPE_Q8_ROW), ("0.bias", TYPE_F32),
                                 ("2.weight", TYPE_Q8_ROW), ("2.bias", TYPE_F32),
                                 ("3.weight", TYPE_F32)):
                write_tensor(stream, state[f"{prefix}.{suffix}"], kind)
        for layer in range(layers):
            prefix = f"talker.layers.{layer}."
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
        write_tensor(stream, state["talker.norm.weight"], TYPE_F32)
        write_tensor(stream, state["talker.lm_head.base.weight"], TYPE_Q8_ROW)
        for index in range(adapters):
            write_tensor(stream, state[f"talker.lm_head.adapters.{index}.0.weight"], TYPE_Q8_ROW)
            write_tensor(stream, state[f"talker.lm_head.adapters.{index}.2.weight"], TYPE_Q8_ROW)
        write_tensor(stream, state["talker.spk_proj.weight"], TYPE_Q8_ROW)
        write_tensor(stream, state["talker.text_scale"].reshape(1), TYPE_F32)
        write_tensor(stream, state["talker.audio_scale"].reshape(1), TYPE_F32)
        file_bytes = stream.tell()
        header = struct.pack(
            "<8s13I2f4x2Q64s",
            MAGIC, VERSION, HEADER_BYTES, layers, hidden, heads, kv_heads,
            head_dim, intermediate, vocab, adapters, rank,
            int(config["audio_pad_token"]), tensor_count,
            float(config["rms_norm_eps"]), float(config["rope_theta"]),
            HEADER_BYTES, file_bytes, actual_sha.encode("ascii"),
        )
        stream.seek(0)
        stream.write(header)
        stream.flush()
    print(json.dumps({"output": str(args.output), "bytes": file_bytes,
                      "checkpoint_sha256": actual_sha,
                      "image_sha256": file_sha256(args.output),
                      "tensors": tensor_count}))


if __name__ == "__main__":
    main()
