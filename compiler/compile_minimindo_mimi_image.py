#!/usr/bin/env python3
"""Pack the MiniMind-O Mimi decode-only graph for the native C runtime."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


MAGIC = b"MMOMIMI1"
VERSION = 1
HEADER_BYTES = 4096
TYPE_F32 = 1
TYPE_Q8_ROW = 2
EXPECTED_SHA256 = "7542ee039d3025d5089cf227d21df64b6b8eff08fcd376a11a1fbd178dd9d3f5"


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
    values = tensor.detach().float().reshape(tensor.shape[0], -1).contiguous().cpu()
    output = bytearray()
    for row in values:
        maximum = float(row.abs().max())
        scale = maximum / 127.0 if maximum else 1.0
        quantized = torch.clamp(torch.round(row / scale), -127, 127).to(torch.int8)
        output.extend(struct.pack("<f", scale))
        output.extend(quantized.numpy().astype(np.int8, copy=False).tobytes())
    return bytes(output)


def write_tensor(stream, tensor, kind):
    values = tensor.detach().float()
    rows = values.shape[0] if values.ndim > 1 else 1
    cols = values.numel() // rows
    payload = f32_payload(values) if kind == TYPE_F32 else q8_payload(values)
    stream.write(struct.pack("<IIIIQ", kind, rows, cols, 0, len(payload)))
    stream.write(payload)
    stream.write(b"\0" * (align64(stream.tell()) - stream.tell()))


def codebook(state, prefix):
    return state[prefix + ".embed_sum"] / state[prefix + ".cluster_usage"].clamp(min=1e-5)[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()
    actual_sha = file_sha256(args.model)
    if actual_sha != EXPECTED_SHA256 and not args.allow_unpinned:
        raise SystemExit(f"Mimi SHA-256 mismatch: {actual_sha}")
    config = json.loads(args.config.read_text())
    with safe_open(args.model, framework="pt", device="cpu") as source:
        state = {name: source.get_tensor(name) for name in source.keys()}

    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    codebook_dim = int(config["codebook_dim"])
    codebook_size = int(config["codebook_size"])
    codebooks = 8
    ratios = list(map(int, config["upsampling_ratios"]))
    tensor_count = codebooks + 2 + 1 + layers * 12 + 28

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w+b") as stream:
        stream.write(b"\0" * HEADER_BYTES)
        semantic = "quantizer.semantic_residual_vector_quantizer"
        acoustic = "quantizer.acoustic_residual_vector_quantizer"
        write_tensor(stream, codebook(state, semantic + ".layers.0.codebook"), TYPE_F32)
        for index in range(7):
            write_tensor(stream, codebook(state, acoustic + f".layers.{index}.codebook"), TYPE_F32)
        write_tensor(stream, state[semantic + ".output_proj.weight"], TYPE_Q8_ROW)
        write_tensor(stream, state[acoustic + ".output_proj.weight"], TYPE_Q8_ROW)
        # grouped transposed convolution; its stored [in, 1, kernel] layout is
        # already one independent output row per channel.
        write_tensor(stream, state["upsample.conv.weight"], TYPE_Q8_ROW)

        for layer in range(layers):
            p = f"decoder_transformer.layers.{layer}."
            for suffix, kind in (
                ("input_layernorm.weight", TYPE_F32),
                ("input_layernorm.bias", TYPE_F32),
                ("post_attention_layernorm.weight", TYPE_F32),
                ("post_attention_layernorm.bias", TYPE_F32),
                ("self_attn_layer_scale.scale", TYPE_F32),
                ("mlp_layer_scale.scale", TYPE_F32),
                ("self_attn.q_proj.weight", TYPE_Q8_ROW),
                ("self_attn.k_proj.weight", TYPE_Q8_ROW),
                ("self_attn.v_proj.weight", TYPE_Q8_ROW),
                ("self_attn.o_proj.weight", TYPE_Q8_ROW),
                ("mlp.fc1.weight", TYPE_Q8_ROW),
                ("mlp.fc2.weight", TYPE_Q8_ROW),
            ):
                write_tensor(stream, state[p + suffix], kind)

        conv_names = ["decoder.layers.0.conv"]
        for base in (2, 5, 8, 11):
            conv_names.extend((f"decoder.layers.{base}.conv",
                               f"decoder.layers.{base + 1}.block.1.conv",
                               f"decoder.layers.{base + 1}.block.3.conv"))
        conv_names.append("decoder.layers.14.conv")
        for index, name in enumerate(conv_names):
            weight = state[name + ".weight"]
            # ConvTranspose1d is stored [in, out, kernel]; transpose it to the
            # decoder runtime's output-major row representation.
            if index in (1, 4, 7, 10):
                weight = weight.permute(1, 0, 2).contiguous()
            write_tensor(stream, weight, TYPE_Q8_ROW)
            write_tensor(stream, state[name + ".bias"], TYPE_F32)

        file_bytes = stream.tell()
        header = struct.pack(
            "<8s16I2f4I2Q64s",
            MAGIC, VERSION, HEADER_BYTES, layers, hidden, heads, head_dim,
            intermediate, codebooks, codebook_size, codebook_dim,
            int(config["sampling_rate"]), int(round(float(config["frame_rate"]) * 10)),
            tensor_count, int(config["sliding_window"]), len(ratios), 0,
            float(config["norm_eps"]), float(config["rope_theta"]),
            ratios[0], ratios[1], ratios[2], ratios[3],
            HEADER_BYTES, file_bytes, actual_sha.encode("ascii"),
        )
        stream.seek(0); stream.write(header); stream.flush()
    print(json.dumps({"output": str(args.output), "bytes": file_bytes,
                      "source_sha256": actual_sha,
                      "image_sha256": file_sha256(args.output),
                      "tensors": tensor_count}))


if __name__ == "__main__":
    main()
