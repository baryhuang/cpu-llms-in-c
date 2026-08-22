#!/usr/bin/env python3
"""Pack MiniMind-O's frozen SenseVoice encoder plus audio projector."""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"MMOSENS1"
HEADER_BYTES = 4096
TYPE_F32 = 1
TYPE_Q8_ROW = 2
SENSE_SHA = "218811976815c1673e1b852dc383d78987b229268f78e7ebd0a1fe67229c83dd"
MINIMIND_SHA = "21530f9bbc540f461e2c0e29292ad359781d4d984d1e0c994510945f9b0edaab"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align64(value): return (value + 63) & ~63


def f32_payload(tensor):
    return tensor.detach().float().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()


def q8_payload(tensor):
    values = tensor.detach().float().reshape(tensor.shape[0], -1).contiguous().cpu()
    output = bytearray()
    for row in values:
        maximum = float(row.abs().max())
        scale = maximum / 127.0 if maximum else 1.0
        q = torch.clamp(torch.round(row / scale), -127, 127).to(torch.int8)
        output.extend(struct.pack("<f", scale)); output.extend(q.numpy().astype(np.int8, copy=False).tobytes())
    return bytes(output)


def write_tensor(stream, tensor, kind):
    tensor = tensor.detach().float()
    rows = tensor.shape[0] if tensor.ndim > 1 else 1
    cols = tensor.numel() // rows
    payload = f32_payload(tensor) if kind == TYPE_F32 else q8_payload(tensor)
    stream.write(struct.pack("<IIIIQ", kind, rows, cols, 0, len(payload)))
    stream.write(payload); stream.write(b"\0" * (align64(stream.tell()) - stream.tell()))


def c_array(path, name, count):
    text = path.read_text()
    match = re.search(rf"{re.escape(name)}\s*\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match: raise ValueError(f"missing C array {name}")
    values = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    if len(values) != count: raise ValueError(f"{name}: expected {count}, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def layer_tensors(state, prefix):
    qkv_weight = state[prefix + ".self_attn.linear_q_k_v.weight"]
    qkv_bias = state[prefix + ".self_attn.linear_q_k_v.bias"]
    return [
        (state[prefix + ".norm1.weight"], TYPE_F32),
        (state[prefix + ".norm1.bias"], TYPE_F32),
        (qkv_weight[0:512], TYPE_Q8_ROW), (qkv_bias[0:512], TYPE_F32),
        (qkv_weight[512:1024], TYPE_Q8_ROW), (qkv_bias[512:1024], TYPE_F32),
        (qkv_weight[1024:1536], TYPE_Q8_ROW), (qkv_bias[1024:1536], TYPE_F32),
        (state[prefix + ".self_attn.fsmn_block.weight"], TYPE_F32),
        (state[prefix + ".self_attn.linear_out.weight"], TYPE_Q8_ROW),
        (state[prefix + ".self_attn.linear_out.bias"], TYPE_F32),
        (state[prefix + ".norm2.weight"], TYPE_F32),
        (state[prefix + ".norm2.bias"], TYPE_F32),
        (state[prefix + ".feed_forward.w_1.weight"], TYPE_Q8_ROW),
        (state[prefix + ".feed_forward.w_1.bias"], TYPE_F32),
        (state[prefix + ".feed_forward.w_2.weight"], TYPE_Q8_ROW),
        (state[prefix + ".feed_forward.w_2.bias"], TYPE_F32),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sense-checkpoint", type=Path, required=True)
    parser.add_argument("--minimind-checkpoint", type=Path, required=True)
    parser.add_argument("--mel-header", type=Path, required=True)
    parser.add_argument("--cmvn-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()
    sense_sha, minimind_sha = sha256(args.sense_checkpoint), sha256(args.minimind_checkpoint)
    if not args.allow_unpinned and (sense_sha != SENSE_SHA or minimind_sha != MINIMIND_SHA):
        raise SystemExit(f"source SHA mismatch: SenseVoice={sense_sha}, MiniMind={minimind_sha}")
    sense = torch.load(args.sense_checkpoint, map_location="cpu", weights_only=True)
    minimind = torch.load(args.minimind_checkpoint, map_location="cpu", weights_only=True)
    mel = c_array(args.mel_header, "LogMelFilterMelArray", 80 * 256).reshape(80, 256)
    means = c_array(args.cmvn_header, "CMVN_MEANS", 560)
    variances = c_array(args.cmvn_header, "CMVN_VARS", 560)
    tensor_count = 4 + 70 * 17 + 2 + 2 + 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w+b") as stream:
        stream.write(b"\0" * HEADER_BYTES)
        for tensor in (mel, means, variances, sense["embed.weight"]): write_tensor(stream, tensor, TYPE_F32)
        prefixes = ["encoder.encoders0.0"] + [f"encoder.encoders.{i}" for i in range(49)]
        prefixes += [f"encoder.tp_encoders.{i}" for i in range(20)]
        for prefix in prefixes:
            for tensor, kind in layer_tensors(sense, prefix): write_tensor(stream, tensor, kind)
        for name in ("encoder.after_norm.weight", "encoder.after_norm.bias",
                     "encoder.tp_norm.weight", "encoder.tp_norm.bias"):
            write_tensor(stream, sense[name], TYPE_F32)
        for name, kind in (
            ("audio_proj.mlp.0.weight", TYPE_F32), ("audio_proj.mlp.0.bias", TYPE_F32),
            ("audio_proj.mlp.1.weight", TYPE_Q8_ROW), ("audio_proj.mlp.1.bias", TYPE_F32),
            ("audio_proj.mlp.3.weight", TYPE_Q8_ROW), ("audio_proj.mlp.3.bias", TYPE_F32)):
            write_tensor(stream, minimind[name], kind)
        file_bytes = stream.tell()
        # 16 u32 fields, 2 floats, 2 offsets, and two pinned source hashes.
        header = struct.pack("<8s16I2f2Q64s64s", MAGIC, 1, HEADER_BYTES,
            70, 50, 20, 560, 512, 4, 2048, 11, 80, 256, 7, 6,
            tensor_count, 0, 1e-5, 10000.0, HEADER_BYTES, file_bytes,
            sense_sha.encode(), minimind_sha.encode())
        stream.seek(0); stream.write(header); stream.flush()
    print(json.dumps({"output":str(args.output),"bytes":file_bytes,"tensors":tensor_count,
                      "sense_sha256":sense_sha,"minimind_sha256":minimind_sha,
                      "image_sha256":sha256(args.output)}))


if __name__ == "__main__": main()
