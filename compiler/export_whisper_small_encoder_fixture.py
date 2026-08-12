#!/usr/bin/env python3
"""Export real small.en encoder boundaries from the pinned F32 checkpoint.

The fixture contains deterministic log-Mel input and three independent NumPy
reference boundaries: encoder stem, block 0, and all 12 blocks plus final LN.
Weights are never copied into the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

try:
    from .q4_image import sha256_file
    from .safetensors_file import SafetensorsFile
except ImportError:
    from q4_image import sha256_file
    from safetensors_file import SafetensorsFile


HEADER = struct.Struct("<8s6I")


def tensor(source: SafetensorsFile, name: str) -> np.ndarray:
    info = source.tensors[name]
    if info.dtype != "F32":
        raise ValueError(f"expected F32 tensor: {name}")
    return np.frombuffer(source.read_bytes(name), dtype="<f4").reshape(info.shape)


def gelu(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    result = np.empty(flat.shape, dtype=np.float32)
    root_two = math.sqrt(2.0)
    for index, value in enumerate(flat):
        number = float(value)
        result[index] = 0.5 * number * (1.0 + math.erf(number / root_two))
    return result.reshape(values.shape)


def layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    wide = values.astype(np.float64)
    mean = wide.mean(axis=-1, keepdims=True)
    variance = ((wide - mean) ** 2).mean(axis=-1, keepdims=True)
    return (((wide - mean) / np.sqrt(variance + 1.0e-5)) *
            weight.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)


def linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    result = values.astype(np.float64) @ weight.astype(np.float64).T
    if bias is not None:
        result += bias.astype(np.float64)
    return result.astype(np.float32)


def stem(source: SafetensorsFile, mel: np.ndarray) -> np.ndarray:
    conv1_weight = tensor(source, "model.encoder.conv1.weight")
    conv1_bias = tensor(source, "model.encoder.conv1.bias")
    conv2_weight = tensor(source, "model.encoder.conv2.weight")
    conv2_bias = tensor(source, "model.encoder.conv2.bias")
    frames = mel.shape[1]
    padded = np.pad(mel, ((0, 0), (1, 1)))
    first = np.broadcast_to(conv1_bias[:, None], (768, frames)).astype(np.float64).copy()
    for tap in range(3):
        first += conv1_weight[:, :, tap].astype(np.float64) @ padded[:, tap:tap + frames].astype(np.float64)
    first = gelu(first.astype(np.float32))
    output_frames = (frames + 1) // 2
    padded = np.pad(first, ((0, 0), (1, 1)))
    second = np.broadcast_to(conv2_bias[:, None], (768, output_frames)).astype(np.float64).copy()
    sources = np.arange(output_frames) * 2
    for tap in range(3):
        second += conv2_weight[:, :, tap].astype(np.float64) @ padded[:, sources + tap].astype(np.float64)
    second = gelu(second.astype(np.float32)).T
    positions = tensor(source, "model.encoder.embed_positions.weight")[:output_frames]
    return (second + positions).astype(np.float32)


def block(source: SafetensorsFile, layer: int, values: np.ndarray) -> np.ndarray:
    prefix = f"model.encoder.layers.{layer}."
    norm = layer_norm(
        values,
        tensor(source, prefix + "self_attn_layer_norm.weight"),
        tensor(source, prefix + "self_attn_layer_norm.bias"),
    )
    query = linear(norm, tensor(source, prefix + "self_attn.q_proj.weight"),
                   tensor(source, prefix + "self_attn.q_proj.bias"))
    key = linear(norm, tensor(source, prefix + "self_attn.k_proj.weight"), None)
    value = linear(norm, tensor(source, prefix + "self_attn.v_proj.weight"),
                   tensor(source, prefix + "self_attn.v_proj.bias"))
    frames = values.shape[0]
    context = np.empty((frames, 768), dtype=np.float32)
    for head in range(12):
        channels = slice(head * 64, (head + 1) * 64)
        logits = query[:, channels].astype(np.float64) @ key[:, channels].astype(np.float64).T / 8.0
        logits -= logits.max(axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=-1, keepdims=True)
        context[:, channels] = (probability @ value[:, channels].astype(np.float64)).astype(np.float32)
    values = (values + linear(
        context,
        tensor(source, prefix + "self_attn.out_proj.weight"),
        tensor(source, prefix + "self_attn.out_proj.bias"),
    )).astype(np.float32)
    norm = layer_norm(
        values,
        tensor(source, prefix + "final_layer_norm.weight"),
        tensor(source, prefix + "final_layer_norm.bias"),
    )
    hidden = gelu(linear(norm, tensor(source, prefix + "fc1.weight"),
                         tensor(source, prefix + "fc1.bias")))
    return (values + linear(hidden, tensor(source, prefix + "fc2.weight"),
                            tensor(source, prefix + "fc2.bias"))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-frames", type=int, default=4)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.input_frames <= 3000:
        raise ValueError("input-frames must be in 1..3000")

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/whisper-small.en/pins.json").read_text())
    checkpoint_pin = pins["model"]["compiler_checkpoint"]["files"]["model.safetensors"]
    if args.checkpoint.stat().st_size != checkpoint_pin["bytes"]:
        raise ValueError("checkpoint byte count does not match the pin")
    if not args.skip_checkpoint_hash and sha256_file(args.checkpoint) != checkpoint_pin["sha256"]:
        raise ValueError("checkpoint digest does not match the pin")

    started = time.monotonic()
    indices = np.arange(80 * args.input_frames, dtype=np.float64)
    mel = (0.21 * np.sin(indices * 0.013 + 0.2) +
           0.07 * np.cos(indices * 0.031 - 0.4)).astype("<f4").reshape(80, args.input_frames)
    with SafetensorsFile(args.checkpoint) as source:
        stem_output = stem(source, mel)
        values = block(source, 0, stem_output)
        layer0_output = values.copy()
        for layer in range(1, 12):
            values = block(source, layer, values)
            print(f"reference layers={layer + 1}/12 elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
        final_output = layer_norm(
            values,
            tensor(source, "model.encoder.layer_norm.weight"),
            tensor(source, "model.encoder.layer_norm.bias"),
        )

    output_frames = stem_output.shape[0]
    header = HEADER.pack(b"WHREAL01", 1, args.input_frames, output_frames, 768, 3, 0)
    payload = header + b"".join(
        array.astype("<f4").tobytes(order="C")
        for array in (mel, stem_output, layer0_output, final_output)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "format": "WHREAL01",
        "checkpoint_sha256": checkpoint_pin["sha256"],
        "input": "deterministic synthetic 80-bin log-Mel; not an audio quality test",
        "input_frames": args.input_frames,
        "output_frames": output_frames,
        "boundaries": ["stem", "layer_0", "layer_11_plus_final_layer_norm"],
        "fixture_bytes": len(payload),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "reference": "independent NumPy implementation of the pinned graph using float64 accumulations",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
