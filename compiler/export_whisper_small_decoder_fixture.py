#!/usr/bin/env python3
"""Export cached-decoder boundaries from the pinned real small.en weights."""

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
    return np.frombuffer(source.read_bytes(name), dtype="<f4").reshape(info.shape)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    wide = x.astype(np.float64)
    mean = wide.mean(axis=-1, keepdims=True)
    variance = ((wide - mean) ** 2).mean(axis=-1, keepdims=True)
    return (((wide - mean) / np.sqrt(variance + 1.0e-5)) *
            weight.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    result = x.astype(np.float64) @ weight.astype(np.float64).T
    if bias is not None:
        result += bias.astype(np.float64)
    return result.astype(np.float32)


def gelu(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape, dtype=np.float32)
    for index, value in enumerate(values.reshape(-1)):
        number = float(value)
        result.reshape(-1)[index] = 0.5 * number * (1.0 + math.erf(number / math.sqrt(2.0)))
    return result


def attention(query: np.ndarray, key: np.ndarray, value: np.ndarray,
              causal: bool) -> np.ndarray:
    tokens = query.shape[0]
    output = np.empty_like(query)
    for head in range(12):
        channels = slice(head * 64, (head + 1) * 64)
        logits = query[:, channels].astype(np.float64) @ key[:, channels].astype(np.float64).T / 8.0
        if causal:
            mask = np.triu(np.ones(logits.shape, dtype=bool), k=1)
            logits[mask] = -np.inf
        logits -= logits.max(axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=-1, keepdims=True)
        output[:, channels] = (probability @ value[:, channels].astype(np.float64)).astype(np.float32)
    return output


def decoder(source: SafetensorsFile, encoder: np.ndarray,
            tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embedding = tensor(source, "model.decoder.embed_tokens.weight")
    positions = tensor(source, "model.decoder.embed_positions.weight")
    hidden = (embedding[tokens] + positions[:len(tokens)]).astype(np.float32)
    for layer in range(12):
        prefix = f"model.decoder.layers.{layer}."
        normalized = layer_norm(
            hidden, tensor(source, prefix + "self_attn_layer_norm.weight"),
            tensor(source, prefix + "self_attn_layer_norm.bias"))
        query = linear(normalized, tensor(source, prefix + "self_attn.q_proj.weight"),
                       tensor(source, prefix + "self_attn.q_proj.bias"))
        key = linear(normalized, tensor(source, prefix + "self_attn.k_proj.weight"), None)
        value = linear(normalized, tensor(source, prefix + "self_attn.v_proj.weight"),
                       tensor(source, prefix + "self_attn.v_proj.bias"))
        context = attention(query, key, value, causal=True)
        hidden = (hidden + linear(context,
                  tensor(source, prefix + "self_attn.out_proj.weight"),
                  tensor(source, prefix + "self_attn.out_proj.bias"))).astype(np.float32)

        normalized = layer_norm(
            hidden, tensor(source, prefix + "encoder_attn_layer_norm.weight"),
            tensor(source, prefix + "encoder_attn_layer_norm.bias"))
        query = linear(normalized, tensor(source, prefix + "encoder_attn.q_proj.weight"),
                       tensor(source, prefix + "encoder_attn.q_proj.bias"))
        key = linear(encoder, tensor(source, prefix + "encoder_attn.k_proj.weight"), None)
        value = linear(encoder, tensor(source, prefix + "encoder_attn.v_proj.weight"),
                       tensor(source, prefix + "encoder_attn.v_proj.bias"))
        context = attention(query, key, value, causal=False)
        hidden = (hidden + linear(context,
                  tensor(source, prefix + "encoder_attn.out_proj.weight"),
                  tensor(source, prefix + "encoder_attn.out_proj.bias"))).astype(np.float32)

        normalized = layer_norm(
            hidden, tensor(source, prefix + "final_layer_norm.weight"),
            tensor(source, prefix + "final_layer_norm.bias"))
        mlp = gelu(linear(normalized, tensor(source, prefix + "fc1.weight"),
                          tensor(source, prefix + "fc1.bias")))
        hidden = (hidden + linear(mlp, tensor(source, prefix + "fc2.weight"),
                                  tensor(source, prefix + "fc2.bias"))).astype(np.float32)
    hidden = layer_norm(hidden, tensor(source, "model.decoder.layer_norm.weight"),
                        tensor(source, "model.decoder.layer_norm.bias"))
    logits = hidden.astype(np.float64) @ embedding.astype(np.float64).T
    return hidden, logits.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/whisper-small.en/pins.json").read_text())
    checkpoint_pin = pins["model"]["compiler_checkpoint"]["files"]["model.safetensors"]
    if args.checkpoint.stat().st_size != checkpoint_pin["bytes"]:
        raise ValueError("checkpoint byte count does not match the pin")
    if not args.skip_checkpoint_hash and sha256_file(args.checkpoint) != checkpoint_pin["sha256"]:
        raise ValueError("checkpoint digest does not match the pin")

    started = time.monotonic()
    audio_frames = 3
    index = np.arange(audio_frames * 768, dtype=np.float64)
    encoder = (0.17 * np.sin(index * 0.017 + 0.3) +
               0.04 * np.cos(index * 0.041 - 0.2)).astype("<f4").reshape(audio_frames, 768)
    tokens = np.asarray([50257, 50362, 464], dtype="<u4")
    hidden_steps = []
    top_tokens = []
    top_logits = []
    with SafetensorsFile(args.checkpoint) as source:
        for length in range(1, len(tokens) + 1):
            hidden, logits = decoder(source, encoder, tokens[:length])
            top = int(np.argmax(logits[-1]))
            hidden_steps.append(hidden[-1].copy())
            top_tokens.append(top)
            top_logits.append(float(logits[-1, top]))
            print(f"reference decoder_steps={length}/{len(tokens)} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)

    header = HEADER.pack(b"WHDEC001", 1, audio_frames, len(tokens), 768, 51864, 3)
    payload = (header + encoder.astype("<f4").tobytes() + tokens.tobytes() +
               np.asarray(hidden_steps, dtype="<f4").tobytes() +
               np.asarray(top_tokens, dtype="<u4").tobytes() +
               np.asarray(top_logits, dtype="<f4").tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "format": "WHDEC001",
        "checkpoint_sha256": checkpoint_pin["sha256"],
        "audio_frames": audio_frames,
        "input_tokens": tokens.tolist(),
        "expected_next_tokens": top_tokens,
        "fixture_bytes": len(payload),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "reference": "independent NumPy full-prefix causal decoder; float64 accumulations",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
