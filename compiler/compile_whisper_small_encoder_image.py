#!/usr/bin/env python3
"""Compile the pinned Whisper small.en graph into an mmap-ready image.

Encoder-only F32 is the exact correctness image. Full-graph images also carry
the decoder and byte-level tokenizer tables. Quantized images remain derived
artifacts that must be evaluated against the exact boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .q4_image import (align, q4_byte_count, q8_byte_count,
                           quantize_q4_grouped, quantize_q8_grouped, sha256_file)
    from .safetensors_file import SafetensorsFile
except ImportError:
    from q4_image import (align, q4_byte_count, q8_byte_count,
                          quantize_q4_grouped, quantize_q8_grouped, sha256_file)
    from safetensors_file import SafetensorsFile


MAGIC = b"WHSENC01"
KIND_F32 = 1
KIND_U32 = 2
KIND_U8 = 3
KIND_Q4_GROUPED = 4
KIND_Q8_GROUPED = 5
HEADER = struct.Struct("<8s8I4Q")
DESCRIPTOR = struct.Struct("<96sII4IQQQ")
MODEL_PREFIX = "model."


@dataclass
class Entry:
    name: str
    shape: tuple[int, ...]
    kind: int = KIND_F32
    payload: bytes | None = None
    source_name: str | None = None
    offset: int = 0
    byte_count: int = 0


def descriptor_bytes(entry: Entry) -> bytes:
    encoded = entry.name.encode("utf-8")
    if len(encoded) >= 96 or len(entry.shape) > 4:
        raise ValueError(f"tensor cannot be represented: {entry.name} {entry.shape}")
    shape = list(entry.shape) + [0] * (4 - len(entry.shape))
    return DESCRIPTOR.pack(
        encoded, entry.kind, len(entry.shape), *shape,
        entry.offset, entry.byte_count, 0,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def matrix_kind(name: str, shape: tuple[int, ...], precision: str) -> int:
    if (len(shape) != 2 or not name.endswith(".weight") or
            shape[1] % 128 != 0):
        return KIND_F32
    if precision in ("q4", "mixed-q4-q8") and name.startswith("model.encoder.layers."):
        return KIND_Q4_GROUPED
    if precision == "q4" and (name.startswith("model.decoder.layers.") or
                              name == "model.decoder.embed_tokens.weight"):
        return KIND_Q4_GROUPED
    if precision == "mixed-q4-q8" and (name.startswith("model.decoder.layers.") or
                                       name == "model.decoder.embed_tokens.weight"):
        return KIND_Q8_GROUPED
    return KIND_F32


def bytes_to_unicode() -> dict[int, str]:
    values = (list(range(ord("!"), ord("~") + 1)) +
              list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    mapped = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(values, mapped)}


def tokenizer_payloads(path: Path) -> list[Entry]:
    spec = json.loads(path.read_text())
    if spec["model"]["type"] != "BPE":
        raise ValueError("expected Whisper BPE tokenizer")
    inverse = {character: byte for byte, character in bytes_to_unicode().items()}
    token_count = 51864
    pieces: list[bytes] = [b""] * token_count
    special = np.zeros(token_count, dtype=np.uint8)
    for piece, token_id in spec["model"]["vocab"].items():
        pieces[token_id] = bytes(inverse[character] for character in piece)
    for token in spec.get("added_tokens", []):
        token_id = token["id"]
        if token_id < token_count:
            pieces[token_id] = token["content"].encode("utf-8")
            special[token_id] = 1 if token.get("special") else 0
    offsets = np.zeros(token_count + 1, dtype="<u4")
    for index, piece in enumerate(pieces):
        offsets[index + 1] = offsets[index] + len(piece)
    blob = b"".join(pieces)
    return [
        Entry("tokenizer.offsets", (token_count + 1,), kind=KIND_U32,
              payload=offsets.tobytes()),
        Entry("tokenizer.bytes", (len(blob),), kind=KIND_U8, payload=blob),
        Entry("tokenizer.special", (token_count,), kind=KIND_U8,
              payload=special.tobytes()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mel-filters", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--precision", choices=("f32", "q4", "mixed-q4-q8"), default="f32")
    parser.add_argument("--graph", choices=("encoder", "full"), default="encoder")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    if args.precision == "mixed-q4-q8" and args.graph != "full":
        raise ValueError("mixed-q4-q8 is defined only for the full graph")
    if args.graph == "full" and args.tokenizer is None:
        raise ValueError("--tokenizer is required for the full graph")

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/whisper-small.en/pins.json").read_text())
    source_pin = pins["model"]["compiler_checkpoint"]
    checkpoint_pin = source_pin["files"]["model.safetensors"]
    config_pin = source_pin["files"]["config.json"]
    if args.checkpoint.stat().st_size != checkpoint_pin["bytes"]:
        raise ValueError("checkpoint byte count does not match the pin")
    if not args.skip_checkpoint_hash and sha256_file(args.checkpoint) != checkpoint_pin["sha256"]:
        raise ValueError("checkpoint digest does not match the pin")
    if args.config.stat().st_size != config_pin["bytes"] or sha256_file(args.config) != config_pin["sha256"]:
        raise ValueError("config does not match the pin")
    if sha256_file(args.mel_filters) != pins["model"]["reference_files"]["whisper/assets/mel_filters.npz"]:
        raise ValueError("mel filter file does not match the pin")
    if args.tokenizer is not None:
        tokenizer_pin = source_pin["files"]["tokenizer.json"]
        if (args.tokenizer.stat().st_size != tokenizer_pin["bytes"] or
                sha256_file(args.tokenizer) != tokenizer_pin["sha256"]):
            raise ValueError("tokenizer does not match the pin")

    config = json.loads(args.config.read_text())
    required = {
        "d_model": 768,
        "encoder_attention_heads": 12,
        "encoder_ffn_dim": 3072,
        "encoder_layers": 12,
        "max_source_positions": 1500,
        "num_mel_bins": 80,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"unexpected config value: {key}={config.get(key)!r}")

    with np.load(args.mel_filters) as archive:
        mel = np.ascontiguousarray(archive["mel_80"], dtype="<f4")
    if mel.shape != (80, 201):
        raise ValueError(f"unexpected mel_80 shape: {mel.shape}")

    started = time.monotonic()
    entries = [Entry("frontend.mel_80", tuple(mel.shape), payload=mel.tobytes())]
    if args.graph == "full":
        entries.extend(tokenizer_payloads(args.tokenizer))
    with SafetensorsFile(args.checkpoint) as source:
        for full_name in sorted(source.tensors):
            if not (full_name.startswith("model.encoder.") or
                    (args.graph == "full" and full_name.startswith("model.decoder."))):
                continue
            info = source.tensors[full_name]
            if info.dtype != "F32":
                raise ValueError(f"exact encoder image requires F32: {full_name} is {info.dtype}")
            shape = tuple(info.shape)
            kind = matrix_kind(full_name, shape, args.precision)
            if kind == KIND_Q4_GROUPED:
                byte_count = q4_byte_count(shape)
            elif kind == KIND_Q8_GROUPED:
                byte_count = q8_byte_count(shape)
            else:
                byte_count = info.byte_count
            entries.append(Entry(full_name[len(MODEL_PREFIX):], shape, kind=kind,
                                 source_name=full_name, byte_count=byte_count))

        directory_offset = HEADER.size
        data_offset = align(directory_offset + len(entries) * DESCRIPTOR.size)
        cursor = data_offset
        for entry in entries:
            if entry.payload is not None:
                entry.byte_count = len(entry.payload)
            entry.offset = cursor
            cursor = align(cursor + entry.byte_count)
        file_bytes = cursor

        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w+b") as output:
            output.truncate(file_bytes)
            output.write(HEADER.pack(
                MAGIC, ((5 if args.precision == "mixed-q4-q8" else
                         (4 if args.precision == "q4" else 3))
                        if args.graph == "full" else
                        (2 if args.precision == "q4" else 1)),
                len(entries), 12, 80, 768, 12, 3072, 1500,
                directory_offset, data_offset, file_bytes, 0,
            ))
            output.seek(directory_offset)
            for entry in entries:
                output.write(descriptor_bytes(entry))
            for index, entry in enumerate(entries, 1):
                if entry.payload is not None:
                    payload = entry.payload
                elif entry.kind == KIND_Q4_GROUPED:
                    raw = source.read_bytes(entry.source_name)
                    array = np.frombuffer(raw, dtype="<f4").reshape(entry.shape)
                    payload = quantize_q4_grouped(array)
                elif entry.kind == KIND_Q8_GROUPED:
                    raw = source.read_bytes(entry.source_name)
                    array = np.frombuffer(raw, dtype="<f4").reshape(entry.shape)
                    payload = quantize_q8_grouped(array)
                else:
                    payload = source.read_bytes(entry.source_name)
                if len(payload) != entry.byte_count:
                    raise AssertionError(f"payload length mismatch: {entry.name}")
                output.seek(entry.offset)
                output.write(payload)
                if index % 24 == 0 or index == len(entries):
                    print(f"copied tensors={index}/{len(entries)} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)

    manifest = {
        "schema_version": 1,
        "format": MAGIC.decode(),
        "graph": args.graph,
        "precision": ({
            "f32": "float32 exact correctness baseline",
            "q4": "signed Q4 group 128 with BF16 scale for selected matrices; other tensors F32",
            "mixed-q4-q8": "encoder Transformer matrices Q4; decoder matrices and tied embedding/output Q8; other tensors F32",
        }[args.precision]),
        "checkpoint_repository": source_pin["repository"],
        "checkpoint_revision": source_pin["revision"],
        "checkpoint_sha256": checkpoint_pin["sha256"],
        "mel_filters_sha256": sha256_file(args.mel_filters),
        "tokenizer_sha256": (sha256_file(args.tokenizer) if args.tokenizer else None),
        "image_bytes": args.output.stat().st_size,
        "image_sha256": sha256_file(args.output),
        "tensor_count": len(entries),
        "q4_matrix_count": sum(entry.kind == KIND_Q4_GROUPED for entry in entries),
        "q4_matrix_bytes": sum(entry.byte_count for entry in entries if entry.kind == KIND_Q4_GROUPED),
        "q8_matrix_count": sum(entry.kind == KIND_Q8_GROUPED for entry in entries),
        "q8_matrix_bytes": sum(entry.byte_count for entry in entries if entry.kind == KIND_Q8_GROUPED),
        "payload_bytes": sum(entry.byte_count for entry in entries),
        "padding_and_directory_bytes": args.output.stat().st_size - sum(entry.byte_count for entry in entries),
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
