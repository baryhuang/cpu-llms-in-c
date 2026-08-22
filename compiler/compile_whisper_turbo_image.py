#!/usr/bin/env python3
"""Compile the pinned Whisper large-v3-turbo graph into an mmap-ready image.

The pinned checkpoint stores F16 weights; the F32 image upcasts them exactly
(F16 -> F32 is lossless), so it is the correctness baseline. Q4/Q8/Q5 images
are derived artifacts that must be evaluated against that baseline. Every
image carries the full graph: encoder, decoder, byte-level tokenizer tables,
and the generation suppress list from the pinned generation config.
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
    from .q4_image import (align, q4_byte_count, q5_byte_count, q8_byte_count,
                           quantize_q4_grouped, quantize_q5_grouped,
                           quantize_q8_grouped, sha256_file)
    from .safetensors_file import SafetensorsFile
except ImportError:
    from q4_image import (align, q4_byte_count, q5_byte_count, q8_byte_count,
                          quantize_q4_grouped, quantize_q5_grouped,
                          quantize_q8_grouped, sha256_file)
    from safetensors_file import SafetensorsFile


MAGIC = b"WHTRBO01"
KIND_F32 = 1
KIND_U32 = 2
KIND_U8 = 3
KIND_Q4_GROUPED = 4
KIND_Q8_GROUPED = 5
KIND_Q5_GROUPED = 6
HEADER = struct.Struct("<8s8I4Q")
DESCRIPTOR = struct.Struct("<96sII4IQQQ")
MODEL_PREFIX = "model."
VOCABULARY = 51866
VERSION_BY_PRECISION = {"f32": 1, "q4": 2, "q8": 3, "q5": 4}
KIND_BY_PRECISION = {"q4": KIND_Q4_GROUPED, "q8": KIND_Q8_GROUPED,
                     "q5": KIND_Q5_GROUPED}

# large-v3 multilingual control tokens, asserted against the pinned tokenizer
# so the C runtime's hardcoded ids can never drift from the packed image.
SPECIAL_ASSERTS = {
    "<|endoftext|>": 50257,
    "<|startoftranscript|>": 50258,
    "<|en|>": 50259,
    "<|transcribe|>": 50360,
    "<|notimestamps|>": 50364,
}
TIMESTAMP_BEGIN = 50365


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


def matrix_kind(name: str, shape: tuple[int, ...], precision: str) -> int:
    if (precision == "f32" or len(shape) != 2 or not name.endswith(".weight") or
            shape[1] % 128 != 0):
        return KIND_F32
    if (name.startswith("model.encoder.layers.") or
            name.startswith("model.decoder.layers.") or
            name == "model.decoder.embed_tokens.weight"):
        return KIND_BY_PRECISION[precision]
    return KIND_F32


def quantized_byte_count(kind: int, shape: tuple[int, ...]) -> int:
    if kind == KIND_Q4_GROUPED:
        return q4_byte_count(shape)
    if kind == KIND_Q8_GROUPED:
        return q8_byte_count(shape)
    return q5_byte_count(shape)


def tensor_f32(source: SafetensorsFile, name: str) -> np.ndarray:
    info = source.tensors[name]
    raw = source.read_bytes(name)
    if info.dtype == "F32":
        array = np.frombuffer(raw, dtype="<f4")
    elif info.dtype == "F16":
        array = np.frombuffer(raw, dtype="<f2").astype("<f4")
    else:
        raise ValueError(f"unsupported dtype for {name}: {info.dtype}")
    return np.ascontiguousarray(array.reshape(info.shape))


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


def tokenizer_payloads(tokenizer_path: Path, generation_path: Path) -> list[Entry]:
    spec = json.loads(tokenizer_path.read_text())
    if spec["model"]["type"] != "BPE":
        raise ValueError("expected Whisper BPE tokenizer")
    inverse = {character: byte for byte, character in bytes_to_unicode().items()}
    pieces: list[bytes] = [b""] * VOCABULARY
    special = np.zeros(VOCABULARY, dtype=np.uint8)
    for piece, token_id in spec["model"]["vocab"].items():
        pieces[token_id] = bytes(inverse[character] for character in piece)
    added_ids: dict[str, int] = {}
    for token in spec.get("added_tokens", []):
        token_id = token["id"]
        added_ids[token["content"]] = token_id
        if token_id < VOCABULARY:
            pieces[token_id] = token["content"].encode("utf-8")
            special[token_id] = 1 if token.get("special") else 0
    for content, expected in SPECIAL_ASSERTS.items():
        if added_ids.get(content) != expected:
            raise ValueError(
                f"tokenizer drift: {content} is {added_ids.get(content)!r}, "
                f"expected {expected}")
    offsets = np.zeros(VOCABULARY + 1, dtype="<u4")
    for index, piece in enumerate(pieces):
        offsets[index + 1] = offsets[index] + len(piece)
    blob = b"".join(pieces)

    generation = json.loads(generation_path.read_text())
    suppress = sorted(int(token) for token in generation["suppress_tokens"])
    if any(token < 0 or token >= VOCABULARY for token in suppress):
        raise ValueError("suppress token outside the vocabulary")
    if sorted(generation.get("begin_suppress_tokens", [])) != [220, 50257]:
        raise ValueError("unexpected begin_suppress_tokens in generation config")
    suppress_payload = np.asarray(suppress, dtype="<u4")

    return [
        Entry("tokenizer.offsets", (VOCABULARY + 1,), kind=KIND_U32,
              payload=offsets.tobytes()),
        Entry("tokenizer.bytes", (len(blob),), kind=KIND_U8, payload=blob),
        Entry("tokenizer.special", (VOCABULARY,), kind=KIND_U8,
              payload=special.tobytes()),
        Entry("tokenizer.suppress", (len(suppress),), kind=KIND_U32,
              payload=suppress_payload.tobytes()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mel-filters", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--precision", choices=("f32", "q4", "q8", "q5"),
                        default="f32")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads(
        (repository / "models/whisper-large-v3-turbo/pins.json").read_text())
    source_pin = pins["model"]["compiler_checkpoint"]
    pinned_files = {
        "model.safetensors": args.checkpoint,
        "config.json": args.config,
        "tokenizer.json": args.tokenizer,
        "generation_config.json": args.generation_config,
    }
    for pin_name, path in pinned_files.items():
        pin = source_pin["files"][pin_name]
        if path.stat().st_size != pin["bytes"]:
            raise ValueError(f"{pin_name} byte count does not match the pin")
        if pin_name == "model.safetensors" and args.skip_checkpoint_hash:
            continue
        if sha256_file(path) != pin["sha256"]:
            raise ValueError(f"{pin_name} digest does not match the pin")
    if sha256_file(args.mel_filters) != pins["model"]["reference_files"]["whisper/assets/mel_filters.npz"]:
        raise ValueError("mel filter file does not match the pin")

    config = json.loads(args.config.read_text())
    required = {
        "d_model": 1280,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "encoder_layers": 32,
        "decoder_attention_heads": 20,
        "decoder_ffn_dim": 5120,
        "decoder_layers": 4,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 128,
        "vocab_size": VOCABULARY,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"unexpected config value: {key}={config.get(key)!r}")

    with np.load(args.mel_filters) as archive:
        mel = np.ascontiguousarray(archive["mel_128"], dtype="<f4")
    if mel.shape != (128, 201):
        raise ValueError(f"unexpected mel_128 shape: {mel.shape}")

    started = time.monotonic()
    entries = [Entry("frontend.mel_128", tuple(mel.shape), payload=mel.tobytes())]
    entries.extend(tokenizer_payloads(args.tokenizer, args.generation_config))
    with SafetensorsFile(args.checkpoint) as source:
        for full_name in sorted(source.tensors):
            if not (full_name.startswith("model.encoder.") or
                    full_name.startswith("model.decoder.")):
                continue
            info = source.tensors[full_name]
            if info.dtype not in ("F16", "F32"):
                raise ValueError(f"unsupported dtype: {full_name} is {info.dtype}")
            shape = tuple(info.shape)
            kind = matrix_kind(full_name, shape, args.precision)
            if kind == KIND_F32:
                byte_count = info.element_count * 4
            else:
                byte_count = quantized_byte_count(kind, shape)
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
                MAGIC, VERSION_BY_PRECISION[args.precision],
                len(entries), 32, 128, 1280, 20, 5120, 1500,
                directory_offset, data_offset, file_bytes, 0,
            ))
            output.seek(directory_offset)
            for entry in entries:
                output.write(descriptor_bytes(entry))
            for index, entry in enumerate(entries, 1):
                if entry.payload is not None:
                    payload = entry.payload
                elif entry.kind == KIND_Q4_GROUPED:
                    payload = quantize_q4_grouped(tensor_f32(source, entry.source_name))
                elif entry.kind == KIND_Q8_GROUPED:
                    payload = quantize_q8_grouped(tensor_f32(source, entry.source_name))
                elif entry.kind == KIND_Q5_GROUPED:
                    payload = quantize_q5_grouped(tensor_f32(source, entry.source_name))
                else:
                    payload = tensor_f32(source, entry.source_name).tobytes(order="C")
                if len(payload) != entry.byte_count:
                    raise AssertionError(f"payload length mismatch: {entry.name}")
                output.seek(entry.offset)
                output.write(payload)
                if index % 64 == 0 or index == len(entries):
                    print(f"copied tensors={index}/{len(entries)} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)

    manifest = {
        "schema_version": 1,
        "format": MAGIC.decode(),
        "graph": "full",
        "precision": ({
            "f32": "float32 exact upcast of the pinned F16 checkpoint (lossless)",
            "q4": "signed Q4 group 128 with BF16 scale for Transformer matrices and the tied embedding; other tensors F32",
            "q8": "signed Q8 group 128 with BF16 scale for Transformer matrices and the tied embedding; other tensors F32",
            "q5": "signed Q5 group 128 (Q4 nibbles + high-bit plane) with BF16 scale for Transformer matrices and the tied embedding; other tensors F32",
        }[args.precision]),
        "checkpoint_repository": source_pin["repository"],
        "checkpoint_revision": source_pin["revision"],
        "checkpoint_sha256": source_pin["files"]["model.safetensors"]["sha256"],
        "mel_filters_sha256": sha256_file(args.mel_filters),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "generation_config_sha256": sha256_file(args.generation_config),
        "image_bytes": args.output.stat().st_size,
        "image_sha256": sha256_file(args.output),
        "tensor_count": len(entries),
        "quantized_matrix_count": sum(entry.kind in (KIND_Q4_GROUPED, KIND_Q8_GROUPED, KIND_Q5_GROUPED) for entry in entries),
        "quantized_matrix_bytes": sum(entry.byte_count for entry in entries if entry.kind in (KIND_Q4_GROUPED, KIND_Q8_GROUPED, KIND_Q5_GROUPED)),
        "payload_bytes": sum(entry.byte_count for entry in entries),
        "padding_and_directory_bytes": args.output.stat().st_size - sum(entry.byte_count for entry in entries),
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
