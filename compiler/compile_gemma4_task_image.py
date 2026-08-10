#!/usr/bin/env python3
"""Compile a bounded Gemma 4 task profile into a framework-free Q4 image."""

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
from tokenizers import Tokenizer

try:
    from .evaluate_gemma4_task_reference import canonical_prompt
    from .gemma4_task_reference import (
        MODEL_PREFIX,
        build_compiled_token_rows,
        decode_float32,
        load_rows,
        validate_text_config,
    )
    from .safetensors_file import SafetensorsFile
except ImportError:
    from evaluate_gemma4_task_reference import canonical_prompt
    from gemma4_task_reference import (
        MODEL_PREFIX,
        build_compiled_token_rows,
        decode_float32,
        load_rows,
        validate_text_config,
    )
    from safetensors_file import SafetensorsFile


MAGIC = b"G4TASK01"
VERSION = 1
GROUP_SIZE = 128
HEADER = struct.Struct("<8s8I4Q")
DESCRIPTOR = struct.Struct("<96sII4IQQQ")
KIND_F32 = 1
KIND_U32 = 2
KIND_U8 = 3
KIND_Q4_GROUPED = 4


@dataclass
class Entry:
    name: str
    kind: int
    shape: tuple[int, ...]
    offset: int = 0
    byte_count: int = 0


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype="<f4").view("<u4").copy()
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits >> np.uint32(16)).astype("<u2")


def q4_byte_count(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or shape[1] % GROUP_SIZE:
        raise ValueError(f"Q4 matrix shape must be [rows, multiple of {GROUP_SIZE}]: {shape}")
    return shape[0] * (shape[1] // GROUP_SIZE) * (2 + GROUP_SIZE // 2)


def quantize_q4_grouped(weights: np.ndarray) -> bytes:
    rows, columns = weights.shape
    if columns % GROUP_SIZE:
        raise ValueError(f"matrix width {columns} is not divisible by {GROUP_SIZE}")
    groups = columns // GROUP_SIZE
    blocks = np.asarray(weights, dtype=np.float32).reshape(rows, groups, GROUP_SIZE)
    minimum = np.min(blocks, axis=-1)
    maximum = np.max(blocks, axis=-1)
    scale = np.maximum(-minimum / np.float32(8.0), maximum / np.float32(7.0))
    scale[scale == 0] = np.float32(1.0)
    quantized = np.rint(blocks / scale[..., None]).clip(-8, 7).astype(np.int8)
    unsigned = (quantized.astype(np.int16) & 0xF).astype(np.uint8)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << np.uint8(4))
    records = np.empty((rows, groups, 2 + GROUP_SIZE // 2), dtype=np.uint8)
    records[..., :2] = float32_to_bf16(scale).view(np.uint8).reshape(rows, groups, 2)
    records[..., 2:] = packed
    return records.tobytes(order="C")


def encode_profile(profile: dict, tokenizer: Tokenizer):
    sequences = []
    for case in profile["cases"]:
        encoded = tokenizer.encode(
            canonical_prompt(profile["system_prompt"], case["observation"]),
            add_special_tokens=False,
        )
        sequences.append(encoded.ids)
    label_ids = []
    label_names = []
    for label in profile["labels"]:
        encoded = tokenizer.encode(label["text"], add_special_tokens=False)
        if len(encoded.ids) != 1:
            raise ValueError(f"label {label['name']} is not one token: {encoded.ids}")
        label_names.append(label["name"])
        label_ids.append(encoded.ids[0])
    expected = []
    for case in profile["cases"]:
        try:
            expected.append(label_names.index(case["expected"]))
        except ValueError as error:
            raise ValueError(f"unknown expected label in {case['id']}") from error
    return sequences, label_ids, expected


def raw_entry(name: str, kind: int, array: np.ndarray) -> tuple[Entry, bytes]:
    if kind == KIND_F32:
        payload = np.asarray(array, dtype="<f4").tobytes(order="C")
    elif kind == KIND_U32:
        payload = np.asarray(array, dtype="<u4").tobytes(order="C")
    elif kind == KIND_U8:
        payload = np.asarray(array, dtype=np.uint8).tobytes(order="C")
    else:
        raise ValueError("unsupported raw entry kind")
    return Entry(name, kind, tuple(array.shape), byte_count=len(payload)), payload


def pack_descriptor(entry: Entry) -> bytes:
    encoded = entry.name.encode("utf-8")
    if len(encoded) >= 96:
        raise ValueError(f"tensor name is too long: {entry.name}")
    shape = list(entry.shape) + [0] * (4 - len(entry.shape))
    return DESCRIPTOR.pack(
        encoded,
        entry.kind,
        len(entry.shape),
        *shape,
        entry.offset,
        entry.byte_count,
        0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/gemma-4-e2b/pins.json").read_text())
    if args.checkpoint.stat().st_size != pins["model"]["model_bytes"]:
        raise ValueError("checkpoint size does not match the pinned revision")
    if sha256_file(args.config) != pins["model"]["files"]["config.json"]:
        raise ValueError("config does not match the pinned revision")
    if sha256_file(args.tokenizer) != pins["model"]["files"]["tokenizer.json"]:
        raise ValueError("tokenizer does not match the pinned revision")

    text = validate_text_config(json.loads(args.config.read_text()))
    profile = json.loads(args.profile.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sequences, label_ids, expected = encode_profile(profile, tokenizer)
    unique_token_ids = sorted(set(label_ids).union(*(set(sequence) for sequence in sequences)))
    token_to_row = {token: index for index, token in enumerate(unique_token_ids)}
    case_rows = [[token_to_row[token] for token in sequence] for sequence in sequences]
    case_offsets = [0]
    for rows in case_rows:
        case_offsets.append(case_offsets[-1] + len(rows))

    entries: list[Entry] = []
    raw_payloads: dict[str, bytes] = {}
    started = time.monotonic()
    with SafetensorsFile(args.checkpoint) as source:
        layer_names = sorted(
            (
                name
                for name in source.tensors
                if name.startswith(f"{MODEL_PREFIX}.layers.")
                and not (
                    int(name.split(".layers.", 1)[1].split(".", 1)[0]) >= 15
                    and name.endswith(
                        (
                            ".self_attn.k_proj.weight",
                            ".self_attn.v_proj.weight",
                            ".self_attn.k_norm.weight",
                        )
                    )
                )
            ),
            key=lambda name: (
                int(name.split(".layers.", 1)[1].split(".", 1)[0]),
                name,
            ),
        )
        for name in layer_names:
            info = source.tensors[name]
            if len(info.shape) == 2:
                entries.append(Entry(name, KIND_Q4_GROUPED, info.shape, byte_count=q4_byte_count(info.shape)))
            else:
                array = decode_float32(source.read_bytes(name), info.dtype, info.shape)
                entry, payload = raw_entry(name, KIND_F32, array)
                entries.append(entry)
                raw_payloads[name] = payload

        final_norm_name = f"{MODEL_PREFIX}.norm.weight"
        final_norm = decode_float32(
            source.read_bytes(final_norm_name),
            source.tensors[final_norm_name].dtype,
            source.tensors[final_norm_name].shape,
        )
        entry, payload = raw_entry(final_norm_name, KIND_F32, final_norm)
        entries.append(entry)
        raw_payloads[entry.name] = payload

        embeddings, folded_ple = build_compiled_token_rows(source, text, unique_token_ids)
        label_rows = load_rows(source, f"{MODEL_PREFIX}.embed_tokens.weight", label_ids)

        arrays = [
            ("profile.token_ids", KIND_U32, np.asarray(unique_token_ids, dtype=np.uint32)),
            ("profile.embeddings", KIND_F32, embeddings),
            ("profile.folded_ple", KIND_F32, folded_ple),
            ("profile.label_token_ids", KIND_U32, np.asarray(label_ids, dtype=np.uint32)),
            ("profile.label_weights", KIND_F32, label_rows),
            ("profile.case_offsets", KIND_U32, np.asarray(case_offsets, dtype=np.uint32)),
            (
                "profile.case_token_rows",
                KIND_U32,
                np.asarray([row for rows in case_rows for row in rows], dtype=np.uint32),
            ),
            ("profile.case_expected", KIND_U32, np.asarray(expected, dtype=np.uint32)),
            (
                "profile.source_sha256",
                KIND_U8,
                np.frombuffer(hashlib.sha256(args.profile.read_bytes()).digest(), dtype=np.uint8),
            ),
        ]
        for name, kind, array in arrays:
            entry, payload = raw_entry(name, kind, array)
            entries.append(entry)
            raw_payloads[name] = payload

        directory_offset = HEADER.size
        data_offset = align(directory_offset + len(entries) * DESCRIPTOR.size)
        cursor = data_offset
        for entry in entries:
            entry.offset = cursor
            cursor = align(cursor + entry.byte_count)
        file_bytes = cursor

        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w+b") as output:
            output.truncate(file_bytes)
            output.seek(0)
            output.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    len(entries),
                    len(unique_token_ids),
                    len(sequences),
                    len(label_ids),
                    max(len(sequence) for sequence in sequences),
                    GROUP_SIZE,
                    0,
                    directory_offset,
                    data_offset,
                    file_bytes,
                    0,
                )
            )
            output.seek(directory_offset)
            for entry in entries:
                output.write(pack_descriptor(entry))

            matrix_index = 0
            matrix_count = sum(entry.kind == KIND_Q4_GROUPED for entry in entries)
            for entry in entries:
                output.seek(entry.offset)
                if entry.kind == KIND_Q4_GROUPED:
                    info = source.tensors[entry.name]
                    weights = decode_float32(source.read_bytes(entry.name), info.dtype, info.shape)
                    payload = quantize_q4_grouped(weights)
                    if len(payload) != entry.byte_count:
                        raise AssertionError(f"Q4 byte count mismatch for {entry.name}")
                    output.write(payload)
                    matrix_index += 1
                    elapsed = time.monotonic() - started
                    print(
                        f"quantized matrix={matrix_index}/{matrix_count} name={entry.name} elapsed_seconds={elapsed:.3f}",
                        flush=True,
                    )
                else:
                    output.write(raw_payloads[entry.name])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)

    image_sha256 = sha256_file(args.output)
    manifest = {
        "schema_version": 1,
        "format": "G4TASK01",
        "profile_id": profile["profile_id"],
        "checkpoint_revision": pins["model"]["revision"],
        "checkpoint_sha256": pins["model"]["files"]["model.safetensors"],
        "tokenizer_sha256": pins["model"]["files"]["tokenizer.json"],
        "quantization": "signed Q4, group size 128, BF16 scale",
        "image_bytes": args.output.stat().st_size,
        "image_sha256": image_sha256,
        "tensor_count": len(entries),
        "matrix_count": sum(entry.kind == KIND_Q4_GROUPED for entry in entries),
        "quantized_matrix_bytes": sum(
            entry.byte_count for entry in entries if entry.kind == KIND_Q4_GROUPED
        ),
        "non_matrix_and_padding_bytes": args.output.stat().st_size
        - sum(entry.byte_count for entry in entries if entry.kind == KIND_Q4_GROUPED),
        "compiled_token_count": len(unique_token_ids),
        "case_count": len(sequences),
        "maximum_prompt_tokens": max(len(sequence) for sequence in sequences),
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
