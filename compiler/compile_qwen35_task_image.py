#!/usr/bin/env python3
"""Compile the pinned Qwen3.5-0.8B checkpoint into a framework-free Q4 image.

Image v1 carries the full text graph for the prompt-defined runtime:
every text-layer matrix and the complete tied embedding/output table in
signed Q4 group-128, small tensors in float32. The vision tower and the
MTP head never enter the image. Tokenizer tables land in a later image
version; until then the runtime accepts token ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .q4_image import (
        align,
        q4_byte_count,
        q8_byte_count,
        quantize_q4_grouped,
        quantize_q8_grouped,
        sha256_file,
    )
    from .gemma4_task_reference import decode_float32
    from .safetensors_file import SafetensorsFile
except ImportError:
    from q4_image import (
        align,
        q4_byte_count,
        q8_byte_count,
        quantize_q4_grouped,
        quantize_q8_grouped,
        sha256_file,
    )
    from gemma4_task_reference import decode_float32
    from safetensors_file import SafetensorsFile


MAGIC = b"QW35TSK1"
VERSION = 2
GROUP_SIZE = 128
TEXT_PREFIX = "model.language_model."
HEADER = struct.Struct("<8s8I4Q")
DESCRIPTOR = struct.Struct("<96sII4IQQQ")
KIND_F32 = 1
KIND_U32 = 2
KIND_U8 = 3
KIND_Q4_GROUPED = 4
KIND_Q8_GROUPED = 5

LAYER_COUNT = 24
FULL_ATTENTION_INTERVAL = 4
HIDDEN_SIZE = 1024
VOCAB_ROWS = 248320

# Matrices small enough that float32 costs little and quantization noise
# on the DeltaNet gates is avoided.
F32_MATRIX_SUFFIXES = (
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.in_proj_a.weight",
)

# The DeltaNet projections are the Q4-sensitive class: with every class at
# Q4 the smoke decisions flip versus BF16, and the class ablation recovers
# them only when these matrices go to Q8 (recorded in the model record).
Q8_MATRIX_SUFFIXES = (
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
)


@dataclass
class Entry:
    name: str
    kind: int
    shape: tuple[int, ...]
    offset: int = 0
    byte_count: int = 0


def pack_descriptor(entry: Entry) -> bytes:
    encoded = entry.name.encode("utf-8")
    if len(encoded) >= 96:
        raise ValueError(f"tensor name is too long: {entry.name}")
    shape = list(entry.shape) + [0] * (4 - len(entry.shape))
    return DESCRIPTOR.pack(
        encoded, entry.kind, len(entry.shape), *shape, entry.offset, entry.byte_count, 0
    )


def classify(name: str, shape: tuple[int, ...]) -> int:
    if len(shape) == 2 and shape[1] % GROUP_SIZE == 0:
        if any(name.endswith(suffix) for suffix in F32_MATRIX_SUFFIXES):
            return KIND_F32
        if any(name.endswith(suffix) for suffix in Q8_MATRIX_SUFFIXES):
            return KIND_Q8_GROUPED
        return KIND_Q4_GROUPED
    return KIND_F32


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level mapping from raw bytes to printable unicode chars."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    mapped = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, mapped)}


def build_tokenizer_entries(tokenizer_path: Path) -> list[tuple[str, int, np.ndarray]]:
    """Pack vocab byte strings, ranked merges, and special tokens for the C runtime."""
    spec = json.loads(tokenizer_path.read_text())
    if spec["model"]["type"] != "BPE":
        raise ValueError("expected a BPE tokenizer")
    unicode_to_byte = {char: byte for byte, char in bytes_to_unicode().items()}

    def token_to_bytes(token: str) -> bytes:
        return bytes(unicode_to_byte[char] for char in token)

    vocab: dict[str, int] = spec["model"]["vocab"]
    added = spec.get("added_tokens", [])
    token_count = max(
        max(vocab.values()), max((t["id"] for t in added), default=0)
    ) + 1

    token_bytes: list[bytes] = [b""] * token_count
    for token, token_id in vocab.items():
        token_bytes[token_id] = token_to_bytes(token)
    for token in added:
        token_bytes[token["id"]] = token["content"].encode("utf-8")

    offsets = np.zeros(token_count + 1, dtype=np.uint32)
    for index, data in enumerate(token_bytes):
        offsets[index + 1] = offsets[index] + len(data)
    blob = np.frombuffer(b"".join(token_bytes), dtype=np.uint8)

    byte_tokens = np.full(256, 0xFFFFFFFF, dtype=np.uint32)
    for token_id, data in enumerate(token_bytes):
        if len(data) == 1:
            if byte_tokens[data[0]] == 0xFFFFFFFF:
                byte_tokens[data[0]] = token_id
    if int((byte_tokens == 0xFFFFFFFF).sum()) != 0:
        raise ValueError("byte-level vocabulary is missing single-byte tokens")

    merges = []
    for rank, merge in enumerate(spec["model"]["merges"]):
        left, right = merge.split(" ") if isinstance(merge, str) else merge
        merged = left + right
        merges.append((vocab[left], vocab[right], vocab[merged], rank))
    merges.sort(key=lambda entry: (entry[0], entry[1]))
    merge_array = np.asarray(merges, dtype=np.uint32)

    specials = sorted(
        (t for t in added if t.get("special")), key=lambda t: -len(t["content"])
    )
    special_ids = np.asarray([t["id"] for t in specials], dtype=np.uint32)
    special_strings = [t["content"].encode("utf-8") for t in specials]
    special_offsets = np.zeros(len(specials) + 1, dtype=np.uint32)
    for index, data in enumerate(special_strings):
        special_offsets[index + 1] = special_offsets[index] + len(data)
    special_blob = np.frombuffer(b"".join(special_strings), dtype=np.uint8)

    return [
        ("tokenizer.token_offsets", KIND_U32, offsets),
        ("tokenizer.token_bytes", KIND_U8, blob),
        ("tokenizer.byte_tokens", KIND_U32, byte_tokens),
        ("tokenizer.merges", KIND_U32, merge_array),
        ("tokenizer.special_ids", KIND_U32, special_ids),
        ("tokenizer.special_offsets", KIND_U32, special_offsets),
        ("tokenizer.special_bytes", KIND_U8, special_blob),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="trust the pinned size check only; skip hashing the 1.7 GB file",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/qwen3.5-0.8b/pins.json").read_text())
    checkpoint_name = "model.safetensors-00001-of-00001.safetensors"
    if args.checkpoint.stat().st_size != pins["model"]["model_bytes"]:
        raise ValueError("checkpoint size does not match the pinned revision")
    if not args.skip_checkpoint_hash:
        digest = sha256_file(args.checkpoint)
        if digest != pins["model"]["files"][checkpoint_name]:
            raise ValueError("checkpoint hash does not match the pinned revision")
    if sha256_file(args.config) != pins["model"]["files"]["config.json"]:
        raise ValueError("config does not match the pinned revision")
    if sha256_file(args.tokenizer) != pins["model"]["files"]["tokenizer.json"]:
        raise ValueError("tokenizer does not match the pinned revision")

    config = json.loads(args.config.read_text())["text_config"]
    if (
        config["num_hidden_layers"] != LAYER_COUNT
        or config["hidden_size"] != HIDDEN_SIZE
        or config["vocab_size"] != VOCAB_ROWS
        or config["full_attention_interval"] != FULL_ATTENTION_INTERVAL
    ):
        raise ValueError("config does not describe the expected Qwen3.5-0.8B text graph")

    started = time.monotonic()
    entries: list[Entry] = []
    with SafetensorsFile(args.checkpoint) as source:
        selected = []
        for name in sorted(source.tensors):
            if not name.startswith(TEXT_PREFIX):
                continue  # vision tower and MTP head never enter the image
            info = source.tensors[name]
            short = name[len(TEXT_PREFIX) :]
            shape = tuple(info.shape)
            if short.endswith("linear_attn.conv1d.weight"):
                shape = (shape[0], shape[2])  # squeeze the grouped-conv middle axis
            selected.append((short, name, shape))

        for short, _, shape in selected:
            kind = classify(short, shape)
            if kind == KIND_Q4_GROUPED:
                byte_count = q4_byte_count(shape)
            elif kind == KIND_Q8_GROUPED:
                byte_count = q8_byte_count(shape)
            else:
                byte_count = int(np.prod(shape)) * 4
            entries.append(Entry(short, kind, shape, byte_count=byte_count))

        raw_payloads: dict[str, bytes] = {}
        for name, kind, array in build_tokenizer_entries(args.tokenizer):
            dtype = "<u4" if kind == KIND_U32 else np.uint8
            payload = np.ascontiguousarray(array, dtype=dtype).tobytes()
            entries.append(Entry(name, kind, tuple(array.shape), byte_count=len(payload)))
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
                    LAYER_COUNT,
                    FULL_ATTENTION_INTERVAL,
                    VOCAB_ROWS,
                    HIDDEN_SIZE,
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

            matrix_total = sum(
                entry.kind in (KIND_Q4_GROUPED, KIND_Q8_GROUPED) for entry in entries
            )
            matrix_index = 0
            for (short, full_name, shape), entry in zip(selected, entries):
                info = source.tensors[full_name]
                array = decode_float32(source.read_bytes(full_name), info.dtype, info.shape)
                array = np.asarray(array, dtype=np.float32).reshape(entry.shape)
                output.seek(entry.offset)
                if entry.kind in (KIND_Q4_GROUPED, KIND_Q8_GROUPED):
                    quantize = (
                        quantize_q4_grouped
                        if entry.kind == KIND_Q4_GROUPED
                        else quantize_q8_grouped
                    )
                    payload = quantize(array)
                    if len(payload) != entry.byte_count:
                        raise AssertionError(f"quantized byte count mismatch for {short}")
                    output.write(payload)
                    matrix_index += 1
                    print(
                        f"quantized matrix={matrix_index}/{matrix_total} name={short} "
                        f"kind={'q4' if entry.kind == KIND_Q4_GROUPED else 'q8'} "
                        f"elapsed_seconds={time.monotonic() - started:.3f}",
                        flush=True,
                    )
                else:
                    output.write(array.astype("<f4").tobytes(order="C"))
            for entry in entries[len(selected) :]:
                output.seek(entry.offset)
                output.write(raw_payloads[entry.name])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)

    manifest = {
        "schema_version": 1,
        "format": MAGIC.decode(),
        "checkpoint_revision": pins["model"]["revision"],
        "checkpoint_sha256": pins["model"]["files"][checkpoint_name],
        "quantization": (
            "signed Q4 group 128 with BF16 scale; DeltaNet projections signed Q8 "
            "group 128; small tensors float32"
        ),
        "image_bytes": args.output.stat().st_size,
        "image_sha256": sha256_file(args.output),
        "tensor_count": len(entries),
        "q4_matrix_count": sum(entry.kind == KIND_Q4_GROUPED for entry in entries),
        "q8_matrix_count": sum(entry.kind == KIND_Q8_GROUPED for entry in entries),
        "quantized_matrix_bytes": sum(
            entry.byte_count
            for entry in entries
            if entry.kind in (KIND_Q4_GROUPED, KIND_Q8_GROUPED)
        ),
        "tokenizer_sha256": pins["model"]["files"]["tokenizer.json"],
        "tokenizer_tables": "vocab byte strings, ranked merges, byte map, special tokens",
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
