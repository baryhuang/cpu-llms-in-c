#!/usr/bin/env python3
"""Pack the pinned MiniMind-O byte-level BPE for the native C runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"MMOTOK1\0"
VERSION = 1
HEADER_BYTES = 4096
EXPECTED_SHA256 = "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668"
DIRECTORY = struct.Struct("<QII")
MERGE = struct.Struct("<IIII")


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) & -alignment


def byte_unicode() -> tuple[dict[int, str], dict[str, int]]:
    direct = list(range(ord("!"), ord("~") + 1))
    direct += list(range(ord("¡"), ord("¬") + 1))
    direct += list(range(ord("®"), ord("ÿ") + 1))
    mapped = direct[:]
    extra = 0
    for byte in range(256):
        if byte not in direct:
            direct.append(byte)
            mapped.append(256 + extra)
            extra += 1
    forward = {byte: chr(codepoint) for byte, codepoint in zip(direct, mapped)}
    return forward, {text: byte for byte, text in forward.items()}


def decode_bytelevel(token: str, reverse: dict[str, int]) -> bytes:
    try:
        return bytes(reverse[character] for character in token)
    except KeyError as exc:
        raise ValueError(f"token is not byte-level decodable: {token!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()

    source = args.tokenizer.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != EXPECTED_SHA256 and not args.allow_unpinned:
        raise SystemExit(f"tokenizer SHA-256 mismatch: {source_sha}")
    document = json.loads(source)
    if document.get("normalizer") is not None:
        raise SystemExit("native tokenizer requires the pinned null normalizer")
    pre = document.get("pre_tokenizer", {})
    if pre.get("type") != "ByteLevel" or pre.get("add_prefix_space"):
        raise SystemExit("unsupported MiniMind-O pre-tokenizer")

    vocab_by_text = document["model"]["vocab"]
    vocab_size = len(vocab_by_text)
    tokens: list[str | None] = [None] * vocab_size
    for text, token_id in vocab_by_text.items():
        if not 0 <= token_id < vocab_size or tokens[token_id] is not None:
            raise SystemExit("invalid tokenizer vocabulary IDs")
        tokens[token_id] = text
    if any(token is None for token in tokens):
        raise SystemExit("tokenizer vocabulary IDs are not dense")

    forward, reverse = byte_unicode()
    byte_ids = []
    for byte in range(256):
        try:
            byte_ids.append(vocab_by_text[forward[byte]])
        except KeyError as exc:
            raise SystemExit(f"missing byte token {byte}") from exc

    added_ids = {int(item["id"]) for item in document["added_tokens"]}
    decoded: list[bytes] = []
    flags: list[int] = []
    for token_id, token in enumerate(tokens):
        assert token is not None
        decoded.append(token.encode() if token_id in added_ids else decode_bytelevel(token, reverse))
        flags.append(1 if token_id in added_ids else 0)

    merges = []
    for rank, pair in enumerate(document["model"]["merges"]):
        if isinstance(pair, str):
            left, right = pair.split(" ", 1)
        else:
            left, right = pair
        result = left + right
        try:
            merges.append((vocab_by_text[left], vocab_by_text[right],
                           vocab_by_text[result], rank))
        except KeyError as exc:
            raise SystemExit(f"invalid merge at rank {rank}: {pair!r}") from exc
    merges.sort(key=lambda item: (item[0], item[1]))

    directory_offset = HEADER_BYTES
    directory_bytes = vocab_size * DIRECTORY.size
    blob_offset = align(directory_offset + directory_bytes)
    blob = bytearray()
    entries = []
    for value, flag in zip(decoded, flags):
        entries.append((len(blob), len(value), flag))
        blob.extend(value)
    merges_offset = align(blob_offset + len(blob))
    file_bytes = merges_offset + len(merges) * MERGE.size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        header = struct.pack(
            "<8s8I6Q64s256I",
            MAGIC, VERSION, HEADER_BYTES, vocab_size, len(merges), len(added_ids),
            DIRECTORY.size, MERGE.size, 0,
            directory_offset, directory_bytes, blob_offset, len(blob),
            merges_offset, file_bytes, source_sha.encode(), *byte_ids,
        )
        if len(header) > HEADER_BYTES:
            raise SystemExit("tokenizer header exceeds one page")
        stream.write(header)
        stream.write(b"\0" * (HEADER_BYTES - len(header)))
        for entry in entries:
            stream.write(DIRECTORY.pack(*entry))
        stream.write(b"\0" * (blob_offset - stream.tell()))
        stream.write(blob)
        stream.write(b"\0" * (merges_offset - stream.tell()))
        for merge in merges:
            stream.write(MERGE.pack(*merge))
    image_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "bytes": file_bytes,
                      "source_sha256": source_sha, "image_sha256": image_sha,
                      "vocab": vocab_size, "merges": len(merges)}))


if __name__ == "__main__":
    main()
