#!/usr/bin/env python3
"""Create a sparse safetensors file by fetching only selected HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "BF16": 2,
    "F16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def fetch_range(
    url: str, start: int, end: int, timeout: int = 300, attempts: int = 4
) -> tuple[bytes, int]:
    last_error = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or match is None:
                    raise ValueError(f"server did not honor range {start}:{end}")
                actual_start, actual_end, total_size = map(int, match.groups())
                if actual_start != start or actual_end + 1 != end or len(payload) != end - start:
                    raise ValueError(f"incorrect response range for {start}:{end}")
                return payload, total_size
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch range {start}:{end}") from last_error


def read_remote_header(url: str) -> tuple[bytes, dict, int, int]:
    prefix, total_size = fetch_range(url, 0, 8)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size == 0 or header_size > 256 * 1024 * 1024:
        raise ValueError(f"invalid remote safetensors header size: {header_size}")
    encoded, confirmed_size = fetch_range(url, 8, 8 + header_size)
    if confirmed_size != total_size:
        raise ValueError("remote file size changed while reading its header")
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid remote safetensors header") from error
    return prefix + encoded, header, total_size, 8 + header_size


def tensor_range(header: dict, data_start: int, name: str) -> tuple[int, int]:
    record = header.get(name)
    if not isinstance(record, dict):
        raise KeyError(f"tensor not found: {name}")
    start, end = record["data_offsets"]
    return data_start + start, data_start + end


def tensor_slice_range(
    header: dict, data_start: int, name: str, element_offset: int, element_count: int
) -> tuple[int, int]:
    record = header.get(name)
    if not isinstance(record, dict):
        raise KeyError(f"tensor not found: {name}")
    dtype = record["dtype"]
    shape = record["shape"]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype for {name}: {dtype}")
    total_elements = math.prod(shape)
    if (
        element_offset < 0
        or element_count < 0
        or element_offset > total_elements
        or element_count > total_elements - element_offset
    ):
        raise ValueError(f"element slice is outside {name}")
    tensor_start = data_start + record["data_offsets"][0]
    byte_width = DTYPE_BYTES[dtype]
    return (
        tensor_start + element_offset * byte_width,
        tensor_start + (element_offset + element_count) * byte_width,
    )


def coalesce(ranges: list[tuple[int, int]], maximum_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start == end:
            continue
        if merged and start <= merged[-1][1] + maximum_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def split_ranges(ranges: list[tuple[int, int]], maximum_bytes: int) -> list[tuple[int, int]]:
    if maximum_bytes <= 0:
        raise ValueError("maximum range size must be positive")
    result = []
    for start, end in ranges:
        while start < end:
            chunk_end = min(start + maximum_bytes, end)
            result.append((start, chunk_end))
            start = chunk_end
    return result


def parse_slice(value: str) -> tuple[str, int, int]:
    try:
        name, offset, count = value.rsplit(":", 2)
        return name, int(offset), int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("slice must be TENSOR:ELEMENT_OFFSET:ELEMENT_COUNT") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--tensor-prefix", action="append", default=[])
    parser.add_argument("--slice", action="append", default=[], type=parse_slice)
    parser.add_argument("--maximum-gap", type=int)
    parser.add_argument("--chunk-bytes", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text()) if args.spec is not None else {}
    url = args.url or spec.get("url")
    if url is None:
        parser.error("one of --url or --spec with a URL is required")
    tensor_names = [*spec.get("tensors", []), *args.tensor]
    tensor_prefixes = [*spec.get("tensor_prefixes", []), *args.tensor_prefix]
    slices = [
        (item["tensor"], item["element_offset"], item["element_count"])
        for item in spec.get("slices", [])
    ] + args.slice
    maximum_gap = args.maximum_gap if args.maximum_gap is not None else spec.get("maximum_gap", 64 * 1024)
    chunk_bytes = args.chunk_bytes if args.chunk_bytes is not None else spec.get("chunk_bytes", 4 * 1024 * 1024)
    workers = args.workers if args.workers is not None else spec.get("workers", 12)

    header_bytes, header, total_size, data_start = read_remote_header(url)
    selected_names = set(tensor_names)
    for prefix in tensor_prefixes:
        selected_names.update(name for name in header if name.startswith(prefix))
    ranges = [(0, data_start)]
    ranges.extend(tensor_range(header, data_start, name) for name in sorted(selected_names))
    ranges.extend(
        tensor_slice_range(header, data_start, name, offset, count)
        for name, offset, count in slices
    )
    merged = coalesce(ranges, maximum_gap)
    remote_ranges = split_ranges(
        [(start, end) for start, end in merged if not (start == 0 and end == data_start)],
        chunk_bytes,
    )

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    manifest_path = args.output.with_suffix(args.output.suffix + ".ranges.json")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with temporary.open("w+b") as output:
        output.truncate(total_size)
        output.seek(0)
        output.write(header_bytes)
        records.append(
            {
                "start": 0,
                "end": data_start,
                "bytes": data_start,
                "sha256": hashlib.sha256(header_bytes).hexdigest(),
            }
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_range, url, start, end): (start, end)
                for start, end in remote_ranges
            }
            completed_bytes = 0
            total_remote_bytes = sum(end - start for start, end in remote_ranges)
            for future in concurrent.futures.as_completed(futures):
                start, end = futures[future]
                payload, confirmed_size = future.result()
                if confirmed_size != total_size:
                    raise ValueError("remote file size changed during range fetch")
                output.seek(start)
                output.write(payload)
                records.append(
                    {
                        "start": start,
                        "end": end,
                        "bytes": end - start,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                completed_bytes += end - start
                print(
                    f"fetched {completed_bytes}/{total_remote_bytes} bytes",
                    file=sys.stderr,
                    flush=True,
                )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.output)

    manifest = {
        "schema_version": 1,
        "url": url,
        "logical_bytes": total_size,
        "fetched_bytes": sum(record["bytes"] for record in records),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "tensors": sorted(selected_names),
        "slices": [
            {"tensor": name, "element_offset": offset, "element_count": count}
            for name, offset, count in slices
        ],
        "ranges": sorted(records, key=lambda record: record["start"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
