#!/usr/bin/env python3
"""Build an RK3588 MatMul-API W4A16 weight package.

The RKNN graph compiler does not expose W4A16 for RK3588.  RKNPU2's MatMul
API does expose FP16 x INT4 with per-group scales, so the native target uses a
small, mmap-friendly container instead of pretending that an incompatible
RK3576 .rknn graph can be deployed on RK3588.

Python is a build-time dependency only.  The target runtime reads this format
directly from C++ and does not embed Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from safetensors import safe_open


MAGIC = b"LLMCW4A\0"
VERSION = 2
ALIGNMENT = 64
GROUP_SIZE = 32
ENCODING_FP16 = 0
ENCODING_W4A16_GROUP = 1
HEADER = struct.Struct("<8sIIQ")
ENTRY = struct.Struct("<IBBH4IIIQQQQ")

QUANTIZED_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".out_proj.weight",
    ".fc1.weight",
    ".fc2.weight",
    "proj_out.weight",
)

QUANTIZED_CONVOLUTIONS = (
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
)


@dataclass
class TensorEntry:
    name: str
    encoding: int
    dims: tuple[int, ...]
    group_size: int
    data_offset: int
    data_size: int
    scale_offset: int
    scale_size: int


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_aligned(stream: BinaryIO, payload: bytes) -> tuple[int, int]:
    offset = align(stream.tell())
    if offset > stream.tell():
        stream.write(b"\0" * (offset - stream.tell()))
    stream.write(payload)
    return offset, len(payload)


def selected(name: str, scope: str) -> bool:
    if scope == "encoder":
        return name.startswith("model.encoder.") or (
            name.startswith("model.decoder.layers.")
            and ".encoder_attn." in name
            and (".k_proj." in name or ".v_proj." in name)
        )
    if scope == "decoder":
        return name.startswith("model.decoder.")
    return name.startswith("model.encoder.") or name.startswith("model.decoder.")


def quantized(name: str, tensor: torch.Tensor) -> bool:
    return (tensor.ndim == 2 and name.endswith(QUANTIZED_SUFFIXES)) or (
        name in QUANTIZED_CONVOLUTIONS
    )


def as_matmul_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name in QUANTIZED_CONVOLUTIONS:
        return tensor.reshape(tensor.shape[0], -1)
    if name == "model.proj_out.weight":
        padded_n = align(tensor.shape[0], 64)
        if padded_n != tensor.shape[0]:
            padded = torch.zeros((padded_n, tensor.shape[1]), dtype=tensor.dtype)
            padded[: tensor.shape[0]] = tensor
            return padded
    return tensor


def fp16_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.to(dtype=torch.float16).contiguous().numpy().tobytes()


def quantize_w4a16_group32(
    tensor: torch.Tensor,
) -> tuple[bytes, bytes, float, float]:
    # PyTorch Linear stores [N, K], while rknn_matmul consumes B=[K, N].
    weight = tensor.to(dtype=torch.float32)
    n, k = weight.shape
    if k % GROUP_SIZE:
        raise ValueError(f"K={k} is not divisible by group size {GROUP_SIZE}")

    grouped = weight.view(n, k // GROUP_SIZE, GROUP_SIZE)
    scales = grouped.abs().amax(dim=2) / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q_nk = torch.round(grouped / scales.unsqueeze(2)).clamp(-8, 7).to(torch.int8)

    dequantized = q_nk.to(torch.float32) * scales.unsqueeze(2)
    error = dequantized - grouped
    sum_squared_error = float(torch.sum(error * error).item())
    max_abs_error = float(torch.max(torch.abs(error)).item())

    q_kn = q_nk.view(n, k).transpose(0, 1).contiguous().view(-1)
    if q_kn.numel() % 2:
        q_kn = torch.cat((q_kn, torch.zeros(1, dtype=torch.int8)))
    low = torch.bitwise_and(q_kn[0::2].to(torch.int16), 0x0F)
    high = torch.bitwise_left_shift(
        torch.bitwise_and(q_kn[1::2].to(torch.int16), 0x0F), 4
    )
    packed = torch.bitwise_or(low, high).to(torch.uint8).numpy().tobytes()
    # RKNPU2 consumes per-group scales as [K / group_size, N], not the
    # [N, K / group_size] order used naturally while quantizing PyTorch's
    # Linear [N, K] weights.  Version 1 packages used the latter order and
    # must not be accepted by the native runtime.
    scale_bytes = scales.transpose(0, 1).contiguous().numpy().tobytes()
    return packed, scale_bytes, sum_squared_error, max_abs_error


def validate_package(path: Path) -> None:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        magic, version, tensor_count, payload_start = HEADER.unpack(
            stream.read(HEADER.size)
        )
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{path}: invalid package header")
        if payload_start % ALIGNMENT or payload_start > file_size:
            raise ValueError(f"{path}: invalid payload offset {payload_start}")

        names: set[str] = set()
        for _ in range(tensor_count):
            fields = ENTRY.unpack(stream.read(ENTRY.size))
            name_len, encoding, ndim = fields[:3]
            dims = tuple(fields[4:8])[:ndim]
            group_size = fields[8]
            data_offset, data_size, scale_offset, scale_size = fields[10:14]
            name = stream.read(name_len).decode("utf-8")
            if name in names:
                raise ValueError(f"{path}: duplicate tensor {name}")
            names.add(name)
            elements = 1
            for dim in dims:
                elements *= dim
            if data_offset < payload_start or data_offset + data_size > file_size:
                raise ValueError(f"{path}: data range out of bounds for {name}")
            if encoding == ENCODING_FP16:
                if data_size != elements * 2 or scale_size != 0:
                    raise ValueError(f"{path}: invalid FP16 payload for {name}")
            elif encoding == ENCODING_W4A16_GROUP:
                expected_scales = dims[0] * (dims[1] // group_size) * 4
                if (
                    ndim != 2
                    or group_size != GROUP_SIZE
                    or data_size != (elements + 1) // 2
                    or scale_size != expected_scales
                    or scale_offset < payload_start
                    or scale_offset + scale_size > file_size
                ):
                    raise ValueError(f"{path}: invalid W4A16 payload for {name}")
            else:
                raise ValueError(f"{path}: invalid encoding {encoding} for {name}")
        if stream.tell() > payload_start:
            raise ValueError(f"{path}: tensor table overlaps payload")


def build(checkpoint: Path, output: Path, scope: str) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    entries: list[TensorEntry] = []
    quantized_elements = 0
    fp16_elements = 0
    sum_squared_error = 0.0
    max_abs_error = 0.0

    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".payload", dir=output.parent, delete=False
    ) as payload_stream:
        payload_path = Path(payload_stream.name)
        try:
            with safe_open(checkpoint, framework="pt", device="cpu") as source:
                names = sorted(name for name in source.keys() if selected(name, scope))
                if scope in ("decoder", "all"):
                    names.append("model.proj_out.weight")
                for index, name in enumerate(names, 1):
                    source_name = (
                        "model.decoder.embed_tokens.weight"
                        if name == "model.proj_out.weight"
                        else name
                    )
                    tensor = source.get_tensor(source_name)
                    tensor = as_matmul_weight(name, tensor)
                    dims = tuple(int(value) for value in tensor.shape)
                    if len(dims) > 4:
                        raise ValueError(f"{name}: at most 4 dimensions are supported")

                    if quantized(name, tensor):
                        data, scales, squared_error, tensor_max_error = (
                            quantize_w4a16_group32(tensor)
                        )
                        encoding = ENCODING_W4A16_GROUP
                        group_size = GROUP_SIZE
                        quantized_elements += tensor.numel()
                        sum_squared_error += squared_error
                        max_abs_error = max(max_abs_error, tensor_max_error)
                    else:
                        data = fp16_bytes(tensor)
                        scales = b""
                        encoding = ENCODING_FP16
                        group_size = 0
                        fp16_elements += tensor.numel()

                    data_offset, data_size = write_aligned(payload_stream, data)
                    if scales:
                        scale_offset, scale_size = write_aligned(payload_stream, scales)
                    else:
                        scale_offset, scale_size = 0, 0
                    entries.append(
                        TensorEntry(
                            name=name,
                            encoding=encoding,
                            dims=dims,
                            group_size=group_size,
                            data_offset=data_offset,
                            data_size=data_size,
                            scale_offset=scale_offset,
                            scale_size=scale_size,
                        )
                    )
                    print(
                        f"[{index:03d}/{len(names):03d}] "
                        f"{'w4a16' if encoding else 'fp16  '} {name} {list(dims)}",
                        flush=True,
                    )

            payload_stream.flush()
            os.fsync(payload_stream.fileno())
            table_size = sum(ENTRY.size + len(item.name.encode("utf-8")) for item in entries)
            payload_start = align(HEADER.size + table_size)
            with output.open("wb") as target:
                target.write(HEADER.pack(MAGIC, VERSION, len(entries), payload_start))
                for item in entries:
                    name_bytes = item.name.encode("utf-8")
                    padded_dims = item.dims + (0,) * (4 - len(item.dims))
                    target.write(
                        ENTRY.pack(
                            len(name_bytes),
                            item.encoding,
                            len(item.dims),
                            0,
                            *padded_dims,
                            item.group_size,
                            0,
                            payload_start + item.data_offset,
                            item.data_size,
                            payload_start + item.scale_offset if item.scale_size else 0,
                            item.scale_size,
                        )
                    )
                    target.write(name_bytes)
                target.write(b"\0" * (payload_start - target.tell()))
                with payload_path.open("rb") as source_payload:
                    while chunk := source_payload.read(16 * 1024 * 1024):
                        target.write(chunk)
        finally:
            payload_path.unlink(missing_ok=True)

    validate_package(output)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    total_quantized = max(quantized_elements, 1)
    summary = {
        "format": "llmc-rknpu2-w4a16",
        "version": VERSION,
        "scope": scope,
        "source": str(checkpoint),
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "tensor_count": len(entries),
        "quantized_tensors": sum(
            item.encoding == ENCODING_W4A16_GROUP for item in entries
        ),
        "quantized_elements": quantized_elements,
        "fp16_elements": fp16_elements,
        "group_size": GROUP_SIZE,
        "quantization": "symmetric INT4 weights, FP16 activations, per-K-group scales",
        "scale_layout": "K-group-major [K/32, N] (RKNPU2 order)",
        "rmse": (sum_squared_error / total_quantized) ** 0.5,
        "max_abs_error": max_abs_error,
        "rknpu2_matmul_type": "RKNN_FLOAT16_MM_INT4_TO_FLOAT16",
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("encoder", "decoder", "all"), required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = build(args.checkpoint, args.output, args.scope)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
