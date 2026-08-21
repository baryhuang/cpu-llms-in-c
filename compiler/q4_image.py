#!/usr/bin/env python3
"""Model-independent Q4 image helpers shared by the task-image compilers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

GROUP_SIZE = 128


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


def q8_byte_count(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or shape[1] % GROUP_SIZE:
        raise ValueError(f"Q8 matrix shape must be [rows, multiple of {GROUP_SIZE}]: {shape}")
    return shape[0] * (shape[1] // GROUP_SIZE) * (2 + GROUP_SIZE)


def _output_block_records(records: np.ndarray, output_block: int) -> bytes:
    """Serialize [output, group, record] in kernel traversal order.

    Output-blocked kernels visit every quantization group for a small block of
    output rows and consume all rows in that group before advancing. The
    resulting on-disk order is [output_block, group, output_lane, record].
    """
    rows, groups, record_bytes = records.shape
    if output_block <= 1:
        raise ValueError("output block must be greater than one")
    if rows % output_block:
        raise ValueError(
            f"matrix row count {rows} is not divisible by output block {output_block}"
        )
    blocked = records.reshape(
        rows // output_block, output_block, groups, record_bytes
    ).transpose(0, 2, 1, 3)
    return blocked.tobytes(order="C")


def _quantize_q8_records(weights: np.ndarray) -> np.ndarray:
    rows, columns = weights.shape
    if columns % GROUP_SIZE:
        raise ValueError(f"matrix width {columns} is not divisible by {GROUP_SIZE}")
    groups = columns // GROUP_SIZE
    blocks = np.asarray(weights, dtype=np.float32).reshape(rows, groups, GROUP_SIZE)
    scale = np.abs(blocks).max(axis=-1) / np.float32(127.0)
    scale[scale == 0] = np.float32(1.0)
    quantized = np.rint(blocks / scale[..., None]).clip(-127, 127).astype(np.int8)
    records = np.empty((rows, groups, 2 + GROUP_SIZE), dtype=np.uint8)
    records[..., :2] = float32_to_bf16(scale).view(np.uint8).reshape(rows, groups, 2)
    records[..., 2:] = quantized.view(np.uint8)
    return records


def quantize_q8_grouped(weights: np.ndarray) -> bytes:
    return _quantize_q8_records(weights).tobytes(order="C")


def quantize_q8_grouped_output_blocked(
    weights: np.ndarray, output_block: int
) -> bytes:
    return _output_block_records(_quantize_q8_records(weights), output_block)


def _quantize_q4_records(weights: np.ndarray) -> np.ndarray:
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
    return records


def quantize_q4_grouped(weights: np.ndarray) -> bytes:
    return _quantize_q4_records(weights).tobytes(order="C")


def quantize_q4_grouped_output_blocked(
    weights: np.ndarray, output_block: int
) -> bytes:
    return _output_block_records(_quantize_q4_records(weights), output_block)
