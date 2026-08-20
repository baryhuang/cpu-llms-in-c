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


def quantize_q8_grouped(weights: np.ndarray) -> bytes:
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
    return records.tobytes(order="C")


def q5_byte_count(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or shape[1] % GROUP_SIZE:
        raise ValueError(f"Q5 matrix shape must be [rows, multiple of {GROUP_SIZE}]: {shape}")
    return shape[0] * (shape[1] // GROUP_SIZE) * (2 + GROUP_SIZE // 2 + GROUP_SIZE // 8)


def quantize_q5_grouped(weights: np.ndarray) -> bytes:
    """Signed 5-bit groups: Q4-style low nibbles plus a 128-bit high-bit plane."""
    rows, columns = weights.shape
    if columns % GROUP_SIZE:
        raise ValueError(f"matrix width {columns} is not divisible by {GROUP_SIZE}")
    groups = columns // GROUP_SIZE
    blocks = np.asarray(weights, dtype=np.float32).reshape(rows, groups, GROUP_SIZE)
    minimum = np.min(blocks, axis=-1)
    maximum = np.max(blocks, axis=-1)
    scale = np.maximum(-minimum / np.float32(16.0), maximum / np.float32(15.0))
    scale[scale == 0] = np.float32(1.0)
    quantized = np.rint(blocks / scale[..., None]).clip(-16, 15).astype(np.int8)
    five_bit = (quantized.astype(np.int16) & 0x1F).astype(np.uint8)
    nibbles = five_bit & 0x0F
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << np.uint8(4))
    plane = np.packbits((five_bit >> np.uint8(4)) & np.uint8(1),
                        axis=-1, bitorder="little")
    records = np.empty((rows, groups, 2 + GROUP_SIZE // 2 + GROUP_SIZE // 8),
                       dtype=np.uint8)
    records[..., :2] = float32_to_bf16(scale).view(np.uint8).reshape(rows, groups, 2)
    records[..., 2:2 + GROUP_SIZE // 2] = packed
    records[..., 2 + GROUP_SIZE // 2:] = plane
    return records.tobytes(order="C")


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
