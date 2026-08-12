"""Minimal, selective safetensors reader used by the offline compiler."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


_DTYPE_BYTES = {
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


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.end - self.start


class SafetensorsFile:
    """Read safetensors metadata and selected tensor payloads without a framework."""

    def __init__(self, path: str | Path, maximum_header_bytes: int = 256 * 1024 * 1024):
        self.path = Path(path)
        self._file = self.path.open("rb")
        try:
            self.file_size = self.path.stat().st_size
            self.header_size = self._read_header_size(maximum_header_bytes)
            self.data_start = 8 + self.header_size
            self.metadata, self.tensors = self._read_header()
        except Exception:
            self._file.close()
            raise

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _read_header_size(self, maximum_header_bytes: int) -> int:
        prefix = self._file.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors file is shorter than its length prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > maximum_header_bytes:
            raise ValueError(f"invalid safetensors header size: {header_size}")
        if header_size > self.file_size - 8:
            raise ValueError("safetensors header extends beyond the file")
        return header_size

    def _read_header(self) -> tuple[dict[str, str], dict[str, TensorInfo]]:
        encoded = self._file.read(self.header_size)
        try:
            header = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid safetensors JSON header") from error
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")

        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        ):
            raise ValueError("invalid safetensors metadata")

        data_bytes = self.file_size - self.data_start
        tensors: dict[str, TensorInfo] = {}
        occupied: list[tuple[int, int, str]] = []
        for name, record in header.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ValueError("invalid safetensors tensor record")
            dtype = record.get("dtype")
            shape = record.get("shape")
            offsets = record.get("data_offsets")
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"unsupported dtype for {name}: {dtype}")
            if not isinstance(shape, list) or not all(
                isinstance(dimension, int) and dimension >= 0 for dimension in shape
            ):
                raise ValueError(f"invalid shape for {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) for offset in offsets)
            ):
                raise ValueError(f"invalid data offsets for {name}")
            start, end = offsets
            if start < 0 or end < start or end > data_bytes:
                raise ValueError(f"out-of-range data offsets for {name}")
            expected = math.prod(shape) * _DTYPE_BYTES[dtype]
            if end - start != expected:
                raise ValueError(
                    f"payload size mismatch for {name}: expected {expected}, got {end - start}"
                )
            tensors[name] = TensorInfo(name, dtype, tuple(shape), start, end)
            if end > start:
                occupied.append((start, end, name))

        occupied.sort()
        for previous, current in zip(occupied, occupied[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"overlapping tensor payloads: {previous[2]} and {current[2]}")
        return metadata, tensors

    def read_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        self._file.seek(self.data_start + info.start)
        payload = self._file.read(info.byte_count)
        if len(payload) != info.byte_count:
            raise ValueError(f"short tensor payload for {name}")
        return payload

    def read_element_bytes(self, name: str, element_offset: int, element_count: int) -> bytes:
        """Read a contiguous range from a flattened tensor."""

        info = self.tensors[name]
        if element_offset < 0 or element_count < 0:
            raise ValueError("tensor element range cannot be negative")
        if element_offset > info.element_count or element_count > info.element_count - element_offset:
            raise ValueError(f"tensor element range is outside {name}")
        element_bytes = _DTYPE_BYTES[info.dtype]
        byte_offset = element_offset * element_bytes
        byte_count = element_count * element_bytes
        self._file.seek(self.data_start + info.start + byte_offset)
        payload = self._file.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError(f"short tensor payload for {name}")
        return payload

    def read_float32(self, name: str) -> list[float]:
        """Convert a small F32, F16 or BF16 tensor to Python float values."""

        info = self.tensors[name]
        payload = self.read_bytes(name)
        if info.dtype == "F32":
            return list(struct.unpack(f"<{info.element_count}f", payload))
        if info.dtype == "F16":
            return [value[0] for value in struct.iter_unpack("<e", payload)]
        if info.dtype == "BF16":
            values = []
            for (bits,) in struct.iter_unpack("<H", payload):
                values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
            return values
        raise ValueError(f"tensor {name} is {info.dtype}, not F32, F16 or BF16")

    def read_float32_range(self, name: str, element_offset: int, element_count: int) -> list[float]:
        info = self.tensors[name]
        payload = self.read_element_bytes(name, element_offset, element_count)
        if info.dtype == "F32":
            return list(struct.unpack(f"<{element_count}f", payload))
        if info.dtype == "F16":
            return [value[0] for value in struct.iter_unpack("<e", payload)]
        if info.dtype == "BF16":
            values = []
            for (bits,) in struct.iter_unpack("<H", payload):
                values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
            return values
        raise ValueError(f"tensor {name} is {info.dtype}, not F32, F16 or BF16")

    def find_unique_suffix(self, suffix: str) -> str:
        matches = [name for name in self.tensors if name.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"expected one tensor ending in {suffix!r}, found {len(matches)}")
        return matches[0]
