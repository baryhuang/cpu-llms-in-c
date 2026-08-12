#!/usr/bin/env python3
"""Generate a pinned OpenAI Whisper 80- or 128-bin front-end fixture."""

import argparse
import hashlib
import io
import struct
import urllib.request
from pathlib import Path

import numpy as np

OPENAI_REVISION = "5f86d1d86363843179951550570367b37c5d6f78"
FILTER_URL = (
    "https://raw.githubusercontent.com/openai/whisper/"
    f"{OPENAI_REVISION}/whisper/assets/mel_filters.npz"
)
FILTER_FILE_SHA256 = "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c"
HEADER = struct.Struct("<8s7I")
N_FFT = 400
HOP = 160
N_FREQ = N_FFT // 2 + 1


def reflect_indices(indices: np.ndarray, count: int) -> np.ndarray:
    result = indices.copy()
    last = count - 1
    while np.any((result < 0) | (result > last)):
        result = np.where(result < 0, -result, result)
        result = np.where(result > last, 2 * last - result, result)
    return result


def reference_log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    frames = len(audio) // HOP
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    powers = np.empty((N_FREQ, frames), dtype=np.float32)
    offsets = np.arange(N_FFT, dtype=np.int64)

    for frame in range(frames):
        source = reflect_indices(frame * HOP - N_FFT // 2 + offsets, len(audio))
        values = (audio[source] * window).astype(np.float32)
        spectrum = np.fft.rfft(values, n=N_FFT)
        powers[:, frame] = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)

    mel = (filters.astype(np.float64) @ powers.astype(np.float64)).astype(np.float32)
    log_spec = np.log10(np.maximum(mel, np.float32(1.0e-10))).astype(np.float32)
    log_spec = np.maximum(log_spec, np.float32(log_spec.max() - 8.0)).astype(np.float32)
    return ((log_spec + np.float32(4.0)) / np.float32(4.0)).astype("<f4")


def load_filters(path: Path | None, n_mels: int) -> np.ndarray:
    if path is None:
        payload = urllib.request.urlopen(FILTER_URL, timeout=60).read()
    else:
        payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FILTER_FILE_SHA256:
        raise ValueError(f"mel filter hash mismatch: {digest}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        key = f"mel_{n_mels}"
        filters = archive[key].astype("<f4")
    if filters.shape != (n_mels, N_FREQ):
        raise ValueError(f"unexpected mel_{n_mels} shape: {filters.shape}")
    return filters


def build_audio(sample_count: int = 960) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / 16000.0
    phase = 2.0 * np.pi * (180.0 * t + 0.5 * 900.0 * t * t)
    audio = (
        0.31 * np.sin(2.0 * np.pi * 223.0 * t)
        + 0.17 * np.cos(2.0 * np.pi * 997.0 * t + 0.2)
        + 0.11 * np.sin(phase)
    )
    audio[0] += 0.29
    audio[319] -= 0.23
    audio[-1] += 0.19
    return audio.astype("<f4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mel-filters", type=Path)
    parser.add_argument("--n-mels", type=int, choices=(80, 128), default=128)
    args = parser.parse_args()

    filters = load_filters(args.mel_filters, args.n_mels)
    audio = build_audio()
    expected = reference_log_mel(audio, filters)
    frames = expected.shape[1]
    header = HEADER.pack(
        b"WHMEL001", 1, len(audio), frames, N_FFT, HOP, args.n_mels, N_FREQ
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + audio.tobytes() + filters.tobytes() + expected.tobytes())


if __name__ == "__main__":
    main()
