# Whisper large-v3-turbo on Jetson Orin Nano Super

Turbo-variant record for the Jetson Orin target. The machine profile,
public methodology, and the native CUDA patch are shared with the
[large-v3 target](../../../whisper-large-v3/targets/jetson-orin/README.md)
— one patched whisper.cpp build (`4834a23` + the
[llmc kernel patch](../../../whisper-large-v3/targets/jetson-orin/patches/0001-whisper-shape-cuda-fusions.patch))
serves both models. This directory holds the turbo-specific measurements
and raw records; the increment-by-increment optimization ledger lives with
the shared patch in the large-v3 target.

## Certified result (2026-08-19, one session, upstream vs llmc patch, fp16)

| Arm | E2E RTFx | Encoder / 30 s | WER | Transcripts |
|---|---:|---:|---:|---|
| upstream whisper.cpp | 7.09 | 673 ms | 0.301 % | reference |
| **llmc patch** | **7.79 (+9.8 %)** | **590 ms (−12.4 %)** | 0.301 % | 32/32 byte-identical |

## Quantization modes (2026-08-19, 10 arms one session, upstream vs llmc per mode)

| Mode | File | E2E RTFx (upstream → llmc) | Decode ms/tok | WER (llmc) | Peak RSS |
|---|---:|---|---:|---:|---:|
| fp16 | 1.62 GB | 7.12 → **7.74** (+8.7 %) | 10.50 | 0.301 % | 2,352 MB |
| q8_0 (i8-weights) | 0.87 GB | 7.60 → **7.89** (+3.8 %) | 5.20 | 0.301 % | 1,610 MB |
| q5_0 | 0.57 GB | 7.39 → 7.54 (+2.1 %) | 3.18 | 0.301 % | 1,323 MB |
| q4_K | 0.47 GB | 6.97 → 7.12 (+2.1 %) | 3.25 | 0.150 % | 1,227 MB |
| q4_0 (w4a16-class) | 0.47 GB | 7.65 → 7.82 (+2.2 %) | **2.70** | 0.301 %† | 1,229 MB |

† q4_0-llmc shows 2/665 word errors vs upstream q4_0's 1/665 — word-level
flutter on the small subset, not a collapse; recorded honestly.

Reading: turbo's e2e is encoder-bound (~0.6 s per file), so quantization
moves e2e little — the e2e leader is q8_0-llmc (7.89). For streaming and
low latency the metric is decode ms/token, where q4_0-llmc reaches
**2.70 ms/token at a 1.2 GB footprint**. whisper.cpp quantization is
weights-only (activations stay fp16), so the int8-activation decoder
collapse seen on NPU int8 paths structurally cannot occur here; all five
modes transcribe with zero per-file regression except the flagged q4_0
word.

Cross-device anchor (encoder per fixed 30 s window, content-independent):
turbo fp16-llmc **589 ms** vs A311Y3 NPU w4a16 1,921 ms (3.3×), RK3588 NPU
i8 7,623 ms (collapsed), RK3588 CPU q5 48,671 ms.

Raw per-file records for every arm: [`benchmarks/`](benchmarks/)
(baseline, increments 3–6 A/B pairs, certification, quant batteries).
Machine profile, methodology definition, and the optimization ledger:
[large-v3 target record](../../../whisper-large-v3/targets/jetson-orin/README.md).
