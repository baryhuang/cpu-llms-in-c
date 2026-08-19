# Whisper large-v3 / large-v3-turbo on Jetson Orin Nano Super

Baseline and incremental optimization of Whisper large-v3 and large-v3-turbo
on one exact machine, measured end to end by one public methodology for every
arm and every increment.

## The machine

| Fact | Value |
|---|---|
| Board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (tegra234) |
| CPU / RAM | 6 cores, 7 GB unified (CPU+GPU shared) |
| GPU | Orin, compute capability 8.7, VMM |
| Software | L4T R39.2.1, kernel 6.8.12-tegra, CUDA 13.2 (nvcc 13.2.r13.2) |
| Power state | `nvpmodel` MAXN_SUPER, `jetson_clocks` pinned — recorded with every run |

## Methodology (public tools only)

- **Inference arm**: `whisper.cpp` commit `4834a23` (2026-08-18), built with
  `cmake -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=87`. GPU offload and flash
  attention are whisper.cpp defaults.
- **Stage timings**: whisper.cpp's own public `whisper-bench`
  (load / encode / decode / batch / prompt).
- **End to end**: the pinned 32-file LibriSpeech `test-clean` subset
  (251.1 s audio, list and references in [`benchmarks/`](benchmarks/),
  list SHA-256 `19bcb463…`), one `whisper-cli` invocation per file with
  `-t 6 --language en`, default beam search. Per-file processing time is
  whisper.cpp's own reported `total time − load time`; **RTFx = audio
  seconds / summed processing seconds**. WER via `jiwer` + OpenAI English
  text normalizer (Open ASR Leaderboard method).
- Same audio, same normalizer, same machine state for all arms; compared
  arms run back to back in the same session.

## Baseline (2026-08-19, MAXN_SUPER + jetson_clocks)

| Model | Encode (ms) | Decode (ms/tok) | E2E RTFx | WER | Model load (s) |
|---|---:|---:|---:|---:|---:|
| large-v3 fp16 | 759.9 | 55.85 | 2.92 | 0.301 % | 35.8 |
| large-v3-turbo fp16 | 678.5 | 10.64 | **7.16** | 0.301 % | 3.9 |

Reading of the baseline: the two encoders cost nearly the same (both 32
layers); turbo's entire advantage is the 4-layer decoder (10.6 vs 55.9
ms/token single-stream). Decode is the memory-bound loop on 7 GB unified
RAM, and large-v3's 3.1 GB fp16 image also makes cold model load (35.8 s)
the dominant cost of any non-resident invocation. Both models transcribe
the clean subset at 2 word errors / 665 words, on different files —
per-file records are in `benchmarks/baseline-e2e-*.json`.

Optimization levers this baseline points at, in order: shrink decoder
bytes/token (quantization, gated on WER), then amortize or shrink model
load, then encoder.

## Optimization ledger

Every increment appends here with before/after measured by the same
methodology; negative results are kept. Memory footprint (peak RSS under
GNU `time -v`, swap after run) is recorded with every arm; swap stayed 0
in all runs.

| # | Change | Model | E2E RTFx | WER | Peak RSS | Verdict |
|---|---|---|---:|---:|---:|---|
| 0 | Baseline: whisper.cpp CUDA fp16, defaults | large-v3 | 2.78 | 0.301 % | 4,439 MB | reference |
| 0 | Baseline: whisper.cpp CUDA fp16, defaults | turbo | 7.18 | 0.301 % | 2,322 MB | reference |
| 1 | q8_0 quantization (`whisper-quantize`) | large-v3 | 3.94 | 0.150 % | 3,034 MB | gate passed; reference arm, not mainline |
| 1 | q5_0 quantization | large-v3 | 3.83 | 0.150 % | 2,485 MB | gate passed; reference arm, not mainline |
| 1 | q8_0 quantization | turbo | 7.58 | 0.301 % | 1,579 MB | gate passed; reference arm, not mainline |
| 1 | q5_0 quantization | turbo | 7.42 | 0.301 % | 1,293 MB | gate passed; reference arm, not mainline |

Baseline row 0 here is the same-window re-measure from the six-arm battery
(2.78 vs the first session's 2.92 — ~5 % session drift, which is exactly why
compared arms run back to back); all six arms above are one back-to-back
session.

**Quant gate detail**: no per-file WER regression in any arm; large-v3
q8_0/q5_0 each fixed one of the two baseline word errors (file
`1089-134686-0008`), turbo's error pattern is identical across fp16/q8/q5.
Single-stream decode fell 55.8 → 28.6 (q8) → 20.6 ms/token (q5) on
large-v3 and 10.6 → 5.3 → 3.3 on turbo.

**Why quantization is a reference arm, not the mainline** (owner decision,
2026-08-19): peak RSS shows memory capacity is not the constraint (4.4 GB
of 7 GB at worst), so the mainline is native C/CUDA optimization at fp16
quality, and the remaining headroom is available to trade for speed.
The battery also shows where the native work must aim: cutting decode
2.7× moved turbo e2e only 7.18 → 7.42–7.58 RTFx, because per-invocation
fixed costs — the ~660–760 ms encoder pass and per-file setup — now
dominate the end-to-end path. Encoder and fixed-cost elimination are the
levers with headroom; decode-side quant savings are largely amortized
already.

Raw records: [`results.json`](results.json), per-file evidence in
[`benchmarks/`](benchmarks/).
