# Target: Amlogic A113X — Whisper large-v3-turbo

Status: the from-scratch C11 runtime executes the complete 32-layer encoder,
four-layer cached decoder and tokenizer on the 2 GB Cortex-A53 board. The
recommended Q4 compact-window path transcribes the 11-second JFK sample in
236.934 seconds at 448.2 MiB peak RSS and zero swap. It is an offline fallback,
not a real-time path.

## Current result

The final CPU increment vectorizes the two encoder convolutions across four
audio frames while retaining double-precision accumulation. It reuses the
same Cortex-A53 technique explored for Whisper small.en without repeating that
target's rejected float-accumulation implementation.

| Q4 compact JFK | Generic double stem | A113X double-NEON stem | Change |
|---|---:|---:|---:|
| Stem | 24.764 s | **15.767 s** | **1.571x** |
| Full encoder | 241.722 s | **229.763 s** | 1.052x |
| End to end | 249.212 s | **236.934 s** | **1.052x** |
| Peak RSS | 458,760 KiB | 458,956 KiB | +196 KiB |
| Process swap | 0 | 0 | unchanged |

Q5 independently reproduces the end-to-end result: 255.093 to 240.125
seconds, a 1.062x speedup. In both precision modes the generated token IDs,
printed top logits and final transcript are identical before and after the
stem change.

The compact window itself is the largest structural gain for short audio.
Against the original Q4 fixed-30-second JFK baseline of 1,323.840 seconds,
compact plus the double-NEON stem is **5.587x** faster. `fixed30` remains
available for explicitly padded comparisons.

## Fixed-window precision baseline

All rows are adjacent same-boot device runs with four OpenMP threads and the
performance governor pinned at 1.416 GHz. They use the pre-stem-optimization
binary and establish the quantization choice.

| Precision | Image | JFK 11 s total / RSS | std30 27.27 s total / RSS |
|---|---:|---:|---:|
| Q4 | 447,416,512 B | 1,323.840 s / 496.8 MiB | 1,300.079 s / 500.6 MiB |
| Q5 | 547,465,472 B | 1,280.763 s / 591.9 MiB | 1,309.350 s / 595.6 MiB |
| Q8 | 847,612,352 B | 1,396.278 s / 877.4 MiB | 1,271.225 s / 880.6 MiB |

Speed differences cross over between the two files, so they do not justify
the much larger Q5/Q8 memory footprints. Q4 is the deployment default.
Switching the governor from `ondemand` to `performance` reduced the Q4 JFK
fixed-window duration from 1,614.587 to 1,323.840 seconds.

## Optimization ledger

The initial target port already imports the successful A113X techniques from
Whisper small.en: grouped Q4/Q5/Q8 decode, four-output NEON FMA, two-row weight
reuse, 64-column output tiles, four-thread static partitioning and vectorized
self-attention.

| Experiment | Micro result | Full-transcription result | Decision |
|---|---:|---:|---|
| Q5 output tile 32 vs 64, two-row kernel | 4-layer sum 4.627 vs 4.777 s | not promoted | reject; interaction changed with row count |
| Four-row × four-output GEMM | up to 14.8% faster on a 150-frame microbench | 257.360 vs 255.093 s Q5 compact | **reject**, 0.89% regression |
| Double-precision NEON stem | 1100-frame A/B/B/A mean 19.074 vs 28.807 s | Q4 236.934 vs 249.212 s; Q5 240.125 vs 255.093 s | **keep** |

This repeats the repository's main CPU lesson: a kernel-only win is not a
release result until the complete transcription path also wins.

## Quality scope

- Q4, Q5 and Q8 fixed-window JFK transcripts are correct and word-identical.
- The two std30 renderings differ in case and punctuation by precision but
  normalize to the same words as the F32 x86 reference.
- The double-NEON stem has an identical 1100-frame synthetic checksum to the
  scalar-double stem across A/B/B/A runs.
- Q4 and Q5 compact A/B runs have identical generated IDs, printed logits and
  transcripts.
- Per owner direction, no 32-file LibriSpeech battery is run on this small-box
  target. The evidence is a two-file quantization check plus the JFK
  optimization gate, not a corpus WER certification.

## Reproduce

Build on the target with GCC 12.2:

```sh
make build/whisper-turbo-encoder-bench-a113x \
  build/whisper-turbo-transcribe-a113x \
  CC=gcc CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic' \
  OMPFLAGS=-fopenmp
```

Run the recommended compact Q4 path:

```sh
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  ./build/whisper-turbo-transcribe-a113x \
  turbo-q4.whtrbo jfk.wav 96 compact
```

Raw device records are in [`benchmarks/device/`](benchmarks/device/); scalar
x86 quantization-validation records are in
[`benchmarks/x86-validation/`](benchmarks/x86-validation/). Machine-readable
results are in [`results.json`](results.json).
