# Target: Amlogic A113X — Whisper large-v3

Status: target plan recorded. The generic scalar 128-bin log-Mel boundary passes on this board. `tiny.en` Q5_1 has been measured as a feasibility reference. No large-v3 encoder/decoder artifact has been built or measured.

## CPU pin

| Field | Value |
|---|---|
| SoC | Amlogic A113X |
| Chip year | 2017 |
| CPU | 4x Cortex-A53, ARMv8-A/AArch64, maximum observed 1.416 GHz |
| SIMD | NEON/ASIMD and FMA |
| Missing ISA | dotprod, i8mm, SVE |
| RAM | 2,059,239,424 physical bytes (~1.92 GiB) |
| Storage | 6.9 GiB root eMMC filesystem |
| OS | Armbian / Debian bookworm, Linux 6.6.120-current-meson64 |
| Governor during reference runs | `ondemand` |
| Read bandwidth | 1.753 GiB/s one thread; 3.591 GiB/s four threads, from the existing target probe |

## Existing ASR feasibility reference

These rows use Whisper `tiny.en`, not large-v3. They establish target behavior and measurement procedure only. The public 11-second JFK sample was looped byte-exactly to 30 and 300 seconds. Text content is public; no CareMojo audio or transcript was read.

Machine-readable hashes and fields: [`results-tiny-reference.json`](results-tiny-reference.json).

Runtime: pinned `whisper.cpp` v1.8.0, CPU-only, four threads unless stated. RTF is wall/audio and uses the full original duration.

| Model / policy | Audio | Threads | Wall | RTF | Audio/wall | CPU | Peak RSS | Swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny.en Q5_1, beam 5 / best-of 5 | 11 s | 4 | 20.31 s | 1.846 | 0.542x | 376% | 132,956 KiB | 0 |
| tiny.en Q5_1, beam 5 / best-of 5 | 30 s | 4 | 24.79 s | 0.826 | 1.210x | 372% | 135,068 KiB | 0 |
| tiny.en Q5_1, beam 5 / best-of 5 | 300 s | 4 | 261.75 s | 0.873 | 1.146x | 376% | 251,472 KiB | 0 |
| tiny.en Q5_1, greedy | 30 s | 4 | 19.62 s | 0.654 | 1.529x | 373% | 105,396 KiB | 0 |
| tiny.en Q5_1, greedy | 300 s | 4 | 227.10 s | 0.757 | 1.321x | 378% | 223,000 KiB | 0 |
| tiny.en Q5_1, beam 5 / best-of 5 | 30 s | 3 | 33.29 s | 1.110 | 0.901x | 287% | 134,116 KiB | 0 |
| tiny.en F16, greedy | 30 s | 4 | 20.57 s | 0.686 | 1.458x | 373% | 151,784 KiB | 0 |

The 300-second greedy result is 13.2% faster than the beam row. Three threads do not keep up with full-duration input. Sequential scheduling with the downstream model is therefore the initial policy.

## Capacity gate for large-v3

| Item | Known value | Status on A113X |
|---|---:|---|
| F16 converted image | 3,095,033,483 bytes | cannot fit in 1.92 GiB RAM |
| Q5_0 converted image | 1,081,140,203 bytes | file fits; process peak and zero swap unverified |
| Decoder self-KV, cross-KV and work buffers | hundreds of MiB for full 32-layer decoder | must be measured together with image residency |
| Root filesystem free space at reference run | about 4.5 GiB | enough for one image plus build artifacts, but not a RAM result |

The first large-v3 target action is a load-only probe with `/usr/bin/time -v`, followed by one 30-second public-audio run. If swap is nonzero, execution stops; an RTF number obtained while paging is not retained as a usable baseline.

## CPU-axis incremental stages

All rows are cumulative and use the same model-axis artifact. A model-axis change, including a new per-layer precision manifest, starts a new table.

| Stage | Cumulative target implementation | Intended effect | Correctness gate | Status |
|---|---|---|---|---|
| T0 | generic C M3 artifact, scalar kernels, four threads | target baseline after capacity gate | segment/token agreement with generic C | blocked on generic runtime |
| T1 | T0 + Cortex-A53 NEON kernels for Q4/Q5/Q6/Q8 and F16/F32 accumulation | replace scalar projection, attention and MLP inner loops | primitive tails/alignment; graph boundaries | planned |
| T2 | T1 + compiler-selected encoder tiles and packed QKV/MLP layouts | reuse each weight tile across the 1500 audio positions and reduce repacking | exact-to-T1 selected tensors; same transcript | planned |
| T3 | T2 + fused layer normalization/projection, GELU and residual writeback schedules | reduce intermediate DRAM traffic | tensor tolerances and unchanged decode tokens | planned |
| T4 | T3 + static four-core partitions by head/output row and tuned prefetch distances | keep four A53 cores occupied without dynamic scheduling | deterministic reductions; repeat-run variance | planned |
| T5 | T4 + exact output-head search tree specialized for NEON and legal-token masks | avoid full vocabulary scan when bounds prove the top-k set | exact top-k equality for every decoder step | planned experiment |
| T6 | T5 + resident-image and sequential Whisper/Qwen scheduler | avoid eMMC faults and CPU contention | zero swap, queue-age and downstream coexistence test | planned |

`dotprod`, `i8mm` and SVE kernels are excluded: the silicon does not expose those instructions. Quantized kernels must use Cortex-A53 NEON widening multiply/accumulate and be measured; a smaller image is not assumed to be faster.

The pre-T0 M1 front-end boundary was compiled on target with GCC 12.2 and `-mcpu=cortex-a53 -mtune=cortex-a53`. It matches the committed 768-value fixture with maximum absolute delta `1.74045563e-5`; binary SHA-256 is `801e2ffbbd7d057b0e47c28ab2138574b83ca3e9b435852acaa239ca5608586d`. This verifies portability only. The scalar direct DFT is not retained as a performance row.

## Large-v3 feasibility estimate

The existing tiny encoder is four layers at width 384; large-v3 is 32 layers at width 1280. Dense encoder work scales approximately with `layers * width^2`:

| Quantity | tiny.en Q5_1 measured/reference | large-v3 analytical projection |
|---|---:|---:|
| Encoder layers / width | 4 / 384 | 32 / 1280 |
| Encoder work ratio | 1.0x | 88.9x |
| Encoder time per 30-second window | about 16.9 s | about 1,500 s |
| Encoder-only RTF | about 0.56 | about 50 |

Decoder work and timestamp handling are additional. This table is not a benchmark. It is the decision boundary: target stages T1-T6 can yield constant-factor gains, but full large-v3 requires about two orders of magnitude to approach real time. Model stage M8 — teacher-guided structured reduction — is therefore part of the A113X path, not an optional polish step.

## Benchmark matrix

When a large-v3 artifact exists, each retained row must include:

| Dimension | Required values |
|---|---|
| Audio | 30-second public fixture; 300-second public fixture; speech/silence mixture for VAD |
| Decoder | timestamped greedy; timestamped confidence fallback |
| Duration | front end, encoder, decoder, total wall |
| Throughput | original-audio RTF and audio/wall speed |
| Resources | CPU, peak RSS, process swap, device swap before/after, page faults |
| Quality | human-readable transcript, WER/CER, segment boundary error, repetition/hallucination flags |
| Identity | source revision, image hash, layer-manifest hash, binary hash, CPU pin |

Verification results appear in a separate section and are not included in benchmark duration.
