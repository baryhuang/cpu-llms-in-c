# Target: RK3588 RKNPU2 — Whisper large-v3-turbo

Status: measured on an Orange Pi 5 Plus with Rockchip's proprietary RKNPU2
stack. The final LLMC target runtime is C++17 and has no Python dependency.
Python is used only by the community baseline runner.

On the owner-provided 27.27-second `std30.wav`, the RKNPU2 FP16 baseline takes
45.774 seconds end to end, excluding model load. The LLMC median is **17.933
seconds**, a **2.552× speedup**. Encoder time falls from 12.840 to **8.725
seconds**, decoder latency falls from 408.33 to **111.56 ms/token**, and peak
RSS falls from 4,222.1 to **2,064.8 MiB**. Baseline and every native tuning run
produce the same 77-token transcript.

This target deliberately has no 32-file result. The only workload is
`std30.wav`; this is a same-file performance and exact-transcript gate, not
corpus WER certification.

## Proprietary RKNPU2 baseline versus LLMC

Both arms use the same precompiled FP16 encoder/decoder graphs, model revision,
`librknnrt` 2.3.2, driver 0.9.8 and audio hash. Model loading is excluded from
the inference figures.

| Metric | Python RKNPU2 baseline | Native C++ LLMC median | Gain |
|---|---:|---:|---:|
| Encoder / fixed 30 s | 12.840 s | **8.725 s** | **32.0% / 1.472×** |
| Decoder step, 80 steps | 408.33 ms | **111.56 ms** | **72.7% / 3.660×** |
| End to end | 45.774 s | **17.933 s** | **60.8% / 2.552×** |
| Peak RSS | 4,222.1 MiB | **2,064.8 MiB** | **51.1% lower** |

The best-profile LLMC end-to-end trials are 17.933, 17.957 and 17.901
seconds. Their median values are reported above rather than selecting the
fastest run.

## PDF quantization variants

The requested labels are GGML weight formats. They are not interchangeable
with RKNN graph quantization modes, so unsupported rows are recorded as such
instead of inventing measurements or relabeling another format.

| Requested variant | RKNPU2 Toolkit2 2.3.2 status | Device result |
|---|---|---:|
| FP16 | Supported by the pinned precompiled graphs | **17.933 s LLMC** |
| q8_0 | Rejected as `quantized_dtype`; native `w8a8` also quantizes activations and needs calibration | unsupported exact format |
| q5_0 | No native RK3588 5-bit conversion mode | unsupported exact format |
| q4_K | No native groupwise GGML q4_K conversion mode | unsupported exact format |
| q4_0 | Rejected; it is not the custom group-32 package below | unsupported exact GGML format |
| w4a16 · group 32 | Fused vendor dtype unsupported; custom CPU-W4→NPU-FP16 runtime supported | **79.596 s LLMC; correct output** |

Toolkit configuration accepts `w8a8`, `w16a16i` and `w16a16i_dfp`, but those
are not the PDF's q8/q4/q5 formats. No calibrated large-v3-turbo `w8a8` graph
was produced, so no native INT8 number is presented. The exact support probe is
in [`toolkit-2.3.2-quantization-support.log`](benchmarks/rknpu2/toolkit-2.3.2-quantization-support.log).

### Fused W4A16 capability and custom mixed-precision path

Toolkit2 2.3.2 rejects `quantized_dtype=w4a16` when compiling an RK3588
`.rknn` graph. Its generic MatMul header exposes
`RKNN_FLOAT16_MM_INT4_TO_FLOAT16`, but the RK3588 backend rejects that type at
context creation with `-5` (`unsupported ... in this platform`). Rockchip's
current changelog documents W4A16 for RK3576; the RK3588 INT4 addition is
`INT4 x INT4 -> INT16`, not W4A16. The official SDK demo reproduces the same
rejection at a canonical 1x64x64 shape while its FP16 control passes.

That fused-dtype rejection is retained as a capability result. It does not
prevent the Python-free C++17 runtime from owning mixed precision itself:

1. packed signed INT4 group-32 weights remain the stored model format;
2. C++ applies the group scales and expands each output-column shard to FP16;
3. RKNPU2 receives only supported FP16 x FP16 -> FP16 MatMul jobs;
4. output N is split at 16-column boundaries across three independent
   contexts pinned to Core0, Core1 and Core2;
5. linear and attention shards execute concurrently and are concatenated by
   the runtime.

Combined MatMul masks `0+1` and `0+1+2` return `-1` on this device. Individual
Core0, Core1 and Core2 masks all pass the same CPU-reference gate, so the
runtime implements parallelism explicitly instead of relying on
`RKNN_NPU_CORE_ALL`.

Two comparable modes are implemented in the same binary:

- `baseline` expands and prepares each three-way FP16 B shard on every use.
- `llmc` expands each decoder weight once, keeps all three B shards resident
  and reuses the three MatMul contexts and DMA allocations.

Both modes use the same v2 W4A16 files, FP16 activation/attention MatMuls,
native audio preprocessing, greedy decoder and `std30.wav`. Their 87 generated
tokens match exactly.

Measured on 2026-08-21:

| Item | Result |
|---|---:|
| Encoder W4A16 v2 | 409,684,160 bytes; 202 INT4 tensors; SHA-256 `22f02d37…30efc4fe` |
| Decoder W4A16 v2 | 241,173,248 bytes; 41 INT4 tensors; SHA-256 `8ff480c2…c5d3061` |
| Target executable | ARM64 Linux, pure C++17, maximum required glibc 2.29; FFTW linked statically; no Python, libsndfile or external FFTW runtime dependency |
| MatMul gate | 1x1280x51904: Core0 12.305 ms; parallel3 5.351 ms (**2.300x**); CPU/NPU cosine 0.999999978 |
| Baseline | encoder 67.177 s; decoder 216.308 s; end to end 283.728 s |
| LLMC | encoder 66.512 s; decoder 12.853 s; end to end **79.596 s** |
| LLMC gain | decoder **16.829x**; end to end **3.565x** |
| NPU gate | all three cores active; LLMC encoder max 45/44/44%, decoder max 17/18/17% |

The scale payload is deliberately versioned. v2 stores scales in RKNPU2's
`[K/32, N]` order; the native loader rejects the earlier v1 ordering.
The fixed `std30.wav` gate also uses an in-tree PCM16 WAV reader and validates
the exact 436,320-sample payload before initializing the NPU.

The fused capability evidence is in
[`rk3588-w4a16-capability-20260821.log`](benchmarks/rknpu2/w4a16/rk3588-w4a16-capability-20260821.log),
the package/audio gate is in
[`w4a16-package-validation-20260821.log`](benchmarks/rknpu2/w4a16/w4a16-package-validation-20260821.log).
The measured custom path is recorded in
[`matmul-parallel3-resident-gate.log`](benchmarks/rknpu2/w4a16/matmul-parallel3-resident-gate.log),
[`std30-w4fp16-parallel3-baseline.log`](benchmarks/rknpu2/w4a16/std30-w4fp16-parallel3-baseline.log)
and
[`std30-w4fp16-parallel3-llmc.log`](benchmarks/rknpu2/w4a16/std30-w4fp16-parallel3-llmc.log).

Build the two packages on the host (Python is build-time only):

```sh
python scripts/build_w4a16_weights.py \
  --checkpoint model.safetensors --scope encoder \
  --output whisper-large-v3-turbo-encoder-w4a16-v2.llmc
python scripts/build_w4a16_weights.py \
  --checkpoint model.safetensors --scope decoder \
  --output whisper-large-v3-turbo-decoder-w4a16-v2.llmc
```

The target-side gate is [`native/run_w4a16_std30.sh`](native/run_w4a16_std30.sh).
The full implementation is
[`native/whisper_rknpu2_w4a16.cc`](native/whisper_rknpu2_w4a16.cc), with the
RKNPU2 backend in [`native/w4a16_matmul.cc`](native/w4a16_matmul.cc) and the
versioned loader in [`native/w4a16_model.cc`](native/w4a16_model.cc).

## What LLMC changes

- The encoder's two 15.36 MB cross-attention KV outputs are imported into the
  decoder context through their DMA-BUF file descriptors. There is no host
  round trip between models.
- Decoder key and value self-caches are bound in place: each 4.59 MB buffer is
  both the next-step input and the present-cache output.
- Global RKNN input/output cache flushes are disabled. The runtime explicitly
  synchronizes only CPU-written features, token IDs and cache positions, plus
  CPU-read logits. Device-resident KV buffers are not invalidated every step.
- Encoder and decoder use `RKNN_NPU_CORE_ALL`. The process runs on Cortex-A76
  cores 4–7 with CPU, NPU and DMC performance governors during measurement.
  A trap restores `ondemand`, `rknpu_ondemand` and `dmc_ondemand` afterwards.
- Feature extraction, Slaney mel filters, byte-level token decoding and greedy
  suppression are implemented natively in C++17. Target inference imports no
  Python runtime or Python extension.

## Tuning ledger

| Native profile | Encoder | Decoder step | End to end | Decision |
|---|---:|---:|---:|---|
| Zero-copy, automatic NPU core, ondemand | 12.394 s | 143.21 ms | 24.736 s | keep zero-copy; tune cores |
| Zero-copy, all NPU cores, ondemand | 8.803 s | 128.86 ms | 19.933 s | keep all-core mask |
| All cores + A76/performance profile, median | **8.725 s** | **111.56 ms** | **17.933 s** | final |

## Exact pins

| Component | Revision |
|---|---|
| RKNPU2 large-v3-turbo graphs | `happyme531/whisper-large-v3-turbo-RKNN2` at `791cf8c7152a3882fc3fb7d3fb8d6718dfbea889` |
| RKNN-Toolkit2 | 2.3.2, `42aa1d426c0a9e0869b6374edba009f7208a1926` |
| RKNN Model Zoo | `bad6c7334531becaf90a561988519b7bec34d0ab` |
| RKNPU kernel driver | 0.9.8 |
| `librknnrt` | 2.3.2, SHA-256 `d31fc19c…bac738e8` |
| Encoder graph | 1,380,111,287 bytes, SHA-256 `987ebf7d…8625db2` |
| Decoder graph | 456,033,863 bytes, SHA-256 `663b2200…76082e6` |
| `std30.wav` | SHA-256 `659b371d…2429a2d` |

The pinned model repository is AGPL-3.0. Model files are not committed here.

## Build and run the Python-free LLMC target

Build on an aarch64 RK3588 system with the pinned Toolkit2 and Model Zoo
checkouts:

```sh
cmake -S native -B native-build \
  -DRKNN_TOOLKIT_ROOT=/path/to/rknn-toolkit2 \
  -DRKNN_MODEL_ZOO_ROOT=/path/to/rknn_model_zoo
cmake --build native-build -j4
```

Run one file with the measured governor/affinity profile; the script restores
all governors on exit and has no corpus or directory loop:

```sh
sudo native/run_std30_tuned.sh \
  native-build/whisper_rknpu2_llmc \
  encoder_with_kv.rknn decoder_static_kv.rknn vocab.json std30.wav \
  /path/to/rknn/runtime
```

The full implementation is
[`native/whisper_rknpu2_llmc.cc`](native/whisper_rknpu2_llmc.cc), with the
measured profile in [`native/run_std30_tuned.sh`](native/run_std30_tuned.sh).
Raw baseline,
native tuning, final repeats, tensor layouts and environment evidence live in
[`benchmarks/rknpu2/`](benchmarks/rknpu2/).

## Open-stack exploratory appendix

Earlier work under [`benchmarks/baseline/`](benchmarks/baseline/) and
[`benchmarks/llmc/`](benchmarks/llmc/) uses mainline Rocket plus
whisper.cpp/GGML. It remains useful for exact q8_0/q5_0/q4_K/q4_0 feasibility,
but it is a different stack and kernel. Those numbers are not used as the
proprietary RKNPU2 headline or as the fresh cross-device RK3588 row.
