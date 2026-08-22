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

## RKLLM 1.3.0 W8A8 decoder experiment

An appended 2026-08-22 experiment uses the official RKLLM 1.3.0 custom-model
converter and runtime for the four-layer Whisper decoder. The fixed
30-second encoder remains the same FP16 RKNPU2 graph; `W8A8` therefore labels
the RKLLM decoder, not the encoder. Both measured arms are target-side C++ and
have no Python runtime dependency. They use the same 297,423,548-byte model,
the same cross-attention cache, all three declared NPU cores and the same
`std30.wav`.

The final export uses the official accuracy-priority
`optimization_level=1`. It is byte-identical to the preliminary level-0
export (297,423,548 bytes, SHA-256 `2268bb17…44c4c`), and a separate level-1
device validation reproduces the same degenerate output.

| Metric | Native RKLLM baseline median | Native C++ LLMC median | Change |
|---|---:|---:|---:|
| Encoder / fixed 30 s | 9.119 s | **8.989 s** | 1.0145× |
| RKLLM decoder / 128 generated steps | **2.635 s** | 2.645 s | 0.37% slower |
| End to end | 12.208 s | **12.146 s** | 1.0051× |
| Peak RSS | 2,151,620 KiB | **2,054,232 KiB** | 4.53% lower |
| Prefix calls + generated-token calls | 132 | **129** | four prefix tokens batched |
| Recognition | failed: one comma token repeated 128 times | same failure | not deployable |

The three baseline end-to-end trials are 12.162, 12.292 and 12.208 seconds;
the LLMC trials are 12.146, 12.183 and 12.040 seconds. Core0/1/2 were all
observed active. In the median LLMC encoder record their average loads are
87.41/21.92/21.92%; in the decoder they are 43.33/24.16/24.50%. This confirms
three-core dispatch, although work remains strongly concentrated on Core0.

The unmodified 1.3.0 runtime cannot execute this valid all-no-RoPE custom
decoder: its first decoder call asserts while copying an implicit
`position_ids` tensor whose buffer is intentionally absent. The official
runtime failure is retained as a separate gate. Formal timing uses a pinned,
four-byte compatibility patch that replaces only that unused copy call with
one AArch64 NOP; the original runtime is preserved. The patcher verifies the
expected instruction before writing a separate library. Original and patched
runtime SHA-256 values are `6a9e4fc…22a6e6` and `6524cf2b…63d4e8`.

The compatibility patch makes the graph executable but does not repair W8A8
accuracy. Both paths deterministically emit the same degenerate output, so
these timings are recorded as a negative capability/performance result and
must not be presented as a usable Whisper transcription. Raw trials and the
unmodified-runtime assertion are under [`benchmarks/rkllm/w8a8/`](benchmarks/rkllm/w8a8/).
The target runner is [`native/whisper_rkllm_w8a8.cc`](native/whisper_rkllm_w8a8.cc),
and the pinned compatibility utility is
[`native/rkllm_no_rope_patch.cc`](native/rkllm_no_rope_patch.cc).

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
| w4a16 · group 32 · retained v2 | Fused vendor dtype unsupported; custom CPU-W4→NPU-FP16 runtime supported | **79.596 s LLMC; correct output** |
| w4a16 · group 32 · AOT v3 | CPU-sequential tiled storage, fixed three-thread scheduler, shape-shared AOT and static head scheduling | **42.480 s LLMC median; correct output** |

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

The retained v2 measurements used two comparable modes in the same binary:

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

### AOT v3 fixed-thread result

The current appended path retains all v2 rows and adds a new versioned package
and scheduler:

> **Scope:** this is not native packed-W4 execution on the RK3588 NPU. The
> package stores W4, the native C++ runtime expands it to FP16 once during
> initialization, and Core0/Core1/Core2 execute supported FP16 MatMul shards.
> The accurate label is **AOT W4 storage -> resident FP16 -> three-core RKNPU2
> FP16 MatMul**. The packed INT4 values never enter the NPU.

- Each 640-byte W4 record contains 32 FP32 scales followed by one packed
  32x32 INT4 tile. N32 is outermost and K32 is innermost, so CPU expansion is
  a sequential read instead of separate scale and nibble streams.
- Nine shape-shared AOT plans own the RKNPU2 A/C contexts. All 235 W4 tensors
  are expanded during initialization into 1,611,694,080 resident FP16 B bytes;
  inference creates no context, thread, queue or DMA allocation.
- One long-lived C++ worker thread owns each of Core0, Core1 and Core2. Context
  creation, binding, execution and destruction stay on the owning thread. A
  fixed three-slot scheduler replaces per-call threads, promises and queues.
- Attention heads use a static round-robin mapping (`head % 3`). Encoder K/V
  for decoder cross-attention are transposed and packed once after encoding.

The same final binary was run once in baseline mode and three adjacent times
in LLMC mode. Model loading is excluded; `std30.wav` is the only workload.

| Metric | AOT v3 baseline | AOT v3 LLMC median | Gain |
|---|---:|---:|---:|
| Encoder / fixed 30 s | 34.930 s | **21.796 s** | **1.603x** |
| Decoder / 90 steps | 125.994 s | **20.733 s** | **6.077x** |
| Decoder step | 1,399.94 ms | **230.37 ms** | **6.077x** |
| End to end | 161.158 s | **42.480 s** | **3.794x** |

The LLMC end-to-end trials are 43.776, 42.441 and 42.480 seconds. All three
and the baseline emit the exact same 87-token sequence. The 1000-loop
1x1280x51904 gate completed 3000 NPU jobs without a crash or stale DMA data;
Core0/Core1/Core2 each reached 90% load, CPU/NPU cosine was 0.999999978, and
average three-core MatMul time was 4.730 ms.

The three cores are useful inside this custom path: the retained canonical
1x1280x51904 gate takes 5.351 ms on Core0/Core1/Core2 versus 12.305 ms on
Core0 alone, a 2.300x gain. That comparison must not be confused with the
17.933-second vendor FP16 graph, which uses `RKNN_NPU_CORE_ALL` and can keep
intermediate tensors inside a compiled, fused graph. The AOT W4 runtime makes
3172 W4 linear calls, 30080 attention MatMul calls and 752 host attention
dispatches. In the 42.480-second median trial, W4 linear wall time is 24.628
seconds and attention wall time is 13.682 seconds; decoder average load is
only 7.88/7.88/7.52% on Core0/Core1/Core2. Therefore the remaining gap is
dominated by fragmented FP16 jobs, host/NPU boundaries and expanded-weight
bandwidth, not by failure to activate the three cores.

Two measured alternatives were rejected: complete-N Q/K/V jobs statically
assigned one per core took 43.227 s, and retaining another 30.8 MB of
cross-attention B buffers took 42.583 s. Neither beat the lighter N-sharded
42.480 s median path. The custom path remains 2.369x slower than the 17.933 s
fused FP16 graph because RK3588 cannot fuse runtime W4 weights into that graph.

Raw records are under
[`benchmarks/rknpu2/w4a16/aot-v3/`](benchmarks/rknpu2/w4a16/aot-v3/).

### Mali-G610 packed-W4 decoder hybrid

The appended GPU path is native C++/OpenCL and directly consumes the same v3
N32/K32 packed INT4 records; decoder weights are not expanded on the CPU or
presented to the GPU as FP16. Static scope selection keeps the measured winner
for each stage:

- encoder W4 linear and attention stay on the three RKNPU2 cores;
- decoder W4 linear runs on all four Mali-G610 compute units;
- decoder attention is three fixed OpenCL kernels: QK, in-place softmax and PV;
- `M=1,N<=5120` uses a four-way K-split reduction, while the 51,904-column
  vocabulary projection uses an eight-output work-item;
- nine shape plans share A/C buffers. Encoder expanded FP16 NPU weights plus
  decoder packed GPU weights occupy 1,402,378,240 resident bytes.

The three adjacent `std30.wav` trials all emitted the exact same 87-token
sequence as the AOT v3 baseline and NPU-only LLMC path.

| Metric | Three-NPU AOT v3 | NPU/GPU hybrid median | Gain |
|---|---:|---:|---:|
| Encoder / fixed 30 s | 21.796 s | **21.717 s** | 1.004x |
| Decoder / 90 steps | 20.733 s | **4.267 s** | **4.859x** |
| Decoder step | 230.37 ms | **47.41 ms** | **4.859x** |
| End to end | 42.480 s | **26.213 s** | **1.621x** |

The hybrid trials are 26.213, 26.252 and 26.189 seconds. In the median trial,
decoder W4 linear wall time is 3.094 seconds versus 18.014 seconds in the
NPU-only median, a 5.823x gain. Decoder GPU load averages 49.83%, reaches 100%
and runs at 1 GHz. The result remains 1.462x slower than the separately fused
17.933-second FP16 graph.

Final 100-loop GPU gates at 1 GHz are:

| Shape `(M,K,N)` | Kernel mean | Maximum sampled absolute error |
|---|---:|---:|
| `(1,1280,1280)` | 0.441 ms | 0.00000918 |
| `(1,1280,5120)` | 1.506 ms | 0.00002345 |
| `(1,5120,1280)` | 1.729 ms | 0.00002201 |
| `(1,1280,51904)` | 8.446 ms | 0.00002687 |

Two broad GPU schedules were measured and rejected: moving every W4 linear to
the GPU took 51.118 seconds because the batched encoder was slower, while
moving encoder attention to the GPU took 41.127 seconds because its PV kernel
alone consumed 16.150 seconds. These measurements led to the fixed per-stage
hybrid rather than a dynamic scheduler.

The tested board exposes a proprietary Arm OpenCL 3.0 platform on Mali-G610
r0p0 with four compute units and `cl_khr_fp16`. Its kernel DDK is
`g25p0-00eac0`; the privately deployed `g24p0-00eac0` userspace blob has
SHA-256 `07cc993b13d6591161b0a12a270c75e06f25d56437016c6fbff8ea36e47c6614`.
That proprietary blob is subject to its vendor EULA and is deliberately not
committed. The implementation links the normal OpenCL ICD loader and requires
the operator to provide an appropriately licensed Mali ICD.

The target runtime is [`native/whisper_rknpu2_w4a16.cc`](native/whisper_rknpu2_w4a16.cc),
the packed-W4 kernels and shape plans are in
[`native/opencl_w4a16.cc`](native/opencl_w4a16.cc), decoder attention is in
[`native/opencl_attention.cc`](native/opencl_attention.cc), and the reversible
performance-profile runner is
[`native/run_opencl_w4a16_std30.sh`](native/run_opencl_w4a16_std30.sh).
Raw records are under
[`benchmarks/opencl/w4a16/`](benchmarks/opencl/w4a16/).

Build the two packages on the host (Python is build-time only):

```sh
python scripts/build_w4a16_weights.py \
  --checkpoint model.safetensors --scope encoder \
  --output whisper-large-v3-turbo-encoder-w4a16-v3.llmc
python scripts/build_w4a16_weights.py \
  --checkpoint model.safetensors --scope decoder \
  --output whisper-large-v3-turbo-decoder-w4a16-v3.llmc
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
