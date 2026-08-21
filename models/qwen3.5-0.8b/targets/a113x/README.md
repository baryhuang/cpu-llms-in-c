# Target: Amlogic A113X (ThirdReality LinuxBox)

Status: measured on the ThirdReality TRHub-V3. The generic scalar runtime and two cumulative A113X CPU increments use the same model image, 12 prompts, answer set, and four-thread process. Raw fields and per-case final outputs are in [`results.json`](results.json). The human-readable free-generation input, output, and timing are in [`GENERATION_REVIEW.html`](GENERATION_REVIEW.html).

## CPU pin

| Field | Value |
|---|---|
| SoC | Amlogic A113X |
| Chip year | 2017 — OpenLinux MP release for A113D/A113X on 2017-08-31 |
| Cores | 4x ARM Cortex-A53 (ARMv8-A, AArch64) |
| SIMD | NEON/ASIMD 128-bit, includes TBL byte table lookup |
| RAM | 2,059,239,424 physical bytes (~1.92 GiB) |
| Storage | 6.9 GiB root eMMC filesystem |
| OS | Armbian / Debian bookworm, Linux 6.6.120-current-meson64 |
| Page size | 4,096 bytes |
| Measured read bandwidth | 1.753 GiB/s one thread; 3.591 GiB/s four threads |

Year evidence: [Amlogic OpenLinux release notes mirror](https://manuals.plus/m/66778c57bce54f4fd8afa6ff632d78c9ded060489bae66094d81b01c8d0b215a.pdf). The year identifies the chip software MP release, not the LinuxBox board revision.

The board exposes no temperature node through the tested `/sys/class/hwmon` paths. Frequency scaling ranges from 100 MHz to 1,416 MHz under the `ondemand` governor.

## Unified result table

Rows are cumulative implementation stages. Columns are separate workloads: classification prefill and free-generation decode are not divided by or compared with each other. An incremental factor is always `current stage throughput / previous stage throughput` inside the same column; cumulative factor is `current stage throughput / scalar throughput` inside the same column.

| Stage | Cumulative CPU implementation | Classification prefill | Increment | Cumulative | Steady greedy decode | Increment | Cumulative | Peak RSS: classification / generation | Output gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | Generic scalar | 0.8230 prompt tokens/s | 1.00x | 1.00x | not measured | — | — | 375,764 KiB / not measured | classification decisions match x86 12/12 |
| 1 | Stage 0 + Cortex-A53 NEON Q4/Q8 GEMV | 2.4297 prompt tokens/s | **2.95x** | **2.95x** | not measured | — | — | 375,972 KiB / not measured | classification decisions match x86 12/12 |
| 2 | Stage 1 + contiguous NEON DeltaNet state + four-thread head partition | **3.6353 prompt tokens/s** | **1.50x** | **4.42x** | **2.6005 generated tokens/s** | baseline pending | baseline pending | 376,152 KiB / 499,968 KiB | classification 12/12; generation IDs match generic C 19/19 |

The missing Stage 0 and Stage 1 decode cells require runs of the same generation prompt on the device. They are left explicit rather than inferred from classification measurements.

## Increment composition

| Stage | What changes | What does not change |
|---|---|---|
| 0 → 1 | Q4 nibble unpack and Q4/Q8 dot products become Cortex-A53 NEON kernels | model image, quantization, graph, prompt, tokenizer, answer set, thread count |
| 1 → 2 | DeltaNet state traversal becomes contiguous and its 16 independent heads are statically partitioned across four cores | Stage 1 NEON GEMV remains enabled; model and workload inputs remain unchanged |

The stages are additive, not three unrelated builds. Every reported increment uses the immediately preceding cumulative stage as its denominator. Verification is a gate after each implementation change and is excluded from benchmark duration.

## A113X-native weight image

`QW35TSK1` version 3 is the target-specific image format. Q4 and Q8 records
are stored as `[4-output block][input group][output lane][record]`, exactly the
order consumed by the four-output A113X GEMV loop. Tensor payloads are also
stored in numeric layer/execution order, followed by final norm and the tied
embedding/output head. Tokenizer tables precede the graph because prompt
tokenization runs first.

The runtime still accepts the measured row-major version 2 image. Version 3
has packing and record-order tests, but has not yet been benchmarked on
the A113X; no speedup is claimed from the layout alone.

```sh
python3 compiler/compile_qwen35_task_image.py \
  --checkpoint model.safetensors-00001-of-00001.safetensors \
  --config config.json --tokenizer tokenizer.json \
  --target a113x --output qwen35-a113x-v3.qtask
```

## Incremental benchmark

Benchmark and verification are separate. Every row below is a warm single-process run over the same 488 prompt tokens in 12 cases with `OMP_NUM_THREADS=4`. `Duration` is model classification time summed from the runtime; `Wall` comes from `/usr/bin/time -v`. Throughput is prompt prefill/classification throughput, not free-generation decode throughput.

| Cumulative implementation | Duration | Wall | Throughput | Increment vs previous | Increment vs baseline | CPU | Peak RSS | Swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Generic scalar baseline | 592.942 s | 593.10 s | 0.8230 token/s | 1.00x | 1.00x | 340% | 375,764 KiB | 0 |
| + Cortex-A53 NEON Q4/Q8 GEMV | 200.846 s | 201.04 s | 2.4297 token/s | **2.95x** | **2.95x** | 269% | 375,972 KiB | 0 |
| + contiguous NEON DeltaNet state kernel and static head partition | 134.241 s | 134.44 s | **3.6353 token/s** | **1.50x** | **4.42x** | 384% | 376,152 KiB | 0 |

The final CPU layer removes 77.36% of baseline classification duration without changing the image or task contract. RSS changes by 388 KiB and all runs use zero swap.

## Free-generation decode benchmark

This is a separate greedy generation run. Unlike answer-set scoring, every generated token scans all 248,320 rows of the tied output head. The runtime stops at `<|im_end|>`.

Input:

```text
<|im_start|>system
You are a concise assistant.<|im_end|>
<|im_start|>user
Name three primary colors.<|im_end|>
<|im_start|>assistant
```

Output:

```text
<think>

</think>

The primary colors are **Red**, **Blue**, and **Yellow**.
```

| Quantity | Result |
|---|---:|
| Prompt | 24 tokens |
| Requested / generated | 24 / 19 tokens; stopped on EOS |
| Prefill duration / throughput | 6.6597 s / 3.6038 prompt tokens/s |
| Full-head duration for first token | 0.1104 s |
| Time to first token | 6.7701 s |
| Steady decode | 18 tokens / 6.9218 s = **2.6005 tokens/s** |
| Generation after prefill, including first token | 19 tokens / 7.0322 s = 2.7018 tokens/s |
| Wall duration | 13.77 s |
| CPU | 386% |
| Peak RSS | 499,968 KiB (~488 MiB) |
| Swap | 0 |

The complete 19-token sequence matches the local generic C runtime 19/19. Review the exact English input, human-readable output, and per-token decoded pieces in [`GENERATION_REVIEW.html`](GENERATION_REVIEW.html). Machine-readable durations and IDs remain in [`results.json`](results.json). This is one functional decode measurement, not a latency distribution.

### ARC-Easy five-case generative smoke subset

Five fixed, source-order ARC-Easy validation prompts were supplied at run time. This is a generated-answer smoke adaptation, not an official ARC-Easy likelihood score. Exact inputs and outputs: [`../../benchmarks/arc-easy-5/REVIEW.html`](../../benchmarks/arc-easy-5/REVIEW.html). Raw A113X fields: [`../../benchmarks/arc-easy-5/results-a113x.json`](../../benchmarks/arc-easy-5/results-a113x.json).

| Quality | Prefill | Steady decode | Mean TTFT | Wall | CPU | Peak RSS | Swap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4/5 | 2.9182 tokens/s | 1.8220 tokens/s | 32.0487 s | 182.57 s | 333.84% wall-weighted | 502,016 KiB | 0 |

CPU and memory came from `/usr/bin/time -v` for each invocation. Individual CPU utilization was 316–354% of a four-core ceiling near 400%; process swaps were zero in every case, and device swap usage was zero both before and after the batch.

## Verification

Verification time is not included in the benchmark table.

| Check | Result |
|---|---|
| Final A113X decisions vs x86 C runtime | 12/12 equal |
| Maximum absolute answer-logit delta vs x86 | 1.72e-5 |
| Decisions vs written smoke labels | 10/12 |
| Wrong cases | `danger_uncontrolled_pressure`, `danger_robot_entry` — the x86 image makes the same decisions |

This is prompt-defined answer scoring: `safe,danger` is supplied for this run. Neither the target runtime nor the CPU kernels hard-code a binary classifier.

## Build and run

The measured optimized binary was compiled on the target with GCC 12.2:

```sh
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp -static \
  -mcpu=cortex-a53 -mtune=cortex-a53 \
  models/qwen3.5-0.8b/targets/a113x/qwen35_task.c \
  -o qwen35-task-a113x -lm

OMP_NUM_THREADS=4 ./qwen35-task-a113x qwen35-v2.qtask \
  --prompts-file hazard-v1.prompts --answers-text safe,danger

OMP_NUM_THREADS=4 ./qwen35-task-a113x qwen35-v2.qtask \
  --prompt "$PROMPT" --generate 24
```

## CPU-axis status

Measured steps are not estimates. Future steps remain unmeasured.

1. **Done — scalar baseline:** 0.8230 token/s.
2. **Done — NEON Q4/Q8 GEMV:** signed nibble unpacking and 16-lane float multiply-accumulate; 2.4297 token/s, 2.95x incremental.
3. **Done — DeltaNet state kernel:** contiguous `state[key][value]` row traversal plus 16 independent heads statically partitioned across four cores; 3.6353 token/s, 1.50x incremental.
4. **Next — NEON TBL lookup:** compare a table-lookup low-bit GEMV with the current multiply kernel; keep it only after full logit and 12-case gates.
5. **Next — multi-token prefill:** reuse weights or lookup tables across prompt tokens. DeltaNet recurrence stays ordered, but projections and MLPs can batch.
6. **Implemented, measurement pending — A53 weight layout:** Q4/Q8 records are output-blocked offline in the kernel's group/lane traversal order; the version 3 image still needs same-window device measurement.
7. **Experimental — Q3/Q2:** reduce traffic only if a larger held-out quality set passes.

## Feasibility notes

- Measured four-thread read bandwidth is 3.591 GiB/s. The final runtime remains compute-bound: it uses 3.84 cores while staying far below the bandwidth-only ceiling.
- The answer-set run uses ~367 MiB peak RSS because it does not touch the full embedding/head table. Free generation touches the whole tied head and raises peak RSS to ~488 MiB. Both remain at zero swap on the 2 GiB board.
