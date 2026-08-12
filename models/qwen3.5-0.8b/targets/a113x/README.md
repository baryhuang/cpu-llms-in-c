# Target: Amlogic A113X (ThirdReality LinuxBox)

Status: measured on the ThirdReality TRHub-V3. The generic scalar runtime and two cumulative A113X CPU increments use the same model image, 12 prompts, answer set, and four-thread process. Raw fields and per-case final outputs are in [`results.json`](results.json).

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

The complete 19-token ID sequence matches the local generic C runtime 19/19. Full per-token durations and IDs are in [`results.json`](results.json). This is one functional decode measurement, not a latency distribution.

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
6. **Next — A53 weight layout and prefetch:** reorder Q4 records offline for the measured cache and DRAM path.
7. **Experimental — Q3/Q2:** reduce traffic only if a larger held-out quality set passes.

## Feasibility notes

- Measured four-thread read bandwidth is 3.591 GiB/s. The final runtime remains compute-bound: it uses 3.84 cores while staying far below the bandwidth-only ceiling.
- The answer-set run uses ~367 MiB peak RSS because it does not touch the full embedding/head table. Free generation touches the whole tied head and raises peak RSS to ~488 MiB. Both remain at zero swap on the 2 GiB board.
