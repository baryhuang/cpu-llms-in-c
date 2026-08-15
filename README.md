# cpu-llms-in-c

An offline compiler turns a pinned language model into a packed image, and a small C11 runtime executes it on a pinned CPU/SoC. A target may dispatch compiler-selected graph regions to an on-SoC accelerator. The deployed target needs no Python, PyTorch, llama.cpp, or ONNX Runtime. Task outputs are defined by the prompt at run time — the runtime is not hardwired to one task.

## Organization

Everything is classified along two axes, model first, CPU/SoC target second. The target pins the CPU and, when selected, an on-SoC accelerator. A released artifact is one model x target pair, and results never transfer between pairs. The full contract is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Path | Contents |
|---|---|
| [`tools/`](tools/) | On-target probe: ISA, topology, and measured memory bandwidth for CPU pins |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| `models/<model>/` | Model axis: pins, profile, graph record, reference outputs, model-only optimizations |
| `models/<model>/targets/generic/` | The model's C runtime with model-axis optimizations only, portable to any CPU |
| `models/<model>/targets/<soc>/` | Target axis: CPU/SoC pin, accelerator boundary, specialized kernels, and results measured for that pair |
| [`tests/`](tests/) | Committed correctness tests (`make test`) |

Checkpoints, generated images, binaries, and credentials are never committed.

## Status

| Model | CPU / SoC target | Chip year | Status | Verification | Measured performance | Record |
|---|---|---:|---|---|---|---|
| Gemma 4 E2B | two-vCPU x86-64 dev machine (unpinned) | not pinned | implemented | 12/12 written labels, 10/10 layer-0 boundaries | 0.598 tokens/s scalar, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [inputs/outputs](REVIEW.html) · [raw data](models/gemma-4-e2b/results.json) |
| Qwen3.5-0.8B | Amlogic A113X: 4x Cortex-A53 | 2017 | target runtime measured; answer scoring and greedy free generation implemented | target vs local generic generation IDs 19/19; classification decisions vs x86 12/12; tokenizer parity 20/20 | **3.6353 prompt tokens/s** classification prefill; **2.6005 tokens/s** steady decode; 488 MiB generation RSS; zero swap | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) · [generation review](models/qwen3.5-0.8b/targets/a113x/GENERATION_REVIEW.html) · [ARC-Easy 5-case HTML](models/qwen3.5-0.8b/benchmarks/arc-easy-5/REVIEW.html) · [PDF](output/pdf/qwen35-arc-easy-5-review.pdf) · [raw data](models/qwen3.5-0.8b/targets/a113x/results.json) |
| Qwen3.5-0.8B | Rockchip RK3588S: 4x Cortex-A76 + 4x Cortex-A55, NPU | 2022 | target plan recorded | external baseline only; no target run | nothing measured by this repository | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/rk3588s/README.md) |
| Qwen3.5-0.8B | Rockchip RK3576: 4x Cortex-A72 + 4x Cortex-A53, NPU | 2024 | target plan recorded | external baseline only; no target run | nothing measured by this repository | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/rk3576/README.md) |
| Qwen3.6-27B | Apple M3 Pro: 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | 2023 | free-text end-to-end C/Metal runtime measured; adaptive multi-step MTP speculation, half-tile MMA prefill (S64..S4), FP16 KV, conversation continuation, per-request sampling, thinking mode; three-way vs mlx-lm/oMLX: decode 1.41x/1.44x faster | C vs oMLX: prompt IDs 36/36 and visible output IDs 30/30; MTP output token-identical on every battery; ARC-Easy-5 smoke 5/5 | **9.72 tok/s decode** aggregate over a 3,394-token five-case battery with adaptive MTP (8.09 plain; code up to 13.5 tok/s); warm TTFT 0.81 s; multi-turn TTFT up to 6.1x via continuation | [model](models/qwen3.6-27b/README.md) · [target](models/qwen3.6-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.6-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.6-27b/targets/apple-m3-pro/results.json) |
| Qwen3.8-27B | Apple M3 Pro: same pin | 2023 | released 2026-08; verified architecture-identical to Qwen3.6-27B, so the same image format, kernels and binaries serve it — the port added source pins, two MTP packers, and full 3.8 template semantics (reasoning_effort thinking mode with reasoning_content streaming, preserve_thinking) | api-state and prefill parity pass against the 3.8 images; MTP output token-identical on 5/5 matrix cases; ARC-Easy-5 smoke 3/5 strict (misses carry correct content without the format line) | **9.98 tok/s decode** aggregate over a 3,305-token five-case battery with adaptive MTP (8.13 plain; code 1.42-1.47x) | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.8-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.8-27b/targets/apple-m3-pro/results.json) |
| Whisper small.en | generic CPU | not pinned | complete C FFT front end, encoder, cached decoder, tokenizer and full-graph image compiler | F32 boundaries pass; three real-weight decoder tokens match NumPy; quantized error recorded separately | arbitrary PCM16 WAV to English text implemented; target-neutral speed is not a release result | [model](models/whisper-small.en/README.md) · [decision record](models/whisper-small.en/DECISIONS.md) · [generic target](models/whisper-small.en/targets/generic/README.md) |
| Whisper small.en | Amlogic A113X: 4x Cortex-A53 | 2017 | mixed Q4/Q8 full graph; compact window, Q4 row/output reuse, NEON kernels and four-thread scheduling measured | public JFK smoke case normalized word edits 0/22; relative-WER suite remains open | 11 s WAV, 3-run median: **45.047 s, RTF 4.095**, 388% CPU, 251,396 KiB RSS, zero swap; **13.08x** vs fixed30 | [model](models/whisper-small.en/README.md) · [target](models/whisper-small.en/targets/a113x/README.md) · [HTML review](models/whisper-small.en/targets/a113x/benchmarks/jfk-11s/REVIEW.html) · [PDF review](output/pdf/whisper-small-en-a113x-jfk-11s-review.pdf) · [raw data](models/whisper-small.en/targets/a113x/results.json) |

Chip year means first public MP release, official launch, or official development-board sale; it is not the board manufacture year. The evidence and exact event are recorded in each target file.

The Gemma artifact predates the prompt-defined output contract and compiles its two labels in — now the restricted special case. The Qwen artifact carries the default contract: runtime tokenizer, full output head, per-call answer sets.

## Evaluation isolation

A result belongs to one exact tuple: model revision x compiled-image hash x runtime-binary hash x CPU/SoC target x benchmark-profile hash. Changing the model or input data starts a new record. Old results remain evidence for the old tuple only.

| Reuse for a new model | Reset for a new model and new data |
|---|---|
| source revision and file-hash pinning method | model graph, tokenizer, chat template, and image |
| exact-input/output review format | prompts, cases, answer keys, and output parser |
| prefill, TTFT, decode, wall, CPU, RSS, swap, page-fault definitions | quality score and correctness conclusions |
| per-case JSON plus HTML/PDF review structure | throughput, latency, CPU, and memory measurements |
| verification separated from timed benchmark sections | target correctness agreement and quantization conclusions |

The ARC-Easy five-case profile is closed evidence for Qwen3.5-0.8B. It is not a default input set for the next model.

## Memory is the governing factor

Decode must stream every visited weight byte from DRAM for each token, so the hard ceiling is:

```text
tokens/s  ≤  usable memory bandwidth / weight bytes visited per token
```

| Quantity | Gemma 4 E2B on dev x86 (measured) | Qwen3.5-0.8B on A113X (measured) | Qwen3.6-27B on M3 Pro (measured) |
|---|---:|---:|---:|
| Weight bytes visited per decode token | 960 MB | ~350 MB (Q8 DeltaNet projections included) | 15,138.6 MB mapped text image |
| Usable/effective memory rate | not the limit yet | 3.591 GiB/s, four-thread read probe | 117.2 GB/s complete Delta layer; 118.3 GB/s complete attention layer |
| Achieved measured paths | 0.598 token/s (different model/workload) | 3.6353 prompt tokens/s classification; 2.6005 tokens/s steady greedy decode | 8.4227 decode tok/s; 3.1855 sequential prompt tok/s |
| Target optimization result | — | 4.42x cumulative; still compute-bound | decode 1.4396x oMLX; prompt 4.2554x slower than oMLX |
| Image / state vs RAM | 966 MB image, 926 MiB RSS, zero swap | 470 MiB image; 367 MiB answer scoring / 488 MiB generation RSS vs 1.92 GiB; zero swap | 15.139 GB mapped weights + 158.9 MB recurrent/conv state + 16.8 MB FP32 KV at capacity 128 vs 36 GB |

Two regimes follow, and they map exactly onto the two optimization axes:

| Regime | Binding constraint | Lever | Axis |
|---|---|---|---|
| Compute-bound | achieved runtime below the bandwidth-only ceiling | faster kernels, batching, layout | CPU |
| Bandwidth-bound (after SIMD kernels) | DRAM bandwidth itself | fewer bytes per token: smaller model, answer-set scoring, lower bits | model |

Capacity is a gate, not a tunable: the image and state must stay resident with zero swap — exceeding RAM means paging from eMMC at ~100-300 MB/s and an order-of-magnitude collapse.

The Rockchip NPU changes the memory/compute trade. The following numbers are external RKLLM v1.3 reference data with sequence length 128 and 64 generated tokens. They are not measurements by this repository.

| Model | Target | Quantization | TTFT | Decode | Reported memory | 1 GB implication |
|---|---|---|---:|---:|---:|---|
| Qwen3.5-0.8B | RK3588 | W8A8 | 587.74 ms | 27.05 tokens/s | 1039.66 MB | no OS/runtime headroom |
| Qwen3.5-0.8B | RK3576 | W4A16 | 1369.31 ms | 18.79 tokens/s | 689.50 MB | fits on paper; board RSS and zero swap still unverified |

Source: [Rockchip RKLLM benchmark, revision `878f936`](https://github.com/airockchip/rknn-llm/blob/878f9361fd3afa7e167b7079918918f78d2c1c2a/benchmark.md). RK3588S shares the RK3588 compute/NPU block used by this planning baseline, but the exact board must still be measured.

## Optimization roadmap

Measured and planned steps are labeled separately. Factors do not multiply cleanly and the stack is capped by the target's memory bandwidth. A step lands only after output/logit correctness checks for the code it touches.

| Step | Axis | Mechanism | Result or estimate |
|---|---|---|---|
| 1. Switch to Qwen3.5-0.8B | model | per-token matrix traffic 960 MB → ~350 MB (Q8 DeltaNet projections — see the model record) | ~2.7x per token |
| 2. Batched prefill | model | read each matrix once per layer per prompt, not per token | up to ~40x on prompt prefill |
| 3. Per-call answer-set scoring | model | skip the 248K-row output head unless generating | ~130 MB saved per decision |
| 4. NEON Q4/Q8 GEMV | CPU | Cortex-A53 signed nibble unpack and vector MAC | **measured prefill: 2.95x**, 0.8230 → 2.4297 prompt tokens/s |
| 5. DeltaNet state kernel | CPU | contiguous row traversal and four-thread static head partition | **measured prefill: 1.50x**, 2.4297 → 3.6353 prompt tokens/s |
| 6. Lower-bit LUT (experimental) | CPU | Q3/Q2 with linear LUT cost scaling, only if task quality survives | further 1.3-2x |

Model-axis details: [`models/qwen3.5-0.8b/README.md`](models/qwen3.5-0.8b/README.md). Target details: [A113X](models/qwen3.5-0.8b/targets/a113x/README.md) · [RK3588S](models/qwen3.5-0.8b/targets/rk3588s/README.md) · [RK3576](models/qwen3.5-0.8b/targets/rk3576/README.md).

### Whisper transcription paths

Transcript remains an open natural-language input to the downstream model. Whisper therefore retains its tokenizer, reachable output vocabulary and ordered timestamp segments. It is not compiled into a fixed classifier.

The active A113X deployment challenge is `small.en`-derived: English-only, continuous RTF `<=1.0`, less than or equal to 10% relative WER increase against the pinned unmodified `small.en`, peak RSS below 1 GiB and zero swap. `tiny.en` and `base.en` remain speed/quality controls. `medium.en` can be an offline English teacher. The feature losses, size comparison and structural plan are recorded in the [`small.en` decision record](models/whisper-small.en/DECISIONS.md).

| Step | Axis | Mechanism | Status / evidence |
|---|---|---|---|
| 1. Portable C graph | model | 80-bin Mel front end, 12-layer encoder, cached 12-layer decoder, tokenizer and no-timestamps decode | implemented and tensor-verified |
| 2. Compact audio window | model + workload | compile the encoder frame count from actual audio instead of unconditional 30-second padding | measured on 11-second JFK input: 589.056 → 160.803 seconds |
| 3. Cortex-A53 packed Q4/NEON | CPU | output-column tiling, four-output dot and two-row weight reuse | measured E9 median: 45.047 seconds, RTF 4.095, 13.08x cumulative |
| 4. Teacher-guided structural reduction | model | prune/distill encoder layers using `medium.en` or the pinned `small.en` baseline | required research branch; derived artifact needs a new identity and full WER rerun |

Full definitions, incremental gates, raw process output and review artifacts are in the [`small.en` A113X target record](models/whisper-small.en/targets/a113x/README.md).

## Why rewrite instead of using an existing stack

A general inference stack must accept any model and any prompt at load time. This repository instead compiles one pinned model, one task contract, and one CPU/SoC target ahead of time. A113X numbers explicitly labeled as measured come from the 2 GB ThirdReality board; stack-comparison numbers remain estimates until llama.cpp is measured on that same board.

| Stack | Fits the 1 GB board | Decode traffic per token (est.) | Disqualifier or cost |
|---|---|---:|---|
| This C runtime (measured capacity) | yes | ~350 MB | 367 MiB peak RSS and zero swap on the 2 GB A113X; per-call answer sets skip the head |
| llama.cpp (Q4 GGUF) | yes | ~420 MB | pays the full 248,320-row head every token; DeltaNet CPU op is freshly landed, unspecialized, and not tensor-verified |
| PyTorch + Transformers | no | ~1.6 GB | BF16 weights alone exceed RAM — kept only as the numerical oracle |
| ONNX Runtime | no | — | cannot run the architecture: no operators for non-softmax attention as of 2026 |

That leaves llama.cpp as the only stack that runs at all, and kernels are not what separate us from it: its NEON vectorization is imported here as CPU-axis roadmap step 4, after which both stacks face the same DRAM bandwidth wall. The remaining gap is structural — each row below depends on information a generic runtime does not have at load time:

| Structural advantage | Estimated effect on the A113X | Why a generic stack cannot absorb it |
|---|---|---|
| Per-call answer-set scoring | skips untouched head pages; final measured process is 367 MiB RSS for a 470 MiB image | needs the task contract in the runtime API; a generic decode loop computes all 248,320 logits every token |
| Compile-time exact rewrites | text-only input contract: MRoPE provably reduces to RoPE, vision tower and MTP head never enter the image | must keep run-time paths for inputs that never arrive |
| Fixed non-weight memory | ~30 MB bounded arena vs ~100-300 MB framework overhead — the headroom that decides fit on 1 GB | allocator, context, and graph machinery sized for generality |
| Dependency-free static binary | 948,296-byte measured A113X ELF; no Python or C++ runtime on the target | frameworks ship their runtime with the model |
| Tensor-verified DeltaNet path | per-boundary comparison against the pinned oracle before any kernel lands | upstream op is freshly landed with no equivalent gate |
| Per-target specialization | measured A113X gain is 4.42x from NEON GEMV plus DeltaNet state layout/thread scheduling | one build must serve every CPU |

The A113X capacity and CPU-specialization rows are measured. The llama.cpp traffic and overhead rows remain hypotheses until a same-board run is recorded.

## Where the differentiation is

The section above compares stacks on the same board. This one compares boards: the value of this repository falls monotonically with hardware price, because expensive edge hardware already has a solved LLM story. Prices are street prices observed 2026-08, not pinned quotes.

| Tier | Representative hardware | Street price | Existing LLM story | Value of this repository |
|---|---|---:|---|---|
| High-end edge | Jetson Orin Nano Super, 67 TOPS | $249 kit | mature — CUDA/TensorRT/Ollama, sub-1B models at 25-40 tokens/s | ~none |
| Mid SoC | RK3588S / RK3576, 6 TOPS NPU | $75-220 board | RKLLM works (27.05 / 18.79 tokens/s external baseline) | weak — kept as external reference baselines only |
| Low with NPU | RK3562, 1 TOPS NPU | ~$20-40 class | RKLLM supported, but Qwen3.5-0.8B w8a8 needs 1,021 MB — a 1 GB board cannot host it | real — Q4 CPU path is the only one that fits 1 GB |
| Low, no NPU path | A113X, RK3566-class | $10-20 SoC | none: no vendor stack targets these chips for LLMs | all of it |

The battleground is the bottom two rows: low-cost boxes already deployed in the field (smart-home hubs, gateways), 1 GB RAM, A53/A55 cores, hardware no vendor LLM stack serves or plans to serve. A $249 Jetson cannot reach a $20 BOM, and RKLLM requires chips and memory these boxes do not have — a dependency-free Q4 C runtime with ~30 MB overhead is the only path onto them.

Consequences for target priority: A113X stays first. RK3588S and RK3576 remain external reference baselines with no kernel investment. RK3562 is the interesting middle case — its A53-class CPU can reuse this kernel family, but every target still needs its own measurement.

## Build and test

```sh
make test
```

Run the compiled Qwen3.6-27B target interactively on the pinned Apple machine:

```sh
tools/qwen36_chat.sh
```

Interactive mode is resident: the model loads and wires once at startup (about 7 s), then every prompt answers at the ready-state latency — about 1.4 s to first token, streaming under `Model>` as tokens complete, with a `[first token …, tok/s]` status line after each reply. Defaults allow long replies: 4,096-token context and up to 3,072 new tokens per reply, generation stopping at the model's end token; a long prompt shrinks that reply budget instead of erroring (override with `QWEN36_CONTEXT` / `QWEN36_MAX_TOKENS`). Enter `/quit` to exit. A prompt passed as an argument runs the one-shot generator instead. The script uses the local compiled image under `tmp/qwen36-27b-runtime`; weights remain outside Git.

Running `tools/qwen36_chat.sh` with no arguments is the one-command app experience: it starts the OpenAI-compatible server if it is not already running, installs the Chatbox client on first use (Homebrew cask), copies the API address to the clipboard, and opens the client. First time only, add a provider in Chatbox: API host `http://127.0.0.1:8199/v1`, any API key, model `qwen3.6-27b`. `--terminal` keeps the resident terminal chat instead.

The server can also be run directly and used from any OpenAI-compatible client — Cherry Studio, Open WebUI, Raycast, Continue, Cline:

```sh
tools/qwen36_serve.py            # http://127.0.0.1:8199/v1, model id qwen3.6-27b
```

The shim (standard library only) renders multi-turn history into the pinned no-thinking template and streams SSE deltas; all model execution stays in the resident C/Metal runtime, so after the one-time startup wiring every request answers at ready-state latency.

To watch CPU, memory, GPU utilization and memory pressure while a run is active, start the stdlib-only monitor in a second terminal (no root needed). On a terminal it draws a live sparkline dashboard; it can also record a run and render it as an HTML chart page:

```sh
tools/qwen36_monitor.py                          # live sparkline dashboard
tools/qwen36_monitor.py python3.13               # follow an mlx-lm / oMLX run
tools/qwen36_monitor.py --record run.jsonl       # save samples while watching
tools/qwen36_monitor.py --render run.jsonl       # write run.html with SVG charts
```

Model-specific compile and run commands live in each model record, e.g. the [Gemma 4 E2B build](models/gemma-4-e2b/README.md#build).

## Current limits

| Limit | Where it is addressed |
|---|---|
| Gemma runtime has no tokenizer or free-text input | restricted legacy artifact; Qwen implements the current contract |
| Qwen3.6-27B pays 6.5 s of one-time weight wiring at model open | Apple M3 Pro target: MTLResidencySet moves it out of the first request; amortizes in a long-lived process |
| Qwen3.6-27B has representative layer and end-to-end token parity, not all 64 layer boundaries | extend the independent oracle before changing numerical kernels |
| No held-out application-quality evaluation | open |
| Cross-model compiler remains a design | the second model record starts generalizing it |

## License

No license has been selected.
