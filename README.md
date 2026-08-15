# cpu-llms-in-c

An offline compiler turns one exact revision of a language model into a packed image, and a small C11 runtime executes it on one specific CPU/SoC. A target may dispatch compiler-selected graph regions to an on-SoC accelerator. The deployed machine needs no Python, PyTorch, llama.cpp, or ONNX Runtime. Task outputs are defined by the prompt at run time — the runtime is not hardwired to one task.

## Organization

Everything is classified along two axes, model first, CPU/SoC target second. A target names one exact CPU or SoC (and, when selected, its on-SoC accelerator); every result is measured on that hardware and transfers to no other. A released artifact is one model x target pair. The full contract is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

| Path | Contents |
|---|---|
| [`tools/`](tools/) | On-target probe: ISA, topology, and measured memory bandwidth of each target CPU |
| [`compiler/`](compiler/) | Offline compiler and independent reference tools |
| `models/<model>/` | Model axis: exact source revisions and hashes, profile, graph record, reference outputs, model-only optimizations |
| `models/<model>/targets/generic/` | The model's C runtime with model-axis optimizations only, portable to any CPU |
| `models/<model>/targets/<soc>/` | Target axis: the exact CPU/SoC, accelerator boundary, specialized kernels, and results measured for that pair |
| [`tests/`](tests/) | Committed correctness tests (`make test`) |

Checkpoints, generated images, binaries, and credentials are never committed.

## Status

| Model | CPU / SoC target | Chip year | Status | Verification | Measured performance | Record |
|---|---|---:|---|---|---|---|
| Gemma 4 E2B | two-vCPU x86-64 dev machine | — | implemented | 12/12 written labels, 10/10 layer-0 boundaries | 0.598 tokens/s scalar, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [inputs/outputs](REVIEW.html) · [raw data](models/gemma-4-e2b/results.json) |
| Qwen3.5-0.8B | Amlogic A113X: 4x Cortex-A53 | 2017 | measured on device: answer scoring and greedy generation | output matches the x86 reference token for token (19/19 generation, 12/12 classification, 20/20 tokenizer) | **3.64 prompt tok/s**, **2.60 decode tok/s**; 488 MiB RSS, zero swap | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) · [generation review](models/qwen3.5-0.8b/targets/a113x/GENERATION_REVIEW.html) · [ARC-Easy 5-case HTML](models/qwen3.5-0.8b/benchmarks/arc-easy-5/REVIEW.html) · [PDF](output/pdf/qwen35-arc-easy-5-review.pdf) · [raw data](models/qwen3.5-0.8b/targets/a113x/results.json) |
| Qwen3.6-27B | Apple M3 Pro: 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | 2023 | complete chat runtime: batched prompt reading, lossless speculative decoding, FP16 KV cache, multi-turn serving, OpenAI-compatible API, thinking mode | output matches oMLX token for token; speculative output identical to plain decoding on every test set; ARC-Easy 5-question check 5/5 | **9.42 tok/s end-to-end** on a 3,394-token workload set (7.91 without speculation); **1.5-1.6x faster than mlx-lm and oMLX**; ahead of llama.cpp end to end without speculation and **1.46x ahead with it**; first token in ~1 s with the model resident | [model](models/qwen3.6-27b/README.md) · [target](models/qwen3.6-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.6-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.6-27b/targets/apple-m3-pro/results.json) |
| Qwen3.8-27B | Apple M3 Pro: 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | 2023 | runs on the Qwen3.6 runtime unchanged (architecture verified identical tensor by tensor); adds reasoning-effort thinking mode | all runtime test suites pass; speculative output identical to plain decoding; ARC-Easy 5-question check 3/5 (each miss answers correctly but skips the required format line) | **9.66 tok/s end-to-end** on a 3,305-token workload set (7.94 without speculation) | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.8-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.8-27b/targets/apple-m3-pro/results.json) |
| Whisper small.en | generic CPU | — | complete C FFT front end, encoder, cached decoder, tokenizer and full-graph image compiler | F32 boundaries pass; three real-weight decoder tokens match NumPy; quantized error recorded separately | arbitrary PCM16 WAV to English text implemented; target-neutral speed is not a release result | [model](models/whisper-small.en/README.md) · [decision record](models/whisper-small.en/DECISIONS.md) · [generic target](models/whisper-small.en/targets/generic/README.md) |
| Whisper small.en | Amlogic A113X: 4x Cortex-A53 | 2017 | mixed Q4/Q8 full graph with NEON kernels and four-thread scheduling, measured on device | JFK reference sample transcribed with 0 word errors in 22; broader word-error-rate suite still open | 11 s WAV in **45.0 s** (3-run median), 251 MiB RSS, zero swap; **13.08x** faster than the unoptimized baseline | [model](models/whisper-small.en/README.md) · [target](models/whisper-small.en/targets/a113x/README.md) · [HTML review](models/whisper-small.en/targets/a113x/benchmarks/jfk-11s/REVIEW.html) · [PDF review](output/pdf/whisper-small-en-a113x-jfk-11s-review.pdf) · [raw data](models/whisper-small.en/targets/a113x/results.json) |

Chip year means first public MP release, official launch, or official development-board sale; it is not the board manufacture year. The evidence and exact event are recorded in each target file.

The Gemma build predates the current design and hard-codes its two output labels; every later model takes any prompt at run time and produces free text through the full output head.

## Throughput vs llama.cpp, mlx-lm and oMLX, same machine, same model

Four stacks measured on the Apple M3 Pro above, all serving the same
request — a 36-token prompt answered with a 30-token reply, greedy
decoding, model already resident, four measured rounds each. The
mlx-lm 0.31.3 and oMLX 0.5.7 runs compute on the same Q4 weight values
as this runtime and produced identical output tokens in all 12 runs.
llama.cpp (build 10360, Metal, flash attention) runs the Unsloth
Q4_K_M file — a different Q4 quantization of the same model, 15.65 GiB
of tensors vs this runtime's 14.1 GiB — and its reply text matched on
this prompt. Speculative decoding was off in all four stacks for a
like-for-like comparison; the row below the table shows it on. Raw
record:
[`results.json`](models/qwen3.6-27b/targets/apple-m3-pro/results.json).

| Tokens per second, higher is better | This runtime | llama.cpp | mlx-lm | oMLX |
|---|---:|---:|---:|---:|
| **End-to-end: reply tokens / total request time** | **6.54** | 6.06 | 3.96 | 3.74 |
| Prompt reading, long prompt | 63.6 | 73.0 | 13.9 | 11.6 |
| Generation (decode) | 8.48 | 7.44 | 6.03 | 5.90 |
| **End-to-end with speculative decoding on (this runtime's default)** | **8.83** | — | — | — |

| Supporting latencies in seconds, lower is better | This runtime | llama.cpp | mlx-lm | oMLX |
|---|---:|---:|---:|---:|
| Total request, model already loaded | 4.59 | 4.95 | 7.58 | 8.03 |
| First token | 0.99 | 0.92 | 2.60 | 3.11 |

End-to-end is the primary number: reply tokens divided by the whole
request time, prompt reading and first-token wait included — the rate a
user actually experiences. Prompt reading and generation are its two
components; any workload's end-to-end rate lands between them depending
on how much of the time is spent reading versus writing. When first
measured, this runtime and llama.cpp tied end to end from opposite
strengths; the techniques behind llama.cpp's faster prompt reading
(wide GEMM output tiles, vectorized quant loads, whole-prompt batching)
were then ported into this runtime's kernels, closing the long-prompt
gap to 13% and putting it ahead end to end without speculation. With
speculative decoding on, output stays token-identical to plain decoding
and the same request reaches 8.83 tok/s — 1.46x over llama.cpp, which
has no equivalent mode for this architecture — and 9.42 tok/s on a
3,394-token mixed workload set. Both native stacks remain 1.6-2.4x
faster than the Python stacks end to end. mlx-lm and oMLX prompt rates
were measured on the original 36-token request; this runtime and
llama.cpp on ~600-token prompts, where prompt reading is
steady-state.

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
| Achieved measured paths | 0.598 token/s (different model/workload) | 3.6353 prompt tokens/s classification; 2.6005 tokens/s steady greedy decode | 9.42 tok/s end-to-end with speculative decoding (7.91 without); ~52 prompt tok/s sustained on long prompts |
| Target optimization result | — | 4.42x cumulative; still compute-bound | 1.5-1.6x faster end to end than mlx-lm and oMLX on the same machine (comparison table above) |
| Image / state vs RAM | 966 MB image, 926 MiB RSS, zero swap | 470 MiB image; 367 MiB answer scoring / 488 MiB generation RSS vs 1.92 GiB; zero swap | 15.139 GB mapped weights + 158.9 MB recurrent/conv state + FP16 KV at 64 KiB per context token (0.27 GB at capacity 4096) vs 36 GB |

Two regimes follow, and they map exactly onto the two optimization axes:

| Regime | Binding constraint | Lever | Axis |
|---|---|---|---|
| Compute-bound | achieved runtime below the bandwidth-only ceiling | faster kernels, batching, layout | CPU |
| Bandwidth-bound (after SIMD kernels) | DRAM bandwidth itself | fewer bytes per token: smaller model, answer-set scoring, lower bits | model |

Capacity is a gate, not a tunable: the image and state must stay resident with zero swap — exceeding RAM means paging from eMMC at ~100-300 MB/s and an order-of-magnitude collapse.

## Why rewrite instead of using an existing stack

A general inference stack must accept any model and any prompt at load time. This repository instead compiles one pinned model, one task contract, and one CPU/SoC target ahead of time. A113X numbers explicitly labeled as measured come from the 2 GB ThirdReality board; stack-comparison numbers remain estimates until llama.cpp is measured on that same board.

| Stack | Fits the 1 GB board | Decode traffic per token (est.) | Disqualifier or cost |
|---|---|---:|---|
| This C runtime (measured capacity) | yes | ~350 MB | 367 MiB peak RSS and zero swap on the 2 GB A113X; per-call answer sets skip the head |
| llama.cpp (Q4 GGUF) | yes | ~420 MB | pays the full 248,320-row head every token; DeltaNet CPU op is freshly landed, unspecialized, and not tensor-verified |
| PyTorch + Transformers | no | ~1.6 GB | BF16 weights alone exceed RAM — kept only as the numerical oracle |
| ONNX Runtime | no | — | cannot run the architecture: no operators for non-softmax attention as of 2026 |

That leaves llama.cpp as the only stack that runs at all, and kernels are not what separate us from it: its NEON vectorization is imported here as the measured A113X NEON kernels (2.95x prefill), after which both stacks face the same DRAM bandwidth wall. The remaining gap is structural — each row below depends on information a generic runtime does not have at load time:

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

The section above compares stacks on the same board. This one compares boards: the value of this repository falls monotonically with hardware price, because expensive edge hardware already has a solved LLM story. Prices are street prices observed 2026-08, not fixed quotes.

| Tier | Representative hardware | Street price | Existing LLM story | Value of this repository |
|---|---|---:|---|---|
| High-end edge | Jetson Orin Nano Super, 67 TOPS | $249 kit | mature — CUDA/TensorRT/Ollama, sub-1B models at 25-40 tokens/s | ~none |
| Mid SoC | RK3588S / RK3576, 6 TOPS NPU | $75-220 board | RKLLM works (27.05 / 18.79 tokens/s external baseline) | weak — kept as external reference baselines only |
| Low with NPU | RK3562, 1 TOPS NPU | ~$20-40 class | RKLLM supported, but Qwen3.5-0.8B w8a8 needs 1,021 MB — a 1 GB board cannot host it | real — Q4 CPU path is the only one that fits 1 GB |
| Low, no NPU path | A113X, RK3566-class | $10-20 SoC | none: no vendor stack targets these chips for LLMs | all of it |

The battleground is the bottom two rows: low-cost boxes already deployed in the field (smart-home hubs, gateways), 1 GB RAM, A53/A55 cores, hardware no vendor LLM stack serves or plans to serve. A $249 Jetson cannot reach a $20 BOM, and RKLLM requires chips and memory these boxes do not have — a dependency-free Q4 C runtime with ~30 MB overhead is the only path onto them.

## Build and test

```sh
make test
```

Run the compiled Qwen3.6-27B target interactively on the Apple M3 Pro machine:

```sh
tools/qwen36_chat.sh
```

The chat stays resident: the model loads once at startup (about 7 s), and every prompt after that reaches its first token in about 1 s, streaming under `Model>` as tokens complete, with a `[first token …, tok/s]` status line after each reply. Defaults allow long replies: 4,096-token context and up to 3,072 new tokens per reply, generation stopping at the model's end token; a long prompt shrinks that reply budget instead of erroring (override with `QWEN36_CONTEXT` / `QWEN36_MAX_TOKENS`). Enter `/quit` to exit. A prompt passed as an argument runs the one-shot generator instead. The script uses the local compiled image under `tmp/qwen36-27b-runtime` (`QWEN36_MODEL_DIR=tmp/qwen38-27b-runtime` serves Qwen3.8-27B); weights remain outside Git.

Running `tools/qwen36_chat.sh` with no arguments is the one-command app experience: it starts the OpenAI-compatible server if it is not already running, installs the Chatbox client on first use (Homebrew cask), writes the provider configuration into Chatbox automatically, and opens the client. `--terminal` keeps the resident terminal chat instead.

The server can also be run directly and used from any OpenAI-compatible client — Cherry Studio, Open WebUI, Raycast, Continue, Cline:

```sh
tools/qwen36_serve.py            # http://127.0.0.1:8199/v1, model id qwen3.6-27b
```

The server is a single standard-library Python file that renders multi-turn history into the official template, passes each request's sampling fields (`temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `max_tokens`) through to the C sampler, and streams SSE deltas; a per-request `reasoning_effort` field (low/medium/xhigh, `none` to disable) switches to thinking mode with the think block streamed as `reasoning_content`. All model execution stays in the resident C/Metal runtime. The runtime remembers the conversation, so a follow-up request only processes the new turn — measured up to 6.1x faster to first token on later turns — and greedy requests keep speculative decoding with output identical to plain decoding.

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
