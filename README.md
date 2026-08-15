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
| Qwen3.5-0.8B | Amlogic A113X: 4x Cortex-A53 | 2017 | answer scoring + greedy generation, measured on device | matches x86 reference: 19/19 generation, 12/12 classification, 20/20 tokenizer | **3.64 prompt tok/s**, **2.60 decode tok/s**; 488 MiB RSS, zero swap | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) · [generation review](models/qwen3.5-0.8b/targets/a113x/GENERATION_REVIEW.html) · [ARC-Easy 5-case HTML](models/qwen3.5-0.8b/benchmarks/arc-easy-5/REVIEW.html) · [PDF](output/pdf/qwen35-arc-easy-5-review.pdf) · [raw data](models/qwen3.5-0.8b/targets/a113x/results.json) |
| Qwen3.6-27B | Apple M3 Pro: 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | 2023 | full chat runtime: batched prefill, lossless speculative decoding, FP16 KV, multi-turn serving, OpenAI API, thinking mode | matches oMLX token-for-token; speculation lossless; ARC-Easy 5/5 | **9.42 tok/s end-to-end**; 1.5-1.6x mlx-lm/oMLX; 1.46x llama.cpp; first token ~1 s | [model](models/qwen3.6-27b/README.md) · [target](models/qwen3.6-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.6-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.6-27b/targets/apple-m3-pro/results.json) |
| Qwen3.8-27B | Apple M3 Pro: 11-core CPU + 14-core Metal 3 GPU, 36 GB unified memory | 2023 | Qwen3.6 runtime unchanged (architecture verified identical); adds reasoning-effort control | all suites pass; speculation lossless; ARC-Easy 3/5 (misses correct in content, wrong format) | **9.66 tok/s end-to-end**; 1.56x llama.cpp | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/apple-m3-pro/README.md) · [review](models/qwen3.8-27b/targets/apple-m3-pro/REVIEW.html) · [raw data](models/qwen3.8-27b/targets/apple-m3-pro/results.json) |
| Whisper small.en | generic CPU | — | complete C FFT front end, encoder, cached decoder, tokenizer and full-graph image compiler | F32 boundaries pass; decoder tokens match NumPy; quantized error recorded | WAV → English text; target-neutral speed not a release result | [model](models/whisper-small.en/README.md) · [decision record](models/whisper-small.en/DECISIONS.md) · [generic target](models/whisper-small.en/targets/generic/README.md) |
| Whisper small.en | Amlogic A113X: 4x Cortex-A53 | 2017 | mixed Q4/Q8 full graph, NEON kernels, four threads, measured on device | JFK sample: 0/22 word errors; WER suite open | 11 s WAV in **45.0 s**; 251 MiB RSS, zero swap; **13.08x** vs baseline | [model](models/whisper-small.en/README.md) · [target](models/whisper-small.en/targets/a113x/README.md) · [HTML review](models/whisper-small.en/targets/a113x/benchmarks/jfk-11s/REVIEW.html) · [PDF review](output/pdf/whisper-small-en-a113x-jfk-11s-review.pdf) · [raw data](models/whisper-small.en/targets/a113x/results.json) |

Chip year means first public MP release, official launch, or official development-board sale; it is not the board manufacture year. The evidence and exact event are recorded in each target file.

The Gemma build predates the current design and hard-codes its two output labels; every later model takes any prompt at run time and produces free text through the full output head.

## Throughput

Qwen3.6-27B on the Apple M3 Pro, one request: 36-token prompt,
30-token greedy reply, model resident, four measured rounds per stack.
mlx-lm 0.31.3 and oMLX 0.5.7 compute on the same Q4 weight values as
this runtime and produced identical output; llama.cpp build 10360
(Metal) runs the Unsloth Q4_K_M file, a different Q4 quantization of
the same model. Raw record:
[`results.json`](models/qwen3.6-27b/targets/apple-m3-pro/results.json).

| Qwen3.6-27B, one request | This runtime | llama.cpp | mlx-lm | oMLX |
|---|---:|---:|---:|---:|
| End-to-end tok/s | 6.54 | 6.06 | 3.96 | 3.74 |
| End-to-end tok/s, speculative | **8.83** | — | — | — |
| Prompt tok/s | 36.4 | 39.3 | 13.9 | 11.6 |
| Decode tok/s | 8.48 | 7.44 | 6.03 | 5.90 |
| Request wall, s | 4.59 | 4.95 | 7.58 | 8.03 |
| First token, s | 0.99 | 0.92 | 2.60 | 3.11 |

| Five-workload set | End-to-end tok/s, speculative | Plain |
|---|---:|---:|
| Qwen3.6-27B | **9.42** | 7.91 |
| Qwen3.8-27B | **9.66** | 7.94 |

End-to-end = reply tokens / full request time. Speculative decoding is
this runtime's default and its output is token-identical to plain
decoding; the other stacks have no equivalent mode. The workload-set
rates are higher because long replies amortize the prompt. Per-case
tables: [Qwen3.6](models/qwen3.6-27b/targets/apple-m3-pro/README.md) ·
[Qwen3.8](models/qwen3.8-27b/targets/apple-m3-pro/README.md).

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

The chat stays resident: the model loads once (about 7 s), then every prompt reaches its first token in about 1 s. Follow-up turns reprocess only the new turn. With no arguments the script also starts the OpenAI-compatible server, installs and auto-configures the Chatbox client, and opens it; `--terminal` keeps the terminal chat.

The server also works standalone with any OpenAI-compatible client:

```sh
tools/qwen36_serve.py            # http://127.0.0.1:8199/v1
```

It honors each request's sampling fields, and a `reasoning_effort` field enables thinking mode with the reasoning streamed separately. Greedy requests keep lossless speculative decoding. All model execution stays in the resident C/Metal runtime.

```sh
tools/qwen36_monitor.py          # live CPU/GPU/memory dashboard; can record and render charts
```

Serving options, Qwen3.8 usage, and model-specific compile commands live in each model record.

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
