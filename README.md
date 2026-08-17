# cpu-llms-in-c

Compile a pinned model for a pinned hardware target, then run it without a general inference framework.

This repository explores how far model-aware and hardware-aware compilation can push local inference. The offline toolchain fixes tensor formats, layouts, graph rewrites, memory placement and schedules ahead of deployment. The target receives a compact C runtime, immutable model images and only the kernels needed by that model–hardware pair.

Runtime APIs and graph control are written in C. A target backend may add the minimum platform layer required by its hardware—for example, Objective-C and Metal shaders on Apple Silicon or NEON intrinsics on Arm.

| Compile time | Deployment time |
|---|---|
| Model revision, tokenizer and source hashes | Verified packed model images |
| Architecture-specific graph rewrites | C runtime with fixed state lifetimes |
| Quantization and tensor layout | CPU or on-SoC accelerator kernels |
| Target topology and memory constraints | No Python or training framework |

The project is not a wrapper around llama.cpp, MLX or ONNX Runtime. Those projects remain useful reference implementations and benchmark peers. The code here owns checkpoint import, image formats, runtime state, tokenization, sampling and target kernels.

## Implemented model × target pairs

Every performance claim belongs to one exact model revision and one exact target. Numbers do not transfer to a different chip or model.

| Model | CPU / SoC | Execution path | Measured result | Evidence |
|---|---|---|---|---|
| Qwen3.8-27B | Apple M3 Pro, 36 GB unified memory | C runtime + Metal kernels, affine Q4, FP16 KV, adaptive MTP | **10.82 end-to-end tok/s** five-workload aggregate, up to **14.86** on long code; 7.78 tok/s without MTP | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/apple-m3-pro/README.md) · [raw results](models/qwen3.8-27b/targets/apple-m3-pro/results.json) · [review](models/qwen3.8-27b/targets/apple-m3-pro/REVIEW.html) |
| MiniMax-H3 | Apple M3 Pro, 36 GB unified memory | C/Metal tokenizer, streamed Q8 conditioner, affine-Q4/BF16 H3, optimized Video VAE, Audio VAE | Exact 864×480×124 Turbo-4: **2,418.71 s**; frame-safe late-layer candidate: **2,344.73 s**, matching all three exact scene cuts without the old tree's ghosting; zero swap | [model](models/minimax-h3/README.md) · [target](models/minimax-h3/targets/apple-m3-pro/README.md) · [raw results](models/minimax-h3/targets/apple-m3-pro/results.json) · [review](models/minimax-h3/targets/apple-m3-pro/H3-ATTENTION-QUALITY-REVIEW.html) |
| Qwen3.5-0.8B | Amlogic A113X, 4× Cortex-A53, 2 GB | C11 + NEON, model-specialized DeltaNet state | **3.64 prompt tok/s**, **2.60 decode tok/s**, 488 MiB generation RSS, zero swap | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) · [raw results](models/qwen3.5-0.8b/targets/a113x/results.json) |
| Whisper small.en | Amlogic A113X, 4× Cortex-A53, 2 GB | C11 + NEON, mixed Q4/Q8 encoder and cached decoder | 11 s audio in **45.0 s**, 251 MiB RSS, zero swap; 0/22 word errors on the pinned JFK sample | [model](models/whisper-small.en/README.md) · [target](models/whisper-small.en/targets/a113x/README.md) · [raw results](models/whisper-small.en/targets/a113x/results.json) |
| Gemma 4 E2B | Unpinned two-vCPU x86-64 development machine | Legacy restricted C artifact | 0.598 token/s, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [raw results](models/gemma-4-e2b/results.json) |

The MiniMax-H3 Video VAE now uses static tensor bindings, precomputed RoPE,
grouped command buffers, simdgroup-matrix GEMM and exact tiled attention. The
same 480p N-to-N workload fell from 9,294.870 to 2,418.708 seconds (3.843×);
Video VAE decode fell from 7,387.292 to 487.274 seconds (15.160×). The
[next H3 attention round](models/minimax-h3/targets/apple-m3-pro/README.md#h3-attention-round-precompiled-data-and-metal-work)
removes a 448.4 MB exact-attention scratch buffer with bit-identical output and
compiles fixed tree topology and routes into an opt-in Metal execution image.
That aggressive approximate path reduces H3 denoise from 1,898.120 to 974.545
seconds and N-to-N runtime to 1,489.401 seconds, but it is not the default
because temporal stability and visual detail regress. A second candidate keeps
all conditioning exact, never averages K/V across frames, and approximates only
layers 40–49 of the last Turbo evaluation. It completed in 2,344.734 seconds;
the three scene cuts match exact, adjacent-frame luma difference is 4.283 versus
4.496 exact and 9.814 for the rejected tree, and sampled transition review
shows no double exposure. Exact remains the default until this result repeats
across a prompt suite. The
[target record](models/minimax-h3/targets/apple-m3-pro/README.md#video-vae-structure-and-optimization)
separates measured end-to-end results, component gates and projections.
Its [optimization ledger](models/minimax-h3/targets/apple-m3-pro/README.md#optimization-ledger)
records each observation, reason, isolated or combined timing, correctness
boundary and rejected branch. The optimized run recomputed all 200 H3 layer
calls and all 3,780 Video VAE block calls; local cache hits skipped weight
download/import, not inference.

Gemma is the early restricted artifact and accepts only its compiled test
inputs. Qwen3.5, Qwen3.8, Whisper and MiniMax-H3 execute their full model
paths; task behavior is selected by input or prompt rather than compiled
labels.

## Qwen3.8-27B on Apple M3 Pro

Qwen3.8 is the most complete target in the repository. It includes free-text chat, batched prefill, incremental multi-turn state, FP16 KV cache, greedy and sampled decoding, adaptive multi-token prediction, thinking-mode streaming and an OpenAI-compatible local server.

| Five-workload resident run | Plain greedy | Adaptive MTP | Increment |
|---|---:|---:|---:|
| End-to-end throughput | 7.78 tok/s | **10.82 tok/s** | **1.39×** |
| Decode throughput | 7.89 tok/s | **11.12 tok/s** | **1.41×** |
| Long-code case end-to-end | 7.82 tok/s | **14.86 tok/s** | **1.90×** |
| Total request wall | 424.6 s | **316.0 s** | 108.6 s less |

Speculative output is token-identical to plain greedy on three of the five cases; the two prose cases each flip one numerical near-tie onto an alternate fluent greedy continuation, and an exact-verify mode restores full token identity at 9.81 aggregate tok/s.

One separate 36-token-prompt, 28-token-reply comparison used resident models and four measured rounds:

| Runtime | End-to-end throughput | Request wall |
|---|---:|---:|
| This runtime, plain | 6.29 tok/s | 4.450 s |
| This runtime, adaptive MTP | **9.01 tok/s** | **3.109 s** |
| llama.cpp build 10360, Unsloth Q4_K_M | 5.76 tok/s | 4.864 s |

The llama.cpp checkpoint uses a different Q4 format, so this is a runtime-level comparison, not token-level quantization parity. Exact sources, hashes, prompts, output checks and per-case timings are in the [target record](models/qwen3.8-27b/targets/apple-m3-pro/results.json).

## How the repository is organized

Classification is model first, hardware target second:

```text
checkpoint + tokenizer + model rules + target profile
                         |
                         v
                  offline compiler
                         |
                         v
          packed images + target runtime + kernels
```

| Path | Responsibility |
|---|---|
| [`models/<model>/`](models/) | Source pins, architecture facts, model-level decisions and validation records |
| `models/<model>/targets/generic/` | Portable model runtime before CPU-specific specialization, where available |
| `models/<model>/targets/<target>/` | Target kernels, layouts, schedules, runtime code and measured results |
| [`compiler/`](compiler/README.md) | Offline checkpoint inspection, packing, fixture generation and target image compilers |
| [`tools/`](tools/README.md) | Commands users run directly: chat, serving, monitoring, comparison and hardware probing |
| [`tests/`](tests/) | Import, hashing, packing, sampler, state-machine and numerical parity tests |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Artifact contract and validation rules |

Checkpoints, generated model images, binaries and credentials are not committed.

## Why memory topology matters

Autoregressive decode repeatedly visits model weights. Once kernels are efficient, token generation approaches a bandwidth problem:

```text
decode tokens/s <= usable memory bandwidth / bytes visited per token
```

| Target | Resident model/image | Measured memory fact | Resulting engineering focus |
|---|---:|---:|---|
| Apple M3 Pro / Qwen3.8-27B | 15.139 GB mapped text image | 117–118 GB/s measured across complete representative layers | reduce bytes and dispatches per token; keep state in unified memory; speculate only when acceptance pays |
| A113X / Qwen3.5-0.8B | 470 MiB image | 3.591 GiB/s four-thread read probe | NEON GEMV, contiguous recurrent state, static head partition and zero swap |
| A113X / Whisper small.en | mixed Q4/Q8 image | 2 GB system RAM | bounded working memory, compact audio windows and cached decoder state |

The compiler therefore treats DIMM/channel bandwidth, unified-memory behavior, cache topology and storage speed as target inputs—not incidental machine details.

## Build and run

Run the committed tests:

```sh
make test
```

Build the Qwen3.8 Apple M3 Pro runtime and image tools:

```sh
make qwen38-m3-chat qwen38-tools qwen38-mtp-pack
```

After compiling the pinned checkpoint as described in the [target guide](models/qwen3.8-27b/targets/apple-m3-pro/README.md), start a resident terminal chat:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_chat.sh --terminal
```

Or start the OpenAI-compatible server:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_serve.py
# http://127.0.0.1:8199/v1
```

Model execution stays in the resident C/Metal process. The Python server is an optional standard-library HTTP adapter and is not part of the inference path.

## Good places to contribute

| Area | Concrete work |
|---|---|
| MiniMax-H3 / Apple Silicon | quality-gate hierarchical H3 attention and sparse projections against the corrected 480p dense trajectory; then offline-pack 64×32 VAE weight tiles |
| Qwen3.8 / Apple Silicon | fused batched prefill, DeltaNet scheduling, attention kernels, sampling and streaming overlap |
| Low-cost Arm CPUs | NEON kernels, recurrent-state layout, cache-aware thread partitioning and memory probes |
| New model support | add a model directory, independent numerical oracle, packed format and generic C runtime |
| New hardware support | add one target directory with an exact machine profile, specialized kernels and on-device measurements |
| Verification | expand layer-boundary parity, held-out quality suites, RSS/page-fault accounting and reproducible benchmark reports |

Performance patches need a correctness gate and raw before/after measurements. Approximate changes must record their quality effect; smoke examples are never presented as product-quality evaluation.

## License

No license has been selected yet.
