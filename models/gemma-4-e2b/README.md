# Gemma 4 E2B on Intel Celeron J3455

Status: scalar layer bring-up; no checkpoint-level inference.

This document records the first model-specific validation path for `cpu-llms-in-c`. It covers the Gemma 4 E2B text graph, an Intel Celeron J3455 target, proposed transformations, memory budgets, and throughput estimates.

The common compiler and runtime contracts are defined in [Compiler and Runtime Architecture](../../docs/ARCHITECTURE.md).

## 1. Evidence status

Statements in this document belong to one of four classes.

| Class | Meaning |
|---|---|
| Published fact | Taken from a primary model, implementation, or hardware source |
| Machine observation | Collected through read-only inspection of the target system |
| Analytical estimate | Derived from parameter counts, assumed formats, or bandwidth models |
| Project measurement | Produced by code in this repository on controlled hardware |

There are no checkpoint-level quality or performance measurements. The repository contains one miniature scalar layer correctness result; memory and throughput numbers below remain analytical estimates or design limits.

## 2. Target system

The machine was inspected read-only. No package was installed, no model was downloaded, and no inference process was run.

| Item | Observed value |
|---|---|
| System class | Schneider tablet-class x86 system; bare metal |
| CPU | Intel Celeron J3455, Apollo Lake, 4 cores / 4 threads |
| Frequency | 1.50 GHz nominal; 800 MHz to 2.30 GHz reported cpufreq range |
| SIMD | SSE, SSE2, SSSE3, SSE4.1, SSE4.2; no AVX, AVX2, FMA, or AVX-512 observed |
| Cache | 24 KiB L1d and 32 KiB L1i per core; 2 MiB L2 in two 1 MiB instances; no L3 reported |
| NUMA | One node, CPUs 0-3 |
| Visible RAM | 3,589,599,232 bytes, approximately 3.34 GiB |
| Swap | 3,923,767,296-byte swap file, approximately 3.65 GiB; unused during inspection |
| Memory DMI | Two 2 GB devices, ChannelA-DIMM0 and ChannelB-DIMM0, DDR3-1866, 16-bit data width |
| Storage | 62,545,461,248-byte eMMC, approximately 58.25 GiB; about 45 GiB free during inspection |
| Integrated GPU | Intel Apollo Lake GT1 / HD Graphics 500 |
| Operating system | Ubuntu 26.04 LTS, x86-64 |
| Kernel | Linux 7.0.0-14-generic |
| Baseline userland | Python 3.14.4, glibc 2.43; GCC and CMake not installed during inspection |
| CPU governor | `schedutil` on all four CPUs |
| Idle temperature | Approximately 37 degrees Celsius in one idle reading |

The DMI table also contained placeholder manufacturer, serial, part-number, and voltage values. Only the two device entries, channel locators, reported speed, and reported data width are retained as firmware observations. Physical bus width and sustained bandwidth remain unverified.

## 3. Fixed validation scope

```text
model:             Gemma 4 E2B IT
modality:          text input and text output
batch:             1
context:           at most 512 tokens
CPU:               Intel Celeron J3455, four cores, SSE4.2
steady-state RSS:  target 960 MiB; reject above 1,024 MiB
swap:              forbidden during inference
runtime:           C executable plus immutable packed model image
```

The target runtime is written in C. It does not depend on llama.cpp, LiteRT-LM, ONNX Runtime, PyTorch, or another inference framework on the target machine. Existing implementations are permitted as graph references and correctness oracles on development systems.

The HD Graphics 500, vision encoder, audio encoder, general 128K context, and MTP draft model are outside the first implementation stage.

## 4. Model selection context

Gemma 4 E2B IT is the primary bring-up target because it is the smallest Gemma 4 model and contains the architecture-specific features that the runtime must support.

Gemma 3 1B IT may be used as an optional scalar and quantized-kernel smoke test. It cannot validate Gemma 4 PLE, hybrid attention, shared K/V layers, proportional RoPE, or the Gemma 4 chat format.

Smaller general models exist. Qwen3-0.6B established a runnable 0.6B class, and Qwen3.5-0.8B is a relevant lower-bound reference for constrained instruction following. These models do not replace the Gemma 4 architecture target.

The following scale bands are working engineering categories, not standardized benchmark conclusions:

```text
runnable lower bound:                 approximately 0.6B parameters
limited general-purpose lower bound: approximately 0.8B parameters
broader practical range:             approximately 3B to 4B parameters
```

Capability depends on language, quantization, workload, and evaluation data.

## 5. Gemma 4 E2B published characteristics

Published Gemma 4 material describes E2B as follows:

- 2.3B effective parameters;
- 5.1B total parameters including Per-Layer Embeddings (PLE);
- 35 decoder layers;
- 262,144-token vocabulary;
- 512-token local sliding-attention window;
- 128K maximum context in the general model;
- text, image, and audio input with text output;
- an approximately 150M-parameter vision encoder;
- an approximately 300M-parameter audio encoder.

Published static-load estimates for E2B are 11.4 GB for BF16, 5.7 GB for SFP8, and 2.9 GB for Q4_0. Those estimates include 20% load overhead and exclude context KV cache. The official Q4_0 GGUF main model is 3.35 GB; its multimodal projector is 987 MB. The 0.84 GB LiteRT-LM mobile text-only estimate is not directly comparable to a general GGUF load.

Gemma 4 features that affect this runtime include:

- Per-Layer Embeddings;
- interleaved local sliding-window and global attention;
- unified K/V reuse in later layers;
- proportional RoPE for global attention;
- thinking mode and a new chat template;
- a separate MTP draft model;
- optional multimodal encoders and projector.

The official E2B QAT Q4_0 GGUF is the planned output and performance reference. The official E2B QAT assistant/draft checkpoint is approximately 76M to 78M parameters and will be considered only after base decode is correct.

Published E2B IT thinking-mode results include MMLU Pro 60.0%, AIME 2026 37.5%, LiveCodeBench v6 44.0%, GPQA Diamond 43.4%, an average of 24.5% across three Tau2 results, and MMMLU 67.4%. These are Google results, not measurements from this project. They do not predict target-workload quality or CPU speed.

## 6. Text-graph inventory

The current official configuration reports the following constants.

| Constant | E2B value |
|---|---:|
| Vocabulary | 262,144 |
| Hidden size | 1,536 |
| Decoder layers | 35 |
| Base MLP intermediate size | 6,144 |
| Late double-wide MLP intermediate size | 12,288 |
| Query heads | 8 |
| KV heads | 1 |
| Local head dimension | 256 |
| Global head dimension | 512 |
| PLE dimension per layer | 256 |
| Local window | 512 tokens |
| K/V-shared layers | Final 20 layers |
| Attention pattern | Four local layers followed by one global layer, repeated seven times |
| Local RoPE | Theta 10,000 over all local head dimensions |
| Global proportional RoPE | Theta 1,000,000; rotary factor 0.25 |
| Final logit soft cap | 30.0 |

The 35 layers are not one homogeneous loop. With zero-based indices:

| Layer class | Indices | Count | K/V source | MLP width |
|---|---|---:|---|---:|
| Early local | 0-13 except 4 and 9 | 12 | Generated by the layer | 6,144 |
| Early global | 4, 9, 14 | 3 | Generated by the layer | 6,144 |
| Late local | 15-33 except 19, 24, and 29 | 16 | Reuses local K/V from layer 13 | 12,288 |
| Late global | 19, 24, 29, 34 | 4 | Reuses global K/V from layer 14 | 12,288 |

Layers 15-34 have no independent K projection, V projection, K normalization, or V normalization. They still compute their own Q and attention output. Their MLPs are double-wide.

Ignoring small normalization and scalar tensors, the principal matrix counts are approximately:

| Layer class | Principal matrix parameters per layer |
|---|---:|
| Early local | 36.2M |
| Early global | 43.3M |
| Late local | 63.7M |
| Late global | 70.0M |

The total is approximately 1.87B parameters, consistent with the technical report's `Einsums: 1,870M` entry. The late double-wide MLPs dominate sequential weight traffic during token decode.

A text block is not treated as a generic pre-normalized Transformer block:

```text
x
 -> RMSNorm -> Q/K/V attention -> RMSNorm -> residual add
 -> RMSNorm -> GELU-gated MLP   -> RMSNorm -> residual add
 -> PLE gate * per-token PLE -> projection -> RMSNorm -> residual add
 -> layer scalar
```

Pre-normalization, post-normalization, QK normalization, and V normalization occur at distinct positions and must be preserved by the correctness implementation.

## 7. Exact and architecture-preserving transformations

### 7.1 Fold text-only PLE preprocessing offline

The packed E2B PLE table has shape:

```text
[262144, 35 * 256]
```

This is 2,348,810,240 parameters. It is large in storage, but one decode token accesses one row containing 8,960 values.

For text token `t`, the published forward path can be written as:

```text
main_input(t) = main_embedding[t] * sqrt(1536)
context_l(t)  = RMSNorm(W_context_l * main_input(t) / sqrt(1536))
token_l(t)    = PLE_table[t, l] * sqrt(256)
ple_l(t)      = (context_l(t) + token_l(t)) / sqrt(2)
```

The two `sqrt(1536)` factors around the context projection cancel. `ple_l(t)` depends on the token ID and fixed weights, not on the changing hidden state inside layer execution.

A text-only compiler can therefore precompute the final per-token, per-layer PLE representation and remove the following runtime work:

- the `1536 -> 35*256` per-layer model projection;
- the 35 post-projection RMSNorm operations;
- the runtime addition and scaling of the two PLE branches.

This inference does not apply to image or audio soft tokens. It must be validated against exported layer tensors with an explicit floating-point tolerance.

The folded PLE remains large. The proposed representation is token-row vector quantization:

- one row contains 35 layer slices;
- each layer slice has 256 dimensions;
- codebooks remain resident;
- codes and indices are read by token ID;
- one 8,960-value row is decoded once per token or incrementally by layer.

A third-party checkpoint reports AQLM-style compression of the approximately 4.7 GB BF16 PLE to approximately 0.26 GB. That figure has not been reproduced by this project. It is a size reference, not a result.

### 7.2 Store 15 physical K/V states, not 35

Layers 15-34 reuse layer 13 local K/V or layer 14 global K/V according to attention type. The runtime must not allocate independent KV caches for those 20 layers or store K/V weights that do not exist.

For the initial 512-token context limit:

- 12 early local layers use 512-entry circular KV buffers;
- 3 early global layers retain full-context KV;
- 16 late local and 4 late global layers retain references only.

The model uses MQA with one KV head. K/V data must not be physically replicated for the eight query heads.

### 7.3 Generate separate local and global attention kernels

Local and global attention differ in head width, RoPE, and cache access:

- local: 256-dimensional heads, full RoPE, 512-entry ring buffer;
- global: 512-dimensional heads, rotary transformation over only the first 25% of dimensions, full-context access.

The unrotated 75% of each global head must not enter the sine, cosine, or rotation loops. Position-dependent values can be advanced incrementally and reused across layers of the same attention type.

### 7.4 Fuse the MLP gate, up, and down paths

The MLP is:

```text
down(GELU(gate(x)) * up(x))
```

Gate and up projections share one normalized input. Their weights should be interleaved by tile. The input is quantized once. GELU, elementwise multiplication, and the down projection are processed by intermediate-dimension stripe, avoiding a complete 6,144- or 12,288-element temporary array.

For a late double-wide layer, four persistent worker threads each process a fixed intermediate stripe, produce a 1,536-element partial output, and enter a fixed-size reduction.

### 7.5 Fuse normalization, residual, and quantization boundaries

The J3455 has no FP16 or BF16 arithmetic. The initial numerical plan is:

- FP32 for hidden state, residuals, normalization, and output accumulation;
- dynamic int8 quantization for projection inputs;
- tensor-sensitive 2-, 3-, or 4-bit packed weights;
- int32 dot-product accumulation;
- FP32 projection output.

When Q/K/V or gate/up share a normalized input, that input is quantized once. Post-attention normalization plus residual, post-MLP normalization plus residual, and post-PLE normalization plus residual are each one streaming pass.

### 7.6 Generate an SSE4.2 GEMV path

The target has no AVX2 or VNNI. The initial kernel plan is a 128-bit SSE path:

- SSSE3 `pshufb` for 4-bit or 2-bit unpacking;
- `pmaddubsw` followed by `pmaddwd` for byte dot products;
- unsigned activations with a fixed zero point and a precomputed correction sum per weight row;
- bounded block accumulation to prevent `pmaddubsw` 16-bit saturation;
- 64-byte alignment for weights, scales, and correction data;
- a scalar reference kernel for each packed format.

Four cores use a persistent pthread pool and static partitions. The token loop does not create threads or call a general allocator.

### 7.7 Store tied embedding and LM-head weights once

The main embedding is `262144 x 1536`, approximately 402.7M parameters, and is tied to the LM head. The packed image stores one copy:

- input lookup reads one row by token ID;
- general output scans vocabulary rows in four fixed partitions.

The final soft cap is the monotonic function `30*tanh(logit/30)`. Therefore:

- greedy argmax does not need to evaluate the soft cap;
- top-k selection may be performed on raw logits, followed by soft cap and softmax only for the selected values;
- a 262,144-entry FP32 logit buffer is not required.

This preserves ranking.

## 8. Approximate transformations under investigation

### 8.1 Shared-codebook matrix representation

The first size-oriented representation is a shared-codebook directed acyclic graph rather than conditional neuron skipping.

For a group of input columns, the runtime computes the input tile against a small set of codewords. Each output row then reads compact code indices and accumulates values from the resulting partial-sum table.

```text
activation tile
        |
dot products against K codewords
        |
small partial-sum table
        |
packed code indices for output rows
        |
dense output vector
```

All output rows remain present. Relative to the quantized matrix, lookup execution can be exact; relative to the BF16 teacher, the approximation is the codebook quantization itself.

Offline search may:

- optimize codebooks jointly within a Transformer block;
- share codebooks across compatible layers;
- assign residual refinement codes to sensitive blocks;
- optimize against teacher layer output and final-logit loss rather than weight L2 alone;
- search layouts compatible with 4-bit selectors, 16-entry tables, and `pshufb`;
- co-design code indices, scales, corrections, and output-row ordering for sequential reads.

The J3455 has no gather instruction. A LUT method that performs well on a newer CPU or GPU may be slower than packed integer dot products on this target. The compiler must generate both packed-dot and codebook-LUT candidates and select from target measurements.

### 8.2 Conditional expert trees for late MLPs

For one gated MLP:

```text
a_j(x) = GELU(gate_j dot x) * (up_j dot x)
y(x)   = sum_j a_j(x) * down_column_j
```

The gate row, up row, and matching down column for one intermediate neuron are one atomic unit. The compiler may cluster these units by co-activation, output direction, and teacher-loss sensitivity, then construct a small routing tree:

```text
normalized hidden state
        |
small oblique routing tree
        |
top-k leaf identifiers
        |
packed gate/up rows and matching down columns
        |
1,536-element MLP output
```

The first candidates are layers 15-34. Their MLP parameter distribution is approximately:

| Portion | Parameters |
|---|---:|
| Layers 0-14, 6,144-wide MLPs | 425M |
| Layers 15-34, 12,288-wide MLPs | 1,132M |
| Total MLP parameters | 1,557M |

The late MLPs account for approximately 61% of the 1.87B einsum parameters. Routing them addresses the largest continuous decode stream without changing attention, K/V sharing, or PLE.

Candidate searches include 32, 64, 128, and 256 leaves, with per-layer top-k selection. Internal nodes use low-dimensional or sparse oblique projections. Each leaf occupies one contiguous cache-line-aligned region. Frequently used leaves may receive higher precision.

Reordering dense neurons into leaves does not reduce the total model image. Conditional routing reduces bytes read and operations only when leaves are skipped. If every leaf remains resident, routing alone does not reduce RSS. The 1 GiB target still requires low-bit coding, shared codebooks, or actual parameter removal after distillation.

This is not an exact rewrite. Most GELU-gated contributions are not exactly zero. The required control process is:

1. Collect real per-layer hidden states from a dense BF16 teacher.
2. Score units using `abs(a_j(x)) * norm(down_column_j)` and final-logit loss.
3. Search partitions, routing trees, top-k values, and leaf precision.
4. Fit layer residuals, then distill router, leaf scales, and any low-rank correction end to end.
5. Emit a confidence threshold per layer; low confidence increases the leaf count.
6. Report quality, bytes read, and latency across complete validation, boundary, long-input, and atypical-token sets.

Without training or distillation, this method is approximate structured pruning, not a compiler-equivalent optimization.

### 8.3 Vocabulary search tree

The LM head is maximum-inner-product search over 262,144 rows of dimension 1,536. A hierarchical clustering tree may store centroids, radii, or tighter inner-product bounds in internal nodes and contiguous vocabulary rows in leaves.

An exact branch-and-bound mode can reject a subtree only when its upper bound is below the current kth logit. It preserves top-k, but high dimensionality may make the bound ineffective. Worst-case work remains a full vocabulary scan.

An approximate mode can search a fixed number of clusters and rerank exact rows. It requires an exact fallback when the candidate margin is small, a bound is inconclusive, or routing confidence is low.

The vocabulary tree is intended to reduce per-token LM-head traffic. It is not the primary mechanism for reducing resident memory because the tied table is also needed for input embeddings unless the input token set is proven finite.

### 8.4 Hierarchical PLE codebooks

PLE is indexed directly by token ID and does not need hidden-state routing. It may use residual or additive vector quantization:

- decode a coarse code;
- apply zero or more refinement codes;
- allocate additional refinement to sensitive token/layer slices.

Code selection should minimize downstream teacher loss, not only reconstruction error. AQLM provides evidence for sub-3-bit additive quantization and block-level optimization, but it does not establish results for Gemma 4 PLE.

### 8.5 Components not routed in the first version

- local and global attention remain structurally exact;
- RMSNorm, residual, RoPE, and the PLE gate remain in the original graph;
- complete Transformer blocks are not replaced with general decision trees;
- hierarchical softmax is excluded because it changes the model's probability factorization.

### 8.6 MTP after base correctness

The published Gemma 4 MTP drafter is a small four-layer model with dimension 256, three local layers, one global layer, and cross-attention to main-model K/V. It requires no independent prefill and can propose an arbitrary draft length.

MTP may convert main-model verification into a small batch and improve arithmetic intensity. Acceptance rate, memory cost, and verification cost are unknown. Draft lengths 2, 3, and 4 will be tested only after single-token decode agrees with the oracle and steady-state execution uses no swap.

## 9. Implementation plan

Implementation has started with a deterministic miniature early-local decoder
layer. This is a graph and numerical bring-up artifact. It is not checkpoint
inference and is not a performance result.

### Implemented scalar slice

The current C path covers:

```text
input RMSNorm
Q/K/V projection
Q/K RMSNorm and V RMSNorm
local RoPE
causal sliding-window multi-query attention
attention output normalization and residual
pre-MLP normalization
GELU-tanh gated MLP
MLP output normalization and residual
PLE gate and projection
PLE output normalization and residual
layer scalar
```

The committed fixture uses three positions, hidden size 8, two query heads, one K/V
head, head dimension 4, intermediate size 12, PLE size 4, and sliding window
2. It is deliberately too small for performance measurement.

Validation results:

| Environment | Build | Result |
|---|---|---|
| macOS arm64 | Clang C11, optimized and ASan/UBSan variants | 10 of 10 declared tensor boundaries passed; maximum absolute error 0 |
| Ubuntu 24.04 x86-64 | GCC 13.3, static `x86-64-v3` binary | 10 of 10 declared tensor boundaries passed; maximum absolute error 0 |

Committed fixture SHA-256:
`e9e5db23c57fe09a3dac1863e199abf5cb787404e8727c25d3942160da2c15f7`.
The validated Linux binary SHA-256 was
`262d7291bcd1d056a3ee63ec63a47218c0e58ea20a016aef466d7db727772574`;
the binary is a generated artifact and is not committed.

### Official layer-0 weight validation

The offline range fetcher read 73,443,610 bytes from the pinned
10,246,621,918-byte checkpoint. It selected all 17 layer-0 records, the PLE
projection normalization, two main-embedding rows, two token-PLE slices, and
the layer-0 block of the context PLE projection. A separate Linux validation
downloaded the complete 10,246,621,918-byte checkpoint and independently
verified SHA-256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
Exporting from that complete file produced a byte-identical fixture to the
range extraction.

The input token IDs were 2 and 1. The exported layer has the published E2B
dimensions: hidden size 1536, eight query heads, one K/V head, head dimension
256, intermediate size 6144, PLE size 256, and sliding window 512. The actual
checkpoint layer scalar is 0.017822265625.

The 144,857,148-byte fixture has SHA-256
`2844d6c61f46d9b5a7ec51b89210c137c1b94aee7a098c31e4716a0fafe0a732`.
Independent local and Linux exports produced the same hash. The scalar C runner
passed all ten tensor boundaries with maximum absolute error 0 on both systems.
The Linux correctness run used 145,752,064 bytes maximum RSS, reported no swap,
and completed in 0.21 seconds. The macOS run used 146,407,424 bytes maximum RSS
and completed in 0.08 seconds. These single runs are correctness observations,
not inference benchmarks or token-throughput measurements.

The oracle expands the official BF16 records to FP32 and evaluates the pinned
layer equations with NumPy. It is not a direct BF16 Transformers forward pass.
The test does not validate a safety-task prompt, tokenizer, later local layers,
global proportional RoPE, shared K/V, final normalization, the LM head, or
autoregressive decode. Exact machine-readable results are in
`layer0-validation.json`.

### Stage 0: pin model and oracle data

1. Pin one Gemma 4 E2B IT checkpoint and tokenizer revision. Complete; hashes
   are recorded in `pins.json`.
2. Fix text-only mode, thinking disabled, context 512, and batch 1.
3. Export layer tensors from the reference implementation for small inputs: embedding, PLE, attention residual, MLP residual, PLE residual, final normalization, and selected logits.
4. Keep the reference framework outside the target artifact.

### Stage 1: offline model compiler

1. Read the pinned checkpoint.
2. Remove vision, audio, and first-stage MTP weights.
3. Fold context-aware and token PLE for text tokens.
4. Deduplicate the main embedding and LM head.
5. Reorder weights by the four layer classes; omit K/V records for layers 15-34.
6. Generate mixed-bit or additive codes, scales, zero-point corrections, and PLE codebooks.
7. Search late-MLP expert trees and exact/approximate vocabulary trees.
8. Calibrate every approximate candidate against the dense teacher.
9. Emit a model-specific image with fixed offsets and no general tensor-name graph.

### Stage 2: scalar C correctness runtime

1. Implement the tokenizer and pinned chat format.
2. Implement the packed-image loader.
3. Implement scalar FP32 RMSNorm, RoPE, attention, MLP, PLE, and LM head.
   RMSNorm, local RoPE, early-local attention, MLP, and the runtime PLE path
   are implemented for miniature and real-weight layer-0 fixtures. The LM head
   is not implemented.
4. Run complete prefill and decode at context 16.
5. Compare every declared tensor boundary with the oracle.

The scalar runtime is evaluated for correctness, not speed.

### Stage 3: SSE4.2 kernels

Replace scalar operations in this order:

1. quantized GEMV;
2. fused gate/up;
3. partial down-projection reduction;
4. Q/K/V and Q-only attention projections;
5. streaming LM head and top-k;
6. PLE vector decode and PLE tail;
7. normalization and residual fusion.

Each optimized kernel retains a scalar differential test.

### Stage 4: prefill, threading, and I/O

1. Stabilize batch-one decode.
2. Add a multi-token tiled prefill kernel.
3. Measure one, two, and four worker threads.
4. Map the two L2 instances before choosing CPU affinity.
5. Use sequential memory-map and prefetch policies for dense layer records.
6. Measure router recall, active-leaf count, and actual bytes read per token.
7. Do not lock the entire image into memory and do not permit eMMC paging during steady-state inference.

### Stage 5: MTP

Add and measure the published draft model only after base correctness and memory gates pass.

## 10. Resident-memory budget

The first packed-image target is:

```text
folded PLE codes and codebooks             220 MiB
Transformer weights                        460 MiB
tied embedding and LM head                 105 MiB
codebook, routing, scale, correction data   35 MiB
KV, activations, tokenizer, threads          60 MiB
unassigned margin                            80 MiB
---------------------------------------------------
target steady-state RSS                     960 MiB
hard rejection threshold                  1,024 MiB
swap                                             0
```

This is a design target, not an achieved result. RSS must be measured during sustained inference on the target machine. Model-file size and virtual address size are not substitutes.

Supporting arithmetic:

- 1.87B einsum parameters at a pure 2-bit payload require approximately 446 MiB;
- 402.7M tied embedding/head parameters at a pure 2-bit payload require approximately 96 MiB;
- the third-party approximately 0.26 GB PLE report corresponds to approximately 248 MiB;
- these payloads total approximately 790 MiB before codebooks, scales, alignment, routing, KV, arenas, and executable data.

The 220 MiB PLE allocation is tighter than the unverified 248 MiB reference. Any layer that requires 3- or 4-bit weights must be offset through lower precision elsewhere, codebook sharing, factorization, or validated parameter removal. Conditional inactivity does not reduce resident bytes.

## 11. Decode traffic model

This section is an analytical model under the following conditions:

```text
Gemma 4 E2B IT, text only
batch = 1
decode after prefill
context <= 512
four CPU cores
steady state
swap = 0
```

Intel lists the J3455 as a four-core part with 1.5 GHz base frequency, 2.3 GHz burst frequency, 2 MiB cache, at most two memory channels, and DDR3L/LPDDR3-1866 support. Intel lists 14.9 GB/s maximum memory bandwidth for the related NUC6CAYS platform.

The target machine is not that NUC. Its DMI table reports two 16-bit DDR3-1866 devices. If that width is accurate, the total physical data width is 32 bits:

```text
1866 MT/s * 4 bytes = 7.46 GB/s theoretical peak
                    = 6.95 GiB/s theoretical peak
```

Until a memory benchmark is run, this document uses 3.5 to 5.2 GiB/s as a working sustained-bandwidth interval. It does not use the platform maximum as a target-machine measurement.

### 11.1 Minimum weight payload per token

At a pure 2-bit payload:

| Component | Parameters | Payload |
|---|---:|---:|
| Early MLP, layers 0-14 | 424.7M | 101.3 MiB |
| Late MLP, layers 15-34 | 1,132.5M | 270.0 MiB |
| Attention and other einsums | 312.9M | 74.6 MiB |
| Transformer total | 1,870.0M | 445.8 MiB |
| Tied embedding and LM head | 402.7M | 96.0 MiB |

Scales, code indices, corrections, routing data, and alignment increase actual traffic. The scenarios below include an engineering allowance.

| Scenario | Active late MLP | LM-head rows | Estimated RAM traffic per token |
|---|---:|---:|---:|
| A: dense correctness fallback | 100% | 100% | 550-570 MiB |
| B: conservative expert routing | 50% | 100% | 415-435 MiB |
| C: target expert and vocabulary routing | 25% | approximately 10% | 260-280 MiB |
| D: aggressive expert and vocabulary routing | 12.5% | approximately 2% | 220-240 MiB |

PLE does not require a sequential scan of the full table. One token reads one PLE row, below 1 KiB on average in the proposed representation. Input embedding also reads one row. KV, normalization, routing, and activation traffic are smaller than the weight stream and are covered by the scenario allowance.

Scenarios C and D depend on approximate routing and require quality validation. A full-head fallback adds approximately 96 to 105 MiB and approximately 403M weight products for that token.

### 11.2 Bandwidth required by target decode rates

```text
required RAM bandwidth = bytes read per token * tokens per second
```

Using each scenario midpoint:

| Scenario | Traffic per token | At 3 tok/s | At 5 tok/s | At 8 tok/s |
|---|---:|---:|---:|---:|
| A | 560 MiB | 1.64 GiB/s | 2.73 GiB/s | 4.38 GiB/s |
| B | 425 MiB | 1.25 GiB/s | 2.08 GiB/s | 3.32 GiB/s |
| C | 270 MiB | 0.79 GiB/s | 1.32 GiB/s | 2.11 GiB/s |
| D | 230 MiB | 0.67 GiB/s | 1.12 GiB/s | 1.80 GiB/s |

At the assumed 3.5 to 5.2 GiB/s sustained range, bandwidth-only ceilings are:

| Scenario | Bandwidth-only ceiling |
|---|---:|
| A | approximately 6-9 tok/s |
| B | approximately 8-12 tok/s |
| C | approximately 13-20 tok/s |
| D | approximately 15-23 tok/s |

These are not attainable-rate predictions. They assign zero cost to unpacking, lookup, dot products, GELU, attention, routing, and reduction. The J3455 lacks AVX2, VNNI, FMA, FP16, and BF16 arithmetic. Instruction throughput is expected to bind before these memory ceilings.

### 11.3 CPU-aware decode estimate

Scenario A processes approximately 2.27B weight products per token, including the full Transformer and LM head. Scenario C still processes approximately 1.06B active weight products per token. Codebook lookup can replace part of the multiplication work with partial-sum lookup, but it adds index decoding and accumulation.

Before target microbenchmarks, the working estimates are:

| Scenario | RAM ceiling | CPU-aware sustained estimate | Primary risk |
|---|---:|---:|---|
| A | 6-9 tok/s | 0.8-1.8 tok/s | Full GEMV and full LM head |
| B | 8-12 tok/s | 1.3-2.8 tok/s | LM head still processes 403M products |
| C | 13-20 tok/s | 2.5-4.5 tok/s | Router recall and LUT/SSE efficiency |
| D | 15-23 tok/s | 3.5-6.0 tok/s | Quality loss and fallback rate |

Working engineering thresholds:

```text
minimum usable floor        1.0 tok/s sustained decode
engineering target          3.0 tok/s sustained decode
stretch target              5.0 tok/s sustained decode
```

Three tokens per second is the current point estimate, not a commitment. It requires approximately scenario C: 25% average late-MLP activation, about 10% average LM-head row evaluation, acceptable routing quality, and effective kernels. Five tokens per second requires behavior closer to scenario D, infrequent fallback, an effective LUT implementation, and stable CPU frequency under load.

Prefill is excluded. It uses multi-token tiled matrix multiplication and requires a separate model after the decode kernels exist.

## 12. Storage model

The normal design loads the approximately 960 MiB model image before inference and keeps it resident:

```text
required steady-state eMMC bandwidth = 0 MiB/s
```

Sequential storage speed affects cold start:

| Sequential read rate | Ideal time to read 960 MiB |
|---:|---:|
| 25 MiB/s | approximately 38 s |
| 50 MiB/s | approximately 19 s |
| 100 MiB/s | approximately 9.6 s |
| 200 MiB/s | approximately 4.8 s |

Fifty MiB/s is the minimum acceptable cold-start rate; 100 MiB/s is the target. One image requires at least approximately 1.5 GiB free. Two candidates, a temporary update, and rollback require at least 4 GiB; 8 GiB is operationally safer.

Cold expert leaves on eMMC are excluded from the normal design. At 3 tok/s, 20 MiB of cold reads per token requires 60 MiB/s, while 50 MiB requires 150 MiB/s before random-access penalties across 20 late layers. This cannot support a predictable rate on the target storage.

## 13. First validation workload

The initial workload is delayed danger assessment, not interactive chat. It accepts a finite observation or event window and returns a risk decision after bounded processing delay.

Current constraints:

- no requirement for token-by-token interactive output;
- no long conversation history;
- one decision consumes one finite observation window;
- the primary output is a risk result, not open-ended continuation;
- short aggregation, periodic evaluation, or event-triggered execution is permitted;
- the final latency limit, input data types, and risk taxonomy remain undefined.

Sustained decode rate is an implementation diagnostic. The product metric is:

```text
time to decision = input preparation
                 + prompt prefill
                 + risk classification
                 + optional bounded explanation
```

### 13.1 Fixed output path

An early interface may use a small structured result set such as:

```text
SAFE
REVIEW
DANGER
```

These labels are examples. The risk taxonomy and thresholds have not been defined.

Two candidate output implementations are:

1. retain the Gemma final hidden state and evaluate only the vocabulary rows for the allowed labels;
2. distill a small `1536 -> N` classification head from the full model and labeled data.

The first remains closer to the checkpoint. The second is cheaper at runtime but creates a new derived classifier and requires independent validation. Full input embedding may still be required for arbitrary input tokens even when full-vocabulary output scanning is removed.

Only `REVIEW` or `DANGER` may request a short bounded explanation. The normal `SAFE` path does not generate natural language.

### 13.2 Provisional latency bands

These bands are workload placeholders, not performance predictions:

| Band | End-to-end decision latency | Intended mode |
|---|---:|---|
| Fast check | 5-15 s | Short input, resident model, fixed label |
| Normal asynchronous check | 15-60 s | Longer input or one review pass |
| Background review | 1-5 min | Multiple windows, prompts, or a second explanation pass |

Required measurements include:

- prefill time for 32, 64, 128, 256, and 512 input tokens;
- fixed-label decision time;
- separate `SAFE` and explanation-path latency;
- cold-start and warm-resident execution;
- queue delay under accumulated events.

### 13.3 Safety boundary

Model output cannot be the only fail-safe. Deterministic sensor thresholds, device fault codes, emergency-stop conditions, and other explicit rules must trigger directly and cannot be overridden by a `SAFE` result.

The model path is intended for semantic combinations, event context, and delayed review not covered by deterministic rules. Its interface should include:

```text
risk_class
confidence_or_margin
reason_codes
evidence_references
model_and_runtime_version
timestamp
```

Natural-language explanation is not evidence. The system must retain references to the original observations used for each decision.

## 14. Bounded-input optimization

The first workload can constrain its instruction, schema, field types, length range, output schema, and risk taxonomy without fixing every input byte.

```text
fixed instruction prefix
+ one of a small number of templates
+ variable observation fields under a schema
+ fixed classification request
```

This profile is a compiler input at the same level as the checkpoint and CPU profile. The artifact records the supported profile revision. A profile change requires recompilation and reevaluation.

### 14.1 Exact fixed-prefix optimization

When the instruction is a causal prefix, the compiler can execute it offline under the compiled model and package its state:

```text
fixed prefix tokens
        |
offline prefill
        |
15 physical KV snapshots and position state
        |
runtime starts at the first variable token
```

The snapshot is bound to:

- checkpoint and quantization revision;
- input-profile and template revision;
- exact token IDs;
- prefix length and position convention;
- KV precision and layout;
- local and global attention type.

Text that follows a variable observation cannot be fully precomputed because its hidden state depends on that observation. Input layout may put the fixed instruction first only when doing so preserves the intended semantics.

Restricting output to a fixed label set also permits evaluation of only those LM-head rows. That is exact for a decoder explicitly defined over the label set. It is not equivalent to unrestricted open-vocabulary argmax.

### 14.2 Distribution-specific approximate optimization

Within the validated input distribution, the compiler may search:

- layer sensitivity and bit allocation;
- MLP clustering, routing, and active-leaf budgets;
- codebooks, low-rank corrections, and structured pruning;
- classification-head distillation;
- context buckets and thread schedules.

Expected hidden states may occupy a narrower distribution than general chat. Any additional pruning or routing benefit must be measured on held-out and boundary data, not only the search corpus.

Separate artifacts may be generated for different input lengths:

```text
artifact A: short event summary, at most 64 variable tokens
artifact B: normal observation window, at most 256 variable tokens
artifact C: extended review, at most 512 variable tokens
```

The first version may instead use one artifact for the most common window if multiple resident images are operationally expensive.

### 14.3 Conditions for vocabulary and PLE row pruning

A bounded input profile is not automatically a bounded token set:

- instruction, template, and label tokens are exactly known;
- enumerated fields and reason codes may have a finite token set;
- free-text observations may use most of the vocabulary;
- numbers, timestamps, and identifiers have large combinations even if their subtoken set is measurable.

Embedding or PLE rows may be removed only when the input schema prohibits free text or an allowed-token grammar proves that the rows are unreachable. Otherwise, the complete input embedding and PLE vocabulary remains in the image, or an explicit unknown/out-of-domain path is required.

The baseline 960 MiB budget does not assume this saving.

### 14.4 Out-of-domain handling

Before model execution, the runtime checks:

- template identifier and schema version;
- required fields;
- token count and context bucket;
- enumerated-value ranges;
- allowed-token grammar, if row pruning is enabled;
- agreement between the input-profile revision and artifact revision.

Invalid input is rejected, marked for review, sent to a more conservative artifact, or delegated to an external system. It must not enter a heavily specialized model silently.

## 15. Required work before implementation

1. Pin the checkpoint and tokenizer revisions.
2. Define the exact input schema and whether free text is allowed.
3. Define the risk taxonomy, actions, and hard deterministic rules.
4. Define false-negative, false-positive, and decision-latency requirements.
5. Define context buckets, update frequency, and queue limits.
6. Define the output schema and explanation policy.
7. Establish disjoint compiler-search, held-out, boundary, and distribution-shift datasets.
8. Export the reference tensor oracle.
9. Measure target memory bandwidth, storage throughput, thread scaling, and sustained clock rate.
10. Recalculate the image and traffic budgets from actual packed formats.

## 16. Primary references

### Model and implementation

- [Gemma documentation](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 technical report](https://arxiv.org/abs/2607.02770)
- [Gemma 4 E2B configuration](https://huggingface.co/google/gemma-4-E2B/blob/main/config.json)
- [Gemma 4 E2B IT QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Gemma 4 QAT Q4_0 collection](https://huggingface.co/collections/google/gemma-4-qat-q4-0)
- [Transformers Gemma 4 model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [Transformers Gemma 4 configuration implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py)
- [Gemma 3 1B IT model card](https://huggingface.co/google/gemma-3-1b-it)
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)

### Compression and conditional execution

- [AQLM](https://arxiv.org/abs/2401.06118)
- [LUT-GEMM](https://arxiv.org/abs/2206.09557)
- [MoEfication](https://arxiv.org/abs/2110.01786)
- [Deja Vu](https://arxiv.org/abs/2310.17157)
- [PowerInfer](https://arxiv.org/abs/2312.12456)
- [Tree-based maximum inner-product search](https://arxiv.org/abs/1202.6101)
- [Third-party Gemma 4 E2B PLE quantization experiment](https://huggingface.co/TheStageAI/gemma-4-E2B-it)

The third-party PLE checkpoint is not a Google checkpoint and has not been reproduced by this project.

### Hardware

- [Intel Celeron J3455 specifications](https://www.intel.com/content/www/us/en/products/sku/95594/intel-celeron-processor-j3455-2m-cache-up-to-2-30-ghz/specifications.html)
- [Intel NUC6CAYS specifications](https://www.intel.com/content/www/us/en/products/sku/95078/intel-nuc-kit-nuc6cays/specifications.html)
