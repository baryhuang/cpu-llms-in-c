# Qwen3.6-27B Apple M3 Pro handoff

Date: 2026-08-14 (Asia/Shanghai), updated after the streaming increment
verified and landed.

This is a factual handoff for continuing work on this target. It is not a
release record. Published numbers live in `README.md`, `REVIEW.html` and
`results.json`; do not copy provisional measurements into those files until
their stated checks pass.

## Objective

Implement Qwen3.6-27B inference as a model- and Apple-M3-Pro-specific C/Metal
runtime. The deployment must contain no Python, PyTorch, `llama.cpp`, ONNX
Runtime, MLX, or `mlx-lm`. A thin Objective-C system bridge for Metal is
allowed; model execution and orchestration remain C interfaces.

Streaming ultimately covers the full input/output pipeline:

```text
input bytes -> C tokenizer -> chunked prefill -> decode -> C detokenizer -> output
```

Stages should overlap where dependencies allow. Do not describe dependent
stages as fully parallel: decode for one request cannot begin before that
request's prefill state exists, and decode token N+1 depends on token N.

## User constraints and decisions

- Write the deployment in C with Metal kernels; no Python in the deployed path.
- Optimize for the pinned model and pinned hardware, beyond a general framework.
- Keep free-text generation. Binary classification is not a hard-coded goal.
- Prompt/task restrictions may be compiler inputs, but the runtime must not be
  permanently restricted to one baked test prompt.
- Treat encode, prefill, decode, detokenize, and output as a streaming pipeline.
- Learn from `mlx-lm`; do not blindly duplicate framework machinery.
- Keep benchmark timing separate from correctness verification.
- Record exact inputs, visible outputs, token IDs, durations, CPU/memory data,
  and raw review evidence.
- Do not include credentials or model weights in Git.
- Do not use remote Claude. Work in this local repository and on the local Mac.
- Do not touch the unrelated untracked `openmemory.sqlite` file.

## Landed: streaming increment (2026-08-14)

The first streaming increment is implemented, verified and committed:

- `qwen36_m3_decode.h/.m`: `qwen36_m3_model_forward_submit` /
  `qwen36_m3_model_forward_wait` around exactly one in-flight Metal command.
  The synchronous `qwen36_m3_model_forward` remains as a wrapper. Reset and
  close drain a pending command. Double submit and wait-without-submit are
  checked errors.
- `tools/qwen36_m3_generate.c`: samples on CPU, submits the next forward,
  flushes the newest complete UTF-8 suffix to the descriptor named by
  `QWEN36_STREAM_FD`, then waits. JSON schema 2 adds per-token
  `stream_emitted_after_prompt_start_ms` and a `streaming` object.
  Descriptor 1 is rejected for streaming to protect the JSON contract.
- `tools/qwen36_chat.sh` (formerly root `qwen36-chat.sh`): streams
  descriptor 3; `jq` no longer required;
  `QWEN36_RAW=1` preserves raw JSON.
- `tests/qwen36_m3_api_state_test.c` + `make qwen36-m3-api-state-test`:
  16-check live API-state test (needs the model images; not part of the
  fixture-only `make test`). 16/16 pass.

Verification of the increment (raw data in `results.json`
`streaming_increment`):

- Controlled A/B: 13 fresh-process runs, 1 discarded warmup, 4 rounds x
  3 variants (parent d99ef9f sync, async no-stream, async stream) with
  rotated order. 12/12 runs identical token IDs and text. Means 8.545635 /
  8.576762 / 8.540470 decisions/s; max mean gap 0.037 vs per-variant round
  spread 0.115-0.254. No regression; the earlier informal 7.679 reading was
  run-to-run variance.
- Streaming runs: 30 flushes per 30 visible tokens, inter-flush P50
  116.317-117.166 ms, P95 121.173-125.286 ms, total CPU decode-and-flush
  0.115-0.211 ms per run.
- The stream decoder re-decodes the visible prefix each step: O(T^2) in
  generated length, measured at most 0.211 ms total at 30 tokens. Replace
  with a stateful incremental detokenizer when generation lengths grow.

## Landed: batched prefill (2026-08-14, second increment)

- `qwen36_prefill.metal` + `qwen36_m3_model_prefill`: S32/S16 chunk graphs
  specialized through a Metal function constant, one-token tail under 16.
  Batched Q4 GEMM reads each weight group once per chunk; GDN conv and the
  delta rule keep time sequential inside blocked kernels; attention chunks
  run causal batched attention on the shared KV cache. The final prompt
  token stays on the logits-producing decode forward.
- `QWEN36_PREFILL=0` restores the sequential path;
  `QWEN36_PREFILL_MAX_CHUNK=16` restricts to the S16 bucket.
- Fixed a latent threadgroup race in the decode attention softmax reduction
  (missing barrier between the max read and the sum-reduction overwrite);
  ordering-only change, both kernels.
- Verified: S16 bitwise state+logits parity; S32 argmax-identical with
  fast-math drift <= 0.0841 abs and zero NaN
  (`make qwen36-m3-prefill-parity-test`); token parity on 5/5 smoke prompts
  and 8/8 interleaved A/B runs; TTFT 11,547.6 -> 9,725.9 ms mean
  (raw data in `results.json` `prefill_increment`).
- Do not chase bitwise parity across kernel shape variants: fast-math FMA
  contraction differs per specialization. The published gate is end-to-end
  token parity plus bounded state drift, matching the C-vs-oMLX standard.

## Model and target facts

| Item | Value |
|---|---:|
| Model | Qwen3.6-27B text path |
| Target | Apple M3 Pro, 14-core Metal 3 GPU, 36 GB unified memory |
| Transformer layers | 64 |
| Gated DeltaNet layers | 48 |
| Full-attention layers | 16; every fourth layer |
| Hidden width | 5,120 |
| MLP width | 17,408 |
| Deployed weights | affine Q4, group size 64, FP16 scale and bias |
| Mapped model image | 15,138,643,968 bytes |
| GDN recurrent/conv state | 158,859,264 bytes |
| FP32 attention KV at capacity 128 | 16,777,216 bytes |

The reported 215-277 MB RSS/physical-footprint values are not the total model
memory cost. File-backed Metal-visible model pages are accounted separately.
The real machine-memory pressure is approximately the 15.139 GB model plus
state, KV, workspace, process memory, and resident file-backed pages.

## Framework distinction

Do not conflate these layers in future explanations:

```text
MLX      = Apple's array/graph/C++/Metal execution framework
mlx-lm   = Apple's LLM model and generation library built on MLX
oMLX     = a separate third-party server that depends on and extends mlx-lm
```

The published comparison in this repository was run against the complete oMLX
stack, not bare `mlx-lm`. Its numbers must remain labelled oMLX. Techniques can
still be learned from the open `mlx-lm` implementation.

## Techniques learned from mlx-lm

### Suitable for this runtime

| Technique | C/Metal translation |
|---|---|
| Separate full-sequence prefill from one-token decode | Landed: Q4 GEMM chunk graphs beside the Q4 GEMV decode graph |
| Preserve the sequence dimension through model layers | Landed: `[S, 5120]` activations through the chunk graph |
| Chunk long prefill | Landed for S16/S32 with a one-token tail; S64/S128 open |
| Async evaluate the next decode step | Landed: submit/wait decode API |
| Use model-specific cache types | Separate GDN conv/recurrent state from attention KV state |
| Reuse prompt prefix state | Snapshot all layer states at compiler-approved prefix boundaries |
| Reuse allocation storage | Perform tensor-lifetime analysis and emit a fixed static arena |
| Direct quantized matrix execution | Dequantize Q4 inside GEMV/GEMM kernels, never materialize full FP16 weights |

### Not suitable as the first step

- oMLX's Paged SSD cache, multi-model LRU pool, web server, and scheduler are
  not needed for the current single-stream target.
- A large generic allocator cache is contrary to the static-arena goal.
- Continuous batching is useful later for multiple concurrent streams, but it
  must not increase single-stream inter-token latency.
- Speculative decoding/MTP waited until a batched verification path and an
  acceptance/quality gate existed; both landed and MTP is now implemented
  (fourth increment below).

## Streaming dependency model

The intended pipeline is overlap, not dependency violation:

```text
CPU tokenize chunk N+1     || GPU prefill chunk N
CPU detokenize/output N    || GPU decode token N+1
decode request A           || bounded prefill chunk for request B
```

For multiple requests, decode ticks should have priority over long prefill.
Long prompts must be split so an active output stream is not stalled for
seconds by one monolithic prefill command.

Inside one model layer, combine projections sharing the same input rather than
issuing several small competing kernels:

```text
GDN: packed Q/K/V/Z/A/B -> convolution -> blocked recurrence -> gated norm -> output
Attention: packed Q/K/V -> QK norm + RoPE -> causal attention -> output
MLP: packed gate/up -> activation -> down
```

## Landed: prefill kernel optimization + weight residency (2026-08-14, third increment)

- Tiled simdgroup-matrix GEMM for the prefill projections: 32x32 output
  tile per 128-thread threadgroup, 64-wide K blocks (one Q4 group) staged
  in threadgroup memory, inline dequantization, 8x8
  `simdgroup_multiply_accumulate` with float accumulators. Warm S32 chunk
  2,035 -> 864 ms (2.35x). `QWEN36_PREFILL_MMA=0` keeps the exact
  decode-identical kernels; the parity test verifies both modes.
- `MTLResidencySet` with all 644 mapped weight buffers, committed and
  attached to the queue at model open: TTFT after open 8,116 -> 1,389 ms;
  wiring moves into model open (6.5 s one-time); cold total slightly
  better (7,882 vs 8,183 ms). `QWEN36_RESIDENCY=0` disables.
- Headline: TTFT after ready 1,389 ms = 1.91x faster than the oMLX
  server's published 2,655.8 ms; ready-state prompt throughput 25.9 tok/s
  (oMLX 13.56, bare mlx-lm 27.3); cold-start total at parity with both.
- Technique sources: the tiled-MMA prefill shape matches the 2026 BaseRT
  paper (arxiv 2607.00501) and MLX's own quantized kernels; the root
  cause insight (activation re-reads scaling with batch in the naive
  batched GEMM) came from our own traffic arithmetic.
- Considered and deferred: indirect-command-buffer pre-encoding of the
  static decode graph (compile-time graph precompute). Measured CPU
  encode is roughly 2 ms/token, under 2 percent of decode; not worth the
  kernel-signature churn until it dominates.

## Measured negative result: CPU prefault does not help cold TTFT

Touching all mapped image pages from 8 parallel CPU threads at model open
(plus `MADV_WILLNEED`) took 7.1 s and did NOT shrink the first S32 chunk,
which still took 11.2 s; the control without prefault ran model open in
0.12 s and the first chunk in 8.9 s. Total time to first token got about
7 s worse. Conclusion: the first-chunk cost is Metal's first-use residency
wiring of the `newBufferWithBytesNoCopy` mapped buffers, not process page
faults, and CPU touching cannot satisfy it. The change was reverted. A
future attempt should target GPU-side residency (for example
`MTLResidencySet` on macOS 15) or accept the cost as a one-time
per-process constant that amortizes in a server process.

## Tuning state of the batched kernels

Per-layer GPU profiling (QWEN36_PROFILE=1/2, one command buffer per
layer on the serial queue) attributed the chunk to kernel classes and
showed the float MMA path was FP32-ALU-bound (~13-14 ms/layer on BOTH
layer types — so the GEMMs, not the recurrence, dominated). The
half-MMA path (half tiles, device-direct activation fragments,
per-thread float accumulators spilled every 64 columns) brings chunk32
to 576 ms GPU (delta layers avg 9.3 ms, attention layers avg 8.2 ms).
Remaining headroom sits in tile staging/barrier overhead over a ~300 ms
half-ALU bound plus the ~115 ms streaming floor: candidates are
double-buffered weight tiles and wider K tiles. Decode itself measures
118-123 ms GPU (delta48 ~87, attn16 ~27, head ~5.5) against a ~101 ms
streaming floor — near the memory bound, which is why MTP is the decode
lever. The batch-2 MTP verify measures ~168 ms with the +50 ms over a
single forward spread uniformly across both layer types (batch-2 GEMV
ALU), snapshot blit only 2.5 ms.

## Landed: MTP speculative decoding (2026-08-14, fourth increment)

Greedy, output-lossless speculation using the checkpoint's `mtp.*` head.
Each step drafts one token (fused embed+norm kernel, fc, one attention
layer at index 64, shared Q4 head; ~8 ms) and verifies pending+draft in
one batch-2 forward (~168 ms including a GDN state snapshot blitted
inside the same command buffer; a reject restores the snapshot and
re-verifies). Batch<=2 GEMM uses the exact decode kernels
(`use_mma && batch > 2`), so verify arithmetic matches plain decode.
Prefill fills the draft layer's KV cache per chunk with tokens shifted
one position; with MTP open, `qwen36_m3_model_prefill` requires
token_count + 1 entries in token_ids.

Measured (fresh-process A/B, parity battery): identical output 4/4;
accepts 15/15 on code (8.26 -> 11.98 tok/s, 1.45x), 186/223 = 83 % on
446-token prose (8.39 -> 9.55 tok/s, 1.14x). Chat auto-enables when
`MODEL_DIR/mtp-layer.q36att` + `mtp.q36mtp` exist and sampling is
greedy; `QWEN36_MTP=0` disables. serve.py defaults are greedy, so the
Chatbox path uses it.

Multi-step drafting landed on top (same day): the single MTP layer runs
recursively (chain hidden = previous step's post-norm MTP hidden, per
the reference implementation), one batch-(depth+1) verify, partial
accepts restore the GDN snapshot and re-verify the accepted prefix plus
the correction. Depths measured on the battery (decode tok/s
code/prose): off 8.4/8.4, depth1 11.93/9.60, depth2 14.18/8.83,
depth3 15.40/8.10, adaptive default 13.49/9.49. Adaptive = per-draft
acceptance EMA (alpha 0.15, thresholds 0.90/0.95), reset at prompt
start; QWEN36_MTP_DEPTH fixes 1..3. Output token-identical to plain
greedy at every depth. Two verify-kernel experiments measured and
rejected (small-batch MMA tile; half-math GEMV at 196 vs 168 ms) — the
exact float GEMV stays, keeping verify bitwise-anchored to decode.

Hard-won fact, encoded in `tools/qwen36_mtp_pack.c`: the seven `mtp.*`
norm vectors are HF delta weights (GemmaRMSNorm, effective multiplier
1 + w) even though every main-model norm in the deployed conversion is a
direct multiplier. Packed direct, drafts are garbage (0/40 accepts);
with 1 + w folded, 31/40 on the Python reference and 15/15 on the code
prompt in C. Isolated with an independent MLX reference
(vLLM/SGLang sources confirm; Transformers ignores mtp.*). Source
tensors: BF16 assembly of shards 13+15, SHA-256 pinned in
`qwen36_m3_image.h` (`713b0faf...`); images are generated artifacts, not
committed.

## Work not yet implemented

- Per-dispatch profiling of the remaining 864 ms warm chunk (recurrence,
  softmax, elementwise kernels).
- Indirect-command-buffer pre-encoding of the static decode graph
  (about 2 ms/token CPU encode today; deferred until it matters).
- Batched kernel tuning: the warm S32 chunk spends about 1.4 s of compute
  on 32 tokens; profile GEMM tiling and the blocked recurrence.
- S64/S128 prompt buckets for long prompts, after tuning.
- Input-byte streaming tokenizer API.
- Token ring buffer between tokenizer and prefill.
- Decode sampling/argmax on GPU; CPU sampling is still a serial dependency.
- GPU-to-GPU token ID -> embedding chaining.
- More than one in-flight command/workspace.
- Multi-request decode-priority scheduler and continuous batching.
- Prefix-state snapshots.
- Stateful O(1)-per-token incremental detokenizer.

## Recommended continuation order

1. Profile the S32 chunk (GPU counters or per-dispatch timing) and tune the
   dominant batched kernels; re-run the prefill parity test after every
   kernel change. CPU prefaulting was tried and rejected; see the negative
   result above. The same profiling applies to the batch-2 MTP verify
   (~168 ms vs ~118 ms single forward): closing that gap raises the MTP
   speedup toward its accept-rate bound (1.83x at 83 % accept).
2. Add S64/S128 buckets once tuned kernels justify them; verify with the
   same parity gates.
3. Continue the streaming pipeline: input-byte tokenizer API, token ring
   buffer, GPU sampling, prefix-state snapshots.
4. Update docs/results only with measured incremental effects and exact raw
   commands. Commit and push after verification.

## Relevant files

| File | Role |
|---|---|
| `tools/qwen36_chat.sh` | human-facing shell; interactive mode execs the resident chat binary |
| `tools/qwen36_m3_chat.c` | resident chat: model opens and wires once, prompt loop at ready-state latency; `QWEN36_MACHINE=1` switches to a JSON-line protocol (R/D/E/X lines) for serving shims |
| `tools/qwen36_serve.py` | OpenAI-compatible `/v1/chat/completions` server over the resident chat; multi-turn template rendering, SSE streaming, stdlib-only |
| `tools/qwen36_monitor.py` | live CPU/memory/GPU monitor with sparkline dashboard, JSONL recording and HTML chart rendering |
| `tools/qwen36_m3_generate.c` | tokenizer, prompt loop, sampler, stream output, JSON report |
| `qwen36_m3_decode.h` | public sync and async C runtime API |
| `qwen36_m3_decode.m` | Objective-C Metal runtime implementation |
| `qwen36_q4.metal` | Q4 matrix kernels |
| `qwen36_layer.metal` | GDN/MLP layer kernels |
| `qwen36_attention.metal` | full-attention kernels |
| `qwen36_prefill.metal` | batched S32/S16 prefill kernels and the MTP fuse kernel |
| `qwen36_m3_mtp_image.h` | MTP extras image format (fc + three norms) |
| `tools/qwen36_mtp_pack.c` | BF16 -> Q4 packer for the two MTP images; folds the HF delta-norm convention |
| `tests/qwen36_m3_api_state_test.c` | live async API state-machine test |
| `tests/qwen36_m3_prefill_parity_test.c` | live prefill-vs-decode state and logits parity test |
| `results.json` | published machine-readable record, includes `streaming_increment` |
| `REVIEW.html` | published human review |

All target source paths above are relative to
`models/qwen3.6-27b/targets/apple-m3-pro/` except the root shell, the tool
and the test.
