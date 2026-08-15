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
- `qwen36-chat.sh`: streams descriptor 3; `jq` no longer required;
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
- Speculative decoding/MTP should wait until a batched verification path and an
  acceptance/quality gate exist.

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

## Work not yet implemented

- Cold-start weight wiring: about 7.4-8.9 s of the ~9.9 s cold TTFT is
  Metal first-use residency wiring of the mapped weight buffers, now the
  controlling limit. See the negative result above before retrying.
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
   result above.
2. Add S64/S128 buckets once tuned kernels justify them; verify with the
   same parity gates.
3. Continue the streaming pipeline: input-byte tokenizer API, token ring
   buffer, GPU sampling, prefix-state snapshots.
4. Update docs/results only with measured incremental effects and exact raw
   commands. Commit and push after verification.

## Relevant files

| File | Role |
|---|---|
| `qwen36-chat.sh` | human-facing prompt shell; streams descriptor 3 |
| `tools/qwen36_m3_generate.c` | tokenizer, prompt loop, sampler, stream output, JSON report |
| `qwen36_m3_decode.h` | public sync and async C runtime API |
| `qwen36_m3_decode.m` | Objective-C Metal runtime implementation |
| `qwen36_q4.metal` | Q4 matrix kernels |
| `qwen36_layer.metal` | GDN/MLP layer kernels |
| `qwen36_attention.metal` | full-attention kernels |
| `qwen36_prefill.metal` | batched S32/S16 prefill kernels |
| `tests/qwen36_m3_api_state_test.c` | live async API state-machine test |
| `tests/qwen36_m3_prefill_parity_test.c` | live prefill-vs-decode state and logits parity test |
| `results.json` | published machine-readable record, includes `streaming_increment` |
| `REVIEW.html` | published human review |

All target source paths above are relative to
`models/qwen3.6-27b/targets/apple-m3-pro/` except the root shell, the tool
and the test.
