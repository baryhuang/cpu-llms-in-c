# Qwen3.6-27B on Apple M3 Pro

Status: free-text generation runs end to end. The deployment opens compiled images, tokenizes one UTF-8 user prompt, renders the pinned no-thinking chat template, prefills the prompt through batched S32/S16 graphs, executes the complete 64-layer text graph, samples, and decodes text. Generated text streams incrementally while the next GPU forward is in flight. It does not load Python, MLX, llama.cpp, or a C++ runtime.

## Target pin

| Property | Value |
|---|---|
| Machine | MacBook Pro `Mac15,6` |
| SoC | Apple M3 Pro, 2023 |
| CPU | 5 performance + 6 efficiency cores |
| GPU | 14 cores, Metal 3 |
| Unified memory | 36 GB / 38,654,705,664 bytes |
| OS used for measurement | macOS 15.7.3, build 24G419 |
| Metal compiler | Apple Metal 32023.864 |
| Model input | `mlx-community/Qwen3.6-27B-4bit` revision `c000ac2c2057d94be3fa931000c31723aac53282` |

## Implemented path

| Stage | Implementation | Fixed at compile time |
|---|---|---|
| Source import | C safetensors reader and SHA-256 validation | repository revision, three shard hashes, tensor names, shapes and dtypes |
| Model image | one global image, 48 Delta layer images, 16 attention layer images | text-only graph, Q4 group-64 layout, direct norm convention, layer order |
| Tokenizer image | C tokenizer compiler and C runtime using system ICU for Unicode classes | pinned tokenizer JSON and special-token table |
| Chat input | one arbitrary UTF-8 user string | official `enable_thinking=false` template |
| Graph | fixed 64-layer loop: three GatedDeltaNet layers then one full-attention layer, repeated | all dimensions and kernel dispatches |
| State | FP32 Delta recurrent state, convolution history and FP16 attention KV cache | head counts, head dimensions and cache stride |
| Output | complete 248,320-row Q4 language-model head | padded IDs above tokenizer vocabulary are masked |
| Sampling | greedy or temperature/top-k sampling in C | vocabulary bound and stop IDs |
| Speculation | optional greedy MTP draft-and-verify, output-lossless | draft layer graph and batch-2 verify graph |
| Decode | C tokenizer decode to UTF-8 | special-token behavior |

The `.m` files are thin Objective-C calls into the Apple Metal system API. Model control, tokenizer, sampling, image compilers and public runtime interfaces are C; compute kernels are Metal. `otool -L` reports only Foundation, Metal, CoreFoundation, `libicucore`, `libSystem` and `libobjc`.

## Artifact

| Item | Value |
|---|---:|
| Global plus 64 layer images | 65 files |
| Mapped model bytes | 15,138,643,968 |
| Tokenizer image | 9,781,808 bytes |
| Compiled-image manifest SHA-256 | `dadbd9457bcfa809befa2f28087cb89be495de4063b5ebb16da7e58ed6771d63` |
| Generator SHA-256 | `92e61119cb58c10595544ade204642bbd793461e824c8fcd9a7d5b2e0ff0db53` |
| Metallib SHA-256 | `d43fe32c1082161f181acd6cdf3eb59bf12c08a5fdebe493fd183d8aac8c25bf` |

All 65 local model-image hashes matched the remote compiled-artifact manifest. Images and checkpoint shards are generated artifacts and are not committed.

## Deployment size and resident memory

### Standalone install

Measured on disk. The deployed path carries no Python, PyTorch,
llama.cpp or ONNX Runtime; it links only the macOS system frameworks
(Metal, Foundation, CoreFoundation, libicucore).

| Component | Bytes |
|---|---:|
| Compiled model images, 65 files, Q4 group-64 | 15,138,643,968 |
| Tokenizer image | 9,781,808 |
| `qwen36-m3-q4.metallib` GPU kernels | 229,760 |
| `qwen36-m3-chat` + `qwen36-m3-generate` binaries | 237,760 |
| Serving, monitor and setup scripts | 56,127 |
| **Total** | **15,148,949,423 (15.15 GB)** |
| Optional MTP draft images (`mtp-layer.q36att` + `mtp.q36mtp`) | +238,993,408 (+1.6 %) |

Everything except the model images totals 10.3 MB.

| Target machine requirement | Value |
|---|---|
| Hardware | Apple Silicon with Metal 3 |
| OS | macOS 15+ for `MTLResidencySet` wiring at model open; older macOS skips it and pays wiring in the first prefill chunk instead |
| Dependencies to install for the core runtime | none |
| Optional tool dependencies | system `python3` (standard library) for `qwen36_serve.py` and `qwen36_monitor.py`; Chatbox client |
| Resident memory while serving | 15.139 GB wired weights + ~0.5 GB process footprint at context 4096 |
| Practical machine memory | 24 GB minimum, 32 GB or more comfortable |
| Performance claims | pinned to the M3 Pro target; results do not transfer to other chips |

### Reading the monitor while the server is resident

`tools/qwen36_monitor.py` attached to the resident engine on the 36 GB
pin, model loaded, idle between requests:

| Column | Typical reading | Meaning for this runtime |
|---|---:|---|
| foot | ~0.5 GB | process-private memory: 0.27 GB FP16 KV cache at capacity 4096 + 0.16 GB GDN recurrent/conv state + workspace |
| rss | near 0 | not meaningful here; the mapped weights are file-backed and never counted in RSS |
| wired | ~20 GB | macOS baseline of roughly 4 GB plus the 15.14 GB weight residency; drops back the moment the server exits |
| filebk | 2–3 GB | page cache for other files; wired weights are not in this column |
| anon | varies | other applications' heap memory |
| compr | varies | compressed inactive memory; grows when total demand exceeds machine memory |
| free | 0–1 GB | macOS keeps free memory near zero by design; not a signal on its own |
| pressure | normal / warn | warn near the boundary is expected with the model resident next to large applications |

Attributable total while serving: about 16 GB, constant across
requests. `pkill -f qwen36_serve.py` releases the wired weights
immediately; the engine child exits with the server.

## End-to-end example

Command:

```sh
build/qwen36-m3-generate MODEL_DIR build/qwen36-m3-q4.metallib \
  MODEL_DIR/tokenizer.q36tok 128 64 0 1 42 \
  'Write a C function int max2(int a, int b) that returns the larger value. Output only the function.'
```

| Field | Raw value |
|---|---|
| Input | `Write a C function int max2(int a, int b) that returns the larger value. Output only the function.` |
| Prompt tokens | 36 |
| Output token IDs | `71093 66 198 390 1866 17 1494 264 11 514 292 8 313 198 262 460 318 64 835 292 8 907 264 535 292 26 198 92 198 71093` |
| Stop token | `248046` |
| Output | <pre>```c
int max2(int a, int b) {
    return (a &gt; b) ? a : b;
}
```</pre> |

The code is semantically correct. It fails the strict formatting instruction because the model emitted Markdown fences. Raw prompts, outputs and timings for all five cases are in [`REVIEW.html`](REVIEW.html).

## Timed benchmark

All duration fields below are part of the timed path. Correctness checks are listed separately in the next section.

The same-machine comparison uses the prompt above, the same 36 prompt IDs, greedy sampling, the same 30 visible output IDs and the same text. The oMLX checkpoint was exported from the deployed C images so both paths see the same Q4 payloads and deployed metadata values.

| Metric | C/Metal runtime | oMLX 0.5.7 | Ratio |
|---|---:|---:|---:|
| Process/model startup | 97.968 ms image open | 4,971.697 ms engine start | different loading models |
| TTFT after model open/request | 11,303.791 ms | 2,655.758 ms | C is 4.256x slower |
| Prompt throughput, 36 tokens | 3.185502 tok/s | 13.555450 tok/s | C is 4.255x slower |
| C prompt throughput after cold first page touch | 8.389157 tok/s | not separately exposed | — |
| Decode, 29 intervals between 30 visible tokens | **8.422709 tok/s** | **5.850744 tok/s** | **C is 1.439596x faster** |
| Request continuation including the EOS decision | 3,568.987 ms / 8.405747 decisions/s | EOS timestamp not exposed | not compared |
| Process real / user / system time | 15.09 / 0.03 / 2.16 s | 14.43 / 2.96 / 5.52 s | includes startup and request |
| Model tensor memory | 15.139 GB file-mapped | 15.135 GB MLX active + 0.827 GB MLX cache | both exclude comparison-stack code pages |
| Peak process physical footprint | 0.277 GB plus file-backed mapped pages | 16.376 GB | accounting models differ; see Memory |

The table above is the published baseline record from before batched prefill
landed; it ran the one-token graph 36 times. With the batched prefill,
tiled-GEMM kernels and weight residency described below, time to first token
after the model is ready measures 1,389.0 ms mean — 1.91x faster than the
oMLX server's 2,655.8 ms on the same prompt — and the cold-start total is
7,882 ms, at parity with the mlx-lm and oMLX stacks.

oMLX 0.5.7 was built with its native `qwen35_prefill` extension available. Its default Q4 and FA256 routes start at 2,048 prompt tokens, so those native long-prefill kernels did not engage for this 36-token case.

### Published upstream native reference

The Qwen model card does not publish a token-throughput number for Qwen3.6-27B. It recommends vLLM or SGLang and shows an eight-GPU tensor-parallel serving configuration, but gives no hardware-specific speed result. The closest published upstream implementation result is in the official Hugging Face Transformers Qwen3.5/3.6 documentation.

| Metric | This C/Metal run | Official Transformers native path |
|---|---:|---:|
| Model | Qwen3.6-27B text-only Q4 group-64 | `Qwen/Qwen3.6-27B` BF16 |
| Hardware | Apple M3 Pro, 14-core GPU, 36 GB | NVIDIA GB10 / SM121 |
| Prompt / generation | 36 / 30 visible tokens | 1,024 / 256 tokens |
| Mode | greedy, no thinking | greedy |
| TTFT | 11.304 s | 1.11 s with the official GDN Hub kernel; 1.66 s fallback |
| Decode | **8.423 tok/s** | 4.14 tok/s with the official kernel; 4.11 tok/s fallback |
| Reported memory | 15.139 GB mapped model; 0.277 GB process footprint plus file-backed pages | not reported |

No speedup ratio is calculated: the hardware, precision, prompt length, generated length and memory accounting differ. Sources: [Qwen official model card](https://huggingface.co/Qwen/Qwen3.6-27B) and [official Transformers Qwen3.5/3.6 implementation notes](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md#usage-tips-and-notes).

### Same-window three-way measurement

One session, all three stacks on the same value-equivalent checkpoint and
the same 36-token prompt, greedy: one discarded warmup per stack, then
four rounds with the stack order rotated each round, fresh process per
run, all wall-clock. All 12 measured runs produced the published output.
Ready means model open (C), library load (mlx-lm) or engine start (oMLX);
TTFT is request start to first visible output. oMLX streams coalesced
events, so its decode rate uses first-to-last arrival time.

| Metric, mean of 4 | C/Metal (this repo) | Bare mlx-lm 0.31.3 | oMLX 0.5.7 |
|---|---:|---:|---:|
| Ready (open / load / engine start) | 7.093 s | 3.925 s | 4.671 s |
| TTFT after ready | **1.385 s** | 2.598 s | 3.113 s |
| Cold start to first token | 8.478 s | **6.524 s** | 7.784 s |
| Visible decode | **8.479 tok/s** | 6.028 tok/s | 5.898 tok/s |
| Prompt throughput after ready | **26.0 tok/s** | 13.9 tok/s | 11.6 tok/s |

The C runtime decodes 1.41x faster than bare mlx-lm and 1.44x faster than
oMLX, and answers a ready model 1.88x and 2.25x sooner respectively. The
one loss is cold start: the C ready phase carries the one-time 7 s weight
wiring, which amortizes in a long-lived process. Raw data:
`same_window_three_way` in [`results.json`](results.json).

### Bare mlx-lm reference

The published same-machine baseline above is the complete oMLX server
stack. This section measures the bare `mlx-lm` library (mlx-lm 0.31.3,
mlx 0.32.0, Python 3.13, comparison-only environment) on the same
value-equivalent exported checkpoint: one discarded warmup, then four
rounds of bare mlx-lm and the C runtime interleaved with alternating
order, fresh process each run, the same 36-token prompt, greedy. All
eight runs produced the published 36 prompt IDs and 30 visible output IDs
plus the stop token.

| Metric | C/Metal, same rounds | Bare mlx-lm 0.31.3 | Same-window ratio |
|---|---:|---:|---|
| Model load | 0.101 s mmap open | 4.588 s | different loading models |
| First token after load | 9.716 s | 3.030 s | — |
| Cold start to first token | 9.816 s | 7.618 s | C 1.289x slower |
| Visible decode intervals | **7.735 tok/s** | 5.660 tok/s | **C 1.367x faster** |
| Reported prompt throughput | 3.71 tok/s cold-inclusive | 27.33 tok/s | see note |
| Peak process memory | 0.27 GB plus mapped pages | 15.89 GB | accounting differs |

These rounds ran on a thermally loaded machine: the C decode measured
7.735 tok/s here versus its published 8.4227 tok/s from the cooler
baseline session, and mlx-lm measured 5.660 tok/s versus the oMLX
server's 5.8507 tok/s; the same-window ratios are the comparable facts.
mlx-lm reported 27.3 prompt tok/s with a warm page cache versus 13.16 in
its own cold warmup and 13.56 through the oMLX server; the C
cold-inclusive figure carries the roughly 7.4 s first-use weight wiring
described above, while the warm C S32 chunk sustains about 15.7 tok/s
(32 tokens in 2.035 s). Raw data: `bare_mlx_lm_baseline` in
[`results.json`](results.json).

### Memory

| Metric | C/Metal runtime | oMLX 0.5.7 |
|---|---:|---:|
| Model tensor bytes | 15,138,643,968 file-mapped bytes | 15,134,588,968 MLX active bytes |
| Recurrent plus convolution state | 158,859,264 bytes | included in MLX allocation accounting |
| KV cache | 16,777,216 bytes at FP32 capacity 128 | included in MLX allocation accounting |
| MLX cache | none | 826,530,040 bytes |
| Peak process physical footprint | 277,030,400 bytes | 16,376,235,264 bytes |
| Maximum RSS from `/usr/bin/time -lp` | 215,007,232 bytes | 10,625,630,208 bytes |

The C footprint and RSS rows do not include all file-backed model pages as anonymous process memory. They must be read together with the 15,138,643,968 mapped-byte row; they are not claims that the model consumes only 277 MB of machine memory.

### Complete layer medians

Five independent processes were run for each layer. Each process used two warmups and 20 measured iterations.

| Layer | Scope | Median duration | Effective packed-weight rate |
|---|---|---:|---:|
| Delta layer 0 | complete one-token layer, including projections, convolution, recurrent update, gated norm, MLP and residuals | 1.840846 ms | 117.211300 GB/s |
| Attention layer 3 | complete one-token layer at context 1, including RoPE, KV update, attention, MLP and residuals | 1.770885 ms | 118.263171 GB/s |

## Streaming delivery

Decode is an asynchronous submit/wait pair around exactly one in-flight Metal
command; the model owns one workspace, so a second submit before wait is a
checked error rather than a race.

```c
qwen36_m3_model_forward_submit(model, token, position, ...);
/* CPU decodes and flushes the newest visible text while the GPU runs */
qwen36_m3_model_forward_wait(model, &result, &logits, &logit_count, ...);
```

The generator samples on CPU, submits the next forward, flushes the newest
complete UTF-8 suffix to the file descriptor named by `QWEN36_STREAM_FD`, then
waits. Machine-readable JSON keeps standard output to itself; descriptor 1 is
rejected for streaming so the two contracts never share a byte stream.
`tools/qwen36_chat.sh` passes descriptor 3 and no longer requires `jq`. The
synchronous `qwen36_m3_model_forward` remains as a wrapper.

### Measured effect

Thirteen fresh-process runs in one sitting: one discarded warmup, then four
rounds of three variants with the variant order rotated each round to cancel
thermal and load drift. Same model images, byte-identical metallib, the same
36-token prompt as the timed benchmark, greedy seed 42. All 12 measured runs
produced identical token IDs and identical text.

| Variant | Decode decisions/s, mean of 4 rounds | Rounds |
|---|---:|---|
| Parent commit `d99ef9f`, synchronous forward | 8.545635 | 8.5665 / 8.5885 / 8.3870 / 8.6405 |
| Async submit/wait, streaming off | 8.576762 | 8.5568 / 8.6154 / 8.6450 / 8.4899 |
| Async submit/wait, streaming on | 8.540470 | 8.5405 / 8.5891 / 8.5586 / 8.4737 |

The three means differ by at most 0.037 decisions/s while the per-variant
spread across rounds is 0.115 to 0.254 decisions/s: the async API and the
stream flush change decode throughput by less than run-to-run variance. An
earlier informal 7.679 decisions/s reading from the first streaming smoke run
is attributed to that variance, not to the API change.

Streaming behavior across the four streaming runs:

| Metric | Value |
|---|---:|
| Flushes per 30 visible tokens | 30 |
| First streamed text after prompt start | 11,228.909 to 11,789.067 ms |
| Inter-flush interval P50 | 116.317 to 117.166 ms |
| Inter-flush interval P95 | 121.173 to 125.286 ms |
| Total CPU decode-and-flush work per run | 0.115 to 0.211 ms |

First text is still dominated by sequential prompt processing: the cold
first forward plus 35 one-token forwards. The batched prefill graph in
Current limits remains the controlling fix. The stream decoder re-decodes the
visible token prefix each step, which is O(T^2) in generated length but
measured at most 0.211 ms total for 30 tokens; a stateful incremental
detokenizer replaces it when generation lengths grow.

## Batched prefill

The prompt no longer replays the one-token decode graph per token. A
`qwen36_m3_model_prefill` call pushes prompt tokens through compiled batch
graphs: S32 chunks first, then an S16 chunk, then one-token forwards for a
tail shorter than 16. The final prompt token stays on the normal decode
forward so its logits feed the sampler. Batched Q4 GEMM kernels read each
weight group once per chunk instead of once per token; the GDN convolution
and delta-rule recurrence keep their time dependency inside blocked
sequential kernels; the attention chunk performs causal batched attention
against the shared KV cache. `QWEN36_PREFILL=0` restores the sequential
path and `QWEN36_PREFILL_MAX_CHUNK=16` restricts chunking to the S16 bucket.

### Measured effect

Nine fresh-process runs: one discarded warmup, then four rounds of
prefill-on and prefill-off with the order alternating each round; the same
36-token published prompt, greedy seed 42. All eight measured runs produced
identical token IDs and text.

| Metric | Prefill off | Prefill on |
|---|---:|---:|
| Time to first token after model open, mean of 4 | 11,547.6 ms | **9,725.9 ms (1.187x faster)** |
| First weight-touching unit, includes cold page faults | 7,370.1 ms first forward | 8,831.5 ms first S32 chunk, 32 tokens |
| Prompt work after that unit | 4,177.1 ms, 35 sequential forwards | 894.2 ms, 3 tail forwards plus the final prompt forward |
| Decode decisions/s | 8.3588 | 8.2408; gap 0.118 vs combined round spread 0.39 |

### Numeric parity

`build/qwen36-m3-prefill-parity-test` compares every persistent layer state
buffer and the next-token logits between prefilled and sequentially pushed
token runs of 16, 19, 32, 35 and 48 tokens:

- S16-only runs are bitwise identical in all 256 state buffers and logits.
- S32 runs decide the identical next token in every case; their states show
  fast-math drift bounded by 0.0841 absolute on state magnitudes around 20,
  with zero NaN. The cause is FMA contraction differing between the unrolled
  batch positions of the S32 kernel specialization and the one-token kernel;
  end-to-end token parity is the published gate, matching the C-vs-oMLX
  comparison standard.
- Prefill on versus off produced identical token IDs and text on all five
  published smoke prompts.

Building the batched graph also exposed a latent threadgroup race in the
decode attention softmax kernel: the same threadgroup array carried the max
and sum reductions with no barrier between reading the maximum and
overwriting slot 0. The 24-threadgroup decode dispatch never manifested it;
the 24-by-batch prefill dispatch did. Both kernels now carry the barrier,
which changes ordering only, not arithmetic.

### Optimized kernels and weight residency

Two further increments, both measured with rotated-round A/B and gated by
the same parity standard:

**Tiled simdgroup-matrix GEMM.** The first-generation batched GEMM kept
decode-identical arithmetic but re-read every batch activation row from
device memory once per weight group per simdgroup, multiplying activation
traffic by the batch size. The tiled path stages a [batch x 64] activation
tile and a dequantized [64 x 32] weight tile in threadgroup memory once per
threadgroup and consumes them with 8x8 `simdgroup_multiply_accumulate`
operations — the standard prefill kernel shape on Apple GPUs. The warm S32
chunk drops from 2,035 ms to 864 ms for 32 tokens (2.35x).
`QWEN36_PREFILL_MMA=0` restores the exact decode-identical kernels.

**Half-precision MMA tiles.** Per-layer GPU profiling (`QWEN36_PROFILE=1`
splits the graph into one command buffer per layer and reads the GPU
timestamp spans) showed the float MMA chunk is FP32-ALU-bound, not
bandwidth-bound: ~13-14 ms per layer for a 32-token chunk against a
~1.8 ms weight-streaming floor, on both layer types. The half path
dequantizes weight tiles to half, reads activation fragments directly
from device memory with strided `simdgroup_load` (float activations are
converted once into a half scratch), and multiplies with half 8x8 MMAs;
per-thread float accumulators take a spill every 64 columns, which
bounds the half accumulation window. Measured chunk32 GPU time drops
from 889 ms to 576 ms (1.54x; 3.53x over the first-generation batched
kernels), warm time-to-first-token from 1.115 s to 0.806 s. Parity:
state drift bounded by 0.249, argmax identical on every prefill-parity
run, end-to-end token IDs and text identical on the full battery with
and without MTP. `QWEN36_PREFILL_MMA=1` restores the float MMA path,
`=0` the exact decode-identical kernels.

**Weight residency at model open.** A CPU prefault of the mapped pages was
measured and rejected (see Current limits history): the first-chunk cost is
Metal first-use residency wiring, not page faults. Adding all 644 mapped
weight buffers to an `MTLResidencySet` committed, requested and attached to
the command queue at model open moves that wiring out of the first chunk.
`QWEN36_RESIDENCY=0` disables it.

Interleaved four-round A/B, MMA prefill active in both arms, identical
token IDs and text in all eight runs:

| Metric | Residency off | Residency on |
|---|---:|---:|
| Model open | 67.4 ms | 6,493.5 ms, absorbs wiring |
| Time to first token after open, mean of 4 | 8,116.0 ms | **1,389.0 ms** |
| First S32 chunk | 7,249.1 ms | 905.3 ms |
| Cold start to first token, total | 8,183.2 ms | 7,882.4 ms |
| Decode decisions/s | 8.4521 | 8.3910 |

Prompt throughput once the model is ready: 36 tokens in 1,389 ms is
25.9 tok/s versus 13.56 tok/s for the oMLX server and 27.3 tok/s for bare
mlx-lm on this machine. In a long-lived process the one-time 6.5 s wiring
at open amortizes across requests.

## MTP speculative decoding

Qwen3.6-27B ships a multi-token-prediction head (`mtp.*` tensors: an fc
projection, one standard full-attention decoder layer and three extra RMS
norms, sharing the main embedding table and language-model head).
Transformers ignores these tensors; vLLM and SGLang use them for
speculative decoding. This runtime uses them the same way, greedy-only
and lossless: each step drafts one token with the cheap MTP pass, then
verifies the pending token and the draft in one batch-2 forward through
the main model. An accepted draft yields two tokens for roughly one
main-model pass plus overhead; a rejected draft is replaced through a
one-token re-verify, so emitted tokens are always bitwise identical to
plain greedy decoding. That identity is the gate, checked per release.

| Piece | Implementation |
|---|---:|
| Draft input | `fc([rms(embed(token)); rms(hidden_post_final_norm)])`, fused embed+norm kernel |
| Draft body | one standard attention layer (layer index 64) with its own KV cache, then MTP final norm and the shared Q4 head |
| Verify | batch-2 prefill graph through all 64 layers plus a batched two-row head |
| State rollback | GDN recurrent and convolution states snapshot into a shadow buffer inside the verify command buffer; a reject blits them back. Attention KV needs no rollback: the re-verify overwrites the same cache rows |
| Draft KV during prompt | filled by the same S32/S16 prefill chunks, shifted one position |

### Packing the draft weights

`make qwen36-mtp-pack` builds the packer; it reads the `mtp.*` tensors
(assembled from ranged reads of official shards 13 and 15, pinned SHA-256
`713b0faf…`) and emits `mtp-layer.q36att` (a standard attention image,
209,436,672 bytes) plus `mtp.q36mtp` (fc and norms, 29,556,736 bytes) —
238,993,408 bytes total, +1.6 % on the model. One convention trap cost a
day and is now encoded in the packer: the seven `mtp.*` norm vectors are
Hugging Face delta weights (GemmaRMSNorm, effective multiplier `1 + w`),
while every norm in the main checkpoint conversion is a direct
multiplier. Packed as direct weights the drafts are garbage (0/40
accepts); with `1 + w` folded at pack time the same battery accepts
31/40. The finding was isolated with an independent MLX reference
implementation before touching the C path.

### Measured effect

Fresh-process A/B, same battery as the parity gate, greedy seed 42.
`qwen36-m3-chat` enables MTP automatically when both images sit next to
the model and sampling is greedy; `QWEN36_MTP=0` disables it. The decode
rate is tokens divided by time after the first token.

| Prompt | Accepted drafts | Decode, MTP off | Decode, MTP on | Speedup |
|---|---:|---:|---:|---:|
| C `max2` function, 30 tokens | 15/15 (100 %) | 8.26 tok/s | 11.98 tok/s | 1.45x |
| Hash-table explanation, 446 tokens | 186/223 (83 %) | 8.39 tok/s | 9.55 tok/s | 1.14x |

Structured output (code) accepts nearly everything; free prose accepts
about 83 %. Per step the draft pass costs ~8 ms and the batch-2
verify (snapshot included) ~168 ms against ~118 ms for a plain one-token
forward. Resident cost of MTP: 239 MB more mapped weights and one more
attention layer's KV cache (34 MB at context 4096).

### Multi-step drafting

The reference implementations run the single MTP layer recursively
(Qwen recommends three speculative tokens for this family), each step
consuming the previous step's post-norm MTP hidden. This runtime chains
up to three drafts, verifies pending plus all drafts in one
batch-(depth+1) forward, and on a partial accept restores the
pre-verify GDN snapshot and re-verifies the accepted prefix plus the
corrected token — so output stays token-identical to plain greedy at
every depth (verified on the full battery). Deeper chains trade verify
rows and replays against acceptance:

| Decode tok/s | MTP off | depth 1 | depth 2 | depth 3 | adaptive (default) |
|---|---:|---:|---:|---:|---:|
| C function, 30 tokens | 8.4 | 11.93 | 14.18 | **15.40** | 13.49 |
| Prose, 446 tokens | 8.4 | **9.60** | 8.83 | 8.10 | 9.49 |

Per-draft acceptance falls with chain position (prose: 83 % at depth 1,
71 % cumulative at 2, 57 % at 3), so a fixed deep chain loses on prose.
The default is adaptive: a per-draft acceptance EMA (alpha 0.15,
deepen above 0.90/0.95) keeps prose at depth 1 within noise of its
optimum and lets code climb to depth 3. `QWEN36_MTP_DEPTH=1..3` fixes
the depth; `0` is the adaptive default.

Two batch-2..4 verify kernel experiments were measured and rejected:
an 8x8-MMA tile variant (barrier/staging bound at small batch) and a
half-math GEMV (196 ms vs 168 ms at batch 2 — Metal scalar half is not
faster here). The verify keeps the exact decode-identical float GEMV
kernels, which also keeps verify logits bitwise-anchored to decode.

## Verification

Verification is outside the timed benchmark above.

| Gate | Result |
|---|---|
| Compiled artifact transfer | 65/65 local model-image SHA-256 entries matched the remote compiled artifact |
| Tokenizer | five English, chat-template, contraction/newline, Chinese and NFC cases matched the pinned official Rust tokenizer |
| Same prompt, C vs oMLX | 36/36 token IDs matched |
| Same visible generation, C vs oMLX | 30/30 token IDs and decoded text matched |
| Delta layer 0 output vs independent C | maximum absolute error `1.49011612e-6` |
| Delta recurrent state vs independent C | maximum absolute error `1.69873238e-6` |
| Delta convolution state vs independent C | maximum absolute error `1.09672546e-5` |
| Attention layer 3 output vs independent C | maximum absolute error `9.65595245e-6` |
| Attention q / k / v vs independent C | `1.07288361e-5` / `1.04904175e-5` / `6.91413879e-6` |
| Five free-text smoke cases | semantic 5/5; strict requested format 4/5 |
| Async decode API state machine | 16/16 checks: double submit, wait without submit, invalid token, position bound, reset and close with a pending command (`make qwen36-m3-api-state-test`, needs the model images) |
| Streaming A/B token parity | 12/12 runs identical IDs and text across parent, async and streaming variants |
| Prefill state parity, exact path | S16 bitwise identical in all 256 layer-state buffers and logits; S32 argmax-identical with drift bounded by 0.0841 and zero NaN (`make qwen36-m3-prefill-parity-test`) |
| Prefill state parity, tiled-GEMM path | argmax-identical on all five token runs, drift bounded by 0.108, zero NaN |
| Prefill end-to-end token parity | identical IDs and text on 5/5 smoke prompts and 8/8 A/B runs, repeated after each kernel change |
| MTP end-to-end token parity | MTP on vs off produced identical IDs and text on 4/4 battery prompts including a 446-token generation |
| MTP draft correctness vs independent reference | fused embed+norm kernel bitwise vs CPU oracle; fc within Q4 quantization error; accept behavior matches an MLX reference implementation |

The norm oracle uses the deployed checkpoint convention: standard RMSNorm and q/k norm tensors are direct multiplicative weights, not Hugging Face-style delta weights. Delta q/k normalization uses the exact 128-dimensional epsilon algebra.

## Build and reproduce

No Python command is used to compile or run the deployment.

For an interactive prompt loop after the image exists:

```sh
tools/qwen36_chat.sh
```

Interactive mode runs the resident chat binary (`make qwen36-m3-chat`):
the model opens and wires once at startup, then every prompt answers at
the ready-state latency (about 1.4 s to first token) with layer state
reset between prompts. Enter a prompt at `You>`; text streams under
`Model>`. Enter `/quit` to exit. One-shot and `QWEN36_RAW=1` invocations
keep the per-run generator and its JSON contract.

`tools/qwen36_monitor.py` (stdlib-only, no root) samples the run's CPU,
RSS, physical footprint, GPU utilization and the system's
free/wired/file-backed/compressed memory and pressure level. On a
terminal it draws a live sparkline dashboard; `--log` prints plain lines;
`--record run.jsonl` saves samples and `--render run.jsonl` turns a
recording into a standalone HTML page with SVG line charts (utilization,
process memory, system memory, with pressure intervals shaded). The
weight-wiring phase and the GPU-bound generation phase are clearly
separable in the charts. Remember the accounting note above: this
runtime's mapped weights appear in the wired/file-backed columns, not in
its RSS.

```sh
make qwen36-tools qwen36-m3-generate

tools/qwen36_compile_text_image.sh \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  tokenizer.json MODEL_DIR

build/qwen36-m3-generate MODEL_DIR build/qwen36-m3-q4.metallib \
  MODEL_DIR/tokenizer.q36tok 128 64 0 1 42 \
  'What is 2+2? Answer with only the number.'
```

The compile script is restartable: an existing image is validated by the runtime and is not overwritten. Every source shard is checked against the pinned SHA-256 before tensor import.

For MTP speculative decoding, assemble the `mtp.*` tensors into one
BF16 safetensors file (they live in official shards 13 and 15; ranged
HTTP reads suffice) and pack:

```sh
make qwen36-mtp-pack
build/qwen36-mtp-pack qwen36-mtp-bf16.safetensors \
  MODEL_DIR/mtp-layer.q36att MODEL_DIR/mtp.q36mtp \
  713b0fafacc94c9e541925872de3bcc3507cf5af73abbce525298467b8b4b10f
```

The packer refuses any source file whose SHA-256 differs from the pinned
value and folds the Hugging Face delta-norm convention (`1 + w`) into
the stored vectors. The chat binary picks the images up automatically.

The value-equivalent checkpoint used for the same-machine baseline can be reconstructed in C:

```sh
build/qwen36-m3-export-omlx MODEL_DIR model.safetensors
```

The exporter hard-checks 1,847 tensors and 15,132,802,048 tensor-data bytes. The baseline itself uses Python because oMLX is the comparison stack; Python is not in the deployment path.

## Current limits

| Limit | Consequence | Next work |
|---|---|---|
| One-time weight wiring at open | 6.5 s of MTLResidencySet wiring per process; CPU prefault was measured and rejected before this | amortizes in a long-lived process; batch it against other startup work if a server lands |
| Warm S32 chunk at 576 ms | profiling shows the half-MMA GEMMs still carry staging/barrier overhead over their ~300 ms compute bound | double-buffered weight tiles, wider K tiles |
| Per-token CPU encoding of the static decode graph | roughly 2 ms per token, under 2 percent of decode | pre-encode with indirect command buffers if it ever dominates |
| Prompts under 16 tokens | still run the sequential one-token path | add smaller buckets only if short-prompt TTFT matters |
| Single user-message CLI | free text works, but system and multi-turn message APIs do not | expose a message-array C API without changing the graph |
| FP16 KV cache | 64 KiB per context token | verify a Q8 cache path if longer contexts need it |
| Text-only image | vision inputs are unsupported | separate artifact if vision is required |
| MTP verify pass at ~168 ms vs ~118 ms single forward | speedup is 1.45x on code, 1.14x on prose instead of the accept-rate bound | tune batch-2 attention and GDN kernels; the GEMM path already uses the exact decode kernels |
| MTP is greedy-only | temperature or top-k sampling disables speculation | lossless sampled speculation needs rejection sampling against full distributions |
| Five smoke prompts only | not a general quality score | run a pinned standardized text benchmark |
| No 1K or 4K profile | short-prompt results do not establish long-context behavior | measure after batched prefill lands |

Machine-readable measurements are in [`results.json`](results.json).
