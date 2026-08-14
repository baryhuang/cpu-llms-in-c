# Qwen3.6-27B on Apple M3 Pro

Status: free-text generation runs end to end. The deployment opens compiled images, tokenizes one UTF-8 user prompt, renders the pinned no-thinking chat template, executes the complete 64-layer text graph, samples, and decodes text. It does not load Python, MLX, llama.cpp, or a C++ runtime.

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
| State | FP32 Delta recurrent state, convolution history and FP32 attention KV cache | head counts, head dimensions and cache stride |
| Output | complete 248,320-row Q4 language-model head | padded IDs above tokenizer vocabulary are masked |
| Sampling | greedy or temperature/top-k sampling in C | vocabulary bound and stop IDs |
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

The C TTFT result is not competitive yet. It runs the one-token graph 36 times; oMLX submits a batched prompt graph. The first C forward also takes 7,129.150 ms because it faults the mapped model pages into the process. The remaining 35 prompt forwards take 4,172.052 ms.

oMLX 0.5.7 was built with its native `qwen35_prefill` extension available. Its default Q4 and FA256 routes start at 2,048 prompt tokens, so those native long-prefill kernels did not engage for this 36-token case.

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

The norm oracle uses the deployed checkpoint convention: standard RMSNorm and q/k norm tensors are direct multiplicative weights, not Hugging Face-style delta weights. Delta q/k normalization uses the exact 128-dimensional epsilon algebra.

## Build and reproduce

No Python command is used to compile or run the deployment.

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

The value-equivalent checkpoint used for the same-machine baseline can be reconstructed in C:

```sh
build/qwen36-m3-export-omlx MODEL_DIR model.safetensors
```

The exporter hard-checks 1,847 tensors and 15,132,802,048 tensor-data bytes. The baseline itself uses Python because oMLX is the comparison stack; Python is not in the deployment path.

## Current limits

| Limit | Consequence | Next work |
|---|---|---|
| Sequential prompt processing | TTFT is 4.256x slower than oMLX on the measured 36-token prompt | emit a separate batched prefill graph |
| Single user-message CLI | free text works, but system and multi-turn message APIs do not | expose a message-array C API without changing the graph |
| FP32 KV cache | 128 KiB per context token | verify FP16, then Q8 cache paths |
| Text-only image | vision inputs are unsupported | separate artifact if vision is required |
| MTP omitted | one target token decision per model forward | add only after independent acceptance and quality gates |
| Five smoke prompts only | not a general quality score | run a pinned standardized text benchmark |
| No 1K or 4K profile | short-prompt results do not establish long-context behavior | measure after batched prefill lands |

Machine-readable measurements are in [`results.json`](results.json).
