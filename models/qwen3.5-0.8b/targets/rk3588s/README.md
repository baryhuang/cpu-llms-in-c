# Target: Rockchip RK3588S

Status: target plan recorded. No RK3588S board is pinned, no artifact exists, and this repository has not measured this target.

## Target pin

| Field | Value |
|---|---|
| SoC | Rockchip RK3588S |
| Chip year | 2022 — RK3588S official EVB sale announced 2022-03-03 |
| CPU | 4x Cortex-A76 + 4x Cortex-A55 |
| NPU | 6 TOPS, triple core; INT4, INT8, INT16, FP16, BF16, TF32 |
| Memory interface | four-channel LPDDR4/LPDDR4X/LPDDR5 |
| Board RAM, storage, OS | not selected |
| Runtime boundary | generated C graph/runtime; Rockchip userspace/driver is an explicit dependency when the NPU is used |

Sources: [Rockchip RK3588 specifications](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html) and [official RK3588/RK3588S EVB sale notice](https://www.rock-chips.com/a/cn/news/rockchip/2022/0303/1544.html). The year is the official EVB availability event, not a later board manufacture date.

## External reference baseline

These are not measurements by this repository. Rockchip reports them for RK3588 at maximum CPU/NPU frequency, sequence length 128, 64 new tokens, and `optimization_level=0`.

| Model | Data path | TTFT | Decode | Reported memory |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | W8A8 | 587.74 ms | 27.05 tokens/s | 1039.66 MB |
| Qwen3.5-2B | W8A8 | 776.95 ms | 13.59 tokens/s | 2121.91 MB |
| Gemma 4 E2B | W8A8 | 599.06 ms | 11.12 tokens/s | 2498.70 MB |

Source: [RKLLM benchmark, revision `878f936`](https://github.com/airockchip/rknn-llm/blob/878f9361fd3afa7e167b7079918918f78d2c1c2a/benchmark.md). RKLLM v1.3 lists Qwen3.5 and Gemma 4 support in its [pinned release record](https://github.com/airockchip/rknn-llm/tree/878f9361fd3afa7e167b7079918918f78d2c1c2a). RK3588 is used as a planning proxy for the shared RK3588-series compute block; the exact RK3588S board remains a separate measurement target.

## NPU use for this model

| Graph region | Initial backend | Reason |
|---|---|---|
| Q/K/V and attention output projections | NPU | large dense matrix operations |
| MLP gate/up/down projections | NPU | dominant dense matrix traffic |
| DeltaNet input, gate, value and output projections | NPU | batched projection work is regular |
| DeltaNet convolution and recurrent state update | CPU first | small sequential state operations; per-op NPU dispatch can dominate |
| RMSNorm, RoPE, tokenizer, sampling and answer-set selection | CPU | small or control-heavy operations |

Do not dispatch every tensor operation separately. We first lower whole projection groups with persistent shared buffers. A later compiler pass may move a complete DeltaNet or attention block only after boundary tests show that removing synchronization is worth the added kernel work.

Fixed or schema-bounded prompt prefixes are compiled independently of the task output. The compiler may precompute the prefix's KV/DeltaNet state and start each call from that snapshot. This reduces prefill without hardwiring binary classification or any other output set.

## Target-specific plan

| Step | Change | Required evidence |
|---:|---|---|
| 1 | Run official RKLLM Qwen3.5-0.8B only as an external-stack board baseline | exact board, SDK, frequencies, RSS, TTFT, prefill and decode |
| 2 | Run the model's generic C runtime on Cortex-A76/A55 | tensor agreement and CPU-only baseline |
| 3 | Lower one projection group through Rockchip's `rknn_matmul_api` | output differential test and dispatch timing |
| 4 | Emit NPU-native weight order and create contexts/buffers once | zero repacking or allocation in token loop |
| 5 | Search static work partition across the three NPU cores | per-layout timing and core utilization |
| 6 | Compile fixed-prefix state snapshots for declared prompt templates | exact replay comparison against full prefill |
| 7 | Search per-layer W8/W4 choices with outlier handling | held-out quality, perplexity and memory |
| 8 | Fuse larger graph regions only where synchronization remains material | end-to-end improvement, not kernel-only timing |

The public repository must not vendor proprietary Rockchip SDK headers or libraries without a license that permits redistribution. Generated code owns the model graph, tokenizer, prompt contract and scheduling; the vendor runtime is a target dependency, not the model runtime.

## Current conclusion

The NPU is useful primarily for projection-heavy prefill and for energy efficiency. An independent experimental [Rockchip NPU backend](https://github.com/invisiofficial/rk-llama.cpp/tree/rknpu2/ggml/src/ggml-rknpu2) reports large prefill gains on RK3588 but shows that small-model single-token decode can remain CPU-competitive. Therefore the acceptance metric is end-to-end TTFT, decode, power and quality, not the 6 TOPS label.

The official Qwen3.5-0.8B W8A8 reference already reports 1039.66 MB. It is not a safe fit for a physical 1 GB system after OS and runtime memory. Select a board with more RAM or establish a lower-bit path before treating this target as deployable.
