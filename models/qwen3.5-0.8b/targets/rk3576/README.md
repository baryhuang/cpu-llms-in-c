# Target: Rockchip RK3576

Status: target plan recorded. No RK3576 board is pinned, no artifact exists, and this repository has not measured this target.

## Target pin

| Field | Value |
|---|---|
| SoC | Rockchip RK3576 |
| Chip year | 2024 — officially released in March 2024 |
| CPU | 4x Cortex-A72 + 4x Cortex-A53, up to 2.2 GHz |
| NPU | 6 TOPS RKNN at INT8; supports Transformer-model operators |
| Memory interface | 32-bit LPDDR4/LPDDR4X/LPDDR5 |
| Board RAM, storage, OS | not selected |
| Runtime boundary | generated C graph/runtime; Rockchip userspace/driver is an explicit dependency when the NPU is used |

Sources: [Rockchip RK3576 specifications](https://www.rock-chips.com/a/en/products/RK35_Series/2024/1212/2033.html), [official brief datasheet](https://www.rock-chips.com/uploads/pdf/2024.3.18/191/RK3576%20Brief%20Datasheet.pdf), and the [Rockchip 2023 annual report](https://www.rock-chips.com/uploads/pdf/2024.4.15/178/%E7%91%9E%E8%8A%AF%E5%BE%AE%EF%BC%9A2023%E5%B9%B4%E5%B9%B4%E5%BA%A6%E6%8A%A5%E5%91%8A.pdf), which states that RK3576 was officially released in March 2024. The year is the chip launch, not a board manufacture date.

## External reference baseline

These are not measurements by this repository. Rockchip reports them at maximum CPU/NPU frequency, sequence length 128, 64 new tokens, and `optimization_level=0`.

| Model | Data path | TTFT | Decode | Reported memory |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | W4A16 | 1369.31 ms | 18.79 tokens/s | 689.50 MB |
| Qwen3.5-0.8B | W4A16 group-128 | 1392.09 ms | 17.88 tokens/s | 687.38 MB |
| Qwen3.5-0.8B | W8A8 | 1342.54 ms | 13.18 tokens/s | 1047.08 MB |
| Qwen3.5-2B | W4A16 | 1661.01 ms | 11.03 tokens/s | 1236.12 MB |
| Gemma 4 E2B | W4A16 | 1219.25 ms | 9.23 tokens/s | 1463.42 MB |

Source: [RKLLM benchmark, revision `878f936`](https://github.com/airockchip/rknn-llm/blob/878f9361fd3afa7e167b7079918918f78d2c1c2a/benchmark.md). The result fixes the first implementation choice: W4A16, not W8A8, is the initial RK3576 NPU path for the approximately 1 GB resident-memory objective. That choice remains subject to our own quality tests.

## NPU use for this model

| Graph region | Initial backend | Reason |
|---|---|---|
| Q/K/V and attention output projections | NPU | large dense matrix operations |
| MLP gate/up/down projections | NPU | dominant dense matrix traffic |
| DeltaNet input, gate, value and output projections | NPU | regular batched projection work |
| DeltaNet convolution and recurrent state update | CPU first | sequential state and dispatch overhead |
| RMSNorm, RoPE, tokenizer, sampling and answer-set selection | CPU | small or control-heavy operations |

The compiler should group projections and keep shared buffers resident instead of crossing the CPU/NPU boundary for each small operator. Fixed or schema-bounded prompt prefixes may be compiled into KV/DeltaNet state snapshots. Output behavior remains prompt-defined; no classification labels are compiled into the generic target.

## Target-specific plan

| Step | Change | Required evidence |
|---:|---|---|
| 1 | Reproduce RKLLM W4A16, group-128 and W8A8 on the selected board | exact SDK, clocks, RSS, TTFT, prefill, decode and outputs |
| 2 | Run the model's generic C runtime on Cortex-A72/A53 | tensor agreement and CPU-only baseline |
| 3 | Lower one projection group through Rockchip's `rknn_matmul_api` | output differential test and dispatch timing |
| 4 | Emit NPU-native W4A16 layout and create contexts/buffers once | zero repacking or allocation in token loop |
| 5 | Search layer placement and CPU/NPU overlap | end-to-end timing including synchronization |
| 6 | Compile fixed-prefix state snapshots for declared prompt templates | exact replay comparison against full prefill |
| 7 | Compare W4A16, group-128 and selective W8 by layer | held-out quality, perplexity, RSS and decode |
| 8 | Fuse complete DeltaNet/attention regions only when boundary cost remains material | end-to-end gain and tensor-boundary agreement |

The public repository must not vendor proprietary Rockchip SDK headers or libraries without a license that permits redistribution. Generated C owns the graph, tokenizer, prompt contract and schedule; the Rockchip runtime remains an explicit target dependency.

## Current conclusion

RK3576 is the better first Rockchip NPU target for the approximately 1 GB resident-memory objective. Its official Qwen3.5-0.8B W4A16 reference uses 689.50 MB and reaches 18.79 tokens/s, leaving substantially more system headroom than RK3588 W8A8. This is a selection hypothesis based on external data, not a repository result. A physical board must still demonstrate peak RSS below the declared ceiling, zero swap, acceptable storage loading, output quality and thermal stability.
