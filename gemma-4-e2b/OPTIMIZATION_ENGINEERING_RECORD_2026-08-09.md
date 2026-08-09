# Gemma 4 E2B / J3455 专项优化工程记录

日期：2026-08-09  
时区：Asia/Shanghai  
状态：第一条垂直验证路径的架构与资源预算阶段；尚未开始 runtime 编码

本文只记录 Gemma 4 E2B 在 Intel J3455 上的模型结构、专项优化、资源预算和第一条任务验证路径。通用 task-to-binary compiler 方法见根目录框架文档。

## 1. 最小可行模型的工程边界

调查日期：2026-08-09。

订正记录：本节初稿把 `Gemma 3 1B IT` 写成项目的首要 compiler/runtime target。该判断是在没有先核对 2026 年 4 月发布的 Gemma 4 的情况下形成的，不能保留为当前代模型选择。下文已将首要 target 改为 `Gemma 4 E2B IT`；Gemma 3 1B 只保留为可选的早期正确性冒烟测试。

本节区分两个问题：

1. 参数量最小、仍能执行基本指令的通用模型；
2. 适合本项目 bring-up 和性能验证的最小模型。

这两个问题的答案不同。

资料：

- Qwen3.5-0.8B 官方模型卡：<https://huggingface.co/Qwen/Qwen3.5-0.8B>
- Qwen3 官方发布记录：<https://qwenlm.github.io/blog/qwen3/>
- Gemma 4 官方模型总览：<https://ai.google.dev/gemma/docs/core>
- Gemma 4 官方模型卡：<https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 官方 QAT Q4_0 模型集合：<https://huggingface.co/collections/google/gemma-4-qat-q4-0>
- Gemma 4 E2B IT 官方 Q4_0 GGUF：<https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf>
- Gemma 3 官方介绍：<https://developers.googleblog.com/en/introducing-gemma3/>
- Gemma 3 1B IT 官方模型卡：<https://huggingface.co/google/gemma-3-1b-it>
- Gemma 3 1B IT 的 llama.cpp GGUF：<https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF>
- SmolLM3 官方发布记录：<https://huggingface.co/blog/smollm3>

### 1.1 通用模型的参数量下限

当前可作为“稍微能用”下限观察的模型是 `Qwen3.5-0.8B`。

这里的“稍微能用”仅指：

- 简短中文问答；
- 分类和意图识别；
- 从短文本中抽取字段；
- 短文本改写；
- 格式固定、范围受限的指令。

不能由 0.8B 参数量推导出以下能力可靠：

- 多步推理；
- 代码生成和修改；
- 长文档理解；
- 可靠事实问答；
- 自主工具调用；
- 无约束对话。

Qwen 官方也提供过 `Qwen3-0.6B`。它证明 0.6B 级别可以运行和执行指令，但本记录不把 0.6B 作为通用可用下限。模型规模继续下降以后，任务约束和专门微调对结果的影响会大于通用能力。

因此，本文采用以下工程判断：

```text
能运行的下限：约 0.6B
勉强通用可用的下限：约 0.8B
开始具有较广实际用途的区间：约 3B–4B
```

最后一项是工程分档，不是公开 benchmark 的统一结论。具体结果取决于语言、量化、任务和评测集。

### 1.2 本项目的最小 bring-up 模型

如果项目目标是针对当前 Gemma 架构做专用 compiler/runtime，首要 target 应采用 `Gemma 4 E2B IT`。Gemma 4 于 2026 年 4 月发布；E2B 是当前 Gemma 4 家族最小型号。

E2B 不能按普通 2B dense 模型理解：

- effective parameters 为 2.3B；
- 加入 Per-Layer Embeddings（PLE）后的总参数为 5.1B；
- 35 个 decoder layers；
- vocabulary size 为 262K；
- local sliding-window attention window 为 512 tokens；
- context length 为 128K；
- text、image、audio 输入，text 输出；
- vision encoder 约 150M 参数；
- audio encoder 约 300M 参数。

官方给出的 E2B 静态权重加载估算为：BF16 11.4 GB、SFP8 5.7 GB、Q4_0 2.9 GB。该表包含 20% load overhead，不包含 context KV cache。官方 Hugging Face 仓库中的 Q4_0 GGUF 主模型文件为 3.35 GB，multimodal projector 文件为 987 MB。LiteRT-LM mobile text-only 配置的官方估算为 0.84 GB，但该格式不能与通用 GGUF Q4_0 的内存占用直接比较。

Gemma 4 增加了本项目需要明确实现或明确排除的结构：

- PLE；
- local sliding-window 与 global attention 交错；
- global layers 的 unified Keys/Values；
- Proportional RoPE（p-RoPE）；
- thinking mode 和新的 chat template；
- 独立 MTP draft model，用于 speculative decoding；
- 可选 multimodal encoder/projector。

官方提供 `google/gemma-4-E2B-it-qat-q4_0-gguf`，当前 llama.cpp 可以直接加载，适合作为结果和性能基线。官方还提供约 78M 参数的 E2B QAT assistant/draft checkpoint，可在基础 decode 正确以后单独验证 MTP。

`Gemma 3 1B IT Q4_K_M` 仍可用于以下早期工作：

- graph lowering 是否正确；
- tensor layout 是否正确；
- 量化 kernel 是否正确；
- generated runtime 是否能完成端到端 decode；
- 输出是否可与 llama.cpp 对照；
- 单机 profiling 和回归测试是否稳定。

但 Gemma 3 1B 只能称为 smoke-test model，不能再称为项目的当前代主 target。

### 1.3 最小性能实验模型

建议将模型梯度固定为：

| 阶段 | 模型 | 用途 |
|---|---|---|
| 可选 smoke test | Gemma 3 1B IT，Q4_K_M | 先打通最小 text-only 路径；不作为当前架构结果 |
| 当前代 bring-up 和第一轮性能实验 | Gemma 4 E2B IT，官方 QAT Q4_0 GGUF | 实现 PLE、hybrid attention 和 text-only decode；与 llama.cpp 对照 |
| 当前代规模扩展 | Gemma 4 E4B IT，官方 QAT Q4_0 | 观察更大 PLE 权重、layout、prefetch 和线程调度 |
| 服务器架构比较 | Gemma 4 12B IT，官方 QAT Q4_0 | 放大 memory-channel 和持续带宽差异；评估 unified architecture |
| NUMA/MoE 专项 | Gemma 4 26B A4B IT，官方 QAT Q4_0 | 验证 expert placement、routing 和跨 NUMA 流量 |

因此，本项目里“最小可行”应写成两个明确结论：

```text
面向用户的最小通用模型：Qwen3.5-0.8B
面向当前 Gemma compiler/runtime 的最小主模型：Gemma 4 E2B IT QAT Q4_0
可选的早期正确性冒烟模型：Gemma 3 1B IT Q4_K_M
```

如果目标是验证 CPU + RAM 专用编译对 decode bandwidth 的收益，Gemma 4 E2B 已经有 5.1B 总参数和 3.35 GB Q4_0 主 GGUF，可用于第一轮测试。NUMA 和服务器内存通道结论仍应继续用 12B 或更大的模型验证。

### 1.4 Gemma 4 E2B 的官方能力数据

Gemma 4 官方模型卡给出的 E2B IT thinking 结果包括：MMLU Pro 60.0%、AIME 2026 37.5%、LiveCodeBench v6 44.0%、GPQA Diamond 43.4%、Tau2 三项平均 24.5%、MMMLU 67.4%。这些是 Google 的公开 benchmark，不是本项目复测结果。

这些指标不能合并成一个通用能力分数，也不能直接预测中文特定任务或 CPU inference 性能。本节没有在本地或云端运行上述模型。模型质量和速度均未实测。Qwen3.5-0.8B 的精确 GGUF 文件尺寸也未在本节确认。

## 2. 小型本地 x86 机器

调查日期：2026-08-09。

本次只执行只读检查。没有安装软件、下载模型、修改系统配置或运行 inference。

连接目标：

```text
SSH: root@100.102.47.48
Hostname: caremojo-hub-410e4a20
```

### 2.1 实际观察到的规格

| 项目 | 观察结果 |
|---|---|
| 机器类型 | Schneider tablet-class x86 system；非虚拟机 |
| CPU | Intel Celeron J3455，Apollo Lake，4 cores / 4 threads |
| CPU 频率 | 1.50 GHz 标称；800 MHz–2.30 GHz cpufreq 范围 |
| SIMD | SSE、SSE2、SSSE3、SSE4.1、SSE4.2；没有观察到 AVX、AVX2、FMA 或 AVX-512 |
| Cache | 每核 24 KiB L1d、32 KiB L1i；2 MiB L2，共 2 个 1 MiB instance；未报告 L3 |
| NUMA | 1 node，CPU 0–3 |
| RAM | 3,589,599,232 bytes，约 3.34 GiB 可见内存 |
| Swap | `/swap.img`，3,923,767,296 bytes，约 3.65 GiB；检查时使用量为 0 |
| Memory DMI | 两个 2 GB device，分别标为 ChannelA-DIMM0 和 ChannelB-DIMM0；DDR3-1866，16-bit data width |
| Storage | 62,545,461,248-byte eMMC，约 58.25 GiB；root filesystem 检查时约 45 GiB 可用 |
| GPU | Intel Apollo Lake GT1 / HD Graphics 500；存在 `/dev/dri/renderD128` |
| Network | Realtek RTL8111/8168 千兆以太网；Intel Wireless 7265 |
| OS | Ubuntu 26.04 LTS，x86-64 |
| Kernel | Linux 7.0.0-14-generic |
| Runtime baseline | Python 3.14.4，glibc 2.43；没有安装 gcc 或 cmake |
| CPU governor | 四个 CPU 均为 `schedutil` |
| 检查时温度 | CPU/package 约 37°C；这是空闲时单次读数，不是负载温度 |

DMI 数据中还出现了占位 manufacturer、serial、part number 和不合理的 voltage 值。因此，本记录只把“两项 2 GB device、channel locator、DDR3-1866”写为固件报告值，不把它们当成已经验证的物理颗粒规格。实际内存通道宽度和持续带宽需要单独测量。

### 2.2 修正后的目标

目标不是寻找能在本机启动 Gemma 4 的现成 framework。目标是针对以下三个固定对象生成专用 runtime：

```text
固定模型：Gemma 4 E2B IT，text-only
固定硬件：Intel Celeron J3455，4 cores，SSE4.2，3.34 GiB RAM
限定任务：危险判断使用的有界 Prompt family，不支持任意聊天 Prompt
```

runtime 从零以 C 实现，不把 llama.cpp、LiteRT-LM、ONNX Runtime 或 PyTorch 作为目标机器依赖。现有实现只作为 graph 定义、权重来源和 correctness oracle。

HD Graphics 500、vision encoder、audio encoder 和通用 128K context 不进入第一阶段范围。

主要资料：

- Gemma 4 Technical Report：<https://arxiv.org/abs/2607.02770>
- Gemma 4 官方模型卡：<https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 E2B 官方 config：<https://huggingface.co/google/gemma-4-E2B/blob/main/config.json>
- Transformers Gemma 4 参考实现：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py>
- Transformers Gemma 4 config 实现：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py>
- 第三方 PLE vector-quantization 实验：<https://huggingface.co/TheStageAI/gemma-4-E2B-it>
- AQLM 2–3 bit additive quantization：<https://arxiv.org/abs/2401.06118>
- LUT-GEMM codebook lookup matrix multiplication：<https://arxiv.org/abs/2206.09557>
- MoEfication dense-FFN-to-experts：<https://arxiv.org/abs/2110.01786>
- Deja Vu contextual sparsity：<https://arxiv.org/abs/2310.17157>
- PowerInfer hot/cold neuron locality：<https://arxiv.org/abs/2312.12456>
- Tree maximum-inner-product search：<https://arxiv.org/abs/1202.6101>

其中 TheStageAI 项不是 Google 官方 checkpoint，也不是本项目复测结果。它只证明“PLE 与 transformer weights 采用不同压缩方法”已经有人实际尝试；不能直接采用其质量声明。

### 2.3 E2B text graph 的固定结构

官方 config 当前给出的 text model 常量如下：

| 常量 | E2B 值 |
|---|---:|
| Vocabulary | 262,144 |
| Hidden size | 1,536 |
| Layers | 35 |
| Base MLP intermediate | 6,144 |
| Late double-wide MLP intermediate | 12,288 |
| Query heads | 8 |
| KV heads | 1 |
| Local head dimension | 256 |
| Global head dimension | 512 |
| PLE dimension per layer | 256 |
| Local window | 512 tokens |
| KV-shared layers | 后 20 层 |
| Local/global pattern | 4 local + 1 global，重复 7 次 |
| Local RoPE | theta 10,000，全部 local head dimensions |
| Global p-RoPE | theta 1,000,000，partial rotary factor 0.25 |
| Final logit softcap | 30.0 |

35 层不是同构循环。按 zero-based layer index，至少分为四类：

| Layer 类别 | Index | 数量 | K/V | MLP width |
|---|---|---:|---|---:|
| Early local | 0–13 中除 4、9 | 12 | 本层生成 | 6,144 |
| Early global | 4、9、14 | 3 | 本层生成 | 6,144 |
| Late local | 15–33 中除 19、24、29 | 16 | 复用 layer 13 local KV | 12,288 |
| Late global | 19、24、29、34 | 4 | 复用 layer 14 global KV | 12,288 |

后 20 层没有自己的 K projection、V projection、K norm 和 V norm。它们仍计算各自的 Q 和 attention output。官方实现同时将这 20 层的 MLP 扩为 double-wide，因此减少的 attention 参数被部分转移到了 MLP。

不计 norm 和 scalar 等小 tensor，四类 layer 的主要矩阵参数量约为：

| Layer 类别 | 单层主要矩阵参数 |
|---|---:|
| Early local | 约 36.2M |
| Early global | 约 43.3M |
| Late local | 约 63.7M |
| Late global | 约 70.0M |

总数约 1.87B，与 technical report 的 E2B `Einsums: 1,870M` 对应。Late double-wide MLP 是 decode 期间最主要的连续权重流。

单层 text block 的顺序不能简化为普通 pre-norm Transformer：

```text
x
 -> RMSNorm -> Q/K/V attention -> RMSNorm -> residual add
 -> RMSNorm -> GELU-gated MLP   -> RMSNorm -> residual add
 -> PLE gate * per-token PLE -> projection -> RMSNorm -> residual add
 -> layer scalar
```

模型同时使用 pre-norm、post-norm、QKNorm 和 V normalization。正确性实现必须保留这些位置。

### 2.4 可以利用的 Gemma 4 特性

#### 2.4.1 在编译期合并 text-only PLE

E2B 的 packed PLE table 形状为：

```text
[262144, 35 * 256]
```

即 2,348,810,240 个参数。Technical report 将其记为约 2,340M parameters。它大，但对单个 decode token 只访问一行，共 8,960 个值。

官方 forward 对纯文本 token `t` 的 PLE 计算可以写为：

```text
main_input(t) = main_embedding[t] * sqrt(1536)
context_l(t)  = RMSNorm(W_context_l * main_input(t) / sqrt(1536))
token_l(t)    = PLE_table[t, l] * sqrt(256)
ple_l(t)      = (context_l(t) + token_l(t)) / sqrt(2)
```

`sqrt(1536)` 在 projection 前后抵消。`ple_l(t)` 只依赖 token ID 和固定权重，不依赖 layer 执行期间变化的 hidden state。

因此，text-only compiler 可以离线生成最终 `ple_l(t)`，删除 runtime 中的：

- `1536 -> 35*256` 的 `per_layer_model_projection`；
- projection 后的 35 组 RMSNorm；
- 两个 PLE 分支的运行时相加和缩放。

这是从官方实现推导出的 graph specialization，不适用于 image/audio soft tokens。它必须通过逐层 golden tensor 验证浮点误差。

最终 PLE 仍然很大，不能使用普通逐元素 BF16。候选格式是按 token row 组织的 vector quantization：

- 一行包含 35 个 layer slice；
- 每个 slice 为 256 dimensions；
- codebook 常驻 RAM；
- code/index row 按 token ID 随机读取；
- 当前 token 的 8,960-value row 只解码一次或逐 layer 解码。

第三方 TheStageAI 声称使用 AQLM-style vector quantization 将约 4.7 GB BF16 PLE 压到约 0.26 GB。这个数字未由本项目验证，但可作为自定义格式的尺寸参考。

#### 2.4.2 只保存 15 份 KV，而不是 35 份

Layer 15–34 按 attention type 复用 layer 13 或 layer 14 的 KV。专用 runtime 不应为后 20 层分配 KV cache，也不应在文件中保留不存在的 K/V weights。

初始 context 上限固定为 512：

- 12 个 early local layer 使用 512-entry circular KV buffer；
- 3 个 early global layer保存完整 context KV；
- 16 个 late local 和 4 个 late global layer只保存引用，不分配新 KV。

MQA 只有 1 个 KV head。计算 8 个 query heads 时不能物理复制 K/V 八次。

#### 2.4.3 分开生成 local 和 global attention kernel

Local 和 global attention 的 head dimension、RoPE 和 cache 访问不同：

- local：head dim 256、完整 RoPE、512 ring buffer；
- global：head dim 512、只旋转前 25% dimensions、读取完整 context。

global p-RoPE 的未旋转 75% dimensions 不应进入 sin/cos 或 rotate loop。两套 RoPE 的 sin/cos 可以按 position 增量生成；不需要每层重新计算三角函数。

#### 2.4.4 融合 MLP 的 gate/up/down 路径

MLP 是：

```text
down(GELU(gate(x)) * up(x))
```

gate 和 up 使用同一个 normalized input。权重应按同一 tile 交错存储，一次 activation quantization 后同时计算 gate/up。GELU、elementwise multiply 和 down projection 按 intermediate stripe 融合，避免产生完整的 6,144 或 12,288-element temporary buffer。

对于后 20 个 double-wide layer，四个线程各处理一段 intermediate dimension，生成 1,536-element partial output，最后做固定大小 reduction。这里不需要通用 task scheduler。

#### 2.4.5 融合 RMSNorm、residual 和量化边界

J3455 没有 FP16/BF16 arithmetic。计划采用：

- hidden/residual/norm accumulation：FP32；
- matrix activation：每个 projection input 动态量化为 int8；
- weights：按 tensor sensitivity 使用 2/3/4-bit packed representation；
- dot accumulation：int32；
- projection output：恢复到 FP32。

同一个 normalized input 被 Q/K/V 或 gate/up 共用时只量化一次。post-attention norm + residual、post-MLP norm + residual、post-PLE norm + residual分别实现为单次 streaming pass。

#### 2.4.6 为 SSE4.2 生成独立 GEMV kernel

目标 CPU 没有 AVX2/VNNI。kernel 只生成 128-bit SSE 路径：

- SSSE3 `pshufb` 解包 4-bit 或 2-bit weights；
- `pmaddubsw` + `pmaddwd` 做 byte dot product；
- unsigned activation 使用固定 zero-point，并为每个 weight row 预存 correction sum；
- 限制 block accumulation，避免 `pmaddubsw` 的 16-bit saturation；
- weights、scale 和 correction 按 64-byte cache line 对齐；
- scalar reference kernel 验证每种 quant block。

四个 core 使用常驻 pthread pool和静态分工。运行时不按 layer 创建线程，不在 token loop 中调用通用 allocator。

#### 2.4.7 Tied embedding/LM head 只保存一份

Main embedding 为 262,144 x 1,536，约 402.7M parameters，并与 LM head tied。自定义文件只保存一份量化权重：

- input 时按 token ID读取单行；
- output 时四线程顺序扫描 vocabulary rows。

`final_logit_softcapping=30` 使用单调的 `30*tanh(logit/30)`。因此：

- greedy argmax 完全不需要执行 softcap；
- top-k=64 时先按 raw logits 选择 top 64，再只对 64 个值执行 softcap 和 softmax；
- 不需要 materialize 262,144-entry FP32 logits buffer。

这是 exact ranking optimization，不改变 top-k 集合。

#### 2.4.8 编译期资源不设上限

本项目采用以下非对称约束：

```text
compiler：时间、CPU、GPU、RAM 和临时存储不设工程上限
runtime：固定 J3455；steady-state RSS 约 1 GiB；不使用 swap；不能依赖 eMMC 持续换页
```

因此，离线 compiler 可以执行通常不经济的工作：

- 在大规模代表性 token corpus 上保存每层输入、输出和 neuron contribution；
- 对每一层分别穷举 quantization bit allocation、codebook、outlier set、tree fan-out、leaf width 和 top-k leaf count；
- 对离散 code 和 tree partition 反复做全模型 teacher-logit 校准；
- 同时使用 layer-output error、最终 logit KL、下一个 token ranking 和任务评测选择候选；
- 为这一个 checkpoint 和这一颗 CPU 生成固定 layout、固定路由树和固定 kernel schedule。

编译资源无限不等于模型可以无损任意压缩。运行时文件尺寸、每 token 读取字节数和近似误差仍然是硬约束。compiler 的作用是搜索更好的 Pareto 点，并对所有近似变换提供可复现的误差数据。

compiler 不需要覆盖全部可能的聊天 prompt；它只需要覆盖已经定义的危险判断 Prompt family。但这个 family 中的 variable observation 和 hidden state 仍不可能全部枚举。任何从 corpus 学到的路由树都仍有 distribution-shift 风险；无限算力只允许扩大任务内 corpus、搜索和验证，不能把经验误差自动变成形式化正确性证明。

#### 2.4.9 将 late MLP 编译为条件专家树

这是“把层变成树”最直接、最值得实验的部分。Gemma 4 E2B 的一个 gated MLP 可写为：

```text
a_j(x) = GELU(gate_j dot x) * (up_j dot x)
y(x)   = sum_j a_j(x) * down_column_j
```

每个 intermediate neuron 的 `gate row`、`up row` 和对应 `down column` 必须作为一个原子单元移动。compiler 将这些 neuron 按共同激活、输出方向和 teacher-loss sensitivity 分组为 leaf expert，再生成一棵小路由树。runtime 只访问选中的 leaves：

```text
normalized hidden
        |
small oblique routing tree
        |
top-k leaf IDs
        |
packed gate/up rows + matching down columns
        |
1,536-element MLP output
```

优先处理 layer 15–34 的 12,288-wide MLP。原因是 E2B 的 MLP 参数约为：

| 部分 | 估算参数量 |
|---|---:|
| 前 15 层、6,144-wide MLP | 约 425M |
| 后 20 层、12,288-wide MLP | 约 1,132M |
| 合计 | 约 1,557M |

后 20 层的 MLP 约占全部 1.87B einsum 参数的 61%，也是单 token decode 最大的连续权重流。先树化这 20 层可以在不改变 attention、KV sharing 和 PLE graph 的情况下处理主要流量。

候选 leaf layout 不预先固定为一个数。compiler 应搜索例如 32、64、128、256 个 leaves，以及每层不同的 top-k。每个内部节点使用低维或稀疏 oblique projection 评分；每个 leaf 在文件中占连续、cache-line-aligned 区域。高频 leaf 可以采用较高 bit width，低频 leaf 采用更低 bit width。

仅将 dense neurons 重排成树不会减少模型总字节数；它只可能减少单 token 实际访问的 leaves。若全部 leaves 都留在 RAM，resident memory 不会因为路由而下降。1 GiB 目标仍要求 additive quantization、共享 codebook、低 bit width 或经蒸馏后实际删除部分参数。由于 steady-state 不允许 eMMC paging，本项目不把“冷 leaf 放磁盘”计入正常内存方案。

MoEfication 已经证明预训练 dense FFN 可以被划分为功能 expert，并用 router 条件选择；Deja Vu 和 PowerInfer 则分别给出 contextual sparsity 与 hot/cold neuron locality 的实证。但这些结果没有验证 Gemma 4 E2B 的 `gelu_pytorch_tanh` gated MLP，也没有验证 J3455。因此这里只把它们作为方法依据，不把论文中的稀疏率或速度数字写成本项目预期。

这项变换不是 exact rewrite。GELU gated MLP 的大多数 contribution 不会严格等于零。若 runtime 不计算未选 leaf，输出就与 dense teacher 不同。必须通过以下编译流程控制误差：

1. 用 dense BF16 teacher 采集每个 layer 的真实 hidden distribution。
2. 以 `|a_j(x)| * ||down_column_j||` 和最终 logit loss 联合衡量 neuron/leaf 重要性。
3. 搜索 neuron partition、routing tree、top-k 和每-leaf bit width。
4. 先拟合逐层 residual，再端到端蒸馏 router、leaf scale 和必要的低秩 correction。
5. 对每层生成 confidence threshold；低置信度时增加 leaf count，而不是固定只走一条路径。
6. 对完整 benchmark、长文本和异常 token 分布报告 quality/bytes-read/latency，不只报告平均 layer error。

如果没有训练或蒸馏，只能将这项方案称为有损 structured pruning，不能称为编译优化。

#### 2.4.10 LM head 可以建立词表搜索树

LM head 是 262,144 个 1,536-dimensional row 与 final hidden 的 maximum-inner-product search。可以对 tied embedding rows 建立 hierarchical clustering tree：

- 内部节点保存 centroid、radius 或更紧的 inner-product bound；
- 叶子保存连续排列的 vocabulary rows；
- 先搜索节点，再对候选 leaf 中的真实 rows 重算 logits；
- raw top-k 确定后再执行单调 softcap。

存在 exact branch-and-bound 形式：如果一个节点的严格 upper bound 已低于当前第 k 名 logit，可以安全剪掉该 subtree。它不改变 top-k。但是 hidden dimension 为 1,536，边界可能很松；最坏情况仍需扫描全部 vocabulary。编译算力再多也不能保证运行时 query 一定容易剪枝。

近似形式可以只搜索 top-n clusters 后 rerank，速度收益更确定，但可能漏掉真实 top-k。compiler 应同时生成 exact fallback：当 candidate margin 小、bound 无法证明或 router confidence 低时扩大搜索，必要时扫描全部 head。该树首先是减少每 token LM-head 字节流的实验，不是 1 GiB resident budget 的主要来源。

#### 2.4.11 PLE 使用层次化码本，不使用 hidden-state 路由

PLE 已经由 token ID 直接索引。它没有必要再由 hidden state 选择 expert。可以使用 residual/additive vector quantization 或层次化 codebook：先解 coarse code，再叠加若干 residual code。这种“树”减少存储，并允许 compiler 对高敏感 token/layer slice 分配更多 refinement code；它不会减少 transformer MLP 的计算量。

编译期可以联合优化全部 262,144 token rows、35 layer slices 和 codebooks，并用下游 teacher loss 而不是单纯 L2 reconstruction error 选择 code。AQLM 提供了低于 3 bit/parameter 和 block-level joint optimization 的方法依据，但其论文没有覆盖 Gemma 4 PLE。第三方约 0.26 GB PLE 数字仍需独立复测。

#### 2.4.12 Dense matrix 编译为共享码本 DAG

对 1 GiB 目标，更低风险的第一类“树”不是跳过 neuron，而是把量化后的 dense matrix 表示为共享码本。逻辑上它更接近 DAG：大量 weight blocks 引用相同 codeword，而不是每个 leaf 保存一份重复权重。

对矩阵的一组 input columns，runtime 先计算 activation tile 与该组少量 codeword 的 inner products；随后每个 output row 读取短 code index，并累加对应 lookup value：

```text
activation tile x_g
        |
dot(x_g, codeword_0 ... codeword_K-1)
        |
small partial-sum table
        |
packed code indices for all output rows
        |
dense output vector
```

这保留全部 output rows 和 dense topology。相对于编译后的量化矩阵，lookup 执行可以是 exact；相对于原 BF16 teacher，误差只来自 codebook quantization。AQLM 和 LUT-GEMM 都给出了 codebook/LUT 形式的低 bit matrix multiplication，但没有验证 Apollo Lake、SSE4.2 或 Gemma 4。

无限编译算力允许做硬件约束的全局 codebook 搜索：

- 联合优化同一 transformer block，必要时跨同类 layers 共享 codebook；
- 对敏感 block 使用 coarse code 加 residual refinement，对普通 block 只保留 coarse code；
- 直接以 teacher layer output 和最终 logit loss 优化离散 code，不只最小化 weight L2；
- 搜索能让 4-bit selector、16-entry table 和 `pshufb` 路径成立的 codebook shape；
- 把 code index、scale、correction 和 output-row order 一起优化为顺序读取格式。

J3455 没有 gather 指令。一个在新 CPU/GPU 上有效的任意大小 LUT，在本机可能比直接 `pmaddubsw` 更慢。因此 compiler 必须同时输出至少两个候选：packed integer dot kernel 和 codebook-LUT kernel；最终按目标机的 layer-by-layer 实测选择，不能仅凭压缩率决定。

该 DAG 直接减少 resident payload；条件 expert tree 主要减少每 token 读取和计算。实施顺序应为：先做共享码本 DAG，确认 1 GiB 和基本质量，再评估是否需要有损 expert routing。

#### 2.4.13 当前不树化的部分

- local/global attention 已经通过 512 window、MQA 和后 20 层 KV sharing 降低成本；第一版保持 exact；
- RMSNorm、residual、RoPE 和 PLE gate 保持原 graph；
- 不把完整 transformer block 替换为一个通用 decision tree。1536-dimensional hidden 空间的 piecewise approximation 可能需要大量 leaves，模型尺寸不可控；
- 不采用 hierarchical softmax。它改变训练目标和概率分解，不是对现有 tied LM head 的等价 rewrite。

#### 2.4.14 将 MTP 放在正确性版本之后

Gemma 4 官方 MTP drafter 约 76M parameters，是 4-layer、model dimension 256 的小模型，含 3 local + 1 global layer，并 cross-attend main-model KV。它不需要单独 prefill，可以生成任意 draft length。

MTP 对本机可能有价值：多个 draft token 的 main-model verification 可以转为小 batch，增加 GEMM arithmetic intensity。但 acceptance rate、额外内存和 verification 成本都未知。第一版单-token decoder正确以后再加入，不能用 MTP 掩盖基础 kernel 错误。

### 2.5 自定义 C runtime 的实施计划

本节是计划，没有开始编码。

#### 阶段 0：固定模型与 golden data

1. 固定一个 Gemma 4 E2B IT checkpoint revision 和 tokenizer revision。
2. 固定 text-only、thinking off、context 512、batch 1。
3. 在开发机用官方实现导出小输入的逐层 tensor：embedding、PLE、每层 attention/MLP/PLE residual、final norm 和 selected logits。
4. 现有 framework 只生成 oracle，不进入目标 binary。

#### 阶段 1：离线 model compiler

1. 读取官方 checkpoint。
2. 删除 vision、audio 和第一阶段不使用的 MTP weights。
3. 离线合并 context-aware PLE 与 token PLE。
4. main embedding/LM head去重。
5. 按四类 layer 顺序重排 weights；后 20 层不生成 K/V records。
6. 生成 mixed-bit/additive quantization、scale、zero-point correction 和 PLE codebooks。
7. 对 layer 15–34 搜索 MLP expert tree；对 LM head 搜索 exact/approximate MIPS tree。
8. 用 dense teacher 做逐层与端到端校准；只保留通过质量门槛的候选。
9. 输出固定模型专用文件，不保存通用 tensor name graph。

1 GiB steady-state resident budget：

```text
folded PLE codes + codebooks              220 MiB
transformer weights                       460 MiB
tied embedding / LM-head                  105 MiB
tree/router/scale/correction metadata       35 MiB
KV/activation/tokenizer/thread/runtime      60 MiB
unassigned safety margin                    80 MiB
--------------------------------------------------
target steady-state RSS                    960 MiB
hard rejection threshold                 1024 MiB
swap used = 0
```

这些是设计目标，不是已经达到的结果。RSS 必须在目标机器上用固定 prompt、512 context、持续 decode 实测；不能用 model file size 或 virtual address size 代替。

尺寸的基础算术如下：

- 1.87B einsum parameters 的纯 2-bit payload 约 446 MiB；
- 402.7M tied embedding/head 的纯 2-bit payload 约 96 MiB；
- 第三方报告的约 0.26 GB PLE 相当于约 248 MiB；
- 三者裸 payload 合计约 790 MiB，尚未包含 codebook、scale、alignment、router、KV、arena 和程序本身。

因此 1 GiB 在纸面上不是矛盾目标，但几乎没有使用通用格式或保留大块高精度权重的空间。220 MiB PLE 预算还比第三方约 248 MiB 参考值更紧。若某一敏感 layer 必须使用 3/4 bit，compiler 必须通过其他 layer 的更低 bit、共享 codebook、低秩分解或经蒸馏后实际删除参数找回预算；仅让 leaf 在当前 token 不激活并不会减少 resident memory。

#### 阶段 2：scalar C correctness runtime

1. tokenizer 和固定 chat template；
2. custom model loader；
3. FP32 scalar RMSNorm、RoPE、attention、MLP、PLE 和 LM head；
4. context 16 的完整 prefill/decode；
5. 每个边界与 golden tensor 比较。

Scalar 版本的目标是正确，不以 tokens/s 评价。

#### 阶段 3：SSE4.2 layer kernels

按风险从低到高替换：

1. quantized GEMV；
2. fused gate/up；
3. down partial reduction；
4. Q/K/V 和 Q-only attention projection；
5. streaming LM head + top-k；
6. PLE vector decode + PLE tail；
7. norm/residual fusion。

每替换一个 kernel，都保留 scalar differential test。

#### 阶段 4：prefill、threading 和 I/O

1. decode batch-1 先稳定；
2. 再增加多-token tiled prefill kernel；
3. 测量 1、2、4 threads；
4. 映射两个 L2 instance 的 core 归属，再决定线程 pinning；
5. dense layer weights 采用 sequential mmap/prefetch；PLE token rows 和条件 expert leaves 采用显式布局；
6. 测量 expert-tree router recall、每 token 实际 leaf 数和实际读取字节数；
7. 不 mlock 整个模型，不允许 eMMC 参与 steady-state decode paging。

#### 阶段 5：MTP

只有 single-token output 与 oracle 一致、内存不使用 swap 后，才加入官方 76M drafter并测试 draft length 2、3、4。

### 2.6 当前工程判断

Gemma 4 E2B 对这台机器不是普通“5.1B dense model”：

- 2.34B PLE parameters 每个 token 只查一行；
- 后 20/35 layers 不生成自己的 KV；
- attention 是 8Q/1KV 的 MQA；
- local attention 固定 512 window；
- global RoPE 只旋转 25% dimensions；
- text-only PLE projection可以离线折叠；
- tied LM head只需一份权重；
- logit softcap 可在 top-k 之后执行；
- 76M MTP drafter可在后续把 sequential decode转为小批 verification。

这些特性共同构成从零写 C runtime 的理由。最大剩余问题仍是 1.87B einsum parameters 和 402.7M LM-head parameters 每个 token 的权重流量。当前最具体的 tree candidate 是：只把 layer 15–34 的 double-wide MLP 编译为条件 expert tree，同时为 LM head 建立带 exact fallback 的 vocabulary MIPS tree；attention 和其余 residual graph 保持原结构。

是否能达到可用速度，最终取决于 mixed-bit image 的实际字节数、expert router 的 recall/active ratio、J3455 的持续内存带宽、SSE4.2 unpack/dot 效率和 MTP acceptance rate。编译期资源不设上限可以扩大搜索和校准规模，但不能替代这些目标机实测。

目前没有性能数字，也没有在目标机器上安装、下载或运行任何新内容。

### 2.7 理论吞吐量、RAM 带宽和存储带宽预算

本节是分析模型，不是 benchmark。范围固定为：

```text
Gemma 4 E2B IT text-only
batch = 1
decode，已经完成 prefill
context <= 512
4 cores
steady-state，不使用 swap
```

Intel 给出的 J3455 规格是 4 cores、1.5 GHz base、2.3 GHz burst、2 MiB cache、最多 2 memory channels，并支持 DDR3L/LPDDR3-1866。作为同代平台参考，Intel NUC6CAYS 规格列出 14.9 GB/s max memory bandwidth：

- <https://www.intel.com/content/www/us/en/products/sku/95594/intel-celeron-processor-j3455-2m-cache-up-to-2-30-ghz/specifications.html>
- <https://www.intel.com/content/www/us/en/products/sku/95078/intel-nuc-kit-nuc6cays/specifications.html>

目标机器不是该 NUC。目标机器 DMI 报告两个 16-bit DDR3-1866 device；若该宽度可信，总物理 data width 是 32 bit：

```text
1866 MT/s * 4 bytes = 7.46 GB/s theoretical peak
                    = 6.95 GiB/s theoretical peak
```

控制器效率、refresh、CPU 访问模式和 GPU 共享内存都会降低可用带宽。在没有 STREAM 或专用 read benchmark 前，本节使用 `3.5–5.2 GiB/s sustained` 作为工作区间。若 DMI 宽度不可信、实际 memory bus 更宽，应在实测后上调；当前不使用 14.9 GB/s 作为本机结论。

#### 2.7.1 每 token 的最低权重流量

按纯 2-bit payload，1.87B einsum parameters 可以分成：

| 部分 | 参数量 | 2-bit payload |
|---|---:|---:|
| Early MLP，layer 0–14 | 424.7M | 101.3 MiB |
| Late MLP，layer 15–34 | 1,132.5M | 270.0 MiB |
| Attention 和其他 einsum | 312.9M | 74.6 MiB |
| Transformer 合计 | 1,870.0M | 445.8 MiB |
| Tied embedding/LM head | 402.7M | 96.0 MiB |

这只是 packed code payload。实际还要读取 scale、codebook index、zero-point correction、router 和 alignment padding，因此以下场景使用略高的工程流量：

| Decode 场景 | Late MLP active | LM-head rows | 预测 RAM 流量/token |
|---|---:|---:|---:|
| A：dense correctness fallback | 100% | 100% | 550–570 MiB |
| B：保守 expert tree | 50% | 100% | 415–435 MiB |
| C：目标 expert + vocab tree | 25% | 约 10% | 260–280 MiB |
| D：激进 expert + vocab tree | 12.5% | 约 2% | 220–240 MiB |

PLE 不需要扫描 220 MiB table。按 token ID 每次只读取一行，平均不到 1 KiB；input embedding 也只读取一行。KV、RMSNorm、router 和 activation 流量相对上述几百 MiB 权重流较小，但已经通过工程余量计入区间。

场景 C、D 都是有损路由假设，必须先达到质量门槛。若 vocabulary tree 经常触发 full-head fallback，每个 fallback token 额外读取约 96–105 MiB，并额外计算约 403M weight products。

#### 2.7.2 目标 token rate 对 RAM 带宽的要求

计算式为：

```text
required_RAM_BW = bytes_read_per_token * decode_tokens_per_second
```

取每个场景区间中值：

| 场景 | 中值流量/token | 3 tok/s 需要 | 5 tok/s 需要 | 8 tok/s 需要 |
|---|---:|---:|---:|---:|
| A：dense/full head | 560 MiB | 1.64 GiB/s | 2.73 GiB/s | 4.38 GiB/s |
| B：50% late/full head | 425 MiB | 1.25 GiB/s | 2.08 GiB/s | 3.32 GiB/s |
| C：25% late/10% head | 270 MiB | 0.79 GiB/s | 1.32 GiB/s | 2.11 GiB/s |
| D：12.5% late/2% head | 230 MiB | 0.67 GiB/s | 1.12 GiB/s | 1.80 GiB/s |

如果本机能持续提供 3.5–5.2 GiB/s，单纯的 memory-bandwidth ceiling 为：

| 场景 | 只按 RAM 带宽计算的 ceiling |
|---|---:|
| A | 约 6–9 tok/s |
| B | 约 8–12 tok/s |
| C | 约 13–20 tok/s |
| D | 约 15–23 tok/s |

这些不是可实现的 token rate。它们假设 CPU 可以零成本完成 unpack、lookup、dot、GELU、attention、router 和 reduction。J3455 没有 AVX2、VNNI、FMA、FP16 或 BF16 arithmetic；实际 decode 会在达到这些 RAM ceilings 之前受到 instruction throughput 限制。

#### 2.7.3 CPU-aware token/s 预测

场景 A 每 token 需要处理约 2.27B weight products，包括 full transformer 和 full LM head。场景 C 在 25% late MLP、10% LM-head rows 时仍约需要 1.06B active weight products。codebook-LUT 可以把一部分 multiply 转为共享 partial-sum lookup，但会增加 index decode 和 accumulation；不能假设它把这些操作全部消除。

未做目标机 microbenchmark 前，当前预测如下：

| 场景 | RAM ceiling | CPU-aware sustained decode 预测 | 主要风险 |
|---|---:|---:|---|
| A：dense correctness fallback | 6–9 tok/s | 0.8–1.8 tok/s | full GEMV 和 full LM head |
| B：50% late/full head | 8–12 tok/s | 1.3–2.8 tok/s | LM head 仍占 403M products |
| C：25% late/10% head | 13–20 tok/s | 2.5–4.5 tok/s | router recall、LUT/SSE 效率 |
| D：12.5% late/2% head | 15–23 tok/s | 3.5–6.0 tok/s | 最大质量风险、fallback 频率 |

据此设置三个工程数字：

```text
minimum usable floor       1.0 tok/s sustained decode
current engineering target 3.0 tok/s sustained decode
stretch target             5.0 tok/s sustained decode
```

`3 tok/s` 是当前最合理的单点预测。它要求场景 C 大致成立：late MLP 平均只激活约 25%，LM head 平均只重算约 10% rows，且路由后质量可接受。`5 tok/s` 需要更接近场景 D、较少 full-head fallback、有效 LUT kernel 和负载下稳定频率；现在不能把它写成承诺。

本预测不包括 prompt prefill throughput。Prefill 是多-token tiled matrix multiplication，计算/带宽比例不同，应在 decode kernel 完成后单独建模。

#### 2.7.4 SD/eMMC 吞吐量和容量

正常设计中，约 960 MiB model image 在 decode 前进入 RAM，并在 steady-state 保持 resident。因此：

```text
steady-state decode 所需 SD/eMMC throughput = 0 MiB/s
```

存储顺序读取速度只决定冷启动时间：

| 存储顺序读取 | 读取 960 MiB 的理想时间 |
|---:|---:|
| 25 MiB/s | 约 38 s |
| 50 MiB/s | 约 19 s |
| 100 MiB/s | 约 9.6 s |
| 200 MiB/s | 约 4.8 s |

工程要求可以定为：`>= 50 MiB/s` 可接受，`>= 100 MiB/s` 为目标。单个 compiled image 的最低可用空闲容量约 1.5 GiB；考虑两个候选 image、临时更新和回滚，建议至少保留 4 GiB，8 GiB 更实际。目标机器当前约 45 GiB 可用，容量不是问题。

若把 cold expert leaves 放到 SD/eMMC，存储需求会变成：

```text
storage_BW = cold_bytes_per_token * token_rate
```

例如 3 tok/s 下，每 token 读取 20 MiB cold leaves 就需要 60 MiB/s；50 MiB 则需要 150 MiB/s。leaves 又分散在 20 个 late layers，实际还会受随机读取延迟和 IOPS 影响。这个路径不能稳定支持目标吞吐量，因此仍维持“不允许 eMMC 参与 steady-state decode paging”的设计约束。

### 2.8 预定用途：允许延迟的危险判断

这个 runtime 的预定用途不是实时聊天。目标 workload 是接收一段有限输入，在允许一定延迟的条件下判断是否存在危险情况。

当前可以确定的产品约束：

- 不要求逐字实时输出；
- 不需要长对话历史；
- 一次任务是一个有限 observation/event window；
- 输出重点是风险判断，不是开放式续写；
- 系统可以采用短时聚合、周期检查或事件触发；
- 允许的最大端到端延迟、输入数据类型和危险类别仍待定义。

因此，上一节的 `tok/s` 只用于判断底层 runtime 是否达到基本能力，不应成为最终产品指标。最终指标应改为：

```text
time_to_decision = input preparation
                 + prompt prefill
                 + risk classification
                 + optional short explanation
```

#### 2.8.1 主路径不做开放词表生成

第一版主路径可以固定为少量结构化结果，例如：

```text
SAFE
REVIEW
DANGER
```

具体 label、危险 taxonomy 和阈值尚未确定，上述三个词只是接口示例。

如果使用固定 label，runtime 不需要为每次判断扫描全部 262,144 LM-head rows。可比较两种编译方案：

1. 保留 Gemma final hidden，只计算固定 label token 所需的 vocabulary rows；
2. 离线训练一个 `1536 -> N risk classes` 的小 classification head，并用原模型和人工标注数据蒸馏。

方案 1 更接近原 checkpoint；方案 2 的运行时成本更小，但它是新的任务模型，必须单独验证。完整 tied embedding 仍可能需要保留，因为任意输入 token 都需要 embedding lookup；省掉的是每次输出的 full-vocabulary scan，不是自动删除全部 embedding storage。

只有 `REVIEW` 或 `DANGER` 路径需要可选的短解释。解释可以限制为固定最大 token 数。正常 `SAFE` 路径不生成自然语言。

#### 2.8.2 性能目标从 decode throughput 改为 decision latency

对于短标签任务，持续 decode 不是主要时间。一个判断可能有 128–512 input tokens，但只输出 1–4 个 label tokens；此时 prefill 很可能占主要延迟。

暂时保留以下非承诺性工作档位，等待真实业务给出 SLO：

| 档位 | 端到端 decision latency | 适用方式 |
|---|---:|---|
| 快速检查 | 5–15 s | 输入窗口短，模型常驻，固定 label |
| 普通异步判断 | 15–60 s | 允许较长输入或一次复核 |
| 后台复核 | 1–5 min | 多窗口、多个 prompt 或二次解释 |

这些不是现有性能预测。当前只预测过 decode；没有建立 J3455 prefill 模型。下一轮性能分析必须报告：

- 32、64、128、256、512-token input 的 prefill time；
- fixed-label decision time；
- `SAFE` 主路径和异常解释路径分别的总延迟；
- 冷启动与 warm resident 两种情况；
- 单任务和事件积压时的 queue latency。

由于允许延迟，scheduler 可以先收集一个短窗口，再一次执行 tiled prefill；不需要为每个新 observation 立即逐 token 重跑完整模型。

#### 2.8.3 危险判断的安全边界

Gemma 输出不能作为唯一的 fail-safe。明确的传感器阈值、设备故障码、急停条件或其他确定性危险规则应直接触发告警，不能等待 LLM，也不能被 `SAFE` 输出覆盖。

LLM 适合处理确定性规则难以覆盖的语义组合、事件上下文和需要延迟复核的情况。接口至少应返回：

```text
risk_class
confidence_or_margin
reason_codes
evidence references
model/runtime version
timestamp
```

自然语言解释不是证据本身。系统需要保留原始输入引用，才能审计一次危险判断是由哪些 observation 触发。

#### 2.8.4 下一步定义工作

在开始 runtime 编码前，先补齐以下任务定义：

1. 定义输入：文本、事件、传感器摘要、日志或其组合。
2. 定义危险 taxonomy 和每类处置动作。
3. 定义最大允许漏报率、误报率和必须立即绕过模型的硬规则。
4. 定义 decision-latency SLO，而不是只定义 token/s。
5. 定义 input window 长度、更新频率和事件积压上限。
6. 定义固定输出 schema，以及是否需要异常解释。
7. 建立正常、危险、边界和 distribution-shift 样本集。
8. 用 dense teacher、压缩模型和 tree-routed 模型比较同一批 decision results。

这一步完成后，才能判断 1 tok/s 是否已经满足业务，还是必须继续追求 3–5 tok/s；也才能判断 MLP expert tree 带来的质量损失是否可以接受。

### 2.9 一级前提：只优化有界 Prompt family

本项目不要求 compiled model 对任意 Prompt、任意聊天风格和任意任务保持通用能力。编译目标是一个有边界的危险判断任务集合：Prompt 不是逐字完全固定，但其 instruction、输入 schema、字段类型、长度范围、输出 schema 和危险 taxonomy 可以被约束。

可以把运行时输入写为：

```text
fixed instruction prefix
+ one of a small number of task templates
+ variable observation fields within a defined schema
+ fixed classification request
```

这是 model compiler 的一级输入，和 checkpoint、CPU topology 同等重要。compiler artifact 必须记录所支持的 Prompt-family version；换 Prompt family 需要重新编译和重新评测。

#### 2.9.1 Prompt specialization 可以做的 exact 优化

如果固定 instruction 位于 causal sequence 的前缀，它不依赖后续 variable observation。compiler 可以离线执行该前缀，并保存编译后量化模型的 KV snapshot：

```text
fixed prefix tokens
        |
offline prefill
        |
15 physical KV-cache snapshots + position state
        |
runtime starts from first variable token
```

该优化相对于同一个 compiled quantized model 可以保持 exact；它直接删除每次判断中固定前缀的 prefill 时间。snapshot 必须绑定以下内容：

- checkpoint 和 quantization revision；
- Prompt-family/template revision；
- 精确 token IDs；
- prefix length 和 position convention；
- KV precision/layout；
- local/global attention 类型。

如果固定文字位于 variable observation 之后，它的 hidden state 已经依赖 observation，不能离线生成完整 KV。Prompt schema 应尽量组织为“固定 instruction prefix 在前、variable fields 在后”；不能为了套用 cache 而改变任务语义。

固定 label output 也允许只计算少量指定 LM-head rows。这是对完整 logits scan 的 exact restriction，前提是 decoder 被定义为只能在该 label set 内选择；它不等价于原模型在开放词表上的 argmax。

#### 2.9.2 可以做的任务域有损优化

compiler 可以只用任务内 Prompt distribution 进行：

- layer sensitivity 和 quantization bit allocation；
- MLP neuron clustering、expert-tree routing 和 active-leaf budget；
- codebook、低秩 correction 和 structured pruning；
- fixed-label classification-head distillation；
- context-length bucket 和 thread schedule 选择。

相比通用 Prompt 校准，任务内 hidden-state distribution 更窄，理论上允许更激进的剪枝和更稳定的路由。这个收益必须通过任务内 held-out data 和边界样本验证，不能只看 compiler training corpus。

可以为少量 template/length bucket 分别生成 artifact，例如：

```text
artifact A: short event summary, <= 64 variable tokens
artifact B: normal observation window, <= 256 variable tokens
artifact C: extended review, <= 512 variable tokens
```

多个 artifact 是否值得保留取决于总存储、内存切换成本和实际输入比例。第一版也可以只生成一个覆盖最常见窗口的 artifact。

#### 2.9.3 Vocabulary 和 PLE row pruning 的条件

有界 Prompt family 不自动等于有界 token set。需要分别处理：

- 固定 instruction/template/label tokens：集合完全已知；
- 枚举字段、设备码、reason code：可以得到有限 token set；
- 自由文本 observation：可能触及 tokenizer 的大部分 vocabulary；
- 数字、时间、标识符：组合很多，但实际 subtoken set 可能可统计。

只有当输入 schema 明确禁止自由文本，或者存在完整的 allowed-token grammar 时，compiler 才能安全删除未使用的 embedding 和 PLE rows。若仍允许自由文本，完整输入 embedding/PLE vocabulary 必须保留，或者定义明确的 unknown/out-of-domain 路径。

这是一个潜在的大内存收益点：当前预算中 folded PLE 约 220 MiB、tied embedding/head 约 105 MiB。如果 allowed-token set 只占完整 vocabulary 的一小部分，这两项可以显著缩小；在 Prompt schema 未定义前，不把该收益计入 960 MiB 基线。

#### 2.9.4 范围外输入必须显式处理

task-specialized artifact 对范围外 Prompt 的行为不作通用模型保证。runtime 在进入模型前应检查：

- template ID 和 schema version；
- required fields；
- token count 和 context bucket；
- allowed enum/value ranges；
- allowed-token grammar（如果启用 vocabulary pruning）；
- Prompt-family/compiler-artifact revision match。

不符合条件的输入不得静默送入高度剪枝的模型。处理方式只能是明确拒绝、转 `REVIEW`、使用较保守 artifact，或交给外部系统。范围外输入比例也应进入运行指标。

#### 2.9.5 下一步先冻结 Prompt contract

在量化、树搜索或 C runtime 编码前，先产生一个版本化 Prompt contract：

1. 固定 instruction prefix 的精确文本和 token IDs。
2. 支持的 template IDs。
3. variable fields、类型、顺序和最大长度。
4. 是否允许自由文本。
5. 固定 label set 和输出 grammar。
6. 允许的 context-length buckets。
7. out-of-domain 判定与处置。
8. 用于 compiler search、held-out validation 和危险边界测试的三个互斥数据集。

Prompt contract 冻结后，重新计算三项预算：可离线跳过的 prefix prefill、实际需要保留的 vocabulary/PLE rows，以及任务内 expert-tree active ratio。此前的 1 GiB 和 3 tok/s 仍作为不使用 Prompt-domain memory savings 的保守基线。

## 3. 与框架文档的关系

通用 task frontend、model-adapter contract、hardware-backend contract、compiler pipeline 和 artifact 规则见根目录 [框架文档](../TASK_SPECIALIZED_CPU_COMPILER_FRAMEWORK_2026-08-09.md)。本文不重复定义这些跨模型接口。
