# 任务专用 CPU 模型 Compiler 框架

日期：2026-08-09  
时区：Asia/Shanghai  
状态：框架定义阶段；尚未开始 runtime 编码

## 1. 框架目标

本框架把范围明确的任务、一个小模型 checkpoint 和目标 CPU 编译为任务专用的本地部署版本。

```text
task contract
+ Prompt family
+ model checkpoint
+ target hardware profile
+ memory / latency / quality constraints
+ compiler-search and validation data
              |
              v
       offline task compiler
              |
              v
versioned local deployment bundle
```

compiler 可以使用大量 CPU、GPU、RAM、存储和时间。部署端不承担训练、搜索或通用 graph 解释成本，只执行已经选择好的模型 graph、layout、kernels、Prompt policy 和输出协议。

框架不绑定 Gemma 4。Gemma 4 E2B 与 Intel J3455 只是当前第一条垂直验证路径。其他小模型通过 model adapter 接入同一套 task specialization、compiler search、quality gates、hardware lowering 和 artifact packaging 流程。

## 2. 稳定分层

```text
Task frontend
  task contract / Prompt family / input-output schema / quality gates

Model adapter
  checkpoint / tokenizer / graph inventory / teacher oracle
  architecture-specific exact rewrites and approximate candidates

Hardware backend
  ISA kernels / memory layout / static arena / threading / packaging ABI
```

Task frontend 不包含某个模型的 layer 名称。Hardware backend 不包含任务的危险 taxonomy 或 Prompt 文本。Model adapter 是模型结构与通用 compiler 之间的边界。

## 3. Compiler 输入

最小输入不是单个自然语言 Prompt，而是一份版本化 task contract：

| 输入 | 内容 |
|---|---|
| Task identity | task ID、revision、用途 |
| Input contract | schema、字段类型、最大长度、是否允许自由文本 |
| Prompt family | 固定 prefix、templates、变量插槽、tokenization 规则 |
| Output contract | labels、结构化 schema、是否允许短解释 |
| Safety policy | 硬规则、OOD 处置、不能被模型覆盖的条件 |
| Model source | checkpoint、tokenizer、reference implementation revision |
| Target profile | CPU ISA、cores、cache、RAM、storage、OS ABI |
| Resource limits | RSS、swap、decision latency、artifact size |
| Quality gates | task metrics、teacher divergence、漏报/误报上限 |
| Data | search、held-out validation、boundary/OOD 三类数据 |

框架可以提供 task contract 模板和静态检查器，但不能从一个 Prompt 自动推断真实任务的标签语义、危险定义或验收标准。

## 4. 有界 Prompt family

框架不要求 compiled artifact 对任意聊天 Prompt 保持能力。一个任务可以声明有边界的 Prompt family：

```text
fixed instruction prefix
+ one of a small number of templates
+ variable fields within a defined schema
+ fixed output request
```

这允许 compiler：

- 离线执行 causal fixed prefix，并保存绑定模型 revision 的 KV snapshot；
- 只针对任务内 hidden-state distribution 搜索量化、低秩、剪枝和条件路由；
- 生成固定 label head 或限制 LM-head candidate rows；
- 按已知 context-length buckets 生成 layout 和 schedule；
- 在 token grammar 有严格边界时删除不可能使用的 embedding rows。

有界 Prompt family 不自动等于有界 token set。只要 variable input 允许自由文本，模型仍可能需要完整 tokenizer 和 input embedding vocabulary。

范围外输入必须由 schema/version/token-count/OOD checks 显式拒绝、转人工复核或转到更保守的 artifact。高度专用的 artifact 不对范围外 Prompt 提供通用模型保证。

## 5. 编译期资源假设

编译期时间和资源不设工程上限。compiler 可以进行通常不经济的工作：

- 保存大量任务内 teacher traces 和中间 activations；
- 对各层分别搜索 bit allocation、codebook、低秩 correction 和 tree topology；
- 联合优化离散 codes、router、leaf precision 和 memory layout；
- 使用逐层 error、最终 task loss、输出 ranking 和真实任务指标筛选 candidate；
- 为单一 checkpoint、Prompt family 和 CPU profile 生成固定 schedule。

编译资源无限不等于模型可以无损任意压缩，也不等于能够枚举全部 variable input。运行时尺寸、读取字节数、instruction throughput 和任务误差仍是硬约束。

## 6. Compiler pipeline

一个完整 compile job 至少包括：

1. 验证 task contract、数据隔离和目标硬件 profile。
2. 固定 checkpoint/tokenizer revision，建立 teacher oracle。
3. 展开 Prompt family，生成 token IDs、context buckets 和 fixed-prefix states。
4. 通过 model adapter 枚举 exact rewrites 和 approximate candidates。
5. 搜索 quantization、codebooks、low-rank correction、structured pruning 和 expert trees。
6. 通过 hardware backend 生成 kernels、layout、thread schedule 和 static arenas。
7. 在 held-out、boundary 和 OOD 数据上比较 teacher 与 compiled candidates。
8. 在目标硬件或等价 runner 上验证 RSS、swap、latency 和输出 schema。
9. 只有全部 hard gates 通过时才生成 deployable release。
10. 未通过时输出失败原因和 candidate 数据，不把结果标记为可部署。

目标机不负责动态训练 router、重新选择 kernel 或搜索未知配置。

## 7. Model adapter contract

每种模型架构需要一个独立 adapter：

| Adapter 能力 | 要求 |
|---|---|
| Checkpoint import | 固定 revision，验证 tensor shapes/dtypes |
| Tokenizer import | 产生稳定 token IDs，支持 allowed-token 分析 |
| Graph inventory | 列出 layers、attention/sequence module、MLP、norm、embedding/head 和 cache 语义 |
| Reference execution | 在指定 tensor boundaries 产生 teacher outputs |
| Exact rewrites | 声明可证明保持语义的 constant folding、weight tying、prefix cache 等 |
| Approximate candidates | 声明可量化、剪枝、低秩化或条件路由的 components |
| Error boundaries | 定义逐层和端到端比较位置 |
| Runtime lowering | 降到 hardware backend 支持的 primitive set |

不同模型可以暴露不同的结构机会，例如 GQA/MQA、MoE、shared layers、state-space blocks、局部 attention 或模型特有 embedding。通用 compiler 不假设这些结构一定存在。

能够读取权重文件不等于已经支持该模型。新增 adapter 必须完成 graph/oracle 对齐、exact rewrite 验证、approximation sensitivity、primitive coverage 和 task-level regression。

## 8. Hardware backend contract

hardware profile 至少记录：

- ISA 与可用 SIMD instructions；
- core/thread topology；
- cache sizes 和共享关系；
- NUMA、memory capacity 和持续 bandwidth；
- storage capacity、顺序读取和随机访问约束；
- OS ABI、page size、mmap 和 thread-affinity 能力。

backend 负责生成或选择：

- packed integer、codebook-LUT 和必要的 floating-point primitives；
- layer-specific fusion；
- weight/code/index layout；
- static memory arena；
- thread partition 和 reduction schedule；
- prefetch、mmap 和 residency policy。

目标硬件没有实测数据时，compiler 可以产生理论预算，但 artifact 不能仅凭理论 bandwidth 标为达到 latency gate。

## 9. 本地部署版本

生成物逻辑上是一个 versioned local binary。物理 bundle 可以包含：

```text
task executable
immutable packed model image
fixed-prefix state snapshot(s)
task/schema manifest
checksums and provenance
optional license/notices
```

executable 与大型 model image 分离便于 mmap、原子更新和回滚。交付层可以将它们封装成单文件，但 runtime sections 和 checksum 仍应独立。

目标机器不需要 Python、训练代码、calibration data、通用推理框架或远程 inference service。第一阶段接口优先采用可记录、可重放的 stdin/stdout record、Unix socket 或嵌入式 C ABI，不在 runtime 中捆绑通用 HTTP server。

相同 bundle、输入和明确的 decision policy 应产生可复现结果。固定分类主路径不使用开放式随机采样。

## 10. Recompile 边界

以下任一变化都产生新 compile job：

- checkpoint 或 tokenizer revision；
- Prompt prefix、template、input/output schema；
- task label、quality gate 或 safety policy；
- target CPU/ISA、RAM limit 或 OS ABI；
- quantization/tree configuration；
- 主要 compiler-search distribution。

部署端不增量修改自身权重。新版本由 compiler 生成并离线验证，目标机校验 checksum 后原子切换；旧版本保留用于回滚和 decision audit。

## 11. 文档分层

根目录只存放跨模型框架文档。每个模型使用独立目录，记录其 graph、专项优化、目标硬件实验和资源预算：

```text
TASK_SPECIALIZED_CPU_COMPILER_FRAMEWORK_2026-08-09.md
<model-name>/
  OPTIMIZATION_ENGINEERING_RECORD_<date>.md
```

模型目录名称使用稳定、可识别的模型名称。模型文档不得重新定义通用 task contract 或 hardware-backend API；框架文档也不保存某个模型的逐层常量和专项 kernel 细节。

当前模型记录：

- [Gemma 4 E2B 专项优化](gemma-4-e2b/OPTIMIZATION_ENGINEERING_RECORD_2026-08-09.md)

## 12. 第一条垂直验证路径

```text
task:       delayed danger assessment
model:      Gemma 4 E2B IT text-only
hardware:   Intel Celeron J3455 / SSE4.2 / 3.34 GiB RAM
resident:   target 960 MiB, reject above 1024 MiB
output:     fixed risk labels, optional bounded explanation
runtime:    generated C executable + immutable packed image
```

该路径用于验证 task contract、model adapter、hardware backend、compiler search、artifact generation 和本地 decision execution 能否闭环。它不是框架已经支持其他模型、CPU 或任务的证明。
