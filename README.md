# cpu-llms-in-c

`cpu-llms-in-c` is an experimental compiler and runtime project for running language models on CPU-only systems.

The compiler takes a model checkpoint, a target CPU profile, deployment constraints, and an optional bounded input profile. It produces a model-specific C runtime and a packed model image. Existing inference frameworks may be used as correctness oracles during development, but they are not runtime dependencies.

## Status

Bounded-profile scalar prototype.

- The offline compiler emits a 966,579,776-byte task image from the pinned Gemma 4 E2B IT checkpoint. It retains 112 profile-reachable tokens, folds their PLE rows, quantizes 275 executed matrices to group-128 Q4, and omits the unused multimodal graph and late shared-K/V weights.
- The C11 runtime executes all 35 text layers, local and global RoPE, 15 physical K/V states, late shared K/V, double-wide late MLPs, PLE, final normalization, and a two-row constrained LM head. It has no inference-framework runtime dependency.
- On one measured Ubuntu x86-64 two-vCPU system, the twelve-case warm run used 948,224 KiB peak RSS with no swap. Aggregate sequential prefill was 0.598 tokens/s and one-step decode was 0.621 tokens/s.
- The twelve obvious smoke examples produced 12/12 Q4 decisions against their written labels. The independent BF16-weight NumPy reference produced 11/12, and Q4/reference decision agreement was 11/12. This is not a safety benchmark.
- A general runtime tokenizer, arbitrary free-text input, fixed-prefix snapshots, tensor-by-tensor full-graph differential tests, SIMD kernels, and a real held-out safety evaluation are not implemented.
- The earlier real-weight layer-0 test still passes ten declared tensor boundaries with maximum absolute error 0.

## Scope

The project investigates offline specialization across the full inference stack:

- model graph lowering;
- architecture-specific graph rewrites;
- mixed-bit and codebook quantization;
- tensor and model-image layout;
- CPU-specific kernels;
- static memory allocation;
- thread placement and scheduling;
- bounded-input specialization when the deployment permits it;
- target-machine validation and artifact packaging.

The intended runtime is a small C program with no Python interpreter, training stack, graph interpreter, or general-purpose inference framework on the target machine.

## Current validation target

The first planned validation target is text-only Gemma 4 E2B inference on an Intel Celeron J3455 system:

| Item | Target |
|---|---|
| CPU | Intel Celeron J3455, 4 cores, SSE4.2 |
| Available RAM | 3.34 GiB |
| Steady-state RSS target | 960 MiB |
| Hard RSS limit | 1,024 MiB |
| Swap during inference | 0 |
| Context limit | 512 tokens |
| Runtime | Generated C executable and immutable packed model image |

This target was selected to expose memory, instruction-set, storage, and scheduling constraints. It is not the only architecture intended for the framework.

## Repository layout

```text
.
├── README.md
├── bench/
│   ├── README.md
│   └── target_probe.c
├── docs/
│   └── ARCHITECTURE.md
├── include/cpu_llms/
├── src/
├── tests/
├── tools/
└── models/
    └── gemma-4-e2b/
        ├── README.md
        ├── layer0-ranges.json
        ├── layer0-validation.json
        ├── pins.json
        └── task-profiles/
            ├── hazard-v1.json
            └── hazard-v1-results.json
```

- [Compiler and runtime architecture](docs/ARCHITECTURE.md)
- [Gemma 4 E2B engineering plan](models/gemma-4-e2b/README.md)
- [Target probe](bench/README.md)

## Correctness test

```sh
make fixture
make test
```

The committed fixture is synthetic and small. A separate ignored fixture validates layer 0 with pinned official weights and real embedding/PLE rows. The bounded task profile additionally exercises the complete text graph, but it does not replace full tensor-boundary or product-quality validation.

## Implementation rule

Claims are separated into four categories:

1. facts taken from primary model or hardware documentation;
2. observations collected from the target machine;
3. analytical estimates;
4. measured project results.

An estimate is not reported as a benchmark. An approximate graph transformation is not reported as an exact rewrite. A model is not reported as supported until its graph, numerical boundaries, output behavior, memory use, and target-machine execution have been validated.

## License

No license has been selected.
