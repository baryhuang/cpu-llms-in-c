# cpu-llms-in-c

`cpu-llms-in-c` is an experimental compiler and runtime project for running language models on CPU-only systems.

The compiler takes a model checkpoint, a target CPU profile, deployment constraints, and an optional bounded input profile. It produces a model-specific C runtime and a packed model image. Existing inference frameworks may be used as correctness oracles during development, but they are not runtime dependencies.

## Status

Early scalar bring-up.

- A framework-independent scalar C implementation of one Gemma 4 early local decoder layer is present.
- The layer passes both a committed miniature fixture and an uncommitted real-weight Gemma 4 E2B layer-0 fixture at ten tensor boundaries on macOS and Linux x86-64.
- The offline tools can selectively read and range-fetch safetensors records without downloading an unused multimodal checkpoint in full.
- No tokenizer, runtime checkpoint loader, complete model runtime, or token decoder has been implemented.
- No model has passed end-to-end correctness validation.
- No performance result in this repository is a benchmark.
- Memory and throughput figures are engineering estimates until measured on the target hardware.

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
        └── pins.json
```

- [Compiler and runtime architecture](docs/ARCHITECTURE.md)
- [Gemma 4 E2B engineering plan](models/gemma-4-e2b/README.md)
- [Target probe](bench/README.md)

## Correctness test

```sh
make fixture
make test
```

The committed fixture is synthetic and small. A separate ignored fixture validates layer 0 with pinned official weights and real embedding/PLE rows. Neither result is end-to-end model validation.

## Implementation rule

Claims are separated into four categories:

1. facts taken from primary model or hardware documentation;
2. observations collected from the target machine;
3. analytical estimates;
4. measured project results.

An estimate is not reported as a benchmark. An approximate graph transformation is not reported as an exact rewrite. A model is not reported as supported until its graph, numerical boundaries, output behavior, memory use, and target-machine execution have been validated.

## License

No license has been selected.
