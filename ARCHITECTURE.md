# Compiler and Runtime Architecture

Status: the repository contains measured model-target implementations for Qwen3.8-27B, Qwen3.5-0.8B and Whisper small.en, plus an earlier restricted Gemma 4 E2B artifact. Packing and runtime code still use model adapters; a single declarative cross-model compiler is not complete.

## Output

```text
checkpoint + tokenizer + workload profile + target profile
                         |
                         v
                  offline compiler
                         |
                         v
             C executable + packed image
```

Compilation may use large development machines and long searches. The deployment target executes a fixed graph and image. It does not load Python, a training stack, or a general inference framework.

## Components

| Component | Owns |
|---|---|
| Deployment profile | prompt/input schema, output schema, limits, validation data; label sets only for restricted specializations |
| Model adapter | checkpoint import, tokenizer, graph, exact rewrites, numerical oracle |
| Target backend | CPU/NPU kernels, tensor layout, static memory, transfer boundary, threads, prefetch and mapping policy |
| Runtime | input validation, fixed graph execution, output formatting |

Application labels do not belong in the target backend. Model layer names do not belong in the deployment profile.

## Compile inputs

A release must pin:

- checkpoint, tokenizer, and reference revisions;
- input and output contracts;
- fixed prompt text and variable-field rules;
- CPU ISA, core/cache topology, accelerator API/data types, shared-memory topology, RAM and storage constraints;
- image-size, RSS, swap, latency, and quality gates;
- search, held-out, boundary, and out-of-domain data partitions.

Changing any item creates a new artifact and requires validation again.

## Compile pipeline

1. Validate and pin every input.
2. Import the architecture graph and reference implementation.
3. Apply architecture-preserving rewrites.
4. Expand any bounded input profile and record reachable states or tokens.
5. Search quantization, codebooks, pruning, layout, and schedules.
6. Emit the packed image and C runtime configuration.
7. Compare numerical boundaries and task outputs with the reference.
8. Measure memory, swap, cold/warm latency, and output behavior on the target.
9. Release only when the declared gates pass.

The target does not repeat steps 1-6.

## Bounded-input specialization

A deployment may contain a fixed prefix, a finite set of templates, and schema-limited fields. The compiler may then precompute fixed-prefix state, remove unreachable embedding rows, restrict output rows, or select length-specific layouts.

These removals require proof from the input contract. A free-text field normally requires the full tokenizer and all reachable token rows. Inputs outside the compiled contract must be rejected or sent to another artifact.

The current Gemma 4 artifact is more restricted: it stores twelve complete token sequences and accepts only their numeric case indices. See [`REVIEW.html`](REVIEW.html).

Task outputs are defined by the prompt at run time, not by the compiled artifact. The default runtime carries the tokenizer and the full output head, accepts a prompt and an optional per-invocation set of allowed answers, and scores only the requested answer rows after prefill; with no answer set it generates tokens normally. A fixed compiled-in label set — including the binary one-bit case with the `W_1 - W_0` decision-only rewrite — is an optional restricted specialization for extreme targets, not the contract. The first Gemma 4 artifact is such a special case.

## Model-adapter requirements

- verify tensor names, shapes, data types, and hashes;
- preserve state sharing, local windows, tied weights, and architecture-specific embeddings;
- expose scalar or independent reference values at declared boundaries;
- distinguish exact rewrites from approximate transformations;
- lower every executed operation to backend primitives.

Reading a checkpoint is not model support. Support requires graph execution, reference comparison, target measurements, and workload validation.

## Target-backend requirements

- record CPU ISA, cores, caches, accelerator, NUMA, RAM, storage, ABI, driver/runtime versions, and page size;
- generate scalar, ISA-specific, and selected accelerator kernels;
- define matrix order, alignment, block format, and prefetch policy;
- define which graph regions stay on CPU and which execute on an accelerator; account for dispatch, synchronization, and copies;
- use a bounded arena and fixed state lifetimes;
- define thread partitions and deterministic reductions;
- keep the model image resident when decode repeatedly visits its weights.

Analytical bandwidth estimates are not benchmark results.

## Artifact contract

```text
generated executable
immutable packed model image
deployment manifest
source and artifact checksums
optional fixed-prefix snapshots
```

Large images may remain separate from the executable for memory mapping and atomic replacement. The target verifies the artifact version and checksum before execution.

The first runtime interface is record-oriented or an embedded C ABI. HTTP serving is outside the runtime core.

## Validation gates

| Level | Required checks |
|---|---|
| Primitive | packing round-trip, kernel differential tests, tails, overflow and alignment |
| Graph | tensor-boundary comparison, rewrite equivalence, cache/state replay |
| Model | token/logit agreement and task metrics for each approximate artifact |
| Target | peak RSS, swap, cold/warm timing, throughput, page faults and reproducibility |

Smoke examples are recorded as smoke examples. They do not establish product quality or safety.

## Repository contract

A released artifact is one point in a cross product with two axes: one pinned model and one pinned CPU/SoC target. The classification order is fixed — model first, CPU/SoC target second:

```text
tools/                            shared on-target tooling: hardware and memory-bandwidth probe
                                  used to pin every target
compiler/                         offline compiler and reference tools
models/<model>/                   model axis: profile, pins, graph record, reference outputs
models/<model>/targets/generic/   the model's C runtime with model-axis optimizations only,
                                  portable to any CPU
models/<model>/targets/<soc>/     target axis: CPU/SoC and accelerator pin, specialized kernels,
                                  layouts, and results measured for this model x target pair
tests/                            committed small fixtures and tests
```

- The model directory holds everything independent of the target: input profile, checkpoint and tokenizer pins, graph constants, and reference outputs.
- `targets/generic/` holds the model's runtime source carrying only model-axis optimizations; it is the portable CPU starting point every pinned target builds from.
- Each `targets/<soc>/` directory holds everything specific to one model x target combination: CPU/SoC pin, optional accelerator boundary, kernel, layout and schedule choices, and benchmarks measured on that hardware.
- Changing either axis creates a new combination that requires its own validation. Results never transfer between combinations.

No Gemma 4 CPU target has been selected yet. The current Gemma 4 E2B results under `models/gemma-4-e2b/` were measured on an unpinned two-vCPU development machine; they move into the first pinned target directory when a target is selected.

Cross-model contracts stay in this file. Checkpoints, generated model images, binaries, machine credentials, and access tokens are not committed.

Current model records:

| Model | Measured target |
|---|---|
| [Qwen3.8-27B](models/qwen3.8-27b/) | [Apple M3 Pro](models/qwen3.8-27b/targets/apple-m3-pro/) |
| [Qwen3.5-0.8B](models/qwen3.5-0.8b/) | [Amlogic A113X](models/qwen3.5-0.8b/targets/a113x/) |
| [Whisper small.en](models/whisper-small.en/) | [Amlogic A113X](models/whisper-small.en/targets/a113x/) |
| [Gemma 4 E2B](models/gemma-4-e2b/) | unpinned legacy development machine |
