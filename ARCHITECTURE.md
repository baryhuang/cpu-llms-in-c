# Compiler and Runtime Architecture

Status: one bounded Gemma 4 E2B artifact is implemented. The cross-model compiler is not complete.

## Output

```text
checkpoint + tokenizer + workload profile + CPU profile
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
| Deployment profile | prompt/input schema, labels, output schema, limits, validation data |
| Model adapter | checkpoint import, tokenizer, graph, exact rewrites, numerical oracle |
| CPU backend | kernels, tensor layout, static memory, threads, prefetch and mapping policy |
| Runtime | input validation, fixed graph execution, output formatting |

Application labels do not belong in the CPU backend. Model layer names do not belong in the deployment profile.

## Compile inputs

A release must pin:

- checkpoint, tokenizer, and reference revisions;
- input and output contracts;
- fixed prompt text and variable-field rules;
- CPU ISA, core/cache topology, RAM and storage constraints;
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

## Model-adapter requirements

- verify tensor names, shapes, data types, and hashes;
- preserve state sharing, local windows, tied weights, and architecture-specific embeddings;
- expose scalar or independent reference values at declared boundaries;
- distinguish exact rewrites from approximate transformations;
- lower every executed operation to backend primitives.

Reading a checkpoint is not model support. Support requires graph execution, reference comparison, target measurements, and workload validation.

## CPU-backend requirements

- record ISA, cores, caches, NUMA, RAM, storage, ABI, and page size;
- generate scalar and ISA-specific packed kernels;
- define matrix order, alignment, block format, and prefetch policy;
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

```text
include/   public C interfaces
src/       C runtimes
tools/     offline compiler and reference tools
models/    one directory per model family or checkpoint
tests/     committed small fixtures and tests
```

Model-specific constants and results stay under `models/<model>/`. Cross-model contracts stay in this file. Checkpoints, generated model images, binaries, machine credentials, and access tokens are not committed.

Current model record: [`models/gemma-4-e2b/`](models/gemma-4-e2b/).
