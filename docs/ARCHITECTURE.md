# Compiler and Runtime Architecture

Status: design phase; runtime implementation has not started.

## 1. Objective

The project compiles a language-model checkpoint and a target hardware profile into a local CPU deployment artifact.

```text
model checkpoint
+ tokenizer revision
+ target hardware profile
+ memory, latency, and quality constraints
+ optional bounded workload profile
+ compiler search and validation data
                |
                v
          offline compiler
                |
                v
       versioned deployment bundle
```

Offline compilation may use large amounts of CPU, GPU, RAM, storage, and time. The target system does not train, search, select kernels, or interpret a general model graph. It executes a graph, layout, kernel set, and schedule selected in advance.

The framework is not tied to Gemma 4. Each supported model family provides an adapter. Each CPU family provides a hardware backend.

## 2. System boundaries

```text
Deployment frontend
  input contract / prompt domain / output schema / quality gates

Model adapter
  checkpoint / tokenizer / graph inventory / reference oracle
  exact rewrites / approximate candidates / numerical boundaries

Hardware backend
  ISA kernels / data layout / static arena / threading / artifact ABI
```

The deployment frontend must not contain model layer names. The hardware backend must not contain application labels, policy text, or prompt templates. The model adapter is the boundary between an architecture-specific graph and the compiler's common intermediate representation.

## 3. Compiler inputs

A compile job requires a versioned deployment specification.

| Input | Required content |
|---|---|
| Deployment identity | ID, revision, intended use |
| Input contract | Schema, field types, maximum lengths, free-text policy |
| Prompt domain | Fixed prefixes, templates, variable slots, tokenization rules |
| Output contract | Labels, structured schema, optional bounded explanation |
| Safety policy | Hard rules, out-of-domain handling, non-overridable conditions |
| Model source | Checkpoint, tokenizer, reference implementation revision |
| Target profile | ISA, cores, caches, RAM, storage, operating-system ABI |
| Resource limits | RSS, swap, decision latency, artifact size |
| Quality gates | Application metrics, teacher divergence, error limits |
| Data partitions | Search, held-out validation, boundary, and out-of-domain sets |

The compiler may validate this specification. It cannot infer the intended labels, safety policy, or acceptance thresholds from a prompt string.

## 4. Optional bounded-input specialization

A deployment may define a bounded prompt or input domain:

```text
fixed instruction prefix
+ one of a finite set of templates
+ variable fields governed by a schema
+ fixed output request
```

This condition can permit the compiler to:

- precompute a causal fixed prefix and package a revision-bound state snapshot;
- search quantization, low-rank, pruning, and routing choices on the expected hidden-state distribution;
- generate a fixed label head or restrict LM-head candidate rows;
- generate separate layouts and schedules for known context-length buckets;
- remove embedding rows only when an allowed-token grammar proves that they are unreachable.

A bounded prompt domain does not imply a bounded token set. Free-text input can still require the complete tokenizer, input embedding, and token-indexed model data.

The runtime must reject, escalate, or route inputs outside the compiled contract. A specialized artifact provides no general-model guarantee outside its validated domain.

This specialization is an optimization input. It is not the identity or purpose of the project.

## 5. Compile-time resource model

The design places no practical limit on offline search resources. A compile job may:

- record teacher traces and intermediate activations over large corpora;
- search per-layer bit allocation, codebooks, outliers, low-rank corrections, and routing structures;
- jointly optimize discrete codes, memory layout, and hardware schedules;
- score candidates using layer error, final loss, output ranking, and application metrics;
- emit a fixed schedule for one checkpoint and one CPU profile.

Unlimited offline compute does not imply arbitrary lossless compression. Runtime image size, bytes read per inference step, instruction throughput, and approximation error remain hard constraints. Finite validation data also cannot establish formal correctness for all variable inputs.

## 6. Compile pipeline

1. Validate the deployment specification, data partitions, and target profile.
2. Pin the checkpoint, tokenizer, and reference implementation revisions.
3. Build a reference oracle at defined tensor boundaries.
4. Expand the bounded input domain, when supplied, into token sequences and context buckets.
5. Ask the model adapter for exact rewrites and approximate candidates.
6. Search quantization, codebooks, low-rank corrections, structured pruning, and conditional routing.
7. Lower the selected graph through the hardware backend.
8. Generate kernels, data layout, thread schedules, and static memory arenas.
9. Compare candidates against the oracle on held-out, boundary, and out-of-domain data.
10. Measure RSS, swap, latency, and output conformance on the target machine or an equivalent runner.
11. Produce a deployable release only if every hard gate passes.
12. Otherwise, preserve failure reports and candidate measurements without marking the model as supported.

The target machine does not retrain routers or search for a different kernel configuration.

## 7. Model-adapter contract

Each model architecture requires an adapter with the following capabilities.

| Capability | Requirement |
|---|---|
| Checkpoint import | Pin the revision and validate tensor names, shapes, and data types |
| Tokenizer import | Produce stable token IDs and support allowed-token analysis |
| Graph inventory | Describe layers, sequence operators, MLPs, normalization, embeddings, heads, and state semantics |
| Reference execution | Export oracle values at declared tensor boundaries |
| Exact rewrites | Declare transformations that preserve the defined semantics |
| Approximate candidates | Declare quantizable, prunable, factorable, or conditionally routed components |
| Error boundaries | Define layer-level and end-to-end comparison points |
| Runtime lowering | Lower the graph to the backend primitive set |

Adapters may expose GQA or MQA, MoE routing, shared layers, state-space blocks, local attention, multimodal branches, or model-specific embeddings. The common compiler must not assume that any one of these structures exists.

Reading a checkpoint is not sufficient for model support. Support requires graph and oracle agreement, validation of exact rewrites, sensitivity data for approximate transformations, primitive coverage, and end-to-end regression tests.

## 8. Hardware-backend contract

The hardware profile records at least:

- the ISA and available SIMD instructions;
- core and thread topology;
- cache sizes and sharing relationships;
- NUMA topology, memory capacity, and sustained memory bandwidth;
- storage capacity, sequential throughput, and random-access constraints;
- operating-system ABI, page size, memory-mapping behavior, and affinity support.

The backend generates or selects:

- packed integer, codebook-LUT, and required floating-point primitives;
- architecture- and layer-specific fusion;
- weight, code, index, and metadata layout;
- a static memory arena;
- thread partitions and reduction schedules;
- prefetch, memory-map, and residency policy.

When the target machine has not been measured, the compiler may emit an analytical resource budget. It must not claim that a latency requirement has been met from theoretical bandwidth alone.

## 9. Intermediate representation

The common intermediate representation must retain the properties required for validation and lowering:

- tensor shape, logical axes, and physical layout;
- state lifetime and aliasing rules;
- exact versus approximate operation status;
- numerical accumulation and rounding requirements;
- token-, layer-, and context-dependent access patterns;
- target quality gates for approximate subgraphs;
- provenance back to checkpoint tensors and compiler passes.

Architecture-specific information is preserved until the backend no longer needs it. Prematurely lowering every operation to generic matrix multiplication would discard reusable KV relationships, local-window structure, tied weights, and other model-specific opportunities.

## 10. Deployment artifact

The logical output is a versioned local deployment bundle:

```text
generated executable
immutable packed model image
optional fixed-prefix state snapshots
deployment and schema manifest
checksums and source provenance
license and notice files when required
```

The executable and large model image may remain separate to support memory mapping, atomic updates, and rollback. A distribution format may wrap them in one package, but the runtime sections and checksums remain independent.

The target machine does not require Python, training code, calibration data, a general inference framework, or a remote inference service. Initial interfaces should be record-oriented stdin/stdout, a Unix socket, or an embedded C ABI. A general HTTP server is outside the runtime core.

Given the same bundle, input, and deterministic decision policy, the runtime should produce reproducible output. A fixed classification path does not require open-ended random sampling.

## 11. Recompilation boundary

Any of the following changes creates a new compile job:

- checkpoint or tokenizer revision;
- prompt prefix, template, input schema, or output schema;
- labels, quality gates, or safety policy;
- target CPU, ISA, RAM limit, or operating-system ABI;
- quantization, codebook, pruning, or routing configuration;
- a material change to the compiler-search distribution.

The deployed artifact does not modify its own weights. A new version is generated and validated offline. The target verifies its checksum before an atomic switch. The previous version remains available for rollback and decision audit.

## 12. Validation requirements

Validation is divided into four levels.

### 12.1 Primitive validation

- scalar reference implementation;
- packed-format encode/decode tests;
- kernel differential tests over boundary values;
- overflow, saturation, alignment, and tail handling tests.

### 12.2 Graph validation

- tensor-by-tensor comparison against the pinned oracle;
- explicit tolerances for each numerical boundary;
- exact-rewrite equivalence tests;
- state and cache replay tests.

### 12.3 Model validation

- teacher-forced token agreement;
- greedy decode agreement for the correctness configuration;
- quality measurements for every approximate configuration;
- regression sets covering context lengths and atypical tokens.

### 12.4 Target validation

- peak and steady-state RSS;
- swap usage;
- cold-start and warm execution time;
- prefill and decode measurements;
- sustained frequency and temperature;
- bytes read per layer and per token;
- one-, two-, and full-core scaling;
- artifact checksum and rollback tests.

## 13. Repository organization

Cross-model contracts are stored under `docs/`. Model-specific graph details, budgets, and experiments are stored under `models/<model-name>/`.

```text
docs/
  ARCHITECTURE.md
models/
  <model-name>/
    README.md
```

A model document must not redefine the common adapter or backend contracts. The architecture document must not contain model-specific layer constants or kernel schedules.

Current model record:

- [Gemma 4 E2B](../models/gemma-4-e2b/README.md)

## 14. First validation path

```text
model:        Gemma 4 E2B IT, text only
hardware:     Intel Celeron J3455, four cores, SSE4.2, 3.34 GiB RAM
resident set: target 960 MiB; reject above 1,024 MiB
context:      at most 512 tokens
output path:  fixed risk labels; optional bounded explanation
runtime:      generated C executable and immutable packed image
```

This path tests whether the adapter, backend, compiler search, artifact generation, and local execution form a complete system. It does not establish support for another model, CPU, or workload.
