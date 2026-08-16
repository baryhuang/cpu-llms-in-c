# MiniMax-H3 Apple M3 Pro compiler

Status: geometry, AOT AdaLN image layout, tree routes, branch-cache schedule
validation, sparse projection work accounting, local/HTTP safetensors
inspection and one-layer affine-Q4 packing are implemented. A complete
official-checkpoint quantizer and target artifact emitter are not implemented.

## Inputs

| Input | Pinned source |
|---|---|
| H3 Base FL2VA transformer | `MiniMaxAI/MiniMax-H3@42ed227e`, `FL2VA/transformer/` |
| T2VA conditioner | same revision, `FL2VA/text_encoder/` and `FL2VA/processor/` |
| Video VAE | same revision, `FL2VA/video_vae/source/` |
| Audio VAE | same revision, `FL2VA/audio_vae/` |
| Turbo challenge | `larryvrh/MiniMax-H3-Turbo-Lora@43a74557`, v4 step-600 EMA |
| Source manifest | [`models/minimax-h3/source.json`](../../../models/minimax-h3/source.json) |
| Target | [`models/minimax-h3/targets/apple-m3-pro/`](../../../models/minimax-h3/targets/apple-m3-pro/) |

The compiler must reject a source file whose size or SHA-256 does not match the manifest. It must read only the selected FL2VA workflow; duplicated root and Ref2VA weights are outside the input set.

## Compile phases

| Phase | Output |
|---|---|
| Safetensors inspection | tensor name, dtype, shape, source shard and byte range |
| Graph pruning | Qwen layers 0-49 only; no LM head; no unused H3 graph branch |
| Workload specialization | fixed benchmark prompt/context, 864×480×124 layout, explicit Turbo-4 base sigmas `[1,.75,.5,.25,0]` and separate shift-12/shift-3 schedules |
| AdaLN evaluation | exact per-block `(step,modality)` tables after applying the Turbo LoRA contribution |
| Quantization search | per-tensor W4/W8 choice, group size, scales, error record and fallback format |
| Activation-subspace search | optional task-profile low-rank residuals and exception rows, accepted only by dense-teacher trajectory gates |
| Attention compilation | static tree geometry, exact sink rows, route-refresh policy and dense fallback |
| Packing | page-aligned immutable phase images and checksums |
| Validation export | small independent boundary fixtures and expected hashes |

Compilation may use remote ranges and process one source tensor at a time. It must not require all 144 GB of released weights to coexist on local storage.

## Planned artifacts

```text
prompt-context.h3ctx
transformer-global.h3m3
transformer-layer-00.h3m3 ... transformer-layer-49.h3m3
adaln-turbo4.h3aot
attention-tree-864x480x124.h3tree
branch-cache-schedule.h3cache
video-vae.h3m3
audio-vae.h3m3
manifest.json
```

Large generated artifacts remain outside Git. Their format headers, compiler and validation code belong in this repository.

The current C layout planner specializes only combinations that the fixed T2VA
graph reaches. Across four evaluations that is 12 block-modulation slots:
`(video timestep, video)`, `(video timestep, text)` and
`(audio timestep, audio)` per evaluation. The final norm depends only on time;
the shared zero timestep is deduplicated, leaving seven rows. At FP16 the
50-block AdaLN data is 38,707,200 bytes and the final-norm data is 150,528
bytes; the complete 16 KiB-aligned image plan is 38,895,616 bytes.

The static tree compiler follows H3 geometry rather than generic sequence
blocks. For 864×480×124 it emits a 500-row exact conditioning range and a
342-node video tree: 296 at-most-8×8 spatial leaves, 37 frame nodes, eight
five-latent temporal groups, and one root. Leaf metadata maps directly to the
packed video rows; no padded token is introduced.

The measured performance route stores the residual stream in tree-major
physical order. A retained logical-to-physical row map preserves H3's packed
semantics. The initial compiled-prompt seed uses 11 contiguous text summaries,
13 audio summaries, and expands one video leaf. Contiguous text groups are not
a release policy: offline dense-teacher search must choose grouping and route
budgets against all four denoise evaluations before the compiler emits them.

The route search records one budget per `(Turbo evaluation, 10-layer band,
modality)`. Initial selection features are previous-evaluation LSE,
input-token delta, consecutive reuse count and temporally smoothed
neighboring-frame scores. For reused rows the artifact stores ungated
attention/MLP branch outputs; runtime applies the current evaluation's exact
AOT gate. Dense boundary error and final media metrics, not the search score
alone, decide whether a schedule is valid.

Evaluation zero has no prior branch cache and is dense in the conservative
schedule. Evaluation one may reduce its keep ratio sharply; ratios must then
be nondecreasing through evaluations two and three. A separate future
bootstrap predictor may reconstruct evaluation-zero omitted rows from spatial
and temporal tree representatives, but it is not encoded as cache reuse.

For sparse K/V, the compiler emits post-RMSNorm/AdaLN summary reducers before
the projection images. Since the released H3 K/V projections are bias-free,
projecting one summary is identical to averaging the corresponding projected
rows. The emitted route must still be compared with dense attention because a
single summary key/value is not equivalent to retaining every attention
candidate.
