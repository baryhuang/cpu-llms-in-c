# Handoff: finish the five-workload cross-stack comparison

Written 2026-08-15 for a Codex session to continue. Everything below is
verified state, not intention. All prior work is committed and pushed
on `main` (HEAD `44a1c74` plus the `tools/compare/` commit carrying
this file). All processes from the previous session are killed; nothing
is running.

## The task in one sentence

Measure the remaining five of six "five-workload set" runs
(mlx-lm/oMLX/llama.cpp on Qwen3.6-27B and Qwen3.8-27B), then rewrite
the root `README.md` **Throughput** section so both tables are complete
cross-stack comparisons with model names and terse cells.

## Why (reviewer feedback that drives the doc rewrite)

The repo owner's review rules for the root README — follow them
exactly:

- Final results only. No roadmap, no development history, no internal
  jargon ("battery", "pin", "landed", env-variable inventories).
- A comparison table must carry model names in it; the reader must not
  need prose to know what was measured.
- Table cells are words or numbers, not sentences.
- One workload per table; do not mix workloads in one table (that
  produced three contradictory "end-to-end" values before).
- End-to-end tokens/s (reply tokens / full request wall) is the primary
  metric; decode and prompt rates are components.
- Every number in prose must appear in a table or be labeled with its
  workload.
- Edit documents by reading them, not by scripted string replacement.

The current root Throughput section (read it first) has one complete
table (single request, four stacks) and one incomplete table
(five-workload set, only this runtime). The reader questions it fails:
"what is one request", "what is five-workload", "why no cross-stack
comparison in the second table". The measurements below fill the second
table; then fix the labels: name the request shape in the header (e.g.
"36-token prompt, 30-token reply") and the set (e.g. "five requests,
28-1,757-token replies").

## Measurement method (already established; keep identical)

Five fixed workloads, greedy, temperature 0, max 1,800 new tokens,
model resident, one short warmup request first. Per request record
reply tokens and wall seconds; the set aggregate is
`sum(tokens)/sum(wall)`. The workload texts and the shared template
renderer are in `tools/compare/five_workloads.py`. End-to-end rate =
reply tokens / full request wall (prompt reading included).

Runner scripts (paths inside are absolute to this machine):

| Script | Stack | Usage |
|---|---|---|
| `tools/compare/run_mlxlm_set.py` | bare mlx-lm | `tmp/compare-venv/bin/python tools/compare/run_mlxlm_set.py <model_dir> <out.json>` |
| `tools/compare/run_omlx_set.py` | oMLX BatchedEngine, `enable_thinking=False` | same arg shape |
| `tools/compare/run_llamacpp_set.py` | llama-server via HTTP | `python3 tools/compare/run_llamacpp_set.py <port> <out.json>` (start `llama-server -m <gguf> -c 4096 -fa 1 --port <port>` first) |
| `tools/compare/run_all_sets.sh` | orchestrates all six sequentially, resumable (skips existing out files) | `sh tools/compare/run_all_sets.sh` |
| `tools/compare/dl_q36_mlx.sh` | parallel-ranged download of the 3.6 mlx checkpoint, resumable | called by the orchestrator |

Caveat found while writing `run_omlx_set.py`: it counts stream events
as tokens with `stream_interval=1` and prefers a usage/token-count
attribute when the final output object has one — verify against the
first run's numbers before trusting.

## Which runs exist, which remain

Done (result committed): `tools/compare/set-llama-36.json` —
llama.cpp b10360 + Unsloth Qwen3.6 Q4_K_M, five-workload aggregate
**6.87 tok/s** over 3,217 reply tokens.

Remaining five (write results next to the scripts):

1. mlx-lm on Qwen3.6 → `set-mlxlm-36.json` (needs the 3.6 mlx
   checkpoint, see below)
2. oMLX on Qwen3.6 → `set-omlx-36.json` (same checkpoint)
3. mlx-lm on Qwen3.8 → `set-mlxlm-38.json` (checkpoint ready:
   `tmp/qwen38-27b-4bit`)
4. oMLX on Qwen3.8 → `set-omlx-38.json` (same)
5. llama.cpp on Qwen3.8 → `set-llama-38.json` (GGUF ready:
   `tmp/qwen38-27b-unsloth/Qwen3.8-27B-Q4_K_M.gguf`, sha256 verified
   `7e78da5d…6fe169`)

Run them one at a time — each stack loads ~15-17 GB into a 36 GB
machine. Kill any resident `qwen36-m3-chat`/`qwen36_serve.py` first.

## Machine state

- `tmp/compare-venv`: python3.13, mlx 0.32.0, mlx-lm 0.31.3,
  oMLX 0.5.7 (installed from `git+https://github.com/jundot/omlx@v0.5.7`;
  not on PyPI). These are the versions pinned by the published
  comparison — do not upgrade.
- llama.cpp b10360 via Homebrew (`llama-server`, `-fa 1`, context 4096).
- 3.6 mlx checkpoint download in progress at `tmp/qwen36-27b-mlx/`:
  shards 1-2 complete, shard 3 partial in `*.part??` files; rerun
  `sh tools/compare/dl_q36_mlx.sh` to resume and finish (it verifies
  sizes; source `mlx-community/Qwen3.6-27B-4bit` revision `c000ac2c…`,
  the exact revision this runtime's images were compiled from, so
  mlx-lm/oMLX compute on the same weight values).
- `tmp/qwen36-27b-4bit/` is NOT a checkpoint (bring-up artifacts only)
  — do not point mlx-lm at it.
- The 3.6 Unsloth GGUF was deleted for disk space after its run; to
  re-measure, re-download `unsloth/Qwen3.6-27B-GGUF` Q4_K_M revision
  `82d411ac…`, sha256 `5ed60d0a…392a0`.
- This runtime's own five-workload numbers (already published,
  `results.json` of each target): Qwen3.6 9.42 tok/s speculative /
  7.91 plain over 3,394 reply tokens; Qwen3.8 9.66 / 7.94 over 3,305.
  Note the caveat when comparing: reply token counts differ per stack
  because each stack generates its own reply (llama.cpp runs a
  different Q4 quantization; mlx-lm/oMLX share our weight values).

## After the numbers land

1. Root `README.md` Throughput section: make the five-workload table
   cross-stack — rows Qwen3.6-27B and Qwen3.8-27B, columns This
   runtime (speculative), This runtime (plain), llama.cpp, mlx-lm,
   oMLX. Rename both table headers so they self-describe the workload.
   Keep the four-line closing note (end-to-end definition, lossless
   speculation, why set rates exceed the single-request rate,
   per-case links).
2. Add the raw set results to
   `models/qwen3.6-27b/targets/apple-m3-pro/results.json` and
   `models/qwen3.8-27b/targets/apple-m3-pro/results.json` (new block,
   e.g. `five_workload_cross_stack`), and mirror the table into each
   target README's comparison section and REVIEW.html if the numbers
   change any stated ratio.
3. Status-table performance cells in the root: update the ratios if
   the five-workload comparison changes them (currently quoted from
   the single request: 1.5-1.6x mlx-lm/oMLX, 1.46x llama.cpp for 3.6;
   1.56x llama.cpp for 3.8).
4. Commit style: plain-language messages, facts and numbers, no
   co-author lines, author Bury Huang only.

## Repo conventions that bit us before (avoid re-learning)

- Every performance claim needs the measured number next to it and a
  raw record in `results.json`.
- Greedy speculative decoding must stay token-identical to plain
  decoding; any kernel change reruns
  `build/qwen36-m3-prefill-parity-test` (36 checks) and the 4-prompt
  machine-mode battery for both models.
- The tokenizer/pack/runtime pin every source by SHA-256; new inputs
  need their hash added, not a bypassed check.
- `tmp/` is uncommitted scratch; model weights never enter Git.
