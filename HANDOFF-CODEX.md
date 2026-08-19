# Handoff: Whisper large-v3 / large-v3-turbo on Jetson Orin

## Mission

1. **Baseline test** for `openai/whisper-large-v3` and `openai/whisper-large-v3-turbo` on the Jetson below, using public inference stacks.
2. **Then optimize incrementally.** Record every step: what changed, measured before/after with raw records, negative results included. One commit per measured increment.
3. **One unified whisper bench for everything** — baselines, every increment, and the final result must be measured by the same methodology. Use **public** tools only; do not build a custom benchmark. (Precedent in this repo: NVIDIA AIPerf was used for LLM serving numbers — pick the ASR equivalent.)

Suggested public benching, subject to your verification on-device:

- End-to-end: a pinned LibriSpeech `test-clean` subset (record the exact file list), WER via `jiwer` with the Whisper text normalizer, throughput as RTFx (audio seconds / wall seconds). The Hugging Face `open_asr_leaderboard` harness is the closest public standard.
- Kernel/stage level: `whisper.cpp`'s public `whisper-bench` where whisper.cpp is the arm being measured.
- Report both models in every table. Same audio set, same normalizer, same machine state for all arms.

Reasonable baseline arms (verify what installs cleanly on this JetPack): `whisper.cpp` (CUDA backend), `faster-whisper` (CTranslate2 CUDA), Hugging Face `transformers` as reference. Measure WER and RTFx per arm per model.

## The machine

```
ssh root@100.76.66.42        # hostname: caremojo
```

- Jetson Orin ("Orin Super" per owner), aarch64, **6 CPU cores, 7 GB unified RAM total** (CPU+GPU shared — watch memory: large-v3 fp16 weights are ~3 GB before activations)
- L4T R39.2.1 (2026-08-07 build), kernel 6.8.12-tegra, driver 595.78, **CUDA 13.2**
- Storage: 467 GB, 435 GB free at `/`
- `nvidia-smi` is limited on Tegra; use `tegrastats` for utilization/power
- **Pin the power mode before benching** (`nvpmodel -q`, `jetson_clocks`) and record it in results; all comparisons same-mode

## Repository conventions (must follow)

- Commits authored solely by **Bury Huang** — no co-sign, no AI attribution, no `Co-Authored-By`.
- Every performance claim needs a measured number and a raw record in the target's `results.json`. Honest negative results are recorded, not deleted.
- Directory layout: model first, target second — follow `models/whisper-small.en/targets/a113x/` as the pattern; this work goes under `models/whisper-large-v3/targets/jetson-orin/` (and turbo either as a second model dir or a documented variant — decide and be consistent).
- Pin sources: checkpoint revisions and SHA-256 in the model README; packers verify hashes. No weights, checkpoints, or credentials in git. Big artifacts live on the device or in `tmp/` (untracked).
- Root README gets only final results (a row in the "Implemented model × target pairs" table); method details live in the target README + results.json.
- Quality gate for ASR: WER on the pinned set must not regress versus the baseline of the same model. State the gate result with every increment.

## Hard-won lessons from this machine-pair's sibling work (read before optimizing)

1. **A kernel-bench win does not transfer automatically.** A GEMM kernel measured 1.78× faster standalone ran 2.6× slower inside the real pipeline (commit `12b9736`). Confirm every kernel change with in-pipeline timings before committing.
2. **Same-window A/B only.** Thermal drift makes cross-window comparisons unreliable; run compared arms back-to-back and record machine state.
3. **Killed GPU runs can leak wired memory** (19 GB seen on the Mac sibling). Check free memory/swap before believing any timing; reboot clears it. On 7 GB unified memory this machine will be far more sensitive.
4. **Quality gates need the right granularity.** Whole-run averages hid localized quality regressions in the video work; for ASR, per-file WER deltas catch what corpus-average WER hides.
5. Numerics changes in kernels can be individually tiny yet change outputs; for ASR the gate is WER-neutrality on the pinned set, and bit-identical transcripts are the gold standard when claiming "lossless".

## State at handoff

- Recon done (numbers above); nothing installed or benched yet on the Jetson.
- Repo: `github.com/baryhuang/llm-in-c`, main at `c920b5f`, tree clean.
- The sibling Whisper target (`models/whisper-small.en/targets/a113x/`) shows the intended shape of a finished target: pinned sources, packed images, C runtime, results.json with WER + RTF + RSS.
