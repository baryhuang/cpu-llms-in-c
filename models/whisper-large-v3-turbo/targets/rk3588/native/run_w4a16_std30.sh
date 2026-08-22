#!/bin/sh
set -eu

ROOT=${1:-.}
NPU_SCHEDULER=${NPU_SCHEDULER:-parallel3}
OUT=${OUT:-"$ROOT/logs"}

ENCODER="$ROOT/models/whisper-large-v3-turbo-encoder-w4a16-v2.llmc"
DECODER="$ROOT/models/whisper-large-v3-turbo-decoder-w4a16-v2.llmc"
VOCAB="$ROOT/vocab.json"
AUDIO="$ROOT/std30.wav"
BIN="$ROOT/bin"
LD_LIBRARY_PATH="$ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

mkdir -p "$OUT"

sha256sum -c "$ROOT/SHA256SUMS"
"$BIN/w4a16_package_test" "$ENCODER" "$DECODER" |
  tee "$OUT/package-validation.log"
"$BIN/pcm16_wav_test" "$AUDIO" | tee "$OUT/wav-validation.log"

if [ ! -r /sys/kernel/debug/rknpu/load ]; then
  echo "warning: /sys/kernel/debug/rknpu/load is not readable; load fields will be unavailable" >&2
fi

"$BIN/w4a16_matmul_smoke" "$DECODER" model.proj_out.weight baseline \
  1 3 parallel3 >"$OUT/matmul-baseline.log" 2>&1
"$BIN/w4a16_matmul_smoke" "$DECODER" model.proj_out.weight llmc \
  1 3 parallel3 >"$OUT/matmul-llmc.log" 2>&1

baseline_checksum=$(awk '/^output_checksum:/ {print $2}' \
  "$OUT/matmul-baseline.log")
llmc_checksum=$(awk '/^output_checksum:/ {print $2}' \
  "$OUT/matmul-llmc.log")
if [ -z "$baseline_checksum" ] || [ "$baseline_checksum" != "$llmc_checksum" ]; then
  echo "error: baseline/LLMC MatMul checksums differ" >&2
  exit 1
fi

"$BIN/whisper_rknpu2_w4a16" "$ENCODER" "$DECODER" "$VOCAB" "$AUDIO" \
  baseline --npu-scheduler "$NPU_SCHEDULER" 2>&1 |
  tee "$OUT/std30-w4a16-baseline.log"

"$BIN/whisper_rknpu2_w4a16" "$ENCODER" "$DECODER" "$VOCAB" "$AUDIO" \
  llmc --npu-scheduler "$NPU_SCHEDULER" 2>&1 |
  tee "$OUT/std30-w4a16-llmc.log"

baseline_tokens=$(sed -n 's/^token_ids://p' \
  "$OUT/std30-w4a16-baseline.log")
llmc_tokens=$(sed -n 's/^token_ids://p' "$OUT/std30-w4a16-llmc.log")
if [ -z "$baseline_tokens" ] || [ "$baseline_tokens" != "$llmc_tokens" ]; then
  echo "error: baseline/LLMC Whisper token sequences differ" >&2
  exit 1
fi

echo "w4a16_std30_validation: passed"
