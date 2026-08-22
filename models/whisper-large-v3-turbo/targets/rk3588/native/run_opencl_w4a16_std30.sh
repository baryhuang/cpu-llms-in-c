#!/bin/sh
set -eu

GPU_ROOT=${1:-.}
MODEL_ROOT=${2:-$GPU_ROOT}
OUT=${OUT:-"$GPU_ROOT/logs"}

CPU0=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
CPU4=/sys/devices/system/cpu/cpufreq/policy4/scaling_governor
CPU6=/sys/devices/system/cpu/cpufreq/policy6/scaling_governor
GPU=/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/governor
NPU=/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/governor
DMC=/sys/class/devfreq/dmc/governor

cpu0_old=$(cat "$CPU0")
cpu4_old=$(cat "$CPU4")
cpu6_old=$(cat "$CPU6")
gpu_old=$(cat "$GPU")
npu_old=$(cat "$NPU")
dmc_old=$(cat "$DMC")

restore_governors() {
  echo "$cpu0_old" >"$CPU0"
  echo "$cpu4_old" >"$CPU4"
  echo "$cpu6_old" >"$CPU6"
  echo "$gpu_old" >"$GPU"
  echo "$npu_old" >"$NPU"
  echo "$dmc_old" >"$DMC"
}
trap restore_governors EXIT HUP INT TERM

echo performance >"$CPU0"
echo performance >"$CPU4"
echo performance >"$CPU6"
echo performance >"$GPU"
echo performance >"$NPU"
echo performance >"$DMC"

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$MODEL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OCL_ICD_VENDORS="$GPU_ROOT/icd"

taskset -c 4-7 "$GPU_ROOT/bin/whisper_opencl_w4a16" \
  "$MODEL_ROOT/models/whisper-large-v3-turbo-encoder-w4a16-v3.llmc" \
  "$MODEL_ROOT/models/whisper-large-v3-turbo-decoder-w4a16-v3.llmc" \
  "$MODEL_ROOT/vocab.json" "$MODEL_ROOT/std30.wav" llmc \
  --linear-backend "${LINEAR_BACKEND:-hybrid}" \
  --attention-backend "${ATTENTION_BACKEND:-hybrid}" \
  --npu-scheduler parallel3 2>&1 |
  tee "$OUT/std30-opencl-w4a16.log"
