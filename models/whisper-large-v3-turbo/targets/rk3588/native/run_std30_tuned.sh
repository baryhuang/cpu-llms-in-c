#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RUNNER ENCODER.rknn DECODER.rknn vocab.json std30.wav RKNN_LIB_DIR" >&2
  exit 2
fi

runner_path=$1
encoder_path=$2
decoder_path=$3
vocab_path=$4
audio_path=$5
rknn_library_dir=$6

cpu_governor_files=(/sys/devices/system/cpu/cpufreq/policy*/scaling_governor)
old_cpu_governors=()
for governor_file in "${cpu_governor_files[@]}"; do
  old_cpu_governors+=("$(cat "$governor_file")")
done

npu_governor_file=/sys/class/devfreq/fdab0000.npu/governor
dmc_governor_file=/sys/class/devfreq/dmc/governor
old_npu_governor=$(cat "$npu_governor_file")
old_dmc_governor=$(cat "$dmc_governor_file")

restore_governors() {
  for index in "${!cpu_governor_files[@]}"; do
    printf '%s' "${old_cpu_governors[$index]}" > "${cpu_governor_files[$index]}"
  done
  printf '%s' "$old_npu_governor" > "$npu_governor_file"
  printf '%s' "$old_dmc_governor" > "$dmc_governor_file"
}
trap restore_governors EXIT

for governor_file in "${cpu_governor_files[@]}"; do
  printf '%s' performance > "$governor_file"
done
printf '%s' performance > "$npu_governor_file"
printf '%s' performance > "$dmc_governor_file"

LD_LIBRARY_PATH="$rknn_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  taskset -c 4-7 "$runner_path" \
  "$encoder_path" "$decoder_path" "$vocab_path" "$audio_path" \
  --core-mask all
