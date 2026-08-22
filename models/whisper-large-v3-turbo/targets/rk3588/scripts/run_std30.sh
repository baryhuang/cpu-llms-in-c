#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 /path/to/whisper-rk3588 /path/to/std30.wav /path/to/output-dir" >&2
    exit 1
fi

root=$1
audio=$2
output=$3
whisper=$root/whisper.cpp
backend=$root/ggml-rocket/build-dl-whisper-source/libggml-rocket.so
cli=$whisper/build/bin/whisper-cli

test -x "$cli"
test -f "$backend"
test -f "$audio"
mkdir -p "$output"

for policy in /sys/devices/system/cpu/cpufreq/policy*; do
    if [ -w "$policy/scaling_governor" ]; then
        echo performance > "$policy/scaling_governor"
    fi
done

export LD_LIBRARY_PATH=$whisper/build/bin:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_BACKEND_PATH=$backend
export ROCKET_CPU_AFFINITY=4-7
export ROCKET_N_THREADS=5
export ROCKET_MM_PROFILE=1

for mode in fp16 q8_0 q5_0 q4_k q4_0; do
    case $mode in
        fp16) model=$whisper/models/ggml-large-v3-turbo.bin ;;
        *) model=$whisper/models/ggml-large-v3-turbo-$mode.bin ;;
    esac
    log=$output/std30-$mode.log
    test -f "$model"
    /usr/bin/time -v taskset -c 4-7 "$cli" \
        -m "$model" -f "$audio" -t 4 -l en --no-timestamps >"$log" 2>&1
    awk -v mode="$mode" '
        /load time/   { load=$5 }
        /encode time/ { encode=$5 }
        /total time/  { total=$5 }
        END { printf "%s encoder=%.2fms e2e_excluding_load=%.2fms\n", mode, encode, total-load }
    ' "$log"
done
