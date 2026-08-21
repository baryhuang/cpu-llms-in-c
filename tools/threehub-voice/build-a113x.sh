#!/bin/sh

set -eu

if [ "$#" -ne 2 ]; then
    printf 'usage: %s OUTPUT_DIR WHISPER_CPP_SOURCE\n' "$0" >&2
    exit 2
fi

output_dir=$1
whisper_source=$2
script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repository=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
whisper_commit=41fc9dea6a4fe056424be86f61164413903fcff4

case "$(uname -s):$(uname -m)" in
    Linux:aarch64|Linux:arm64) ;;
    *)
        printf 'error: build inside a Linux/AArch64 environment\n' >&2
        exit 1
        ;;
esac

for command_name in cmake gcc git strip; do
    command -v "$command_name" >/dev/null 2>&1 || {
        printf 'error: required build command is missing: %s\n' "$command_name" >&2
        exit 1
    }
done

if [ "$(git -C "$whisper_source" rev-parse HEAD)" != "$whisper_commit" ]; then
    printf 'error: whisper.cpp must be pinned to %s\n' "$whisper_commit" >&2
    exit 1
fi

mkdir -p "$output_dir"

gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -static \
    -mcpu=cortex-a53 -mtune=cortex-a53 \
    "${script_dir}/threehub_audio_capture.c" \
    -o "${output_dir}/threehub-audio-capture-v1.0.0-linux-aarch64-a113x" -lm

gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp -static \
    -mcpu=cortex-a53 -mtune=cortex-a53 \
    "${repository}/models/qwen3.5-0.8b/targets/a113x/qwen35_task.c" \
    -o "${output_dir}/qwen35-task-resident-v1.0.0-linux-aarch64-a113x" -lm

cmake -S "$whisper_source" -B "${output_dir}/whisper-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_BUILD_SERVER=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=ON \
    -DGGML_CPU_ARM_ARCH=armv8-a+crypto+crc \
    -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG -mcpu=cortex-a53 -mtune=cortex-a53' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -mcpu=cortex-a53 -mtune=cortex-a53'
cmake --build "${output_dir}/whisper-build" --target whisper-server -j4
cp "${output_dir}/whisper-build/bin/whisper-server" \
    "${output_dir}/whisper-server-v1.8.0-linux-aarch64-a113x"

strip \
    "${output_dir}/threehub-audio-capture-v1.0.0-linux-aarch64-a113x" \
    "${output_dir}/qwen35-task-resident-v1.0.0-linux-aarch64-a113x" \
    "${output_dir}/whisper-server-v1.8.0-linux-aarch64-a113x"
sha256sum \
    "${output_dir}/threehub-audio-capture-v1.0.0-linux-aarch64-a113x" \
    "${output_dir}/qwen35-task-resident-v1.0.0-linux-aarch64-a113x" \
    "${output_dir}/whisper-server-v1.8.0-linux-aarch64-a113x"
