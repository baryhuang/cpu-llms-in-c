#!/bin/sh

set -eu

RUNTIME_REPOSITORY="https://github.com/0xShug0/audio.cpp.git"
RUNTIME_COMMIT="26dcb5c4cf5aa016ae6285096a7b45f2671e5d17"

usage() {
    printf 'Usage: %s OUTPUT_DIR\n' "$0" >&2
    printf 'Set AUDIOCPP_SOURCE_DIR to reuse a pristine checkout at the pinned commit.\n' >&2
    exit 2
}

[ "$#" -eq 1 ] || usage

case "$(uname -s):$(uname -m)" in
    Linux:aarch64|Linux:arm64) ;;
    *)
        printf 'error: build on Linux AArch64 (a linux/arm64 container is supported)\n' >&2
        exit 1
        ;;
esac

for command_name in git cmake ninja cc c++ strip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'error: required build command is missing: %s\n' "$command_name" >&2
        exit 1
    fi
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
patch_file="${script_dir}/patches/0001-moss-nano-incremental-kv-cache.patch"
output_dir=$1

case "$output_dir" in
    /*) ;;
    *) output_dir="$(pwd)/${output_dir}" ;;
esac

if [ ! -r "$patch_file" ]; then
    printf 'error: runtime patch is missing: %s\n' "$patch_file" >&2
    exit 1
fi

temporary_root=
cleanup() {
    if [ -n "$temporary_root" ] && [ -d "$temporary_root" ]; then
        rm -rf -- "$temporary_root"
    fi
}
trap cleanup EXIT HUP INT TERM

if [ -n "${AUDIOCPP_SOURCE_DIR:-}" ]; then
    source_dir=$AUDIOCPP_SOURCE_DIR
    actual_commit=$(git -C "$source_dir" rev-parse HEAD)
    if [ "$actual_commit" != "$RUNTIME_COMMIT" ]; then
        printf 'error: AUDIOCPP_SOURCE_DIR is at %s, expected %s\n' \
            "$actual_commit" "$RUNTIME_COMMIT" >&2
        exit 1
    fi
    if [ -n "$(git -C "$source_dir" status --porcelain)" ]; then
        printf 'error: AUDIOCPP_SOURCE_DIR must be a pristine checkout\n' >&2
        exit 1
    fi
else
    temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/moss-a113x-build.XXXXXX")
    source_dir="${temporary_root}/audio.cpp"
    git clone --filter=blob:none --no-checkout "$RUNTIME_REPOSITORY" "$source_dir"
    git -C "$source_dir" fetch --depth 1 origin "$RUNTIME_COMMIT"
    git -C "$source_dir" checkout --detach "$RUNTIME_COMMIT"
fi

git -C "$source_dir" apply --check "$patch_file"
git -C "$source_dir" apply "$patch_file"

build_dir="${source_dir}/build-a113x-kvcache"
common_flags='-O3 -mcpu=cortex-a53+crypto+crc -mtune=cortex-a53 -fomit-frame-pointer -fno-semantic-interposition -fno-plt -ffunction-sections -fdata-sections'

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAUDIOCPP_MODEL_SET=custom \
    -DAUDIOCPP_MODELS=moss_tts_nano \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
    -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DENGINE_ENABLE_LLAMAFILE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8-a+crypto+crc \
    -DGGML_LTO=ON \
    -DCMAKE_C_FLAGS="$common_flags" \
    -DCMAKE_CXX_FLAGS="$common_flags" \
    -DCMAKE_EXE_LINKER_FLAGS='-flto=auto -Wl,--gc-sections -Wl,-O1'

cmake --build "$build_dir" --target audiocpp_cli audiocpp_server -j4

mkdir -p "$output_dir"
install -m 0755 "$build_dir/bin/audiocpp_cli" "$output_dir/audiocpp_cli"
install -m 0755 "$build_dir/bin/audiocpp_server" "$output_dir/audiocpp_server"
strip --strip-all "$output_dir/audiocpp_cli" "$output_dir/audiocpp_server"

printf 'Built native A113X artifacts in %s\n' "$output_dir"
