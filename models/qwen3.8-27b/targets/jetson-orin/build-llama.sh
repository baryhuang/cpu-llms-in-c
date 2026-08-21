#!/bin/sh
set -eu

llama_source=${LLAMA_CPP_SOURCE:-/root/qwen38-jetson/src/llama.cpp}
build_directory=${LLAMA_CPP_BUILD:-"$llama_source/build-orin-lean"}
expected_commit=749f688fcaa4c472ec034b08cb8a907c45cfaa02

actual_commit=$(git -C "$llama_source" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
    echo "llama.cpp revision mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 1
fi

CUDACXX=/usr/local/cuda/bin/nvcc cmake -S "$llama_source" -B "$build_directory" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_directory" -j6 --target \
    llama-cli llama-server llama-bench
