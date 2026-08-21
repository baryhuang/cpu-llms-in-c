#!/bin/sh
set -eu

model_directory=${1:-/root/qwen38-jetson/models}
model_name=Qwen3.8-27B-UD-IQ1_S.gguf
model_sha256=3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887
model_url=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ1_S.gguf

mkdir -p "$model_directory"
cd "$model_directory"
curl -fL --retry 5 --retry-delay 3 --continue-at - \
    --output "$model_name.part" "$model_url"
printf '%s  %s\n' "$model_sha256" "$model_name.part" | sha256sum -c -
mv "$model_name.part" "$model_name"
