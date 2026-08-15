#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runner="$repository/build/qwen36-m3-generate"
chat="$repository/build/qwen36-m3-chat"
model_directory=${QWEN36_MODEL_DIR:-"$repository/tmp/qwen36-27b-runtime"}
metallib=${QWEN36_METALLIB:-"$repository/build/qwen36-m3-q4.metallib"}
tokenizer=${QWEN36_TOKENIZER:-"$model_directory/tokenizer.q36tok"}
context=${QWEN36_CONTEXT:-4096}
maximum_new=${QWEN36_MAX_TOKENS:-2048}
temperature=${QWEN36_TEMPERATURE:-0}
top_k=${QWEN36_TOP_K:-1}
seed=${QWEN36_SEED:-42}

if [ ! -x "$runner" ] || [ ! -x "$chat" ] || [ ! -f "$metallib" ]; then
    echo "Building the Qwen3.6 runner..." >&2
    make -C "$repository" qwen36-m3-generate qwen36-m3-chat
fi

if [ ! -f "$model_directory/global.q36global" ] ||
   [ ! -f "$model_directory/layer-00.q36delta" ] ||
   [ ! -f "$model_directory/layer-63.q36att" ] ||
   [ ! -f "$tokenizer" ]; then
    echo "Compiled Qwen3.6-27B image not found: $model_directory" >&2
    echo "See models/qwen3.6-27b/targets/apple-m3-pro/README.md" >&2
    exit 2
fi

run_prompt() {
    if [ "${QWEN36_RAW:-0}" = "1" ]; then
        "$runner" "$model_directory" "$metallib" "$tokenizer" \
            "$context" "$maximum_new" "$temperature" "$top_k" \
            "$seed" "$1"
    else
        QWEN36_STREAM_FD=3 \
            "$runner" "$model_directory" "$metallib" "$tokenizer" \
                "$context" "$maximum_new" "$temperature" "$top_k" \
                "$seed" "$1" 3>&1 >/dev/null
    fi
}

# One-shot mode (a prompt as arguments) and QWEN36_RAW keep the
# per-invocation generator with its JSON contract.
if [ "$#" -gt 0 ]; then
    run_prompt "$*"
    exit 0
fi

# Interactive mode runs the resident chat: the model loads and wires once
# at startup, then every prompt answers at the ready-state latency.
exec "$chat" "$model_directory" "$metallib" "$tokenizer" \
    "$context" "$maximum_new" "$temperature" "$top_k" "$seed"
