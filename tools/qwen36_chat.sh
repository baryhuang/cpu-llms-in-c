#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runner="$repository/build/qwen36-m3-generate"
model_directory=${QWEN36_MODEL_DIR:-"$repository/tmp/qwen36-27b-runtime"}
metallib=${QWEN36_METALLIB:-"$repository/build/qwen36-m3-q4.metallib"}
tokenizer=${QWEN36_TOKENIZER:-"$model_directory/tokenizer.q36tok"}
context=${QWEN36_CONTEXT:-512}
maximum_new=${QWEN36_MAX_TOKENS:-128}
temperature=${QWEN36_TEMPERATURE:-0}
top_k=${QWEN36_TOP_K:-1}
seed=${QWEN36_SEED:-42}

if [ ! -x "$runner" ] || [ ! -f "$metallib" ]; then
    echo "Building the Qwen3.6 runner..." >&2
    make -C "$repository" qwen36-m3-generate
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

if [ "$#" -gt 0 ]; then
    run_prompt "$*"
    exit 0
fi

echo "Qwen3.6-27B on Apple M3 Pro"
echo "Enter /quit to exit."

while :; do
    printf '\nYou> '
    if ! IFS= read -r prompt; then
        printf '\n'
        break
    fi
    case "$prompt" in
        /quit|/exit) break ;;
        '') continue ;;
    esac
    printf '\nModel>\n'
    run_prompt "$prompt"
done
