#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runner="$repository/build/qwen36-m3-generate"
chat="$repository/build/qwen36-m3-chat"
model_directory=${QWEN36_MODEL_DIR:-"$repository/tmp/qwen36-27b-runtime"}
metallib=${QWEN36_METALLIB:-"$repository/build/qwen36-m3-q4.metallib"}
tokenizer=${QWEN36_TOKENIZER:-"$model_directory/tokenizer.q36tok"}
context=${QWEN36_CONTEXT:-4096}
maximum_new=${QWEN36_MAX_TOKENS:-3072}
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

# --terminal keeps the resident terminal chat. One-shot mode (a prompt as
# arguments) and QWEN36_RAW keep the per-invocation generator with its
# JSON contract.
if [ "$#" -gt 0 ] && [ "$1" = "--terminal" ]; then
    exec "$chat" "$model_directory" "$metallib" "$tokenizer" \
        "$context" "$maximum_new" "$temperature" "$top_k" "$seed"
fi
if [ "$#" -gt 0 ]; then
    run_prompt "$*"
    exit 0
fi

# Default: the one-command app experience. Start the OpenAI-compatible
# server if it is not already running, install the Chatbox client if it
# is missing, then open the client.
port=${QWEN36_PORT:-8199}
base_url="http://127.0.0.1:$port/v1"
log="$repository/tmp/qwen36-serve.log"

if ! curl -sf --max-time 2 "http://127.0.0.1:$port/health" \
        > /dev/null 2>&1; then
    echo "Starting the model server (one-time weight wiring, ~10 s)..." >&2
    nohup python3 "$repository/tools/qwen36_serve.py" --port "$port" \
        >> "$log" 2>&1 &
    waited=0
    until curl -sf --max-time 2 "http://127.0.0.1:$port/health" \
            > /dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 120 ]; then
            echo "Server did not become ready; see $log" >&2
            exit 3
        fi
    done
    echo "Server ready at $base_url" >&2
else
    echo "Server already running at $base_url" >&2
fi

if [ ! -d "/Applications/Chatbox.app" ]; then
    if ! command -v brew > /dev/null 2>&1; then
        echo "Homebrew not found; install the Chatbox client manually" \
             "from https://chatboxai.app and connect it to $base_url" >&2
        exit 4
    fi
    echo "Installing the Chatbox client (one time)..." >&2
    brew install --cask chatbox >&2
fi

if python3 "$repository/tools/qwen36_chatbox_config.py" "$base_url" >&2
then
    echo "Chatbox is opening, already connected to the local model." >&2
else
    printf '%s' "$base_url" | pbcopy 2>/dev/null || true
    cat >&2 <<SETTINGS
Automatic configuration failed; add a provider in Chatbox settings:
  Provider:  OpenAI API Compatible
  API Host:  $base_url   (already copied to your clipboard)
  API Key:   anything, e.g. local
  Model:     qwen3.6-27b
SETTINGS
fi
echo "Server log: $log. Stop it with: pkill -f qwen36_serve.py" >&2
open "/Applications/Chatbox.app"
# A second open after startup reliably surfaces the window when the app
# relaunched without one (Electron reopen behavior).
sleep 2
exec open "/Applications/Chatbox.app"
