#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runner="$repository/build/qwen38-m3-generate"
chat="$repository/build/qwen38-m3-chat"
model_directory=${QWEN38_MODEL_DIR:-"$repository/tmp/qwen38-27b-runtime"}
metallib=${QWEN38_METALLIB:-"$repository/build/qwen38-m3-q4.metallib"}
tokenizer=${QWEN38_TOKENIZER:-"$model_directory/tokenizer.q38tok"}
context=${QWEN38_CONTEXT:-4096}
maximum_new=${QWEN38_MAX_TOKENS:-3072}
temperature=${QWEN38_TEMPERATURE:-0}
top_k=${QWEN38_TOP_K:-1}
seed=${QWEN38_SEED:-42}

if [ ! -x "$runner" ] || [ ! -x "$chat" ] || [ ! -f "$metallib" ]; then
    echo "Building the Qwen3.8 runner..." >&2
    make -C "$repository" qwen38-m3-generate qwen38-m3-chat
fi

if [ ! -f "$model_directory/global.q38global" ] ||
   [ ! -f "$model_directory/layer-00.q38delta" ] ||
   [ ! -f "$model_directory/layer-63.q38att" ] ||
   [ ! -f "$tokenizer" ]; then
    echo "Compiled Qwen3.8-27B image not found: $model_directory" >&2
    echo "See models/qwen3.8-27b/targets/apple-m3-pro/README.md" >&2
    exit 2
fi

run_prompt() {
    if [ "${QWEN38_RAW:-0}" = "1" ]; then
        "$runner" "$model_directory" "$metallib" "$tokenizer" \
            "$context" "$maximum_new" "$temperature" "$top_k" \
            "$seed" "$1"
    else
        QWEN38_STREAM_FD=3 \
            "$runner" "$model_directory" "$metallib" "$tokenizer" \
                "$context" "$maximum_new" "$temperature" "$top_k" \
                "$seed" "$1" 3>&1 >/dev/null
    fi
}

# --terminal keeps the resident terminal chat. One-shot mode (a prompt as
# arguments) and QWEN38_RAW keep the per-invocation generator with its
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
port=${QWEN38_PORT:-8199}
base_url="http://127.0.0.1:$port/v1"
log="$repository/tmp/qwen38-serve.log"

# Exactly one server, one engine and one client: stop whatever is
# already running before starting fresh.
if pgrep -f qwen38_serve.py > /dev/null 2>&1 ||
   pgrep -x qwen38-m3-chat > /dev/null 2>&1; then
    echo "Stopping the existing server..." >&2
    pkill -f qwen38_serve.py 2>/dev/null || true
    pkill -x qwen38-m3-chat 2>/dev/null || true
    waited=0
    while pgrep -f qwen38_serve.py > /dev/null 2>&1 ||
          pgrep -x qwen38-m3-chat > /dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 15 ]; then
            pkill -9 -f qwen38_serve.py 2>/dev/null || true
            pkill -9 -x qwen38-m3-chat 2>/dev/null || true
            sleep 1
            break
        fi
    done
fi
if pgrep -x Chatbox > /dev/null 2>&1; then
    echo "Restarting the running Chatbox..." >&2
    osascript -e 'quit app "Chatbox"' > /dev/null 2>&1 ||
        pkill -x Chatbox 2>/dev/null || true
    waited=0
    while pgrep -x Chatbox > /dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 10 ]; then
            pkill -9 -x Chatbox 2>/dev/null || true
            sleep 1
            break
        fi
    done
fi

echo "Starting the model server (one-time weight wiring, ~10 s)..." >&2
nohup python3 "$repository/tools/qwen38_serve.py" --port "$port" \
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

if [ ! -d "/Applications/Chatbox.app" ]; then
    if ! command -v brew > /dev/null 2>&1; then
        echo "Homebrew not found; install the Chatbox client manually" \
             "from https://chatboxai.app and connect it to $base_url" >&2
        exit 4
    fi
    echo "Installing the Chatbox client (one time)..." >&2
    brew install --cask chatbox >&2
fi

if python3 "$repository/tools/qwen38_chatbox_config.py" "$base_url" >&2
then
    echo "Chatbox is opening, already connected to the local model." >&2
else
    printf '%s' "$base_url" | pbcopy 2>/dev/null || true
    cat >&2 <<SETTINGS
Automatic configuration failed; add a provider in Chatbox settings:
  Provider:  OpenAI API Compatible
  API Host:  $base_url   (already copied to your clipboard)
  API Key:   anything, e.g. local
  Model:     qwen3.8-27b
SETTINGS
fi
echo "Server log: $log. Stop it with: pkill -f qwen38_serve.py" >&2
open "/Applications/Chatbox.app"
# A second open after startup reliably surfaces the window when the app
# relaunched without one (Electron reopen behavior).
sleep 2
exec open "/Applications/Chatbox.app"
