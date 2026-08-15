#!/bin/sh
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_directory/../.." && pwd)
venv_python=${COMPARE_PYTHON:-"$repository/tmp/compare-venv/bin/python"}
mlx_model=${QWEN38_MLX_MODEL:-"$repository/tmp/qwen38-27b-4bit"}
gguf_model=${QWEN38_GGUF_MODEL:-"$repository/tmp/qwen38-27b-unsloth/Qwen3.8-27B-Q4_K_M.gguf"}

cd "$repository"

run_llama() {
    output=$1
    llama-server -m "$gguf_model" -c 4096 -fa 1 --port 8760 \
        > "$script_directory/qwen38-llama-server.log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM

    attempt=0
    while [ "$attempt" -lt 200 ]; do
        if curl -s -m 1 http://127.0.0.1:8760/health 2>/dev/null \
            | grep -q ok; then
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    if [ "$attempt" -eq 200 ]; then
        echo "llama-server did not become healthy" >&2
        exit 3
    fi

    python3 "$script_directory/run_llamacpp_set.py" 8760 "$output"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    trap - EXIT INT TERM
}

echo "Qwen3.8 / mlx-lm"
[ -f "$script_directory/set-mlxlm-38.json" ] || \
    "$venv_python" "$script_directory/run_mlxlm_set.py" \
        "$mlx_model" "$script_directory/set-mlxlm-38.json"

echo "Qwen3.8 / oMLX"
[ -f "$script_directory/set-omlx-38.json" ] || \
    "$venv_python" "$script_directory/run_omlx_set.py" \
        "$mlx_model" "$script_directory/set-omlx-38.json"

echo "Qwen3.8 / llama.cpp"
[ -f "$script_directory/set-llama-38.json" ] || \
    run_llama "$script_directory/set-llama-38.json"

echo "complete"
