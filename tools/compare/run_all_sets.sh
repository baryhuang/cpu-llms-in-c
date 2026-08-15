#!/bin/sh
set -u
REPO=/Users/buryhuang/git/cpullama
SCRATCH=/Users/buryhuang/git/cpullama/tools/compare
VENVPY=$REPO/tmp/compare-venv/bin/python
cd "$REPO"

run_llama() {  # gguf outfile
  llama-server -m "$1" -c 4096 -fa 1 --port 8760 > "$SCRATCH/llsrv.log" 2>&1 &
  SRV=$!
  i=0; while [ $i -lt 200 ]; do curl -s -m 1 http://127.0.0.1:8760/health 2>/dev/null | grep -q ok && break; sleep 2; i=$((i+1)); done
  python3 "$SCRATCH/run_llamacpp_set.py" 8760 "$2"
  kill $SRV 2>/dev/null; sleep 3
}

echo "=A= llama.cpp 3.6"
[ -f "$SCRATCH/set-llama-36.json" ] || run_llama tmp/qwen36-27b-unsloth/Qwen3.6-27B-Q4_K_M.gguf "$SCRATCH/set-llama-36.json"
echo "=B= free 3.6 gguf, download 3.6 mlx checkpoint"
if [ ! -f tmp/qwen36-27b-mlx/download.done ]; then
  rm -f tmp/qwen36-27b-unsloth/Qwen3.6-27B-Q4_K_M.gguf
  sh "$SCRATCH/dl_q36_mlx.sh"
fi
echo "=C= mlx-lm 3.6"
[ -f "$SCRATCH/set-mlxlm-36.json" ] || "$VENVPY" "$SCRATCH/run_mlxlm_set.py" tmp/qwen36-27b-mlx "$SCRATCH/set-mlxlm-36.json"
echo "=D= oMLX 3.6"
[ -f "$SCRATCH/set-omlx-36.json" ] || "$VENVPY" "$SCRATCH/run_omlx_set.py" tmp/qwen36-27b-mlx "$SCRATCH/set-omlx-36.json"
echo "=E= mlx-lm 3.8"
[ -f "$SCRATCH/set-mlxlm-38.json" ] || "$VENVPY" "$SCRATCH/run_mlxlm_set.py" tmp/qwen38-27b-4bit "$SCRATCH/set-mlxlm-38.json"
echo "=F= oMLX 3.8"
[ -f "$SCRATCH/set-omlx-38.json" ] || "$VENVPY" "$SCRATCH/run_omlx_set.py" tmp/qwen38-27b-4bit "$SCRATCH/set-omlx-38.json"
echo "=G= llama.cpp 3.8"
[ -f "$SCRATCH/set-llama-38.json" ] || run_llama tmp/qwen38-27b-unsloth/Qwen3.8-27B-Q4_K_M.gguf "$SCRATCH/set-llama-38.json"
echo done > "$SCRATCH/run_all_sets.done"
