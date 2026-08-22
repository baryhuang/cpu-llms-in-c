#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_node="$repository/tools/comfyui/minimax_h3_native"
comfy_root=${COMFYUI_ROOT:-"$HOME/ComfyUI-Installs/Bary's Host/ComfyUI"}
custom_nodes="$comfy_root/custom_nodes"
destination="$custom_nodes/minimax_h3_native"

if [ ! -d "$custom_nodes" ]; then
    echo "ComfyUI custom_nodes directory not found: $custom_nodes" >&2
    echo "Set COMFYUI_ROOT to the active ComfyUI instance." >&2
    exit 2
fi

if [ -e "$destination" ] && [ ! -L "$destination" ]; then
    echo "Refusing to replace an existing non-symlink: $destination" >&2
    exit 3
fi

ln -sfn "$source_node" "$destination"
echo "Installed MiniMax-H3 native node: $destination -> $source_node"
echo "Restart the ComfyUI server to load it."
