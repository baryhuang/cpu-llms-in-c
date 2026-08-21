#!/bin/sh
set -eu

threshold_kib=${QWEN38_SWAP_RESTART_KIB:-1048576}
server_pid=$(systemctl show --property MainPID --value qwen38.service)
if [ -z "$server_pid" ] || [ "$server_pid" -le 0 ] || \
   [ ! -r "/proc/$server_pid/status" ]; then
    exit 0
fi

server_swap_kib=$(awk '/^VmSwap:/ { print $2 }' "/proc/$server_pid/status")
server_swap_kib=${server_swap_kib:-0}
if [ "$server_swap_kib" -lt "$threshold_kib" ]; then
    exit 0
fi

slots=$(curl --fail --silent --show-error --max-time 3 \
    http://127.0.0.1:8199/slots) || exit 0
idle=$(printf '%s' "$slots" | python3 -c \
    'import json,sys; print("yes" if all(not x.get("is_processing", False) for x in json.load(sys.stdin)) else "no")')
if [ "$idle" != yes ]; then
    exit 0
fi

logger -t qwen38-memory-guard \
    "restarting idle qwen38.service at ${server_swap_kib} KiB process swap"
systemctl restart qwen38.service
