#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runner="$repository/build/minimax-h3-m3-e2e"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "MiniMax-H3 currently requires Apple Silicon and Metal." >&2
    exit 2
fi

if [ "$#" -gt 0 ]; then
    prompt=$*
elif [ -t 0 ]; then
    printf "Prompt: " >&2
    IFS= read -r prompt
else
    prompt=$(cat)
fi

if [ -z "$prompt" ]; then
    echo "Prompt cannot be empty." >&2
    exit 2
fi

width=${MINIMAX_H3_WIDTH:-128}
height=${MINIMAX_H3_HEIGHT:-128}
frames=${MINIMAX_H3_FRAMES:-22}
seed=${MINIMAX_H3_SEED:-42}
run_id=$(date +%Y%m%d-%H%M%S)
output_directory=${MINIMAX_H3_OUTPUT_DIR:-"$repository/tmp/minimax-h3-runs/$run_id"}

if [ ! -x "$runner" ] ||
   [ ! -f "$repository/build/minimax-h3-m3-attention.metallib" ] ||
   [ ! -f "$repository/build/minimax-h3-tokenizer.h3tok" ]; then
    echo "Building the MiniMax-H3 C/Metal runner..." >&2
    make -C "$repository" minimax-h3-m3-e2e
fi
if [ ! -f "$repository/build/minimax-h3-tokenizer.h3tok" ]; then
    echo "Local compiled tokenizer image is required: $repository/build/minimax-h3-tokenizer.h3tok" >&2
    echo "Pack it offline with build/minimax-h3-tokenizer-pack and the pinned tokenizer.json." >&2
    exit 3
fi

text_encoder=${MINIMAX_H3_TEXT_ENCODER:-"$repository/tmp/minimax-h3-text-encoder.safetensors"}
case "$text_encoder" in
    /*) ;;
    *) text_encoder="$repository/$text_encoder" ;;
esac
if [ ! -f "$text_encoder" ]; then
    echo "Local MiniMax-H3 text encoder is required: $text_encoder" >&2
    echo "Required size: 28,222,740,739 bytes. Inference never reads weights over the network." >&2
    echo "Copy or download it first, then set MINIMAX_H3_TEXT_ENCODER to its absolute path." >&2
    exit 3
fi
expected_text_encoder_bytes=28222740739
expected_text_encoder_sha256=0d1aebb6c57e6da06dd40423f65c1cdd95a914dd2e9b3a5626ac9f1083c5dc2b
actual_text_encoder_bytes=$(stat -f '%z' "$text_encoder")
if [ "$actual_text_encoder_bytes" -ne "$expected_text_encoder_bytes" ]; then
    echo "Incomplete MiniMax-H3 text encoder: $text_encoder" >&2
    echo "Expected $expected_text_encoder_bytes bytes; found $actual_text_encoder_bytes." >&2
    exit 3
fi
echo "Verifying local MiniMax-H3 text encoder SHA-256..." >&2
actual_text_encoder_sha256=$(shasum -a 256 "$text_encoder" | awk '{print $1}')
if [ "$actual_text_encoder_sha256" != "$expected_text_encoder_sha256" ]; then
    echo "MiniMax-H3 text encoder SHA-256 mismatch: $text_encoder" >&2
    echo "Expected $expected_text_encoder_sha256; found $actual_text_encoder_sha256." >&2
    exit 3
fi

if [ -n "${MINIMAX_H3_TURBO_ADAPTER:-}" ]; then
    case "$MINIMAX_H3_TURBO_ADAPTER" in
        file:///*) turbo_adapter=${MINIMAX_H3_TURBO_ADAPTER#file://} ;;
        *)
            echo "MINIMAX_H3_TURBO_ADAPTER must be an absolute file:// URL." >&2
            exit 3
            ;;
    esac
    if [ ! -f "$turbo_adapter" ]; then
        echo "Local MiniMax-H3 Turbo adapter is required: $turbo_adapter" >&2
        exit 3
    fi
    expected_turbo_bytes=779849816
    expected_turbo_sha256=5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
    actual_turbo_bytes=$(stat -f '%z' "$turbo_adapter")
    if [ "$actual_turbo_bytes" -ne "$expected_turbo_bytes" ]; then
        echo "Incomplete MiniMax-H3 Turbo adapter: $turbo_adapter" >&2
        echo "Expected $expected_turbo_bytes bytes; found $actual_turbo_bytes." >&2
        exit 3
    fi
    echo "Verifying local MiniMax-H3 Turbo adapter SHA-256..." >&2
    actual_turbo_sha256=$(shasum -a 256 "$turbo_adapter" | awk '{print $1}')
    if [ "$actual_turbo_sha256" != "$expected_turbo_sha256" ]; then
        echo "MiniMax-H3 Turbo adapter SHA-256 mismatch: $turbo_adapter" >&2
        echo "Expected $expected_turbo_sha256; found $actual_turbo_sha256." >&2
        exit 3
    fi
fi

revision=e1244ad93d60c737c7e0f065a1c9372f3de7caf8
cache_directory="$repository/tmp/minimax-h3-m3-cache"
missing_cache=0
for filename in transformer.safetensors adaln_cache.safetensors \
                video_vae.safetensors audio_vae.safetensors; do
    cache="$cache_directory/$filename.$revision.h3cache"
    if [ ! -f "$cache" ]; then
        echo "Local MiniMax-H3 runtime cache is required: $cache" >&2
        missing_cache=1
    fi
done
if [ "$missing_cache" -ne 0 ]; then
    echo "Inference is offline. Prepare the local caches before generating." >&2
    exit 3
fi

mkdir -p "$output_directory"
echo "Output: $output_directory/minimax-h3.mp4" >&2
echo "Geometry: ${width}x${height}, ${frames} frames at 24 fps; seed $seed" >&2

cd "$repository"
MINIMAX_H3_TEXT_ENCODER_URL="file://$text_encoder" \
    "$runner" "$prompt" "$output_directory" \
    "$width" "$height" "$frames" "$seed"
