#!/bin/sh
set -eu

if [ "$#" -ne 5 ]; then
    echo "usage: $0 SHARD1 SHARD2 SHARD3 TOKENIZER_JSON OUTPUT_DIR" >&2
    exit 2
fi

shard1=$1
shard2=$2
shard3=$3
tokenizer_json=$4
output_dir=$5

sha1=2689680915661f040c50c35244d08b336def279e509e1ca11873f8dd1b0e7ce0
sha2=983ed9b84ef9469d099e5300d957e6fdecffe21b87096c0f52c9f272a97490c6
sha3=ac23bf70b1f239a040921d6f93770d74176fd435dbf44e42317053d06c68d702

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_dir/.." && pwd)

for source_file in "$shard1" "$shard2" "$shard3" "$tokenizer_json"; do
    if [ ! -f "$source_file" ]; then
        echo "missing source: $source_file" >&2
        exit 3
    fi
done

make -C "$repository" qwen36-tools
mkdir -p "$output_dir"

pack_if_missing() {
    output=$1
    shift
    if [ -e "$output" ]; then
        echo "resume: $output already exists" >&2
    else
        "$@"
    fi
}

pack_if_missing "$output_dir/global.q36global" \
    "$repository/build/qwen36-m3-global-pack" \
    "$shard1" "$shard3" "$output_dir/global.q36global" "$sha1" "$sha3"

pack_if_missing "$output_dir/tokenizer.q36tok" \
    "$repository/build/qwen36-tokenizer-pack" \
    "$tokenizer_json" "$output_dir/tokenizer.q36tok"

layer=0
while [ "$layer" -lt 64 ]; do
    if [ "$layer" -le 17 ]; then
        core_source=$shard1
        core_sha=$sha1
    elif [ "$layer" -le 42 ]; then
        core_source=$shard2
        core_sha=$sha2
    else
        core_source=$shard3
        core_sha=$sha3
    fi
    padded=$(printf '%02d' "$layer")
    remainder=$((layer % 4))
    if [ "$remainder" -eq 3 ]; then
        output="$output_dir/layer-$padded.q36att"
        pack_if_missing "$output" \
            "$repository/build/qwen36-m3-attention-pack" \
            "$core_source" "$output" "$core_sha" "$layer"
    else
        output="$output_dir/layer-$padded.q36delta"
        if [ "$layer" -eq 17 ]; then
            pack_if_missing "$output" \
                "$repository/build/qwen36-m3-pack" \
                "$core_source" "$output" "$core_sha" "$layer" \
                "$shard2" "$sha2"
        elif [ "$layer" -eq 42 ]; then
            pack_if_missing "$output" \
                "$repository/build/qwen36-m3-pack" \
                "$core_source" "$output" "$core_sha" "$layer" \
                "$shard3" "$sha3"
        else
            pack_if_missing "$output" \
                "$repository/build/qwen36-m3-pack" \
                "$core_source" "$output" "$core_sha" "$layer"
        fi
    fi
    layer=$((layer + 1))
done

echo "compiled Qwen3.6-27B text image: $output_dir"
