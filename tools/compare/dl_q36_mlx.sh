#!/bin/sh
# Parallel-ranged download of mlx-community/Qwen3.6-27B-4bit at the
# pinned revision into tmp/qwen36-27b-mlx (clean dir; the old
# tmp/qwen36-27b-4bit holds bring-up artifacts and no full shards).
set -u
REV=c000ac2c2057d94be3fa931000c31723aac53282
BASE="https://huggingface.co/mlx-community/Qwen3.6-27B-4bit/resolve/$REV"
DIR=/Users/buryhuang/git/cpullama/tmp/qwen36-27b-mlx
mkdir -p "$DIR"

for small in model.safetensors.index.json config.json tokenizer.json \
             tokenizer_config.json generation_config.json \
             chat_template.jinja; do
  [ -s "$DIR/$small" ] || curl -sSL -o "$DIR/$small" "$BASE/$small"
done

dl_shard() {  # name size
  NAME=$1; TOTAL=$2; PARTS=12
  CHUNK=$(( (TOTAL + PARTS - 1) / PARTS ))
  i=0
  while [ $i -lt $PARTS ]; do
    START=$(( i * CHUNK )); END=$(( START + CHUNK - 1 ))
    [ $END -ge $TOTAL ] && END=$(( TOTAL - 1 ))
    PART="$DIR/$NAME.part$(printf %02d $i)"
    EXPECT=$(( END - START + 1 ))
    HAVE=$(stat -f%z "$PART" 2>/dev/null || echo 0)
    if [ "$HAVE" -lt "$EXPECT" ]; then
      RESUME=$(( START + HAVE ))
      curl -sSL -r "$RESUME-$END" "$BASE/$NAME" >> "$PART" &
    fi
    i=$(( i + 1 ))
  done
  wait
  cat "$DIR/$NAME".part?? > "$DIR/$NAME"
  SIZE=$(stat -f%z "$DIR/$NAME")
  if [ "$SIZE" -eq "$TOTAL" ]; then
    rm -f "$DIR/$NAME".part??
    return 0
  fi
  return 1
}

attempt=0
while [ $attempt -lt 100 ]; do
  ok=1
  [ "$(stat -f%z "$DIR/model-00001-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5343268752 ] || dl_shard model-00001-of-00003.safetensors 5343268752 || ok=0
  [ "$(stat -f%z "$DIR/model-00002-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5354185100 ] || dl_shard model-00002-of-00003.safetensors 5354185100 || ok=0
  [ "$(stat -f%z "$DIR/model-00003-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5357087747 ] || dl_shard model-00003-of-00003.safetensors 5357087747 || ok=0
  if [ "$ok" -eq 1 ] && \
     [ "$(stat -f%z "$DIR/model-00001-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5343268752 ] && \
     [ "$(stat -f%z "$DIR/model-00002-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5354185100 ] && \
     [ "$(stat -f%z "$DIR/model-00003-of-00003.safetensors" 2>/dev/null || echo 0)" -eq 5357087747 ]; then
    echo done > "$DIR/download.done"
    exit 0
  fi
  attempt=$(( attempt + 1 ))
  sleep 5
done
echo "gave up" > "$DIR/download.err"
