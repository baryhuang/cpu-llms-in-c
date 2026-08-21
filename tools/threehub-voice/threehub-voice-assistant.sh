#!/bin/sh

# Resident, no-Python voice pipeline for ThirdReality TRHub-V3 / A113X.
# The ALSA capture process never closes the microphone. Whisper, Qwen, and
# MOSS keep their model processes resident and warmed between turns.

set -eu

VOICE_RELEASE="threehub-resident-voice-a113x-v1.0.0"
VOICE_BASE_URL="https://github.com/baryhuang/llm-in-c/releases/download/${VOICE_RELEASE}"
CAPTURE_ASSET="threehub-audio-capture-v1.0.0-linux-aarch64-a113x"
WHISPER_SERVER_ASSET="whisper-server-v1.8.0-linux-aarch64-a113x"
QWEN_BINARY_ASSET="qwen35-task-resident-v1.0.0-linux-aarch64-a113x"
QWEN_IMAGE_ASSET="qwen35-a113x-v3.qtask"
CAPTURE_SHA256="c5a424380917fa14ffc07e573963a307911d6b57d390583ce8cbf3e673c9f1b4"
WHISPER_SERVER_SHA256="25f25ad837f108642505c15f21c37a4496466572e98033789d34b600057c0a5b"
QWEN_BINARY_SHA256="4cbfb51b04f5f0c515638a6a6cf5c19894f9e688fd78f8e7dd1eca6f8b97859b"
QWEN_IMAGE_SHA256="793f666415da402d459dfaaa7060c10021346ab3d1d4cc91c511f40848093438"

WHISPER_RELEASE="whisper-base-q5_1-vad-a113x-v1.8.0"
WHISPER_BASE_URL="https://github.com/baryhuang/llm-in-c/releases/download/${WHISPER_RELEASE}"
WHISPER_MODEL_ASSET="ggml-base-q5_1.bin"
WHISPER_VAD_ASSET="ggml-silero-v5.1.2.bin"
WHISPER_MODEL_SHA256="422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898"
WHISPER_VAD_SHA256="29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf"

MOSS_RELEASE="moss-tts-nano-q8-a113x-kvcache-r2"
MOSS_BASE_URL="https://github.com/baryhuang/llm-in-c/releases/download/${MOSS_RELEASE}"
MOSS_LAUNCHER_ASSET="run-moss-server-a113x.sh"
MOSS_LAUNCHER_SHA256="deb6a06012b219b9d118e0fc967202d000af572d29b1b7910797e8d4cfd6e1be"

cache_dir=${THREEHUB_VOICE_CACHE_DIR:-/dev/shm/threehub-resident-voice-v1}
state_dir=${VOICE_ASSISTANT_STATE_DIR:-/dev/shm/threehub-voice}
install_dir=${THREEHUB_VOICE_INSTALL_DIR:-/root/threehub-voice}
voice_reference=${MOSS_VOICE_REF:-${install_dir}/voice_ref.wav}
moss_cache=${MOSS_NANO_CACHE_DIR:-/dev/shm/moss-tts-nano-q8-a113x-v0.6.1}
qwen_image=${QWEN35_IMAGE:-/root/llm-in-c-a113x/deploy/qwen35-a113x-v3.qtask}
max_tokens=${VOICE_ASSISTANT_MAX_TOKENS:-32}
moss_max_tokens=${MOSS_MAX_TOKENS:-40}
whisper_port=${WHISPER_SERVER_PORT:-8178}
pid_file=${VOICE_ASSISTANT_PID_FILE:-${install_dir}/voice-assistant-live.pid}
capture_status=${state_dir}/capture-status.json
pipeline_status=${state_dir}/pipeline-status.json
playback_mute=${state_dir}/playback.active
spool_dir=${state_dir}/spool-$$
qwen_request_fifo=${state_dir}/qwen-request-$$.fifo
qwen_response_fifo=${state_dir}/qwen-response-$$.fifo

capture_pid=
whisper_pid=
qwen_pid=

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download() {
    url=$1
    output=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 --output "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
    else
        printf 'error: curl or wget is required\n' >&2
        exit 1
    fi
}

ensure_asset() {
    base_url=$1
    asset=$2
    destination=$3
    expected_sha256=$4
    stamp="${destination}.sha256-ok"
    mkdir -p "$(dirname "$destination")"

    if [ -f "$destination" ] && [ -f "$stamp" ] &&
       [ "$(sed -n '1p' "$stamp")" = "$expected_sha256" ]; then
        return
    fi
    if [ -f "$destination" ] && [ "$(sha256_file "$destination")" = "$expected_sha256" ]; then
        printf '%s\n' "$expected_sha256" > "$stamp"
        return
    fi

    temporary="${destination}.part.$$"
    rm -f "$temporary"
    printf 'Downloading %s...\n' "$asset" >&2
    download "${base_url}/${asset}" "$temporary"
    if [ "$(sha256_file "$temporary")" != "$expected_sha256" ]; then
        rm -f "$temporary"
        printf 'error: SHA-256 mismatch for %s\n' "$asset" >&2
        exit 1
    fi
    mv "$temporary" "$destination"
    printf '%s\n' "$expected_sha256" > "$stamp"
}

detect_microphone() {
    if [ -n "${VOICE_MIC_DEVICE:-}" ]; then
        printf '%s\n' "$VOICE_MIC_DEVICE"
        return
    fi
    arecord -l 2>/dev/null |
        sed -n 's/^card \([0-9][0-9]*\):.*device \([0-9][0-9]*\):.*/plughw:\1,\2/p' |
        sed -n '1p'
}

detect_speaker() {
    if [ -n "${VOICE_SPEAKER_DEVICE:-}" ]; then
        printf '%s\n' "$VOICE_SPEAKER_DEVICE"
        return
    fi
    aplay -l 2>/dev/null |
        sed -n 's/^card \([0-9][0-9]*\):.*device \([0-9][0-9]*\):.*/plughw:\1,\2/p' |
        sed -n '1p'
}

normalize_lines() {
    awk 'NF {
        line = $0
        sub(/^[[:space:]]+/, "", line)
        sub(/[[:space:]]+$/, "", line)
        if (line != "") {
            if (text != "") text = text " "
            text = text line
        }
    } END { print text }'
}

write_pipeline_status() {
    stage=$1
    detail=${2:-}
    temporary="${pipeline_status}.tmp.$$"
    jq -n --arg stage "$stage" --arg detail "$detail" \
        --argjson timestamp "$(date +%s)" \
        '{stage:$stage,detail:$detail,timestamp:$timestamp}' > "$temporary"
    mv "$temporary" "$pipeline_status"
}

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    touch "$playback_mute"
    for child in "$capture_pid" "$qwen_pid" "$whisper_pid"; do
        [ -n "$child" ] || continue
        kill "$child" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    rm -f "$qwen_request_fifo" "$qwen_response_fifo" "$playback_mute"
    if [ -f "$pid_file" ] && [ "$(sed -n '1p' "$pid_file")" = "$$" ]; then
        rm -f "$pid_file"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM HUP

for dependency in aplay arecord awk curl jq sed sox; do
    command -v "$dependency" >/dev/null 2>&1 || {
        printf 'error: required command is missing: %s\n' "$dependency" >&2
        exit 1
    }
done

case "$max_tokens:$moss_max_tokens" in
    *[!0-9:]*|:*|*:)
        printf 'error: token limits must be positive integers\n' >&2
        exit 1
        ;;
esac

mkdir -p "$cache_dir" "$state_dir" "$install_dir" "$spool_dir" "$moss_cache"
printf '%s\n' "$$" > "$pid_file"
rm -f "$playback_mute"

capture_binary=${cache_dir}/${CAPTURE_ASSET}
whisper_server=${cache_dir}/${WHISPER_SERVER_ASSET}
qwen_binary=${cache_dir}/${QWEN_BINARY_ASSET}
whisper_model=${cache_dir}/${WHISPER_MODEL_ASSET}
whisper_vad=${cache_dir}/${WHISPER_VAD_ASSET}
moss_server_launcher=${moss_cache}/${MOSS_LAUNCHER_ASSET}

write_pipeline_status starting "verifying release artifacts"
ensure_asset "$VOICE_BASE_URL" "$CAPTURE_ASSET" "$capture_binary" "$CAPTURE_SHA256"
ensure_asset "$VOICE_BASE_URL" "$WHISPER_SERVER_ASSET" "$whisper_server" \
    "$WHISPER_SERVER_SHA256"
ensure_asset "$VOICE_BASE_URL" "$QWEN_BINARY_ASSET" "$qwen_binary" "$QWEN_BINARY_SHA256"
ensure_asset "$VOICE_BASE_URL" "$QWEN_IMAGE_ASSET" "$qwen_image" "$QWEN_IMAGE_SHA256"
ensure_asset "$WHISPER_BASE_URL" "$WHISPER_MODEL_ASSET" "$whisper_model" \
    "$WHISPER_MODEL_SHA256"
ensure_asset "$WHISPER_BASE_URL" "$WHISPER_VAD_ASSET" "$whisper_vad" \
    "$WHISPER_VAD_SHA256"
ensure_asset "$MOSS_BASE_URL" "$MOSS_LAUNCHER_ASSET" "$moss_server_launcher" \
    "$MOSS_LAUNCHER_SHA256"
chmod 0755 "$capture_binary" "$whisper_server" "$qwen_binary" "$moss_server_launcher"

if [ ! -r "$voice_reference" ]; then
    printf 'error: MOSS voice reference is unavailable: %s\n' "$voice_reference" >&2
    exit 1
fi

microphone=$(detect_microphone)
speaker=$(detect_speaker)
if [ -z "$microphone" ] || [ -z "$speaker" ]; then
    printf 'error: microphone or speaker is unavailable\n' >&2
    exit 1
fi

write_pipeline_status starting "continuous microphone capture"
arecord -q -D "$microphone" -t raw -f S16_LE -r 16000 -c 1 |
    "$capture_binary" \
        --spool "$spool_dir" \
        --status "$capture_status" \
        --mute-file "$playback_mute" \
        --start-percent "${VOICE_START_PERCENT:-1.2}" \
        --stop-percent "${VOICE_STOP_PERCENT:-0.7}" \
        --start-ms "${VOICE_START_MS:-100}" \
        --stop-ms "${VOICE_STOP_MS:-1200}" \
        --pre-roll-ms "${VOICE_PRE_ROLL_MS:-300}" \
        --max-seconds "${VOICE_MAX_UTTERANCE_SECONDS:-35}" \
        --max-queued "${VOICE_MAX_QUEUED_UTTERANCES:-8}" \
        >> "${state_dir}/capture.log" 2>&1 &
capture_pid=$!

write_pipeline_status starting "resident Whisper Base"
env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
    "$whisper_server" \
        --model "$whisper_model" \
        --host 127.0.0.1 \
        --port "$whisper_port" \
        --threads 4 \
        --processors 1 \
        --language auto \
        --no-gpu \
        --no-timestamps \
        --beam-size 1 \
        --best-of 1 \
        --no-context \
        --vad \
        --vad-model "$whisper_vad" \
        > "${state_dir}/whisper-server.log" 2>&1 &
whisper_pid=$!

health_attempt=0
until curl -fsS "http://127.0.0.1:${whisper_port}/health" >/dev/null 2>&1; do
    health_attempt=$((health_attempt + 1))
    if ! kill -0 "$whisper_pid" 2>/dev/null || [ "$health_attempt" -ge 180 ]; then
        printf 'error: resident Whisper server did not become healthy\n' >&2
        exit 1
    fi
    sleep 1
done

write_pipeline_status starting "resident Qwen 0.8B"
rm -f "$qwen_request_fifo" "$qwen_response_fifo"
mkfifo "$qwen_request_fifo" "$qwen_response_fifo"
env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
    "$qwen_binary" "$qwen_image" --serve "$max_tokens" \
    < "$qwen_request_fifo" > "$qwen_response_fifo" \
    2> "${state_dir}/qwen-server.log" &
qwen_pid=$!
exec 8> "$qwen_request_fifo"
exec 9< "$qwen_response_fifo"

# Touch every Qwen execution path before declaring the pipeline ready.
printf '%s\n' '<|im_start|>system You are a concise voice assistant.<|im_end|> <|im_start|>user Reply only: Ready.<|im_end|> <|im_start|>assistant' >&8
IFS= read -r qwen_warm_json <&9
if ! printf '%s\n' "$qwen_warm_json" |
    jq -e '.mode == "greedy_generate" and .generated_tokens > 0' >/dev/null; then
    printf 'error: resident Qwen warm-up failed\n' >&2
    exit 1
fi

write_pipeline_status starting "resident MOSS voice clone"
env MOSS_NANO_CACHE_DIR="$moss_cache" MOSS_VOICE_REF="$voice_reference" \
    MOSS_MAX_TOKENS="$moss_max_tokens" "$moss_server_launcher" start

# Warm Whisper with actual reference speech, then discard the transcript.
warm_wav=${state_dir}/whisper-warm-$$.wav
sox "$voice_reference" -r 16000 -c 1 -b 16 -e signed-integer "$warm_wav" trim 0 1
curl -fsS "http://127.0.0.1:${whisper_port}/inference" \
    -F "file=@${warm_wav}" -F response_format=json -F temperature=0 \
    > "${state_dir}/whisper-warm.json"
rm -f "$warm_wav"

printf 'Resident pipeline ready: capture=%s whisper=%s qwen=%s moss=localhost speaker=%s\n' \
    "$capture_pid" "$whisper_pid" "$qwen_pid" "$speaker"
write_pipeline_status listening "microphone held continuously"

turn_number=0
while :; do
    next_audio=
    while [ -z "$next_audio" ]; do
        set -- "$spool_dir"/utterance-*.wav
        if [ -f "$1" ]; then
            next_audio=$1
            break
        fi
        for child in "$capture_pid" "$whisper_pid" "$qwen_pid"; do
            if ! kill -0 "$child" 2>/dev/null; then
                printf 'error: resident child exited: %s\n' "$child" >&2
                exit 1
            fi
        done
        sleep 0.1
    done

    turn_number=$((turn_number + 1))
    turn_dir="${state_dir}/turn-$(date +%Y%m%d-%H%M%S)-$$-${turn_number}"
    mkdir -p "$turn_dir"
    mv "$next_audio" "$turn_dir/input.wav"

    turn_start_ns=$(date +%s%N)
    write_pipeline_status whisper "turn ${turn_number}"
    curl -fsS "http://127.0.0.1:${whisper_port}/inference" \
        -F "file=@${turn_dir}/input.wav" \
        -F response_format=json \
        -F temperature=0 \
        > "${turn_dir}/whisper.json"
    transcript=$(jq -r '.text // empty' "${turn_dir}/whisper.json" | normalize_lines)

    if [ -z "$transcript" ]; then
        printf 'No speech recognized.\n' >&2
        write_pipeline_status listening "Silero VAD rejected empty audio"
        continue
    fi
    printf 'You: %s\n' "$transcript"

    write_pipeline_status qwen "$transcript"
    prompt="<|im_start|>system You are a concise voice assistant running on an embedded device. Reply in the language used by the user. Answer directly in at most two short sentences. Do not include analysis, thinking, or XML tags.<|im_end|> <|im_start|>user ${transcript}<|im_end|> <|im_start|>assistant"
    printf '%s\n' "$prompt" >&8
    IFS= read -r qwen_json <&9
    printf '%s\n' "$qwen_json" > "${turn_dir}/qwen.json"
    answer=$(printf '%s\n' "$qwen_json" | jq -r '.text // empty' |
        sed 's/<think>[^<]*<\/think>//g; /<think>/,/<\/think>/d; s/<|im_end|>//g' |
        normalize_lines)
    printf 'Qwen: %s\n' "$answer"

    if [ -n "$answer" ]; then
        write_pipeline_status moss "$answer"
        env MOSS_NANO_CACHE_DIR="$moss_cache" MOSS_VOICE_REF="$voice_reference" \
            MOSS_MAX_TOKENS="$moss_max_tokens" MOSS_DO_SAMPLE=true \
            "$moss_server_launcher" synthesize "$answer" "${turn_dir}/response.wav" \
            > "${turn_dir}/moss.stdout" 2> "${turn_dir}/moss.stderr"

        # Keep draining ALSA during playback, but discard these frames so the
        # assistant cannot answer its own cloned voice.
        write_pipeline_status playback "capture held; segmentation muted"
        touch "$playback_mute"
        aplay -q -D "$speaker" "${turn_dir}/response.wav"
        sleep 1
        rm -f "$playback_mute"
    fi

    turn_end_ns=$(date +%s%N)
    turn_ms=$(( (turn_end_ns - turn_start_ns) / 1000000 ))
    jq -n --argjson turn "$turn_number" --argjson end_to_end_ms "$turn_ms" \
        --arg transcript "$transcript" --arg answer "$answer" \
        '{turn:$turn,end_to_end_ms:$end_to_end_ms,transcript:$transcript,answer:$answer}' \
        > "${turn_dir}/timing.json"
    printf 'Turn %s end-to-end: %s ms\n' "$turn_number" "$turn_ms"
    write_pipeline_status listening "microphone held continuously"
done
