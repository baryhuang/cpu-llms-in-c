#!/bin/sh

set -eu

REPOSITORY="baryhuang/llm-in-c"
SERVER_RELEASE="moss-tts-nano-q8-a113x-kvcache-r1"
MODEL_RELEASE="moss-tts-nano-q8-a113x-audiocpp-v0.6.1"
SERVER_ASSET="audiocpp-server-v0.6.1-linux-aarch64-a113x-kvcache-r1"
MODEL_ASSET="moss-tts-nano-100m-q8_0.gguf"
SERVER_SHA256="17037774d2b9fdab5fac3093c50c7fd4d43d085394744c426946e53bcd0c80f0"
MODEL_SHA256="3b9c138ce4093514b6493b77abaf3e62f4d2ac58412616d2f7ee89cca28be388"

cache_dir=${MOSS_NANO_CACHE_DIR:-/dev/shm/moss-tts-nano-q8-a113x-v0.6.1}
voice_reference=${MOSS_VOICE_REF:-/root/threehub-voice/voice_ref.wav}
server_port=${MOSS_SERVER_PORT:-18088}
server_pid_file="${cache_dir}/moss-server.pid"
server_log="${MOSS_SERVER_LOG:-/root/threehub-voice/moss-server.log}"
server_config="${cache_dir}/moss-server.json"
server_binary="${cache_dir}/${SERVER_ASSET}"
model_file="${cache_dir}/${MODEL_ASSET}"

usage() {
    printf 'Usage:\n' >&2
    printf '  %s start\n' "$0" >&2
    printf '  %s synthesize TEXT OUTPUT.wav\n' "$0" >&2
    printf '  %s status\n' "$0" >&2
    printf '  %s stop\n' "$0" >&2
    exit 2
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'error: sha256sum or shasum is required\n' >&2
        exit 1
    fi
}

download() {
    url=$1
    destination=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 --output "$destination" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$destination" "$url"
    else
        printf 'error: curl or wget is required\n' >&2
        exit 1
    fi
}

ensure_asset() {
    release=$1
    asset=$2
    destination=$3
    expected_sha256=$4
    stamp="${destination}.sha256-ok"

    if [ -f "$destination" ] && [ -f "$stamp" ] &&
       [ "$(sed -n '1p' "$stamp")" = "$expected_sha256" ]; then
        return
    fi
    if [ -f "$destination" ] && [ "$(sha256_file "$destination")" = "$expected_sha256" ]; then
        printf '%s\n' "$expected_sha256" > "$stamp"
        return
    fi

    temporary="${destination}.part.$$"
    rm -f -- "$temporary"
    download \
        "https://github.com/${REPOSITORY}/releases/download/${release}/${asset}" \
        "$temporary"
    if [ "$(sha256_file "$temporary")" != "$expected_sha256" ]; then
        rm -f -- "$temporary"
        printf 'error: SHA-256 mismatch for %s\n' "$asset" >&2
        exit 1
    fi
    mv "$temporary" "$destination"
    printf '%s\n' "$expected_sha256" > "$stamp"
}

server_is_healthy() {
    command -v curl >/dev/null 2>&1 &&
        curl -fsS "http://127.0.0.1:${server_port}/health" >/dev/null 2>&1
}

ensure_runtime() {
    case "$(uname -s):$(uname -m)" in
        Linux:aarch64|Linux:arm64) ;;
        *)
            printf 'error: this runtime targets Linux AArch64\n' >&2
            exit 1
            ;;
    esac
    if [ ! -r "$voice_reference" ]; then
        printf 'error: voice reference is unavailable: %s\n' "$voice_reference" >&2
        exit 1
    fi
    mkdir -p "$cache_dir"
    ensure_asset "$SERVER_RELEASE" "$SERVER_ASSET" "$server_binary" "$SERVER_SHA256"
    ensure_asset "$MODEL_RELEASE" "$MODEL_ASSET" "$model_file" "$MODEL_SHA256"
    chmod 0755 "$server_binary"
    if ldd "$server_binary" 2>/dev/null | grep -q 'not found'; then
        ldd "$server_binary" >&2 || true
        printf 'error: native runtime libraries are missing (Debian package: libgomp1)\n' >&2
        exit 1
    fi
}

write_server_config() {
    case "${model_file}${voice_reference}" in
        *'"'*|*'\'*)
            printf 'error: model and reference paths cannot contain quotes or backslashes\n' >&2
            exit 1
            ;;
    esac
    temporary="${server_config}.part.$$"
    {
        printf '{\n'
        printf '  "host": "127.0.0.1",\n'
        printf '  "port": %s,\n' "$server_port"
        printf '  "backend": "cpu",\n'
        printf '  "threads": 4,\n'
        printf '  "lazy_load": false,\n'
        printf '  "ui_enabled": false,\n'
        printf '  "ui_management": false,\n'
        printf '  "busy_timeout_ms": 600000,\n'
        printf '  "models": [{\n'
        printf '    "id": "moss-nano",\n'
        printf '    "family": "moss_tts_nano",\n'
        printf '    "path": "%s",\n' "$model_file"
        printf '    "task": "clon",\n'
        printf '    "mode": "offline",\n'
        printf '    "default_voice_preset": {"voice_ref": "%s"}\n' "$voice_reference"
        printf '  }]\n'
        printf '}\n'
    } > "$temporary"
    mv "$temporary" "$server_config"
}

request_json() {
    text=$1
    max_tokens=${MOSS_MAX_TOKENS:-40}
    do_sample=${MOSS_DO_SAMPLE:-true}
    case "$max_tokens" in
        ''|*[!0-9]*)
            printf 'error: MOSS_MAX_TOKENS must be a positive integer\n' >&2
            exit 1
            ;;
    esac
    case "$do_sample" in
        true|false) ;;
        *)
            printf 'error: MOSS_DO_SAMPLE must be true or false\n' >&2
            exit 1
            ;;
    esac
    jq -cn \
        --arg input "$text" \
        --argjson max_tokens "$max_tokens" \
        --argjson do_sample "$do_sample" \
        '{model:"moss-nano", input:$input, max_tokens:$max_tokens, seed:1234, options:{do_sample:$do_sample}}'
}

start_server() {
    if server_is_healthy; then
        return
    fi
    ensure_runtime
    command -v jq >/dev/null 2>&1 || {
        printf 'error: jq is required for JSON-safe speech requests\n' >&2
        exit 1
    }
    write_server_config
    nohup env \
        OMP_NUM_THREADS=4 \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        GOMP_CPU_AFFINITY=0-3 \
        "$server_binary" --config "$server_config" --no-ui \
        >"$server_log" 2>&1 </dev/null &
    printf '%s\n' "$!" > "$server_pid_file"

    attempt=0
    while [ "$attempt" -lt 60 ]; do
        if server_is_healthy; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    if ! server_is_healthy; then
        tail -80 "$server_log" >&2 || true
        printf 'error: MOSS server did not become healthy\n' >&2
        exit 1
    fi

    if [ "${MOSS_SKIP_WARMUP:-0}" != 1 ]; then
        warm_request=$(MOSS_DO_SAMPLE=false request_json 'Ready.')
        curl -fsS \
            -o /dev/null \
            "http://127.0.0.1:${server_port}/v1/audio/speech" \
            -H 'Content-Type: application/json' \
            -d "$warm_request"
    fi
}

stop_server() {
    if [ ! -r "$server_pid_file" ]; then
        printf 'MOSS server is not managed by this launcher\n'
        return
    fi
    server_pid=$(sed -n '1p' "$server_pid_file")
    case "$server_pid" in
        ''|*[!0-9]*)
            printf 'error: invalid server PID file: %s\n' "$server_pid_file" >&2
            exit 1
            ;;
    esac
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
    fi
    rm -f -- "$server_pid_file"
}

command_name=${1:-}
case "$command_name" in
    start)
        [ "$#" -eq 1 ] || usage
        start_server
        printf 'MOSS server ready at http://127.0.0.1:%s\n' "$server_port"
        ;;
    synthesize)
        [ "$#" -eq 3 ] || usage
        start_server
        payload=$(request_json "$2")
        curl -fsS \
            -o "$3" \
            "http://127.0.0.1:${server_port}/v1/audio/speech" \
            -H 'Content-Type: application/json' \
            -d "$payload"
        ;;
    status)
        [ "$#" -eq 1 ] || usage
        if server_is_healthy; then
            curl -fsS "http://127.0.0.1:${server_port}/health"
            printf '\n'
        else
            printf 'MOSS server is not healthy\n' >&2
            exit 1
        fi
        ;;
    stop)
        [ "$#" -eq 1 ] || usage
        stop_server
        ;;
    *) usage ;;
esac
