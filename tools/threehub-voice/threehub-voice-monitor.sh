#!/bin/sh

set -eu

state_dir=${VOICE_ASSISTANT_STATE_DIR:-/dev/shm/threehub-voice}
capture_status=${state_dir}/capture-status.json
pipeline_status=${state_dir}/pipeline-status.json
full_scale=${VOICE_MONITOR_FULL_SCALE_PERCENT:-10}

command -v jq >/dev/null 2>&1 || {
    printf 'error: jq is required\n' >&2
    exit 1
}

while :; do
    if [ ! -r "$capture_status" ] || [ ! -r "$pipeline_status" ]; then
        printf '\033[2J\033[HThreeHub resident voice monitor\n\nStarting...\n'
        sleep 0.25
        continue
    fi

    # One jq process renders the entire frame. This matters on the four-core
    # A53 while Whisper or MOSS owns the compute budget.
    jq -rs --argjson scale "$full_scale" '
        .[0] as $capture | .[1] as $pipeline |
        (($capture.peak_percent * 40 / $scale + 0.5) | floor) as $raw_filled |
        ([[$raw_filled, 0] | max, 40] | min) as $filled |
        ("#" * $filled + "." * (40 - $filled)) as $bar |
        "\u001b[2J\u001b[HThreeHub resident voice monitor\n\n" +
        "Capture:  \($capture.state) [\($bar)]\n" +
        "Peak:     \($capture.peak_percent)%   RMS: \($capture.rms_percent)%   " +
        "trigger: \($capture.start_threshold_percent)%   bar scale: \($scale)%\n" +
        "Pipeline: \($pipeline.stage) \($pipeline.detail)\n" +
        "Turns captured since start: \($capture.utterances)\n\n" +
        "Ctrl-C exits the monitor; it does not stop the assistant."
    ' "$capture_status" "$pipeline_status"
    sleep 0.25
done
