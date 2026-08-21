# ThreeHub resident voice assistant

This is the always-on A113X pipeline deployed on the ThirdReality TRHub-V3.
One `arecord` process holds ALSA continuously and feeds the native C
segmenter. Whisper Base multilingual, Qwen3.5-0.8B, and MOSS-TTS-Nano each
remain alive with their model mapped between turns. There is no Python process
in capture, VAD, inference, synthesis, monitoring, or supervision.

```text
ALSA (held continuously) -> C RMS segmenter -> resident Whisper + Silero VAD
                                              -> resident Qwen
                                              -> resident MOSS -> ALSA speaker
```

Speaker playback creates `playback.active`. The capture process continues to
drain ALSA while that file exists but discards those frames, preventing the
cloned response from becoming a new user turn. Up to eight complete utterances
can wait while a slow inference stage is active; additional turns are counted
as dropped instead of filling `/dev/shm`.

## Live monitor

The capture process atomically updates peak and RMS levels ten times per
second. The monitor reads that same state and never opens the microphone:

```sh
ssh root@100.123.75.40 /root/threehub-voice/threehub-voice-monitor.sh
```

The production defaults start after 100 ms at 1.2% frame RMS and stop after
1.2 seconds below 0.7% RMS. `VOICE_START_PERCENT`, `VOICE_STOP_PERCENT`,
`VOICE_START_MS`, `VOICE_STOP_MS`, and `VOICE_MAX_QUEUED_UTTERANCES` tune the
installation without rebuilding.

## Build and release

All target executables are built before upload. Pin whisper.cpp v1.8.0 at
commit `41fc9dea6a4fe056424be86f61164413903fcff4`, then run in a Linux/AArch64
Debian 12 environment:

```sh
tools/threehub-voice/build-a113x.sh build/threehub-release /path/to/whisper.cpp
```

The build uses GCC/CMake only. The assistant launcher downloads every public
runtime/model artifact from GitHub Releases and verifies its SHA-256. The
private voice reference stays at `/root/threehub-voice/voice_ref.wav` and is
never uploaded.

Install the launcher, monitor, and unit, then enable supervision:

```sh
install -m 0755 threehub-voice-assistant.sh threehub-voice-monitor.sh \
  /root/threehub-voice/
install -m 0644 threehub-voice-assistant.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now threehub-voice-assistant.service
```

`systemd` restarts the pipeline on failure. Startup does one real request
through each model before setting the pipeline state to `listening`; requests
after that use the already-running servers. Raw validation measurements are in
[`results.json`](results.json).
