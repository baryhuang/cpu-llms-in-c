# User tools

`tools/` contains commands a user can run directly. Model compilers, runtime source, kernel benchmarks and numerical validation programs do not live here.

## Qwen3.8

| Command | Direct use |
|---|---|
| `qwen38_chat.sh` | Run one prompt, start resident terminal chat, or launch the local Chatbox setup |
| `qwen38_serve.py` | Expose the resident C/Metal runtime through an OpenAI-compatible HTTP API |
| `qwen38_monitor.py` | Record CPU, GPU, process footprint and system memory on Apple Silicon |

`support/qwen38_chatbox_config.py` is an implementation detail called by `qwen38_chat.sh`; it is separated because users do not invoke it directly.

## Hardware and comparison

| Path | Direct use |
|---|---|
| `target_probe.c` | Built by `make linux-tools`; records CPU topology, ISA and memory bandwidth for a target |
| `compare/run_all_sets.sh` | Runs the fixed Qwen3.8 workload set through mlx-lm, oMLX and llama.cpp |
| `compare/*.py` | Support programs called by `run_all_sets.sh` |

## Code that is intentionally elsewhere

| Code type | Location |
|---|---|
| Qwen3.8 offline image compiler | [`compiler/qwen3.8-27b/apple-m3-pro/`](../compiler/qwen3.8-27b/apple-m3-pro/) |
| Qwen3.8 runtime implementation | [`models/qwen3.8-27b/targets/apple-m3-pro/`](../models/qwen3.8-27b/targets/apple-m3-pro/) |
| Qwen3.8 compiled command front ends | [`commands/`](../models/qwen3.8-27b/targets/apple-m3-pro/commands/) |
| Qwen3.8 kernel benchmarks | [`benchmarks/`](../models/qwen3.8-27b/targets/apple-m3-pro/benchmarks/) |
| Qwen3.8 numerical/debug validation | [`validation/`](../models/qwen3.8-27b/targets/apple-m3-pro/validation/) |
| Whisper small.en command source | [`models/whisper-small.en/commands/`](../models/whisper-small.en/commands/) |
| Whisper small.en benchmarks and checks | [`benchmarks/`](../models/whisper-small.en/benchmarks/) · [`validation/`](../models/whisper-small.en/validation/) |
| Automated regression tests | [`tests/`](../tests/) |
