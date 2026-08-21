# MOSS-TTS-Nano-100M on Amlogic A113X

## Outcome

The production optimization replaces MOSS Nano's quadratic global-transformer
decode with an exact incremental KV cache. A batched prefill graph evaluates
the text/reference prompt once and copies every layer's post-RoPE K/V tensors
directly into a persistent cache. A fixed one-row graph then appends one audio
frame and attends over the cache for each generation step.

The deterministic `Ready.` fixture is a strict correctness gate: the baseline,
KV-cache, and A113X-tuned binaries produce the same 122,924-byte WAV with
SHA-256 `5d09fd812e60f4be7eb6e64b50c5422f25006d317f5167ec184aec2d9c5b35fa`.

## A113X specialization

The deployed CPU reports `fp asimd aes pmull sha1 sha2 crc32`; it does not
report dot-product or Armv8.2 FP16 arithmetic. The build therefore uses:

- `-mcpu=cortex-a53+crypto+crc -mtune=cortex-a53`;
- ARMv8-A NEON/FMA through GGML;
- link-time optimization, dead-section removal, and no semantic interposition;
- four OpenMP threads pinned close to cores at runtime;
- no host-native ISA discovery, dot-product, FP16-vector, i8mm, SVE, or
  llamafile kernels.

`GGML_CPU_ARM_ARCH=armv8-a+crypto+crc` is intentional. Setting only
`GGML_NATIVE=OFF` is insufficient because audio.cpp derives that cache entry
from `ENGINE_ENABLE_NATIVE_CPU`; a build performed under an Apple-Silicon
Linux VM can otherwise inherit newer host ISA features and fault on Cortex-A53.

## Reproduce

The patch is pinned to audio.cpp commit
`26dcb5c4cf5aa016ae6285096a7b45f2671e5d17`:

```sh
models/moss-tts-nano/targets/a113x/build-a113x.sh build/moss-a113x
```

On macOS, run the script in a `linux/arm64` Debian 12 container with Git, GCC,
CMake, Ninja, and binutils installed. The script refuses non-Linux-AArch64
hosts, validates a pristine pinned source tree, checks the patch before applying
it, and produces stripped CLI and server binaries. No Python is used.

The production launcher automatically downloads and SHA-256 verifies the
server binary and Q8_0 model, starts a localhost-only persistent server, pins its
four OpenMP workers, and performs one warm-up request before accepting speech:

```sh
models/moss-tts-nano/targets/a113x/run-moss-server-a113x.sh start
models/moss-tts-nano/targets/a113x/run-moss-server-a113x.sh synthesize \
  'Ready.' ready.wav
```

Set `MOSS_VOICE_REF` to select a private reference file. The default is
`/root/threehub-voice/voice_ref.wav`; no reference audio is downloaded or
published.

## Benchmark contract

- Board: ThirdReality TRHub-V3 / Amlogic A113X, four Cortex-A53 cores, 2 GB RAM.
- Runtime: CPU only, four threads, model Q8_0.
- Reference: private 10-second mono PCM16 48 kHz voice reference; hash recorded
  in the private deployment log, never published.
- Text: `Ready.`
- Generation: greedy, `--max-tokens 20`.
- Output: 0.64 seconds, stereo PCM16 48 kHz.
- Memory: `/usr/bin/time` maximum RSS; no swap.

Raw before/after measurements are in [`results.json`](results.json).

| Runtime | Process/request wall | Session wall | RTF | Peak RSS | Output |
|---|---:|---:|---:|---:|---|
| audio.cpp 0.6.1, full prompt every frame | 125.73 s | 121.611 s | 190.017 | 858,236 KiB | reference hash |
| + exact incremental KV cache + A113X build | **72.75 s** | **69.520 s** | **108.625** | 859,460 KiB | byte-identical |
| + persistent production server, warm | **9.83 s** median | n/a | **15.359** | 861,584 KiB | byte-identical, repeatable |

The cold process-wall increment is **1.73x** and the session-wall increment is
**1.75x**. The production warm path is **12.79x** faster than the original cold
CLI and **7.40x** faster than the optimized cold CLI. The server reuses the
unchanged reference encoding, loaded weights, and fixed decode graph while
resetting and prefilling the global-transformer KV cache for each request.

The two measured warm requests were 9.82 and 9.84 seconds and produced the same
WAV hash as each other and the baseline. Server peak RSS was 861,584 KiB;
post-request idle RSS was 506,712 KiB and the process reported zero swap. This
is substantially more usable, but still not real-time synthesis: the remaining
reference-independent local frame and waveform decoders dominate on Cortex-A53.
