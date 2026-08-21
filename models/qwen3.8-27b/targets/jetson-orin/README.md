# Qwen3.8-27B on Jetson Orin Nano 8 GB

This target runs the Qwen3.8-27B text model as a resident, loopback-only
OpenAI-compatible service on the 8 GB Jetson at `100.76.66.42`. The deployed
profile uses full CUDA weight residency, direct-I/O GGUF loading, a 1,024-token
single-user context and the smallest valid recurrent-state checkpoint pool.

## Deployed result

| Property | Selected value |
|---|---:|
| Artifact | Unsloth `Qwen3.8-27B-UD-IQ1_S.gguf` |
| File size | 6,192,222,208 bytes |
| Effective quantization | mixed dynamic IQ1, reported as 1.5625 bpw |
| Runtime | llama.cpp `749f688fcaa4c472ec034b08cb8a907c45cfaa02` |
| Decode benchmark | **4.17 tok/s**, 16 tokens, three repetitions, FlashAttention on |
| Resident API smoke | **4.07–4.09 tok/s** decode |
| Completed Chatbox turns | **114.41 s** for 164 prompt + 377 generated tokens; **156.47 s** for 202 + 525 |
| CPU-only baseline | 0.75 tok/s |
| Load mode | direct I/O (`--load-mode dio`) |
| Context / concurrency | 1,024 tokens / one slot |
| Service | `qwen38.service`, `127.0.0.1:8199` |

The host is an Orin Nano Super with 7.3 GiB usable unified DRAM, L4T R39.2.1,
Ubuntu 24.04.4, CUDA 13.2 and an 85.8 MiB/s measured model volume. It runs in
`MAXN_SUPER`; `jetson_clocks` pins the six CPU cores to 1.728 GHz, GPU to
1.020 GHz and EMC to 3.199 GHz before the service starts.

## Why this weight and loading format

The repository's 15.139 GB affine-Q4 image cannot fit. Among the current public
GGUFs, UD-IQ1_S is the only tier that leaves enough physical memory for the
159 MB recurrent state, KV cache, CUDA work buffers and operating system. It is
not uniformly one-bit: its 851 tensors use a dynamic mix led by 264 `IQ1_S`,
59 `IQ2_XXS`, 25 `IQ1_M`, 96 `Q8_0` and 353 small `F32` tensors. Embedding and
output tensors are protected at higher bit widths.

Direct I/O is important on this machine. Full GPU offload with mmap faulted the
GGUF while copying it into CUDA allocations, took 98 seconds to start and
reported an 11,926,484 KiB RSS peak. Direct I/O bypassed that duplicate page
cache pass: startup fell to 82 seconds and peak RSS to 6,744,916 KiB while
decode rose from 3.29 to 3.92 tok/s in the short probe.

The production service additionally uses:

- full GPU layer placement; a 32-layer CPU/GPU split reached only 1.67 tok/s;
- ordinary CUDA allocations; managed allocation was slightly slower and larger;
- FlashAttention, worth 1.5% in the repeated decode benchmark;
- CUDA graphs disabled, avoiding retained graph allocations for a roughly 2%
  resident-decode cost;
- batch and microbatch 2, the smallest values accepted by the server's internal
  two-token sequence-removal check;
- one recurrent context checkpoint; zero checkpoints corrupted DeltaNet output;
- prompt caching and slot LCP reuse disabled to avoid retaining large recurrent
  state snapshots between unrelated requests.

A 4 GiB swap file is reserved for cold service and OS pages, not model-weight
streaming. Layer streaming from the 85.8 MiB/s model volume would need roughly
72 seconds for every weight pass and is not viable for token decoding.

## Use the service

The server intentionally binds only to loopback. Forward it over SSH:

```sh
ssh -L 8199:127.0.0.1:8199 root@100.76.66.42
```

Then call it locally:

```sh
curl http://127.0.0.1:8199/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0,
    "max_tokens": 96
  }'
```

Thinking mode is the default because the extreme IQ1 checkpoint returned an
immediate EOS in one no-thinking probe. Even for short answers, allow at least
96 completion tokens so the reasoning trace can finish before final content.

Useful operations on the Jetson:

```sh
systemctl status qwen38.service
systemctl restart qwen38.service
curl http://127.0.0.1:8199/health
journalctl -u qwen38.service -f
```

Current llama.cpp retains approximately 300 MB of recurrent state on some
successive Qwen3.8 requests even with the minimized cache settings. The
`qwen38-memory-guard.timer` checks once per minute and restarts the server only
while it is idle if the server's own `VmSwap` reaches 1 GiB. This prevents the
multi-gigabyte swap storm seen with the default 32-checkpoint prompt cache.
Reloading takes about 75–85 seconds from eMMC.

## Interactive trial and final disposition

The final trial used Chatbox through an SSH tunnel with the deployed 1,024-token
context. Two consecutive, unrestricted-thinking turns completed as follows:

| Turn | Prompt | Generation | Total wall time |
|---|---:|---:|---:|
| 1 | 164 tokens in 21.78 s (7.53 tok/s) | 377 tokens in 92.63 s (4.06 tok/s) | **114.41 s** |
| 2 | 202 tokens in 28.22 s (7.16 tok/s) | 525 tokens in 128.26 s (4.09 tok/s) | **156.47 s** |

A later conversation turn replayed 310 prompt tokens in 41.71 seconds before
generation because prompt-state reuse is disabled for bounded recurrent-state
memory. Near the end of the session the server occupied 6,814,568 KiB RSS and
1,012,092 KiB process swap; the complete system had only 272,711,680 bytes of
available DRAM. Crossing the 1 GiB process-swap guard threshold causes an idle
restart followed by another 75–85 second model load.

For context, the separate Apple M3 Pro 36 GB target reports 7.96 tok/s without
MTP and 11.95 tok/s with adaptive MTP on its five-workload aggregate. Those are
not same-prompt measurements, but they explain the observed user experience:
the M3 can retain a 15.139 GB affine-Q4 image and an MTP draft model, whereas
this Jetson must use a 6.192 GB, 1.5625-bpw IQ1 image without MTP.

**Disposition:** the experiment proves that Qwen3.8-27B can run fully offloaded
on an 8 GB Orin Nano, and direct I/O materially improves startup residency, but
the resulting service is not recommended for interactive chat. On-disk layout
cannot remove the approximately 4.1 tok/s decode ceiling, repeated-history
prefill cost, extreme-quant quality loss or recurrent-state swap pressure. Use
a smaller Q4-class model on this device, or run the 27B model on the 36 GB M3.

## Build and deploy

`build-llama.sh` enforces the tested llama.cpp revision and builds native
Cortex-A78AE and `sm_87` code. `download-model.sh` resumes the model download
and refuses to rename it until this SHA-256 passes:

```text
3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887
```

Install the supplied units and guard script as root, then enable them:

```sh
install -m 0644 qwen38-model-swap.service qwen38.service \
  qwen38-memory-guard.service qwen38-memory-guard.timer \
  /etc/systemd/system/
install -m 0755 qwen38-memory-guard.sh /usr/local/sbin/
systemctl daemon-reload
systemctl enable --now qwen38-model-swap.service \
  qwen38.service qwen38-memory-guard.timer
```

The optional `llama.cpp-jetson-integrated-host.patch` exposes upstream's
disabled CUDA integrated-host path behind
`GGML_CUDA_ENABLE_INTEGRATED_HOST`. It is retained only to reproduce the
experiment. Do not enable it in production: pinning the complete image failed
with Jetson `NvMap` error 12.

## 2026 research disposition

- NVIDIA's current CUDA for Tegra guidance confirms that device, host and
  unified allocations share physical DRAM on Orin, and recommends mapped pinned
  memory for coalesced one-pass integrated-GPU access. The local 256 MiB probe
  measured 30.18 GB/s device, 24.61 GB/s registered mmap and 22.95 GB/s managed
  reads. The full pinned model nevertheless exceeded the driver's `NvMap`
  allocation limit. See the [CUDA for Tegra application note](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html).
- A 2026 Qwen3.8 GGUF study reports that stock IQ1_S is a severe quality tier
  and that adding MTP can tip a marginal GPU layout into a 40% slowdown. The
  separate MTP image was therefore not installed. See the
  [Qwen3.8 quantization discussion](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/49).
- Current llama.cpp ARM investigation notes that decode is weight-bandwidth
  bound and that repacking makes bytes read differ from the on-disk GGUF size.
  See [llama.cpp issue 26484](https://github.com/ggml-org/llama.cpp/issues/26484).
- T-MAC's CPU lookup-table kernels and BitNet's ternary kernels are promising
  for models encoded for those runtimes, but neither accepts this Qwen3.8 mixed
  IQ GGUF. See [T-MAC](https://github.com/microsoft/T-MAC/) and
  [BitNet](https://github.com/microsoft/BitNet).
- FluxBin reports strong binary-weight CUDA results, but its reported A100 path
  is not an available Orin/Qwen3.8 deployment stack. See the
  [FluxBin preprint](https://arxiv.org/abs/2608.15602).

Exact raw measurements and rejected layouts are in [`results.json`](results.json).
The five-case ARC script is a generative smoke profile, not an official ARC
score; the long run was stopped after it exposed unsafe default cache growth.
