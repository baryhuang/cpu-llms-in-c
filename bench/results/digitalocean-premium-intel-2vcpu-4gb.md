# DigitalOcean Premium Intel: 2 vCPU, 4 GiB

Date: 2026-08-09

## Environment

| Item | Observed value |
|---|---|
| DigitalOcean size | `s-2vcpu-4gb-intel` |
| Region | `sfo3` |
| CPU label | `DO-Premium-Intel` |
| CPU topology | 2 vCPU, one socket, two cores, one thread per core |
| CPU identification | Family 6, model 85, stepping 7 |
| Virtualization | KVM |
| ISA exposed to the guest | SSE4.2, AVX, AVX2, FMA |
| Visible memory | 4,106,104,832 bytes |
| Swap | None |
| Storage | 80 GiB virtual disk |
| Operating system | Ubuntu 24.04 x86-64 |
| Kernel | Linux 6.8.0-124-generic |

The Droplet uses shared vCPUs. Results may vary with host placement and neighboring workloads.

## Binary

```text
compiler: GCC 13.3.0
flags:    -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror -march=x86-64-v3 -static
format:   ELF 64-bit x86-64, statically linked
sha256:   8f6ee22c381c241c82241311600f943b630422a9a26176c715bba628c9cf3013
```

The initial local cross-toolchain downloads did not complete. This binary was compiled on the Ubuntu target, copied to the development machine, checked by SHA-256, and copied back to `/root/target-probe`. The locally stored and redeployed binary hashes matched.

## Method

`target-probe` sequentially reads a buffer and accumulates every 64-bit word. Worker threads are pinned to logical CPUs in index order. Each worker receives a disjoint portion of the buffer.

This is a read-and-sum kernel. It includes integer accumulation cost. It is not STREAM, storage throughput, cache bandwidth, or language-model inference.

Command for the repeated 512 MiB runs:

```sh
/root/target-probe --mib 512 --seconds 3
```

## 512 MiB results

| Run | One thread | Two threads |
|---:|---:|---:|
| 1 | 10.231 GiB/s | 19.947 GiB/s |
| 2 | 9.830 GiB/s | 18.524 GiB/s |
| 3 | 9.272 GiB/s | 18.610 GiB/s |
| Median | 9.830 GiB/s | 18.610 GiB/s |

Median two-thread scaling over one thread was 1.89x.

## 1,024 MiB result

Command:

```sh
/root/target-probe --mib 1024 --seconds 3
```

| One thread | Two threads |
|---:|---:|
| 9.614 GiB/s | 18.455 GiB/s |

No swap device existed during the run. After the run, the guest reported 3,641,618,432 bytes available.

## Interpretation

The 512 MiB and 1,024 MiB results are close. The measured path is not dominated by the reported 8 MiB aggregate L2 cache.

The two-vCPU result is approximately twice the one-vCPU result. This establishes a hardware-backend baseline for sequential packed-weight reads on this Droplet class. It does not predict token throughput because unpacking, dot products, activation functions, attention, routing, and reductions are absent.

These measurements do not validate the Intel Celeron J3455 estimates. The DigitalOcean CPU exposes AVX2 and FMA, while the J3455 target does not.
