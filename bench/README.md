# Target probe

`target-probe` records the Linux CPU and memory environment and measures sequential read-and-sum bandwidth. It is a bring-up tool for hardware backends. It is not an inference benchmark.

Build on a Linux x86-64 host:

```sh
make CFLAGS="-O3 -std=c11 -Wall -Wextra -Wpedantic -march=x86-64-v3"
```

Run:

```sh
./build/target-probe
./build/target-probe --mib 512 --seconds 3
```

Defaults:

- buffer: 512 MiB;
- duration: 2 seconds per thread-count measurement;
- measurements: one thread, then all visible logical CPUs up to four threads.

The benchmark reads a buffer larger than the target cache and accumulates every 64-bit word. Reported bandwidth includes the accumulation cost. It must not be presented as STREAM bandwidth or as a model throughput result.

Recorded runs:

- [DigitalOcean Premium Intel, 2 vCPU, 4 GiB](results/digitalocean-premium-intel-2vcpu-4gb.md)
