#!/usr/bin/env python3
"""Live resource monitor for local model runs on Apple Silicon.

Prints one line per interval with per-process CPU, RSS and physical
footprint, GPU utilization, and system memory state. No root and no
third-party packages: per-process numbers come from proc_pid_rusage via
ctypes, system memory from vm_stat, GPU utilization from the
IOAccelerator performance statistics in the IO registry.

Usage:
  tools/qwen36_monitor.py                  # watch the first qwen36* process
  tools/qwen36_monitor.py python3.13       # watch mlx-lm / oMLX runs
  tools/qwen36_monitor.py 12345            # watch a specific pid
  tools/qwen36_monitor.py -i 0.5 qwen36    # 0.5 s interval

The monitor waits for a matching process to appear, follows it until it
exits, then goes back to waiting, so it can be started once and left
running across repeated benchmark runs. Stop with Ctrl-C.

Note on accounting: this runtime maps the 15.1 GB model file-backed, so
its RSS stays small; the honest per-process number is the physical
footprint plus the file-backed page count in the system columns. The
Python stacks allocate anonymously and show up directly in RSS.
"""

import argparse
import ctypes
import os
import plistlib
import re
import subprocess
import sys
import time

PAGE_SIZE = 16384
RUSAGE_INFO_V2 = 2

_libc = ctypes.CDLL(None, use_errno=True)


class _MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_timebase = _MachTimebase()
_libc.mach_timebase_info(ctypes.byref(_timebase))


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def process_usage(pid):
    """Return (cpu_ns_total, rss_bytes, footprint_bytes) or None."""
    info = _RusageInfoV2()
    status = _libc.proc_pid_rusage(pid, RUSAGE_INFO_V2,
                                   ctypes.byref(info))
    if status != 0:
        return None
    ticks = info.ri_user_time + info.ri_system_time
    cpu_ns = ticks * _timebase.numer // _timebase.denom
    return cpu_ns, info.ri_resident_size, info.ri_phys_footprint


_UNINTERESTING = {"sh", "bash", "zsh", "make", "pgrep", "tee", "time"}


def find_pid(target):
    if target.isdigit():
        return int(target) if process_usage(int(target)) else None
    # Match the executable name first; fall back to full command lines but
    # skip wrapper shells so a harness script never shadows the model run.
    for arguments in (["pgrep", "-n", target],
                      ["pgrep", "-f", target]):
        try:
            output = subprocess.run(
                arguments, capture_output=True, text=True,
                timeout=5).stdout.strip()
        except subprocess.TimeoutExpired:
            continue
        for line in reversed(output.splitlines()):
            pid = int(line)
            if pid == 0 or pid == os.getpid():
                continue
            if process_name(pid) in _UNINTERESTING:
                continue
            return pid
    return None


def process_name(pid):
    try:
        return subprocess.run(
            ["ps", "-o", "comm=", "-p", str(pid)], capture_output=True,
            text=True, timeout=5).stdout.strip().split("/")[-1]
    except subprocess.TimeoutExpired:
        return "?"


_VM_STAT_PATTERN = re.compile(r"^(.+?):\s+(\d+)\.$", re.MULTILINE)


def system_memory():
    try:
        output = subprocess.run(["vm_stat"], capture_output=True,
                                text=True, timeout=5).stdout
    except subprocess.TimeoutExpired:
        return {}
    pages = {name: int(value) for name, value in
             _VM_STAT_PATTERN.findall(output)}
    def gb(name):
        return pages.get(name, 0) * PAGE_SIZE / 1e9
    return {
        "free": gb("Pages free") + gb("Pages speculative"),
        "wired": gb("Pages wired down"),
        "filebacked": gb("File-backed pages"),
        "anonymous": gb("Anonymous pages"),
        "compressor": gb("Pages occupied by compressor"),
    }


def memory_pressure():
    try:
        value = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except subprocess.TimeoutExpired:
        return "?"
    return {"1": "normal", "2": "warn", "4": "critical"}.get(value, value)


def gpu_utilization():
    try:
        output = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a"],
            capture_output=True, timeout=5).stdout
        entries = plistlib.loads(output)
    except (subprocess.TimeoutExpired, plistlib.InvalidFileException,
            ValueError):
        return None
    for entry in entries:
        statistics = entry.get("PerformanceStatistics", {})
        value = statistics.get("Device Utilization %")
        if value is not None:
            return int(value)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Live CPU/memory/GPU monitor for local model runs.")
    parser.add_argument("target", nargs="?", default="qwen36",
                        help="process name substring or pid "
                             "(default: qwen36)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="sampling interval in seconds (default: 1)")
    arguments = parser.parse_args()

    print(f"monitoring '{arguments.target}' every "
          f"{arguments.interval:g}s; Ctrl-C to stop", flush=True)
    pid = None
    previous_cpu_ns = None
    previous_wall = None
    try:
        while True:
            if pid is None:
                pid = find_pid(arguments.target)
                if pid is not None:
                    print(f"attached to pid {pid} "
                          f"({process_name(pid)})", flush=True)
                    previous_cpu_ns = None
            line = time.strftime("%H:%M:%S")
            if pid is not None:
                usage = process_usage(pid)
                if usage is None:
                    print(f"{line}  pid {pid} exited; waiting", flush=True)
                    pid = None
                    previous_cpu_ns = None
                else:
                    cpu_ns, rss, footprint = usage
                    wall = time.monotonic()
                    if previous_cpu_ns is not None:
                        cpu_percent = ((cpu_ns - previous_cpu_ns) /
                                       ((wall - previous_wall) * 1e9)
                                       * 100.0)
                        cpu_text = f"{cpu_percent:6.1f}%"
                    else:
                        cpu_text = "     -"
                    previous_cpu_ns = cpu_ns
                    previous_wall = wall
                    line += (f"  pid {pid}  cpu {cpu_text}  "
                             f"rss {rss / 1e9:6.2f}G  "
                             f"foot {footprint / 1e9:6.2f}G")
            else:
                line += "  [waiting for process]"
            gpu = gpu_utilization()
            line += f" | gpu {gpu:3d}%" if gpu is not None else " | gpu  ?"
            memory = system_memory()
            if memory:
                line += (f" | free {memory['free']:5.2f}G"
                         f"  wired {memory['wired']:5.2f}G"
                         f"  filebk {memory['filebacked']:5.2f}G"
                         f"  anon {memory['anonymous']:5.2f}G"
                         f"  compr {memory['compressor']:5.2f}G"
                         f"  pressure {memory_pressure()}")
            print(line, flush=True)
            time.sleep(arguments.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
