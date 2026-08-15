#!/usr/bin/env python3
"""Live resource monitor for local model runs on Apple Silicon.

Three modes, standard library only, no root:

  live dashboard (default on a terminal)
      one sparkline row per metric, redrawn in place every interval.
  line log (--log, or when stdout is not a terminal)
      one plain text line per interval, suitable for tee.
  chart render (--render samples.jsonl [-o run.html])
      turns a recording into a standalone HTML page with SVG line charts:
      utilization, process memory, and system memory over time.

Add --record samples.jsonl to either monitoring mode to save every sample
for later rendering.

Per-process numbers come from proc_pid_rusage via ctypes (CPU percent is
computed from user+system time deltas), system memory from vm_stat, and
GPU utilization from the IOAccelerator performance statistics in the IO
registry. The gpu column is the whole-device utilization of the built-in
GPU, not a per-process figure.

Usage:
  tools/qwen36_monitor.py                  # watch the next qwen36* process
  tools/qwen36_monitor.py python3.13       # watch mlx-lm / oMLX runs
  tools/qwen36_monitor.py 12345            # watch a specific pid
  tools/qwen36_monitor.py -i 0.5 --record run.jsonl
  tools/qwen36_monitor.py --render run.jsonl -o run.html

The monitor waits for a matching process to appear, follows it until it
exits, then goes back to waiting, so it can be started once and left
running across repeated benchmark runs. Stop with Ctrl-C.

Note on accounting: this runtime maps the 15.1 GB model file-backed, so
its RSS stays small; the honest per-process number is the physical
footprint plus the wired/file-backed movement in the system columns. The
Python stacks allocate anonymously and show up directly in RSS.
"""

import argparse
import ctypes
import json
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
    """Device-level GPU statistics. macOS exposes no per-GPU-core
    counters (the 14 cores report as one device), but the driver splits
    utilization into renderer and tiler engines."""
    try:
        output = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a"],
            capture_output=True, timeout=5).stdout
        entries = plistlib.loads(output)
    except (subprocess.TimeoutExpired, plistlib.InvalidFileException,
            ValueError):
        return {}
    for entry in entries:
        statistics = entry.get("PerformanceStatistics", {})
        if "Device Utilization %" in statistics:
            return {
                "device": int(statistics["Device Utilization %"]),
                "renderer": statistics.get("Renderer Utilization %"),
                "tiler": statistics.get("Tiler Utilization %"),
            }
    return {}


def cpu_topology():
    """(efficiency_count, performance_count). On Apple Silicon the
    efficiency cluster occupies the low processor indices; verified on
    this M3 Pro by pinning background-QoS spinners to cpu0-5."""
    def level(name):
        try:
            return int(subprocess.run(
                ["sysctl", "-n", f"hw.{name}.logicalcpu"],
                capture_output=True, text=True, timeout=5).stdout)
        except (subprocess.TimeoutExpired, ValueError):
            return 0
    performance = efficiency = 0
    try:
        names = {index: subprocess.run(
            ["sysctl", "-n", f"hw.perflevel{index}.name"],
            capture_output=True, text=True, timeout=5).stdout.strip()
            for index in (0, 1)}
    except subprocess.TimeoutExpired:
        names = {}
    for index, name in names.items():
        count = level(f"perflevel{index}")
        if name == "Efficiency":
            efficiency = count
        else:
            performance = count
    return efficiency, performance


PROCESSOR_CPU_LOAD_INFO = 2


class CoreTicks:
    """Per-core busy percentages from host_processor_info deltas."""

    def __init__(self):
        self.previous = None

    def read(self):
        host = _libc.mach_host_self()
        count = ctypes.c_uint()
        info = ctypes.POINTER(ctypes.c_uint32)()
        info_count = ctypes.c_uint()
        if _libc.host_processor_info(
                host, PROCESSOR_CPU_LOAD_INFO, ctypes.byref(count),
                ctypes.byref(info), ctypes.byref(info_count)) != 0:
            return None
        ticks = [(info[i * 4] + info[i * 4 + 1] + info[i * 4 + 3],
                  info[i * 4 + 2]) for i in range(count.value)]
        task = ctypes.c_uint.in_dll(_libc, "mach_task_self_")
        _libc.vm_deallocate(task,
                            ctypes.cast(info, ctypes.c_void_p).value,
                            info_count.value * 4)
        return ticks

    def sample(self):
        current = self.read()
        if current is None:
            return None
        previous, self.previous = self.previous, current
        if previous is None or len(previous) != len(current):
            return None
        percentages = []
        for (busy_now, idle_now), (busy_then, idle_then) in zip(
                current, previous):
            busy = busy_now - busy_then
            total = busy + idle_now - idle_then
            percentages.append(
                round(100.0 * busy / total, 1) if total > 0 else 0.0)
        return percentages


class Sampler:
    """Attach to the target process (waiting if needed) and produce one
    sample dictionary per call."""

    def __init__(self, target):
        self.target = target
        self.pid = None
        self.name = None
        self.previous_cpu_ns = None
        self.previous_wall = None
        self.events = []
        self.cores = CoreTicks()
        self.efficiency_cores, self.performance_cores = cpu_topology()

    def sample(self):
        if self.pid is None:
            self.pid = find_pid(self.target)
            if self.pid is not None:
                self.name = process_name(self.pid)
                self.previous_cpu_ns = None
                self.events.append(f"attached to pid {self.pid} "
                                   f"({self.name})")
        cpu = rss = footprint = None
        if self.pid is not None:
            usage = process_usage(self.pid)
            if usage is None:
                self.events.append(f"pid {self.pid} exited; waiting")
                self.pid = None
                self.previous_cpu_ns = None
            else:
                cpu_ns, rss, footprint = usage
                wall = time.monotonic()
                if self.previous_cpu_ns is not None:
                    cpu = ((cpu_ns - self.previous_cpu_ns) /
                           ((wall - self.previous_wall) * 1e9) * 100.0)
                self.previous_cpu_ns = cpu_ns
                self.previous_wall = wall
        memory = system_memory()
        gpu = gpu_utilization()
        return {
            "t": time.time(),
            "pid": self.pid,
            "name": self.name if self.pid is not None else None,
            "cpu": cpu,
            "rss": rss,
            "foot": footprint,
            "gpu": gpu.get("device"),
            "gpu_renderer": gpu.get("renderer"),
            "gpu_tiler": gpu.get("tiler"),
            "cores": self.cores.sample(),
            "ecores": self.efficiency_cores,
            "pressure": memory_pressure(),
            **memory,
        }

    def drain_events(self):
        events, self.events = self.events, []
        return events


def format_log_line(sample):
    line = time.strftime("%H:%M:%S", time.localtime(sample["t"]))
    if sample["pid"] is not None:
        cpu = (f"{sample['cpu']:6.1f}%" if sample["cpu"] is not None
               else "     -")
        line += (f"  pid {sample['pid']}  cpu {cpu}  "
                 f"rss {sample['rss'] / 1e9:6.2f}G  "
                 f"foot {sample['foot'] / 1e9:6.2f}G")
    else:
        line += "  [waiting for process]"
    gpu = sample["gpu"]
    line += f" | gpu {gpu:3d}%" if gpu is not None else " | gpu  ?"
    if "free" in sample:
        line += (f" | free {sample['free']:5.2f}G"
                 f"  wired {sample['wired']:5.2f}G"
                 f"  filebk {sample['filebacked']:5.2f}G"
                 f"  anon {sample['anonymous']:5.2f}G"
                 f"  compr {sample['compressor']:5.2f}G"
                 f"  pressure {sample['pressure']}")
    return line


# ---------------------------------------------------------------------------
# Live terminal dashboard

BLOCKS = " ▁▂▃▄▅▆▇█"

DASHBOARD_ROWS = [
    ("cpu %", "cpu", 100.0, 1e-2),
    ("gpu %", "gpu", 100.0, 1.0),
    ("foot G", "foot", None, 1e-9),
    ("rss G", "rss", None, 1e-9),
    ("wired G", "wired", None, 1.0),
    ("filebk G", "filebacked", None, 1.0),
    ("anon G", "anonymous", None, 1.0),
    ("free G", "free", None, 1.0),
    ("compr G", "compressor", None, 1.0),
]


def sparkline(values, width, floor_maximum):
    values = values[-width:]
    present = [v for v in values if v is not None]
    maximum = max(present) if present else 0.0
    if floor_maximum is not None:
        maximum = max(maximum, floor_maximum)
    cells = []
    for value in values:
        if value is None:
            cells.append(" ")
        else:
            index = 0 if maximum <= 0 else min(
                8, max(0, int(round(value / maximum * 8))))
            cells.append(BLOCKS[index])
    return "".join(cells).rjust(width), maximum


def run_dashboard(sampler, arguments, record_file):
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 100
    width = max(20, min(90, columns - 34))
    history = {key: [] for _, key, _, _ in DASHBOARD_ROWS}
    count = 0
    sys.stdout.write("\x1b[2J\x1b[?25l")
    try:
        while True:
            sample = sampler.sample()
            if record_file is not None:
                record_file.write(json.dumps(sample) + "\n")
                record_file.flush()
            count += 1
            for _, key, _, scale in DASHBOARD_ROWS:
                value = sample.get(key)
                history[key].append(None if value is None
                                    else value * scale)
                del history[key][:-width]
            frame = ["\x1b[H"]
            attachment = (f"pid {sample['pid']} ({sample['name']})"
                          if sample["pid"] is not None
                          else "waiting for process")
            frame.append(
                f"qwen36 monitor  target '{sampler.target}'  "
                f"every {arguments.interval:g}s  {attachment}\x1b[K")
            frame.append("\x1b[K")
            for label, key, floor_maximum, _ in DASHBOARD_ROWS:
                line, maximum = sparkline(history[key], width,
                                          floor_maximum)
                current = history[key][-1]
                current_text = ("     -" if current is None
                                else f"{current:6.1f}")
                frame.append(f"{label:>8s}  {line}  {current_text}"
                             f"  peak {maximum:6.1f}\x1b[K")
            cores = sample.get("cores")
            if cores:
                efficiency = sample.get("ecores", 0)
                def meter(values):
                    return "".join(
                        BLOCKS[min(8, max(0, int(round(v / 100 * 8))))]
                        for v in values)
                frame.append(
                    f"{'cores':>8s}  E {meter(cores[:efficiency])}"
                    f"  P {meter(cores[efficiency:])}"
                    f"   (per-core now)\x1b[K")
            frame.append("\x1b[K")
            recording = (f"  recording {arguments.record}"
                         if arguments.record else "")
            frame.append(
                f"pressure {sample['pressure']:8s}  samples {count}"
                f"{recording}  Ctrl-C to stop\x1b[K")
            for event in sampler.drain_events():
                frame.append(f"{event}\x1b[K")
            sys.stdout.write("\n".join(frame) + "\n")
            sys.stdout.flush()
            time.sleep(arguments.interval)
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


def run_log(sampler, arguments, record_file):
    while True:
        sample = sampler.sample()
        if record_file is not None:
            record_file.write(json.dumps(sample) + "\n")
            record_file.flush()
        for event in sampler.drain_events():
            print(event, flush=True)
        print(format_log_line(sample), flush=True)
        time.sleep(arguments.interval)


# ---------------------------------------------------------------------------
# HTML chart rendering

CHART_WIDTH = 960
CHART_HEIGHT = 200
MARGIN_LEFT = 46
MARGIN_RIGHT = 12
MARGIN_TOP = 8
MARGIN_BOTTOM = 22


def nice_ceiling(value):
    if value <= 0:
        return 1.0
    from math import floor, log10
    magnitude = 10 ** floor(log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def polyline_segments(values, top, x_of, y_of):
    """Split a value list at None gaps into SVG point strings."""
    segments = []
    current = []
    for index, value in enumerate(values):
        if value is None:
            if len(current) > 1:
                segments.append(current)
            current = []
            continue
        current.append(f"{x_of(index):.1f},{y_of(value, top):.1f}")
    if len(current) > 1:
        segments.append(current)
    return segments


def render_panel(panel, seconds, pressure_spans=None):
    count = len(seconds)
    width = panel.get("width", CHART_WIDTH)
    height = panel.get("height", CHART_HEIGHT)
    tick_count = panel.get("ticks", 4)
    plot_width = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM
    peak = 0.0
    for _, _, values in panel["series"]:
        peak = max([peak] + [v for v in values if v is not None])
    top = max(nice_ceiling(peak), panel.get("floor_top", 0.0))

    def x_of(index):
        if count <= 1:
            return MARGIN_LEFT
        return MARGIN_LEFT + plot_width * index / (count - 1)

    def y_of(value, top_value):
        return (MARGIN_TOP + plot_height *
                (1.0 - min(value, top_value) / top_value))

    parts = [f'<figure class="panel" data-panel="{panel["key"]}">']
    legend = "".join(
        f'<span class="key"><i style="background:var({color})"></i>'
        f'{name}</span>'
        for name, color, _ in panel["series"]) if (
        len(panel["series"]) > 1) else ""
    extra = panel.get("legend_extra", "")
    parts.append(f'<figcaption><span class="ptitle">{panel["title"]}'
                 f'</span><span class="legend">{legend}{extra}</span>'
                 f'</figcaption>')
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" role="img" '
        f'aria-label="{panel["title"]}">')
    if pressure_spans:
        for start, end, level in pressure_spans:
            color = ("var(--status-warn)" if level == "warn"
                     else "var(--status-critical)")
            parts.append(
                f'<rect x="{x_of(start):.1f}" y="{MARGIN_TOP}" '
                f'width="{x_of(end) - x_of(start):.1f}" '
                f'height="{plot_height}" fill="{color}" '
                f'opacity="0.13"/>')
    for tick in range(tick_count + 1):
        value = top * tick / tick_count
        y = y_of(value, top)
        parts.append(f'<line class="grid" x1="{MARGIN_LEFT}" '
                     f'x2="{width - MARGIN_RIGHT}" '
                     f'y1="{y:.1f}" y2="{y:.1f}"/>')
        text = f"{value:g}"
        parts.append(f'<text class="tick" x="{MARGIN_LEFT - 6}" '
                     f'y="{y + 3.5:.1f}" text-anchor="end">{text}</text>')
    total = seconds[-1] if seconds else 0
    fractions = ((0.0, 1.0) if panel.get("mini")
                 else (0.0, 0.25, 0.5, 0.75, 1.0))
    for fraction in fractions:
        index = int(fraction * (count - 1)) if count > 1 else 0
        moment = seconds[index] if seconds else 0
        label = f"{int(moment // 60)}:{int(moment % 60):02d}"
        anchor = ("start" if fraction == 0.0 else
                  "end" if fraction == 1.0 else "middle")
        parts.append(
            f'<text class="tick" x="{x_of(index):.1f}" '
            f'y="{height - 6}" text-anchor="{anchor}">{label}</text>')
    labels = []
    for name, color, values in panel["series"]:
        for points in polyline_segments(values, top, x_of, y_of):
            parts.append(f'<polyline class="line" points='
                         f'"{" ".join(points)}" '
                         f'stroke="var({color})"/>')
        last_index = max((i for i, v in enumerate(values)
                          if v is not None), default=None)
        if last_index is not None:
            x = x_of(last_index)
            y = y_of(values[last_index], top)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                         f'fill="var({color})" class="enddot"/>')
            labels.append([x, y - 5, name])
    # Push colliding end labels apart vertically so every name stays legible.
    labels.sort(key=lambda item: item[1])
    for index in range(1, len(labels)):
        if labels[index][1] - labels[index - 1][1] < 13:
            labels[index][1] = labels[index - 1][1] + 13
    bottom = MARGIN_TOP + plot_height - 2
    if labels and not panel.get("mini"):
        # Keep the stack inside the plot: clamp the lowest label to the
        # baseline, then walk upward so no pair collapses back together.
        labels[-1][1] = min(labels[-1][1], bottom)
        for index in range(len(labels) - 2, -1, -1):
            labels[index][1] = min(labels[index][1],
                                   labels[index + 1][1] - 13)
        for x, y, name in labels:
            anchor = "end" if x > width - 90 else "start"
            offset = -7 if anchor == "end" else 7
            parts.append(f'<text class="dlabel" x="{x + offset:.1f}" '
                         f'y="{y:.1f}" text-anchor="{anchor}">{name}</text>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{MARGIN_TOP}" '
                 f'y2="{MARGIN_TOP + plot_height}" opacity="0"/>')
    parts.append("</svg></figure>")
    parts.append(f"<!--total {total:.0f}s-->")
    return "\n".join(parts)


def render_html(samples, target, output_path):
    seconds = [s["t"] - samples[0]["t"] for s in samples]

    def series(key, scale):
        return [None if s.get(key) is None else s[key] * scale
                for s in samples]

    pressure_spans = []
    open_span = None
    for index, sample in enumerate(samples):
        level = sample.get("pressure", "normal")
        if level in ("warn", "critical"):
            if open_span is None or open_span[2] != level:
                if open_span is not None:
                    pressure_spans.append(
                        (open_span[0], index, open_span[2]))
                open_span = (index, index, level)
        elif open_span is not None:
            pressure_spans.append((open_span[0], index, open_span[2]))
            open_span = None
    if open_span is not None:
        pressure_spans.append((open_span[0], len(samples) - 1,
                               open_span[2]))

    efficiency = next((s.get("ecores", 0) for s in samples
                       if s.get("ecores")), 0)
    core_count = max((len(s["cores"]) for s in samples
                      if s.get("cores")), default=0)

    def core_series(index):
        return [None if not s.get("cores") or index >= len(s["cores"])
                else s["cores"][index] for s in samples]

    core_panels = []
    for index in range(core_count):
        is_efficiency = index < efficiency
        label = (f"E{index + 1}" if is_efficiency
                 else f"P{index - efficiency + 1}")
        kind = "efficiency" if is_efficiency else "performance"
        core_panels.append({
            "key": f"core{index}", "mini": True, "width": 470,
            "height": 96, "ticks": 2, "floor_top": 100.0,
            "title": f"{label} · {kind} core",
            "series": [(label,
                        "--s2" if is_efficiency else "--s1",
                        core_series(index))]})

    panels = [
        {"key": "util", "title": "Utilization %", "floor_top": 100.0,
         "series": [("process cpu", "--s1", series("cpu", 1.0)),
                    ("gpu device", "--s2", series("gpu", 1.0))]},
        {"key": "gpu", "title":
            "GPU engines % (14-core device; macOS exposes no per-core "
            "GPU counters)", "floor_top": 100.0,
         "series": [("device", "--s1", series("gpu", 1.0)),
                    ("renderer", "--s2", series("gpu_renderer", 1.0)),
                    ("tiler", "--s3", series("gpu_tiler", 1.0))]},
        {"key": "proc", "title": "Process memory GB",
         "series": [("footprint", "--s1", series("foot", 1e-9)),
                    ("rss", "--s2", series("rss", 1e-9))]},
        {"key": "system", "title": "System memory GB",
         "legend_extra":
             ('<span class="key"><i class="warnchip"></i>'
              'pressure warn/critical</span>' if pressure_spans else ""),
         "series": [("wired", "--s1", series("wired", 1.0)),
                    ("file-backed", "--s2", series("filebacked", 1.0)),
                    ("anonymous", "--s3", series("anonymous", 1.0)),
                    ("free", "--s4", series("free", 1.0)),
                    ("compressed", "--s5", series("compressor", 1.0))]},
    ]

    payload = {
        "seconds": [round(v, 2) for v in seconds],
        "panels": {
            panel["key"]: {
                "unit": "GB" if panel["key"] in ("proc", "system")
                        else "%",
                "left": MARGIN_LEFT,
                "width": panel.get("width", CHART_WIDTH),
                "right": MARGIN_RIGHT,
                "series": [(name, color, [
                    None if v is None else round(v, 2) for v in values])
                    for name, color, values in panel["series"]],
            } for panel in panels + core_panels
        },
    }

    started = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime(samples[0]["t"]))
    duration = seconds[-1] if seconds else 0
    names = sorted({s["name"] for s in samples if s.get("name")})
    rendered = {panel["key"]: render_panel(
        panel, seconds,
        pressure_spans if panel["key"] == "system" else None)
        for panel in panels}
    core_grid = ""
    if core_panels:
        minis = "\n".join(render_panel(panel, seconds)
                          for panel in core_panels)
        core_grid = (
            '<section class="coresec">'
            '<div class="corehead"><span class="ptitle">CPU cores %'
            f'</span><span class="legend">'
            f'<span class="key"><i style="background:var(--s1)"></i>'
            f'performance ({core_count - efficiency})</span>'
            f'<span class="key"><i style="background:var(--s2)"></i>'
            f'efficiency ({efficiency})</span></span></div>'
            f'<div class="grid">{minis}</div></section>')
    body = "\n".join([rendered["util"], core_grid, rendered["gpu"],
                      rendered["proc"], rendered["system"]])

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qwen36 monitor — {target}</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4;
  --status-warn: #fab219; --status-critical: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 28px 20px 48px; }}
h1 {{ font-size: 19px; margin: 0 0 2px; }}
.meta {{ color: var(--ink-2); margin: 0 0 18px; font-size: 13px; }}
.panel {{ margin: 0 0 18px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px 8px; }}
.coresec {{ margin: 0 0 18px; }}
.corehead {{ display: flex; justify-content: space-between;
  align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 8px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.grid .panel {{ margin: 0; padding: 8px 10px 4px; }}
.grid .ptitle {{ font-size: 12px; font-weight: 600;
  color: var(--ink-2); }}
@media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
figcaption {{ display: flex; justify-content: space-between;
  align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 6px; }}
.ptitle {{ font-weight: 600; font-size: 13px; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap;
  font-size: 12px; color: var(--ink-2); }}
.key i {{ display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 5px; vertical-align: -1px; }}
.key .warnchip {{ background: var(--status-warn); opacity: 0.55; }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 10px;
  font-variant-numeric: tabular-nums;
  font-family: system-ui, sans-serif; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round;
  stroke-linecap: round; vector-effect: non-scaling-stroke; }}
.dlabel {{ fill: var(--ink-2); font-size: 11px;
  font-family: system-ui, sans-serif; paint-order: stroke;
  stroke: var(--surface); stroke-width: 3; }}
.cross {{ stroke: var(--muted); stroke-width: 1;
  stroke-dasharray: 3 3; }}
.tip {{ position: fixed; pointer-events: none; z-index: 10;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,0.18);
  padding: 7px 10px; font-size: 12px; display: none;
  color: var(--ink); }}
.tip b {{ font-variant-numeric: tabular-nums; }}
.tip .row {{ display: flex; gap: 8px; align-items: center;
  white-space: nowrap; }}
.tip i {{ display: inline-block; width: 8px; height: 8px;
  border-radius: 2px; }}
.tip .when {{ color: var(--ink-2); margin-bottom: 3px; }}
</style>
</head>
<body>
<main>
<h1>qwen36 resource monitor</h1>
<p class="meta">target '{target}' &middot; {started} &middot;
{duration:.0f}s &middot; {len(samples)} samples &middot;
processes: {", ".join(names) if names else "none attached"}</p>
{body}
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {json.dumps(payload)};
const tip = document.getElementById("tip");
for (const figure of document.querySelectorAll(".panel")) {{
  const svg = figure.querySelector("svg");
  const cross = figure.querySelector(".cross");
  const panel = DATA.panels[figure.dataset.panel];
  const count = DATA.seconds.length;
  svg.addEventListener("mousemove", (event) => {{
    const box = svg.getBoundingClientRect();
    const viewX = (event.clientX - box.left) / box.width * panel.width;
    const plot = panel.width - panel.left - panel.right;
    let index = Math.round((viewX - panel.left) / plot * (count - 1));
    index = Math.max(0, Math.min(count - 1, index));
    const x = panel.left + plot * index / (count - 1 || 1);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x);
    cross.setAttribute("opacity", 1);
    const moment = DATA.seconds[index];
    const clock = `${{Math.floor(moment / 60)}}:` +
      String(Math.floor(moment % 60)).padStart(2, "0");
    let rows = `<div class="when">t+${{clock}}</div>`;
    for (const [name, color, values] of panel.series) {{
      const value = values[index];
      rows += `<div class="row"><i style="background:var(${{color}})">` +
        `</i>${{name}} <b>${{value === null ? "–" : value}}` +
        `${{panel.unit}}</b></div>`;
    }}
    tip.innerHTML = rows;
    tip.style.display = "block";
    tip.style.left = Math.min(event.clientX + 14,
      window.innerWidth - 170) + "px";
    tip.style.top = (event.clientY + 14) + "px";
  }});
  svg.addEventListener("mouseleave", () => {{
    cross.setAttribute("opacity", 0);
    tip.style.display = "none";
  }});
}}
</script>
</body>
</html>
"""
    with open(output_path, "w") as handle:
        handle.write(html)
    return output_path


def render_command(arguments):
    samples = []
    with open(arguments.render) as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        print("no samples in recording", file=sys.stderr)
        return 1
    output = arguments.output or (
        os.path.splitext(arguments.render)[0] + ".html")
    render_html(samples, arguments.target, output)
    print(f"wrote {output} ({len(samples)} samples, "
          f"{samples[-1]['t'] - samples[0]['t']:.0f}s)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Live CPU/memory/GPU monitor for local model runs.")
    parser.add_argument("target", nargs="?", default="qwen36",
                        help="process name substring or pid "
                             "(default: qwen36)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="sampling interval in seconds (default: 1)")
    parser.add_argument("--log", action="store_true",
                        help="plain line output instead of the live "
                             "dashboard")
    parser.add_argument("--record", metavar="FILE",
                        help="append one JSON sample per interval to FILE")
    parser.add_argument("--render", metavar="FILE",
                        help="render a recording to an HTML chart page "
                             "and exit")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="output path for --render "
                             "(default: recording name .html)")
    arguments = parser.parse_args()

    if arguments.render:
        return render_command(arguments)

    record_file = open(arguments.record, "a") if arguments.record else None
    sampler = Sampler(arguments.target)
    try:
        if arguments.log or not sys.stdout.isatty():
            run_log(sampler, arguments, record_file)
        else:
            run_dashboard(sampler, arguments, record_file)
    except KeyboardInterrupt:
        return 0
    finally:
        if record_file is not None:
            record_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
