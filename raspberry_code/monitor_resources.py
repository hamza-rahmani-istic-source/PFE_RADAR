#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor Raspberry Pi resource usage for matching processes.")
    p.add_argument(
        "--match",
        action="append",
        required=True,
        help="Substring to match in process command line. Repeat for multiple processes.",
    )
    p.add_argument("--output-csv", default="monitor_resources.csv")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=0.0, help="0 means run until Ctrl+C.")
    p.add_argument("--disk-path", default="/home/pi/ai_camera")
    return p.parse_args()


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def read_meminfo() -> Dict[str, int]:
    out: Dict[str, int] = {}
    raw = read_text("/proc/meminfo") or ""
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if not parts:
            continue
        try:
            out[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return out


def read_temperature_c() -> Optional[float]:
    raw = read_text("/sys/class/thermal/thermal_zone0/temp")
    if raw:
        try:
            return int(raw.strip()) / 1000.0
        except ValueError:
            return None
    return None


def list_matches(patterns: List[str]) -> List[int]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return []

    pids: List[int] = []
    lowered = [p.lower() for p in patterns]
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1].lower()
        if any(p in cmd for p in lowered):
            pids.append(pid)
    return sorted(set(pids))


def rss_bytes(pid: int) -> int:
    raw = read_text(f"/proc/{pid}/statm")
    if not raw:
        return 0
    parts = raw.split()
    if len(parts) < 2:
        return 0
    try:
        return int(parts[1]) * PAGE_SIZE
    except ValueError:
        return 0


def cpu_percent(pid: int) -> float:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return float(proc.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def total_rss(pids: Iterable[int]) -> int:
    return sum(rss_bytes(pid) for pid in pids)


def total_cpu(pids: Iterable[int]) -> float:
    return round(sum(cpu_percent(pid) for pid in pids), 2)


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    disk_path = Path(args.disk_path)
    start = time.time()

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "elapsed_s",
                "pid_count",
                "proc_cpu_percent",
                "proc_rss_mib",
                "system_mem_used_mib",
                "system_mem_available_mib",
                "disk_used_mib",
                "disk_free_mib",
                "temperature_c",
            ]
        )
        try:
            while True:
                elapsed = time.time() - start
                if args.duration > 0 and elapsed >= args.duration:
                    break

                pids = list_matches(args.match)
                meminfo = read_meminfo()
                temp = read_temperature_c()
                usage = shutil.disk_usage(disk_path)
                mem_total = meminfo.get("MemTotal", 0)
                mem_available = meminfo.get("MemAvailable", 0)
                mem_used = max(0, mem_total - mem_available)

                writer.writerow(
                    [
                        round(elapsed, 2),
                        len(pids),
                        total_cpu(pids),
                        mib(total_rss(pids)),
                        mib(mem_used),
                        mib(mem_available),
                        mib(usage.used),
                        mib(usage.free),
                        "" if temp is None else round(temp, 2),
                    ]
                )
                fh.flush()
                time.sleep(max(0.2, args.interval))
        except KeyboardInterrupt:
            pass

    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
