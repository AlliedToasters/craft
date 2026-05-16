"""Sample system + per-agent load while another command runs.

Polls every INTERVAL seconds and writes one JSONL line per sample with:
  - timestamp (epoch + relative)
  - system: total/used/available/buff_cache MB, swap_used MB, load1
  - per-java: pid, port (from -Dhomunculus.port=...), rss_mb, pcpu

Usage: python -m scripts.sample_load --out results/load.jsonl --interval 2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PORT_RE = re.compile(r"-Dhomunculus\.port=(\d+)")


def _system_snapshot() -> dict:
    free = subprocess.run(
        ["free", "-m"], capture_output=True, text=True, check=True
    ).stdout
    lines = free.strip().splitlines()
    mem_parts = lines[1].split()
    swap_parts = lines[2].split()
    mem = {
        "total": int(mem_parts[1]),
        "used": int(mem_parts[2]),
        "free": int(mem_parts[3]),
        "shared": int(mem_parts[4]),
        "buff_cache": int(mem_parts[5]),
        "available": int(mem_parts[6]),
    }
    swap = {"total": int(swap_parts[1]), "used": int(swap_parts[2])}
    with open("/proc/loadavg") as f:
        load1 = float(f.read().split()[0])
    return {"mem_mb": mem, "swap_mb": swap, "load1": load1}


def _java_snapshot() -> list[dict]:
    out = subprocess.run(
        ["ps", "-eo", "pid,rss,pcpu,cmd"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows: list[dict] = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, rss_kb, pcpu, cmd = parts
        if "PrismLauncher" not in cmd and "homunculus.port" not in cmd:
            continue
        m = PORT_RE.search(cmd)
        port = int(m.group(1)) if m else None
        try:
            rows.append({
                "pid": int(pid),
                "port": port,
                "rss_mb": round(int(rss_kb) / 1024, 1),
                "pcpu": float(pcpu),
            })
        except ValueError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="0 = until SIGTERM")
    args = ap.parse_args()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + args.duration if args.duration > 0 else None

    with path.open("w") as f:
        while True:
            now = time.time()
            rec = {
                "t": round(now, 2),
                "dt": round(now - t0, 2),
                "system": _system_snapshot(),
                "java": _java_snapshot(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if deadline is not None and now >= deadline:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
