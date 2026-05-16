"""Sample MC server TPS/MSPT/CPU while another workload runs.

Every INTERVAL seconds:
  1. POST /cmd {"cmd":"spark tps"}
     (Purpur prints a ⚡ block: TPS + tick durations + CPU usage)
  2. sleep briefly so the server flushes to console
  3. GET /log?n=40, find the most recent ⚡ block and parse it
  4. write one JSONL line per sample, deduped by [HH:MM:SS] timestamp

Run before kicking off the suite; SIGTERM to stop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import os

import requests

SERVER = os.environ.get("MC_SERVER_CMD_BASE", "http://127.0.0.1:4747")

TS_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\]")
TPS_VALS = re.compile(r"\*?(\d+\.\d+),\s*\*?(\d+\.\d+),\s*\*?(\d+\.\d+),\s*\*?(\d+\.\d+),\s*\*?(\d+\.\d+)")
MSPT_VALS = re.compile(
    r"(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+);\s+(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
)
CPU_VALS = re.compile(r"(\d+)%,\s*(\d+)%,\s*(\d+)%\s*\((system|process)\)")
LIST_RE = re.compile(r"There are (\d+) of a max of (\d+) players online")


def _fire(cmd: str) -> None:
    requests.post(f"{SERVER}/cmd", json={"cmd": cmd}, timeout=3.0)


def _fetch_log(n: int = 40) -> list[str]:
    return requests.get(f"{SERVER}/log", params={"n": n}, timeout=3.0).json().get(
        "lines", []
    )


def _parse_latest_block(lines: list[str]) -> dict:
    """Find the newest ⚡ TPS-block (by line-position) and parse it.

    Spark/Purpur output:
      [HH:MM:SS] ... [⚡] TPS from last 5s, 10s, 1m, 5m, 15m:
      [HH:MM:SS] ... [⚡]  *20.0, *20.0, 20.0, *20.0, *20.0
      [HH:MM:SS] ... [⚡]
      [HH:MM:SS] ... [⚡] Tick durations (min/med/95%ile/max ms) from last 10s, 1m:
      [HH:MM:SS] ... [⚡]  0.7/1.8/2.0/2.3;  0.6/1.8/2.1/9.7
      [HH:MM:SS] ... [⚡]
      [HH:MM:SS] ... [⚡] CPU usage from last 10s, 1m, 15m:
      [HH:MM:SS] ... [⚡]  0%, 0%, 0%  (system)
      [HH:MM:SS] ... [⚡]  0%, 0%, 0%  (process)
    """
    # Find the most recent line containing "TPS from last 5s, 10s,"
    # — that's the unambiguous header for the spark/purpur extended block.
    header_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "TPS from last 5s, 10s, 1m, 5m, 15m" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        return {}

    # Block is roughly the next ~10 lines
    block = lines[header_idx : header_idx + 11]
    out: dict = {}
    ts_m = TS_RE.search(lines[header_idx])
    if ts_m:
        out["server_ts"] = ts_m.group(1)
    for line in block:
        if "tps" not in out:
            m = TPS_VALS.search(line)
            if m and "TPS from" not in line:
                out["tps"] = {
                    "5s": float(m.group(1)),
                    "10s": float(m.group(2)),
                    "1m": float(m.group(3)),
                    "5m": float(m.group(4)),
                    "15m": float(m.group(5)),
                }
        if "mspt" not in out:
            m = MSPT_VALS.search(line)
            if m:
                out["mspt"] = {
                    "last_10s": {
                        "min": float(m.group(1)),
                        "med": float(m.group(2)),
                        "p95": float(m.group(3)),
                        "max": float(m.group(4)),
                    },
                    "last_1m": {
                        "min": float(m.group(5)),
                        "med": float(m.group(6)),
                        "p95": float(m.group(7)),
                        "max": float(m.group(8)),
                    },
                }
        m = CPU_VALS.search(line)
        if m:
            key = "cpu_sys" if m.group(4) == "system" else "cpu_proc"
            if key not in out:
                out[key] = {
                    "10s": int(m.group(1)),
                    "1m": int(m.group(2)),
                    "15m": int(m.group(3)),
                }

    # players: walk full log backward for the freshest list-line
    for line in reversed(lines):
        m = LIST_RE.search(line)
        if m:
            out["players"] = int(m.group(1))
            out["max_players"] = int(m.group(2))
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last_server_ts: str | None = None
    with path.open("w") as f:
        i = 0
        while True:
            try:
                _fire("spark tps")
                if i % 6 == 0:
                    _fire("list")
                time.sleep(0.6)
                lines = _fetch_log(n=40)
                parsed = _parse_latest_block(lines)
                is_fresh = (
                    parsed.get("server_ts") is not None
                    and parsed.get("server_ts") != last_server_ts
                )
                rec = {
                    "t": round(time.time(), 2),
                    "dt": round(time.time() - t0, 2),
                    "fresh": is_fresh,
                    **parsed,
                }
                if is_fresh:
                    last_server_ts = parsed["server_ts"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
            except Exception as e:
                f.write(json.dumps({"t": time.time(), "error": repr(e)}) + "\n")
                f.flush()
            i += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
