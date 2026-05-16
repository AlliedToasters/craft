"""Summarize server-side TPS/MSPT/CPU sampler output."""

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/server_load_concurrent.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "tps" in r and r.get("fresh") is not False]
    if not rows:
        print("no fresh samples")
        return 1
    print(f"fresh samples: {len(rows)}  duration: {rows[-1]['dt']:.1f}s")
    print()

    tps_5s = [r["tps"]["5s"] for r in rows]
    tps_1m = [r["tps"]["1m"] for r in rows]
    mspt_med = [r["mspt"]["last_10s"]["med"] for r in rows]
    mspt_p95 = [r["mspt"]["last_10s"]["p95"] for r in rows]
    mspt_max = [r["mspt"]["last_10s"]["max"] for r in rows]
    cpu_proc = [r.get("cpu_proc", {}).get("10s", 0) for r in rows]
    cpu_sys = [r.get("cpu_sys", {}).get("10s", 0) for r in rows]

    def stats(name: str, xs: list, unit: str = "") -> None:
        print(f"  {name:<28}  min={min(xs):.2f}{unit}  median={statistics.median(xs):.2f}{unit}  max={max(xs):.2f}{unit}")

    print("TPS:")
    stats("tps (5s window)", tps_5s)
    stats("tps (1m window)", tps_1m)
    print()
    print("MSPT (Paper, last 10s) — 50ms budget per tick:")
    stats("med ms/tick", mspt_med, " ms")
    stats("p95 ms/tick", mspt_p95, " ms")
    stats("max ms/tick", mspt_max, " ms")
    print()
    print("CPU (Paper reports % across all cores combined):")
    stats("process CPU (10s)", cpu_proc, "%")
    stats("system CPU (10s)", cpu_sys, "%")
    print()

    # Saturation hint
    over_25 = sum(1 for m in mspt_max if m > 25)
    over_50 = sum(1 for m in mspt_max if m > 50)
    print(f"Tick-max excursions: {over_25}/{len(mspt_max)} samples >25ms, {over_50} >50ms (lag)")

    # Peak sample
    peak = max(rows, key=lambda r: r["mspt"]["last_10s"]["max"])
    print(f"\nPeak mspt sample @ dt={peak['dt']}s (server {peak.get('server_ts')}):")
    print(f"  tps_5s={peak['tps']['5s']} mspt_med={peak['mspt']['last_10s']['med']} max={peak['mspt']['last_10s']['max']}")
    print(f"  cpu_proc={peak.get('cpu_proc',{})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
