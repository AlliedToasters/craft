"""Summarize results/load_concurrent.jsonl."""

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/load_concurrent.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"samples: {len(rows)}  duration: {rows[-1]['dt']:.1f}s")
    print()

    # System
    mem_used = [r["system"]["mem_mb"]["used"] for r in rows]
    mem_avail = [r["system"]["mem_mb"]["available"] for r in rows]
    load1 = [r["system"]["load1"] for r in rows]
    swap = [r["system"]["swap_mb"]["used"] for r in rows]
    print("System memory (MB):")
    print(f"  used      min={min(mem_used)} median={int(statistics.median(mem_used))} max={max(mem_used)}")
    print(f"  available min={min(mem_avail)} median={int(statistics.median(mem_avail))} max={max(mem_avail)}")
    print(f"  swap_used max={max(swap)}")
    print(f"Load1: min={min(load1):.2f} median={statistics.median(load1):.2f} max={max(load1):.2f}")
    print()

    # Per-agent
    by_port: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        for j in r["java"]:
            if j["port"] is None:
                continue
            d = by_port.setdefault(j["port"], {"rss": [], "pcpu": []})
            d["rss"].append(j["rss_mb"])
            d["pcpu"].append(j["pcpu"])
    print("Per-agent (RSS MB, CPU %):")
    print(f"  {'port':>6}  {'rss min':>8} {'rss med':>8} {'rss max':>8}   {'cpu min':>8} {'cpu med':>8} {'cpu max':>8}")
    total_rss_max = 0
    total_cpu_max = 0.0
    for port in sorted(by_port):
        d = by_port[port]
        rss_max = max(d["rss"])
        cpu_max = max(d["pcpu"])
        total_rss_max += rss_max
        total_cpu_max += cpu_max
        print(f"  {port:>6}  {min(d['rss']):>8.1f} {statistics.median(d['rss']):>8.1f} {rss_max:>8.1f}   {min(d['pcpu']):>8.1f} {statistics.median(d['pcpu']):>8.1f} {cpu_max:>8.1f}")
    print()
    print(f"Sum-of-per-agent-peak RSS: {total_rss_max:.0f} MB ({total_rss_max/1024:.2f} GB)")
    print(f"Sum-of-per-agent-peak CPU: {total_cpu_max:.1f}% (of 800% on 8 cores)")
    print()

    # Find peak sample (max load1)
    peak = max(rows, key=lambda r: r["system"]["load1"])
    print(f"Peak load sample @ dt={peak['dt']}s: load1={peak['system']['load1']}")
    print(f"  mem used={peak['system']['mem_mb']['used']} avail={peak['system']['mem_mb']['available']} swap={peak['system']['swap_mb']['used']}")
    for j in peak["java"]:
        if j["port"]:
            print(f"  port {j['port']}: rss={j['rss_mb']} MB, cpu={j['pcpu']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
