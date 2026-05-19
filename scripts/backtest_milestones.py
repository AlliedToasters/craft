#!/usr/bin/env python3
"""Replay rolling-rollout JSONLs through the milestone framework and report
when each milestone would have fired.

Usage: backtest_milestones.py [results-dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from craft.milestones import Milestones


def replay(path: Path) -> dict:
    ms = Milestones()
    fires = {}
    final_turn = 0
    died = False
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("_type") != "turn":
                continue
            t = d.get("turn", 0)
            final_turn = max(final_turn, t)
            if d.get("died"):
                died = True
            stats = {k: d.get(k) for k in ("day_ticks", "day_count", "health", "food")}
            inv = d.get("inventory") or {}
            ev = ms.check(stats, inv, t)
            if ev:
                fires[ev.name] = ev.turn
    return {"name": path.stem, "final_turn": final_turn, "died": died, "fires": fires}


def bucket_label(t: int) -> str:
    if t < 20:
        return "A_dead_early"
    if t < 50:
        return "B_short"
    if t < 100:
        return "C_medium"
    if t < 200:
        return "D_long"
    return "E_hero"


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/rolling-20260517")
    files = sorted(base.glob("agent*.jsonl"))
    rows = []
    for fp in files:
        r = replay(fp)
        if r["final_turn"] > 0:
            rows.append(r)
    print(f"# {len(rows)} rollouts replayed")

    # M1 fire counts by survival bucket
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[bucket_label(r["final_turn"])].append(r)

    print("\n=== M1_iron_goal fire rate by survival bucket ===")
    print(f"  {'bucket':16s} {'n':>4s} {'fired':>5s}  {'pct':>5s}  avg_fire_turn  median_fire_turn")
    for k in sorted(by_bucket):
        rs = by_bucket[k]
        fired = [r["fires"]["M1_iron_goal"] for r in rs if "M1_iron_goal" in r["fires"]]
        pct = (100 * len(fired) / len(rs)) if rs else 0
        avg = sum(fired) / len(fired) if fired else 0
        med = sorted(fired)[len(fired) // 2] if fired else 0
        print(f"  {k:16s} {len(rs):4d} {len(fired):5d}  {pct:4.1f}%  {avg:13.1f}  {med:7d}")

    print("\n=== When did M1 fire? (all rollouts) ===")
    fired_all = [r["fires"]["M1_iron_goal"] for r in rows if "M1_iron_goal" in r["fires"]]
    print(f"  total fires: {len(fired_all)} / {len(rows)} ({100*len(fired_all)/len(rows):.1f}%)")
    if fired_all:
        print(f"  min: {min(fired_all)}, median: {sorted(fired_all)[len(fired_all)//2]}, max: {max(fired_all)}")
        # Histogram by 10-turn buckets
        hist = defaultdict(int)
        for f in fired_all:
            hist[(f // 10) * 10] += 1
        for k in sorted(hist):
            bar = "█" * hist[k]
            print(f"  T{k:3d}-{k+9:3d}: {hist[k]:3d} {bar}")

    print("\n=== Among rollouts where M1 fired: post-fire survival ===")
    fired_rows = [r for r in rows if "M1_iron_goal" in r["fires"]]
    deltas = [r["final_turn"] - r["fires"]["M1_iron_goal"] for r in fired_rows]
    if deltas:
        print(f"  n={len(deltas)}")
        print(f"  mean turns survived AFTER M1: {sum(deltas)/len(deltas):.1f}")
        print(f"  median: {sorted(deltas)[len(deltas)//2]}")
        died_after_fire = sum(1 for r in fired_rows if r["died"])
        print(f"  died after M1: {died_after_fire}/{len(fired_rows)} ({100*died_after_fire/len(fired_rows):.0f}%)")


if __name__ == "__main__":
    main()
