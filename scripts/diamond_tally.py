"""Per-brain diamond / tier-progression tally for goal=diamond waves.

Scans each agent rollout's step records (inventory = {item_id: count}) and the
summary record, reporting per result-dir:
  - n rollouts, deaths
  - diamonds: rollouts that ever held minecraft:diamond (raw) or a diamond tool
  - peak pickaxe tier histogram (none/wooden/stone/iron/diamond)
  - mine_diamond attempts vs successes
  - peak iron-ingot count (proxy for how deep the iron economy got)

Usage:
    python -m scripts.diamond_tally results/bigN20-easy-haiku-<ts> results/bigN20-easy-qwen-<ts>
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter

TIERS = ["none", "wooden", "stone", "iron", "diamond"]
PICK = {f"minecraft:{t}_pickaxe": i for i, t in enumerate(["wooden", "stone", "iron", "diamond"], start=1)}


def _peak_tier(invs: list[dict]) -> int:
    peak = 0
    for inv in invs:
        for item in inv:
            if item in PICK:
                peak = max(peak, PICK[item])
    return peak


def _has_diamond(invs: list[dict]) -> bool:
    for inv in invs:
        for item, n in inv.items():
            if item == "minecraft:diamond" and n > 0:
                return True
            if "diamond" in item and ("pickaxe" in item or "sword" in item
                                       or "shovel" in item or "axe" in item):
                return True  # crafted a diamond tool → had diamonds
    return False


def analyze(result_dir: str) -> dict:
    rollouts = []
    for f in glob.glob(f"{result_dir}/agent*.jsonl"):
        invs, died, mine_d_try, mine_d_ok, peak_iron = [], False, 0, 0, 0
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("inventory"):
                invs.append(r["inventory"])
                peak_iron = max(peak_iron, r["inventory"].get("minecraft:iron_ingot", 0))
            if r.get("died") or r.get("rollout_had_death"):
                died = True
            if r.get("tool") == "mine_diamond":
                mine_d_try += 1
                if not (r.get("outcome", "") or "").startswith("FAILED"):
                    mine_d_ok += 1
        if not invs:
            continue
        rollouts.append({
            "file": f.split("/")[-1],
            "diamond": _has_diamond(invs),
            "peak_tier": _peak_tier(invs),
            "died": died,
            "mine_d_try": mine_d_try,
            "mine_d_ok": mine_d_ok,
            "peak_iron": peak_iron,
        })

    n = len(rollouts)
    diamonds = sum(r["diamond"] for r in rollouts)
    deaths = sum(r["died"] for r in rollouts)
    tier_hist = Counter(TIERS[r["peak_tier"]] for r in rollouts)
    md_try = sum(r["mine_d_try"] for r in rollouts)
    md_ok = sum(r["mine_d_ok"] for r in rollouts)

    print(f"\n=== {result_dir} ===")
    print(f"  rollouts={n}  deaths={deaths}  DIAMONDS={diamonds}/{n} ({100*diamonds/max(n,1):.0f}%)")
    print(f"  peak pickaxe tier: " + "  ".join(f"{t}={tier_hist.get(t,0)}" for t in TIERS))
    print(f"  mine_diamond: {md_ok}/{md_try} success/attempts")
    iron_reached = sum(1 for r in rollouts if r["peak_tier"] >= 3)
    print(f"  reached iron-pickaxe tier: {iron_reached}/{n}")
    diamond_files = [r["file"] for r in rollouts if r["diamond"]]
    if diamond_files:
        print(f"  diamond rollouts: {diamond_files}")
    return {"n": n, "diamonds": diamonds, "deaths": deaths, "tier_hist": dict(tier_hist)}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for d in argv:
        analyze(d.rstrip("/"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
