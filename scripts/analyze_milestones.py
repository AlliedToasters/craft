#!/usr/bin/env python3
"""Mine rolling-rollout JSONLs for early indicators of long-term survival.

For each rollout, extract:
  - final_turns (death turn or last turn alive)
  - tier at T=N for N in {10, 15, 20, 30}
  - food/HP at T=N
  - first turn shelter/wall_in was used
  - first turn each tool was successfully used
  - did the agent survive past first dusk (day_ticks≥12000 before death)

Then bucket by final survival and compute predictor means per bucket.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Tier ordering
TIER_RANK = {"none": 0, "wood": 1, "stone": 2, "iron": 3, "diamond": 4}


def parse_inventory(inv_dict):
    """inv_dict is {item_id: count} — return set of item_ids with count>=1."""
    items = set()
    if not inv_dict:
        return items
    if isinstance(inv_dict, dict):
        for k, v in inv_dict.items():
            if isinstance(v, int) and v >= 1:
                items.add(k)
            elif isinstance(v, dict):
                items.add(v.get("item", k))
            else:
                items.add(k)
    return items


def tier_of(items: set) -> str:
    if any("diamond_pickaxe" in i or "diamond_sword" in i for i in items):
        return "diamond"
    if any("iron_pickaxe" in i or "iron_sword" in i for i in items):
        return "iron"
    if any("stone_pickaxe" in i or "stone_sword" in i for i in items):
        return "stone"
    if any("wooden_pickaxe" in i or "wooden_sword" in i for i in items):
        return "wood"
    return "none"


def has_food(items: set) -> bool:
    foods = ("mutton", "beef", "porkchop", "chicken", "rabbit", "bread", "apple",
             "cooked_", "sweet_berries", "wheat", "carrot", "potato")
    return any(any(f in i for f in foods) for i in items)


def shelter_blocks(items: set) -> int:
    """Estimate cobblestone/dirt for shelter."""
    # Crude — would need counts. For now binary: "has any cobble".
    return any("cobblestone" in i or "dirt" in i for i in items)


def analyze(jsonl_path: Path) -> dict | None:
    """Parse one rollout, return a feature dict."""
    rec = {
        "path": str(jsonl_path),
        "name": jsonl_path.stem,
        "final_turn": 0,
        "died": False,
        "death_phase_alive": "unknown",  # before/after first dusk
    }
    # best tier ever seen by turn T (monotone)
    best_tier_so_far = 0  # 0=none, 1=wood, 2=stone, 3=iron, 4=diamond
    tier_at = {}  # snapshot of best_tier_so_far at first turn >= N
    food_at = {}
    hp_at = {}
    shelter_built_at = None
    wall_in_at = None
    first_look_around = None
    first_dusk_alive = None  # turn at which day_ticks crossed 12000 first time

    last_state = {}

    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("_type") != "turn":
                    continue
                t = d.get("turn", 0)
                rec["final_turn"] = max(rec["final_turn"], t)
                if d.get("died"):
                    rec["died"] = True
                # Track first dusk survival
                dt = d.get("day_ticks")
                if dt is not None and dt >= 12000 and first_dusk_alive is None:
                    first_dusk_alive = t

                # Inventory item set + monotone-best tier tracking
                inv = d.get("inventory")
                items = parse_inventory(inv) if inv else set()
                cur_tier = TIER_RANK[tier_of(items)]
                if cur_tier > best_tier_so_far:
                    best_tier_so_far = cur_tier
                last_state["items"] = items
                last_state["hp"] = d.get("health")
                last_state["food"] = d.get("food")
                last_state["best_tier"] = best_tier_so_far

                # Snapshot at thresholds (records the BEST tier seen by turn N)
                tier_inv = {v: k for k, v in TIER_RANK.items()}
                for N in (10, 15, 20, 30, 50):
                    if t >= N and N not in tier_at:
                        tier_at[N] = tier_inv[best_tier_so_far]
                        food_at[N] = d.get("food")
                        hp_at[N] = d.get("health")

                # Track tool use
                tool = d.get("tool", "")
                outcome = (d.get("outcome") or "")
                if tool == "build_shelter" and "FAILED" not in outcome.upper() and shelter_built_at is None:
                    # Crude: any non-FAILED build_shelter call. Real validation would
                    # check 'armed' or follow-up shelter_armed flags but this is
                    # a starting signal.
                    if "shelter" in outcome.lower() or "armed" in outcome.lower() or "complete" in outcome.lower():
                        shelter_built_at = t
                if tool == "wall_in" and wall_in_at is None and "FAILED" not in outcome.upper():
                    wall_in_at = t
                if tool == "look_around" and first_look_around is None:
                    first_look_around = t

    except FileNotFoundError:
        return None

    rec["tier_at_10"] = tier_at.get(10, "none")
    rec["tier_at_15"] = tier_at.get(15, "none")
    rec["tier_at_20"] = tier_at.get(20, "none")
    rec["tier_at_30"] = tier_at.get(30, "none")
    rec["tier_at_50"] = tier_at.get(50, "none")
    rec["food_at_20"] = food_at.get(20)
    rec["hp_at_20"] = hp_at.get(20)
    rec["shelter_built_at"] = shelter_built_at
    rec["wall_in_at"] = wall_in_at
    rec["first_look_around"] = first_look_around
    rec["first_dusk_alive"] = first_dusk_alive  # turn at which dusk first observed
    rec["survived_first_dusk"] = first_dusk_alive is not None and rec["final_turn"] > first_dusk_alive + 5
    rec["final_tier"] = tier_of(last_state.get("items", set()))
    return rec


def bucket_label(final_turn: int) -> str:
    if final_turn < 20:
        return "A_dead_early (<20)"
    if final_turn < 50:
        return "B_short (20-50)"
    if final_turn < 100:
        return "C_medium (50-100)"
    if final_turn < 200:
        return "D_long (100-200)"
    return "E_hero (200+)"


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/rolling-20260517")
    files = sorted(base.glob("agent*.jsonl"))
    print(f"# Found {len(files)} JSONLs in {base}", file=sys.stderr)
    rows = []
    for fp in files:
        r = analyze(fp)
        if r and r["final_turn"] > 0:
            rows.append(r)
    print(f"# Parsed {len(rows)} rollouts with at least 1 turn", file=sys.stderr)

    # Survival-bucket × tier_at_15
    buckets = defaultdict(list)
    for r in rows:
        buckets[bucket_label(r["final_turn"])].append(r)

    print("\n=== Distribution by survival bucket ===")
    for k in sorted(buckets):
        print(f"  {k:24s} n={len(buckets[k])}")

    print("\n=== Tier ceiling reached @ T=15 by bucket ===")
    print(f"  {'bucket':24s} {'n':>4s}  none  wood  stone  iron  diamond")
    for k in sorted(buckets):
        rs = buckets[k]
        counts = defaultdict(int)
        for r in rs:
            counts[r["tier_at_15"]] += 1
        print(f"  {k:24s} {len(rs):4d}  "
              f"{counts['none']:4d}  {counts['wood']:4d}  {counts['stone']:5d}  "
              f"{counts['iron']:4d}  {counts['diamond']:7d}")

    print("\n=== Tier ceiling reached @ T=30 by bucket ===")
    print(f"  {'bucket':24s} {'n':>4s}  none  wood  stone  iron  diamond")
    for k in sorted(buckets):
        rs = [r for r in buckets[k] if r["final_turn"] >= 30]
        if not rs:
            continue
        counts = defaultdict(int)
        for r in rs:
            counts[r["tier_at_30"]] += 1
        print(f"  {k:24s} {len(rs):4d}  "
              f"{counts['none']:4d}  {counts['wood']:4d}  {counts['stone']:5d}  "
              f"{counts['iron']:4d}  {counts['diamond']:7d}")

    print("\n=== Survived first dusk?  (T>first_dusk_observed+5) by bucket ===")
    print(f"  {'bucket':24s} {'n':>4s}  yes  no")
    for k in sorted(buckets):
        rs = buckets[k]
        yes = sum(1 for r in rs if r["survived_first_dusk"])
        print(f"  {k:24s} {len(rs):4d}  {yes:3d}  {len(rs)-yes:3d}")

    print("\n=== shelter_built or wall_in fired by T<=20  by bucket ===")
    print(f"  {'bucket':24s} {'n':>4s}  shelter wall_in either")
    for k in sorted(buckets):
        rs = buckets[k]
        sh = sum(1 for r in rs if r["shelter_built_at"] and r["shelter_built_at"] <= 20)
        wi = sum(1 for r in rs if r["wall_in_at"] and r["wall_in_at"] <= 20)
        either = sum(1 for r in rs
                     if (r["shelter_built_at"] and r["shelter_built_at"] <= 20)
                     or (r["wall_in_at"] and r["wall_in_at"] <= 20))
        print(f"  {k:24s} {len(rs):4d}  {sh:7d} {wi:7d} {either:6d}")

    print("\n=== Joint predictor: tier_at_15=stone+ AND survived_first_dusk ===")
    print(f"  {'bucket':24s} {'n':>4s}  both  pct")
    for k in sorted(buckets):
        rs = buckets[k]
        joint = sum(1 for r in rs
                    if TIER_RANK.get(r["tier_at_15"], 0) >= 2 and r["survived_first_dusk"])
        pct = (100 * joint / len(rs)) if rs else 0
        print(f"  {k:24s} {len(rs):4d}  {joint:4d}  {pct:5.1f}%")

    print("\n=== Marginal: tier_at_15==stone (alone) by bucket ===")
    for k in sorted(buckets):
        rs = buckets[k]
        n = sum(1 for r in rs if TIER_RANK.get(r["tier_at_15"], 0) >= 2)
        pct = (100 * n / len(rs)) if rs else 0
        print(f"  {k:24s} n={len(rs):4d}  stone+@T15={n:4d}  ({pct:5.1f}%)")

    print("\n=== Marginal: tier_at_30==stone+ (alone) by bucket ===")
    for k in sorted(buckets):
        rs = [r for r in buckets[k] if r["final_turn"] >= 30]
        if not rs:
            continue
        n = sum(1 for r in rs if TIER_RANK.get(r["tier_at_30"], 0) >= 2)
        pct = (100 * n / len(rs)) if rs else 0
        print(f"  {k:24s} n={len(rs):4d}  stone+@T30={n:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
