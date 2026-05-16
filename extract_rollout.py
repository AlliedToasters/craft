"""Extract survival metrics from a rollout log.

Usage:
    python extract_rollout.py <log_path> [--variant VARIANT] [--rep N]
    python extract_rollout.py --summary <results.jsonl>

In row mode, emits a single JSON line on stdout suitable for jsonl aggregation.
In summary mode, prints a per-variant table.

Row fields:
    final_turn         — last completed turn (int)
    completed          — bool, whether '=== rollout complete ===' was reached
    first_death_turn   — turn at first death, or None
    total_deaths       — count of '[death]' lines
    death_causes       — list of cause strings
    max_tier           — highest pickaxe tier reached (0=none, 1=wood, 2=stone, 3=iron, 4=diamond)
    max_tier_turn      — turn at which max_tier was first reached
    iron_armor_seen    — bool, whether any iron armor piece appeared in inventory
    max_diamonds       — largest single-slot diamond count seen (rough proxy for diamond acquisition)
    min_hp             — lowest HP value observed (proxy for danger close-calls)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PICKAXE_TIER = {
    "wooden_pickaxe": 1,
    "stone_pickaxe": 2,
    "iron_pickaxe": 3,
    "diamond_pickaxe": 4,
}
ARMOR_ITEMS = {"iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots"}

TURN_RE = re.compile(r"=== turn (\d+)/\d+: planning ===")
DEATH_RE = re.compile(r"\[death\] YOU DIED:.*?\(cause:\s*([^)]+)\)")
HP_RE = re.compile(r"HP=([\d.]+)/")
SLOT_RE = re.compile(r"slot \d+:\s+(\d+)x\s+minecraft:(\w+)")
COMPLETE_RE = re.compile(r"=== rollout complete ===")


def extract(log_path: Path) -> dict:
    text = log_path.read_text()

    final_turn = 0
    current_turn = 0
    first_death_turn: int | None = None
    death_causes: list[str] = []
    max_tier = 0
    max_tier_turn: int | None = None
    iron_armor_seen = False
    max_diamonds = 0
    min_hp = 20.0
    completed = False

    for line in text.splitlines():
        if m := TURN_RE.search(line):
            current_turn = int(m.group(1))
            final_turn = current_turn
            continue
        if m := DEATH_RE.search(line):
            if first_death_turn is None:
                first_death_turn = current_turn
            death_causes.append(m.group(1).strip())
            continue
        if m := HP_RE.search(line):
            hp = float(m.group(1))
            if hp < min_hp:
                min_hp = hp
        if m := SLOT_RE.search(line):
            count = int(m.group(1))
            item = m.group(2)
            tier = PICKAXE_TIER.get(item)
            if tier and tier > max_tier:
                max_tier = tier
                max_tier_turn = current_turn
            if item in ARMOR_ITEMS:
                iron_armor_seen = True
            if item == "diamond" and count > max_diamonds:
                max_diamonds = count
        if COMPLETE_RE.search(line):
            completed = True

    return {
        "final_turn": final_turn,
        "completed": completed,
        "first_death_turn": first_death_turn,
        "total_deaths": len(death_causes),
        "death_causes": death_causes,
        "max_tier": max_tier,
        "max_tier_turn": max_tier_turn,
        "iron_armor_seen": iron_armor_seen,
        "max_diamonds": max_diamonds,
        "min_hp": min_hp,
    }


def summarize(jsonl_path: Path) -> None:
    agg: dict[str, list[dict]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agg[r.get("variant", "?")].append(r)

    if not agg:
        print(f"(no rows in {jsonl_path})")
        return

    header = f"{'variant':<22} {'n':>3}  {'avg_tier':>8}  {'iron_armor':>10}  {'avg_dia':>7}  {'no_death':>8}  {'min_hp':>6}"
    print(header)
    print("-" * len(header))
    for v in sorted(agg):
        rows = agg[v]
        n = len(rows)
        avg_tier = mean(r["max_tier"] for r in rows)
        iron_armor = sum(1 for r in rows if r["iron_armor_seen"])
        avg_dia = mean(r["max_diamonds"] for r in rows)
        survived = sum(1 for r in rows if r["first_death_turn"] is None)
        avg_min_hp = mean(r["min_hp"] for r in rows)
        print(
            f"{v:<22} {n:>3}  {avg_tier:>8.2f}  {f'{iron_armor}/{n}':>10}  "
            f"{avg_dia:>7.1f}  {f'{survived}/{n}':>8}  {avg_min_hp:>6.1f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="rollout log (row mode) or results.jsonl (--summary)")
    p.add_argument("--variant", help="variant id to tag the emitted row with")
    p.add_argument("--rep", type=int, help="rep number to tag the emitted row with")
    p.add_argument("--summary", action="store_true", help="aggregate a results.jsonl into a per-variant table")
    args = p.parse_args()

    if args.summary:
        summarize(Path(args.path))
        return

    row = extract(Path(args.path))
    if args.variant:
        row["variant"] = args.variant
    if args.rep is not None:
        row["rep"] = args.rep
    row["log"] = str(args.path)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
