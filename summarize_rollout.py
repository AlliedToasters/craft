"""Post-hoc summary of a craft.agent rollout log.

Usage:
    python summarize_rollout.py results/rollout-shelter-pd-<ts>.log
    python summarize_rollout.py results/*.log  # compare mode

What it extracts:
  - Per-turn: tool, args, outcome, plan/exec timing, hp, day_ticks, biome,
    inventory snapshot
  - Categories: gather / craft / movement / shelter / failed / other
  - Milestones: first occurrence of each tool/armor/material in inventory
  - Death cause + turn (if permadeath)

What it reports:
  - Per-rollout: tech-tree timing breakdown (early-craft cost vs amortization),
    milestone timeline (turn + cumulative wall-time), category-wall-time table
  - Compare mode: side-by-side milestone arrival across rollouts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# Tool name → category. Anything not listed is "other".
_CATEGORIES = {
    "mine_wood": "gather", "mine_stone": "gather", "mine_coal": "gather",
    "mine_iron": "gather", "mine_diamond": "gather", "mine_copper": "gather",
    "craft": "craft", "smelt": "craft", "collect_smelt": "craft",
    "surface": "movement", "descend": "movement", "travel": "movement",
    "goto_corpse": "movement",
    "build_shelter": "shelter", "place": "shelter",
}
_CAT_ORDER = ["gather", "craft", "movement", "shelter", "other"]

# Items we care about as milestones. Order matters for stable output.
_MILESTONE_ITEMS = [
    # Wooden tools
    "wooden_pickaxe", "wooden_axe", "wooden_shovel", "wooden_sword",
    # Stone tier
    "stone_pickaxe", "stone_axe", "stone_sword",
    # Copper armor
    "copper_chestplate", "copper_helmet", "copper_leggings", "copper_boots",
    # Iron tier + armor
    "iron_ingot", "iron_pickaxe", "iron_sword", "iron_axe",
    "iron_chestplate", "iron_helmet", "iron_leggings", "iron_boots",
    # Diamond
    "diamond", "diamond_pickaxe", "diamond_chestplate", "diamond_helmet",
    # Other notable
    "coal", "oak_door", "crafting_table", "furnace",
]


@dataclass
class Turn:
    n: int
    tool: str | None = None
    args: str | None = None
    outcome: str | None = None
    plan_s: float | None = None
    exec_s: float | None = None
    total_s: float | None = None
    hp: float | None = None
    food: int | None = None
    day_ticks: int | None = None
    biome: str | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    failed: bool = False


@dataclass
class Rollout:
    path: Path
    goal: str | None = None
    max_turns: int | None = None
    permadeath: bool = False
    setup: dict = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    death_turn: int | None = None
    death_cause: str | None = None
    death_pos: tuple[int, int, int] | None = None
    completed: bool = False  # saw "=== rollout complete ==="


_RE_GOAL = re.compile(r"=== goal=(\S+), max_turns=(\d+)(.*?)\s*===")
_RE_TURN_PLAN = re.compile(r"=== turn (\d+)/\d+: planning ===")
_RE_TURN_EXEC = re.compile(r"=== turn (\d+): executing (\w+)\((.*)\) ===")
_RE_TURN_OUT = re.compile(r"=== turn (\d+) outcome: (.*?) ===\s*$")
_RE_TIMING = re.compile(
    r"\[timing\] turn (\d+): plan=([\d.]+)s exec=([\d.]+)s"
)
_RE_TIMING_TOTAL = re.compile(
    r"\[timing\] turn (\d+) total=([\d.]+)s"
)
_RE_STATS_HP = re.compile(r"\[stats\] Stats: HP=([\d.]+)/[\d.]+ food=(\d+)")
_RE_STATS_TIME = re.compile(
    r"time=(DAY|NIGHT)\s*([\d.]+)min until (?:dusk|dawn)(?:\s*\(day (\d+)\))?"
)
_RE_STATS_BIOME = re.compile(r"biome=(\S+)")
_RE_INV_HEADER = re.compile(r"^\[inventory\]\s*$")
_RE_INV_LINE = re.compile(r"^\s+slot \d+:\s+(\d+)x minecraft:(\S+)")
_RE_INV_EMPTY = re.compile(r"^\s+\(empty\)")
_RE_INV_CURHEADER = re.compile(r"^Current inventory:\s*$")
# An inventory block ends at the first line that looks like a NEW log section.
_RE_BLOCK_END = re.compile(r"^(?:\[[a-z_]+\]|\s*===\s|$)")
_RE_DEATH = re.compile(
    r"\[death\] YOU DIED: .*?(?:by\s+)?(\w+)?\s*\(cause:\s*(\S+)\).*?"
    r"Died at \((-?\d+),(-?\d+),(-?\d+)\)"
)
_RE_PERMA = re.compile(r"=== PERMADEATH: trajectory terminated at turn (\d+) ===")
_RE_COMPLETE = re.compile(r"=== rollout complete ===")
_RE_SETUP_TP = re.compile(r"\[setup\] tp to \((-?\d+),\d+,(-?\d+)\)")
_RE_SETUP_PHASE = re.compile(r"\[setup\] time set (\d+) \(phase=(\w+)\)")


def parse(path: Path) -> Rollout:
    r = Rollout(path=path)
    cur: Turn | None = None
    in_inv = False
    inv_buf: dict[str, int] = {}

    def finish_inv():
        nonlocal in_inv, inv_buf
        if cur is not None and in_inv:
            cur.inventory = dict(inv_buf)
        in_inv = False
        inv_buf = {}

    with path.open() as f:
        for line in f:
            line_stripped = line.rstrip()

            m = _RE_GOAL.search(line)
            if m:
                r.goal = m.group(1)
                r.max_turns = int(m.group(2))
                r.permadeath = "permadeath" in m.group(3).lower()
                continue
            m = _RE_SETUP_TP.search(line)
            if m:
                r.setup["tp_xz"] = (int(m.group(1)), int(m.group(2)))
                continue
            m = _RE_SETUP_PHASE.search(line)
            if m:
                r.setup["start_phase"] = m.group(2)
                continue

            m = _RE_TURN_PLAN.search(line)
            if m:
                finish_inv()
                n = int(m.group(1))
                cur = Turn(n=n)
                r.turns.append(cur)
                continue
            m = _RE_TURN_EXEC.search(line)
            if m and cur is not None and cur.n == int(m.group(1)):
                cur.tool = m.group(2)
                cur.args = m.group(3)
                continue
            m = _RE_TURN_OUT.search(line)
            if m and cur is not None and cur.n == int(m.group(1)):
                cur.outcome = m.group(2).strip()
                cur.failed = (
                    cur.outcome.startswith("FAILED")
                    or cur.outcome.startswith("ABORTED")
                )
                continue
            m = _RE_TIMING.search(line)
            if m and cur is not None and cur.n == int(m.group(1)):
                cur.plan_s = float(m.group(2))
                cur.exec_s = float(m.group(3))
                continue
            m = _RE_TIMING_TOTAL.search(line)
            if m and cur is not None and cur.n == int(m.group(1)):
                cur.total_s = float(m.group(2))
                continue
            m = _RE_STATS_HP.search(line)
            if m and cur is not None:
                try:
                    cur.hp = float(m.group(1))
                except (TypeError, ValueError):
                    pass
                try:
                    cur.food = int(m.group(2))
                except (TypeError, ValueError):
                    pass
                mt = _RE_STATS_TIME.search(line)
                if mt:
                    phase, mins_s = mt.group(1), float(mt.group(2))
                    # Derive day_ticks from "Xmin until dusk/dawn".
                    cur.day_ticks = int(12000 - mins_s * 1200) if phase == "DAY" else int(24000 - mins_s * 1200)
                mb = _RE_STATS_BIOME.search(line)
                if mb:
                    cur.biome = mb.group(1).split(":")[-1]
                continue

            if _RE_INV_HEADER.match(line):
                in_inv = True
                inv_buf = {}
                continue
            if in_inv:
                m = _RE_INV_LINE.match(line)
                if m:
                    inv_buf[m.group(2)] = inv_buf.get(m.group(2), 0) + int(m.group(1))
                    continue
                if _RE_INV_EMPTY.match(line):
                    continue
                if _RE_INV_CURHEADER.match(line):
                    continue
                # A new log-section header ends the inv block.
                if _RE_BLOCK_END.match(line):
                    finish_inv()

            m = _RE_DEATH.search(line)
            if m:
                r.death_cause = m.group(2)
                r.death_pos = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
                if cur is not None:
                    r.death_turn = cur.n
                continue
            m = _RE_PERMA.search(line)
            if m:
                r.death_turn = int(m.group(1))
                continue
            if _RE_COMPLETE.search(line):
                r.completed = True

    finish_inv()
    return r


def categorize(turn: Turn) -> str:
    if turn.failed:
        return "failed"
    if turn.tool is None:
        return "other"
    return _CATEGORIES.get(turn.tool, "other")


def milestones(r: Rollout) -> dict[str, tuple[int, float]]:
    """First turn each milestone item appeared in inventory + cumulative wall-time."""
    seen: dict[str, tuple[int, float]] = {}
    cum_t = 0.0
    for t in r.turns:
        if t.total_s is not None:
            cum_t += t.total_s
        for item in t.inventory:
            if item in _MILESTONE_ITEMS and item not in seen:
                seen[item] = (t.n, cum_t)
    return seen


def summarize(r: Rollout) -> None:
    print(f"\n=== {r.path.name} ===")
    print(f"  goal={r.goal} max_turns={r.max_turns} permadeath={r.permadeath} "
          f"setup={r.setup}")
    n = len(r.turns)
    if n == 0:
        print("  (no turns parsed)")
        return
    last = r.turns[-1]
    final_biome = next((t.biome for t in reversed(r.turns) if t.biome), None)
    print(f"  reached turn {last.n}/{r.max_turns}, final biome={final_biome}")
    if r.death_turn:
        print(f"  DEATH at turn {r.death_turn}: cause={r.death_cause} "
              f"pos={r.death_pos}")
    if r.completed and not r.death_turn:
        print(f"  rollout completed without death")

    # Category timing
    cat_time: dict[str, float] = Counter()
    cat_count: dict[str, int] = Counter()
    total = 0.0
    for t in r.turns:
        c = categorize(t)
        if t.total_s is not None:
            cat_time[c] += t.total_s
            total += t.total_s
        cat_count[c] += 1

    print(f"\n  per-category wall time (total {total:.1f}s = {total/60:.1f}min):")
    for c in _CAT_ORDER + (["failed"] if "failed" in cat_time else []):
        if cat_count.get(c, 0) == 0:
            continue
        pct = (cat_time[c] / total * 100) if total > 0 else 0
        print(f"    {c:10s} {cat_time[c]:6.1f}s ({pct:4.0f}%)  "
              f"{cat_count[c]:>2d} turn(s)")

    # Milestones
    ms = milestones(r)
    print(f"\n  milestones reached ({len(ms)}/{len(_MILESTONE_ITEMS)}):")
    if not ms:
        print("    (none)")
    else:
        for item in _MILESTONE_ITEMS:
            if item in ms:
                turn_n, cum_t = ms[item]
                print(f"    {item:22s} turn {turn_n:>3d}  @ {cum_t:>6.1f}s")

    # Tool-call mix (top tools by count)
    tool_count = Counter(t.tool for t in r.turns if t.tool)
    print(f"\n  tool calls ({len(r.turns)} turns):")
    for tool, c in tool_count.most_common():
        avg_s = sum(t.total_s or 0 for t in r.turns if t.tool == tool) / c
        fail_n = sum(1 for t in r.turns if t.tool == tool and t.failed)
        fail_tag = f" ({fail_n} failed)" if fail_n else ""
        print(f"    {tool:18s} ×{c:>2d}  avg {avg_s:>5.1f}s/call{fail_tag}")


def compare(rs: list[Rollout]) -> None:
    print(f"\n=== compare ({len(rs)} rollouts) ===")
    # Milestone arrival table.
    all_ms: dict[str, list[tuple[int, float] | None]] = {}
    for item in _MILESTONE_ITEMS:
        row: list[tuple[int, float] | None] = []
        for r in rs:
            ms = milestones(r)
            row.append(ms.get(item))
        if any(c is not None for c in row):
            all_ms[item] = row

    if not all_ms:
        print("  no shared milestones — skipping table")
        return

    header = "  " + "milestone".ljust(22)
    for r in rs:
        header += r.path.stem[-12:].rjust(15)
    print(header)
    for item, row in all_ms.items():
        line = "  " + item.ljust(22)
        for cell in row:
            if cell is None:
                line += "—".rjust(15)
            else:
                line += f"t{cell[0]}/{cell[1]:.0f}s".rjust(15)
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true",
                    help="emit parsed rollout(s) as JSON instead of human summary")
    args = ap.parse_args()

    rs = [parse(p) for p in args.paths]

    if args.json:
        out = []
        for r in rs:
            ms = milestones(r)
            out.append({
                "path": str(r.path),
                "goal": r.goal,
                "max_turns": r.max_turns,
                "permadeath": r.permadeath,
                "setup": r.setup,
                "turns_reached": len(r.turns),
                "death_turn": r.death_turn,
                "death_cause": r.death_cause,
                "completed": r.completed,
                "milestones": {k: {"turn": v[0], "cum_s": v[1]} for k, v in ms.items()},
                "category_seconds": {
                    c: sum(t.total_s or 0 for t in r.turns if categorize(t) == c)
                    for c in _CAT_ORDER + ["failed"]
                },
            })
        print(json.dumps(out, indent=2))
        return 0

    for r in rs:
        summarize(r)
    if len(rs) > 1:
        compare(rs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
