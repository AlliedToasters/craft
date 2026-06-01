"""Aggregate rollout wall-clock by tool x outcome (pass/fail).

The turn-vs-time conflation (2026-06-01): measuring failures by TURN COUNT
overstates cheap-but-frequent friction (placement) and understates slow friction
(failed mining at ~40-46s/call). This script aggregates the per-step `total_s`
from a bigN rollout dir so "where does the time actually go" is answered by
wall-clock, not turn counts.

Usage:
    python -m scripts.wallclock_by_tool results/bigN20-easy-qwen-<ts> [more dirs...]

Prints, per dir:
  - total wall-clock across all steps, split LLM-plan / tool-exec / ctx-fetch
  - per (tool, pass/fail): n, sum_total_s, mean_total_s, % of all wall-clock
  - MINING focus: failed-mine mean per-call total_s (the watchdog headline)
  - PLACEMENT focus: no_placeable_spot / no_space / make-room counts
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

MINE_TOOLS = {"mine_wood", "mine_stone", "mine_iron", "mine_diamond", "mine_coal", "mine"}
PLACE_FAIL_TOKENS = ("no_placeable_spot", "no_space", "no flat ground", "make_room")


def _is_failed(outcome: str) -> bool:
    return outcome is None or outcome.startswith("FAILED")


def analyze(result_dir: str) -> None:
    steps: list[dict] = []
    for f in glob.glob(f"{result_dir}/agent*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "tool" in r and "total_s" in r:
                steps.append(r)

    if not steps:
        print(f"\n=== {result_dir} ===\n  (no step records found)")
        return

    total_wall = sum(s.get("total_s", 0.0) for s in steps)
    plan_wall = sum(s.get("plan_s", 0.0) for s in steps)
    exec_wall = sum(s.get("exec_s", 0.0) for s in steps)
    ctx_wall = sum(s.get("ctx_s", 0.0) for s in steps)

    print(f"\n=== {result_dir} ===")
    print(f"  steps={len(steps)}  total_wall={total_wall:.0f}s")
    print(f"  split: plan={plan_wall:.0f}s ({100*plan_wall/total_wall:.0f}%)  "
          f"exec={exec_wall:.0f}s ({100*exec_wall/total_wall:.0f}%)  "
          f"ctx={ctx_wall:.0f}s ({100*ctx_wall/total_wall:.0f}%)")

    # Per (tool, pass/fail), with the plan(think)/exec/ctx split folded in.
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in steps:
        key = (s["tool"], "FAIL" if _is_failed(s.get("outcome", "")) else "ok")
        agg[key].append(s)

    print(f"\n  {'tool':22s} {'res':4s} {'n':>4s} {'sum_s':>8s} {'mean_s':>7s} "
          f"{'%wall':>6s} {'think_s':>8s} {'exec_s':>8s} {'ctx_s':>7s}")
    rows = sorted(agg.items(), key=lambda kv: -sum(x.get("total_s", 0.0) for x in kv[1]))
    for (tool, res), ss in rows:
        ssum = sum(x.get("total_s", 0.0) for x in ss)
        think = sum(x.get("plan_s", 0.0) for x in ss)
        ex = sum(x.get("exec_s", 0.0) for x in ss)
        cx = sum(x.get("ctx_s", 0.0) for x in ss)
        print(f"  {tool:22s} {res:4s} {len(ss):>4d} {ssum:>8.0f} "
              f"{ssum/len(ss):>7.1f} {100*ssum/total_wall:>5.1f}% "
              f"{think:>8.0f} {ex:>8.0f} {cx:>7.0f}")

    # Mining focus.
    print("\n  -- MINING (the watchdog headline) --")
    for tool in sorted(MINE_TOOLS):
        fails = [s.get("total_s", 0.0) for s in steps
                 if s["tool"] == tool and _is_failed(s.get("outcome", ""))]
        oks = [s.get("total_s", 0.0) for s in steps
               if s["tool"] == tool and not _is_failed(s.get("outcome", ""))]
        if fails or oks:
            fmean = f"{sum(fails)/len(fails):.1f}s" if fails else "—"
            omean = f"{sum(oks)/len(oks):.1f}s" if oks else "—"
            print(f"    {tool:14s} FAIL n={len(fails):<3d} mean={fmean:<7s} "
                  f"sum={sum(fails):.0f}s   |  ok n={len(oks):<3d} mean={omean}")
    all_mine_fail = [s.get("total_s", 0.0) for s in steps
                     if s["tool"] in MINE_TOOLS and _is_failed(s.get("outcome", ""))]
    if all_mine_fail:
        print(f"    ALL failed-mine: n={len(all_mine_fail)} mean={sum(all_mine_fail)/len(all_mine_fail):.1f}s "
              f"sum={sum(all_mine_fail):.0f}s ({100*sum(all_mine_fail)/total_wall:.1f}% of wall)")

    # Placement focus.
    print("\n  -- PLACEMENT --")
    place_steps = [s for s in steps if s["tool"] in ("place", "craft", "smelt", "build_shelter")]
    place_fail = [s for s in place_steps
                  if _is_failed(s.get("outcome", ""))
                  and any(tok in (s.get("outcome", "") or "") for tok in PLACE_FAIL_TOKENS)]
    pf_wall = sum(s.get("total_s", 0.0) for s in place_fail)
    print(f"    placement-spot failures (no_placeable_spot/no_space): n={len(place_fail)} "
          f"sum={pf_wall:.0f}s ({100*pf_wall/total_wall:.1f}% of wall)")
    cleared = [s for s in steps if "cleared" in (s.get("outcome", "") or "").lower()
               and s["tool"] in ("place", "craft", "smelt")]
    print(f"    make-room clears observed in outcomes: n={len(cleared)}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for d in argv:
        analyze(d.rstrip("/"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
