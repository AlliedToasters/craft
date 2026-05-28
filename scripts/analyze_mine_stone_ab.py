#!/usr/bin/env python3
"""Analyze a mine_stone descend-nudge A/B run.

Reads the on/ and off/ basket subdirs of a run produced by
scripts/mine_stone_nudge_ab.sh and prints a per-arm metric table. All metrics
derive from the JSONL `tool`/`outcome` fields, so this can be re-run on the
artifacts after the fact with no rollout cost.

Per-arm metrics:
  attempted%          agents that called mine_stone at all (informativeness gate)
  stone_acquired%     agents where mine_stone returned a positive acquire (binary)
  mean_cobble         mean cobblestone-drops collected per rollout (over all n)
  cobble_if_acquired  mean cobble among rollouts that acquired any (magnitude)
  descend_after_fail% agents where a descend followed a mine_stone FAILED
                      (the exact behavior the nudge is meant to induce)
  mean_fails          mean count of mine_stone FAILED outcomes per rollout (friction)

Usage:
  scripts/analyze_mine_stone_ab.py [RUN_DIR]   # default: latest results/mine-stone-nudge-ab-*
"""
import glob
import json
import os
import re
import sys

# Cumulative stone-drop count reported in mine_stone success/partial/target-met
# outcomes, e.g. "...now have 10 mine_stone-drops..." / "...already had 3...".
_HAVE_RE = re.compile(r"(?:now have|already had) (\d+)")


def analyze_iter(path):
    attempted = False
    fails = 0
    acquired = False
    descend_after_fail = False
    pending_fail = False
    cobble = 0
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("_type") == "header" or "tool" not in r:
            continue
        tool = r.get("tool")
        out = str(r.get("outcome") or "")
        if tool == "mine_stone":
            attempted = True
            if out.startswith("FAILED"):
                fails += 1
                pending_fail = True
            else:
                m = _HAVE_RE.search(out)
                if m:
                    cobble = max(cobble, int(m.group(1)))
                if "acquired 0" not in out and "acquired" in out:
                    acquired = True
                    pending_fail = False
        elif tool == "descend":
            if pending_fail:
                descend_after_fail = True
            pending_fail = False
    return {
        "attempted": attempted,
        "fails": fails,
        "acquired": acquired,
        "descend_after_fail": descend_after_fail,
        "cobble": cobble,
    }


def arm_stats(run_dir, arm):
    paths = sorted(glob.glob(os.path.join(run_dir, arm, "*.jsonl")))
    rows = [analyze_iter(p) for p in paths]
    n = len(rows)
    att = sum(r["attempted"] for r in rows)
    acq = sum(r["acquired"] for r in rows)
    daf = sum(r["descend_after_fail"] for r in rows)
    tf = sum(r["fails"] for r in rows)
    cobble_total = sum(r["cobble"] for r in rows)
    cobble_acq = [r["cobble"] for r in rows if r["acquired"]]
    return {
        "n": n,
        "attempted": att,
        "acquired": acq,
        "descend_after_fail": daf,
        "mean_fails": tf / n if n else 0.0,
        "mean_cobble": cobble_total / n if n else 0.0,
        "cobble_if_acquired": (sum(cobble_acq) / len(cobble_acq)) if cobble_acq else 0.0,
    }


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "  n/a"


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if run_dir is None:
        candidates = sorted(glob.glob("results/mine-stone-nudge-ab-*"))
        if not candidates:
            print("no results/mine-stone-nudge-ab-* dir found; pass RUN_DIR")
            return 1
        run_dir = candidates[-1]
    print(f"=== MINE_STONE DESCEND-NUDGE A/B  ({run_dir}) ===")
    hdr = (f"{'arm':<5} {'n':>3} {'attempted':>10} {'stone_acq':>10} "
           f"{'mean_cobble':>12} {'cobble/acq':>11} {'descend_after_fail':>20} {'mean_fails':>11}")
    print(hdr)
    arms = {}
    for arm in ("on", "off"):
        s = arm_stats(run_dir, arm)
        arms[arm] = s
        n = s["n"]
        print(f"{arm:<5} {n:>3} {pct(s['attempted'], n):>10} {pct(s['acquired'], n):>10} "
              f"{s['mean_cobble']:>12.2f} {s['cobble_if_acquired']:>11.2f} "
              f"{pct(s['descend_after_fail'], n):>20} {s['mean_fails']:>11.2f}")
    print()
    print("If the nudge helps: ON shows higher descend_after_fail% (behavior changed)")
    print("→ higher stone_acq% / mean_cobble (it paid off) → lower mean_fails (less flailing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
