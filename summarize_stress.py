"""Summarize a stress_test_shelter JSONL run.

Usage:
    python summarize_stress.py results/stress-shelter-<ts>.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def get_track(r: dict) -> dict:
    return r.get("track") or {}


def outcome(r: dict) -> str:
    if r.get("fatal_error"):
        return "fatal"
    if r.get("build_error"):
        return "build_err"
    br = (r.get("build_result") or "")
    if r.get("aborted") or br.startswith("ABORTED"):
        return "aborted"
    if br.startswith("FAILED"):
        return "build_failed"
    if r.get("died"):
        return "died"
    if get_track(r).get("breach"):
        return "breached"
    if get_track(r).get("interrupted"):
        return "interrupted"
    return "ok"


def shelter_status(r: dict) -> str:
    """How the SHELTER reported itself, independent of ambush outcome.

    - errored:    build raised
    - aborted:    pre-flight bail (no shelter attempted)
    - partial:    handle_build_shelter returned PARTIAL prefix
    - holes:      success prefix BUT post-inspect reported hole(s)
    - complete:   success prefix AND inspect clean
    """
    if r.get("build_error") or r.get("fatal_error"):
        return "errored"
    br = (r.get("build_result") or "")
    if r.get("aborted") or br.startswith("ABORTED") or br.startswith("FAILED"):
        return "aborted"
    if br.startswith("PARTIAL"):
        return "partial"
    if "hole(s)" in br:
        return "holes"
    return "complete"


def ambush_outcome(r: dict) -> str:
    """Did the player actually die or get breached during the ambush?

    - died:     /deaths registered a new entry during track
    - breached: mob crossed into interior AABB but player survived
    - survived: no death, no breach
    - skipped:  ambush didn't run (aborted/errored builds)
    """
    if r.get("aborted") or r.get("build_error") or r.get("fatal_error"):
        return "skipped"
    if r.get("died"):
        return "died"
    t = get_track(r)
    if t.get("breach"):
        return "breached"
    if t.get("interrupted") or not t:
        return "skipped"
    return "survived"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    rows = load(args.path)
    if not rows:
        print(f"no rows in {args.path}")
        return 1

    n = len(rows)
    outcomes = Counter(outcome(r) for r in rows)

    print(f"=== {args.path} ({n} iters) ===\n")
    print("outcomes:")
    for k, v in outcomes.most_common():
        print(f"  {k:14s} {v:3d}  {v/n:>5.0%}")

    # Shelter-status × ambush-outcome calibration table.
    # The interesting cells are the OFF-DIAGONALS:
    #   complete + died/breached → false positive (we claimed seal, mob got in)
    #   partial/holes + survived → false negative (we cried partial, held anyway)
    statuses = ["complete", "holes", "partial", "aborted", "errored"]
    amb_cols = ["survived", "breached", "died", "skipped"]
    grid: dict[tuple[str, str], int] = Counter(
        (shelter_status(r), ambush_outcome(r)) for r in rows
    )
    print("\nshelter_status × ambush_outcome:")
    header = " " * 12 + " ".join(f"{c:>9s}" for c in amb_cols) + "   total"
    print(header)
    for s in statuses:
        row_total = sum(grid.get((s, c), 0) for c in amb_cols)
        if row_total == 0:
            continue
        cells = " ".join(f"{grid.get((s, c), 0):>9d}" for c in amb_cols)
        print(f"  {s:10s}{cells}   {row_total:>5d}")

    # Flag the calibration failures explicitly.
    false_pos = [r for r in rows
                 if shelter_status(r) == "complete"
                 and ambush_outcome(r) in ("died", "breached")]
    false_neg = [r for r in rows
                 if shelter_status(r) in ("partial", "holes")
                 and ambush_outcome(r) == "survived"]
    if false_pos:
        print(f"\nFALSE POSITIVES — complete shelter, mob got in ({len(false_pos)}):")
        for r in false_pos:
            t = get_track(r)
            br = (r.get("build_result") or "")[:120]
            print(f"  iter {r.get('iter')}: ambush={ambush_outcome(r)} "
                  f"breach@={t.get('breach_first_t')} hp_min={t.get('player_hp_min')}")
            print(f"    {br}")
    if false_neg:
        print(f"\nFALSE NEGATIVES — partial/holes shelter, held anyway ({len(false_neg)}):")
        for r in false_neg:
            br = (r.get("build_result") or "")[:120]
            print(f"  iter {r.get('iter')}: {shelter_status(r)}")
            print(f"    {br}")

    # Per-iter table
    print("\nper-iter:")
    print(f"  {'#':>3} {'outcome':14s} {'mat':>22s} {'door':>14s} "
          f"{'build_s':>7s} {'spawn':>6s} {'final':>5s} {'hp_min':>6s} "
          f"{'breach@':>7s} {'wall_s':>6s}")
    build_secs: list[float] = []
    wall_secs: list[float] = []
    for r in rows:
        t = get_track(r)
        amb = r.get("ambush") or {}
        spawned = len(amb.get("spawned") or [])
        skipped = len(amb.get("skipped") or [])
        mat = (r.get("material") or "?").replace("minecraft:", "")
        door = (r.get("door_item") or "?").replace("minecraft:", "")
        bs = r.get("build_seconds")
        ws = r.get("wall_seconds")
        if isinstance(bs, (int, float)):
            build_secs.append(bs)
        if isinstance(ws, (int, float)):
            wall_secs.append(ws)
        hp_min = t.get("player_hp_min")
        hp_str = f"{hp_min:.1f}" if isinstance(hp_min, (int, float)) else "-"
        breach_t = t.get("breach_first_t")
        breach_str = f"{breach_t:.1f}" if isinstance(breach_t, (int, float)) else "-"
        print(f"  {r.get('iter','?'):>3} {outcome(r):14s} {mat:>22s} {door:>14s} "
              f"{bs if bs is not None else '-':>7} "
              f"{f'{spawned}/{spawned+skipped}':>6s} "
              f"{t.get('final_alive','-'):>5} "
              f"{hp_str:>6s} {breach_str:>7s} "
              f"{ws if ws is not None else '-':>6}")

    # Timings
    if build_secs:
        print(f"\nbuild_seconds: n={len(build_secs)} "
              f"min={min(build_secs):.1f} "
              f"median={statistics.median(build_secs):.1f} "
              f"max={max(build_secs):.1f} "
              f"mean={statistics.mean(build_secs):.1f}")
    if wall_secs:
        print(f"wall_seconds:  n={len(wall_secs)} "
              f"sum={sum(wall_secs):.1f}s "
              f"({sum(wall_secs)/60:.1f}min)")

    # Failure breakdown
    deaths = [r for r in rows if r.get("died")]
    if deaths:
        print(f"\ndeaths ({len(deaths)}):")
        for r in deaths:
            d = r.get("death") or {}
            print(f"  iter {r.get('iter')}: {d.get('message','?')} "
                  f"@ {d.get('death_pos','?')}")

    breaches = [r for r in rows if get_track(r).get("breach")]
    if breaches:
        print(f"\nbreaches ({len(breaches)}):")
        for r in breaches:
            t = get_track(r)
            print(f"  iter {r.get('iter')}: first_t={t.get('breach_first_t')} "
                  f"max_in_interior={t.get('max_in_interior')} "
                  f"build={outcome(r)}")

    build_failed = [r for r in rows if outcome(r) in ("build_err", "build_failed")]
    if build_failed:
        print(f"\nbuild failures ({len(build_failed)}):")
        for r in build_failed:
            br = (r.get("build_result") or "")[:160]
            be = r.get("build_error")
            print(f"  iter {r.get('iter')} @ {r.get('build_center')}: "
                  f"{br or be}")

    cmd_errs = [(r, r.get("cmd_errors") or []) for r in rows]
    cmd_errs = [(r, errs) for r, errs in cmd_errs if errs]
    if cmd_errs:
        print(f"\niters with cmd_errors ({len(cmd_errs)}):")
        for r, errs in cmd_errs:
            print(f"  iter {r.get('iter')}: {len(errs)} cmd error(s)")
            for e in errs[:3]:
                print(f"    {e.get('cmd')!r}: {(e.get('log') or [''])[-1]}")

    # Material / door correlation
    print("\nmaterial × outcome:")
    by_mat: dict[str, Counter] = {}
    for r in rows:
        m = (r.get("material") or "?").replace("minecraft:", "")
        by_mat.setdefault(m, Counter())[outcome(r)] += 1
    for m, c in sorted(by_mat.items()):
        line = ", ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  {m:22s} {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
