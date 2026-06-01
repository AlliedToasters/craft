"""Constrained-geometry regression test for the make-room placement fix.

The existing place/smelt e2e tests all `build_arena` a flat clearing, so they
CANNOT reproduce the `no_placeable_spot` / `no_space` friction that dominates
qwen's turn-1 crafting-table wall (slope/canopy spawns). This test builds the
hostile geometry by hand with `setblock`/`fill` — no arena — to exercise the
two-tier Placer fix:

  - **slope** (Tier 1, vertical search): player on a 1-block pedestal over a
    deep pit with a one-step-down terrace to the east. Every feet-Y candidate
    is open-but-unsupported; the only valid spot is one block DOWN. Pre-fix
    (feet-Y-only search) → `no_placeable_spot`. Post-fix (y∈{0,-1,+1}) → places
    on the terrace (placed_y == feet_y - 1).

  - **make_room** (Tier 2, Excavate-one-cell): player on a pedestal over a pit;
    the only supported candidate is a clearable dirt block cardinally adjacent
    at feet-Y, capped by a ceiling so it can't be placed-on-top. No Tier-1 spot
    exists at any y. Post-fix clears the dirt then places into the freed cell
    (placed at the dirt's old coord). Tier-2 needs Baritone to break an
    adjacent block, so reachability is the genuine risk this scenario probes.

Pass criteria per iter (for the chosen scenario):
  - outcome does NOT start with FAILED
  - a placed_at coord is parsed and matches the scenario's expected cell
  - wall_s <= timeout

Run:  python -m e2e.test_place_constrained --scenario slope --iters 3
      python -m e2e.test_place_constrained --scenario make_room --iters 3
      python -m e2e.test_place_constrained --scenario all --iters 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from craft.testkit import (
    PLAYER_NAME,
    TestLogger,
    cmd,
    pos,
    preflight,
    setup_clean,
)
from craft.tools import dispatch

# Fixed build site, far from spawn/village and the burrow/doorway fixed arenas
# (5000,100,5000). Same world seed across wipes → reproducible.
ANCHOR = (6100, 100, 6100)
DEFAULT_ITEM = "crafting_table"
DEFAULT_TIMEOUT_S = 40.0
DEFAULT_PASS_RATE = 0.9

SCENARIOS = ("slope", "make_room")


def _c(s: str) -> None:
    cmd(s)


def _clear_volume(ax: int, ay: int, az: int) -> None:
    """Wipe an 11x11 column from 4 below feet to 6 above → blank canvas."""
    _c(f"fill {ax-5} {ay-4} {az-5} {ax+5} {ay+6} {az+5} minecraft:air")
    # Safety floor 4 below feet: a fall off the pedestal is ~3 blocks (survivable
    # with the setup_clean instant_health), and it sits at feet_y-4 so the
    # Tier-1 dy=-1 support check (feet_y-2) sees air everywhere it shouldn't.
    _c(f"fill {ax-5} {ay-4} {az-5} {ax+5} {ay-4} {az+5} minecraft:stone")


def _build_slope(anchor: tuple[int, int, int]) -> None:
    """Pedestal + one-step-down terrace to the east. Recovery is at feet_y-1."""
    ax, ay, az = anchor
    _clear_volume(ax, ay, az)
    _c(f"setblock {ax} {ay-1} {az} minecraft:stone")          # player pedestal
    _c(f"setblock {ax+1} {ay-2} {az} minecraft:stone")        # terrace floor (dy=-1)
    _c(f"setblock {ax+2} {ay-2} {az} minecraft:stone")
    time.sleep(0.3)


def _build_make_room(anchor: tuple[int, int, int]) -> None:
    """Pedestal + a single clearable dirt block (capped) cardinally adjacent."""
    ax, ay, az = anchor
    _clear_volume(ax, ay, az)
    _c(f"setblock {ax} {ay-1} {az} minecraft:stone")          # player pedestal
    _c(f"setblock {ax+1} {ay-1} {az} minecraft:stone")        # dirt's sturdy floor
    _c(f"setblock {ax+1} {ay}   {az} minecraft:dirt")         # clearable cell (feet-Y)
    _c(f"setblock {ax+1} {ay+1} {az} minecraft:stone")        # ceiling: blocks place-on-top
    time.sleep(0.3)


def _extract_placed_at(outcome: str) -> tuple[int, int, int] | None:
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", outcome)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def run_iter(rec: dict, *, scenario: str, item: str, timeout_s: float, verbose: bool) -> None:
    ax, ay, az = ANCHOR
    rec["scenario"] = scenario
    rec["anchor"] = list(ANCHOR)
    rec["item"] = item

    # Force the target chunk to load before we setblock into it. First-iter
    # race: fills/setblocks at a far coord silently no-op until a player loads
    # the chunk — without this the player tps onto a pedestal that was never
    # built and falls into the void.
    cmd(f"gamemode creative {PLAYER_NAME}")
    cmd(f"tp {PLAYER_NAME} {ax} {ay + 30} {az}")
    time.sleep(1.0)

    # Build geometry first, then drop the player onto the pedestal. setup_clean
    # tps to feet=(ax,ay,az) in creative, settles, switches to survival.
    if scenario == "slope":
        _build_slope(ANCHOR)
    elif scenario == "make_room":
        _build_make_room(ANCHOR)
    else:
        rec["passed"] = False
        rec["fail_reason"] = f"unknown_scenario:{scenario}"
        return

    setup_clean(ANCHOR)
    # Rebuild any geometry setup_clean's tp/gamemode toggles may have disturbed
    # (creative tp shouldn't, but the player now occupies the pedestal cell).
    cmd(f"give {PLAYER_NAME} minecraft:{item} 1")
    time.sleep(0.3)

    after = pos()
    if after is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_position"
        return
    rec["start_pos"] = list(after)
    # Guard: player must actually be on the pedestal (didn't fall into the pit).
    if abs(after[1] - ay) > 1.5:
        rec["passed"] = False
        rec["fail_reason"] = f"player_off_pedestal (y={after[1]:.1f}, want ~{ay})"
        return

    t0 = time.monotonic()
    try:
        outcome = dispatch("place", json.dumps({"item": item}))
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["place_wall_s"] = elapsed
    if verbose:
        print(f"[test:{scenario}] outcome ({elapsed}s): {outcome}", flush=True)

    placed_at = _extract_placed_at(outcome) if outcome else None
    rec["placed_at"] = list(placed_at) if placed_at else None

    is_failed = outcome is None or outcome.startswith("FAILED")
    has_placed = placed_at is not None

    # Expected cell per scenario.
    expected_ok = False
    if placed_at is not None:
        px, py, pz = placed_at
        if scenario == "slope":
            # Recovery is one step down (feet_y - 1), on the east terrace.
            expected_ok = (py == ay - 1) and (px in (ax + 1, ax + 2)) and (pz == az)
        elif scenario == "make_room":
            # The cleared dirt cell, at feet-Y.
            expected_ok = (px, py, pz) == (ax + 1, ay, az)
        rec["expected_cell_ok"] = expected_ok
        # Cleanup the placed block.
        _c(f"setblock {px} {py} {pz} minecraft:air")

    within_budget = elapsed <= timeout_s
    cmd(f"clear {PLAYER_NAME}")
    cmd("kill @e[type=item,distance=..32]")

    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "placed_at_parsed": has_placed,
        "expected_cell": expected_ok,
        "within_timeout": within_budget,
    }
    rec["passed"] = (not is_failed) and has_placed and expected_ok and within_budget
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not has_placed:
            rec["fail_reason"] = "placed_at_not_parsed"
        elif not expected_ok:
            rec["fail_reason"] = "placed_wrong_cell"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="slope")
    ap.add_argument("--item", default=DEFAULT_ITEM)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--spawn-range", type=int, default=0)  # accepted, unused (fixed arena)
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    logger = TestLogger("place_constrained", path=Path(args.out) if args.out else None)

    idx = 0
    for i in range(args.iters):
        for scenario in scenarios:
            try:
                with logger.iter_record(idx) as rec:
                    run_iter(
                        rec,
                        scenario=scenario,
                        item=args.item,
                        timeout_s=args.timeout,
                        verbose=not args.quiet,
                    )
            except Exception as e:
                print(f"[test] iter {idx} ({scenario}) raised: {e!r}", flush=True)
            idx += 1

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
