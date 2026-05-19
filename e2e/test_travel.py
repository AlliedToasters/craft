"""Multi-iter test of the travel tool.

travel(direction, distance) walks Baritone a configurable number of blocks
along a cardinal axis. Test: build a clear stone arena, record start xz,
dispatch travel("north", 10), assert |Δz| ≈ 10 (north = -z).

Pass criteria per iter:
  - outcome doesn't start with FAILED
  - |Δz| >= 0.7 * distance  (Baritone may stop slightly short — partial OK)
  - |Δx| < 4  (we asked for a straight cardinal walk)
  - wall_s <= timeout
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from pathlib import Path

from craft.testkit import (
    PLAYER_NAME,
    TestLogger,
    build_arena,
    cmd,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_DISTANCE = 10
DEFAULT_DIRECTION = "north"  # north = -z
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_PASS_RATE = 0.9


DIR_DELTA = {
    "north": (0, -1),  # dx_axis, dz_axis sign that should be non-trivial
    "south": (0, 1),
    "east":  (1, 0),
    "west":  (-1, 0),
}


def run_iter(
    rec: dict,
    *,
    direction: str,
    distance: int,
    timeout_s: float,
    spawn_range: int,
    rng: _random.Random,
    verbose: bool,
) -> None:
    if spawn_range > 0:
        spawn_result = random_spawn(
            range_blocks=spawn_range, rng=rng, verbose=verbose
        )
        rec["spawn"] = spawn_result
        if not spawn_result.get("ok"):
            rec["passed"] = False
            rec["fail_reason"] = "spawn_retry_exhausted"
            return

    start = pos()
    if start is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_position"
        return
    anchor = (
        int(math.floor(start[0])),
        int(math.floor(start[1])),
        int(math.floor(start[2])),
    )
    rec["anchor"] = list(anchor)
    if verbose:
        print(f"[test] anchor = {anchor}, direction={direction}, distance={distance}",
              flush=True)

    setup_clean(anchor)
    # Arena wide enough for the requested travel + 4 block buffer.
    radius = max(distance + 4, 8)
    build_arena(anchor, x_radius=radius)

    # Re-read pos AFTER setup_clean's /tp settles, so deltas are measured
    # from the actual departure spot rather than the pre-tp position.
    start2 = pos()
    if start2 is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_setup_position"
        return
    sx, sy, sz = start2
    rec["start_pos"] = [sx, sy, sz]

    t0 = time.monotonic()
    outcome: str | None = None
    try:
        outcome = dispatch(
            "travel",
            json.dumps({"direction": direction, "distance": distance}),
        )
    except Exception as e:
        outcome = f"FAILED: dispatch threw {e!r}"
    elapsed = round(time.monotonic() - t0, 2)
    rec["outcome"] = outcome
    rec["travel_wall_s"] = elapsed
    if verbose:
        print(f"[test] outcome ({elapsed}s): {outcome}", flush=True)

    end = pos()
    if end is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_position"
        return
    ex, ey, ez = end
    rec["end_pos"] = [ex, ey, ez]

    dx = ex - sx
    dz = ez - sz
    rec["dx"] = round(dx, 2)
    rec["dz"] = round(dz, 2)

    # Project the (dx, dz) vector onto the requested axis. Positive = went
    # the right way; negative = went backwards.
    axis_sign_x, axis_sign_z = DIR_DELTA[direction]
    along = dx * axis_sign_x + dz * axis_sign_z
    perp = abs(dx) if axis_sign_x == 0 else abs(dz)
    rec["along"] = round(along, 2)
    rec["perp"] = round(perp, 2)

    is_failed = outcome is None or outcome.startswith("FAILED")
    moved_along = along >= 0.7 * distance
    stayed_on_axis = perp < 4
    within_budget = elapsed <= timeout_s

    rec["distance"] = distance
    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "moved_along_axis": moved_along,
        "stayed_on_axis": stayed_on_axis,
        "within_timeout": within_budget,
    }
    rec["passed"] = (
        (not is_failed) and moved_along and stayed_on_axis and within_budget
    )
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not moved_along:
            rec["fail_reason"] = "did_not_travel_far_enough"
        elif not stayed_on_axis:
            rec["fail_reason"] = "drifted_off_axis"
        else:
            rec["fail_reason"] = "timeout"

    cmd(f"clear {PLAYER_NAME}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--direction", choices=list(DIR_DELTA.keys()),
                    default=DEFAULT_DIRECTION)
    ap.add_argument("--distance", type=int, default=DEFAULT_DISTANCE)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--spawn-range", type=int, default=0)
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight()
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger("travel", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    direction=args.direction,
                    distance=args.distance,
                    timeout_s=args.timeout,
                    spawn_range=args.spawn_range,
                    rng=rng,
                    verbose=not args.quiet,
                )
        except Exception as e:
            print(f"[test] iter {i} raised: {e!r}", flush=True)

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
