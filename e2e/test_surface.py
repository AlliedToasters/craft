"""Multi-iter test of the surface tool.

surface() ascends to open sky in the current column. The tool is CHUNKED
(SURFACE_MAX_PER_CALL=40), but Baritone's per-chunk path can still timeout
through dense biomes (trees, leaves, ravines). The agent is documented to
call surface() repeatedly until the outcome stops saying "call surface()
again"; the test mirrors that pattern with a small max-calls cap.

Setup: drop the player to spawn_y - 20 in a stone-filled column so the
tool has actual work to do (rather than just walking out of grass).

Pass criteria per iter:
  - last outcome doesn't start with FAILED
  - new_y >= initial_surface_y - 2  (reached sky)
  - total wall_s <= timeout
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
    cmd,
    pos,
    preflight,
    random_spawn,
    setup_clean,
)
from craft.tools import dispatch


DEFAULT_DEPTH = 20  # blocks below natural surface — under SURFACE_MAX_PER_CALL=40
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_PASS_RATE = 0.9
MAX_CALLS = 4  # safety cap on chunked-ascent retries


def _carve_underground_pocket(
    px: int, surface_y: int, depth: int, pz: int
) -> tuple[int, tuple, tuple]:
    """Carve a 1×3 air pocket at (px, surface_y - depth, pz) and stone-fill
    the column above so the tool has natural-feeling stone to dig up through.

    Returns (target_y, shell_lo, shell_hi) for cleanup.
    """
    target_y = surface_y - depth
    # Stone column from just above pocket up to ~3 blocks below surface, so
    # the tool's ascent has continuous solid material to clear (matches the
    # natural underground state after a descend()).
    shell_lo = (px, target_y + 3, pz)
    shell_hi = (px, surface_y - 1, pz)
    cmd(f"fill {shell_lo[0]} {shell_lo[1]} {shell_lo[2]} "
        f"{shell_hi[0]} {shell_hi[1]} {shell_hi[2]} minecraft:stone")
    # 1×3 air pocket for the player.
    cmd(f"fill {px} {target_y} {pz} {px} {target_y + 2} {pz} minecraft:air")
    # Stone floor so they don't fall out.
    cmd(f"setblock {px} {target_y - 1} {pz} minecraft:stone")
    time.sleep(0.3)
    return target_y, shell_lo, shell_hi


def _cleanup_pocket(lo: tuple, hi: tuple) -> None:
    cmd(f"fill {lo[0]} {lo[1]} {lo[2]} {hi[0]} {hi[1]} {hi[2]} minecraft:air")
    cmd("kill @e[type=item,distance=..32]")


def run_iter(
    rec: dict,
    *,
    depth: int,
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
    px = int(math.floor(start[0]))
    surface_y = int(math.floor(start[1]))
    pz = int(math.floor(start[2]))
    rec["surface_y"] = surface_y

    # Carve the underground pocket BEFORE setup_clean so the player tps
    # straight into air (in creative), avoiding suffocation when survival
    # kicks back in.
    target_y, lo, hi = _carve_underground_pocket(px, surface_y, depth, pz)
    rec["target_y"] = target_y
    rec["pocket_lo"] = list(lo)
    rec["pocket_hi"] = list(hi)
    if verbose:
        print(f"[test] surface_y={surface_y}, target_y={target_y}", flush=True)

    setup_clean((px, target_y, pz))

    # Re-read pos to confirm the player landed underground.
    after_setup = pos()
    if after_setup is None:
        _cleanup_pocket(lo, hi)
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_setup_position"
        return
    sx, sy, sz = after_setup
    rec["start_pos"] = [sx, sy, sz]

    cmd(f"give {PLAYER_NAME} minecraft:stone_pickaxe 1")
    cmd(f"give {PLAYER_NAME} minecraft:stone_shovel 1")
    time.sleep(0.3)

    t0 = time.monotonic()
    outcomes: list[str] = []
    calls = 0
    try:
        while calls < MAX_CALLS:
            o = dispatch("surface", json.dumps({}))
            outcomes.append(o)
            calls += 1
            if verbose:
                print(f"[test] call {calls}: {o}", flush=True)
            # Success signals from handle_surface.
            if o.startswith("surfaced") or o.startswith("already at surface"):
                break
            # PARTIAL ascent — keep going.
            if o.startswith("ascended") and "call surface() again" in o:
                continue
            # FAILED or unexpected — bail.
            break
    except Exception as e:
        outcomes.append(f"FAILED: dispatch threw {e!r}")
    elapsed = round(time.monotonic() - t0, 2)
    outcome = outcomes[-1] if outcomes else None
    rec["outcomes"] = outcomes
    rec["outcome"] = outcome
    rec["calls"] = calls
    rec["surface_wall_s"] = elapsed

    end = pos()
    if end is None:
        _cleanup_pocket(lo, hi)
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_position"
        return
    ex, ey, ez = end
    rec["end_pos"] = [ex, ey, ez]
    delta_y = ey - surface_y
    rec["delta_y"] = round(delta_y, 2)

    _cleanup_pocket(lo, hi)
    cmd(f"clear {PLAYER_NAME}")

    is_failed = outcome is None or outcome.startswith("FAILED")
    # Allow Δy = -2 since trees/grass can cap surface_y; tool reports when
    # within 1 of detected surface.
    reached_surface = ey >= surface_y - 2
    within_budget = elapsed <= timeout_s

    rec["checks"] = {
        "outcome_not_failed": not is_failed,
        "reached_surface": reached_surface,
        "within_timeout": within_budget,
    }
    rec["passed"] = (not is_failed) and reached_surface and within_budget
    if not rec["passed"]:
        if is_failed:
            rec["fail_reason"] = "outcome_failed"
        elif not reached_surface:
            rec["fail_reason"] = "did_not_reach_surface"
        else:
            rec["fail_reason"] = "timeout"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
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
    logger = TestLogger("surface", path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(
                    rec,
                    depth=args.depth,
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
