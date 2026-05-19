"""End-to-end test for handle_burrow.

Synthetic arena: 7×7 stone floor at y=99 with a 7-deep × 3-tall × 7-wide
stone wall to the east starting at x=5004. The player TPs to center
(5000, 100, 5000) facing east, then handle_burrow() is invoked directly
(it's a Python handler, not an HTTP route).

PASS when:
  - excavated cells 1..3 (x=5005..5007, y=100..101, z=5000) are air
  - seal cell 2 (x=5006, y=100..101, z=5000) is solid (placed cobble)
  - back-cavity cell 3 (x=5007, y=100..101, z=5000) is air
  - player position is inside back cavity
  - foyer cell 1 (x=5005, y=100..101, z=5000) is air (mob-occupiable
    pocket between outside and seal)

Run (against agent0):
  HOMUNCULUS_PORT=25570 MC_PLAYER_NAME=agent0 \\
  MC_SERVER_CMD_BASE=http://10.0.0.222:4747 \\
  .venv/bin/python -m e2e.test_burrow
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from craft.testkit import (
    HOMUNCULUS_BASE,
    PLAYER_NAME,
    TestLogger,
    cmd,
    pos,
    set_gamemode,
)
from craft.tools import (
    clear_burrow_state,
    get_burrow_state,
    handle_burrow,
    handle_expand_burrow,
)


ARENA_X = 5000
ARENA_Y = 100
ARENA_Z = 5000

# Wall starts directly east of the player (ARENA_X + 1), runs 7 cells.
# Burrow checks the *adjacent* cell; the player must be flush against the
# wall for any direction to be viable.
WALL_X0 = ARENA_X + 1
WALL_X1 = WALL_X0 + 6
WALL_Z0 = ARENA_Z - 3
WALL_Z1 = ARENA_Z + 3

# Expected geometry after a successful burrow facing east:
#   cell 1 (foyer)  = (ARENA_X + 1, y, ARENA_Z)
#   cell 2 (seal)   = (ARENA_X + 2, y, ARENA_Z)
#   cell 3 (back)   = (ARENA_X + 3, y, ARENA_Z)


def _setup_arena() -> None:
    # Clear a generous air box covering both the player area and the wall.
    cmd(f"fill {ARENA_X - 3} {ARENA_Y} {WALL_Z0 - 1} "
        f"{WALL_X1 + 1} {ARENA_Y + 3} {WALL_Z1 + 1} air")
    # 4-wide × 7-deep stone floor under player (x in [-3,0] relative to ARENA_X).
    cmd(f"fill {ARENA_X - 3} {ARENA_Y - 1} {WALL_Z0} "
        f"{ARENA_X} {ARENA_Y - 1} {WALL_Z1} stone")
    # Stone wall east of player area, 7-wide × 4-tall × 7-deep (covers
    # floor-1 through head+1 — fully solid hill for the tunnel to bite into).
    cmd(f"fill {WALL_X0} {ARENA_Y - 1} {WALL_Z0} "
        f"{WALL_X1} {ARENA_Y + 2} {WALL_Z1} stone")


def _verify_arena(retries: int = 4) -> tuple[bool, str]:
    """Confirm the platform + wall fills actually placed.

    Watches for the silent-fill-on-unloaded-chunk failure mode by scanning
    one sample cell from the platform and one from the wall.
    """
    for attempt in range(retries):
        scan = _scan(ARENA_X, ARENA_Y - 1, ARENA_Z, WALL_X0 + 1, ARENA_Y, ARENA_Z)
        if scan.get("success") is not False:
            platform_ok = _block_at(scan, ARENA_X, ARENA_Y - 1, ARENA_Z) is not None
            wall_ok = _block_at(scan, WALL_X0 + 1, ARENA_Y, ARENA_Z) is not None
            if platform_ok and wall_ok:
                return True, ""
            print(f"  [verify] attempt {attempt+1}: platform={platform_ok} wall={wall_ok}; "
                  f"re-filling and retrying...", flush=True)
        time.sleep(1.0)
        _setup_arena()
        time.sleep(0.8)
    return False, "platform/wall fills did not stick after retries"


def _teardown_arena() -> None:
    cmd(f"fill {ARENA_X - 3} {ARENA_Y - 1} {WALL_Z0 - 1} "
        f"{WALL_X1 + 1} {ARENA_Y + 3} {WALL_Z1 + 1} air")


def _scan(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> dict:
    r = requests.get(
        f"{HOMUNCULUS_BASE}/scan_blocks",
        params={"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2},
        timeout=10.0,
    )
    return r.json()


def _is_air(scan: dict, x: int, y: int, z: int) -> bool:
    """/scan_blocks omits air cells, so absence == air."""
    for b in scan.get("blocks", []):
        if b["x"] == x and b["y"] == y and b["z"] == z:
            return False
    return True


def _block_at(scan: dict, x: int, y: int, z: int) -> str | None:
    for b in scan.get("blocks", []):
        if b["x"] == x and b["y"] == y and b["z"] == z:
            return b.get("id")
    return None


def run_iter(rec: dict, verbose: bool = True, with_expand: bool = False) -> None:
    clear_burrow_state()

    # Load chunks first by TPing above the destination, then fill, then TP
    # again to settle on the floor. /fill silently no-ops on unloaded chunks.
    set_gamemode("creative")
    cmd(f"tp {PLAYER_NAME} {ARENA_X}.5 {ARENA_Y + 5} {ARENA_Z}.5 -90 0")
    time.sleep(1.5)
    _setup_arena()
    time.sleep(0.8)
    arena_ok, arena_reason = _verify_arena()
    if not arena_ok:
        rec["passed"] = False
        rec["fail_reason"] = f"arena_build_failed: {arena_reason}"
        return
    cmd(f"tp {PLAYER_NAME} {ARENA_X}.5 {ARENA_Y} {ARENA_Z}.5 -90 0")
    landed = False
    for _ in range(20):
        time.sleep(0.3)
        p = pos()
        if p is None:
            continue
        if (abs(p[0] - (ARENA_X + 0.5)) < 1.5
                and abs(p[2] - (ARENA_Z + 0.5)) < 1.5
                and abs(p[1] - ARENA_Y) < 0.5):
            landed = True
            break
    if not landed:
        rec["passed"] = False
        rec["fail_reason"] = f"player did not land on platform pos={pos()}"
        return
    set_gamemode("survival")
    cmd(f"clear {PLAYER_NAME}")
    # Seed with 6 cobble so the seal works even if tunnel drops are flaky.
    # (Tunnel of 6 stone normally → 6 cobble after pickup, but loose drops
    # can be missed if Baritone overshoots.)
    cmd(f"give {PLAYER_NAME} cobblestone 6")
    # Iron pickaxe so the tunnel mines stone quickly.
    cmd(f"give {PLAYER_NAME} iron_pickaxe 1")
    time.sleep(0.8)

    p = pos()
    if p is None:
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_position"
        return

    result_str = handle_burrow({})
    if verbose:
        print(f"[burrow] result: {result_str}")

    rec["player_pos_before"] = list(p)
    rec["burrow_result"] = result_str

    if result_str.startswith("FAILED") or result_str.startswith("PARTIAL"):
        rec["passed"] = False
        rec["fail_reason"] = f"handle_burrow non-success: {result_str}"
        return

    # Verify geometry. Scan the entire corridor + seal + back cavity.
    time.sleep(0.5)
    scan = _scan(ARENA_X + 1, ARENA_Y, ARENA_Z, ARENA_X + 3, ARENA_Y + 1, ARENA_Z)
    if scan.get("success") is False:
        rec["passed"] = False
        rec["fail_reason"] = f"verification scan failed: {scan.get('reason')}"
        return

    foyer_air = (_is_air(scan, ARENA_X + 1, ARENA_Y, ARENA_Z)
                 and _is_air(scan, ARENA_X + 1, ARENA_Y + 1, ARENA_Z))
    seal_foot = _block_at(scan, ARENA_X + 2, ARENA_Y, ARENA_Z)
    seal_head = _block_at(scan, ARENA_X + 2, ARENA_Y + 1, ARENA_Z)
    back_air = (_is_air(scan, ARENA_X + 3, ARENA_Y, ARENA_Z)
                and _is_air(scan, ARENA_X + 3, ARENA_Y + 1, ARENA_Z))

    rec["foyer_air"] = foyer_air
    rec["seal_foot"] = seal_foot
    rec["seal_head"] = seal_head
    rec["back_air"] = back_air

    p_after = pos()
    rec["player_pos_after"] = list(p_after) if p_after else None
    in_back_cavity = (
        p_after is not None
        and abs(p_after[0] - (ARENA_X + 3 + 0.5)) < 1.5
        and abs(p_after[2] - (ARENA_Z + 0.5)) < 1.5
    )
    rec["in_back_cavity"] = in_back_cavity

    bstate = get_burrow_state()
    rec["burrow_state"] = bstate

    failures = []
    if not foyer_air:
        failures.append(f"foyer (x={ARENA_X+1}) not air: {_block_at(scan, ARENA_X+1, ARENA_Y, ARENA_Z)}")
    if seal_foot is None:
        failures.append(f"seal foot (x={ARENA_X+2}) is air (expected placed block)")
    if seal_head is None:
        failures.append(f"seal head (x={ARENA_X+2}, y={ARENA_Y+1}) is air")
    if not back_air:
        failures.append(f"back cavity (x={ARENA_X+3}) not air")
    if not in_back_cavity:
        failures.append(f"player not in back cavity: pos={p_after}")
    if bstate is None:
        failures.append("burrow_state not set")

    rec["passed"] = len(failures) == 0
    rec["fail_reason"] = "; ".join(failures) if failures else None

    if not rec["passed"] or not with_expand:
        return

    # ── expand_burrow phase ────────────────────────────────────────────
    # Burrow PASS → call expand_burrow, verify the 2×3 alcove cells are
    # air, seal still cobble, ceiling/floor still stone.
    expand_str = handle_expand_burrow({})
    if verbose:
        print(f"[expand_burrow] result: {expand_str}")
    rec["expand_result"] = expand_str

    if expand_str.startswith("FAILED") or expand_str.startswith("PARTIAL"):
        rec["passed"] = False
        rec["fail_reason"] = f"handle_expand_burrow non-success: {expand_str}"
        return

    # Alcove (facing east) spans x=[5004,5005], z=[4999,5001], y=[100,101].
    # Sample interior, ceiling, floor, seal.
    time.sleep(0.5)
    scan2 = _scan(ARENA_X + 4, ARENA_Y - 1, ARENA_Z - 1,
                  ARENA_X + 5, ARENA_Y + 2, ARENA_Z + 1)
    if scan2.get("success") is False:
        rec["passed"] = False
        rec["fail_reason"] = f"post-expand scan failed: {scan2.get('reason')}"
        return

    alcove_cells = [
        (ARENA_X + 4, ARENA_Y, ARENA_Z - 1),
        (ARENA_X + 4, ARENA_Y, ARENA_Z + 1),
        (ARENA_X + 5, ARENA_Y, ARENA_Z),
        (ARENA_X + 5, ARENA_Y + 1, ARENA_Z),
    ]
    floor_cells = [
        (ARENA_X + 4, ARENA_Y - 1, ARENA_Z),
        (ARENA_X + 5, ARENA_Y - 1, ARENA_Z),
    ]
    ceiling_cells = [
        (ARENA_X + 4, ARENA_Y + 2, ARENA_Z),
        (ARENA_X + 5, ARENA_Y + 2, ARENA_Z),
    ]

    failures2 = []
    for (cx, cy, cz) in alcove_cells:
        if not _is_air(scan2, cx, cy, cz):
            failures2.append(f"alcove cell ({cx},{cy},{cz}) not air: "
                             f"{_block_at(scan2, cx, cy, cz)}")
    for (cx, cy, cz) in floor_cells:
        if _is_air(scan2, cx, cy, cz):
            failures2.append(f"floor ({cx},{cy},{cz}) carved away")
    for (cx, cy, cz) in ceiling_cells:
        if _is_air(scan2, cx, cy, cz):
            failures2.append(f"ceiling ({cx},{cy},{cz}) carved away")

    # Re-scan the seal to confirm still intact.
    seal_scan = _scan(ARENA_X + 2, ARENA_Y, ARENA_Z, ARENA_X + 2, ARENA_Y + 1, ARENA_Z)
    seal_still = (_block_at(seal_scan, ARENA_X + 2, ARENA_Y, ARENA_Z) is not None
                  and _block_at(seal_scan, ARENA_X + 2, ARENA_Y + 1, ARENA_Z) is not None)
    if not seal_still:
        failures2.append("seal carved away during expand")

    bstate2 = get_burrow_state()
    if bstate2 is None or "alcove_aabb" not in bstate2:
        failures2.append("alcove_aabb not set in burrow_state")

    rec["expand_burrow_state"] = bstate2
    rec["passed"] = len(failures2) == 0
    rec["fail_reason"] = "; ".join(failures2) if failures2 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep-arena", action="store_true",
                    help="skip teardown (useful for in-game inspection)")
    ap.add_argument("--with-expand", action="store_true",
                    help="chain expand_burrow after burrow and verify alcove")
    ap.add_argument("--pass-rate", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                    help="JSONL output path (default: derived from test name)")
    args = ap.parse_args()

    verbose = not args.quiet
    logger = TestLogger("burrow",
                        path=Path(args.out) if args.out else None)
    try:
        for i in range(args.iters):
            print(f"\n=== iter {i + 1}/{args.iters} (player={PLAYER_NAME}, "
                  f"homunculus={HOMUNCULUS_BASE}) ===")
            try:
                with logger.iter_record(i) as rec:
                    run_iter(rec, verbose=verbose, with_expand=args.with_expand)
            except Exception as e:
                print(f"[test] iter {i} raised: {e!r}", flush=True)
                continue
            tag = "PASS" if rec.get("passed") else "FAIL"
            reason = rec.get("fail_reason") or "ok"
            print(f"[{tag}] iter {i + 1}: {reason}")
            if verbose:
                print(json.dumps(
                    {k: v for k, v in rec.items() if k != "burrow_result"},
                    indent=2, default=str))
    finally:
        if not args.keep_arena:
            _teardown_arena()
            clear_burrow_state()

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
