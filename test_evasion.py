"""Multi-iter test of the outside-the-handler evasion path.

Per iteration:
  1. (Optional) biome-aware random TP for sample diversity.
  2. setup_clean — peaceful, heal, +resistance, TP to anchor.
  3. build_arena — long stone+air corridor covering anchor + displaced ring.
  4. /evasion/arm at the anchor.
  5. Displace east by displace_dx, /difficulty easy, ambush adult zombies.
  6. Poll /evasion/status: capture fired-latency, flee_state transitions.
  7. Settle, verify position back at anchor.
  8. Cleanup; write JSONL record.

Pass criteria per iter:
  - fired_within_timeout: watcher saw a hostile-mob hit within FIRE_TIMEOUT_S
  - flee_state_arrived: terminal state is "arrived" (not "timeout"/"failed")
  - player_within_tolerance_of_anchor: manhattan distance ≤ ARRIVAL_TOL

Overall verdict: pass-rate across iters ≥ --pass-rate.

--no-baby is load-bearing: adult zombies are slower than the player so the
flee outruns pursuit. Baby zombies would pile on during the flee and break
timing reliability.
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from pathlib import Path

import requests

from craft.ambush import ambush
from craft.testkit import (
    HOMUNCULUS_BASE,
    PLAYER_NAME,
    SERVER_CMD_BASE,
    TestLogger,
    build_arena,
    cmd,
    pos,
    preflight,
    random_spawn,
    set_difficulty,
    set_gamemode,
    setup_clean,
    stats,
)


# Fire-detection budget. Watcher polls at ~20 Hz, so detection should be
# well under 5s in practice; 15s is the give-up threshold.
FIRE_TIMEOUT_S = 15.0
# Java-side flee caps at 60s; we wait beyond that for the terminal state.
FLEE_TIMEOUT_S = 75.0
# Manhattan tolerance for "player is at the anchor."
ARRIVAL_TOL = 3.0
# Default east displacement: far enough for a real flee path, close enough
# that chunks are loaded + the ambush ring doesn't overlap anchor.
DEFAULT_DISPLACE_DX = 15
# Pass-rate threshold. Evasion is more deterministic than shelter but less
# than mine_wood (depends on Baritone goto timing) — hold to 0.9.
DEFAULT_PASS_RATE = 0.9


def _evasion_status() -> dict | None:
    try:
        r = requests.get(f"{HOMUNCULUS_BASE}/evasion/status", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _evasion_arm(x: float, y: float, z: float) -> bool:
    try:
        r = requests.post(
            f"{HOMUNCULUS_BASE}/evasion/arm",
            json={"x": x, "y": y, "z": z},
            timeout=3.0,
        )
        return r.ok and r.json().get("success") is True
    except (requests.RequestException, ValueError):
        return False


def _evasion_disarm() -> None:
    try:
        requests.post(f"{HOMUNCULUS_BASE}/evasion/disarm", timeout=3.0)
    except requests.RequestException:
        pass


def _displace(anchor: tuple[int, int, int], dx: int) -> None:
    ax, ay, az = anchor
    cmd(f"tp {PLAYER_NAME} {ax + dx} {ay} {az}")
    time.sleep(1.0)


def _kill_zombies() -> None:
    cmd("kill @e[type=zombie,distance=..96]")


def run_iter(
    rec: dict,
    *,
    mob: str,
    displace_dx: int,
    poll_interval_s: float,
    spawn_range: int,
    rng: _random.Random,
    verbose: bool,
) -> None:
    """Run a single iteration and populate `rec` in place."""
    if spawn_range > 0:
        spawn_result = random_spawn(
            range_blocks=spawn_range, rng=rng, verbose=verbose)
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
    anchor = (int(math.floor(start[0])),
              int(math.floor(start[1])),
              int(math.floor(start[2])))
    rec["anchor"] = list(anchor)
    rec["displace_dx"] = displace_dx
    if verbose:
        print(f"[test] anchor = {anchor}", flush=True)

    # Resistance 4 keeps the player alive through the zombie volley long
    # enough to test the flee path — we're testing the trigger + flee, not
    # survival under fire.
    setup_clean(anchor, extra_effects=("minecraft:resistance 120 4 true",))

    # Carve a corridor: anchor through displaced ring, with margin on z.
    # x: anchor.x - 2  .. anchor.x + displace_dx + 5
    # z: anchor.z - 5  .. anchor.z + 5
    # Asymmetric arena; pass x and z radii to build_arena.
    half_x_low = 2
    half_x_high = displace_dx + 5
    x_radius_sym = max(half_x_low, half_x_high)
    # build_arena is centered on anchor, so use the larger radius. Some
    # west-of-anchor floor gets carved harmlessly.
    build_arena(anchor, x_radius=x_radius_sym, z_radius=5)
    # TP back to anchor in case the air-fill dropped the player.
    set_gamemode("creative")
    cmd(f"tp {PLAYER_NAME} {anchor[0]} {anchor[1]} {anchor[2]}")
    time.sleep(0.5)
    set_gamemode("survival")
    time.sleep(0.5)

    if not _evasion_arm(float(anchor[0]) + 0.5, float(anchor[1]),
                        float(anchor[2]) + 0.5):
        rec["passed"] = False
        rec["fail_reason"] = "evasion_arm_failed"
        return
    if verbose:
        print(f"[test] evasion armed at {anchor}", flush=True)

    _displace(anchor, displace_dx)
    pos_at_ambush = pos()
    if pos_at_ambush is None:
        _evasion_disarm()
        rec["passed"] = False
        rec["fail_reason"] = "could_not_read_post_displace_position"
        return
    rec["pos_at_ambush"] = list(pos_at_ambush)

    set_difficulty("easy")
    time.sleep(0.5)

    ambush_t0 = time.monotonic()
    amb_result = ambush(
        agent_name=PLAYER_NAME,
        anchor=None,
        mob=mob,
        baby=False,
        verbose=verbose,
        homunculus_base=HOMUNCULUS_BASE,
        server_cmd_base=SERVER_CMD_BASE,
    )
    ambush_dt = round(time.monotonic() - ambush_t0, 2)
    spawned_count = len(amb_result.get("spawned", []))
    rec["ambush_spawned"] = spawned_count
    rec["ambush_attempts"] = amb_result.get("attempts")
    rec["ambush_wall_s"] = ambush_dt
    rec["ambush_spawn_coords"] = amb_result.get("spawned", [])
    if verbose:
        print(f"[test] ambush done in {ambush_dt}s — "
              f"{spawned_count} spawned, {len(amb_result.get('skipped', []))} skipped",
              flush=True)

    if spawned_count == 0:
        _kill_zombies()
        _evasion_disarm()
        set_difficulty("peaceful")
        rec["passed"] = False
        rec["fail_reason"] = "ambush_spawned_zero_mobs"
        return

    fire_t0 = time.monotonic()
    fired_at: float | None = None
    flee_terminal: str | None = None
    transitions: list[tuple[float, str, bool]] = []
    last_state = "idle"
    last_fired = False
    final_status: dict | None = None
    while True:
        elapsed = time.monotonic() - fire_t0
        if fired_at is None and elapsed > FIRE_TIMEOUT_S:
            break
        if fired_at is not None and elapsed - fired_at > FLEE_TIMEOUT_S:
            break
        s = _evasion_status()
        if s is None:
            time.sleep(poll_interval_s)
            continue
        final_status = s
        st = s.get("flee_state", "idle")
        fired = bool(s.get("fired"))
        if st != last_state or fired != last_fired:
            transitions.append((round(elapsed, 2), st, fired))
            last_state = st
            last_fired = fired
            if verbose:
                print(f"[test] t={elapsed:5.2f}s  fired={fired}  "
                      f"flee_state={st}  attackers={s.get('attackers')}",
                      flush=True)
        if fired and fired_at is None:
            fired_at = elapsed
        if st in ("arrived", "timeout", "failed"):
            flee_terminal = st
            break
        time.sleep(poll_interval_s)

    time.sleep(0.5)
    end_pos = pos()
    end_stats = stats() or {}
    _kill_zombies()
    _evasion_disarm()
    set_difficulty("peaceful")

    if end_pos is not None:
        ex, ey, ez = end_pos
        dist_to_anchor = (abs(ex - (anchor[0] + 0.5))
                          + abs(ey - anchor[1])
                          + abs(ez - (anchor[2] + 0.5)))
    else:
        dist_to_anchor = None

    pass_fired = fired_at is not None
    pass_arrived = flee_terminal == "arrived"
    pass_anchor = (dist_to_anchor is not None and dist_to_anchor <= ARRIVAL_TOL)

    rec["end_pos"] = list(end_pos) if end_pos else None
    rec["end_hp"] = end_stats.get("health")
    rec["mob"] = mob
    rec["fired_latency_s"] = fired_at
    rec["flee_terminal_state"] = flee_terminal
    rec["dist_to_anchor"] = dist_to_anchor
    rec["transitions"] = transitions
    rec["final_status"] = final_status
    rec["checks"] = {
        "fired_within_timeout": pass_fired,
        "flee_state_arrived": pass_arrived,
        "player_within_tolerance_of_anchor": pass_anchor,
    }
    rec["passed"] = pass_fired and pass_arrived and pass_anchor
    if not rec["passed"]:
        if not pass_fired:
            rec["fail_reason"] = "watcher_did_not_fire"
        elif not pass_arrived:
            rec["fail_reason"] = f"flee_terminal_{flee_terminal}"
        else:
            rec["fail_reason"] = "settled_outside_anchor_tolerance"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mob", default="minecraft:zombie")
    ap.add_argument("--displace", type=int, default=DEFAULT_DISPLACE_DX)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--spawn-range", type=int, default=0,
                    help="if >0, biome-aware random TP within ±range each iter")
    ap.add_argument("--pass-rate", type=float, default=DEFAULT_PASS_RATE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    err = preflight(require_paths=("/evasion/status",))
    if err is not None:
        print(f"[test] preflight FAIL: {err}", flush=True)
        return 2

    rng = _random.Random(args.seed)
    logger = TestLogger("evasion",
                        path=Path(args.out) if args.out else None)

    for i in range(args.iters):
        try:
            with logger.iter_record(i) as rec:
                run_iter(rec, mob=args.mob, displace_dx=args.displace,
                         poll_interval_s=args.poll_interval,
                         spawn_range=args.spawn_range, rng=rng,
                         verbose=not args.quiet)
        except Exception as e:
            print(f"[test] iter {i} raised: {e!r}", flush=True)

    summary = logger.summary()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rate"] >= args.pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
