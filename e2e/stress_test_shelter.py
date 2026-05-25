"""Stress-test the shelter build tool on random terrain, then ambush.

Per iteration:
  1. TP agent to a random (dx, dz) offset from anchor (default: current pos).
  2. /difficulty peaceful: clears hostiles, isolates BUILD-time from ambush.
  3. Clear inventory, give random throwaway block + random door + tools.
  4. Brief resistance/saturation so the fall doesn't kill them.
  5. Build shelter via craft.tools.handle_build_shelter.
  6. /difficulty easy: enables zombie damage.
  7. Run ambush (craft.ambush.ambush) — 17 baby zombies at the ring.
  8. Poll mobs+player via track_ambush for N seconds (breach detection).
  9. Record JSONL line; difficulty back to peaceful for cleanup.

Test failure modes captured:
  - build_error / build_seconds (completion-time pressure)
  - breach=True: any zombie reached interior AABB → wall/door failure
  - died=True: zombie killed player → shelter inadequate
  - skipped count: how many ring points were non-air (encased/underground info)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

from craft import ambush as _ambush_mod
from craft.testkit import HOMUNCULUS_BASE, PLAYER_NAME, SERVER_CMD_BASE
from craft.tools import (
    _DOOR_ITEMS,
    _SHELTER_THROWAWAY_ITEMS,
    _position,
    handle_build_shelter,
)
from craft.world import set_difficulty, set_gamemode, set_time


def _cmd(cmd: str, timeout: float = 5.0, *, capture: bool = True) -> dict:
    """POST /cmd, then fetch /log to surface the async server response.

    Returns {"cmd","ok","log","error"}. Prints a warning if `error` is true.
    """
    resp = _ambush_mod._server_cmd(
        SERVER_CMD_BASE, cmd, timeout=timeout, capture=capture,
    )
    if resp.get("error"):
        tail = (resp.get("log") or [""])[-1]
        print(f"[stress] ⚠ cmd failed: {cmd!r} :: {tail}", flush=True)
    return resp


def _stats() -> dict:
    r = requests.get(f"{HOMUNCULUS_BASE}/stats", timeout=5)
    r.raise_for_status()
    return r.json()


def _deaths() -> list[dict]:
    r = requests.get(f"{HOMUNCULUS_BASE}/deaths", timeout=5)
    r.raise_for_status()
    return r.json().get("deaths", [])


def _pos_xyz() -> tuple[int, int, int] | None:
    try:
        p = _position()
    except Exception:
        return None
    if p.get("success") is False:
        return None
    try:
        # Block-coord truncation: MC entity x/z are float center-of-block,
        # block coord is floor(x).
        import math
        return (math.floor(p["x"]), int(round(p["y"])), math.floor(p["z"]))
    except (KeyError, TypeError):
        return None


def _pick_anchor() -> tuple[int, int, int]:
    """Use agent's current position as anchor (truncated to int block)."""
    p = _position()
    import math
    return (math.floor(p["x"]), int(round(p["y"])), math.floor(p["z"]))


def stress_iteration(
    iter_idx: int,
    anchor: tuple[int, int, int],
    *,
    rng: random.Random,
    range_xz: int,
    drop_y: int,
    ambush_seconds: int,
    mob: str,
    baby: bool,
    ambush_difficulty: str,
) -> dict:
    t_start = time.time()
    ax, _, az = anchor

    material = rng.choice(_SHELTER_THROWAWAY_ITEMS)
    door_item = rng.choice(_DOOR_ITEMS)

    print(f"\n===== [stress] iter {iter_idx}: anchor=({ax},_,{az}) range=±{range_xz} "
          f"material={material} door={door_item} =====", flush=True)

    cmd_errors: list[dict] = []

    def run_cmd(c: str) -> dict:
        r = _cmd(c)
        if r.get("error"):
            cmd_errors.append({"cmd": c, "log": r.get("log")})
        return r

    # --- Setup: peaceful difficulty wipes hostiles and isolates build phase ---
    set_difficulty("peaceful")
    # Force dawn at iter start so concurrent shelter builders all get a full
    # in-game day to construct. Without this, a fast builder finishing first
    # could trigger world transitions (or natural nightfall) that interrupt
    # slower builders mid-shelter. Idempotent across 3x concurrent agents.
    from craft.world import set_time as _set_time
    _set_time("dawn")
    run_cmd(f"effect clear {PLAYER_NAME}")
    run_cmd(f"effect give {PLAYER_NAME} minecraft:resistance 30 4 true")
    run_cmd(f"effect give {PLAYER_NAME} minecraft:saturation 3 9 true")
    run_cmd(f"effect give {PLAYER_NAME} minecraft:instant_health 1 9")

    # Biome-aware spawn with retry. Rejects in_water/in_lava/bad_biome/
    # stuck-no-ground/HP-drop-in-survival — same logic the rollouts and other
    # tests use. Naive single-shot TP previously landed agents in water at
    # y=49 or on dark_forest 6/25 floor footprints; this filters those out.
    from craft.spawn import random_spawn as _random_spawn
    spawn_result = _random_spawn(
        range_blocks=range_xz,
        homunculus_base=HOMUNCULUS_BASE,
        server_cmd_base=SERVER_CMD_BASE,
        player_name=PLAYER_NAME,
        anchor_xz=(ax, az),
        rng=rng,
        verbose=True,
    )
    dx, dz = spawn_result.get("offset") or (0, 0)
    tp_to = spawn_result.get("tp_to") or (ax, drop_y, az)
    tx, ty, tz = tp_to
    biome = spawn_result.get("biome")

    if not spawn_result.get("ok"):
        print(f"[stress] spawn-retry exhausted; returning early "
              f"(last={spawn_result.get('attempts')[-1] if spawn_result.get('attempts') else None})",
              flush=True)
        return {
            "iter": iter_idx,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tp_target": [tx, ty, tz],
            "tp_offset": [dx, dz],
            "biome": biome,
            "spawn_attempts": spawn_result.get("attempts"),
            "build_center": None,
            "post_build_pos": None,
            "material": material,
            "door_item": door_item,
            "build_seconds": 0.0,
            "build_result": None,
            "build_error": None,
            "aborted": True,
            "ambush": None,
            "track": None,
            "died": False,
            "death": None,
            "cmd_errors": cmd_errors,
            "wall_seconds": round(time.time() - t_start, 1),
            "passed": False,
            "fail_reason": "spawn_retry_exhausted",
        }

    # random_spawn leaves the player in survival mode + cleared inventory.
    # Briefly back to creative to grant test inventory (material + tools).
    set_gamemode("creative")
    run_cmd(f"give {PLAYER_NAME} {material} 128")
    run_cmd(f"give {PLAYER_NAME} {door_item} 1")
    run_cmd(f"give {PLAYER_NAME} minecraft:diamond_pickaxe 1")
    run_cmd(f"give {PLAYER_NAME} minecraft:diamond_shovel 1")
    # Axe to chew through leaves+wood when the drop-in lands the bot in a
    # tree (common at forested biomes that pass random_spawn's bad-biome filter).
    run_cmd(f"give {PLAYER_NAME} minecraft:diamond_axe 1")

    set_gamemode("survival")
    time.sleep(0.5)  # brief settle; random_spawn already verified on_ground

    pre_xyz = _pos_xyz()
    pre_stats = _stats()
    print(f"[stress] pre-build: pos={pre_xyz} hp={pre_stats.get('health')}",
          flush=True)

    # ---- Build phase (peaceful) ----
    t_build_start = time.time()
    try:
        build_result_text = handle_build_shelter({})
        build_err = None
    except Exception as e:
        build_result_text = ""
        build_err = repr(e)
    t_build = time.time() - t_build_start
    print(f"[stress] build done in {t_build:.1f}s (err={build_err})", flush=True)
    if build_result_text:
        head = build_result_text.replace("\n", " | ")[:240]
        print(f"[stress] build_result: {head}", flush=True)

    post_xyz = _pos_xyz()
    print(f"[stress] post-build: pos={post_xyz}", flush=True)

    # If build aborted (unsuitable terrain), short-circuit the iter.
    # Ambush + track would just measure how fast the player drowns or
    # gets killed in open terrain — not useful shelter signal.
    if (build_result_text or "").startswith("ABORTED"):
        print(f"[stress] aborted — skipping ambush+track", flush=True)
        set_difficulty("peaceful")
        return {
            "iter": iter_idx,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tp_target": [tx, ty, tz],
            "tp_offset": [dx, dz],
            "biome": biome,
            "build_center": list(pre_xyz) if pre_xyz else None,
            "post_build_pos": list(post_xyz) if post_xyz else None,
            "material": material,
            "door_item": door_item,
            "build_seconds": round(t_build, 1),
            "build_result": build_result_text,
            "build_error": build_err,
            "aborted": True,
            "ambush": None,
            "track": None,
            "died": False,
            "death": None,
            "cmd_errors": cmd_errors,
            "wall_seconds": round(time.time() - t_start, 1),
            # Aborted = environmental rejection (terrain wouldn't fit a shelter),
            # not a shelter-tool failure. Counted as a non-pass for the suite
            # but a different fail_reason than a real breach/death.
            "passed": False,
            "fail_reason": "build_aborted_unsuitable_terrain",
        }

    # Build CENTER is what the shelter is anchored on (pre-build pos). If
    # final_goto failed, post_xyz is in the doorway — DON'T anchor the
    # ambush ring on post_xyz, the ring would be miscentered.
    ambush_anchor = pre_xyz if pre_xyz is not None else (tx, drop_y, tz)

    # ---- Ambush phase (easy → zombies damage) ----
    set_difficulty(ambush_difficulty)  # type: ignore[arg-type]
    # Night → zombies don't burn in sunlight. 60s ambush << 20-min MC day,
    # so a single set holds for the whole window.
    set_time("midnight")
    # Top up player so the test measures shelter, not residual fall damage.
    run_cmd(f"effect give {PLAYER_NAME} minecraft:instant_health 1 9")
    run_cmd(f"effect clear {PLAYER_NAME} minecraft:resistance")
    time.sleep(0.5)

    # Capture latest death timestamp BEFORE ambush as the baseline.
    deaths_before = _deaths()
    death_baseline_ms = max((d.get("timestamp", 0) for d in deaths_before),
                            default=0)

    try:
        amb = _ambush_mod.ambush(
            PLAYER_NAME,
            anchor=ambush_anchor,
            mob=mob,
            baby=baby,
            verbose=True,
        )
        ambush_err = amb.get("error")
    except Exception as e:
        amb = {"spawned": [], "skipped": [], "attempts": 17}
        ambush_err = repr(e)
    print(f"[stress] ambush spawned={len(amb.get('spawned', []))}/"
          f"{amb.get('attempts')} skipped={len(amb.get('skipped', []))}"
          + (f" (err={ambush_err})" if ambush_err else ""), flush=True)

    # ---- Track phase: poll mobs + player + deaths ----
    try:
        track = _ambush_mod.track_ambush(
            ambush_anchor,
            duration=ambush_seconds,
            poll_interval=2.0,
            mob=mob,
            death_baseline_ms=death_baseline_ms,
            verbose=True,
        )
        track_err = None
    except Exception as e:
        track = {
            "timeline": [], "max_in_interior": 0, "breach": False,
            "breach_first_t": None, "final_alive": 0,
            "player_hp_min": None, "player_hp_final": None,
            "per_uuid": {}, "ended_early": False,
            "death": None, "interrupted": False,
        }
        track_err = repr(e)

    # ---- Verdict: track["death"] is authoritative (catches respawn races) ----
    death_record = track.get("death")
    died = death_record is not None

    breach = track["breach"]
    print(f"[stress] verdict: died={died} breach={breach} "
          f"hp_min={track['player_hp_min']} final_alive={track['final_alive']}",
          flush=True)

    # ---- Cleanup ----
    set_difficulty("peaceful")  # wipes any remaining hostiles
    if died:
        # Respawn-ish: spawnpoint near anchor, instant heal.
        run_cmd(f"spawnpoint {PLAYER_NAME} {ax} 90 {az}")
        run_cmd(f"effect give {PLAYER_NAME} minecraft:instant_health 1 9")

    spawned_any = bool(amb.get("spawned") or [])
    # Pass criteria mirror the previous run_tests.py shelter_jsonl judge:
    # build worked, player survived, no zombie reached interior, ambush
    # actually exercised the test (a 0-spawn iter is a degenerate "pass"
    # that hides bugs).
    passed = (
        not build_err
        and not died
        and not breach
        and spawned_any
    )
    fail_reason = None
    if not passed:
        if build_err:
            fail_reason = "build_error"
        elif died:
            fail_reason = "player_died"
        elif breach:
            fail_reason = "shelter_breached"
        elif not spawned_any:
            fail_reason = "ambush_spawned_zero_mobs"

    return {
        "iter": iter_idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tp_target": [tx, ty, tz],
        "tp_offset": [dx, dz],
        "biome": biome,
        "build_center": list(pre_xyz) if pre_xyz else None,
        "post_build_pos": list(post_xyz) if post_xyz else None,
        "material": material,
        "door_item": door_item,
        "build_seconds": round(t_build, 1),
        "build_result": build_result_text,
        "build_error": build_err,
        "ambush": {
            "mob": mob, "baby": baby,
            "attempts": amb.get("attempts"),
            "spawned": amb.get("spawned"),
            "skipped": amb.get("skipped"),
            "error": ambush_err,
        },
        "track": {
            "breach": breach,
            "breach_first_t": track["breach_first_t"],
            "max_in_interior": track["max_in_interior"],
            "final_alive": track["final_alive"],
            "player_hp_min": track["player_hp_min"],
            "player_hp_final": track["player_hp_final"],
            "ended_early": track["ended_early"],
            "early_stop_reason": track.get("early_stop_reason"),
            "interrupted": track.get("interrupted", False),
            "timeline": track["timeline"],
            "per_uuid": track["per_uuid"],
            "error": track_err,
        },
        "died": died,
        "death": death_record,
        "cmd_errors": cmd_errors,
        "wall_seconds": round(time.time() - t_start, 1),
        "passed": passed,
        "fail_reason": fail_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--range", type=int, default=1500,
                    help="random ±xz offset range from anchor (blocks)")
    ap.add_argument("--anchor", type=int, nargs=3, metavar=("X", "Y", "Z"),
                    help="anchor (x,y,z); default = agent's current pos")
    ap.add_argument("--drop-y", type=int, default=100,
                    help="Y to TP player to (default 100, falls to surface)")
    ap.add_argument("--ambush-seconds", type=int, default=60)
    ap.add_argument("--ambush-difficulty", default="easy",
                    choices=("easy", "normal", "hard"),
                    help="difficulty during ambush phase (build is peaceful)")
    ap.add_argument("--mob", default="minecraft:zombie")
    ap.add_argument("--no-baby", dest="baby", action="store_false")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Wurst preflight: fresh agent JVMs boot with all hacks off. Tests
    # silently fail in subtle ways (AutoEat etc.) without this, so we run
    # the same preflight as the other tests. Reaches HOMUNCULUS_BASE which
    # is env-driven, so a concurrent runner targets the right agent.
    from craft.testkit import preflight as _preflight
    err = _preflight()
    if err is not None:
        print(f"[stress] preflight FAIL: {err}", flush=True)
        return 2

    rng = random.Random(args.seed)
    anchor = tuple(args.anchor) if args.anchor else _pick_anchor()
    out_path = Path(args.out) if args.out \
        else Path(f"results/stress-shelter-{int(time.time())}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stress] anchor={anchor} range=±{args.range} iters={args.iters} "
          f"drop_y={args.drop_y} seed={args.seed} → {out_path}", flush=True)

    with out_path.open("w") as f:
        for i in range(args.iters):
            try:
                result = stress_iteration(
                    i, anchor,
                    rng=rng,
                    range_xz=args.range,
                    drop_y=args.drop_y,
                    ambush_seconds=args.ambush_seconds,
                    mob=args.mob,
                    baby=args.baby,
                    ambush_difficulty=args.ambush_difficulty,
                )
            except Exception as e:
                result = {"iter": i, "fatal_error": repr(e),
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "passed": False, "fail_reason": "fatal_error"}
                print(f"[stress] FATAL iter {i}: {e}", flush=True)
            f.write(json.dumps(result) + "\n")
            f.flush()
            if result.get("aborted"):
                print(f"[stress] iter {i} verdict: ABORTED "
                      f"wall={result.get('wall_seconds')}s", flush=True)
            else:
                print(f"[stress] iter {i} verdict: "
                      f"died={result.get('died')} "
                      f"breach={(result.get('track') or {}).get('breach')} "
                      f"spawned={len((result.get('ambush') or {}).get('spawned') or [])}/"
                      f"{(result.get('ambush') or {}).get('attempts')} "
                      f"wall={result.get('wall_seconds')}s", flush=True)

    print(f"\n[stress] done → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
