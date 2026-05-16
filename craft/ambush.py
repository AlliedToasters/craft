"""Ambush primitive: spawn mobs around an agent in a 17-point ring.

The 17 spawn points trace the perimeter + roof of a default 5x2x5 shelter
(see zombie_spawn_points_y{0,2}.txt). Even on open ground (no shelter), the
ring is a useful "surround me with hostiles" debug tool — pass the agent
name and an optional mob/center override and it spawns the ring.

CLI:
    python -m craft.ambush                    # default: zombies, baby, around $MC_PLAYER_NAME
    python -m craft.ambush --mob skeleton     # adult skeletons
    python -m craft.ambush --no-baby
    python -m craft.ambush --anchor 12 64 -5  # override shelter center
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
from typing import Iterable

import requests

from craft.config import HOMUNCULUS_BASE, PLAYER_NAME, SERVER_CMD_BASE


# 12 ground-level points, 1 block outside each wall face.
# Coords relative to shelter center (px, py, pz); py is the shelter floor's
# air row (where the player stands).
_Y0_OFFSETS: tuple[tuple[int, int, int, str], ...] = (
    # North face (z = pz - 4)
    (-2, 0, -4, "n_w"), (0, 0, -4, "n_c"), (+2, 0, -4, "n_e"),
    # West face (x = px - 4)
    (-4, 0, -2, "w_n"), (-4, 0, 0, "w_c"), (-4, 0, +2, "w_s"),
    # East face (x = px + 4)
    (+4, 0, -2, "e_n"), (+4, 0, 0, "e_c"), (+4, 0, +2, "e_s"),
    # South face (z = pz + 4)
    (-2, 0, +4, "s_w"), (0, 0, +4, "s_c"), (+2, 0, +4, "s_e"),
)

# 5 elevated points: 4 directly above each wall-middle (y+2) and 1 on top of
# roof center (y+3, since the roof slab occupies y+2). y2 wall-middle points
# typically fail the air check because ceiling_extras occupy them — that's
# itself a useful test signal.
_Y2_OFFSETS: tuple[tuple[int, int, int, str], ...] = (
    (0, +2, -3, "top_n"),
    (-3, +2, 0, "top_w"),
    (0, +3, 0, "top_roof"),
    (+3, +2, 0, "top_e"),
    (0, +2, +3, "top_s"),
)


# MC console error patterns. The /cmd endpoint always returns {"ok": true};
# the actual feedback is async-written to the server console, fetched via
# /log. We sleep ~50ms (1 tick) between cmd and log, then take the tail.
_CMD_ERROR_PATTERNS = (
    "Unknown or incomplete command",
    "Incorrect argument for command",
    "Unable to",
    "No entity was found",
    "No player was found",
    "Could not ",        # "Could not parse...", "Could not teleport..."
    "Expected ",         # arg-validator messages
    "Invalid ",          # "Invalid position for teleport", "Invalid name", ...
)


def _server_log(base: str, n: int = 3, timeout: float = 3.0) -> list[str]:
    try:
        r = requests.get(f"{base}/log", params={"n": n}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("lines", []) or []
    except Exception:
        return []


def _server_cmd(
    base: str, cmd: str, *,
    timeout: float = 5.0,
    capture: bool = True,
    log_wait_s: float = 0.05,
    log_n: int = 2,
) -> dict:
    """POST to /cmd, then fetch /log to capture the async response.

    Returns: {"cmd": str, "ok": bool, "log": [str], "error": bool}.
    On error, callers should surface and decide.
    """
    r = requests.post(f"{base}/cmd", json={"cmd": cmd}, timeout=timeout)
    r.raise_for_status()
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    out = {"cmd": cmd, "ok": body.get("ok") is True}
    if capture:
        time.sleep(log_wait_s)
        lines = _server_log(base, n=log_n)
        out["log"] = lines
        joined = " | ".join(lines)
        out["error"] = any(p in joined for p in _CMD_ERROR_PATTERNS)
    return out


def _scan_blocks(
    base: str,
    box: tuple[int, int, int, int, int, int],
    timeout: float = 8.0,
) -> dict:
    x1, y1, z1, x2, y2, z2 = box
    params = {"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2}
    url = f"{base}/scan_blocks?" + urllib.parse.urlencode(params)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _position(base: str, timeout: float = 5.0) -> dict:
    r = requests.get(f"{base}/position", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _scan_entities(
    base: str, mob: str, *, radius: int = 64, limit: int = 64, timeout: float = 4.0,
) -> list[dict]:
    params = {"type": mob, "radius": radius, "limit": limit}
    url = f"{base}/scan_entities?" + urllib.parse.urlencode(params)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("success") is False:
        return []
    return body.get("entities", []) or []


def _stats(base: str, timeout: float = 4.0) -> dict:
    r = requests.get(f"{base}/stats", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _spawn_box(anchor: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    ax, ay, az = anchor
    return (ax - 4, ay, az - 4, ax + 4, ay + 3, az + 4)


def spawn_points(
    anchor: tuple[int, int, int],
) -> list[dict]:
    """Return the 17 spawn-point coords as dicts {x,y,z,tag,ring}."""
    ax, ay, az = anchor
    out = []
    for dx, dy, dz, tag in _Y0_OFFSETS:
        out.append({"x": ax + dx, "y": ay + dy, "z": az + dz,
                    "tag": tag, "ring": "y0"})
    for dx, dy, dz, tag in _Y2_OFFSETS:
        out.append({"x": ax + dx, "y": ay + dy, "z": az + dz,
                    "tag": tag, "ring": "y2"})
    return out


def ambush(
    agent_name: str = PLAYER_NAME,
    *,
    anchor: tuple[int, int, int] | None = None,
    mob: str = "minecraft:zombie",
    baby: bool = True,
    extra_nbt: str = "",
    server_cmd_base: str = SERVER_CMD_BASE,
    homunculus_base: str = HOMUNCULUS_BASE,
    verbose: bool = False,
) -> dict:
    """Spawn `mob` at every air cell in the 17-point ambush ring around the agent.

    Args:
        agent_name: player name used by the server (for the future when we
            target a specific agent via NBT — currently only used in messages).
        anchor: (x, y, z) center to anchor the ring around. If None, looks up
            the agent's current /position via homunculus.
        mob: namespaced mob id (e.g. "minecraft:zombie", "minecraft:skeleton").
        baby: if True, append `IsBaby:1b` to spawn NBT.
        extra_nbt: extra NBT to merge inside the `{}` payload (e.g. effects,
            equipment). Comma-prefixed automatically if non-empty.
        server_cmd_base: console-cmd HTTP base (default home server).
        homunculus_base: client-side mod HTTP base.
        verbose: print per-point spawn results.

    Returns:
        {
          "agent": str, "anchor": [x,y,z], "mob": str, "baby": bool,
          "attempts": 17,
          "spawned": [{"pos":[x,y,z], "tag":..., "ring":...}, ...],
          "skipped": [{"pos":[x,y,z], "tag":..., "ring":..., "reason":"non_air", "id":...}, ...],
        }
    """
    if anchor is None:
        pos = _position(homunculus_base)
        anchor = (
            int(round(pos["x"] - 0.5)),  # floor-rounding for block coord
            int(round(pos["y"])),
            int(round(pos["z"] - 0.5)),
        )
        if verbose:
            print(f"[ambush] anchor (from agent pos): {anchor}", flush=True)

    points = spawn_points(anchor)
    box = _spawn_box(anchor)
    scan = _scan_blocks(homunculus_base, box)
    if scan.get("success") is False:
        return {
            "agent": agent_name, "anchor": list(anchor), "mob": mob, "baby": baby,
            "attempts": len(points), "spawned": [], "skipped": [],
            "error": f"scan_blocks failed: {scan.get('reason')}: {scan.get('message')}",
        }
    solid_by_pos = {(b["x"], b["y"], b["z"]): b["id"] for b in scan.get("blocks", [])}

    nbt_bits: list[str] = []
    if baby:
        nbt_bits.append("IsBaby:1b")
    if extra_nbt:
        nbt_bits.append(extra_nbt.strip().strip(","))
    nbt = "{" + ",".join(nbt_bits) + "}" if nbt_bits else ""

    spawned: list[dict] = []
    skipped: list[dict] = []
    for p in points:
        key = (p["x"], p["y"], p["z"])
        if key in solid_by_pos:
            skipped.append({
                "pos": [p["x"], p["y"], p["z"]],
                "tag": p["tag"], "ring": p["ring"],
                "reason": "non_air", "id": solid_by_pos[key],
            })
            if verbose:
                print(f"[ambush]   skip {p['tag']:>8} @{key}  ({solid_by_pos[key]})", flush=True)
            continue
        cmd = f"summon {mob} {p['x']} {p['y']} {p['z']} {nbt}".strip()
        resp = _server_cmd(server_cmd_base, cmd)
        rec = {
            "pos": [p["x"], p["y"], p["z"]],
            "tag": p["tag"], "ring": p["ring"],
            "log": resp.get("log"),
            "error": resp.get("error", False),
        }
        spawned.append(rec)
        if verbose:
            err_tag = " ERROR" if rec["error"] else ""
            print(f"[ambush]  spawn {p['tag']:>8} @{key}{err_tag} :: "
                  f"{(resp.get('log') or [''])[-1]}", flush=True)

    return {
        "agent": agent_name, "anchor": list(anchor),
        "mob": mob, "baby": baby,
        "attempts": len(points),
        "spawned": spawned,
        "skipped": skipped,
    }


def track_ambush(
    anchor: tuple[int, int, int],
    *,
    duration: float = 60.0,
    poll_interval: float = 2.0,
    mob: str = "minecraft:zombie",
    homunculus_base: str = HOMUNCULUS_BASE,
    interior_xz_radius: int = 2,
    interior_y_lo_offset: int = 0,
    interior_y_hi_offset: int = 1,
    death_baseline_ms: int | None = None,
    early_stop_grace_s: float = 20.0,
    verbose: bool = False,
) -> dict:
    """Poll mob positions + player HP for `duration` seconds.

    `anchor` should be the shelter build-center (the position the shelter
    was built around) — NOT post-build player position, which may be in
    the doorway or otherwise off-center if final_goto failed.

    Breach detection: an entity is "inside" the shelter when the BLOCK
    its foot occupies (math.floor of its (x,y,z) position) lies in:
    ax±interior_xz_radius, ay+[lo..hi], az±interior_xz_radius.
    Defaults match the 5x2x5 shelter cavity (interior_xz_radius=2, y in
    [py, py+1]). Strict block-occupancy — no AABB inflation — so a zombie
    standing in the doorway threshold (z=az-3) does NOT count as inside.

    Persistence requirement: a uuid must be observed inside for ≥2
    *consecutive* polls before it counts as a confirmed breach. Single-
    poll blips (mob path-passes through cavity edge, brief AABB clip)
    don't trip `breach`. If a uuid is missing from a scan, its consec
    counter resets to 0.

    Death detection: if `death_baseline_ms` is provided, polls /deaths
    each tick. Any death record with timestamp > baseline ends the loop
    immediately with `death` populated. This is the authoritative signal
    — HP polling can miss a death-then-respawn between samples.

    Returns:
        {
          "anchor": [x,y,z], "mob": str, "duration": float,
          "timeline": [{t, alive, in_interior, hp, breach_uuids}],
          "max_in_interior": int,
          "breach": bool,
          "breach_first_t": float | None,
          "final_alive": int,
          "player_hp_min": float | None,
          "player_hp_final": float | None,
          "per_uuid": { uuid: {first_t, last_t, last_pos, last_hp, min_hp,
                                ever_in_interior,           # True iff confirmed
                                consec_in,                  # current consec count
                                breach_first_t,             # first poll observed inside
                                breach_first_pos,           # entity pos at breach_first_t
                                breach_first_hp_player,     # player HP at breach_first_t
                                confirmed_t,                # poll when promoted to breach
                                confirmed_pos} },           # entity pos at confirmed_t
          "ended_early": bool,    # broke on death/HP/interrupt
          "death": dict | None,   # /deaths record if we caught a new one
          "interrupted": bool,    # True if KeyboardInterrupt mid-loop
        }
    """
    ax, ay, az = anchor
    x_lo = ax - interior_xz_radius
    x_hi = ax + interior_xz_radius
    z_lo = az - interior_xz_radius
    z_hi = az + interior_xz_radius
    y_lo = ay + interior_y_lo_offset
    y_hi = ay + interior_y_hi_offset

    def inside(px: float, py: float, pz: float) -> bool:
        # Floor entity position to block coords, then strict containment.
        # Old code inflated the AABB by 0.5 to catch bbox overlap, but that
        # falsely flagged zombies standing at the door threshold (z=az-3,
        # outside the cavity) and zombies one block beyond a wall.
        bx, by, bz = math.floor(px), math.floor(py), math.floor(pz)
        return (x_lo <= bx <= x_hi
                and z_lo <= bz <= z_hi
                and y_lo <= by <= y_hi)

    t_start = time.time()
    deadline = t_start + duration
    timeline: list[dict] = []
    per_uuid: dict[str, dict] = {}
    player_hp_min: float | None = None
    player_hp_max: float | None = None  # initial reading; used by early-stop
    player_hp_final: float | None = None
    breach_first_t: float | None = None
    ended_early = False
    early_stop_reason: str | None = None
    death_record: dict | None = None
    interrupted = False

    try:
        while time.time() < deadline:
            t = round(time.time() - t_start, 1)
            try:
                zombies = _scan_entities(homunculus_base, mob, radius=64, limit=64)
            except Exception:
                zombies = []
            try:
                stats = _stats(homunculus_base)
                hp = stats.get("health")
            except Exception:
                hp = None

            in_int_now = 0
            breach_uuids: list[str] = []
            seen_uuids: set[str] = set()
            for z in zombies:
                uuid = z["uuid"]
                seen_uuids.add(uuid)
                pos = z.get("position", [None, None, None])
                health = z.get("health")
                here_inside = pos[0] is not None and inside(pos[0], pos[1], pos[2])

                rec = per_uuid.get(uuid)
                if rec is None:
                    rec = {
                        "first_t": t, "last_t": t,
                        "last_pos": pos, "last_hp": health,
                        "min_hp": health, "ever_in_interior": False,
                        "consec_in": 0,
                        "breach_first_t": None,
                        "breach_first_pos": None,
                        "breach_first_hp_player": None,
                        "confirmed_t": None,
                        "confirmed_pos": None,
                    }
                    per_uuid[uuid] = rec
                rec["last_t"] = t
                rec["last_pos"] = pos
                rec["last_hp"] = health
                if health is not None and (rec["min_hp"] is None or health < rec["min_hp"]):
                    rec["min_hp"] = health

                if here_inside:
                    rec["consec_in"] += 1
                    if rec["breach_first_t"] is None:
                        rec["breach_first_t"] = t
                        rec["breach_first_pos"] = list(pos)
                        rec["breach_first_hp_player"] = hp
                    # Promote to confirmed breach on the 2nd consecutive
                    # in-interior poll. ever_in_interior + the run-level
                    # breach_first_t reflect *confirmed* breaches only.
                    if not rec["ever_in_interior"] and rec["consec_in"] >= 2:
                        rec["ever_in_interior"] = True
                        rec["confirmed_t"] = t
                        rec["confirmed_pos"] = list(pos)
                        if breach_first_t is None:
                            breach_first_t = t
                else:
                    rec["consec_in"] = 0

                # in_int_now counts uuids that are confirmed AND currently
                # in the cavity — this is the live-threat signal the
                # early-stop "shelter held" check needs.
                if rec["ever_in_interior"] and here_inside:
                    in_int_now += 1
                    breach_uuids.append(uuid)

            # Any uuid not observed this poll loses its consec streak — a
            # zombie that walked out (or despawned, or fled scan range)
            # must rebuild from zero to count as a new confirmed breach.
            for uuid, rec in per_uuid.items():
                if uuid not in seen_uuids:
                    rec["consec_in"] = 0

            if hp is not None:
                if player_hp_min is None or hp < player_hp_min:
                    player_hp_min = hp
                if player_hp_max is None:
                    player_hp_max = hp
                player_hp_final = hp

            sample = {
                "t": t, "alive": len(zombies),
                "in_interior": in_int_now, "hp": hp,
            }
            if breach_uuids:
                sample["breach_uuids"] = breach_uuids
            timeline.append(sample)
            if verbose:
                tag = " BREACH" if in_int_now else ""
                print(f"[track] t={t:5.1f}s alive={len(zombies):2d} "
                      f"interior={in_int_now:2d} hp={hp}{tag}", flush=True)

            # Authoritative death check — /deaths catches death-then-respawn
            # between HP polls.
            if death_baseline_ms is not None:
                try:
                    r = requests.get(f"{homunculus_base}/deaths", timeout=3)
                    for d in r.json().get("deaths", []) or []:
                        if d.get("timestamp", 0) > death_baseline_ms:
                            death_record = d
                            break
                except Exception:
                    pass
                if death_record is not None:
                    if verbose:
                        print(f"[track] DEATH: {death_record.get('message')}",
                              flush=True)
                    ended_early = True
                    break

            if hp is not None and hp <= 0:
                ended_early = True
                early_stop_reason = "player_dead"
                break

            # Early stop on "shelter held": after the grace period, if the
            # player hasn't taken any damage AND no zombie is currently in
            # the interior AND there are mobs to fight, the shelter is
            # functionally protective — no need to wait out the full window.
            if (
                t >= early_stop_grace_s
                and len(zombies) > 0
                and in_int_now == 0
                and player_hp_min is not None
                and player_hp_max is not None
                and player_hp_min >= player_hp_max - 0.01
            ):
                if verbose:
                    print(f"[track] early-stop: shelter held "
                          f"(HP {player_hp_min}, {len(zombies)} alive, "
                          f"no interior breach at t={t}s)", flush=True)
                ended_early = True
                early_stop_reason = "shelter_held"
                break

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        interrupted = True
        ended_early = True
        if verbose:
            print("[track] KeyboardInterrupt — returning partial timeline",
                  flush=True)

    max_in_interior = max((s["in_interior"] for s in timeline), default=0)
    if interrupted:
        early_stop_reason = early_stop_reason or "interrupted"
    return {
        "anchor": list(anchor), "mob": mob, "duration": duration,
        "timeline": timeline,
        "max_in_interior": max_in_interior,
        "breach": max_in_interior > 0,
        "breach_first_t": breach_first_t,
        "final_alive": timeline[-1]["alive"] if timeline else 0,
        "player_hp_min": player_hp_min,
        "player_hp_final": player_hp_final,
        "per_uuid": per_uuid,
        "ended_early": ended_early,
        "early_stop_reason": early_stop_reason,
        "death": death_record,
        "interrupted": interrupted,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from craft.config import PLAYER_NAME as _DEFAULT_AGENT
    ap.add_argument("agent", nargs="?", default=_DEFAULT_AGENT,
                    help=f"agent (player) name (default: {_DEFAULT_AGENT}, from $MC_PLAYER_NAME)")
    ap.add_argument("--anchor", type=int, nargs=3, metavar=("X", "Y", "Z"),
                    help="override anchor center; else use agent's current pos")
    ap.add_argument("--mob", default="minecraft:zombie",
                    help="namespaced mob id (default: minecraft:zombie)")
    ap.add_argument("--no-baby", dest="baby", action="store_false",
                    help="spawn adults (default: babies)")
    ap.add_argument("--extra-nbt", default="",
                    help="extra NBT inside the {} payload")
    ap.add_argument("--track", type=float, default=0.0,
                    help="if >0, after spawning poll mobs+player for N seconds")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    anchor = tuple(args.anchor) if args.anchor else None
    result = ambush(
        args.agent,
        anchor=anchor,
        mob=args.mob,
        baby=args.baby,
        extra_nbt=args.extra_nbt,
        verbose=not args.quiet,
    )

    if args.track > 0 and "error" not in result:
        track = track_ambush(
            tuple(result["anchor"]),
            duration=args.track,
            poll_interval=args.poll_interval,
            mob=args.mob,
            verbose=not args.quiet,
        )
        result["track"] = track

    if not args.quiet:
        summary = {
            "agent": result["agent"], "anchor": result["anchor"],
            "mob": result["mob"], "baby": result["baby"],
            "attempts": result["attempts"],
            "spawned_count": len(result["spawned"]),
            "skipped_count": len(result["skipped"]),
        }
        if "track" in result:
            t = result["track"]
            summary["track"] = {
                "breach": t["breach"],
                "breach_first_t": t["breach_first_t"],
                "max_in_interior": t["max_in_interior"],
                "final_alive": t["final_alive"],
                "player_hp_min": t["player_hp_min"],
                "player_hp_final": t["player_hp_final"],
                "ended_early": t["ended_early"],
            }
        print(json.dumps(summary, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
