"""Biome-aware random spawn for live-MC tests + rollouts.

One canonical implementation of the spawn-retry loop. Both the agent
rollout setup (`craft.agent._apply_setup`) and the integration test
fixtures (`craft.testkit.random_spawn`) call into here so behavior stays
identical across the substrate.

Spectator column-spawn (the "find the real surface" approach)
-------------------------------------------------------------
The previous design TP'd the player to a fixed drop_y (=100) in creative
and reactively rejected the column afterward (encased-on-peak / cave-fall
heuristics + an adaptive drop_y bump). Two structural problems:

  1. **Encasement race.** Terrain does NOT generate until a player loads
     the chunk. A creative player TP'd into an ungenerated chunk lands on
     half-generated terrain, passes the survival HP probe, then the chunk
     finishes generating *around* it mid-rollout → suffocation. The probe
     window simply raced chunk-gen.
  2. **Biome sampling bias.** Any column whose surface peaks above drop_y
     got classified `column_inverted` and rejected, so high-terrain biomes
     (mountains, plateaus) were systematically excluded from the spawn
     distribution.

Both fall out of "guess the surface Y up front." Instead we *measure* it:

  - Put the player in **spectator** (no collision, no damage, no gravity)
    and TP to a high Y (`probe_y`, default 320 — above the Y=256 terrain
    cap and the ~Y=264 feature ceiling). The player's presence forces the
    chunk to generate; spectator can't be encased while it settles.
  - Poll `/scan_blocks` on the 1×1 column at (x, z) until it returns blocks
    (generation complete) and locate the topmost solid ground block.
  - TP to surface+1 (still spectator), then switch to survival. Because the
    chunk is now fully generated, the survival HP probe is reliable, not
    racy.

Per-attempt reject reasons:
  - gen_timeout            (chunk never generated within the poll budget)
  - in_water / in_lava     (topmost column block is liquid)
  - bad biome              (BAD_BIOMES — empirically unsurvivable)
  - damage_in_survival     (HP dropped after survival switch — encased,
                            e.g. a tree trunk sat at the chosen column)

BAD_BIOMES grows from observed failure modes, not aesthetic preference.
"""

from __future__ import annotations

import math
import random as _random
import time
from typing import Callable, Optional

import requests

from craft.world import set_gamemode


# Overworld vertical bounds (MC 1.21.4). probe_y sits above the Y=256
# natural-terrain cap and the ~Y=264 feature ceiling so a spectator TP'd
# there is always above the surface; the column scan reaches the world
# floor so caves/overhangs below the surface don't confuse detection.
WORLD_BOTTOM_Y: int = -64
SPECTATOR_PROBE_Y: int = 320

# Blocks that are not a standing surface. Air variants are obvious;
# leaves/logs/bark are filtered so a column that happens to fall on a tree
# resolves to the ground *under* the trunk — placing there puts the player
# inside the trunk, which the survival probe then rejects (so tree columns
# retry instead of spawning the agent up a canopy).
_AIR_LIKE: frozenset[str] = frozenset({"air", "cave_air", "void_air"})


BAD_BIOMES: tuple[str, ...] = (
    # Ocean variants: tiny island might be habitable but typically no wood.
    "ocean", "deep_ocean", "frozen_ocean", "warm_ocean",
    "lukewarm_ocean", "cold_ocean", "deep_frozen_ocean",
    "deep_cold_ocean", "deep_lukewarm_ocean",
    # Desert: no trees; long walks for wood miss dusk shelter window.
    "desert",
    # Badlands: terracotta spires fail floor_footprint (1-2/25 cells solid)
    # — surfaced by haiku R3 2026-05-14.
    "badlands", "wooded_badlands", "eroded_badlands",
    # Frozen / no-tree spawn traps (kept even after tag-aware crafting).
    "ice_spikes", "frozen_river", "frozen_peaks",
    "windswept_hills", "windswept_gravelly_hills", "windswept_forest",
    # Bamboo jungle: mine_wood fails (bamboo ≠ log; crafting support deferred).
    "bamboo_jungle", "sparse_bamboo_jungle",
    # Stony shore: technically habitable but trees are out of scan radius;
    # 2026-05-16 concurrent rollout agent2 stuck T50 with 45/50 FAILED
    # mine_wood. Until salvage-from-structures (issue #7) lands, treat as
    # spawn trap.
    "stony_shore",
)


def _server_cmd(server_cmd_base: str, cmd: str, *, timeout: float = 5.0) -> dict:
    try:
        r = requests.post(
            f"{server_cmd_base}/cmd",
            json={"cmd": cmd},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": str(e)}


def _stats(homunculus_base: str) -> Optional[dict]:
    try:
        r = requests.get(f"{homunculus_base}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _scan_player_y(homunculus_base: str) -> Optional[float]:
    """Player's current Y from /position (None on error). Used to confirm
    the player settled at the spawn surface rather than fell elsewhere."""
    try:
        r = requests.get(f"{homunculus_base}/position", timeout=5.0)
        r.raise_for_status()
        return float(r.json().get("y"))
    except (requests.RequestException, ValueError, TypeError):
        return None


def _scan_column(
    homunculus_base: str,
    x: int,
    z: int,
    *,
    y_top: int = SPECTATOR_PROBE_Y,
    y_bot: int = WORLD_BOTTOM_Y,
    timeout: float = 10.0,
) -> Optional[list[dict]]:
    """Scan the 1×1 vertical column at (x, z), returning the block list.

    Returns None on transport error OR when the chunk hasn't generated yet
    (homunculus reports success=False / empty when the chunk isn't loaded).
    The caller polls on this to detect generation. Volume is
    (y_top - y_bot + 1) cells (≤385 for the default range) — under
    homunculus's MAX_VOLUME=2000 cap.
    """
    try:
        r = requests.get(
            f"{homunculus_base}/scan_blocks",
            params={"x1": x, "y1": y_bot, "z1": z, "x2": x, "y2": y_top, "z2": z},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    if data.get("success") is False:
        return None
    return data.get("blocks") or None


def _surface_from_column(blocks: list[dict]) -> tuple[Optional[int], Optional[str]]:
    """Topmost standing surface in a column block list.

    Returns (surface_y, block_id_short). Skips air variants and tree
    canopy (leaves/logs/bark) so the result is true ground; the caller
    treats a water/lava result as a liquid-surface reject. Returns
    (None, None) for an all-air / empty column.
    """
    best_y: Optional[int] = None
    best_id: Optional[str] = None
    for b in blocks:
        bid = str(b.get("id", "")).split(":")[-1]
        if (
            bid in _AIR_LIKE
            or bid.endswith("_leaves")
            or bid.endswith("_log")
            or bid.endswith("_wood")
        ):
            continue
        y = b.get("y")
        if y is None:
            continue
        y = int(y)
        if best_y is None or y > best_y:
            best_y = y
            best_id = bid
    return best_y, best_id


def random_spawn(
    *,
    range_blocks: int,
    homunculus_base: str,
    server_cmd_base: str,
    player_name: str,
    probe_y: int = SPECTATOR_PROBE_Y,
    max_retries: int = 12,
    bad_biomes: tuple[str, ...] = BAD_BIOMES,
    gen_timeout_s: float = 20.0,
    gen_poll_interval_s: float = 0.5,
    settle_timeout_s: float = 3.0,
    survival_probe_s: float = 1.0,
    min_spawn_hp: float = 18.0,
    anchor_xz: Optional[tuple[int, int]] = None,
    rng: Optional[_random.Random] = None,
    verbose: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Pick a survivable random spawn within ±range_blocks of an anchor.

    Anchor defaults to the player's current (x, z). After return the player
    is always in survival mode. On exhaustion the player remains at the
    last attempted position — caller chooses whether to proceed or bail.

    Returns:
        {
            "ok": bool,                            # at least one attempt landed cleanly
            "anchor_xz": (ax, az) | None,          # the offsets are measured from this
            "offset": (dx, dz) | None,             # the chosen offset (last attempt if !ok)
            "tp_to": (tx, surface_y+1, tz) | None, # absolute spawn coord
            "biome": str | None,                   # biome at chosen position
            "attempts": [                          # per-attempt audit trail
                {"dx", "dz", "ok": bool, "reason": str, "surface_y": int | None},
                ...
            ],
        }
    """
    if rng is None:
        rng = _random.Random()

    if anchor_xz is None:
        try:
            p = requests.get(f"{homunculus_base}/position", timeout=5.0).json()
            ax = math.floor(p["x"])
            az = math.floor(p["z"])
        except (requests.RequestException, ValueError, KeyError):
            return {
                "ok": False, "anchor_xz": None, "offset": None,
                "tp_to": None, "biome": None,
                "attempts": [{"reason": "could_not_read_anchor_position"}],
            }
    else:
        ax, az = anchor_xz

    def _attempt(dx: int, dz: int) -> tuple[bool, str, Optional[int]]:
        tx = ax + dx
        tz = az + dz
        if verbose:
            log(f"[spawn] probe ({tx},?,{tz}) (offset {dx},{dz} from {ax},{az})")

        # 1. Spectator at high Y forces chunk-gen without collision/damage.
        set_gamemode("spectator", player_name=player_name,
                     server_cmd_base=server_cmd_base)
        _server_cmd(server_cmd_base, f"tp {player_name} {tx} {probe_y} {tz}")

        # 2. Poll the column until terrain generates and a surface appears.
        surf_y: Optional[int] = None
        top_id: Optional[str] = None
        deadline = time.time() + gen_timeout_s
        while time.time() < deadline:
            time.sleep(gen_poll_interval_s)
            blocks = _scan_column(homunculus_base, tx, tz,
                                  y_top=probe_y, y_bot=WORLD_BOTTOM_Y)
            if blocks:
                surf_y, top_id = _surface_from_column(blocks)
                if surf_y is not None:
                    break
        if surf_y is None:
            return False, "gen_timeout", None

        # 3. Liquid surface → not a viable standing spot.
        if top_id == "water":
            return False, "in_water", surf_y
        if top_id == "lava":
            return False, "in_lava", surf_y

        spawn_y = surf_y + 1

        # 4. Place via SURVIVAL collision, not a spectator TP. The homunculus
        #    spectator player drifts downward with noclip, so a spectator TP
        #    sinks the player *through* the surface; switching to survival
        #    first gives them collision so the TP lands them cleanly on top.
        #    The double-TP zeroes the residual fall velocity from the
        #    gamemode-switch tick (a single TP leaves a ~1-tick drop → ~0.3
        #    HP of phantom fall damage that muddies the encasement probe).
        #    TP to the block CENTER (+0.5): integer coords put the player's
        #    feet on a block corner, so their bounding box pokes into the
        #    adjacent (unscanned) column — at a terrain/biome boundary that
        #    neighbor can be a cliff face, encasing the player in a column we
        #    never verified (observed killing agent2 at a plains/forest edge,
        #    2026-05-25). Centering keeps the player inside the scanned column.
        cx_, cz_ = tx + 0.5, tz + 0.5
        set_gamemode("survival", player_name=player_name,
                     server_cmd_base=server_cmd_base)
        _server_cmd(server_cmd_base, f"tp {player_name} {cx_} {spawn_y} {cz_}")
        _server_cmd(server_cmd_base, f"clear {player_name}")
        time.sleep(0.4)
        _server_cmd(server_cmd_base, f"tp {player_name} {cx_} {spawn_y} {cz_}")

        # 5. Settle: wait until the player is grounded at the surface.
        settle_deadline = time.time() + settle_timeout_s
        s: dict = {}
        while time.time() < settle_deadline:
            time.sleep(0.3)
            s = _stats(homunculus_base) or {}
            p = _scan_player_y(homunculus_base)
            if s.get("on_ground") and (p is None or abs(p - spawn_y) <= 2):
                break

        # 6. Biome gate.
        biome = (s.get("biome") or "").split(":")[-1]
        if biome in bad_biomes:
            set_gamemode("spectator", player_name=player_name,
                         server_cmd_base=server_cmd_base)
            return False, f"biome_{biome}", surf_y

        # 7. Encasement probe. With collision placement there is no settle
        #    fall on clean ground (HP stays pinned at 20 under peaceful
        #    regen), so any sustained HP deficit means the spawn slot itself
        #    is bad — e.g. the column fell on a tree trunk and spawn_y is
        #    inside the log, where suffocation keeps HP below full.
        time.sleep(survival_probe_s)
        s = _stats(homunculus_base) or {}
        hp = float(s.get("health") or 0.0)
        # HP alone is the encasement signal: with centered collision placement
        # a clean spawn holds 20 (peaceful regen pins it) while an encased slot
        # bleeds well below the threshold. on_ground is intentionally NOT
        # required here — under fleet contention it can lag past the probe even
        # for a perfectly good full-HP spawn (false-rejects observed at C=20).
        if hp < min_spawn_hp:
            set_gamemode("spectator", player_name=player_name,
                         server_cmd_base=server_cmd_base)
            return False, f"damage_in_survival(hp={hp:.1f})", surf_y
        return True, biome, surf_y

    attempts: list[dict] = []
    last_tp: tuple[int, int, int] | None = None
    # Exhaustion fallback: the best *land* surface seen across attempts, so an
    # exhausted spawn lands on solid ground instead of wherever the last
    # (possibly water) probe left the player. A bad-biome surface is dry and
    # survivable; an encased (damage) surface is the last resort but still
    # beats drowning. Priority: biome reject (2) > damage reject (1).
    fallback: tuple[int, tuple[int, int, int]] | None = None
    for attempt in range(1, max_retries + 1):
        dx = rng.randint(-range_blocks, range_blocks)
        dz = rng.randint(-range_blocks, range_blocks)
        ok, info, surf_y = _attempt(dx, dz)
        if surf_y is not None:
            last_tp = (ax + dx, surf_y + 1, az + dz)
            prio = 2 if info.startswith("biome_") else (
                1 if info.startswith("damage_in_survival") else 0)
            if prio and (fallback is None or prio >= fallback[0]):
                fallback = (prio, (ax + dx, surf_y + 1, az + dz))
        attempts.append({
            "dx": dx, "dz": dz, "ok": ok, "reason": info,
            "surface_y": surf_y,
        })
        if ok:
            if verbose:
                log(f"[spawn] spawned ok at attempt {attempt}: "
                    f"biome={info} surface_y={surf_y}")
            set_gamemode("survival", player_name=player_name,
                         server_cmd_base=server_cmd_base)
            return {
                "ok": True,
                "anchor_xz": (ax, az),
                "offset": (dx, dz),
                "tp_to": last_tp,
                "biome": info,
                "attempts": attempts,
            }
        if verbose:
            log(f"[spawn] attempt {attempt}/{max_retries} rejected: {info}")

    set_gamemode("survival", player_name=player_name,
                 server_cmd_base=server_cmd_base)
    if fallback is not None:
        # Land the player on the best dry surface we found rather than leaving
        # them mid-air / in water (an exhausted spawn used to drown — agent17,
        # 2026-05-25). Centered TP, same as the success placement.
        fx, fy, fz = fallback[1]
        _server_cmd(server_cmd_base, f"tp {player_name} {fx + 0.5} {fy} {fz + 0.5}")
        last_tp = fallback[1]
        if verbose:
            log(f"[spawn] WARNING: exhausted {max_retries} retries; "
                f"placing at best land surface {fallback[1]} (prio {fallback[0]})")
    elif verbose:
        log(f"[spawn] WARNING: exhausted {max_retries} retries; "
            f"no land surface found — proceeding at last position")
    return {
        "ok": False,
        "anchor_xz": (ax, az),
        "offset": (attempts[-1]["dx"], attempts[-1]["dz"]) if attempts else None,
        "tp_to": last_tp,
        "biome": None,
        "attempts": attempts,
    }
