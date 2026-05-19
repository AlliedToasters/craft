"""Biome-aware random spawn for live-MC tests + rollouts.

One canonical implementation of the spawn-retry loop. Both the agent
rollout setup (`craft.agent._apply_setup`) and the integration test
fixtures (`craft.testkit.random_spawn`) call into here so behavior stays
identical across the substrate.

Behavior per attempt: pick (dx, dz) ∈ [-range, range]^2, /tp the player
to (anchor.x + dx, drop_y, anchor.z + dz) in creative, wait for
on_ground, then reject if any of:
  - stuck_no_ground   (TP'd inside terrain; suffocation risk)
  - column_inverted   (landed too high — terrain extended above drop_y;
                       agent clipped onto a peak or encased mid-block)
  - cave_fall         (fell too far below drop_y — column had an open
                       cave pocket; agent dropped into a deep cave)
  - in_water          (typically ocean; rollout wood-starves)
  - in_lava           (immediate death)
  - bad biome         (BAD_BIOMES — empirically unsurvivable)
  - HP drop in survival (creative shielded a suffocation slot)

BAD_BIOMES grows from observed failure modes, not aesthetic preference.
"""

from __future__ import annotations

import math
import random as _random
import time
from typing import Callable, Optional

import requests

from craft.world import set_gamemode


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


def _classify_landing(
    landing_y: int,
    drop_y: int,
    *,
    inverted_margin: int = 5,
    cave_fall_max: int = 50,
) -> Optional[str]:
    """Reject landings that signal a bad column (encased peak or cave pocket).

    The current spawn-retry loop catches water/lava/biome but is blind to
    column shape:
      - High terrain above drop_y → agent clips onto a peak. on_ground=true
        but the spawn is effectively encased mid-block; build_shelter
        aborts downstream with `encased=true`.
      - Open column with a cave pocket → agent falls *through* the natural
        surface into a cave. on_ground=true, biome valid, but surface() now
        has to ascend 9-20 blocks of stone (Baritone vertical-mine timeout).

    Returns a string reason (used as the `attempts[].reason` audit code) if
    the landing should be rejected, else None.

    Thresholds: with drop_y=100, defaults reject landing_y>=95 (inverted)
    and landing_y<50 (cave-fall). The 50-block cave-fall threshold leaves
    normal plains (y≈64) and forest (y≈70) alone while catching the
    observed-failure landings (y=44/50/55/59 in deep caves below
    y=70-79 surfaces). Tunable per call. Doesn't catch all cave-fall
    cases — a column-scan for the natural surface would, but that needs
    a homunculus API we don't have yet.
    """
    if landing_y >= drop_y - inverted_margin:
        return f"column_inverted(y={landing_y})"
    if drop_y - landing_y > cave_fall_max:
        return f"cave_fall(y={landing_y})"
    return None


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


def _position(homunculus_base: str) -> Optional[dict]:
    """Read /position for landing_y. /stats does not include y."""
    try:
        r = requests.get(f"{homunculus_base}/position", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def random_spawn(
    *,
    range_blocks: int,
    homunculus_base: str,
    server_cmd_base: str,
    player_name: str,
    drop_y: int = 100,
    max_retries: int = 8,
    bad_biomes: tuple[str, ...] = BAD_BIOMES,
    column_inverted_margin: int = 5,
    column_cave_fall_max: int = 50,
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
            "tp_to": (tx, drop_y, tz) | None,      # absolute target coord
            "biome": str | None,                   # biome at chosen position
            "attempts": [                          # per-attempt audit trail
                {"dx", "dz", "ok": bool, "reason": str},
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

    def _attempt(dx: int, dz: int) -> tuple[bool, str]:
        tx = ax + dx
        tz = az + dz
        if verbose:
            log(f"[spawn] tp to ({tx},{drop_y},{tz}) (offset {dx},{dz} from {ax},{az})")
        set_gamemode("creative", player_name=player_name,
                     server_cmd_base=server_cmd_base)
        _server_cmd(server_cmd_base, f"tp {player_name} {tx} {drop_y} {tz}")
        _server_cmd(server_cmd_base, f"clear {player_name}")
        landed = False
        for _ in range(20):
            time.sleep(0.5)
            s = _stats(homunculus_base)
            if s and s.get("on_ground"):
                landed = True
                break
        if not landed:
            return False, "stuck_no_ground"
        # Column-quality check: /stats omits coords, so read /position for y.
        # Catches both encased-on-peak (landing_y≈drop_y) and cave-fall
        # (landing_y << drop_y) which the existing checks miss.
        pos = _position(homunculus_base) or {}
        ly_raw = pos.get("y")
        if ly_raw is not None:
            landing_y = int(math.floor(float(ly_raw)))
            reason = _classify_landing(
                landing_y, drop_y,
                inverted_margin=column_inverted_margin,
                cave_fall_max=column_cave_fall_max,
            )
            if reason is not None:
                return False, reason
        s = _stats(homunculus_base) or {}
        if s.get("in_water"):
            return False, "in_water"
        if s.get("in_lava"):
            return False, "in_lava"
        biome = (s.get("biome") or "").split(":")[-1]
        if biome in bad_biomes:
            return False, f"biome_{biome}"
        # Survival probe catches "on_ground=true but stuck inside a wall" —
        # creative shielded a suffocation slot while we sampled. ~24 ticks
        # is enough for at least one suffocation tick to register.
        set_gamemode("survival", player_name=player_name,
                     server_cmd_base=server_cmd_base)
        time.sleep(1.2)
        s = _stats(homunculus_base) or {}
        hp = float(s.get("health") or 0.0)
        if hp < 20.0:
            set_gamemode("creative", player_name=player_name,
                         server_cmd_base=server_cmd_base)
            return False, f"damage_in_survival(hp={hp:.1f})"
        return True, biome

    attempts: list[dict] = []
    last_tp: tuple[int, int, int] | None = None
    for attempt in range(1, max_retries + 1):
        dx = rng.randint(-range_blocks, range_blocks)
        dz = rng.randint(-range_blocks, range_blocks)
        last_tp = (ax + dx, drop_y, az + dz)
        ok, info = _attempt(dx, dz)
        attempts.append({"dx": dx, "dz": dz, "ok": ok, "reason": info})
        if ok:
            if verbose:
                log(f"[spawn] landed ok at attempt {attempt}: biome={info}")
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

    if verbose:
        log(f"[spawn] WARNING: exhausted {max_retries} retries; "
            f"proceeding with last position")
    set_gamemode("survival", player_name=player_name,
                 server_cmd_base=server_cmd_base)
    return {
        "ok": False,
        "anchor_xz": (ax, az),
        "offset": (attempts[-1]["dx"], attempts[-1]["dz"]) if attempts else None,
        "tp_to": last_tp,
        "biome": None,
        "attempts": attempts,
    }
