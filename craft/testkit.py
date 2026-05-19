"""Shared helpers for live-MC integration tests.

Every `test_*.py` should import from here instead of duplicating HTTP
wrappers, setup, arena prep, or logging. New helpers belong here when
they're useful to two or more tests; test-specific logic (plant_logs,
ambush rings, etc.) stays in the test module.

Design principles:
  - Defaults match the rest of the substrate (homunculus on localhost,
    mc_api on localhost, MC_PLAYER_NAME from env). Override only when
    needed.
  - Functions are pure call-and-return; no module-level state.
  - JSONL records use one file per test invocation; each iteration is
    one line. Path is `results/test-<name>-<ts>.jsonl` by default.

Tests are inherently flaky because the underlying environment is a live
MC client + Baritone. The runner judges by pass-rate over iterations,
not by demanding zero failures. Persist enough context per record
(world coords, biome, planted positions, etc.) that failures can be
reproduced — the MC world seed is fixed across wipes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from craft.spawn import random_spawn as _random_spawn
# Re-exported so tests can `from craft.testkit import set_difficulty, ...`
# without needing to know that world.py is the home for these.
from craft.world import (
    set_difficulty,
    set_gamemode,
    set_time,
    resolve_phase_ticks,
    PHASE_TICKS,
    Difficulty,
    Gamemode,
    Phase,
)

from craft.config import HOMUNCULUS_BASE, SERVER_CMD_BASE, PLAYER_NAME  # noqa: F401

__all__ = [
    "HOMUNCULUS_BASE", "SERVER_CMD_BASE", "PLAYER_NAME",
    "cmd", "pos", "stats", "inventory",
    "preflight", "setup_clean", "build_arena", "random_spawn",
    "TestLogger",
    # re-exports from craft.world
    "set_difficulty", "set_gamemode", "set_time",
    "resolve_phase_ticks", "PHASE_TICKS",
    "Difficulty", "Gamemode", "Phase",
]


# ---- HTTP wrappers ---------------------------------------------------------

def cmd(s: str, *, timeout: float = 5.0,
        server_cmd_base: str = SERVER_CMD_BASE) -> dict:
    """POST one server console command. Returns {"ok": bool, ...}."""
    try:
        r = requests.post(f"{server_cmd_base}/cmd",
                          json={"cmd": s}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": str(e)}


def pos(homunculus_base: str = HOMUNCULUS_BASE
        ) -> Optional[tuple[float, float, float]]:
    """Player (x, y, z) as floats, or None on transport error."""
    try:
        r = requests.get(f"{homunculus_base}/position", timeout=3.0)
        r.raise_for_status()
        p = r.json()
        return float(p["x"]), float(p["y"]), float(p["z"])
    except (requests.RequestException, ValueError, KeyError):
        return None


def stats(homunculus_base: str = HOMUNCULUS_BASE) -> Optional[dict]:
    """Full /stats payload, or None on transport error."""
    try:
        r = requests.get(f"{homunculus_base}/stats", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def inventory(homunculus_base: str = HOMUNCULUS_BASE) -> Optional[dict]:
    try:
        r = requests.get(f"{homunculus_base}/inventory", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


# ---- Test setup ------------------------------------------------------------

def preflight(
    *,
    require_paths: tuple[str, ...] = (),
    homunculus_base: str = HOMUNCULUS_BASE,
    ensure_hacks: bool = True,
) -> Optional[str]:
    """Return None if homunculus is reachable and required paths exist.

    `require_paths` lets a test demand specific routes (e.g. evasion test
    requires `/evasion/status`). A 404 on a required path is reported as
    a stale-jar problem — the usual fix.

    `ensure_hacks` (default True) turns on the substrate-required Wurst
    modules (KillAura, AutoEat, AutoTool, AutoRespawn, ...) once
    per test process. Fresh PrismLauncher instances boot with all Wurst
    hacks off, so concurrent agents need this every run. A wurst-not-loaded
    or transport error is logged but does not fail preflight — tests
    should still run, just with degraded behavior.
    """
    try:
        r = requests.get(f"{homunculus_base}/stats", timeout=3.0)
    except requests.RequestException as e:
        return f"homunculus unreachable: {e}"
    if not r.ok:
        return f"/stats returned {r.status_code}"
    for p in require_paths:
        try:
            r2 = requests.get(f"{homunculus_base}{p}", timeout=3.0)
            if r2.status_code == 404:
                return (f"homunculus is running an old jar — {p} missing. "
                        "Rebuild + deploy + restart MC.")
        except requests.RequestException:
            pass
    if ensure_hacks:
        # Import lazily — craft.wurst imports from craft.config, which uses
        # the same env vars. Lazy import keeps preflight cheap for the rare
        # caller that disables this.
        from craft.wurst import ensure_hacks_on
        report = ensure_hacks_on(verbose=False)
        if not report.get("ok"):
            failed = [r["name"] for r in report.get("results", [])
                      if not r.get("ok")]
            print(f"[testkit] wurst-preflight WARN: failed={failed} "
                  f"wurst_loaded={report.get('wurst_loaded')}", flush=True)
    return None


def setup_clean(
    anchor: tuple[int, int, int],
    *,
    player_name: str = PLAYER_NAME,
    extra_effects: tuple[str, ...] = (),
    server_cmd_base: str = SERVER_CMD_BASE,
) -> None:
    """Per-player heal/clear/tp + arena prep. Gamemode toggles for fall safety.

    Standard prep before any test exercises a tool. `extra_effects` is a
    tuple of strings to append after `effect give <player>` — use it when
    a test needs e.g. ("minecraft:resistance 120 4 true",) on top of the
    standard saturation + instant_health.

    DOES NOT mutate global world state (difficulty, time). The phase-grouped
    concurrent runner (run_tests.py) sets difficulty per group; setup_clean
    should never override that. For standalone test runs that need a clean
    peaceful start, set SUITE_COORDINATOR_MANAGES_DIFFICULTY=0 (or unset it),
    and the legacy peaceful+kill@e behavior runs as before — preserved for
    tests run outside the suite runner.

    Kill@e historically used distance=..64 from console (centered on world
    origin), which is meaningless once agents are scattered. With the env
    flag, neither the difficulty set nor that kill@e runs — coordinator
    handles cross-agent cleanup between phases.
    """
    ax, ay, az = anchor
    c = lambda s: cmd(s, server_cmd_base=server_cmd_base)
    coord_managed = os.environ.get(
        "SUITE_COORDINATOR_MANAGES_DIFFICULTY", "0") == "1"
    if not coord_managed:
        # Standalone-run path: keep legacy global-state cleanup so existing
        # invocations (python -m e2e.test_mine_wood ...) keep working.
        set_difficulty("peaceful", server_cmd_base=server_cmd_base)
        c("kill @e[type=!player,type=!item_frame,type=!armor_stand,distance=..64]")
    c(f"effect clear {player_name}")
    c(f"effect give {player_name} minecraft:saturation 30 9 true")
    c(f"effect give {player_name} minecraft:instant_health 1 9")
    for e in extra_effects:
        c(f"effect give {player_name} {e}")
    set_gamemode("creative", player_name=player_name, server_cmd_base=server_cmd_base)
    c(f"clear {player_name}")
    c(f"tp {player_name} {ax} {ay} {az}")
    time.sleep(1.5)
    set_gamemode("survival", player_name=player_name, server_cmd_base=server_cmd_base)
    time.sleep(0.5)


def build_arena(
    anchor: tuple[int, int, int],
    *,
    x_radius: int = 8,
    z_radius: Optional[int] = None,
    floor_block: str = "minecraft:stone",
    height: int = 3,
    server_cmd_base: str = SERVER_CMD_BASE,
) -> None:
    """Stone floor + air column around anchor, terrain-independent.

    Non-symmetric x/z is supported — evasion needs a long corridor east,
    mine_wood needs a square. `height` is air blocks above the floor.
    """
    ax, ay, az = anchor
    if z_radius is None:
        z_radius = x_radius
    c = lambda s: cmd(s, server_cmd_base=server_cmd_base)
    c(f"fill {ax - x_radius} {ay - 1} {az - z_radius} "
      f"{ax + x_radius} {ay - 1} {az + z_radius} {floor_block}")
    c(f"fill {ax - x_radius} {ay} {az - z_radius} "
      f"{ax + x_radius} {ay + height} {az + z_radius} minecraft:air")
    time.sleep(0.3)


def random_spawn(
    *,
    range_blocks: int,
    drop_y: int = 100,
    homunculus_base: str = HOMUNCULUS_BASE,
    server_cmd_base: str = SERVER_CMD_BASE,
    player_name: str = PLAYER_NAME,
    rng=None,
    verbose: bool = True,
) -> dict:
    """Thin wrapper around `craft.spawn.random_spawn` with testkit defaults."""
    return _random_spawn(
        range_blocks=range_blocks,
        drop_y=drop_y,
        homunculus_base=homunculus_base,
        server_cmd_base=server_cmd_base,
        player_name=player_name,
        rng=rng,
        verbose=verbose,
    )


# ---- JSONL logging ---------------------------------------------------------

def default_jsonl_path(test_name: str) -> Path:
    """`results/test-<name>-<unix_ts>.jsonl`."""
    return Path(f"results/test-{test_name}-{int(time.time())}.jsonl")


class TestLogger:
    """Append-only JSONL logger.

    Usage:
        logger = TestLogger("mine_wood")
        for i in range(iters):
            with logger.iter_record(i) as rec:
                rec["anchor"] = [...]
                # ... run the test, populate fields
                rec["passed"] = True
        logger.summary()  # returns {"iters", "passed", "rate"}

    `iter_record` automatically records `timestamp`, `wall_s`, and writes
    on context exit (even when the body raises). Each iter line includes
    a `passed: bool` field — tests are responsible for setting it; the
    default of False fails closed if the body doesn't get that far.
    """

    def __init__(self, test_name: str, path: Optional[Path] = None):
        self.test_name = test_name
        self.path = path if path is not None else default_jsonl_path(test_name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        # Truncate any existing file so a fresh run never inherits stale lines.
        self.path.write_text("")

    def write(self, record: dict) -> None:
        self._records.append(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    def iter_record(self, idx: int) -> "_IterRecord":
        return _IterRecord(self, idx)

    def summary(self) -> dict:
        """Pass-rate snapshot over written iters."""
        n = len(self._records)
        passed = sum(1 for r in self._records if r.get("passed"))
        return {
            "test": self.test_name,
            "iters": n,
            "passed": passed,
            "rate": (passed / n) if n else 0.0,
            "path": str(self.path),
        }


class _IterRecord:
    """Context manager that builds + writes one iter record."""

    def __init__(self, logger: TestLogger, idx: int):
        self.logger = logger
        self.record: dict = {
            "test": logger.test_name,
            "iter": idx,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "passed": False,  # fail-closed if the body never sets it
        }
        self._t0: float = 0.0

    def __enter__(self) -> dict:
        self._t0 = time.monotonic()
        return self.record

    def __exit__(self, exc_type, exc, tb):
        self.record["wall_s"] = round(time.monotonic() - self._t0, 2)
        if exc is not None:
            self.record["fatal_error"] = repr(exc)
            self.record["passed"] = False
        self.logger.write(self.record)
        return False  # never swallow — the caller's loop decides
