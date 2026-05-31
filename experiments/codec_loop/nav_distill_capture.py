#!/usr/bin/env python3
"""§21.0 capture — local-r navigation distillation (neural_interface.md §21).

The §21 arc climbs from §20.1a's "select a substrate goal" to "PRODUCE the local
plan": distill Baritone's window-exit subgoal (where the planned path crosses the
agent's action radius r) from local terrain + a goal bearing. The design invariant
is FIX the prediction target / MIGRATE the conditioning oracle→perception across
rungs — so this one capture records EVERYTHING the later rungs need (capture once,
ablate many) though §21.0 itself consumes only terrain + bearing.

What the dataset is: the TickSidecarRecorder line, which (after the §21.0 substrate
add) already carries the three things this rung needs, joined by tick —
  * block_grid        — the air-filtered r=10 cube around the player = local TERRAIN
  * baritone_state.path_fwd — the forward slice of the PLANNED PATH from the
        executor's current node = the TARGET source (the window-exit subgoal at any
        radius r≤10 is computed offline from this + the player origin, so one
        capture replays the whole r-sweep)
  * baritone_state.path_dest — the goal terminus = the BEARING source
There is no separate packet recording: the move stream is §16's deterministic
follower null; the neural object here is the path-level subgoal, which lives in the
sidecar. Frames are a PARALLEL throttled stream (forward-investment for the §21.2
visual rung) joined offline by wall-clock ms.

Capture recipe (terrain VARIETY is load-bearing — flat terrain makes the path a
straight line so the window-exit is trivially "toward the goal" at every r and the
horizon curve is flat by construction; we WANT hills/water/trees/cliffs that force
LOCAL detours): random_spawn across biomes, then long random-heading gotos. Each
rollout fires several legs so the body keeps moving across natural obstacles.

Peaceful for the capture (mobs would interrupt the legs / kill the idle body);
difficulty restored after.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_distill_capture \
        --port 25570 --rollouts 8 --legs 3 --leg-dist 90 \
        --spawn-range 20000 --frames --out-root results/sprint21
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import threading
import time
from pathlib import Path

from craft.spawn import random_spawn
from experiments.codec_loop.aim_carrier import _relay
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world


# --- sidecar + intent control (same substrate as §20.0) ----------------------
def _arm_sidecar(base: str, path: Path) -> dict:
    return _http("POST", f"{base}/obs/sidecar/arm",
                 {"path": str(path), "gzip": True}, timeout=15)


def _disarm_sidecar(base: str) -> dict:
    return _http("POST", f"{base}/obs/sidecar/disarm", timeout=25)


def _stop(base: str) -> dict:
    return _http("POST", f"{base}/baritone/stop", timeout=8)


def _set_baritone_render(base: str, visible: bool) -> dict:
    """Toggle Baritone's in-world overlay (path line + goal beacon + selection
    boxes). The pathfinder still runs — only the visuals change. CRITICAL for the
    §21.2 VISUAL rung: Baritone ships these ON, so a recorded frame has the planned
    PATH drawn straight at the goal — the window-exit subgoal painted on the input.
    Predicting the subgoal from such a frame is OCR of the answer, not perception of
    terrain. We force it OFF whenever frames are captured (the structured §21.0/§21.1
    channel is unaffected — overlay only touches pixels)."""
    return _http("POST", f"{base}/baritone/render", {"visible": visible}, timeout=8)


def _set_hud(base: str, visible: bool) -> dict:
    """Show/hide the whole vanilla HUD (health/food/air/hotbar/effects/xp/crosshair/
    selected-item). For §21.2 frames we hide it: the HUD doesn't leak the subgoal,
    but the model gets health/hunger/etc through the structured obs channel, not the
    pixels, so a terrain-only frame is the honest visual input. In-memory flag, resets
    to visible on client restart, so set per-run."""
    return _http("POST", f"{base}/hud", {"all": visible}, timeout=8)


def _ensure_fullbright(base: str) -> dict:
    """Force Wurst Fullbright ON so the frames have UNIFORM lighting — time-of-day
    and cave depth become non-confounds, so the only signal in the pixels is terrain
    geometry/material (what §21.2 tests perception against), not "is it dark → am I
    underground". Fresh clients boot all hacks OFF and it currently rides on Wurst's
    persisted state; pin it here so a fresh agent / post-bounce capture can't silently
    mix un-fullbright (darker) frames into the dataset. ON-only, no restore — Fullbright
    is part of the standard hack set; leaving it on is the correct resting state."""
    return _http("POST", f"{base}/wurst/hack", {"name": "Fullbright", "enabled": True}, timeout=8)


def _goto_blocking(base: str, x: int, y: int, z: int, tol: int, t: int) -> dict:
    return _http("POST", f"{base}/baritone/goto",
                 {"x": x, "y": y, "z": z, "timeout_seconds": t, "arrival_tolerance": tol},
                 timeout=t + 15)


def _position(base: str) -> dict:
    return _http("GET", f"{base}/position", timeout=8)


# --- throttled frame grabber (forward-investment for §21.2) ------------------
def _display_geometry(display: str) -> tuple[int, int]:
    """(w, h) of the Xvfb root on `display`; falls back to 1280x720."""
    try:
        out = subprocess.run(["xwininfo", "-root", "-display", display],
                             capture_output=True, text=True, timeout=5).stdout
        w = h = None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Width:"):
                w = int(s.split()[1])
            elif s.startswith("Height:"):
                h = int(s.split()[1])
        if w and h:
            return w, h
    except Exception:
        pass
    return 1280, 720


def _mc_window_rect(display: str) -> tuple[int, int, int, int] | None:
    """Absolute (w, h, x, y) of the Minecraft game window on `display`, parsed from
    `xwininfo -root -tree`. Lets the grab CROP to the game view and exclude the
    PrismLauncher console window (it floats at the display origin and otherwise
    occludes the frame's top-left — a §21.2 contaminant nearly as bad as the path
    overlay). Returns None if the MC window isn't found (caller falls back to the
    full display)."""
    try:
        out = subprocess.run(["xwininfo", "-root", "-tree", "-display", display],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "Minecraft" not in line:
                continue
            # e.g.  ...  854x480+213+120  +213+120   (last +X+Y is absolute)
            m = re.search(r"(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)", line)
            if m:
                w, h, x, y = (int(m.group(i)) for i in (1, 2, 3, 4))
                if w > 1 and h > 1:
                    return w, h, x, y
    except Exception:
        pass
    return None


class FrameGrabber:
    """Single-frame ffmpeg x11grab off the agent's Xvfb every `interval` s, on a
    worker thread. Each frame is named by the wall-clock ms at grab → joined to the
    nearest sidecar row's `captured_at_ms` offline. Degrades to no-frames silently
    (the §21.0 horizon curve does not depend on frames; they are §21.2 insurance)."""

    def __init__(self, display: str, frames_dir: Path, interval: float,
                 crop_to_mc: bool = True):
        self.display = display
        self.frames_dir = frames_dir
        self.interval = interval
        # Crop to the MC game window so the PrismLauncher console (at the display
        # origin) is excluded; fall back to the full display if not located.
        rect = _mc_window_rect(display) if crop_to_mc else None
        if rect is not None:
            self.w, self.h, self.gx, self.gy = rect
            self.cropped = True
        else:
            self.w, self.h = _display_geometry(display)
            self.gx, self.gy = 0, 0
            self.cropped = False
        self.grab_input = f"{display}+{self.gx},{self.gy}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.grabbed = 0
        self.failed = 0
        self.index: list[dict] = []

    def _grab_once(self) -> None:
        ts = int(time.time() * 1000)
        out = self.frames_dir / f"f-{ts}.png"
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "x11grab", "-video_size", f"{self.w}x{self.h}",
                 "-i", self.grab_input, "-frames:v", "1", str(out)],
                capture_output=True, timeout=8)
            if r.returncode == 0 and out.exists():
                self.grabbed += 1
                self.index.append({"ms": ts, "file": out.name})
            else:
                self.failed += 1
        except Exception:
            self.failed += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._grab_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        (self.frames_dir / "frame_index.json").write_text(
            json.dumps({"display": self.display, "geometry": [self.w, self.h],
                        "grab_offset": [self.gx, self.gy], "cropped_to_mc": self.cropped,
                        "interval_s": self.interval, "grabbed": self.grabbed,
                        "failed": self.failed, "frames": self.index}, indent=2))


# --- legs --------------------------------------------------------------------
def _leg_goal(px: float, pz: float, py: int, dist: int,
              rng: random.Random) -> tuple[int, int, int]:
    """A goal `dist` blocks away at a random heading. Y is left near the current
    surface (Baritone climbs/descends to reach it); the goal need not be reachable
    exactly — the planned path toward it across natural terrain is the signal."""
    theta = rng.uniform(0, 2 * math.pi)
    gx = int(px + dist * math.cos(theta))
    gz = int(pz + dist * math.sin(theta))
    return gx, py, gz


# --- per-rollout -------------------------------------------------------------
def run_rollout(base: str, idx: int, rdir: Path, legs: int, leg_dist: int,
                tol: int, leg_t: int, rng: random.Random,
                grabber: FrameGrabber | None) -> dict:
    rdir.mkdir(parents=True, exist_ok=True)
    sidecar_path = rdir / "sidecar.jsonl.gz"

    p0 = _position(base)
    sx, sy, sz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    print(f"\n=== rollout {idx} @ spawn=({sx},{sy},{sz}) legs={legs} → {rdir} ===",
          flush=True)

    _arm_sidecar(base, sidecar_path)
    time.sleep(0.3)                  # sidecar leads → full tick coverage
    if grabber is not None:
        grabber.start()

    leg_log = []
    t0 = time.time()
    try:
        for li in range(legs):
            p = _position(base)
            gx, gy, gz = _leg_goal(p["x"], p["z"], int(p["y"]), leg_dist, rng)
            r = _goto_blocking(base, gx, gy, gz, tol, leg_t)
            pe = _position(base)
            moved = math.hypot(pe["x"] - p["x"], pe["z"] - p["z"])
            leg_log.append({"leg": li, "goal": [gx, gy, gz],
                            "moved_blocks": round(moved, 1),
                            "reason": (r.get("reason") or r.get("message"))})
            print(f"  leg {li}: goal=({gx},{gy},{gz}) moved={moved:.1f}b "
                  f"reason={leg_log[-1]['reason']}", flush=True)
            _stop(base)
            time.sleep(0.3)
    finally:
        _stop(base)
        if grabber is not None:
            grabber.stop()
        sc_final = _disarm_sidecar(base)
    wall = round(time.time() - t0, 1)

    seg = _sidecar_summary(sidecar_path)
    entry = {"index": idx, "spawn": [sx, sy, sz], "wall_s": wall, "legs": leg_log,
             "sidecar": seg,
             "sc_drops": sc_final.get("dropped_queue_full"),
             "frames_grabbed": grabber.grabbed if grabber else 0}
    (rdir / "rollout.json").write_text(json.dumps(entry, indent=2))
    print(f"  ticks={seg['ticks']} with_path={seg['ticks_with_path']} "
          f"frames={entry['frames_grabbed']} wall={wall}s", flush=True)
    return entry


def _sidecar_summary(path: Path) -> dict:
    """Sanity: how many ticks landed and how many carried a usable forward path."""
    import gzip
    ticks = with_path = 0
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ticks += 1
                bs = d.get("baritone_state") or {}
                fwd = bs.get("path_fwd")
                if fwd and len(fwd) >= 2:
                    with_path += 1
    return {"ticks": ticks, "ticks_with_path": with_path}


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="§21.0 local-r nav distillation capture")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--cmd-base", default="http://127.0.0.1:4747",
                    help="MC server console relay (random_spawn gamemode/tp)")
    ap.add_argument("--player", default=None,
                    help="player name for spawn tp (default agentN from port)")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--legs", type=int, default=3, help="gotos per rollout")
    ap.add_argument("--leg-dist", type=int, default=90, help="goal distance per leg")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--leg-timeout", type=int, default=60)
    ap.add_argument("--spawn-range", type=int, default=20000,
                    help="random_spawn radius for biome variety (0 = stay put)")
    ap.add_argument("--keep-overlay", action="store_true",
                    help="keep Baritone's path/goal overlay ON during frame capture "
                         "(debug only — CONTAMINATES §21.2: the path is the answer)")
    ap.add_argument("--frames", action="store_true",
                    help="grab throttled Xvfb frames (forward-investment for §21.2)")
    ap.add_argument("--frame-interval", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out-root", default="results/sprint21")
    ap.add_argument("--no-peaceful", action="store_true")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    out_root = Path(args.out_root).absolute() / "capture"
    print(f"[nav_distill_capture] base={base} out_root={out_root}")
    if not _wait_in_world(base):
        print("[nav_distill_capture] FATAL: no player in world")
        return 2

    rng = random.Random(args.seed)
    display = f":{200 + (args.port - 25570)}" if 25570 <= args.port <= 25589 else None
    player = args.player or (f"agent{args.port - 25570}"
                             if 25570 <= args.port <= 25589 else "Player")

    if not args.no_peaceful:
        _relay(args.relay, "say §21.0 local-nav distillation capture")
        _relay(args.relay, "difficulty peaceful")
        time.sleep(0.3)

    # §21.2 confound guard: kill Baritone's path/goal/selection overlay so the
    # recorded frames don't carry the answer (the planned path drawn at the goal).
    # Only when actually grabbing frames; restored in the finally block.
    render_was_off = False
    hud_was_off = False
    if args.frames and not args.keep_overlay:
        r = _set_baritone_render(base, False)
        render_was_off = True
        print(f"[nav_distill_capture] baritone overlay OFF for clean frames: "
              f"renderPath={(r.get('settings') or {}).get('renderPath')}", flush=True)
        h = _set_hud(base, False)
        hud_was_off = True
        print(f"[nav_distill_capture] HUD hidden for clean frames: "
              f"success={h.get('success')}", flush=True)
        fb = _ensure_fullbright(base)
        print(f"[nav_distill_capture] Fullbright pinned ON (uniform lighting): "
              f"success={fb.get('success')}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    entries = []
    try:
        for i in range(args.rollouts):
            if args.spawn_range > 0:
                sp = random_spawn(
                    range_blocks=args.spawn_range, homunculus_base=base,
                    server_cmd_base=args.cmd_base, player_name=player,
                    rng=rng, verbose=False, log=lambda _m: None)
                if not sp.get("ok"):
                    print(f"  rollout {i}: spawn FAILED ({sp.get('attempts',[])[-1:]}); skip",
                          flush=True)
                    continue
                print(f"  rollout {i}: spawned biome={sp.get('biome')} "
                      f"@ {sp.get('tp_to')}", flush=True)
                time.sleep(0.5)
            rdir = out_root / f"rollout-{i}"
            grabber = (FrameGrabber(display, rdir / "frames", args.frame_interval)
                       if (args.frames and display) else None)
            entries.append(run_rollout(
                base, i, rdir, args.legs, args.leg_dist, args.tol,
                args.leg_timeout, rng, grabber))
    finally:
        _stop(base)
        if render_was_off:
            _set_baritone_render(base, True)   # restore Baritone's debugging overlay
        if hud_was_off:
            _set_hud(base, True)               # restore the vanilla HUD
        if not args.no_peaceful:
            _relay(args.relay, "difficulty easy")

    summary = {"base": base, "params": vars(args), "rollouts": entries,
               "total_ticks": sum(e["sidecar"]["ticks"] for e in entries),
               "total_ticks_with_path": sum(e["sidecar"]["ticks_with_path"] for e in entries)}
    (out_root / "capture_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 68)
    print("§21.0 CAPTURE — local-r nav distillation")
    print("=" * 68)
    print(f"  rollouts={len(entries)} total_ticks={summary['total_ticks']} "
          f"with_path={summary['total_ticks_with_path']}")
    print(f"wrote {out_root}/capture_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
