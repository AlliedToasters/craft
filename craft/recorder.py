"""Best-effort rollout screen recording — a video artifact per rollout.

Each fleet agent renders its Minecraft client to its own headless Xvfb
(display :200+N, software-GL via llvmpipe; see launch_agent.sh). The pixels
exist even with no monitor attached, so we can grab them off the virtual
framebuffer with ffmpeg's x11grab and write an .mp4 alongside the per-turn
JSONL transcript. This gives retroactive review of a rollout (impossible with
live-only monitoring) and is the first stepping stone toward multimodal
datasets pairing frames with the LLM transcript (and, later, homunculus
telemetry).

Design constraints:
- **Never break a rollout.** Recording is a side artifact: every failure path
  (no ffmpeg, undetectable display, ffmpeg dies) degrades to "no video" with a
  warning, never an exception into the agent loop.
- **Crash-resilient.** The mp4 is written with fragmented-moov flags so even a
  hard kill (SIGKILL, OOM) leaves a file playable up to the last flushed
  fragment — a rollout that dies mid-run still yields footage.
- **Cheap-ish.** The box is CPU-bound (MC rasterizes on CPU), so we default to
  a low frame rate + ultrafast x264. Tunable via env for heavier/lighter runs.

Enable with CRAFT_RECORD_VIDEO=1 (or --record-video). Env knobs:
  CRAFT_RECORD_VIDEO    truthy to enable (default off).
  CRAFT_RECORD_FPS      capture frame rate (default 10).
  CRAFT_RECORD_CRF      x264 quality, lower=better/bigger (default 30).
  CRAFT_RECORD_DISPLAY  explicit X display to grab (default: derived from the
                        homunculus port via the fleet :200+N convention, else
                        $DISPLAY).
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from craft.config import HOMUNCULUS_PORT, PLAYER_NAME


def _truthy(raw: str | None) -> bool:
    return bool(raw) and raw.strip().lower() not in ("0", "false", "off", "no", "")


def _resolve_display() -> str | None:
    """Which X display holds this agent's MC client.

    The fleet convention (launch_agent.sh) is authoritative for fleet ports:
    agentN's Xvfb is :200+N and N = homunculus_port - 25570. We prefer that over
    a stray inherited $DISPLAY so a headless fleet agent never records the wrong
    screen. Non-fleet ports fall back to $DISPLAY (e.g. a desktop dev session).
    """
    explicit = os.environ.get("CRAFT_RECORD_DISPLAY")
    if explicit:
        return explicit
    if 25570 <= HOMUNCULUS_PORT <= 25589:
        return f":{200 + (HOMUNCULUS_PORT - 25570)}"
    disp = os.environ.get("DISPLAY")
    return disp or None


_GEOM_RE = re.compile(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)")


def _detect_mc_geometry(display: str) -> tuple[int, int, int, int] | None:
    """Locate the Minecraft window on `display` → (w, h, x, y), or None.

    Grabbing exactly the game window (854x480 by default, centered in the
    1280x720 Xvfb with no WM) keeps frames clean for a future dataset and
    cheaper to encode than the full root. Falls back to None (full display) if
    xwininfo is absent or the window isn't found.
    """
    if shutil.which("xwininfo") is None:
        return None
    try:
        out = subprocess.run(
            ["xwininfo", "-display", display, "-root", "-tree"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        if "Minecraft" not in line:
            continue
        m = _GEOM_RE.search(line)
        if m:
            w, h, x, y = (int(g) for g in m.groups())
            if w > 1 and h > 1:  # skip the 1x1 placeholder windows
                return w, h, x, y
    return None


def _video_path_for(jsonl_path: str | None) -> str:
    """Derive the video artifact path so it sits beside the transcript."""
    if jsonl_path:
        stem = re.sub(r"\.jsonl$", "", jsonl_path)
        return stem + ".mp4"
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"results/video-{PLAYER_NAME}-{ts}.mp4"


class RolloutRecorder:
    """Wraps a single ffmpeg x11grab subprocess for one rollout."""

    def __init__(self, display: str, path: str, *, fps: int, crf: int,
                 region: tuple[int, int, int, int] | None):
        self.display = display
        self.path = path
        self.fps = fps
        self.crf = crf
        self.region = region
        self.proc: subprocess.Popen | None = None
        self._stopped = False

    def _build_cmd(self) -> list[str]:
        if self.region is not None:
            w, h, x, y = self.region
            inp = f"{self.display}+{x},{y}"
            size = ["-video_size", f"{w}x{h}"]
        else:
            inp = self.display
            size = []  # x11grab grabs the whole display when size is omitted
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "x11grab", "-framerate", str(self.fps), *size, "-i", inp,
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-crf", str(self.crf),
            # Fragmented moov → the file stays playable if we're hard-killed.
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            self.path,
        ]

    def start(self) -> bool:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.path + ".ffmpeg.log", "w")
        try:
            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_fh,
            )
        except OSError as e:
            print(f"[recorder] failed to start ffmpeg: {e}", flush=True)
            self.proc = None
            return False
        atexit.register(self.stop)  # safety net for the exception/return paths
        reg = f" region={self.region}" if self.region else " (full display)"
        print(f"[recorder] recording {self.display}{reg} @ {self.fps}fps → {self.path}", flush=True)
        return True

    def stop(self, timeout: float = 10.0) -> None:
        """Finalize the recording. Idempotent; safe from atexit and inline."""
        if self._stopped:
            return
        self._stopped = True
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            # 'q' on stdin = graceful ffmpeg shutdown (writes the trailer).
            p.communicate(b"q", timeout=timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        try:
            self._log_fh.close()
        except Exception:
            pass
        print(f"[recorder] stopped → {self.path}", flush=True)


def start_rollout_recording(jsonl_path: str | None) -> RolloutRecorder | None:
    """Start a recorder if CRAFT_RECORD_VIDEO is enabled, else return None.

    Best-effort: any precondition failure (disabled, no ffmpeg, no display)
    returns None with a one-line note. Never raises.
    """
    if not _truthy(os.environ.get("CRAFT_RECORD_VIDEO")):
        return None
    if shutil.which("ffmpeg") is None:
        print("[recorder] CRAFT_RECORD_VIDEO set but ffmpeg not found — skipping", flush=True)
        return None
    display = _resolve_display()
    if not display:
        print("[recorder] could not resolve an X display to record — skipping", flush=True)
        return None
    try:
        fps = int(os.environ.get("CRAFT_RECORD_FPS", "10"))
        crf = int(os.environ.get("CRAFT_RECORD_CRF", "30"))
    except ValueError:
        fps, crf = 10, 30
    region = _detect_mc_geometry(display)
    rec = RolloutRecorder(display, _video_path_for(jsonl_path), fps=fps, crf=crf, region=region)
    if not rec.start():
        return None
    return rec
