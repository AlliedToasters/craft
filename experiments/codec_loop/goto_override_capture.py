#!/usr/bin/env python3
"""§20.0 capture — forced-override + completion navigation move-streams.

The stateful rung (neural_interface.md §20). §16 closed the per-tick MOVE packet
in ISOLATION (no learned headroom; a deterministic obs-relative reparam). But a
Baritone move STREAM is a deterministic function of ONE committed goal, so the
stream compresses to its goal — the temporal analog of §18's "predict the
decision, not the packet," now at the plan level. And the corrigibility question
that was TRIVIAL in §19 (a feedforward controller re-reads g_t every tick → can't
slip authority) becomes non-trivial here: a COMMITTED controller can ride its old
plan when g_t changes mid-commitment. That latency is the moat.

This harness produces the offline tranche §13.2.4 explicitly DEFERRED. That study
(rung_c_transition) measured handover latency on PEACEFUL frozen rollouts whose
seams are almost all COMPLETION seams (goal done → next), and flagged: "Override
is the corrigibility-relevant seam; a non-peaceful recapture is a next-sprint
input." We make BOTH seam types head-on and on demand:

  completion  — blocking /baritone/goto to ARRIVAL, then stamp the next g_t and
                goto again. At the g_t flip the body is AT REST at the old goal:
                the old-segment momentum signature is already gone.
  override    — fire /baritone/goto in a background thread (it blocks holding
                Baritone.SESSION_LOCK while the body walks toward A), and ~T
                seconds in — body at speed, mid-path, far from A — stamp the new
                g_t then /baritone/stop (which DELIBERATELY bypasses the session
                lock, see StopHandler) and re-path to B. At the g_t flip the body
                still carries toward-A momentum; it must decelerate/reverse.

The substrate this rides (no new homunculus code):
  * /obs/meta {g_t, current_tool}  — AgentMeta stamps g_t + ticks_since_g_t_issued
        onto every per-tick PlayerObsSnapshot, so each recorded packet carries the
        active NAV goal as its segment label. We stamp g_t = "goto(x,y,z)".
  * /obs/sidecar                   — TickSidecarRecorder writes baritone_state.goal
        ("GoalBlock{x=..,y=..,z=..}") per tick = the GROUND-TRUTH goal.
  * /packets/recording             — PacketRecorder writes the outbound MoveAction
        fields + obs; joins the sidecar by tick. This IS the rung_c_transition
        input schema (obs.g_t / obs.ticks_since_g_t_issued / move fields).

Output layout mirrors experiments.next_packet.capture so the §20.0 measure step
reuses rung_c_moat / rung_c_transition verbatim — each mode writes its OWN data
root (results/sprint20/<mode>/) holding rollout-<i>/{packets.jsonl,
sidecar.jsonl.gz, seams.json}. All seams in the override root are overrides; all
in the completion root are completions → two crossovers → moat = override −
completion.

Run NON-peaceful is NOT needed (this is pure navigation): a peaceful window keeps
mobs from interrupting the legs. The harness sets peaceful for the capture and
restores difficulty after.

Usage:
    .venv/bin/python -m experiments.codec_loop.goto_override_capture \
        --port 25570 --reps 4 --waypoints 6 --leg 24 --override-after 2.0 \
        --out-root results/sprint20
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from pathlib import Path

from experiments.codec_loop.aim_carrier import _relay
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world


# --- recorder + intent control ----------------------------------------------
def _arm_packets(base: str, path: Path) -> dict:
    return _http("POST", f"{base}/packets/recording/arm", {"path": str(path)}, timeout=15)


def _disarm_packets(base: str) -> dict:
    return _http("POST", f"{base}/packets/recording/disarm", timeout=20)


def _arm_sidecar(base: str, path: Path) -> dict:
    return _http("POST", f"{base}/obs/sidecar/arm",
                 {"path": str(path), "gzip": True}, timeout=15)


def _disarm_sidecar(base: str) -> dict:
    return _http("POST", f"{base}/obs/sidecar/disarm", timeout=25)


def _meta(base: str, g_t: str | None, tool: str = "goto") -> dict:
    """Stamp the active nav intent. AgentMeta restamps ticks_since_g_t_issued only
    when the g_t STRING changes, so a distinct string per waypoint == a clean
    segment boundary at the issue tick."""
    return _http("POST", f"{base}/obs/meta",
                 {"g_t": g_t, "current_tool": (None if g_t is None else tool)}, timeout=8)


def _stop(base: str) -> dict:
    return _http("POST", f"{base}/baritone/stop", timeout=8)


def _goto_blocking(base: str, x: int, y: int, z: int, tol: int, t: int) -> dict:
    return _http("POST", f"{base}/baritone/goto",
                 {"x": x, "y": y, "z": z, "timeout_seconds": t, "arrival_tolerance": tol},
                 timeout=t + 15)


def _goto_background(base: str, x: int, y: int, z: int, tol: int, t: int) -> threading.Thread:
    """Fire a blocking goto on a worker thread and return immediately. The call
    holds Baritone.SESSION_LOCK for its duration; we interrupt it with /baritone/stop
    (lock-bypassing) to force the override, then join the thread."""
    th = threading.Thread(target=_goto_blocking, args=(base, x, y, z, tol, t), daemon=True)
    th.start()
    return th


# --- waypoints ---------------------------------------------------------------
def _ring(sx: int, sy: int, sz: int, leg: int, n: int) -> list[tuple[int, int, int]]:
    """A square ring of waypoints centred on spawn, side = leg, so each leg is a
    full `leg`-block axis-aligned traversal (long enough that an override fired a
    couple seconds in lands the body solidly mid-path)."""
    d = leg // 2
    corners = [(sx + d, sy, sz + d), (sx - d, sy, sz + d),
               (sx - d, sy, sz - d), (sx + d, sy, sz - d)]
    return [corners[i % 4] for i in range(n)]


def _gstr(w: tuple[int, int, int]) -> str:
    return f"goto({w[0]},{w[1]},{w[2]})"


# --- per-rollout drivers -----------------------------------------------------
def _run_completion(base: str, wps, tol: int, leg_t: int) -> list[dict]:
    """Blocking gotos: g_t flips with the body AT REST at the prior goal."""
    seams = []
    for i, w in enumerate(wps):
        _meta(base, _gstr(w))
        meta = _http("GET", f"{base}/obs/meta", timeout=8)
        outcome = _goto_blocking(base, w[0], w[1], w[2], tol, leg_t)
        seams.append({"i": i, "type": "completion", "goal": _gstr(w),
                      "issued_tick": meta.get("g_t_issued_tick"),
                      "reason": (outcome.get("reason") or outcome.get("message"))})
    return seams


def _run_override(base: str, wps, tol: int, leg_t: int, after_s: float,
                  settle_s: float) -> list[dict]:
    """Fire goto in the background; ~after_s in (body at speed, mid-path) stamp the
    next g_t and /baritone/stop, then re-path. g_t flips while the body still
    carries toward-prior momentum."""
    seams = []
    th = None
    for i, w in enumerate(wps):
        if th is not None:
            # interrupt the in-flight (still-moving) leg, stamp the NEW goal first
            # so the symbolic flip precedes the body redirect (the lag we measure).
            _meta(base, _gstr(w))
            meta = _http("GET", f"{base}/obs/meta", timeout=8)
            _stop(base)                 # lock-bypassing cancel of the bg goto
            time.sleep(settle_s)        # let the bg goto unwind through finally → unlock
            th.join(timeout=2.0)
            seams.append({"i": i, "type": "override", "goal": _gstr(w),
                          "issued_tick": meta.get("g_t_issued_tick")})
        else:
            _meta(base, _gstr(w))       # first leg: just start moving
        th = _goto_background(base, w[0], w[1], w[2], tol, leg_t)
        time.sleep(after_s)
    # final leg: stop the dangling bg goto and release the lock before disarm
    _stop(base)
    time.sleep(settle_s)
    if th is not None:
        th.join(timeout=2.0)
    return seams


# --- join check (tick coverage of packets by the sidecar) --------------------
def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _verify(packets: Path, sidecar: Path) -> dict:
    pk, sc = set(), set()
    if packets.exists():
        with _open_text(packets) as f:
            for line in f:
                try:
                    pk.add(json.loads(line)["obs"]["tick"])
                except (ValueError, KeyError):
                    pass
    if sidecar.exists():
        with _open_text(sidecar) as f:
            for line in f:
                try:
                    sc.add(json.loads(line)["tick"])
                except (ValueError, KeyError):
                    pass
    joined = pk & sc
    return {"packet_ticks": len(pk), "sidecar_ticks": len(sc),
            "joined": len(joined),
            "join_pct": round(100.0 * len(joined) / len(pk), 1) if pk else None}


def _seg_summary(packets: Path) -> dict:
    """How many distinct g_t segments + move packets actually landed (sanity that
    the legs produced traffic and the stamps flipped)."""
    gts, moves, n = [], 0, 0
    prev = object()
    if packets.exists():
        with open(packets, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                n += 1
                if "move" in (d.get("id") or ""):
                    moves += 1
                g = d["obs"].get("g_t")
                if g != prev:
                    gts.append(g)
                    prev = g
    return {"packets": n, "move_packets": moves, "segments": max(0, len(gts))}


# --- main --------------------------------------------------------------------
def run_mode(base: str, mode: str, root: Path, reps: int, wp_n: int, leg: int,
             tol: int, leg_t: int, after_s: float, settle_s: float) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(reps):
        p = _http("GET", f"{base}/position", timeout=8)
        sx, sy, sz = int(p["x"]), int(p["y"]), int(p["z"])
        wps = _ring(sx, sy, sz, leg, wp_n)
        rdir = root / f"rollout-{i}"
        rdir.mkdir(parents=True, exist_ok=True)
        packets_path = rdir / "packets.jsonl"
        sidecar_path = rdir / "sidecar.jsonl.gz"

        print(f"\n=== {mode} rollout {i+1}/{reps} @ spawn=({sx},{sy},{sz}) "
              f"wps={len(wps)} → {rdir} ===", flush=True)
        _arm_sidecar(base, sidecar_path)
        time.sleep(0.25)                 # sidecar leads packets → 100% join (capture.py rationale)
        _arm_packets(base, packets_path)
        _meta(base, None)                # clear any stale intent before the run

        t0 = time.time()
        try:
            if mode == "completion":
                seams = _run_completion(base, wps, tol, leg_t)
            else:
                seams = _run_override(base, wps, tol, leg_t, after_s, settle_s)
        finally:
            _meta(base, None)
            pk_final = _disarm_packets(base)
            sc_final = _disarm_sidecar(base)
        wall = round(time.time() - t0, 1)

        (rdir / "seams.json").write_text(json.dumps(
            {"mode": mode, "spawn": [sx, sy, sz], "waypoints": wps, "seams": seams},
            indent=2))
        join = _verify(packets_path, sidecar_path)
        seg = _seg_summary(packets_path)
        entry = {"index": i, "mode": mode, "spawn": [sx, sy, sz], "wall_s": wall,
                 "n_seams": len([s for s in seams if s.get("type")]),
                 "join": join, "seg": seg,
                 "pkt_drops": pk_final.get("dropped_queue_full"),
                 "sc_drops": sc_final.get("dropped_queue_full")}
        entries.append(entry)
        print(f"  packets={seg['packets']} moves={seg['move_packets']} "
              f"segments={seg['segments']} seams={entry['n_seams']} "
              f"join={join['join_pct']}% wall={wall}s", flush=True)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="§20.0 override+completion nav capture")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--modes", default="completion,override")
    ap.add_argument("--reps", type=int, default=4, help="rollouts per mode")
    ap.add_argument("--waypoints", type=int, default=6, help="legs per rollout")
    ap.add_argument("--leg", type=int, default=24, help="ring side (block traversal per leg)")
    ap.add_argument("--tol", type=int, default=2, help="arrival tolerance (blocks)")
    ap.add_argument("--leg-timeout", type=int, default=40, help="per-goto timeout (s)")
    ap.add_argument("--override-after", type=float, default=2.0,
                    help="seconds into a leg before forcing the override")
    ap.add_argument("--settle", type=float, default=0.2,
                    help="seconds after /baritone/stop for the bg goto to release the lock")
    ap.add_argument("--out-root", default="results/sprint20")
    ap.add_argument("--no-peaceful", action="store_true",
                    help="skip the peaceful window (default sets peaceful for clean legs)")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    print(f"[goto_override_capture] base={base} out_root={args.out_root}")
    if not _wait_in_world(base):
        print("[goto_override_capture] FATAL: no player in world")
        return 2

    if not args.no_peaceful:
        _relay(args.relay, "say §20.0 nav goal-commitment capture")
        _relay(args.relay, "difficulty peaceful")  # clean legs, no mob interrupts
        time.sleep(0.3)

    # Absolute: the homunculus process resolves recording paths against ITS cwd
    # (the agent's MC instance dir), not ours — a relative path silently lands
    # under .../minecraft/ and we'd record nothing here.
    out_root = Path(args.out_root).absolute()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary: dict = {"base": base, "modes": modes,
                     "params": {"reps": args.reps, "waypoints": args.waypoints,
                                "leg": args.leg, "override_after": args.override_after}}
    try:
        for mode in modes:
            summary[mode] = run_mode(
                base, mode, out_root / mode, args.reps, args.waypoints, args.leg,
                args.tol, args.leg_timeout, args.override_after, args.settle)
    finally:
        _stop(base)
        _meta(base, None)
        if not args.no_peaceful:
            _relay(args.relay, "difficulty easy")  # restore fleet difficulty

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "capture_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 68)
    print("§20.0 CAPTURE — override + completion nav move-streams")
    print("=" * 68)
    for mode in modes:
        es = summary.get(mode, [])
        tot_seams = sum(e["n_seams"] for e in es)
        worst_join = min((e["join"]["join_pct"] for e in es
                          if e["join"]["join_pct"] is not None), default=None)
        tot_moves = sum(e["seg"]["move_packets"] for e in es)
        print(f"  {mode:>10}: rollouts={len(es)} seams={tot_seams} "
              f"move_pkts={tot_moves} worst_join={worst_join}%")
    print(f"wrote {out_root}/capture_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
