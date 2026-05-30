#!/usr/bin/env python3
"""Sprint A — CONTROLLED parity-vs-bits sweep on a flat built arena.

Replaces the n=1 out-and-back `run_rungs` sweep, which had two fatal confounds
(see results/sprintA/RESULTS.md):

  1. ok-gate artifact (FIXED in craft/codec/server.py): the codec returned
     ok=false on every quantized packet, so homunculus counted drift and passed
     the ORIGINAL packet through — the controller ran effectively lossless and
     every bit level "passed". server.py now returns ok=true in lossy mode so
     the quantized fields actually reach the wire.

  2. mutating course (this file's reason to exist): Baritone runs with allowBreak
     and terraforms the spawn area into a pit; back-to-back trials on the same
     ground meant each rollout started from more-degraded terrain than the last.
     "All rollouts look similar" = the agent thrashing in the same hole. Order
     confounds bits.

This harness controls the course:
  * Flat stone arena built FAR from spawn (terrain-independent fill), so natural
    terrain and the spawn pit are never touched.
  * Baritone allowBreak forced OFF — the agent cannot reshape the test surface.
  * The player is TP'd back to a fixed arena start before EVERY trial — no
    terraform/position carryover between trials or between bit levels.
  * N repeated trials per bit level — failure is cliff-shaped (a quantized pos
    delta compounds and tips a ledge-jump from success to blocked; observed
    live), so we need a RATE per b, not one Bernoulli draw.
  * Continuous metric is PRIMARY: per-leg final distance-to-target. The binary
    "reached" flag absorbs ~1 block of slack and is a poor estimator on a jagged
    surface; we keep it as secondary.

CAVEAT (documented, not hidden): a FLAT arena under-tests the jump/ledge failure
mode that was the most visible live symptom. The continuous drift metric still
captures accumulating positional error; the discrete ledge-cliff is reported
qualitatively. A natural-terrain obstacle-course variant is the follow-up if the
flat-arena drift curve is too smooth to locate a knee.

Brief alignment: movement-only quantization, ONE target type, NOT a learned-codec
baseline (anti-pattern #1). Deliverable = the parity-vs-bits knee.

Usage:
    .venv/bin/python -m experiments.codec_loop.sprintA_arena \
        --port 25570 --player agent0 --trials 15 --bits 8,6,5,4,3 \
        --out results/sprintA/arena.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


# ---- HTTP --------------------------------------------------------------------

def _http(method: str, url: str, body: dict | None = None, timeout: float = 65.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
    except Exception as e:
        return {"_transport_error": str(e)}
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {"_raw": raw}


# ---- substrate helpers (thin; reuse homunculus + relay routes) ---------------

def _pos(base: str) -> dict:
    return _http("GET", f"{base}/position", timeout=8)


def _in_world(base: str) -> bool:
    return _pos(base).get("x") is not None


def _wait_in_world(base: str, timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _in_world(base):
            return True
        time.sleep(5)
    return False


def _server_cmd(relay: str, cmd: str, timeout: float = 8.0) -> dict:
    return _http("POST", f"{relay}/cmd", {"cmd": cmd}, timeout=timeout)


def _say(relay: str, msg: str) -> None:
    """Emit a line to the MC server chat so the user can observe the sweep live
    in-client (user request, 2026-05-29). Best-effort; never blocks the trial."""
    try:
        _server_cmd(relay, f"say {msg}", timeout=4.0)
    except Exception:
        pass


def _set_allow_break(base: str, value: bool) -> dict:
    return _http("POST", f"{base}/baritone/allow_break", {"value": value})


def _set_quant(codec_cfg_url: str, bits: int | None) -> dict:
    return _http("POST", codec_cfg_url, {"quant_bits": bits})


def _arm_codec(base: str, codec_url: str) -> dict:
    return _http("POST", f"{base}/codec/passthrough/arm",
                 {"endpoint": codec_url, "substitute": True})


def _disarm_codec(base: str) -> dict:
    return _http("POST", f"{base}/codec/passthrough/disarm")


def _codec_status(base: str) -> dict:
    return _http("GET", f"{base}/codec/passthrough/status")


def _goto(base: str, x: int, y: int, z: int, tol: int, t: int, allow_place: bool) -> dict:
    return _http(
        "POST", f"{base}/baritone/goto",
        {"x": x, "y": y, "z": z, "timeout_seconds": t,
         "arrival_tolerance": tol, "allow_place": allow_place},
        timeout=t + 15,
    )


def _stop(base: str) -> dict:
    return _http("POST", f"{base}/stop", {})


# ---- arena setup -------------------------------------------------------------

def build_flat_arena(relay: str, player: str, anchor: tuple[int, int, int],
                     radius: int) -> None:
    """Stone floor + air column at a far anchor (terrain-independent fill).

    Mirrors craft.testkit.build_arena but driven through the relay directly so
    this harness has no import-time dependency on test fixtures.
    """
    ax, ay, az = anchor
    _server_cmd(relay, f"fill {ax-radius} {ay-1} {az-radius} {ax+radius} {ay-1} {az+radius} minecraft:stone")
    _server_cmd(relay, f"fill {ax-radius} {ay} {az-radius} {ax+radius} {ay+4} {az+radius} minecraft:air")
    time.sleep(0.4)


def tp_to(relay: str, player: str, x: float, y: float, z: float) -> None:
    _server_cmd(relay, f"tp {player} {x} {y} {z}")
    time.sleep(0.5)


def heal(relay: str, player: str) -> None:
    # keep the agent alive across many trials (peaceful arena, but fall/edge safety)
    _server_cmd(relay, f"effect give {player} minecraft:instant_health 1 10 true")


# ---- one trial ---------------------------------------------------------------

def _dist(fp, target) -> float | None:
    if not fp or len(fp) != 3:
        return None
    return math.sqrt(sum((fp[i] - target[i]) ** 2 for i in range(3)))


def run_trial(base: str, relay: str, player: str, start: tuple[int, int, int],
              out_xyz: tuple[int, int, int], tol: int, leg_timeout: int) -> dict:
    """One out-and-back on the flat arena from a freshly-reset start.

    Returns per-leg final positions + distances. Continuous metric is the
    distance; reached is secondary.
    """
    # hard reset position so no carryover between trials
    _stop(base)
    tp_to(relay, player, start[0] + 0.5, start[1], start[2] + 0.5)

    a = _goto(base, *out_xyz, tol=tol, t=leg_timeout, allow_place=False)
    b = _goto(base, *start, tol=tol, t=leg_timeout, allow_place=False)

    da = _dist(a.get("final_position"), out_xyz)
    db = _dist(b.get("final_position"), start)
    return {
        "out": {"target": out_xyz, "final_position": a.get("final_position"),
                "dist": round(da, 3) if da is not None else None,
                "reason": a.get("reason"),
                "reached": da is not None and da <= tol + 1.5},
        "home": {"target": start, "final_position": b.get("final_position"),
                 "dist": round(db, 3) if db is not None else None,
                 "reason": b.get("reason"),
                 "reached": db is not None and db <= tol + 1.5},
    }


# ---- sweep -------------------------------------------------------------------

def sweep_bit_level(base: str, relay: str, codec_cfg_url: str, codec_url: str,
                    player: str, bits: int | None, start, out_xyz, tol,
                    leg_timeout, trials) -> dict:
    """N trials at one bit level (None = lossless control). Codec armed in
    substitute mode for the whole level; quant level set once."""
    _set_quant(codec_cfg_url, bits)
    _arm_codec(base, codec_url)
    label = "lossless control" if bits is None else f"b={bits}"
    _say(relay, f"[Sprint A] === {label} ({trials} trials) ===")
    legs = []
    for i in range(trials):
        _say(relay, f"[Sprint A] {label} trial {i+1}/{trials}")
        t = run_trial(base, relay, player, start, out_xyz, tol, leg_timeout)
        legs.append(t)
        d_out = t["out"]["dist"]
        d_home = t["home"]["dist"]
        print(f"    b={bits!s:>4} trial {i+1}/{trials}: "
              f"out_dist={d_out} home_dist={d_home} "
              f"reached={int(t['out']['reached'])+int(t['home']['reached'])}/2",
              flush=True)
    status = _codec_status(base)
    final = _disarm_codec(base)

    # aggregate
    out_dists = [t["out"]["dist"] for t in legs if t["out"]["dist"] is not None]
    home_dists = [t["home"]["dist"] for t in legs if t["home"]["dist"] is not None]
    all_dists = out_dists + home_dists
    reached = sum(int(t["out"]["reached"]) + int(t["home"]["reached"]) for t in legs)
    total_legs = 2 * len(legs)

    def _mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    def _p(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(q * len(s)))], 3)

    return {
        "bits": bits,
        "trials": trials,
        "reached": reached,
        "total_legs": total_legs,
        "reached_rate": round(reached / total_legs, 3) if total_legs else None,
        "dist_mean": _mean(all_dists),
        "dist_p50": _p(all_dists, 0.5),
        "dist_p90": _p(all_dists, 0.9),
        "dist_max": round(max(all_dists), 3) if all_dists else None,
        "counters": {
            "substituted": final.get("substituted"),
            "drift": final.get("drift"),
            "substitute_errors": final.get("substitute_errors"),
            "transport_errors": final.get("transport_errors"),
            "subst_latency_p99_ms": status.get("subst_latency_p99_ms"),
            "subst_latency_mean_ms": status.get("subst_latency_mean_ms"),
        },
        "legs": legs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A controlled flat-arena parity sweep")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--player", default="agent0")
    ap.add_argument("--relay", default=os.environ.get("MC_SERVER_CMD_BASE", "http://127.0.0.1:4747"))
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--codec-cfg-url", default="http://127.0.0.1:25600/config")
    ap.add_argument("--anchor", default="5000,100,5000", help="arena center x,y,z")
    ap.add_argument("--radius", type=int, default=12, help="arena half-width (flat floor)")
    ap.add_argument("--delta", type=int, default=8, help="out-and-back hop length (kept inside arena)")
    ap.add_argument("--tol", type=int, default=1, help="arrival tolerance (blocks)")
    ap.add_argument("--leg-timeout", type=int, default=25)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--bits", default="8,6,5,4,3", help="comma list of bit levels (control always run)")
    ap.add_argument("--out", default="results/sprintA/arena.json")
    ap.add_argument("--no-build", action="store_true",
                    help="skip build_flat_arena (preserve a hand-placed obstacle course); "
                         "still disables allowBreak + TP-resets per trial")
    args = ap.parse_args()

    host = os.environ.get("HOMUNCULUS_HOST", "127.0.0.1")
    base = f"http://{host}:{args.port}"
    ax, ay, az = (int(v) for v in args.anchor.split(","))
    anchor = (ax, ay, az)
    start = (ax, ay, az)
    out_xyz = (ax + args.delta, ay, az + args.delta)
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]

    print(f"[arena] base={base} relay={args.relay} player={args.player}")
    print(f"[arena] anchor={anchor} radius={args.radius} start={start} out={out_xyz} "
          f"tol={args.tol} trials={args.trials} bits={bit_levels}")

    if not _wait_in_world(base):
        print(f"[arena] FATAL: no player in world at {base}", file=sys.stderr)
        return 2

    # Build the controlled course and lock it down. --no-build preserves a
    # hand-placed obstacle course (re-filling would destroy it).
    if args.no_build:
        print("[arena] --no-build: preserving existing course (NOT re-filling)")
        _say(args.relay, f"[Sprint A] obstacle run @ {anchor} (no rebuild), allowBreak off")
    else:
        print("[arena] building flat arena + disabling allowBreak ...")
        _say(args.relay, f"[Sprint A] building flat arena @ {anchor}, allowBreak off")
        build_flat_arena(args.relay, args.player, anchor, args.radius)
    ab = _set_allow_break(base, False)
    print(f"[arena] allow_break -> {ab}")
    tp_to(args.relay, args.player, start[0] + 0.5, start[1], start[2] + 0.5)
    time.sleep(1.0)
    p = _pos(base)
    print(f"[arena] player at {p.get('x'):.1f},{p.get('y'):.1f},{p.get('z'):.1f}" if p.get("x") is not None else "[arena] WARN no pos")

    results = []
    # control first (lossless), then descending bits
    print("\n[arena] === CONTROL (lossless) ===")
    results.append(sweep_bit_level(base, args.relay, args.codec_cfg_url, args.codec_url,
                                   args.player, None, start, out_xyz, args.tol,
                                   args.leg_timeout, args.trials))
    for b in bit_levels:
        print(f"\n[arena] === b={b} ===")
        results.append(sweep_bit_level(base, args.relay, args.codec_cfg_url, args.codec_url,
                                       args.player, b, start, out_xyz, args.tol,
                                       args.leg_timeout, args.trials))

    # restore lossless so nothing stays silently lossy
    _set_quant(args.codec_cfg_url, None)
    _say(args.relay, "[Sprint A] sweep complete")

    out_doc = {
        "base": base, "anchor": anchor, "start": start, "out_xyz": out_xyz,
        "tol": args.tol, "trials": args.trials, "delta": args.delta,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\n[arena] wrote {args.out}")

    # compact table
    print("\n" + "=" * 64)
    print("SPRINT A — flat-arena parity vs bits (CONTINUOUS metric primary)")
    print("=" * 64)
    print(f"{'bits':>5} {'reach_rate':>10} {'dist_mean':>9} {'dist_p90':>8} {'dist_max':>8} "
          f"{'subst':>6} {'drift':>6} {'p99ms':>6}")
    for r in results:
        c = r["counters"]
        print(f"{r['bits']!s:>5} {r['reached_rate']!s:>10} {r['dist_mean']!s:>9} "
              f"{r['dist_p90']!s:>8} {r['dist_max']!s:>8} "
              f"{c['substituted']!s:>6} {c['drift']!s:>6} {c['subst_latency_p99_ms']!s:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
