#!/usr/bin/env python3
"""§14 codec-in-the-loop rung driver — sequences Rungs 0→1→2 over a LIVE agent.

The codec experiment's "close the loop" test ladder (neural_interface.md §14):

  Rung 0  observer dry-run    — /codec/passthrough {substitute:false}.
          The Python semantic codec round-trips every allowlisted outbound
          packet but does NOT touch the wire. PASS = drift==0 and
          transport_errors==0. Proves the codec is identity-in-practice at
          live data rate. Safe precondition for substitution.

  Rung 1  byte-identity ctrl  — /packets/roundtrip {enabled:true}.
          In-process Mojang StreamCodec round-trip (no Python, no network).
          PASS = byte_mismatch==0, encode_failed==0, decode_failed==0, no
          disconnect. Isolates plumbing faults from encoding faults: if
          behaviour moves HERE, the substitution machinery is the bug.

  Rung 2  THE TEST           — /codec/passthrough {substitute:true}.
          The Python codec's decoded fields are reconstructed into packets
          that go on the wire IN PLACE of the originals — the controller
          plays through the full identity codec. PASS = the controller
          reaches its targets (position-based, see below), substitute_errors
          and drift ~0, and substitute latency (the sync POST on the netty
          send thread @20Hz) within a no-desync budget.

ARRIVAL IS POSITION-BASED, NOT reason=="arrived".  Baritone sometimes reports
PathEvent.CANCELED even when the player ends up at the target within tolerance
(and that cancel reproduces with the codec OFF — it is target-pathing noise,
not a codec fault). So we judge each goto by final distance-to-target, not by
the success/reason string. Byte-precise equality is explicitly NOT the bar for
Rung 2 (fields_close reconstructs from floats); behavioural parity — the
controller reaching its goals — is.

Each rung drives traffic with two Baritone gotos (out-and-back from spawn) so
movement packets actually flow, reads the counters, and judges PASS/FAIL.
Reuses existing homunculus routes — does NOT reimplement spawn/agent.

Usage:
    .venv/bin/python -m experiments.codec_loop.run_rungs               # canonical agent
    .venv/bin/python -m experiments.codec_loop.run_rungs --port 25570  # fleet agent0
    .venv/bin/python -m experiments.codec_loop.run_rungs --rungs 0,2   # subset
    HOMUNCULUS_PORT=25570 .venv/bin/python -m experiments.codec_loop.run_rungs

Exit code is informational (0 = all selected rungs PASS); the printed table is
the truth.
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


def _resolve_base(port_override: int | None) -> str:
    """Homunculus base URL. Prefer craft.config; else --port / env; else 25566."""
    if port_override is not None:
        host = os.environ.get("HOMUNCULUS_HOST", "127.0.0.1")
        return f"http://{host}:{port_override}"
    try:
        from craft import config as _cfg  # type: ignore
        base = getattr(_cfg, "HOMUNCULUS_BASE", None)
        if base:
            return base.rstrip("/")
    except Exception:
        pass
    host = os.environ.get("HOMUNCULUS_HOST", "127.0.0.1")
    port = os.environ.get("HOMUNCULUS_PORT", "25566")
    return f"http://{host}:{port}"


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


def _pos(base: str) -> dict:
    return _http("GET", f"{base}/position", timeout=8)


def _in_world(base: str) -> bool:
    return _pos(base).get("x") is not None


def _wait_in_world(base: str, timeout_s: float = 240.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _in_world(base):
            return True
        time.sleep(6)
    return False


def _goto(base: str, x: int, y: int, z: int, tol: int = 2, t: int = 50) -> dict:
    return _http(
        "POST", f"{base}/baritone/goto",
        {"x": x, "y": y, "z": z, "timeout_seconds": t, "arrival_tolerance": tol},
        timeout=t + 15,
    )


def _reached(outcome: dict, target: tuple[int, int, int], tol: float) -> bool:
    """Position-based arrival: did the player END UP at the target within tol?

    Tolerant of Baritone's spurious PathEvent.CANCELED (see module docstring) —
    we trust the final position, not the reason string. Uses horizontal+vertical
    distance with a small slack over the requested tolerance for float wobble.
    """
    fp = outcome.get("final_position")
    if not fp or len(fp) != 3:
        return False
    dx, dy, dz = fp[0] - target[0], fp[1] - target[1], fp[2] - target[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist <= tol + 1.5


def _drive(base: str, out_xyz, home_xyz, tol: int) -> list[dict]:
    a = _goto(base, *out_xyz, tol=tol)
    b = _goto(base, *home_xyz, tol=tol)
    return [
        {"target": out_xyz, "outcome": a, "reached": _reached(a, out_xyz, tol)},
        {"target": home_xyz, "outcome": b, "reached": _reached(b, home_xyz, tol)},
    ]


def _all_reached(legs: list[dict]) -> bool:
    return all(l["reached"] for l in legs)


# ---- Rungs --------------------------------------------------------------

def rung0(base, codec_url, out_xyz, home_xyz, tol):
    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": False})
    legs = _drive(base, out_xyz, home_xyz, tol)
    final = _http("POST", f"{base}/codec/passthrough/disarm")
    passed = (final.get("drift") == 0 and final.get("transport_errors") == 0
              and final.get("attempted", 0) > 0 and _all_reached(legs))
    return {"rung": "0 observer", "passed": passed, "counters": final, "legs": legs}


def rung1(base, out_xyz, home_xyz, tol):
    _http("POST", f"{base}/packets/roundtrip", {"enabled": True})
    legs = _drive(base, out_xyz, home_xyz, tol)
    final = _http("GET", f"{base}/packets/roundtrip")
    _http("POST", f"{base}/packets/roundtrip", {"enabled": False})
    passed = (final.get("byte_mismatch") == 0 and final.get("encode_failed") == 0
              and final.get("decode_failed") == 0 and final.get("roundtripped", 0) > 0
              and _all_reached(legs))
    return {"rung": "1 byte-identity", "passed": passed, "counters": final, "legs": legs}


def rung2(base, codec_url, out_xyz, home_xyz, tol, latency_budget_ms):
    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": True})
    legs = _drive(base, out_xyz, home_xyz, tol)
    # snapshot latency BEFORE disarm (disarm body omits latency fields)
    snap = _http("GET", f"{base}/codec/passthrough/status")
    final = _http("POST", f"{base}/codec/passthrough/disarm")
    p99 = snap.get("subst_latency_p99_ms")
    passed = (_all_reached(legs) and final.get("substituted", 0) > 0
              and final.get("drift") == 0 and final.get("substitute_errors") == 0
              and final.get("transport_errors") == 0
              and (p99 is None or p99 <= latency_budget_ms))
    # carry the latency fields forward from the live snapshot
    merged = dict(final)
    for k in ("subst_latency_count", "subst_latency_mean_ms", "subst_latency_p50_ms",
              "subst_latency_p99_ms", "subst_latency_max_ms"):
        if k in snap:
            merged[k] = snap[k]
    return {"rung": "2 THE TEST", "passed": passed, "counters": merged, "legs": legs}


# ---- Reporting ----------------------------------------------------------

def _print_table(results):
    print("\n" + "=" * 72)
    print("§14 CODEC-IN-THE-LOOP — RUNG RESULTS")
    print("=" * 72)
    for r in results:
        verdict = "PASS" if r["passed"] else "FAIL"
        reached = sum(l["reached"] for l in r.get("legs", []))
        total = len(r.get("legs", []))
        print(f"\n[{verdict}] Rung {r['rung']}   targets_reached={reached}/{total}")
        c = r.get("counters", {})
        if "_transport_error" in c:
            print(f"    transport error: {c['_transport_error']}")
            continue
        keys = [
            "attempted", "ok", "drift", "transport_errors",
            "roundtripped", "passed_through", "encode_failed", "decode_failed",
            "byte_mismatch",
            "substituted", "substitute_fallbacks", "substitute_errors", "no_obs",
            "subst_latency_count", "subst_latency_mean_ms",
            "subst_latency_p50_ms", "subst_latency_p99_ms", "subst_latency_max_ms",
        ]
        for k in keys:
            if k in c:
                print(f"    {k:24s} = {c[k]}")
    allpass = all(r["passed"] for r in results)
    print("\n" + "-" * 72)
    print(f"OVERALL: {'ALL PASS' if allpass else 'SOME FAILED'} "
          f"({sum(r['passed'] for r in results)}/{len(results)})")
    print("-" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="§14 codec-in-the-loop rung driver")
    ap.add_argument("--port", type=int, default=None,
                    help="homunculus port (default: craft.config / HOMUNCULUS_PORT / 25566)")
    ap.add_argument("--codec-url", default=None,
                    help="codec roundtrip endpoint (default: CODEC_URL env or http://127.0.0.1:25600/codec/roundtrip)")
    ap.add_argument("--rungs", default="0,1,2", help="comma list of rungs (default 0,1,2)")
    ap.add_argument("--latency-budget-ms", type=float, default=10.0,
                    help="Rung-2 p99 substitute-latency budget (default 10ms ≈ 20%% of a 50ms tick)")
    ap.add_argument("--delta", type=int, default=28, help="goto displacement from spawn (out-and-back)")
    ap.add_argument("--tol", type=int, default=2, help="arrival tolerance (blocks)")
    ap.add_argument("--out", default=None, help="write results JSON to PATH")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    codec_url = (args.codec_url or os.environ.get("CODEC_URL")
                 or "http://127.0.0.1:25600/codec/roundtrip")
    want = {s.strip() for s in args.rungs.split(",") if s.strip()}

    print(f"[driver] homunculus base = {base}")
    print(f"[driver] codec url       = {codec_url}")
    print(f"[driver] rungs           = {sorted(want)}")

    if not _wait_in_world(base):
        print(f"[driver] FATAL: no player in world at {base} after wait — "
              f"is the agent up and joined? (curl {base}/position)", file=sys.stderr)
        return 2

    p0 = _pos(base)
    sx, sy, sz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    d = args.delta
    out_xyz = (sx + d, sy, sz + d)
    home_xyz = (sx, sy, sz)
    print(f"[driver] spawn=({sx},{sy},{sz}) out={out_xyz} home={home_xyz} tol={args.tol}")

    results = []
    if "0" in want:
        print("\n[driver] === Rung 0: observer dry-run ===")
        results.append(rung0(base, codec_url, out_xyz, home_xyz, args.tol))
    if "1" in want:
        print("[driver] === Rung 1: byte-identity control ===")
        results.append(rung1(base, out_xyz, home_xyz, args.tol))
    if "2" in want:
        print("[driver] === Rung 2: THE TEST (semantic substitution) ===")
        results.append(rung2(base, codec_url, out_xyz, home_xyz, args.tol,
                             args.latency_budget_ms))

    _print_table(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"base": base, "codec_url": codec_url, "results": results}, f, indent=2)
        print(f"\n[driver] wrote {args.out}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
