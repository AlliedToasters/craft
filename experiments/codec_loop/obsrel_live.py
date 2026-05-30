#!/usr/bin/env python3
"""§16.1 — LIVE behavioral-parity sweep for the obs-relative-rotation baseline.

Offline (obsrel_baseline.py) showed obs-relative rotation dominates the RD
frontier and holds heading at-rest exactly. But offline fidelity != behavioral
parity (the §15 discipline). THE QUESTION: at low rotation bits the obs-relative
codec heavily DEADBANDS turns (zero_preserving step = 180/(2^(b-1)-1): 5.8°@b6,
12°@b5, 26°@b4, 60°@b3). Does the controller still reach its goto targets — i.e.
is rotation DEADBAND benign like position deadband (§15), because Baritone
re-issues the heading every tick? Or is there a rotation knee absolute coding
hid?

Method = the §14 Rung-2 harness (run_rungs.rung2: arm substitute:true, drive an
out-and-back Baritone goto, judge by POSITION-based arrival, not Baritone reason).
POSITION is held near-lossless (pos_bits high) so the test isolates ROTATION.
Per cell: set the sidecar /config, run N trials, aggregate reach-rate + drift +
substitute latency. Control = lossless identity (§14 codec-off-equivalent).

Run under PEACEFUL (no mobs perturbing the gotos). One agent, sequential cells.

Usage:
    .venv/bin/python -m experiments.codec_loop.obsrel_live --port 25570 \
        --bits 6,5,4,3,2 --trials 3 --out results/sprint16/obsrel_live.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from experiments.codec_loop.run_rungs import (
    _resolve_base, _wait_in_world, _pos, rung2,
)


def _sidecar_base(codec_url: str) -> str:
    # codec_url is http://host:port/codec/roundtrip -> strip the path
    i = codec_url.find("/codec/roundtrip")
    return codec_url[:i] if i > 0 else codec_url


def _config(sidecar: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{sidecar}/config", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _rot_step_deg(bits: int) -> float:
    k = (1 << (bits - 1)) - 1
    return 180.0 / k if k > 0 else 360.0


def _run_cell(base, codec_url, out_xyz, home_xyz, tol, trials, latency_budget_ms):
    """trials × rung2; return aggregate reach + worst-case counters."""
    legs_reached = 0
    legs_total = 0
    drift = 0
    subst_errors = 0
    transport_errors = 0
    substituted = 0
    p99s = []
    per_trial = []
    for t in range(trials):
        r = rung2(base, codec_url, out_xyz, home_xyz, tol, latency_budget_ms)
        c = r.get("counters", {})
        reached = sum(l["reached"] for l in r.get("legs", []))
        legs_reached += reached
        legs_total += len(r.get("legs", []))
        drift += c.get("drift", 0) or 0
        subst_errors += c.get("substitute_errors", 0) or 0
        transport_errors += c.get("transport_errors", 0) or 0
        substituted += c.get("substituted", 0) or 0
        if c.get("subst_latency_p99_ms") is not None:
            p99s.append(c["subst_latency_p99_ms"])
        # final positions for drift inspection
        finals = [l["outcome"].get("final_position") for l in r.get("legs", [])]
        per_trial.append({"reached": reached, "finals": finals,
                          "substituted": c.get("substituted"),
                          "drift": c.get("drift")})
        time.sleep(2)
    return {
        "reach_rate": round(legs_reached / legs_total, 3) if legs_total else 0.0,
        "legs_reached": legs_reached, "legs_total": legs_total,
        "drift": drift, "substitute_errors": subst_errors,
        "transport_errors": transport_errors, "substituted": substituted,
        "p99_latency_ms_max": max(p99s) if p99s else None,
        "per_trial": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--bits", default="6,5,4,3,2")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pos-bits", type=int, default=16,
                    help="position bits — kept HIGH to isolate the rotation effect")
    ap.add_argument("--delta", type=int, default=28)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--latency-budget-ms", type=float, default=10.0)
    ap.add_argument("--out", default="results/sprint16/obsrel_live.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    sidecar = _sidecar_base(args.codec_url)
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]

    print(f"[obsrel_live] base={base} sidecar={sidecar} pos_bits={args.pos_bits} "
          f"rot_bits={bit_levels} trials={args.trials} delta={args.delta}")
    if not _wait_in_world(base):
        print("[obsrel_live] FATAL: no player in world")
        return 2
    p0 = _pos(base)
    sx, sy, sz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    d = args.delta
    out_xyz = (sx + d, sy, sz + d)
    home_xyz = (sx, sy, sz)
    print(f"[obsrel_live] spawn=({sx},{sy},{sz}) out={out_xyz}")

    cells = []

    # control: lossless identity (codec-off-equivalent)
    print("\n[obsrel_live] === CONTROL (lossless identity) ===")
    _config(sidecar, {"quant_bits": None, "obsrel": False})
    agg = _run_cell(base, args.codec_url, out_xyz, home_xyz, args.tol,
                    args.trials, args.latency_budget_ms)
    print(f"  reach={agg['reach_rate']} ({agg['legs_reached']}/{agg['legs_total']}) "
          f"drift={agg['drift']} subst_err={agg['substitute_errors']} "
          f"substituted={agg['substituted']} p99={agg['p99_latency_ms_max']}ms")
    cells.append({"cell": "control", "rot_bits": None, "rot_step_deg": 0.0, **agg})

    # obs-relative rotation sweep
    for b in bit_levels:
        step = _rot_step_deg(b)
        print(f"\n[obsrel_live] === obs-rel rotation b{b} (step {step:.1f}°), "
              f"pos b{args.pos_bits} ===")
        _config(sidecar, {"obsrel": True, "pos_bits": args.pos_bits,
                          "yaw_bits": b, "pitch_bits": b, "pos_mode": "zero_preserving"})
        agg = _run_cell(base, args.codec_url, out_xyz, home_xyz, args.tol,
                        args.trials, args.latency_budget_ms)
        print(f"  reach={agg['reach_rate']} ({agg['legs_reached']}/{agg['legs_total']}) "
              f"drift={agg['drift']} subst_err={agg['substitute_errors']} "
              f"substituted={agg['substituted']} p99={agg['p99_latency_ms_max']}ms")
        cells.append({"cell": f"b{b}", "rot_bits": b, "rot_step_deg": round(step, 2), **agg})

    # reset sidecar to lossless
    _config(sidecar, {"quant_bits": None, "obsrel": False})

    out = {"base": base, "sidecar": sidecar, "pos_bits": args.pos_bits,
           "delta": args.delta, "tol": args.tol, "trials": args.trials, "cells": cells}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("§16.1 obs-relative rotation — LIVE PARITY SWEEP")
    print("=" * 60)
    print(f"{'cell':>8} {'rot_step':>9} {'reach':>6} {'drift':>6} {'subErr':>7} {'p99ms':>7}")
    for c in cells:
        print(f"{c['cell']:>8} {c['rot_step_deg']:>8.1f}° {c['reach_rate']:>6.2f} "
              f"{c['drift']:>6} {c['substitute_errors']:>7} "
              f"{str(c['p99_latency_ms_max']):>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
