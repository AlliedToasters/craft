#!/usr/bin/env python3
"""Sprint A — POS_RANGE sweep: the CONTINUOUS knee lever.

The bit sweep showed the parity knee is intrinsically sharp: one bit = 2x
resolution, and the single step from b6 (min step 0.254 ~ walk delta, reach 1.0)
to b5 (0.516, reach 0.0) straddles the walk-delta threshold with no integer bit
between. The obstacle sweep confirmed geometry is NOT the lever.

The TRUE lever is the quantizer span POS_RANGE: min step = 2*range/(2**bits-1) is
range-linear, so at FIXED bits, shrinking the range slides the knee continuously
and lets several settings land inside the b5-b6 transition -> a GRADED reach
curve. This sweeps range at fixed bits and reports reach vs the resulting min
step (the real x-axis of the parity curve).

CAVEAT: shrinking range below the true delta range CLIPS large excursions. The
flat out-and-back arena has only ~0.21 walk deltas (no falls), so clipping never
fires here; this curve is valid for locomotion, not for fall-bearing motion.

Range is a GLOBAL server setting (like bits) -> barrier-synchronized: set
(bits,range) once, release all agents, aggregate, advance. Obstacle fixed at
size 1 (the obstacle sweep proved size doesn't move the knee).

Usage:
    .venv/bin/python -m experiments.codec_loop.range_sweep \
        --agents 0-14 --bits 5 --ranges 8,6,5,4,3,2,1 --trials 3 \
        --out results/sprintA/range_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from experiments.codec_loop.sprintA_arena import _say, _wait_in_world
from experiments.codec_loop.sprintA_fleet import (
    _run_concurrent,
    aggregate_level,
    parse_agents,
    run_level_for_agent,
    setup_agent,
)

CODEC_BASE = "http://127.0.0.1:25600"


def _codec_get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{CODEC_BASE}{path}", timeout=6) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def _set_quant_range(cfg_url: str, bits: int | None, pos_range: float | None) -> dict:
    """Set the global quant level AND the pos quantizer span in one POST.
    bits=None -> lossless (range ignored). bits set + range set -> lossy with the
    given span."""
    body = {"quant_bits": bits, "pos_range": pos_range}
    data = json.dumps(body).encode()
    req = urllib.request.Request(cfg_url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def min_step(bits: int, pos_range: float) -> float:
    """The smallest representable pos delta = the resolution that gates parity."""
    levels = (1 << bits) - 1
    return round(2.0 * pos_range / levels, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A POS_RANGE (continuous knee) sweep")
    ap.add_argument("--agents", default="0-14")
    ap.add_argument("--bits", type=int, default=5, help="FIXED bit width; range is the swept lever")
    ap.add_argument("--ranges", default="8,6,5,4,3,2,1", help="pos_range values to sweep")
    ap.add_argument("--host", default=os.environ.get("HOMUNCULUS_HOST", "127.0.0.1"))
    ap.add_argument("--port-base", type=int, default=25570)
    ap.add_argument("--relay", default=os.environ.get("MC_SERVER_CMD_BASE", "http://127.0.0.1:4747"))
    ap.add_argument("--codec-url", default=f"{CODEC_BASE}/codec/roundtrip")
    ap.add_argument("--codec-cfg-url", default=f"{CODEC_BASE}/config")
    ap.add_argument("--anchor-base", default="6000,100,6000")
    ap.add_argument("--spacing", type=int, default=512)
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--leg-timeout", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--obstacle-size", type=int, default=1)
    ap.add_argument("--obstacle-height", type=int, default=3)
    ap.add_argument("--fail-thresh", type=float, default=2.5)
    ap.add_argument("--out", default="results/sprintA/range_sweep.json")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    ranges = [float(r) for r in args.ranges.split(",") if r.strip()]
    bx, by, bz = (int(v) for v in args.anchor_base.split(","))

    ctx_all = []
    for i, n in enumerate(agents):
        ctx_all.append({
            "n": n, "player": f"agent{n}",
            "base": f"http://{args.host}:{args.port_base + n}",
            "anchor": (bx + args.spacing * i, by, bz),
        })

    print(f"[rangesweep] agents={agents} bits={args.bits} ranges={ranges} trials={args.trials}")
    print(f"[rangesweep] min_step by range: " +
          ", ".join(f"r{r}={min_step(args.bits, r)}" for r in ranges))
    missing = [c["n"] for c in ctx_all if not _wait_in_world(c["base"], 8)]
    if missing:
        print(f"[rangesweep] FATAL: not in world: {missing}", file=sys.stderr)
        return 2

    # setup once, sequential + robust
    _say(args.relay, f"[rangesweep] building {len(ctx_all)} arenas (size {args.obstacle_size}) ...")
    print("[rangesweep] === SETUP ===")
    setup = {}
    good = []
    for c in ctx_all:
        s = setup_agent(c["base"], args.relay, c["player"], c["anchor"],
                        args.delta, args.radius, "wall", args.obstacle_size,
                        args.obstacle_height)
        setup[c["n"]] = s
        if s.get("ok"):
            good.append(c)
        print(f"[rangesweep]   setup agent{c['n']}: ok={s.get('ok')} y_off={s.get('y_off')}")
    print(f"[rangesweep] setup_ok = {len(good)}/{len(ctx_all)}")
    if not good:
        print("[rangesweep] FATAL: no arenas", file=sys.stderr)
        return 3

    # sweep: lossless control, then each range at fixed bits
    levels: list = [("loss", None, None)] + [("b%d_r%s" % (args.bits, r), args.bits, r) for r in ranges]
    results = []
    for label, bits, rng in levels:
        _codec_get("/stats/reset")
        cfg = _set_quant_range(args.codec_cfg_url, bits, rng)
        time.sleep(0.3)
        ms = None if bits is None else min_step(bits, rng)
        t0 = time.time()
        per = _run_concurrent(
            lambda c, b=bits: run_level_for_agent(
                c["base"], args.relay, args.codec_url, c["player"], c["anchor"],
                args.delta, args.tol, args.leg_timeout, args.trials, b,
                args.fail_thresh),
            good, label)
        grp = [per[c["n"]] for c in good if "_error" not in per.get(c["n"], {})]
        agg = aggregate_level(bits, grp)
        agg.pop("per_agent", None)
        agg["label"] = label
        agg["pos_range"] = rng
        agg["min_step"] = ms
        agg["wall_s"] = round(time.time() - t0, 1)
        agg["cfg_applied"] = cfg.get("quant")
        health = _codec_get("/healthz")
        agg["codec_stats"] = health.get("stats")
        results.append(agg)
        _say(args.relay, f"[rangesweep] {label}: min_step={ms} reach={agg['reach_rate']}")
        print(f"[rangesweep] {label:>10} range={rng} min_step={ms} "
              f"reach={agg['reach_rate']} dist_mean={agg['dist_mean']} "
              f"subst={agg['substituted']} ({agg['wall_s']}s)")

    _set_quant_range(args.codec_cfg_url, None, None)  # restore lossless
    _say(args.relay, "[rangesweep] complete")

    out_doc = {
        "agents": agents, "bits": args.bits, "ranges": ranges,
        "obstacle_size": args.obstacle_size, "trials": args.trials,
        "delta": args.delta, "leg_timeout": args.leg_timeout,
        "fail_thresh": args.fail_thresh, "setup_ok": len(good),
        "setup": setup, "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[rangesweep] wrote {args.out}")

    # headline: reach vs min_step (the continuous parity curve)
    print("\n" + "=" * 64)
    print(f"POS_RANGE SWEEP @ b={args.bits} — reach vs min representable step")
    print("=" * 64)
    print(f"{'label':>12} {'range':>6} {'min_step':>9} {'reach':>6} {'dist_mn':>8} {'wall_s':>7}")
    for r in results:
        print(f"{r['label']:>12} {str(r['pos_range']):>6} {str(r['min_step']):>9} "
              f"{str(r['reach_rate']):>6} {str(r['dist_mean']):>8} {r['wall_s']:>7}")
    print("\n(a column of intermediate reach values between 0 and 1 = the GRADED knee)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
