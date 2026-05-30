#!/usr/bin/env python3
"""Sprint A — pos-quantizer grid A/B: zero_biased vs zero_preserving.

THE QUESTION: the b5 cliff (reach 1.0@b6 -> 0.0@b5) was traced to the
zero_biased pos grid injecting a DC drift on stationary packets (recon_hist.py:
0.0 not representable -> ±R/(2^b-1) on 56% of traffic -> rubberband). The
principled fix is a zero_PRESERVING grid (a code sits on 0, so a still player
reconstructs to exactly 0). Does it buy back a bit-level?

THE CATCH (offline /tmp/predict_zeromode.json, verified before this run):
zero_preserving removes the drift (stat_rmse -> 0 at every range) but introduces
a DEADBAND -- the half-step swallows slow-walk deltas (~0.21): 54% of walk-band
deltas round to 0 at R8, falling to 6% at R3. So the two grids have OPPOSITE
failure horns at coarse resolution:
  * zero_biased   fails by PHANTOM DRIFT  (still player twitches)
  * zero_preserving fails by DEADBAND     (walking player can't register motion)
A single b5/R8 A/B point would show BOTH failing and be misread. So we sweep
RANGE at fixed b5 for BOTH modes: this separates the horns and finds each grid's
knee. If zero_preserving's knee sits at a LARGER range than zero_biased's, the
fix buys resolution headroom; if equal/smaller, the cliff is raw-resolution-bound
and zero-phase is a red herring.

Barrier-synchronized like range_sweep (bits+range+mode are GLOBAL server config).
Obstacle fixed at size 1 (obstacle sweep proved size doesn't move the knee).

Usage:
    .venv/bin/python -m experiments.codec_loop.zero_mode_ab \
        --agents 0-14 --bits 5 --ranges 8,6,5,4,3 --trials 2 \
        --out results/sprintA/zero_mode_ab.json
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


def _set_cfg(cfg_url: str, bits: int | None, pos_range: float | None,
             pos_mode: str) -> dict:
    """Set global quant level + pos span + pos grid mode in one POST.
    bits=None -> lossless (range/mode irrelevant). Always sends pos_mode so the
    server's grid is explicit per cell (no carry-over from a prior cell)."""
    body = {"quant_bits": bits, "pos_range": pos_range, "pos_mode": pos_mode}
    data = json.dumps(body).encode()
    req = urllib.request.Request(cfg_url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def step_for(mode: str, bits: int, pos_range: float) -> float:
    """Min representable step for each grid (the resolution that gates parity).
    zero_biased: 2R/(2^b-1) over 2^b points. zero_preserving: R/(2^(b-1)-1)."""
    if mode == "zero_preserving":
        k = (1 << (bits - 1)) - 1
        return round(pos_range / k, 4) if k else float("inf")
    return round(2.0 * pos_range / ((1 << bits) - 1), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A zero_biased vs zero_preserving grid A/B")
    ap.add_argument("--agents", default="0-14")
    ap.add_argument("--bits", type=int, default=5, help="FIXED bit width for the A/B")
    ap.add_argument("--ranges", default="8,6,5,4,3", help="pos_range values swept per mode")
    ap.add_argument("--modes", default="zero_biased,zero_preserving")
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
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--obstacle-size", type=int, default=1)
    ap.add_argument("--obstacle-height", type=int, default=3)
    ap.add_argument("--fail-thresh", type=float, default=2.5)
    ap.add_argument("--out", default="results/sprintA/zero_mode_ab.json")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    ranges = [float(r) for r in args.ranges.split(",") if r.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bx, by, bz = (int(v) for v in args.anchor_base.split(","))

    ctx_all = []
    for i, n in enumerate(agents):
        ctx_all.append({
            "n": n, "player": f"agent{n}",
            "base": f"http://{args.host}:{args.port_base + n}",
            "anchor": (bx + args.spacing * i, by, bz),
        })

    print(f"[zeroAB] agents={agents} bits={args.bits} ranges={ranges} modes={modes} trials={args.trials}")
    for m in modes:
        print(f"[zeroAB] step[{m}] by range: " +
              ", ".join(f"r{r}={step_for(m, args.bits, r)}" for r in ranges))
    missing = [c["n"] for c in ctx_all if not _wait_in_world(c["base"], 8)]
    if missing:
        print(f"[zeroAB] FATAL: not in world: {missing}", file=sys.stderr)
        return 2

    _say(args.relay, f"[zeroAB] building {len(ctx_all)} arenas (size {args.obstacle_size}) ...")
    print("[zeroAB] === SETUP ===")
    setup = {}
    good = []
    for c in ctx_all:
        s = setup_agent(c["base"], args.relay, c["player"], c["anchor"],
                        args.delta, args.radius, "wall", args.obstacle_size,
                        args.obstacle_height)
        setup[c["n"]] = s
        if s.get("ok"):
            good.append(c)
        print(f"[zeroAB]   setup agent{c['n']}: ok={s.get('ok')} y_off={s.get('y_off')}")
    print(f"[zeroAB] setup_ok = {len(good)}/{len(ctx_all)}")
    if not good:
        print("[zeroAB] FATAL: no arenas", file=sys.stderr)
        return 3

    # lossless control, then each (mode, range) cell at fixed bits. Range-major
    # so the per-range mode PAIR is adjacent in the output (easy horn compare).
    levels: list = [("loss", None, None, "zero_biased")]
    for rng in ranges:
        for mode in modes:
            levels.append((f"{mode[:2]}_r{rng}", args.bits, rng, mode))

    results = []
    for label, bits, rng, mode in levels:
        _codec_get("/stats/reset")
        cfg = _set_cfg(args.codec_cfg_url, bits, rng, mode)
        time.sleep(0.3)
        st = None if bits is None else step_for(mode, bits, rng)
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
        agg["pos_mode"] = mode
        agg["step"] = st
        agg["wall_s"] = round(time.time() - t0, 1)
        agg["cfg_applied"] = cfg.get("quant")
        health = _codec_get("/healthz")
        agg["codec_stats"] = health.get("stats")
        results.append(agg)
        _say(args.relay, f"[zeroAB] {label}: step={st} reach={agg['reach_rate']}")
        print(f"[zeroAB] {label:>14} mode={mode:>15} range={rng} step={st} "
              f"reach={agg['reach_rate']} dist_mean={agg['dist_mean']} "
              f"subst={agg['substituted']} ({agg['wall_s']}s)")

    _set_cfg(args.codec_cfg_url, None, None, "zero_biased")  # restore lossless
    _say(args.relay, "[zeroAB] complete")

    out_doc = {
        "agents": agents, "bits": args.bits, "ranges": ranges, "modes": modes,
        "obstacle_size": args.obstacle_size, "trials": args.trials,
        "delta": args.delta, "leg_timeout": args.leg_timeout,
        "fail_thresh": args.fail_thresh, "setup_ok": len(good),
        "setup": setup, "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[zeroAB] wrote {args.out}")

    print("\n" + "=" * 70)
    print(f"ZERO-GRID A/B @ b={args.bits} — reach by (mode, range)")
    print("=" * 70)
    print(f"{'label':>14} {'mode':>16} {'range':>6} {'step':>7} {'reach':>6} {'dist_mn':>8} {'wall_s':>7}")
    for r in results:
        print(f"{r['label']:>14} {str(r['pos_mode']):>16} {str(r['pos_range']):>6} "
              f"{str(r['step']):>7} {str(r['reach_rate']):>6} {str(r['dist_mean']):>8} {r['wall_s']:>7}")
    print("\nKnee = highest range with reach 1.0, per mode. zero_preserving knee at a "
          "LARGER range than zero_biased => the fix buys headroom; equal/smaller => "
          "the cliff is raw-resolution-bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
