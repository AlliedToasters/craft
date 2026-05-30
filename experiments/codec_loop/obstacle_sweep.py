#!/usr/bin/env python3
"""Sprint A — obstacle-size x bits sweep: recover the GRADED knee.

The n-sweep proved the b=5 knee is n-INDEPENDENT (real loss-intolerance) and that
the auto 3x3x3 box turned the historical graded b5-6 knee into a hard cliff
(reach 0.0 even at n=1). So the obstacle footprint is the lever on knee SHARPNESS.
This sweeps it.

Concurrency design (the speedup the user noted): the codec quant is ONE global
level, but the obstacle is PER-ARENA. So assign different obstacle sizes to
different agent GROUPS and run them all at the same global bit level at once:
  agents split into len(sizes) contiguous groups -> group g gets sizes[g].
  For each bit level: set global quant, release ALL agents concurrently, then
  aggregate results grouped by obstacle size.
With 15 agents / 3 sizes that's 5 agents per (size,bit) cell (30 legs) and the
whole size x bit MATRIX costs ~one n-sweep column of wall-clock.

Output: a (size x bit) reach matrix. The size whose reach degrades GRADUALLY
across bits (not 1.0 -> 0.0 in one step) is the geometry that reproduces the
graded knee for the lossy parity curve.

Usage:
    .venv/bin/python -m experiments.codec_loop.obstacle_sweep \
        --agents 0-14 --sizes 1,2,3 --bits 8,7,6,5,4 --trials 3 \
        --out results/sprintA/obstacle_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from experiments.codec_loop.sprintA_arena import _say, _set_quant, _wait_in_world
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


def assign_sizes(agents: list[int], sizes: list[int]) -> dict[int, int]:
    """Split agents into len(sizes) contiguous groups; group g -> sizes[g].
    Contiguous (not round-robin) so each size's arenas are spatially clustered,
    which keeps any residual chunk effects size-correlated and visible rather
    than smeared across the fleet."""
    out: dict[int, int] = {}
    g = len(sizes)
    per = len(agents) // g
    rem = len(agents) % g
    idx = 0
    for gi, sz in enumerate(sizes):
        # distribute remainder onto the earliest groups
        count = per + (1 if gi < rem else 0)
        for _ in range(count):
            out[agents[idx]] = sz
            idx += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A obstacle-size x bits sweep")
    ap.add_argument("--agents", default="0-14")
    ap.add_argument("--sizes", default="1,2,3", help="obstacle footprints to compare (odd -> centered)")
    ap.add_argument("--bits", default="8,7,6,5,4", help="bit levels swept at every size")
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
    ap.add_argument("--obstacle-height", type=int, default=3)
    ap.add_argument("--fail-thresh", type=float, default=2.5)
    ap.add_argument("--out", default="results/sprintA/obstacle_sweep.json")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]
    bx, by, bz = (int(v) for v in args.anchor_base.split(","))
    size_of = assign_sizes(agents, sizes)

    ctx_all = []
    for i, n in enumerate(agents):
        ctx_all.append({
            "n": n, "player": f"agent{n}",
            "base": f"http://{args.host}:{args.port_base + n}",
            "anchor": (bx + args.spacing * i, by, bz),
            "size": size_of[n],
        })

    print(f"[obssweep] agents={agents} sizes={sizes} bits={bit_levels} trials={args.trials}")
    for c in ctx_all:
        print(f"[obssweep]   agent{c['n']:>2} size={c['size']} anchor={c['anchor']}")
    missing = [c["n"] for c in ctx_all if not _wait_in_world(c["base"], 8)]
    if missing:
        print(f"[obssweep] FATAL: not in world: {missing} (launch them first)", file=sys.stderr)
        return 2

    # --- setup ONCE, sequential + re-fill-on-fail (robust 15/15) --------------
    _say(args.relay, f"[obssweep] building {len(ctx_all)} arenas, sizes={sizes} ...")
    print("[obssweep] === SETUP (sequential, robust) ===")
    setup = {}
    good = []
    for c in ctx_all:
        s = setup_agent(c["base"], args.relay, c["player"], c["anchor"],
                        args.delta, args.radius, "wall", c["size"],
                        args.obstacle_height)
        s["size"] = c["size"]
        setup[c["n"]] = s
        if s.get("ok"):
            good.append(c)
        print(f"[obssweep]   setup agent{c['n']} size={c['size']}: ok={s.get('ok')} y_off={s.get('y_off')}")
    print(f"[obssweep] setup_ok = {len(good)}/{len(ctx_all)}")
    if not good:
        print("[obssweep] FATAL: no arenas set up", file=sys.stderr)
        return 3

    # --- sweep: each bit level runs ALL agents at once; group results by size --
    levels = [None] + bit_levels
    results = []  # one entry per (bits, size) cell
    for bits in levels:
        _codec_get("/stats/reset")
        _set_quant(args.codec_cfg_url, bits)
        time.sleep(0.3)
        label = "loss" if bits is None else f"b{bits}"
        t0 = time.time()
        per = _run_concurrent(
            lambda c, b=bits: run_level_for_agent(
                c["base"], args.relay, args.codec_url, c["player"], c["anchor"],
                args.delta, args.tol, args.leg_timeout, args.trials, b,
                args.fail_thresh),
            good, label)
        wall = round(time.time() - t0, 1)
        health = _codec_get("/healthz")
        cstats = health.get("stats")
        # group this level's per-agent results by obstacle size
        for sz in sizes:
            grp = [per[c["n"]] for c in good
                   if c["size"] == sz and "_error" not in per.get(c["n"], {})]
            if not grp:
                continue
            agg = aggregate_level(bits, grp)
            agg.pop("per_agent", None)
            agg["size"] = sz
            agg["wall_s"] = wall
            agg["codec_stats"] = cstats
            results.append(agg)
        # progress line per level
        row = " | ".join(
            f"sz{sz}:{next((r['reach_rate'] for r in results if r['bits']==bits and r['size']==sz), None)}"
            for sz in sizes)
        _say(args.relay, f"[obssweep] {label}: {row}")
        print(f"[obssweep] {label} ({wall}s): {row}")

    _set_quant(args.codec_cfg_url, None)  # restore lossless
    _say(args.relay, "[obssweep] complete")

    out_doc = {
        "agents": agents, "sizes": sizes, "bits": bit_levels,
        "size_of": {str(k): v for k, v in size_of.items()},
        "trials": args.trials, "delta": args.delta, "leg_timeout": args.leg_timeout,
        "obstacle_height": args.obstacle_height, "fail_thresh": args.fail_thresh,
        "setup_ok": len(good), "setup": setup, "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[obssweep] wrote {args.out}")

    # --- headline: reach matrix (rows=bits incl lossless, cols=size) ----------
    print("\n" + "=" * 60)
    print("OBSTACLE-SIZE x BITS — reach matrix (knee-shape per geometry)")
    print("=" * 60)
    hdr = f"{'level':>6} " + " ".join(f"{'sz'+str(sz):>7}" for sz in sizes)
    print(hdr)
    for bits in levels:
        lvl = "loss" if bits is None else f"b{bits}"
        cells = []
        for sz in sizes:
            r = next((x for x in results if x["bits"] == bits and x["size"] == sz), None)
            cells.append(f"{r['reach_rate']:>7}" if r else f"{'-':>7}")
        print(f"{lvl:>6} " + " ".join(cells))
    print("\n(legs/cell = trials*2*agents_per_size; a GRADED column = the geometry "
          "that gives a real knee)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
