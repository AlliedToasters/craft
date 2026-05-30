#!/usr/bin/env python3
"""Sprint A — does the b=5 knee MOVE with fleet size n?

Decides the one thing the codec load test left entangled: the single-agent run
got reach=0.63 at b=5 while the n=7 fleet got 0.0 — same bits, same resolution.
Either the 3x3x3 obstacle is harder than the hand-placed one, OR concurrency
(codec-server contention / scheduling) tips marginal legs into timeout. The load
test says the hardened server has ~13x throughput headroom and no tail, which
favors "obstacle", but doesn't prove n-independence.

This proves it: build the arenas ONCE, then run the SAME bit level on subsets of
size n in {1,4,8,15}. Same obstacle, same resolution, only n varies.
  * reach@b5 rises as n falls  -> concurrency IS the operational ceiling.
  * reach@b5 flat across n      -> it's real loss-intolerance + the obstacle.

Concurrency witness: only ARMED agents substitute packets, so the codec server's
max_inflight during a level == n. We reset /stats before each level and read it
after — an independent, fabrication-resistant confirmation that load actually
scaled with n.

Reuses the hardened fleet harness verbatim (setup_agent now re-fills on fail).

Usage:
    .venv/bin/python -m experiments.codec_loop.nsweep \
        --agents 0-14 --n-list 1,4,8,15 --bit 5 --trials 3 \
        --out results/sprintA/nsweep.json
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


def run_n_level(subset, codec_cfg_url, codec_url, relay, bits, args) -> dict:
    """One (n, bit-level) cell: reset codec stats, set the global quant, run all
    `subset` agents concurrently, snapshot the codec strain gauge, aggregate."""
    _codec_get("/stats/reset")
    _set_quant(codec_cfg_url, bits)
    time.sleep(0.3)
    label = f"n{len(subset)}-{'loss' if bits is None else f'b{bits}'}"
    t0 = time.time()
    per = _run_concurrent(
        lambda c, b=bits: run_level_for_agent(
            c["base"], relay, codec_url, c["player"], c["anchor"],
            args.delta, args.tol, args.leg_timeout, args.trials, b,
            args.fail_thresh),
        subset, label)
    per_agent = [per[c["n"]] for c in subset if "_error" not in per.get(c["n"], {})]
    agg = aggregate_level(bits, per_agent)
    agg["n"] = len(subset)
    agg["wall_s"] = round(time.time() - t0, 1)
    agg["errors"] = {n: per[n]["_error"] for n in per if "_error" in per[n]}
    health = _codec_get("/healthz")
    agg["codec_stats"] = health.get("stats")
    # drop the heavy per-leg payloads from the stored agg (keep it readable)
    agg.pop("per_agent", None)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A n-sweep: knee-vs-fleet-size")
    ap.add_argument("--agents", default="0-14")
    ap.add_argument("--n-list", default="1,4,8,15", help="fleet sizes to test (subsets of good arenas)")
    ap.add_argument("--bit", type=int, default=5, help="the (suspected-knee) bit level swept across n")
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
    ap.add_argument("--trials", type=int, default=3, help="MUST be uniform across n (reach resolution)")
    ap.add_argument("--barrier", choices=["wall", "none"], default="wall")
    ap.add_argument("--obstacle-size", type=int, default=3)
    ap.add_argument("--obstacle-height", type=int, default=3)
    ap.add_argument("--fail-thresh", type=float, default=2.5)
    ap.add_argument("--out", default="results/sprintA/nsweep.json")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    n_list = sorted({int(x) for x in args.n_list.split(",") if x.strip()})
    bx, by, bz = (int(v) for v in args.anchor_base.split(","))
    max_n = max(n_list)
    if len(agents) < max_n:
        print(f"[nsweep] FATAL: need >= {max_n} agents, have {len(agents)}: {agents}",
              file=sys.stderr)
        return 2

    ctx_all = []
    for i, n in enumerate(agents):
        ctx_all.append({
            "n": n, "player": f"agent{n}",
            "base": f"http://{args.host}:{args.port_base + n}",
            "anchor": (bx + args.spacing * i, by, bz),
        })

    print(f"[nsweep] agents={agents} n_list={n_list} bit={args.bit} trials={args.trials}")
    missing = [c["n"] for c in ctx_all if not _wait_in_world(c["base"], 8)]
    if missing:
        print(f"[nsweep] FATAL: not in world: {missing} (launch them first)", file=sys.stderr)
        return 2

    # --- setup ONCE, SEQUENTIALLY (no fill-storm), re-fill-on-fail in setup_agent
    _say(args.relay, f"[nsweep] building {len(ctx_all)} arenas sequentially ...")
    print("[nsweep] === SETUP (sequential, robust) ===")
    setup = {}
    good = []
    for c in ctx_all:
        s = setup_agent(c["base"], args.relay, c["player"], c["anchor"],
                        args.delta, args.radius, args.barrier,
                        args.obstacle_size, args.obstacle_height)
        setup[c["n"]] = s
        if s.get("ok"):
            good.append(c)
        print(f"[nsweep]   setup agent{c['n']}: ok={s.get('ok')} y_off={s.get('y_off')}")
    print(f"[nsweep] setup_ok = {len(good)}/{len(ctx_all)}")
    if len(good) < max_n:
        print(f"[nsweep] WARN: only {len(good)} good arenas; n-values > {len(good)} "
              f"will be clamped", file=sys.stderr)

    # --- n-sweep: subsets of the good arenas, lossless control + the swept bit --
    results = []
    for n in n_list:
        if len(good) < n:
            print(f"[nsweep] skip n={n} (only {len(good)} good)", file=sys.stderr)
            continue
        subset = good[:n]
        for bits in (None, args.bit):
            agg = run_n_level(subset, args.codec_cfg_url, args.codec_url,
                              args.relay, bits, args)
            results.append(agg)
            cs = agg.get("codec_stats") or {}
            label = "loss" if bits is None else f"b{bits}"
            _say(args.relay, f"[nsweep] n={n} {label}: reach={agg['reach_rate']} "
                             f"max_inflight={cs.get('max_inflight')}")
            print(f"[nsweep] n={n:>2} {label:>4}: reach={agg['reach_rate']} "
                  f"dist_mean={agg['dist_mean']} rb/ksub={agg['rb_per_ksub']} "
                  f"subst={agg['substituted']} max_inflight={cs.get('max_inflight')} "
                  f"roundtrips={cs.get('roundtrips')} ({agg['wall_s']}s)")

    _set_quant(args.codec_cfg_url, None)  # restore lossless
    _say(args.relay, "[nsweep] complete")

    out_doc = {
        "agents": agents, "n_list": n_list, "bit": args.bit,
        "trials": args.trials, "delta": args.delta, "leg_timeout": args.leg_timeout,
        "barrier": args.barrier, "obstacle_size": args.obstacle_size,
        "obstacle_height": args.obstacle_height, "fail_thresh": args.fail_thresh,
        "setup_ok": len(good), "setup": setup, "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[nsweep] wrote {args.out}")

    # compact table: the headline — reach@bit vs n, with the concurrency witness
    print("\n" + "=" * 70)
    print(f"N-SWEEP — reach vs fleet size at b={args.bit} (knee-mobility test)")
    print("=" * 70)
    print(f"{'n':>3} {'level':>5} {'reach':>6} {'dist_mn':>7} {'rb/ksub':>8} "
          f"{'subst':>7} {'max_infl':>8} {'wall_s':>7}")
    for r in results:
        cs = r.get("codec_stats") or {}
        lvl = "loss" if r["bits"] is None else f"b{r['bits']}"
        print(f"{r['n']:>3} {lvl:>5} {str(r['reach_rate']):>6} "
              f"{str(r['dist_mean']):>7} {str(r['rb_per_ksub']):>8} "
              f"{r['substituted']:>7} {str(cs.get('max_inflight')):>8} {r['wall_s']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
