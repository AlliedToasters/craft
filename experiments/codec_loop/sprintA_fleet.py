#!/usr/bin/env python3
"""Sprint A — CONCURRENT parity-vs-bits sweep across the agent fleet.

Scales the single-agent `sprintA_arena.py` to N agents sharing ONE MC server, so
the bit-sweep finishes in ~1/N the wall-clock (user: "trials are taking too
long"). The experiment is unchanged; only the parallelism + arena-construction
are new:

  * Each agent gets its OWN arena at a far, non-overlapping anchor
    (anchor_i = base + i*spacing on X; 512-block spacing >> sim distance), so 15
    players on one server never share chunks and never interfere.
  * The obstacle ("barrier") is BUILT PROGRAMMATICALLY in every arena (user:
    "i won't build 15 manually"). `--barrier wall` stamps the proven avoidable
    obstacle (a stone box on the goto diagonal) that reproduces the graded
    knee (b=5-6) from results/sprintA/RESULTS.md; `--barrier none` = flat.
  * The codec server (:25600) holds ONE global quant level under a lock, so all
    agents MUST be at the same bit level at the same instant. The sweep is
    therefore BARRIER-SYNCHRONIZED by bit-level: set b once globally, release all
    agents to run their trials concurrently, join, then advance b. The wire path
    is byte-identical across levels (only the sidecar math changes).
  * Rubberband telemetry (new this sprint): per (agent, bit-level) we snapshot
    /packets/feedback before/after the level, so corrections-per-movement-packet
    becomes a parity metric alongside dist/reach (colleague: the categorical
    server-side rubberband-rate may have a cleaner knee than continuous drift).

Reuses sprintA_arena's HTTP + trial helpers verbatim (no fork of the validated
single-agent path).

Usage:
    .venv/bin/python -m experiments.codec_loop.sprintA_fleet \
        --agents 0-14 --bits 8,6,5,4,3 --trials 4 --barrier wall \
        --out results/sprintA/fleet.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from experiments.codec_loop.sprintA_arena import (
    _arm_codec,
    _codec_status,
    _disarm_codec,
    _http,
    _pos,
    _say,
    _set_allow_break,
    _set_quant,
    _wait_in_world,
    build_flat_arena,
    run_trial,
    tp_to,
)


# ---- agent spec --------------------------------------------------------------

def parse_agents(spec: str) -> list[int]:
    """'0-14' or '0,1,2' or '0-4,7,9' -> sorted unique list of agent numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _feedback(base: str) -> dict:
    """Snapshot the inbound corrective-feedback counters (cumulative since the
    agent's homunculus booted). Delta across a level = corrections that level."""
    r = _http("GET", f"{base}/packets/feedback", timeout=6)
    return {
        "rubberbands": r.get("rubberbands", 0) or 0,
        "motion_overrides": r.get("motion_overrides", 0) or 0,
        "total": r.get("total", 0) or 0,
    }


# ---- arena construction (programmatic obstacle) ------------------------------

def build_obstacle(relay: str, anchor: tuple[int, int, int], delta: int,
                   size: int, height: int) -> dict:
    """Stamp an AVOIDABLE stone box at the midpoint of the goto diagonal.

    Footprint `size`x`size` (odd -> centered), `height` tall, sitting ON the
    floor. With allowBreak OFF the agent can neither dig through nor (without
    allow_place) climb a >=2 box, so Baritone routes AROUND it on a curved path
    — exactly the constrained-but-recoverable course that produces the graded
    knee. Small relative to the radius-12 platform, so there is room to go
    around either end.
    """
    ax, ay, az = anchor
    mx = ax + delta // 2
    mz = az + delta // 2
    half = max(0, size // 2)
    return _server_fill(relay, mx - half, ay, mz - half,
                        mx + half, ay + height - 1, mz + half, "minecraft:stone")


def _server_fill(relay: str, x0, y0, z0, x1, y1, z1, block: str) -> dict:
    return _http("POST", f"{relay}/cmd",
                 {"cmd": f"fill {x0} {y0} {z0} {x1} {y1} {z1} {block}"}, timeout=10)


def setup_agent(base: str, relay: str, player: str, anchor: tuple[int, int, int],
                delta: int, radius: int, barrier: str, size: int,
                height: int) -> dict:
    """Place the agent on its anchor, build its arena + obstacle, lock allowBreak
    OFF. Returns a small status dict for the setup log."""
    ax, ay, az = anchor
    start = (ax + 0.5, ay, az + 0.5)
    # 1. force chunk load at the (possibly never-visited) far anchor.
    tp_to(relay, player, *start)
    if not _wait_in_world(base, 90):
        return {"player": player, "ok": False, "reason": "not_in_world"}
    # CRITICAL: let fresh/far chunks finish generating+loading BEFORE filling.
    # Without this the /fill races chunk-load, silently no-ops, and the agent is
    # left on natural terrain BELOW the (unbuilt) platform -> Baritone reports
    # 'unreachable' to the y=anchor target. This was the root cause of the v1
    # fleet failure: 14/15 distant arenas had no floor (only agent0's pre-loaded
    # chunks worked), so reach collapsed to ~0.14 even at lossless.
    time.sleep(12)
    # 2. build the platform + obstacle (chunks now loaded -> fill applies).
    build_flat_arena(relay, player, anchor, radius)
    obs = None
    if barrier == "wall":
        obs = build_obstacle(relay, anchor, delta, size, height)
    time.sleep(0.5)
    ab = _set_allow_break(base, False)
    # 3. place the agent ON the platform and VERIFY it is at floor level, with
    # retries. An agent stuck below the platform would only feed 'unreachable',
    # so we mark it not-ok and exclude it from the sweep rather than poison data.
    y_off = None
    ok = False
    for _ in range(6):
        tp_to(relay, player, *start)
        time.sleep(1.2)
        p = _pos(base)
        y = p.get("y")
        if y is not None:
            y_off = round(y - ay, 2)
            if abs(y - ay) <= 1.5:
                ok = True
                break
        # Verification failed: the floor almost certainly didn't fill (the
        # chunk-load race) so the player fell to natural terrain. RE-FILL — the
        # player is now standing here forcing the chunks resident, so the second
        # fill lands. Re-filling on every failed probe (not just sleeping longer)
        # is what turns the 7/14 concurrent result into a robust build.
        time.sleep(2.0)
        build_flat_arena(relay, player, anchor, radius)
        if barrier == "wall":
            build_obstacle(relay, anchor, delta, size, height)
        time.sleep(0.8)
    p = _pos(base)
    return {
        "player": player, "ok": ok, "anchor": list(anchor),
        "pos": [p.get("x"), p.get("y"), p.get("z")], "y_off": y_off,
        "allow_break": ab.get("value", ab), "obstacle": obs is not None,
    }


# ---- one agent, one bit level ------------------------------------------------

def run_level_for_agent(base: str, relay: str, codec_url: str, player: str,
                        anchor: tuple[int, int, int], delta: int, tol: int,
                        leg_timeout: int, trials: int, bits, fail_thresh: float) -> dict:
    """Arm the codec (global quant already set by the orchestrator), snapshot the
    rubberband counter, run N out-and-back trials, snapshot again, disarm.

    `fail_thresh` is the per-leg distance (blocks) above which a leg counts as a
    failure (matches the obstacle-run convention in RESULTS.md: fail = dist>2.5
    or no-arrival)."""
    start = (anchor[0], anchor[1], anchor[2])
    out_xyz = (anchor[0] + delta, anchor[1], anchor[2] + delta)

    fb0 = _feedback(base)
    _arm_codec(base, codec_url)
    legs = []
    for _ in range(trials):
        legs.append(run_trial(base, relay, player, start, out_xyz, tol, leg_timeout))
    status = _codec_status(base)
    final = _disarm_codec(base)
    fb1 = _feedback(base)

    out_dists = [t["out"]["dist"] for t in legs if t["out"]["dist"] is not None]
    home_dists = [t["home"]["dist"] for t in legs if t["home"]["dist"] is not None]

    def _failed(leg):
        d = leg["dist"]
        return d is None or d > fail_thresh

    out_fail = sum(_failed(t["out"]) for t in legs)
    home_fail = sum(_failed(t["home"]) for t in legs)
    n_legs = 2 * len(legs)
    n_fail = out_fail + home_fail

    substituted = final.get("substituted") or 0
    rb_delta = fb1["rubberbands"] - fb0["rubberbands"]
    motion_delta = fb1["motion_overrides"] - fb0["motion_overrides"]

    return {
        "player": player, "bits": bits, "trials": len(legs),
        "out_fail": out_fail, "home_fail": home_fail,
        "n_legs": n_legs, "n_fail": n_fail,
        "reach_rate": round((n_legs - n_fail) / n_legs, 3) if n_legs else None,
        "out_dists": out_dists, "home_dists": home_dists,
        "substituted": substituted,
        "drift": final.get("drift"),
        "rubberbands": rb_delta, "motion_overrides": motion_delta,
        "rb_per_leg": round(rb_delta / n_legs, 3) if n_legs else None,
        "rb_per_ksub": round(1000.0 * rb_delta / substituted, 3) if substituted else None,
        "subst_latency_p99_ms": status.get("subst_latency_p99_ms"),
        "legs": legs,
    }


# ---- aggregation -------------------------------------------------------------

def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def _p(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def aggregate_level(bits, per_agent: list[dict]) -> dict:
    """Pool all agents' legs at one bit level into the headline row."""
    all_out = [d for a in per_agent for d in a["out_dists"]]
    all_home = [d for a in per_agent for d in a["home_dists"]]
    all_dists = all_out + all_home
    n_legs = sum(a["n_legs"] for a in per_agent)
    n_fail = sum(a["n_fail"] for a in per_agent)
    rb = sum(a["rubberbands"] for a in per_agent)
    motion = sum(a["motion_overrides"] for a in per_agent)
    subst = sum(a["substituted"] for a in per_agent)
    drift = sum((a["drift"] or 0) for a in per_agent)
    return {
        "bits": bits,
        "n_agents": len(per_agent),
        "n_legs": n_legs,
        "n_fail": n_fail,
        "reach_rate": round((n_legs - n_fail) / n_legs, 3) if n_legs else None,
        "out_fail": sum(a["out_fail"] for a in per_agent),
        "home_fail": sum(a["home_fail"] for a in per_agent),
        "dist_mean": _mean(all_dists),
        "out_mean": _mean(all_out),
        "home_mean": _mean(all_home),
        "dist_p90": _p(all_dists, 0.9),
        "dist_max": round(max(all_dists), 3) if all_dists else None,
        "substituted": subst,
        "drift": drift,
        "rubberbands": rb,
        "motion_overrides": motion,
        "rb_per_leg": round(rb / n_legs, 3) if n_legs else None,
        "rb_per_ksub": round(1000.0 * rb / subst, 3) if subst else None,
        "per_agent": per_agent,
    }


# ---- orchestration -----------------------------------------------------------

def _run_concurrent(fn, agents_ctx: list[dict], label: str) -> dict[int, dict]:
    """Run fn(ctx) for every agent in its own thread; join all (barrier).
    Returns {agent_n: result}. Exceptions become {'_error': str} so one bad
    agent never kills the level."""
    results: dict[int, dict] = {}
    lock = threading.Lock()

    def _worker(ctx):
        n = ctx["n"]
        try:
            r = fn(ctx)
        except Exception as e:  # noqa: BLE001 - isolate one agent's failure
            r = {"_error": f"{type(e).__name__}: {e}", "player": ctx["player"]}
        with lock:
            results[n] = r

    threads = [threading.Thread(target=_worker, args=(c,), name=f"{label}-a{c['n']}")
               for c in agents_ctx]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint A concurrent fleet parity sweep")
    ap.add_argument("--agents", default="0-14", help="agent numbers: '0-14' or '0,1,2'")
    ap.add_argument("--host", default=os.environ.get("HOMUNCULUS_HOST", "127.0.0.1"))
    ap.add_argument("--port-base", type=int, default=25570, help="homunculus port = base + agent_n")
    ap.add_argument("--relay", default=os.environ.get("MC_SERVER_CMD_BASE", "http://127.0.0.1:4747"))
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--codec-cfg-url", default="http://127.0.0.1:25600/config")
    ap.add_argument("--anchor-base", default="6000,100,6000", help="arena 0 center x,y,z")
    ap.add_argument("--spacing", type=int, default=512, help="X-gap between adjacent arenas (blocks)")
    ap.add_argument("--radius", type=int, default=12, help="arena half-width")
    ap.add_argument("--delta", type=int, default=8, help="out-and-back hop length")
    ap.add_argument("--tol", type=int, default=1, help="arrival tolerance (blocks)")
    ap.add_argument("--leg-timeout", type=int, default=25)
    ap.add_argument("--trials", type=int, default=4, help="trials/agent/level (total/level = trials*agents)")
    ap.add_argument("--bits", default="8,6,5,4,3", help="comma list of bit levels (lossless control always run)")
    ap.add_argument("--barrier", choices=["wall", "none"], default="wall")
    ap.add_argument("--obstacle-size", type=int, default=3, help="obstacle footprint (odd -> centered)")
    ap.add_argument("--obstacle-height", type=int, default=3)
    ap.add_argument("--fail-thresh", type=float, default=2.5, help="per-leg dist (blocks) above which a leg is a fail")
    ap.add_argument("--out", default="results/sprintA/fleet.json")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    bx, by, bz = (int(v) for v in args.anchor_base.split(","))
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]

    # Build per-agent context (port, player, anchor).
    ctx = []
    for i, n in enumerate(agents):
        anchor = (bx + args.spacing * i, by, bz)
        ctx.append({
            "n": n,
            "player": f"agent{n}",
            "base": f"http://{args.host}:{args.port_base + n}",
            "anchor": anchor,
        })

    print(f"[fleet] {len(agents)} agents: {agents}")
    for c in ctx:
        print(f"[fleet]   agent{c['n']:>2} -> {c['base']}  anchor={c['anchor']}")
    print(f"[fleet] bits={bit_levels} trials/agent/level={args.trials} "
          f"total/level={args.trials * len(agents)} barrier={args.barrier}")

    # Preflight: every agent must be reachable + in-world.
    missing = [c["n"] for c in ctx if not _wait_in_world(c["base"], 8)]
    if missing:
        print(f"[fleet] FATAL: agents not in world: {missing} "
              f"(launch them first: ./launch_agent.sh N)", file=sys.stderr)
        return 2

    # --- setup phase (concurrent): build each arena + obstacle ---------------
    _say(args.relay, f"[Sprint A fleet] building {len(agents)} arenas ({args.barrier}) ...")
    print("\n[fleet] === SETUP: building arenas concurrently ===")
    setup = _run_concurrent(
        lambda c: setup_agent(c["base"], args.relay, c["player"], c["anchor"],
                              args.delta, args.radius, args.barrier,
                              args.obstacle_size, args.obstacle_height),
        ctx, "setup")
    for n in sorted(setup):
        print(f"[fleet]   setup agent{n}: {setup[n]}")
    bad = [n for n in setup if not setup[n].get("ok")]
    if bad:
        print(f"[fleet] WARN: setup failed for agents {bad}; continuing with the rest",
              file=sys.stderr)
        ctx = [c for c in ctx if c["n"] not in bad]
        if not ctx:
            print("[fleet] FATAL: no agents set up", file=sys.stderr)
            return 3

    # --- sweep phase: barrier-synchronized by bit level ----------------------
    levels = [None] + bit_levels  # control first
    results = []
    for bits in levels:
        label = "lossless" if bits is None else f"b={bits}"
        _set_quant(args.codec_cfg_url, bits)   # ONE global level for all agents
        time.sleep(0.3)
        _say(args.relay, f"[Sprint A fleet] === {label} : {len(ctx)} agents x {args.trials} trials ===")
        print(f"\n[fleet] === {label} (global quant set; {len(ctx)} agents concurrently) ===")
        t0 = time.time()
        per = _run_concurrent(
            lambda c, b=bits: run_level_for_agent(
                c["base"], args.relay, args.codec_url, c["player"], c["anchor"],
                args.delta, args.tol, args.leg_timeout, args.trials, b,
                args.fail_thresh),
            ctx, label)
        per_agent = [per[c["n"]] for c in ctx if "_error" not in per.get(c["n"], {})]
        errs = {n: per[n]["_error"] for n in per if "_error" in per[n]}
        if errs:
            print(f"[fleet]   {label} agent errors: {errs}", file=sys.stderr)
        agg = aggregate_level(bits, per_agent)
        agg["wall_s"] = round(time.time() - t0, 1)
        agg["errors"] = errs
        results.append(agg)
        print(f"[fleet]   {label}: reach={agg['reach_rate']} dist_mean={agg['dist_mean']} "
              f"rb={agg['rubberbands']} rb/leg={agg['rb_per_leg']} "
              f"rb/ksub={agg['rb_per_ksub']} subst={agg['substituted']} "
              f"drift={agg['drift']} ({agg['wall_s']}s)")

    # restore lossless so nothing stays silently lossy
    _set_quant(args.codec_cfg_url, None)
    _say(args.relay, "[Sprint A fleet] sweep complete")

    out_doc = {
        "agents": agents, "anchor_base": [bx, by, bz], "spacing": args.spacing,
        "delta": args.delta, "tol": args.tol, "trials_per_agent": args.trials,
        "barrier": args.barrier, "obstacle_size": args.obstacle_size,
        "obstacle_height": args.obstacle_height, "fail_thresh": args.fail_thresh,
        "bits": bit_levels, "setup": setup, "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\n[fleet] wrote {args.out}")

    # compact table — both curves side by side
    print("\n" + "=" * 78)
    print("SPRINT A FLEET — parity vs bits  (continuous dist + categorical rubberband)")
    print("=" * 78)
    print(f"{'bits':>8} {'legs':>5} {'reach':>6} {'dist_mn':>7} {'home_mn':>7} "
          f"{'rb':>5} {'rb/leg':>6} {'rb/ksub':>7} {'subst':>7} {'drift':>5}")
    for r in results:
        print(f"{str(r['bits']):>8} {r['n_legs']:>5} {str(r['reach_rate']):>6} "
              f"{str(r['dist_mean']):>7} {str(r['home_mean']):>7} "
              f"{r['rubberbands']:>5} {str(r['rb_per_leg']):>6} "
              f"{str(r['rb_per_ksub']):>7} {r['substituted']:>7} {r['drift']:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
