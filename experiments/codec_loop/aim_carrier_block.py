#!/usr/bin/env python3
"""§17.0 — AIM carrier, BLOCK channel: does the move-packet or use_item_on carry
the place target? And does corrupting block_pos on the wire break the place?

Companion to aim_carrier.py (the attack channel). The block-targeted action is a
clean DOUBLE control because use_item_on.block_pos is plain ints — the Java
reconstructor rebuilds it with NO entity lookup, so we can both (a) deadband the
move-packet rotation (§16 obsrel) and (b) corrupt block_pos itself on the wire,
and watch which one breaks aim.

Fixed aim task = /place_at one stone block at a known target T on flat ground.
`Placer.verifyPlacement` reads `level.getBlockState(T)` — so a place "success"
means the block ACTUALLY landed at T (world-truth, not just a sent packet). We
also scan T and T+delta directly to see WHERE a perturbed place lands.

Cells:
  control       lossless identity codec.
  obsrel@bN     move-packet yaw/pitch DEADBANDED on the wire (§16), use_item_on
                untouched, position near-lossless. Tests whether the place is
                gated on the wire ROTATION.
  perturb+Dx    use_item_on.block_pos offset by +D in x on the wire (rotation &
                position lossless). The CARRIER control: if the block lands at
                T+D instead of T, block_pos carries the place target.

VERDICT:
  obsrel benign  + perturb breaks  → block_pos (the action packet's own field)
                                      carries aim; move-rotation is render. The
                                      §16 "rotation near-free" result extends to
                                      block actions → 17.2 (lossy block_pos codec).
  obsrel breaks                     → rotation is load-bearing for placing → 17.1.

Run under PEACEFUL, on flat ground (the agent's desert spawn is flat stone).

Usage:
    .venv/bin/python -m experiments.codec_loop.aim_carrier_block --port 25570 \
        --bits 2,3 --delta 2 --trials 5 --out results/sprint17/aim_carrier_block.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import (
    _config, _relay, _rot_step_deg, _sidecar_base,
)
from experiments.codec_loop.run_rungs import _http, _pos, _resolve_base, _wait_in_world

ITEM = "minecraft:stone"


def _block_at(base: str, x: int, y: int, z: int) -> str:
    """Block id at one cell via a 1×1×1 scan; 'minecraft:air' if empty/absent."""
    r = _http("GET", f"{base}/scan_blocks?x1={x}&y1={y}&z1={z}&x2={x}&y2={y}&z2={z}",
              timeout=8)
    for b in r.get("blocks", []):
        if b.get("x") == x and b.get("y") == y and b.get("z") == z:
            return b.get("id", "minecraft:air")
    return "minecraft:air"


def _clear_strip(relay, tx, ty, tz, delta):
    """Air out the place target and the perturbed landing cells; ensure the
    supports below are solid stone so a place at either can take."""
    lo, hi = min(0, delta) - 1, max(0, delta) + 1
    for dx in range(lo, hi + 1):
        _relay(relay, f"setblock {tx+dx} {ty} {tz} air replace")
        _relay(relay, f"setblock {tx+dx} {ty-1} {tz} minecraft:stone replace")
    time.sleep(0.3)


def _run_cell(base, relay, codec_url, target, delta, trials, label, cfg):
    sidecar = _sidecar_base(codec_url)
    tx, ty, tz = target
    _config(sidecar, cfg)
    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)

    landed_at_T = 0
    landed_at_Tdelta = 0
    place_ok = 0
    per_trial = []
    for _ in range(trials):
        _clear_strip(relay, tx, ty, tz, delta)
        r = _http("POST", f"{base}/place_at",
                  {"item": ITEM, "x": tx, "y": ty, "z": tz}, timeout=12)
        # Placement is CLIENT-PREDICTED: the client shows the block at T
        # immediately regardless of the wire packet. Wait out the server
        # round-trip so the scan below reads server-corrected truth (a
        # perturbed/rotation-rejected place gets rolled back here).
        time.sleep(0.7)
        at_T = _block_at(base, tx, ty, tz)
        at_Td = _block_at(base, tx + delta, ty, tz)
        ok = bool(r.get("success"))
        hitT = at_T == ITEM
        hitTd = at_Td == ITEM
        if ok:
            place_ok += 1
        if hitT:
            landed_at_T += 1
        if hitTd:
            landed_at_Tdelta += 1
        per_trial.append({"place_success": ok, "reason": r.get("reason"),
                          "block_at_T": at_T, "block_at_T_plus_delta": at_Td})

    # cleanup
    _clear_strip(relay, tx, ty, tz, delta)
    for dx in (0, delta):
        _relay(relay, f"setblock {tx+dx} {ty} {tz} air replace")

    status = _http("GET", f"{base}/codec/passthrough/status", timeout=8)
    _http("POST", f"{base}/codec/passthrough/disarm")
    return {
        "cell": label, "config": cfg, "trials": trials,
        "place_success_rate": round(place_ok / trials, 3) if trials else 0.0,
        "landed_at_T_rate": round(landed_at_T / trials, 3) if trials else 0.0,
        "landed_at_T_plus_delta_rate": round(landed_at_Tdelta / trials, 3) if trials else 0.0,
        "substituted": status.get("substituted"),
        "substitute_errors": status.get("substitute_errors"),
        "drift": status.get("drift"),
        "per_trial": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§17.0 aim-carrier block-channel probe")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--bits", default="2,3", help="obs-rel rotation bit levels")
    ap.add_argument("--pos-bits", type=int, default=16)
    ap.add_argument("--delta", type=int, default=1, help="block_pos +x offset for the perturb cell")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--target-dx", type=int, default=3, help="target +x from player")
    ap.add_argument("--player", default=os.environ.get("MC_PLAYER_NAME", "agent0"),
                    help="player name for /give top-up")
    ap.add_argument("--out", default="results/sprint17/aim_carrier_block.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    codec_url = args.codec_url
    bit_levels = [int(b) for b in args.bits.split(",") if b.strip()]

    print(f"[aim_block] base={base} codec={codec_url} relay={args.relay}")
    if not _wait_in_world(base):
        print("[aim_block] FATAL: no player in world")
        return 2

    p0 = _pos(base)
    px, py, pz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    target = (px + args.target_dx, py, pz)  # py is feet level; support is py-1 (ground)
    print(f"[aim_block] player=({px},{py},{pz}) target={target} delta=+{args.delta}x "
          f"trials={args.trials}")
    # Top up stone so no cell runs dry (each place consumes one).
    _relay(args.relay, f"give {args.player} {ITEM} 64")
    time.sleep(0.3)

    cells = []

    print("\n[aim_block] === CONTROL (lossless identity) ===")
    agg = _run_cell(base, args.relay, codec_url, target, args.delta, args.trials,
                    "control", {"quant_bits": None, "obsrel": False, "perturb": None})
    print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
          f"@T+d={agg['landed_at_T_plus_delta_rate']} subst={agg['substituted']}")
    cells.append({"rot_bits": None, "rot_step_deg": 0.0, **agg})

    for b in bit_levels:
        step = _rot_step_deg(b)
        print(f"\n[aim_block] === obsrel rotation b{b} (step {step:.1f}°) ===")
        agg = _run_cell(base, args.relay, codec_url, target, args.delta, args.trials,
                        f"obsrel_b{b}",
                        {"obsrel": True, "pos_bits": args.pos_bits, "yaw_bits": b,
                         "pitch_bits": b, "pos_mode": "zero_preserving", "perturb": None})
        print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
              f"@T+d={agg['landed_at_T_plus_delta_rate']} subst={agg['substituted']}")
        cells.append({"rot_bits": b, "rot_step_deg": round(step, 2), **agg})

    print(f"\n[aim_block] === PERTURB block_pos +{args.delta}x (rot+pos lossless) ===")
    agg = _run_cell(base, args.relay, codec_url, target, args.delta, args.trials,
                    f"perturb_+{args.delta}x",
                    {"quant_bits": None, "obsrel": False,
                     "perturb": {"block_pos_delta": [args.delta, 0, 0]}})
    print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
          f"@T+d={agg['landed_at_T_plus_delta_rate']} subst={agg['substituted']}")
    cells.append({"rot_bits": None, "rot_step_deg": 0.0, **agg})

    _config(_sidecar_base(codec_url), {"quant_bits": None, "obsrel": False, "perturb": None})

    out = {"base": base, "codec_url": codec_url, "target": list(target),
           "delta": args.delta, "trials": args.trials, "cells": cells}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 66)
    print("§17.0 AIM CARRIER — block channel")
    print("=" * 66)
    print(f"{'cell':>14} {'place_ok':>9} {'@T':>6} {'@T+d':>6} {'subst':>6} {'subErr':>7}")
    for c in cells:
        print(f"{c['cell']:>14} {c['place_success_rate']:>9.2f} {c['landed_at_T_rate']:>6.2f} "
              f"{c['landed_at_T_plus_delta_rate']:>6.2f} {str(c.get('substituted')):>6} "
              f"{str(c.get('substitute_errors')):>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
