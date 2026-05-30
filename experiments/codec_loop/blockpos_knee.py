#!/usr/bin/env python3
"""§17.2.1 — live block_pos KNEE: how few bits address the place target on the
wire, and is the obs-relative reparam the entire win?

Confirmatory sweep (the spec says so). §17.0 already proved the carrier: a +1
perturbation of block_pos is a deterministic miss. This converts that into a
bits-vs-parity knee for the real codec (craft.codec.server blockpos mode), and
contrasts the two codings the §17.2 plan named:

  control            block_pos lossless identity.
  obsrel@bN          block_pos coded as round(block_pos - player) over ±reach,
                     N bits/axis. Predicted: lossless to ~b4 (step<1 over ±6),
                     cliff below — the §16 "every field a zero_preserving delta
                     vs obs" reparam applied to the discrete target.
  absolute@bN        block_pos quantized over a fixed ±abs_range world window.
                     Predicted: misses until ~13-15 bits/axis (it must resolve
                     the world-coordinate magnitude) — the foil that shows the
                     obs-relative reparam, not lossy tolerance, is the win.

Metric = place-@T rate via post-server-sync scan (placement is client-predicted;
we wait out the round-trip then scan world truth), reused from aim_carrier_block.

VERDICT (pre-registered): obsrel holds parity to a ~4-bit floor then cliffs;
absolute needs ~3x the bits for the same parity. No graded sub-unit tolerance.
=> the only block_pos compression is the lossless pointer reparam => a learned
codec's only remaining play is dropping the pointer and reconstructing from the
block grid in obs (§18, which needs that obs channel). Mining (player_action dig)
carries the identical block_pos field -> covered by symmetry, no separate driver.

Run under PEACEFUL, on flat ground.

Usage:
    .venv/bin/python -m experiments.codec_loop.blockpos_knee --port 25570 \
        --obsrel-bits 6,5,4,3,2 --abs-bits 14,8 --trials 5 \
        --out results/sprint17/blockpos_knee.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.codec_loop.aim_carrier import _config, _relay, _sidecar_base
from experiments.codec_loop.aim_carrier_block import _block_at, _clear_strip, ITEM
from experiments.codec_loop.run_rungs import _http, _pos, _resolve_base, _wait_in_world

# A clean reset POSTed before every cell so no prior lossy mode leaks across cells.
_LOSSLESS = {"quant_bits": None, "obsrel": False, "perturb": None, "blockpos": None}


def _run_cell(base, relay, codec_url, target, trials, label, blockpos_cfg):
    sidecar = _sidecar_base(codec_url)
    tx, ty, tz = target
    _config(sidecar, {**_LOSSLESS, "blockpos": blockpos_cfg})
    _http("POST", f"{base}/codec/passthrough/arm",
          {"endpoint": codec_url, "substitute": True})
    time.sleep(0.3)

    landed_at_T = 0
    place_ok = 0
    per_trial = []
    for _ in range(trials):
        _clear_strip(relay, tx, ty, tz, 0)
        r = _http("POST", f"{base}/place_at",
                  {"item": ITEM, "x": tx, "y": ty, "z": tz}, timeout=12)
        # Placement is client-predicted; wait out the server round-trip so the
        # scan reads world-corrected truth (a mis-targeted place rolls back).
        time.sleep(0.7)
        at_T = _block_at(base, tx, ty, tz)
        ok = bool(r.get("success"))
        hitT = at_T == ITEM
        if ok:
            place_ok += 1
        if hitT:
            landed_at_T += 1
        per_trial.append({"place_success": ok, "reason": r.get("reason"),
                          "block_at_T": at_T})

    _clear_strip(relay, tx, ty, tz, 0)
    status = _http("GET", f"{base}/codec/passthrough/status", timeout=8)
    _http("POST", f"{base}/codec/passthrough/disarm")
    return {
        "cell": label, "blockpos": blockpos_cfg, "trials": trials,
        "place_success_rate": round(place_ok / trials, 3) if trials else 0.0,
        "landed_at_T_rate": round(landed_at_T / trials, 3) if trials else 0.0,
        "substituted": status.get("substituted"),
        "substitute_errors": status.get("substitute_errors"),
        "drift": status.get("drift"),
        "per_trial": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§17.2.1 block_pos live knee")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--codec-url", default="http://127.0.0.1:25600/codec/roundtrip")
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--obsrel-bits", default="6,5,4,3,2")
    ap.add_argument("--abs-bits", default="14,8", help="absolute-coding bit levels (the foil)")
    ap.add_argument("--reach", type=float, default=6.0)
    ap.add_argument("--abs-range", type=float, default=8192.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--target-dx", type=int, default=3, help="target +x from player")
    ap.add_argument("--player", default=os.environ.get("MC_PLAYER_NAME", "agent0"))
    ap.add_argument("--out", default="results/sprint17/blockpos_knee.json")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    codec_url = args.codec_url
    obsrel_bits = [int(b) for b in args.obsrel_bits.split(",") if b.strip()]
    abs_bits = [int(b) for b in args.abs_bits.split(",") if b.strip()]

    print(f"[bp_knee] base={base} codec={codec_url} relay={args.relay}")
    if not _wait_in_world(base):
        print("[bp_knee] FATAL: no player in world")
        return 2

    p0 = _pos(base)
    px, py, pz = int(p0["x"]), int(p0["y"]), int(p0["z"])
    target = (px + args.target_dx, py, pz)
    print(f"[bp_knee] player=({px},{py},{pz}) target={target} dx=+{args.target_dx} "
          f"trials={args.trials} reach={args.reach} abs_range={args.abs_range}")
    _relay(args.relay, f"give {args.player} {ITEM} 64")
    time.sleep(0.3)

    cells = []

    print("\n[bp_knee] === CONTROL (lossless identity) ===")
    agg = _run_cell(base, args.relay, codec_url, target, args.trials, "control", None)
    print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
          f"subst={agg['substituted']}")
    cells.append({"mode": "control", "bits": None, **agg})

    for b in obsrel_bits:
        print(f"\n[bp_knee] === obsrel b{b} (reach ±{args.reach}) ===")
        agg = _run_cell(base, args.relay, codec_url, target, args.trials, f"obsrel_b{b}",
                        {"bits": b, "mode": "obsrel", "reach": args.reach})
        print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
              f"subst={agg['substituted']}")
        cells.append({"mode": "obsrel", "bits": b, **agg})

    for b in abs_bits:
        print(f"\n[bp_knee] === absolute b{b} (±{args.abs_range}) ===")
        agg = _run_cell(base, args.relay, codec_url, target, args.trials, f"absolute_b{b}",
                        {"bits": b, "mode": "absolute", "abs_range": args.abs_range})
        print(f"  place_ok={agg['place_success_rate']} @T={agg['landed_at_T_rate']} "
              f"subst={agg['substituted']}")
        cells.append({"mode": "absolute", "bits": b, **agg})

    _config(_sidecar_base(codec_url), dict(_LOSSLESS))

    out = {"base": base, "codec_url": codec_url, "target": list(target),
           "reach": args.reach, "abs_range": args.abs_range, "trials": args.trials,
           "cells": cells}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("§17.2.1 block_pos KNEE")
    print("=" * 60)
    print(f"{'cell':>14} {'place_ok':>9} {'@T':>6} {'subst':>6} {'subErr':>7}")
    for c in cells:
        print(f"{c['cell']:>14} {c['place_success_rate']:>9.2f} {c['landed_at_T_rate']:>6.2f} "
              f"{str(c.get('substituted')):>6} {str(c.get('substitute_errors')):>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
