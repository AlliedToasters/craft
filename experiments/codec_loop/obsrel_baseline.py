#!/usr/bin/env python3
"""§16.1 — the obs-relative-rotation BASELINE: rate-distortion + at-rest gate.

The honest baseline the §16.2 learned AE must beat is NOT the §15 quantizer's
18.45-bit allocation — it is obs-relative-rotation coding + entropy coding (§16.0
verdict, ≈1.8 b/pkt). This script measures that baseline OFFLINE over the frozen
corpus, on three axes, comparing obs-RELATIVE rotation coding vs the §15 ABSOLUTE
coding at matched bits:

  1. FIDELITY  — reconstruction error (yaw/pitch RMSE°, wrapped) at each bit level.
  2. RATE      — achievable entropy-coded bits/symbol (= empirical code entropy).
                 The obs-relative residual is peaked at 0, so its entropy is far
                 below the (near-uniform) absolute code's entropy at matched bits.
  3. ZERO-MEAN-AT-REST — reconstruction deviation on NOT-turning packets. obs-rel
                 zero_preserving holds heading EXACTLY (residual 0 -> obs.{yaw,
                 pitch}); absolute coding snaps to its grid -> a static ±half-step
                 heading offset (the camera analog of the §15 pos zero-bias).

The deliverable is the rate-distortion frontier (rate vs RMSE) for obs-rel vs
absolute: obs-rel should DOMINATE (lower rate at equal fidelity), and that frontier
is the line the learned codec must beat. NO training; deterministic reparam only.

Usage:
    .venv/bin/python -m experiments.codec_loop.obsrel_baseline            # narrated
    .venv/bin/python -m experiments.codec_loop.obsrel_baseline --set combat
    .venv/bin/python -m experiments.codec_loop.obsrel_baseline --out results/sprint16/obsrel_baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

from experiments.codec_loop.offline_fidelity import SETS, load_move_actions
from experiments.codec_loop.obsrel import wrap180, ROT_RANGE
from experiments.codec_loop.quantize import (
    YAW_LO, YAW_HI, PITCH_LO, PITCH_HI,
    quant_scalar, quant_scalar_zero_preserving, _wrap360,
)

BIT_LEVELS = [10, 8, 6, 5, 4, 3, 2]
STAT_ROT = 0.5     # |per-tick turn| below this (deg) = player is holding heading


def _entropy_bits(codes) -> float:
    c = Counter(codes)
    n = sum(c.values())
    if not n:
        return 0.0
    return -sum((k / n) * math.log2(k / n) for k in c.values() if k)


def _rms(vals) -> float:
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def _p99(vals) -> float:
    if not vals:
        return float("nan")
    s = sorted(abs(v) for v in vals)
    return s[min(len(s) - 1, int(0.99 * len(s)))]


def analyze(root: str) -> dict:
    actions, n_tp = load_move_actions(root)
    # collect raw (yaw, obs_yaw, pitch, obs_pitch) for every rot-bearing packet
    rot_rows = []
    for action, obs, _orig in actions:
        if action.rot is None:
            continue
        yaw, pitch = action.rot
        rot_rows.append((float(yaw), float(obs["yaw"]), float(pitch), float(obs["pitch"])))

    n_rot = len(rot_rows)
    # at-rest mask: holding heading (both axes barely moving)
    restset = {i for i, (y, oy, p, op) in enumerate(rot_rows)
               if abs(wrap180(y - oy)) < STAT_ROT and abs(p - op) < STAT_ROT}
    pct_rest = round(100.0 * len(restset) / n_rot, 1) if n_rot else 0.0

    rows = []
    for b in BIT_LEVELS:
        # --- ABSOLUTE coding (the §15 way) ---
        abs_yaw_err, abs_pitch_err = [], []
        abs_yaw_codes, abs_pitch_codes = [], []
        abs_rest_err = []
        # --- obs-RELATIVE coding (this baseline) ---
        rel_yaw_err, rel_pitch_err = [], []
        rel_yaw_codes, rel_pitch_codes = [], []
        rel_rest_err = []
        for i, (yaw, oy, pitch, op) in enumerate(rot_rows):
            # absolute
            yrec, yc = quant_scalar(_wrap360(yaw), YAW_LO, YAW_HI, b)
            prec, pc = quant_scalar(pitch, PITCH_LO, PITCH_HI, b)
            ya_err = wrap180(yrec - _wrap360(yaw))
            pa_err = prec - pitch
            abs_yaw_err.append(ya_err); abs_pitch_err.append(pa_err)
            abs_yaw_codes.append(yc); abs_pitch_codes.append(pc)
            # obs-relative
            yres_rec, yrc = quant_scalar_zero_preserving(wrap180(yaw - oy), -ROT_RANGE, ROT_RANGE, b)
            pres_rec, prc = quant_scalar_zero_preserving(pitch - op, -ROT_RANGE, ROT_RANGE, b)
            yr_err = wrap180((oy + yres_rec) - yaw)
            pr_err = (op + pres_rec) - pitch
            rel_yaw_err.append(yr_err); rel_pitch_err.append(pr_err)
            rel_yaw_codes.append(yrc); rel_pitch_codes.append(prc)
            if i in restset:
                abs_rest_err.append(ya_err); abs_rest_err.append(pa_err)
                rel_rest_err.append(yr_err); rel_rest_err.append(pr_err)
        rows.append({
            "bits": b,
            "absolute": {
                "yaw_rmse": round(_rms(abs_yaw_err), 4), "yaw_p99": round(_p99(abs_yaw_err), 4),
                "pitch_rmse": round(_rms(abs_pitch_err), 4), "pitch_p99": round(_p99(abs_pitch_err), 4),
                "yaw_entropy_bits": round(_entropy_bits(abs_yaw_codes), 4),
                "pitch_entropy_bits": round(_entropy_bits(abs_pitch_codes), 4),
                "rot_rate_bits": round(_entropy_bits(abs_yaw_codes) + _entropy_bits(abs_pitch_codes), 4),
                "at_rest_rmse": round(_rms(abs_rest_err), 4),
            },
            "obs_relative": {
                "yaw_rmse": round(_rms(rel_yaw_err), 4), "yaw_p99": round(_p99(rel_yaw_err), 4),
                "pitch_rmse": round(_rms(rel_pitch_err), 4), "pitch_p99": round(_p99(rel_pitch_err), 4),
                "yaw_entropy_bits": round(_entropy_bits(rel_yaw_codes), 4),
                "pitch_entropy_bits": round(_entropy_bits(rel_pitch_codes), 4),
                "rot_rate_bits": round(_entropy_bits(rel_yaw_codes) + _entropy_bits(rel_pitch_codes), 4),
                "at_rest_rmse": round(_rms(rel_rest_err), 4),
            },
        })
    return {
        "set": root, "n_move_packets": len(actions), "n_rot_packets": n_rot,
        "n_tp_excluded": n_tp, "pct_holding_heading": pct_rest,
        "stat_rot_deg": STAT_ROT, "rot_range": ROT_RANGE, "rows": rows,
    }


def _print(res: dict) -> None:
    print(f"\n===== §16.1 obs-relative baseline: {res['set']} "
          f"(n_rot={res['n_rot_packets']}, {res['pct_holding_heading']}% holding heading) =====")
    print("rate = entropy-coded bits/packet for (yaw+pitch); fidelity = RMSE°; "
          "at_rest = RMSE° while holding heading")
    print(f"\n{'bits':>4} | {'ABS rate':>8} {'ABS yawRMSE':>11} {'ABS rest':>8} | "
          f"{'REL rate':>8} {'REL yawRMSE':>11} {'REL rest':>8}")
    for r in res["rows"]:
        a, rl = r["absolute"], r["obs_relative"]
        print(f"{r['bits']:>4} | {a['rot_rate_bits']:>8.3f} {a['yaw_rmse']:>11.3f} "
              f"{a['at_rest_rmse']:>8.3f} | {rl['rot_rate_bits']:>8.3f} "
              f"{rl['yaw_rmse']:>11.3f} {rl['at_rest_rmse']:>8.3f}")
    print("\nobs-relative should give LOWER rate at EQUAL fidelity (RD frontier "
          "dominance) and at_rest≈0 (holds heading exactly; absolute snaps to grid).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="narrated", choices=list(SETS))
    ap.add_argument("--out", default="/tmp/obsrel_baseline.json")
    args = ap.parse_args()
    res = analyze(SETS[args.set])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    _print(res)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
