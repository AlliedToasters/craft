#!/usr/bin/env python3
"""Sprint A — reconstruction-error structure of the move quantizer.

WHY: the live b-sweep showed reach 1.0 @ b6 -> 0.0 @ b5 -> 0.267 @ b4 (NON-
monotonic). My first explanation ("min step 0.516 @ b5 > 0.21 walk delta, so the
step rounds to ZERO") is FALSE: a 0.21 delta at range=8/b5 reconstructs to 0.258
(error ~0.05). Moving deltas survive. So what breaks b5?

This dissects the quantizer's reconstruction error over the REAL frozen delta
corpus, per bit level, separating two error modes a single RMSE hides:
  1. ZERO-BIAS: quant_scalar is mid-tread over [-R,+R] with (2**b - 1) levels;
     0.0 is representable iff (2**b-1) is even-coded at the midpoint — it is NOT,
     so a true-zero (or tiny) pos delta reconstructs to +/- (R/(2**b-1)). That DC
     offset is injected on EVERY near-stationary packet -> persistent jitter the
     server sees as the player twitching -> rubberband. This is range-linear, so
     it is what the POS_RANGE sweep moves.
  2. RESOLUTION error on genuinely-moving deltas (the textbook +/- half-step).

Reports, per bit level (at the live POS_RANGE=8): the reconstruction of a ZERO
delta (the DC bias), the |error| distribution split by |delta|<0.02 (stationary)
vs >=0.02 (moving), and a coarse error histogram. The b4-vs-b5 asymmetry should
surface in how the zero-bias and the moving-error interact across the grid.

Offline only; no MC. Uses the same frozen corpus + codec as offline_fidelity.py.

Usage:
    .venv/bin/python -m experiments.codec_loop.recon_hist            # narrated
    .venv/bin/python -m experiments.codec_loop.recon_hist --set combat
"""
from __future__ import annotations

import argparse
import json
import math

from experiments.codec_loop.offline_fidelity import SETS, load_move_actions
from experiments.codec_loop.quantize import POS_RANGE, quant_scalar

BIT_LEVELS = [8, 7, 6, 5, 4, 3]
STATIONARY = 0.02  # |delta| below this = effectively standing still


def zero_bias(bits: int, pos_range: float = POS_RANGE) -> float:
    """What a TRUE-ZERO pos delta reconstructs to at this bit width — the DC
    offset injected on every stationary packet."""
    rec, _ = quant_scalar(0.0, -pos_range, pos_range, bits)
    return rec


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = sorted(abs(v) for v in vals)
    n = len(a)
    return {
        "n": n,
        "rmse": round(math.sqrt(sum(v * v for v in vals) / n), 5),
        "mean_abs": round(sum(a) / n, 5),
        "p50_abs": round(a[n // 2], 5),
        "p99_abs": round(a[min(n - 1, int(0.99 * n))], 5),
        "max_abs": round(a[-1], 5),
    }


def analyze(root: str, pos_range: float = POS_RANGE) -> dict:
    actions, n_tp = load_move_actions(root)
    # collect the raw pos deltas once (action.pos is the (dx,dy,dz) delta)
    deltas = []  # flat list of per-axis deltas across all move-with-pos packets
    for action, _obs, orig in actions:
        if orig.get("has_pos") and action.pos is not None:
            deltas.extend(action.pos)
    rows = []
    for b in BIT_LEVELS:
        zb = zero_bias(b, pos_range)
        min_step = round(2.0 * pos_range / ((1 << b) - 1), 5)
        stat_err, move_err = [], []
        for d in deltas:
            rec, _ = quant_scalar(d, -pos_range, pos_range, b)
            err = rec - d
            if abs(d) < STATIONARY:
                stat_err.append(err)
            else:
                move_err.append(err)
        rows.append({
            "bits": b,
            "min_step": min_step,
            "zero_bias": round(zb, 5),
            "stationary": _stats(stat_err),
            "moving": _stats(move_err),
            "all": _stats(stat_err + move_err),
        })
    # delta population shape (context: how much of the traffic is near-zero)
    n_all = len(deltas)
    n_stat = sum(1 for d in deltas if abs(d) < STATIONARY)
    return {
        "n_move_packets": len(actions), "n_tp_excluded": n_tp,
        "n_axis_deltas": n_all, "pct_stationary": round(100.0 * n_stat / n_all, 1) if n_all else 0,
        "pos_range": pos_range, "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="narrated", choices=list(SETS))
    ap.add_argument("--pos-range", type=float, default=POS_RANGE)
    ap.add_argument("--out", default="/tmp/recon_hist.json")
    args = ap.parse_args()

    res = analyze(SETS[args.set], args.pos_range)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"set={args.set} range={res['pos_range']} "
          f"n_move={res['n_move_packets']} n_axis_deltas={res['n_axis_deltas']} "
          f"pct_stationary(|d|<{STATIONARY})={res['pct_stationary']}%")
    print(f"{'bits':>4} {'min_step':>8} {'zero_bias':>9} | "
          f"{'statRMSE':>8} {'statP99':>8} | {'moveRMSE':>8} {'moveP99':>8} | {'allRMSE':>8}")
    for r in res["rows"]:
        s, m, a = r["stationary"], r["moving"], r["all"]
        print(f"{r['bits']:>4} {r['min_step']:>8} {r['zero_bias']:>9} | "
              f"{str(s.get('rmse')):>8} {str(s.get('p99_abs')):>8} | "
              f"{str(m.get('rmse')):>8} {str(m.get('p99_abs')):>8} | {str(a.get('rmse')):>8}")
    print(f"\nwrote {args.out}")
    print("zero_bias = DC offset injected on a STATIONARY packet (the rubberband "
          "driver); range-linear, which is why the POS_RANGE sweep moves the knee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
