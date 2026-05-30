#!/usr/bin/env python3
"""§16.1 — AE headroom preflight: what can a learned codec win over the baseline?

The obs-relative baseline (obsrel_baseline.py) is already a per-field, LAG-1
conditional entropy coder — every move field is coded as a residual vs the
previous obs snapshot, then entropy-coded. So before training, decompose exactly
what headroom a learned codec has LEFT over that baseline:

  * A per-timestep conditional AE (the §16 spec form: 5 floats + obs -> latent)
    can only beat a per-field entropy coder by the CROSS-FIELD correlation among
    the reparam'd residuals = the TOTAL CORRELATION (Σ marginal H − joint H).
    Measured here as pairwise mutual informations among {pos-moving gate,
    yaw_residual, pitch_residual, on_ground}. If these are ~0, a per-timestep AE
    is analytically doomed to ~tie the baseline (the §16.2 null), no training
    needed.

  * A TEMPORAL codec could additionally exploit turn-rate / velocity MOMENTUM:
    H(residual_t) vs H(residual_t | residual_{t-1}). If conditioning the residual
    on the PREVIOUS residual reduces entropy materially, there is real headroom a
    per-timestep AE cannot reach but a recurrent/contextual one could.

All at the §15 parity-safe grids so bits are comparable. Plug-in entropy/MI; with
n~35k and ~15-30 codes/field the bias is <~0.02 bits (noted). This is the
"measure before building" gate one level deeper: it picks the AE architecture.

Usage:
    .venv/bin/python -m experiments.codec_loop.ae_headroom            # narrated
    .venv/bin/python -m experiments.codec_loop.ae_headroom --out results/sprint16/ae_headroom.json
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
    POS_RANGE, quant_scalar_zero_preserving,
)

B = 5
STATIONARY = 0.02


def _H(counter: Counter) -> float:
    n = sum(counter.values())
    if not n:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def _MI(pairs) -> float:
    """Mutual information I(X;Y) in bits from an iterable of (x,y) tuples."""
    jxy = Counter(pairs)
    n = sum(jxy.values())
    if not n:
        return 0.0
    px, py = Counter(), Counter()
    for (x, y), c in jxy.items():
        px[x] += c
        py[y] += c
    mi = 0.0
    for (x, y), c in jxy.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((px[x] / n) * (py[y] / n)))
    return mi


def _H_cond_prev(seq) -> tuple[float, float]:
    """(H(x_t), H(x_t | x_{t-1})) for a code sequence."""
    marg = Counter(seq)
    trans = Counter(zip(seq[:-1], seq[1:]))
    # H(x_t | x_{t-1}) = H(x_{t-1}, x_t) - H(x_{t-1})
    prev = Counter(seq[:-1])
    return _H(marg), _H(trans) - _H(prev)


def _pos_code(d: float) -> int:
    return quant_scalar_zero_preserving(d, -POS_RANGE, POS_RANGE, B)[1]


def _rot_code(res: float) -> int:
    return quant_scalar_zero_preserving(res, -ROT_RANGE, ROT_RANGE, B)[1]


def analyze(root: str) -> dict:
    actions, n_tp = load_move_actions(root)

    # per-packet reparam'd codes (in corpus/time order)
    pmove, og = [], []                 # all move packets
    yaw_seq, pitch_seq = [], []        # rot-bearing packets only (time-ordered subset)
    pos_axis_marg = Counter()
    # cross-field tuples (rot-bearing packets, where all fields coexist)
    cf = []                            # (pmove, yaw_code, pitch_code, on_ground)
    yc_pc = []                         # (yaw_code, pitch_code)
    for action, obs, _orig in actions:
        moving = 0
        if action.pos is not None:
            for d in action.pos:
                pos_axis_marg[_pos_code(d)] += 1
                if abs(d) >= STATIONARY:
                    moving = 1
        pmove.append(moving)
        og.append(1 if action.on_ground else 0)
        if action.rot is not None:
            yaw, pitch = action.rot
            yc = _rot_code(wrap180(yaw - float(obs["yaw"])))
            pc = _rot_code(pitch - float(obs["pitch"]))
            yaw_seq.append(yc)
            pitch_seq.append(pc)
            cf.append((moving, yc, pc, 1 if action.on_ground else 0))
            yc_pc.append((yc, pc))

    # --- marginal entropies (the per-field-entropy baseline components) ---
    H_pos_axis = _H(pos_axis_marg)
    H_yaw = _H(Counter(yaw_seq))
    H_pitch = _H(Counter(pitch_seq))
    H_og = _H(Counter(og))
    H_pmove = _H(Counter(pmove))

    # --- CROSS-FIELD correlation (per-timestep AE headroom) ---
    mi = {
        "yaw__pitch": _MI(yc_pc),
        "pmove__yaw": _MI([(m, y) for (m, y, _p, _o) in cf]),
        "pmove__pitch": _MI([(m, p) for (m, _y, p, _o) in cf]),
        "onground__yaw": _MI([(o, y) for (_m, y, _p, o) in cf]),
        "onground__pmove": _MI([(o, m) for (m, _y, _p, o) in cf]),
    }
    # upper bound on per-timestep AE win over per-field entropy coder:
    # total correlation >= max pairwise MI; report the SUM of a non-overlapping
    # estimate is unsafe, so report the dominant pairwise MIs + their sum as a
    # loose indicator (true TC is between max and sum).
    tc_lo = max(mi.values())
    tc_hi_loose = sum(mi.values())

    # --- TEMPORAL momentum (recurrent-codec headroom) ---
    Hy, Hy_cond = _H_cond_prev(yaw_seq)
    Hp, Hp_cond = _H_cond_prev(pitch_seq)
    # pos: per-axis sequence isn't clean (3/pkt); use pmove gate temporal instead
    Hm, Hm_cond = _H_cond_prev(pmove)

    return {
        "set": root, "n_move": len(actions), "n_rot": len(yaw_seq), "n_tp": n_tp,
        "grid_bits": B,
        "marginals_bits": {
            "pos_per_axis": round(H_pos_axis, 4), "yaw_res": round(H_yaw, 4),
            "pitch_res": round(H_pitch, 4), "on_ground": round(H_og, 4),
            "pmove_gate": round(H_pmove, 4),
        },
        "cross_field_MI_bits": {k: round(v, 4) for k, v in mi.items()},
        "per_timestep_AE_headroom_bits": {
            "min_total_correlation": round(tc_lo, 4),
            "loose_upper_sum_pairwise": round(tc_hi_loose, 4),
        },
        "temporal_momentum_bits": {
            "yaw_res": {"H": round(Hy, 4), "H_given_prev": round(Hy_cond, 4),
                        "reduction": round(Hy - Hy_cond, 4)},
            "pitch_res": {"H": round(Hp, 4), "H_given_prev": round(Hp_cond, 4),
                          "reduction": round(Hp - Hp_cond, 4)},
            "pmove_gate": {"H": round(Hm, 4), "H_given_prev": round(Hm_cond, 4),
                           "reduction": round(Hm - Hm_cond, 4)},
        },
    }


def _print(r: dict) -> None:
    print(f"\n===== §16.1 AE headroom: {r['set']} (n_move={r['n_move']}, n_rot={r['n_rot']}) =====")
    print("marginals (per-field entropy baseline, bits):", r["marginals_bits"])
    print("\nCROSS-FIELD MI (per-timestep AE's ONLY headroom over per-field coding):")
    for k, v in r["cross_field_MI_bits"].items():
        print(f"    I({k:18s}) = {v:.4f} bits")
    h = r["per_timestep_AE_headroom_bits"]
    print(f"  -> per-timestep AE headroom in [{h['min_total_correlation']:.4f}, "
          f"{h['loose_upper_sum_pairwise']:.4f}] bits/pkt (total correlation band)")
    print("\nTEMPORAL momentum (recurrent-codec headroom, H -> H|prev):")
    for k, v in r["temporal_momentum_bits"].items():
        print(f"    {k:11s}: {v['H']:.4f} -> {v['H_given_prev']:.4f}  (−{v['reduction']:.4f})")
    print("\nVerdict logic: if cross-field MI ~0 AND temporal reduction ~0, both AE forms "
          "tie the baseline (analytical null). Temporal reduction >> cross-field => the "
          "headroom is RECURRENT, not per-timestep.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="narrated", choices=list(SETS))
    ap.add_argument("--out", default="/tmp/ae_headroom.json")
    args = ap.parse_args()
    res = analyze(SETS[args.set])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    _print(res)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
