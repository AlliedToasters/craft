#!/usr/bin/env python3
"""§16.0 — Characterize the CONDITIONAL residual of the move stream.

Deliverable (neural_interface.md §16.0): size the prize a *conditional* learned
codec could win, and from that pick/confirm the family + freeze the objective.
NO net code, NO training — pure entropy measurement over the frozen corpus, at
the SAME quantization grids §15 proved behaviorally parity-safe, so every number
is in bits comparable to the §15 `float_bits` axis.

THE THREE-LEVEL LADDER (the framing this script measures):

  1. FIXED-POINT ALLOCATION (§15 quantizer): spends a fixed bit budget per field
     regardless of the value distribution. zero_preserving@b5 = 5 bits/axis pos,
     5 yaw, 5 pitch = 25 float bits/full packet. This is the parity-safe floor we
     already pay.
  2. MARGINAL ENTROPY on the same grid (= an arithmetic coder on the quantizer's
     own symbols, NO conditioning, NO learning). H(code) < allocated bits because
     the distributions are peaked (56% of pos deltas are stationary; turns are
     small). The gap 1->2 is a FREE win available to any codec — it is NOT what
     justifies learning.
  3. CONDITIONAL ENTROPY given the obs the decoder already holds (obs.{x,y,z}
     reference frame, obs.{yaw,pitch} last-known rotation, a still/moving gate).
     H(code | obs). The gap 2->3 is what *conditioning* buys — THE prize that
     justifies (or refutes) the conditional autoencoder. If 2->3 is small, the
     stream is near its memoryless floor and a learned codec "buys nothing"
     (the §16.2 pre-registered null branch) — itself the finding.

CONDITIONING SIGNALS (all decoder-available — no rollout-id, no Baritone path;
§B leakage anti-pattern #2 holds):
  * pos: a still/moving GATE (|true delta| < STATIONARY). At rest the codes
    collapse to exactly 0 (zero_preserving) -> 0 bits. This is the structural
    at-rest gate the §15 drift-fatal prior demands, measured here as an entropy
    reduction. The gate's own cost is tiny AND temporally near-constant (run
    lengths), so it is ~free to a recurrent decoder.
  * rot: code the value RELATIVE to obs.{yaw,pitch} (the decoder's last-known
    rotation) instead of absolute. Same grid/step, but the residual distribution
    is peaked at 0 (the player is usually not mid-turn) -> lower entropy. This is
    pure conditional coding, zero learning.

NOTE ON SPLITS: marginal/conditional ENTROPY is a population statistic; pooling
all rollouts is correct here. (§B's by-rollout mandate governs *learned*
predictor generalization in 16.1, not this 16.0 measurement.)

Usage:
    .venv/bin/python -m experiments.codec_loop.cond_residual            # narrated
    .venv/bin/python -m experiments.codec_loop.cond_residual --set combat
    .venv/bin/python -m experiments.codec_loop.cond_residual --out results/sprint16/cond_residual.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

from experiments.codec_loop.offline_fidelity import SETS, load_move_actions
from experiments.codec_loop.quantize import (
    POS_RANGE, YAW_LO, YAW_HI, PITCH_LO, PITCH_HI,
    quant_scalar, quant_scalar_zero_preserving, _wrap360,
)

# Parity-safe grids (from §15 RESULTS): zero_preserving pos @ b5 holds lossless
# behavioral parity; yaw/pitch @ b5 are within the validated fidelity band.
B = 5
STATIONARY = 0.02   # |axis delta| below this = standing still (recon_hist.py)


def _wrap180(d: float) -> float:
    """Signed shortest angular difference in (-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


# ---- discrete entropy helpers (plug-in; n>>alphabet so bias is negligible) ----
def H(counter: Counter) -> float:
    """Shannon entropy in BITS of a symbol-count distribution."""
    n = sum(counter.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def H_cond(joint: Counter) -> float:
    """H(X | Y) in bits from a Counter keyed by (y, x)."""
    n = sum(joint.values())
    if n == 0:
        return 0.0
    py: Counter = Counter()
    for (y, _), c in joint.items():
        py[y] += c
    h = 0.0
    for y, ny in py.items():
        cond = Counter()
        for (yy, x), c in joint.items():
            if yy == y:
                cond[x] += c
        h += (ny / n) * H(cond)
    return h


def _pos_code(d: float) -> int:
    _, code = quant_scalar_zero_preserving(d, -POS_RANGE, POS_RANGE, B)
    return code


def _yaw_abs_code(yaw: float) -> int:
    _, code = quant_scalar(_wrap360(yaw), YAW_LO, YAW_HI, B)
    return code


def _yaw_rel_code(yaw: float, obs_yaw: float) -> int:
    # residual coded on a zero_preserving grid over [-180,180] (mode = no turn)
    _, code = quant_scalar_zero_preserving(_wrap180(yaw - obs_yaw), -180.0, 180.0, B)
    return code


def _pitch_abs_code(pitch: float) -> int:
    _, code = quant_scalar(pitch, PITCH_LO, PITCH_HI, B)
    return code


def _pitch_rel_code(pitch: float, obs_pitch: float) -> int:
    # pitch is bounded [-90,90]; residual rarely large. Same step as absolute
    # (5.8 deg) on a zero_preserving grid over the bounded residual span [-180,180].
    _, code = quant_scalar_zero_preserving(pitch - obs_pitch, -180.0, 180.0, B)
    return code


def analyze(root: str) -> dict:
    actions, n_tp = load_move_actions(root)

    # --- POS: per-axis marginal vs conditional-on-gate ------------------------
    pos_marg: Counter = Counter()                 # code
    pos_joint_gate: Counter = Counter()           # (gate, code)
    pkt_gate_seq: list[int] = []                   # packet-level moving? for autocorr
    n_pos_axis = 0
    n_stat = 0

    # --- ROT: absolute vs obs-relative ---------------------------------------
    yaw_abs: Counter = Counter()
    yaw_rel: Counter = Counter()
    pitch_abs: Counter = Counter()
    pitch_rel: Counter = Counter()

    # --- BOOLS ---------------------------------------------------------------
    onground_marg: Counter = Counter()
    onground_joint: Counter = Counter()           # (obs_on_ground, on_ground)
    hcoll_marg: Counter = Counter()

    # per-packet-type field presence (for volume-weighted bits/packet)
    type_counts: Counter = Counter()

    for action, obs, _orig in actions:
        type_counts[action.packet_type] += 1

        # bools (always present)
        og = bool(action.on_ground)
        onground_marg[og] += 1
        obs_og = bool(obs.get("on_ground", og))
        onground_joint[(obs_og, og)] += 1
        hcoll_marg[bool(action.horizontal_collision)] += 1

        if action.pos is not None:
            pkt_moving = 0
            for d in action.pos:
                code = _pos_code(d)
                gate = 1 if abs(d) >= STATIONARY else 0
                pos_marg[code] += 1
                pos_joint_gate[(gate, code)] += 1
                n_pos_axis += 1
                if gate == 0:
                    n_stat += 1
                else:
                    pkt_moving = 1
            # packet-level moving? gate (any axis moving) — the temporally
            # coherent signal a recurrent decoder tracks (NOT axis-interleaved).
            pkt_gate_seq.append(pkt_moving)

        if action.rot is not None:
            yaw, pitch = action.rot
            oy = float(obs.get("yaw", yaw))
            op = float(obs.get("pitch", pitch))
            yaw_abs[_yaw_abs_code(yaw)] += 1
            yaw_rel[_yaw_rel_code(yaw, oy)] += 1
            pitch_abs[_pitch_abs_code(pitch)] += 1
            pitch_rel[_pitch_rel_code(pitch, op)] += 1

    # gate temporal structure (PACKET-level, not axis-interleaved): H(gate) vs
    # H(gate_t | gate_{t-1}); run-length mean. Shows the moving/still phase is
    # temporally coherent -> the gate is ~free side-info to a recurrent decoder.
    gate_marg = Counter(pkt_gate_seq)
    gate_trans: Counter = Counter()
    for a, b in zip(pkt_gate_seq, pkt_gate_seq[1:]):
        gate_trans[(a, b)] += 1
    runs = 0
    prev = None
    for g in pkt_gate_seq:
        if g != prev:
            runs += 1
            prev = g
    mean_run = (len(pkt_gate_seq) / runs) if runs else 0.0

    # --- assemble per-field bit numbers --------------------------------------
    pos_H_marg = H(pos_marg)
    pos_H_cond = H_cond(pos_joint_gate)
    gate_H = H(gate_marg)
    gate_H_cond = H_cond(gate_trans)

    yaw_H_abs = H(yaw_abs)
    yaw_H_rel = H(yaw_rel)
    pitch_H_abs = H(pitch_abs)
    pitch_H_rel = H(pitch_rel)

    og_H = H(onground_marg)
    og_H_cond = H_cond(onground_joint)
    hcoll_H = H(hcoll_marg)

    fields = {
        "pos_per_axis": {
            "alloc_bits": B,
            "H_marginal": round(pos_H_marg, 4),
            "H_cond_gate": round(pos_H_cond, 4),
            "gate_cost_bits": round(gate_H, 4),
            "gate_cost_bits_given_prev": round(gate_H_cond, 4),
            "gate_mean_run_len": round(mean_run, 2),
            "pct_stationary": round(100.0 * n_stat / n_pos_axis, 1) if n_pos_axis else 0.0,
            "free_win_alloc_to_marg": round(B - pos_H_marg, 4),
            "cond_win_marg_to_cond": round(pos_H_marg - pos_H_cond, 4),
        },
        "yaw": {
            "alloc_bits": B,
            "H_marginal_abs": round(yaw_H_abs, 4),
            "H_cond_obs_relative": round(yaw_H_rel, 4),
            "free_win_alloc_to_marg": round(B - yaw_H_abs, 4),
            "cond_win_abs_to_rel": round(yaw_H_abs - yaw_H_rel, 4),
        },
        "pitch": {
            "alloc_bits": B,
            "H_marginal_abs": round(pitch_H_abs, 4),
            "H_cond_obs_relative": round(pitch_H_rel, 4),
            "free_win_alloc_to_marg": round(B - pitch_H_abs, 4),
            "cond_win_abs_to_rel": round(pitch_H_abs - pitch_H_rel, 4),
        },
        "on_ground": {
            "alloc_bits": 1,
            "H_marginal": round(og_H, 4),
            "H_cond_obs": round(og_H_cond, 4),
        },
        "horizontal_collision": {
            "alloc_bits": 1,
            "H_marginal": round(hcoll_H, 4),
        },
    }

    # --- volume-weighted bits/packet at each ladder level --------------------
    # A move packet carries pos iff its type has pos; rot iff its type has rot;
    # bools always. Weight each field's per-occurrence bits by its presence rate.
    n = len(actions)
    f_pos = sum(type_counts[t] for t in
                ("minecraft:move_player_pos", "minecraft:move_player_pos_rot")) / n
    f_rot = sum(type_counts[t] for t in
                ("minecraft:move_player_rot", "minecraft:move_player_pos_rot")) / n

    def per_pkt(pos_bits, yaw_bits, pitch_bits, og_bits, hc_bits):
        return (f_pos * 3 * pos_bits + f_rot * (yaw_bits + pitch_bits)
                + og_bits + hc_bits)

    alloc = per_pkt(B, B, B, 1, 1)
    marg = per_pkt(pos_H_marg, yaw_H_abs, pitch_H_abs, og_H, hcoll_H)
    cond = per_pkt(pos_H_cond + gate_H_cond / 3.0,   # gate shared across 3 axes, ~free
                   yaw_H_rel, pitch_H_rel, og_H_cond, hcoll_H)

    ladder = {
        "n_move_packets": n,
        "frac_with_pos": round(f_pos, 3),
        "frac_with_rot": round(f_rot, 3),
        "bits_per_packet_alloc_fixedpoint": round(alloc, 3),
        "bits_per_packet_marginal_entropy": round(marg, 3),
        "bits_per_packet_conditional_entropy": round(cond, 3),
        "free_win_alloc_to_marg": round(alloc - marg, 3),
        "cond_win_marg_to_cond": round(marg - cond, 3),
        "total_compression_x_vs_alloc": round(alloc / cond, 2) if cond else None,
    }

    return {
        "set": root, "n_move_packets": n, "n_tp_excluded": n_tp,
        "grid": {"bits": B, "pos_mode": "zero_preserving", "pos_range": POS_RANGE,
                 "stationary_thresh": STATIONARY},
        "type_counts": dict(type_counts),
        "fields": fields,
        "ladder": ladder,
    }


def _print(res: dict) -> None:
    print(f"\n===== §16.0 conditional-residual: {res['set']} "
          f"(n={res['n_move_packets']} move pkts, {res['n_tp_excluded']} TP excl) =====")
    print(f"grid: zero_preserving b{res['grid']['bits']} R{res['grid']['pos_range']} "
          f"| stationary |d|<{res['grid']['stationary_thresh']}")
    f = res["fields"]
    print("\n  field           alloc  H_marg   H_cond   free(a→m)  cond(m→c)")
    p = f["pos_per_axis"]
    print(f"  pos/axis        {p['alloc_bits']:>5}  {p['H_marginal']:>6}  "
          f"{p['H_cond_gate']:>6}   {p['free_win_alloc_to_marg']:>7}   "
          f"{p['cond_win_marg_to_cond']:>7}   (gate {p['gate_cost_bits']}b, "
          f"run~{p['gate_mean_run_len']}, stat {p['pct_stationary']}%)")
    y = f["yaw"]
    print(f"  yaw             {y['alloc_bits']:>5}  {y['H_marginal_abs']:>6}  "
          f"{y['H_cond_obs_relative']:>6}   {y['free_win_alloc_to_marg']:>7}   "
          f"{y['cond_win_abs_to_rel']:>7}")
    pt = f["pitch"]
    print(f"  pitch           {pt['alloc_bits']:>5}  {pt['H_marginal_abs']:>6}  "
          f"{pt['H_cond_obs_relative']:>6}   {pt['free_win_alloc_to_marg']:>7}   "
          f"{pt['cond_win_abs_to_rel']:>7}")
    og = f["on_ground"]
    print(f"  on_ground       {og['alloc_bits']:>5}  {og['H_marginal']:>6}  "
          f"{og['H_cond_obs']:>6}")
    hc = f["horizontal_collision"]
    print(f"  horiz_collision {hc['alloc_bits']:>5}  {hc['H_marginal']:>6}")
    L = res["ladder"]
    print(f"\n  BITS/PACKET (vol-weighted: {L['frac_with_pos']*100:.0f}% carry pos, "
          f"{L['frac_with_rot']*100:.0f}% carry rot):")
    print(f"    1. fixed-point alloc (§15 quantizer floor) : {L['bits_per_packet_alloc_fixedpoint']:>7}")
    print(f"    2. marginal entropy (free, arith-coder)    : {L['bits_per_packet_marginal_entropy']:>7}"
          f"   (−{L['free_win_alloc_to_marg']} free)")
    print(f"    3. conditional entropy (the codec target)  : {L['bits_per_packet_conditional_entropy']:>7}"
          f"   (−{L['cond_win_marg_to_cond']} from conditioning)")
    print(f"    total compression vs alloc: {L['total_compression_x_vs_alloc']}×")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="narrated", choices=list(SETS))
    ap.add_argument("--out", default="/tmp/cond_residual.json")
    args = ap.parse_args()
    res = analyze(SETS[args.set])
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    _print(res)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
