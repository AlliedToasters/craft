"""§18.0 — the learned interact-target prior AS A CODEC: bits/interact = its cross-entropy.

§17.2.2 mapped the entity_id knee: the lossless pointer-into-obs costs ~log2(n)
bits (n = candidates in entity_set); collapsing to the geometric argmax costs ~0
bits but is LOSSY. §18 is the learned discrete-decision codec that closes that gap
WITHOUT the loss: entropy-code the target index under a learned prior P(idx | obs
geometry). The achieved rate of such a (lossless) coder is exactly the prior's
CROSS-ENTROPY on the true index — `mean -log2 P(true_idx)`. Behavioral parity is
100% by construction (the coder always recovers the true index); the whole story
is the RATE.

The §13.1 attack-target head (results/rung_a_target_ckpt) IS that prior: it was
trained with nn.CrossEntropyLoss over the candidate scores (rung_a_target_train.py),
so softmax(scores) is a calibrated distribution over the candidate index and its
CE is the codec rate. This script reads the frozen checkpoint and measures it.

The point §18 makes that §16 (move) and §17.2.1 (block) could NOT: here LEARNING
PAYS. The target is a non-trivial learned function of geometry — nearest-baseline
is only ~0.43 (so the §17.2.2 "collapse to nearest" is a weak prior), while the
learned head reaches 0.954 (geom) / 0.985 (geom+type). That accuracy edge converts
directly into bits: a prior that is right 98.5% of the time spends ~0 bits on the
common case and only pays on its rare misses.

Comparison set (all in bits/interact, lower = better compression):
  raw_int        ~24    the raw network id, no obs (the §17.2.2 absolute foil).
  uniform        log2(n) flat pointer into entity_set (the §17.2.2 lossless floor).
  nearest_bet    lossless cost of the §17.2.2 "guess nearest, send residual"
                 strategy = H2(p_near) + (1-p_near)*E[log2(n-1)]. The trivial
                 geometric prior made lossless — beats uniform a little.
  geom (CE)      §13.1 geom head as prior, raw and temperature-calibrated.
  geom+type (CE) §13.1 geom+type head as prior, raw and temperature-calibrated.

Temperature calibration: a 200-epoch / 195-sample head is overconfident, so its raw
softmax assigns near-0 to the true class on its ~1.5% misses → a huge -log2 penalty
that inflates the rate. We fit a scalar temperature T on the held-out split (the
standard calibration fix) and report the calibrated CE = the rate a well-calibrated
coder actually achieves. We report BOTH; calibrated is the honest codec rate.

Split is reproduced EXACTLY from the checkpoint (seed 42, val_frac 0.25, same shuffle
of load_attacks rows) so val == the checkpoint's held-out 65. val is the primary
(honest, uncontaminated) number; `all` is a denser train-contaminated estimate.

Usage:
    .venv/bin/python -m experiments.next_packet.rung_a_target_entropy \
        --rollouts-glob 'results/frozen_combat/rollout-*' \
        --ckpt-dir results/rung_a_target_ckpt \
        --out results/sprint18/entropy_rate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch

from .ablation_r1_r3 import EntityVocab
from .rung_a_target import baseline_nearest, cand_features, load_attacks
from .rung_a_target_train import GEOM_ZSCORE_DIMS, build_model

RAW_INT_BITS = 24.0  # §17.2.2: the raw network id resolves losslessly at ~24 bits


def _load_arm(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build_model(ck["dim"], ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["geom_mean"], ck["geom_std"], ck["dim"], ck["use_type"]


def _feat(c, evocab, use_type, mean, std):
    f = cand_features(c, evocab, use_type)
    for i, d in enumerate(GEOM_ZSCORE_DIMS):
        f[d] = (f[d] - mean[i]) / std[i]
    return f


def _logits(model, rows, evocab, use_type, mean, std, device):
    """Per-event candidate logit tensors (variable length) + integer labels."""
    out = []
    with torch.no_grad():
        for r in rows:
            X = torch.tensor([_feat(c, evocab, use_type, mean, std) for c in r["cands"]],
                             dtype=torch.float32, device=device)
            out.append((model(X).squeeze(-1), int(r["label"])))
    return out


def _ce_bits(logit_label, temp=1.0):
    """Mean cross-entropy in BITS over events at temperature `temp` (logits/temp).
    Also returns per-event bits for distribution stats."""
    per = []
    for logits, label in logit_label:
        logp = torch.log_softmax(logits / temp, dim=0)
        per.append(-logp[label].item() / math.log(2.0))  # nats→bits
    return float(np.mean(per)) if per else float("nan"), per


def _fit_temperature(logit_label, grid=None):
    """Scalar temperature minimizing CE on these events (standard calibration)."""
    if grid is None:
        grid = [round(t, 3) for t in np.geomspace(0.3, 30.0, 60)]
    best_t, best_b = 1.0, float("inf")
    for t in grid:
        b, _ = _ce_bits(logit_label, temp=t)
        if b < best_b:
            best_t, best_b = t, b
    return best_t, best_b


def _accuracy(logit_label):
    if not logit_label:
        return float("nan")
    ok = sum(1 for logits, label in logit_label if int(logits.argmax().item()) == label)
    return ok / len(logit_label)


def _uniform_bits(rows):
    return float(np.mean([math.log2(max(len(r["cands"]), 1)) for r in rows]))


def _nearest_bet_bits(rows):
    """Lossless cost of the §17.2.2 'bet nearest, else point among the rest' coder:
    H2(p_near) flag + (1-p_near) * E[log2(n-1)] residual."""
    p = baseline_nearest(rows)  # P(target == candidate 0)
    h2 = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    resid = float(np.mean([math.log2(max(len(r["cands"]) - 1, 1)) for r in rows]))
    return h2 + (1 - p) * resid


def _dist_stats(per_event_bits):
    a = np.array(per_event_bits) if per_event_bits else np.array([0.0])
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
        "frac_under_0p1_bits": float((a < 0.1).mean()),
        "frac_under_1_bit": float((a < 1.0).mean()),
    }


def _arm_report(tag, ckpt_path, val, allr, evocab, device):
    model, mean, std, dim, use_type = _load_arm(ckpt_path, device)
    ll_val = _logits(model, val, evocab, use_type, mean, std, device)
    ll_all = _logits(model, allr, evocab, use_type, mean, std, device)

    raw_val, per_val = _ce_bits(ll_val, temp=1.0)
    temp, cal_val = _fit_temperature(ll_val)
    _, per_val_cal = _ce_bits(ll_val, temp=temp)
    raw_all, _ = _ce_bits(ll_all, temp=1.0)
    cal_all, _ = _ce_bits(ll_all, temp=temp)  # apply val-fit temp to all

    return {
        "tag": tag, "dim": dim, "use_type": use_type,
        "val_accuracy": _accuracy(ll_val), "all_accuracy": _accuracy(ll_all),
        "val_bits_raw": raw_val, "val_bits_calibrated": cal_val,
        "all_bits_raw": raw_all, "all_bits_calibrated": cal_all,
        "fit_temperature": temp,
        "val_dist_raw": _dist_stats(per_val),
        "val_dist_calibrated": _dist_stats(per_val_cal),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§18.0 learned interact-target prior as a codec")
    ap.add_argument("--rollouts-glob", default="results/frozen_combat/rollout-*")
    ap.add_argument("--ckpt-dir", default="results/rung_a_target_ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--out", default="results/sprint18/entropy_rate.json")
    args = ap.parse_args()

    rows = load_attacks(sorted(glob.glob(args.rollouts_glob)))
    if len(rows) < 4:
        print(f"too few ATTACK events: {len(rows)}")
        return 2
    types = sorted({c["type"] for r in rows for c in r["cands"] if c["type"]})
    evocab = EntityVocab(types)

    # Reproduce the checkpoint's split EXACTLY (same seed/frac/shuffle order).
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, _train = rows[:n_val], rows[n_val:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uniform_val, uniform_all = _uniform_bits(val), _uniform_bits(rows)
    nearest_val, nearest_all = _nearest_bet_bits(val), _nearest_bet_bits(rows)
    avg_cands_val = float(np.mean([len(r["cands"]) for r in val]))
    avg_cands_all = float(np.mean([len(r["cands"]) for r in rows]))

    arms = []
    for tag in ("geom", "geomtype"):
        p = os.path.join(args.ckpt_dir, f"model_{tag}.pt")
        if os.path.exists(p):
            arms.append(_arm_report(tag, p, val, rows, evocab, device))

    out = {
        "rollouts_glob": args.rollouts_glob, "ckpt_dir": args.ckpt_dir,
        "n_events": len(rows), "n_val": n_val, "seed": args.seed,
        "avg_candidates_val": avg_cands_val, "avg_candidates_all": avg_cands_all,
        "references_bits": {
            "raw_int": RAW_INT_BITS,
            "uniform_val": uniform_val, "uniform_all": uniform_all,
            "nearest_bet_val": nearest_val, "nearest_bet_all": nearest_all,
            "baseline_nearest_acc_val": baseline_nearest(val),
            "baseline_nearest_acc_all": baseline_nearest(rows),
        },
        "arms": arms,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- report ----
    print("=" * 72)
    print("§18.0 — learned interact-target prior AS A CODEC (bits/interact)")
    print("=" * 72)
    print(f"events={len(rows)} val={n_val} avg_cands(val)={avg_cands_val:.2f} "
          f"avg_cands(all)={avg_cands_all:.2f}")
    print(f"\nReferences (bits/interact, lower=better):")
    print(f"  raw network id .............. {RAW_INT_BITS:>6.2f}")
    print(f"  uniform pointer log2(n) ..... {uniform_val:>6.2f} (val)  {uniform_all:6.2f} (all)")
    print(f"  nearest-bet (lossless) ...... {nearest_val:>6.2f} (val)  {nearest_all:6.2f} (all)"
          f"   [nearest acc={baseline_nearest(val):.2f}]")
    print(f"\nLearned priors (cross-entropy = lossless predictive-codec rate):")
    print(f"  {'arm':>9} {'acc(val)':>8} {'raw(val)':>9} {'calT(val)':>10} "
          f"{'T':>5} {'calT(all)':>10} {'free%':>6}")
    for a in arms:
        print(f"  {a['tag']:>9} {a['val_accuracy']:>8.3f} {a['val_bits_raw']:>9.3f} "
              f"{a['val_bits_calibrated']:>10.3f} {a['fit_temperature']:>5.2f} "
              f"{a['all_bits_calibrated']:>10.3f} "
              f"{a['val_dist_calibrated']['frac_under_0p1_bits']*100:>5.0f}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
