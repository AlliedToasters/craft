#!/usr/bin/env python3
"""§18.2 — train + SAVE the g_t-conditioned interact-target prior for live serving.

filter_bits.py trains its arms in-memory for the offline gap measurement. 18.2
needs the trained prior on disk so the codec sidecar can serve it live: per
outbound interact, score the entity_set candidates under P(idx | geom, type,
obs.policy) and entropy-code the target index (rate = -log2 P(true idx)).

Trains on the FULL filter_dataset.jsonl (no val holdout — we want the best prior
to deploy) and saves both arms used live:
  geom+type         the MODE-BLIND prior (control: can't see g_t).
  geom+type+policy  the MODE-AWARE prior (the g_t codec).
Each bundle carries everything the sidecar needs to rebuild the exact feature
vector: state_dict, z-score mean/std, species vocab, arm tag, dim, hidden.

Usage:
    .venv/bin/python -m experiments.codec_loop.filter_prior_train \
        --data results/sprint18/filter_dataset.jsonl --out-dir results/sprint18/prior
"""
from __future__ import annotations

import argparse
import os
import random

import torch

from experiments.codec_loop.filter_bits import (
    ARMS, _build, _fit_temp, _load, _logits_labels, _train,
    _zscore_apply, _zscore_fit,
)

SAVE_ARMS = ["geom+type", "geom+type+policy"]


def main():
    ap = argparse.ArgumentParser(description="§18.2 train+save the interact-target prior")
    ap.add_argument("--data", default="results/sprint18/filter_dataset.jsonl")
    ap.add_argument("--out-dir", default="results/sprint18/prior")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert set(SAVE_ARMS).issubset(ARMS)
    rows, vocab = _load(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Held-out calib split (by scene) to fit the temperature — these small models
    # are overconfident (§18.0/18.1), so the SERVED prior must carry a calibration
    # temperature or its live raw CE is hugely inflated (esp. the sharper policy
    # arm). Train on the train split, fit T on calib, deploy model + T.
    scenes = sorted({r["scene_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_cal = max(1, int(len(scenes) * 0.2))
    cal_s = set(scenes[:n_cal])
    tr_rows = [r for r in rows if r["scene_id"] not in cal_s]
    cal_rows = [r for r in rows if r["scene_id"] in cal_s]

    for arm in SAVE_ARMS:
        tr = _build(tr_rows, arm, vocab)
        cal = _build(cal_rows, arm, vocab)
        mean, std = _zscore_fit(tr)
        _zscore_apply(tr, mean, std)
        _zscore_apply(cal, mean, std)
        dim = len(tr[0][0][0])
        m = _train(tr, dim, hidden=args.hidden, epochs=args.epochs,
                   lr=args.lr, seed=args.seed, device=device)
        temp = _fit_temp(_logits_labels(m, cal, device))
        path = os.path.join(args.out_dir, f"prior_{arm.replace('+', '_')}.pt")
        torch.save({
            "arm": arm, "dim": dim, "hidden": args.hidden, "temp": temp,
            "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
            "zscore_mean": mean, "zscore_std": std,
            "vocab": list(vocab),                # species order for the one-hot
            "n_train_rows": len(tr_rows),
        }, path)
        print(f"saved {arm:>18}  dim={dim}  temp={temp:.2f}  -> {path}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
