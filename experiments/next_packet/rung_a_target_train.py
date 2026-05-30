"""Rung A · target head — TRAIN→CHECKPOINT (§13.1.1, the closed-loop swap prereq).

`rung_a_target.py` is eval-only: `train_arm` returns (best_val_acc, dim) and never
persists a model. §13.1 puts that attack-target pointer into the *live* loop, so we
need a frozen artifact a separate process can load:

  (a) a `.pt` checkpoint  (scorer weights + the geom z-score stats + entity vocab)
  (b) a `feature_spec.json` the live selector reads to build per-candidate features
      in the EXACT order/normalization the model was trained on.

We REUSE the validated feature pipeline from `rung_a_target` (`load_attacks`,
`cand_features`, the baselines) so features are byte-identical to the eval head, and
mirror `train_arm`'s architecture/normalization here while adding model capture +
persistence.

Feature contract (load-bearing): every `cand_features` field —
[dx, dy, dz, dist, sin/cos(off_yaw), sin/cos(off_pitch)] (+ optional type one-hot) —
is computed from one live `entity_set` snapshot + the player's pos/yaw/pitch at
inference time. No `delta_tick`, no teacher-forced cadence (§11d). The attack target
is gaze-independent (KillAura auto-aims), so this decision needs no servo.

Run as a package module (the eval script uses relative imports):
    .venv/bin/python -m experiments.next_packet.rung_a_target_train \
        --rollouts-glob 'results/frozen_combat/rollout-*' \
        --out-dir results/rung_a_target_ckpt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from .ablation_r1_r3 import EntityVocab
from .rung_a_target import (
    baseline_nearest,
    baseline_nearest_hostile,
    cand_features,
    load_attacks,
)

# names for the geom block, matching cand_features() order in rung_a_target
FEATURE_NAMES = ["dx", "dy", "dz", "dist",
                 "off_yaw_sin", "off_yaw_cos", "off_pitch_sin", "off_pitch_cos"]
GEOM_ZSCORE_DIMS = [0, 1, 2, 3]  # dx,dy,dz,dist are z-scored; angles pass through


def build_model(dim, hidden):
    """Same architecture as rung_a_target.train_arm."""
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def geom_stats(train, evocab, use_type):
    """Mean/std of the 4 geom dims over all training candidates (mirrors train_arm)."""
    cols = [[] for _ in GEOM_ZSCORE_DIMS]
    for r in train:
        for c in r["cands"]:
            f = cand_features(c, evocab, use_type)
            for i, d in enumerate(GEOM_ZSCORE_DIMS):
                cols[i].append(f[d])
    mean = [float(np.mean(col)) for col in cols]
    std = [float(max(np.std(col), 1e-6)) for col in cols]
    return mean, std


def make_feat(evocab, use_type, mean, std):
    def feat(c):
        f = cand_features(c, evocab, use_type)
        for i, d in enumerate(GEOM_ZSCORE_DIMS):
            f[d] = (f[d] - mean[i]) / std[i]
        return f
    return feat


def train_capture(train, val, evocab, use_type, *, hidden, epochs, lr, seed, device):
    """Train the pointer head; return (model, mean, std, dim, final_acc, best_acc)."""
    mean, std = geom_stats(train, evocab, use_type)
    feat = make_feat(evocab, use_type, mean, std)
    dim = len(cand_features(train[0]["cands"][0], evocab, use_type))

    torch.manual_seed(seed)
    model = build_model(dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    tr = [(torch.tensor([feat(c) for c in r["cands"]], dtype=torch.float32, device=device),
           torch.tensor(r["label"], device=device)) for r in train]

    def eval_acc():
        model.eval()
        ok = 0
        with torch.no_grad():
            for r in val:
                X = torch.tensor([feat(c) for c in r["cands"]],
                                 dtype=torch.float32, device=device)
                if int(model(X).squeeze(-1).argmax().item()) == r["label"]:
                    ok += 1
        return ok / len(val) if val else float("nan")

    rng = random.Random(seed)
    best = 0.0
    for _e in range(epochs):
        model.train()
        rng.shuffle(tr)
        for X, y in tr:
            opt.zero_grad()
            ce(model(X).squeeze(-1).unsqueeze(0), y.unsqueeze(0)).backward()
            opt.step()
        best = max(best, eval_acc())
    final = eval_acc()
    return model, mean, std, dim, final, best


def feature_spec(evocab, use_type, dim, hidden, mean, std):
    return {
        "version": 1,
        "head": "rung_a_target",
        "canonical_extraction": "rung_a_target.cand_features",
        "feature_names": FEATURE_NAMES,        # geom block, in order
        "geom_zscore_dims": GEOM_ZSCORE_DIMS,  # which dims are z-scored
        "geom_mean": mean,
        "geom_std": std,
        "use_type": use_type,
        "type_onehot_vocab": list(evocab.types),  # type -> onehot index = .index
        "type_onehot_size": evocab.size,
        "input_dim": dim,
        "model": {
            "arch": "Linear(dim,h)->ReLU->Linear(h,h)->ReLU->Linear(h,1)",
            "hidden": hidden,
            "scores": "per-candidate scalar; argmax over candidates = pick",
        },
        "extraction_params": {
            "off_yaw": "radians((bearing_yaw - player_yaw + 180)%360 - 180); "
                       "bearing_yaw = degrees(atan2(-dx, dz))",
            "off_pitch": "radians((bearing_pitch - player_pitch + 180)%360 - 180); "
                         "bearing_pitch = degrees(atan2(-dy, horiz))",
            "dist": "euclidean(entity_pos - player_pos)",
            "candidate_set": "entity_set rows (player excluded), dist-sorted",
        },
        "live_readable_only": True,
        "no_delta_tick": True,
        "notes": "All features come from one live entity_set snapshot + player "
                 "pos/yaw/pitch. Gaze-independent decision (KillAura aims).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-glob", default="results/frozen_combat/rollout-*")
    ap.add_argument("--out-dir", default="results/rung_a_target_ckpt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary", choices=["geom", "geomtype"], default="geom",
                    help="arm frozen as the deployable checkpoint; geom default "
                         "(fewer live fields, no entity-type OOV).")
    args = ap.parse_args()

    rows = load_attacks(sorted(glob.glob(args.rollouts_glob)))
    if len(rows) < 4:
        print(f"too few ATTACK events: {len(rows)} from {args.rollouts_glob}",
              file=sys.stderr)
        sys.exit(2)
    types = sorted({c["type"] for r in rows for c in r["cands"] if c["type"]})
    evocab = EntityVocab(types)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {
        "n_events": len(rows), "n_train": len(train), "n_val": n_val,
        "entity_types": evocab.size, "seed": args.seed, "epochs": args.epochs,
        "hidden": args.hidden, "lr": args.lr,
        "baseline_nearest_val": baseline_nearest(val),
        "baseline_nearest_hostile_val": baseline_nearest_hostile(val),
        "baseline_nearest_all": baseline_nearest(rows),
        "baseline_nearest_hostile_all": baseline_nearest_hostile(rows),
        "arms": {},
    }

    saved = {}
    for use_type, tag in [(False, "geom"), (True, "geomtype")]:
        model, mean, std, dim, final, best = train_capture(
            train, val, evocab, use_type, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, seed=args.seed, device=device)
        metrics["arms"][tag] = {"use_type": use_type, "dim": dim,
                                "val_acc_final": final, "val_acc_best": best}
        ckpt = os.path.join(args.out_dir, f"model_{tag}.pt")
        torch.save({"state_dict": model.state_dict(), "dim": dim,
                    "hidden": args.hidden, "use_type": use_type,
                    "geom_mean": mean, "geom_std": std}, ckpt)
        saved[tag] = (model, mean, std, dim, use_type)
        print(f"[{tag}] dim={dim} val_acc final={final:.3f} best={best:.3f} -> {ckpt}")

    pmodel, pmean, pstd, pdim, puse = saved[args.primary]
    spec = feature_spec(evocab, puse, pdim, args.hidden, pmean, pstd)
    with open(os.path.join(args.out_dir, "feature_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    torch.save({"state_dict": pmodel.state_dict(), "dim": pdim,
                "hidden": args.hidden, "use_type": puse,
                "geom_mean": pmean, "geom_std": pstd,
                "primary_arm": args.primary},
               os.path.join(args.out_dir, "model.pt"))
    metrics["primary_arm"] = args.primary
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"primary={args.primary} -> {os.path.join(args.out_dir, 'model.pt')}")
    print(f"baselines(all): nearest={metrics['baseline_nearest_all']:.3f} "
          f"nearest_hostile={metrics['baseline_nearest_hostile_all']:.3f}")


if __name__ == "__main__":
    main()
