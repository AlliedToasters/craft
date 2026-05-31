#!/usr/bin/env python3
"""§18.1 — the bits g_t buys: mode-blind vs mode-aware interact-target prior.

Reads the mixed-filter dataset (filter_capture.py) and trains the §13.1
per-candidate target scorer under four feature arms, reporting each arm's
cross-entropy in BITS = its lossless predictive-codec rate (the §18.0 metric):

  geom                 dx,dy,dz,dist (z-scored) + sin/cos off_yaw/off_pitch.
  geom+type            + species one-hot.
  geom+type+policy     + the g_t bit (filter_passive) broadcast to every
                       candidate. THE mode-aware arm.
  geom+type+attackable + the per-candidate `attackable` ORACLE (policy x type,
                       logged-not-fed by design) — the rate ceiling.

The point: a matched scene captured under both g_t modes has IDENTICAL geom+type
features but (when flipped) DIFFERENT labels. So the mode-blind arms (geom,
geom+type) MUST incur loss on the flip — they cannot tell the two modes apart.
The mode-aware arm sees the g_t bit and resolves it. The BITS GAP
(geom+type  −  geom+type+policy) = what modeling in-world context (the operator's
filter policy) buys on the discrete-target channel. The dual, on the g_t axis, of
§18.0's "learning is the compression": here CONTEXT is the compression.

Split is BY SCENE (a scene's two mode-rows go to the same side) so the val gap
measures generalization, not memorized scene geometry. We also report the gap
restricted to FLIPPED scenes, where mode is the whole story.

Usage:
    .venv/bin/python -m experiments.codec_loop.filter_bits \
        --data results/sprint18/filter_dataset.jsonl \
        --out results/sprint18/filter_bits.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

GEOM_Z = [0, 1, 2, 3]  # dx,dy,dz,dist z-scored; angles pass through
ARMS = ["geom", "geom+type", "geom+type+policy", "geom+type+attackable"]


def _load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    species = sorted({c["type"] for r in rows for c in r["cands"]})
    return rows, {s: i for i, s in enumerate(species)}


def _structural_mi(rows):
    """The training-free headline: bits g_t buys = I(target ; g_t | scene).

    The executor is deterministic — g_t fixes the target given the scene — so
    H(target | scene, g_t) = 0 and the gap reduces to H(target | scene): a prior
    that sees the (geom,type) scene but NOT g_t must spread its mass over the
    targets the scene takes across modes. Per scene, that is the entropy of its
    target distribution over the captured modes; mean over rows = the bits a
    mode-blind prior MUST pay that a g_t-aware one need not. (For balanced 2-mode
    capture this equals flip_rate × 1 bit.) Robust to model/overfit — it is a
    property of the data's matched-pair structure, not of any trained model."""
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene_id"], []).append(r["label"])
    total_rows = sum(len(v) for v in by_scene.values())
    mi = 0.0
    for labels in by_scene.values():
        n = len(labels)
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        mi += h * n  # weight by rows in this scene
    return mi / total_rows if total_rows else float("nan")


def _feat(c, gt, arm, vocab):
    f = [c["dx"], c["dy"], c["dz"], c["dist"],
         math.sin(c["off_yaw"]), math.cos(c["off_yaw"]),
         math.sin(c["off_pitch"]), math.cos(c["off_pitch"])]
    if "type" in arm:
        oh = [0.0] * len(vocab)
        oh[vocab[c["type"]]] = 1.0
        f += oh
    if "policy" in arm:
        f += [float(gt["filter_passive"])]
    if "attackable" in arm:
        f += [float(c["attackable"])]
    return f


def _build(rows, arm, vocab):
    """List of (feature matrix [n_cand, dim], label int) per event."""
    out = []
    for r in rows:
        X = [_feat(c, r["gt"], arm, vocab) for c in r["cands"]]
        out.append((X, int(r["label"])))
    return out


def _zscore_fit(events):
    cols = [[] for _ in GEOM_Z]
    for X, _ in events:
        for row in X:
            for i, d in enumerate(GEOM_Z):
                cols[i].append(row[d])
    mean = [float(np.mean(c)) for c in cols]
    std = [float(np.std(c)) or 1e-6 for c in cols]
    return mean, std


def _zscore_apply(events, mean, std):
    for X, _ in events:
        for row in X:
            for i, d in enumerate(GEOM_Z):
                row[d] = (row[d] - mean[i]) / std[i]


def _model(dim, hidden=32):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def _train(train, dim, *, hidden, epochs, lr, seed, device):
    torch.manual_seed(seed)
    m = _model(dim, hidden).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss()
    tens = [(torch.tensor(X, dtype=torch.float32, device=device),
             torch.tensor([y], device=device)) for X, y in train]
    rng = random.Random(seed)
    for _ in range(epochs):
        m.train()
        rng.shuffle(tens)
        for X, y in tens:
            opt.zero_grad()
            logits = m(X).squeeze(-1).unsqueeze(0)  # [1, n_cand]
            ce(logits, y).backward()
            opt.step()
    return m


def _logits_labels(m, events, device):
    out = []
    m.eval()
    with torch.no_grad():
        for X, y in events:
            t = torch.tensor(X, dtype=torch.float32, device=device)
            out.append((m(t).squeeze(-1), y))
    return out


def _bits_acc(ll, temp=1.0):
    """Mean CE in bits (codec rate) at temperature `temp`, plus accuracy."""
    bits, correct = [], 0
    for logits, y in ll:
        logp = torch.log_softmax(logits / temp, dim=0)
        bits.append(-logp[y].item() / math.log(2.0))
        correct += int(logits.argmax().item() == y)
    return (float(np.mean(bits)) if bits else float("nan"),
            (correct / len(ll) if ll else float("nan")))


def _fit_temp(ll):
    """Scalar temperature minimizing CE on a held-out calib split (overconfidence
    fix — the §18.0 calibration). Returns the best T."""
    best_t, best_b = 1.0, float("inf")
    for t in np.geomspace(0.3, 30.0, 60):
        b, _ = _bits_acc(ll, temp=float(t))
        if b < best_b:
            best_t, best_b = float(t), b
    return best_t


def main():
    ap = argparse.ArgumentParser(description="§18.1 bits g_t buys")
    ap.add_argument("--data", default="results/sprint18/filter_dataset.jsonl")
    ap.add_argument("--out", default="results/sprint18/filter_bits.json")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=5, help="models averaged per arm (denoise)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, vocab = _load(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split BY SCENE (both mode-rows of a scene stay together): train / calib / val.
    # calib is held out to fit the temperature (overconfidence fix); val is honest.
    scenes = sorted({r["scene_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * args.val_frac))
    n_cal = max(1, int(len(scenes) * args.calib_frac))
    val_s = set(scenes[:n_val])
    cal_s = set(scenes[n_val:n_val + n_cal])
    tr_rows = [r for r in rows if r["scene_id"] not in val_s and r["scene_id"] not in cal_s]
    cal_rows = [r for r in rows if r["scene_id"] in cal_s]
    va_rows = [r for r in rows if r["scene_id"] in val_s]
    va_flip = [r for r in va_rows if r.get("scene_flipped")]
    n_flip_scenes = len({r["scene_id"] for r in rows if r.get("scene_flipped")})

    report = {"data": args.data, "n_rows": len(rows), "n_scenes": len(scenes),
              "n_flipped_scenes": n_flip_scenes, "n_train": len(tr_rows),
              "n_calib": len(cal_rows), "n_val": len(va_rows), "n_val_flip": len(va_flip),
              "seeds": args.seeds, "species": list(vocab), "arms": {}}

    for arm in ARMS:
        tr = _build(tr_rows, arm, vocab)
        cal = _build(cal_rows, arm, vocab)
        va = _build(va_rows, arm, vocab)
        vaf = _build(va_flip, arm, vocab)
        mean, std = _zscore_fit(tr)
        for ev in (tr, cal, va, vaf):
            _zscore_apply(ev, mean, std)
        dim = len(tr[0][0][0])
        vb, vacc, fb, facc, temps = [], [], [], [], []
        for s in range(args.seeds):
            m = _train(tr, dim, hidden=args.hidden, epochs=args.epochs,
                       lr=args.lr, seed=args.seed + s, device=device)
            t = _fit_temp(_logits_labels(m, cal, device))
            b, acc = _bits_acc(_logits_labels(m, va, device), temp=t)
            fbits, faccs = _bits_acc(_logits_labels(m, vaf, device), temp=t)
            vb.append(b); vacc.append(acc); fb.append(fbits); facc.append(faccs); temps.append(t)
        report["arms"][arm] = {
            "dim": dim, "temp_mean": float(np.mean(temps)),
            "val_bits": float(np.mean(vb)), "val_bits_std": float(np.std(vb)),
            "val_acc": float(np.mean(vacc)),
            "flip_bits": float(np.mean(fb)), "flip_acc": float(np.mean(facc))}

    a = report["arms"]
    gap = a["geom+type"]["val_bits"] - a["geom+type+policy"]["val_bits"]
    gap_flip = a["geom+type"]["flip_bits"] - a["geom+type+policy"]["flip_bits"]
    report["bits_gt_buys_val"] = gap
    report["bits_gt_buys_flip"] = gap_flip
    report["structural_mi_bits"] = _structural_mi(rows)
    report["flip_rate"] = n_flip_scenes / len(scenes) if scenes else float("nan")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print("=" * 72)
    print("§18.1 — bits g_t buys (mode-blind vs mode-aware interact-target prior)")
    print("=" * 72)
    print(f"rows={len(rows)} scenes={len(scenes)} flipped_scenes={n_flip_scenes}  "
          f"train/calib/val rows={report['n_train']}/{report['n_calib']}/{report['n_val']} "
          f"(val_flip={report['n_val_flip']})  seeds={args.seeds}")
    print(f"\n{'arm':>22} {'val_bits':>9} {'±std':>6} {'val_acc':>8} {'flip_bits':>10} "
          f"{'flip_acc':>9} {'T':>5}")
    for arm in ARMS:
        x = a[arm]
        print(f"{arm:>22} {x['val_bits']:>9.3f} {x['val_bits_std']:>6.3f} {x['val_acc']:>8.3f} "
              f"{x['flip_bits']:>10.3f} {x['flip_acc']:>9.3f} {x['temp_mean']:>5.2f}")
    print(f"\nBITS g_t BUYS:")
    print(f"  STRUCTURAL  I(target;g_t|scene) = {report['structural_mi_bits']:.3f} "
          f"bits/interact   (flip_rate={report['flip_rate']:.2f}; training-free headline)")
    print(f"  learned codec (geom+type − geom+type+policy):")
    print(f"     all val:  {gap:+.3f}   flipped:  {gap_flip:+.3f}   (corroboration, data-limited)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
