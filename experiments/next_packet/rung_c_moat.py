"""rung C — intent half-life / moat-width (neural_interface.md §12.3).

Framing (a), within-rollout segment recovery:
  Per rollout, the active intent g_t is re-issued each LLM turn. A *segment* is a
  maximal run of packets sharing one g_t string. We train a per-rollout classifier
  to recover WHICH segment is active from the embodied obs alone, then plot decode
  accuracy as a function of `ticks_since_g_t_issued`. The decay (or lack of it) is
  the moat width: how long the planner's command stays legible in the body's
  fast-loop stream before its dynamics wash it out.

Key design choices
  * Features are EMBODIED state only — kinematics, velocity, stats, inventory, and
    the wire packet-type. `g_t` is the label; `current_tool` is EXCLUDED by default
    (it is the control-stack's own intent label, ~1:1 with the segment, so including
    it would trivialize "is intent legible in the *body*"). `--with-tool` adds it as
    an ablation contrast.
  * Classes are per-rollout segment indices, so each rollout gets its own model.
    The decay curve pools test packets across rollouts, binned by ticks-since-issued.
  * Two splits, both reported:
      random  — stratified-by-segment random holdout (upper bound; leaks via temporal
                autocorrelation but the curve *shape* is still informative).
      block   — hold out the last `--block-frac` of each segment by tick (the honest
                generalization test; no adjacent-packet leakage).

Classifier: a torch multinomial-logistic head (linear softmax) per rollout, trained
full-batch with Adam. torch is the available numeric stack here (sklearn absent);
GPU is used when present.

Usage:
  .venv/bin/python -m experiments.next_packet.rung_c_moat \
      --data results/frozen_narrated --seed 0 --bins 12 [--with-tool] [--split block]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- IO
def load_rollout(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def segments(rows: list[dict]) -> list[int]:
    """Assign a 0-based segment id per packet: a new segment starts when g_t changes.

    Rows are assumed tick-sorted (verified: packets.jsonl is monotonic in tick).
    """
    seg = []
    sid = -1
    prev = object()
    for r in rows:
        g = r["obs"].get("g_t")
        if g != prev:
            sid += 1
            prev = g
        seg.append(sid)
    return seg


# ----------------------------------------------------------------- featurization
STAT_KEYS = [
    "health", "food", "saturation", "air", "armor", "xp_level",
    "in_water", "in_lava", "on_fire",
]
STAT_NORM = {"health": 20.0, "food": 20.0, "saturation": 20.0, "air": 300.0,
             "armor": 20.0}


def build_item_vocab(all_rows: list[list[dict]], top_k: int = 40) -> list[str]:
    c: Counter = Counter()
    for rows in all_rows:
        for r in rows:
            inv = r["obs"].get("inventory") or {}
            for it in (inv.get("main") or []):
                if it and it.get("id"):
                    c[it["id"]] += 1
    return [k for k, _ in c.most_common(top_k)]


def build_pkt_vocab(all_rows: list[list[dict]]) -> list[str]:
    c: Counter = Counter()
    for rows in all_rows:
        for r in rows:
            c[r.get("id", "?")] += 1
    return [k for k, _ in c.most_common()]


def featurize(rows: list[dict], item_vocab: list[str], pkt_vocab: list[str],
              tool_vocab: list[str] | None) -> np.ndarray:
    item_ix = {k: i for i, k in enumerate(item_vocab)}
    pkt_ix = {k: i for i, k in enumerate(pkt_vocab)}
    tool_ix = {k: i for i, k in enumerate(tool_vocab)} if tool_vocab else {}
    feats = []
    prev_xyz = None
    for r in rows:
        o = r["obs"]
        x, y, z = o.get("x", 0.0), o.get("y", 0.0), o.get("z", 0.0)
        yaw = math.radians(o.get("yaw", 0.0) or 0.0)
        pitch = math.radians(o.get("pitch", 0.0) or 0.0)
        row = [x, y, z,
               math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
               1.0 if o.get("on_ground") else 0.0]
        # velocity (Δ since previous packet in this rollout)
        if prev_xyz is None:
            row += [0.0, 0.0, 0.0]
        else:
            row += [x - prev_xyz[0], y - prev_xyz[1], z - prev_xyz[2]]
        prev_xyz = (x, y, z)
        # stats
        st = o.get("stats") or {}
        for k in STAT_KEYS:
            v = st.get(k, 0)
            if isinstance(v, bool):
                row.append(1.0 if v else 0.0)
            elif k == "xp_level":
                row.append(math.log1p(float(v or 0)))
            else:
                row.append(float(v or 0) / STAT_NORM.get(k, 1.0))
        # inventory: log1p counts over item vocab + total
        invvec = [0.0] * len(item_vocab)
        inv = o.get("inventory") or {}
        total = 0
        for it in (inv.get("main") or []):
            if it and it.get("id") in item_ix:
                invvec[item_ix[it["id"]]] += it.get("count", 0)
            if it:
                total += it.get("count", 0)
        row += [math.log1p(v) for v in invvec]
        row.append(math.log1p(total))
        # wire packet type one-hot
        pkt = [0.0] * len(pkt_vocab)
        if r.get("id") in pkt_ix:
            pkt[pkt_ix[r["id"]]] = 1.0
        row += pkt
        # optional: current_tool one-hot (ablation)
        if tool_vocab is not None:
            tv = [0.0] * len(tool_vocab)
            ct = o.get("current_tool")
            if ct in tool_ix:
                tv[tool_ix[ct]] = 1.0
            row += tv
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def zscore(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    out = [(train - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out


# ------------------------------------------------------------------- classifier
def fit_predict(Xtr, ytr, Xte, nseg, seed, *, epochs=300, lr=0.05, wd=1e-4):
    """Multinomial-logistic head (linear softmax), full-batch Adam. Returns the
    argmax class for each test row as a numpy array."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    model = nn.Linear(Xtr.shape[1], nseg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=1).cpu().numpy()
    return pred


# ------------------------------------------------------------------- experiment
def split_indices(seg: np.ndarray, tsi: np.ndarray, mode: str, block_frac: float,
                  seed: int):
    """Return (train_idx, test_idx). Stratified by segment."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        if len(idx) < 4:
            train.extend(idx)  # too small to hold out; keep in train
            continue
        if mode == "random":
            perm = rng.permutation(idx)
            k = max(1, int(round(0.3 * len(idx))))
            test.extend(perm[:k]); train.extend(perm[k:])
        else:  # block: hold out the latest block_frac by ticks-since-issued
            order = idx[np.argsort(tsi[idx])]
            k = max(1, int(round(block_frac * len(idx))))
            test.extend(order[-k:]); train.extend(order[:-k])
    return np.asarray(train), np.asarray(test)


def run(args):
    dirs = sorted(glob.glob(os.path.join(args.data, "rollout-*/")))
    all_rows = [load_rollout(os.path.join(d, "packets.jsonl")) for d in dirs]
    item_vocab = build_item_vocab(all_rows, top_k=args.items)
    pkt_vocab = build_pkt_vocab(all_rows)
    tool_vocab = None
    if args.with_tool:
        tc: Counter = Counter()
        for rows in all_rows:
            for r in rows:
                tc[r["obs"].get("current_tool")] += 1
        tool_vocab = [k for k in tc if k is not None]

    bin_correct = defaultdict(int)
    bin_total = defaultdict(int)
    bin_chance = defaultdict(list)
    per_rollout = []

    all_tsi = np.concatenate([
        np.array([r["obs"].get("ticks_since_g_t_issued", 0) or 0 for r in rows])
        for rows in all_rows])
    tmax = float(np.percentile(all_tsi, 99))  # clip tail
    edges = np.linspace(0, tmax, args.bins + 1)

    def binof(t):
        b = int(np.searchsorted(edges, t, side="right") - 1)
        return min(max(b, 0), args.bins - 1)

    for d, rows in zip(dirs, all_rows):
        seg = np.asarray(segments(rows))
        tsi = np.asarray([r["obs"].get("ticks_since_g_t_issued", 0) or 0 for r in rows],
                         dtype=np.float64)
        nseg = int(seg.max()) + 1
        X = featurize(rows, item_vocab, pkt_vocab, tool_vocab)
        tr, te = split_indices(seg, tsi, args.split, args.block_frac, args.seed)
        Xtr, Xte = zscore(X[tr], X[te])
        yhat = fit_predict(Xtr, seg[tr], Xte, nseg, args.seed,
                           epochs=args.epochs, lr=args.lr)
        acc = float((yhat == seg[te]).mean())
        per_rollout.append((os.path.basename(d.rstrip("/")), nseg, len(te), acc))
        for i, gi in enumerate(te):
            b = binof(tsi[gi])
            bin_total[b] += 1
            bin_correct[b] += int(yhat[i] == seg[gi])
            bin_chance[b].append(1.0 / nseg)

    # ---- report
    print("=== rung C — moat width (within-rollout segment recovery) ===")
    print(f"split={args.split} with_tool={args.with_tool} seed={args.seed} "
          f"epochs={args.epochs} cuda={torch.cuda.is_available()}")
    for name, nseg, nte, acc in per_rollout:
        print(f"  {name}: nseg={nseg} test={nte} acc={acc:.3f} chance={1/nseg:.3f}")
    overall = sum(bin_correct.values()) / max(1, sum(bin_total.values()))
    print(f"  OVERALL test acc={overall:.3f}")
    print("--- decay curve: ticks_since_g_t_issued bin -> acc (vs chance) ---")
    rows_out = []
    for b in range(args.bins):
        if bin_total[b] == 0:
            continue
        lo, hi = edges[b], edges[b + 1]
        acc = bin_correct[b] / bin_total[b]
        ch = float(np.mean(bin_chance[b])) if bin_chance[b] else float("nan")
        lift = acc - ch
        print(f"  [{lo:6.0f},{hi:6.0f})  n={bin_total[b]:5d}  acc={acc:.3f}  "
              f"chance={ch:.3f}  lift={lift:+.3f}")
        rows_out.append((lo, hi, bin_total[b], acc, ch, lift))

    if args.out:
        with open(args.out, "w") as f:
            f.write("lo,hi,n,acc,chance,lift\n")
            for lo, hi, n, acc, ch, lift in rows_out:
                f.write(f"{lo:.1f},{hi:.1f},{n},{acc:.4f},{ch:.4f},{lift:.4f}\n")
        print(f"[wrote curve -> {args.out}]")

    # one-line read
    if len(rows_out) >= 3:
        q = max(1, len(rows_out) // 4)
        first = float(np.mean([r[5] for r in rows_out[:q]]))
        last = float(np.mean([r[5] for r in rows_out[-q:]]))
        trend = "DECAYS" if last < first - 0.05 else (
            "RISES" if last > first + 0.05 else "FLAT")
        print(f"READ: intent legibility lift early={first:+.3f} late={last:+.3f} "
              f"-> {trend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/frozen_narrated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--items", type=int, default=40, help="inventory vocab size")
    ap.add_argument("--split", choices=["random", "block"], default="random")
    ap.add_argument("--block-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--with-tool", action="store_true",
                    help="ablation: add current_tool one-hot to features")
    ap.add_argument("--out", default=None, help="CSV path for the decay curve")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
