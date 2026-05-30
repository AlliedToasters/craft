"""Rung C · transition-seam study (§13.2) — how fast does the embodied-decodable
intent flip from the OLD segment to the NEW one across a g_t boundary?

§12.3 measured the segment *interior* (flat-high → NO moat decay: intent stays
legible the full segment width). §7 predicts the LLM's value is *originating* goals,
not *sustaining* them, so any degradation should localize at *transitions*. The
number we want is the **handover latency = the moat width measured head-on**.

Reuses the validated §12.3 pipeline from `rung_c_moat` (`load_rollout`, `segments`,
`featurize` with `tool_vocab=None` → embodied-only features; no current_tool, no
delta_tick) so the inputs are identical to the no-decay result.

Instrument (13.2.2, honest). A per-rollout multinomial-logistic segment decoder (the
§12.3 classifier) trained on segment INTERIORS only (rows ≥ `holdout` ticks from
either of their own boundaries) and evaluated ON the held-out seam. At each seam row
we read `rel = p_new / (p_old + p_new)` and bin by the signed offset (tick − t0).

Offsets need no absolute tick: within the new segment offset = ticks_since_g_t_issued
(tsi); within the old segment offset = tsi − len(old) (negative). Each seam window is
restricted to the two adjacent segments, so the readout is a clean old-vs-new contest.

Crossover (13.2.1). The true label is a step at t0 (latency 0), so the crossover's
offset from 0 IS the handover latency. The current_tool label also flips at t0 (it is
set per LLM turn), so tool-switch − embodied-switch = crossover = the literal §1 rate
gap: symbolic intent flips instantly, the body's decodable state lags it.

13.2.3. Split boundaries by new-segment length (≥/< median): a real latency persists
for long new segments; a short-segment artifact vanishes once length is controlled —
this disentangles the §12.3 fresh-tick dip.

13.2.4. Data caveat: frozen_narrated is peaceful → transitions are mostly *completion*
(goal done → next), not *override* (interrupt). Override is the corrigibility-relevant
seam (§6); a non-peaceful recapture is a next-sprint input.

Run as a package module:
    .venv/bin/python -m experiments.next_packet.rung_c_transition \
        --data results/frozen_narrated --margin 60 --holdout 20 \
        --out results/rung_c_transition/crossover.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from .rung_c_moat import (
    build_item_vocab,
    build_pkt_vocab,
    featurize,
    load_rollout,
    segments,
)

TICKS_PER_SEC = 20.0


def tsi_of(rows):
    return np.asarray(
        [r["obs"].get("ticks_since_g_t_issued", 0) or 0 for r in rows],
        dtype=np.int64,
    )


def tool_of(rows):
    return [r["obs"].get("current_tool") for r in rows]


def fit_proba(X, y, nseg, *, seed, epochs, lr, wd, device):
    """Multinomial-logistic (§12.3 classifier). Returns (model, mu, sd, train_acc)."""
    torch.manual_seed(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    model = nn.Linear(Xn.shape[1], nseg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        ce(model(Xn), yt).backward()
        opt.step()
    with torch.no_grad():
        acc = float((model(Xn).argmax(1) == yt).float().mean().item())
    return model, mu, sd, acc


def proba(model, X, mu, sd, device):
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(device)
    with torch.no_grad():
        return torch.softmax(model(Xn), dim=1).cpu().numpy()


def _smooth(vals, k=1):
    out = []
    for i in range(len(vals)):
        win = [v for v in vals[max(0, i - k):i + k + 1] if v == v]
        out.append(float(np.mean(win)) if win else float("nan"))
    return out


def crossover(centers, rel):
    filled, last = list(rel), None
    for i in range(len(filled)):
        if filled[i] == filled[i]:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    sm = _smooth(filled, 1)
    for i in range(1, len(centers)):
        a, b = sm[i - 1], sm[i]
        if a != a or b != b:
            continue
        if (a < 0.5 <= b) or (a <= 0.5 < b):
            frac = (0.5 - a) / (b - a) if b != a else 0.0
            return float(centers[i - 1] + frac * (centers[i] - centers[i - 1]))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/frozen_narrated")
    ap.add_argument("--out", default="results/rung_c_transition/crossover.json")
    ap.add_argument("--margin", type=int, default=60, help="seam half-window (ticks)")
    ap.add_argument("--holdout", type=int, default=20,
                    help="ticks from a segment's own boundaries excluded from training")
    ap.add_argument("--min-train", type=int, default=20)
    ap.add_argument("--bin", type=int, default=5)
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.data, "rollout-*")))
    all_rows = [load_rollout(os.path.join(d, "packets.jsonl"))
                for d in dirs if os.path.exists(os.path.join(d, "packets.jsonl"))]
    all_rows = [r for r in all_rows if r]
    if not all_rows:
        print(f"no rollouts under {args.data}", file=sys.stderr)
        sys.exit(2)
    item_vocab = build_item_vocab(all_rows, top_k=args.items)
    pkt_vocab = build_pkt_vocab(all_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = list(range(-args.margin, args.margin + 1, args.bin))
    centers_arr = np.asarray(centers)

    # first pass: per-rollout features, seg, tsi, seg lengths, boundaries
    packs = []
    new_lens = []
    for rows in all_rows:
        seg = np.asarray(segments(rows))
        tsi = tsi_of(rows)
        tool = tool_of(rows)
        X = featurize(rows, item_vocab, pkt_vocab, None)  # embodied-only
        nseg = int(seg.max()) + 1
        seg_len = np.array([int(tsi[seg == s].max()) if (seg == s).any() else 0
                            for s in range(nseg)])
        packs.append(dict(seg=seg, tsi=tsi, tool=tool, X=X, nseg=nseg, seg_len=seg_len))
        for s in range(1, nseg):
            new_lens.append(int(seg_len[s]))
    med = float(np.median(new_lens)) if new_lens else 0.0

    def curve(which):
        acc = [[] for _ in centers]
        used = dropped = 0
        train_accs = []
        for p in packs:
            seg, tsi, X, nseg, seg_len = p["seg"], p["tsi"], p["X"], p["nseg"], p["seg_len"]
            # interior = far from BOTH of a row's own boundaries
            dist_start = tsi.astype(np.int64)               # ticks since this seg began
            dist_end = seg_len[seg] - tsi                   # ticks until this seg ends
            interior = (dist_start >= args.holdout) & (dist_end >= args.holdout)
            if interior.sum() < args.min_train or len(np.unique(seg[interior])) < 2:
                continue
            model, mu, sd, tacc = fit_proba(
                X[interior], seg[interior], nseg, seed=args.seed,
                epochs=args.epochs, lr=args.lr, wd=args.wd, device=device)
            train_accs.append(tacc)
            for s in range(1, nseg):
                old_s, new_s = s - 1, s
                if which == "long" and seg_len[new_s] < med:
                    continue
                if which == "short" and seg_len[new_s] >= med:
                    continue
                used += 1
                # seam rows from the two adjacent segments only
                old_mask = seg == old_s
                new_mask = seg == new_s
                old_off = tsi[old_mask] - seg_len[old_s]    # negative, →0 at boundary
                new_off = tsi[new_mask]                      # 0,1,2,... after boundary
                rows_X = np.concatenate([X[old_mask], X[new_mask]], axis=0)
                offs = np.concatenate([old_off, new_off])
                keep = np.abs(offs) <= args.margin
                if not keep.any():
                    continue
                sm = proba(model, rows_X[keep], mu, sd, device)
                for row, off in zip(sm, offs[keep]):
                    p_old, p_new = float(row[old_s]), float(row[new_s])
                    denom = p_old + p_new
                    if denom <= 0:
                        continue
                    bi = int(np.argmin(np.abs(centers_arr - off)))
                    acc[bi].append(p_new / denom)
        mean_rel = [float(np.mean(a)) if a else float("nan") for a in acc]
        n_per = [len(a) for a in acc]
        dec = float(np.mean(train_accs)) if train_accs else float("nan")
        return mean_rel, n_per, used, dropped, dec

    mean_rel, n_per, used, dropped, dec_acc = curve("all")
    xo = crossover(centers, mean_rel)
    mr_long, *_ = curve("long")
    mr_short, *_ = curve("short")
    xo_long, xo_short = crossover(centers, mr_long), crossover(centers, mr_short)

    # tool reference: fraction of boundaries where current_tool differs across the seam
    tool_changed = 0
    n_bnd = 0
    for p in packs:
        seg, tool = p["seg"], p["tool"]
        for s in range(1, p["nseg"]):
            n_bnd += 1
            old_t = next((tool[i] for i in range(len(seg)) if seg[i] == s - 1), None)
            new_t = next((tool[i] for i in range(len(seg)) if seg[i] == s), None)
            if old_t != new_t:
                tool_changed += 1

    result = {
        "n_rollouts": len(packs),
        "n_segments": int(sum(p["nseg"] for p in packs)),
        "n_boundaries": n_bnd,
        "margin_ticks": args.margin, "holdout_ticks": args.holdout,
        "bin_ticks": args.bin,
        "instrument": "per-rollout multiclass §12.3 decoder; rel=p_new/(p_old+p_new) "
                      "at seam; trained on interiors, evaluated on held-out seam",
        "decoder_interior_train_acc": dec_acc,
        "boundaries_used": used,
        "centers_ticks": centers, "mean_p_new": mean_rel, "n_per_bin": n_per,
        "crossover_tick": xo,
        "crossover_sec": (xo / TICKS_PER_SEC) if xo == xo else float("nan"),
        "new_len_median_ticks": med,
        "crossover_tick_long_new": xo_long,
        "crossover_tick_short_new": xo_short,
        "tool_changed_at_boundary": tool_changed,
        "tool_switch_offset_ticks": 0,  # tool set per LLM turn → flips at t0
        "rate_gap_ticks": xo,           # crossover − tool_switch(=0)
        "data_caveat": "peaceful frozen_narrated → mostly completion, not override, "
                       "transitions; override seam needs a non-peaceful recapture.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"rollouts={len(packs)} segments={result['n_segments']} "
          f"boundaries={n_bnd} used={used} decoder_interior_acc={dec_acc:.3f}")
    print(f"HANDOVER crossover={xo:.1f} ticks ({result['crossover_sec']:.2f}s)")
    print(f"long_new={xo_long:.1f}  short_new={xo_short:.1f}  (median new_len={med:.0f})")
    print(f"tool_changed_at_boundary={tool_changed}/{n_bnd} → rate_gap={xo:.1f} ticks")
    cur = " ".join(f"{centers[i]}:{mean_rel[i]:.2f}"
                   for i in range(0, len(centers), max(1, len(centers) // 8))
                   if mean_rel[i] == mean_rel[i])
    print("curve " + cur)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
