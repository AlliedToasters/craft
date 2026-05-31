#!/usr/bin/env python3
"""§21.1 analysis — the BEARING-PRECISION KNEE (neural_interface.md §21).

§21.0 pinned two corners of the (terrain × bearing) square: `full` (terrain +
exact bearing) and `bearing_only` (bearing, terrain ablated). §21.1 fills the
bottom-left cell — drop / coarsen the GOAL SIGNAL and ask how much of the local
plan survives. The north-star reading: is the bearing SCENE-INFERABLE from local
structured terrain, or must perception (§21.2 vision) supply it?

MECHANISM (reuses the §21.0 capture verbatim — pure re-analysis, no recapture):
the §21.0 head sees terrain in a frame ROTATED so +forward points at the goal,
and predicts the subgoal as a DEVIATION from the true straight-line bearing. We
ablate by aligning that frame to the bearing QUANTIZED to k sectors, while the
TARGET stays the deviation from the TRUE bearing. The quantization error
θ_true−θ_q ∈ [−π/k, π/k] is unknown to the head → it is exactly the irreducible
noise of "knowing the goal direction only to resolution 2π/k". So:

    k = ∞ (exact)  → §21.0 `full`
    k = 8 (45°), 4 (90°), 2 (180°)  → the graded knee
    k = 1          → no alignment at all = the FULL bearing ablation; the world
                     frame carries no consistent relation to a goal-relative
                     deviation, so terrain collapses to the straight-line prior.

The gvec [log-dist, beyond-window] is bearing-FREE (§21.0 dropped sin/cos into
the rotation) so it stays constant across all arms — we ablate only the
direction precision, nothing else.

READINGS:
  * detour-subset recovery vs k  — the headline knee: how precise must the goal
    direction be before local-planning detours stop being recoverable.
  * (full_exact − k=1) on the detour subset = the part of the local plan that
    is ONLY accessible via the bearing = NOT scene-inferable from local terrain
    = the job handed to §21.2 vision.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_bearing_ablation \
        --capture results/sprint21/capture --feat-r 6 --target-r 5 \
        --seeds 0 1 --out results/sprint21/bearing_ablation.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.codec_loop.nav_horizon import (
    N_SECTORS,
    Head,
    _aligned_terrain,
    _circ_sector_correct,
    _column_maps,
    _exit_dev,
    _global_vec,
    _goal_angle,
    load_samples,
)


def _quantize_angle(ang: float, k: int) -> float:
    """Snap a bearing to one of k evenly-spaced directions. k>=1; k==1 collapses
    every bearing to 0 (world-north frame = no alignment = full ablation)."""
    if k <= 1:
        return 0.0
    step = 2 * math.pi / k
    return round(ang / step) * step


def train_eval_bearing(rollouts, feat_r, target_r, bearing_k, *, terrain=True,
                       epochs=60, lr=1e-3, seed=0, device):
    """One ablation cell. Identical to nav_horizon.train_eval_r except the terrain
    frame is aligned to the bearing QUANTIZED to `bearing_k` sectors, while the
    target deviation stays defined against the TRUE bearing. bearing_k=None → exact
    (the §21.0 `full`). terrain=False → bearing_only (terrain ablated)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(rollouts)
    n_test = max(1, n // 3)
    test_names = {rollouts[i][0] for i in range(n - n_test, n)}
    straight_class = N_SECTORS // 2

    def build(split_rows):
        X, Ydev, Ydy = [], [], []
        for d, origin, fwd, dest in split_rows:
            goal_ang = _goal_angle(dest, origin)
            if goal_ang is None:
                continue
            tgt = _exit_dev(fwd, origin, target_r, goal_ang)   # TRUE bearing
            if tgt is None:
                continue
            devc, dyc = tgt
            gvec = _global_vec(dest, origin, target_r)
            if terrain:
                solid, water = _column_maps(d)
                align_ang = goal_ang if bearing_k is None else _quantize_angle(goal_ang, bearing_k)
                tvec = _aligned_terrain(solid, water, feat_r, align_ang)   # QUANTIZED frame
                feat = np.concatenate([tvec, np.asarray(gvec, np.float32)])
            else:
                feat = np.asarray(gvec, np.float32)
            X.append(feat)
            Ydev.append(devc)
            Ydy.append(dyc)
        if not X:
            return None
        Bsec = torch.full((len(X),), straight_class, dtype=torch.long)
        return (torch.tensor(np.stack(X), dtype=torch.float32),
                torch.tensor(Ydev), torch.tensor(Ydy), Bsec)

    train_rows = [row for name, rows in rollouts if name not in test_names for row in rows]
    test_rows = [row for name, rows in rollouts if name in test_names for row in rows]
    tr = build(train_rows)
    te = build(test_rows)
    if tr is None or te is None:
        return None
    Xtr, Ytr_s, Ytr_d, _ = tr
    Xte, Yte_s, Yte_d, Bte_s = te
    Xtr, Ytr_s, Ytr_d = Xtr.to(device), Ytr_s.to(device), Ytr_d.to(device)
    Xte, Yte_s, Yte_d = Xte.to(device), Yte_s.to(device), Yte_d.to(device)

    Bte_dev = Bte_s.to(device)
    detour_mask = (_circ_sector_correct(Bte_dev, Yte_s) == False)  # noqa: E712

    model = Head(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bs = 4096
    tail = max(1, epochs // 6)
    acc = {k: 0.0 for k in ("sector_exact", "sector_within1", "dy_acc",
                            "both_exact", "sector_ce_bits", "detour_within1")}
    seen = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            ps, pd = model(Xtr[idx])
            loss = ce(ps, Ytr_s[idx]) + ce(pd, Ytr_d[idx])
            loss.backward()
            opt.step()
        if ep < epochs - tail:
            continue
        model.eval()
        with torch.no_grad():
            ps, pd = model(Xte)
            sec_pred = ps.argmax(1)
            dy_pred = pd.argmax(1)
            hit1 = _circ_sector_correct(sec_pred, Yte_s)
            acc["sector_exact"] += (sec_pred == Yte_s).float().mean().item()
            acc["sector_within1"] += hit1.float().mean().item()
            acc["dy_acc"] += (dy_pred == Yte_d).float().mean().item()
            acc["both_exact"] += ((sec_pred == Yte_s) & (dy_pred == Yte_d)).float().mean().item()
            acc["sector_ce_bits"] += ce(ps, Yte_s).item() / math.log(2)
            acc["detour_within1"] += (hit1[detour_mask].float().mean().item()
                                      if detour_mask.any() else float("nan"))
        seen += 1
    best = {k: v / seen for k, v in acc.items()}
    best["straight_within1"] = _circ_sector_correct(Bte_s, Yte_s.cpu()).float().mean().item()
    best["detour_frac"] = detour_mask.float().mean().item()
    best["n_train"] = int(Xtr.shape[0])
    best["n_test"] = int(Xte.shape[0])
    best["n_detour"] = int(detour_mask.sum().item())
    best["feat_dim"] = int(Xtr.shape[1])
    return best


def _avg(dicts):
    """Mean across seeds of every numeric field (ignore NaNs in detour_within1).
    For the two headline metrics also keep the across-seed std so 'flat = within
    noise' is checkable rather than asserted."""
    out = {}
    keys = dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in dicts]
        finite = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        out[k] = (sum(finite) / len(finite)) if finite else float("nan")
    for k in ("detour_within1", "sector_within1"):
        finite = [d[k] for d in dicts
                  if isinstance(d.get(k), (int, float)) and not math.isnan(d[k])]
        if len(finite) > 1:
            m = sum(finite) / len(finite)
            out[k + "_std"] = (sum((v - m) ** 2 for v in finite) / (len(finite) - 1)) ** 0.5
        else:
            out[k + "_std"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="§21.1 bearing-precision knee")
    ap.add_argument("--capture", default="results/sprint21/capture")
    ap.add_argument("--feat-r", type=int, default=6,
                    help="terrain window radius (>= target-r so terrain isn't under-powered)")
    ap.add_argument("--target-r", type=int, default=5, help="fixed action radius")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 4, 2, 1],
                    help="bearing quantization sectors to sweep (exact + these)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/sprint21/bearing_ablation.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = Path(args.capture)
    rollouts = load_samples(cap)
    total = sum(len(r) for _, r in rollouts)
    print(f"[bearing_ablation] {len(rollouts)} rollouts, {total} rows, device={device}")
    if len(rollouts) < 3:
        print("[bearing_ablation] need >=3 rollouts for a held-out split")
        return 2
    print(f"[bearing_ablation] feat_r={args.feat_r} target_r={args.target_r} "
          f"seeds={args.seeds}\n")

    def run_arm(label, bearing_k, terrain):
        per_seed = []
        for s in args.seeds:
            r = train_eval_bearing(rollouts, args.feat_r, args.target_r, bearing_k,
                                    terrain=terrain, epochs=args.epochs, seed=s, device=device)
            if r is not None:
                per_seed.append(r)
        if not per_seed:
            return None
        a = _avg(per_seed)
        deg = "—" if bearing_k is None else (f"{360/bearing_k:.0f}°" if bearing_k > 1 else "none")
        print(f"  {label:14s} (prec={deg:>5s})  sec_±1={a['sector_within1']:.3f}"
              f"±{a['sector_within1_std']:.3f} ce={a['sector_ce_bits']:.2f}b | "
              f"DETOUR(frac={a['detour_frac']:.2f},n={a['n_detour']}): "
              f"{a['detour_within1']:.3f}±{a['detour_within1_std']:.3f}  "
              f"(straight={a['straight_within1']:.3f})", flush=True)
        a["seeds"] = list(args.seeds)
        return a

    arms = {}
    arms["full_exact"] = run_arm("full(exact)", None, True)
    for k in args.ks:
        arms[f"k{k}"] = run_arm(f"coarse k={k}", k, True)
    arms["bearing_only"] = run_arm("bearing_only", None, False)

    # the headline scene-inferability gap
    fe = arms.get("full_exact") or {}
    k1 = arms.get("k1") or {}
    gap = (fe.get("detour_within1", float("nan")) - k1.get("detour_within1", float("nan")))
    terrain_floor = k1.get("detour_within1", float("nan"))
    print("\n=== BEARING-PRECISION KNEE ===")
    print(f"  full(exact) detour±1   = {fe.get('detour_within1', float('nan')):.3f}")
    print(f"  no-bearing  detour±1   = {terrain_floor:.3f}  (k=1: terrain in world frame)")
    print(f"  bearing-dependent gap  = {gap:+.3f}")
    if abs(gap) < 0.03:
        print("     → FLAT: detour recovery is invariant to bearing precision. The bearing\n"
              "       only resolves aim-at-goal (the bearing-trivial majority); it adds nothing\n"
              "       to detours. The §21.0 residual is NOT a goal-signal problem — local terrain\n"
              "       lacks the detour cause. §21.2's job is richer TERRAIN, not a better bearing.")
    else:
        print("     = detours recoverable ONLY with the goal signal (the goal-precision job)")

    out = {"capture": str(cap), "n_rollouts": len(rollouts), "n_rows": total,
           "feat_r": args.feat_r, "target_r": args.target_r, "device": device,
           "ks": args.ks, "seeds": list(args.seeds), "arms": arms,
           "scene_inferability_gap": gap, "no_bearing_floor": terrain_floor}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
