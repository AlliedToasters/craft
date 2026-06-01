#!/usr/bin/env python3
"""§22 Rung 2 — is the recompute residual PREDICTABLE from plan-state?

Rung 1 located the residual: the path-stream is ≈free within a commit-run (0.996
coverage) and pays only at RECOMPUTES (0.43%/tick), of which 94% are benign
extensions straight at the goal and only ~6% are detours. Rung 2 asks the hinge
question — can a decoder anticipate the recompute from plan-state ALONE (no terrain,
no re-running A*)? Three parts, each answering a different sub-residual:

  PART A — TIMING (does data; 66k ticks). A classifier predicts "a recompute occurs
    within the next K ticks" from causal plan-state (nodes-ahead, idx/len, run-age,
    Baritone's own ticks_to_goal ETA, distances). By-rollout split, AUC vs a
    nodes-ahead-only baseline vs chance. Tests whether the recompute EVENT is a
    predictable clock/threshold — Rung-1 probes said NOT cleanly (recompute fires at
    ~27 nodes still ahead, bursty, run-length CV ~0.6-1.3), so we expect modest AUC:
    the event is weakly anticipated, not free.

  PART B — CONTENT (entropy decomposition; data-starved on detours, stated honestly).
    The new segment's heading deviation from the goal bearing (16-way, the §21-locked
    representation). marginal H(dev) [predict straight-at-goal] vs predictive-coding
    H(turn) [predict continue-old-heading] vs conditional H(dev | coarse plan-state).
    Rung-1 probe: predictive-coding is WORSE (1.46 > 1.33 b) — recomputes re-aim at the
    goal — and the 14 detours are ~6 episodes (9 in one rollout), so detour DIRECTION
    is un-learnable here AND is terrain-caused (the §21 negative). Content direction is
    not in plan-state.

  PART C — THE VERDICT + the A*-decoder framing. The recompute is the planner's
    deterministic-given-goal output. Per the spine's thesis (DON'T relearn A* — §20),
    the right codec lets Baritone regenerate segments and transmits only the GOAL.
    So the true residual collapses to the GOAL/intent stream (operator re-commands):
    74 goal-changes / 66k ticks = 0.11%/tick — 4× fewer events than recomputes, and
    each is a real decision (the §19/§20.1a override surface), not a planner artifact.
    The residual ladder: raw sector/tick → recompute-marginal (Rung 1) → goal-only.
    (A* determinism — that a re-run from captured (pos,goal) reproduces the segment, so
    recomputes truly cost 0 — is the LIVE Phase B check, flagged not claimed.)

Resolution of the §21 negative: you can't predict the detour from perception (§21) OR
from plan-state (§22 Rung 2) — so don't. Transmit the goal; let A* regenerate it.

Usage:
    .venv/bin/python -m experiments.codec_loop.path_codec_rung2 \
        --capture results/sprint21_visual/capture --out results/sprint22/rung2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.codec_loop.path_codec import (
    load_rollouts, _segment, _bearing, _dev_class, _sectors_off, _entropy_bits,
    N_SECTORS,
)

K_AHEAD = 10                      # a recompute "soon" = within this many ticks


# --- feature extraction ------------------------------------------------------
def _hypot_xz(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _tick_features(t, run_age):
    """Causal plan-state at a tick — everything a stream decoder already knows, no
    terrain, no A* re-run. run_age = ticks since the last recompute (the decoder saw it)."""
    plen = max(t["plen"], 1)
    ahead = t["plen"] - t["idx"]
    ttg = t["ttg"]
    ttg = float(ttg) if ttg is not None else -1.0
    return [
        float(ahead),                              # committed nodes still ahead
        t["idx"] / plen,                           # fraction consumed
        float(plen),                               # segment length
        float(run_age),                            # ticks since last recompute
        ttg,                                       # Baritone's own ETA (−1 if absent)
        _hypot_xz(t["origin"], t["dest"]),         # distance to segment end
        _hypot_xz(t["origin"], t["goal"]),         # distance to far goal
        _hypot_xz(t["dest"], t["goal"]),           # how far the segment falls short
    ]


FEAT_NAMES = ["nodes_ahead", "frac_consumed", "plen", "run_age", "ttg",
              "dist_to_dest", "dist_to_goal", "dest_short_of_goal"]


def build_timing_samples(rollouts):
    """Per rollout: (X features, y recompute-within-K, ahead-only column). A tick is a
    positive if a SEGMENT recompute (same goal) starts within the next K ticks. Ticks in
    the last K of a rollout are dropped (label undefined)."""
    per_rollout = []
    for name, ticks in rollouts:
        runs, _ = _segment(ticks)
        # recompute boundary tick-indices (same-goal segment re-extensions only)
        rc_idx = set()
        for k in range(1, len(runs)):
            s, _ = runs[k]
            if ticks[s]["goal"] == ticks[s - 1]["goal"]:
                rc_idx.add(s)
        rc_sorted = sorted(rc_idx)
        # run_age per tick
        last_rc = 0
        X, y = [], []
        n = len(ticks)
        for i in range(n - K_AHEAD):
            if i in rc_idx:
                last_rc = i
            feat = _tick_features(ticks[i], i - last_rc)
            # positive if any recompute in (i, i+K]
            soon = any(i < r <= i + K_AHEAD for r in rc_sorted
                       if r <= i + K_AHEAD and r > i)
            X.append(feat)
            y.append(1 if soon else 0)
        if X and any(y):
            per_rollout.append((name, np.asarray(X, np.float32), np.asarray(y, np.float32)))
    return per_rollout


# --- AUC (rank-based, no sklearn) --------------------------------------------
def auc_score(scores, labels):
    """Mann-Whitney AUC with tie-averaged ranks (no sklearn dependency)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    n_p, n_n = int((y == 1).sum()), int((y == 0).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    avg_rank = starts + (counts + 1) / 2.0        # 1-based average rank per unique value
    ranks = avg_rank[inv]
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


# --- model -------------------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_timing(per_rollout, device, *, epochs=40, seed=0, ahead_only=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(per_rollout)
    n_test = max(1, n // 3)
    test_names = {per_rollout[i][0] for i in range(n - n_test, n)}
    cols = [0] if ahead_only else list(range(len(FEAT_NAMES)))

    Xtr = np.concatenate([X[:, cols] for nm, X, y in per_rollout if nm not in test_names])
    ytr = np.concatenate([y for nm, X, y in per_rollout if nm not in test_names])
    Xte = np.concatenate([X[:, cols] for nm, X, y in per_rollout if nm in test_names])
    yte = np.concatenate([y for nm, X, y in per_rollout if nm in test_names])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)

    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device,
                         dtype=torch.float32)
    model = Tiny(len(cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bs = 8192
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t), device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xte_t)).cpu().numpy()
    return {
        "auc": auc_score(scores, yte),
        "base_rate": float(yte.mean()),
        "n_test": int(len(yte)),
        "n_train": int(len(ytr)),
    }


# --- content entropy ---------------------------------------------------------
def content_decomposition(rollouts):
    """Entropy (bits) of the new segment heading at recomputes under three predictors,
    plus the detour-episode breakdown (data-starvation made explicit)."""
    devs, turns = [], []
    feats_for_cond = []          # (dev, coarse nodes_ahead bin, coarse frac bin)
    det_by_rollout = {}
    for name, ticks in rollouts:
        runs, _ = _segment(ticks)
        for k in range(1, len(runs)):
            s, _ = runs[k]
            cur, prev = ticks[s], ticks[s - 1]
            if cur["goal"] != prev["goal"]:
                continue
            ga = _bearing(cur["origin"], cur["goal"])
            nh = _bearing(cur["origin"], cur["dest"])
            oh = _bearing(prev["origin"], prev["dest"])
            if ga is None or nh is None:
                continue
            dev = _dev_class(nh, ga)
            devs.append(dev)
            if oh is not None:
                turns.append(_dev_class(nh, oh))
            ahead_bin = min(int((prev["plen"] - prev["idx"]) // 10), 4)
            feats_for_cond.append((dev, ahead_bin))
            if _sectors_off(dev) > 1:
                det_by_rollout[name] = det_by_rollout.get(name, 0) + 1

    h_marginal = _entropy_bits(devs)
    h_turn = _entropy_bits(turns)
    # conditional H(dev | coarse nodes-ahead bin): weighted average of per-bin entropies
    bins = {}
    for dev, ab in feats_for_cond:
        bins.setdefault(ab, []).append(dev)
    tot = sum(len(v) for v in bins.values())
    h_cond = sum(len(v) / tot * _entropy_bits(v) for v in bins.values()) if tot else float("nan")
    return {
        "n_recompute": len(devs),
        "h_marginal_goal_anchor_bits": h_marginal,
        "h_predictive_coding_bits": h_turn,
        "h_conditional_planstate_bits": h_cond,
        "detour_episodes_by_rollout": det_by_rollout,
        "n_detour": int(sum(det_by_rollout.values())),
        "n_detour_rollouts": len(det_by_rollout),
    }


# --- driver ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="§22 Rung 2 — recompute predictability")
    ap.add_argument("--capture", default="results/sprint21_visual/capture")
    ap.add_argument("--out", default="results/sprint22/rung2.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rollouts = load_rollouts(Path(args.capture))
    total_ticks = sum(len(t) for _, t in rollouts)
    print(f"[rung2] {len(rollouts)} rollouts, {total_ticks} pathing ticks, device={device}\n")

    # PART A — timing
    per_rollout = build_timing_samples(rollouts)
    timing_full = train_timing(per_rollout, device, epochs=args.epochs, seed=args.seed)
    timing_ahead = train_timing(per_rollout, device, epochs=args.epochs, seed=args.seed,
                                ahead_only=True)

    # PART B — content
    content = content_decomposition(rollouts)

    # PART C — goal / intent stream (the true residual under a Baritone-A* decoder)
    goal_changes = 0
    for _, ticks in rollouts:
        _, gc = _segment(ticks)
        goal_changes += gc
    # recompute rate from rung-1 logic
    recomputes = content["n_recompute"]
    goal_rate = goal_changes / total_ticks
    rc_rate = recomputes / total_ticks
    ladder = {
        "raw_sector_bits_per_tick": math.log2(N_SECTORS),
        "recompute_marginal_bits_per_tick": content["h_marginal_goal_anchor_bits"] * rc_rate,
        "goal_event_rate_per_tick": goal_rate,
        "recompute_event_rate_per_tick": rc_rate,
        "goal_vs_recompute_event_ratio": rc_rate / goal_rate if goal_rate else float("inf"),
    }

    out = {
        "capture": args.capture, "total_ticks": total_ticks,
        "n_rollouts": len(rollouts), "K_ahead": K_AHEAD, "device": device,
        "timing": {"full": timing_full, "nodes_ahead_only": timing_ahead},
        "content": content,
        "goal_residual": {"goal_changes": goal_changes, **ladder},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    tf, ta = timing_full, timing_ahead
    c = content
    print("=== §22 RUNG 2 — is the recompute predictable from plan-state? ===")
    print(f"PART A  TIMING (recompute within {K_AHEAD} ticks, base rate {tf['base_rate']*100:.1f}%)")
    print(f"   AUC plan-state-full = {tf['auc']:.3f}  |  nodes-ahead-only = {ta['auc']:.3f}  "
          f"(0.5=chance)  [n_test={tf['n_test']}]")
    print(f"   -> the recompute EVENT is {'weakly' if tf['auc']<0.75 else 'fairly'} anticipated, "
          f"not a clean clock/threshold")
    print(f"PART B  CONTENT (new heading deviation at recompute, {c['n_recompute']} events)")
    print(f"   H(dev | goal-anchor)      = {c['h_marginal_goal_anchor_bits']:.3f} b   (predict straight-at-goal)")
    print(f"   H(turn | predictive-code) = {c['h_predictive_coding_bits']:.3f} b   "
          f"({'WORSE' if c['h_predictive_coding_bits']>c['h_marginal_goal_anchor_bits'] else 'better'} "
          f"— recomputes re-aim at goal, not continue)")
    print(f"   H(dev | plan-state bin)   = {c['h_conditional_planstate_bits']:.3f} b   "
          f"(Δ vs marginal = {c['h_conditional_planstate_bits']-c['h_marginal_goal_anchor_bits']:+.3f} b)")
    print(f"   DETOURS data-starved: {c['n_detour']} events across {c['n_detour_rollouts']} rollouts "
          f"{c['detour_episodes_by_rollout']} -> direction un-learnable + terrain-caused (§21)")
    print(f"PART C  VERDICT — residual collapses to the GOAL/intent stream (Baritone decodes)")
    print(f"   recompute events {rc_rate*100:.2f}%/tick  ->  goal events {goal_rate*100:.2f}%/tick "
          f"({ladder['goal_vs_recompute_event_ratio']:.1f}× fewer; each a real decision, not a planner artifact)")
    print(f"   ladder bits/tick: raw {ladder['raw_sector_bits_per_tick']:.0f}  ->  "
          f"recompute-marginal {ladder['recompute_marginal_bits_per_tick']:.4f}  ->  "
          f"goal-only event-rate {goal_rate:.5f}")
    print(f"   (A* determinism = recomputes cost 0 under a re-running decoder: LIVE Phase B check)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
