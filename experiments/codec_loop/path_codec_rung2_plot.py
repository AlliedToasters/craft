#!/usr/bin/env python3
"""§22 Rung 2 plot — is the recompute predictable, and what's the true residual?

Three panels from rung2.json:
  LEFT   — TIMING: AUC for "recompute within K ticks" — chance vs nodes-ahead-only vs
           full plan-state. The event is fairly anticipated (~0.90), a distributed
           plan-state signal (drop-any-one stays ≥0.88), well above the single-feature
           baseline. The WHEN is mostly free.
  MIDDLE — CONTENT: entropy (bits) of the new heading at recompute under three
           predictors. Goal-anchor (predict straight-at-goal) beats predictive-coding
           (recomputes re-aim at the goal); plan-state conditioning barely helps and
           the detour DIRECTION is data-starved (~6 episodes) + terrain-caused (§21).
  RIGHT  — RESIDUAL LADDER: bits/tick, raw → recompute-marginal (Rung 1) → goal-only
           event rate. Under a Baritone-A* decoder the recompute is deterministic-
           given-goal, so the residual collapses to the GOAL/intent stream.

Usage:
    .venv/bin/python -m experiments.codec_loop.path_codec_rung2_plot \
        --json results/sprint22/rung2.json --out results/sprint22/rung2.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/sprint22/rung2.json")
    ap.add_argument("--out", default="results/sprint22/rung2.png")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    t, c, g = d["timing"], d["content"], d["goal_residual"]

    plt.rcParams.update({"axes.titlesize": 9.5})
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 5.0))

    # LEFT: timing AUC
    labels = ["chance", "nodes-ahead\nonly", "full\nplan-state"]
    vals = [0.5, t["nodes_ahead_only"]["auc"], t["full"]["auc"]]
    axL.bar(labels, vals, color=["#999999", "#ff7f0e", "#1f77b4"], alpha=0.85)
    axL.set_ylim(0.4, 1.0)
    axL.axhline(0.5, color="k", lw=0.8, ls="--")
    axL.set_title(f"PART A — recompute TIMING (AUC, within {d['K_ahead']} ticks)\n"
                  f"the WHEN is mostly free (base rate {t['full']['base_rate']*100:.1f}%)")
    axL.set_ylabel("AUC")
    for i, v in enumerate(vals):
        axL.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # MIDDLE: content entropy
    clab = ["goal-anchor\n(straight)", "predictive\ncoding", "plan-state\nconditional"]
    cval = [c["h_marginal_goal_anchor_bits"], c["h_predictive_coding_bits"],
            c["h_conditional_planstate_bits"]]
    axM.bar(clab, cval, color=["#1f77b4", "#d62728", "#2ca02c"], alpha=0.85)
    axM.set_title(f"PART B — recompute CONTENT entropy (bits)\n"
                  f"detour direction data-starved ({c['n_detour']} ev / "
                  f"{c['n_detour_rollouts']} rollouts) + terrain-caused (§21)")
    axM.set_ylabel("bits / recompute")
    axM.set_ylim(0, max(cval) * 1.25)
    for i, v in enumerate(cval):
        axM.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # RIGHT: residual ladder
    rlab = ["raw\n(sector/tick)", "recompute\nmarginal", "goal-only\n(A* decoder)"]
    rval = [g["raw_sector_bits_per_tick"], g["recompute_marginal_bits_per_tick"],
            g["goal_event_rate_per_tick"]]
    axR.bar(rlab, rval, color=["#999999", "#ff7f0e", "#1f77b4"], alpha=0.85)
    axR.set_yscale("log")
    axR.set_title(f"PART C — residual ladder (bits/tick)\n"
                  f"goal events {g['goal_vs_recompute_event_ratio']:.1f}× sparser than recomputes "
                  f"— each a real decision")
    axR.set_ylabel("bits / tick (log)")
    for i, v in enumerate(rval):
        axR.text(i, v * 1.2, f"{v:.4g}", ha="center", fontsize=9)

    fig.suptitle(f"§22 Rung 2 — recompute predictability  (n={d['total_ticks']} ticks, "
                 f"{d['n_rollouts']} rollouts, {c['n_recompute']} recomputes, "
                 f"{g['goal_changes']} goal-changes)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
