#!/usr/bin/env python3
"""§22 Rung 1 plot — where the path-stream residual lives, and how small it is.

Three panels from rung1.json:
  LEFT   — commit-run length histogram: the stream is long committed segments. Mean
           run length IS the §20.0 commit-length compression factor (one transmitted
           event reproduces the whole run; within-run coverage 0.996 = pure consumption).
  MIDDLE — recompute residual: sectors-off-goal at each recompute. Overwhelmingly 0-1
           (aligned extensions, "head at the goal"); the thin >1 tail is the DETOUR
           origination — the §21-uncrackable bend, relocated to a sparse plan event.
  RIGHT  — amortised bits/tick: raw (send a sector every tick) vs the marginal residual
           (entropy at recomputes × recompute rate). Rung 2 conditions on plan-state to
           lower the residual further; the gap it can close = "is the detour predictable".

Usage:
    .venv/bin/python -m experiments.codec_loop.path_codec_plot \
        --json results/sprint22/rung1.json --out results/sprint22/rung1.png
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
    ap.add_argument("--json", default="results/sprint22/rung1.json")
    ap.add_argument("--out", default="results/sprint22/rung1.png")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    c, b, r = d["compression"], d["bits"], d["residual"]
    h = d["hist"]

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 5.0))
    plt.rcParams.update({"axes.titlesize": 9.5})

    # LEFT: run-length histogram
    rl = h["run_len_bins"]
    axL.bar(rl["labels"], rl["counts"], color="#1f77b4", alpha=0.85)
    axL.set_title(f"Commit-run length (ticks/segment)\n"
                  f"mean {c['mean_run_len_pooled']:.0f}× = §20.0 commit factor  "
                  f"(within-run coverage {d['within_run_freeness']:.3f})")
    axL.set_xlabel("ticks per committed segment")
    axL.set_ylabel("# runs")
    axL.tick_params(axis="x", rotation=45)

    # MIDDLE: sectors-off-goal at recompute
    so = h["sectors_off_counts"]
    xs = list(range(len(so)))
    colors = ["#2ca02c" if i <= 1 else "#d62728" for i in xs]
    axM.bar(xs, so, color=colors, alpha=0.85)
    det = r["detour_frac_of_recomputes"] * 100
    axM.set_title(f"Recompute residual: sectors off goal\n"
                  f"green ≤1 (aligned extension) | red >1 = DETOUR ({det:.1f}% of recomputes)")
    axM.set_xlabel("sectors off straight-to-goal at recompute")
    axM.set_ylabel("# recomputes")
    axM.axvline(1.5, color="k", lw=0.8, ls="--")

    # RIGHT: bits/tick
    bars = ["raw\n(sector/tick)", "residual\n(marginal)"]
    vals = [b["raw_sector_bits_per_tick"], b["residual_bits_per_tick_marginal"]]
    axR.bar(bars, vals, color=["#999999", "#1f77b4"], alpha=0.85)
    axR.set_yscale("log")
    axR.set_title(f"Amortised bits/tick\n{b['compression_x_vs_raw_sector']:.0f}× "
                  f"(marginal upper bound — Rung 2 lowers the residual)")
    axR.set_ylabel("bits / tick (log)")
    for i, v in enumerate(vals):
        axR.text(i, v * 1.15, f"{v:.4g}", ha="center", fontsize=9)

    fig.suptitle(f"§22 Rung 1 — path-state navigation codec residual  "
                 f"(n={d['total_ticks']} ticks, {d['n_rollouts']} rollouts, "
                 f"{d['total_recomputes']} recomputes, {d['total_goal_changes']} goal-changes)",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
