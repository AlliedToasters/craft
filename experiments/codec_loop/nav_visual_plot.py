#!/usr/bin/env python3
"""§21.2 plot — does the frame add detour info beyond the camera angle?

Two panels from visual.json:
  LEFT  — detour-subset recovery: structured (§21.0 ~0.12) vs cam_only (camera yaw =
          Baritone's heading readout) vs full_visual (CNN + cam vec). The pixel gain
          (full − cam_only) is ~0: vision doesn't crack the horizontal detour.
  RIGHT — Δy accuracy: full_visual vs cam_only — the one place pixels help (elevation
          is legible in the frame, not in the horizontal heading).

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_visual_plot \
        --json results/sprint21_visual/visual.json --out results/sprint21_visual/visual.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

STRUCTURED_DETOUR = 0.12   # §21.0 floor/block/water @ r6 (cross-rung reference)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/sprint21_visual/visual.json")
    ap.add_argument("--out", default="results/sprint21_visual/visual.png")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    fv, co = d["arms"]["full_visual"], d["arms"]["cam_only"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # LEFT: detour recovery
    labels = ["structured\n(§21.0)", "cam_only\n(heading)", "full_visual\n(CNN+cam)"]
    vals = [STRUCTURED_DETOUR, co["detour_within1"], fv["detour_within1"]]
    errs = [0, co.get("detour_std", 0), fv.get("detour_std", 0)]
    colors = ["#999999", "#2ca02c", "#1f77b4"]
    axL.bar(labels, vals, yerr=errs, color=colors, capsize=4, alpha=0.85)
    axL.axhline(0, color="k", lw=0.8)
    gain = fv["detour_within1"] - co["detour_within1"]
    axL.set_title(f"Detour-subset recovery (±1 sector)\n"
                  f"pixel gain full−cam_only = {gain:+.3f}  (≈0: frame adds no horizontal routing)")
    axL.set_ylabel("detour subgoal recovered")
    axL.set_ylim(0, max(vals) * 1.4)
    for i, v in enumerate(vals):
        axL.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    # RIGHT: Δy accuracy
    dyl = ["cam_only", "full_visual"]
    dyv = [co["dy_acc"], fv["dy_acc"]]
    axR.bar(dyl, dyv, color=["#2ca02c", "#1f77b4"], alpha=0.85)
    axR.set_title("Δy (elevation) accuracy\nthe one place pixels help — terrain height is legible")
    axR.set_ylabel("Δy class accuracy")
    axR.set_ylim(0, 1)
    for i, v in enumerate(dyv):
        axR.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"§21.2 visual subgoal distillation  "
                 f"(n={d['n_samples']} frames, {d['n_rollouts']} rollouts, img={d['img_size']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
