#!/usr/bin/env python3
"""§21.1 plot — the bearing-precision knee.

Reads bearing_ablation.json and renders detour-subset recovery (and aggregate
accuracy) as the goal direction is coarsened: exact → 45° → 90° → 180° → none.
The shaded span = the bearing-dependent gap (full_exact − no-bearing) = the part
of the local plan recoverable ONLY with the goal signal = NOT scene-inferable
from local terrain = the §21.2 vision job.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_bearing_plot \
        --json results/sprint21/bearing_ablation.json --out results/sprint21/bearing_knee.png
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
    ap.add_argument("--json", default="results/sprint21/bearing_ablation.json")
    ap.add_argument("--out", default="results/sprint21/bearing_knee.png")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    arms = data["arms"]
    ks = data["ks"]

    # ordinal x-axis: exact, then each coarsening k (label by angular precision)
    order = ["full_exact"] + [f"k{k}" for k in ks]
    labels = ["exact"] + [(f"{360//k}°" if k > 1 else "none") for k in ks]
    xs = list(range(len(order)))
    det = [arms[a]["detour_within1"] if arms.get(a) else float("nan") for a in order]
    agg = [arms[a]["sector_within1"] if arms.get(a) else float("nan") for a in order]

    bo = arms.get("bearing_only") or {}
    straight = arms["full_exact"]["straight_within1"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(xs, det, "-o", color="#1f77b4", lw=2, label="detour subset (±1 sector)")
    ax.plot(xs, agg, "-s", color="#999999", lw=1.5, alpha=0.8, label="aggregate (±1 sector)")

    # reference floors
    ax.axhline(straight, color="#d62728", ls=":", alpha=0.7,
               label=f"straight-line floor ({straight:.2f})")
    if "detour_within1" in bo:
        ax.axhline(bo["detour_within1"], color="#2ca02c", ls="--", alpha=0.6,
                   label=f"bearing-only, terrain ablated ({bo['detour_within1']:.2f})")

    # the headline is the FLATNESS: detour recovery is invariant to bearing
    # precision, so the gap exact→none is ~0 — the bearing is NOT the bottleneck.
    if len(det) >= 2:
        gap = data.get("scene_inferability_gap", det[0] - det[-1])
        ax.annotate(f"detour recovery is FLAT in bearing precision\n"
                    f"(exact→none gap = {gap:+.2f}) — coarsening the\n"
                    f"goal to nothing doesn't change it. The bearing\n"
                    f"only resolves aim-at-goal; detours need richer\n"
                    f"TERRAIN, not a better bearing (→ §21.2).",
                    xy=(xs[-1], det[-1]), xytext=(1.0, 0.30),
                    color="#ff7f0e", fontsize=8, va="center",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", alpha=0.6))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("goal-direction precision fed to the head  (coarser →)")
    ax.set_ylabel("subgoal recovered (±1 sector)")
    ax.set_title(f"§21.1 bearing-precision knee  (feat_r={data['feat_r']}, "
                 f"target_r={data['target_r']})\n"
                 "how precisely must the goal direction be known for local planning?")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
