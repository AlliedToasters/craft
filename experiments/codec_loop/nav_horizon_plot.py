#!/usr/bin/env python3
"""§21.0 plot — the navigation horizon curve(s).

Reads the per-(target_r, seed) horizon JSONs and renders two panels:
  LEFT  — detour-subset recovery vs feature radius, one curve per action radius
          target_r; a dashed vertical at each target_r shows the saturation point
          TRACKS the action radius (the headline: horizon = action envelope).
  RIGHT — aggregate accuracy vs feature radius with the straight-line floor, +
          the detour fraction per target_r (navigation is mostly bearing-trivial).

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_horizon_plot \
        --sweeps results/sprint21/sweeps --out results/sprint21/horizon.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="results/sprint21/sweeps")
    ap.add_argument("--out", default="results/sprint21/horizon.png")
    args = ap.parse_args()

    # group seeds by target_r, average the detour curve across seeds
    by_t = defaultdict(list)
    for f in sorted(Path(args.sweeps).glob("h_t*_s*.json")):
        m = re.match(r"h_t(\d+)_s(\d+)\.json", f.name)
        if not m:
            continue
        by_t[int(m.group(1))].append(json.loads(f.read_text()))

    if not by_t:
        print(f"no sweep JSONs in {args.sweeps}")
        return 2

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis([0.15, 0.5, 0.82])
    for ci, t in enumerate(sorted(by_t)):
        runs = by_t[t]
        rs = [c["r"] for c in runs[0]["curve"]]
        # average detour_within1 and aggregate across seeds
        det = [sum(run["curve"][i]["full"]["detour_within1"] for run in runs) / len(runs)
               for i in range(len(rs))]
        agg = [sum(run["curve"][i]["full"]["sector_within1"] for run in runs) / len(runs)
               for i in range(len(rs))]
        straight = runs[0]["curve"][0]["full"]["straight_within1"]
        dfrac = runs[0]["curve"][0]["full"]["detour_frac"]
        c = colors[ci % 3]
        axL.plot(rs, det, "-o", color=c, label=f"action r={t} (detour {dfrac:.0%})")
        axL.axvline(t, color=c, ls="--", alpha=0.4)
        axR.plot(rs, agg, "-o", color=c, label=f"action r={t}  (straight={straight:.2f})")
        axR.axhline(straight, color=c, ls=":", alpha=0.4)

    axL.set_title("Horizon: detour-subset recovery vs feature radius\n"
                  "(dashed = action radius; saturation tracks it)")
    axL.set_xlabel("feature window radius (how far the head sees)")
    axL.set_ylabel("detour subgoal recovered (±1 sector)")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)

    axR.set_title("Aggregate accuracy vs feature radius\n"
                  "(dotted = straight-line floor; most nav is bearing-trivial)")
    axR.set_xlabel("feature window radius")
    axR.set_ylabel("subgoal accuracy (±1 sector)")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
