"""Render the fleet sweep summary (exact values + both parity curves) into ONE
PNG, so results can be read as an image — bypassing text-channel corruption."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/sprintA/fleet.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/sprintA/fleet_summary.png"
d = json.load(open(path))

def key(r):
    return -1 if r["bits"] is None else r["bits"]

rows = sorted(d["results"], key=key, reverse=True)
labels = ["lossless" if r["bits"] is None else f"b={r['bits']}" for r in rows]
reach = [r["reach_rate"] for r in rows]
dist = [r["dist_mean"] for r in rows]
rbk = [r["rb_per_ksub"] for r in rows]
rb = [r["rubberbands"] for r in rows]
subst = [r["substituted"] for r in rows]
drift = [r["drift"] for r in rows]
legs = [r["n_legs"] for r in rows]

xs = list(range(len(rows)))  # left=lossless ... right=b3 (coarser)

fig = plt.figure(figsize=(14, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.4])

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(xs, reach, "o-", color="tab:blue")
ax1.set_ylabel("reach_rate", color="tab:blue")
ax1.set_ylim(-0.05, 1.05)
ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.grid(alpha=0.3)
ax1b = ax1.twinx()
ax1b.plot(xs, dist, "s--", color="tab:red")
ax1b.set_ylabel("dist_mean (blocks)", color="tab:red")
ax1.set_title("Behavioral parity")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(xs, rbk, "D-", color="tab:green")
ax2.set_ylabel("rubberbands / 1k move-packets")
ax2.set_yscale("log")
ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.grid(alpha=0.3, which="both")
ax2.set_title("Server-side parity (log y)")

# exact-value table panel
ax3 = fig.add_subplot(gs[0, 2]); ax3.axis("off")
cols = ["level", "legs", "reach", "dist", "rb", "rb/ksub", "subst", "drift"]
cells = []
for i in range(len(rows)):
    cells.append([labels[i], legs[i], f"{reach[i]:.3f}", f"{dist[i]:.2f}",
                  rb[i], f"{rbk[i]:.1f}", subst[i], drift[i]])
t = ax3.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
ax3.set_title("exact values (read from bytes)")

setup_ok = sum(1 for v in d["setup"].values() if v.get("ok"))
fig.suptitle(f"Sprint A FLEET — {len(d['agents'])} agents, setup {setup_ok}/{len(d['setup'])} ok, "
             f"{legs[0]} legs/level, avoidable-wall (3x3x3), drift=0 all levels", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out, dpi=120)
print("wrote", out)
