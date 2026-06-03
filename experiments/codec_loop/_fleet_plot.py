"""Two parity-vs-bits curves from the fleet sweep: continuous behavioral
(reach_rate / dist) vs categorical server-side (rubberbands per 1k movement
packets). Colleague's question: which knee is more diagnostic?"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/sprintA/fleet.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/sprintA/fleet_curves.png"
d = json.load(open(path))

def xb(r):
    return 10 if r["bits"] is None else r["bits"]

rows = sorted(d["results"], key=xb)
bits = [xb(r) for r in rows]
labels = ["loss" if r["bits"] is None else str(r["bits"]) for r in rows]
reach = [r["reach_rate"] for r in rows]
dist = [r["dist_mean"] for r in rows]
rbk = [r["rb_per_ksub"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

ax1.plot(bits, reach, "o-", color="tab:blue", label="reach_rate")
ax1.set_xlabel("quantization bits (lossless=10)")
ax1.set_ylabel("reach_rate", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(-0.05, 1.05)
ax1.invert_xaxis()
ax1b = ax1.twinx()
ax1b.plot(bits, dist, "s--", color="tab:red", label="dist_mean")
ax1b.set_ylabel("dist_mean to target (blocks)", color="tab:red")
ax1b.tick_params(axis="y", labelcolor="tab:red")
ax1.set_title("Behavioral parity (continuous)")
ax1.set_xticks(bits); ax1.set_xticklabels(labels); ax1.grid(alpha=0.3)

ax2.plot(bits, rbk, "D-", color="tab:green")
ax2.set_xlabel("quantization bits (lossless=10)")
ax2.set_ylabel("rubberbands / 1k movement packets")
ax2.set_yscale("log")
ax2.invert_xaxis()
ax2.set_title("Server-side parity (categorical, log y)")
ax2.set_xticks(bits); ax2.set_xticklabels(labels); ax2.grid(alpha=0.3, which="both")

fig.suptitle(f"Sprint A fleet sweep — {len(d['agents'])} agents, "
             f"{rows[0]['n_legs']} legs/level, avoidable-wall arena (drift=0)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=110)
print(f"wrote {out}")
