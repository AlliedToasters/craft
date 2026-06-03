"""Render the fleet sweep results table from fleet.json to /tmp/fleet_table.txt.
Disk-truth only: reads the JSON the harness wrote, writes a plain table to a file
we then Read back (avoids trusting any streamed/rendered stdout)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/sprintA/fleet.json"
d = json.load(open(path))
out = []
out.append(f"agents={len(d['agents'])} trials/agent={d['trials_per_agent']} "
           f"barrier={d['barrier']} obstacle={d['obstacle_size']}x{d['obstacle_size']}"
           f"x{d['obstacle_height']} spacing={d['spacing']}")
out.append(f"setup ok: {sum(1 for v in d['setup'].values() if v.get('ok'))}/{len(d['setup'])}")
out.append("")
hdr = (f"{'bits':>8} {'legs':>5} {'reach':>6} {'dist_mn':>7} {'home_mn':>7} "
       f"{'rb':>6} {'rb/leg':>7} {'rb/ksub':>8} {'subst':>7} {'drift':>5} {'wall_s':>6}")
out.append(hdr)
out.append("-" * len(hdr))

def key(r):
    return 999 if r["bits"] is None else r["bits"]

for r in sorted(d["results"], key=key, reverse=True):
    out.append(f"{str(r['bits']):>8} {r['n_legs']:>5} {str(r['reach_rate']):>6} "
               f"{str(r['dist_mean']):>7} {str(r['home_mean']):>7} {r['rubberbands']:>6} "
               f"{str(r['rb_per_leg']):>7} {str(r['rb_per_ksub']):>8} {r['substituted']:>7} "
               f"{r['drift']:>5} {str(r.get('wall_s')):>6}")
    if r.get("errors"):
        out.append(f"         errors: {r['errors']}")
out.append("")
out.append("per-agent reach_rate spread per level (agents should track together):")
for r in sorted(d["results"], key=key, reverse=True):
    rates = sorted(round(a["reach_rate"], 3) for a in r["per_agent"]
                   if a.get("reach_rate") is not None)
    if rates:
        out.append(f"  b={str(r['bits']):>8}: min={rates[0]} "
                   f"med={rates[len(rates)//2]} max={rates[-1]} n={len(rates)}")
total_wall = sum(r.get("wall_s", 0) or 0 for r in d["results"])
total_legs = sum(r["n_legs"] for r in d["results"])
out.append("")
out.append(f"total sweep wall-clock: {total_wall:.0f}s ({total_wall/60:.1f} min) for {total_legs} legs")
open("/tmp/fleet_table.txt", "w").write("\n".join(out) + "\n")
print("OK rows", len(d["results"]))
