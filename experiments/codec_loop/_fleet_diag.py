"""Diagnose the fleet sweep: goto-reason distribution per level + one sample leg +
per-level error dict. Writes plain text to /tmp/fleet_diag.txt."""
import collections
import json

d = json.load(open("results/sprintA/fleet.json"))
out = ["NONCE_DIAG_Z8"]

def key(r):
    return -1 if r["bits"] is None else r["bits"]

for r in sorted(d["results"], key=key, reverse=True):
    b = "lossless" if r["bits"] is None else f"b={r['bits']}"
    reasons = collections.Counter()
    finals = 0
    for a in r["per_agent"]:
        for leg in a.get("legs", []):
            for side in ("out", "home"):
                reasons[str(leg[side].get("reason"))] += 1
                if leg[side].get("final_position") is not None:
                    finals += 1
    out.append(f"\n=== {b}: reach={r['reach_rate']} dist_mean={r['dist_mean']} "
               f"wall_s={r.get('wall_s')} n_per_agent={len(r['per_agent'])} ===")
    out.append(f"  errors={r.get('errors')}")
    out.append(f"  legs_with_final_pos={finals}")
    for reason, c in reasons.most_common():
        out.append(f"  reason {reason!r}: {c}")

# one concrete sample leg from the lossless level
loss = [r for r in d["results"] if r["bits"] is None][0]
if loss["per_agent"] and loss["per_agent"][0].get("legs"):
    a0 = loss["per_agent"][0]
    out.append(f"\nSAMPLE lossless agent {a0.get('player')} leg[0]:")
    out.append(json.dumps(a0["legs"][0], indent=2)[:1200])

# setup detail for a couple agents
out.append("\nSETUP sample:")
for k in list(d["setup"])[:3]:
    out.append(f"  {k}: {d['setup'][k]}")

open("/tmp/fleet_diag.txt", "w").write("\n".join(out) + "\n")
print("OK_DIAG")
