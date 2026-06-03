"""Read the REAL sanity numbers from fleet_sanity.json -> /tmp/sanity_check.txt."""
import collections
import json

d = json.load(open("results/sprintA/fleet_sanity.json"))
out = ["NONCE_SANITY_Q5"]
out.append(f"agents={d['agents']} trials/agent={d['trials_per_agent']} barrier={d['barrier']}")
for r in d["results"]:
    b = "lossless" if r["bits"] is None else f"b={r['bits']}"
    reasons = collections.Counter()
    for a in r["per_agent"]:
        for leg in a.get("legs", []):
            for side in ("out", "home"):
                reasons[str(leg[side].get("reason"))] += 1
    out.append(f"{b}: reach={r['reach_rate']} dist_mean={r['dist_mean']} "
               f"wall_s={r.get('wall_s')} reasons={dict(reasons)}")
# sample leg
r0 = d["results"][0]
if r0["per_agent"] and r0["per_agent"][0].get("legs"):
    out.append("SAMPLE leg[0] lossless:")
    out.append(json.dumps(r0["per_agent"][0]["legs"][0], indent=2)[:900])
open("/tmp/sanity_check.txt", "w").write("\n".join(out) + "\n")
print("OK_SANITY")
