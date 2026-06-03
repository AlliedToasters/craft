"""Sprint A/B reconnaissance over frozen captures.

A: movement-packet field dynamic ranges (sets quantization levels).
B: g_t content-population rate (the true Sprint-B gate).

Writes a plain-text report to stdout; caller redirects to a file.
"""
import json, glob, math, collections

SETS = {
    "narrated": "results/frozen_narrated",
    "combat": "results/frozen_combat",
    "dryrun": "results/frozen_dryrun",
}

MOVE_TYPES = {
    "minecraft:move_player_pos", "minecraft:move_player_rot",
    "minecraft:move_player_pos_rot", "minecraft:move_player_status_only",
}


def packet_files(root):
    return sorted(glob.glob(f"{root}/rollout-*/packets.jsonl"))


def stats(vals):
    if not vals:
        return "  (none)"
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n

    def pct(q):
        return vals[min(n - 1, int(q * n))]

    return (f"n={n} min={vals[0]:.4f} p01={pct(.01):.4f} p50={pct(.5):.4f} "
            f"p99={pct(.99):.4f} max={vals[-1]:.4f} sd={math.sqrt(var):.4f}")


def recon_set(name, root):
    print(f"\n========== {name}  ({root}) ==========")
    type_counts = collections.Counter()
    gt_populated = 0
    gt_total = 0
    gt_distinct = collections.Counter()
    dpos = {"x": [], "y": [], "z": []}
    rot = {"yaw": [], "pitch": []}
    n_lines = 0
    sample_keys_printed = False
    for pf in packet_files(root):
        for line in open(pf):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_lines += 1
            if not sample_keys_printed:
                obs0 = d.get("obs", {}) or {}
                print(f"  [schema] top={list(d.keys())}")
                print(f"  [schema] obs={list(obs0.keys())}")
                print(f"  [schema] fields(ex)={d.get('fields')}")
                sample_keys_printed = True
            pid = d.get("id")
            type_counts[pid] += 1
            obs = d.get("obs", {}) or {}
            gt = obs.get("g_t")
            gt_total += 1
            if gt not in (None, "", "null"):
                gt_populated += 1
                gt_distinct[gt] += 1
            fields = d.get("fields", {}) or {}
            if pid in MOVE_TYPES:
                opos = obs.get("pos") or obs.get("position")
                # obs.pos is a list [x,y,z]; map to axis index for delta.
                if isinstance(opos, (list, tuple)) and len(opos) == 3:
                    obase = {"x": opos[0], "y": opos[1], "z": opos[2]}
                elif isinstance(opos, dict):
                    obase = opos
                else:
                    obase = None
                for axis in ("x", "y", "z"):
                    v = fields.get(axis)
                    if v is None:
                        continue
                    if obase and axis in obase:
                        dpos[axis].append(v - obase[axis])
                    else:
                        dpos[axis].append(v)
                for rk in ("yaw", "pitch"):
                    if fields.get(rk) is not None:
                        rot[rk].append(fields[rk])
    print(f"lines={n_lines}")
    print("--- packet type mix ---")
    for t, c in type_counts.most_common():
        print(f"  {t:28s} {c:7d}  {100*c/max(n_lines,1):5.1f}%")
    print("--- g_t population (Sprint B gate) ---")
    print(f"  g_t populated: {gt_populated}/{gt_total} = "
          f"{100*gt_populated/max(gt_total,1):.1f}%")
    print(f"  distinct g_t values: {len(gt_distinct)}")
    for g, c in gt_distinct.most_common(12):
        print(f"    {c:6d}  {g!r}")
    print("--- movement field ranges (Sprint A target) ---")
    for axis in ("x", "y", "z"):
        print(f"  dpos.{axis}: {stats(dpos[axis])}")
    for rk in ("yaw", "pitch"):
        print(f"  {rk}: {stats(rot[rk])}")


def summary_dict(root):
    out = {"type_mix": {}, "g_t_pct": None, "g_t_distinct": None,
           "fields": {}, "n": 0}
    type_counts = collections.Counter()
    gt_pop = gt_tot = 0
    gt_distinct = set()
    acc = {"dpos.x": [], "dpos.y": [], "dpos.z": [], "yaw": [], "pitch": []}
    n = 0
    for pf in packet_files(root):
        for line in open(pf):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            pid = d.get("id")
            type_counts[pid] += 1
            obs = d.get("obs", {}) or {}
            gt = obs.get("g_t")
            gt_tot += 1
            if gt not in (None, "", "null"):
                gt_pop += 1
                gt_distinct.add(gt)
            fields = d.get("fields", {}) or {}
            # capture nests the actual fields under fields["fields"].
            if isinstance(fields.get("fields"), dict):
                fields = fields["fields"]
            if pid in MOVE_TYPES:
                # obs carries flat x/y/z; the codec delta-codes pos vs obs.
                # obs is per-tick and can be stale vs a teleported packet, so
                # |delta|>=10 blocks are TP/spawn discontinuities, not a tick
                # of locomotion — exclude them from the locomotion stats.
                for axis in ("x", "y", "z"):
                    v = fields.get(axis)
                    o = obs.get(axis)
                    if v is not None and o is not None and abs(v - o) < 10:
                        acc[f"dpos.{axis}"].append(v - o)
                for rk in ("yaw", "pitch"):
                    if fields.get(rk) is not None:
                        acc[rk].append(fields[rk])
    out["n"] = n
    out["type_mix"] = {t: round(100 * c / max(n, 1), 2)
                       for t, c in type_counts.most_common(8)}
    out["g_t_pct"] = round(100 * gt_pop / max(gt_tot, 1), 1)
    out["g_t_distinct"] = len(gt_distinct)
    for k, vals in acc.items():
        if not vals:
            continue
        vals.sort()
        m = len(vals)
        mean = sum(vals) / m
        sd = (sum((v - mean) ** 2 for v in vals) / m) ** 0.5
        out["fields"][k] = {
            "n": m, "min": round(vals[0], 4),
            "p01": round(vals[int(.01 * m)], 4),
            "p50": round(vals[int(.5 * m)], 4),
            "p99": round(vals[min(m - 1, int(.99 * m))], 4),
            "max": round(vals[-1], 4), "sd": round(sd, 4)}
    return out


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        res = {name: summary_dict(root) for name, root in SETS.items()
               if glob.glob(f"{root}/rollout-*")}
        json.dump(res, open("/tmp/recon_summary.json", "w"), indent=1)
        print("wrote /tmp/recon_summary.json")
    else:
        for name, root in SETS.items():
            if glob.glob(f"{root}/rollout-*"):
                recon_set(name, root)
            else:
                print(f"(skip {name}: no rollouts at {root})")
