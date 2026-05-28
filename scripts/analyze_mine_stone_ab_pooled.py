#!/usr/bin/env python3
"""Pool multiple mine_stone descend-nudge A/B runs and test significance.

Pools per-rollout results across all given run dirs (default: every
results/mine-stone-nudge-ab-* dir), then reports pooled per-arm rates and a
Fisher's exact test on the primary metric (stone_acquired: yes/no per rollout).

Contamination guard: any run dir containing a transport_error (the jar-corruption
signature, see memory: jar-deploy-over-running-corrupts) is AUTO-EXCLUDED, since
its arms are confounded. Pass dirs explicitly to override.

Usage:
  scripts/analyze_mine_stone_ab_pooled.py [RUN_DIR ...]
"""
import glob
import sys
from math import comb

sys.path.insert(0, "scripts")
from analyze_mine_stone_ab import analyze_iter  # noqa: E402


def run_has_transport_error(run_dir):
    import json
    for arm in ("on", "off"):
        for p in glob.glob(f"{run_dir}/{arm}/*.jsonl"):
            for ln in open(p):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if "transport_error" in str(r.get("outcome") or ""):
                    return True
    return False


def fisher_2x2(a, b, c, d):
    """Two-sided + one-sided (on>off) Fisher exact for [[a,b],[c,d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p_table(x):
        b_, c_, d_ = r1 - x, c1 - x, r2 - (c1 - x)
        if min(b_, c_, d_, x) < 0:
            return 0.0
        return comb(r1, x) * comb(r2, c_) / comb(n, c1)
    p_obs = p_table(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p_two = sum(p_table(x) for x in range(lo, hi + 1) if p_table(x) <= p_obs + 1e-12)
    p_one = sum(p_table(x) for x in range(a, hi + 1))
    return min(1.0, p_two), min(1.0, p_one)


def main():
    args = sys.argv[1:]
    if args:
        dirs = args
    else:
        dirs = sorted(glob.glob("results/mine-stone-nudge-ab-*"))
    pooled = {"on": [], "off": []}
    used, skipped = [], []
    for d in dirs:
        if not args and run_has_transport_error(d):
            skipped.append(d)
            continue
        used.append(d)
        for arm in ("on", "off"):
            for p in sorted(glob.glob(f"{d}/{arm}/*.jsonl")):
                pooled[arm].append(analyze_iter(p))

    print("=== POOLED MINE_STONE DESCEND-NUDGE A/B ===")
    print("runs pooled:")
    for d in used:
        print(f"  + {d}")
    for d in skipped:
        print(f"  - {d}  (SKIPPED: transport_error contamination)")
    print()

    agg = {}
    for arm in ("on", "off"):
        rows = pooled[arm]
        n = len(rows)
        att = sum(r["attempted"] for r in rows)
        acq = sum(r["acquired"] for r in rows)
        daf = sum(r["descend_after_fail"] for r in rows)
        cob = sum(r["cobble"] for r in rows)
        agg[arm] = dict(n=n, att=att, acq=acq, daf=daf,
                        mean_cobble=cob / n if n else 0.0,
                        mean_fails=sum(r["fails"] for r in rows) / n if n else 0.0)

    def pct(x, n):
        return f"{100*x/n:5.1f}%" if n else "n/a"
    print(f"{'arm':<5} {'n':>3} {'attempted':>10} {'stone_acq':>12} {'mean_cobble':>12} "
          f"{'descend_after_fail':>20} {'mean_fails':>11}")
    for arm in ("on", "off"):
        s = agg[arm]
        n = s["n"]
        print(f"{arm:<5} {n:>3} {pct(s['att'],n):>10} "
              f"{s['acq']}/{n} ({pct(s['acq'],n).strip()}) ".rjust(13)
              + f"{s['mean_cobble']:>12.2f} {pct(s['daf'],n):>20} {s['mean_fails']:>11.2f}")

    on, off = agg["on"], agg["off"]
    a, b = on["acq"], on["n"] - on["acq"]
    c, d = off["acq"], off["n"] - off["acq"]
    if on["n"] and off["n"]:
        p_two, p_one = fisher_2x2(a, b, c, d)
        print()
        print(f"stone_acquired: ON {a}/{on['n']} ({100*a/on['n']:.0f}%) vs "
              f"OFF {c}/{off['n']} ({100*c/off['n']:.0f}%)")
        print(f"Fisher exact:  two-sided p = {p_two:.4f}   one-sided (ON>OFF) p = {p_one:.4f}")
        verdict = "SIGNIFICANT (p<0.05)" if p_two < 0.05 else "not yet significant"
        print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
