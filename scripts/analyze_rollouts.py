#!/usr/bin/env python3
"""Pareto frontier analysis for quant-comparison rollouts.

Reads per-agent JSONL rollout files from a results directory and outputs:
  - per-rollout CSV  (PREFIX.rollouts.csv)
  - per-quant aggregate summary (printed + PREFIX.summary.csv)

Usage:
  .venv/bin/python scripts/analyze_rollouts.py results/pareto_overnight_YYYYMMDD/
  .venv/bin/python scripts/analyze_rollouts.py results/pareto_overnight_YYYYMMDD/ --min-rollouts 5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


# ────────────────────────── tech tier helpers ──────────────────────────────

_TIER_ORDER = ["diamond", "iron", "stone", "wooden", "none"]
_TIER_RANK = {t: i for i, t in enumerate(_TIER_ORDER)}

_PICKAXES = ["diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe"]
def _tech_tier(inv: dict | None) -> str:
    """Highest crafting tier present in inventory (pickaxe-based)."""
    if not inv:
        return "none"
    for px in _PICKAXES:
        for k, v in inv.items():
            if isinstance(v, int) and v >= 1 and k.endswith(px):
                return px.split("_")[0]  # diamond / iron / stone / wooden
    return "none"


# ───────────────────────────── JSONL parser ──────────────────────────────

def _parse_jsonl(path: Path) -> dict | None:
    """Parse a single rollout JSONL.  Returns a flat metrics dict or None on error."""
    turns = []
    end_rec = None
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("_type")
                if t == "turn":
                    turns.append(rec)
                elif t == "end":
                    end_rec = rec
    except OSError:
        return None

    if not turns:
        return None

    # Survival
    death_turn = next((r for r in turns if r.get("died")), None)
    survived = death_turn is None

    last = turns[-1]
    n_turns = last.get("turn", len(turns))
    day_count = last.get("day_count") or 0

    # Milestones fired
    milestones = [r.get("milestone_fired") for r in turns if r.get("milestone_fired")]
    m1 = "M1_iron_goal" in milestones
    m2 = "M2_diamond_goal" in milestones

    # Tech tier (best seen across all turns — use inventory at the last surviving turn)
    final_inv = last.get("inventory") or {}
    tier = _tech_tier(final_inv)

    # Cause of death
    cause = None
    death_msg = None
    if death_turn:
        d = death_turn.get("death") or {}
        cause = d.get("cause", "unknown")
        death_msg = d.get("message", "")

    # "Died while thinking": died turn with exec_s ≈ 0 (dispatch was skipped)
    died_thinking = False
    if death_turn:
        exec_s = death_turn.get("exec_s") or 0.0
        plan_s = death_turn.get("plan_s") or 0.0
        # exec_s absent or near-zero AND plan_s non-trivial → killed during planning
        died_thinking = exec_s < 0.05 and plan_s > 0.1

    # LLM timing (from end record)
    plan_s_mean = (end_rec or {}).get("plan_s_mean") or 0.0
    llm_idle_pct = (end_rec or {}).get("llm_idle_pct") or 0.0
    wall_s = (end_rec or {}).get("wall_s") or 0.0

    # Evasion fire rate
    evasion_fires = sum(1 for r in turns if (r.get("evasion") or {}).get("fired"))
    evasion_rate = evasion_fires / len(turns) if turns else 0.0

    # Tool call distribution (from each turn's tool field)
    tool_counts: dict[str, int] = {}
    for r in turns:
        t = r.get("tool")
        if t:
            tool_counts[t] = tool_counts.get(t, 0) + 1

    return {
        "file": path.name,
        "survived": survived,
        "turns": n_turns,
        "day_count": day_count,
        "m1": m1,
        "m2": m2,
        "tier": tier,
        "tier_rank": _TIER_RANK.get(tier, 4),
        "cause": cause or "",
        "death_msg": (death_msg or "")[:80],
        "died_thinking": died_thinking,
        "plan_s_mean": round(plan_s_mean, 3),
        "llm_idle_pct": round(llm_idle_pct, 1),
        "wall_s": round(wall_s, 1),
        "evasion_rate": round(evasion_rate, 3),
        "tool_counts": json.dumps(tool_counts),
    }


# ──────────────────────────── aggregation ──────────────────────────────────

def _pct(xs, q=0.5):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    survived = [r for r in rows if r["survived"]]
    deaths = [r for r in rows if not r["survived"]]
    survival_rate = len(survived) / n

    turns = [r["turns"] for r in rows]
    days = [r["day_count"] for r in rows]
    plan_s = [r["plan_s_mean"] for r in rows if r["plan_s_mean"] > 0]
    idle = [r["llm_idle_pct"] for r in rows if r["llm_idle_pct"] > 0]
    ev_rate = [r["evasion_rate"] for r in rows]
    tiers = [r["tier_rank"] for r in rows]
    died_thinking = sum(1 for r in deaths if r["died_thinking"])

    # Cause of death breakdown
    causes: dict[str, int] = {}
    for r in deaths:
        c = r["cause"] or "unknown"
        causes[c] = causes.get(c, 0) + 1
    top_cause = max(causes, key=lambda k: causes[k]) if causes else "n/a"

    return {
        "n": n,
        "survival_rate": round(survival_rate, 3),
        "survival_pct": f"{survival_rate*100:.0f}%",
        "m1_rate": round(sum(1 for r in rows if r["m1"]) / n, 3),
        "m2_rate": round(sum(1 for r in rows if r["m2"]) / n, 3),
        "turns_mean": round(statistics.mean(turns), 1),
        "turns_p50": _pct(turns),
        "days_mean": round(statistics.mean(days), 2),
        "tier_mean": round(statistics.mean(tiers), 2),  # lower = better
        "plan_s_mean": round(statistics.mean(plan_s), 3) if plan_s else 0,
        "plan_s_p95": round(_pct(plan_s, 0.95), 3) if plan_s else 0,
        "llm_idle_pct": round(statistics.mean(idle), 1) if idle else 0,
        "evasion_rate": round(statistics.mean(ev_rate), 3),
        "died_thinking": died_thinking,
        "died_thinking_pct": f"{died_thinking}/{len(deaths)} deaths",
        "top_cause": top_cause,
    }


def _quant_from_filename(name: str) -> str:
    """Extract quant tag from filename like Q4_K_M_w3_a5.jsonl"""
    m = re.match(r"^(Q\d+_K_M|Q\d+_\d+|F16|Q8_0)_", name)
    return m.group(1) if m else "unknown"


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="Results directory containing JSONL rollout files")
    ap.add_argument("--min-rollouts", type=int, default=1,
                    help="Skip quant conditions with fewer than N rollouts")
    ap.add_argument("--out", default=None,
                    help="Output prefix for CSV files (default: <dir>/analysis)")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return

    # Parse all rollouts
    rollouts: list[dict] = []
    errors = 0
    for f in jsonl_files:
        r = _parse_jsonl(f)
        if r is None:
            errors += 1
            continue
        r["quant"] = _quant_from_filename(f.name)
        rollouts.append(r)

    print(f"Parsed {len(rollouts)} rollouts ({errors} errors) from {results_dir}")

    # Group by quant
    by_quant: dict[str, list[dict]] = {}
    for r in rollouts:
        by_quant.setdefault(r["quant"], []).append(r)

    # Print aggregate summary — ordered by latency (fastest first)
    quant_order = ["Q4_K_M", "Q8_0", "F16"]
    present = [q for q in quant_order if q in by_quant]
    present += [q for q in by_quant if q not in quant_order]

    print("\n=== Pareto frontier: latency vs survival ===")
    print(f"{'Quant':<10} {'N':>4} {'Surv%':>6} {'plan_s':>7} {'idle%':>6} "
          f"{'turns':>6} {'days':>5} {'M1%':>5} {'M2%':>5} {'tier':>5} "
          f"{'died_thinking':>14} {'top_cause'}")
    agg_rows = []
    for q in present:
        rows = by_quant[q]
        if len(rows) < args.min_rollouts:
            print(f"  {q}: only {len(rows)} rollouts, skipping (--min-rollouts {args.min_rollouts})")
            continue
        a = _agg(rows)
        a["quant"] = q
        agg_rows.append(a)
        print(f"{q:<10} {a['n']:>4} {a['survival_pct']:>6} {a['plan_s_mean']:>7.3f} "
              f"{a['llm_idle_pct']:>6.1f} {a['turns_mean']:>6.1f} {a['days_mean']:>5.2f} "
              f"{a['m1_rate']*100:>5.0f}% {a['m2_rate']*100:>5.0f}% "
              f"{4-a['tier_mean']:>5.1f} "  # invert rank so higher=better
              f"{a['died_thinking_pct']:>14} {a['top_cause']}")

    # Tier legend
    print("\nTier scale (inverted rank, higher=better): 4=diamond 3=iron 2=stone 1=wood 0=none")

    # Write CSVs
    prefix = args.out or str(results_dir / "analysis")
    if rollouts:
        cols = [c for c in rollouts[0] if c != "tool_counts"] + ["tool_counts"]
        with open(f"{prefix}.rollouts.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rollouts:
                fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"\nwrote {prefix}.rollouts.csv")

    if agg_rows:
        acols = list(agg_rows[0].keys())
        with open(f"{prefix}.summary.csv", "w") as fh:
            fh.write(",".join(acols) + "\n")
            for a in agg_rows:
                fh.write(",".join(str(a.get(c, "")) for c in acols) + "\n")
        print(f"wrote {prefix}.summary.csv")


if __name__ == "__main__":
    main()
