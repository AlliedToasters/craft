#!/usr/bin/env python3
"""§22 Rung 1 — the path-state navigation codec: LOCATE + SIZE the residual.

The §21 arc closed a clean negative: LOCAL PERCEPTION (terrain, bearing, pixels)
cannot predict the navigation detour — the only learnable local signal was a LEAK
of Baritone's own heading (a plan readout). §22 turns that around and asks the dual
question at the PLAN level, in the spine's "predict the decision, transmit the
residual" frame (§18/§19/§20.0): how cheaply does the controller's own plan-state
encode its near-future, and WHERE do the irreducible bits live?

The structural fact (verified on the capture): Baritone paths in bounded SEGMENTS.
`path_dest` is a segment endpoint ~16-30 blocks ahead of a far `goal`; the agent
walks the committed `path_fwd` (full node list, fwd[0]≈current, fwd[-1]==dest) and
re-invokes A* for a NEW segment only near the segment's end. So:

  * Within a commit-run (constant path_dest), every tick's plan-state is reconstructable
    from the run's committed segment + the tick's position along it — the player advances
    DETERMINISTICALLY along nodes already transmitted. Within-run cost ≈ 0 (index coding,
    the §20.0 mechanism). This is what makes the move STREAM compress 37-437×.
  * ALL the residual bits live at RECOMPUTES — the run boundaries where path_dest jumps
    and genuinely new committed nodes appear. That is the navigation analog of §18's
    "operator departs the argmax": between commitments the codec is free; at a commitment
    it pays.

Rung 1 is descriptive — it MEASURES the residual, it does not yet try to predict it:

  (1) COMPRESSION  — run-length (ticks per commit-run) distribution. Mean run length is
      the §20.0 commit-length compression factor, now on real multi-biome nav data: you
      transmit one event per run to reproduce every tick in it.
  (2) RECOMPUTE RATE — recomputes per tick (segment re-extensions, same goal) reported
      separately from GOAL-CHANGE events (operator re-commands — a different residual,
      the actual intent input, which Rung 2/Phase B will model).
  (3) WITHIN-RUN FREE-NESS — empirical check that the committed future is pure
      consumption within a run (later path_fwd nodes ⊆ the run-start committed set),
      validating the "within-run ≈ 0 bits" claim instead of asserting it.
  (4) RESIDUAL DECOMPOSITION — at each recompute, the NEW committed heading's deviation
      from the straight-line goal bearing (16-way sector, 8=straight, the §21-locked
      representation), split into ALIGNED EXTENSIONS (head at goal, dev ≤1 sector —
      benign, predictable) vs DETOURS (dev >1 — the §21-uncrackable bend, now relocated
      to the plan layer). Plus segment growth.
  (5) RESIDUAL BITS (upper bound) — marginal entropy of the recompute deviation
      distribution × recompute rate = amortised bits/tick the codec must transmit if it
      knows NOTHING about context. Rung 2 lowers this with a predictor; the gap it can
      close is exactly "is the detour predictable from plan-state".

Usage:
    .venv/bin/python -m experiments.codec_loop.path_codec \
        --capture results/sprint21_visual/capture \
        --out results/sprint22/rung1.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

N_SECTORS = 16
STRAIGHT = N_SECTORS // 2                 # dev class 8 == zero deviation (head at goal)
_GOAL_RE = re.compile(r"x=(-?\d+).*?y=(-?\d+).*?z=(-?\d+)")


# --- loading -----------------------------------------------------------------
def _parse_goal(goal_str):
    """GoalBlock{x=..,y=..,z=..} -> (x,y,z) or None."""
    if not goal_str:
        return None
    m = _GOAL_RE.search(goal_str)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def load_rollouts(capture_dir: Path):
    """Per rollout: an ORDERED list of plan-state ticks while actively pathing.

    Each tick is a dict with origin + the baritone_state fields we need. Order is
    load order (the sidecar is append-per-tick), which is what segmentation needs."""
    out = []
    for rdir in sorted(capture_dir.glob("rollout-*")):
        sc = rdir / "sidecar.jsonl.gz"
        if not sc.exists():
            continue
        ticks = []
        try:
            with gzip.open(sc, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    bs = d.get("baritone_state") or {}
                    if not bs.get("pathing"):
                        continue
                    dest = bs.get("path_dest")
                    fwd = bs.get("path_fwd")
                    goal = _parse_goal(bs.get("goal"))
                    if not dest or not fwd or goal is None:
                        continue
                    ticks.append({
                        "origin": d["origin"],
                        "dest": tuple(dest),
                        "fwd": [tuple(n) for n in fwd],
                        "idx": bs.get("path_idx", 0),
                        "plen": bs.get("path_len", len(fwd)),
                        "goal": goal,
                    })
        except (EOFError, OSError, gzip.BadGzipFile):
            pass            # truncated tail of an in-flight rollout — keep clean prefix
        if len(ticks) >= 2:
            out.append((rdir.name, ticks))
    return out


# --- geometry ----------------------------------------------------------------
def _bearing(frm, to):
    """atan2 heading in the XZ plane from `frm` to `to`, or None if coincident."""
    dx, dz = to[0] - frm[0], to[2] - frm[2]
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(dz, dx)


def _dev_class(heading, goal_ang):
    """Heading minus goal bearing, wrapped to a 16-way deviation class centred on 8
    (=straight, head-at-goal). |class-8| = sectors off straight (0..8)."""
    rel = (heading - goal_ang) % (2 * math.pi)
    dev = int(round(rel / (2 * math.pi) * N_SECTORS)) % N_SECTORS
    return (dev + STRAIGHT) % N_SECTORS


def _sectors_off(dev_class):
    """Circular distance of a dev class from straight (0..8)."""
    d = abs(dev_class - STRAIGHT)
    return min(d, N_SECTORS - d)


# --- segmentation + measurement ----------------------------------------------
def _segment(ticks):
    """Split a rollout's tick stream into commit-runs and goal-epochs.

    Returns (runs, goal_changes) where runs is a list of (start_i, end_i) half-open
    index ranges over `ticks` sharing one path_dest, and goal_changes counts operator
    re-commands (path goal target changed). A recompute = the boundary BETWEEN two
    runs that share the same goal (a segment re-extension); a goal change starts a
    fresh run that is NOT counted as a recompute (it's the operator's input, a
    different residual)."""
    runs = []
    goal_changes = 0
    start = 0
    for i in range(1, len(ticks)):
        new_dest = ticks[i]["dest"] != ticks[i - 1]["dest"]
        new_goal = ticks[i]["goal"] != ticks[i - 1]["goal"]
        if new_goal:
            goal_changes += 1
        if new_dest or new_goal:
            runs.append((start, i))
            start = i
    runs.append((start, len(ticks)))
    return runs, goal_changes


def _within_run_freeness(ticks, runs):
    """For each run, what fraction of nodes in later ticks' path_fwd were ALREADY in
    the run-start committed segment (pure consumption, no churn). Coverage→1 means the
    committed future is fixed within the run = within-run bits ≈ 0. Averaged over all
    within-run later ticks (run-start tick excluded; runs of length 1 contribute none)."""
    covs = []
    for s, e in runs:
        committed = {tuple(n) for n in ticks[s]["fwd"]}
        if not committed:
            continue
        for j in range(s + 1, e):
            fwd = ticks[j]["fwd"]
            if not fwd:
                continue
            inside = sum(1 for n in fwd if tuple(n) in committed)
            covs.append(inside / len(fwd))
    return float(np.mean(covs)) if covs else float("nan"), len(covs)


def _recompute_residuals(ticks, runs):
    """One residual record per RECOMPUTE (run boundary with unchanged goal). The new
    committed heading = bearing(origin -> new path_dest) at the recompute tick; its
    deviation from the straight-line goal bearing is the residual the codec carries."""
    recs = []
    for k in range(1, len(runs)):
        s, _ = runs[k]
        cur, prev = ticks[s], ticks[s - 1]
        if cur["goal"] != prev["goal"]:
            continue                       # goal change, not a segment recompute
        goal_ang = _bearing(cur["origin"], cur["goal"])
        new_head = _bearing(cur["origin"], cur["dest"])
        old_head = _bearing(prev["origin"], prev["dest"])
        if goal_ang is None or new_head is None:
            continue
        dev = _dev_class(new_head, goal_ang)
        off = _sectors_off(dev)
        turn = (None if old_head is None
                else _sectors_off(_dev_class(new_head, old_head)))
        recs.append({
            "dev_class": dev,
            "sectors_off": off,
            "is_detour": off > 1,
            "turn_from_prev": turn,
            "dlen": cur["plen"] - prev["plen"],
            "run_ticks": runs[k - 1][1] - runs[k - 1][0],
        })
    return recs


def _entropy_bits(classes, k=N_SECTORS):
    """Empirical Shannon entropy (bits) of a list of integer class labels in [0,k)."""
    if not classes:
        return float("nan")
    counts = np.bincount(np.asarray(classes, dtype=int), minlength=k).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


# --- driver ------------------------------------------------------------------
def analyse(capture_dir: Path):
    rollouts = load_rollouts(capture_dir)
    if not rollouts:
        raise SystemExit(f"no usable rollouts under {capture_dir}")

    per_rollout = []
    all_run_lens, all_recs = [], []
    total_ticks = total_recomputes = total_goal_changes = 0
    free_cov_w, free_n = 0.0, 0

    for name, ticks in rollouts:
        runs, gchg = _segment(ticks)
        recs = _recompute_residuals(ticks, runs)
        cov, ncov = _within_run_freeness(ticks, runs)
        run_lens = [e - s for s, e in runs]
        all_run_lens.extend(run_lens)
        all_recs.extend(recs)
        total_ticks += len(ticks)
        total_recomputes += len(recs)
        total_goal_changes += gchg
        if not math.isnan(cov):
            free_cov_w += cov * ncov
            free_n += ncov
        per_rollout.append({
            "name": name, "ticks": len(ticks), "runs": len(runs),
            "recomputes": len(recs), "goal_changes": gchg,
            "mean_run_len": float(np.mean(run_lens)),
            "freeness": cov, "n_detour": sum(r["is_detour"] for r in recs),
        })

    run_lens = np.asarray(all_run_lens, dtype=float)
    devs = [r["dev_class"] for r in all_recs]
    detours = [r for r in all_recs if r["is_detour"]]
    dlens = np.asarray([r["dlen"] for r in all_recs], dtype=float)

    # rollout-balanced compression factor (don't let one long rollout dominate)
    rollout_mean_run = float(np.mean([p["mean_run_len"] for p in per_rollout]))
    recompute_rate = total_recomputes / total_ticks

    dev_entropy = _entropy_bits(devs)
    resid_bits_per_tick = dev_entropy * recompute_rate
    raw_bits_per_tick = math.log2(N_SECTORS)           # naive: send a sector every tick

    summary = {
        "capture": str(capture_dir),
        "n_rollouts": len(rollouts),
        "total_ticks": total_ticks,
        # (1) compression
        "compression": {
            "mean_run_len_pooled": float(run_lens.mean()),
            "median_run_len": float(np.median(run_lens)),
            "mean_run_len_rollout_balanced": rollout_mean_run,
            "max_run_len": float(run_lens.max()),
            "note": "mean run length == §20.0 commit-length compression factor on real nav data",
        },
        # (2) recompute rate
        "recompute_rate_per_tick": recompute_rate,
        "total_recomputes": total_recomputes,
        "total_goal_changes": total_goal_changes,
        # (3) within-run free-ness
        "within_run_freeness": (free_cov_w / free_n) if free_n else float("nan"),
        "within_run_n": free_n,
        # (4) residual decomposition
        "residual": {
            "detour_frac_of_recomputes": (len(detours) / len(all_recs)) if all_recs else float("nan"),
            "n_detour": len(detours),
            "mean_sectors_off": float(np.mean([r["sectors_off"] for r in all_recs])) if all_recs else float("nan"),
            "mean_dlen": float(dlens.mean()) if len(dlens) else float("nan"),
        },
        # (5) residual bits (marginal upper bound)
        "bits": {
            "dev_entropy_per_recompute": dev_entropy,
            "residual_bits_per_tick_marginal": resid_bits_per_tick,
            "raw_sector_bits_per_tick": raw_bits_per_tick,
            "compression_x_vs_raw_sector": raw_bits_per_tick / resid_bits_per_tick
            if resid_bits_per_tick > 0 else float("inf"),
            "note": "marginal entropy is the UPPER bound; Rung 2 conditions on plan-state to lower it",
        },
        "per_rollout": per_rollout,
        # histograms for the plot
        "hist": {
            "run_len_bins": _hist(run_lens, [0, 5, 10, 20, 40, 80, 160, 320, 1e9]),
            "dev_class_counts": np.bincount(np.asarray(devs, dtype=int),
                                            minlength=N_SECTORS).tolist() if devs else [],
            "sectors_off_counts": np.bincount(
                np.asarray([r["sectors_off"] for r in all_recs], dtype=int),
                minlength=9).tolist() if all_recs else [],
        },
    }
    return summary


def _hist(arr, edges):
    h, _ = np.histogram(arr, bins=edges)
    labels = [f"{int(edges[i])}-{int(edges[i+1]) if edges[i+1] < 1e8 else '+'}"
              for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": h.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description="§22 Rung 1 — path-codec residual")
    ap.add_argument("--capture", default="results/sprint21_visual/capture")
    ap.add_argument("--out", default="results/sprint22/rung1.json")
    args = ap.parse_args()

    s = analyse(Path(args.capture))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(s, indent=2))

    c = s["compression"]
    b = s["bits"]
    r = s["residual"]
    print(f"[path_codec] {s['n_rollouts']} rollouts, {s['total_ticks']} pathing ticks\n")
    print("=== §22 RUNG 1 — path-stream residual ===")
    print(f"(1) COMPRESSION  mean run = {c['mean_run_len_pooled']:.1f} ticks (pooled), "
          f"{c['mean_run_len_rollout_balanced']:.1f} (rollout-balanced), "
          f"median {c['median_run_len']:.0f}, max {c['max_run_len']:.0f}")
    print(f"    -> one transmitted event reproduces ~{c['mean_run_len_pooled']:.0f} ticks "
          f"(§20.0 commit-length factor, real nav data)")
    print(f"(2) RECOMPUTES   {s['total_recomputes']} segment re-extensions "
          f"= {s['recompute_rate_per_tick']*100:.2f}%/tick  |  "
          f"{s['total_goal_changes']} operator goal-changes (separate residual)")
    print(f"(3) WITHIN-RUN   committed-future coverage = {s['within_run_freeness']:.3f} "
          f"(→1 = within-run is pure consumption, ~0 bits)  [n={s['within_run_n']}]")
    print(f"(4) RESIDUAL     {r['detour_frac_of_recomputes']*100:.1f}% of recomputes are DETOURS "
          f"(>1 sector off goal); mean {r['mean_sectors_off']:.2f} sectors off; "
          f"mean Δlen {r['mean_dlen']:+.1f} nodes")
    print(f"(5) BITS         {b['dev_entropy_per_recompute']:.2f} b/recompute (marginal) "
          f"× {s['recompute_rate_per_tick']:.4f} = "
          f"{b['residual_bits_per_tick_marginal']:.4f} b/tick  "
          f"(vs {b['raw_sector_bits_per_tick']:.0f} b/tick raw = "
          f"{b['compression_x_vs_raw_sector']:.0f}× ; UPPER bound, Rung 2 lowers it)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
