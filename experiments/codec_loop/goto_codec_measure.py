#!/usr/bin/env python3
"""§20.0 measure — the goal codec: stream-vs-goal compression + the OVERRIDE moat.

Consumes the two capture roots written by goto_override_capture.py
(results/sprint20/{completion,override}/rollout-*/) and produces the §20.0 PRIMARY
deliverable (neural_interface.md §20.0):

  PART A — compression headline (predict the PLAN, not the packet stream).
    A move SEGMENT is the run of move packets under one committed goal (one g_t
    string). For each segment:
      * stream_bits = Σ over its move packets of the §16 obs-relative per-packet
        code cost (pos zero_preserving delta + yaw/pitch residual + the has_*/
        on_ground flags), entropy-coded at `--bits` (default 5 = the §16.1
        zero_preserving@b5 baseline-to-beat). This is the cost of sending EVERY
        move — the §16 per-tick codec applied to the whole stream.
      * goal_bits = the cost of the ONE goal that generates the whole stream,
        two honest codings:
          - index  : log2(#distinct goals in the rollout) — a pointer into the
                     controller's small waypoint set (the navigation analog of
                     §17.2.2's index-into-obs entity pointer).
          - delta  : the GoalBlock coded as a quantized obs-relative 3-delta from
                     the segment-start position (the §17.2.1 block_pos reparam).
    Headline = stream_bits / goal_bits, and how it SCALES with segment length: the
    longer the controller commits, the more the stream compresses to its fixed
    goal. This is the plan-level headroom §16 never measured (it coded each packet
    in isolation and found a per-tick null; the stream pays ~2 b/pkt PER PACKET,
    the goal pays it ONCE).

  PART B — the override moat (THE HEADLINE; corrigibility made quantitative).
    Runs the validated §13.2 rel-crossover instrument (rung_c_transition) VERBATIM
    on each root. By construction every seam in completion/ is a completion (g_t
    flips with the body AT REST at the prior goal) and every seam in override/ is a
    forced override (g_t flips with the body AT SPEED mid-path). The crossover tick
    is the handover latency (offset from the g_t-issue tick = offset 0, since the
    capture stamps g_t AT the transition). Then:
        moat = override_handover_latency − completion_handover_latency
    Strictly positive = a committed plan carries real interruption inertia (the
    recurrence/corrigibility boundary made quantitative). ≈ 0 = corrigible-by-
    default even when stateful (an equally publishable null). Completion latency
    here is the live-recaptured analog of §13.2's ~6.4 ticks on frozen rollouts.

No new homunculus code; reuses offline_fidelity.load helpers, the quantize codec,
and rung_c_transition. Run AFTER goto_override_capture.py.

Usage:
    .venv/bin/python -m experiments.codec_loop.goto_codec_measure \
        --root results/sprint20 --bits 5 --margin 40 --holdout 10 --bin 4 \
        --out results/sprint20/measure.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

from craft.codec import encode
from craft.codec.move import MoveAction
from experiments.codec_loop.obsrel import wrap180
from experiments.codec_loop.quantize import (
    POS_MODES, POS_RANGE, quant_scalar_zero_preserving,
)

MOVE_TYPES = {
    "minecraft:move_player_pos", "minecraft:move_player_rot",
    "minecraft:move_player_pos_rot", "minecraft:move_player_status_only",
}
TP_THRESHOLD = 10.0
ROT_RANGE = 180.0


# --- load: per-rollout move packets carrying (codes, segment id, pos) ---------
def _load_rollout(path: str, bits: int):
    """Return per-move-packet rows: {seg, codes:{field:code}, flags:{...}, pos}.
    seg = 0-based segment index (new segment when obs.g_t changes). Skips TP
    artifacts (stale obs vs teleported packet) like offline_fidelity."""
    rows = []
    goals = []          # distinct goal strings in order of appearance
    pos_q = POS_MODES["zero_preserving"]
    seg = -1
    prev_g = object()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            obs = d.get("obs", {}) or {}
            g = obs.get("g_t")
            if g != prev_g:
                seg += 1
                prev_g = g
                if g not in goals:
                    goals.append(g)
            pid = d.get("id")
            if pid not in MOVE_TYPES:
                continue
            fields = d.get("fields", {}) or {}
            if fields.get("has_pos"):
                try:
                    dmax = max(abs(float(fields["x"]) - float(obs["x"])),
                               abs(float(fields["y"]) - float(obs["y"])),
                               abs(float(fields["z"]) - float(obs["z"])))
                except (KeyError, TypeError, ValueError):
                    dmax = 0.0
                if dmax >= TP_THRESHOLD:
                    continue
            action = encode(pid, fields, obs)
            if not isinstance(action, MoveAction):
                continue
            codes: dict = {}
            if action.pos is not None:
                for ax, c in zip("xyz", action.pos):
                    codes[f"pos_{ax}"] = pos_q(c, -POS_RANGE, POS_RANGE, bits)[1]
            if action.rot is not None:
                yaw, pitch = action.rot
                oy, op = float(obs["yaw"]), float(obs["pitch"])
                codes["yaw"] = quant_scalar_zero_preserving(
                    wrap180(yaw - oy), -ROT_RANGE, ROT_RANGE, bits)[1]
                codes["pitch"] = quant_scalar_zero_preserving(
                    pitch - op, -ROT_RANGE, ROT_RANGE, bits)[1]
            flags = {"has_pos": bool(fields.get("has_pos")),
                     "has_rot": bool(fields.get("has_rot")),
                     "on_ground": bool(fields.get("on_ground"))}
            rows.append({"seg": seg,
                         "codes": codes, "flags": flags,
                         "pos": (float(obs.get("x", 0.0)), float(obs.get("y", 0.0)),
                                 float(obs.get("z", 0.0)))})
    return rows, goals


# --- entropy bookkeeping ------------------------------------------------------
def _entropy_tables(all_rows):
    """Empirical -log2 p tables, one per code field and per boolean flag, pooled
    over the whole capture (the entropy coder's shared model)."""
    field_counts: dict = defaultdict(Counter)
    flag_counts: dict = defaultdict(Counter)
    for r in all_rows:
        for k, c in r["codes"].items():
            field_counts[k][c] += 1
        for k, v in r["flags"].items():
            flag_counts[k][v] += 1
    field_bits, flag_bits = {}, {}
    for k, ctr in field_counts.items():
        n = sum(ctr.values())
        field_bits[k] = {code: -math.log2(cnt / n) for code, cnt in ctr.items()}
    for k, ctr in flag_counts.items():
        n = sum(ctr.values())
        flag_bits[k] = {v: -math.log2(cnt / n) for v, cnt in ctr.items()}
    return field_bits, flag_bits


def _packet_bits(r, field_bits, flag_bits) -> float:
    b = 0.0
    for k, c in r["codes"].items():
        b += field_bits[k][c]
    for k, v in r["flags"].items():
        b += flag_bits[k][v]
    return b


# --- PART A: compression ------------------------------------------------------
def _goal_delta_bits(bits_per_axis: int = 6) -> float:
    """Code the goal as a quantized obs-relative 3-delta (the §17.2.1 reparam):
    block-resolution over the leg range. A fixed per-axis budget; what matters is
    it is O(10s of bits) ONCE, vs O(bits) PER PACKET for the stream."""
    return 3.0 * bits_per_axis


def compress(root: str, bits: int) -> dict:
    files = sorted(glob.glob(f"{root}/rollout-*/packets.jsonl"))
    per_rollout = []
    all_rows = []
    rollouts = []
    for f in files:
        rows, goals = _load_rollout(f, bits)
        rollouts.append((f, rows, goals))
        all_rows.extend(rows)
    if not all_rows:
        return {"root": root, "n_move_packets": 0}
    field_bits, flag_bits = _entropy_tables(all_rows)
    bpp_mean = sum(_packet_bits(r, field_bits, flag_bits) for r in all_rows) / len(all_rows)

    seg_records = []
    for f, rows, goals in rollouts:
        n_goals = max(1, len(goals))
        index_bits = math.log2(n_goals) if n_goals > 1 else 1.0
        by_seg: dict = defaultdict(list)
        for r in rows:
            by_seg[r["seg"]].append(r)
        rollout_segs = []
        for _s, sr in sorted(by_seg.items()):
            if not sr:
                continue
            stream_bits = sum(_packet_bits(r, field_bits, flag_bits) for r in sr)
            delta_bits = _goal_delta_bits()
            rollout_segs.append({
                "n_move": len(sr),
                "stream_bits": round(stream_bits, 1),
                "goal_index_bits": round(index_bits, 2),
                "goal_delta_bits": round(delta_bits, 1),
                "ratio_vs_index": round(stream_bits / index_bits, 1),
                "ratio_vs_delta": round(stream_bits / delta_bits, 1),
            })
        per_rollout.append({"rollout": os.path.basename(os.path.dirname(f)),
                            "n_goals": n_goals, "segments": rollout_segs})
        seg_records.extend(rollout_segs)

    # aggregate (exclude degenerate 0-move segments from ratio stats)
    nz = [s for s in seg_records if s["n_move"] > 0]
    tot_stream = sum(s["stream_bits"] for s in nz)
    tot_index = sum(s["goal_index_bits"] for s in nz)
    tot_delta = sum(s["goal_delta_bits"] for s in nz)
    lens = [s["n_move"] for s in nz]
    return {
        "root": root, "bits": bits,
        "n_move_packets": len(all_rows), "n_segments": len(nz),
        "mean_bits_per_packet": round(bpp_mean, 3),
        "mean_segment_len_moves": round(sum(lens) / len(lens), 1) if lens else 0,
        "total_stream_bits": round(tot_stream, 1),
        "total_goal_index_bits": round(tot_index, 1),
        "total_goal_delta_bits": round(tot_delta, 1),
        "ratio_stream_over_index": round(tot_stream / tot_index, 1) if tot_index else None,
        "ratio_stream_over_delta": round(tot_stream / tot_delta, 1) if tot_delta else None,
        "per_rollout": per_rollout,
    }


# --- PART B: override moat (reuse rung_c_transition verbatim) ------------------
def crossover(root: str, margin: int, holdout: int, binw: int, seed: int) -> dict:
    """Shell out to rung_c_transition on `root`; return its result JSON."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out = tf.name
    cmd = [sys.executable, "-m", "experiments.next_packet.rung_c_transition",
           "--data", root, "--out", out,
           "--margin", str(margin), "--holdout", str(holdout),
           "--bin", str(binw), "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        res = json.load(open(out))
    except (ValueError, FileNotFoundError):
        res = {"error": "rung_c_transition produced no result",
               "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-1000:]}
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="§20.0 goal-codec measure")
    ap.add_argument("--root", default="results/sprint20")
    ap.add_argument("--bits", type=int, default=5,
                    help="obsrel quant bit-depth (5 = §16.1 zero_preserving baseline)")
    ap.add_argument("--margin", type=int, default=40, help="seam half-window (ticks)")
    ap.add_argument("--holdout", type=int, default=10,
                    help="ticks from a segment's own boundaries excluded from training")
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp_root = os.path.join(args.root, "completion")
    over_root = os.path.join(args.root, "override")

    print("[measure] PART A — compression (stream vs goal bits)")
    comp_A = compress(comp_root, args.bits)
    over_A = compress(over_root, args.bits)

    print("[measure] PART B — handover crossover (override vs completion)")
    comp_B = crossover(comp_root, args.margin, args.holdout, args.bin, args.seed)
    over_B = crossover(over_root, args.margin, args.holdout, args.bin, args.seed)

    xo_comp = comp_B.get("crossover_tick")
    xo_over = over_B.get("crossover_tick")
    moat = (xo_over - xo_comp) if (isinstance(xo_comp, (int, float))
                                   and isinstance(xo_over, (int, float))
                                   and xo_comp == xo_comp and xo_over == xo_over) else None

    result = {
        "root": args.root, "bits": args.bits,
        "compression": {"completion": comp_A, "override": over_A},
        "crossover": {
            "completion": {k: comp_B.get(k) for k in
                           ("crossover_tick", "crossover_sec", "boundaries_used",
                            "decoder_interior_train_acc", "crossover_tick_long_new",
                            "crossover_tick_short_new", "n_boundaries")},
            "override": {k: over_B.get(k) for k in
                         ("crossover_tick", "crossover_sec", "boundaries_used",
                          "decoder_interior_train_acc", "crossover_tick_long_new",
                          "crossover_tick_short_new", "n_boundaries")},
        },
        "moat_ticks": round(moat, 2) if moat is not None else None,
        "moat_sec": round(moat / 20.0, 3) if moat is not None else None,
    }

    out = args.out or os.path.join(args.root, "measure.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)

    print("\n" + "=" * 70)
    print("§20.0 — THE GOAL CODEC: predict the PLAN, not the packet stream")
    print("=" * 70)
    print("\nPART A — compression (the move stream compresses to its goal):")
    for name, A in (("completion", comp_A), ("override", over_A)):
        if A.get("n_move_packets"):
            print(f"  {name:>10}: {A['n_move_packets']} moves / {A['n_segments']} segs, "
                  f"{A['mean_bits_per_packet']} b/pkt, seg≈{A['mean_segment_len_moves']} moves")
            print(f"              stream={A['total_stream_bits']}b  "
                  f"goal(index)={A['total_goal_index_bits']}b → {A['ratio_stream_over_index']}× | "
                  f"goal(delta)={A['total_goal_delta_bits']}b → {A['ratio_stream_over_delta']}×")
    print("\nPART B — handover latency (the override moat):")
    print(f"  completion crossover = {xo_comp} ticks "
          f"({comp_B.get('crossover_sec')}s)  "
          f"[decoder_acc={comp_B.get('decoder_interior_train_acc')}, "
          f"seams={comp_B.get('boundaries_used')}]")
    print(f"  override   crossover = {xo_over} ticks "
          f"({over_B.get('crossover_sec')}s)  "
          f"[decoder_acc={over_B.get('decoder_interior_train_acc')}, "
          f"seams={over_B.get('boundaries_used')}]")
    if moat is not None:
        verdict = ("MOAT (committed plan resists interrupt)" if moat > 1.0
                   else "NO MOAT (corrigible-by-default even when stateful)")
        print(f"  -> MOAT = override − completion = {moat:+.1f} ticks "
              f"({moat/20.0:+.2f}s)  {verdict}")
    else:
        print("  -> MOAT: could not compute (a crossover was NaN; see measure.json)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
