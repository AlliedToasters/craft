"""Sprint A step 1: offline decode(quantize(encode(x))) fidelity gate.

Before touching the wire, measure how much each bit-budget corrupts the
reconstructed movement values, over the frozen corpus. This is the precursor
to the live parity sweep: the offline gate answers "at B bits, how far does the
reconstructed wire value drift from the original?"; the live sweep then answers
"does that drift change behavior?" (run via the §14 Rung-2 harness).

Per the brief, Sprint A picks ONE target: the move family (volume-dominant).
Error is pooled over all move packets (so it is naturally volume-weighted) with
TP/spawn discontinuities (|pos delta| >= 10 blocks, an obs/packet staleness
artifact of the *recording* -- the live wire sees fresh per-tick obs) excluded
and counted.

Usage:
    .venv/bin/python -m experiments.codec_loop.offline_fidelity            # narrated
    .venv/bin/python -m experiments.codec_loop.offline_fidelity --set combat
    .venv/bin/python -m experiments.codec_loop.offline_fidelity --json     # -> /tmp/offline_fidelity.json
"""

from __future__ import annotations

import glob
import json
import math

from craft.codec import decode, encode
from craft.codec.move import MoveAction
from experiments.codec_loop.quantize import float_bits, quantize_move

SETS = {
    "narrated": "results/frozen_narrated",
    "combat": "results/frozen_combat",
    "dryrun": "results/frozen_dryrun",
}
MOVE_TYPES = {
    "minecraft:move_player_pos", "minecraft:move_player_rot",
    "minecraft:move_player_pos_rot", "minecraft:move_player_status_only",
}
# uniform bits/field; the knee is read off this curve
BIT_LEVELS = [12, 10, 8, 7, 6, 5, 4, 3, 2]
TP_THRESHOLD = 10.0  # blocks; |pos delta| beyond this = TP/spawn artifact


def _packet_files(root: str):
    return sorted(glob.glob(f"{root}/rollout-*/packets.jsonl"))


def _wrap_deg(d: float) -> float:
    """Signed shortest angular difference, in (-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def _p99_abs(vals):
    if not vals:
        return float("nan")
    s = sorted(abs(v) for v in vals)
    return s[min(len(s) - 1, int(0.99 * len(s)))]


def load_move_actions(root: str):
    """Yield (MoveAction, obs, original_fields) for every non-TP move packet."""
    out = []
    n_tp = 0
    for pf in _packet_files(root):
        with open(pf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid = d.get("id")
                if pid not in MOVE_TYPES:
                    continue
                fields = d.get("fields", {}) or {}
                obs = d.get("obs", {}) or {}
                if fields.get("has_pos"):
                    # skip TP/spawn artifacts (stale obs vs teleported packet)
                    try:
                        dmax = max(abs(float(fields["x"]) - float(obs["x"])),
                                   abs(float(fields["y"]) - float(obs["y"])),
                                   abs(float(fields["z"]) - float(obs["z"])))
                    except (KeyError, TypeError, ValueError):
                        dmax = 0.0
                    if dmax >= TP_THRESHOLD:
                        n_tp += 1
                        continue
                action = encode(pid, fields, obs)
                if isinstance(action, MoveAction):
                    out.append((action, obs, fields))
    return out, n_tp


def sweep(root: str):
    actions, n_tp = load_move_actions(root)
    rows = []
    for b in BIT_LEVELS:
        pos_err, yaw_err, pitch_err, fbits = [], [], [], []
        for action, obs, orig in actions:
            q = quantize_move(action, pos_bits=b, yaw_bits=b, pitch_bits=b)
            rec = decode(q, obs)
            fbits.append(float_bits(action, pos_bits=b, yaw_bits=b, pitch_bits=b))
            if orig.get("has_pos"):
                for ax in ("x", "y", "z"):
                    pos_err.append(rec[ax] - float(orig[ax]))
            if orig.get("has_rot"):
                yaw_err.append(_wrap_deg(rec["yaw"] - float(orig["yaw"])))
                pitch_err.append(rec["pitch"] - float(orig["pitch"]))
        rows.append({
            "bits_per_field": b,
            "float_bits_per_packet": round(sum(fbits) / len(fbits), 2) if fbits else 0,
            "pos_rmse_blocks": round(_rms(pos_err), 5),
            "pos_p99_abs": round(_p99_abs(pos_err), 5),
            "yaw_rmse_deg": round(_rms(yaw_err), 4),
            "yaw_p99_abs": round(_p99_abs(yaw_err), 4),
            "pitch_rmse_deg": round(_rms(pitch_err), 4),
            "pitch_p99_abs": round(_p99_abs(pitch_err), 4),
        })
    return {"n_move_packets": len(actions), "n_tp_excluded": n_tp, "rows": rows}


def _print_report(name, res):
    print(f"\n========== {name}: offline fidelity "
          f"(n={res['n_move_packets']} move pkts, {res['n_tp_excluded']} TP excluded) ==========")
    print(f"{'bits':>4} {'fbits/pkt':>9} | {'posRMSE':>8} {'posP99':>8} | "
          f"{'yawRMSE':>8} {'yawP99':>8} | {'pitRMSE':>8} {'pitP99':>8}")
    for r in res["rows"]:
        print(f"{r['bits_per_field']:>4} {r['float_bits_per_packet']:>9} | "
              f"{r['pos_rmse_blocks']:>8.4f} {r['pos_p99_abs']:>8.4f} | "
              f"{r['yaw_rmse_deg']:>8.3f} {r['yaw_p99_abs']:>8.3f} | "
              f"{r['pitch_rmse_deg']:>8.3f} {r['pitch_p99_abs']:>8.3f}")


if __name__ == "__main__":
    import sys
    as_json = "--json" in sys.argv
    which = "narrated"
    if "--set" in sys.argv:
        which = sys.argv[sys.argv.index("--set") + 1]
    if as_json:
        res = {name: sweep(root) for name, root in SETS.items()
               if glob.glob(f"{root}/rollout-*")}
        json.dump(res, open("/tmp/offline_fidelity.json", "w"), indent=1)
        print("wrote /tmp/offline_fidelity.json")
    else:
        _print_report(which, sweep(SETS[which]))
