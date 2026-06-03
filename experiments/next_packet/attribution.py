"""Attribute server corrective feedback to the outbound actions that provoked it.

The homunculus packet recorder now writes a single time-ordered JSONL stream
that interleaves two kinds of line (see homunculus PacketRecorder /
ServerFeedbackTap):

  {"dir":"out", "ts_ms":..., "tick":..., "id":"minecraft:move_player_pos",
                "fields":{...}, "obs":{...}}            # the action the client emitted
  {"dir":"in",  "ts_ms":..., "tick":..., "id":"minecraft:player_position",
                "fields":{"teleport_id":..,"x":..,"relatives":[..],..}, "obs":{...}}
                                                        # the server's correction (rubber-band)

This module joins the two: for every outbound action it labels whether a
corrective packet arrived within N ticks, and for every correction it names the
most recent preceding action as its best-guess cause. It is deliberately the
*measurement* layer — it computes labels and rates, it does NOT train anything.

Why both directions of the join (per the brief):
  - per-outbound (effect view): "did this action get corrected within N ticks?"
    These are the labels a future online-learning system would penalize on.
  - per-correction (cause view): "which action most likely caused this snap-back?"
    The attribution is best-effort — the server's reply is not tightly aligned
    in time with the offending packet, so we report the *distribution* of
    latencies, not a single point estimate, and we flag corrections we cannot
    attribute (none in window) or that are ambiguous (several candidates).

Windowed labels (N=5, 20, 100 ticks by default) are emitted from the start
because temporal cause here is messy: a correction can lag its trigger by a
variable number of ticks, so a single window would hide the structure.

Caveat baked in: this is a *constraint* signal (stay inside the server's
movement envelope), not a *task* signal. A slightly-too-fast legit walk and a
genuine teleport both show up as a "correction". We add a heuristic `kind`
(rubberband vs teleport) from the relative-flags + snap magnitude, but it is a
hint, not ground truth.

Usage:
  # offline: join a recording and write a summary
  python -m experiments.next_packet.attribution path/to/recording.jsonl \\
         --windows 5,20,100 --out results/feedback/attr.json

  # live: poll a fleet agent's rubber-band counters during a sweep
  python -m experiments.next_packet.attribution --watch --port 25570 --interval 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

MS_PER_TICK = 50.0  # MC server tick = 50ms; used only to derive ms windows / fallback deltas

# Outbound packet ids that can plausibly *cause* a movement correction. Used to
# restrict the cause-view attribution; the effect-view labels every outbound.
MOVEMENT_OUT_IDS = {
    "minecraft:move_player_pos",
    "minecraft:move_player_pos_rot",
    "minecraft:move_player_rot",
    "minecraft:move_player_status_only",
    "minecraft:player_input",
    "minecraft:player_command",
}

# Inbound corrective ids (mirror of homunculus ServerFeedbackExtractor).
RUBBERBAND_ID = "minecraft:player_position"
MOTION_ID = "minecraft:set_entity_motion"


@dataclass
class Line:
    dir: str
    tick: int | None
    ts_ms: int | None
    id: str
    fields: dict[str, Any]
    obs: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)


def load(path: str) -> list[Line]:
    """Parse a recorder JSONL into normalized Lines, sorted causally.

    Legacy recordings predate the ``dir`` field; those lines are all outbound
    actions, so a missing ``dir`` defaults to "out" (and there will be no "in"
    lines to join against — the summary will simply show zero corrections).
    """
    out: list[Line] = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.append(
                Line(
                    dir=d.get("dir", "out"),
                    tick=d.get("tick"),
                    ts_ms=d.get("ts_ms"),
                    id=d.get("id", "?"),
                    fields=d.get("fields") or {},
                    obs=d.get("obs"),
                    raw=d,
                )
            )
    # Stable sort by (tick, ts_ms) so the greedy nearest-cause scan is causal.
    out.sort(key=lambda x: (x.tick if x.tick is not None else math.inf,
                            x.ts_ms if x.ts_ms is not None else math.inf))
    return out


def _delta_ticks(out_line: Line, corr: Line) -> float | None:
    if out_line.tick is not None and corr.tick is not None:
        return corr.tick - out_line.tick
    if out_line.ts_ms is not None and corr.ts_ms is not None:
        return (corr.ts_ms - out_line.ts_ms) / MS_PER_TICK
    return None


def _delta_ms(out_line: Line, corr: Line) -> float | None:
    if out_line.ts_ms is not None and corr.ts_ms is not None:
        return corr.ts_ms - out_line.ts_ms
    return None


def _snap_magnitude(corr: Line) -> float | None:
    """Distance between the corrected pose and the client's believed pose.

    Uses the correction line's obs (the player's pose at receive time). Only
    meaningful for player_position corrections. None when obs or coords absent.
    """
    if corr.id != RUBBERBAND_ID or corr.obs is None:
        return None
    try:
        dx = corr.fields["x"] - corr.obs["x"]
        dy = corr.fields["y"] - corr.obs["y"]
        dz = corr.fields["z"] - corr.obs["z"]
    except (KeyError, TypeError):
        return None
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _kind(corr: Line, mag: float | None) -> str:
    """Heuristic label: rubber-band (anti-cheat snap-back) vs teleport vs motion.

    Hint only. A rubber-band is a small absolute position correction; a teleport
    (spawn, /tp, dimension change) is a large jump. Threshold is conservative.
    """
    if corr.id == MOTION_ID:
        return "motion_override"
    if mag is None:
        return "position_unknown"
    return "rubberband" if mag < 4.0 else "teleport"


def attribute(lines: list[Line], windows: list[int]) -> dict[str, Any]:
    outs = [ln for ln in lines if ln.dir == "out"]
    corrs = [ln for ln in lines if ln.dir == "in"]
    maxwin = max(windows) if windows else 100

    # --- effect view: per-outbound, was it corrected within each window? ---
    # corrs already sorted causally; for each out, scan forward for the first
    # correction with 0 <= delta_tick <= maxwin and delta_ms >= 0 (causal).
    per_out_rate = {w: defaultdict(lambda: [0, 0]) for w in windows}  # id -> [corrected, total]
    latencies_ticks: list[float] = []
    latencies_ms: list[float] = []
    for o in outs:
        # nearest correction at or after this action within the max window
        nearest_dt: float | None = None
        nearest_dms: float | None = None
        for c in corrs:
            dt = _delta_ticks(o, c)
            if dt is None or dt < 0 or dt > maxwin:
                continue
            dms = _delta_ms(o, c)
            if dms is not None and dms < 0:
                continue
            if nearest_dt is None or dt < nearest_dt:
                nearest_dt, nearest_dms = dt, dms
        for w in windows:
            rec = per_out_rate[w][o.id]
            rec[1] += 1
            if nearest_dt is not None and nearest_dt <= w:
                rec[0] += 1
        if nearest_dt is not None:
            latencies_ticks.append(nearest_dt)
            if nearest_dms is not None:
                latencies_ms.append(nearest_dms)

    # --- cause view: per-correction, name the most recent preceding movement ---
    cause_tally: dict[str, int] = defaultdict(int)
    unattributed = 0
    ambiguous = 0
    kinds: dict[str, int] = defaultdict(int)
    mags: list[float] = []
    for c in corrs:
        mag = _snap_magnitude(c)
        if mag is not None:
            mags.append(mag)
        kinds[_kind(c, mag)] += 1
        # candidate causes: movement outs within [tick-maxwin, tick]
        cands = []
        for o in outs:
            if o.id not in MOVEMENT_OUT_IDS:
                continue
            dt = _delta_ticks(o, c)
            if dt is None or dt < 0 or dt > maxwin:
                continue
            dms = _delta_ms(o, c)
            if dms is not None and dms < 0:
                continue
            cands.append((dt, o))
        if not cands:
            unattributed += 1
            continue
        cands.sort(key=lambda t: t[0])  # smallest delta = most recent = best guess
        if len(cands) > 1 and cands[0][0] == cands[1][0]:
            ambiguous += 1
        cause_tally[cands[0][1].id] += 1

    return {
        "n_out": len(outs),
        "n_corrections": len(corrs),
        "rubberbands": sum(1 for c in corrs if c.id == RUBBERBAND_ID),
        "motion_overrides": sum(1 for c in corrs if c.id == MOTION_ID),
        "windows_ticks": windows,
        "effect_view": {
            str(w): {
                pid: {
                    "corrected": rec[0],
                    "total": rec[1],
                    "rate": (rec[0] / rec[1]) if rec[1] else 0.0,
                }
                for pid, rec in sorted(per_out_rate[w].items())
            }
            for w in windows
        },
        "latency_ticks": _dist(latencies_ticks),
        "latency_ms": _dist(latencies_ms),
        "cause_view": {
            "by_outbound_id": dict(sorted(cause_tally.items(), key=lambda kv: -kv[1])),
            "unattributed": unattributed,
            "ambiguous": ambiguous,
        },
        "correction_kinds": dict(kinds),
        "snap_magnitude_blocks": _dist(mags),
    }


def _dist(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        k = p * (n - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {
        "n": n,
        "min": s[0],
        "median": pct(0.5),
        "p90": pct(0.9),
        "max": s[-1],
        "mean": sum(s) / n,
    }


def _print_summary(summ: dict[str, Any]) -> None:
    print(f"outbound actions : {summ['n_out']}")
    print(f"corrections      : {summ['n_corrections']} "
          f"(rubberband={summ['rubberbands']}, motion={summ['motion_overrides']})")
    print(f"correction kinds : {summ['correction_kinds']}")
    lat = summ["latency_ticks"]
    if lat:
        print(f"latency (ticks)  : median={lat['median']:.1f} p90={lat['p90']:.1f} "
              f"max={lat['max']:.0f} (n={lat['n']})")
    mag = summ["snap_magnitude_blocks"]
    if mag:
        print(f"snap (blocks)    : median={mag['median']:.3f} p90={mag['p90']:.3f} "
              f"max={mag['max']:.3f}")
    print("\neffect view — corrected-within-N-ticks, by outbound packet:")
    for w in summ["windows_ticks"]:
        rows = summ["effect_view"][str(w)]
        print(f"  N={w} ticks:")
        for pid, r in rows.items():
            print(f"    {pid:32s} {r['corrected']:5d}/{r['total']:<5d} "
                  f"rate={r['rate']:.3f}")
    cv = summ["cause_view"]
    print(f"\ncause view — best-guess trigger of each correction "
          f"(unattributed={cv['unattributed']}, ambiguous={cv['ambiguous']}):")
    for pid, n in cv["by_outbound_id"].items():
        print(f"    {pid:32s} {n}")


def _watch(port: int, interval: float) -> None:
    """Poll /packets/feedback and print the running rubber-band counters."""
    import requests  # local import: offline path needs no network dep

    base = f"http://127.0.0.1:{port}"
    print(f"watching {base}/packets/feedback every {interval}s (ctrl-c to stop)")
    prev_rb = 0
    while True:
        try:
            r = requests.get(f"{base}/packets/feedback", timeout=4)
            d = r.json()
            rb = d.get("rubberbands", 0)
            mo = d.get("motion_overrides", 0)
            drb = rb - prev_rb
            prev_rb = rb
            print(f"  rubberbands={rb} (+{drb})  motion_overrides={mo}  total={d.get('total', 0)}")
        except Exception as e:  # noqa: BLE001 — watcher must not die on a transient
            print(f"  [poll failed: {e}]")
        time.sleep(interval)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?", help="recorder JSONL to attribute")
    ap.add_argument("--windows", default="5,20,100",
                    help="comma-separated tick windows (default 5,20,100)")
    ap.add_argument("--out", help="write the full summary JSON here")
    ap.add_argument("--watch", action="store_true",
                    help="live mode: poll /packets/feedback instead of parsing a file")
    ap.add_argument("--port", type=int, default=25570, help="agent homunculus port (watch mode)")
    ap.add_argument("--interval", type=float, default=2.0, help="poll interval s (watch mode)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.watch:
        try:
            _watch(args.port, args.interval)
        except KeyboardInterrupt:
            print("\nstopped.")
        return 0

    if not args.recording:
        ap.error("a recording path is required (or use --watch)")
    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    lines = load(args.recording)
    summ = attribute(lines, windows)
    _print_summary(summ)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summ, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
