#!/usr/bin/env python3
"""§20.1a — neural takes the wheel on NAVIGATION (MVP authority-loop, live).

The §19 rung one step up the embodiment ladder. §19 served the g_t-prior as the
live CONTROLLER of a MEMORYLESS decision (the attack target) and proved it on the
server by whose-HP-drops. §20.1 does it for the STATEFUL decision — the Baritone
GOAL — and proves it by **whose-waypoint-the-body-converges-on**.

Architecture (neural_interface.md §20.1, MVP staging):
  * The decision is a LATENT goal: it lives in Baritone's customGoalProcess, on NO
    move packet, so the wheel is at the CONTROL layer, not a wire rewrite. The
    served codec ISSUES the goal via the §20.0 stop+repath override (the lock-
    bypassing /baritone/stop + re-path; /baritone/goto itself holds SESSION_LOCK
    and blocks). A control-loop driver — poll the authority command, decide,
    command Baritone — no packet sidecar.
  * g_t = the authority interface. `gt_override` decouples the served goal from the
    operator's command, EXACTLY like §19's filter_prior gt_override (None = honor
    the operator; set = override). MVP: the "decision" is the trivial g_t→beacon
    map; §20.1b grafts a trained nav goal-prior into this same rig (only the policy
    block swaps).
  * Scene = two typed beacons (the §19/§20.0 duel geometry): beacon A (cow) at +D
    on X, beacon B (pig) at -D on X. The body walks to whichever beacon the served
    goal selects; whose-waypoint = the nearer beacon at rest.

Note the honest asymmetry from §19: Baritone does NOT autonomously navigate (it only
goes where commanded), unlike KillAura which autonomously swings. So §20.1's "take
the wheel" is less a wresting from an active competing heuristic and more "the codec
is the goal source AND it is corrigible — it obeys a g_t change mid-commitment." The
operator's command is the gt_override=None baseline the codec honors; setting
gt_override is the override.

Three measurements:
  TEST A (effectiveness): gt_override=None, the codec honors the operator command —
    operator=A → body converges A; operator=B → body converges B. The wheel
    faithfully executes the commanded goal.
  TEST B (corrigibility, HEADLINE): operator command FIXED at A; gt_override=A →
    body→A (codec agrees); gt_override=B → body→B (codec OVERRIDES the operator).
    whose-waypoint flips with the operator command unchanged = the §19 passive_hit
    1.0→0.0 analog at the plan level.
  LIVE MOAT: operator=A, gt_override=None → body heads to A; mid-path flip
    gt_override=B and time ticks-until-the-body-changes-course, decomposed into
    DECISION LAG (controller notices g_t at its poll cadence ≤ cadence) + BODY
    REDIRECT (stop+repath, the §20.0 ~2t). Swept over cadence to show the moat is
    set by the controller's DECISION CADENCE, not body inertia (§20.0 found no
    body-level moat) — the corrigibility property §20.0's offline decoder, having
    instantaneous g_t, could not see.

No new homunculus code. Run on agent0.

Usage:
    .venv/bin/python -m experiments.codec_loop.nav_wheel \
        --port 25570 --reps 4 --dist 12 --out results/sprint20/nav_wheel.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

from experiments.codec_loop.aim_carrier import _relay
from experiments.codec_loop.run_rungs import _http, _resolve_base, _wait_in_world
from experiments.codec_loop.filter_capture import _clear, _position

BEACON_A = "cow"     # waypoint A at +D
BEACON_B = "pig"     # waypoint B at -D
_TAGS = "{NoAI:1b,Silent:1b,PersistenceRequired:1b,Invulnerable:1b," \
        "Attributes:[{Name:\"minecraft:generic.knockback_resistance\",Base:1.0}]}"


def _summon_beacon(relay: str, species: str, x: float, y: float, z: float) -> None:
    _relay(relay, f"summon minecraft:{species} {x:.2f} {y:.2f} {z:.2f} {_TAGS}")


def _dist_xz(x: float, z: float, wp) -> float:
    return math.hypot(x - wp[0], z - wp[2])


def _stop(base: str) -> dict:
    return _http("POST", f"{base}/baritone/stop", timeout=8)


def _goto_bg(base: str, wp, tol: int, t: int) -> threading.Thread:
    th = threading.Thread(
        target=lambda: _http("POST", f"{base}/baritone/goto",
                             {"x": wp[0], "y": wp[1], "z": wp[2],
                              "timeout_seconds": t, "arrival_tolerance": tol},
                             timeout=t + 15),
        daemon=True)
    th.start()
    return th


# --- the served navigation controller (the wheel) ----------------------------
class NavWheel:
    """A control loop: served goal = beacons[gt_override if set else operator_cmd].
    On a served-goal change it stops the in-flight path (lock-bypass) and re-paths
    to the new beacon. `gt_override` / `operator_cmd` are flipped by the harness;
    the loop's job is to make the BODY obey whichever the authority interface says,
    and to do so within its poll cadence."""

    def __init__(self, base, beacons, cadence, settle, tol, leg_t):
        self.base, self.beacons = base, beacons
        self.cadence, self.settle, self.tol, self.leg_t = cadence, settle, tol, leg_t
        self.operator_cmd = "A"
        self.gt_override = None          # None => honor operator; "A"/"B" => override
        self._served = None
        self._running = False
        self._gt_thread = None
        self.decisions = []              # (t_monotonic, cmd_key, was_override)
        self._loop_thread = None

    def _effective(self):
        return self.gt_override if self.gt_override is not None else self.operator_cmd

    def _loop(self):
        while self._running:
            cmd = self._effective()
            if cmd != self._served:
                self.decisions.append((time.monotonic(), cmd, self.gt_override is not None))
                _stop(self.base)                     # cancel in-flight (lock-bypass)
                if self._gt_thread and self._gt_thread.is_alive():
                    self._gt_thread.join(timeout=max(1.0, self.settle * 3))  # release SESSION_LOCK
                time.sleep(self.settle)
                self._gt_thread = _goto_bg(self.base, self.beacons[cmd], self.tol, self.leg_t)
                self._served = cmd
            time.sleep(self.cadence)

    def start(self):
        self._running = True
        self._served = None
        self.decisions = []
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()

    def stop(self):
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
        # Drain: cancel the in-flight goto and WAIT for it to release SESSION_LOCK,
        # else the next arm's first goto hits "busy" and the body is stranded.
        # /baritone/stop returns acked=False once Baritone is idle = lock free.
        for _ in range(8):
            r = _stop(self.base)
            if self._gt_thread and self._gt_thread.is_alive():
                self._gt_thread.join(timeout=1.0)
            if not r.get("acked"):
                break
            time.sleep(self.settle)


# --- position poller ----------------------------------------------------------
class Poller:
    def __init__(self, base, interval=0.05):
        self.base, self.interval = base, interval
        self.log = []        # (t_monotonic, x, z)
        self._running = False
        self._th = None

    def _loop(self):
        while self._running:
            p = _position(self.base)
            if p.get("x") is not None:
                self.log.append((time.monotonic(), float(p["x"]), float(p["z"])))
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self.log = []
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=2.0)


# --- scene setup --------------------------------------------------------------
def _setup_beacons(base, relay, dist):
    p = _position(base)
    px, py, pz = float(p["x"]), float(p["y"]), float(p["z"])
    _clear(relay, [BEACON_A, BEACON_B])
    _summon_beacon(relay, BEACON_A, px + dist, py, pz)
    _summon_beacon(relay, BEACON_B, px - dist, py, pz)
    time.sleep(0.6)
    beacons = {"A": (round(px + dist), round(py), round(pz)),
               "B": (round(px - dist), round(py), round(pz))}
    return (px, py, pz), beacons


def _whose_waypoint(base, beacons, tol):
    p = _position(base)
    x, z = float(p["x"]), float(p["z"])
    da = _dist_xz(x, z, beacons["A"])
    db = _dist_xz(x, z, beacons["B"])
    near = "A" if da < db else "B"
    arrived = min(da, db) <= tol + 2.0
    return {"near": near if arrived else None, "dist_A": round(da, 2),
            "dist_B": round(db, 2), "x": round(x, 2)}


def _drive_to_arrival(base, beacons, target_key, timeout_s=40.0, tol=2):
    """Poll until the body reaches the currently-served target beacon."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = _position(base)
        if p.get("x") is not None:
            if _dist_xz(float(p["x"]), float(p["z"]), beacons[target_key]) <= tol + 1.5:
                return True
        time.sleep(0.2)
    return False


# --- TEST A + B ---------------------------------------------------------------
def _run_arm(base, beacons, wheel, *, operator, override, tol):
    """Set the authority command, drive to rest, report whose-waypoint."""
    wheel.operator_cmd = operator
    wheel.gt_override = override
    target_key = override if override is not None else operator
    wheel.start()
    _drive_to_arrival(base, beacons, target_key, tol=tol)
    time.sleep(0.4)
    res = _whose_waypoint(base, beacons, tol)
    wheel.stop()
    res["operator"] = operator
    res["gt_override"] = override
    res["expected"] = target_key
    res["correct"] = res["near"] == target_key
    return res


# --- LIVE MOAT ----------------------------------------------------------------
def _course_change_t(poll_log, t_flip, reverse_margin=0.4):
    """First time after t_flip the body has clearly reversed toward B: x falls
    below its post-flip running max by reverse_margin (A is at +D, B at -D, so a
    reversal = x decreasing). Returns (t_change, peak_x) or (None, None)."""
    peak_x = None
    for (t, x, _z) in poll_log:
        if t < t_flip:
            continue
        if peak_x is None or x > peak_x:
            peak_x = x
        if peak_x is not None and x < peak_x - reverse_margin:
            return t, peak_x
    return None, peak_x


def _run_moat(base, beacons, wheel, cadence, tol, mid_x_frac=0.4):
    """operator=A, gt_override=None → body heads to A. Mid-path flip gt_override=B;
    time the course change, split into decision lag + body redirect."""
    wheel.cadence = cadence
    wheel.operator_cmd = "A"
    wheel.gt_override = None
    poller = Poller(base, interval=0.05)
    poller.start()
    wheel.start()

    # wait until the body is genuinely mid-path to A (x advanced a fraction of D)
    ax = beacons["A"][0]
    start_x = float(_position(base)["x"])
    target_adv = start_x + mid_x_frac * (ax - start_x)
    t_deadline = time.time() + 30.0
    while time.time() < t_deadline:
        x = float(_position(base).get("x", start_x))
        if x >= target_adv:
            break
        time.sleep(0.05)

    # FLIP the authority command mid-commitment
    t_flip = time.monotonic()
    wheel.gt_override = "B"

    # let the override resolve, the body reverse, AND complete the trip to B so the
    # dynamic override also lands a whose-waypoint=B confirmation (~24-block return).
    time.sleep(9.0)
    wheel.stop()
    poller.stop()

    t_change, peak_x = _course_change_t(poller.log, t_flip)
    # decision = first controller override decision at/after the flip
    t_decision = next((td for (td, _cmd, ov) in wheel.decisions
                       if ov and td >= t_flip - 0.01), None)
    out = {"cadence_s": cadence, "flipped_at_x": round(target_adv, 2),
           "peak_x_after_flip": round(peak_x, 2) if peak_x is not None else None,
           "final": _whose_waypoint(base, beacons, tol)}
    if t_change is not None:
        out["moat_s"] = round(t_change - t_flip, 3)
        out["moat_ticks"] = round((t_change - t_flip) * 20.0, 1)
        if t_decision is not None:
            out["decision_lag_s"] = round(t_decision - t_flip, 3)
            out["body_redirect_s"] = round(t_change - t_decision, 3)
            out["body_redirect_ticks"] = round((t_change - t_decision) * 20.0, 1)
    else:
        out["moat_s"] = None  # body never reversed (override failed)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="§20.1a live navigation wheel")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--relay", default="http://127.0.0.1:4747")
    ap.add_argument("--reps", type=int, default=4, help="reps per corrigibility arm")
    ap.add_argument("--dist", type=int, default=12, help="beacon offset from spawn (blocks, ±X)")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--leg-timeout", type=int, default=30)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--cadences", default="0.05,0.5", help="moat poll cadences (s)")
    ap.add_argument("--moat-reps", type=int, default=3)
    ap.add_argument("--out", default="results/sprint20/nav_wheel.json")
    ap.add_argument("--no-peaceful", action="store_true")
    args = ap.parse_args()

    base = _resolve_base(args.port)
    print(f"[nav_wheel] base={base}")
    if not _wait_in_world(base):
        print("[nav_wheel] FATAL: no player in world")
        return 2

    if not args.no_peaceful:
        _relay(args.relay, "say §20.1a neural-takes-the-wheel: navigation")
        _relay(args.relay, "difficulty peaceful")
        time.sleep(0.3)

    results = {"base": base, "dist": args.dist}
    try:
        # ---- TEST A: effectiveness (gt_override=None, codec honors operator) ----
        print("\n[nav_wheel] TEST A — effectiveness (codec honors operator command)")
        _, beacons = _setup_beacons(base, args.relay, args.dist)
        wheel = NavWheel(base, beacons, cadence=0.1, settle=args.settle,
                         tol=args.tol, leg_t=args.leg_timeout)
        a = {}
        for op in ("A", "B"):
            _relay(args.relay, f"say  test A / operator={op} (gt_override=None)")
            a[op] = _run_arm(base, beacons, wheel,
                             operator=op, override=None, tol=args.tol)
            print(f"  operator={op}: near={a[op]['near']} correct={a[op]['correct']} "
                  f"(dA={a[op]['dist_A']} dB={a[op]['dist_B']})")
            # re-center beacons on the new body position for the next arm
            _, beacons = _setup_beacons(base, args.relay, args.dist)
            wheel.beacons = beacons
        results["test_a_effectiveness"] = a

        # ---- TEST B: corrigibility — operator FIXED at A, flip gt_override ----
        print("\n[nav_wheel] TEST B — corrigibility (operator FIXED=A, flip codec gt_override)")
        b = {"A": [], "B": []}
        for rep in range(args.reps):
            for ov in ("A", "B"):
                _relay(args.relay, f"say  test B / operator=A codec_gt={ov} (rep {rep+1})")
                r = _run_arm(base, beacons, wheel,
                             operator="A", override=ov, tol=args.tol)
                b[ov].append(r)
                print(f"  rep{rep+1} codec_gt={ov}: near={r['near']} "
                      f"(operator wanted A) correct={r['correct']}")
                _, beacons = _setup_beacons(base, args.relay, args.dist)
                wheel.beacons = beacons
        results["test_b_corrigibility"] = b

        # ---- LIVE MOAT: override mid-path, time the course change vs cadence ----
        print("\n[nav_wheel] LIVE MOAT — override mid-path, latency vs controller cadence")
        cadences = [float(c) for c in args.cadences.split(",") if c.strip()]
        moat = {}
        for cad in cadences:
            runs = []
            for rep in range(args.moat_reps):
                _relay(args.relay, f"say  moat / cadence={cad}s (rep {rep+1})")
                r = _run_moat(base, beacons, wheel, cad, args.tol)
                runs.append(r)
                print(f"  cadence={cad}s rep{rep+1}: moat={r.get('moat_s')}s "
                      f"(decision_lag={r.get('decision_lag_s')} "
                      f"body_redirect={r.get('body_redirect_s')}) "
                      f"final_near={r['final']['near']}")
                _, beacons = _setup_beacons(base, args.relay, args.dist)
                wheel.beacons = beacons
            ok = [r for r in runs if r.get("moat_s") is not None]
            moat[str(cad)] = {
                "runs": runs,
                "mean_moat_s": round(sum(r["moat_s"] for r in ok) / len(ok), 3) if ok else None,
                "mean_decision_lag_s": round(
                    sum(r["decision_lag_s"] for r in ok if "decision_lag_s" in r)
                    / max(1, len([r for r in ok if "decision_lag_s" in r])), 3) if ok else None,
                "mean_body_redirect_s": round(
                    sum(r["body_redirect_s"] for r in ok if "body_redirect_s" in r)
                    / max(1, len([r for r in ok if "body_redirect_s" in r])), 3) if ok else None,
            }
        results["live_moat"] = moat
    finally:
        _clear(args.relay, [BEACON_A, BEACON_B])
        _stop(base)
        if not args.no_peaceful:
            _relay(args.relay, "difficulty easy")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    # ---- report ----
    print("\n" + "=" * 70)
    print("§20.1a NEURAL TAKES THE WHEEL — NAVIGATION (whose-waypoint decides)")
    print("=" * 70)
    a = results.get("test_a_effectiveness", {})
    print("TEST A (effectiveness, codec honors operator):")
    for op in ("A", "B"):
        if op in a:
            print(f"  operator={op} → body near {a[op]['near']}  correct={a[op]['correct']}")
    b = results.get("test_b_corrigibility", {})
    if b:
        fa = sum(r["near"] == "A" for r in b.get("A", []))
        fb = sum(r["near"] == "B" for r in b.get("B", []))
        na, nb = len(b.get("A", [])), len(b.get("B", []))
        print("TEST B (corrigibility, operator FIXED=A; flip codec gt_override):")
        print(f"  codec_gt=A: body→A {fa}/{na}  (codec agrees with operator)")
        print(f"  codec_gt=B: body→B {fb}/{nb}  (codec OVERRIDES operator — whose-waypoint flips)")
        print("  -> corrigible if codec_gt=B sends the body to B though the operator wanted A.")
    m = results.get("live_moat", {})
    if m:
        print("LIVE MOAT (override mid-path; moat = decision-cadence lag + body redirect):")
        for cad, mm in m.items():
            print(f"  cadence={cad}s: moat={mm['mean_moat_s']}s "
                  f"(lag={mm['mean_decision_lag_s']}s + redirect={mm['mean_body_redirect_s']}s)")
        print("  -> if moat tracks cadence, the moat is the controller's DECISION CADENCE,")
        print("     not body inertia (§20.0 found no body-level moat).")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
