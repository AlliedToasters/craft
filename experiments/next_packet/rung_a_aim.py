"""Rung A — the aim head (neural_interface.md §11). The cadence-proof driver test.

The rung-A *discriminator* (rung_a_driver.py) hit 0.86, but ~95% of the lift was
`delta_tick` — a teacher-forcing crutch (the executor's own emission cadence) that
evaporates the moment a net has to *decide when* to act. So packet-TYPE is the
wrong target. This script predicts a packet FIELD that timing cannot fake: the
yaw/pitch the executor turns to on every `*_rot` packet.

Aim is the executor's core continuous choice — Baritone looks toward the path/
mining target, Wurst KillAura snaps to the mob. A rotation *value* can only come
from world-state. So if `prox` (relative entity geometry) predicts the aim and
`timing` does not, the executor's look-control is genuinely world-driven and a
small net is a real neural aim driver — not a cadence echo.

Target: (sin,cos) of (yaw,pitch) of each has_rot packet → decoded to a wrapped
angular error in degrees. Reported against two no-train baselines:
  persistence  echo the current pose look      (rotations are small per-packet)
  oracle_near  point straight at nearest entity (combat upper bound for "aim at mob")

Arms (inputs to the learned head):
  pose          current look + on_ground
  pose+prox     + nearest-3 entity relative (dx,dy,dz) + count
  pose+prox+dt  + delta_tick                 (must NOT beat pose+prox — exposes the crutch)

Usage:
  .venv/bin/python -m experiments.next_packet.rung_a_aim \
      --rollouts-glob "results/frozen_combat/rollout-*" --epochs 60
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

from .ablation_r0_r1 import _open_text
from .features import PACKET_TYPE_INDEX

ROT_TYPES = {"minecraft:move_player_rot", "minecraft:move_player_pos_rot"}
N_NEAR = 3            # nearest entities exposed to the prox arm
FAR = 64.0           # padding distance for missing neighbors


def load_aim(rollout_dirs: list[str]) -> list[dict]:
    """One row per has_rot packet: current pose, target yaw/pitch, nearest-K
    entity geometry (joined from sidecar by tick), delta_tick (per-file)."""
    rows: list[dict] = []
    for d in rollout_dirs:
        dp = Path(d)
        packets = dp / "packets.jsonl"
        sidecar = next(iter(glob.glob(str(dp / "sidecar.jsonl*"))), None)
        if not packets.exists():
            continue
        ent_by_tick: dict[int, list] = {}
        if sidecar:
            with _open_text(sidecar) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    t = r.get("tick")
                    if isinstance(t, int):
                        ent_by_tick[t] = r.get("entity_set")
        prev_tick: int | None = None
        with _open_text(str(packets)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                ptype = rec.get("id")
                if ptype not in PACKET_TYPE_INDEX:
                    continue
                obs = rec.get("obs") or {}
                tick = obs.get("tick")
                dt = max(0, tick - prev_tick) if (isinstance(tick, int) and prev_tick is not None) else 0
                if isinstance(tick, int):
                    prev_tick = tick
                if ptype not in ROT_TYPES:
                    continue
                f_ = rec.get("fields") or {}
                if f_.get("yaw") is None or f_.get("pitch") is None:
                    continue
                ent = ent_by_tick.get(tick) if isinstance(tick, int) else None
                rows.append({
                    "cur_yaw": float(obs.get("yaw", 0.0)),
                    "cur_pitch": float(obs.get("pitch", 0.0)),
                    "on_ground": 1.0 if obs.get("on_ground") else 0.0,
                    "tgt_yaw": float(f_["yaw"]),
                    "tgt_pitch": float(f_["pitch"]),
                    "near": _nearest_rel(ent),     # [(dx,dy,dz,dist)] sorted, len N_NEAR
                    "n_within8": _count_within(ent, 8.0),
                    "delta_tick": float(dt),
                })
    return rows


def _nearest_rel(ent: list | None) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    if ent and len(ent) >= 1:
        p = ent[0]
        px, py, pz = float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))
        cand = []
        for e in ent[1:]:
            dx, dy, dz = float(e.get("x", 0)) - px, float(e.get("y", 0)) - py, float(e.get("z", 0)) - pz
            cand.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))
        cand.sort(key=lambda c: c[3])
        out = cand[:N_NEAR]
    while len(out) < N_NEAR:
        out.append((0.0, 0.0, 0.0, FAR))
    return out


def _count_within(ent: list | None, r: float) -> float:
    if not ent or len(ent) < 1:
        return 0.0
    p = ent[0]
    px, py, pz = float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))
    n = 0
    for e in ent[1:]:
        dx, dy, dz = float(e.get("x", 0)) - px, float(e.get("y", 0)) - py, float(e.get("z", 0)) - pz
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= r:
            n += 1
    return float(n)


def _ang_target(yaw_deg: float, pitch_deg: float) -> list[float]:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return [math.sin(y), math.cos(y), math.sin(p), math.cos(p)]


def _decode_angle(sin_v: float, cos_v: float) -> float:
    return math.degrees(math.atan2(sin_v, cos_v))


def _ang_err(a_deg: float, b_deg: float) -> float:
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)


def featurize(row: dict, groups: set[str]) -> list[float]:
    cy, cp = math.radians(row["cur_yaw"]), math.radians(row["cur_pitch"])
    feats = [math.sin(cy), math.cos(cy), math.sin(cp), math.cos(cp), row["on_ground"]]
    if "prox" in groups:
        for dx, dy, dz, dist in row["near"]:
            feats += [dx, dy, dz, dist]
        feats += [row["n_within8"]]
    if "timing" in groups:
        feats += [row["delta_tick"]]
    return feats


def _scale(rows_feats: list[list[float]]):
    n = len(rows_feats)
    dim = len(rows_feats[0])
    mean = [0.0] * dim
    std = [1.0] * dim
    for i in range(dim):
        col = [v[i] for v in rows_feats]
        m = sum(col) / n
        var = sum(x * x for x in col) / n - m * m
        mean[i], std[i] = m, max(math.sqrt(max(var, 0.0)), 1e-6)
    return mean, std


ARMS: dict[str, set[str]] = {
    "pose":        set(),
    "pose+prox":   {"prox"},
    "pose+prox+dt": {"prox", "timing"},
}


def eval_baselines(val: list[dict]) -> dict[str, tuple[float, float]]:
    """No-train baselines: (mean yaw err, mean pitch err) in degrees."""
    persist_y = sum(_ang_err(r["tgt_yaw"], r["cur_yaw"]) for r in val) / len(val)
    persist_p = sum(_ang_err(r["tgt_pitch"], r["cur_pitch"]) for r in val) / len(val)
    # oracle: aim at nearest entity (only rows with one in range)
    oy = op = 0.0
    n = 0
    for r in val:
        dx, dy, dz, dist = r["near"][0]
        if dist >= FAR:
            continue
        bear_yaw = math.degrees(math.atan2(-dx, dz))          # MC yaw convention
        horiz = math.sqrt(dx * dx + dz * dz)
        bear_pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-6 else 0.0
        oy += _ang_err(r["tgt_yaw"], bear_yaw)
        op += _ang_err(r["tgt_pitch"], bear_pitch)
        n += 1
    oracle = (oy / n, op / n) if n else (float("nan"), float("nan"))
    return {"persistence": (persist_y, persist_p), "oracle_near": oracle, "_oracle_n": (float(n), 0.0)}


def eval_model(model, val_subset, groups, normfn):
    import torch
    if not val_subset:
        return float("nan"), float("nan")
    device = next(model.parameters()).device
    xv = torch.tensor([normfn(featurize(r, groups)) for r in val_subset],
                      dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(xv).cpu().tolist()
    ey = ep = 0.0
    for pr, r in zip(pred, val_subset):
        ey += _ang_err(_decode_angle(pr[0], pr[1]), r["tgt_yaw"])
        ep += _ang_err(_decode_angle(pr[2], pr[3]), r["tgt_pitch"])
    return ey / len(val_subset), ep / len(val_subset)


def train_arm(train, val, groups, *, hidden, epochs, lr, batch_size, seed, return_model=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    Xtr = [featurize(r, groups) for r in train]
    mean, std = _scale(Xtr)
    def norm(x): return [(x[i] - mean[i]) / std[i] for i in range(len(x))]
    Ytr = [_ang_target(r["tgt_yaw"], r["tgt_pitch"]) for r in train]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = len(Xtr[0])
    model = nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 4),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    xt = torch.tensor([norm(x) for x in Xtr], dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)

    rng = random.Random(seed)
    order = list(range(len(train)))
    best = (float("inf"), float("inf"))
    for _e in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            crit(model(xt[idx]), yt[idx]).backward()
            opt.step()
        if not return_model:
            ey, ep = eval_model(model, val, groups, norm)
            if ey + ep < best[0] + best[1]:
                best = (ey, ep)
    if return_model:
        return model, norm, dim
    return best, dim


def main() -> None:
    ap = argparse.ArgumentParser(description="Rung A aim head — cadence-proof driver test (§11)")
    ap.add_argument("--rollouts-glob", default="results/frozen_combat/rollout-*")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retarget-deg", type=float, default=15.0,
                    help="also evaluate on the subset where |tgt_yaw-cur_yaw| >= this (the re-targeting events)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(glob.glob(args.rollouts_glob))
    rows = load_aim(dirs)
    if not rows:
        print(f"No has_rot packets from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    n_mob = sum(1 for r in val if r["near"][0][3] < FAR)

    print(f"has_rot packets={len(rows)}  train={len(train)}  val={n_val}  "
          f"val_rows_with_entity_in_range={n_mob}")

    # Models are trained ONCE (on `train`, all rows); we evaluate the same fitted
    # model on the full val and on the re-targeting subset. The subset is a slice
    # of val, so there is no train/eval leak.
    trained = {}
    for label in ARMS:
        trained[label] = train_arm(
            train, val, ARMS[label], hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size, seed=args.seed,
            return_model=True,
        )

    def report(val_subset: list[dict], tag: str) -> None:
        if not val_subset:
            print(f"\n  [{tag}] empty subset"); return
        base = eval_baselines(val_subset)
        print(f"\n  === {tag}  (n={len(val_subset)}) ===")
        print("  baseline (no training)          yaw_err°   pitch_err°")
        print(f"    {'persistence (echo pose)':<30}{base['persistence'][0]:>8.2f}{base['persistence'][1]:>12.2f}")
        print(f"    {'oracle: aim @ nearest ent':<30}{base['oracle_near'][0]:>8.2f}{base['oracle_near'][1]:>12.2f}"
              f"   (n={int(base['_oracle_n'][0])})")
        print("  learned head                    yaw_err°   pitch_err°   dim")
        for label in ARMS:
            ey, ep = eval_model(trained[label][0], val_subset, ARMS[label], trained[label][1])
            print(f"    {label:<30}{ey:>8.2f}{ep:>12.2f}{trained[label][2]:>6}")

    report(val, "ALL has_rot")
    retarget = [r for r in val if _ang_err(r["tgt_yaw"], r["cur_yaw"]) >= args.retarget_deg]
    report(retarget, f"RE-TARGETS only (|Δyaw|>={args.retarget_deg:.0f}°)")

    print("\n  read: per-packet aim is inertia-dominated (persistence wins on ALL);")
    print("        the test is whether world-state beats persistence on RE-TARGETS.")


if __name__ == "__main__":
    main()
