"""Rung A — the attack-target pointer head (neural_interface.md §6/§8c/§11).

The rung-A payoff. Type-prediction was faked by cadence (rung_a_driver) and aim by
positional inertia (rung_a_aim) — both are packet-stream autocorrelation, not
control. The genuinely world-driven decisions are the *sparse discrete events*.
The best-powered one is `interact.ATTACK`: which entity does the executor (Wurst
KillAura) strike? This is the §6 pointer gap — a pointer INTO the entity_set, not
a fixed-width field — and it cannot be faked by cadence or persistence.

Setup (combat data): for each ATTACK packet, the candidates are the entity_set
(joined by tick); the label is the index whose runtime_id == fields.entity_id
(100% recoverable in this capture). A shared per-candidate MLP scores each
candidate; segment-softmax over the event's candidates; cross-entropy to the true
index (a pointer network).

Baselines (analytic, no training):
  nearest         pick the closest entity                 (= 0.43 here: target is
                                                            NOT just nearest 57% of the time)
  nearest_hostile pick the closest of a hardcoded hostile set (sanity ceiling for
                                                            "threat priority")

Arms (learned scorer):
  geom        per-candidate relative pos + off-axis angle   (is it geometric?)
  geom+type   + entity-type one-hot                          (is threat priority learned?)

If geom+type clears `nearest` and approaches `nearest_hostile`, KillAura target
selection is a learnable function of geometry + type — the entity pointer gap
closes, threat-agnostic (the net learns which types matter; we don't label them).

Usage:
  .venv/bin/python -m experiments.next_packet.rung_a_target \
      --rollouts-glob "results/frozen_combat/rollout-*" --epochs 200
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
from .ablation_r1_r3 import EntityVocab

# Only used for the analytic nearest_hostile *baseline* — the learned arms never
# see this set (type is fed as a raw one-hot; the net learns priority itself).
_HOSTILE = {"minecraft:zombie", "minecraft:skeleton", "minecraft:creeper",
            "minecraft:spider", "minecraft:witch", "minecraft:drowned",
            "minecraft:zombie_villager", "minecraft:enderman"}


def load_attacks(rollout_dirs: list[str]) -> list[dict]:
    """One row per ATTACK with its candidate entity_set (joined by tick) and the
    target index (rank in the candidate list)."""
    rows: list[dict] = []
    for d in rollout_dirs:
        dp = Path(d)
        packets = dp / "packets.jsonl"
        sidecar = next(iter(glob.glob(str(dp / "sidecar.jsonl*"))), None)
        if not packets.exists() or not sidecar:
            continue
        ent_by_tick: dict[int, list] = {}
        with _open_text(sidecar) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("tick")
                if isinstance(t, int):
                    ent_by_tick[t] = r.get("entity_set")
        with _open_text(str(packets)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("id") != "minecraft:interact":
                    continue
                fld = rec.get("fields") or {}
                if fld.get("action") != "ATTACK" or fld.get("entity_id") is None:
                    continue
                obs = rec.get("obs") or {}
                tick = obs.get("tick")
                ent = ent_by_tick.get(tick) if isinstance(tick, int) else None
                if not ent or len(ent) < 2:
                    continue
                p = ent[0]
                px, py, pz = float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))
                cur_yaw = float(obs.get("yaw", 0.0))
                cur_pitch = float(obs.get("pitch", 0.0))
                cands = []
                for e in ent[1:]:
                    dx, dy, dz = float(e.get("x", 0)) - px, float(e.get("y", 0)) - py, float(e.get("z", 0)) - pz
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    horiz = math.sqrt(dx * dx + dz * dz)
                    bear_yaw = math.degrees(math.atan2(-dx, dz))
                    bear_pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-6 else 0.0
                    off_yaw = math.radians((bear_yaw - cur_yaw + 180.0) % 360.0 - 180.0)
                    off_pitch = math.radians((bear_pitch - cur_pitch + 180.0) % 360.0 - 180.0)
                    cands.append({
                        "rid": e.get("runtime_id"), "type": e.get("type", ""),
                        "dx": dx, "dy": dy, "dz": dz, "dist": dist,
                        "off_yaw": off_yaw, "off_pitch": off_pitch,
                    })
                cands.sort(key=lambda c: c["dist"])
                ids = [c["rid"] for c in cands]
                if fld["entity_id"] not in ids:
                    continue
                rows.append({"cands": cands, "label": ids.index(fld["entity_id"])})
    return rows


def cand_features(c: dict, evocab: EntityVocab, use_type: bool) -> list[float]:
    f = [c["dx"], c["dy"], c["dz"], c["dist"],
         math.sin(c["off_yaw"]), math.cos(c["off_yaw"]),
         math.sin(c["off_pitch"]), math.cos(c["off_pitch"])]
    if use_type:
        oh = [0.0] * evocab.size
        ti = evocab.index.get(c["type"])
        if ti is not None:
            oh[ti] = 1.0
        f += oh
    return f


def baseline_nearest(rows) -> float:
    # candidates are dist-sorted, so the nearest is index 0.
    return sum(1 for r in rows if r["label"] == 0) / len(rows)


def baseline_nearest_hostile(rows) -> float:
    hit = 0
    for r in rows:
        pick = next((i for i, c in enumerate(r["cands"]) if c["type"] in _HOSTILE), 0)
        if pick == r["label"]:
            hit += 1
    return hit / len(rows)


def train_arm(train, val, evocab, use_type, *, hidden, epochs, lr, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    dim = len(cand_features(train[0]["cands"][0], evocab, use_type))
    # z-score the 4 geom dims (dx,dy,dz,dist) over all candidates; rest pass through.
    geom = [[c[k] for c in (cc for r in train for cc in r["cands"])]
            for k in ("dx", "dy", "dz", "dist")]
    mean = [sum(col) / len(col) for col in geom]
    std = [max(math.sqrt(sum(x * x for x in col) / len(col) - (sum(col) / len(col)) ** 2), 1e-6)
           for col in geom]

    def feat(c):
        f = cand_features(c, evocab, use_type)
        for i in range(4):
            f[i] = (f[i] - mean[i]) / std[i]
        return f

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    # Pre-tensor each event's candidate matrix (variable length → per-event).
    tr = [(torch.tensor([feat(c) for c in r["cands"]], dtype=torch.float32, device=device),
           torch.tensor(r["label"], device=device)) for r in train]

    rng = random.Random(seed)
    best = 0.0
    for _e in range(epochs):
        model.train()
        rng.shuffle(tr)
        for X, y in tr:
            opt.zero_grad()
            scores = model(X).squeeze(-1).unsqueeze(0)   # [1, n_cand]
            ce(scores, y.unsqueeze(0)).backward()
            opt.step()
        # eval with normalized features
        model.eval()
        correct = 0
        with torch.no_grad():
            for r in val:
                X = torch.tensor([feat(c) for c in r["cands"]], dtype=torch.float32, device=device)
                if int(model(X).squeeze(-1).argmax().item()) == r["label"]:
                    correct += 1
        acc = correct / len(val)
        best = max(best, acc)
    return best, dim


def main() -> None:
    ap = argparse.ArgumentParser(description="Rung A attack-target pointer head (§6/§8c/§11)")
    ap.add_argument("--rollouts-glob", default="results/frozen_combat/rollout-*")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    rows = load_attacks(sorted(glob.glob(args.rollouts_glob)))
    if not rows:
        print(f"No ATTACK events from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)
    types = sorted({c["type"] for r in rows for c in r["cands"] if c["type"]})
    evocab = EntityVocab(types)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    avg_cand = sum(len(r["cands"]) for r in rows) / len(rows)

    print(f"ATTACK events={len(rows)}  train={len(train)}  val={n_val}  "
          f"avg_candidates={avg_cand:.1f}  entity_types={evocab.size}")
    print("\n  baseline (no training)            val_acc")
    print(f"    {'nearest':<30}{baseline_nearest(val):>8.3f}")
    print(f"    {'nearest_hostile':<30}{baseline_nearest_hostile(val):>8.3f}")

    print("\n  learned pointer head              val_acc   dim")
    for label, use_type in [("geom", False), ("geom+type", True)]:
        acc, dim = train_arm(train, val, evocab, use_type,
                             hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed)
        print(f"    {label:<30}{acc:>8.3f}{dim:>6}")

    print("\n  read: if geom+type clears `nearest` and nears `nearest_hostile`, the")
    print("        KillAura target policy (geometry + threat) is a learnable pointer.")


if __name__ == "__main__":
    main()
