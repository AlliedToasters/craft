"""R1→R3 ablation on `interact` — does entity_set predict combat packets where
g_t didn't? (neural_interface.md §8c-bis redirect, §2b entity_set.)

The R0→R1 result showed g_t gives ~0 on interact: interact is KillAura-reflexive
(fires when a mob enters range), not goal-driven. The mechanism predicts the
signal lives in `entity_set` (R3), not `g_t` (R1). This script tests that by
joining the heavy tick sidecar's entity_set to each packet (by tick) and adding
an entity feature group.

Entity features are deliberately threat-agnostic (ml.MD §5b — don't bake the
answer into the sensor): a histogram of entity *types* within a radius + nearest
distance + count. The model learns which types matter; we don't hand-label
"hostile".

Arms (shared val split, same packets):
  R0        pose
  R1        pose + g_t one-hot + temporal           (prior best for interact)
  ent_only  pose + entity                            (isolates the entity channel)
  R3        pose + g_t + temporal + entity           (full)

Usage:
  .venv/bin/python -m experiments.next_packet.ablation_r1_r3 \
      --rollouts-glob "results/frozen_combat/rollout-*" --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

from .ablation_r0_r1 import _R0, _R1_TEMPORAL, GoalVocab, Normalizer, _open_text
from .features import PACKET_TYPE_INDEX, PACKET_TYPES
from .metrics import TypeMetrics

_DIM_ORDER = {"minecraft:overworld": 0.0, "minecraft:the_nether": 1.0, "minecraft:the_end": 2.0}

ENTITY_RADIUS = 8.0   # histogram "within" radius (KillAura engages close)
NEAREST_CAP = 64.0    # nearest-distance sentinel when no entity in range


def load_examples(rollout_dirs: list[str]) -> list[tuple[dict, list | None, str]]:
    """(obs, entity_set, packet_type). entity_set joined from the tick sidecar;
    delta_tick computed per-file. Only entity_set is retained from each sidecar
    row (block_grid is dropped to keep memory bounded)."""
    out: list[tuple[dict, list | None, str]] = []
    for d in rollout_dirs:
        dp = Path(d)
        packets = dp / "packets.jsonl"
        sidecar = next(iter(glob.glob(str(dp / "sidecar.jsonl*"))), None)
        if not packets.exists():
            continue
        # tick → entity_set
        ent_by_tick: dict[int, list] = {}
        if sidecar:
            with _open_text(sidecar) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    t = row.get("tick")
                    if isinstance(t, int):
                        ent_by_tick[t] = row.get("entity_set")
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
                obs = dict(rec.get("obs") or {})
                tick = obs.get("tick")
                obs["delta_tick"] = max(0, tick - prev_tick) if (isinstance(tick, int) and prev_tick is not None) else 0
                if isinstance(tick, int):
                    prev_tick = tick
                ent = ent_by_tick.get(tick) if isinstance(tick, int) else None
                out.append((obs, ent, ptype))
    return out


class EntityVocab:
    """Non-player entity types seen in training. Threat-agnostic — raw types."""

    def __init__(self, types: list[str]) -> None:
        self.types = list(types)
        self.index = {t: i for i, t in enumerate(types)}

    @classmethod
    def fit(cls, examples) -> "EntityVocab":
        s: set[str] = set()
        for _obs, ent, _p in examples:
            if not ent:
                continue
            for e in ent[1:]:  # skip index 0 (player)
                t = e.get("type")
                if t:
                    s.add(t)
        return cls(sorted(s))

    @property
    def size(self) -> int:
        return len(self.types)

    def names(self) -> list[str]:
        return [f"ent:{t}" for t in self.types]


def entity_features(ent: list | None, evocab: EntityVocab) -> list[float]:
    """Histogram of entity types within ENTITY_RADIUS + nearest dist + count.
    ent[0] is the player (self) — used as the distance origin, excluded from
    the histogram."""
    hist = [0.0] * evocab.size
    nearest = NEAREST_CAP
    n_within = 0
    if ent and len(ent) >= 1:
        p = ent[0]
        px, py, pz = float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))
        for e in ent[1:]:
            dx = float(e.get("x", 0.0)) - px
            dy = float(e.get("y", 0.0)) - py
            dz = float(e.get("z", 0.0)) - pz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            nearest = min(nearest, dist)
            if dist <= ENTITY_RADIUS:
                n_within += 1
                ti = evocab.index.get(e.get("type"))
                if ti is not None:
                    hist[ti] += 1.0
    return hist + [nearest, float(n_within)]


def layout(gvocab: GoalVocab, evocab: EntityVocab, groups: set[str]) -> list[tuple[str, bool]]:
    cols = list(_R0)
    if "goal" in groups:
        cols += [(n, False) for n in gvocab.names()]
    if "temporal" in groups:
        cols += _R1_TEMPORAL
    if "entity" in groups:
        cols += [(n, False) for n in evocab.names()]          # type counts: raw
        cols += [("nearest_entity_dist", True), ("n_entities_within", True)]
    return cols


def featurize(obs: dict, ent: list | None, gvocab: GoalVocab, evocab: EntityVocab,
              groups: set[str]) -> list[float]:
    yaw = math.radians(float(obs.get("yaw", 0.0)))
    pitch = math.radians(float(obs.get("pitch", 0.0)))
    feats = [
        float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("z", 0.0)),
        math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
        1.0 if obs.get("on_ground") else 0.0,
        _DIM_ORDER.get(str(obs.get("dim", "")), -1.0),
    ]
    if "goal" in groups:
        feats += gvocab.onehot(obs.get("g_t"))
    if "temporal" in groups:
        feats += [float(obs.get("ticks_since_g_t_issued") or 0), float(obs.get("delta_tick") or 0)]
    if "entity" in groups:
        feats += entity_features(ent, evocab)
    return feats


ARMS: dict[str, set[str]] = {
    "R0":       set(),
    "R1":       {"goal", "temporal"},
    "ent_only": {"entity"},
    "R3":       {"goal", "temporal", "entity"},
}


def train_arm(examples, gvocab, evocab, label, groups, *, val_idx, hidden, epochs, lr, batch_size, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    cols = layout(gvocab, evocab, groups)
    raw = [(featurize(o, ent, gvocab, evocab, groups), PACKET_TYPE_INDEX[p], p) for o, ent, p in examples]
    train_raw = [raw[i] for i in range(len(raw)) if i not in val_idx]
    val_raw = [raw[i] for i in range(len(raw)) if i in val_idx]

    norm = Normalizer(cols).fit([f for f, _, _ in train_raw])
    train = [(norm.transform(f), y, p) for f, y, p in train_raw]
    val = [(norm.transform(f), y, p) for f, y, p in val_raw]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(len(cols), hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, len(PACKET_TYPES)),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    def tens(batch):
        xs = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        return xs, ys

    rng = random.Random(seed)
    best_acc, best_m = 0.0, TypeMetrics()
    for _e in range(epochs):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), batch_size):
            xs, ys = tens(train[i:i + batch_size])
            opt.zero_grad()
            crit(model(xs), ys).backward()
            opt.step()
        model.eval()
        m = TypeMetrics()
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch = val[i:i + batch_size]
                xs, _ = tens(batch)
                preds = model(xs).argmax(dim=-1).tolist()
                for (_, _, pt), pi in zip(batch, preds):
                    m.update(pt, PACKET_TYPES[pi])
        acc = m.overall_accuracy()
        if acc >= best_acc:
            best_acc, best_m = acc, m
    return best_acc, best_m, len(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="R1→R3 entity_set ablation on interact (§8c-bis)")
    ap.add_argument("--rollouts-glob", default="results/frozen_combat/rollout-*")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(glob.glob(args.rollouts_glob))
    examples = load_examples(dirs)
    if not examples:
        print(f"No examples from {args.rollouts_glob}", file=sys.stderr)
        sys.exit(1)
    n_with_ent = sum(1 for _, e, _ in examples if e)
    gvocab = GoalVocab.fit([(o, p) for o, _, p in examples])
    evocab = EntityVocab.fit(examples)

    rng = random.Random(args.seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_idx = set(idx[:n_val])

    print(f"examples={len(examples)} (entity-joined={n_with_ent})  val={n_val}")
    print(f"entity_vocab ({evocab.size}): {evocab.types}")
    print(f"entity_radius={ENTITY_RADIUS}")

    results = {}
    for label in ARMS:
        acc, m, dim = train_arm(
            examples, gvocab, evocab, label, ARMS[label], val_idx=val_idx,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
        )
        results[label] = (acc, m, dim)
        print(f"  {label:<9} dim={dim:<4} overall_val_acc={acc:.4f}")

    print("\n" + "=" * 72)
    print("R1 → R3 ablation — focus: interact (does entity_set predict it?)")
    print("=" * 72)
    r0 = results["R0"][0]
    print(f"  {'arm':<10}{'dim':>5}{'overall':>10}{'Δ vs R0':>10}")
    for label in ARMS:
        acc, _m, dim = results[label]
        print(f"  {label:<10}{dim:>5}{acc:>10.4f}{acc - r0:>+10.4f}")

    print(f"\n  per-type accuracy by arm (types with val support):")
    ref_m = results["R3"][1]
    print("  " + f"{'type':<34}{'n':>5}" + "".join(f"{a:>11}" for a in ARMS))
    for t in PACKET_TYPES:
        n = ref_m._total[t]
        if n == 0:
            continue
        cells = "".join(
            f"{(results[a][1].accuracy(t) if results[a][1].accuracy(t) is not None else float('nan')):>11.3f}"
            for a in ARMS
        )
        print(f"  {t:<34}{n:>5}{cells}")


if __name__ == "__main__":
    main()
