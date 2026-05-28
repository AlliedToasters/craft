"""R0→R1 obs-ablation for the next-packet discriminator (neural_interface.md §8b/§8c).

Self-contained so it doesn't disturb the train.py/checkpoint.py scaffolds (which
pin the old fixed-width FeatureNormalizer API). Trains the *discriminator* —
predict the wire type from obs — at two rungs on the SAME packets:

  R0  minimal pose only (x,y,z, sin/cos yaw+pitch, on_ground, dim)
  R1  R0 + goal identity (g_t one-hot) + ticks_since_g_t_issued + delta_tick

The label is the packet id directly (`line["id"]`), so the codec is not needed
for the discriminator. Reads the per-packet recordings (packets.jsonl[.gz])
produced by the frozen-capture runner; delta_tick is computed per-file from the
tick gap to the previous packet (§3d temporal frame).

Rung gating is real: the model input width changes (R0 = 9 dims, R1 = 9 +
|goal_vocab| + 2), architecture held constant (§8b). Continuous features are
z-scored on the train split; booleans / one-hots pass through unnormalized so a
rare goal's tiny variance can't blow up.

Usage:
  .venv/bin/python -m experiments.next_packet.ablation_r0_r1 \
      --recordings "results/frozen_dryrun/rollout-*/packets.jsonl" \
      --epochs 40 --hidden 128
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
from pathlib import Path

from .features import PACKET_TYPES, PACKET_TYPE_INDEX
from .metrics import TypeMetrics

_DIM_ORDER = {"minecraft:overworld": 0.0, "minecraft:the_nether": 1.0, "minecraft:the_end": 2.0}


def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def load_examples(globs: list[str]) -> list[tuple[dict, str]]:
    """(obs, packet_type) per allowlisted packet. delta_tick computed per-file."""
    files: list[str] = []
    for g in globs:
        files.extend(sorted(glob.glob(g)))
    out: list[tuple[dict, str]] = []
    for fp in files:
        prev_tick: int | None = None
        with _open_text(fp) as f:
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
                if isinstance(tick, int) and prev_tick is not None:
                    obs["delta_tick"] = max(0, tick - prev_tick)
                else:
                    obs["delta_tick"] = 0
                if isinstance(tick, int):
                    prev_tick = tick
                out.append((obs, ptype))
    return out


class GoalVocab:
    """g_t string → index. Index 0 reserved for none/unknown (closed vocab)."""

    def __init__(self, goals: list[str]) -> None:
        self.goals = list(goals)
        self.index = {g: i + 1 for i, g in enumerate(self.goals)}

    @classmethod
    def fit(cls, examples: list[tuple[dict, str]]) -> "GoalVocab":
        s = {o["g_t"] for o, _ in examples if o.get("g_t") is not None}
        return cls(sorted(s))

    @property
    def size(self) -> int:
        return len(self.goals) + 1

    def names(self) -> list[str]:
        return ["goal=<none/unk>"] + [f"goal={g}" for g in self.goals]

    def onehot(self, g_t) -> list[float]:
        v = [0.0] * self.size
        v[self.index.get(g_t, 0) if g_t is not None else 0] = 1.0
        return v


# (name, normalize?) — booleans/one-hots/sin-cos pass through unnormalized.
_R0 = [("x", True), ("y", True), ("z", True),
       ("sin_yaw", False), ("cos_yaw", False), ("sin_pitch", False), ("cos_pitch", False),
       ("on_ground", False), ("dim_id", False)]
_R1_TEMPORAL = [("ticks_since_g_t_issued", True), ("delta_tick", True)]


# Arms disentangle the two things R1 adds, so an R0→R1 gain can be attributed
# to goal *identity* vs *timing* (both mining goals emit swing+rot, so temporal
# features, not the goal, are the likely separator there — §8c needs this split).
ARMS: dict[str, set[str]] = {
    "R0":          set(),                 # base pose only
    "R1_goal":     {"goal"},              # + g_t one-hot
    "R1_temporal": {"temporal"},          # + ticks_since_g_t_issued, delta_tick
    "R1_full":     {"goal", "temporal"},  # + both (the spec's R1)
}


def layout(vocab: GoalVocab, groups: set[str]) -> list[tuple[str, bool]]:
    cols = list(_R0)
    if "goal" in groups:
        cols += [(n, False) for n in vocab.names()]
    if "temporal" in groups:
        cols += _R1_TEMPORAL
    return cols


def featurize(obs: dict, vocab: GoalVocab, groups: set[str]) -> list[float]:
    yaw = math.radians(float(obs.get("yaw", 0.0)))
    pitch = math.radians(float(obs.get("pitch", 0.0)))
    feats = [
        float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("z", 0.0)),
        math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
        1.0 if obs.get("on_ground") else 0.0,
        _DIM_ORDER.get(str(obs.get("dim", "")), -1.0),
    ]
    if "goal" in groups:
        feats += vocab.onehot(obs.get("g_t"))
    if "temporal" in groups:
        feats += [float(obs.get("ticks_since_g_t_issued") or 0), float(obs.get("delta_tick") or 0)]
    return feats


class Normalizer:
    """Z-score only the columns flagged for normalization; pass the rest through."""

    def __init__(self, cols: list[tuple[str, bool]]) -> None:
        self.flags = [norm for _, norm in cols]
        self.mean = [0.0] * len(cols)
        self.std = [1.0] * len(cols)

    def fit(self, vecs: list[list[float]]) -> "Normalizer":
        n = len(vecs)
        if n == 0:
            return self
        for i, flag in enumerate(self.flags):
            if not flag:
                continue
            col = [v[i] for v in vecs]
            mean = sum(col) / n
            var = sum(x * x for x in col) / n - mean * mean
            self.mean[i] = mean
            self.std[i] = max(math.sqrt(max(var, 0.0)), 1e-6)
        return self

    def transform(self, vec: list[float]) -> list[float]:
        return [(vec[i] - self.mean[i]) / self.std[i] if self.flags[i] else vec[i]
                for i in range(len(vec))]


def train_arm(
    examples: list[tuple[dict, str]],
    vocab: GoalVocab,
    label: str,
    groups: set[str],
    *,
    val_idx: set[int],
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[float, TypeMetrics, int]:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    cols = layout(vocab, groups)
    raw = [(featurize(o, vocab, groups), PACKET_TYPE_INDEX[p], p) for o, p in examples]
    train_raw = [raw[i] for i in range(len(raw)) if i not in val_idx]
    val_raw = [raw[i] for i in range(len(raw)) if i in val_idx]

    norm = Normalizer(cols).fit([f for f, _, _ in train_raw])
    train = [(norm.transform(f), y, p) for f, y, p in train_raw]
    val = [(norm.transform(f), y, p) for f, y, p in val_raw]

    input_dim = len(cols)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(input_dim, hidden), nn.ReLU(),
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
    best_acc, best_metrics = 0.0, TypeMetrics()
    for _epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), batch_size):
            xs, ys = tens(train[i:i + batch_size])
            opt.zero_grad()
            loss = crit(model(xs), ys)
            loss.backward()
            opt.step()
        model.eval()
        m = TypeMetrics()
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch = val[i:i + batch_size]
                xs, _ = tens(batch)
                preds = model(xs).argmax(dim=-1).tolist()
                for (_, _, ptype), pi in zip(batch, preds):
                    m.update(ptype, PACKET_TYPES[pi])
        acc = m.overall_accuracy()
        if acc >= best_acc:
            best_acc, best_metrics = acc, m
    return best_acc, best_metrics, input_dim


def main() -> None:
    ap = argparse.ArgumentParser(description="R0→R1 next-packet discriminator ablation (§8c)")
    ap.add_argument("--recordings", nargs="+",
                    default=["results/frozen_dryrun/rollout-*/packets.jsonl"])
    ap.add_argument("--epochs", type=int, default=40)
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

    globs = [str(Path(p)) for p in args.recordings]
    examples = load_examples(globs)
    if not examples:
        print(f"No examples from {globs}", file=sys.stderr)
        sys.exit(1)

    vocab = GoalVocab.fit(examples)
    # Shared val split so every arm is evaluated on identical packets.
    rng = random.Random(args.seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_idx = set(idx[:n_val])

    print(f"examples={len(examples)}  val={n_val}  goal_vocab={vocab.names()}")
    freq = TypeMetrics()
    for _, p in examples:
        freq.update_counts_only(p)
    print("\n" + freq.frequency_baseline_report())

    arm_order = list(ARMS.keys())
    results = {}
    for label in arm_order:
        acc, m, dim = train_arm(
            examples, vocab, label, ARMS[label], val_idx=val_idx,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
        )
        results[label] = (acc, m, dim)
        print(f"\n===== {label} (input_dim={dim}) best val_acc={acc:.4f} =====")
        print(m.report(rung=label))

    # Comparison table: overall + per-type across all arms, vs R0.
    print("\n" + "=" * 78)
    print("R0 → R1 ablation (val top-1 accuracy) — goal vs temporal disentangled")
    print("=" * 78)
    r0_acc = results["R0"][0]
    head = "  " + f"{'arm':<14}" + f"{'dim':>5}" + f"{'overall':>10}" + f"{'Δ vs R0':>10}"
    print(head)
    for label in arm_order:
        acc, _m, dim = results[label]
        print(f"  {label:<14}{dim:>5}{acc:>10.4f}{acc - r0_acc:>+10.4f}")

    print(f"\n  per-type accuracy by arm:")
    print("  " + f"{'type':<34}{'n':>5}" + "".join(f"{a:>13}" for a in arm_order))
    r1_m = results["R1_full"][1]
    for t in PACKET_TYPES:
        n = r1_m._total[t]
        if n == 0:
            continue
        cells = ""
        for label in arm_order:
            a = results[label][1].accuracy(t)
            cells += f"{(a if a is not None else float('nan')):>13.3f}"
        print(f"  {t:<34}{n:>5}{cells}")


if __name__ == "__main__":
    main()
