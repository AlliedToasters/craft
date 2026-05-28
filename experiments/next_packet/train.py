"""Next-packet prediction training loop (neural_interface.md §8).

Requires: torch (not in craft's default deps — install separately).

Architecture: MLP trunk (obs features → hidden) + 11 type-conditioned
parameter heads. At training time all 11 bundles emit in parallel; loss
is masked per instance by packet_type (discriminator) and semantic_fields
(within-type). At inference, sample the discriminator then run heads for
the picked type only.

This file handles:
  - Data loading + train/val split
  - Feature extraction (features.py)
  - MLP trunk + discriminator head (§8, "MLP trunk + 11 heads")
  - Per-type metrics accumulation (metrics.py)
  - Discriminator cross-entropy loss (the first and cheapest signal)

Parameter heads beyond the discriminator are scaffolded as stubs — they
are present in the model class but contribute zero loss until wired up.
The discriminator is the first learning signal; parameter heads are the
second. Run with ``--discriminator-only`` (the default) to train just the
discriminator and get the R0 per-type accuracy table. Pass
``--all-heads`` to enable parameter head losses once the stubs are wired.

Usage::

    # Dry run — exercises data pipeline, no torch needed:
    python -m experiments.next_packet.train --dry-run

    # R0 discriminator training:
    python -m experiments.next_packet.train \
        --recordings "~/.homunculus/recordings/*.jsonl" \
        --epochs 20 --hidden 256 --lr 1e-3

    # With explicit rung label (for pre-registered comparison table):
    python -m experiments.next_packet.train --rung R0 ...
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from .checkpoint import save_checkpoint
from .dataset import LoadStats, load_recordings
from .features import (
    FEATURE_NAMES,
    NUM_PACKET_TYPES,
    PACKET_TYPES,
    FeatureNormalizer,
    FeatureVec,
    obs_to_features,
    packet_type_label,
)
from .metrics import TypeMetrics


def _build_dataset(
    recordings: list[str],
    *,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[tuple[FeatureVec, int, str]], list[tuple[FeatureVec, int, str]], LoadStats, FeatureNormalizer]:
    """Load recordings → (features, disc_label, packet_type) triples, split train/val.

    Returns a fitted FeatureNormalizer alongside the splits. Features are
    z-scored using stats from the training split only (no val leakage).
    """
    stats = LoadStats()
    raw: list[tuple[dict, int, str]] = []
    for obs, action in load_recordings(recordings, stats=stats):
        label = packet_type_label(action.packet_type)
        raw.append((obs, label, action.packet_type))

    rng = random.Random(seed)
    rng.shuffle(raw)
    n_val = max(1, int(len(raw) * val_frac))
    raw_val = raw[:n_val]
    raw_train = raw[n_val:]

    norm = FeatureNormalizer()
    norm.fit([obs for obs, _, _ in raw_train])

    train = [(norm.transform(obs), label, pt) for obs, label, pt in raw_train]
    val = [(norm.transform(obs), label, pt) for obs, label, pt in raw_val]
    return train, val, stats, norm


def run_dry(recordings: list[str]) -> None:
    """Exercise the full data pipeline without torch. Prints load stats +
    dataset distribution and exits 0."""
    print("=== dry run — no training ===")
    train, val, stats, norm = _build_dataset(recordings)
    print(stats.summary())
    print()

    m = TypeMetrics()
    for _, _, pt in train + val:
        m.update_counts_only(pt)
    print(m.frequency_baseline_report())
    print()

    # Show feature vector for first example
    print("\nNormalizer (fit on train split):")
    print(norm.summary())
    if train:
        fv, _label, pt = train[0]
        print(f"\nNormalized feature vector ({len(fv)} dims) for first example ({pt!r}):")
        for name, val_f in zip(fv.names, fv.values):
            print(f"  {name:<32}  {val_f:+.6f}")
    print("dry run OK")


def train_loop(
    train: list[tuple[FeatureVec, int, str]],
    val: list[tuple[FeatureVec, int, str]],
    *,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    rung: str,
    checkpoint_path: str | None = None,
    normalizer: "FeatureNormalizer | None" = None,
) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print(
            "ERROR: torch not installed. Run:\n"
            "  pip install torch\n"
            "or use --dry-run to exercise the data pipeline only.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dim = len(FEATURE_NAMES)

    class NextPacketModel(nn.Module):
        def __init__(self, input_dim: int, hidden: int, n_types: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.discriminator = nn.Linear(hidden, n_types)
            # Parameter heads: stubs for now. Each will be a separate
            # nn.Linear(hidden, head_output_dim) when wired up.
            # Not contributed to loss until --all-heads is passed.

        def forward(self, x: Any) -> Any:
            h = self.trunk(x)
            return self.discriminator(h)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = NextPacketModel(input_dim, hidden, NUM_PACKET_TYPES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def to_tensor(batch: list[tuple[FeatureVec, int, str]]) -> tuple[Any, Any]:
        xs = torch.tensor([ex[0].values for ex in batch], dtype=torch.float32).to(device)
        ys = torch.tensor([ex[1] for ex in batch], dtype=torch.long).to(device)
        return xs, ys

    best_val_acc = 0.0
    rng = random.Random(0)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train), batch_size):
            batch = train[i : i + batch_size]
            xs, ys = to_tensor(batch)
            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        m = TypeMetrics()
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch = val[i : i + batch_size]
                xs, ys = to_tensor(batch)
                logits = model(xs)
                preds = logits.argmax(dim=-1).tolist()
                for (_, _, true_type), pred_idx in zip(batch, preds):
                    m.update(true_type, PACKET_TYPES[pred_idx])

        avg_loss = total_loss / max(n_batches, 1)
        val_acc = m.overall_accuracy()
        print(f"\nEpoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.3f}")
        print(m.report(rung=rung))

        if checkpoint_path and normalizer and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, normalizer, checkpoint_path,
                rung=rung, epoch=epoch, val_acc=val_acc,
                train_loss=avg_loss, n_train=len(train), n_val=len(val),
            )
            print(f"  ✓ checkpoint saved (val_acc={val_acc:.3f}) → {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-packet prediction trainer")
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["~/.homunculus/recordings/*.jsonl"],
        help="Glob patterns or paths to recording JSONL files",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Exercise data pipeline only; no training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rung", type=str, default="R0",
                        help="Obs rung label for metrics table (e.g. R0, R1)")
    parser.add_argument("--save-checkpoint", type=str, default=None,
                        metavar="PATH",
                        help="Save best-val-acc checkpoint to this path (e.g. checkpoints/r0.pt)")
    args = parser.parse_args()

    # Expand ~ in paths
    recordings = [str(Path(p).expanduser()) for p in args.recordings]

    if args.dry_run:
        run_dry(recordings)
        return

    print(f"Loading recordings: {recordings}")
    train, val, stats, norm = _build_dataset(recordings, val_frac=args.val_frac, seed=args.seed)
    print(stats.summary())
    print("\nNormalizer (fit on train split):")
    print(norm.summary())
    print(f"train={len(train)}  val={len(val)}")
    if not train:
        print("No training examples found. Use --dry-run to debug.", file=sys.stderr)
        sys.exit(1)

    train_loop(
        train, val,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rung=args.rung,
        checkpoint_path=args.save_checkpoint,
        normalizer=norm,
    )


if __name__ == "__main__":
    main()
