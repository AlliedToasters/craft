"""Checkpoint save/load for the next-packet prediction model.

Saves model state dict + normalizer parameters + run metadata so the
inference server can reconstruct the model without re-importing the
training loop.

Checkpoint format (torch .pt file, top-level dict):
  {
    "model_state": OrderedDict,          # model.state_dict()
    "arch": {
      "input_dim": int,
      "hidden": int,
      "n_types": int,
    },
    "normalizer": {
      "mean": list[float],
      "std": list[float],
      "feature_names": list[str],
    },
    "metadata": {
      "rung": str,
      "epoch": int,
      "val_acc": float,
      "train_loss": float,
      "n_train": int,
      "n_val": int,
      "packet_types": list[str],        # PACKET_TYPES ordering used at training
    },
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES, NUM_PACKET_TYPES, PACKET_TYPES, FeatureNormalizer


def save_checkpoint(
    model: Any,
    normalizer: FeatureNormalizer,
    path: str | Path,
    *,
    rung: str = "R?",
    epoch: int = 0,
    val_acc: float = 0.0,
    train_loss: float = 0.0,
    n_train: int = 0,
    n_val: int = 0,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": {
                "input_dim": len(FEATURE_NAMES),
                "hidden": _infer_hidden(model),
                "n_types": NUM_PACKET_TYPES,
            },
            "normalizer": {
                "mean": normalizer.mean,
                "std": normalizer.std,
                "feature_names": list(FEATURE_NAMES),
            },
            "metadata": {
                "rung": rung,
                "epoch": epoch,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "n_train": n_train,
                "n_val": n_val,
                "packet_types": list(PACKET_TYPES),
            },
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[Any, FeatureNormalizer, dict]:
    """Load checkpoint → (model, normalizer, metadata).

    Returns a model in eval mode on the best available device. The caller
    does not need to know the architecture — it's reconstructed from the
    checkpoint.
    """
    import torch
    import torch.nn as nn

    path = Path(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    arch = ckpt["arch"]

    class _Model(nn.Module):
        def __init__(self, input_dim: int, hidden: int, n_types: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.discriminator = nn.Linear(hidden, n_types)

        def forward(self, x: Any) -> Any:
            return self.discriminator(self.trunk(x))

    model = _Model(arch["input_dim"], arch["hidden"], arch["n_types"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = FeatureNormalizer()
    norm.mean = ckpt["normalizer"]["mean"]
    norm.std = ckpt["normalizer"]["std"]
    norm._fitted = True

    return model, norm, ckpt["metadata"]


def _infer_hidden(model: Any) -> int:
    """Extract the hidden dim from the trunk's first Linear layer."""
    try:
        return model.trunk[0].out_features
    except (AttributeError, IndexError):
        return 256
