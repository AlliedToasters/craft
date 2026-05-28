"""Per-type accuracy / entropy tracker for next-packet prediction.

Pure Python — no ML deps. Accumulates (true_type, predicted_type) pairs
and prints the pre-registered (type × rung) table described in
neural_interface.md §8c. The tracker is designed to be used from a
training loop (accumulate per batch, report per epoch) and from the
baseline frequency analysis (zero-model baseline, no training needed).

Usage::

    from experiments.next_packet.metrics import TypeMetrics

    m = TypeMetrics()
    for obs, action in pairs:
        # training loop: call m.update(true_type, predicted_type) per example
        m.update(action.packet_type, predicted_type)

    print(m.report())

Zero-model baseline (frequency prior)::

    m = TypeMetrics()
    for obs, action in pairs:
        m.update_counts_only(action.packet_type)
    print(m.frequency_baseline_report())
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .features import PACKET_TYPES


class TypeMetrics:
    """Accumulates per-type correct/total counts for the discriminator head."""

    def __init__(self) -> None:
        self._total: Counter[str] = Counter()
        self._correct: Counter[str] = Counter()
        # For entropy / calibration: count predicted types per true type.
        # _confusion[true][pred] = count
        self._confusion: dict[str, Counter[str]] = {}

    def update(self, true_type: str, predicted_type: str) -> None:
        self._total[true_type] += 1
        if true_type == predicted_type:
            self._correct[true_type] += 1
        self._confusion.setdefault(true_type, Counter())[predicted_type] += 1

    def update_counts_only(self, true_type: str) -> None:
        """Record a true type without a prediction (frequency analysis only)."""
        self._total[true_type] += 1

    def accuracy(self, packet_type: str) -> float | None:
        total = self._total[packet_type]
        if total == 0:
            return None
        return self._correct[packet_type] / total

    def overall_accuracy(self) -> float:
        total = sum(self._total.values())
        correct = sum(self._correct.values())
        return correct / total if total else 0.0

    def frequency_prior(self) -> dict[str, float]:
        """Return per-type empirical frequency (p(type)) over seen examples."""
        total = sum(self._total.values())
        if total == 0:
            return {}
        return {t: self._total[t] / total for t in self._total}

    def frequency_baseline_accuracy(self) -> float:
        """Accuracy of a model that always predicts the most common type."""
        if not self._total:
            return 0.0
        majority = self._total.most_common(1)[0][0]
        total = sum(self._total.values())
        return self._total[majority] / total

    def entropy_of_prior(self) -> float:
        """Shannon entropy (bits) of the empirical type distribution."""
        prior = self.frequency_prior()
        h = 0.0
        for p in prior.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def report(self, *, rung: str = "R?") -> str:
        """Multi-line table: per-type accuracy + overall + frequency baseline."""
        lines = [f"  Next-packet discriminator — rung {rung}"]
        lines.append(f"  {'Type':<45}  {'N':>6}  {'Acc':>7}  {'Freq':>7}")
        lines.append("  " + "-" * 72)
        prior = self.frequency_prior()
        all_types = sorted(
            set(PACKET_TYPES) | set(self._total.keys()),
            key=lambda t: -self._total.get(t, 0),
        )
        for t in all_types:
            n = self._total.get(t, 0)
            acc = self.accuracy(t)
            freq = prior.get(t, 0.0)
            acc_s = f"{acc:.3f}" if acc is not None else "   —  "
            lines.append(f"  {t:<45}  {n:>6d}  {acc_s:>7}  {freq:>7.3f}")
        lines.append("  " + "-" * 72)
        lines.append(
            f"  {'OVERALL':<45}  {sum(self._total.values()):>6d}  "
            f"{self.overall_accuracy():>7.3f}  "
        )
        lines.append(
            f"  {'frequency baseline':<45}  {'':>6}  "
            f"{self.frequency_baseline_accuracy():>7.3f}  "
        )
        lines.append(
            f"  {'prior entropy (bits)':<45}  {'':>6}  "
            f"{self.entropy_of_prior():>7.3f}  "
        )
        return "\n".join(lines)

    def frequency_baseline_report(self) -> str:
        """Report without accuracy (count + frequency only). For dataset analysis."""
        lines = ["  Dataset type distribution"]
        lines.append(f"  {'Type':<45}  {'N':>6}  {'Freq':>7}")
        lines.append("  " + "-" * 60)
        prior = self.frequency_prior()
        for t, n in self._total.most_common():
            lines.append(f"  {t:<45}  {n:>6d}  {prior.get(t,0):.3f}")
        lines.append("  " + "-" * 60)
        total = sum(self._total.values())
        lines.append(f"  {'TOTAL':<45}  {total:>6d}")
        lines.append(f"  majority baseline acc: {self.frequency_baseline_accuracy():.3f}")
        lines.append(f"  prior entropy (bits):  {self.entropy_of_prior():.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-registered (type × rung) hypotheses from neural_interface.md §8c
# ---------------------------------------------------------------------------

# Each entry: (wire_type, rung_from, rung_to, kind, description)
# kind: "step" = predicted to move; "flat" = predicted to stay flat
PREREGISTERED: list[tuple[str, str, str, str, str]] = [
    ("minecraft:interact",
     "R0", "R1", "step",
     "ATTACK vs INTERACT vs INTERACT_AT: g_t disambiguates intent"),
    ("minecraft:player_command",
     "R0", "R1", "step",
     "sprint/sneak edges are goal-driven; goal switch → sprint change"),
    ("minecraft:use_item_on",
     "R2", "R3", "step",
     "block_pos pointer gap closes; pointer head replaces absolute MSE"),
    ("minecraft:interact",
     "R2", "R3", "step",
     "entity_id pointer gap closes"),
    ("minecraft:move_player_pos_rot",
     "R0", "R1", "flat",
     "movement deltas are Baritone path-following, not LLM intent"),
    ("minecraft:move_player_pos",
     "R0", "R1", "flat",
     "movement deltas are Baritone path-following, not LLM intent"),
    ("minecraft:move_player_rot",
     "R0", "R1", "flat",
     "movement deltas are Baritone path-following, not LLM intent"),
    ("minecraft:swing",
     "R0", "R3", "flat",
     "hand is near-deterministic from inventory state; high acc at R0"),
]


def preregistered_report(
    rung_metrics: dict[str, TypeMetrics],
    rung_order: Sequence[str] = ("R0", "R1", "R2", "R3", "R4"),  # noqa: ARG001
) -> str:
    """Compare per-type accuracy across rungs against pre-registered predictions.

    ``rung_metrics`` maps rung name → TypeMetrics with predictions recorded.
    Only prints rows where both the from-rung and to-rung are in the dict.
    """
    lines = ["  Pre-registered (type × rung) predictions"]
    lines.append(
        f"  {'Type':<45}  {'Kind':<5}  {'From':>4}  {'To':>4}  {'ΔAcc':>7}  "
        f"{'Confirmed':<9}  Description"
    )
    lines.append("  " + "-" * 120)
    for wire_type, r_from, r_to, kind, desc in PREREGISTERED:
        m_from = rung_metrics.get(r_from)
        m_to = rung_metrics.get(r_to)
        acc_from = m_from.accuracy(wire_type) if m_from else None
        acc_to = m_to.accuracy(wire_type) if m_to else None
        if acc_from is None or acc_to is None:
            delta_s = "   n/a "
            conf = "n/a      "
        else:
            delta = acc_to - acc_from
            delta_s = f"{delta:+.3f}"
            if kind == "step":
                confirmed = "YES" if delta > 0.05 else ("NO" if delta < -0.02 else "~")
            else:  # flat
                confirmed = "YES" if abs(delta) < 0.02 else ("NO" if delta > 0.05 else "~")
            conf = f"{confirmed:<9}"
        lines.append(
            f"  {wire_type:<45}  {kind:<5}  {r_from:>4}  {r_to:>4}  "
            f"{delta_s:>7}  {conf}  {desc}"
        )
    return "\n".join(lines)
