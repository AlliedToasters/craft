"""JSONL dataset for next-packet prediction (neural_interface.md §8).

Reads PacketRecorder JSONL files (one line per captured outbound packet),
calls ``craft.codec.encode`` to build the structured Action label, and
yields ``(obs, action)`` pairs. Pure Python — no ML deps.

Data contract (each JSONL line):
  {
    "ts_ms": int,
    "id":    str,       -- the packet_type / discriminator
    "fields": {...},    -- raw wire fields
    "obs":   {
      "tick": int,
      "captured_at_ms": int,
      "x", "y", "z": float,
      "yaw", "pitch": float,
      "on_ground": bool,
      "dim": str,
      -- R1 additions (optional, null when absent):
      "g_t": str | null,
      "ticks_since_g_t_issued": int | null,
    }
  }

Lines whose ``id`` has no registered codec are skipped (counted in
``skipped_unknown``). Lines where encode raises are also skipped
(counted in ``skipped_encode_error``), with a warning on the first
occurrence per type.

Usage::

    from experiments.next_packet.dataset import load_recordings

    pairs = load_recordings(["/path/to/recording-*.jsonl"])
    for obs, action in pairs:
        print(action.packet_type, action.semantic_fields)
"""

from __future__ import annotations

import glob
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

# craft.codec.__init__ imports all per-type modules, triggering registration.
from craft.codec import Action, encode, is_registered


@dataclass
class LoadStats:
    files: int = 0
    lines_read: int = 0
    lines_yielded: int = 0
    skipped_unknown: int = 0
    skipped_unimplemented: int = 0   # Java extractor returned {"_unimplemented": True}
    skipped_encode_error: int = 0
    # per-type counts of successfully yielded examples
    per_type: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_type is None:
            self.per_type = {}

    def record(self, packet_type: str) -> None:
        self.per_type[packet_type] = self.per_type.get(packet_type, 0) + 1
        self.lines_yielded += 1

    def summary(self) -> str:
        lines = [
            f"files={self.files}  read={self.lines_read}  "
            f"yielded={self.lines_yielded}  "
            f"skip_unknown={self.skipped_unknown}  "
            f"skip_unimplemented={self.skipped_unimplemented}  "
            f"skip_encode_err={self.skipped_encode_error}"
        ]
        if self.per_type:
            lines.append("per type:")
            for t, n in sorted(self.per_type.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {n:6d}  {t}")
        return "\n".join(lines)


def iter_jsonl(path: Path) -> Generator[dict, None, None]:
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                warnings.warn(f"{path}: bad JSON line: {e}")


def load_recordings(
    sources: Iterable[str | Path],
    *,
    stats: LoadStats | None = None,
) -> Generator[tuple[dict, Action], None, None]:
    """Yield ``(obs_dict, Action)`` pairs from one or more recording paths.

    ``sources`` may contain file paths or glob patterns. Files are read in
    the order given; within a file, lines are yielded in order (chronological).

    ``stats`` is updated in-place if provided; construct a ``LoadStats()``
    and pass it to get a breakdown by type after iteration.
    """
    if stats is None:
        stats = LoadStats()

    warned_encode: set[str] = set()

    for source in sources:
        paths = sorted(glob.glob(str(source))) if "*" in str(source) else [Path(source)]
        for path in paths:
            path = Path(path)
            if not path.exists():
                warnings.warn(f"recording not found: {path}")
                continue
            stats.files += 1
            for entry in iter_jsonl(path):
                stats.lines_read += 1
                packet_id = entry.get("id", "")
                fields = entry.get("fields", {})
                obs = entry.get("obs", {})

                if not is_registered(packet_id):
                    stats.skipped_unknown += 1
                    continue

                # Old recordings captured before Java-side extraction was wired
                # for this packet type. Not a codec bug — skip cleanly.
                if fields.get("_unimplemented"):
                    stats.skipped_unimplemented += 1
                    continue

                try:
                    action = encode(packet_id, fields, obs)
                except Exception as e:
                    stats.skipped_encode_error += 1
                    if packet_id not in warned_encode:
                        warned_encode.add(packet_id)
                        warnings.warn(
                            f"encode failed for {packet_id!r} "
                            f"(first occurrence): {type(e).__name__}: {e}"
                        )
                    continue

                stats.record(packet_id)
                yield obs, action


def load_from_default_recordings(glob_pattern: str = "~/.homunculus/recordings/*.jsonl") -> tuple[
    list[tuple[dict, Action]], LoadStats
]:
    """Convenience: load all recordings matching the default path. Returns
    the full list in memory (suitable for small datasets) + stats."""
    expanded = str(Path(glob_pattern).expanduser())
    stats = LoadStats()
    pairs = list(load_recordings([expanded], stats=stats))
    return pairs, stats
