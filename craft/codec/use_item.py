"""use_item codec — right-click in empty space (ml.MD §4a).

Distinct from ``use_item_on``: this packet fires when the right-click hits
nothing (no block target). Examples: eating food, drawing a bow, throwing an
ender pearl, snowball, splash potion. The held item determines what happens
server-side.

Wire fields:

  * ``hand``    — MAIN_HAND / OFF_HAND.
  * ``yaw``,
    ``pitch``   — the player's look direction at click time. Anti-cheat
                  bookkeeping the server uses to validate the action;
                  reconstructable from the obs but the wire carries it
                  explicitly so the codec records it for round-trip parity.
  * ``sequence`` — plumbing, mechanically generated at packet construction
                  (same status as in ``use_item_on`` / ``player_action``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:use_item"]

_HANDS = frozenset({"MAIN_HAND", "OFF_HAND"})


@dataclass(frozen=True)
class UseItemAction:
    packet_type: str
    hand: str
    yaw: float
    pitch: float
    sequence: int

    _is_plumbing: ClassVar[tuple[str, ...]] = ("sequence",)

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(f"UseItemAction: invalid packet_type {self.packet_type!r}")
        if self.hand not in _HANDS:
            raise ValueError(f"UseItemAction: invalid hand {self.hand!r}")

    @property
    def semantic_fields(self) -> frozenset[str]:
        """Sequence is plumbing per ``_is_plumbing`` and excluded."""
        return frozenset({"hand", "yaw", "pitch"})


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    return UseItemAction(
        packet_type=packet_type,
        hand=str(fields["hand"]),
        yaw=float(fields["yaw"]),
        pitch=float(fields["pitch"]),
        sequence=int(fields["sequence"]),
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, UseItemAction):
        raise TypeError(
            f"use_item._decode: expected UseItemAction, got {type(action).__name__}"
        )
    return {
        "hand": action.hand,
        "sequence": action.sequence,
        "yaw": action.yaw,
        "pitch": action.pitch,
    }


register(PACKET_TYPES, _encode, _decode)

__all__ = ["UseItemAction", "PACKET_TYPES"]
