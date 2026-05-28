"""swing codec — left-click arm swing (ml.MD §4a).

Single-field packet: the hand that swung. The wire-level swing is mostly
cosmetic (visual animation broadcast to other clients) but it's the only
thing KillAura's targetless-click pipeline emits, so it's behaviorally
meaningful for recording even though it doesn't carry pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:swing"]

_HANDS = frozenset({"MAIN_HAND", "OFF_HAND"})


@dataclass(frozen=True)
class SwingAction:
    packet_type: str
    hand: str

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(f"SwingAction: invalid packet_type {self.packet_type!r}")
        if self.hand not in _HANDS:
            raise ValueError(f"SwingAction: invalid hand {self.hand!r}")

    @property
    def semantic_fields(self) -> frozenset[str]:
        return frozenset({"hand"})


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    return SwingAction(packet_type=packet_type, hand=str(fields["hand"]))


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, SwingAction):
        raise TypeError(
            f"swing._decode: expected SwingAction, got {type(action).__name__}"
        )
    return {"hand": action.hand}


register(PACKET_TYPES, _encode, _decode)

__all__ = ["SwingAction", "PACKET_TYPES"]
