"""player_input codec — keyboard-state packet (ml.MD §4a).

Seven booleans wrapping the client's input bitmask. These are *held* keyboard
states (continuous edges), distinct from ``player_command``'s discrete events.
Every tick the client may send a fresh ``player_input`` if the bitmask
changed; this is the policy's primary motor channel.

All 7 fields are semantically meaningful — there's no plumbing here, no
conditional structure. The neural head predicts a 7-dim multi-binary tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:player_input"]

_INPUT_FIELDS = (
    "forward",
    "backward",
    "left",
    "right",
    "jump",
    "shift",
    "sprint",
)


@dataclass(frozen=True)
class PlayerInputAction:
    packet_type: str
    forward: bool
    backward: bool
    left: bool
    right: bool
    jump: bool
    shift: bool
    sprint: bool

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(
                f"PlayerInputAction: invalid packet_type {self.packet_type!r}"
            )

    @property
    def semantic_fields(self) -> frozenset[str]:
        return frozenset(_INPUT_FIELDS)


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    return PlayerInputAction(
        packet_type=packet_type,
        **{name: bool(fields[name]) for name in _INPUT_FIELDS},
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, PlayerInputAction):
        raise TypeError(
            f"player_input._decode: expected PlayerInputAction, got {type(action).__name__}"
        )
    return {name: getattr(action, name) for name in _INPUT_FIELDS}


register(PACKET_TYPES, _encode, _decode)

__all__ = ["PlayerInputAction", "PACKET_TYPES"]
