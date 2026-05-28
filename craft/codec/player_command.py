"""player_command codec — discrete state-edge events (ml.MD §4a).

Distinct from ``player_input`` (which is *held* keyboard state): this packet
fires once per discrete event — sprint started/stopped, shift pressed/released,
horse-jump charged, fall-flying engaged, etc.

The wire fields:

  * ``entity_id`` — the subject. The player's own runtime id when the event
    is about the player (the common case); the vehicle's id when riding.
    Same pointer gap as ``interact.entity_id`` — encoded absolute pending the
    local entity-set observation channel.
  * ``action`` — 9-way enum (PRESS_SHIFT_KEY, RELEASE_SHIFT_KEY, STOP_SLEEPING,
    START/STOP_SPRINTING, START/STOP_RIDING_JUMP, OPEN_INVENTORY,
    START_FALL_FLYING).
  * ``data`` — action-specific int payload (e.g. horse jump charge). Zero
    for most actions; carried verbatim.

No plumbing fields — sequence numbers don't apply to this packet type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:player_command"]

_ACTIONS = frozenset({
    "PRESS_SHIFT_KEY",
    "RELEASE_SHIFT_KEY",
    "STOP_SLEEPING",
    "START_SPRINTING",
    "STOP_SPRINTING",
    "START_RIDING_JUMP",
    "STOP_RIDING_JUMP",
    "OPEN_INVENTORY",
    "START_FALL_FLYING",
})


@dataclass(frozen=True)
class PlayerCommandAction:
    packet_type: str
    entity_id: int
    action: str
    data: int

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(
                f"PlayerCommandAction: invalid packet_type {self.packet_type!r}"
            )
        if self.action not in _ACTIONS:
            raise ValueError(f"PlayerCommandAction: invalid action {self.action!r}")

    @property
    def semantic_fields(self) -> frozenset[str]:
        return frozenset({"entity_id", "action", "data"})


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    return PlayerCommandAction(
        packet_type=packet_type,
        entity_id=int(fields["entity_id"]),
        action=str(fields["action"]),
        data=int(fields["data"]),
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, PlayerCommandAction):
        raise TypeError(
            f"player_command._decode: expected PlayerCommandAction, "
            f"got {type(action).__name__}"
        )
    return {
        "entity_id": action.entity_id,
        "action": action.action,
        "data": action.data,
    }


register(PACKET_TYPES, _encode, _decode)

__all__ = ["PlayerCommandAction", "PACKET_TYPES"]
