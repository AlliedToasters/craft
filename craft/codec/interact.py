"""interact codec — entity right-click / attack (ml.MD §4a).

Three sub-actions distinguished by the recorder's dispatch visitor:

  * ``ATTACK``      — left-click an entity. Wire: ``entity_id`` +
                      ``using_secondary_action`` only.
  * ``INTERACT``    — right-click an entity (with hand). Wire adds ``hand``.
  * ``INTERACT_AT`` — right-click at a specific spot on the entity
                      (e.g. saddle area on a horse). Wire adds an
                      *entity-relative* ``at`` vec3 in entity-local axes.

**The conditional-fields pattern**, now spanning three codec types — the
neural head treats them uniformly via ``semantic_fields``, but the codec
carrying differs depending on what the wire actually does:

  ┌─────────────────────────────────────┬────────────────────────────┬──────────────────────────────┐
  │ Wire behavior                       │ Codec representation       │ Codec example                │
  ├─────────────────────────────────────┼────────────────────────────┼──────────────────────────────┤
  │ Field literally absent for some     │ ``Optional[T]`` field;     │ interact: ``hand``, ``at``   │
  │ action values                       │ ``__post_init__`` validates│                              │
  │                                     │ presence matches action    │                              │
  ├─────────────────────────────────────┼────────────────────────────┼──────────────────────────────┤
  │ Field always on wire; value is      │ Non-optional field carried │ player_action: ``block_pos`` │
  │ sentinel-zero for some action       │ verbatim; meaningfulness   │ and ``face`` for non-spatial │
  │ values                              │ exposed via                │ actions                      │
  │                                     │ ``semantic_fields``        │                              │
  ├─────────────────────────────────────┼────────────────────────────┼──────────────────────────────┤
  │ Field always on wire; presence is   │ Wire-type discriminator    │ move: ``pos`` and ``rot``    │
  │ a function of the wire type         │ + Optional fields; both    │ across the 4 wire types      │
  │ discriminator                       │ validated against type     │                              │
  └─────────────────────────────────────┴────────────────────────────┴──────────────────────────────┘

All three converge on ``semantic_fields()`` — the set of field names the
neural head predicts and downstream consumers treat as meaningful. The
codec carrying is the substrate detail; the prediction surface is uniform.

**Pointer gap**: ``entity_id`` is encoded as the raw client-side entity id
pending a local entity-set observation channel. When that lands the codec
swaps ``entity_id: int`` for ``entity_idx: int`` (pointer into observed
entities). Localized change — same pattern as ``block_pos`` in
``use_item_on`` / ``player_action``.

The ``at`` vec3 for ``INTERACT_AT`` is already entity-relative on the wire
(client computes it from raycast), so no transform happens at the codec
boundary — straight pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:interact"]

_ACTIONS = frozenset({"ATTACK", "INTERACT", "INTERACT_AT"})
_HANDS = frozenset({"MAIN_HAND", "OFF_HAND"})


@dataclass(frozen=True)
class InteractAction:
    """Structured form of a ``ServerboundInteractPacket``.

    Optional fields encode "literally absent on the wire" semantics:
    ``hand=None`` for ``ATTACK``; ``at=None`` for ``ATTACK`` and
    ``INTERACT``. ``__post_init__`` enforces the wire-shape contract so
    constructing an inconsistent action raises immediately.
    """

    packet_type: str
    entity_id: int
    action: str
    using_secondary_action: bool
    hand: str | None
    at: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(f"InteractAction: invalid packet_type {self.packet_type!r}")
        if self.action not in _ACTIONS:
            raise ValueError(f"InteractAction: invalid action {self.action!r}")
        if self.hand is not None and self.hand not in _HANDS:
            raise ValueError(f"InteractAction: invalid hand {self.hand!r}")
        # Action-conditional presence rules — straight from the wire shape.
        expects_hand = self.action in ("INTERACT", "INTERACT_AT")
        expects_at = self.action == "INTERACT_AT"
        if expects_hand != (self.hand is not None):
            raise ValueError(
                f"InteractAction: action={self.action!r} hand presence mismatch "
                f"(hand={self.hand!r})"
            )
        if expects_at != (self.at is not None):
            raise ValueError(
                f"InteractAction: action={self.action!r} at presence mismatch "
                f"(at={self.at!r})"
            )
        if self.at is not None and len(self.at) != 3:
            raise ValueError(f"InteractAction: at must be 3 floats, got {self.at!r}")

    @property
    def semantic_fields(self) -> frozenset[str]:
        """Field names whose values are semantically meaningful for this
        action — what the neural head predicts and what consumers treat as
        load-bearing. ``hand`` and ``at`` participate only when present."""
        fields = {"entity_id", "action", "using_secondary_action"}
        if self.hand is not None:
            fields.add("hand")
        if self.at is not None:
            fields.add("at")
        return frozenset(fields)


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    hand = fields.get("hand")
    at_raw = fields.get("at")
    at: tuple[float, float, float] | None = None
    if at_raw is not None:
        at = (float(at_raw[0]), float(at_raw[1]), float(at_raw[2]))
    return InteractAction(
        packet_type=packet_type,
        entity_id=int(fields["entity_id"]),
        action=str(fields["action"]),
        using_secondary_action=bool(fields["using_secondary_action"]),
        hand=str(hand) if hand is not None else None,
        at=at,
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, InteractAction):
        raise TypeError(
            f"interact._decode: expected InteractAction, got {type(action).__name__}"
        )
    # Match the recorder's JSONL shape exactly: omit hand/at keys when absent
    # (the Java extractor uses ``if (x != null) m.put(...)``). Round-trip
    # equality is checked with ``fields_close`` which compares keys first,
    # so emitting None-keyed entries would spuriously fail.
    out: dict[str, Any] = {
        "entity_id": action.entity_id,
        "using_secondary_action": action.using_secondary_action,
        "action": action.action,
    }
    if action.hand is not None:
        out["hand"] = action.hand
    if action.at is not None:
        out["at"] = [action.at[0], action.at[1], action.at[2]]
    return out


register(PACKET_TYPES, _encode, _decode)

__all__ = ["InteractAction", "PACKET_TYPES"]
