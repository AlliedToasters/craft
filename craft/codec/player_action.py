"""player_action codec — block-break lifecycle + inventory edges (ml.MD §4a).

The 7 wire actions split into two semantic groups:

  * **Spatial (block-targeted)**: ``START_DESTROY_BLOCK``, ``ABORT_DESTROY_BLOCK``,
    ``STOP_DESTROY_BLOCK`` — the dig-progress lifecycle. ``block_pos`` and
    ``face`` are meaningful pointers; the neural policy points at a block
    and a face.
  * **Non-spatial (no block target)**: ``DROP_ALL_ITEMS``, ``DROP_ITEM``,
    ``RELEASE_USE_ITEM``, ``SWAP_ITEM_WITH_OFFHAND`` — inventory / hand
    state edges. The wire still carries ``block_pos`` and ``face``
    fields (the protocol shape is fixed), but the client conventionally
    sets them to ``(0, 0, 0)`` / ``DOWN``. The codec preserves those values
    verbatim for round-trip parity.

Implication for the neural head: the head emits a structured action where
``block_pos`` and ``face`` are conditionally consumed downstream based on
the action enum. For spatial actions the policy attends over the local
block grid; for non-spatial actions the parameters are filled mechanically
at packet-construction. Phase 2 codec keeps both groups in one tagged
record (the wire-level packet is a single type — ``player_action``) and
flags the conditioning via ``_is_spatial`` so the head implementation can
mask appropriately.

**Pointer gap** (same as ``use_item_on``): ``block_pos`` is encoded as
absolute coordinates pending the local-block-grid observation channel.

**Sequence number (plumbing per §4a)**: present only for the dig-lifecycle
actions in practice (the server uses it to ack destroy-progress), zero
for the others. Carried for round-trip, flagged in ``_is_plumbing`` —
neural policy never predicts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:player_action"]

_SPATIAL_ACTIONS = frozenset({
    "START_DESTROY_BLOCK",
    "ABORT_DESTROY_BLOCK",
    "STOP_DESTROY_BLOCK",
})
_NON_SPATIAL_ACTIONS = frozenset({
    "DROP_ALL_ITEMS",
    "DROP_ITEM",
    "RELEASE_USE_ITEM",
    "SWAP_ITEM_WITH_OFFHAND",
})
_ACTIONS = _SPATIAL_ACTIONS | _NON_SPATIAL_ACTIONS

_FACES = frozenset({"DOWN", "UP", "NORTH", "SOUTH", "WEST", "EAST"})


@dataclass(frozen=True)
class PlayerActionAction:
    """Structured form of a ``ServerboundPlayerActionPacket``.

    All four wire fields are carried explicitly: the protocol shape is fixed,
    so even non-spatial actions get the convention-zero ``block_pos`` /
    ``face: DOWN``. The codec preserves them verbatim for byte-equivalent
    round-trip; the neural head will mask block_pos/face for non-spatial
    actions at inference time.
    """

    packet_type: str
    action: str
    block_pos: tuple[int, int, int]
    face: str
    sequence: int

    _is_plumbing: ClassVar[tuple[str, ...]] = ("sequence",)

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(
                f"PlayerActionAction: invalid packet_type {self.packet_type!r}"
            )
        if self.action not in _ACTIONS:
            raise ValueError(f"PlayerActionAction: invalid action {self.action!r}")
        if self.face not in _FACES:
            raise ValueError(f"PlayerActionAction: invalid face {self.face!r}")
        if len(self.block_pos) != 3 or any(
            not isinstance(v, int) for v in self.block_pos
        ):
            raise ValueError(
                f"PlayerActionAction: block_pos must be 3 ints, got {self.block_pos!r}"
            )

    @property
    def is_spatial(self) -> bool:
        """``True`` if this action's ``block_pos``/``face`` are pointer-rich;
        ``False`` if they're convention-zero filler. Used by the neural head
        to mask the spatial parameter outputs for non-block actions."""
        return self.action in _SPATIAL_ACTIONS

    @property
    def semantic_fields(self) -> frozenset[str]:
        """Field names the neural head predicts. ``action`` is always semantic;
        ``block_pos`` / ``face`` participate only when ``is_spatial``. Sequence
        is plumbing and excluded."""
        fields = {"action"}
        if self.is_spatial:
            fields.add("block_pos")
            fields.add("face")
        return frozenset(fields)


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    bp_raw = fields["block_pos"]
    bp = (int(bp_raw[0]), int(bp_raw[1]), int(bp_raw[2]))
    return PlayerActionAction(
        packet_type=packet_type,
        action=str(fields["action"]),
        block_pos=bp,
        face=str(fields["face"]),
        sequence=int(fields["sequence"]),
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, PlayerActionAction):
        raise TypeError(
            f"player_action._decode: expected PlayerActionAction, got {type(action).__name__}"
        )
    return {
        "action": action.action,
        "block_pos": [action.block_pos[0], action.block_pos[1], action.block_pos[2]],
        "face": action.face,
        "sequence": action.sequence,
    }


register(PACKET_TYPES, _encode, _decode)

__all__ = ["PlayerActionAction", "PACKET_TYPES"]
