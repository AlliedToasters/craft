"""use_item_on codec — right-click on a block (ml.MD §4a).

This is the first **pointer-rich** packet in the codec. Its parameter shape
is exactly what motivates the "actions are pointers into observations"
framing:

  * ``block_pos`` is the target block. In the eventual neural policy, this
    is a pointer into the local block-grid observation channel ("attention
    over the observed block tokens, pick one"). The neural model never
    regresses absolute coordinates — it points at an already-observed cell.

  * ``cursor`` is the sub-block hit point. Encoded as ``(block-relative
    offset)`` in ``[0, 1]^3``, with the axis determined by ``face`` pinned
    to 0 or 1 (DOWN: y=0; UP: y=1; NORTH: z=0; SOUTH: z=1; WEST: x=0;
    EAST: x=1). On the wire MC sends these as block-relative floats too —
    the codec preserves that shape exactly.

  * ``face`` is one of 6 directions (categorical, 6-way head).

  * ``hand`` is MAIN_HAND / OFF_HAND (categorical, 2-way head).

  * ``inside`` and ``world_border_hit`` are booleans the client computes
    from raycast geometry. We carry them through; the neural policy can
    set them mechanically (inside=False, world_border_hit=False) for any
    open-world placement and only learn them when they matter.

**Pointer gap (intentional scaffolding limitation):** we do not yet have a
local block-grid observation channel. Until that observation exists,
``block_pos`` is encoded as an *absolute* world-coord triple, not a
pointer into the obs token set. This is sufficient for the offline static
round-trip test (§4a step 1) — every captured packet round-trips by
exact identity. It is **not** sufficient for the neural training target;
the codec interface must change to consume an "observed blocks" tensor
when that observation channel lands. The dataclass shape is intentionally
designed so swapping ``block_pos: (int,int,int)`` for ``block_idx: int``
later is a localized change.

**Sequence number (plumbing per §4a):** captured for round-trip parity
but flagged via ``_is_plumbing = ("sequence",)`` on the action. The
neural policy never predicts these — they're generated mechanically at
packet-construction from the local sequence counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from craft.codec.base import Action, register


PACKET_TYPES = ["minecraft:use_item_on"]

_HANDS = frozenset({"MAIN_HAND", "OFF_HAND"})
_FACES = frozenset({"DOWN", "UP", "NORTH", "SOUTH", "WEST", "EAST"})


@dataclass(frozen=True)
class UseItemOnAction:
    """Structured form of a ``ServerboundUseItemOnPacket``.

    ``cursor`` is **block-relative** here (each component in ``[0, 1]``);
    the wire/recorded shape uses world-absolute cursor coords and the
    codec converts at the boundary. The relative form is what the neural
    policy will emit — it's the natural representation given the action
    targets a specific block face.
    """

    packet_type: str
    hand: str
    block_pos: tuple[int, int, int]
    face: str
    cursor: tuple[float, float, float]  # block-relative; each component in [0, 1]
    inside: bool
    world_border_hit: bool
    sequence: int

    # Tag fields the neural head should NOT predict. Sequence numbers are
    # generated at packet-construction; the codec carries them for round-trip
    # parity but training/inference treats them as plumbing.
    _is_plumbing: ClassVar[tuple[str, ...]] = ("sequence",)

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(f"UseItemOnAction: invalid packet_type {self.packet_type!r}")
        if self.hand not in _HANDS:
            raise ValueError(f"UseItemOnAction: invalid hand {self.hand!r}")
        if self.face not in _FACES:
            raise ValueError(f"UseItemOnAction: invalid face {self.face!r}")
        if len(self.block_pos) != 3 or any(not isinstance(v, int) for v in self.block_pos):
            raise ValueError(f"UseItemOnAction: block_pos must be 3 ints, got {self.block_pos!r}")
        if len(self.cursor) != 3:
            raise ValueError(f"UseItemOnAction: cursor must be 3 floats, got {self.cursor!r}")


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    bp_raw = fields["block_pos"]
    bp = (int(bp_raw[0]), int(bp_raw[1]), int(bp_raw[2]))
    cursor_raw = fields["cursor"]
    # Convert the absolute world-space cursor to a block-relative offset.
    # Single subtract per axis — FP-stable when cursor is within the block
    # (the only legitimate range; client validates this before sending).
    cursor_rel = (
        float(cursor_raw[0]) - bp[0],
        float(cursor_raw[1]) - bp[1],
        float(cursor_raw[2]) - bp[2],
    )
    return UseItemOnAction(
        packet_type=packet_type,
        hand=str(fields["hand"]),
        block_pos=bp,
        face=str(fields["face"]),
        cursor=cursor_rel,
        inside=bool(fields["inside"]),
        world_border_hit=bool(fields["world_border_hit"]),
        sequence=int(fields["sequence"]),
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, UseItemOnAction):
        raise TypeError(
            f"use_item_on._decode: expected UseItemOnAction, got {type(action).__name__}"
        )
    bp = action.block_pos
    cx, cy, cz = action.cursor
    return {
        "hand": action.hand,
        # JSONL parses the recorder's array as a list; match that shape so
        # round-trip equality with the recorder output holds without an
        # ad-hoc list/tuple normalization in the test.
        "block_pos": [bp[0], bp[1], bp[2]],
        "face": action.face,
        "cursor": [bp[0] + cx, bp[1] + cy, bp[2] + cz],
        "inside": action.inside,
        "world_border_hit": action.world_border_hit,
        "sequence": action.sequence,
    }


register(PACKET_TYPES, _encode, _decode)

__all__ = ["UseItemOnAction", "PACKET_TYPES"]
