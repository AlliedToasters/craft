"""Move-family codec: covers all 4 wire types in
``ServerboundMovePlayerPacket`` (ml.MD §4a).

Wire types and what they carry:

  * ``move_player_pos``         — position only, no rotation
  * ``move_player_pos_rot``     — position + rotation
  * ``move_player_rot``         — rotation only, no position
  * ``move_player_status_only`` — neither position nor rotation (just the
                                  on-ground flag and horizontal-collision)

Encoding choices:

  * **Position is encoded as a delta** ``(dx, dy, dz)`` against
    ``obs.{x, y, z}``. This is the literal "pointer into observation": the
    structured action says *"move +0.21 blocks in x relative to where I am"*,
    not *"move to absolute coord 100.42"*. Same physical action across
    different starting points → same structured representation.

  * **Rotation is left absolute** ``(yaw, pitch)``. A delta encoding for
    angles requires wrap-around handling (a +359° turn equals -1°); for a
    scaffolding pass the absolute form is simpler, lossless, and trivially
    invertible. The neural model will normalize via sin/cos features.

  * **on_ground / horizontal_collision pass through** as booleans.

  * ``packet_type`` is **carried explicitly** on the action. It's the
    discriminator that lets the decoder emit the correct on-wire shape — we
    could derive it from ``(pos is not None, rot is not None)`` but that
    couples the wire-type choice to "are the values zero" rather than "did
    the client decide to send this on the wire," which is a different fact
    and the wrong source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from craft.codec.base import Action, fields_close, register


PACKET_TYPES = [
    "minecraft:move_player_pos",
    "minecraft:move_player_pos_rot",
    "minecraft:move_player_rot",
    "minecraft:move_player_status_only",
]


@dataclass(frozen=True)
class MoveAction:
    """Structured form of any ``ServerboundMovePlayerPacket`` wire type.

    Invariants enforced at construction time:

      * ``packet_type`` is one of the four move wire ids.
      * ``pos is not None`` iff the wire type carries position
        (``move_player_pos`` / ``move_player_pos_rot``).
      * ``rot is not None`` iff the wire type carries rotation
        (``move_player_pos_rot`` / ``move_player_rot``).
    """

    packet_type: str
    pos: tuple[float, float, float] | None  # (dx, dy, dz) against obs.pos
    rot: tuple[float, float] | None  # (yaw_abs, pitch_abs)
    on_ground: bool
    horizontal_collision: bool

    def __post_init__(self) -> None:
        if self.packet_type not in PACKET_TYPES:
            raise ValueError(f"MoveAction: invalid packet_type {self.packet_type!r}")
        expects_pos = self.packet_type in (
            "minecraft:move_player_pos",
            "minecraft:move_player_pos_rot",
        )
        expects_rot = self.packet_type in (
            "minecraft:move_player_pos_rot",
            "minecraft:move_player_rot",
        )
        if expects_pos != (self.pos is not None):
            raise ValueError(
                f"MoveAction: {self.packet_type!r} pos presence mismatch "
                f"(pos={self.pos!r})"
            )
        if expects_rot != (self.rot is not None):
            raise ValueError(
                f"MoveAction: {self.packet_type!r} rot presence mismatch "
                f"(rot={self.rot!r})"
            )


def _encode(packet_type: str, fields: Mapping[str, Any], obs: Mapping[str, Any]) -> Action:
    has_pos = bool(fields.get("has_pos", False))
    has_rot = bool(fields.get("has_rot", False))
    pos: tuple[float, float, float] | None = None
    rot: tuple[float, float] | None = None
    if has_pos:
        # Validate obs supplies the reference frame we need to subtract against.
        # Missing keys here are a codec contract bug, not a recoverable runtime case.
        ox, oy, oz = float(obs["x"]), float(obs["y"]), float(obs["z"])
        pos = (
            float(fields["x"]) - ox,
            float(fields["y"]) - oy,
            float(fields["z"]) - oz,
        )
    if has_rot:
        rot = (float(fields["yaw"]), float(fields["pitch"]))
    return MoveAction(
        packet_type=packet_type,
        pos=pos,
        rot=rot,
        on_ground=bool(fields.get("on_ground", False)),
        horizontal_collision=bool(fields.get("horizontal_collision", False)),
    )


def _decode(action: Action, obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(action, MoveAction):
        raise TypeError(f"move._decode: expected MoveAction, got {type(action).__name__}")
    has_pos = action.pos is not None
    has_rot = action.rot is not None
    out: dict[str, Any] = {"has_pos": has_pos, "has_rot": has_rot}
    if has_pos:
        assert action.pos is not None  # narrow for the type checker
        ox, oy, oz = float(obs["x"]), float(obs["y"]), float(obs["z"])
        dx, dy, dz = action.pos
        out["x"] = ox + dx
        out["y"] = oy + dy
        out["z"] = oz + dz
    if has_rot:
        assert action.rot is not None
        out["yaw"], out["pitch"] = action.rot
    out["on_ground"] = action.on_ground
    out["horizontal_collision"] = action.horizontal_collision
    return out


register(PACKET_TYPES, _encode, _decode)

# Re-export the generic comparator so callers that already imported it from
# ``craft.codec.move`` don't break.
__all__ = ["MoveAction", "PACKET_TYPES", "fields_close"]
