"""Structured codec between serverbound packets and neural actions (ml.MD §4a).

Maps allowlisted packets to a tagged-union structured action whose parameters
are pointers/deltas into the observation. Round-trip invariant:

    decode(encode(packet_fields, obs), obs) ≈ packet_fields  (modulo plumbing)

Encode/decode are observation-conditioned — that's the formal statement of
"actions are pointers into observations." The codec is independently testable
with NO ML; correctness here is the contract every learning experiment binds
against.

Current scope: ``move_player_pos / pos_rot / rot / status_only`` — the four
wire types in the ``ServerboundMovePlayerPacket`` family, ~70% of allowlisted
volume per Phase 0 stats. The remaining 7 types (player_input, player_command,
use_item, use_item_on, player_action, interact, swing) will register against
the same dispatch as their codecs land.
"""

from craft.codec.base import (
    Action,
    decode,
    encode,
    fields_close,
    is_registered,
    registered_types,
)
from craft.codec.interact import InteractAction
from craft.codec.move import MoveAction
from craft.codec.player_action import PlayerActionAction
from craft.codec.player_command import PlayerCommandAction
from craft.codec.player_input import PlayerInputAction
from craft.codec.swing import SwingAction
from craft.codec.use_item import UseItemAction
from craft.codec.use_item_on import UseItemOnAction

__all__ = [
    "Action",
    "InteractAction",
    "MoveAction",
    "PlayerActionAction",
    "PlayerCommandAction",
    "PlayerInputAction",
    "SwingAction",
    "UseItemAction",
    "UseItemOnAction",
    "decode",
    "encode",
    "fields_close",
    "is_registered",
    "registered_types",
]
