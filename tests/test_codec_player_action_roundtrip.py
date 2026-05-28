"""Static round-trip for the ``player_action`` codec (ml.MD §4a, step 1).

Inline coverage:

  * The 3 spatial dig-lifecycle actions with meaningful ``block_pos``/``face``.
  * The 4 non-spatial actions with convention-zero filler.
  * ``is_spatial`` property partitions actions correctly.
  * Enum validation on bad action / bad face.

Corpus replay tolerates ``{"_unimplemented": true}`` entries from older
recorder builds so a single JSONL can drive every codec test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.player_action import PACKET_TYPES, PlayerActionAction


_OBS = {
    "tick": 12345,
    "captured_at_ms": 1_700_000_000_000,
    "x": -102_600.0,
    "y": 70.0,
    "z": 16_126.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


def _fields(
    *,
    action: str = "STOP_DESTROY_BLOCK",
    block_pos: tuple[int, int, int] = (-102_602, 68, 16_126),
    face: str = "SOUTH",
    sequence: int = 2,
) -> dict:
    return {
        "action": action,
        "block_pos": list(block_pos),
        "face": face,
        "sequence": sequence,
    }


# All 7 wire actions exercised — the 3 spatial with real block coords +
# the 4 non-spatial with convention-zero filler the client actually sends.
@pytest.mark.parametrize(
    "fields",
    [
        # Real captured packet
        _fields(action="STOP_DESTROY_BLOCK"),
        _fields(action="START_DESTROY_BLOCK", face="UP", sequence=1),
        _fields(action="ABORT_DESTROY_BLOCK", face="EAST", sequence=4),
        # Non-spatial: client conventionally sends (0,0,0) + DOWN + 0
        _fields(action="DROP_ALL_ITEMS", block_pos=(0, 0, 0), face="DOWN", sequence=0),
        _fields(action="DROP_ITEM", block_pos=(0, 0, 0), face="DOWN", sequence=0),
        _fields(action="RELEASE_USE_ITEM", block_pos=(0, 0, 0), face="DOWN", sequence=0),
        _fields(action="SWAP_ITEM_WITH_OFFHAND", block_pos=(0, 0, 0), face="DOWN", sequence=0),
    ],
)
def test_inline_roundtrip(fields: dict) -> None:
    action = codec.encode("minecraft:player_action", fields, _OBS)
    assert isinstance(action, PlayerActionAction)
    decoded = codec.decode(action, _OBS)
    assert codec.fields_close(decoded, fields), f"\nexpected: {fields}\ngot:      {decoded}"


def test_is_spatial_partition() -> None:
    """The neural head will mask spatial parameters based on this property —
    make sure the partition is exactly the 3 dig-lifecycle actions."""
    spatial = {"START_DESTROY_BLOCK", "ABORT_DESTROY_BLOCK", "STOP_DESTROY_BLOCK"}
    non_spatial = {
        "DROP_ALL_ITEMS", "DROP_ITEM", "RELEASE_USE_ITEM", "SWAP_ITEM_WITH_OFFHAND",
    }
    for a in spatial:
        action = PlayerActionAction(
            packet_type="minecraft:player_action",
            action=a,
            block_pos=(1, 2, 3),
            face="UP",
            sequence=0,
        )
        assert action.is_spatial, a
    for a in non_spatial:
        action = PlayerActionAction(
            packet_type="minecraft:player_action",
            action=a,
            block_pos=(0, 0, 0),
            face="DOWN",
            sequence=0,
        )
        assert not action.is_spatial, a


def test_invalid_action_rejected() -> None:
    with pytest.raises(ValueError, match="invalid action"):
        PlayerActionAction(
            packet_type="minecraft:player_action",
            action="DANCE",
            block_pos=(0, 0, 0),
            face="DOWN",
            sequence=0,
        )


def test_invalid_face_rejected() -> None:
    with pytest.raises(ValueError, match="invalid face"):
        PlayerActionAction(
            packet_type="minecraft:player_action",
            action="START_DESTROY_BLOCK",
            block_pos=(0, 0, 0),
            face="SIDEWAYS",
            sequence=0,
        )


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt), f"{pt!r} missing from registry"


# ---------------------------------------------------------------------------
# Corpus replay
# ---------------------------------------------------------------------------


def test_corpus_player_action_roundtrip() -> None:
    env = os.environ.get("CRAFT_CODEC_CORPUS") or ""
    if not env:
        pytest.skip("set CRAFT_CODEC_CORPUS to a recording JSONL to enable")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"CRAFT_CODEC_CORPUS {path} does not exist")
    total = 0
    failures: list[tuple[dict, dict]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] != "minecraft:player_action":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            obs = entry["obs"]
            action = codec.encode(entry["id"], fields, obs)
            decoded = codec.decode(action, obs)
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no player_action packets in corpus {path}")
    assert not failures, (
        f"{len(failures)} / {total} player_action round-trips failed; "
        f"first: {failures[0]}"
    )
