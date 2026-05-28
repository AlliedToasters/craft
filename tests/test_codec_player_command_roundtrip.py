"""Static round-trip for ``player_command`` (ml.MD §4a, step 1). 9-way enum
plus entity_id + data int."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.player_command import PACKET_TYPES, PlayerCommandAction


_OBS = {
    "tick": 1,
    "captured_at_ms": 0,
    "x": 0.0, "y": 64.0, "z": 0.0,
    "yaw": 0.0, "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


_ALL_ACTIONS = (
    "PRESS_SHIFT_KEY", "RELEASE_SHIFT_KEY", "STOP_SLEEPING",
    "START_SPRINTING", "STOP_SPRINTING",
    "START_RIDING_JUMP", "STOP_RIDING_JUMP",
    "OPEN_INVENTORY", "START_FALL_FLYING",
)


@pytest.mark.parametrize("action", _ALL_ACTIONS)
def test_inline_roundtrip(action: str) -> None:
    fields = {"entity_id": 1_544_476, "action": action, "data": 0}
    a = codec.encode("minecraft:player_command", fields, _OBS)
    assert isinstance(a, PlayerCommandAction)
    assert a.semantic_fields == frozenset({"entity_id", "action", "data"})
    assert codec.fields_close(codec.decode(a, _OBS), fields)


def test_nonzero_data_payload() -> None:
    """``data`` carries action-specific payload (horse jump charge etc).
    Verify a non-zero int round-trips exactly."""
    fields = {
        "entity_id": 42,
        "action": "START_RIDING_JUMP",
        "data": 70,  # jump charge
    }
    a = codec.encode("minecraft:player_command", fields, _OBS)
    assert codec.fields_close(codec.decode(a, _OBS), fields)


def test_invalid_action_rejected() -> None:
    with pytest.raises(ValueError, match="invalid action"):
        PlayerCommandAction(
            packet_type="minecraft:player_command",
            entity_id=1, action="DANCE", data=0,
        )


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt)


def test_corpus_player_command_roundtrip() -> None:
    env = os.environ.get("CRAFT_CODEC_CORPUS") or ""
    if not env:
        pytest.skip("set CRAFT_CODEC_CORPUS to a recording JSONL")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    total = 0
    failures: list[tuple[dict, dict]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] != "minecraft:player_command":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            decoded = codec.decode(codec.encode(entry["id"], fields, entry["obs"]), entry["obs"])
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no player_command packets in corpus {path}")
    assert not failures, f"{len(failures)}/{total} player_command round-trips failed; first: {failures[0]}"
