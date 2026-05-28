"""Static round-trip for ``player_input`` (ml.MD §4a, step 1). 7 booleans;
exercise the empty + full bitmask plus a representative motion bitmask."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.player_input import PACKET_TYPES, PlayerInputAction


_OBS = {
    "tick": 1,
    "captured_at_ms": 0,
    "x": 0.0, "y": 64.0, "z": 0.0,
    "yaw": 0.0, "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}

_INPUT_KEYS = ("forward", "backward", "left", "right", "jump", "shift", "sprint")


def _input_fields(**overrides: bool) -> dict:
    fields = {k: False for k in _INPUT_KEYS}
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    "fields",
    [
        _input_fields(),                                          # all off
        _input_fields(forward=True),                              # walk forward
        _input_fields(forward=True, sprint=True),                 # sprint
        _input_fields(forward=True, jump=True),                   # jump
        _input_fields(left=True, sprint=True, jump=True),         # sprint-jump turn
        _input_fields(**{k: True for k in _INPUT_KEYS}),          # all on (anti-cheat would reject but codec must round-trip)
    ],
)
def test_inline_roundtrip(fields: dict) -> None:
    action = codec.encode("minecraft:player_input", fields, _OBS)
    assert isinstance(action, PlayerInputAction)
    assert action.semantic_fields == frozenset(_INPUT_KEYS)
    assert codec.fields_close(codec.decode(action, _OBS), fields)


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt)


def test_corpus_player_input_roundtrip() -> None:
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
            if entry["id"] != "minecraft:player_input":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            decoded = codec.decode(codec.encode(entry["id"], fields, entry["obs"]), entry["obs"])
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no player_input packets in corpus {path}")
    assert not failures, f"{len(failures)}/{total} player_input round-trips failed; first: {failures[0]}"
