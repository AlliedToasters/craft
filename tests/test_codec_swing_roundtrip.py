"""Static round-trip for ``swing`` (ml.MD §4a, step 1). Trivial 1-field
codec; tests are correspondingly small."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.swing import PACKET_TYPES, SwingAction


_OBS = {
    "tick": 1,
    "captured_at_ms": 0,
    "x": 0.0, "y": 64.0, "z": 0.0,
    "yaw": 0.0, "pitch": 0.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


@pytest.mark.parametrize("hand", ["MAIN_HAND", "OFF_HAND"])
def test_inline_roundtrip(hand: str) -> None:
    fields = {"hand": hand}
    action = codec.encode("minecraft:swing", fields, _OBS)
    assert isinstance(action, SwingAction)
    assert action.semantic_fields == frozenset({"hand"})
    assert codec.fields_close(codec.decode(action, _OBS), fields)


def test_invalid_hand_rejected() -> None:
    with pytest.raises(ValueError, match="invalid hand"):
        SwingAction(packet_type="minecraft:swing", hand="THIRD_HAND")


def test_registered() -> None:
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt)


def test_corpus_swing_roundtrip() -> None:
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
            if entry["id"] != "minecraft:swing":
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            decoded = codec.decode(codec.encode(entry["id"], fields, entry["obs"]), entry["obs"])
            if not codec.fields_close(decoded, fields, atol=1e-6):
                failures.append((fields, decoded))
    if total == 0:
        pytest.skip(f"no swing packets in corpus {path}")
    assert not failures, f"{len(failures)}/{total} swing round-trips failed; first: {failures[0]}"
