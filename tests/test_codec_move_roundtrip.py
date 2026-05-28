"""Static round-trip test for the move-family codec (ml.MD §4a, test-ladder
step 1).

Two kinds of test:

  * **Inline fixtures** — small hand-built (fields, obs) pairs that cover the
    four wire types plus the zero-delta edge case. Self-contained, fast,
    deterministic, exercise the contract regardless of recording availability.

  * **Corpus replay** — point at a real JSONL produced by the homunculus
    recorder (``CRAFT_CODEC_CORPUS`` env var). Skips if not set. This is the
    real test once the substrate is wired; the inline fixtures keep CI honest.

The round-trip contract: ``decode(encode(fields, obs), obs) == fields`` modulo
floating-point drift on the position channel (delta-encode introduces one
subtract+add per axis).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from craft import codec
from craft.codec.move import MoveAction, PACKET_TYPES, fields_close


# Realistic-ish obs frame; round-trip is invariant in obs so the values just
# need to be representative.
_OBS = {
    "tick": 12345,
    "captured_at_ms": 1_700_000_000_000,
    "x": -102_611.3625,
    "y": 64.0,
    "z": 16_129.4725,
    "yaw": 272.55,
    "pitch": 9.9,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


@pytest.mark.parametrize(
    "packet_type, fields",
    [
        (
            "minecraft:move_player_pos_rot",
            {
                "has_pos": True,
                "has_rot": True,
                "x": -102_611.3628,
                "y": 64.0,
                "z": 16_129.4730,
                "yaw": 273.10,
                "pitch": 9.85,
                "on_ground": True,
                "horizontal_collision": False,
            },
        ),
        (
            "minecraft:move_player_pos",
            {
                "has_pos": True,
                "has_rot": False,
                "x": -102_611.0,
                "y": 64.0,
                "z": 16_130.0,
                "on_ground": False,
                "horizontal_collision": False,
            },
        ),
        (
            "minecraft:move_player_rot",
            {
                "has_pos": False,
                "has_rot": True,
                "yaw": 197.4,
                "pitch": 7.5,
                "on_ground": True,
                "horizontal_collision": False,
            },
        ),
        (
            "minecraft:move_player_status_only",
            {
                "has_pos": False,
                "has_rot": False,
                "on_ground": True,
                "horizontal_collision": False,
            },
        ),
        (
            # Zero-delta edge case: client sends a pos packet with the same
            # coords as the obs frame (player hasn't actually moved). The
            # delta encode is exactly (0, 0, 0) — important to verify that
            # case round-trips without spurious drift.
            "minecraft:move_player_pos",
            {
                "has_pos": True,
                "has_rot": False,
                "x": _OBS["x"],
                "y": _OBS["y"],
                "z": _OBS["z"],
                "on_ground": True,
                "horizontal_collision": False,
            },
        ),
    ],
)
def test_inline_roundtrip(packet_type: str, fields: dict) -> None:
    action = codec.encode(packet_type, fields, _OBS)
    assert isinstance(action, MoveAction)
    assert action.packet_type == packet_type
    decoded = codec.decode(action, _OBS)
    assert fields_close(decoded, fields), f"\nexpected: {fields}\ngot:      {decoded}"


def test_all_move_types_registered() -> None:
    """Sanity that the registry side-effect ran for every move wire type —
    if the import didn't land in the right order, the dispatch would miss."""
    for pt in PACKET_TYPES:
        assert codec.is_registered(pt), f"{pt!r} missing from registry"


def test_unknown_packet_raises() -> None:
    with pytest.raises(KeyError):
        codec.encode("minecraft:client_tick_end", {}, _OBS)


# ---------------------------------------------------------------------------
# Optional corpus replay against a real homunculus recording.
# ---------------------------------------------------------------------------

_MOVE_IDS = set(PACKET_TYPES)


def test_corpus_move_roundtrip() -> None:
    env = os.environ.get("CRAFT_CODEC_CORPUS") or ""
    if not env:
        pytest.skip("set CRAFT_CODEC_CORPUS to a recording JSONL to enable")
    path = Path(env)
    if not path.exists():
        pytest.skip(f"CRAFT_CODEC_CORPUS {path} does not exist")
    total = 0
    failures: list[tuple[str, dict, dict]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] not in _MOVE_IDS:
                continue
            fields = entry["fields"]
            if fields.get("_unimplemented"):
                continue
            total += 1
            obs = entry["obs"]
            action = codec.encode(entry["id"], fields, obs)
            decoded = codec.decode(action, obs)
            if not fields_close(decoded, fields, atol=1e-6):
                failures.append((entry["id"], fields, decoded))
    assert total > 0, f"no move packets in corpus {path}"
    assert not failures, (
        f"{len(failures)} / {total} move round-trips failed; "
        f"first: {failures[0]}"
    )
