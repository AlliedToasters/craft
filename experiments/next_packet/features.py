"""Obs dict → flat float feature vector (neural_interface.md §8b, rungs R0–R2).

Converts the obs snapshot captured alongside each packet into a flat
list of floats. The function is obs-shape-agnostic: it reads whatever
keys are present and returns a named list so callers can inspect which
features were active. The model's input width is ``len(features(obs))``.

Rung R0 (§2a minimal, always present):
  x, y, z               3 floats (absolute world coords — used as-is here;
                         the delta encoding lives in the Action, not the obs)
  sin(yaw), cos(yaw)    2 floats (wrap-safe angle encoding)
  sin(pitch), cos(pitch) 2 floats
  on_ground             1 float (0/1)
  dim_id                1 float (ordinal over registered dims; unknown → -1)

Rung R1 (§8a additions, present only in LLM-driven recordings):
  g_t_present           1 float (0/1 flag — is g_t non-null?)
  ticks_since_g_t_issued 1 float (0 if absent)
  Note: ``g_t`` itself is a string and needs an embedding layer, NOT a
  float. The rung R1 float contribution is just the two temporal scalars;
  the goal string goes to a separate encoder outside this function.

Rung R2 (stats + inventory — TBD; scaffolded as zeroes for now):
  health, hunger, saturation, air  4 floats (zeroed until infra lands)

Total width:
  R0: 9
  R0+R1: 12
  R0+R1+R2: 16

Returns a ``FeatureVec`` named tuple so downstream code can inspect the
field list without parsing positional indices.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple

# Canonical dimension ordering. Unknown dims map to -1; new dims can be
# appended (the model's dim-feature won't generalise to them without
# retraining, but at least the feature doesn't crash).
_DIM_ORDER: dict[str, float] = {
    "minecraft:overworld": 0.0,
    "minecraft:the_nether": 1.0,
    "minecraft:the_end": 2.0,
}

# Names of all features in order. Index i <-> FEATURE_NAMES[i].
FEATURE_NAMES: list[str] = [
    # R0
    "x", "y", "z",
    "sin_yaw", "cos_yaw",
    "sin_pitch", "cos_pitch",
    "on_ground",
    "dim_id",
    # R1
    "g_t_present",
    "ticks_since_g_t_issued",
    # R1 temporal
    "delta_tick",           # Δticks_since_last (§3d); 0 when absent
    # R2 (zeroed until infra)
    "health",
    "hunger",
    "saturation",
    "air",
]


class FeatureVec(NamedTuple):
    values: list[float]
    names: list[str]   # same length as values; FEATURE_NAMES subset

    def __len__(self) -> int:
        return len(self.values)


def obs_to_features(obs: Mapping[str, Any]) -> FeatureVec:
    """Convert an obs dict to a flat float feature vector.

    Missing keys are handled gracefully: R1/R2 fields that aren't present
    are filled with sentinel values (0 / -1) rather than raising. The
    ``names`` field of the returned ``FeatureVec`` always matches
    FEATURE_NAMES regardless of which rungs contributed.
    """
    yaw_rad = math.radians(float(obs.get("yaw", 0.0)))
    pitch_rad = math.radians(float(obs.get("pitch", 0.0)))

    g_t_raw = obs.get("g_t")
    g_t_present = 0.0 if g_t_raw is None else 1.0
    ticks_since = float(obs.get("ticks_since_g_t_issued") or 0)
    delta_tick = float(obs.get("delta_tick") or 0)

    values: list[float] = [
        # R0
        float(obs.get("x", 0.0)),
        float(obs.get("y", 0.0)),
        float(obs.get("z", 0.0)),
        math.sin(yaw_rad),
        math.cos(yaw_rad),
        math.sin(pitch_rad),
        math.cos(pitch_rad),
        1.0 if obs.get("on_ground") else 0.0,
        _DIM_ORDER.get(str(obs.get("dim", "")), -1.0),
        # R1
        g_t_present,
        ticks_since,
        delta_tick,
        # R2 (zeroed)
        float(obs.get("health", 0.0)),
        float(obs.get("hunger", 0.0)),
        float(obs.get("saturation", 0.0)),
        float(obs.get("air", 0.0)),
    ]
    assert len(values) == len(FEATURE_NAMES), "FEATURE_NAMES out of sync"
    return FeatureVec(values=values, names=list(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Action tensorisation helpers (no torch dependency — returns plain lists)
# ---------------------------------------------------------------------------

# Canonical packet_type ordering. Index = class label for the discriminator.
PACKET_TYPES: list[str] = [
    "minecraft:interact",
    "minecraft:move_player_pos",
    "minecraft:move_player_pos_rot",
    "minecraft:move_player_rot",
    "minecraft:move_player_status_only",
    "minecraft:player_action",
    "minecraft:player_command",
    "minecraft:player_input",
    "minecraft:swing",
    "minecraft:use_item",
    "minecraft:use_item_on",
]
PACKET_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(PACKET_TYPES)}
NUM_PACKET_TYPES = len(PACKET_TYPES)


def packet_type_label(packet_type: str) -> int:
    """Return the integer discriminator label for ``packet_type``.

    Raises ``KeyError`` for unregistered types — the training loop should
    only see types that passed through ``encode``, so this is a bug if it
    fires.
    """
    return PACKET_TYPE_INDEX[packet_type]
