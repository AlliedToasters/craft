"""§16.1 — obs-relative rotation reparameterization (the dumb conditional baseline).

The §16.0 characterization (results/sprint16/RESULTS_cond_residual.md) found the
entire conditional-coding prize is in ROTATION: yaw/pitch carry ~4 bits coded
ABSOLUTELY but only 0.2-0.5 bits coded RELATIVE to obs.{yaw,pitch}. The existing
codec already carries POSITION as a delta vs obs (craft/codec/move.py) — i.e. it
already codes position obs-relative. This module does the SAME for rotation, so
the whole MoveAction becomes "every field is a zero_preserving delta vs obs":

  * pos   : delta vs obs.{x,y,z}        (already so; quantized zero_preserving)
  * yaw   : wrap180(yaw - obs.yaw)      (NEW: residual, quantized zero_preserving)
  * pitch : pitch - obs.pitch           (NEW: residual, quantized zero_preserving)

Two properties this buys, both load-bearing:
  1. RATE — the residual is sharply peaked at 0 (per-tick turn median 0.6°), so an
     entropy coder spends ~0.5 bits where absolute coding spends ~4. This is the
     honest baseline-to-beat for the learned AE (NOT the 18.45-bit allocation).
  2. ZERO-MEAN-AT-REST for rotation — zero_preserving makes a still-looking player
     (residual 0) reconstruct to obs.{yaw,pitch} EXACTLY. Absolute coding injects
     the same ±zero_bias DC offset that rubberbanded position in §15, but on the
     camera. This module inherits the §15 drift-fatal/dropout-benign fix for rot.

This is a DETERMINISTIC reparameterization — zero learning. It is the baseline the
§16.2 learned AE must beat; if it can't, the stream's compressibility was a free
reparam, not learnable structure (the pre-registered null).

`quantize_move_obsrel` mirrors `quantize.quantize_move` but takes obs and quantizes
the rotation RESIDUAL; it returns a MoveAction with ABSOLUTE rot reconstructed, so
the downstream `decode` path is unchanged (the reparam is internal to coding).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Any

from craft.codec.move import MoveAction
from experiments.codec_loop.quantize import (
    POS_RANGE, POS_MODES, quant_scalar_zero_preserving,
)

# Residual is the per-tick change; [-180,180] never clips a turn (yaw wraps; pitch
# is bounded ±90 so its residual is ±180 worst case). zero_preserving => 0 is a code.
ROT_RANGE = 180.0


def wrap180(d: float) -> float:
    """Signed shortest angular difference in (-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


def yaw_residual(yaw: float, obs_yaw: float) -> float:
    return wrap180(yaw - obs_yaw)


def yaw_from_residual(res: float, obs_yaw: float) -> float:
    return obs_yaw + res


def quantize_move_obsrel(action: MoveAction, obs: Mapping[str, Any], *,
                         pos_bits: int, yaw_bits: int, pitch_bits: int,
                         pos_range: float = POS_RANGE,
                         pos_mode: str = "zero_preserving",
                         rot_range: float = ROT_RANGE) -> MoveAction:
    """Lossy copy of ``action`` with pos AND rotation coded as zero_preserving
    deltas vs ``obs``. Rotation is reconstructed back to ABSOLUTE so ``decode``
    is unchanged. Booleans/packet_type pass through.

    obs must supply yaw/pitch when the action carries rotation (it always does on
    the live wire — obs is the pre-packet per-tick snapshot). pos uses the
    existing delta (action.pos is already obs-relative)."""
    pos_q = POS_MODES[pos_mode]
    pos = action.pos
    if pos is not None:
        pos = tuple(pos_q(c, -pos_range, pos_range, pos_bits)[0] for c in pos)  # type: ignore[assignment]
    rot = action.rot
    if rot is not None:
        yaw, pitch = rot
        oy = float(obs["yaw"])
        op = float(obs["pitch"])
        yres, _ = quant_scalar_zero_preserving(wrap180(yaw - oy), -rot_range, rot_range, yaw_bits)
        pres, _ = quant_scalar_zero_preserving(pitch - op, -rot_range, rot_range, pitch_bits)
        rot = (oy + yres, op + pres)
    return replace(action, pos=pos, rot=rot)
