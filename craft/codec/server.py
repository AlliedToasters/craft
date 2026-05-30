"""HTTP shim wrapping ``craft.codec.encode/decode`` for live use (ml.MD §4a,
test-ladder step 2).

The homunculus mod is JVM-side; the codec is Python. Step 2 needs the codec
exercised on every allowlisted outbound packet *while gameplay runs*, so the
mod ships fields + obs over the local network and reads back the round-tripped
fields. A standalone server keeps the codec a single source of truth (no Java
re-implementation drift) and lets one server back the entire 10-agent fleet —
the codec is pure-function, no per-agent state.

Endpoint:

  ``POST /codec/roundtrip``  body ``{"id": <packet_id>, "fields": {...}, "obs": {...}}``
  → ``{"ok": bool, "decoded": {...}|null, "error": str|null}``

  ``ok=true`` iff ``decode(encode(fields, obs), obs)`` returns a fields dict
  structurally equal to the input (``fields_close`` with the codec's default
  tolerance). On any exception, ``ok=false`` and ``error`` is the message —
  the caller increments its drift / transport-error counter accordingly.

Why this is just a thin shim:
  The codec round-trip itself is what's being validated. Anything more
  (caching, batching, schema validation) would either add latency or paper
  over the very failures we want to surface. The server is stateless; the
  Java side does all the bookkeeping.

Run with ``python -m craft.codec.server [--port 25600] [--host 127.0.0.1]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from craft.codec.base import decode, encode, fields_close, registered_types
from craft.codec.move import MoveAction
from experiments.codec_loop.quantize import POS_MODES, float_bits, quantize_move
from experiments.codec_loop.obsrel import quantize_move_obsrel
from experiments.codec_loop.blockpos import BLOCK_REACH, ABS_RANGE, quantize_block_pos

DEFAULT_PORT = 25600
DEFAULT_HOST = "127.0.0.1"
_MAX_BODY_BYTES = 64 * 1024

_log = logging.getLogger("craft.codec.server")

# --- Sprint A lossy mode -----------------------------------------------------
# The default server is a LOSSLESS identity shim (§14). Sprint A turns it into a
# loss-tolerance probe by quantizing the MOVEMENT family between encode and
# decode, so the decoded fields the harness substitutes on the wire carry the
# quantization error. Movement only (volume-dominant; brief: pick ONE target
# type). This is NOT a learned-codec baseline (anti-pattern #1) — it baselines
# nothing; the deliverable is the parity-vs-bits knee.
#
# Config is a module-global tuned at startup (--quant-bits) and live via
# POST /config, so a sweep retunes b without restarting (wire path stays byte-
# identical across levels — only the sidecar math changes). None => lossless.
_QUANT_LOCK = threading.Lock()
# pos_bits/yaw_bits/pitch_bits: None => that field is lossless.
_QUANT: dict[str, int | None] = {"pos_bits": None, "yaw_bits": None, "pitch_bits": None}
# pos_range: the symmetric ±span the pos delta is quantized over (None =>
# quantize.py default 8.0). Kept in its OWN global (not in _QUANT) so _QUANT stays
# int-typed: pos_range is a float span, not a bit budget. It is the CONTINUOUS
# resolution lever — at fixed bits, min step = 2*pos_range/(2**pos_bits-1), so
# sweeping it slides the parity knee in sub-bit increments — and it is orthogonal
# to the lossless gate: it only takes effect when pos_bits is also set.
_POS_RANGE: float | None = None
# pos_mode: which pos quantizer grid (quantize.POS_MODES). "zero_biased" (default)
# is the original mid-rise grid (0 not representable -> DC drift on stationary
# packets); "zero_preserving" puts a code on 0 so a still player reconstructs to
# exactly 0 (the b5-cliff fix). Own global, like _POS_RANGE; only takes effect
# when pos_bits is set. Default preserves every prior sweep byte-for-byte.
_POS_MODE: str = "zero_biased"

# obsrel: §16.1 baseline. When True, the MOVE family is quantized via
# `quantize_move_obsrel` — rotation is coded as a zero_preserving residual vs
# obs.{yaw,pitch} (the decoder's last-known rotation) instead of absolute, then
# reconstructed back to absolute so `decode` is unchanged. §16.0/16.1 found the
# whole conditional-coding prize is here: yaw/pitch carry ~4 bits absolute but
# ~0.4 obs-relative, and obs-relative holds heading at-rest exactly (no camera
# zero-bias). This live mode tests whether rotation DEADBAND is benign like pos
# deadband (§15). Only takes effect when quant is active (yaw/pitch bits set).
# Default False preserves every prior §15 sweep byte-for-byte.
_OBSREL: bool = False

# perturb: §17.0 aim-carrier DIAGNOSTIC (not a shippable codec). When set, the
# decoded TARGET field of an action packet is deliberately corrupted before it is
# reconstructed onto the wire — the positive-carrier control: if corrupting field
# X breaks the aim-dependent action, X carries aim. Spec (all optional):
#   {"block_pos_delta": [dx,dy,dz]}  -> offset block_pos for use_item_on and the
#                                        spatial (dig-lifecycle) player_action
#                                        actions. block_pos is plain ints, so the
#                                        Java reconstructor rebuilds cleanly (no
#                                        entity lookup) and the action targets the
#                                        WRONG block.
# entity_id is intentionally NOT perturbable here: the Java reconstructor resolves
# entity_id -> Entity (level.getEntity); a bogus id yields null and the substitute
# silently falls back to the ORIGINAL packet, so an entity_id perturbation cannot
# be cleanly attributed. The block channel is the clean carrier probe. Default off.
_PERTURB: dict[str, Any] | None = None
_PERTURB_SPATIAL_ACTIONS = frozenset({
    "START_DESTROY_BLOCK", "ABORT_DESTROY_BLOCK", "STOP_DESTROY_BLOCK",
})

# blockpos: §17.2.1 lossy DISCRETE-TARGET codec (a real codec, unlike perturb).
# When set, the block_pos field of a block-targeted action (use_item_on place /
# spatial player_action dig) is quantized between encode and decode so the
# substituted wire packet carries the quantization error — the discrete-channel
# analog of the §16 move quantizer. Spec (all optional bar bits):
#   {"bits": 4, "mode": "obsrel"|"absolute", "reach": 6.0, "abs_range": 8192.0}
# mode=obsrel codes block_pos - round(player_pos) over ±reach (the §16 reparam,
# ~4 bits lossless); mode=absolute quantizes the raw world coord over ±abs_range
# (~14 bits to resolve 1 block — the foil). Reconstructs to integer block_pos so
# the Java reconstructor rebuilds a valid packet. None => block_pos lossless.
_BLOCKPOS: dict[str, Any] | None = None

# --- lightweight load instrumentation (fleet strain gauge) -------------------
# Counts roundtrips and tracks concurrent in-flight requests so a sweep can read
# /healthz and SEE whether the single shared server is the operational ceiling
# on fleet size n. Cheap dict ops under the GIL — negligible vs the ~0.3ms
# roundtrip; max_inflight has a benign read-modify-write race (diagnostic only).
_STATS_LOCK = threading.Lock()
_STATS: dict[str, int] = {"roundtrips": 0, "inflight": 0, "max_inflight": 0}


def _stats_enter() -> None:
    with _STATS_LOCK:
        _STATS["inflight"] += 1
        _STATS["roundtrips"] += 1
        if _STATS["inflight"] > _STATS["max_inflight"]:
            _STATS["max_inflight"] = _STATS["inflight"]


def _stats_exit() -> None:
    with _STATS_LOCK:
        _STATS["inflight"] -= 1


def _stats_snapshot() -> dict[str, int]:
    with _STATS_LOCK:
        return dict(_STATS)


def _quant_snapshot() -> dict[str, int | None]:
    with _QUANT_LOCK:
        return dict(_QUANT)


def _full_snapshot() -> dict[str, Any]:
    """quant bits + pos_range span + pos_mode + obsrel, for /healthz and /config."""
    with _QUANT_LOCK:
        snap: dict[str, Any] = dict(_QUANT)
        snap["pos_range"] = _POS_RANGE
        snap["pos_mode"] = _POS_MODE
        snap["obsrel"] = _OBSREL
        snap["perturb"] = _PERTURB
        snap["blockpos"] = _BLOCKPOS
        return snap


def _set_quant(pos_bits: int | None, yaw_bits: int | None, pitch_bits: int | None) -> None:
    with _QUANT_LOCK:
        _QUANT["pos_bits"] = pos_bits
        _QUANT["yaw_bits"] = yaw_bits
        _QUANT["pitch_bits"] = pitch_bits


def _set_pos_range(pos_range: float | None) -> None:
    global _POS_RANGE
    with _QUANT_LOCK:
        _POS_RANGE = pos_range


def _set_pos_mode(pos_mode: str) -> None:
    global _POS_MODE
    with _QUANT_LOCK:
        _POS_MODE = pos_mode


def _set_obsrel(obsrel: bool) -> None:
    global _OBSREL
    with _QUANT_LOCK:
        _OBSREL = obsrel


def _set_perturb(perturb: dict[str, Any] | None) -> None:
    global _PERTURB
    with _QUANT_LOCK:
        _PERTURB = perturb


def _set_blockpos(blockpos: dict[str, Any] | None) -> None:
    global _BLOCKPOS
    with _QUANT_LOCK:
        _BLOCKPOS = blockpos


def _apply_perturb(packet_id: str, decoded: dict[str, Any],
                   perturb: dict[str, Any]) -> bool:
    """§17.0 carrier diagnostic: corrupt the TARGET field of an action packet
    in-place on the decoded dict. Returns True if anything was perturbed (so the
    caller forces ok=True and homunculus substitutes the corrupted packet)."""
    delta = perturb.get("block_pos_delta")
    if delta and packet_id in ("minecraft:use_item_on", "minecraft:player_action"):
        # player_action carries block_pos for ALL actions but it's only meaningful
        # (and on the wire from a real dig) for the spatial dig-lifecycle ones.
        if (packet_id == "minecraft:player_action"
                and decoded.get("action") not in _PERTURB_SPATIAL_ACTIONS):
            return False
        bp = decoded.get("block_pos")
        if bp and len(bp) == 3:
            decoded["block_pos"] = [bp[0] + int(delta[0]),
                                    bp[1] + int(delta[1]),
                                    bp[2] + int(delta[2])]
            return True
    return False


def _quant_active(q: dict[str, int | None]) -> bool:
    return any(v is not None for v in q.values())


def roundtrip(packet_id: str, fields: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    """The whole point: ``decode(encode(...), ...)`` and report whether the
    output matches the input. Exposed at module scope so unit tests can hit
    the same surface as the HTTP endpoint without standing up a server.

    When lossy mode is active (Sprint A), MoveAction packets are quantized
    between encode and decode so the returned ``decoded`` fields carry the
    quantization error onto the wire. ``ok`` then reports fidelity vs the
    original input (it will be False once quantization perturbs a field — that
    is expected and not a fault; the harness judges behavioral parity, not
    fields_close)."""
    try:
        action = encode(packet_id, fields, obs)
    except KeyError as e:
        return {"ok": False, "decoded": None, "error": f"encode KeyError: {e}"}
    except Exception as e:
        return {"ok": False, "decoded": None, "error": f"encode {type(e).__name__}: {e}"}

    quantized_bits: int | None = None
    q = _quant_snapshot()
    if _quant_active(q) and isinstance(action, MoveAction):
        # Missing per-axis bits fall back to the smallest specified width so a
        # single quant_bits=N tunes all three; default unfilled to an
        # effectively-lossless width. pos_range is a separate float global.
        bit_vals = [v for v in (q["pos_bits"], q["yaw_bits"], q["pitch_bits"]) if v is not None]
        fill = min(bit_vals) if bit_vals else 32
        pb = q["pos_bits"] if q["pos_bits"] is not None else fill
        yb = q["yaw_bits"] if q["yaw_bits"] is not None else fill
        ptb = q["pitch_bits"] if q["pitch_bits"] is not None else fill
        with _QUANT_LOCK:
            pr = _POS_RANGE
            pm = _POS_MODE
            obsrel = _OBSREL
        try:
            if obsrel:
                # §16.1 baseline: rotation coded obs-relative (residual vs
                # obs.{yaw,pitch}), reconstructed to absolute. obs is the
                # pre-packet per-tick snapshot the decoder holds live.
                kw = {"pos_bits": pb, "yaw_bits": yb, "pitch_bits": ptb, "pos_mode": pm}
                if pr is not None:
                    kw["pos_range"] = pr
                action = quantize_move_obsrel(action, obs, **kw)  # type: ignore[arg-type]
            elif pr is not None:
                action = quantize_move(action, pos_bits=pb, yaw_bits=yb,
                                       pitch_bits=ptb, pos_range=pr, pos_mode=pm)
            else:
                action = quantize_move(action, pos_bits=pb, yaw_bits=yb,
                                       pitch_bits=ptb, pos_mode=pm)
            quantized_bits = float_bits(action, pos_bits=pb, yaw_bits=yb, pitch_bits=ptb)
        except Exception as e:
            return {"ok": False, "decoded": None,
                    "error": f"quantize {type(e).__name__}: {e}"}

    # §17.2.1 discrete-target codec: quantize block_pos on block-targeted actions
    # (place / spatial dig). Mutually exclusive with the MoveAction quant path
    # above — different packet families. Like the move quant, the substituted
    # decoded fields carry the quantization error onto the wire.
    blockpos_lossy = False
    with _QUANT_LOCK:
        bp_cfg = _BLOCKPOS
    if bp_cfg is not None:
        try:
            new_action = quantize_block_pos(
                action, obs,
                bits=int(bp_cfg["bits"]),
                mode=bp_cfg.get("mode", "obsrel"),
                reach=float(bp_cfg.get("reach", BLOCK_REACH)),
                abs_range=float(bp_cfg.get("abs_range", ABS_RANGE)),
            )
        except Exception as e:
            return {"ok": False, "decoded": None,
                    "error": f"blockpos {type(e).__name__}: {e}"}
        # quantize_block_pos returns the SAME object for non-block actions; only
        # flag lossy when it actually touched a block-targeted packet.
        blockpos_lossy = new_action is not action
        action = new_action

    try:
        decoded = decode(action, obs)
    except Exception as e:
        return {"ok": False, "decoded": None, "error": f"decode {type(e).__name__}: {e}"}

    # §17.0 carrier diagnostic: deliberately corrupt an action packet's target
    # field. Independent of quant/obsrel (those touch only the MoveAction path).
    with _QUANT_LOCK:
        perturb = _PERTURB
    perturbed = False
    if perturb:
        perturbed = _apply_perturb(packet_id, decoded, perturb)
    if perturbed:
        # The corruption IS the experiment — force substitution so the wrong
        # target goes on the wire (fidelity is intentionally broken).
        return {"ok": True, "decoded": decoded, "error": None,
                "perturbed": True, "fidelity_ok": False}

    fidelity_ok = fields_close(decoded, fields, atol=1e-6)

    # `ok` is the SUBSTITUTION signal homunculus gates on (CodecPassthrough.java:330):
    # ok=true → it reconstructs `decoded` onto the wire; ok=false → it counts drift
    # and sends the ORIGINAL packet unchanged. In lossless/identity mode ok=fidelity
    # (an unintended mismatch must NOT be put on the wire). In LOSSY mode the
    # quantization error IS the experiment, so the mismatch is intentional and we
    # MUST substitute — otherwise homunculus silently passes the original through and
    # the controller runs effectively lossless (the parity curve would be a flat
    # artifact). So ok=true under lossy; true fidelity is reported as `fidelity_ok`.
    if quantized_bits is not None or blockpos_lossy:
        out: dict[str, Any] = {
            "ok": True, "decoded": decoded, "error": None,
            "lossy": True, "fidelity_ok": fidelity_ok, "float_bits": quantized_bits,
        }
    else:
        out = {"ok": fidelity_ok, "decoded": decoded,
               "error": None if fidelity_ok else "fields_close mismatch",
               "fidelity_ok": fidelity_ok}
    return out


class _Handler(BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler's default log line goes to stderr on every
    # request — at 20Hz × 10 agents that's a flood. Silence it; we log only
    # on errors.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - signature
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path in ("/healthz", "/health"):
            self._respond(200, {"ok": True, "registered": registered_types(),
                                "quant": _full_snapshot(), "stats": _stats_snapshot()})
            return
        if self.path == "/stats/reset":
            with _STATS_LOCK:
                _STATS["roundtrips"] = 0
                _STATS["max_inflight"] = 0
            self._respond(200, {"ok": True, "stats": _stats_snapshot()})
            return
        if self.path == "/config":
            self._respond(200, {"ok": True, "quant": _full_snapshot()})
            return
        self._respond(404, {"ok": False, "error": f"unknown route {self.path}"})

    def _handle_config(self) -> None:
        """Live-retune the Sprint A lossy level without restarting the server.

        Body forms (all optional, missing axes left unchanged unless
        ``quant_bits`` is given which sets all three):
          {"quant_bits": 4}                  -> pos=yaw=pitch=4
          {"quant_bits": null}               -> lossless (all None)
          {"pos_bits": 5, "yaw_bits": 6}     -> per-axis
        """
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length < 0 or length > _MAX_BODY_BYTES:
            self._respond(400, {"ok": False, "error": f"bad content-length: {length}"})
            return
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
        except Exception as e:
            self._respond(400, {"ok": False, "error": f"bad json: {e}"})
            return
        if not isinstance(body, dict):
            self._respond(400, {"ok": False, "error": "body must be a JSON object"})
            return
        cur = _quant_snapshot()
        if "quant_bits" in body:
            qb = body["quant_bits"]
            if qb is None:
                _set_quant(None, None, None)
            elif isinstance(qb, int):
                _set_quant(qb, qb, qb)
            else:
                self._respond(400, {"ok": False, "error": "quant_bits must be int or null"})
                return
        else:
            def _pick(key: str) -> int | None:
                if key not in body:
                    return cur[key]
                v = body[key]
                return v if (v is None or isinstance(v, int)) else "BAD"  # type: ignore[return-value]
            pb, yb, ptb = _pick("pos_bits"), _pick("yaw_bits"), _pick("pitch_bits")
            if "BAD" in (pb, yb, ptb):
                self._respond(400, {"ok": False, "error": "*_bits must be int or null"})
                return
            _set_quant(pb, yb, ptb)  # type: ignore[arg-type]
        if "pos_range" in body:
            pr = body["pos_range"]
            if not (pr is None or isinstance(pr, (int, float))):
                self._respond(400, {"ok": False, "error": "pos_range must be number or null"})
                return
            _set_pos_range(float(pr) if pr is not None else None)
        if "pos_mode" in body:
            pm = body["pos_mode"]
            if pm not in POS_MODES:
                self._respond(400, {"ok": False,
                                    "error": f"pos_mode must be one of {sorted(POS_MODES)}"})
                return
            _set_pos_mode(pm)
        if "obsrel" in body:
            orl = body["obsrel"]
            if not isinstance(orl, bool):
                self._respond(400, {"ok": False, "error": "obsrel must be a bool"})
                return
            _set_obsrel(orl)
        if "perturb" in body:
            pt = body["perturb"]
            if pt is not None and not isinstance(pt, dict):
                self._respond(400, {"ok": False, "error": "perturb must be an object or null"})
                return
            if isinstance(pt, dict) and "block_pos_delta" in pt:
                d = pt["block_pos_delta"]
                if not (isinstance(d, list) and len(d) == 3
                        and all(isinstance(v, (int, float)) for v in d)):
                    self._respond(400, {"ok": False,
                                        "error": "block_pos_delta must be 3 numbers"})
                    return
            _set_perturb(pt)
        if "blockpos" in body:
            bp = body["blockpos"]
            if bp is not None and not isinstance(bp, dict):
                self._respond(400, {"ok": False, "error": "blockpos must be an object or null"})
                return
            if isinstance(bp, dict):
                if not isinstance(bp.get("bits"), int):
                    self._respond(400, {"ok": False, "error": "blockpos.bits must be an int"})
                    return
                if bp.get("mode", "obsrel") not in ("obsrel", "absolute"):
                    self._respond(400, {"ok": False,
                                        "error": "blockpos.mode must be 'obsrel' or 'absolute'"})
                    return
                for k in ("reach", "abs_range"):
                    if k in bp and not isinstance(bp[k], (int, float)):
                        self._respond(400, {"ok": False,
                                            "error": f"blockpos.{k} must be a number"})
                        return
            _set_blockpos(bp)
        snap = _full_snapshot()
        _log.info("quant config -> %s", snap)
        self._respond(200, {"ok": True, "quant": snap})

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        if self.path == "/config":
            self._handle_config()
            return
        if self.path != "/codec/roundtrip":
            self._respond(404, {"ok": False, "error": f"unknown route {self.path}"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._respond(400, {"ok": False, "error": f"bad content-length: {length}"})
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw)
        except Exception as e:
            self._respond(400, {"ok": False, "error": f"bad json: {e}"})
            return
        if not isinstance(body, dict):
            self._respond(400, {"ok": False, "error": "body must be a JSON object"})
            return
        packet_id = body.get("id")
        fields = body.get("fields")
        obs = body.get("obs")
        if not isinstance(packet_id, str) or not isinstance(fields, dict) or not isinstance(obs, dict):
            self._respond(400, {"ok": False, "error": "id:str, fields:obj, obs:obj required"})
            return
        _stats_enter()
        try:
            result = roundtrip(packet_id, fields, obs)
        finally:
            _stats_exit()
        self._respond(200, result)

    def _respond(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _Server(ThreadingHTTPServer):
    # The stock backlog of 5 (socketserver default) drops connections under a
    # fleet burst (15 agents x ~20Hz substituted packets). A dropped SYN costs a
    # ~1-3s TCP retransmit, which surfaces as multi-second tail stalls that freeze
    # a 20Hz control loop into a leg timeout — measured at k>=8 in the codec load
    # test even with zero application errors. A deep accept queue absorbs the
    # burst, killing both the connection drops AND the retransmit tail.
    request_queue_size = 256
    # Don't let lingering request threads block process shutdown on restart.
    daemon_threads = True


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Build and return the server. Caller decides whether to run forever or
    drive it from a test via ``serve_forever`` in a thread. ThreadingHTTPServer
    is fine here — the codec is pure-function so per-request concurrency has no
    shared state to corrupt."""
    server = _Server((host, port), _Handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="craft codec round-trip server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quant-bits", type=int, default=None,
                        help="Sprint A lossy mode: quantize the movement family to N "
                             "bits/field (pos+yaw+pitch). Omit for lossless. Retune "
                             "live via POST /config {\"quant_bits\": N}.")
    parser.add_argument("--pos-bits", type=int, default=None,
                        help="per-axis override of --quant-bits for pos dx/dy/dz")
    parser.add_argument("--yaw-bits", type=int, default=None,
                        help="per-axis override of --quant-bits for yaw")
    parser.add_argument("--pitch-bits", type=int, default=None,
                        help="per-axis override of --quant-bits for pitch")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.quant_bits is not None or any(
        x is not None for x in (args.pos_bits, args.yaw_bits, args.pitch_bits)
    ):
        base = args.quant_bits
        _set_quant(
            args.pos_bits if args.pos_bits is not None else base,
            args.yaw_bits if args.yaw_bits is not None else base,
            args.pitch_bits if args.pitch_bits is not None else base,
        )

    server = serve(host=args.host, port=args.port)
    _log.info(
        "codec server listening on http://%s:%d (codecs: %s) quant=%s",
        args.host, args.port, ",".join(registered_types()), _full_snapshot(),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("shutdown")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
