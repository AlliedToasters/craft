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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from craft.codec.base import decode, encode, fields_close, registered_types

DEFAULT_PORT = 25600
DEFAULT_HOST = "127.0.0.1"
_MAX_BODY_BYTES = 64 * 1024

_log = logging.getLogger("craft.codec.server")


def roundtrip(packet_id: str, fields: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    """The whole point: ``decode(encode(...), ...)`` and report whether the
    output matches the input. Exposed at module scope so unit tests can hit
    the same surface as the HTTP endpoint without standing up a server."""
    try:
        action = encode(packet_id, fields, obs)
    except KeyError as e:
        return {"ok": False, "decoded": None, "error": f"encode KeyError: {e}"}
    except Exception as e:
        return {"ok": False, "decoded": None, "error": f"encode {type(e).__name__}: {e}"}
    try:
        decoded = decode(action, obs)
    except Exception as e:
        return {"ok": False, "decoded": None, "error": f"decode {type(e).__name__}: {e}"}
    ok = fields_close(decoded, fields, atol=1e-6)
    return {"ok": ok, "decoded": decoded, "error": None if ok else "fields_close mismatch"}


class _Handler(BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler's default log line goes to stderr on every
    # request — at 20Hz × 10 agents that's a flood. Silence it; we log only
    # on errors.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - signature
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path in ("/healthz", "/health"):
            self._respond(200, {"ok": True, "registered": registered_types()})
            return
        self._respond(404, {"ok": False, "error": f"unknown route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
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
        result = roundtrip(packet_id, fields, obs)
        self._respond(200, result)

    def _respond(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Build and return the server. Caller decides whether to run forever or
    drive it from a test via ``serve_forever`` in a thread. ThreadingHTTPServer
    is fine here — the codec is pure-function so per-request concurrency has no
    shared state to corrupt."""
    server = ThreadingHTTPServer((host, port), _Handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="craft codec round-trip server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    server = serve(host=args.host, port=args.port)
    _log.info(
        "codec server listening on http://%s:%d (codecs: %s)",
        args.host, args.port, ",".join(registered_types()),
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
