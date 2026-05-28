"""Unit tests for ``craft.codec.server`` (ml.MD §4a step 2 plumbing).

Two layers of testing:

  1. **In-process** — call ``roundtrip(...)`` directly. Validates the codec
     wiring without the HTTP loopback. Cheap, fast.
  2. **Loopback** — start an actual ``ThreadingHTTPServer`` on a random port,
     POST one sample per codec via ``urllib``, verify the response shape.
     Catches body-parsing / Content-Length / status-code regressions that
     in-process tests miss.

Why both: the Java side hits the HTTP surface, so even though the codec
itself is exercised by the per-codec round-trip tests, an HTTP-layer
regression in this server would silently break step 2 in the live rollout.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from typing import Any

import pytest

from craft.codec import server


_OBS = {
    "tick": 1,
    "captured_at_ms": 0,
    "x": 100.5,
    "y": 64.0,
    "z": -200.25,
    "yaw": 45.0,
    "pitch": -10.0,
    "on_ground": True,
    "dim": "minecraft:overworld",
}


# One minimal-but-valid fields dict per codec. Used by both the in-process
# tests and the HTTP loopback test. If a new codec lands without an entry
# here, the all-codecs test below will fail loudly.
_SAMPLES: list[tuple[str, dict[str, Any]]] = [
    ("minecraft:move_player_pos_rot", {
        "has_pos": True, "has_rot": True,
        "x": _OBS["x"], "y": _OBS["y"], "z": _OBS["z"],
        "yaw": _OBS["yaw"], "pitch": _OBS["pitch"],
        "on_ground": True, "horizontal_collision": False,
    }),
    ("minecraft:move_player_pos", {
        "has_pos": True, "has_rot": False,
        "x": _OBS["x"], "y": _OBS["y"], "z": _OBS["z"],
        "on_ground": True, "horizontal_collision": False,
    }),
    ("minecraft:move_player_rot", {
        "has_pos": False, "has_rot": True,
        "yaw": _OBS["yaw"], "pitch": _OBS["pitch"],
        "on_ground": True, "horizontal_collision": False,
    }),
    ("minecraft:move_player_status_only", {
        "has_pos": False, "has_rot": False,
        "on_ground": True, "horizontal_collision": False,
    }),
    ("minecraft:swing", {"hand": "MAIN_HAND"}),
    ("minecraft:player_input", {
        "forward": True, "backward": False, "left": False, "right": False,
        "jump": False, "shift": False, "sprint": True,
    }),
    ("minecraft:player_command", {"entity_id": 42, "action": "START_SPRINTING", "data": 0}),
    ("minecraft:use_item", {"hand": "MAIN_HAND", "sequence": 7, "yaw": 0.0, "pitch": 0.0}),
    ("minecraft:use_item_on", {
        "hand": "MAIN_HAND",
        "block_pos": [100, 63, -200],
        "face": "UP",
        "cursor": [0.5, 1.0, 0.5],
        "inside": False,
        "world_border_hit": False,
        "sequence": 1,
    }),
    ("minecraft:player_action", {
        "action": "START_DESTROY_BLOCK",
        "block_pos": [100, 63, -200],
        "face": "UP",
        "sequence": 1,
    }),
    ("minecraft:interact", {
        "entity_id": 42, "using_secondary_action": False, "action": "ATTACK",
    }),
]


@pytest.mark.parametrize("packet_id,fields", _SAMPLES)
def test_roundtrip_inprocess(packet_id: str, fields: dict[str, Any]) -> None:
    """The codec is what's being tested; the server is just a wrapper. If
    this fails, the per-codec round-trip test would already have caught it
    — included here so the server's call shape is exercised before HTTP."""
    result = server.roundtrip(packet_id, fields, _OBS)
    assert result["ok"] is True, f"roundtrip failed: {result}"
    assert result["error"] is None
    assert isinstance(result["decoded"], dict)


def test_roundtrip_unknown_packet_type() -> None:
    """Unknown packet ids produce a structured error rather than raising —
    the Java side relies on the failure path being graceful."""
    result = server.roundtrip("minecraft:not_a_real_packet", {}, _OBS)
    assert result["ok"] is False
    assert result["error"] is not None
    assert "KeyError" in result["error"] or "no codec" in result["error"]
    assert result["decoded"] is None


def test_roundtrip_drift_when_obs_mismatch() -> None:
    """Move packets use pos-as-delta-against-obs; if we feed obs that
    contradicts the packet's claimed coordinates, the codec still round-trips
    because both encode and decode use the SAME obs (the delta cancels). The
    drift test is more interesting: change the fields to claim ``has_pos:True``
    with no x/y/z — that's a malformed packet and the codec should reject."""
    fields = {
        "has_pos": True, "has_rot": False,
        # Missing x/y/z — codec should raise during encode.
        "on_ground": True, "horizontal_collision": False,
    }
    result = server.roundtrip("minecraft:move_player_pos", fields, _OBS)
    assert result["ok"] is False
    assert result["error"] is not None


def _pick_free_port() -> int:
    """Bind to port 0 to get an unused port, then close. Race: another
    process could grab the port before we restart, but for tests on a
    single-user box this is fine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server():
    """Start the codec server on a random port and tear it down at test end.
    Yields (host, port) for the test to hit."""
    port = _pick_free_port()
    srv = server.serve(host="127.0.0.1", port=port)
    thread = threading.Thread(target=srv.serve_forever, name="test-codec-server", daemon=True)
    thread.start()
    # Tiny settle so the listening socket is ready. ThreadingHTTPServer
    # accepts on the main thread before serve_forever blocks, so this is
    # belt-and-braces, not load-bearing.
    time.sleep(0.05)
    try:
        yield ("127.0.0.1", port)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


def _post(host: str, port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data) if data else {}
    finally:
        conn.close()


def _get(host: str, port: int, path: str) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data) if data else {}
    finally:
        conn.close()


def test_health_endpoint(running_server) -> None:
    host, port = running_server
    status, body = _get(host, port, "/healthz")
    assert status == 200
    assert body["ok"] is True
    # Sanity: all 11 codecs must show up in the registered list — this is
    # the same surface the operator hits to confirm the server is sane.
    assert len(body["registered"]) == 11


@pytest.mark.parametrize("packet_id,fields", _SAMPLES)
def test_roundtrip_over_http(running_server, packet_id: str, fields: dict[str, Any]) -> None:
    host, port = running_server
    status, body = _post(host, port, "/codec/roundtrip", {
        "id": packet_id, "fields": fields, "obs": _OBS,
    })
    assert status == 200, f"unexpected status {status}: {body}"
    assert body["ok"] is True, f"drift over HTTP: {body}"
    assert body["error"] is None


def test_bad_json_returns_400(running_server) -> None:
    host, port = running_server
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        conn.request(
            "POST", "/codec/roundtrip",
            body=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        status = resp.status
        body = json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()
    assert status == 400
    assert body["ok"] is False
    assert "json" in body["error"].lower()


def test_unknown_route_returns_404(running_server) -> None:
    host, port = running_server
    status, body = _get(host, port, "/nope")
    assert status == 404
    assert body["ok"] is False


def test_missing_fields_returns_400(running_server) -> None:
    host, port = running_server
    # Missing 'fields' key — handler must reject.
    status, body = _post(host, port, "/codec/roundtrip", {
        "id": "minecraft:swing", "obs": _OBS,
    })
    assert status == 400
    assert body["ok"] is False


def test_all_codecs_have_a_sample() -> None:
    """If a new codec lands in the registry without a sample here, this
    fails — keeps the server-level coverage in lockstep with the codec set."""
    from craft.codec.base import registered_types
    sample_ids = {pid for pid, _ in _SAMPLES}
    registered = set(registered_types())
    missing = registered - sample_ids
    assert not missing, f"server test missing samples for: {sorted(missing)}"
