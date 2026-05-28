"""Inference HTTP server for next-packet prediction (neural_interface.md §8).

Loads a checkpoint at startup and serves predictions from live obs snapshots.
Same transport pattern as craft/codec/server.py — a ThreadingHTTPServer so
concurrent requests from multiple Java clients don't queue.

Routes:
  POST /infer
    Body: {"obs": {...}}        -- obs dict from homunculus PlayerObsSnapshot
    Response: {
      "predicted_type": str,   -- top-1 predicted packet type
      "confidence": float,     -- softmax probability of top-1
      "top3": [
        {"type": str, "prob": float}, ...
      ],
      "latency_ms": float,     -- Python-side inference time (not round-trip)
    }

  GET /healthz
    Response: {"ok": true, "checkpoint": str, "metadata": {...}}

Usage:
    python -m experiments.next_packet.inference_server \
        --checkpoint checkpoints/r0.pt \
        --port 25601

Then from anywhere:
    curl -s -X POST http://127.0.0.1:25601/infer \
      -H 'Content-Type: application/json' \
      -d '{"obs": {"tick":100,"x":100.0,"y":64.0,"z":-200.0,"yaw":45.0,"pitch":-10.0,"on_ground":true,"dim":"minecraft:overworld"}}'
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from .checkpoint import load_checkpoint
from .features import PACKET_TYPES


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


_model: Any = None
_normalizer: Any = None  # FeatureNormalizer; typed as Any to avoid forward-ref issues
_metadata: dict = {}
_checkpoint_path: str = ""
_device: Any = None


def _ensure_loaded(checkpoint_path: str) -> None:
    global _model, _normalizer, _metadata, _checkpoint_path, _device
    import torch
    _model, _normalizer, _metadata = load_checkpoint(checkpoint_path)
    _checkpoint_path = checkpoint_path
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference_server] loaded checkpoint: {checkpoint_path}")
    print(f"[inference_server] metadata: {_metadata}")
    print(f"[inference_server] device: {_device}")


def predict(obs: dict[str, Any]) -> dict[str, Any]:
    """Run inference on a single obs dict. Returns the prediction payload."""
    import torch

    t0 = time.perf_counter()
    fv = _normalizer.transform(obs)
    x = torch.tensor([fv.values], dtype=torch.float32).to(_device)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    top3_idx = probs.topk(min(3, len(PACKET_TYPES))).indices.tolist()
    top3 = [{"type": PACKET_TYPES[i], "prob": float(probs[i])} for i in top3_idx]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "predicted_type": top3[0]["type"],
        "confidence": top3[0]["prob"],
        "top3": top3,
        "latency_ms": latency_ms,
    }


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/infer":
            self._handle_infer()
        else:
            self._respond(404, {"ok": False, "error": f"unknown route: {self.path}"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._respond(200, {
                "ok": True,
                "checkpoint": _checkpoint_path,
                "metadata": _metadata,
            })
        else:
            self._respond(404, {"ok": False, "error": f"unknown route: {self.path}"})

    def _handle_infer(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as e:
            self._respond(400, {"ok": False, "error": f"bad json: {e}"})
            return
        obs = payload.get("obs")
        if obs is None:
            self._respond(400, {"ok": False, "error": "missing 'obs' key"})
            return
        try:
            result = predict(obs)
        except Exception as e:
            self._respond(500, {"ok": False, "error": f"inference error: {e}"})
            return
        self._respond(200, result)

    def _respond(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # silence per-request log
        pass


def serve(checkpoint_path: str, host: str = "127.0.0.1", port: int = 25601) -> ThreadingHTTPServer:
    _ensure_loaded(checkpoint_path)
    server = ThreadingHTTPServer((host, port), _Handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-packet inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25601)
    args = parser.parse_args()

    server = serve(args.checkpoint, host=args.host, port=args.port)
    print(f"[inference_server] listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[inference_server] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
