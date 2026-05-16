#!/usr/bin/env python3
"""HTTP bridge to the Minecraft server console.

POST /cmd   {"cmd": "say hello"}   → {"ok": true}
GET  /log?n=50                     → {"lines": [...]}

Expects the Minecraft server (Purpur / Paper / vanilla) to be running
inside a named ``screen`` session, with ``logs/latest.log`` reachable
from the script's working directory. See server/README.md.

Config (env vars):
  MC_SCREEN_SESSION   default: server   — name of the screen session
                                          hosting the MC server console.
  MC_LOG_PATH         default: ./logs/latest.log (relative to CWD)
  MC_API_PORT         default: 4747     — HTTP listen port.
  MC_API_HOST         default: 0.0.0.0  — HTTP listen address.
"""

import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCREEN_SESSION = os.environ.get("MC_SCREEN_SESSION", "server")
LOG_PATH = Path(os.environ.get("MC_LOG_PATH", "logs/latest.log"))
PORT = int(os.environ.get("MC_API_PORT", "4747"))
HOST = os.environ.get("MC_API_HOST", "0.0.0.0")


def send_cmd(cmd: str) -> None:
    subprocess.run(
        ["screen", "-S", SCREEN_SESSION, "-X", "stuff", cmd + "\r"],
        check=True,
    )


def tail_log(n: int) -> list[str]:
    if not LOG_PATH.exists():
        return []
    return LOG_PATH.read_text(errors="replace").splitlines()[-n:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default access log noise

    def _respond(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/cmd":
            self._respond(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
            cmd = str(body["cmd"])
        except (json.JSONDecodeError, KeyError):
            self._respond(400, {"error": "expected JSON body with 'cmd' key"})
            return
        try:
            send_cmd(cmd)
            self._respond(200, {"ok": True})
        except subprocess.CalledProcessError as e:
            self._respond(500, {"error": str(e)})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/log":
            self._respond(404, {"error": "not found"})
            return
        params = parse_qs(parsed.query)
        try:
            n = int(params.get("n", ["50"])[0])
            n = max(1, min(n, 2000))
        except ValueError:
            self._respond(400, {"error": "n must be an integer"})
            return
        self._respond(200, {"lines": tail_log(n)})


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"mc_api listening on {HOST}:{PORT}")
    print(f"  screen session: {SCREEN_SESSION}")
    print(f"  log path:       {LOG_PATH}")
    server.serve_forever()
