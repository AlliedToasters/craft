"""Direct load test of the codec roundtrip server (:25600).

Isolates the server's throughput/latency ceiling with NO MC / Baritone in the
loop, to test the hypothesis: packet-intervention via the single shared codec
server is the operational ceiling on fleet size n (the b=5 cliff may be a
server-saturation artifact, not true loss-intolerance).

Closed-loop: K worker threads each loop {send /codec/roundtrip, time it, repeat}
for DURATION seconds. Achieved req/s at concurrency K = sustained throughput;
as K rises, the throughput plateau is the server ceiling and latency p99 is what
each agent's homunculus waits per substituted packet.

Realistic offered load reference: 15 agents x ~20Hz = ~300 req/s baseline.

Output: JSON to --out. Prints a fresh nonce first (anti-fabrication discipline).
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

# A realistic move_player_pos_rot packet: small walk delta + abs rotation.
# obs supplies the reference frame the pos delta is subtracted against.
_BODY = {
    "id": "minecraft:move_player_pos_rot",
    "fields": {
        "has_pos": True, "has_rot": True,
        "x": 100.21, "y": 64.0, "z": 100.05,
        "yaw": 134.5, "pitch": 12.0,
        "on_ground": True, "horizontal_collision": False,
    },
    "obs": {"x": 100.0, "y": 64.0, "z": 100.0},
}
_PAYLOAD = json.dumps(_BODY).encode("utf-8")


def _one_request(url: str, timeout: float) -> tuple[float, bool]:
    """Return (latency_s, ok). ok=False on any transport error/timeout."""
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=_PAYLOAD,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return time.perf_counter() - t0, True
    except Exception:
        return time.perf_counter() - t0, False


def _worker(url: str, deadline: float, timeout: float,
            lats: list[float], errs: list[int]) -> None:
    local_lats: list[float] = []
    local_errs = 0
    while time.perf_counter() < deadline:
        lat, ok = _one_request(url, timeout)
        local_lats.append(lat)
        if not ok:
            local_errs += 1
    lats.extend(local_lats)
    errs.append(local_errs)


def run_level(url: str, k: int, duration: float, timeout: float) -> dict:
    lats: list[float] = []
    errs: list[int] = []
    threads = [threading.Thread(target=_worker,
                                args=(url, time.perf_counter() + duration, timeout, lats, errs))
               for _ in range(k)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    n = len(lats)
    total_err = sum(errs)
    lats_ms = sorted(x * 1000.0 for x in lats)

    def pct(p: float) -> float:
        if not lats_ms:
            return 0.0
        i = min(len(lats_ms) - 1, int(round(p / 100.0 * (len(lats_ms) - 1))))
        return round(lats_ms[i], 2)

    return {
        "k": k, "elapsed_s": round(elapsed, 3), "requests": n,
        "errors": total_err,
        "req_per_s": round(n / elapsed, 1) if elapsed else 0.0,
        "lat_ms_p50": pct(50), "lat_ms_p90": pct(90),
        "lat_ms_p99": pct(99), "lat_ms_max": round(lats_ms[-1], 2) if lats_ms else 0.0,
        "lat_ms_mean": round(statistics.mean(lats_ms), 2) if lats_ms else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25600)
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--nonce", default="NONONCE")
    ap.add_argument("--out", default="/tmp/codec_loadtest.json")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/codec/roundtrip"
    ks = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(args.nonce, flush=True)
    print(f"url={url} duration={args.duration}s ks={ks}", flush=True)

    results = []
    for k in ks:
        r = run_level(url, k, args.duration, args.timeout)
        results.append(r)
        print(f"k={r['k']:>3}  req/s={r['req_per_s']:>8}  "
              f"p50={r['lat_ms_p50']:>7}ms  p99={r['lat_ms_p99']:>8}ms  "
              f"max={r['lat_ms_max']:>8}ms  errs={r['errors']}", flush=True)

    out = {"nonce": args.nonce, "url": url, "duration_s": args.duration,
           "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
