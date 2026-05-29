"""Frozen-validation-set capture runner (neural_interface.md §8e).

Orchestrates N LLM-driven rollouts, arming BOTH obs streams per rollout — the
light per-packet recording (`/packets/recording`) and the heavy tick sidecar
(`/obs/sidecar`) — then disarming, verifying the two streams join by `tick`,
and writing a manifest with per-file content hashes.

The point of the frozen set: every obs-ablation rung (§8b R0–R4) is measured
on the *same* eval packets. So this runner is what produces the immutable
artifact — disjoint from training rollouts, manifested, hashed. The
"--purpose dry_run" mode is for a 2–3 rollout shakedown: it measures real
footprint and confirms the join holds across full rollouts before we commit to
a large capture (and before deciding whether the §8e gzip/dense-array levers
are needed).

Each rollout writes to ``<out>/rollout-<i>/``:
  packets.jsonl   — per-packet recording (PacketRecorder)
  sidecar.jsonl   — per-tick heavy channels (TickSidecarRecorder)
  agent.jsonl     — the agent's own turn/llm records (--jsonl-out)

Usage:
  .venv/bin/python -m experiments.next_packet.capture \
      --rollouts 3 --turns 8 --goal minimal \
      --model "$QWEN" --port 25570 --player agent0 \
      --out results/frozen_dryrun

The agent subprocess inherits the scout-model env from this process, so set
CRAFT_SCOUT_FANOUT_MODEL / CRAFT_SCOUT_UNIFY_MODEL / CRAFT_LOOK_AROUND_MAX_RADIUS
before invoking (the daily-driver config).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def _base(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _open_text(path: Path):
    """Text handle, transparently gunzipping ``.gz`` (the sidecar is gzipped)."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arm(base: str, route: str, path: Path, gzip_flag: bool = False) -> dict:
    body: dict = {"path": str(path)}
    if gzip_flag:
        body["gzip"] = True
    r = requests.post(f"{base}{route}/arm", json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def _disarm(base: str, route: str) -> dict:
    r = requests.post(f"{base}{route}/disarm", timeout=15)
    r.raise_for_status()
    return r.json()


def _spawn_info(packets_path: Path, agent_path: Path) -> dict:
    """Approximate spawn: first packet obs at a survival Y (the spawn dance
    probes in spectator at y≈320, so skip y≥200), plus the biome the agent
    logs at turn 1. Biome is the load-bearing spawn confound; coords are a
    convenience."""
    info: dict = {"x": None, "y": None, "z": None, "biome": None}
    if packets_path.exists():
        with open(packets_path, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)["obs"]
                    y = o.get("y")
                    if isinstance(y, (int, float)) and y < 200:
                        info["x"], info["y"], info["z"] = o.get("x"), y, o.get("z")
                        break
                except (ValueError, KeyError):
                    continue
    if agent_path.exists():
        with open(agent_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("_type") == "turn":
                        info["biome"] = rec.get("biome")
                        break
                except ValueError:
                    continue
    return info


def _verify_join(packets_path: Path, sidecar_path: Path) -> dict:
    """Every packet's obs.tick should have a matching sidecar row (§8e join)."""
    pk_ticks: set[int] = set()
    if packets_path.exists():
        with _open_text(packets_path) as f:
            for line in f:
                try:
                    pk_ticks.add(json.loads(line)["obs"]["tick"])
                except (ValueError, KeyError):
                    continue
    sc_ticks: set[int] = set()
    if sidecar_path.exists():
        with _open_text(sidecar_path) as f:
            for line in f:
                try:
                    sc_ticks.add(json.loads(line)["tick"])
                except (ValueError, KeyError):
                    continue
    joined = pk_ticks & sc_ticks
    missing = sorted(pk_ticks - sc_ticks)
    return {
        "packet_ticks": len(pk_ticks),
        "sidecar_ticks": len(sc_ticks),
        "joined": len(joined),
        "join_pct": round(100.0 * len(joined) / len(pk_ticks), 2) if pk_ticks else None,
        "packet_ticks_missing_sidecar": len(missing),
        "missing_sample": missing[:10],
    }


def _file_stats(path: Path, line_key: str = "lines") -> dict:
    lines = 0
    if path.exists():
        with _open_text(path) as f:  # gz-aware: counts decompressed rows
            lines = sum(1 for _ in f)
    return {
        "path": str(path),
        line_key: lines,
        "bytes": path.stat().st_size if path.exists() else 0,  # on-disk (compressed)
        "sha256": _sha256(path),  # over stored bytes — integrity of the artifact as written
    }


def run_capture(args: argparse.Namespace) -> dict:
    base = _base(args.port)
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=True)

    # Preflight: client up?
    try:
        requests.get(f"{base}/position", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"FATAL: client at {base} not responding ({e})", file=sys.stderr)
        sys.exit(2)

    craft_root = Path(__file__).resolve().parents[2]
    homunculus_root = craft_root.parent / "homunculus"

    manifest: dict = {
        "schema": "frozen_capture_v1",
        "purpose": args.purpose,
        "model": args.model,
        "substrate": {
            "port": args.port,
            "player": args.player,
            "scout_fanout_model": os.environ.get("CRAFT_SCOUT_FANOUT_MODEL"),
            "scout_unify_model": os.environ.get("CRAFT_SCOUT_UNIFY_MODEL"),
            "look_around_max_radius": os.environ.get("CRAFT_LOOK_AROUND_MAX_RADIUS"),
        },
        "commits": {
            "craft": _git_head(craft_root),
            "homunculus": _git_head(homunculus_root),
        },
        "agent": {
            "turns": args.turns, "goal": args.goal,
            "spawn_range": args.spawn_range,
            "start_phase": args.start_phase, "difficulty": args.difficulty,
            "narrate": args.narrate,
        },
        "rollouts": [],
        "notes": [
            "Capture includes the spawn dance (recorders armed before the agent "
            "subprocess); spectator-phase packets carry g_t=null and can be "
            "filtered by early tick. A post-spawn arm is a later refinement.",
        ],
    }

    env = dict(os.environ)
    env["HOMUNCULUS_PORT"] = str(args.port)
    env["MC_PLAYER_NAME"] = args.player

    for i in range(args.rollouts):
        rdir = out / f"rollout-{i}"
        rdir.mkdir(parents=True, exist_ok=True)
        packets_path = rdir / "packets.jsonl"
        sidecar_path = rdir / "sidecar.jsonl.gz"  # gzipped (§8e footprint)
        agent_path = rdir / "agent.jsonl"

        print(f"\n=== rollout {i+1}/{args.rollouts} → {rdir} ===", flush=True)
        # Arm sidecar FIRST (and disarm it LAST, below) so its per-tick coverage
        # is a strict superset of the packet stream — no arm-gap boundary tick
        # where a packet is recorded before the sidecar starts → 100% join.
        _arm(base, "/obs/sidecar", sidecar_path, gzip_flag=True)
        # Let the sidecar run a few ticks before packets start. A packet emitted
        # in the same game-tick the recorders arm references the pose snapshot
        # from the *previous* tick — one older than the sidecar's first row — so
        # without this gap it would be the lone unjoinable packet per rollout.
        time.sleep(0.25)
        _arm(base, "/packets/recording", packets_path)

        cmd = [
            sys.executable, "-m", "craft.agent",
            str(args.turns), args.goal,
            "--model", args.model,
            "--random-spawn-range", str(args.spawn_range),
            "--start-phase", args.start_phase,
            "--difficulty", args.difficulty,
            "--jsonl-out", str(agent_path),
        ]
        if args.narrate:
            cmd.append("--narrate")  # §12.2: g_t = free-text intent (not tool name)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(craft_root), env=env)
        wall = time.time() - t0

        # Disarm packets first, sidecar last (mirror of the arm order) so the
        # sidecar covers every tick the packet recorder saw.
        pk_final = _disarm(base, "/packets/recording")
        sc_final = _disarm(base, "/obs/sidecar")

        join = _verify_join(packets_path, sidecar_path)
        entry = {
            "index": i,
            "agent_exit": proc.returncode,
            "wall_s": round(wall, 1),
            "spawn": _spawn_info(packets_path, agent_path),
            "packets": {**_file_stats(packets_path, "lines"),
                        "drops_queue_full": pk_final.get("dropped_queue_full"),
                        "drops_no_obs": pk_final.get("dropped_no_obs")},
            "sidecar": {**_file_stats(sidecar_path, "rows"),
                        "drops_queue_full": sc_final.get("dropped_queue_full"),
                        "drops_no_player": sc_final.get("dropped_no_player")},
            "agent_jsonl": _file_stats(agent_path, "lines"),
            "join": join,
        }
        manifest["rollouts"].append(entry)
        print(
            f"  packets={entry['packets']['lines']} "
            f"sidecar={entry['sidecar']['rows']} "
            f"join={join['join_pct']}% "
            f"bytes(pkt/side)={entry['packets']['bytes']}/{entry['sidecar']['bytes']} "
            f"wall={entry['wall_s']}s exit={proc.returncode}",
            flush=True,
        )

    # Totals + content hash (hash of the per-file sha256s, order-stable).
    total_bytes = 0
    file_hashes: list[str] = []
    for r in manifest["rollouts"]:
        for k in ("packets", "sidecar", "agent_jsonl"):
            total_bytes += r[k]["bytes"]
            if r[k]["sha256"]:
                file_hashes.append(r[k]["sha256"])
    manifest["totals"] = {
        "rollouts": len(manifest["rollouts"]),
        "packets": sum(r["packets"]["lines"] for r in manifest["rollouts"]),
        "sidecar_rows": sum(r["sidecar"]["rows"] for r in manifest["rollouts"]),
        "bytes": total_bytes,
    }
    manifest["content_hash"] = hashlib.sha256(
        "".join(sorted(file_hashes)).encode()
    ).hexdigest()
    manifest["created_at_ms"] = int(time.time() * 1000)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n=== manifest → {manifest_path} ===")
    print(f"  rollouts={manifest['totals']['rollouts']} "
          f"packets={manifest['totals']['packets']} "
          f"sidecar_rows={manifest['totals']['sidecar_rows']} "
          f"total_bytes={total_bytes} ({total_bytes/1e6:.1f} MB)")
    print(f"  content_hash={manifest['content_hash'][:16]}…")
    worst_join = min((r["join"]["join_pct"] for r in manifest["rollouts"]
                      if r["join"]["join_pct"] is not None), default=None)
    print(f"  worst per-rollout join: {worst_join}%")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-validation-set capture runner (§8e).")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--goal", default="minimal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--player", default="agent0")
    ap.add_argument("--spawn-range", type=int, default=5000)
    ap.add_argument("--start-phase", default="none",
                    choices=["dawn", "noon", "dusk", "midnight", "random", "none"])
    ap.add_argument("--difficulty", default="easy",
                    choices=["peaceful", "easy", "normal", "hard"])
    ap.add_argument("--out", default="results/frozen_dryrun")
    ap.add_argument("--purpose", default="dry_run")
    ap.add_argument("--narrate", action="store_true",
                    help="§12.2: pass --narrate to the agent so g_t is free-text intent "
                         "(not the tool name); prerequisite for the §12.3 moat-width decode")
    run_capture(ap.parse_args())


if __name__ == "__main__":
    main()
