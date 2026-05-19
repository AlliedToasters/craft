"""End-to-end integration test runner.

Tests are flaky by design — the substrate is a live MC client plus
Baritone. The runner judges by **pass-rate over N iterations**, not by
demanding every iter succeed.

Each test is a subprocess (fresh interpreter, killable on timeout) that
writes one JSONL line per iteration. Each line includes `"passed": bool`
and `"fail_reason": str | null`. The runner reads the JSONL and computes
iters_passed / iters_total against the test's threshold.

Adding a test: extend TESTS below. The fields:
  name        — short, matches the test_<name>.py module
  cmd_base    — argv list; `--iters N --out <path>` get appended at run time
  threshold   — minimum pass-rate (0..1) for the suite to mark it PASS
  iters       — default iters when --iters isn't passed at the suite level
  timeout_s   — generous wall-clock subprocess cap per iter * iters
  summary     — one-liner for --list

Per-test JSONL goes to `results/suite-<name>.jsonl` so the runner knows
where to read.

NOT parallelizable: all tests share one player, one world, one Baritone
session lock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from craft.config import SERVER_CMD_BASE


def _preflight(base: str) -> str | None:
    """Confirm a homunculus base is reachable + has the routes the tests need."""
    try:
        r = requests.get(f"{base}/stats", timeout=3.0)
    except requests.RequestException as e:
        return f"homunculus unreachable at {base}: {e}"
    if not r.ok:
        return f"{base}/stats returned {r.status_code}"
    # Stale-jar check: /evasion/* would 404 if homunculus wasn't rebuilt.
    try:
        r2 = requests.get(f"{base}/evasion/status", timeout=3.0)
        if r2.status_code == 404:
            return (f"homunculus at {base} is running an old jar — /evasion/* "
                    "missing. Rebuild + deploy + restart MC.")
    except requests.RequestException:
        pass
    return None


def _judge_pass_rate(path: Path, threshold: float) -> tuple[bool, str, dict]:
    """Parse a test JSONL and compute pass-rate vs threshold.

    Each line is expected to have `passed: bool`. Missing == fail.
    Returns (passed_bool, summary_string, details_dict).
    """
    if not path.exists():
        return False, f"no output JSONL at {path}", {"iters": 0}
    iters: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                iters.append(json.loads(line))
            except json.JSONDecodeError as e:
                return False, f"malformed JSONL line: {e}", {"iters": 0}
    if not iters:
        return False, "JSONL empty (no iters ran)", {"iters": 0}
    n = len(iters)
    passed = sum(1 for r in iters if r.get("passed"))
    rate = passed / n
    fail_reasons: list[str] = []
    for r in iters:
        if not r.get("passed"):
            i = r.get("iter", "?")
            reason = r.get("fail_reason") or r.get("fatal_error") or "unknown"
            fail_reasons.append(f"iter {i}: {reason}")
    ok = rate >= threshold
    details = {
        "iters": n,
        "passed": passed,
        "rate": round(rate, 3),
        "threshold": threshold,
        "fail_reasons": fail_reasons,
    }
    summary = (f"{passed}/{n} passed (rate={rate:.2f}, threshold={threshold:.2f})"
               + (f" — {'; '.join(fail_reasons[:3])}"
                  + (f" (+{len(fail_reasons) - 3} more)" if len(fail_reasons) > 3 else "")
                  if fail_reasons else ""))
    return ok, summary, details


# Concurrent-mode agent assignment.
# launch_agent.sh N -> 1.21.4.agentN, port 25570+N, offline account agentN.
# Sequential mode uses no `concurrent_agent` key; defaults flow from env
# (MC_PLAYER_NAME / HOMUNCULUS_PORT) or the canonical main client on 25566.
# Default spawn-radius in concurrent mode. Agents must physically scatter
# so that one test's mob-spawn / arena-fill / @e selectors don't touch
# another's space. 3000 blocks gives ~36M square-block separation per pair.
CONCURRENT_SPAWN_RANGE = 3000

# Phase ordering for grouped concurrent runs. Tests with the same `world_state`
# run together; phases run sequentially. `peaceful` first so we can warm up
# without hostile interference; `non_peaceful` next; `mixed` last because tests
# in that phase self-manage difficulty (and may race with each other when N>1).
PHASE_ORDER: tuple[str, ...] = ("peaceful", "non_peaceful", "mixed")

# Coordinator-applied difficulty per phase. `mixed` is absent — tests own their
# transitions inside that phase (e.g., shelter does peaceful->easy internally).
PHASE_DIFFICULTY: dict[str, str] = {
    "peaceful": "peaceful",
    "non_peaceful": "easy",
}

# Concurrent-mode agent assignment.
# launch_agent.sh N -> 1.21.4.agentN, port 25570+N, offline account agentN.
#
# `world_state` decides which phase the test runs in; `concurrent_agents` is
# the list of agent indices it fans out to inside that phase. shelter fans
# to 3 agents because we want iter throughput on the slowest test — the
# remaining tests stay 1-agent until we add more iters or more tests.
#
# `concurrent_extra_args` is appended to cmd_base ONLY in --concurrent mode.
TESTS: list[dict] = [
    {
        "name": "mine_wood",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_mine_wood", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 180,
        "summary": "Plant 4 oak_logs, dispatch mine_wood(2), verify inventory delta.",
        "world_state": "peaceful",
        "concurrent_agents": [0],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "evasion",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_evasion", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 180,
        "summary": "Ambush 17 adult zombies, verify watcher fires + flee reaches anchor.",
        "world_state": "non_peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    # ---- Tool coverage suite (2026-05-15): one test per agent tool call.
    # All peaceful. Distributed across agents 0/1/2 to balance wall time;
    # within a single agent, tests run sequentially (per-agent queue).
    # Estimated wall budgets (smoke): mine_wood 20s, mine_stone 13s,
    # mine_iron 7s, mine_diamond 5s, mine_coal 9s, surface 31s, descend 11s,
    # travel 2s, place 1s, craft 2s, smelt 3s, collect_smelt 17s.
    #
    # Bin-packed: agent0 ~40s, agent1 ~37s, agent2 ~41s.
    {
        "name": "mine_stone",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_mine_stone", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 120,
        "summary": "Underground at spawn_y-10: stone-encase, dispatch, verify cobble delta.",
        "world_state": "peaceful",
        "concurrent_agents": [2],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "mine_iron",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_mine_ore",
                     "--species", "iron", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 90,
        "summary": "Plant 4 iron_ore at cardinals, dispatch mine_iron(2), verify raw_iron delta.",
        "world_state": "peaceful",
        "concurrent_agents": [2],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "mine_diamond",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_mine_ore",
                     "--species", "diamond", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 90,
        "summary": "Plant 4 diamond_ore at cardinals (iron pickaxe), dispatch, verify delta.",
        "world_state": "peaceful",
        "concurrent_agents": [2],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "mine_coal",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_mine_ore",
                     "--species", "coal", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 90,
        "summary": "Plant 4 coal_ore at cardinals (wooden pickaxe), dispatch, verify delta.",
        "world_state": "peaceful",
        "concurrent_agents": [2],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "surface",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_surface", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 180,
        "summary": "Bury at spawn_y-20 in encased pocket, dispatch surface() (chunked retry), verify ascent.",
        "world_state": "peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "descend",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_descend", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 120,
        "summary": "Dispatch descend(spawn_y-15), verify final y within 2 of target.",
        "world_state": "peaceful",
        "concurrent_agents": [0],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "travel",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_travel", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 60,
        "summary": "Build arena, dispatch travel(north, 10), verify Δz≈-10, |Δx|<4.",
        "world_state": "peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "place",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_place", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 60,
        "summary": "/give chest, dispatch place, verify placed_at within 5 blocks of player.",
        "world_state": "peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "craft",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_craft", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 60,
        "summary": "/give 4 oak_log, dispatch craft(oak_planks, 16), verify 16-plank delta.",
        "world_state": "peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "smelt",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_smelt", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 60,
        "summary": "/give furnace+raw_iron+coal, dispatch smelt, verify 'smelt started' + nearby furnace.",
        "world_state": "peaceful",
        "concurrent_agents": [1],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "collect_smelt",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.test_collect_smelt", "--quiet"],
        "threshold": 0.9,
        "iters": 1,
        "timeout_s": 90,
        "summary": "smelt 1x raw_iron, wait ~15s, dispatch collect_smelt, verify iron_ingot delta.",
        "world_state": "peaceful",
        "concurrent_agents": [0],
        "concurrent_extra_args": ["--spawn-range", str(CONCURRENT_SPAWN_RANGE)],
    },
    {
        "name": "shelter",
        "cmd_base": ["uv", "run", "python", "-m", "e2e.stress_test_shelter",
                     "--range", "0", "--ambush-seconds", "30"],
        "threshold": 0.66,  # shelter is the flakiest; stress over multiple iters helps
        "iters": 1,
        "timeout_s": 360,
        "summary": "Build shelter, ambush babies for 30s, judge no breach/death.",
        "world_state": "mixed",
        "concurrent_agents": [0, 1, 2, 3, 4],  # fan-out — 5x iters per --iters
        # shelter uses `--range` (different flag name); override sequential 0.
        # The leading replacement is intentional — we don't keep the 0 entry.
        "concurrent_extra_args": ["--range", str(CONCURRENT_SPAWN_RANGE)],
    },
]


def _agent_env(agent_n: int) -> dict[str, str]:
    """Return env-var overrides that retarget the substrate at agent<N>.

    Combined with os.environ in subprocess launch.
    """
    return {
        "HOMUNCULUS_PORT": str(25570 + agent_n),
        "MC_PLAYER_NAME": f"agent{agent_n}",
    }


def _base_for_agent(agent_n: int) -> str:
    return f"http://127.0.0.1:{25570 + agent_n}"


def _resolve_cmd(spec: dict, iters: int, *, concurrent: bool = False,
                 agent_n: int | None = None
                 ) -> tuple[list[str], Path]:
    """Append --iters, --out, and concurrent extras to the spec's cmd_base.

    In concurrent mode, output paths get an -agent<N> suffix so fan-out
    across multiple agents writes to separate JSONLs we can later combine.
    `concurrent_extra_args` (typically --spawn-range overrides) is appended
    so the test physically scatters its anchor. Later args win in argparse,
    so this overrides any baked-in defaults like `--range 0` in cmd_base.
    """
    if concurrent and agent_n is not None:
        out_path = Path(f"results/suite-{spec['name']}-agent{agent_n}.jsonl")
    else:
        out_path = Path(f"results/suite-{spec['name']}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = list(spec["cmd_base"]) + ["--iters", str(iters), "--out", str(out_path)]
    if concurrent:
        cmd.extend(spec.get("concurrent_extra_args", []))
    return cmd, out_path


def _judge_combined(name: str, threshold: float
                    ) -> tuple[bool, str, dict, list[Path]]:
    """Combine results/suite-<name>-agent*.jsonl + results/suite-<name>.jsonl
    into one pass-rate. Used by the phased runner when a spec fans out
    across multiple agents.
    """
    candidates = list(Path("results").glob(f"suite-{name}-agent*.jsonl"))
    single = Path(f"results/suite-{name}.jsonl")
    if not candidates and single.exists():
        candidates = [single]
    if not candidates:
        return False, f"no output JSONL for '{name}'", {"iters": 0}, []
    iters: list[dict] = []
    for p in candidates:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    iters.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not iters:
        return False, f"JSONL(s) empty for '{name}'", {"iters": 0}, candidates
    n = len(iters)
    passed = sum(1 for r in iters if r.get("passed"))
    rate = passed / n
    fail_reasons: list[str] = []
    for r in iters:
        if not r.get("passed"):
            i = r.get("iter", "?")
            reason = r.get("fail_reason") or r.get("fatal_error") or "unknown"
            fail_reasons.append(f"iter {i}: {reason}")
    ok = rate >= threshold
    details = {
        "iters": n,
        "passed": passed,
        "rate": round(rate, 3),
        "threshold": threshold,
        "fail_reasons": fail_reasons,
        "agent_count": len(candidates),
    }
    summary = (f"{passed}/{n} passed across {len(candidates)} agent(s) "
               f"(rate={rate:.2f}, threshold={threshold:.2f})"
               + (f" — {'; '.join(fail_reasons[:3])}"
                  + (f" (+{len(fail_reasons) - 3} more)"
                     if len(fail_reasons) > 3 else "")
                  if fail_reasons else ""))
    return ok, summary, details, candidates


def _coordinator_cleanup_phase() -> None:
    """Between-phase cleanup: kill all hostile/throwaway entities globally,
    nudge difficulty to peaceful to despawn any stragglers. Called before
    each phase change so leftover state from the previous phase doesn't
    bleed into the next.
    """
    requests.post(f"{SERVER_CMD_BASE}/cmd",
                  json={"cmd": "kill @e[type=!player,type=!item_frame,"
                                       "type=!armor_stand,type=!leash_knot]"},
                  timeout=10.0)
    requests.post(f"{SERVER_CMD_BASE}/cmd",
                  json={"cmd": "difficulty peaceful"}, timeout=5.0)


def _coordinator_set_phase_state(phase: str) -> None:
    """Apply the global state required by `phase` before launching tests."""
    if phase in PHASE_DIFFICULTY:
        diff = PHASE_DIFFICULTY[phase]
        requests.post(f"{SERVER_CMD_BASE}/cmd",
                      json={"cmd": f"difficulty {diff}"}, timeout=5.0)
    # mixed: no coordinator-set state; tests do their own peaceful->easy.


def run_test(spec: dict, *, iters: int, threshold: float,
             verbose: bool = True, env_overrides: dict[str, str] | None = None
             ) -> dict:
    """Run a single test subprocess and return its result record.

    `env_overrides` retargets the subprocess at a specific homunculus
    port + player name (see _agent_env). Empty/None = inherit the
    suite-runner's env (defaults: localhost:25566 + $MC_PLAYER_NAME).
    """
    name = spec["name"]
    cmd, out_path = _resolve_cmd(spec, iters)
    timeout_s = spec.get("timeout_s", 300)
    # Scale timeout by iter count — the per-iter budget is the base value.
    effective_timeout = timeout_s * max(1, iters)
    env = {**os.environ, **(env_overrides or {})}
    if verbose:
        target = ""
        if env_overrides:
            target = (f" [HOMUNCULUS_PORT={env_overrides.get('HOMUNCULUS_PORT')} "
                      f"player={env_overrides.get('MC_PLAYER_NAME')}]")
        print(f"\n{'=' * 78}\n[suite] running test '{name}'{target} :: "
              f"{' '.join(cmd)}\n"
              f"[suite] {spec.get('summary', '')} "
              f"(iters={iters}, threshold={threshold:.2f}, "
              f"timeout={effective_timeout}s)\n{'=' * 78}", flush=True)

    # Clear stale JSONL so we judge a fresh run.
    if out_path.exists():
        out_path.unlink()

    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(cmd, timeout=effective_timeout,
                              stdout=None, stderr=None, env=env)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = -1
    wall_s = round(time.monotonic() - t0, 1)

    if timed_out:
        passed = False
        summary = f"timeout after {effective_timeout}s"
        details = {"iters": 0, "passed": 0, "rate": 0.0, "threshold": threshold}
    else:
        passed, summary, details = _judge_pass_rate(out_path, threshold)
        # A non-zero exit code from the test process is informational but the
        # JSONL is the source of truth. Test subprocesses already self-judge
        # by pass-rate; sometimes the test exits 1 (rate < its own default
        # threshold) but the suite-level threshold is met or vice versa.

    if verbose:
        verdict = "PASS" if passed else "FAIL"
        print(f"\n[suite] '{name}' {verdict} in {wall_s}s — {summary}", flush=True)

    return {
        "name": name,
        "passed": passed,
        "wall_s": wall_s,
        "returncode": returncode,
        "timed_out": timed_out,
        "summary": summary,
        "details": details,
        "jsonl_path": str(out_path),
    }


def _launch_one(
    spec: dict, agent_n: int, phase: str, *,
    iters: int, threshold: float, coord_managed: bool, verbose: bool,
) -> dict:
    """Spawn one subprocess for `spec` against `agent_n`. Returns a record."""
    name = spec["name"]
    cmd, out_path = _resolve_cmd(spec, iters, concurrent=True, agent_n=agent_n)
    env = {**os.environ, **_agent_env(agent_n)}
    if coord_managed:
        env["SUITE_COORDINATOR_MANAGES_DIFFICULTY"] = "1"
    log_path = Path(f"results/suite-{name}-agent{agent_n}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    log_fh = log_path.open("w")
    timeout_s = spec.get("timeout_s", 300) * max(1, iters)
    if verbose:
        print(f"[suite][phase={phase}] launching '{name}' on agent{agent_n} "
              f"({_base_for_agent(agent_n)}) -> log {log_path} "
              f"(timeout={timeout_s}s, coord_managed={coord_managed})",
              flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=log_fh,
                         stderr=subprocess.STDOUT)
    return {
        "name": name, "spec": spec, "popen": p, "log_fh": log_fh,
        "log_path": log_path, "out_path": out_path,
        "agent_n": agent_n, "iters": iters, "threshold": threshold,
        "t0": time.monotonic(), "timeout_s": timeout_s,
        "wall_s": None, "timed_out": False, "returncode": None,
        "phase": phase,
    }


def _launch_phase(phase: str, phase_specs: list[dict],
                  iters_for: dict[str, int], thresholds: dict[str, float],
                  *, verbose: bool = True) -> list[dict]:
    """Per-agent serial, cross-agent parallel.

    Each spec's `concurrent_agents` is unrolled into per-agent work items.
    Within a single agent, items run sequentially (one MC client can't host
    two tests at once — Baritone session lock, single player). Across agents,
    items advance in parallel. The phase completes when every agent's queue
    drains.

    Spec fan-out (e.g. shelter on [0,1,2]) emits one queue entry per agent.
    Multiple distinct specs targeting the same agent (e.g. mine_iron and
    travel both on agent2) get serialized in declaration order.
    """
    coord_managed = (phase in PHASE_DIFFICULTY)

    # Build per-agent queues, preserving spec declaration order.
    agent_queues: dict[int, list[dict]] = {}
    for spec in phase_specs:
        agents = spec.get("concurrent_agents") or []
        if not agents:
            print(f"[suite] '{spec['name']}' has no concurrent_agents — "
                  "skipping in phased concurrent mode", flush=True)
            continue
        for agent_n in agents:
            agent_queues.setdefault(agent_n, []).append(spec)

    running: dict[int, dict] = {}  # agent_n -> active proc record
    all_records: list[dict] = []

    def _kick_next(agent_n: int) -> None:
        queue = agent_queues.get(agent_n, [])
        if not queue:
            return
        spec = queue.pop(0)
        rec = _launch_one(
            spec, agent_n, phase,
            iters=iters_for[spec["name"]],
            threshold=thresholds[spec["name"]],
            coord_managed=coord_managed,
            verbose=verbose,
        )
        running[agent_n] = rec
        all_records.append(rec)

    # Seed: launch the head of each agent's queue.
    for agent_n in list(agent_queues.keys()):
        _kick_next(agent_n)

    # Poll: when a proc completes (or times out), launch the next on its agent.
    while running:
        for agent_n in list(running.keys()):
            p = running[agent_n]
            rc = p["popen"].poll()
            elapsed = time.monotonic() - p["t0"]
            done = False
            if rc is not None:
                p["returncode"] = rc
                p["wall_s"] = round(elapsed, 1)
                done = True
                if verbose:
                    print(f"[suite][phase={phase}] '{p['name']}'@agent{agent_n} "
                          f"exited rc={rc} after {p['wall_s']}s", flush=True)
            elif elapsed > p["timeout_s"]:
                p["popen"].kill()
                p["popen"].wait(timeout=5)
                p["returncode"] = -1
                p["wall_s"] = round(elapsed, 1)
                p["timed_out"] = True
                done = True
                if verbose:
                    print(f"[suite][phase={phase}] '{p['name']}'@agent{agent_n} "
                          f"TIMED OUT after {p['wall_s']}s — killed", flush=True)
            if done:
                p["log_fh"].close()
                del running[agent_n]
                _kick_next(agent_n)
        time.sleep(1.0)

    return all_records


def run_concurrent_phased(specs: list[dict], *, iters_for: dict[str, int],
                          thresholds: dict[str, float], verbose: bool = True
                          ) -> list[dict]:
    """Phase-grouped concurrent runner.

    Tests are grouped by `world_state` and phases run sequentially in
    PHASE_ORDER. Inside each phase, tests run in parallel and each spec
    fans out across `concurrent_agents`. Between phases, the coordinator
    cleans up entities + resets difficulty.

    Per-spec results are combined across all agents using _judge_combined.
    """
    # Group specs by phase.
    groups: dict[str, list[dict]] = {p: [] for p in PHASE_ORDER}
    for spec in specs:
        phase = spec.get("world_state", "peaceful")
        if phase not in groups:
            print(f"[suite] '{spec['name']}' has unknown world_state "
                  f"{phase!r} — defaulting to peaceful", flush=True)
            phase = "peaceful"
        groups[phase].append(spec)

    all_procs: list[dict] = []
    for phase in PHASE_ORDER:
        phase_specs = groups[phase]
        if not phase_specs:
            continue
        if verbose:
            print(f"\n{'=' * 78}", flush=True)
            print(f"[suite] PHASE '{phase}' — "
                  f"{[s['name'] for s in phase_specs]}", flush=True)
            print(f"{'=' * 78}", flush=True)
        _coordinator_cleanup_phase()
        _coordinator_set_phase_state(phase)
        # Brief settle so difficulty change propagates before subprocesses
        # start their own setup. Empirically 1s is enough.
        time.sleep(1.0)
        phase_procs = _launch_phase(phase, phase_specs, iters_for, thresholds,
                                    verbose=verbose)
        all_procs.extend(phase_procs)

    # Final cleanup so the world is in a known state for human inspection
    # / next run. peaceful + entity kill.
    _coordinator_cleanup_phase()

    # Aggregate per-spec. Multiple sub-process rows per spec when fanned out.
    by_spec: dict[str, list[dict]] = {}
    for p in all_procs:
        by_spec.setdefault(p["name"], []).append(p)

    results: list[dict] = []
    for name, sub_procs in by_spec.items():
        # Per-spec wall = max(sub_walls) since they ran in parallel within phase.
        wall_s = max((p["wall_s"] or 0.0) for p in sub_procs)
        any_timeout = any(p["timed_out"] for p in sub_procs)
        threshold = sub_procs[0]["threshold"]
        if any_timeout:
            passed = False
            summary = (f"timeout in {sum(1 for p in sub_procs if p['timed_out'])}"
                       f"/{len(sub_procs)} agent(s)")
            details = {"iters": 0, "passed": 0, "rate": 0.0,
                       "threshold": threshold,
                       "agent_count": len(sub_procs)}
        else:
            passed, summary, details, _ = _judge_combined(name, threshold)
        if verbose:
            verdict = "PASS" if passed else "FAIL"
            agents_used = sorted(p["agent_n"] for p in sub_procs)
            print(f"[suite][phased] '{name}' {verdict} on agents={agents_used} — "
                  f"{summary}", flush=True)
        results.append({
            "name": name,
            "passed": passed,
            "wall_s": wall_s,
            "returncode": [p["returncode"] for p in sub_procs],
            "timed_out": any_timeout,
            "summary": summary,
            "details": details,
            "jsonl_paths": [str(p["out_path"]) for p in sub_procs],
            "log_paths": [str(p["log_path"]) for p in sub_procs],
            "agents": [p["agent_n"] for p in sub_procs],
            "phase": sub_procs[0]["phase"],
        })
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these tests by name (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="list available tests and exit")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--iters", type=int, default=None,
                    help="override per-test iters (multi-iter estimates failure rate)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override per-test pass-rate threshold (uniform across tests)")
    ap.add_argument("--concurrent", action="store_true",
                    help="phase-grouped concurrent runner: tests group by "
                         "world_state (peaceful / non_peaceful / mixed), phases "
                         "run sequentially, tests within a phase fan out across "
                         "their concurrent_agents list. Coordinator owns "
                         "difficulty for non-mixed phases; mixed-phase tests "
                         "self-manage. Per-agent JSONLs at "
                         "results/suite-<name>-agent<N>.jsonl.")
    args = ap.parse_args(argv)

    if args.list:
        for spec in TESTS:
            print(f"  {spec['name']:<12} (iters={spec['iters']:>2}, "
                  f"threshold={spec['threshold']:.2f}) — {spec.get('summary', '')}")
        return 0

    selected = TESTS if args.only is None else [
        s for s in TESTS if s["name"] in args.only
    ]
    if not selected:
        print(f"[suite] no tests match {args.only}; available: "
              f"{[s['name'] for s in TESTS]}", file=sys.stderr)
        return 2

    # In concurrent mode, preflight every agent that any selected test wants
    # across all phases. Sequential mode preflights the single canonical client.
    if args.concurrent:
        required_agents: set[int] = set()
        for spec in selected:
            agents = spec.get("concurrent_agents")
            if not agents:
                print(f"[suite] '{spec['name']}' has no concurrent_agents — "
                      "cannot run in --concurrent mode", file=sys.stderr)
                return 2
            required_agents.update(agents)
        for agent_n in sorted(required_agents):
            err = _preflight(_base_for_agent(agent_n))
            if err is not None:
                print(f"[suite] preflight FAIL on agent{agent_n}: {err}",
                      file=sys.stderr)
                return 2
        if args.fail_fast:
            print("[suite] --fail-fast is ignored in --concurrent mode "
                  "(subprocesses launch in phase batches)", flush=True)
    else:
        err = _preflight("http://127.0.0.1:25566")
        if err is not None:
            print(f"[suite] preflight FAIL: {err}", file=sys.stderr)
            return 2

    print(f"[suite] running {len(selected)} test(s): "
          f"{[s['name'] for s in selected]}"
          + (" [CONCURRENT]" if args.concurrent else "")
          + (f" (--iters {args.iters})" if args.iters else "")
          + (f" (--threshold {args.threshold})" if args.threshold else ""),
          flush=True)

    iters_for = {s["name"]: (args.iters if args.iters is not None else s["iters"])
                 for s in selected}
    thresholds = {s["name"]: (args.threshold if args.threshold is not None
                              else s["threshold"]) for s in selected}

    if args.concurrent:
        results = run_concurrent_phased(selected, iters_for=iters_for,
                                        thresholds=thresholds)
    else:
        results = []
        for spec in selected:
            iters = iters_for[spec["name"]]
            threshold = thresholds[spec["name"]]
            r = run_test(spec, iters=iters, threshold=threshold)
            results.append(r)
            if args.fail_fast and not r["passed"]:
                print("[suite] --fail-fast: stopping after first failure",
                      flush=True)
                break

    print(f"\n{'=' * 78}\n[suite] SUMMARY\n{'=' * 78}", flush=True)
    width_name = max(len(r["name"]) for r in results)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        d = r["details"]
        rate_str = (f"{d.get('passed', 0)}/{d.get('iters', 0)} "
                    f"({d.get('rate', 0.0):.2f})") if d.get("iters") else "n/a"
        print(f"  {tag}  {r['name']:<{width_name}}  {r['wall_s']:>6.1f}s  "
              f"rate={rate_str}  {r['summary']}", flush=True)
    total_wall = sum(r["wall_s"] for r in results)
    failed = [r for r in results if not r["passed"]]
    print(f"\n[suite] {len(results) - len(failed)}/{len(results)} tests passed "
          f"({total_wall:.1f}s total)", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
