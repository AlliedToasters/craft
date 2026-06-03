# Standing up the N-agent headless fleet

Canonical recipe for running N concurrent rollouts on the headless box. The
executable form is `./fleet.sh`; this file is the narrative + the gotchas that
cost us time. **TL;DR for a clean N=20 stand-up:**

```bash
./fleet.sh cycle 20        # preflight -> down -> deploy -> up -> wait -> status
# then drive the rollouts:
./scripts/bigN20_easy_qwen.sh      # pure-qwen daily driver
# or  ./scripts/bigN20_easy_haiku.sh   # Haiku planner, same substrate
```

If `cycle` reports `in-world: 20/20` you're ready. If it reports stragglers,
see **Troubleshooting**.

## The pieces

Three layers, two of them external to the fleet itself:

1. **MC server** — Purpur/Paper running inside a GNU `screen` session named
   `server` (started elsewhere; survives across fleet cycles).
2. **Console relay** — `gemmacraft/server_1.21.4/mc_api.py` on port **4747**.
   `craft/world.py` routes `set_difficulty` / `set_time` (so `--start-phase`) /
   `set_gamemode` / `give` / `clear` through it by injecting into the `server`
   screen session. **If 4747 is down these silently no-op** — a rollout then
   runs at whatever the world's current time/difficulty is, ignoring your
   flags. Not auto-started; `./fleet.sh preflight` checks it.
   Start: `cd ../gemmacraft/server_1.21.4 && nohup python3 mc_api.py &`
3. **The fleet** — N PrismLauncher clients, agent`i` → homunculus port
   `2557(0+i)`, headless Xvfb display `:200+i`, offline account `agent<i>`.

## The four steps (what `fleet.sh cycle` does)

### 1. `down` — kill everything, clean stale sockets
- Kills `launch_agent.sh` wrappers first (their EXIT traps reap each agent's
  Xvfb + dialog watcher), then the game clients (graceful, then `-9`), then any
  orphaned Xvfb `:2xx`.
- **Then clears `/tmp/pl*` single-instance sockets** — see gotcha #2. Only does
  this once no launcher is running (deleting a live socket would orphan it).
- All `pkill` patterns use the bracket trick (`[o]rg.prismlauncher...`) so they
  never match their own argv.

### 2. `deploy` — rebuild + distribute homunculus
- `cd ../homunculus && ./move_to_instance.sh` (gradle build, then `cp` the jar
  into all ~42 `1.21.4*/minecraft/mods` dirs — both the canonical
  `~/.local/share/PrismLauncher/instances/*` templates **and** the live
  per-agent roots under `~/.local/share/pl-agents/agent*`).
- **Must run with no client live** — `cp`'ing the jar over a *running* instance
  corrupts its lazy class loading (silent `transport_errors`). `down` first.
  `deploy` refuses to run if it sees a live client.

### 3. `up [N]` — launch the fleet
- Loops `launch_agent.sh i` for `i` in `0..N-1`, each **backgrounded** with a
  ~4s stagger, boot logs to `/tmp/fleet-boot/agent<i>.boot.log`.
- `launch_agent.sh` launches one agent and **blocks** (waits on the Prism PID so
  its EXIT trap can reap the Xvfb + watcher). That's why a fleet needs the loop:
  one backgrounded wrapper per agent. This loop is the bit that wasn't written
  down before.
- Per agent it: forces software GL (llvmpipe — keeps the GPU free for Qwen),
  starts a private Xvfb `:200+i`, isolates the launcher with `-d <per-agent
  root>` (single-instance fix, gotcha #2), and passes `-a agent<i>`.

### 4. `status [N]` — verify in-world
- Polls every `2557x/stats`. **Keys on a live HP, not on the port being bound**:
  the homunculus HTTP socket binds early (client still loading), but `/stats`
  only returns health once the player has actually joined. "Port listening but
  no HP" = booted-but-not-joined, the usual half-up state.
- Expect `in-world: N/N`, `gpu=0 %` (render is on CPU).

## Troubleshooting

### Gotcha #1 — straggler stuck in account-refresh (never joins)
**Symptom:** one agent never reaches in-world; its boot log loops
`RefreshSchedule: Background account refresh ... Processing account <Offline>`
every ~20s and no game JVM spawns.
**Cause:** that agent's per-agent root was built *before* the account-stripping
logic, so its `accounts.json` carries many accounts with `activeAccount: None`.
PrismLauncher cycles all of them forever and never settles. (A clean root has
exactly one account + the right `activeAccount`.)
**Fix:** `./fleet.sh fix <i>` — wipes the stale root and relaunches; the rebuild
regenerates a single-account `accounts.json`.

### Gotcha #2 — "Unable to redirect command to already running instance"
**Symptom:** a (re)launched agent prints `QLocalSocket::setServerName() called
while not in unconnected state` / `Unable to redirect command to already running
instance` and exits without booting.
**Cause:** PrismLauncher's single-instance is scoped per app-root and backed by
a `/tmp/pl<hash>` socket. `kill -9` leaks these (66 had accumulated vs 20 live
during this write-up). A leftover socket for that root intercepts the launch.
**Fix:** `down` clears `/tmp/pl*` wholesale once nothing is running. For a single
relaunch that races the kill's socket teardown, just retry once (`fleet.sh fix`
does this implicitly by relaunching after the wipe).

### Gotcha #3 — flags ignored, agent runs into night despite `--start-phase dawn`
The relay (port 4747) is down. `./fleet.sh preflight`. Restart `mc_api.py`.

### Verifying a single agent really launched (vs the staleness trap)
`~/.local/share/pl-agents/agent<i>/instances/1.21.4.agent<i>/minecraft/logs/latest.log`
should show a **fresh** `Setting user: agent<i>`, `OpenGL Renderer: llvmpipe`
(not NVIDIA), and `agent<i> joined the game`. If it names a different user or has
old timestamps you're reading a stale log from a prior generation.

## Capacity (this box: headless Ryzen + RTX 5090, 128 GB, **swap=0**)

- Rendering is forced to CPU (llvmpipe) so the GPU stays reserved for Qwen —
  concurrency is **CPU-bound**, not RAM-bound. At N=20: ~46 GB used, GPU ~0%.
- N≈20 is the validated sweet spot; robust to ~32 but per-agent latency
  balloons and homunculus HTTP timeouts are the strain canary.
- **No swap = hard cliff** — RAM exhaustion won't degrade gracefully. Watch
  `free -g` if you push N up.
