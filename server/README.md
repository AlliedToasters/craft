# server/

The Minecraft-side half of craft. Ships one tiny stdlib-only HTTP wrapper
(`mc_api.py`) that lets the agent box poke the MC server console over a
loopback or LAN URL — `POST /cmd` to inject a console command, `GET /log`
to tail `logs/latest.log`.

Everything in this directory runs on the **Minecraft server machine** (which
may or may not be the same box as the agent). The rest of `craft/` doesn't
need to live here.

## What craft expects from the server

1. A Minecraft 1.21.4 server (Purpur recommended; Paper / vanilla should work).
2. The server runs **inside a named `screen` session** with its console
   attached — `mc_api` injects commands via `screen -X stuff`, so the MC
   console has to actually be reading from screen's stdin.
3. `mc_api.py` running alongside the server, with `logs/latest.log`
   reachable from its working directory.

That's it. No mod, no plugin — every command in the wrapper is a vanilla
server command (`tp`, `effect`, `gamemode`, etc.), so it works on any
flavor of the server that ships a normal console.

## Install

On the MC server box:

```bash
# 1. Install prerequisites
sudo apt install screen openjdk-21-jre-headless python3   # or your distro's equivalent

# 2. Grab a server jar (Purpur shown; substitute Paper / vanilla as you like)
mkdir -p ~/mc-server && cd ~/mc-server
curl -L -o server.jar https://api.purpurmc.org/v2/purpur/1.21.4/latest/download

# 3. First-run EULA dance
java -Xms2G -Xmx4G -jar server.jar nogui
# → exits, writes eula.txt; flip eula=true:
sed -i 's/eula=false/eula=true/' eula.txt

# 4. Copy mc_api.py here from this repo
#    (the wrapper resolves logs/ relative to CWD, so it needs to run
#    from the server directory — or set MC_LOG_PATH explicitly)
cp /path/to/craft/server/mc_api.py .
```

## Run

Two long-running processes: the MC server inside a named screen, and the
HTTP wrapper alongside it.

```bash
# Start the MC server in a detached screen session named "server"
screen -dmS server java -Xms2G -Xmx4G -jar server.jar nogui

# (attach with: screen -r server   — Ctrl-A D to detach)

# Start the HTTP bridge in another screen (or systemd unit, or nohup)
screen -dmS mc_api python3 mc_api.py
```

Smoke test:

```bash
curl http://localhost:4747/log?n=5
curl -X POST http://localhost:4747/cmd -H 'Content-Type: application/json' \
     -d '{"cmd":"say hello from craft"}'
```

You should see "hello from craft" land in the MC console.

## Config (env vars)

All optional — defaults match a single-box setup with the server running
in screen session `server`:

| Var | Default | Notes |
|-----|---------|-------|
| `MC_SCREEN_SESSION` | `server` | name passed to `screen -S` |
| `MC_LOG_PATH` | `logs/latest.log` | resolved against CWD |
| `MC_API_PORT` | `4747` | HTTP listen port |
| `MC_API_HOST` | `0.0.0.0` | bind address — use `127.0.0.1` if you don't need LAN access |

Example: a non-default screen name and loopback-only bind:

```bash
MC_SCREEN_SESSION=mc-survival MC_API_HOST=127.0.0.1 python3 mc_api.py
```

On the **agent box**, point craft at the wrapper via `MC_SERVER_CMD_BASE`
(see the top-level README) — defaults to `http://127.0.0.1:4747` for a
single-box install. For a separate MC server box, set e.g.
`MC_SERVER_CMD_BASE=http://mc-host.local:4747`.

## Server settings worth tuning

Drop these into `server.properties` before first launch:

- `online-mode=false` — required if you use offline accounts for the agent
  fleet (`agent0`..`agent9`).
- `connection-throttle=0` — Purpur default of 4000ms rejects rapid same-IP
  rejoins, which kills concurrent rollouts. (Public servers want the
  throttle on; private rigs don't.)
- `spawn-protection=0` — otherwise the agent can't break blocks near spawn.
- `view-distance` — agent doesn't need much. 8 is plenty.

## Why a screen pipe and not RCON?

RCON works for command injection, but it doesn't expose the console log,
which craft's smoke tests + observability tools depend on. The screen +
log-tail combo is one moving part instead of two, and the wrapper has zero
non-stdlib dependencies.

## Troubleshooting

- **`curl /log` returns `{"lines": []}`** → the wrapper can't find
  `logs/latest.log`. Either run it from the MC server's working directory
  or set `MC_LOG_PATH` to the absolute path.
- **`curl /cmd` 500s with `No screen session found`** → the MC server isn't
  running inside the expected screen session, or screen renamed it
  (`screen -ls` to check).
- **Commands don't take effect** → `screen -r server` to attach, type
  `/say test` directly. If the prompt isn't accepting keystrokes, the
  console isn't actually attached to screen's stdin (common with
  `start.sh` wrappers that pipe through `tee` or similar).
