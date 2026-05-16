#!/usr/bin/env bash
# Minimal installer for craft. Idempotent — re-running is safe.
#
#   ./setup.sh
#
# What it does:
#   1. Verifies Python 3.10+.
#   2. Creates .venv/ if missing.
#   3. Installs craft (editable) and its dependencies.
#   4. Copies .env.example to .env if .env doesn't already exist.
#
# Doesn't do:
#   - Install Ollama, Java, PrismLauncher, or the MC server. Those live on
#     different machines in the general case; see README.md + server/README.md.

set -euo pipefail
cd "$(dirname "$0")"

# ─── 1. Python version check ──────────────────────────────────────────────
need_py() {
  echo "Need Python 3.10 or newer. Found: $(python3 --version 2>&1 || echo none)" >&2
  exit 1
}
command -v python3 >/dev/null || need_py
python3 - <<'PY' || need_py
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

# ─── 2. venv ──────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
  echo "Creating .venv/ ..."
  python3 -m venv .venv
fi

# ─── 3. install ───────────────────────────────────────────────────────────
echo "Installing craft (editable) ..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .

# ─── 4. .env scaffold ─────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "Created .env from .env.example."
  echo "Edit it now — fill in ANTHROPIC_API_KEY (if using Claude) and"
  echo "MC_PLAYER_NAME / MC_SERVER_CMD_BASE / HOMUNCULUS_PORT to match"
  echo "your setup. See README.md for the field-by-field walkthrough."
else
  echo ".env already exists — leaving it alone."
fi

echo
echo "Done. Activate the venv with:  source .venv/bin/activate"
echo "Next: read README.md for the first-rollout walkthrough."
