#!/usr/bin/env bash
# One-shot local install: Python venv + backend, frontend deps, database, and free REAL
# market data (daily + recent intraday bars for SPY/QQQ/IWM and VIX - a few MB, no account).
# Skip the download with:  STRATA_SKIP_FETCH=1 scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
if command -v uv >/dev/null 2>&1; then
  uv venv -p 3.12 backend/.venv
  uv pip install -p backend/.venv/bin/python -e 'backend[dev]'
else
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -e 'backend[dev]'
fi
if command -v pnpm >/dev/null 2>&1; then (cd frontend && pnpm install); else (cd frontend && npm install); fi
if [ "${STRATA_SKIP_FETCH:-0}" = "1" ]; then
  backend/.venv/bin/strata init
else
  backend/.venv/bin/strata init --fetch || echo "Could not download market data (offline?). Run 'strata fetch' later, or use Data > Get free data in the app."
fi
echo
echo "Setup complete. Start everything with:  scripts/dev.sh"
echo "Optional: real SPY end-of-day option chains (free, ~350 MB):  backend/.venv/bin/strata fetch SPY --options"
