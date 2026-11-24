#!/usr/bin/env bash
# One-shot local install: Python venv + backend, frontend deps, database and demo data.
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
backend/.venv/bin/strata init --demo
echo
echo "Setup complete. Start everything with:  scripts/dev.sh"
