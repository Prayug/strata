#!/usr/bin/env bash
# Run the API (port 8765) and the Vite dev server (port 5199) together.
set -euo pipefail
cd "$(dirname "$0")/.."
export STRATA_API=http://127.0.0.1:8765
backend/.venv/bin/strata serve --port 8765 --reload &
API=$!
trap 'kill $API 2>/dev/null' EXIT
cd frontend && (command -v pnpm >/dev/null && pnpm dev --port 5199 || npm run dev -- --port 5199)
