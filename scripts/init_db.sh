#!/usr/bin/env bash
# Create (or migrate) the SQLite application database. Add --fetch to also download
# free real market data (daily + recent intraday bars for SPY, QQQ, IWM and VIX).
set -euo pipefail
cd "$(dirname "$0")/.."
backend/.venv/bin/strata init "$@"
