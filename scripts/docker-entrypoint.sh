#!/bin/sh
set -e
# Create the SQLite schema. Market data is never invented: on first start the container
# downloads free real daily/intraday history (a few MB) unless STRATA_SKIP_FETCH=1.
if [ "${STRATA_SKIP_FETCH:-0}" = "1" ]; then
  strata init
else
  strata init --fetch || echo "[strata] market data download failed; use Data > Get free data in the app"
fi
exec "$@"
