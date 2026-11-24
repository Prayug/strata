#!/bin/sh
set -e
# Create the SQLite schema and, on first start, the SYNTHETIC demo dataset (~1 minute).
if [ "${STRATA_SKIP_DEMO:-0}" = "1" ]; then strata init; else strata init --demo; fi
exec "$@"
