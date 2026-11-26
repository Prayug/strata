#!/bin/sh
# Create the SQLite schema. Market data is never invented: on first start the container
# downloads free real daily/intraday history (a few MB) unless STRATA_SKIP_FETCH=1.
#
# Exit code 3 from `strata init` means another copy of Strata (the app running directly on
# the computer) holds this data directory. A database cannot be shared across the container
# boundary, so nothing is initialised or downloaded and the server starts on standby: it
# explains this in the browser and takes over by itself once the other copy has stopped.
rc=0
if [ "${STRATA_SKIP_FETCH:-0}" = "1" ]; then
  strata init || rc=$?
else
  strata init --fetch || rc=$?
fi
if [ "$rc" = "3" ]; then
  echo "[strata] another copy of Strata holds this data directory: starting on standby"
elif [ "$rc" != "0" ]; then
  if [ "${STRATA_SKIP_FETCH:-0}" = "1" ]; then
    echo "[strata] the database could not be prepared (exit $rc)"
    exit "$rc"
  fi
  echo "[strata] market data download failed; use Data > Get free data in the app"
fi
exec "$@"
