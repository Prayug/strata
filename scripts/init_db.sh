#!/usr/bin/env bash
# Create (or migrate) the SQLite application database. Add --demo to also generate
# the SYNTHETIC demonstration dataset if it is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
backend/.venv/bin/strata init "$@"
