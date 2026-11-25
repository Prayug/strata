"""Seed a throw-away data directory for the browser end-to-end tests.

    python -m tests.e2e_seed /path/to/empty/dir

Writes the same small hand-built store the unit tests use (symbol TST, eight sessions of
one-minute bars with two 3% down days, and option trade prints for one of the qualifying
sessions). It is a TEST FIXTURE with known answers, not market data: it is only ever
written to the directory given here, and the server that reads it runs with downloads off.
"""
from __future__ import annotations

import sys
from pathlib import Path


def seed(root: Path) -> None:
    root = root.expanduser().resolve()
    if any(p for p in root.rglob("*") if p.is_file()):
        raise SystemExit(f"{root} is not empty; refusing to write test fixtures into it")
    root.mkdir(parents=True, exist_ok=True)
    from strata.api import db
    from strata.api.settings import set_setting
    from strata.config import override_settings
    override_settings(data_dir=root)
    db.init_db()
    from tests.test_v3_agent import trade_store
    st = trade_store(root)
    set_setting("data.default_symbol", "TST")
    print(f"seeded {root}: symbols {sorted(st.stored_symbols())}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    seed(Path(sys.argv[1]))
