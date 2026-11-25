"""Seed a throw-away data directory for the browser end-to-end tests.

    python -m tests.e2e_seed /path/to/empty/dir

Writes the same small hand-built store the unit tests use (symbol TST, eight sessions of
one-minute bars with two 3% down days, and option trade prints for one of the qualifying
sessions), plus the V4 fixtures described in `seed`. It is a TEST FIXTURE with known answers, not market data: it is only ever
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
    # V4 fixtures: LNG has 1,500 daily sessions in which a fall of 2% or more is followed by a session that closes
    # about 1.2% above its open (so a hypothesis plan has something real to find), plus a second symbol and a
    # volatility index; and a small vendor-format archive for the bulk importer.
    from tests.test_v4_research import rich_store
    rich_store(root, sessions=1500, effect=1.2, intraday=0, sym="LNG")
    import zipfile
    from datetime import date, timedelta

    import polars as pl

    from strata.core import calendar as cal
    fx = root / "fixtures"
    fx.mkdir(exist_ok=True)
    rows = [{"underlying": "LNG", "optionroot": "x", "type": "call" if r == "C" else "put", "expiration": (d + timedelta(days=7)).strftime("%m/%d/%Y"), "quotedate": d.strftime("%m/%d/%Y"), "strike": float(k), "bid": 1.0, "ask": 1.2}
            for d in cal.trading_days(date(2024, 1, 2), date(2024, 1, 12)) for k in (95, 100, 105) for r in ("C", "P")]
    pl.DataFrame(rows).write_csv(fx / "quotes.csv")
    with zipfile.ZipFile(fx / "vendor_quotes.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(fx / "quotes.csv", "LNG_2024_01.csv")
    (fx / "quotes.csv").unlink()
    from strata.data.store import MarketStore
    st = MarketStore(root / "market")
    print(f"seeded {root}: symbols {sorted(st.stored_symbols())}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    seed(Path(sys.argv[1]))
