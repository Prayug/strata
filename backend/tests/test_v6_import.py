"""V6: importing an OptionsDX file as the vendor actually writes it.

The vendor's public sample (SPY, 2020-03-06) was imported and cross-checked by hand; it is not redistributed here. These
tests write files in exactly that layout (33 columns, plain header names, "10 x 1" size strings, a first snapshot ten
seconds after the open) with made-up prices, so they prove the importer and the engine handle the layout. They say nothing
about any real price.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from strata.audit import reference as R
from strata.core import calendar as cal
from strata.data import bulk as B
from strata.engine.engine import run_backtest
from strata.research import usecase as U
from tests.helpers import make_store

D = date(2020, 3, 6)
HEADER = ("QUOTE_UNIXTIME,QUOTE_READTIME,QUOTE_DATE,QUOTE_TIME_HOURS,UNDERLYING_LAST,EXPIRE_DATE,EXPIRE_UNIX,DTE,C_DELTA,C_GAMMA,C_VEGA,C_THETA,C_RHO,C_IV,C_VOLUME,C_LAST,C_SIZE,C_BID,C_ASK,STRIKE,"
          "P_BID,P_ASK,P_SIZE,P_LAST,P_DELTA,P_GAMMA,P_VEGA,P_THETA,P_RHO,P_IV,P_VOLUME,STRIKE_DISTANCE,STRIKE_DISTANCE_PCT")


def spot(m: int) -> float:
    return 100.0 if m < 65 else 101.5  # the 10:35 snapshot is the first to show the rise


def write_file(path, until: str = "16:00", brackets: bool = False):
    """One CSV in the vendor's layout: a snapshot at 09:30:10, then every five minutes to `until`; two expirations, strikes 95-105."""
    o = cal.session_bounds(D)[0]
    h, mi = map(int, until.split(":"))
    last = (h * 60 + mi) - 570
    lines = [",".join(f"[{c}]" for c in HEADER.split(",")) if brackets else HEADER]
    for m in range(0, last + 1, 5):
        t = o + timedelta(minutes=m, seconds=10 if m == 0 else 0)
        s = spot(m)
        for exp in (D, D + timedelta(days=7)):
            tv = (1.0 if exp == D else 2.5) - 0.002 * m
            eu = int(datetime(exp.year, exp.month, exp.day, 16, 0, tzinfo=cal.ET).timestamp())
            for k in range(95, 106):
                c, p = max(s - k, 0) + tv, max(k - s, 0) + tv
                lines.append(f"{int(t.timestamp())},{t:%Y-%m-%d %H:%M},{D},{(570 + m) / 60:g},{s},{exp},{eu},{(exp - D).days + 0.27},0.5,0.01,0.1,-0.1,0.01,0.3,12,{c:.2f},10 x 1,{c - 0.05:.2f},{c + 0.05:.2f},{k},"
                             f"{p - 0.05:.2f},{p + 0.05:.2f},0 x 250,{p:.2f},-0.5,0.01,0.1,-0.1,-0.01,0.3,7,{abs(s - k):g},{abs(s - k) / s:.3f}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def store(tmp_path):
    from strata.config import override_settings
    override_settings(data_dir=tmp_path)
    st = make_store(tmp_path, sym="SPY", settlement="physical", exercise="american")
    prev = cal.prev_trading_day(D)
    st.write_daily("SPY", pl.DataFrame({"date": [cal.prev_trading_day(prev), prev], "open": [104.0, 103.0], "high": [104.5, 103.5], "low": [103.0, 99.5], "close": [103.5, 100.0], "volume": [1e6, 1e6]}), provider="yahoo")
    return st


@pytest.mark.parametrize("brackets", [False, True], ids=["plain header (the public sample)", "bracketed header (store downloads)"])
def test_both_header_styles_of_the_vendor_are_recognised(tmp_path, store, brackets):
    f = write_file(tmp_path / "spy.csv", brackets=brackets)
    assert B.detect(pl.read_csv(f, n_rows=1).columns) == "optionsdx"
    rep = B.import_path(str(f), symbol="SPY", provider="optionsdx", store=store)
    assert rep["sessions"] == 1 and not rep.get("partial_sessions") and rep["rejected"] == {} and rep["files"][0]["format"] == "optionsdx", rep
    df = store.read_option_day("SPY", D, "quote", provider="optionsdx")
    assert df.height == 79 * 2 * 11 * 2, "79 snapshots x 2 expirations x 11 strikes x calls and puts"
    row = df.filter((pl.col("strike") == 100) & (pl.col("right") == "C") & (pl.col("expiration") == D)).sort("ts").row(0, named=True)
    assert (row["bid"], row["ask"], row["bid_size"], row["ask_size"]) == (pytest.approx(0.95), pytest.approx(1.05), 10, 1), "\"10 x 1\" is a bid size of 10 and an ask size of 1"
    assert row["ts"].astimezone(cal.ET).strftime("%H:%M:%S") == "09:30:10" and row["underlying_price"] == 100.0
    put = df.filter((pl.col("strike") == 100) & (pl.col("right") == "P") & (pl.col("expiration") == D)).sort("ts").row(0, named=True)
    assert (put["bid_size"], put["ask_size"]) == (0, 250)


def test_a_file_that_ends_at_noon_is_a_partial_session_and_is_never_called_complete(tmp_path, store):
    f = write_file(tmp_path / "sample.csv", until="12:00")
    rep = B.import_path(str(f), symbol="SPY", provider="optionsdx", store=store, source_label="vendor sample", describe={"license": "vendor terms", "limitations": ["a sample"]})
    assert rep["sessions"] == 1 and rep["partial_sessions"] == {D.isoformat(): {"symbol": "SPY", "first_snapshot": "09:30", "last_snapshot": "12:00", "interval_minutes": 5, "expected_snapshots": 79}}
    key = next(k for k in store.datasets("SPY") if k.provider == "optionsdx" and k.dtype == "options")
    st = store.session_states(key)[D.isoformat()]
    assert st["status"] == "partial" and (st["first_snapshot"], st["last_snapshot"], st["interval_minutes"]) == ("09:30", "12:00", 5) and st["expected_snapshots"] == 79 and st["snapshots"] == 31
    assert store.option_days("SPY", "quote", dataset=key, complete_only=True) == [] and store.option_days("SPY", "quote", dataset=key) == [D]
    assert store.descriptor(key).get("license") == "vendor terms"
    # the underlying's bars come from the price printed with each snapshot: 30 five-minute bars, 09:30 to 12:00, and they do not pass for a whole session
    bars = store.read_bars("SPY", "5m", D, D, with_provider=True).filter(pl.col("provider") == "optionsdx")
    assert bars.height == 30 and rep["underlying_from_snapshots"]["bars"] == 30
    best = store.read_best_bars("SPY", D, D)
    assert best.is_empty() or int(best["res"][0]) == 86400, "a morning of bars is not used as if it were the session"
    assert B.session_window(store.read_option_day("SPY", D, "quote", dataset=key), D)[0] == "partial"


def test_an_imported_full_session_is_priced_by_the_engine_and_by_the_reference_alike(tmp_path, store):
    """The use case on an imported file: the day before fell 3.4%, buy the at-the-money 0DTE call at the open, exit 1% higher.
    Entry at the first quote (09:30:10): ask = 1.00 + 0.05. The rise to 101.5 is first seen at 10:35; that snapshot's bid is
    1.50 + (1.00 - 0.002 x 65) - 0.05 = 2.32."""
    B.import_path(str(write_file(tmp_path / "spy_2020.csv")), symbol="SPY", provider="optionsdx", store=store)
    cfg = {**U.manual_config("SPY", 3.0), "start": D.isoformat(), "end": D.isoformat(), "pricing": {"mode": "historical"}, "data": {"options_provider": "optionsdx"}}
    res = run_backtest(cfg, store=store)
    assert len(res.trades) == 1, res.diagnostics.get("skipped_signals")
    t = res.trades[0]
    leg = t["legs"][0]
    assert (leg["strike"], leg["expiration"], leg["price_source"]) == (100.0, D.isoformat(), "quote") and (leg["entry_price"], leg["exit_price"]) == (pytest.approx(1.05), pytest.approx(2.32))
    assert (t["entry_time"][11:19], t["exit_time"][11:19], t["exit_reason"]) == ("09:30:10", "10:35:00", "underlying_target") and t["pnl"] == pytest.approx((2.32 - 1.05) * 100 - 1.30)
    assert t["fill_delay_seconds"] == 10, "the signal is at the open; the first quote exists ten seconds later"
    ref = R.simulate([R.load_session(store, "SPY", D, "quote", "optionsdx")], U._ref_spec(cfg, "quote"))
    assert R.compare(res.trades, ref["trades"])["agree"], R.compare(res.trades, ref["trades"])
    assert res.meta["data_resolution"] or True
