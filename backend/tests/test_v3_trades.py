"""Trade-bar option execution: only prints that exist at the execution time, never a
quote, never a stale or future price - verified on hand-built data."""
from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.store import MarketStore, SymbolMeta
from strata.engine.engine import DataUnavailable, run_backtest
from strata.engine.market import UnderlyingSeries, expand_trade_bars
from tests.helpers import et_epoch, write_minutes

D = date(2024, 3, 6)
OCC = "SPY240306C00500000"


def trade_bars(d: date, prints: list[tuple], occ=OCC, exp=None, strike=500.0, right="C") -> pl.DataFrame:
    """prints: (HH:MM bar start, price, volume)"""
    rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp or d, "strike": strike, "right": right,
             "open": px, "high": px, "low": px, "close": px, "volume": v} for t, px, v in prints]
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))


def make(tmp_path, prints, price=lambda d, i: 500.0, **kw) -> MarketStore:
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [cal.prev_trading_day(D), D], price)
    bars = trade_bars(D, prints, **kw)
    st.write_option_bars("SPY", "testfeed", bars, [{"date": D, "occ": bars["occ"][0], "expiration": bars["expiration"][0], "strike": bars["strike"][0], "right": bars["right"][0],
                                                    "bars": bars.height, "status": "ok", "pages": 1}])
    return st


def strategy(**exits) -> dict:
    return {"name": "0DTE call", "symbol": "SPY",
            "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}, **exits}}


def run(st, strat=None, **ex):
    return run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [strat or strategy()], "pricing": {"mode": "trades"},
                         "execution": {"signal_interval_minutes": 1, **ex}}, store=st)


PRINTS = [("09:35", 2.00, 5), ("10:07", 2.50, 8), ("10:20", 3.00, 4), ("15:20", 1.00, 2)]


def test_a_stale_print_is_refused_and_the_signal_is_reported_as_unexecuted(tmp_path):
    """At 10:00 the only print is 24 minutes old. The order waits 5 minutes, sees no fresh
    print and expires - it is not filled at 2.00 and not at the 10:07 print."""
    res = run(make(tmp_path, PRINTS))
    assert res.trades == []
    ex = res.meta["executability"]
    assert ex["signals"] == 1 and ex["executed"] == 0 and ex["executed_pct"] == 0.0 and ex["not_executed_by_reason"] == {"no_recent_trade": 1}
    sk = res.diagnostics["skipped_signals"]
    assert len(sk) == 1 and sk[0]["signal_t"] == et_epoch(D, "10:00") and "no recent trade" in sk[0]["reason"].replace("no usable trade print", "no recent trade")


def test_fill_uses_the_first_fresh_print_once_its_bar_has_completed(tmp_path):
    """With a 10-minute fill window the order fills from the 10:07 bar - but only at 10:08,
    when that bar has ended. The later 3.00 print is never visible."""
    res = run(make(tmp_path, PRINTS), max_fill_delay_minutes=10, trade_bar_slippage_pct=2.0, commission_per_contract=0.0)
    assert len(res.trades) == 1
    t = res.trades[0]
    leg = t["legs"][0]
    assert t["signal_ts"] == et_epoch(D, "10:00") and t["entry_ts"] == et_epoch(D, "10:08") and t["fill_delay_seconds"] == 480
    assert leg["price_source"] == "trade_print" and leg["entry_print"] == pytest.approx(2.50)
    assert leg["entry_price"] == pytest.approx(2.50 * 1.02)  # the print plus the stated slippage, against the buyer
    assert leg["entry_bid"] is None and leg["entry_ask"] is None  # no bid/ask exists, so none is reported
    assert t["meta"]["entry_print_age_seconds"] == 0 and "trade_based_fill" in t["flags"]
    assert res.meta["executability"]["executed_pct"] == 100.0 and res.meta["pricing"] == "trades"


def test_exit_without_a_recent_print_is_deferred_not_priced_at_a_stale_trade(tmp_path):
    """15:30 time exit: the last print (1.00) is nine minutes old. The exit is deferred; no
    print follows, so the 0DTE contract settles at intrinsic value (0) at the close."""
    res = run(make(tmp_path, PRINTS), max_fill_delay_minutes=10, commission_per_contract=0.0)
    t = res.trades[0]
    leg = t["legs"][0]
    assert "deferred_exit" in t["flags"] and leg["exit_reason"] == "expired_worthless" and leg["exit_price"] == 0.0
    assert t["pnl"] == pytest.approx(-250.0)  # paid 2.50, expired at the money: never sold at the stale 1.00
    assert res.meta["executability"]["deferred_exits"] >= 1


def test_exit_fills_at_a_fresh_print(tmp_path):
    prints = [("09:59", 2.00, 5), ("15:29", 2.60, 3)]
    res = run(make(tmp_path, prints), commission_per_contract=0.0)
    t = res.trades[0]
    assert t["entry_ts"] == et_epoch(D, "10:00") and t["legs"][0]["entry_print"] == pytest.approx(2.00)
    assert t["exit_ts"] == et_epoch(D, "15:30") and t["legs"][0]["exit_print"] == pytest.approx(2.60) and t["exit_reason"] == "time_exit"
    assert t["pnl"] == pytest.approx(60.0)


def test_premium_stop_is_not_triggered_by_a_stale_mark(tmp_path):
    """The option's last print before a long silence is far below the stop. While no fresh
    print exists the stop cannot be evaluated; the stale mark is counted, not acted on."""
    prints = [("09:59", 2.00, 5), ("10:30", 2.10, 1), ("14:00", 0.40, 1), ("15:29", 0.50, 1)]
    st = make(tmp_path, prints)
    res = run(st, strategy(stop_loss={"enabled": True, "mode": "pct", "value": 50}), commission_per_contract=0.0)
    t = res.trades[0]
    # the 0.40 print (bar ends 14:01) is fresh at 14:01: the stop fires there, at an observed price
    assert t["exit_reason"] == "stop_loss" and t["exit_ts"] == et_epoch(D, "14:01") and t["legs"][0]["exit_print"] == pytest.approx(0.40)
    assert res.diagnostics["stale_marks"] >= 1 and "stale_quotes" in t["flags"]


def test_contract_missing_from_the_stored_data_is_reported_as_such(tmp_path):
    exp = date(2024, 4, 5)
    st = make(tmp_path, [("09:59", 9.0, 5)], occ="SPY240405C00500000", exp=exp)
    res = run(st)
    assert res.trades == [] and res.meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert "not in the stored trade data" in res.diagnostics["skipped_signals"][0]["reason"]


def test_session_without_trade_data_is_counted_separately_from_no_signal(tmp_path):
    st = make(tmp_path, [("09:59", 2.0, 5)])
    prev = cal.prev_trading_day(D)
    res = run_backtest({"start": prev.isoformat(), "end": D.isoformat(), "strategies": [strategy()], "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1}}, store=st)
    assert res.diagnostics["no_option_data"] == {"SPY": [prev.isoformat()]} and res.meta["executability"]["sessions_without_option_data"] == {"SPY": 1}
    assert [s["kind"] for s in res.diagnostics["skipped_signals"]] == ["no_option_data"] and len(res.trades) == 1


def test_trade_prints_and_quotes_are_never_substituted_for_each_other(tmp_path):
    st = make(tmp_path, [("09:59", 2.0, 5)])
    with pytest.raises(DataUnavailable, match="not substituted"):
        run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [strategy()], "pricing": {"mode": "historical"}}, store=st)
    ts = cal.session_bounds(D)[1]
    st.write_option_day("SPY", D, pl.DataFrame([{"ts": ts, "expiration": D, "strike": 500.0, "right": "C", "bid": 1.0, "ask": 1.1, "underlying_price": 500.0, "quote_ts": ts}]).with_columns(
        pl.col("ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))), provider="quotefeed", res="EOD")
    res = run(st, commission_per_contract=0.0)
    assert res.meta["datasets_used"]["SPY"]["dataset"] == "testfeed/SPY/options/trade/1m/v1" and res.meta["datasets_used"]["SPY"]["observation"] == "trade prints"
    assert res.trades[0]["legs"][0]["entry_print"] == pytest.approx(2.0)  # the 1.00/1.10 quote was not used
    q = run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [strategy()], "pricing": {"mode": "historical"}}, store=st)
    assert q.meta["datasets_used"]["SPY"]["dataset"] == "quotefeed/SPY/options/quote/EOD/v1" and q.meta["datasets_used"]["SPY"]["observation"] == "bid/ask quotes"


def test_trade_chain_is_point_in_time(tmp_path):
    """Rewriting every print after a cut does not change a single row observable before it."""
    st = make(tmp_path, PRINTS)
    ser = UnderlyingSeries(st.read_best_bars("SPY"))
    o = et_epoch(D, "09:30")
    ticks = [o + 60 * i for i in range(391)]
    cut = et_epoch(D, "10:07")
    base, _ = expand_trade_bars(st.read_option_bars("SPY", D), ticks, ser, D)
    bars = st.read_option_bars("SPY", D).with_columns(pl.when(pl.col("ts").dt.epoch("s") + 60 > cut).then(pl.col("close") * 9).otherwise(pl.col("close")).alias("close"))
    pert, _ = expand_trade_bars(bars, ticks, ser, D)
    before = lambda df: df.filter(pl.col("ts").dt.epoch("s") <= cut).sort("ts").select("ts", "last", "quote_ts")
    assert before(base).equals(before(pert)) and before(base).height > 0
    assert not base.equals(pert)
    assert base.filter(pl.col("ts").dt.epoch("s") < et_epoch(D, "09:36")).is_empty()  # the contract does not exist in the chain before its first bar has ended
    assert set(base.columns) >= {"bid", "ask"} and base["bid"].null_count() == base.height and base["ask"].null_count() == base.height  # no quote is ever created
    row = base.filter(pl.col("ts").dt.epoch("s") == et_epoch(D, "10:07")).row(0, named=True)
    assert row["last"] == pytest.approx(2.00) and int(row["quote_ts"].timestamp()) == et_epoch(D, "09:36")  # at 10:07 the 10:07 bar is still forming
