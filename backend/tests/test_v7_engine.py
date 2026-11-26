"""V7: the engine does less work for the same answers, and refuses a configuration it does not understand.

Every shortcut measured on the stored dataset (docs/V7_REPORT.md) is pinned here by the thing it must not change: the trades."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.engine.engine import Backtester, run_backtest
from strata.engine.models import BacktestConfig
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401
from tests.test_v7_experiment import DAYS, PREV, SPOT, grid_store


def cfg(entry_children=None, exits=None, legs=None, **entry) -> dict:
    kids = entry_children or [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]
    return {"start": DAYS[0].isoformat(), "end": DAYS[-1].isoformat(), "pricing": {"mode": "trades"}, "data": {"options_provider": "testfeed"}, "execution": {"signal_interval_minutes": 1},
            "strategies": [{"name": "call", "symbol": "SPY", "legs": legs or [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
                            "entry": {"conditions": {"kind": "group", "op": "and", "children": kids}, **entry},
                            "exits": exits or {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}]}


def rows(trades) -> list[tuple]:
    return [(t["entry_date"], t["entry_time"], t.get("exit_time"), t.get("exit_reason"), round(float(t["pnl"]), 4), tuple((l["strike"], l["expiration"], l["entry_price"], l["exit_price"]) for l in t["legs"])) for t in trades]


# ----------------------------------------------------------------------------- only the sessions where the rule fired
def test_a_run_limited_to_the_sessions_of_its_signals_has_the_same_trades(tmp_settings):
    st = grid_store(tmp_settings)
    full = run_backtest(cfg(), store=st, record_paths=False)
    assert len(full.trades) == 10
    assert rows(run_backtest(cfg(), store=st, record_paths=False, sessions=set(DAYS)).trades) == rows(full.trades)
    some = run_backtest(cfg(), store=st, record_paths=False, sessions={DAYS[2], DAYS[5]})
    assert rows(some.trades) == [r for r in rows(full.trades) if r[0] in (DAYS[2].isoformat(), DAYS[5].isoformat())] and len(some.trades) == 2
    assert run_backtest(cfg(), store=st, record_paths=False, sessions=set()).trades == []


def test_a_position_that_is_open_is_managed_on_sessions_outside_the_hint(tmp_settings):
    st = grid_store(tmp_settings)
    exp = DAYS[4]
    frames, index = [], []
    for n, d in enumerate(DAYS[:5]):  # one contract expiring on the fifth session, printing on each session before it
        occ = S.occ_symbol("SPY", exp, "C", 500.0)
        r = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp, "strike": 500.0, "right": "C", "open": 6.0 + n, "high": 6.0 + n, "low": 6.0 + n, "close": 6.0 + n, "volume": 5}
             for t in ("09:59", "15:28")]
        frames.append(pl.DataFrame(r).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))))
        index.append({"date": d, "occ": occ, "expiration": exp, "strike": 500.0, "right": "C", "bars": 2, "status": "ok", "pages": 1})
    st.write_option_bars("SPY", "testfeed", pl.concat(frames), index)
    c = cfg(legs=[{"id": "lc", "side": "buy", "right": "C", "expiration": exp.isoformat(), "strike": {"method": "fixed", "value": 500.0}}],
            exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": 2}}, max_entries_per_day=1, max_open_positions=1)
    c["end"] = DAYS[4].isoformat()
    full = run_backtest(c, store=st, record_paths=False)
    first = [r for r in rows(full.trades) if r[0] == DAYS[0].isoformat()]
    assert first and first[0][2][:10] == DAYS[2].isoformat()  # bought on the first session, sold two sessions later
    hinted = run_backtest(c, store=st, record_paths=False, sessions={DAYS[0]})
    assert rows(hinted.trades) == first  # the two sessions after it were not in the hint, and the position was still closed on time


# ----------------------------------------------------------------------------- lazy rule evaluation
def wave_store(tmp):
    """The option grid over an underlying that moves: up through the morning on even sessions, down on odd ones."""
    st = grid_store(tmp)
    write_minutes(st, "SPY", [PREV] + DAYS, lambda d, i: SPOT + (1 if d.toordinal() % 2 == 0 else -1) * 2.0 * np.sin(max(i, 0) / 60.0))
    return st


def test_stopping_at_the_first_plain_no_takes_the_same_trades_as_evaluating_every_rule(tmp_settings, monkeypatch):
    from strata.engine.rules import RuleStrategy
    st = wave_store(tmp_settings)
    kids = [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00,10:30,11:00"}}, {"kind": "cond", "type": "expr", "params": {"expr": "price > vwap"}},
            {"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct >= 0.05 and previous_session.return_pct <= 5"}},
            {"kind": "group", "op": "or", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "price > open"}}, {"kind": "cond", "type": "expr", "params": {"expr": "vix < 5"}}]}]
    c = cfg(entry_children=kids, max_entries_per_day=2, max_open_positions=2)
    lazy = run_backtest(c, store=st, record_paths=False)
    assert 0 < len(lazy.trades) < 30  # the rule holds on some sessions and not on others
    monkeypatch.setattr(RuleStrategy, "explain_every_tick", True, raising=False)
    assert rows(run_backtest(c, store=st, record_paths=False).trades) == rows(lazy.trades)


def test_a_rule_that_cannot_be_evaluated_is_still_not_a_rule_that_is_false():
    from strata.engine.conditions import evaluate
    from strata.engine.models import CondNode
    node = CondNode.model_validate({"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "vix < 18"}}, {"kind": "cond", "type": "expr", "params": {"expr": "price > 1"}}]})

    class Ctx:  # prices are known, the VIX is not stored
        def __getattr__(self, name):
            raise AttributeError(name)
    for lazy in (False, True):
        ok, _explain, _na = evaluate(node, Ctx(), lazy=lazy)
        assert ok is not True  # unknown AND anything is never "passed", lazily or not


# ----------------------------------------------------------------------------- the chain from trade bars
def test_the_vectorised_chain_is_the_per_contract_definition_row_for_row():
    from strata.engine.market import expand_trade_bars
    d = DAYS[0]
    o = int(cal.session_bounds(d)[0].timestamp())
    close = int(cal.session_bounds(d)[1].timestamp())
    rng = np.random.default_rng(11)
    bars = []
    for k, right in ((495.0, "C"), (500.0, "C"), (500.0, "P"), (505.0, "P")):
        mins = sorted(set(rng.integers(0, 390, size=14).tolist()) | ({389} if k == 500.0 else set()))
        for m in mins:
            bars.append({"ts": datetime.fromtimestamp(o + 60 * m, tz=timezone.utc), "occ": S.occ_symbol("SPY", d, right, k), "expiration": d, "strike": k, "right": right, "close": round(float(rng.uniform(0.5, 9)), 2), "volume": int(rng.integers(1, 50))})
    df = pl.DataFrame(bars).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    und_end = o + 60 * (np.arange(390) + 1)
    und_c = 500.0 + np.arange(390) * 0.01
    ticks = [o + 60 * m for m in (0, 1, 5, 30, 31, 119, 240, 389, 390)]
    out, stats = expand_trade_bars(df, ticks, SimpleNamespace(end=und_end, c=und_c), d)
    got = {(int(t.timestamp()), e, k, r): (l, v, u, int(q.timestamp())) for t, e, k, r, l, v, u, q in out.select("ts", "expiration", "strike", "right", "last", "volume", "underlying_price", "quote_ts").iter_rows()}
    want = {}
    for (occ,), g in df.sort("ts").group_by(["occ"], maintain_order=True):
        ends = [min(int(t.timestamp()) + 60, close) for t in g["ts"]]
        for T in ticks:
            seen = [j for j, e in enumerate(ends) if e <= T]
            if not seen:
                continue  # it has not printed yet: it is not in the chain
            j = seen[-1]
            ui = int(np.searchsorted(und_end, ends[j], side="right")) - 1
            want[(T, g["expiration"][0], g["strike"][0], g["right"][0])] = (g["close"][j], int(g["volume"][: j + 1].sum()), float(und_c[ui]) if ui >= 0 else float("nan"), ends[j])
    assert got.keys() == want.keys() and len(got) > 20
    for key, w in want.items():
        g_ = got[key]
        assert g_[0] == w[0] and g_[1] == w[1] and g_[3] == w[3] and (g_[2] == w[2] or (np.isnan(g_[2]) and np.isnan(w[2]))), key
    assert all(q <= t for (t, *_), (_l, _v, _u, q) in got.items())  # nothing is priced from a print that had not happened
    assert out["bid"].null_count() == out.height and out["ask"].null_count() == out.height  # a print is not a quote
    assert sum(s["bars"] for s in stats.values()) == df.height


def test_greeks_are_worked_out_for_the_rows_that_are_read_and_equal_the_whole_chain(tmp_settings):
    from strata.engine.market import DayChain, expand_trade_bars
    st = grid_store(tmp_settings)
    d = DAYS[0]
    o = int(cal.session_bounds(d)[0].timestamp())
    bars = st.read_option_bars("SPY", d, dataset=st.option_dataset("SPY", "trade", "testfeed"))
    ser = SimpleNamespace(end=o + 60 * (np.arange(390) + 1), c=np.full(390, SPOT))
    frame, _ = expand_trade_bars(bars, [o + 60 * m for m in (30, 45, 60, 90)], ser, d)
    lazy, whole = DayChain(frame), DayChain(frame)
    n = len(lazy.ts)
    assert lazy._unfilled == n and not lazy.greeks_computed  # a chain of prints carries no greeks, and none are computed for loading it
    full_delta, full_iv = np.asarray(whole.delta), np.asarray(whole.iv)
    assert whole._unfilled == 0 and np.isfinite(full_delta).all()
    assert lazy.delta[5] == full_delta[5] and lazy._unfilled == n - 1
    assert lazy.delta[5] == full_delta[5] and lazy._unfilled == n - 1  # read again: nothing is recomputed
    np.testing.assert_array_equal(lazy.iv[10:14], full_iv[10:14])
    mask = np.zeros(n, bool)
    mask[[0, 7, 21]] = True
    np.testing.assert_array_equal(lazy.delta[mask], full_delta[mask])
    np.testing.assert_array_equal(lazy.gamma[np.array([3, 3, 8])], np.asarray(whole.gamma)[[3, 3, 8]])
    assert lazy._unfilled == n - 1 - 4 - 3 - 2
    np.testing.assert_array_equal(np.asarray(lazy.delta), full_delta)  # asking for the column completes it
    assert lazy._unfilled == 0


def test_the_single_row_kernels_are_the_same_arithmetic_as_the_parallel_ones():
    from strata.core.pricing import bs_greeks_serial, bs_greeks_vec, implied_vol_serial, implied_vol_vec
    rng = np.random.default_rng(3)
    n = 400
    Sp, K, T = rng.uniform(400, 600, n), rng.uniform(380, 620, n), rng.uniform(0.0002, 0.3, n)
    call = rng.random(n) < 0.5
    price = np.maximum(0.01, rng.uniform(0.05, 30, n))
    np.testing.assert_array_equal(implied_vol_serial(price, Sp, K, T, 0.04, 0.0, call), implied_vol_vec(price, Sp, K, T, 0.04, 0.0, call))
    sig = rng.uniform(0.05, 0.9, n)
    np.testing.assert_array_equal(bs_greeks_serial(Sp, K, T, 0.04, 0.0, sig, call), bs_greeks_vec(Sp, K, T, 0.04, 0.0, sig, call))


def test_only_the_expirations_a_strategy_can_select_are_loaded_and_the_trades_do_not_change(tmp_settings, monkeypatch):
    st = grid_store(tmp_settings)
    far = DAYS[-1]
    frames, index = [], []
    for d in DAYS[:6]:  # a second expiration stored on six sessions: a same-day strategy never reads it
        for k in (495.0, 500.0, 505.0):
            occ = S.occ_symbol("SPY", far, "C", k)
            r = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": far, "strike": k, "right": "C", "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 5} for t in ("09:59", "15:28")]
            frames.append(pl.DataFrame(r).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))))
            index.append({"date": d, "occ": occ, "expiration": far, "strike": k, "right": "C", "bars": 2, "status": "ok", "pages": 1})
    st.write_option_bars("SPY", "testfeed", pl.concat(frames), index)
    asked = []
    real = Backtester._expirations_in_play

    def spy(self, sym, d, ds):
        out = real(self, sym, d, ds)
        asked.append(out)
        return out
    monkeypatch.setattr(Backtester, "_expirations_in_play", spy)
    filtered = run_backtest(cfg(), store=st, record_paths=False)
    assert asked and all(a is not None and a[0] == {d_} for a, d_ in zip(asked, DAYS)) and len(filtered.trades) == 10
    monkeypatch.setattr(Backtester, "_expirations_in_play", lambda self, sym, d, ds: None)  # the whole stored chain, as before
    assert rows(run_backtest(cfg(), store=st, record_paths=False).trades) == rows(filtered.trades)
    # a strategy that reads the chain as a whole keeps the whole chain, and a "nearest expiration" leg keeps what selection accepts
    monkeypatch.setattr(Backtester, "_expirations_in_play", real)
    ds = st.option_dataset("SPY", "trade", "testfeed")

    def play(c):
        bt = Backtester(BacktestConfig.model_validate(c), st, None, record_paths=False)
        return bt._expirations_in_play("SPY", DAYS[0], ds)
    assert play(cfg()) == ({DAYS[0]}, [])
    assert play(cfg(entry_children=[{"kind": "cond", "type": "expr", "params": {"expr": "iv_rank(20) >= 30"}}])) is None
    near = play(cfg(legs=[{"id": "lc", "side": "buy", "right": "C", "dte": 30, "dte_mode": "nearest", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}]))
    from datetime import timedelta
    from strata.engine.selection import dte_tolerance
    assert dte_tolerance(30) == 8 and near == (set(), [(DAYS[0] + timedelta(days=22), DAYS[0] + timedelta(days=38))])


# ----------------------------------------------------------------------------- a configuration it does not understand
def test_a_setting_the_engine_does_not_know_is_refused_and_never_silently_ignored(tmp_settings):
    c = cfg(exits={"profit_target": {"enabled": True, "pct": 30}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}})
    with pytest.raises(ValidationError) as e:
        BacktestConfig.model_validate(c)
    assert "pct" in str(e.value) and "Extra inputs are not permitted" in str(e.value)
    with pytest.raises(ValidationError):
        run_backtest(c, store=grid_store(tmp_settings))
    for bad in ({"execution": {"signal_interval_minutes": 1, "slipage_pct": 2}}, {"pricing": {"mode": "trades", "max_age": 60}}, {"capitol": {"initial": 1000}}):
        with pytest.raises(ValidationError):
            BacktestConfig.model_validate({**cfg(), **bad})
    assert BacktestConfig.model_validate(cfg()).strategies[0].legs[0].dte == 0  # what is spelled as the engine spells it still loads


# ----------------------------------------------------------------------------- another symbol's own levels
def test_a_level_of_another_symbol_is_read_on_that_symbols_own_bars(tmp_settings):
    from strata.data.store import SymbolMeta
    from strata.engine import expr as X
    from strata.engine.scan import scan_signals
    st = grid_store(tmp_settings)
    st.upsert_symbol(SymbolMeta(symbol="QQQ", kind="etf", multiplier=100.0))
    # SPY leaves its first five minutes' range at minute 20; QQQ leaves its own at minute 40
    write_minutes(st, "SPY", [PREV] + DAYS, lambda d, i: 500.0 + (1.0 if i >= 20 else 0.0))
    write_minutes(st, "QQQ", [PREV] + DAYS, lambda d, i: 400.0 + (1.0 if i >= 40 else 0.0))

    def first(expr: str) -> set[int]:
        c = cfg(entry_children=[{"kind": "cond", "type": "expr", "params": {"expr": expr}}], max_entries_per_day=1)
        return {s["minute"] for s in scan_signals(c, st)["signals"]}
    own, both = first("price > opening_range_high(5)"), first("price > opening_range_high(5) and QQQ.price > QQQ.opening_range_high(5)")
    assert len(own) == 1 and len(both) == 1 and min(both) - min(own) == 20  # the confirmation arrives twenty minutes later, on QQQ's own bars
    assert first("QQQ.price > QQQ.opening_range_high(5)") == both
    assert X.validate("QQQ.opening_range_high(30) > 0") == [] and X.validate('IWM.consecutive_bearish_candles("5m") >= 3') == []
    assert X.validate("QQQ.not_a_level(3) > 0") != []


def test_reading_some_expirations_of_a_session_gives_exactly_the_rows_a_filter_afterwards_gives(tmp_settings):
    st = grid_store(tmp_settings)
    far = DAYS[-1]
    occ = S.occ_symbol("SPY", far, "C", 500.0)
    r = [{"ts": datetime.fromtimestamp(et_epoch(DAYS[0], t), tz=timezone.utc), "occ": occ, "expiration": far, "strike": 500.0, "right": "C", "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 5} for t in ("09:59", "15:28")]
    st.write_option_bars("SPY", "testfeed", pl.DataFrame(r).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": DAYS[0], "occ": occ, "expiration": far, "strike": 500.0, "right": "C", "bars": 2, "status": "ok", "pages": 1}])
    ds = st.option_dataset("SPY", "trade", "testfeed")
    cols = ["ts", "occ", "expiration", "strike", "right", "close", "volume"]
    whole = st.read_option_bars("SPY", DAYS[0], dataset=ds)
    part = st.read_option_bars("SPY", DAYS[0], dataset=ds, columns=cols, where=pl.col("expiration") == DAYS[0])
    assert whole.height == 57 and part.height == 55 and part.columns == cols and part.equals(whole.filter(pl.col("expiration") == DAYS[0]).select(cols))
    assert st.read_option_bars("SPY", DAYS[0], dataset=ds, where=pl.col("expiration") == date(2030, 1, 1)).is_empty() and st.read_option_bars("SPY", date(2030, 1, 2), dataset=ds, columns=cols) is None
