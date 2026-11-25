"""Stateful rules: event sequences, cross-symbol relationships, scale-ins, per-position
state and entry-dependent exits - each verified on a hand-built price path where a
loosely related static condition would give a different answer."""
from __future__ import annotations

from datetime import date

import pytest

from strata.core import calendar as cal
from strata.data.store import MarketStore, SymbolMeta
from strata.engine import expr as X
from strata.engine.engine import run_backtest
from strata.engine.models import StrategyConfig
from strata.engine.scan import scan_signals
from tests.helpers import write_minutes, write_options

D0, D = date(2024, 3, 5), date(2024, 3, 6)


def path(points: dict[str, float], start: float):
    """Piecewise-linear 1-minute closes through (HH:MM -> price) knots; the bar ending at a knot closes there."""
    knots = sorted((cal.parse_hhmm(k) - 570, v) for k, v in points.items())

    def f(d, i):
        if i < 0:
            return start
        m = i + 1  # the bar with index i ends at minute i+1 after the open
        prev = (0, start)
        for km, kv in knots:
            if m <= km:
                return prev[1] + (kv - prev[1]) * (m - prev[0]) / max(km - prev[0], 1)
            prev = (km, kv)
        return prev[1]
    return f


def store(tmp_path, today, yesterday=None, other: dict | None = None) -> MarketStore:
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    yesterday = yesterday or (lambda d, i: 100.0 if i < 0 else 100.0 + 0.001 * i)
    write_minutes(st, "SPY", [cal.prev_trading_day(D0)], lambda d, i: 100.0)
    write_minutes(st, "SPY", [D0], yesterday)
    write_minutes(st, "SPY", [D], today)
    for sym, (y, t) in (other or {}).items():
        write_minutes(st, sym, [cal.prev_trading_day(D0)], lambda d, i: 100.0)
        write_minutes(st, sym, [D0], y)
        write_minutes(st, sym, [D], t)
    return st


def base(entry: dict, exits: dict | None = None) -> dict:
    return {"name": "t", "symbol": "SPY", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "fixed", "value": 100}}],
            "entry": {"conditions": {"kind": "group", "op": "and", "children": []}, **entry},
            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}, **(exits or {})}}


def signals(st, strat) -> list[str]:
    r = scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [strat], "execution": {"signal_interval_minutes": 1}}, st)
    return [cal.fmt_hhmm(s["minute"]) for s in r["signals"]]


# ----------------------------------------------------------------------------- 1. drop, then recover half
DROP_RECOVER = {"sequence": [{"name": "drop", "when": "from_open_pct <= -0.5"},
                             {"name": "recover", "when": 'price >= low_since("drop") + 0.5 * (open - low_since("drop"))'}]}


def test_wait_for_a_drop_then_enter_when_half_of_it_is_recovered(tmp_path):
    """Open 100 -> 99.5 at 10:00 (the drop) -> low 99.0 at 10:30 -> back up. Half of the
    decline from the open (100) to the low (99.0) is 99.5, first regained at 11:00."""
    st = store(tmp_path, path({"10:00": 99.5, "10:30": 99.0, "11:00": 99.5, "12:00": 99.9}, 100.0))
    assert signals(st, base(DROP_RECOVER)) == ["11:00"]
    # the static look-alike "down 0.5% from the open" fires an hour earlier, at the drop itself
    static = base({"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct <= -0.5"}}]}})
    assert signals(st, static) == ["10:00"]


def test_the_recovery_is_measured_from_the_low_made_after_the_drop(tmp_path):
    """The decline keeps going to 98.0, so half of it is 99.0 - not the 99.75 that half of the
    initial 0.5% drop would suggest. Price touching 99.6 on the way down is not a recovery."""
    st = store(tmp_path, path({"09:45": 99.5, "10:00": 99.6, "11:00": 98.0, "12:00": 99.0, "13:00": 99.4}, 100.0))
    assert signals(st, base(DROP_RECOVER)) == ["12:00"]


def test_no_drop_means_no_entry_and_a_stage_cannot_complete_on_the_tick_that_armed_it(tmp_path):
    st = store(tmp_path, path({"10:00": 99.7, "12:00": 100.4}, 100.0))  # never 0.5% below the open
    assert signals(st, base(DROP_RECOVER)) == []
    # the drop arms the sequence at 09:31 (price 99.5, half-recovery level 99.75). Price stays at 99.5 until 09:41,
    # so the entry is 09:41 - never the tick that armed it
    st2 = store(tmp_path / "b", path({"09:31": 99.5, "09:40": 99.5, "09:41": 99.8}, 100.0))
    assert signals(st2, base(DROP_RECOVER)) == ["09:41"]


def test_sequence_state_is_reported_and_resets_each_session(tmp_path):
    st = store(tmp_path, path({"10:00": 99.5, "10:30": 99.0, "11:00": 99.5, "12:00": 99.9}, 100.0))
    r = scan_signals({"start": D0.isoformat(), "end": D.isoformat(), "strategies": [base(DROP_RECOVER)], "execution": {"signal_interval_minutes": 1}}, st)
    assert [s["date"] for s in r["signals"]] == [D]  # yesterday drifted up: nothing carried over into today
    why = " | ".join(r["signals"][0]["why"])
    assert "drop 10:00 @ 99.50 -> recover 11:00 @ 99.50" in why and "low_since('drop')" in why


# ----------------------------------------------------------------------------- 2. failed breakout, then VWAP reclaim
FAILED_BREAKOUT = {"sequence": [{"name": "breakout", "when": "price > opening_range_high(15)"},
                                {"name": "failed", "when": "price < opening_range_high(15)"},
                                {"name": "reclaim", "when": "crosses_above(price, vwap)"}]}


def test_enter_only_after_the_first_breakout_fails_and_price_reclaims_vwap(tmp_path):
    """Opening range 100.0-100.3 (first 15 min). Breaks out to 100.6 at 10:00, falls back
    through the range high and below VWAP by 10:40, then crosses back above VWAP."""
    today = path({"09:45": 100.3, "10:00": 100.6, "10:20": 100.2, "10:40": 99.9, "11:10": 100.5, "12:00": 100.6}, 100.0)
    st = store(tmp_path, today)
    r = scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(FAILED_BREAKOUT)], "execution": {"signal_interval_minutes": 1}}, st)
    assert len(r["signals"]) == 1
    sig = r["signals"][0]
    order = [x.split(" ")[0] for x in sig["why"][0].replace("sequence: ", "").split(" -> ")]
    assert order == ["breakout", "failed", "reclaim"]
    times = [x.split(" ")[1] for x in sig["why"][0].replace("sequence: ", "").split(" -> ")]
    assert times[0] < times[1] < times[2] and "10:40" < times[2] <= "11:10"
    # a day that breaks out and never fails produces no entry, although price is above VWAP all along
    st2 = store(tmp_path / "b", path({"09:45": 100.3, "10:00": 100.6, "12:00": 101.5}, 100.0))
    assert signals(st2, base(FAILED_BREAKOUT)) == []


def test_within_minutes_restarts_a_sequence_that_takes_too_long(tmp_path):
    seq = {"sequence": [{"name": "drop", "when": "from_open_pct <= -0.5"}, {"name": "recover", "when": "from_open_pct >= 0", "within_minutes": 30}]}
    st = store(tmp_path, path({"10:00": 99.5, "11:30": 100.0, "12:00": 99.4, "12:20": 100.1}, 100.0))
    # the first recovery (11:30) comes 90 minutes after the drop: too late, so the sequence starts over. The second drop (just
    # before 12:00) is followed by a recovery that crosses the open at 12:18 on this path - inside the 30 minutes
    assert signals(st, base(seq)) == ["12:18"]


# ----------------------------------------------------------------------------- 3. candles, unless yesterday was red
def test_three_red_five_minute_candles_unless_yesterday_was_also_red(tmp_path):
    rule = {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'consecutive_bearish_candles("5m") >= 3 and not (previous_session.open_to_close_pct < 0)'}}]}}
    today = path({"10:00": 100.0, "10:15": 99.7, "11:00": 99.9}, 100.0)  # three falling 5-minute candles 10:00-10:15
    green_yday = store(tmp_path, today, yesterday=lambda d, i: 100.0 if i < 0 else 100.0 + 0.001 * i)
    assert signals(green_yday, base(rule)) == ["10:15"]
    red_yday = store(tmp_path / "b", today, yesterday=lambda d, i: 100.0 if i < 0 else 100.0 - 0.001 * i)
    assert signals(red_yday, base(rule)) == []


# ----------------------------------------------------------------------------- 4. cross-symbol
CROSS = {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "crosses_above(price, prev_high) and QQQ.high <= QQQ.prev_high"}}]}}


def test_break_of_yesterdays_high_only_if_the_other_symbol_has_not_broken_its_own(tmp_path):
    yday = lambda d, i: 100.0 if i < 0 else min(100.0 + 0.002 * i, 100.4)  # yesterday's high 100.4 for both symbols
    spy = path({"10:30": 100.3, "10:31": 100.5, "12:00": 100.6}, 100.1)  # SPY takes out 100.4 at 10:31
    qqq_lagging = path({"12:00": 100.2}, 100.0)  # QQQ stays below its own high
    st = store(tmp_path, spy, yesterday=yday, other={"QQQ": (yday, qqq_lagging)})
    assert signals(st, base(CROSS)) == ["10:31"]
    qqq_broke_first = path({"10:00": 100.6, "12:00": 100.6}, 100.0)  # QQQ already above its high at 10:00
    st2 = store(tmp_path / "b", spy, yesterday=yday, other={"QQQ": (yday, qqq_broke_first)})
    assert signals(st2, base(CROSS)) == []
    assert StrategyConfig.model_validate(base(CROSS)).symbols() == {"QQQ"}


def test_missing_cross_symbol_data_excludes_the_session_instead_of_saying_no(tmp_path):
    st = store(tmp_path, path({"10:31": 100.5}, 100.1))  # QQQ is not stored at all
    r = scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(CROSS)], "execution": {"signal_interval_minutes": 1}}, st)
    assert r["signals"] == [] and r["sessions_evaluable"] == 0
    assert any("QQQ prices" in k for k in r["excluded"])


# ----------------------------------------------------------------------------- 5. scale in, exit both at VWAP
def chain_every_minute(st, d, price_fn, strike=100.0):
    q = []
    for i in range(cal.session_minutes(d) + 1):
        m = 570 + i
        s = price_fn(d, i - 1) if i else price_fn(d, -1)
        intr = max(s - strike, 0.0)
        mid = round(intr + 0.50, 2)
        q.append({"time": cal.fmt_hhmm(m), "expiration": d, "strike": strike, "right": "C", "bid": mid - 0.01, "ask": mid + 0.01, "delta": 0.5, "underlying_price": s})
    write_options(st, "SPY", d, q)


SCALE = {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct <= -1"}}]},
         "adds": [{"when": "price <= entry.underlying_price * (1 - 0.005)", "quantity": 1, "max_adds": 1, "text": "a further 0.5% decline"}]}
VWAP_EXIT = {"time_exit": {"enabled": True, "time": "15:55", "dte_lte": None},
             "custom": {"kind": "group", "op": "or", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "price >= vwap", "label": "price returns to VWAP"}}]}}


def test_enter_add_after_a_further_decline_and_close_both_lots_at_vwap(tmp_path):
    """100 -> 99.0 at 10:00 (enter) -> 98.5 at 10:30 (99.0 * 0.995 = 98.505: add) -> 97.9 at
    11:00 (no second add: max 1) -> rallies; both lots close on the same tick when price is back at VWAP."""
    today = path({"10:00": 99.0, "10:30": 98.5, "11:00": 97.9, "13:00": 99.6, "14:00": 99.7}, 100.0)
    st = store(tmp_path, today)
    chain_every_minute(st, D, today)
    res = run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(SCALE, VWAP_EXIT)], "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0}}, store=st)
    lots = sorted(res.trades, key=lambda t: t["entry_ts"])
    assert [t["lot"] for t in lots] == [1, 2] and lots[0]["group"] == lots[1]["group"] == lots[0]["id"]
    assert lots[0]["entry_time"][11:16] == "10:00" and lots[0]["entry_spot"] == pytest.approx(99.0)
    assert lots[1]["entry_time"][11:16] == "10:30" and lots[1]["entry_spot"] == pytest.approx(98.5) and lots[1]["tag"] == "add"
    assert "a further 0.5% decline" in lots[1]["entry_explain"][0]["text"]
    assert lots[0]["exit_ts"] == lots[1]["exit_ts"] and {t["exit_reason"] for t in lots} == {"custom_exit"}  # one rule, both lots, same tick
    exit_min = cal.minute_of_day_et(lots[0]["exit_ts"] * 1_000_000_000)
    assert 11 * 60 < exit_min < 14 * 60 and "price returns to VWAP" in lots[0]["exit_explain"][0]["text"]
    assert res.diagnostics["signals"] == 1 and res.diagnostics["entries"] == 2  # the add is a lot of the same position, not a second signal


def test_add_rule_reads_the_first_lots_entry_not_the_latest_price(tmp_path):
    """Without the further decline there is no add, however long the position is open."""
    today = path({"10:00": 99.0, "11:00": 98.6, "13:00": 99.8}, 100.0)  # 98.6 > 98.505: the add level is never reached
    st = store(tmp_path, today)
    chain_every_minute(st, D, today)
    res = run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(SCALE, VWAP_EXIT)], "execution": {"signal_interval_minutes": 1}}, store=st)
    assert [t["lot"] for t in res.trades] == [1]


def test_entry_dependent_exit_and_position_state(tmp_path):
    """Exit when the underlying gives back half of its best move since entry - a rule that
    only exists relative to this position's own history."""
    today = path({"10:00": 100.0, "11:00": 102.0, "12:00": 100.9, "13:00": 100.2}, 100.0)
    st = store(tmp_path, today)
    chain_every_minute(st, D, today)
    entry = {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}}
    exits = {"time_exit": {"enabled": True, "time": "15:55", "dte_lte": None},
             "custom": {"kind": "group", "op": "or", "children": [{"kind": "cond", "type": "expr", "params": {
                 "expr": "position.underlying_high >= entry.underlying_price * 1.01 and price <= entry.underlying_price + 0.5 * (position.underlying_high - entry.underlying_price)"}}]}}
    res = run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(entry, exits)], "execution": {"signal_interval_minutes": 1}}, store=st)
    t = res.trades[0]
    # best move: 100 -> 102; half given back at 101.0, reached between 11:00 (102) and 12:00 (100.9)
    assert t["exit_reason"] == "custom_exit" and "11:50" <= t["exit_time"][11:16] <= "11:56" and t["exit_spot"] == pytest.approx(101.0, abs=0.03)


# ----------------------------------------------------------------------------- validation
def test_rules_that_read_unknown_state_are_rejected_with_the_exact_reason():
    with pytest.raises(ValueError, match="no such stage"):
        StrategyConfig.model_validate(base({"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'price > stage_price("drop")'}}]}}))
    with pytest.raises(ValueError, match="not an earlier stage"):
        StrategyConfig.model_validate(base({"sequence": [{"name": "a", "when": 'price > stage_price("b")'}, {"name": "b", "when": "price > open"}]}))
    with pytest.raises(ValueError, match="only exist once a position is open"):
        StrategyConfig.model_validate(base({"sequence": [{"name": "a", "when": "price > entry.underlying_price"}]}))
    with pytest.raises(ValueError, match="no stage captures"):
        StrategyConfig.model_validate(base({"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'price > ref("lvl")'}}]}}))
    ok = StrategyConfig.model_validate(base({"sequence": [{"name": "a", "when": "from_open_pct <= -0.5", "capture": {"lvl": "low"}}, {"name": "b", "when": 'price > ref("lvl") + 0.2'}]}))
    assert [s.name for s in ok.entry.sequence] == ["a", "b"]
    assert X.validate("QQQ.rsi > 50") and X.validate("SPY.__dict__ == 1") and not X.validate("IWM.change_pct < QQQ.change_pct")
    assert X.symbols_used("IWM.change_pct < QQQ.change_pct") == {"IWM", "QQQ"} and X.needs_intraday("QQQ.vwap > 1") and not X.needs_intraday("QQQ.prev_close > 1")


def test_captured_reference_price_is_fixed_at_the_stage(tmp_path):
    seq = {"sequence": [{"name": "drop", "when": "from_open_pct <= -0.5", "capture": {"floor": "low"}}, {"name": "break", "when": 'price < ref("floor") - 0.2'}]}
    st = store(tmp_path, path({"10:00": 99.5, "10:30": 99.4, "11:00": 99.25, "12:00": 99.0}, 100.0))
    r = scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(seq)], "execution": {"signal_interval_minutes": 1}}, st)
    # floor captured at the drop = 99.5; the break needs price < 99.3, first true just before 11:00
    assert len(r["signals"]) == 1 and r["signals"][0]["price"] < 99.3 and "10:40" < cal.fmt_hhmm(r["signals"][0]["minute"]) <= "11:00"


# ----------------------------------------------------------------------------- coarse bars
def _hourly_store(tmp_path, closes: list[float]) -> MarketStore:
    """A session stored only as 60-minute bars (plus the daily bar they imply)."""
    import polars as pl
    from datetime import timezone
    from strata.data.providers.base import minutes_to_daily
    st = store(tmp_path, lambda d, i: 100.0)  # D0 and the session before it have 1-minute bars
    # replace D's minute bars with seven hourly ones: remove the stored minutes for D first
    for p in (tmp_path / "market").rglob("*.parquet"):
        if "/1m/" in str(p):
            df = pl.read_parquet(p)
            if "ts" in df.columns and df["ts"].dtype != pl.Date:
                keep = df.filter(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date() != D)
                keep.write_parquet(p) if not keep.is_empty() else p.unlink()
    o_ts = int(cal.session_bounds(D)[0].timestamp())
    rows, prev = [], 100.0
    for i, c in enumerate(closes):
        rows.append((o_ts + 3600 * i, prev, max(prev, c), min(prev, c), c, 1000.0))
        prev = c
    bars = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", timezone.utc)))
    st.write_bars("SPY", "60m", bars)
    st.write_daily("SPY", minutes_to_daily(bars))
    return MarketStore(tmp_path / "market")


def test_a_sequence_is_not_evaluated_on_hourly_bars_and_the_session_is_reported_as_excluded(tmp_path):
    """On hourly bars the drop and the recovery would each land on one of seven ticks. Such a
    session is excluded with the reason - it is neither a signal nor a 'did not qualify'."""
    st = _hourly_store(tmp_path, [99.4, 99.0, 99.6, 99.8, 99.9, 99.9, 99.9])
    r = scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(DROP_RECOVER)], "execution": {"signal_interval_minutes": 1}}, st)
    assert r["signals"] == []
    assert r["sessions"] == 1 and r["sessions_evaluable"] == 0
    (reason, days), = r["excluded"].items()
    assert "5-minute or finer bars" in reason and "60-minute" in reason and days == [D.isoformat()]
    # the planner says the same thing before anything runs
    from strata.data import planner as P
    req = P.requirements("options", {"start": D.isoformat(), "end": D.isoformat(), "strategies": [base(DROP_RECOVER)]}, "rules")
    assert req["underlying"]["SPY"]["seconds"] == 300
    cov = P.underlying_coverage(st, req, D, D)
    assert cov["usable"]["sessions"] == 0 and "60-minute" in cov["excluded"][0]["reason"]
    # a rule without a sequence still runs on hourly bars: only step-by-step rules need fine ones
    static = base({"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct <= -0.5"}}]}})
    assert len(scan_signals({"start": D.isoformat(), "end": D.isoformat(), "strategies": [static]}, st)["signals"]) == 1


def test_a_study_with_a_sequence_excludes_hourly_sessions_too(tmp_path):
    from strata.research.study import run_study
    st = _hourly_store(tmp_path, [99.4, 99.0, 99.6, 99.8, 99.9, 99.9, 99.9])
    res = run_study({"symbol": "SPY", "start": D.isoformat(), "end": D.isoformat(), "entry_time": "signal", "sequence": DROP_RECOVER["sequence"],
                     "target": {"kind": "pct", "value": 0.2}, "horizon": {"kind": "session"}}, st)
    assert res["stats"]["qualifying_sessions"] == 0 and res["stats"]["excluded_sessions"] == 1
    assert any("5-minute or finer bars" in x["reason"] for x in res["excluded"])


# ----------------------------------------------------------------------------- plain wording of expirations and periods
@pytest.mark.parametrize("text,dte", [
    ("buy a call expiring in 7 days", 7), ("calls expiring in about two weeks", 14), ("buy 0dte calls", 0), ("45 days to expiration", 45), ("a month out", 30),
    ("7-day atm calls", 7), ("buy spy weeklies", 7), ("options with 10 days left", 10), ("buy a 30dte call and hold for 5 days", 30),
    ("hold for 5 days", None), ("sell after 3 days", None),  # a holding period is not an expiration
])
def test_days_to_expiration_is_read_from_plain_wording(text, dte):
    from strata.ai.nl_rules import find_dte
    assert find_dte(text) == dte


@pytest.mark.parametrize("text,start,end", [
    ("from 2026-11-17 to 2026-11-23", date(2026, 11, 17), date(2026, 11, 23)), ("between 3/1/2024 and 6/30/2024", date(2024, 3, 1), date(2024, 6, 30)),
    ("since march 2020", date(2020, 3, 1), None), ("from january 2022 to june 2023", date(2022, 1, 1), date(2023, 6, 30)), ("on 2026-11-23", date(2026, 11, 23), date(2026, 11, 23)),
    ("through 2024-12-31", None, date(2024, 12, 31)), ("from 2018 to 2021", date(2018, 1, 1), date(2021, 12, 31)), ("buy at 10:00 and sell at 15:55", None, None),
    ("from 2024-02-31 to 2024-03-05", None, None),  # not a date: nothing is guessed
])
def test_a_stated_period_is_read_exactly(text, start, end):
    from strata.ai.nl_rules import find_period
    assert find_period(text)[:2] == (start, end)


def test_a_dated_weekly_call_request_compiles_to_what_was_said():
    from strata.ai.draft import compile_draft
    from strata.ai.nl_rules import parse_fresh
    d = parse_fresh("Buy one at-the-money SPY call expiring in 7 days at 10:00 ET and sell it at 15:55 ET, using option trade prices, from 2026-11-17 to 2026-11-23.", ["SPY"])
    assert (d.dte, d.start, d.end, d.entry_time, d.exit_time) == (7, date(2026, 11, 17), date(2026, 11, 23), "10:00", "15:55")
    c = compile_draft(d, None)
    assert not any(a["field"] == "dte" for a in c["assumptions"])  # stated, so not assumed
    leg = c["strategy"]["legs"][0]
    assert (leg["side"], leg["right"], leg["dte"], leg["strike"]["method"], leg["strike"]["value"]) == ("buy", "C", 7, "atm_offset", 0.0)
