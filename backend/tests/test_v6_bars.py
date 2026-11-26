"""V6: the audit of stored intraday bars against the exchange calendar and the daily record."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from strata.audit import bars as AB
from strata.core import calendar as cal
from strata.data.quality import cumulative_volume_artefacts
from tests.helpers import make_store

FRI, MON, TUE = date(2024, 3, 8), date(2024, 3, 11), date(2024, 3, 12)  # the clocks went forward on Sunday 2024-03-10
HALF = date(2024, 7, 3)  # 13:00 close


def minute_bars(d: date, start="09:30", end=None, px=100.0, vol=1000.0, shift_minutes=0) -> list[tuple]:
    o, c = cal.session_bounds(d)
    h, m = map(int, start.split(":"))
    a = datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET)
    b = c if end is None else datetime(d.year, d.month, d.day, int(end[:2]), int(end[3:]), tzinfo=cal.ET)
    out, t = [], a
    while t < b:
        out.append((t + timedelta(minutes=shift_minutes), px, px + 0.1, px - 0.1, px, vol))
        t += timedelta(minutes=1)
    return out


def write(st, rows, sym="TST", provider="yahoo", res="1m"):
    df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(pl.col("ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
    st.write_bars(sym, res, df, provider=provider)


def daily(st, days, close=100.0, volume=None, sym="TST"):
    st.write_daily(sym, pl.DataFrame({"date": days, "open": [100.0] * len(days), "high": [100.1] * len(days), "low": [99.9] * len(days), "close": [close] * len(days),
                                      "volume": [volume or 1000.0 * cal.session_minutes(d) for d in days]}), provider="yahoo")


def finding(rep, check):
    return next((f for r in rep["datasets"] for f in r["findings"] if f["check"] == check), None)


def test_a_bar_that_carries_the_days_running_total_is_recognised_and_nothing_else_is():
    rng = np.random.default_rng(1)
    v = rng.integers(500, 4000, 390).astype(float)
    v[0] = 60_000  # a heavy opening minute is real
    assert not cumulative_volume_artefacts(v, 60).any()
    bad = v.copy()
    for i in (40, 111, 250):
        bad[i] = v[:i].sum() + 123  # everything traded so far, in one bar
    assert np.flatnonzero(cumulative_volume_artefacts(bad, 60)).tolist() == [40, 111, 250], "each is found although the ones before it inflated the raw running total"
    early = v.copy()
    early[5] = v[:5].sum() * 3
    assert not cumulative_volume_artefacts(early, 60).any(), "nothing is judged in the first quarter of an hour"
    assert not cumulative_volume_artefacts(np.array([1e6, 1e6, 9e6, 1e6, 1e6, 1e6, 1e6]), 3600).any(), "an hourly bar can hold most of a day: it is not judged"
    assert cumulative_volume_artefacts(np.array([]), 60).size == 0


def test_the_engine_gives_such_a_bar_no_weight(tmp_path):
    from strata.engine.market import UnderlyingSeries
    st = make_store(tmp_path)
    rows = minute_bars(FRI)
    rows[200] = (rows[200][0], 100.0, 100.1, 99.9, 100.0, 200_000.0 + 55)  # the 200 bars before it traded 200,000 shares
    write(st, rows)
    daily(st, [FRI])
    ser = UnderlyingSeries(st.read_best_bars("TST"), official=st.official_closes("TST"))
    assert ser.volume_artefacts == 1 and ser.v[200] == 0.0 and ser.v[199] == 1000.0
    assert ser.d_v[0] == pytest.approx(389 * 1000.0), "the session's volume is the 389 bars whose volume is known"


def test_a_clean_dataset_has_no_finding_across_a_clock_change_and_an_early_close(tmp_path):
    st = make_store(tmp_path)
    days = [FRI, MON, TUE, HALF]
    write(st, [r for d in days for r in minute_bars(d)])
    daily(st, cal.trading_days(FRI, HALF))
    rep = AB.audit(st, "TST")
    r = rep["datasets"][0]
    assert r["sessions"] == 4 and r["bars"] == 390 * 3 + 210 and r["daylight_saving"] == {"changes_in_range": [MON.isoformat()], "sessions_not_starting_at_0930": 0}
    assert r["early_closes"] == {"sessions": 1, "with_bars_after_the_close": 0} and r["intervals"] == {"expected": 390 * 3 + 210, "missing": 0}
    assert [f["check"] for f in r["findings"]] == ["missing sessions"], "only the sessions between March and July that were never stored"
    assert r["compared_with_daily"]["largest_pct"] == 0.0 and r["volume_share_of_daily_pct"] == 100.0 and r["volume_artefacts"] == {"bars": 0, "sessions": 0}
    assert rep["finest_resolution_by_session"]["1-minute"]["sessions"] == 4 and "daily only" in rep["finest_resolution_by_session"]


def test_every_kind_of_fault_is_found_and_named(tmp_path):
    st = make_store(tmp_path)
    days = cal.trading_days(date(2024, 3, 4), date(2024, 3, 15))  # ten sessions; the clock change falls between the 8th and the 11th
    rows = []
    for d in days:
        if d == date(2024, 3, 5):
            continue  # never stored
        if d == date(2024, 3, 6):
            rows += minute_bars(d, start="11:00")  # starts late
        elif d == date(2024, 3, 7):
            rows += minute_bars(d, start="09:00")  # half an hour of pre-market bars
        elif d == MON:
            rows += minute_bars(d, start="09:30", shift_minutes=60)  # timestamps an hour late after the clock change: a fixed-offset conversion
        elif d == TUE:
            b = minute_bars(d)
            b[300] = (b[300][0], 100.0, 100.1, 99.9, 100.0, 300_000.0)
            rows += b
        elif d == date(2024, 3, 13):
            rows += minute_bars(d, px=100.6)  # the bars end 0.6% above the official close
        elif d == date(2024, 3, 14):
            rows += minute_bars(d, px=200.0)  # twice the daily price: not split-adjusted
        else:
            rows += minute_bars(d)
    rows += minute_bars(date(2024, 3, 9), start="10:00", end="10:05")  # a Saturday
    rows += minute_bars(HALF, end="16:00")  # an early-close session with bars all afternoon
    write(st, rows)
    daily(st, days + [HALF])
    rep = AB.audit(st, "TST")
    assert rep["warnings"] >= 8
    f = lambda check: finding(rep, check)
    assert f("missing sessions")["sample"][0] == "2024-03-05" or "2024-03-05" in f("missing sessions")["detail"] or f("missing sessions")["sessions"] >= 1
    assert f("bars on a non-session")["sample"] == ["2024-03-09"]
    assert "2024-03-06" in f("partial sessions")["sample"] and "2024-03-11" in f("partial sessions")["sample"], "a late start, and a session whose shifted bars begin at 10:30"
    out = f("bars outside regular hours")
    assert set(out["sample"]) >= {"2024-03-07", "2024-03-11", "2024-07-03"} and "extended-hours" in out["detail"]
    assert f("bars after an early close")["sample"] == ["2024-07-03"]
    assert f("cumulative volume in single bars")["sessions"] == 1 and f("cumulative volume in single bars")["sample"] == ["2024-03-12"]
    assert f("last bar away from the official close")["sample"] == ["2024-03-13"] and "+0.60%" in f("last bar away from the official close")["detail"]
    assert f("different price scale")["sample"] == ["2024-03-14"] and "split" in f("different price scale")["detail"]
    text = AB.render(rep)
    assert "WARNING: bars on a non-session" in text and "Routes to more intraday history" in text


def test_two_providers_at_one_resolution_are_reported_as_mixed_feeds_and_a_thin_feed_is_called_thin(tmp_path):
    st = make_store(tmp_path)
    write(st, minute_bars(FRI), provider="yahoo")
    write(st, minute_bars(MON, vol=25.0), provider="alpaca")  # one exchange: 2.5% of the consolidated volume
    daily(st, [FRI, MON])
    rep = AB.audit(st, "TST")
    assert rep["mixed_feeds"][0]["providers"] == ["alpaca", "yahoo"] or sorted(rep["mixed_feeds"][0]["providers"]) == ["alpaca", "yahoo"]
    assert "never blended" in rep["mixed_feeds"][0]["note"] and "one exchange" in rep["mixed_feeds"][0]["note"]
    thin = next(r for r in rep["datasets"] if r["provider"] == "alpaca")
    assert thin["volume_share_of_daily_pct"] == 2.5 and any(x["check"] == "volume far below the daily total" for x in thin["findings"])
