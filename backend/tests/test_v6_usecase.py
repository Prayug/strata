"""V6: the original use case, end to end on a small market whose answer is known by hand.

    Buy at-the-money 0DTE calls at the open, exit when the underlying is 1% above its entry (or at 15:55),
    only on sessions that follow a fall of at least 3%.

Thirteen sessions, four of them after a fall of 3% or more. Two have a full session of bid/ask quotes, one had no 0DTE
contract listed, one has no intraday option prices at all. Each of the four must come out as exactly that.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.research import usecase as U
from tests.helpers import chain, make_store, write_minutes, write_options
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 3, 20))[:13]
CLOSE = [500.0, 500.0, 480.0, 486.0, 486.0, 470.0, 470.0, 470.0, 454.0, 454.0, 438.0, 438.0, 438.0]
S3, S6, S9, S11 = DAYS[3], DAYS[6], DAYS[9], DAYS[11]


def price(d: date, i: int) -> float:
    k = DAYS.index(d)
    if d == S3:
        return 480.0 if i < 120 else 486.0  # opens where it closed the day before, then rises 1.25% at 11:30
    return CLOSE[k]


def quotes(d: date, expiration: date, every: int = 5) -> list[dict]:
    """A chain every 5 minutes: call = intrinsic + a time value that loses 0.004 a minute; bid/ask 0.05 either side."""
    q = []
    for m in range(0, cal.session_minutes(d), every):
        s = price(d, m - 1) if m else price(d, 0)
        tv = 2.0 - 0.004 * m
        hh = f"{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}"
        q += chain(hh, {(float(k), "C"): (round(max(s - k, 0) + tv - 0.05, 2), round(max(s - k, 0) + tv + 0.05, 2), 0.5) for k in range(int(s) - 12, int(s) + 13)}, expiration, underlying=s)
    return q


@pytest.fixture
def store(tmp_settings):
    st = make_store(tmp_settings, sym="SPY", settlement="physical", exercise="american")
    write_minutes(st, "SPY", DAYS, price)
    for d in (S3, S6):
        write_options(st, "SPY", d, quotes(d, d))
    eod = lambda d, exp: pl.DataFrame([{"ts": cal.session_bounds(d)[1], "expiration": exp, "strike": 480.0, "right": "C", "bid": 1.0, "ask": 1.1}]).with_columns(pl.col("ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
    for d in (S3, S6, S11):
        st.write_option_day("SPY", d, eod(d, d), provider="eod_archive", res="EOD")  # the end-of-day chain lists a contract expiring that session
    st.write_option_day("SPY", S9, eod(S9, cal.next_trading_day(S9)), provider="eod_archive", res="EOD")  # on S9 the nearest expiration is the next day: no 0DTE existed
    return st


def test_the_falls_are_found_the_same_way_by_arithmetic_and_by_the_engine(store):
    assert [(q["date"], q["fall_pct"]) for q in U.down_days(store, "SPY", 3.0)] == [(S3, -4.0), (S6, -3.292), (S9, -3.404), (S11, -3.524)]
    assert [q["date"] for q in U.down_days(store, "SPY", 4.0)] == [S3] and U.down_days(store, "SPY", 5.0) == []


def test_the_sentence_and_the_hand_written_strategy_are_the_same_strategy(store):
    ai = U.ai_config(store, "SPY", 3.0)
    assert ai["config"] is not None and ai["open"] == [] and {a["field"] for a in ai["assumptions"]} == {"stop", "exit_time"}, "what the sentence leaves unsaid is listed as assumed, not hidden"
    assert U.config_difference(U.manual_config("SPY", 3.0), ai["config"]) == []
    # and the comparison can tell them apart: an exit ten minutes earlier, a looser expiration rule
    assert U.config_difference(U.manual_config("SPY", 3.0, time_exit="15:45"), ai["config"]) == ["exits"]
    loose = U.manual_config("SPY", 3.0)
    loose["strategies"][0]["legs"][0]["dte_mode"] = "nearest"
    assert U.config_difference(loose, ai["config"]) == ["legs"]
    assert U.config_difference(U.manual_config("SPY", 2.0), ai["config"]) == ["entry"]


def test_every_qualifying_session_is_accounted_for_and_priced_twice_where_it_can_be(store, tmp_settings):
    rep = U.run(store, "SPY", thresholds=(3.0, 4.0), today=date(2024, 3, 21))
    r = rep["thresholds"][0]
    assert (r["qualifying"], r["engine_scan"], r["lists_identical"]) == (4, 4, True) and r["interpreter"]["identical"] is True and r["from_sentence_identical"] is True
    by = {s["date"]: s for s in r["sessions"]}
    assert [by[d.isoformat()]["status"] for d in (S3, S6, S9, S11)] == ["priced", "priced", "no 0DTE contract was listed", "no intraday option prices stored"]
    assert [by[d.isoformat()]["zero_dte_listed"] for d in (S3, S6, S9, S11)] == [True, True, False, True]
    # S3: bought the 480 call at the 09:30 ask (2.05). The rise to 486 is first known at 11:31; the 11:30 quote is one minute old and still usable: sold at its bid, 1.47
    a = by[S3.isoformat()]
    assert a["agree"] is True and a["same_from_sentence"] is True and a["engine"]["exit_reason"] == a["reference"]["exit_reason"] == "underlying_target"
    assert (a["engine"]["entry_time"][11:16], a["engine"]["exit_time"][11:16]) == ("09:30", "11:31")
    assert a["engine"]["legs"] == [{"right": "C", "strike": 480.0, "expiration": S3.isoformat(), "entry_price": 2.05, "exit_price": 1.47, "price_source": "quote"}]
    assert a["engine"]["pnl"] == a["reference"]["pnl"] == pytest.approx((1.47 - 2.05) * 100 - 1.30)
    # S6: flat all day, closed at the 15:55 bid (time value 2.0 - 0.004 x 385 = 0.46, less 0.05)
    b = by[S6.isoformat()]
    assert b["agree"] is True and b["engine"]["exit_reason"] == "time_exit" and b["engine"]["pnl"] == pytest.approx((0.41 - 2.05) * 100 - 1.30) and b["reference"]["session_observed"] == "whole session"
    assert r["counts"] == {"priced_by_engine": 2, "priced_by_reference_only": 0, "agree": 2, "disagree": 0, "no_contract": 1, "missing": 1, "signal_not_filled": 0, "partial_at_provider": 0, "pnl_engine": pytest.approx(-224.6),
                           "pnl_reference": 0.0, "two_providers": 0, "two_providers_same_trade": 0, "wins": 0, "pnl_worst_print": pytest.approx(-224.6), "pnl_best_print": pytest.approx(-224.6), "with_print_range": 0}
    whole = r["periods"][0]
    assert (whole["qualifying"], whole["no_0dte_contract"], whole["testable_at_most"], whole["priced"], whole["missing"]) == (4, 1, 3, 2, 1) and whole["priced_pct_of_qualifying"] == 50.0 and whole["priced_pct_of_testable"] == 66.7
    alp = next(p for p in r["periods"] if "Alpaca" in p["label"])
    assert alp["from"] == "2024-01-18" and alp["qualifying"] == 4, "the same sessions against the period a free provider reaches (its first session with data, as measured), reported next to the whole history"
    assert by[S11.isoformat()]["reach"] == ["Alpaca free account (trade prints)", "Databento free credit (1-minute bid/ask)"], "a missing session lists the routes whose history includes its date"
    four = rep["thresholds"][1]
    assert four["qualifying"] == 1 and four["counts"]["priced_by_engine"] == 1 and four["sessions"][0]["date"] == S3.isoformat()
    text = U.render(rep)
    assert "| 3% | 4 | yes | 2 | -224.60 |  | 0 of 2 | 0 | 2 of 2 | none held twice | 0 | 0 | 1 | 1 | yes |" in text, "bid/ask sessions carry no print range: the quotes are the fill" and f"| {S3.isoformat()} | -4.0% |" in text and "engine and reference agree" in text
    assert U.saved()["thresholds"][0]["qualifying"] == 4 and (tmp_settings / "reports" / "usecase.md").exists()


def test_a_partly_observed_session_is_priced_by_the_reference_only_and_says_so(tmp_settings):
    """A vendor sample: quotes from 09:30 to 12:00 only. The engine does not run a partial session; the reference prices what was observed."""
    st = make_store(tmp_settings, sym="SPY", settlement="physical", exercise="american")
    write_minutes(st, "SPY", DAYS[:4], price)
    part = [x for x in quotes(S3, S3) if x["time"] <= "12:00"]
    from tests.helpers import et_epoch
    rows = [{"ts": et_epoch(S3, x["time"]), "expiration": x["expiration"], "strike": x["strike"], "right": x["right"], "bid": x["bid"], "ask": x["ask"], "underlying_price": x["underlying_price"]} for x in part]
    df = pl.DataFrame(rows).with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
    st.write_option_day("SPY", S3, df, provider="vendor_sample", res="5m", status="partial", session_info={"first_snapshot": "09:30", "last_snapshot": "12:00"})
    from strata.data.bulk import underlying_from_snapshots
    ub, res = underlying_from_snapshots(df.select("ts", "underlying_price"))
    st.write_bars("SPY", res, ub, provider="vendor_sample")
    rep = U.run(st, "SPY", thresholds=(3.0,), today=date(2024, 3, 8), save=False)
    s = rep["thresholds"][0]["sessions"][0]
    assert s["status"] == "priced (reference only)" and s["engine"] is None and s["agree"] is None
    assert s["reference"]["session_observed"] == "09:30 to 12:00 ET" and s["reference"]["outcome_observed"] is True and s["reference"]["exit_reason"] == "underlying_target"
    assert s["stored"] == [{"provider": "vendor_sample", "observation": "bid/ask quotes", "resolution": "5m", "status": "partial", "window": ["09:30", "12:00"]}]
    assert rep["thresholds"][0]["counts"]["priced_by_reference_only"] == 1 and rep["stored_option_data"]["by_provider"]["vendor_sample"]["partial_sessions"] == 1
    assert "reference simulator only (09:30 to 12:00 ET)" in U.render(rep)
