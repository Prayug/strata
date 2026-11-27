"""V6: how well Strata understands what it is told, measured.

`strata.ai.evalset` holds 110 development sentences and 40 held-out ones (written after the readers were corrected and
scored once before anything was changed in response: 30 of 40 were right on that first run). Each carries the meaning a
careful reader takes from it. This test runs both sets through the application's own routing on a fixture store and
holds the line on two things: accuracy, and above all the number of SILENT misreadings.

The named regressions below are the V5 failure classes: a condition dropped without a word, a clock time in the wrong
zone, session VWAP, a comparison between two symbols, an outcome size that vanished.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from strata.ai import evalset as E
from strata.ai import intent as I
from strata.core import calendar as cal
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def ag(tmp_path_factory):
    from strata.ai.agent import Agent
    from strata.api import db
    from strata.config import get_settings, override_settings
    from strata.data.store import MarketStore, SymbolMeta
    tmp = tmp_path_factory.mktemp("nl")
    old = get_settings().data_dir
    override_settings(data_dir=tmp)
    db.init_db()
    st = MarketStore(tmp / "market")
    rng = np.random.default_rng(4)
    days, d = [], date(2021, 1, 4)
    while len(days) < 1000:
        if cal.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    for sym, px0, vol in (("SPY", 400.0, 0.011), ("QQQ", 320.0, 0.014), ("IWM", 200.0, 0.013)):
        r = rng.standard_t(4, len(days)) * vol
        c = px0 * np.cumprod(1 + r)
        o = np.r_[px0, c[:-1]] * (1 + rng.normal(0, 0.003, len(days)))
        st.upsert_symbol(SymbolMeta(symbol=sym, kind="etf", vol_index="VIX"))
        st.write_daily(sym, pl.DataFrame({"date": days, "open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997, "close": c, "volume": [1e7] * len(days)}), provider="yahoo")
        rows = []
        for i in range(len(days) - 15, len(days)):  # five-minute bars for the last fifteen sessions: clock times need them
            t0 = int(cal.session_bounds(days[i])[0].timestamp())
            n = cal.session_minutes(days[i]) // 5
            path = np.linspace(o[i], c[i], n + 1)
            rows += [(t0 + 300 * k, path[k], max(path[k], path[k + 1]) * 1.0004, min(path[k], path[k + 1]) * 0.9996, path[k + 1], 5e4) for k in range(n)]
        st.write_bars(sym, "5m", pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))), provider="yahoo")
        st.rebuild_manifest(sym)
    v = 14 + 12 * (0.5 + 0.5 * np.sin(np.arange(len(days)) / 30)) + rng.normal(0, 1.5, len(days))
    st.upsert_symbol(SymbolMeta(symbol="VIX", kind="volatility_index", has_options=False))
    st.write_daily("VIX", pl.DataFrame({"date": days, "open": v, "high": v + 1, "low": v - 1, "close": v, "volume": [0.0] * len(days)}), provider="cboe")
    st.rebuild_manifest("VIX")
    yield Agent(None, st, llm_factory=lambda: None)
    override_settings(data_dir=old)


def read(ag, text, current=None):
    return I.understand(ag, text, "SPY", current=current)


# ----------------------------------------------------------------------------- the measured sets
def test_the_development_set_is_understood_and_nothing_in_it_is_misread_silently(ag):
    r = E.evaluate(ag)
    assert r["cases"] == len(E.CASES) >= 100
    assert r["silent"] <= 1, [(f["text"], f["differences"]) for f in r["failures"] if f["status"] == "silent"]
    assert r["correct"] >= 105, [(f["text"], f["status"], f["differences"]) for f in r["failures"]]
    assert set(r["by_tag"]) == {"study", "strategy", "time zones", "follow-up", "conversation", "must be flagged"}
    assert r["by_tag"]["must be flagged"]["correct"] == r["by_tag"]["must be flagged"]["cases"], "every sentence that cannot be represented says so"
    assert r["by_tag"]["time zones"]["correct"] == r["by_tag"]["time zones"]["cases"]


def test_the_held_out_set_holds_up(ag):
    """These sentences were not used to tune the readers before their first scoring (75% then). They are now, so this is a regression line, not an estimate."""
    r = E.evaluate(ag, cases=E.HELDOUT)
    assert r["cases"] == 40 and r["correct"] >= 34 and r["silent"] <= 2, [(f["text"], f["status"], f["differences"]) for f in r["failures"]]


def test_an_evaluation_row_says_whether_a_failure_was_flagged_or_silent(ag):
    bad = [{"text": "Buy ATM SPY 0DTE calls at 6:30 AM Pacific.", "kind": "strategy", "symbol": "SPY", "conditions": [], "structure": "long_put", "tag": "x"},
           E.U("Buy ATM SPY 0DTE calls at 6:30 AM Pacific.")]
    r = E.evaluate(ag, cases=bad)
    assert [x["status"] for x in r["rows"]] == ["silent", "silent"] and "structure: long_call for long_put" in r["rows"][0]["differences"][0]
    assert r["rows"][1]["differences"] == ["the part that cannot be represented was not flagged"] and r["accuracy_pct"] == 0.0


# ----------------------------------------------------------------------------- named regressions
def test_a_condition_is_never_dropped_without_a_word(ag):
    for text, key, op, val in (("Buy ATM SPY 1DTE calls at the open after SPY closes down 3% or more.", "prev_return", "<=", -3.0),
                               ("Buy ATM 0DTE SPY puts at the open on days after SPY rallied 2% or more, exit at noon Eastern.", "prev_return", ">=", 2.0),
                               ("Buy 5 contracts of the SPY 0DTE at-the-money call at 10:45 AM Central if SPY is down more than 0.8% from the open, with a 35% premium stop.", "from_open", "<=", -0.8),
                               ("Buy SPY 0DTE calls at the opening bell following a session where SPY dropped at least 3.5%.", "prev_return", "<=", -3.5)):
        it = read(ag, text)
        got = [(c["key"], c["op"], c["value"]) for c in it["conditions"]]
        assert (key, op, val) in got or it["unsupported"], (text, got)
    it = read(ag, "Buy ATM SPY 1DTE calls at the open after SPY closes down 3% or more.")
    assert [(c["key"], c["op"], c["value"]) for c in it["conditions"]] == [("prev_return", "<=", -3.0)] and it["unsupported"] == []
    # what cannot be read stays in front of the user, in their words
    for text, words in (("Buy SPY 0DTE calls at the open when Mercury is in retrograde.", "Mercury is in retrograde"), ("Buy SPY 0DTE calls at the open after a 3% down day, but only if Twitter sentiment is bullish.", "Twitter sentiment is bullish"),
                        ("How often does SPY rise 1% from the open when retail sentiment is euphoric?", "retail sentiment is euphoric")):
        it = read(ag, text)
        assert any(words.lower() in str(u).lower() for u in it["unsupported"]), (text, it["unsupported"])
    it = read(ag, "Buy SPY 0DTE calls at the open after a 3% down day, but only if Twitter sentiment is bullish.")
    assert ("prev_return", "<=", -3.0) in [(c["key"], c["op"], c["value"]) for c in it["conditions"]], "the part that WAS understood is kept"


def test_a_number_is_used_once_and_for_what_it_says(ag):
    it = read(ag, "Buy 0DTE SPY calls at 10:00 AM New York time if SPY is up at least 0.5% from the open; exit at 50% profit or 25% loss.")
    assert [(c["key"], c["op"], c["value"]) for c in it["conditions"]] == [("from_open", ">=", 0.5)]
    assert it["exits"]["underlying_for"] is None and it["exits"]["premium_target"] == 50.0 and it["exits"]["premium_stop"] == 25.0, "0.5% is the condition, not also a target"
    it = read(ag, "Sell an iron condor on SPY with 16-delta shorts and 5-point wings, 30 DTE, close at 50% profit.")
    assert it["exits"]["premium_target"] == 50.0 and it["exits"]["premium_stop"] is None and it["width"] == 5.0, "'close' contains 'lose': it is not a stop"
    it = read(ag, "Sell a 30-delta SPY put with 7 days to expiration every Monday at 10:00 AM Eastern, take profit at 50% and stop at 200%.")
    assert it["structure"] == "short_put", "a put that is SOLD is a short put"
    it = read(ag, "Short strangle on SPY, 20 delta, 14 DTE, entered Tuesdays at 11:00 AM Eastern, stop at 200% of the credit.")
    assert [(c["key"], c["value"]) for c in it["conditions"]] == [("weekday", ["Tue"])], "'entered Tuesdays' is not a red Tuesday"
    assert all(c["ok"] for c in read(ag, "Buy SPY 0DTE calls at 6:30 AM Pacific after a 3% down day, 2 contracts, stop at 40% loss, target SPY up 1.5%.")["checks"])


def test_clock_times_are_new_york_time_whatever_zone_they_were_said_in(ag):
    it = read(ag, "Buy ATM SPY 0DTE calls at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move.")
    assert it["entry"] == "open" and it["exits"]["time"] == "15:45" and it["exits"]["underlying_for"] == 1.0 and it["entry_stated"].startswith("6:30")
    base = ag._rules(E.ORIGINAL, None, sorted(ag.store.stored_symbols()), None, "SPY")["draft"]
    it = read(ag, "Enter at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move.", current=base)
    assert it["kind"] == "modify" and it["entry"] == "open" and it["exits"]["time"] == "15:45" and it["exits"]["underlying_for"] == 1.0, "one entry time and one exit time, not two entry times to compare"
    assert [(c["key"], c["value"]) for c in it["conditions"]] == [("prev_return", -3.0)], "the condition of the strategy being edited is kept"
    for said, ny in (("7 AM PST", "10:00"), ("8:45 AM Central", "09:45"), ("7:35 AM Mountain", "09:35"), ("1 PM ET", "13:00")):
        assert read(ag, f"Buy ATM SPY 0DTE calls at {said}.")["entry"] == ny
    it = read(ag, "Buy ATM SPY 0DTE calls at 9:45 AM and sell them at 11:15 AM.")
    assert it["tz_assumed"] and any(c["check"] == "Time zone" and c.get("assumed") for c in it["checks"]), "a zone that was assumed is said"


def test_vwap_is_the_sessions_and_a_comparison_between_symbols_is_kept(ag):
    it = read(ag, "Buy SPY 0DTE calls at 10:00 AM Eastern when SPY is above VWAP.")
    assert [c["expr"] for c in it["conditions"]] == ["price > vwap"] and "session VWAP" in it["conditions"][0]["text"]
    it = read(ag, "Buy SPY 0DTE calls at 10:30 AM ET when QQQ is up more than SPY from the open.")
    assert [c["expr"] for c in it["conditions"]] == ["((QQQ.from_open_pct) - (from_open_pct)) > 0"], "not 'QQQ is up': the comparison is the condition"
    it = read(ag, "Buy SPY 0DTE puts at 11:00 AM ET when IWM is down more than SPY.")
    assert [c["expr"] for c in it["conditions"]] == ["((IWM.from_open_pct) - (from_open_pct)) < 0"]


def test_the_size_of_an_outcome_is_part_of_the_question(ag):
    it = read(ag, "When VIX is above 30 and SPY gaps down more than 1%, how often does SPY rise 1% from the open?")
    # V8: the semantic layer reads this sentence itself ("frame"); before V8 the research-graph reader did
    assert it["reader"] in ("graph", "frame") and {k: it["outcome"][k] for k in ("direction", "size", "measure", "from", "horizon")} == {"direction": "up", "size": 1.0, "measure": "touch", "from": "open", "horizon": "session"}
    assert sorted((c["key"], c["value"]) for c in it["conditions"]) == [("gap", -1.0), ("vix", 30.0)]
    it = read(ag, "How often does SPY rise 1% from the open after a 3% down day?")
    assert it["outcome"]["size"] == 1.0 and it["outcome"]["measure"] == "touch" and it["outcome"]["from"] == "open" and [(c["key"], c["value"]) for c in it["conditions"]] == [("prev_return", -3.0)]
    it = read(ag, "How often does QQQ rise 2% within 5 days after falling more than 1%?")
    assert it["symbol"] == "QQQ" and it["outcome"]["size"] == 2.0 and it["outcome"]["horizon"] == "sessions:5" and it["outcome"]["measure"] == "touch"
    it = read(ag, "How often does SPY close at least 1% higher the day after a 2% drop?")
    assert it["outcome"]["size"] == 1.0 and it["outcome"]["measure"] == "close"
    # the guard itself: a reading that has no place for a size in the sentence fails validation
    checks = I.validate({"kind": "study", "conditions": [{"key": "prev_return", "op": "<=", "value": -3.0}], "outcome": {"direction": "up", "size": None}}, "How often does SPY rise 1% after a 3% down day?")
    assert checks[0]["ok"] is False and "1%" in checks[0]["detail"]


def test_every_request_goes_through_the_same_stages(ag):
    it = read(ag, E.ORIGINAL)
    assert [s["stage"] for s in it["stages"]] == ["understand", "validate"] and it["stages"][1]["status"] == "done"
    I.stage(it, "data", "attention", "1 of 105 sessions can be priced")
    I.stage(it, "follow_up")
    assert [s["stage"] for s in it["stages"]] == ["understand", "validate", "data", "follow_up"] and I.STAGES == ["understand", "validate", "data", "plan", "execute", "explain", "follow_up"]
    with pytest.raises(ValueError):
        I.stage(it, "guess")
    assert read(ag, "What have we found so far?")["kind"] == "action" and read(ag, "Buy when the stars align.")["kind"] in ("unreadable", "strategy", "action")
