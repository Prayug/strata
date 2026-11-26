"""V7: what was typed into the running application, and answered wrongly.

On 2026-11-26 the application's own user asked it, three times in different words, how often the first candle of the
session predicts the day. Each time it ran a study with no condition and no outcome and reported the share of ALL sessions
that closed higher: a number for a question nobody had asked, with nothing flagged. The first attempt also read "9:30" as
Pacific time, and the correction that followed ("no I meant 6:30a, PST") got the list of example prompts for an answer.

These tests hold the sentences as they were typed. They are not part of the sealed set (strata/ai/evalset_v7.py), and
nothing here says anything about sentences nobody has typed yet.
"""
from __future__ import annotations

from datetime import date

import pytest

from strata.ai import nl_rules as NL
from strata.ai import research as RS
from strata.ai.agent import Agent
from strata.ai.draft import compile_draft
from strata.ai.nl_stateful import enrich
from strata.core import calendar as cal
from strata.data.store import MarketStore, SymbolMeta
from strata.research.study import run_study
from tests.helpers import write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture: a temporary data directory)

TYPED = ["How often is SPY's overall day, predicted by the first 9:30 open candle?",
         "How often is SPY's overall day, predicted by the first 9:30ET open candle?",
         "How often does SPY's first 9:30ET candle, predict its overall day?"]

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 8, 30))[:80]
# what each session does: (price at the end of the first five minutes, the close), from an open of 100
KINDS = {"A": (100.2, 100.5),   # candle up,   session up,   rest up
         "B": (100.2, 100.1),   # candle up,   session up,   rest DOWN: the whole session agrees only because the candle is in it
         "C": (100.2, 99.5),    # candle up,   session down, rest down
         "D": (99.8, 99.5),     # candle down, session down, rest down
         "E": (99.8, 99.9),     # candle down, session down, rest UP
         "F": (99.8, 100.5)}    # candle down, session up,   rest up
ORDER = ("A" * 20 + "B" * 12 + "C" * 8 + "D" * 20 + "E" * 12 + "F" * 8)
KIND = {d: ORDER[(i * 7) % 80] for i, d in enumerate(DAYS)}  # 7 and 80 share no factor: every letter keeps its count, spread over the weeks


PREV = cal.prev_trading_day(DAYS[0])  # a flat session before the first: a study does not evaluate the first session it holds


def price(d: date, i: int) -> float:
    first, close = KINDS[KIND[d]] if d in KIND else (100.0, 100.0)
    return 100.0 if i < 0 else first if i < 300 else close  # the candle's direction is set in the first minute and held


@pytest.fixture
def store(tmp_settings):
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="TST", kind="etf", multiplier=100.0))
    write_minutes(st, "TST", [PREV] + DAYS, price)
    return st


def route(store, text: str, current=None):
    return Agent(None, store)._rules(text, current, ["TST", "SPY", "QQQ", "IWM"], None, "TST")


# ----------------------------------------------------------------------------- the question
@pytest.mark.parametrize("text", TYPED)
def test_the_question_as_it_was_typed_is_a_comparison_of_two_directions(text, store):
    fc = RS.first_candle(text)
    assert fc == {"minutes": [5, 15, 30], "stated": False, "scope": "both"}, "no candle length was named, so none is assumed: three are measured"
    r = route(store, text)
    assert r["action"] == "investigate", "not a study with nothing in it"
    plan = r["plan"]
    from strata.ai.agent import MAX_VARIANTS
    assert plan["symbol"] == "SPY" and plan["kind"] == "first_candle" and len(plan["variants"]) == 12 <= MAX_VARIANTS and len(plan["pairs"]) == 6, "every planned study is run: none is cut off by the bound"
    assert {(p["minutes"], p["scope"]) for p in plan["pairs"]} == {(m, s) for m in (5, 15, 30) for s in ("day", "rest")}
    assert any("was not stated" in n and "5-, 15- and 30-minute" in n for n in plan["interpretation_notes"])
    assert any("measured twice" in n and "part of it" in n for n in plan["interpretation_notes"])
    # every variant is an ordinary study: the first N minutes against the open, then the session or the rest of it
    by = {v["label"]: compile_draft(v["draft"], store)["study"] for v in plan["variants"]}
    up, rest_dn = by["first 5 min up: session closes above its open"], by["first 5 min down: close is below the 09:35 price"]
    assert up["entry_time"] == "09:35" and up["reference"] == "open" and up["measure"] == "close" and up["direction"] == "up" and up["target"]["value"] == 0.0
    assert up["condition"]["children"][0]["params"]["expr"] == "from_open_pct > 0"
    assert rest_dn["entry_time"] == "09:35" and rest_dn["reference"] == "entry" and rest_dn["direction"] == "down" and rest_dn["condition"]["children"][0]["params"]["expr"] == "from_open_pct < 0"


def test_other_wordings_and_what_is_left_alone(store):
    assert RS.first_candle("How often does the first 15-minute candle predict the rest of the session for QQQ?") == {"minutes": [15], "stated": True, "scope": "rest"}
    assert RS.first_candle("Does the first hour set the tone for the day?")["minutes"] == [60]
    assert RS.first_candle("Is the opening 5m candle a good predictor of the daily close?") == {"minutes": [5], "stated": True, "scope": "both"}
    assert RS.first_candle("How often does the first half hour predict the close?")["minutes"] == [30]
    assert RS.first_candle("What does the first 30 minutes tell us about the rest of the day?") == {"minutes": [30], "stated": True, "scope": "rest"}
    assert route(store, "Does the opening candle predict the day?")["action"] == "investigate"
    assert route(store, "Does the first 15-minute candle predict the day?")["plan"]["pairs"][0]["minutes"] == 15
    # a strategy that mentions the first candle is a strategy
    assert RS.first_candle("Buy QQQ puts when the first 15-minute candle breaks the previous day's low.") is None
    assert RS.first_candle("Buy SPY calls when the first 5 minute candle is green and matches the previous day") is None
    # a candle and no word for agreement or prediction is not this question
    assert RS.first_candle("How big is the first 5 minute candle on average?") is None
    # a frequency question the readers find nothing to count in is not run as a study of nothing
    r = route(store, "How often does SPY do the thing with the wick?")
    assert r["action"] == "clarify" and "could not find anything to count" in r["text"] and "nothing was run" in r["text"]
    assert route(store, "How often does TST rise 1% from the open after a 3% down day?")["action"] == "new"


def test_agreement_is_counted_from_the_sessions_and_the_candle_is_not_credited_for_itself(store):
    """80 sessions made so that the answer is known: the first five minutes and the whole session agree on 64 of them, and
    on 24 of those the agreement is nothing but the candle (the rest of the session went the other way). After the candle,
    its direction matches the rest of the session on 40 of 80: exactly what unrelated directions give."""
    plan = RS.first_candle_plan("How often does the first 5 minute candle predict the day for TST?", "TST", {"minutes": [5], "stated": True, "scope": "both"})
    entries = []
    for i, v in enumerate(plan["variants"]):
        c = compile_draft(v["draft"], store)
        assert not c["errors"]
        entries.append({"label": v["label"], "run": {"kind": "study", "id": f"r{i}"}, "result": run_study(c["study"], store), "mode": "study"})
    rep = RS.investigation_report(plan, entries, len(entries))
    day, rest = (next(t for t in rep["agreement"] if t["scope"] == s) for s in ("day", "rest"))
    assert (day["sessions"], day["agree"], day["agreement"]) == (80, 64, 0.8) and day["if_unrelated"] == pytest.approx(0.5)
    assert (day["up"]["n"], day["up"]["probability"], day["down"]["n"], day["down"]["probability"]) == (40, 0.8, 40, 0.8)
    assert (rest["sessions"], rest["agree"], rest["agreement"]) == (80, 40, 0.5) and rest["if_unrelated"] == pytest.approx(0.5) and rest["distinguishable"] is False
    assert (rest["up"]["probability"], rest["down"]["probability"]) == (0.5, 0.5) and rest["from"] == "09:35"
    assert rep["level"] == "none", "what the candle is credited with is the rest of the session, and that is a coin"
    v = rep["verdict"]
    assert "matched the direction of the whole session (open to close) on 80.0% of sessions" in v and "against 50.0% if the two were unrelated" in v
    assert "Most of that is the candle itself" in v and "From the end of the candle to the close it matched on 50.0%" in v and "no more than chance produces" in v
    assert rep["findings"] == ["First 5 minutes: matched the whole session (open to close) on 80.0% of 80 sessions (95% interval 70.0% to 87.3%; 50.0% if unrelated); "
                               "matched the rest of the session (09:35 to the close) on 50.0% of 80 sessions (95% interval 39.3% to 60.7%; 50.0% if unrelated)."]
    assert any("not a forecast" in x and "inside the open-to-close move" in x for x in rep["limitations"])
    assert any(f"80 of them, {DAYS[0]} to {DAYS[-1]}" in x for x in rep["limitations"])
    assert len(rep["rows"]) == 4 and all(r["hits"] is not None for r in rep["rows"]), "the four studies stay inspectable one by one"
    # only the rest of the session, when that is what was asked
    only = RS.first_candle_plan("x", "TST", {"minutes": [5], "stated": True, "scope": "rest"})
    assert [p["scope"] for p in only["pairs"]] == ["rest"] and len(only["variants"]) == 2


def test_a_candle_that_closes_at_the_open_has_no_direction_and_nothing_is_put_in_its_place(tmp_settings):
    from tests.test_v3_agent import trade_store
    st = trade_store(tmp_settings)  # eight sessions of minute bars whose first half hour is flat
    plan = route(st, "How often does TST's first 9:30 candle predict its overall day?")["plan"]
    entries = [{"label": v["label"], "run": {"kind": "study", "id": f"r{i}"}, "result": run_study(compile_draft(v["draft"], st)["study"], st), "mode": "study"} for i, v in enumerate(plan["variants"])]
    rep = RS.investigation_report(plan, entries, len(entries))
    assert rep["agreement"] == [] and rep["level"] == "insufficient"
    assert rep["verdict"] == "On the 7 TST sessions with intraday bars the first candle closed exactly at the session's open every time, so it has no direction to compare with the day's."
    assert all(r["n"] == 0 and r["evaluated"] == 7 for r in rep["rows"]), "the sessions were observed; none qualified"


# ----------------------------------------------------------------------------- the clock, and the correction
def test_clock_times_as_people_type_them():
    t = lambda s: [(x["et"], x["stated"]) for x in NL.find_times(s, None)]
    assert t("no I meant 6:30a, PST") == [("09:30", "6:30a, PST")]
    assert t("enter at 6:45a PST") == [("09:45", "6:45a PST")]
    assert t("exit at 12:45p Pacific") == [("15:45", "12:45p Pacific")]
    assert t("first 9:30ET candle") == [("09:30", "9:30ET")]
    # 9:30 with no zone is the opening bell, whatever zone the reader's settings would assume for other times
    assert t("the first 9:30 open candle")[0][0] == "09:30"
    # a comma after a time is not part of it; "a" is a letter of the time only when it touches the digits
    assert t("buy at 10:00, exit at 15:30") == [("10:00", "10:00 (Eastern assumed)"), ("15:30", "15:30 (Eastern assumed)")]
    assert t("at 9:30 a trade is opened")[0] == ("09:30", "9:30 (Eastern assumed)")
    assert t("enter at 6:30 AM Pacific time and exit at 12:45 PM Pacific") == [("09:30", "6:30 AM Pacific time"), ("15:45", "12:45 PM Pacific")]


def test_what_counts_as_a_correction():
    assert NL.correction("no I meant 6:30a, PST") == "6:30a, PST"
    assert NL.correction("No, 2%") == "2%" and NL.correction("sorry, I meant puts") == "puts" and NL.correction("actually make it 1DTE") == "1DTE"
    for not_one in ("How often does SPY rise 1%?", "Now compare 0DTE and 7DTE", "note the stop", "Buy calls at the open", "no", "compare that with 2% down days"):
        assert NL.correction(not_one) is None, not_one


def test_a_value_on_its_own_replaces_the_one_thing_it_can(store):
    study = route(store, "How often does TST rise 1% from the open after a 3% down day?")["draft"]
    calls = enrich("Buy ATM TST 0DTE calls at 10:00 AM Eastern after a 1% down day", ["TST"], "TST")
    both = enrich("Buy ATM TST 0DTE calls at 10:00 AM Eastern after a 1% down day, exit at 3:55 PM Eastern", ["TST"], "TST")
    # the correction that was typed: a clock, with its zone, becomes the time the observation starts
    r = route(store, "no I meant 6:45a, PST", study)
    assert r["action"] == "modify" and r["changed"] == ["entry_time"] and (r["draft"].entry_time, r["draft"].entry_time_stated) == ("09:45", "6:45a, PST")
    r = route(store, "no I meant 6:30a, PST", calls)
    assert r["action"] == "modify" and r["draft"].entry_time == "open" and r["draft"].entry_time_stated == "6:30a, PST"
    # an entry time and an exit time are both stated: the clock could be either, so nothing is changed and the reply says how to say which
    r = route(store, "no I meant 6:45a PST", both)
    assert r["action"] == "clarify" and "could replace either" in r["text"] and '"enter at 6:45a PST" or "exit at 6:45a PST"' in r["text"]
    # one percentage in the reading: it is replaced. Two: nothing is, and both are named
    r = route(store, "no, 2%", calls)
    assert r["action"] == "modify" and r["changed"] == ["prev_ret"] and r["draft"].conditions[0].expr == "previous_session.return_pct <= -2"
    r = route(store, "no I meant 2%", study)
    assert r["action"] == "clarify" and "could replace more than one thing" in r["text"] and "previous session fell 3% or more" in r["text"] and "the 1% target" in r["text"] and "Nothing was changed" in r["text"]
    assert route(store, "actually make it puts", calls)["draft"].structure == "long_put"
    assert route(store, "sorry, 1DTE", calls)["draft"].dte == 1
    assert route(store, "no I meant QQQ", calls)["draft"].symbol == "QQQ"
    # words that say what they change go to the readers as they always did
    stop = enrich("Buy ATM TST 0DTE calls at the open after a 1% down day with a 20% premium stop", ["TST"], "TST")
    r = route(store, "no stop loss", stop)
    assert r["action"] == "modify" and r["draft"].premium_stop_pct is None
    r = route(store, "actually enter at 10:15 AM Eastern", calls)
    assert r["action"] == "modify" and r["draft"].entry_time == "10:15"
    # a whole new request after "no," is the request
    r = route(store, "no, buy QQQ puts at 10:00 AM Eastern after a 2% up day", calls)
    assert r["action"] == "new" and r["draft"].symbol == "QQQ" and r["draft"].structure == "long_put"
    # a correction nothing can place says so in terms of what was typed, instead of listing example prompts
    r = route(store, "no I meant something else entirely", calls)
    assert r["action"] == "clarify" and 'I took "something else entirely" as a correction' in r["text"] and "nothing was changed" in r["text"]


def test_changing_the_entry_time_in_words_keeps_the_clock(store):
    """ "change the entry time to 6:45 AM Pacific" set the entry to "6.0": the number after "to" was taken for the value."""
    both = enrich("Buy ATM TST 0DTE calls at 10:00 AM Eastern after a 1% down day, exit at 3:55 PM Eastern", ["TST"], "TST")
    r = route(store, "change the entry time to 6:45 AM Pacific", both)
    assert r["action"] == "modify" and r["changed"] == ["entry_time"] and (r["draft"].entry_time, r["draft"].entry_time_stated, r["draft"].exit_time) == ("09:45", "6:45 AM Pacific", "15:55")
    study = route(store, "How often does TST rise 1% from the open after a 3% down day?")["draft"]
    r = route(store, "observe from 10:00 AM Eastern instead", study)
    assert r["action"] == "modify" and r["draft"].entry_time == "10:00"
