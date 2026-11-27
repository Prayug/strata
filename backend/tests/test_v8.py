"""V8: understanding the question, keeping the conversation, answering first, and showing what every number is made of.

The sentences here are development material (the brief's own examples and sentences found while building). They say
nothing about sentences nobody has written yet: that is what the sealed set (strata/ai/evalset_v8.py) is for, and this
file only checks that it has not been edited.
"""
from __future__ import annotations

import pytest

from strata.ai import answer as ANS
from strata.ai import converse as CV
from strata.ai import semantics as SM
from strata.ai import teach as TEACH
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v3_agent import agent, session, trade_store
from tests.test_v7_questions import store  # noqa: F401  (fixture: 80 sessions whose first candle and rest of day are known)

KNOWN = ["SPY", "QQQ", "IWM", "TST", "VIX"]


def read(text: str, sym: str = "SPY") -> SM.Reading:
    return SM.read(text, KNOWN, sym)


# ----------------------------------------------------------------------------- the sealed set
def test_the_sealed_set_is_the_one_that_was_written():
    from strata.ai import evalset_v8 as E
    assert E.sealed_hash() == "be3d93236350fdff99e1367b81dfbcbdaf79c5b8290fe55afed71ce89a1a464c", "the sealed V8 set must not change after it was written"
    assert len(E.SEALED) == 52 and len(E.SEALED_CONVERSATIONS) == 8


# ----------------------------------------------------------------------------- reading one sentence
def test_the_first_candle_question_measures_the_whole_day_and_the_rest_of_it():
    r = read("Is the first five-minute candle a good predictor of the day?")
    f = r.frame
    assert r.complete and f["kind"] == "agreement" and f["candle"] == [5] and f["scope"] in ("day", "both")
    plan = SM.agreement_plan(f)
    assert {(p["minutes"], p["scope"]) for p in plan["pairs"]} == {(5, "day"), (5, "rest")}, "the whole day alone would credit the candle for its own move"
    r = read("How often is SPY's first candle right about the rest of the day?")
    assert r.frame["scope"] == "rest" and r.frame["candle"] == [5, 15, 30] and any("not stated" in a["text"] for a in r.frame["assumptions"])


@pytest.mark.parametrize("text,conds,outcome,end", [
    ("How often does SPY finish higher on the day following a decline of 2.5% or more?", [("prev_return", "<=", -2.5)], ("close_vs", "prev_close"), "close"),
    ("how often does spy drop 1% from the open before noon eastern", [], ("touch", None), "12:00"),
    ("How often is SPY up more than 2% five sessions after it falls 5% in a day?", [("prev_return", "<=", -5.0)], ("touch", "prev_close"), "sessions:5"),
    ("How often does QQQ rise 2% within 5 days after falling more than 1%?", [("prev_return", "<=", -1.0)], ("touch", "start"), "sessions:5"),
    ("how often does qqq drop 1% from the open and still close green", [("from_open", "<=", -1.0)], ("close_vs", "open"), "close"),
    ("after 3 red days in a row how often is the next day green for spy", [("down_streak", ">=", 3.0)], ("close_vs", "prev_close"), "close"),
])
def test_every_part_of_the_question_has_a_place(text, conds, outcome, end):
    r = read(text)
    f = r.frame
    assert r.ok and not r.unread and not r.unclaimed, (text, r.unread, [m.text for m in r.unclaimed])
    assert [(c["key"], c["op"], c["value"]) for c in f["conditions"]] == conds
    assert f["outcome"]["key"] == outcome[0] and (outcome[1] is None or f["outcome"].get("ref", f["outcome"].get("level")) == outcome[1])
    assert str(f.get("end")) == end


def test_a_symbol_inside_a_condition_is_not_the_subject():
    f = read("If VIX is above 25, how often does SPY fall more than 1% that day?").frame
    assert f["symbol"] == "SPY" and [(c["key"], c["value"]) for c in f["conditions"]] == [("vix", 25.0)]


def test_recover_is_measured_in_each_of_its_readings_not_guessed():
    r = read("Does QQQ usually recover after falling 1%?")
    f = r.frame
    assert r.ok and len(f["outcome_options"]) == 2 and any("can mean more than one thing" in a["text"] for a in f["assumptions"])
    route = SM.to_route(r)
    assert route["action"] == "investigate" and len(route["plan"]["variants"]) == 2
    gap = read("Does SPY recover after gapping down 1%?").frame
    assert gap["outcome_options"][0]["outcome"] == {"key": "touch_level", "level": "prev_close", "direction": "up", "text": "recover"}, "after a gap, recovering first means the gap fills"


def test_words_that_were_not_understood_stop_the_run():
    r = read("How often does SPY rise 1% from the open when retail sentiment is euphoric?")
    assert [u["text"] for u in r.unread] == ["retail", "sentiment", "euphoric"] and not r.complete


def test_what_cannot_be_answered_is_said_rather_than_substituted():
    assert read("will spy go up tomorrow").flag
    assert read("how often does spy rise after an fomc announcement").flag
    assert read("how does TSLA trade after earnings").flag  # no TSLA history and no real earnings calendar: either is said, nothing is run


# ----------------------------------------------------------------------------- the conversation
def test_the_brief_conversation_edits_one_investigation(store):
    ag, sid = agent(store), session()
    m = ag.post(sid, "How often is TST's first candle right about the rest of the day?")[-1]
    assert m["kind"] == "investigation"
    ag.finalize(sid)
    steps = [("Use 5 min.", "Changed the candle length"), ("What about when yesterday was red?", "Added: the previous session closed down."),
             ("Nah, I meant down at least 1%.", "the previous session fell 1% or more")]
    for text, said in steps:
        out = ag.post(sid, text)
        first = next(x for x in out if x["role"] == "assistant")
        ch = (first["content"].get("change") or {}).get("text") or ""
        assert said in ch and ch.endswith("Everything else stays the same."), (text, ch)
    st = ag._state(sid)[0]["state"]
    assert st["frame"]["candle"] == [5] and [(c["key"], c["value"]) for c in st["frame"]["conditions"]] == [("prev_return", -1.0)]
    ag.post(sid, "undo")
    assert [(c["key"], c["value"]) for c in ag._state(sid)[0]["state"]["frame"]["conditions"]] == [("prev_return", 0.0)], "undo goes back exactly one change"


def test_a_named_condition_is_edited_not_the_outcome(tmp_settings):
    ag, sid = agent(trade_store(tmp_settings)), session()
    ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    ag.finalize(sid)
    m = ag.post(sid, "Change the previous-day decline to 3.8% and compare the results.")[-1]
    c = m["content"]
    assert m["kind"] == "plan" and c["draft"]["conditions"][0]["expr"] == "previous_session.return_pct <= -3.8" and c["compare_with"]
    assert c["change"]["text"].startswith("Now: the previous session fell 3.8% or more (was 3%).")
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "compare" and res["answer"]["headline"].startswith("When the previous session fell 3.8% or more, it happened more often")


def test_a_comparison_request_on_its_own_is_not_an_edit():
    from strata.ai.agent_v8 import bare_compare
    assert bare_compare("Compare both.") and bare_compare("compare them side by side")
    assert not bare_compare("Change the previous-day decline to 3.8% and compare the results.") and not bare_compare("Compare the results with 7DTE calls.")


def test_turning_the_question_into_a_strategy_says_what_was_assumed(store):
    ag, sid = agent(store), session()
    ag.post(sid, "Is TST's first 5-minute candle a good predictor of the day?")
    ag.finalize(sid)
    first = next(x for x in ag.post(sid, "What if I bought calls instead?") if x["role"] == "assistant")
    c = first["content"]
    assert c["draft"]["kind"] == "options" and c["draft"]["structure"] == "long_call" and c["draft"]["entry_time"] == "09:35"
    assert c["change"]["lines"] == ["Turned the question into an option strategy, with the same conditions."], "one change, not a list of every slot"
    assert any("09:35" in a for a in c["assumed"])


# ----------------------------------------------------------------------------- answers
def test_the_answer_leads_with_the_rest_of_the_day_and_a_measured_baseline(store):
    ag, sid = agent(store), session()
    ag.post(sid, "Is the first five-minute candle a good predictor of the day for TST?")
    a = ag.finalize(sid)[0]["content"]["answer"]
    assert a["kind"] == "agreement"
    assert "whole day 80.0% of the time" in a["headline"] and "only 50.0% of the time" in a["headline"], a["headline"]
    m = next(r for r in a["viz"]["rows"] if r["scope"] == "rest" and r["minutes"] == 5)
    assert (m["up_up"], m["up_down"], m["down_down"], m["down_up"]) == (20, 20, 20, 20) and abs(m["always_up"] - 0.5) < 1e-9
    assert all(n.get("explore") for n in a["numbers"] if n["label"] != "If unrelated"), "every count opens the days behind it"


def test_a_gap_fill_is_not_compared_with_days_that_had_no_gap():
    res = {"stats": {"qualifying_sessions": 100, "target_first": 41, "ambiguous": 0, "probability": 0.41, "probability_ci95": [0.32, 0.51], "baseline_probability": 0.67, "baseline_sessions": 8000,
                     "p_value_vs_rest": 0.0001}, "symbol": "SPY", "period": ["2016-01-04", "2026-11-25"]}
    f = read("how often does spy fill the gap when it gaps down more than 0.5%?").frame
    a = ANS.frequency(res, f, "r1")
    assert "any day" not in a["headline"] and a["viz"]["baseline"] is None and a["matters"] is None
    assert any("no fair" in h for h in a["happened"])


def test_statistics_are_worded_correctly():
    assert "not the probability" in ANS.TERMS["p_value"]["meaning"]
    rob = {"level": "none", "checks": [{"name": "Sample size", "ok": True, "detail": "50 trades"}], "verdict": ""}
    a = ANS.reliability(rob, 252)
    assert a["headline"].startswith("It may just have been luck") and any("We tried 252 variations" in x for x in a["limits"])
    assert any("not the probability that the strategy works" in x for x in a["limits"])


def test_versions_are_named_by_what_differs():
    a = {"symbol": "SPY", "structure": "long_call", "dte": 0, "exit_time": "15:55", "conditions": []}
    b = {**a, "exit_time": "14:55"}
    assert ANS.draft_phrases(a, b) == ("exit at 3:55 PM ET", "exit at 2:55 PM ET")
    assert CV.summary_text(CV.summary({"kind": "frequency", "symbol": "SPY"}, {"kind": "frequency", "symbol": "QQQ"})) == "Changed SPY to QQQ. Everything else stays the same."


# ----------------------------------------------------------------------------- explore, teaching
def test_the_days_behind_a_number_and_what_selected_days_share(tmp_settings):
    from strata.research import explore as X
    st = trade_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    res = ag.finalize(sid)[0]["content"]
    spec = next(n["explore"] for n in res["answer"]["numbers"] if n["label"] == "Days it happened")
    t = X.observations(spec, st)
    assert t["kind"] == "sessions" and t["count"] == 1 and all(r["happened"] == "yes" for r in t["rows"])
    ag.post(sid, "Buy ATM TST 0DTE calls at the open after a 3% down day using option trade prices, exit at 15:55")
    run = ag.finalize(sid)[0]["content"]["run_id"]
    tr = X.observations({"kind": "trades", "run_id": run, "filter": "all"}, st)
    assert tr["count"] == 1 and tr["rows"][0]["multiplier"] == 100 and "one contract is 100 shares" in tr["caption"]


def test_teaching_answers_come_from_the_runs_own_trades(tmp_settings):
    st = trade_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "Buy ATM TST 0DTE calls at the open after a 3% down day using option trade prices, exit at 15:55")
    ag.finalize(sid)
    for text, topic in (("Explain this like I'm new to options", "new_to_options"), ("Why did the option lose value if SPY went up?", "why_lost_value"), ("What does this chart mean?", "chart")):
        assert TEACH.topic_of(text) == topic
        m = ag.post(sid, text)[-1]
        assert m["kind"] == "result" and m["content"]["view"] == "explain" and m["content"]["answer"]["topic"] == topic
    a = ag.post(sid, "Explain this like I'm new to options")[-1]["content"]["answer"]
    assert any("One contract covers 100 shares" in x for x in a["happened"]) and a["illustration"]["illustration"] is True


def test_a_loss_on_a_wrong_way_move_is_not_blamed_on_time_decay():
    t = {"id": 1, "symbol": "QQQ", "entry_date": "2026-06-10", "entry_time": "2026-06-10T09:45:00-04:00", "entry_spot": 707.10, "exit_spot": 695.93, "pnl": -481.3, "gross_pnl": -480.0,
         "commissions": 1.3, "units": 1, "exit_reason": "time_exit", "legs": [{"kind": "option", "strike": 707.0, "right": "C", "entry_price": 4.81, "exit_price": 0.01, "multiplier": 100, "expiration": "2026-06-10"}]}
    cause = " ".join(ANS.why_lost(t)["cause"])
    assert "moved against the position" in cause and "time value running out" not in cause


# ----------------------------------------------------------------------------- the measured sets
def test_the_development_sets_are_read_without_silent_errors():
    """A regression line: these sets were used while building (see the module docstring of strata/ai/evalset_v8). They are
    read against the stored market data (SPY, QQQ, IWM), so this runs only where that data is: not on a fresh checkout."""
    from strata.ai import eval_v8 as EV
    from strata.config import get_settings
    from strata.data.store import MarketStore
    st = MarketStore(get_settings().data_dir / "market")
    if not {"SPY", "QQQ", "IWM"} <= set(st.stored_symbols()) or st.is_synthetic("SPY"):
        pytest.skip("the development sets are read against real stored SPY/QQQ/IWM history, which this data directory does not have")
    for which in ("development", "probe"):
        s = EV.evaluate(which, store=st, answers=False)["summary"]
        assert s["single"]["silent"] == 0 and s["single"]["correct"] == s["single"]["n"], (which, s)
        assert s["conversation_turns"]["silent"] == 0, (which, s)
