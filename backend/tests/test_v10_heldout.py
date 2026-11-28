"""Regressions from the V10 held-out run (docs/V10_REPORT.md, "The held-out result").

The held-out set was scored once, on 2026-11-27, and read by the developer only afterwards. Each test here is a defect
that run or its numerical audit exposed. They are regressions, not a measure: nothing here says how fresh questions are read.
"""
from __future__ import annotations

import pytest

from strata.ai import answer as ANS
from strata.ai import intent as I
from strata.ai import semantics as SM
from tests.test_v10 import KNOWN, _confirming, ag, draft_of, market, post, route, session  # noqa: F401  (fixtures)


def test_a_condition_the_study_engine_cannot_express_is_never_left_out():
    # sam/sh-02: "when it is below the open at 12:00: how often it closes above its open" was counted on all 2,741 sessions
    f = SM.read(I.tidy("when SPY is below its open at noon ET, how often does it close above its open?"), KNOWN, "SPY")
    assert [c["key"] for c in f.frame["conditions"]] == ["rel"]
    r = SM.to_route(f)
    assert r["action"] == "profile" and r["frame"]["outcome"] == {"key": "rel", "rel": {"a": "close", "op": ">", "b": "open"}, "text": r["frame"]["outcome"]["text"]}, "run by the engine that applies the condition"
    assert r["frame"] is f.frame, "the frame that ran is the frame that is shown"
    with pytest.raises(ValueError, match="cannot apply the condition"):
        SM.study_draft({"kind": "frequency", "symbol": "SPY", "conditions": [{"key": "rel", "rel": {"a": "at:12:00", "op": "<", "b": "open"}}], "outcome": {"key": "touch", "value": 1.0, "ref": "open"}})
    g = {"kind": "frequency", "symbol": "SPY", "text": "t", "conditions": [{"key": "rel", "rel": {"a": "at:12:00", "op": "<", "b": "open"}}], "outcome": {"key": "touch", "value": 1.0, "ref": "open"}, "assumptions": []}
    r2 = SM.to_route(SM.Reading(text="t", frame=g, mentions=[]))
    assert r2["action"] == "clarify" and "cannot be combined with this measurement" in r2["text"], "said, not dropped"


def test_an_older_readers_empty_study_is_not_a_reading(ag):
    # riley/rh-01, rh-04, alex/ah-15, ah-18, jordan/jh-21: a restatement nobody could read ran as an unconditioned study and
    # was answered "the outcome happened on 0 of N days"
    r = route(ag, "For IWM since 2000, for each calendar month give the average percent return from the last close of the prior month to the last close of the month, ranked by the average.")
    assert r["action"] in ("unknown", "clarify") and r.get("draft") is None
    res = {"symbol": "SPY", "stats": {"qualifying_sessions": 100, "target_first": 0, "probability": 0.0, "avg_end_return_pct": 0.02, "median_end_return_pct": 0.05, "pct_positive_end": 0.54}}
    from strata.ai.agent_v8 import ResearchV8  # noqa: F401
    content = {"view": "single"}
    ag._v8_answer("result", {}, content, [{"run": {"id": "r", "kind": "study", "config": {"target": None}}, "result": res}])
    assert content["answer"]["kind"] == "movement" and "the outcome happened" not in content["answer"]["headline"] and "moved +0.02% on average" in content["answer"]["headline"]


def test_n_sessions_later_is_measured_from_the_close_of_the_day_the_condition_was_met():
    # morgan/mh-18, jordan/jh-16: the reading said "from that session's close", the calculation started at the next open
    f = SM.read(I.tidy("spy closes down 2%+ on the day. where is it 5 trading days later on avg, and how often higher. all history"), KNOWN, "SPY").frame
    assert f["kind"] == "movement" and f["end"] == "sessions:5" and f["outcome"].get("ref") == "prev_close"
    d = SM.study_draft(f)
    assert (d.study.reference, d.study.horizon_kind, d.study.horizon_sessions) == ("prev_close", "sessions", 4), "the observation session is the day after: 4 more sessions end 5 after the fall"
    a = ANS.movement({"symbol": "SPY", "stats": {"qualifying_sessions": 341, "avg_end_return_pct": 0.79, "median_end_return_pct": 0.69, "pct_positive_end": 0.595}}, f, "r")
    assert "from that day's close to the close 5 sessions later" in a["headline"]
    # alex/ah-24: with a horizon in sessions there is one reading of the outcome, labelled with the horizon
    g = SM.read(I.tidy("after 4 down days in a row does IWM usually bounce 5 sessions later"), KNOWN, "SPY").frame
    opts = SM.options_of(g)
    assert len(opts) == 1 and opts[0]["label"] == "it closes above the close of the day the condition was met, 5 sessions later" and "measured separately" not in SM.describe(g)


def test_a_position_strata_cannot_test_is_declined_not_run_as_another(ag):
    # alex/ah-21, morgan/mh-14, jordan/jh-02, riley/rh-14: a call butterfly was run as one long call
    r = route(ag, "buy a SPY call butterfly with 5-dollar wings at 1pm ET 0dte, out at 3:50 pm ET")
    assert r["action"] == "clarify" and r["declined"] and "\"call butterfly\" is not a position Strata can backtest" in r["text"] and "iron butterflies" in r["text"]
    assert route(ag, "sell a SPY iron butterfly 0dte at 10 ET, 5 wide, hold to expiration")["draft"].structure == "iron_butterfly"
    sid = session()
    post(ag, sid, "buy SPY 0dte atm calls at 10 ET, exit at 3 pm ET")
    m = post(ag, sid, "make it a butterfly instead")
    assert m["content"]["declined"] and draft_of(ag, sid).structure == "long_call"


def test_a_correction_to_a_waiting_reading_waits_again(ag):
    # 20 of the 41 silent failures: a correction fixed one part of a reading that was wrong in several, and it ran
    _confirming()
    sid = session()
    post(ag, sid, "when vix is above 15 how often does qqq close green after a gap down since 2024")
    m = post(ag, sid, "use SPY instead")
    assert m["content"]["confirm"] == {"reasons": [], "complex": True} and m["content"]["text"].startswith("Changed QQQ to SPY") and "Nothing was run" in m["content"]["text"]
    st = ag._state(sid)[0]["state"]
    assert st.get("frame") is None and st["pending"]["frame"]["symbol"] == "SPY"
    ag.post(sid, "yes")
    assert ag._state(sid)[0]["state"]["frame"]["symbol"] == "SPY"
    sid = session()
    post(ag, sid, "sell a 5 wide SPY put credit spread 0dte at 10:30 ET if SPY is above VWAP, 20 delta, take profit at 50% of the credit, hold to expiration")
    assert ag._plan_message(sid, ag._state(sid)[1]["id"])["content"]["state"] == "awaiting_confirmation"
    post(ag, sid, "make it 2 contracts")
    v = ag._state(sid)[1]
    assert draft_of(ag, sid).quantity == 2 and ag._plan_message(sid, v["id"])["content"]["state"] == "awaiting_confirmation" and "has not run yet" in draft_of(ag, sid).review[0]


def test_days_the_conversation_does_not_hold_are_asked_for(ag):
    # riley/rh-23: "buy a put at 10:30 on those mornings" after a question that was never answered ran on all 711 sessions
    r = route(ag, "Suppose I had bought an at-the-money SPY put expiring that day at 10:30 ET on those mornings and sold it at 3:00 pm ET, two contracts each time?")
    assert r["action"] == "clarify" and "points at the days of an earlier question" in r["text"] and r.get("draft") is None


def test_sentences_the_judges_found_wrong_are_not_said():
    from strata.ai.draft import Draft, diff_drafts
    a = Draft(kind="options", symbol="SPY", structure="long_call")
    b = a.model_copy(deep=True)
    b.review = ["check it"]
    assert diff_drafts(a, b) == [], "why a reading waits is not a change to it (a comparison was labelled 'Changed review, start')"
    res = {"symbol": "SPY", "stats": {"qualifying_sessions": 8515, "target_first": 3881, "probability": 0.456, "probability_ci95": [0.445, 0.466], "baseline_probability": None}}
    txt = " ".join(ANS.frequency(res, {"kind": "frequency", "symbol": "SPY", "outcome": {"key": "close_vs", "ref": "open", "direction": "up"}}, "r").get("happened") or [])
    assert "how few days" not in txt and "8,515 days" in txt
    trades = [{"pnl": -500.0, "entry_cash": 120.0, "units": 1, "max_risk": 43121.6, "commissions": 1.3}, {"pnl": 80.0, "entry_cash": 110.0, "units": 1, "max_risk": 40000.0, "commissions": 1.3}]
    run = {"id": "r", "config": {"strategies": [{"legs": [{"side": "sell", "kind": "option", "right": "C"}], "sizing": {}}]}}
    lines = " ".join(ANS.sizing_story(run, {"config": run["config"]}, trades, None)["lines"])
    assert "no fixed maximum loss" in lines and "distance between the strikes" not in lines, "a single short call has one strike and no such cap"
