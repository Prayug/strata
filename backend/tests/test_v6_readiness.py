"""V6: research readiness is counted from stored files, and a strategy is checked against them before anything is planned."""
from __future__ import annotations

from datetime import date

from strata.data import readiness as RD
from strata.research import usecase as U
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v3_agent import agent, session
from tests.test_v6_usecase import DAYS, S3, S6, S9, S11, store  # noqa: F401  (fixture: 13 sessions, 4 after a fall of 3% or more)


def test_readiness_counts_what_is_stored_and_nothing_a_provider_advertises(store):
    r = RD.symbol_readiness(store, "SPY")
    c = {x["id"]: x for x in r["capabilities"]}
    assert [x["id"] for x in r["capabilities"]] == ["daily", "intraday", "eod_options", "option_trades", "option_quotes"]
    assert (c["daily"]["sessions"], c["intraday"]["sessions"], c["eod_options"]["sessions"], c["option_trades"]["sessions"], c["option_quotes"]["sessions"]) == (13, 13, 4, 0, 2)
    assert c["option_trades"]["ready"] is False and c["option_quotes"]["ready"] is True and c["option_quotes"]["first"] == S3.isoformat() and c["intraday"]["by_resolution"] == {"1-minute": 13}
    assert r["matrix"] == [{"year": 2024, "daily": 13, "intraday": 13, "eod_options": 4, "option_trades": 0, "option_quotes": 2}]
    assert "intraday option backtests on 0 sessions of trade prints and 2 of bid/ask" in r["summary"]
    whole = RD.readiness(store)
    assert [s["symbol"] for s in whole["symbols"]] == ["SPY"] and "advertised coverage is not counted" in whole["note"]


def test_a_strategy_is_checked_against_stored_observations_before_it_is_planned(store):
    cfg = {**U.manual_config("SPY", 3.0), "start": DAYS[0].isoformat(), "end": DAYS[-1].isoformat()}
    pc = RD.precheck(cfg, store)
    assert pc["signal_sessions"] == 4 and pc["priced"] == 2 and pc["priced_pct"] == 50.0 and pc["executable"] is True and pc["blocking"] is False
    assert pc["counts"] == {"quotes": 2, "trades": 0, "eod": 0, "exit_uncertain": 0, "stale": 0, "contract": 0, "expiration": 0, "partial": 0, "no_contract": 1, "missing": 1}
    assert pc["potential"] == 2 and pc["fetchable"] == 0
    assert [(s["date"], s["status"]) for s in pc["sessions"]] == [(S3.isoformat(), "quotes"), (S6.isoformat(), "quotes"), (S9.isoformat(), "no_contract"), (S11.isoformat(), "missing")]
    assert pc["untested"] == [S9.isoformat(), S11.isoformat()]
    assert "fires on 4 sessions" in pc["text"] and "2 pass every check on stored option observations (2 on intraday bid/ask): 50.0%" in pc["text"] and "On 1 no contract with the required expiration was listed" in pc["text"]
    assert "A run confirms each one" in pc["text"], "what the stored data supports is not called a trade"
    ids = [a["id"] for a in pc["alternatives"]]
    assert ids[:2] == ["study", "subset"] and ids[-2:] == ["import", "modeled"] and "connect" in ids
    sub = next(a for a in pc["alternatives"] if a["id"] == "subset")
    assert sub["sessions"] == [S3.isoformat(), S6.isoformat()] and "never filled in" in sub["detail"]
    assert next(a for a in pc["alternatives"] if a["id"] == "modeled")["warning"] is True and "Not observed option prices" in next(a for a in pc["alternatives"] if a["id"] == "modeled")["detail"]
    # an end-of-day chain cannot price an entry at the open or a 0DTE contract: those sessions are not counted as priced
    assert pc["intraday"] is True and pc["counts"]["eod"] == 0
    # a strategy whose rule never fires says so instead of offering to run on nothing
    never = RD.precheck({**U.manual_config("SPY", 9.0), "start": DAYS[0].isoformat(), "end": DAYS[-1].isoformat()}, store)
    assert never["signal_sessions"] == 0 and never["executable"] is False and "never fires" in never["text"] and [a["id"] for a in never["alternatives"]] == ["import", "modeled"]


def test_the_agent_states_data_availability_before_the_plan(store):
    ag, sid = agent(store), session()
    m = ag.post(sid, "Buy ATM SPY 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in SPY, only on days when SPY fell at least 3% the previous trading day.")[-1]
    c = m["content"]
    assert m["kind"] == "plan" and c["data_check"]["signal_sessions"] == 4 and c["data_check"]["priced"] == 2 and c["data_check"]["counts"]["no_contract"] == 1
    assert c["text"].index("fires on 4 sessions") < c["text"].index("assumed"), "what can be priced is said first, before anything about the plan"
    assert [b["budget"] for b in c["data_check"]["budget"]["budgets"]] == [0, 25, 50, 100, 200] and "offers" not in c["data_check"]["budget"]
    # one intent, and the same stages for every request: understood, checked against the sentence, data checked, planned, started
    it = c["intent"]
    assert it["kind"] == "strategy" and [(x["key"], x["value"]) for x in it["conditions"]] == [("prev_return", -3.0)] and all(k["ok"] for k in it["checks"])
    st_ = {x["stage"]: x for x in it["stages"]}
    assert list(st_) == ["understand", "validate", "data", "plan", "execute"] and st_["data"]["status"] == "attention" and (st_["data"]["priced"], st_["data"]["of"]) == (2, 4)
    # an opted-in theoretical run needs no option prices, so nothing is checked for it; a study has no option prices to check
    q = ag.post(sid, "How often does SPY close higher after falling 3%?")[-1]
    assert "data_check" not in (q["content"] if isinstance(q["content"], dict) else {})
    qs = {x["stage"]: x for x in q["content"]["intent"]["stages"]}
    assert q["content"]["intent"]["kind"] == "study" and qs["data"]["detail"].startswith("underlying prices only")
