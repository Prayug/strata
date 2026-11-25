"""V3 research agent: clarifying questions, confirmation, coverage decisions, trade-priced
follow-ups, investigations, discovery, provider reliability and credential handling.
Everything runs offline against hand-built stores (jobs execute inline)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from strata.core import calendar as cal
from tests.helpers import et_epoch
from tests.test_v2_research import DAYS, FakeLLM, InlineJobs, eod_store, study_store, tmp_settings  # noqa: F401  (fixtures)


def agent(st, llm=None):
    from strata.ai.agent import Agent
    return Agent(InlineJobs(), st, llm_factory=lambda: llm)


def session():
    from strata.api import db
    return db.create_session("t")["id"]


def add_trade_bars(st, d: date, strike: float, prints: list[tuple], exp: date | None = None, right="C"):
    exp = exp or d
    occ = f"TST{exp:%y%m%d}{right}{int(strike * 1000):08d}"
    rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp, "strike": strike, "right": right, "open": px, "high": px, "low": px, "close": px, "volume": 5} for t, px in prints]
    st.write_option_bars("TST", "testfeed", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))),
                         [{"date": d, "occ": occ, "expiration": exp, "strike": strike, "right": right, "bars": len(rows), "status": "ok", "pages": 1}])


# ----------------------------------------------------------------------------- questions, unsupported parts, review mode
def test_an_ambiguous_target_is_asked_before_anything_runs(tmp_settings):
    from strata.api import db
    ag, sid = agent(study_store(tmp_settings)), session()
    plan = ag.post(sid, "Buy TST 0DTE calls at the open with a 1% target after a 3% down day")[-1]["content"]
    assert plan["state"] == "awaiting_answer" and plan["runs"] == [] and db.list_runs() == []
    q = plan["questions"][0]
    assert q["text"] == "You said target 1%. Do you mean a 1% move in TST or a 1% gain on the option?" and [o["label"] for o in q["options"]] == ["1% move in TST", "1% gain on the option"]
    assert plan["draft"]["underlying_target_pct"] == 1.0  # the plausible reading is shown, but nothing has run on it
    msg = ag.answer(sid, plan["version_id"], "target", 1)[-1]
    c = msg["content"]
    assert c["draft"]["premium_target_pct"] == 1.0 and c["draft"]["underlying_target_pct"] is None and c["questions"][0]["answered"] == 1
    assert c["state"] == "running" and len(c["runs"]) == 1 and any("1% of the option premium" in str(i["value"]) for i in c["items"])
    assert len(db.list_versions(sid)) == 1  # the answer completes the same version; it does not fork history


def test_a_large_bare_percentage_defaults_to_the_option_but_is_still_asked(tmp_settings):
    from strata.ai import nl_rules as NL
    from strata.ai.nl_stateful import enrich
    d = enrich("Buy SPY 0DTE calls at the open with a 50% stop and take profit at 100% of the premium", NL.TICKERS)
    assert [q["id"] for q in d.questions] == ["stop"] and d.questions[0]["options"][0]["label"] == "50% loss on the option" and d.premium_stop_pct == 50 and d.underlying_stop_pct is None
    assert d.premium_target_pct == 100  # "of the premium" was said, so no question about the target
    assert enrich("Buy SPY calls at 6:30 AM Pacific, targeting a 1% underlying price increase", NL.TICKERS).questions == []


def test_a_part_that_cannot_be_represented_is_named_and_nothing_runs_until_confirmed(tmp_settings):
    from strata.api import db
    ag, sid = agent(study_store(tmp_settings)), session()
    plan = ag.post(sid, "Buy TST 0DTE calls at 10:00 ET after a 3% down day when dealers are short gamma and the realized skew is inverted")[-1]["content"]
    assert plan["unsupported"] == ["when dealers are short gamma and the realized skew is inverted"]
    assert plan["state"] == "awaiting_confirmation" and plan["runs"] == [] and db.list_runs() == [] and "Nothing was substituted" in plan["text"]
    assert [c["expr"] for c in plan["draft"]["conditions"]] == ["previous_session.return_pct <= -3"]  # no loosely related stand-in was invented
    assert any(i["status"] == "unsupported" for i in plan["items"])
    c = ag.confirm(sid, plan["version_id"])[-1]["content"]
    assert c["state"] == "running" and len(c["runs"]) == 1


def test_review_mode_waits_for_confirmation_and_quick_mode_does_not(tmp_settings):
    st = study_store(tmp_settings)
    ag, sid = agent(st), session()
    q = "How often does TST rise 1% from the open after falling 3% or more the previous trading day?"
    assert ag.post(sid, q)[-1]["content"]["state"] == "running"
    sid2 = session()
    ag.set_mode(sid2, "review")
    plan = ag.post(sid2, q)[-1]["content"]
    assert plan["state"] == "awaiting_confirmation" and plan["runs"] == [] and "Review mode" in plan["text"] and ag.payload(sid2)["exec_mode"] == "review"
    assert ag.confirm(sid2, plan["version_id"])[-1]["content"]["state"] == "running"
    res = ag.finalize(sid2)[0]["content"]
    assert res["view"] == "single" and "1 of the 2 sessions" in res["text"] and res["coverage"]["signals"]["qualifying"] == 2


def test_missing_position_is_a_question_not_a_guess(tmp_settings):
    ag, sid = agent(study_store(tmp_settings)), session()
    plan = ag.post(sid, "Enter after a 1% decline, add another contract after a further 0.5% decline, and close both positions when price returns to VWAP.")[-1]["content"]
    assert plan["state"] == "awaiting_answer" and plan["draft"]["structure"] is None and plan["questions"][0]["id"] == "structure"
    assert plan["draft"]["adds"][0]["expr"] == "price <= entry.underlying_price * (1 - 0.005)" and plan["draft"]["exit_conditions"][0]["expr"] == "price >= vwap"
    assert plan["draft"]["conditions"][0]["expr"] == "from_open_pct <= -1"
    c = ag.answer(sid, plan["version_id"], "structure", 0)[-1]["content"]
    assert c["draft"]["structure"] == "long_call" and c["config"]["strategy"]["entry"]["adds"][0]["when"] == "price <= entry.underlying_price * (1 - 0.005)"
    assert c["config"]["strategy"]["exits"]["custom"]["children"][-1]["params"]["expr"] == "price >= vwap"


# ----------------------------------------------------------------------------- stateful language -> explicit engine constructs
@pytest.mark.parametrize("text,check", [
    ("Wait until SPY drops 0.5% from the open, then buy calls when it recovers half of that decline.",
     lambda d: [s.expr for s in d.sequence] == ["from_open_pct <= -0.5", 'price >= low_since("drop") + 0.5 * (open - low_since("drop"))'] and d.structure == "long_call" and d.conditions == []),
    ("Buy SPY puts only after the first breakout fails and price reclaims VWAP.",
     lambda d: [s.name for s in d.sequence] == ["breakout", "failed", "reclaim"] and d.sequence[2].expr == "crosses_above(price, vwap)" and d.structure == "long_put"),
    ("Buy puts after three consecutive five-minute red candles, unless yesterday was also red.",
     lambda d: [c.expr for c in d.conditions] == ['consecutive_bearish_candles("5m") >= 3', "not (previous_session.open_to_close_pct < 0)"] and d.unsupported == []),
    ("Enter calls when SPY breaks yesterday's high, but only if QQQ hasn't broken its own high.",
     lambda d: [c.expr for c in d.conditions] == ["crosses_above(price, prev_high)", "QQQ.high <= QQQ.prev_high"] and d.structure == "long_call" and d.unsupported == []),
    ("Buy SPY calls when SPY rallies 1% from the open and then gives back half of that rally",
     lambda d: [s.expr for s in d.sequence] == ["from_open_pct >= 1", 'price <= high_since("rally") - 0.5 * (high_since("rally") - open)']),
])
def test_temporal_requests_become_sequences_and_relationships(tmp_settings, text, check):
    from strata.ai import nl_rules as NL
    from strata.ai.draft import compile_draft
    from strata.ai.nl_stateful import enrich
    d = enrich(text, NL.TICKERS)
    assert check(d), d.model_dump(exclude_none=True)
    c = compile_draft(d)
    assert c["errors"] == []
    if d.sequence:
        assert [s["when"] for s in c["strategy"]["entry"]["sequence"]] == [s.expr for s in d.sequence]
        assert any(i["label"].startswith("Step 1") for i in c["items"]) and any(i["label"] == "Order" for i in c["items"])


# ----------------------------------------------------------------------------- trade-priced follow-ups, costs, skipped signals, export
def trade_store(tmp):
    st = study_store(tmp)
    # qualifying sessions are DAYS[2] (after -4%) and DAYS[5] (after -3.6%). Only DAYS[2] has option trade prints.
    add_trade_bars(st, DAYS[2], 96.0, [("09:30", 1.00), ("09:31", 1.05), ("10:29", 1.20), ("10:30", 2.10), ("10:31", 2.00), ("15:54", 1.60)])
    return st


def test_option_trades_on_the_sessions_of_a_study_with_costs_skips_and_export(tmp_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from strata.api import app as appmod
    from strata.api import db
    st = trade_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    ag.finalize(sid)
    assert ag.post(sid, "Show every qualifying historical session.")[-1]["kind"] == "trades"
    plan = ag.post(sid, "Test ATM 0DTE calls on those days using historical option trade prices.")[-1]["content"]
    d = plan["draft"]
    assert d["kind"] == "options" and d["structure"] == "long_call" and d["dte"] == 0 and d["mode"] == "trades" and d["conditions"][0]["expr"] == "previous_session.return_pct <= -3"
    tgt = next(i for i in plan["items"] if i["label"] == "Target")
    assert tgt["status"] == "assumed" and d["underlying_target_pct"] == 1.0  # carried over from the study and disclosed as such
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "single" and res["mode"] == "trades" and res["coverage"]["signals"]["qualifying"] == 2 and res["coverage"]["executable"]["signals"] == 1
    assert "only 1 of them has usable observed option prices" in res["text"] and "One trade is an observation, not evidence" in res["text"]
    run = db.get_run(res["run_id"])
    from strata.api.results import load_result, load_trades
    r = load_result(run["id"])
    t = load_trades(run["id"])[0]
    leg = t["legs"][0]
    assert r["meta"]["pricing"] == "trades" and r["meta"]["datasets_used"]["TST"]["observation"] == "trade prints"
    assert leg["price_source"] == "trade_print" and leg["entry_bid"] is None and leg["entry_print"] == pytest.approx(1.00) and t["entry_time"][11:16] == "09:31"  # the 09:30 bar is visible once it has ended
    assert t["exit_reason"] == "underlying_target" and leg["exit_print"] == pytest.approx(2.10)  # +1% reached at 10:30; the print observed at 10:31
    sk = r["coverage"]["skipped_signals"]
    assert sk["count"] == 1 and sk["rows"][0]["date"] == DAYS[5].isoformat() and sk["rows"][0]["class"] == "data"
    # ---- conservative costs: a new version, compared with the previous one, worse by construction
    c2 = ag.post(sid, "Include commissions and conservative slippage.")[-1]["content"]
    assert c2["changed"] == ["costs"] and c2["draft"]["costs"]["preset"] == "conservative" and c2["compare_with"] and c2["config"]["strategy"] is not None
    cmp_ = ag.finalize(sid)[0]["content"]
    rows = cmp_["table"]["rows"]
    assert cmp_["view"] == "compare" and rows[1]["values"]["net_pnl"] < rows[0]["values"]["net_pnl"] and cmp_["compatibility"]["differences"][0]["field"] == "execution"
    sk_msg = ag.post(sid, "Show me every skipped signal and why it was skipped.")[-1]["content"]
    assert sk_msg["view"] == "skipped" and "1 qualifying session did not become a trade" in sk_msg["text"] and "none of them is counted as a losing or winning trade" in sk_msg["text"]
    rob = ag.post(sid, "Is this statistically significant?")[-1]["content"]
    assert rob["view"] == "robustness" and rob["robustness"]["level"] == "insufficient" and rob["robustness"]["trials"] == ag.payload(sid)["trials"] >= 3
    ex = ag.post(sid, "Export the complete research report.")[-1]["content"]
    assert ex["view"] == "export" and len(ex["links"]) == 6
    monkeypatch.setattr(appmod, "jobs", InlineJobs())
    c = TestClient(appmod.app)
    html = c.get(ex["links"][0]["href"]).text
    assert "Research question" in html and "trade prints" in html and "Signals that did not become trades" in html and "Methodology and reproduction" in html and "previous_session.return_pct" in html
    md = c.get(ex["links"][1]["href"]).text
    assert md.startswith("# ") and "## Data sources and coverage" in md and "## Robustness" in md and "## Limitations" in md
    rep = c.get(ex["links"][2]["href"]).json()
    assert rep["data"]["executable"] == {"signals": 1, "of": 2, "window": [DAYS[2].isoformat(), DAYS[2].isoformat()]} and rep["interpretation"]["items"] and rep["reproduce"]["config"]["pricing"]["mode"] == "trades"
    assert c.get(ex["links"][3]["href"]).text.splitlines()[0].startswith("id,date,entry_time") and "no_option_data" in c.get(ex["links"][4]["href"]).text
    cal_ = c.get(f"/api/runs/{rep['reproduce']['run_id']}/calendar").json()
    assert cal_["days"][DAYS[2].isoformat()]["s"] in ("win", "loss") and cal_["days"][DAYS[5].isoformat()]["s"] == "skipped"
    day = c.get(f"/api/runs/{rep['reproduce']['run_id']}/day", params={"day": DAYS[5].isoformat()}).json()
    assert day["status"] == "skipped" and "data limitation, not a failed rule" in day["why"] and day["rules"]["signal"]["time"] == "09:30"
    day = c.get(f"/api/runs/{rep['reproduce']['run_id']}/day", params={"day": DAYS[3].isoformat()}).json()
    assert day["status"] == "none" and "was never true" in day["why"]
    day = c.get(f"/api/runs/{rep['reproduce']['run_id']}/day", params={"day": DAYS[2].isoformat()}).json()
    assert day["trades"][0]["signal_time"] == "09:30" and day["option_series"][0]["kind"].startswith("trade prints") and len(day["option_series"][0]["points"]) == 6


def test_changing_only_the_expiration_keeps_the_exit_that_was_in_force(tmp_settings):
    st = trade_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "Buy ATM TST 0DTE calls at the open after a 3% down day using option trade prices, exit when TST moves up 1% from entry")
    ag.finalize(sid)
    c = ag.post(sid, "Compare the results with 7DTE calls.")[-1]["content"]
    assert c["draft"]["dte"] == 7 and c["draft"]["exit_time"] == "15:55" and "exit_time" in c["draft"]["carried"]
    assert next(i for i in c["items"] if i["label"] == "Time exit")["status"] == "assumed"  # disclosed, not silently changed to "hold to expiration"


# ----------------------------------------------------------------------------- coverage decisions
def test_a_stated_period_that_cannot_be_covered_stops_and_offers_the_choices(tmp_settings):
    from strata.api import db
    st = eod_store(tmp_settings)  # daily bars Jan-Mar 2024; end-of-day option quotes for the last 25 sessions only
    ag, sid = agent(st), session()
    plan = ag.post(sid, "Buy a 30DTE ATM SPY call at the close after a down day and hold for 2 days, in 2024")[-1]["content"]
    assert plan["state"] == "running" and plan["draft"]["start"] == "2024-01-01"
    dec = ag.finalize(sid)[0]
    c = dec["content"]
    assert dec["kind"] == "decision" and "You asked for 2024-01-01 to 2024-12-31" in c["text"] and "but only" in c["text"]
    assert [x["id"] for x in c["choices"]][:1] == ["reduced_window"] and db.get_run(c["run_id"])["summary"]["outcome"] == "needs_decision"
    ag.decide(sid, dec["id"], "reduced_window")
    out = ag.finalize(sid)
    # the underlying shortfall was accepted; the option quotes still cover only some signals, which the same approval allows
    res = out[0]["content"]
    assert out[0]["kind"] == "result" and res["view"] == "single" and res["mode"] == "historical"
    cov = res["coverage"]
    assert cov["executable"]["signals"] < cov["signals"]["qualifying"] and cov["requested"]["stated"] is True
    assert any("no option quotes stored for the entry session" in i["reason"] for i in cov["options"]["issues"])


def test_without_a_stated_period_the_run_proceeds_and_reports_what_was_left_out(tmp_settings):
    st = eod_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "Buy a 30DTE ATM SPY call at the close after a down day and hold for 2 days")
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "single" and res["mode_label"].startswith("Historical options backtest (bid/ask quotes)") and "Chosen automatically" in res["mode_reason"]
    cov = res["coverage"]
    assert 0 < cov["executable"]["signals"] < cov["signals"]["qualifying"] and "could not be tested" in res["text"]


# ----------------------------------------------------------------------------- investigations and discovery
def test_an_investigation_runs_related_studies_and_adjusts_for_the_comparisons(tmp_settings):
    st = study_store(tmp_settings)
    ag, sid = agent(st), session()
    m = ag.post(sid, "Find out whether TST tends to reverse after large down days.")[-1]
    c = m["content"]
    assert m["kind"] == "investigation" and len(c["runs"]) == 4 and "closes above its open more often" in c["plan"]["hypothesis"] and len(c["plan"]["variants"]) == 4
    assert all(v["items"] for v in c["plan"]["variants"])  # every variant shows its own interpretation
    res = ag.finalize(sid)[0]["content"]
    rep = res["report"]
    assert res["view"] == "investigation" and len(rep["rows"]) == 4 and rep["level"] == "insufficient" and "fewer than 30" in rep["verdict"]
    assert [r["n"] for r in rep["rows"]] == [2, 2, 2, 1] and all(r["adjusted_p_value"] is None or r["adjusted_p_value"] >= r["p_value"] for r in rep["rows"] if r["p_value"] is not None)
    assert ag.payload(sid)["trials"] == 4 and "not option returns" in " ".join(rep["limitations"])


def long_store(tmp, effect: bool):
    """600 daily sessions. A fall of 2% happens every 10th session; with `effect` the following
    session opens flat and closes +1.5% (a real, directional reversal), otherwise it is flat."""
    from strata.data.store import MarketStore, SymbolMeta
    st = MarketStore(tmp / "market")
    st.upsert_symbol(SymbolMeta(symbol="TST", kind="etf"))
    days, d = [], date(2021, 1, 4)
    while len(days) < 600:
        if cal.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    rows, px = [], 100.0
    for i, d in enumerate(days):
        o = px
        if i % 10 == 5:
            c = o * 0.98
            hi, lo = o, c
        elif i % 10 == 6 and effect:
            c = o * 1.015
            hi, lo = c, o
        else:
            c = o * (1.001 if i % 2 else 0.999)
            hi, lo = max(o, c), min(o, c)
        rows.append({"date": d, "open": o, "high": hi, "low": lo, "close": c, "volume": 1.0})
        px = c
    st.write_daily("TST", pl.DataFrame(rows), provider="yahoo")
    return st


def _grid(sym="TST"):
    from strata.research.discovery import build_grid
    dims = [{"name": "move", "label": "Decline", "values": [1, 2, 3]}, {"name": "target", "label": "Target", "values": [0.5, 1.0]}]

    def mk(p):
        return ({"symbol": sym, "condition": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": f"previous_session.return_pct <= -{p['move']}"}}]},
                 "entry_time": "open", "target": {"kind": "pct", "value": p["target"]}, "reference": "entry", "horizon": {"kind": "session"}}, f"down {p['move']}%, +{p['target']}%")
    return {"symbol": sym, "question": "q", "hypothesis": "h", "direction": "up", "dims": dims, "configs": build_grid(sym, dims, mk)}


def test_discovery_keeps_every_configuration_and_only_candidates_touch_the_final_period(tmp_path):
    from strata.research.discovery import run_discovery
    res = run_discovery(_grid(), long_store(tmp_path, effect=True))
    assert res["tested"] == 6 and len(res["configs"]) == 6  # nothing is dropped, including configurations that show nothing
    by = {c["key"]: c for c in res["configs"]}
    assert set(res["candidates"]) == {"move=1|target=0.5", "move=1|target=1.0", "move=2|target=0.5", "move=2|target=1.0"} and res["confirmed"] == res["candidates"]
    assert all("test" in by[k] and by[k]["status_label"] == "confirmed on untouched data" for k in res["candidates"])
    quiet = by["move=3|target=0.5"]
    assert quiet["explore"]["n"] == 0 and quiet["candidate"] is False and "test" not in quiet and quiet["status_label"] == "exploratory" and "only 0 observations" in quiet["rejected_because"][0]
    s = res["splits"]
    assert s["explore"][1] < s["validate"][0] <= s["validate"][1] < s["test"][0] and res["test_period_used"] is True
    assert all(c["q_value"] is None or c["q_value"] >= c["explore"]["p_value"] for c in res["configs"])  # adjusted p-values are never smaller than the raw ones


def test_discovery_finds_nothing_in_noise_and_leaves_the_final_period_untouched(tmp_path):
    from strata.research.discovery import run_discovery
    res = run_discovery(_grid(), long_store(tmp_path, effect=False))
    assert res["candidates"] == [] and res["confirmed"] == [] and res["test_period_used"] is False
    assert all("test" not in c for c in res["configs"]) and "remains untouched" in " ".join(res["notes"])


def test_discovery_request_goes_through_the_agent(tmp_path, tmp_settings):
    st = long_store(tmp_settings, effect=True)
    ag, sid = agent(st), session()
    m = ag.post(sid, "Investigate mean reversion after unusually large TST declines.")[-1]
    assert m["kind"] == "discovery" and m["content"]["tested"] == 64 and "final test period" in m["content"]["text"]
    res = ag.finalize(sid)[0]["content"]
    d = res["discovery"]
    assert res["view"] == "discovery" and d["tested"] == 64 and len(d["configs"]) == 64 and ag.payload(sid)["trials"] == 64
    assert "Everything else is exploratory" in res["text"] and d["confirmed"] and all(k.endswith("entry=open|target=0.5|regime=any") or "entry=open" in k for k in d["confirmed"])


def test_statistics_helpers():
    from strata.research import stats as ST
    assert ST.bh_adjust([0.01, 0.04, 0.03, None, 0.5]) == [pytest.approx(0.04), pytest.approx(0.0533333), pytest.approx(0.0533333), None, pytest.approx(0.5)]
    assert ST.bonferroni(0.02, 10) == pytest.approx(0.2) and ST.bonferroni(0.2, 10) == 1.0
    assert ST.two_prop_p(50, 100, 50, 100) == pytest.approx(1.0) and ST.two_prop_p(90, 100, 50, 100) < 1e-6 and ST.two_prop_p(0, 0, 1, 2) is None
    assert ST.mean_test([1.0, 1.0, 1.0])["p_value"] is None and ST.mean_test([1, 2, 3, 4, 5])["mean"] == 3.0


# ----------------------------------------------------------------------------- language-model reliability
def test_tool_arguments_are_validated_against_strict_schemas(tmp_settings):
    from strata.ai.agent import llm_interpret
    from strata.ai.llm import LLMError
    st = study_store(tmp_settings)
    good = {"tool": "sweep", "args": {"field": "dte", "values": [0, 7]}}
    m = FakeLLM(objs=[{"tool": "sweep", "args": {"field": "leverage", "values": [1]}}, {"tool": "check_coverage", "args": {"symbol": "SPY; DROP TABLE runs"}}, good])
    assert llm_interpret(m, "x", None, [], st, lambda: None) == {"action": "sweep", "field": "dte", "values": [0, 7]}
    assert "field" in m.calls[1][1][-1]["content"] and "symbol" in m.calls[2][1][-1]["content"]
    with pytest.raises(LLMError, match="kept returning an invalid action"):
        llm_interpret(FakeLLM(objs=[{"tool": "propose", "args": {"draft": {}, "execute_sql": "x"}}] * 4), "x", None, [], st, lambda: None)
    seq = {"kind": "options", "symbol": "SPY", "structure": "long_call", "sequence": [{"name": "a", "expr": 'price > stage_price("later")'}, {"name": "later", "expr": "price > open"}]}
    ok = {**seq, "sequence": [{"name": "a", "expr": "from_open_pct <= -0.5"}, {"name": "b", "expr": 'price >= low_since("a") + 0.1'}], "unsupported": ["dealers are short gamma"]}
    schemas = []

    class Recording(FakeLLM):
        def json(self, system, messages, schema=None, **kw):
            schemas.append(schema)
            return super().json(system, messages, schema=schema, **kw)
    m = Recording(objs=[{"tool": "propose", "args": {"draft": seq}}, {"tool": "propose", "args": {"draft": ok}}])
    r = llm_interpret(m, "wait for a drop then buy calls", None, [], st, lambda: None)
    assert "not an earlier stage" in m.calls[1][1][-1]["content"] and [s.name for s in r["draft"].sequence] == ["a", "b"] and r["draft"].unsupported == ["dealers are short gamma"]
    assert schemas[0]["properties"]["tool"]["enum"][0] == "list_datasets"  # the provider's structured-output mode is asked for the action envelope


def test_a_model_failure_is_never_a_silent_downgrade(tmp_settings):
    from strata.ai.llm import LLMError
    from strata.api import db
    from strata.api.settings import set_setting

    class Broken(FakeLLM):
        def json(self, *a, **k):
            raise LLMError("provider down")
    ag, sid = agent(study_store(tmp_settings), Broken()), session()
    plan = ag.post(sid, "Wait until TST drops 0.5% from the open, then buy calls when it recovers half of that decline")[-1]["content"]
    assert plan["interpreter"] == "built-in rules" and "provider down" in plan["text"] and "understands fewer phrasings" in plan["text"]
    set_setting("llm.fallback", "none")
    n = len(db.list_runs())
    m = ag.post(session(), "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")[-1]
    assert m["kind"] == "error" and "nothing was run" in m["content"]["text"] and len(db.list_runs()) == n


def test_provider_calls_retry_transient_failures_record_usage_and_can_be_cancelled(tmp_settings, monkeypatch):
    from strata.ai import llm
    from strata.api import db

    class R:
        def __init__(self, code, body):
            self.status_code, self._b, self.text = code, body, json.dumps(body)

        def json(self):
            return self._b
    seq = [R(503, {"error": {"message": "overloaded"}}), R(200, {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 11, "completion_tokens": 4}})]
    sent = []
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm.httpx, "request", lambda *a, **k: (sent.append(k.get("json")), seq.pop(0))[1])
    m = llm.OpenAICompat("openai", "https://api.openai.com/v1", "sk-test-1234567890abcd", "gpt-x")
    assert m.json("s", [{"role": "user", "content": "u"}], schema={"type": "object"}) == {"ok": True} and len(sent) == 2
    assert sent[0]["response_format"]["type"] == "json_schema" and m.last_usage == {"input_tokens": 11, "output_tokens": 4}
    u = db.usage_summary()
    assert u["calls"] == 1 and u["input_tokens"] == 11 and u["rows"][0]["provider"] == "openai"
    # a provider that rejects the schema mode falls back to plain JSON mode instead of failing
    seq[:] = [R(400, {"error": {"message": "response_format json_schema is not supported"}}), R(200, {"choices": [{"message": {"content": '{"ok": 1}'}}]})]
    sent.clear()
    assert m.json("s", [], schema={"type": "object"}) == {"ok": 1} and sent[1]["response_format"] == {"type": "json_object"}
    m.cancel = lambda: True
    with pytest.raises(llm.LLMError, match="cancelled"):
        m.text("s", [])
    monkeypatch.setattr(llm.httpx, "request", lambda *a, **k: R(401, {"error": {"message": "Incorrect API key provided: sk-test-1234567890abcd"}}))
    m.cancel = None
    with pytest.raises(llm.LLMError) as e:
        m.text("s", [])
    assert "sk-test" not in str(e.value) and "[redacted]" in str(e.value)


# ----------------------------------------------------------------------------- credentials and remote access
def test_credentials_live_outside_the_database_and_old_ones_are_moved(tmp_settings):
    import stat
    from strata.api import db, settings
    with db.connect() as con:  # a Strata V2 database: the key sits in the settings table
        con.execute("INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)", ("secret.ALPACA_API_KEY", json.dumps("PKLEGACY1234567890AB"), db.now()))
    assert settings.get_secret("ALPACA_API_KEY") == "PKLEGACY1234567890AB"  # still readable before the migration
    settings._cache.clear()
    assert settings.migrate_legacy_secrets() == 1
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) AS n FROM settings WHERE key LIKE 'secret.%'").fetchone()["n"] == 0
    f = tmp_settings / "secrets.json"
    assert json.loads(f.read_text()) == {"ALPACA_API_KEY": "PKLEGACY1234567890AB"} and stat.S_IMODE(os.stat(f).st_mode) == 0o600
    assert settings.get_secret("ALPACA_API_KEY") == "PKLEGACY1234567890AB" and settings.secret_source("ALPACA_API_KEY") == "protected file (owner-only)"
    pub = json.dumps(settings.public_settings(), ensure_ascii=False)
    assert "PKLEGACY" not in pub and "…90AB" in pub
    assert settings.redact("failed with key PKLEGACY1234567890AB and sk-ant-abcdefgh12345678") == "failed with key [redacted] and [redacted]"
    settings.set_secret("ALPACA_API_KEY", None)
    assert settings.get_secret("ALPACA_API_KEY") is None and json.loads(f.read_text()) == {}
    with pytest.raises(ValueError):
        settings.set_secret("NOT_A_KEY", "x")


def test_keychain_backend_never_puts_the_secret_on_a_command_line(tmp_settings, monkeypatch):
    from strata.api import settings
    calls = []
    vault = {}

    def fake(cmd, data=None):
        calls.append((cmd, data))
        if cmd[:2] == ["security", "-i"]:
            name = data.split(" -a ")[1].split(" ")[0]
            vault[name] = data.strip().rsplit(" -w ", 1)[1]
            return 0, ""
        if "find-generic-password" in cmd:
            v = vault.get(cmd[cmd.index("-a") + 1])
            return (0, v + "\n") if v else (44, "")
        if "delete-generic-password" in cmd:
            vault.pop(cmd[cmd.index("-a") + 1], None)
        return 0, ""
    monkeypatch.setenv("STRATA_SECRET_STORE", "keychain")
    monkeypatch.setattr(settings, "_run", fake)
    settings.set_secret("OPENAI_API_KEY", "sk-test-keychain-000111222")
    assert settings.get_secret("OPENAI_API_KEY") == "sk-test-keychain-000111222" and settings.secret_source("OPENAI_API_KEY") == "macOS Keychain"
    assert all("sk-test-keychain-000111222" not in " ".join(cmd) for cmd, _ in calls)  # only ever passed on stdin
    assert not (tmp_settings / "secrets.json").exists()
    with pytest.raises(RuntimeError):
        settings.set_secret("OPENAI_API_KEY", "bad value with spaces")


def test_requests_from_another_machine_are_refused_without_a_token(tmp_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from strata.api import app as appmod
    monkeypatch.setattr(appmod, "jobs", InlineJobs())
    local = TestClient(appmod.app)
    assert local.get("/api/settings").status_code == 200
    remote = TestClient(appmod.app, client=("203.0.113.9", 50000))
    r = remote.get("/api/settings")
    assert r.status_code == 403 and "Remote access is disabled" in r.json()["detail"]
    assert remote.put("/api/settings", json={"secrets": {"OPENAI_API_KEY": "sk-evil-000000000000"}}).status_code == 403
    monkeypatch.setenv("STRATA_API_TOKEN", "s3cret-token")
    assert remote.get("/api/settings").status_code == 401
    assert remote.get("/api/settings", headers={"X-Strata-Token": "wrong"}).status_code == 401
    assert remote.get("/api/settings", headers={"X-Strata-Token": "s3cret-token"}).status_code == 200
    # a loopback client arriving under a foreign Host name (DNS rebinding) is treated as remote
    rebind = TestClient(appmod.app, client=("127.0.0.1", 50000), base_url="http://attacker.example")
    assert rebind.get("/api/settings").status_code == 401
    assert TestClient(appmod.app, client=("127.0.0.1", 50000), base_url="http://localhost:8765").get("/api/settings").status_code == 200
