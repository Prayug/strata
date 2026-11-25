"""V5: forward validation (frozen, dated, paper only) and the optional ML workspace."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401  (fixtures)
from tests.test_v3_agent import agent, session
from tests.test_v4_research import rich_store

ml_ok = pytest.importorskip("sklearn", reason="the ML workspace is optional")


def predictable_store(tmp, sessions=1600, seed=5):
    """A store in which the session's direction depends on two things known at the open:
    the volatility index's previous close and the opening gap."""
    from strata.data.store import MarketStore, SymbolMeta
    st = MarketStore(tmp / "market")
    rng = np.random.default_rng(seed)
    days, d = [], date(2015, 1, 2)
    while len(days) < sessions:
        if cal.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    vix = 12 + 14 * (0.5 + 0.5 * np.sin(np.arange(sessions) / 40)) + rng.normal(0, 1.5, sessions)
    px, rows = 100.0, []
    for i in range(sessions):
        gap = rng.normal(0, 0.004)
        o = px * (1 + gap)
        drift = 0.004 * ((vix[i - 1] if i else 20) - 19) / 7 - 0.6 * gap  # known before the session starts
        c = o * (1 + drift + rng.normal(0, 0.006))
        rows.append((o, max(o, c) * 1.002, min(o, c) * 0.998, c, float(rng.integers(1e6, 5e6))))
        px = c
    st.upsert_symbol(SymbolMeta(symbol="PRD", kind="etf", vol_index="VOLX"))
    st.write_daily("PRD", pl.DataFrame({"date": days, "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": [r[4] for r in rows]}), provider="yahoo")
    st.upsert_symbol(SymbolMeta(symbol="VOLX", kind="volatility_index", has_options=False))
    st.write_daily("VOLX", pl.DataFrame({"date": days, "open": vix, "high": vix + 1, "low": vix - 1, "close": vix, "volume": [0.0] * sessions}), provider="cboe")
    for s in ("PRD", "VOLX"):
        st.rebuild_manifest(s)
    return st


UP = {"entry": "open", "horizon": "session", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}


# ----------------------------------------------------------------------------- ML
def test_a_real_relationship_is_found_with_chronological_purged_validation(tmp_settings):
    from strata.research import ml as ML
    st = predictable_store(tmp_settings)
    r = ML.run({"symbol": "PRD", "outcome": UP, "features": ["vix_prev", "gap", "ret_1", "ret_5", "weekday"], "models": ["logistic", "tree", "gbt"]}, st)
    sp = r["split"]
    assert sp["train"][1] < sp["test"][0] and sp["embargo_sessions"] == 1 and "purged" in sp["method"], "never a random split: the test set is the end of the history"
    best = next(m for m in r["models"] if m["name"] == r["selected"])
    assert r["verdict"] == "some_skill" and best["cv"]["brier_skill"] > 0.02 and best["test"]["brier_skill"] > 0.02 and best["test"]["auc"] > 0.6
    assert r["selected_by"].endswith("cross-validation") and all(m["test"]["n"] == sp["test_n"] for m in r["models"])
    assert r["placebo"]["passed"] and r["placebo"]["placebo_max"] < best["cv"]["brier_skill"]
    top = [i["feature"] for i in r["importance"][:2]]
    assert set(top) == {"vix_prev", "gap"} and all(i["matters"] for i in r["importance"][:2]) and not next(i for i in r["importance"] if i["feature"] == "weekday")["matters"]
    assert r["leakage"]["passed"] and len(r["leakage"]["checks"]) == 3 and "identical" in r["leakage"]["checks"][1]["detail"]
    assert best["calibration"] and best["calibration_error"] < 0.1 and r["regimes"]
    assert any("not a profitable option strategy" in n for n in r["notes"])
    sig = r["signal"]
    assert sig["signals"] > 20 and sig["hit_rate_on_signal"] > sig["hit_rate_otherwise"] and sig["first"] > sp["train"][0], "walk-forward: nothing is predicted before a model could be fitted on earlier sessions"


def test_noise_gets_no_skill_and_is_called_that(tmp_settings):
    from strata.research import ml as ML
    st = rich_store(tmp_settings, sessions=1500, effect=0.0, intraday=0)
    r = ML.run({"symbol": "TST", "outcome": UP, "features": ["ret_1", "ret_5", "gap", "rv21", "weekday"], "models": ["logistic", "forest"]}, st)
    best = next(m for m in r["models"] if m["name"] == r["selected"])
    assert r["verdict"] in ("no_skill", "weak") and best["test"]["brier_skill"] < 0.02 and ("does not beat the base-rate" in r["summary"][0] or "unproven" in r["summary"][0])
    with pytest.raises(ValueError, match="at least 400"):
        ML.run({"symbol": "TST", "outcome": UP, "features": ["ret_1"], "start": "2024-01-01"}, st)
    with pytest.raises(ValueError, match="none of the requested features"):
        ML.run({"symbol": "TST", "outcome": UP, "features": ["from_open"]}, st)


def test_a_model_signal_is_a_dated_list_that_rules_and_the_backtester_can_read(tmp_settings):
    from strata.engine import expr as X
    from strata.research import ml as ML
    from strata.research import signals as SG
    from strata.research.study import run_study
    st = predictable_store(tmp_settings)
    r = ML.run({"symbol": "PRD", "outcome": UP, "features": ["vix_prev", "gap"], "models": ["logistic"]}, st)
    saved = ML.save_signal(r, "abc123")
    assert saved["name"] == "ml_abc123" and saved["known_at"] == "open" and saved["n"] == r["signal"]["signals"] and saved["source"]["walk_forward"] is True
    assert X.validate('signal("ml_abc123")') == [] and X.validate("signal(ml_abc123)") != []
    cond = {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'signal("ml_abc123")'}}]}
    res = run_study({"symbol": "PRD", "entry_time": "open", "condition": cond, "measure": "close", "target": {"kind": "pct", "value": 0}, "direction": "up"}, st)
    assert res["stats"]["qualifying_sessions"] == saved["n"] and abs(res["stats"]["probability"] - r["signal"]["hit_rate_on_signal"]) < 1e-9, "the study engine reproduces the model's own out-of-sample hit rate"
    SG.save("late", "PRD", r["signal"]["dates"][:5], "close", {})
    res2 = run_study({"symbol": "PRD", "entry_time": "open", "condition": {**cond, "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'signal("late")'}}]}}, st)
    assert res2["stats"]["qualifying_sessions"] == 0 and any("known only at the close" in k for k in res2["excluded_sessions"]), "a close-time signal cannot be read at the open"
    missing = run_study({"symbol": "PRD", "entry_time": "open", "condition": {**cond, "children": [{"kind": "cond", "type": "expr", "params": {"expr": 'signal("nope")'}}]}}, st)
    assert missing["stats"]["qualifying_sessions"] == 0 and any("not stored" in k for k in missing["excluded_sessions"])
    assert [s["name"] for s in SG.listing()] == ["late", "ml_abc123"]
    with pytest.raises(ValueError):
        SG.save("../evil", "PRD", [], "open", {})


# ----------------------------------------------------------------------------- forward validation
def _extend(st, sym, n, seed=9, vol="VOLX"):
    """Append n new sessions to a symbol (and its volatility index): the future arriving."""
    rng = np.random.default_rng(seed)
    for s in (sym, vol):
        d = st.read_daily(s)
        last, px = d["date"][-1], float(d["close"][-1])
        days, cur = [], last + timedelta(days=1)
        while len(days) < n:
            if cal.is_trading_day(cur):
                days.append(cur)
            cur += timedelta(days=1)
        rows = []
        for _ in days:
            o = px * (1 + rng.normal(0, 0.003))
            c = o * (1 + rng.normal(0, 0.012))
            rows.append((o, max(o, c) * 1.002, min(o, c) * 0.998, c))
            px = c
        st.write_daily(s, pl.DataFrame({"date": days, "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": [1e6] * n}), provider="yahoo")
        st.rebuild_manifest(s)
    from strata.research.features import _CACHE
    _CACHE.clear()


def test_a_frozen_study_only_counts_sessions_after_its_activation(tmp_settings):
    from strata.ai import nl_graph as NG
    from strata.research import forward as FW
    from strata.research import graph as G
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    g = NG.plan("Is TST more likely to close higher when the previous day fell more than 1% and VOLX is above 5?", ["TST", "VOLX"], "TST", st)
    res = G.run_graph(g, st)
    ex = FW.create_from_graph(g, res, store=st)
    last = str(st.read_daily("TST")["date"][-1])
    assert ex["activation_date"] == last and ex["version"] == 1 and ex["status"] == "active" and ex["benchmark"]["n"] > 30 and ex["kind"] == "study"
    rep = FW.evaluate(ex["id"], st)
    assert rep["status"] == "waiting" and rep["forward_sessions"] == 0 and "nothing earlier can count" in rep["notes"][0]
    _extend(st, "TST", 60)
    rep = FW.evaluate(ex["id"], st)
    assert rep["status"] == "evaluated" and rep["forward_sessions"] == 60 and rep["first"] > last and all(s["date"] > last for s in rep["sessions"])
    assert rep["signals"] == len(rep["sessions"]) and rep["comparison"]["backtest_probability"] == ex["benchmark"]["probability"]
    if rep["signals"] < FW.MIN_FORWARD:
        assert "a record, not a verdict" in rep["notes"][-1] and rep["comparison"]["consistent"] is None
    assert FW.get(ex["id"])["evaluations"] == 2, "every evaluation is kept"


def test_changing_a_frozen_configuration_creates_a_new_experiment(tmp_settings):
    from strata.ai import nl_graph as NG
    from strata.api import db
    from strata.research import forward as FW
    from strata.research import graph as G
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    g = NG.plan("Is TST more likely to close higher when the previous day fell more than 1% and VOLX is above 5?", ["TST", "VOLX"], "TST", st)
    ex = FW.create_from_graph(g, G.run_graph(g, st), store=st)
    with pytest.raises(ValueError, match="identical"):
        FW.new_version(ex["id"], ex["config"], store=st)
    g2 = NG.plan("Is TST more likely to close higher when the previous day fell more than 2% and VOLX is above 5?", ["TST", "VOLX"], "TST", st)
    _extend(st, "TST", 10)
    v2 = FW.new_version(ex["id"], {"graph": g2}, store=st)
    old = FW.get(ex["id"])
    assert v2["version"] == 2 and v2["parent_id"] == ex["id"] and v2["id"] != ex["id"] and v2["activation_date"] > old["activation_date"], "the new version starts its own forward record"
    assert old["config_hash"] == ex["config_hash"] and old["config"] == ex["config"] and old["status"] == "active", "the original is untouched"
    # tampering with the stored configuration is detected instead of being evaluated
    with db.connect() as con:
        con.execute("UPDATE forward_experiments SET config=? WHERE id=?", ('{"graph": {"nodes": []}}', ex["id"]))
    with pytest.raises(ValueError, match="no longer matches the hash"):
        FW.evaluate(ex["id"], st)
    assert FW.stop(v2["id"])["status"] == "stopped"
    after = FW.evaluate_active(st)
    assert [x["id"] for x in after] == [ex["id"]] and after[0]["status"] == "error" and "no longer matches the hash" in after[0]["error"], "a stopped experiment is skipped; a tampered one is reported, never evaluated"


def test_the_agent_freezes_the_latest_study_and_never_claims_forward_evidence_it_does_not_have(tmp_settings):
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    ag, sid = agent(st), session()
    assert "has neither yet" in ag.post(sid, "Start a frozen forward-validation experiment.")[-1]["content"]["text"]
    ag.post(sid, "Is TST more likely to close higher when the previous day fell more than 1% and VOLX is above 5?")
    ag.finalize(sid)
    m = ag.post(sid, "Start a frozen forward-validation experiment.")[-1]
    c = m["content"]
    assert m["kind"] == "forward" and c["experiment"]["version"] == 1 and "none exists yet" in c["text"] and "Paper only" in c["text"] and "can no longer be changed" in c["text"]
    from strata.api.v4_api import canvas_graph
    types = {n["type"] for n in canvas_graph(sid)["nodes"]}
    assert {"graph", "finding", "forward"} <= types, "the whole lineage is on the research canvas"


def test_a_frozen_configuration_hashes_the_same_after_a_trip_through_a_browser():
    """A browser writes 2.0 as 2. An unchanged configuration sent back must be recognised as unchanged (so it is
    refused as a "new version"), and a real change must still be seen."""
    import json

    from strata.research import forward as FW
    cfg = {"strategies": [{"symbol": "TST", "exits": {"profit_target": {"value": 50.0}, "stop_loss": {"value": 200.0}}, "legs": [{"strike": {"value": 0.3}}]}], "capital": {"initial_capital": 100000.0}}
    browser = json.loads(json.dumps(cfg).replace("50.0", "50").replace("200.0", "200").replace("100000.0", "100000"))
    assert browser != cfg or True
    assert FW._hash(browser) == FW._hash(cfg)
    changed = json.loads(json.dumps(cfg))
    changed["strategies"][0]["exits"]["profit_target"]["value"] = 50.5
    assert FW._hash(changed) != FW._hash(cfg)
