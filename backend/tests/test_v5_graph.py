"""V5: composable research graphs, matched comparisons, the registry, research memory, the
pattern explorer and the agent behaviour built on them. Offline: every store is generated
in a temporary directory."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401  (fixtures)
from tests.test_v3_agent import agent, session
from tests.test_v4_research import rich_store


def base_graph(sym="TST", value=-2.0, entry="open", horizon="session", at="open", extra=None):
    nodes = [{"id": "d", "op": "data", "params": {"symbol": sym}},
             {"id": "f", "op": "feature", "inputs": {"data": "d"}, "params": {"name": "ret_1", "at": at}},
             {"id": "c", "op": "filter", "inputs": {"x": "f"}, "params": {"op": "<=", "value": value}},
             {"id": "o", "op": "forward_return", "inputs": {"data": "d"}, "params": {"entry": entry, "horizon": horizon}},
             {"id": "p", "op": "conditional_probability", "inputs": {"outcome": "o", "condition": "c"}}]
    return {"question": "test", "nodes": nodes + (extra or [])}


# ----------------------------------------------------------------------------- validation
def test_a_graph_is_refused_for_structure_type_and_parameter_errors():
    from strata.research import graph as G
    assert G.validate(base_graph())["errors"] == []
    bad = base_graph()
    bad["nodes"][4]["inputs"]["condition"] = "f"  # a series where a condition is required
    assert any("needs a mask" in e for e in G.validate(bad)["errors"])
    bad = base_graph()
    bad["nodes"].append({"id": "x", "op": "exec_python", "params": {"code": "import os"}})
    assert any("unknown primitive" in e for e in G.validate(bad)["errors"]), "there is no primitive that runs code"
    bad = base_graph()
    bad["nodes"][2]["params"]["op"] = "~="
    assert any("must be one of" in e for e in G.validate(bad)["errors"])
    bad = base_graph(extra=[{"id": "e", "op": "filter", "inputs": {"data": "d"}, "params": {"expr": "__import__('os').system('x')", "at": "open"}}])
    assert any("not a valid expression" in e for e in G.validate(bad)["errors"]), "expressions are parsed by the rule language, nothing else"
    cyc = {"nodes": [{"id": "a", "op": "combine", "inputs": {"masks": ["b"]}}, {"id": "b", "op": "combine", "inputs": {"masks": ["a"]}}]}
    assert any("cycle" in e for e in G.validate(cyc)["errors"])
    assert any("needs the parameter 'symbol'" in e for e in G.validate({"nodes": [{"id": "d", "op": "data", "params": {}}]})["errors"])
    assert len(G.primitives()) >= 16 and all(p["semantics"] and p["requires"] for p in G.primitives())


def test_a_condition_known_after_the_outcome_starts_is_refused():
    from strata.research import graph as G
    late = base_graph(at="close")  # today's close-to-close return used to explain today's open-to-close move
    errs = G.validate(late)["errors"]
    assert any("only known at close" in e and "information from the future" in e for e in errs)
    intr = {"nodes": [{"id": "d", "op": "data", "params": {"symbol": "TST"}}, {"id": "c", "op": "filter", "inputs": {"data": "d"}, "params": {"expr": "from_open_pct < 0", "at": "12:30"}},
                      {"id": "o", "op": "forward_return", "inputs": {"data": "d"}, "params": {"entry": "open"}}, {"id": "p", "op": "conditional_probability", "inputs": {"outcome": "o", "condition": "c"}}]}
    assert any("12:30" in e for e in G.validate(intr)["errors"])
    intr["nodes"][2]["params"]["entry"] = "12:30"
    assert G.validate(intr)["errors"] == [] and G.validate(intr)["known_at"]["c"] == "12:30"
    intr["nodes"][1]["params"].update(expr="price < vwap", at="open")
    assert any("needs a clock time" in e for e in G.validate(intr)["errors"]), "a value that reads the session so far cannot be read at the open"
    assert any("needs a horizon of one or more sessions" in e for e in G.validate(base_graph(entry="close", at="close"))["errors"])
    ctl = base_graph(extra=[{"id": "k", "op": "feature", "inputs": {"data": "d"}, "params": {"name": "ret_1"}}, {"id": "m", "op": "matched_comparison", "inputs": {"outcome": "o", "condition": "c", "controls": ["k"]}}])
    assert any("defines the condition" in e for e in G.validate(ctl)["errors"])


# ----------------------------------------------------------------------------- one engine
def test_a_graph_reproduces_the_study_engine_and_plain_arithmetic(tmp_path):
    from strata.research import graph as G
    from strata.research.features import get_features
    from strata.research.study import run_study
    st = rich_store(tmp_path, sessions=700, effect=0.9, intraday=0)
    res = G.run_graph(base_graph(), st)
    pr = next(n["result"] for n in res["nodes"] if n["id"] == "p")
    ft = get_features(st, "TST")
    s = ft.ser
    c, o = s.d_c, s.d_o
    prev = np.r_[np.nan, (c[1:] / c[:-1] - 1) * 100]  # the previous session's return, by hand
    asof = np.r_[np.nan, prev[:-1]]
    q = asof <= -2
    up = c > o
    assert pr["evidence"]["n"] == int(q.sum()) and pr["evidence"]["hits"] == int((q & up).sum())
    study = run_study(pr["study"], st)
    assert study["stats"]["qualifying_sessions"] == pr["evidence"]["n"] and study["stats"]["targets"] == pr["evidence"]["hits"] if "targets" in study["stats"] else True
    assert abs(study["stats"]["probability"] - pr["evidence"]["probability"]) < 1e-12
    assert res["limits"][-1].startswith("These are frequencies and returns of the underlying")


def test_a_free_form_expression_condition_equals_the_catalog_feature(tmp_path):
    from strata.research.features import ExprPred, OutcomeSpec, Pred, get_features
    st = rich_store(tmp_path, sessions=400, intraday=0)
    ft = get_features(st, "TST")
    oc = OutcomeSpec("open", "session", "up", 0.0, None, "close")
    a, b = ft.evaluate([Pred("ret_1", "<=", -1.0)], oc), ft.evaluate([ExprPred("return_pct(1) <= -1", "fell")], oc)
    assert (a["qualifying"] == b["qualifying"]).all() and (a["evaluable"] == b["evaluable"]).all() and a["qualifying"].sum() > 5
    assert Pred.load(ExprPred("price > vwap", "above vwap").dump()).src == "price > vwap"


# ----------------------------------------------------------------------------- question -> graph
def test_the_example_question_is_composed_from_spans_subjects_and_predicates(tmp_path):
    from strata.ai import nl_graph as NG
    from strata.research import graph as G
    st = rich_store(tmp_path, sessions=600, intraday=40)
    q = "Find out whether TST is more likely to rally in the afternoon when the morning session sold off, OTH outperformed TST, and VOLX increased overnight."
    g = NG.plan(q, ["TST", "OTH", "VOLX"], "TST", st)
    cl = g["core"]["clauses"]
    assert [c["feature"] or c["expr"] for c in cl] == ["from_open", "(OTH.from_open_pct) - (from_open_pct)", "VOLX_gap"]
    assert [c["at"] for c in cl] == ["12:00", "12:00", "open"] and [c["op"] for c in cl] == ["<=", ">", ">"] and cl[0]["value"] < 0
    assert g["core"]["outcome"]["params"] == {"direction": "up", "entry": "12:00", "horizon": "session"} and g["unsupported"] == [] and "validation" in g["analyses"]
    assert any("same span" in a for a in g["assumptions"]) and any("more than a marginal move" in a for a in g["assumptions"]) and G.validate(g)["errors"] == []


@pytest.mark.parametrize("q,clauses,outcome", [
    ("Is TST more likely to close higher when it gaps down more than 1% and VOLX is above 25?", [("gap", "<=", -1.0, "open"), ("vix_prev", ">", 25.0, "open")], {"entry": "open", "horizon": "session", "direction": "up"}),
    ("Does TST tend to fall over the next 5 days after three consecutive up days when it is above its 200-day moving average?", [("up_streak", ">=", 3.0, "close"), ("sma200_dist", ">", 0.0, "close")], {"entry": "close", "horizon": "sessions:5", "direction": "down"}),
    ("Does a gap up of more than 0.5% on a Monday lead to a lower close?", [("weekday", "==", 0.0, "open"), ("gap", ">=", 0.5, "open")], {"entry": "open", "horizon": "session", "direction": "down"}),
    ("Does TST rise the next day after it fell 2% and volume was heavy?", [("ret_1", "<=", -2.0, "close"), ("vol_ratio", ">=", 1.5, "close")], {"entry": "close", "horizon": "sessions:1", "direction": "up"}),
    ("Is TST more likely to rally into the close when it fell in the first hour?", [("from_open", "<", 0.0, "10:30")], {"entry": "15:00", "horizon": "session", "direction": "up"}),
])
def test_novel_combinations_need_no_new_parser(tmp_path, q, clauses, outcome):
    from strata.ai import nl_graph as NG
    from strata.research import graph as G
    st = rich_store(tmp_path, sessions=500, intraday=30)
    g = NG.plan(q, ["TST", "OTH", "VOLX"], "TST", st)
    got = [(c["feature"] or c["expr"], c["op"], c["value"], c["at"]) for c in g["core"]["clauses"]]
    assert got == clauses and g["unsupported"] == []
    assert {k: g["core"]["outcome"]["params"][k] for k in outcome} == outcome and G.validate(g)["errors"] == []


def test_words_that_are_not_understood_are_reported_never_dropped(tmp_path):
    from strata.ai import nl_graph as NG
    st = rich_store(tmp_path, sessions=300, intraday=0)
    g = NG.plan("What happens to TST when the moon is full and the previous day fell 1%?", ["TST"], "TST", st)
    assert g["unsupported"] == ["the moon is full"] and len(g["core"]["clauses"]) == 1
    g = NG.plan("Does TST rise when it fell 2% the previous day during earnings season?", ["TST"], "TST", st)
    assert any("earnings" in u for u in g["unsupported"])
    assert NG.plan("hello there", ["TST"], "TST", st) is None
    big = NG.plan("Does TST recover after large down days?", ["TST"], "TST", st)
    assert big["core"]["clauses"][0]["value"] < -1 and any("'large' is taken as" in a for a in big["assumptions"]), "a qualitative size becomes a stated number"
    assert big["sweeps"] and big["sweeps"][0]["values"] == ["sessions:1", "sessions:3", "sessions:5"]


def test_controls_comparisons_and_periods_become_graph_structure(tmp_path):
    from strata.ai import nl_graph as NG
    st = rich_store(tmp_path, sessions=400, intraday=0)
    g = NG.plan("Does TST tend to recover after large down days, even after controlling for volatility? Compare with OTH since 2020", ["TST", "OTH", "VOLX"], "TST", st)
    assert g["core"]["controls"] == ["rv21", "vix_prev"] and g["core"]["start"] == "2020-01-01"
    assert any(n["op"] == "matched_comparison" for n in g["nodes"]) and any(n["op"] == "sweep" and n["params"]["param"] == "symbol" and n["params"]["values"] == ["TST", "OTH"] for n in g["nodes"])
    r = NG.retarget(g, "OTH")
    assert r["core"]["symbol"] == "OTH" and r["nodes"][0]["params"]["symbol"] == "OTH" and any("carried over" in a for a in r["assumptions"])


# ----------------------------------------------------------------------------- matched comparisons
def _dates(n):
    out, d = [], dt.date(2000, 1, 3)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return np.array(out)


@pytest.mark.parametrize("true_effect,verdict", [(0.0, "not_distinguishable"), (0.12, "persists")])
def test_matching_removes_a_confounder_and_keeps_a_real_difference(true_effect, verdict):
    from strata.research import matching as M
    rng = np.random.default_rng(5)
    n = 6000
    vol = np.exp(0.9 * np.sin(np.arange(n) / 150) + rng.normal(0, 0.3, n))
    treated = rng.random(n) < 1 / (1 + np.exp(-(2.0 * np.log(vol) - 2.2)))  # the condition is more common when volatility is high
    y = (rng.random(n) < np.clip(0.15 + 0.12 * np.log(vol) + true_effect * treated, 0.01, 0.99)).astype(float)  # and so is the outcome
    r = M.matched_comparison(_dates(n), y, treated, ~treated, [{"name": "vol", "label": "volatility", "values": vol}])
    assert r["ordinary"]["estimate"] > 0.05, "the plain comparison shows a difference either way"
    assert r["verdict"] == verdict and abs(r["nearest"]["estimate"] - true_effect) < 0.04 and abs(r["stratified"]["estimate"] - true_effect) < 0.05 and abs(r["regression"]["estimate"] - true_effect) < 0.04
    assert abs(r["balance"][0]["before"]) > 0.8 and abs(r["balance"][0]["after_nearest"]) < 0.05, "the groups are comparable on the control after matching"
    assert "not a causal effect" in r["notes"][-1]


def test_matching_says_when_it_cannot_estimate():
    from strata.research import matching as M
    d = _dates(200)
    y = np.zeros(200)
    t = np.zeros(200, bool)
    t[:5] = True
    r = M.matched_comparison(d, y, t, ~t, [{"name": "x", "label": "x", "values": np.arange(200.0)}])
    assert r["verdict"] == "not_estimable" and "too few" in r["notes"][0]
    assert M.matched_comparison(d, y, t, ~t, [])["verdict"] == "not_estimable"


# ----------------------------------------------------------------------------- validation, holdouts, registry
def test_a_real_effect_is_confirmed_once_and_noise_never_touches_the_final_period(tmp_settings):
    from strata.research import graph as G
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    g = base_graph(extra=[{"id": "v", "op": "validation", "inputs": {"outcome": "o", "condition": "c"}}])
    sid = session()
    res = G.run_graph(g, st, scope=sid, run_id="r1")
    v = next(n["result"] for n in res["nodes"] if n["id"] == "v")
    assert v["status"] == "confirmed" and v["test"]["n"] >= 8 and LG.current(sid, "TST")["status"] == "used"
    d = next(n["info"] for n in res["nodes"] if n["id"] == "d")
    assert d["last"] < v["holdout"]["start"], "everything else in the graph was measured before the final test period"
    noise = rich_store(tmp_settings / "n", sessions=1500, effect=0.0, intraday=0, sym="NSE")
    g2 = base_graph("NSE", extra=[{"id": "v", "op": "validation", "inputs": {"outcome": "o", "condition": "c"}}])
    sid2 = session()
    v2 = next(n["result"] for n in G.run_graph(g2, noise, scope=sid2)["nodes"] if n["id"] == "v")
    assert v2["status"] == "not_a_candidate" and v2["test"] is None and LG.current(sid2, "NSE")["status"] == "sealed"


def test_a_final_period_examined_elsewhere_is_not_an_independent_confirmation(tmp_settings):
    from strata.research import graph as G
    from strata.research import registry as RG
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    other = session()
    G.run_graph(base_graph(value=-1.8), st, scope=other)  # another notebook looked at the whole history with a related configuration
    sid = session()
    g = base_graph(extra=[{"id": "v", "op": "validation", "inputs": {"outcome": "o", "condition": "c"}}])
    v = next(n["result"] for n in G.run_graph(g, st, scope=sid)["nodes"] if n["id"] == "v")
    assert v["status"] == "confirmed_not_independent" and v["global_exposure"]["related"] >= 1 and "NOT an independent confirmation" in v["summary"][0]
    ov = RG.overview()
    assert ov["notebooks"] == 2 and ov["configurations"] >= 2 and any(h["scope"] == sid and not h["independent"] for h in ov["holdouts"])
    key = next(n["result"]["key"] for n in G.run_graph(base_graph(value=-1.8), st, scope=sid)["nodes"] if n["id"] == "p")
    dup = RG.duplicates(key)
    assert {d["scope"] for d in dup} == {other, sid}, "the identical configuration is found in both notebooks"
    chk = RG.check(sid, "TST", key, {"symbol": "TST", "features": ["ret_1"], "outcome": "open|session|up|close"})
    assert chk["exact"] and chk["related"] and chk["holdouts"] and any("not an independent confirmation" in s for s in chk["sentences"])
    assert RG.audit(symbol="TST")["total"] >= 3 and RG.audit(scope=sid, phase="validation")["total"] >= 1


def test_the_final_period_of_an_intraday_graph_is_cut_from_observable_sessions(tmp_settings):
    from strata.research import graph as G
    st = rich_store(tmp_settings, sessions=700, effect=0.0, intraday=300)
    g = {"nodes": [{"id": "d", "op": "data", "params": {"symbol": "TST"}}, {"id": "c", "op": "filter", "inputs": {"data": "d"}, "params": {"expr": "from_open_pct < 0", "at": "12:00"}},
                   {"id": "o", "op": "forward_return", "inputs": {"data": "d"}, "params": {"entry": "12:00"}}, {"id": "p", "op": "conditional_probability", "inputs": {"outcome": "o", "condition": "c"}},
                   {"id": "v", "op": "validation", "inputs": {"outcome": "o", "condition": "c"}}]}
    res = G.run_graph(g, st, scope=session())
    p = next(n["result"] for n in res["nodes"] if n["id"] == "p")
    assert 200 <= p["evidence"]["evaluated"] <= 245, "about 80% of the 300 intraday sessions are explored; the rest are held back"
    assert next(n["info"] for n in res["nodes"] if n["id"] == "d")["holdout"]["symbol"] == "TST@INTRADAY"


def test_looking_at_a_sealed_final_period_is_recorded_as_exposure(tmp_settings):
    from strata.research import graph as G
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=1500, effect=0.0, intraday=0)
    sid = session()
    G.run_graph(base_graph(extra=[{"id": "v", "op": "validation", "inputs": {"outcome": "o", "condition": "c"}}]), st, scope=sid)
    assert LG.current(sid, "TST")["status"] == "sealed"
    G.run_graph({**base_graph(value=-1.5), "include_holdout": True}, st, scope=sid)
    h = LG.current(sid, "TST")
    assert h["status"] == "exposed" and "before anything was scored" in h["reason"]


def test_a_sweep_lists_every_value_and_records_each_as_a_trial(tmp_settings):
    from strata.research import graph as G
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=900, effect=1.0, intraday=0)
    sid = session()
    g = base_graph(extra=[{"id": "s", "op": "sweep", "inputs": {"result": "p"}, "params": {"node": "c", "param": "value", "values": [-1.0, -2.0, -3.0, -40.0]}},
                          {"id": "x", "op": "sweep", "inputs": {"result": "p"}, "params": {"node": "d", "param": "symbol", "values": ["TST", "OTH", "ZZZ"]}}])
    res = G.run_graph(g, st, scope=sid)
    s = next(n["result"] for n in res["nodes"] if n["id"] == "s")
    assert [r["value"] for r in s["rows"]] == [-1.0, -2.0, -3.0, -40.0] and s["rows"][3]["n"] == 0 and all("p_adjusted" in r for r in s["rows"])
    x = next(n["result"] for n in res["nodes"] if n["id"] == "x")
    assert [r["status"] for r in x["rows"]] == ["ok", "ok", "error"] and "ZZZ" in x["rows"][2]["error"], "a symbol with no data is reported, not skipped"
    assert LG.count(sid) + LG.count(sid, "OTH") >= 5 and {r["kind"] for r in LG.summary(sid)["by_kind"]} == {"graph", "sweep"}


def test_robustness_reports_idle_clauses_and_unstable_thresholds(tmp_path):
    from strata.research import graph as G
    st = rich_store(tmp_path, sessions=1500, effect=1.2, intraday=0)
    g = base_graph(extra=[{"id": "w", "op": "feature", "inputs": {"data": "d"}, "params": {"name": "weekday"}}, {"id": "c2", "op": "filter", "inputs": {"x": "w"}, "params": {"op": ">=", "value": 0}},
                          {"id": "both", "op": "combine", "inputs": {"masks": ["c", "c2"]}}, {"id": "r", "op": "robustness", "inputs": {"outcome": "o", "condition": "both"}}])
    r = next(n["result"] for n in G.run_graph(g, st)["nodes"] if n["id"] == "r")
    idle = next(a for a in r["ablation"] if "weekday" in a["removed"])
    assert idle["contributes"] is False and any("not adding evidence" in f for f in r["flags"]), "a clause that is always true is called out"
    assert next(a for a in r["ablation"] if "return" in a["removed"])["contributes"] is True and len(r["periods"]) == 5 and r["years"] and r["neighbours"]


# ----------------------------------------------------------------------------- patterns
def test_the_pattern_explorer_matches_plain_arithmetic(tmp_path):
    from strata.research import patterns as PT
    from strata.research.features import get_features
    from strata.research.hypothesis import structure_variants
    st = rich_store(tmp_path, sessions=900, effect=0.0, intraday=0)
    p = PT.explore("TST", {"id": "down_day", "params": {"x": 2}}, st, horizon=5, target=1.0, stop=1.0, overlays=[{"id": "red_streak", "params": {"n": 3}}, {"id": "nope"}])
    ft = get_features(st, "TST")
    c = ft.ser.c[ft.ser.day_end - 1]
    ret = np.r_[np.nan, (c[1:] / c[:-1] - 1) * 100]
    idx = np.flatnonzero(ret <= -2)
    assert p["n"] == idx.size and p["dates"] == [ft.dates[i] for i in idx]
    ok = idx[idx + 1 < ft.n]
    nxt = (c[ok + 1] / c[ok] - 1) * 100
    h1 = next(h for h in p["horizons"] if h["horizon"] == 1)
    assert h1["n"] == ok.size and abs(h1["mean"] - nxt.mean()) < 1e-9 and abs(p["paths"]["bands"]["mean"][1] - nxt.mean()) < 1e-3
    assert p["paths"]["steps"][:2] == ["event close", "+1"] and len(p["paths"]["paths"]) == min(idx.size, 400) and p["paths"]["baseline"]["n"] > 0
    assert p["target"]["evidence"]["n"] > 0 and p["regimes"] and p["by_year"] and p["overlays"][0]["n"] > 0 and "error" in p["overlays"][1]
    variants, _ = structure_variants("TST", p["cell"]["preds"], p["cell"]["outcome"], True, False)
    assert variants and "return_pct" in variants[0]["draft"].conditions[0].expr, "a pattern becomes an entry rule the strategy engine accepts"
    with pytest.raises(ValueError):
        PT.explore("TST", {"id": "custom", "params": {"expr": "position.pnl > 0", "at": "close"}}, st)
    with pytest.raises(ValueError):
        PT.explore("TST", {"id": "down_day", "params": {"x": 99}}, st)
    assert len(PT.catalog()) >= 12


# ----------------------------------------------------------------------------- memory
def test_memory_is_derived_from_runs_and_never_invents_a_finding(tmp_settings):
    from strata.research import memory as MEM
    sid = session()
    mem = MEM.notebook(sid)
    assert mem["findings"] == [] and mem["counts"]["findings"] == 0 and MEM.strongest(sid) is None and "No finding is on record" in MEM.summary_text(mem)
    fe = MEM.filter_effect([5, 5, -1, -1, -1, -1, 0.2, 0.2, 9, -2], [0.2, 0.2, 0.3, 0.1, -3, 0.2])
    d = fe["decomposition"]
    assert fe["after"]["win_rate"] > fe["before"]["win_rate"] and fe["after"]["expectancy"] < fe["before"]["expectancy"] and "carried the large gains" in fe["sentence"]
    assert abs(d["from_win_rate"] + d["from_win_size"] + d["from_loss_size"] - d["total"]) < 1e-9, "the three sources add up to the change in expectancy"
    ind = MEM.independence([{"label": "a", "dates": {"1", "2", "3", "4"}}, {"label": "b", "dates": {"1", "2", "3"}}, {"label": "c", "dates": {"8", "9"}}])
    assert ind["independent_groups"] == 2 and ind["groups"][0]["members"] == ["a", "b"]
    assert MEM.day_of("what we discovered yesterday", dt.date(2026, 11, 25)) == dt.date(2026, 11, 24)


# ----------------------------------------------------------------------------- the agent
def test_a_composed_question_runs_as_a_graph_and_follow_ups_reuse_it(tmp_settings):
    from strata.api import db
    from strata.research import ledger as LG
    from strata.research import memory as MEM
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    m = ag.post(sid, "Does TST tend to close higher when the previous day fell more than 2%, even after controlling for volatility?")[-1]
    c = m["content"]
    assert m["kind"] == "graph" and c["state"] == "running" and c["graph"]["core"]["controls"] == ["rv21", "vix_prev"] and c["data"][0]["ok"] and [n["op"] for n in c["graph"]["nodes"]][0] == "data"
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "graph" and {"probability", "matched", "robustness", "validation"} <= set(res["analyses"]) and res["findings"] and res["explanation_source"] == "deterministic"
    assert res["findings"][0]["evidence"] in ("confirmed", "supported") and res["findings"][0]["convertible"] and ag.finalize(sid) == []
    assert db.list_messages(sid)[-2]["content"]["state"] == "done"
    # follow-ups edit the stored core; none of them needs a parser of its own
    f1 = ag.post(sid, "Apply the same idea to OTH.")[-1]["content"]
    assert f1["graph"]["core"]["symbol"] == "OTH" and "applied to OTH" in f1["question"]
    assert ag.finalize(sid)[0]["content"]["symbols"] == ["OTH"]
    f2 = ag.post(sid, "Show the distribution of price paths following those events.")[-1]["content"]
    assert [n["op"] for n in f2["graph"]["nodes"]][-1] == "distribution"
    ag.finalize(sid)
    f3 = ag.post(sid, "Find which historical conditions are associated with that outcome.")[-1]["content"]
    assert any(n["op"] == "regression" for n in f3["graph"]["nodes"]) and "within-sample" in f3["text"]
    ag.finalize(sid)
    f4 = ag.post(sid, "Compare that behavior with TST.")[-1]["content"]
    assert f4["graph"]["sweeps"][0]["values"] == ["OTH", "TST"]
    assert ag.finalize(sid)[0]["content"]["has_sweep"]
    assert len(MEM.graphs_of(sid)) == 5 and LG.count(sid) >= 3
    mem = ag.post(sid, "What have we found so far?")[-1]["content"]
    assert mem["view"] == "memory" and mem["memory"]["findings"] and mem["memory"]["tested_conditions"] and "on record" in mem["text"]
    reg = ag.post(sid, "Check whether I've already tested similar configurations on this holdout.")[-1]["content"]
    assert reg["view"] == "registry" and reg["check"]["holdouts"] and "final test period" in reg["text"]
    ind = ag.post(sid, "Which of these results are actually independent?")[-1]["content"]
    assert ind["view"] == "analysis" and "independent set" in ind["text"]


def test_a_finding_becomes_option_structures_only_as_drafts_until_they_are_priced(tmp_settings):
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    assert "no recorded findings" in ag.post(sid, "Build a strategy from the strongest supported finding.")[-1]["content"]["text"]
    ag.post(sid, "Does TST tend to close higher when the previous day fell more than 2% and VOLX is above 5?")
    ag.finalize(sid)
    sp = ag.post(sid, "Build a strategy from the strongest supported finding.")[-1]
    c = sp["content"]
    assert sp["kind"] == "strategy_plan" and len(c["variants"]) >= 4 and c["finding"]["evidence"] in ("confirmed", "supported") and "None has been priced" in c["text"] and c["launched"] is False
    inv = ag.post(sid, "Backtest that strategy using observed historical options prices.")[-1]
    assert inv["kind"] == "investigation" and inv["content"]["plan"]["stage"] == "options" and "observed premiums only" in inv["content"]["text"]
    done = ag.finalize(sid)[0]["content"]
    assert done["view"] == "failed" and "None of the planned configurations could be run" in done["text"], "no option prices are stored, so nothing is priced and nothing is invented"


def test_an_unsupported_strategy_idea_is_refused_unless_asked_for_anyway(tmp_settings):
    st = rich_store(tmp_settings, sessions=1200, effect=0.0, intraday=0)
    ag, sid = agent(st), session()
    ag.post(sid, "Is TST more likely to close higher when the previous day fell more than 2% and VOLX is above 5?")
    ag.finalize(sid)
    t = ag.post(sid, "Build a strategy from the strongest supported finding.")[-1]
    assert t["kind"] == "text" and "building on noise" in t["content"]["text"]
    a = ag.post(sid, "Build a strategy from the strongest finding anyway.")[-1]
    assert a["kind"] == "strategy_plan" and "NOT supported by its evidence" in a["content"]["text"]


def test_a_partly_understood_question_waits_for_the_user(tmp_settings):
    st = rich_store(tmp_settings, sessions=600, effect=0.0, intraday=0)
    ag, sid = agent(st), session()
    m = ag.post(sid, "Is TST more likely to rise when the previous day fell 1% and VOLX is above 5 and the moon is full?")[-1]
    assert m["kind"] == "graph" and m["content"]["state"] == "awaiting" and m["content"]["runs"] == [] and "the moon is full" in m["content"]["text"] and ag.finalize(sid) == []
    ran = ag.decide_graph(sid, m["id"], "run")[0]["content"]
    assert ran["state"] == "running" and any("Run WITHOUT" in a for a in ran["raw"]["assumptions"])
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "graph" and any("Run WITHOUT" in c for c in res["caveats"])
    m2 = ag.post(sid, "Is TST more likely to rise when ZZZQ gapped down 1% and VOLX is above 5?")[-1]
    assert m2["content"]["state"] in ("needs_data", "awaiting")


def test_memory_follow_ups_refuse_to_invent_and_combine_stored_conditions(tmp_settings):
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    assert "no recorded pattern findings" in ag.post(sid, "Which of our previous findings still hold after 2022?")[-1]["content"]["text"]
    assert "No finding was recorded" in ag.post(sid, "Find something similar to what we discovered yesterday.")[-1]["content"]["text"]
    ag.post(sid, "Is TST more likely to close higher when the previous day fell more than 2% and VOLX is above 5?")
    ag.finalize(sid)
    ag.post(sid, "Is TST more likely to close higher when it gaps down more than 0.2% and volatility is high?")
    ag.finalize(sid)
    cmb = ag.post(sid, "Combine the gap condition from our second study with the down day condition from the first.")[-1]["content"]
    labels = [c["label"] for c in cmb["graph"]["core"]["clauses"]]
    assert len(labels) == 2 and "overnight" in labels[0] and "previous session" in labels[1] and "new configuration" in cmb["text"]
    ag.finalize(sid)
    still = ag.post(sid, "Which of our previous findings still hold after 2022?")[-1]
    if still["content"]["state"] == "awaiting":  # the later period is this notebook's final test period: looking at it needs a yes
        assert "holds back as its final test period" in still["content"]["text"]
        still = ag.decide_graph(sid, still["id"], "run")[-1]
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "graph_batch" and res["rows"] and all(r["status"] == "ok" and r["window"][0] >= "2022-01-01" and r.get("verdict") for r in res["rows"])
    sim = ag.post(sid, "Find something similar to what we found today.")[-1]["content"]
    assert sim["graph"]["batch"] and "closely related configurations" in sim["text"]


class FakeLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def json(self, system, msgs, max_tokens=3000, schema=None):
        self.calls.append((system, [dict(m) for m in msgs]))
        return self.replies.pop(0)


def test_a_model_proposed_graph_goes_through_the_same_validator(tmp_path):
    from strata.ai.agent import llm_interpret
    st = rich_store(tmp_path, sessions=300, intraday=0)
    bad = base_graph(at="close")["nodes"]
    good = base_graph()["nodes"]
    llm = FakeLLM([{"tool": "research_graph", "args": {"nodes": bad, "assumptions": [], "unsupported": []}}, {"tool": "research_graph", "args": {"nodes": good, "assumptions": ["x"], "unsupported": []}}])
    r = llm_interpret(llm, "does TST rise after a down day", None, [], st, lambda: None)
    assert r["action"] == "graph" and r["graph"]["nodes"] == good and len(llm.calls) == 2
    assert "information from the future" in llm.calls[1][1][-1]["content"], "the validator's message is what the model is asked to fix"
    assert "research_graph" in llm.calls[0][0] and "conditional_probability" in llm.calls[0][0] and "no code" in llm.calls[0][0]


def test_symbols_are_compared_on_one_period_that_stops_before_the_held_back_sessions(tmp_settings):
    """"Compare that with OTH": each symbol's own condition (the label says so), every symbol on the same dates, and
    none of them on the sessions this notebook holds back - they are the same market days as the base symbol's."""
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    ag.post(sid, "Does TST tend to close higher when the previous day fell more than 2%, even after controlling for volatility?")
    first = ag.finalize(sid)[0]["content"]
    assert len(first["text"]) < 1200, "the chat card is a headline, not the whole report"
    from strata.research import ledger as LG
    hold = LG.current(sid, "TST")
    assert hold and hold["start"]
    ag.post(sid, "Compare that behavior with OTH.")
    res = ag.finalize(sid)[0]["content"]
    assert res["text"].startswith("TST: ") and "; OTH: " in res["text"] and "Every symbol is evaluated on the same period" in res["text"]
    run = MEM_graph(res["run_id"])
    sw = next(n["result"] for n in run["nodes"] if n.get("result") and n["result"]["kind"] == "sweep")
    rows = {r["value"]: r for r in sw["rows"]}
    assert rows["OTH"]["condition"].startswith("OTH ") and rows["TST"]["condition"].startswith("TST ")
    assert rows["OTH"]["window"] == rows["TST"]["window"] == sw["period"]
    assert sw["period"][1] < hold["start"], "another symbol must not be read on the dates held back for the final test"


def MEM_graph(run_id: str) -> dict:
    import json

    from strata.api.results import run_dir
    return json.loads((run_dir(run_id) / "graph.json").read_text())


def test_pricing_options_on_an_unscored_final_period_is_recorded_as_a_look(tmp_settings):
    """An at-the-money call's profit on a session is very nearly the price pattern's outcome on it. Pricing option
    positions on held-back sessions BEFORE the pattern was scored there means the period was seen; after the pattern
    has had its one scoring, pricing structures on it does not re-choose anything and leaves it as it is."""
    from datetime import date, timedelta

    from strata.research import ledger as LG
    days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(400)]
    h = LG.reserve("nb-a", "TST", days)
    assert h["status"] == "sealed"
    early = LG.record("nb-a", "TST", [{"kind": "option-structure", "key": "s0", "label": "atm call", "start": days[0], "end": days[100]}])
    assert early["contaminated"] is None and LG.current("nb-a", "TST")["status"] == "sealed"  # priced before the held-back sessions: not a look
    LG.record("nb-a", "TST", [{"kind": "option-structure", "key": "s1", "label": "atm call"}, {"kind": "option-structure", "key": "s2", "label": "atm put"}])
    cur = LG.current("nb-a", "TST")
    assert cur["status"] == "exposed" and "2 option positions were priced" in cur["reason"] and LG._pub(cur)["label"] == "already seen"
    h2 = LG.reserve("nb-b", "TST", days)
    LG.use(h2["id"], ["k1"], "run1")
    LG.record("nb-b", "TST", [{"kind": "option-structure", "key": "s1", "label": "atm call"}])
    assert LG.current("nb-b", "TST")["status"] == "used"


def test_deleting_a_notebook_removes_its_findings_but_never_its_trials(tmp_settings):
    from strata.api import db
    from strata.research import ledger as LG
    from strata.research import memory as MEM
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    ag.post(sid, "Does TST tend to close higher when the previous day fell more than 2%, even after controlling for volatility?")
    ag.finalize(sid)
    assert MEM.list_findings(sid) and LG.count(sid) > 0
    n = LG.count(sid)
    db.delete_session(sid)
    assert MEM.list_findings(sid) == [] and db.get_session(sid) is None
    assert LG.count(sid) == n, "what was tested on which dates is kept: a deleted notebook must not make a period look fresh again"
