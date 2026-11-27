"""V4 research: the session-feature engine against the rule engine and against an independent
calculation, look-ahead protection, hypothesis plans, the research ledger (trial counting and
contamination of the final test period), the agent flow, reasoning tools and the reviewer."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401  (fixtures)
from tests.test_v3_agent import agent, session

QUESTIONS = ["Investigate whether SPY tends to reverse following large down days.", "Does the first 15-minute candle predict the rest of the session?", "What conditions are associated with large intraday moves?",
             "Compare mean reversion and momentum after overnight gaps.", "How does the effectiveness of opening range breakouts change with volatility?",
             "Investigate whether calls or puts perform differently following consecutive red days.", "What historical conditions are associated with a 1% rally before a 0.5% decline?"]


def rich_store(tmp, sessions: int = 900, effect: float = 0.0, intraday: int = 14, seed: int = 3, sym: str = "TST"):
    """Daily bars for TST and a second symbol plus a volatility index; the last `intraday`
    sessions also have 5-minute bars. With `effect`, a session after a fall of 2% or more
    closes that many percent above its open (a real, directional reversal)."""
    from strata.data.store import MarketStore, SymbolMeta
    st = MarketStore(tmp / "market")
    rng = np.random.default_rng(seed)
    days, d = [], date(2019, 1, 2)
    while len(days) < sessions:
        if cal.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)

    def walk(vol, eff=0.0):
        rows, px, prev_ret = [], 100.0, 0.0
        for _ in days:
            gap = rng.standard_normal() * 0.003
            o = px * (1 + gap)
            body = rng.standard_t(4) * vol
            if eff and prev_ret <= -0.02:
                body = eff / 100 + abs(rng.standard_normal()) * 0.001
            c = o * (1 + body)
            hi = max(o, c) * (1 + abs(rng.standard_normal()) * 0.003)
            lo = min(o, c) * (1 - abs(rng.standard_normal()) * 0.003)
            rows.append((o, hi, lo, c, float(rng.integers(1e6, 5e6))))
            prev_ret, px = c / px - 1, c
        return rows
    for s, vol, eff in ((sym, 0.011, effect), ("OTH", 0.013, 0.0)):
        r = walk(vol, eff)
        st.upsert_symbol(SymbolMeta(symbol=s, kind="etf", vol_index="VOLX"))
        st.write_daily(s, pl.DataFrame({"date": days, "open": [x[0] for x in r], "high": [x[1] for x in r], "low": [x[2] for x in r], "close": [x[3] for x in r], "volume": [x[4] for x in r]}), provider="yahoo")
        if s == sym and intraday:
            bars = []
            for d_, x in list(zip(days, r))[-intraday:]:
                o_ts = int(cal.session_bounds(d_)[0].timestamp())
                n = cal.session_minutes(d_) // 5
                path = np.linspace(x[0], x[3], n + 1) + np.r_[0, rng.standard_normal(n - 1) * x[0] * 0.0015, 0]
                for i in range(n):
                    a, b = path[i], path[i + 1]
                    bars.append((o_ts + 300 * i, a, max(a, b) * 1.0004, min(a, b) * 0.9996, b, float(rng.integers(1e4, 9e4))))
            df = pl.DataFrame(bars, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
            st.write_bars(s, "5m", df, provider="yahoo")
    v = 14 + np.abs(np.cumsum(rng.standard_normal(len(days)) * 0.7)) % 28
    st.upsert_symbol(SymbolMeta(symbol="VOLX", kind="volatility_index", has_options=False))
    st.write_daily("VOLX", pl.DataFrame({"date": days, "open": v, "high": v + 1, "low": v - 1, "close": v, "volume": [0.0] * len(days)}), provider="cboe")
    for s in (sym, "OTH", "VOLX"):
        st.rebuild_manifest(s)
    return st


# ----------------------------------------------------------------------------- features == rule engine == independent arithmetic
def test_every_feature_equals_the_rule_language_expression_it_is_tied_to(tmp_path):
    from strata.research.features import FEATURES, SessionFeatures, feature_def
    ft = SessionFeatures(rich_store(tmp_path), "TST", ("OTH",))
    sample = np.r_[np.arange(1, 30), np.arange(260, 420, 7), np.arange(ft.n - 30, ft.n)]
    names = [n for n, f in FEATURES.items() if f.kind != "intraday"] + ["OTH_ret_1", "OTH_gap", "rs5_OTH"]
    for entry in ("open", "close"):
        for nm in names:
            fd = feature_def(nm, "TST")
            expr = fd.expr_close if (entry == "close" and fd.expr_close) else fd.expr
            a, b = ft.col(nm, entry)[sample], ft.engine_values(expr, entry, sample)[sample]
            assert (np.isfinite(a) == np.isfinite(b)).all(), f"{nm} at the {entry}: evaluable on different sessions"
            ok = np.isfinite(a)
            assert np.allclose(a[ok], b[ok], atol=1e-9), f"{nm} at the {entry} differs from `{expr}`"
    for nm in ("from_open", "range_so_far", "or15_pos", "or30_pos", "vwap_dist"):  # intraday features ARE the engine's values, on sessions with bars
        x = ft.col(nm, "10:30")
        assert np.isfinite(x[-10:]).all() and not np.isfinite(x[:ft.n - 14]).any(), nm


@pytest.mark.parametrize("preds,oc", [
    ([("ret_1", "<=", -2)], ("open", "session", "up", 0.0, None, "close")),
    ([("ret_1", "<=", -1)], ("open", "session", "up", 1.0, 0.5, "touch")),
    ([("gap", "<=", -0.3), ("vix_prev", ">=", 20)], ("open", "sessions:3", "up", 2.0, 1.0, "touch")),
    ([("down_streak", ">=", 2)], ("close", "sessions:5", "up", 0.0, None, "close")),
    ([("ret_1", "<=", -1)], ("close", "sessions:2", "down", 1.5, 1.5, "touch")),
    ([("from_open", ">", 0)], ("10:30", "session", "up", 0.0, None, "close")),
    ([("sma200_dist", "<", 0), ("OTH_ret_1", "<=", -1)], ("open", "sessions:1", "up", 0.0, None, "close")),
])
def test_vectorised_outcomes_reproduce_the_session_by_session_study(tmp_path, preds, oc):
    from strata.research.features import CODE_NAMES, OutcomeSpec, Pred, SessionFeatures, to_study
    from strata.research.study import run_study
    st = rich_store(tmp_path)
    ft = SessionFeatures(st, "TST", ("OTH",))
    ps, o = [Pred(*p) for p in preds], OutcomeSpec(*oc)
    ev = ft.evaluate(ps, o)
    mine = {ft.dates[i]: CODE_NAMES[ev["outcome"].code[i]] for i in np.flatnonzero(ev["qualifying"])}
    res = run_study(to_study("TST", ps, o), st, max_sessions_out=100000)
    theirs = {r["date"]: r["outcome"] for r in res["sessions"]}
    assert mine == theirs and len(mine) > 0
    assert int(ev["evaluable"].sum()) == res["stats"]["evaluated_sessions"]
    for r in res["sessions"]:
        assert abs(ev["outcome"].end_ret[ft.dates.index(r["date"])] - r["end_return_pct"]) < 1e-3


def test_a_study_matches_an_independent_plain_python_calculation(tmp_path):
    """No shared code: the conditional frequency is recomputed from the stored daily rows
    with ordinary loops and compared with the study engine and the feature engine."""
    from strata.research.features import OutcomeSpec, Pred, SessionFeatures, to_study
    from strata.research.study import run_study
    st = rich_store(tmp_path, intraday=0)
    rows = st.read_daily("TST").sort("date").to_dicts()
    q = hit = base = base_hit = amb = t_first = 0
    for i in range(2, len(rows)):
        prev_ret = (rows[i - 1]["close"] / rows[i - 2]["close"] - 1) * 100
        o, h, l, c = (rows[i][k] for k in ("open", "high", "low", "close"))
        up = c > o
        base += 1
        base_hit += up
        if prev_ret <= -1.5:
            q += 1
            hit += up
            tt, ss = h >= o * 1.01, l <= o * 0.995  # +1% before -0.5%, one daily bar: both touched = unknowable
            amb += tt and ss
            t_first += tt and not ss
    res = run_study(to_study("TST", [Pred("ret_1", "<=", -1.5)], OutcomeSpec("open", "session", "up", 0.0, None, "close")), st)["stats"]
    assert res["qualifying_sessions"] == q and res["target_first"] == hit and res["evaluated_sessions"] == base
    assert math.isclose(res["probability"], hit / q) and math.isclose(res["baseline_probability"], base_hit / base)
    touch = run_study(to_study("TST", [Pred("ret_1", "<=", -1.5)], OutcomeSpec("open", "session", "up", 1.0, 0.5, "touch")), st)["stats"]
    assert touch["ambiguous"] == amb and touch["target_first"] == t_first
    ev = SessionFeatures(st, "TST").evaluate([Pred("ret_1", "<=", -1.5)], OutcomeSpec("open", "session", "up", 1.0, 0.5, "touch"))
    assert int((ev["outcome"].code[ev["qualifying"]] == 3).sum()) == amb and int((ev["outcome"].code[ev["qualifying"]] == 0).sum()) == t_first


# ----------------------------------------------------------------------------- look-ahead
def test_a_lookback_of_zero_or_less_is_rejected_before_it_can_read_the_future(tmp_path):
    from strata.engine import expr as X
    from strata.research.features import SessionFeatures
    for bad in ("close(0) > open", "return_pct(-1) > 0", "session_high(0) > prev_high", "sma(0) > 1", "rolling_high(-5) > price", 'return_pct("OTH", 0) > 0', "rvol(0) > 5", "open_to_close(-2) > 0"):
        assert X.validate(bad), bad
    for good in ("close(1) > open", 'return_pct("OTH", 1) > 0', "sma(20) > sma(200)", "bb_upper(20, 0.5) > price", 'volume_ratio("5m", 20) > 2'):
        assert X.validate(good) == [], good
    ft = SessionFeatures(rich_store(tmp_path, intraday=0), "TST")
    n = 2  # a computed count is caught at evaluation: nothing is returned for today's unfinished close
    assert not np.isfinite(ft.engine_values(f"close({n} - 2)", "open", np.arange(50, 60))).any()


def test_scrambling_the_future_changes_no_feature_and_exposes_a_leaky_rule(tmp_path, monkeypatch):
    from strata.research.challenge import ShockedStore, lookahead_probe_study
    from strata.research.features import FEATURES, OutcomeSpec, Pred, SessionFeatures, to_study
    from strata.research.study import run_study
    st = rich_store(tmp_path, intraday=0)
    ft = SessionFeatures(st, "TST", ("OTH",))
    i = 500
    shocked = SessionFeatures(ShockedStore(st, int(ft.t_open[i])), "TST", ("OTH",))
    assert not np.isclose(shocked.ser.d_c[i], ft.ser.d_c[i]) and np.isclose(shocked.ser.d_c[i - 1], ft.ser.d_c[i - 1]) and np.isclose(shocked.ser.d_o[i], ft.ser.d_o[i])
    for nm, f in FEATURES.items():
        if f.kind != "intraday":
            a, b = ft.col(nm, "open")[i], shocked.col(nm, "open")[i]
            assert (not np.isfinite(a) and not np.isfinite(b)) or np.isclose(a, b), f"{nm} read a price from after the open"
    cfg = to_study("TST", [Pred("ret_1", "<=", -1.5)], OutcomeSpec("open", "session", "up", 0.0, None, "close"))
    res = run_study(cfg, st)
    clean = lookahead_probe_study(cfg, res, st)
    assert clean["checked"] >= 8 and clean["changed"] == 0
    # a rule that peeks: today's final close, readable only because the count check is switched off
    from strata.engine import expr as X
    monkeypatch.setattr(X.Evaluator, "_count", staticmethod(lambda n, what="n", lo=1: int(n)))
    monkeypatch.setattr(X, "count_errors", lambda tree: [])
    leaky = {**cfg, "condition": {"id": "root", "kind": "group", "op": "and", "children": [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": "close(0) > open * 1.004"}}]}}
    bad = lookahead_probe_study(leaky, run_study(leaky, st), st)
    assert bad["changed"] > 0, "the future-scramble test must catch a rule that reads the session's own close at the open"


# ----------------------------------------------------------------------------- plans
def test_each_example_question_becomes_an_explicit_bounded_plan(tmp_path):
    from strata.research import hypothesis as HY
    st = rich_store(tmp_path, intraday=0)
    for q in QUESTIONS:
        plan = HY.build_plan(q.replace("SPY", "TST"), "TST", st, ["TST", "OTH", "VOLX"])
        assert plan is not None, q
        assert 4 <= len(plan["configs"]) <= plan["max_configs"] == 60 and HY.validate_plan(plan) == []
        for k in ("hypothesis", "null", "baseline", "outcomes", "primary_metric", "min_sample", "notes", "families"):
            assert plan[k], (q, k)
        assert len({c["key"] for c in plan["configs"]}) == len(plan["configs"])
        assert all(c["preds"] and c["outcome"]["measure"] in ("touch", "close") for c in plan["configs"])
        data = HY.describe_data(plan, st)
        assert data["required"][0]["what"] == "Daily bars" and data["sessions"] == 900
    assert HY.build_plan("What is the weather like tomorrow?", "TST", st) is None
    big = dict(plan, configs=plan["configs"] * 6)
    assert any("bound" in e for e in HY.validate_plan(big))
    assert any("unknown feature" in e for e in HY.validate_plan(dict(plan, configs=[{**plan["configs"][0], "preds": [{"feature": "astrology", "op": ">", "value": 1}]}])))


def test_a_real_effect_is_selected_on_training_data_and_confirmed_once_on_the_final_period(tmp_path):
    from strata.research import hypothesis as HY
    st = rich_store(tmp_path, sessions=1500, effect=1.2, intraday=0)
    plan = HY.build_plan("Investigate whether TST tends to reverse following large down days.", "TST", st, ["TST", "OTH"])
    res = HY.run_plan(plan, st)
    assert res["tested"] == len(plan["configs"]) == len(res["configs"]), "every configuration is reported, including the ones that showed nothing"
    by = {r["key"]: r for r in res["configs"]}
    p = res["periods"]["daily"]
    assert p["train"][1] < p["validate"][0] <= p["validate"][1] < p["test"][0] and p["test_sessions"] > 250
    assert res["confirmed"], "a 1.2% reversal after every 2% decline should be found and confirmed"
    for k in res["candidates"]:
        r = by[k]
        assert r["train"]["n"] >= 30 and r["p_adjusted"] < 0.10 and not r["rejected_because"] and "test" in r
    assert all("test" not in r for r in res["configs"] if r["key"] not in res["candidates"]), "only candidates may be scored on the final period"
    top = by[res["confirmed"][0]]
    assert top["preds"][0]["feature"] == "ret_1" and top["train"]["lift"] > 0.15 and top["study"]["symbol"] == "TST"
    assert [f["n"] for f in res["funnel"]] == sorted([f["n"] for f in res["funnel"]], reverse=True)
    assert res["selection_cv"]["daily"]["k"] >= 3 and "UNDERLYING" in " ".join(res["notes"])
    obs = HY.observations(res, top["key"], st)
    assert len(obs["rows"]) >= top["train"]["n"] and {"train", "validate", "test"} >= {r["period"] for r in obs["rows"]}
    hidden = HY.observations(res, next(r["key"] for r in res["configs"] if r["status"] == "ok" and "test" not in r), st)
    assert hidden["hidden_test_period"] and all(r["period"] != "test" for r in hidden["rows"])


def test_noise_produces_no_candidate_and_the_final_period_is_never_looked_at(tmp_path):
    from strata.research import hypothesis as HY
    st = rich_store(tmp_path, sessions=1500, effect=0.0, intraday=0, seed=11)
    for q in (QUESTIONS[0], QUESTIONS[6]):
        res = HY.run_plan(HY.build_plan(q.replace("SPY", "TST"), "TST", st, ["TST", "OTH"]), st)
        confirmed = [r for r in res["configs"] if r.get("status_label", "").startswith("confirmed")]
        assert not confirmed, f"noise was confirmed: {[r['label'] for r in confirmed]}"
        if not res["candidates"]:
            assert not res["test_period_used"] and all("test" not in r for r in res["configs"])
        assert all(r.get("grade", {}).get("level") != "strong" for r in res["configs"] if r["status"] == "ok")


def test_configurations_that_need_bars_which_are_not_stored_are_reported_not_dropped(tmp_path):
    from strata.research import hypothesis as HY
    st = rich_store(tmp_path, intraday=14)
    res = HY.run_plan(HY.build_plan(QUESTIONS[1], "TST", st), st)
    assert res["tested"] == 18 and res["evaluable"] == 0 and all(r["status"] == "no_data" and "250 sessions" in r["error"] for r in res["configs"])
    assert res["periods"]["intraday"] is None and any("could not be evaluated" in n for n in res["notes"])


# ----------------------------------------------------------------------------- ledger
def test_the_ledger_counts_distinct_configurations_and_corrections_grow_with_them(tmp_settings):
    from strata.research import hypothesis as HY
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    sid = session()
    LG.record(sid, "TST", [{"kind": "study", "key": "a", "start": "2019-01-02", "end": "2020-01-02"}, {"kind": "study", "key": "a"}, {"kind": "backtest", "key": "b", "start": "2019-01-02", "end": "2020-06-01"}])
    assert LG.count(sid) == 2 and LG.summary(sid)["evaluations"] == 3
    assert LG.config_key({"x": 1, "label": "one", "start": "2020-01-01"}) == LG.config_key({"x": 1, "label": "two", "start": "2021-01-01"}) != LG.config_key({"x": 2})
    plan = HY.build_plan(QUESTIONS[0].replace("SPY", "TST"), "TST", st, ["TST", "OTH"])
    alone = HY.run_plan(plan, st)
    LG.record(sid, "TST", [{"kind": "study", "key": f"earlier{i}", "start": "2019-01-02", "end": "2020-06-01"} for i in range(400)])
    crowded = HY.run_plan(plan, st, scope=sid)
    assert crowded["trials"]["session"] == 402 + len(plan["configs"]) and alone["trials"]["session"] == len(plan["configs"])
    a, c = {r["key"]: r for r in alone["configs"]}, {r["key"]: r for r in crowded["configs"]}
    assert all(c[k]["p_adjusted"] >= a[k]["p_adjusted"] - 1e-12 for k in a if a[k].get("p_adjusted") is not None)
    assert any(c[k]["p_adjusted"] > a[k]["p_adjusted"] for k in a if a[k].get("p_adjusted") not in (None, 0.0, 1.0))
    assert LG.count(sid) == 402 + len(plan["configs"]), "every configuration of the plan is in the ledger, not only the ones shown as candidates"


def test_tuning_after_seeing_the_final_period_contaminates_it_and_a_new_holdout_is_required(tmp_settings):
    from strata.research import hypothesis as HY
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    sid = session()
    plan = HY.build_plan(QUESTIONS[0].replace("SPY", "TST"), "TST", st, ["TST", "OTH"])
    first = HY.run_plan(plan, st, scope=sid)
    h = first["holdouts"]["daily"]
    assert first["confirmed"] and h["status"] == "used" and len(h["uses"]) == 1 and sorted(h["uses"][0]["keys"]) == sorted(first["candidates"])
    # running the very same configuration again is not a new hypothesis
    again = LG.record(sid, "TST", [{"kind": "study", "key": first["candidates"][0], "start": "2019-01-02", "end": None}])
    assert again["contaminated"] is None and LG.current(sid, "TST")["status"] == "used"
    # a run that ends before the final period cannot contaminate it either
    assert LG.record(sid, "TST", [{"kind": "study", "key": "early-only", "start": "2019-01-02", "end": "2019-12-31"}])["contaminated"] is None
    # a NEW configuration tested on data that includes the final period, after its results were shown
    hit = LG.record(sid, "TST", [{"kind": "study", "key": "tuned-threshold", "label": "ret_1 <= -2.25", "start": "2019-01-02", "end": None}])
    assert hit["contaminated"]["status"] == "contaminated" and "after its results had been shown" in hit["contaminated"]["reason"]
    tuned = HY.build_plan(QUESTIONS[6].replace("SPY", "TST"), "TST", st, ["TST", "OTH"])
    second = HY.run_plan(tuned, st, scope=sid)
    assert second["holdouts"]["daily"]["status"] == "contaminated" and second["holdouts"]["daily"].get("usable") is False
    assert not second["confirmed"] and not second["test_period_used"] and all("test" not in r for r in second["configs"])
    for k in second["candidates"]:
        r = next(x for x in second["configs"] if x["key"] == k)
        assert r["status_label"] == "candidate - no untouched data" and "new holdout is required" in r["test_note"] and r["grade"]["level"] != "strong"
    assert any("contaminated" in n for n in second["notes"])
    need = LG.reserve(sid, "TST", SessionFeaturesDates(st))
    assert need["usable"] is False and need["needed"] == LG.MIN_HOLDOUT_SESSIONS
    later = SessionFeaturesDates(st) + [(date(2030, 1, 1) + timedelta(days=i)).isoformat() for i in range(45)]
    fresh = LG.reserve(sid, "TST", later)
    assert fresh["status"] == "sealed" and fresh["usable"] and fresh["start"] > h["end"] and "replaces the contaminated period" in fresh["reason"]


def SessionFeaturesDates(st) -> list[str]:
    from strata.research.features import SessionFeatures
    return SessionFeatures(st, "TST").dates[1:]


def test_a_period_that_earlier_runs_already_covered_is_never_called_untouched(tmp_settings):
    from strata.research import ledger as LG
    sid = session()
    days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(500)]
    LG.record(sid, "TST", [{"kind": "study", "key": "full-period-look", "start": days[0], "end": days[-1]}])
    h = LG.reserve(sid, "TST", days)
    assert h["status"] == "exposed" and not h["usable"] and "already covered" in h["reason"] and "not untouched" in LG.holdout_note(h)
    other = session()
    clean = LG.reserve(other, "TST", days)
    assert clean["status"] == "sealed" and clean["usable"] and clean["start"] == days[400]
    used = LG.use(clean["id"], ["k1"], "run1")
    assert used["status"] == "used" and LG.use(clean["id"], ["k1"], "run1")["status"] == "used"  # showing the same candidate again reveals nothing
    assert LG.use(clean["id"], ["k2"], "run2")["status"] == "contaminated"  # a second, different look


# ----------------------------------------------------------------------------- the agent
def test_a_broad_question_runs_as_a_plan_and_a_candidate_can_be_taken_to_a_study_and_to_options(tmp_settings):
    from strata.api import db
    from strata.research import ledger as LG
    st = rich_store(tmp_settings, sessions=1500, effect=1.2, intraday=0)
    ag, sid = agent(st), session()
    m = ag.post(sid, "Investigate whether TST tends to reverse following large down days.")[-1]
    c = m["content"]
    assert m["kind"] == "hypothesis" and len(c["plan"]["configs"]) == 36 and c["plan"]["hypothesis"] and c["data"]["required"] and "all of them will be reported" in c["text"]
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "hypothesis" and res["summary"]["tested"] == 36 and res["summary"]["confirmed"] >= 1 and res["candidates"] and "separate test on observed premiums" in res["text"]
    assert ag.payload(sid)["trials"] == 36 and ag.payload(sid)["ledger"]["holdouts"][0]["status"] == "used"
    cand = res["candidates"][0]
    out = ag.select_hypothesis(sid, res["run_id"], cand["key"], "study")
    plan = out[-1]["content"]
    assert out[-1]["kind"] == "plan" and plan["kind"] == "study" and plan["runs"] and "not counted as a new hypothesis" in plan["text"]
    study = ag.finalize(sid)[0]["content"]
    assert study["view"] == "single" and LG.current(sid, "TST")["status"] == "used", "the same configuration, run session by session, must not contaminate the final period"
    full = json.loads((__import__("strata.api.results", fromlist=["run_dir"]).run_dir(res["run_id"]) / "hypothesis.json").read_text())
    row = next(r for r in full["configs"] if r["key"] == cand["key"])
    got = __import__("strata.api.results", fromlist=["load_run_result"]).load_run_result(db.get_run(plan["runs"][0]["run_id"]))
    assert got["stats"]["qualifying_sessions"] == row["train"]["sessions"] + row["validate"]["sessions"] + row["test"]["sessions"], "the drill-down study covers exactly the plan's sessions"
    # reasoning about the result, from stored observations
    a = ag.post(sid, "How much of the result is explained by volatility?")[-1]["content"]
    assert a["view"] == "analysis" and a["tool"] == "volatility_explained" and a["analysis"]["tables"] and a["explanation_source"] == "deterministic"
    a = ag.post(sid, "Why did it perform worse after 2022?")[-1]["content"]
    assert a["tool"] == "subperiod" and "Before 2022-01-01" in a["text"] and "From 2022-01-01" in a["text"]
    # stage two: option positions on the same sessions; nothing is stored, so nothing is priced and nothing is invented
    opt = ag.select_hypothesis(sid, res["run_id"], cand["key"], "options")
    inv = opt[-1]["content"]
    assert opt[-1]["kind"] == "investigation" and inv["plan"]["stage"] == "options" and len(inv["runs"]) >= 6 and "observed premiums only" in inv["text"]
    done = ag.finalize(sid)[0]["content"]
    assert done["view"] == "failed" and "None of the planned configurations could be run" in done["text"]
    assert LG.current(sid, "TST")["status"] == "used", "pricing option positions on an already-selected condition does not re-tune the condition"
    assert ag.payload(sid)["trials"] > 36, "the option positions are still counted as configurations tried"
    # a different threshold after the final period was shown: the ledger notices
    ag.post(sid, "How often does TST close above its open after falling 2.2% or more the previous day?")
    assert LG.current(sid, "TST")["status"] == "contaminated"


def test_stateful_strategy_requests_are_not_mistaken_for_hypotheses(tmp_settings):
    st = rich_store(tmp_settings, sessions=900, intraday=0)
    ag, sid = agent(st), session()
    for text in ("Wait until TST drops 0.5% from the open, then buy calls when it recovers half of that decline.", "Buy TST puts after three consecutive five-minute red candles, unless yesterday was also red.",
                 "How often does TST rise 1% after a 3% down day?"):
        r, _, _ = ag.interpret(sid, text, None, None, None)
        assert r["action"] == "new", (text, r["action"])
    # V9: a question whether the first candle predicts the rest of the session is answered by the agreement reading (the same
    # engine as every other phrasing of it); the hypothesis lab keeps requests worded as hypotheses to investigate
    r, _, _ = ag.interpret(sid, "Does the first 15-minute candle predict the rest of the session for TST?", None, None, None)
    assert r["action"] == "investigate" and r["plan"]["kind"] in ("first_candle", "agreement") and r["plan"]["symbol"] == "TST"
    r, _, _ = ag.interpret(sid, "Investigate the hypothesis that a large gap down on TST tends to reverse by the close", None, None, None)
    assert r["action"] in ("hypothesis", "investigate", "discover", "graph"), r["action"]


def test_structure_variants_cover_both_sides_and_the_stated_exits():
    from strata.research.hypothesis import structure_variants
    v, notes = structure_variants("SPY", [{"feature": "ret_1", "op": "<=", "value": -2}], {"entry": "close", "horizon": "sessions:1", "direction": "up", "target": 1.0, "stop": 0.5, "measure": "touch"}, True, False)
    ds = [x["draft"] for x in v]
    assert {d.structure for d in ds} >= {"long_call", "long_put", "bull_call_spread", "bull_put_spread", "long_straddle", "iron_condor"} and len(v) <= 12
    assert all(d.entry_time == "close" and d.hold_sessions == 1 and d.conditions[0].expr == "return_pct(1) <= -2" for d in ds)
    assert {d.dte for d in ds} == {7, 14, 30} and any(d.premium_target_pct == 50 for d in ds) and any(d.underlying_target_pct == 1.0 and d.underlying_stop_pct == 0.5 for d in ds)
    assert any(d.strike.method == "delta" for d in ds) and "end-of-day bid/ask" in notes[0]
    v2, n2 = structure_variants("SPY", [{"feature": "vix_prev", "op": ">=", "value": 25}], {"entry": "open", "horizon": "session", "direction": "either", "target": 1.0, "stop": None, "measure": "touch"}, False, False)
    assert {x["draft"].structure for x in v2} == {"long_straddle", "iron_condor", "long_strangle"} and {x["draft"].dte for x in v2} <= {0, 1, 7} and any("No intraday option prices" in x for x in n2)


# ----------------------------------------------------------------------------- reasoning and the reviewer
def test_reasoning_tools_answer_from_observations_and_say_when_they_cannot(tmp_path):
    from strata.research import reasoning as RQ
    from strata.research.features import OutcomeSpec, Pred, to_study
    from strata.research.study import run_study
    st = rich_store(tmp_path, sessions=1500, effect=1.2, intraday=0)
    oc = OutcomeSpec("open", "session", "up", 0.0, None, "close")
    r2, r3 = run_study(to_study("TST", [Pred("ret_1", "<=", -2)], oc), st), run_study(to_study("TST", [Pred("ret_1", "<=", -3)], oc), st)
    td = RQ.threshold_difference(r2, r3, "2% down", "3% down")
    assert td["answerable"] and "also a session of" in td["answer"] and "NOT a distinguishable difference" in td["answer"], td["answer"]
    sp = RQ.subperiod(r2, None, "2022")
    assert sp["tables"][0]["rows"][0]["sessions"] + sp["tables"][0]["rows"][1]["sessions"] == len([s for s in r2["sessions"] if s["outcome"] != "ambiguous"])
    assert not RQ.subperiod(r2, None, "1990")["answerable"]
    assert not RQ.cost_survival(r2, None)["answerable"] and "no trades" in RQ.cost_survival(r2, None)["answer"]
    assert RQ.suspicious(r2, None)["flags"] and "nearly implied by the condition" in RQ.suspicious(r2, None)["answer"]  # the built-in effect makes it ~100%
    ai = RQ.assumption_impact(r2, None, None, st)
    assert ai["answerable"] and ai["variants"] >= 2 and "Sign reversals" in [f["label"] for f in ai["facts"]]
    trades = [{"entry_date": f"2024-0{1 + i // 20}-{1 + i % 20:02d}", "pnl": p, "entry_vix": 15 + i % 20} for i, p in enumerate([900, 700, 500] + [-40] * 37)]
    cz = RQ.concentration({"metrics": {}}, trades)
    assert "depends on three trades" in cz["answer"] and cz["facts"][0]["value"] == "+620"
    pr = RQ.paired_runs(trades[:30], [{**t, "pnl": t["pnl"] - 10} for t in trades[10:]], "A", "B")
    assert "On the 20 sessions both traded" in pr["answer"] and "excluded" in pr["answer"]
    for q, tool in [("Why did the strategy perform worse after 2020?", "subperiod"), ("Is the 3% threshold genuinely different from the 2% threshold?", "threshold_difference"),
                    ("Would these results survive higher transaction costs?", "cost_survival"), ("Are most profits concentrated in a handful of sessions?", "concentration"),
                    ("How much of the result is explained by volatility?", "volatility_explained"), ("Which assumptions have the biggest effect on this backtest?", "assumption_impact")]:
        assert RQ.detect(q)["tool"] == tool, q
    assert RQ.detect("Buy calls at 10:00 after a 3% down day") is None


def test_the_reviewer_examines_every_topic_and_grounds_each_finding(tmp_path):
    from strata.research import challenge as CH
    from strata.research.features import OutcomeSpec, Pred, to_study
    from strata.research.study import run_study
    st = rich_store(tmp_path, sessions=1500, effect=0.0, intraday=0)
    res = run_study(to_study("TST", [Pred("ret_1", "<=", -1.5)], OutcomeSpec("open", "session", "up", 1.0, 1.0, "touch")), st)
    out = CH.challenge_study({"id": "r1"}, res, st, trials=25, holdout={"status": "contaminated", "start": "2023-01-03", "end": "2024-12-18", "reason": "3 new configurations were tested"})
    assert [f["topic"] for f in sorted(out["findings"], key=lambda f: [t for t, _ in CH.TOPICS].index(f["topic"]))] == [t for t, _ in CH.TOPICS]
    f = {x["topic"]: x for x in out["findings"]}
    assert all(x["evidence"] and x["severity"] in CH.ORDER for x in out["findings"])
    assert f["lookahead"]["severity"] == "ok" and f["lookahead"]["test"] == "future-scramble test" and "sampled sessions" in f["lookahead"]["evidence"]
    assert "25 configurations" in f["selection"]["evidence"] and f["selection"]["severity"] in ("medium", "high")
    assert f["intrabar"]["severity"] in ("medium", "high") and "daily bar" in f["intrabar"]["evidence"]  # target and stop on daily bars: the order inside the day is unknown
    assert f["leverage"]["severity"] == "na" and f["fills"]["severity"] == "na" and f["contracts"]["severity"] == "na"
    assert f["parameters"]["table"]["rows"] and f["costs"]["severity"] == "medium" and "no trades" in f["costs"]["evidence"]
    assert any("contaminated" in n for n in out["notes"]) and sum(out["counts"].values()) == 16
    assert "should not be relied on" in out["verdict"] or "limit what can be claimed" in out["verdict"]


def test_large_plans_stay_fast(tmp_path):
    import time

    from strata.research import hypothesis as HY
    st = rich_store(tmp_path, sessions=4200, intraday=0)
    plan = HY.build_plan(QUESTIONS[6].replace("SPY", "TST"), "TST", st, ["TST", "OTH"])
    t0 = time.time()
    res = HY.run_plan(plan, st)
    assert res["evaluable"] >= 35 and time.time() - t0 < 30, "a scan of some forty configurations over 4,200 sessions should take seconds"


_ = (datetime, timezone)
