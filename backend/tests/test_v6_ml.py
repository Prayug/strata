"""V6: the ML workspace cannot be used to pick a winner from the test set.

Before version 6 every candidate model was scored on the reserved test period, so the best-looking one could be chosen
from it. Now: candidates are compared on purged validation only, the selected configuration is frozen, the test period is
scored for that one model when asked, and every such scoring is recorded for good. These tests use a relationship that
is really there, pure noise, and a feature that knows the answer.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v4_research import rich_store
from tests.test_v5_forward_ml import UP, predictable_store

ml_ok = pytest.importorskip("sklearn", reason="the ML workspace is optional")
SPEC = {"symbol": "PRD", "outcome": UP, "features": ["vix_prev", "gap", "ret_1"], "models": ["logistic", "tree"]}


def test_the_test_set_is_scored_only_for_the_frozen_model_and_every_scoring_is_recorded_for_good(tmp_settings):
    from strata.api import db
    from strata.research import ml as ML
    st = predictable_store(tmp_settings)
    v = ML.run(SPEC, st)
    assert v["final_test"] is False and v["test_exposure"] == {"scored": False, "test": v["split"]["test"], "previous_scorings": 0, "other_configurations": 0, "this_configuration_scored_before": False, "history": []}
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM ml_test_exposures").fetchone()[0] == 0, "validation leaves no trace on the test set because it never touched it"
    a = ML.run({**SPEC, "final_test": True}, st, run_id="run-a")
    assert a["confirmation"] == "confirmed_once" and a["test_exposure"]["scored"] and a["test_exposure"]["configurations_scored_on_this_period"] == 1
    assert sum(m["test"] is not None for m in a["models"]) == 1
    # the same frozen configuration again: the same single look, not a second one
    a2 = ML.run({**SPEC, "final_test": True}, st, run_id="run-a2")
    assert a2["confirmation"] == "confirmed_once" and a2["test_exposure"]["this_configuration_scored_before"] is True
    with db.connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM ml_test_exposures ORDER BY id")]
    assert len(rows) == 1 and rows[0]["run_id"] == "run-a" and rows[0]["config_key"] == a["frozen"]["config_key"] and rows[0]["test_start"] == a["split"]["test"][0] and rows[0]["brier_skill"] == pytest.approx(next(m for m in a["models"] if m["test"])["test"]["brier_skill"])
    # a different feature set scored on the same period is a second look, and the result may no longer be called confirmed
    b = ML.run({**SPEC, "features": ["vix_prev", "gap", "ret_1", "ret_5"], "final_test": True}, st, run_id="run-b")
    assert b["frozen"]["config_key"] != a["frozen"]["config_key"] and b["test_exposure"]["other_configurations"] == 1 and b["test_exposure"]["configurations_scored_on_this_period"] == 2
    assert b["confirmation"] == "tested_repeatedly" and "2 different configurations" in b["summary"][0] and "cannot be called independently confirmed" in b["summary"][0]
    # and the first configuration is tainted by it too: the period has been looked at twice whichever is reported
    a3 = ML.run({**SPEC, "final_test": True}, st)
    assert a3["confirmation"] == "tested_repeatedly" and len(a3["test_exposure"]["history"]) == 2
    # validation alone still shows what has already happened to that period
    v2 = ML.run(SPEC, st)
    assert v2["test_exposure"]["scored"] is False and v2["test_exposure"]["previous_scorings"] == 2 and v2["confirmation"] == "not_tested"
    with db.connect() as con:
        for sql in ("UPDATE ml_test_exposures SET brier_skill=0.9", "DELETE FROM ml_test_exposures"):
            with pytest.raises(sqlite3.DatabaseError, match="permanent"):
                con.execute(sql)
    # a later run whose test period has merely shifted by a few sessions is the same period for this purpose
    e = ML.exposures("PRD", rows[0]["outcome_key"], str(date.fromisoformat(rows[0]["test_start"]) + timedelta(days=9)), str(date.fromisoformat(rows[0]["test_end"]) + timedelta(days=9)))
    assert len(e) == 2 and ML.exposures("PRD", rows[0]["outcome_key"], "2030-01-02", "2030-12-31") == [] and ML.exposures("OTHER", rows[0]["outcome_key"], rows[0]["test_start"], rows[0]["test_end"]) == []


def test_on_pure_noise_no_candidate_is_confirmed_and_the_choice_never_depends_on_the_test_set(tmp_settings):
    from strata.research import ml as ML
    st = rich_store(tmp_settings, sessions=1500, effect=0.0, intraday=0)
    spec = {"symbol": "TST", "outcome": UP, "features": ["ret_1", "ret_5", "gap", "rv21", "weekday", "range_1"], "models": ["logistic", "ridge", "lasso", "tree", "forest", "gbt"]}
    v = ML.run(spec, st)
    cvs = {m["name"]: m["cv"]["log_loss"] for m in v["models"]}
    assert v["selected"] == min(cvs, key=cvs.get) and all(m["test"] is None for m in v["models"]), "six candidates, zero looks at the test set"
    f = ML.run({**spec, "final_test": True}, st)
    scored = [m for m in f["models"] if m["test"] is not None]
    assert [m["name"] for m in scored] == [v["selected"]] and f["selected"] == v["selected"]
    assert f["verdict"] in ("no_skill", "weak") and f["confirmation"] in ("not_confirmed", "confirmed_once") and scored[0]["test"]["brier_skill"] < 0.02
    assert "does not beat the base-rate" in f["summary"][0] or "did not hold up" in f["summary"][0] or "unproven" in f["summary"][0]
    # six looks would have been taken before version 6; exactly one was
    from strata.api import db
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM ml_test_exposures").fetchone()[0] == 1


def test_a_feature_that_knows_the_answer_is_called_a_leak_not_a_discovery(tmp_settings):
    """LEAK's return on day t-1 is +1% when PRD goes on to rise on day t and -1% when it falls: a series stamped a day too early."""
    from strata.data.store import SymbolMeta
    from strata.research import ml as ML
    st = predictable_store(tmp_settings)
    d = st.read_daily("PRD")
    days, up = d["date"].to_list(), (d["close"] > d["open"]).to_list()
    px, rows = 50.0, []
    for i in range(len(days)):
        nxt = up[i + 1] if i + 1 < len(days) else True
        o = px
        px = px * (1.01 if nxt else 0.99)
        rows.append((days[i], o, max(o, px), min(o, px), px))
    st.upsert_symbol(SymbolMeta(symbol="LEAK", kind="etf"))
    st.write_daily("LEAK", pl.DataFrame({"date": [r[0] for r in rows], "open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows], "close": [r[4] for r in rows], "volume": [1e6] * len(rows)}), provider="yahoo")
    st.rebuild_manifest("LEAK")
    r = ML.run({"symbol": "PRD", "outcome": UP, "features": ["gap", "LEAK_ret_1"], "models": ["logistic", "tree"], "final_test": True}, st)
    best = next(m for m in r["models"] if m["name"] == r["selected"])
    assert best["cv"]["auc"] > 0.95, "the model looks superb, which is the problem"
    assert r["leakage"]["passed"] is False and "alone separates the outcome" in r["leakage"]["checks"][0]["detail"]
    assert r["verdict"] == "leak_suspected" and r["confirmation"] == "not_applicable" and "leakage check failed" in r["summary"][0] and "not evidence of predictive skill" in r["summary"][0]
    clean = ML.run({"symbol": "PRD", "outcome": UP, "features": ["gap", "vix_prev"], "models": ["logistic"]}, st)
    assert clean["leakage"]["passed"] is True and clean["verdict"] == "some_skill"


def test_scaling_calibration_and_the_signal_threshold_are_fitted_on_training_rows_only():
    from strata.research import ml as ML
    rng = np.random.default_rng(3)
    Xtr = rng.normal(0, 1, (800, 3))
    ytr = (Xtr[:, 0] + rng.normal(0, 1, 800) > 0).astype(int)
    Xte = rng.normal(50, 9, (200, 3))  # the rows to predict are on a wildly different scale
    p, f = ML._fit_predict("logistic", Xtr, ytr, Xte, False)
    sc = f.model.steps[0][1]
    assert np.allclose(sc.mean_, Xtr.mean(0)) and sc.n_samples_seen_ == 800, "the scaler has never seen the rows it predicts"
    p, f = ML._fit_predict("logistic", Xtr, ytr, Xte, True, embargo=5)
    assert f.iso is not None and f.model.steps[0][1].n_samples_seen_ == 595, "the model stops 5 sessions before the 200 calibration rows begin (600 - 5)"
    assert np.allclose(f.model.steps[0][1].mean_, Xtr[:595].mean(0))
    raw = f.model.predict_proba(Xtr)[:, 1]
    assert np.allclose(f.proba(Xtr), np.clip(f.iso.predict(raw), 0, 1)) and not np.allclose(f.proba(Xtr), raw), "every prediction goes through the calibration map"
    # the walk-forward threshold is a quantile of those same calibrated predictions, on rows before the block it applies to
    X = rng.normal(0, 1, (900, 2))
    y = (X[:, 0] + rng.normal(0, 1.2, 900) > 0).astype(int)
    idx = np.arange(900)
    dates = np.array([str(d) for d in cal.trading_days(date(2018, 1, 2), date(2022, 12, 30))[:900]])
    wf = ML._walk_forward("tree", X, y, idx, idx, dates, 1, 0.8, True)
    assert 0.1 < wf["signals"] / wf["predicted"] < 0.5, "a minority of sessions clear an 80th-percentile threshold when both sides are on the same scale (a shallow tree has few distinct levels, so ties add some)"
    part = ML._walk_forward("tree", X, y, idx, idx, dates, 1, 0.8, True, upto=600)
    assert part["covers_final_test"] is False and part["last"] <= dates[599] and part["predicted"] == 300
    assert [s["leak"] for s in ML.PREPROCESSING] == [False] * 7 and {s["step"] for s in ML.PREPROCESSING} >= {"Scaling", "Missing values", "Calibration", "Signal threshold", "Model selection"}


def test_what_was_tested_on_a_period_survives_the_deletion_of_the_notebook_that_tested_it(tmp_settings):
    """The registry is what makes "this final period has not been looked at" true. A notebook can be tidied away; its trials cannot."""
    from strata.api import db
    from strata.research import ledger as LG
    from strata.research import registry as RG
    sid = db.create_session("to be deleted")["id"]
    sig = {"symbol": "TST", "clauses": ["ret_1<=-1"], "outcome": {"direction": "up"}}
    LG.record(sid, "TST", [{"kind": "graph", "key": "k1", "label": "after a 1% fall", "start": "2019-01-02", "end": "2022-07-28", "metric": "lift", "value": 0.04, "n": 120, "p_value": 0.03, "signature": sig}], family="q")
    assert RG.exposure("TST", "2022-01-03", "2022-07-28", exclude_scope="another")["configurations"] == 1
    db.delete_session(sid)
    assert db.get_session(sid) is None
    after = RG.exposure("TST", "2022-01-03", "2022-07-28", exclude_scope="another", signature=sig)
    assert after["configurations"] == 1 and after["notebooks"] == 1, "the deleted notebook's look at that period still counts"
    assert not hasattr(LG, "delete_scope")
    with db.connect() as con, pytest.raises(sqlite3.DatabaseError, match="permanent"):
        con.execute("DELETE FROM trials")


def test_the_final_test_is_pinned_to_what_validation_froze(tmp_settings):
    """Step two scores the configuration step one froze. If the model, its features or the rows of the split are not the
    same any more, it stops before a test score exists, and nothing is recorded as a look."""
    from fastapi.testclient import TestClient

    from strata.api import db
    from strata.api.app import app
    from strata.research import ml as ML
    st = predictable_store(tmp_settings)
    v = ML.run(SPEC, st)
    assert v["spec"] == SPEC and v["validation_run"] is None, "the result carries the request that produced it"
    pin = {"expect_frozen": {k: v["frozen"][k] for k in ("config_key", "train", "test")}, "end": v["frozen"]["test"][1], "final_test": True}
    f = ML.run({**v["spec"], **pin, "validation_run": "run-v"}, st, run_id="run-f")
    assert f["final_test"] and f["frozen"]["config_key"] == v["frozen"]["config_key"] and f["split"] == v["split"] and f["validation_run"] == "run-v"
    assert f["test_exposure"]["scored"] and sum(m["test"] is not None for m in f["models"]) == 1
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM ml_test_exposures").fetchone()[0] == 1
    # other features, or a different period, under the same pin: refused, and the test set is not looked at
    for change, what in (({"features": ["vix_prev", "gap", "ret_1", "ret_5"]}, "the selected model or its features"), ({"end": "2019-12-31"}, "the test period")):
        with pytest.raises(ValueError, match="the final test was not run") as e:
            ML.run({**v["spec"], **pin, **change}, st, run_id="run-x")
        assert what in str(e.value)
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM ml_test_exposures").fetchone()[0] == 1, "a refused step is not a look at the test set"
    # through the application: a finished validation run offers the step once; a run that holds the score does not
    rid = db.create_run("ml", "validation", {"symbol": "PRD"})
    ML.save_result(rid, v)
    db.update_run(rid, status="done")
    done = db.create_run("ml", "final", {"symbol": "PRD"})
    ML.save_result(done, f)
    old = db.create_run("ml", "from before", {"symbol": "PRD"})
    ML.save_result(old, {k: x for k, x in v.items() if k != "spec"})
    leaky = db.create_run("ml", "leaky", {"symbol": "PRD"})
    ML.save_result(leaky, {**v, "verdict": "leak_suspected"})
    for x in (done, old, leaky):
        db.update_run(x, status="done")
    with TestClient(app) as c:  # the job runner starts with the application
        assert c.post(f"/api/runs/{done}/ml/final-test").status_code == 409
        r = c.post(f"/api/runs/{old}/ml/final-test")
        assert r.status_code == 409 and "run it again" in r.json()["detail"]
        assert c.post(f"/api/runs/{leaky}/ml/final-test").status_code == 409
        started = c.post(f"/api/runs/{rid}/ml/final-test")
        assert started.status_code == 200 and started.json()["run_id"] != rid
        new = started.json()["run_id"]
        for _ in range(600):
            if db.get_run(new)["status"] in ("done", "error", "cancelled"):
                break
            import time
            time.sleep(0.1)
        assert db.get_run(new)["status"] == "done", db.get_run(new).get("error")
        got = c.get(f"/api/runs/{new}/ml").json()
        assert got["final_test"] is True and got["validation_run"] == rid and got["frozen"]["config_key"] == v["frozen"]["config_key"]
