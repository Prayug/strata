"""The same request on the same data gives the same answer, byte for byte.

A research result is only worth storing if it can be produced again: these tests run every
kind of research twice on an unchanged store and compare everything that is a result
(observations, statistics, trades, coverage) while ignoring only wall-clock fields."""
from __future__ import annotations

import json

from tests.test_v2_research import DAYS, tmp_settings  # noqa: F401  (fixture)
from tests.test_v3_agent import _grid, long_store, trade_store

CLOCK = {"elapsed_seconds", "runtime_seconds", "started", "finished", "updated", "created_at", "generated", "run_id", "id", "pipeline", "seconds", "duration_seconds", "timing"}


def canon(x):
    """A result with wall-clock fields removed, as canonical JSON."""
    def strip(v):
        if isinstance(v, dict):
            return {k: strip(w) for k, w in sorted(v.items()) if k not in CLOCK}
        return [strip(w) for w in v] if isinstance(v, list) else v
    return json.dumps(strip(json.loads(json.dumps(x, default=str))), sort_keys=True)


def _pipeline(spec, st, rid):
    from strata.api.results import load_run_result, load_trades
    from strata.research.pipeline import Pipeline, fingerprint
    out = Pipeline(rid, spec, store=st).run()
    kind = "study" if spec["kind"] == "study" else "backtest"
    return out, load_run_result({"id": rid, "kind": kind}), load_trades(rid), fingerprint(spec, st)


STUDY = {"kind": "study", "mode": "study", "symbol": "TST", "label": "s", "backtest": None, "provider": None, "approvals": {}, "interpretation": {},
         "requested": {"start": None, "end": None, "stated": False},
         "study": {"symbol": "TST", "entry_time": "open", "target": {"kind": "pct", "value": 1.0}, "horizon": {"kind": "session"},
                   "condition": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "previous_session.return_pct <= -3"}}]}}}

CALLS = {"name": "0DTE call after a down day", "symbol": "TST",
         "legs": [{"id": "lc", "side": "buy", "right": "C", "quantity": 1, "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0}}],
         "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "open"}},
                                                                             {"kind": "cond", "type": "expr", "params": {"expr": "previous_session.return_pct <= -3"}}]}},
         "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}}}
OPTIONS = {"kind": "options", "mode": "trades", "symbol": "TST", "label": "o", "study": None, "provider": None, "approvals": {"reduced_window": True}, "interpretation": {},
           "requested": {"start": None, "end": None, "stated": False}, "backtest": {"strategies": [CALLS], "execution": {"signal_interval_minutes": 1}}}


def test_a_market_study_is_reproducible(tmp_settings):
    st = trade_store(tmp_settings)
    (o1, r1, _, f1), (o2, r2, _, f2) = _pipeline(STUDY, st, "a1"), _pipeline(STUDY, st, "a2")
    assert f1 == f2  # the same request on the same data has the same fingerprint (this is the cache key)
    assert r1["stats"]["qualifying_sessions"] == 2 and r1["stats"]["target_first"] == 1
    assert canon(r1) == canon(r2)
    assert canon({k: v for k, v in o1.items()}) == canon({k: v for k, v in o2.items()})


def test_a_trade_priced_option_backtest_is_reproducible(tmp_settings):
    st = trade_store(tmp_settings)
    (o1, r1, t1, f1), (o2, r2, t2, f2) = _pipeline(OPTIONS, st, "b1"), _pipeline(OPTIONS, st, "b2")
    assert o1["outcome"] == "completed" and len(t1) == 1 and t1[0]["legs"][0]["price_source"] == "trade_print"
    assert f1 == f2 and canon(t1) == canon(t2)
    assert canon(r1["metrics"]) == canon(r2["metrics"]) and canon(r1["coverage"]) == canon(r2["coverage"]) and canon(r1["series"]) == canon(r2["series"])


def test_the_fingerprint_changes_when_the_request_or_the_data_changes(tmp_settings):
    from strata.research.pipeline import fingerprint
    from tests.test_v3_agent import add_trade_bars
    st = trade_store(tmp_settings)
    base = fingerprint(OPTIONS, st)
    other = json.loads(json.dumps(OPTIONS))
    other["backtest"]["strategies"][0]["exits"]["time_exit"]["time"] = "15:30"
    assert fingerprint(other, st) != base  # a different rule is a different run
    add_trade_bars(st, DAYS[5], 94.0, [("09:30", 0.80), ("15:54", 0.50)])
    assert fingerprint(OPTIONS, st) != base  # new observations invalidate the cached result


def test_an_exploration_is_reproducible(tmp_path):
    from strata.research.discovery import run_discovery
    st = long_store(tmp_path, effect=True)
    a, b = run_discovery(_grid(), st), run_discovery(_grid(), st)
    assert a["tested"] == b["tested"] and a["candidates"] == b["candidates"] and a["confirmed"] == b["confirmed"]
    assert canon(a["configs"]) == canon(b["configs"]) and canon(a["splits"]) == canon(b["splits"])
