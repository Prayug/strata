"""V7: the performance card, every fill against the stored bar behind it, and the same position without the signal."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from strata.data.providers import sources as S
from tests.helpers import et_epoch
from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401
from tests.test_v7_engine import cfg, wave_store
from tests.test_v7_experiment import DAYS, grid_store
from tests.test_v7_replay import stored_run

ORDER = ["trades", "coverage", "net", "average", "expectancy", "drawdown", "streaks", "distribution", "calendar", "time_of_day", "execution", "concentration", "baseline"]


def card_of(rid, trials=1, **kw):
    from strata.api import db
    from strata.api.results import load_result, load_trades
    from strata.research import performance as PF
    return PF.card(db.get_run(rid), load_result(rid), load_trades(rid), trials, **kw)


def test_the_card_puts_the_sample_and_the_coverage_first_and_every_figure_is_the_stored_trades(tmp_settings):
    from strata.api.results import load_trades
    st = grid_store(tmp_settings)
    rid = stored_run(st, cfg(exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "max_hold": {"enabled": True, "minutes": 15}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}))
    trades = load_trades(rid)
    p = np.array([t["pnl"] for t in trades])
    c = card_of(rid)
    sec = {s["id"]: s for s in c["sections"]}
    assert [s["id"] for s in c["sections"]] == ORDER and c["observation"] == "trade prints (no bid/ask)" and c["window"] == [DAYS[0].isoformat(), DAYS[-1].isoformat()]
    assert sec["trades"]["figures"] == {"trades": 10, "winners": 5, "losers": 5, "flat": 0, "sessions_with_a_trade": 10} and "Fewer than 30 trades" in sec["trades"]["text"]
    cov = sec["coverage"]["figures"]
    assert (cov["signals"], cov["priced"], cov["priced_pct"], cov["sessions_in_window"], cov["not_priced_by_reason"]) == (10, 10, 100.0, 10, {})
    net = sec["net"]["figures"]
    assert net["net"] == round(float(p.sum()), 2) and net["gross"] == round(sum(t["gross_pnl"] for t in trades), 2) and net["commissions"] == round(sum(t["commissions"] for t in trades), 2)
    # the account's own books and the trades agree, and the card would say so if they did not
    assert net["account_net"] == net["net"] and net["account_minus_trades"] == 0.0 and sec["net"]["warning"] is False and "own books" not in sec["net"]["text"]
    from strata.api import db
    from strata.api.results import load_result
    from strata.research import performance as PF
    res = load_result(rid)
    res["metrics"] = {**res["metrics"], "net_pnl": res["metrics"]["net_pnl"] + 12_823.35}  # something held that never became a trade
    bad = next(x for x in PF.card(db.get_run(rid), res, trades)["sections"] if x["id"] == "net")
    assert bad["warning"] is True and bad["figures"]["account_minus_trades"] == 12_823.35 and "+12,823.35 that these trades do not account for" in bad["text"] and "do not rely on either figure" in bad["text"]
    avg = sec["average"]["figures"]
    assert avg["mean"] == round(float(p.mean()), 2) and avg["win_rate"] == 50.0 and avg["average_win"] == round(float(p[p > 0].mean()), 2) and avg["average_loss"] == round(float(p[p < 0].mean()), 2)
    ex = sec["expectancy"]["figures"]
    assert ex["ci95"][0] < ex["mean"] < ex["ci95"][1] and "The interval includes zero" in sec["expectancy"]["text"]
    # even sessions win 10 before costs and odd ones lose 10: the path saws, one loss at a time
    assert sec["streaks"]["figures"] == {"max_consecutive_losses": 1, "max_consecutive_wins": 1}
    eq = np.cumsum(p)
    assert sec["drawdown"]["figures"]["max_drawdown"] == round(float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()), 2)
    assert sum(sec["distribution"]["figures"]["histogram"]["counts"]) == 10 and sec["distribution"]["figures"]["best"] == round(float(p.max()), 2)
    assert [(m["key"], m["trades"]) for m in sec["calendar"]["figures"]["months"]] == [("2024-03", 10)] and sec["time_of_day"]["figures"]["half_hours"][0]["key"] == "10:00" and "All entries fall in the half hour from 10:00" in sec["time_of_day"]["text"]
    turn = sum((l["entry_price"] + l["exit_price"]) * 100 for t in trades for l in t["legs"])
    rows = sec["execution"]["figures"]["rows"]
    assert [r["worse_fills_pct"] for r in rows] == [1.0, 2.0, 5.0, 10.0] and rows[1]["net"] == round(float(p.sum()) - turn * 0.02, 2) and "bid/ask spread is not in any of these figures" in sec["execution"]["text"]
    conc = sec["concentration"]["figures"]
    assert conc["best_trade"] == round(float(p.max()), 2) and conc["net_without_best"] == round(float(p.sum() - p.max()), 2) and conc["net_without_best_3"] == round(float(p.sum() - np.sort(p)[-3:].sum()), 2)
    assert sec["baseline"]["figures"] is None and "Not run yet" in sec["baseline"]["text"]
    # a historical result is not evidence of an edge, and the card says which one it is showing
    assert c["evidence"]["level"] == "insufficient" and c["evidence"]["text"].startswith("Too few trades") and "what these trades did on this data" in c["evidence"]["reading"]
    many = card_of(rid, trials=40)
    e2 = next(s for s in many["sections"] if s["id"] == "expectancy")
    assert e2["figures"]["configurations_tried"] == 40 and e2["figures"]["p_adjusted"] == min(1.0, e2["figures"]["p_value"] * 40) and "40 configurations tried" in e2["text"]


def test_signals_the_stored_data_could_not_price_are_counted_as_that_and_no_trades_is_said(tmp_settings):
    st = grid_store(tmp_settings, skip_days={DAYS[3], DAYS[4]})
    c = card_of(stored_run(st, cfg()))
    cov = next(s for s in c["sections"] if s["id"] == "coverage")
    assert (cov["figures"]["signals"], cov["figures"]["priced"], cov["figures"]["priced_pct"]) == (10, 8, 80.0) and sum(cov["figures"]["not_priced_by_reason"].values()) == 2
    assert "The others are not losses" in cov["text"] and next(s for s in c["sections"] if s["id"] == "trades")["figures"]["trades"] == 8
    far = cfg(legs=[{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "fixed", "value": 560.0}}])
    none = card_of(stored_run(st, far))
    assert [s["id"] for s in none["sections"]] == ["trades", "coverage"] and none["sections"][0]["text"].startswith("No trade was taken.") and none["evidence"]["level"] == "insufficient"


def test_every_fill_can_be_checked_against_the_stored_bar_it_was_priced_from(tmp_settings):
    from fastapi.testclient import TestClient
    import strata.api.app as appmod
    from strata.research import performance as PF
    st = grid_store(tmp_settings)
    c = cfg()
    c["end"] = DAYS[1].isoformat()
    rid = stored_run(st, c)
    f = PF.fills(rid, st)
    assert len(f["fills"]) == 4 and f["checked_against_stored_bars"] == 4 and f["matching"] == 4
    e, x = f["fills"][0], f["fills"][1]
    occ = S.occ_symbol("SPY", DAYS[0], "C", 500.0)
    assert (e["action"], e["side"], e["contract"], e["decided_et"], e["stored_bar_start_et"], e["stored_bar_close"], e["print_used"], e["matches_stored_bar"]) == ("entry", "buy", occ, f"{DAYS[0]} 10:00:00", "09:59:00", 2.75, 2.75, True)
    assert (x["action"], x["side"], x["stored_bar_start_et"], x["print_age_seconds"], x["reason"]) == ("exit", "sell", "15:28:00", 60, "time_exit")
    # the stored observation is revised after the run: the fill no longer matches what is on disk, and that is shown, not hidden
    row = {"ts": datetime.fromtimestamp(et_epoch(DAYS[0], "09:59"), tz=timezone.utc), "occ": occ, "expiration": DAYS[0], "strike": 500.0, "right": "C", "open": 2.9, "high": 2.9, "low": 2.9, "close": 2.9, "volume": 7}
    st.write_option_bars("SPY", "testfeed", pl.DataFrame([row]).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": DAYS[0], "occ": occ, "expiration": DAYS[0], "strike": 500.0, "right": "C", "bars": 5, "status": "ok", "pages": 1}])
    g = PF.fills(rid, st)
    assert g["matching"] == 3 and g["fills"][0]["matches_stored_bar"] is False and g["fills"][0]["stored_bar_close"] == 2.9 and g["fills"][0]["print_used"] == 2.75
    api = TestClient(appmod.app)
    rows = list(csv.DictReader(io.StringIO(api.get(f"/api/backtests/{rid}/fills.csv").text)))
    assert len(rows) == 4 and rows[0]["contract"] == occ and rows[0]["matches_stored_bar"] == "False" and rows[1]["matches_stored_bar"] == "True" and set(g["columns"]) == set(rows[0])
    assert api.get(f"/api/backtests/{rid}/fills").json()["matching"] == 3 and api.get("/api/backtests/nope/fills").status_code == 404
    card = api.get(f"/api/backtests/{rid}/performance").json()
    assert [s["id"] for s in card["sections"]] == ORDER and api.get("/api/backtests/nope/performance").status_code == 404


def test_the_baseline_is_the_same_position_at_the_same_time_without_the_signal(tmp_settings, monkeypatch):
    from fastapi.testclient import TestClient
    import strata.api.app as appmod
    from strata.api import db
    from strata.api.results import load_trades
    from strata.api.v4_api import _jobs
    from strata.research import performance as PF
    st = wave_store(tmp_settings)
    kids = [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}, {"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct >= 0.05"}}]
    rid = stored_run(st, cfg(entry_children=kids))
    mine = load_trades(rid)
    assert 0 < len(mine) < 10  # the signal holds on the sessions that rise through the morning, and only on those
    made, said = PF.baseline_config(db.get_run(rid)["config"], mine)
    assert said == "at 10:00 New York time on every session" and made["strategies"][0]["entry"]["conditions"]["children"] == [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]
    assert made["strategies"][0]["legs"] == db.get_run(rid)["config"]["strategies"][0]["legs"] or made["strategies"][0]["legs"][0]["strike"]["method"] == "atm_offset"  # the position is the run's
    b = PF.run_baseline(rid, st)
    assert b["trades"] == 10 and b["strategy_trades"] == len(mine) and b["definition"] == said and b["difference_in_average"] == round(b["strategy_average"] - b["average"], 2) and "not counted among the configurations tried" in b["note"]
    sec = next(s for s in card_of(rid, baseline=PF.saved_baseline(rid))["sections"] if s["id"] == "baseline")
    assert sec["figures"]["trades"] == 10 and "The same position entered at 10:00 New York time on every session" in sec["text"]
    # a rule with no clock time: the baseline enters at the minute the rule's own trades most often entered
    rid2 = stored_run(st, cfg(entry_children=[{"kind": "cond", "type": "expr", "params": {"expr": "crosses_above(from_open_pct, 0.1)"}}], max_entries_per_day=1))
    t2 = load_trades(rid2)
    if t2:
        m = max({t["entry_minute"] for t in t2}, key=[t["entry_minute"] for t in t2].count)
        assert PF.baseline_config(db.get_run(rid2)["config"], t2)[1] == f"at {m // 60:02d}:{m % 60:02d} New York time (the minute at which its own trades most often entered) on every session"
    # as a job, through the application
    monkeypatch.setattr("strata.api.v7_api._jobs", lambda: InlineJobs())
    api = TestClient(appmod.app)
    jid = api.post(f"/api/backtests/{rid}/baseline").json()["id"]
    assert db.get_run(jid)["status"] == "done" and db.get_run(jid)["summary"]["trades"] == 10
    assert next(s for s in api.get(f"/api/backtests/{rid}/performance").json()["sections"] if s["id"] == "baseline")["figures"]["trades"] == 10
    assert api.post("/api/backtests/nope/baseline").status_code == 404


def test_a_request_wider_than_the_stored_observations_says_so_before_any_figure(tmp_settings):
    """The run is made on the sessions that could be priced. The card states the period that was asked for, how many times
    the rule fired in it and how few of those the result describes, with the reasons for the rest."""
    from strata.api import db
    from strata.api.results import load_result, load_trades
    from strata.research import performance as PF
    st = grid_store(tmp_settings)
    rid = stored_run(st, cfg())
    res = load_result(rid)
    res["coverage"] = {"requested": {"start": "1993-01-29", "end": "2024-03-15", "sessions": 7800, "stated": False}, "signals": {"qualifying": 1079}, "executable": {"signals": 10, "of": 1079, "window": [DAYS[0].isoformat(), DAYS[-1].isoformat()]},
                       "options": {"issues": [{"reason": "no option trade data is stored for the session", "signals": 1024}, {"reason": "no contract with the required expiration was listed that day", "signals": 45}]}}
    c = PF.card(db.get_run(rid), res, load_trades(rid))
    cov = next(s for s in c["sections"] if s["id"] == "coverage")
    assert cov["figures"]["signals"] == 1079 and cov["figures"]["priced"] == 10 and cov["figures"]["priced_pct"] == 0.9 and cov["figures"]["period_stated"] is False
    assert cov["figures"]["not_priced_by_reason"] == {"no option trade data is stored for the session": 1024, "no contract with the required expiration was listed that day": 45}
    for phrase in ("Over the period asked for (1993-01-29 to 2024-03-15, all stored history: no period was stated)", "fired on 1,079 sessions", "10 of them (0.9%) could be priced", "Everything below describes those 10 and no others",
                   "Not priced, and not losses: 1,024 no option trade data is stored for the session; 45 no contract"):
        assert phrase in cov["text"], phrase


def test_what_is_stored_is_summarised_per_symbol_and_the_quote_steps_are_always_the_current_ones(tmp_settings):
    import json

    from fastapi.testclient import TestClient
    import strata.api.app as appmod
    from strata.config import get_settings
    from strata.data.providers import databento as DBN
    grid_store(tmp_settings, symbols=("SPY", "QQQ"))
    api = TestClient(appmod.app)
    d = api.get("/api/dataset/options").json()
    assert [(r["symbol"], r["sessions"], r["contract_sessions"], r["bars"], r["provider"]) for r in d["symbols"]] == [("QQQ", 10, 110, 550, "testfeed"), ("SPY", 10, 110, 550, "testfeed")]
    assert d["bars"] == 1100 and d["collector"]["enabled"] is False and d["quotes"] == {"connected": False, "text": "no historical bid/ask source is connected"}
    # a check saved by an earlier version carried that version's steps: the page shows the steps as they are written now
    p = get_settings().data_dir / "access"
    p.mkdir(parents=True, exist_ok=True)
    (p / "databento.json").write_text(json.dumps({"provider": "databento", "status": "blocked", "verified": False, "stages": [{"stage": "key", "ok": False, "detail": "no key stored"}], "how": ["an older instruction"], "notes": ["an older note"]}))
    q = api.get("/api/data/access/databento").json()
    assert q["key_stored"] is False and q["how"] == DBN.SETUP and q["notes"] == DBN.NOTES and "an older instruction" not in q["how"]
    assert api.post("/api/data/access/databento", json={"budget_usd": 500, "sessions": 20, "symbol": "SPY"}).status_code == 422  # a budget beyond the free credit is refused outright


def test_a_position_settled_at_expiration_is_a_settlement_and_is_not_compared_with_a_print(tmp_settings):
    from strata.research import performance as PF
    st = grid_store(tmp_settings)
    c = cfg(exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": False}})
    c["end"] = DAYS[0].isoformat()
    rid = stored_run(st, c)
    f = PF.fills(rid, st)
    entry, out = f["fills"]
    assert entry["action"] == "entry" and entry["matches_stored_bar"] is True and f["checked_against_stored_bars"] == 1 and f["matching"] == 1 and f["settled_at_expiration"] == 1
    assert out["action"] == "settlement" and out["price_source"] == "intrinsic value at expiration" and out["print_used"] is None and "matches_stored_bar" not in out
    assert "not a trade" in out["note"] and "intrinsic value" in out["note"]
