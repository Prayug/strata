"""V7: one sentence -> a bounded grid of configurations on stored observations.

The reader turns lists, ranges and steps into dimensions and never trims a grid; the runner scans each entry rule once, runs
every configuration only on the sessions where its rule fired, and reports the grid as a map (regions, earlier against later,
the search discounted) rather than as a winner. Everything here runs on a hand-written store of trade bars."""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 3, 29))[:10]
PREV = cal.prev_trading_day(DAYS[0])
SPOT = 500.0
# the print each contract makes in the minute that ENDS at the time shown, as a move from its first print of the day
PATH = {"09:59": 0.0, "10:14": 0.10, "10:29": -0.05, "10:59": 0.20, "15:28": 0.0}


def grid_store(tmp, symbols=("SPY",), skip_days: set | None = None) -> MarketStore:
    """Ten sessions of same-day calls, strikes 495-505. Odd sessions move the other way, so every cell has wins and losses."""
    st = MarketStore(tmp / "market")
    for sym in symbols:
        st.upsert_symbol(SymbolMeta(symbol=sym, kind="etf", multiplier=100.0))
        write_minutes(st, sym, [PREV] + DAYS, lambda d, i: SPOT)
        frames, index = [], []
        for n, d in enumerate(DAYS):
            if skip_days and d in skip_days:
                continue
            sign = 1 if n % 2 == 0 else -1
            for k in range(495, 506):
                occ = S.occ_symbol(sym, d, "C", float(k))
                base = max(0.5, 505.5 - k) / 2
                rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": d, "strike": float(k), "right": "C", "open": base + sign * m, "high": base + sign * m, "low": base + sign * m,
                         "close": round(base + sign * m, 2), "volume": 5} for t, m in PATH.items()]
                frames.append(pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))))
                index.append({"date": d, "occ": occ, "expiration": d, "strike": float(k), "right": "C", "bars": len(rows), "status": "ok", "pages": 1})
        st.write_option_bars(sym, "testfeed", pl.concat(frames), index)
    return st


def base_draft(text="Buy ATM SPY 0DTE calls at 10:00 AM Eastern"):
    from strata.ai import nl_rules as NL
    from strata.ai.nl_stateful import enrich
    return enrich(text, NL.TICKERS)


def read(text, current=None, known=("SPY", "QQQ", "IWM")):
    from strata.ai import nl_experiment as NX
    return NX.read(text, current, list(known))


# ----------------------------------------------------------------------------- the reader
def test_the_sentence_of_the_brief_is_a_grid_of_252_configurations_with_its_assumptions_said():
    r = read("Compare entry times from 9:30 to 11:00 in 15-minute increments, prior-day declines from 0.5% to 3%, 0DTE versus 1DTE, and several profit targets.", base_draft())
    dims = {d["field"]: d for d in r["dims"]}
    assert r["action"] == "experiment" and r["configs"] == 7 * 6 * 2 * 3 == 252 and r["base_from"] == "current"
    assert dims["entry_time"]["values"] == ["open", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00"] and not dims["entry_time"]["assumed"]
    assert dims["prev_decline"]["values"] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] and dims["prev_decline"]["assumed"]  # the step was not stated: half a percent is assumed, and said
    assert dims["dte"]["values"] == [0, 1] and dims["premium_target"]["values"] == [25.0, 50.0, 100.0] and dims["premium_target"]["assumed"]
    assert any("several profit targets" in n for n in r["notes"]) and any("No step was given for the decline" in n for n in r["notes"])
    # the base carries the first value of the decline it varies: the grid changes that one condition and nothing else
    assert [c.expr for c in r["draft"].conditions] == ["previous_session.return_pct <= -0.5"]


def test_a_list_of_values_is_a_comparison_and_a_comparison_needs_a_strategy():
    from strata.ai import nl_experiment as NX
    r = read("Compare exits after 15, 30, 60, and 120 minutes.", base_draft())
    assert r["action"] == "experiment" and r["dims"] == [{"field": "exit_minutes", "values": [15, 30, 60, 120], "text": "exits after 15, 30, 60, and 120 minutes", "assumed": False}]
    cfgs = NX.configurations(r)
    assert [c["label"] for c in cfgs] == ["15 min", "30 min", "60 min", "120 min"] and [c["draft"].hold_minutes for c in cfgs] == [15, 30, 60, 120]
    assert all(c["draft"].entry_time == "10:00" and c["draft"].structure == "long_call" for c in cfgs)  # nothing else changed
    alone = read("Compare exits after 15, 30, 60, and 120 minutes.")
    assert alone["action"] == "experiment_needs_base" and "Which strategy" in alone["question"] and "draft" not in alone
    # the sentence can carry its own strategy, and a clock in it is read in the zone of the sentence
    own = read("Buy SPY 0DTE calls at the open after a 1% down day and compare 20, 40 and 80 delta strikes with exits after 5, 15 and 45 minutes.")
    assert own["base_from"] == "sentence" and own["configs"] == 9 and own["draft"].entry_time == "open" and [c.expr for c in own["draft"].conditions] == ["previous_session.return_pct <= -1"]
    assert {d["field"]: d["values"] for d in own["dims"]} == {"strike": [("delta", 0.2), ("delta", 0.4), ("delta", 0.8)], "exit_minutes": [5, 15, 45]}
    close = read("Hold for 20 minutes versus 90 minutes versus to the close.", base_draft())
    assert close["dims"][0]["values"] == [20, 90, "close"] and NX.configurations(close)[2]["draft"].exit_time == "close" and NX.configurations(close)[2]["draft"].hold_minutes is None


def test_a_grid_that_is_too_large_is_refused_whole_and_never_trimmed():
    from strata.ai import nl_experiment as NX
    r = read("Compare entry times from 9:30 to 15:30 in 5-minute increments with exits after 5, 10, 15, 20, 30, 45, 60, 90 and 120 minutes and 0DTE, 1DTE, 2DTE and 7DTE.", base_draft())
    assert r["configs"] > NX.MAX_CONFIGS and "nothing was trimmed" in r["too_many"] and f"{r['configs']:,}" in r["too_many"]
    assert len(next(d for d in r["dims"] if d["field"] == "exit_minutes")["values"]) == 9  # the dimensions are still what was asked


def test_times_of_one_sentence_are_read_in_one_zone_and_the_bell_is_new_york(tmp_settings):
    from strata.api.settings import set_setting
    set_setting("ui.timezone", "America/Los_Angeles")
    r = read("Compare entry times from 9:30 to 10:30 in 30-minute increments.", base_draft())
    assert r["dims"][0]["values"] == ["open", "10:00", "10:30"]  # 9:30 with no zone is the opening bell, not 12:30 in New York
    p = read("Compare entering at 7:00 AM, 8:00 AM and 9:00 AM Pacific.", base_draft())
    assert p["dims"][0]["values"] == ["10:00", "11:00", "12:00"]


# ----------------------------------------------------------------------------- the runner
def test_configurations_share_one_signal_scan_and_each_equals_the_same_run_on_its_own(tmp_settings, monkeypatch):
    from strata.engine import scan as SC
    from strata.engine.engine import run_backtest
    from strata.research import experiment as EXP
    st = grid_store(tmp_settings)
    plan = read("Compare exits after 15, 30 and 60 minutes and ATM versus 1% OTM contracts.", base_draft())
    assert plan["configs"] == 6
    payload = EXP.build(plan, st)
    assert len(payload["configs"]) == 6 and payload["errors"] == [] and {c["config"]["start"] for c in payload["configs"]} == {DAYS[0].isoformat()}
    assert len({EXP.entry_signature(c["config"]) for c in payload["configs"]}) == 1  # strike and exit do not change when the rule fires
    scans, real = [], SC.scan_signals
    monkeypatch.setattr(SC, "scan_signals", lambda cfg, store=None, **kw: (scans.append(1), real(cfg, store, **kw))[1])
    res = EXP.run_experiment(payload, st, pool=False)
    assert len(scans) == 1 and res["summary"]["timing"]["entry_rules"] == 1 and res["summary"]["timing"]["scans_shared_by"] == 6.0
    for c in res["cells"]:
        alone = run_backtest(next(x["config"] for x in payload["configs"] if x["i"] == c["i"]), store=st, record_paths=False)
        assert c["status"] == "ok" and len(c["trades"]) == len(alone.trades) == 10 and [t["pnl"] for t in c["trades"]] == [round(float(t["pnl"]), 2) for t in alone.trades], c["label"]
        assert c["metrics"]["coverage"] == {"signals": 10, "executed": 10, "pct": 100.0, "not_executed": {}, "deferred_exits": c["metrics"]["coverage"]["deferred_exits"], "fetchable": 0}
    atm15 = next(c for c in res["cells"] if c["key"] == {"exit_minutes": 15, "strike": ["atm", 0.0]})
    # bought on the 09:59 print, sold on the 10:14 print: +0.10 on even sessions, -0.10 on odd ones, less the costs of a round trip
    assert [t["exit_time"] for t in atm15["trades"]] == ["10:15:00"] * 10 and {round(t["exit"] - t["entry"], 2) for t in atm15["trades"]} == {0.10, -0.10}
    assert atm15["metrics"]["trades"] == 10 and atm15["metrics"]["win_rate"] == 50.0 and atm15["metrics"]["max_consecutive_losses"] == 1


def test_the_split_is_by_date_and_the_same_for_every_configuration_and_worse_fills_are_first_order(tmp_settings):
    from strata.research import experiment as EXP
    st = grid_store(tmp_settings)
    res = EXP.run_experiment(EXP.build(read("Compare exits after 15, 30 and 60 minutes.", base_draft()), st), st, pool=False)
    assert res["split"] == DAYS[7].isoformat() and res["window"] == [DAYS[0].isoformat(), DAYS[-1].isoformat()]  # the last 30% of ten sessions
    for c in res["cells"]:
        m = c["metrics"]
        assert m["earlier"]["trades"] == 7 and m["later"]["trades"] == 3 and round(m["earlier"]["net"] + m["later"]["net"], 2) == m["net"]
        turn = sum(t["turnover"] for t in c["trades"])
        assert m["sensitivity"] == {"2%": round(m["net"] - turn * 0.02, 2), "5%": round(m["net"] - turn * 0.05, 2)} and m["sensitivity"]["5%"] < m["sensitivity"]["2%"] < m["net"]
    assert "Both parts are history that has now been examined; neither is evidence of an edge." in res["summary"]["text"]
    assert res["observation"].startswith("trade prints (no bid/ask)")


def test_a_configuration_reports_the_signals_the_stored_observations_could_not_price(tmp_settings):
    from strata.research import experiment as EXP
    st = grid_store(tmp_settings, skip_days={DAYS[3], DAYS[4]})
    res = EXP.run_experiment(EXP.build(read("Compare ATM versus 3% OTM contracts.", base_draft()), st), st, pool=False)
    atm, far = (next(c for c in res["cells"] if c["key"]["strike"][0] == m) for m in ("atm", "pct_otm"))
    assert atm["metrics"]["coverage"]["signals"] == 10 and atm["metrics"]["coverage"]["executed"] == 8 and atm["metrics"]["coverage"]["pct"] == 80.0 and atm["metrics"]["trades"] == 8
    # 3% out of the money is 515: beyond the stored strikes on every session. Not one trade, and not one loss
    assert far["metrics"]["trades"] == 0 and far["metrics"]["net"] == 0.0 and far["metrics"]["coverage"]["executed"] == 0 and far["metrics"]["coverage"]["fetchable"] == 10
    assert "configurations are not all tested on the same sessions" in res["summary"]["text"] and res["summary"]["coverage"]["fetchable_signals"] == 12


def test_the_same_signal_on_several_underlyings_scans_each_one_once(tmp_settings):
    from strata.research import experiment as EXP
    st = grid_store(tmp_settings, symbols=("SPY", "QQQ"))
    plan = read("Compare the same signal across SPY and QQQ with exits after 15 and 30 minutes.", base_draft(), known=("SPY", "QQQ"))
    res = EXP.run_experiment(EXP.build(plan, st), st, pool=False)
    assert res["summary"]["timing"]["entry_rules"] == 2 and sorted({c["symbol"] for c in res["cells"]}) == ["QQQ", "SPY"] and all(c["metrics"]["trades"] == 10 for c in res["cells"])
    assert res["stability"]["rows"] and all(r["neighbours"] == 1 for r in res["stability"]["rows"])  # symbols have no order: only the exit has neighbours


# ----------------------------------------------------------------------------- the report
def cells(grid: dict[tuple, tuple[float, int]], split=(0.0, 0.0)) -> list[dict]:
    out = []
    for i, ((a, b), (ex, n)) in enumerate(grid.items()):
        m = {"trades": n, "expectancy": ex, "net": round(ex * n, 2), "t": 2.9 if ex == max(v[0] for v in grid.values()) else 0.5, "earlier": {"trades": 20, "expectancy": ex + split[0]},
             "later": {"trades": 9, "expectancy": -ex + split[1]}, "coverage": {"pct": 100.0, "fetchable": 0}}
        out.append({"i": i, "key": {"x": a, "y": b}, "label": f"x{a} y{b}", "status": "ok", "metrics": m})
    return out


def test_a_peak_that_stands_alone_is_called_one_and_the_stable_region_is_named_beside_it():
    from strata.research import experiment as EXP
    dims = [{"field": "x", "values": [1, 2, 3]}, {"field": "y", "values": [1, 2, 3]}]
    grid = {(x, y): (5.0, 30) for x in (1, 2, 3) for y in (1, 2, 3)}
    grid.update({(1, 1): (40.0, 30), (1, 2): (-8.0, 30), (2, 1): (-6.0, 30), (3, 3): (9.0, 30), (3, 2): (8.0, 30), (2, 3): (7.0, 30), (2, 2): (4.0, 30)})
    s = EXP.stability(dims, cells(grid))
    best, stable = next(r for r in s["rows"] if r["best"]), next(r for r in s["rows"] if r["stable"])
    assert best["label"] == "x1 y1" and best["region_min"] == -8.0 and "a peak that stands alone" in s["text"]
    assert stable["label"] == "x3 y3" and stable["region_min"] == 7.0 and stable["same_sign_pct"] == 100 and "The most stable region is around x3 y3" in s["text"]
    # a cell with too few trades is not compared, and is not a neighbour either
    thin = cells({**grid, (3, 3): (90.0, 4)})
    assert all(r["label"] != "x3 y3" for r in EXP.stability(dims, thin)["rows"]) and "No configuration has" in EXP.stability(dims, cells({k: (1.0, 3) for k in grid}))["text"]


def test_the_best_cell_is_discounted_for_the_search_and_the_ranking_is_compared_across_the_split():
    from strata.research import experiment as EXP
    grid = {(x, y): (float(x * 3 + y), 30) for x in (1, 2, 3) for y in (1, 2, 3)}
    cs = cells(grid)
    sm = EXP.summarise({}, cs, "2026-08-18", {"seconds": 1.0})
    p_raw = math.erfc(2.9 / math.sqrt(2))
    assert sm["best"]["label"] == "x3 y3" and sm["best"]["p_unadjusted"] == pytest.approx(p_raw) and sm["best"]["p_adjusted_for_configurations"] == pytest.approx(p_raw * 9)
    assert sm["earlier_later_rank_correlation"] == -1.0 and "tends to REVERSE" in sm["text"] and sm["chosen_on_earlier"]["label"] == "x3 y3" and sm["chosen_on_earlier"]["later"]["expectancy"] == -12.0
    assert "as the best of 9 configurations" in sm["text"] and sm["profitable"] == 9 and sm["configurations"] == 9
    none = EXP.summarise({}, [{**c, "metrics": {**c["metrics"], "trades": 0}} for c in cs], "2026-08-18", {})
    assert none["text"] == "9 of 9 configurations ran and none of them traded on the stored observations." and "best" not in none


def test_statistics_of_one_cell():
    from strata.research import experiment as EXP
    s = EXP.stats([{"pnl": x} for x in (10, -5, -5, 20, -10, 30)])
    assert s["trades"] == 6 and s["net"] == 40.0 and s["expectancy"] == 6.67 and s["win_rate"] == 50.0 and s["profit_factor"] == 3.0 and s["max_consecutive_losses"] == 2
    assert s["max_drawdown"] == 10.0 and s["largest_share"] == 75.0 and s["ci95"][0] < 6.67 < s["ci95"][1] and EXP.stats([{"pnl": 1}] * 4)["ci95"] is None
    assert EXP.stats([]) == {"trades": 0, "net": 0.0, "expectancy": None, "win_rate": None, "profit_factor": None, "max_drawdown": 0.0, "ci95": None, "t": None, "largest_share": None}


# ----------------------------------------------------------------------------- through the application
def test_a_comparison_said_in_the_conversation_runs_as_one_job_and_counts_every_cell_as_a_trial(tmp_settings, monkeypatch):
    from strata.ai.agent import Agent
    from strata.api import db
    from strata.research import experiment as EXP
    from strata.research import ledger as LG
    monkeypatch.setattr(EXP, "POOL", False)
    st = grid_store(tmp_settings)
    ag, sid = Agent(InlineJobs(), st, llm_factory=lambda: None), db.create_session("t")["id"]
    first = ag.post(sid, "Buy ATM SPY 0DTE calls at 10:00 AM Eastern, sell at 3:30 PM Eastern")[-1]
    assert first["kind"] == "plan" and LG.count(sid) == 1
    msg = ag.post(sid, "Compare exits after 15, 30 and 60 minutes.")[-1]
    c = msg["content"]
    assert msg["kind"] == "experiment" and c["configurations"] == 3 and c["base_from"] == "current" and "Every configuration counts as a trial" in c["text"] and len(c["runs"]) == 1
    run = db.get_run(c["runs"][0]["run_id"])
    assert run["kind"] == "experiment" and run["status"] == "done" and run["summary"]["configurations"] == 3 and run["summary"]["with_trades"] == 3
    assert LG.count(sid) == 4  # the first run and the three cells
    res = EXP.load(run["id"])
    assert [x["label"] for x in res["cells"]] == ["15 min", "30 min", "60 min"] and "trades" not in res["cells"][0] and len(EXP.trades_of(run["id"], res["cells"][0]["i"])) == 10
    rows = EXP.csv_rows(res)
    assert rows[0]["setting_exit_minutes"] == 15 and rows[0]["trades"] == 10 and rows[0]["coverage_pct"] == 100.0 and "net_with_2%_worse_fills" in rows[0]
    done = [m for m in ag.finalize(sid) if m["content"].get("view") == "experiment"]
    assert len(done) == 1 and done[0]["content"]["run_id"] == run["id"] and done[0]["content"]["summary"]["configurations"] == 3
    # a comparison with nothing to apply it to asks; it does not invent a strategy
    sid2 = db.create_session("u")["id"]
    q = ag.post(sid2, "Compare exits after 15, 30 and 60 minutes.")[-1]
    assert q["kind"] == "text" and "Which strategy" in q["content"]["text"] and len(db.list_runs()) == 2
