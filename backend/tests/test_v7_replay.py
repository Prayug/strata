"""V7: market replay shows stored observations, and only those.

An option chart made from trade bars has a bar where the contract traded and nothing where it did not; a mark carried across
a gap says how old it is; a level on the chart is the level the engine tested a rule against; and a trade is laid over the
observations with the time it was decided, the time it was filled and the age of the print it was filled on."""
from __future__ import annotations

import numpy as np
import pytest

from strata.core import calendar as cal
from strata.data.providers import sources as S
from tests.helpers import et_epoch
from tests.test_v2_research import tmp_settings  # noqa: F401
from tests.test_v7_engine import cfg, wave_store
from tests.test_v7_experiment import DAYS, PATH, PREV, grid_store

D = DAYS[0]
OCC = S.occ_symbol("SPY", D, "C", 500.0)


# ----------------------------------------------------------------------------- an option contract: bars where it traded
def test_a_contract_has_a_bar_in_each_minute_it_traded_and_nothing_in_the_others(tmp_settings):
    from strata.research import replay as RP
    st = grid_store(tmp_settings)
    c = RP.contract(st, "SPY", OCC, D)
    assert [b[0] for b in c["bars"]] == [et_epoch(D, t) for t in PATH] and c["candle_seconds"] == 60 and not c["aggregated"]
    assert [b[6] - b[0] for b in c["bars"]] == [60] * 5  # a print is observable when its minute has ended
    assert [b[4] for b in c["bars"]] == [2.75, 2.85, 2.7, 2.95, 2.75] and all(b[5] == 5 and b[7] == 1 for b in c["bars"])
    assert c["sessions"] == [{"date": D.isoformat(), "stored": True, "bars": 5, "volume": 25, "minutes_with_prints_pct": round(5 / 390 * 100, 1)}]
    assert c["observation"].startswith("trade prints (no bid/ask)") and "a missing minute is a minute without a trade" in c["note"]
    assert (c["strike"], c["right"], c["expiration"], c["provider"]) == (500.0, "C", D.isoformat(), "testfeed")
    # a contract that did not trade has no price here, and none is made for it
    none = RP.contract(st, "SPY", S.occ_symbol("SPY", D, "C", 530.0), D)
    assert none["bars"] == [] and "none is constructed" in none["note"] and none["sessions"][0]["bars"] == 0
    # the bell: a print in the last minute is observable at the close, not a minute after it
    late = S.occ_symbol("SPY", D, "P", 500.0)
    import polars as pl
    from datetime import datetime, timezone
    row = {"ts": datetime.fromtimestamp(et_epoch(D, "15:59"), tz=timezone.utc), "occ": late, "expiration": D, "strike": 500.0, "right": "P", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 3}
    st.write_option_bars("SPY", "testfeed", pl.DataFrame([row]).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": D, "occ": late, "expiration": D, "strike": 500.0, "right": "P", "bars": 1, "status": "ok", "pages": 1}])
    assert RP.contract(st, "SPY", late, D)["bars"][0][6] == int(cal.session_bounds(D)[1].timestamp())


def test_the_contracts_of_a_session_come_from_the_stored_index_and_long_spans_are_aggregated(tmp_settings):
    from strata.research import replay as RP
    st = grid_store(tmp_settings)
    s = RP.sessions(st, "SPY")
    assert (s["count"], s["first"], s["last"], s["provider"]) == (10, DAYS[0].isoformat(), DAYS[-1].isoformat(), "testfeed") and s["sessions"][0] == {"date": DAYS[-1].isoformat(), "contracts": 11, "bars": 55, "expirations": 1}
    k = RP.contracts(st, "SPY", D)
    e = k["expirations"][0]
    assert len(k["expirations"]) == 1 and (e["expiration"], e["dte"], e["contracts"], e["traded"], e["strikes"]) == (D.isoformat(), 0, 11, 11, [495.0, 505.0]) and e["rows"][5]["occ"] == OCC
    assert k["previous_close_as_traded"] == 500.0 and RP.contracts(st, "SPY", PREV)["expirations"] == [] and RP.sessions(st, "QQQ")["sessions"] == []
    # asked for a coarser candle, the stored bars are aggregated and the reply says so; a candle exists only where a print does
    agg = RP.contract(st, "SPY", OCC, D, bucket=1800)
    assert agg["aggregated"] and agg["candle_seconds"] == 1800 and "aggregated to 30-minute candles" in agg["note"]
    assert [(b[0], b[7]) for b in agg["bars"]] == [(et_epoch(D, "09:59"), 1), (et_epoch(D, "10:14"), 2), (et_epoch(D, "10:59"), 1), (et_epoch(D, "15:28"), 1)]
    assert agg["bars"][1][1:5] == [2.85, 2.85, 2.7, 2.7] and agg["bars"][1][5] == 10 and agg["bars"][1][6] == et_epoch(D, "10:30")


# ----------------------------------------------------------------------------- the underlying: the engine's own levels
def test_the_levels_on_the_chart_are_computed_from_the_stored_bars_as_the_engine_computes_them(tmp_settings):
    from strata.core import indicators as IND
    from strata.research import replay as RP
    st = wave_store(tmp_settings)
    u = RP.underlying(st, "SPY", D)
    bars = st.read_minutes("SPY", D, D).sort("ts")
    assert u["stored"] == "1m" and len(u["candles"]) == 390 and not u["aggregated"] and u["candles"][0][0] == int(cal.session_bounds(D)[0].timestamp())
    h, l, c, v = (bars[k].to_numpy().astype(float) for k in ("high", "low", "close", "volume"))
    want = IND.session_vwap(h, l, c, v, np.zeros(390, np.int64))
    assert u["vwap"] == [float(x) for x in want] and u["ema9"][-1] == float(IND.ema(c, 9)[-1])
    ses = u["sessions"][0]
    assert ses["opening_range"]["15"] == {"high": float(h[:15].max()), "low": float(l[:15].min()), "complete_at": u["candles"][0][0] + 900} and set(ses["opening_range"]) == {"5", "15", "30", "60"}
    prev = st.read_daily("SPY").filter(__import__("polars").col("date") == PREV)
    assert ses["previous"] == {"date": PREV.isoformat(), "close": float(prev["close"][0]), "high": float(prev["high"][0]), "low": float(prev["low"][0])}
    assert (ses["open"], ses["close"], ses["bars"]) == (float(bars["open"][0]), float(c[-1]), 390)
    # ten sessions do not fit one chart at one minute: they are aggregated, and each candle is exactly its stored bars
    span = RP.underlying(st, "SPY", DAYS[0], DAYS[-1])
    assert span["aggregated"] and span["candle_seconds"] == 300 and len(span["candles"]) == 780 and "aggregated to 5-minute candles to draw 10 sessions" in span["note"]
    first = span["candles"][0]
    assert first[1:] == [float(bars["open"][0]), float(h[:5].max()), float(l[:5].min()), float(c[4]), float(v[:5].sum())] and span["vwap"][0] == float(want[4])
    with pytest.raises(ValueError, match="not a trading session"):
        RP.underlying(st, "SPY", cal.prev_trading_day(D) if False else __import__("datetime").date(2024, 3, 9))


# ----------------------------------------------------------------------------- a trade over the observations
def stored_run(st, c: dict) -> str:
    from strata.api import db
    from strata.api.results import save_backtest
    from strata.engine.engine import run_backtest
    rid = db.create_run("backtest", "replay", c)
    db.update_run(rid, status="done", summary=save_backtest(rid, run_backtest(c, store=st, record_paths=False)))
    return rid


def test_a_trade_is_shown_with_its_decision_its_fill_the_age_of_its_print_and_marks_from_stored_prints(tmp_settings):
    from strata.api.results import load_trades
    from strata.research import replay as RP
    st = grid_store(tmp_settings)
    c = cfg(exits={"profit_target": {"enabled": True, "mode": "pct", "value": 30}, "stop_loss": {"enabled": True, "mode": "pct", "value": 50}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None},
                   "underlying": {"enabled": True, "move_for_pct": 1.0, "move_against_pct": 0.5}})
    c["end"] = D.isoformat()
    rid = stored_run(st, c)
    t = load_trades(rid)[0]
    r = RP.trade(rid, t["id"], st)
    kinds = {m["kind"]: m for m in r["markers"]}
    assert kinds["decision"]["t"] == t["signal_ts"] == et_epoch(D, "10:00") and kinds["entry"]["t"] == t["entry_ts"] and kinds["exit"]["t"] == t["exit_ts"] and kinds["exit"]["text"] == "time exit"
    age = t["meta"]["entry_print_age_seconds"]
    assert kinds["entry"]["print_age_seconds"] == age and kinds["entry"]["print_ts"] == t["entry_ts"] - age and f"on a print made {age}s before the fill" in kinds["entry"]["text"]
    leg = r["legs"][0]
    assert (leg["occ"], leg["entry_price"], len(leg["bars"]), leg["side"], leg["contracts"]) == (OCC, 2.75, 5, "buy", 1)
    # the running profit is the last stored print against the entry price: flat between prints, and the age of the print grows
    at = dict(zip(r["marks"]["t"], zip(r["marks"]["pnl"], r["marks"]["oldest_print_seconds"])))
    assert at[et_epoch(D, "10:15")] == (10.0, 0) and at[et_epoch(D, "10:20")] == (10.0, 300) and at[et_epoch(D, "10:30")] == (-5.0, 0) and at[et_epoch(D, "11:00")] == (20.0, 0) and at[et_epoch(D, "14:00")] == (20.0, 10800)
    assert at[t["exit_ts"]] == (round(t["gross_pnl"], 2), t["exit_ts"] - et_epoch(D, "15:29")) and r["marks"]["t"][0] > t["entry_ts"] - 60 and r["marks"]["t"][-1] == t["exit_ts"]
    lv = r["levels"]
    assert [(x["rule"], x["price"]) for x in lv["premium"]] == [("profit target", round(2.75 * 1.3, 4)), ("stop", round(2.75 * 0.5, 4))] and "30% of the 2.75 paid" in lv["premium"][0]["text"]
    assert [(x["rule"], x["price"]) for x in lv["underlying"]] == [("underlying target", 505.0), ("underlying stop", 497.5)] and lv["clock"] == {"rule": "clock exit", "time": "15:30"}
    assert r["underlying"]["stored"] == "1m" and len(r["underlying"]["candles"]) == 390 and r["entry_explain"] == t["entry_explain"] and "not quotes" in r["reading"]
    with pytest.raises(KeyError):
        RP.trade(rid, 999, st)


def test_a_position_held_over_several_sessions_is_marked_only_where_its_contract_printed(tmp_settings):
    import polars as pl
    from datetime import datetime, timezone
    from strata.api.results import load_trades
    from strata.research import replay as RP
    st = grid_store(tmp_settings)
    exp = DAYS[4]
    occ = S.occ_symbol("SPY", exp, "C", 500.0)
    frames, index = [], []
    for n, d in enumerate(DAYS[:5]):
        if n == 1:
            continue  # the contract does not trade at all on the second session
        r = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp, "strike": 500.0, "right": "C", "open": 6.0 + n, "high": 6.0 + n, "low": 6.0 + n, "close": 6.0 + n, "volume": 5} for t in ("09:59", "15:28")]
        frames.append(pl.DataFrame(r).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))))
        index.append({"date": d, "occ": occ, "expiration": exp, "strike": 500.0, "right": "C", "bars": 2, "status": "ok", "pages": 1})
    st.write_option_bars("SPY", "testfeed", pl.concat(frames), index)
    c = cfg(legs=[{"id": "lc", "side": "buy", "right": "C", "expiration": exp.isoformat(), "strike": {"method": "fixed", "value": 500.0}}],
            exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": 2}}, max_entries_per_day=1, max_open_positions=1)
    c["end"] = DAYS[2].isoformat()
    rid = stored_run(st, c)
    t = load_trades(rid)[0]
    assert t["entry_date"] == DAYS[0].isoformat() and t["exit_time"][:10] == DAYS[2].isoformat()
    r = RP.trade(rid, t["id"], st)
    assert [s["date"] for s in r["legs"][0]["sessions"]] == [d.isoformat() for d in DAYS[:3]] and [s["bars"] for s in r["legs"][0]["sessions"]] == [2, 0, 2]
    assert r["mark_step_seconds"] == 300 and len(r["underlying"]["sessions"]) == 3
    at = dict(zip(r["marks"]["t"], zip(r["marks"]["pnl"], r["marks"]["oldest_print_seconds"])))
    mid = et_epoch(DAYS[1], "12:00")
    # all through the session without a trade the mark is the first session's last print, and it says how old that is
    assert at[mid] == (0.0, mid - et_epoch(DAYS[0], "15:29")) and at[et_epoch(DAYS[2], "10:00")] == (200.0, 0)


def test_the_replay_routes(tmp_settings):
    from fastapi.testclient import TestClient
    import strata.api.app as appmod
    st = grid_store(tmp_settings)
    c = cfg()
    c["end"] = D.isoformat()
    rid = stored_run(st, c)
    api = TestClient(appmod.app)
    assert api.get("/api/replay/SPY/sessions").json()["count"] == 10
    day = api.get(f"/api/replay/SPY/{D.isoformat()}").json()
    assert len(day["underlying"]["candles"]) == 390 and day["contracts"]["expirations"][0]["contracts"] == 11
    assert len(api.get(f"/api/replay/SPY/contract/{OCC}", params={"start": D.isoformat()}).json()["bars"]) == 5
    assert api.get(f"/api/backtests/{rid}/trades/1/observed").json()["legs"][0]["occ"] == OCC
    assert api.get(f"/api/backtests/{rid}/trades/77/observed").status_code == 404 and api.get("/api/backtests/nope/trades/1/observed").status_code == 404
    assert api.get("/api/replay/SPY/not-a-date").status_code == 400 and api.get("/api/replay/SPY/contract/NOTACONTRACT", params={"start": D.isoformat()}).status_code == 400
    assert api.get("/api/replay/SPY/2024-03-09").status_code == 400  # a Saturday
