"""V4 execution uncertainty, adaptive contract selection against complete chains, streaming
bulk imports, provider verification and the data collector. Everything is offline."""
from __future__ import annotations

import json
import zipfile
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.store import MarketStore, SymbolMeta
from strata.engine.engine import run_backtest
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)

D = date(2024, 3, 6)
PREV = cal.prev_trading_day(D)


def occ(strike: float, right: str = "C", exp: date = D, root: str = "SPY") -> str:
    return f"{root}{exp:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def chain_store(tmp, strikes, spot=500.0, exp=D, prints=("09:35", "09:58", "10:00", "10:03", "12:00", "15:20"), provider="testfeed", volume=40, complete=False, vol=0.01) -> MarketStore:
    """Trade bars for calls at the given strikes. A contract's price is its intrinsic value
    plus a time value that decays with distance from the money, so deltas implied by the
    prints are monotone in strike."""
    st = MarketStore(tmp / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [PREV, D], lambda d, i: spot)
    add_strikes(st, strikes, spot, exp, prints, provider, volume, complete, vol)
    return st


def add_strikes(st, strikes, spot=500.0, exp=D, prints=("09:35", "09:58", "10:00", "10:03", "12:00", "15:20"), provider="testfeed", volume=40, complete=False, vol=0.01):
    rows, idx = [], []
    for k in strikes:
        px = max(spot - k, 0.0) + 3.0 * float(np.exp(-0.5 * ((k - spot) / (spot * vol)) ** 2)) + 0.02
        o = occ(k, "C", exp)
        for i, t in enumerate(prints):
            p = round(px * (1 + 0.01 * i), 2)
            rows.append({"ts": datetime.fromtimestamp(et_epoch(D, t), tz=timezone.utc), "occ": o, "expiration": exp, "strike": float(k), "right": "C", "open": p, "high": p, "low": p, "close": p, "volume": volume})
        idx.append({"date": D, "occ": o, "expiration": exp, "strike": float(k), "right": "C", "bars": len(prints), "status": "ok", "pages": 1})
    st.write_option_bars("SPY", provider, pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), idx, **({"complete_chain": True} if complete else {}))


def strat(method="atm_offset", value=0.0, **kw) -> dict:
    return {"name": "call", "symbol": "SPY", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": method, "value": value, "unit": "points"}, **kw}],
            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}


def bt(st, s=None, **ex):
    return run_backtest({"start": D.isoformat(), "end": D.isoformat(), "strategies": [s or strat()], "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1, **ex}}, store=st)


# ----------------------------------------------------------------------------- execution assumptions
def test_worse_execution_assumptions_change_fills_missed_orders_and_size(tmp_path):
    st = chain_store(tmp_path, [498, 499, 500, 501, 502])
    base = bt(st)
    assert len(base.trades) == 1 and base.trades[0]["legs"][0]["strike"] == 500.0
    e0, x0 = base.trades[0]["legs"][0]["entry_price"], base.trades[0]["legs"][0]["exit_price"]
    worse = bt(st, extra_slippage_pct=5.0).trades[0]["legs"][0]
    assert np.isclose(worse["entry_price"], e0 * 1.05) and np.isclose(worse["exit_price"], x0 * 0.95), "a buy fills higher and the closing sale lower"
    assert worse["entry_print"] == base.trades[0]["legs"][0]["entry_print"], "the observed print is reported unchanged; only the assumed fill moves"
    pn = [bt(st, extra_slippage_pct=x).trades[0]["pnl"] for x in (0, 1, 2, 5, 10)]
    assert all(a > b for a, b in zip(pn, pn[1:])), "more slippage must never help"
    # a delayed order uses the print that exists when it arrives - the 10:03 one, not the 10:00 one
    late = bt(st, latency_seconds=240, max_fill_delay_minutes=10).trades[0]
    assert late["fill_delay_seconds"] >= 240 and late["legs"][0]["entry_print"] > base.trades[0]["legs"][0]["entry_print"]
    # minimum traded volume: 80 contracts have traded at 10:00, 120 once the 10:00 bar completes - the order waits for it
    waited = bt(st, strat(min_volume=100)).trades
    assert len(waited) == 1 and waited[0]["fill_delay_seconds"] >= 60
    missed = bt(st, strat(min_volume=500))
    assert not missed.trades and missed.meta["executability"]["not_executed"] == 1
    # participation: at 10:00 two bars have completed (80 contracts; the 10:00 bar is still forming) -> at most 8 contracts
    big = strat()
    big["sizing"] = {"method": "fixed", "contracts": 30}
    part = bt(st, big, max_volume_pct=10.0, allow_partial_fills=True)
    assert part.trades[0]["units"] == 8 and part.diagnostics["partial_fills"] == 1
    none = bt(st, big, max_volume_pct=10.0, allow_partial_fills=False)
    assert not none.trades and "insufficient displayed liquidity" in " ".join(none.diagnostics["skipped_entries"])


def test_multi_leg_positions_can_be_charged_for_legging_and_single_legs_are_not(tmp_path):
    st = chain_store(tmp_path, [498, 499, 500, 501, 502, 505])
    spread = strat()
    spread["legs"].append({"id": "sc", "side": "sell", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "leg_offset", "value": 5, "unit": "points", "ref_leg": "lc"}})
    a, b = bt(st, spread).trades[0], bt(st, spread, multi_leg_extra_pct=4.0).trades[0]
    la, lb = {l["id"]: l for l in a["legs"]}, {l["id"]: l for l in b["legs"]}
    assert np.isclose(lb["lc"]["entry_price"], la["lc"]["entry_price"] * 1.04) and np.isclose(lb["sc"]["entry_price"], la["sc"]["entry_price"] * 0.96) and b["pnl"] < a["pnl"]
    assert bt(st, multi_leg_extra_pct=4.0).trades[0]["pnl"] == bt(st).trades[0]["pnl"], "a single-leg position has no legging risk"


def test_the_stress_ladder_is_built_from_what_the_data_can_and_cannot_test(tmp_path):
    from strata.research import execution as EX
    cfg = {"start": D.isoformat(), "end": D.isoformat(), "strategies": [strat()], "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1}}
    sc = {s["id"]: s for s in EX.build_scenarios(cfg, quotes_available=False, eod=False)}
    assert {"signal", "base", "slip1", "slip10", "delay60", "age60", "vol50", "part10", "part10x", "adverse", "quotes"} <= set(sc)
    assert "skipped" in sc["quotes"] and "spread is unknown" in sc["quotes"]["skipped"] and "not estimated" in sc["quotes"]["skipped"]
    assert "an assumption about the unobserved spread" in sc["slip2"]["assumption"] and "legs2" not in sc
    assert sc["signal"]["config"]["execution"]["commission_per_contract"] == 0 and sc["adverse"]["config"]["strategies"][0]["legs"][0]["min_volume"] == 10
    eod = {s["id"]: s for s in EX.build_scenarios({**cfg, "pricing": {"mode": "historical"}}, eod=True)}
    assert "skipped" in eod["delay"] and "end-of-day" in eod["delay"]["skipped"] and "delay60" not in eod and "latency_seconds" not in eod["adverse"]["config"]["execution"]
    assert eod["signal"]["config"]["execution"]["fill_model"] == "midpoint" and "quotes" not in eod
    assert "config" in {s["id"]: s for s in EX.build_scenarios(cfg, quotes_available=True)}["quotes"]

    st = chain_store(tmp_path, [498, 499, 500, 501, 502])
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    try:
        out = EX.run_stress(cfg, str(tmp_path), workers=1, only=["slip2", "slip10", "vol100", "part10x", "adverse"])
    finally:
        override_settings(data_dir=old)
    rows = {r["id"]: r for r in out["rows"]}
    assert rows["signal"]["net_pnl"] > rows["base"]["net_pnl"] > rows["slip2"]["net_pnl"] > rows["slip10"]["net_pnl"], "signal-only >= as simulated >= worse fills"
    assert rows["slip2"]["paired"] == {"pairs": 1, "mean_diff": rows["slip2"]["delta_net_pnl"], "worse": 1, "better": 0} or rows["slip2"]["paired"] is None
    assert rows["quotes"]["status"] == "not_testable" and rows["base"]["thin"] and out["summary"]["thin"]
    assert rows["adverse"]["trades"] in (0, 1) and out["summary"]["execution_cost"] > 0
    assert "NOT observed" in out["notes"][0] and any("re-simulation" in n for n in out["notes"]) and any("Only 1 trades" in n for n in out["notes"])
    del st


def test_an_option_backtest_is_reviewed_with_its_own_stress_runs_benchmark_and_lookahead_probe(tmp_settings):
    """The whole review of a real pipeline run: the stress ladder, one-step parameter changes, the
    unconditional benchmark and the future-scramble probe all run, and every topic is reported."""
    from strata.api import db
    from strata.api.results import run_dir
    from tests.test_v3_agent import agent, session, trade_store
    st = trade_store(tmp_settings)
    ag, sid = agent(st), session()
    ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    ag.finalize(sid)
    ag.post(sid, "Test ATM 0DTE calls on those days using historical option trade prices.")
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "single" and res["mode"] == "trades"
    m = ag.post(sid, "Challenge this strategy")[-1]
    assert m["kind"] == "challenge" and m["content"]["target"] == res["run_id"]
    out = ag.finalize(sid)[0]["content"]
    assert out["view"] == "challenge", out.get("text")
    full = json.loads((run_dir(out["run_id"]) / "challenge.json").read_text())
    f = {x["topic"]: x for x in full["findings"]}
    assert len(f) == 16 and full["subject"] == "backtest" and full["stress"]["summary"]["trades"] == 1
    assert f["lookahead"]["severity"] == "ok" and "scrambled" in f["lookahead"]["evidence"]
    assert f["sample"]["severity"] == "high" and "1 trades" in f["sample"]["evidence"]
    assert f["coverage"]["severity"] in ("medium", "high") and "1 of 2 signals could be priced" in f["coverage"]["title"]
    assert f["fills"]["severity"] == "high" and "never observed" in f["fills"]["evidence"], "trade prints with no slippage are flagged as optimistic"
    assert f["contracts"]["severity"] in ("medium", "high") and "band of strikes" in f["contracts"]["evidence"]
    assert f["costs"]["test"] == "execution stress test" and f["costs"]["table"]["rows"]
    assert "one market environment" in f["regime"]["evidence"] and f["leverage"]["evidence"]
    rows = {r["id"]: r for r in full["stress"]["rows"]}
    assert rows["signal"]["net_pnl"] > rows["base"]["net_pnl"] > rows["slip10"]["net_pnl"] and rows["quotes"]["status"] == "not_testable"
    assert full["counts"]["high"] >= 2 and "should not be relied on" in full["verdict"]
    # the stress ladder alone, and the reasoning tools that read it
    m = ag.post(sid, "Stress-test the execution")[-1]
    ex = ag.finalize(sid)[0]["content"]
    assert m["kind"] == "stress" and ex["view"] == "execution" and ex["summary"]["signal_net_pnl"] > ex["summary"]["simulated_net_pnl"]
    a = ag.post(sid, "Which assumptions have the biggest effect on this backtest?")[-1]["content"]
    assert a["view"] == "analysis" and a["analysis"]["answerable"] and a["analysis"]["tables"][0]["rows"], a["text"]
    a = ag.post(sid, "Would these results survive higher transaction costs?")[-1]["content"]
    assert a["tool"] == "cost_survival" and "Re-simulated under worse execution" in a["analysis"]["tables"][0]["title"]
    assert db.get_run(out["run_id"])["summary"]["target_run"] == res["run_id"]


# ----------------------------------------------------------------------------- contract selection: selective band vs complete chain
FULL = list(range(470, 531))


@pytest.mark.parametrize("method,value", [("atm_offset", 0.0), ("atm_offset", 2.0), ("pct_otm", 0.4), ("delta", 0.30), ("delta", 0.45), ("premium", 1.5)])
def test_a_selective_download_selects_the_same_contract_as_the_complete_chain(tmp_path, method, value):
    """The band the planner would download (around the price at the signal, from information
    known then) must lead to the same contract, fills and P&L as the full chain."""
    from strata.data.planner import leg_targets
    leg = strat(method, value)["legs"][0]
    k0, n = leg_targets([leg], 500.0, 0.16, 1.0)["lc"]
    band = [k for k in FULL if k0 - n - 1e-6 <= k <= k0 + n + 1e-6]
    assert 0 < len(band) < len(FULL)
    full = bt(chain_store(tmp_path / "full", FULL, complete=True), strat(method, value))
    sel = bt(chain_store(tmp_path / "sel", band), strat(method, value))
    assert len(full.trades) == len(sel.trades) == 1
    a, b = full.trades[0]["legs"][0], sel.trades[0]["legs"][0]
    assert (a["strike"], a["entry_price"], a["exit_price"]) == (b["strike"], b["entry_price"], b["exit_price"]) and full.trades[0]["pnl"] == sel.trades[0]["pnl"]
    cs_full, cs_sel = full.meta["contract_selection"], sel.meta["contract_selection"]
    assert cs_full["complete_chain"] and not cs_full["selective_storage"] and cs_full["at_band_edge"] == 0
    assert cs_sel["selective_storage"] and cs_sel["at_band_edge"] == 0, "inside the band nothing sits on its edge"


def test_a_selection_on_the_edge_of_the_band_is_reported_and_widening_restores_equivalence(tmp_path):
    """A low delta target lies outside a narrow band. Since V7 the outermost stored strike is not bought in its place (it
    was, with a flag): the signal is left unfilled and its session is handed to the widening loop. Once the band is widened
    the selection equals the complete chain's. A target the band does hold on its edge is still traded and still flagged."""
    s = strat("delta", 0.05)
    full = bt(chain_store(tmp_path / "full", FULL, complete=True), s)
    want = full.trades[0]["legs"][0]["strike"]
    narrow = chain_store(tmp_path / "sel", list(range(497, 504)))
    r1 = bt(narrow, s)
    cs = r1.meta["contract_selection"]
    assert r1.trades == [] and want > 503.0, "the 503 strike is not passed off as the 5-delta call"
    assert r1.meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert cs["beyond_band"] == 1 and cs["beyond_dates"] == [D.isoformat()] and cs["at_band_edge"] == 0 and cs["selective_storage"]
    add_strikes(narrow, list(range(504, 520)))  # what the adaptive widening downloads
    r2 = bt(narrow, s)
    cs2 = r2.meta["contract_selection"]
    assert r2.trades[0]["legs"][0]["strike"] == want and cs2["at_band_edge"] == 0 and cs2["beyond_band"] == 0 and r2.trades[0]["pnl"] == full.trades[0]["pnl"]
    # a target the band holds on its edge, within tolerance: traded, and flagged so that the loop looks one step further
    edge = chain_store(tmp_path / "edge", list(range(497, 504)))
    near = float(bt(chain_store(tmp_path / "full2", FULL, complete=True), strat("fixed", 503.0)).meta["contract_selection"]["selections"])  # the full chain trades the 503 strike
    assert near == 1
    d503 = bt(edge, strat("fixed", 503.0))
    assert d503.trades[0]["legs"][0]["strike"] == 503.0
    # the complete chain is never refused or flagged, even when the best contract is its outermost strike: there is nothing beyond it
    tiny = bt(chain_store(tmp_path / "tiny", list(range(497, 504)), complete=True), s)
    assert tiny.trades[0]["legs"][0]["strike"] == 503.0 and tiny.meta["contract_selection"]["at_band_edge"] == 0 and tiny.meta["contract_selection"]["beyond_band"] == 0


def test_the_planner_widens_the_band_only_from_information_known_at_the_signal(tmp_path):
    from strata.data import planner as P
    from strata.data.providers import sources as S
    st = chain_store(tmp_path, [500])
    seen = []

    class Src:
        def list_contracts(self, client, sym, as_of, e_lo, e_hi, lo, hi, rights):
            seen.append((as_of, e_lo, e_hi, round(lo, 3), round(hi, 3), rights))
            return [S.contract_from_occ(occ(k, rights[0], e_lo)) for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    leg = strat("delta", 0.30)["legs"][0]
    sig = {"date": D, "price": 500.0}
    w0, _ = P.trade_contracts(Src(), None, st, "SPY", [leg], sig, 0)
    w1, _ = P.trade_contracts(Src(), None, st, "SPY", [leg], sig, 0, widen=1)
    w2, _ = P.trade_contracts(Src(), None, st, "SPY", [leg], sig, 0, widen=2)
    n0, n1, n2 = (len(w[D]) for w in (w0, w1, w2))
    assert n1 > n0 and n2 > n1 and {c.strike for c in w0[D]} < {c.strike for c in w1[D]} < {c.strike for c in w2[D]}
    assert all(s[0] == D and s[1] >= D for s in seen), "contracts are listed as of the signal session, for expirations on or after it"
    centre = (seen[0][3] + seen[0][4]) / 2
    assert all(abs((s[3] + s[4]) / 2 - centre) < 1e-6 for s in seen), "widening keeps the band centred on the target chosen at the signal"
    fixed = strat("fixed", 505.0)["legs"][0]
    f0, _ = P.trade_contracts(Src(), None, st, "SPY", [fixed], sig, 0, widen=3)
    assert {c.strike for c in f0[D]} == {505.0}, "an explicitly fixed strike is never widened"


def test_adjusted_contracts_and_constructed_identifiers_are_distinguished(tmp_path):
    from strata.data.providers import sources as S
    std, adj = S.contract_from_occ("SPY240306C00500000", source="provider listing"), S.contract_from_occ("SPY1240306C00500000")
    assert not std.adjusted and adj.adjusted and adj.root == "SPY1" and adj.source == "constructed" and std.multiplier == 100.0
    st = chain_store(tmp_path, [499, 500, 501])
    key = st.option_dataset("SPY", "trade")
    st.write_contracts(key, pl.DataFrame([{**std.row("SPY")}, {**adj.row("SPY"), "multiplier": 150.0}, {**S.contract_from_occ(occ(501)).row("SPY")}]))
    cs = bt(st).meta["contract_selection"]
    assert cs["contracts_constructed"] == 2 and cs["contracts_adjusted"] == 1 and cs["multipliers"] == {"100": 2, "150": 1}


# ----------------------------------------------------------------------------- bulk imports
def _quotes(days, strikes=(495, 500, 505), bad=0):
    rows = []
    for d in days:
        for k in strikes:
            for r in ("C", "P"):
                rows.append({"d": d, "exp": d + timedelta(days=7), "k": float(k), "r": r, "bid": 1.0 + k / 1000, "ask": 1.2 + k / 1000})
    for i in range(bad):
        rows[i]["bid"], rows[i]["ask"] = 2.0, 1.0  # crossed
    return rows


def test_vendor_layouts_are_recognised_from_their_columns(tmp_path):
    from strata.data import bulk
    days = cal.trading_days(date(2024, 1, 2), date(2024, 1, 5))
    q = _quotes(days)
    files = {}
    pl.DataFrame({"[QUOTE_UNIXTIME]": [et_epoch(r["d"], "16:00") for r in q if r["r"] == "C"], " [EXPIRE_DATE]": [str(r["exp"]) for r in q if r["r"] == "C"], " [STRIKE]": [r["k"] for r in q if r["r"] == "C"],
                  " [C_BID]": [r["bid"] for r in q if r["r"] == "C"], " [C_ASK]": [r["ask"] for r in q if r["r"] == "C"], " [P_BID]": [r["bid"] for r in q if r["r"] == "C"], " [P_ASK]": [r["ask"] for r in q if r["r"] == "C"],
                  " [UNDERLYING_LAST]": [500.0] * (len(q) // 2)}).write_csv(tmp_path / "optionsdx.csv")
    files["optionsdx.csv"] = ("optionsdx", "option_quotes")
    pl.DataFrame({"underlying_symbol": ["SPY"] * len(q), "quote_date": [str(r["d"]) for r in q], "root": ["SPY"] * len(q), "expiration": [str(r["exp"]) for r in q], "strike": [r["k"] for r in q],
                  "option_type": [r["r"] for r in q], "bid_1545": [r["bid"] for r in q], "ask_1545": [r["ask"] for r in q], "bid_eod": [r["bid"] for r in q], "ask_eod": [r["ask"] for r in q]}).write_csv(tmp_path / "cboe.csv")
    files["cboe.csv"] = ("cboe_eod", "option_quotes")
    pl.DataFrame({"underlying": ["SPY"] * len(q), "underlying_last": [500.0] * len(q), "exchange": ["*"] * len(q), "optionroot": ["x"] * len(q), "optionext": [""] * len(q), "type": ["call" if r["r"] == "C" else "put" for r in q],
                  "expiration": [r["exp"].strftime("%m/%d/%Y") for r in q], "quotedate": [r["d"].strftime("%m/%d/%Y") for r in q], "strike": [r["k"] for r in q], "last": [1.1] * len(q), "bid": [r["bid"] for r in q],
                  "ask": [r["ask"] for r in q], "volume": [10] * len(q), "openinterest": [100] * len(q), "impliedvol": [0.2] * len(q), "delta": [0.5] * len(q)}).write_csv(tmp_path / "hod.csv")
    files["hod.csv"] = ("historicaloptiondata", "option_quotes")
    ns = [et_epoch(D, "10:00") * 10**9 + i * 60 * 10**9 for i in range(4)]
    pl.DataFrame({"ticker": ["O:" + occ(500)] * 4, "volume": [5, 6, 7, 8], "open": [2.0] * 4, "close": [2.1] * 4, "high": [2.2] * 4, "low": [1.9] * 4, "window_start": ns, "transactions": [1] * 4}).write_csv(tmp_path / "poly_opt.csv")
    files["poly_opt.csv"] = ("polygon_options", "option_trades")
    pl.DataFrame({"ticker": ["SPY"] * 4, "volume": [5, 6, 7, 8], "open": [500.0] * 4, "close": [500.1] * 4, "high": [500.2] * 4, "low": [499.9] * 4, "window_start": ns, "transactions": [1] * 4}).write_csv(tmp_path / "poly_stk.csv")
    files["poly_stk.csv"] = ("polygon_stocks", "underlying_bars")
    pl.DataFrame({"ts_event": [datetime.fromtimestamp(n / 1e9, tz=timezone.utc).isoformat() for n in ns], "rtype": [33] * 4, "publisher_id": [1] * 4, "instrument_id": [9] * 4, "open": [2.0] * 4, "high": [2.2] * 4, "low": [1.9] * 4,
                  "close": [2.1] * 4, "volume": [5] * 4, "symbol": ["SPY   240306C00500000"] * 4}).write_csv(tmp_path / "dbn.csv")
    files["dbn.csv"] = ("databento_ohlcv", "option_trades")
    (tmp_path / "stooq.txt").write_text("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\nSPY.US,D,20240102,000000,470,473,468,472,1000,0\nSPY.US,D,20240103,000000,472,474,470,471,1100,0\n")
    files["stooq.txt"] = ("stooq", "underlying_bars")
    (tmp_path / "kibot.txt").write_text("01/02/2024,09:30,470.1,470.5,469.9,470.3,1200\n01/02/2024,09:31,470.3,470.6,470.2,470.4,900\n")
    files["kibot.txt"] = ("headerless_bars", "underlying_bars")
    (tmp_path / "notes.csv").write_text("name,comment\na,b\n")
    info = bulk.inspect(str(tmp_path))
    got = {f["path"].rsplit("/", 1)[1]: (f["format"], f["kind"]) for f in info["files"]}
    assert got == files, got
    assert [u["path"].rsplit("/", 1)[1] for u in info["unrecognised"]] == ["notes.csv"] and info["kinds"] == ["option_quotes", "option_trades", "underlying_bars"]


def test_quotes_stream_in_bounded_memory_from_a_zip_and_bad_rows_are_counted_not_repaired(tmp_path, monkeypatch):
    from strata.data import bulk
    days = cal.trading_days(date(2023, 1, 3), date(2023, 4, 28))
    q = _quotes(days, strikes=tuple(range(400, 601, 2)), bad=7)
    half = len(days) // 2
    for name, part in (("a_2023_q1.csv", [r for r in q if r["d"] in days[:half]]), ("b_2023_q2.csv", [r for r in q if r["d"] in days[half:]])):
        pl.DataFrame({"underlying": ["SPY"] * len(part), "optionroot": ["x"] * len(part), "type": ["call" if r["r"] == "C" else "put" for r in part], "expiration": [r["exp"].strftime("%m/%d/%Y") for r in part],
                      "quotedate": [r["d"].strftime("%m/%d/%Y") for r in part], "strike": [r["k"] for r in part], "bid": [r["bid"] for r in part], "ask": [r["ask"] for r in part], "impliedvol": [20.0] * len(part)}).write_csv(tmp_path / name)
    z = tmp_path / "history.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("a_2023_q1.csv", "b_2023_q2.csv"):
            zf.write(tmp_path / name, name)
            (tmp_path / name).unlink()
    monkeypatch.setattr(bulk, "BATCH", 500)  # 202 rows per session: sessions straddle batches constantly
    st = MarketStore(tmp_path / "market")
    info = bulk.inspect(str(z))
    assert info["files"][0]["archive"] and info["files"][0]["members"] == 2 and info["files"][0]["format"] == "historicaloptiondata"
    rep = bulk.import_path(str(z), provider="vendorfile", store=st)
    assert rep["rows_in"] == len(q) and rep["rows_written"] == len(q) - 7 and rep["sessions"] == len(days) and rep["symbols"] == ["SPY"]
    assert list(rep["rejected"].values()) == [7] and "crossed" in list(rep["rejected"])[0]
    assert rep["peak_rows_in_memory"] < len(q) / 4, f"the importer must never hold the whole file (peak {rep['peak_rows_in_memory']} of {len(q)} rows)"
    got = st.option_days("SPY", "quote", "vendorfile")
    assert got == days and st.read_option_day("SPY", days[1], "quote", "vendorfile").height == 202, "a session split across batches is written whole"
    one = st.read_option_day("SPY", days[-1], "quote", "vendorfile")
    assert one["ts"].dt.convert_time_zone("America/New_York").dt.hour().unique().to_list() == [16] and one["iv"].max() < 1, "end-of-day rows are stamped at the close; percent volatilities become decimals"
    assert st.option_dataset("SPY", "trade") is None, "quotes never create a trade dataset"
    ds = st.option_dataset("SPY", "quote", "vendorfile")
    assert ds.provider == "vendorfile" and str(ds.res).upper() == "EOD"


def test_a_session_that_reappears_later_in_the_input_is_merged_and_trade_bars_stay_trades(tmp_path, monkeypatch):
    from strata.data import bulk
    monkeypatch.setattr(bulk, "BATCH", 6)
    d1, d2 = date(2024, 3, 5), date(2024, 3, 6)
    rows = _quotes([d1], (495, 500)) + _quotes([d2], (495, 500)) + _quotes([d1], (505, 510))  # the first session comes back after the second
    pl.DataFrame({"date": [str(r["d"]) for r in rows], "expiration": [str(r["exp"]) for r in rows], "strike": [r["k"] for r in rows], "right": [r["r"] for r in rows], "bid": [r["bid"] for r in rows],
                  "ask": [r["ask"] for r in rows]}).write_csv(tmp_path / "unsorted.csv")
    st = MarketStore(tmp_path / "market")
    rep = bulk.import_path(str(tmp_path / "unsorted.csv"), symbol="TST", store=st)
    assert rep["sessions"] == 2 and sorted(st.read_option_day("TST", d1)["strike"].unique().to_list()) == [495.0, 500.0, 505.0, 510.0]
    ns = [et_epoch(D, "10:00") * 10**9 + i * 60 * 10**9 for i in range(5)] + [et_epoch(D, "08:00") * 10**9]
    pl.DataFrame({"ticker": ["O:" + occ(500)] * 6, "volume": [5] * 6, "open": [2.0] * 6, "close": [2.1] * 5 + [9.0], "high": [2.2] * 5 + [9.0], "low": [1.9] * 5 + [9.0], "window_start": ns, "transactions": [1] * 6}).write_csv(tmp_path / "trades.csv")
    rep = bulk.import_path(str(tmp_path / "trades.csv"), store=st)
    assert rep["symbols"] == ["SPY"] and rep["rows_written"] == 5 and rep["rejected"] == {"bars outside the regular session": 1}
    ds = st.option_dataset("SPY", "trade")
    assert ds is not None and st.option_dataset("SPY", "quote") is None, "trade bars are stored as trades and never appear as quotes"
    assert st.descriptor(ds).get("complete_chain") is True and st.option_bar_index("SPY")["occ"].to_list() == [occ(500)]
    with pytest.raises(ValueError):
        bulk.import_path(str(tmp_path / "nowhere.csv"), store=st)


# ----------------------------------------------------------------------------- providers
def test_a_source_without_credentials_is_unverified_whatever_its_mocked_tests_say(tmp_settings):
    from strata.data.providers import bench as PB
    r = PB.benchmark_source("alpaca")
    assert r["status"] == "unverified" and r["verified"] is False and r["steps"] == [] and "Mocked tests cover the request and parsing logic only" in r["reason"]
    cat = PB.catalog()
    al = next(s for s in cat["sources"] if s["name"] == "alpaca")
    assert al["verification"] == "unverified" and not al["available"] and al["benchmark"]["status"] == "unverified"
    assert next(s for s in cat["sources"] if s["name"] == "yahoo")["verification"] == "not checked yet"
    # the researched list of other sources now lives in one place, the provider marketplace
    from strata.data.providers import market as MK
    names = {e["name"] for e in MK.ENTRIES}
    assert names >= {"Cboe DataShop", "OptionsDX", "Databento OPRA", "Massive (formerly Polygon.io)"} and all(e["url"].startswith("https://") and e["pricing"]["summary"] for e in MK.ENTRIES) and "external" not in cat


def test_a_live_check_records_depth_missing_bars_and_speed(tmp_settings, monkeypatch):
    from strata.data.providers import bench as PB
    from strata.data.providers import sources as S
    days = cal.trading_days(date(2024, 3, 1), date(2024, 3, 7))

    def chart(self, client, sym, **p):
        if p.get("interval") == "1d":
            ts = [et_epoch(d, "09:30") for d in days]
            return {"timestamp": ts, "indicators": {"quote": [{k: [500.0] * len(ts) for k in ("open", "high", "low", "close")} | {"volume": [1] * len(ts)}], "adjclose": [{"adjclose": [500.0] * len(ts)}]}, "meta": {}}
        step = {"1m": 60, "5m": 300, "60m": 3600}[p["interval"]]
        ts = [et_epoch(d, "09:30") + i * step for d in days for i in range(int(390 * 60 / step))]
        if p["interval"] == "5m":
            ts = ts[::2]  # the provider returned half of the bars
        return {"timestamp": ts, "indicators": {"quote": [{k: [500.0] * len(ts) for k in ("open", "high", "low", "close")} | {"volume": [10] * len(ts)}]}, "meta": {}}
    monkeypatch.setattr(S.Yahoo, "_chart", chart)
    r = PB.benchmark_source("yahoo")
    steps = {s["name"]: s for s in r["steps"]}
    assert r["status"] == "verified" and r["requests_failed"] == 0 and r["bars_per_second"] > 0
    assert steps["daily history"]["first"] == "2024-03-01" and steps["1-minute bars"]["missing_pct"] == 0 and steps["1-minute bars"]["sessions"] == len(days)
    assert 45 < steps["5-minute bars"]["missing_pct"] < 55, "half of the expected bars were not delivered and that is measured"
    assert {d["dataset"] for d in r["historical_depth"]} >= {"daily bars", "1-minute bars"}
    assert PB.saved()["yahoo"]["measured_at"] and PB.catalog()["sources"][0]["verification"] == "verified"

    def boom(self, client, sym, **p):
        raise S.SourceError("HTTP 429 from query1.finance.yahoo.com")
    monkeypatch.setattr(S.Yahoo, "_chart", boom)
    bad = PB.benchmark_source("yahoo")
    assert bad["status"] == "failed" and not bad["verified"] and all("429" in s["error"] for s in bad["steps"])


# ----------------------------------------------------------------------------- collector
class FakeOpt:
    """A source that serves contracts within two strikes of 500 and records what it is asked for."""
    name, label, tier, rate = "yahoo_options", "fake options", "free", 50.0

    def __init__(self, store, fail_after=None):
        self.store, self.asked, self.fail_after = store, [], fail_after

    def available(self):
        return True, "ok"

    def bars_window(self, today=None):
        return date(2024, 1, 1), date(2030, 1, 1)

    def estimate(self, wanted, sym):
        n = sum(len(v) for v in wanted.values())
        return {"requests": n, "pairs_to_fetch": n, "pairs_cached": 0, "pairs_requested": n}

    def acquire(self, sym, wanted, progress=None, should_cancel=None, force=False):
        from strata.data.providers import sources as S
        idx = self.store.option_bar_index(sym, provider=self.name)
        have = set() if idx.is_empty() else set(zip(idx["date"].to_list(), idx["occ"].to_list()))
        rep = {"pairs_requested": sum(len(v) for v in wanted.values()), "pairs_cached": 0, "pairs_ok": 0, "pairs_empty": 0, "pairs_unavailable": 0, "pairs_failed": 0, "bars": 0, "requests": 0, "rejected": {}, "errors": []}
        for d, cs in sorted(wanted.items()):
            rows, index = [], []
            for c in cs:
                if (d, c.occ) in have:
                    rep["pairs_cached"] += 1
                    continue
                if self.fail_after is not None and rep["requests"] >= self.fail_after:
                    rep["pairs_failed"] += 1
                    rep["errors"].append("HTTP 503")
                    continue
                rep["requests"] += 1
                self.asked.append((d, c.occ))
                if abs(c.strike - 500) <= 2:
                    rows.append({"ts": datetime.fromtimestamp(et_epoch(d, "10:00"), tz=timezone.utc), "occ": c.occ, "expiration": c.expiration, "strike": c.strike, "right": c.right, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 3})
                    index.append({"date": d, "occ": c.occ, "expiration": c.expiration, "strike": c.strike, "right": c.right, "bars": 1, "status": "ok", "pages": 1})
                    rep["pairs_ok"] += 1
                    rep["bars"] += 1
                else:
                    index.append({"date": d, "occ": c.occ, "expiration": c.expiration, "strike": c.strike, "right": c.right, "bars": 0, "status": "empty", "pages": 1})
                    rep["pairs_empty"] += 1
            if index:
                self.store.write_option_bars(sym, self.name, pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))) if rows else pl.DataFrame(), index)
        del S
        return rep


class FakeYahoo:
    name, rate = "yahoo", 2.0

    def __init__(self):
        self.calls = []

    def ingest(self, ds, syms, **kw):
        self.calls.append((ds, tuple(syms)))
        return {"symbols": {s: {"rows": 1} for s in syms}}


def collector_env(tmp, monkeypatch, fail_after=None):
    from strata.collector import service as CS
    from strata.data.providers import sources as S
    st = MarketStore(tmp / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf"))
    days = cal.trading_days(date(2024, 2, 20), date(2024, 3, 8))
    st.write_daily("SPY", pl.DataFrame({"date": days, "open": [500.0] * len(days), "high": [501.0] * len(days), "low": [499.0] * len(days), "close": [500.0] * len(days), "volume": [1.0] * len(days)}), provider="yahoo")
    opt, yh = FakeOpt(st, fail_after), FakeYahoo()
    monkeypatch.setattr(S, "get_source", lambda name, store=None: yh)
    monkeypatch.setattr(S, "option_bar_sources", lambda store=None: [opt])
    monkeypatch.setattr(S, "STRIKE_STEP", {"SPY": 1.0})
    CS.save_config({"symbols": ["spy", "ZZZ"], "options": {"dte": [0, 1, 7], "strike_pct": 1.0, "max_strikes_each_side": 4, "rights": ["C", "P"]}, "lookback_sessions": 2, "max_requests_per_cycle": 5000})
    return CS, st, opt, yh


def test_the_collector_asks_only_for_contracts_chosen_from_information_before_each_session(tmp_settings, monkeypatch):
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    cfg = CS.load_config()
    assert cfg["symbols"] == ["SPY", "ZZZ"] and cfg["enabled"] is False, "the collector is off until it is switched on"
    w, info = CS.wanted_contracts(st, "SPY", [date(2024, 3, 5), date(2024, 3, 6)], cfg, today=date(2024, 3, 6))
    cs = w[date(2024, 3, 6)]
    assert {c.strike for c in cs} == {float(k) for k in range(496, 505)} and {c.right for c in cs} == {"C", "P"}
    assert sorted({(c.expiration - date(2024, 3, 6)).days for c in cs}) == [0, 1, 7] and info["contracts"] == sum(len(v) for v in w.values())
    assert all(c.source.startswith("constructed") for c in cs)
    assert [c.strike for c in cs[:2]] == [500.0, 500.0], "strikes nearest the money come first, so a request budget trims the wings"
    none, why = CS.wanted_contracts(st, "ZZZ", [date(2024, 3, 6)], cfg)
    assert none == {} and "no strike grid" in why["skipped"][0]
    with pytest.raises(ValueError):
        CS.save_config({"bar_resolutions": ["2m"]})
    with pytest.raises(ValueError):
        CS.save_config({"place_orders": True})


def test_a_cycle_is_recorded_resumes_and_marks_sessions_it_never_reached_as_missed(tmp_settings, monkeypatch):
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch, fail_after=30)
    today = date(2024, 3, 8)
    monkeypatch.setattr(CS, "datetime", type("dt", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2024, 3, 8, 17, 0, tzinfo=tz))}))
    r1 = CS.run_once(store=st, today=today, trigger="test")
    assert r1["status"] == "partial" and r1["symbols"]["SPY"]["options"]["pairs_failed"] > 0 and any("will be retried next cycle" in e for e in r1["errors"])
    assert ("daily", ("SPY",)) in yh.calls and ("intraday", ("SPY",)) in yh.calls and r1["symbols"]["ZZZ"]["options"]["contracts"] == 0
    first = len(opt.asked)
    opt.fail_after = None
    r2 = CS.run_once(store=st, today=today, trigger="test")
    assert r2["status"] == "done" and r2["symbols"]["SPY"]["options"]["pairs_cached"] == first, "the second cycle skips what the first stored"
    assert len(opt.asked) == r1["symbols"]["SPY"]["options"]["pairs_requested"] == len(set(opt.asked)), "nothing is requested twice"
    r3 = CS.run_once(store=st, today=today, trigger="test")
    assert r3["requests"] == 8 and r3["bars"] == 0, "a third cycle only refreshes the underlying bars (four requests per symbol)"
    hist = CS.history(10)
    assert [h["status"] for h in hist] == ["done", "done", "partial"] and hist[-1]["report"]["bars"] == r1["bars"] and hist[0]["trigger"] == "test"
    cov = CS.coverage(st, CS.load_config(), sessions=10, today=today)
    rows = {r["date"]: r for r in cov["symbols"]["SPY"]}
    assert rows["2024-03-08"]["state"] == rows["2024-03-07"]["state"] == "collected" and rows["2024-03-08"]["contracts_with_trades"] > 0 and rows["2024-03-08"]["contracts_without_trades"] > 0
    assert rows["2024-03-01"]["state"] == "missed" and {m["date"] for m in cov["missed"] if m["symbol"] == "SPY"} >= {"2024-03-01", "2024-03-04"}
    assert all(r["state"] == "not applicable" for r in cov["symbols"]["ZZZ"])
    q = CS.quality(st, CS.load_config())[0]
    assert q["with_trades"] + q["without_trades"] == q["contract_sessions"] and 0 < q["without_trades_pct"] < 100
    state = CS.state(st)
    assert state["safety"].startswith("The collector only reads market data") and state["license"] and state["next_due"] is None and len(state["history"]) == 3


def test_the_request_budget_trims_the_wings_and_a_full_disk_stops_collection_without_deleting(tmp_settings, monkeypatch):
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    CS.save_config({"max_requests_per_cycle": 60, "symbols": ["SPY"]})
    r = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    asked = {float(o[-8:]) / 1000 for _, o in opt.asked}
    assert len(opt.asked) <= 60 and r["symbols"]["SPY"]["options"]["trimmed"] and 500.0 in asked and 504.0 not in asked and any("nearest the money" in n for n in r["notes"])
    before = st.storage_bytes()
    CS.save_config({"storage_limit_gb": 0.5})
    monkeypatch.setattr(MarketStore, "storage_bytes", lambda self, sym=None, key=None: 10**12)
    n = len(opt.asked)
    full = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert full["status"] == "skipped" and len(opt.asked) == n and "never deleted unless clean-up is switched on" in full["notes"][0]
    monkeypatch.undo()
    assert MarketStore(st.root).storage_bytes() == before
    dry = CS.cleanup(st, dry_run=True)
    assert dry["dry_run"] and dry["files"] == 0, "with no retention period configured nothing is eligible"


def test_the_collector_schedule_follows_the_close_and_backs_off_after_a_failure(tmp_settings, monkeypatch):
    from strata.api import db
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    cfg = CS.save_config({"enabled": True, "run_after": "16:20"})["config"]
    ET = cal.ET
    assert CS.next_due({**cfg, "enabled": False}) is None
    # a completed session that no cycle has covered is due at once: waiting for the next close can lose a contract that has expired
    morning, saturday = datetime(2024, 3, 8, 10, 0, tzinfo=ET), datetime(2024, 3, 9, 10, 0, tzinfo=ET)
    assert CS.next_due(cfg, morning) == morning, "just switched on, before the close: yesterday's session has never been collected, so it is due now"
    assert CS.next_due(cfg, saturday) == saturday, "a Saturday with Friday's session uncollected: due now, not after Monday's close"
    late = datetime(2024, 3, 8, 18, 0, tzinfo=ET)
    assert CS.next_due(cfg, late) == late, "after the close with nothing collected yet: due now"
    with db.connect() as con:
        con.execute("INSERT INTO collector_runs (started_at, finished_at, status, trigger, report) VALUES (?,?,?,?,?)", (datetime(2024, 3, 8, 21, 30, tzinfo=timezone.utc).isoformat(), db.now(), "done", "schedule", "{}"))
    monday = datetime(2024, 3, 11, 16, 20, tzinfo=ET)
    assert CS.next_due(cfg, late) == monday, "today's cycle already ran"
    assert CS.next_due(cfg, saturday) == monday, "a Saturday after Friday's cycle: due after Monday's close"
    assert CS.next_due(cfg, datetime(2024, 3, 11, 10, 0, tzinfo=ET)) == monday, "a trading day before the close, with every completed session covered: due after the close"
    tuesday = datetime(2024, 3, 12, 10, 0, tzinfo=ET)
    assert CS.next_due(cfg, tuesday) == tuesday, "the computer was off on Monday evening: that session is due as soon as the collector is running again"

    class Jobs:
        def __init__(self):
            self.sent = []

        def submit(self, rid, kind, payload):
            self.sent.append((kind, payload))
    j = Jobs()
    s = CS.Scheduler(j)
    monkeypatch.setattr(CS, "next_due", lambda cfg, now=None: datetime(2020, 1, 1, tzinfo=ET))
    assert s.tick() and j.sent == [("collector", {"trigger": "schedule"})]
    assert s.tick() is None, "one cycle at a time: the queued run blocks another"
    CS.save_config({"enabled": False})
    for r in db.list_runs(kind="collector"):
        db.update_run(r["id"], status="done")
    assert s.tick() is None, "nothing is scheduled while the collector is switched off"


def test_no_request_can_reach_a_trading_endpoint():
    """The collector, and every source it uses, can only read market data."""
    import inspect
    import re

    import httpx

    from strata.collector import service as CS
    from strata.data.providers import sources as S
    with httpx.Client() as c:
        for url in ("https://paper-api.alpaca.markets/v2/orders", "https://api.alpaca.markets/v2/positions", "https://api.alpaca.markets/v2/account/activities", "https://api.tradier.com/v1/accounts/1/orders"):
            with pytest.raises(S.SourceError, match="never calls a trading endpoint"):
                S.http_get(c, url)
    src = inspect.getsource(S) + inspect.getsource(CS)
    assert not re.search(r"\.(post|put|patch|delete)\(", src), "provider and collector code may only issue GET requests"
    assert not re.search(r"client\.(get|request|send|stream)\(", inspect.getsource(CS)), "the collector makes no request of its own; it goes through the sources"
    assert len(re.findall(r"client\.get\(", inspect.getsource(S))) == 1, "every provider request goes through http_get"
    assert "/v2/orders" not in src and "submit_order" not in src and "place_order" not in src
    assert "never" in CS.__doc__ and "trading endpoint" in CS.__doc__


_ = json
