"""Provider normalisers (recorded sample payloads), CSV import, research statistics,
analytics and a small end-to-end optimisation on a hand-built dataset."""
from datetime import date

import numpy as np
import polars as pl
import pytest

from strata.data.providers.vendors import Orats, Polygon, ThetaData
from tests.helpers import chain, et_epoch, make_store, write_minutes, write_options

D1, D2 = date(2024, 3, 4), date(2024, 3, 5)


# ---------------------------------------------------------------- providers
def test_thetadata_normalisation():
    quotes = {"header": {"format": ["ms_of_day", "bid_size", "bid_exchange", "bid", "bid_condition", "ask_size", "ask_exchange", "ask", "ask_condition", "date"]},
              "response": [{"contract": {"root": "SPY", "expiration": 20240304, "strike": 500000, "right": "C"},
                            "ticks": [[36000000, 10, 1, 1.25, 0, 12, 1, 1.30, 0, 20240304], [36900000, 0, 0, 0, 0, 0, 0, 0, 0, 20240304]]}]}
    greeks = {"header": {"format": ["ms_of_day", "bid", "ask", "delta", "theta", "vega", "rho", "epsilon", "lambda", "implied_vol", "iv_error", "ms_of_day2", "underlying_price", "date"]},
              "response": [{"contract": {"root": "SPY", "expiration": 20240304, "strike": 500000, "right": "C"},
                            "ticks": [[36000000, 1.25, 1.30, 0.45, -0.3, 0.05, 0, 0, 0, 0.14, 0, 36000000, 499.5, 20240304]]}]}
    df = ThetaData.normalize_options(ThetaData.rows_from(quotes), ThetaData.rows_from(greeks), [], D1)
    assert df.height == 1  # the empty 0/0 quote is dropped, not invented
    r = df.row(0, named=True)
    assert r["strike"] == 500.0 and r["right"] == "C" and r["bid"] == 1.25 and r["delta"] == 0.45 and r["underlying_price"] == 499.5
    assert int(r["ts"].timestamp()) == et_epoch(D1, "10:00")


def test_polygon_snapshot_sampling_keeps_quote_age():
    ct = {"ticker": "O:SPY240304P00495000", "expiration_date": "2024-03-04", "strike_price": 495, "contract_type": "put"}
    q = [{"sip_timestamp": et_epoch(D1, "09:58") * 10**9, "bid_price": 0.5, "ask_price": 0.55, "bid_size": 3, "ask_size": 4},
         {"sip_timestamp": et_epoch(D1, "10:07") * 10**9, "bid_price": 0.45, "ask_price": 0.5, "bid_size": 3, "ask_size": 4}]
    rows = Polygon.normalize_quotes(ct, q, [et_epoch(D1, "09:30"), et_epoch(D1, "10:00"), et_epoch(D1, "10:15")])
    assert [r["ts"] for r in rows] == [et_epoch(D1, "10:00"), et_epoch(D1, "10:15")]  # nothing before the first quote
    assert rows[0]["quote_ts"] == et_epoch(D1, "09:58") and rows[1]["bid"] == 0.45


def test_orats_strike_rows_split_into_calls_and_puts():
    row = {"expirDate": "2024-03-15", "strike": 500, "stockPrice": 501.0, "snapShotEstTime": 1545, "delta": 0.55, "gamma": 0.02, "vega": 0.3, "theta": -0.2,
           "callBidPrice": 6.1, "callAskPrice": 6.2, "putBidPrice": 5.0, "putAskPrice": 5.1, "callMidIv": 0.15, "putMidIv": 0.16,
           "callVolume": 10, "putVolume": 20, "callOpenInterest": 100, "putOpenInterest": 200}
    df = Orats.normalize_strikes([row], D1, 30)
    assert df.height == 2
    p = df.filter(pl.col("right") == "P").row(0, named=True)
    assert p["delta"] == pytest.approx(-0.45) and p["iv"] == pytest.approx(0.16)
    assert int(p["ts"].timestamp()) == et_epoch(D1, "15:45")


# ---------------------------------------------------------------- import
def test_csv_option_import_roundtrip(tmp_path):
    from strata.data.importers import import_file
    from strata.data.store import MarketStore
    st = MarketStore(tmp_path / "m")
    csv = tmp_path / "q.csv"
    csv.write_text("quote_datetime,expiration,strike,option_type,bid,ask,underlying_last\n"
                   "2024-03-04 10:00:00,2024-03-04,500,call,1.2,1.3,500\n"
                   "2024-03-04 10:00:00,2024-03-04,500,put,1.1,1.0,500\n"  # crossed quote: rejected
                   "2024-03-05 10:00:00,2024-03-05,500,P,0.9,1.0,500\n")
    rep = import_file(str(csv), "option_quotes", "IMP", store=st)
    assert rep["rows"] == 2 and rep["rejected"] == 1 and rep["sessions"] == 2
    d = st.read_option_day("IMP", D1)
    assert int(d["ts"][0].timestamp()) == et_epoch(D1, "10:00")  # interpreted as ET wall clock


# ---------------------------------------------------------------- research
def test_research_intraday_move_frequencies(tmp_path):
    from strata.research.queries import Research, wilson
    st = make_store(tmp_path)
    # day 1: +1% by 10:00 then flat; day 2: flat
    write_minutes(st, "TST", [D1, D2], lambda d, i: (100.0 if i < 20 else 101.0) if d == D1 else 100.0)
    r = Research(st).intraday_move("TST", thresholds=(0.5,))
    rows = {x["horizon"]: x for x in r["tables"][0]["rows"]}
    assert rows["15m"]["> 0.5%"]["k"] == 0  # the jump happens in minute 20
    assert rows["30m"]["> 0.5%"]["k"] == 1 and rows["30m"]["n"] == 2
    lo, hi = wilson(1, 2)
    assert rows["30m"]["> 0.5%"]["ci"] == pytest.approx([lo, hi])


def test_metrics_hand_values():
    from strata.analytics.metrics import compute_metrics, max_streak
    daily = pl.DataFrame({"date": [date(2024, 1, d) for d in (2, 3, 4, 5)], "equity": [1000.0, 1100.0, 990.0, 1089.0], "bp_used": [0.0, 100, 0, 0],
                          "open_positions": [0, 1, 0, 0], "benchmark": [1.0, 1.0, 1.0, 1.0], "delta": [0.0] * 4, "gamma": [0.0] * 4, "theta": [0.0] * 4, "vega": [0.0] * 4, "vix": [20.0] * 4})
    trades = [{"pnl": p, "max_risk": 100, "return_on_risk": p / 100, "holding_minutes": 30, "commissions": 1, "slippage": 2, "mfe": 1, "mae": -1, "flags": []} for p in (100, -110, 99)]
    m = compute_metrics(trades, daily, 1000.0)
    assert m["net_pnl"] == pytest.approx(89)
    assert m["max_drawdown"] == pytest.approx(990 / 1100 - 1)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(199 / 110)
    assert max_streak([1, 1, -1, 1, 1, 1], 1) == 3


def test_monte_carlo_is_seeded():
    from strata.analytics.montecarlo import block_bootstrap, trade_bootstrap
    a = trade_bootstrap([10, -5, 3, -2, 8], 1000, 500, seed=3)
    b = trade_bootstrap([10, -5, 3, -2, 8], 1000, 500, seed=3)
    assert a["final_equity"] == b["final_equity"]
    r = np.random.default_rng(0).normal(0.0005, 0.01, 300)
    bb = block_bootstrap(r.tolist(), 1000, 300, block=10, seed=1)
    assert 0 <= bb["prob_loss"] <= 1 and bb["block"] == 10


def test_deflated_sharpe_penalises_many_trials():
    from strata.optimize.overfitting import deflated_sharpe, pbo_cscv
    rng = np.random.default_rng(1)
    r = rng.normal(0.0008, 0.01, 500)
    trials = rng.normal(0, 0.03, 200)
    few = deflated_sharpe(r, 2, trials[:2])
    many = deflated_sharpe(r, 200, trials)
    assert many["dsr"] < few["dsr"]
    M = rng.normal(0, 0.01, (400, 20))
    noise = pbo_cscv(M, 8)["pbo"]
    M[:, 3] += 0.004  # one configuration with a genuine, persistent edge
    edge = pbo_cscv(M, 8)["pbo"]
    assert 0.0 <= edge < 0.05 and edge <= noise <= 1.0


# ---------------------------------------------------------------- optimisation
def test_small_grid_optimisation(tmp_path):
    from strata.config import get_settings, override_settings
    from strata.optimize.optimizer import OptimizationConfig, run_optimization
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    try:
        st = make_store(tmp_path)
        write_minutes(st, "TST", [D1, D2], lambda d, i: 100.0)
        for d in (D1, D2):
            write_options(st, "TST", d, chain("10:00", {(95, "P"): (1.00, 1.10, -0.20), (90, "P"): (0.40, 0.50, -0.10), (85, "P"): (0.10, 0.20, -0.05)}, d)
                          + chain("12:00", {(95, "P"): (0.30, 0.40, -0.1), (90, "P"): (0.10, 0.20, -0.05), (85, "P"): (0.0, 0.10, -0.02)}, d))
        base = {"start": D1.isoformat(), "end": D2.isoformat(), "strategies": [{"name": "s", "symbol": "TST",
                "legs": [{"id": "sp", "side": "sell", "right": "P", "dte": 0, "strike": {"method": "delta", "value": 0.2}},
                         {"id": "lp", "side": "buy", "right": "P", "dte": 0, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}}],
                "entry": {"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}}}]}
        oc = OptimizationConfig.model_validate({"base": base, "params": [{"path": "strategies.0.legs.0.strike.value", "values": [0.1, 0.2]},
                                                                        {"path": "strategies.0.exits.profit_target.value", "values": [25, 50]}],
                                                "method": "grid", "objective": "net_pnl", "min_trades": 1, "validation": "none", "workers": 1})
        out = run_optimization(oc, str(tmp_path))
        assert out["n_configs"] == 4
        best = out["configs"][out["best_index"]]
        assert best["params"]["strategies.0.legs.0.strike.value"] == 0.2  # 0.10-delta leg is the 85/80 spread with no 80 strike -> no trades
        assert any("configurations evaluated" in w["text"] for w in out["warnings"])
    finally:
        override_settings(data_dir=old)
