"""V5 strategy capabilities, each against a tiny hand-calculated dataset: staged partial
exits, ATR-based and trailing underlying exits, expression and ATR sizing, independent
entry systems (calls on one signal, puts on another), portfolio-level conditions and delta
hedging. One long call on a cash-settled index, Monday 2024-03-04:

    10:00  100C 2.00/2.10 (delta 0.50)     100P 2.00/2.10 (delta -0.50)
    11:00  100C 3.00/3.10 (0.60)           100P 1.20/1.30
    12:00  100C 4.00/4.10 (0.70)           100P 0.80/0.90
    13:00  100C 2.50/2.60 (0.55)           100P 1.60/1.70
    15:30  100C 1.00/1.10 (0.30)           100P 2.40/2.50
"""
from datetime import date, timedelta

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.engine.engine import run_backtest
from tests.helpers import chain, make_store, write_minutes, write_options
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)

D1, D2 = date(2024, 3, 4), date(2024, 3, 5)
QUOTES = [("10:00", 2.00, 2.10, 0.50, 2.00, 2.10), ("11:00", 3.00, 3.10, 0.60, 1.20, 1.30), ("12:00", 4.00, 4.10, 0.70, 0.80, 0.90), ("13:00", 2.50, 2.60, 0.55, 1.60, 1.70), ("15:30", 1.00, 1.10, 0.30, 2.40, 2.50)]


def flat(d, i):
    return 100.0


def setup(tmp, price=flat, history=0, tr=2.0):
    st = make_store(tmp)
    if history:  # daily bars before D1 with a constant true range, so ATR is known exactly
        days, d = [], D1 - timedelta(days=1)
        while len(days) < history:
            if cal.is_trading_day(d):
                days.append(d)
            d -= timedelta(days=1)
        days = days[::-1]
        st.write_daily("TST", pl.DataFrame({"date": days, "open": [100.0] * history, "high": [100.0 + tr / 2] * history, "low": [100.0 - tr / 2] * history, "close": [100.0] * history, "volume": [1e6] * history}), provider="test")
    write_minutes(st, "TST", [D1, D2], price)
    q = []
    for t, cb, ca, cd, pb, pa in QUOTES:
        q += chain(t, {(100, "C"): (cb, ca, cd), (100, "P"): (pb, pa, cd - 1.0)}, D1)
    write_options(st, "TST", D1, q)
    write_options(st, "TST", D2, chain("10:00", {(100, "C"): (1.0, 1.1, 0.5)}, D2))
    return st


def call(contracts=4, at="10:00", **over):
    s = {"name": "long call", "symbol": "TST", "legs": [{"id": "c", "side": "buy", "right": "C", "dte": 0, "strike": {"method": "fixed", "value": 100}}],
         "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": at}}]}},
         "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}}, "sizing": {"method": "fixed", "contracts": contracts}}
    for k, v in over.items():
        s[k] = {**s[k], **v} if isinstance(s.get(k), dict) and isinstance(v, dict) else v
    return s


def bt(st, strat, **cfg):
    return run_backtest({"start": D1.isoformat(), "end": D1.isoformat(), "strategies": [strat], "execution": {"commission_per_contract": 0.65, "stock_slippage_bps": 0.0, **cfg.pop("execution", {})}, **cfg}, store=st)


def ramp(points):
    """Piecewise-linear price by minute index from the open: [(minute, price), ...]."""
    def f(d, i):
        i = max(i, 0)
        for (a, pa), (b, pb) in zip(points, points[1:]):
            if a <= i <= b:
                return pa + (pb - pa) * (i - a) / max(1, b - a)
        return points[-1][1]
    return f


# ----------------------------------------------------------------------------- staged exits
def test_a_stage_takes_partial_profit_and_moves_the_stop_to_breakeven(tmp_path):
    s = call(exits={"stages": [{"name": "half", "profit_pct": 40, "close_pct": 50, "then_stop": "breakeven"}]})
    res = bt(setup(tmp_path), s)
    t = res.trades[0]
    # in at 2.10 x 4 = -840. 11:00 bid 3.00: P&L +360 = 42.9% >= 40% -> 2 units out at 3.00 (+600).
    # 15:30 bid 1.00: -840 + 600 + 200 = -40 <= 0 -> breakeven stop closes the other 2 at 1.00.
    assert t["entry_cash"] == pytest.approx(-840.0) and t["original_units"] == 4 and t["units"] == 2
    assert [(x["stage"], x["units"]) for x in t["stage_exits"]] == [("half", 2)] and t["stage_exits"][0]["pnl_pct_then"] == pytest.approx(42.857, abs=1e-3)
    assert t["exit_reason"] == "stop_loss" and t["exit_time"].startswith("2024-03-04T15:30") and "breakeven stop set by exit stage half" in t["exit_explain"][0]["text"]
    assert t["exit_cash"] == pytest.approx(800.0) and t["pnl"] == pytest.approx(-840 + 800 - 8 * 0.65) and "scaled_out" in t["flags"]
    assert t["legs"][0]["contracts"] == 4 and t["legs"][0]["exit_price"] == pytest.approx(2.0), "the leg's exit price is the average of both fills"
    assert any(e["kind"] == "scale_out" for e in res.events)


def test_stages_fire_in_order_and_a_full_exit_rule_always_wins(tmp_path):
    s = call(exits={"stages": [{"name": "a", "profit_pct": 40, "close_pct": 25}, {"name": "b", "when": "position.pnl_pct >= 60", "close_pct": 25, "then_stop_pct": 10}]})
    t = bt(setup(tmp_path), s).trades[0]
    # 11:00: +42.9% -> stage a closes 1 (of 4) at 3.00. 12:00: -840 + 300 + 3 x 400 = +660 = 78.6% -> stage b closes 1 at 4.00.
    # 13:00: -840 + 300 + 400 + 2 x 250 = +360 (42.9%): the tightened 10% stop is not hit. 15:30: -840 + 700 + 200 = +60 (7.1%): not hit either -> held to expiry (worthless at 100).
    assert [(x["stage"], x["units"]) for x in t["stage_exits"]] == [("a", 1), ("b", 1)] and t["units"] == 2 and t["exit_reason"] == "expiration"
    assert t["pnl"] == pytest.approx(-840 + 300 + 400 - 6 * 0.65)
    full = call(exits={"profit_target": {"enabled": True, "value": 40}, "stages": [{"profit_pct": 40, "close_pct": 50}]})
    t2 = bt(setup(tmp_path / "b"), full).trades[0]
    assert t2["exit_reason"] == "profit_target" and t2["stage_exits"] == [] and t2["units"] == 4
    with pytest.raises(ValueError, match="exactly one trigger"):
        bt(setup(tmp_path / "c"), call(exits={"stages": [{"profit_pct": 40, "when": "price > 1"}]}))


# ----------------------------------------------------------------------------- ATR and trailing exits on the underlying
def test_an_atr_target_is_fixed_at_entry_and_fills_at_the_next_option_quote(tmp_path):
    up = ramp([(0, 100.0), (60, 100.0), (120, 103.5), (389, 103.5)])  # flat until 10:30, +3.5 by 11:30
    s = call(1, exits={"underlying": {"enabled": True, "target_atr": 1.5, "atr_period": 14}})
    res = bt(setup(tmp_path, up, history=30), s)
    t = res.trades[0]
    # ATR(14) of the history is exactly 2.00, so the target is 3.00 above the entry price of 100: reached at 11:22.
    assert t["entry_atr"] == pytest.approx(2.0) and t["exit_reason"] == "underlying_target" and "1.5 x ATR(14) 2.00 fixed at entry" in t["exit_explain"][0]["text"]
    assert t["exit_time"].startswith("2024-03-04T12:00") and t["exit_cash"] == pytest.approx(400.0), "the signal is on the underlying; the fill waits for an observed option quote"
    down = ramp([(0, 100.0), (60, 100.0), (120, 96.0), (389, 96.0)])
    t2 = bt(setup(tmp_path / "d", down, history=30), call(1, exits={"underlying": {"enabled": True, "stop_atr": 1.0}})).trades[0]
    assert t2["exit_reason"] == "underlying_stop" and "against the position" in t2["exit_explain"][0]["text"]


def test_an_entry_whose_atr_cannot_be_computed_is_skipped_and_reported(tmp_path):
    res = bt(setup(tmp_path, history=0), call(1, exits={"underlying": {"enabled": True, "target_atr": 1.5}}))
    assert res.trades == [] and any(x["kind"] == "no_atr" and "ATR(14) cannot be computed" in x["reason"] for x in res.diagnostics["skipped_signals"])


def test_a_trailing_threshold_follows_the_best_underlying_price_since_entry(tmp_path):
    path = ramp([(0, 100.0), (30, 100.0), (120, 104.0), (180, 102.5), (389, 102.5)])  # peak 104.00 at 11:30, back to 102.50 by 12:30
    t = bt(setup(tmp_path, path), call(1, exits={"underlying": {"enabled": True, "trail_pct": 1.0}})).trades[0]
    # 1% off the best level of 104.00 is 102.96, crossed at 12:12: the exit fills at the 13:00 quote
    assert t["exit_reason"] == "underlying_stop" and "gave back" in t["exit_explain"][0]["text"] and "104.00" in t["exit_explain"][0]["text"]
    assert t["exit_time"].startswith("2024-03-04T13:00") and t["exit_cash"] == pytest.approx(250.0)


# ----------------------------------------------------------------------------- sizing
def test_sizing_by_expression_and_by_atr_risk(tmp_path):
    t = bt(setup(tmp_path), call(sizing={"method": "expr", "expr": "min(5, 350 / price)"})).trades[0]
    assert t["original_units"] == 3 and t["entry_cash"] == pytest.approx(-630.0)  # floor(350 / 100) = 3
    res = bt(setup(tmp_path / "b"), call(sizing={"method": "expr", "expr": "price / 1000"}))
    assert res.trades == [] and any("below one" in x["reason"] for x in res.diagnostics["skipped_signals"])
    # 1% of 100,000 = 1,000 budget; one unit moves 0.50 delta x 100 x ATR 2.00 = 100 per ATR -> 10 units
    t = bt(setup(tmp_path / "c", history=30), call(sizing={"method": "atr_risk", "pct": 1.0, "atr_multiple": 1.0})).trades[0]
    assert t["original_units"] == 10
    with pytest.raises(ValueError, match="cannot read entry"):
        bt(setup(tmp_path / "d"), call(sizing={"method": "expr", "expr": "position.pnl"}))


# ----------------------------------------------------------------------------- independent entry systems
def _systems():
    put = [{"id": "p", "side": "buy", "right": "P", "dte": 0, "strike": {"method": "fixed", "value": 100}}]
    ex = lambda e: {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": e}}]}  # noqa: E731
    return call(1, entry={"systems": [{"name": "dip", "conditions": ex("from_open_pct <= -0.5"), "text": "buy the dip with a call"},
                                      {"name": "rip", "conditions": ex("from_open_pct >= 0.5"), "legs": put, "text": "fade the rally with a put"}]})


@pytest.mark.parametrize("move,system,right", [(-1.0, "dip", "C"), (1.0, "rip", "P")])
def test_entry_systems_choose_the_position_and_are_recorded_on_the_trade(tmp_path, move, system, right):
    path = ramp([(0, 100.0), (20, 100.0 + move), (389, 100.0 + move)])
    t = bt(setup(tmp_path, path), _systems()).trades[0]
    assert t["system"] == system and [l["right"] for l in t["legs"]] == [right]
    assert any(e.get("system") == system for e in t["entry_explain"])


def test_no_system_firing_means_no_signal_and_system_names_must_be_unique(tmp_path):
    assert bt(setup(tmp_path), _systems()).trades == []
    s = _systems()
    s["entry"]["systems"][1]["name"] = "dip"
    with pytest.raises(ValueError, match="unique"):
        bt(setup(tmp_path / "b"), s)


# ----------------------------------------------------------------------------- portfolio-level conditions
def test_a_portfolio_condition_blocks_an_entry_while_another_position_is_open(tmp_path):
    two = {"max_entries_per_day": 2, "max_open_positions": 2}
    free = call(1, at="10:00,11:00", entry=two)
    assert len(bt(setup(tmp_path), free).trades) == 2
    gated = call(1, at="10:00,11:00", entry=two)
    gated["entry"]["conditions"]["children"].append({"kind": "cond", "type": "expr", "params": {"expr": "portfolio.open_positions < 1"}})
    res = bt(setup(tmp_path / "b"), gated)
    assert len(res.trades) == 1 and res.trades[0]["entry_time"].startswith("2024-03-04T10:00")


def test_portfolio_values_do_not_exist_in_a_market_study(tmp_path):
    from strata.research.study import run_study
    st = setup(tmp_path)
    r = run_study({"symbol": "TST", "entry_time": "open", "condition": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "portfolio.open_positions < 1"}}]}}, st)
    assert r["stats"]["qualifying_sessions"] == 0 and any("portfolio" in k for k in r["excluded_sessions"]), "not evaluable, never read as false"


# ----------------------------------------------------------------------------- delta hedging
def test_a_delta_hedge_trades_shares_to_minus_delta_and_its_cash_is_part_of_the_trade(tmp_path):
    path = ramp([(0, 100.0), (30, 100.0), (150, 104.0), (360, 101.0), (389, 101.0)])  # 100 at 10:00, 104 at 12:00, 101 at 15:30
    s = call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}}, hedge={"enabled": True, "rebalance_minutes": 60, "band": 0})
    res = bt(setup(tmp_path, path), s, execution={"commission_per_contract": 0.0, "stock_commission_per_share": 0.0, "stock_slippage_bps": 0.0})
    t = res.trades[0]
    # at a tick the engine knows the close of the last COMPLETED one-minute bar: the 10:00 price is the 09:59 bar's close
    px = {m: path(D1, m - 1) for m in (30, 90, 150, 210, 360)}  # 10:00, 11:00, 12:00, 13:00, 15:30
    # shares: -50 at 10:00 (delta .50), -60 at 11:00, -70 at 12:00, -55 at 13:00; all 55 bought back at 15:30 with the option
    flows = 50 * px[30] + 10 * px[90] + 10 * px[150] - 15 * px[210] - 55 * px[360]
    # V6: the buy-back that closes the hedge is a hedge trade too. It used to be left out of the record, which then showed
    # only the opening proceeds as the hedge's result and a share count that never returned to zero.
    assert t["hedge"]["trades"] == 5 and t["hedge"]["shares_traded"] == 140 and t["hedge"]["shares"] == 0 and "delta_hedged" in t["flags"]
    assert t["hedge"]["cash"] == pytest.approx(flows, abs=1e-6) and next(l for l in t["legs"] if l["id"] == "_hedge")["pnl"] == pytest.approx(flows, abs=1e-6)
    assert t["exit_reason"] == "time_exit" and t["pnl"] == pytest.approx(-210 + 100 + flows, abs=1e-6), "option -110 plus every share trade at the price of its tick"
    hedges = [e for e in res.events if e["kind"] == "hedge"]
    assert [("sold" in e["msg"], e["msg"].split("hedge now ")[1].split(" ")[0]) for e in hedges] == [(True, "-50"), (True, "-60"), (True, "-70"), (False, "-55")]
    unhedged = bt(setup(tmp_path / "u", path), call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}}), execution={"commission_per_contract": 0.0}).trades[0]
    assert unhedged["pnl"] == pytest.approx(-110.0) and unhedged["hedge"] is None


# ----------------------------------------------------------------------------- capabilities that already existed keep their semantics
def test_cross_symbol_confirmation_rolls_and_multi_day_holds_still_work(tmp_path):
    from strata.data.store import SymbolMeta
    st = setup(tmp_path)
    st.upsert_symbol(SymbolMeta(symbol="OTH", kind="etf"))
    write_minutes(st, "OTH", [D1, D2], ramp([(0, 50.0), (20, 50.6), (389, 50.6)]))
    s = call(1)
    s["entry"]["conditions"]["children"].append({"kind": "cond", "type": "expr", "params": {"expr": "OTH.from_open_pct >= 1"}})
    t = bt(st, s).trades[0]
    assert any("OTH" in e["text"] for e in t["entry_explain"]), "another symbol's value confirmed the entry"
    s["entry"]["conditions"]["children"][-1]["params"]["expr"] = "OTH.from_open_pct >= 5"
    assert bt(st, s).trades == []


# ----------------------------------------------------------------------------- execution fidelity
def test_auction_windows_block_entries_and_charge_exits_that_fill_inside_them(tmp_path):
    st = setup(tmp_path)
    late = call(1, at="15:30", exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}})
    assert len(bt(st, late).trades) == 1
    res = bt(st, late, execution={"no_entry_last_minutes": 45})
    assert res.trades == [] and any(x["kind"] == "auction_window" for x in res.diagnostics["skipped_signals"]), "an entry inside the closing window is not filled"
    # an exit is never blocked, it is charged: time exit at 15:30 sells at the 1.00 bid, 5% worse inside the window
    s = call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}})
    plain = bt(st, s).trades[0]
    worse = bt(st, s, execution={"no_entry_last_minutes": 45, "auction_extra_pct": 5.0}).trades[0]
    assert plain["exit_cash"] == pytest.approx(100.0) and worse["exit_cash"] == pytest.approx(95.0) and worse["entry_cash"] == plain["entry_cash"]


def test_expiration_day_fills_are_worse_only_in_the_final_stretch_of_the_contracts_last_day(tmp_path):
    st = setup(tmp_path)
    s = call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}})
    t = bt(st, s, execution={"expiry_day_extra_pct": 10.0, "expiry_day_minutes": 60}).trades[0]
    assert t["entry_cash"] == pytest.approx(-210.0) and t["exit_cash"] == pytest.approx(90.0), "the 10:00 entry is untouched; the 15:30 exit, 30 minutes before expiry, is 10% worse"
    early = call(1, exits={"time_exit": {"enabled": True, "time": "13:00"}})
    assert bt(st, early, execution={"expiry_day_extra_pct": 10.0}).trades[0]["exit_cash"] == pytest.approx(250.0)


def test_the_stress_ladder_covers_auctions_expiry_assignment_and_the_three_cases(tmp_settings):
    from strata.research import execution as EX
    tmp_path = tmp_settings
    setup(tmp_path)
    cfg = {"start": D1.isoformat(), "end": D1.isoformat(), "strategies": [call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}})], "execution": {"commission_per_contract": 0.65}}
    sc = {s["id"]: s for s in EX.build_scenarios(cfg)}
    assert {"best", "base", "signal", "impatient", "auction", "expiry5", "expiry10", "adverse"} <= set(sc)
    assert "skipped" in sc["assign"] and "no short option" in sc["assign"]["skipped"], "assignment is reported as not applicable rather than left out"
    short = {**cfg, "strategies": [{**cfg["strategies"][0], "legs": [{"id": "c", "side": "sell", "right": "C", "dte": 0, "strike": {"method": "fixed", "value": 100}}]}]}
    assert {"assign_early", "assign_never", "settle"} <= {s["id"] for s in EX.build_scenarios(short)}
    assert "skipped" in {s["id"]: s for s in EX.build_scenarios(cfg, eod=True)}["auction"]
    out = EX.run_stress(cfg, str(tmp_path), workers=1)
    cases = {c["case"]: c for c in out["summary"]["cases"]}
    # best: mid fills (2.05 in, 1.05 out) = -100 less 1.30 commissions; base: ask in, bid out = -110 less 1.30
    assert cases["best"]["net_pnl"] == pytest.approx(-101.30) and cases["base"]["net_pnl"] == pytest.approx(-111.30)
    # adverse: the order arrives a minute late and no quote exists then, so the trade is missed - reported as missed, not as a better result
    assert cases["adverse"]["trades"] == 0 and next(r for r in out["rows"] if r["id"] == "adverse")["missed"] == 1
    rows = {r["id"]: r for r in out["rows"]}
    assert rows["expiry10"]["net_pnl"] == pytest.approx(-121.30) and rows["signal"]["net_pnl"] == pytest.approx(-100.0)


def test_the_execution_quality_score_grades_evidence_not_profit(tmp_path):
    from strata.research import execution as EX
    res = bt(setup(tmp_path), call(1, exits={"time_exit": {"enabled": True, "time": "15:30"}}))
    cfg = {"pricing": {"mode": "historical"}, "execution": {}}
    q = EX.quality(res.trades, res.meta, res.diagnostics, cfg)
    comps = {c["id"]: c for c in q["components"]}
    assert comps["evidence"]["score"] == 100 and comps["spread"]["measurable"] and comps["spread"]["value"].startswith("median 4.9%"), "2.00 / 2.10 is a 4.9% spread"
    assert comps["coverage"]["value"] == "1 of 1" and comps["liquidity"]["measurable"] and 0 < q["score"] <= 100 and "nothing about whether the strategy made or lost money" in q["notes"][0]
    flipped = [{**t, "pnl": -t["pnl"] * 50} for t in res.trades]
    assert EX.quality(flipped, res.meta, res.diagnostics, cfg)["score"] == q["score"], "profit and loss never enter the score"
    # the same fills described as trade prints: the spread becomes unmeasurable and the evidence weaker, and the report says so
    prints = [{**t, "legs": [{**l, "price_source": "trade_print", "entry_bid": None, "entry_ask": None} for l in t["legs"]], "meta": {**t["meta"], "entry_print_age_seconds": 150, "contract_minutes_traded_pct": 5.0}} for t in res.trades]
    q2 = EX.quality(prints, res.meta, res.diagnostics, {"pricing": {"mode": "trades"}, "execution": {"max_trade_age_seconds": 300}})
    c2 = {c["id"]: c for c in q2["components"]}
    assert c2["evidence"]["score"] == 40 and not c2["spread"]["measurable"] and "not estimated" in c2["spread"]["detail"] and c2["freshness"]["score"] == 50 and q2["score"] < q["score"]
    assert q2["measured_weight"] == 90 and any("could not be measured" in n for n in q2["notes"])
    assert EX.quality([], {}, {}, cfg)["score"] is None


def test_session_vwap_has_no_value_before_the_sessions_first_bar(tmp_path):
    """At the opening tick the last completed bar belongs to the previous session. Its VWAP is not today's VWAP so far:
    the name must be "not evaluable" there, never yesterday's figure."""
    from strata.api import ide as IDE
    st = setup(tmp_path, price=lambda d, i: 110.0 if d == D1 else 100.0)  # D1 trades flat at 110, D2 flat at 100
    r = IDE.debug_session(call(2), D2, st, ["vwap", "price"], None)
    tl = r["timeline"]
    assert tl[0]["time"] == "09:30" and tl[0]["watch"]["vwap"] is None        # no bar of D2 has completed: not D1's 110
    assert tl[1]["time"] == "09:31" and tl[1]["watch"]["vwap"] == pytest.approx(100.0, abs=0.01)
    assert tl[-1]["watch"]["vwap"] == pytest.approx(100.0, abs=0.01)
    # a rule that reads it is "not evaluable" at the open, which is not the same as false
    s = call(2)
    s["entry"]["conditions"]["children"] = [{"kind": "cond", "type": "expr", "params": {"expr": "price < vwap + 5"}}]
    r = IDE.debug_session(s, D2, st)
    first = r["timeline"][0]
    assert first["items"][0]["passed"] is None and "not evaluable" in first["items"][0]["text"] and first["passed"] is False and r["timeline"][1]["passed"] is True


def test_a_debugger_watch_can_read_a_symbol_the_rules_do_not(tmp_path):
    from strata.api import ide as IDE
    st = setup(tmp_path)
    write_minutes(st, "OTH", [D1, D2], lambda d, i: 50.0 + 0.01 * max(i, 0))
    r = IDE.debug_session(call(2), D1, st, ["OTH.price", "OTH.from_open_pct"], "OTH.from_open_pct > 0.3")
    assert r["timeline"][5]["watch"]["OTH.price"] == pytest.approx(50.04, abs=0.011) and r["breakpoint"]["hit"] is True
