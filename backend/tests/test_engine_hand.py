"""Engine validation against tiny hand-calculated datasets.

Base scenario (cash-settled index TST, Monday 2024-03-04, 0DTE puts):
  10:00  95P 1.00/1.10 (d -0.20)   90P 0.40/0.50 (d -0.10)
  12:00  95P 0.50/0.60             90P 0.20/0.30
  15:30  95P 0.05/0.10             90P 0.00/0.05
Strategy: sell the 0.20-delta put, buy the put 5 points lower, entered at 10:00.
Conservative entry credit = (1.00 - 0.50) * 100 = $50, commissions $0.65/contract.
"""
from datetime import date

import pytest

from strata.engine.engine import run_backtest
from tests.helpers import chain, make_store, put_spread_strategy, write_minutes, write_options

D1, D2, D3 = date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)


def base_quotes(d=D1, exp=D1, missing_90_at_1530=False, stale=False, bid_size=500):
    extra = {"quote_time": "09:40"} if stale else {}
    q = chain("10:00", {(100, "P"): (3.0, 3.2, -0.5), (95, "P"): (1.00, 1.10, -0.20, {"bid_size": bid_size}), (90, "P"): (0.40, 0.50, -0.10), (85, "P"): (0.10, 0.20, -0.05)}, exp, **extra)
    q += chain("12:00", {(100, "P"): (2.0, 2.2, -0.45), (95, "P"): (0.50, 0.60, -0.15), (90, "P"): (0.20, 0.30, -0.07), (85, "P"): (0.05, 0.10, -0.03)}, exp)
    q += chain("15:30", {(100, "P"): (0.5, 0.6, -0.4), (95, "P"): (0.05, 0.10, -0.05), (90, "P"): None if missing_90_at_1530 else (0.00, 0.05, -0.02), (85, "P"): (0.0, 0.05, -0.01)}, exp)
    q += chain("15:45", {(100, "P"): (0.4, 0.5, -0.4), (95, "P"): (0.04, 0.08, -0.04), (90, "P"): (0.00, 0.05, -0.02), (85, "P"): (0.0, 0.05, -0.01)}, exp)
    return q


def setup(tmp_path, price=lambda d, i: 100.0, **qkw):
    st = make_store(tmp_path)
    write_minutes(st, "TST", [D1, D2], price)
    write_options(st, "TST", D1, base_quotes(**qkw))
    write_options(st, "TST", D2, chain("10:00", {(100, "P"): (1.0, 1.1, -0.5)}, D2))
    return st


def bt(st, strat=None, **cfg):
    c = {"start": D1.isoformat(), "end": D2.isoformat(), "strategies": [strat or put_spread_strategy()], **cfg}
    return run_backtest(c, store=st)


def test_hold_to_expiration_otm(tmp_path):
    res = bt(setup(tmp_path))
    assert len(res.trades) == 1
    t = res.trades[0]
    assert [l["strike"] for l in t["legs"]] == [95.0, 90.0]
    assert t["entry_cash"] == pytest.approx(50.0)
    assert t["commissions"] == pytest.approx(1.30)
    assert t["pnl"] == pytest.approx(48.70)
    assert t["exit_reason"] == "expiration"
    assert res.daily["equity"][-1] == pytest.approx(100_048.70)
    assert t["max_risk"] == pytest.approx(450.0)  # width 500 - credit 50


def test_profit_target_fills_at_snapshot_quotes(tmp_path):
    s = put_spread_strategy(exits={"profit_target": {"enabled": True, "value": 50}, "stop_loss": {"enabled": False}})
    t = bt(setup(tmp_path), s).trades[0]
    # 12:00 cost to close 0.60-0.20=0.40 (20%); 15:30 cost 0.10-0.00=0.10 (80%) -> exit
    assert t["exit_reason"] == "profit_target"
    assert t["exit_time"].startswith("2024-03-04T15:30")
    assert t["pnl"] == pytest.approx(50 - 10 - 2.60)


def test_midpoint_execution_model(tmp_path):
    res = bt(setup(tmp_path), execution={"fill_model": "midpoint"})
    assert res.trades[0]["entry_cash"] == pytest.approx(60.0)
    assert res.trades[0]["pnl"] == pytest.approx(58.70)


def test_cash_settlement_itm(tmp_path):
    st = setup(tmp_path, price=lambda d, i: 100.0 if (d == D1 and i < 300) else 93.0)
    t = bt(st).trades[0]
    # 95P settles at 2.00 intrinsic, 90P worthless -> -$200
    assert t["pnl"] == pytest.approx(50 - 200 - 1.30)
    assert t["legs"][0]["exit_reason"] == "cash_settled"


def test_stop_and_target_same_snapshot_is_resolved_conservatively(tmp_path):
    st = setup(tmp_path, price=lambda d, i: 94.0 if (d == D1 and 330 <= i) else 100.0)
    s = put_spread_strategy(exits={"profit_target": {"enabled": True, "value": 50}, "stop_loss": {"enabled": False},
                                   "underlying": {"enabled": True, "move_against_pct": 5}})
    t = bt(st, s).trades[0]
    assert t["exit_reason"] == "underlying_stop"
    assert "ambiguous_exit" in t["flags"]


def test_missing_quote_defers_exit_instead_of_using_model_price(tmp_path):
    s = put_spread_strategy(exits={"profit_target": {"enabled": True, "value": 50}, "stop_loss": {"enabled": False}})
    res = bt(setup(tmp_path, missing_90_at_1530=True), s)
    t = res.trades[0]
    assert t["exit_time"].startswith("2024-03-04T15:45")
    assert t["pnl"] == pytest.approx(50 - 8 - 2.60)
    assert "missing_quotes" in t["flags"]
    assert res.diagnostics["missing_marks"] >= 1


def test_stale_quotes_rejected_at_entry(tmp_path):
    res = bt(setup(tmp_path, stale=True))
    assert res.trades == []
    assert any("stale_quote" in k for k in res.diagnostics["skipped_entries"]), res.diagnostics["skipped_entries"]


def test_insufficient_buying_power_rejects_entry(tmp_path):
    res = bt(setup(tmp_path), capital={"initial_capital": 300})
    assert res.trades == []
    assert any("buying power" in k for k in res.diagnostics["skipped_entries"])


def test_partial_fill_limited_by_displayed_size(tmp_path):
    s = put_spread_strategy(sizing={"method": "fixed", "contracts": 5})
    t = bt(setup(tmp_path, bid_size=3), s).trades[0]
    assert t["units"] == 3 and "partial_fill" in t["flags"]
    assert t["pnl"] == pytest.approx(3 * 48.70)


def test_physical_assignment_at_expiration(tmp_path):
    st = make_store(tmp_path, settlement="physical", exercise="american")
    write_minutes(st, "TST", [D1, D2], lambda d, i: (100.0 if i < 300 else 93.0) if d == D1 else 92.0)
    write_options(st, "TST", D1, base_quotes())
    write_options(st, "TST", D2, chain("10:00", {(100, "P"): (8.0, 8.2, -0.9)}, D2))
    s = put_spread_strategy(legs=[{"id": "sp", "side": "sell", "right": "P", "dte": 0, "strike": {"method": "fixed", "value": 95}}],
                            entry={"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}, {"kind": "cond", "type": "day_of_week", "params": {"days": ["Mon"]}}]}})
    res = bt(st, s, execution={"stock_slippage_bps": 0})
    t = res.trades[0]
    # +100 credit, assigned 100 sh @95, sold at next open 92 -> -300, commission 0.65
    assert t["pnl"] == pytest.approx(100 - 300 - 0.65)
    assert t["exit_reason"] == "assigned" and "assigned" in t["flags"]
    assert t["exit_time"].startswith("2024-03-05T09:30")


def test_early_assignment_of_deep_itm_short_put(tmp_path):
    st = make_store(tmp_path, settlement="physical", exercise="american")
    write_minutes(st, "TST", [D1, D2, D3], lambda d, i: 95.0)
    q = chain("10:00", {(105, "P"): (9.98, 10.02, -0.99)}, D3, underlying=95.0) + chain("15:30", {(105, "P"): (9.98, 10.02, -0.99)}, D3, underlying=95.0)
    write_options(st, "TST", D1, q)
    write_options(st, "TST", D2, chain("10:00", {(105, "P"): (9.98, 10.02, -0.99)}, D3, underlying=95.0))
    s = put_spread_strategy(legs=[{"id": "sp", "side": "sell", "right": "P", "dte": 2, "strike": {"method": "fixed", "value": 105}}],
                            entry={"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}, {"kind": "cond", "type": "day_of_week", "params": {"days": ["Mon"]}}]}})
    res = run_backtest({"start": D1.isoformat(), "end": D2.isoformat(), "strategies": [s], "execution": {"stock_slippage_bps": 0}}, store=st)
    t = res.trades[0]
    assert "early_assignment" in t["flags"]
    assert t["pnl"] == pytest.approx(998 - 1000 - 0.65)
    assert t["exit_time"].startswith("2024-03-05T09:30")


def test_split_adjusts_open_contracts(tmp_path):
    import polars as pl
    st = make_store(tmp_path)
    write_minutes(st, "TST", [D1, D2], lambda d, i: 100.0 if d == D1 else 50.0)
    write_options(st, "TST", D1, chain("10:00", {(100, "C"): (1.90, 2.00, 0.5)}, D3))
    write_options(st, "TST", D2, chain("10:00", {(50, "C"): (0.80, 0.85, 0.5)}, D3, underlying=50.0) + chain("11:00", {(50, "C"): (1.00, 1.05, 0.5)}, D3, underlying=50.0))
    st.write_corporate_actions("TST", pl.DataFrame({"ex_date": [D2], "action": ["split"], "ratio": [2.0], "amount": [None]}))
    s = put_spread_strategy(legs=[{"id": "lc", "side": "buy", "right": "C", "dte": 2, "strike": {"method": "fixed", "value": 100}}],
                            entry={"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}, {"kind": "cond", "type": "day_of_week", "params": {"days": ["Mon"]}}]}},
                            exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "11:00", "dte_lte": None}})
    t = bt(st, s).trades[0]
    leg = t["legs"][0]
    assert leg["strike"] == 50.0 and leg["contracts"] == 2
    assert "contract_adjusted" in t["flags"]
    assert t["pnl"] == pytest.approx(-200 + 200 - 0.65 - 1.30)


def test_entry_conditions_and_reentry_rules(tmp_path):
    st = setup(tmp_path)
    s = put_spread_strategy(
        entry={"conditions": {"kind": "group", "op": "or", "children": [
            {"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}},
            {"kind": "cond", "type": "entry_times", "params": {"times": "12:00"}}]},
            "max_entries_per_day": 2, "max_open_positions": 1},
        exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "11:00", "dte_lte": None}})
    res = bt(st, s)
    # 10:00 entry, 12:00 time-exit (first snapshot >= 11:00) then re-entry at 12:00 is a new entry
    times = [t["entry_time"][11:16] for t in res.trades]
    assert times == ["10:00", "12:00"]
    s["entry"]["reentry_after"] = ["stop"]
    res2 = bt(st, s)
    assert [t["entry_time"][11:16] for t in res2.trades] == ["10:00"]


def test_explanations_recorded(tmp_path):
    t = bt(setup(tmp_path)).trades[0]
    texts = " | ".join(e["text"] for e in t["entry_explain"])
    assert "10:00" in texts and "95P" in texts and "90P" in texts


def test_accounting_identity(tmp_path):
    s = put_spread_strategy(exits={"profit_target": {"enabled": True, "value": 50}, "stop_loss": {"enabled": False}})
    res = bt(setup(tmp_path), s, capital={"initial_capital": 50_000})
    assert res.daily["equity"][-1] == pytest.approx(50_000 + sum(t["pnl"] for t in res.trades))


def test_expiring_legs_are_never_traded_at_or_after_the_close(tmp_path):
    """A garbage 16:00 quote on the expiring contract must not be used: the leg settles."""
    st = make_store(tmp_path)
    write_minutes(st, "TST", [D1, D2], lambda d, i: 100.0)
    q = base_quotes() + chain("16:00", {(95, "P"): (0.0, 99.0, -0.9), (90, "P"): (0.0, 50.0, -0.9), (100, "P"): (0, 1, -0.5), (85, "P"): (0, 1, 0)}, D1)
    write_options(st, "TST", D1, q)
    write_options(st, "TST", D2, chain("10:00", {(100, "P"): (1.0, 1.1, -0.5)}, D2))
    s = put_spread_strategy(exits={"profit_target": {"enabled": False}, "stop_loss": {"enabled": True, "value": 100}})
    t = bt(st, s).trades[0]
    assert t["pnl"] == pytest.approx(48.70)  # expired worthless, not stopped at the bogus 99.00 ask
    assert t["exit_reason"] == "expiration"
