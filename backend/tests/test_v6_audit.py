"""V6: an audit of execution and accounting.

Three kinds of check, none of which asks the engine to agree with itself:

  reference   a second simulator (`strata.audit.reference`), written from the stated execution
              assumptions without the engine's code, must produce the same trades
  invariants  identities that must hold for ANY data and any strategy of the family:
              cash accounting, fees counted once, no trade that needs the future
  regressions the defects this audit found, pinned
"""
from __future__ import annotations

import math
import os
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hs

from strata.audit import reference as R
from strata.core import calendar as cal
from strata.engine.engine import run_backtest
from tests.helpers import chain, et_epoch, make_store, write_minutes, write_options

D1, D2, D3 = date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)
N = int(os.environ.get("STRATA_AUDIT_EXAMPLES", "40"))  # random cases per property; set it higher for a longer search
STRIKES = [float(k) for k in range(88, 113)]


# ----------------------------------------------------------------------------- a small market, fully specified
def market(tmp, days: list[date], path: dict[date, np.ndarray], snap_every: int = 30, spread: float = 0.06, tv: float = 1.2, with_delta: bool = True, sym: str = "TST"):
    """1-minute bars following `path[d]` (closes), and a 0DTE chain every `snap_every` minutes from the open:
    mid = intrinsic + a time value that decays through the session, bid/ask = mid -/+ spread."""
    st = make_store(tmp, sym)
    write_minutes(st, sym, days, lambda d, i: float(path[d][max(i, 0)]))
    for d in days:
        n = cal.session_minutes(d)
        q = []
        for m in range(0, n, snap_every):
            s = float(path[d][m - 1]) if m > 0 else float(path[d][0])  # the price known at the snapshot: the close of the last completed bar (the open at the open)
            left = 1 - m / n
            hh = f"{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}"
            spec = {}
            for k in STRIKES:
                t_val = tv * math.sqrt(left) * math.exp(-abs(s - k) / 3)
                for right in ("C", "P"):
                    mid = max(0.0, (s - k) if right == "C" else (k - s)) + t_val
                    dl = min(0.99, max(0.01, 0.5 + (s - k) / 8)) * (1 if right == "C" else 1) - (0 if right == "C" else 1)
                    spec[(k, right)] = (round(max(mid - spread, 0.0), 2), round(mid + spread, 2), round(dl, 3) if with_delta else None)
            q += chain(hh, spec, d, underlying=s)
        write_options(st, sym, d, q)
    return st


def walk(seed: int, days: list[date], start: float = 100.0, vol: float = 0.06, drift: float = 0.0) -> dict[date, np.ndarray]:
    rng = np.random.default_rng(seed)
    out, px = {}, start
    for d in days:
        n = cal.session_minutes(d)
        steps = rng.normal(drift, vol, n)
        p = px + np.cumsum(steps)
        out[d] = np.round(np.clip(p, 90.5, 109.5), 2)
        px = float(out[d][-1])
    return out


def engine_cfg(spec: R.Spec, days: list[date], sym: str = "TST") -> dict:
    ex = {"profit_target": {"enabled": spec.profit_target_pct is not None, "mode": "pct", "value": spec.profit_target_pct or 50.0},
          "stop_loss": {"enabled": spec.stop_loss_pct is not None, "mode": "pct", "value": spec.stop_loss_pct or 100.0},
          "time_exit": {"enabled": spec.time_exit is not None, "time": spec.time_exit or "15:55", "dte_lte": None},
          "underlying": {"enabled": spec.underlying_for_pct is not None or spec.underlying_against_pct is not None, "move_for_pct": spec.underlying_for_pct, "move_against_pct": spec.underlying_against_pct}}
    legs = [{"id": f"l{i}", "side": l.side, "right": l.right, "dte": l.dte, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": l.offset, "unit": "points"}} for i, l in enumerate(spec.legs)]
    return {"start": days[0].isoformat(), "end": days[-1].isoformat(), "execution": {"signal_interval_minutes": spec.signal_interval_min, "fill_model": spec.fill, "commission_per_contract": spec.commission},
            "strategies": [{"name": "audit", "symbol": sym, "legs": legs, "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": spec.entry}}]}}, "exits": ex}]}


def both(st, spec: R.Spec, days: list[date]):
    res = run_backtest(engine_cfg(spec, days), store=st, record_paths=False)
    ref = R.simulate([R.load_session(st, "TST", d, "quote") for d in days], spec)
    return res, ref, R.compare(res.trades, ref["trades"])


# ----------------------------------------------------------------------------- the reference agrees with the engine
def test_a_long_call_is_priced_identically_by_hand_by_the_reference_and_by_the_engine(tmp_path):
    """Flat at 100 until 11:00, then 101.2. Buy the 100 call at 10:00: ask = time value + 0.06. Exit when the underlying is 1% above its entry."""
    n = cal.session_minutes(D1)
    path = {D1: np.array([100.0] * 90 + [101.2] * (n - 90))}
    st = market(tmp_path, [D1], path)
    spec = R.Spec(legs=[R.Leg("buy", "C", 0, 0.0)], entry="10:00", underlying_for_pct=1.0, time_exit="15:55")
    res, ref, cmp_ = both(st, spec, [D1])
    assert cmp_["agree"] and cmp_["matched"] == 1, cmp_
    t = ref["trades"][0]
    tv_1000 = 1.2 * math.sqrt(1 - 30 / n)  # the 10:00 snapshot, 30 minutes into the session, at the money
    # The bar that ends at 11:01 is the first to show 101.2, so the move is first known at the 11:05 decision time. The 11:00 quote
    # is then five minutes old (the limit is two), so the close waits for the next fresh quote: the 11:30 snapshot, 1.2 in the money.
    tv_1130 = 1.2 * math.sqrt(1 - 120 / n) * math.exp(-1.2 / 3)
    assert t["legs"][0]["strike"] == 100.0 and t["entry_time"].startswith("2024-03-04T10:00") and t["exit_reason"] == "underlying_target"
    assert t["legs"][0]["entry_price"] == pytest.approx(round(tv_1000 + 0.06, 2)) and t["legs"][0]["exit_price"] == pytest.approx(round(1.2 + tv_1130 - 0.06, 2))
    assert t["fees"] == pytest.approx(1.30) and t["pnl"] == pytest.approx((t["legs"][0]["exit_price"] - t["legs"][0]["entry_price"]) * 100 - 1.30)
    assert t["exit_time"].startswith("2024-03-04T11:30") and res.trades[0]["exit_reason"] == "underlying_target" and "deferred_exit" in res.trades[0]["flags"]


@pytest.mark.parametrize("spec", [
    R.Spec(legs=[R.Leg("buy", "P", 0, 0.0)], entry="open", time_exit="15:30"),
    R.Spec(legs=[R.Leg("sell", "P", 0, -2.0), R.Leg("buy", "P", 0, -5.0)], entry="10:00", profit_target_pct=40.0, stop_loss_pct=150.0),
    R.Spec(legs=[R.Leg("sell", "C", 0, 2.0)], entry="11:00", stop_loss_pct=80.0, time_exit="15:00"),
    R.Spec(legs=[R.Leg("buy", "C", 0, 1.0)], entry="10:30", fill="midpoint", profit_target_pct=30.0, underlying_against_pct=0.4),
    R.Spec(legs=[R.Leg("buy", "C", 0, 0.0), R.Leg("sell", "C", 0, 3.0)], entry="open"),  # held to cash settlement
], ids=["long put, time exit", "put credit spread, target and stop", "short call, stop and time", "long call at the midpoint", "debit spread to expiration"])
def test_the_reference_and_the_engine_agree_on_five_sessions_of_random_prices(tmp_path, spec):
    days = [d for d in cal.trading_days(D1, D1 + timedelta(days=8))][:5]
    st = market(tmp_path, days, walk(11, days, vol=0.09))
    res, ref, cmp_ = both(st, spec, days)
    assert len(res.trades) == 5 and cmp_["agree"], cmp_["differences"]


SPECS = hs.builds(
    lambda side, right, off, entry, pt, sl, uf, ua, te, fill: R.Spec(legs=[R.Leg(side, right, 0, float(off))], entry=entry, profit_target_pct=pt, stop_loss_pct=sl, underlying_for_pct=uf, underlying_against_pct=ua, time_exit=te, fill=fill),
    hs.sampled_from(["buy", "sell"]), hs.sampled_from(["C", "P"]), hs.integers(-3, 3), hs.sampled_from(["open", "10:00", "11:30", "13:00"]), hs.sampled_from([None, 15.0, 40.0, 90.0]), hs.sampled_from([None, 25.0, 60.0, 150.0]),
    hs.sampled_from([None, 0.25, 0.6]), hs.sampled_from([None, 0.25, 0.6]), hs.sampled_from([None, "14:00", "15:30", "15:55"]), hs.sampled_from(["conservative", "midpoint"]))


@settings(max_examples=N, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(spec=SPECS, seed=hs.integers(0, 10_000), vol=hs.sampled_from([0.03, 0.08, 0.15]), snap=hs.sampled_from([5, 15, 30]))
def test_for_any_strategy_of_the_family_and_any_price_path_the_two_simulators_agree(tmp_path_factory, spec, seed, vol, snap):
    days = [D1, D2]
    st = market(tmp_path_factory.mktemp("m"), days, walk(seed, days, vol=vol), snap_every=snap)
    res, ref, cmp_ = both(st, spec, days)
    assert cmp_["agree"], {"differences": cmp_["differences"], "spec": spec, "seed": seed, "vol": vol, "snap": snap}


# ----------------------------------------------------------------------------- accounting invariants
@settings(max_examples=N, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(spec=SPECS, seed=hs.integers(0, 10_000), vol=hs.sampled_from([0.05, 0.12]), capital=hs.sampled_from([25_000.0, 100_000.0]))
def test_cash_fees_and_equity_always_reconcile(tmp_path_factory, spec, seed, vol, capital):
    days = [D1, D2, D3]
    st = market(tmp_path_factory.mktemp("m"), days, walk(seed, days, vol=vol))
    cfg = {**engine_cfg(spec, days), "capital": {"initial_capital": capital}}
    res = run_backtest(cfg, store=st, record_paths=False)
    fees = 0.0
    for t in res.trades:
        # the identity of one trade: what came in, what went out, what was paid
        assert t["pnl"] == pytest.approx(t["entry_cash"] + t["exit_cash"] - t["commissions"], abs=1e-6)
        assert t["gross_pnl"] == pytest.approx(t["pnl"] + t["commissions"], abs=1e-6)
        legs = [l for l in t["legs"] if l["kind"] == "option"]
        # each leg's own profit adds up to the trade's gross profit: nothing is counted twice or dropped
        assert sum(l["pnl"] for l in legs) == pytest.approx(t["gross_pnl"], abs=1e-6)
        for l in legs:
            sign = 1 if l["side"] == "buy" else -1
            assert l["pnl"] == pytest.approx(sign * l["contracts"] * l["multiplier"] * (l["exit_price"] - l["entry_price"]), abs=1e-6)
            assert l["contracts"] == t["original_units"] * abs(l["ratio"]), "no more is ever closed than was opened"
            assert l["entry_price"] >= 0 and l["exit_price"] >= 0
            if l["price_source"] == "quote" and spec.fill == "conservative":
                assert l["entry_price"] == pytest.approx(l["entry_ask"] if sign > 0 else l["entry_bid"]), "a buy pays the ask, a sale receives the bid"
        # commissions: one per contract per fill; a leg that settles at expiration pays none on the way out
        fills = sum(l["contracts"] * (1 + (0 if l["exit_reason"] in ("cash_settled", "expired_worthless") else 1)) for l in legs)
        assert t["commissions"] == pytest.approx(0.65 * fills, abs=1e-6), (t["commissions"], fills, [l["exit_reason"] for l in legs])
        assert t["entry_ts"] >= t["signal_ts"] and t["exit_ts"] > t["entry_ts"], "a fill never precedes its signal and an exit never precedes its entry"
        fees += t["commissions"]
    # the account: every dollar of equity is the starting capital plus realised profit (nothing is left open at the end)
    final = float(res.daily["equity"][-1])
    assert final == pytest.approx(capital + sum(t["pnl"] for t in res.trades), abs=1e-4)
    assert float(res.daily["cash"][-1]) == pytest.approx(final, abs=1e-4), "with no position open, cash is the whole of equity"
    assert len(res.trades) <= len(days), "one entry per session at most"


@settings(max_examples=max(10, N * 5 // 8), deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(spec=SPECS, seed=hs.integers(0, 10_000), cut=hs.integers(1, 2))
def test_what_happened_on_a_session_never_depends_on_later_sessions(tmp_path_factory, spec, seed, cut):
    """No look-ahead, stated as a property: replace everything after session `cut` by a different future and the trades up to it do not change."""
    days = [D1, D2, D3]
    past = walk(seed, days)
    other = {**past, **{d: v for d, v in walk(seed + 7919, days, vol=0.2).items() if d in days[cut:]}}
    a = run_backtest({**engine_cfg(spec, days[:cut])}, store=market(tmp_path_factory.mktemp("a"), days, past), record_paths=False)
    b = run_backtest({**engine_cfg(spec, days[:cut])}, store=market(tmp_path_factory.mktemp("b"), days, other), record_paths=False)
    key = lambda t: (t["entry_time"], t["exit_time"], t["exit_reason"], round(t["pnl"], 6), tuple((l["strike"], l["entry_price"], l["exit_price"]) for l in t["legs"]))
    assert [key(t) for t in a.trades] == [key(t) for t in b.trades]


# ----------------------------------------------------------------------------- the same identities with everything switched on
def market_multi(tmp, days: list[date], path: dict[date, np.ndarray], settlement: str = "cash", snap_every: int = 30, spread: float = 0.06):
    """As `market`, with three expirations listed every session (today, the next session and the one after) and either settlement style."""
    st = make_store(tmp, "TST", settlement=settlement, exercise="european" if settlement == "cash" else "american")
    write_minutes(st, "TST", days, lambda d, i: float(path[d][max(i, 0)]))
    for d in days:
        n = cal.session_minutes(d)
        q = []
        for m in range(0, n, snap_every):
            s = float(path[d][m - 1]) if m > 0 else float(path[d][0])
            hh = f"{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}"
            for ahead in (0, 1, 2):
                exp = cal.next_trading_day(d, ahead) if ahead else d
                spec = {}
                for k in STRIKES:
                    tv = 1.2 * math.sqrt(ahead + 1 - m / n) * math.exp(-abs(s - k) / (3 + 2 * ahead))
                    for right in ("C", "P"):
                        mid = max(0.0, (s - k) if right == "C" else (k - s)) + tv
                        spec[(k, right)] = (round(max(mid - spread, 0.0), 2), round(mid + spread, 2), round(min(0.99, max(0.01, 0.5 + (s - k) / 8)) - (0 if right == "C" else 1), 3))
                q += chain(hh, spec, exp, underlying=s)
        write_options(st, "TST", d, q)
    return st


STRUCTURES = {"short put": [("sell", "P", 1.0)], "long call": [("buy", "C", 0.0)], "put credit spread": [("sell", "P", -1.0), ("buy", "P", -4.0)], "short strangle": [("sell", "P", -3.0), ("sell", "C", 3.0)],
              "long straddle": [("buy", "C", 0.0), ("buy", "P", 0.0)]}
FULL = hs.fixed_dictionaries({"structure": hs.sampled_from(sorted(STRUCTURES)), "dte": hs.sampled_from([0, 1, 2]), "contracts": hs.sampled_from([1, 2, 4]), "max_open": hs.sampled_from([1, 3]), "staged": hs.booleans(),
                              "hedged": hs.booleans(), "target": hs.sampled_from([None, 20.0, 60.0]), "stop": hs.sampled_from([None, 50.0, 150.0]), "entry": hs.sampled_from(["open", "10:00", "13:00"])})


def full_strategy(o: dict, name: str) -> dict:
    legs = [{"id": f"l{i}", "side": side, "right": right, "dte": o["dte"], "dte_mode": "nearest", "strike": {"method": "atm_offset", "value": off, "unit": "points"}} for i, (side, right, off) in enumerate(STRUCTURES[o["structure"]])]
    exits = {"profit_target": {"enabled": o["target"] is not None, "mode": "pct", "value": o["target"] or 50.0}, "stop_loss": {"enabled": o["stop"] is not None, "mode": "pct", "value": o["stop"] or 100.0}}
    if o["staged"]:
        exits["stages"] = [{"name": "half", "profit_pct": 8.0, "close_pct": 50.0, "then_stop": "breakeven"}]
    return {"name": name, "symbol": "TST", "legs": legs, "sizing": {"method": "fixed", "contracts": o["contracts"]}, "exits": exits, "hedge": {"enabled": o["hedged"], "rebalance_minutes": 60, "band": 10},
            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": o["entry"]}}]}, "max_open_positions": o["max_open"]}}


@settings(max_examples=N, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(a=FULL, b=hs.one_of(hs.none(), FULL), seed=hs.integers(0, 10_000), settlement=hs.sampled_from(["cash", "physical"]), capital=hs.sampled_from([30_000.0, 100_000.0]), vol=hs.sampled_from([0.06, 0.14]))
def test_the_books_balance_with_overlapping_positions_scaling_hedging_assignment_and_two_strategies(tmp_path_factory, a, b, seed, settlement, capital, vol):
    """Two strategies sharing one account, positions held over several sessions and overlapping, contracts sold in two stages, shares
    traded as a delta hedge, options exercised and assigned into stock. Whatever happens, every dollar must be accounted for."""
    days = cal.trading_days(D1, D1 + timedelta(days=8))[:6]
    st = market_multi(tmp_path_factory.mktemp("f"), days, walk(seed, days, vol=vol), settlement)
    strategies = [full_strategy(a, "first")] + ([full_strategy(b, "second")] if b else [])
    res = run_backtest({"start": days[0].isoformat(), "end": days[-1].isoformat(), "capital": {"initial_capital": capital}, "strategies": strategies}, store=st, record_paths=False)
    for t in res.trades:
        assert t["pnl"] == pytest.approx(t["entry_cash"] + t["exit_cash"] - t["commissions"], abs=1e-6)
        opts = [l for l in t["legs"] if l["kind"] == "option"]
        o = (a, b)[t["strategy_idx"]]
        assert t["original_units"] == o["contracts"] and 1 <= t["units"] <= t["original_units"]
        staged = sum(int(x.get("units", 0)) for x in t["stage_exits"])
        assert staged + t["units"] == t["original_units"], "what was sold in stages plus what was closed at the end is exactly what was opened"
        if not o["staged"]:
            assert t["stage_exits"] == []
        for l in opts:
            assert l["contracts"] == t["original_units"] * abs(l["ratio"]), "every contract opened is closed once, none twice"
            assert l["entry_price"] >= 0 and l["exit_price"] >= 0 and math.isfinite(l["exit_price"])
        if not {"exercised", "assigned", "early_assignment"} & set(t["flags"]):
            # with no stock delivered, the legs' own results (the hedge's net cash included) are the whole of the trade's gross result
            assert sum(l["pnl"] or 0.0 for l in t["legs"]) == pytest.approx(t["gross_pnl"], abs=1e-4), (t["flags"], [(l["id"], l["pnl"]) for l in t["legs"]])
        if "delta_hedged" in t["flags"]:
            assert o["hedged"] and t["hedge"]["shares"] == 0, "a hedge never outlives the options it was held against"
        else:
            assert t["hedge"] is None
        assert t["exit_ts"] >= t["entry_ts"] >= t["signal_ts"]
    final = float(res.daily["equity"][-1])
    assert final == pytest.approx(capital + sum(t["pnl"] for t in res.trades), abs=1e-3), "equity is the starting capital plus the result of every trade, to the cent"
    assert float(res.daily["cash"][-1]) == pytest.approx(final, abs=1e-3) and int(res.daily["open_positions"][-1]) == 0
    assert float(res.daily["commissions"][-1]) == pytest.approx(sum(t["commissions"] for t in res.trades), abs=1e-6), "fees are counted once"
    assert all(math.isfinite(x) for x in res.daily["equity"].to_list()) and min(res.daily["bp_used"].to_list()) >= -1e-9
    assert int(res.daily["positions_max"].max()) <= a["max_open"] + (b["max_open"] if b else 0)
    # (a hedge buys or shorts up to 100 shares a contract, which can genuinely exhaust the account: only unhedged cases are asserted)
    if capital >= 100_000 and all(STRUCTURES[x["structure"]] != STRUCTURES["short strangle"] and not x["hedged"] for x in (a, b) if x):
        assert res.diagnostics["margin_calls"] == 0, "with ample capital nothing here is liquidated for margin"


# ----------------------------------------------------------------------------- regressions found by this audit
def test_day_level_rules_and_settlement_use_the_official_close_even_when_intraday_bars_end_far_from_it(tmp_path):
    """SPY, 2025-04-09: the stored hourly bars ended at 542.84 and the official close was 548.62 (1.06% apart). The engine
    only trusted an official close within 1% of the last bar, so the next session's "fell 4% yesterday" was computed from
    542.84 and did not fire. Here: the bars of day 2 end at 95.00, the official close is 93.80 (-6.2% on the day)."""
    st = make_store(tmp_path)
    n = cal.session_minutes(D1)
    write_minutes(st, "TST", [D1, D2, D3], lambda d, i: 100.0 if d == D1 else 95.0 if d == D2 else 94.0)
    daily = st.read_daily("TST")
    assert daily.filter(pl.col("date") == D2)["close"][0] == 95.0
    st.write_daily("TST", pl.DataFrame({"date": [D2], "open": [95.0], "high": [95.0], "low": [93.8], "close": [93.8], "volume": [1e6]}))  # the closing auction printed lower than the last bar
    for d in (D2, D3):
        write_options(st, "TST", d, chain("10:00", {(94.0, "P"): (0.50, 0.60, -0.4), (94.5, "P"): (0.70, 0.80, -0.45), (94.0, "C"): (0.9, 1.0, 0.5), (95.0, "C"): (0.4, 0.5, 0.45), (95.0, "P"): (0.9, 1.0, -0.5)}, d, underlying=95.0))
    rule = {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}, {"kind": "cond", "type": "expr", "params": {"expr": "previous_session.return_pct <= -6"}}]}
    strat = {"name": "after a 6% fall", "symbol": "TST", "legs": [{"id": "c", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0.0}}], "entry": {"conditions": rule},
             "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}}}
    res = run_backtest({"start": D1.isoformat(), "end": D3.isoformat(), "strategies": [strat]}, store=st)
    assert [t["entry_date"] for t in res.trades] == [D3.isoformat()], "the official close was 93.80, a fall of 6.2%: the rule fires on the next session"
    gaps = res.meta["data_quality"]["official_close_gaps"]
    assert gaps == [{"symbol": "TST", "date": D2.isoformat(), "last_intraday_bar": 95.0, "official_close": 93.8, "difference_pct": 1.279}]
    assert any("official close" in w for w in res.diagnostics["warnings"])
    # settlement: a 94.5 put expiring on day 2 is 0.70 in the money at the official close, and out of the money at the last bar
    put = {"name": "short put", "symbol": "TST", "legs": [{"id": "p", "side": "sell", "right": "P", "dte": 0, "dte_mode": "exact", "strike": {"method": "fixed", "value": 94.5}}],
           "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}}, "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}}}
    t = run_backtest({"start": D2.isoformat(), "end": D2.isoformat(), "strategies": [put]}, store=st).trades[0]
    assert t["legs"][0]["exit_price"] == pytest.approx(0.70) and t["pnl"] == pytest.approx((0.70 - 0.70) * 100 - 0.65), "sold at the 0.70 bid, settled 0.70 in the money, one commission"
    # a split-sized difference is a different price scale, not a late print: the official close is not forced onto those bars
    st.write_daily("TST", pl.DataFrame({"date": [D2], "open": [47.5], "high": [47.5], "low": [46.9], "close": [47.5], "volume": [1e6]}))
    from strata.engine.market import UnderlyingSeries
    ser = UnderlyingSeries(st.read_best_bars("TST"), official=st.official_closes("TST"))
    assert ser.scale_mismatches == 1 and ser.session_close(D2) == 95.0 and ser.close_gaps == []
    assert n == 390


def test_a_position_is_neutral_only_when_its_legs_offset_never_because_its_delta_is_small(tmp_path):
    """The exit "the underlying moved X% in the position's favour" needs the position's direction. It came from an absolute
    threshold on the entry delta (|delta| < 0.05 per unit meant "neutral"), so a far out-of-the-money long call had no
    direction: its target never fired, and its stop fired on a move either way."""
    from types import SimpleNamespace as NS

    from strata.engine.rules import position_direction
    leg = lambda ratio, call, delta, kind="option": NS(kind=kind, ratio=ratio, is_call=call, entry_greeks=(delta, 0.0, 0.0, 0.0), multiplier=100.0)
    pos = lambda *legs: NS(legs=list(legs))
    nan = float("nan")
    assert position_direction(pos(leg(1, True, 0.01)))[1] is False and position_direction(pos(leg(1, True, 0.01)))[0] > 0, "a 1-delta long call is still long"
    assert position_direction(pos(leg(1, False, -0.02)))[0] < 0 and position_direction(pos(leg(-1, False, -0.03)))[0] > 0
    assert position_direction(pos(leg(-1, False, -0.04), leg(1, False, -0.02))) == (pytest.approx(2.0), False), "a far out-of-the-money put credit spread is bullish"
    assert position_direction(pos(leg(1, True, 0.5), leg(1, False, -0.5)))[1] is True, "a straddle is neutral"
    assert position_direction(pos(leg(-1, False, -0.2), leg(1, False, -0.1), leg(-1, True, 0.2), leg(1, True, 0.1)))[1] is True, "an iron condor is neutral"
    assert position_direction(pos(leg(1, True, nan)))[1] is False and position_direction(pos(leg(1, True, nan)))[0] > 0, "an unknown delta: the leg itself says long"
    assert position_direction(pos(leg(1, True, nan), leg(1, False, nan)))[1] is True and position_direction(pos(leg(1, True, 1e-70)))[1] is False
    # in the engine: buy the call 10 points out of the money (delta 0.01 in this chain), the underlying then rises 1.5%
    n = cal.session_minutes(D1)
    path = {D1: np.array([100.0] * 90 + [101.5] * (n - 90))}
    st = market(tmp_path, [D1], path, snap_every=5)
    spec = R.Spec(legs=[R.Leg("buy", "C", 0, 10.0)], entry="10:00", underlying_for_pct=1.0, time_exit="15:55")
    res, ref, cmp_ = both(st, spec, [D1])
    assert res.trades[0]["legs"][0]["strike"] == 110.0 and abs(res.trades[0]["legs"][0]["entry_delta"]) < 0.05
    assert res.trades[0]["exit_reason"] == "underlying_target" and res.trades[0]["exit_time"][11:16] == "11:05" and cmp_["agree"], cmp_
    # and a stop is a move AGAINST the position, not any move: a long put is not stopped out by a fall
    path = {D1: np.array([100.0] * 90 + [98.5] * (n - 90))}
    st = market(tmp_path / "b", [D1], path, snap_every=5)
    spec = R.Spec(legs=[R.Leg("buy", "P", 0, -10.0)], entry="10:00", underlying_for_pct=1.0, underlying_against_pct=1.0, time_exit="15:55")
    res, ref, cmp_ = both(st, spec, [D1])
    assert res.trades[0]["exit_reason"] == "underlying_target" and cmp_["agree"], cmp_


_ = et_epoch
