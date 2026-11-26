"""V7: expanding beyond the stored strike and expiration universe, without substituting a different trade.

Selectively stored trade data holds a band of strikes and a few expirations. When a strategy asks for a contract outside
them, three things must hold: the outermost stored contract is never passed off as the one asked for; the readiness check
and the run say the same thing about it; and the planner asks the provider for exactly what is missing, centred on
information that existed at the decision.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.core.pricing import bs_price_scalar
from strata.data import planner as P
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from strata.engine.engine import run_backtest
from strata.engine.scan import scan_signals
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture: a temporary data directory)

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 3, 22))[:10]
PREV = cal.prev_trading_day(DAYS[0])
D_BAND, D_LOW, D_WEEK = DAYS[0], DAYS[1], DAYS[2]
WEEK = D_WEEK + timedelta(days=7)
SPOT, VOL = 500.0, 0.20


def premium(strike: float, days: float, right: str = "C") -> float:
    """A print the smile-free model would give: enough for the engine to recover a sensible delta from it."""
    return max(0.01, round(bs_price_scalar(SPOT, strike, days / 365.0, 0.04, 0.0, VOL, right == "C"), 2))


def put_chain(st: MarketStore, d: date, exp: date, strikes, right: str = "C", times=("09:59", "10:30", "15:28"), skip: set | None = None):
    frames, index = [], []
    for k in strikes:
        if skip and k in skip:
            continue
        occ = S.occ_symbol("SPY", exp, right, float(k))
        px = premium(float(k), (exp - d).days + 0.25, right)
        rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp, "strike": float(k), "right": right, "open": px, "high": px, "low": px, "close": px, "volume": 5} for t in times]
        frames.append(pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))))
        index.append({"date": d, "occ": occ, "expiration": exp, "strike": float(k), "right": right, "bars": len(rows), "status": "ok", "pages": 1})
    st.write_option_bars("SPY", "testfeed", pl.concat(frames), index)


@pytest.fixture
def store(tmp_settings):
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [PREV] + DAYS, lambda d, i: SPOT)
    put_chain(st, D_BAND, D_BAND, range(495, 506))               # a band of eleven strikes around the money, expiring that day
    put_chain(st, D_LOW, D_LOW, range(495, 501))                 # only the money and below: every stored call is at or in the money
    put_chain(st, D_WEEK, WEEK, range(495, 506))                 # one expiration, a week out
    return st


def cfg(d: date, strike: dict, **leg) -> dict:
    return {"start": d.isoformat(), "end": d.isoformat(), "pricing": {"mode": "trades"}, "data": {"options_provider": "testfeed"},
            "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0},
            "strategies": [{"name": "call at 10:00", "symbol": "SPY",
                            "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": strike, **leg}],
                            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
                            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}]}


def outcome(store, c: dict) -> tuple[list[dict], dict, dict]:
    """(trades, the run's own account of what it did not execute, the readiness verdict for the same signal)."""
    res = run_backtest(c, store=store, record_paths=False)
    rd = P.signal_readiness(store, c, scan_signals(c, store)["signals"], "testfeed")["rows"][0]
    return res.trades, res.meta, rd


# ----------------------------------------------------------------------------- a strike beyond the stored band
def test_a_percent_out_of_the_money_strike_beyond_the_band_is_not_replaced_by_the_outermost_one(store):
    inside, meta, rd = outcome(store, cfg(D_BAND, {"method": "pct_otm", "value": 0.8}))
    assert [t["legs"][0]["strike"] for t in inside] == [504.0] and rd["status"] == "trades", "0.8% out of the money is 504: stored, so it trades"
    # 3% out of the money is 515. The stored strikes stop at 505: before V7 the 505 call was bought and called the 3% one
    trades, meta, rd = outcome(store, cfg(D_BAND, {"method": "pct_otm", "value": 3.0}))
    assert trades == []
    assert meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    why = meta["executability"]["skipped"][0]["reason"] if meta["executability"].get("skipped") else ""
    assert rd["status"] == "contract" and "3% out of the money is strike 515.00, which lies beyond the stored strike range 495-505" in rd["detail"], why
    sel = meta["contract_selection"]
    assert sel["beyond_dates"] == [D_BAND.isoformat()] and sel["beyond_band"] == 1 and sel["at_band_edge"] == 0, "the session is handed to the widening loop although nothing was filled"
    assert "never replaced by the outermost stored one" in sel["note"]


def test_a_fixed_strike_is_that_strike_or_no_trade(store):
    got, _, rd = outcome(store, cfg(D_BAND, {"method": "fixed", "value": 503}))
    assert [t["legs"][0]["strike"] for t in got] == [503.0] and rd["status"] == "trades"
    trades, meta, rd = outcome(store, cfg(D_BAND, {"method": "fixed", "value": 510}))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert rd["status"] == "contract" and "strike 510 lies beyond the stored strike range 495-505" in rd["detail"]
    # stored but not printing: the neighbouring strike is not bought in its place
    put_chain(store, DAYS[3], DAYS[3], range(495, 506), skip={503})
    put_chain(store, DAYS[3], DAYS[3], [503], times=("09:35", "15:29"))  # 503 trades early and late, never near 10:00
    trades, meta, rd = outcome(store, cfg(DAYS[3], {"method": "fixed", "value": 503}))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"no_recent_trade": 1}
    assert rd["status"] == "stale" and "strike 503 has no usable trade print at this time" in rd["detail"]


def test_a_delta_or_premium_target_the_band_cannot_reach_is_not_filled_at_its_edge(store):
    # the whole band is stored: a 30-delta call sits inside it and is chosen by the delta its own print implies
    trades, meta, rd = outcome(store, cfg(D_BAND, {"method": "delta", "value": 0.30}))
    pick = meta["contract_selection"]
    assert len(trades) == 1 and 500 < trades[0]["legs"][0]["strike"] < 505 and pick["delta_error_max"] < 0.10 and pick["beyond_band"] == 0 and rd["status"] == "trades"
    # only strikes at or below the money are stored: the nearest call to 30 delta is the 500 strike at about 50 delta
    trades, meta, rd = outcome(store, cfg(D_LOW, {"method": "delta", "value": 0.30}))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert rd["status"] == "contract" and "|delta| 0.3 (the outermost stored strike 500 has |delta| 0.5" in rd["detail"] and "lies beyond the stored strike range 495-500" in rd["detail"]
    assert meta["contract_selection"]["beyond_dates"] == [D_LOW.isoformat()]
    # a premium of 0.10 is far out of the money: the cheapest stored call (the 500 strike) costs about ten times that
    trades, meta, rd = outcome(store, cfg(D_LOW, {"method": "premium", "value": 0.10}))
    assert trades == [] and rd["status"] == "contract" and "a premium near 0.1 (the outermost stored strike 500 has a last trade at" in rd["detail"]
    # a premium the band does hold is traded, interior or not
    want = premium(497.0, 0.25)
    trades, meta, rd = outcome(store, cfg(D_LOW, {"method": "premium", "value": want}))
    assert [t["legs"][0]["strike"] for t in trades] == [497.0] and rd["status"] == "trades"
    # an edge selection that IS within tolerance of the target stays a trade, and stays flagged for widening
    trades, meta, rd = outcome(store, cfg(D_LOW, {"method": "delta", "value": 0.45}))
    assert [t["legs"][0]["strike"] for t in trades] == [500.0] and meta["contract_selection"]["at_band_edge"] == 1 and meta["contract_selection"]["edge_dates"] == [D_LOW.isoformat()]


def test_a_strike_whose_neighbours_have_not_traded_is_not_filled_somewhere_else(store):
    """Inside the stored band, but the strikes either side of the target have no recent print. Before V7 the nearest strike
    WITH a print was bought, however far away: live, an 8% in-the-money call was filled 49 strikes from its target."""
    d = DAYS[4]
    put_chain(store, d, d, [495, 496, 497, 498], times=("09:35", "15:29"))  # stored, traded early and late, nothing near 10:00
    put_chain(store, d, d, range(499, 506))
    # 0.6% in the money is strike 497: stored on both sides (497 exactly), none of them fresh; the nearest fresh strike is 499
    trades, meta, rd = outcome(store, cfg(d, {"method": "pct_otm", "value": -0.6}))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"no_recent_trade": 1}
    assert rd["status"] == "stale" and "neither stored strike next to -0.6% out of the money (strike 497.00) (497) has a usable trade print at this time (the nearest with one is 499)" in rd["detail"]
    assert rd["downloadable"] is None, "the contract is stored: there is nothing to download, it simply did not trade"
    # a delta the fresh strikes cannot show: 80 delta is around strike 497, the nearest fresh call is the 499 at about 62
    trades, meta, rd = outcome(store, cfg(d, {"method": "delta", "value": 0.85}))
    assert trades == [] and rd["status"] == "stale" and "no contract near |delta| 0.85 has a usable trade print at this time" in rd["detail"]
    # a target that falls between two adjacent stored strikes is the nearer of them even when the miss is large: nothing closer exists
    wide = DAYS[5]
    put_chain(store, wide, wide, [490, 500, 510])  # a ten-point grid: deltas of roughly 1.0, 0.5 and 0.0
    trades, meta, rd = outcome(store, cfg(wide, {"method": "delta", "value": 0.25}))
    assert [t["legs"][0]["strike"] for t in trades] in ([500.0], [510.0]) and rd["status"] == "trades" and meta["contract_selection"]["delta_error_max"] > 0.10


def test_a_far_expiration_is_not_replaced_by_the_nearest_one_that_happens_to_be_stored(store):
    """Only a one-week expiration is stored. A leg that asks for 30 days out was given the one-week contract ("nearest");
    live, every 30DTE signal traded a 7DTE contract until the right expiration had been fetched."""
    atm = {"method": "atm_offset", "value": 0, "unit": "points"}
    trades, meta, rd = outcome(store, cfg(D_WEEK, atm, dte=30, dte_mode="nearest"))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert rd["status"] == "expiration" and "no call expiration within 8 days of 30DTE is stored for the session (stored: 7DTE)" in rd["detail"]
    # within the distance listed expirations are apart, the stored one is used: 5 days out asked, 7 stored
    trades, meta, rd = outcome(store, cfg(D_WEEK, atm, dte=5, dte_mode="nearest"))
    assert len(trades) == 1 and trades[0]["legs"][0]["expiration"] == WEEK.isoformat()
    from strata.engine.selection import dte_tolerance
    assert [dte_tolerance(x) for x in (1, 7, 30, 90, 365)] == [6, 6, 8, 22, 91]


# ----------------------------------------------------------------------------- an exact expiration date
def test_an_exact_expiration_date_is_that_expiration_or_no_trade(store):
    atm = {"method": "atm_offset", "value": 0, "unit": "points"}
    c = cfg(D_WEEK, atm, expiration=WEEK.isoformat())
    trades, meta, rd = outcome(store, c)
    assert len(trades) == 1 and trades[0]["legs"][0]["expiration"] == WEEK.isoformat() and rd["status"] in ("trades", "exit_uncertain")
    assert rd["entry"]["contracts"] == [f"500C {WEEK.isoformat()}"]
    # the day after is not stored: the stored expiration next to it is not used instead
    other = WEEK + timedelta(days=1)
    trades, meta, rd = outcome(store, cfg(D_WEEK, atm, expiration=other.isoformat()))
    assert trades == [] and meta["executability"]["not_executed_by_reason"] == {"contract_not_stored": 1}
    assert rd["status"] == "expiration" and f"the {other} call expiration asked for is not stored for the session (stored: {WEEK})" in rd["detail"]
    # a date that had already passed on the session cannot be traded or downloaded
    gone = D_WEEK - timedelta(days=3)
    trades, meta, rd = outcome(store, cfg(D_WEEK, atm, expiration=gone.isoformat()))
    assert trades == [] and rd["status"] == "no_contract" and "had already expired on this session" in rd["detail"] and rd["downloadable"] is None
    # the holding period of a dated leg is counted from the run's first session, and never past the expiration
    from strata.engine.models import BacktestConfig
    st = BacktestConfig.model_validate({**c, "strategies": [{**c["strategies"][0], "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": False}}}]}).strategies[0]
    assert P.leg_days(st.legs[0], D_WEEK) == 7 and P.hold_sessions(st, D_WEEK) == 5 and P.hold_sessions(st) == 0
    assert P._dte_ok([0, 7, 14], {"expiration": WEEK.isoformat()}, D_WEEK) and not P._dte_ok([0, 14], {"expiration": WEEK.isoformat()}, D_WEEK)


# ----------------------------------------------------------------------------- what the planner asks the provider for
class Listing(S.OptionBarSource):
    """A provider that lists whole strikes for any expiration and records what it was asked."""
    name, label, tier = "testfeed", "Test feed", "free_key"

    def __init__(self, store):
        super().__init__(store)
        self.asked = []

    def available(self):
        return True, "key configured"

    def bars_window(self, today=None):
        return date(2024, 1, 1), date(2026, 12, 31)

    def list_contracts(self, client, sym, as_of, exp_from, exp_to, strike_lo, strike_hi, rights=("C", "P")):
        self.asked.append({"as_of": as_of, "exp": (exp_from, exp_to), "strikes": (round(strike_lo), round(strike_hi)), "rights": rights})
        exps = [e for e in cal.trading_days(exp_from, exp_to) if e.weekday() == 4 or (e - as_of).days <= 7]  # dailies for a week, then Fridays
        # whole strikes near the front; five points apart from two months out, as far expirations are listed
        return [S.contract_from_occ(S.occ_symbol(sym, e, r, float(k)), source="test") for e in exps for r in rights for k in range(int(np.ceil(strike_lo)), int(np.floor(strike_hi)) + 1)
                if (e - as_of).days < 60 or k % 5 == 0]


def leg(strike: dict, dte: int = 0, **kw) -> dict:
    return {"id": "lc", "side": "buy", "right": "C", "dte": dte, "dte_mode": "exact" if dte == 0 else "nearest", "strike": strike, **kw}


def test_the_planner_asks_for_the_contracts_a_signal_needs_and_no_others(store):
    src = Listing(store)
    sig = {"date": D_BAND, "price": 500.0}
    # at the money, same day: three strikes either side, that expiration only
    w, notes = P.trade_contracts(src, None, store, "SPY", [leg({"method": "atm_offset", "value": 0, "unit": "points"})], sig, 0)
    assert notes == [] and sorted(c.strike for c in w[D_BAND]) == [497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0] and {c.expiration for c in w[D_BAND]} == {D_BAND}
    # ten percent out of the money: the band is centred on 550, far outside anything a bulk download holds
    w, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "pct_otm", "value": 10.0})], sig, 0)
    assert sorted(c.strike for c in w[D_BAND]) == [float(k) for k in range(547, 554)]
    # deep in the money by the same rule
    w, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "pct_otm", "value": -8.0})], sig, 0)
    assert min(c.strike for c in w[D_BAND]) == 457.0 and max(c.strike for c in w[D_BAND]) == 463.0
    # ninety days out, nearest expiration: one expiration is chosen among those listed around day 90, and held contracts are needed on later sessions
    w, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "atm_offset", "value": 0, "unit": "points"}, dte=90)], sig, 2)
    exps = {c.expiration for cs in w.values() for c in cs}
    assert len(exps) == 1 and abs((next(iter(exps)) - D_BAND).days - 90) <= 6 and sorted(w) == [D_BAND, cal.next_trading_day(D_BAND, 1), cal.next_trading_day(D_BAND, 2)]
    # that expiration is listed five points apart: the strikes kept are the LISTED ones around the money, not "three points either side" (one strike)
    assert sorted(c.strike for c in w[D_BAND]) == [490.0, 495.0, 500.0, 505.0, 510.0]
    # a fixed strike is that strike alone
    w, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "fixed", "value": 512})], sig, 0)
    assert [c.strike for c in w[D_BAND]] == [512.0]
    # an exact date: that expiration and nothing around it
    on = date(2024, 4, 19)
    src.asked.clear()
    w, notes = P.trade_contracts(src, None, store, "SPY", [leg({"method": "atm_offset", "value": 0, "unit": "points"}, expiration=on.isoformat())], sig, 0)
    assert src.asked[0]["exp"] == (on, on) and {c.expiration for c in w[D_BAND]} == {on} and notes == []
    w, notes = P.trade_contracts(src, None, store, "SPY", [leg({"method": "atm_offset", "value": 0, "unit": "points"}, expiration="2024-02-16")], sig, 0)
    assert w == {} and "had already expired on this session" in notes[0]
    # a wider band only when a selection came back on the edge of the last one
    w1, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "pct_otm", "value": 2.0})], sig, 0, widen=0)
    w2, _ = P.trade_contracts(src, None, store, "SPY", [leg({"method": "pct_otm", "value": 2.0})], sig, 0, widen=1)
    assert len(w2[D_BAND]) > len(w1[D_BAND]) and {c.occ for c in w1[D_BAND]} <= {c.occ for c in w2[D_BAND]}


def test_delta_and_premium_bands_are_centred_for_the_time_to_expiry():
    """The band that is downloaded is centred where the target should sit for the time left, from a volatility level known
    before the session. It was centred a fixed eight strikes from the money whatever the expiration."""
    call = lambda strike, dte: {"id": "lc", "side": "buy", "right": "C", "dte": dte, "strike": strike}
    k0, n0 = P.leg_targets([call({"method": "delta", "value": 0.30}, 0)], 700.0, 0.20, 1.0)["lc"]
    k30, n30 = P.leg_targets([call({"method": "delta", "value": 0.30}, 30)], 700.0, 0.20, 1.0)["lc"]
    k90, n90 = P.leg_targets([call({"method": "delta", "value": 0.10}, 90)], 700.0, 0.20, 1.0)["lc"]
    assert 700 < k0 <= 704 and n0 == 6, "same day: a 30-delta call is a few strikes out"
    assert 715 <= k30 <= 730 and n30 >= 8, "a month out it is twenty strikes away, and the band is wider because the guess is rougher"
    assert 780 <= k90 <= 830 and n90 >= 40, "a 10-delta call three months out is more than sixty strikes away: it was unreachable when only sixty strikes either side were scanned"
    # a premium target
    p0, m0 = P.leg_targets([call({"method": "premium", "value": 1.00}, 0)], 700.0, 0.20, 1.0)["lc"]
    p30, m30 = P.leg_targets([call({"method": "premium", "value": 1.00}, 30)], 700.0, 0.20, 1.0)["lc"]
    assert 700 <= p0 <= 705 and m0 == 12 and 740 <= p30 <= 775 and m30 >= 20, "a one-dollar call a month out is some fifty strikes from the money"
    put = {"id": "lp", "side": "buy", "right": "P", "dte": 30, "strike": {"method": "premium", "value": 1.00}}
    assert P.leg_targets([put], 700.0, 0.20, 1.0)["lp"][0] < 660, "and a one-dollar put sits on the other side"
    # a higher volatility level known before the session moves the centre further out
    assert P.leg_targets([call({"method": "delta", "value": 0.30}, 30)], 700.0, 0.40, 1.0)["lc"][0] > k30


# ----------------------------------------------------------------------------- the stored universe after a large gap
def test_the_stored_strike_range_follows_a_large_gap_at_the_open_and_nothing_later(tmp_settings):
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf"))
    d0, d1, d2, d3 = cal.trading_days(date(2024, 3, 4), date(2024, 3, 7))
    rows = [{"date": d0, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0},
            {"date": d1, "open": 505.0, "high": 540.0, "low": 504.0, "close": 500.0, "volume": 1.0},   # opens 1% up, then runs 8%: not a gap
            {"date": d2, "open": 525.0, "high": 526.0, "low": 520.0, "close": 500.0, "volume": 1.0},   # opens 5% up
            {"date": d3, "open": 470.0, "high": 472.0, "low": 468.0, "close": 470.0, "volume": 1.0}]   # opens 6% down
    st.write_daily("SPY", pl.DataFrame(rows), provider="yahoo")
    lo, hi, info = S.session_strikes(st, "SPY", d1, 3.0, 1.0)
    assert (lo, hi) == (485.0, 515.0) and "widened_for_the_open" not in info, "the range does not follow where the session went after the open"
    lo, hi, info = S.session_strikes(st, "SPY", d2, 3.0, 1.0)
    assert (lo, hi) == (485.0, 533.0), "half the band beyond the open: 525 x 1.015 = 532.9"
    assert info["widened_for_the_open"] == {"open": 525.0, "gap_pct": 5.0, "reach_pct": 1.5, "to": [485.0, 533.0]} and info["from_previous_close"] == [485.0, 515.0]
    lo, hi, info = S.session_strikes(st, "SPY", d3, 3.0, 1.0)
    assert (lo, hi) == (462.0, 515.0) and info["widened_for_the_open"]["gap_pct"] == -6.0
    # the session's own bar is not stored yet (a cycle that runs before the daily history is refreshed): the previous close alone
    nxt = cal.next_trading_day(d3, 1)
    lo, hi, info = S.session_strikes(st, "SPY", nxt, 3.0, 1.0)
    assert (lo, hi) == S.strike_band(470.0, 3.0, 1.0) and "not stored yet" in info["open"]
    with pytest.raises(S.SourceError, match="never from where the session went"):
        S.session_strikes(st, "SPY", d0, 3.0, 1.0)
    # the history is split-adjusted to today, strikes are not: before a ten-for-one split the band is ten times higher
    st.write_corporate_actions("SPY", pl.DataFrame([{"ex_date": d3, "action": "split", "ratio": 10.0, "amount": None}]))
    lo, hi, info = S.session_strikes(st, "SPY", d1, 3.0, 1.0)
    assert (lo, hi) == (4850.0, 5150.0) and info["as_traded_factor"] == 10.0 and info["previous_close"] == 5000.0
    assert S.session_strikes(st, "SPY", nxt, 3.0, 1.0)[:2] == S.strike_band(470.0, 3.0, 1.0), "after the split the stored prices are the traded ones"


# ----------------------------------------------------------------------------- a strike the first fetch did not reach
class Feed(S.OptionBarSource):
    """A provider that lists every whole strike it is asked about and serves each contract three prints a session."""
    name, label, tier = "testfeed", "Test feed", "free_key"
    times: tuple = ("09:59", "10:30", "15:28")
    asked: list = []

    def available(self):
        return True, "key configured"

    def bars_window(self, today=None):
        return date(2024, 1, 1), date(2026, 12, 31)

    def capabilities(self, today=None):
        return [{"dataset": "option_trades", "note": "a test feed"}]

    def list_contracts(self, client, sym, as_of, e_lo, e_hi, lo, hi, rights):
        return [S.contract_from_occ(S.occ_symbol(sym, e_lo, r, float(k))) for r in rights for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]

    def acquire(self, sym, wanted, progress=None, should_cancel=None, force=False):
        idx = self.store.option_bar_index(sym, provider=self.name)
        have = set() if idx.is_empty() else set(zip(idx["date"].to_list(), idx["occ"].to_list()))
        new = {day: [c for c in cs if (day, c.occ) not in have] for day, cs in wanted.items()}
        type(self).asked.append(sorted(c.strike for cs in new.values() for c in cs))
        for day, cs in new.items():
            if cs:
                put_chain(self.store, day, cs[0].expiration, [c.strike for c in cs], times=self.times)
        n, total = sum(len(v) for v in new.values()), sum(len(v) for v in wanted.values())
        return {"pairs_requested": total, "pairs_cached": total - n, "pairs_ok": n, "pairs_empty": 0, "pairs_unavailable": 0, "pairs_failed": 0, "bars": 3 * n, "requests": 1 if n else 0}


@pytest.fixture
def online(monkeypatch):
    """Downloads on, with the class above as the only option source and every other source unreachable: a request for the
    underlying's history fails the test instead of reaching a real service. (The suite itself runs with downloads off.)"""
    from strata.research.pipeline import Pipeline

    def use(st, times=("09:59", "10:30", "15:28")):
        monkeypatch.setattr(Feed, "times", times)
        monkeypatch.setattr(Feed, "asked", [])
        monkeypatch.setattr(S, "option_bar_sources", lambda store=None: [Feed(store or st)])
        return Feed.asked

    def no_network(*a, **k):
        raise AssertionError("a test must not reach a real provider")
    monkeypatch.delenv("STRATA_OFFLINE")
    monkeypatch.setattr(Pipeline, "_ensure_daily", lambda self, req: None)
    monkeypatch.setattr(P, "underlying_actions", lambda *a, **k: ([], []))
    monkeypatch.setattr(S, "get_source", no_network)
    return use


def request(d: date, strike: dict, **approvals) -> dict:
    """The request the lab sends for `cfg(d, strike)`: trade prints, the period stated, nothing approved in advance."""
    c = cfg(d, strike)
    return {"kind": "options", "mode": "trades", "symbol": "SPY", "label": "call at 10:00", "study": None, "provider": "testfeed", "approvals": approvals, "interpretation": {},
            "requested": {"start": d.isoformat(), "end": d.isoformat(), "stated": True}, "backtest": {"strategies": c["strategies"], "execution": c["execution"]}}


def test_a_strike_outside_the_first_fetched_range_is_fetched_before_coverage_is_reported(tmp_settings, online):
    """Found on the stored SPY history (2024-12-19): the contracts were fetched around the price at the signal, the price
    had moved four strikes by the first print, and the strike the entry then needed was one below the fetched range. The
    run asked for a go-ahead on "74 of 75 sessions", fetched the strike while simulating, traded all 75, and its summary
    still said 74. The range is widened before anything is reported."""
    from strata.api.results import load_run_result, load_trades
    from strata.research.pipeline import Pipeline
    d = DAYS[0]
    hist = cal.trading_days(date(2024, 2, 5), DAYS[1])  # enough sessions before it for the rule's warm-up
    path = lambda day, i: SPOT if day != d or i <= 30 else 495.6  # 500 at the 10:00 signal, 495.6 from 10:01
    late = ("10:02", "10:30", "15:28")  # the first print of the session comes after the price has moved

    def world(name: str) -> MarketStore:
        st = MarketStore(tmp_settings / name)
        st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
        write_minutes(st, "SPY", hist, path)
        return st

    st = world("market")
    asked = online(st, late)
    out = Pipeline("widen1", request(d, {"method": "atm_offset", "value": 0}), store=st).run()
    # the first fetch is centred on 500, the price at the signal; the entry needs 496, one strike below it; the second fetch reaches it
    assert asked == [[497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0], [494.0, 495.0, 496.0, 504.0, 505.0, 506.0]]
    assert out["outcome"] == "completed" and out["qualifying"] == 1 and out["executable"] == 1, "nobody was asked to give the session up"
    trades = load_trades("widen1")
    assert len(trades) == 1 and trades[0]["legs"][0]["strike"] == 496.0, "the contract the rule asked for at the price it was filled at, not the nearest one that happened to be stored"
    cov = load_run_result({"id": "widen1", "kind": "backtest"})["coverage"]
    assert cov["executable"] == {"signals": 1, "of": 1, "window": [d.isoformat(), d.isoformat()]} and cov["options"]["issues"] == [] and cov["options"]["outside_stored_strikes"] == []
    assert cov["acquisition"]["widened"] == {"rounds": 1, "sessions": [d.isoformat()], "stopped": None}
    # the replay of a run made this way (through the lab's pipeline, which stores the request and not the engine's
    # configuration) draws the exit rules the run was made with
    from strata.research import replay as RP
    seen = RP.trade("widen1", trades[0]["id"], st)
    assert seen["levels"]["clock"] == {"rule": "clock exit", "time": "15:30"} and seen["legs"][0]["occ"].endswith("C00496000")
    assert [x["why"] for x in cov["downloads"]] == ["option trade prints for 1 qualifying session(s): 7 contract-sessions requested", "a wider strike range for 1 session(s) whose strike was outside the stored one (round 1)"]
    # with downloads off the same request reports the session as not priced, fetches nothing and substitutes nothing
    st2 = world("market2")
    put_chain(st2, d, d, range(497, 504), times=late)
    asked = online(st2, late)
    off = Pipeline("widen2", request(d, {"method": "atm_offset", "value": 0}, no_download=True), store=st2).run()
    assert off["outcome"] == "needs_decision" and asked == [] and load_trades("widen2") == []
    assert "None of the 1 qualifying sessions has usable trade prints: the strike the strategy needs is outside the strikes stored for the session (1 signal)" in off["decision"]["reason"]


def test_a_run_that_widens_the_range_while_simulating_counts_its_coverage_again(store, online):
    """A filled selection on the outermost stored strike may have a better contract beyond it: the run fetches a wider range
    and simulates again. What it then says about coverage is counted from the store as it stands after that fetch."""
    from strata.api.results import load_run_result, load_trades
    from strata.research.pipeline import Pipeline
    asked = online(store)
    # D_LOW stores the money and below. A 45-delta call is the 500 strike there: the outermost one, within tolerance, so it trades
    out = Pipeline("edge1", request(D_LOW, {"method": "delta", "value": 0.45}), store=store).run()
    assert out["outcome"] == "completed" and out["qualifying"] == out["executable"] == 1 and len(load_trades("edge1")) == 1
    res = load_run_result({"id": "edge1", "kind": "backtest"})
    ad = res["coverage"]["contract_selection"]["adaptive"]
    assert ad["rounds"] == 1 and ad["widened_sessions"] == [D_LOW.isoformat()] and ad["remaining"] == 0 and ad["stopped"] is None
    assert len(asked) == 1 and not set(asked[0]) & {float(k) for k in range(495, 501)} and {501.0, 502.0, 503.0} <= set(asked[0]), "one wider fetch, of strikes that were not stored"
    stage = next(s for s in res["pipeline"] if s["id"] == "contracts")
    assert stage["detail"] == "1 of 1 qualifying sessions have usable trade prints, counted again after the strike range was widened for 1 session(s)"
    assert res["coverage"]["executable"]["signals"] == 1 and res["coverage"]["options"]["executable"] == 1 and res["coverage"]["options"]["issues"] == []
    assert res["coverage"]["contract_selection"]["at_band_edge"] == 0, "after the wider fetch the selection has stored strikes on both sides"


# ----------------------------------------------------------------------------- a contract dated on a day the market was closed
def test_a_contract_dated_on_a_closed_day_expires_the_session_before_and_is_never_carried_on(tmp_settings):
    """2025-01-09 was a national day of mourning: the market was closed and the contracts dated that day expired at the
    close of the 8th. Found on the stored SPY history: a call dated the 9th, bought on the 8th and left to expire, never
    met its expiration date in the engine. It stayed open for two years, marked at intrinsic value, and the run's net
    result came to $19,796 where its 74 trades added up to $6,973."""
    from strata.engine.market import day_num, from_day_num
    from strata.engine.selection import effective_expiration
    d8, d9, d10 = date(2025, 1, 8), date(2025, 1, 9), date(2025, 1, 10)
    assert not cal.is_trading_day(d9) and cal.next_trading_day(d8) == d10
    assert from_day_num(effective_expiration(day_num(d9))) == d8 and effective_expiration(day_num(d8)) == day_num(d8)
    assert from_day_num(effective_expiration(day_num(date(2025, 1, 11)))) == d10, "a Saturday date expires on the Friday"
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    days = cal.trading_days(date(2024, 12, 16), date(2025, 1, 17))
    write_minutes(st, "SPY", days, lambda day, i: 503.0 if day > d8 or (day == d8 and i >= 60) else SPOT)  # 500 until 10:30 on the 8th, 503 from then on
    for exp in (d8, d9, d10):
        put_chain(st, d8, exp, range(495, 506))
    put_chain(st, d10, d10, range(495, 506))

    def account(res) -> float:
        """What the account made over the run, from its own last equity: the figure a result's net is taken from."""
        return float(res.daily["equity"][-1]) - 100_000.0

    def run(leg: dict, exits: dict | None = None, last: date = d10):
        c = cfg(d8, {"method": "atm_offset", "value": 0}, **leg)
        c["end"] = last.isoformat()
        c["strategies"][0]["exits"]["time_exit"] = (exits or {}).get("time_exit", {"enabled": False, "time": "15:30", "dte_lte": None})
        # one entry, on the 8th: the rule would fire again on the 10th
        c["strategies"][0]["entry"]["conditions"]["children"].append({"kind": "cond", "type": "expr", "params": {"expr": "price < 501"}})
        return run_backtest(c, store=st, record_paths=False)

    # the contract dated the 9th, named by its date and left alone: settled at the close of the 8th, 3.00 in the money
    res = run({"dte": 1, "dte_mode": "nearest", "expiration": d9})
    assert len(res.trades) == 1 and res.diagnostics["entries"] == 1
    t = res.trades[0]
    leg = t["legs"][0]
    assert leg["expiration"] == d9.isoformat() and leg["strike"] == 500.0 and t["exit_reason"] in ("exercised", "expiration") and leg["exit_price"] == pytest.approx(3.0)
    assert t["exit_time"][:10] in (d8.isoformat(), d10.isoformat()), "settled on the 8th (delivered shares, if any, go at the next open)"
    assert account(res) == pytest.approx(sum(x["pnl"] for x in res.trades)), "what the account made is what its trades made: nothing is left open and marked"
    assert int(res.daily["open_positions"][-1]) == 0
    assert not [w for w in res.diagnostics.get("warnings", []) if "settled at this session's open" in w]
    # the same contract with "close at 15:30 on the day it expires": that day is the 8th
    res = run({"dte": 1, "dte_mode": "nearest", "expiration": d9}, {"time_exit": {"enabled": True, "time": "15:30", "dte_lte": 0}})
    assert [(x["exit_reason"], x["exit_time"][:16]) for x in res.trades] == [("time_exit", f"{d8}T15:30")]
    # asked for by days: on the 8th the contract dated the 9th has no day left. "0DTE" is the one dated the 8th, an exact
    # "1DTE" does not exist, and the nearest to one day is the contract dated the 10th
    res = run({"dte": 0, "dte_mode": "exact"})
    assert [x["legs"][0]["expiration"] for x in res.trades] == [d8.isoformat()]
    res = run({"dte": 1, "dte_mode": "exact"})
    assert res.trades == [] and res.meta["executability"]["not_executed"] == 1
    res = run({"dte": 1, "dte_mode": "nearest"})
    assert [x["legs"][0]["expiration"] for x in res.trades] == [d10.isoformat()] and account(res) == pytest.approx(sum(x["pnl"] for x in res.trades))
