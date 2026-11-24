from datetime import date, datetime

import numpy as np
import pytest

from strata.core import calendar as cal
from strata.core import indicators as ind
from strata.core.pricing import bs_greeks_scalar, bs_price_scalar, implied_vol_scalar


# ---------------------------------------------------------------- calendar
def test_nyse_holidays_2024():
    expected = {date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29), date(2024, 5, 27),
                date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28), date(2024, 12, 25)}
    assert set(cal.holidays(2024)) == expected


def test_observed_rules_and_special_closures():
    assert date(2022, 12, 26) in cal.holidays(2022)  # Christmas on Sunday -> Monday
    assert date(2021, 12, 31) not in cal.holidays(2021)  # NY Day on Saturday: not observed Friday
    assert date(2026, 7, 3) in cal.holidays(2026)  # July 4 on Saturday -> Friday
    assert not cal.is_trading_day(date(2025, 1, 9))  # Carter national day of mourning
    assert not cal.is_trading_day(date(2012, 10, 29))  # Sandy


def test_early_closes():
    assert cal.session_close_time(date(2024, 7, 3)).hour == 13
    assert cal.session_close_time(date(2024, 11, 29)).hour == 13
    assert cal.session_close_time(date(2024, 12, 24)).hour == 13
    assert cal.session_close_time(date(2023, 12, 22)).hour == 16  # Dec 24 2023 was a Sunday
    assert cal.session_minutes(date(2024, 11, 29)) == 210
    assert cal.session_minutes(date(2024, 11, 27)) == 390


def test_dst_session_bounds_in_utc():
    o_est, _ = cal.session_bounds(date(2024, 3, 8))  # before DST
    o_edt, _ = cal.session_bounds(date(2024, 3, 11))  # after DST
    assert o_est.astimezone(cal.UTC).hour == 14 and o_est.astimezone(cal.UTC).minute == 30
    assert o_edt.astimezone(cal.UTC).hour == 13 and o_edt.astimezone(cal.UTC).minute == 30


def test_trading_day_navigation():
    assert cal.next_trading_day(date(2024, 3, 28)) == date(2024, 4, 1)  # skips Good Friday + weekend
    assert cal.prev_trading_day(date(2024, 1, 2)) == date(2023, 12, 29)
    assert cal.trading_days_between(date(2024, 1, 2), date(2024, 1, 9)) == 5


# ---------------------------------------------------------------- pricing
def test_black_scholes_reference_values():
    # Hull, Options Futures & Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    assert bs_price_scalar(100, 100, 1.0, 0.05, 0.0, 0.2, True) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price_scalar(100, 100, 1.0, 0.05, 0.0, 0.2, False) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_and_greeks():
    S, K, T, r, s = 102.0, 100.0, 0.25, 0.03, 0.25
    c = bs_price_scalar(S, K, T, r, 0.0, s, True)
    p = bs_price_scalar(S, K, T, r, 0.0, s, False)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)
    _, dc, gc, _, vc = bs_greeks_scalar(S, K, T, r, 0.0, s, True)
    _, dp, gp, _, vp = bs_greeks_scalar(S, K, T, r, 0.0, s, False)
    assert dc - dp == pytest.approx(1.0, abs=1e-9)
    assert gc == pytest.approx(gp) and vc == pytest.approx(vp)
    # finite-difference delta
    h = 1e-3
    fd = (bs_price_scalar(S + h, K, T, r, 0.0, s, True) - bs_price_scalar(S - h, K, T, r, 0.0, s, True)) / (2 * h)
    assert dc == pytest.approx(fd, abs=1e-6)


def test_implied_vol_roundtrip_and_bounds():
    for sig in (0.08, 0.2, 0.65):
        for is_call in (True, False):
            px = bs_price_scalar(100, 95, 0.1, 0.02, 0.0, sig, is_call)
            assert implied_vol_scalar(px, 100, 95, 0.1, 0.02, 0.0, is_call) == pytest.approx(sig, abs=1e-6)
    assert np.isnan(implied_vol_scalar(0.5, 100, 90, 0.1, 0.0, 0.0, True))  # below intrinsic
    assert np.isnan(implied_vol_scalar(1.0, 100, 100, 0.0, 0.0, 0.0, True))  # expired


def test_expired_option_is_intrinsic():
    assert bs_price_scalar(105, 100, 0.0, 0.05, 0.0, 0.3, True) == 5.0
    assert bs_price_scalar(105, 100, 0.0, 0.05, 0.0, 0.3, False) == 0.0


# ---------------------------------------------------------------- indicators
def test_indicators_hand_values():
    x = np.array([1, 2, 3, 4, 5], float)
    assert np.allclose(ind.sma(x, 3)[2:], [2, 3, 4])
    e = ind.ema(x, 3)
    assert e[2] == pytest.approx(2.0) and e[3] == pytest.approx(3.0) and e[4] == pytest.approx(4.0)
    r = ind.rsi(np.array([1, 2, 3, 4, 5, 6], float), 3)
    assert r[3] == 100.0  # only gains


@pytest.mark.parametrize("name,params", [("rsi", {"period": 14}), ("ema", {"period": 9}), ("macd", {}), ("bbands", {}), ("atr", {"period": 14}), ("vwap", {})])
def test_indicators_are_causal(name, params):
    """Value at i must not change when future bars are altered (no look-ahead)."""
    rng = np.random.default_rng(1)
    n = 300
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    h, l, v = c + 1, c - 1, rng.uniform(1, 2, n)
    sess = np.repeat(np.arange(3), 100)
    full = ind.compute(name, params, c, h, l, c, v, sess)
    c2 = c.copy()
    c2[200:] *= 1.5
    part = ind.compute(name, params, c2, c2 + 1, c2 - 1, c2, v, sess)
    for k in full:
        assert np.allclose(full[k][:200], part[k][:200], equal_nan=True), k
