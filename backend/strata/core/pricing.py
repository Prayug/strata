"""Black-Scholes-Merton pricing, Greeks and implied volatility (Numba-compiled).

These functions are used for (a) computing IV/Greeks from *observed* historical quotes
when a provider does not supply them, (b) theoretical payoff/T+0 curves in the strategy
builder, and (c) the clearly-labelled synthetic demo dataset generator. They are never
used to fill in missing historical quotes during a backtest.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SQRT2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


@njit(cache=True, fastmath=False)
def ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


@njit(cache=True)
def npdf(x: float) -> float:
    return INV_SQRT2PI * math.exp(-0.5 * x * x)


@njit(cache=True)
def bs_price_scalar(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = S - K if is_call else K - S
        return intrinsic if intrinsic > 0.0 else 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    if is_call:
        return S * math.exp(-q * T) * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
    return K * math.exp(-r * T) * ncdf(-d2) - S * math.exp(-q * T) * ncdf(-d1)


@njit(cache=True)
def bs_greeks_scalar(S, K, T, r, q, sigma, is_call):
    """Returns (price, delta, gamma, theta_per_day, vega_per_vol_point)."""
    if T <= 1e-10 or sigma <= 0.0:
        intrinsic = (S - K) if is_call else (K - S)
        price = intrinsic if intrinsic > 0.0 else 0.0
        if is_call:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return price, delta, 0.0, 0.0, 0.0
    sqT = math.sqrt(T)
    sq = sigma * sqT
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    eqT = math.exp(-q * T)
    erT = math.exp(-r * T)
    pdf1 = npdf(d1)
    gamma = eqT * pdf1 / (S * sq)
    vega = S * eqT * pdf1 * sqT / 100.0
    if is_call:
        price = S * eqT * ncdf(d1) - K * erT * ncdf(d2)
        delta = eqT * ncdf(d1)
        theta = (-S * eqT * pdf1 * sigma / (2.0 * sqT) - r * K * erT * ncdf(d2) + q * S * eqT * ncdf(d1)) / 365.0
    else:
        price = K * erT * ncdf(-d2) - S * eqT * ncdf(-d1)
        delta = -eqT * ncdf(-d1)
        theta = (-S * eqT * pdf1 * sigma / (2.0 * sqT) + r * K * erT * ncdf(-d2) - q * S * eqT * ncdf(-d1)) / 365.0
    return price, delta, gamma, theta, vega


@njit(cache=True)
def implied_vol_scalar(price, S, K, T, r, q, is_call):
    """Implied volatility via safeguarded Newton/bisection. Returns NaN when the price is
    outside no-arbitrage bounds or the option has expired."""
    if T <= 1e-10 or price <= 0.0 or S <= 0.0 or K <= 0.0:
        return np.nan
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    lower = max(disc_s - disc_k, 0.0) if is_call else max(disc_k - disc_s, 0.0)
    upper = disc_s if is_call else disc_k
    if price < lower - 1e-9 or price >= upper:
        return np.nan
    lo, hi = 1e-4, 8.0
    sigma = 0.25
    for _ in range(100):
        p = bs_price_scalar(S, K, T, r, q, sigma, is_call)
        diff = p - price
        if abs(diff) < 1e-8:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        sqT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT)
        vega = S * math.exp(-q * T) * npdf(d1) * sqT
        nxt = sigma - diff / vega if vega > 1e-12 else 0.5 * (lo + hi)
        if nxt <= lo or nxt >= hi:
            nxt = 0.5 * (lo + hi)
        sigma = nxt
        if hi - lo < 1e-10:
            break
    return sigma


@njit(cache=True, parallel=True)
def bs_price_vec(S, K, T, r, q, sigma, is_call):
    n = K.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = bs_price_scalar(S[i], K[i], T[i], r, q, sigma[i], is_call[i])
    return out


@njit(cache=True, parallel=True)
def bs_greeks_vec(S, K, T, r, q, sigma, is_call):
    n = K.shape[0]
    out = np.empty((n, 5))
    for i in prange(n):
        p, d, g, t, v = bs_greeks_scalar(S[i], K[i], T[i], r, q, sigma[i], is_call[i])
        out[i, 0] = p
        out[i, 1] = d
        out[i, 2] = g
        out[i, 3] = t
        out[i, 4] = v
    return out


@njit(cache=True, parallel=True)
def implied_vol_vec(price, S, K, T, r, q, is_call):
    n = price.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = implied_vol_scalar(price[i], S[i], K[i], T[i], r, q, is_call[i])
    return out


def intrinsic(S, K, is_call):
    S = np.asarray(S, dtype=float)
    return np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
