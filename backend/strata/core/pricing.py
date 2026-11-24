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


# ---------------------------------------------------------------- American options (BAW)
@njit(cache=True)
def baw_price_scalar(S, K, T, r, q, sigma, is_call):
    """Barone-Adesi & Whaley (1987) quadratic approximation for American options with a
    continuous dividend yield q. Falls back to Black-Scholes when early exercise is never
    optimal (calls with q <= 0) or inputs are degenerate."""
    euro = bs_price_scalar(S, K, T, r, q, sigma, is_call)
    intrinsic = (S - K) if is_call else (K - S)
    if intrinsic < 0.0:
        intrinsic = 0.0
    if T <= 1e-8 or sigma <= 1e-8:
        return intrinsic
    if r <= 1e-9 or (is_call and q <= 1e-9):
        return euro if euro > intrinsic else intrinsic
    b = r - q
    s2 = sigma * sigma
    M = 2.0 * r / s2
    N = 2.0 * b / s2
    Kc = 1.0 - math.exp(-r * T)
    sqT = sigma * math.sqrt(T)
    disc = (N - 1.0) ** 2 + 4.0 * M / Kc
    if is_call:
        q2 = (-(N - 1.0) + math.sqrt(disc)) / 2.0
        Sx = K * 1.2
        for _ in range(60):  # critical price S*: S* - K = c(S*) + (1 - e^{(b-r)T} N(d1)) S*/q2
            d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / sqT
            c = bs_price_scalar(Sx, K, T, r, q, sigma, True)
            lhs = Sx - K
            rhs = c + (1.0 - math.exp((b - r) * T) * ncdf(d1)) * Sx / q2
            slope = math.exp((b - r) * T) * ncdf(d1) * (1.0 - 1.0 / q2) + (1.0 - math.exp((b - r) * T) * npdf(d1) / sqT) / q2
            if abs(lhs - rhs) < 1e-7 * K:
                break
            Sx = (K + rhs - slope * Sx) / (1.0 - slope)
            if Sx < K:
                Sx = K * 1.0001
        if S >= Sx:
            return S - K
        d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / sqT
        A2 = (Sx / q2) * (1.0 - math.exp((b - r) * T) * ncdf(d1))
        v = euro + A2 * (S / Sx) ** q2
        return v if v > intrinsic else intrinsic
    q1 = (-(N - 1.0) - math.sqrt(disc)) / 2.0
    Sx = K * 0.85
    for _ in range(60):  # S**: K - S** = p(S**) - (1 - e^{(b-r)T} N(-d1)) S**/q1
        d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / sqT
        p = bs_price_scalar(Sx, K, T, r, q, sigma, False)
        lhs = K - Sx
        rhs = p - (1.0 - math.exp((b - r) * T) * ncdf(-d1)) * Sx / q1
        slope = -math.exp((b - r) * T) * ncdf(-d1) * (1.0 - 1.0 / q1) - (1.0 + math.exp((b - r) * T) * npdf(-d1) / sqT) / q1
        if abs(lhs - rhs) < 1e-7 * K:
            break
        Sx = (K - rhs + slope * Sx) / (1.0 + slope)
        if Sx <= 0.0:
            Sx = K * 0.01
        if Sx > K:
            Sx = K * 0.9999
    if S <= Sx:
        return K - S
    d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / sqT
    A1 = -(Sx / q1) * (1.0 - math.exp((b - r) * T) * ncdf(-d1))
    v = euro + A1 * (S / Sx) ** q1
    return v if v > intrinsic else intrinsic


@njit(cache=True, parallel=True)
def model_chain_vec(S, K, T, r, q, sigma, is_call, american):
    """Columns: price, delta, gamma, theta/day, vega/pt. Greeks are Black-Scholes."""
    n = K.shape[0]
    out = np.empty((n, 5))
    for i in prange(n):
        p, d, g, t, v = bs_greeks_scalar(S[i], K[i], T[i], r, q, sigma[i], is_call[i])
        if american:
            p = baw_price_scalar(S[i], K[i], T[i], r, q, sigma[i], is_call[i])
        out[i, 0] = p
        out[i, 1] = d
        out[i, 2] = g
        out[i, 3] = t
        out[i, 4] = v
    return out
