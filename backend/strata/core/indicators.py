"""Causal technical indicators (Numba). Every output at index i depends only on inputs
at indices <= i, which is what lets the engine look values up by "last completed bar"
without look-ahead."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def sma(x, n):
    out = np.full(x.shape[0], np.nan)
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i]
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


@njit(cache=True)
def ema(x, n):
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] < n:
        return out
    a = 2.0 / (n + 1.0)
    s = 0.0
    for i in range(n):
        s += x[i]
    v = s / n
    out[n - 1] = v
    for i in range(n, x.shape[0]):
        v = a * x[i] + (1 - a) * v
        out[i] = v
    return out


@njit(cache=True)
def rsi(x, n):
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] <= n:
        return out
    gain = 0.0
    loss = 0.0
    for i in range(1, n + 1):
        d = x[i] - x[i - 1]
        if d > 0:
            gain += d
        else:
            loss -= d
    ag = gain / n
    al = loss / n
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, x.shape[0]):
        d = x[i] - x[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + l) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


@njit(cache=True)
def atr(h, l, c, n):
    m = c.shape[0]
    out = np.full(m, np.nan)
    if m <= n:
        return out
    tr = np.empty(m)
    tr[0] = h[0] - l[0]
    for i in range(1, m):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    s = 0.0
    for i in range(1, n + 1):
        s += tr[i]
    v = s / n
    out[n] = v
    for i in range(n + 1, m):
        v = (v * (n - 1) + tr[i]) / n
        out[i] = v
    return out


@njit(cache=True)
def rolling_std(x, n):
    out = np.full(x.shape[0], np.nan)
    for i in range(n - 1, x.shape[0]):
        m = 0.0
        for j in range(i - n + 1, i + 1):
            m += x[j]
        m /= n
        v = 0.0
        for j in range(i - n + 1, i + 1):
            v += (x[j] - m) ** 2
        out[i] = np.sqrt(v / n)
    return out


@njit(cache=True)
def session_vwap(h, l, c, v, session_id):
    out = np.full(c.shape[0], np.nan)
    pv = 0.0
    vv = 0.0
    cur = -1
    for i in range(c.shape[0]):
        if session_id[i] != cur:
            cur = session_id[i]
            pv = 0.0
            vv = 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        w = v[i] if v[i] > 0 else 1.0
        pv += tp * w
        vv += w
        out[i] = pv / vv
    return out


def macd(x, fast=12, slow=26, signal=9):
    line = ema(x, fast) - ema(x, slow)
    sig = np.full_like(line, np.nan)
    valid = ~np.isnan(line)
    if valid.sum() >= signal:
        idx = np.argmax(valid)
        sig[idx:] = ema(line[idx:], signal)
    return line, sig, line - sig


def bollinger(x, n=20, k=2.0):
    mid = sma(x, n)
    sd = rolling_std(x, n)
    up, lo = mid + k * sd, mid - k * sd
    width = (up - lo) / mid
    with np.errstate(invalid="ignore", divide="ignore"):
        pctb = (x - lo) / (up - lo)
    return mid, up, lo, pctb, width


def realized_vol(close, n=21, periods_per_year=252):
    r = np.diff(np.log(close), prepend=np.nan)
    sd = np.full_like(close, np.nan)
    if close.shape[0] > n:
        sd[1:] = rolling_std(np.nan_to_num(r[1:]), n)
    return sd * np.sqrt(periods_per_year)


INDICATORS = {
    "sma": {"params": {"period": 20}, "fields": ["value"]},
    "ema": {"params": {"period": 20}, "fields": ["value"]},
    "rsi": {"params": {"period": 14}, "fields": ["value"]},
    "atr": {"params": {"period": 14}, "fields": ["value", "pct"]},
    "macd": {"params": {"fast": 12, "slow": 26, "signal": 9}, "fields": ["line", "signal", "hist"]},
    "bbands": {"params": {"period": 20, "k": 2.0}, "fields": ["mid", "upper", "lower", "pctb", "width"]},
    "vwap": {"params": {}, "fields": ["value", "dist_pct"]},
    "rvol": {"params": {"period": 21}, "fields": ["value"]},
}


SESSION_ANCHORED = {"vwap"}  # restart with each session, so they have no value until that session's first bar has completed


def compute(name: str, params: dict, o, h, l, c, v, session_id) -> dict[str, np.ndarray]:
    p = {**INDICATORS[name]["params"], **(params or {})}
    c = np.ascontiguousarray(c, dtype=np.float64)
    if name == "sma":
        return {"value": sma(c, int(p["period"]))}
    if name == "ema":
        return {"value": ema(c, int(p["period"]))}
    if name == "rsi":
        return {"value": rsi(c, int(p["period"]))}
    if name == "atr":
        a = atr(np.asarray(h, float), np.asarray(l, float), c, int(p["period"]))
        return {"value": a, "pct": a / c * 100}
    if name == "macd":
        line, sig, hist = macd(c, int(p["fast"]), int(p["slow"]), int(p["signal"]))
        return {"line": line, "signal": sig, "hist": hist}
    if name == "bbands":
        mid, up, lo, pctb, width = bollinger(c, int(p["period"]), float(p["k"]))
        return {"mid": mid, "upper": up, "lower": lo, "pctb": pctb, "width": width}
    if name == "vwap":
        vw = session_vwap(np.asarray(h, float), np.asarray(l, float), c, np.asarray(v, float), np.asarray(session_id, np.int64))
        return {"value": vw, "dist_pct": (c / vw - 1) * 100}
    if name == "rvol":
        return {"value": realized_vol(c, int(p["period"])) * 100}
    raise ValueError(f"unknown indicator {name}")
