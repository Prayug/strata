"""MODELED option chains: theoretical premiums on top of real underlying prices.

Used only by the explicitly selected "Modeled options backtest" mode. Nothing here is an
observed option price: premiums come from Black-Scholes / Barone-Adesi-Whaley with the
volatility, skew and spread assumptions in `Pricing`, and every result built on them is
labelled MODELED. Historical mode never reads from this module.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.core.pricing import model_chain_vec
from strata.engine.models import Pricing

EPOCH = date(1970, 1, 1)


def listed_expirations(d: date, max_dte: int) -> list[date]:
    """Assumed listing schedule: an expiration every session for the next week, weekly
    Fridays after that, third-Friday monthlies further out."""
    out, dd = set(), d
    while (dd - d).days <= min(max_dte, 7):
        if cal.is_trading_day(dd):
            out.add(dd)
        dd += timedelta(days=1)
    f = d + timedelta(days=(4 - d.weekday()) % 7)
    while (f - d).days <= max_dte + 7:
        e = f if cal.is_trading_day(f) else cal.prev_trading_day(f)
        if e >= d:
            out.add(e)
        f += timedelta(days=7)
    return sorted(e for e in out if (e - d).days <= max_dte + 7)


def strike_step(spot: float, override: float | None) -> float:
    if override:
        return override
    return 0.5 if spot < 50 else 1.0 if spot < 300 else 1.0 if spot < 1000 else 5.0


def base_vol(pr: Pricing, vix: float, rv: float) -> tuple[float, str]:
    if pr.iv_source == "fixed":
        return pr.iv_fixed * pr.iv_scale, f"fixed {pr.iv_fixed:.0%} x {pr.iv_scale:g}"
    if pr.iv_source == "realized" or not math.isfinite(vix):
        if math.isfinite(rv) and rv > 0:
            return rv * pr.iv_scale, f"21-session realised vol x {pr.iv_scale:g}"
        return pr.iv_fixed * pr.iv_scale, f"fallback fixed {pr.iv_fixed:.0%} (no volatility input available)"
    return vix / 100.0 * pr.iv_scale, f"VIX x {pr.iv_scale:g}"


def modeled_chain(d: date, ticks: list[int], spots: list[float], vols: list[float], pr: Pricing, max_dte: int) -> pl.DataFrame:
    """One session of modeled quotes. ticks: epoch seconds; spots/vols: underlying price and
    ATM vol (decimal) observable at each tick."""
    close_t = int(cal.session_bounds(d)[1].timestamp())
    exps = listed_expirations(d, max_dte)
    rows_t, rows_S, rows_K, rows_T, rows_sig, rows_call, rows_exp = [], [], [], [], [], [], []
    for t, S, v in zip(ticks, spots, vols):
        if not (math.isfinite(S) and S > 0 and math.isfinite(v) and v > 0):
            continue
        step = strike_step(S, pr.strike_step)
        for e in exps:
            exp_close = int(cal.session_bounds(e)[1].timestamp())
            if exp_close <= t:
                continue
            T = (exp_close - t) / (365.0 * 86400.0)
            band = max(0.03, 3.0 * v * math.sqrt(max(T, 1 / 365)))
            ks = np.arange(math.floor(S * (1 - band) / step) * step, S * (1 + band) + step, step)
            ks = ks[ks > 0]
            sig = np.clip(v - pr.skew_per_10pct * np.log(ks / S) / 0.10, 0.03, 3.0)
            for is_call in (True, False):
                n = ks.size
                rows_t.append(np.full(n, t, np.int64)); rows_S.append(np.full(n, S)); rows_K.append(ks); rows_T.append(np.full(n, T))
                rows_sig.append(sig); rows_call.append(np.full(n, is_call)); rows_exp.append(np.full(n, (e - EPOCH).days, np.int32))
    if not rows_t:
        return pl.DataFrame()
    t = np.concatenate(rows_t); S = np.concatenate(rows_S); K = np.concatenate(rows_K); T = np.concatenate(rows_T)
    sig = np.concatenate(rows_sig); is_call = np.concatenate(rows_call); exp = np.concatenate(rows_exp)
    G = model_chain_vec(S, K, T, pr.rate, pr.dividend_yield, sig, is_call, pr.model == "american_baw")
    mid = G[:, 0]
    half = np.maximum(pr.min_spread, mid * pr.spread_pct / 100.0) / 2.0
    bid = np.maximum(0.0, np.round(mid - half, 2))
    ask = np.maximum(bid + pr.min_spread, np.round(mid + half, 2))
    df = pl.DataFrame({"ts": t, "expiration": exp, "strike": K, "right": np.where(is_call, "C", "P"), "bid": bid, "ask": ask, "iv": sig,
                       "delta": G[:, 1], "gamma": G[:, 2], "theta": G[:, 3], "vega": G[:, 4], "underlying_price": S, "quote_ts": t})
    return df.with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
        pl.from_epoch("quote_ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
        pl.col("expiration").cast(pl.Date),
        pl.lit(None, dtype=pl.Int32).alias("bid_size"), pl.lit(None, dtype=pl.Int32).alias("ask_size"), pl.lit(None, dtype=pl.Float64).alias("last"),
        pl.lit(None, dtype=pl.Int64).alias("volume"), pl.lit(None, dtype=pl.Int64).alias("open_interest"),
    )


def assumptions_text(pr: Pricing) -> list[str]:
    return [
        f"Pricing model: {'Barone-Adesi-Whaley American approximation' if pr.model == 'american_baw' else 'Black-Scholes (European)'}; Greeks are Black-Scholes.",
        f"ATM volatility: {'VIX level x %g' % pr.iv_scale if pr.iv_source == 'vix' else ('21-session realised volatility x %g' % pr.iv_scale if pr.iv_source == 'realized' else 'fixed %.0f%% x %g' % (pr.iv_fixed * 100, pr.iv_scale))}, flat across expirations.",
        f"Skew: {pr.skew_per_10pct * 100:+.1f} vol points per 10% of downside moneyness (linear in log-moneyness).",
        f"Bid-ask spread: {pr.spread_pct:g}% of the modeled premium (minimum ${pr.min_spread:.2f}); fills use the selected execution model on these modeled quotes.",
        f"Rates: r = {pr.rate:.2%}, dividend yield = {pr.dividend_yield:.2%}. Assumed listings: daily expirations for a week, weekly Fridays beyond.",
    ]
