"""Derived series computed from *stored* historical chains (never from a model):
daily constant-maturity ATM implied volatility used for IV rank / IV percentile."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from strata.config import get_settings
from strata.core.pricing import implied_vol_vec
from strata.data.store import MarketStore


def ensure_iv(df: pl.DataFrame, r: float | None = None) -> pl.DataFrame:
    """Fill the iv column from observed mid quotes where the provider did not supply it.
    Uses calendar time to the 16:00 ET expiration close."""
    r = get_settings().risk_free_rate if r is None else r
    if "iv" in df.columns and df["iv"].null_count() == 0:
        return df
    exp_close = pl.col("expiration").cast(pl.Datetime("us")).dt.replace_time_zone("America/New_York").dt.offset_by("16h").dt.convert_time_zone("UTC")
    tmp = df.with_columns(((exp_close - pl.col("ts")).dt.total_seconds() / (365 * 86400)).alias("_T"), ((pl.col("bid") + pl.col("ask")) / 2).alias("_mid"))
    need = tmp["iv"].is_null().to_numpy() if "iv" in tmp.columns else np.ones(tmp.height, bool)
    S = tmp["underlying_price"].to_numpy().astype(float)
    ivs = implied_vol_vec(tmp["_mid"].to_numpy().astype(float), S, tmp["strike"].to_numpy().astype(float), tmp["_T"].to_numpy().astype(float), r, 0.0, (tmp["right"] == "C").to_numpy())
    cur = tmp["iv"].to_numpy().astype(float) if "iv" in tmp.columns else np.full(tmp.height, np.nan)
    cur = np.where(need, ivs, cur)
    return df.with_columns(pl.Series("iv", cur))


def atm_iv_by_expiration(snap: pl.DataFrame, spot: float) -> list[tuple[date, float]]:
    out = []
    for (exp,), g in snap.group_by(["expiration"], maintain_order=True):
        g = g.filter(pl.col("iv").is_not_null() & pl.col("iv").is_finite() & (pl.col("bid") > 0))
        if g.is_empty():
            continue
        ks = g["strike"].unique().to_numpy()
        k = ks[np.argmin(np.abs(ks - spot))]
        v = g.filter(pl.col("strike") == k)["iv"].mean()
        if v is not None and math.isfinite(v):
            out.append((exp, float(v)))
    return sorted(out)


def constant_maturity_iv(points: list[tuple[float, float]], target_days: float = 30.0) -> float | None:
    """Interpolate ATM IV in total variance across days-to-expiry."""
    pts = sorted((d, v) for d, v in points if d > 0.05)
    if not pts:
        return None
    if target_days <= pts[0][0]:
        return pts[0][1]
    if target_days >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= target_days <= d1:
            w0, w1 = v0 * v0 * d0, v1 * v1 * d1
            tv = w0 + (w1 - w0) * (target_days - d0) / (d1 - d0)
            return math.sqrt(tv / target_days)
    return None


def derive_iv_history(store: MarketStore, sym: str, full: bool = False, dataset=None) -> pl.DataFrame:
    """Daily constant-maturity ATM IV from each session's last stored QUOTE snapshot, kept
    beside the option-quote dataset it was derived from (never from trade prints).
    Incremental: sessions already derived are kept unless `full`."""
    schema = {"date": pl.Date, "atm_iv_30d": pl.Float64, "atm_iv_front": pl.Float64, "close": pl.Float64}
    dataset = dataset or store.option_dataset(sym, "quote")
    if dataset is None:
        return pl.DataFrame(schema=schema)
    p = store.iv_path(sym, dataset)
    prev = pl.read_parquet(p) if p.exists() and not full else pl.DataFrame(schema=schema)
    have = set(prev["date"].to_list())
    rows = []
    for d in store.option_days(sym, dataset=dataset):
        if d in have:
            continue
        df = store.read_option_day(sym, d, dataset=dataset)
        if df is None or df.is_empty():
            continue
        last_ts = df["ts"].max()
        snap = df.filter(pl.col("ts") == last_ts)
        und = snap["underlying_price"].drop_nulls()
        if und.is_empty() or "bid" not in snap.columns or snap["bid"].null_count() == snap.height:
            continue
        spot = float(und[0])
        ks = snap["strike"].unique().sort()
        near = ks.filter((ks > spot * 0.9) & (ks < spot * 1.1))
        snap = ensure_iv(snap.filter(pl.col("strike").is_in(near)))
        pts = atm_iv_by_expiration(snap, spot)
        if not pts:
            continue
        dpts = []
        for exp, v in pts:
            secs = (pl.Series([exp]).cast(pl.Datetime("us")).dt.replace_time_zone("America/New_York").dt.offset_by("16h").dt.convert_time_zone("UTC")[0] - last_ts).total_seconds()
            dpts.append((secs / 86400.0, v))
        front = next((v for dd, v in dpts if dd > 0.05), None)
        rows.append({"date": d, "atm_iv_30d": constant_maturity_iv(dpts, 30.0), "atm_iv_front": front, "close": spot})
    out = pl.concat([prev.select(list(schema)), pl.DataFrame(rows, schema=schema)]).unique("date", keep="last").sort("date") if rows else prev
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(p)
    return out
