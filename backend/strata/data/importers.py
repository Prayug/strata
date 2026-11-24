"""CSV / Parquet import for user-supplied datasets.

kinds:
  underlying_minutes  ts, open, high, low, close[, volume]       (bar START time)
  underlying_daily    date, open, high, low, close[, volume]
  option_quotes       ts, expiration, strike, right, bid, ask[, bid_size, ask_size, volume,
                      open_interest, iv, delta, gamma, theta, vega, underlying_price, quote_ts]
  events              date, event_type[, time, description]
  corporate_actions   ex_date, action (split|dividend)[, ratio, amount]
Timestamps without a timezone are interpreted in `tz` (default America/New_York).
Rows are validated (bid <= ask, non-negative prices, known rights) and invalid rows are
reported, never silently fixed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from strata.data.store import MarketStore, SymbolMeta, minutes_to_daily

KINDS = {
    "underlying_minutes": ["ts", "open", "high", "low", "close"],
    "underlying_daily": ["date", "open", "high", "low", "close"],
    "option_quotes": ["ts", "expiration", "strike", "right", "bid", "ask"],
    "events": ["date", "event_type"],
    "corporate_actions": ["ex_date", "action"],
}
ALIASES = {
    "timestamp": "ts", "time": "ts", "datetime": "ts", "quote_datetime": "ts", "expiry": "expiration", "expiration_date": "expiration", "exp": "expiration",
    "option_type": "right", "type": "right", "cp": "right", "call_put": "right", "strike_price": "strike", "bid_price": "bid", "ask_price": "ask",
    "implied_volatility": "iv", "impl_vol": "iv", "underlying": "underlying_price", "underlying_last": "underlying_price", "active_underlying_price": "underlying_price",
    "oi": "open_interest", "vol": "volume", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
}


def _read(path: str) -> pl.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pl.read_parquet(p)
    return pl.read_csv(p, infer_schema_length=20000, try_parse_dates=True)


def _ts(df: pl.DataFrame, col: str, tz: str) -> pl.DataFrame:
    s = df[col]
    if s.dtype == pl.String:
        df = df.with_columns(pl.col(col).str.to_datetime(strict=False, time_unit="us"))
        s = df[col]
    if s.dtype in (pl.Int64, pl.Float64):
        unit = "ms" if s.max() > 10**11 else "s"
        df = df.with_columns(pl.from_epoch(col, time_unit=unit).dt.replace_time_zone("UTC"))
    elif isinstance(s.dtype, pl.Datetime) and s.dtype.time_zone is None:
        df = df.with_columns(pl.col(col).dt.replace_time_zone(tz, ambiguous="earliest"))
    return df.with_columns(pl.col(col).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))


def import_file(path: str, kind: str, symbol: str | None, tz: str = "America/New_York", column_map: dict | None = None,
                meta: dict | None = None, store: MarketStore | None = None, progress=None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown import kind {kind}; expected one of {list(KINDS)}")
    store = store or MarketStore()
    df = _read(path)
    df = df.rename({c: c.strip().lower() for c in df.columns})
    ren = {c: ALIASES[c] for c in df.columns if c in ALIASES and ALIASES[c] not in df.columns}
    ren.update({k.lower(): v for k, v in (column_map or {}).items() if k.lower() in df.columns})
    df = df.rename(ren)
    missing = [c for c in KINDS[kind] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}; found {df.columns}")
    report: dict = {"kind": kind, "symbol": symbol, "input_rows": df.height}
    if kind in ("underlying_minutes", "underlying_daily", "option_quotes", "corporate_actions") and not symbol:
        raise ValueError("symbol is required for this import kind")
    if kind == "underlying_minutes":
        df = _ts(df, "ts", tz)
        bad = df.filter((pl.col("high") < pl.col("low")) | (pl.col("close") <= 0))
        df = df.filter(~((pl.col("high") < pl.col("low")) | (pl.col("close") <= 0)))
        store.write_minutes(symbol, df)
        store.write_daily(symbol, minutes_to_daily(store.read_minutes(symbol)))
        report.update(rows=df.height, rejected=bad.height)
    elif kind == "underlying_daily":
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False))
        store.write_daily(symbol, df)
        report.update(rows=df.height, rejected=0, note="daily bars only: intraday conditions and 0DTE logic need minute bars")
    elif kind == "option_quotes":
        df = _ts(df, "ts", tz)
        if "quote_ts" in df.columns:
            df = _ts(df, "quote_ts", tz)
        df = df.with_columns(pl.col("expiration").cast(pl.Date, strict=False), pl.col("right").cast(pl.String).str.to_uppercase().str.slice(0, 1))
        valid = (pl.col("bid") >= 0) & (pl.col("ask") >= pl.col("bid")) & pl.col("right").is_in(["C", "P"]) & (pl.col("strike") > 0) & pl.col("expiration").is_not_null()
        bad = df.filter(~valid)
        df = df.filter(valid)
        if "iv" in df.columns:
            mx = df["iv"].drop_nulls().max()
            if mx is not None and mx > 5:  # percent -> decimal
                df = df.with_columns(pl.col("iv") / 100)
        df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))
        parts = df.partition_by("_d", as_dict=True)
        for i, ((d,), part) in enumerate(sorted(parts.items(), key=lambda kv: kv[0])):
            store.write_option_day(symbol, d, part.drop("_d"))
            if progress:
                progress((i + 1) / len(parts), f"wrote {d}")
        from strata.data.derived import derive_iv_history
        derive_iv_history(store, symbol)
        report.update(rows=df.height, rejected=bad.height, sessions=len(parts), rejected_examples=bad.head(5).to_dicts() if bad.height else [])
    elif kind == "events":
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False), pl.lit("import").alias("source") if "source" not in df.columns else pl.col("source"))
        store.write_events(df)
        report.update(rows=df.height)
    elif kind == "corporate_actions":
        df = df.with_columns(pl.col("ex_date").cast(pl.Date, strict=False), pl.col("action").str.to_lowercase())
        store.write_corporate_actions(symbol, df)
        report.update(rows=df.height)
    if symbol and kind != "events":
        cur = store.symbols().get(symbol)
        if cur is None:
            from strata.data.store import KNOWN_SPECS
            m = SymbolMeta(symbol=symbol, provider="import", **KNOWN_SPECS.get(symbol, {}))
            for k, v in (meta or {}).items():
                if hasattr(m, k):
                    setattr(m, k, v)
            store.upsert_symbol(m)
        store.rebuild_manifest(symbol)
    return report
