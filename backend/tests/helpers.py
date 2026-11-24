"""Builders for tiny hand-specified market datasets whose correct results are known."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable

import polars as pl

from strata.core import calendar as cal
from strata.data.store import MarketStore, SymbolMeta, minutes_to_daily

UTC = "UTC"


def et_epoch(d: date, hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return int(datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET).timestamp())


def write_minutes(store: MarketStore, sym: str, days: list[date], price: Callable[[date, int], float]):
    """price(day, minute_index) -> close of that 1-minute bar; open = previous close."""
    rows = []
    for d in days:
        o_ts = int(cal.session_bounds(d)[0].timestamp())
        n = cal.session_minutes(d)
        prev = price(d, -1)
        for i in range(n):
            c = price(d, i)
            rows.append((o_ts + 60 * i, prev, max(prev, c), min(prev, c), c, 1000.0))
            prev = c
    df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone(UTC).cast(pl.Datetime("us", UTC)))
    store.write_minutes(sym, df)
    store.write_daily(sym, minutes_to_daily(df))


def write_options(store: MarketStore, sym: str, d: date, quotes: list[dict]):
    """quotes: dicts with time (HH:MM ET), expiration, strike, right, bid, ask and optional
    delta/iv/bid_size/ask_size/volume/quote_time/underlying_price."""
    rows = []
    for q in quotes:
        ts = et_epoch(d, q["time"])
        rows.append({
            "ts": ts, "expiration": q.get("expiration", d), "strike": float(q["strike"]), "right": q["right"],
            "bid": float(q["bid"]), "ask": float(q["ask"]), "bid_size": q.get("bid_size", 500), "ask_size": q.get("ask_size", 500),
            "volume": q.get("volume", 1000), "open_interest": q.get("open_interest", 5000),
            "iv": q.get("iv", 0.2), "delta": q.get("delta"), "gamma": q.get("gamma", 0.0), "theta": q.get("theta", 0.0), "vega": q.get("vega", 0.0),
            "underlying_price": q.get("underlying_price", 100.0),
            "quote_ts": et_epoch(d, q["quote_time"]) if q.get("quote_time") else ts,
        })
    df = pl.DataFrame(rows).with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone(UTC).cast(pl.Datetime("us", UTC)),
        pl.from_epoch("quote_ts", time_unit="s").dt.replace_time_zone(UTC).cast(pl.Datetime("us", UTC)),
        pl.col("delta").cast(pl.Float64),
    )
    store.write_option_day(sym, d, df)


def make_store(tmp: Path, sym="TST", settlement="cash", exercise="european") -> MarketStore:
    st = MarketStore(tmp / "market")
    st.upsert_symbol(SymbolMeta(symbol=sym, kind="index" if settlement == "cash" else "etf", exercise=exercise, settlement=settlement, multiplier=100.0, provider="test"))
    return st


def chain(time: str, spec: dict, expiration: date, underlying: float = 100.0, **extra) -> list[dict]:
    """spec: {(strike, right): (bid, ask, delta)} -> quote rows at one snapshot."""
    out = []
    for (k, r), v in spec.items():
        if v is None:
            continue
        b, a, dl = v[:3]
        row = {"time": time, "expiration": expiration, "strike": k, "right": r, "bid": b, "ask": a, "delta": dl, "underlying_price": underlying}
        if len(v) > 3:
            row.update(v[3])
        row.update(extra)
        out.append(row)
    return out


def put_spread_strategy(**over) -> dict:
    s = {
        "name": "test put spread", "symbol": "TST",
        "legs": [
            {"id": "sp", "side": "sell", "right": "P", "dte": 0, "strike": {"method": "delta", "value": 0.20}},
            {"id": "lp", "side": "buy", "right": "P", "dte": 0, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}},
        ],
        "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
        "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}},
    }
    s.update(over)
    return s
