"""Auditor's own loaders. Reads the raw price files only; nothing from Strata."""
import glob
import math

import polars as pl

DATA = "/Users/prayugsigdel/Coding/strata/data"

OPEN_MOD = 570   # 09:30 New York, minutes after midnight
CLOSE_MOD = 960  # 16:00
HALF_CLOSE_MOD = 780  # 13:00 on scheduled early closes


def daily(sym: str) -> pl.DataFrame:
    p = f"{DATA}/market/datasets/yahoo/{sym}/underlying/trade/1d/v1/data/daily.parquet"
    return pl.read_parquet(p).sort("date")


def vix() -> pl.DataFrame:
    paths = glob.glob(f"{DATA}/market/datasets/cboe/VIX/underlying/*/1d/v1/data/daily.parquet")
    assert len(paths) == 1, paths
    return pl.read_parquet(paths[0]).sort("date")


def minute(sym: str) -> pl.DataFrame:
    """All one-minute bars with New York local date and minute-of-day (mod)."""
    files = sorted(glob.glob(f"{DATA}/market/datasets/alpaca/{sym}/underlying/trade/1m/v1/data/*.parquet"))
    m = pl.concat([pl.read_parquet(f) for f in files]).sort("ts")
    m = m.with_columns(ny=pl.col("ts").dt.convert_time_zone("America/New_York"))
    m = m.with_columns(
        date=pl.col("ny").dt.date(),
        mod=(pl.col("ny").dt.hour().cast(pl.Int32) * 60 + pl.col("ny").dt.minute().cast(pl.Int32)),
    )
    return m.filter((pl.col("mod") >= OPEN_MOD) & (pl.col("mod") < CLOSE_MOD))


def session_table(m: pl.DataFrame) -> pl.DataFrame:
    """One row per minute-bar session: first/last bar, count, and OHLC from the minute bars.

    whole = the bars run from 09:30 to the session's last minute: 15:59 on a normal day, 12:59 on a
    scheduled 13:00 early close.  (The only session this drops for SPY and IWM is 2019-08-12, whose
    bars stop at 15:30/15:31.)
    """
    s = m.group_by("date").agg(
        n=pl.len(), first=pl.col("mod").min(), last=pl.col("mod").max(),
        m_open=pl.col("open").sort_by("mod").first(),
        m_high=pl.col("high").max(), m_low=pl.col("low").min(),
        m_close=pl.col("close").sort_by("mod").last(),
    ).sort("date")
    s = s.with_columns(
        half=(pl.col("last") == HALF_CLOSE_MOD - 1),
        whole=(pl.col("first") == OPEN_MOD) & ((pl.col("last") == CLOSE_MOD - 1) | (pl.col("last") == HALF_CLOSE_MOD - 1)),
    )
    return s


def base(sym: str) -> pl.DataFrame:
    """Daily record joined with the minute-session table, with Strata's stated session open:
    the 09:30 minute bar's open where a whole session of minute bars is stored, else the daily open.
    The close is always the daily record's close."""
    d = daily(sym)
    s = session_table(minute(sym))
    j = d.join(s, on="date", how="left").sort("date")
    j = j.with_columns(whole=pl.col("whole").fill_null(False))
    j = j.with_columns(
        s_open=pl.when(pl.col("whole")).then(pl.col("m_open")).otherwise(pl.col("open")),
        prev_close=pl.col("close").shift(1),
        prev_high=pl.col("high").shift(1),
        prev_low=pl.col("low").shift(1),
        prev_open=pl.col("open").shift(1),
    )
    j = j.with_columns(
        ret=pl.col("close") / pl.col("prev_close") - 1,
    )
    j = j.with_columns(prev_ret=pl.col("ret").shift(1))
    return j


def streaks(close: list[float]) -> tuple[list[int], list[int]]:
    """Up/down streak ending AT each session: the number of consecutive closes above/below the close before."""
    up, dn = [0] * len(close), [0] * len(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            up[i] = up[i - 1] + 1
        elif close[i] < close[i - 1]:
            dn[i] = dn[i - 1] + 1
    return up, dn


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"
