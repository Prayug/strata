"""Point-in-time market data access for the engine.

All accessors take the current decision time `t` (epoch seconds, UTC) and only return
information that was observable at or before `t`:
  * underlying price = close of the last 1-minute bar that *ended* at or before t
  * indicator values = value on the last aggregated bar that ended at or before t
  * daily indicators/features = values as of the previous completed session
  * option chain = the latest stored snapshot with ts <= t
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.core import indicators as ind
from strata.core.pricing import bs_greeks_vec, implied_vol_vec
from strata.data.derived import constant_maturity_iv
from strata.data.store import MarketStore

EPOCH = date(1970, 1, 1)


def day_num(d: date) -> int:
    return (d - EPOCH).days


def from_day_num(n: int) -> date:
    return EPOCH + timedelta(days=int(n))


def contract_key(exp_days, right_is_call, strike):
    """int64 key ordered like the stored files: expiration, then calls before puts, then strike."""
    return (np.asarray(exp_days, np.int64) * 2 + (1 - np.asarray(right_is_call, np.int64))) * 1_000_000_000 + np.round(np.asarray(strike, float) * 1000).astype(np.int64)


def exp_close_epoch(exp_days: np.ndarray) -> np.ndarray:
    """Epoch seconds of 16:00 ET (or the early close) on each expiration date."""
    out = np.empty(exp_days.shape[0], np.int64)
    cache: dict[int, int] = {}
    for i, e in enumerate(exp_days.tolist()):
        if e not in cache:
            d = from_day_num(e)
            cache[e] = int(cal.session_bounds(d)[1].timestamp())
        out[i] = cache[e]
    return out


class DayChain:
    """All option snapshots of one symbol for one session as sorted numpy arrays."""

    def __init__(self, df: pl.DataFrame, r: float = 0.04):
        df = df.with_columns(
            pl.col("ts").dt.epoch("s").alias("_t"),
            pl.col("expiration").cast(pl.Int32).alias("_e"),
            pl.when(pl.col("quote_ts").is_null()).then(pl.col("ts")).otherwise(pl.col("quote_ts")).dt.epoch("s").alias("_q"),
        )
        is_call = (df["right"] == "C").to_numpy()
        key = contract_key(df["_e"].to_numpy(), is_call, df["strike"].to_numpy())
        t_raw = df["_t"].to_numpy()
        dt, dk = np.diff(t_raw), np.diff(key)
        if np.all((dt > 0) | ((dt == 0) & (dk > 0))):
            order = np.arange(len(key))  # files are written pre-sorted; skip the sort
        else:
            order = np.lexsort((key, t_raw))
        g = lambda c, dt=float: np.nan_to_num(df[c].to_numpy().astype(dt), nan=np.nan)[order] if c in df.columns else None
        self.ts = df["_t"].to_numpy()[order].astype(np.int64)
        self.exp = df["_e"].to_numpy()[order].astype(np.int64)
        self.strike = df["strike"].to_numpy()[order].astype(float)
        self.is_call = is_call[order]
        self.key = key[order]
        self.bid = g("bid")
        self.ask = g("ask")
        last = df["last"].to_numpy().astype(float)[order] if "last" in df.columns else np.full(len(order), np.nan)
        # rows with a trade print but no quote: priced at the print, flagged trade-based.
        # A bid/ask spread is never invented for them.
        self.trade_based = (~np.isfinite(self.bid) | ~np.isfinite(self.ask)) & np.isfinite(last)
        if self.trade_based.any():
            self.bid = np.where(self.trade_based, last, self.bid)
            self.ask = np.where(self.trade_based, last, self.ask)
        self.bid_size = df["bid_size"].fill_null(-1).to_numpy()[order].astype(np.int64)
        self.ask_size = df["ask_size"].fill_null(-1).to_numpy()[order].astype(np.int64)
        self.volume = df["volume"].fill_null(-1).to_numpy()[order].astype(np.int64)
        self.oi = df["open_interest"].fill_null(-1).to_numpy()[order].astype(np.int64)
        self.quote_ts = df["_q"].to_numpy()[order].astype(np.int64)
        self.under = df["underlying_price"].to_numpy()[order].astype(float) if "underlying_price" in df.columns else np.full(len(order), np.nan)
        self.mid = (self.bid + self.ask) / 2
        self.iv = df["iv"].to_numpy()[order].astype(float)
        self.delta = df["delta"].to_numpy()[order].astype(float)
        self.gamma = df["gamma"].to_numpy()[order].astype(float)
        self.theta = df["theta"].to_numpy()[order].astype(float)
        self.vega = df["vega"].to_numpy()[order].astype(float)
        self.r = r
        self.kind = "quotes"  # "trades" when built from trade bars (set by the engine)
        self.contract_stats: dict = {}  # trade bars: per-contract session statistics (reporting only)
        self.multipliers: dict = {}  # contract key -> multiplier from the provider's contract definition
        self.snap_ts, self.snap_start = np.unique(self.ts, return_index=True)
        self.snap_end = np.r_[self.snap_start[1:], len(self.ts)]
        self.greeks_computed = False
        self._fill_missing_greeks()

    def _fill_missing_greeks(self):
        """Compute IV/Greeks from observed mid quotes where the provider omitted them."""
        need_iv = ~np.isfinite(self.iv)
        need_g = ~np.isfinite(self.delta)
        if not need_iv.any() and not need_g.any():
            return
        self.greeks_computed = True
        T = np.maximum((exp_close_epoch(self.exp) - self.ts) / (365 * 86400.0), 0.0)
        S = self.under
        if need_iv.any():
            ivs = implied_vol_vec(self.mid, S, self.strike, T, self.r, 0.0, self.is_call)
            self.iv = np.where(need_iv, ivs, self.iv)
        if need_g.any():
            sig = np.where(np.isfinite(self.iv), self.iv, 0.0)
            G = bs_greeks_vec(S, self.strike, T, self.r, 0.0, sig, self.is_call)
            self.delta = np.where(need_g, G[:, 1], self.delta)
            self.gamma = np.where(~np.isfinite(self.gamma), G[:, 2], self.gamma)
            self.theta = np.where(~np.isfinite(self.theta), G[:, 3], self.theta)
            self.vega = np.where(~np.isfinite(self.vega), G[:, 4], self.vega)

    def snapshot_at(self, t: int) -> int:
        """Index of the latest snapshot with ts <= t, or -1."""
        return int(np.searchsorted(self.snap_ts, t, side="right")) - 1

    def rows(self, si: int) -> slice:
        return slice(int(self.snap_start[si]), int(self.snap_end[si]))

    def find(self, si: int, key: int) -> int:
        s = self.rows(si)
        j = int(np.searchsorted(self.key[s], key))
        if j < s.stop - s.start and self.key[s.start + j] == key:
            return s.start + j
        return -1

    def row_dict(self, i: int) -> dict:
        return {
            "expiration": from_day_num(self.exp[i]).isoformat(), "strike": float(self.strike[i]), "right": "C" if self.is_call[i] else "P",
            "bid": float(self.bid[i]), "ask": float(self.ask[i]), "mid": float(self.mid[i]), "iv": _f(self.iv[i]), "delta": _f(self.delta[i]),
            "gamma": _f(self.gamma[i]), "theta": _f(self.theta[i]), "vega": _f(self.vega[i]), "volume": int(self.volume[i]),
            "open_interest": int(self.oi[i]), "bid_size": int(self.bid_size[i]), "ask_size": int(self.ask_size[i]),
            "quote_age": int(self.ts[i] - self.quote_ts[i]),
        }


def _f(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def expand_trade_bars(bars: pl.DataFrame, ticks: list[int], ser: "UnderlyingSeries", d: date) -> tuple[pl.DataFrame, dict]:
    """Per-contract option TRADE bars -> the chain as it could be observed at each engine tick.

    At tick T a contract appears only if it has printed today in a bar that ENDED at or
    before T, and it is priced at that bar's close; `quote_ts` is the end of that bar, so the
    age of the print is always known and stale prints can be refused. Nothing is carried
    over from earlier sessions, no bid/ask is created (rows carry `last` only) and a
    contract that has not traded yet does not exist in the chain. The underlying price
    attached to each row is the one at the print, not at the tick.

    Returns the snapshot frame and per-contract statistics for the whole session (used for
    reporting only, never for decisions)."""
    if bars is None or bars.is_empty() or not ticks:
        return pl.DataFrame(), {}
    T = np.asarray(sorted(ticks), np.int64)
    b = bars.sort(["occ", "ts"]).with_columns((pl.col("ts").dt.epoch("s") + 60).alias("_end"), pl.col("expiration").cast(pl.Int32).alias("_e"))
    close_t = int(cal.session_bounds(d)[1].timestamp())
    cols: dict[str, list] = {k: [] for k in ("ts", "expiration", "strike", "right", "last", "volume", "underlying_price", "quote_ts")}
    stats: dict[int, dict] = {}
    for (occ,), g in b.group_by(["occ"], maintain_order=True):
        end = np.minimum(g["_end"].to_numpy().astype(np.int64), close_t)
        px = g["close"].to_numpy().astype(float)
        cum = np.cumsum(g["volume"].fill_null(0).to_numpy().astype(np.int64))
        idx = np.searchsorted(end, T, side="right") - 1
        m = idx >= 0
        if not m.any():
            continue
        ii = idx[m]
        n = int(m.sum())
        e, k, r = int(g["_e"][0]), float(g["strike"][0]), str(g["right"][0])
        ui = np.searchsorted(ser.end, end[ii], side="right") - 1
        und = np.where(ui >= 0, ser.c[np.maximum(ui, 0)], np.nan)
        cols["ts"].append(T[m])
        cols["expiration"].append(np.full(n, e, np.int32))
        cols["strike"].append(np.full(n, k))
        cols["right"].append(np.full(n, r))
        cols["last"].append(px[ii])
        cols["volume"].append(cum[ii])
        cols["underlying_price"].append(und)
        cols["quote_ts"].append(end[ii])
        key = int(contract_key(np.array([e]), np.array([r == "C"]), np.array([k]))[0])
        stats[key] = {"occ": occ, "bars": int(len(end)), "volume": int(cum[-1]), "first_print": int(end[0]), "last_print": int(end[-1])}
    if not cols["ts"]:
        return pl.DataFrame(), stats
    df = pl.DataFrame({
        "ts": pl.from_epoch(pl.Series(np.concatenate(cols["ts"])), time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
        "expiration": pl.Series(np.concatenate(cols["expiration"])).cast(pl.Date),
        "strike": np.concatenate(cols["strike"]),
        "right": np.concatenate(cols["right"]),
        "last": np.concatenate(cols["last"]),
        "volume": np.concatenate(cols["volume"]),
        "underlying_price": np.concatenate(cols["underlying_price"]),
        "quote_ts": pl.from_epoch(pl.Series(np.concatenate(cols["quote_ts"])), time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
    })
    null = lambda t: pl.lit(None, dtype=t)
    df = df.with_columns(null(pl.Float64).alias("bid"), null(pl.Float64).alias("ask"), null(pl.Int32).alias("bid_size"), null(pl.Int32).alias("ask_size"), null(pl.Int64).alias("open_interest"),
                         null(pl.Float32).alias("iv"), null(pl.Float32).alias("delta"), null(pl.Float32).alias("gamma"), null(pl.Float32).alias("theta"), null(pl.Float32).alias("vega"))
    return df, stats


SCALE_BAND = 0.2  # an official close more than 20% from the last intraday bar means different price scales (an unadjusted split), not a late print
CLOSE_GAP_PCT = 0.25  # a gap larger than this between the last intraday bar and the official close is reported as a data-quality finding


class UnderlyingSeries:
    """Price bars for the whole run (+warmup) with point-in-time lookups.

    Bars may have mixed resolution (the finest stored per session: 1m ... 60m, or a single
    daily bar). Every bar carries its own end time; a bar is only visible once t >= end.
    """

    def __init__(self, df: pl.DataFrame, bar_seconds: int = 60, official: pl.DataFrame | None = None):
        df = df.sort("ts").with_columns(
            pl.col("ts").dt.epoch("s").alias("_t"),
            pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().cast(pl.Int32).alias("_d"),
        )
        self.ts = df["_t"].to_numpy().astype(np.int64)
        self.end = df["end"].dt.epoch("s").to_numpy().astype(np.int64) if "end" in df.columns else self.ts + bar_seconds
        self.o = df["open"].to_numpy().astype(float)
        self.h = df["high"].to_numpy().astype(float)
        self.l = df["low"].to_numpy().astype(float)
        self.c = df["close"].to_numpy().astype(float)
        self.v = df["volume"].fill_null(0).to_numpy().astype(float)
        self.day = df["_d"].to_numpy().astype(np.int64)
        self.volume_artefacts = 0
        self.bar_seconds = bar_seconds
        # nominal resolution of each bar in seconds (86400 marks a daily bar)
        self.res = df["res"].to_numpy().astype(np.int64) if "res" in df.columns else (self.end - self.ts)
        self.days, self.day_start = np.unique(self.day, return_index=True)
        self.day_end = np.r_[self.day_start[1:], len(self.ts)]
        n = len(self.ts)
        if n:
            # a bar that carries the day's running total instead of its own volume gets no weight (in VWAP, in volume sums): its volume is unknown
            # (only on feeds that can produce it: the consolidated tape's own volume is kept as it is, large blocks included)
            from strata.data.quality import artefact_prone, cumulative_volume_artefacts
            prov = df["provider"].to_list() if "provider" in df.columns else None
            fine = np.flatnonzero(self.res[self.day_start] <= 300)
            if fine.size:
                self.v = self.v.copy()
                for k in fine:
                    a, b = int(self.day_start[k]), int(self.day_end[k])
                    if prov is not None and not artefact_prone(prov[a]):
                        continue
                    bad = cumulative_volume_artefacts(self.v[a:b], int(self.res[a]))
                    if bad.any():
                        self.v[a:b][bad] = 0.0
                        self.volume_artefacts += int(bad.sum())
        self.d_o = self.o[self.day_start] if n else np.array([])
        self.d_c = self.c[self.day_end - 1] if n else np.array([])
        self.d_h = np.maximum.reduceat(self.h, self.day_start) if n else np.array([])
        self.d_l = np.minimum.reduceat(self.l, self.day_start) if n else np.array([])
        self.d_v = np.add.reduceat(self.v, self.day_start) if n else np.array([])
        self.d_end = self.end[self.day_end - 1] if n else np.array([], np.int64)
        self.d_open_t = self.ts[self.day_start] if n else np.array([], np.int64)
        # finest bar length (seconds) available in each session
        self.d_res = np.minimum.reduceat(self.res, self.day_start) if n else np.array([], np.int64)
        self.official_closes = 0
        self.close_gaps: list[tuple[int, float, float]] = []  # (day number, close of the last intraday bar, official close) where they differ by more than CLOSE_GAP_PCT
        self.scale_mismatches = 0
        if official is not None and n and not official.is_empty():
            # Session closes come from the daily record (the official closing print) wherever one exists: day-over-day
            # returns, "fell 3% yesterday" and expiration settlement must match the published close. The last intraday
            # bar usually agrees to a few cents, but not always: free intraday feeds can end before the closing auction
            # (SPY, 2025-04-09: last hourly bar 542.84, official close 548.62). The official close wins unless the two are
            # on different price scales (an unadjusted split), which only a large ratio can indicate. Intraday bars are untouched.
            od = official["date"].cast(pl.Int32).to_numpy().astype(np.int64)
            oc = official["close"].to_numpy().astype(float)
            idx = np.searchsorted(od, self.days)
            ok = (idx < len(od)) & (self.d_res < 86400)
            ok[ok] &= od[idx[ok]] == self.days[ok]
            ok[ok] &= np.isfinite(oc[idx[ok]])
            ratio = np.ones(len(self.days))
            ratio[ok] = oc[idx[ok]] / self.d_c[ok]
            same_scale = np.abs(ratio - 1) < SCALE_BAND
            self.scale_mismatches = int((ok & ~same_scale).sum())
            ok &= same_scale
            for i in np.flatnonzero(ok & (np.abs(ratio - 1) * 100 > CLOSE_GAP_PCT)):
                self.close_gaps.append((int(self.days[i]), float(self.d_c[i]), float(oc[idx[i]])))
            self.d_c = self.d_c.copy()
            self.d_c[ok] = oc[idx[ok]]
            # the official close is a print of the session: its range cannot exclude it
            self.d_h = np.where(ok, np.maximum(self.d_h, self.d_c), self.d_h)
            self.d_l = np.where(ok, np.minimum(self.d_l, self.d_c), self.d_l)
            self.official_closes = int(ok.sum())
        self._tf_cache: dict = {}
        self._ind_cache: dict = {}
        self._ind_days: dict = {}
        self._memo: dict = {}

    def empty(self) -> bool:
        return len(self.ts) == 0

    def rescale(self, k: float) -> None:
        """Re-express all prices in different share terms (used across stock splits so
        history stays continuous while today's prices match as-traded option strikes)."""
        if abs(k - 1.0) < 1e-12:
            return
        for name in ("o", "h", "l", "c", "d_o", "d_h", "d_l", "d_c"):
            setattr(self, name, getattr(self, name) * k)
        self._tf_cache.clear()
        self._ind_cache.clear()
        self._memo.clear()

    # ----- index helpers
    def day_index(self, d: date) -> int:
        n = day_num(d)
        i = int(np.searchsorted(self.days, n))
        return i if i < len(self.days) and self.days[i] == n else -1

    def prev_day_index(self, d: date) -> int:
        """Index of the last session strictly before d."""
        return int(np.searchsorted(self.days, day_num(d))) - 1

    def completed_day(self, t: int) -> int:
        """Index of the last session whose final bar has ended by t."""
        return int(np.searchsorted(self.d_end, t, side="right")) - 1

    def resolution(self, d: date) -> int:
        di = self.day_index(d)
        return int(self.d_res[di]) if di >= 0 else 0

    def last_bar(self, t: int) -> int:
        return int(np.searchsorted(self.end, t, side="right")) - 1

    def price_at(self, t: int, d: date | None = None) -> float:
        i = self.last_bar(t)
        if d is not None:
            di = self.day_index(d)
            if di >= 0:
                s = self.day_start[di]
                if i < s and self.ts[s] <= t:
                    return float(self.o[s])  # first trade of the session is known at the open
        return float(self.c[i]) if i >= 0 else float("nan")

    def session_open(self, d: date) -> float:
        di = self.day_index(d)
        return float(self.o[self.day_start[di]]) if di >= 0 else float("nan")

    def session_close(self, d: date) -> float:
        """The session's closing price: the official close where a daily record exists (settlement uses this), else the last bar's close."""
        di = self.day_index(d)
        return float(self.d_c[di]) if di >= 0 else float("nan")

    def range_between(self, t0: int, t1: int) -> tuple[float, float]:
        """High/low of bars that completed in (t0, t1]."""
        a = int(np.searchsorted(self.end, t0, side="right"))
        b = int(np.searchsorted(self.end, t1, side="right"))
        if b <= a:
            return float("nan"), float("nan")
        return float(self.h[a:b].max()), float(self.l[a:b].min())

    def today_slice(self, d: date, t: int) -> tuple[int, int]:
        """[start, stop) of today's bars that have completed by t."""
        di = self.day_index(d)
        if di < 0:
            return 0, 0
        s, e = self.day_start[di], self.day_end[di]
        return int(s), int(s + np.searchsorted(self.end[s:e], t, side="right"))

    def session_range_until(self, d: date, t: int, minutes: int | None = None) -> tuple[float, float, bool]:
        """High/low of today's bars that ended by t (optionally only the first `minutes`).
        Returns (high, low, complete) where complete means the window has fully elapsed."""
        di = self.day_index(d)
        if di < 0:
            return float("nan"), float("nan"), False
        s, e = self.day_start[di], self.day_end[di]
        lim = self.ts[s] + minutes * 60 if minutes else None
        b = int(np.searchsorted(self.end[s:e], t if lim is None else min(t, lim), side="right"))
        if b == 0:
            return float("nan"), float("nan"), False
        complete = lim is not None and t >= lim
        return float(self.h[s:s + b].max()), float(self.l[s:s + b].min()), complete

    # ----- aggregated bars + indicators -----
    def _tf(self, minutes: int):
        if minutes in self._tf_cache:
            return self._tf_cache[minutes]
        mins_from_open = (self.ts - self.ts[np.repeat(self.day_start, self.day_end - self.day_start)]) // 60
        bucket = self.day * 10_000 + mins_from_open // max(minutes, 1)
        uniq, start = np.unique(bucket, return_index=True)
        end_idx = np.r_[start[1:], len(self.ts)]
        res = (self.ts[start], self.end[end_idx - 1], self.o[start], np.maximum.reduceat(self.h, start), np.minimum.reduceat(self.l, start),
               self.c[end_idx - 1], np.add.reduceat(self.v, start), self.day[start])
        self._tf_cache[minutes] = res
        return res

    @staticmethod
    def tf_minutes(tf: str) -> int:
        return int(tf[:-1]) * (60 if tf.endswith("h") else 1)

    def indicator(self, name: str, params: dict, tf: str, field: str | None, t: int, d: date, offset: int = 0) -> float:
        key = (name, json.dumps(params or {}, sort_keys=True), tf)
        if key not in self._ind_cache:
            if tf == "1d":
                vals = ind.compute(name, params, self.d_o, self.d_h, self.d_l, self.d_c, self.d_v, np.zeros(len(self.d_c), np.int64))
                self._ind_cache[key] = (self.d_end, vals)
            else:
                ts, end, o, h, l, c, v, dd = self._tf(self.tf_minutes(tf))
                vals = ind.compute(name, params, o, h, l, c, v, dd)
                self._ind_cache[key] = (end, vals)
                self._ind_days[key] = dd
        end, vals = self._ind_cache[key]
        if tf != "1d" and self.resolution(d) > self.tf_minutes(tf) * 60:
            return float("nan")  # the session's bars are coarser than the requested timeframe
        fld = field or next(iter(vals))
        arr = vals[fld]
        i = int(np.searchsorted(end, t, side="right")) - 1 - offset
        if i < 0 or i >= len(arr):
            return float("nan")
        # a session-anchored indicator has no value before the session's first bar has completed: at the opening tick the
        # last completed bar belongs to the PREVIOUS session, and its VWAP is not "today's VWAP so far"
        if name in ind.SESSION_ANCHORED and tf != "1d" and int(self._ind_days[key][i]) != day_num(d):
            return float("nan")
        return float(arr[i])

    def daily_sma(self, d: date, n: int) -> float:
        i = self.prev_day_index(d)
        if i + 1 < n:
            return float("nan")
        return float(self.d_c[i - n + 1:i + 1].mean())


@dataclass
class IvHistory:
    dates: np.ndarray  # day numbers
    iv: np.ndarray

    @classmethod
    def load(cls, store: MarketStore, sym: str, dataset=None) -> "IvHistory":
        df = store.read_iv_history(sym, dataset)
        if df.is_empty():
            return cls(np.array([], np.int64), np.array([]))
        df = df.filter(pl.col("atm_iv_30d").is_not_null())
        return cls(df["date"].cast(pl.Int32).to_numpy().astype(np.int64), df["atm_iv_30d"].to_numpy().astype(float))

    def window(self, d: date, lookback: int = 252) -> np.ndarray:
        i = int(np.searchsorted(self.dates, day_num(d)))  # strictly before d
        return self.iv[max(0, i - lookback):i]


def atm_iv_from_chain(ch: DayChain, si: int, spot: float, target_days: float = 30.0) -> float | None:
    s = ch.rows(si)
    t = ch.snap_ts[si]
    exps = np.unique(ch.exp[s])
    pts = []
    closes = exp_close_epoch(exps)
    for e, ce in zip(exps, closes):
        days = (ce - t) / 86400.0
        if days <= 0.05:
            continue
        m = (ch.exp[s] == e) & np.isfinite(ch.iv[s]) & (ch.bid[s] > 0)
        if not m.any():
            continue
        ks = ch.strike[s][m]
        k = ks[np.argmin(np.abs(ks - spot))]
        sel = m & (ch.strike[s] == k)
        pts.append((days, float(np.nanmean(ch.iv[s][sel]))))
    return constant_maturity_iv(pts, target_days)


class Events:
    def __init__(self, df: pl.DataFrame):
        self.by_date: dict[date, list[dict]] = {}
        for r in df.iter_rows(named=True):
            self.by_date.setdefault(r["date"], []).append(r)

    def on(self, d: date, types: list[str] | None = None) -> list[dict]:
        evs = self.by_date.get(d, [])
        if types:
            evs = [e for e in evs if e["event_type"] in types]
        return evs

    def within(self, d: date, n_sessions: int, types: list[str] | None = None) -> list[dict]:
        out, dd = [], d
        for _ in range(n_sessions + 1):
            out += self.on(dd, types)
            dd = cal.next_trading_day(dd)
        return out
