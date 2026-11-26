"""Faults in stored observations that can be recognised from the observations themselves."""
from __future__ import annotations

from datetime import date

import numpy as np

WARMUP_SECONDS = 900  # the opening minutes are legitimately heavy: nothing is judged before 15 minutes of the session have traded

# Feeds whose bar volume is the consolidated tape's own figure. There a single minute CAN hold a large share of the
# session: SPY's tape shows one-minute blocks of 8% to 18% of the day on monthly expiration Fridays (2018-09-21 10:43,
# 18.7 million shares; measured on 2,741 sessions of Alpaca's consolidated bars, where the sequential test below
# flagged 32 such real prints). The test is therefore applied only to feeds that are known to produce the artefact
# or whose origin is unknown.
CONSOLIDATED_FEEDS = frozenset({"alpaca", "polygon", "databento", "thetadata"})


def artefact_prone(provider: str | None) -> bool:
    """Whether a provider's intraday volume can carry the running-total artefact (see below)."""
    return (provider or "") not in CONSOLIDATED_FEEDS


def cumulative_volume_artefacts(volume: np.ndarray, bar_seconds: int) -> np.ndarray:
    """Bars of ONE session (in time order) whose volume is not that bar's volume but the day's running total.

    Some free intraday feeds occasionally return, in a single bar, everything traded so far that day (Yahoo's 1-minute
    bars: 5 to 9 bars a session; SPY on 2026-11-20 shows 64 million shares in the 13:38 minute of a 151 million share
    day). A real bar after the first quarter of an hour never carries more than half of everything traded before it,
    so the test is sequential: a bar larger than half of the clean running total is an artefact and is left out of
    that total. Measured on the stored SPY bars, what remains sums to 83% of the daily volume at 1 and at 5 minutes
    alike, where the raw 1-minute bars summed to 289%.

    Returns a boolean mask. The volume of a flagged bar is unknown: callers give it no weight, they do not invent one."""
    v = np.nan_to_num(np.asarray(volume, dtype=float), nan=0.0)
    out = np.zeros(len(v), dtype=bool)
    if bar_seconds > 300 or len(v) == 0:
        return out  # at coarser bars a single bar can legitimately hold a large share of the day
    warm = max(1, WARMUP_SECONDS // max(1, bar_seconds))
    cum = 0.0
    for i, x in enumerate(v):
        if i >= warm and cum > 0 and x > 0.5 * cum:
            out[i] = True
        else:
            cum += x
    return out


# ----------------------------------------------------------------------------- option trade bars: is the provider's session whole?
OPEN_SLACK = 5  # minutes after 09:30 by which SOME near-dated contract around the money has traded on any normal session
CLOSE_SLACK = 6  # minutes before the close after which some near-dated contract still trades
JUDGE_FROM = 10  # near-dated contracts with trades needed before a session's hours are judged at all
NEAR_DAYS = 2  # "near-dated": expiring within this many days of the session


def assess_trade_session(bars, d: date, asked: int, traded: int) -> dict | None:
    """Whether a provider's option trade history for one session covers the session.

    A provider can hold a session only in part (measured on Alpaca: 2024-01-23 stops at 11:02 New York time, 2024-02-02 is
    absent, and five early sessions start between 09:36 and 10:30). On such a session "no print at the open" is a hole in the
    history, not a quiet market, and a simulated entry would silently slide to the first print the provider happens to have.

    Judged only when enough was asked for that silence means absence: ten or more near-dated contracts with trades. Returns
    {"status": "partial", "reason", ...}, {"status": "complete", ...} when the hours are whole, or None when it cannot be judged."""
    import polars as pl

    from strata.core import calendar as cal
    if asked >= 2 * JUDGE_FROM and traded == 0:
        return {"status": "partial", "kind": "empty", "reason": f"the provider returned no trade for any of the {asked} contracts asked for: it has no option history for this session", "contracts_asked": asked}
    if bars is None or bars.is_empty():
        return None
    near = bars.filter((pl.col("expiration") - pl.lit(d)).dt.total_days() <= NEAR_DAYS)
    if near["occ"].n_unique() < JUDGE_FROM:
        return None
    et = near.select((pl.col("ts").dt.convert_time_zone("America/New_York").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.convert_time_zone("America/New_York").dt.minute().cast(pl.Int32)).alias("m"))["m"]
    first, last = int(et.min()), int(et.max())
    close = cal.session_close_time(d).hour * 60 + cal.session_close_time(d).minute
    hhmm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    info = {"first_bar": hhmm(first), "last_bar": hhmm(last), "near_dated_contracts": int(near["occ"].n_unique())}
    why = []
    if first > 570 + OPEN_SLACK:
        why.append(f"its first trade print of any near-dated contract is at {hhmm(first)} New York time")
    if last < close - CLOSE_SLACK:
        why.append(f"its last is at {hhmm(last)}, before the {hhmm(close)} close")
    if why:
        return {"status": "partial", "kind": "hours", "reason": "the provider's option history for this session is incomplete: " + " and ".join(why), **info}
    return {"status": "complete", **info}


def assess_trade_sessions(store, key, days=None) -> dict:
    """Judge the sessions of one option trade-bar dataset and record the partial ones in the dataset's session states (a session
    found whole is only recorded when that clears an earlier "partial"). Returns {iso date: verdict} for the partial sessions."""
    import polars as pl
    idx = store.option_bar_index(key.symbol, dataset=key)
    if idx.is_empty():
        return {}
    per = idx.group_by("date").agg(pl.len().alias("asked"), (pl.col("status") == "ok").sum().alias("traded"))
    want = set(days) if days is not None else None
    states = store.session_states(key)
    partial, entries = {}, {}
    for r in per.sort("date").iter_rows(named=True):
        d = r["date"]
        if want is not None and d not in want:
            continue
        bars = store.read_option_bars(key.symbol, d, dataset=key) if r["traded"] else None
        v = assess_trade_session(bars, d, int(r["asked"]), int(r["traded"]))
        prev = (states.get(d.isoformat()) or {}).get("status")
        if v is None:
            continue
        if v["status"] == "partial":
            partial[d.isoformat()] = v
            entries[d.isoformat()] = v
        elif prev == "partial":
            entries[d.isoformat()] = v
    if entries:
        store.mark_sessions(key, entries)
    return partial
