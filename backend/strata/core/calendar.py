"""NYSE/Cboe trading calendar: holidays, early closes, DST-aware session times.

Rules are implemented from the published NYSE holiday schedule so the calendar is
deterministic and does not need a network lookup. Special one-off closures are listed
explicitly.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

SPECIAL_CLOSURES = {
    date(1994, 4, 27),  # President Nixon funeral
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),  # President Reagan funeral
    date(2007, 1, 2),  # President Ford funeral
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),  # President G.H.W. Bush funeral
    date(2025, 1, 9),  # President Carter funeral
}


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[date]:
    hs: set[date] = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        hs.add(date(year, 1, 2))
    elif ny.weekday() != 5:  # NYSE does not observe New Year's on the prior Friday
        hs.add(ny)
    if year >= 1998:
        hs.add(_nth_weekday(year, 1, 0, 3))  # MLK
    hs.add(_nth_weekday(year, 2, 0, 3))  # Presidents' Day
    hs.add(_easter(year) - timedelta(days=2))  # Good Friday
    hs.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        hs.add(_observed(date(year, 6, 19)))
    hs.add(_observed(date(year, 7, 4)))
    hs.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    hs.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    hs.add(_observed(date(year, 12, 25)))
    hs |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return frozenset(x for x in hs if x is not None)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


@lru_cache(maxsize=None)
def early_closes(year: int) -> frozenset[date]:
    out = set()
    j3 = date(year, 7, 3)
    if j3.weekday() < 4 and is_trading_day(j3):  # Mon-Thu July 3 is a half day
        out.add(j3)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    out.add(thanksgiving + timedelta(days=1))
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5 and is_trading_day(xmas_eve):
        out.add(xmas_eve)
    return frozenset(out)


def session_close_time(d: date) -> time:
    return EARLY_CLOSE if d in early_closes(d.year) else REGULAR_CLOSE


def session_bounds(d: date) -> tuple[datetime, datetime]:
    """Session open/close as timezone-aware ET datetimes (DST handled by zoneinfo)."""
    return (datetime.combine(d, REGULAR_OPEN, tzinfo=ET), datetime.combine(d, session_close_time(d), tzinfo=ET))


def session_minutes(d: date) -> int:
    o, c = session_bounds(d)
    return int((c - o).total_seconds() // 60)


def trading_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def next_trading_day(d: date, n: int = 1) -> date:
    step = 1 if n >= 0 else -1
    k = abs(n)
    while k:
        d += timedelta(days=step)
        if is_trading_day(d):
            k -= 1
    return d


def prev_trading_day(d: date) -> date:
    return next_trading_day(d, -1)


def trading_days_between(a: date, b: date) -> int:
    """Number of trading sessions in (a, b]."""
    if b <= a:
        return 0
    return len(trading_days(a + timedelta(days=1), b))


def to_et(ts_utc_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_utc_ns / 1e9, tz=UTC).astimezone(ET)


def et_to_utc_ns(dt_et: datetime) -> int:
    return int(dt_et.astimezone(UTC).timestamp() * 1_000_000_000)


def minute_of_day_et(ts_utc_ns: int) -> int:
    t = to_et(ts_utc_ns)
    return t.hour * 60 + t.minute


def parse_hhmm(s: str) -> int:
    h, m = s.strip().split(":")[:2]
    return int(h) * 60 + int(m)


def fmt_hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"
