"""Daily-bar frequency items: streaks, closes above previous close, previous high/low relations.

Items: casey/ch-21#1, #2, #3 ; alex/ah-01#1 ; morgan/mh-01#1 ; sam/sh-21#1 ; alex/ah-27#1
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import polars as pl
from common import base, streaks, wilson, pct

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def show(label, k, n):
    lo, hi = wilson(k, n)
    print(f"  {label}: {k} of {n} = {pct(k / n)} (2dp {100 * k / n:.2f}%)   Wilson95 {pct(lo)} .. {pct(hi)}   didn't {n - k}")


def streak_item(sym, thr, key):
    print("=" * 90)
    print(key, sym, f"up_streak>={thr}, outcome close>prev_close")
    b = base(sym)
    up, dn = streaks(b["close"].to_list())
    b = b.with_columns(up=pl.Series(up), dn=pl.Series(dn))
    b = b.with_columns(up_prev=pl.col("up").shift(1), dn_prev=pl.col("dn").shift(1))
    v = b.filter(pl.col("prev_close").is_not_null())
    print("  days with a previous close:", v.height)
    allk = v.filter(pl.col("close") > pl.col("prev_close")).height
    show("any day close>prev_close", allk, v.height)
    c = v.filter(pl.col("up_prev") >= thr)
    k = c.filter(pl.col("close") > pl.col("prev_close")).height
    show(f"after up-streak>={thr} (counted up to the previous session)", k, c.height)
    # alternative: streak counted including today (wrong reading) for reference
    c2 = v.filter(pl.col("up") >= thr)
    print("  [ref] sessions that themselves end a streak>=thr:", c2.height)
    eq = v.filter(pl.col("close") == pl.col("prev_close")).height
    print("  [ref] days with close == prev_close:", eq)
    print("  first/last matching day:", c["date"].min(), c["date"].max())


streak_item("IWM", 5, "casey/ch-21#3")
streak_item("SPY", 3, "casey/ch-21#1")
streak_item("SPY", 5, "casey/ch-21#2")

print("=" * 90)
print("alex/ah-01#1 QQQ closes above the previous session's close")
b = base("QQQ")
v = b.filter(pl.col("prev_close").is_not_null())
show("close>prev_close", v.filter(pl.col("close") > pl.col("prev_close")).height, v.height)

print("=" * 90)
print("morgan/mh-01#1 SPY: high>prev_high at some point -> close<prev_high")
b = base("SPY")
v = b.filter(pl.col("prev_close").is_not_null())
for name, hi_col in [("daily-record high", pl.col("high")),
                     ("minute high where whole session else daily", pl.when(pl.col("whole")).then(pl.col("m_high")).otherwise(pl.col("high")))]:
    c = v.filter(hi_col > pl.col("prev_high"))
    k = c.filter(pl.col("close") < pl.col("prev_high")).height
    show(f"[{name}] close<prev_high given high>prev_high", k, c.height)
    k2 = c.filter(pl.col("close") <= pl.col("prev_high")).height
    print(f"      (<= instead of <: {k2}); first {c['date'].min()} last {c['date'].max()}")
    c3 = v.filter(hi_col >= pl.col("prev_high"))
    print(f"      (cond >= instead of >: {c3.height}, of which close<prev_high {c3.filter(pl.col('close') < pl.col('prev_high')).height})")

print("=" * 90)
print("sam/sh-21#1 SPY since 2016-01-01: low<prev_low at some point -> close>prev_close")
v2 = v.filter(pl.col("date") >= dt.date(2016, 1, 1))
print("  sessions since 2016-01-01:", v2.height, v2["date"].min(), v2["date"].max())
for name, lo_col in [("daily-record low", pl.col("low")),
                     ("minute low where whole session else daily", pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low")))]:
    c = v2.filter(lo_col < pl.col("prev_low"))
    k = c.filter(pl.col("close") > pl.col("prev_close")).height
    show(f"[{name}] close>prev_close given low<prev_low", k, c.height)
    print(f"      first {c['date'].min()} last {c['date'].max()}")
    c3 = v2.filter(lo_col <= pl.col("prev_low"))
    print(f"      (cond <= instead of <: {c3.height}, of which close>prev_close {c3.filter(pl.col('close') > pl.col('prev_close')).height})")

print("=" * 90)
print("alex/ah-27#1 SPY: how often it trades below the previous session's low (all history)")
m_lo = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
m_lo_any = pl.when(pl.col("m_low").is_not_null()).then(pl.col("m_low")).otherwise(pl.col("low"))
variants = {
    "daily low <  prev daily low": pl.col("low") < pl.col("prev_low"),
    "daily low <= prev daily low": pl.col("low") <= pl.col("prev_low"),
    "minute low (whole sessions) else daily low <  prev daily low": m_lo < pl.col("prev_low"),
    "minute low (whole sessions) else daily low <= prev daily low": m_lo <= pl.col("prev_low"),
    "minute low (any minute session) else daily low <  prev daily low": m_lo_any < pl.col("prev_low"),
    "minute low (any minute session) else daily low <= prev daily low": m_lo_any <= pl.col("prev_low"),
}
for name, e in variants.items():
    show(name, v.filter(e).height, v.height)
# where the two sources disagree
dis = v.filter((pl.col("low") < pl.col("prev_low")) != (m_lo < pl.col("prev_low")))
print("  sessions where the daily-record low and the minute-bar low give a different answer (strict <):", dis.height)
print(dis.select("date", "open", "low", "prev_low", "m_low", "n", "whole").with_columns(
    daily_says=pl.col("low") < pl.col("prev_low"), minute_says=pl.col("m_low") < pl.col("prev_low"),
    d_minus_prev=pl.col("low") - pl.col("prev_low"), m_minus_prev=pl.col("m_low") - pl.col("prev_low")))
dis2 = v.filter((pl.col("low") <= pl.col("prev_low")) != (m_lo <= pl.col("prev_low")))
print("  same with <= :", dis2.height)
eq = v.filter(pl.col("low") == pl.col("prev_low"))
print("  sessions with daily low exactly equal to prev daily low:", eq.height, " of which before 2016:", eq.filter(pl.col("date") < dt.date(2016, 1, 1)).height)
eqm = v.filter(pl.col("whole") & (pl.col("m_low") == pl.col("prev_low")))
print("  whole minute sessions with minute low exactly equal to prev daily low:", eqm.height)
