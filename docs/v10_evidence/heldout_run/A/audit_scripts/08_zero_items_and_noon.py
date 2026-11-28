"""The 'outcome happened on 0 of N days' items (type movement, outcome end_return) and the 12:00 item.

alex/ah-18#1   SPY  start 14:30 -> close                     Strata 0 of 2,719
riley/rh-01#1  IWM  open -> close, since 2000-01-01          Strata 0 of 6,665
alex/ah-15#1   SPY  open -> close, since 2016-01-01          Strata 0 of 2,742
riley/rh-04#1  SPY  open -> close                            Strata 0 of 8,515
jordan/jh-21#2 SPY  gap<0, open -> close                     Strata 0 of 3,698
jordan/jh-21#6 QQQ  gap<0, open -> close                     Strata 0 of 3,056
sam/sh-02#1    SPY  below the open at 12:00 -> closes above its open     Strata 1,487 of 2,741 (any day 54.3% of 2,741)
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import numpy as np
import polars as pl
from common import base, minute, wilson, pct

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def move_stats(df, start_col, label):
    r = (df["close"] / df[start_col] - 1).to_numpy()
    n = len(r)
    up, down, flat = int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())
    print(f"  {label}: n={n}  mean {100 * r.mean():+.3f}%  median {100 * np.median(r):+.3f}%  "
          f"ended higher {up} ({100 * up / n:.1f}%)  lower {down} ({100 * down / n:.1f}%)  unchanged {flat}")
    return n, r.mean(), np.median(r), up, down, flat


def price_at(m, mod):
    """Close of the last one-minute bar that ended by `mod` (i.e. the bar starting at mod-1), per session."""
    return m.filter(pl.col("mod") == mod - 1).select("date", px="close")


# ---------------- open -> close items
for key, sym, since, gapneg in [
    ("riley/rh-01#1", "IWM", dt.date(2000, 1, 1), False),
    ("alex/ah-15#1", "SPY", dt.date(2016, 1, 1), False),
    ("riley/rh-04#1", "SPY", None, False),
    ("jordan/jh-21#2", "SPY", None, True),
    ("jordan/jh-21#6", "QQQ", None, True),
]:
    print("=" * 100)
    print(key, sym, "since", since, "gap<0" if gapneg else "")
    b = base(sym)
    v = b.filter(pl.col("prev_close").is_not_null())
    if since:
        v = v.filter(pl.col("date") >= since)
    print("  sessions with a previous close in the period:", v.height, v["date"].min(), v["date"].max())
    if gapneg:
        g_s = v.filter(pl.col("s_open") < pl.col("prev_close"))
        g_d = v.filter(pl.col("open") < pl.col("prev_close"))
        g_le = v.filter(pl.col("s_open") <= pl.col("prev_close"))
        print(f"  opened below the previous close: session open (09:30 bar where whole) {g_s.height}; daily-record open {g_d.height}; (<=: {g_le.height})")
        v = g_s
    move_stats(v, "s_open", "open -> close (session open per the stated convention, daily close)")
    move_stats(v, "open", "open -> close (daily-record open)")

# ---------------- 14:30 -> close
print("=" * 100)
print("alex/ah-18#1 SPY start 14:30 -> close")
b = base("SPY")
m = minute("SPY")
whole = b.filter(pl.col("whole")).select("date", "close", "s_open", "m_close", "half", "last")
print("  whole minute sessions:", whole.height, " of which early closes:", whole.filter(pl.col("half")).height)
p1430 = price_at(m, 14 * 60 + 30)
j = whole.join(p1430, on="date", how="inner")
print("  whole sessions with a price at 14:30 (the 14:29 bar):", j.height)
j_full = j.filter(~pl.col("half"))
print("  ... excluding early closes:", j_full.height)
has_1430_bar = whole.join(m.filter(pl.col("mod") == 14 * 60 + 30).select("date", b1430="open"), on="date", how="inner")
print("  whole sessions with a 14:30 bar:", has_1430_bar.height)
move_stats(j, "px", "14:30 price (close of the 14:29 bar) -> daily close")
jm = j.with_columns(close=pl.col("m_close"))
move_stats(jm, "px", "14:30 price -> last minute bar's close")
j2 = has_1430_bar.with_columns(px=pl.col("b1430"))
move_stats(j2, "px", "open of the 14:30 bar -> daily close")

# ---------------- 12:00 item
print("=" * 100)
print("sam/sh-02#1 SPY: below the open at 12:00 -> closes above its open")
p12 = price_at(m, 12 * 60)
j = whole.join(p12, on="date", how="left")
print("  whole sessions:", j.height, " with a price at 12:00:", j.filter(pl.col("px").is_not_null()).height)
allk = j.filter(pl.col("close") > pl.col("s_open")).height
lo, hi = wilson(allk, j.height)
print(f"  ANY whole session: daily close > session open: {allk} of {j.height} = {pct(allk / j.height)}  Wilson {pct(lo)}..{pct(hi)}")
allk_m = j.filter(pl.col("m_close") > pl.col("s_open")).height
print(f"  [ref] last minute close > session open: {allk_m} of {j.height} = {pct(allk_m / j.height)}")
below = j.filter(pl.col("px") < pl.col("s_open"))
k = below.filter(pl.col("close") > pl.col("s_open")).height
lo, hi = wilson(k, below.height)
print(f"  below the open at 12:00 (close of the 11:59 bar < 09:30 open): {below.height} sessions; of those the daily close is above the open on {k} = {pct(k / below.height)}  Wilson {pct(lo)}..{pct(hi)}")
k_m = below.filter(pl.col("m_close") > pl.col("s_open")).height
print(f"     [ref] using the last minute bar's close: {k_m} = {pct(k_m / below.height)}")
at_or_below = j.filter(pl.col("px") <= pl.col("s_open"))
print(f"     [ref] at or below at 12:00: {at_or_below.height}; closes above open {at_or_below.filter(pl.col('close') > pl.col('s_open')).height}")
above = j.filter(pl.col("px") > pl.col("s_open"))
print(f"     [ref] above the open at 12:00: {above.height}; closes above open {above.filter(pl.col('close') > pl.col('s_open')).height}")
# measured from 12:00 instead: closes above the 12:00 price
k12 = j.filter(pl.col("close") > pl.col("px")).height
print(f"  [ref] daily close above the 12:00 price: {k12} of {j.filter(pl.col('px').is_not_null()).height}")
# 12:00 bar open variant
p12b = m.filter(pl.col("mod") == 12 * 60).select("date", px2="open")
j3 = whole.join(p12b, on="date", how="left")
below2 = j3.filter(pl.col("px2") < pl.col("s_open"))
print(f"     [ref] below using the 12:00 bar's open: {below2.height}; closes above open {below2.filter(pl.col('close') > pl.col('s_open')).height}")
