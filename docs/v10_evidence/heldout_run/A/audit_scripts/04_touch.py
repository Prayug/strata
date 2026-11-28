"""Touch items.
riley/rh-22#6  QQQ touch:+1.5 (open -> close), all history      Strata 1,163 of 6,973
casey/ch-18#1  QQQ touch:+1   (open -> close), all history      Strata 1,202 happened, 4,991 didn't, 780 undecidable
morgan/mh-23#2 SPY touch:+5   (open -> close), all history      Strata 24 of 8,515
casey/ch-07#1  IWM gap>=1, touch:prev_close, since 2016-01-01   Strata 54 of 220
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import polars as pl
from common import base, wilson, pct

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def show(label, k, n):
    lo, hi = wilson(k, n)
    print(f"  {label}: {k} of {n} = {pct(k / n)} ({100 * k / n:.2f}%)  Wilson95 {pct(lo)}..{pct(hi)}  didn't {n - k}")


def touch_up(sym, x, key):
    print("=" * 100)
    print(key, sym, f"touch +{x}% from the open, open -> close")
    b = base(sym)
    v = b.filter(pl.col("prev_close").is_not_null())
    f = 1 + x / 100
    hi_min = pl.when(pl.col("whole")).then(pl.col("m_high")).otherwise(pl.col("high"))
    variants = {
        "daily open, daily high (>=)": pl.col("high") >= pl.col("open") * f,
        "daily open, daily high (>)": pl.col("high") > pl.col("open") * f,
        "session open (09:30 bar where whole), minute high where whole else daily (>=)": hi_min >= pl.col("s_open") * f,
        "session open (09:30 bar where whole), minute high where whole else daily (>)": hi_min > pl.col("s_open") * f,
        "session open, daily high (>=)": pl.col("high") >= pl.col("s_open") * f,
        "daily open, minute high where whole (>=)": hi_min >= pl.col("open") * f,
    }
    for name, e in variants.items():
        show(name, v.filter(e).height, v.height)
    # all rows including the very first (no previous close needed for this question)
    print("  [ref] including the first stored day: daily", b.filter(pl.col("high") >= pl.col("open") * f).height, "of", b.height)
    return b, v


touch_up("QQQ", 1.5, "riley/rh-22#6")
touch_up("SPY", 5.0, "morgan/mh-23#2")
bq, vq = touch_up("QQQ", 1.0, "casey/ch-18#1")

print()
print("casey/ch-18#1: Strata reports 1,202 happened / 4,991 didn't / 780 'order inside one bar cannot be known'.")
print("Looking for a second level whose same-bar collision with +1% gives those numbers (daily bars before 2016):")
pre = vq.filter(~pl.col("whole"))
post = vq.filter(pl.col("whole"))
print("  days without whole minute session:", pre.height, " with:", post.height)
up_pre = pre.filter(pl.col("high") >= pl.col("open") * 1.01)
print("  pre: touch +1% (daily):", up_pre.height)
up_post = post.filter(pl.col("m_high") >= pl.col("s_open") * 1.01)
print("  post: touch +1% (minute):", up_post.height)
for y in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    both = up_pre.filter(pl.col("low") <= pl.col("open") * (1 - y / 100)).height
    print(f"    stop -{y}%: pre days with both in one daily bar: {both}; only +1%: {up_pre.height - both}")
print("  close-based second events on daily bars:")
print("    pre: +1% touched and close < open:", up_pre.filter(pl.col("close") < pl.col("open")).height)
print("    pre: +1% touched and close < open*1.01:", up_pre.filter(pl.col("close") < pl.col("open") * 1.01).height)
print("    pre: +1% touched and close >= open*1.01:", up_pre.filter(pl.col("close") >= pl.col("open") * 1.01).height)
print("    pre: +1% touched and low < open:", up_pre.filter(pl.col("low") < pl.col("open")).height)
print("    pre: +1% touched and low <= prev_close:", up_pre.filter(pl.col("low") <= pl.col("prev_close")).height)
print("    pre: +1% touched and low == open:", up_pre.filter(pl.col("low") == pl.col("open")).height)
print("    pre: +1% touched and low < open, i.e. order unknown vs open:", up_pre.filter(pl.col("low") < pl.col("open")).height)

print("=" * 100)
print("casey/ch-07#1 IWM: opened 1% or more above the previous close -> trades back to the previous close, since 2016-01-01")
b = base("IWM")
v = b.filter(pl.col("prev_close").is_not_null() & (pl.col("date") >= dt.date(2016, 1, 1)))
print("  sessions since 2016:", v.height, " whole:", v.filter(pl.col("whole")).height)
lo_min = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
b2 = v.with_columns(prev_m_close=pl.lit(None))
for gname, gexpr in {
    "gap from session open (09:30 bar) vs daily prev close": pl.col("s_open") / pl.col("prev_close") - 1,
    "gap from daily open vs daily prev close": pl.col("open") / pl.col("prev_close") - 1,
}.items():
    for thr_name, cond in {">=1%": gexpr >= 0.01, ">1%": gexpr > 0.01, ">=1% (rounded 6dp)": gexpr.round(6) >= 0.01}.items():
        c = v.filter(cond)
        for lname, lexpr in {"minute low (whole) else daily": lo_min, "daily low": pl.col("low")}.items():
            k_le = c.filter(lexpr <= pl.col("prev_close")).height
            k_lt = c.filter(lexpr < pl.col("prev_close")).height
            print(f"  [{gname}] {thr_name}: matched {c.height}; fills ({lname}) <= {k_le}, < {k_lt}")
c = v.filter((pl.col("s_open") / pl.col("prev_close") - 1) >= 0.01)
k = c.filter(lo_min <= pl.col("prev_close")).height
show("stated convention (09:30 open vs daily prev close, minute lows)", k, c.height)
print("  only whole sessions:", c.filter(pl.col("whole")).height, " first/last", c["date"].min(), c["date"].max())
