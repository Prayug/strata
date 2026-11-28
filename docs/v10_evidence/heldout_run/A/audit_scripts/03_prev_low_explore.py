"""alex/ah-27#1: SPY trades below the previous session's low.  Strata 3,881; its own check 3,870.
Try every plausible definition to see which (if any) reproduces either figure."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import polars as pl
from common import base, minute, session_table

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)

b = base("SPY")
b = b.with_columns(prev_m_low=pl.col("m_low").shift(1), prev_whole=pl.col("whole").shift(1), prev_n=pl.col("n").shift(1))
v = b.filter(pl.col("prev_close").is_not_null())
N = v.height
print("days", N)

cur_daily = pl.col("low")
cur_min_whole = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
cur_min_any = pl.when(pl.col("m_low").is_not_null()).then(pl.col("m_low")).otherwise(pl.col("low"))
cur_both = pl.min_horizontal(pl.col("low"), pl.col("m_low").fill_null(1e18))
prev_daily = pl.col("prev_low")
prev_min_whole = pl.when(pl.col("prev_whole").fill_null(False)).then(pl.col("prev_m_low")).otherwise(pl.col("prev_low"))
prev_min_any = pl.when(pl.col("prev_m_low").is_not_null()).then(pl.col("prev_m_low")).otherwise(pl.col("prev_low"))

curs = {"cur=daily": cur_daily, "cur=minute(whole)": cur_min_whole, "cur=minute(any)": cur_min_any, "cur=min(daily,minute)": cur_both}
prevs = {"prev=daily": prev_daily, "prev=minute(whole)": prev_min_whole, "prev=minute(any)": prev_min_any}
for cn, ce in curs.items():
    for pn, pe in prevs.items():
        lt = v.filter(ce < pe).height
        le = v.filter(ce <= pe).height
        for eps in (0.0,):
            pass
        flag = ""
        if lt in (3881, 3870) or le in (3881, 3870):
            flag = "   <<<<<< MATCH"
        print(f"{cn:26s} {pn:20s}  <: {lt}   <=: {le}{flag}")

print()
print("tolerances on the daily-record comparison (low <= prev_low + eps):")
for eps in (0.0, 1e-6, 1e-4, 0.001, 0.005, 0.0051, 0.01, 0.0101, 0.02, 0.03, 0.05):
    a = v.filter(pl.col("low") <= pl.col("prev_low") + eps).height
    c = v.filter(cur_min_whole <= pl.col("prev_low") + eps).height
    print(f"  eps={eps}: daily {a}   minute(whole)/daily {c}")

print()
print("rounded to cents:")
for nd in (2, 3, 4):
    a = v.filter(pl.col("low").round(nd) <= pl.col("prev_low").round(nd)).height
    a2 = v.filter(pl.col("low").round(nd) < pl.col("prev_low").round(nd)).height
    c = v.filter(cur_min_whole.round(nd) <= pl.col("prev_low").round(nd)).height
    c2 = v.filter(cur_min_whole.round(nd) < pl.col("prev_low").round(nd)).height
    print(f"  round {nd}: daily <= {a}, < {a2};   minute(whole)/daily <= {c}, < {c2}")

print()
print("open-based variants (the level counts as touched if the session opens at or below it):")
for on, oe in {"daily open": pl.col("open"), "s_open": pl.col("s_open")}.items():
    a = v.filter((pl.col("low") <= pl.col("prev_low")) | (oe <= pl.col("prev_low"))).height
    c = v.filter((cur_min_whole <= pl.col("prev_low")) | (oe <= pl.col("prev_low"))).height
    print(f"  {on}: daily {a}  minute {c}")

print()
print("float32 comparisons (the daily file looks like float32 values widened to float64):")
f32 = v.with_columns(l32=pl.col("low").cast(pl.Float32), pl32=pl.col("prev_low").cast(pl.Float32), ml32=cur_min_whole.cast(pl.Float32))
print("  daily f32 <=", f32.filter(pl.col("l32") <= pl.col("pl32")).height, " <", f32.filter(pl.col("l32") < pl.col("pl32")).height)
print("  minute f32 <=", f32.filter(pl.col("ml32") <= pl.col("pl32")).height, " <", f32.filter(pl.col("ml32") < pl.col("pl32")).height)

# Year by year, to see where an 11-day or 6-10 day gap could sit
print()
yr = v.with_columns(y=pl.col("date").dt.year()).group_by("y").agg(
    n=pl.len(),
    d_lt=(pl.col("low") < pl.col("prev_low")).sum(), d_le=(pl.col("low") <= pl.col("prev_low")).sum(),
    m_lt=(cur_min_whole < pl.col("prev_low")).sum(), m_le=(cur_min_whole <= pl.col("prev_low")).sum(),
).sort("y")
print(yr)
print("pre-2016: daily <", v.filter(pl.col("date") < dt.date(2016, 1, 1)).filter(pl.col("low") < pl.col("prev_low")).height,
      " <=", v.filter(pl.col("date") < dt.date(2016, 1, 1)).filter(pl.col("low") <= pl.col("prev_low")).height)
post = v.filter(pl.col("date") >= dt.date(2016, 1, 1))
print("2016+: daily <", post.filter(pl.col("low") < pl.col("prev_low")).height, " <=", post.filter(pl.col("low") <= pl.col("prev_low")).height,
      "; minute(whole) <", post.filter(cur_min_whole < pl.col("prev_low")).height, " <=", post.filter(cur_min_whole <= pl.col("prev_low")).height,
      "; minute vs prev minute <", post.filter(cur_min_whole < prev_min_whole).height, " <=", post.filter(cur_min_whole <= prev_min_whole).height)
