"""How complete are the minute-bar sessions? Which sessions are 'whole'?"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import polars as pl
from common import daily, minute, rth

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

for sym in ["SPY", "QQQ", "IWM"]:
    print("=" * 100)
    print(sym)
    d = daily(sym)
    m = minute(sym)
    print("minute rows", m.height, "outside 09:30-16:00:", m.filter((pl.col("mod") < 570) | (pl.col("mod") >= 960)).height)
    r = rth(m)
    s = r.group_by("date").agg(
        n=pl.len(), first=pl.col("mod").min(), last=pl.col("mod").max(),
        o=pl.col("open").first(), h=pl.col("high").max(), l=pl.col("low").min(), c=pl.col("close").last(),
    ).sort("date")
    print("minute sessions:", s.height, s["date"].min(), s["date"].max())
    d16 = d.filter(pl.col("date") >= dt.date(2016, 1, 1))
    print("daily sessions since 2016:", d16.height)
    md = set(s["date"].to_list()); dd = set(d16["date"].to_list())
    print("in daily not minute:", sorted(dd - md))
    print("in minute not daily:", sorted(md - dd))
    print("bar-count distribution:")
    print(s.group_by("n").agg(k=pl.len()).sort("n"))
    print("first-bar distribution:", s.group_by("first").agg(k=pl.len()).sort("first").to_dicts())
    print("last-bar distribution:", s.group_by("last").agg(k=pl.len()).sort("last").to_dicts())
    odd = s.filter((pl.col("n") != 390))
    print("sessions with n != 390:", odd.height)
    print(odd)
    # compare the session OHLC from minute bars with the daily record
    j = s.join(d, on="date", how="inner")
    j = j.with_columns(
        do=(pl.col("o") - pl.col("open")), dh=(pl.col("h") - pl.col("high")), dl=(pl.col("l") - pl.col("low")), dc=(pl.col("c") - pl.col("close")),
    )
    for col in ["do", "dh", "dl", "dc"]:
        a = j[col].abs()
        print(col, "max abs diff", a.max(), "mean abs", a.mean(), "n exact(<1e-6)", (a < 1e-6).sum(), "n within 1c", (a <= 0.0100001).sum(), "of", j.height)
