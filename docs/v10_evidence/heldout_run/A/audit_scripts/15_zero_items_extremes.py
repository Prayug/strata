"""For the six '0 of N' items: how extreme would an unstated threshold have to be for '0 days' to be a true count?
Prints the largest rise and fall of the summary's end return, and how many days exceed round thresholds."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import numpy as np
import polars as pl
from common import base, minute

B = {s: base(s) for s in ("SPY", "QQQ", "IWM")}
V = {s: B[s].filter(pl.col("prev_close").is_not_null()) for s in B}


def show(key, df, start_col):
    r = (df["close"] / df[start_col] - 1).to_numpy() * 100
    d = df["date"].to_list()
    i_max, i_min = int(np.argmax(r)), int(np.argmin(r))
    print(f"{key}: n={len(r)}  largest rise {r[i_max]:+.2f}% on {d[i_max]}  largest fall {r[i_min]:+.2f}% on {d[i_min]}")
    print("    days with a rise of at least 1/2/3/5/10%:", [int((r >= x).sum()) for x in (1, 2, 3, 5, 10)],
          "  a fall of at least 1/2/3/5/10%:", [int((r <= -x).sum()) for x in (1, 2, 3, 5, 10)],
          "  |move| >= 0.5%:", int((np.abs(r) >= 0.5).sum()))


show("riley/rh-01#1 IWM open->close since 2000", V["IWM"].filter(pl.col("date") >= dt.date(2000, 1, 1)), "s_open")
show("alex/ah-15#1 SPY open->close since 2016", V["SPY"].filter(pl.col("date") >= dt.date(2016, 1, 1)), "s_open")
show("riley/rh-04#1 SPY open->close", V["SPY"], "s_open")
show("jordan/jh-21#2 SPY gap<0 open->close", V["SPY"].filter(pl.col("s_open") < pl.col("prev_close")), "s_open")
show("jordan/jh-21#6 QQQ gap<0 open->close", V["QQQ"].filter(pl.col("s_open") < pl.col("prev_close")), "s_open")
whole = B["SPY"].filter(pl.col("whole")).select("date", "close")
p = minute("SPY").filter(pl.col("mod") == 14 * 60 + 29).select("date", px="close")
show("alex/ah-18#1 SPY 14:30->close", whole.join(p, on="date", how="inner"), "px")
