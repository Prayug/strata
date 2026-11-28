"""alex/ah-03#1  SPY: average close-to-close move by weekday since 2015-01-01.
Strata: 2,994 sessions; Wednesday +0.096% (614 days) best; Friday +0.002% (600 days) worst; all days +0.048%;
        closed higher on Wednesdays 56.2%, Fridays 52.8%;
        open->close: Mon +0.073%, Tue -0.016%, Wed +0.028%, Thu +0.003%, Fri +0.004%; Jan 2, 2015 to Nov 27, 2026;
        'no weekday stands out' (label shuffling).
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import numpy as np
import polars as pl
from common import base

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)

b = base("SPY")
v = b.filter(pl.col("prev_close").is_not_null() & (pl.col("date") >= dt.date(2015, 1, 1)))
v = v.with_columns(
    wd=pl.col("date").dt.weekday(),  # 1 = Monday
    c2c=(pl.col("close") / pl.col("prev_close") - 1) * 100,
    o2c_s=(pl.col("close") / pl.col("s_open") - 1) * 100,
    o2c_d=(pl.col("close") / pl.col("open") - 1) * 100,
    up=(pl.col("close") > pl.col("prev_close")),
)
print("sessions:", v.height, "first", v["date"].min(), "last", v["date"].max())
print("all days: mean close-to-close", f"{v['c2c'].mean():+.4f}%", " mean open->close (session open)", f"{v['o2c_s'].mean():+.4f}%",
      " (daily open)", f"{v['o2c_d'].mean():+.4f}%")
names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday"}
g = v.group_by("wd").agg(
    n=pl.len(), c2c=pl.col("c2c").mean(), up=pl.col("up").mean() * 100, k_up=pl.col("up").sum(),
    o2c_s=pl.col("o2c_s").mean(), o2c_d=pl.col("o2c_d").mean(),
).sort("wd")
for r in g.iter_rows(named=True):
    print(f"  {names[r['wd']]:10s} n={r['n']:4d}  close-to-close {r['c2c']:+.4f}% ({r['c2c']:+.3f}%)   closed higher {r['k_up']} = {r['up']:.2f}%   "
          f"open->close: session-open {r['o2c_s']:+.4f}% ({r['o2c_s']:+.3f}%)   daily-open {r['o2c_d']:+.4f}% ({r['o2c_d']:+.3f}%)")

# label-shuffling test on the spread of weekday means (max - min) and on an F-like statistic
x = v["c2c"].to_numpy()
wd = v["wd"].to_numpy()
rng = np.random.default_rng(7)


def stat(lbl):
    means = np.array([x[lbl == k].mean() for k in range(1, 6)])
    return means.max() - means.min(), float(((means - x.mean()) ** 2 * np.array([(lbl == k).sum() for k in range(1, 6)])).sum())


obs = stat(wd)
cnt = np.zeros(2)
R = 5000
for _ in range(R):
    s = stat(rng.permutation(wd))
    cnt += (np.array(s) >= np.array(obs))
print(f"label shuffling ({R} shuffles): spread of weekday means {obs[0]:.4f} pct-pts, p = {cnt[0] / R:.3f};  between-group SS p = {cnt[1] / R:.3f}")
