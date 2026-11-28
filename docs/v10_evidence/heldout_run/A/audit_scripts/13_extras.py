"""Extra figures quoted in the notes.
 (a) casey/ch-18#1: in the minute-bar years, of the sessions that touched both +1% and -0.5% from the open, how often did +1% come first?
 (b) movement items: favourable/adverse medians measured strictly inside the window for each start/end choice.
 (c) sizes of the daily-record vs minute-bar disagreements used in notes.
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import numpy as np
import polars as pl
from common import base, minute, vix

# (a)
b = base("QQQ")
v = b.filter(pl.col("prev_close").is_not_null())
m = minute("QQQ")
post = v.filter(pl.col("whole")).select("date", "s_open")
mm = m.join(post, on="date", how="inner").with_columns(
    hu=pl.col("high") >= pl.col("s_open") * 1.01, hd=pl.col("low") <= pl.col("s_open") * 0.995)
g = mm.group_by("date").agg(tu=pl.col("mod").filter(pl.col("hu")).min(), td=pl.col("mod").filter(pl.col("hd")).min())
both = g.filter(pl.col("tu").is_not_null() & pl.col("td").is_not_null())
print("minute-bar sessions:", g.height, "; touched +1%:", g.filter(pl.col("tu").is_not_null()).height,
      "; touched both levels:", both.height, "; of those +1% first:", both.filter(pl.col("tu") < pl.col("td")).height,
      "; same minute:", both.filter(pl.col("tu") == pl.col("td")).height)
print("minute-bar sessions where +1% came first overall:", g.filter(pl.col("tu").is_not_null() & (pl.col("td").is_null() | (pl.col("tu") < pl.col("td")))).height)
pre = v.filter(~pl.col("whole"))
print("daily-bar days:", pre.height, "; touched +1% only:", pre.filter((pl.col("high") >= pl.col("open") * 1.01) & (pl.col("low") > pl.col("open") * 0.995)).height,
      "; both inside one bar:", pre.filter((pl.col("high") >= pl.col("open") * 1.01) & (pl.col("low") <= pl.col("open") * 0.995)).height)
print("bounds for the all-days rate: ", round(100 * 1202 / 6973, 1), "to", round(100 * (1202 + 780) / 6973, 1))

# (b)
b = base("SPY")
vx = vix().select("date", vix_close="close")
b = b.join(vx, on="date", how="left").sort("date")
c = b["close"].to_numpy(); o = b["s_open"].to_numpy(); h = b["high"].to_numpy(); l = b["low"].to_numpy(); n = len(c)
whole = b["whole"].to_numpy()
hm = np.where(whole, np.nan_to_num(b["m_high"].to_numpy(), nan=0.0), h)
lm = np.where(whole, np.nan_to_num(b["m_low"].to_numpy(), nan=0.0), l)
prev_ret = np.full(n, np.nan); prev_ret[2:] = c[1:-1] / c[:-2] - 1
pv = np.full(n, np.nan); pv[1:] = b["vix_close"].to_numpy()[:-1]


def stats(t, start, first_sess, last_sess, label):
    r = c[last_sess] / start - 1
    fav = np.array([h[a:z + 1].max() for a, z in zip(first_sess, last_sess)]) / start - 1
    adv = np.array([l[a:z + 1].min() for a, z in zip(first_sess, last_sess)]) / start - 1
    favm = np.array([hm[a:z + 1].max() for a, z in zip(first_sess, last_sess)]) / start - 1
    advm = np.array([lm[a:z + 1].min() for a, z in zip(first_sess, last_sess)]) / start - 1
    print(f"  {label}: n={len(r)} mean {100 * r.mean():+.2f}% median {100 * np.median(r):+.2f}% higher {100 * (r > 0).mean():.1f}% "
          f"q25 {100 * np.quantile(r, .25):+.2f}% q75 {100 * np.quantile(r, .75):+.2f}%  fav {100 * np.median(fav):+.2f}% adv {100 * np.median(adv):+.2f}% "
          f"(minute-bar highs/lows where stored: fav {100 * np.median(favm):+.2f}% adv {100 * np.median(advm):+.2f}%)")


for K, cond, name in [(5, lambda t: prev_ret[t] <= -0.02, "morgan/mh-18#1"), (10, lambda t: prev_ret[t] <= -0.02 and pv[t] > 35, "jordan/jh-16#1")]:
    t = np.array([i for i in range(2, n) if cond(i) and i + K < n])
    print(name, "K =", K, "sessions:", len(t))
    stats(t, o[t], t, t + K, f"open[t] -> close[t+{K}] (Strata's figures)")
    stats(t, c[t], t + 1, t + K, f"close[t] -> close[t+{K}] (reading, 'that session' = the session after the fall)")
    stats(t, c[t - 1], t, t + K - 1, f"close[t-1] -> close[t+{K - 1}] (reading, 'that session' = the session that fell)")
    stats(t, o[t], t, t + K - 1, f"open[t] -> close[t+{K - 1}] (next {K} sessions from the open)")

# (c)
for sym in ("SPY", "QQQ", "IWM"):
    b = base(sym).filter(pl.col("whole"))
    for col_d, col_m in (("low", "m_low"), ("high", "m_high"), ("open", "m_open")):
        d = (b[col_m] - b[col_d]).abs()
        print(sym, col_d, "minute vs daily differ by more than 1 cent on", int((d > 0.0100001).sum()), "of", b.height, "whole sessions; max", round(float(d.max()), 2))
