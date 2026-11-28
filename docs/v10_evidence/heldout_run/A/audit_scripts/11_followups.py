"""Follow-ups:
 (a) bootstrap ranges for the two movement items (which method gives Strata's +0.36..+1.33 and +0.07..+3.16?)
 (b) more digits for the weekday 'all days' mean
 (c) alex/ah-27#1: which sessions separate 3,881 / 3,875 / 3,871 / 3,824
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import datetime as dt
import math
import numpy as np
import polars as pl
from common import base, vix

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)

b = base("SPY")
vx = vix().select("date", vix_close="close")
b = b.join(vx, on="date", how="left").sort("date")
c = b["close"].to_numpy(); o = b["s_open"].to_numpy(); n = len(c)
date = b["date"].to_list()
prev_ret = np.full(n, np.nan); prev_ret[2:] = c[1:-1] / c[:-2] - 1
pv = np.full(n, np.nan); pv[1:] = b["vix_close"].to_numpy()[:-1]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi_inv(p):
    # Acklam's rational approximation is overkill; bisection is fine here
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ranges(r, t_idx, label):
    print("-" * 80)
    print(label, "n =", len(r), "mean", f"{100 * r.mean():+.3f}%")
    se = r.std(ddof=1) / math.sqrt(len(r))
    print(f"  normal +/-1.96se: {100 * (r.mean() - 1.96 * se):+.2f}% .. {100 * (r.mean() + 1.96 * se):+.2f}%")
    for seed in (0, 1, 2, 42, 12345):
        for B in (1000, 2000, 10000):
            rng = np.random.default_rng(seed)
            boots = rng.choice(r, size=(B, len(r)), replace=True).mean(axis=1)
            lo, hi = np.quantile(boots, [.025, .975])
            # BCa
            z0 = phi_inv(float((boots < r.mean()).mean()))
            jk = np.array([np.delete(r, i).mean() for i in range(len(r))])
            a = ((jk.mean() - jk) ** 3).sum() / (6 * (((jk.mean() - jk) ** 2).sum() ** 1.5))
            zl, zu = phi_inv(.025), phi_inv(.975)
            al = phi(z0 + (z0 + zl) / (1 - a * (z0 + zl))); au = phi(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
            bl, bu = np.quantile(boots, [al, au])
            if B == 10000 or seed == 0:
                print(f"  seed {seed:5d} B={B:5d}: percentile {100 * lo:+.2f}% .. {100 * hi:+.2f}%   BCa {100 * bl:+.2f}% .. {100 * bu:+.2f}%")
    # moving-block bootstrap over the calendar order of the matching sessions (clusters of nearby dates)
    for L in (2, 3, 5, 10):
        rng = np.random.default_rng(3)
        m = len(r)
        nb = math.ceil(m / L)
        means = []
        for _ in range(5000):
            starts = rng.integers(0, m - L + 1, size=nb)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:m]
            means.append(r[idx].mean())
        lo, hi = np.quantile(means, [.025, .975])
        print(f"  moving-block bootstrap, block {L:2d} matching sessions: {100 * lo:+.2f}% .. {100 * hi:+.2f}%")
    # cluster bootstrap: resample runs of matching sessions that lie within K trading sessions of each other
    for gap in (5, 10):
        clusters, cur = [], [0]
        for i in range(1, len(t_idx)):
            if t_idx[i] - t_idx[i - 1] <= gap:
                cur.append(i)
            else:
                clusters.append(cur); cur = [i]
        clusters.append(cur)
        rng = np.random.default_rng(4)
        means = []
        for _ in range(5000):
            pick = rng.integers(0, len(clusters), size=len(clusters))
            idx = np.concatenate([clusters[p] for p in pick])
            means.append(r[idx].mean())
        lo, hi = np.quantile(means, [.025, .975])
        print(f"  cluster bootstrap (sessions within {gap} trading days grouped, {len(clusters)} clusters): {100 * lo:+.2f}% .. {100 * hi:+.2f}%")


t5 = np.array([t for t in range(2, n) if prev_ret[t] <= -0.02 and t + 5 < n])
ranges(c[t5 + 5] / o[t5] - 1, t5, "morgan/mh-18#1 open[t] -> close[t+5]   (Strata range +0.36% .. +1.33%)")
t10 = np.array([t for t in range(2, n) if prev_ret[t] <= -0.02 and pv[t] > 35 and t + 10 < n])
ranges(c[t10 + 10] / o[t10] - 1, t10, "jordan/jh-16#1 open[t] -> close[t+10]   (Strata range +0.07% .. +3.16%)")

print("=" * 80)
v = b.filter(pl.col("prev_close").is_not_null() & (pl.col("date") >= dt.date(2015, 1, 1)))
x = ((v["close"] / v["prev_close"] - 1) * 100)
print("weekday item, all days mean close-to-close:", f"{x.mean():+.6f}%", " n", v.height)

print("=" * 80)
print("alex/ah-27#1 details")
b = b.with_columns(prev_m_low=pl.col("m_low").shift(1), prev_whole=pl.col("whole").shift(1).fill_null(False))
v = b.filter(pl.col("prev_close").is_not_null())
cur_m = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
prev_m = pl.when(pl.col("prev_whole")).then(pl.col("prev_m_low")).otherwise(pl.col("prev_low"))
v = v.with_columns(
    A=(pl.col("low") < pl.col("prev_low")),            # strict, daily record
    B=(pl.col("low") <= pl.col("prev_low")),           # at or below, daily record
    C=(cur_m <= pl.col("prev_low")),                   # at or below, minute lows vs daily previous low
    D=(cur_m <= prev_m),                               # at or below, minute lows vs minute previous low  (3,881)
    E=(cur_m < pl.col("prev_low")),                    # strict, minute lows vs daily previous low
    F=(pl.col("low") <= prev_m),
)
for k in "ABCDEF":
    print(" ", k, v.filter(pl.col(k)).height)
print("  D and not B:", v.filter(pl.col("D") & ~pl.col("B")).height, "  B and not D:", v.filter(pl.col("B") & ~pl.col("D")).height)
print("  D and not C:", v.filter(pl.col("D") & ~pl.col("C")).height, "  C and not D:", v.filter(pl.col("C") & ~pl.col("D")).height)
print("  D == F on every session:", v.filter(pl.col("D") != pl.col("F")).height == 0, " (sessions where they differ:", v.filter(pl.col("D") != pl.col("F")).height, ")")
print("  sessions equal to the previous low exactly (daily record):", v.filter(pl.col("low") == pl.col("prev_low")).height)
print("  days in D but not in A (counted by Strata's figure, not strictly below the daily-record previous low):", v.filter(pl.col("D") & ~pl.col("A")).height)
print("  days in A but not in D:", v.filter(pl.col("A") & ~pl.col("D")).height)
x = v.filter(pl.col("D") != pl.col("C")).select("date", "low", "m_low", "prev_low", "prev_m_low", "C", "D")
print(x)
print("  2016+ sessions where the previous session's minute-bar low differs from its daily-record low by more than 1 cent:",
      v.filter(pl.col("prev_whole") & ((pl.col("prev_m_low") - pl.col("prev_low")).abs() > 0.0100001)).height,
      " (minute low higher:", v.filter(pl.col("prev_whole") & ((pl.col("prev_m_low") - pl.col("prev_low")) > 0.0100001)).height, ")")
