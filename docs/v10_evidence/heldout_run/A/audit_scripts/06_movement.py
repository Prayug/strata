"""Movement items over several sessions.
morgan/mh-18#1  SPY, prev session fell 2%+ -> move over the next 5 sessions   Strata: 341 days, avg +0.85%, median +0.78%, higher 60.1%,
                quartiles -1.73%/+3.38%, favourable +2.71%, adverse -2.54%, likely range of the average +0.36%..+1.33%
jordan/jh-16#1  SPY, prev session fell 2%+ and VIX closed above 35 the session before -> next 10 sessions
                Strata: 106 days, avg +1.72%, median +3.08%, higher 67.0%, quartiles -1.81%/+6.98%, fav +6.95%, adv -4.77%, range +0.07%..+3.16%
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import numpy as np
import polars as pl
from common import base, vix

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)

b = base("SPY")
vx = vix().select("date", vix_close="close")
b = b.join(vx, on="date", how="left").sort("date")
b = b.with_columns(prev_vix=pl.col("vix_close").shift(1), prev2_vix=pl.col("vix_close").shift(2))
print("SPY rows", b.height, "VIX missing on SPY dates:", b.filter(pl.col("vix_close").is_null()).height)

date = b["date"].to_list()
o_d = b["open"].to_numpy()
o_s = b["s_open"].to_numpy()
h = b["high"].to_numpy()
l = b["low"].to_numpy()
c = b["close"].to_numpy()
# session highs/lows with minute bars where a whole session is stored
hm = np.where(b["whole"].to_numpy(), np.nan_to_num(b["m_high"].to_numpy(), nan=0.0), h)
lm = np.where(b["whole"].to_numpy(), np.nan_to_num(b["m_low"].to_numpy(), nan=0.0), l)
n = len(c)
prev_ret = np.full(n, np.nan)
prev_ret[2:] = c[1:-1] / c[:-2] - 1  # return of session t-1
pv = b["prev_vix"].to_numpy()


def q(x, p):
    return float(np.quantile(x, p))


def describe(idx, K, label):
    """idx = observation sessions t (the session after the fall).  K sessions horizon."""
    print("-" * 100)
    print(label, "| matching sessions:", len(idx), "| first", date[idx[0]] if len(idx) else None, "last", date[idx[-1]] if len(idx) else None)
    variants = {}
    for name, start_arr, start_shift, end_shift in [
        (f"A: session open (09:30 bar where whole) of t -> close of t+{K - 1}  ({K} sessions incl. t)", o_s, 0, K - 1),
        (f"B: session open of t -> close of t+{K}", o_s, 0, K),
        (f"C: close of t-1 (the down session) -> close of t+{K - 1}  ('that session's close to the close {K} sessions later')", c, -1, K - 1),
        (f"D: close of t -> close of t+{K}", c, 0, K),
        (f"E: daily-record open of t -> close of t+{K - 1}", o_d, 0, K - 1),
        (f"F: daily-record open of t -> close of t+{K}", o_d, 0, K),
        (f"G: close of t-1 -> close of t+{K}", c, -1, K),
    ]:
        ok = [t for t in idx if t + end_shift < n and t + start_shift >= 0]
        if not ok:
            continue
        t = np.array(ok)
        start = start_arr[t + start_shift]
        end = c[t + end_shift]
        r = end / start - 1
        # favourable / adverse excursion over the window (from the first session whose bars lie after the start)
        first = t if start_shift == 0 else t  # sessions t..t+end_shift are inside the window either way
        fav_d = np.array([h[a:b_ + 1].max() for a, b_ in zip(first, t + end_shift)]) / start - 1
        adv_d = np.array([l[a:b_ + 1].min() for a, b_ in zip(first, t + end_shift)]) / start - 1
        fav_m = np.array([hm[a:b_ + 1].max() for a, b_ in zip(first, t + end_shift)]) / start - 1
        adv_m = np.array([lm[a:b_ + 1].min() for a, b_ in zip(first, t + end_shift)]) / start - 1
        se = r.std(ddof=1) / np.sqrt(len(r))
        rng = np.random.default_rng(12345)
        boots = rng.choice(r, size=(20000, len(r)), replace=True).mean(axis=1)
        print(f"  {name}")
        print(f"     n={len(r)}  mean {100 * r.mean():+.2f}%  median {100 * np.median(r):+.2f}%  higher {100 * (r > 0).mean():.1f}%  "
              f"(>=0: {100 * (r >= 0).mean():.1f}%)  q25 {100 * q(r, .25):+.2f}%  q75 {100 * q(r, .75):+.2f}%")
        print(f"     fav median (daily highs) {100 * np.median(fav_d):+.2f}%  adv median (daily lows) {100 * np.median(adv_d):+.2f}%   "
              f"| with minute highs/lows where whole: fav {100 * np.median(fav_m):+.2f}%  adv {100 * np.median(adv_m):+.2f}%")
        print(f"     mean +/- 1.96 se: {100 * (r.mean() - 1.96 * se):+.2f}% .. {100 * (r.mean() + 1.96 * se):+.2f}%   "
              f"iid bootstrap 2.5/97.5: {100 * np.quantile(boots, .025):+.2f}% .. {100 * np.quantile(boots, .975):+.2f}%")
        variants[name[0]] = r
    return variants


# ---- morgan/mh-18#1
print("=" * 100)
print("morgan/mh-18#1  SPY: previous session fell 2% or more -> next 5 sessions")
idx_all = [t for t in range(2, n) if prev_ret[t] <= -0.02]
print("sessions whose previous session fell 2%+ (<= -2%):", len(idx_all), " strictly < -2%:", sum(1 for t in range(2, n) if prev_ret[t] < -0.02))
print("down sessions of 2%+ in the whole file (incl. the last session, which has no next session):", int(np.nansum((c[1:] / c[:-1] - 1) <= -0.02)))
for K in (5,):
    describe(idx_all, K, f"K={K}")

# ---- jordan/jh-16#1
print("=" * 100)
print("jordan/jh-16#1  SPY: previous session fell 2%+ and VIX closed above 35 the session before -> next 10 sessions")
idx_v1 = [t for t in range(2, n) if prev_ret[t] <= -0.02 and pv[t] > 35]
print("prev_ret<=-2% and VIX close of t-1 > 35:", len(idx_v1))
p2 = b["prev2_vix"].to_numpy()
vt = b["vix_close"].to_numpy()
print("  [ref] VIX close of t-2 > 35:", len([t for t in range(2, n) if prev_ret[t] <= -0.02 and p2[t] > 35]),
      "; VIX close of t > 35:", len([t for t in range(2, n) if prev_ret[t] <= -0.02 and vt[t] > 35]),
      "; VIX t-1 >= 35:", len([t for t in range(2, n) if prev_ret[t] <= -0.02 and pv[t] >= 35]))
describe(idx_v1, 10, "K=10, VIX close of t-1 > 35")
