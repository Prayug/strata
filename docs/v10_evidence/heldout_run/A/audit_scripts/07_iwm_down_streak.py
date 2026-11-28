"""alex/ah-24#1  IWM, 4+ down sessions in a row (up to the previous session), since 2000-01-01.
Strata: 'closes above the previous close' 56.7% of 261 days (any day 55.1%);
        'closes above its own open' 53.3% of 261 days (any day 54.6%).
Reading: closes above the previous session's close 5 sessions later (at the close); separately: the next session closes above its own open.
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import numpy as np
import polars as pl
from common import base, streaks, wilson, pct

b = base("IWM")
c = b["close"].to_numpy()
o_d = b["open"].to_numpy()
o_s = b["s_open"].to_numpy()
date = b["date"].to_list()
n = len(c)
up, dn = streaks(list(c))
dn = np.array(dn)

# session t matches when the down streak counted up to the previous session (t-1) is >= 4
match = np.zeros(n, dtype=bool)
match[1:] = dn[:-1] >= 4
valid = np.zeros(n, dtype=bool)
valid[1:] = True  # has a previous close
print("IWM sessions with a previous close:", valid.sum(), "; first", date[1], "last", date[-1])
print("sessions after 4+ down closes in a row:", int(match.sum()), "; first", date[int(np.argmax(match))], "; last", date[int(np.where(match)[0][-1])])
print("  (>=3:", int((np.r_[False, dn[:-1] >= 3]).sum()), " >=5:", int((np.r_[False, dn[:-1] >= 5]).sum()), ")")


def rate(event, cond, label):
    """event: array of floats (1.0/0.0/nan when not measurable)."""
    ok = cond & ~np.isnan(event)
    k, m = int(np.nansum(event[ok])), int(ok.sum())
    alln = valid & ~np.isnan(event)
    ka, ma = int(np.nansum(event[alln])), int(alln.sum())
    lo, hi = wilson(k, m)
    print(f"  {label}: {k} of {m} = {100 * k / m:.1f}%  (Wilson {pct(lo)}..{pct(hi)})   any day: {ka} of {ma} = {100 * ka / ma:.1f}%")
    return k, m, ka, ma


def ev(a, b_):
    out = np.full(n, np.nan)
    good = ~np.isnan(a) & ~np.isnan(b_)
    out[good] = (a[good] > b_[good]).astype(float)
    return out


def shift_fwd(x, k):
    out = np.full(n, np.nan)
    if k == 0:
        return x.astype(float).copy()
    out[:-k] = x[k:]
    return out


prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
print()
print("Variant 1: close K sessions later above the previous session's close (close[t-1])")
for K in (0, 1, 4, 5, 6):
    rate(ev(shift_fwd(c, K), prev_c), match, f"close[t+{K}] > close[t-1]")
print()
print("Variant 2: closes above its own open")
for K in (0, 4, 5):
    rate(ev(shift_fwd(c, K), o_s), match, f"close[t+{K}] > session open[t] (09:30 bar where whole)")
    rate(ev(shift_fwd(c, K), o_d), match, f"close[t+{K}] > daily-record open[t]")
print()
print("Reference: the session K later closes above ITS OWN open")
for K in (0, 1, 4, 5):
    rate(ev(shift_fwd(c, K), shift_fwd(o_s, K)), match, f"close[t+{K}] > session open[t+{K}]")
    rate(ev(shift_fwd(c, K), shift_fwd(o_d, K)), match, f"close[t+{K}] > daily open[t+{K}]")
