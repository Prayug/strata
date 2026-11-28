"""casey/ch-18#1 (QQQ touch:+1, 1,202 / 4,991 / 780).
The 780 undecidable days equal the pre-2016 daily bars that hold both +1% and -0.5% from the open.
Test the full hypothesis: 'reaches +1% above the open before -0.5% below it, by the close',
minute bars deciding the order where a whole session of them is stored."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import polars as pl
from common import base, minute, wilson, pct

SYM = "QQQ"
b = base(SYM)
v = b.filter(pl.col("prev_close").is_not_null())
m = minute(SYM)


def run(up, dn, verbose=True):
    fu, fd = 1 + up / 100, 1 - dn / 100
    pre = v.filter(~pl.col("whole"))
    pre_up = pre.filter(pl.col("high") >= pl.col("open") * fu)
    pre_both = pre_up.filter(pl.col("low") <= pl.col("open") * fd)
    pre_hit = pre_up.height - pre_both.height
    pre_no = pre.height - pre_up.height

    post = v.filter(pl.col("whole")).select("date", "s_open")
    mm = m.join(post, on="date", how="inner")
    mm = mm.with_columns(
        hit_up=pl.col("high") >= pl.col("s_open") * fu,
        hit_dn=pl.col("low") <= pl.col("s_open") * fd,
    )
    g = mm.group_by("date").agg(
        t_up=pl.col("mod").filter(pl.col("hit_up")).min(),
        t_dn=pl.col("mod").filter(pl.col("hit_dn")).min(),
    )
    up_first = g.filter(pl.col("t_up").is_not_null() & (pl.col("t_dn").is_null() | (pl.col("t_up") < pl.col("t_dn")))).height
    same = g.filter(pl.col("t_up").is_not_null() & pl.col("t_dn").is_not_null() & (pl.col("t_up") == pl.col("t_dn"))).height
    post_no = g.height - up_first - same
    happened = pre_hit + up_first
    undec = pre_both.height + same
    didnt = pre_no + post_no
    if verbose:
        print(f"+{up}% before -{dn}%:  happened {happened} (daily-only days {pre_hit} + minute days {up_first}); "
              f"didn't {didnt}; undecidable {undec} (daily bars {pre_both.height} + same minute bar {same}); total {happened + didnt + undec}")
        if happened + didnt:
            lo, hi = wilson(happened, happened + didnt)
            print(f"    rate among decided days {happened}/{happened + didnt} = {pct(happened / (happened + didnt))}  Wilson95 {pct(lo)}..{pct(hi)};"
                  f"   of all {v.height} days = {pct(happened / v.height)}")
    return happened, didnt, undec


for dn in (0.25, 0.5, 0.75, 1.0, 2.0):
    run(1.0, dn)
print()
print("Plain touch of +1% (no second level) for reference:")
hi_min = pl.when(pl.col("whole")).then(pl.col("m_high")).otherwise(pl.col("high"))
k = v.filter(hi_min >= pl.col("s_open") * 1.01).height
lo, hi = wilson(k, v.height)
print(f"  {k} of {v.height} = {pct(k / v.height)}  Wilson95 {pct(lo)}..{pct(hi)}")
print()
print("Strata: happened 1202, didn't 4991, undecidable 780; 1202/6973 =", pct(1202 / 6973), "; 1202/(1202+4991) =", pct(1202 / 6193))
print("Wilson for 1202/6193:", [pct(x) for x in wilson(1202, 6193)], " for 1202/6973:", [pct(x) for x in wilson(1202, 6973)])
