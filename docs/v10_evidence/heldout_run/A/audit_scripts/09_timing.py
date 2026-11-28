"""Half-hour 'parts of the session' items, from the one-minute bars (whole sessions only).

alex/ah-08#1   IWM  when the session's low was made        Strata: 09:30-10:00 33.8%, 15:30-16:00 13.2%, 2,741 sessions, 7.7% if uniform
casey/ch-03#1  SPY  how much the price moves per part      Strata: 09:30-10:00 0.21%, quietest 12:30-13:00 0.11%, next 10:00-10:30 0.18%,
                                                                   largest move in 09:30-10:00 on 21.1% of sessions, 2,741 sessions
morgan/mh-12#1 SPY  range (high-low)/first price per part  Strata: 09:30-10:00 0.42%, quietest 13:30-14:00 0.23%, next 10:00-10:30 0.37%,
                                                                   widest range in 09:30-10:00 on 32.1% of sessions, 2,741 sessions
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit")
import polars as pl
from common import minute, session_table, OPEN_MOD

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def lab(bk):
    a = OPEN_MOD + 30 * bk
    return f"{a // 60:02d}:{a % 60:02d}-{(a + 30) // 60:02d}:{(a + 30) % 60:02d}"


def whole_minutes(sym):
    m = minute(sym)
    s = session_table(m)
    w = s.filter(pl.col("whole")).select("date", "half")
    mm = m.join(w, on="date", how="inner").with_columns(bk=((pl.col("mod") - OPEN_MOD) // 30).cast(pl.Int32))
    return mm, w, s


# ------------------------------------------------------------------ low time (IWM)
print("=" * 100)
print("alex/ah-08#1 IWM: when the session's low was made (30-minute parts)")
mm, w, s = whole_minutes("IWM")
print("  whole sessions:", w.height, "(early closes:", w.filter(pl.col("half")).height, ") first", w["date"].min(), "last", w["date"].max(),
      "| all minute sessions:", s.height, "| not whole:", s.filter(~pl.col("whole"))["date"].to_list())
for tie in ("first", "last"):
    g = mm.sort(["date", "mod"]).group_by("date", maintain_order=True).agg(
        lowmin=pl.col("low").min(),
        t_first=pl.col("mod").filter(pl.col("low") == pl.col("low").min()).min(),
        t_last=pl.col("mod").filter(pl.col("low") == pl.col("low").min()).max(),
    )
    g = g.with_columns(t=pl.col("t_first") if tie == "first" else pl.col("t_last"))
    g = g.with_columns(bk=((pl.col("t") - OPEN_MOD) // 30).cast(pl.Int32))
    dist = g.group_by("bk").agg(k=pl.len()).sort("bk").with_columns(share=pl.col("k") / g.height * 100)
    print(f"  ties -> {tie} bar holding the low:")
    for r in dist.iter_rows(named=True):
        print(f"     {lab(r['bk'])}: {r['k']:5d}  {r['share']:.1f}%  ({r['share']:.2f})")
    print("     sessions:", g.height, " tie sessions whose first/last low bars fall in different parts:",
          g.filter(((pl.col('t_first') - OPEN_MOD) // 30) != ((pl.col('t_last') - OPEN_MOD) // 30)).height)
print("  uniform share over 13 parts:", round(100 / 13, 2))
# excluding early closes, for reference
g = mm.filter(~pl.col("half")).sort(["date", "mod"]).group_by("date", maintain_order=True).agg(
    t=pl.col("mod").filter(pl.col("low") == pl.col("low").min()).min())
g = g.with_columns(bk=((pl.col("t") - OPEN_MOD) // 30).cast(pl.Int32))
d2 = g.group_by("bk").agg(k=pl.len()).sort("bk").with_columns(share=pl.col("k") / g.height * 100)
print("  [ref] excluding early closes (", g.height, "sessions): first part", round(d2.filter(pl.col("bk") == 0)["share"][0], 2),
      " last part", round(d2.filter(pl.col("bk") == 12)["share"][0], 2))


# ------------------------------------------------------------------ SPY per-part move and range
def per_part(sym):
    mm, w, s = whole_minutes(sym)
    print("  whole sessions:", w.height, "(early closes:", w.filter(pl.col("half")).height, ") first", w["date"].min(), "last", w["date"].max(),
          "| not whole:", s.filter(~pl.col("whole"))["date"].to_list())
    mm = mm.sort(["date", "mod"])
    # price just before each bar (previous bar's close within the session) for the 'price at hh:mm' convention
    mm = mm.with_columns(prev_bar_close=pl.col("close").shift(1).over("date"))
    p = mm.group_by(["date", "bk"], maintain_order=True).agg(
        nb=pl.len(),
        first_open=pl.col("open").first(),
        start_conv=pl.col("prev_bar_close").first(),   # close of the last bar before the part (null for the first part)
        last_close=pl.col("close").last(),
        hi=pl.col("high").max(), lo=pl.col("low").min(),
        first_mod=pl.col("mod").first(), last_mod=pl.col("mod").last(),
    )
    p = p.with_columns(start_conv=pl.coalesce(pl.col("start_conv"), pl.col("first_open")))
    p = p.with_columns(
        abs_move=((pl.col("last_close") / pl.col("first_open") - 1).abs() * 100),
        abs_move_conv=((pl.col("last_close") / pl.col("start_conv") - 1).abs() * 100),
        rng=((pl.col("hi") - pl.col("lo")) / pl.col("first_open") * 100),
        rng_conv=((pl.col("hi") - pl.col("lo")) / pl.col("start_conv") * 100),
        rng_lo=((pl.col("hi") - pl.col("lo")) / pl.col("lo") * 100),
    )
    return p, w


for key, measure, alt in [("casey/ch-03#1", "abs_move", "abs_move_conv"), ("morgan/mh-12#1", "rng", "rng_conv")]:
    print("=" * 100)
    print(key, "SPY", measure)
    p, w = per_part("SPY")
    print("  parts with fewer than 30 bars:", p.filter(pl.col("nb") < 30).height, " sessions with fewer than 13 parts (early closes etc.):",
          p.group_by("date").agg(k=pl.len()).filter(pl.col("k") < 13).height)
    for col in (measure, alt) + (("rng_lo",) if measure == "rng" else ()):
        a = p.group_by("bk").agg(mean=pl.col(col).mean(), median=pl.col(col).median(), n=pl.len()).sort("bk")
        print(f"  average of `{col}` by part (first price = {'open of the first bar' if col == measure else 'see code'}):")
        for r in a.iter_rows(named=True):
            print(f"     {lab(r['bk'])}: mean {r['mean']:.4f}%  ({r['mean']:.2f}%)   median {r['median']:.3f}%   sessions {r['n']}")
        # which part holds the session's largest value
        top = p.sort(["date", "bk"]).group_by("date", maintain_order=True).agg(
            best_first=pl.col("bk").filter(pl.col(col) == pl.col(col).max()).min(),
            best_last=pl.col("bk").filter(pl.col(col) == pl.col(col).max()).max(),
            nmax=(pl.col(col) == pl.col(col).max()).sum(),
        )
        k0 = top.filter(pl.col("best_first") == 0).height
        k0l = top.filter(pl.col("best_last") == 0).height
        print(f"     first part holds the session's largest `{col}` on {k0} of {top.height} sessions = {100 * k0 / top.height:.1f}% "
              f"({100 * k0 / top.height:.2f}); ties resolved to the later part: {100 * k0l / top.height:.2f}; sessions with ties: {top.filter(pl.col('nmax') > 1).height}")
