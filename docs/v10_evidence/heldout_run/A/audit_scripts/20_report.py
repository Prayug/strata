"""Builds the audit report lines.  Every number in the report is recomputed here from the raw price files;
Strata's own figures are typed in from the sample.  For each item marked as agreeing, the script asserts
that the recalculated figures equal Strata's at the precision shown.

Output: audit_report.generated.jsonl next to this script (the report file itself is written separately).
"""
import sys
HERE = "/private/tmp/claude-501/-Users-prayugsigdel/933fc6e8-8ed8-4560-b611-d85a5d318c01/scratchpad/audit"
sys.path.insert(0, HERE)
import datetime as dt
import json
import math
from statistics import NormalDist

import numpy as np
import polars as pl

from common import base, minute, session_table, streaks, vix, wilson, OPEN_MOD

LINES = []


def r1(x):
    return round(float(x) + 0.0, 1)


def r2(x):
    return round(float(x) + 0.0, 2)


def r3(x):
    return round(float(x) + 0.0, 3)


def pc(k, n, nd=1):
    return round(100.0 * k / n, nd)


def wl(k, n):
    lo, hi = wilson(k, n)
    return round(100 * lo, 1), round(100 * hi, 1)


def th(n):
    return f"{n:,}"


def emit(key, recalculated, strata, agrees, note):
    assert all(ord(ch) < 128 for ch in note), [ch for ch in note if ord(ch) >= 128]
    LINES.append(json.dumps({"key": key, "recalculated": recalculated, "strata": strata, "agrees": agrees, "note": note}))


def same(a: dict, b: dict, keys):
    for k in keys:
        assert a[k] == b[k], (k, a[k], b[k])


B = {s: base(s) for s in ("SPY", "QQQ", "IWM")}
M = {s: minute(s) for s in ("SPY", "QQQ", "IWM")}
V = {s: B[s].filter(pl.col("prev_close").is_not_null()) for s in B}


# ------------------------------------------------------------------------------------------------ streak items
def streak_item(key, sym, thr, strata):
    b = B[sym]
    up, _ = streaks(b["close"].to_list())
    b = b.with_columns(up_prev=pl.Series(up).shift(1))
    v = b.filter(pl.col("prev_close").is_not_null())
    c = v.filter(pl.col("up_prev") >= thr)
    n, k = c.height, c.filter(pl.col("close") > pl.col("prev_close")).height
    an, ak = v.height, v.filter(pl.col("close") > pl.col("prev_close")).height
    lo, hi = wl(k, n)
    rec = {"days_matched": n, "days_happened": k, "days_did_not": n - k, "rate_pct": pc(k, n), "any_day_pct": pc(ak, an),
           "any_day_days": an, "interval_low_pct": lo, "interval_high_pct": hi}
    same(rec, strata, rec.keys())
    note = (f"Reproduced exactly from the daily closes. {th(n)} sessions follow {thr} or more closes in a row above the close before "
            f"(the run counted up to the previous session); {th(k)} of them closed above the previous close ({pc(k, n)}%). "
            f"On any day: {th(ak)} of {th(an)} = {pc(ak, an)}%. The {lo}%-{hi}% range is the 95% Wilson interval for {th(k)} of {th(n)}.")
    return rec, note


S1 = {"days_matched": 234, "days_happened": 116, "days_did_not": 118, "rate_pct": 49.6, "any_day_pct": 52.9, "any_day_days": 6665,
      "interval_low_pct": 43.2, "interval_high_pct": 55.9}
rec1, note1 = streak_item("casey/ch-21#3", "IWM", 5, S1)

S6 = {"days_matched": 1250, "days_happened": 623, "days_did_not": 627, "rate_pct": 49.8, "any_day_pct": 53.5, "any_day_days": 8515,
      "interval_low_pct": 47.1, "interval_high_pct": 52.6}
rec6, note6 = streak_item("casey/ch-21#1", "SPY", 3, S6)

S9 = {"days_matched": 308, "days_happened": 148, "days_did_not": 160, "rate_pct": 48.1, "any_day_pct": 53.5, "any_day_days": 8515,
      "interval_low_pct": 42.5, "interval_high_pct": 53.6}
rec9, note9 = streak_item("casey/ch-21#2", "SPY", 5, S9)


# ------------------------------------------------------------------------------------------------ movement items
def movement_item(K, with_vix):
    b = B["SPY"]
    if with_vix:
        b = b.join(vix().select("date", vix_close="close"), on="date", how="left").sort("date")
    c = b["close"].to_numpy(); o = b["s_open"].to_numpy(); h = b["high"].to_numpy(); l = b["low"].to_numpy(); n = len(c)
    prev_ret = np.full(n, np.nan); prev_ret[2:] = c[1:-1] / c[:-2] - 1
    if with_vix:
        pv = np.full(n, np.nan); pv[1:] = b["vix_close"].to_numpy()[:-1]
        cond = lambda t: prev_ret[t] <= -0.02 and pv[t] > 35
    else:
        cond = lambda t: prev_ret[t] <= -0.02
    all_t = np.array([t for t in range(2, n) if cond(t)])
    t = np.array([x for x in all_t if x + K < n])

    def st(start, first_sess, last_sess):
        r = c[last_sess] / start - 1
        fav = np.array([h[a:z + 1].max() for a, z in zip(first_sess, last_sess)]) / start - 1
        adv = np.array([l[a:z + 1].min() for a, z in zip(first_sess, last_sess)]) / start - 1
        return {"n": len(r), "avg": r2(100 * r.mean()), "median": r2(100 * np.median(r)), "higher": r1(100 * (r > 0).mean()),
                "q25": r2(100 * np.quantile(r, .25)), "q75": r2(100 * np.quantile(r, .75)),
                "fav": r2(100 * np.median(fav)), "adv": r2(100 * np.median(adv)), "r": r}

    out = {
        "open": st(o[t], t, t + K),               # open of the session after the fall -> close K sessions after that session
        "close_t": st(c[t], t + 1, t + K),        # close of the session after the fall -> close K sessions later
        "close_fall": st(c[t - 1], t, t + K - 1), # close of the session that fell -> close K sessions later
        "open_k": st(o[t], t, t + K - 1),         # open of the session after the fall -> close of the K-th session
    }
    # bootstrap of the mean for the figures Strata reports (percentile and BCa), fixed seed
    r = out["open"]["r"]
    rng = np.random.default_rng(1)
    boots = rng.choice(r, size=(10000, len(r)), replace=True).mean(axis=1)
    out["boot_pct"] = (r2(100 * np.quantile(boots, .025)), r2(100 * np.quantile(boots, .975)))
    # bias-corrected and accelerated (BCa) interval from the same resamples
    nd = NormalDist()
    z0 = nd.inv_cdf(float((boots < r.mean()).mean()))
    jk = (r.sum() - r) / (len(r) - 1)
    acc = ((jk.mean() - jk) ** 3).sum() / (6 * (((jk.mean() - jk) ** 2).sum() ** 1.5))
    adj = lambda z: nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    out["boot_bca"] = (r2(100 * np.quantile(boots, adj(nd.inv_cdf(.025)))), r2(100 * np.quantile(boots, adj(nd.inv_cdf(.975)))))
    se = r.std(ddof=1) / math.sqrt(len(r))
    out["normal"] = (r2(100 * (r.mean() - 1.96 * se)), r2(100 * (r.mean() + 1.96 * se)))
    return len(all_t), len(t), out


for key, K, with_vix, S in [
    ("morgan/mh-18#1", 5, False, {"days_matched": 341, "avg_move_pct": 0.85, "median_move_pct": 0.78, "ended_higher_pct": 60.1,
                                  "quartile_low_pct": -1.73, "quartile_high_pct": 3.38, "favourable_median_pct": 2.71,
                                  "adverse_median_pct": -2.54, "avg_range_low_pct": 0.36, "avg_range_high_pct": 1.33}),
    ("jordan/jh-16#1", 10, True, {"days_matched": 106, "avg_move_pct": 1.72, "median_move_pct": 3.08, "ended_higher_pct": 67.0,
                                  "quartile_low_pct": -1.81, "quartile_high_pct": 6.98, "favourable_median_pct": 6.95,
                                  "adverse_median_pct": -4.77, "avg_range_low_pct": 0.07, "avg_range_high_pct": 3.16}),
]:
    n_all, n_used, o_ = movement_item(K, with_vix)
    A, Ct, Cf, Ok = o_["open"], o_["close_t"], o_["close_fall"], o_["open_k"]
    # Strata's figures are the open-based ones: assert that, so the note's claim is backed
    assert (n_used, A["avg"], A["median"], A["higher"], A["q25"], A["q75"], A["fav"], A["adv"]) == (
        S["days_matched"], S["avg_move_pct"], S["median_move_pct"], S["ended_higher_pct"], S["quartile_low_pct"], S["quartile_high_pct"],
        S["favourable_median_pct"], S["adverse_median_pct"]), (key, A)
    rec = {
        "days_matched": n_used,
        "from_that_sessions_close_avg_move_pct": Ct["avg"], "from_that_sessions_close_median_pct": Ct["median"],
        "from_that_sessions_close_ended_higher_pct": Ct["higher"],
        "from_that_sessions_close_quartile_low_pct": Ct["q25"], "from_that_sessions_close_quartile_high_pct": Ct["q75"],
        "from_fall_sessions_close_avg_move_pct": Cf["avg"], "from_fall_sessions_close_median_pct": Cf["median"],
        "from_fall_sessions_close_ended_higher_pct": Cf["higher"],
        "from_open_to_close_%d_sessions_later_avg_move_pct" % K: A["avg"],
        "from_open_to_close_%d_sessions_later_median_pct" % K: A["median"],
        "from_open_to_close_%d_sessions_later_ended_higher_pct" % K: A["higher"],
    }
    cond_txt = ("a session that fell 2% or more with the VIX closing above 35 that day" if with_vix else "a session that fell 2% or more")
    note = (
        f"The day count agrees: {th(n_all)} sessions follow {cond_txt}, and {th(n_used)} of them have {K} later sessions to measure. "
        f"The move does not match the reading, which says it is measured 'from that session's close to the close {K} sessions later'. "
        f"From the close of the session after the fall to the close {K} sessions later I get {Ct['avg']:+.2f}% on average "
        f"(median {Ct['median']:+.2f}%, higher on {Ct['higher']}%, middle half {Ct['q25']:+.2f}% to {Ct['q75']:+.2f}%); "
        f"if 'that session' is the one that fell, {Cf['avg']:+.2f}% (median {Cf['median']:+.2f}%, higher on {Cf['higher']}%). "
        f"Strata's {S['avg_move_pct']:+.2f}% / {S['median_move_pct']:+.2f}% / {S['ended_higher_pct']}%, its quartiles "
        f"({S['quartile_low_pct']:+.2f}%, {S['quartile_high_pct']:+.2f}%) and its favourable/adverse medians "
        f"({S['favourable_median_pct']:+.2f}%, {S['adverse_median_pct']:+.2f}%) are all reproduced exactly when the move runs from the OPEN "
        f"of the session after the fall to the close {K} sessions after that session, as the headline's 'from the open' says. "
        f"So the arithmetic is sound, but the start point calculated (the open) is not the one the reading states (the close), and the window "
        f"then spans {K + 1} sessions rather than {K} (open to the close of the {K}th session would be {Ok['avg']:+.2f}%, higher on {Ok['higher']}%). "
        f"The 'likely range' of the average ({S['avg_range_low_pct']:+.2f}% to {S['avg_range_high_pct']:+.2f}%) is consistent with a bootstrap of the "
        f"open-based mean (mine, 10,000 resamples: bias-corrected {o_['boot_bca'][0]:+.2f}% to {o_['boot_bca'][1]:+.2f}%, plain percentile "
        f"{o_['boot_pct'][0]:+.2f}% to {o_['boot_pct'][1]:+.2f}%; normal approximation {o_['normal'][0]:+.2f}% to {o_['normal'][1]:+.2f}%); "
        f"its method is not stated and resampling is random, so it cannot be matched to the last digit."
    )
    if key == "morgan/mh-18#1":
        rec2, note2 = rec, note; S2 = S
    else:
        rec24, note24 = rec, note; S24 = S
        note24 = note24 + (" The reading repeats the 2% condition twice; that has no effect. 'The VIX closed above 35 the session before' is taken as "
                           "the VIX close on the day of the fall.")


# ------------------------------------------------------------------------------------------------ touch items
def touch_plain(sym, x):
    v = V[sym]
    hi = pl.when(pl.col("whole")).then(pl.col("m_high")).otherwise(pl.col("high"))
    k = v.filter(hi >= pl.col("s_open") * (1 + x / 100)).height
    k_daily = v.filter(pl.col("high") >= pl.col("open") * (1 + x / 100)).height
    return v.height, k, k_daily


n3, k3, k3d = touch_plain("QQQ", 1.5)
lo3, hi3 = wl(k3, n3)
rec3 = {"days_matched": n3, "days_happened": k3, "days_did_not": n3 - k3, "rate_pct": pc(k3, n3), "interval_low_pct": lo3, "interval_high_pct": hi3}
S3 = {"days_matched": 6973, "days_happened": 1163, "days_did_not": 5810, "rate_pct": 16.7, "interval_low_pct": 15.8, "interval_high_pct": 17.6}
same(rec3, S3, rec3.keys())
note3 = (f"No reading is shown, so this is checked against the summary: the price trades 1.5% or more above the session's open at some point "
         f"between the open and the close. Reproduced exactly: {th(k3)} of {th(n3)} days ({pc(k3, n3)}%), {th(n3 - k3)} not, using the 09:30 bar's "
         f"open and the one-minute highs where a whole session of minute bars is stored and the daily record before that. "
         f"The {lo3}%-{hi3}% range is the 95% Wilson interval. (From the daily record alone it would be {th(k3d)}, also {pc(k3d, n3)}%.)")

n15, k15, k15d = touch_plain("SPY", 5.0)
lo15, hi15 = wl(k15, n15)
rec15 = {"days_matched": n15, "days_happened": k15, "days_did_not": n15 - k15, "rate_pct": pc(k15, n15), "interval_low_pct": lo15, "interval_high_pct": hi15}
S15 = {"days_matched": 8515, "days_happened": 24, "days_did_not": 8491, "rate_pct": 0.3, "interval_low_pct": 0.2, "interval_high_pct": 0.4}
same(rec15, S15, rec15.keys())
assert k15 == k15d
note15 = (f"No reading is shown, so this is checked against the summary: the price trades 5% or more above the session's open at some point "
          f"between the open and the close. Reproduced exactly: {th(k15)} of {th(n15)} days ({pc(k15, n15)}%), {th(n15 - k15)} not; the count is the "
          f"same from the daily record and from the one-minute bars. The {lo15}%-{hi15}% range is the 95% Wilson interval.")

# casey/ch-18#1: plain touch +1% and the two-level rule that reproduces Strata's counts
n7, k7, k7d = touch_plain("QQQ", 1.0)
lo7, hi7 = wl(k7, n7)
v = V["QQQ"]
pre = v.filter(~pl.col("whole"))
pre_up = pre.filter(pl.col("high") >= pl.col("open") * 1.01)
pre_both = pre_up.filter(pl.col("low") <= pl.col("open") * 0.995).height
pre_hit = pre_up.height - pre_both
post = v.filter(pl.col("whole")).select("date", "s_open")
mm = M["QQQ"].join(post, on="date", how="inner").with_columns(
    hu=pl.col("high") >= pl.col("s_open") * 1.01, hd=pl.col("low") <= pl.col("s_open") * 0.995)
g = mm.group_by("date").agg(tu=pl.col("mod").filter(pl.col("hu")).min(), td=pl.col("mod").filter(pl.col("hd")).min())
post_hit = g.filter(pl.col("tu").is_not_null() & (pl.col("td").is_null() | (pl.col("tu") < pl.col("td")))).height
post_same = g.filter(pl.col("tu").is_not_null() & pl.col("td").is_not_null() & (pl.col("tu") == pl.col("td"))).height
post_both = g.filter(pl.col("tu").is_not_null() & pl.col("td").is_not_null())
post_both_n, post_both_up_first = post_both.height, post_both.filter(pl.col("tu") < pl.col("td")).height
hap, und = pre_hit + post_hit, pre_both + post_same
didnt = n7 - hap - und
S7 = {"days_matched": 6973, "days_happened": 1202, "days_did_not": 4991, "days_order_unknown": 780, "rate_pct": 19.4,
      "interval_low_pct": 18.4, "interval_high_pct": 20.4}
assert (hap, didnt, und) == (S7["days_happened"], S7["days_did_not"], S7["days_order_unknown"]), (hap, didnt, und)
assert pc(hap, hap + didnt) == S7["rate_pct"] and wl(hap, hap + didnt) == (S7["interval_low_pct"], S7["interval_high_pct"])
rec7 = {"days_matched": n7, "days_touched_plus_1pct_from_open": k7, "days_did_not": n7 - k7, "rate_pct": pc(k7, n7),
        "interval_low_pct": lo7, "interval_high_pct": hi7,
        "inferred_rule_plus_1pct_before_minus_0_5pct_days_happened": hap, "inferred_rule_days_did_not": didnt,
        "inferred_rule_days_order_unknown": und, "inferred_rule_rate_of_decided_days_pct": pc(hap, hap + didnt),
        "inferred_rule_rate_of_all_days_pct": pc(hap, n7)}
note7 = (
    f"No reading is shown and the summary says only 'touch +1% from the open, open to close'. By that definition QQQ traded 1% or more above "
    f"its open on {th(k7)} of {th(n7)} days ({pc(k7, n7)}%, interval {lo7}%-{hi7}%), not {th(S7['days_happened'])}, and with a single level there "
    f"is no order of events to be unsure about. Best explanation: Strata's three counts are reproduced exactly by a rule with a second "
    f"level that is stated nowhere in what I was given - 'reaches +1% above the open before -0.5% below it, by the close': happened {th(hap)} "
    f"({th(pre_hit)} daily-bar days plus {th(post_hit)} minute-bar days), did not {th(didnt)}, both levels inside one daily bar {th(und)}. "
    f"Even under that rule the headline is inconsistent: {th(hap)} of {th(n7)} is {pc(hap, n7)}%; the 19.4% (and the 18.4%-20.4% interval) is "
    f"{th(hap)} of the {th(hap + didnt)} decided days. Convention concern: leaving the {th(und)} undecidable days out is not neutral - every one of "
    f"them did reach +1% - so the rate over all days lies somewhere between {pc(hap, n7)}% and {pc(hap + und, n7)}% "
    f"(in the minute-bar years, where the order is known, +1% came first on {post_both_up_first} of the {post_both_n} sessions that touched both levels)."
)

# ------------------------------------------------------------------------------------------------ IWM gap fill
v = V["IWM"].filter(pl.col("date") >= dt.date(2016, 1, 1))
lo_min = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
c17 = v.filter(pl.col("s_open") / pl.col("prev_close") - 1 >= 0.01)
n17, k17 = c17.height, c17.filter(lo_min <= pl.col("prev_close")).height
c17d = v.filter(pl.col("open") / pl.col("prev_close") - 1 >= 0.01)
n17d, k17d = c17d.height, c17d.filter(pl.col("low") <= pl.col("prev_close")).height
lo17, hi17 = wl(k17, n17)
rec17 = {"days_matched": n17, "days_happened": k17, "days_did_not": n17 - k17, "rate_pct": pc(k17, n17), "interval_low_pct": lo17, "interval_high_pct": hi17}
S17 = {"days_matched": 220, "days_happened": 54, "days_did_not": 166, "rate_pct": 24.5, "interval_low_pct": 19.3, "interval_high_pct": 30.6}
same(rec17, S17, rec17.keys())
note17 = (f"Reproduced exactly under the stated conventions. Since 2016 the 09:30 bar's open was 1% or more above the previous daily close on "
          f"{n17} sessions; the one-minute lows came back to the previous close on {k17} of them ({pc(k17, n17)}%), {n17 - k17} did not. "
          f"The {lo17}%-{hi17}% range is the 95% Wilson interval. (With the daily record's open and low instead it would be {k17d} of {n17d}.)")

# ------------------------------------------------------------------------------------------------ previous high / low items
v = V["SPY"]
c14 = v.filter(pl.col("high") > pl.col("prev_high"))
n14, k14 = c14.height, c14.filter(pl.col("close") < pl.col("prev_high")).height
hi_min = pl.when(pl.col("whole")).then(pl.col("m_high")).otherwise(pl.col("high"))
c14m = v.filter(hi_min > pl.col("prev_high"))
n14m, k14m = c14m.height, c14m.filter(pl.col("close") < pl.col("prev_high")).height
rec14 = {"sessions_matched": n14, "sessions_happened": k14, "sessions_did_not": n14 - k14, "share_pct": pc(k14, n14),
         "first_session": str(c14["date"].min()), "last_session": str(c14["date"].max())}
S14 = {"sessions_matched": 4579, "sessions_happened": 1854, "sessions_did_not": 2725, "share_pct": 40.5,
       "first_session": "1993-02-01", "last_session": "2026-11-27"}
same(rec14, S14, rec14.keys())
note14 = (f"Reproduced exactly from the daily record: the high was above the previous session's high on {th(n14)} sessions and the close was below "
          f"that previous high on {th(k14)} of them ({pc(k14, n14)}%); {th(n14 - k14)} did not; the first and last such sessions are "
          f"{c14['date'].min()} and {c14['date'].max()}. Taking the session's high from the one-minute bars where a whole session is stored "
          f"gives {th(k14m)} of {th(n14m)}, also {pc(k14m, n14m)}%.")

v16 = v.filter(pl.col("date") >= dt.date(2016, 1, 1))
c22 = v16.filter(pl.col("low") < pl.col("prev_low"))
n22, k22 = c22.height, c22.filter(pl.col("close") > pl.col("prev_close")).height
lo_min = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
c22m = v16.filter(lo_min < pl.col("prev_low"))
n22m, k22m = c22m.height, c22m.filter(pl.col("close") > pl.col("prev_close")).height
rec22 = {"sessions_matched": n22, "sessions_happened": k22, "sessions_did_not": n22 - k22, "share_pct": pc(k22, n22),
         "first_session": str(c22["date"].min()), "last_session": str(c22["date"].max())}
S22 = {"sessions_matched": 1198, "sessions_happened": 277, "sessions_did_not": 921, "share_pct": 23.1,
       "first_session": "2016-01-04", "last_session": "2026-11-25"}
same(rec22, S22, rec22.keys())
note22 = (f"Reproduced exactly from the daily record: since 2016 the low was below the previous session's low on {th(n22)} sessions and the close "
          f"was above the previous close on {k22} of them ({pc(k22, n22)}%); {n22 - k22} did not; the first and last such sessions are "
          f"{c22['date'].min()} and {c22['date'].max()}. With the session's low taken from the one-minute bars the figure is {k22m} of {th(n22m)} "
          f"({pc(k22m, n22m)}%): one session differs between the two price sources.")

# alex/ah-27#1
b = B["SPY"].with_columns(prev_m_low=pl.col("m_low").shift(1), prev_whole=pl.col("whole").shift(1).fill_null(False))
v = b.filter(pl.col("prev_close").is_not_null())
cur_m = pl.when(pl.col("whole")).then(pl.col("m_low")).otherwise(pl.col("low"))
prev_m = pl.when(pl.col("prev_whole")).then(pl.col("prev_m_low")).otherwise(pl.col("prev_low"))
n20 = v.height
a_lt_d = v.filter(pl.col("low") < pl.col("prev_low")).height
a_lt_m = v.filter(cur_m < pl.col("prev_low")).height
a_le_d = v.filter(pl.col("low") <= pl.col("prev_low")).height
a_le_m = v.filter(cur_m <= pl.col("prev_low")).height
a_le_mm = v.filter(cur_m <= prev_m).height
ties = v.filter(pl.col("low") == pl.col("prev_low")).height
src_diff = v.filter(pl.col("prev_whole") & ((pl.col("prev_m_low") - pl.col("prev_low")).abs() > 0.0100001)).height
ex = b.filter(pl.col("date") == dt.date(2022, 10, 4)).select("low", "m_low").row(0)
assert a_lt_d == a_lt_m
S20 = {"days_matched": 8515, "days_happened": 3881, "days_did_not": 4634, "rate_pct": 45.6, "interval_low_pct": 44.5, "interval_high_pct": 46.6,
       "strata_own_check_days_happened": 3870}
assert a_le_mm == S20["days_happened"]
lo20, hi20 = wl(a_lt_d, n20)
rec20 = {"days_matched": n20, "days_happened": a_lt_d, "days_did_not": n20 - a_lt_d, "rate_pct": pc(a_lt_d, n20),
         "interval_low_pct": lo20, "interval_high_pct": hi20,
         "days_at_or_below_daily_record_lows": a_le_d, "days_at_or_below_minute_bar_lows": a_le_m, "at_or_below_rate_pct": pc(a_le_d, n20),
         "days_at_or_below_previous_low_taken_from_minute_bars": a_le_mm}
assert pc(a_le_d, n20) == pc(a_le_m, n20)
note20 = (
    f"By the convention stated for the previous session's high, applied here to the low (the session's lows against the previous session's low "
    f"from the daily record), SPY traded below the "
    f"previous session's low on {th(a_lt_d)} of {th(n20)} days ({pc(a_lt_d, n20)}%, interval {lo20}%-{hi20}%); the count is the same whether the "
    f"session's low comes from the one-minute bars (2016 on) or the daily record. Counting a low equal to the previous low as well gives "
    f"{th(a_le_d)} (daily lows) or {th(a_le_m)} (minute lows from 2016), {pc(a_le_d, n20)}%. Strata's {th(S20['days_happened'])} ({S20['rate_pct']}%) "
    f"is reproduced exactly only by counting 'at or below' AND taking the previous session's low from that session's one-minute bars (2016 on) "
    f"instead of the daily record. So two things depart from what is stated: 'trades below' is counted as 'at or below' ({ties} sessions have a low "
    f"exactly equal to the previous low), and the previous low is not the daily record's (the two sources differ by more than a cent on {src_diff} "
    f"sessions since 2016, e.g. 2022-10-04: daily-record low {ex[0]:.2f}, minute-bar low {ex[1]:.2f}). Strata's own independent check also "
    f"disagreed (3,870); my nearest figure to that is {th(a_le_m)}. The answer sam/sh-21#1 counts the same phrase strictly and against the daily "
    f"record ({th(n22)} sessions since 2016), so the two answers use different rules for 'trades below the previous session's low'."
)

# alex/ah-01#1
v = V["QQQ"]
n21, k21 = v.height, v.filter(pl.col("close") > pl.col("prev_close")).height
lo21, hi21 = wl(k21, n21)
rec21 = {"days_matched": n21, "days_happened": k21, "days_did_not": n21 - k21, "rate_pct": pc(k21, n21), "interval_low_pct": lo21, "interval_high_pct": hi21}
S21 = {"days_matched": 6973, "days_happened": 3785, "days_did_not": 3188, "rate_pct": 54.3, "interval_low_pct": 53.1, "interval_high_pct": 55.4}
same(rec21, S21, rec21.keys())
note21 = (f"Reproduced exactly from the daily closes: QQQ closed above the previous session's close on {th(k21)} of {th(n21)} days "
          f"({pc(k21, n21)}%), {th(n21 - k21)} not. The {lo21}%-{hi21}% range is the 95% Wilson interval.")

# ------------------------------------------------------------------------------------------------ IWM down streak, two measures
b = B["IWM"]
c = b["close"].to_numpy(); o_s = b["s_open"].to_numpy(); n = len(c)
_, dn = streaks(list(c)); dn = np.array(dn)
match = np.zeros(n, dtype=bool); match[1:] = dn[:-1] >= 4
valid = np.zeros(n, dtype=bool); valid[1:] = True
prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]


def fwd(x, k):
    out = np.full(n, np.nan)
    if k == 0:
        return x.astype(float).copy()
    out[:-k] = x[k:]
    return out


def rate(a, b_):
    ok = ~np.isnan(a) & ~np.isnan(b_)
    e = np.zeros(n, dtype=bool); e[ok] = a[ok] > b_[ok]
    return int((e & match & ok).sum()), int((match & ok).sum()), int((e & valid & ok).sum()), int((valid & ok).sum())


v1 = rate(fwd(c, 5), prev_c)       # close 5 sessions later above the previous session's close
v1b = rate(fwd(c, 4), prev_c)      # counting the 5 sessions from the previous session
v1n = rate(fwd(c, 0), prev_c)      # the next session itself closes above the previous close
v2 = rate(fwd(c, 0), o_s)          # the next session closes above its own open
v2s = rate(fwd(c, 5), o_s)         # close 5 sessions later above that session's open (what Strata's figures equal)
S16 = {"days_matched": 261, "closes_above_previous_close_pct": 56.7, "closes_above_previous_close_any_day_pct": 55.1,
       "closes_above_its_own_open_pct": 53.3, "closes_above_its_own_open_any_day_pct": 54.6}
assert v1[1] == 261 and pc(v1[0], v1[1]) == 56.7 and pc(v1[2], v1[3]) == 55.1
assert pc(v2s[0], v2s[1]) == 53.3 and pc(v2s[2], v2s[3]) == 54.6
rec16 = {"days_matched": v1[1],
         "closes_above_previous_close_pct": pc(v1[0], v1[1]), "closes_above_previous_close_any_day_pct": pc(v1[2], v1[3]),
         "closes_above_its_own_open_pct": pc(v2[0], v2[1]), "closes_above_its_own_open_any_day_pct": pc(v2[2], v2[3]),
         "close_5_sessions_later_above_that_sessions_open_pct": pc(v2s[0], v2s[1]),
         "close_5_sessions_later_above_that_sessions_open_any_day_pct": pc(v2s[2], v2s[3])}
note16 = (
    f"The day count agrees ({v1[1]} sessions follow 4 or more down closes in a row) and so does the first measure: the close 5 sessions after "
    f"the session that follows the streak is above the previous session's close on {v1[0]} of {v1[1]} = {pc(v1[0], v1[1])}%, any day "
    f"{th(v1[2])} of {th(v1[3])} = {pc(v1[2], v1[3])}%. (Counting the 5 sessions from the previous session instead gives {v1b[0]} of {v1b[1]} = "
    f"{pc(v1b[0], v1b[1])}%, any day {pc(v1b[2], v1b[3])}%.) The second measure differs: 'the next session closes above its own open' is "
    f"{v2[0]} of {v2[1]} = {pc(v2[0], v2[1])}%, any day {th(v2[2])} of {th(v2[3])} = {pc(v2[2], v2[3])}%. Strata's 53.3% (any day 54.6%) is "
    f"reproduced exactly as 'the close 5 sessions later is above that session's open' ({v2s[0]} of {v2s[1]}; any day {th(v2s[2])} of {th(v2s[3])}): "
    f"the 5-session horizon was applied to the second measure too, which is not what its label says. The headline's first label also drops the "
    f"horizon ('the next session closes above the previous close'); read literally that would be {v1n[0]} of {v1n[1]} = {pc(v1n[0], v1n[1])}%, "
    f"any day {pc(v1n[2], v1n[3])}%."
)

# ------------------------------------------------------------------------------------------------ '0 of N' movement items
def mv(df, start_col):
    r = (df["close"] / df[start_col] - 1).to_numpy()
    d = df["date"].to_list()
    i_max, i_min = int(np.argmax(r)), int(np.argmin(r))
    return {"n": len(r), "avg": r3(100 * r.mean()), "median": r3(100 * np.median(r)), "up": int((r > 0).sum()), "down": int((r < 0).sum()),
            "flat": int((r == 0).sum()), "max": r2(100 * r[i_max]), "max_date": str(d[i_max]), "min": r2(100 * r[i_min]), "min_date": str(d[i_min])}


def zero_item(sym, n_strata, m, what, extra=""):
    assert m["n"] == n_strata, (m["n"], n_strata)
    assert m["up"] > 0 and m["down"] > 0
    rec = {"days_matched": m["n"], "avg_move_pct": m["avg"], "median_move_pct": m["median"], "days_ended_higher": m["up"],
           "days_ended_lower": m["down"], "days_unchanged": m["flat"], "ended_higher_pct": pc(m["up"], m["n"]),
           "largest_rise_pct": m["max"], "largest_fall_pct": m["min"]}
    strata = {"days_matched": n_strata, "days_happened": 0, "days_did_not": n_strata, "rate_pct": 0.0}
    note = (f"The day count agrees exactly ({th(m['n'])}{extra}). The '0 days it happened (0.0%)' cannot be reproduced: no reading is shown and the "
            f"summary describes a movement ({what}), which is a size, not an event that happens or does not. Measured as the summary "
            f"describes, {sym} moved {m['avg']:+.3f}% on average (median {m['median']:+.3f}%), ending higher on {th(m['up'])} of {th(m['n'])} days "
            f"({pc(m['up'], m['n'])}%), lower on {th(m['down'])} and unchanged on {m['flat']}. The largest rise was {m['max']:+.2f}% ({m['max_date']}) "
            f"and the largest fall {m['min']:+.2f}% ({m['min_date']}), so 0 could be a true count only for a move beyond those, and nothing shown "
            f"states such an event. The headline is therefore not a real count; it looks like a movement question answered through the "
            f"'how often' output with no event defined.")
    return rec, strata, note


m8 = mv(V["IWM"].filter(pl.col("date") >= dt.date(2000, 1, 1)), "s_open")
rec8, S8, note8 = zero_item("IWM", 6665, m8, "the return from the session's open to its close, since 2000-01-01",
                            "; IWM's record starts 2000-05-26")
m10 = mv(V["SPY"].filter(pl.col("date") >= dt.date(2016, 1, 1)), "s_open")
rec10, S10, note10 = zero_item("SPY", 2742, m10, "the return from the session's open to its close, since 2016-01-01")
m18 = mv(V["SPY"], "s_open")
rec18, S18, note18 = zero_item("SPY", 8515, m18, "the return from the session's open to its close")
g19 = V["SPY"].filter(pl.col("s_open") < pl.col("prev_close"))
g19d = V["SPY"].filter(pl.col("open") < pl.col("prev_close")).height
rec19, S19, note19 = zero_item("SPY", 3698, mv(g19, "s_open"), "the return from the session's open to its close on days that opened below the previous close",
                               f" sessions whose open - the 09:30 bar's from 2016, the daily record's before - was below the previous daily close; "
                               f"{th(g19d)} with the daily record's open throughout")
g23 = V["QQQ"].filter(pl.col("s_open") < pl.col("prev_close"))
g23d = V["QQQ"].filter(pl.col("open") < pl.col("prev_close")).height
rec23, S23, note23 = zero_item("QQQ", 3056, mv(g23, "s_open"), "the return from the session's open to its close on days that opened below the previous close",
                               f" sessions whose open - the 09:30 bar's from 2016, the daily record's before - was below the previous daily close; "
                               f"{th(g23d)} with the daily record's open throughout")
whole = B["SPY"].filter(pl.col("whole")).select("date", "close", "s_open", "half")
p1430 = M["SPY"].filter(pl.col("mod") == 14 * 60 + 29).select("date", px="close")
j5 = whole.join(p1430, on="date", how="inner")
n_whole_spy, n_half_spy = whole.height, whole.filter(pl.col("half")).height
rec5, S5, note5 = zero_item("SPY", 2719, mv(j5, "px"), "the return from 14:30 to the close",
                            f" = the {th(n_whole_spy)} whole minute-bar sessions less the {n_half_spy} early closes, which have no 14:30 price; "
                            f"the 14:30 price is the close of the 14:29 bar and the close is the daily record's")

# ------------------------------------------------------------------------------------------------ 12:00 item
p12 = M["SPY"].filter(pl.col("mod") == 12 * 60 - 1).select("date", px="close")
j = whole.join(p12, on="date", how="inner")
assert j.height == whole.height
n_any, k_any = j.height, j.filter(pl.col("close") > pl.col("s_open")).height
below = j.filter(pl.col("px") < pl.col("s_open"))
n11, k11 = below.height, below.filter(pl.col("close") > pl.col("s_open")).height
lo11, hi11 = wl(k11, n11)
S11 = {"days_matched": 2741, "days_happened": 1487, "days_did_not": 1254, "rate_pct": 54.3, "any_day_pct": 54.3, "any_day_days": 2741,
       "interval_low_pct": 52.4, "interval_high_pct": 56.1}
assert (n_any, k_any) == (S11["days_matched"], S11["days_happened"])
rec11 = {"days_matched": n11, "days_happened": k11, "days_did_not": n11 - k11, "rate_pct": pc(k11, n11), "any_day_pct": pc(k_any, n_any),
         "any_day_days": n_any, "interval_low_pct": lo11, "interval_high_pct": hi11}
note11 = (
    f"The 12:00 condition was not applied. SPY was below its open at 12:00 (close of the 11:59 bar below the 09:30 bar's open) on "
    f"{th(n11)} of the {th(n_any)} whole sessions, and closed above its open on {k11} of those = {pc(k11, n11)}% (interval {lo11}%-{hi11}%). "
    f"Strata's {th(S11['days_happened'])} of {th(S11['days_matched'])} (54.3%) is the count over ALL {th(n_any)} sessions - exactly the figure it "
    f"also gives as 'any day' - so the headline's 'when it is below the open at 12:00' is not true of the days counted. (Strata's 'days it "
    f"didn't' of 1,254 is 2,741 minus 1,487; that it equals the number of sessions below the open at 12:00 is a coincidence.)"
)
assert n_any - k_any == 1254 and n11 == 1254

# ------------------------------------------------------------------------------------------------ timing items
def whole_minutes(sym):
    m = M[sym]
    s = session_table(m)
    w = s.filter(pl.col("whole")).select("date")
    not_whole = [str(d) for d in s.filter(~pl.col("whole"))["date"].to_list()]
    return m.join(w, on="date", how="inner").with_columns(bk=((pl.col("mod") - OPEN_MOD) // 30).cast(pl.Int32)).sort(["date", "mod"]), w.height, not_whole


mm, nw, not_whole_iwm = whole_minutes("IWM")
g = mm.group_by("date", maintain_order=True).agg(
    t_first=pl.col("mod").filter(pl.col("low") == pl.col("low").min()).min(),
    t_last=pl.col("mod").filter(pl.col("low") == pl.col("low").min()).max())
g = g.with_columns(bf=((pl.col("t_first") - OPEN_MOD) // 30), bl=((pl.col("t_last") - OPEN_MOD) // 30))
cnt = {r["bf"]: r["k"] for r in g.group_by("bf").agg(k=pl.len()).iter_rows(named=True)}
cntl = {r["bl"]: r["k"] for r in g.group_by("bl").agg(k=pl.len()).iter_rows(named=True)}
order = sorted(cnt, key=lambda k_: -cnt[k_])
assert order[0] == 0 and order[1] == 12
rec4 = {"sessions": nw, "share_0930_1000_pct": pc(cnt[0], nw), "share_1530_1600_pct": pc(cnt[12], nw), "uniform_share_pct": round(100 / 13, 1)}
S4 = {"sessions": 2741, "share_0930_1000_pct": 33.8, "share_1530_1600_pct": 13.2, "uniform_share_pct": 7.7}
same(rec4, S4, rec4.keys())
note4 = (f"Reproduced from the one-minute bars: {th(nw)} whole sessions from {mm['date'].min()} to {mm['date'].max()} ({', '.join(not_whole_iwm)} is "
         f"left out because its bars stop at 15:30). The session's low was first made between 09:30 and 10:00 on {cnt[0]} sessions "
         f"({pc(cnt[0], nw)}%) and between 15:30 and 16:00 on {cnt[12]} ({pc(cnt[12], nw)}%), the two largest shares; one part in 13 is 7.7%. "
         f"Where the same low price occurs in more than one bar the first is taken; taking the last would give {pc(cntl[0], nw)}% and "
         f"{pc(cntl[12], nw)}%. The 22 early-close sessions are included although they have no afternoon parts.")


def lab(bk):
    a = OPEN_MOD + 30 * bk
    return f"{a // 60:02d}:{a % 60:02d}-{(a + 30) // 60:02d}:{(a + 30) % 60:02d}"


mm, nws, not_whole_spy = whole_minutes("SPY")
p = mm.group_by(["date", "bk"], maintain_order=True).agg(
    first_open=pl.col("open").first(), last_close=pl.col("close").last(), hi=pl.col("high").max(), lo=pl.col("low").min())
p = p.with_columns(abs_move=((pl.col("last_close") / pl.col("first_open") - 1).abs() * 100),
                   rng=((pl.col("hi") - pl.col("lo")) / pl.col("first_open") * 100))


def part_stats(col):
    a = {r["bk"]: r["mean"] for r in p.group_by("bk").agg(mean=pl.col(col).mean()).iter_rows(named=True)}
    ranked = sorted(a, key=lambda k_: -a[k_])
    top = p.group_by("date", maintain_order=True).agg(best=pl.col("bk").filter(pl.col(col) == pl.col(col).max()).min())
    k0 = top.filter(pl.col("best") == 0).height
    return a, ranked, k0, top.height


a13, rk13, k13, n13 = part_stats("abs_move")
assert rk13[0] == 0 and rk13[1] == 1 and rk13[-1] == 6, rk13
rec13 = {"sessions": n13, "avg_move_0930_1000_pct": r2(a13[0]), "avg_move_1000_1030_pct": r2(a13[1]), "avg_move_quietest_1230_1300_pct": r2(a13[6]),
         "largest_move_in_0930_1000_share_pct": pc(k13, n13)}
S13 = {"sessions": 2741, "avg_move_0930_1000_pct": 0.21, "avg_move_1000_1030_pct": 0.18, "avg_move_quietest_1230_1300_pct": 0.11,
       "largest_move_in_0930_1000_share_pct": 21.1}
same(rec13, S13, rec13.keys())
note13 = (f"Reproduced from the one-minute bars on {th(n13)} whole sessions ({mm['date'].min()} to {mm['date'].max()}): the average absolute move "
          f"from the first bar's open to the last bar's close of each half hour is {a13[0]:.3f}% for 09:30-10:00, {a13[1]:.3f}% for 10:00-10:30 "
          f"(second) and lowest for {lab(rk13[-1])} at {a13[rk13[-1]]:.3f}%; 09:30-10:00 held the session's largest move on {k13} sessions "
          f"({pc(k13, n13)}%). All agree at the precision shown.")

a26, rk26, k26, n26 = part_stats("rng")
assert rk26[0] == 0 and rk26[1] == 1 and rk26[-1] == 8, rk26
rec26 = {"sessions": n26, "avg_range_0930_1000_pct": r2(a26[0]), "avg_range_1000_1030_pct": r2(a26[1]), "avg_range_quietest_1330_1400_pct": r2(a26[8]),
         "widest_range_in_0930_1000_share_pct": pc(k26, n26)}
S26 = {"sessions": 2741, "avg_range_0930_1000_pct": 0.42, "avg_range_1000_1030_pct": 0.37, "avg_range_quietest_1330_1400_pct": 0.23,
       "widest_range_in_0930_1000_share_pct": 32.1}
same(rec26, S26, rec26.keys())
note26 = (f"Reproduced from the one-minute bars on {th(n26)} whole sessions: the average of (highest high minus lowest low) as a percentage of the "
          f"part's first price is {a26[0]:.3f}% for 09:30-10:00, {a26[1]:.3f}% for 10:00-10:30 (second) and lowest for {lab(rk26[-1])} at "
          f"{a26[rk26[-1]]:.3f}%; 09:30-10:00 held the session's widest range on {k26} sessions ({pc(k26, n26)}%). All agree at the precision shown.")

# ------------------------------------------------------------------------------------------------ weekday item
v = V["SPY"].filter(pl.col("date") >= dt.date(2015, 1, 1)).with_columns(
    wd=pl.col("date").dt.weekday(), c2c=(pl.col("close") / pl.col("prev_close") - 1) * 100,
    o2c=(pl.col("close") / pl.col("s_open") - 1) * 100, up=pl.col("close") > pl.col("prev_close"))
g = {r["wd"]: r for r in v.group_by("wd").agg(n=pl.len(), c2c=pl.col("c2c").mean(), k_up=pl.col("up").sum(), o2c=pl.col("o2c").mean()).iter_rows(named=True)}
best = max(g, key=lambda k_: g[k_]["c2c"]); worst = min(g, key=lambda k_: g[k_]["c2c"])
assert best == 3 and worst == 5
rec25 = {"sessions": v.height, "wednesday_avg_pct": r3(g[3]["c2c"]), "wednesday_days": g[3]["n"], "friday_avg_pct": r3(g[5]["c2c"]),
         "friday_days": g[5]["n"], "all_days_avg_pct": r3(v["c2c"].mean()),
         "wednesday_closed_higher_pct": pc(g[3]["k_up"], g[3]["n"]), "friday_closed_higher_pct": pc(g[5]["k_up"], g[5]["n"]),
         "open_to_close_monday_pct": r3(g[1]["o2c"]), "open_to_close_tuesday_pct": r3(g[2]["o2c"]), "open_to_close_wednesday_pct": r3(g[3]["o2c"]),
         "open_to_close_thursday_pct": r3(g[4]["o2c"]), "open_to_close_friday_pct": r3(g[5]["o2c"]),
         "first_session": str(v["date"].min()), "last_session": str(v["date"].max())}
S25 = {"sessions": 2994, "wednesday_avg_pct": 0.096, "wednesday_days": 614, "friday_avg_pct": 0.002, "friday_days": 600, "all_days_avg_pct": 0.048,
       "wednesday_closed_higher_pct": 56.2, "friday_closed_higher_pct": 52.8,
       "open_to_close_monday_pct": 0.073, "open_to_close_tuesday_pct": -0.016, "open_to_close_wednesday_pct": 0.028,
       "open_to_close_thursday_pct": 0.003, "open_to_close_friday_pct": 0.004, "first_session": "2015-01-02", "last_session": "2026-11-27"}
same(rec25, S25, rec25.keys())
x = v["c2c"].to_numpy(); wdv = v["wd"].to_numpy()
rng = np.random.default_rng(7)
spread = lambda lbl: (lambda ms: ms.max() - ms.min())(np.array([x[lbl == k_].mean() for k_ in range(1, 6)]))
obs = spread(wdv)
pval = float(np.mean([spread(rng.permutation(wdv)) >= obs for _ in range(2000)]))
note25 = (f"Reproduced exactly from the daily closes: {th(v.height)} sessions from {v['date'].min()} to {v['date'].max()}; Wednesday "
          f"{g[3]['c2c']:+.3f}% ({g[3]['n']} days) is the highest weekday average and Friday {g[5]['c2c']:+.3f}% ({g[5]['n']} days) the lowest; "
          f"all days {v['c2c'].mean():+.3f}%; closed higher on {g[3]['k_up']} of {g[3]['n']} Wednesdays ({pc(g[3]['k_up'], g[3]['n'])}%) and "
          f"{g[5]['k_up']} of {g[5]['n']} Fridays ({pc(g[5]['k_up'], g[5]['n'])}%). Open to close (09:30 bar's open from 2016, the daily record's "
          f"open in 2015): Monday {g[1]['o2c']:+.3f}%, Tuesday {g[2]['o2c']:+.3f}%, Wednesday {g[3]['o2c']:+.3f}%, Thursday {g[4]['o2c']:+.3f}%, "
          f"Friday {g[5]['o2c']:+.3f}%. My own label-shuffling test on the gap between the highest and lowest weekday average gives p = {pval:.2f}, "
          f"so 'no weekday stands out' holds.")

# ------------------------------------------------------------------------------------------------ compare item
rec12 = {}
S12 = {"days_first_side": 289, "rate_first_side_pct": 0.0, "days_second_side": 289, "rate_second_side_pct": 0.0}
note12 = ("Cannot be checked: no reading is shown and the summary gives only the type 'compare' - no symbol, condition, outcome or period - so "
          "there is nothing to recalculate from the price files, and I could not tell which calculation produced 289 days. For the record, the "
          "headline carries what look like internal labels ('Changed review, start' and 'previous') and reports 0.0% on both sides, the same "
          "pattern as the '0 of N days' answers in this sample, which suggests the figures are not a real count; I cannot confirm that.")

# ------------------------------------------------------------------------------------------------ emit in sample order
emit("casey/ch-21#3", rec1, S1, True, note1)
emit("morgan/mh-18#1", rec2, S2, False, note2)
emit("riley/rh-22#6", rec3, S3, True, note3)
emit("alex/ah-08#1", rec4, S4, True, note4)
emit("alex/ah-18#1", rec5, S5, False, note5)
emit("casey/ch-21#1", rec6, S6, True, note6)
emit("casey/ch-18#1", rec7, S7, False, note7)
emit("riley/rh-01#1", rec8, S8, False, note8)
emit("casey/ch-21#2", rec9, S9, True, note9)
emit("alex/ah-15#1", rec10, S10, False, note10)
emit("sam/sh-02#1", rec11, S11, False, note11)
emit("morgan/mh-09#1", rec12, S12, "cannot_check", note12)
emit("casey/ch-03#1", rec13, S13, True, note13)
emit("morgan/mh-01#1", rec14, S14, True, note14)
emit("morgan/mh-23#2", rec15, S15, True, note15)
emit("alex/ah-24#1", rec16, S16, False, note16)
emit("casey/ch-07#1", rec17, S17, True, note17)
emit("riley/rh-04#1", rec18, S18, False, note18)
emit("jordan/jh-21#2", rec19, S19, False, note19)
emit("alex/ah-27#1", rec20, S20, False, note20)
emit("alex/ah-01#1", rec21, S21, True, note21)
emit("sam/sh-21#1", rec22, S22, True, note22)
emit("jordan/jh-21#6", rec23, S23, False, note23)
emit("jordan/jh-16#1", rec24, S24, False, note24)
emit("alex/ah-03#1", rec25, S25, True, note25)
emit("morgan/mh-12#1", rec26, S26, True, note26)

# sample order / completeness check against the sample file's keys
sample_keys = [json.loads(line)["key"] for line in open("/Users/prayugsigdel/Coding/strata/docs/v10_evidence/heldout_run/A/audit_sample.jsonl") if line.strip()]
out_keys = [json.loads(line)["key"] for line in LINES]
assert sample_keys == out_keys, (sample_keys, out_keys)

with open(f"{HERE}/audit_report.generated.jsonl", "w") as f:
    f.write("\n".join(LINES) + "\n")
for line in LINES:
    print(line)
tally = {}
for line in LINES:
    a = json.loads(line)["agrees"]
    tally[str(a)] = tally.get(str(a), 0) + 1
print("TALLY", len(LINES), tally, file=sys.stderr)
