"""One definition of "the condition against everything else".

Every research surface asks the same thing of a set of sessions: on the sessions where a
condition held, how often did the outcome happen, how does that compare with the other
evaluated sessions, and how sure can anyone be. Hypothesis plans, sensitivity maps, the
market-behaviour studies, research graphs, matched comparisons and the pattern explorer all
call `compare` so a frequency, its interval and its p-value cannot differ between screens.

Conventions (unchanged from the modules this replaced)
  * AMBIGUOUS observations (target and stop inside one bar) are excluded from frequencies
    and reported as a count; they still contribute their end-of-horizon return
  * `baseline_probability` is the frequency over ALL evaluated sessions; the test compares
    the qualifying sessions with the OTHER evaluated sessions
  * intervals and p-values resample whole calendar weeks, because observations in the same
    week are not independent draws. The independent-sample p-value is reported next to it
"""
from __future__ import annotations

import numpy as np

from strata.research import robust as R

T_, A_ = 0, 3  # outcome codes shared with features.py (target, ambiguous)


def compare(dates, code, end_ret, qual, base, direction: str = "up", reps: int = 400, robust: bool = True,
            mean_diff: bool = False, return_ci: bool = False, mae=None, mfe=None, by: str = "week") -> dict:
    """Evidence for one condition inside one period.

    dates   ISO dates per session            code     outcome code per session (-1 = not evaluated)
    end_ret end-of-horizon return, %         qual     sessions where the condition held (already inside the period)
    base    every evaluated session of the period (the baseline); qual must be a subset
    """
    dates = np.asarray(dates)
    res_q, res_b = qual & (code != A_), base & (code != A_)
    rest_all = base & ~qual
    rest = res_b & ~qual
    n, k = int(res_q.sum()), int((code[res_q] == T_).sum())
    bn, bk = int(res_b.sum()), int((code[res_b] == T_).sum())
    rn, rk = int(rest.sum()), int((code[rest] == T_).sum())
    er = end_ret[qual]
    er = er[np.isfinite(er)]
    rr = end_ret[rest_all]
    rr = rr[np.isfinite(rr)]
    sgn = -1.0 if direction == "down" else 1.0
    out = {"n": n, "hits": k, "probability": (k / n) if n else None, "ambiguous": int((qual & (code == A_)).sum()), "sessions": int(qual.sum()), "evaluated": int(base.sum()),
           "baseline_n": bn, "baseline_probability": (bk / bn) if bn else None, "rest_n": rn, "rest_probability": (rk / rn) if rn else None,
           "lift": (k / n - bk / bn) if n and bn else None, "lift_vs_rest": (k / n - rk / rn) if n and rn else None,
           "ci95": R.wilson(k, n), "p_value": None, "p_iid": R.two_prop_p(k, n, rk, rn), "clusters": 0, "design_effect": None, "lift_ci95": [None, None],
           "avg_end_return_pct": float(er.mean()) if er.size else None, "median_end_return_pct": float(np.median(er)) if er.size else None,
           "rest_avg_end_return_pct": float(rr.mean()) if rr.size else None, "end_return_diff_pct": None, "directional_diff_pct": None, "end_return_p_value": None,
           "return_ci95": [None, None], "q05_end_return_pct": float(np.quantile(er, 0.05)) if er.size >= 20 else None, "q95_end_return_pct": float(np.quantile(er, 0.95)) if er.size >= 20 else None,
           "avg_mae_pct": float(np.nanmean(mae[qual])) if mae is not None and qual.any() else None, "avg_mfe_pct": float(np.nanmean(mfe[qual])) if mfe is not None and qual.any() else None,
           "first": None, "last": None}
    if qual.any():
        qd = dates[qual]
        out["first"], out["last"] = str(qd[0]), str(qd[-1])
    if er.size and rr.size:
        out["end_return_diff_pct"] = out["avg_end_return_pct"] - out["rest_avg_end_return_pct"]
        out["directional_diff_pct"] = sgn * out["end_return_diff_pct"]
    if not n or not rn:
        return out
    hq, hr = (code[res_q] == T_).astype(float), (code[rest] == T_).astype(float)
    if robust:
        lt = R.lift_test(hq, dates[res_q], hr, dates[rest], by, reps)
        ci = R.cluster_prop_ci(hq, dates[res_q], by, reps)
        out.update(p_value=lt["p_value"], p_iid=lt["p_iid"], clusters=lt["cond_clusters"], design_effect=ci["design_effect"], ci95=ci["ci"], lift_ci95=lt["ci"])
        if mean_diff and direction != "either":
            out["end_return_p_value"] = R.mean_diff_test(end_ret[qual], dates[qual], end_ret[rest_all], dates[rest_all], by, reps)["p_value"]
        if return_ci and er.size >= 6:
            out["return_ci95"] = R.cluster_mean_ci(end_ret[qual], dates[qual], by, reps)["ci"]
    else:
        out["clusters"] = int(np.unique(R.cluster_ids(dates[res_q], by)).size)
    return out


def of(ft, ev: dict, pm, direction: str = "up", **kw) -> dict:
    """`compare` for a SessionFeatures evaluation (`ft.evaluate`) restricted to the period mask `pm`."""
    out = ev["outcome"]
    return compare(ft.dates, out.code, out.end_ret, ev["qualifying"] & pm, ev["evaluable"] & pm, direction, mae=out.mae, mfe=out.mfe, **kw)
