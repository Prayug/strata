"""Deterministic statistics for research conclusions.

Everything a report says about significance, robustness or out-of-sample behaviour is
computed here from stored observations (qualifying sessions, baseline sessions, trades).
No language model is involved, and nothing here ranks by profitability alone.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

CODE = {"t": "target", "s": "stop", "n": "neither", "a": "ambiguous"}


from strata.research.robust import two_prop_p, wilson  # noqa: E402,F401  (one definition; re-exported for callers)


def bh_adjust(pvals: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control across a family of tests)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(pvals)
    if not m:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        out[i] = float(min(1.0, prev))
    return out


def bonferroni(p: float | None, m: int) -> float | None:
    return None if p is None else float(min(1.0, p * max(1, m)))


def mean_test(x) -> dict:
    """Mean of a sample with a bootstrap 95% interval and a one-sample t-test against zero."""
    x = np.asarray([v for v in x if v is not None and math.isfinite(v)], float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": [None, None], "p_value": None, "median": None}
    out = {"n": n, "mean": float(x.mean()), "median": float(np.median(x)), "ci95": [None, None], "p_value": None, "std": float(x.std(ddof=1)) if n > 1 else None}
    if n >= 3:
        rng = np.random.default_rng(7)
        m = x[rng.integers(0, n, (4000, n))].mean(axis=1)
        out["ci95"] = [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
        sd = x.std(ddof=1)
        if sd > 0:
            from scipy.stats import t as tdist
            out["p_value"] = float(2 * (1 - tdist.cdf(abs(x.mean() / (sd / math.sqrt(n))), n - 1)))
    return out


def diff_test(a, b) -> dict:
    """Difference in means of two groups (Welch t-test), e.g. Monday trades versus the rest."""
    a = np.asarray([v for v in a if v is not None and math.isfinite(v)], float)
    b = np.asarray([v for v in b if v is not None and math.isfinite(v)], float)
    out = {"n_a": int(a.size), "n_b": int(b.size), "mean_a": float(a.mean()) if a.size else None, "mean_b": float(b.mean()) if b.size else None, "p_value": None}
    if a.size >= 3 and b.size >= 3 and (a.std(ddof=1) > 0 or b.std(ddof=1) > 0):
        from scipy.stats import ttest_ind
        out["p_value"] = float(ttest_ind(a, b, equal_var=False).pvalue)
    return out


# ----------------------------------------------------------------------------- studies
def study_block(res: dict, start: str | None = None, end: str | None = None) -> dict:
    """Conditional probability, baseline and their difference on the sessions inside [start, end]."""
    rows = [r for r in res.get("sessions", []) if (start is None or r["date"] >= start) and (end is None or r["date"] <= end)]
    base = res.get("baseline_sessions") or {"dates": [], "outcomes": ""}
    pairs = [(d, o) for d, o in zip(base["dates"], base["outcomes"]) if (start is None or d >= start) and (end is None or d <= end)]
    n = sum(1 for r in rows if r["outcome"] != "ambiguous")
    k = sum(1 for r in rows if r["outcome"] == "target")
    bn = sum(1 for _, o in pairs if o != "a")
    bk = sum(1 for _, o in pairs if o == "t")
    signal = (res.get("config") or {}).get("entry_time") == "signal" or bool((res.get("config") or {}).get("sequence"))
    # condition sessions against all other evaluated sessions (for a signal-timed study the baseline is measured
    # from the open on every session, so the two samples are compared as they are)
    p_iid = two_prop_p(k, n, bk, bn) if signal else two_prop_p(k, n, bk - k, bn - n) if bn - n > 0 else None
    pval, clusters, deff = p_iid, None, None
    if n >= 10 and pairs:
        # Observations in the same week are not independent draws (large moves cluster), so the test resamples whole
        # calendar weeks. The larger of the two p-values is used: clustering may weaken the evidence, never strengthen it.
        from strata.research import robust as R
        qd = [r["date"] for r in rows if r["outcome"] != "ambiguous"]
        qset = set(qd)
        rest = [(d, o) for d, o in pairs if o != "a" and (signal or d not in qset)]
        lt = R.lift_test([1.0 if r["outcome"] == "target" else 0.0 for r in rows if r["outcome"] != "ambiguous"], qd, [1.0 if o == "t" else 0.0 for _, o in rest], [d for d, _ in rest], "week", 400)
        clusters, deff = lt["cond_clusters"], lt.get("design_effect")
        if lt["p_value"] is not None and p_iid is not None:
            pval = max(p_iid, lt["p_value"])
    ends = [r["end_return_pct"] for r in rows]
    # direction check: do qualifying sessions END higher than the others, or do they merely have wider ranges?
    b_end = base.get("end_returns") or []
    cond_dates = {r["date"] for r in rows}
    rest_end = [e for (d, _), e in zip(zip(base["dates"], base["outcomes"]), b_end) if (start is None or d >= start) and (end is None or d <= end) and d not in cond_dates] if b_end else []
    dt = diff_test(ends, rest_end) if rest_end else {"p_value": None, "mean_b": None}
    return {"start": start, "end": end, "n": n, "hits": k, "probability": k / n if n else None, "ci95": wilson(k, n), "baseline_n": bn, "baseline_probability": bk / bn if bn else None,
            "lift": (k / n - bk / bn) if n and bn else None, "p_value": pval, "p_independent": p_iid, "weeks": clusters, "design_effect": deff, "avg_end_return_pct": float(np.mean(ends)) if ends else None, "sessions": len(rows),
            "other_sessions_avg_end_return_pct": dt["mean_b"], "end_return_diff_pct": (float(np.mean(ends)) - dt["mean_b"]) if ends and dt["mean_b"] is not None else None, "end_return_p_value": dt["p_value"]}


def split_date(dates: list[str], frac: float) -> str | None:
    ds = sorted(dates)
    return ds[min(len(ds) - 1, int(len(ds) * frac))] if ds else None


def study_robustness(res: dict, trials: int = 1) -> dict:
    """Is the study's conditional probability distinguishable from its baseline, and does it
    hold up in later data and across years? `trials` is the number of configurations tested
    in the research session (the more that were tried, the stronger the evidence must be)."""
    st = res["stats"]
    full = study_block(res)
    base = (res.get("baseline_sessions") or {}).get("dates") or []
    cut = split_date(base, 0.7)
    ins = study_block(res, None, _day_before(cut)) if cut else None
    oos = study_block(res, cut, None) if cut else None
    years = [y for y in res.get("by_year", []) if y["n"] >= 5 and y.get("probability") is not None]
    pb = st.get("baseline_probability")
    above = sum(1 for y in years if pb is not None and y["probability"] > pb)
    adj = bonferroni(full["p_value"], trials)
    checks = []
    n = full["n"]
    wk = full.get("weeks")
    checks.append({"name": "Sample size", "ok": n >= 30 and (wk is None or wk >= 12), "detail": f"{n} qualifying sessions" + (f" in {wk} separate weeks" if wk else "")
                   + ("" if n >= 30 else " - fewer than 30, so any estimate is fragile") + (" - too few separate weeks to treat them as independent observations" if wk is not None and wk < 12 and n >= 30 else "")})
    if full["p_value"] is not None:
        checks.append({"name": "Different from the baseline", "ok": adj is not None and adj < 0.05,
                       "detail": f"p = {full['p_value']:.4f} against all other sessions, with whole weeks resampled" + (f" (treating sessions as independent would give {full['p_independent']:.4f})" if full.get("p_independent") is not None and abs(full["p_independent"] - full["p_value"]) > 1e-4 else "") + (f"; {adj:.4f} after allowing for the {trials} configurations tested in this session (Bonferroni)" if trials > 1 else "")})
    if oos and ins and oos["n"] and ins["n"] and oos["lift"] is not None and ins["lift"] is not None:
        same = (oos["lift"] > 0) == (ins["lift"] > 0)
        checks.append({"name": "Holds in later data", "ok": same and oos["n"] >= 10,
                       "detail": f"first 70% of history: {ins['probability'] * 100:.1f}% of {ins['n']} vs baseline {ins['baseline_probability'] * 100:.1f}%; last 30% (from {cut}): {oos['probability'] * 100:.1f}% of {oos['n']} vs baseline {oos['baseline_probability'] * 100:.1f}%"
                                 + ("" if oos["n"] >= 10 else " - too few later sessions to judge")})
    if years:
        checks.append({"name": "Consistent across years", "ok": above / len(years) >= 0.6, "detail": f"above the overall baseline in {above} of {len(years)} years with at least 5 qualifying sessions"})
    if full.get("end_return_diff_pct") is not None and (res.get("config") or {}).get("target") and (res.get("config") or {}).get("measure", "touch") == "touch":
        up = ((res["config"]["target"].get("value") or 0) >= 0) if res["config"]["target"].get("kind") != "expr" else True
        d_ = full["end_return_diff_pct"] * (1 if up else -1)
        ok_dir = d_ > 0 and (full["end_return_p_value"] is not None and full["end_return_p_value"] < 0.05)
        checks.append({"name": "Direction, not just a wider range", "ok": ok_dir,
                       "detail": f"qualifying sessions end {full['avg_end_return_pct']:+.2f}% from the entry on average versus {full['other_sessions_avg_end_return_pct']:+.2f}% on other sessions"
                                 + (f" (p = {full['end_return_p_value']:.3f})" if full["end_return_p_value"] is not None else "")
                                 + ("" if ok_dir else ". A target is touched more often when ranges are wider, so a higher touch rate alone does not show a directional edge")})
    if st.get("ambiguous"):
        checks.append({"name": "Order of target and stop known", "ok": st["ambiguous"] / max(st["qualifying_sessions"], 1) < 0.1, "detail": f"{st['ambiguous']} of {st['qualifying_sessions']} observations are ambiguous (both levels inside one bar)"})
    if n < 30:
        verdict, level = "Too few observations to draw a conclusion.", "insufficient"
    elif adj is None or adj >= 0.05:
        verdict, level = "Not distinguishable from the baseline: the difference is within what chance alone produces.", "none"
    elif all(c["ok"] for c in checks):
        verdict, level = "Statistically different from the baseline, consistent across years and in later data. This is evidence about the underlying's behaviour, not a profitability claim.", "supported"
    else:
        verdict, level = "Statistically different from the baseline over the whole period, but not consistently - see the checks that did not pass.", "mixed"
    return {"kind": "study", "trials": trials, "full": full, "in_sample": ins, "out_of_sample": oos, "split_date": cut, "adjusted_p_value": adj, "years_above_baseline": [above, len(years)],
            "checks": checks, "verdict": verdict, "level": level}


def _day_before(iso: str) -> str:
    from datetime import timedelta
    return (date.fromisoformat(iso) - timedelta(days=1)).isoformat()


# ----------------------------------------------------------------------------- backtests
def backtest_robustness(result: dict, trades: list[dict], trials: int = 1) -> dict:
    """Is the average trade distinguishable from zero, and does that survive costs, later
    data and the removal of the few best trades?"""
    pnl = [t["pnl"] for t in trades]
    full = mean_test(pnl)
    adj = bonferroni(full["p_value"], trials)
    ds = sorted(t["entry_date"] for t in trades)
    cut = split_date(ds, 0.7)
    a = [t["pnl"] for t in trades if cut and t["entry_date"] < cut]
    b = [t["pnl"] for t in trades if cut and t["entry_date"] >= cut]
    ins, oos = mean_test(a), mean_test(b)
    total = sum(pnl)
    top = sorted(pnl, reverse=True)[:3]
    cs = result.get("cost_sensitivity") or {}
    checks = []
    n = full["n"]
    checks.append({"name": "Sample size", "ok": n >= 30, "detail": f"{n} trades" + ("" if n >= 30 else " - fewer than 30, so the average is fragile")})
    if full["p_value"] is not None:
        checks.append({"name": "Average trade different from zero", "ok": adj is not None and adj < 0.05 and full["mean"] > 0,
                       "detail": f"mean {full['mean']:+,.2f} per trade, 95% interval {full['ci95'][0]:+,.2f} to {full['ci95'][1]:+,.2f}, p = {full['p_value']:.4f}" + (f" ({adj:.4f} after allowing for {trials} configurations tested)" if trials > 1 else "")})
    if ins["n"] >= 3 and oos["n"] >= 3:
        same = (ins["mean"] > 0) == (oos["mean"] > 0)
        checks.append({"name": "Holds in later data", "ok": same and oos["mean"] > 0 and oos["n"] >= 10,
                       "detail": f"first 70% of trades: mean {ins['mean']:+,.2f} ({ins['n']}); last 30% (from {cut}): mean {oos['mean']:+,.2f} ({oos['n']})" + ("" if oos["n"] >= 10 else " - too few later trades to judge")})
    if n >= 5 and total > 0:
        share = sum(top) / total
        checks.append({"name": "Not carried by a few trades", "ok": share < 0.6, "detail": f"the three best trades account for {share * 100:.0f}% of the total profit"})
    be = cs.get("breakeven_slippage_pct")
    if cs.get("rows"):
        worse = next((r for r in cs["rows"] if r["extra_slippage_pct"] == 2), None)
        checks.append({"name": "Survives worse fills", "ok": bool(worse and worse["net_pnl"] > 0),
                       "detail": (f"with every fill 2% worse the net result is {worse['net_pnl']:+,.2f}" if worse else "") + (f"; it reaches zero at about {be:.1f}% worse fills" if be is not None else "")})
    if n < 30:
        verdict, level = "Too few trades to draw a conclusion about an edge.", "insufficient"
    elif adj is None or adj >= 0.05 or full["mean"] <= 0:
        verdict, level = "No statistically significant edge: the average trade is not distinguishable from zero.", "none"
    elif all(c["ok"] for c in checks):
        verdict, level = "The average trade is significantly above zero, and that holds in later data, with worse fills and without the best trades. Past results on this data only - not a forecast.", "supported"
    else:
        verdict, level = "The average trade is significantly above zero over the whole period, but not robustly - see the checks that did not pass.", "mixed"
    return {"kind": "backtest", "trials": trials, "full": full, "in_sample": ins, "out_of_sample": oos, "split_date": cut, "adjusted_p_value": adj, "checks": checks, "verdict": verdict, "level": level}


def weekday_breakdown(rows: list[dict], value: str, weekday_of) -> dict:
    """Per-weekday summary of `value` plus a test of each day against all other days."""
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    groups: dict[int, list[float]] = {i: [] for i in range(5)}
    for r in rows:
        v = r.get(value)
        if v is not None:
            groups.setdefault(weekday_of(r), []).append(float(v))
    out = []
    for i in range(5):
        rest = [v for j in range(5) if j != i for v in groups[j]]
        t = diff_test(groups[i], rest)
        out.append({"day": names[i], "n": len(groups[i]), "mean": t["mean_a"], "rest_mean": t["mean_b"], "p_value": t["p_value"],
                    "positive_share": (sum(1 for v in groups[i] if v > 0) / len(groups[i])) if groups[i] else None})
    ps = bh_adjust([x["p_value"] for x in out])
    for x, q in zip(out, ps):
        x["adjusted_p_value"] = q
    return {"rows": out, "note": "Each day is tested against all other days (Welch t-test); p-values are adjusted for the five comparisons (Benjamini-Hochberg)."}
