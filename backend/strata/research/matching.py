"""Matched historical comparisons.

A raw comparison ("after large down days SPY rose more often than on other days") mixes two
things: the condition itself, and everything that tends to come with it. Large declines
arrive in volatile markets, and volatile markets have wider ranges and different drift
whether or not anything fell. This module asks the narrower question: compared with
sessions that looked the SAME on a list of observable variables but where the condition
did not hold, is the outcome still different?

Three estimates are computed from the same observations and shown together

  * stratification    the controls are cut into quantile bins; each qualifying session is
                      compared with the non-qualifying sessions of its own stratum
  * nearest neighbour each qualifying session is paired with its k most similar
                      non-qualifying sessions (standardised distance, with a caliper)
  * regression        outcome ~ condition + controls, with calendar-week clustered errors

What it does not do
  * it is not a causal estimate. Matching removes the part of the difference explained by
    the LISTED variables at the resolution they were measured; anything unlisted, and any
    variable that is itself moved by the condition, is still in the estimate
  * controls must be known at the decision tick. The caller passes values read at that
    tick; a variable that defines the condition is refused rather than matched on
  * qualifying sessions with no comparable session are reported as unmatched, never forced
"""
from __future__ import annotations

import math

import numpy as np

from strata.research import robust as R

MAX_BINS, MIN_MATCHED = 3, 20


def _smd(xt: np.ndarray, xc: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """Standardised mean difference of one control between the two groups."""
    if xt.size < 2 or xc.size < 2:
        return None
    mc = float(np.average(xc, weights=w)) if w is not None else float(xc.mean())
    sd = math.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
    return None if sd <= 0 else float((xt.mean() - mc) / sd)


def _mean_test(d: np.ndarray, dates: np.ndarray, reps: int) -> dict:
    """Mean of per-observation differences with a calendar-week interval and p-value."""
    out = {"estimate": float(d.mean()) if d.size else None, "ci95": [None, None], "p_value": None, "clusters": 0}
    if d.size < 6:
        return out
    r = R.cluster_mean_ci(d, dates, "week", reps)
    out["clusters"] = r.get("clusters", 0)
    if r["ci"][0] is None or not r.get("se"):
        return out
    from scipy.stats import t as tdist
    out["ci95"] = r["ci"]
    out["p_value"] = float(2 * (1 - tdist.cdf(abs(d.mean()) / r["se"], max(2, out["clusters"] - 1))))
    return out


def _strata(Z: np.ndarray, bins: int) -> np.ndarray:
    """A stratum id per row from quantile bins of every control (computed on the pooled sample)."""
    key = np.zeros(Z.shape[0], np.int64)
    for j in range(Z.shape[1]):
        col = Z[:, j]
        uniq = np.unique(col)
        if uniq.size <= bins:  # a discrete control (weekday, a regime flag): its own values are the bins
            b = np.searchsorted(uniq, col)
            width = uniq.size
        else:
            edges = np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1])
            b = np.searchsorted(edges, col, side="right")
            width = bins
        key = key * max(width, 1) + b
    return key


def _regression(y: np.ndarray, treated: np.ndarray, Z: np.ndarray, dates: np.ndarray) -> dict:
    """Coefficient on the condition in y ~ 1 + condition + controls, week-clustered errors."""
    out = {"estimate": None, "ci95": [None, None], "p_value": None}
    n = y.size
    X = np.column_stack([np.ones(n), treated.astype(float), Z])
    if n <= X.shape[1] + 10:
        return out
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return out
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    cl = R.cluster_ids(dates, "week")
    uniq, inv = np.unique(cl, return_inverse=True)
    G = uniq.size
    if G < 12:
        return out
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, inv, X * resid[:, None])
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / max(1, n - X.shape[1]))
    se = math.sqrt(max(V[1, 1], 0.0))
    if se <= 0:
        return out
    from scipy.stats import t as tdist
    q = float(tdist.ppf(0.975, G - 1))
    out.update(estimate=float(beta[1]), ci95=[float(beta[1] - q * se), float(beta[1] + q * se)], p_value=float(2 * (1 - tdist.cdf(abs(beta[1]) / se, G - 1))), clusters=int(G))
    return out


def matched_comparison(dates, y, treated, pool, controls: list[dict], k: int = 5, caliper: float = 0.5, bins: int = MAX_BINS, reps: int = 400, binary: bool = True) -> dict:
    """Compare `y` on the treated sessions with comparable sessions from `pool`.

    dates     ISO date per session            y        outcome per session (0/1 for a frequency, else a return)
    treated   sessions where the condition held (resolved outcomes only)
    pool      evaluated sessions where it did not hold
    controls  [{"name", "label", "values": array, "unit"}] read at the decision tick
    """
    dates = np.asarray(dates)
    names = [c["name"] for c in controls]
    labels = {c["name"]: c.get("label") or c["name"] for c in controls}
    X = np.column_stack([np.asarray(c["values"], float) for c in controls]) if controls else np.zeros((len(dates), 0))
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    ti, ci = np.flatnonzero(treated & fin), np.flatnonzero(pool & fin)
    out = {"controls": [{"name": n, "label": labels[n]} for n in names], "treated": int(treated.sum()), "treated_usable": int(ti.size), "pool": int(pool.sum()), "pool_usable": int(ci.size),
           "dropped_missing_controls": int((treated & ~fin).sum()), "ordinary": None, "stratified": None, "nearest": None, "regression": None, "balance": [], "verdict": "not_estimable",
           "sentence": "", "notes": [], "k": k, "caliper": caliper, "metric": "frequency" if binary else "mean return"}
    if not controls:
        out["notes"].append("No control variables were given, so there is nothing to match on.")
        return out
    if ti.size < MIN_MATCHED or ci.size < MIN_MATCHED:
        out["notes"].append(f"Only {ti.size} qualifying and {ci.size} other sessions have every control available: too few to match (at least {MIN_MATCHED} of each are needed).")
        return out
    yt, yc = y[ti], y[ci]
    ordinary = _mean_test(np.r_[yt - yc.mean()], dates[ti], reps)  # the plain difference, on exactly the sessions the matched estimates use
    ordinary.update(treated_value=float(yt.mean()), comparison_value=float(yc.mean()), n=int(ti.size), comparison_n=int(ci.size))
    out["ordinary"] = ordinary
    # standardise on the pooled sample
    allx = X[np.r_[ti, ci]]
    mu, sd = allx.mean(axis=0), allx.std(axis=0, ddof=1)
    keep = sd > 0
    if not keep.all():
        out["notes"].append("Controls with no variation were ignored: " + ", ".join(labels[n] for n, k_ in zip(names, keep) if not k_) + ".")
    Zt, Zc = (X[ti] - mu)[:, keep] / sd[keep], (X[ci] - mu)[:, keep] / sd[keep]
    used = [n for n, k_ in zip(names, keep) if k_]
    if not used:
        return out
    # ----- stratification (coarsened exact matching, weighted to the qualifying sessions)
    nb = max(2, min(bins, int(round((ti.size / 8) ** (1 / len(used)))) or 2))
    key = _strata(np.vstack([Zt, Zc]), nb)
    kt, kc = key[:ti.size], key[ti.size:]
    diffs, d_dates, w_c = [], [], np.zeros(ci.size)
    unmatched = 0
    for s in np.unique(kt):
        a, b = kt == s, kc == s
        if not b.any():
            unmatched += int(a.sum())
            continue
        diffs.append(yt[a] - yc[b].mean())
        d_dates.append(dates[ti][a])
        w_c[b] += a.sum() / b.sum()
    if diffs:
        d = np.concatenate(diffs)
        st = _mean_test(d, np.concatenate(d_dates), reps)
        st.update(matched=int(d.size), unmatched=unmatched, strata=int(np.unique(kt).size), bins_per_control=nb,
                  treated_value=float(yt[np.isin(kt, np.unique(kc))].mean()) if d.size else None)
        st["comparison_value"] = None if st["treated_value"] is None or st["estimate"] is None else st["treated_value"] - st["estimate"]
        out["stratified"] = st
    # ----- nearest neighbours with replacement, inside a caliper
    kk = min(k, ci.size)
    cal = caliper * math.sqrt(len(used))
    nn_d, nn_dates, w_n = [], [], np.zeros(ci.size)
    far = 0
    for a0 in range(0, ti.size, 512):  # chunked so a long history cannot build a huge distance matrix
        D = np.sqrt(((Zt[a0:a0 + 512, None, :] - Zc[None, :, :]) ** 2).sum(axis=2))
        idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        dist = np.take_along_axis(D, idx, axis=1)
        for r in range(idx.shape[0]):
            ok = dist[r] <= cal
            if not ok.any():
                far += 1
                continue
            j = idx[r][ok]
            nn_d.append(yt[a0 + r] - yc[j].mean())
            nn_dates.append(dates[ti][a0 + r])
            w_n[j] += 1.0 / j.size
    if nn_d:
        d = np.asarray(nn_d, float)
        nn = _mean_test(d, np.asarray(nn_dates), reps)
        nn.update(matched=int(d.size), unmatched=far, comparison_sessions_used=int((w_n > 0).sum()))
        nn["treated_value"] = float(d.mean() + np.average(yc, weights=w_n)) if w_n.sum() > 0 else None
        nn["comparison_value"] = float(np.average(yc, weights=w_n)) if w_n.sum() > 0 else None
        out["nearest"] = nn
    # ----- regression adjustment as a cross-check
    out["regression"] = _regression(np.r_[yt, yc], np.r_[np.ones(ti.size, bool), np.zeros(ci.size, bool)], np.vstack([Zt, Zc]), np.r_[dates[ti], dates[ci]])
    # ----- balance: how different the groups were on each control, before and after
    j = 0
    for n_, k_ in zip(names, keep):
        if not k_:
            continue
        xt, xc = X[ti][:, names.index(n_)], X[ci][:, names.index(n_)]
        out["balance"].append({"name": n_, "label": labels[n_], "treated_mean": float(xt.mean()), "comparison_mean": float(xc.mean()), "before": _smd(xt, xc),
                               "after_stratified": _smd(xt, xc, w_c) if w_c.sum() > 0 else None, "after_nearest": _smd(xt, xc, w_n) if w_n.sum() > 0 else None,
                               "matched_mean": float(np.average(xc, weights=w_n)) if w_n.sum() > 0 else None})
        j += 1
    out.update(_verdict(out, binary))
    return out


def _verdict(o: dict, binary: bool) -> dict:
    """Does the difference survive? Stated from the nearest-neighbour estimate, with the
    stratified and regression estimates as agreement checks."""
    notes = list(o["notes"])
    raw, nn, st, rg = o["ordinary"], o.get("nearest"), o.get("stratified"), o.get("regression")
    unit = " points" if binary else "%"
    scale = 100.0 if binary else 1.0
    main = nn if nn and nn.get("matched", 0) >= MIN_MATCHED else st if st and st.get("matched", 0) >= MIN_MATCHED else None
    if main is None or main.get("estimate") is None or raw.get("estimate") is None:
        notes.append("Too few qualifying sessions have a comparable session on these controls: the matched comparison cannot be estimated.")
        return {"verdict": "not_estimable", "sentence": "The matched comparison could not be estimated: too few qualifying sessions have a comparable session on these controls.", "notes": notes}
    e, r = main["estimate"], raw["estimate"]
    lo, hi = main["ci95"]
    excl = lo is not None and (lo > 0 or hi < 0)
    same = (e > 0) == (r > 0)
    share = None if abs(r) < 1e-12 else 1 - e / r
    # the three estimates answer the same question in different ways; one of them alone is not allowed to carry the conclusion
    alt = [x for x in (nn, st, rg) if x and x is not main and x.get("estimate") is not None and x.get("ci95", [None])[0] is not None]
    agree = [x for x in alt if (x["estimate"] > 0) == (e > 0) and (x["ci95"][0] > 0 or x["ci95"][1] < 0)]
    if excl and same and alt and not agree:
        names = {id(nn): "nearest-neighbour", id(st): "stratified", id(rg): "regression"}
        verdict = "mixed"
        s = (f"The plain difference of {r * scale:+.1f}{unit} becomes {e * scale:+.1f}{unit} by the {names.get(id(main), 'main')} estimate (95% interval {lo * scale:+.1f} to {hi * scale:+.1f}), but the "
             + " and ".join(names.get(id(x), "other") for x in alt) + f" estimate{'s do' if len(alt) != 1 else ' does'} not distinguish it from zero ("
             + ", ".join(f"{x['estimate'] * scale:+.1f}{unit}" for x in alt) + "). The methods do not agree, so whether the difference survives the controls is unsettled.")
        notes.append("This is an association measured on historical sessions, not a causal effect: only the listed variables are held comparable.")
        return {"verdict": verdict, "sentence": s, "share_explained": share, "notes": notes}
    if excl and same:
        verdict = "persists"
        tail = "persists after matching"
    elif same:
        verdict = "not_distinguishable"
        tail = "is no longer distinguishable from zero after matching"
    else:
        verdict = "reverses"
        tail = "changes sign after matching"
    by = ", ".join(c["label"] for c in o["controls"])
    s = (f"The plain difference of {r * scale:+.1f}{unit} becomes {e * scale:+.1f}{unit} against sessions matched on {by}"
         + (f" (95% interval {lo * scale:+.1f} to {hi * scale:+.1f})" if lo is not None else "") + f": the difference {tail}.")
    if share is not None and same and 0 < share < 1.5:
        s += f" About {share * 100:.0f}% of the plain difference is accounted for by the controls."
    others = [x["estimate"] for x in (st, rg) if x and x is not main and x.get("estimate") is not None]
    if others and any((x > 0) != (e > 0) for x in others):
        notes.append("The stratified and regression estimates do not all agree on the sign: the conclusion depends on the method, so treat it as unsettled.")
    worst = max((abs(b["after_nearest"]) for b in o["balance"] if b.get("after_nearest") is not None), default=None)
    if worst is not None and worst > 0.25:
        notes.append(f"After matching, the groups still differ by up to {worst:.2f} standard deviations on a control: the match is imperfect.")
    um = (nn or {}).get("unmatched", 0)
    if um:
        notes.append(f"{um} qualifying session{'s' if um != 1 else ''} had no comparable session within the caliper and {'are' if um != 1 else 'is'} not part of the matched estimate.")
    notes.append("This is an association measured on historical sessions, not a causal effect: only the listed variables are held comparable.")
    return {"verdict": verdict, "sentence": s, "share_explained": share, "notes": notes}


DEFAULT_CONTROLS = {"volatility": ["rv21", "vix_prev"], "vix": ["vix_prev"], "realized volatility": ["rv21"], "gap": ["gap"], "overnight gap": ["gap"], "trend": ["sma50_dist", "sma200_dist"],
                    "drawdown": ["dist_high252"], "day of week": ["weekday"], "weekday": ["weekday"], "recent returns": ["ret_5", "ret_20"], "regime": ["rv_pctile", "sma200_dist"],
                    "volume": ["vol_ratio"], "range": ["range_1"]}
