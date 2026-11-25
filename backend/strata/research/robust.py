"""Statistics that stay honest when observations are not independent.

Market observations cluster: a 3% decline is usually followed by more large moves, holding
periods overlap, and a week with one qualifying session often has several. Formulas that
assume independent draws (Wilson intervals, the two-proportion z-test, an ordinary
bootstrap) then report intervals that are too narrow and p-values that are too small.

Everything here is deterministic (fixed seeds) and works on stored observations only:

  * moving-block and stationary bootstraps for time-ordered series
  * calendar-cluster bootstraps (session / week / month) for event samples
  * a circular-shift permutation test for "condition versus baseline" claims
  * Newey-West (HAC) standard errors and an effective sample size
  * paired comparisons on identical sessions
  * multiple-hypothesis corrections (Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli)
  * purged, embargoed chronological cross-validation and walk-forward splits
  * uniqueness weights for overlapping holding periods
  * regime breakdowns with a heterogeneity test
  * an evidence grade that refuses strong language on thin samples
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

REPS = 2000
CHUNK = 3_000_000  # resampled elements held in memory at once
MIN_OBS, MIN_CLUSTERS = 30, 12  # below either, no statistical conclusion is drawn


# ----------------------------------------------------------------------------- basics
def _arr(x) -> np.ndarray:
    a = np.asarray([v for v in x if v is not None], float) if not isinstance(x, np.ndarray) else x.astype(float)
    return a[np.isfinite(a)]


def acf(x, lag: int = 1) -> float:
    x = _arr(x)
    if x.size <= lag + 1:
        return 0.0
    d = x - x.mean()
    den = float((d * d).sum())
    return float((d[lag:] * d[:-lag]).sum() / den) if den > 0 else 0.0


def nw_lags(n: int) -> int:
    """Newey-West rule-of-thumb truncation lag."""
    return int(max(1, math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))) if n > 2 else 0


def hac_se(x, lags: int | None = None) -> float | None:
    """Newey-West standard error of the mean (Bartlett kernel)."""
    x = _arr(x)
    n = x.size
    if n < 3:
        return None
    L = min(nw_lags(n) if lags is None else int(lags), n - 1)
    d = x - x.mean()
    s = float((d * d).sum() / n)
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * float((d[k:] * d[:-k]).sum() / n)
    return math.sqrt(max(s, 0.0) / n)


def effective_n(x, lags: int | None = None) -> float:
    """Sample size an independent sample would need to be as informative about the mean."""
    x = _arr(x)
    n = x.size
    if n < 3 or x.std() == 0:
        return float(n)
    iid = x.std(ddof=0) / math.sqrt(n)
    se = hac_se(x, lags)
    if not se or se <= 0:
        return float(n)
    return float(min(n, max(1.0, n * (iid / se) ** 2)))


def hac_mean_test(x) -> dict:
    """Mean with a Newey-West standard error: the t-test that tolerates serial correlation."""
    x = _arr(x)
    n = x.size
    out = {"n": int(n), "mean": float(x.mean()) if n else None, "se": None, "t": None, "p_value": None, "lags": nw_lags(n), "n_eff": float(n)}
    se = hac_se(x)
    if se and se > 0:
        from scipy.stats import t as tdist
        ne = effective_n(x)
        tt = float(x.mean() / se)
        out.update(se=float(se), t=tt, n_eff=ne, p_value=float(2 * (1 - tdist.cdf(abs(tt), max(1.0, ne - 1)))))
    return out


def block_length(x) -> int:
    """Block length for a block bootstrap, from the stronger of the serial correlation of the
    values and of their magnitudes (volatility clustering). 1 means the ordinary bootstrap."""
    x = _arr(x)
    n = x.size
    if n < 12:
        return 1
    rho = max(acf(x, 1), acf(np.abs(x - x.mean()), 1), 0.0)
    rho = min(rho, 0.97)
    if rho < 0.03:
        return 1
    b = (1.5 ** (1 / 3)) * ((2 * rho / (1 - rho * rho)) ** (2 / 3)) * n ** (1 / 3)
    return int(min(max(2, math.ceil(b)), max(2, n // 4)))


# ----------------------------------------------------------------------------- block bootstraps
def moving_block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap: blocks of `block` consecutive observations."""
    block = max(1, min(int(block), n))
    k = math.ceil(n / block)
    starts = rng.integers(0, n, (reps, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(reps, k * block)[:, :n]


def stationary_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: block lengths are geometric with the given mean."""
    p = 1.0 / max(1.0, float(mean_block))
    new = rng.random((reps, n)) < p
    new[:, 0] = True
    pos = np.arange(n)[None, :]
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    starts = rng.integers(0, n, (reps, n))
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (pos - last)) % n


def _draws(x: np.ndarray, method: str, block: int, reps: int, seed: int, stat) -> np.ndarray:
    n = x.size
    rng = np.random.default_rng(seed)
    out = []
    per = max(1, min(reps, CHUNK // max(n, 1)))
    left = reps
    while left > 0:
        r = min(per, left)
        if method == "iid" or block <= 1:
            idx = rng.integers(0, n, (r, n))
        elif method == "moving":
            idx = moving_block_indices(n, block, r, rng)
        else:
            idx = stationary_indices(n, block, r, rng)
        out.append(stat(x[idx]))
        left -= r
    return np.concatenate(out)


def _mean_rows(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1)


def _sharpe_rows(a: np.ndarray) -> np.ndarray:
    sd = a.std(axis=1, ddof=1)
    return np.where(sd > 0, a.mean(axis=1) / np.where(sd > 0, sd, 1), np.nan)


STATS = {"mean": _mean_rows, "sharpe": _sharpe_rows, "median": lambda a: np.median(a, axis=1), "sum": lambda a: a.sum(axis=1)}


def bootstrap_ci(x, stat: str = "mean", method: str = "stationary", block: int | None = None, reps: int = REPS, seed: int = 7, level: float = 0.95) -> dict:
    """Bootstrap interval for a statistic of a TIME-ORDERED series. `method`: stationary,
    moving or iid. The block length is chosen from the data unless given."""
    x = _arr(x)
    n = x.size
    f = STATS[stat]
    out = {"n": int(n), "stat": stat, "method": method, "block": 1, "estimate": None, "ci": [None, None], "se": None, "iid_ci": [None, None], "n_eff": float(n), "level": level}
    if n == 0:
        return out
    out["estimate"] = float(f(x[None, :])[0])
    if n < 3:
        return out
    b = int(block) if block else block_length(x)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    d = _draws(x, method, b, reps, seed, f)
    d = d[np.isfinite(d)]
    i = _draws(x, "iid", 1, reps, seed + 1, f)
    i = i[np.isfinite(i)]
    out.update(block=b if method != "iid" else 1, n_eff=effective_n(x))
    if d.size:
        out.update(ci=[float(np.quantile(d, lo)), float(np.quantile(d, hi))], se=float(d.std(ddof=1)))
        if stat == "mean" and method != "iid":
            se = hac_se(x)
            if se:
                from scipy.stats import t as tdist
                tq = float(tdist.ppf(hi, max(2.0, out["n_eff"] - 1)))
                out["ci"] = [float(min(out["ci"][0], x.mean() - tq * se)), float(max(out["ci"][1], x.mean() + tq * se))]
                out["se"] = float(max(out["se"], se))
    if i.size:
        out["iid_ci"] = [float(np.quantile(i, lo)), float(np.quantile(i, hi))]
    return out


def bootstrap_p(x, null: float = 0.0, method: str = "stationary", block: int | None = None, reps: int = REPS, seed: int = 7) -> float | None:
    """Two-sided bootstrap p-value for 'mean == null' (shift method): the series is recentred
    on the null and the observed distance is compared with the resampled ones."""
    x = _arr(x)
    if x.size < 3 or x.std() == 0:
        return None
    b = int(block) if block else block_length(x)
    d = _draws(x - x.mean(), method, b, reps, seed, _mean_rows)
    est = float(x.mean() - null)
    return _p_from_draws(d + est, est, max(float(d.std(ddof=1)), hac_se(x) or 0.0), effective_n(x) - 1)


def _p_from_draws(draws: np.ndarray, est: float, se: float | None, dof: float) -> float:
    """Two-sided p-value of 'the true value is zero' from bootstrap draws centred on the
    estimate. A resampling p-value cannot be smaller than 1/(draws+1); when it is within a
    few steps of that floor, the t-approximation with the bootstrap standard error and the
    number of independent clusters as degrees of freedom is used instead, so that small
    p-values keep their resolution when many hypotheses are corrected for."""
    p = float((1 + (np.abs(draws - est) >= abs(est) - 1e-15).sum()) / (1 + draws.size))
    if p > 10.0 / (1 + draws.size) or not se or se <= 0:
        return p
    from scipy.stats import t as tdist
    return float(min(p, max(2 * (1 - tdist.cdf(abs(est) / se, max(2.0, dof))), 1e-12)))


# ----------------------------------------------------------------------------- calendar clusters
def _to_date(d) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d)[:10])


def cluster_ids(dates, by: str = "week") -> np.ndarray:
    """Integer cluster label per observation: session (each date), ISO week, or month."""
    out = np.empty(len(dates), np.int64)
    for i, d in enumerate(dates):
        d = _to_date(d)
        if by == "session":
            out[i] = d.toordinal()
        elif by == "month":
            out[i] = d.year * 100 + d.month
        else:
            iso = d.isocalendar()
            out[i] = iso[0] * 100 + iso[1]
    return out


def _cluster_sums(values: np.ndarray, weights: np.ndarray, clusters: np.ndarray, uniq: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if uniq is None:
        uniq = np.unique(clusters)
    pos = np.searchsorted(uniq, clusters)
    S = np.bincount(pos, weights=values * weights, minlength=uniq.size)
    N = np.bincount(pos, weights=weights, minlength=uniq.size)
    return S, N, uniq


def _cluster_draw(G: int, reps: int, rng: np.random.Generator, block: int = 1):
    """Index draws over G time-ordered clusters. With block > 1, runs of consecutive clusters
    are kept together (dependence that crosses a week boundary survives the resampling)."""
    per = max(1, min(reps, CHUNK // max(G, 1)))
    left = reps
    while left > 0:
        r = min(per, left)
        yield rng.integers(0, G, (r, G)) if block <= 1 else moving_block_indices(G, block, r, rng)
        left -= r


def _cluster_block(resid: np.ndarray) -> int:
    """Block length over clusters, from the serial correlation of the cluster residuals."""
    G = resid.size
    if G < 16:
        return 1
    rho = min(max(acf(resid, 1), acf(np.abs(resid), 1), 0.0), 0.95)
    if rho < 0.05:
        return 1
    return int(min(max(2, math.ceil((1.5 ** (1 / 3)) * ((2 * rho / (1 - rho * rho)) ** (2 / 3)) * G ** (1 / 3))), max(2, G // 5)))


def cluster_mean_ci(values, dates, by: str = "week", reps: int = REPS, seed: int = 7, level: float = 0.95) -> dict:
    """Mean of event observations with an interval from resampling whole calendar clusters:
    observations that share a week (or month) are kept together, as they arrived together."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = v[ok]
    cl = cluster_ids([d for d, m in zip(dates, ok) if m], by)
    n = v.size
    out = {"n": int(n), "mean": float(v.mean()) if n else None, "ci": [None, None], "se": None, "clusters": int(np.unique(cl).size) if n else 0, "by": by, "design_effect": None, "n_eff": float(n),
           "iid_ci": [None, None]}
    if n < 3 or out["clusters"] < 2:
        return out
    S, N, uniq = _cluster_sums(v, np.ones(n), cl)
    resid = S - v.mean() * N
    blk = _cluster_block(resid)
    rng = np.random.default_rng(seed)
    draws = []
    for idx in _cluster_draw(uniq.size, reps, rng, blk):
        den = N[idx].sum(axis=1)
        draws.append(np.where(den > 0, S[idx].sum(axis=1) / np.where(den > 0, den, 1), np.nan))
    d = np.concatenate(draws)
    d = d[np.isfinite(d)]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    se = float(d.std(ddof=1))
    # linearised cluster-robust standard error with a Newey-West correction across neighbouring clusters
    G = uniq.size
    se_lin = (hac_se(resid * G / n) or 0.0)
    se = max(se, se_lin)
    from scipy.stats import t as tdist
    tq = float(tdist.ppf(hi, max(2, G - 1)))
    iid_se = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    deff = (se / iid_se) ** 2 if iid_se > 0 else None
    # the reported interval is the wider of the percentile and the t interval: clustering may widen it, never narrow it
    out.update(ci=[float(min(np.quantile(d, lo), v.mean() - tq * se)), float(max(np.quantile(d, hi), v.mean() + tq * se))], se=se, design_effect=deff,
               n_eff=float(n / deff) if deff and deff > 0 else float(n), iid_ci=[float(v.mean() - 1.96 * iid_se), float(v.mean() + 1.96 * iid_se)], cluster_block=blk)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """Two-sided p-value of a two-proportion z-test that assumes independent observations
    (None when it cannot be computed)."""
    if n1 <= 0 or n2 <= 0:
        return None
    pp = (k1 + k2) / (n1 + n2)
    if not 0 < pp < 1:
        return None
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se <= 0:
        return None
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs((k1 / n1 - k2 / n2) / se))))


def iid_mean_ci(x, reps: int = 2000, seed: int = 1) -> list[float | None]:
    """Percentile bootstrap interval of a mean that treats observations as independent.
    Used only for descriptive tables; conclusions use the cluster intervals."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (reps, x.size))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def cluster_prop_ci(hits, dates, by: str = "week", reps: int = REPS, seed: int = 7) -> dict:
    """A frequency (hits / observations) with a calendar-cluster interval next to the Wilson
    interval that assumes independence. `design_effect` > 1 means the sample carries less
    information than its size suggests."""
    h = np.asarray(hits, float)
    n = int(h.size)
    k = int(h.sum())
    out = {"n": n, "hits": k, "p": (k / n) if n else None, "wilson": wilson(k, n), "ci": wilson(k, n), "clusters": 0, "by": by, "design_effect": None, "n_eff": float(n), "method": "wilson"}
    if n < 3:
        return out
    r = cluster_mean_ci(h, dates, by, reps, seed)
    out["clusters"] = r["clusters"]
    if r["ci"][0] is None or r["clusters"] < 5:
        return out
    # widen, never narrow: the reported interval is the union of the two, so a small cluster count cannot make it look tighter
    w = out["wilson"]
    out.update(ci=[max(0.0, min(w[0], r["ci"][0])), min(1.0, max(w[1], r["ci"][1]))], design_effect=r["design_effect"], n_eff=r["n_eff"], method=f"{by}-cluster bootstrap")
    return out


def lift_test(cond_hits, cond_dates, rest_hits, rest_dates, by: str = "week", reps: int = REPS, seed: int = 7) -> dict:
    """Difference between the frequency on qualifying sessions and on all OTHER evaluated
    sessions, with whole calendar clusters resampled jointly (so a volatile week contributes
    to both groups or to neither). Returns the cluster interval and p-value next to the
    two-proportion z-test that assumes independence."""
    ch, rh = np.asarray(cond_hits, float), np.asarray(rest_hits, float)
    n1, n2 = int(ch.size), int(rh.size)
    out = {"n": n1, "rest_n": n2, "p": float(ch.mean()) if n1 else None, "rest_p": float(rh.mean()) if n2 else None, "lift": None, "ci": [None, None], "p_value": None, "p_iid": None,
           "clusters": 0, "cond_clusters": 0, "design_effect": None, "by": by}
    if not n1 or not n2:
        return out
    lift = float(ch.mean() - rh.mean())
    out["lift"] = lift
    pp = (ch.sum() + rh.sum()) / (n1 + n2)
    se_iid = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else 0.0
    if se_iid > 0:
        from scipy.stats import norm
        out["p_iid"] = float(2 * (1 - norm.cdf(abs(lift) / se_iid)))
    c1, c2 = cluster_ids(cond_dates, by), cluster_ids(rest_dates, by)
    uniq = np.unique(np.concatenate([c1, c2]))
    out["clusters"], out["cond_clusters"] = int(uniq.size), int(np.unique(c1).size)
    if out["cond_clusters"] < 5 or uniq.size < 6 or n1 < 10 or n2 < 10:
        return out  # too few independent periods for any dependence-robust statement
    S1, N1, _ = _cluster_sums(ch, np.ones(n1), c1, uniq)
    S2, N2, _ = _cluster_sums(rh, np.ones(n2), c2, uniq)
    blk = _cluster_block((S1 - ch.mean() * N1) + (S2 - rh.mean() * N2))
    rng = np.random.default_rng(seed)
    draws = []
    for idx in _cluster_draw(uniq.size, reps, rng, blk):
        a, b = N1[idx].sum(axis=1), N2[idx].sum(axis=1)
        ok = (a > 0) & (b > 0)
        d = np.full(idx.shape[0], np.nan)
        d[ok] = S1[idx].sum(axis=1)[ok] / a[ok] - S2[idx].sum(axis=1)[ok] / b[ok]
        draws.append(d)
    d = np.concatenate(draws)
    d = d[np.isfinite(d)]
    if d.size < 200:
        return out
    se = float(d.std(ddof=1))
    p = _p_from_draws(d, lift, se, out["cond_clusters"] - 1)
    out.update(ci=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], p_value=p, se=se, design_effect=((se / se_iid) ** 2 if se_iid > 0 else None))
    return out


def mean_diff_test(a_vals, a_dates, b_vals, b_dates, by: str = "week", reps: int = REPS, seed: int = 7) -> dict:
    """Difference in means between two groups of sessions, calendar clusters resampled jointly."""
    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a, b = a[ma], b[mb]
    da, db = [d for d, m in zip(a_dates, ma) if m], [d for d, m in zip(b_dates, mb) if m]
    out = {"n_a": int(a.size), "n_b": int(b.size), "mean_a": float(a.mean()) if a.size else None, "mean_b": float(b.mean()) if b.size else None, "diff": None, "ci": [None, None], "p_value": None, "p_iid": None, "by": by}
    if a.size < 3 or b.size < 3:
        return out
    diff = float(a.mean() - b.mean())
    out["diff"] = diff
    if a.std(ddof=1) > 0 or b.std(ddof=1) > 0:
        from scipy.stats import ttest_ind
        out["p_iid"] = float(ttest_ind(a, b, equal_var=False).pvalue)
    c1, c2 = cluster_ids(da, by), cluster_ids(db, by)
    uniq = np.unique(np.concatenate([c1, c2]))
    if np.unique(c1).size < 3 or np.unique(c2).size < 3:
        return out
    S1, N1, _ = _cluster_sums(a, np.ones(a.size), c1, uniq)
    S2, N2, _ = _cluster_sums(b, np.ones(b.size), c2, uniq)
    blk = _cluster_block((S1 - a.mean() * N1) + (S2 - b.mean() * N2))
    rng = np.random.default_rng(seed)
    draws = []
    for idx in _cluster_draw(uniq.size, reps, rng, blk):
        x, y = N1[idx].sum(axis=1), N2[idx].sum(axis=1)
        ok = (x > 0) & (y > 0)
        d = np.full(idx.shape[0], np.nan)
        d[ok] = S1[idx].sum(axis=1)[ok] / x[ok] - S2[idx].sum(axis=1)[ok] / y[ok]
        draws.append(d)
    d = np.concatenate(draws)
    d = d[np.isfinite(d)]
    if d.size >= 200:
        out.update(ci=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], p_value=_p_from_draws(d, diff, float(d.std(ddof=1)), min(np.unique(c1).size, np.unique(c2).size) - 1))
    return out


def shift_test(mask, outcome, reps: int = 1000, seed: int = 7, min_shift: int = 20) -> dict:
    """Circular-shift permutation test of 'the outcome is different when the condition is on'.

    `mask` and `outcome` are aligned, in session order, over every evaluated session. The
    condition series is rotated in time by a random number of sessions: this keeps the
    clustering of the condition and the serial structure of the outcome exactly as they
    are and only breaks their alignment. The p-value is the share of rotations whose
    difference is at least as large as the observed one."""
    m = np.asarray(mask, bool)
    y = np.asarray(outcome, float)
    ok = np.isfinite(y)
    m, y = m[ok], y[ok]
    n = y.size
    k = int(m.sum())
    out = {"n": n, "n_condition": k, "diff": None, "p_value": None, "rotations": 0}
    if k < 3 or n - k < 3 or n < 4 * min_shift:
        return out
    tot = y.sum()
    obs = y[m].mean() - (tot - y[m].sum()) / (n - k)
    rng = np.random.default_rng(seed)
    shifts = np.unique(rng.integers(min_shift, n - min_shift, reps))
    cs = np.cumsum(np.r_[y, y])  # rotation of the mask == a sum over rotated positions of the outcome
    pos = np.flatnonzero(m)
    yy = np.r_[y, y]
    sums = np.array([yy[pos + s].sum() for s in shifts]) if pos.size * shifts.size <= 4_000_000 else np.array([yy[(pos + s)].sum() for s in shifts[:400]])
    del cs
    d = sums / k - (tot - sums) / (n - k)
    out.update(diff=float(obs), p_value=float((1 + (np.abs(d) >= abs(obs) - 1e-15).sum()) / (1 + d.size)), rotations=int(d.size))
    return out


# ----------------------------------------------------------------------------- paired comparisons
def paired_compare(a: dict, b: dict, by: str = "week", reps: int = REPS, seed: int = 7) -> dict:
    """Compare two configurations on the sessions (or keys) BOTH have an observation for.

    a, b: {date: value}. Differences are taken within each shared session, so anything that
    affects both alike cancels; the interval resamples calendar clusters of the differences."""
    keys = sorted(set(a) & set(b))
    out = {"pairs": len(keys), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)), "mean_a": None, "mean_b": None, "mean_diff": None, "ci": [None, None], "p_value": None,
           "sign_p": None, "a_better": 0, "b_better": 0, "ties": 0, "by": by}
    if not keys:
        return out
    x = np.array([float(a[k]) for k in keys])
    y = np.array([float(b[k]) for k in keys])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    keys = [k for k, m in zip(keys, ok) if m]
    d = x - y
    out.update(pairs=int(d.size), mean_a=float(x.mean()) if d.size else None, mean_b=float(y.mean()) if d.size else None, mean_diff=float(d.mean()) if d.size else None,
               a_better=int((d > 0).sum()), b_better=int((d < 0).sum()), ties=int((d == 0).sum()))
    nz = int((d != 0).sum())
    if nz >= 1:
        from scipy.stats import binomtest
        out["sign_p"] = float(binomtest(int((d > 0).sum()), nz, 0.5).pvalue)
    if d.size >= 6:
        ci = cluster_mean_ci(d, keys, by, reps, seed)
        out["ci"] = ci["ci"]
        if ci["se"] and ci["se"] > 0:
            cl = cluster_ids(keys, by)
            S, N, uniq = _cluster_sums(d - d.mean(), np.ones(d.size), cl)
            rng = np.random.default_rng(seed + 3)
            dr = np.concatenate([S[i].sum(axis=1) / np.maximum(N[i].sum(axis=1), 1) for i in _cluster_draw(uniq.size, reps, rng)])
            out["p_value"] = _p_from_draws(dr + d.mean(), float(d.mean()), ci["se"], uniq.size - 1)
    return out


def mcnemar(a_hits: dict, b_hits: dict) -> dict:
    """Paired comparison of two yes/no outcomes on shared sessions (exact McNemar test)."""
    keys = sorted(set(a_hits) & set(b_hits))
    only_a = sum(1 for k in keys if a_hits[k] and not b_hits[k])
    only_b = sum(1 for k in keys if b_hits[k] and not a_hits[k])
    p = None
    if only_a + only_b > 0:
        from scipy.stats import binomtest
        p = float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {"pairs": len(keys), "a_only": only_a, "b_only": only_b, "both": sum(1 for k in keys if a_hits[k] and b_hits[k]), "neither": sum(1 for k in keys if not a_hits[k] and not b_hits[k]), "p_value": p}


# ----------------------------------------------------------------------------- multiple hypotheses
def adjust(pvals: list[float | None], method: str = "holm", m: int | None = None) -> list[float | None]:
    """Adjusted p-values. `m` (default: the number of tests given) can be raised to the number
    of hypotheses tried in the whole research session, including ones not in this list.

    bonferroni / holm control the chance of ANY false positive; bh controls the expected
    share of false positives among the discoveries and assumes independent or positively
    dependent tests; by (Benjamini-Yekutieli) holds under arbitrary dependence."""
    idx = [i for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    out: list[float | None] = [None] * len(pvals)
    k = len(idx)
    if not k:
        return out
    M = max(int(m or k), k)
    order = sorted(idx, key=lambda i: pvals[i])
    if method == "bonferroni":
        for i in idx:
            out[i] = float(min(1.0, pvals[i] * M))
    elif method == "holm":
        prev = 0.0
        for rank, i in enumerate(order):
            prev = max(prev, min(1.0, pvals[i] * (M - rank)))
            out[i] = float(prev)
    elif method in ("bh", "by"):
        c = sum(1.0 / j for j in range(1, M + 1)) if method == "by" else 1.0
        prev = 1.0
        for rank in range(k, 0, -1):
            i = order[rank - 1]
            prev = min(prev, pvals[i] * M * c / rank)
            out[i] = float(min(1.0, prev))
    else:
        raise ValueError(f"unknown adjustment {method}")
    return out


# ----------------------------------------------------------------------------- chronological validation
def uniqueness(starts, ends) -> np.ndarray:
    """Average uniqueness of each observation over the sessions it spans (Lopez de Prado):
    1 when nothing else is open at the same time, 1/k when k observations always overlap.
    `starts`, `ends`: inclusive integer session indices."""
    s, e = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    if s.size == 0:
        return np.array([])
    lo = int(s.min())
    conc = np.zeros(int(e.max()) - lo + 2)
    np.add.at(conc, s - lo, 1)
    np.add.at(conc, e - lo + 1, -1)
    conc = np.cumsum(conc)[:-1]
    inv = np.r_[0.0, np.cumsum(1.0 / np.maximum(conc, 1))]
    return (inv[e - lo + 1] - inv[s - lo]) / (e - s + 1)


def overlap_summary(starts, ends) -> dict:
    s, e = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    n = int(s.size)
    if not n:
        return {"n": 0, "overlapping": 0, "overlapping_share": None, "max_concurrent": 0, "effective_n": 0.0, "avg_uniqueness": None}
    u = uniqueness(s, e)
    lo = int(s.min())
    conc = np.zeros(int(e.max()) - lo + 2)
    np.add.at(conc, s - lo, 1)
    np.add.at(conc, e - lo + 1, -1)
    conc = np.cumsum(conc)[:-1]
    over = int((u < 1 - 1e-12).sum())
    return {"n": n, "overlapping": over, "overlapping_share": over / n, "max_concurrent": int(conc.max()), "effective_n": float(u.sum()), "avg_uniqueness": float(u.mean())}


def purged_kfold(starts, ends, k: int = 5, embargo: int = 0) -> list[dict]:
    """Chronological K-fold for observations that span time. Each test fold is a contiguous
    block; training drops (purges) every observation whose span overlaps the test block and
    (embargo) every observation that starts within `embargo` sessions after it, so nothing
    in training shares information with what it is scored on."""
    s, e = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    n = int(s.size)
    order = np.argsort(s, kind="stable")
    k = max(2, min(int(k), n))
    folds = []
    for part in np.array_split(order, k):
        if part.size == 0:
            continue
        t0, t1 = int(s[part].min()), int(e[part].max())
        test = np.zeros(n, bool)
        test[part] = True
        overlap = (s <= t1) & (e >= t0) & ~test
        emb = (s > t1) & (s <= t1 + int(embargo)) & ~test
        train = ~test & ~overlap & ~emb
        folds.append({"test": np.flatnonzero(test), "train": np.flatnonzero(train), "purged": int(overlap.sum()), "embargoed": int(emb.sum()), "span": [t0, t1]})
    return folds


def walk_forward(n: int, train: int, test: int, step: int | None = None, anchored: bool = False, embargo: int = 0) -> list[dict]:
    """Rolling (or anchored) train/test windows over n ordered sessions, with an embargo gap."""
    step = int(step or test)
    out = []
    a = 0
    while a + train + embargo + test <= n:
        out.append({"train": [0 if anchored else a, a + train - 1], "test": [a + train + embargo, a + train + embargo + test - 1]})
        a += step
    return out


def selection_cv(M: np.ndarray, starts, ends, k: int = 5, embargo: int = 0, min_obs: int = 8) -> dict:
    """Does choosing the best configuration on some of the history pay off on the rest?

    M: observations x configurations, the outcome of each observation under each
    configuration (NaN where the configuration does not include the observation).
    For every purged fold the configuration with the best mean on the TRAINING rows is
    scored on the test rows; the result shows how the chosen one ranks there."""
    M = np.asarray(M, float)
    n, c = M.shape
    if c < 2 or n < 4 * min_obs:
        return {"folds": [], "reason": "too few observations or configurations"}
    folds = purged_kfold(starts, ends, k, embargo)
    rows = []
    for f in folds:
        tr, te = M[f["train"]], M[f["test"]]
        ntr, nte = np.isfinite(tr).sum(axis=0), np.isfinite(te).sum(axis=0)
        mtr = np.where(ntr >= min_obs, np.nansum(tr, axis=0) / np.maximum(ntr, 1), np.nan) if tr.size else np.full(c, np.nan)
        mte = np.where(nte >= max(3, min_obs // 2), np.nansum(te, axis=0) / np.maximum(nte, 1), np.nan) if te.size else np.full(c, np.nan)
        if not np.isfinite(mtr).any() or not np.isfinite(mte).any():
            continue
        best = int(np.nanargmax(mtr))
        if not math.isfinite(mte[best]):
            continue
        valid = np.isfinite(mte)
        rank = float((mte[valid] < mte[best]).sum() + 0.5 * ((mte[valid] == mte[best]).sum() - 1)) / max(1, valid.sum() - 1) if valid.sum() > 1 else None
        rows.append({"span": f["span"], "chosen": best, "train_mean": float(mtr[best]), "test_mean": float(mte[best]), "test_rank": rank, "test_median_of_all": float(np.nanmedian(mte)),
                     "purged": f["purged"], "embargoed": f["embargoed"], "train_n": int(ntr[best]), "test_n": int(nte[best])})
    if not rows:
        return {"folds": [], "reason": "no fold had enough observations on both sides"}
    below = sum(1 for r in rows if r["test_rank"] is not None and r["test_rank"] < 0.5)
    ranked = [r for r in rows if r["test_rank"] is not None]
    return {"folds": rows, "k": len(rows), "embargo": int(embargo), "chosen_below_median": below, "share_below_median": (below / len(ranked)) if ranked else None,
            "mean_train": float(np.mean([r["train_mean"] for r in rows])), "mean_test": float(np.mean([r["test_mean"] for r in rows])),
            "distinct_choices": len({r["chosen"] for r in rows})}


# ----------------------------------------------------------------------------- regimes
def regime_table(values, labels, dates, order: list[str] | None = None, binary: bool = False, by: str = "week") -> dict:
    """The same statistic inside each regime, with a test of whether the regimes differ at
    all (chi-square for frequencies, Kruskal-Wallis for returns). Regimes with fewer than
    ten observations are shown but excluded from the test."""
    v = np.asarray(values, float)
    lab = [str(x) for x in labels]
    names = (order or []) + sorted(set(lab) - set(order or []))
    rows, groups = [], []
    for name in names:
        m = np.array([x == name for x in lab]) & np.isfinite(v)
        if not m.any():
            continue
        x = v[m]
        ds = [d for d, k in zip(dates, m) if k]
        if binary:
            ci = cluster_prop_ci(x, ds, by)
            rows.append({"regime": name, "n": int(x.size), "estimate": ci["p"], "ci": ci["ci"], "thin": x.size < MIN_OBS})
        else:
            ci = cluster_mean_ci(x, ds, by)
            rows.append({"regime": name, "n": int(x.size), "estimate": ci["mean"], "ci": ci["ci"] if ci["ci"][0] is not None else [None, None], "thin": x.size < MIN_OBS})
        if x.size >= 10:
            groups.append(x)
    p = None
    test = None
    if len(groups) >= 2:
        try:
            if binary:
                from scipy.stats import chi2_contingency
                tab = np.array([[g.sum(), g.size - g.sum()] for g in groups])
                if (tab.sum(axis=0) > 0).all():
                    p, test = float(chi2_contingency(tab)[1]), "chi-square"
            else:
                from scipy.stats import kruskal
                p, test = float(kruskal(*groups).pvalue), "Kruskal-Wallis"
        except ValueError:
            p = None
    ests = [r["estimate"] for r in rows if r["estimate"] is not None and not r["thin"]]
    return {"rows": rows, "heterogeneity_p": p, "test": test, "spread": (max(ests) - min(ests)) if len(ests) >= 2 else None,
            "note": "The heterogeneity test treats observations as independent; read it together with the per-regime cluster intervals."}


# ----------------------------------------------------------------------------- evidence grade
def grade(n: int, clusters: int | None, p_adj: float | None, effect_ok: bool = True, holdout: str | None = None, stable: bool | None = None, contaminated: bool = False) -> dict:
    """How strongly the evidence may be worded. Thin samples never get past 'insufficient',
    whatever the p-value; 'strong' additionally needs confirmation on untouched data."""
    reasons = []
    if n < MIN_OBS:
        return {"level": "insufficient", "label": "Too few observations", "reasons": [f"{n} observations (at least {MIN_OBS} are needed before any statistical statement)"]}
    if clusters is not None and clusters < MIN_CLUSTERS:
        return {"level": "insufficient", "label": "Too few independent periods", "reasons": [f"the {n} observations fall in only {clusters} separate weeks, so they are far from {n} independent draws"]}
    if p_adj is None or p_adj >= 0.10 or not effect_ok:
        return {"level": "none", "label": "Not distinguishable from chance", "reasons": ["the difference is within what chance produces once dependence and the number of hypotheses tried are allowed for"]}
    level = "weak" if p_adj >= 0.05 else "moderate"
    if p_adj < 0.01 and n >= 100 and (clusters is None or clusters >= 30) and holdout == "confirmed" and stable is not False and not contaminated:
        level = "strong"
    if n < 100:
        reasons.append(f"{n} observations: enough to test, not enough for a precise estimate")
    if holdout != "confirmed":
        reasons.append({"unconfirmed": "the final test period did not confirm it", "untested": "too few observations in the final test period to confirm or reject it", None: "it has not been checked on untouched data"}.get(holdout, "it has not been confirmed on untouched data"))
    if contaminated:
        reasons.append("the final test period has been used for tuning, so it no longer counts as untouched")
    if stable is False:
        reasons.append("neighbouring parameter values disagree")
    label = {"weak": "Weak evidence", "moderate": "Moderate evidence", "strong": "Strong evidence"}[level]
    return {"level": level, "label": label, "reasons": reasons}


# ----------------------------------------------------------------------------- helpers for tests / calibration
def simulate_clustered_binary(n: int, p: float, persistence: float, rng: np.random.Generator) -> np.ndarray:
    """A two-state Markov chain with stationary probability p: hits arrive in runs when
    persistence > 0 (used to check that the intervals above keep their stated coverage)."""
    a = p + persistence * (1 - p)  # P(1 | 1)
    b = p * (1 - persistence)      # P(1 | 0)
    out = np.empty(n, np.int8)
    x = rng.random() < p
    u = rng.random(n)
    for i in range(n):
        x = u[i] < (a if x else b)
        out[i] = x
    return out
