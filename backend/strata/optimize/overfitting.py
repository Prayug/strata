"""Multiple-testing and overfitting diagnostics.

* Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): probability that the selected
  configuration's Sharpe exceeds the maximum Sharpe expected from N unskilled trials.
* Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
  (Bailey, Borwein, Lopez de Prado & Zhu, 2015).
* Parameter stability: how much the objective degrades in the neighbourhood of a
  configuration on the parameter grid.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    if n_trials < 2 or sr_var <= 0:
        return 0.0
    return math.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sharpes: np.ndarray) -> dict:
    """returns: daily returns of the selected config; trial_sharpes: per-period (non-annualised)
    Sharpe ratios of all trials tested."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 10 or r.std(ddof=1) == 0:
        return {"dsr": None, "reason": "insufficient data"}
    sr = r.mean() / r.std(ddof=1)
    skew = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    kurt = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    ts = np.asarray(trial_sharpes, float)
    ts = ts[np.isfinite(ts)]
    var = ts.var(ddof=1) if ts.size > 1 else 0.0
    sr0 = expected_max_sharpe(max(n_trials, 1), var)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    psr0 = norm.cdf(sr * math.sqrt(T - 1) / denom)
    return {"dsr": float(norm.cdf(z)), "psr": float(psr0), "sharpe_annual": float(sr * math.sqrt(252)),
            "expected_max_sharpe_annual": float(sr0 * math.sqrt(252)), "n_trials": int(n_trials), "observations": int(T)}


def pbo_cscv(M: np.ndarray, S: int = 10) -> dict:
    """M: T x N matrix of per-period returns (rows aligned by date) for N configurations."""
    T, N = M.shape
    if N < 2 or T < S * 5:
        return {"pbo": None, "reason": f"need >=2 configurations and >= {S * 5} aligned observations"}
    S = S - (S % 2)
    groups = np.array_split(np.arange(T), S)

    def perf(rows):
        x = M[rows]
        sd = x.std(axis=0, ddof=1)
        return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)

    logits = []
    degr = []
    for combo in itertools.combinations(range(S), S // 2):
        is_rows = np.concatenate([groups[i] for i in combo])
        oos_rows = np.concatenate([groups[i] for i in range(S) if i not in combo])
        p_is, p_oos = perf(is_rows), perf(oos_rows)
        best = int(np.argmax(p_is))
        rank = (p_oos < p_oos[best]).sum() + 0.5 * ((p_oos == p_oos[best]).sum() - 1)
        w = (rank + 1) / (N + 1)
        logits.append(math.log(w / (1 - w)))
        degr.append((float(p_is[best]), float(p_oos[best])))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": int(len(logits)), "logits": logits.tolist(),
            "is_vs_oos": degr[:200], "blocks": S}


def stability_scores(params: list[dict], scores: np.ndarray, space: list[dict]) -> np.ndarray:
    """For each config: mean objective of its grid neighbours / its own objective (1 = flat
    plateau, <0.5 = isolated spike). Neighbours differ by one step in exactly one param."""
    keys = [p["path"] for p in space]
    vals = {p["path"]: list(p["values"]) for p in space}
    index = {}
    for i, pr in enumerate(params):
        try:
            index[tuple(vals[k].index(pr[k]) for k in keys)] = i
        except (ValueError, KeyError):
            continue
    out = np.full(len(params), np.nan)
    for coord, i in index.items():
        nb = []
        for d in range(len(keys)):
            for step in (-1, 1):
                c = list(coord)
                c[d] += step
                j = index.get(tuple(c))
                if j is not None and np.isfinite(scores[j]):
                    nb.append(scores[j])
        if nb and np.isfinite(scores[i]) and scores[i] > 0:
            out[i] = float(np.mean(nb) / scores[i])
    return out


def pareto_front(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of non-dominated points maximising both x and y."""
    idx = []
    for i in range(len(x)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        dominated = np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])) & np.isfinite(x) & np.isfinite(y))
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: x[i])
