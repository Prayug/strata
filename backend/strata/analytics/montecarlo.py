"""Monte Carlo resampling of historical outcomes (Numba).

trade bootstrap : i.i.d. resampling of realised trade P&Ls (ignores serial dependence)
block bootstrap : moving-block resampling of daily returns, preserving autocorrelation
                  and volatility clustering within blocks of `block` sessions
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _trade_paths(pnls, n_sims, n_trades, initial, seed):
    np.random.seed(seed)
    finals = np.empty(n_sims)
    mdds = np.empty(n_sims)
    fan = np.empty((n_sims, n_trades + 1))
    for s in range(n_sims):
        eq = initial
        peak = initial
        mdd = 0.0
        fan[s, 0] = eq
        for i in range(n_trades):
            eq += pnls[np.random.randint(0, pnls.shape[0])]
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0 if peak > 0 else -1.0
            if dd < mdd:
                mdd = dd
            fan[s, i + 1] = eq
        finals[s] = eq
        mdds[s] = mdd
    return finals, mdds, fan


@njit(cache=True)
def _block_paths(rets, n_sims, horizon, block, seed):
    np.random.seed(seed)
    n = rets.shape[0]
    out = np.empty((n_sims, horizon))
    for s in range(n_sims):
        i = 0
        while i < horizon:
            start = np.random.randint(0, max(1, n - block + 1))
            for j in range(block):
                if i >= horizon:
                    break
                out[s, i] = rets[(start + j) % n]
                i += 1
    return out


def _fan(paths: np.ndarray, points: int = 120) -> dict:
    idx = np.unique(np.linspace(0, paths.shape[1] - 1, min(points, paths.shape[1])).astype(int))
    q = np.quantile(paths[:, idx], [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return {"x": idx.tolist(), "p05": q[0].tolist(), "p25": q[1].tolist(), "p50": q[2].tolist(), "p75": q[3].tolist(), "p95": q[4].tolist()}


def _hist(x: np.ndarray, bins=40) -> dict:
    c, e = np.histogram(x[np.isfinite(x)], bins=bins)
    return {"edges": e.tolist(), "counts": c.tolist()}


def trade_bootstrap(pnls: list[float], initial: float, n_sims: int = 2000, seed: int = 7, ruin_pct: float = 0.5) -> dict:
    p = np.asarray(pnls, float)
    if p.size < 2:
        return {"error": "need at least 2 trades"}
    finals, mdds, fan = _trade_paths(p, n_sims, p.size, float(initial), seed)
    ruin = (fan.min(axis=1) <= initial * (1 - ruin_pct)).mean()
    return {
        "method": "trade_bootstrap", "n_sims": n_sims, "n_trades": int(p.size), "seed": seed,
        "final_equity": {k: float(v) for k, v in zip(["p05", "p25", "p50", "p75", "p95"], np.quantile(finals, [0.05, 0.25, 0.5, 0.75, 0.95]))},
        "max_drawdown": {k: float(v) for k, v in zip(["p05", "p25", "p50", "p75", "p95"], np.quantile(mdds, [0.05, 0.25, 0.5, 0.75, 0.95]))},
        "prob_loss": float((finals < initial).mean()),
        "prob_ruin": float(ruin), "ruin_threshold": ruin_pct,
        "fan": _fan(fan), "final_hist": _hist(finals), "dd_hist": _hist(mdds),
    }


def block_bootstrap(daily_returns: list[float], initial: float, n_sims: int = 2000, block: int = 10, seed: int = 7, horizon: int | None = None) -> dict:
    r = np.asarray([x for x in daily_returns if x is not None and math.isfinite(x)], float)
    if r.size < block * 2:
        return {"error": f"need at least {2 * block} daily returns"}
    h = horizon or r.size
    paths_r = _block_paths(r, n_sims, h, block, seed)
    eq = initial * np.cumprod(1 + paths_r, axis=1)
    eq = np.concatenate([np.full((n_sims, 1), initial), eq], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (eq / peak - 1).min(axis=1)
    sd = paths_r.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, paths_r.mean(axis=1) / sd * math.sqrt(252), np.nan)
    years = h / 252
    cagr = (eq[:, -1] / initial) ** (1 / years) - 1
    q = lambda x: {k: float(v) for k, v in zip(["p05", "p25", "p50", "p75", "p95"], np.nanquantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]))}
    return {
        "method": "block_bootstrap", "n_sims": n_sims, "block": block, "horizon": h, "seed": seed,
        "final_equity": q(eq[:, -1]), "max_drawdown": q(mdd), "sharpe": q(sharpe), "cagr": q(cagr),
        "prob_loss": float((eq[:, -1] < initial).mean()), "prob_sharpe_below_0": float(np.nanmean(sharpe < 0)),
        "fan": _fan(eq), "final_hist": _hist(eq[:, -1]), "dd_hist": _hist(mdd), "sharpe_hist": _hist(sharpe),
    }
