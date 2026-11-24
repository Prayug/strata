"""Combined portfolio testing.

The combined run executes every strategy inside ONE engine loop with a shared account, so
cash, buying power, margin and the portfolio position limit are contended realistically.
For contrast we also run each strategy on its own with its allocated slice of capital and
sum them ("naive sum") - the gap between the two is what independent backtests miss.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import polars as pl

from strata.analytics.metrics import compute_metrics, drawdown
from strata.engine.engine import BacktestResult, run_backtest
from strata.engine.models import BacktestConfig


def portfolio_analysis(cfg: BacktestConfig, combined: BacktestResult, singles: list[BacktestResult]) -> dict:
    d = combined.daily
    names = [s.name for s in cfg.strategies]
    dates = [x.isoformat() for x in d["date"].to_list()]
    init = cfg.capital.initial_capital
    per_pnl = {names[i]: d[f"s{i}_pnl"].to_numpy().astype(float) for i in range(len(names)) if f"s{i}_pnl" in d.columns}
    daily_pnl = {k: np.diff(np.r_[0.0, v]) for k, v in per_pnl.items()}
    keys = list(daily_pnl)
    corr = np.full((len(keys), len(keys)), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            x, y = daily_pnl[a], daily_pnl[b]
            if x.std() > 0 and y.std() > 0:
                corr[i, j] = float(np.corrcoef(x, y)[0, 1])
    naive = np.full(len(dates), init, float)
    single_rows = []
    for i, (s, r) in enumerate(zip(cfg.strategies, singles)):
        sd = r.daily
        alloc_cap = init * s.allocation
        eq = dict(zip([x.isoformat() for x in sd["date"].to_list()], sd["equity"].to_numpy().astype(float)))
        series = np.array([eq.get(dt, np.nan) for dt in dates], float)
        series = np.where(np.isfinite(series), series, alloc_cap)
        naive += series - alloc_cap
        m = compute_metrics(r.trades, sd, alloc_cap, cfg.capital.risk_free_rate)
        comb_tr = [t for t in combined.trades if t["strategy_idx"] == i]
        single_rows.append({
            "strategy": s.name, "allocation": s.allocation, "standalone_trades": len(r.trades), "combined_trades": len(comb_tr),
            "standalone_pnl": m.get("net_pnl"), "combined_pnl": float(sum(t["pnl"] for t in comb_tr)),
            "standalone_sharpe": m.get("sharpe"), "standalone_max_dd": m.get("max_drawdown"),
            "contribution_pct": None,
        })
    total = sum(r["combined_pnl"] for r in single_rows)
    for r in single_rows:
        r["contribution_pct"] = r["combined_pnl"] / total if abs(total) > 1e-9 else None
    comb_eq = d["equity"].to_numpy().astype(float)
    _, comb_mdd, _ = drawdown(comb_eq)
    _, naive_mdd, _ = drawdown(naive)
    concurrent = {names[i]: d[f"s{i}_open"].to_list() for i in range(len(names)) if f"s{i}_open" in d.columns}
    skipped = combined.diagnostics.get("skipped_entries", {})
    capital_skips = {k: v for k, v in skipped.items() if "buying power" in k or "concurrent" in k or "capital" in k}
    return {
        "dates": dates,
        "strategies": names,
        "per_strategy_pnl": {k: v.tolist() for k, v in per_pnl.items()},
        "correlation": {"labels": keys, "matrix": [[None if not math.isfinite(x) else x for x in row] for row in corr.tolist()]},
        "combined_equity": comb_eq.tolist(),
        "naive_sum_equity": naive.tolist(),
        "combined_max_dd": comb_mdd, "naive_max_dd": naive_mdd,
        "rows": single_rows,
        "concurrent_positions": concurrent,
        "capital_constrained_skips": capital_skips,
        "notes": [
            "Combined = all strategies in one engine loop sharing cash, buying power and position limits.",
            "Naive sum = each strategy backtested alone on its allocated capital, P&L added together (ignores contention).",
        ],
    }


def run_portfolio(cfg: BacktestConfig, progress=None, should_cancel=None) -> tuple[BacktestResult, dict]:
    n = len(cfg.strategies)
    steps = n + 1

    def prog(stage):
        def f(i, total, d):
            if progress:
                progress((stage + i / max(total, 1)) / steps, f"{'combined portfolio' if stage == 0 else cfg.strategies[stage - 1].name}: {d}")
        return f

    combined = run_backtest(cfg, progress=prog(0), should_cancel=should_cancel)
    singles = []
    for i, s in enumerate(cfg.strategies):
        c = copy.deepcopy(cfg)
        sc = s.model_copy(update={"allocation": 1.0})
        c = c.model_copy(update={"strategies": [sc], "capital": cfg.capital.model_copy(update={"initial_capital": cfg.capital.initial_capital * s.allocation})})
        singles.append(run_backtest(c, progress=prog(i + 1), should_cancel=should_cancel, record_paths=False))
    return combined, portfolio_analysis(cfg, combined, singles)
