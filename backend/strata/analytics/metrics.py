"""Performance analytics computed from trades and the daily account record."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

TRADING_DAYS = 252


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def max_streak(signs: list[int], target: int) -> int:
    best = cur = 0
    for s in signs:
        cur = cur + 1 if s == target else 0
        best = max(best, cur)
    return best


def drawdown(equity: np.ndarray) -> tuple[np.ndarray, float, int]:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    # longest drawdown duration in sessions
    longest = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return dd, float(dd.min()) if dd.size else 0.0, longest


def compute_metrics(trades: list[dict], daily: pl.DataFrame, initial: float, rf: float = 0.0) -> dict:
    m: dict = {"initial_capital": initial}
    pnls = np.array([t["pnl"] for t in trades], float)
    n = len(pnls)
    m["trades"] = n
    if daily.is_empty():
        return m
    eq = daily["equity"].to_numpy().astype(float)
    eq_prev = np.r_[initial, eq[:-1]]
    rets = eq / eq_prev - 1
    days = daily["date"].to_list()
    years = max((days[-1] - days[0]).days / 365.25, 1 / 365.25) if len(days) > 1 else 1 / 252
    final = eq[-1]
    m["net_pnl"] = final - initial
    m["total_return"] = final / initial - 1
    m["cagr"] = (final / initial) ** (1 / years) - 1 if final > 0 else -1.0
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ex = rets - rf_d
    sd = rets.std(ddof=1) if rets.size > 1 else 0
    m["volatility"] = sd * math.sqrt(TRADING_DAYS)
    m["sharpe"] = ex.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else None
    downside = np.minimum(ex, 0)
    dsd = math.sqrt((downside ** 2).mean()) if rets.size else 0
    m["sortino"] = ex.mean() / dsd * math.sqrt(TRADING_DAYS) if dsd > 0 else None
    dd, mdd, dd_len = drawdown(eq)
    m["max_drawdown"] = mdd
    m["max_drawdown_dollars"] = float((eq - np.maximum.accumulate(eq)).min())
    m["longest_drawdown_sessions"] = dd_len
    m["calmar"] = m["cagr"] / abs(mdd) if mdd < 0 else None
    q = np.quantile(rets, 0.05) if rets.size else 0
    m["var_95_daily"] = float(q)
    m["cvar_95_daily"] = float(rets[rets <= q].mean()) if (rets <= q).any() else None
    m["worst_day"] = float(rets.min()) if rets.size else None
    m["best_day"] = float(rets.max()) if rets.size else None
    m["skew_daily"] = float(((rets - rets.mean()) ** 3).mean() / sd ** 3) if sd > 0 else None
    m["kurtosis_daily"] = float(((rets - rets.mean()) ** 4).mean() / sd ** 4 - 3) if sd > 0 else None
    if n:
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        m["win_rate"] = len(wins) / n
        m["avg_winner"] = wins.mean() if wins.size else 0.0
        m["avg_loser"] = losses.mean() if losses.size else 0.0
        m["largest_winner"] = pnls.max()
        m["largest_loser"] = pnls.min()
        m["expectancy"] = pnls.mean()
        m["expectancy_std_err"] = pnls.std(ddof=1) / math.sqrt(n) if n > 1 else None
        gl = -losses.sum()
        m["profit_factor"] = wins.sum() / gl if gl > 0 else None
        m["payoff_ratio"] = (wins.mean() / -losses.mean()) if wins.size and losses.size and losses.mean() < 0 else None
        signs = [1 if p > 0 else -1 for p in pnls]
        m["max_consecutive_wins"] = max_streak(signs, 1)
        m["max_consecutive_losses"] = max_streak(signs, -1)
        risks = np.array([t["max_risk"] for t in trades], float)
        ror = np.array([t["return_on_risk"] for t in trades if t["return_on_risk"] is not None], float)
        m["avg_return_on_risk"] = ror.mean() if ror.size else None
        m["return_on_capital_at_risk"] = pnls.sum() / risks.sum() if risks.sum() > 0 else None
        hold = np.array([t["holding_minutes"] or 0 for t in trades], float)
        m["avg_holding_minutes"] = hold.mean()
        m["median_holding_minutes"] = float(np.median(hold))
        tq = np.quantile(pnls, 0.05)
        m["trade_cvar_95"] = pnls[pnls <= tq].mean()
        m["total_commissions"] = float(sum(t["commissions"] for t in trades))
        m["total_slippage"] = float(sum(t["slippage"] for t in trades))
        m["avg_mfe"] = float(np.mean([t["mfe"] for t in trades]))
        m["avg_mae"] = float(np.mean([t["mae"] for t in trades]))
        m["ambiguous_trades"] = sum(1 for t in trades if "ambiguous_exit" in t["flags"] or "intrabar_ambiguous" in t["flags"])
        m["flagged_trades"] = sum(1 for t in trades if t["flags"])
        # t-stat of mean trade P&L (does the edge survive noise?)
        m["t_stat"] = (pnls.mean() / (pnls.std(ddof=1) / math.sqrt(n))) if n > 1 and pnls.std(ddof=1) > 0 else None
    bp = (daily["bp_used_max"] if "bp_used_max" in daily.columns else daily["bp_used"]).to_numpy().astype(float)
    m["max_bp_used"] = float(np.nanmax(bp)) if bp.size else 0
    m["avg_capital_utilization"] = float(np.nanmean(bp / eq)) if bp.size else 0
    m["max_capital_utilization"] = float(np.nanmax(bp / eq)) if bp.size else 0
    exposed = (daily["positions_max"] if "positions_max" in daily.columns else daily["open_positions"]).to_numpy() > 0
    m["exposure"] = float(exposed.mean())
    m["exposure_adjusted_return"] = m["cagr"] / m["exposure"] if m["exposure"] > 0 else None
    m["return_on_avg_bp"] = m["net_pnl"] / np.nanmean(bp[bp > 0]) if (bp > 0).any() else None
    m["sessions"] = len(days)
    m["years"] = years
    b = daily["benchmark"].to_numpy().astype(float)
    if np.isfinite(b).sum() > 1:
        m["benchmark_return"] = b[-1] / b[np.isfinite(b)][0] - 1
        br = np.diff(b) / b[:-1]
        if br.size > 2 and rets.size > 2:
            c = np.corrcoef(rets[1:], br)[0, 1]
            m["benchmark_correlation"] = c if math.isfinite(c) else None
            beta = np.cov(rets[1:], br)[0, 1] / br.var(ddof=1) if br.var(ddof=1) > 0 else None
            m["beta"] = beta
    if "interest" in daily.columns:
        m["interest_earned"] = float(daily["interest"][-1])
    small = n < 30
    m["warnings"] = []
    if small:
        m["warnings"].append(f"Only {n} trades: statistics are not reliable (< 30).")
    if m.get("max_capital_utilization", 0) > 0.8:
        m["warnings"].append("Buying-power usage exceeded 80% of equity at some point.")
    if m.get("ambiguous_trades"):
        m["warnings"].append(f"{m['ambiguous_trades']} trades had stop/target ambiguity resolved conservatively.")
    return {k: (_f(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in m.items()}


def series_payload(trades: list[dict], daily: pl.DataFrame, initial: float, rf: float = 0.04) -> dict:
    """Chart-ready series for the results dashboard."""
    if daily.is_empty():
        return {}
    eq = daily["equity"].to_numpy().astype(float)
    dates = [d.isoformat() for d in daily["date"].to_list()]
    dd, _, _ = drawdown(eq)
    b = daily["benchmark"].to_numpy().astype(float)
    first = b[np.isfinite(b)][0] if np.isfinite(b).any() else np.nan
    bench_eq = initial * b / first
    years = np.array([(d - daily["date"][0]).days / 365.25 for d in daily["date"].to_list()])
    cash_eq = initial * (1 + rf) ** years
    rets = eq / np.r_[initial, eq[:-1]] - 1
    win = 63
    rs = [None] * len(rets)
    for i in range(win, len(rets)):
        w = rets[i - win + 1:i + 1]
        s = w.std(ddof=1)
        rs[i] = _f(w.mean() / s * math.sqrt(TRADING_DAYS)) if s > 0 else None
    # monthly returns
    dfm = daily.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym")).group_by("ym", maintain_order=True).agg(pl.col("equity").last())
    me = dfm["equity"].to_numpy()
    mprev = np.r_[initial, me[:-1]]
    monthly = [{"month": ym, "return": _f(e / p - 1)} for ym, e, p in zip(dfm["ym"].to_list(), me, mprev)]
    bp = (daily["bp_used_max"] if "bp_used_max" in daily.columns else daily["bp_used"]).to_numpy().astype(float)
    return {
        "dates": dates,
        "equity": [_f(x) for x in eq],
        "drawdown": [_f(x) for x in dd],
        "benchmark": [_f(x) for x in bench_eq],
        "cash": [_f(x) for x in cash_eq],
        "daily_return": [_f(x) for x in rets],
        "rolling_sharpe": rs,
        "monthly": monthly,
        "bp_used": [_f(x) for x in bp],
        "open_positions": (daily["positions_max"] if "positions_max" in daily.columns else daily["open_positions"]).to_list(),
        "greeks": {k: [_f(x) for x in daily[k].to_numpy()] for k in ["delta", "gamma", "theta", "vega"]},
        "vix": [_f(x) for x in daily["vix"].to_numpy()],
    }


def _bucket(v, edges, labels):
    if v is None or not math.isfinite(v):
        return "n/a"
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


def breakdowns(trades: list[dict]) -> dict:
    """P&L grouped by entry time, weekday, DTE, delta, VIX regime, exit reason, month."""
    if not trades:
        return {}
    groups: dict[str, dict[str, list[dict]]] = {k: {} for k in ["time_of_day", "weekday", "dte", "delta", "vix", "exit_reason", "regime", "strategy"]}
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for t in trades:
        hh = t["entry_minute"] // 30 * 30
        keys = {
            "time_of_day": f"{hh // 60:02d}:{hh % 60:02d}",
            "weekday": wd[t["entry_weekday"]],
            "dte": _bucket(t["entry_dte"], [1, 2, 8, 22, 46, 91], ["0", "1", "2-7", "8-21", "22-45", "46-90", "90+"]),
            "delta": _bucket(t["short_delta"], [0.05, 0.10, 0.16, 0.25, 0.35, 0.5], ["<.05", ".05-.10", ".10-.16", ".16-.25", ".25-.35", ".35-.50", ">.50"]),
            "vix": _bucket(t["entry_vix"], [12, 15, 20, 25, 30], ["<12", "12-15", "15-20", "20-25", "25-30", "30+"]),
            "exit_reason": t["exit_reason"],
            "regime": t.get("entry_regime") or "n/a",
            "strategy": t["strategy"],
        }
        for g, k in keys.items():
            groups[g].setdefault(k, []).append(t)
    order = {"weekday": wd, "dte": ["0", "1", "2-7", "8-21", "22-45", "46-90", "90+", "n/a"], "delta": ["<.05", ".05-.10", ".10-.16", ".16-.25", ".25-.35", ".35-.50", ">.50", "n/a"], "vix": ["<12", "12-15", "15-20", "20-25", "25-30", "30+", "n/a"]}
    out = {}
    for g, d in groups.items():
        keys = sorted(d, key=lambda k: order[g].index(k) if g in order and k in order[g] else k)
        rows = []
        for k in keys:
            ts = d[k]
            p = np.array([t["pnl"] for t in ts])
            rr = np.array([t["return_on_risk"] for t in ts if t["return_on_risk"] is not None])
            rows.append({"key": k, "trades": len(ts), "pnl": float(p.sum()), "avg_pnl": float(p.mean()), "win_rate": float((p > 0).mean()),
                         "avg_ror": float(rr.mean()) if rr.size else None, "std_pnl": float(p.std(ddof=1)) if len(p) > 1 else None})
        out[g] = rows
    return out


def distribution(trades: list[dict], mode: str = "dollars", bins: int = 40) -> dict:
    if not trades:
        return {"edges": [], "counts": []}
    if mode == "pct":
        x = np.array([t["pnl_pct_premium"] for t in trades], float)
    elif mode == "risk":
        x = np.array([t["return_on_risk"] if t["return_on_risk"] is not None else 0 for t in trades], float)
    else:
        x = np.array([t["pnl"] for t in trades], float)
    x = x[np.isfinite(x)]
    lo, hi = np.quantile(x, [0.005, 0.995]) if x.size > 10 else (x.min(), x.max())
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": edges.tolist(), "counts": counts.tolist(), "mean": float(x.mean()), "median": float(np.median(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
