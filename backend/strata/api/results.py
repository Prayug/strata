"""Persisting backtest results and building trade-replay payloads."""
from __future__ import annotations

import json
import math
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from strata.analytics.metrics import breakdowns, compute_metrics, distribution, series_payload
from strata.config import get_settings
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine.engine import BacktestResult
from strata.engine.market import DayChain, from_day_num


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    return o


def dumps(o) -> str:
    return json.dumps(_clean(o), separators=(",", ":"))


def run_dir(rid: str) -> Path:
    return get_settings().results_dir / rid


SUMMARY_KEYS = ["net_pnl", "total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor", "expectancy", "trades", "max_bp_used", "avg_capital_utilization"]


def save_backtest(rid: str, res: BacktestResult, extra: dict | None = None) -> dict:
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    cap = res.config.capital
    metrics = compute_metrics(res.trades, res.daily, cap.initial_capital, cap.risk_free_rate)
    payload = {
        "meta": res.meta, "metrics": metrics, "series": series_payload(res.trades, res.daily, cap.initial_capital, cap.cash_benchmark_rate),
        "breakdowns": breakdowns(res.trades),
        "distributions": {m: distribution(res.trades, m) for m in ("dollars", "pct", "risk")},
        "diagnostics": res.diagnostics, "config": res.config.model_dump(mode="json"),
        "strategies": [s.name for s in res.config.strategies],
        **(extra or {}),
    }
    (d / "result.json").write_text(dumps(payload))
    (d / "trades.json").write_text(dumps(res.trades))
    (d / "events.json").write_text(dumps(res.events))
    if res.paths is not None:
        res.paths.write_parquet(d / "paths.parquet")
        res.leg_paths.write_parquet(d / "legs.parquet")
    res.intraday.write_parquet(d / "intraday.parquet")
    summary = {k: metrics.get(k) for k in SUMMARY_KEYS}
    summary["equity_spark"] = _spark(res.daily["equity"].to_list() if not res.daily.is_empty() else [])
    summary["start"] = res.config.start.isoformat()
    summary["end"] = res.config.end.isoformat()
    summary["symbols"] = sorted({s.symbol for s in res.config.strategies})
    summary["synthetic"] = res.meta.get("synthetic")
    return _clean(summary)


def _spark(xs: list, n: int = 60) -> list:
    if not xs:
        return []
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [round(float(xs[i]), 2) for i in idx]


def load_result(rid: str) -> dict:
    p = run_dir(rid) / "result.json"
    if not p.exists():
        raise FileNotFoundError(rid)
    return json.loads(p.read_text())


def load_trades(rid: str) -> list[dict]:
    p = run_dir(rid) / "trades.json"
    return json.loads(p.read_text()) if p.exists() else []


def delete_results(rid: str) -> None:
    d = run_dir(rid)
    if d.exists():
        shutil.rmtree(d)


def trade_replay(rid: str, tid: int, store: MarketStore | None = None) -> dict:
    store = store or MarketStore()
    trades = load_trades(rid)
    t = next((x for x in trades if x["id"] == tid), None)
    if t is None:
        raise KeyError(tid)
    d = run_dir(rid)
    path = leg = None
    if (d / "paths.parquet").exists():
        p = pl.read_parquet(d / "paths.parquet").filter(pl.col("pos") == tid).sort("t")
        path = {c: p[c].to_list() for c in p.columns if c != "pos"}
        lp = pl.read_parquet(d / "legs.parquet").filter(pl.col("pos") == tid).sort(["t", "leg"])
        leg = {}
        for li in sorted(set(lp["leg"].to_list())):
            g = lp.filter(pl.col("leg") == li)
            leg[str(int(li))] = {c: [None if (isinstance(v, float) and not math.isfinite(v)) else v for v in g[c].to_list()] for c in ["t", "bid", "ask", "mid", "iv", "delta"]}
    events = [e for e in json.loads((d / "events.json").read_text()) if e.get("pos") == tid] if (d / "events.json").exists() else []
    # underlying candles around the trade
    ent, ext = t["entry_ts"], t["exit_ts"] or t["entry_ts"]
    span_days = (ext - ent) / 86400
    tf = 1 if span_days < 1 else 5 if span_days < 4 else 30 if span_days < 25 else 390
    et_ent = cal.to_et(ent * 1_000_000_000).date()
    et_ext = cal.to_et(ext * 1_000_000_000).date()
    pad = 0 if tf < 390 else 10
    m = store.read_minutes(t["symbol"], et_ent - timedelta(days=pad), et_ext + timedelta(days=pad // 2))
    candles = []
    if not m.is_empty():
        m = m.with_columns(pl.col("ts").dt.epoch("s").alias("t"))
        if tf < 390:
            m = m.with_columns((pl.col("t") // (tf * 60) * (tf * 60)).alias("b")).group_by("b", maintain_order=True).agg(
                pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum()).rename({"b": "t"})
        else:
            m = m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d", maintain_order=True).agg(
                pl.col("t").first(), pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum())
        candles = m.select(["t", "open", "high", "low", "close", "volume"]).rows()
    # original chain at entry for the legs' expirations
    chain = None
    df = store.read_option_day(t["symbol"], et_ent)
    if df is not None:
        ch = DayChain(df)
        si = ch.snapshot_at(ent)
        if si >= 0:
            s = ch.rows(si)
            exps = sorted({l["expiration"] for l in t["legs"] if l["expiration"]})
            spot = t["entry_spot"]
            rows = []
            for e in exps:
                en = (date.fromisoformat(e) - date(1970, 1, 1)).days
                idx = [i for i in range(s.start, s.stop) if ch.exp[i] == en]
                ks = sorted({float(ch.strike[i]) for i in idx})
                if not ks:
                    continue
                leg_ks = [l["strike"] for l in t["legs"] if l["expiration"] == e]
                center = np.argmin(np.abs(np.array(ks) - spot))
                lo_k = min(leg_ks + [ks[max(0, center - 12)]]) if leg_ks else ks[max(0, center - 12)]
                hi_k = max(leg_ks + [ks[min(len(ks) - 1, center + 12)]]) if leg_ks else ks[min(len(ks) - 1, center + 12)]
                for k in ks:
                    if k < lo_k - 1e-9 or k > hi_k + 1e-9:
                        continue
                    row = {"expiration": e, "strike": k}
                    for i in idx:
                        if ch.strike[i] == k:
                            side = "call" if ch.is_call[i] else "put"
                            r = ch.row_dict(i)
                            row[side] = {kk: r[kk] for kk in ("bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega", "volume", "open_interest", "quote_age")}
                    rows.append(row)
            chain = {"ts": int(ch.snap_ts[si]), "spot": spot, "rows": rows}
    return {"trade": t, "path": path, "legs": leg, "events": events, "candles": candles, "candle_minutes": tf, "chain": chain}
