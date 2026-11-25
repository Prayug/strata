"""Sensitivity maps: how a result changes when one or two assumptions change.

A single best configuration says little; what matters is whether its neighbours agree.
A map evaluates a two-dimensional grid of an assumption pair and reports, for EVERY cell,
the estimate, its sample, a dependence-aware interval and explicit flags:

  thin        fewer observations than the minimum - the estimate is shown but not trusted
  incomplete  a large share of sessions could not be evaluated (missing bars / history)
  significant distinguishable from the baseline after correcting for every cell of the map
  stable      the cell and most of its neighbours agree in direction

Flags are data, so the interface can draw them as marks and text instead of relying on
colour. Every cell carries the exact configuration behind it, so it can be opened as its
observations or as a full study / backtest.

Two kinds of map:
  * underlying maps (`underlying_map`) are answered instantly by the session-feature engine
  * option maps (`option_map`) re-run an existing option backtest over a grid of its own
    parameters on the option data already stored - nothing is downloaded or modelled
"""
from __future__ import annotations

import numpy as np

from strata.data.store import MarketStore
from strata.research import robust as R
from strata.research.features import A, CODE_NAMES, OutcomeSpec, Pred, T, cond_text, get_features, to_study

MIN_CELL = 30
MAX_CELLS = 144

# ----------------------------------------------------------------------------- underlying maps
PRESETS: dict[str, dict] = {
    "decline_vs_horizon": {
        "title": "Previous-day decline versus what follows", "question": "After a decline of at least X%, how often is the price higher N sessions later?",
        "x": {"kind": "threshold", "feature": "ret_1", "op": "<=", "values": [-0.5, -1, -1.5, -2, -2.5, -3, -4], "label": "Previous session's return at most", "unit": "%"},
        "y": {"kind": "horizon", "values": ["sessions:1", "sessions:2", "sessions:3", "sessions:5", "sessions:10"], "label": "Sessions held from the close"},
        "outcome": {"entry": "close", "horizon": "sessions:1", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}},
    "entry_vs_target": {
        "title": "Entry time versus target distance", "question": "Entering at time T, how often does price reach +X% before the close?",
        "x": {"kind": "entry", "values": ["open", "10:30", "11:30", "12:30", "13:30", "14:30"], "label": "Entry time (ET)"},
        "y": {"kind": "target", "values": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "label": "Target above the entry price", "unit": "%"},
        "outcome": {"entry": "open", "horizon": "session", "direction": "up", "target": 0.5, "stop": None, "measure": "touch"}},
    "target_vs_stop": {
        "title": "Profit target versus stop distance", "question": "From the open, how often is +X% reached before -Y%?",
        "x": {"kind": "target", "values": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "label": "Target above the open", "unit": "%"},
        "y": {"kind": "stop", "values": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "label": "Stop below the open", "unit": "%"},
        "outcome": {"entry": "open", "horizon": "session", "direction": "up", "target": 1.0, "stop": 0.5, "measure": "touch"}},
    "holding_vs_risk": {
        "title": "Holding period versus risk", "question": "Holding N sessions after a decline of at least X%: return and downside",
        "x": {"kind": "horizon", "values": ["sessions:1", "sessions:2", "sessions:3", "sessions:5", "sessions:10", "sessions:20"], "label": "Sessions held from the close"},
        "y": {"kind": "threshold", "feature": "ret_1", "op": "<=", "values": [-1, -2, -3], "label": "Previous session's return at most", "unit": "%"},
        "outcome": {"entry": "close", "horizon": "sessions:1", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}, "layer": "avg_end_return_pct"},
    "volatility_vs_condition": {
        "title": "Volatility regime versus performance", "question": "Does the pattern depend on the level of volatility?",
        "x": {"kind": "band", "feature": "vix_prev", "edges": [0, 15, 20, 25, 30, 200], "label": "Volatility index, previous close"},
        "y": {"kind": "threshold", "feature": "ret_1", "op": "<=", "values": [-0.5, -1, -2, -3], "label": "Previous session's return at most", "unit": "%"},
        "outcome": {"entry": "close", "horizon": "sessions:1", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}},
    "gap_vs_entry": {
        "title": "Gap size versus entry time", "question": "After a gap down of at least X%, entering at time T: how often does the session finish higher?",
        "x": {"kind": "threshold", "feature": "gap", "op": "<=", "values": [-0.25, -0.5, -1, -1.5, -2], "label": "Opening gap at most", "unit": "%"},
        "y": {"kind": "entry", "values": ["open", "10:30", "11:30", "12:30"], "label": "Entry time (ET)"},
        "outcome": {"entry": "open", "horizon": "session", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}},
    "streak_vs_horizon": {
        "title": "Losing streak versus what follows", "question": "After N consecutive lower closes, how often is the price higher later?",
        "x": {"kind": "threshold", "feature": "down_streak", "op": ">=", "values": [1, 2, 3, 4, 5, 6], "label": "Consecutive lower closes, at least", "unit": ""},
        "y": {"kind": "horizon", "values": ["sessions:1", "sessions:3", "sessions:5", "sessions:10"], "label": "Sessions held from the close"},
        "outcome": {"entry": "close", "horizon": "sessions:1", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}},
}
LAYERS = {"probability": "Frequency of the outcome", "lift": "Frequency minus the baseline (percentage points)", "avg_end_return_pct": "Average return over the horizon (%)",
          "q05_end_return_pct": "5th-percentile return over the horizon (%)", "avg_mae_pct": "Average adverse excursion (%)", "n": "Observations"}


def _apply(dim: dict, v, preds: list[Pred], oc: dict) -> tuple[list[Pred], dict, str]:
    k = dim["kind"]
    if k == "threshold":
        return preds + [Pred(dim["feature"], dim["op"], float(v))], oc, f"{v:g}{dim.get('unit', '')}"
    if k == "band":
        lo, hi = v
        return preds + [Pred(dim["feature"], "between", float(lo), float(hi))], oc, (f"below {hi:g}" if lo <= 0 else f"{lo:g}+" if hi >= 100 else f"{lo:g} to {hi:g}")
    if k == "entry":
        return preds, {**oc, "entry": v}, str(v)
    if k == "target":
        return preds, {**oc, "target": float(v)}, f"{v:g}{dim.get('unit', '%')}"
    if k == "stop":
        return preds, {**oc, "stop": float(v)}, f"{v:g}{dim.get('unit', '%')}"
    if k == "horizon":
        return preds, {**oc, "horizon": v, **({"entry": "close"} if oc.get("entry") == "open" and str(v).startswith("sessions:") and False else {})}, (str(v).split(":")[1] if ":" in str(v) else str(v))
    raise ValueError(f"unknown dimension kind {k}")


def _values(dim: dict) -> list:
    if dim["kind"] == "band":
        e = dim["edges"]
        return [(e[i], e[i + 1]) for i in range(len(e) - 1)]
    return list(dim["values"])


def underlying_map(symbol: str, spec: dict, store: MarketStore | None = None, start: str | None = None, end: str | None = None, cross: tuple[str, ...] = (), reps: int = 400) -> dict:
    """spec: {x, y, outcome, preds?, layer?} (a PRESETS entry or a custom one)."""
    store = store or MarketStore()
    sym = symbol.upper()
    ft = get_features(store, sym, cross)
    xs, ys = _values(spec["x"]), _values(spec["y"])
    if len(xs) * len(ys) > MAX_CELLS:
        raise ValueError(f"the map has {len(xs) * len(ys)} cells; the limit is {MAX_CELLS}")
    base_preds = [Pred.load(p) for p in spec.get("preds") or []]
    pm = ft.window(start, end)
    dates = np.array(ft.dates)
    cells = []
    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            preds, oc, xl = _apply(spec["x"], xv, list(base_preds), dict(spec["outcome"]))
            preds, oc, yl = _apply(spec["y"], yv, preds, oc)
            o = OutcomeSpec.load(oc)
            cell = {"xi": xi, "yi": yi, "x": xl, "y": yl, "preds": [p.dump() for p in preds], "outcome": o.dump(), "condition": cond_text(preds, sym), "outcome_text": o.text()}
            try:
                ev = ft.evaluate(preds, o)
            except (ValueError, KeyError) as e:
                cells.append({**cell, "status": "error", "error": str(e)[:160]})
                continue
            out = ev["outcome"]
            q = ev["qualifying"] & pm
            base = ev["evaluable"] & pm
            rq, rb = q & (out.code != A), base & (out.code != A)
            rest = rb & ~ev["qualifying"]
            n, k = int(rq.sum()), int((out.code[rq] == T).sum())
            bn, bk = int(rb.sum()), int((out.code[rb] == T).sum())
            window_n = int(pm[1:].sum())
            er = out.end_ret[q]
            cell.update(status="ok", n=n, hits=k, probability=(k / n) if n else None, baseline_n=bn, baseline_probability=(bk / bn) if bn else None, lift=((k / n - bk / bn) * 100) if n and bn else None,
                        ambiguous=int((q & (out.code == A)).sum()), evaluated=int(base.sum()), not_evaluated=window_n - int(base.sum()),
                        avg_end_return_pct=float(er.mean()) if er.size else None, q05_end_return_pct=float(np.quantile(er, 0.05)) if er.size >= 20 else None,
                        q50_end_return_pct=float(np.median(er)) if er.size else None, q95_end_return_pct=float(np.quantile(er, 0.95)) if er.size >= 20 else None,
                        avg_mae_pct=float(out.mae[q].mean()) if er.size else None, thin=n < MIN_CELL, incomplete=window_n > 0 and (window_n - int(base.sum())) / window_n > 0.2,
                        study=to_study(sym, preds, o, start, end) if o.direction != "either" else None)
            if n and rest.any():
                lt = R.lift_test((out.code[rq] == T).astype(float), dates[rq], (out.code[rest] == T).astype(float), dates[rest], "week", reps)
                ci = R.cluster_prop_ci((out.code[rq] == T).astype(float), dates[rq], "week", reps)
                cell.update(p_value=lt["p_value"], ci95=ci["ci"], clusters=lt["cond_clusters"], lift_ci95=[None if v is None else v * 100 for v in lt["ci"]])
                if er.size >= 6:
                    cell["return_ci95"] = R.cluster_mean_ci(er, dates[q], "week", reps)["ci"]
            cells.append(cell)
    ok = [c for c in cells if c["status"] == "ok"]
    adj = R.adjust([c.get("p_value") for c in ok], "holm")
    for c, a in zip(ok, adj):
        c["p_adjusted"] = a
        c["significant"] = bool(a is not None and a < 0.05 and not c["thin"])
    layer = spec.get("layer") or "lift"
    grid = {(c["xi"], c["yi"]): c for c in ok}
    for c in ok:
        v = c.get(layer if layer != "probability" else "lift")
        nb = [grid.get((c["xi"] + dx, c["yi"] + dy)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nb = [x for x in nb if x is not None and not x["thin"] and x.get(layer if layer != "probability" else "lift") is not None]
        agree = sum(1 for x in nb if v is not None and (x[layer if layer != "probability" else "lift"] > 0) == (v > 0))
        c["neighbours"], c["neighbours_agree"] = len(nb), agree
        c["stable"] = bool(v is not None and not c["thin"] and len(nb) >= 2 and agree / len(nb) >= 0.75)
    thin = sum(1 for c in ok if c["thin"])
    sig = sum(1 for c in ok if c["significant"])
    notes = [f"{len(cells)} cells were evaluated; significance is corrected for all of them (Holm) and uses a test that resamples whole calendar weeks.",
             "Underlying price behaviour only - these are not option returns."]
    if thin:
        notes.append(f"{thin} cell{'s have' if thin != 1 else ' has'} fewer than {MIN_CELL} observations: shown, but not to be relied on.")
    if any(c.get("incomplete") for c in ok):
        notes.append("Some cells could evaluate only part of the period (usually an intraday entry time, for which free sources keep only recent bars).")
    best = max((c for c in ok if not c["thin"] and c.get(layer) is not None), key=lambda c: abs(c[layer]), default=None)
    if best is not None and not best["stable"]:
        notes.append(f"The most extreme cell ({best['x']} / {best['y']}) is not in a stable region: its neighbours do not agree with it, so it should not be singled out.")
    return {"kind": "sensitivity", "map": "underlying", "symbol": sym, "title": spec.get("title", ""), "question": spec.get("question", ""), "x": {**spec["x"], "labels": [c["x"] for c in cells[:len(xs)]]},
            "y": {**spec["y"], "labels": [cells[i * len(xs)]["y"] for i in range(len(ys))]}, "cells": cells, "layer": layer, "layers": LAYERS, "window": [start or ft.dates[1], end or ft.dates[-1]],
            "summary": {"cells": len(cells), "thin": thin, "significant": sig, "stable": sum(1 for c in ok if c["stable"])}, "notes": notes, "synthetic": store.is_synthetic(sym)}


def cell_detail(symbol: str, preds: list[dict], outcome: dict, store: MarketStore | None = None, start: str | None = None, end: str | None = None, cross: tuple[str, ...] = (), limit: int = 3000) -> dict:
    """Observations and return distribution behind one cell (qualifying sessions against the baseline)."""
    store = store or MarketStore()
    ft = get_features(store, symbol, cross)
    ps = [Pred.load(p) for p in preds]
    oc = OutcomeSpec.load(outcome)
    ev = ft.evaluate(ps, oc)
    out = ev["outcome"]
    pm = ft.window(start, end)
    q, base = ev["qualifying"] & pm, ev["evaluable"] & pm
    cols = {p.feature: ft.col(p.feature, oc.entry) for p in ps}
    rows = [{"date": ft.dates[i], "outcome": CODE_NAMES[out.code[i]], "entry_price": round(float(out.entry_px[i]), 4), "end_return_pct": round(float(out.end_ret[i]), 4), "mfe_pct": round(float(out.mfe[i]), 4),
             "mae_pct": round(float(out.mae[i]), 4), **{k: round(float(v[i]), 4) for k, v in cols.items()}} for i in np.flatnonzero(q)[-limit:]]
    a, b = out.end_ret[q], out.end_ret[base & ~ev["qualifying"]]
    hist = None
    if a.size and b.size:
        lo, hi = np.quantile(np.r_[a, b], [0.005, 0.995])
        if hi <= lo:
            hi = lo + 1e-6
        ha, edges = np.histogram(np.clip(a, lo, hi), bins=31, range=(lo, hi))
        hb, _ = np.histogram(np.clip(b, lo, hi), bins=31, range=(lo, hi))
        hist = {"x": ((edges[:-1] + edges[1:]) / 2).tolist(), "condition": (ha / max(ha.sum(), 1)).tolist(), "baseline": (hb / max(hb.sum(), 1)).tolist(), "condition_n": int(a.size), "baseline_n": int(b.size)}
    years: dict[str, list] = {}
    for r in rows:
        y = years.setdefault(r["date"][:4], [0, 0])
        y[0] += 1
        y[1] += r["outcome"] == "target"
    return {"condition": cond_text(ps, symbol.upper()), "outcome_text": oc.text(), "columns": ["date", "outcome", "entry_price", "end_return_pct", "mfe_pct", "mae_pct"] + list(cols), "rows": rows, "histogram": hist,
            "by_year": [{"year": y, "n": v[0], "hits": v[1]} for y, v in sorted(years.items())], "study": to_study(symbol.upper(), ps, oc, start, end) if oc.direction != "either" else None,
            "excluded": {k: int(v) for k, v in out.reasons.items()}}


# ----------------------------------------------------------------------------- option maps
OPTION_PRESETS = {
    "dte_vs_delta": {"title": "Days to expiration versus delta", "question": "How do return and drawdown change with the contract chosen?",
                     "x": {"path": "legs.*.dte", "values": [0, 1, 3, 7, 14, 30], "label": "Days to expiration"},
                     "y": {"path": "legs.*.strike", "method": "delta", "values": [0.2, 0.3, 0.4, 0.5, 0.6], "label": "Delta of the contract"}},
    "target_vs_stop": {"title": "Profit target versus stop loss", "question": "How sensitive is the result to the premium-based exits?",
                       "x": {"path": "exits.profit_target", "values": [25, 50, 100, 200], "label": "Profit target (% of premium)"},
                       "y": {"path": "exits.stop_loss", "values": [25, 50, 75, 100], "label": "Stop loss (% of premium)"}},
    "slippage_vs_delay": {"title": "Slippage versus order delay", "question": "How quickly does the edge disappear as execution gets worse?",
                          "x": {"path": "execution.slippage", "values": [0, 1, 2, 5, 10], "label": "Extra slippage against you (% of the price)"},
                          "y": {"path": "execution.latency_seconds", "values": [0, 60, 300, 900], "label": "Order delay (seconds)"}},
    "hold_vs_dte": {"title": "Holding period versus days to expiration", "question": "How does risk change with how long the position is held?",
                    "x": {"path": "exits.max_hold.days", "values": [1, 2, 3, 5], "label": "Sessions held"},
                    "y": {"path": "legs.*.dte", "values": [7, 14, 30], "label": "Days to expiration"}},
}
OPTION_LAYERS = {"expectancy": "Average profit per trade ($)", "net_pnl": "Net profit ($)", "max_drawdown": "Maximum drawdown", "win_rate": "Win rate", "trades": "Trades"}


def set_param(cfg: dict, dim: dict, v) -> dict:
    """Apply one map value to a backtest configuration (returns a modified deep copy)."""
    import copy
    c = copy.deepcopy(cfg)
    s = c["strategies"][0]
    path = dim["path"]
    quotes = (c.get("data") or {}).get("pricing") != "trades"
    if path == "legs.*.dte":
        for leg in s["legs"]:
            if leg.get("kind", "option") == "option":
                leg["dte"] = int(v)
                leg["dte_mode"] = "nearest"
                leg.pop("dte_min", None)
                leg.pop("dte_max", None)
    elif path == "legs.*.strike":
        for leg in s["legs"]:
            if leg.get("kind", "option") == "option" and (leg.get("strike") or {}).get("method") != "leg_offset":
                leg["strike"] = {"method": dim.get("method", "delta"), "value": float(v), "unit": "points"}
    elif path in ("exits.profit_target", "exits.stop_loss"):
        s.setdefault("exits", {})[path.split(".")[1]] = {"enabled": True, "mode": "pct", "value": float(v)}
    elif path == "exits.max_hold.days":
        s.setdefault("exits", {})["max_hold"] = {"enabled": True, "days": int(v), "minutes": None}
    elif path == "execution.slippage":
        ex = c.setdefault("execution", {})
        if quotes:
            ex["fill_model"], ex["custom_fraction"] = "custom", 1.0
            ex["slippage_pct_of_price"] = float(v)
        else:
            ex["trade_bar_slippage_pct"] = float(v)
    elif path.startswith("execution."):
        c.setdefault("execution", {})[path.split(".", 1)[1]] = v
    else:
        from strata.optimize.optimizer import set_path
        set_path(c, path, v)
    return c


def option_map(base: dict, spec: dict, data_dir: str, progress=None, should_cancel=None) -> dict:
    """Re-run a backtest configuration over a grid of two of its own parameters, on the option
    data already stored. Each cell is a complete backtest by the same engine."""
    from strata.optimize.optimizer import Evaluator
    xs, ys = list(spec["x"]["values"]), list(spec["y"]["values"])
    if len(xs) * len(ys) > 64:
        raise ValueError("an option map is limited to 64 backtests")
    cfgs, meta = [], []
    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            cfgs.append(set_param(set_param(base, spec["x"], xv), spec["y"], yv))
            meta.append((xi, yi, xv, yv))
    with Evaluator(None, data_dir, (lambda d, t, l: progress(d / max(t, 1), f"{d} of {t} backtests")) if progress else None, should_cancel) as ev:
        ev.total = len(cfgs)
        res = ev.run(cfgs, "sensitivity")
    cells = []
    for (xi, yi, xv, yv), r, cfg in zip(meta, res, cfgs):
        m = r.get("metrics") or {}
        n = int(m.get("trades") or 0)
        cells.append({"xi": xi, "yi": yi, "x": f"{xv:g}" if isinstance(xv, (int, float)) else str(xv), "y": f"{yv:g}" if isinstance(yv, (int, float)) else str(yv), "status": "ok" if r.get("ok") else "error",
                      "error": r.get("error"), "trades": n, "n": n, "thin": n < MIN_CELL, "config": cfg, **{k: m.get(k) for k in ("net_pnl", "expectancy", "win_rate", "max_drawdown", "profit_factor", "sharpe")}})
    ok = [c for c in cells if c["status"] == "ok"]
    grid = {(c["xi"], c["yi"]): c for c in ok}
    for c in ok:
        nb = [grid.get((c["xi"] + dx, c["yi"] + dy)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nb = [x for x in nb if x is not None and x["trades"] > 0 and x.get("expectancy") is not None]
        agree = sum(1 for x in nb if c.get("expectancy") is not None and (x["expectancy"] > 0) == (c["expectancy"] > 0))
        c["neighbours"], c["neighbours_agree"] = len(nb), agree
        c["stable"] = bool(c["trades"] >= MIN_CELL and len(nb) >= 2 and agree / len(nb) >= 0.75)
        c["significant"] = False  # a per-cell significance claim would need its own test on the trades; none is made here
    thin = sum(1 for c in ok if c["thin"])
    notes = [f"{len(cells)} backtests were run on the option data already stored; nothing was downloaded or modelled.",
             "Each cell is a separate configuration. Picking the best cell and reporting it alone is a search over " + str(len(cells)) + " configurations; they are all counted in the research ledger."]
    if thin:
        notes.append(f"{thin} of {len(ok)} cells have fewer than {MIN_CELL} trades. With samples this small the map shows what happened, not what to expect.")
    return {"kind": "sensitivity", "map": "options", "title": spec.get("title", ""), "question": spec.get("question", ""), "x": {**spec["x"], "labels": [f"{v:g}" for v in xs]}, "y": {**spec["y"], "labels": [f"{v:g}" for v in ys]},
            "cells": cells, "layer": "expectancy", "layers": OPTION_LAYERS, "summary": {"cells": len(cells), "thin": thin, "significant": 0, "stable": sum(1 for c in ok if c.get("stable"))}, "notes": notes}
