"""Read models for the research workspace: the session calendar, the observations behind a
statistic, and the side-by-side comparison of runs. Everything is derived from stored
results; nothing here computes a new financial number except differences between runs.
"""
from __future__ import annotations

import math
from datetime import date

from strata.api.results import load_trades
from strata.core import calendar as cal

EXIT_CLASS = {"profit_target": "target", "underlying_target": "target", "leg_target": "target", "stop_loss": "stop", "trailing_stop": "stop", "underlying_stop": "stop", "leg_stop": "stop",
              "strike_breach": "stop", "greek_stop": "stop", "margin_call": "stop"}
LEGEND = [
    {"id": "target", "label": "Qualified - target reached"}, {"id": "stop", "label": "Qualified - stop reached"}, {"id": "win", "label": "Profitable trade"}, {"id": "loss", "label": "Losing trade"},
    {"id": "neither", "label": "Qualified - neither level"}, {"id": "ambiguous", "label": "Ambiguous order of events"}, {"id": "skipped", "label": "Qualified - not executed"},
    {"id": "missing", "label": "Missing data"}, {"id": "none", "label": "Did not qualify"},
]


def run_calendar(run: dict, res: dict) -> dict:
    """One status per session. Days are only listed when they are something other than
    "evaluated, did not qualify"; `start`/`end` bound the evaluated period."""
    days: dict[str, dict] = {}
    if run["kind"] == "study":
        a, b = res["period"]
        for r in res.get("sessions", []):
            st = r["outcome"] if r["outcome"] in ("target", "stop", "ambiguous") else "neither"
            days[r["date"]] = {"s": st, "v": r.get("end_return_pct"), "q": True}
        for reason, ds in (res.get("excluded_sessions") or {}).items():
            for d in ds:
                days.setdefault(d, {"s": "missing", "why": reason})
        base = set((res.get("baseline_sessions") or {}).get("dates") or [])
        value = "end return %"
    else:
        cov = res.get("coverage") or {}
        a = (cov.get("requested") or {}).get("start") or res["config"]["start"]
        b = (cov.get("requested") or {}).get("end") or res["config"]["end"]
        sig = set((cov.get("signals") or {}).get("dates") or [])
        for t in load_trades(run["id"]):
            d = t["entry_date"]
            cur = days.get(d)
            pnl = (cur["v"] if cur else 0.0) + t["pnl"]
            flags = set(cur.get("flags", [])) if cur else set()
            flags |= {f for f in t.get("flags", []) if f in ("ambiguous_exit", "intrabar_ambiguous", "deferred_exit", "stale_quotes", "trade_based_fill", "modeled_prices")}
            cls = EXIT_CLASS.get(t.get("exit_reason") or "")
            days[d] = {"s": "ambiguous" if flags & {"ambiguous_exit"} else ("win" if pnl > 0 else "loss"), "v": pnl, "q": True, "n": (cur.get("n", 0) if cur else 0) + 1, "exit": cls or (cur or {}).get("exit"),
                       "flags": sorted(flags), "trade": (cur or {}).get("trade") or t["id"]}
        for r in ((cov.get("skipped_signals") or {}).get("rows") or []):
            if r["date"] not in days:
                days[r["date"]] = {"s": "skipped", "q": True, "why": r["reason"], "class": r.get("class")}
        for d in sig:
            days.setdefault(d, {"s": "skipped", "q": True, "why": "qualified; not simulated"})
        for reason, ds in ((res.get("diagnostics") or {}).get("excluded_sessions") or {}).items():
            for d in ds:
                days.setdefault(d, {"s": "missing", "why": reason})
        base = None
        value = "P&L $"
    counts: dict[str, int] = {}
    for v in days.values():
        counts[v["s"]] = counts.get(v["s"], 0) + 1
    tdays = cal.trading_days(date.fromisoformat(a), date.fromisoformat(b))
    total = len(tdays)
    # weekdays on which the market was closed, so the calendar does not show them as sessions
    from datetime import timedelta
    tset, closed, x = set(tdays), [], date.fromisoformat(a)
    while x <= date.fromisoformat(b):
        if x.weekday() < 5 and x not in tset:
            closed.append(x.isoformat())
        x += timedelta(days=1)
    counts["none"] = max(0, (len(base) if base is not None else total) - sum(1 for v in days.values() if v.get("q")))
    return {"start": a, "end": b, "days": days, "counts": counts, "legend": [x for x in LEGEND if counts.get(x["id"])], "value": value, "sessions": total, "kind": run["kind"], "closed": closed,
            "years": sorted({d[:4] for d in days} | {a[:4], b[:4]})}


def observations(run: dict, res: dict, which: str = "all", limit: int = 5000) -> dict:
    """The observations a statistic was computed from.

    study:     qualifying | target | stop | neither | ambiguous | baseline | baseline_target | excluded
    backtest:  all | win | loss | target | stop | time | skipped | excluded"""
    if run["kind"] == "study":
        rows = res.get("sessions", [])
        if which in ("target", "stop", "neither", "ambiguous"):
            rows = [r for r in rows if r["outcome"] == which]
            title = f"Qualifying sessions with outcome '{which}'"
        elif which in ("baseline", "baseline_target"):
            b = res.get("baseline_sessions") or {"dates": [], "outcomes": "", "end_returns": []}
            names = {"t": "target", "s": "stop", "n": "neither", "a": "ambiguous"}
            ends = b.get("end_returns") or [None] * len(b["dates"])
            rows = [{"date": d, "outcome": names[o], "end_return_pct": e} for d, o, e in zip(b["dates"], b["outcomes"], ends) if which == "baseline" or o == "t"]
            title = "Every evaluated session (the baseline)" if which == "baseline" else "Baseline sessions where the outcome occurred"
        elif which == "excluded":
            rows = [{"date": d, "reason": k} for k, ds in (res.get("excluded_sessions") or {}).items() for d in ds]
            title = "Sessions excluded because they could not be evaluated"
        else:
            title = "Sessions that met the condition"
        cols = [c for c in ("date", "outcome", "entry_minute", "entry_price", "target_level", "stop_level", "minutes_to_target", "mfe_pct", "mae_pct", "end_return_pct", "prev_return_pct", "gap_pct", "vix_prev", "resolution", "reason") if rows and c in rows[0]]
    else:
        trades = load_trades(run["id"])
        cov = res.get("coverage") or {}
        if which in ("skipped", "excluded"):
            if which == "skipped":
                rows = (cov.get("skipped_signals") or {}).get("rows") or []
                title = "Qualifying sessions that did not become a trade"
            else:
                rows = [{"date": d, "reason": k} for k, ds in ((res.get("diagnostics") or {}).get("excluded_sessions") or {}).items() for d in ds]
                title = "Sessions excluded because a rule could not be evaluated"
            cols = [c for c in ("date", "minute", "price", "kind", "class", "reason") if rows and c in rows[0]]
        else:
            sel = {"win": lambda t: t["pnl"] > 0, "loss": lambda t: t["pnl"] <= 0, "target": lambda t: EXIT_CLASS.get(t.get("exit_reason") or "") == "target",
                   "stop": lambda t: EXIT_CLASS.get(t.get("exit_reason") or "") == "stop", "time": lambda t: (t.get("exit_reason") or "") in ("time_exit", "dte_exit", "max_hold", "expiration")}.get(which, lambda t: True)
            rows = [{"id": t["id"], "date": t["entry_date"], "entry_time": t["entry_time"][11:16], "exit_time": (t.get("exit_time") or "")[:16].replace("T", " "), "contract": _contract(t), "entry_price": t["legs"][0].get("entry_price"),
                     "exit_price": t["legs"][0].get("exit_price"), "price_source": t["legs"][0].get("price_source"), "pnl": t["pnl"], "exit_reason": t.get("exit_reason"), "entry_spot": t.get("entry_spot"), "exit_spot": t.get("exit_spot"),
                     "fill_delay_s": t.get("fill_delay_seconds"), "lot": t.get("lot"), "flags": ",".join(t.get("flags") or [])} for t in trades if sel(t)]
            title = {"win": "Winning trades", "loss": "Losing trades", "target": "Trades closed at a target", "stop": "Trades closed at a stop", "time": "Trades closed by time"}.get(which, "Every trade")
            cols = ["id", "date", "entry_time", "exit_time", "contract", "price_source", "entry_price", "exit_price", "pnl", "exit_reason", "entry_spot", "exit_spot", "fill_delay_s", "lot", "flags"]
    return {"which": which, "title": title, "columns": cols, "rows": rows[-limit:], "count": len(rows), "truncated": len(rows) > limit}


def _contract(t: dict) -> str:
    out = []
    for l in t["legs"]:
        if l["kind"] == "stock":
            out.append(f"{'+' if l['side'] == 'buy' else '-'}stock")
        else:
            out.append(f"{'+' if l['side'] == 'buy' else '-'}{l['strike']:g}{l['right']} {l['expiration']}")
    return " / ".join(out)


# ----------------------------------------------------------------------------- comparison workspace
def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def compare_runs(entries: list[dict]) -> dict:
    """Side-by-side view of finished runs: [{label, run, result}]. Curves are returned on
    their own dates (the client aligns them); paired rows match observations by date."""
    from strata.ai.agent import compatibility
    from strata.ai.explain import compare_table
    kinds = {e["run"]["kind"] for e in entries}
    comp = compatibility([{**e, "mode": (e["run"].get("summary") or {}).get("mode")} for e in entries])
    out: dict = {"runs": [], "compatibility": comp, "mixed_kinds": len(kinds) > 1}
    if len(kinds) > 1:
        comp["warning"] = "A market study and an options backtest measure different things (how the underlying moved versus what an option position earned). They are shown together but are not comparable."
        comp["compatible"] = False
    else:
        out["table"] = compare_table(entries)
    for e in entries:
        run, res = e["run"], e["result"]
        item = {"id": run["id"], "label": e["label"], "kind": run["kind"], "name": run.get("name"), "created_at": run.get("created_at")}
        if run["kind"] == "study":
            st = res["stats"]
            item.update({"period": res["period"], "pricing": "underlying prices (study)", "metrics": {k: _num(st.get(k)) for k in ("qualifying_sessions", "probability", "baseline_probability", "lift", "avg_end_return_pct", "p_value_vs_rest")},
                         "ci95": st.get("probability_ci95"), "by_year": res.get("by_year"), "curve": _cum([(r["date"], r["end_return_pct"]) for r in res.get("sessions", [])]), "curve_label": "cumulative end return of qualifying sessions (%)",
                         "obs": {r["date"]: r["end_return_pct"] for r in res.get("sessions", [])}})
        else:
            m = res.get("metrics") or {}
            ser = res.get("series") or {}
            dates, eq, dd = ser.get("dates") or [], ser.get("equity") or [], ser.get("drawdown") or []
            trades = load_trades(run["id"])
            cap = (res["config"].get("capital") or {}).get("initial_capital", 100000)
            item.update({"period": [res["config"]["start"], res["config"]["end"]], "pricing": (res.get("meta") or {}).get("pricing"),
                         "metrics": {k: _num(m.get(k)) for k in ("trades", "net_pnl", "win_rate", "expectancy", "profit_factor", "max_drawdown", "sharpe", "total_return")},
                         "curve": [[d, (v - cap) if v is not None else None] for d, v in zip(dates, eq)], "curve_label": "net profit ($)", "drawdown": [[d, v] for d, v in zip(dates, dd)],
                         "obs": _sum_by_date(trades), "execution": {k: (res["config"].get("execution") or {}).get(k) for k in ("fill_model", "slippage_per_contract", "commission_per_contract", "trade_bar_slippage_pct", "latency_seconds")},
                         "coverage": {"qualifying": ((res.get("coverage") or {}).get("signals") or {}).get("qualifying"), "executable": ((res.get("coverage") or {}).get("executable") or {}).get("signals")}})
        out["runs"].append(item)
    # paired observations: dates present in every run
    if len(entries) >= 2 and len(kinds) == 1:
        common = set(out["runs"][0]["obs"])
        for r in out["runs"][1:]:
            common &= set(r["obs"])
        dates = sorted(common)
        pairs = [{"date": d, "values": [r["obs"][d] for r in out["runs"]]} for d in dates]
        diffs = [p["values"][1] - p["values"][0] for p in pairs] if len(out["runs"]) == 2 else []
        from strata.research.stats import mean_test
        out["paired"] = {"dates_in_common": len(dates), "rows": pairs[-600:], "unit": "end return %" if "study" in kinds else "P&L $",
                         "only_in": [{"label": r["label"], "count": len(set(r["obs"]) - common)} for r in out["runs"]],
                         "difference": ({**mean_test(diffs), "label": f"{out['runs'][1]['label']} minus {out['runs'][0]['label']} on the same dates"} if diffs else None)}
    for r in out["runs"]:
        r.pop("obs", None)
    return out


def _cum(pairs: list[tuple]) -> list[list]:
    out, tot = [], 0.0
    for d, v in sorted(pairs):
        tot += v or 0.0
        out.append([d, round(tot, 4)])
    return out


def _sum_by_date(trades: list[dict]) -> dict:
    out: dict[str, float] = {}
    for t in trades:
        out[t["entry_date"]] = out.get(t["entry_date"], 0.0) + t["pnl"]
    return out
