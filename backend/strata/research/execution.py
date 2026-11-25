"""Execution uncertainty: how an option backtest changes as fills get worse.

A historical trade print shows that SOMEONE traded at that price. It does not show that a
simulated order would have been filled there, at that size, at that moment. Bid/ask quotes
are better evidence, and even they say nothing about delay or queue position.

`run_stress` re-runs the SAME configuration through the SAME engine under an explicit
ladder of execution assumptions and reports each one next to the run as it was simulated:

  signal only       no fees, no spread, no slippage - what the signal was worth on paper
  as simulated      the run's own assumptions
  slippage          every option fill 1 / 2 / 5 / 10 % worse
  order delay       the order reaches the market 30 s / 1 min / 5 min after the signal
  price age         only quotes / prints newer than a limit may be used
  minimum volume    the contract must have traded at least N contracts before the fill
  participation     the order may take at most 10% of that volume (partial or missed fills)
  multi-leg         each leg of a spread fills separately, a further x% worse
  bid/ask           the same signals priced on stored bid/ask quotes instead of prints
  adverse           several of the above at once

Nothing is invented: where only trade prints exist the spread is reported as UNKNOWN, and
the slippage rows are labelled as assumptions about it rather than as measurements.
Every scenario decides entries, exits and fills again; it is not a repricing of the
original trades.
"""
from __future__ import annotations

import copy
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from strata.research import robust as R


def _set_exec(cfg: dict, **kw) -> dict:
    c = copy.deepcopy(cfg)
    c.setdefault("execution", {}).update(kw)
    return c


def _legs(cfg: dict) -> list[dict]:
    return [l for s in cfg.get("strategies", []) for l in s.get("legs", []) if l.get("kind", "option") == "option"]


def build_scenarios(cfg: dict, quotes_available: bool = False, eod: bool = False) -> list[dict]:
    """The ladder for one configuration. Each scenario: {id, group, label, assumption, config}."""
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    trades = mode == "trades"
    ex = cfg.get("execution") or {}
    n_legs = max((len([l for l in s.get("legs", []) if l.get("kind", "option") == "option"]) for s in cfg.get("strategies", [])), default=1)
    out = []

    def add(sid, group, label, assumption, c, severity=1):
        out.append({"id": sid, "group": group, "label": label, "assumption": assumption, "config": c, "severity": severity})

    add("signal", "Reference", "Signal only (no costs)", "No commissions or fees; fills at the " + ("observed print" if trades else "midpoint of the quote") + " with no slippage. Not achievable - it shows what the signal was worth before execution.",
        _set_exec(cfg, commission_per_contract=0.0, fee_per_contract=0.0, commission_per_order=0.0, slippage_per_contract=0.0, trade_bar_slippage_pct=0.0, extra_slippage_pct=0.0, multi_leg_extra_pct=0.0,
                  **({} if trades else {"fill_model": "midpoint"})), 0)
    add("base", "Reference", "As simulated", "The run's own execution assumptions.", copy.deepcopy(cfg), 0)
    if not trades and ex.get("fill_model", "conservative") != "conservative":
        add("bidask", "Fill price", "Buy at the ask, sell at the bid", "Every fill crosses the full quoted spread.", _set_exec(cfg, fill_model="conservative"))
    for x in (1, 2, 5, 10):
        add(f"slip{x}", "Slippage", f"Every fill {x}% worse", f"Each option fill is {x}% worse than the {'print' if trades else 'quote'} it was priced from" + (" (an assumption about the unobserved spread)." if trades else "."),
            _set_exec(cfg, extra_slippage_pct=float(ex.get("extra_slippage_pct") or 0) + x), x)
    if eod:
        out.append({"id": "delay", "group": "Order delay", "label": "Order delay", "skipped": "The option data is one snapshot per session at the close, so a delayed order has no later price that day to fill at. Delay cannot be tested on end-of-day data."})
    else:
        for sec, lbl in ((30, "30 seconds"), (60, "1 minute"), (300, "5 minutes")):
            add(f"delay{sec}", "Order delay", f"Order arrives {lbl} late", f"The order reaches the market {lbl} after the signal and fills at the price observed then (or not at all).",
                _set_exec(cfg, latency_seconds=float(sec), max_fill_delay_minutes=max(float(ex.get("max_fill_delay_minutes", 5.0)), sec / 60 + 5)), sec / 30)
        if trades:
            for sec in (120, 60):
                if sec < int(ex.get("max_trade_age_seconds", 300)):
                    add(f"age{sec}", "Print age", f"Prints no older than {sec // 60} min" if sec >= 60 else f"Prints no older than {sec} s",
                        f"A fill may only use a trade print from the last {sec} seconds (the run allowed {int(ex.get('max_trade_age_seconds', 300))}); otherwise the order is missed.", _set_exec(cfg, max_trade_age_seconds=sec), 2)
        elif int(ex.get("max_quote_age_seconds", 120)) > 30:
            add("age30", "Quote age", "Quotes no older than 30 s", "A fill may only use a quote from the last 30 seconds.", _set_exec(cfg, max_quote_age_seconds=30), 2)
    for v in (10, 50, 100):
        c = copy.deepcopy(cfg)
        for l in _legs(c):
            l["min_volume"] = max(int(l.get("min_volume") or 0), v)
        add(f"vol{v}", "Minimum volume", f"At least {v} contracts traded", f"Each contract must have traded at least {v} contracts {'in the session before the fill' if trades else 'that day'}; otherwise the entry is missed.", c, v / 25)
    add("part10", "Participation", "At most 10% of volume, partial fills", "The order may take at most 10% of the contract's traded volume; a smaller size is accepted.", _set_exec(cfg, max_volume_pct=10.0, allow_partial_fills=True), 2)
    add("part10x", "Participation", "At most 10% of volume, all or nothing", "The order may take at most 10% of the contract's traded volume; if the full size does not fit, the entry is missed.",
        _set_exec(cfg, max_volume_pct=10.0, allow_partial_fills=False), 3)
    if n_legs >= 2:
        for x in (2, 5):
            add(f"legs{x}", "Multi-leg", f"Legs fill separately, {x}% worse each", f"The {n_legs} legs do not fill as one order: each leg is {x}% worse." + (" With trade prints the legs' prices were observed at different moments, so the combined price was never quoted." if trades else ""),
                _set_exec(cfg, multi_leg_extra_pct=float(x)), x)
    if trades:
        if quotes_available:
            c = copy.deepcopy(cfg)
            c["pricing"] = {**(c.get("pricing") or {}), "mode": "historical"}
            c["data"] = {"options_provider": None}
            add("quotes", "Bid/ask", "Priced on bid/ask quotes", "The same rules priced on the stored bid/ask quotes (buy at the ask, sell at the bid) instead of trade prints.", _set_exec(c, fill_model="conservative"), 2)
        else:
            out.append({"id": "quotes", "group": "Bid/ask", "label": "Priced on bid/ask quotes", "skipped": "No bid/ask quotes are stored for these sessions, so the spread is unknown. It is not estimated: the slippage rows show what different spreads would have cost."})
    adverse = _set_exec(cfg, extra_slippage_pct=float(ex.get("extra_slippage_pct") or 0) + 5, max_volume_pct=10.0, allow_partial_fills=False, **({"multi_leg_extra_pct": 2.0} if n_legs >= 2 else {}),
                        **({} if eod else {"latency_seconds": 60.0, "max_fill_delay_minutes": max(float(ex.get("max_fill_delay_minutes", 5.0)), 6.0)}))
    for l in _legs(adverse):
        l["min_volume"] = max(int(l.get("min_volume") or 0), 10)
    add("adverse", "Combined", "Adverse combination", "5% worse fills" + ("" if eod else ", a 1-minute delay") + ", at least 10 contracts traded, at most 10% of volume all-or-nothing" + (", legs 2% worse each" if n_legs >= 2 else "") + ".", adverse, 9)
    return out


def _worker_init(data_dir: str):
    from strata.config import override_settings
    for k in ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[k] = "1"
    override_settings(data_dir=data_dir)


def evaluate_scenario(cfg_dict: dict) -> dict:
    """One full backtest; returns what the comparison needs (metrics, fills, per-trade P&L)."""
    from strata.analytics.metrics import compute_metrics
    from strata.engine.engine import run_backtest
    from strata.engine.models import BacktestConfig
    t0 = time.time()
    try:
        cfg = BacktestConfig.model_validate(cfg_dict)
        res = run_backtest(cfg, record_paths=False)
        m = compute_metrics(res.trades, res.daily, cfg.capital.initial_capital, cfg.capital.risk_free_rate)
        exe = res.meta.get("executability") or {}
        tr = [{"key": f"{t['signal_ts']}", "date": t["entry_date"], "pnl": t["pnl"], "units": t["units"], "commissions": t["commissions"], "slippage": t["slippage"], "delay": t.get("fill_delay_seconds") or 0} for t in res.trades]
        return {"ok": True, "metrics": {k: m.get(k) for k in ("net_pnl", "trades", "win_rate", "expectancy", "profit_factor", "max_drawdown", "sharpe")}, "signals": exe.get("signals"), "executed": exe.get("executed"),
                "not_executed": exe.get("not_executed"), "by_reason": exe.get("not_executed_by_reason") or {}, "partial_fills": (res.diagnostics or {}).get("partial_fills", 0), "trades": tr,
                "commissions": float(sum(t["commissions"] for t in res.trades)), "contracts": float(sum(abs(l["contracts"] or 0) for t in res.trades for l in t["legs"] if l["kind"] == "option")),
                "datasets": res.meta.get("datasets_used"), "runtime": time.time() - t0}
    except Exception as e:  # noqa: BLE001  (one scenario failing must not hide the others)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "runtime": time.time() - t0}


def run_stress(cfg: dict, data_dir: str, quotes_available: bool = False, eod: bool = False, progress=None, should_cancel=None, workers: int | None = None, only: list[str] | None = None) -> dict:
    scen = build_scenarios(cfg, quotes_available, eod)
    todo = [s for s in scen if "config" in s and (only is None or s["id"] in only or s["id"] in ("signal", "base"))]
    results: dict[str, dict] = {}
    workers = max(1, min(workers or 4, os.cpu_count() or 1, len(todo)))
    if workers == 1:
        for i, s in enumerate(todo):
            if should_cancel and should_cancel():
                from strata.engine.engine import BacktestCancelled
                raise BacktestCancelled()
            if progress:
                progress(i / len(todo), f"{s['label']} ({i + 1} of {len(todo)})")
            results[s["id"]] = evaluate_scenario(s["config"])
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_worker_init, initargs=(data_dir,)) as pool:
            futs = {pool.submit(evaluate_scenario, s["config"]): s for s in todo}
            done = 0
            for f in as_completed(futs):
                if should_cancel and should_cancel():
                    pool.shutdown(wait=False, cancel_futures=True)
                    from strata.engine.engine import BacktestCancelled
                    raise BacktestCancelled()
                s = futs[f]
                try:
                    results[s["id"]] = f.result()
                except Exception as e:  # noqa: BLE001
                    results[s["id"]] = {"ok": False, "error": str(e)[:300]}
                done += 1
                if progress:
                    progress(done / len(todo), f"{done} of {len(todo)} execution scenarios")
    return assemble(cfg, scen, results, eod)


def assemble(cfg: dict, scen: list[dict], results: dict[str, dict], eod: bool = False) -> dict:
    base = results.get("base") or {}
    bm = base.get("metrics") or {}
    b_tr = {t["key"]: t["pnl"] for t in base.get("trades", [])} if base.get("ok") else {}
    rows = []
    for s in scen:
        row = {k: s[k] for k in ("id", "group", "label") if k in s} | {"assumption": s.get("assumption"), "severity": s.get("severity")}
        if "skipped" in s:
            rows.append({**row, "status": "not_testable", "note": s["skipped"]})
            continue
        r = results.get(s["id"])
        if r is None:
            continue
        if not r.get("ok"):
            rows.append({**row, "status": "error", "note": r.get("error")})
            continue
        m = r["metrics"]
        n = int(m.get("trades") or 0)
        tr = {t["key"]: t["pnl"] for t in r["trades"]}
        common = sorted(set(tr) & set(b_tr))
        pc = None
        if s["id"] not in ("base",) and len(common) >= 3:
            p = R.paired_compare({k: tr[k] for k in common}, {k: b_tr[k] for k in common}, by="session") if False else None
            d = [tr[k] - b_tr[k] for k in common]
            pc = {"pairs": len(common), "mean_diff": sum(d) / len(d), "worse": sum(1 for x in d if x < 0), "better": sum(1 for x in d if x > 0)}
            del p
        rows.append({**row, "status": "ok", "trades": n, "signals": r.get("signals"), "missed": r.get("not_executed"), "missed_by_reason": r.get("by_reason"), "partial_fills": r.get("partial_fills"),
                     "net_pnl": m.get("net_pnl"), "expectancy": m.get("expectancy"), "win_rate": m.get("win_rate"), "profit_factor": m.get("profit_factor"), "max_drawdown": m.get("max_drawdown"),
                     "commissions": r.get("commissions"), "delta_net_pnl": (m.get("net_pnl") - bm.get("net_pnl")) if m.get("net_pnl") is not None and bm.get("net_pnl") is not None else None,
                     "lost_trades": len(set(b_tr) - set(tr)) if b_tr else None, "new_trades": len(set(tr) - set(b_tr)) if b_tr else None, "paired": pc, "thin": n < 30, "datasets": r.get("datasets")})
    ok = {r["id"]: r for r in rows if r["status"] == "ok"}
    sig, bas, adv = ok.get("signal"), ok.get("base"), ok.get("adverse")
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    slip = sorted([(0.0, bas["net_pnl"])] + [(float(r["id"][4:]), r["net_pnl"]) for r in ok.values() if r["id"].startswith("slip") and r["net_pnl"] is not None]) if bas and bas.get("net_pnl") is not None else []
    breakeven = None
    for (x0, y0), (x1, y1) in zip(slip, slip[1:]):
        if y0 > 0 >= y1:
            breakeven = x0 + (x1 - x0) * y0 / (y0 - y1)
            break
    summary = {"pricing": mode, "trades": bas["trades"] if bas else 0, "signal_net_pnl": sig["net_pnl"] if sig else None, "simulated_net_pnl": bas["net_pnl"] if bas else None, "adverse_net_pnl": adv["net_pnl"] if adv else None,
               "execution_cost": (sig["net_pnl"] - bas["net_pnl"]) if sig and bas and sig["net_pnl"] is not None and bas["net_pnl"] is not None else None,
               "breakeven_slippage_pct": breakeven, "still_profitable": [r["label"] for r in ok.values() if r["id"] not in ("signal",) and (r["net_pnl"] or 0) > 0],
               "turns_negative": [r["label"] for r in ok.values() if bas and (bas["net_pnl"] or 0) > 0 and (r["net_pnl"] or 0) <= 0], "thin": bool(bas and bas["trades"] < 30)}
    notes = []
    if mode == "trades":
        notes.append("Priced on trade prints: the bid/ask spread at these moments is NOT observed. \"As simulated\" assumes the order would have traded at the last print; the slippage rows show the result if it traded worse. None of them is a measurement of the spread.")
    elif mode == "historical":
        notes.append("Priced on observed bid/ask quotes" + (" at the close (one snapshot per session)" if eod else "") + ": buying at the ask and selling at the bid is already the conservative case; the slippage rows go beyond the quoted spread.")
    else:
        notes.append("Priced on MODELLED premiums: every row here inherits that assumption, so none of them is evidence about real fills.")
    if summary["thin"]:
        notes.append(f"Only {summary['trades']} trades: differences between rows are descriptions of these trades, not estimates of future execution costs.")
    if sig and bas and summary["execution_cost"] is not None and sig["net_pnl"]:
        share = summary["execution_cost"] / abs(sig["net_pnl"]) * 100 if sig["net_pnl"] else None
        notes.append(f"With no costs the signal shows {sig['net_pnl']:+,.0f}; as simulated the result is {bas['net_pnl']:+,.0f}. Spread, fees and slippage account for {summary['execution_cost']:,.0f}"
                     + (f", {share:.0f}% of what the signal showed." if share is not None and sig["net_pnl"] > 0 and share <= 100 else ", more than the signal showed." if sig["net_pnl"] > 0 else "."))
    if breakeven is not None:
        notes.append(f"The result reaches zero when every fill is about {breakeven:.1f}% worse than simulated.")
    notes.append("Each row is a complete re-simulation (entries, exits and fills decided again), not a repricing of the original trades.")
    return {"kind": "execution_stress", "rows": rows, "summary": summary, "notes": notes, "config_hash": _hash(cfg)}


def _hash(cfg: dict) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def context_for(run: dict, result: dict, store=None) -> dict:
    """What a stress test of a finished run needs: its config, whether its option data is
    end-of-day, and whether bid/ask quotes exist for the sessions it traded."""
    from strata.data.store import MarketStore
    store = store or MarketStore()
    cfg = result.get("config")
    if not cfg:
        raise ValueError("this run has no option backtest to stress (market studies have no fills)")
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    sym = cfg["strategies"][0]["symbol"]
    eod = False
    quotes = False
    if mode == "historical":
        ds = store.option_dataset(sym, "quote", (cfg.get("data") or {}).get("options_provider"))
        eod = ds is not None and str(ds.res).lower() in ("1d", "eod", "daily")
    elif mode == "trades":
        from strata.api.results import load_trades
        try:
            days = {t["entry_date"] for t in load_trades(run["id"])}
        except Exception:  # noqa: BLE001
            days = set()
        qd = {d.isoformat() for d in store.option_days(sym, "quote")}
        quotes = bool(days) and days <= qd
    return {"config": cfg, "eod": eod, "quotes_available": quotes, "mode": mode, "symbol": sym}


def save_stress(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "execution.json").write_text(dumps(res))
    s = res["summary"]
    return {"execution_stress": True, "scenarios": len(res["rows"]), "trades": s["trades"], "signal_net_pnl": s["signal_net_pnl"], "simulated_net_pnl": s["simulated_net_pnl"], "adverse_net_pnl": s["adverse_net_pnl"],
            "breakeven_slippage_pct": s["breakeven_slippage_pct"]}


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)
