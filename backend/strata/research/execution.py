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
  missing prices    the order is cancelled after one minute without a usable price
  auctions          no entries in the first and last minutes; fills there are worse
  expiration day    fills in the final hour of a contract's last day are worse
  assignment        short legs are assigned earlier / stock is settled differently
  best / adverse    the most favourable assumptions that are still achievable, and several
                    unfavourable ones at once

`quality` is separate from all of that: an EXECUTION-QUALITY SCORE that says how much
observed evidence stands behind the simulated fills (were prices bid/ask quotes or prints,
how fresh, how liquid, how many signals could be executed at all). It is computed from
measurable facts about the data and the fills and has nothing to do with profitability:
a losing backtest can have excellent execution evidence and a winning one almost none.

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
    add("best", "Reference", "Best case that is still achievable", "Commissions as configured; every fill at the " + ("observed print with no slippage" if trades else "midpoint of the quote") + ". Achievable only with patient, lucky orders.",
        _set_exec(cfg, slippage_per_contract=0.0, trade_bar_slippage_pct=0.0, extra_slippage_pct=0.0, multi_leg_extra_pct=0.0, **({} if trades else {"fill_model": "midpoint"})), 0)
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
    if not eod:
        add("impatient", "Missing prices", "Cancelled after 1 minute without a price", "The order is cancelled if no usable " + ("trade print" if trades else "quote") + " appears within one minute of the signal (the run waited "
            + f"{float(ex.get('max_fill_delay_minutes', 5.0)):g} minutes).", _set_exec(cfg, max_fill_delay_minutes=1.0), 2)
        add("auction", "Auctions", "No entries in the first and last 5 minutes", "The opening rotation and the closing minutes have wide, one-sided markets: no entry fills there, and any exit that fills there is 5% worse.",
            _set_exec(cfg, no_entry_first_minutes=5, no_entry_last_minutes=5, auction_extra_pct=5.0), 2)
    else:
        out.append({"id": "auction", "group": "Auctions", "label": "Opening and closing auctions", "skipped": "The option data is one snapshot per session at the close, so every fill is already a closing-print fill and no other moment can be compared with it."})
    dtes = [int(l.get("dte") or 0) for l in _legs(cfg)]
    holds_to_expiry = any(d <= 1 for d in dtes) or True
    for x in (5, 10):
        add(f"expiry{x}", "Expiration day", f"Final-hour fills {x}% worse on a contract's last day", f"Any fill of a contract during the last hour of its own expiration day is {x}% worse (thin, fast markets near expiry). Fills at other times are unchanged.",
            _set_exec(cfg, expiry_day_extra_pct=float(x)), x / 2)
    del holds_to_expiry
    shorts = [l for l in _legs(cfg) if l.get("side") == "sell"]
    if shorts:
        add("assign_early", "Assignment", "Short legs assigned earlier", "A short in-the-money option is assigned once its remaining time value falls below $0.25 (the run used $" + f"{float(ex.get('assignment_extrinsic_threshold', 0.05)):.2f}).",
            _set_exec(cfg, early_assignment=True, assignment_extrinsic_threshold=0.25), 2)
        add("assign_never", "Assignment", "No early assignment", "Short options are never assigned before expiration.", _set_exec(cfg, early_assignment=False), 1)
        other = "intrinsic" if ex.get("physical_settlement", "close_next_open") == "close_next_open" else "close_next_open"
        add("settle", "Assignment", "Delivered stock settled " + ("at intrinsic value" if other == "intrinsic" else "at the next open"), "Stock delivered by exercise or assignment is " + ("valued at the expiration close with no overnight risk." if other == "intrinsic"
            else "sold at the next session's open, taking the overnight gap."), _set_exec(cfg, physical_settlement=other), 2)
    else:
        out.append({"id": "assign", "group": "Assignment", "label": "Exercise and assignment", "skipped": "The strategy holds no short option, so it cannot be assigned. Long options that finish in the money are exercised at expiration as simulated."})
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
            d = [tr[k] - b_tr[k] for k in common]
            pc = {"pairs": len(common), "mean_diff": sum(d) / len(d), "worse": sum(1 for x in d if x < 0), "better": sum(1 for x in d if x > 0)}
        rows.append({**row, "status": "ok", "trades": n, "signals": r.get("signals"), "missed": r.get("not_executed"), "missed_by_reason": r.get("by_reason"), "partial_fills": r.get("partial_fills"),
                     "net_pnl": m.get("net_pnl"), "expectancy": m.get("expectancy"), "win_rate": m.get("win_rate"), "profit_factor": m.get("profit_factor"), "max_drawdown": m.get("max_drawdown"),
                     "commissions": r.get("commissions"), "delta_net_pnl": (m.get("net_pnl") - bm.get("net_pnl")) if m.get("net_pnl") is not None and bm.get("net_pnl") is not None else None,
                     "lost_trades": len(set(b_tr) - set(tr)) if b_tr else None, "new_trades": len(set(tr) - set(b_tr)) if b_tr else None, "paired": pc, "thin": n < 30, "datasets": r.get("datasets")})
    ok = {r["id"]: r for r in rows if r["status"] == "ok"}
    sig, bas, adv, best = ok.get("signal"), ok.get("base"), ok.get("adverse"), ok.get("best")
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    slip = sorted([(0.0, bas["net_pnl"])] + [(float(r["id"][4:]), r["net_pnl"]) for r in ok.values() if r["id"].startswith("slip") and r["net_pnl"] is not None]) if bas and bas.get("net_pnl") is not None else []
    breakeven = None
    for (x0, y0), (x1, y1) in zip(slip, slip[1:]):
        if y0 > 0 >= y1:
            breakeven = x0 + (x1 - x0) * y0 / (y0 - y1)
            break
    summary = {"pricing": mode, "trades": bas["trades"] if bas else 0, "signal_net_pnl": sig["net_pnl"] if sig else None, "simulated_net_pnl": bas["net_pnl"] if bas else None, "adverse_net_pnl": adv["net_pnl"] if adv else None,
               "best_net_pnl": best["net_pnl"] if best else None,
               "cases": [{"case": "best", "label": "Best case that is still achievable", "net_pnl": best["net_pnl"] if best else None, "trades": best["trades"] if best else None, "expectancy": best["expectancy"] if best else None},
                         {"case": "base", "label": "As simulated", "net_pnl": bas["net_pnl"] if bas else None, "trades": bas["trades"] if bas else None, "expectancy": bas["expectancy"] if bas else None},
                         {"case": "adverse", "label": "Adverse combination", "net_pnl": adv["net_pnl"] if adv else None, "trades": adv["trades"] if adv else None, "expectancy": adv["expectancy"] if adv else None}],
               "execution_cost": (sig["net_pnl"] - bas["net_pnl"]) if sig and bas and sig["net_pnl"] is not None and bas["net_pnl"] is not None else None,
               "breakeven_slippage_pct": breakeven, "still_profitable": [r["label"] for r in ok.values() if r["id"] not in ("signal", "best") and (r["net_pnl"] or 0) > 0],
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
            "best_net_pnl": s.get("best_net_pnl"), "breakeven_slippage_pct": s["breakeven_slippage_pct"], "quality_score": (res.get("quality") or {}).get("score")}


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)


# ----------------------------------------------------------------------------- execution-quality score
def _lin(x: float, bad: float, good: float) -> float:
    """0 at `bad`, 100 at `good`, linear and clipped."""
    if good == bad:
        return 100.0
    return float(max(0.0, min(100.0, (x - bad) / (good - bad) * 100.0)))


def _median(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def quality(trades: list[dict], meta: dict, diagnostics: dict, cfg: dict, eod: bool = False) -> dict:
    """How much observed evidence stands behind the simulated fills, 0 to 100.

    Each component is a measured fact about the data or the fills; one that cannot be
    measured with this data (the spread of a trade print, for instance) is reported as
    not measurable and left out of the average - and the report says how much of the
    score that removes. Profit and loss never enter it."""
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    ex = cfg.get("execution") or {}
    legs = [l for t in trades for l in t.get("legs", []) if l.get("kind") == "option"]
    comps = []

    def comp(id_, label, weight, score, value, detail, measurable=True):
        comps.append({"id": id_, "label": label, "weight": weight, "score": None if not measurable or score is None else round(float(score), 1), "value": value, "detail": detail, "measurable": bool(measurable and score is not None)})

    # 1. what kind of price each fill used
    n = len(legs)
    src = {k: sum(1 for l in legs if l.get("price_source") == k) for k in ("quote", "trade_print", "model")}
    if n:
        base = {"quote": 70.0 if eod else 100.0, "trade_print": 40.0, "model": 0.0}
        sc = sum(base[k] * v for k, v in src.items()) / n
        what = ("modelled premiums (no observed option price at all)" if src["model"] == n else "observed bid/ask quotes" + (" at the close only (one snapshot per session)" if eod else "") if src["quote"] == n
                else "observed trade prints (the bid/ask at those moments is not observed)" if src["trade_print"] == n else "a mix of quotes and prints")
        comp("evidence", "Price evidence", 30, sc, f"{src['quote']} quote, {src['trade_print']} print, {src['model']} model", f"Fills were priced on {what}.")
    else:
        comp("evidence", "Price evidence", 30, None, "no fills", "There are no fills to assess.", False)
    # 2. how many signals could be executed at all
    exe = meta.get("executability") or {}
    sig, done = exe.get("signals"), exe.get("executed")
    if sig:
        comp("coverage", "Signals that could be executed", 20, 100.0 * done / sig, f"{done} of {sig}",
             f"{sig - done} signal{'s' if sig - done != 1 else ''} had no usable option price or size and never became a trade" + (": " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted((exe.get("not_executed_by_reason") or {}).items(), key=lambda kv: -kv[1])[:3]) if sig - done else "") + ".")
    else:
        comp("coverage", "Signals that could be executed", 20, None, "not recorded", "The run did not record its signals.", False)
    # 3. how old the price was when it was used
    ages = [t.get("meta", {}).get("entry_print_age_seconds") for t in trades]
    med_age = _median(ages)
    delays = _median([t.get("fill_delay_seconds") for t in trades])
    if mode == "trades" and med_age is not None:
        lim = float(ex.get("max_trade_age_seconds", 300))
        comp("freshness", "Freshness of the price used", 15, _lin(med_age, lim, 0.0), f"median print age {med_age:.0f} s", f"Half of the entries used a print older than {med_age:.0f} seconds (the limit was {lim:.0f}).")
    elif mode == "historical" and not eod and delays is not None:
        lim = float(ex.get("max_fill_delay_minutes", 5.0)) * 60
        comp("freshness", "Wait for a usable quote", 15, _lin(delays, lim, 0.0), f"median wait {delays:.0f} s", f"Half of the entries waited {delays:.0f} seconds or more between the signal and the quote they filled at.")
    else:
        comp("freshness", "Freshness of the price used", 15, None, "not measurable", "One closing snapshot per session: there is no later or earlier price to compare its age with." if eod else "The age of the price at the fill was not recorded for this data.", False)
    # 4. liquidity: order size against what traded, and how often the contract traded
    shares, mins = [], [t.get("meta", {}).get("contract_minutes_traded_pct") for t in trades]
    for t in trades:
        for l in t.get("legs", []):
            v = l.get("entry_volume")
            if l.get("kind") == "option" and v and l.get("contracts"):
                shares.append(abs(l["contracts"]) / v * 100)
    subs, bits = [], []
    ms = _median(shares)
    if ms is not None:
        subs.append(_lin(ms, 25.0, 1.0))
        bits.append(f"the median order was {ms:.1f}% of the contract's traded volume")
    mm = _median(mins)
    if mm is not None:
        subs.append(_lin(mm, 1.0, 50.0))
        bits.append(f"the median contract traded in {mm:.0f}% of the session's minutes")
    pf = sum(1 for t in trades if "partial_fill" in (t.get("flags") or []))
    if trades:
        subs.append(100.0 * (1 - pf / len(trades)))
        if pf:
            bits.append(f"{pf} entr{'ies were' if pf != 1 else 'y was'} filled only in part")
    if (ms is not None or mm is not None) and subs:
        comp("liquidity", "Liquidity of the contracts traded", 20, sum(subs) / len(subs), "; ".join(bits), ("Measured from the volume observed up to each fill: " + "; ".join(bits) + ".").replace("  ", " "))
    else:
        comp("liquidity", "Liquidity of the contracts traded", 20, None, "not measurable", "The stored data carries no volume for the contracts traded, so order size cannot be compared with it.", False)
    # 5. the quoted spread paid
    spreads = [(l["entry_ask"] - l["entry_bid"]) / ((l["entry_ask"] + l["entry_bid"]) / 2) * 100 for l in legs if l.get("entry_bid") is not None and l.get("entry_ask") is not None and (l["entry_ask"] + l["entry_bid"]) > 0]
    msp = _median(spreads)
    if msp is not None and mode == "historical":
        comp("spread", "Quoted spread at the entry", 10, _lin(msp, 20.0, 1.0), f"median {msp:.1f}% of the mid", f"Half of the entry fills crossed a quoted spread of {msp:.1f}% of the midpoint or more.")
    else:
        comp("spread", "Quoted spread at the entry", 10, None, "not measurable", "Trade prints carry no bid or ask, so the spread is unknown. It is not estimated." if mode == "trades" else "No bid/ask was recorded at the entries.", False)
    # 6. exits that could not be filled when the rule asked
    bad = {"deferred_exit", "closed_at_last_known_quote", "unfilled_at_end", "ambiguous_exit", "intrabar_ambiguous"}
    if trades:
        nb = sum(1 for t in trades if bad & set(t.get("flags") or []))
        comp("exits", "Exits filled when the rule asked", 5, 100.0 * (1 - nb / len(trades)), f"{len(trades) - nb} of {len(trades)}", f"{nb} trade{'s' if nb != 1 else ''} had an exit that was deferred, ambiguous or filled at the last known price.")
    else:
        comp("exits", "Exits filled when the rule asked", 5, None, "no trades", "There are no trades.", False)
    meas = [c for c in comps if c["measurable"]]
    w = sum(c["weight"] for c in meas)
    score = sum(c["weight"] * c["score"] for c in meas) / w if w else None
    grade = None if score is None else "strong" if score >= 80 else "reasonable" if score >= 60 else "weak" if score >= 40 else "very weak"
    notes = ["This score measures the evidence behind the fills (what kind of prices, how fresh, how liquid, how many signals could trade). It says nothing about whether the strategy made or lost money."]
    if w < 100:
        notes.append(f"{100 - w}% of the score's weight could not be measured with this data and is left out rather than assumed: " + "; ".join(c["label"].lower() for c in comps if not c["measurable"]) + ".")
    if len(trades) < 30:
        notes.append(f"Only {len(trades)} trades: the medians above describe a small number of fills.")
    return {"score": None if score is None else round(score, 1), "grade": grade, "measured_weight": w, "components": comps, "notes": notes, "trades": len(trades), "pricing": mode}
