"""A bounded experiment: one strategy over a grid of settings, on stored observations.

What it is for: seeing how a result depends on the settings that were chosen, and where it does not depend on them much.
What it is not for: finding the best cell. With enough cells one of them is always good, so the report is built around the
opposite question: which REGION of the grid gives a similar answer whichever cell you stand in, and how much of what the
earlier part of the history shows is still there in the later part.

How it runs:

  one scan per entry rule   configurations that share their entry rule (they differ in expiration, strike or exits) share
                            ONE signal scan. Each is then simulated only on the sessions where that rule fired
  stored data only          nothing is downloaded while a grid runs. Each configuration reports how many of its signals the
                            stored observations could price, and why the others could not be (never counted as losses)
  the same split for all    the first 70% of the sessions are "earlier", the last 30% "later". Every configuration is run once
                            over the whole period and its trades are divided by date, so the later part was not used to
                            choose anything inside the run. It IS still history that has been looked at: the report says so
  every cell is a trial     the number of configurations is recorded, and the best cell's evidence is discounted for it

Results are trade prints unless the run says otherwise: the spread is not in them.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timezone

import numpy as np

from strata.core import calendar as cal
from strata.data.store import MarketStore

OOS_FRACTION = 0.30
SLIPPAGE_STEPS = (2.0, 5.0)  # % of each print, against the order: the first-order cost of worse fills
BOOT = 2000


# ----------------------------------------------------------------------------- building the grid
def window_for(store: MarketStore, sym: str, provider: str | None, start: date | None, end: date | None) -> tuple[date, date, str] | None:
    """The sessions an experiment on `sym` runs over: where option trade data is stored, cut to the period asked for."""
    ds = store.option_dataset(sym, "trade", provider)
    if ds is None:
        return None
    idx = store.option_bar_index(sym, dataset=ds)
    if idx.is_empty():
        return None
    a, b = idx["date"].min(), idx["date"].max()
    a, b = max(a, start) if start else a, min(b, end) if end else b
    return (a, b, ds.provider) if a <= b else None


def build(plan: dict, store: MarketStore) -> dict:
    """The plan of `strata.ai.nl_experiment` as a JSON-safe payload: every configuration compiled to an engine configuration."""
    from strata.ai import nl_experiment as NX
    from strata.ai.draft import compile_draft, execution_overrides
    base = plan["draft"]
    rows, errors = [], []
    for i, c in enumerate(NX.configurations(plan)):
        d = c["draft"]
        comp = compile_draft(d, store)
        if comp["errors"] or not comp.get("strategy"):
            errors.append({"label": c["label"], "error": "; ".join(comp["errors"]) or "the configuration does not compile"})
            continue
        win = window_for(store, d.symbol, d.provider, d.start, d.end)
        if win is None:
            errors.append({"label": c["label"], "error": f"no option trade data is stored for {d.symbol}"})
            continue
        cfg = {"strategies": [comp["strategy"]], "start": win[0].isoformat(), "end": win[1].isoformat(), "pricing": {"mode": "trades"}, "data": {"options_provider": win[2]},
               "execution": {"signal_interval_minutes": 1, **execution_overrides(d)}, "label": c["label"]}
        rows.append({"i": i, "key": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c["key"].items()}, "label": c["label"], "config": cfg, "symbol": d.symbol,
                     "assumptions": [a["field"] for a in comp["assumptions"]][:8]})
    return {"title": (base.title or "Experiment")[:120], "base": base.model_dump(mode="json"), "dims": [{**d, "values": [list(v) if isinstance(v, tuple) else v for v in d["values"]]} for d in plan["dims"]],
            "notes": list(plan.get("notes") or []), "configs": rows, "errors": errors, "mode": "trades", "oos_fraction": OOS_FRACTION}


def entry_signature(cfg: dict) -> str:
    """Configurations with the same signature fire on the same ticks: the symbol, the entry rule and the clock."""
    s = cfg["strategies"][0]
    # what decides WHEN the rule fires. How many positions may be open at once does not: the scan does not simulate positions
    e = {k: v for k, v in s["entry"].items() if k not in ("adds", "min_credit", "max_debit", "max_open_positions")}
    return hashlib.sha256(json.dumps({"symbol": s["symbol"], "entry": e, "interval": cfg["execution"].get("signal_interval_minutes"), "start": cfg["start"], "end": cfg["end"]}, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- one configuration (runs in a worker process)
def _trade_row(t: dict) -> dict:
    l = t["legs"][0]
    return {"date": t["entry_date"], "entry_time": t["entry_time"][11:19], "exit_time": (t.get("exit_time") or "")[11:19], "exit_date": (t.get("exit_time") or "")[:10], "pnl": round(float(t["pnl"]), 2),
            "gross": round(float(t.get("gross_pnl") or t["pnl"]), 2), "reason": t.get("exit_reason"), "strike": l.get("strike"), "expiration": l.get("expiration"), "right": l.get("right"),
            "entry": l.get("entry_price"), "exit": l.get("exit_price"), "legs": len(t["legs"]),
            # what a fill 1% worse on every leg would cost: buys pay more, sells receive less (first order: decisions unchanged)
            "turnover": round(sum(abs(float(x.get("contracts") or 1)) * float(x.get("multiplier") or 100.0) * (float(x.get("entry_price") or 0.0) + float(x.get("exit_price") or 0.0)) for x in t["legs"] if x.get("kind") == "option"), 2)}


def _scan(job: dict) -> dict:
    from strata.config import override_settings
    if job.get("data_dir"):
        override_settings(data_dir=job["data_dir"])
    from strata.engine.scan import scan_signals
    t0 = time.time()
    try:
        sc = scan_signals(job["config"], MarketStore())
    except Exception as e:  # noqa: BLE001
        return {"sig": job["sig"], "error": f"{type(e).__name__}: {e}"[:240], "sessions": [], "signals": 0, "seconds": round(time.time() - t0, 2)}
    return {"sig": job["sig"], "sessions": sorted({s["date"].isoformat() for s in sc["signals"]}), "signals": len(sc["signals"]), "evaluable": sc.get("sessions_evaluable"), "seconds": round(time.time() - t0, 2)}


def _run(job: dict) -> dict:
    from strata.config import override_settings
    if job.get("data_dir"):
        override_settings(data_dir=job["data_dir"])
    from strata.engine.engine import run_backtest
    t0 = time.time()
    try:
        res = run_backtest(job["config"], store=MarketStore(), record_paths=False, sessions={date.fromisoformat(x) for x in job["sessions"]} if job.get("sessions") is not None else None)
    except Exception as e:  # noqa: BLE001
        return {"i": job["i"], "error": f"{type(e).__name__}: {e}"[:240], "trades": [], "seconds": round(time.time() - t0, 2)}
    ex = res.meta.get("executability") or {}
    cs = res.meta.get("contract_selection") or {}
    return {"i": job["i"], "trades": [_trade_row(t) for t in res.trades], "signals": ex.get("signals"), "executed": ex.get("executed"), "not_executed": ex.get("not_executed_by_reason") or {},
            "deferred_exits": ex.get("deferred_exits"), "beyond": cs.get("beyond_band"), "edge": cs.get("at_band_edge"), "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------------------- statistics of one cell
def stats(trades: list[dict], seed: int = 7) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net": 0.0, "expectancy": None, "win_rate": None, "profit_factor": None, "max_drawdown": 0.0, "ci95": None, "t": None, "largest_share": None}
    p = np.array([t["pnl"] for t in trades], float)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(np.maximum(eq, 0.0)) - eq).max()) if n else 0.0
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    sd = float(p.std(ddof=1)) if n > 1 else 0.0
    out = {"trades": n, "net": round(float(p.sum()), 2), "expectancy": round(float(p.mean()), 2), "win_rate": round(float((p > 0).mean() * 100), 1),
           "profit_factor": round(float(wins / losses), 2) if losses > 0 else None, "max_drawdown": round(dd, 2), "avg_win": round(float(p[p > 0].mean()), 2) if (p > 0).any() else None,
           "avg_loss": round(float(p[p < 0].mean()), 2) if (p < 0).any() else None, "t": round(float(p.mean() / (sd / math.sqrt(n))), 2) if sd > 0 and n > 1 else None,
           "largest_share": round(float(p.max() / p.sum() * 100), 1) if p.sum() > 0 and p.max() > 0 else None,
           "max_consecutive_losses": int(max((len(r) for r in "".join("L" if x <= 0 else "W" for x in p).split("W")), default=0))}
    if n >= 5:
        rng = np.random.default_rng(seed)
        m = p[rng.integers(0, n, size=(BOOT, n))].mean(axis=1)
        out["ci95"] = [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]
    else:
        out["ci95"] = None
    return out


def cell(r: dict, cut: str) -> dict:
    tr = r.get("trades") or []
    whole = stats(tr)
    early, late = [t for t in tr if t["date"] < cut], [t for t in tr if t["date"] >= cut]
    turn = sum(t["turnover"] for t in tr)
    sig = r.get("signals") or 0
    return {**whole, "earlier": {k: v for k, v in stats(early).items() if k in ("trades", "net", "expectancy", "win_rate", "profit_factor")}, "later": {k: v for k, v in stats(late).items() if k in ("trades", "net", "expectancy", "win_rate", "profit_factor", "ci95")},
            "coverage": {"signals": sig, "executed": r.get("executed"), "pct": round((r.get("executed") or 0) / sig * 100, 1) if sig else None, "not_executed": r.get("not_executed") or {}, "deferred_exits": r.get("deferred_exits"),
                         "fetchable": int(sum(v for k, v in (r.get("not_executed") or {}).items() if k in ("contract_not_stored", "no_option_data")))},
            "sensitivity": {f"{s:g}%": round(whole["net"] - turn * s / 100.0, 2) for s in SLIPPAGE_STEPS} if tr else {}}


# ----------------------------------------------------------------------------- the grid as a whole
def _neighbours(dims: list[dict], key: dict) -> list[dict]:
    """The cells one step away along one dimension (an ordered dimension has a next and a previous value; symbols have none)."""
    out = []
    for d in dims:
        if d["field"] == "symbol":
            continue
        vals = d["values"]
        try:
            j = vals.index(key[d["field"]])
        except ValueError:
            continue
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                out.append({**key, d["field"]: vals[jj]})
    return out


def _k(key: dict) -> str:
    return json.dumps(key, sort_keys=True, default=str)


def stability(dims: list[dict], cells: list[dict], min_trades: int = 10) -> dict:
    """Which region of the grid gives a similar answer whichever cell you stand in.

    Each cell is scored by itself AND its neighbours: the lowest expectancy among them (a region is only as good as its worst
    cell) and how many of them have the same sign. The best single cell is reported beside the most stable one, with what its
    own neighbours did: a peak that stands alone is what a search over many cells produces by chance."""
    by = {_k(c["key"]): c for c in cells}
    rows = []
    for c in cells:
        m = c["metrics"]
        if m["trades"] < min_trades or m["expectancy"] is None:
            continue
        nb = [by[_k(k)] for k in _neighbours(dims, c["key"]) if _k(k) in by]
        nb = [x for x in nb if x["metrics"]["trades"] >= min_trades and x["metrics"]["expectancy"] is not None]
        ex = [m["expectancy"]] + [x["metrics"]["expectancy"] for x in nb]
        rows.append({"i": c["i"], "label": c["label"], "expectancy": m["expectancy"], "neighbours": len(nb), "region_min": round(min(ex), 2), "region_mean": round(float(np.mean(ex)), 2),
                     "region_spread": round(float(np.max(ex) - np.min(ex)), 2), "same_sign_pct": round(sum(1 for e in ex if (e > 0) == (m["expectancy"] > 0)) / len(ex) * 100, 0)})
    if not rows:
        return {"rows": [], "text": f"No configuration has {min_trades} trades or more, so no region can be compared."}
    best = max(rows, key=lambda r: r["expectancy"])
    with_nb = [r for r in rows if r["neighbours"] >= 1]
    stable = max(with_nb, key=lambda r: (r["region_min"], r["region_mean"])) if with_nb else None
    for r in rows:
        r["best"], r["stable"] = r["i"] == best["i"], bool(stable and r["i"] == stable["i"])
    text = f"The highest expectancy is {best['label']}: {best['expectancy']:+,.2f} a trade"
    if best["neighbours"]:
        text += f"; with the {best['neighbours']} configuration{'s' if best['neighbours'] != 1 else ''} next to it the figure ranges down to {best['region_min']:+,.2f}" + (", so it is a peak that stands alone" if best["region_min"] <= 0 < best["expectancy"] else "")
    text += ". "
    if stable is None:
        text += "The grid has no ordered setting, so there are no neighbouring configurations to compare."
    else:
        same = stable["i"] == best["i"]
        where = "that same cell" if same else stable["label"]
        nb = f"its {stable['neighbours']} neighbour{'s' if stable['neighbours'] != 1 else ''}"
        if stable["region_min"] > 0:
            text += f"The most stable region is around {where}: the cell and {nb} all make at least {stable['region_min']:+,.2f} a trade (mean {stable['region_mean']:+,.2f})."
        else:
            text += (f"No cell of this grid is profitable together with all of its neighbours. The least fragile region is around {where}: the worst of the cell and {nb} makes {stable['region_min']:+,.2f} a trade "
                     f"(mean {stable['region_mean']:+,.2f}).")
    return {"rows": rows, "best": best["i"], "stable": stable["i"] if stable else None, "text": text.strip()}


def _rank(x: list[float]) -> np.ndarray:
    a = np.asarray(x, float)
    order = a.argsort()
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def summarise(payload: dict, cells: list[dict], cut: str, timing: dict) -> dict:
    ok = [c for c in cells if c["status"] == "ok"]
    traded = [c for c in ok if c["metrics"]["trades"] > 0]
    n = len(cells)
    out: dict = {"configurations": n, "ran": len(ok), "with_trades": len(traded), "failed": len([c for c in cells if c["status"] == "error"]), "split": cut, "timing": timing}
    if not traded:
        out["text"] = f"{len(ok)} of {n} configurations ran and none of them traded on the stored observations."
        return out
    best = max(traded, key=lambda c: c["metrics"]["net"])
    pos = [c for c in traded if c["metrics"]["net"] > 0]
    both = [c for c in traded if c["metrics"]["earlier"]["trades"] >= 5 and c["metrics"]["later"]["trades"] >= 5]
    rho = None
    if len(both) >= 4:
        a, b = _rank([c["metrics"]["earlier"]["expectancy"] for c in both]), _rank([c["metrics"]["later"]["expectancy"] for c in both])
        rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    chosen = max(both, key=lambda c: c["metrics"]["earlier"]["expectancy"]) if both else None
    # the best cell's t-statistic, discounted for the number of cells it was the best of (Bonferroni on a two-sided normal test)
    t = best["metrics"].get("t")
    p_raw = float(math.erfc(abs(t) / math.sqrt(2))) if t is not None else None
    p_adj = min(1.0, p_raw * len(traded)) if p_raw is not None else None
    cov = [c["metrics"]["coverage"]["pct"] for c in ok if c["metrics"]["coverage"]["pct"] is not None]
    out.update(best={"i": best["i"], "label": best["label"], "net": best["metrics"]["net"], "trades": best["metrics"]["trades"], "expectancy": best["metrics"]["expectancy"], "t": t, "p_unadjusted": p_raw,
                     "p_adjusted_for_configurations": p_adj},
               profitable=len(pos), profitable_pct=round(len(pos) / len(traded) * 100, 1), median_net=round(float(np.median([c["metrics"]["net"] for c in traded])), 2),
               earlier_later_rank_correlation=round(rho, 2) if rho is not None else None, compared_on_both=len(both),
               chosen_on_earlier=None if chosen is None else {"i": chosen["i"], "label": chosen["label"], "earlier": chosen["metrics"]["earlier"], "later": chosen["metrics"]["later"]},
               coverage={"min_pct": min(cov) if cov else None, "median_pct": float(np.median(cov)) if cov else None, "fetchable_signals": int(sum(c["metrics"]["coverage"]["fetchable"] for c in ok))})
    txt = (f"{len(traded)} of {n} configurations traded. {len(pos)} of them ({out['profitable_pct']:g}%) end above zero; the median result is {out['median_net']:+,.2f}. "
           f"The best, {best['label']}, made {best['metrics']['net']:+,.2f} over {best['metrics']['trades']} trades")
    if p_adj is not None:
        txt += f" (t {t:+.2f}; as the best of {len(traded)} configurations that is p = {p_adj:.2f} after adjusting for the search, {p_raw:.3f} before)"
    txt += ". "
    if chosen is not None:
        e, l = chosen["metrics"]["earlier"], chosen["metrics"]["later"]
        txt += (f"Chosen on the earlier sessions alone, the best would have been {chosen['label']} ({e['expectancy']:+,.2f} a trade over {e['trades']} trades before {cut}); "
                f"on the later sessions it made {l['expectancy']:+,.2f} a trade over {l['trades']}. ")
    if rho is not None:
        txt += (f"Across the {len(both)} configurations with enough trades in both parts, the ranking by the earlier part and by the later part correlate at {rho:+.2f}: "
                + ("what looked better earlier mostly looked better later." if rho >= 0.5 else "a weak relationship." if rho >= 0.2 else "the earlier ranking says little about the later one." if rho > -0.2
                   else "the ranking tends to REVERSE: what looked better earlier mostly looked worse later.") + " ")
    if cov and min(cov) < 100:
        txt += (f"Every configuration could price {min(cov):g}% of its signals on stored observations. " if max(cov) - min(cov) < 0.05 else
                f"Coverage ranges from {min(cov):g}% to {max(cov):g}% of the signals: configurations are not all tested on the same sessions. ")
    txt += "Both parts are history that has now been examined; neither is evidence of an edge."
    out["text"] = txt
    return out


# ----------------------------------------------------------------------------- running it
POOL = True  # worker processes for the grid (one configuration per core). Off: every configuration in this process


def run_experiment(payload: dict, store: MarketStore | None = None, progress=None, should_cancel=None, workers: int | None = None, pool: bool | None = None) -> dict:
    """Run every configuration of a built experiment. Returns the result (JSON-safe)."""
    from strata.config import get_settings
    store = store or MarketStore()
    pool = POOL if pool is None else pool
    cfgs = payload["configs"]
    t0 = time.time()
    data_dir = str(get_settings().data_dir)
    groups: dict[str, list[dict]] = {}
    for c in cfgs:
        groups.setdefault(entry_signature(c["config"]), []).append(c)
    scan_jobs = [{"sig": sig, "config": {**g[0]["config"], "pricing": {"mode": "trades"}}, "data_dir": data_dir} for sig, g in groups.items()]
    n_jobs = len(scan_jobs) + len(cfgs)
    done = [0]

    def tick(msg: str):
        done[0] += 1
        if progress:
            progress(min(0.99, done[0] / max(1, n_jobs)), msg)

    def each(fn, jobs, label):
        if should_cancel and should_cancel():
            raise _Cancelled()
        if not pool or len(jobs) < 3:
            for j in jobs:
                if should_cancel and should_cancel():
                    raise _Cancelled()
                r = fn(j)
                tick(label(r))
                yield r
            return
        import multiprocessing as mp
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        nw = max(1, min(workers or max(1, (mp.cpu_count() or 4) - 2), len(jobs), 10))
        # each worker is one configuration on one core. Left alone, every worker starts a thread pool as wide as the machine
        # (the data-frame library, the compiled pricing kernels), and eight workers then fight over ten cores with eighty
        # threads: measured on this grid, a load average over 100 and less than half the throughput.
        caps = {"POLARS_MAX_THREADS": "1", "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        before = {k: os.environ.get(k) for k in caps}
        os.environ.update(caps)  # a spawned worker takes the environment as it is when it starts
        try:
            with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
                futs = [ex.submit(fn, j) for j in jobs]
                for f in as_completed(futs):
                    if should_cancel and should_cancel():
                        for x in futs:
                            x.cancel()
                        raise _Cancelled()
                    r = f.result()
                    tick(label(r))
                    yield r
        finally:
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    scans = {r["sig"]: r for r in each(_scan, scan_jobs, lambda r: f"signals for one entry rule: {r.get('signals', 0)} on {len(r.get('sessions') or [])} sessions")}
    t_scan = time.time() - t0
    jobs = []
    for sig, g in groups.items():
        sc = scans.get(sig) or {}
        for c in g:
            jobs.append({"i": c["i"], "config": c["config"], "sessions": None if sc.get("error") else sc.get("sessions", []), "data_dir": data_dir})
    by_i = {c["i"]: c for c in cfgs}
    results = {r["i"]: r for r in each(_run, jobs, lambda r: f"{by_i[r['i']]['label']}: {len(r.get('trades') or [])} trades")}
    # the same split for every configuration: by session, over the union of the windows
    a, b = min(c["config"]["start"] for c in cfgs), max(c["config"]["end"] for c in cfgs)
    days = cal.trading_days(date.fromisoformat(a), date.fromisoformat(b))
    cut = days[int(len(days) * (1 - float(payload.get("oos_fraction") or OOS_FRACTION)))].isoformat() if days else b
    cells = []
    for c in cfgs:
        r = results.get(c["i"]) or {"error": "not run", "trades": []}
        sig = entry_signature(c["config"])
        cells.append({"i": c["i"], "key": c["key"], "label": c["label"], "symbol": c["symbol"], "status": "error" if r.get("error") else "ok", "error": r.get("error"), "metrics": cell(r, cut),
                      "signal_sessions": len((scans.get(sig) or {}).get("sessions") or []), "seconds": r.get("seconds"), "trades": r.get("trades") or []})
    timing = {"seconds": round(time.time() - t0, 1), "scan_seconds": round(t_scan, 1), "entry_rules": len(groups), "configurations": len(cfgs), "scans_shared_by": round(len(cfgs) / max(1, len(groups)), 1),
              "cpu_seconds": round(sum((r.get("seconds") or 0) for r in results.values()) + sum((s.get("seconds") or 0) for s in scans.values()), 1)}
    out = {"title": payload["title"], "base": payload["base"], "dims": payload["dims"], "notes": payload["notes"], "errors": payload.get("errors") or [], "window": [a, b], "split": cut, "mode": payload.get("mode", "trades"),
           "cells": cells, "stability": stability(payload["dims"], cells), "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "observation": "trade prints (no bid/ask): a fill is the last print the engine had seen, with the run's slippage assumption"}
    out["summary"] = summarise(payload, cells, cut, timing)
    if progress:
        progress(1.0, "done")
    return out


class _Cancelled(Exception):
    pass


def slim(res: dict) -> dict:
    """The result without each cell's trade list (the table and the heatmap do not need them)."""
    return {**res, "cells": [{k: v for k, v in c.items() if k != "trades"} for c in res["cells"]]}


def save(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "experiment.json").write_text(dumps(slim(res)))
    (d / "experiment_trades.json").write_text(dumps({str(c["i"]): c["trades"] for c in res["cells"]}))
    sm = res["summary"]
    return {"experiment": True, "configurations": sm["configurations"], "with_trades": sm.get("with_trades"), "best": sm.get("best"), "profitable_pct": sm.get("profitable_pct"), "text": sm.get("text"),
            "start": res["window"][0], "end": res["window"][1], "symbols": sorted({c["symbol"] for c in res["cells"]}), "seconds": sm["timing"]["seconds"]}


def load(rid: str) -> dict:
    from strata.api.results import run_dir
    p = run_dir(rid) / "experiment.json"
    if not p.exists():
        raise FileNotFoundError(rid)
    return json.loads(p.read_text())


def trades_of(rid: str, i: int) -> list[dict]:
    from strata.api.results import run_dir
    p = run_dir(rid) / "experiment_trades.json"
    return (json.loads(p.read_text()) if p.exists() else {}).get(str(i), [])


def csv_rows(res: dict) -> list[dict]:
    rows = []
    for c in res["cells"]:
        m = c["metrics"]
        rows.append({**{f"setting_{k}": (v if not isinstance(v, list) else f"{v[0]} {v[1]}") for k, v in c["key"].items()}, "label": c["label"], "status": c["status"], "trades": m["trades"], "net": m["net"], "expectancy": m["expectancy"],
                     "win_rate": m["win_rate"], "profit_factor": m["profit_factor"], "max_drawdown": m["max_drawdown"], "ci95_low": (m.get("ci95") or [None, None])[0], "ci95_high": (m.get("ci95") or [None, None])[1], "t": m.get("t"),
                     "earlier_trades": m["earlier"]["trades"], "earlier_expectancy": m["earlier"]["expectancy"], "later_trades": m["later"]["trades"], "later_expectancy": m["later"]["expectancy"],
                     "signals": m["coverage"]["signals"], "executed": m["coverage"]["executed"], "coverage_pct": m["coverage"]["pct"], **{f"net_with_{k}_worse_fills": v for k, v in m["sensitivity"].items()}})
    return rows
