"""Job bodies executed inside worker processes."""
from __future__ import annotations

import json
from datetime import date

from strata.api import db
from strata.api.results import dumps, run_dir, save_backtest
from strata.config import get_settings


def execute(rid: str, kind: str, payload: dict, progress, should_cancel) -> dict:
    if kind == "backtest":
        from strata.engine.engine import run_backtest
        from strata.engine.models import BacktestConfig
        cfg = BacktestConfig.model_validate(payload["config"])
        res = run_backtest(cfg, progress=lambda i, n, d: progress(i / n, f"session {d} ({i}/{n})"), should_cancel=should_cancel)
        progress(0.99, "computing analytics")
        return save_backtest(rid, res)
    if kind == "portfolio":
        from strata.engine.models import BacktestConfig
        from strata.portfolio.portfolio import run_portfolio
        cfg = BacktestConfig.model_validate(payload["config"])
        combined, analysis = run_portfolio(cfg, progress=progress, should_cancel=should_cancel)
        s = save_backtest(rid, combined, extra={"portfolio": analysis})
        s["portfolio"] = True
        s["naive_net_pnl"] = analysis["naive_sum_equity"][-1] - cfg.capital.initial_capital if analysis["naive_sum_equity"] else None
        return s
    if kind == "optimization":
        from strata.optimize.optimizer import OptimizationConfig, run_optimization
        oc = OptimizationConfig.model_validate(payload["config"])
        out = run_optimization(oc, str(get_settings().data_dir), progress=lambda d, t, label: progress(d / max(t, 1), f"{label.replace('_', ' ')}: {d}/{t}"), should_cancel=should_cancel)
        d = run_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "optimization.json").write_text(dumps(out | {"config": oc.model_dump(mode="json")}))
        if payload.get("strategy_id"):
            db.add_trials(payload["strategy_id"], out["n_configs"])
        best = out["configs"][out["best_index"]] if out.get("best_index") is not None else None
        return {"n_configs": out["n_configs"], "objective": out["objective"], "method": out["method"], "validation": out["validation"],
                "best_params": best["params"] if best else None, "best_objective": best["objective"] if best else None,
                "dsr": (out.get("deflated_sharpe") or {}).get("dsr"), "pbo": (out.get("pbo") or {}).get("pbo"), "warnings": len(out["warnings"]),
                "oos_degradation": out.get("oos_degradation"), "synthetic": out.get("synthetic", False)}
    if kind == "entry_timing":
        from strata.optimize.optimizer import Evaluator, apply_params
        cfg = payload["config"]
        path, times = payload["path"], payload["times"]
        cfgs = [apply_params(cfg, {path: t}) for t in times]
        with Evaluator(None, str(get_settings().data_dir), lambda d, t, l: progress(d / max(t, 1), f"{d}/{t} entry times"), should_cancel) as ev:
            ev.total = len(cfgs)
            res = ev.run(cfgs, "entry_timing")
        rows = []
        for t, r in zip(times, res):
            m = r.get("metrics") or {}
            rows.append({"time": t, "ok": r.get("ok"), "error": r.get("error"), **{k: m.get(k) for k in ("trades", "net_pnl", "sharpe", "win_rate", "expectancy", "max_drawdown", "profit_factor")}})
        out = {"rows": rows, "path": path}
        d = run_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "entry_timing.json").write_text(dumps(out))
        return {"times": len(times)}
    if kind == "research":
        from strata.research.pipeline import run_pipeline
        return run_pipeline(rid, payload["spec"], progress, should_cancel)
    if kind == "discovery":
        from strata.research.discovery import run_discovery, save_discovery
        return save_discovery(rid, run_discovery(payload["spec"], progress=progress, should_cancel=should_cancel, scope=payload.get("scope"), run_id=rid))
    if kind == "hypothesis":
        from strata.research.hypothesis import run_plan, save_plan_result
        return save_plan_result(rid, run_plan(payload["plan"], scope=payload.get("scope"), progress=progress, should_cancel=should_cancel, run_id=rid))
    if kind == "graph":
        from strata.research import graph as GR
        if payload.get("batch"):
            return GR.save_graph_result(rid, GR.run_batch(payload["batch"], scope=payload.get("scope"), run_id=rid, progress=progress, should_cancel=should_cancel, title=payload.get("title", ""), what=payload.get("what", ""),
                                                          include_holdout=bool(payload.get("include_holdout"))))
        return GR.save_graph_result(rid, GR.run_graph(payload["graph"], scope=payload.get("scope"), run_id=rid, progress=progress, should_cancel=should_cancel))
    if kind == "ml":
        from strata.research import ml as ML
        return ML.save_result(rid, ML.run(payload["spec"], scope=payload.get("scope"), run_id=rid, progress=progress, should_cancel=should_cancel))
    if kind in ("stress", "challenge", "sensitivity"):
        from strata.api.results import load_run_result, load_trades
        from strata.data.store import MarketStore
        target = db.get_run(payload["target"])
        if target is None:
            raise LookupError("the run this analysis refers to no longer exists")
        res = load_run_result(target)
        store = MarketStore()
        data_dir = str(get_settings().data_dir)
        if kind == "stress":
            from strata.research import execution as EX
            ctx = EX.context_for(target, res, store)
            out = EX.run_stress(ctx["config"], data_dir, ctx["quotes_available"], ctx["eod"], progress=progress, should_cancel=should_cancel)
            out["target"] = target["id"]
            out["quality"] = EX.quality(load_trades(target["id"]), res.get("meta") or {}, res.get("diagnostics") or {}, ctx["config"], ctx["eod"])
            return EX.save_stress(rid, out)
        if kind == "challenge":
            from strata.research import challenge as CH
            from strata.research import ledger as LG
            trades = None if target["kind"] == "study" or res.get("kind") == "study" else load_trades(target["id"])
            hold = None
            if payload.get("scope"):
                sym = (res.get("symbol") or ((res.get("config") or {}).get("strategies") or [{}])[0].get("symbol") or "").upper()
                hold = LG.current(payload["scope"], sym) if sym else None
            out = CH.run_challenge(target, res, trades, store, data_dir, int(payload.get("trials") or 1), hold, progress, should_cancel)
            return CH.save_challenge(rid, out)
        from strata.research import ledger as LG
        from strata.research import sensitivity as SN
        cfg = res.get("config")
        if not cfg or "strategies" not in cfg:
            raise ValueError("an option sensitivity map needs a finished option backtest")
        out = SN.option_map(cfg, payload["spec"], data_dir, progress, should_cancel)
        out["target"] = target["id"]
        if payload.get("scope"):
            sym = cfg["strategies"][0]["symbol"]
            LG.record(payload["scope"], sym, [{"kind": "sensitivity", "key": LG.config_key(c["config"]), "label": f"{out['title']}: {c['x']} / {c['y']}", "run_id": rid, "start": cfg.get("start"), "end": cfg.get("end"),
                                              "metric": "expectancy", "value": c.get("expectancy"), "n": c.get("trades")} for c in out["cells"] if c["status"] == "ok"], family=out["title"])
        d = run_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sensitivity.json").write_text(dumps(out))
        return {"sensitivity": True, "cells": len(out["cells"]), "thin": out["summary"]["thin"], "target": target["id"], "title": out["title"]}
    if kind == "collector":
        from strata.collector.service import run_once
        return run_once(progress=progress, should_cancel=should_cancel, trigger=payload.get("trigger", "manual"))
    if kind == "alpaca_validation":
        from strata.data.providers.market import alpaca_validation
        out = alpaca_validation(progress=progress, should_cancel=should_cancel, symbol=payload.get("symbol") or "SPY")
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "provider_benchmark":
        from strata.data.providers.bench import benchmark_source
        return benchmark_source(payload["source"], progress=progress, should_cancel=should_cancel)
    if kind == "bulk_import":
        from strata.data.bulk import import_path
        out = import_path(payload["path"], symbol=payload.get("symbol"), fmt=payload.get("format"), tz=payload.get("tz", "America/New_York"), provider=payload.get("provider") or "import", progress=progress, should_cancel=should_cancel)
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "study":
        from strata.api.results import save_study
        from strata.research.study import run_study
        progress(0.05, "evaluating sessions")
        res = run_study(payload["config"])
        progress(0.95, "saving")
        return save_study(rid, res)
    if kind == "source_ingest":
        from strata.data.providers.sources import get_source
        src = get_source(payload["source"])
        start = date.fromisoformat(payload["start"]) if payload.get("start") else None
        end = date.fromisoformat(payload["end"]) if payload.get("end") else None
        out = src.ingest(payload["dataset"], [x.upper() for x in payload.get("symbols") or []], start, end, progress=progress, should_cancel=should_cancel, force=bool(payload.get("force")))
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "quickstart":
        from strata.data.providers.sources import quickstart
        out = quickstart([x.upper() for x in payload.get("symbols") or []] or None, progress=progress, should_cancel=should_cancel, intraday=bool(payload.get("intraday", True)))
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "demo_data":
        from strata.data.store import MarketStore
        from strata.data.synthetic import SyntheticConfig, generate_demo_dataset
        cfg = SyntheticConfig(start=date.fromisoformat(payload.get("start", "2023-01-03")), end=date.fromisoformat(payload.get("end", "2024-12-31")),
                              seed=int(payload.get("seed", 20240101)), option_interval=int(payload.get("interval", 15)))
        nsyms = len(cfg.underlyings)
        state = {"k": 0}

        def prog(sym, i, n):
            k = [u.symbol for u in cfg.underlyings].index(sym)
            progress(0.1 + 0.85 * (k + i / n) / nsyms, f"{sym} chains {i}/{n}")
        progress(0.02, "simulating underlying paths")
        out = generate_demo_dataset(MarketStore(), cfg, log=lambda *a: None, progress=prog)
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "ingest":
        from strata.data.providers import get_provider
        prov = get_provider(payload["provider"])
        out = prov.ingest(payload["symbol"], date.fromisoformat(payload["start"]), date.fromisoformat(payload["end"]),
                          interval=int(payload.get("interval", 15)), include_options=payload.get("include_options", True), progress=progress, should_cancel=should_cancel)
        from strata.data.catalog import invalidate
        invalidate()
        return out
    if kind == "import":
        from strata.data.importers import import_file
        out = import_file(payload["path"], payload["kind"], payload.get("symbol"), tz=payload.get("tz", "America/New_York"),
                          column_map=payload.get("column_map") or {}, meta=payload.get("meta") or {}, progress=progress)
        from strata.data.catalog import invalidate
        invalidate()
        return out
    raise ValueError(f"unknown job kind {kind}")
