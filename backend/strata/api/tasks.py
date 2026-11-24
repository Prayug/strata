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
