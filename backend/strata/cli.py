"""Command-line interface: `strata <command>` (or `python -m strata.cli`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path


def cmd_init(a):
    from strata.api import db
    from strata.config import get_settings
    from strata.data.store import MarketStore
    p = db.init_db()
    print(f"database ready: {p}")
    st = MarketStore()
    if a.demo and "DEMO" not in st.symbols():
        cmd_demo(a)
    else:
        print(f"market data dir: {get_settings().market_dir} ({len(st.symbols())} symbols)")


def cmd_demo(a):
    from strata.data.store import MarketStore
    from strata.data.synthetic import SyntheticConfig, generate_demo_dataset
    cfg = SyntheticConfig(start=date.fromisoformat(a.start), end=date.fromisoformat(a.end), seed=a.seed, option_interval=a.interval)
    t = time.time()

    def prog(sym, i, n):
        print(f"\r  {sym}: {i}/{n} sessions", end="", flush=True)
    out = generate_demo_dataset(MarketStore(), cfg, progress=prog)
    print(f"\nSYNTHETIC demo dataset generated in {time.time() - t:.0f}s: {out}")


def cmd_serve(a):
    import uvicorn
    pkg = str(Path(__file__).resolve().parent)
    uvicorn.run("strata.api.app:app", host=a.host, port=a.port, reload=a.reload, reload_dirs=[pkg] if a.reload else None, workers=1)


def cmd_backtest(a):
    """Run a backtest from a JSON BacktestConfig file, or a Python file defining
    STRATEGIES (list of Strategy subclasses instances) and CONFIG (dict)."""
    from strata.analytics.metrics import compute_metrics
    from strata.engine.engine import run_backtest
    from strata.engine.models import BacktestConfig
    path = Path(a.file)
    strategies = None
    if path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_strategy", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # the user's own local file
        cfg = BacktestConfig.model_validate(mod.CONFIG)
        strategies = mod.STRATEGIES
    else:
        cfg = BacktestConfig.model_validate(json.loads(path.read_text()))
    t = time.time()
    res = run_backtest(cfg, strategies=strategies, progress=lambda i, n, d: print(f"\r  {d} ({i}/{n})", end="", flush=True))
    print()
    m = compute_metrics(res.trades, res.daily, cfg.capital.initial_capital, cfg.capital.risk_free_rate)
    keys = ["trades", "net_pnl", "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "expectancy", "total_commissions"]
    for k in keys:
        v = m.get(k)
        print(f"  {k:20s} {v:,.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    print(f"  {'synthetic data':20s} {res.meta['synthetic']}")
    print(f"  {'fingerprint':20s} {res.fingerprint()}  (config {res.meta['config_hash']}, data {res.meta['dataset_versions']})")
    if a.save:
        from strata.api import db
        from strata.api.results import save_backtest
        db.init_db()
        rid = db.create_run("backtest", a.name or ("python: " + path.stem if strategies else cfg.label or path.stem), cfg.model_dump(mode="json"),
                            config_hash=cfg.config_hash(), dataset_versions=res.meta["dataset_versions"], engine_version=res.meta["engine_version"], synthetic=res.meta["synthetic"])
        summary = save_backtest(rid, res)
        db.update_run(rid, status="done", progress=1.0, summary=summary, started_at=db.now(), finished_at=db.now())
        print(f"saved as run {rid} (open it in the UI under Backtest Results)")
    print(f"done in {time.time() - t:.1f}s")


def cmd_import(a):
    from strata.data.importers import import_file
    print(import_file(a.file, a.kind, a.symbol, tz=a.tz))


def cmd_ingest(a):
    from strata.data.providers import get_provider
    p = get_provider(a.provider)
    print(p.ingest(a.symbol, date.fromisoformat(a.start), date.fromisoformat(a.end), interval=a.interval, include_options=not a.no_options,
                   progress=lambda f, m: print(f"\r  {f:5.1%} {m}", end="", flush=True)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strata", description="Strata options research platform")
    ap.add_argument("--data-dir", help="override STRATA_DATA_DIR")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create the database (and the synthetic demo dataset with --demo)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--start", default="2023-01-03")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seed", type=int, default=20240101)
    p.add_argument("--interval", type=int, default=15)
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("demo-data", help="(re)generate the SYNTHETIC demo dataset")
    p.add_argument("--start", default="2023-01-03")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seed", type=int, default=20240101)
    p.add_argument("--interval", type=int, default=15)
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("backtest", help="run a backtest from a JSON config or a Python strategy file")
    p.add_argument("file")
    p.add_argument("--save", action="store_true", help="store the result so it appears in the UI")
    p.add_argument("--name", default="")
    p.set_defaults(fn=cmd_backtest)
    p = sub.add_parser("import", help="import a CSV/Parquet file")
    p.add_argument("file")
    p.add_argument("--kind", required=True, choices=["underlying_minutes", "underlying_daily", "option_quotes", "events", "corporate_actions"])
    p.add_argument("--symbol")
    p.add_argument("--tz", default="America/New_York")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("ingest", help="download from a configured provider")
    p.add_argument("provider", choices=["thetadata", "polygon", "databento", "orats"])
    p.add_argument("symbol")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--interval", type=int, default=15)
    p.add_argument("--no-options", action="store_true")
    p.set_defaults(fn=cmd_ingest)
    a = ap.parse_args(argv)
    if a.data_dir:
        from strata.config import override_settings
        override_settings(data_dir=a.data_dir)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
