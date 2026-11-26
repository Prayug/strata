"""Command-line interface: `strata <command>` (or `python -m strata.cli`)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path


def cmd_init(a):
    from strata.api import db
    from strata.config import get_settings
    from strata.data.store import MarketStore
    p = db.init_db()
    print(f"database ready: {p}")
    st = MarketStore()
    real = [k for k in st.symbols() if not st.is_synthetic(k)]
    print(f"market data dir: {get_settings().market_dir} ({len(real)} real symbols stored)")
    if getattr(a, "fetch", False):
        a.symbols, a.no_intraday, a.options, a.years = ["SPY", "QQQ", "IWM"], False, False, None
        cmd_fetch(a)
    elif not real:
        print("No market data yet. Get free real data with:  strata fetch            (daily + recent intraday, a few MB)")
        print("                                              strata fetch --options  (adds SPY end-of-day option chains)")


def cmd_fetch(a):
    """Free, keyless real market data: Yahoo daily/intraday bars, CBOE VIX, and optionally
    the open end-of-day option-chain archive. Nothing here costs money or needs an account."""
    from strata.data.providers.sources import get_source, quickstart, verify_daily
    syms = [x.upper() for x in a.symbols]
    t = time.time()

    def prog(f, m=""):
        print(f"\r  {f:5.1%} {str(m)[:70]:70s}", end="", flush=True)
    out = quickstart(syms, progress=prog, intraday=not a.no_intraday)
    print()
    for step in out["steps"]:
        print("  ", json.dumps(step, default=str)[:300])
    if a.options:
        src = get_source("github_eod")
        osyms = [x for x in syms if x in ("SPY", "QQQ", "IWM")]
        if not osyms:
            print("  the free option archive covers SPY, QQQ and IWM only")
        else:
            start = date(int(a.years.split("-")[0]), 1, 1) if a.years else date(date.today().year - 5, 1, 1)
            end = date(int(a.years.split("-")[-1]), 12, 31) if a.years else None
            plan = src.plan("options_eod", osyms, start, end)
            print(f"  option chains: {json.dumps(plan, default=str)[:400]}")
            print("  ", json.dumps(src.ingest("options_eod", osyms, start, end, progress=prog), default=str)[:400])
            print()
    if "SPY" in syms:
        try:
            v = verify_daily(__import__("strata.data.store", fromlist=["MarketStore"]).MarketStore(), "SPY")
            print(f"  verified SPY daily closes against an independent source: {v.get('compared')} sessions compared, max difference {v.get('max_close_diff_pct', 0):.3f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  (verification skipped: {e})")
    print(f"done in {time.time() - t:.0f}s")


def cmd_demo(a):
    from strata.data.store import MarketStore
    from strata.data.synthetic import SyntheticConfig, generate_demo_dataset
    cfg = SyntheticConfig(start=date.fromisoformat(a.start), end=date.fromisoformat(a.end), seed=a.seed, option_interval=a.interval)
    t = time.time()

    def prog(sym, i, n):
        print(f"\r  {sym}: {i}/{n} sessions", end="", flush=True)
    out = generate_demo_dataset(MarketStore(), cfg, progress=prog)
    print(f"\nSIMULATED sandbox symbols generated in {time.time() - t:.0f}s: {out}")
    print("These are not market data. Runs that use them must set allow_synthetic and are labelled SIMULATED everywhere.")


def cmd_serve(a):
    import uvicorn
    pkg = str(Path(__file__).resolve().parent)
    # where this copy can be opened: shown by another copy that finds the data directory in use
    os.environ.setdefault("STRATA_PUBLIC_URL", f"http://localhost:{a.port}")
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


def cmd_bulk_import(a):
    from strata.data import bulk
    info = bulk.inspect(a.path)
    for f in info["files"][:20]:
        print(f"  {f['label']:60s} {f['path']}")
    for f in info["unrecognised"][:10]:
        print(f"  not recognised: {f['path']}  columns: {', '.join(f['columns'][:8])}")
    if a.inspect or not info["files"]:
        return
    rep = bulk.import_path(a.path, symbol=a.symbol, fmt=a.format, tz=a.tz, provider=a.provider, progress=lambda f, m: print(f"\r  {f:5.1%} {m[:70]:70s}", end="", flush=True))
    print(f"\n{rep['rows_written']:,} rows written in {rep['sessions']} option sessions for {', '.join(rep['symbols'])}; at most {rep['peak_rows_in_memory']:,} rows were held in memory")
    for k, v in rep["rejected"].items():
        print(f"  rejected {v:,}: {k}")
    for sk in rep["skipped"][:10]:
        print(f"  skipped {sk['file']}: {sk['reason']}")


def cmd_collect(a):

    from strata.api import db
    from strata.collector import service as CS
    from strata.config import get_settings
    if a.health:
        # for a container health check: is the daemon loop alive (collecting, or waiting on standby)? Reads one file's age, nothing else.
        import time as _t
        beat = get_settings().data_dir / "collector.heartbeat"
        ok = beat.exists() and _t.time() - beat.stat().st_mtime < 300
        print("collector daemon is alive" if ok else "collector daemon has not reported for five minutes (or was never started)")
        sys.exit(0 if ok else 1)
    db.init_db()
    if a.status:
        st = CS.state()
        print(f"collector {'ENABLED' if st['config']['enabled'] else 'disabled'}; symbols {', '.join(st['config']['symbols'])}; next due {st['next_due'] or 'n/a'}")
        for r in st["history"][:8]:
            print(f"  {r['started_at'][:19]}  {r['status']:9s} {r['report'].get('bars', 0):>8,} bars  {r['report'].get('requests', 0):>5} requests  {'; '.join(r['report'].get('errors', [])[:1])[:80]}")
        print(f"missed sessions: {len(st['coverage']['missed'])}")
        return
    beat = get_settings().data_dir / "collector.heartbeat"
    if a.daemon:
        import time as _t
        from datetime import datetime as _dt

        from strata.core import calendar as cal
        print("collector running; press Ctrl-C to stop. It collects after each close while enabled in its settings, and it never trades.", flush=True)
        if not CS.load_config()["enabled"]:
            print("NOTE: the collector is switched off in its settings, so nothing will be collected until it is enabled (Data > Collector, or STRATA_COLLECTOR_ENABLE=1).", flush=True)
        if os.environ.get("STRATA_COLLECTOR_ENABLE") == "1" and not CS.load_config()["enabled"]:
            CS.save_config({"enabled": True})
            print("enabled by STRATA_COLLECTOR_ENABLE=1", flush=True)
        for rid in CS.recover_interrupted():
            print(f"cycle {rid} was cut off by a restart; continuing from what it stored", flush=True)
        while True:
            beat.parent.mkdir(parents=True, exist_ok=True)
            beat.write_text(_dt.now(cal.ET).isoformat())
            try:  # one bad cycle (an outage, a full disk, another process mid-cycle) must not stop the daemon
                cfg = CS.load_config()
                nd = CS.next_due(cfg)
                last = CS.history(1)
                failed_recently = bool(last and last[0]["status"] in ("error", "skipped") and (_dt.now(cal.ET) - _dt.fromisoformat(last[0]["started_at"]).astimezone(cal.ET)).total_seconds() < 3600)
                if cfg["enabled"] and nd is not None and nd <= _dt.now(cal.ET) and not failed_recently:
                    rep = CS.run_once(progress=lambda f, m: beat.write_text(_dt.now(cal.ET).isoformat()), trigger="daemon")
                    print(f"{rep['started_at'][:19]}  {rep['status']}  {rep['bars']:,} bars  {rep['requests']} requests" + (f"  recovered {len(rep['recovered'])}" if rep.get("recovered") else "")
                          + (f"  unrecoverable {len(rep['unrecoverable'])}" if rep.get("unrecoverable") else "") + ("  " + "; ".join(rep["errors"][:1])[:120] if rep.get("errors") else ""), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{_dt.now(cal.ET).isoformat()[:19]}  cycle not run: {type(e).__name__}: {e}"[:300], flush=True)
            _t.sleep(60)
    rep = CS.run_once(progress=lambda f, m: print(f"\r  {f:5.1%} {m[:70]:70s}", end="", flush=True), trigger="cli")
    print(f"\n{rep['status']}: {rep['bars']:,} option bars, {rep['requests']} requests, {rep['seconds']}s")
    for e in rep["errors"][:8]:
        print("  " + e)


def cmd_bench(a):
    from strata import bench
    sys.exit(bench.main(quick=a.quick, real=a.real, check=a.check, update_baseline=a.update_baseline, as_json=a.json))


def cmd_providers(a):
    from strata.data.providers import bench as PB
    names = a.sources or [n for n in PB.BENCHABLE]
    for n in names:
        r = PB.benchmark_source(n, progress=lambda f, m: None)
        print(f"{r.get('label', n):34s} {r['status'].upper():16s} " + (r.get("reason", "")[:110] if r["status"] == "unverified" else f"{r.get('requests_ok', 0)} requests ok, {r.get('requests_failed', 0)} failed"))
        for s_ in r.get("steps", []):
            print(f"    {'ok ' if s_['ok'] else 'ERR'} {s_['name']:42s} {s_.get('rows', '')!s:>8} rows  {s_.get('first') or ''} .. {s_.get('last') or ''}  {s_.get('error', '')[:90]}")
        for e in r.get("entitlements", []):
            print(f"    entitlement: {e['what']}: {'granted' if e['granted'] else 'NOT granted'} ({str(e['detail'])[:80]})")


def cmd_download(a):
    """Download from a source that needs no payment (the same code the Data page runs), outside the web application:
    1-minute underlying bars, or the bulk set of option trade bars (every contract within a band of the previous
    close that expires within a few days). Resumable: what is already stored is skipped."""
    from datetime import timedelta
    from strata.core import calendar as cal
    from strata.data.providers import sources as S
    from strata.data.store import MarketStore
    from strata.core import instance
    instance.ensure()  # the download writes market files: one machine at a time
    store = MarketStore()
    try:
        src = S.get_source(a.source, store)
    except (KeyError, ValueError) as e:
        sys.exit(f"unknown source {a.source!r}: {e}")
    ok, why = src.available()
    if not ok:
        print(f"BLOCKED: {src.label}: {why}")
        sys.exit(2)
    today = datetime.now(cal.ET).date()
    end = date.fromisoformat(a.end) if a.end else today
    t0 = time.time()
    last = [0.0]

    def prog(prefix=""):
        def f(frac, msg):
            if time.time() - last[0] >= a.every:
                last[0] = time.time()
                print(f"{time.time() - t0:7.0f}s {prefix}{frac:6.1%} {msg}", flush=True)
        return f

    failed = 0
    for sym in [x.upper() for x in a.symbols]:
        S.ensure_symbol(store, sym)
        if a.dataset == "option_trades" and isinstance(src, S.OptionBarSource):
            first, lastday = src.bars_window(today)
            start = max(date.fromisoformat(a.start) if a.start else first, first)
            tot = {"pairs_requested": 0, "pairs_cached": 0, "pairs_ok": 0, "pairs_empty": 0, "pairs_unavailable": 0, "pairs_failed": 0, "bars": 0, "requests": 0, "pages": 0}
            days = cal.trading_days(start, min(end, lastday))
            months = sorted({(d.year, d.month) for d in days})
            for mi, (y, m) in enumerate(months):
                chunk = [d for d in days if (d.year, d.month) == (y, m)]
                tag = f"[{sym} {y}-{m:02d}, {mi + 1}/{len(months)}] "
                try:
                    wanted = src.bulk_wanted(sym, chunk[0], chunk[-1], a.band, a.max_dte)
                    if a.rights != "both":
                        wanted = {d: [c for c in cs if c.right == a.rights[0].upper()] for d, cs in wanted.items()}
                    rep = src.acquire(sym, wanted, prog(tag), None, a.force)
                except S.SourceError as e:
                    print(f"{tag}FAILED: {str(e)[:300]}", flush=True)
                    failed += 1
                    continue
                for k in tot:
                    tot[k] += int(rep.get(k) or 0)
                failed += int(rep.get("pairs_failed") or 0)
                print(f"{time.time() - t0:7.0f}s {tag}{len(chunk)} sessions: {rep['pairs_ok']} contract-sessions with trades, {rep['pairs_empty']} without, {rep['pairs_cached']} already stored, "
                      f"{rep['pairs_failed']} failed, {rep['bars']:,} bars, {rep['requests']} requests" + (f" | rejected {rep['rejected']}" if rep.get("rejected") else "") + (f" | errors {rep['errors'][:2]}" if rep.get("errors") else ""), flush=True)
            print(f"DONE {sym} {a.dataset}: " + json.dumps(tot) + f" in {time.time() - t0:.0f}s", flush=True)
        else:
            start = date.fromisoformat(a.start) if a.start else None
            if a.dataset == "bars_1m" and hasattr(src, "stock_bars"):
                rep = src.stock_bars(sym, start or src.STOCKS_FIRST, end, prog(f"[{sym}] "), None, a.force)
                S._finish(store, sym)
            else:
                rep = src.ingest(a.dataset, [sym], start, end, prog(f"[{sym}] "), None, a.force)
            print(f"DONE {sym} {a.dataset}: " + json.dumps(rep, default=str)[:1200] + f" in {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if failed else 0)


def cmd_access(a):
    """What the stored credentials can really retrieve: the expired-contract matrix, or BLOCKED and what to supply."""
    from strata.data.providers import access as A
    rep = A.alpaca_matrix(progress=lambda f, m: print(f"\r  {f:5.1%} {m:60s}", end="", flush=True), symbol=a.symbol.upper())
    print("\r" + " " * 80 + "\r", end="")
    print(json.dumps(rep, indent=1, default=str) if a.json else A.render(rep))
    sys.exit(0 if rep["verified"] else 2 if rep["status"] == "blocked" else 1)


def cmd_usecase(a):
    """The original use case on the observations that are stored: qualifying sessions, what is priced, and a second independent computation."""
    from strata.research import usecase as U
    if getattr(a, "acquire", False):
        got = U.acquire(symbol=a.symbol.upper(), source=a.source, progress=lambda f, m: print(f"\r  {f:5.1%} {m[:70]:70s}", end="", flush=True))
        print("\r" + " " * 84 + "\r", end="")
        if got["status"] == "blocked":
            print(got["reason"])
            sys.exit(2)
        for v in got["variants"]:
            print(f"{v['dte']}DTE, sessions after a fall of {got['threshold_pct']:g}% or more inside {got['window'][0]}..{got['window'][1]}: {v['signals_in_window']} sessions, {v['contract_sessions']} contract-sessions "
                  f"({v['already_stored']} already stored, {v['downloaded_with_trades']} downloaded with trades, {v['downloaded_without_trades']} without, {v['failed']} failed; {v['bars'] or 0:,} bars)"
                  + (f"; notes: {v['notes'][:2]}" if v["notes"] else ""))
    rep = U.run(symbol=a.symbol.upper(), progress=lambda f, m: print(f"\r  {f:5.1%} {m:60s}", end="", flush=True))
    print("\r" + " " * 80 + "\r", end="")
    print(json.dumps(rep, indent=1, default=str) if a.json else U.render(rep))
    bad = [r for r in rep["thresholds"] if not r["lists_identical"] or r["counts"]["disagree"]]
    sys.exit(1 if bad else 0)


def cmd_options_audit(a):
    """Stored option trade bars: what is there session by session, a sample compared with the provider's own trades, and two providers against each other."""
    from strata.audit import options as AO
    sym = a.symbol.upper()
    prog = lambda f, m: print(f"\r  {f:5.1%} {m:60s}", end="", flush=True)
    aud = AO.audit_option_bars(sym=sym, provider=a.provider, progress=prog)
    ver = AO.verify_against_trades(sym=sym, sample=a.verify, progress=prog) if a.verify else AO.saved(sym)
    cross = AO.cross_provider(sym=sym)
    print("\r" + " " * 80 + "\r", end="")
    print(json.dumps({"audit": aud, "verification": ver, "cross_provider": cross}, indent=1, default=str) if a.json else AO.render(aud, ver, cross))
    sys.exit(2 if (ver or {}).get("status") == "blocked" and a.verify else 1 if (a.verify and not (ver or {}).get("verified")) else 0)


def cmd_bars_audit(a):
    """Stored intraday bars checked against the exchange calendar and the daily record."""
    from strata.audit import bars as AB
    bad = 0
    for sym in a.symbols or ["SPY"]:
        rep = AB.audit(sym=sym.upper())
        bad += rep["warnings"]
        print(json.dumps(rep, indent=1, default=str) if a.json else AB.render(rep))
    sys.exit(1 if bad and a.strict else 0)


def cmd_eval_nl(a):
    """How well requests are understood: 110 development and 40 held-out sentences through the application's own routing."""
    from strata.ai import evalset as EV
    from strata.ai.agent import Agent
    from strata.data.store import MarketStore
    ag = Agent(None, MarketStore()) if a.llm else Agent(None, MarketStore(), llm_factory=lambda: None)
    try:
        rep = EV.report(ag, llm=a.llm)
    except RuntimeError as e:
        print(f"NOT RUN: {e}")
        sys.exit(2)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        for k, title in (("development", "Development sentences (used to correct the readers)"), ("held_out", "Held-out sentences (now also used)")):
            r = rep[k]
            print(f"{title}: {r['correct']} of {r['cases']} correct ({r['accuracy_pct']}%), {r['flagged']} flagged, {r['silent']} silent  [{r['interpreter']}]")
            for f in r["failures"]:
                print(f"   [{f['status']}] {f['text'][:100]}\n      " + "; ".join(f["differences"][:3]))
        fr = rep["held_out_first_run"]
        print(f"Held-out, first run ({fr['date']}, before any fix): {fr['correct']} of {fr['cases']} correct ({fr['accuracy_pct']}%), {fr['silent']} silent. This is the estimate for an unseen sentence.")
    sys.exit(0 if rep["development"]["silent"] == 0 else 1)


def cmd_backup(a):
    from strata import backup as BK
    out = BK.backup(a.dest, market=not a.no_market, progress=lambda f, m: print(f"\r  {f:5.1%} {m:40s}", end="", flush=True))
    print(f"\rbackup written to {out['path']}: {out['files']:,} files, {out['bytes'] / 2 ** 20:,.1f} MB" + ("" if out["market_data"] else " (market data not included)") + ". Credentials are never part of a backup.")


def cmd_restore(a):
    from strata import backup as BK
    if a.verify:
        r = BK.verify(a.src)
        print(("verified" if r["ok"] else "DAMAGED") + f": {r['files']:,} files, created {r['created']}" + ("" if r["ok"] else "\n  " + "\n  ".join(r["damaged"] + [f"{x}: missing" for x in r["missing"]])))
        sys.exit(0 if r["ok"] else 1)
    try:
        r = BK.restore(a.src, force=a.force)
    except ValueError as e:
        print(f"not restored: {e}")
        sys.exit(1)
    print(f"restored {r['files']:,} files into {r['restored_to']}" + (f"; the previous database is kept as {r['previous_database_kept_as']}" if r["previous_database_kept_as"] else "") + f". {r['note']}")


def cmd_ingest(a):
    from strata.data.providers import get_provider
    p = get_provider(a.provider)
    print(p.ingest(a.symbol, date.fromisoformat(a.start), date.fromisoformat(a.end), interval=a.interval, include_options=not a.no_options,
                   progress=lambda f, m: print(f"\r  {f:5.1%} {m}", end="", flush=True)))


def cmd_validate(a):
    """End-to-end validation on actual market observations, from an empty data directory."""
    from strata.validate import run_validation
    rep = run_validation(a.dir, keep=a.keep or bool(a.dir), sessions=a.sessions)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["ok"] else 1)


def cmd_migrate(a):
    """Move a data directory written by an older version to the current storage layout."""
    from strata.config import get_settings
    from strata.data.migrate import migrate_layout
    root = get_settings().market_dir
    out = migrate_layout(root, dry_run=a.dry_run)
    print(f"{root}: layout {out['from']} -> {out['to']}" + (" (dry run, nothing moved)" if a.dry_run else "") + (f": {out['note']}" if out.get("note") else ""))
    for d in out.get("datasets", []):
        print(f"   {d['dataset']}: {d['files']} file(s)")
    if out.get("moved_files"):
        print(f"moved {out['moved_files']} files; a record of the migration is in {root / 'migrations'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strata", description="Strata options research platform")
    ap.add_argument("--data-dir", help="override STRATA_DATA_DIR")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create the database (add --fetch to also download free real market data)")
    p.add_argument("--fetch", action="store_true", help="download free daily + recent intraday history for SPY, QQQ, IWM and VIX")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("fetch", help="download free real market data (no account or key needed)")
    p.add_argument("symbols", nargs="*", default=["SPY", "QQQ", "IWM"])
    p.add_argument("--no-intraday", action="store_true", help="daily bars only")
    p.add_argument("--options", action="store_true", help="also download end-of-day option chains for SPY/QQQ/IWM (hundreds of MB)")
    p.add_argument("--years", help="option-chain years, e.g. 2020-2025 (default: the last five years)")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("sandbox", help="developer sandbox: generate SIMULATED symbols (DEMO/DEMOX) for testing without market data")
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
    p = sub.add_parser("validate", help="prove the whole chain on real data: empty directory -> free download -> manual vs AI strategy -> premiums checked against the source")
    p.add_argument("--dir", help="empty directory to use (kept afterwards); default: a temporary one that is removed")
    p.add_argument("--keep", action="store_true", help="keep the temporary data directory for inspection")
    p.add_argument("--sessions", type=int, default=3, help="recent sessions to backtest (default 3)")
    p.add_argument("--json", action="store_true", help="also print the full report as JSON")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("migrate", help="move stored market data to the current storage layout (also done automatically on start)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_migrate)
    p = sub.add_parser("import", help="import a CSV/Parquet file")
    p.add_argument("file")
    p.add_argument("--kind", required=True, choices=["underlying_minutes", "underlying_daily", "option_quotes", "events", "corporate_actions"])
    p.add_argument("--symbol")
    p.add_argument("--tz", default="America/New_York")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("bulk-import", help="import historical files of any size: CSV, ZIP or Parquet, a file or a whole folder; vendor layouts are recognised")
    p.add_argument("path")
    p.add_argument("--symbol", help="underlying symbol (only needed when the files do not name it)")
    p.add_argument("--format", help="force a layout id instead of detecting it")
    p.add_argument("--tz", default="America/New_York", help="timezone of timestamps that carry none")
    p.add_argument("--provider", default="import", help="name of the dataset the files form (default: import)")
    p.add_argument("--inspect", action="store_true", help="only report what would be imported")
    p.set_defaults(fn=cmd_bulk_import)
    p = sub.add_parser("collect", help="run the market-data collector: one cycle now, or stay running (it never trades)")
    p.add_argument("--once", action="store_true", help="run one collection cycle and exit (the default)")
    p.add_argument("--daemon", action="store_true", help="stay running and collect after each close while the collector is enabled")
    p.add_argument("--status", action="store_true", help="show the configuration, recent cycles and missed sessions")
    p.add_argument("--health", action="store_true", help="exit 0 if the daemon reported in the last five minutes (for container health checks)")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("bench", help="time realistic research workloads on a reproducible fixture and record the result")
    p.add_argument("--quick", action="store_true", help="a smaller fixture (for a fast sanity run; not comparable with the baseline)")
    p.add_argument("--real", action="store_true", help="also time read-only workloads on the real data directory")
    p.add_argument("--check", action="store_true", help="exit non-zero if a workload is slower than its baseline tolerance")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("providers", help="live check of what each data source actually delivers (depth, speed, entitlements)")
    p.add_argument("sources", nargs="*", help="yahoo cboe yahoo_options github_eod dolthub alpaca (default: all)")
    p.set_defaults(fn=cmd_providers)
    p = sub.add_parser("download", help="download from a free or keyed source outside the web app: 1-minute bars, or the bulk set of option trade bars (resumable; exit 2 = no key)")
    p.add_argument("source", help="alpaca, yahoo, yahoo_options, ...")
    p.add_argument("dataset", help="bars_1m, option_trades, ... (the ids the Data page shows for the source)")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--start", help="YYYY-MM-DD (default: the first date the source has)")
    p.add_argument("--end", help="YYYY-MM-DD (default: today)")
    p.add_argument("--band", type=float, default=2.0, help="option_trades: strikes within this percentage of the PREVIOUS close (default 2)")
    p.add_argument("--max-dte", type=int, default=7, help="option_trades: expirations up to this many calendar days out (default 7)")
    p.add_argument("--rights", choices=["both", "call", "put"], default="both")
    p.add_argument("--force", action="store_true", help="download again what is already stored")
    p.add_argument("--every", type=float, default=15.0, help="seconds between progress lines")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("access", help="test what the stored provider key can really retrieve: expired SPY option contracts across 2024-2026 (exit 2 = blocked, no key)")
    p.add_argument("provider", nargs="?", default="alpaca", choices=["alpaca"])
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_access)
    p = sub.add_parser("usecase", help="the 0DTE-calls-after-a-down-day strategy on stored observations: coverage, engine result and an independent recomputation (exit 1 if the two disagree)")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--json", action="store_true")
    p.add_argument("--acquire", action="store_true", help="first download, from the keyed free source, the contracts this strategy needs on its qualifying sessions (chosen around the price at the signal)")
    p.add_argument("--source", default="alpaca")
    p.set_defaults(fn=cmd_usecase)
    p = sub.add_parser("options-audit", help="check stored option trade bars: sessions missing, empty or thin; --verify N refetches N contract-sessions as individual trades and rebuilds the bars (exit 1 if they differ, 2 = no key)")
    p.add_argument("symbol", nargs="?", default="SPY")
    p.add_argument("--provider", help="default: the dataset the engine would use")
    p.add_argument("--verify", type=int, default=0, metavar="N", help="compare N stored contract-sessions with the provider's own trades (read-only requests)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_options_audit)
    p = sub.add_parser("bars-audit", help="check stored intraday bars: missing and partial sessions, hours, clock changes, official close, volume, mixed feeds")
    p.add_argument("symbols", nargs="*", help="default: SPY")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 if any warning is found")
    p.set_defaults(fn=cmd_bars_audit)
    p = sub.add_parser("eval-nl", help="measure how well requests are understood (150 sentences; --llm uses the configured language model)")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval_nl)
    p = sub.add_parser("backup", help="copy the database, the stored market data and the reports to a folder, with a checksum manifest (credentials are never included)")
    p.add_argument("dest", help="a folder OUTSIDE the data directory; a dated sub-folder is created in it")
    p.add_argument("--no-market", action="store_true", help="leave the market data out (database and reports only)")
    p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("restore", help="restore a backup after verifying every file against its manifest")
    p.add_argument("src", help="the backup folder (the one that contains MANIFEST.json)")
    p.add_argument("--verify", action="store_true", help="only check the backup; restore nothing")
    p.add_argument("--force", action="store_true", help="restore into a data directory that already holds data")
    p.set_defaults(fn=cmd_restore)
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
    from strata.core import instance
    if a.cmd != "serve":
        # a command waits out a lease whose holder may have gone; the server goes on standby instead
        instance.DEFAULT_WAIT = float(os.environ.get("STRATA_INSTANCE_WAIT") or 25.0)
        instance.configure(role="collector" if a.cmd == "collect" else "command")
    try:
        health = a.cmd == "collect" and getattr(a, "health", False)
        if a.cmd == "collect" and getattr(a, "daemon", False):
            # a daemon has nothing else to do: it waits on standby until the copy that holds the data directory has gone
            told = False
            while True:
                try:
                    instance.ensure(wait=0.0)
                    break
                except instance.Busy as e:
                    if not told:
                        print(f"STANDBY. {e}", flush=True)
                        told = True
                    hb = __import__("strata.config", fromlist=["get_settings"]).get_settings().data_dir / "collector.heartbeat"
                    try:  # alive and waiting: the container's health check must not restart it
                        hb.parent.mkdir(parents=True, exist_ok=True)
                        hb.write_text("standby")
                    except OSError:
                        pass
                    time.sleep(5.0)
        elif a.cmd not in ("serve", "validate", "bench") and not health:  # those two work in a directory of their own; the server goes on standby by itself
            instance.ensure()  # before anything is read or written: one machine at a time on a data directory
        a.fn(a)
    except instance.Busy as e:
        hint = (f"The command can be run inside that copy: docker compose exec strata strata {a.cmd} ..." if "container" in str(e.holder.get("where", ""))
                else "Run the command where that copy is running (on the computer itself, outside the container).")
        print(f"\nNOT RUN. {e}\n{hint}", file=sys.stderr)
        sys.exit(3)
    finally:
        if a.cmd != "serve":
            instance.release()


if __name__ == "__main__":
    main(sys.argv[1:])
