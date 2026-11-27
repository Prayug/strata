"""Research performance benchmarks: realistic workloads, measured and kept.

`strata bench` builds a deterministic fixture (a simulated two-symbol option market plus a
long daily history) in a temporary directory and times the workloads research actually
runs: ingestion, loading cached data, a single backtest, a parameter sweep, hypothesis
plans (vectorised) against the session-by-session study engine, a multi-symbol study, a
streaming import, and concurrent jobs. It also records peak memory and storage.

Every run is appended to `benchmarks/history.jsonl` with the commit it was measured on.
`--check` compares the run with `benchmarks/baseline.json` and exits non-zero when a
workload slowed down more than its tolerance compared with the rest of the same run (the
run's median slowdown measures the machine, so a slower or faster machine is not reported
as a regression), or when the whole suite became slower than its tolerance after
normalising by a fixed calibration loop. `--update-baseline` rewrites the baseline from the
current run.

The fixture is simulated on purpose: benchmarks must be reproducible offline and must not
depend on what the user happens to have downloaded. `--real` additionally times the
read-only workloads on the real data directory.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import date, datetime, timezone
from pathlib import Path

# a workload may slow down this many times more than the rest of its run, and the whole suite this many times
# (normalised by the calibration loop), before --check fails; CI runs on different hardware from the machine that
# recorded the baseline, so it sets a wider tolerance
TOLERANCE = float(os.environ.get("STRATA_BENCH_TOLERANCE", "1.75"))
FLOOR_SECONDS = 0.05  # workloads faster than this are too noisy to gate on
MIN_FOR_MEDIAN = 5  # with fewer gated workloads a run's median says little about the machine: the calibration loop alone


def bench_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists() or (root / "benchmarks").exists():
        return root / "benchmarks"
    from strata.config import get_settings
    return get_settings().data_dir / "benchmarks"


def _commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _rss_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def calibrate() -> float:
    """Seconds for a fixed numerical workload: the unit other timings are expressed in."""
    import numpy as np
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        rng = np.random.default_rng(1)
        x = rng.standard_normal((400, 400))
        for _ in range(12):
            x = np.tanh(x @ x.T / 400)
        np.sort(rng.standard_normal(2_000_000))
        best = min(best, time.perf_counter() - t0)
    return best


class _Timer:
    def __init__(self):
        self.rows: list[dict] = []

    def run(self, name: str, fn, repeat: int = 1, **info) -> object:
        best, out = 1e18, None
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        extra = out if isinstance(out, dict) and out.get("_info") else {}
        self.rows.append({"name": name, "seconds": round(best, 4), **info, **{k: v for k, v in extra.items() if k != "_info"}})
        return out


def _fixture(root: Path, quick: bool):
    """A simulated option market (two symbols with chains) and a long daily history."""
    import numpy as np
    import polars as pl

    from strata.core import calendar as cal
    from strata.data.store import MarketStore, SymbolMeta
    from strata.data.synthetic import SyntheticConfig, generate_demo_dataset
    store = MarketStore(root / "market") if not (root / "market").exists() else MarketStore(root / "market")
    cfg = SyntheticConfig(start=date(2024, 1, 2), end=date(2024, 2, 9) if quick else date(2024, 3, 28), seed=7, option_interval=30)
    t0 = time.perf_counter()
    info = generate_demo_dataset(store, cfg, log=lambda *a: None)
    gen = time.perf_counter() - t0
    rng = np.random.default_rng(42)
    days = cal.trading_days(date(2008, 1, 2), date(2023, 12, 29))
    for sym, vol in (("BENCH", 0.011), ("BENCHB", 0.014)):
        r = rng.standard_t(4, len(days)) * vol * 0.7
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[100.0, c[:-1]] * np.exp(rng.standard_normal(len(days)) * 0.003)
        hi = np.maximum(o, c) * (1 + np.abs(rng.standard_normal(len(days))) * 0.004)
        lo = np.minimum(o, c) * (1 - np.abs(rng.standard_normal(len(days))) * 0.004)
        store.write_daily(sym, pl.DataFrame({"date": days, "open": o, "high": hi, "low": lo, "close": c, "volume": rng.integers(1e6, 9e6, len(days)).astype(float)}), provider="synthetic")
        store.upsert_symbol(SymbolMeta(symbol=sym, vol_index="BENCHVIX", synthetic=True, has_options=False))
    v = 18 + np.abs(np.cumsum(rng.standard_normal(len(days)) * 0.6)) % 25
    store.write_daily("BENCHVIX", pl.DataFrame({"date": days, "open": v, "high": v + 1, "low": v - 1, "close": v, "volume": np.zeros(len(days))}), provider="synthetic")
    store.upsert_symbol(SymbolMeta(symbol="BENCHVIX", kind="volatility_index", synthetic=True, has_options=False))
    for s in ("BENCH", "BENCHB", "BENCHVIX"):
        store.rebuild_manifest(s)
    return store, {"generate_seconds": gen, "sessions": len(cal.trading_days(cfg.start, cfg.end)), "info": {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str))}}


def _bt_config(symbol: str, start: date, end: date) -> dict:
    from strata.engine.templates import entry_at, exits, leg
    return {"strategies": [{"name": "bench put credit spread", "symbol": symbol, "legs": [leg("short", "sell", "P", 7, "delta", 0.25), leg("long", "buy", "P", 7, "leg_offset", -10, ref="short")],
                            "entry": entry_at("10:00"), "exits": exits(50, 200)}], "start": start.isoformat(), "end": end.isoformat(), "allow_synthetic": True}


def _concurrent_bt(cfg: dict):
    from strata.engine.engine import run_backtest
    from strata.engine.models import BacktestConfig
    return len(run_backtest(BacktestConfig.model_validate(cfg), record_paths=False).trades)


def _init_worker(data_dir: str):
    from strata.config import override_settings
    override_settings(data_dir=data_dir)


def run_benchmarks(quick: bool = False, real: bool = False, out=print) -> dict:
    """Run the suite and return the record that is appended to the history."""
    from strata.config import ENGINE_VERSION, get_settings, override_settings
    real_dir = get_settings().data_dir
    cal_s = calibrate()
    tm = _Timer()
    rec: dict = {"at": datetime.now(timezone.utc).isoformat(), "commit": _commit(), "engine_version": ENGINE_VERSION, "python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}",
                 "cpus": os.cpu_count(), "calibration_seconds": round(cal_s, 4), "quick": quick}
    with tempfile.TemporaryDirectory(prefix="strata-bench-") as td:
        root = Path(td)
        override_settings(data_dir=root)
        try:
            from strata.api import db
            db.init_db()
            out("building the fixture (simulated option market + 16 years of daily bars)")
            store, fx = _fixture(root, quick)
            tm.rows.append({"name": "cold_ingest", "seconds": round(fx["generate_seconds"], 4), "what": f"generate and store {fx['sessions']} sessions of intraday bars and option chains for two symbols"})
            from strata.data.store import MarketStore
            from strata.engine.engine import run_backtest
            from strata.engine.market import UnderlyingSeries
            from strata.engine.models import BacktestConfig
            from strata.research import features as FT
            from strata.research import hypothesis as HY
            from strata.research.study import run_study
            days = store.option_days("DEMO")
            a, b = days[0], days[-1]

            def cached():
                s2 = MarketStore(store.root)
                df = s2.read_best_bars("DEMO", None, None)
                UnderlyingSeries(df, official=s2.official_closes("DEMO", None, None))
                s2.read_option_day("DEMO", days[len(days) // 2])
                return {"_info": True, "bars": df.height}
            out("cached-data loading")
            tm.run("cached_load", cached, repeat=3, what="read stored bars, build the price series, read one session's option chain")

            cfg = _bt_config("DEMO", a, b)
            out("single backtest")
            res = tm.run("single_backtest", lambda: run_backtest(BacktestConfig.model_validate(cfg), store, record_paths=False), what=f"put credit spread, {len(days)} sessions of 30-minute chains")
            tm.rows[-1]["trades"] = len(res.trades)

            def sweep():
                from strata.research.sensitivity import set_param
                n = 0
                for dte in (3, 7, 14):
                    for delta in (0.15, 0.25, 0.35, 0.45):
                        c = set_param(set_param(cfg, {"path": "legs.*.dte"}, dte), {"path": "exits.profit_target"}, 50)
                        c["strategies"][0]["legs"][0]["strike"]["value"] = delta
                        n += len(run_backtest(BacktestConfig.model_validate(c), store, record_paths=False).trades)
                return {"_info": True, "backtests": 12, "trades": n}
            if not quick:
                out("parameter sweep (12 backtests)")
                tm.run("sweep_12", sweep, what="12 backtests over expiration and delta, one process")

            FT._CACHE.clear()
            out("hypothesis plan (vectorised)")
            plan = HY.build_plan("Investigate whether BENCH tends to reverse following large down days.", "BENCH", store, ["BENCH", "BENCHB"])
            tm.run("feature_table_build", lambda: FT.SessionFeatures(store, "BENCH", ("BENCHB",)), repeat=2, what="load 16 years of daily bars and build the session feature table")
            r = tm.run("hypothesis_plan", lambda: HY.run_plan(plan, store), what=f"{len(plan['configs'])} configurations with calendar-cluster tests on 16 years of sessions")
            tm.rows[-1]["configurations"] = r["tested"]
            scan = HY.build_plan("What historical conditions are associated with a 1% rally before a 0.5% decline?", "BENCH", store, ["BENCH", "BENCHB"])
            tm.run("condition_scan", lambda: HY.run_plan(scan, store), what=f"{len(scan['configs'])} single-condition configurations")

            row = next(x for x in r["configs"] if x.get("study"))
            out("session-by-session study engine (the reference implementation)")
            tm.run("study_engine_one", lambda: run_study(row["study"], store, max_sessions_out=0), what="one configuration evaluated session by session through the rule engine")
            ft = FT.get_features(store, "BENCH", ("BENCHB",))
            preds, oc = [FT.Pred.load(p) for p in row["preds"]], FT.OutcomeSpec.load(row["outcome"])
            ft._outcomes.clear()
            tm.run("vectorised_one", lambda: (ft._outcomes.clear(), ft.evaluate(preds, oc))[1], repeat=3, what="the same configuration from the session feature table")
            se, ve = tm.rows[-2]["seconds"], max(tm.rows[-1]["seconds"], 1e-6)
            rec["vectorised_speedup"] = round(se / ve, 1)

            def multi():
                n = 0
                for s in ("BENCH", "BENCHB"):
                    n += run_study({**row["study"], "symbol": s}, store, max_sessions_out=0)["stats"]["qualifying_sessions"]
                return {"_info": True, "symbols": 2, "qualifying": n}
            out("multi-symbol study")
            tm.run("multi_symbol_study", multi, what="the same study on two symbols")

            def imp():
                import polars as pl

                from strata.data import bulk
                src = store.read_option_day("DEMO", days[0])
                rows = pl.concat([src.with_columns(pl.col("ts") + pl.duration(days=k * 7)) for k in range(2 if quick else 8)])
                rows = rows.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None).alias("quote_datetime"), "expiration", "strike", pl.col("right").alias("option_type"), "bid", "ask",
                                   pl.lit("IMPB").alias("underlying_symbol"))
                f = root / "import.csv"
                rows.write_csv(f)
                tracemalloc.start()
                rep = bulk.import_path(str(f), provider="bench_import", store=store)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                return {"_info": True, "rows": rep["rows_in"], "rows_per_second": None, "peak_python_mb": round(peak / 1e6, 1), "file_mb": round(f.stat().st_size / 1e6, 1)}
            out("streaming import")
            tm.run("bulk_import", imp, what="stream a CSV of option quotes into per-session files")
            last = tm.rows[-1]
            last["rows_per_second"] = round(last["rows"] / max(last["seconds"], 1e-6))

            # ----- V5: composed research graphs, the pattern explorer, the IDE's debugger, ML research
            from strata.ai import nl_graph as NG
            from strata.research import graph as GR
            from strata.research import patterns as PT
            out("research graph (question -> typed graph -> evidence, matched comparison, robustness, validation)")
            question = "Does BENCH tend to recover after down days of 1.5% or more, even after controlling for volatility?"
            tm.run("graph_plan", lambda: NG.plan(question, ["BENCH", "BENCHB"], "BENCH", store), repeat=3, what="read a question into a validated research graph")
            g = NG.plan(question, ["BENCH", "BENCHB"], "BENCH", store)
            gres = tm.run("graph_run", lambda: GR.run_graph(g, store, scope=None), what=f"run its {len(g['nodes'])} steps on 16 years of sessions")
            tm.rows[-1]["analyses"] = sum(1 for n in gres["nodes"] if n.get("result"))
            out("pattern explorer")
            pres = tm.run("pattern_explore", lambda: PT.explore("BENCH", {"id": "down_day", "x": 1.5}, store, horizon=10), repeat=2, what="every occurrence of an event with forward paths, horizons and regimes")
            tm.rows[-1]["occurrences"] = pres["n"]

            def dbg():
                from strata.api import ide as IDE
                st_cfg = {**cfg["strategies"][0], "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": "from_open_pct < -0.2"}}]}}}
                r_ = IDE.debug_session(st_cfg, days[len(days) // 2], store, ["price", "from_open_pct"], "from_open_pct < -0.5")
                return {"_info": True, "ticks": len(r_.get("timeline") or [])}
            out("condition debugger")
            tm.run("ide_debug_session", dbg, what="replay one session's entry rule tick by tick with two watches and a breakpoint")
            if not quick:
                from strata.research import ml as ML
                if ML.available()[0]:
                    out("ML study (purged cross-validation, placebo, leakage probe, walk-forward signal)")
                    spec = {"symbol": "BENCH", "outcome": {"entry": "close", "horizon": "sessions:5", "direction": "up", "target": 0.0, "stop": None, "measure": "close"}, "models": ["logistic", "tree"]}
                    mres = tm.run("ml_study", lambda: ML.run(spec, store), what="two models on 16 years of sessions, scored out of sample")
                    tm.rows[-1]["rows"] = mres["n"]
                else:
                    rec["ml_skipped"] = ML.available()[1]

            if not quick:
                out("concurrent jobs")
                from concurrent.futures import ProcessPoolExecutor
                from multiprocessing import get_context
                t0 = time.perf_counter()
                for _ in range(3):
                    _concurrent_bt(cfg)
                seq = time.perf_counter() - t0
                t0 = time.perf_counter()
                with ProcessPoolExecutor(3, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(str(root),)) as pool:
                    list(pool.map(_concurrent_bt, [cfg] * 3))
                par = time.perf_counter() - t0
                tm.rows.append({"name": "concurrent_jobs_3", "seconds": round(par, 4), "sequential_seconds": round(seq, 4), "speedup": round(seq / par, 2), "what": "three backtests in three worker processes (including process start-up)"})
            rec["storage_mb"] = round(store.storage_bytes() / 1e6, 1)
        finally:
            override_settings(data_dir=real_dir)
    if real:
        try:
            from strata.data.store import MarketStore
            from strata.research import features as FT
            from strata.research import hypothesis as HY
            st = MarketStore()
            sym = "SPY" if "SPY" in st.stored_symbols() else sorted(st.stored_symbols())[0]
            FT._CACHE.clear()
            tm.run("real_feature_table", lambda: FT.SessionFeatures(st, sym, ("QQQ", "IWM")), what=f"real data: {sym} feature table")
            p = HY.build_plan(f"Investigate whether {sym} tends to reverse following large down days.", sym, st, sorted(st.stored_symbols()))
            tm.run("real_hypothesis_plan", lambda: HY.run_plan(p, st), what=f"real data: {len(p['configs'])} configurations on {sym}")
        except Exception as e:  # noqa: BLE001
            rec["real_error"] = str(e)[:200]
    rec["peak_rss_mb"] = round(_rss_mb(), 1)
    for r in tm.rows:
        r["normalised"] = round(r["seconds"] / cal_s, 3)
    rec["workloads"] = tm.rows
    return rec


def assess(rec: dict, baseline: dict, tolerance: float = TOLERANCE) -> dict:
    """Which workloads slowed down more than the tolerance, and whether the whole suite did.

    A workload is measured against the rest of its own run: its time over the baseline's, divided by the median of that
    ratio over every gated workload (how much slower this machine runs the suite). The calibration loop alone misjudged
    shared runners: on the V9 push one ran the loop in 0.061 s where others took 0.12 to 0.18 s, every normalised ratio
    rose by about a third, and a backtest as fast as V8's was reported as a regression (3.04x against 3.0).
    The whole suite is measured against the calibration loop (the median normalised ratio), which catches what a
    comparison within the run cannot: everything slowing down together.
    `ratio` stays the calibration-normalised ratio of earlier versions; `relative` is the one gated on.
    """
    base = {w["name"]: w for w in baseline.get("workloads", [])}
    rows, raw = [], {}
    for w in rec["workloads"]:
        b = base.get(w["name"])
        if not b or w["name"].startswith("real_"):
            continue
        ratio = w["normalised"] / b["normalised"] if b.get("normalised") else None
        gate = max(b["seconds"], w["seconds"]) >= FLOOR_SECONDS
        raw[w["name"]] = w["seconds"] / b["seconds"] if b.get("seconds") else None
        rows.append({"name": w["name"], "seconds": w["seconds"], "baseline_seconds": b["seconds"], "ratio": ratio, "gated": gate})
    gated = [r for r in rows if r["gated"] and raw[r["name"]] and r["ratio"] is not None]
    machine = statistics.median(raw[r["name"]] for r in gated) if len(gated) >= MIN_FOR_MEDIAN else None
    suite = statistics.median(r["ratio"] for r in gated) if machine else None
    for r in rows:
        rel = (raw[r["name"]] / machine if raw[r["name"]] is not None else None) if machine else r["ratio"]
        r["relative"] = None if rel is None else round(rel, 2)
        r["ratio"] = None if r["ratio"] is None else round(r["ratio"], 2)
        r["regressed"] = bool(r["gated"] and rel is not None and rel > tolerance)
    return {"workloads": rows, "tolerance": tolerance, "basis": "the run's median" if machine else "the calibration loop",
            "machine": None if machine is None else round(machine, 2), "suite_ratio": None if suite is None else round(suite, 2),
            "suite_regressed": bool(suite is not None and suite > tolerance)}


def compare(rec: dict, baseline: dict, tolerance: float = TOLERANCE) -> list[dict]:
    """The per-workload rows of assess() (the shape callers have used since V4)."""
    return assess(rec, baseline, tolerance)["workloads"]


def save(rec: dict) -> Path:
    d = bench_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return d / "history.jsonl"


def history(limit: int = 30) -> dict:
    d = bench_dir()
    rows = []
    p = d / "history.jsonl"
    if p.exists():
        for line in p.read_text().splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    base = json.loads((d / "baseline.json").read_text()) if (d / "baseline.json").exists() else None
    fe = []
    fp = d / "frontend.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines()[-limit:]:
            try:
                fe.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    a = assess(rows[-1], base) if rows and base else None
    return {"runs": rows, "baseline": base, "frontend": fe, "tolerance": TOLERANCE, "directory": str(d), "comparison": a["workloads"] if a else [],
            "suite": {k: v for k, v in a.items() if k != "workloads"} if a else None}


def main(quick=False, real=False, check=False, update_baseline=False, as_json=False) -> int:
    rec = run_benchmarks(quick, real, out=(lambda *a: None) if as_json else print)
    d = bench_dir()
    base_p = d / "baseline.json"
    base = json.loads(base_p.read_text()) if base_p.exists() else None
    a = assess(rec, base) if base else None
    cmp_ = a["workloads"] if a else []
    rec["comparison"] = cmp_
    if a:
        rec["suite"] = {k: v for k, v in a.items() if k != "workloads"}
    save(rec)
    if update_baseline or (base is None and not quick):
        d.mkdir(parents=True, exist_ok=True)
        base_p.write_text(json.dumps({k: rec[k] for k in ("at", "commit", "engine_version", "calibration_seconds", "machine", "workloads")}, indent=1))
    if as_json:
        print(json.dumps(rec, indent=1))
    else:
        print(f"\ncalibration {rec['calibration_seconds']:.3f}s   peak memory {rec['peak_rss_mb']:.0f} MB   fixture storage {rec.get('storage_mb')} MB   vectorised vs session engine: {rec.get('vectorised_speedup')}x")
        by = {c["name"]: c for c in cmp_}
        for w in rec["workloads"]:
            c = by.get(w["name"])
            tail = "" if not c or c["ratio"] is None else f"   x{c['ratio']:.2f} vs baseline" + (f", x{c['relative']:.2f} vs this run" if a["machine"] and c["relative"] is not None else "") + ("   REGRESSION" if c["regressed"] else "")
            print(f"  {w['name']:22s} {w['seconds']:8.3f}s{tail}   {w.get('what', '')}")
        if a and a["machine"]:
            print(f"\nthis machine ran the suite at x{a['machine']:.2f} the baseline's times (median of the gated workloads); each workload is "
                  f"gated against that, the whole suite against the calibration loop (x{a['suite_ratio']:.2f}); tolerance x{a['tolerance']:.2f}")
        print(f"\nrecorded in {d / 'history.jsonl'}" + (f"; baseline written to {base_p}" if update_baseline or base is None else ""))
    bad = [f"{c['name']} slowed down {c['relative']}x more than the rest of the suite" if a["machine"] else f"{c['name']} is {c['ratio']}x its baseline"
           for c in cmp_ if c["regressed"]]
    if a and a["suite_regressed"]:
        bad.append(f"the whole suite is {a['suite_ratio']}x its baseline after the calibration loop")
    if check and bad:
        print("performance regression: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0
