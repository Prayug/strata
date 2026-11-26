"""Benchmarks on the stored dataset (`strata bench --dataset`).

The fixture benchmarks in `strata.bench` are reproducible anywhere and gate regressions. These measure the thing a person
waits for: research on the option history that is actually on this disk. They only read it. Each workload runs in a process
of its own, started fresh, so that its time includes loading what it needs and its peak memory is its own; "again" is the
same workload a second time in that process, with the data-frame and file caches warm.

Wall time on a shared machine is noisy, so every row carries the CPU seconds it used as well. Nothing here is compared with
a baseline or fails a check: the numbers describe this machine and this dataset on the day they were taken, and they are
appended to `benchmarks/dataset_history.jsonl` so that a change can be seen against earlier days.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ATM = {"method": "atm_offset", "value": 0.0}
NO_PT = {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}}


def _strategy(sym: str, name: str, legs: list[dict], entry: dict, exits: dict) -> dict:
    return {"name": name, "symbol": sym, "legs": legs, "entry": entry, "exits": {**NO_PT, **exits}}


def _at(times: str, *exprs: str) -> dict:
    return {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": times}}] + [{"kind": "cond", "type": "expr", "params": {"expr": e}} for e in exprs]}}


def call_every_session(sym: str) -> dict:
    """An at-the-money same-day call bought at 10:00 and sold at 15:30 on every session: the engine prices a trade a day."""
    return _strategy(sym, f"{sym} 0DTE call every session", [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": ATM}], _at("10:00"), {"time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}})


def put_spread(sym: str) -> dict:
    """A put credit spread a week out, sold after a down day and held for days: two legs, positions open across sessions."""
    return _strategy(sym, f"{sym} 7DTE put credit spread after a down day", [{"id": "sp", "side": "sell", "right": "P", "dte": 7, "strike": {"method": "delta", "value": 0.30}},
                                                                             {"id": "lp", "side": "buy", "right": "P", "dte": 7, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}}],
                     _at("10:00", "previous_session.return_pct <= -0.5"), {"profit_target": {"enabled": True, "mode": "pct", "value": 50}, "stop_loss": {"enabled": True, "mode": "pct", "value": 200}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": 1}})


def vwap_reclaim(sym: str) -> dict:
    """A stateful rule checked every minute: down 0.5% from the open, then back through VWAP; a 30% premium target against a 15% stop."""
    entry = {"conditions": {"kind": "group", "op": "and", "children": []}, "sequence": [{"name": "drop", "when": "from_open_pct <= -0.5"}, {"name": "reclaim", "when": "crosses_above(price, vwap)"}]}
    return _strategy(sym, f"{sym} 0DTE call on a VWAP reclaim after a 0.5% decline", [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": ATM}], entry,
                     {"profit_target": {"enabled": True, "mode": "pct", "value": 30}, "stop_loss": {"enabled": True, "mode": "pct", "value": 15}, "time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}})


def config(strategies: list[dict], start: str, end: str, provider: str) -> dict:
    return {"start": start, "end": end, "pricing": {"mode": "trades"}, "data": {"options_provider": provider}, "execution": {"signal_interval_minutes": 1}, "strategies": strategies}


# ----------------------------------------------------------------------------- one workload in one fresh process
def _usage() -> tuple[float, float]:
    r, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = max(r.ru_maxrss, c.ru_maxrss)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime, (rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024)


def _backtest(cfg: dict) -> dict:
    from strata.data.store import MarketStore
    from strata.engine.engine import run_backtest
    res = run_backtest(cfg, store=MarketStore(), record_paths=False)
    ex = res.meta.get("executability") or {}
    return {"trades": len(res.trades), "signals": ex.get("signals"), "priced": ex.get("executed"), "net": round(sum(float(t["pnl"]) for t in res.trades), 2)}


def _sweep(args: dict) -> dict:
    from strata.ai import nl_experiment as NX
    from strata.ai import nl_rules as NL
    from strata.ai.nl_stateful import enrich
    from strata.data.store import MarketStore
    from strata.research import experiment as EXP
    st = MarketStore()
    base = enrich(f"Buy ATM {args['symbol']} 0DTE calls at 10:00 AM Eastern after a 0.5% down day", NL.TICKERS)
    base.start, base.end = date.fromisoformat(args["start"]), date.fromisoformat(args["end"])
    plan = NX.read("Compare entry times from 9:45 to 10:45 in 15-minute increments, exits after 15, 30, 60, 120 and 240 minutes, 0DTE versus 1DTE and ATM versus 1% OTM contracts.", base, sorted(st.stored_symbols()))
    res = EXP.run_experiment(EXP.build(plan, st), st)
    sm = res["summary"]
    return {"configurations": sm["configurations"], "with_trades": sm.get("with_trades"), "entry_rules": sm["timing"]["entry_rules"], "worker_cpu_seconds": sm["timing"]["cpu_seconds"], "trades": sum(c["metrics"]["trades"] for c in res["cells"])}


def _replay(args: dict) -> dict:
    from strata.data.store import MarketStore
    from strata.research import replay as RP
    st = MarketStore()
    d = date.fromisoformat(args["date"])
    t = {}
    t0 = time.perf_counter(); u = RP.underlying(st, args["symbol"], d); t["session_seconds"] = round(time.perf_counter() - t0, 4)
    t0 = time.perf_counter(); c = RP.contracts(st, args["symbol"], d); t["contract_list_seconds"] = round(time.perf_counter() - t0, 4)
    occ = next((r["occ"] for e in c["expirations"] for r in e["rows"] if r["traded"] and e["dte"] == 0 and r["right"] == "C" and abs(r["strike"] - (u["sessions"][0]["open"] or 0)) < 1.01), None)
    if occ:
        t0 = time.perf_counter(); k = RP.contract(st, args["symbol"], occ, d); t["contract_seconds"] = round(time.perf_counter() - t0, 4)
        t["contract_bars"] = len(k["bars"])
    if args.get("run"):
        try:
            t0 = time.perf_counter(); tr = RP.trade(args["run"], 1, st); t["trade_overlay_seconds"] = round(time.perf_counter() - t0, 4)
            t["trade_payload_kb"] = round(len(json.dumps(tr)) / 1024, 1)
        except Exception as e:  # noqa: BLE001
            t["trade_overlay"] = f"not measured ({type(e).__name__})"
    t0 = time.perf_counter(); span = RP.underlying(st, args["symbol"], date(d.year, 1, 2), d); t["year_span_seconds"] = round(time.perf_counter() - t0, 4)
    return {**t, "candles": len(u["candles"]), "contracts": sum(e["contracts"] for e in c["expirations"]), "year_span_candles": len(span["candles"]), "year_span_candle_seconds": span["candle_seconds"]}


def _hires(args: dict) -> dict:
    """The check on individual trades for a stored run, from the provider cache only: what the check costs once the prints are here."""
    from strata.api import db
    from strata.api.results import load_trades
    from strata.data.store import MarketStore
    from strata.engine import hires as H
    run = db.get_run(args["run"])
    trades = load_trades(args["run"])
    src = H.AlpacaTrades(MarketStore())
    rep = H.replay(trades, run["config"], src)
    return {"positions": len(trades), "repriced": (rep.get("summary") or {}).get("repriced"), "provider_requests": src.requests, "from_cache": src.requests == 0}


JOBS = {"backtest": _backtest, "sweep": _sweep, "replay": _replay, "hires": _hires}


def _child(kind: str, arg, conn, twice: bool) -> None:
    os.environ.setdefault("STRATA_INSTANCE_GUARD", "0")  # read-only: no lease is taken and nothing is written to the data directory
    try:
        t0, c0 = time.perf_counter(), _usage()[0]
        info = JOBS[kind](arg)
        first, cpu1 = time.perf_counter() - t0, _usage()[0] - c0
        again = None
        if twice:
            t1 = time.perf_counter()
            JOBS[kind](arg)
            again = time.perf_counter() - t1
        conn.send({"ok": True, "seconds": round(first, 3), "cpu_seconds": round(cpu1, 3), "again_seconds": None if again is None else round(again, 3), "peak_rss_mb": round(_usage()[1], 1), **(info or {})})
    except Exception as e:  # noqa: BLE001
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    finally:
        conn.close()


def measure(kind: str, arg, twice: bool = False) -> dict:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(kind, arg, b, twice))
    t0 = time.perf_counter()
    p.start()
    b.close()
    try:
        out = a.recv()
    except EOFError:
        out = {"ok": False, "error": "the workload's process ended without a result"}
    p.join()
    out["process_seconds"] = round(time.perf_counter() - t0, 3)  # with interpreter start and imports: what a job run from cold costs
    return out


def run(out=print, quick: bool = False) -> dict:
    """Every workload once. `quick` uses the most recent year instead of the whole stored period."""
    from strata.config import ENGINE_VERSION, get_settings
    from strata.data.store import MarketStore
    st = MarketStore()
    have = {}
    for sym in ("SPY", "QQQ", "IWM"):
        ds = st.option_dataset(sym, "trade", None)
        idx = st.option_bar_index(sym, dataset=ds) if ds is not None else None
        if idx is not None and not idx.is_empty():
            have[sym] = {"provider": ds.provider, "first": idx["date"].min().isoformat(), "last": idx["date"].max().isoformat(), "sessions": int(idx["date"].n_unique()), "contract_sessions": idx.height, "bars": int(idx["bars"].sum())}
    if not have:
        return {"status": "nothing to measure", "note": "No option trade observations are stored. Download some first (Data page, or `strata download alpaca option_trades SPY`)."}
    rec: dict = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "engine_version": ENGINE_VERSION, "machine": f"{platform.system()} {platform.machine()}", "cpus": os.cpu_count(), "quick": quick,
                 "load_average_at_start": [round(x, 2) for x in os.getloadavg()], "dataset": have, "observations": sum(v["bars"] for v in have.values()), "workloads": []}
    first_sym = next(iter(have))

    def window(sym: str) -> tuple[str, str]:
        a, b = have[sym]["first"], have[sym]["last"]
        return (max(a, f"{b[:4]}-01-01"), b) if quick else (a, b)

    def add(name: str, what: str, kind: str, arg, twice: bool = False) -> dict:
        out(f"  {name} ...")
        r = measure(kind, arg, twice)
        row = {"name": name, "what": what, **r}
        rec["workloads"].append(row)
        out(f"    {row.get('seconds', '-')}s wall, {row.get('cpu_seconds', '-')}s cpu" + (f", again {row['again_seconds']}s" if row.get("again_seconds") is not None else "") + (f", {row.get('peak_rss_mb')} MB" if row.get("peak_rss_mb") else "")
            + ("" if row.get("ok") else f"  FAILED: {row.get('error')}"))
        return row

    for sym in have:
        a, b = window(sym)
        add(f"full_period_{sym}", f"{sym}: a same-day call every session, {a} to {b} ({have[sym]['sessions'] if not quick else 'this year'} sessions)", "backtest", config([call_every_session(sym)], a, b, have[sym]["provider"]), twice=(sym == first_sym))
    if len(have) > 1:
        a, b = max(window(s)[0] for s in have), min(window(s)[1] for s in have)
        add("several_symbols_one_run", f"{', '.join(have)} in one run: one account, a strategy per symbol, {a} to {b}", "backtest", config([call_every_session(s) for s in have], a, b, have[first_sym]["provider"]))
    a, b = window(first_sym)
    add("multi_leg_held_for_days", f"{first_sym}: a two-leg credit spread a week out, held across sessions, {a} to {b}", "backtest", config([put_spread(first_sym)], a, b, have[first_sym]["provider"]))
    add("stateful_rule_every_minute", f"{first_sym}: an ordered rule (decline, then VWAP reclaim) checked every minute with a premium target and stop, {a} to {b}", "backtest", config([vwap_reclaim(first_sym)], a, b, have[first_sym]["provider"]))
    ya, yb = max(have[first_sym]["first"], f"{have[first_sym]['last'][:4]}-01-01"), have[first_sym]["last"]
    add("sweep_100_configurations", f"{first_sym}: 100 configurations of one strategy (5 entry times x 5 exits x 2 expirations x 2 strikes), {ya} to {yb}, in worker processes", "sweep", {"symbol": first_sym, "start": ya, "end": yb})
    from strata.api import db
    runs = [r for r in db.list_runs() if r.get("kind") == "backtest" and r.get("status") == "done"]
    hr = next((r["id"] for r in runs if (get_settings().results_dir / r["id"] / "hires.json").exists()), None)
    add("replay_loading", f"{first_sym} {have[first_sym]['last']}: the session's bars and levels, its contract list, one contract's prints, one trade laid over them, and a year of candles", "replay",
        {"symbol": first_sym, "date": have[first_sym]["last"], "run": hr or (runs[0]["id"] if runs else None)})
    if hr:
        add("individual_trade_check", "the execution check on individual trades for a stored run, from the provider cache (no request is made when every window is cached)", "hires", {"run": hr})
    else:
        rec["individual_trade_check"] = "not measured: no stored run has been checked on individual trades yet"
    by = {}
    for key in st.datasets():
        by[f"{key.symbol} {key.provider} {key.dtype} {key.obs} {key.res}"] = st.storage_bytes(key=key)
    ticks = get_settings().cache_dir / "alpaca" / "ticks"
    rec["disk"] = {"market_mb": round(st.storage_bytes() / 1e6, 1), "largest": {k: round(v / 1e6, 1) for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:8]},
                   "individual_trade_cache_mb": round(sum(f.stat().st_size for f in ticks.rglob("*.parquet")) / 1e6, 1) if ticks.exists() else 0.0,
                   "bytes_per_observation": round(sum(v for k, v in by.items() if " options trade " in k) / max(1, rec["observations"]), 2)}
    rec["load_average_at_end"] = [round(x, 2) for x in os.getloadavg()]
    rec["peak_rss_mb"] = max((w.get("peak_rss_mb") or 0) for w in rec["workloads"])
    return rec


def save(rec: dict) -> Path:
    from strata.bench import bench_dir
    d = bench_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset_latest.json").write_text(json.dumps(rec, indent=1))
    with open(d / "dataset_history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return d / "dataset_latest.json"


def latest() -> dict | None:
    from strata.bench import bench_dir
    p = bench_dir() / "dataset_latest.json"
    return json.loads(p.read_text()) if p.exists() else None


def render(rec: dict) -> str:
    if not rec.get("workloads"):
        return rec.get("note") or "nothing was measured"
    lines = [f"Stored dataset: {rec['observations']:,} option trade bars ({', '.join(f'{s} {v['sessions']} sessions' for s, v in rec['dataset'].items())}); {rec['machine']}, {rec['cpus']} cores, load {rec['load_average_at_start'][0]} at the start",
             f"{'workload':30s} {'wall s':>8s} {'cpu s':>8s} {'again s':>8s} {'peak MB':>8s}  result"]
    for w in rec["workloads"]:
        if not w.get("ok"):
            lines.append(f"{w['name']:30s} FAILED: {w.get('error')}")
            continue
        info = ", ".join(f"{k} {v}" for k, v in w.items() if k in ("trades", "signals", "priced", "configurations", "entry_rules", "worker_cpu_seconds", "session_seconds", "contract_seconds", "trade_overlay_seconds", "year_span_seconds", "repriced", "provider_requests"))
        lines.append(f"{w['name']:30s} {w['seconds']:8.2f} {w['cpu_seconds']:8.2f} {(w['again_seconds'] if w.get('again_seconds') is not None else float('nan')):8.2f} {w.get('peak_rss_mb', 0):8.0f}  {info}")
    d = rec.get("disk") or {}
    lines.append(f"disk: {d.get('market_mb')} MB of market data, {d.get('bytes_per_observation')} bytes per option bar; individual-trade cache {d.get('individual_trade_cache_mb')} MB")
    return "\n".join(lines)
