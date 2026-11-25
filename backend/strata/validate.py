"""`strata validate`: a reproducible end-to-end check on actual market observations.

Starts from an EMPTY data directory, downloads a small real dataset from free keyless
sources, and proves the chain from download to reported profit:

 1. the data directory starts empty
 2. a small actual SPY dataset is downloaded
 3. the stored candles agree with an independent source and with each other
 4. legitimate option observations (trade prints) are downloaded, selectively
 5. a hand-written strategy is backtested on them
 6. the same strategy, written as a sentence and read by the interpreter, is backtested
 7. both produce identical trades
 8. every premium equals the stored record AND a fresh copy of the provider's record
 9. an interrupted download marks nothing as complete, and resumes without duplicates
10. no simulated data is involved anywhere

Nothing here needs an account or costs money. It needs network access to the free
providers; a step that cannot reach them FAILS (it is never skipped or faked).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

SENTENCE = "Buy one at-the-money SPY call expiring in 7 days at 10:00 ET and sell it at 15:55 ET, using option trade prices, from {start} to {end}."

# What a person would build in the Strategy Builder for that sentence, written by hand.
MANUAL = {
    "name": "Manual: weekly ATM call, 10:00 to 15:55", "symbol": "SPY",
    "legs": [{"id": "lc", "side": "buy", "right": "C", "quantity": 1, "dte": 7, "dte_mode": "nearest", "strike": {"method": "atm_offset", "value": 0}}],
    "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
    # dte_lte None = "on every session" (the builder's "any"); its default of 0 would only close on expiration day
    "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}},
    "sizing": {"method": "fixed", "contracts": 1},
}


class Check(Exception):
    """A validation step failed."""


def need(cond, msg: str):
    if not cond:
        raise Check(msg)


def _trade_key(t: dict) -> tuple:
    return (t["entry_ts"], t["exit_ts"], t["units"], t["exit_reason"], round(t["pnl"], 6), round(t["commissions"], 6),
            tuple((l["right"], l["strike"], l["expiration"], l["side"], l["contracts"], round(l["entry_price"], 6), round(l["exit_price"], 6), l.get("price_source")) for l in t["legs"]))


def run_validation(workdir: str | Path | None = None, keep: bool = False, sessions: int = 3, out=print) -> dict:
    """Run the ten steps. Returns {ok, steps: [{n, name, ok, seconds, detail, error}], data_dir}."""
    made = workdir is None
    root = Path(workdir).expanduser().resolve() if workdir else Path(tempfile.mkdtemp(prefix="strata-validate-"))
    # hermetic: only keyless free sources, no stored credentials, no model
    os.environ["STRATA_SECRET_STORE"] = "file"
    os.environ.pop("STRATA_OFFLINE", None)
    for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "POLYGON_API_KEY", "DATABENTO_API_KEY", "ORATS_API_KEY"):
        os.environ.pop(k, None)
    from strata.config import override_settings
    override_settings(data_dir=root)
    from strata.api import settings as app_settings
    getattr(app_settings, "_cache", {}).clear()  # credentials read earlier in this process are forgotten

    report: dict = {"data_dir": str(root), "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": [], "ok": False}
    ctx: dict = {}

    def step(n: int, name: str, fn):
        t0 = time.time()
        rec = {"n": n, "name": name, "ok": False, "seconds": 0.0, "detail": "", "error": None}
        report["steps"].append(rec)
        out(f"[{n:>2}/10] {name} ...")
        try:
            rec["detail"] = fn() or ""
            rec["ok"] = True
        except Check as e:
            rec["error"] = str(e)
        except Exception as e:  # noqa: BLE001  (a crash is a failed step, reported with its type)
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(time.time() - t0, 1)
        out(f"        {'PASS' if rec['ok'] else 'FAIL'}  {rec['detail'] if rec['ok'] else rec['error']}  ({rec['seconds']}s)")
        return rec["ok"]

    try:
        ok = (step(1, "The data directory starts empty", lambda: _s1(root, ctx))
              and step(2, "Download a small actual SPY dataset", lambda: _s2(ctx))
              and step(3, "Verify the stored candles", lambda: _s3(ctx))
              and step(4, "Download legitimate option observations", lambda: _s4(ctx, sessions))
              and step(5, "Run a hand-written strategy", lambda: _s5(ctx))
              and step(6, "Run the same strategy written as a sentence", lambda: _s6(ctx))
              and step(7, "Both produce identical results", lambda: _s7(ctx))
              and step(8, "Premiums match the source records", lambda: _s8(ctx))
              and step(9, "Interrupted downloads are never marked complete", lambda: _s9(ctx))
              and step(10, "No simulated data anywhere", lambda: _s10(ctx)))
        report["ok"] = bool(ok)
    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            (root / "validation-report.json").write_text(json.dumps(report, indent=1, default=str))
        except OSError:
            pass
        if made and not keep:
            shutil.rmtree(root, ignore_errors=True)
            report["data_dir"] = None
    n_ok = sum(1 for s in report["steps"] if s["ok"])
    out(f"\n{'VALIDATION PASSED' if report['ok'] else 'VALIDATION FAILED'}: {n_ok} of 10 steps passed" + ("" if report["ok"] or len(report["steps"]) == 10 else f" (stopped at step {len(report['steps'])})"))
    if report["data_dir"]:
        out(f"data directory kept at {report['data_dir']}")
    return report


# ----------------------------------------------------------------------------- steps
def _s1(root: Path, ctx: dict) -> str:
    from strata.api import db
    from strata.data.store import MarketStore
    leftovers = [p for p in root.rglob("*") if p.is_file() and p.name != "validation-report.json"] if root.exists() else []
    need(not leftovers, f"{root} is not empty ({len(leftovers)} files); pass an empty directory")
    root.mkdir(parents=True, exist_ok=True)
    db.init_db()
    st = MarketStore()
    need(not st.stored_symbols() and not st.datasets(), "a fresh store already reports datasets")
    need(db.list_runs(limit=5) == [], "a fresh database already has runs")
    ctx["store"] = st
    return f"{root} (no datasets, no runs, no credentials)"


def _s2(ctx: dict) -> str:
    from strata.data.providers.sources import quickstart
    from strata.data.store import MarketStore
    out = quickstart(["SPY"], store=ctx["store"], intraday=True)
    errs = [s for s in out["steps"] if s.get("error")]
    need(not errs, "download failed: " + "; ".join(str(e.get("error"))[:160] for e in errs))
    st = ctx["store"] = MarketStore()
    daily = st.read_daily("SPY")
    need(daily.height > 5000, f"only {daily.height} daily bars were stored")
    mins = st.read_bars("SPY", "1m")
    need(mins is not None and mins.height > 1000, "no 1-minute bars were stored")
    ctx["daily"], ctx["minutes"] = daily, mins
    size = sum(p.stat().st_size for p in (Path(st.root) if hasattr(st, "root") else Path(".")).rglob("*.parquet"))
    return f"{daily.height:,} daily bars ({daily['date'][0]} to {daily['date'][-1]}), {mins.height:,} one-minute bars, {size / 1e6:.1f} MB, from Yahoo and Cboe"


def _s3(ctx: dict) -> str:
    import polars as pl
    from strata.core import calendar as cal
    from strata.data.providers.sources import verify_daily
    st, daily, mins = ctx["store"], ctx["daily"], ctx["minutes"]
    # (a) internal consistency of every stored candle
    bad = daily.filter((pl.col("low") > pl.min_horizontal("open", "close")) | (pl.col("high") < pl.max_horizontal("open", "close")) | (pl.col("low") <= 0))
    need(bad.is_empty(), f"{bad.height} daily bars violate low <= open/close <= high")
    need(daily["date"].n_unique() == daily.height, "duplicate daily bars")
    need(mins["ts"].n_unique() == mins.height, "duplicate one-minute bars")
    not_sessions = [d for d in daily["date"].to_list() if not cal.is_trading_day(d)]
    need(not not_sessions, f"bars stored for days the exchange was closed: {not_sessions[:3]}")
    # (b) minute bars and daily bars come from separate requests: they must describe the same session
    m = mins.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date"))
    agg = m.group_by("date").agg(pl.col("high").max().alias("m_high"), pl.col("low").min().alias("m_low"), pl.col("close").last().alias("m_close"), pl.len().alias("n")).sort("date")
    full = agg.filter(pl.col("n") >= 380).join(daily.select("date", "high", "low", "close"), on="date")
    need(full.height >= 2, "fewer than two complete minute sessions to cross-check")
    worst = float(full.select(((pl.col("m_close") / pl.col("close") - 1).abs() * 100).max()).item())
    need(worst < 0.25, f"minute bars and daily bars disagree on a close by {worst:.3f}%")
    span = float(full.select((((pl.col("m_high") - pl.col("high")).abs() / pl.col("high")) * 100).max()).item())
    need(span < 0.5, f"minute bars and daily bars disagree on a session high by {span:.3f}%")
    # (c) an independent publisher (the option archive ships its own underlying price table)
    v = verify_daily(st, "SPY")
    need(v.get("compared", 0) > 500, "the independent reference could not be compared")
    need(v["within_0_1pct"] > 0.99, f"only {v['within_0_1pct']:.1%} of closes are within 0.1% of the independent source")
    ctx["verify"] = v
    return (f"{v['compared']:,} daily closes compared with an independent source: {v['within_0_1pct']:.2%} within 0.1% (largest difference {v['max_close_diff_pct']:.3f}%); "
            f"{full.height} minute sessions agree with their daily bar (largest close difference {worst:.3f}%)")


def _window(ctx: dict, sessions: int) -> tuple[date, date]:
    """The most recent completed sessions that have one-minute underlying bars."""
    import polars as pl
    from strata.data import planner as P
    last = P.last_completed_session()
    m = ctx["minutes"].with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date"))
    days = sorted(d for d, n in m.group_by("date").agg(pl.len().alias("n")).iter_rows() if n >= 380 and d <= last)
    need(len(days) >= sessions, f"only {len(days)} complete one-minute sessions are available")
    return days[-sessions], days[-1]


def _spec(strategy: dict, start: date, end: date, label: str) -> dict:
    return {"kind": "options", "mode": "trades", "study": None, "symbol": "SPY", "label": label, "provider": None,
            "backtest": {"strategies": [strategy], "execution": {"signal_interval_minutes": 1}, "label": label},
            "requested": {"start": start.isoformat(), "end": end.isoformat(), "stated": True},
            # a session whose contract did not trade near the entry is reported as skipped rather than stopping the run
            "approvals": {"reduced_window": True}, "interpretation": {"interpreter": "", "detail": label}}


def _run(spec: dict, name: str, ctx: dict) -> tuple[str, dict]:
    import hashlib
    from strata.api import db
    from strata.config import ENGINE_VERSION
    from strata.research.pipeline import run_pipeline
    cfg = spec["backtest"]
    body = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    rid = db.create_run("backtest", name, {**cfg, "_research": {k: spec.get(k) for k in ("mode", "requested", "approvals", "provider")}}, None, None, hashlib.sha256(body.encode()).hexdigest()[:16], {}, ENGINE_VERSION, False)
    out = run_pipeline(rid, spec, store=ctx["store"])
    db.update_run(rid, status="done", progress=1.0, summary=out)
    return rid, out


def _s4(ctx: dict, sessions: int) -> str:
    """Stage two of the pipeline for the hand-written strategy: only the sessions that
    produced a signal, only the contracts those signals could select."""
    import polars as pl
    from strata.core import calendar as cal
    from strata.data.providers.sources import parse_occ
    from strata.data.store import MarketStore
    from strata.research.pipeline import load_pipeline
    a, b = _window(ctx, sessions)
    ctx["window"] = (a, b)
    rid, out = _run(_spec(MANUAL, a, b, MANUAL["name"]), MANUAL["name"], ctx)
    ctx["manual"] = (rid, out)
    pipe = load_pipeline(rid) or {}
    need(out.get("outcome") == "completed", "the run stopped for a decision: " + str((pipe.get("decision") or {}).get("text") or out)[:300])
    st = ctx["store"] = MarketStore()
    ds = st.option_dataset("SPY", "trade", None)
    need(ds is not None, "no option trade dataset was created")
    idx = st.option_bar_index("SPY", dataset=ds)
    need(not idx.is_empty(), "the dataset index is empty: no (session, contract) pair was fetched to completion")
    rows, contracts, days = 0, set(), []
    for d in cal.trading_days(a, b):
        bars = st.read_option_bars("SPY", d, dataset=ds)
        if bars is None or bars.is_empty():
            continue
        days.append(d)
        rows += bars.height
        contracts |= set(bars["occ"].unique().to_list())
        o, c = (int(x.timestamp()) for x in cal.session_bounds(d))
        ts = bars["ts"].dt.epoch("s")
        need(int(ts.min()) >= o and int(ts.max()) < c, f"{d}: an option bar lies outside the trading session")
        need(bars.filter(pl.col("volume") <= 0).is_empty(), f"{d}: a bar with no traded volume was stored as a trade print")
        need(bars.filter((pl.col("low") > pl.min_horizontal("open", "close")) | (pl.col("high") < pl.max_horizontal("open", "close")) | (pl.col("low") <= 0)).is_empty(), f"{d}: malformed option bar")
        need(bars.select(pl.struct("occ", "ts").n_unique()).item() == bars.height, f"{d}: duplicate option bars")
        need(bars.filter(pl.col("expiration") < d).is_empty(), f"{d}: a bar of a contract that had already expired")
        need({"bid", "ask"}.isdisjoint(bars.columns), "trade bars carry bid/ask columns: trades must never be stored as quotes")
        for occ, e, k, r in bars.select("occ", "expiration", "strike", "right").unique().iter_rows():
            root_, exp_, right_, strike_ = parse_occ(occ)  # raises if the identifier is not an OCC symbol
            need((root_, exp_, right_) == ("SPY", e, r) and abs(strike_ - k) < 1e-6, f"contract {occ} does not parse back to its own columns")
    need(days, "no option trade bars were stored for the window")
    desc = st.descriptor(ds)
    ctx["dataset"] = ds
    acq = next((s for s in pipe.get("stages", []) if s["id"] == "acquire"), {})
    return (f"{rows:,} one-minute trade bars for {len(contracts)} contracts on {len(days)} sessions ({days[0]} to {days[-1]}) from {desc.get('provider', ds.provider)}; "
            f"dataset {ds.id}; {acq.get('detail', '')}"[:420])


def _s5(ctx: dict) -> str:
    from strata.api.results import load_trades
    rid, out = ctx["manual"]
    trades = load_trades(rid)
    need(trades, "the hand-written strategy produced no trades on the downloaded observations")
    for t in trades:
        for l in t["legs"]:
            need(l.get("price_source") == "trade_print", f"trade {t['id']} was not priced on a trade print ({l.get('price_source')})")
            need(l.get("entry_bid") is None and l.get("entry_ask") is None, "a bid/ask was attached to a trade-print fill")
    ctx["manual_trades"] = trades
    pnl = sum(t["pnl"] for t in trades)
    return f"{len(trades)} trade{'s' if len(trades) != 1 else ''}, net {pnl:+,.2f} after {sum(t['commissions'] for t in trades):.2f} commissions; every fill is an observed trade print (run {rid})"


def _s6(ctx: dict) -> str:
    from strata.ai.agent import Agent
    from strata.ai.draft import build_spec, compile_draft
    from strata.api.results import load_trades
    from strata.research.pipeline import load_pipeline
    a, b = ctx["window"]
    text = SENTENCE.format(start=a.isoformat(), end=b.isoformat())
    st = ctx["store"]
    ag = Agent(None, st, llm_factory=lambda: None)  # the built-in interpreter: deterministic, no model, no network
    r = ag._rules(text, None, sorted(st.stored_symbols()), None, "SPY")
    need(r["action"] == "new", f"the sentence was not read as a new strategy ({r['action']})")
    d = r["draft"]
    need(not d.unsupported and not [q for q in d.questions if q.get("answered") is None], f"the interpreter left something open: {d.unsupported or d.questions}")
    compiled = compile_draft(d, st)
    need(not compiled["errors"] and compiled.get("strategy"), "the interpreted strategy does not compile: " + "; ".join(compiled["errors"]))
    spec = build_spec(d, compiled, None, {"reduced_window": True}, d.title, "built-in rules")
    need(spec["mode"] == "trades" and spec["requested"] == {"start": a.isoformat(), "end": b.isoformat(), "stated": True}, f"the sentence's mode/period were not read as written: {spec['mode']} {spec['requested']}")
    rid, out = _run(spec, "AI: " + d.title[:60], ctx)
    need(out.get("outcome") == "completed", "the run stopped for a decision: " + str(((load_pipeline(rid) or {}).get("decision") or {}).get("text") or out)[:300])
    ctx["ai"] = (rid, out)
    ctx["ai_strategy"] = compiled["strategy"]
    ctx["ai_trades"] = load_trades(rid)
    ctx["ai_assumed"] = compiled["assumptions"]
    need(ctx["ai_trades"], "the interpreted strategy produced no trades")
    return f"\"{text}\" -> {len(ctx['ai_trades'])} trade{'s' if len(ctx['ai_trades']) != 1 else ''} (run {rid}); assumed by the interpreter: {', '.join(x['field'] for x in compiled['assumptions']) or 'nothing'}"


def _s7(ctx: dict) -> str:
    from strata.engine.models import StrategyConfig
    m, a = ctx["manual_trades"], ctx["ai_trades"]
    need(len(m) == len(a), f"{len(m)} trades by hand versus {len(a)} from the sentence")
    for x, y in zip(m, a):
        need(_trade_key(x) == _trade_key(y), f"trade {x['id']} differs: {_trade_key(x)} vs {_trade_key(y)}")
    # the configurations themselves: everything that affects behaviour must be equal. Labels (the name, the
    # editor's node ids) and the parameters of rules that are switched off do not.
    def norm(cfg: dict) -> dict:
        c = StrategyConfig.model_validate(cfg).model_dump(mode="json")
        c.pop("name", None)

        def strip_ids(node):
            if isinstance(node, dict):
                return {k: strip_ids(v) for k, v in node.items() if k != "id"}
            return [strip_ids(x) for x in node] if isinstance(node, list) else node
        c["entry"]["conditions"] = strip_ids(c["entry"]["conditions"])
        c["exits"] = {k: ({"enabled": False} if isinstance(v, dict) and v.get("enabled") is False else v) for k, v in c["exits"].items()}
        return c
    x, y = norm(MANUAL), norm(ctx["ai_strategy"])
    diff = [k for k in x if json.dumps(x[k], sort_keys=True) != json.dumps(y.get(k), sort_keys=True)]
    need(not diff, "the two strategies trade identically here but their configurations differ in: " + ", ".join(diff))
    return f"{len(m)} trades identical in contract, entry and exit time, both prices, commissions and profit; every setting that affects behaviour is equal in the two configurations"


def _s8(ctx: dict) -> str:
    """Each fill is recomputed from the stored bars without the engine, then the bars
    themselves are downloaded again and compared with what was stored."""
    import polars as pl
    from strata.data.providers import sources as S
    from strata.data.providers.sources import Contract, occ_symbol
    st, ds = ctx["store"], ctx["dataset"]
    src = next(s for s in S.option_bar_sources(st) if s.name == ds.provider)
    checked, refetched, max_age = 0, 0, 300
    with S._client() as client:
        for t in ctx["manual_trades"]:
            d = date.fromisoformat(t["entry_date"])
            stored = st.read_option_bars("SPY", d, dataset=ds)
            for l in t["legs"]:
                exp = date.fromisoformat(l["expiration"])
                occ = occ_symbol("SPY", exp, l["right"], l["strike"])
                mine = stored.filter(pl.col("occ") == occ).sort("ts").with_columns((pl.col("ts").dt.epoch("s") + 60).alias("end"))
                need(not mine.is_empty(), f"the traded contract {occ} has no stored bars on {d}")
                fresh, status, _ = src.fetch_batch(client, "SPY", [Contract(occ, "SPY", exp, l["strike"], l["right"])], [d])
                need((d, occ) in status and not fresh.is_empty(), f"{src.label} no longer serves {occ} for {d}; the stored record could not be re-checked")
                fresh = fresh.filter(pl.col("occ") == occ).with_columns((pl.col("ts").dt.epoch("s") + 60).alias("end"))
                for which, ts, px in (("entry", t["entry_ts"], l["entry_price"]), ("exit", l.get("exit_ts") or t["exit_ts"], l["exit_price"])):
                    bar = mine.filter(pl.col("end") <= ts).tail(1)  # the last print that had finished by the fill
                    need(not bar.is_empty(), f"{occ}: no stored print at or before the {which} at {ts}")
                    end, close = int(bar["end"][0]), float(bar["close"][0])
                    need(ts - end <= max_age, f"{occ}: the {which} used a print {ts - end}s old (limit {max_age}s)")
                    need(abs(close - px) < 5e-5, f"{occ}: {which} price {px} is not the stored print {close}")
                    again = fresh.filter(pl.col("end") == end)
                    need(not again.is_empty(), f"{occ}: the provider's fresh copy has no bar ending at {end}")
                    need(abs(float(again["close"][0]) - close) < 5e-3, f"{occ}: stored print {close} differs from the provider's fresh copy {float(again['close'][0])}")
                    checked += 1
                refetched += 1
    return f"{checked} fill prices recomputed from the stored bars without the engine (each within {max_age}s of its print) and matched against {refetched} freshly downloaded contract histories from {src.label}"


def _s9(ctx: dict) -> str:
    """A network failure in the middle of a download: nothing half-fetched is recorded as
    complete, and running it again fetches only what is missing, without duplicates."""
    import polars as pl
    from strata.data.providers import sources as S
    from strata.data.providers.sources import Contract, occ_symbol
    from strata.data.store import MarketStore
    st, ds = ctx["store"], ctx["dataset"]
    src = next(s for s in S.option_bar_sources(st) if s.name == ds.provider)
    t = ctx["manual_trades"][-1]
    d, leg = date.fromisoformat(t["entry_date"]), t["legs"][0]
    exp, k0 = date.fromisoformat(leg["expiration"]), float(leg["strike"])
    have = set(st.option_bar_index("SPY", dataset=ds).filter(pl.col("date") == d)["occ"].to_list())
    # neighbouring strikes of the traded expiration that were not needed, so not downloaded yet
    new = [Contract(occ_symbol("SPY", exp, r, k), "SPY", exp, k, r) for k in (k0 - 25, k0 + 25, k0 - 30, k0 + 30) for r in ("C", "P")]
    new = [c for c in new if c.occ not in have][:6]
    need(len(new) >= 4, "could not find contracts that are not stored yet")
    real_get, calls = S.http_get, {"n": 0}

    def flaky(client, url, params=None, headers=None, retries=4):
        calls["n"] += 1
        if calls["n"] > 2:  # the connection drops after two successful requests
            raise S.SourceError("simulated network failure (connection reset)")
        return real_get(client, url, params=params, headers=headers, retries=retries)
    S.http_get = flaky
    try:
        rep1 = src.acquire("SPY", {d: new})
    finally:
        S.http_get = real_get
    idx1 = MarketStore().option_bar_index("SPY", dataset=ds).filter(pl.col("date") == d)
    done1 = set(idx1["occ"].to_list()) & {c.occ for c in new}
    need(rep1["pairs_failed"] >= 1, f"the failure was not reported ({rep1})")
    need(len(done1) == rep1["pairs_ok"] + rep1["pairs_empty"], f"{len(done1)} contracts are marked complete but only {rep1['pairs_ok'] + rep1['pairs_empty']} were fetched to the end")
    need(len(done1) < len(new), "every contract is marked complete although the download was cut off")
    # resume: only the missing pairs are requested, and nothing is stored twice
    rep2 = src.acquire("SPY", {d: new})
    need(rep2["pairs_cached"] == len(done1), f"the resume refetched completed contracts (cached {rep2['pairs_cached']}, expected {len(done1)})")
    need(rep2["pairs_failed"] == 0, f"the resume failed: {rep2['errors'][:2]}")
    st2 = ctx["store"] = MarketStore()
    idx2 = st2.option_bar_index("SPY", dataset=ds).filter(pl.col("date") == d)
    resolved = set(idx2["occ"].to_list()) & {c.occ for c in new}
    need(len(resolved) + rep2["pairs_unavailable"] == len(new), f"after resuming, {len(new) - len(resolved) - rep2['pairs_unavailable']} contracts are still unaccounted for")
    bars = st2.read_option_bars("SPY", d, dataset=ds)
    need(bars.select(pl.struct("occ", "ts").n_unique()).item() == bars.height, "the resume stored duplicate bars")
    need(idx2.select(pl.struct("date", "occ").n_unique()).item() == idx2.height, "the resume stored duplicate index rows")
    rep3 = src.acquire("SPY", {d: new})
    need(rep3["requests"] == 0 or rep3["pairs_cached"] == len(resolved), "a third run requested data that was already complete")
    return (f"connection cut after 2 requests: {rep1['pairs_ok'] + rep1['pairs_empty']} of {len(new)} contracts recorded as complete, {rep1['pairs_failed']} reported as failed; "
            f"the resume fetched only the missing {rep2['pairs_ok'] + rep2['pairs_empty']} ({rep2['pairs_cached']} reused), no duplicate bars")


def _s10(ctx: dict) -> str:
    from strata.api import db
    from strata.api.results import load_run_result
    st = ctx["store"]
    dsets = st.datasets()
    need(dsets, "no datasets to inspect")
    for k in dsets:
        desc = st.descriptor(k)
        need(not desc.get("synthetic") and k.provider != "synthetic", f"dataset {k.id} is simulated")
        need(desc.get("license") or desc.get("url"), f"dataset {k.id} carries no source or licence")
        need((st.dataset_dir(k) / "HEAD").exists() and any((st.dataset_dir(k) / "manifests").glob("*.json")), f"dataset {k.id} has no manifest")
    for sym in st.stored_symbols():
        need(not st.is_synthetic(sym), f"{sym} is flagged as simulated")
    n = 0
    for run in db.list_runs(limit=50):
        need(not run.get("synthetic"), f"run {run['id']} is flagged as using simulated data")
        if run["kind"] == "backtest" and run["status"] == "done":
            res = load_run_result(run)
            meta = res.get("meta") or {}
            need(meta.get("pricing") == "trades", f"run {run['id']} was priced on {meta.get('pricing')}, not on observed trade prints")
            need(not meta.get("synthetic") and not (res["config"].get("allow_synthetic")), f"run {run['id']} allowed simulated data")
            n += 1
    provs = sorted({k.provider for k in dsets})
    return f"{len(dsets)} datasets from {', '.join(provs)}, each with a licence and manifest; {n} runs priced on observed trade prints; no modeled or simulated price anywhere"
