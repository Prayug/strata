"""Strata REST API (FastAPI). Interactive docs at /docs, schema at /openapi.json."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from strata.api import db
from strata.api.jobs import JobManager
from strata.api.results import _clean, delete_results, dumps, load_result, load_trades, run_dir, trade_replay
from strata.config import ENGINE_VERSION, get_settings
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine.models import BacktestConfig, Execution, StrategyConfig

jobs: JobManager | None = None


standby = False  # True while another machine holds the data directory: this copy has not opened the database


@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs, standby
    import os
    import threading

    from strata.core import instance
    instance.configure(role="server", url=os.environ.get("STRATA_PUBLIC_URL"))
    loop = asyncio.get_running_loop()
    st: dict = {"sched": None}
    stop = threading.Event()

    def startup():
        global jobs, standby
        db.init_db()
        from strata.api.settings import backfill_where, migrate_legacy_secrets
        moved = migrate_legacy_secrets()  # V2 kept API keys in the database; they now live in the OS keychain / a protected file
        if moved:
            print(f"[strata] moved {moved} credential(s) from the database into the credential store")
        backfill_where()  # which store holds each key (never the key), so a copy on the other side of a container boundary can say so
        MarketStore()  # brings an older data directory to the current storage layout
        n = db.mark_interrupted()
        jobs = JobManager(max_concurrent=int(os.environ.get("STRATA_MAX_JOBS", "2")))
        jobs.loop = loop
        if n:
            print(f"[strata] marked {n} interrupted runs as error")
        if os.environ.get("STRATA_COLLECTOR", "1") != "0":
            from strata.collector.service import Scheduler
            st["sched"] = Scheduler(jobs)  # does nothing unless the collector is switched on in its settings
            st["sched"].start()
        standby = False

    def wait_then_start():
        # standby: the database is not opened until the copy that holds the data directory has gone
        while not stop.is_set():
            try:
                instance.ensure()
                startup()
                print("[strata] the other copy has gone: this one now holds the data directory and is running", flush=True)
                return
            except instance.Busy:
                stop.wait(2.0)

    try:
        startup()
    except instance.Busy as e:
        standby = True
        print(f"[strata] STANDBY. {e}", flush=True)
        threading.Thread(target=wait_then_start, name="strata-standby", daemon=True).start()
    yield
    stop.set()
    if st["sched"] is not None:
        st["sched"].stop()
    instance.release()


app = FastAPI(title="Strata API", version=ENGINE_VERSION, lifespan=lifespan,
              description="Options backtesting & research platform. All long-running work (backtests, optimisations, ingestion) runs as background jobs; poll /api/jobs or stream /api/jobs/{id}/stream.")
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins), allow_methods=["*"], allow_headers=["*"])

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _local_host_header(host: str) -> bool:
    h = (host or "").rsplit(":", 1)[0].strip("[]").lower() if not (host or "").startswith("[") else (host or "").split("]")[0].strip("[").lower()
    return h in LOOPBACK or h.endswith(".localhost")


@app.middleware("http")
async def standby_guard(request: Request, call_next):
    """While another machine holds the data directory this copy answers nothing but its health: it has not opened the database."""
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        from strata.core import instance
        s_ = instance.status()  # cheap while the lease is held; also catches a lease lost after startup
        if standby or s_.get("standby"):
            msg = s_.get("message") or "The other copy of Strata has gone: this one is starting. Try again in a moment."
            return JSONResponse({"detail": msg, "standby": s_ if s_.get("standby") else {"standby": True, "starting": True, "message": msg}}, status_code=503, headers={"Retry-After": "3"})
    return await call_next(request)


@app.middleware("http")
async def remote_access_guard(request: Request, call_next):
    """Strata is a local application. A request that does not come from this machine - or that
    arrives under a non-local Host name (DNS rebinding) - is refused unless it carries the
    access token configured in STRATA_API_TOKEN. With no token configured, remote access is off."""
    client = request.client.host if request.client else ""
    in_process = client == "testclient"  # Starlette's in-process test client: there is no socket, so no remote peer
    # Inside a container the computer's own browser arrives from the container network's gateway, not from loopback, and
    # cannot be told apart from another machine by its address. When the port is PUBLISHED ON LOOPBACK ONLY (the compose
    # file does this: "127.0.0.1:8000:8000") nothing but the computer itself can reach it, and the compose file says so
    # with STRATA_LOOPBACK_PUBLISHED=1. The Host name must still be local, which keeps DNS rebinding out.
    import os as _os
    local_peer = client in LOOPBACK or _os.environ.get("STRATA_LOOPBACK_PUBLISHED") == "1"
    if request.url.path.startswith("/api/") and not in_process and not (local_peer and _local_host_header(request.headers.get("host", ""))):
        import hmac
        import os
        token = os.environ.get("STRATA_API_TOKEN", "")
        given = request.headers.get("x-strata-token") or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if not token:
            return JSONResponse({"detail": "Remote access is disabled. Strata only answers requests from this computer unless STRATA_API_TOKEN is set on the server."}, status_code=403)
        if not given or not hmac.compare_digest(token, given):
            return JSONResponse({"detail": "A valid access token is required (header X-Strata-Token)."}, status_code=401)
    return await call_next(request)


def _busy(request: Request, exc: Exception):
    return JSONResponse({"detail": str(exc), "standby": {"standby": True, "holder": getattr(exc, "holder", {}), "message": str(exc)}}, status_code=503, headers={"Retry-After": "3"})


from strata.core.instance import Busy as _Busy  # noqa: E402

app.add_exception_handler(_Busy, _busy)

from strata.api.research_api import router as research_router  # noqa: E402

app.include_router(research_router)

from strata.api.v4_api import router as v4_router  # noqa: E402

app.include_router(v4_router)

from strata.api.v5_api import router as v5_router  # noqa: E402

app.include_router(v5_router)

from strata.api.v6_api import router as v6_router  # noqa: E402

app.include_router(v6_router)

from strata.api.v7_api import router as v7_router  # noqa: E402

app.include_router(v7_router)

from strata.api.v8_api import router as v8_router  # noqa: E402

app.include_router(v8_router)


def J(o) -> JSONResponse:
    return JSONResponse(content=json.loads(dumps(o)))


def store() -> MarketStore:
    return MarketStore()


# ======================================================================= meta
@app.get("/api/health")
def health():
    """Answers in every state. `standby` describes the copy of Strata that holds the data directory when this one may not open it."""
    from strata.core import instance
    s_ = instance.status()
    if standby and not s_.get("standby"):
        s_ = {"standby": True, "starting": True, "message": "The other copy of Strata has gone: this one is starting."}
    return {"ok": True, "engine_version": ENGINE_VERSION, "data_dir": str(get_settings().data_dir), "standby": s_ if s_.get("standby") else None}


@app.get("/api/meta")
def meta():
    from strata.core.indicators import INDICATORS
    from strata.data.providers import PROVIDERS
    from strata.engine.conditions import INDICATOR_NAMES, LEAF_TYPES, OPERAND_FEATURES, TIMEFRAMES
    from strata.engine.templates import templates
    from strata.optimize.optimizer import OBJECTIVES
    from strata.research.queries import FILTER_FIELDS, TARGETS, TEMPLATES
    return J({
        "conditions": LEAF_TYPES, "indicators": INDICATORS, "indicator_names": INDICATOR_NAMES, "timeframes": TIMEFRAMES, "features": OPERAND_FEATURES,
        "templates": templates(*_template_args()), "objectives": OBJECTIVES, "research": TEMPLATES, "research_targets": TARGETS, "research_filters": FILTER_FIELDS,
        "providers": [{"name": k, "label": v.label, "docs": v.docs, "resolution": v.resolution} for k, v in PROVIDERS.items()],
        "execution_defaults": Execution().model_dump(), "engine_version": ENGINE_VERSION,
        "expressions": _expr_reference(), "default_symbol": _default_symbol(), "pricing_defaults": __import__("strata.engine.models", fromlist=["Pricing"]).Pricing().model_dump(),
    })


def _template_args(symbol: str | None = None) -> tuple[str, bool, float]:
    """(symbol, option data is end-of-day?, strike-width scale) for building presets on stored data."""
    from strata.data.catalog import profile
    s = store()
    sym = symbol or _default_symbol()
    prof = profile(s, sym) or {}
    opt = prof.get("options") or {}
    eod = opt.get("resolution") == "EOD" or not opt
    spot = None
    try:
        d = s.read_daily(sym)
        spot = float(d["close"][-1]) if not d.is_empty() else None
    except Exception:  # noqa: BLE001
        pass
    scale = 1.0 if not spot else (10.0 if spot >= 2500 else 1.0 if spot >= 150 else 0.5 if spot >= 60 else 0.2)
    return sym, eod, scale


@app.get("/api/templates")
def templates_for(symbol: Optional[str] = None):
    from strata.engine.templates import templates
    sym, eod, scale = _template_args(symbol.upper() if symbol else None)
    return J({"symbol": sym, "eod": eod, "templates": templates(sym, eod, scale)})


def _expr_reference() -> list[dict]:
    from strata.engine.expr import REGISTRY
    return [{"name": k, **v} for k, v in REGISTRY.items()]


def _default_symbol() -> str:
    """Preferred real symbol that is actually stored; never a synthetic one unless nothing else exists."""
    from strata.api.settings import get_setting
    syms = store().symbols()
    real = [k for k in syms if not store().is_synthetic(k) and store().has_daily(k)]
    pref = [str(get_setting("data.default_symbol") or "SPY"), "SPY", "QQQ", "IWM", "SPX", "AAPL", "NVDA", "TSLA"]
    return next((x for x in pref if x in real and x not in ("VIX",)), next((x for x in real if x != "VIX"), pref[0]))


@app.get("/api/overview")
def overview():
    from strata.data.catalog import catalog
    cat = catalog(store())
    runs = db.list_runs(limit=12)

    def span(c):
        o, d = c.get("options"), c.get("daily") or c.get("underlying")
        return (o or d or {}).get("first"), (o or d or {}).get("last")
    return J({
        "symbols": [{"symbol": c["symbol"], "synthetic": c["meta"].get("synthetic"), "provider": c["meta"].get("provider"), "has_options": "options" in c,
                     "first": span(c)[0], "last": span(c)[1], "storage_bytes": c.get("storage_bytes")} for c in cat],
        "recent_runs": runs, "strategies": len(db.list_strategies()), "active_jobs": jobs.active() if jobs else [],
        "storage_bytes": store().storage_bytes(), "engine_version": ENGINE_VERSION, "default_symbol": _default_symbol(),
        "has_real_data": any(not c["meta"].get("synthetic") and (c.get("daily") or c.get("underlying")) for c in cat),
    })


# ======================================================================= data
@app.get("/api/data/catalog")
def data_catalog(refresh: bool = False):
    from strata.data.catalog import catalog
    return J(catalog(store(), refresh))


@app.get("/api/data/symbols/{sym}")
def data_symbol(sym: str):
    from strata.data.catalog import symbol_coverage
    s = store()
    if sym not in s.symbols() and not s.datasets(sym):
        raise HTTPException(404, f"unknown symbol {sym}")
    # data files across every dataset of the symbol (each dataset's current manifest lists its own)
    n_files = sum(len(s.head(k).get("files") or {}) for k in s.datasets(sym))
    return J(symbol_coverage(s, sym) | {"manifest": s.manifest(sym), "files": n_files})


@app.delete("/api/data/symbols/{sym}")
def data_delete(sym: str):
    from strata.data.catalog import invalidate
    store().remove_symbol(sym)
    invalidate()
    return {"ok": True}


@app.get("/api/data/candles/{sym}")
def candles(sym: str, start: Optional[date] = None, end: Optional[date] = None, tf: str = "1d", limit: int = 5000):
    """OHLCV bars. Intraday requests are served from the finest stored resolution at or below
    the requested timeframe; if only coarser bars exist they are returned as-is and labelled."""
    from strata.data.store import INTRADAY_RES
    s = store()
    daily = s.read_daily(sym)
    if tf == "1d":
        d = daily
        if start:
            d = d.filter(pl.col("date") >= start)
        if end:
            d = d.filter(pl.col("date") <= end)
        d = d.tail(limit)
        return {"tf": tf, "served": "1d", "bars": [[int(datetime(x.year, x.month, x.day).timestamp()), o, h, l, c, v] for x, o, h, l, c, v in d.select(["date", "open", "high", "low", "close", "volume"]).rows()]}
    want = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "60m": 3600}.get(tf)
    if want is None:
        raise HTTPException(422, f"unknown timeframe {tf}")
    have = s.bar_resolutions(sym)
    finer = [r for r in have if INTRADAY_RES[r] <= want]
    src = max(finer, key=lambda r: INTRADAY_RES[r]) if finer else (min(have, key=lambda r: INTRADAY_RES[r]) if have else None)
    if src is None:
        return {"tf": tf, "served": None, "bars": [], "note": f"No intraday bars are stored for {sym}."}
    full = s.read_bars(sym, src)
    last = full["ts"].max().date() if not full.is_empty() else date.today()
    end = end or last
    start = start or end - timedelta(days=5 if want <= 300 else 45)
    m = s.read_bars(sym, src, start, end)
    if m.is_empty():
        return {"tf": tf, "served": src, "bars": []}
    m = m.with_columns(pl.col("ts").dt.epoch("s").alias("t"))
    if want > INTRADAY_RES[src]:
        # bucket from each session's open so 30-minute bars start at 09:30, not on the hour
        m = m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("et")).with_columns(
            (pl.col("et").dt.hour().cast(pl.Int32) * 3600 + pl.col("et").dt.minute().cast(pl.Int32) * 60 - 34200).alias("so"))
        m = m.with_columns((pl.col("t") - pl.col("so") % want).alias("t")).group_by("t", maintain_order=True).agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum())
    served = src if want <= INTRADAY_RES[src] else tf
    note = f"{tf} bars are not stored for {sym}; showing {src} bars." if INTRADAY_RES[src] > want else None
    return {"tf": tf, "served": served, "source_resolution": src, "note": note, "bars": m.select(["t", "open", "high", "low", "close", "volume"]).tail(limit).rows()}


@app.get("/api/data/chain/{sym}")
def chain_snapshot(sym: str, day: Optional[date] = None, time: str = "10:00"):
    from strata.engine.market import DayChain, from_day_num
    s = store()
    days = s.option_days(sym)
    if not days:
        raise HTTPException(404, f"no option data for {sym}")
    d = max([x for x in days if not day or x <= day], default=days[0])
    ch = DayChain(s.read_option_day(sym, d))
    t = int(datetime.combine(d, datetime.strptime(time, "%H:%M").time(), tzinfo=cal.ET).timestamp())
    si = max(ch.snapshot_at(t), 0)
    sl = ch.rows(si)
    rows = [ch.row_dict(i) for i in range(sl.start, sl.stop)]
    exps = sorted({r["expiration"] for r in rows})
    return J({"date": d.isoformat(), "ts": int(ch.snap_ts[si]), "spot": float(np.nanmedian(ch.under[sl])), "expirations": exps, "rows": rows,
              "snapshots": [int(x) for x in ch.snap_ts]})


@app.get("/api/data/events")
def events(start: Optional[date] = None, end: Optional[date] = None):
    e = store().read_events()
    if start:
        e = e.filter(pl.col("date") >= start)
    if end:
        e = e.filter(pl.col("date") <= end)
    return J(e.to_dicts())


@app.get("/api/data/providers")
def providers(check: bool = False):
    from strata.data.providers import PROVIDERS
    out = []
    for k, cls in PROVIDERS.items():
        p = cls()
        st = p.check() if check else {"ok": p.configured(), "message": "configured" if p.configured() else "not configured"}
        out.append({"name": k, "label": p.label, "configured": p.configured(), "status": st, "docs": p.docs, "resolution": p.resolution,
                    "cache_bytes": sum(f.stat().st_size for f in p.cache.rglob("*") if f.is_file())})
    return J(out)


class IngestReq(BaseModel):
    provider: str
    symbol: str
    start: date
    end: date
    interval: int = 15
    include_options: bool = True


@app.post("/api/data/ingest")
def ingest(req: IngestReq):
    rid = db.create_run("ingest", f"{req.provider}:{req.symbol} {req.start}..{req.end}", req.model_dump(mode="json"))
    jobs.submit(rid, "ingest", req.model_dump(mode="json"))
    return {"id": rid}


@app.post("/api/data/import")
async def import_data(file: UploadFile = File(...), kind: str = Form(...), symbol: str = Form(""), tz: str = Form("America/New_York"), meta: str = Form("{}")):
    up = get_settings().cache_dir / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload").name
    dest = up / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{name}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    payload = {"path": str(dest), "kind": kind, "symbol": symbol.upper() or None, "tz": tz, "meta": json.loads(meta or "{}")}
    rid = db.create_run("import", f"import {kind} {symbol}", payload)
    jobs.submit(rid, "import", payload)
    return {"id": rid}


class DemoReq(BaseModel):
    start: date = date(2023, 1, 3)
    end: date = date(2024, 12, 31)
    seed: int = 20240101
    interval: int = 15


@app.post("/api/data/demo")
def demo(req: DemoReq):
    """Developer sandbox only: generates SIMULATED symbols (DEMO, DEMOX). Never started automatically."""
    rid = db.create_run("demo_data", "Developer sandbox: generate SIMULATED symbols", req.model_dump(mode="json"))
    jobs.submit(rid, "demo_data", req.model_dump(mode="json"))
    return {"id": rid}


@app.get("/api/data/storage")
def storage():
    s = store()
    # by kind of observation, from the dataset tree (provider / symbol / data type / observation / resolution / version)
    parts = {"underlying": 0, "option quotes": 0, "option trades": 0, "reference": 0}
    for k in s.datasets():
        size = s.storage_bytes(key=k)
        parts["underlying" if k.dtype == "underlying" else "option trades" if k.obs == "trade" else "option quotes"] += size
    ref = s.root / "reference"
    parts["reference"] = sum(f.stat().st_size for f in ref.rglob("*") if f.is_file()) if ref.exists() else 0
    res = get_settings().results_dir
    cache = get_settings().cache_dir
    return {"market": parts, "results": sum(f.stat().st_size for f in res.rglob("*") if f.is_file()) if res.exists() else 0,
            "cache": sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) if cache.exists() else 0, "data_dir": str(get_settings().data_dir)}


# ======================================================================= free data sources
@app.get("/api/data/sources")
def data_sources():
    from strata.data.providers.sources import SOURCES
    return J([cls(store()).info() for cls in SOURCES.values()])


class SourceReq(BaseModel):
    source: str
    dataset: str
    symbols: list[str] = []
    start: Optional[date] = None
    end: Optional[date] = None
    force: bool = False


@app.post("/api/data/sources/plan")
def data_source_plan(req: SourceReq):
    """Coverage and size estimate before anything is downloaded."""
    from strata.data.providers.sources import SourceError, get_source
    try:
        return J(get_source(req.source, store()).plan(req.dataset, [x.upper() for x in req.symbols], req.start, req.end))
    except SourceError as e:
        raise HTTPException(400, str(e))


@app.post("/api/data/sources/ingest")
def data_source_ingest(req: SourceReq):
    from strata.data.providers.sources import SOURCES
    if req.source not in SOURCES:
        raise HTTPException(404, f"unknown source {req.source}")
    src = SOURCES[req.source](store())
    ok, why = src.available()
    if not ok:
        raise HTTPException(400, why)
    if src.tier == "paid":
        raise HTTPException(400, "Paid sources are never started automatically; use the provider's own controls.")
    payload = req.model_dump(mode="json")
    rid = db.create_run("source_ingest", f"{src.label}: {req.dataset} {' '.join(req.symbols)}".strip(), payload)
    jobs.submit(rid, "source_ingest", payload)
    return {"id": rid}


class QuickReq(BaseModel):
    symbols: list[str] = ["SPY", "QQQ", "IWM"]
    intraday: bool = True


@app.post("/api/data/quickstart")
def data_quickstart(req: QuickReq):
    """Free, keyless first download: daily history + recent intraday bars + VIX (a few MB)."""
    payload = req.model_dump()
    rid = db.create_run("quickstart", "Free market data: " + ", ".join(req.symbols), payload)
    jobs.submit(rid, "quickstart", payload)
    return {"id": rid}


@app.get("/api/data/verify/{sym}")
def data_verify(sym: str):
    from strata.data.providers.sources import SourceError, verify_daily
    try:
        return J(verify_daily(store(), sym.upper()))
    except SourceError as e:
        raise HTTPException(502, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.get("/api/data/coverage")
def data_coverage():
    """Coverage matrix: one row per symbol, one column per dataset, with what is missing spelled out."""
    from strata.data.catalog import catalog
    rows = []
    for c in catalog(store()):
        d, intr, o, tr = c.get("daily"), c.get("intraday") or {}, c.get("options"), c.get("option_trades")
        gaps = []
        if not d:
            gaps.append("no daily history")
        if not intr:
            gaps.append("no intraday bars: time-of-day rules and intraday targets cannot be evaluated")
        elif "1m" not in intr:
            gaps.append("no 1-minute bars: entries are limited to times the stored bars can observe")
        if c["meta"].get("has_options") or o:
            if not o and not tr:
                gaps.append("no option prices stored: they are downloaded for the sessions a strategy needs, where a free source has them")
            elif o and o["resolution"] == "EOD" and not tr:
                gaps.append("option quotes are end-of-day only: intraday and 0DTE option backtests use option trade prints, fetched per strategy")
            if tr:
                gaps.append(f"option trade prints are stored selectively: {tr.get('contract_sessions') or 0} contract-sessions on {tr['sessions']} sessions (no bid/ask)")
        rows.append({"symbol": c["symbol"], "synthetic": bool(c["meta"].get("synthetic")), "kind": c["meta"].get("kind"),
                     "daily": d and {"first": d["first"], "last": d["last"], "sessions": d["sessions"], "missing": d["missing_sessions"], "provider": d.get("provider")},
                     "intraday": {k: {"first": v["first"], "last": v["last"], "sessions": v["sessions"], "incomplete": v["incomplete_sessions"], "provider": v.get("provider")} for k, v in intr.items()},
                     "options": o and {"first": o["first"], "last": o["last"], "sessions": o["sessions"], "missing": o["missing_count"], "resolution": o["resolution"], "quote_type": o["quote_type"],
                                       "provider": o["provider"], "license": o.get("license"), "rows": o["rows"]},
                     "option_trades": tr and {"first": tr["first"], "last": tr["last"], "sessions": tr["sessions"], "contracts": tr.get("contracts"), "contract_sessions": tr.get("contract_sessions"),
                                              "with_trades": tr.get("contract_sessions_with_trades"), "rows": tr.get("rows"), "provider": tr["provider"], "license": tr.get("license")},
                     "datasets": [{k: x.get(k) for k in ("id", "provider", "dtype", "obs", "resolution", "version", "first", "last", "sessions", "rows", "bytes", "license", "limitations", "synthetic", "head", "manifests")}
                                  for x in c.get("datasets", [])],
                     "iv_history": c.get("iv_history"), "storage_bytes": c.get("storage_bytes"), "dataset_version": c.get("dataset_version"), "sources": c.get("sources") or [], "gaps": gaps})
    return J(rows)


# ======================================================================= market studies
class StudyReq(BaseModel):
    config: dict
    name: str = ""
    force: bool = False


@app.post("/api/study/run")
def study_run(req: StudyReq):
    from strata.api.launch import launch_study
    from strata.engine.conditions import validate_tree
    from strata.research.study import StudyConfig
    try:
        cfg = StudyConfig.model_validate(req.config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    errs = validate_tree(cfg.condition)
    if errs:
        raise HTTPException(422, "; ".join(errs))
    if not store().has_daily(cfg.symbol) and not store().bar_resolutions(cfg.symbol):
        raise HTTPException(404, f"No historical price data is stored for {cfg.symbol}. Download it under Data; nothing is simulated in its place.")
    return launch_study(jobs, cfg, req.name, req.force)


@app.get("/api/study/session")
def study_session(symbol: str, day: date, context_days: int = 0):
    from strata.research.study import session_chart
    try:
        return J(session_chart(store(), symbol.upper(), day, context_days))
    except LookupError as e:
        raise HTTPException(404, str(e))


@app.get("/api/study/runs")
def study_runs(limit: int = 100):
    return J(db.list_runs(kind="study", limit=limit))


# ======================================================================= expressions
@app.post("/api/expr/validate")
def expr_validate(expr: str = Body(..., embed=True)):
    from strata.ai.agent import tool_validate_expression
    return tool_validate_expression(expr)


# ======================================================================= settings
@app.get("/api/settings")
def settings_get():
    from strata.ai.llm import status
    from strata.api.settings import public_settings
    return J({**public_settings(), "llm": status()})


class SettingsReq(BaseModel):
    settings: dict[str, Any] = {}
    secrets: dict[str, Optional[str]] = {}


@app.put("/api/settings")
def settings_put(req: SettingsReq):
    """Keys are write-only: they are stored server-side and never sent back to the browser."""
    from strata.api.settings import DEFAULTS, SECRETS, set_secret, set_setting
    for k, v in req.settings.items():
        if k not in DEFAULTS:
            raise HTTPException(422, f"unknown setting {k}")
        set_setting(k, v)
    for k, v in req.secrets.items():
        if k not in SECRETS:
            raise HTTPException(422, f"unknown secret {k}")
        try:
            set_secret(k, v)
        except RuntimeError as e:
            raise HTTPException(500, f"The credential could not be stored: {e}")
    return settings_get()


@app.get("/api/settings/ollama")
def settings_ollama():
    from strata.ai.llm import ollama_models
    return {"models": ollama_models()}


@app.post("/api/settings/llm/test")
def settings_llm_test(provider: str = Body(..., embed=True)):
    """One tiny request to confirm the provider answers. No result or strategy is involved."""
    from strata.ai.llm import LLMError, build
    try:
        m = build(provider)
        out = m.json("You are a connectivity check.", [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}], max_tokens=200)
        return {"ok": bool(out.get("ok", True)), "model": m.name}
    except LLMError as e:
        return {"ok": False, "error": str(e)}


# ======================================================================= AI research lab
def _agent():
    from strata.ai.agent import Agent
    return Agent(jobs, store())


def _ai(fn):
    try:
        return J(fn())
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/ai/meta")
def ai_meta():
    from strata.ai.llm import status
    from strata.ai.nl_rules import EXAMPLES, SWEEP_FIELDS
    from strata.api.settings import get_setting
    groups = [
        {"title": "Ask a market question", "items": ["How often does SPY rise 1% after a 3% down day?", "How often does SPY touch yesterday's high before touching yesterday's low?", "Does the first 15-minute candle predict the rest of the trading session?"]},
        {"title": "Investigate a hypothesis", "items": ["Investigate whether SPY tends to reverse following large down days.", "What historical conditions are associated with a 1% rally before a 0.5% decline?",
                                                       "Compare mean reversion and momentum after overnight gaps."]},
        {"title": "Describe a strategy", "items": ["Wait until SPY drops 0.5% from the open, then buy calls when it recovers half of that decline.", "Buy a 30DTE ATM SPY call at the close after a 2% down day and hold for 5 days",
                                                   "Enter calls when SPY breaks yesterday's high, but only if QQQ hasn't broken its own high."]},
    ]
    follow = ["Challenge this result", "How much of the result is explained by volatility?", "Compare that with 2% and 4% down days", "What time of day does it usually hit the target?", "Show every qualifying historical session", "Test ATM 0DTE calls on those days using historical option trade prices",
              "Include commissions and conservative slippage", "Show me every skipped signal and why", "Is this statistically significant?", "Test whether the results hold up out of sample", "Export the complete research report"]
    return J({"examples": EXAMPLES, "example_groups": groups, "followups": follow, "sweep_fields": SWEEP_FIELDS, "llm": status(), "default_symbol": _default_symbol(), "exec_mode": get_setting("ai.exec_mode") or "quick"})


@app.get("/api/ai/sessions")
def ai_sessions():
    return J(db.list_sessions())


@app.post("/api/ai/sessions")
def ai_session_create(title: str = Body("New research", embed=True)):
    from strata.api.settings import get_setting
    return J(db.create_session(title[:80] or "New research", {"exec_mode": get_setting("ai.exec_mode") or "quick"}))


@app.get("/api/ai/sessions/{sid}")
def ai_session_get(sid: str):
    return _ai(lambda: _agent().payload(sid))


@app.patch("/api/ai/sessions/{sid}")
def ai_session_rename(sid: str, title: str = Body(..., embed=True)):
    db.update_session(sid, title=title[:80])
    return {"ok": True}


@app.delete("/api/ai/sessions/{sid}")
def ai_session_delete(sid: str):
    db.delete_session(sid)
    return {"ok": True}


@app.post("/api/ai/sessions/{sid}/messages")
def ai_message(sid: str, text: str = Body(..., embed=True)):
    return _ai(lambda: {"messages": _agent().post(sid, text)})


class AiRunReq(BaseModel):
    version_id: int
    mode: str


@app.post("/api/ai/sessions/{sid}/run")
def ai_run(sid: str, req: AiRunReq):
    return _ai(lambda: {"messages": _agent().run(sid, req.version_id, req.mode)})


class AiEditReq(BaseModel):
    draft: dict
    run: bool = True


@app.post("/api/ai/sessions/{sid}/edit")
def ai_edit(sid: str, req: AiEditReq):
    def go():
        try:
            return {"messages": _agent().edit(sid, req.draft, req.run)}
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "ValidationError":
                raise ValueError(str(e)) from None
            raise
    return _ai(go)


class AiApplyReq(BaseModel):
    field: str
    value: Any


@app.post("/api/ai/sessions/{sid}/apply")
def ai_apply(sid: str, req: AiApplyReq):
    return _ai(lambda: {"messages": _agent().apply(sid, req.field, req.value)})


@app.post("/api/ai/sessions/{sid}/restore")
def ai_restore(sid: str, version_id: int = Body(..., embed=True)):
    return _ai(lambda: {"messages": _agent().restore(sid, version_id)})


@app.post("/api/ai/sessions/{sid}/finalize")
def ai_finalize(sid: str):
    return _ai(lambda: {"messages": _agent().finalize(sid)})


@app.post("/api/ai/interpret")
def ai_interpret(text: str = Body(..., embed=True)):
    """Stateless preview: how a request is read, what is assumed, and which mode the data supports. Runs nothing."""
    from strata.ai.draft import compile_draft
    s = store()
    r = _agent()._rules(text, None, sorted(s.stored_symbols()), None, _default_symbol())
    if r["action"] != "new":
        return J({"understood": r["action"] in ("investigate", "discover"), "action": r["action"]})
    d = r["draft"]
    compiled = compile_draft(d, s)
    return J({"understood": True, "action": "new", "draft": d.model_dump(mode="json"), "items": compiled["items"], "assumptions": compiled["assumptions"], "missing": compiled["missing"], "errors": compiled["errors"],
              "unsupported": d.unsupported, "questions": d.questions})


# ======================================================================= strategies
@app.get("/api/strategies")
def strategies_list():
    out = db.list_strategies()
    runs = db.list_runs(kind="backtest", limit=1000)
    last = {}
    for r in runs:
        if r["strategy_id"] and r["strategy_id"] not in last and r["status"] == "done":
            last[r["strategy_id"]] = {"id": r["id"], "summary": r["summary"], "created_at": r["created_at"]}
    for s in out:
        s["last_run"] = last.get(s["id"])
    return J(out)


@app.get("/api/strategies/{sid}")
def strategy_get(sid: str):
    s = db.get_strategy(sid)
    if not s:
        raise HTTPException(404)
    return J(s)


class StrategySave(BaseModel):
    config: dict
    notes: str = ""
    tags: list[str] = []


@app.post("/api/strategies")
def strategy_create(body: StrategySave):
    cfg = StrategyConfig.model_validate(body.config)
    return J(db.save_strategy(cfg.model_dump(mode="json"), notes=body.notes, tags=body.tags))


@app.put("/api/strategies/{sid}")
def strategy_update(sid: str, body: StrategySave):
    cfg = StrategyConfig.model_validate(body.config)
    return J(db.save_strategy(cfg.model_dump(mode="json"), sid=sid, notes=body.notes, tags=body.tags))


@app.post("/api/strategies/{sid}/duplicate")
def strategy_dup(sid: str):
    s = db.get_strategy(sid)
    if not s:
        raise HTTPException(404)
    cfg = dict(s["config"], name=s["config"]["name"] + " (copy)")
    return J(db.save_strategy(cfg, notes=s.get("notes", ""), tags=s.get("tags") or []))


@app.delete("/api/strategies/{sid}")
def strategy_delete(sid: str):
    db.delete_strategy(sid)
    return {"ok": True}


@app.post("/api/strategies/examples")
def strategy_examples():
    """Seed the curated example strategies (skips names that already exist)."""
    from strata.engine.templates import examples
    have = {s["name"] for s in db.list_strategies()}
    out = []
    sym, eod, _ = _template_args()
    for ex in examples(sym, eod):
        if ex["name"] in have:
            continue
        cfg = StrategyConfig.model_validate(ex)
        out.append(db.save_strategy(cfg.model_dump(mode="json"), notes="example", tags=["example"]))
    return J(out)


@app.post("/api/strategies/validate")
def strategy_validate(config: dict = Body(...)):
    try:
        StrategyConfig.model_validate(config)
        return {"ok": True, "errors": []}
    except Exception as e:  # noqa: BLE001
        errs = getattr(e, "errors", lambda: [{"msg": str(e), "loc": []}])()
        return J({"ok": False, "errors": [{"loc": ".".join(str(x) for x in er.get("loc", [])), "msg": er.get("msg")} for er in errs]})


@app.post("/api/strategies/optimizable")
def strategy_optimizable(config: dict = Body(...)):
    from strata.optimize.optimizer import optimizable_params
    return J(optimizable_params(config))


class PreviewReq(BaseModel):
    strategy: dict
    day: Optional[date] = None
    time: str = "10:00"
    execution: Optional[dict] = None


@app.post("/api/strategies/preview")
def strategy_preview(req: PreviewReq):
    from strata.api.builder import preview
    try:
        return J(preview(req.strategy, req.day, req.time, req.execution))
    except Exception as e:  # noqa: BLE001
        return J({"error": f"{type(e).__name__}: {e}"})


# ======================================================================= backtests
def _cache_key(cfg: BacktestConfig) -> tuple[str, dict, bool]:
    from strata.api.launch import backtest_cache_key
    return backtest_cache_key(cfg, store())


class BacktestReq(BaseModel):
    config: dict
    strategy_id: Optional[str] = None
    name: str = ""
    force: bool = False


@app.post("/api/backtests")
def backtest_create(req: BacktestReq):
    try:
        cfg = BacktestConfig.model_validate(req.config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    from strata.api.launch import launch_backtest
    return launch_backtest(jobs, cfg, req.name, req.strategy_id, req.force)


@app.get("/api/backtests")
def backtest_list(kind: str = "backtest,portfolio", strategy_id: Optional[str] = None, limit: int = 200):
    return J(db.list_runs(kind=kind, strategy_id=strategy_id, limit=limit))


@app.get("/api/runs/{rid}")
def run_get(rid: str):
    r = db.get_run(rid)
    if not r:
        raise HTTPException(404, "run not found")
    if r["status"] == "done" and r["kind"] in ("backtest", "portfolio"):
        try:
            r["result"] = load_result(rid)
        except FileNotFoundError:
            if (r.get("summary") or {}).get("outcome") != "needs_decision":  # a research run that stopped to ask has no result by design
                r["status"], r["error"] = "error", "result files missing"
    elif r["status"] == "done" and r["kind"] in ("study", "discovery"):
        p = run_dir(rid) / f"{r['kind']}.json"
        if p.exists():
            r["result"] = json.loads(p.read_text())
        elif (r.get("summary") or {}).get("outcome") != "needs_decision":
            r["status"], r["error"] = "error", "result files missing"
    elif r["status"] == "done" and r["kind"] == "optimization":
        p = run_dir(rid) / "optimization.json"
        if p.exists():
            r["result"] = json.loads(p.read_text())
    elif r["status"] == "done" and r["kind"] == "entry_timing":
        p = run_dir(rid) / "entry_timing.json"
        if p.exists():
            r["result"] = json.loads(p.read_text())
    return J(r)


@app.delete("/api/runs/{rid}")
def run_delete(rid: str):
    if jobs:
        jobs.cancel(rid)
    delete_results(rid)
    db.delete_run(rid)
    return {"ok": True}


@app.get("/api/backtests/{rid}/trades")
def backtest_trades(rid: str, compact: bool = True):
    tr = load_trades(rid)
    if compact:
        for t in tr:
            t.pop("entry_explain", None)
            t.pop("exit_explain", None)
            t.pop("meta", None)
    return J(tr)


@app.get("/api/backtests/{rid}/trades.csv")
def backtest_trades_csv(rid: str):
    tr = load_trades(rid)
    cols = ["id", "strategy", "symbol", "entry_time", "exit_time", "units", "entry_cash", "exit_cash", "commissions", "slippage", "pnl", "max_risk", "return_on_risk",
            "exit_reason", "entry_spot", "exit_spot", "entry_dte", "short_delta", "entry_vix", "entry_iv", "mfe", "mae", "flags"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols + ["legs"])
    for t in tr:
        w.writerow([t.get(c) if c != "flags" else "|".join(t.get("flags", [])) for c in cols] + [" ".join(f"{l['side']}{l['contracts']}x{l['strike']}{l['right'] or ''}@{l['entry_price']:.2f}->{(l['exit_price'] or 0):.2f}" for l in t["legs"])])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=trades_{rid}.csv"})


@app.get("/api/backtests/{rid}/trades/{tid}")
def backtest_trade(rid: str, tid: int):
    try:
        return J(trade_replay(rid, tid, store()))
    except KeyError:
        raise HTTPException(404, "trade not found")


@app.get("/api/backtests/{rid}/intraday")
def backtest_intraday(rid: str, max_points: int = 4000):
    p = run_dir(rid) / "intraday.parquet"
    if not p.exists():
        return {"t": [], "equity": []}
    df = pl.read_parquet(p)
    step = max(1, df.height // max_points)
    df = df.gather_every(step)
    return {"t": df["t"].cast(pl.Int64).to_list(), "equity": df["equity"].to_list()}


class MCReq(BaseModel):
    method: str = "trade"
    n_sims: int = 2000
    block: int = 10
    seed: int = 7
    ruin_pct: float = 0.5


@app.post("/api/backtests/{rid}/montecarlo")
def backtest_mc(rid: str, req: MCReq):
    from strata.analytics.montecarlo import block_bootstrap, trade_bootstrap
    res = load_result(rid)
    init = res["config"]["capital"]["initial_capital"]
    if req.method == "block":
        return J(block_bootstrap(res["series"]["daily_return"], init, min(req.n_sims, 20000), max(2, req.block), req.seed))
    return J(trade_bootstrap([t["pnl"] for t in load_trades(rid)], init, min(req.n_sims, 20000), req.seed, req.ruin_pct))


# ======================================================================= portfolio / optimisation
@app.post("/api/portfolios")
def portfolio_create(req: BacktestReq):
    cfg = BacktestConfig.model_validate(req.config)
    if sum(s.allocation for s in cfg.strategies) > 1.0 + 1e-9:
        raise HTTPException(422, "strategy allocations must sum to at most 100%")
    key, vers, synth = _cache_key(cfg)
    rid = db.create_run("portfolio", req.name or "Portfolio: " + " + ".join(s.name for s in cfg.strategies), cfg.model_dump(mode="json"), None, "pf:" + key, cfg.config_hash(), vers, ENGINE_VERSION, synth)
    jobs.submit(rid, "portfolio", {"config": cfg.model_dump(mode="json")})
    return {"id": rid}


class OptReq(BaseModel):
    config: dict
    strategy_id: Optional[str] = None
    name: str = ""


@app.post("/api/optimizations")
def optimization_create(req: OptReq):
    from strata.optimize.optimizer import OptimizationConfig
    cfg = dict(req.config)
    if req.strategy_id:
        s = db.get_strategy(req.strategy_id)
        if s:
            cfg["prior_trials"] = int(s.get("trials") or 0)
    oc = OptimizationConfig.model_validate(cfg)
    key, vers, synth = _cache_key(oc.base)
    rid = db.create_run("optimization", req.name or f"Optimise {oc.base.strategies[0].name} ({oc.method}, {oc.objective})", oc.model_dump(mode="json"), req.strategy_id,
                        None, oc.base.config_hash(), vers, ENGINE_VERSION, synth)
    jobs.submit(rid, "optimization", {"config": oc.model_dump(mode="json"), "strategy_id": req.strategy_id})
    return {"id": rid}


@app.get("/api/optimizations")
def optimization_list():
    return J(db.list_runs(kind="optimization"))


@app.post("/api/optimizations/{rid}/apply")
def optimization_apply(rid: str, index: int = Body(..., embed=True)):
    """Return the base config with the chosen configuration's parameters applied."""
    from strata.optimize.optimizer import apply_params
    p = run_dir(rid) / "optimization.json"
    out = json.loads(p.read_text())
    params = out["configs"][index]["params"]
    return J(apply_params(out["config"]["base"], params))


# ======================================================================= research
class ResearchReq(BaseModel):
    template: str
    params: dict = {}


@app.post("/api/research/run")
def research_run(req: ResearchReq):
    from strata.research.queries import Research
    from strata.research.queries import TEMPLATES
    r = Research(store())
    if req.template not in TEMPLATES:  # the named studies only: never an arbitrary attribute of the object
        raise HTTPException(404, f"unknown research template {req.template}")
    fn = getattr(r, req.template)
    p = dict(req.params)
    try:
        for k in ("start", "end"):
            if p.get(k):
                p[k] = date.fromisoformat(str(p[k]))
    except ValueError:
        raise HTTPException(422, "start and end must be dates (YYYY-MM-DD)")
    try:
        return J(fn(**p))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e)[:400])


class SqlReq(BaseModel):
    sql: str
    limit: int = 5000


@app.post("/api/research/sql")
def research_sql(req: SqlReq):
    from strata.research.queries import sql_query
    try:
        return J(sql_query(req.sql, store(), min(req.limit, 50000)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")


class TimingReq(BaseModel):
    config: dict
    path: str
    times: list[str]


@app.post("/api/research/entry-timing")
def research_timing(req: TimingReq):
    BacktestConfig.model_validate(req.config)
    rid = db.create_run("entry_timing", f"Entry-time sweep ({len(req.times)} times)", req.model_dump())
    jobs.submit(rid, "entry_timing", req.model_dump())
    return {"id": rid}


@app.get("/api/research/runs")
def research_runs():
    return J(db.list_runs(kind="entry_timing", limit=50))


# ======================================================================= jobs
@app.get("/api/jobs")
def jobs_list():
    recent = db.list_runs(limit=30)
    return J({"active": jobs.active() if jobs else [], "recent": recent})


@app.post("/api/jobs/{rid}/cancel")
def job_cancel(rid: str):
    return {"ok": bool(jobs and jobs.cancel(rid))}


@app.get("/api/jobs/{rid}/stream")
async def job_stream(rid: str):
    """Server-sent events: one JSON message per progress update until a terminal state."""
    async def gen():
        q = jobs.subscribe(rid)
        try:
            st = jobs.snapshot(rid) or db.get_run(rid) or {"id": rid, "status": "unknown"}
            yield f"data: {json.dumps(_clean({k: st.get(k) for k in ('id', 'status', 'progress', 'message', 'error')}))}\n\n"
            if st.get("status") in ("done", "error", "cancelled", "unknown"):
                return
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(_clean(msg))}\n\n"
                if msg.get("status") in ("done", "error", "cancelled"):
                    if msg.get("status") == "error":
                        r = db.get_run(rid) or {}
                        yield f"data: {json.dumps({'id': rid, 'status': 'error', 'message': r.get('message'), 'error': (r.get('error') or '')[:2000]})}\n\n"
                    return
        finally:
            jobs.unsubscribe(rid, q)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ======================================================================= static frontend
def mount_frontend(application: FastAPI, dist: Path) -> bool:
    """Serve the built interface: its files, and the application shell for every other path (client-side routing).

    Only files INSIDE the build directory are ever served. The path arrives percent-decoded, so `..%2f..%2f` reaches this
    function as `../../`: it is resolved and must still lie under the build directory, otherwise the shell is returned.
    An unknown /api path is a 404, not the shell: a caller must not mistake a page of HTML for an answer."""
    dist = Path(dist)
    if not dist.exists():
        return False
    root = dist.resolve()
    if (root / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @application.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "no such endpoint")
        if path:
            try:
                f = (root / path).resolve()
            except (OSError, ValueError):
                f = None
            if f is not None and f.is_relative_to(root) and f.is_file():
                return FileResponse(f)
        return FileResponse(root / "index.html")

    return True


_dist = Path(__import__("os").environ.get("STRATA_FRONTEND_DIST") or Path(__file__).resolve().parents[3] / "frontend" / "dist")
mount_frontend(app, _dist)
