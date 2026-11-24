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
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    db.init_db()
    n = db.mark_interrupted()
    jobs = JobManager(max_concurrent=int(__import__("os").environ.get("STRATA_MAX_JOBS", "2")))
    jobs.loop = asyncio.get_running_loop()
    if n:
        print(f"[strata] marked {n} interrupted runs as error")
    yield


app = FastAPI(title="Strata API", version=ENGINE_VERSION, lifespan=lifespan,
              description="Options backtesting & research platform. All long-running work (backtests, optimisations, ingestion) runs as background jobs; poll /api/jobs or stream /api/jobs/{id}/stream.")
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins) + ["*"], allow_methods=["*"], allow_headers=["*"])


def J(o) -> JSONResponse:
    return JSONResponse(content=json.loads(dumps(o)))


def store() -> MarketStore:
    return MarketStore()


# ======================================================================= meta
@app.get("/api/health")
def health():
    return {"ok": True, "engine_version": ENGINE_VERSION, "data_dir": str(get_settings().data_dir)}


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
    real = [k for k, m in syms.items() if not m.synthetic and (store().daily_path(k).exists())]
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
    if sym not in s.symbols() and not (s.root / "underlying" / sym).exists():
        raise HTTPException(404, f"unknown symbol {sym}")
    return J(symbol_coverage(s, sym) | {"manifest": {k: v for k, v in s.manifest(sym).items() if k != "files"}, "files": len(s.manifest(sym).get("files", {}))})


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
    root = s.root
    parts = {}
    for sub in ["underlying", "options", "daily", "derived", "events", "corporate_actions"]:
        p = root / sub
        parts[sub] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
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
        d, intr, o = c.get("daily"), c.get("intraday") or {}, c.get("options")
        gaps = []
        if not d:
            gaps.append("no daily history")
        if not intr:
            gaps.append("no intraday bars: time-of-day rules and intraday targets cannot be evaluated")
        elif "1m" not in intr:
            gaps.append("no 1-minute bars: entries are limited to times the stored bars can observe")
        if c["meta"].get("has_options") or o:
            if not o:
                gaps.append("no option quotes: only market studies and modeled estimates are possible")
            elif o["resolution"] == "EOD":
                gaps.append("option quotes are end-of-day only: no intraday or 0DTE option backtests")
        rows.append({"symbol": c["symbol"], "synthetic": bool(c["meta"].get("synthetic")), "kind": c["meta"].get("kind"),
                     "daily": d and {"first": d["first"], "last": d["last"], "sessions": d["sessions"], "missing": d["missing_sessions"], "provider": d.get("provider")},
                     "intraday": {k: {"first": v["first"], "last": v["last"], "sessions": v["sessions"], "incomplete": v["incomplete_sessions"], "provider": v.get("provider")} for k, v in intr.items()},
                     "options": o and {"first": o["first"], "last": o["last"], "sessions": o["sessions"], "missing": o["missing_count"], "resolution": o["resolution"], "quote_type": o["quote_type"],
                                       "provider": o["provider"], "license": o.get("license"), "rows": o["rows"]},
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
    if not store().daily_path(cfg.symbol).exists() and not store().bar_resolutions(cfg.symbol):
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
    from strata.api.settings import DEFAULTS, SECRETS, set_setting
    for k, v in req.settings.items():
        if k not in DEFAULTS:
            raise HTTPException(422, f"unknown setting {k}")
        set_setting(k, v)
    for k, v in req.secrets.items():
        if k not in SECRETS:
            raise HTTPException(422, f"unknown secret {k}")
        set_setting(f"secret.{k}", (v or "").strip() or None)
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
    return J({"examples": EXAMPLES, "sweep_fields": SWEEP_FIELDS, "llm": status(), "default_symbol": _default_symbol()})


@app.get("/api/ai/sessions")
def ai_sessions():
    return J(db.list_sessions())


@app.post("/api/ai/sessions")
def ai_session_create(title: str = Body("New research", embed=True)):
    return J(db.create_session(title[:80] or "New research"))


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
    from strata.ai import nl_rules as NL
    from strata.ai.draft import assess, compile_draft
    s = store()
    r = NL.interpret(text, None, list(s.symbols().keys()))
    if r["action"] != "new":
        return {"understood": False}
    d = r["draft"]
    compiled = compile_draft(d, s)
    cover = assess(d, compiled, s)
    return J({"understood": True, "draft": d.model_dump(mode="json"), "items": compiled["items"], "assumptions": compiled["assumptions"], "missing": compiled["missing"], "errors": compiled["errors"],
              "coverage": {"modes": cover["modes"], "recommended": cover["recommended"], "messages": cover["messages"]}})


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
            r["status"], r["error"] = "error", "result files missing"
    elif r["status"] == "done" and r["kind"] == "study":
        p = run_dir(rid) / "study.json"
        if p.exists():
            r["result"] = json.loads(p.read_text())
        else:
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
    r = Research(store())
    fn = getattr(r, req.template, None)
    if fn is None or req.template.startswith("_"):
        raise HTTPException(404, f"unknown research template {req.template}")
    p = dict(req.params)
    for k in ("start", "end"):
        if p.get(k):
            p[k] = date.fromisoformat(p[k])
    try:
        return J(fn(**p))
    except ValueError as e:
        raise HTTPException(400, str(e))


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
_dist = Path(__import__("os").environ.get("STRATA_FRONTEND_DIST") or Path(__file__).resolve().parents[3] / "frontend" / "dist")
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _dist / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
