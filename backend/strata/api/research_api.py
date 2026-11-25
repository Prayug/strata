"""REST endpoints of the research workspace (V3): interacting with a research session,
inspecting a run (pipeline, calendar, a single day, observations, robustness), comparing
runs, exporting, and the provider connection checks. Mounted by api/app.py.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from strata.api import db
from strata.api.results import dumps, load_run_result, load_trades, run_dir
from strata.data.store import MarketStore

router = APIRouter()


def J(o) -> JSONResponse:
    return JSONResponse(content=json.loads(dumps(o)))


def _agent():
    from strata.api import app as appmod
    from strata.ai.agent import Agent
    return Agent(appmod.jobs, MarketStore())


def _ai(fn):
    try:
        return J(fn())
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


def _run(rid: str) -> tuple[dict, dict]:
    run = db.get_run(rid)
    if not run:
        raise HTTPException(404, "run not found")
    if run["status"] != "done" or (run.get("summary") or {}).get("outcome") == "needs_decision":
        raise HTTPException(409, "this run has no result")
    try:
        return run, load_run_result(run)
    except FileNotFoundError:
        raise HTTPException(404, "result files are missing")


# ----------------------------------------------------------------------------- research sessions
class AnswerReq(BaseModel):
    version_id: int
    question_id: str
    option: int


@router.post("/api/ai/sessions/{sid}/answer")
def ai_answer(sid: str, req: AnswerReq):
    return _ai(lambda: {"messages": _agent().answer(sid, req.version_id, req.question_id, req.option)})


class ConfirmReq(BaseModel):
    version_id: int
    mode: Optional[str] = None


@router.post("/api/ai/sessions/{sid}/confirm")
def ai_confirm(sid: str, req: ConfirmReq):
    return _ai(lambda: {"messages": _agent().confirm(sid, req.version_id, req.mode)})


class DecideReq(BaseModel):
    message_id: int
    choice: str


@router.post("/api/ai/sessions/{sid}/decide")
def ai_decide(sid: str, req: DecideReq):
    return _ai(lambda: {"messages": _agent().decide(sid, req.message_id, req.choice)})


@router.post("/api/ai/sessions/{sid}/mode")
def ai_mode(sid: str, exec_mode: str = Body(..., embed=True)):
    return _ai(lambda: _agent().set_mode(sid, exec_mode))


class PinReq(BaseModel):
    run_id: str
    on: bool = True


@router.post("/api/ai/sessions/{sid}/pin")
def ai_pin(sid: str, req: PinReq):
    return _ai(lambda: _agent().pin(sid, req.run_id, req.on))


@router.get("/api/ai/search")
def ai_search(q: str = "", limit: int = 40):
    """Saved research notebooks (sessions) and their runs, searchable by any word in the
    title, a question that was asked, or a run's name."""
    words = [w for w in q.lower().split() if w]
    out = []
    for s in db.list_sessions(limit=300):
        msgs = db.list_messages(s["id"])
        asked = [m["content"].get("text", "") for m in msgs if m["role"] == "user" and isinstance(m["content"], dict)]
        hay = (s["title"] + " " + " ".join(asked)).lower()
        if words and not all(w in hay for w in words):
            continue
        runs = []
        for m in msgs:
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] == "result" and c.get("run_id"):
                runs.append({"run_id": c["run_id"], "label": c.get("label"), "view": c.get("view"), "mode": c.get("mode")})
        hit = next((a for a in asked if words and any(w in a.lower() for w in words)), asked[0] if asked else "")
        out.append({"id": s["id"], "title": s["title"], "updated_at": s["updated_at"], "created_at": s["created_at"], "questions": len(asked), "results": len(runs), "match": hit[:160], "runs": runs[-4:]})
        if len(out) >= limit:
            break
    return J(out)


class PreviewReq(BaseModel):
    draft: dict
    day: Optional[date] = None


@router.post("/api/ai/preview")
def ai_preview(req: PreviewReq):
    """Interpretation preview: how every rule of a draft evaluates on an actual historical
    session, tick by tick. Nothing is traded and nothing is stored."""
    from strata.ai.draft import Draft, compile_draft
    from strata.engine.scan import scan_signals
    from strata.research.explain_day import explain_day, nearest_session, sample_dates, study_as_strategy
    s = MarketStore()
    try:
        d = Draft.model_validate(req.draft)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e)[:400])
    compiled = compile_draft(d, s)
    cfg = {"strategies": [compiled["strategy"]], "execution": {"signal_interval_minutes": 1}} if compiled.get("strategy") else study_as_strategy(compiled["study"]) if compiled.get("study") else None
    if cfg is None:
        raise HTTPException(422, "; ".join(compiled["errors"]) or "this request cannot be previewed yet")
    daily = s.read_daily(d.symbol)
    if daily.is_empty():
        return J({"available": False, "reason": f"No prices are stored for {d.symbol} yet. They are downloaded automatically when this is run."})
    last = daily["date"][-1]
    samples = None
    if req.day is None:
        # recent sessions on which the rule fires and on which it does not, so both can be inspected
        from datetime import timedelta
        try:
            scan = scan_signals({**cfg, "start": (last - timedelta(days=900)).isoformat(), "end": last.isoformat()}, s)
            samples = sample_dates(s, d.symbol, [x["date"].isoformat() for x in scan["signals"]])
        except Exception:  # noqa: BLE001
            samples = {"qualifying": [], "not_qualifying": []}
    day = req.day or (date.fromisoformat(samples["qualifying"][0]) if samples and samples["qualifying"] else last)
    out = explain_day(cfg, nearest_session(day), s)
    out.update({"available": True, "samples": samples, "last_session": last.isoformat()})
    return J(out)


@router.get("/api/ai/sessions/{sid}/report")
def ai_report(sid: str, format: str = "json", run: Optional[str] = None):
    from strata.research.report import build_report, render_html, render_markdown
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    msgs = db.list_messages(sid)
    rid = run
    if rid is None:
        rid = next((m["content"].get("run_id") for m in reversed(msgs) if m["kind"] == "result" and isinstance(m["content"], dict) and m["content"].get("run_id")), None)
    if rid is None:
        raise HTTPException(404, "this session has no finished result yet")
    r, res = _run(rid)
    ctx = report_context(sid, rid, msgs, s)
    rep = build_report(r, res, ctx, trials=max(1, int((s.get("state") or {}).get("trials") or 1)))
    if format == "html":
        return Response(render_html(rep), media_type="text/html", headers={"Content-Disposition": f'inline; filename="strata-report-{rid}.html"'})
    if format == "md":
        return Response(render_markdown(rep), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="strata-report-{rid}.md"'})
    return J(rep)


def report_context(sid: str | None, rid: str, msgs: list[dict] | None = None, session: dict | None = None) -> dict:
    """The question and interpretation that led to a run, read from its research session."""
    if sid is None:
        return {}
    msgs = msgs if msgs is not None else db.list_messages(sid)
    session = session or db.get_session(sid) or {}
    plan, question = None, None
    for i, m in enumerate(msgs):
        c = m["content"] if isinstance(m["content"], dict) else {}
        refs = [x.get("run_id") for x in (c.get("runs") or []) + (c.get("compare_with") or [])]
        if rid in refs:
            plan = c
            question = next((x["content"].get("text") for x in reversed(msgs[:i]) if x["role"] == "user" and isinstance(x["content"], dict)), None)
            if m["kind"] == "plan":
                break
    if plan is None:
        return {"title": session.get("title")}
    if plan.get("items") is None and plan.get("version_id"):
        base = next((m["content"] for m in msgs if m["kind"] == "plan" and isinstance(m["content"], dict) and m["content"].get("version_id") == plan["version_id"]), None)
        plan = {**(base or {}), **{k: v for k, v in plan.items() if v is not None and k not in ("items", "assumptions")}} if base else plan
    return {"title": session.get("title") or plan.get("title"), "question": question or plan.get("title"), "items": plan.get("items") or [], "assumptions": plan.get("assumptions") or [],
            "unsupported": plan.get("unsupported") or [], "notes": plan.get("notes") or [], "interpreter": plan.get("interpreter"), "version_id": plan.get("version_id"), "session_id": sid}


# ----------------------------------------------------------------------------- one run
@router.get("/api/runs/{rid}/pipeline")
def run_pipeline_state(rid: str):
    """Stage-by-stage progress of a research run (also readable while it is running)."""
    from strata.research.pipeline import STAGES, load_pipeline
    run = db.get_run(rid)
    if not run:
        raise HTTPException(404, "run not found")
    p = load_pipeline(rid)
    if p is None:
        p = {"run_id": rid, "stages": [{"id": i, "label": l, "status": "pending", "detail": "", "data": {}} for i, l in STAGES] if run["status"] in ("queued", "running") else [], "outcome": None, "downloads": []}
    return J({**p, "status": run["status"], "progress": run.get("progress"), "message": run.get("message"), "error": (run.get("error") or "").split("\n")[0][:400] or None})


@router.get("/api/runs/{rid}/calendar")
def run_calendar_view(rid: str):
    from strata.research.views import run_calendar
    run, res = _run(rid)
    return J(run_calendar(run, res))


@router.get("/api/runs/{rid}/observations")
def run_observations(rid: str, which: str = "all", limit: int = 5000):
    from strata.research.views import observations
    run, res = _run(rid)
    return J(observations(run, res, which, limit))


@router.get("/api/runs/{rid}/robustness")
def run_robustness(rid: str, trials: int = 1):
    from strata.research import stats as ST
    run, res = _run(rid)
    return J(ST.study_robustness(res, max(1, trials)) if run["kind"] == "study" else ST.backtest_robustness(res, load_trades(rid), max(1, trials)))


@router.get("/api/runs/{rid}/report")
def run_report(rid: str, format: str = "json", sid: Optional[str] = None, trials: int = 1):
    from strata.research.report import build_report, render_html, render_markdown
    run, res = _run(rid)
    rep = build_report(run, res, report_context(sid, rid), trials=max(1, trials))
    if format == "html":
        return Response(render_html(rep), media_type="text/html")
    if format == "md":
        return Response(render_markdown(rep), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="strata-report-{rid}.md"'})
    return J(rep)


@router.get("/api/runs/{rid}/export")
def run_export(rid: str, what: str = "observations"):
    """Download the raw material of a result: observations, skipped signals, the baseline,
    or the configuration that reproduces it."""
    from strata.research.report import csv_text
    from strata.research.views import observations
    run, res = _run(rid)
    if what == "config":
        body = {"run_id": rid, "kind": run["kind"], "engine_version": run.get("engine_version"), "config_hash": run.get("config_hash"), "dataset_versions": run.get("dataset_versions"), "created_at": run.get("created_at"),
                "config": res.get("config"), "coverage": {k: v for k, v in (res.get("coverage") or {}).items() if k not in ("skipped_signals",)}}
        return Response(json.dumps(json.loads(dumps(body)), indent=1), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="strata-{rid}-config.json"'})
    which = {"observations": "all", "skipped": "skipped", "baseline": "baseline", "excluded": "excluded"}.get(what)
    if which is None:
        raise HTTPException(422, "what must be observations, skipped, baseline, excluded or config")
    o = observations(run, res, which, limit=10**7)
    return Response(csv_text(o["columns"], o["rows"]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="strata-{rid}-{what}.csv"'})


@router.get("/api/runs/{rid}/day")
def run_day(rid: str, day: date):
    """Everything about one session of a run: the bars, how each rule evaluated through the
    day, whether a signal fired, what was (or was not) executed and why, and the observed
    option prices of the traded contracts."""
    from strata.core import calendar as cal
    from strata.research.explain_day import explain_day, study_as_strategy
    from strata.research.study import session_chart
    run, res = _run(rid)
    s = MarketStore()
    study = run["kind"] == "study"
    cfg = study_as_strategy(res["config"]) if study else {"strategies": res["config"]["strategies"][:1], "execution": {"signal_interval_minutes": (res["config"].get("execution") or {}).get("signal_interval_minutes", 5)}}
    out: dict = {"date": day.isoformat(), "kind": run["kind"]}
    out["rules"] = explain_day(cfg, day, s)
    sym = res["symbol"] if study else res["config"]["strategies"][0]["symbol"]
    try:
        out["chart"] = session_chart(s, sym, day, 0)
    except LookupError as e:
        out["chart"] = None
        out["chart_error"] = str(e)
    if study:
        row = next((r for r in res.get("sessions", []) if r["date"] == day.isoformat()), None)
        out["observation"] = row
        ex = next((k for k, v in (res.get("excluded_sessions") or {}).items() if day.isoformat() in v), None)
        base = res.get("baseline_sessions") or {"dates": [], "outcomes": ""}
        bi = base["dates"].index(day.isoformat()) if day.isoformat() in base["dates"] else -1
        if row:
            what = {"target": "the target was reached first", "stop": "the stop was reached first", "neither": "neither level was reached", "ambiguous": "target and stop fell inside the same bar, so their order is unknown"}[row["outcome"]]
            out["status"], out["why"] = row["outcome"], f"Qualified: the condition held. Observed from {cal.fmt_hhmm(row.get('entry_minute', 570))} ET at {row['entry_price']:.2f}; {what}. The session ended {row['end_return_pct']:+.2f}% from that price."
        elif ex:
            out["status"], out["why"] = "missing", f"Excluded: {ex}. A session that cannot be evaluated is not counted as one where the condition failed."
        elif bi >= 0:
            out["status"], out["why"] = "none", out["rules"].get("summary") or "The condition did not hold on this session."
        else:
            out["status"], out["why"] = "missing", "This session is outside the evaluated period or has no stored prices."
        return J(out)
    trades = [t for t in load_trades(rid) if t["entry_date"] == day.isoformat() or (t.get("exit_time") or "")[:10] == day.isoformat()]
    cov = res.get("coverage") or {}
    skip = next((r for r in ((cov.get("skipped_signals") or {}).get("rows") or []) if r["date"] == day.isoformat()), None)
    ex = next((k for k, v in ((res.get("diagnostics") or {}).get("excluded_sessions") or {}).items() if day.isoformat() in v), None)
    out["trades"] = [{"id": t["id"], "lot": t.get("lot"), "signal_time": cal.fmt_hhmm(cal.minute_of_day_et(t["signal_ts"] * 1_000_000_000)), "entry_time": t["entry_time"][11:16], "exit_time": (t.get("exit_time") or "")[11:16],
                      "exit_date": (t.get("exit_time") or "")[:10], "fill_delay_seconds": t.get("fill_delay_seconds"), "exit_delay_seconds": t.get("exit_delay_seconds"), "pnl": t["pnl"], "exit_reason": t.get("exit_reason"), "entry_spot": t.get("entry_spot"),
                      "exit_spot": t.get("exit_spot"), "signal_ts": t["signal_ts"], "entry_ts": t["entry_ts"], "exit_ts": t.get("exit_ts"), "flags": t.get("flags"), "legs": [{k: l.get(k) for k in ("right", "strike", "expiration", "side", "contracts", "price_source",
                      "entry_price", "exit_price", "entry_print", "exit_print", "entry_bid", "entry_ask", "exit_bid", "exit_ask", "exit_reason")} for l in t["legs"]],
                      "entry_explain": [x["text"] for x in (t.get("entry_explain") or [])][:10], "exit_explain": [x["text"] for x in (t.get("exit_explain") or [])][:6]} for t in trades]
    # observed prices of the traded contracts on that day (trade prints or quote marks recorded by the engine)
    series = []
    pricing = (res.get("meta") or {}).get("pricing")
    used = ((res.get("meta") or {}).get("datasets_used") or {}).get(sym) or {}
    if trades and pricing == "trades" and used.get("format") == "bars":
        from strata.data.store import DatasetKey
        bars = s.read_option_bars(sym, day, dataset=DatasetKey.parse(used["dataset"]))
        if bars is not None:
            import polars as pl
            for l in {(x["expiration"], x["strike"], x["right"]) for t in trades for x in t["legs"] if x.get("expiration")}:
                b = bars.filter((pl.col("expiration") == date.fromisoformat(l[0])) & (pl.col("strike") == l[1]) & (pl.col("right") == l[2])).sort("ts")
                series.append({"label": f"{l[1]:g}{l[2]} {l[0]}", "kind": "trade prints (1-minute bars)", "points": [[int(t.timestamp()) + 60, float(c), int(v or 0)] for t, c, v in zip(b["ts"].to_list(), b["close"].to_list(), b["volume"].to_list())]})
    elif trades:
        import polars as pl
        lp = run_dir(rid) / "legs.parquet"
        if lp.exists():
            legs = pl.read_parquet(lp)
            a, b_ = cal.session_bounds(day)
            for t in trades:
                g = legs.filter((pl.col("pos") == t["id"]) & (pl.col("t") >= a.timestamp()) & (pl.col("t") <= b_.timestamp())).sort(["leg", "t"])
                for li in sorted(set(g["leg"].to_list())):
                    x = g.filter(pl.col("leg") == li)
                    leg = t["legs"][int(li)]
                    series.append({"label": f"{leg['strike']:g}{leg['right']} {leg['expiration']}", "kind": "theoretical marks" if pricing == "modeled" else "quote mid marks",
                                   "points": [[int(tt), float(m)] for tt, m in zip(x["t"].to_list(), x["mid"].to_list()) if m is not None and m == m]})
    out["option_series"] = series
    out["pricing"] = pricing
    if trades:
        t0 = out["trades"][0]
        tot = sum(t["pnl"] for t in trades)
        out["status"] = "win" if tot > 0 else "loss"
        out["why"] = (f"Signal at {t0['signal_time']} ET, filled at {t0['entry_time']} ET" + (f" ({int(t0['fill_delay_seconds'] // 60)} min later: the first usable option price)" if t0.get("fill_delay_seconds") else "")
                      + f". Closed {('the same day at ' + t0['exit_time']) if t0['exit_date'] == day.isoformat() else 'on ' + t0['exit_date']} ({(t0.get('exit_reason') or '').replace('_', ' ')}). Result {tot:+,.2f}.")
    elif skip:
        out["status"], out["skip"] = "skipped", skip
        out["why"] = f"The entry rule fired at {cal.fmt_hhmm(skip['minute'])} ET, but no trade was made: {skip['reason']}." + (" This is a data limitation, not a failed rule." if skip.get("class") == "data" else "")
    elif ex:
        out["status"], out["why"] = "missing", f"Excluded: {ex}. A session that cannot be evaluated is not counted as one without a signal."
    else:
        out["status"], out["why"] = "none", out["rules"].get("summary") or "The entry rule did not fire on this session."
    return J(out)


# ----------------------------------------------------------------------------- comparison workspace
@router.get("/api/compare")
def compare(ids: str = Query(..., description="comma-separated run ids")):
    from strata.research.views import compare_runs
    entries = []
    for rid in [x for x in ids.split(",") if x][:8]:
        run, res = _run(rid)
        entries.append({"label": run.get("name") or rid, "run": run, "result": res})
    if len(entries) < 2:
        raise HTTPException(422, "pick at least two finished runs to compare")
    return J(compare_runs(entries))


class RerunReq(BaseModel):
    ids: list[str]
    start: date
    end: date


@router.post("/api/compare/rerun")
def compare_rerun(req: RerunReq):
    """Run the given runs again on one shared window so they can be compared like for like."""
    from strata.api import app as appmod
    from strata.api.launch import launch_research
    out = []
    for rid in req.ids[:8]:
        run = db.get_run(rid)
        if not run or run["kind"] not in ("study", "backtest"):
            continue
        cfg = dict(run.get("config") or {})
        meta = cfg.pop("_research", None) or {}
        cfg.pop("start", None)
        cfg.pop("end", None)
        res = None
        try:
            res = load_run_result(run)
        except FileNotFoundError:
            pass
        mode = (run.get("summary") or {}).get("mode") or meta.get("mode") or ("study" if run["kind"] == "study" else ((res or {}).get("meta") or {}).get("pricing") or "auto")
        spec = {"kind": "study" if run["kind"] == "study" else "options", "mode": mode, "study": cfg if run["kind"] == "study" else None,
                "backtest": {k: v for k, v in cfg.items() if k in ("strategies", "execution", "capital", "label", "benchmark", "allow_synthetic")} if run["kind"] != "study" else None,
                "requested": {"start": req.start.isoformat(), "end": req.end.isoformat(), "stated": True}, "approvals": {"reduced_window": True}, "provider": meta.get("provider"),
                "symbol": cfg.get("symbol") if run["kind"] == "study" else cfg["strategies"][0]["symbol"], "label": f"{run.get('name')} [{req.start} to {req.end}]"[:90], "interpretation": {"detail": "re-run on a shared window"}}
        r = launch_research(appmod.jobs, spec, spec["label"])
        out.append({"from": rid, "run_id": r["id"], "cached": r["cached"], "label": spec["label"]})
    return J({"runs": out})


@router.get("/api/runs/search")
def runs_search(q: str = "", kind: Optional[str] = None, limit: int = 80):
    """Finished and running runs, searchable by name, symbol or id."""
    words = [w for w in q.lower().split() if w]
    out = []
    for r in db.list_runs(kind=kind or "backtest,study,discovery,portfolio", limit=600):
        s = r.get("summary") or {}
        hay = f"{r.get('name', '')} {r['id']} {' '.join(s.get('symbols') or [])} {s.get('mode', '')} {r['kind']}".lower()
        if words and not all(w in hay for w in words):
            continue
        out.append({"id": r["id"], "kind": r["kind"], "name": r.get("name"), "status": r["status"], "created_at": r.get("created_at"), "mode": s.get("mode"), "outcome": s.get("outcome"),
                    "symbols": s.get("symbols"), "start": s.get("start"), "end": s.get("end"), "trades": s.get("trades"), "net_pnl": s.get("net_pnl"), "probability": s.get("probability"),
                    "qualifying": s.get("qualifying_sessions") or s.get("qualifying"), "synthetic": r.get("synthetic")})
        if len(out) >= limit:
            break
    return J(out)


# ----------------------------------------------------------------------------- providers and model settings
@router.post("/api/data/alpaca/test")
def alpaca_test():
    """Guided connection check: one small authenticated request to each Alpaca API."""
    from strata.data.providers.sources import Alpaca
    return J(Alpaca(MarketStore()).check())


@router.get("/api/data/datasets")
def datasets(symbol: Optional[str] = None):
    """Every stored dataset with its provider, observation type, resolution, version and coverage."""
    from strata.data.catalog import catalog
    out = []
    for c in catalog(MarketStore()):
        if symbol and c["symbol"] != symbol.upper():
            continue
        for d in c.get("datasets", []):
            out.append({k: v for k, v in d.items() if k not in ("daily_snapshots", "snapshot_gaps", "per_session")})
    return J(out)


@router.get("/api/settings/llm/models")
def llm_models(provider: str):
    from strata.ai.llm import LLMError, list_models
    try:
        return {"ok": True, "models": list_models(provider)[:400]}
    except LLMError as e:
        return {"ok": False, "error": str(e), "models": []}


@router.get("/api/settings/llm/usage")
def llm_usage(days: int = 30):
    return J(db.usage_summary(days))
