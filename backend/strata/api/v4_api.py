"""V4 routes: hypothesis plans, the research ledger, execution stress, the adversarial
review, sensitivity maps, the market-behaviour encyclopedia, the research canvas, the data
collector, provider verification and bulk imports."""
from __future__ import annotations

import json
import re
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from strata.api import db
from strata.api.research_api import J, _agent, _ai
from strata.api.results import run_dir
from strata.data.store import MarketStore

router = APIRouter()


def _jobs():
    from strata.api import app as appmod
    return appmod.jobs


def _file(rid: str, name: str) -> dict:
    run = db.get_run(rid)
    if not run:
        raise HTTPException(404, "run not found")
    p = run_dir(rid) / name
    if run["status"] != "done" or not p.exists():
        raise HTTPException(409, "this run has no result yet")
    return json.loads(p.read_text())


def _guard(fn):
    try:
        return J(fn())
    except LookupError as e:
        raise HTTPException(404, str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))


# ----------------------------------------------------------------------------- hypothesis lab
class PlanReq(BaseModel):
    question: str
    symbol: Optional[str] = None


@router.post("/api/hypothesis/plan")
def hypothesis_plan(req: PlanReq):
    """A bounded research plan for a broad question (nothing is run)."""
    from strata.ai import nl_rules as NL
    from strata.api.settings import get_setting
    from strata.research import hypothesis as HY
    store = MarketStore()
    known = sorted(store.stored_symbols())
    sym = (req.symbol or NL.find_symbol(req.question, [k for k in known if k not in NL.NOT_UNDERLYING]) or str(get_setting("data.default_symbol") or "SPY")).upper()
    plan = HY.build_plan(req.question, sym, store, known)
    if plan is None:
        raise HTTPException(422, "That question is not one of the hypothesis families Strata can plan yet. Try: reversals after large moves, overnight gaps, opening-range breakouts, the first N-minute candle, "
                                 "consecutive red or green days, large intraday moves, or \"what conditions are associated with a X% rally before a Y% decline\".")
    return _guard(lambda: {"plan": plan, "data": HY.describe_data(plan, store)})


class PlanRun(BaseModel):
    plan: dict
    sid: Optional[str] = None


@router.post("/api/hypothesis/run")
def hypothesis_run(req: PlanRun):
    from strata.research import hypothesis as HY
    errs = HY.validate_plan(req.plan)
    if errs:
        raise HTTPException(422, "; ".join(errs))
    rid = db.create_run("hypothesis", ("Hypothesis: " + req.plan.get("question", ""))[:90], {"question": req.plan.get("question"), "symbol": req.plan.get("symbol"), "configs": len(req.plan["configs"]), "session": req.sid})
    _jobs().submit(rid, "hypothesis", {"plan": req.plan, "scope": req.sid})
    return {"id": rid}


@router.get("/api/runs/{rid}/hypothesis")
def hypothesis_result(rid: str):
    return J(_file(rid, "hypothesis.json"))


@router.get("/api/runs/{rid}/hypothesis/observations")
def hypothesis_observations(rid: str, key: str, period: Optional[str] = None):
    from strata.research import hypothesis as HY
    res = _file(rid, "hypothesis.json")
    return _guard(lambda: HY.observations(res, key, MarketStore(), period))


class SelectReq(BaseModel):
    run_id: str
    key: str
    what: str = "study"


@router.post("/api/ai/sessions/{sid}/hypothesis/select")
def hypothesis_select(sid: str, req: SelectReq):
    return _ai(lambda: _agent().select_hypothesis(sid, req.run_id, req.key, req.what))


class CellSession(BaseModel):
    symbol: str
    preds: list[dict]
    outcome: dict
    label: str = ""
    prior_trials: int = 0
    start: Optional[str] = None
    end: Optional[str] = None


@router.post("/api/ai/sessions/from-cell")
def session_from_cell(req: CellSession):
    """A new research notebook that starts from one cell of a map or one row of an encyclopedia entry."""
    def go():
        from strata.research.features import OutcomeSpec, Pred, cond_text
        title = f"{req.symbol.upper()}: {cond_text([Pred.load(p) for p in req.preds], req.symbol.upper())}"[:70]
        OutcomeSpec.load(req.outcome)
        s = db.create_session(req.label[:70] or title)
        _agent().propose_cell(s["id"], req.symbol, req.preds, req.outcome, req.label, req.prior_trials, req.start, req.end)
        return {"id": s["id"]}
    return _ai(go)


@router.get("/api/ledger")
def ledger(scope: str, symbol: Optional[str] = None):
    from strata.research import ledger as LG
    syms = [symbol.upper()] if symbol else sorted({t["symbol"] for t in LG.list_trials(scope, limit=5000)})
    return J({"summary": LG.summary(scope), "holdouts": [h for s in syms for h in [LG.state(scope, s)["holdout"]] if h], "history": [h for s in syms for h in LG.state(scope, s)["history"]],
              "trials": LG.list_trials(scope, limit=400)})


# ----------------------------------------------------------------------------- stress / challenge
def _launch_analysis(kind: str, rid: str, sid: Optional[str], force: bool = False) -> dict:
    from strata.research import ledger as LG
    run = db.get_run(rid)
    if not run or run["status"] != "done":
        raise HTTPException(404, "run not found or not finished")
    if kind == "stress" and run["kind"] == "study":
        raise HTTPException(422, "A market study has no orders or fills, so there is no execution to stress.")
    if not force:
        for r in db.runs_targeting(rid, (kind,)):
            if r["status"] in ("done", "running", "queued"):
                return {"id": r["id"], "cached": r["status"] == "done"}
    trials = max(1, LG.count(sid)) if sid else 1
    name = ("Execution stress: " if kind == "stress" else "Challenge: ") + re.sub(r"^v[0-9]+ - ", "", run.get("name") or rid)
    nid = db.create_run(kind, name[:90], {"target": rid, "session": sid, "trials": trials})
    _jobs().submit(nid, kind, {"target": rid, "scope": sid, "trials": trials})
    return {"id": nid, "cached": False}


@router.post("/api/runs/{rid}/stress")
def run_stress(rid: str, sid: Optional[str] = Body(None, embed=True), force: bool = Body(False, embed=True)):
    return _launch_analysis("stress", rid, sid, force)


@router.post("/api/runs/{rid}/challenge")
def run_challenge(rid: str, sid: Optional[str] = Body(None, embed=True), force: bool = Body(False, embed=True)):
    return _launch_analysis("challenge", rid, sid, force)


@router.get("/api/runs/{rid}/execution")
def execution_result(rid: str):
    return J(_file(rid, "execution.json"))


@router.get("/api/runs/{rid}/challenge")
def challenge_result(rid: str):
    return J(_file(rid, "challenge.json"))


@router.get("/api/runs/{rid}/analyses")
def analyses_of(rid: str):
    """Stress tests, reviews and sensitivity maps that refer to a run."""
    return J([{"id": r["id"], "kind": r["kind"], "status": r["status"], "name": r["name"], "summary": r.get("summary"), "created_at": r["created_at"], "progress": r.get("progress"), "message": r.get("message")}
              for r in db.runs_targeting(rid)])


class AskReq(BaseModel):
    question: str
    other: Optional[str] = None  # a second run for comparisons


@router.post("/api/runs/{rid}/ask")
def ask_run(rid: str, req: AskReq):
    """A quantitative question about a finished run, answered by the deterministic reasoning tools."""
    from strata.api.results import load_run_result, load_trades
    from strata.research import reasoning as RQ
    run = db.get_run(rid)
    if not run or run["status"] != "done":
        raise HTTPException(404, "run not found or not finished")
    t = RQ.detect(req.question)
    if t is None:
        raise HTTPException(422, "No analysis matches that question. Try: after 2020, concentration, costs, volatility, assumptions, benchmark, too good.")
    res = load_run_result(run)
    trades = None if run["kind"] == "study" else load_trades(rid)
    store = MarketStore()

    def go():
        tool = t["tool"]
        if tool == "subperiod":
            return RQ.subperiod(res, trades, t.get("split") or "2020")
        if tool in ("threshold_difference", "options_vs_underlying"):
            o = db.get_run(req.other or "")
            if not o or o["status"] != "done":
                raise ValueError("this comparison needs a second finished run")
            r2 = load_run_result(o)
            if tool == "threshold_difference":
                return RQ.threshold_difference(res, r2, run.get("name") or "first", o.get("name") or "second")
            st, bt, bid = (res, r2, o["id"]) if run["kind"] == "study" else (r2, res, rid)
            return RQ.options_vs_underlying(st, bt, load_trades(bid))
        if tool == "cost_survival":
            return RQ.cost_survival(res, trades, _agent()._stress_of(rid) if trades is not None else None)
        if tool == "concentration":
            return RQ.concentration(res, trades)
        if tool == "volatility_explained":
            return RQ.volatility_explained(res, trades, store)
        if tool == "assumption_impact":
            return RQ.assumption_impact(res, trades, _agent()._stress_of(rid) if trades is not None else None, store)
        if tool == "suspicious":
            return RQ.suspicious(res, trades)
        return RQ.benchmarks(res, trades)
    return _guard(go)


# ----------------------------------------------------------------------------- sensitivity
@router.get("/api/sensitivity/presets")
def sensitivity_presets():
    from strata.research import sensitivity as SN
    return J({"underlying": [{"id": k, **{x: v[x] for x in ("title", "question", "x", "y", "outcome")}} for k, v in SN.PRESETS.items()], "options": [{"id": k, **v} for k, v in SN.OPTION_PRESETS.items()],
              "layers": SN.LAYERS, "option_layers": SN.OPTION_LAYERS})


class MapReq(BaseModel):
    symbol: str = "SPY"
    preset: Optional[str] = None
    spec: Optional[dict] = None
    start: Optional[str] = None
    end: Optional[str] = None
    preds: Optional[list[dict]] = None
    scope: Optional[str] = None


@router.post("/api/sensitivity/underlying")
def sensitivity_underlying(req: MapReq):
    from strata.research import ledger as LG
    from strata.research import sensitivity as SN
    spec = dict(req.spec or SN.PRESETS.get(req.preset or "") or {})
    if not spec:
        raise HTTPException(422, "unknown map")
    if req.preds is not None:
        spec["preds"] = req.preds

    def go():
        out = SN.underlying_map(req.symbol, spec, MarketStore(), req.start, req.end, cross=("QQQ", "IWM"))
        scope = req.scope or f"explorer:{req.symbol.upper()}"
        LG.record(scope, req.symbol, [{"kind": "sensitivity", "key": LG.config_key({"symbol": req.symbol.upper(), "preds": c["preds"], "outcome": c["outcome"]}), "label": f"{c['condition']} -> {c['outcome_text']}"[:200],
                                       "start": req.start, "end": req.end, "metric": "lift", "value": c.get("lift"), "n": c.get("n"), "p_value": c.get("p_value")} for c in out["cells"] if c["status"] == "ok"], family=out["title"])
        out["ledger"] = {"scope": scope, "configurations": LG.count(scope)}
        return out
    return _guard(go)


class CellReq(BaseModel):
    symbol: str
    preds: list[dict]
    outcome: dict
    start: Optional[str] = None
    end: Optional[str] = None


@router.post("/api/sensitivity/cell")
def sensitivity_cell(req: CellReq):
    from strata.research import sensitivity as SN
    return _guard(lambda: SN.cell_detail(req.symbol, req.preds, req.outcome, MarketStore(), req.start, req.end, cross=("QQQ", "IWM")))


class OptMapReq(BaseModel):
    run_id: str
    preset: Optional[str] = None
    spec: Optional[dict] = None
    sid: Optional[str] = None


@router.post("/api/sensitivity/options")
def sensitivity_options(req: OptMapReq):
    from strata.research import sensitivity as SN
    run = db.get_run(req.run_id)
    if not run or run["status"] != "done" or run["kind"] != "backtest":
        raise HTTPException(422, "An option sensitivity map starts from a finished option backtest.")
    spec = req.spec or SN.OPTION_PRESETS.get(req.preset or "")
    if not spec:
        raise HTTPException(422, "unknown map")
    rid = db.create_run("sensitivity", f"Sensitivity: {spec.get('title', 'map')} - {run.get('name', '')}"[:90], {"target": req.run_id, "session": req.sid, "preset": req.preset})
    _jobs().submit(rid, "sensitivity", {"target": req.run_id, "spec": spec, "scope": req.sid})
    return {"id": rid}


@router.get("/api/runs/{rid}/sensitivity")
def sensitivity_result(rid: str):
    return J(_file(rid, "sensitivity.json"))


@router.get("/api/features")
def features(symbol: str = "SPY"):
    from strata.research.features import catalog
    return J(catalog(symbol.upper(), tuple(x for x in ("QQQ", "IWM", "SPY") if x != symbol.upper())))


# ----------------------------------------------------------------------------- encyclopedia
@router.get("/api/encyclopedia")
def encyclopedia():
    from strata.research import encyclopedia as EN
    store = MarketStore()
    syms = sorted(s for s in store.stored_symbols() if not store.is_synthetic(s) and store.has_daily(s))
    return J({"entries": EN.listing(), "symbols": syms})


@router.post("/api/encyclopedia/{entry_id}")
def encyclopedia_run(entry_id: str, params: dict = Body(default_factory=dict)):
    from strata.research import encyclopedia as EN
    return _guard(lambda: EN.run_entry(entry_id, params, MarketStore()))


# ----------------------------------------------------------------------------- research canvas
def canvas_graph(sid: str) -> dict:
    """The session as a graph: every node is a stored message / version / run."""
    msgs = db.list_messages(sid)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    last_q = None
    by_version: dict[int, str] = {}
    by_run: dict[str, str] = {}
    by_source: dict[int, str] = {}

    def add(nid, typ, title, parent=None, **kw):
        nodes[nid] = {"id": nid, "type": typ, "title": (title or "")[:110], **kw}
        if parent and parent in nodes:
            edges.append({"id": f"{parent}>{nid}", "source": parent, "target": nid})
        return nid

    def status_of(runs):
        st = [(db.get_run(r["run_id"]) or {}).get("status") for r in runs or []]
        return "error" if any(x == "error" for x in st) else "running" if any(x in ("running", "queued") for x in st) else "done" if st else "idle"

    for m in msgs:
        c = m["content"] if isinstance(m["content"], dict) else {}
        k = m["kind"]
        if m["role"] == "user":
            last_q = add(f"q{m['id']}", "question", c.get("text", ""), None, message_id=m["id"])
            continue
        if k == "hypothesis":
            pl = c.get("plan") or {}
            nid = add(f"plan{m['id']}", "plan", pl.get("hypothesis") or c.get("question"), last_q, message_id=m["id"], run_id=c["runs"][0]["run_id"], status=status_of(c.get("runs")),
                      stats=[{"label": "Configurations", "value": str(len(pl.get("configs") or []))}, {"label": "Minimum sample", "value": str(pl.get("min_sample"))}, {"label": "Bound", "value": str(pl.get("max_configs"))}],
                      assumptions=(pl.get("notes") or [])[:4], subtitle="Research plan")
            by_source[m["id"]] = nid
            by_run[c["runs"][0]["run_id"]] = nid
        elif k == "plan":
            d = c.get("draft") or {}
            parent = by_version.get(c.get("parent_id")) if c.get("parent_id") else None
            src = (d.get("carried") or {}).get("from_plan")
            if not parent and src:
                parent = f"cand:{src}:{(d.get('carried') or {}).get('ledger_key')}" if f"cand:{src}:{(d.get('carried') or {}).get('ledger_key')}" in nodes else by_run.get(src)
            typ = "study" if c.get("kind") == "study" else "backtest"
            nid = add(f"v{c['version_id']}", typ, c.get("title") or d.get("title") or "Strategy", parent or last_q, message_id=m["id"], version_id=c["version_id"], status=status_of(c.get("runs")) if c.get("runs") else c.get("state"),
                      run_id=(c.get("runs") or [{}])[-1].get("run_id"), subtitle="Market study" if typ == "study" else "Option backtest",
                      assumptions=[f"{a.get('label')}: {a.get('value')}" for a in (c.get("assumptions") or [])[:5]], items=[{"label": i.get("label"), "value": i.get("value"), "status": i.get("status")} for i in (c.get("items") or [])[:8]])
            by_version[c["version_id"]] = nid
            for r in c.get("runs") or []:
                by_run[r["run_id"]] = nid
            by_source[m["id"]] = nid
        elif k in ("sweep", "investigation", "discovery"):
            parent = by_version.get(c.get("version_id")) or last_q
            if k == "investigation" and (c.get("plan") or {}).get("from_plan_key"):
                cand = next((n for n in nodes if n.startswith("cand:") and n.endswith(":" + c["plan"]["from_plan_key"])), None)
                parent = cand or parent
            stage = (c.get("plan") or {}).get("stage")
            typ = "structures" if stage == "options" else "sweep" if k == "sweep" else "discovery" if k == "discovery" else "investigation"
            title = {"structures": "Option structures", "sweep": f"Sweep: {c.get('what') or c.get('field')}", "discovery": c.get("hypothesis") or "Exploration", "investigation": (c.get("plan") or {}).get("question") or "Investigation"}[typ]
            if typ == "structures":  # option structures launched from a drafted strategy hang under that draft
                drafts = [n for n, v in nodes.items() if v["type"] == "strategy_draft"]
                if drafts and not (c.get("plan") or {}).get("from_plan_key"):
                    parent = drafts[-1]
            nid = add(f"g{m['id']}", typ, title, parent if parent in nodes else last_q, message_id=m["id"], status=status_of(c.get("runs")), group=True, subtitle={"structures": "Options profitability (stage two)", "sweep": "Parameter sweep", "discovery": "Exploratory grid",
                                                                                                                                                          "investigation": "Related studies"}[typ],
                      stats=[{"label": "Variants", "value": str(len(c.get("runs") or []))}], version_id=c.get("version_id"))
            by_source[m["id"]] = nid
            if k != "discovery":
                for r in (c.get("runs") or [])[:14]:
                    child = add(f"r{r['run_id']}", "backtest" if (r.get("draft") or {}).get("kind") == "options" else "study", r.get("label"), nid, run_id=r["run_id"], status=(db.get_run(r["run_id"]) or {}).get("status"), parent_group=nid,
                                subtitle="variant")
                    by_run[r["run_id"]] = child
            else:
                by_run[c["runs"][0]["run_id"]] = nid
        elif k == "graph":
            g = c.get("graph") or {}
            src_node = next((by_run[r] for r in c.get("derived_from") or [] if r in by_run), None)
            nid = add(f"gr{m['id']}", "graph", c.get("question") or g.get("question"), src_node or last_q, message_id=m["id"], run_id=(c.get("runs") or [{}])[0].get("run_id"),
                      status={"awaiting": "idle", "needs_data": "idle", "blocked": "error", "cancelled": "idle"}.get(c.get("state"), status_of(c.get("runs"))), subtitle="Research graph",
                      stats=[{"label": "Steps", "value": str(len(g.get("nodes") or []) or len(g.get("batch") or []))}, {"label": "Conditions", "value": str(len((g.get("core") or {}).get("clauses") or []))}],
                      assumptions=(g.get("assumptions") or [])[:4])
            if src_node and last_q in nodes:
                edges.append({"id": f"{last_q}>{nid}", "source": last_q, "target": nid})
            by_source[m["id"]] = nid
            for r in c.get("runs") or []:
                if r.get("mode") == "graph":
                    by_run[r["run_id"]] = nid
        elif k == "strategy_plan":
            f = c.get("finding") or {}
            parent = f"find{f.get('id')}" if f"find{f.get('id')}" in nodes else by_run.get(f.get("run_id")) or last_q
            last_plan = add(f"sp{m['id']}", "strategy_draft", c.get("label") or "Option structures", parent, message_id=m["id"], status="done" if c.get("launched") else "idle", subtitle="Drafted strategies (not priced)",
                            stats=[{"label": "Structures", "value": str(len(c.get("variants") or []))}, {"label": "Built on", "value": f.get("evidence") or ""}], summary=(c.get("text") or "")[:240])
            nodes[last_plan]["plan_message"] = m["id"]
        elif k == "forward":
            ex = c.get("experiment") or {}
            parent = by_run.get(ex.get("source_run_id")) or last_q
            add(f"fw{ex.get('id')}", "forward", ex.get("name") or "Forward validation", parent, message_id=m["id"], status="done", subtitle=f"Frozen v{ex.get('version')} - forward validation",
                stats=[{"label": "Active from", "value": ex.get("activation_date") or ""}], experiment_id=ex.get("id"))
        elif k in ("stress", "challenge"):
            parent = by_run.get(c.get("target"))
            nid = add(f"a{m['id']}", "stress" if k == "stress" else "challenge", "Execution stress test" if k == "stress" else "Adversarial review", parent or last_q, message_id=m["id"], run_id=c["runs"][0]["run_id"], status=status_of(c.get("runs")),
                      subtitle="Robustness")
            by_source[m["id"]] = nid
        elif k == "result":
            view = c.get("view")
            src = by_source.get(c.get("source_message"))
            facts = [{"label": f.get("label"), "value": f.get("shown")} for f in (c.get("facts") or [])[:4]]
            if view in ("single", "compare") and src in nodes:
                nodes[src].update(stats=facts or nodes[src].get("stats"), result_message=m["id"], status="done", mode=c.get("mode"), summary=(c.get("text") or "")[:240])
                if view == "compare":
                    cmp_id = add(f"c{m['id']}", "comparison", "Comparison with the previous version", src, message_id=m["id"], status="done", subtitle="Comparison", summary=(c.get("compare_text") or "")[:240])
                    prev = by_version.get((db.get_message(c.get("source_message")) or {}).get("content", {}).get("parent_id"))
                    if prev:
                        edges.append({"id": f"{prev}>{cmp_id}", "source": prev, "target": cmp_id})
            elif view == "hypothesis" and src in nodes:
                s_ = c.get("summary") or {}
                nodes[src].update(status="done", result_message=m["id"], summary=(c.get("text") or "")[:240],
                                  stats=[{"label": "Tested", "value": str(s_.get("tested"))}, {"label": "Candidates", "value": str(s_.get("candidates"))}, {"label": "Confirmed", "value": str(s_.get("confirmed"))}])
                grp = add(f"all:{c['run_id']}", "tested", f"All {s_.get('tested')} tested configurations", src, run_id=c["run_id"], status="done", group=True, subtitle="Including the ones that showed nothing",
                          stats=[{"label": x["step"], "value": str(x["n"])} for x in (s_.get("funnel") or [])[2:6]])
                for cd in c.get("candidates") or []:
                    add(f"cand:{c['run_id']}:{cd['key']}", "candidate", cd["label"], grp, run_id=c["run_id"], key=cd["key"], status="done", subtitle=cd.get("status_label"), parent_group=grp,
                        stats=[{"label": "Outcome", "value": cd.get("outcome_text")}, {"label": "Evidence", "value": (cd.get("grade") or {}).get("label")}])
            elif view in ("graph", "graph_batch") and src in nodes:
                nodes[src].update(status="done", result_message=m["id"], summary=(c.get("text") or "")[:240],
                                  stats=[{"label": "Analyses", "value": str(len(c.get("analyses") or c.get("rows") or []))}, {"label": "New trials", "value": str(c.get("trials", ""))}])
                for f in c.get("findings") or []:
                    add(f"find{f['id']}", "finding", f["statement"], src, message_id=m["id"], run_id=c.get("run_id"), status="done", subtitle=f"Finding - {f['evidence']}", finding_id=f["id"],
                        stats=[{"label": "Evidence", "value": f["evidence"]}], summary="; ".join(f.get("why") or [])[:240])
            elif view in ("investigation", "sweep", "discovery") and src in nodes:
                nodes[src].update(status="done", result_message=m["id"], summary=(c.get("text") or "")[:240])
            elif view in ("challenge", "execution") and src in nodes:
                cnt = c.get("counts") or {}
                s_ = c.get("summary") or {}
                nodes[src].update(status="done", result_message=m["id"], summary=(c.get("text") or "")[:240],
                                  stats=([{"label": "Serious", "value": str(cnt.get("high", 0))}, {"label": "Weaknesses", "value": str(cnt.get("medium", 0))}, {"label": "Clear", "value": str(cnt.get("ok", 0))}] if view == "challenge" else
                                         [{"label": "Signal only", "value": _money(s_.get("signal_net_pnl"))}, {"label": "As simulated", "value": _money(s_.get("simulated_net_pnl"))}, {"label": "Adverse", "value": _money(s_.get("adverse_net_pnl"))}]))
            elif view in ("robustness", "analysis", "breakdown", "timing", "skipped"):
                parent = by_run.get(c.get("run_id")) or by_version.get(c.get("version_id")) or last_q
                title = {"robustness": "Robustness checks", "breakdown": "Breakdown", "timing": "Timing", "skipped": "Skipped signals"}.get(view) or (c.get("analysis") or {}).get("title") or "Analysis"
                add(f"an{m['id']}", "validation" if view == "robustness" else "analysis", title, parent, message_id=m["id"], status="done", subtitle="Out-of-sample and robustness" if view == "robustness" else "Reasoning",
                    summary=(c.get("text") or "")[:240], stats=[{"label": f["label"], "value": f["value"]} for f in ((c.get("analysis") or {}).get("facts") or [])[:3]], run_id=c.get("run_id"))
            elif view == "failed" and src in nodes:
                nodes[src].update(status="error", summary=(c.get("text") or "")[:240])
        elif k == "export":
            parent = by_version.get(c.get("version_id")) or last_q
            add(f"rep{m['id']}", "report", "Research report", parent, message_id=m["id"], status="done", subtitle="Final report", run_id=c.get("run_id"))
    # drop questions that led to nothing drawable (plain answers) unless they are the only content
    used = {e["source"] for e in edges} | {e["target"] for e in edges}
    keep = {n: v for n, v in nodes.items() if v["type"] != "question" or n in used}
    # default layout: depth from the roots, siblings stacked
    kids: dict[str, list[str]] = {}
    indeg = {n: 0 for n in keep}
    for e in edges:
        if e["source"] in keep and e["target"] in keep and e["target"] not in kids.get(e["source"], []):
            kids.setdefault(e["source"], []).append(e["target"])
            indeg[e["target"]] += 1
    y = [0.0]
    pos: dict[str, dict] = {}

    def place(n, depth):
        if n in pos:
            return pos[n]["y"]
        ch = kids.get(n, [])
        if not ch:
            pos[n] = {"x": depth * 330.0, "y": y[0]}
            y[0] += 150.0
            return pos[n]["y"]
        ys = [place(c_, depth + 1) for c_ in ch]
        pos[n] = {"x": depth * 330.0, "y": sum(ys) / len(ys)}
        return pos[n]["y"]
    for n in [n for n in keep if indeg[n] == 0]:
        place(n, 0)
        y[0] += 40.0
    for n, v in keep.items():
        v["default_position"] = pos.get(n, {"x": 0.0, "y": 0.0})
    with db.connect() as con:
        row = con.execute("SELECT layout FROM canvases WHERE session_id=?", (sid,)).fetchone()
    layout = json.loads(row["layout"]) if row else {"positions": {}, "collapsed": [], "viewport": None}
    return {"nodes": list(keep.values()), "edges": [e for e in edges if e["source"] in keep and e["target"] in keep], "layout": layout}


def _money(x):
    return "n/a" if x is None else f"{x:+,.0f}"


@router.get("/api/ai/sessions/{sid}/canvas")
def canvas_get(sid: str):
    if db.get_session(sid) is None:
        raise HTTPException(404, "session not found")
    return J(canvas_graph(sid))


@router.put("/api/ai/sessions/{sid}/canvas")
def canvas_put(sid: str, layout: dict = Body(...)):
    if db.get_session(sid) is None:
        raise HTTPException(404, "session not found")
    clean = {"positions": {str(k): {"x": float(v["x"]), "y": float(v["y"])} for k, v in (layout.get("positions") or {}).items() if isinstance(v, dict) and "x" in v and "y" in v},
             "collapsed": [str(x) for x in (layout.get("collapsed") or [])][:500], "viewport": layout.get("viewport") if isinstance(layout.get("viewport"), dict) else None}
    with db.connect() as con:
        con.execute("INSERT INTO canvases (session_id, layout, updated_at) VALUES (?,?,?) ON CONFLICT(session_id) DO UPDATE SET layout=excluded.layout, updated_at=excluded.updated_at", (sid, json.dumps(clean), db.now()))
    return {"ok": True}


class BranchReq(BaseModel):
    text: str
    version_id: Optional[int] = None


@router.post("/api/ai/sessions/{sid}/branch")
def canvas_branch(sid: str, req: BranchReq):
    """Continue the research from an earlier node: that version becomes current, then the message is interpreted against it."""
    def go():
        ag = _agent()
        out = []
        if req.version_id is not None:
            ag.restore(sid, req.version_id)
        out += ag.post(sid, req.text)
        return out
    return _ai(go)


class AnalysisReq(BaseModel):
    kind: str
    run_id: str


@router.post("/api/ai/sessions/{sid}/analysis")
def session_analysis(sid: str, req: AnalysisReq):
    """Start an execution stress test or an adversarial review of a run, inside a research session."""
    if req.kind not in ("stress", "challenge"):
        raise HTTPException(422, "kind must be stress or challenge")
    return _ai(lambda: _agent()._analysis_job(sid, None, req.kind, req.run_id))


# ----------------------------------------------------------------------------- collector
@router.get("/api/collector")
def collector_state():
    from strata.collector import service as CS
    return J(CS.state(MarketStore()))


@router.put("/api/collector/config")
def collector_config(cfg: dict = Body(...)):
    from strata.collector import service as CS
    return _guard(lambda: CS.save_config(cfg))


@router.post("/api/collector/run")
def collector_run():
    from strata.collector import service as CS
    if CS.running():
        raise HTTPException(409, "a collection cycle is already running")
    rid = db.create_run("collector", "Data collection cycle", {"trigger": "manual"})
    _jobs().submit(rid, "collector", {"trigger": "manual"})
    return {"id": rid}


@router.get("/api/collector/health")
def collector_health():
    """One small read-only request to each source the collector uses, and when the unattended daemon last reported."""
    import time as _t

    from strata.api.settings import get_settings
    from strata.collector import service as CS
    from strata.data.providers import sources as S

    def go():
        cfg = CS.load_config()
        store = MarketStore()
        out = {"sources": [CS.health(S.get_source("yahoo", store))], "daemon": {"alive": False, "seconds_ago": None}, "checked_at": db.now()}
        if cfg["options"]["enabled"]:
            src = next((x for x in S.option_bar_sources(store) if x.name == cfg["options"]["provider"]), None)
            if src is not None:
                ok, why = src.available()
                out["sources"].append(CS.health(src) if ok else {"source": src.name, "ok": False, "detail": why})
        hb = get_settings().data_dir / "collector.heartbeat"
        if hb.exists():
            age = _t.time() - hb.stat().st_mtime
            out["daemon"] = {"alive": age < 300, "seconds_ago": round(age)}
        return out
    return _guard(go)


@router.post("/api/collector/cleanup")
def collector_cleanup(dry_run: bool = Body(True, embed=True)):
    from strata.collector import service as CS
    return _guard(lambda: CS.cleanup(MarketStore(), dry_run=dry_run))


# ----------------------------------------------------------------------------- providers
@router.get("/api/data/providers/catalog")
def provider_catalog():
    from strata.data.providers import bench as PB
    return J(PB.catalog(MarketStore()))


@router.post("/api/data/providers/benchmark")
def provider_benchmark(source: str = Body(..., embed=True)):
    from strata.data.providers import bench as PB
    if source not in PB.BENCHABLE:
        raise HTTPException(422, f"unknown source {source}")
    rid = db.create_run("provider_benchmark", f"Provider check: {source}", {"source": source})
    _jobs().submit(rid, "provider_benchmark", {"source": source})
    return {"id": rid}


# ----------------------------------------------------------------------------- bulk import
@router.post("/api/data/import/inspect")
def import_inspect(path: str = Body(..., embed=True)):
    from strata.data import bulk as BK
    return _guard(lambda: BK.inspect(path))


class BulkReq(BaseModel):
    path: str
    symbol: Optional[str] = None
    format: Optional[str] = None
    tz: str = "America/New_York"
    provider: Optional[str] = None


@router.post("/api/data/import/bulk")
def import_bulk(req: BulkReq):
    from strata.data import bulk as BK
    info = BK.inspect(req.path)
    if not info["files"]:
        raise HTTPException(422, "no importable files found at that path")
    rid = db.create_run("bulk_import", f"Import {req.symbol or info.get('symbol') or ''} {info['files'][0]['format'] if info['files'] else ''}"[:90], {"path": req.path, "symbol": req.symbol, "format": req.format})
    _jobs().submit(rid, "bulk_import", req.model_dump())
    return {"id": rid, "inspect": info}


# ----------------------------------------------------------------------------- benchmarks
@router.get("/api/bench")
def bench_history(limit: int = 30):
    from strata import bench as BN
    return J(BN.history(limit))
