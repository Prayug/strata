"""V5 routes: research graphs, the pattern explorer, the experiment registry, research
memory and the findings knowledge base, forward validation, ML research, the provider
marketplace and the strategy IDE helpers."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from strata.api import db
from strata.api.research_api import J, _agent, _ai
from strata.api.v4_api import _file, _guard, _jobs
from strata.data.store import MarketStore

router = APIRouter()


def _default_symbol() -> str:
    from strata.api.settings import get_setting
    return str(get_setting("data.default_symbol") or "SPY").upper()


# ----------------------------------------------------------------------------- research graphs
@router.get("/api/graph/primitives")
def graph_primitives():
    """The primitive catalog, the feature catalog and the rule-language reference: everything
    a graph can be made of (the graph editor and the planner's instructions use the same list)."""
    from strata.engine import expr as X
    from strata.research import graph as G
    from strata.research.features import catalog
    return J({"primitives": G.primitives(), "categories": G.CATEGORIES, "features": catalog(_default_symbol()),
              "expressions": [{"name": k, **v} for k, v in X.REGISTRY.items() if v["category"] != "Position"], "limits": {"nodes": G.MAX_NODES, "sweep_values": G.MAX_SWEEP}})


class GraphPlanReq(BaseModel):
    question: str
    symbol: Optional[str] = None


@router.post("/api/graph/plan")
def graph_plan(req: GraphPlanReq):
    """A question -> a validated graph and what it needs. Nothing is run."""
    from strata.ai import nl_graph as NG
    from strata.research import graph as G
    store = MarketStore()
    g = NG.plan(req.question, sorted(store.stored_symbols()), (req.symbol or _default_symbol()).upper(), store)
    if g is None:
        raise HTTPException(422, "That could not be read as a market question with a condition and an outcome. Example: \"Is SPY more likely to close higher when it gaps down more than 1% and VIX is above 25?\"")
    return J({"graph": g, "describe": G.describe(g, store) if not g.get("errors") else {"ok": False, "errors": g["errors"]}})


class GraphReq(BaseModel):
    graph: dict
    session_id: Optional[str] = None


@router.post("/api/graph/validate")
def graph_validate(req: GraphReq):
    from strata.research import graph as G
    return J(G.describe(req.graph, MarketStore()))


@router.post("/api/graph/run")
def graph_run(req: GraphReq):
    """Run a graph (hand-edited or planned). With a session it is recorded in that notebook's
    ledger; without one nothing is recorded and no final test period is reserved in a ledger."""
    from strata.research import graph as G
    chk = G.validate(req.graph)
    if chk["errors"]:
        raise HTTPException(422, "; ".join(chk["errors"][:6]))
    if req.session_id:
        return _ai(lambda: {"messages": _agent()._graph(req.session_id, req.graph, "graph editor")})
    rid = db.create_run("graph", ("Graph: " + (req.graph.get("question") or "custom graph"))[:90], {"question": req.graph.get("question"), "nodes": len(req.graph["nodes"])})
    _jobs().submit(rid, "graph", {"graph": req.graph, "scope": None})
    return {"run_id": rid}


@router.get("/api/runs/{rid}/graph")
def graph_result(rid: str):
    return J(_file(rid, "graph.json"))


class ChoiceReq(BaseModel):
    message_id: int
    choice: str


@router.post("/api/ai/sessions/{sid}/graph/decide")
def graph_decide(sid: str, req: ChoiceReq):
    return _ai(lambda: {"messages": _agent().decide_graph(sid, req.message_id, req.choice)})


class FindingReq(BaseModel):
    finding_id: Optional[int] = None
    anyway: bool = False


@router.post("/api/ai/sessions/{sid}/strategy-from-finding")
def strategy_from_finding(sid: str, req: FindingReq):
    return _ai(lambda: {"messages": _agent()._strategy_from_finding(sid, req.anyway, req.finding_id)})


@router.post("/api/ai/sessions/{sid}/backtest-plan")
def backtest_plan(sid: str):
    return _ai(lambda: {"messages": _agent()._backtest_plan(sid)})


# ----------------------------------------------------------------------------- pattern explorer
@router.get("/api/patterns/events")
def pattern_events():
    from strata.research import patterns as PT
    store = MarketStore()
    return J({"events": PT.catalog(), "symbols": sorted(s for s in store.stored_symbols() if not store.is_synthetic(s)), "default_symbol": _default_symbol(), "horizons": PT.HORIZONS})


class PatternReq(BaseModel):
    symbol: str
    event: dict
    start: Optional[str] = None
    end: Optional[str] = None
    horizon: int = 10
    main_horizon: Optional[int] = None
    target: Optional[float] = None
    stop: Optional[float] = None
    direction: str = "up"
    overlays: list[dict] = []


@router.post("/api/patterns/explore")
def pattern_explore(req: PatternReq):
    from strata.research import ledger as LG
    from strata.research import patterns as PT

    def go():
        out = PT.explore(req.symbol, req.event, MarketStore(), req.start, req.end, req.horizon, req.target, req.stop, req.direction, req.overlays[:4], main_horizon=req.main_horizon)
        if out["n"]:  # looking at a pattern is a trial like any other: it is recorded so the registry can say the dates were seen
            sig = {"symbol": out["symbol"], "features": sorted({p.get("feature") or p.get("expr", "") for p in out["cell"]["preds"]}), "outcome": "|".join(str(out["cell"]["outcome"].get(k)) for k in ("entry", "horizon", "direction", "measure"))}
            LG.record(f"patterns:{out['symbol']}", out["symbol"] + ("" if out["event"]["anchor"] in ("open", "close") else "@intraday"),
                      [{"kind": "study", "key": LG.config_key(out["cell"]), "label": out["event"]["text"], "start": out["window"][0], "end": out["window"][1], "metric": "mean_forward_return", "n": out["n"], "signature": sig}], family="pattern explorer")
        return out
    return _guard(go)


# ----------------------------------------------------------------------------- experiment registry
@router.get("/api/registry")
def registry_overview():
    from strata.research import registry as RG
    return J(RG.overview())


@router.get("/api/registry/audit")
def registry_audit(scope: Optional[str] = None, symbol: Optional[str] = None, phase: Optional[str] = None, q: Optional[str] = None, limit: int = 300, offset: int = 0):
    from strata.research import registry as RG
    return J(RG.audit(scope, symbol, phase, q, min(limit, 1000), offset))


class RegistryCheckReq(BaseModel):
    symbol: str
    scope: Optional[str] = None
    key: Optional[str] = None
    signature: Optional[dict] = None
    config: Optional[dict] = None


@router.post("/api/registry/check")
def registry_check(req: RegistryCheckReq):
    from strata.research import ledger as LG
    from strata.research import registry as RG
    key = req.key or (LG.config_key(req.config) if req.config else None)
    return J(RG.check(req.scope, req.symbol, key, req.signature))


# ----------------------------------------------------------------------------- memory and the findings knowledge base
@router.get("/api/ai/sessions/{sid}/memory")
def session_memory(sid: str):
    from strata.research import memory as MEM
    if db.get_session(sid) is None:
        raise HTTPException(404, "session not found")
    mem = MEM.notebook(sid)
    return J({**mem, "text": MEM.summary_text(mem)})


@router.get("/api/findings")
def findings(q: Optional[str] = None, symbol: Optional[str] = None, session: Optional[str] = None, evidence: Optional[str] = None, limit: int = 200):
    """Search every notebook's recorded findings (the knowledge base)."""
    from strata.research import memory as MEM
    rows = MEM.list_findings(session, symbol, q, evidence.split(",") if evidence else None, min(limit, 500))
    return J({"findings": [{k: r.get(k) for k in ("id", "session_id", "notebook", "run_id", "symbol", "kind", "statement", "evidence", "rank", "why", "numbers", "tags", "period_start", "period_end", "created_at", "fingerprint")}
                           | {"convertible": bool((r.get("config") or {}).get("cell")), "reproducible": bool((r.get("config") or {}).get("graph") or (r.get("config") or {}).get("study") or (r.get("config") or {}).get("backtest"))} for r in rows],
              "grades": MEM.GRADES})


@router.get("/api/findings/{fid}")
def finding(fid: int):
    from strata.research import memory as MEM
    f = MEM.get_finding(fid)
    if f is None:
        raise HTTPException(404, "finding not found")
    return J(f)


# ----------------------------------------------------------------------------- execution quality
@router.get("/api/runs/{rid}/execution-quality")
def execution_quality(rid: str):
    """How much observed evidence stands behind a backtest's fills (not its profitability)."""
    from strata.api.results import load_run_result, load_trades
    from strata.research import execution as EX
    run = db.get_run(rid)
    if run is None or run["status"] != "done":
        raise HTTPException(404, "run not found")
    if run["kind"] == "study":
        raise HTTPException(422, "A market study has no fills, so there is no execution to assess.")

    def go():
        res = load_run_result(run)
        ctx = EX.context_for(run, res, MarketStore())
        return EX.quality(load_trades(rid), res.get("meta") or {}, res.get("diagnostics") or {}, ctx["config"], ctx["eod"])
    return _guard(go)


# ----------------------------------------------------------------------------- ML research (optional dependency)
@router.get("/api/ml/meta")
def ml_meta():
    from strata.research import ml as ML
    from strata.research.features import catalog
    ok, info = ML.available()
    store = MarketStore()
    return J({"available": ok, "detail": info if not ok else f"scikit-learn {info}", "models": [{"name": k, **v} for k, v in ML.MODELS.items()], "default_features": ML.DEFAULT_FEATURES,
              "features": catalog(_default_symbol()), "symbols": sorted(s for s in store.stored_symbols() if not store.is_synthetic(s)), "default_symbol": _default_symbol(), "min_rows": ML.MIN_ROWS})


class MlReq(BaseModel):
    spec: dict
    session_id: Optional[str] = None


@router.post("/api/ml/run")
def ml_run(req: MlReq):
    from strata.research import ml as ML
    from strata.research.features import OutcomeSpec
    ok, info = ML.available()
    if not ok:
        raise HTTPException(409, info)
    if not req.spec.get("symbol") or not isinstance(req.spec.get("outcome"), dict):
        raise HTTPException(422, "an ML study needs a symbol and an outcome")
    oc = OutcomeSpec.load(req.spec["outcome"])
    sym = str(req.spec["symbol"]).upper()
    rid = db.create_run("ml", f"ML: {sym} - {oc.text()}"[:90], {"symbol": sym, "outcome": oc.dump(), "models": req.spec.get("models"), "session": req.session_id})
    _jobs().submit(rid, "ml", {"spec": req.spec, "scope": req.session_id or f"ml:{sym}"})
    return {"run_id": rid}


@router.get("/api/runs/{rid}/ml")
def ml_result(rid: str):
    return J(_file(rid, "ml.json"))


@router.post("/api/runs/{rid}/ml/final-test")
def ml_final_test(rid: str):
    """Step two of an ML study: score the configuration that run `rid` froze on its final test set, once, and record that it was looked at.

    The new run is pinned to what was frozen (the selected model, its features and the exact rows of the split): if any of them
    no longer matches, it stops before a test score exists. The test set of a study whose leakage check failed is not scored."""
    from strata.research import ml as ML
    ok, info = ML.available()
    if not ok:
        raise HTTPException(409, info)
    res = _file(rid, "ml.json")
    spec, frozen = res.get("spec"), res.get("frozen")
    if not isinstance(spec, dict) or not isinstance(frozen, dict):
        raise HTTPException(409, "this study was run before the two-step procedure existed: run it again, then score its final test")
    if res.get("final_test"):
        raise HTTPException(409, "this run already holds the final test score")
    if res.get("verdict") == "leak_suspected":
        raise HTTPException(409, "a leakage check failed: a score on the final test set would mean nothing, and looking at it would spend it")
    sym = str(spec["symbol"]).upper()
    new = db.create_run("ml", f"ML final test: {sym} - {res.get('outcome_text', '')}"[:90], {"symbol": sym, "outcome": spec.get("outcome"), "models": spec.get("models"), "session": None, "validation_run": rid})
    pinned = {**spec, "end": frozen["test"][1], "final_test": True, "validation_run": rid, "expect_frozen": {k: frozen[k] for k in ("config_key", "train", "test")}}
    _jobs().submit(new, "ml", {"spec": pinned, "scope": f"ml:{sym}"})
    return {"run_id": new}


class SignalReq(BaseModel):
    name: Optional[str] = None


@router.post("/api/runs/{rid}/ml/signal")
def ml_save_signal(rid: str, req: SignalReq):
    """Store the model's walk-forward signal as a named list that any rule can read with signal("name")."""
    from strata.research import ml as ML
    return _guard(lambda: ML.save_signal(_file(rid, "ml.json"), rid, req.name))


@router.get("/api/signals")
def signals_list():
    from strata.research import signals as SG
    return J({"signals": SG.listing()})


@router.delete("/api/signals/{name}")
def signals_delete(name: str):
    from strata.research import signals as SG
    SG.delete(name)
    return {"ok": True}


# ----------------------------------------------------------------------------- forward validation (paper only)
@router.get("/api/forward")
def forward_list():
    from strata.research import forward as FW
    return J({"experiments": FW.listing(), "min_forward": FW.MIN_FORWARD})


class ForwardReq(BaseModel):
    run_id: Optional[str] = None
    name: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/api/forward")
def forward_create(req: ForwardReq):
    """Freeze a finished backtest (or a research graph) as a forward-validation experiment."""
    from strata.research import forward as FW

    def go():
        run = db.get_run(req.run_id or "")
        if run is None:
            raise LookupError("run not found")
        if run["kind"] == "graph":
            res = _file(run["id"], "graph.json")
            return FW.create_from_graph(res["graph"], res, req.session_id or (run.get("config") or {}).get("session"), run["id"], req.name)
        return FW.create_from_run(run, req.session_id or (run.get("config") or {}).get("session"), req.name)
    return _guard(go)


@router.get("/api/forward/{eid}")
def forward_get(eid: str):
    from strata.research import forward as FW
    ex = FW.get(eid)
    if ex is None:
        raise HTTPException(404, "experiment not found")
    with db.connect() as con:
        hist = [{"id": r["id"], "evaluated_at": r["evaluated_at"], "through_date": r["through_date"]} for r in con.execute("SELECT id, evaluated_at, through_date FROM forward_evaluations WHERE experiment_id=? ORDER BY id DESC LIMIT 200", (eid,))]
        kids = [dict(r) for r in con.execute("SELECT id, version, activation_date, first_session, status FROM forward_experiments WHERE parent_id=? ORDER BY version", (eid,))]
    return J({**ex, "history": hist, "newer_versions": kids, "ledger": FW.verify(eid), "pending_sessions": FW.pending(ex, MarketStore())})


@router.get("/api/forward/{eid}/ledger")
def forward_ledger(eid: str, limit: int = 500):
    """The append-only record: what was frozen, then every sealed session's entries in the order they were written."""
    from strata.research import forward as FW
    if FW.get(eid) is None:
        raise HTTPException(404, "experiment not found")
    return J({"entries": FW.ledger(eid, limit=max(1, min(int(limit), 5000))), "verification": FW.verify(eid)})


@router.post("/api/forward/{eid}/evaluate")
def forward_evaluate(eid: str):
    from strata.research import forward as FW
    # what the frozen configuration needs for the sessions about to be sealed is fetched first (free sources only, never offline): a session is sealed once
    return _guard(lambda: FW.evaluate(eid, refresh=True))


class VersionReq(BaseModel):
    config: dict
    name: Optional[str] = None


@router.post("/api/forward/{eid}/version")
def forward_version(eid: str, req: VersionReq):
    """A change to a frozen configuration creates a new experiment; the old one is untouched."""
    from strata.research import forward as FW
    return _guard(lambda: FW.new_version(eid, req.config, req.name))


@router.post("/api/forward/{eid}/stop")
def forward_stop(eid: str):
    from strata.research import forward as FW
    return _guard(lambda: FW.stop(eid))


# ----------------------------------------------------------------------------- provider marketplace
@router.get("/api/data/marketplace")
def marketplace():
    """Every source found: what it advertises, and what this installation has verified."""
    from strata.data.providers import market as MK
    return J(MK.listing(MarketStore()))


class CheckReq(BaseModel):
    source: str


@router.post("/api/data/marketplace/check")
def marketplace_check(req: CheckReq):
    """Start the live check of one source (a handful of small read-only requests)."""
    from strata.data.providers import bench as B
    if req.source not in B.BENCHABLE:
        raise HTTPException(422, "this source has no live check: it is used through file imports or is not integrated")
    rid = db.create_run("provider_benchmark", f"Live check: {req.source}", {"source": req.source})
    _jobs().submit(rid, "provider_benchmark", {"source": req.source})
    return {"run_id": rid}


@router.post("/api/data/alpaca/validate")
def alpaca_validate():
    """The guided connection: credentials, entitlements, an expired contract, its trade bars, a small backtest."""
    rid = db.create_run("alpaca_validation", "Alpaca guided connection", {"symbol": "SPY"})
    _jobs().submit(rid, "alpaca_validation", {"symbol": "SPY"})
    return {"run_id": rid}


@router.get("/api/data/alpaca/validation")
def alpaca_validation_result():
    from strata.data.providers import bench as B
    return J(B.saved().get("alpaca_validation") or {"provider": "alpaca", "ok": False, "verified": False, "steps": [], "summary": "The guided connection has not been run on this installation."})


class EstimateReq(BaseModel):
    config: Optional[dict] = None
    run_id: Optional[str] = None
    strategy_id: Optional[str] = None


@router.post("/api/data/marketplace/estimate")
def marketplace_estimate(req: EstimateReq):
    """What it would take, per source, to obtain only the option observations one strategy needs."""
    from strata.api.results import load_run_result
    from strata.data.providers import market as MK

    def go():
        cfg = req.config
        if cfg is None and req.run_id:
            run = db.get_run(req.run_id)
            if run is None:
                raise LookupError("run not found")
            cfg = (load_run_result(run).get("config") if run["status"] == "done" else None) or (run.get("config") or {}).get("config") or run.get("config")
        if cfg is None and req.strategy_id:
            st = db.get_strategy(req.strategy_id)
            if st is None:
                raise LookupError("strategy not found")
            from datetime import date, timedelta
            cfg = {"strategies": [st["config"]], "start": (date.today() - timedelta(days=730)).isoformat(), "end": date.today().isoformat()}
        if not cfg or "strategies" not in cfg:
            raise ValueError("an estimate needs a strategy: pass a backtest configuration, a finished run or a saved strategy")
        store = MarketStore()
        out = MK.estimate(MK.needs_of(cfg, store))
        status = {x["id"]: x["verification"]["status"] for x in MK.listing(store)["sources"]}
        for row in out["sources"]:  # a price is not a promise that the source works here: say what has actually been verified
            row["verification"] = status.get(row["id"], "unverified")
            if row["verification"] != "verified":
                row["ready"] = False
        return out
    return _guard(go)


# ----------------------------------------------------------------------------- notifications (in-app only)
@router.get("/api/notifications")
def notifications(limit: int = 50, unread: bool = False):
    from strata.api import notify as N
    return J(N.listing(min(limit, 200), unread))


class ReadReq(BaseModel):
    ids: Optional[list[int]] = None


@router.post("/api/notifications/read")
def notifications_read(req: ReadReq):
    from strata.api import notify as N
    return {"marked": N.mark_read(req.ids)}


# ----------------------------------------------------------------------------- strategy IDE
class IdeValidateReq(BaseModel):
    strategy: dict


@router.post("/api/ide/validate")
def ide_validate(req: IdeValidateReq):
    from strata.api import ide as IDE
    return J(IDE.validate(req.strategy))


@router.post("/api/ide/compact")
def ide_compact(req: IdeValidateReq):
    """The strategy without the fields at their defaults (what the structured view shows), and the full form."""
    from strata.api import ide as IDE
    return _guard(lambda: {"strategy": IDE.compact(req.strategy), "full": IDE.expand(req.strategy)})


class IdeTextReq(BaseModel):
    text: str


@router.post("/api/ide/from-text")
def ide_from_text(req: IdeTextReq):
    """A description in plain English -> a strategy document. Uses the notebook's interpreter; runs nothing."""
    from strata.api import ide as IDE
    store = MarketStore()
    return _ai(lambda: IDE.from_interpretation(_agent()._rules(req.text, None, sorted(store.stored_symbols()), None, _default_symbol()), store))


class IdeEditReq(BaseModel):
    strategy: dict
    path: str = ""
    instruction: str


@router.post("/api/ide/edit")
def ide_edit(req: IdeEditReq):
    """Apply an instruction to the selected section only; everything else comes back unchanged."""
    from strata.api import ide as IDE
    return _guard(lambda: IDE.edit_section(req.strategy, req.path, req.instruction))


class IdeDebugReq(BaseModel):
    strategy: dict
    day: str
    watch: list[str] = []
    break_when: Optional[str] = None
    execution: Optional[dict] = None


@router.post("/api/ide/debug")
def ide_debug(req: IdeDebugReq):
    from datetime import date as _date

    from strata.api import ide as IDE
    return _guard(lambda: IDE.debug_session(req.strategy, _date.fromisoformat(req.day), MarketStore(), req.watch, req.break_when, req.execution))


class IdePreviewReq(BaseModel):
    symbol: str
    expr: str
    at: str = "open"


@router.post("/api/ide/preview-rule")
def ide_preview(req: IdePreviewReq):
    from strata.api import ide as IDE
    return _guard(lambda: IDE.preview_rule(req.symbol, req.expr, req.at, MarketStore()))


@router.get("/api/ide/reference")
def ide_reference():
    """Everything the editor completes and documents: rule-language names with signatures,
    the fields of each object, other symbols' fields, stored signal lists, and the schema of a strategy."""
    from strata.engine import expr as X
    from strata.engine.conditions import LEAF_TYPES
    from strata.engine.models import StrategyConfig
    from strata.research import signals as SG
    store = MarketStore()
    return J({"names": [{"name": k, **v, "function": k in X.FUNCTIONS} for k, v in X.REGISTRY.items()], "fields": {k: sorted(v) for k, v in X.FIELDS.items()}, "symbol_fields": sorted(X.SYMBOL_FIELDS),
              "symbols": sorted(s for s in store.stored_symbols() if not store.is_synthetic(s)), "signals": [x["name"] for x in SG.listing()], "leaf_types": LEAF_TYPES, "schema": StrategyConfig.model_json_schema(),
              "examples": ["previous_session.return_pct <= -2", "price > vwap and QQQ.from_open_pct > 0", "crosses_above(price, opening_range_high(30))", "rsi(14, \"5m\") < 30", "portfolio.open_positions < 2", "atr_pct(14) > 1.5"]})


# ----------------------------------------------------------------------------- onboarding
@router.get("/api/onboarding")
def onboarding():
    """Where a new installation stands and what its stored data can answer right now."""
    from strata.api.settings import get_setting
    from strata.data.providers import market as MK
    store = MarketStore()
    syms = sorted(s for s in store.stored_symbols() if not store.is_synthetic(s))
    rows = []
    for sym in syms:
        try:
            d = store.read_daily(sym)
        except Exception:  # noqa: BLE001
            continue
        if d.is_empty():
            continue
        intr = {}
        for res in ("1m", "5m", "60m"):
            try:
                b = store.read_bars(sym, res)
                if b is not None and not b.is_empty():
                    intr[res] = int(b["ts"].dt.convert_time_zone("America/New_York").dt.date().n_unique())
            except Exception:  # noqa: BLE001
                pass
        try:
            qd = len(store.option_days(sym, "quote"))
        except Exception:  # noqa: BLE001
            qd = 0
        try:
            idx = store.option_bar_index(sym)
            td = int(idx["date"].n_unique()) if not idx.is_empty() else 0
        except Exception:  # noqa: BLE001
            td = 0
        rows.append({"symbol": sym, "daily_sessions": d.height, "first": str(d["date"][0]), "last": str(d["date"][-1]), "intraday_sessions": intr, "option_quote_sessions": qd, "option_trade_sessions": td})
    main = next((r for r in rows if r["symbol"] == str(get_setting("data.default_symbol") or "SPY")), None) or next((r for r in rows if r["symbol"] not in ("VIX",)), None)
    sugg = []
    if main:
        s = main["symbol"]
        sugg.append({"text": f"Does {s} tend to recover after large down days, even after controlling for volatility?", "needs": "daily bars", "ready": main["daily_sessions"] >= 750})
        sugg.append({"text": f"Is {s} more likely to close higher when it gaps down more than 1% and VIX is above 25?", "needs": "daily bars and the volatility index", "ready": main["daily_sessions"] >= 750 and any(r["symbol"] == "VIX" for r in rows)})
        other = next((r["symbol"] for r in rows if r["symbol"] not in (s, "VIX")), None)
        if other:
            sugg.append({"text": f"Is {s} more likely to rally in the afternoon when the morning sold off and {other} outperformed {s}?", "needs": "intraday bars for both symbols", "ready": bool(main["intraday_sessions"]) and bool(next(r for r in rows if r["symbol"] == other)["intraday_sessions"])})
        sugg.append({"text": f"What happens to {s} after three consecutive down days?", "needs": "daily bars", "ready": main["daily_sessions"] >= 250})
    has_data = bool(main)
    opt_q = sum(r["option_quote_sessions"] for r in rows)
    opt_t = sum(r["option_trade_sessions"] for r in rows)
    mk = MK.listing(store)
    unlock = [{"id": r["id"], "name": r["name"], "what": r["advertised"]["data"], "history": r["advertised"]["history"], "cost": r["pricing"]["summary"], "status": r["verification"]["status"], "integration": r["integration"]}
              for r in mk["sources"] if r["id"] in ("github_eod", "alpaca", "massive", "databento")]
    done = str(get_setting("ui.onboarded") or "") == "1"
    steps = [{"id": "data", "title": "Free price history", "done": has_data, "detail": (f"{len(rows)} symbol(s) stored; {main['symbol']} has {main['daily_sessions']:,} daily sessions from {main['first']}." if has_data else
                                                                                   "Nothing is stored yet. About 3 MB of real daily and recent intraday history for SPY, QQQ, IWM and VIX can be downloaded for free, with no account.")},
             {"id": "question", "title": "A first question", "done": bool(db.list_runs(kind="graph", limit=1) or db.list_runs(kind="study", limit=1) or db.list_runs(kind="hypothesis", limit=1)), "detail": "Ask one of the questions below; each can be answered with what is stored."},
             {"id": "options", "title": "Option prices for backtests", "done": bool(opt_q or opt_t), "detail": (f"{opt_q:,} sessions of option quotes and {opt_t:,} sessions of option trade prints are stored." if (opt_q or opt_t) else
                                                                                                            "No option prices are stored. Market questions work without them; pricing an option strategy needs observed option prices.")}]
    return J({"onboarded": done, "has_data": has_data, "symbols": rows, "suggestions": sugg, "steps": steps, "unlock": unlock, "mode": str(get_setting("ui.mode") or "simple"),
              "free_download": {"symbols": ["SPY", "QQQ", "IWM", "VIX"], "size": "about 3 MB", "source": "Yahoo Finance and Cboe (free, no account)", "what": "daily history since 1993 and recent 1-, 5- and 60-minute bars"}})
