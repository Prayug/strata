"""V6 routes: what the stored credentials can really retrieve (access reports), research
readiness computed from local data, the acquisition options for what is missing, and the
audits (minute-bar coverage, execution accounting)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strata.api import db
from strata.api.research_api import J
from strata.api.v4_api import _jobs

router = APIRouter()


# ----------------------------------------------------------------------------- provider access
@router.get("/api/data/access/alpaca")
def alpaca_access_report():
    """The last access report. Without one, the state is derived from whether a key is stored: nothing is requested here."""
    from strata.data.providers import access as A
    from strata.data.providers import sources as S
    rep = A.saved("alpaca")
    has_key = S.Alpaca().available()[0]
    if rep is None:
        rep = {"provider": "alpaca", "symbol": "SPY", "status": "blocked" if not has_key else "not_run", "verified": False, "requests": [], "matrix": [], "findings": {},
               "summary": ["BLOCKED: no Alpaca key is stored, so nothing about this account can be established."] if not has_key else ["A key is stored. The access check has not been run yet."],
               **({"needs": ["ALPACA_API_KEY", "ALPACA_API_SECRET"], "how": A.ALPACA_HOW} if not has_key else {})}
    rep["key_stored"] = has_key
    # a report made before the key was entered no longer describes this installation
    rep["stale"] = bool(has_key and rep.get("status") == "blocked")
    return J(rep)


@router.post("/api/data/access/alpaca")
def alpaca_access_run():
    """Run the access check: with a key, the expired-contract matrix and the start of the history (about 150 small read-only requests);
    without one, four unauthenticated requests that show the endpoints answer and only the key is missing."""
    rid = db.create_run("alpaca_access", "Alpaca access check", {"symbol": "SPY"})
    _jobs().submit(rid, "alpaca_access", {"symbol": "SPY"})
    return {"run_id": rid}


# ----------------------------------------------------------------------------- what the missing observations would cost
class BudgetReq(BaseModel):
    config: Optional[dict] = None
    run_id: Optional[str] = None
    strategy_id: Optional[str] = None


def _config_of(req) -> dict:
    cfg = req.config
    if cfg is None and req.run_id:
        run = db.get_run(req.run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        cfg = run.get("config")
    if cfg is None and req.strategy_id:
        st = db.get_strategy(req.strategy_id)
        if st is None:
            raise HTTPException(404, "strategy not found")
        cfg = {"strategies": [st["config"]]} if "strategies" not in (st.get("config") or {}) else st["config"]
    if not cfg or "strategies" not in cfg:
        raise HTTPException(422, "a strategy configuration (or a finished backtest) is needed")
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


@router.post("/api/data/budget")
def data_budget(req: BudgetReq):
    """For one strategy: the sessions its entry rule fires on, which of them are priced by what is stored, and what each
    budget ($0 to $200) would make testable. Read-only: signals come from the underlying alone and nothing is requested from any provider."""
    from strata.data.providers import market as MK
    from strata.data.store import MarketStore
    cfg = _config_of(req)
    store = MarketStore()
    try:
        need = MK.needs_of(cfg, store)
    except (LookupError, ValueError) as e:
        raise HTTPException(422, str(e))
    return J({"budget": MK.budget_table(need), "estimate": MK.estimate(need)})


# ----------------------------------------------------------------------------- the original use case, on stored observations
@router.get("/api/research/usecase")
def usecase_report():
    from strata.research import usecase as U
    return J(U.saved() or {"status": "not_run", "thresholds": [], "notes": ["The use-case report has not been run on this installation yet."]})


@router.post("/api/research/usecase")
def usecase_run():
    """Qualifying sessions for falls of 1-4%, what is stored for each, the engine's result and an independent second computation. Reads stored data only."""
    rid = db.create_run("usecase", "Use case: 0DTE calls after a down day", {"symbol": "SPY"})
    _jobs().submit(rid, "usecase", {"symbol": "SPY"})
    return {"run_id": rid}


# ----------------------------------------------------------------------------- audits
@router.get("/api/data/bars/audit")
def bars_audit(symbol: str = "SPY"):
    """Stored intraday bars against the exchange calendar and the daily record: missing and partial sessions, bars outside hours,
    clock changes, the official close, volume, mixed feeds. Read-only."""
    from strata.audit import bars as AB
    from strata.data.store import MarketStore
    st = MarketStore()
    sym = symbol.upper()
    if not st.has_daily(sym) and not st.datasets(sym, "underlying"):
        raise HTTPException(404, f"no price data is stored for {sym}")
    return J(AB.audit(st, sym))


# ----------------------------------------------------------------------------- research readiness
@router.get("/api/data/readiness")
def data_readiness(symbols: Optional[str] = None):
    """Per symbol: on how many sessions each kind of research is possible with the files stored here (never with what a provider advertises)."""
    from strata.data import readiness as RD
    return J(RD.readiness(symbols=[x.strip().upper() for x in symbols.split(",") if x.strip()] if symbols else None))


@router.post("/api/data/precheck")
def data_precheck(req: BudgetReq):
    """One strategy before it is planned: the sessions its entry rule fires on, which can be priced on stored observations, and the alternatives for the rest."""
    from strata.data import readiness as RD
    try:
        return J(RD.precheck(_config_of(req)))
    except (LookupError, ValueError) as e:
        raise HTTPException(422, str(e))


# ----------------------------------------------------------------------------- how well requests are understood
@router.get("/api/ai/evaluation")
def ai_evaluation():
    """The last measured interpretation accuracy (development and held-out sentences), or a note that it has not been run here."""
    import json

    from strata.config import get_settings
    f = get_settings().data_dir / "reports" / "nl_eval.json"
    if not f.exists():
        return J({"status": "not_run", "note": "The interpretation evaluation has not been run on this installation (strata eval-nl, or the button on the Settings page)."})
    return J(json.loads(f.read_text()))


@router.post("/api/ai/evaluation")
def ai_evaluation_run(llm: bool = False):
    """Run the 150-sentence interpretation evaluation. With llm=true the configured language model reads the sentences (refused, not faked, when none is configured)."""
    rid = db.create_run("nl_eval", "Interpretation evaluation" + (" (language model)" if llm else ""), {"llm": llm})
    _jobs().submit(rid, "nl_eval", {"llm": llm})
    return {"run_id": rid}
