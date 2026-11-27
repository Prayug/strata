"""V8 routes: the observations behind any number, what can be tested in plain words, a selection of days described,
starting questions that run on the stored data, and research history as questions and findings."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strata.api import db
from strata.api.research_api import J

router = APIRouter()


def _store():
    from strata.data.store import MarketStore
    return MarketStore()


class ExploreReq(BaseModel):
    spec: dict
    limit: int = 5000


@router.post("/api/v8/explore")
def explore(req: ExploreReq):
    """The rows behind a number: the days of a study, the trades of a backtest, the classified days of an agreement."""
    from strata.research import explore as X
    try:
        out = X.observations(req.spec, _store())
    except LookupError as e:
        raise HTTPException(404, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    out["rows"] = out["rows"][: max(1, min(req.limit, 20000))]
    return J(out)


@router.get("/api/v8/scope/{symbol}")
def scope(symbol: str):
    from strata.research import explore as X
    try:
        return J(X.scope(_store(), symbol.upper()))
    except LookupError as e:
        raise HTTPException(404, str(e))


class DaysReq(BaseModel):
    symbol: str
    dates: list[str]


@router.post("/api/v8/days")
def days(req: DaysReq):
    """What a selection of days have in common, against every stored session of the symbol."""
    from strata.research import explore as X
    try:
        return J(X.describe_days(_store(), req.symbol.upper(), req.dates[:5000]))
    except (LookupError, ValueError) as e:
        raise HTTPException(400, str(e))


STARTERS = [
    {"text": "Is the first five-minute candle a good predictor of the day?", "symbol": "SPY", "needs": "minutes", "kind": "agreement",
     "why": "Measured on every SPY session with minute prices."},
    {"text": "What happens after SPY drops 2%?", "symbol": "SPY", "needs": "minutes", "kind": "movement", "why": "Every 2% down day since 2016, minute by minute the next day."},
    {"text": "How often does QQQ recover from a bad morning?", "symbol": "QQQ", "needs": "minutes", "kind": "frequency", "why": "Two meanings of \"recover\", measured side by side."},
    {"text": "When during the day does SPY tend to move the most?", "symbol": "SPY", "needs": "minutes", "kind": "timing", "why": "Each half hour of every whole session."},
    {"text": "Is buying calls every morning profitable historically?", "symbol": "SPY", "needs": "option_trades", "kind": "strategy",
     "why": "Real option trade prices from 2024 onward."},
    {"text": "Compare buying 0DTE and 7DTE SPY calls at the open", "symbol": "SPY", "needs": "option_trades", "kind": "compare", "why": "The same entry, two expirations, side by side."},
    {"text": "Which weekday has the best open-to-close return for IWM?", "symbol": "IWM", "needs": "daily", "kind": "breakdown", "why": "Every IWM session, grouped by weekday, with a test of whether the groups differ."},
    {"text": "How often does SPY fill the gap when it gaps down more than 0.5%?", "symbol": "SPY", "needs": "minutes", "kind": "frequency", "why": "Every gap down, and whether the price came back to the previous close."},
]


@router.get("/api/v8/starters")
def starters():
    """Starting questions, each checked against what is stored: only questions the data can answer are offered."""
    from strata.data.catalog import catalog
    cat = {c["symbol"]: c for c in catalog(_store())}
    out = []
    for s in STARTERS:
        c = cat.get(s["symbol"])
        if c is None or c["meta"].get("synthetic"):
            continue
        ok = {"daily": bool(c.get("daily")), "minutes": "1m" in (c.get("intraday") or {}), "option_trades": bool(c.get("option_trades"))}[s["needs"]]
        if ok:
            out.append(s)
    return J(out)


@router.get("/api/v8/history")
def history(limit: int = 40):
    """Research history as questions and what was found, newest first (no run ids)."""
    out = []
    for s in db.list_sessions(limit):
        msgs = db.list_messages(s["id"])
        qs = [m for m in msgs if m["role"] == "user"]
        if not qs:
            continue
        found = []
        for m in msgs:
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] == "result" and c.get("view") not in ("failed",):
                a = c.get("answer") or {}
                head = a.get("headline") or (c.get("text") or "").split(". ")[0]
                if head:
                    found.append({"message_id": m["id"], "headline": head[:280], "kind": a.get("kind") or c.get("view")})
        first = qs[0]["content"]
        out.append({"id": s["id"], "title": s.get("title") or "", "question": (first.get("text") if isinstance(first, dict) else str(first))[:280],
                    "questions": len(qs), "findings": found[-3:], "updated": s.get("updated_at"), "created": s.get("created_at")})
    return J(out)


class SelectionReq(BaseModel):
    session_id: str
    ids: list[str] = []
    what: str = "days"


@router.post("/api/v8/selection")
def selection(req: SelectionReq):
    """Observations picked in the explore sheet, so that the next question can be about them ("what sets them apart?"). A
    selection belongs to the result it was made on and is not used once another result is shown."""
    from strata.api import db
    s = db.get_session(req.session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    st = s.get("state") if isinstance(s.get("state"), dict) else {}
    latest = next((m["id"] for m in reversed(db.list_messages(req.session_id)) if m["kind"] == "result"), None)
    st = {**st, "selection": ({"ids": [str(x) for x in req.ids][:5000], "what": req.what, "result": latest} if req.ids else None)}
    db.update_session(req.session_id, state=st)
    return {"ok": True, "count": len(req.ids)}
