"""V7 routes: execution on individual trades (the same decisions filled on the first eligible print after them)."""
from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from strata.api import db
from strata.api.research_api import J
from strata.api.v4_api import _jobs

router = APIRouter()


# ----------------------------------------------------------------------------- execution on individual trades
class HiresReq(BaseModel):
    exit_policy: str = "decision_clock"
    latency_seconds: Optional[float] = None
    max_fill_delay_seconds: Optional[float] = None
    conditions: Optional[str] = None
    min_size: Optional[int] = None
    fill_model: Optional[str] = None
    participation: Optional[float] = None
    open_delay_seconds: Optional[float] = None
    max_leg_gap_seconds: Optional[float] = None
    limit_needs_trade_through: Optional[bool] = None


@router.get("/api/backtests/{rid}/hires")
def hires_get(rid: str):
    """The stored execution checks of a run, what the check does, and whether it can be run here."""
    from strata.data.providers import sources as S
    from strata.engine import hires as H
    run = db.get_run(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    try:
        reports = H.saved(rid)
    except Exception:  # noqa: BLE001
        reports = {}
    ok, why = S.Alpaca().available()
    mode = ((run.get("config") or {}).get("pricing") or {}).get("mode") if isinstance(run.get("config"), dict) else None
    return J({"reports": reports, "available": bool(ok), "blocked": None if ok else f"BLOCKED: {why}. Individual trades come from the provider and need its key.", "pricing_mode": mode,
              "filters": {k: [H.CLASS_TEXT[c] for c in v] for k, v in H.FILTERS.items()},
              "conditions": {k: {"meaning": v[0], "class": v[1]} for k, v in H.CONDITIONS.items()},
              "what": "The backtest filled at the last print it had seen, which happened before its decision. This fills the same decisions on the first eligible print after them, from the provider's individual trades.",
              "limits": "A print is a price at which somebody traded. It is not a bid or an ask and does not guarantee this order would have filled there; no spread is inferred from prints."})


@router.post("/api/backtests/{rid}/hires")
def hires_run(rid: str, req: HiresReq):
    """Start the execution check for a run (a background job: it asks the provider only for the minutes around each fill)."""
    from strata.engine import hires as H
    run = db.get_run(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    settings = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        H.HiresSettings(**settings).validate()
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e))
    jid = db.create_run("hires", f"Execution check on individual trades: {run.get('name') or rid}"[:160], {"target": rid, "settings": settings})
    _jobs().submit(jid, "hires", {"target": rid, "settings": settings})
    return {"id": jid, "target": rid}


@router.get("/api/backtests/{rid}/hires.csv")
def hires_csv(rid: str, policy: str = "decision_clock"):
    """Every fill of the check, with the exact print it used, for independent verification."""
    from strata.engine import hires as H
    rep = H.saved(rid).get(policy)
    if rep is None:
        raise HTTPException(404, "this check has not been run for the run")
    rows = H.csv_rows(rep)
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=execution_check_{rid}_{policy}.csv"})


# ----------------------------------------------------------------------------- bid/ask history: the staged Databento check
class DatabentoReq(BaseModel):
    budget_usd: float = 0.0  # dollars this run may spend. 0: free metadata and the provider's own quote only
    sessions: int = 20
    symbol: str = "SPY"


@router.get("/api/data/access/databento")
def databento_report():
    """The last staged check, or what is needed to run one. Nothing is requested from the provider here."""
    from strata.data.providers import databento as DBN
    rep = DBN.saved()
    has_key = bool(DBN._key())
    if rep is None:
        rep = {"provider": "databento", "status": "blocked" if not has_key else "not_run", "verified": False, "stages": [], "how": DBN.SETUP if not has_key else None, "notes": DBN.NOTES,
               "reason": "BLOCKED: no Databento key is stored." if not has_key else "A key is stored. The check has not been run yet.",
               "documented": {"credit_usd": DBN.CREDIT, "unit_prices_usd_per_gb": DBN.PRICES_DOCUMENTED, "dataset": DBN.DATASET, "schema": DBN.SCHEMA}}
    rep["key_stored"] = has_key
    rep["stale"] = bool(has_key and rep.get("status") == "blocked")
    rep["how"] = DBN.SETUP if not has_key else None  # the steps as they are written now, not as they were when a check was last saved
    rep["notes"] = DBN.NOTES
    return J(rep)


@router.post("/api/data/access/databento")
def databento_run(req: DatabentoReq):
    """Run the staged check. With budget 0 it makes free metadata requests and stops at the price of the sample; it downloads
    the sample only when the budget covers the price the provider quoted. It never spends more than the budget."""
    if req.budget_usd < 0 or req.budget_usd > 125:
        raise HTTPException(422, "the budget must be between 0 and 125 dollars (the size of the free credit)")
    rid = db.create_run("databento_access", "Databento check" + (f" (budget ${req.budget_usd:g})" if req.budget_usd else " (free stages)"), {"budget_usd": req.budget_usd, "sessions": req.sessions, "symbol": req.symbol.upper()})
    _jobs().submit(rid, "databento_access", {"budget_usd": req.budget_usd, "sessions": req.sessions, "symbol": req.symbol.upper()})
    return {"id": rid}


class ObsCompareReq(BaseModel):
    config: dict
    quote_provider: str = "databento"
    trade_provider: str = "alpaca"


@router.get("/api/compare/observations/{symbol}")
def observations_alignment(symbol: str, quote_provider: str = "databento", trade_provider: str = "alpaca"):
    """Do the stored quotes and the stored trade bars of a symbol describe the same market (contracts and clocks)?"""
    from strata.data.store import MarketStore
    from strata.research import obscompare as OC
    return J(OC.alignment(MarketStore(), symbol.upper(), quote_provider, trade_provider))


@router.post("/api/compare/observations")
def observations_execution(req: ObsCompareReq):
    """The same strategy filled at trade prints, at the quote and at the midpoint, on the sessions both datasets cover."""
    from strata.data.store import MarketStore
    from strata.research import obscompare as OC
    try:
        return J(OC.execution(req.config, MarketStore(), req.quote_provider, req.trade_provider))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"{type(e).__name__}: {e}"[:300])


# ----------------------------------------------------------------------------- experiments (one sentence, a grid of settings)
@router.get("/api/experiments/{rid}")
def experiment_get(rid: str):
    """The grid: every configuration with its metrics, the earlier/later split, coverage, and the stable-region analysis."""
    from strata.research import experiment as EXP
    try:
        return J(EXP.load(rid))
    except FileNotFoundError:
        raise HTTPException(404, "no experiment result is stored for this run")


@router.get("/api/experiments/{rid}/trades/{i}")
def experiment_trades(rid: str, i: int):
    """The trades of one configuration of the grid."""
    from strata.research import experiment as EXP
    return J(EXP.trades_of(rid, i))


@router.get("/api/experiments/{rid}/table.csv")
def experiment_csv(rid: str):
    from strata.research import experiment as EXP
    try:
        rows = EXP.csv_rows(EXP.load(rid))
    except FileNotFoundError:
        raise HTTPException(404, "no experiment result is stored for this run")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=experiment_{rid}.csv"})


# ----------------------------------------------------------------------------- forward validation: reconciliation and the stored account
class ReconcileReq(BaseModel):
    force: bool = False  # replay every sealed session from its stored starting state, not only those whose stored data changed


@router.post("/api/forward/{eid}/reconcile")
def forward_reconcile(eid: str, req: ReconcileReq):
    """Compare sealed sessions with the data stored now. Differences are appended as revisions; nothing sealed is altered."""
    from strata.research import forward as FW
    try:
        return J(FW.reconcile(eid, force=req.force))
    except LookupError:
        raise HTTPException(404, "experiment not found")


@router.get("/api/forward/{eid}/account")
def forward_account(eid: str):
    """The stored account of a forward experiment: one state per sealed session (cash, open positions), each with the hash it
    was stored under, the engine that produced it and the fingerprint of the observations the session read."""
    import json as _json

    from strata.research import forward as FW
    if FW.get(eid) is None:
        raise HTTPException(404, "experiment not found")
    rows = []
    for r in FW._states(eid):
        st = _json.loads(r["state"])
        rows.append({"session": r["session"], "origin": r["origin"], "stored_at": r["created_at"], "engine_version": r["engine_version"], "state_hash": r["state_hash"], "data_fingerprint": r["data_fingerprint"],
                     "cash": st.get("cash"), "realised": st.get("realised"), "closed_positions": st.get("closed"),
                     "open_positions": [{"strategy": p.get("strategy"), "entry_ts": p.get("entry_ts"), "units": p.get("units"), "pending_exit": p.get("pending_exit"),
                                         "legs": [{"right": "C" if l.get("is_call") else "P", "strike": l.get("strike"), "qty": l.get("qty"), "entry_price": l.get("entry_price")} for l in p.get("legs") or []]} for p in st.get("positions") or []]})
    return J({"experiment": eid, "states": rows, "note": "Each evaluation starts from the last of these and runs the new sessions only. They are append-only: a stored state cannot be changed or deleted."})


# ----------------------------------------------------------------------------- market replay on stored observations
def _day(x: str):
    from datetime import date
    try:
        return date.fromisoformat(x)
    except ValueError:
        raise HTTPException(400, f"{x!r} is not a date (YYYY-MM-DD)") from None


@router.get("/api/replay/{symbol}/sessions")
def replay_sessions(symbol: str, provider: Optional[str] = None):
    """The sessions of a symbol with stored option observations."""
    from strata.data.store import MarketStore
    from strata.research import replay as RP
    return J(RP.sessions(MarketStore(), symbol.upper(), provider))


@router.get("/api/replay/{symbol}/{day}")
def replay_session(symbol: str, day: str, end: Optional[str] = None, bucket: Optional[int] = None, provider: Optional[str] = None):
    """One session (or a span) of the underlying's stored bars with the engine's levels, and the contracts stored for it."""
    from strata.data.store import MarketStore
    from strata.research import replay as RP
    st, d0 = MarketStore(), _day(day)
    try:
        und = RP.underlying(st, symbol.upper(), d0, _day(end) if end else None, bucket)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return J({"underlying": und, "contracts": RP.contracts(st, symbol.upper(), d0, provider)})


@router.get("/api/replay/{symbol}/contract/{occ}")
def replay_contract(symbol: str, occ: str, start: str, end: Optional[str] = None, bucket: Optional[int] = None, provider: Optional[str] = None):
    """The stored trade bars of one contract: read only when the contract is opened."""
    from strata.data.providers.sources import SourceError
    from strata.data.store import MarketStore
    from strata.research import replay as RP
    try:
        return J(RP.contract(MarketStore(), symbol.upper(), occ.upper(), _day(start), _day(end) if end else None, provider, bucket))
    except SourceError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/api/backtests/{rid}/trades/{tid}/observed")
def replay_trade(rid: str, tid: int):
    """One trade of a run laid over the stored observations of its sessions (decision, fill, print age, levels, marks)."""
    from strata.research import replay as RP
    if db.get_run(rid) is None:
        raise HTTPException(404, "run not found")
    try:
        return J(RP.trade(rid, tid))
    except KeyError:
        raise HTTPException(404, "trade not found") from None
    except FileNotFoundError:
        raise HTTPException(404, "this run has no stored trades") from None


# ----------------------------------------------------------------------------- performance card, fills and the baseline
def _trials_for(rid: str, sid: Optional[str], trials: Optional[int]) -> int:
    """Configurations tried in the research session this run belongs to (the registry is asked; a caller may only raise it)."""
    from strata.research import ledger as LG
    n = int(trials or 1)
    try:
        scope = sid
        if scope is None:
            with db.connect() as con:
                row = con.execute("SELECT scope FROM trials WHERE run_id=? ORDER BY id LIMIT 1", (rid,)).fetchone()
            scope = row[0] if row else None
        if scope:
            n = max(n, LG.count(scope))
    except Exception:  # noqa: BLE001  (a missing registry row never hides a result: the count is then the caller's)
        pass
    return max(1, n)


@router.get("/api/backtests/{rid}/performance")
def performance_card(rid: str, sid: Optional[str] = None, trials: Optional[int] = None):
    """The prioritised performance card of a stored backtest (read from its stored trades)."""
    from strata.api.results import load_result, load_trades
    from strata.engine import hires as H
    from strata.research import performance as PF
    run = db.get_run(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    try:
        res = load_result(rid)
    except FileNotFoundError:
        raise HTTPException(404, "this run has no stored backtest result") from None
    try:
        hr = (H.saved(rid) or {}).get("decision_clock")
    except Exception:  # noqa: BLE001
        hr = None
    return J(PF.card(run, res, load_trades(rid), _trials_for(rid, sid, trials), PF.saved_baseline(rid), hr))


@router.get("/api/backtests/{rid}/fills")
def fills_json(rid: str):
    """Every option fill of a run with the stored bar its price was taken from."""
    from strata.research import performance as PF
    if db.get_run(rid) is None:
        raise HTTPException(404, "run not found")
    return J(PF.fills(rid))


@router.get("/api/backtests/{rid}/fills.csv")
def fills_csv(rid: str):
    from strata.research import performance as PF
    if db.get_run(rid) is None:
        raise HTTPException(404, "run not found")
    out = PF.fills(rid)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=out["columns"], extrasaction="ignore")
    w.writeheader()
    w.writerows(out["fills"])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="fills_{rid}.csv"'})


@router.post("/api/backtests/{rid}/baseline")
def baseline_run(rid: str):
    """Start the baseline of a run: the same position at the same time on every session, whatever the signal says."""
    run = db.get_run(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if not isinstance(run.get("config"), dict) or not run["config"].get("strategies"):
        raise HTTPException(400, "this run has no strategy configuration to take the signal out of")
    jid = db.create_run("baseline", f"Baseline of {run.get('name') or rid}"[:120], {"run": rid}, None, None, None, {}, None, False)
    _jobs().submit(jid, "baseline", {"run": rid})
    return {"id": jid}


# ----------------------------------------------------------------------------- what is on this disk, in one line per symbol
_stored_cache: dict = {}


@router.get("/api/dataset/options")
def dataset_options():
    """The option observations stored here, per symbol: sessions, period, contract-sessions and bars, read from the index of
    each dataset (cached until an index file changes). Also whether the collector is continuing them and whether any
    historical quote source is connected."""
    import polars as pl

    from strata.collector import service as C
    from strata.data.providers import databento as DBN
    from strata.data.store import MarketStore
    st = MarketStore()
    rows = []
    for key in st.datasets(dtype="options", obs="trade"):
        ip = st.dataset_dir(key) / "index.parquet"
        if not ip.exists():
            continue
        stamp = (str(ip), ip.stat().st_mtime_ns)
        if stamp not in _stored_cache:
            idx = pl.read_parquet(ip, columns=["date", "bars", "status"])
            ok = idx.filter(pl.col("status") == "ok")
            _stored_cache[stamp] = None if idx.is_empty() else {"symbol": key.symbol, "provider": key.provider, "sessions": int(idx["date"].n_unique()), "first": idx["date"].min().isoformat(), "last": idx["date"].max().isoformat(),
                                                                "contract_sessions": idx.height, "with_trades": ok.height, "bars": int(ok["bars"].sum()), "observation": "one-minute trade bars (no bid/ask)"}
        if _stored_cache[stamp]:
            rows.append(_stored_cache[stamp])
    rows.sort(key=lambda r: (-r["bars"], r["symbol"]))
    cfg = C.load_config()
    try:
        nd = C.next_due(cfg)
    except Exception:  # noqa: BLE001
        nd = None
    return J({"symbols": rows, "bars": sum(r["bars"] for r in rows),
              "collector": {"enabled": bool(cfg["enabled"]), "symbols": cfg["symbols"], "provider": cfg["options"]["provider"], "next_due": nd.isoformat() if nd else None},
              "quotes": {"connected": bool(DBN._key()), "text": "Databento key stored" if DBN._key() else "no historical bid/ask source is connected"}})
