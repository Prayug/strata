"""Submitting backtests and market studies to the job system, with result caching.

A run is reused only when its configuration, the versions of every dataset it reads, the
event calendar and the engine version are all identical. Shared by the REST API and the
AI research agent so both go through exactly the same execution path.
"""
from __future__ import annotations

import hashlib
import json

from strata.api import db
from strata.config import ENGINE_VERSION
from strata.data.store import MarketStore
from strata.engine.models import BacktestConfig


def backtest_cache_key(cfg: BacktestConfig, store: MarketStore | None = None) -> tuple[str, dict, bool]:
    s = store or MarketStore()
    syms = {x.symbol for x in cfg.strategies}
    for x in cfg.strategies:
        m = s.symbol_meta(x.symbol)
        if x.vol_symbol or m.vol_index:
            syms.add(x.vol_symbol or m.vol_index)
        syms |= x.symbols()
    vers = {k: s.dataset_version(k) for k in sorted(syms)}
    synthetic = any(s.is_synthetic(k) for k in syms)
    h = hashlib.sha256((cfg.canonical_json() + json.dumps(vers, sort_keys=True) + s.events_version() + ENGINE_VERSION).encode()).hexdigest()[:24]
    return h, vers, synthetic


def launch_backtest(jobs, cfg: BacktestConfig, name: str = "", strategy_id: str | None = None, force: bool = False) -> dict:
    key, vers, synth = backtest_cache_key(cfg)
    if not force:
        hit = db.find_cached(key)
        if hit:
            return {"id": hit["id"], "cached": True}
    name = name or cfg.label or " + ".join(s.name for s in cfg.strategies)
    rid = db.create_run("backtest", name, cfg.model_dump(mode="json"), strategy_id, key, cfg.config_hash(), vers, ENGINE_VERSION, synth)
    jobs.submit(rid, "backtest", {"config": cfg.model_dump(mode="json")})
    return {"id": rid, "cached": False}


def launch_study(jobs, cfg, name: str = "", force: bool = False) -> dict:
    from strata.research.study import StudyConfig
    cfg = cfg if isinstance(cfg, StudyConfig) else StudyConfig.model_validate(cfg)
    s = MarketStore()
    meta = s.symbol_meta(cfg.symbol)
    syms = {cfg.symbol} | ({meta.vol_index} if meta.vol_index else set())
    from strata.engine.conditions import extra_symbols
    syms |= set(extra_symbols(cfg.condition))
    vers = {k: s.dataset_version(k) for k in sorted(syms)}
    body = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(("study" + body + json.dumps(vers, sort_keys=True) + s.events_version() + ENGINE_VERSION).encode()).hexdigest()[:24]
    if not force:
        hit = db.find_cached(key)
        if hit:
            return {"id": hit["id"], "cached": True}
    rid = db.create_run("study", name or cfg.label or f"{cfg.symbol} market study", cfg.model_dump(mode="json"), None, key,
                        hashlib.sha256(body.encode()).hexdigest()[:16], vers, ENGINE_VERSION, bool(s.is_synthetic(cfg.symbol)))
    jobs.submit(rid, "study", {"config": cfg.model_dump(mode="json")})
    return {"id": rid, "cached": False}


def launch_research(jobs, spec: dict, name: str = "", force: bool = False) -> dict:
    """Submit a research pipeline run (coverage -> free acquisition -> signals -> selective
    option data -> simulation -> report). The run is stored under the kind of its result
    ("study" or "backtest") so every results view can open it."""
    from strata.research.pipeline import fingerprint
    s = MarketStore()
    kind = "study" if spec["kind"] == "study" else "backtest"
    key = fingerprint(spec, s)
    if not force:
        hit = db.find_cached(key)
        if hit and (hit.get("summary") or {}).get("outcome") == "completed":
            return {"id": hit["id"], "cached": True}
    cfg = spec["study"] if kind == "study" else spec["backtest"]
    sym = (cfg.get("symbol") if kind == "study" else cfg["strategies"][0]["symbol"])
    body = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    rid = db.create_run(kind, name or spec.get("label") or f"{sym} research", {**cfg, "_research": {k: spec.get(k) for k in ("mode", "requested", "approvals", "provider")}}, None, None,
                        hashlib.sha256(body.encode()).hexdigest()[:16], {}, ENGINE_VERSION, bool(s.is_synthetic(sym)))
    jobs.submit(rid, "research", {"spec": spec})
    return {"id": rid, "cached": False}
