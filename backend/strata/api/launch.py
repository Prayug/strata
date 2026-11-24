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
        from strata.engine.conditions import extra_symbols
        try:
            syms |= set(extra_symbols(x.entry.conditions))
        except Exception:  # noqa: BLE001
            pass
    vers = {k: s.dataset_version(k) for k in sorted(syms)}
    synthetic = any(s.symbol_meta(k).synthetic for k in syms)
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
                        hashlib.sha256(body.encode()).hexdigest()[:16], vers, ENGINE_VERSION, bool(meta.synthetic))
    jobs.submit(rid, "study", {"config": cfg.model_dump(mode="json")})
    return {"id": rid, "cached": False}
