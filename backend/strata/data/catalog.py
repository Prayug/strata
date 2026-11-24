"""Dataset catalog: what is stored, at which resolution, from which source, with which gaps."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from strata.core import calendar as cal
from strata.data.store import INTRADAY_RES, MarketStore


def _cache_path(store: MarketStore):
    return store.root / "catalog_cache.json"


def invalidate(store: MarketStore | None = None):
    store = store or MarketStore()
    p = _cache_path(store)
    if p.exists():
        p.unlink()


def _missing(days: list[date]) -> list[str]:
    if not days:
        return []
    have = set(days)
    return [d.isoformat() for d in cal.trading_days(days[0], days[-1]) if d not in have]


def symbol_coverage(store: MarketStore, sym: str) -> dict:
    meta = store.symbol_meta(sym)
    out: dict = {"symbol": sym, "meta": meta.__dict__, "dataset_version": store.dataset_version(sym), "storage_bytes": store.storage_bytes(sym)}
    sources = (meta.extra or {}).get("sources") or {}
    # ---- daily
    if store.daily_path(sym).exists():
        d = store.read_daily(sym)
        if not d.is_empty():
            days = d["date"].to_list()
            miss = _missing(days)
            out["daily"] = {"first": days[0].isoformat(), "last": days[-1].isoformat(), "sessions": len(days), "missing_sessions": len(miss), "missing_sample": miss[-12:],
                            "provider": (sources.get("underlying:1d") or {}).get("provider")}
    # ---- intraday by resolution
    intr = {}
    for res in store.bar_resolutions(sym):
        b = store.read_bars(sym, res)
        if b.is_empty():
            continue
        per = b.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d").agg(pl.len().alias("n")).sort("d")
        days = per["d"].to_list()
        counts = dict(zip(days, per["n"].to_list()))
        secs = INTRADAY_RES[res]
        partial = [d.isoformat() for d in days if counts[d] < -(-cal.session_minutes(d) * 60 // secs)]
        miss = _missing(days)
        intr[res] = {"first": days[0].isoformat(), "last": days[-1].isoformat(), "sessions": len(days), "bars": b.height, "missing_sessions": len(miss), "missing_sample": miss[-12:],
                     "incomplete_sessions": len(partial), "incomplete_sample": partial[-8:], "provider": (sources.get(f"underlying:{res}") or {}).get("provider")}
    if intr:
        out["intraday"] = intr
        best = next(r for r in ("1m", "5m", "15m", "30m", "60m") if r in intr)
        out["underlying"] = {**intr[best], "resolution_seconds": INTRADAY_RES[best], "missing_sessions": intr[best]["missing_sample"], "incomplete_count": intr[best]["incomplete_sessions"], "incomplete_sessions": []}
    # ---- options
    od = store.option_days(sym)
    if od:
        osrc = next((v for k, v in sources.items() if k.startswith("options:")), {})
        expected = cal.trading_days(od[0], od[-1])
        have = set(od)
        missing = [d for d in expected if d not in have]
        rows, snaps, gaps = 0, [], []
        interval = meta.option_interval_minutes
        eod = (osrc.get("resolution") == "EOD")
        for d in od:
            p = store.option_path(sym, d)
            rows += pq.read_metadata(p).num_rows
            if eod:
                snaps.append(1)
                continue
            t = pq.read_table(p, columns=["ts"]).column("ts").to_numpy()
            u = np.unique(t.astype("datetime64[s]").astype(np.int64))
            snaps.append(len(u))
            if len(u) > 2:
                step = int(np.median(np.diff(u)))
                interval = interval or max(1, step // 60)
                exp_n = cal.session_minutes(d) // max(1, step // 60) + 1
                if len(u) < exp_n:
                    gaps.append({"date": d.isoformat(), "snapshots": int(len(u)), "expected": int(exp_n)})
        med = float(np.median(snaps)) if snaps else 0
        is_eod = eod or med <= 1
        out["options"] = {"first": od[0].isoformat(), "last": od[-1].isoformat(), "sessions": len(od), "expected_sessions": len(expected),
                          "missing_sessions": [d.isoformat() for d in missing][-40:], "missing_count": len(missing), "rows": rows, "median_snapshots": med,
                          "resolution": "EOD" if is_eod else f"{interval or '?'}m", "resolution_minutes": None if is_eod else interval, "quote_type": osrc.get("quote_type") or "quotes",
                          "provider": osrc.get("provider") or meta.provider, "license": osrc.get("license", ""), "snapshot_gaps": gaps[:200], "snapshot_gap_count": len(gaps),
                          "daily_snapshots": [{"date": d.isoformat(), "n": int(n)} for d, n in zip(od, snaps)]}
    iv = store.read_iv_history(sym)
    if not iv.is_empty():
        out["iv_history"] = {"first": iv["date"][0].isoformat(), "last": iv["date"][-1].isoformat(), "rows": iv.height}
    out["sources"] = list(sources.values())
    return out


def catalog(store: MarketStore | None = None, refresh: bool = False) -> list[dict]:
    store = store or MarketStore()
    syms = set(store.symbols().keys())
    for sub in ("underlying", "options", "daily"):
        p = store.root / sub
        if p.exists():
            syms |= {x.name if x.is_dir() else x.stem for x in p.iterdir() if not x.name.startswith(".")}
    cache = {}
    cp = _cache_path(store)
    if cp.exists() and not refresh:
        try:
            cache = json.loads(cp.read_text())
        except json.JSONDecodeError:
            cache = {}
    out, changed = [], False
    for s in sorted(syms):
        ver = store.manifest(s).get("version") or store.rebuild_manifest(s).get("version")
        metaver = json.dumps(store.symbol_meta(s).__dict__, sort_keys=True, default=str)
        c = cache.get(s)
        if c and c.get("dataset_version") == ver and c.get("_meta") == metaver:
            out.append(c)
            continue
        c = symbol_coverage(store, s)
        c["_meta"] = metaver
        cache[s] = c
        changed = True
        out.append(c)
    if changed:
        cp.write_text(json.dumps(cache, default=str))
    return out


def profile(store: MarketStore, sym: str) -> dict | None:
    for c in catalog(store):
        if c["symbol"] == sym:
            return c
    return None
