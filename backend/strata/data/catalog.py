"""Dataset catalog: coverage, gaps, resolution and storage per symbol."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from strata.core import calendar as cal
from strata.data.store import MarketStore


def _cache_path(store: MarketStore):
    return store.root / "catalog_cache.json"


def invalidate(store: MarketStore | None = None):
    store = store or MarketStore()
    p = _cache_path(store)
    if p.exists():
        p.unlink()


def symbol_coverage(store: MarketStore, sym: str) -> dict:
    meta = store.symbol_meta(sym)
    out = {"symbol": sym, "meta": meta.__dict__, "dataset_version": store.dataset_version(sym), "storage_bytes": store.storage_bytes(sym)}
    m = store.read_minutes(sym)
    if not m.is_empty():
        per = m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d").agg(pl.len().alias("n")).sort("d")
        days = per["d"].to_list()
        counts = dict(zip(days, per["n"].to_list()))
        expected = cal.trading_days(days[0], days[-1])
        missing_days = [d for d in expected if d not in counts]
        short = [{"date": d.isoformat(), "bars": counts[d], "expected": cal.session_minutes(d)} for d in expected if d in counts and counts[d] < cal.session_minutes(d)]
        ts = m["ts"].dt.epoch("s").to_numpy()
        res = int(np.median(np.diff(ts[:2000]))) if len(ts) > 2 else None
        out["underlying"] = {"first": days[0].isoformat(), "last": days[-1].isoformat(), "sessions": len(days), "expected_sessions": len(expected),
                             "missing_sessions": [d.isoformat() for d in missing_days], "incomplete_sessions": short[:200], "incomplete_count": len(short),
                             "bars": m.height, "resolution_seconds": res}
    od = store.option_days(sym)
    if od:
        expected = cal.trading_days(od[0], od[-1])
        have = set(od)
        missing = [d for d in expected if d not in have]
        rows = 0
        snaps = []
        gaps = []
        interval = meta.option_interval_minutes
        for d in od:
            p = store.option_path(sym, d)
            md = pq.read_metadata(p)
            rows += md.num_rows
            t = pq.read_table(p, columns=["ts"]).column("ts").to_numpy()
            u = np.unique(t.astype("datetime64[s]").astype(np.int64))
            snaps.append(len(u))
            if len(u) > 2:
                step = int(np.median(np.diff(u)))
                interval = interval or step // 60
                exp_n = cal.session_minutes(d) // max(1, step // 60) + 1
                if len(u) < exp_n:
                    gaps.append({"date": d.isoformat(), "snapshots": int(len(u)), "expected": int(exp_n)})
        out["options"] = {"first": od[0].isoformat(), "last": od[-1].isoformat(), "sessions": len(od), "expected_sessions": len(expected),
                          "missing_sessions": [d.isoformat() for d in missing], "rows": rows, "median_snapshots": float(np.median(snaps)),
                          "resolution_minutes": interval, "snapshot_gaps": gaps[:200], "snapshot_gap_count": len(gaps),
                          "daily_snapshots": [{"date": d.isoformat(), "n": int(n)} for d, n in zip(od, snaps)]}
    iv = store.read_iv_history(sym)
    if not iv.is_empty():
        out["iv_history"] = {"first": iv["date"][0].isoformat(), "last": iv["date"][-1].isoformat(), "rows": iv.height}
    return out


def catalog(store: MarketStore | None = None, refresh: bool = False) -> list[dict]:
    store = store or MarketStore()
    syms = set(store.symbols().keys())
    for sub in ("underlying", "options"):
        p = store.root / sub
        if p.exists():
            syms |= {x.name for x in p.iterdir() if x.is_dir()}
    cache = {}
    cp = _cache_path(store)
    if cp.exists() and not refresh:
        try:
            cache = json.loads(cp.read_text())
        except json.JSONDecodeError:
            cache = {}
    out = []
    changed = False
    for s in sorted(syms):
        ver = store.manifest(s).get("version") or store.rebuild_manifest(s).get("version")
        c = cache.get(s)
        if c and c.get("dataset_version") == ver:
            out.append(c)
            continue
        c = symbol_coverage(store, s)
        cache[s] = c
        changed = True
        out.append(c)
    if changed:
        cp.write_text(json.dumps(cache, default=str))
    return out
