"""Dataset catalog: what is stored, at which resolution, from which provider, with which gaps.

Coverage is reported per dataset (provider / observation type / resolution) and, for
convenience, as merged per-symbol views (`daily`, `intraday`, `options`, `option_trades`)
that describe exactly what a reader gets when it does not name a provider.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from strata.core import calendar as cal
from strata.data.store import INTRADAY_RES, DatasetKey, MarketStore


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


def dataset_coverage(store: MarketStore, key: DatasetKey) -> dict:
    """Coverage of one dataset, straight from its files."""
    desc = store.descriptor(key)
    base = store.dataset_dir(key)
    out = {"id": key.id, "provider": key.provider, "symbol": key.symbol, "dtype": key.dtype, "obs": key.obs, "resolution": key.res, "version": key.version,
           "license": desc.get("license", ""), "url": desc.get("url", ""), "synthetic": bool(desc.get("synthetic")), "adjustment": desc.get("adjustment"),
           "format": desc.get("format"), "limitations": desc.get("limitations") or [], "label": desc.get("label", ""),
           "bytes": store.storage_bytes(key=key), "head": store.head(key).get("version"), "manifests": len(list((base / "manifests").glob("*.json"))) if (base / "manifests").exists() else 0,
           "first": None, "last": None, "sessions": 0, "rows": 0}
    if key.dtype == "underlying" and key.res == "1d":
        p = base / "data" / "daily.parquet"
        if p.exists():
            d = pl.read_parquet(p, columns=["date"]).sort("date")
            days = d["date"].to_list()
            miss = _missing(days)
            out.update(first=days[0].isoformat(), last=days[-1].isoformat(), sessions=len(days), rows=len(days), missing_sessions=len(miss), missing_sample=miss[-12:])
    elif key.dtype == "underlying":
        b = store.read_bars(key.symbol, key.res, provider=key.provider)
        if not b.is_empty():
            per = b.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d").agg(pl.len().alias("n")).sort("d")
            days = per["d"].to_list()
            counts = dict(zip(days, per["n"].to_list()))
            secs = INTRADAY_RES[key.res]
            partial = [d.isoformat() for d in days if counts[d] < -(-cal.session_minutes(d) * 60 // secs)]
            miss = _missing(days)
            out.update(first=days[0].isoformat(), last=days[-1].isoformat(), sessions=len(days), rows=b.height, bars=b.height, missing_sessions=len(miss), missing_sample=miss[-12:],
                       incomplete_sessions=len(partial), incomplete_sample=partial[-8:])
    elif key.obs == "trade" and desc.get("format") == "bars":
        idx = store.option_bar_index(key.symbol, dataset=key)
        if not idx.is_empty():
            days = sorted(idx["date"].unique().to_list())
            ok = idx.filter(pl.col("status") == "ok")
            out.update(first=days[0].isoformat(), last=days[-1].isoformat(), sessions=len(days), rows=int(idx["bars"].sum()), contract_sessions=idx.height,
                       contract_sessions_with_trades=ok.height, contract_sessions_without_trades=idx.height - ok.height, contracts=int(idx["occ"].n_unique()),
                       expirations=int(idx["expiration"].n_unique()), selective=True,
                       per_session=[{"date": r["date"].isoformat(), "contracts": r["n"], "bars": r["bars"]} for r in idx.group_by("date").agg(pl.len().alias("n"), pl.col("bars").sum()).sort("date").iter_rows(named=True)][-400:])
    else:
        od = store.option_days(key.symbol, key.obs, dataset=key)
        if od:
            states = store.session_states(key)
            expected = cal.trading_days(od[0], od[-1])
            have = set(od)
            missing = [d for d in expected if d not in have]
            partial = [d.isoformat() for d in od if (states.get(d.isoformat()) or {}).get("status") != "complete"]
            rows, snaps, gaps = 0, [], []
            eod = key.res == "EOD"
            interval = None if eod else int(key.res[:-1]) if key.res[:-1].isdigit() else None
            for d in od:
                p = base / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"
                rows += pq.read_metadata(p).num_rows
                if eod:
                    snaps.append(1)
                    continue
                t = pq.read_table(p, columns=["ts"]).column("ts").to_numpy()
                u = np.unique(t.astype("datetime64[s]").astype(np.int64))
                snaps.append(len(u))
                if len(u) > 2:
                    step = int(np.median(np.diff(u)))
                    exp_n = cal.session_minutes(d) // max(1, step // 60) + 1
                    if len(u) < exp_n:
                        gaps.append({"date": d.isoformat(), "snapshots": int(len(u)), "expected": int(exp_n)})
            out.update(first=od[0].isoformat(), last=od[-1].isoformat(), sessions=len(od), expected_sessions=len(expected), rows=rows,
                       missing_sessions=[d.isoformat() for d in missing][-40:], missing_count=len(missing), partial_sessions=partial[-40:], partial_count=len(partial),
                       median_snapshots=float(np.median(snaps)) if snaps else 0, resolution_minutes=interval, snapshot_gaps=gaps[:200], snapshot_gap_count=len(gaps),
                       daily_snapshots=[{"date": d.isoformat(), "n": int(n)} for d, n in zip(od, snaps)])
    return out


def symbol_coverage(store: MarketStore, sym: str) -> dict:
    meta = store.symbol_meta(sym)
    keys = store.datasets(sym)
    dsets = [dataset_coverage(store, k) for k in keys]
    by = {d["id"]: d for d in dsets}
    out: dict = {"symbol": sym, "meta": {**meta.__dict__, "synthetic": store.is_synthetic(sym)}, "dataset_version": store.dataset_version(sym), "storage_bytes": store.storage_bytes(sym),
                 "datasets": dsets}
    # ---- merged daily view (what read_daily returns)
    if store.has_daily(sym):
        d = store.read_daily(sym, with_provider=True)
        if not d.is_empty():
            days = d["date"].to_list()
            miss = _missing(days)
            provs = {r["provider"]: r["n"] for r in d.group_by("provider").agg(pl.len().alias("n")).iter_rows(named=True)}
            out["daily"] = {"first": days[0].isoformat(), "last": days[-1].isoformat(), "sessions": len(days), "missing_sessions": len(miss), "missing_sample": miss[-12:],
                            "provider": max(provs, key=provs.get), "providers": provs}
    # ---- merged intraday view by resolution
    intr = {}
    for res in store.bar_resolutions(sym):
        b = store.read_bars(sym, res, with_provider=True)
        if b.is_empty():
            continue
        per = b.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d").agg(pl.len().alias("n"), pl.col("provider").first()).sort("d")
        days = per["d"].to_list()
        counts = dict(zip(days, per["n"].to_list()))
        secs = INTRADAY_RES[res]
        partial = [d.isoformat() for d in days if counts[d] < -(-cal.session_minutes(d) * 60 // secs)]
        miss = _missing(days)
        provs = {r["provider"]: r["n"] for r in per.group_by("provider").agg(pl.len().alias("n")).iter_rows(named=True)}
        intr[res] = {"first": days[0].isoformat(), "last": days[-1].isoformat(), "sessions": len(days), "bars": b.height, "missing_sessions": len(miss), "missing_sample": miss[-12:],
                     "incomplete_sessions": len(partial), "incomplete_sample": partial[-8:], "provider": max(provs, key=provs.get), "providers": provs}
    if intr:
        out["intraday"] = intr
        best = next(r for r in ("1m", "5m", "15m", "30m", "60m") if r in intr)
        out["underlying"] = {**intr[best], "resolution_seconds": INTRADAY_RES[best], "missing_sessions": intr[best]["missing_sample"], "incomplete_count": intr[best]["incomplete_sessions"], "incomplete_sessions": []}
    # ---- options: the default quote dataset and the default trade dataset, described separately
    q = store.option_dataset(sym, "quote")
    if q is not None and by.get(q.id, {}).get("sessions"):
        c = by[q.id]
        is_eod = c["resolution"] == "EOD" or c.get("median_snapshots", 0) <= 1
        out["options"] = {**{k: c.get(k) for k in ("first", "last", "sessions", "expected_sessions", "missing_sessions", "missing_count", "partial_sessions", "partial_count", "rows",
                                                   "median_snapshots", "snapshot_gaps", "snapshot_gap_count", "daily_snapshots", "license", "limitations")},
                          "resolution": "EOD" if is_eod else c["resolution"], "resolution_minutes": None if is_eod else c.get("resolution_minutes"), "quote_type": "quotes",
                          "provider": c["provider"], "dataset": c["id"], "alternatives": [d["id"] for d in dsets if d["dtype"] == "options" and d["obs"] == "quote" and d["id"] != c["id"]]}
    t = store.option_dataset(sym, "trade")
    if t is not None and by.get(t.id, {}).get("sessions"):
        c = by[t.id]
        out["option_trades"] = {**{k: c.get(k) for k in ("first", "last", "sessions", "rows", "contract_sessions", "contract_sessions_with_trades", "contract_sessions_without_trades",
                                                         "contracts", "expirations", "per_session", "license", "limitations", "format")},
                                "resolution": c["resolution"], "quote_type": "trades", "provider": c["provider"], "dataset": c["id"], "selective": bool(c.get("selective")),
                                "alternatives": [d["id"] for d in dsets if d["dtype"] == "options" and d["obs"] == "trade" and d["id"] != c["id"]]}
    iv = store.read_iv_history(sym)
    if not iv.is_empty():
        out["iv_history"] = {"first": iv["date"][0].isoformat(), "last": iv["date"][-1].isoformat(), "rows": iv.height}
    out["sources"] = [{"provider": d["provider"], "kind": d["dtype"], "resolution": d["resolution"], "license": d["license"], "url": d["url"],
                       "quote_type": {"quote": "quotes", "trade": "trades", "index": "index"}[d["obs"]], "dataset": d["id"]} for d in dsets]
    return out


def catalog(store: MarketStore | None = None, refresh: bool = False) -> list[dict]:
    store = store or MarketStore()
    cache = {}
    cp = _cache_path(store)
    if cp.exists() and not refresh:
        try:
            cache = json.loads(cp.read_text())
        except json.JSONDecodeError:
            cache = {}
    out, changed = [], False
    for s in sorted(store.stored_symbols()):
        ver = store.dataset_version(s)
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
