"""Storage layout migrations.

Layout 2 (Strata V2) kept one set of files per symbol:
    daily/<SYM>.parquet, underlying/<SYM>/<YYYY>.parquet, bars/<res>/<SYM>/<YYYY>.parquet,
    options/<SYM>/<YYYY>/<date>.parquet, derived/<SYM>_iv.parquet, manifests/<SYM>.json
with the provider recorded only as symbol-level metadata. Layout 3 partitions every dataset
by provider / symbol / data type / observation type / resolution / version (see store.py).

The migration MOVES files (no copy, no rewrite), is idempotent, and can be re-run after an
interruption: LAYOUT is written last. Nothing is deleted except empty legacy directories,
the old manifests and the catalog cache, all of which are rebuilt from the data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LEGACY_DIRS = ("daily", "underlying", "bars", "options", "derived", "manifests", "events", "corporate_actions")


def _legacy_present(root: Path) -> bool:
    return any((root / d).exists() for d in ("daily", "underlying", "bars", "options"))


def _mv(src: Path, dst: Path, moved: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src.unlink()  # an earlier, interrupted run already placed this file
    else:
        src.rename(dst)
    moved.append(str(dst))


def _prune(p: Path) -> None:
    if not p.exists():
        return
    for sub in sorted((x for x in p.rglob("*") if x.is_dir()), key=lambda x: -len(x.parts)):
        try:
            sub.rmdir()
        except OSError:
            pass
    try:
        p.rmdir()
    except OSError:
        pass


def plan_layout(root: Path) -> list[dict]:
    """What a migration would do: one entry per legacy file group (used by --dry-run)."""
    root = Path(root)
    sym_f = root / "symbols.json"
    symbols = json.loads(sym_f.read_text()) if sym_f.exists() else {}
    out = []

    def src_of(sym: str, kind: str, res: str) -> dict:
        m = symbols.get(sym) or {}
        sources = (m.get("extra") or {}).get("sources") or {}
        s = sources.get(f"{kind}:{res}")
        if s is None and kind == "options":
            s = next((v for k, v in sources.items() if k.startswith("options:")), None)
        prov = (s or {}).get("provider") or (m.get("provider") if m.get("provider") not in (None, "", "import") else None) or ("synthetic" if m.get("synthetic") else "import")
        return {"provider": prov, "license": (s or {}).get("license", ""), "url": (s or {}).get("url", ""), "quote_type": (s or {}).get("quote_type", ""),
                "resolution": (s or {}).get("resolution"), "synthetic": bool(m.get("synthetic")),
                "adjustment": "split" if kind == "underlying" and (m.get("extra") or {}).get("price_adjustment") == "split" else None}

    if (root / "daily").exists():
        for f in sorted((root / "daily").glob("*.parquet")):
            s = src_of(f.stem, "underlying", "1d")
            out.append({"symbol": f.stem, "dtype": "underlying", "obs": "index" if s["provider"] == "cboe" else "trade", "res": "1d", **s, "files": [(f, "data/daily.parquet")]})
    if (root / "underlying").exists():
        for d in sorted(x for x in (root / "underlying").iterdir() if x.is_dir()):
            s = src_of(d.name, "underlying", "1m")
            out.append({"symbol": d.name, "dtype": "underlying", "obs": "trade", "res": "1m", **s, "files": [(f, f"data/{f.name}") for f in sorted(d.glob("*.parquet"))]})
    if (root / "bars").exists():
        for rd in sorted(x for x in (root / "bars").iterdir() if x.is_dir()):
            for d in sorted(x for x in rd.iterdir() if x.is_dir()):
                s = src_of(d.name, "underlying", rd.name)
                out.append({"symbol": d.name, "dtype": "underlying", "obs": "trade", "res": rd.name, **s, "files": [(f, f"data/{f.name}") for f in sorted(d.glob("*.parquet"))]})
    if (root / "options").exists():
        for d in sorted(x for x in (root / "options").iterdir() if x.is_dir()):
            m = symbols.get(d.name) or {}
            s = src_of(d.name, "options", "")
            res = s["resolution"] or (f"{m['option_interval_minutes']}m" if m.get("option_interval_minutes") else "EOD")
            obs = "trade" if s["quote_type"] == "trades" else "quote"
            files = [(f, f"data/{f.parent.name}/{f.name}") for f in sorted(d.glob("*/*.parquet"))]
            iv = root / "derived" / f"{d.name}_iv.parquet"
            if iv.exists() and obs == "quote":
                files.append((iv, "derived/iv.parquet"))
            out.append({"symbol": d.name, "dtype": "options", "obs": obs, "res": res, **s, "files": files})
    return out


def migrate_layout(root: Path, dry_run: bool = False) -> dict:
    """Bring a market-data directory to the current layout. Returns a report."""
    import pyarrow.parquet as pq

    root = Path(root)
    report: dict = {"from": "2", "to": "3", "datasets": [], "moved_files": 0, "dry_run": dry_run}
    if (root / "LAYOUT").exists():
        return {**report, "from": (root / "LAYOUT").read_text().strip(), "note": "already current"}
    if not _legacy_present(root):
        if not dry_run:
            root.mkdir(parents=True, exist_ok=True)
            (root / "LAYOUT").write_text("3")
        return {**report, "from": "none", "note": "no data to migrate"}
    plan = plan_layout(root)
    for item in plan:
        ident = f"{item['provider']}/{item['symbol']}/{item['dtype']}/{item['obs']}/{item['res']}/v1"
        entry = {"dataset": ident, "files": len(item["files"])}
        report["datasets"].append(entry)
        report["moved_files"] += len(item["files"])
        if dry_run:
            continue
        base = root / "datasets" / item["provider"] / item["symbol"] / item["dtype"] / item["obs"] / item["res"] / "v1"
        base.mkdir(parents=True, exist_ok=True)
        desc_f = base / "dataset.json"
        legacy_trades = item["dtype"] == "options" and item["obs"] == "trade"
        if not desc_f.exists():
            lim = ["Migrated from storage layout 2."]
            if legacy_trades:
                lim.append("Downloaded by the V2 Alpaca ingester, which did not follow pagination: sessions may be truncated. Every session is marked partial; re-download to replace it.")
            desc_f.write_text(json.dumps({
                "id": ident, "provider": item["provider"], "symbol": item["symbol"], "dtype": item["dtype"], "obs": item["obs"], "resolution": item["res"], "version": 1,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "license": item["license"], "url": item["url"], "adjustment": item["adjustment"],
                "synthetic": item["synthetic"], "format": "snapshots" if item["dtype"] == "options" else None, "limitations": lim, "label": "", "migrated_from_layout": 2}, indent=2, sort_keys=True))
        moved: list = []
        for src, rel in item["files"]:
            if src.exists():
                _mv(src, base / rel, moved)
        if item["dtype"] == "options":
            sess_f = base / "sessions.json"
            sess = json.loads(sess_f.read_text()) if sess_f.exists() else {}
            for f in sorted((base / "data").glob("*/*.parquet")):
                if f.stem not in sess:
                    sess[f.stem] = {"status": "partial" if legacy_trades else "complete", "rows": pq.read_metadata(f).num_rows, "migrated": True}
            sess_f.write_text(json.dumps(sess, sort_keys=True))
    if dry_run:
        return report
    ref = root / "reference"
    if (root / "events" / "events.parquet").exists():
        _mv(root / "events" / "events.parquet", ref / "events" / "events.parquet", [])
    if (root / "corporate_actions").exists():
        for f in (root / "corporate_actions").glob("*.parquet"):
            _mv(f, ref / "corporate_actions" / f.name, [])
    # provider facts leave the symbol table: they now live in each dataset's descriptor
    sym_f = root / "symbols.json"
    if sym_f.exists():
        symbols = json.loads(sym_f.read_text())
        for m in symbols.values():
            ex = dict(m.get("extra") or {})
            ex.pop("sources", None)
            ex.pop("price_adjustment", None)
            m["extra"] = ex
            m["provider"] = ""
            m["option_interval_minutes"] = None
        sym_f.write_text(json.dumps(symbols, indent=2, sort_keys=True))
    for f in (root / "manifests").glob("*.json") if (root / "manifests").exists() else []:
        f.unlink()
    (root / "catalog_cache.json").unlink(missing_ok=True)
    for d in LEGACY_DIRS:
        _prune(root / d)
    mig = root / "migrations"
    mig.mkdir(exist_ok=True)
    (mig / "layout2-to-3.json").write_text(json.dumps({**report, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    (root / "LAYOUT").write_text("3")
    return report
