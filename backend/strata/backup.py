"""Backup and restore of a Strata data directory.

A backup is a folder: a consistent copy of the application database (taken with SQLite's
online backup, so it is safe while the application is running), the stored market data,
the reports, and a manifest listing every file with its size and SHA-256. Credentials are
NOT part of a backup: they live in the operating system's credential store (or a
permission-restricted file) and stay there.

    strata backup  /Volumes/disk/strata-backups            # everything
    strata backup  ~/backups --no-market                   # database and reports only (small)
    strata restore /Volumes/disk/strata-backups/strata-backup-20261125-171500

`restore` verifies every file against the manifest before anything is touched, refuses to
write into a data directory that already holds data unless told to (`--force`), and keeps
the database it replaces beside the new one.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {"cache", "uploads", "tmp"}
SECRET_NAMES = {"secrets.json", ".env"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path, market: bool) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or p.is_symlink() or rel.parts[0] in SKIP_DIRS or p.name in SECRET_NAMES or p.name.startswith(".") and p.suffix in (".tmp", ".lock") or p.name == ".write.lock":
            continue
        if p.name.startswith("app.db"):
            continue  # the database is copied through SQLite, not as a file that may be mid-write
        if not market and rel.parts[0] == "market":
            continue
        out.append(p)
    return out


def backup(dest: str | Path, market: bool = True, data_dir: Path | None = None, progress=None) -> dict:
    from strata.config import ENGINE_VERSION, get_settings
    root = Path(data_dir or get_settings().data_dir)
    if not root.exists():
        raise ValueError(f"there is no data directory at {root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(dest).expanduser() / f"strata-backup-{stamp}"
    if root.resolve() == out.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError("the backup folder must be outside the data directory it is a backup of")
    out.mkdir(parents=True, exist_ok=False)
    files = _files(root, market)
    manifest: dict = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "engine_version": ENGINE_VERSION, "source": str(root), "market_data": market, "files": {}, "database": None,
                      "credentials": "not included: they stay in the credential store of the machine they were entered on"}
    db = root / "app.db"
    if db.exists():
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dst = sqlite3.connect(out / "app.db")
        try:
            src.backup(dst)  # a consistent snapshot even while the application is writing
            ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            dst.close()
            src.close()
        if ok != "ok":
            raise RuntimeError(f"the copied database failed its integrity check: {ok}")
        manifest["database"] = {"file": "app.db", "bytes": (out / "app.db").stat().st_size, "sha256": _sha(out / "app.db"), "integrity_check": ok}
    total = 0
    for i, p in enumerate(files):
        rel = p.relative_to(root)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / rel)
        size = p.stat().st_size
        total += size
        manifest["files"][rel.as_posix()] = {"bytes": size, "sha256": _sha(out / rel)}
        if progress and i % 200 == 0:
            progress(i / max(1, len(files)), f"{i:,} of {len(files):,} files")
    manifest["bytes"] = total + ((manifest["database"] or {}).get("bytes") or 0)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return {"path": str(out), "files": len(files), "bytes": manifest["bytes"], "database": bool(manifest["database"]), "market_data": market}


def verify(src: str | Path) -> dict:
    """Every file of a backup against its manifest. Nothing is written."""
    src = Path(src).expanduser()
    mf = src / "MANIFEST.json"
    if not mf.exists():
        raise ValueError(f"{src} is not a Strata backup (no MANIFEST.json)")
    m = json.loads(mf.read_text())
    bad, missing = [], []
    items = dict(m.get("files") or {})
    if m.get("database"):
        items[m["database"]["file"]] = m["database"]
    for rel, info in items.items():
        if rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
            bad.append(f"{rel}: a path outside the backup")  # a manifest is input too
            continue
        p = src / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != info["bytes"] or _sha(p) != info["sha256"]:
            bad.append(f"{rel}: contents differ from the manifest")
    extra = [p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file() and p.name != "MANIFEST.json" and p.relative_to(src).as_posix() not in items]
    if m.get("database") and not bad and not missing:
        con = sqlite3.connect(f"file:{src / m['database']['file']}?mode=ro", uri=True)
        try:
            chk = con.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            con.close()
        if chk != "ok":
            bad.append(f"app.db: integrity check says {chk}")
    return {"ok": not bad and not missing, "files": len(items), "damaged": bad[:50], "missing": missing[:50], "not_in_manifest": extra[:50], "created": m.get("created"), "engine_version": m.get("engine_version"), "market_data": m.get("market_data")}


def restore(src: str | Path, data_dir: Path | None = None, force: bool = False) -> dict:
    from strata.config import get_settings
    src = Path(src).expanduser()
    root = Path(data_dir or get_settings().data_dir)
    chk = verify(src)
    if not chk["ok"]:
        raise ValueError("the backup did not verify, so nothing was restored: " + "; ".join((chk["damaged"] + [f"{x}: missing" for x in chk["missing"]])[:5]))
    m = json.loads((src / "MANIFEST.json").read_text())
    has_data = root.exists() and any(p.name not in ("cache",) for p in root.iterdir())
    if has_data and not force:
        raise ValueError(f"{root} already holds data. Restoring would replace files in it: pass --force to do that (the current database is kept beside the restored one), or restore into an empty directory with --data-dir.")
    root.mkdir(parents=True, exist_ok=True)
    kept = None
    if m.get("database"):
        cur = root / "app.db"
        if cur.exists():
            kept = root / f"app.db.before-restore-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            con = sqlite3.connect(f"file:{cur}?mode=ro", uri=True)
            dst = sqlite3.connect(kept)
            try:
                con.backup(dst)
            finally:
                dst.close()
                con.close()
            for extra in (root / "app.db-wal", root / "app.db-shm"):
                extra.unlink(missing_ok=True)
        shutil.copy2(src / "app.db", cur)
    n = 0
    for rel in m.get("files") or {}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, root / rel)
        n += 1
    return {"restored_to": str(root), "files": n, "database": bool(m.get("database")), "previous_database_kept_as": str(kept) if kept else None, "market_data": m.get("market_data"),
            "note": "Credentials are not part of a backup. Enter provider keys again in Settings on this machine."}
