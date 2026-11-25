"""SQLite application state (strategies, runs). Market data lives in Parquet."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from strata.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  config TEXT NOT NULL,
  notes TEXT DEFAULT '',
  tags TEXT DEFAULT '[]',
  trials INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  name TEXT DEFAULT '',
  strategy_id TEXT,
  config TEXT,
  config_hash TEXT,
  cache_key TEXT,
  dataset_versions TEXT,
  engine_version TEXT,
  synthetic INTEGER DEFAULT 0,
  progress REAL DEFAULT 0,
  message TEXT DEFAULT '',
  error TEXT,
  summary TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  state TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  kind TEXT NOT NULL,
  content TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  parent_id INTEGER,
  label TEXT,
  draft TEXT NOT NULL,
  runs TEXT DEFAULT '[]',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_session ON versions(session_id, id);
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT DEFAULT '',
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT NOT NULL,
  symbol TEXT NOT NULL,
  family TEXT DEFAULT '',
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  label TEXT DEFAULT '',
  run_id TEXT,
  start TEXT,
  end TEXT,
  metric TEXT,
  value REAL,
  n INTEGER,
  p_value REAL,
  shown INTEGER DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_scope ON trials(scope, symbol, id);
CREATE TABLE IF NOT EXISTS holdouts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT NOT NULL,
  symbol TEXT NOT NULL,
  start TEXT NOT NULL,
  end TEXT NOT NULL,
  status TEXT NOT NULL,
  uses TEXT DEFAULT '[]',
  reason TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS holdouts_scope ON holdouts(scope, symbol, id);
CREATE TABLE IF NOT EXISTS canvases (
  session_id TEXT PRIMARY KEY,
  layout TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collector_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  trigger TEXT DEFAULT 'schedule',
  report TEXT
);
CREATE INDEX IF NOT EXISTS runs_kind ON runs(kind, created_at);
CREATE INDEX IF NOT EXISTS runs_cache ON runs(cache_key);
CREATE INDEX IF NOT EXISTS runs_strategy ON runs(strategy_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    p = get_settings().db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    return con


def init_db() -> Path:
    with connect() as con:
        con.executescript(SCHEMA)
    return get_settings().db_path


JSON_COLS = {"config", "dataset_versions", "summary", "tags", "state", "content", "draft", "runs", "uses", "layout", "report"}


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if d[k]:
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


# ---------------- strategies
def list_strategies() -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM strategies ORDER BY updated_at DESC").fetchall()
    return [_row(r) for r in rows]


def get_strategy(sid: str) -> dict | None:
    with connect() as con:
        return _row(con.execute("SELECT * FROM strategies WHERE id=?", (sid,)).fetchone())


def save_strategy(config: dict, sid: str | None = None, notes: str = "", tags: list | None = None) -> dict:
    sid = sid or uuid.uuid4().hex[:12]
    ts = now()
    with connect() as con:
        cur = con.execute("SELECT id FROM strategies WHERE id=?", (sid,)).fetchone()
        if cur:
            con.execute("UPDATE strategies SET name=?, symbol=?, config=?, notes=?, tags=?, updated_at=? WHERE id=?",
                        (config["name"], config["symbol"], json.dumps(config), notes, json.dumps(tags or []), ts, sid))
        else:
            con.execute("INSERT INTO strategies (id, name, symbol, config, notes, tags, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                        (sid, config["name"], config["symbol"], json.dumps(config), notes, json.dumps(tags or []), ts, ts))
    return get_strategy(sid)


def delete_strategy(sid: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM strategies WHERE id=?", (sid,))


def add_trials(sid: str, n: int) -> None:
    with connect() as con:
        con.execute("UPDATE strategies SET trials = trials + ? WHERE id=?", (n, sid))


# ---------------- runs
def create_run(kind: str, name: str, config: dict | None, strategy_id: str | None = None, cache_key: str | None = None,
               config_hash: str | None = None, dataset_versions: dict | None = None, engine_version: str | None = None, synthetic: bool = False) -> str:
    rid = uuid.uuid4().hex[:12]
    with connect() as con:
        con.execute("INSERT INTO runs (id, kind, status, name, strategy_id, config, config_hash, cache_key, dataset_versions, engine_version, synthetic, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, kind, "queued", name, strategy_id, json.dumps(config) if config is not None else None, config_hash, cache_key,
                     json.dumps(dataset_versions or {}), engine_version, int(synthetic), now()))
    return rid


def update_run(rid: str, **fields) -> None:
    if not fields:
        return
    for k in list(fields):
        if k in JSON_COLS and fields[k] is not None and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), rid))


def get_run(rid: str) -> dict | None:
    with connect() as con:
        return _row(con.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone())


def list_runs(kind: str | None = None, strategy_id: str | None = None, limit: int = 200, status: str | None = None) -> list[dict]:
    q = "SELECT id, kind, status, name, strategy_id, config_hash, dataset_versions, engine_version, synthetic, progress, message, error, summary, created_at, started_at, finished_at FROM runs WHERE 1=1"
    args: list = []
    if kind:
        q += " AND kind IN (%s)" % ",".join("?" * len(kind.split(",")))
        args += kind.split(",")
    if strategy_id:
        q += " AND strategy_id=?"
        args.append(strategy_id)
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with connect() as con:
        return [_row(r) for r in con.execute(q, args).fetchall()]


def runs_targeting(target: str, kinds: tuple[str, ...] = ("stress", "challenge", "sensitivity"), limit: int = 400) -> list[dict]:
    """Analysis runs (stress ladders, reviews, maps) that refer to another run, newest first."""
    q = "SELECT * FROM runs WHERE kind IN (%s) ORDER BY created_at DESC LIMIT ?" % ",".join("?" * len(kinds))
    with connect() as con:
        rows = [_row(r) for r in con.execute(q, [*kinds, limit]).fetchall()]
    return [r for r in rows if isinstance(r.get("config"), dict) and r["config"].get("target") == target]


def find_cached(cache_key: str) -> dict | None:
    with connect() as con:
        return _row(con.execute("SELECT * FROM runs WHERE cache_key=? AND status='done' ORDER BY created_at DESC LIMIT 1", (cache_key,)).fetchone())


def delete_run(rid: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM runs WHERE id=?", (rid,))


def mark_interrupted() -> int:
    with connect() as con:
        cur = con.execute("UPDATE runs SET status='error', error='interrupted: server restarted while the job was running', finished_at=? WHERE status IN ('running','queued')", (now(),))
        return cur.rowcount


# ---------------- language-model usage
def add_usage(provider: str, model: str, purpose: str, input_tokens: int, output_tokens: int) -> None:
    with connect() as con:
        con.execute("INSERT INTO llm_usage (provider, model, purpose, input_tokens, output_tokens, created_at) VALUES (?,?,?,?,?,?)", (provider, model, purpose, int(input_tokens), int(output_tokens), now()))


def usage_summary(days: int = 30) -> dict:
    from datetime import timedelta
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with connect() as con:
        rows = [dict(r) for r in con.execute("SELECT provider, model, purpose, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, MAX(created_at) AS last "
                                             "FROM llm_usage WHERE created_at >= ? GROUP BY provider, model, purpose ORDER BY last DESC", (since,)).fetchall()]
    return {"days": days, "rows": rows, "calls": sum(r["calls"] for r in rows), "input_tokens": sum(r["input_tokens"] or 0 for r in rows), "output_tokens": sum(r["output_tokens"] or 0 for r in rows)}


# ---------------- AI research sessions
def create_session(title: str, state: dict | None = None) -> dict:
    sid = uuid.uuid4().hex[:12]
    t = now()
    with connect() as con:
        con.execute("INSERT INTO sessions (id, title, state, created_at, updated_at) VALUES (?,?,?,?,?)", (sid, title, json.dumps(state or {}), t, t))
    return get_session(sid)


def get_session(sid: str) -> dict | None:
    with connect() as con:
        return _row(con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone())


def list_sessions(limit: int = 100) -> list[dict]:
    with connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()]


def update_session(sid: str, title: str | None = None, state: dict | None = None) -> None:
    with connect() as con:
        if title is not None:
            con.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, now(), sid))
        if state is not None:
            con.execute("UPDATE sessions SET state=?, updated_at=? WHERE id=?", (json.dumps(state), now(), sid))


def delete_session(sid: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        con.execute("DELETE FROM versions WHERE session_id=?", (sid,))
        con.execute("DELETE FROM sessions WHERE id=?", (sid,))


def add_message(sid: str, role: str, kind: str, content: dict) -> dict:
    with connect() as con:
        cur = con.execute("INSERT INTO messages (session_id, role, kind, content, created_at) VALUES (?,?,?,?,?)", (sid, role, kind, json.dumps(content, default=str), now()))
        con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))
        return _row(con.execute("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)).fetchone())


def update_message(mid: int, content: dict) -> None:
    with connect() as con:
        con.execute("UPDATE messages SET content=? WHERE id=?", (json.dumps(content, default=str), mid))


def get_message(mid: int) -> dict | None:
    with connect() as con:
        return _row(con.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone())


def list_messages(sid: str) -> list[dict]:
    with connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,)).fetchall()]


def add_version(sid: str, parent_id: int | None, label: str, draft: dict) -> dict:
    with connect() as con:
        cur = con.execute("INSERT INTO versions (session_id, parent_id, label, draft, runs, created_at) VALUES (?,?,?,?,?,?)", (sid, parent_id, label, json.dumps(draft, default=str), "[]", now()))
        return _row(con.execute("SELECT * FROM versions WHERE id=?", (cur.lastrowid,)).fetchone())


def get_version(vid: int) -> dict | None:
    with connect() as con:
        r = _row(con.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone())
    if r is not None and not isinstance(r.get("runs"), list):
        r["runs"] = []
    return r


def list_versions(sid: str) -> list[dict]:
    with connect() as con:
        out = [_row(r) for r in con.execute("SELECT * FROM versions WHERE session_id=? ORDER BY id", (sid,)).fetchall()]
    for r in out:
        if not isinstance(r.get("runs"), list):
            r["runs"] = []
    return out


def update_version(vid: int, draft: dict | None = None, label: str | None = None) -> None:
    with connect() as con:
        if draft is not None:
            con.execute("UPDATE versions SET draft=? WHERE id=?", (json.dumps(draft, default=str), vid))
        if label is not None:
            con.execute("UPDATE versions SET label=? WHERE id=?", (label, vid))


def set_version_runs(vid: int, runs: list) -> None:
    with connect() as con:
        con.execute("UPDATE versions SET runs=? WHERE id=?", (json.dumps(runs, default=str), vid))
