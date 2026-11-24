-- Strata application database (SQLite). Applied by: strata init
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
CREATE INDEX IF NOT EXISTS runs_kind ON runs(kind, created_at);
CREATE INDEX IF NOT EXISTS runs_cache ON runs(cache_key);
CREATE INDEX IF NOT EXISTS runs_strategy ON runs(strategy_id);
