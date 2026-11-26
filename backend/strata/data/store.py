"""Columnar market-data store, layout version 3 (Parquet files, queried with Polars/DuckDB).

Every dataset is partitioned by provider, symbol, data type, observation type, resolution
and dataset version, so two providers (or a quote dataset and a trade dataset) for the same
security and session never overwrite or relabel each other:

<data_dir>/market/
  LAYOUT                                   "3"
  symbols.json                             contract specifications per symbol (no provider facts)
  datasets/<provider>/<SYMBOL>/<dtype>/<obs>/<res>/v<N>/
      dataset.json                         descriptor: provider, licence, price adjustment, limitations ...
      data/daily.parquet                   underlying, res 1d
      data/<YYYY>.parquet                  underlying intraday bars, ts = bar START (UTC)
      data/<YYYY>/<YYYY-MM-DD>.parquet     options, one file per session
      sessions.json                        options: per-session state (complete | partial) and row counts
      index.parquet                        option trade bars: one row per fully fetched (session, contract)
      contracts.parquet                    option contract definitions with the provider's own identifiers
      derived/iv.parquet                   ATM IV history derived from this option-quote dataset
      provenance.json                      per-file provenance (source request, fetch time)
      manifests/<seq>-<hash>.json          immutable manifests: sha256, bytes and rows of every file
      HEAD                                 name of the current manifest
  reference/events/events.parquet          scheduled economic events
  reference/corporate_actions/<SYM>.parquet

dtype is "underlying" or "options".  obs is the kind of observation:
  underlying: "trade" (bars built from trades) or "index" (published index levels)
  options:    "quote" (bid/ask snapshots) or "trade" (trade prints - never treated as quotes)
res is "1d", "1m" ... "60m" for the underlying; "EOD" or "<n>m" for options.

Readers that are not given a provider resolve one explicitly and deterministically (see
`provider_rank` and `option_dataset`); the engine records which dataset it used.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import os
import re
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from strata.config import get_settings

LAYOUT_VERSION = "3"

OPTION_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "expiration": pl.Date,
    "strike": pl.Float64,
    "right": pl.String,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Int32,
    "ask_size": pl.Int32,
    "last": pl.Float64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
    "iv": pl.Float32,
    "delta": pl.Float32,
    "gamma": pl.Float32,
    "theta": pl.Float32,
    "vega": pl.Float32,
    "underlying_price": pl.Float64,
    "quote_ts": pl.Datetime("us", "UTC"),
}
OPTION_REQUIRED = ["ts", "expiration", "strike", "right"]  # plus bid+ask (quotes) or last (trade prints)
INTRADAY_RES = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600}

# Option TRADE bars: one row per contract and bar in which at least one trade printed.
# There is deliberately no bid/ask column: a trade bar is never a quote.
TRADE_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),  # bar START
    "occ": pl.String,  # the provider's original contract identifier (OCC symbol)
    "expiration": pl.Date,
    "strike": pl.Float64,
    "right": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "trade_count": pl.Int32,
    "vwap": pl.Float64,
}
INDEX_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "occ": pl.String,
    "expiration": pl.Date,
    "strike": pl.Float64,
    "right": pl.String,
    "bars": pl.Int32,  # bars stored for this contract on this session (0 = the provider reported no trades)
    "status": pl.String,  # ok | empty
    "pages": pl.Int32,  # provider pages consumed for the request that covered this row
    "fetched_at": pl.Datetime("us", "UTC"),
    "window": pl.String,  # the request window that was fully consumed, e.g. 09:30-16:00
}
CONTRACT_SCHEMA: dict[str, pl.DataType] = {
    "occ": pl.String,
    "root": pl.String,
    "underlying": pl.String,
    "expiration": pl.Date,
    "expiration_ts": pl.Datetime("us", "UTC"),  # the session close on the expiration date
    "strike": pl.Float64,
    "right": pl.String,
    "style": pl.String,
    "multiplier": pl.Float64,
    "adjusted": pl.Boolean,  # non-standard deliverable (root carries an adjustment suffix)
    "provider_id": pl.String,
    "status_at_fetch": pl.String,
    "first_trade": pl.Date,
    "source": pl.String,
    "fetched_at": pl.Datetime("us", "UTC"),
}

BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
DAILY_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
}
EVENT_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date, "time": pl.String, "event_type": pl.String, "description": pl.String, "source": pl.String,
}
CORP_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "ex_date": pl.Date, "action": pl.String, "ratio": pl.Float64, "amount": pl.Float64,
}

# Which provider wins when several hold the same session at the same resolution. The engine
# reports the provider it used; nothing is blended inside one session.
DAILY_PRIORITY = ["cboe", "yahoo", "alpaca", "polygon", "databento", "thetadata", "orats", "github_eod", "dolthub", "import"]
INTRADAY_PRIORITY = ["alpaca", "polygon", "databento", "thetadata", "yahoo", "import"]


def provider_rank(provider: str, res: str = "1d") -> tuple[int, str]:
    order = DAILY_PRIORITY if res in ("1d", "EOD") else INTRADAY_PRIORITY
    if provider == "synthetic":
        return (99, provider)
    return (order.index(provider) if provider in order else 50, provider)


@dataclass(frozen=True)
class DatasetKey:
    provider: str
    symbol: str
    dtype: str  # underlying | options
    obs: str  # trade | index | quote
    res: str
    version: int = 1

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.symbol}/{self.dtype}/{self.obs}/{self.res}/v{self.version}"

    @classmethod
    def parse(cls, s: str) -> "DatasetKey":
        p = s.strip("/").split("/")
        if len(p) != 6 or not p[5].startswith("v"):
            raise ValueError(f"not a dataset id: {s!r}")
        return cls(p[0], p[1], p[2], p[3], p[4], int(p[5][1:]))


@dataclass
class SymbolMeta:
    symbol: str
    name: str = ""
    kind: str = "etf"  # etf | equity | index | volatility_index
    exercise: str = "american"  # american | european
    settlement: str = "physical"  # physical | cash
    settlement_time: str = "PM"  # PM | AM | AM_MONTHLY
    multiplier: float = 100.0
    vol_index: str | None = None
    provider: str = ""  # legacy (layout 2); provider facts now live in each dataset descriptor
    synthetic: bool = False  # a fictional sandbox symbol
    has_options: bool = True
    option_interval_minutes: int | None = None  # legacy (layout 2)
    description: str = ""
    extra: dict = field(default_factory=dict)


_META_FIELDS = {f.name for f in fields(SymbolMeta)}

KNOWN_SPECS = {
    "SPX": dict(kind="index", exercise="european", settlement="cash", settlement_time="AM_MONTHLY", vol_index="VIX"),
    "SPXW": dict(kind="index", exercise="european", settlement="cash", settlement_time="PM", vol_index="VIX"),
    "XSP": dict(kind="index", exercise="european", settlement="cash", settlement_time="PM", vol_index="VIX"),
    "NDX": dict(kind="index", exercise="european", settlement="cash", settlement_time="AM_MONTHLY", vol_index="VXN"),
    "RUT": dict(kind="index", exercise="european", settlement="cash", settlement_time="AM_MONTHLY", vol_index="RVX"),
    "VIX": dict(kind="volatility_index", exercise="european", settlement="cash", has_options=False),
    "SPY": dict(kind="etf", vol_index="VIX", name="SPDR S&P 500 ETF"),
    "QQQ": dict(kind="etf", vol_index="VIX", name="Invesco QQQ Trust"),
    "IWM": dict(kind="etf", vol_index="VIX", name="iShares Russell 2000 ETF"),
    "DIA": dict(kind="etf", vol_index="VIX", name="SPDR Dow Jones ETF"),
    "AAPL": dict(kind="equity", vol_index="VIX", name="Apple"),
    "NVDA": dict(kind="equity", vol_index="VIX", name="NVIDIA"),
    "TSLA": dict(kind="equity", vol_index="VIX", name="Tesla"),
    "MSFT": dict(kind="equity", vol_index="VIX", name="Microsoft"),
    "AMZN": dict(kind="equity", vol_index="VIX", name="Amazon"),
    "META": dict(kind="equity", vol_index="VIX", name="Meta Platforms"),
    "GOOGL": dict(kind="equity", vol_index="VIX", name="Alphabet"),
    "AMD": dict(kind="equity", vol_index="VIX", name="AMD"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _et_date(col: str = "ts") -> pl.Expr:
    return pl.col(col).dt.convert_time_zone("America/New_York").dt.date()


def _tmp_of(p: Path) -> Path:
    return p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def _atomic_parquet(df, p: Path, **kw) -> None:
    """Write beside the destination and rename: a reader never sees half a file, and an interrupted write leaves the previous one intact."""
    tmp = _tmp_of(Path(p))
    try:
        df.write_parquet(tmp, **kw)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_text(p: Path, text: str) -> None:
    tmp = _tmp_of(Path(p))
    try:
        tmp.write_text(text)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


SAFE_PART = re.compile(r"[A-Za-z0-9^][A-Za-z0-9_.^=-]{0,47}")


def safe_name(name) -> bool:
    """True when a provider, symbol or dataset label can be used as a folder or file name: it comes from requests, imported
    files and language models, so it is checked wherever it is about to become part of a path."""
    return isinstance(name, str) and bool(SAFE_PART.fullmatch(name)) and ".." not in name


def _ref_name(sym) -> str:
    if not safe_name(sym):
        raise ValueError(f"{sym!r} is not a valid symbol name (letters, digits and . _ - ^ = only)")
    return sym
_locks: dict[str, "_WriteLock"] = {}


class _WriteLock:
    """One writer at a time, across threads AND processes. The application and an always-on collector (a second
    process, often a second container) share one data directory: a thread lock alone lets the two interleave a
    data file with its session index. Re-entrant; the file lock is advisory (flock) and is skipped where it is not
    available, leaving the thread lock."""

    def __init__(self):
        self._t = threading.RLock()
        self._depth = 0
        self._fh = None
        self.path: Path | None = None

    def __enter__(self):
        self._t.acquire()
        self._depth += 1
        if self._depth == 1 and self.path is not None:
            try:
                import fcntl
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a+")
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                self._fh = None
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth == 0 and self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            self._fh.close()
            self._fh = None
        self._t.release()
        return False


class MarketStore:
    _lock = _WriteLock()

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_settings().market_dir
        self._lock = _locks.setdefault(str(self.root.resolve()) if self.root.exists() else str(self.root), _WriteLock())
        self._lock.path = self.root / ".write.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self._opt_cache: dict = {}
        self._sess_cache: dict = {}
        self._sess_stamp: dict = {}
        if not (self.root / "LAYOUT").exists():
            from strata.data.migrate import migrate_layout
            with self._lock:
                migrate_layout(self.root)

    # ---------- paths ----------
    def _p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def dataset_dir(self, key: DatasetKey) -> Path:
        for part, what in ((key.provider, "provider"), (key.symbol, "symbol"), (key.dtype, "data type"), (key.obs, "observation type"), (key.res, "resolution")):
            if not safe_name(part):
                # these become folder names: a provider or symbol taken from a request must not be able to name a path
                raise ValueError(f"{part!r} is not a valid {what} name (letters, digits and . _ - ^ = only)")
        return self._p("datasets", key.provider, key.symbol, key.dtype, key.obs, key.res, f"v{int(key.version)}")

    def option_path(self, sym: str, d: date, obs: str = "quote", provider: str | None = None, dataset: DatasetKey | None = None) -> Path | None:
        ds = dataset or self.option_dataset(sym, obs, provider)
        return None if ds is None else self.dataset_dir(ds) / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"

    def iv_path(self, sym: str, dataset: DatasetKey | None = None) -> Path | None:
        ds = dataset or self.option_dataset(sym, "quote")
        return None if ds is None else self.dataset_dir(ds) / "derived" / "iv.parquet"

    # ---------- dataset registry ----------
    def ensure_dataset(self, provider: str, sym: str, dtype: str, obs: str, res: str, version: int = 1, **desc) -> DatasetKey:
        """Create the dataset directory and descriptor if needed. Descriptor fields given
        here (licence, url, adjustment, synthetic, limitations, format, label) are merged in."""
        key = DatasetKey(provider, sym, dtype, obs, res, version)
        d = self.dataset_dir(key)
        f = d / "dataset.json"
        with self._lock:
            cur = json.loads(f.read_text()) if f.exists() else {}
            new = {"id": key.id, "provider": provider, "symbol": sym, "dtype": dtype, "obs": obs, "resolution": res, "version": version,
                   "created": cur.get("created") or _now(), "license": "", "url": "", "adjustment": None, "synthetic": False, "format": None,
                   "limitations": [], "label": "", **{k: v for k, v in cur.items() if k not in ("id",)}}
            for k, v in desc.items():
                if v is not None:
                    new[k] = v
            if new != cur:
                d.mkdir(parents=True, exist_ok=True)
                _atomic_text(f, json.dumps(new, indent=2, sort_keys=True))
                self._opt_cache.clear()
        return key

    def descriptor(self, key: DatasetKey) -> dict:
        f = self.dataset_dir(key) / "dataset.json"
        return json.loads(f.read_text()) if f.exists() else {"id": key.id, "provider": key.provider, "symbol": key.symbol, "dtype": key.dtype, "obs": key.obs, "resolution": key.res, "version": key.version}

    def datasets(self, sym: str | None = None, dtype: str | None = None, obs: str | None = None, res: str | None = None, provider: str | None = None,
                 all_versions: bool = False) -> list[DatasetKey]:
        base = self._p("datasets")
        if not base.exists():
            return []
        if any(x is not None and not safe_name(x) for x in (provider, sym, dtype, obs, res)):
            return []  # such a name cannot be a folder here, and it must not be read as a pattern or a path either
        pat = "/".join([provider or "*", sym or "*", dtype or "*", obs or "*", res or "*", "v*"])
        found: dict[tuple, DatasetKey] = {}
        for p in base.glob(pat):
            if not p.is_dir() or not p.name[1:].isdigit():
                continue
            rel = p.relative_to(base).parts
            k = DatasetKey(rel[0], rel[1], rel[2], rel[3], rel[4], int(rel[5][1:]))
            ident = (k.provider, k.symbol, k.dtype, k.obs, k.res) + ((k.version,) if all_versions else ())
            if ident not in found or found[ident].version < k.version:
                found[ident] = k
        return sorted(found.values(), key=lambda k: (k.symbol, k.dtype, k.obs, provider_rank(k.provider, k.res), k.res, k.version))

    def remove_dataset(self, key: DatasetKey) -> None:
        with self._lock:
            d = self.dataset_dir(key)
            if d.exists():
                shutil.rmtree(d)
            self._opt_cache.clear()
            self._sess_cache.clear()

    # ---------- symbol metadata ----------
    def symbols(self) -> dict[str, SymbolMeta]:
        f = self._p("symbols.json")
        if not f.exists():
            return {}
        raw = json.loads(f.read_text())
        return {k: SymbolMeta(**{a: b for a, b in v.items() if a in _META_FIELDS}) for k, v in raw.items()}

    def symbol_meta(self, sym: str) -> SymbolMeta:
        m = self.symbols().get(sym)
        if m is None:
            spec = KNOWN_SPECS.get(sym, {})
            m = SymbolMeta(symbol=sym, **spec)
        return m

    def upsert_symbol(self, meta: SymbolMeta) -> None:
        with self._lock:
            cur = {k: asdict(v) for k, v in self.symbols().items()}
            cur[meta.symbol] = asdict(meta)
            f = self._p("symbols.json")
            _atomic_text(f, json.dumps(cur, indent=2, sort_keys=True, default=str))

    def stored_symbols(self) -> set[str]:
        return set(self.symbols()) | {k.symbol for k in self.datasets()}

    def remove_symbol(self, sym: str) -> None:
        with self._lock:
            for k in self.datasets(sym, all_versions=True):
                self.remove_dataset(k)
            p = self._p("reference", "corporate_actions", f"{sym}.parquet") if safe_name(sym) else None
            if p is not None and p.exists():
                p.unlink()
            cur = {k: asdict(v) for k, v in self.symbols().items() if k != sym}
            _atomic_text(self._p("symbols.json"), json.dumps(cur, indent=2, sort_keys=True, default=str))

    def is_synthetic(self, sym: str) -> bool:
        """True when the symbol is a sandbox symbol or any of its datasets is simulated."""
        if self.symbol_meta(sym).synthetic:
            return True
        return any(self.descriptor(k).get("synthetic") for k in self.datasets(sym))

    # ---------- provenance ----------
    def _dirty(self, key: DatasetKey) -> None:
        """Mark a dataset as changed since its last manifest. Its content version is part of
        every cache key, so a writer that forgets to commit can never leave a stale version
        behind: the next reader sees the marker and commits first."""
        (self.dataset_dir(key) / "DIRTY").touch()

    def _note(self, key: DatasetKey, path: Path, **info) -> None:
        self._dirty(key)
        f = self.dataset_dir(key) / "provenance.json"
        cur = json.loads(f.read_text()) if f.exists() else {}
        rel = str(path.relative_to(self.dataset_dir(key)))
        cur[rel] = {"provider": key.provider, "written": _now(), **{k: v for k, v in info.items() if v is not None}}
        _atomic_text(f, json.dumps(cur, sort_keys=True))

    # ---------- writing: underlying ----------
    @staticmethod
    def _conform(df: pl.DataFrame, schema: dict[str, pl.DataType], required: list[str]) -> pl.DataFrame:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        cols = []
        for c, t in schema.items():
            if c in df.columns:
                cols.append(pl.col(c).cast(t, strict=False).alias(c))
            else:
                cols.append(pl.lit(None, dtype=t).alias(c))
        return df.select(cols)

    def write_daily(self, sym: str, df: pl.DataFrame, provider: str = "import", obs: str = "trade", source: str | None = None, **desc) -> Path:
        df = self._conform(df, DAILY_SCHEMA, ["date", "open", "high", "low", "close"])
        key = self.ensure_dataset(provider, sym, "underlying", obs, "1d", **desc)
        p = self.dataset_dir(key) / "data" / "daily.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if p.exists():
                df = pl.concat([pl.read_parquet(p), df]).unique(subset=["date"], keep="last")
            _atomic_parquet(df.sort("date"), p, compression="zstd")
            self._note(key, p, source=source, rows=df.height)
        return p

    def write_minutes(self, sym: str, df: pl.DataFrame, provider: str = "import", **kw) -> list[Path]:
        return self.write_bars(sym, "1m", df, provider, paths=True, **kw)

    def write_bars(self, sym: str, res: str, df: pl.DataFrame, provider: str = "import", obs: str = "trade", source: str | None = None, paths: bool = False, **desc):
        """Intraday bars at resolution `res` (bar START time). Merges/dedupes by ts."""
        if res not in INTRADAY_RES:
            raise ValueError(f"unsupported resolution {res}")
        df = self._conform(df, BAR_SCHEMA, ["ts", "open", "high", "low", "close"]).drop_nulls(["open", "high", "low", "close"])
        key = self.ensure_dataset(provider, sym, "underlying", obs, res, **desc)
        n, out = 0, []
        with self._lock:
            for (yr,), part in df.with_columns(pl.col("ts").dt.year().alias("_y")).partition_by("_y", as_dict=True).items():
                p = self.dataset_dir(key) / "data" / f"{int(yr)}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                part = part.drop("_y")
                if p.exists():
                    part = pl.concat([pl.read_parquet(p), part]).unique(subset=["ts"], keep="last")
                _atomic_parquet(part.sort("ts"), p, compression="zstd")
                self._note(key, p, source=source, rows=part.height)
                n += part.height
                out.append(p)
        return out if paths else n

    # ---------- reading: underlying ----------
    def _underlying_sets(self, sym: str, res: str, provider: str | None = None) -> list[DatasetKey]:
        ks = [k for k in self.datasets(sym, "underlying", res=res) if provider is None or k.provider == provider]
        return sorted(ks, key=lambda k: provider_rank(k.provider, res))

    def read_daily(self, sym: str, provider: str | None = None, with_provider: bool = False) -> pl.DataFrame:
        """Daily bars. With several providers, each date comes from the highest-ranked one."""
        frames = []
        for rank, k in enumerate(self._underlying_sets(sym, "1d", provider)):
            p = self.dataset_dir(k) / "data" / "daily.parquet"
            if p.exists():
                frames.append(pl.read_parquet(p).with_columns(pl.lit(rank, dtype=pl.Int32).alias("_r"), pl.lit(k.provider).alias("provider")))
        if frames:
            df = pl.concat(frames).sort(["date", "_r"]).unique(subset=["date"], keep="first", maintain_order=True).drop("_r").sort("date")
            return df if with_provider else df.drop("provider")
        m = self.read_bars(sym, "1m")
        if m.is_empty():
            return pl.DataFrame(schema=DAILY_SCHEMA)
        return minutes_to_daily(m)

    def has_daily(self, sym: str) -> bool:
        return any((self.dataset_dir(k) / "data" / "daily.parquet").exists() for k in self.datasets(sym, "underlying", res="1d"))

    def read_bars(self, sym: str, res: str, start: date | None = None, end: date | None = None, provider: str | None = None, with_provider: bool = False) -> pl.DataFrame:
        """Intraday bars at one resolution. A session's bars always come from a single
        provider: the highest-ranked one that stores that session."""
        frames, taken = [], None
        for k in self._underlying_sets(sym, res, provider):
            d = self.dataset_dir(k) / "data"
            files = sorted(f for f in d.glob("*.parquet") if f.stem.isdigit()) if d.exists() else []
            files = [f for f in files if (not start or int(f.stem) >= start.year) and (not end or int(f.stem) <= end.year)]
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files]).with_columns(_et_date().alias("_d"))
            if start:
                df = df.filter(pl.col("_d") >= start)
            if end:
                df = df.filter(pl.col("_d") <= end)
            if taken is not None:
                df = df.filter(~pl.col("_d").is_in(taken.implode()))
            if df.is_empty():
                continue
            days = df["_d"].unique()
            taken = days if taken is None else pl.concat([taken, days])
            frames.append(df.drop("_d").with_columns(pl.lit(k.provider).alias("provider")))
        if not frames:
            return pl.DataFrame(schema={**BAR_SCHEMA, **({"provider": pl.String} if with_provider else {})})
        out = pl.concat(frames).sort("ts")
        return out if with_provider else out.drop("provider")

    def read_minutes(self, sym: str, start: date | None = None, end: date | None = None, provider: str | None = None) -> pl.DataFrame:
        return self.read_bars(sym, "1m", start, end, provider)

    def bar_resolutions(self, sym: str) -> list[str]:
        have = {k.res for k in self.datasets(sym, "underlying") if k.res in INTRADAY_RES and any((self.dataset_dir(k) / "data").glob("*.parquet"))}
        return [r for r in ("1m", "5m", "15m", "30m", "60m") if r in have]

    def read_best_bars(self, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        """Finest stored bars per session: 1m > 5m > 15m > 30m > 60m > daily. Returns
        ts (bar start), end (bar end), OHLCV, res (seconds) and provider. A session only
        covered by a daily bar becomes one bar spanning the whole session."""
        from strata.core import calendar as cal
        frames, taken = [], None
        for res in self.bar_resolutions(sym):
            b = self.read_bars(sym, res, start, end, with_provider=True)
            if b.is_empty():
                continue
            b = b.with_columns(_et_date().alias("_d"))
            if taken is not None:
                b = b.filter(~pl.col("_d").is_in(taken.implode()))
            if b.is_empty():
                continue
            days = b["_d"].unique()
            prev_taken = taken
            bounds = {d: cal.session_bounds(d) for d in days.to_list()}
            cmap = pl.DataFrame({"_d": list(bounds), "_open": [v[0] for v in bounds.values()], "_close": [v[1] for v in bounds.values()]}).with_columns(
                pl.col("_open").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")), pl.col("_close").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
            secs = INTRADAY_RES[res]
            b = b.join(cmap, on="_d").with_columns(pl.min_horizontal(pl.col("ts") + pl.duration(seconds=secs), pl.col("_close")).alias("end"), pl.lit(secs, dtype=pl.Int32).alias("res"))
            b = b.filter(pl.col("ts") < pl.col("_close"))
            # A session's bars at this resolution are used only if they span the session (start at the
            # open and reach the close, within 15 minutes). A fragment - e.g. a single bar - would give a
            # wrong session high, low and close, so such a session falls through to coarser bars.
            span = b.group_by("_d").agg(pl.col("ts").min().alias("_f"), pl.col("end").max().alias("_l"), pl.col("_open").first(), pl.col("_close").first())
            whole = span.filter((pl.col("_f") <= pl.col("_open") + pl.duration(minutes=15)) & (pl.col("_l") >= pl.col("_close") - pl.duration(minutes=15)))["_d"]
            b = b.filter(pl.col("_d").is_in(whole.implode())).drop("_open", "_close")
            if b.is_empty():
                continue
            taken = whole if prev_taken is None else pl.concat([prev_taken, whole])
            frames.append(b)
        dd = self.read_daily(sym, with_provider=True) if self.has_daily(sym) else pl.DataFrame(schema={**DAILY_SCHEMA, "provider": pl.String})
        if not dd.is_empty():
            if start:
                dd = dd.filter(pl.col("date") >= start)
            if end:
                dd = dd.filter(pl.col("date") <= end)
            if taken is not None:
                dd = dd.filter(~pl.col("date").is_in(taken.implode()))
            dd = dd.drop_nulls(["open", "high", "low", "close"])
            if not dd.is_empty():
                bounds = [cal.session_bounds(x) for x in dd["date"].to_list()]
                frames.append(dd.with_columns(
                    pl.Series("ts", [b[0] for b in bounds]).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                    pl.Series("end", [b[1] for b in bounds]).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                    pl.lit(86400, dtype=pl.Int32).alias("res"), pl.col("date").alias("_d")).drop("date"))
        cols = ["ts", "end", "open", "high", "low", "close", "volume", "res", "provider"]
        if not frames:
            return pl.DataFrame(schema={**BAR_SCHEMA, "end": pl.Datetime("us", "UTC"), "res": pl.Int32, "provider": pl.String})
        return pl.concat([f.select([pl.col(c).cast(pl.Float64) if c in ("open", "high", "low", "close", "volume") else pl.col(c) for c in cols]) for f in frames]).sort("ts")

    def official_closes(self, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame | None:
        """Daily closes (date, close) from the daily record, or None when no daily file is stored."""
        if not self.has_daily(sym):
            return None
        d = self.read_daily(sym).select("date", "close").drop_nulls().sort("date")
        if start:
            d = d.filter(pl.col("date") >= start)
        if end:
            d = d.filter(pl.col("date") <= end)
        return d

    def price_adjustment(self, sym: str) -> str | None:
        """"split" when the stored underlying prices are split-adjusted to today (a property
        of the provider's dataset, not of the symbol)."""
        adj = {self.descriptor(k).get("adjustment") for k in self.datasets(sym, "underlying")}
        return "split" if "split" in adj else None

    def split_factor(self, sym: str, d: date) -> float:
        """Product of split ratios with ex-date after d (vendor prices are split-adjusted
        to today; multiply by this to get as-traded prices on session d)."""
        ca = self.read_corporate_actions(sym)
        f = 1.0
        for r in ca.iter_rows(named=True):
            if r["action"] == "split" and r["ratio"] and r["ex_date"] > d:
                f *= float(r["ratio"])
        return f

    # ---------- options: dataset choice ----------
    def option_dataset(self, sym: str, obs: str = "quote", provider: str | None = None, res: str | None = None) -> DatasetKey | None:
        """The option dataset a reader uses when it does not name one: with several
        candidates, real data beats simulated data, then the one holding the most sessions
        wins (ties by provider name). The choice is explicit and reported, never a blend."""
        ck = (sym, obs, provider, res)
        if ck in self._opt_cache:
            return self._opt_cache[ck]
        cands = [k for k in self.datasets(sym, "options", obs, res) if provider is None or k.provider == provider]
        best = None
        if cands:
            def score(k: DatasetKey):
                d = self.dataset_dir(k) / "data"
                n = sum(1 for _ in d.glob("*/*.parquet")) if d.exists() else 0
                return (0 if self.descriptor(k).get("synthetic") else 1, n, -provider_rank(k.provider, k.res)[0], k.provider)
            best = max(cands, key=score)
        self._opt_cache[ck] = best
        return best

    def _sessions(self, key: DatasetKey) -> dict:
        """The session index of a dataset. It is re-read whenever the file on disk is newer than the copy in memory: another
        process (the collector beside the application) may have written sessions since, and saving a stale copy over the file
        would erase them."""
        f = self.dataset_dir(key) / "sessions.json"
        try:
            stamp = f.stat().st_mtime_ns
        except OSError:
            stamp = None
        if key.id not in self._sess_cache or self._sess_stamp.get(key.id) != stamp:
            self._sess_cache[key.id] = json.loads(f.read_text()) if stamp is not None else {}
            self._sess_stamp[key.id] = stamp
        return self._sess_cache[key.id]

    def _save_sessions(self, key: DatasetKey) -> None:
        f = self.dataset_dir(key) / "sessions.json"
        _atomic_text(f, json.dumps(self._sess_cache.get(key.id, {}), sort_keys=True))
        self._sess_stamp[key.id] = f.stat().st_mtime_ns

    def session_states(self, key: DatasetKey) -> dict:
        return dict(self._sessions(key))

    def mark_sessions(self, key: DatasetKey, entries: dict) -> None:
        """Record per-session state ({iso date: {"status": "complete" | "partial", ...}}).
        Ingesters call this only after the request covering a session was consumed to the end."""
        with self._lock:
            st = self._sessions(key)
            for d, v in entries.items():
                st[d if isinstance(d, str) else d.isoformat()] = {"written": _now(), **v}
            self.dataset_dir(key).mkdir(parents=True, exist_ok=True)
            self._save_sessions(key)

    def complete_sessions(self, key: DatasetKey) -> set[date]:
        return {date.fromisoformat(d) for d, v in self._sessions(key).items() if v.get("status") == "complete"}

    # ---------- options: quote snapshots ----------
    def write_option_day(self, sym: str, d: date, df: pl.DataFrame, provider: str = "import", res: str | None = None, status: str = "complete",
                         source: str | None = None, expected_rows: int | None = None, session_info: dict | None = None, **desc) -> Path:
        """One session of option snapshots. Rows with bid+ask are QUOTES; rows that only
        carry `last` are trade prints and go to a separate (obs=trade) dataset - a session
        is never marked complete unless the caller says so."""
        has_quotes = "bid" in df.columns and "ask" in df.columns
        if not (has_quotes or "last" in df.columns):
            raise ValueError("option data needs bid+ask quotes or last trade prices")
        if status not in ("complete", "partial"):
            raise ValueError("status must be complete or partial")
        obs = "quote" if has_quotes else "trade"
        df = self._conform(df, OPTION_SCHEMA, OPTION_REQUIRED)
        df = df.with_columns(pl.col("right").str.to_uppercase().str.slice(0, 1))
        if res is None:
            cur = [k for k in self.datasets(sym, "options", obs, provider=provider)]
            if cur:
                res = cur[0].res
            else:
                u = df["ts"].unique().sort()
                res = "EOD" if u.len() <= 1 else f"{max(1, int((u.diff().drop_nulls().dt.total_seconds()).median() // 60))}m"
        key = self.ensure_dataset(provider, sym, "options", obs, res, format="snapshots", **desc)
        p = self.dataset_dir(key) / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            _atomic_parquet(df.sort(["ts", "expiration", "right", "strike"]), p, compression="zstd", statistics=True)
            st = self._sessions(key)
            st[d.isoformat()] = {"status": status, "rows": df.height, "snapshots": int(df["ts"].n_unique()), "written": _now(),
                                 **({"expected_rows": expected_rows} if expected_rows is not None else {}), **({"source": source} if source else {}), **(session_info or {})}
            self._save_sessions(key)
            self._note(key, p, source=source, rows=df.height, status=status)
            self._opt_cache.clear()
        return p

    def option_days(self, sym: str, obs: str = "quote", provider: str | None = None, dataset: DatasetKey | None = None, complete_only: bool = False) -> list[date]:
        ds = dataset or self.option_dataset(sym, obs, provider)
        if ds is None:
            return []
        d = self.dataset_dir(ds) / "data"
        days = sorted(date.fromisoformat(f.stem) for f in d.glob("*/*.parquet")) if d.exists() else []
        if complete_only:
            st = self._sessions(ds)
            days = [x for x in days if (st.get(x.isoformat()) or {}).get("status") == "complete"]
        return days

    def read_option_day(self, sym: str, d: date, obs: str = "quote", provider: str | None = None, dataset: DatasetKey | None = None) -> pl.DataFrame | None:
        p = self.option_path(sym, d, obs, provider, dataset)
        if p is None or not p.exists():
            return None
        return pl.read_parquet(p)

    # ---------- options: trade bars ----------
    def write_option_bars(self, sym: str, provider: str, bars: pl.DataFrame, index_rows: list[dict], contracts: pl.DataFrame | None = None,
                          source: str | None = None, **desc) -> dict:
        """Store per-contract option TRADE bars and record which (session, contract) pairs
        were fetched to completion. `index_rows` lists every pair whose request was fully
        consumed - including pairs for which the provider reported no trades (status
        "empty"). A pair that is absent from the index has NOT been downloaded, whatever
        else is stored for that session."""
        key = self.ensure_dataset(provider, sym, "options", "trade", "1m", format="bars", **desc)
        base = self.dataset_dir(key)
        written = 0
        with self._lock:
            if bars is not None and not bars.is_empty():
                bars = self._conform(bars, TRADE_SCHEMA, ["ts", "occ", "expiration", "strike", "right", "close"]).drop_nulls(["ts", "occ", "close"])
                bars = bars.with_columns(pl.col("right").str.to_uppercase().str.slice(0, 1), _et_date().alias("_d"))
                for (d,), part in bars.partition_by("_d", as_dict=True).items():
                    p = base / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    part = part.drop("_d")
                    if p.exists():
                        part = pl.concat([pl.read_parquet(p), part]).unique(subset=["occ", "ts"], keep="last")
                    _atomic_parquet(part.sort(["ts", "expiration", "right", "strike"]), p, compression="zstd", statistics=True)
                    self._note(key, p, source=source, rows=part.height, contracts=int(part["occ"].n_unique()))
                    written += part.height
            if index_rows:
                idx = self._conform(pl.DataFrame(index_rows, infer_schema_length=None), INDEX_SCHEMA, ["date", "occ", "status"])
                ip = base / "index.parquet"
                if ip.exists():
                    idx = pl.concat([pl.read_parquet(ip), idx]).unique(subset=["date", "occ"], keep="last")
                _atomic_parquet(idx.sort(["date", "expiration", "right", "strike"]), ip, compression="zstd")
                self._dirty(key)
            if contracts is not None and not contracts.is_empty():
                self.write_contracts(key, contracts)
                self._dirty(key)
            self._opt_cache.clear()
        return {"dataset": key.id, "bars_written": written, "pairs_indexed": len(index_rows)}

    def option_bar_index(self, sym: str, provider: str | None = None, dataset: DatasetKey | None = None) -> pl.DataFrame:
        ds = dataset or self.option_dataset(sym, "trade", provider)
        ip = self.dataset_dir(ds) / "index.parquet" if ds is not None else None
        if ip is None or not ip.exists():
            return pl.DataFrame(schema=INDEX_SCHEMA)
        return pl.read_parquet(ip)

    def read_option_bars(self, sym: str, d: date, provider: str | None = None, dataset: DatasetKey | None = None) -> pl.DataFrame | None:
        ds = dataset or self.option_dataset(sym, "trade", provider)
        if ds is None:
            return None
        p = self.dataset_dir(ds) / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"
        return pl.read_parquet(p) if p.exists() else None

    def write_contracts(self, key: DatasetKey, df: pl.DataFrame) -> Path:
        df = self._conform(df, CONTRACT_SCHEMA, ["occ", "expiration", "strike", "right"])
        p = self.dataset_dir(key) / "contracts.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if p.exists():
                df = pl.concat([pl.read_parquet(p), df]).unique(subset=["occ"], keep="last")
            _atomic_parquet(df.sort(["expiration", "right", "strike"]), p, compression="zstd")
        return p

    def read_contracts(self, key: DatasetKey) -> pl.DataFrame:
        p = self.dataset_dir(key) / "contracts.parquet"
        return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=CONTRACT_SCHEMA)

    # ---------- derived / reference ----------
    def write_iv_history(self, sym: str, df: pl.DataFrame, dataset: DatasetKey | None = None) -> Path | None:
        p = self.iv_path(sym, dataset)
        if p is None:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(df, p)
        return p

    def read_iv_history(self, sym: str, dataset: DatasetKey | None = None) -> pl.DataFrame:
        p = self.iv_path(sym, dataset)
        if p is not None and p.exists():
            return pl.read_parquet(p).sort("date")
        return pl.DataFrame(schema={"date": pl.Date, "atm_iv_30d": pl.Float64, "atm_iv_front": pl.Float64, "close": pl.Float64})

    def write_events(self, df: pl.DataFrame, replace_source: str | None = None) -> Path:
        df = self._conform(df, EVENT_SCHEMA, ["date", "event_type"])
        p = self._p("reference", "events", "events.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if p.exists():
                old = pl.read_parquet(p)
                if replace_source:
                    old = old.filter(pl.col("source") != replace_source)
                df = pl.concat([old, df]).unique(subset=["date", "event_type", "time"], keep="last")
            _atomic_parquet(df.sort("date"), p)
        return p

    def read_events(self) -> pl.DataFrame:
        p = self._p("reference", "events", "events.parquet")
        return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=EVENT_SCHEMA)

    def write_corporate_actions(self, sym: str, df: pl.DataFrame) -> Path:
        df = self._conform(df, CORP_ACTION_SCHEMA, ["ex_date", "action"])
        p = self._p("reference", "corporate_actions", f"{_ref_name(sym)}.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(df.sort("ex_date"), p)
        return p

    def read_corporate_actions(self, sym: str) -> pl.DataFrame:
        if not safe_name(sym):
            return pl.DataFrame(schema=CORP_ACTION_SCHEMA)
        p = self._p("reference", "corporate_actions", f"{sym}.parquet")
        return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=CORP_ACTION_SCHEMA)

    # ---------- manifests / dataset versions ----------
    def _dataset_files(self, key: DatasetKey) -> list[Path]:
        base = self.dataset_dir(key)
        files = sorted((base / "data").rglob("*.parquet")) if (base / "data").exists() else []
        # sessions.json holds which sessions are whole and which only partly present: a session marked partial is not priced,
        # so the marks are part of what a result depends on and belong to the dataset's version
        for extra in ("index.parquet", "contracts.parquet", "derived/iv.parquet", "sessions.json"):
            if (base / extra).exists():
                files.append(base / extra)
        return files

    def head(self, key: DatasetKey) -> dict:
        base = self.dataset_dir(key)
        h = base / "HEAD"
        if not h.exists():
            return {}
        f = base / "manifests" / h.read_text().strip()
        return json.loads(f.read_text()) if f.exists() else {}

    def commit(self, key: DatasetKey) -> dict:
        """Write a new immutable manifest when the dataset's files changed. Returns HEAD."""
        import pyarrow.parquet as pq
        base = self.dataset_dir(key)
        with self._lock:
            prev = self.head(key)
            prev_files = prev.get("files", {})
            prov_f = base / "provenance.json"
            prov = json.loads(prov_f.read_text()) if prov_f.exists() else {}
            entries = {}
            for f in self._dataset_files(key):
                rel = str(f.relative_to(base))
                st = f.stat()
                old = prev_files.get(rel)
                if old and old.get("bytes") == st.st_size and old.get("mtime") == int(st.st_mtime):
                    entries[rel] = old
                    continue
                entries[rel] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": st.st_size, "mtime": int(st.st_mtime),
                                "rows": pq.read_metadata(f).num_rows if f.suffix == ".parquet" else None, "provenance": prov.get(rel) or {"provider": key.provider}}
            version = hashlib.sha256("".join(f"{k}:{v['sha256']}" for k, v in sorted(entries.items())).encode()).hexdigest()[:16]
            if prev.get("version") == version:
                (base / "DIRTY").unlink(missing_ok=True)
                return prev
            seq = int(prev.get("seq", 0)) + 1
            man = {"dataset": key.id, "seq": seq, "version": version, "parent": prev.get("version"), "created": _now(), "descriptor": self.descriptor(key), "files": entries}
            md = base / "manifests"
            md.mkdir(parents=True, exist_ok=True)
            name = f"{seq:06d}-{version}.json"
            _atomic_text((md / name), json.dumps(man))
            _atomic_text((base / "HEAD"), name)
            (base / "DIRTY").unlink(missing_ok=True)
            return man

    def rebuild_manifest(self, sym: str) -> dict:
        """Commit every dataset of a symbol; returns the combined symbol version."""
        heads = {k.id: self.commit(k).get("version", "none") for k in self.datasets(sym)}
        return {"symbol": sym, "version": self._combine(heads), "datasets": heads}

    @staticmethod
    def _combine(heads: dict[str, str]) -> str:
        return hashlib.sha256(json.dumps(heads, sort_keys=True).encode()).hexdigest()[:16] if heads else "none"

    def manifest(self, sym: str) -> dict:
        keys = self.datasets(sym)
        heads = {k.id: (self.head(k).get("version") or None) for k in keys}
        if not heads or any(v is None for v in heads.values()) or any((self.dataset_dir(k) / "DIRTY").exists() for k in keys):
            return {}  # something was written since the last manifest: the caller commits first
        return {"symbol": sym, "version": self._combine(heads), "datasets": heads}

    def dataset_version(self, sym: str) -> str:
        """Content version of everything stored for a symbol (part of every cache key)."""
        m = self.manifest(sym) or self.rebuild_manifest(sym)
        return m.get("version", "none")

    def dataset_versions(self, sym: str) -> dict[str, str]:
        return (self.manifest(sym) or self.rebuild_manifest(sym)).get("datasets", {})

    def events_version(self) -> str:
        p = self._p("reference", "events", "events.parquet")
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"

    def storage_bytes(self, sym: str | None = None, key: DatasetKey | None = None) -> int:
        if key is not None:
            return sum(f.stat().st_size for f in self.dataset_dir(key).rglob("*") if f.is_file())
        if sym:
            return sum(self.storage_bytes(key=k) for k in self.datasets(sym, all_versions=True))
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def minutes_to_daily(m: pl.DataFrame) -> pl.DataFrame:
    return (
        m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum()
        )
        .sort("date")
    )
