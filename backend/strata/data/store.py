"""Columnar market-data store (Parquet files, queried with Polars/DuckDB).

Layout under <data_dir>/market:
  symbols.json                               symbol metadata / contract specs
  underlying/<SYM>/<YYYY>.parquet            1-minute bars, ts = bar START (UTC)
  daily/<SYM>.parquet                        daily bars (date, open, high, low, close, volume)
  options/<SYM>/<YYYY>/<YYYY-MM-DD>.parquet  option quote snapshots for one session
  derived/<SYM>_iv.parquet                   daily ATM IV history derived from stored chains
  events/events.parquet                      scheduled economic events
  corporate_actions/<SYM>.parquet            splits / dividends
  manifests/<SYM>.json                       per-file sha256 -> dataset version
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from strata.config import get_settings

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
    provider: str = "import"
    synthetic: bool = False
    has_options: bool = True
    option_interval_minutes: int | None = None
    description: str = ""
    extra: dict = field(default_factory=dict)


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


class MarketStore:
    _lock = threading.RLock()

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_settings().market_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------
    def _p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def minute_path(self, sym: str, year: int) -> Path:
        return self._p("underlying", sym, f"{year}.parquet")

    def bars_path(self, sym: str, res: str, year: int) -> Path:
        if res == "1m":
            return self.minute_path(sym, year)
        return self._p("bars", res, sym, f"{year}.parquet")

    def daily_path(self, sym: str) -> Path:
        return self._p("daily", f"{sym}.parquet")

    def option_path(self, sym: str, d: date) -> Path:
        return self._p("options", sym, f"{d.year}", f"{d.isoformat()}.parquet")

    def iv_path(self, sym: str) -> Path:
        return self._p("derived", f"{sym}_iv.parquet")

    # ---------- symbol metadata ----------
    def symbols(self) -> dict[str, SymbolMeta]:
        f = self._p("symbols.json")
        if not f.exists():
            return {}
        raw = json.loads(f.read_text())
        return {k: SymbolMeta(**v) for k, v in raw.items()}

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
            f.write_text(json.dumps(cur, indent=2, sort_keys=True, default=str))

    def remove_symbol(self, sym: str) -> None:
        import shutil

        with self._lock:
            for p in [self._p("underlying", sym), self._p("options", sym)] + [self._p("bars", r, sym) for r in INTRADAY_RES if r != "1m"]:
                if p.exists():
                    shutil.rmtree(p)
            for p in [self.daily_path(sym), self.iv_path(sym), self._p("manifests", f"{sym}.json"), self._p("corporate_actions", f"{sym}.parquet")]:
                if p.exists():
                    p.unlink()
            cur = {k: asdict(v) for k, v in self.symbols().items() if k != sym}
            self._p("symbols.json").write_text(json.dumps(cur, indent=2, sort_keys=True, default=str))

    # ---------- writing ----------
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

    def write_minutes(self, sym: str, df: pl.DataFrame) -> list[Path]:
        df = self._conform(df, BAR_SCHEMA, ["ts", "open", "high", "low", "close"])
        out = []
        with self._lock:
            for (yr,), part in df.with_columns(pl.col("ts").dt.year().alias("_y")).partition_by("_y", as_dict=True).items():
                p = self.minute_path(sym, int(yr))
                p.parent.mkdir(parents=True, exist_ok=True)
                part = part.drop("_y")
                if p.exists():
                    part = pl.concat([pl.read_parquet(p), part]).unique(subset=["ts"], keep="last")
                part.sort("ts").write_parquet(p, compression="zstd")
                out.append(p)
        return out

    def write_bars(self, sym: str, res: str, df: pl.DataFrame) -> int:
        """Intraday bars at resolution `res` (bar START time). Merges/dedupes by ts."""
        if res not in INTRADAY_RES:
            raise ValueError(f"unsupported resolution {res}")
        df = self._conform(df, BAR_SCHEMA, ["ts", "open", "high", "low", "close"]).drop_nulls(["open", "high", "low", "close"])
        n = 0
        with self._lock:
            for (yr,), part in df.with_columns(pl.col("ts").dt.year().alias("_y")).partition_by("_y", as_dict=True).items():
                p = self.bars_path(sym, res, int(yr))
                p.parent.mkdir(parents=True, exist_ok=True)
                part = part.drop("_y")
                if p.exists():
                    part = pl.concat([pl.read_parquet(p), part]).unique(subset=["ts"], keep="last")
                part.sort("ts").write_parquet(p, compression="zstd")
                n += part.height
        return n

    def read_bars(self, sym: str, res: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        if res == "1m":
            return self.read_minutes(sym, start, end)
        d = self._p("bars", res, sym)
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        files = [f for f in files if (not start or int(f.stem) >= start.year) and (not end or int(f.stem) <= end.year)]
        if not files:
            return pl.DataFrame(schema=BAR_SCHEMA)
        df = pl.concat([pl.read_parquet(f) for f in files]).sort("ts")
        et_date = pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()
        if start:
            df = df.filter(et_date >= start)
        if end:
            df = df.filter(et_date <= end)
        return df

    def bar_resolutions(self, sym: str) -> list[str]:
        out = []
        if self._p("underlying", sym).exists() and any(self._p("underlying", sym).glob("*.parquet")):
            out.append("1m")
        for res in ("5m", "15m", "30m", "60m"):
            d = self._p("bars", res, sym)
            if d.exists() and any(d.glob("*.parquet")):
                out.append(res)
        return out

    def read_best_bars(self, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        """Finest stored bars per session: 1m > 5m > 15m > 30m > 60m > daily. Returns
        ts (bar start), end (bar end), OHLCV and res (seconds). A session only covered by
        a daily bar becomes one bar spanning the whole session."""
        from strata.core import calendar as cal
        frames, taken = [], None
        for res in self.bar_resolutions(sym):
            b = self.read_bars(sym, res, start, end)
            if b.is_empty():
                continue
            b = b.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))
            if taken is not None:
                b = b.filter(~pl.col("_d").is_in(taken.implode()))
            if b.is_empty():
                continue
            days = b["_d"].unique()
            taken = days if taken is None else pl.concat([taken, days])
            closes = {d: cal.session_bounds(d)[1] for d in days.to_list()}
            cmap = pl.DataFrame({"_d": list(closes), "_close": list(closes.values())}).with_columns(pl.col("_close").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
            secs = INTRADAY_RES[res]
            b = b.join(cmap, on="_d").with_columns(pl.min_horizontal(pl.col("ts") + pl.duration(seconds=secs), pl.col("_close")).alias("end"), pl.lit(secs, dtype=pl.Int32).alias("res"))
            frames.append(b.filter(pl.col("ts") < pl.col("_close")).drop("_close"))
        dd = self.read_daily(sym) if self.daily_path(sym).exists() else pl.DataFrame(schema=DAILY_SCHEMA)
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
        if not frames:
            return pl.DataFrame(schema={**BAR_SCHEMA, "end": pl.Datetime("us", "UTC"), "res": pl.Int32})
        cols = ["ts", "end", "open", "high", "low", "close", "volume", "res"]
        return pl.concat([f.select([pl.col(c).cast(pl.Float64) if c in ("open", "high", "low", "close", "volume") else pl.col(c) for c in cols]) for f in frames]).sort("ts")

    def official_closes(self, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame | None:
        """Daily closes (date, close) from the daily record, or None when no daily file is stored."""
        if not self.daily_path(sym).exists():
            return None
        d = self.read_daily(sym).select("date", "close").drop_nulls().sort("date")
        if start:
            d = d.filter(pl.col("date") >= start)
        if end:
            d = d.filter(pl.col("date") <= end)
        return d

    def note_source(self, sym: str, kind: str, provider: str, resolution: str, license: str = "", quote_type: str = "", url: str = "") -> None:
        """Record where a dataset came from (shown in the coverage matrix)."""
        with self._lock:
            m = self.symbols().get(sym) or self.symbol_meta(sym)
            src = dict(m.extra.get("sources") or {})
            src[f"{kind}:{resolution}"] = {"provider": provider, "kind": kind, "resolution": resolution, "license": license, "quote_type": quote_type, "url": url,
                                           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            m.extra = {**m.extra, "sources": src}
            if m.provider in ("import", "") or kind == "options":
                m.provider = provider
            self.upsert_symbol(m)

    def split_factor(self, sym: str, d: date) -> float:
        """Product of split ratios with ex-date after d (vendor prices are split-adjusted
        to today; multiply by this to get as-traded prices on session d)."""
        ca = self.read_corporate_actions(sym)
        f = 1.0
        for r in ca.iter_rows(named=True):
            if r["action"] == "split" and r["ratio"] and r["ex_date"] > d:
                f *= float(r["ratio"])
        return f

    def write_daily(self, sym: str, df: pl.DataFrame) -> Path:
        df = self._conform(df, DAILY_SCHEMA, ["date", "open", "high", "low", "close"])
        p = self.daily_path(sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if p.exists():
                df = pl.concat([pl.read_parquet(p), df]).unique(subset=["date"], keep="last")
            df.sort("date").write_parquet(p, compression="zstd")
        return p

    def write_option_day(self, sym: str, d: date, df: pl.DataFrame) -> Path:
        if not (("bid" in df.columns and "ask" in df.columns) or "last" in df.columns):
            raise ValueError("option data needs bid+ask quotes or last trade prices")
        df = self._conform(df, OPTION_SCHEMA, OPTION_REQUIRED)
        df = df.with_columns(pl.col("right").str.to_uppercase().str.slice(0, 1))
        p = self.option_path(sym, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.sort(["ts", "expiration", "right", "strike"]).write_parquet(p, compression="zstd", statistics=True)
        return p

    def write_events(self, df: pl.DataFrame, replace_source: str | None = None) -> Path:
        df = self._conform(df, EVENT_SCHEMA, ["date", "event_type"])
        p = self._p("events", "events.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if p.exists():
                old = pl.read_parquet(p)
                if replace_source:
                    old = old.filter(pl.col("source") != replace_source)
                df = pl.concat([old, df]).unique(subset=["date", "event_type", "time"], keep="last")
            df.sort("date").write_parquet(p)
        return p

    def write_corporate_actions(self, sym: str, df: pl.DataFrame) -> Path:
        df = self._conform(df, CORP_ACTION_SCHEMA, ["ex_date", "action"])
        p = self._p("corporate_actions", f"{sym}.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.sort("ex_date").write_parquet(p)
        return p

    # ---------- reading ----------
    def read_minutes(self, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        d = self._p("underlying", sym)
        if not d.exists():
            return pl.DataFrame(schema=BAR_SCHEMA)
        files = sorted(d.glob("*.parquet"))
        if start or end:
            files = [f for f in files if (not start or int(f.stem) >= start.year) and (not end or int(f.stem) <= end.year)]
        if not files:
            return pl.DataFrame(schema=BAR_SCHEMA)
        df = pl.concat([pl.read_parquet(f) for f in files]).sort("ts")
        if start or end:
            et_date = pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()
            if start:
                df = df.filter(et_date >= start)
            if end:
                df = df.filter(et_date <= end)
        return df

    def read_daily(self, sym: str) -> pl.DataFrame:
        p = self.daily_path(sym)
        if p.exists():
            return pl.read_parquet(p).sort("date")
        m = self.read_minutes(sym)
        if m.is_empty():
            return pl.DataFrame(schema=DAILY_SCHEMA)
        return minutes_to_daily(m)

    def option_days(self, sym: str) -> list[date]:
        d = self._p("options", sym)
        if not d.exists():
            return []
        return sorted(date.fromisoformat(f.stem) for f in d.glob("*/*.parquet"))

    def read_option_day(self, sym: str, d: date) -> pl.DataFrame | None:
        p = self.option_path(sym, d)
        if not p.exists():
            return None
        return pl.read_parquet(p)

    def read_iv_history(self, sym: str) -> pl.DataFrame:
        p = self.iv_path(sym)
        if p.exists():
            return pl.read_parquet(p).sort("date")
        return pl.DataFrame(schema={"date": pl.Date, "atm_iv_30d": pl.Float64, "atm_iv_front": pl.Float64, "close": pl.Float64})

    def read_events(self) -> pl.DataFrame:
        p = self._p("events", "events.parquet")
        return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=EVENT_SCHEMA)

    def read_corporate_actions(self, sym: str) -> pl.DataFrame:
        p = self._p("corporate_actions", f"{sym}.parquet")
        return pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=CORP_ACTION_SCHEMA)

    # ---------- manifests / dataset versions ----------
    def _files_for(self, sym: str) -> list[Path]:
        files = []
        subs = [self._p("underlying", sym), self._p("options", sym)] + [self._p("bars", r, sym) for r in INTRADAY_RES if r != "1m"]
        for sub in subs:
            if sub.exists():
                files += sorted(sub.rglob("*.parquet"))
        for p in [self.daily_path(sym), self.iv_path(sym), self._p("corporate_actions", f"{sym}.parquet")]:
            if p.exists():
                files.append(p)
        return files

    def rebuild_manifest(self, sym: str) -> dict:
        prev = self.manifest(sym)
        prev_files = prev.get("files", {})
        entries = {}
        for f in self._files_for(sym):
            rel = str(f.relative_to(self.root))
            st = f.stat()
            old = prev_files.get(rel)
            if old and old.get("bytes") == st.st_size and old.get("mtime") == int(st.st_mtime):
                entries[rel] = old
                continue
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            entries[rel] = {"sha256": h, "bytes": st.st_size, "mtime": int(st.st_mtime)}
        version = hashlib.sha256("".join(f"{k}:{v['sha256']}" for k, v in sorted(entries.items())).encode()).hexdigest()[:16]
        man = {"symbol": sym, "version": version, "updated": datetime.now(timezone.utc).isoformat(), "files": entries}
        p = self._p("manifests", f"{sym}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(man))
        return man

    def manifest(self, sym: str) -> dict:
        p = self._p("manifests", f"{sym}.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def dataset_version(self, sym: str) -> str:
        m = self.manifest(sym)
        if not m:
            m = self.rebuild_manifest(sym)
        return m.get("version", "none")

    def events_version(self) -> str:
        p = self._p("events", "events.parquet")
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"

    def storage_bytes(self, sym: str | None = None) -> int:
        if sym:
            return sum(f.stat().st_size for f in self._files_for(sym))
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
