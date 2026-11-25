"""Bulk imports of historical files: CSV, compressed CSV, ZIP archives and Parquet, of any size.

`inspect(path)` looks at a file, an archive or a directory and says, per member, which
known layout it is (by its column names, never by guessing from values) and what it would
become in the store. `import_path` then streams it:

  * CSV is read in batches and Parquet row group by row group; an archive is processed one
    member at a time through a temporary file - nothing is ever loaded whole, so several
    years of option data import in bounded memory
  * rows are written per session; a session that straddles two batches is carried over, and
    a session that reappears later in the input is merged with what was already written
  * invalid rows (crossed quotes, non-positive prices, unknown rights) are counted and
    reported, never repaired
  * trade bars are stored as TRADES and quotes as QUOTES; the two are never mixed

Recognised layouts
  option quotes   OptionsDX (wide, [C_BID] / [P_BID] ...), Cboe DataShop (end-of-day and
                  interval), historicaloptiondata.com, DoltHub option_chain, ThetaData
                  exports, and any file with expiration + strike + type + bid + ask
  option trades   Polygon / Massive flat-file aggregates (O:SPY...), Databento OHLCV with
                  OCC symbols, Alpaca bar exports, any OCC symbol + OHLCV file
  underlying      Stooq, Yahoo, Kibot / FirstRate (headerless), Polygon stock aggregates,
                  any timestamp + OHLC file
"""
from __future__ import annotations

import gzip
import re
import shutil
import tempfile
import zipfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from strata.core import calendar as cal
from strata.data.store import MarketStore

BATCH = 400_000
ET = ZoneInfo("America/New_York")
SUFFIXES = (".csv", ".txt", ".csv.gz", ".txt.gz", ".parquet", ".pq")

# id -> (kind, label, required lower-case columns, a description of the vendor layout)
FORMATS: dict[str, dict] = {
    "optionsdx": {"kind": "option_quotes", "label": "OptionsDX (wide: one row per strike, calls and puts side by side)", "need": {"[quote_unixtime]", "[expire_date]", "[strike]", "[c_bid]", "[p_bid]"}},
    "cboe_eod": {"kind": "option_quotes", "label": "Cboe DataShop end-of-day option quotes", "need": {"underlying_symbol", "quote_date", "expiration", "strike", "option_type", "bid_eod", "ask_eod"}},
    "cboe_interval": {"kind": "option_quotes", "label": "Cboe DataShop interval option quotes", "need": {"underlying_symbol", "quote_datetime", "expiration", "strike", "option_type", "bid", "ask"}},
    "historicaloptiondata": {"kind": "option_quotes", "label": "historicaloptiondata.com option quotes", "need": {"underlying", "optionroot", "type", "expiration", "quotedate", "strike", "bid", "ask"}},
    "dolthub": {"kind": "option_quotes", "label": "DoltHub option_chain export", "need": {"act_symbol", "expiration", "strike", "call_put", "bid", "ask", "date"}},
    "thetadata_eod": {"kind": "option_quotes", "label": "ThetaData end-of-day option export", "need": {"root", "expiration", "strike", "right", "bid", "ask", "date"}},
    "polygon_options": {"kind": "option_trades", "label": "Polygon / Massive option aggregates (flat files)", "need": {"ticker", "open", "high", "low", "close", "volume", "window_start"}, "ticker_prefix": "O:"},
    "polygon_stocks": {"kind": "underlying_bars", "label": "Polygon / Massive stock aggregates (flat files)", "need": {"ticker", "open", "high", "low", "close", "volume", "window_start"}},
    "databento_ohlcv": {"kind": "option_trades", "label": "Databento OHLCV bars with OCC symbols", "need": {"ts_event", "open", "high", "low", "close", "volume", "symbol"}},
    "stooq": {"kind": "underlying_bars", "label": "Stooq bars", "need": {"<ticker>", "<date>", "<open>", "<high>", "<low>", "<close>"}},
    "yahoo": {"kind": "underlying_bars", "label": "Yahoo Finance daily CSV", "need": {"date", "open", "high", "low", "close", "adj close"}},
    "generic_option_quotes": {"kind": "option_quotes", "label": "Option quotes (generic columns)", "need": {"strike", "bid", "ask"}},
    "generic_option_trades": {"kind": "option_trades", "label": "Option trade bars (generic columns)", "need": {"open", "high", "low", "close"}},
    "generic_bars": {"kind": "underlying_bars", "label": "Price bars (generic columns)", "need": {"open", "high", "low", "close"}},
    "headerless_bars": {"kind": "underlying_bars", "label": "Headerless bars (Kibot / FirstRate: date[,time],open,high,low,close,volume)", "need": set()},
}
EXP_COLS = ("expiration", "expiry", "expiration_date", "exp", "expire_date", "expirdate")
RIGHT_COLS = ("right", "type", "option_type", "call_put", "cp", "cp_flag", "put_call")
TS_COLS = ("ts", "timestamp", "datetime", "quote_datetime", "quote_time", "time", "date", "quote_date", "quotedate", "t")
OCC_COLS = ("occ", "symbol", "contract", "option_symbol", "ticker", "contract_symbol", "contractid")
OCC_RE = r"^(?:O:)?\s*([A-Z][A-Z0-9.]{0,5})\s*(\d{6})([CP])(\d{8})$"


def detect(columns: list[str], sample: pl.DataFrame | None = None) -> str | None:
    """Layout id for a header (and, to tell option tickers from stock tickers, a few rows)."""
    cols = {c.strip().lower() for c in columns}
    for fid in ("optionsdx", "cboe_eod", "cboe_interval", "historicaloptiondata", "dolthub", "thetadata_eod", "stooq", "yahoo"):
        if FORMATS[fid]["need"] <= cols:
            return fid
    if FORMATS["polygon_options"]["need"] <= cols:
        first = str(sample[_col(sample, "ticker")][0]) if sample is not None and sample.height else ""
        return "polygon_options" if first.startswith("O:") else "polygon_stocks"
    if FORMATS["databento_ohlcv"]["need"] <= cols:
        return "databento_ohlcv"
    has_right, has_exp = bool(cols & set(RIGHT_COLS)), bool(cols & set(EXP_COLS))
    if {"strike", "bid", "ask"} <= cols and has_right and has_exp:
        return "generic_option_quotes"
    if {"open", "high", "low", "close"} <= cols:
        occ = next((c for c in OCC_COLS if c in cols), None)
        if occ and sample is not None and sample.height and re.match(OCC_RE, str(sample[_col(sample, occ)][0]).upper().replace(" ", "")):
            return "generic_option_trades"
        if cols & set(TS_COLS):
            return "generic_bars"
    return None


def _col(df: pl.DataFrame, lower: str) -> str:
    return next(c for c in df.columns if c.strip().lower() == lower)


def _looks_headerless(path: Path) -> bool:
    with _open_text(path) as f:
        first = f.readline().strip()
    parts = re.split(r"[,;\t]", first)
    return len(parts) >= 5 and bool(re.match(r"^\d{1,4}[-/.]?\d{1,2}[-/.]?\d{1,4}", parts[0])) and all(re.fullmatch(r"-?\d+(\.\d+)?", p.strip()) for p in parts[-4:])


def _open_text(path: Path):
    return gzip.open(path, "rt", errors="replace") if path.suffix == ".gz" else open(path, "rt", errors="replace")


def _members(path: Path) -> list[tuple[str, int]]:
    """(name, bytes) of every importable member of a file, archive or directory."""
    if path.is_dir():
        return sorted((str(p), p.stat().st_size) for p in path.rglob("*") if p.is_file() and (p.name.lower().endswith(SUFFIXES) or p.suffix.lower() == ".zip"))
    return [(str(path), path.stat().st_size)]


def _peek(path: Path) -> dict:
    """Header, a few rows and the detected layout of one plain file."""
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        lf = pl.scan_parquet(path)
        cols = lf.collect_schema().names()
        head = lf.head(5).collect()
        fmt = detect(cols, head)
        return {"columns": cols, "format": fmt, "rows": int(pl.scan_parquet(path).select(pl.len()).collect().item()), "sample": head.head(3).to_dicts()}
    if _looks_headerless(path):
        return {"columns": [], "format": "headerless_bars", "rows": None, "sample": []}
    src = path
    tmp = None
    if name.endswith(".gz"):
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        with gzip.open(path, "rb") as f, open(tmp, "wb") as o:
            o.write(b"".join(f.readline() for _ in range(40)))
        src = tmp
    try:
        head = pl.read_csv(src, n_rows=30, infer_schema_length=30, truncate_ragged_lines=True, ignore_errors=True)
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
    return {"columns": head.columns, "format": detect(head.columns, head), "rows": None, "sample": head.head(3).to_dicts()}


def inspect(path: str) -> dict:
    """What is at `path` and how it would be imported. Reads headers only."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"nothing exists at {p}")
    files, total = [], 0
    for name, size in _members(p)[:2000]:
        fp = Path(name)
        total += size
        if fp.suffix.lower() == ".zip":
            with zipfile.ZipFile(fp) as z:
                infos = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(SUFFIXES)]
                shown = None
                for i in infos[:1]:
                    with tempfile.TemporaryDirectory() as td:
                        out = Path(td) / Path(i.filename).name
                        with z.open(i) as src, open(out, "wb") as dst:
                            dst.write(src.read(1_000_000))  # the header and first rows are enough to recognise the layout
                        shown = _peek(out) if not out.name.lower().endswith((".parquet", ".pq")) else None
                files.append({"path": name, "archive": True, "members": len(infos), "bytes": size, "uncompressed_bytes": sum(i.file_size for i in infos), "format": (shown or {}).get("format"),
                              "columns": (shown or {}).get("columns", [])[:40], "first_member": infos[0].filename if infos else None})
        else:
            pk = _peek(fp)
            files.append({"path": name, "archive": False, "bytes": size, "format": pk["format"], "columns": pk["columns"][:40], "rows": pk["rows"], "sample": pk["sample"]})
    for f in files:
        fm = FORMATS.get(f["format"] or "")
        f["label"] = fm["label"] if fm else "Not recognised"
        f["kind"] = fm["kind"] if fm else None
    known = [f for f in files if f["kind"]]
    kinds = sorted({f["kind"] for f in known})
    return {"path": str(p), "files": known, "unrecognised": [{"path": f["path"], "columns": f["columns"]} for f in files if not f["kind"]][:20], "bytes": total, "kinds": kinds,
            "formats": {k: FORMATS[k]["label"] for k in FORMATS},
            "note": ("Trade bars are stored as trade prints and are never read as bid/ask quotes." if "option_trades" in kinds else "") or ("Quotes with a single timestamp per session are stored as end-of-day: they cannot price intraday entries." if "option_quotes" in kinds else "")}


# ----------------------------------------------------------------------------- normalisation
def _to_ts(expr: pl.Expr, dtype, tz: str) -> pl.Expr:
    if dtype == pl.String:
        e = expr.str.strip_chars().str.to_datetime(strict=False, time_unit="us")
        return e.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null").dt.convert_time_zone("UTC")
    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.Float64):
        return pl.when(expr > 10**17).then(pl.from_epoch(expr.cast(pl.Int64), time_unit="ns")).when(expr > 10**14).then(pl.from_epoch(expr.cast(pl.Int64), time_unit="us")) \
            .when(expr > 10**11).then(pl.from_epoch(expr.cast(pl.Int64), time_unit="ms")).otherwise(pl.from_epoch(expr.cast(pl.Int64), time_unit="s")).dt.replace_time_zone("UTC")
    if dtype == pl.Date:
        return expr.cast(pl.Datetime("us")).dt.replace_time_zone(tz).dt.convert_time_zone("UTC")
    if isinstance(dtype, pl.Datetime):
        return (expr.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null") if dtype.time_zone is None else expr).dt.convert_time_zone("UTC")
    raise ValueError(f"cannot read a timestamp from a column of type {dtype}")


def _eod_ts(d: pl.Expr) -> pl.Expr:
    """16:00 New York on the given date, in UTC (early closes are handled by the engine's calendar, not here)."""
    return (d.cast(pl.Datetime("us")) + pl.duration(hours=16)).dt.replace_time_zone("America/New_York", ambiguous="earliest").dt.convert_time_zone("UTC")


def _date(expr: pl.Expr, dtype) -> pl.Expr:
    if dtype == pl.String:
        e = expr.str.strip_chars()
        return pl.coalesce(e.str.to_date("%Y-%m-%d", strict=False), e.str.to_date("%m/%d/%Y", strict=False), e.str.to_date("%Y%m%d", strict=False), e.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False))
    if dtype in (pl.Int64, pl.Int32):
        return expr.cast(pl.String).str.to_date("%Y%m%d", strict=False)
    return expr.cast(pl.Date, strict=False)


def normalise(df: pl.DataFrame, fmt: str, tz: str = "America/New_York", symbol: str | None = None) -> tuple[pl.DataFrame, str | None]:
    """One batch in a vendor layout -> the store's columns. Returns (frame, symbol found in the data)."""
    df = df.rename({c: c.strip().lower() for c in df.columns})
    t = df.schema
    found = None
    f = pl.Float64

    def num(c):
        return pl.col(c).cast(f, strict=False)

    def first(names):
        return next((c for c in names if c in df.columns), None)

    if fmt == "optionsdx":
        ts = pl.from_epoch(pl.col("[quote_unixtime]").cast(pl.Int64, strict=False), time_unit="s").dt.replace_time_zone("UTC")
        exp = _date(pl.col("[expire_date]"), t["[expire_date]"])
        und = num("[underlying_last]") if "[underlying_last]" in df.columns else pl.lit(None, dtype=f)
        parts = []
        for r, p in (("C", "c"), ("P", "p")):
            opt = lambda n: num(f"[{p}_{n}]") if f"[{p}_{n}]" in df.columns else pl.lit(None, dtype=f)  # noqa: E731
            parts.append(df.select(ts.alias("ts"), exp.alias("expiration"), num("[strike]").alias("strike"), pl.lit(r).alias("right"), opt("bid").alias("bid"), opt("ask").alias("ask"), opt("volume").alias("volume"),
                                   opt("iv").alias("iv"), opt("delta").alias("delta"), opt("gamma").alias("gamma"), opt("theta").alias("theta"), opt("vega").alias("vega"), und.alias("underlying_price")))
        out = pl.concat(parts)
    elif fmt in ("cboe_eod", "cboe_interval"):
        found = str(df["underlying_symbol"][0]).lstrip("^") if df.height else None
        if fmt == "cboe_eod":
            ts, bid, ask = _eod_ts(_date(pl.col("quote_date"), t["quote_date"])), num("bid_eod"), num("ask_eod")
            und = ((num("underlying_bid_eod") + num("underlying_ask_eod")) / 2) if "underlying_bid_eod" in df.columns else pl.lit(None, dtype=f)
        else:
            ts, bid, ask = _to_ts(pl.col("quote_datetime"), t["quote_datetime"], tz), num("bid"), num("ask")
            und = ((num("underlying_bid") + num("underlying_ask")) / 2) if "underlying_bid" in df.columns else pl.lit(None, dtype=f)
        opt = lambda n, alt=None: num(n) if n in df.columns else (num(alt) if alt and alt in df.columns else pl.lit(None, dtype=f))  # noqa: E731
        out = df.select(ts.alias("ts"), _date(pl.col("expiration"), t["expiration"]).alias("expiration"), num("strike").alias("strike"), pl.col("option_type").cast(pl.String).alias("right"), bid.alias("bid"), ask.alias("ask"),
                        opt("trade_volume").alias("volume"), opt("open_interest").alias("open_interest"), opt("implied_volatility_1545", "implied_volatility").alias("iv"), opt("delta_1545", "delta").alias("delta"),
                        opt("gamma_1545", "gamma").alias("gamma"), opt("theta_1545", "theta").alias("theta"), opt("vega_1545", "vega").alias("vega"), und.alias("underlying_price"))
    elif fmt == "historicaloptiondata":
        found = str(df["underlying"][0]) if df.height else None
        opt = lambda n: num(n) if n in df.columns else pl.lit(None, dtype=f)  # noqa: E731
        out = df.select(_eod_ts(_date(pl.col("quotedate"), t["quotedate"])).alias("ts"), _date(pl.col("expiration"), t["expiration"]).alias("expiration"), num("strike").alias("strike"), pl.col("type").cast(pl.String).alias("right"),
                        num("bid").alias("bid"), num("ask").alias("ask"), opt("volume").alias("volume"), opt("openinterest").alias("open_interest"), opt("impliedvol").alias("iv"), opt("delta").alias("delta"),
                        opt("gamma").alias("gamma"), opt("theta").alias("theta"), opt("vega").alias("vega"), opt("underlying_last").alias("underlying_price"))
    elif fmt in ("dolthub", "thetadata_eod", "generic_option_quotes"):
        sym_col = first(("act_symbol", "root", "underlying", "underlying_symbol"))
        found = str(df[sym_col][0]) if sym_col and df.height else None
        tc = first(TS_COLS)
        if tc is None:
            raise ValueError("no date or timestamp column")
        daily = t[tc] == pl.Date or (t[tc] == pl.String and df.height and len(str(df[tc][0]).strip()) <= 10) or (t[tc] in (pl.Int64, pl.Int32) and df.height and 19000101 <= int(df[tc][0] or 0) <= 21001231)
        ts = _eod_ts(_date(pl.col(tc), t[tc])) if daily else _to_ts(pl.col(tc), t[tc], tz)
        ec, rc = first(EXP_COLS), first(RIGHT_COLS)
        opt = lambda *ns: next((num(n) for n in ns if n in df.columns), pl.lit(None, dtype=f))  # noqa: E731
        out = df.select(ts.alias("ts"), _date(pl.col(ec), t[ec]).alias("expiration"), num("strike").alias("strike"), pl.col(rc).cast(pl.String).alias("right"), num("bid").alias("bid"), num("ask").alias("ask"),
                        opt("volume", "vol").alias("volume"), opt("open_interest", "oi", "openinterest").alias("open_interest"), opt("iv", "implied_volatility", "impliedvol").alias("iv"), opt("delta").alias("delta"),
                        opt("gamma").alias("gamma"), opt("theta").alias("theta"), opt("vega").alias("vega"), opt("underlying_price", "underlying_last", "spot").alias("underlying_price"))
        if fmt == "thetadata_eod" and df.height and (df["strike"].cast(f, strict=False).max() or 0) > 100_000:
            out = out.with_columns(pl.col("strike") / 1000)  # ThetaData quotes strikes in tenths of a cent
    elif fmt in ("polygon_options", "databento_ohlcv", "generic_option_trades"):
        oc = "ticker" if fmt == "polygon_options" else "symbol" if fmt == "databento_ohlcv" else first(OCC_COLS)
        tc = "window_start" if fmt == "polygon_options" else "ts_event" if fmt == "databento_ohlcv" else first(TS_COLS)
        occ = pl.col(oc).cast(pl.String).str.to_uppercase().str.replace("^O:", "").str.replace_all(" ", "")
        g = occ.str.extract_groups(r"^([A-Z][A-Z0-9.]{0,5}?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
        px = (lambda c: num(c) / 1e9) if (fmt == "databento_ohlcv" and df.height and t["close"] in (pl.Int64, pl.Int32)) else num  # Databento's raw prices are fixed-point 1e-9
        out = df.select(_to_ts(pl.col(tc), t[tc], tz).alias("ts"), occ.alias("occ"), g.struct.field("1").alias("_root"),
                        (pl.lit("20") + g.struct.field("2") + g.struct.field("3") + g.struct.field("4")).str.to_date("%Y%m%d", strict=False).alias("expiration"),
                        (g.struct.field("6").cast(f, strict=False) / 1000).alias("strike"), g.struct.field("5").alias("right"), px("open").alias("open"), px("high").alias("high"), px("low").alias("low"), px("close").alias("close"),
                        num("volume").cast(pl.Int64, strict=False).alias("volume"), (num("transactions").cast(pl.Int64, strict=False) if "transactions" in df.columns else pl.lit(None, dtype=pl.Int64)).alias("trades"))
        roots = out["_root"].drop_nulls().unique().to_list()
        found = (symbol or (roots[0] if len(roots) == 1 else None))
        if symbol:
            out = out.filter(pl.col("_root").str.replace(r"\d$", "") == symbol.upper())
        out = out.drop("_root")
    elif fmt in ("polygon_stocks", "generic_bars", "yahoo", "stooq", "headerless_bars"):
        if fmt == "stooq":
            df = df.rename({c: c.strip("<>") for c in df.columns})
            found = str(df["ticker"][0]).split(".")[0].upper() if df.height else None
            if "time" in df.columns and df.height and int(df["time"].cast(pl.Int64, strict=False).max() or 0) > 0:
                df = df.with_columns((pl.col("date").cast(pl.String) + pl.col("time").cast(pl.String).str.zfill(6)).str.to_datetime("%Y%m%d%H%M%S", strict=False).alias("_ts"))
                tz = "Europe/Warsaw" if tz == "America/New_York" else tz  # Stooq intraday files are stamped in Warsaw time
            else:
                df = df.with_columns(pl.col("date").cast(pl.String).str.to_date("%Y%m%d", strict=False).alias("_ts"))
            tc = "_ts"
            df = df.rename({"vol": "volume"}) if "vol" in df.columns else df
        elif fmt == "polygon_stocks":
            tc = "window_start"
            if symbol:
                df = df.filter(pl.col("ticker") == symbol.upper())
            found = symbol or (str(df["ticker"][0]) if df.height else None)
        else:
            tc = next(c for c in TS_COLS if c in df.columns)
            if tc == "date" and "time" in df.columns and t["date"] == pl.String:
                df = df.with_columns((pl.col("date").str.strip_chars() + pl.lit(" ") + pl.col("time").cast(pl.String).str.strip_chars()).alias("_ts"))
                tc = "_ts"
        t = df.schema
        daily = t[tc] == pl.Date or (t[tc] == pl.String and df.height and len(str(df[tc][0]).strip()) <= 10)
        if daily:
            out = df.select(_date(pl.col(tc), t[tc]).alias("date"), num("open").alias("open"), num("high").alias("high"), num("low").alias("low"), num("close").alias("close"),
                            (num("volume") if "volume" in df.columns else pl.lit(None, dtype=f)).alias("volume"))
        else:
            out = df.select(_to_ts(pl.col(tc), t[tc], tz).alias("ts"), num("open").alias("open"), num("high").alias("high"), num("low").alias("low"), num("close").alias("close"),
                            (num("volume") if "volume" in df.columns else pl.lit(None, dtype=f)).alias("volume"))
    else:
        raise ValueError(f"unknown layout {fmt}")
    return out, (found.upper() if isinstance(found, str) and found else None)


# ----------------------------------------------------------------------------- streaming
def _batches(path: Path, fmt: str):
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=BATCH):
            yield pl.from_arrow(b)
        return
    src, tmp = path, None
    if name.endswith(".gz"):
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        with gzip.open(path, "rb") as f, open(tmp, "wb") as o:
            shutil.copyfileobj(f, o, 4 << 20)
        src = tmp
    try:
        kw = {"infer_schema_length": 5000, "truncate_ragged_lines": True, "ignore_errors": True}
        if fmt == "headerless_bars":
            with open(src, "rt", errors="replace") as f:
                n = len(re.split(r"[,;\t]", f.readline().strip()))
            cols = ["date", "time", "open", "high", "low", "close", "volume"][:n] if n >= 7 else ["ts", "open", "high", "low", "close", "volume"][:n]
            kw.update(has_header=False, new_columns=cols, schema_overrides={c: pl.String for c in ("date", "time", "ts") if c in cols})
        # the streaming engine reads the file in pieces; rows are handed on in order, BATCH at a time
        for b in pl.scan_csv(src, **kw).collect_batches(chunk_size=BATCH, maintain_order=True):
            if b.height:
                yield b
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def _plain_files(path: Path, workdir: Path):
    """Every plain file to import; archive members are extracted one at a time into `workdir`."""
    for name, _size in _members(path):
        fp = Path(name)
        if fp.suffix.lower() == ".zip":
            with zipfile.ZipFile(fp) as z:
                for i in z.infolist():
                    if i.is_dir() or not i.filename.lower().endswith(SUFFIXES):
                        continue
                    out = workdir / Path(i.filename).name
                    with z.open(i) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, 4 << 20)
                    try:
                        yield out, f"{fp.name}:{i.filename}"
                    finally:
                        out.unlink(missing_ok=True)
        else:
            yield fp, fp.name


class _DayWriter:
    """Writes option rows per session, carrying the trailing (possibly incomplete) session
    across batches and merging a session that shows up again later in the input."""

    def __init__(self, store: MarketStore, sym: str, provider: str, kind: str, source: str):
        self.store, self.sym, self.provider, self.kind, self.source = store, sym, provider, kind, source
        self.carry: pl.DataFrame | None = None
        self.written: dict[date, int] = {}
        self.rows = 0
        self.peak_buffer = 0

    def add(self, df: pl.DataFrame, final: bool = False):
        if self.carry is not None and not self.carry.is_empty():
            df = pl.concat([self.carry, df], how="diagonal_relaxed") if df is not None and not df.is_empty() else self.carry
        self.carry = None
        if df is None or df.is_empty():
            return
        df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))
        self.peak_buffer = max(self.peak_buffer, df.height)
        last = df["_d"].max()
        if not final:
            self.carry = df.filter(pl.col("_d") == last).drop("_d")
            df = df.filter(pl.col("_d") != last)
        for (d,), part in sorted(df.partition_by("_d", as_dict=True).items(), key=lambda kv: kv[0]):
            self._write(d, part.drop("_d"))

    def _write(self, d: date, part: pl.DataFrame):
        if self.kind == "option_quotes":
            if d in self.written:
                old = self.store.read_option_day(self.sym, d, "quote", self.provider)
                if old is not None:
                    part = pl.concat([old, part], how="diagonal_relaxed").unique(subset=["ts", "expiration", "strike", "right"], keep="last")
            self.store.write_option_day(self.sym, d, part, provider=self.provider, source=self.source, status="complete")
        else:
            idx = part.group_by("occ").agg(pl.col("expiration").first(), pl.col("strike").first(), pl.col("right").first(), pl.len().alias("bars"), pl.col("volume").sum().alias("volume"))
            rows = [{"date": d, "occ": r["occ"], "expiration": r["expiration"], "strike": r["strike"], "right": r["right"], "status": "ok", "bars": r["bars"], "volume": r["volume"]} for r in idx.to_dicts()]
            con = idx.select("occ", "expiration", "strike", "right").with_columns(pl.lit(self.sym).alias("underlying"), pl.lit("imported file").alias("source"), pl.col("occ").str.extract(r"^([A-Z][A-Z0-9.]*?)\d{6}[CP]", 1).alias("root"),
                                                                                  pl.col("occ").str.contains(r"^[A-Z.]+\d\d{6}[CP]").alias("adjusted"))
            self.store.write_option_bars(self.sym, self.provider, part, rows, con, source=self.source, complete_chain=True)
        self.written[d] = self.written.get(d, 0) + part.height
        self.rows += part.height


def import_path(path: str, symbol: str | None = None, fmt: str | None = None, tz: str = "America/New_York", provider: str = "import", store: MarketStore | None = None, progress=None, should_cancel=None) -> dict:
    """Stream every recognised file at `path` into the store. Returns a report of what was
    written, what was rejected and why, and the peak number of rows held in memory."""
    store = store or MarketStore()
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"nothing exists at {p}")
    total_bytes = max(1, sum(s for _, s in _members(p)))
    done_bytes = 0
    rep = {"path": str(p), "provider": provider, "files": [], "rows_in": 0, "rows_written": 0, "rejected": {}, "sessions": 0, "symbols": [], "skipped": [], "peak_rows_in_memory": 0}
    writers: dict[tuple, _DayWriter] = {}
    bars: dict[str, list[pl.DataFrame]] = {}
    daily: dict[str, list[pl.DataFrame]] = {}

    def reject(why, n):
        if n:
            rep["rejected"][why] = rep["rejected"].get(why, 0) + int(n)

    with tempfile.TemporaryDirectory(prefix="strata-import-") as td:
        for fp, label in _plain_files(p, Path(td)):
            if should_cancel and should_cancel():
                raise InterruptedError("import cancelled")
            try:
                f_fmt = fmt or _peek(fp)["format"]
            except Exception as e:  # noqa: BLE001
                rep["skipped"].append({"file": label, "reason": f"could not be read: {str(e)[:120]}"})
                continue
            if not f_fmt:
                rep["skipped"].append({"file": label, "reason": "layout not recognised (columns do not match any known format)"})
                continue
            kind = FORMATS[f_fmt]["kind"]
            n_in = n_out = 0
            for b in _batches(fp, f_fmt):
                if should_cancel and should_cancel():
                    raise InterruptedError("import cancelled")
                n_in += b.height
                out, found = normalise(b, f_fmt, tz, symbol)
                sym = (symbol or found or "").upper()
                if not sym:
                    raise ValueError(f"{label}: the file does not name its underlying; pass the symbol")
                if kind == "option_quotes":
                    out = out.with_columns(pl.col("right").cast(pl.String).str.to_uppercase().str.slice(0, 1))
                    ok = (pl.col("bid") >= 0) & (pl.col("ask") >= pl.col("bid")) & (pl.col("ask") > 0) & pl.col("right").is_in(["C", "P"]) & (pl.col("strike") > 0) & pl.col("expiration").is_not_null() & pl.col("ts").is_not_null()
                    reject("no usable quote (missing, zero or crossed bid/ask), or unknown contract fields", out.filter(~ok.fill_null(False)).height)
                    out = out.filter(ok.fill_null(False))
                    if "iv" in out.columns and out.height and (out["iv"].drop_nulls().max() or 0) > 5:
                        out = out.with_columns(pl.col("iv") / 100)
                    w = writers.setdefault((sym, kind), _DayWriter(store, sym, provider, kind, str(p)))
                    w.add(out)
                elif kind == "option_trades":
                    from strata.data.providers.sources import validate_trade_bars
                    out = out.drop_nulls(["ts", "occ", "close", "expiration"])
                    clean, st = validate_trade_bars(out)
                    for k in ("outside_session", "invalid_price", "duplicates"):
                        reject({"outside_session": "bars outside the regular session", "invalid_price": "bars with inconsistent or non-positive prices", "duplicates": "duplicate bars"}[k], st[k])
                    w = writers.setdefault((sym, kind), _DayWriter(store, sym, provider, kind, str(p)))
                    w.add(clean)
                    out = clean
                else:
                    bad = (pl.col("high") < pl.col("low")) | (pl.col("close") <= 0)
                    reject("bars with high below low or a non-positive close", out.filter(bad.fill_null(True)).height)
                    out = out.filter(~bad.fill_null(True))
                    (daily if "date" in out.columns else bars).setdefault(sym, []).append(out)
                n_out += out.height
                if sym not in rep["symbols"]:
                    rep["symbols"].append(sym)
            done_bytes += fp.stat().st_size if fp.exists() else 0
            rep["files"].append({"file": label, "format": f_fmt, "label": FORMATS[f_fmt]["label"], "rows_in": n_in, "rows_kept": n_out})
            rep["rows_in"] += n_in
            if progress:
                progress(min(0.95, done_bytes / total_bytes), f"{label}: {n_in:,} rows")
    for (sym, kind), w in writers.items():
        w.add(pl.DataFrame(), final=True)
        rep["rows_written"] += w.rows
        rep["sessions"] += len(w.written)
        rep["peak_rows_in_memory"] = max(rep["peak_rows_in_memory"], w.peak_buffer)
        if kind == "option_quotes" and w.written:
            from strata.data.derived import derive_iv_history
            try:
                derive_iv_history(store, sym)
            except Exception as e:  # noqa: BLE001
                rep.setdefault("warnings", []).append(f"implied-volatility history could not be derived for {sym}: {str(e)[:100]}")
        rep.setdefault("ranges", {})[f"{sym} {kind}"] = [min(w.written).isoformat(), max(w.written).isoformat()] if w.written else None
    for sym, parts in daily.items():
        d = pl.concat(parts).drop_nulls(["date", "close"]).unique(subset=["date"], keep="last").sort("date")
        store.write_daily(sym, d, provider=provider, source=str(p))
        rep["rows_written"] += d.height
        rep.setdefault("ranges", {})[f"{sym} daily bars"] = [str(d["date"][0]), str(d["date"][-1])] if d.height else None
    for sym, parts in bars.items():
        m = pl.concat(parts).drop_nulls(["ts", "close"]).unique(subset=["ts"], keep="last").sort("ts")
        step = int(m["ts"].diff().drop_nulls().dt.total_seconds().median() or 60) if m.height > 2 else 60
        res = "1m" if step <= 60 else "5m" if step <= 300 else "15m" if step <= 900 else "30m" if step <= 1800 else "60m"
        store.write_bars(sym, res, m, provider=provider, source=str(p))
        rep["rows_written"] += m.height
        rep.setdefault("ranges", {})[f"{sym} {res} bars"] = [str(m["ts"][0])[:10], str(m["ts"][-1])[:10]] if m.height else None
    from strata.data.store import KNOWN_SPECS, SymbolMeta
    for sym in rep["symbols"]:
        if store.symbols().get(sym) is None:
            store.upsert_symbol(SymbolMeta(symbol=sym, **KNOWN_SPECS.get(sym, {})))
        store.rebuild_manifest(sym)
    if progress:
        progress(1.0, "done")
    rep["note"] = "Imported files form their own dataset (provider \"" + provider + "\"); they never overwrite data downloaded from a provider."
    return rep


_ = (datetime, time, timezone, cal, ET)
