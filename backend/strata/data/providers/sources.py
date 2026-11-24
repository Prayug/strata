"""Free-first market-data sources.

keyless (work out of the box):
  yahoo       underlying daily history (decades, with dividends/splits) and recent intraday
              bars: 60m for ~730 days, 5m for ~60 days, 1m for ~30 days
  cboe        official VIX daily history since 1990
  github_eod  end-of-day option chains for SPY / QQQ / IWM, 2008-2025 (MIT-licensed
              research dataset by P. Dubach, mirrored on GitHub): bid/ask, sizes, volume,
              open interest, IV and Greeks at the 16:00 ET snapshot
  dolthub     end-of-day option chains for ~2,000 US symbols since 2019 (DoltHub
              post-no-preference/options, CC BY-SA 4.0): bid/ask, IV, Greeks; four
              expirations (~2-8 weeks) and ~27 strikes per expiration; updated daily
free account key:
  alpaca      1-minute stock bars since 2016 and option TRADE bars since Feb 2024 (no
              historical option quotes on the free plan - trade prints only)

Every source is incremental (sessions already stored are skipped), caches raw downloads,
retries with backoff on rate limits, validates rows, and records provenance. None of them
ever falls back to simulated data: a failed download raises.
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from strata.config import get_settings
from strata.core import calendar as cal
from strata.data.store import INTRADAY_RES, KNOWN_SPECS, MarketStore, SymbolMeta

UA = {"User-Agent": "Mozilla/5.0 (compatible; strata-research/2.0)"}
YAHOO_ALIAS = {"VIX": "^VIX", "SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "VXN": "^VXN", "RVX": "^RVX", "DJI": "^DJI", "VIX9D": "^VIX9D", "VVIX": "^VVIX"}
INDEX_SPECS = {"VIX": "volatility_index", "VIX9D": "volatility_index", "VVIX": "volatility_index", "VXN": "volatility_index", "RVX": "volatility_index",
               "SPX": "index", "NDX": "index", "RUT": "index", "DJI": "index"}


class SourceError(Exception):
    pass


class Cancelled(Exception):
    pass


def _client() -> httpx.Client:
    return httpx.Client(timeout=90, headers=UA, follow_redirects=True)


def http_get(client: httpx.Client, url: str, params: dict | None = None, headers: dict | None = None, retries: int = 4) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.get(url, params=params, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("retry-after") or 0) or 1.5 * (2 ** attempt)
                time.sleep(min(wait, 30))
                last = SourceError(f"HTTP {r.status_code} from {httpx.URL(url).host}")
                continue
            if r.status_code >= 400:
                raise SourceError(f"HTTP {r.status_code} from {httpx.URL(url).host}: {r.text[:200]}")
            return r
        except httpx.HTTPError as e:
            last = e
            time.sleep(1 + attempt)
    raise SourceError(f"request failed after {retries} attempts: {last}")


def ensure_symbol(store: MarketStore, sym: str, has_options: bool | None = None) -> SymbolMeta:
    m = store.symbols().get(sym)
    if m is None:
        spec = dict(KNOWN_SPECS.get(sym, {}))
        if sym in INDEX_SPECS and "kind" not in spec:
            spec.update(kind=INDEX_SPECS[sym], exercise="european", settlement="cash", has_options=False)
        m = SymbolMeta(symbol=sym, provider="", has_options=bool(has_options) if has_options is not None else spec.pop("has_options", False), **spec)
        if sym not in KNOWN_SPECS and sym not in INDEX_SPECS:
            m.kind, m.vol_index = "equity", "VIX"
    if has_options:
        m.has_options = True
    store.upsert_symbol(m)
    return m


def _finish(store: MarketStore, sym: str):
    from strata.data.catalog import invalidate
    store.rebuild_manifest(sym)
    invalidate(store)


class Source:
    name = "base"
    label = "Base"
    tier = "free"  # free | free_key | paid
    license = ""
    url = ""
    description = ""
    datasets: list[dict] = []

    def __init__(self, store: MarketStore | None = None):
        self.store = store or MarketStore()
        self.cache = get_settings().cache_dir / self.name
        self.cache.mkdir(parents=True, exist_ok=True)

    def available(self) -> tuple[bool, str]:
        return True, "no key required"

    def info(self) -> dict:
        ok, msg = self.available()
        return {"name": self.name, "label": self.label, "tier": self.tier, "license": self.license, "url": self.url, "description": self.description,
                "datasets": self.datasets, "available": ok, "status": msg}

    def plan(self, dataset: str, symbols: list[str], start: date | None, end: date | None) -> dict:
        raise NotImplementedError

    def ingest(self, dataset: str, symbols: list[str], start: date | None, end: date | None, progress=None, should_cancel=None, force: bool = False) -> dict:
        raise NotImplementedError


# ============================================================================ Yahoo
class Yahoo(Source):
    name, label, tier = "yahoo", "Yahoo Finance", "free"
    url = "https://finance.yahoo.com"
    license = "Yahoo terms of service (personal research use)"
    description = "Underlying prices: full daily history plus recent intraday bars. No option history."
    datasets = [
        {"id": "daily", "label": "Daily bars, dividends & splits", "resolution": "1d", "coverage": "full listing history"},
        {"id": "intraday", "label": "Intraday bars", "resolution": "60m / 5m / 1m", "coverage": "60m ~730 days, 5m ~60 days, 1m ~30 days (rolling)"},
    ]
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"

    @staticmethod
    def ysym(sym: str) -> str:
        return YAHOO_ALIAS.get(sym, sym.replace(".", "-"))

    def _chart(self, client, sym: str, **params) -> dict:
        r = http_get(client, self.BASE + self.ysym(sym), params={"includePrePost": "false", **params})
        c = r.json().get("chart", {})
        if c.get("error"):
            raise SourceError(f"Yahoo: {c['error'].get('description', c['error'])}")
        if not c.get("result"):
            raise SourceError(f"Yahoo returned no data for {sym}")
        return c["result"][0]

    @staticmethod
    def parse_daily(res: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
        ts = res.get("timestamp") or []
        q = res["indicators"]["quote"][0]
        rows = []
        for i, t in enumerate(ts):
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            if None in (o, h, l, c) or h < l or c <= 0:
                continue
            d = datetime.fromtimestamp(t, tz=cal.ET).date()
            rows.append({"date": d, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(q["volume"][i] or 0)})
        ev = res.get("events") or {}
        ca = [{"ex_date": datetime.fromtimestamp(int(v["date"]), tz=cal.ET).date(), "action": "dividend", "ratio": None, "amount": float(v["amount"])} for v in (ev.get("dividends") or {}).values()]
        ca += [{"ex_date": datetime.fromtimestamp(int(v["date"]), tz=cal.ET).date(), "action": "split", "ratio": float(v["numerator"]) / float(v["denominator"]), "amount": None}
               for v in (ev.get("splits") or {}).values() if v.get("denominator")]
        daily = pl.DataFrame(rows, schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}).unique("date", keep="last").sort("date")
        actions = pl.DataFrame(ca, schema={"ex_date": pl.Date, "action": pl.String, "ratio": pl.Float64, "amount": pl.Float64})
        return daily, actions

    @staticmethod
    def parse_intraday(res: dict, res_seconds: int, now: float | None = None) -> pl.DataFrame:
        ts = res.get("timestamp") or []
        q = res["indicators"]["quote"][0]
        now = now or time.time()
        bounds: dict[date, tuple[int, int]] = {}
        rows = []
        for i, t in enumerate(ts):
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            if None in (o, h, l, c) or h < l:
                continue
            d = datetime.fromtimestamp(t, tz=cal.ET).date()
            if d not in bounds:
                if not cal.is_trading_day(d):
                    continue
                b = cal.session_bounds(d)
                bounds[d] = (int(b[0].timestamp()), int(b[1].timestamp()))
            lo, hi = bounds[d]
            if not (lo <= t < hi) or min(t + res_seconds, hi) > now:  # outside the session or bar not complete yet
                continue
            rows.append((t, float(o), float(h), float(l), float(c), float(q["volume"][i] or 0)))
        df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))).unique("ts", keep="last").sort("ts")

    def plan(self, dataset, symbols, start, end):
        n = len(symbols)
        if dataset == "daily":
            return {"requests": n, "est_bytes": 420_000 * n, "est_seconds": 3 * n, "note": "Full daily history per symbol."}
        return {"requests": 7 * n, "est_bytes": 1_300_000 * n, "est_seconds": 12 * n, "note": "Rolling windows: Yahoo only serves recent intraday history; re-run periodically to accumulate."}

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        with _client() as client:
            for k, sym in enumerate(symbols):
                if should_cancel and should_cancel():
                    raise Cancelled()
                if progress:
                    progress(k / max(len(symbols), 1), f"Yahoo {dataset}: {sym}")
                ensure_symbol(self.store, sym)
                if dataset == "daily":
                    res = self._chart(client, sym, interval="1d", period1=0, period2=9999999999, events="div,splits")
                    daily, actions = self.parse_daily(res)
                    if daily.is_empty():
                        raise SourceError(f"Yahoo returned no daily bars for {sym}")
                    today = datetime.now(cal.ET)
                    if today.date() in daily["date"] and today < cal.session_bounds(today.date())[1]:
                        daily = daily.filter(pl.col("date") < today.date())  # session still open
                    self.store.write_daily(sym, daily)
                    if not actions.is_empty():
                        self.store.write_corporate_actions(sym, actions)
                    self.store.note_source(sym, "underlying", self.name, "1d", self.license, "trades", self.url)
                    m = self.store.symbol_meta(sym)
                    m.extra = {**m.extra, "price_adjustment": "split"}  # Yahoo OHLC is split-adjusted (not dividend-adjusted)
                    self.store.upsert_symbol(m)
                    out["symbols"][sym] = {"daily_bars": daily.height, "first": daily["date"][0].isoformat(), "last": daily["date"][-1].isoformat(), "corporate_actions": actions.height}
                else:
                    rep = {}
                    for res_name, rng in (("60m", "730d"), ("5m", "60d")):
                        try:
                            df = self.parse_intraday(self._chart(client, sym, interval=res_name, range=rng), INTRADAY_RES[res_name])
                            if not df.is_empty():
                                self.store.write_bars(sym, res_name, df)
                                self.store.note_source(sym, "underlying", self.name, res_name, self.license, "trades", self.url)
                                rep[res_name] = df.height
                        except SourceError as e:
                            rep[res_name] = f"unavailable: {e}"
                    frames = []
                    now = int(time.time())
                    for back in range(0, 28, 7):
                        p2, p1 = now - back * 86400, now - (back + 7) * 86400
                        try:
                            frames.append(self.parse_intraday(self._chart(client, sym, interval="1m", period1=p1, period2=p2), 60))
                        except SourceError:
                            break
                    frames = [f for f in frames if not f.is_empty()]
                    if frames:
                        m = pl.concat(frames).unique("ts", keep="last").sort("ts")
                        self.store.write_bars(sym, "1m", m)
                        self.store.note_source(sym, "underlying", self.name, "1m", self.license, "trades", self.url)
                        rep["1m"] = m.height
                    out["symbols"][sym] = rep
                _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ CBOE
class Cboe(Source):
    name, label, tier = "cboe", "Cboe", "free"
    url = "https://www.cboe.com/tradable_products/vix/vix_historical_data/"
    license = "Cboe published historical index data"
    description = "Official daily history of the VIX family of volatility indices."
    datasets = [{"id": "vix_daily", "label": "VIX daily OHLC", "resolution": "1d", "coverage": "1990 to the last close"}]
    FILES = {"VIX": "VIX_History.csv", "VIX9D": "VIX9D_History.csv"}

    @staticmethod
    def parse(text: str) -> pl.DataFrame:
        df = pl.read_csv(io.StringIO(text))
        df = df.rename({c: c.strip().lower() for c in df.columns})
        df = df.with_columns(pl.col("date").str.to_date("%m/%d/%Y"), pl.lit(0.0).alias("volume"))
        return df.select(["date", "open", "high", "low", "close", "volume"]).drop_nulls().filter(pl.col("close") > 0).unique("date", keep="last").sort("date")

    def plan(self, dataset, symbols, start, end):
        return {"requests": 1, "est_bytes": 450_000, "est_seconds": 3, "note": "Single CSV."}

    def ingest(self, dataset, symbols=None, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        with _client() as client:
            for sym in [s for s in (symbols or ["VIX"]) if s in self.FILES] or ["VIX"]:
                if progress:
                    progress(0.2, f"Cboe {sym} history")
                df = self.parse(http_get(client, f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{self.FILES[sym]}").text)
                ensure_symbol(self.store, sym)
                self.store.write_daily(sym, df)
                self.store.note_source(sym, "underlying", self.name, "1d", self.license, "index", self.url)
                _finish(self.store, sym)
                out["symbols"][sym] = {"daily_bars": df.height, "first": df["date"][0].isoformat(), "last": df["date"][-1].isoformat()}
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ GitHub EOD options archive
def clean_chain(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Validation shared by the option sources: drop crossed/negative quotes, null out
    implausible vendor IV/Greeks (the engine recomputes them from the observed quotes)."""
    n0 = df.height
    bad = (pl.col("ask") < pl.col("bid")) | (pl.col("bid") < 0) | (pl.col("strike") <= 0)
    crossed = df.filter(bad).height
    df = df.filter(~bad)
    susp = pl.col("iv").is_null() | (pl.col("iv") <= 0.03) | (pl.col("iv") > 5) | pl.col("delta").is_null() | (pl.col("delta").abs() > 1)
    n_susp = df.filter(susp & pl.col("iv").is_not_null()).height
    greeks = [c for c in ("iv", "delta", "gamma", "theta", "vega") if c in df.columns]
    df = df.with_columns([pl.when(susp).then(None).otherwise(pl.col(c)).alias(c) for c in greeks])
    return df, {"input_rows": n0, "rejected_crossed": crossed, "greeks_recomputed_rows": n_susp}


class GithubEod(Source):
    name, label, tier = "github_eod", "EOD options archive (SPY/QQQ/IWM)", "free"
    REPO = "anahatsingh-ui/options-dataset-hist"
    url = f"https://github.com/{REPO}"
    license = "MIT (c) 2025 Philipp D. Dubach; preservation mirror"
    description = "End-of-day option chains 2008-2025: every listed contract with bid/ask, sizes, volume, open interest, IV and Greeks (16:00 ET snapshot)."
    datasets = [{"id": "options_eod", "label": "Full EOD chains", "resolution": "EOD", "coverage": "SPY & IWM 2008-2025, QQQ 2011-2025", "quote_type": "quotes"}]
    SYMBOLS = ("SPY", "QQQ", "IWM")

    def _tree(self, client) -> dict[str, int]:
        cp = self.cache / "tree.json"
        if cp.exists() and time.time() - cp.stat().st_mtime < 86400:
            return json.loads(cp.read_text())
        r = http_get(client, f"https://api.github.com/repos/{self.REPO}/git/trees/HEAD", params={"recursive": "1"}, headers={"Accept": "application/vnd.github+json"})
        tree = {t["path"]: t.get("size", 0) for t in r.json().get("tree", []) if t.get("type") == "blob"}
        cp.write_text(json.dumps(tree))
        return tree

    def years(self, client, sym: str) -> dict[int, int]:
        pre = f"{sym.lower()}/options_"
        return {int(p[len(pre):len(pre) + 4]): s for p, s in self._tree(client).items() if p.startswith(pre) and p.endswith(".parquet")}

    def plan(self, dataset, symbols, start, end):
        with _client() as client:
            total, files, per = 0, 0, {}
            for sym in symbols:
                if sym not in self.SYMBOLS:
                    raise SourceError(f"{self.label} covers {', '.join(self.SYMBOLS)} only")
                ys = {y: s for y, s in self.years(client, sym).items() if (not start or y >= start.year) and (not end or y <= end.year)}
                have = {d.year for d in self.store.option_days(sym)}
                per[sym] = {"years": sorted(ys), "download_bytes": sum(ys.values()), "years_already_stored": sorted(set(ys) & have)}
                total += sum(ys.values())
                files += len(ys)
        return {"requests": files, "est_bytes": int(total * 2.1), "download_bytes": total, "est_seconds": int(files * 25 + total / 2.5e6), "detail": per,
                "note": "Download size shown; the normalised per-session store takes roughly the same again."}

    def _download(self, client, path: str, size: int, progress=None, label="") -> Path:
        dest = self.cache / path.replace("/", "_")
        if dest.exists() and (size == 0 or dest.stat().st_size == size):
            return dest
        part = dest.with_suffix(".part")
        got = 0
        with client.stream("GET", f"https://raw.githubusercontent.com/{self.REPO}/HEAD/{path}") as r:
            if r.status_code >= 400:
                raise SourceError(f"download failed ({r.status_code}) for {path}")
            with part.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if progress and size:
                        progress(None, f"{label} downloading {got / 1e6:.0f}/{size / 1e6:.0f} MB")
        if size and got != size:
            part.unlink(missing_ok=True)
            raise SourceError(f"incomplete download for {path}: {got} of {size} bytes")
        part.rename(dest)
        return dest

    @staticmethod
    def normalize(lf: pl.LazyFrame, underlying: pl.DataFrame | None) -> pl.DataFrame:
        df = lf.select(
            pl.col("date").str.to_date().alias("_d"), pl.col("expiration").str.to_date(), pl.col("strike").cast(pl.Float64),
            pl.col("type").str.slice(0, 1).str.to_uppercase().alias("right"), "bid", "ask",
            pl.col("bid_size").cast(pl.Int32, strict=False), pl.col("ask_size").cast(pl.Int32, strict=False), "last", "volume", "open_interest",
            pl.col("implied_volatility").alias("iv"), "delta", "gamma", "theta", "vega",
        ).collect()
        df = df.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("expiration") >= pl.col("_d")))
        if underlying is not None:
            df = df.join(underlying, on="_d", how="left")
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("underlying_price"))
        return df

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        with _client() as client:
            jobs = []
            for sym in symbols:
                if sym not in self.SYMBOLS:
                    raise SourceError(f"{self.label} covers {', '.join(self.SYMBOLS)} only")
                for y, size in sorted(self.years(client, sym).items()):
                    if (not start or y >= start.year) and (not end or y <= end.year):
                        jobs.append((sym, y, size))
            if not jobs:
                raise SourceError("no files match the requested symbols / years")
            und_cache: dict[str, pl.DataFrame] = {}
            for k, (sym, y, size) in enumerate(jobs):
                if should_cancel and should_cancel():
                    raise Cancelled()
                base = k / len(jobs)
                prog = (lambda f, m, b=base: progress(b if f is None else b + f / len(jobs), m)) if progress else None
                rep = out["symbols"].setdefault(sym, {"sessions_written": 0, "sessions_skipped": 0, "rows": 0, "rejected_crossed": 0, "greeks_recomputed_rows": 0, "years": []})
                ensure_symbol(self.store, sym, has_options=True)
                have = set(self.store.option_days(sym))
                expected = [d for d in cal.trading_days(max(date(y, 1, 1), start or date(y, 1, 1)), min(date(y, 12, 31), end or date(y, 12, 31)))]
                if not force and expected and all(d in have for d in expected):
                    rep["sessions_skipped"] += len(expected)
                    continue
                f = self._download(client, f"{sym.lower()}/options_{y}.parquet", size, prog, f"{sym} {y}")
                if sym not in und_cache:
                    uf = self._download(client, f"{sym.lower()}/underlying_prices.parquet", 0, None)
                    u = pl.read_parquet(uf).select(pl.col("date").str.to_date().alias("_d"), pl.col("close").alias("underlying_price"))
                    und_cache[sym] = u
                    if not self.store.daily_path(sym).exists():  # usable on its own; Yahoo daily can extend it later
                        ud = pl.read_parquet(uf).select(pl.col("date").str.to_date(), "open", "high", "low", "close", pl.col("volume").cast(pl.Float64))
                        self.store.write_daily(sym, ud)
                        self.store.note_source(sym, "underlying", self.name, "1d", self.license, "trades", self.url)
                for mth in range(1, 13):
                    if should_cancel and should_cancel():
                        raise Cancelled()
                    if prog:
                        prog(mth / 13, f"{sym} {y}-{mth:02d} normalising")
                    lf = pl.scan_parquet(f).filter(pl.col("date").str.starts_with(f"{y}-{mth:02d}"))
                    df = self.normalize(lf, und_cache[sym])
                    if df.is_empty():
                        continue
                    df, q = clean_chain(df)
                    rep["rejected_crossed"] += q["rejected_crossed"]
                    rep["greeks_recomputed_rows"] += q["greeks_recomputed_rows"]
                    for (d,), part in df.partition_by("_d", as_dict=True).items():
                        if (start and d < start) or (end and d > end) or not cal.is_trading_day(d):
                            continue
                        if d in have and not force:
                            rep["sessions_skipped"] += 1
                            continue
                        close = cal.session_bounds(d)[1]
                        part = part.drop("_d").with_columns(pl.lit(close).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")).alias("ts"))
                        self.store.write_option_day(sym, d, part.with_columns(pl.col("ts").alias("quote_ts")))
                        rep["sessions_written"] += 1
                        rep["rows"] += part.height
                rep["years"].append(y)
            for sym in out["symbols"]:
                m = self.store.symbol_meta(sym)
                m.option_interval_minutes = None
                self.store.upsert_symbol(m)
                self.store.note_source(sym, "options", self.name, "EOD", self.license, "quotes", self.url)
                if progress:
                    progress(0.97, f"{sym}: deriving IV history")
                from strata.data.derived import derive_iv_history
                derive_iv_history(self.store, sym)
                _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ DoltHub
class DoltHub(Source):
    name, label, tier = "dolthub", "DoltHub options", "free"
    url = "https://www.dolthub.com/repositories/post-no-preference/options"
    license = "CC BY-SA 4.0 (post-no-preference/options)"
    description = "End-of-day option chains since 2019 for ~2,000 US symbols: bid/ask, IV and Greeks. Four expirations (~2-8 weeks out), ~27 strikes each. No volume/open interest."
    datasets = [{"id": "options_eod", "label": "EOD chains (sparse)", "resolution": "EOD", "coverage": "2019-02 to the last close", "quote_type": "quotes"}]
    API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
    FIRST = date(2019, 2, 9)

    def _q(self, client, sql: str) -> list[dict]:
        r = http_get(client, self.API, params={"q": sql})
        d = r.json()
        st = d.get("query_execution_status")
        if st not in ("Success", "RowLimit"):
            raise SourceError(f"DoltHub query failed: {d.get('query_execution_message') or st}")
        rows = d.get("rows") or []
        return rows if st == "Success" else rows + [{"__limit__": True}]

    def latest_date(self, client) -> date:
        return date.fromisoformat(self._q(client, "SELECT MAX(date) AS d FROM option_chain")[0]["d"])

    def plan(self, dataset, symbols, start, end):
        start = max(start or date.today() - timedelta(days=90), self.FIRST)
        end = end or date.today()
        days = cal.trading_days(start, end)
        todo = 0
        for sym in symbols:
            have = set(self.store.option_days(sym))
            todo += sum(1 for d in days if d not in have)
        q = (todo + 3) // 4
        return {"requests": q, "est_bytes": todo * 14_000, "est_seconds": q * 6, "note": f"{todo} sessions to fetch (4 per request, ~6 s each). The public API is rate-limited; large ranges take a while but can be resumed."}

    @staticmethod
    def normalize(rows: list[dict]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        f = lambda v: None if v in (None, "") else float(v)
        recs = [{"_d": date.fromisoformat(r["date"]), "expiration": date.fromisoformat(r["expiration"]), "strike": float(r["strike"]), "right": r["call_put"][:1].upper(),
                 "bid": f(r["bid"]), "ask": f(r["ask"]), "iv": f(r["vol"]), "delta": f(r["delta"]), "gamma": f(r["gamma"]), "theta": f(r["theta"]), "vega": f(r["vega"])} for r in rows]
        return pl.DataFrame(recs).filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        cols = "date, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega"
        with _client() as client:
            last = self.latest_date(client)
            start = max(start or last - timedelta(days=90), self.FIRST)
            end = min(end or last, last)
            days = cal.trading_days(start, end)
            total = max(1, len(days) * len(symbols))
            done = 0
            for sym in symbols:
                ensure_symbol(self.store, sym, has_options=True)
                rep = out["symbols"].setdefault(sym, {"sessions_written": 0, "sessions_skipped": 0, "sessions_empty": [], "rows": 0, "rejected_crossed": 0, "greeks_recomputed_rows": 0})
                have = set(self.store.option_days(sym))
                todo = [d for d in days if d not in have or force]
                rep["sessions_skipped"] = len(days) - len(todo)
                done += rep["sessions_skipped"]
                daily = self.store.read_daily(sym)
                und = daily.select(pl.col("date").alias("_d"), pl.col("close").alias("underlying_price")) if not daily.is_empty() else None
                for i in range(0, len(todo), 4):
                    if should_cancel and should_cancel():
                        raise Cancelled()
                    batch = todo[i:i + 4]
                    if progress:
                        progress(done / total, f"DoltHub {sym} {batch[0]} .. {batch[-1]}")
                    inl = ",".join(f"'{d.isoformat()}'" for d in batch)
                    rows = self._q(client, f"SELECT {cols} FROM option_chain WHERE date IN ({inl}) AND act_symbol='{sym}'")
                    if rows and rows[-1].get("__limit__"):  # over the API row cap: fetch each date on its own, paged
                        rows = []
                        for d in batch:
                            off = 0
                            while True:
                                page = self._q(client, f"SELECT {cols} FROM option_chain WHERE date='{d.isoformat()}' AND act_symbol='{sym}' ORDER BY expiration, strike, call_put LIMIT 1000 OFFSET {off}")
                                page = [r for r in page if not r.get("__limit__")]
                                rows += page
                                if len(page) < 1000:
                                    break
                                off += 1000
                    df = self.normalize(rows)
                    got = set()
                    if not df.is_empty():
                        df = df.join(und, on="_d", how="left") if und is not None else df.with_columns(pl.lit(None, dtype=pl.Float64).alias("underlying_price"))
                        df, q = clean_chain(df)
                        rep["rejected_crossed"] += q["rejected_crossed"]
                        rep["greeks_recomputed_rows"] += q["greeks_recomputed_rows"]
                        for (d,), part in df.partition_by("_d", as_dict=True).items():
                            close = cal.session_bounds(d)[1]
                            part = part.drop("_d").with_columns(pl.lit(close).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")).alias("ts"))
                            self.store.write_option_day(sym, d, part.with_columns(pl.col("ts").alias("quote_ts")))
                            rep["sessions_written"] += 1
                            rep["rows"] += part.height
                            got.add(d)
                    rep["sessions_empty"] += [d.isoformat() for d in batch if d not in got]
                    done += len(batch)
                rep["sessions_empty"] = rep["sessions_empty"][:60]
                self.store.note_source(sym, "options", self.name, "EOD", self.license, "quotes", self.url)
                from strata.data.derived import derive_iv_history
                derive_iv_history(self.store, sym)
                _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ Alpaca (free account)
class Alpaca(Source):
    name, label, tier = "alpaca", "Alpaca Market Data", "free_key"
    url = "https://docs.alpaca.markets/docs/about-market-data-api"
    license = "Alpaca market data agreement"
    description = "Free account: 1-minute stock bars since 2016 and option TRADE bars since Feb 2024. Historical option quotes are not offered on the free plan, so option data from Alpaca is trade-print based."
    datasets = [
        {"id": "bars_1m", "label": "1-minute stock bars", "resolution": "1m", "coverage": "2016 to 15 minutes ago"},
        {"id": "option_trades", "label": "Option 1-minute trade bars", "resolution": "1m", "coverage": "2024-02 onward", "quote_type": "trades"},
    ]
    DATA = "https://data.alpaca.markets"
    TRADING = "https://paper-api.alpaca.markets"

    def _keys(self) -> tuple[str, str]:
        from strata.api.settings import get_secret
        return get_secret("ALPACA_API_KEY") or "", get_secret("ALPACA_API_SECRET") or ""

    def available(self):
        k, s = self._keys()
        return (bool(k and s), "key configured" if k and s else "add a free Alpaca API key and secret in Settings")

    def _h(self) -> dict:
        k, s = self._keys()
        if not (k and s):
            raise SourceError("Alpaca API key/secret not configured (Settings -> Data providers)")
        return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

    @staticmethod
    def parse_bars(bars: list[dict]) -> pl.DataFrame:
        rows = [(b["t"], float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]), float(b.get("v", 0))) for b in bars]
        if not rows:
            return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64})
        df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC").cast(pl.Datetime("us", "UTC")))

    @staticmethod
    def parse_occ(sym: str) -> tuple[date, str, float]:
        body = sym[-15:]
        return date(2000 + int(body[0:2]), int(body[2:4]), int(body[4:6])), body[6], int(body[7:]) / 1000.0

    def plan(self, dataset, symbols, start, end):
        days = len(cal.trading_days(start or date.today() - timedelta(days=30), end or date.today()))
        if dataset == "bars_1m":
            return {"requests": max(1, days * 390 // 10000 + 1) * len(symbols), "est_bytes": days * 9000 * len(symbols), "est_seconds": days // 20 + 5, "note": "Regular-session 1-minute bars (SIP, 15-minute delayed on the free plan)."}
        return {"requests": days * 6 * len(symbols), "est_bytes": days * 400_000 * len(symbols), "est_seconds": days * 8, "note": "Trade bars for contracts within 20% of spot and 45 DTE. Trade prints only: no bid/ask."}

    def _paged(self, client, url, params, key):
        out, token = [], None
        while True:
            p = dict(params)
            if token:
                p["page_token"] = token
            d = http_get(client, url, params=p, headers=self._h()).json()
            chunk = d.get(key) or []
            if isinstance(chunk, dict):
                out.append(chunk)
            else:
                out += chunk
            token = d.get("next_page_token")
            if not token:
                return out

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        start = start or date.today() - timedelta(days=30)
        end = end or date.today()
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        with _client() as client:
            for sym in symbols:
                ensure_symbol(self.store, sym)
                days = cal.trading_days(start, end)
                if dataset == "bars_1m":
                    have = set()
                    m = self.store.read_bars(sym, "1m", start, end)
                    if not m.is_empty():
                        have = set(m["ts"].dt.convert_time_zone("America/New_York").dt.date().unique().to_list())
                    todo = [d for d in days if d not in have or force]
                    n = 0
                    for i in range(0, len(todo), 20):
                        if should_cancel and should_cancel():
                            raise Cancelled()
                        chunk = todo[i:i + 20]
                        if progress:
                            progress(i / max(len(todo), 1), f"Alpaca {sym} {chunk[0]} .. {chunk[-1]}")
                        a, b = cal.session_bounds(chunk[0])[0], cal.session_bounds(chunk[-1])[1]
                        bars = self._paged(client, f"{self.DATA}/v2/stocks/{sym}/bars", {"timeframe": "1Min", "start": a.isoformat(), "end": min(b, datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(), "limit": 10000, "adjustment": "split", "feed": "sip"}, "bars")
                        df = self.parse_bars(bars)
                        if df.is_empty():
                            continue
                        df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et")).filter(
                            (pl.col("_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_et").dt.minute().cast(pl.Int32)).is_between(570, 959)).drop("_et")
                        n += self.store.write_bars(sym, "1m", df) and df.height
                    self.store.note_source(sym, "underlying", self.name, "1m", self.license, "trades", self.url)
                    out["symbols"][sym] = {"bars": n, "sessions": len(todo)}
                else:
                    rep = {"sessions_written": 0, "rows": 0}
                    have = set(self.store.option_days(sym))
                    daily = self.store.read_daily(sym)
                    for k, d in enumerate(days):
                        if should_cancel and should_cancel():
                            raise Cancelled()
                        if d in have and not force:
                            continue
                        if progress:
                            progress(k / max(len(days), 1), f"Alpaca {sym} option trades {d}")
                        spot = daily.filter(pl.col("date") <= d)["close"][-1] if not daily.is_empty() and daily.filter(pl.col("date") <= d).height else None
                        params = {"underlying_symbols": sym, "expiration_date_gte": d.isoformat(), "expiration_date_lte": (d + timedelta(days=45)).isoformat(), "limit": 10000, "status": "inactive" if d < date.today() - timedelta(days=45) else "active"}
                        if spot:
                            params.update(strike_price_gte=round(spot * 0.8, 0), strike_price_lte=round(spot * 1.2, 0))
                        contracts = self._paged(client, f"{self.TRADING}/v2/options/contracts", params, "option_contracts")
                        occ = [c["symbol"] for c in contracts]
                        a, b = cal.session_bounds(d)
                        frames = []
                        for j in range(0, len(occ), 100):
                            d_ = http_get(client, f"{self.DATA}/v1beta1/options/bars", params={"symbols": ",".join(occ[j:j + 100]), "timeframe": "1Min", "start": a.isoformat(), "end": b.isoformat(), "limit": 10000}, headers=self._h()).json()
                            for osym, bars in (d_.get("bars") or {}).items():
                                exp, right, strike = self.parse_occ(osym)
                                bdf = self.parse_bars(bars)
                                if not bdf.is_empty():
                                    frames.append(bdf.select((pl.col("ts") + pl.duration(minutes=1)).alias("ts"), pl.lit(exp).alias("expiration"), pl.lit(strike).alias("strike"), pl.lit(right).alias("right"),
                                                             pl.col("close").alias("last"), pl.col("volume").cast(pl.Int64)))
                        if frames:
                            df = pl.concat(frames)
                            self.store.write_option_day(sym, d, df.with_columns(pl.col("ts").alias("quote_ts")))
                            rep["sessions_written"] += 1
                            rep["rows"] += df.height
                    m = self.store.symbol_meta(sym)
                    m.has_options, m.option_interval_minutes = True, 1
                    self.store.upsert_symbol(m)
                    self.store.note_source(sym, "options", self.name, "1m", self.license, "trades", self.url)
                    out["symbols"][sym] = rep
                _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


SOURCES: dict[str, type[Source]] = {s.name: s for s in (Yahoo, Cboe, GithubEod, DoltHub, Alpaca)}


def get_source(name: str, store: MarketStore | None = None) -> Source:
    if name not in SOURCES:
        raise SourceError(f"unknown data source {name!r}; available: {', '.join(SOURCES)}")
    return SOURCES[name](store)


def quickstart(symbols: list[str] | None = None, store: MarketStore | None = None, progress=None, should_cancel=None, intraday: bool = True) -> dict:
    """First-run download: real daily history (+recent intraday) for a few symbols and VIX.
    Small (a few MB), keyless, and enough to run market studies immediately."""
    store = store or MarketStore()
    symbols = symbols or ["SPY", "QQQ", "IWM"]
    steps = [("cboe", "vix_daily", ["VIX"]), ("yahoo", "daily", symbols)] + ([("yahoo", "intraday", symbols + ["VIX"])] if intraday else [])
    out = {"steps": []}
    for i, (src, ds, syms) in enumerate(steps):
        sub = (lambda f, m, i=i: progress((i + (f or 0)) / len(steps), m)) if progress else None
        out["steps"].append(get_source(src, store).ingest(ds, syms, progress=sub, should_cancel=should_cancel))
    return out


def verify_daily(store: MarketStore, sym: str = "SPY") -> dict:
    """Cross-check stored daily closes against an independent source (the EOD options
    archive ships its own underlying price table)."""
    src = GithubEod(store)
    with _client() as client:
        uf = src._download(client, f"{sym.lower()}/underlying_prices.parquet", 0, None)
    ref = pl.read_parquet(uf).select(pl.col("date").str.to_date(), pl.col("close").alias("ref_close"), pl.col("open").alias("ref_open"))
    mine = store.read_daily(sym).select("date", "close", "open")
    j = mine.join(ref, on="date", how="inner").with_columns(((pl.col("close") / pl.col("ref_close") - 1).abs() * 100).alias("diff_pct"))
    if j.is_empty():
        return {"symbol": sym, "compared": 0}
    worst = j.sort("diff_pct", descending=True).head(3)
    return {"symbol": sym, "compared": j.height, "first": j["date"][0].isoformat(), "last": j["date"][-1].isoformat(), "max_close_diff_pct": float(j["diff_pct"].max()),
            "mean_close_diff_pct": float(j["diff_pct"].mean()), "within_0_1pct": float((j["diff_pct"] < 0.1).mean()),
            "worst": [{"date": r["date"].isoformat(), "stored": r["close"], "reference": r["ref_close"]} for r in worst.iter_rows(named=True)],
            "reference": "EOD options archive underlying_prices.parquet (independent of Yahoo)"}
