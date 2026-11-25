"""Free-first market-data sources.

keyless (work out of the box):
  yahoo          underlying daily history (decades, with dividends/splits) and recent intraday
                 bars: 60m for ~730 days, 5m for ~60 days, 1m for ~30 days
  cboe           official VIX daily history since 1990
  github_eod     end-of-day option chains for SPY / QQQ / IWM, 2008-2025 (MIT-licensed
                 research dataset by P. Dubach, mirrored on GitHub): bid/ask, sizes, volume,
                 open interest, IV and Greeks at the 16:00 ET snapshot
  dolthub        end-of-day option chains for ~2,000 US symbols since 2019 (DoltHub
                 post-no-preference/options, CC BY-SA 4.0): bid/ask, IV, Greeks; four
                 expirations (~2-8 weeks) and ~27 strikes per expiration; updated daily
  yahoo_options  1-minute option TRADE bars for the last ~30 days, only for contracts that are
                 still listed (or expired within about a day). No bid/ask.
free account key:
  alpaca         1-minute stock bars since 2016 and 1-minute option TRADE bars since Feb 2024
                 (no historical option quotes on the free plan - trade prints only)

Guarantees shared by every source
  * provenance: each dataset is stored under its own provider partition with a descriptor
  * resumable: work already recorded as COMPLETE is skipped; a session or a contract is only
    recorded as complete after the request covering it has been consumed to the last page
  * no look-ahead in contract discovery: the strike universe for a session is derived from
    prices that were known before the decision (the previous completed session, or the
    price at the signal time supplied by the planner) - never from that session's close
  * no simulated data: a failed download raises or is reported; nothing is substituted
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from strata.config import get_settings
from strata.core import calendar as cal
from strata.data.store import INTRADAY_RES, KNOWN_SPECS, TRADE_SCHEMA, MarketStore, SymbolMeta

UA = {"User-Agent": "Mozilla/5.0 (compatible; strata-research/3.0)"}
YAHOO_ALIAS = {"VIX": "^VIX", "SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "VXN": "^VXN", "RVX": "^RVX", "DJI": "^DJI", "VIX9D": "^VIX9D", "VVIX": "^VVIX"}
INDEX_SPECS = {"VIX": "volatility_index", "VIX9D": "volatility_index", "VVIX": "volatility_index", "VXN": "volatility_index", "RVX": "volatility_index",
               "SPX": "index", "NDX": "index", "RUT": "index", "DJI": "index"}
ALPACA_OPTIONS_FIRST = date(2024, 2, 1)
# listed strike spacing near the money for short-dated contracts (used only when a provider
# has no contract-definition endpoint; every constructed symbol is verified by the fetch)
STRIKE_STEP = {"SPY": 1.0, "QQQ": 1.0, "IWM": 1.0}


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


class Throttle:
    """At most `per_second` calls per second (sources call .wait() before each request)."""

    def __init__(self, per_second: float):
        self.gap = 1.0 / per_second if per_second > 0 else 0.0
        self.last = 0.0

    def wait(self):
        now = time.monotonic()
        if self.gap and now - self.last < self.gap:
            time.sleep(self.gap - (now - self.last))
        self.last = time.monotonic()


def ensure_symbol(store: MarketStore, sym: str, has_options: bool | None = None) -> SymbolMeta:
    m = store.symbols().get(sym)
    if m is None:
        spec = dict(KNOWN_SPECS.get(sym, {}))
        if sym in INDEX_SPECS and "kind" not in spec:
            spec.update(kind=INDEX_SPECS[sym], exercise="european", settlement="cash", has_options=False)
        m = SymbolMeta(symbol=sym, has_options=bool(has_options) if has_options is not None else spec.pop("has_options", False), **spec)
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


# ============================================================================ option contracts
@dataclass(frozen=True)
class Contract:
    """An option contract definition, with the provider's own identifiers preserved."""
    occ: str
    root: str
    expiration: date
    strike: float
    right: str  # C | P
    multiplier: float = 100.0
    style: str = "american"
    adjusted: bool = False  # non-standard deliverable
    provider_id: str | None = None
    status: str | None = None  # the provider's status when the definition was fetched (informational only)
    source: str = ""
    first_trade: date | None = None

    def row(self, underlying: str) -> dict:
        return {"occ": self.occ, "root": self.root, "underlying": underlying, "expiration": self.expiration,
                "expiration_ts": cal.session_bounds(self.expiration)[1].astimezone(timezone.utc) if cal.is_trading_day(self.expiration) else None,
                "strike": self.strike, "right": self.right, "style": self.style, "multiplier": self.multiplier, "adjusted": self.adjusted,
                "provider_id": self.provider_id, "status_at_fetch": self.status, "first_trade": self.first_trade, "source": self.source,
                "fetched_at": datetime.now(timezone.utc)}


def occ_symbol(root: str, exp: date, right: str, strike: float) -> str:
    return f"{root}{exp:%y%m%d}{right[0].upper()}{int(round(strike * 1000)):08d}"


def parse_occ(sym: str) -> tuple[str, date, str, float]:
    """OCC symbol -> (root, expiration, right, strike). The root is kept verbatim: a digit
    suffix (e.g. SPY1) marks an adjusted contract with a non-standard deliverable."""
    m = re.fullmatch(r"([A-Z][A-Z0-9.]{0,5}?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})", sym.strip().upper())
    if not m:
        raise SourceError(f"not an OCC option symbol: {sym!r}")
    root, yy, mm, dd, right, k = m.groups()
    return root, date(2000 + int(yy), int(mm), int(dd)), right, int(k) / 1000.0


def contract_from_occ(sym: str, source: str = "constructed", **kw) -> Contract:
    root, exp, right, strike = parse_occ(sym)
    return Contract(sym, root, exp, strike, right, adjusted=bool(re.search(r"\d$", root)), source=source, **kw)


def validate_trade_bars(df: pl.DataFrame, require_volume: bool = False) -> tuple[pl.DataFrame, dict]:
    """Validation shared by the option trade-bar sources. Bars must lie inside the regular
    session, be internally consistent and unique per (contract, minute). Returns the clean
    frame and what was rejected - nothing is repaired or filled in."""
    stats = {"input_rows": df.height, "outside_session": 0, "invalid_price": 0, "duplicates": 0, "unconfirmed_zero_volume": 0}
    if df.is_empty():
        return df, stats
    df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et"))
    df = df.with_columns(pl.col("_et").dt.date().alias("_d"), (pl.col("_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_et").dt.minute().cast(pl.Int32)).alias("_m"))
    days = df["_d"].unique().to_list()
    closes = pl.DataFrame({"_d": days, "_c": [(cal.session_bounds(d)[1].hour * 60 + cal.session_bounds(d)[1].minute) if cal.is_trading_day(d) else 0 for d in days]}, schema={"_d": pl.Date, "_c": pl.Int32})
    df = df.join(closes, on="_d", how="left")
    inside = (pl.col("_m") >= 570) & (pl.col("_m") < pl.col("_c"))
    stats["outside_session"] = df.filter(~inside).height
    df = df.filter(inside)
    ok = (pl.col("close") > 0) & (pl.col("high") >= pl.col("low")) & (pl.col("low") > 0) & (pl.col("high") >= pl.col("close")) & (pl.col("low") <= pl.col("close"))
    stats["invalid_price"] = df.filter(~ok).height
    df = df.filter(ok)
    if require_volume:
        stats["unconfirmed_zero_volume"] = df.filter(pl.col("volume").fill_null(0) <= 0).height
        df = df.filter(pl.col("volume").fill_null(0) > 0)
    n = df.height
    df = df.unique(subset=["occ", "ts"], keep="last")
    stats["duplicates"] = n - df.height
    return df.drop("_et", "_d", "_m", "_c").sort(["occ", "ts"]), stats


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

    def capabilities(self, today: date | None = None) -> list[dict]:
        """What this source can deliver, as data ranges the planner can reason about."""
        return []

    def info(self) -> dict:
        ok, msg = self.available()
        caps = [{**c, "first": c["first"].isoformat() if c.get("first") else None, "last": c["last"].isoformat() if c.get("last") else None} for c in self.capabilities()]
        return {"name": self.name, "label": self.label, "tier": self.tier, "license": self.license, "url": self.url, "description": self.description,
                "datasets": self.datasets, "available": ok, "status": msg, "capabilities": caps}

    def plan(self, dataset: str, symbols: list[str], start: date | None, end: date | None) -> dict:
        raise NotImplementedError

    def ingest(self, dataset: str, symbols: list[str], start: date | None, end: date | None, progress=None, should_cancel=None, force: bool = False) -> dict:
        raise NotImplementedError

    def desc(self, **kw) -> dict:
        return {"license": self.license, "url": self.url, **kw}


def _cap(dtype, obs, res, first=None, last=None, symbols=None, rolling_days=None, note="", **kw) -> dict:
    return {"dtype": dtype, "obs": obs, "res": res, "first": first, "last": last, "symbols": symbols, "rolling_days": rolling_days, "note": note, **kw}


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
    ROLLING = {"60m": 729, "5m": 59, "1m": 29}

    def capabilities(self, today=None):
        today = today or datetime.now(cal.ET).date()
        return [_cap("underlying", "trade", "1d", None, today, note="full listing history")] + [
            _cap("underlying", "trade", r, today - timedelta(days=n), today, rolling_days=n, note=f"rolling window: only the last {n} days") for r, n in self.ROLLING.items()]

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
            return {"requests": n, "est_bytes": 420_000 * n, "est_seconds": 3 * n, "cost": 0.0, "note": "Full daily history per symbol."}
        return {"requests": 7 * n, "est_bytes": 1_300_000 * n, "est_seconds": 12 * n, "cost": 0.0, "note": "Rolling windows: Yahoo only serves recent intraday history; re-run periodically to accumulate."}

    def _mark(self, sym: str, res_name: str, df: pl.DataFrame):
        """Sessions whose close has passed and that carry bars are recorded with their bar count."""
        if df.is_empty():
            return
        now = datetime.now(timezone.utc)
        per = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d")).group_by("d").agg(pl.len().alias("n")).sort("d")
        key = self.store.ensure_dataset(self.name, sym, "underlying", "trade", res_name)
        entries = {}
        for d, n in zip(per["d"].to_list(), per["n"].to_list()):
            want = -(-cal.session_minutes(d) * 60 // INTRADAY_RES[res_name])
            closed = cal.session_bounds(d)[1].astimezone(timezone.utc) <= now
            entries[d.isoformat()] = {"status": "complete" if closed and n >= 0.97 * want else "partial", "bars": int(n), "expected": int(want)}
        self.store.mark_sessions(key, entries)

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        d_ = self.desc(adjustment="split", limitations=["OHLC is split-adjusted to today (not dividend-adjusted).", "Intraday history is a rolling window: older sessions must be captured before they age out."])
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
                    self.store.write_daily(sym, daily, provider=self.name, source="chart interval=1d", **d_)
                    if not actions.is_empty():
                        self.store.write_corporate_actions(sym, actions)
                    out["symbols"][sym] = {"daily_bars": daily.height, "first": daily["date"][0].isoformat(), "last": daily["date"][-1].isoformat(), "corporate_actions": actions.height}
                else:
                    rep = {}
                    for res_name, rng in (("60m", "730d"), ("5m", "60d")):
                        try:
                            df = self.parse_intraday(self._chart(client, sym, interval=res_name, range=rng), INTRADAY_RES[res_name])
                            if not df.is_empty():
                                self.store.write_bars(sym, res_name, df, provider=self.name, source=f"chart interval={res_name} range={rng}", **d_)
                                self._mark(sym, res_name, df)
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
                        self.store.write_bars(sym, "1m", m, provider=self.name, source="chart interval=1m (7-day windows)", **d_)
                        self._mark(sym, "1m", m)
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

    def capabilities(self, today=None):
        return [_cap("underlying", "index", "1d", date(1990, 1, 2), today or datetime.now(cal.ET).date(), symbols=list(self.FILES))]

    @staticmethod
    def parse(text: str) -> pl.DataFrame:
        df = pl.read_csv(io.StringIO(text))
        df = df.rename({c: c.strip().lower() for c in df.columns})
        df = df.with_columns(pl.col("date").str.to_date("%m/%d/%Y"), pl.lit(0.0).alias("volume"))
        return df.select(["date", "open", "high", "low", "close", "volume"]).drop_nulls().filter(pl.col("close") > 0).unique("date", keep="last").sort("date")

    def plan(self, dataset, symbols, start, end):
        return {"requests": 1, "est_bytes": 450_000, "est_seconds": 3, "cost": 0.0, "note": "Single CSV."}

    def ingest(self, dataset, symbols=None, start=None, end=None, progress=None, should_cancel=None, force=False):
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        with _client() as client:
            for sym in [s for s in (symbols or ["VIX"]) if s in self.FILES] or ["VIX"]:
                if progress:
                    progress(0.2, f"Cboe {sym} history")
                df = self.parse(http_get(client, f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{self.FILES[sym]}").text)
                ensure_symbol(self.store, sym)
                self.store.write_daily(sym, df, provider=self.name, obs="index", source=self.FILES[sym], **self.desc(limitations=["Index levels, not tradable prices."]))
                _finish(self.store, sym)
                out["symbols"][sym] = {"daily_bars": df.height, "first": df["date"][0].isoformat(), "last": df["date"][-1].isoformat()}
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ GitHub EOD options archive
def clean_chain(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Validation shared by the option-quote sources: drop crossed/negative quotes, null out
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


EOD_LIMITS = ["One snapshot per contract at the 16:00 ET close: cannot answer intraday entries, 0DTE or intraday exits."]


class GithubEod(Source):
    name, label, tier = "github_eod", "EOD options archive (SPY/QQQ/IWM)", "free"
    REPO = "anahatsingh-ui/options-dataset-hist"
    url = f"https://github.com/{REPO}"
    license = "MIT (c) 2025 Philipp D. Dubach; preservation mirror"
    description = "End-of-day option chains 2008-2025: every listed contract with bid/ask, sizes, volume, open interest, IV and Greeks (16:00 ET snapshot)."
    datasets = [{"id": "options_eod", "label": "Full EOD chains", "resolution": "EOD", "coverage": "SPY & IWM 2008-2025, QQQ 2011-2025", "quote_type": "quotes"}]
    SYMBOLS = ("SPY", "QQQ", "IWM")

    def capabilities(self, today=None):
        return [_cap("options", "quote", "EOD", date(2008, 1, 2), date(2025, 12, 12), symbols=list(self.SYMBOLS), note="full chains; one file per year (~100-250 MB each)")]

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

    def _have(self, sym: str) -> set[date]:
        return set(self.store.option_days(sym, "quote", provider=self.name, complete_only=True))

    def plan(self, dataset, symbols, start, end):
        with _client() as client:
            total, files, per = 0, 0, {}
            for sym in symbols:
                if sym not in self.SYMBOLS:
                    raise SourceError(f"{self.label} covers {', '.join(self.SYMBOLS)} only")
                ys = {y: s for y, s in self.years(client, sym).items() if (not start or y >= start.year) and (not end or y <= end.year)}
                have = self._have(sym)
                done = [y for y in ys if (exp := cal.trading_days(max(date(y, 1, 1), start or date(y, 1, 1)), min(date(y, 12, 31), end or date(y, 12, 31)))) and all(d in have for d in exp)]
                todo = {y: s for y, s in ys.items() if y not in done}
                per[sym] = {"years": sorted(ys), "download_bytes": sum(todo.values()), "years_already_stored": sorted(done)}
                total += sum(todo.values())
                files += len(todo)
        return {"requests": files, "est_bytes": int(total * 2.1), "download_bytes": total, "est_seconds": int(files * 25 + total / 2.5e6), "cost": 0.0, "detail": per,
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
                have = self._have(sym)
                expected = [d for d in cal.trading_days(max(date(y, 1, 1), start or date(y, 1, 1)), min(date(y, 12, 31), end or date(y, 12, 31)))]
                if not force and expected and all(d in have for d in expected):
                    rep["sessions_skipped"] += len(expected)
                    continue
                f = self._download(client, f"{sym.lower()}/options_{y}.parquet", size, prog, f"{sym} {y}")
                if sym not in und_cache:
                    uf = self._download(client, f"{sym.lower()}/underlying_prices.parquet", 0, None)
                    u = pl.read_parquet(uf).select(pl.col("date").str.to_date().alias("_d"), pl.col("close").alias("underlying_price"))
                    und_cache[sym] = u
                    # the archive's own price table is stored as its own dataset (independent of Yahoo)
                    ud = pl.read_parquet(uf).select(pl.col("date").str.to_date(), "open", "high", "low", "close", pl.col("volume").cast(pl.Float64))
                    self.store.write_daily(sym, ud, provider=self.name, source=f"{sym.lower()}/underlying_prices.parquet", **self.desc(limitations=["Daily bars shipped with the option archive."]))
                for mth in range(1, 13):
                    if should_cancel and should_cancel():
                        raise Cancelled()
                    if prog:
                        prog(mth / 13, f"{sym} {y}-{mth:02d} normalising")
                    lf = pl.scan_parquet(f).filter(pl.col("date").str.starts_with(f"{y}-{mth:02d}"))
                    df = self.normalize(lf, und_cache[sym])
                    if df.is_empty():
                        continue
                    src_rows = {r["_d"]: r["n"] for r in df.group_by("_d").agg(pl.len().alias("n")).iter_rows(named=True)}
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
                        # the whole year file was downloaded and size-checked, so this session's rows are all of the source's rows
                        self.store.write_option_day(sym, d, part.with_columns(pl.col("ts").alias("quote_ts")), provider=self.name, res="EOD", status="complete",
                                                    source=f"{sym.lower()}/options_{y}.parquet", expected_rows=int(src_rows.get(d, part.height)), **self.desc(limitations=EOD_LIMITS))
                        rep["sessions_written"] += 1
                        rep["rows"] += part.height
                rep["years"].append(y)
            for sym in out["symbols"]:
                if progress:
                    progress(0.97, f"{sym}: deriving IV history")
                from strata.data.derived import derive_iv_history
                ds = self.store.option_dataset(sym, "quote", provider=self.name)
                if ds is not None:
                    derive_iv_history(self.store, sym, dataset=ds)
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

    def capabilities(self, today=None):
        return [_cap("options", "quote", "EOD", self.FIRST, today or datetime.now(cal.ET).date(), note="four expirations (~2-8 weeks out), ~27 strikes each; no 0DTE; rate-limited API")]

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

    def _have(self, sym: str) -> set[date]:
        return set(self.store.option_days(sym, "quote", provider=self.name, complete_only=True))

    def plan(self, dataset, symbols, start, end):
        start = max(start or date.today() - timedelta(days=90), self.FIRST)
        end = end or date.today()
        days = cal.trading_days(start, end)
        todo = 0
        for sym in symbols:
            have = self._have(sym)
            todo += sum(1 for d in days if d not in have)
        q = (todo + 3) // 4
        return {"requests": q, "est_bytes": todo * 14_000, "est_seconds": q * 6, "cost": 0.0, "note": f"{todo} sessions to fetch (4 per request, ~6 s each). The public API is rate-limited; large ranges take a while but can be resumed."}

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
        lim = EOD_LIMITS + ["Sparse chains: four expirations and ~27 strikes per expiration; no volume or open interest."]
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
                have = self._have(sym)
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
                    if rows and rows[-1].get("__limit__"):  # over the API row cap: fetch each date on its own, paged to the end
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
                        src_rows = {r["_d"]: r["n"] for r in df.group_by("_d").agg(pl.len().alias("n")).iter_rows(named=True)}
                        df, q = clean_chain(df)
                        rep["rejected_crossed"] += q["rejected_crossed"]
                        rep["greeks_recomputed_rows"] += q["greeks_recomputed_rows"]
                        for (d,), part in df.partition_by("_d", as_dict=True).items():
                            close = cal.session_bounds(d)[1]
                            part = part.drop("_d").with_columns(pl.lit(close).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")).alias("ts"))
                            self.store.write_option_day(sym, d, part.with_columns(pl.col("ts").alias("quote_ts")), provider=self.name, res="EOD", status="complete",
                                                        source="option_chain table", expected_rows=int(src_rows.get(d, part.height)), **self.desc(limitations=lim))
                            rep["sessions_written"] += 1
                            rep["rows"] += part.height
                            got.add(d)
                    rep["sessions_empty"] += [d.isoformat() for d in batch if d not in got]
                    done += len(batch)
                rep["sessions_empty"] = rep["sessions_empty"][:60]
                from strata.data.derived import derive_iv_history
                ds = self.store.option_dataset(sym, "quote", provider=self.name)
                if ds is not None:
                    derive_iv_history(self.store, sym, dataset=ds)
                _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


# ============================================================================ option trade bars (shared)
TRADE_LIMITS = ["Trade prints only: no bid/ask. A contract has a bar only in minutes when it traded.",
                "Stored selectively: only the contracts and sessions that were requested. The index lists every (session, contract) that was fetched to completion."]


def previous_close(store: MarketStore, sym: str, d: date) -> float | None:
    """Close of the last session strictly BEFORE d - the latest price that was known when
    session d opened. Used to choose a strike universe without looking ahead."""
    daily = store.read_daily(sym)
    if daily.is_empty():
        return None
    prior = daily.filter(pl.col("date") < d)
    return float(prior["close"][-1]) if prior.height else None


def strike_band(ref_price: float, pct: float, step: float) -> tuple[float, float]:
    lo, hi = ref_price * (1 - pct / 100), ref_price * (1 + pct / 100)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


class OptionBarSource(Source):
    """A source of per-contract option trade bars. Subclasses implement contract listing
    and the fetch of one batch; `acquire` owns resumability, validation and the index."""

    batch_size = 100
    require_volume = False
    rate = 3.0

    def bars_window(self, today: date | None = None) -> tuple[date, date]:
        raise NotImplementedError

    def list_contracts(self, client, sym: str, as_of: date, exp_from: date, exp_to: date, strike_lo: float, strike_hi: float, rights: tuple[str, ...] = ("C", "P")) -> list[Contract]:
        raise NotImplementedError

    def fetch_batch(self, client, sym: str, contracts: list[Contract], sessions: list[date]) -> tuple[pl.DataFrame, dict, int]:
        """Returns (bars, {(session, occ): status}, pages). Only pairs present in the status
        dict were fetched to completion; a pair the provider cannot vouch for is omitted."""
        raise NotImplementedError

    def batches(self, todo: dict[date, list[Contract]]) -> list[tuple[list[Contract], list[date]]]:
        """Default grouping: one batch per session and `batch_size` contracts."""
        out = []
        for d in sorted(todo):
            cs = sorted(todo[d], key=lambda c: c.occ)
            for i in range(0, len(cs), self.batch_size):
                out.append((cs[i:i + self.batch_size], [d]))
        return out

    def estimate(self, wanted: dict[date, list[Contract]], sym: str) -> dict:
        idx = self.store.option_bar_index(sym, provider=self.name)
        have = set(zip(idx["date"].to_list(), idx["occ"].to_list())) if not idx.is_empty() else set()
        todo = {d: [c for c in cs if (d, c.occ) not in have] for d, cs in wanted.items()}
        todo = {d: cs for d, cs in todo.items() if cs}
        pairs = sum(len(v) for v in todo.values())
        total = sum(len(v) for v in wanted.values())
        reqs = len(self.batches(todo))
        return {"source": self.name, "pairs_requested": total, "pairs_cached": total - pairs, "pairs_to_fetch": pairs, "requests": reqs,
                "est_bars": pairs * 150, "est_bytes": pairs * 150 * 30, "est_seconds": int(reqs / self.rate * 1.5 + 2), "cost": 0.0}

    def acquire(self, sym: str, wanted: dict[date, list[Contract]], progress=None, should_cancel=None, force: bool = False) -> dict:
        """Download the option trade bars for the requested (session, contract) pairs.

        Resumable: pairs already in the dataset's index are skipped. A pair enters the
        index only after its request was paginated to the end, so an interrupted or failed
        download leaves no pair marked as done and a later run fetches it again."""
        ok, msg = self.available()
        if not ok:
            raise SourceError(f"{self.label}: {msg}")
        first, last = self.bars_window()
        idx = self.store.option_bar_index(sym, provider=self.name)
        have = set() if force or idx.is_empty() else set(zip(idx["date"].to_list(), idx["occ"].to_list()))
        rep = {"source": self.name, "symbol": sym, "pairs_requested": sum(len(v) for v in wanted.values()), "pairs_cached": 0, "pairs_ok": 0, "pairs_empty": 0, "pairs_unavailable": 0,
               "pairs_failed": 0, "bars": 0, "requests": 0, "pages": 0, "rejected": {}, "errors": [], "out_of_window": []}
        todo: dict[date, list[Contract]] = {}
        for d, cs in wanted.items():
            if d < first or d > last:
                rep["out_of_window"].append(d.isoformat())
                rep["pairs_unavailable"] += len(cs)
                continue
            for c in cs:
                if (d, c.occ) in have:
                    rep["pairs_cached"] += 1
                elif c.expiration < d:
                    rep["pairs_unavailable"] += 1  # an expired contract cannot trade on a later session
                else:
                    todo.setdefault(d, []).append(c)
        batches = self.batches(todo)
        ensure_symbol(self.store, sym, has_options=True)
        throttle = Throttle(self.rate)
        with _client() as client:
            for bi, (contracts, sessions) in enumerate(batches):
                if should_cancel and should_cancel():
                    raise Cancelled()
                if progress:
                    progress(bi / max(len(batches), 1), f"{self.label}: {sym} option trades {sessions[0]}{' .. ' + str(sessions[-1]) if len(sessions) > 1 else ''} ({len(contracts)} contract{'s' if len(contracts) != 1 else ''})")
                throttle.wait()
                want = {(d, c.occ) for d in sessions for c in contracts if c in todo.get(d, ())}
                try:
                    bars, status, pages = self.fetch_batch(client, sym, contracts, sessions)
                except SourceError as e:
                    # nothing from this batch is written or indexed: it will be fetched again
                    rep["pairs_failed"] += len(want)
                    if len(rep["errors"]) < 8:
                        rep["errors"].append(f"{sessions[0]}: {str(e)[:220]}")
                    continue
                rep["requests"] += 1
                rep["pages"] += pages
                bars, stats = validate_trade_bars(bars, self.require_volume)
                for k, v in stats.items():
                    if k != "input_rows" and v:
                        rep["rejected"][k] = rep["rejected"].get(k, 0) + v
                counts: dict[tuple, int] = {}
                if not bars.is_empty():
                    g = bars.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).group_by("_d", "occ").agg(pl.len().alias("n"))
                    counts = {(r["_d"], r["occ"]): r["n"] for r in g.iter_rows(named=True)}
                    keep = pl.DataFrame({"_d": [k[0] for k in status], "occ": [k[1] for k in status]}, schema={"_d": pl.Date, "occ": pl.String})
                    bars = bars.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).join(keep, on=["_d", "occ"], how="semi").drop("_d")
                by_occ = {c.occ: c for c in contracts}
                now = datetime.now(timezone.utc)
                rows = []
                for (d, occ), st in status.items():
                    if (d, occ) not in want and (d, occ) in have:
                        continue
                    c = by_occ[occ]
                    n = counts.get((d, occ), 0)
                    o, cl = cal.session_bounds(d)
                    rows.append({"date": d, "occ": occ, "expiration": c.expiration, "strike": c.strike, "right": c.right, "bars": n, "status": "ok" if n else "empty",
                                 "pages": pages, "fetched_at": now, "window": f"{o:%H:%M}-{cl:%H:%M} ET"})
                    if (d, occ) in want:
                        rep["pairs_ok" if n else "pairs_empty"] += 1
                rep["pairs_unavailable"] += len([p for p in want if p not in status])
                defs = pl.DataFrame([c.row(sym) for c in contracts], infer_schema_length=None)
                self.store.write_option_bars(sym, self.name, bars, rows, defs, source=f"{self.label} 1-minute option bars", **self.desc(limitations=TRADE_LIMITS + self.extra_limits()))
                rep["bars"] += bars.height
        if batches:
            _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return rep

    def extra_limits(self) -> list[str]:
        return []

    # ---- bulk mode (Data page): a band around the previous close, short expirations
    def bulk_contracts(self, client, sym: str, d: date, band_pct: float = 2.0, max_dte: int = 7) -> list[Contract]:
        ref = previous_close(self.store, sym, d)
        if ref is None:
            raise SourceError(f"No daily history for {sym} before {d}: download daily bars first (the strike range is taken from the previous close, never from the session itself).")
        lo, hi = strike_band(ref, band_pct, STRIKE_STEP.get(sym, 1.0))
        return self.list_contracts(client, sym, d, d, d + timedelta(days=max_dte), lo, hi)

    def bulk_wanted(self, sym: str, start: date, end: date, band_pct: float = 2.0, max_dte: int = 7) -> dict[date, list[Contract]]:
        first, last = self.bars_window()
        out = {}
        with _client() as client:
            for d in cal.trading_days(max(start, first), min(end, last)):
                out[d] = self.bulk_contracts(client, sym, d, band_pct, max_dte)
        return out


# ============================================================================ Yahoo option trade bars
class YahooOptions(OptionBarSource):
    name, label, tier = "yahoo_options", "Yahoo Finance option trades", "free"
    url = "https://finance.yahoo.com"
    license = "Yahoo terms of service (personal research use)"
    description = ("1-minute option trade bars for the last ~30 days, without an account. Only contracts that are still listed, or expired within about a day, can be "
                   "retrieved - run it regularly to keep 0DTE history. No bid/ask.")
    datasets = [{"id": "option_trades", "label": "Option 1-minute trade bars (recent)", "resolution": "1m", "coverage": "last ~30 days; listed contracts only", "quote_type": "trades"}]
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
    WINDOW_DAYS = 29
    batch_size = 1
    require_volume = True  # bars Yahoo reports with zero volume are not confirmed prints
    rate = 5.0

    def capabilities(self, today=None):
        a, b = self.bars_window(today)
        return [_cap("options", "trade", "1m", a, b, rolling_days=self.WINDOW_DAYS, symbols=None,
                     note="only contracts still listed today or expired within about one day; older expired contracts are no longer served")]

    def bars_window(self, today=None):
        today = today or datetime.now(cal.ET).date()
        return today - timedelta(days=self.WINDOW_DAYS), today

    def extra_limits(self):
        return ["Yahoo serves a contract's minute bars only while it is listed and for about a day after it expires; only the last ~30 days are available.",
                "Bars reported with zero volume are discarded as unconfirmed prints. Contract definitions are constructed from the OCC symbol (standard 100 multiplier assumed)."]

    def list_contracts(self, client, sym, as_of, exp_from, exp_to, strike_lo, strike_hi, rights=("C", "P")):
        step = STRIKE_STEP.get(sym)
        if step is None:
            raise SourceError(f"{self.label} constructs contract symbols from a strike grid, which is only configured for {', '.join(STRIKE_STEP)}.")
        today = datetime.now(cal.ET).date()
        exps = [d for d in cal.trading_days(max(exp_from, as_of), exp_to)]
        ks, k = [], math.ceil(strike_lo / step - 1e-9) * step
        while k <= strike_hi + 1e-9:
            ks.append(round(k, 3))
            k += step
        return [contract_from_occ(occ_symbol(sym, e, r, k), source="constructed from the OCC symbol; existence verified by the fetch") for e in exps for r in rights for k in ks
                if e >= today - timedelta(days=4)]  # older expirations have been purged by the provider

    def batches(self, todo):
        """One request per contract and 7-day window (the endpoint's maximum for 1m bars)."""
        per: dict[str, tuple[Contract, list[date]]] = {}
        for d, cs in todo.items():
            for c in cs:
                per.setdefault(c.occ, (c, []))[1].append(d)
        out = []
        for c, days in sorted(per.values(), key=lambda x: x[0].occ):
            days = sorted(days)
            chunk = [days[0]]
            for d in days[1:]:
                if (d - chunk[0]).days <= 5:
                    chunk.append(d)
                else:
                    out.append(([c], chunk))
                    chunk = [d]
            out.append(([c], chunk))
        return out

    @staticmethod
    def parse_bars(res: dict, occ: str, c: Contract) -> pl.DataFrame:
        ts = res.get("timestamp") or []
        q = (res.get("indicators", {}).get("quote") or [{}])[0]
        rows = []
        for i, t in enumerate(ts):
            cl = (q.get("close") or [None])[i] if i < len(q.get("close") or []) else None
            if cl is None or t % 60:
                continue  # no trade in this minute / the unaligned live bar
            rows.append((t, occ, c.expiration, c.strike, c.right, q["open"][i], q["high"][i], q["low"][i], cl, int(q["volume"][i] or 0)))
        if not rows:
            return pl.DataFrame(schema=TRADE_SCHEMA)
        df = pl.DataFrame(rows, schema=["ts", "occ", "expiration", "strike", "right", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")), pl.lit(None, dtype=pl.Int32).alias("trade_count"), pl.lit(None, dtype=pl.Float64).alias("vwap"))

    def fetch_batch(self, client, sym, contracts, sessions):
        c = contracts[0]
        p1 = int(cal.session_bounds(sessions[0])[0].timestamp())
        p2 = int(cal.session_bounds(sessions[-1])[1].timestamp()) + 60
        today = datetime.now(cal.ET).date()
        now = datetime.now(timezone.utc)
        try:
            r = http_get(client, self.BASE + c.occ, params={"interval": "1m", "period1": p1, "period2": p2, "includePrePost": "false"})
            ch = r.json().get("chart", {})
        except SourceError as e:
            if "No data found" not in str(e) and "delisted" not in str(e):
                raise
            ch = {"error": {"description": "No data found, symbol may be delisted"}}  # the provider answers an unknown symbol with HTTP 404
        # a session that is still open is not recorded: its bars are not final
        done = [d for d in sessions if cal.session_bounds(d)[1].astimezone(timezone.utc) <= now]
        if ch.get("error"):
            desc = str(ch["error"].get("description", ""))
            if "No data found" in desc or "delisted" in desc:
                # the symbol does not exist at the provider. For a contract that has not expired that means it is
                # not listed (no trades); for an expired one it may simply have been purged - which proves nothing.
                return pl.DataFrame(schema=TRADE_SCHEMA), ({(d, c.occ): "empty" for d in done} if c.expiration >= today else {}), 1
            raise SourceError(f"Yahoo: {desc or ch['error']}")
        res = (ch.get("result") or [None])[0]
        if res is None:
            raise SourceError(f"Yahoo returned no result for {c.occ}")
        bars = self.parse_bars(res, c.occ, c)
        if bars.is_empty() and c.expiration < today:
            return bars, {}, 1  # expired and purged: unavailable, not "no trades"
        return bars, {(d, c.occ): "ok" for d in done}, 1

    def plan(self, dataset, symbols, start, end):
        a, b = self.bars_window()
        start, end = max(start or a, a), min(end or b, b)
        days = cal.trading_days(start, end)
        per = 2 * 2 * 30  # +/-2% band, two rights, ~6 expirations at $1 strikes
        reqs = len(days) * per * len(symbols) // 4
        return {"requests": reqs, "est_bytes": len(days) * per * 4500 * len(symbols), "est_seconds": int(reqs / self.rate * 1.4), "cost": 0.0,
                "note": f"Bulk download: every contract within 2% of the previous close expiring within 7 days, {start} to {end}. The research agent downloads far less: only the contracts a strategy needs."}

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        a, b = self.bars_window()
        start, end = max(start or b - timedelta(days=6), a), min(end or b, b)
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        for i, sym in enumerate(symbols):
            wanted = self.bulk_wanted(sym, start, end)
            sub = (lambda f, m, i=i: progress((i + f) / len(symbols), m)) if progress else None
            out["symbols"][sym] = self.acquire(sym, wanted, sub, should_cancel, force)
        return out


# ============================================================================ Alpaca (free account)
class Alpaca(OptionBarSource):
    name, label, tier = "alpaca", "Alpaca Market Data", "free_key"
    url = "https://docs.alpaca.markets/docs/about-market-data-api"
    license = "Alpaca market data agreement"
    description = ("Free account: 1-minute stock bars since 2016 and 1-minute option TRADE bars since February 2024. Historical option quotes are not offered on the free plan, "
                   "so option data from Alpaca is trade-print based.")
    datasets = [
        {"id": "bars_1m", "label": "1-minute stock bars", "resolution": "1m", "coverage": "2016 to 15 minutes ago"},
        {"id": "option_trades", "label": "Option 1-minute trade bars", "resolution": "1m", "coverage": "2024-02 onward", "quote_type": "trades"},
    ]
    DATA = "https://data.alpaca.markets"
    TRADING = "https://paper-api.alpaca.markets"
    STOCKS_FIRST = date(2016, 1, 4)
    rate = 3.0  # the free plan allows 200 historical calls a minute

    def capabilities(self, today=None):
        today = today or datetime.now(cal.ET).date()
        return [_cap("underlying", "trade", "1m", self.STOCKS_FIRST, today, note="15-minute delay on the free plan"),
                _cap("options", "trade", "1m", ALPACA_OPTIONS_FIRST, today, note="trade bars for any listed or expired contract since February 2024; no historical quotes on the free plan")]

    def bars_window(self, today=None):
        return ALPACA_OPTIONS_FIRST, (today or datetime.now(cal.ET).date())

    def _keys(self) -> tuple[str, str]:
        from strata.api.settings import get_secret
        return get_secret("ALPACA_API_KEY") or "", get_secret("ALPACA_API_SECRET") or ""

    def available(self):
        k, s = self._keys()
        return (bool(k and s), "key configured" if k and s else "connect a free Alpaca account (Settings -> Data connections)")

    def _h(self) -> dict:
        k, s = self._keys()
        if not (k and s):
            raise SourceError("Alpaca API key/secret not configured (Settings -> Data connections)")
        return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

    def check(self) -> dict:
        """Connection test used by the guided setup: one cheap authenticated request per API."""
        ok, msg = self.available()
        if not ok:
            return {"ok": False, "message": msg}
        out = {"ok": True, "message": "", "stocks": False, "options": False}
        with _client() as client:
            try:
                end = datetime.now(timezone.utc) - timedelta(days=4)
                http_get(client, f"{self.DATA}/v2/stocks/bars", params={"symbols": "SPY", "timeframe": "1Day", "start": (end - timedelta(days=5)).date().isoformat(), "end": end.date().isoformat(), "limit": 5}, headers=self._h(), retries=2)
                out["stocks"] = True
                http_get(client, f"{self.TRADING}/v2/options/contracts", params={"underlying_symbols": "SPY", "limit": 1}, headers=self._h(), retries=2)
                out["options"] = True
                out["message"] = "Connected: stock bars and option contract data are accessible."
            except SourceError as e:
                out["ok"] = out["stocks"]
                out["message"] = ("Stock bars work, but the options endpoints were refused: " if out["stocks"] else "Alpaca rejected the request: ") + str(e)[:200]
        return out

    @staticmethod
    def parse_bars(bars: list[dict]) -> pl.DataFrame:
        rows = [(b["t"], float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]), float(b.get("v", 0))) for b in bars]
        if not rows:
            return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64})
        df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC").cast(pl.Datetime("us", "UTC")))

    def pages(self, client, url: str, params: dict, key: str, max_pages: int = 5000):
        """Yield every page of a paginated Alpaca endpoint. The caller only treats the
        request as complete when this generator is exhausted (next_page_token is null)."""
        token, n = None, 0
        while True:
            p = dict(params)
            if token:
                p["page_token"] = token
            d = http_get(client, url, params=p, headers=self._h()).json()
            n += 1
            yield d.get(key) or ({} if key == "bars" else [])
            token = d.get("next_page_token")
            if not token:
                return
            if n >= max_pages:
                raise SourceError(f"pagination did not finish after {max_pages} pages")

    # ---- contract definitions (point-in-time safe)
    @staticmethod
    def parse_contract(c: dict) -> Contract:
        root = c.get("root_symbol") or parse_occ(c["symbol"])[0]
        deliv = c.get("deliverables") or []
        nonstd = bool(re.search(r"\d$", root)) or any(str(x.get("type")) != "equity" or float(x.get("amount") or 100) != 100 for x in deliv)
        return Contract(c["symbol"], root, date.fromisoformat(c["expiration_date"]), float(c["strike_price"]), "C" if c["type"] == "call" else "P",
                        multiplier=float(c.get("multiplier") or 100), style=c.get("style") or "american", adjusted=nonstd, provider_id=c.get("id"),
                        status=c.get("status"), source="Alpaca /v2/options/contracts")

    def list_contracts(self, client, sym, as_of, exp_from, exp_to, strike_lo, strike_hi, rights=("C", "P")):
        """Contract definitions that were valid on session `as_of`.

        Both `active` and `inactive` contracts are requested: a contract that has since
        expired is `inactive` today but was tradable on `as_of`, and the reverse filter
        would silently drop it. Whether a contract actually traded on a session is decided
        by its trade bars, not by today's status."""
        exp_from = max(exp_from, as_of)
        ck = self.cache / "contracts" / (hashlib.sha256(f"{sym}|{exp_from}|{exp_to}|{strike_lo}|{strike_hi}|{rights}".encode()).hexdigest()[:24] + ".json")
        today = datetime.now(cal.ET).date()
        if ck.exists() and (exp_to < today or time.time() - ck.stat().st_mtime < 43200):
            raw = json.loads(ck.read_text())
        else:
            raw = {}
            for status in ("active", "inactive"):
                params = {"underlying_symbols": sym, "status": status, "expiration_date_gte": exp_from.isoformat(), "expiration_date_lte": exp_to.isoformat(),
                          "strike_price_gte": strike_lo, "strike_price_lte": strike_hi, "limit": 10000, "show_deliverables": "true"}
                for page in self.pages(client, f"{self.TRADING}/v2/options/contracts", params, "option_contracts"):
                    for c in page:
                        raw[c["symbol"]] = c
            ck.parent.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps(raw))
        out = [self.parse_contract(c) for c in raw.values()]
        out = [c for c in out if c.right in rights and c.expiration >= as_of and not c.adjusted]
        if not out and exp_to < today - timedelta(days=30) and sym in STRIKE_STEP:
            # definitions of long-expired contracts may no longer be listed: OCC symbols are deterministic, so
            # construct the standard grid and let the bars endpoint decide what existed
            step = STRIKE_STEP[sym]
            k = math.ceil(strike_lo / step - 1e-9) * step
            while k <= strike_hi + 1e-9:
                for e in cal.trading_days(exp_from, exp_to):
                    out += [contract_from_occ(occ_symbol(sym, e, r, round(k, 3)), source="constructed from the OCC symbol (definition no longer listed)") for r in rights]
                k += step
        return sorted(out, key=lambda c: (c.expiration, c.right, c.strike))

    def fetch_batch(self, client, sym, contracts, sessions):
        d = sessions[0]
        a, b = cal.session_bounds(d)
        now = datetime.now(timezone.utc)
        if b.astimezone(timezone.utc) + timedelta(minutes=16) > now:
            return pl.DataFrame(schema=TRADE_SCHEMA), {}, 0  # the free plan cannot serve the latest 15 minutes; nothing is recorded
        by = {c.occ: c for c in contracts}
        params = {"symbols": ",".join(by), "timeframe": "1Min", "start": a.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                  "end": b.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "limit": 10000, "sort": "asc"}
        rows, pages = [], 0
        # results are sorted by symbol first: the first page may hold a single contract. Every page is read.
        for page in self.pages(client, f"{self.DATA}/v1beta1/options/bars", params, "bars"):
            pages += 1
            for occ, bars in (page or {}).items():
                c = by.get(occ)
                if c is None:
                    continue
                for x in bars:
                    rows.append((x["t"], occ, c.expiration, c.strike, c.right, float(x["o"]), float(x["h"]), float(x["l"]), float(x["c"]), int(x.get("v") or 0),
                                 int(x["n"]) if x.get("n") is not None else None, float(x["vw"]) if x.get("vw") is not None else None))
        if rows:
            df = pl.DataFrame(rows, schema=["ts", "occ", "expiration", "strike", "right", "open", "high", "low", "close", "volume", "trade_count", "vwap"], orient="row")
            df = df.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC").cast(pl.Datetime("us", "UTC")))
        else:
            df = pl.DataFrame(schema=TRADE_SCHEMA)
        return df, {(d, occ): "ok" for occ in by}, pages

    def extra_limits(self):
        return ["Free plan: derived from the OPRA feed with a 15-minute delay; history starts in February 2024."]

    # ---- data-page entry points
    def plan(self, dataset, symbols, start, end):
        days = len(cal.trading_days(start or date.today() - timedelta(days=30), end or date.today()))
        if dataset == "bars_1m":
            return {"requests": max(1, days * 390 // 10000 + 1) * len(symbols), "est_bytes": days * 9000 * len(symbols), "est_seconds": days // 20 + 5, "cost": 0.0,
                    "note": "Regular-session 1-minute bars (SIP, 15-minute delayed on the free plan)."}
        return {"requests": days * 8 * len(symbols), "est_bytes": days * 600_000 * len(symbols), "est_seconds": days * 6, "cost": 0.0,
                "note": "Bulk download: every contract within 2% of the PREVIOUS close expiring within 7 days. Trade prints only: no bid/ask. The research agent downloads far less: only the contracts a strategy needs."}

    def stock_bars(self, sym: str, start: date, end: date, progress=None, should_cancel=None, force: bool = False) -> dict:
        """1-minute stock bars, 20 sessions per paginated request. A session is recorded as
        complete only after the request that covers it has been read to the last page."""
        desc = self.desc(adjustment="split", limitations=["Split-adjusted. Free plan: data newer than 15 minutes is not served."])
        key = self.store.ensure_dataset(self.name, sym, "underlying", "trade", "1m", **desc)
        done = set() if force else self.store.complete_sessions(key)
        now = datetime.now(timezone.utc)
        days = [d for d in cal.trading_days(max(start, self.STOCKS_FIRST), end) if cal.session_bounds(d)[1].astimezone(timezone.utc) + timedelta(minutes=16) <= now]
        todo = [d for d in days if d not in done]
        rep = {"bars": 0, "sessions": len(todo), "sessions_cached": len(days) - len(todo), "pages": 0, "empty_sessions": [], "feed": "sip"}
        throttle = Throttle(self.rate)
        with _client() as client:
            for i in range(0, len(todo), 20):
                if should_cancel and should_cancel():
                    raise Cancelled()
                chunk = todo[i:i + 20]
                if progress:
                    progress(i / max(len(todo), 1), f"Alpaca {sym} 1-minute bars {chunk[0]} .. {chunk[-1]}")
                a, b = cal.session_bounds(chunk[0])[0], cal.session_bounds(chunk[-1])[1]
                params = {"symbols": sym, "timeframe": "1Min", "start": a.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "end": b.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                          "limit": 10000, "adjustment": "split", "feed": rep["feed"], "sort": "asc"}
                bars: list[dict] = []
                throttle.wait()
                try:
                    for page in self.pages(client, f"{self.DATA}/v2/stocks/bars", params, "bars"):
                        rep["pages"] += 1
                        bars += (page or {}).get(sym, []) if isinstance(page, dict) else list(page)
                except SourceError as e:
                    if "subscription" in str(e).lower() and rep["feed"] == "sip":
                        raise SourceError("Alpaca refused the consolidated (SIP) feed for this account. " + str(e)[:160]) from None
                    raise
                df = self.parse_bars(bars)
                if not df.is_empty():
                    df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et"))
                    df = df.with_columns(pl.col("_et").dt.date().alias("_d"), (pl.col("_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_et").dt.minute().cast(pl.Int32)).alias("_m"))
                    closes = pl.DataFrame({"_d": chunk, "_c": [cal.session_bounds(d)[1].hour * 60 + cal.session_bounds(d)[1].minute for d in chunk]}, schema={"_d": pl.Date, "_c": pl.Int32})
                    df = df.join(closes, on="_d", how="inner").filter((pl.col("_m") >= 570) & (pl.col("_m") < pl.col("_c")))
                    per = {r["_d"]: r["n"] for r in df.group_by("_d").agg(pl.len().alias("n")).iter_rows(named=True)}
                    df = df.drop("_et", "_d", "_m", "_c")
                    self.store.write_bars(sym, "1m", df, provider=self.name, source=f"v2/stocks/bars feed={rep['feed']} adjustment=split", **desc)
                    rep["bars"] += df.height
                else:
                    per = {}
                # the whole chunk was paginated to the end: every session in it is now known
                entries = {}
                for d in chunk:
                    n = per.get(d, 0)
                    if n == 0:
                        rep["empty_sessions"].append(d.isoformat())
                        continue  # no bars at all is reported, not recorded as a complete session
                    entries[d.isoformat()] = {"status": "complete", "bars": int(n), "expected": cal.session_minutes(d)}
                self.store.mark_sessions(key, entries)
        rep["empty_sessions"] = rep["empty_sessions"][:40]
        return rep

    def ingest(self, dataset, symbols, start=None, end=None, progress=None, should_cancel=None, force=False):
        start = start or date.today() - timedelta(days=30)
        end = end or date.today()
        out = {"source": self.name, "dataset": dataset, "symbols": {}}
        for i, sym in enumerate(symbols):
            ensure_symbol(self.store, sym)
            sub = (lambda f, m, i=i: progress((i + f) / len(symbols), m)) if progress else None
            if dataset == "bars_1m":
                out["symbols"][sym] = self.stock_bars(sym, start, end, sub, should_cancel, force)
            else:
                out["symbols"][sym] = self.acquire(sym, self.bulk_wanted(sym, start, end), sub, should_cancel, force)
            _finish(self.store, sym)
        if progress:
            progress(1.0, "done")
        return out


SOURCES: dict[str, type[Source]] = {s.name: s for s in (Yahoo, Cboe, GithubEod, DoltHub, YahooOptions, Alpaca)}


def get_source(name: str, store: MarketStore | None = None) -> Source:
    if name not in SOURCES:
        raise SourceError(f"unknown data source {name!r}; available: {', '.join(SOURCES)}")
    return SOURCES[name](store)


def option_bar_sources(store: MarketStore | None = None) -> list[OptionBarSource]:
    """Sources of option trade bars in order of preference (deepest history first)."""
    return [SOURCES[n](store) for n in ("alpaca", "yahoo_options")]


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
    mine = store.read_daily(sym, provider="yahoo")
    which = "yahoo"
    if mine.is_empty():
        mine, which = store.read_daily(sym), "stored (merged)"
    mine = mine.select("date", "close", "open")
    j = mine.join(ref, on="date", how="inner").with_columns(((pl.col("close") / pl.col("ref_close") - 1).abs() * 100).alias("diff_pct"))
    if j.is_empty():
        return {"symbol": sym, "compared": 0}
    worst = j.sort("diff_pct", descending=True).head(3)
    return {"symbol": sym, "compared": j.height, "first": j["date"][0].isoformat(), "last": j["date"][-1].isoformat(), "max_close_diff_pct": float(j["diff_pct"].max()),
            "mean_close_diff_pct": float(j["diff_pct"].mean()), "within_0_1pct": float((j["diff_pct"] < 0.1).mean()), "checked_dataset": which,
            "worst": [{"date": r["date"].isoformat(), "stored": r["close"], "reference": r["ref_close"]} for r in worst.iter_rows(named=True)],
            "reference": "EOD options archive underlying_prices.parquet (independent of Yahoo)"}
