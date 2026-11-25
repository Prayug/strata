"""Vendor adapters. Each normalises the vendor's documented response format into the store
schema. Normalisers are pure functions so they are unit-tested against recorded sample
payloads (tests/test_providers.py); live endpoints require the user's own subscription."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import polars as pl

from strata.config import get_settings
from strata.core import calendar as cal
from strata.data.providers.base import Provider, ProviderError, et_ms_of_day_to_utc, frame_from_rows


def _snap_last(ticks: list[tuple[int, dict]], snaps: list[int]) -> list[tuple[int, dict]]:
    """For each snapshot time, the last tick at or before it (with its own timestamp kept
    as quote_ts so staleness is measurable)."""
    out, j, last = [], 0, None
    ticks = sorted(ticks, key=lambda x: x[0])
    for s in snaps:
        while j < len(ticks) and ticks[j][0] <= s:
            last = ticks[j]
            j += 1
        if last is not None:
            out.append((s, last))
    return out


# ============================================================== ThetaData (Theta Terminal v2)
class ThetaData(Provider):
    name = "thetadata"
    label = "ThetaData"
    docs = "Run the Theta Terminal locally and set THETADATA_BASE_URL (default http://127.0.0.1:25510)."

    def __init__(self, store=None):
        super().__init__(store)
        self.base = get_settings().thetadata_url.rstrip("/")

    def configured(self) -> bool:
        return bool(self.base)

    def check(self) -> dict:
        try:
            r = self.client.get(f"{self.base}/v2/list/roots/option", timeout=2)
            return {"ok": r.status_code == 200, "message": f"terminal reachable (HTTP {r.status_code})"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"Theta Terminal not reachable at {self.base}: {type(e).__name__}"}

    @staticmethod
    def rows_from(resp: dict) -> list[dict]:
        fmt = resp.get("header", {}).get("format") or []
        out = []
        for item in resp.get("response", []):
            if isinstance(item, dict) and "ticks" in item:
                c = item.get("contract", {})
                for t in item["ticks"]:
                    out.append({**{f"c_{k}": v for k, v in c.items()}, **dict(zip(fmt, t))})
            elif isinstance(item, list):
                out.append(dict(zip(fmt, item)))
        return out

    def _paged(self, path: str, params: dict) -> list[dict]:
        rows, url = [], f"{self.base}{path}"
        p = params
        while url:
            resp = self.get_json(url, p)
            rows += self.rows_from(resp)
            nxt = resp.get("header", {}).get("next_page")
            url, p = (nxt, None) if nxt and nxt != "null" else (None, None)
        return rows

    def fetch_minutes(self, symbol, d):
        ds = d.strftime("%Y%m%d")
        rows = self._paged("/v2/hist/stock/ohlc", {"root": symbol, "start_date": ds, "end_date": ds, "ivl": 60000, "rth": "true"})
        return self.normalize_minutes(rows, d)

    @staticmethod
    def normalize_minutes(rows: list[dict], d: date) -> pl.DataFrame:
        out = []
        for r in rows:
            if not r.get("close"):
                continue
            out.append({"ts": et_ms_of_day_to_utc(d, r["ms_of_day"]), "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "volume": float(r.get("volume", 0))})
        return frame_from_rows(out)

    def fetch_options(self, symbol, d, interval, max_dte):
        exps = self.get_json(f"{self.base}/v2/list/expirations", {"root": symbol}).get("response", [])
        exps = [e for e in exps if 0 <= (datetime.strptime(str(e), "%Y%m%d").date() - d).days <= max_dte]
        ds = d.strftime("%Y%m%d")
        frames = []
        for e in exps:
            q = self._paged("/v2/bulk_hist/option/quote", {"root": symbol, "exp": e, "start_date": ds, "end_date": ds, "ivl": interval * 60000, "rth": "true"})
            g = self._paged("/v2/bulk_hist/option/greeks", {"root": symbol, "exp": e, "start_date": ds, "end_date": ds, "ivl": interval * 60000, "rth": "true"})
            oi = self._paged("/v2/bulk_hist/option/open_interest", {"root": symbol, "exp": e, "start_date": ds, "end_date": ds})
            frames.append(self.normalize_options(q, g, oi, d))
        frames = [f for f in frames if not f.is_empty()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else None

    @staticmethod
    def normalize_options(quotes: list[dict], greeks: list[dict], oi: list[dict], d: date) -> pl.DataFrame:
        def key(r):
            return (r["c_expiration"], r["c_strike"], r["c_right"])
        gmap = {(key(r), r["ms_of_day"]): r for r in greeks}
        oimap = {key(r): r.get("open_interest") for r in oi}
        out = []
        for r in quotes:
            k = key(r)
            g = gmap.get((k, r["ms_of_day"]), {})
            bid, ask = r.get("bid"), r.get("ask")
            if bid is None or ask is None or (bid == 0 and ask == 0):
                continue
            ts = et_ms_of_day_to_utc(d, r["ms_of_day"])
            out.append({
                "ts": ts, "expiration": datetime.strptime(str(r["c_expiration"]), "%Y%m%d").date(), "strike": r["c_strike"] / 1000.0, "right": r["c_right"],
                "bid": float(bid), "ask": float(ask), "bid_size": r.get("bid_size"), "ask_size": r.get("ask_size"),
                "open_interest": oimap.get(k), "iv": g.get("implied_vol"), "delta": g.get("delta"), "theta": g.get("theta"), "vega": g.get("vega"),
                "gamma": g.get("gamma"), "underlying_price": g.get("underlying_price"), "quote_ts": ts,
            })
        return frame_from_rows(out)


# ============================================================== Polygon / Massive
class Polygon(Provider):
    name = "polygon"
    label = "Polygon / Massive"
    docs = "Set POLYGON_API_KEY (Options Advanced plan needed for historical quotes). POLYGON_BASE_URL may point at api.massive.com."

    def __init__(self, store=None):
        super().__init__(store)
        s = get_settings()
        self.key, self.base = s.polygon_api_key, s.polygon_base_url.rstrip("/")

    def configured(self) -> bool:
        return bool(self.key)

    def check(self) -> dict:
        if not self.key:
            return {"ok": False, "message": "POLYGON_API_KEY not set"}
        try:
            r = self.client.get(f"{self.base}/v3/reference/tickers/SPY", params={"apiKey": self.key}, timeout=5)
            return {"ok": r.status_code == 200, "message": f"HTTP {r.status_code}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": str(e)}

    def _paged(self, url: str, params: dict) -> list[dict]:
        out = []
        p = {**params, "apiKey": self.key}
        while url:
            resp = self.get_json(url, p)
            out += resp.get("results", []) or []
            nxt = resp.get("next_url")
            url, p = (nxt, {"apiKey": self.key}) if nxt else (None, None)
        return out

    def fetch_minutes(self, symbol, d):
        res = self._paged(f"{self.base}/v2/aggs/ticker/{symbol}/range/1/minute/{d.isoformat()}/{d.isoformat()}", {"adjusted": "false", "sort": "asc", "limit": 50000})
        return self.normalize_minutes(res, d)

    @staticmethod
    def normalize_minutes(res: list[dict], d: date) -> pl.DataFrame:
        o, c = cal.session_bounds(d)
        lo, hi = int(o.timestamp()), int(c.timestamp())
        out = [{"ts": r["t"] // 1000, "open": r["o"], "high": r["h"], "low": r["l"], "close": r["c"], "volume": float(r.get("v", 0))} for r in res if lo <= r["t"] // 1000 < hi]
        return frame_from_rows(out)

    def fetch_options(self, symbol, d, interval, max_dte):
        contracts = self._paged(f"{self.base}/v3/reference/options/contracts", {
            "underlying_ticker": symbol, "as_of": d.isoformat(), "expired": "true", "expiration_date.gte": d.isoformat(),
            "expiration_date.lte": (d + timedelta(days=max_dte)).isoformat(), "limit": 1000})
        mins = self.fetch_minutes(symbol, d)
        spot = float(mins["close"][0]) if mins is not None and not mins.is_empty() else None
        if spot:
            contracts = [c for c in contracts if abs(c["strike_price"] / spot - 1) <= 0.15]
        snaps = self.snapshot_times(d, interval)
        o, c = cal.session_bounds(d)
        rows = []
        for ct in contracts:
            q = self._paged(f"{self.base}/v3/quotes/{ct['ticker']}", {"timestamp.gte": int(o.timestamp()) * 10**9 - 3600 * 10**9, "timestamp.lte": int(c.timestamp()) * 10**9, "order": "asc", "limit": 50000})
            rows += self.normalize_quotes(ct, q, snaps)
        return frame_from_rows(rows)

    @staticmethod
    def normalize_quotes(contract: dict, quotes: list[dict], snaps: list[int]) -> list[dict]:
        ticks = [(q["sip_timestamp"] // 10**9, q) for q in quotes if q.get("ask_price") is not None]
        out = []
        for s, (qt, q) in _snap_last(ticks, snaps):
            out.append({"ts": s, "expiration": date.fromisoformat(contract["expiration_date"]), "strike": float(contract["strike_price"]),
                        "right": "C" if contract["contract_type"].lower().startswith("c") else "P", "bid": float(q.get("bid_price") or 0.0),
                        "ask": float(q["ask_price"]), "bid_size": q.get("bid_size"), "ask_size": q.get("ask_size"), "quote_ts": qt})
        return out


# ============================================================== Databento (OPRA)
class Databento(Provider):
    name = "databento"
    label = "Databento"
    docs = "Set DATABENTO_API_KEY and `pip install databento` (OPRA.PILLAR cbbo-1m + definition; equities via DBEQ.BASIC ohlcv-1m)."

    def __init__(self, store=None):
        super().__init__(store)
        self.key = get_settings().databento_api_key

    def configured(self) -> bool:
        if not self.key:
            return False
        try:
            import databento  # noqa: F401
            return True
        except ImportError:
            return False

    def check(self) -> dict:
        if not self.key:
            return {"ok": False, "message": "DATABENTO_API_KEY not set"}
        try:
            import databento  # noqa: F401
        except ImportError:
            return {"ok": False, "message": "databento package not installed (pip install 'strata[databento]')"}
        return {"ok": True, "message": "key present; requests are billed by Databento"}

    def _client(self):
        import databento as db
        return db.Historical(self.key)

    def fetch_minutes(self, symbol, d):
        o, c = cal.session_bounds(d)
        data = self._client().timeseries.get_range(dataset="DBEQ.BASIC", schema="ohlcv-1m", symbols=[symbol], start=o.isoformat(), end=c.isoformat())
        df = pl.from_pandas(data.to_df().reset_index())
        return self.normalize_minutes(df)

    @staticmethod
    def normalize_minutes(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.select(pl.col("ts_event").cast(pl.Datetime("us", "UTC")).alias("ts"), "open", "high", "low", "close", pl.col("volume").cast(pl.Float64))

    def fetch_options(self, symbol, d, interval, max_dte):
        o, c = cal.session_bounds(d)
        cl = self._client()
        defs = pl.from_pandas(cl.timeseries.get_range(dataset="OPRA.PILLAR", schema="definition", symbols=[f"{symbol}.OPT"], stype_in="parent", start=d.isoformat(), end=(d + timedelta(days=1)).isoformat()).to_df().reset_index())
        q = pl.from_pandas(cl.timeseries.get_range(dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=[f"{symbol}.OPT"], stype_in="parent", start=o.isoformat(), end=c.isoformat()).to_df().reset_index())
        return self.normalize_options(defs, q, d, interval, max_dte)

    @staticmethod
    def normalize_options(defs: pl.DataFrame, q: pl.DataFrame, d: date, interval: int, max_dte: int) -> pl.DataFrame:
        dd = defs.select("instrument_id", pl.col("expiration").cast(pl.Datetime("us", "UTC")).dt.convert_time_zone("America/New_York").dt.date().alias("expiration"),
                         pl.col("strike_price").cast(pl.Float64).alias("strike"), pl.col("instrument_class").cast(pl.String).alias("right")).unique("instrument_id")
        dd = dd.filter((pl.col("expiration") - pl.lit(d)).dt.total_days() <= max_dte)
        qq = q.select(pl.col("ts_event").cast(pl.Datetime("us", "UTC")).alias("quote_ts"), "instrument_id", pl.col("bid_px_00").alias("bid"), pl.col("ask_px_00").alias("ask"),
                      pl.col("bid_sz_00").cast(pl.Int32).alias("bid_size"), pl.col("ask_sz_00").cast(pl.Int32).alias("ask_size"))
        snaps = pl.DataFrame({"ts": Provider.snapshot_times(d, interval)}).with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
        qq = qq.sort("quote_ts")
        out = snaps.join(dd.select("instrument_id"), how="cross").sort("ts").join_asof(qq, left_on="ts", right_on="quote_ts", by="instrument_id", strategy="backward")
        return out.join(dd, on="instrument_id").drop("instrument_id").drop_nulls(["bid", "ask"])


# ============================================================== ORATS (end-of-day strikes)
class Orats(Provider):
    name = "orats"
    label = "ORATS"
    resolution = "EOD"
    docs = "Set ORATS_API_KEY (Data API). Provides one end-of-day snapshot per session with ORATS greeks/IV; pair with a minute-bar provider for intraday rules."

    def __init__(self, store=None):
        super().__init__(store)
        s = get_settings()
        self.key, self.base = s.orats_api_key, s.orats_base_url.rstrip("/")

    def configured(self) -> bool:
        return bool(self.key)

    def check(self) -> dict:
        return {"ok": bool(self.key), "message": "key present" if self.key else "ORATS_API_KEY not set"}

    def fetch_minutes(self, symbol, d):
        return None  # ORATS has no minute bars; daily bars are written from hist/dailies

    def fetch_options(self, symbol, d, interval, max_dte):
        resp = self.get_json(f"{self.base}/hist/strikes", {"token": self.key, "ticker": symbol, "tradeDate": d.isoformat()})
        return self.normalize_strikes(resp.get("data", []), d, max_dte)

    @staticmethod
    def normalize_strikes(data: list[dict], d: date, max_dte: int) -> pl.DataFrame:
        rows = []
        for r in data:
            exp = date.fromisoformat(r["expirDate"])
            if not 0 <= (exp - d).days <= max_dte:
                continue
            snap = r.get("snapShotEstTime")  # HHMM ET, e.g. 1545
            hh, mm = (int(str(snap)[:-2]), int(str(snap)[-2:])) if snap else (16, 0)
            ts = int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=cal.ET).timestamp())
            common = {"ts": ts, "expiration": exp, "strike": float(r["strike"]), "underlying_price": r.get("stockPrice"), "gamma": r.get("gamma"), "vega": r.get("vega"), "quote_ts": ts}
            cd = r.get("delta")
            rows.append({**common, "right": "C", "bid": r.get("callBidPrice"), "ask": r.get("callAskPrice"), "bid_size": r.get("callBidSize"), "ask_size": r.get("callAskSize"),
                         "volume": r.get("callVolume"), "open_interest": r.get("callOpenInterest"), "iv": r.get("callMidIv"), "delta": cd, "theta": r.get("theta")})
            rows.append({**common, "right": "P", "bid": r.get("putBidPrice"), "ask": r.get("putAskPrice"), "bid_size": r.get("putBidSize"), "ask_size": r.get("putAskSize"),
                         "volume": r.get("putVolume"), "open_interest": r.get("putOpenInterest"), "iv": r.get("putMidIv"), "delta": (cd - 1) if cd is not None else None, "theta": None})
        rows = [x for x in rows if x["bid"] is not None and x["ask"] is not None]
        return frame_from_rows(rows)

    def ingest(self, symbol, start, end, interval=15, include_options=True, max_dte=60, progress=None, should_cancel=None, force=False):
        rep = super().ingest(symbol, start, end, interval, include_options, max_dte, progress, should_cancel, force)
        resp = self.get_json(f"{self.base}/hist/dailies", {"token": self.key, "ticker": symbol, "tradeDate": f"{start.isoformat()},{end.isoformat()}"})
        rows = [{"date": date.fromisoformat(r["tradeDate"]), "open": r.get("open"), "high": r.get("hiPx"), "low": r.get("loPx"), "close": r.get("clsPx"), "volume": float(r.get("stockVolume") or 0)} for r in resp.get("data", [])]
        if rows:
            self.store.write_daily(symbol, pl.DataFrame(rows), provider=self.name, source="ORATS hist/dailies")
        self.store.rebuild_manifest(symbol)
        rep["daily_bars"] = len(rows)
        return rep


PROVIDERS = {p.name: p for p in (ThetaData, Polygon, Databento, Orats)}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name}")
    return PROVIDERS[name]()
