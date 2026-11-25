"""Provider adapters normalise vendor data into the store schema.

All adapters share one ingestion loop: for each trading session in the range, skip it
if it is already stored (columnar cache), otherwise download, normalise, validate and
write it. Raw HTTP responses are cached under <data>/cache/<provider>/ so re-running an
ingest never re-downloads the same payload.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from strata.config import get_settings
from strata.core import calendar as cal
from strata.data.store import KNOWN_SPECS, MarketStore, SymbolMeta, minutes_to_daily


class ProviderError(Exception):
    pass


class Provider:
    name = "base"
    label = "Base"
    resolution = "1m"
    docs = ""
    supports_options = True

    def __init__(self, store: MarketStore | None = None):
        self.store = store or MarketStore()
        self.cache = get_settings().cache_dir / self.name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=60)

    # ---------- to implement
    def configured(self) -> bool:
        return False

    def check(self) -> dict:
        return {"ok": self.configured(), "message": "configured" if self.configured() else "not configured"}

    def fetch_minutes(self, symbol: str, d: date) -> pl.DataFrame | None:
        raise NotImplementedError

    def fetch_options(self, symbol: str, d: date, interval: int, max_dte: int) -> pl.DataFrame | None:
        raise NotImplementedError

    # ---------- helpers
    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3) -> dict | list:
        key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        cp = self.cache / f"{key[:2]}/{key}.json"
        if cp.exists():
            return json.loads(cp.read_text())
        last = None
        for attempt in range(retries):
            try:
                r = self.client.get(url, params=params, headers=headers)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code >= 400:
                    raise ProviderError(f"{self.label} HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data))
                return data
            except httpx.HTTPError as e:
                last = e
                time.sleep(1 + attempt)
        raise ProviderError(f"{self.label} request failed: {last}")

    @staticmethod
    def snapshot_times(d: date, interval: int) -> list[int]:
        o, c = cal.session_bounds(d)
        t0, t1 = int(o.timestamp()), int(c.timestamp())
        return list(range(t0, t1 + 1, interval * 60))

    # ---------- shared ingestion loop
    def ingest(self, symbol: str, start: date, end: date, interval: int = 15, include_options: bool = True, max_dte: int = 60,
               progress=None, should_cancel=None, force: bool = False) -> dict:
        if not self.configured():
            raise ProviderError(f"{self.label} is not configured: {self.docs}")
        days = cal.trading_days(start, end)
        res = "EOD" if self.resolution == "EOD" else f"{interval}m"
        # only sessions recorded as COMPLETE are skipped: a file with some rows is not proof of a full download
        have_opt = set(self.store.option_days(symbol, "quote", provider=self.name, complete_only=True))
        report = {"provider": self.name, "symbol": symbol, "sessions": len(days), "fetched_minutes": 0, "fetched_option_sessions": 0, "skipped_cached": 0, "missing": []}
        minute_frames = []
        for i, d in enumerate(days):
            if should_cancel and should_cancel():
                from strata.engine.engine import BacktestCancelled
                raise BacktestCancelled()
            try:
                m = self.fetch_minutes(symbol, d)
                if m is not None and not m.is_empty():
                    minute_frames.append(m)
                    report["fetched_minutes"] += m.height
                else:
                    report["missing"].append({"date": d.isoformat(), "what": "underlying"})
                if include_options and self.supports_options:
                    if d in have_opt and not force:
                        report["skipped_cached"] += 1
                    else:
                        o = self.fetch_options(symbol, d, interval, max_dte)
                        if o is not None and not o.is_empty():
                            snaps = o["ts"].n_unique()
                            want = 1 if res == "EOD" else cal.session_minutes(d) // max(1, interval) + 1
                            status = "complete" if snaps >= 0.98 * want else "partial"
                            self.store.write_option_day(symbol, d, o, provider=self.name, res=res, status=status, source=f"{self.label} API", license=self.label, url=self.docs)
                            report["fetched_option_sessions"] += 1
                            if status == "partial":
                                report.setdefault("partial_sessions", []).append({"date": d.isoformat(), "snapshots": int(snaps), "expected": int(want)})
                        else:
                            report["missing"].append({"date": d.isoformat(), "what": "options"})
            except ProviderError as e:
                report["missing"].append({"date": d.isoformat(), "what": "error", "error": str(e)[:200]})
            if progress:
                progress((i + 1) / len(days), f"{symbol} {d}")
            if len(minute_frames) >= 20:
                self.store.write_minutes(symbol, pl.concat(minute_frames), provider=self.name, source=f"{self.label} API")
                minute_frames = []
        if minute_frames:
            self.store.write_minutes(symbol, pl.concat(minute_frames), provider=self.name, source=f"{self.label} API")
        mins = self.store.read_minutes(symbol, provider=self.name)
        if not mins.is_empty():
            self.store.write_daily(symbol, minutes_to_daily(mins), provider=self.name, source="aggregated from this provider's 1-minute bars")
        if symbol not in self.store.symbols():
            self.store.upsert_symbol(SymbolMeta(symbol=symbol, **KNOWN_SPECS.get(symbol, {})))
        if include_options:
            from strata.data.derived import derive_iv_history
            ds = self.store.option_dataset(symbol, "quote", provider=self.name)
            if ds is not None:
                derive_iv_history(self.store, symbol, dataset=ds)
        self.store.rebuild_manifest(symbol)
        return report


def et_ms_of_day_to_utc(d: date, ms: int) -> int:
    """Vendor 'milliseconds since local (ET) midnight' -> epoch seconds. DST-safe: the
    clock change happens at 02:00 on Sundays, never inside a trading session."""
    return int(datetime(d.year, d.month, d.day, tzinfo=cal.ET).timestamp()) + int(ms) // 1000


def frame_from_rows(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    conv = [pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))]
    if "quote_ts" in df.columns:
        conv.append(pl.from_epoch("quote_ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")))
    if "expiration" in df.columns:
        conv.append(pl.col("expiration").cast(pl.Date))
    return df.with_columns(conv)
