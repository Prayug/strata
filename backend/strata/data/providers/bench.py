"""Provider verification: what each data source ACTUALLY delivers to this installation.

Advertised coverage and real entitlements differ (plan, feed, account age), so nothing here
is taken from documentation. `benchmark_source` makes a small number of real, read-only
requests and records what came back:

    reachable / authenticated      did the request succeed at all
    entitlements                   which feeds and datasets the account can read
    historical depth               the earliest observation the source returned
    speed                          bars per second and bytes per second
    rate limits                    the limit headers the provider sent, if any
    missing observations           bars expected for the sessions returned versus bars received
    contract coverage              contracts requested versus contracts that had any trade

Results are stored with the time they were measured. A source whose live check has never
run - for example Alpaca without credentials - is reported as UNVERIFIED, whatever its
mocked tests say. Nothing is written to the market store by a benchmark.

The researched list of every legitimate source (free, low-cost, broker, academic, bulk
files) lives in `market.py`, the provider marketplace, which joins it with these measurements.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone

from strata.config import get_settings
from strata.core import calendar as cal

BENCHABLE = ["yahoo", "cboe", "yahoo_options", "github_eod", "dolthub", "alpaca", "massive", "alphavantage"]

def _path():
    return get_settings().data_dir / "provider_benchmarks.json"


def saved() -> dict:
    p = _path()
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}


def _save(name: str, res: dict) -> None:
    cur = saved()
    cur[name] = res
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1, default=str))
    tmp.replace(p)


class _Probe:
    """Collects timed steps; a failing step is recorded and the rest still run."""

    def __init__(self, source: str):
        self.out = {"source": source, "measured_at": datetime.now(timezone.utc).isoformat(), "verified": False, "steps": [], "entitlements": [], "rate_limits": {}, "notes": []}
        self.bytes = 0
        self.seconds = 0.0
        self.bars = 0

    def step(self, name: str, fn):
        t0 = time.time()
        try:
            info = fn() or {}
            ok = True
        except Exception as e:  # noqa: BLE001
            info, ok = {"error": f"{type(e).__name__}: {e}"[:300]}, False
        dt = time.time() - t0
        self.seconds += dt if ok else 0
        self.out["steps"].append({"name": name, "ok": ok, "seconds": round(dt, 2), **info})
        return ok, info

    def headers(self, h) -> None:
        for k, v in dict(h).items():
            if "ratelimit" in k.lower():
                self.out["rate_limits"][k] = v

    def done(self) -> dict:
        ok = [s for s in self.out["steps"] if s["ok"]]
        self.out["verified"] = bool(ok)
        self.out["requests_ok"], self.out["requests_failed"] = len(ok), len(self.out["steps"]) - len(ok)
        if self.bars and self.seconds:
            self.out["bars_per_second"] = round(self.bars / self.seconds)
        if self.bytes and self.seconds:
            self.out["bytes_per_second"] = round(self.bytes / self.seconds)
        return self.out


def _expected_bars(days: list[date], res_seconds: int) -> int:
    n = 0
    for d in days:
        if cal.is_trading_day(d):
            o, c = cal.session_bounds(d)
            n += int((c - o).total_seconds() // res_seconds)
    return n


def benchmark_source(name: str, progress=None, should_cancel=None, symbol: str = "SPY") -> dict:
    """Run the live check of one source and store the result."""
    from strata.data.providers import sources as S
    if name not in BENCHABLE:
        raise ValueError(f"unknown source {name}")
    src = S.get_source(name)
    pr = _Probe(name)
    ok, status = src.available()
    pr.out["label"], pr.out["tier"] = src.label, src.tier
    if name == "alphavantage":
        return _alphavantage(src, pr, ok, progress)
    if not ok:
        pr.out.update(verified=False, status="unverified", reason=f"No live request could be made: {status}. Mocked tests cover the request and parsing logic only; whether this account can actually read the data is unknown.")
        _save(name, pr.out)
        return pr.out
    today = datetime.now(cal.ET).date()
    p = (lambda f, m: progress(f, m)) if progress else (lambda f, m: None)
    with S._client() as client:
        if name == "yahoo":
            def daily():
                r = src._chart(client, symbol, period1=0, period2=int(time.time()), interval="1d", events="div,splits")
                df, _ = S.Yahoo.parse_daily(r)
                pr.bars += df.height
                return {"rows": df.height, "first": str(df["date"][0]), "last": str(df["date"][-1]), "dataset": "daily bars"}
            p(0.1, "daily history")
            pr.step("daily history", daily)
            for i, (iv, rng, sec, lbl) in enumerate((("1m", "7d", 60, "1-minute bars"), ("5m", "60d", 300, "5-minute bars"), ("60m", "730d", 3600, "60-minute bars"))):
                def intraday(iv=iv, rng=rng, sec=sec, lbl=lbl):
                    r = src._chart(client, symbol, range=rng, interval=iv)
                    df = S.Yahoo.parse_intraday(r, sec)
                    days = sorted(set(df["ts"].dt.convert_time_zone("America/New_York").dt.date().to_list()))
                    exp = _expected_bars(days, sec)
                    pr.bars += df.height
                    return {"dataset": lbl, "rows": df.height, "sessions": len(days), "first": str(days[0]) if days else None, "last": str(days[-1]) if days else None,
                            "expected_bars": exp, "missing_pct": round(max(0, exp - df.height) / exp * 100, 2) if exp else None}
                p(0.25 + 0.2 * i, lbl)
                pr.step(lbl, intraday)
            pr.out["notes"].append("Yahoo sends no rate-limit headers; requests are spaced by Strata. Intraday depth is a rolling window: older sessions are not served.")
        elif name == "cboe":
            def vix():
                t0 = time.time()
                r = S.http_get(client, "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
                pr.bytes += len(r.content)
                pr.headers(r.headers)
                df = S.Cboe.parse(r.text)
                pr.bars += df.height
                return {"dataset": "VIX daily history", "rows": df.height, "first": str(df["date"][0]), "last": str(df["date"][-1]), "bytes": len(r.content), "download_seconds": round(time.time() - t0, 2)}
            pr.step("VIX history", vix)
        elif name == "yahoo_options":
            ref = S.previous_close(src.store, symbol, today + timedelta(days=1))
            if ref is None:
                pr.step("reference price", lambda: (_ for _ in ()).throw(RuntimeError(f"no daily history is stored for {symbol}; download daily bars first")))
            else:
                step = S.STRIKE_STEP.get(symbol, 1.0)
                atm = round(ref / step) * step
                days = [d for d in cal.trading_days(today - timedelta(days=9), today)][-4:]
                cs = [S.contract_from_occ(S.occ_symbol(symbol, e, r, k), source="constructed") for e in days[-2:] for r in ("C", "P") for k in (atm - step, atm, atm + step)]
                got = {"requested": 0, "with_trades": 0, "bars": 0, "first": None, "last": None, "zero_volume": 0}

                def bars():
                    for i, c in enumerate(cs):
                        if should_cancel and should_cancel():
                            break
                        p(0.1 + 0.8 * i / len(cs), f"contract {i + 1} of {len(cs)}")
                        src.throttle.wait() if hasattr(src, "throttle") else time.sleep(0.25)
                        df, st, _n = src.fetch_batch(client, symbol, [c], [d for d in days if d <= c.expiration])
                        got["requested"] += 1
                        if df is not None and not df.is_empty():
                            got["with_trades"] += 1
                            got["bars"] += df.height
                            ds = df["ts"].dt.convert_time_zone("America/New_York").dt.date()
                            got["first"] = min(filter(None, [got["first"], str(ds.min())]))
                            got["last"] = max(filter(None, [got["last"], str(ds.max())]))
                        got["zero_volume"] += int(sum((v or {}).get("unconfirmed_zero_volume", 0) for v in (st or {}).values() if isinstance(v, dict)))
                    pr.bars += got["bars"]
                    return {"dataset": "option 1-minute trade bars", "contracts_requested": got["requested"], "contracts_with_trades": got["with_trades"],
                            "contract_coverage_pct": round(got["with_trades"] / got["requested"] * 100, 1) if got["requested"] else None, "rows": got["bars"], "first": got["first"], "last": got["last"],
                            "discarded_zero_volume_bars": got["zero_volume"]}
                pr.step("recent at-the-money contracts", bars)

                def old():
                    e = [d for d in cal.trading_days(today - timedelta(days=40), today - timedelta(days=12))][-1]
                    c = S.contract_from_occ(S.occ_symbol(symbol, e, "C", atm), source="constructed")
                    df, _st, _n = src.fetch_batch(client, symbol, [c], [e])
                    return {"dataset": f"a contract that expired on {e}", "rows": 0 if df is None else df.height, "served": bool(df is not None and not df.is_empty())}
                ok2, info = pr.step("expired contract (depth check)", old)
                if ok2 and not info.get("served"):
                    pr.out["notes"].append("A contract that expired more than a week ago returned nothing: this source only serves contracts that are still listed or expired within about a day. History has to be collected while it is available.")
                pr.out["notes"].append("Trade prints only; no bid/ask is available from this source.")
        elif name == "github_eod":
            def tree():
                yrs = src.years(client, symbol)
                return {"dataset": f"{symbol} end-of-day option chains", "years": sorted(yrs), "first": str(min(yrs)) if yrs else None, "last": str(max(yrs)) if yrs else None, "bytes": sum(yrs.values())}
            pr.step("repository listing", tree)
            pr.out["notes"].append("A static public dataset: one bid/ask snapshot per contract per session at the close. No intraday prices.")
        elif name == "dolthub":
            def latest():
                d = src.latest_date(client)
                return {"dataset": "end-of-day option chains", "last": str(d)}

            def chain():
                d = src.latest_date(client)
                t0 = time.time()
                rows = src._q(client, f"SELECT COUNT(*) AS n, COUNT(DISTINCT expiration) AS e FROM option_chain WHERE act_symbol = '{symbol}' AND `date` = '{d}'")
                return {"dataset": f"{symbol} chain on {d}", "rows": int(rows[0]["n"]) if rows else 0, "expirations": int(rows[0]["e"]) if rows else 0, "query_seconds": round(time.time() - t0, 2)}
            pr.step("latest date", latest)
            pr.step("one chain", chain)
            pr.out["notes"].append("The public dataset lists about three expirations per date, so short-dated and 0DTE contracts are usually absent.")
        elif name == "massive":
            found: list = []
            d0 = [d for d in cal.trading_days(today - timedelta(days=60), today - timedelta(days=30))][-1]

            def contracts():
                ref = S.previous_close(src.store, symbol, d0 + timedelta(days=1)) or 0
                cs = src.list_contracts(client, symbol, d0, d0, d0 + timedelta(days=3), ref * 0.99 if ref else 0, ref * 1.01 if ref else 1e9)
                found.extend(cs[:2])
                return {"dataset": f"contracts listed on {d0} that have since expired", "contracts": len(cs)}
            ok_, info = pr.step("expired contract listing", contracts)
            pr.out["entitlements"].append({"what": "option contract reference data, expired contracts included", "granted": ok_, "detail": info.get("error", f"{info.get('contracts')} contracts")})

            def bars():
                if not found:
                    raise RuntimeError("no contract to request aggregates for")
                df, _st, _n = src.fetch_batch(client, symbol, found[:1], [d0])
                pr.bars += 0 if df is None else df.height
                return {"dataset": f"1-minute option aggregates for {found[0].occ} on {d0}", "rows": 0 if df is None else df.height, "first": str(d0), "last": str(d0)}
            ok_, info = pr.step("minute aggregates of an expired contract", bars)
            pr.out["entitlements"].append({"what": "historical option minute aggregates", "granted": ok_ and bool(info.get("rows")), "detail": info.get("error", f"{info.get('rows')} bars")})

            def depth():
                d1 = [d for d in cal.trading_days(today - timedelta(days=725), today - timedelta(days=700))][-1]
                cs = src.list_contracts(client, symbol, d1, d1, d1 + timedelta(days=3), 0, 1e9)
                return {"dataset": f"contracts listed on {d1} (about two years ago)", "contracts": len(cs), "first": str(d1) if cs else None}
            pr.step("history depth (about two years)", depth)
            pr.out["notes"].append("Trade aggregates only. Bid/ask quotes are a paid tier at this provider and are not read.")
        elif name == "alpaca":
            h = src._h()
            end = datetime.now(timezone.utc) - timedelta(minutes=30)

            def feed(feed_name):
                def go():
                    r = S.http_get(client, f"{src.DATA}/v2/stocks/bars", params={"symbols": symbol, "timeframe": "1Min", "start": (end - timedelta(days=5)).isoformat(), "end": end.isoformat(), "limit": 1000, "feed": feed_name}, headers=h, retries=1)
                    pr.headers(r.headers)
                    pr.bytes += len(r.content)
                    n = len((r.json().get("bars") or {}).get(symbol) or [])
                    pr.bars += n
                    return {"dataset": f"1-minute stock bars, {feed_name.upper()} feed", "rows": n}
                return go
            for f_ in ("iex", "sip"):
                ok_, info = pr.step(f"stock bars ({f_} feed)", feed(f_))
                pr.out["entitlements"].append({"what": f"stock bars, {f_.upper()} feed", "granted": ok_, "detail": info.get("error", f"{info.get('rows')} recent bars")})
            p(0.3, "stock history depth")

            def depth():
                r = S.http_get(client, f"{src.DATA}/v2/stocks/bars", params={"symbols": symbol, "timeframe": "1Min", "start": "2016-01-04T14:30:00Z", "end": "2016-01-08T21:00:00Z", "limit": 10, "feed": "iex"}, headers=h, retries=1)
                b = (r.json().get("bars") or {}).get(symbol) or []
                return {"dataset": "earliest 1-minute stock bars", "rows": len(b), "first": b[0]["t"][:10] if b else None}
            pr.step("stock history depth (2016)", depth)
            p(0.5, "option contracts")
            found: list = []

            def contracts():
                d0 = S.ALPACA_OPTIONS_FIRST + timedelta(days=14)
                cs = src.list_contracts(client, symbol, d0, d0, d0 + timedelta(days=7), 0, 1e9)
                found.extend(cs[:6])
                return {"dataset": f"contracts listed for {d0} (expired)", "contracts": len(cs), "adjusted": sum(1 for c in cs if c.adjusted)}
            ok_, info = pr.step("expired contract listing (February 2024)", contracts)
            pr.out["entitlements"].append({"what": "option contract definitions, including expired contracts", "granted": ok_, "detail": info.get("error", f"{info.get('contracts')} contracts")})

            def opt_bars():
                if not found:
                    raise RuntimeError("no contract to request bars for")
                d0 = S.ALPACA_OPTIONS_FIRST + timedelta(days=14)
                df, _st, _n = src.fetch_batch(client, symbol, found, [d0])
                pr.bars += 0 if df is None else df.height
                return {"dataset": f"option 1-minute trade bars on {d0}", "contracts_requested": len(found), "contracts_with_trades": int(df["occ"].n_unique()) if df is not None and not df.is_empty() else 0, "rows": 0 if df is None else df.height}
            ok_, info = pr.step("option trade bars at the start of the history", opt_bars)
            pr.out["entitlements"].append({"what": "historical option trade bars (indicative feed)", "granted": ok_ and bool(info.get("rows")), "detail": info.get("error", f"{info.get('rows')} bars for {info.get('contracts_with_trades')} of {info.get('contracts_requested')} contracts")})

            def quotes():
                if not found:
                    raise RuntimeError("no contract to request quotes for")
                r = S.http_get(client, f"{src.DATA}/v1beta1/options/quotes/latest", params={"symbols": found[0].occ, "feed": "opra"}, headers=h, retries=1)
                return {"dataset": "option quotes, OPRA feed", "rows": len(r.json().get("quotes") or {})}
            ok_, info = pr.step("option quotes (OPRA feed)", quotes)
            pr.out["entitlements"].append({"what": "option bid/ask quotes, OPRA feed", "granted": ok_, "detail": info.get("error", "accessible")})
            pr.out["notes"].append("Historical option QUOTES are not offered on Alpaca's free plan; option history from Alpaca is trade-print based.")
        p(1.0, "done")
    out = pr.done()
    out["status"] = "verified" if out["verified"] and not out["requests_failed"] else "partly verified" if out["verified"] else "failed"
    depth = [s for s in out["steps"] if s.get("first")]
    out["historical_depth"] = [{"dataset": s.get("dataset"), "first": s.get("first"), "last": s.get("last")} for s in depth]
    _save(name, out)
    return {k: v for k, v in out.items()}


def _alphavantage(src, pr: _Probe, has_key: bool, progress=None) -> dict:
    """Two separate facts: (1) the endpoint and Strata's parser work, verified with the
    public demo key the documentation itself publishes (IBM only); (2) whether THIS
    installation can read other symbols, which needs the user's own premium key."""
    from strata.data.providers import sources as S
    with S._client() as client:
        def demo():
            df = src.fetch(client, "IBM", date(2017, 11, 15), key="demo")
            pr.bars += df.height
            ok_rows = df.filter((df["bid"] <= df["ask"]) & df["delta"].is_not_null())
            return {"dataset": "IBM chain on 2017-11-15 (documentation's demo key)", "rows": df.height, "expirations": int(df["expiration"].n_unique()), "usable_rows": ok_rows.height, "first": "2017-11-15", "last": "2017-11-15"}
        ok_demo, _ = pr.step("endpoint and parser (public demo key, IBM only)", demo)
        pr.out["entitlements"].append({"what": "historical option chains - endpoint reachable and parsed", "granted": ok_demo, "detail": "verified with the provider's public demo key, which serves IBM only"})
        if has_key:
            def own():
                d0 = [d for d in cal.trading_days(date.today() - timedelta(days=20), date.today() - timedelta(days=3))][-1]
                df = src.fetch(client, "SPY", d0)
                return {"dataset": f"SPY chain on {d0} (your key)", "rows": df.height, "first": str(d0), "last": str(d0)}
            ok_own, info = pr.step("SPY chain with your key", own)
            pr.out["entitlements"].append({"what": "historical option chains for SPY with your key", "granted": ok_own and bool(info.get("rows")), "detail": info.get("error", f"{info.get('rows')} rows")})
        else:
            pr.out["entitlements"].append({"what": "historical option chains for SPY, QQQ, IWM", "granted": None, "detail": "not tested: no Alpha Vantage key is configured. The endpoint is premium for every symbol except the demo one."})
    out = pr.done()
    full = has_key and all(e["granted"] for e in out["entitlements"])
    out["status"] = "verified" if full else "parser verified (demo symbol only)" if out["steps"] and out["steps"][0]["ok"] else "failed"
    out["verified"] = bool(full)
    out["historical_depth"] = [{"dataset": s.get("dataset"), "first": s.get("first"), "last": s.get("last")} for s in out["steps"] if s.get("first")]
    out["notes"].append("End-of-day chains only (one snapshot per session). SPY, QQQ and IWM are not confirmed until a key with access is configured and this check is run again.")
    _save("alphavantage", out)
    return out


def catalog(store=None) -> dict:
    """Every source: what it claims, whether it is configured, and what the last live check measured."""
    from strata.data.providers import sources as S
    res = saved()
    rows = []
    for name in BENCHABLE:
        try:
            src = S.get_source(name, store)
        except Exception:  # noqa: BLE001
            continue
        ok, status = src.available()
        b = res.get(name)
        rows.append({"name": name, "label": src.label, "tier": src.tier, "description": src.description, "available": ok, "availability": status, "datasets": src.datasets,
                     "verification": (b or {}).get("status") or ("unverified" if not ok else "not checked yet"), "benchmark": b,
                     "claim": [{"data": c.get("dtype"), "observation": c.get("obs"), "resolution": c.get("res"), "first": str(c.get("first")) if c.get("first") else None, "last": str(c.get("last")) if c.get("last") else None, "note": c.get("note")}
                               for c in src.capabilities()]})
    return {"sources": rows, "note": "\"Verified\" means a live request from this installation succeeded on the date shown. Anything else is a claim from documentation or from mocked tests."}
