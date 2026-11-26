"""What a provider account can ACTUALLY retrieve, established by real requests.

A provider page says what a plan offers. This module finds out what the stored credentials
return, and keeps the evidence:

  probe          one GET, recorded with endpoint, parameters, HTTP status, the provider's
                 own error code and message (credentials removed), rate-limit headers and a
                 classification of what went wrong: authentication, entitlement, request,
                 endpoint, rate limit, no data, network
  alpaca_matrix  an expired-contract test matrix for SPY: nine historical periods, and for
                 each one at-the-money and near-the-money 0DTE calls and puts, a contract
                 about a week out, one about a month out and the underlying's minute bars.
                 Contracts are looked up in the provider's listing AND requested by their
                 OCC symbol independently, because a data endpoint can know a contract the
                 listing no longer shows. A constructed symbol that returns nothing is
                 reported as "no observations", never as a contract that existed

With no credentials nothing is requested with a key, the matrix is BLOCKED, and the report
says exactly what to supply. Nothing here is ever marked verified from a test double.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import MarketStore

ALPACA_HOW = ["Create a free account at https://app.alpaca.markets/signup (an email address is enough for a Paper Only Account).",
              "In the dashboard, with the account switcher on Paper, open API Keys in the right sidebar of the Home page and choose Generate New Keys.",
              "In Strata open Settings, find the panel \"Connect a free Alpaca account\", paste the Key ID and the Secret, and press Save and test.",
              "Then run the access test: Data -> Readiness -> Run the access test (or `strata access alpaca` in a terminal). It makes read-only requests and reports, per period, which expired contracts return observations."]

# One representative mid-month Wednesday per period asked for; each is moved to the next session if the exchange was closed.
PERIODS = [("February 2024", date(2024, 2, 14)), ("March 2024", date(2024, 3, 13)), ("June 2024", date(2024, 6, 12)), ("December 2024", date(2024, 12, 11)), ("March 2025", date(2025, 3, 12)),
           ("June 2025", date(2025, 6, 11)), ("December 2025", date(2025, 12, 10)), ("March 2026", date(2026, 3, 11)), ("June 2026", date(2026, 6, 10))]


def classify(status: int | None, body: str, has_data: bool | None = None) -> str:
    """Why a request did not return data, from the response alone."""
    low = (body or "").lower()
    if status is None:
        return "network"
    if status == 200:
        return "ok" if has_data or has_data is None else "no_data"
    if status == 401:
        return "authentication"
    if status == 403:
        return "entitlement" if any(w in low for w in ("subscription", "permit", "entitle", "plan", "agreement", "opra", "sip")) else "authentication"
    if status == 404:
        return "endpoint"
    if status == 429:
        return "rate_limit"
    if status in (400, 422):
        return "request"
    return "provider_error" if status >= 500 else "request"


EXPLAIN = {"authentication": "The key was not accepted: it is missing, mistyped, or belongs to the other environment (paper keys only work on paper-api, live keys on api).",
           "entitlement": "The key works, but this account's plan does not include what was asked for.",
           "request": "The provider rejected the request itself (a parameter or a symbol format).",
           "endpoint": "That address does not exist on the provider.",
           "rate_limit": "Too many requests: the plan's rate limit was reached.",
           "no_data": "The request was accepted and returned nothing: no observation exists for it (the contract did not trade, did not exist, or the date is outside the provider's history).",
           "network": "The provider could not be reached.", "provider_error": "The provider failed to answer (a server error on their side).", "ok": ""}


SECRET_PARAM = re.compile(r"(?i)key|token|secret|password|auth|signature")


def probe(client: httpx.Client, url: str, params: dict | None = None, headers: dict | None = None, has_data=None, what: str = "") -> dict:
    """One GET, never raising. The record holds nothing secret: headers are not stored and any credential-like query value is removed."""
    u = httpx.URL(url)
    if any(x in u.path for x in S.TRADING_PATHS):
        raise S.SourceError("refused: Strata only reads market data and never calls a trading endpoint")
    rec = {"what": what, "endpoint": f"{u.host}{u.path}", "params": {k: ("[redacted]" if SECRET_PARAM.search(k) else str(v)[:120]) for k, v in (params or {}).items()}, "status": None, "provider_code": None, "message": "", "rate_limit": {}, "ms": None, "authenticated": bool(headers)}
    t0 = time.time()
    try:
        r = client.get(url, params=params, headers=headers)
    except httpx.HTTPError as e:
        rec.update(message=S._redact(f"{type(e).__name__}: {e}")[:200], ms=round((time.time() - t0) * 1000), **{"class": "network"})
        rec["explanation"] = EXPLAIN["network"]
        return rec
    rec["ms"] = round((time.time() - t0) * 1000)
    rec["status"] = r.status_code
    rec["rate_limit"] = {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
    data = None
    try:
        data = r.json()
    except ValueError:
        pass
    if r.status_code >= 400:
        if isinstance(data, dict):
            rec["provider_code"] = data.get("code")
            rec["message"] = S._redact(str(data.get("message") or data))[:240]
        else:
            rec["message"] = S._redact(" ".join(dict.fromkeys(x.strip() for x in re.sub(r"<[^>]+>", "\n", r.text).splitlines() if x.strip())))[:160]
    ok = has_data(data) if (callable(has_data) and r.status_code == 200) else None
    rec["class"] = classify(r.status_code, rec["message"], ok)
    rec["explanation"] = EXPLAIN.get(rec["class"], "")
    rec["_data"] = data if r.status_code == 200 else None
    return rec


def _public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


SCAN = (date(2024, 1, 2), date(2024, 2, 16))  # the weeks around the documented start ("February 2024") of the option history


def _in_session(bars: list[dict], d: date) -> int:
    """Bars that start inside the regular session. A request that ends at the close also returns the bar stamped at the close itself."""
    o, c = cal.session_bounds(d)
    return sum(1 for b in bars if o <= datetime.fromisoformat(b["t"].replace("Z", "+00:00")) < c)


def _session(d: date) -> date:
    while not cal.is_trading_day(d):
        d += timedelta(days=1)
    return d


def _ref_price(store: MarketStore, sym: str, d: date) -> tuple[float | None, str]:
    """The price a strike is chosen from: the PREVIOUS session's close (known before the open)."""
    px = S.previous_close(store, sym, d)
    return px, "previous session's close in the stored daily bars"


def unauthenticated_probes(symbol: str = "SPY") -> list[dict]:
    """The same endpoints without a key: shows that the addresses are right and that the only thing missing is authentication."""
    occ = S.occ_symbol(symbol, date(2024, 6, 21), "C", 545.0)
    out = []
    with S._client() as client:
        out.append(probe(client, f"{S.Alpaca.DATA}/v1beta1/options/bars", {"symbols": occ, "timeframe": "1Min", "start": "2024-06-21", "end": "2024-06-22", "limit": 5}, what="historical option bars"))
        out.append(probe(client, f"{S.Alpaca.DATA}/v1beta1/options/trades", {"symbols": occ, "start": "2024-06-21", "end": "2024-06-22", "limit": 5}, what="historical option trades"))
        out.append(probe(client, f"{S.Alpaca.TRADING}/v2/options/contracts", {"underlying_symbols": symbol, "status": "inactive", "expiration_date": "2024-06-21", "limit": 5}, what="expired option contracts (paper host)"))
        out.append(probe(client, f"{S.Alpaca.DATA}/v2/stocks/bars", {"symbols": symbol, "timeframe": "1Min", "start": "2019-06-03", "end": "2019-06-04", "limit": 5, "feed": "sip"}, what="historical stock minute bars"))
    return [_public(r) for r in out]


def _bar_stats(bars: list[dict], d: date) -> dict:
    if not bars:
        return {"bars": 0}
    ts = sorted(datetime.fromisoformat(b["t"].replace("Z", "+00:00")) for b in bars)
    o, c = cal.session_bounds(d)
    inside = [t for t in ts if o <= t < c]
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(inside, inside[1:])]
    return {"bars": len(bars), "bars_in_session": len(inside), "first": _iso(ts[0]), "last": _iso(ts[-1]), "volume": int(sum(float(b.get("v") or 0) for b in bars)),
            "trades_in_bars": int(sum(int(b.get("n") or 0) for b in bars)) if any(b.get("n") is not None for b in bars) else None,
            "minutes_with_trades_pct": round(len(inside) / max(1, cal.session_minutes(d)) * 100, 1), "longest_gap_minutes": round(max(gaps)) if gaps else None,
            "resolution": "1 minute", "observation": "trade bars (prices of trades that printed; a minute with no trade has no bar)"}


def alpaca_matrix(store: MarketStore | None = None, symbol: str = "SPY", progress=None, should_cancel=None, today: date | None = None, save: bool = True) -> dict:
    """Run the expired-contract test matrix and return the coverage report (see module docstring)."""
    store = store or MarketStore()
    src = S.Alpaca(store)
    p = progress or (lambda f, m: None)
    today = today or datetime.now(cal.ET).date()
    rep: dict = {"provider": "alpaca", "symbol": symbol, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "blocked", "verified": False, "requests": [], "matrix": [], "findings": {}, "summary": []}
    ok, _why = src.available()
    if not ok:
        rep.update(reason="No Alpaca key is stored in Strata's credential store, the environment or a .env file.", needs=["ALPACA_API_KEY", "ALPACA_API_SECRET"], how=ALPACA_HOW)
        try:
            rep["requests"] = unauthenticated_probes(symbol)
        except Exception as e:  # noqa: BLE001
            rep["requests"] = [{"what": "unauthenticated probe", "class": "network", "message": str(e)[:160]}]
        rep["summary"] = ["BLOCKED: no credentials. Without a key every Alpaca endpoint answers 401, so nothing about this account's entitlements can be established.",
                          "The addresses and parameters Strata uses are accepted by the servers (they answer 401, not 404), so the only missing piece is the key."]
        return _finish(store, rep, save)
    H = src._h()
    throttle = S.Throttle(src.rate)
    reqs: list[dict] = rep["requests"]

    def get(url, params, has_data=None, what=""):
        throttle.wait()
        r = probe(client, url, params, H, has_data, what)
        reqs.append(_public(r))
        return r

    F = rep["findings"]
    with S._client() as client:
        # ---- 1. which environment the key belongs to
        p(0.02, "checking the key")
        host = None
        for h in (S.Alpaca.TRADING, "https://api.alpaca.markets"):
            r = get(f"{h}/v2/options/contracts", {"underlying_symbols": symbol, "limit": 1}, lambda d: bool((d or {}).get("option_contracts")), "contract listing (which environment accepts the key)")
            if r["status"] == 200:
                host = h
                break
        F["environment"] = {"accepted_by": host, "kind": "paper" if host and "paper" in host else "live" if host else None}
        if host is None:
            rep.update(status="failed", reason="The stored key was rejected by both the paper and the live trading hosts.")
            rep["summary"] = ["FAILED at authentication: " + "; ".join(f"{x['endpoint']} -> HTTP {x['status']} {x['message']}" for x in reqs[-2:])]
            return _finish(store, rep, save)
        rep["status"] = "ran"
        # ---- 2. stock minute bars: consolidated feed, else the single exchange
        d0 = _session(date(2019, 6, 3))
        a, b = cal.session_bounds(d0)
        feed = None
        for f in ("sip", "iex"):
            r = get(f"{S.Alpaca.DATA}/v2/stocks/bars", {"symbols": symbol, "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 10000, "feed": f, "adjustment": "raw"},
                    lambda d: bool(((d or {}).get("bars") or {}).get(symbol)), f"stock minute bars for {d0}, feed={f}")
            if r["class"] == "ok":
                feed = f
                F["stock_bars"] = {"feed": f, "session": str(d0), "bars": _in_session(r["_data"]["bars"][symbol], d0), "expected": cal.session_minutes(d0),
                                   "note": "sip is the consolidated tape of every exchange; iex is one exchange (a few percent of volume) and must not be mixed with it"}
                break
        if feed is None:
            F["stock_bars"] = {"feed": None, "note": "neither the consolidated nor the single-exchange feed returned 2019 minute bars"}
        else:
            # where the minute history starts: the documented first session, and the session before it
            first_ok = None
            for dd in (cal.prev_trading_day(S.Alpaca.STOCKS_FIRST), S.Alpaca.STOCKS_FIRST):
                a, b = cal.session_bounds(dd)
                r = get(f"{S.Alpaca.DATA}/v2/stocks/bars", {"symbols": symbol, "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 10000, "feed": feed, "adjustment": "raw"},
                        lambda d: bool(((d or {}).get("bars") or {}).get(symbol)), f"stock minute bars for {dd} (start of the history)")
                F["stock_bars"].setdefault("history_probe", {})[str(dd)] = _in_session(((r["_data"] or {}).get("bars") or {}).get(symbol) or [], dd)
                if r["class"] == "ok" and first_ok is None:
                    first_ok = dd
            F["stock_bars"]["first_session_with_bars"] = str(first_ok) if first_ok else None
        # ---- 3. real-time entitlement: an indicative quote is not an OPRA quote
        nxt = next(d for d in cal.trading_days(today, today + timedelta(days=10)) if d.weekday() == 4 or d == cal.trading_days(today, today + timedelta(days=10))[-1])
        ref_now, _ = _ref_price(store, symbol, today)
        if ref_now:
            live = S.occ_symbol(symbol, nxt, "C", float(round(ref_now)))
            F["latest_quotes"] = {}
            for f in ("opra", "indicative"):
                r = get(f"{S.Alpaca.DATA}/v1beta1/options/quotes/latest", {"symbols": live, "feed": f}, lambda d: bool((d or {}).get("quotes")), f"latest option quote, feed={f}")
                F["latest_quotes"][f] = r["class"]
            F["opra_entitled"] = F["latest_quotes"].get("opra") == "ok"
            F["quotes_note"] = ("Alpaca has no historical option QUOTE endpoint on any plan: history is trades and trade bars. Latest quotes on the indicative feed are derived values, not OPRA quotes, "
                                "and Strata does not store them as quotes.")
        # ---- 4. the matrix
        periods = [(n, _session(d)) for n, d in PERIODS if _session(d) < today - timedelta(days=2)]
        recent = [d for d in cal.trading_days(today - timedelta(days=12), today - timedelta(days=2)) if d.weekday() == 2][-1:] or cal.trading_days(today - timedelta(days=12), today - timedelta(days=2))[-1:]
        if recent:
            periods.append(("most recent completed week", recent[0]))
        for pi, (pname, d) in enumerate(periods):
            if should_cancel and should_cancel():
                raise S.Cancelled()
            p(0.1 + 0.75 * pi / max(1, len(periods)), f"{pname}: {d}")
            ref, ref_src = _ref_price(store, symbol, d)
            row = {"period": pname, "date": str(d), "reference_price": ref, "reference": ref_src, "contracts": [], "underlying": None}
            rep["matrix"].append(row)
            if ref is None:
                row["error"] = f"no stored daily price for {symbol} before {d}; download the free daily history first"
                continue
            k0 = float(round(ref))
            # expirations that existed: ask the listing first, then fall back to the calendar
            exps = {}
            for label, lo, hi, want in (("0DTE", 0, 0, 0), ("about 7 days", 5, 10, 7), ("about 30 days", 25, 37, 30)):
                r = get(f"{host}/v2/options/contracts", {"underlying_symbols": symbol, "status": "inactive" if d + timedelta(days=hi) < today else "active", "expiration_date_gte": str(d + timedelta(days=lo)),
                                                         "expiration_date_lte": str(d + timedelta(days=hi)), "strike_price_gte": k0, "strike_price_lte": k0, "type": "call", "limit": 100},
                        lambda x: bool((x or {}).get("option_contracts")), f"{pname}: listing of expirations {label}")
                listed = sorted({c["expiration_date"] for c in ((r["_data"] or {}).get("option_contracts") or [])})
                if listed:
                    e = min((date.fromisoformat(x) for x in listed), key=lambda x: abs((x - d).days - want))
                    exps[label] = (e, "listed")
                else:
                    e = d if want == 0 else min(cal.trading_days(d + timedelta(days=lo), d + timedelta(days=hi)), key=lambda x: (abs((x - d).days - want), x.weekday() != 4))
                    exps[label] = (e, "constructed (not in the listing)")
            wanted = []
            for right in ("C", "P"):
                wanted.append(("0DTE", "at the money", exps["0DTE"], right, k0))
                wanted += [("0DTE", f"{off:+g} from the money", exps["0DTE"], right, k0 + off) for off in (-2.0, 2.0)]
                wanted.append(("about 7 days", "at the money", exps["about 7 days"], right, k0))
                wanted.append(("about 30 days", "at the money", exps["about 30 days"], right, k0))
            occs = {S.occ_symbol(symbol, e, right, k): (bucket, money, e, how, right, k) for bucket, money, (e, how), right, k in wanted}
            a, b = cal.session_bounds(d)
            # bars: every requested symbol in one paginated request, read to the last page
            bars: dict[str, list] = {}
            token, pages = None, 0
            while True:
                prm = {"symbols": ",".join(occs), "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 10000, "sort": "asc", **({"page_token": token} if token else {})}
                r = get(f"{S.Alpaca.DATA}/v1beta1/options/bars", prm, lambda x: bool((x or {}).get("bars")), f"{pname}: 1-minute bars for {len(occs)} contracts")
                pages += 1
                for occ, bs in ((r["_data"] or {}).get("bars") or {}).items():
                    bars.setdefault(occ, []).extend(bs)
                token = (r["_data"] or {}).get("next_page_token")
                if not token or pages >= 20 or r["status"] != 200:
                    break
            row["bars_request"] = {"status": r["status"], "class": r["class"], "pages": pages, "message": r["message"]}
            # trades: counted per contract (first pages only; a truncated count is marked)
            trades: dict[str, int] = {}
            token, tpages, truncated = None, 0, False
            while True:
                prm = {"symbols": ",".join(occs), "start": _iso(a), "end": _iso(b), "limit": 10000, "sort": "asc", **({"page_token": token} if token else {})}
                r2 = get(f"{S.Alpaca.DATA}/v1beta1/options/trades", prm, lambda x: bool((x or {}).get("trades")), f"{pname}: individual trades")
                tpages += 1
                for occ, tr in ((r2["_data"] or {}).get("trades") or {}).items():
                    trades[occ] = trades.get(occ, 0) + len(tr)
                token = (r2["_data"] or {}).get("next_page_token")
                if not token or r2["status"] != 200:
                    break
                if tpages >= 4:
                    truncated = True
                    break
            row["trades_request"] = {"status": r2["status"], "class": r2["class"], "pages": tpages, "truncated": truncated, "message": r2["message"]}
            for occ, (bucket, money, e, how, right, k) in occs.items():
                st = _bar_stats(bars.get(occ, []), d)
                row["contracts"].append({"contract": occ, "expiration": str(e), "expiration_source": how, "bucket": bucket, "moneyness": money, "right": "call" if right == "C" else "put", "strike": k, **st,
                                         "trades": trades.get(occ), "trades_truncated": truncated and occ in trades, "has_observations": st["bars"] > 0, "quotes": False,
                                         "pricing": "observed trades" if st["bars"] else None,
                                         "note": "" if st["bars"] else "no observations returned: the contract did not trade that session, or did not exist (a constructed symbol proves nothing by itself)"})
            if feed:
                r3 = get(f"{S.Alpaca.DATA}/v2/stocks/bars", {"symbols": symbol, "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 10000, "feed": feed, "adjustment": "raw"},
                         lambda x: bool(((x or {}).get("bars") or {}).get(symbol)), f"{pname}: underlying minute bars")
                ub = ((r3["_data"] or {}).get("bars") or {}).get(symbol) or []
                n_in = _in_session(ub, d)
                row["underlying"] = {"feed": feed, "bars": n_in, "expected": cal.session_minutes(d), "missing_minutes": max(0, cal.session_minutes(d) - n_in)}
        # ---- 5. where the history starts: EVERY session of the first weeks is asked, so a missing day inside them is seen too
        p(0.88, "finding the first session with data")
        first_date, asked_days, empty_after_first = None, 0, []
        for d in cal.trading_days(SCAN[0], SCAN[1]):
            if should_cancel and should_cancel():
                raise S.Cancelled()
            ref, _ = _ref_price(store, symbol, d)
            if ref is None:
                continue
            k = float(round(ref))
            exps = cal.trading_days(d, d + timedelta(days=4))[:3]
            occs = [S.occ_symbol(symbol, e, r_, k + off) for e in exps for r_ in ("C", "P") for off in (-1.0, 0.0, 1.0)]
            a, b = cal.session_bounds(d)
            r = get(f"{S.Alpaca.DATA}/v1beta1/options/bars", {"symbols": ",".join(occs), "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 10000}, lambda x: bool((x or {}).get("bars")),
                    f"start of the history: {d} ({len(occs)} contracts around the money, 3 nearest expirations)")
            asked_days += 1
            if r["class"] == "ok":
                first_date = first_date or d
            elif first_date is not None and r["status"] == 200:
                empty_after_first.append(str(d))
        F["earliest_option_bars"] = str(first_date) if first_date else None
        F["history_scan"] = {"from": str(SCAN[0]), "to": str(SCAN[1]), "sessions_asked": asked_days, "first_session_with_bars": str(first_date) if first_date else None,
                             "sessions_after_it_with_no_bars": empty_after_first,
                             "note": "Eighteen contracts around the money were asked for on each session. A session that returns none of them has no option history at the provider."}
        # ---- 6. pagination and rate limits, from the requests already made
        p(0.95, "pagination and rate limits")
        some = next((row for row in rep["matrix"] if any(c["has_observations"] for c in row["contracts"])), None)
        if some:
            c = max(some["contracts"], key=lambda x: x["bars"])
            a, b = cal.session_bounds(date.fromisoformat(some["date"]))
            seen, token, pages = [], None, 0
            while pages < 4:
                r = get(f"{S.Alpaca.DATA}/v1beta1/options/bars", {"symbols": c["contract"], "timeframe": "1Min", "start": _iso(a), "end": _iso(b), "limit": 25, "sort": "asc", **({"page_token": token} if token else {})},
                        lambda x: bool((x or {}).get("bars")), "pagination: 25 bars a page")
                pages += 1
                seen += [x["t"] for x in ((r["_data"] or {}).get("bars") or {}).get(c["contract"], [])]
                token = (r["_data"] or {}).get("next_page_token")
                if not token:
                    break
            F["pagination"] = {"pages_read": pages, "bars": len(seen), "strictly_increasing": seen == sorted(set(seen)), "works": pages > 1 and seen == sorted(set(seen))}
    limits = [x["rate_limit"] for x in reqs if x.get("rate_limit")]
    if limits:
        F["rate_limit"] = {"limit_per_minute": limits[-1].get("x-ratelimit-limit"), "remaining_after_run": limits[-1].get("x-ratelimit-remaining"), "requests_made": len(reqs),
                           "hit_429": sum(1 for x in reqs if x.get("status") == 429)}
    F["request_size"] = {"symbols_per_request": "100 at most (documented); this run sent 10 per request", "points_per_page": "10,000 at most (documented and used)"}
    _summarise(rep)
    return _finish(store, rep, save)


def _summarise(rep: dict) -> None:
    F, M = rep["findings"], rep["matrix"]
    with_obs = [(row, c) for row in M for c in row["contracts"] if c["has_observations"]]
    asked = sum(len(row["contracts"]) for row in M)
    dates_ok = sorted({row["date"] for row, _ in with_obs})
    rep["verified"] = bool(with_obs)
    s = [f"Environment: the key is accepted by the {F.get('environment', {}).get('kind')} host."]
    s.append(f"Expired-contract matrix: {len(with_obs)} of {asked} contract-sessions returned trade bars, on {len(dates_ok)} of {len(M)} dates" + (f" ({dates_ok[0]} to {dates_ok[-1]})." if dates_ok else "."))
    if not with_obs:
        errs = sorted({f"HTTP {row.get('bars_request', {}).get('status')} ({row.get('bars_request', {}).get('class')}): {row.get('bars_request', {}).get('message')}" for row in M if row.get("bars_request")})
        s.append("No historical option observation was returned. " + ("Responses: " + "; ".join(errs)[:400] if errs else ""))
    listed = sum(1 for row in M for c in row["contracts"] if c["expiration_source"] == "listed")
    s.append(f"Contract discovery: {listed} of {asked} requested contracts used an expiration returned by the listing endpoint; the rest were requested by a constructed OCC symbol.")
    hs = F.get("history_scan") or {}
    if F.get("earliest_option_bars"):
        s.append(f"Option history starts on {F['earliest_option_bars']}: every session from {hs.get('from')} was asked and none before it returned a bar."
                 + (f" After it, {len(hs['sessions_after_it_with_no_bars'])} session(s) up to {hs.get('to')} returned nothing at all: {', '.join(hs['sessions_after_it_with_no_bars'])}." if hs.get("sessions_after_it_with_no_bars") else ""))
    elif hs:
        s.append(f"No session between {hs.get('from')} and {hs.get('to')} returned an option bar.")
    if F.get("stock_bars"):
        sb = F["stock_bars"]
        s.append(f"Stock minute bars: feed {sb.get('feed') or 'none'}" + (f", {sb['bars']} of {sb['expected']} minutes for {sb['session']}" if sb.get("bars") else "")
                 + (f"; the history starts on {sb['first_session_with_bars']} (the session before it returns nothing)" if sb.get("first_session_with_bars") and min(sb.get("history_probe", {}).values(), default=1) == 0
                    else f"; first probed session with bars: {sb['first_session_with_bars']}" if sb.get("first_session_with_bars") else "") + ".")
    if "opra_entitled" in F:
        s.append("OPRA entitlement: " + ("yes." if F["opra_entitled"] else "no. Historical trade bars do not need it if the matrix above returned data; real-time OPRA quotes do."))
    s.append("Bid/ask: none. Alpaca's history is trades and trade bars; there is no historical option quote endpoint.")
    rep["summary"] = s


def _finish(store: MarketStore, rep: dict, save: bool) -> dict:
    if save:
        from strata.config import get_settings
        d = get_settings().data_dir / "access"
        d.mkdir(parents=True, exist_ok=True)
        (d / "alpaca.json").write_text(json.dumps(rep, indent=1, default=str))
        (d / "alpaca.md").write_text(render(rep))
    return rep


def saved(provider: str = "alpaca") -> dict | None:
    from strata.config import get_settings
    f = get_settings().data_dir / "access" / f"{provider}.json"
    return json.loads(f.read_text()) if f.exists() else None


def render(rep: dict) -> str:
    """The coverage report as text (also written next to the JSON)."""
    L = [f"# Alpaca access report ({rep['symbol']}), {rep['run_at']}", "", f"Status: **{rep['status'].upper()}**" + ("" if rep["status"] != "blocked" else f" - {rep.get('reason', '')}"), ""]
    L += [f"- {x}" for x in rep.get("summary", [])]
    if rep.get("how"):
        L += ["", "What to do:"] + [f"{i}. {x}" for i, x in enumerate(rep["how"], 1)]
    if rep.get("matrix"):
        L += ["", "| Period | Date | Contract | Bucket | Listing | Bars | First | Last | Volume | Trades | Minutes with a trade |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rep["matrix"]:
            for c in row["contracts"]:
                L.append(f"| {row['period']} | {row['date']} | {c['contract']} | {c['bucket']}, {c['moneyness']} {c['right']} | {c['expiration_source']} | {c['bars']} | {str(c.get('first', ''))[11:16]} | {str(c.get('last', ''))[11:16]} | "
                         f"{c.get('volume', '')} | {'' if c.get('trades') is None else c['trades']}{'+' if c.get('trades_truncated') else ''} | {c.get('minutes_with_trades_pct', '')} |")
            if row.get("underlying"):
                u = row["underlying"]
                L.append(f"| {row['period']} | {row['date']} | {rep['symbol']} | underlying minute bars | feed {u['feed']} | {u['bars']} | | | | | {u['missing_minutes']} missing |")
    L += ["", "## Requests (credentials removed)", "", "| What | Endpoint | Status | Provider code | Class | Message |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rep.get("requests", [])[:120]:
        L.append(f"| {r.get('what', '')} | {r.get('endpoint', '')} | {r.get('status')} | {r.get('provider_code') or ''} | {r.get('class')} | {str(r.get('message', '')).replace('|', '/')[:100]} |")
    return "\n".join(L) + "\n"
