"""V6: what an Alpaca account can actually retrieve.

Everything here runs against a test double of Alpaca's HTTP API. It proves that the access
check reads responses correctly: classification of refusals, pagination, the listing/data
distinction, sanitisation and the blocked state. It proves NOTHING about what a real account
returns. That is `strata access alpaca`, which needs a key and reports BLOCKED without one.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.providers import access as A
from strata.data.providers import sources as S
from strata.data.store import MarketStore

TODAY = date(2026, 11, 25)
KEY, SECRET = "test-key-id", "test-secret-value"


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class FakeAlpaca:
    """A small stand-in for Alpaca: option history starts on `first`, the contract listing has forgotten
    expirations before `listing_from`, puts two strikes below the money never traded, the consolidated
    stock feed and OPRA quotes are refused, and `paper` says which host the key belongs to."""

    def __init__(self, first=date(2024, 2, 5), listing_from=date(2024, 6, 1), paper=True, sip=False, opra=False, valid=True):
        self.first, self.listing_from, self.paper, self.sip, self.opra, self.valid = first, listing_from, paper, sip, opra, valid
        self.calls: list[httpx.Request] = []

    def traded(self, occ: str, d: date) -> bool:
        _root, exp, right, strike = S.parse_occ(occ)
        return d >= self.first and d <= exp and not (right == "P" and strike == 498.0)

    def bars(self, occ: str, d: date) -> list[dict]:
        if not self.traded(occ, d):
            return []
        a, _b = cal.session_bounds(d)
        return [{"t": _iso(a + timedelta(minutes=m)), "o": 2.0, "h": 2.1, "l": 1.9, "c": 2.0 + m / 1000, "v": 5, "n": 2, "vw": 2.0} for m in range(0, cal.session_minutes(d), 3)]

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        q, host, path = req.url.params, req.url.host, req.url.path
        if req.headers.get("APCA-API-KEY-ID") != KEY or req.headers.get("APCA-API-SECRET-KEY") != SECRET or not self.valid:
            return httpx.Response(401, json={"message": "unauthorized."}) if "api.alpaca" in host else httpx.Response(401, text="<html><body><h1>401 Authorization Required</h1></body></html>")
        if path == "/v2/options/contracts":
            if host.startswith("paper") != self.paper:
                return httpx.Response(401, json={"message": "unauthorized."})
            if "expiration_date_gte" not in q:
                return httpx.Response(200, json={"option_contracts": [{"symbol": "SPY261218C00500000", "expiration_date": "2026-12-18"}], "next_page_token": None})
            lo, hi = date.fromisoformat(q["expiration_date_gte"]), date.fromisoformat(q["expiration_date_lte"])
            exps = [e for e in cal.trading_days(lo, hi) if e >= self.listing_from]
            return httpx.Response(200, json={"option_contracts": [{"symbol": S.occ_symbol("SPY", e, "C", float(q["strike_price_gte"])), "expiration_date": e.isoformat(), "status": q["status"], "root_symbol": "SPY",
                                                                        "strike_price": q["strike_price_gte"], "type": "call", "style": "american"} for e in exps], "next_page_token": None})
        d = datetime.fromisoformat(q["start"].replace("Z", "+00:00")).astimezone(cal.ET).date() if "start" in q else None
        if path == "/v1beta1/options/bars":
            flat = [(occ, b) for occ in sorted(q["symbols"].split(",")) for b in self.bars(occ, d)]
            off, lim = int(q.get("page_token") or 0), int(q["limit"])
            out: dict[str, list] = {}
            for occ, b in flat[off:off + lim]:
                out.setdefault(occ, []).append(b)
            return httpx.Response(200, json={"bars": out, "next_page_token": str(off + lim) if off + lim < len(flat) else None}, headers={"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": "150"})
        if path == "/v1beta1/options/trades":
            return httpx.Response(200, json={"trades": {occ: [{"t": q["start"], "p": 2.0, "s": 1, "x": "C", "c": "I"}] * 7 for occ in q["symbols"].split(",") if self.traded(occ, d)}, "next_page_token": None})
        if path == "/v1beta1/options/quotes/latest":
            if q["feed"] == "opra" and not self.opra:
                return httpx.Response(403, json={"message": "OPRA agreement is not signed"})
            return httpx.Response(200, json={"quotes": {q["symbols"]: {"bp": 1.0, "ap": 1.1}}})
        if path == "/v2/stocks/bars":
            if q["feed"] == "sip" and not self.sip:
                return httpx.Response(403, json={"code": 42210000, "message": "subscription does not permit querying recent SIP data"})
            a, _b = cal.session_bounds(d)
            return httpx.Response(200, json={"bars": {"SPY": [{"t": _iso(a + timedelta(minutes=m)), "o": 500, "h": 501, "l": 499, "c": 500, "v": 100} for m in range(cal.session_minutes(d) - 4)]}, "next_page_token": None})
        return httpx.Response(404, json={"message": "not found"})


@pytest.fixture
def store(tmp_path, monkeypatch):
    from strata.config import override_settings
    override_settings(data_dir=tmp_path)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: (KEY, SECRET))
    st = MarketStore(tmp_path / "market")
    days = cal.trading_days(date(2023, 12, 1), TODAY)
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in days]), provider="yahoo")
    return st


def use(monkeypatch, fake):
    monkeypatch.setattr(S, "_client", lambda: httpx.Client(transport=httpx.MockTransport(fake)))
    return fake


# ----------------------------------------------------------------------------- reading a refusal
def test_a_refusal_is_classified_from_the_response():
    assert A.classify(401, "unauthorized.") == "authentication"
    assert A.classify(403, "subscription does not permit querying recent SIP data") == "entitlement"
    assert A.classify(403, "OPRA agreement is not signed") == "entitlement"
    assert A.classify(403, "forbidden.") == "authentication"
    assert A.classify(404, "not found") == "endpoint"
    assert A.classify(422, "invalid symbol") == "request" and A.classify(400, "bad timeframe") == "request"
    assert A.classify(429, "too many requests") == "rate_limit"
    assert A.classify(503, "") == "provider_error" and A.classify(None, "timeout") == "network"
    assert A.classify(200, "", True) == "ok" and A.classify(200, "", False) == "no_data"
    assert set(A.EXPLAIN) >= {"authentication", "entitlement", "request", "endpoint", "rate_limit", "no_data", "network"}


def test_a_probe_records_the_response_and_no_credential():
    def handler(req):
        return httpx.Response(403, json={"code": 42210000, "message": "subscription does not permit querying recent SIP data"}, headers={"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": "199"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        r = A.probe(c, "https://data.alpaca.markets/v2/stocks/bars", {"symbols": "SPY", "feed": "sip", "apikey": "abc"}, {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}, what="stock bars")
    assert (r["status"], r["provider_code"], r["class"]) == (403, 42210000, "entitlement")
    assert r["endpoint"] == "data.alpaca.markets/v2/stocks/bars" and r["params"]["feed"] == "sip" and r["rate_limit"] == {"x-ratelimit-limit": "200", "x-ratelimit-remaining": "199"}
    blob = json.dumps(A._public(r))
    assert KEY not in blob and SECRET not in blob and "abc" not in blob and r["authenticated"] is True


def test_a_probe_survives_a_network_failure_and_refuses_trading_endpoints():
    def boom(req):
        raise httpx.ConnectError("name resolution failed")
    with httpx.Client(transport=httpx.MockTransport(boom)) as c:
        r = A.probe(c, "https://data.alpaca.markets/v1beta1/options/bars", {"symbols": "X"})
        assert r["class"] == "network" and r["status"] is None
        for url in ("https://paper-api.alpaca.markets/v2/orders", "https://api.alpaca.markets/v2/positions"):
            with pytest.raises(S.SourceError, match="never calls a trading endpoint"):
                A.probe(c, url)
    import inspect
    import re
    assert not re.search(r"\.(post|put|patch|delete)\(", inspect.getsource(A)), "the access check only reads"


# ----------------------------------------------------------------------------- no credentials
def test_without_a_key_the_matrix_is_blocked_and_says_what_to_do(store, monkeypatch, tmp_path):
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("", ""))
    fake = use(monkeypatch, FakeAlpaca())
    rep = A.alpaca_matrix(store, today=TODAY)
    assert rep["status"] == "blocked" and rep["verified"] is False and rep["matrix"] == []
    assert rep["needs"] == ["ALPACA_API_KEY", "ALPACA_API_SECRET"] and len(rep["how"]) == 4 and "app.alpaca.markets/signup" in rep["how"][0]
    assert len(rep["requests"]) == 4 and {r["class"] for r in rep["requests"]} == {"authentication"} and all(r["status"] == 401 and not r["authenticated"] for r in rep["requests"])
    assert all("apca-api-key-id" not in {k.lower() for k in c.headers} for c in fake.calls), "nothing is sent with a key when there is none"
    assert "BLOCKED" in rep["summary"][0]
    saved = A.saved("alpaca")
    assert saved["status"] == "blocked" and (tmp_path / "access" / "alpaca.md").read_text().startswith("# Alpaca access report")
    assert "BLOCKED" in A.render(rep) and "What to do" in A.render(rep)


def test_a_rejected_key_is_a_failure_not_a_block(store, monkeypatch):
    use(monkeypatch, FakeAlpaca(valid=False))
    rep = A.alpaca_matrix(store, today=TODAY, save=False)
    assert rep["status"] == "failed" and rep["verified"] is False and rep["matrix"] == []
    assert [r["endpoint"] for r in rep["requests"]] == ["paper-api.alpaca.markets/v2/options/contracts", "api.alpaca.markets/v2/options/contracts"]
    assert "authentication" in rep["summary"][0]


# ----------------------------------------------------------------------------- the matrix
def test_the_matrix_reads_listing_data_feeds_and_pagination(store, monkeypatch):
    fake = use(monkeypatch, FakeAlpaca())
    rep = A.alpaca_matrix(store, today=TODAY, save=False)
    F = rep["findings"]
    assert rep["status"] == "ran" and rep["verified"] is True
    assert F["environment"] == {"accepted_by": "https://paper-api.alpaca.markets", "kind": "paper"}
    # the consolidated feed is refused and the single exchange is used, labelled as such
    assert F["stock_bars"]["feed"] == "iex" and F["stock_bars"]["bars"] == 386 and F["stock_bars"]["expected"] == 390
    assert F["latest_quotes"] == {"opra": "entitlement", "indicative": "ok"} and F["opra_entitled"] is False
    sip = next(r for r in rep["requests"] if r["params"].get("feed") == "sip")
    assert (sip["status"], sip["provider_code"], sip["class"]) == (403, 42210000, "entitlement")
    # nine named periods plus the most recent completed week, each a real session
    assert [r["period"] for r in rep["matrix"]] == [n for n, _ in A.PERIODS] + ["most recent completed week"]
    assert all(cal.is_trading_day(date.fromisoformat(r["date"])) for r in rep["matrix"]) and rep["matrix"][-1]["date"] == "2026-11-18"
    for row in rep["matrix"]:
        cs = row["contracts"]
        assert len(cs) == 10 and {c["bucket"] for c in cs} == {"0DTE", "about 7 days", "about 30 days"} and {c["right"] for c in cs} == {"call", "put"}
        assert sum(1 for c in cs if c["bucket"] == "0DTE") == 6 and all(c["quotes"] is False for c in cs)
        d = date.fromisoformat(row["date"])
        assert all(date.fromisoformat(c["expiration"]) == d for c in cs if c["bucket"] == "0DTE")
        assert all(5 <= (date.fromisoformat(c["expiration"]) - d).days <= 10 for c in cs if c["bucket"] == "about 7 days")
        assert all(25 <= (date.fromisoformat(c["expiration"]) - d).days <= 37 for c in cs if c["bucket"] == "about 30 days")
        dead = [c for c in cs if not c["has_observations"]]
        assert [c["contract"][-9:] for c in dead] == ["P00498000"] and dead[0]["bars"] == 0 and "proves nothing" in dead[0]["note"] and dead[0]["pricing"] is None
        live = [c for c in cs if c["has_observations"]]
        assert all(c["bars"] == 130 and c["trades"] == 7 and c["volume"] == 650 and c["longest_gap_minutes"] == 3 and c["pricing"] == "observed trades" for c in live)
        assert row["underlying"] == {"feed": "iex", "bars": 386, "expected": 390, "missing_minutes": 4}
    # early periods: the listing no longer shows the expirations, the data endpoint still answers for the constructed symbol
    feb = rep["matrix"][0]
    assert {c["expiration_source"] for c in feb["contracts"]} == {"constructed (not in the listing)"} and sum(c["has_observations"] for c in feb["contracts"]) == 9
    assert {c["expiration_source"] for c in rep["matrix"][3]["contracts"]} == {"listed"}
    assert F["earliest_option_bars"] == "2024-02-05"
    assert F["pagination"] == {"pages_read": 4, "bars": 100, "strictly_increasing": True, "works": True}
    assert F["rate_limit"]["limit_per_minute"] == "200" and F["rate_limit"]["hit_429"] == 0
    # nothing secret, and nothing but GETs to read-only endpoints
    blob = json.dumps(rep)
    assert KEY not in blob and SECRET not in blob and '"_data"' not in blob
    assert {c.method for c in fake.calls} == {"GET"} and not any(x in c.url.path for c in fake.calls for x in S.TRADING_PATHS)
    text = A.render(rep)
    assert "| February 2024 | 2024-02-14 | SPY240214C00500000 |" in text and "Bid/ask: none" in text
    assert "90 of 100 contract-sessions returned trade bars, on 10 of 10 dates" in " ".join(rep["summary"])


def test_a_live_key_is_found_on_the_live_host(store, monkeypatch):
    use(monkeypatch, FakeAlpaca(paper=False))
    rep = A.alpaca_matrix(store, today=TODAY, save=False)
    assert rep["findings"]["environment"]["kind"] == "live" and rep["verified"] is True
    assert all(r["endpoint"].startswith("api.alpaca.markets") for r in rep["requests"] if "listing of expirations" in r["what"])


def test_history_that_returns_nothing_is_not_called_verified(store, monkeypatch):
    use(monkeypatch, FakeAlpaca(first=date(2030, 1, 1)))
    rep = A.alpaca_matrix(store, today=TODAY, save=False)
    assert rep["status"] == "ran" and rep["verified"] is False
    assert all(not c["has_observations"] for r in rep["matrix"] for c in r["contracts"]) and rep["findings"]["earliest_option_bars"] is None
    assert "No historical option observation was returned" in " ".join(rep["summary"]) and "pagination" not in rep["findings"]


# ----------------------------------------------------------------------------- the adapter uses the same host rule
def test_contract_listing_falls_back_to_the_live_host_for_a_live_key(store, monkeypatch):
    fake = use(monkeypatch, FakeAlpaca(paper=False, listing_from=date(2020, 1, 1)))
    src = S.Alpaca(store)
    with S._client() as client:
        cs = src.list_contracts(client, "SPY", date(2026, 11, 18), date(2026, 11, 18), date(2026, 11, 20), 500.0, 500.0, rights=("C",))
    assert [c.occ for c in cs] == ["SPY261118C00500000", "SPY261119C00500000", "SPY261120C00500000"]
    hosts = [c.url.host for c in fake.calls]
    assert hosts[0] == "paper-api.alpaca.markets" and set(hosts[1:]) == {"api.alpaca.markets"} and src._host == S.Alpaca.LIVE
    with S._client() as client:
        assert src.contracts_host(client) == S.Alpaca.LIVE


def test_the_start_of_the_history_is_found_by_asking_every_early_session(store, monkeypatch):
    """Measured on the real service: option bars start on 2024-01-18 (documented: "February 2024"), 2024-02-02 returns nothing,
    and stock minute bars start on 2016-01-04. A handful of probe dates had reported 2024-01-24 and missed the hole."""
    class Measured(FakeAlpaca):
        def traded(self, occ, d):
            return super().traded(occ, d) and d != date(2024, 2, 2)

        def __call__(self, req):
            if req.url.path == "/v2/stocks/bars" and req.headers.get("APCA-API-KEY-ID") == KEY:
                d = datetime.fromisoformat(req.url.params["start"].replace("Z", "+00:00")).astimezone(cal.ET).date()
                self.calls.append(req)
                a, b = cal.session_bounds(d)
                n = cal.session_minutes(d) + 1 if d >= date(2016, 1, 4) else 0  # the provider also returns the bar stamped at the close
                return httpx.Response(200, json={"bars": {"SPY": [{"t": _iso(a + timedelta(minutes=m)), "o": 500, "h": 501, "l": 499, "c": 500, "v": 100} for m in range(n)]} if n else {}, "next_page_token": None})
            return super().__call__(req)
    st = store
    days = cal.trading_days(date(2015, 12, 1), date(2023, 11, 30))
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in days]), provider="yahoo")
    use(monkeypatch, Measured(first=date(2024, 1, 18), sip=True))
    rep = A.alpaca_matrix(st, today=TODAY, save=False)
    F = rep["findings"]
    hs = F["history_scan"]
    assert F["earliest_option_bars"] == "2024-01-18" and hs["first_session_with_bars"] == "2024-01-18"
    assert hs["sessions_asked"] == len(cal.trading_days(date(2024, 1, 2), date(2024, 2, 16))), "every session of the first weeks, not a sample of them"
    assert hs["sessions_after_it_with_no_bars"] == ["2024-02-02"], "a hole inside the history is reported, not stepped over"
    sb = F["stock_bars"]
    assert sb["feed"] == "sip" and sb["bars"] == 390 and sb["expected"] == 390, "the bar stamped at the close is not counted as a minute of the session"
    assert sb["first_session_with_bars"] == "2016-01-04" and sb["history_probe"] == {"2015-12-31": 0, "2016-01-04": 390}
    assert all(r["underlying"] == {"feed": "sip", "bars": 390, "expected": 390, "missing_minutes": 0} for r in rep["matrix"])
    text = " ".join(rep["summary"])
    assert "Option history starts on 2024-01-18: every session from 2024-01-02 was asked and none before it returned a bar." in text
    assert "1 session(s) up to 2024-02-16 returned nothing at all: 2024-02-02" in text
    assert "390 of 390 minutes for 2019-06-03; the history starts on 2016-01-04 (the session before it returns nothing)" in text
