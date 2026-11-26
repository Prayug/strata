"""V3 data layer: provider-partitioned storage, migration, and ingestion regressions.

The ingestion tests replay the exact failure modes the V2 Alpaca ingester had: a truncated
(paginated) response, a partially downloaded session, a missing contract and a strike
universe chosen with a price from the future.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import DatasetKey, MarketStore, SymbolMeta

D1, D2 = date(2024, 3, 4), date(2024, 3, 5)


def _ts(d: date, hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bar(d: date, hhmm: str, px: float, v: int = 10) -> dict:
    return {"t": _ts(d, hhmm), "o": px, "h": px + 0.05, "l": px - 0.05, "c": px, "v": v, "n": 3, "vw": px}


@pytest.fixture
def store(tmp_path, monkeypatch):
    from strata.config import override_settings
    override_settings(data_dir=tmp_path)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    st = MarketStore(tmp_path / "market")
    days = cal.trading_days(date(2024, 2, 20), D2)
    # the close jumps on D1: a strike universe built from D1's own close would be centred ~550, not ~500
    closes = {d: 500.0 for d in days}
    closes[D1], closes[D2] = 550.0, 551.0
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": max(501.0, closes[d]), "low": 499.0, "close": closes[d], "volume": 1.0} for d in days]), provider="yahoo")
    return st


def use(monkeypatch, handler):
    """Route every source request through `handler(request) -> httpx.Response`."""
    calls: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return handler(req)
    monkeypatch.setattr(S, "_client", lambda: httpx.Client(transport=httpx.MockTransport(wrapped)))
    return calls


A, B, C = "SPY240304C00500000", "SPY240304C00501000", "SPY240304C00502000"


def contracts(*occ) -> list[S.Contract]:
    return [S.contract_from_occ(o, source="test") for o in occ]


# ----------------------------------------------------------------------------- pagination
def test_option_bars_follow_next_page_token(store, monkeypatch):
    """Alpaca sorts by symbol, so page 1 can hold a single contract. Every page must be read."""
    def handler(req):
        if req.url.params.get("page_token") == "p2":
            return httpx.Response(200, json={"bars": {B: [bar(D1, "09:31", 2.0), bar(D1, "09:40", 2.2)]}, "next_page_token": None})
        return httpx.Response(200, json={"bars": {A: [bar(D1, "09:30", 3.0), bar(D1, "09:31", 3.1), bar(D1, "09:35", 3.3)]}, "next_page_token": "p2"})
    calls = use(monkeypatch, handler)
    rep = S.Alpaca(store).acquire("SPY", {D1: contracts(A, B)})
    assert len(calls) == 2 and calls[1].url.params["page_token"] == "p2"
    assert rep["pages"] == 2 and rep["pairs_ok"] == 2 and rep["pairs_failed"] == 0
    bars = store.read_option_bars("SPY", D1, provider="alpaca")
    assert bars.group_by("occ").len().sort("occ").rows() == [(A, 3), (B, 2)]  # the second page was not lost
    idx = store.option_bar_index("SPY", provider="alpaca")
    assert sorted(zip(idx["occ"].to_list(), idx["bars"].to_list(), idx["pages"].to_list())) == [(A, 3, 2), (B, 2, 2)]
    assert calls[0].url.params["limit"] == "10000" and calls[0].url.params["symbols"] == f"{A},{B}"


def test_truncated_response_is_never_recorded_as_downloaded(store, monkeypatch):
    """The second page fails: nothing from that request may be stored or indexed, and a
    later run must fetch the whole thing again (no half-session marked as done)."""
    state = {"fail": True}

    def handler(req):
        if req.url.params.get("page_token") == "p2":
            if state["fail"]:
                return httpx.Response(503, text="upstream unavailable")
            return httpx.Response(200, json={"bars": {B: [bar(D1, "10:00", 2.0)]}, "next_page_token": None})
        return httpx.Response(200, json={"bars": {A: [bar(D1, "09:30", 3.0)]}, "next_page_token": "p2"})
    use(monkeypatch, handler)
    src = S.Alpaca(store)
    rep = src.acquire("SPY", {D1: contracts(A, B)})
    assert rep["pairs_failed"] == 2 and rep["pairs_ok"] == 0 and rep["bars"] == 0 and rep["errors"]
    assert store.option_bar_index("SPY", provider="alpaca").is_empty()  # contract A's page arrived, but the request was not complete
    assert store.read_option_bars("SPY", D1, provider="alpaca") is None
    state["fail"] = False
    rep = src.acquire("SPY", {D1: contracts(A, B)})
    assert rep["pairs_cached"] == 0 and rep["pairs_ok"] == 2 and rep["bars"] == 2


def test_partially_downloaded_session_is_completed_not_skipped(store, monkeypatch):
    """V2 skipped a session as soon as a file existed for it. Coverage is per (session,
    contract): asking for a contract that was not fetched downloads it."""
    asked: list[str] = []

    def handler(req):
        syms = req.url.params["symbols"].split(",")
        asked.append(",".join(syms))
        return httpx.Response(200, json={"bars": {s: [bar(D1, "09:45", 1.5)] for s in syms}, "next_page_token": None})
    use(monkeypatch, handler)
    src = S.Alpaca(store)
    src.acquire("SPY", {D1: contracts(A)})
    assert store.read_option_bars("SPY", D1, provider="alpaca")["occ"].unique().to_list() == [A]
    rep = src.acquire("SPY", {D1: contracts(A, B)})
    assert rep["pairs_cached"] == 1 and rep["pairs_ok"] == 1 and asked == [A, B]  # only the missing contract was requested
    assert sorted(store.read_option_bars("SPY", D1, provider="alpaca")["occ"].unique().to_list()) == [A, B]
    rep = src.acquire("SPY", {D1: contracts(A, B)})
    assert rep["requests"] == 0 and rep["pairs_cached"] == 2  # resumable: nothing is downloaded twice


def test_contract_without_trades_is_recorded_empty_and_nothing_is_invented(store, monkeypatch):
    use(monkeypatch, lambda req: httpx.Response(200, json={"bars": {A: [bar(D1, "09:45", 1.5)]}, "next_page_token": None}))
    rep = S.Alpaca(store).acquire("SPY", {D1: contracts(A, C)})
    assert rep["pairs_ok"] == 1 and rep["pairs_empty"] == 1
    idx = {r["occ"]: r for r in store.option_bar_index("SPY", provider="alpaca").iter_rows(named=True)}
    assert idx[C]["status"] == "empty" and idx[C]["bars"] == 0 and idx[A]["status"] == "ok"
    bars = store.read_option_bars("SPY", D1, provider="alpaca")
    assert bars["occ"].unique().to_list() == [A]  # no row exists for the contract that did not trade
    assert "bid" not in bars.columns and "ask" not in bars.columns  # a trade bar is never given a quote


def test_invalid_and_out_of_session_bars_are_rejected(store, monkeypatch):
    rows = [bar(D1, "09:29", 1.0), bar(D1, "09:30", 1.0), bar(D1, "09:30", 1.1), bar(D1, "16:00", 1.2), {"t": _ts(D1, "10:00"), "o": 1.0, "h": 0.5, "l": 0.9, "c": 1.0, "v": 1, "n": 1, "vw": 1.0}]
    use(monkeypatch, lambda req: httpx.Response(200, json={"bars": {A: rows}, "next_page_token": None}))
    rep = S.Alpaca(store).acquire("SPY", {D1: contracts(A)})
    assert rep["rejected"] == {"outside_session": 2, "invalid_price": 1, "duplicates": 1}
    bars = store.read_option_bars("SPY", D1, provider="alpaca")
    assert bars.height == 1 and bars["close"][0] == pytest.approx(1.1)  # the later duplicate wins; nothing was repaired


# ----------------------------------------------------------------------------- point-in-time discovery
def _contract_json(occ: str, status: str) -> dict:
    root, exp, right, strike = S.parse_occ(occ)
    return {"id": f"id-{occ}", "symbol": occ, "name": occ, "status": status, "tradable": status == "active", "expiration_date": exp.isoformat(), "root_symbol": root,
            "underlying_symbol": "SPY", "type": "call" if right == "C" else "put", "style": "american", "strike_price": f"{strike:g}", "multiplier": "100", "size": "100"}


def test_strike_universe_comes_from_the_previous_close_not_the_session_close(store, monkeypatch):
    """On D1 the stored close is 550 but the previous session closed at 500. Entering at the
    open, 550 was unknowable: the contract request must be centred on 500."""
    calls = use(monkeypatch, lambda req: httpx.Response(200, json={"option_contracts": [], "next_page_token": None}))
    with S._client() as client:
        S.Alpaca(store).bulk_contracts(client, "SPY", D1, band_pct=2.0, max_dte=3)
    lo, hi = float(calls[0].url.params["strike_price_gte"]), float(calls[0].url.params["strike_price_lte"])
    assert (lo, hi) == (490.0, 510.0)
    assert not (lo <= 550.0 <= hi)  # the session's own close is outside the requested range
    assert S.previous_close(store, "SPY", D1) == 500.0 and S.previous_close(store, "SPY", D2) == 550.0


def test_discovery_is_unchanged_when_the_future_is_rewritten(store, monkeypatch, tmp_path):
    """Perturbation test: change every price from session D1 onwards; the universe requested
    for D1 must be identical."""
    seen = []
    use(monkeypatch, lambda req: (seen.append(dict(req.url.params)), httpx.Response(200, json={"option_contracts": [], "next_page_token": None}))[1])
    src = S.Alpaca(store)
    monkeypatch.setattr(src, "cache", tmp_path / "c1")
    with S._client() as client:
        src.bulk_contracts(client, "SPY", D1)
    first = [dict(x) for x in seen]
    seen.clear()
    d = store.read_daily("SPY")
    store.write_daily("SPY", d.with_columns(pl.when(pl.col("date") >= D1).then(pl.col("close") * 3).otherwise(pl.col("close")).alias("close")), provider="yahoo")
    monkeypatch.setattr(src, "cache", tmp_path / "c2")
    with S._client() as client:
        src.bulk_contracts(client, "SPY", D1)
    assert seen == first


def test_discovery_asks_for_inactive_contracts_too(store, monkeypatch):
    """A contract that expired long ago is `inactive` today, yet it was tradable on its
    session. Filtering on today's status (the endpoint's default is active only) loses it.
    Only the statuses that can match the expiration window are asked for: an expired window
    is `inactive` only, a window around today needs both."""
    def handler(req):
        st = req.url.params["status"]
        return httpx.Response(200, json={"option_contracts": [_contract_json(A, "inactive")] if st == "inactive" else [], "next_page_token": None})
    calls = use(monkeypatch, handler)
    with S._client() as client:
        got = S.Alpaca(store).list_contracts(client, "SPY", D1, D1, D1 + timedelta(days=3), 490, 510)
    assert {c.url.params["status"] for c in calls} == {"inactive"}, "expirations in the past: `active` cannot match and is not asked for"
    assert [c.occ for c in got] == [A] and got[0].status == "inactive" and got[0].provider_id == f"id-{A}" and got[0].multiplier == 100.0
    assert all(c.url.params["expiration_date_gte"] == D1.isoformat() for c in calls)  # never an expiration before the session
    # a window that reaches today: a contract may be either, so both are asked
    today = datetime.now(cal.ET).date()
    calls.clear()
    with S._client() as client:
        S.Alpaca(store).list_contracts(client, "SPY", today - timedelta(days=2), today - timedelta(days=2), today + timedelta(days=3), 490, 510)
    assert {c.url.params["status"] for c in calls} == {"active", "inactive"}
    # entirely in the future: only `active`
    calls.clear()
    with S._client() as client:
        S.Alpaca(store).list_contracts(client, "SPY", today, today + timedelta(days=10), today + timedelta(days=20), 490, 510)
    assert {c.url.params["status"] for c in calls} == {"active"}


def test_contract_listing_is_paginated(store, monkeypatch):
    def handler(req):
        if req.url.params["status"] != "inactive":
            return httpx.Response(200, json={"option_contracts": [], "next_page_token": None})
        if req.url.params.get("page_token") == "t2":
            return httpx.Response(200, json={"option_contracts": [_contract_json(B, "inactive")], "next_page_token": None})
        return httpx.Response(200, json={"option_contracts": [_contract_json(A, "inactive")], "next_page_token": "t2"})
    use(monkeypatch, handler)
    with S._client() as client:
        got = S.Alpaca(store).list_contracts(client, "SPY", D1, D1, D1, 490, 510)
    assert [c.occ for c in got] == [A, B]


def test_identifiers_multiplier_and_expiration_are_preserved(store, monkeypatch):
    use(monkeypatch, lambda req: httpx.Response(200, json={"bars": {A: [bar(D1, "09:45", 1.5)]}, "next_page_token": None}))
    c = S.Alpaca.parse_contract({**_contract_json(A, "inactive"), "multiplier": "100"})
    S.Alpaca(store).acquire("SPY", {D1: [c]})
    key = store.option_dataset("SPY", "trade", "alpaca")
    row = store.read_contracts(key).row(0, named=True)
    assert row["occ"] == A and row["provider_id"] == f"id-{A}" and row["multiplier"] == 100.0 and row["root"] == "SPY" and row["adjusted"] is False
    assert row["expiration_ts"] == cal.session_bounds(D1)[1].astimezone(timezone.utc) and row["status_at_fetch"] == "inactive"
    assert store.read_option_bars("SPY", D1, provider="alpaca")["occ"][0] == A  # the original contract identifier stays on every bar
    desc = store.descriptor(key)
    assert desc["provider"] == "alpaca" and desc["obs"] == "trade" and desc["format"] == "bars" and any("no bid/ask" in x for x in desc["limitations"])


def test_adjusted_contracts_are_recognised():
    c = S.contract_from_occ("SPY1240315C00500000")
    assert c.root == "SPY1" and c.adjusted and c.strike == 500.0 and c.expiration == date(2024, 3, 15)
    assert S.occ_symbol("SPY", date(2024, 3, 15), "P", 512.5) == "SPY240315P00512500"
    with pytest.raises(S.SourceError):
        S.parse_occ("NOT-A-CONTRACT")


# ----------------------------------------------------------------------------- stock bars
def test_stock_bars_are_paginated_and_sessions_marked_complete_only_after_the_last_page(store, monkeypatch):
    state = {"fail": True}
    mins = lambda d, a, b: [bar(d, f"{9 + (570 + i - 540) // 60:02d}:{(570 + i) % 60:02d}", 500.0 + i * 0.01, 100) for i in range(a, b)]

    def handler(req):
        if req.url.params.get("page_token") == "s2":
            if state["fail"]:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"bars": {"SPY": mins(D1, 200, 390)}, "next_page_token": None})
        return httpx.Response(200, json={"bars": {"SPY": mins(D1, 0, 200)}, "next_page_token": "s2"})
    use(monkeypatch, handler)
    src = S.Alpaca(store)
    with pytest.raises(S.SourceError):
        src.stock_bars("SPY", D1, D1)
    key = DatasetKey("alpaca", "SPY", "underlying", "trade", "1m")
    assert store.complete_sessions(key) == set()  # 200 bars arrived, but the session is not complete
    assert store.read_bars("SPY", "1m", provider="alpaca").is_empty()
    state["fail"] = False
    rep = src.stock_bars("SPY", D1, D1)
    assert rep["bars"] == 390 and rep["pages"] == 2 and store.complete_sessions(key) == {D1}
    assert src.stock_bars("SPY", D1, D1)["sessions"] == 0  # cached


# ----------------------------------------------------------------------------- Yahoo option bars
def _yahoo_chart(ts_px_vol: list[tuple]) -> dict:
    return {"chart": {"result": [{"meta": {"instrumentType": "OPTION"}, "timestamp": [t for t, _, _ in ts_px_vol],
                                   "indicators": {"quote": [{"open": [p for _, p, _ in ts_px_vol], "high": [p for _, p, _ in ts_px_vol], "low": [p for _, p, _ in ts_px_vol],
                                                             "close": [p for _, p, _ in ts_px_vol], "volume": [v for _, _, v in ts_px_vol]}]}}], "error": None}}


def test_yahoo_option_bars_keep_only_confirmed_prints(store, monkeypatch):
    today = datetime.now(cal.ET).date()
    d = cal.prev_trading_day(cal.prev_trading_day(today))
    exp = cal.next_trading_day(today, 3)
    occ = S.occ_symbol("SPY", exp, "C", 500)
    o = int(cal.session_bounds(d)[0].timestamp())
    pts = [(o, 2.0, 0), (o + 60, 2.1, 5), (o + 120, None, None), (o + 180, 2.3, 7), (o + 205, 2.4, 9), (o - 3600, 9.9, 4)]  # zero volume, no trade, unaligned live bar, pre-market
    calls = use(monkeypatch, lambda req: httpx.Response(200, json=_yahoo_chart(pts)))
    src = S.YahooOptions(store)
    rep = src.acquire("SPY", {d: [S.contract_from_occ(occ)]})
    assert rep["pairs_ok"] == 1 and rep["bars"] == 2 and rep["rejected"] == {"outside_session": 1, "unconfirmed_zero_volume": 1}
    bars = store.read_option_bars("SPY", d, provider="yahoo_options")
    assert bars["close"].to_list() == pytest.approx([2.1, 2.3]) and bars["occ"].unique().to_list() == [occ]
    assert calls[0].url.path.endswith(occ)  # one request per contract


def test_yahoo_purged_expired_contract_is_unavailable_not_empty(store, monkeypatch):
    """An expired contract the provider no longer serves proves nothing about trading: it
    must not be indexed as 'no trades'. An unknown symbol that has NOT expired is not listed."""
    today = datetime.now(cal.ET).date()
    d = cal.prev_trading_day(cal.prev_trading_day(today))
    use(monkeypatch, lambda req: httpx.Response(404, json={"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"}}}))
    src = S.YahooOptions(store)
    expired = S.contract_from_occ(S.occ_symbol("SPY", d, "C", 500))
    live = S.contract_from_occ(S.occ_symbol("SPY", cal.next_trading_day(today, 5), "C", 500))
    rep = src.acquire("SPY", {d: [expired, live]})
    assert rep["pairs_unavailable"] == 1 and rep["pairs_empty"] == 1 and rep["pairs_failed"] == 0
    idx = store.option_bar_index("SPY", provider="yahoo_options")
    assert idx["occ"].to_list() == [live.occ] and idx["status"].to_list() == ["empty"]


def test_sessions_outside_a_providers_window_are_reported_not_fetched(store, monkeypatch):
    calls = use(monkeypatch, lambda req: httpx.Response(200, json=_yahoo_chart([])))
    rep = S.YahooOptions(store).acquire("SPY", {D1: contracts(A)})  # March 2024 is far outside the 30-day window
    assert not calls and rep["out_of_window"] == [D1.isoformat()] and rep["pairs_unavailable"] == 1
    rep = S.Alpaca(store).acquire("SPY", {date(2023, 6, 1): [S.contract_from_occ("SPY230601C00420000")]})  # before Alpaca's option history starts
    assert rep["out_of_window"] == ["2023-06-01"]


def test_alpaca_without_a_key_refuses_and_never_substitutes(store, monkeypatch):
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("", ""))
    calls = use(monkeypatch, lambda req: httpx.Response(200, json={}))
    with pytest.raises(S.SourceError, match="connect a free Alpaca account"):
        S.Alpaca(store).acquire("SPY", {D1: contracts(A)})
    assert not calls and store.option_dataset("SPY", "trade") is None


# ----------------------------------------------------------------------------- storage
def _quotes(d: date, bid=5.0) -> pl.DataFrame:
    ts = cal.session_bounds(d)[1]
    return pl.DataFrame([{"ts": ts, "expiration": d + timedelta(days=30), "strike": 500.0, "right": r, "bid": bid, "ask": bid + 0.1, "underlying_price": 500.0, "quote_ts": ts} for r in "CP"]).with_columns(
        pl.col("ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")))


def test_two_providers_and_two_observation_types_never_overwrite_each_other(store, monkeypatch):
    store.write_option_day("SPY", D1, _quotes(D1, 5.0), provider="github_eod", res="EOD")
    store.write_option_day("SPY", D1, _quotes(D1, 7.0), provider="dolthub", res="EOD")
    use(monkeypatch, lambda req: httpx.Response(200, json={"bars": {A: [bar(D1, "09:45", 1.5)]}, "next_page_token": None}))
    S.Alpaca(store).acquire("SPY", {D1: contracts(A)})
    ids = {k.id for k in store.datasets("SPY", "options")}
    assert ids == {"github_eod/SPY/options/quote/EOD/v1", "dolthub/SPY/options/quote/EOD/v1", "alpaca/SPY/options/trade/1m/v1"}
    assert store.read_option_day("SPY", D1, provider="github_eod")["bid"][0] == 5.0
    assert store.read_option_day("SPY", D1, provider="dolthub")["bid"][0] == 7.0
    q = store.read_option_day("SPY", D1, "quote")
    assert set(q.columns) >= {"bid", "ask"} and q["bid"].null_count() == 0  # the quote reader never returns trade bars
    # trade bars exist for D1, but a session of selectively stored trade bars is never "complete": coverage is per contract (the index)
    assert store.option_days("SPY", "trade") == [D1] and store.option_days("SPY", "trade", complete_only=True) == []
    assert store.read_option_bars("SPY", D1)["close"][0] == 1.5
    # underlying: the daily record now exists for two providers; each stays intact and the reader ranks them
    store.write_daily("SPY", pl.DataFrame([{"date": D1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}]), provider="import")
    assert store.read_daily("SPY", provider="import").height == 1
    assert store.read_daily("SPY").filter(pl.col("date") == D1)["close"][0] == 550.0  # yahoo outranks an import for the same date
    merged = store.read_daily("SPY", with_provider=True)
    assert merged["provider"].unique().to_list() == ["yahoo"]


def test_provider_facts_live_on_the_dataset_not_the_symbol(store):
    store.write_bars("NVDA", "5m", pl.DataFrame({"ts": [datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}),
                     provider="yahoo", adjustment="split")
    store.write_bars("NVDA", "5m", pl.DataFrame({"ts": [datetime(2024, 3, 5, 14, 30, tzinfo=timezone.utc)], "open": [2.0], "high": [2.0], "low": [2.0], "close": [2.0], "volume": [1.0]}),
                     provider="import")
    assert store.descriptor(DatasetKey("yahoo", "NVDA", "underlying", "trade", "5m"))["adjustment"] == "split"
    assert store.descriptor(DatasetKey("import", "NVDA", "underlying", "trade", "5m"))["adjustment"] is None
    b = store.read_bars("NVDA", "5m", with_provider=True)
    assert b["provider"].to_list() == ["yahoo", "import"]  # one provider per session, labelled
    assert "sources" not in (store.symbol_meta("NVDA").extra or {})


def test_manifests_are_immutable_and_versions_change_with_content(store):
    key = store.write_option_day("SPY", D1, _quotes(D1), provider="github_eod", res="EOD") and store.option_dataset("SPY", "quote")
    m1 = store.commit(key)
    assert store.commit(key)["version"] == m1["version"] and m1["seq"] == 1  # nothing changed: no new manifest
    f = next(iter(m1["files"].values()))
    assert f["rows"] == 2 and len(f["sha256"]) == 64 and f["provenance"]["provider"] == "github_eod"
    v_sym = store.dataset_version("SPY")
    store.write_option_day("SPY", D2, _quotes(D2), provider="github_eod", res="EOD")
    m2 = store.commit(key)
    assert m2["seq"] == 2 and m2["parent"] == m1["version"] and m2["version"] != m1["version"]
    names = sorted(p.name for p in (store.dataset_dir(key) / "manifests").iterdir())
    assert len(names) == 2 and json.loads((store.dataset_dir(key) / "manifests" / names[0]).read_text())["version"] == m1["version"]  # the first manifest is still there, untouched
    assert store.dataset_version("SPY") != v_sym


def test_sessions_are_complete_only_when_the_writer_says_so(store):
    store.write_option_day("SPY", D1, _quotes(D1), provider="thetadata", res="15m", status="partial")
    store.write_option_day("SPY", D2, _quotes(D2), provider="thetadata", res="15m", status="complete")
    assert store.option_days("SPY", provider="thetadata") == [D1, D2]
    assert store.option_days("SPY", provider="thetadata", complete_only=True) == [D2]  # the ingester will fetch D1 again


# ----------------------------------------------------------------------------- migration
def test_layout2_installation_is_migrated_in_place(tmp_path):
    root = tmp_path / "market"
    ts = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
    bars = pl.DataFrame({"ts": [ts], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    daily = pl.DataFrame({"date": [D1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    for p, df in ((root / "daily" / "SPY.parquet", daily), (root / "daily" / "VIX.parquet", daily), (root / "underlying" / "SPY" / "2024.parquet", bars), (root / "bars" / "5m" / "SPY" / "2024.parquet", bars),
                  (root / "options" / "SPY" / "2024" / "2024-03-04.parquet", _quotes(D1)), (root / "options" / "QQQ" / "2024" / "2024-03-04.parquet", _quotes(D1).drop("bid", "ask").with_columns(pl.lit(1.0).alias("last"))),
                  (root / "derived" / "SPY_iv.parquet", pl.DataFrame({"date": [D1], "atm_iv_30d": [0.2], "atm_iv_front": [0.2], "close": [500.0]})),
                  (root / "events" / "events.parquet", pl.DataFrame({"date": [D1], "time": ["08:30"], "event_type": ["CPI"], "description": [""], "source": ["x"]})),
                  (root / "corporate_actions" / "SPY.parquet", pl.DataFrame({"ex_date": [D1], "action": ["dividend"], "ratio": [None], "amount": [1.5]}, schema={"ex_date": pl.Date, "action": pl.String, "ratio": pl.Float64, "amount": pl.Float64}))):
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
    (root / "manifests").mkdir()
    (root / "manifests" / "SPY.json").write_text("{}")
    src = lambda prov, kind, res, qt="trades": {f"{kind}:{res}": {"provider": prov, "kind": kind, "resolution": res, "license": "L", "quote_type": qt, "url": "u"}}
    (root / "symbols.json").write_text(json.dumps({
        "SPY": {"symbol": "SPY", "provider": "github_eod", "option_interval_minutes": None,
                "extra": {"price_adjustment": "split", "sources": {**src("yahoo", "underlying", "1d"), **src("yahoo", "underlying", "1m"), **src("yahoo", "underlying", "5m"), **src("github_eod", "options", "EOD", "quotes")}}},
        "VIX": {"symbol": "VIX", "provider": "", "has_options": False, "extra": {"sources": src("cboe", "underlying", "1d", "index")}},
        "QQQ": {"symbol": "QQQ", "provider": "alpaca", "option_interval_minutes": 1, "extra": {"sources": src("alpaca", "options", "1m", "trades")}}}))
    st = MarketStore(root)  # opening the store migrates it
    assert (root / "LAYOUT").read_text() == "3" and not any((root / d).exists() for d in ("daily", "underlying", "bars", "options", "derived", "manifests", "events", "corporate_actions"))
    ids = {k.id for k in st.datasets()}
    assert ids == {"yahoo/SPY/underlying/trade/1d/v1", "yahoo/SPY/underlying/trade/1m/v1", "yahoo/SPY/underlying/trade/5m/v1", "github_eod/SPY/options/quote/EOD/v1",
                   "cboe/VIX/underlying/index/1d/v1", "alpaca/QQQ/options/trade/1m/v1"}
    assert st.read_daily("SPY").height == 1 and st.read_bars("SPY", "1m").height == 1 and st.read_bars("SPY", "5m").height == 1
    assert st.read_option_day("SPY", D1).height == 2 and st.option_days("SPY", complete_only=True) == [D1]
    assert st.read_iv_history("SPY").height == 1 and st.read_events().height == 1 and st.read_corporate_actions("SPY").height == 1
    assert st.price_adjustment("SPY") == "split" and "sources" not in st.symbol_meta("SPY").extra
    # trade prints downloaded by the V2 Alpaca ingester may be truncated: kept, but never treated as complete
    q = st.option_dataset("QQQ", "trade")
    assert st.option_days("QQQ", "trade") == [D1] and st.option_days("QQQ", "trade", complete_only=True) == []
    assert any("did not follow pagination" in x for x in st.descriptor(q)["limitations"])
    assert st.option_dataset("QQQ", "quote") is None  # and they are not quotes
    report = json.loads((root / "migrations" / "layout2-to-3.json").read_text())
    assert report["moved_files"] == 7 and len(report["datasets"]) == 6
    from strata.data.migrate import migrate_layout
    assert migrate_layout(root)["note"] == "already current"  # idempotent
    assert MarketStore(root).read_daily("SPY").height == 1
