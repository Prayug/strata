"""V7: the collector on a provider that lists its contracts and still serves them after they expire (Alpaca).

What changes against the free source: the universe is the provider's own listing (the same selection as a bulk download, so
a collected session is indistinguishable from a downloaded one), a session is collected whole or not at all, and a session
that was missed while the computer was off is collected by the next cycle instead of being written off.
Every request goes through a mocked transport: what the live service delivers is established separately (docs/V7_REPORT.md)."""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import httpx
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from tests.test_v6_alpaca import bar, use

DAYS = cal.trading_days(date(2024, 2, 20), date(2024, 3, 15))
FRI = date(2024, 3, 8)


class FakeYahoo:
    """Daily and recent intraday bars are not the subject here: the daily history is already in the store."""
    name, rate = "yahoo", 2.0

    def __init__(self):
        self.calls = []

    def ingest(self, ds, syms, **kw):
        self.calls.append((ds, tuple(syms)))
        return {"symbols": {s: {"rows": 1} for s in syms}}


def provider(asked: list, fail_listing: set | None = None):
    """Alpaca as the adapter sees it: a contract listing (whole strikes, an expiration every trading day, calls and puts),
    1-minute option bars for the contracts within three strikes of 500, and 1-minute stock bars."""
    fail_listing = fail_listing if fail_listing is not None else set()

    def handler(req: httpx.Request) -> httpx.Response:
        path, q = req.url.path, req.url.params
        if path.endswith("/options/contracts"):
            a, b = date.fromisoformat(q["expiration_date_gte"]), date.fromisoformat(q["expiration_date_lte"])
            if a in fail_listing:
                return httpx.Response(500, json={"message": "the listing is unavailable"})
            out = []
            for e in cal.trading_days(a, b):
                for k in range(math.ceil(float(q["strike_price_gte"])), math.floor(float(q["strike_price_lte"])) + 1):
                    for typ in ("call", "put"):
                        out.append({"symbol": S.occ_symbol("SPY", e, typ[0].upper(), float(k)), "root_symbol": "SPY", "expiration_date": e.isoformat(), "strike_price": str(k), "type": typ,
                                    "status": q["status"], "id": f"{e}-{k}-{typ}", "size": "100", "multiplier": "100", "style": "american", "deliverables": [{"type": "equity", "symbol": "SPY", "amount": "100"}]})
            return httpx.Response(200, json={"option_contracts": out, "next_page_token": None})
        if "/options/bars" in path:
            d = datetime.fromisoformat(q["start"].replace("Z", "+00:00")).astimezone(cal.ET).date()
            occs = q["symbols"].split(",")
            asked.append((d, len(occs)))
            return httpx.Response(200, json={"bars": {o: [bar(d, "09:30", 2.0), bar(d, "12:00", 2.2), bar(d, "15:59", 1.8)] for o in occs if abs(S.parse_occ(o)[3] - 500) <= 3}, "next_page_token": None})
        if "/stocks/bars" in path:
            a = datetime.fromisoformat(q["start"].replace("Z", "+00:00")).astimezone(cal.ET).date()
            b = datetime.fromisoformat(q["end"].replace("Z", "+00:00")).astimezone(cal.ET).date()
            return httpx.Response(200, json={"bars": {"SPY": [bar(d, t, 500.0) for d in cal.trading_days(a, b) for t in ("09:30", "12:00", "15:59")]}, "next_page_token": None})
        return httpx.Response(404, json={"message": f"unexpected request {path}"})
    return handler


@pytest.fixture
def env(tmp_path, monkeypatch):
    from strata.api import db
    from strata.collector import service as CS
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    db.init_db()
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    monkeypatch.setattr(S.Alpaca, "rate", 1000.0)
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in DAYS]), provider="yahoo")
    yh, real = FakeYahoo(), S.get_source
    monkeypatch.setattr(S, "get_source", lambda name, store=None: yh if name == "yahoo" else real(name, store))
    CS.save_config({"symbols": ["SPY"], "options": {"provider": "alpaca", "dte": [0, 1, 7], "strike_pct": 1.0, "rights": ["C", "P"]}, "lookback_sessions": 2, "max_requests_per_cycle": 5000,
                    "rate_per_second": {"alpaca": 10.0}})
    yield CS, st
    override_settings(data_dir=old)


def seed(st: MarketStore, first: date, last: date) -> dict:
    """An archive as `strata download alpaca option_trades SPY --band 1 --max-dte 7` leaves it."""
    src = S.Alpaca(st)
    return src.acquire("SPY", src.bulk_wanted("SPY", first, last, 1.0, 7))


def sessions_stored(st: MarketStore) -> dict[date, int]:
    idx = st.option_bar_index("SPY", provider="alpaca")
    return {r["date"]: r["n"] for r in idx.group_by("date").agg(pl.len().alias("n")).sort("date").to_dicts()}


# ----------------------------------------------------------------------------- continuing an archive
def test_the_collector_continues_a_downloaded_archive_with_the_same_contracts(env, monkeypatch):
    CS, st = env
    asked: list = []
    use(monkeypatch, provider(asked))
    seed(st, date(2024, 3, 1), date(2024, 3, 4))
    before = sessions_stored(st)
    assert list(before) == [date(2024, 3, 1), date(2024, 3, 4)]
    asked.clear()
    r = CS.run_once(store=st, today=FRI, trigger="test")
    assert r["status"] == "done" and r["errors"] == [] and r["unrecoverable"] == []
    after = sessions_stored(st)
    # every session since the last stored one, although the configured look-back is two sessions
    assert list(after) == [date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6), date(2024, 3, 7), FRI]
    assert {d for d, _ in asked} == {date(2024, 3, 5), date(2024, 3, 6), date(2024, 3, 7), FRI}, "the last stored session was checked against the index and nothing in it was requested again"
    o = r["symbols"]["SPY"]["options"]
    assert o["pairs_cached"] == before[date(2024, 3, 4)] and o["trimmed"] is False and "deferred" not in o
    assert "within 1% of the previous close" in o["universe"] and "7 days" in o["universe"]
    assert any("earlier sessions since the last stored one" in n and "still serves expired contracts" in n for n in r["notes"])
    assert {x["date"] for x in r["recovered"]} == {"2024-03-04", "2024-03-05", "2024-03-06"}, "sessions older than the look-back window were recovered, not written off"
    # a collected session holds exactly what a download of that session asks for
    src = S.Alpaca(st)
    want = src.bulk_wanted("SPY", date(2024, 3, 6), date(2024, 3, 6), 1.0, 7)[date(2024, 3, 6)]
    idx = st.option_bar_index("SPY", provider="alpaca").filter(pl.col("date") == date(2024, 3, 6))
    assert sorted(idx["occ"].to_list()) == sorted(c.occ for c in want) and after[date(2024, 3, 6)] == len(want)
    assert idx.filter(pl.col("status") == "ok").height == 7 * 2 * len({c.expiration for c in want}), "contracts within three strikes of 500 printed; the rest are recorded as not having traded"
    # the consolidated tape's minute bars are continued too
    u = r["symbols"]["SPY"]["underlying"]["alpaca_1m"]
    assert u["feed"] == "sip" and u["sessions"] > 0 and u["through"] == FRI.isoformat()
    key = st.datasets("SPY", "underlying", res="1m", provider="alpaca")[0]
    assert max(st.complete_sessions(key)) == FRI
    # a second cycle finds everything stored and asks for no bars
    asked.clear()
    r2 = CS.run_once(store=st, today=FRI, trigger="test")
    assert r2["status"] == "done" and asked == [] and r2["bars"] == 0 and r2["symbols"]["SPY"]["underlying"]["alpaca_1m"]["sessions"] == 0
    # no order endpoint was reachable: only listing and market-data paths were requested
    assert sessions_stored(st) == after


def test_a_tight_budget_defers_whole_sessions_oldest_first_instead_of_thinning_them(env, monkeypatch):
    CS, st = env
    asked: list = []
    use(monkeypatch, provider(asked))
    seed(st, date(2024, 3, 1), date(2024, 3, 4))
    monkeypatch.setattr(S.Alpaca, "batch_size", 2)  # many small requests, so one session nearly fills the smallest budget
    CS.save_config({"max_requests_per_cycle": 50})
    full = len(S.Alpaca(st).bulk_wanted("SPY", date(2024, 3, 5), date(2024, 3, 5), 1.0, 7)[date(2024, 3, 5)])
    r1 = CS.run_once(store=st, today=FRI, trigger="test")
    o = r1["symbols"]["SPY"]["options"]
    assert o["trimmed"] is False and o["deferred"] == ["2024-03-06", "2024-03-07", "2024-03-08"]
    assert any("left for the next cycle" in n and "nothing is lost" in n for n in r1["notes"])
    got = sessions_stored(st)
    assert max(got) == date(2024, 3, 5) and got[date(2024, 3, 5)] == full, "the oldest missing session was collected whole"
    assert r1["unrecoverable"] == [], "a deferred session is not a lost one"
    r2 = CS.run_once(store=st, today=FRI, trigger="test")
    assert max(sessions_stored(st)) == date(2024, 3, 6) and r2["symbols"]["SPY"]["options"]["deferred"] == ["2024-03-07", "2024-03-08"]
    CS.save_config({"max_requests_per_cycle": 5000})
    CS.run_once(store=st, today=FRI, trigger="test")
    done = sessions_stored(st)
    assert max(done) == FRI and all(done[d] > 0 for d in cal.trading_days(date(2024, 3, 4), FRI))


def test_a_failed_listing_is_retried_and_never_recorded_as_lost(env, monkeypatch):
    from strata.api import db
    CS, st = env
    asked: list = []
    down = {date(2024, 3, 6)}
    use(monkeypatch, provider(asked, down))
    seed(st, date(2024, 3, 1), date(2024, 3, 4))
    r = CS.run_once(store=st, today=FRI, trigger="test")
    assert r["status"] == "partial" and r["unrecoverable"] == []
    assert any("2024-03-06" in e and "the contract listing failed" in e and "queued for retry" in e for e in r["errors"])
    assert date(2024, 3, 6) not in sessions_stored(st) and FRI in sessions_stored(st), "the other sessions were still collected"
    q = CS.queue("pending")
    assert [(x["symbol"], x["session"], x["provider"]) for x in q] == [("SPY", "2024-03-06", "alpaca")]
    assert CS.due_sessions("SPY") == [], "the retry waits for its delay"
    down.clear()
    with db.connect() as con:
        con.execute("UPDATE collector_queue SET not_before=?", ("2020-01-01T00:00:00+00:00",))
    r2 = CS.run_once(store=st, today=FRI, trigger="test")
    assert r2["status"] == "done" and date(2024, 3, 6) in sessions_stored(st) and CS.queue("pending") == []
    assert {"symbol": "SPY", "date": "2024-03-06"} in r2["recovered"]


def test_the_strike_range_is_never_chosen_without_the_previous_close(env, monkeypatch):
    CS, st = env
    asked: list = []
    use(monkeypatch, provider(asked))
    # the daily history stops two sessions before the one being collected
    st2 = MarketStore(st.root.parent / "market2")
    st2.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    short = [d for d in DAYS if d <= date(2024, 3, 6)]
    st2.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in short]), provider="yahoo")
    wanted, info = CS.listed_contracts(st2, S.Alpaca(st2), "SPY", [date(2024, 3, 7), FRI], CS.load_config())
    assert list(wanted) == [date(2024, 3, 7)], "the session after the last stored close can be chosen: its previous close is known"
    assert "the close of 2024-03-07 is not stored" in info["failed"]["2024-03-08"] and "without guessing" in info["failed"]["2024-03-08"]
    r = CS.run_once(store=st2, today=FRI, trigger="test")
    assert r["status"] == "partial" and FRI not in sessions_stored(st2) and [x["session"] for x in CS.queue("pending")] == ["2024-03-08"]


# ----------------------------------------------------------------------------- what the interface is told
def test_the_stored_archive_says_which_settings_continue_it(env, monkeypatch):
    CS, st = env
    use(monkeypatch, provider([]))
    assert CS.continuing_config(st) is None, "nothing is stored yet"
    seed(st, date(2024, 3, 1), date(2024, 3, 6))
    u = CS.stored_universe(st)
    assert u["band_pct"] == 1.0 and u["max_dte"] == 7 and u["symbols"]["SPY"]["sessions"] == 4 and u["symbols"]["SPY"]["rights"] == ["C", "P"]
    c = CS.continuing_config(st)
    assert c["symbols"] == ["SPY"] and c["options"] == {"enabled": True, "provider": "alpaca", "strike_pct": 1.0, "dte": [0, 1, 3, 7], "rights": ["C", "P"]}
    state = CS.state(st)
    assert state["listing"] is True and state["continue"]["in_effect"] is True and "within 1% of the previous close" in state["universe"]
    assert "costs nothing but time" in state["offline"]
    alp = next(x for x in state["sources"] if x["name"] == "alpaca")
    assert alp["available"] and alp["keeps_expired"] and "collected by the next cycle" in alp["after_a_miss"]
    assert next(x for x in state["sources"] if x["name"] == "yahoo_options")["keeps_expired"] is False
    # a different band is not the archive's: the interface is told the settings no longer continue it
    CS.save_config({"options": {"strike_pct": 2.0}})
    assert CS.state(st)["continue"]["in_effect"] is False


def test_a_session_this_provider_has_not_stored_yet_is_waiting_not_missed(env, monkeypatch):
    CS, st = env
    use(monkeypatch, provider([]))
    seed(st, date(2024, 3, 1), date(2024, 3, 4))
    cov = CS.coverage(st, CS.load_config(), sessions=8, today=date(2024, 3, 12))
    by = {r["date"]: r["state"] for r in cov["symbols"]["SPY"]}
    assert by["2024-03-01"] == "collected" and by["2024-03-05"] == "waiting" and by["2024-03-12"] == "pending"
    assert cov["missed"] == [] and cov["recoverable"] is True and {"symbol": "SPY", "date": "2024-03-05"} in cov["waiting"]
    # the free source cannot go back: there the same gap is a missed session
    CS.save_config({"options": {"provider": "yahoo_options"}})
    cov = CS.coverage(st, CS.load_config(), sessions=8, today=date(2024, 3, 12))
    assert cov["recoverable"] is False and cov["waiting"] == []


def test_only_market_data_and_the_contract_listing_are_ever_requested(env, monkeypatch):
    CS, st = env
    calls = use(monkeypatch, provider([]))
    seed(st, date(2024, 3, 6), date(2024, 3, 6))
    CS.run_once(store=st, today=FRI, trigger="test")
    paths = {c.url.path for c in calls}
    assert paths == {"/v2/options/contracts", "/v1beta1/options/bars", "/v2/stocks/bars"}
    assert all(c.method == "GET" for c in calls) and not any("order" in p or "position" in p or "account" in p for p in paths)
    assert datetime.now(timezone.utc).year >= 2026  # the mocked sessions are long past: the provider's 15-minute delay never applies here


def test_a_cycle_whose_write_up_fails_does_not_block_the_next_one(env, monkeypatch):
    CS, st = env
    use(monkeypatch, provider([]))
    seed(st, date(2024, 3, 6), date(2024, 3, 6))
    real, n = CS._notify, {"calls": 0}

    def flaky(rep, status):
        n["calls"] += 1
        if n["calls"] == 1:
            raise RuntimeError("the note could not be written")
        return real(rep, status)
    monkeypatch.setattr(CS, "_notify", flaky)
    with pytest.raises(RuntimeError, match="the note could not be written"):
        CS.run_once(store=st, today=FRI, trigger="test")
    assert CS._other_running() is None, "the cycle was closed in the history before the failure"
    r = CS.run_once(store=st, today=FRI, trigger="test")
    assert r["status"] == "done", "the lock was released: the next cycle runs"
