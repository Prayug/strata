"""What running the Alpaca adapter against the live service changed (2026-11-25), pinned without the network.

Measured there and encoded here: a provider can hold a session only in part (it must be recorded and not priced); the
consolidated tape's own volume is real even when one minute holds a tenth of the day; a daily bar's high or low can sit
outside every regular-session bar; stored bars can be rebuilt from the provider's individual trades.
Every request goes through a mocked transport: nothing here establishes what an account can retrieve."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data import quality as Q
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from tests.helpers import et_epoch, write_minutes

D0, D1, D2, D3 = date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)


def _ts(d: date, hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bar(d, hhmm, px, v=10):
    return {"t": _ts(d, hhmm), "o": px, "h": px + 0.05, "l": px - 0.05, "c": px, "v": v, "n": 3, "vw": px}


@pytest.fixture
def store(tmp_path, monkeypatch):
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    monkeypatch.setattr(S.Alpaca, "rate", 0.0)
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    days = cal.trading_days(date(2024, 2, 20), D3)
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in days]), provider="yahoo")
    yield st
    override_settings(data_dir=old)


def use(monkeypatch, handler):
    calls: list[httpx.Request] = []

    def wrapped(req):
        calls.append(req)
        return handler(req)
    monkeypatch.setattr(S, "_client", lambda: httpx.Client(transport=httpx.MockTransport(wrapped)))
    return calls


def chain(d: date, n: int = 12) -> list[S.Contract]:
    """n calls expiring on the session itself, strikes around 500."""
    return [S.contract_from_occ(S.occ_symbol("SPY", d, "C", 494.0 + i), source="test") for i in range(n)]


def serve(hours: dict[date, tuple[str, str] | None]):
    """A provider that holds each session only between the given times (None: nothing at all)."""
    def handler(req):
        if "/options/bars" not in req.url.path:
            return httpx.Response(200, json={"option_contracts": [], "next_page_token": None})
        start = datetime.fromisoformat(req.url.params["start"].replace("Z", "+00:00")).astimezone(cal.ET).date()
        span = hours.get(start)
        out = {}
        if span:
            for occ in req.url.params["symbols"].split(","):
                out[occ] = [bar(start, span[0], 2.0)] + ([bar(start, "12:00", 2.2)] if span[0] < "12:00" < span[1] else []) + [bar(start, span[1], 1.8)]
        return httpx.Response(200, json={"bars": out, "next_page_token": None})
    return handler


# ----------------------------------------------------------------------------- a session the provider holds only in part
def test_a_session_held_only_in_part_is_recorded_and_is_not_priced(store, monkeypatch):
    """Alpaca, measured: 2024-01-23 stops at 11:02, 2024-02-02 is absent, five early sessions start after 09:35.
    On such a session "no print at the open" is a hole in the history. It is recorded; coverage and the engine leave it out."""
    from strata.data import planner as P
    from strata.data import readiness as RD
    use(monkeypatch, serve({D1: ("10:14", "15:59"), D2: ("09:30", "15:59"), D3: None, D0: ("09:31", "11:02")}))
    src = S.Alpaca(store)
    rep = src.acquire("SPY", {d: chain(d, 24) for d in (D0, D1, D2, D3)})
    assert rep["pairs_ok"] == 72 and rep["pairs_empty"] == 24 and rep["pairs_failed"] == 0
    bad = rep["partial_sessions"]
    assert set(bad) == {D0.isoformat(), D1.isoformat(), D3.isoformat()}, "the whole session is not marked"
    assert "10:14" in bad[D1.isoformat()] and "11:02" in bad[D0.isoformat()] and "no trade for any of the 24 contracts" in bad[D3.isoformat()]
    key = store.option_dataset("SPY", "trade", "alpaca")
    states = store.session_states(key)
    assert states[D1.isoformat()]["status"] == "partial" and states[D1.isoformat()]["first_bar"] == "10:14" and D2.isoformat() not in states
    # the bars themselves are kept exactly as they came: nothing is deleted or filled
    assert store.read_option_bars("SPY", D1, dataset=key).height == 72
    # coverage: a signal on a partial session cannot be priced, and the reason is the provider's history, not the market
    req = {"symbol": "SPY", "options": {"legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}], "hold_sessions": 0}}
    cov = P.trade_coverage(store, req, [{"date": d, "price": 500.0} for d in (D0, D1, D2, D3)], "alpaca")
    assert cov["executable"] == 1
    why = {i["reason"]: i["signals"] for i in cov["issues"]}
    assert why == {"the provider's option trade history for the session is incomplete (it starts late, stops early or is absent)": 3}
    # readiness counts them as partial sessions
    by = RD.stored_option_sessions(store, "SPY")["sessions"]
    assert [by[d][0]["status"] for d in (D0, D1, D2, D3)] == ["partial", "partial", "complete", "partial"] and by[D1][0]["window"] == ["10:14", "15:59"]


def test_too_few_contracts_to_judge_leaves_the_session_alone(store, monkeypatch):
    """Three far-dated contracts with a first print at 10:14 say nothing about the provider: that can be a quiet contract."""
    use(monkeypatch, serve({D1: ("10:14", "15:00")}))
    rep = S.Alpaca(store).acquire("SPY", {D1: chain(D1, 3)})
    assert rep["partial_sessions"] == {} and store.session_states(store.option_dataset("SPY", "trade", "alpaca")) == {}
    # twelve contracts expiring a month out are not "near-dated" either
    far = [S.contract_from_occ(S.occ_symbol("SPY", D1 + timedelta(days=31), "C", 494.0 + i), source="test") for i in range(12)]
    assert S.Alpaca(store).acquire("SPY", {D1: far})["partial_sessions"] == {}
    # and the verdict follows the data: with twelve same-day contracts starting at 10:14 the session IS partial...
    assert set(S.Alpaca(store).acquire("SPY", {D1: chain(D1, 12)})["partial_sessions"]) == {D1.isoformat()}
    # ...until a later download shows the open after all
    use(monkeypatch, serve({D1: ("09:30", "15:59")}))
    again = S.Alpaca(store).acquire("SPY", {D1: chain(D1, 12)}, force=True)
    assert again["partial_sessions"] == {} and store.session_states(store.option_dataset("SPY", "trade", "alpaca"))[D1.isoformat()]["status"] == "complete"


def test_the_engine_does_not_simulate_a_partial_session(tmp_path):
    """The entry would otherwise slide to the first print the provider happens to hold."""
    from strata.engine.engine import run_backtest
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [cal.prev_trading_day(D1), D1, D2], lambda d, i: 500.0)
    for d in (D1, D2):
        occ = S.occ_symbol("SPY", d, "C", 500.0)
        rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": d, "strike": 500.0, "right": "C", "open": px, "high": px, "low": px, "close": px, "volume": 5}
                for t, px in (("09:59", 2.0), ("15:29", 2.6))]
        st.write_option_bars("SPY", "testfeed", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))),
                             [{"date": d, "occ": occ, "expiration": d, "strike": 500.0, "right": "C", "bars": 2, "status": "ok", "pages": 1}])
    key = st.option_dataset("SPY", "trade", "testfeed")
    cfg = {"start": D1.isoformat(), "end": D2.isoformat(), "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0},
           "strategies": [{"name": "0DTE call", "symbol": "SPY", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
                           "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
                           "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}]}
    both = run_backtest(cfg, store=st)
    assert sorted(datetime.fromtimestamp(t["entry_ts"], tz=cal.ET).date() for t in both.trades) == [D1, D2]
    st.mark_sessions(key, {D1.isoformat(): {"status": "partial", "reason": "the provider's option history for this session is incomplete: its first trade print of any near-dated contract is at 10:14 New York time"}})
    res = run_backtest(cfg, store=st)
    assert [datetime.fromtimestamp(t["entry_ts"], tz=cal.ET).date() for t in res.trades] == [D2], "the partial session produced no trade"
    excluded = res.diagnostics["excluded_sessions"]
    assert any("incomplete" in k and D1.isoformat() in v for k, v in excluded.items()), excluded
    used = res.meta.get("datasets_used") or res.diagnostics.get("datasets_used") or {}
    assert used["SPY"]["partial_sessions"] == 1


def test_assessing_one_session():
    mk = lambda d, times, n=12, dte=0: pl.DataFrame([{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": f"C{i}", "expiration": d + timedelta(days=dte)} for i in range(n) for t in times]).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")), pl.col("expiration").cast(pl.Date))
    assert Q.assess_trade_session(mk(D1, ["09:30", "15:59"]), D1, 12, 12)["status"] == "complete"
    assert Q.assess_trade_session(mk(D1, ["09:35", "15:54"]), D1, 12, 12)["status"] == "complete", "five minutes of slack at the open, six at the close"
    late = Q.assess_trade_session(mk(D1, ["09:36", "15:59"]), D1, 12, 12)
    assert late["status"] == "partial" and late["kind"] == "hours" and "09:36" in late["reason"]
    early = Q.assess_trade_session(mk(D1, ["09:30", "15:53"]), D1, 12, 12)
    assert early["status"] == "partial" and "15:53" in early["reason"] and "16:00 close" in early["reason"]
    # an early close: 13:00 is the close, and a last print at 12:58 is whole
    half = date(2024, 11, 29)
    assert Q.assess_trade_session(mk(half, ["09:30", "12:58"]), half, 12, 12)["status"] == "complete"
    assert Q.assess_trade_session(mk(half, ["09:30", "12:40"]), half, 12, 12)["status"] == "partial"
    assert Q.assess_trade_session(mk(D1, ["10:30", "11:00"], n=9), D1, 9, 9) is None, "nine contracts are not enough to judge"
    assert Q.assess_trade_session(mk(D1, ["10:30", "11:00"], dte=30), D1, 12, 12) is None, "contracts a month out trade rarely"
    assert Q.assess_trade_session(None, D1, 24, 0)["kind"] == "empty" and Q.assess_trade_session(None, D1, 6, 0) is None


# ----------------------------------------------------------------------------- volume on the consolidated tape
def test_a_large_block_on_the_consolidated_tape_keeps_its_volume():
    """SPY, 2018-09-21 10:43: 18.7 million shares in one minute, 17.7% of the day, a real print. The rule that removes a
    running-total artefact from Yahoo's bars flagged 32 such prints in 2,741 sessions of consolidated bars."""
    from strata.engine.market import UnderlyingSeries
    assert Q.artefact_prone("yahoo") and Q.artefact_prone("import") and Q.artefact_prone(None) and not Q.artefact_prone("alpaca") and not Q.artefact_prone("databento")
    d = date(2024, 3, 15)
    vol = np.full(390, 100_000.0)
    vol[73] = 18_700_000.0  # 10:43
    assert Q.cumulative_volume_artefacts(vol, 60)[73], "taken alone the pattern is indistinguishable: only the feed tells them apart"

    def series(provider):
        ts = [datetime.fromtimestamp(et_epoch(d, "09:30"), tz=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        df = pl.DataFrame({"ts": ts, "open": 500.0, "high": 500.5, "low": 499.5, "close": 500.0, "volume": vol, "provider": provider}).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
        return UnderlyingSeries(df)

    tape, free = series("alpaca"), series("yahoo")
    assert tape.volume_artefacts == 0 and tape.v[73] == 18_700_000.0 and tape.d_v[0] == pytest.approx(vol.sum())
    assert free.volume_artefacts == 1 and free.v[73] == 0.0, "on a feed known for the artefact the bar's volume is unknown, as before"


def test_the_audit_names_a_daily_extreme_that_no_bar_contains(tmp_path):
    """On 70 of 2,741 SPY sessions the daily high or low lies outside every regular-session minute bar, 65 of them at about
    the previous close: a late or reference-priced trade report. Bars from the consolidated tape are not missing a print."""
    from strata.audit import bars as AB
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    days = [date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)]
    st.write_daily("SPY", pl.DataFrame([{"date": days[0], "open": 500.0, "high": 500.6, "low": 499.4, "close": 500.0, "volume": 39_000.0},
                                        # gap up to 510; the daily low is the previous close, which never traded in the session
                                        {"date": days[1], "open": 510.0, "high": 510.6, "low": 500.02, "close": 510.0, "volume": 39_000.0},
                                        {"date": days[2], "open": 510.0, "high": 510.6, "low": 509.4, "close": 510.0, "volume": 39_000.0}]), provider="yahoo")
    for prov in ("alpaca", "yahoo"):
        rows = []
        for d, px in zip(days, (500.0, 510.0, 510.0)):
            for i in range(390):
                rows.append({"ts": datetime.fromtimestamp(et_epoch(d, "09:30"), tz=timezone.utc) + timedelta(minutes=i), "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 100.0})
        st.write_bars("SPY", "1m", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), provider=prov)
    daily = st.read_daily("SPY").select("date", "open", "high", "low", "close", "volume")
    tape = AB.audit_dataset(st, "SPY", "1m", "alpaca", daily)
    f = next(x for x in tape["findings"] if x["check"] == "daily high or low outside the regular-session bars")
    assert f["level"] == "info" and f["sessions"] == 1 and "on 1 of them that extreme is within 0.1% of the PREVIOUS close" in f["detail"] and "nothing is missing from them" in f["detail"]
    assert tape["daily_extreme_outside_bars"] == {"sessions": 1, "at_previous_close": 1} and tape["volume_rule"].startswith("not applied")
    free = AB.audit_dataset(st, "SPY", "1m", "yahoo", daily)
    g = next(x for x in free["findings"] if x["check"] == "daily high or low outside the regular-session bars")
    assert g["level"] == "warn" and "Either prints are missing" in g["detail"], "on a feed of unknown origin missing prints stay a possibility"
    assert free["shadowed"] == {"sessions": 3, "by": ["alpaca"]} and free["bars"] == 1170, "a dataset is audited on its own bars even when a higher-ranked feed is read first"
    # the two feeds are two records of the same minutes
    agree = AB.cross_feed(st, "SPY", "1m", ["alpaca", "yahoo"])[0]
    assert agree["sessions"] == 3 and agree["bars_compared"] == 1170 and agree["close_within_0_02_pct"] == 100.0 and agree["bars_only_in_a"] == 0


# ----------------------------------------------------------------------------- stored bars rebuilt from individual trades
def test_bars_are_rebuilt_from_trades_and_compared_minute_by_minute():
    from strata.audit import options as AO
    t = lambda hms, p, s: {"t": f"2024-03-04T{hms}Z", "p": p, "s": s, "c": "I"}
    trades = [t("14:30:01.123456789", 2.00, 1), t("14:30:40.5", 2.10, 2), t("14:30:59.999", 1.95, 3), t("14:32:00", 2.20, 5), t("14:32:10.000000001", 2.25, 1)]
    b = AO.bars_from_trades(list(reversed(trades)))  # order of arrival does not matter
    assert b.height == 2
    first, second = b.row(0, named=True), b.row(1, named=True)
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"], first["n"]) == (2.00, 2.10, 1.95, 1.95, 6, 3)
    assert (second["open"], second["high"], second["low"], second["close"], second["volume"]) == (2.20, 2.25, 2.20, 2.25, 6)
    assert first["ts"] == datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc) and second["ts"] == datetime(2024, 3, 4, 14, 32, tzinfo=timezone.utc)
    stored = pl.DataFrame({"ts": [first["ts"], second["ts"], datetime(2024, 3, 4, 14, 40, tzinfo=timezone.utc)], "open": [2.00, 2.20, 3.0], "high": [2.10, 2.30, 3.0], "low": [1.95, 2.20, 3.0], "close": [1.95, 2.25, 3.0],
                           "volume": [6, 6, 1]}).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    c = AO.compare_bars(stored, b)
    assert c == {"stored_bars": 3, "rebuilt_bars": 2, "minutes_in_both": 2, "only_stored": 1, "only_in_trades": 0, "prices_identical": 1, "prices_and_volume_identical": 1, "close_identical": 2, "largest_close_difference": 0.0}
    assert AO.bars_from_trades([]).is_empty()


def test_the_comparison_with_trades_is_blocked_without_a_key_and_exact_with_one(store, monkeypatch):
    from strata.audit import options as AO
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("", ""))
    rep = AO.verify_against_trades(store, "SPY", sample=4, save=False)
    assert rep["status"] == "blocked" and "BLOCKED" in rep["reason"] and rep["pairs"] == [], "no key: nothing is compared and nothing is reported as verified"
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    use(monkeypatch, serve({D2: ("09:30", "15:59")}))
    S.Alpaca(store).acquire("SPY", {D2: chain(D2, 12)})

    def trades(req):
        # the same three minutes as individual trades; one contract has a minute of trades the bars lack
        occ = req.url.params["symbols"]
        px = lambda hhmm, p: {"t": _ts(D2, hhmm), "p": p, "s": 10, "c": "I"}
        tr = [px("09:30", 2.0), px("12:00", 2.2), px("15:59", 1.8)]
        if occ.endswith("C00494000"):
            tr.append(px("13:00", 9.9))
        return httpx.Response(200, json={"trades": {occ: tr}, "next_page_token": None})
    use(monkeypatch, trades)
    monkeypatch.setattr(AO, "_sample", lambda idx, n, seed: idx.filter(pl.col("status") == "ok").head(n).to_dicts())
    rep = AO.verify_against_trades(store, "SPY", sample=5, save=False)
    assert rep["status"] == "ran" and rep["pairs_compared"] == 5 and rep["stored_bars"] == 15
    # the mocked bars carry a high and low five cents around the price; single trades cannot reproduce that, so prices differ
    assert rep["prices_identical"] == 0 and rep["close_identical"] == 15 and rep["only_stored"] == 0 and rep["only_in_trades"] == 1
    assert rep["verified"] is False, "bars that the trades do not reproduce are not reported as verified"


def test_two_providers_holding_the_same_contract_are_compared(tmp_path):
    from strata.audit import options as AO
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    occ = S.occ_symbol("SPY", D1, "C", 500.0)

    def put(provider, prints):
        rows = [{"ts": datetime.fromtimestamp(et_epoch(D1, t), tz=timezone.utc), "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "open": px, "high": px, "low": px, "close": px, "volume": v} for t, px, v in prints]
        st.write_option_bars("SPY", provider, pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))),
                             [{"date": D1, "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "bars": len(rows), "status": "ok", "pages": 1}])
    put("alpaca", [("09:30", 2.0, 5), ("09:31", 2.1, 5), ("09:32", 2.2, 5), ("09:40", 2.5, 1)])
    put("yahoo_options", [("09:30", 2.0, 5), ("09:31", 2.1, 4), ("09:32", 2.25, 5)])
    c = AO.cross_provider(st, "SPY")[0]
    assert {c["a"], c["b"]} == {"alpaca", "yahoo_options"} and c["contract_sessions"] == 1 and c["minutes_in_both"] == 3
    assert c["close_identical_pct"] == pytest.approx(66.67) and c["volume_identical_pct"] == pytest.approx(66.67) and c["largest_close_difference"] == pytest.approx(0.05)
    assert c["only_in_a"] + c["only_in_b"] == 1
    aud = AO.audit_option_bars(st, "SPY", "alpaca")
    assert aud["sessions"] == 1 and aud["bars"] == 4 and aud["contract_sessions_0dte"] == 1 and "no bid or ask" in aud["observation"]
    text = AO.render(aud, None, AO.cross_provider(st, "SPY"))
    assert "against" in text and "one-minute bars" in text


# ----------------------------------------------------------------------------- listing, paging, the command
def test_every_page_of_a_request_is_throttled_and_a_rate_limit_waits_for_the_reset(store, monkeypatch):
    waits, slept = [], []
    monkeypatch.setattr(S.Throttle, "wait", lambda self: waits.append(1))
    monkeypatch.setattr(S.time, "sleep", lambda s: slept.append(s))
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] == 2:
            return httpx.Response(429, headers={"x-ratelimit-reset": str(int(datetime.now(timezone.utc).timestamp()) + 20)}, json={"message": "too many requests"})
        tok = req.url.params.get("page_token")
        nxt = {None: "p2", "p2": "p3", "p3": None}[tok]
        return httpx.Response(200, json={"bars": {"X": [bar(D1, "09:30", 1.0)]}, "next_page_token": nxt})
    use(monkeypatch, handler)
    src = S.Alpaca(store)
    with S._client() as client:
        pages = list(src.pages(client, f"{src.DATA}/v1beta1/options/bars", {"symbols": "X"}, "bars"))
    assert len(pages) == 3 and len(waits) == 3, "one wait per page, not one per request"
    assert len(slept) == 1 and 15 <= slept[0] <= 65, "a 429 waits until the provider's window reopens instead of retrying at once"


def test_the_download_command_is_blocked_without_a_key_and_resumes_with_one(store, monkeypatch, capsys):
    from strata import cli
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("", ""))
    with pytest.raises(SystemExit) as e:
        cli.main(["download", "alpaca", "option_trades", "SPY", "--start", D1.isoformat(), "--end", D2.isoformat()])
    assert e.value.code == 2 and "BLOCKED" in capsys.readouterr().out
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    monkeypatch.setattr(S, "ALPACA_OPTIONS_FIRST", date(2024, 1, 18))

    def handler(req):
        if "/options/contracts" in req.url.path:
            lo, hi = float(req.url.params["strike_price_gte"]), float(req.url.params["strike_price_lte"])
            a, b = date.fromisoformat(req.url.params["expiration_date_gte"]), date.fromisoformat(req.url.params["expiration_date_lte"])
            cs = [{"id": f"id-{i}", "symbol": S.occ_symbol("SPY", e, r, k), "status": "inactive", "expiration_date": e.isoformat(), "root_symbol": "SPY", "underlying_symbol": "SPY", "type": "call" if r == "C" else "put",
                   "style": "american", "strike_price": str(k), "multiplier": "100", "size": "100"} for i, (e, r, k) in enumerate((e, r, float(k)) for e in cal.trading_days(a, b) for r in ("C", "P") for k in range(int(lo), int(hi) + 1))]
            return httpx.Response(200, json={"option_contracts": cs, "next_page_token": None})
        return serve({D1: ("09:30", "15:59"), D2: ("09:30", "15:59")})(req)
    calls = use(monkeypatch, handler)
    with pytest.raises(SystemExit) as e:
        cli.main(["download", "alpaca", "option_trades", "SPY", "--start", D1.isoformat(), "--end", D2.isoformat(), "--band", "1", "--max-dte", "1", "--rights", "call", "--every", "0"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "DONE SPY option_trades" in out
    tot = json.loads(out.split("DONE SPY option_trades: ")[1].split(" in ")[0])
    # +/-1% of the previous close (500) is 495..505: 11 strikes, calls only, the session's expiration and the next one
    assert tot["pairs_ok"] == 2 * 11 * 2 and tot["pairs_failed"] == 0
    assert all(c.url.params.get("status") == "inactive" for c in calls if "/options/contracts" in c.url.path)
    idx = store.option_bar_index("SPY", provider="alpaca")
    assert idx.height == 44 and set(idx["right"].unique().to_list()) == {"C"}
    n = len(calls)
    with pytest.raises(SystemExit) as e:
        cli.main(["download", "alpaca", "option_trades", "SPY", "--start", D1.isoformat(), "--end", D2.isoformat(), "--band", "1", "--max-dte", "1", "--rights", "call", "--every", "0"])
    again = json.loads(capsys.readouterr().out.split("DONE SPY option_trades: ")[1].split(" in ")[0])
    assert again["pairs_cached"] == 44 and again["pairs_ok"] == 0 and not [c for c in calls[n:] if "/options/bars" in c.url.path], "what is stored is not downloaded again"


# ----------------------------------------------------------------------------- a multiplier of zero
def _listed(occ, **kw):
    root, exp, right, strike = S.parse_occ(occ)
    return {"id": f"id-{occ}", "symbol": occ, "status": "inactive", "expiration_date": exp.isoformat(), "root_symbol": root, "underlying_symbol": root, "type": "call" if right == "C" else "put", "style": "american",
            "strike_price": str(strike), "multiplier": "100", "size": "100", "deliverables": [{"type": "equity", "symbol": root, "amount": "100", "allocation_percentage": "100"}], **kw}


def test_a_multiplier_of_zero_in_the_listing_is_not_taken_at_its_word(store, tmp_path):
    """Alpaca lists `multiplier: "0"` for every contract that expired on or before 2024-04-26, next to `size: "100"` and a
    deliverable of 100 shares. The first full run on its data multiplied every such trade by zero and then divided by it."""
    occ = S.occ_symbol("SPY", D1, "C", 500.0)
    assert S.Alpaca.parse_contract(_listed(occ)).multiplier == 100.0
    assert S.Alpaca.parse_contract(_listed(occ, multiplier="0")).multiplier == 100.0, "the contract size says what the zero does not"
    assert S.Alpaca.parse_contract(_listed(occ, multiplier="0", size="0")).multiplier == 100.0, "then the deliverable: 100 shares"
    assert S.Alpaca.parse_contract(_listed(occ, multiplier=None, size=None, deliverables=[])).multiplier == 100.0, "nothing anywhere: the standard contract"
    mini = S.Alpaca.parse_contract(_listed(occ, multiplier="0", size="10", deliverables=[{"type": "equity", "symbol": "SPY", "amount": "10"}]))
    assert mini.multiplier == 10.0 and mini.adjusted is True, "a non-standard size is kept as it is (and such a contract is excluded from selection)"
    # definitions stored before the quirk was understood are rewritten from the cached listing, without a request
    src = S.Alpaca(store)
    import dataclasses
    bad = dataclasses.replace(S.Alpaca.parse_contract(_listed(occ)), multiplier=0.0)
    rows = [{"ts": datetime.fromtimestamp(et_epoch(D1, "09:59"), tz=timezone.utc), "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 5}]
    store.write_option_bars("SPY", "alpaca", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": D1, "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "bars": 1, "status": "ok", "pages": 1}],
                            pl.DataFrame([bad.row("SPY")], infer_schema_length=None))
    key = store.option_dataset("SPY", "trade", "alpaca")
    assert store.read_contracts(key)["multiplier"].to_list() == [0.0]
    (src.cache / "contracts").mkdir(parents=True, exist_ok=True)
    (src.cache / "contracts" / "abc.json").write_text(json.dumps({occ: _listed(occ, multiplier="0")}))
    assert src.repair_definitions("SPY") == {"contracts": 1, "non_positive_multiplier": 1, "repaired": 1, "not_in_cache": 0}
    assert store.read_contracts(key)["multiplier"].to_list() == [100.0] and src.repair_definitions("SPY") == {"contracts": 1, "repaired": 0}


def test_the_engine_does_not_use_a_multiplier_that_is_not_positive(tmp_path):
    """Whatever a stored definition says, a contract cannot control zero shares: the symbol's contract size applies and the run says so."""
    from strata.engine.engine import run_backtest
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [cal.prev_trading_day(D1), D1], lambda d, i: 500.0)
    occ = S.occ_symbol("SPY", D1, "C", 500.0)
    rows = [{"ts": datetime.fromtimestamp(et_epoch(D1, t), tz=timezone.utc), "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "open": px, "high": px, "low": px, "close": px, "volume": 5} for t, px in (("09:59", 2.0), ("15:29", 2.6))]
    import dataclasses
    zero = dataclasses.replace(S.contract_from_occ(occ, source="test"), multiplier=0.0)
    st.write_option_bars("SPY", "testfeed", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": D1, "occ": occ, "expiration": D1, "strike": 500.0, "right": "C", "bars": 2, "status": "ok", "pages": 1}],
                         pl.DataFrame([zero.row("SPY")], infer_schema_length=None))
    cfg = {"start": D1.isoformat(), "end": D1.isoformat(), "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0},
           "strategies": [{"name": "0DTE call", "symbol": "SPY", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
                           "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
                           "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}]}
    res = run_backtest(cfg, store=st)
    assert len(res.trades) == 1 and res.trades[0]["pnl"] == pytest.approx(60.0), "(2.60 - 2.00) x 100 shares, not x 0"
    used = res.meta.get("datasets_used") or res.diagnostics.get("datasets_used") or {}
    assert used["SPY"]["definitions_without_a_usable_multiplier"] == 1
