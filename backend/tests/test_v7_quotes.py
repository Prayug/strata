"""V7: trade prints against bid/ask quotes, and the staged Databento check.

No Databento key has ever been stored here. The provider's answers below are written by hand in the shapes its
documentation gives, and served through a mocked transport: they pin what the code does with such answers, and establish
nothing about what the live service returns. The comparison itself (alignment, execution) runs on fixture datasets."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.providers import databento as DB
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from strata.research import obscompare as OC
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401
from tests.test_v6_alpaca import use

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 3, 8))
PREV = cal.prev_trading_day(DAYS[0])
STRIKES = [498.0, 499.0, 500.0, 501.0, 502.0]
MINUTES = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(60) if (h, m) >= (9, 30)]


def mid(k: float, i: int) -> float:
    """A premium path: worth more in the morning, decaying through the session, cheaper further out of the money."""
    return round(max(0.10, 3.0 - 0.5 * (k - 498) - 0.004 * i), 2)


@pytest.fixture
def store(tmp_settings):
    """Quotes (bid/ask five cents either side of the path, one snapshot a minute) and trade bars (a print at the mid every
    minute) for the same five same-day calls on five sessions."""
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [PREV] + DAYS, lambda d, i: 500.0)
    for d in DAYS:
        q, bars, index = [], [], []
        for k in STRIKES:
            occ = S.occ_symbol("SPY", d, "C", k)
            for i, t in enumerate(MINUTES):
                ts = datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc)
                m = mid(k, i)
                q.append({"ts": ts, "quote_ts": ts, "expiration": d, "strike": k, "right": "C", "bid": round(m - 0.05, 2), "ask": round(m + 0.05, 2), "bid_size": 20, "ask_size": 20})
                bars.append({"ts": ts, "occ": occ, "expiration": d, "strike": k, "right": "C", "open": m, "high": m, "low": m, "close": m, "volume": 9})
            index.append({"date": d, "occ": occ, "expiration": d, "strike": k, "right": "C", "bars": len(MINUTES), "status": "ok", "pages": 1})
        st.write_option_day("SPY", d, pl.DataFrame(q).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").cast(pl.Datetime("us", "UTC"))), provider="quotefeed", res="1m", complete_chain=False)
        st.write_option_bars("SPY", "tradefeed", pl.DataFrame(bars).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), index)
    return st


def strategy() -> dict:
    return {"start": DAYS[0].isoformat(), "end": DAYS[-1].isoformat(), "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0, "max_quote_age_seconds": 120},
            "strategies": [{"name": "call at 10:00", "symbol": "SPY", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
                            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
                            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}]}


# ----------------------------------------------------------------------------- do the two datasets describe the same market
def test_prints_inside_the_quoted_market_show_that_the_clocks_and_the_contracts_agree(store):
    rep = OC.alignment(store, "SPY", "quotefeed", "tradefeed")
    assert rep["status"] == "ran" and rep["sessions"] == 5 and rep["contracts_both"] == 25 and rep["contracts_only_trades"] == 0 and rep["identifiers_aligned"]
    assert rep["inside_pct"] == 100.0 and rep["close_inside_pct"] == 100.0 and rep["clock"] == {"best": "as labelled", "aligned": True}
    assert rep["print_position_in_half_spreads"]["median"] == pytest.approx(0.0, abs=0.11), "the prints are at the midpoint (within the path's cent a minute)"
    assert "so the two clocks agree" in rep["text"] and rep["quoted_spread_pct_of_mid"]["median"] > 0


def test_a_quote_feed_labelled_a_minute_off_is_caught_before_anything_is_priced(tmp_settings):
    """The same market, but the prints move a dollar every minute and the quote records are stamped a minute late: prints
    fall outside the quotes as labelled and inside them a minute earlier."""
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    d = DAYS[0]
    occ = S.occ_symbol("SPY", d, "C", 500.0)
    q, bars = [], []
    for i, t in enumerate(MINUTES[:120]):
        ts = datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc)
        m = 50.0 + i  # the market at the start of minute i: a dollar a minute
        # the quote that was true at ts is stamped a minute late
        q.append({"ts": ts + timedelta(minutes=1), "quote_ts": ts + timedelta(minutes=1), "expiration": d, "strike": 500.0, "right": "C", "bid": m - 0.05, "ask": m + 0.05, "bid_size": 1, "ask_size": 1})
        # the bar that starts at ts trades from m up to just under the next minute's market
        bars.append({"ts": ts, "occ": occ, "expiration": d, "strike": 500.0, "right": "C", "open": m, "high": m + 0.98, "low": m, "close": m + 0.98, "volume": 1})
    st.write_option_day("SPY", d, pl.DataFrame(q).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").cast(pl.Datetime("us", "UTC"))), provider="quotefeed", res="1m")
    st.write_option_bars("SPY", "tradefeed", pl.DataFrame(bars).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": d, "occ": occ, "expiration": d, "strike": 500.0, "right": "C", "bars": 120, "status": "ok", "pages": 1}])
    rep = OC.alignment(st, "SPY", "quotefeed", "tradefeed")
    assert rep["close_inside_pct"] < 5 and rep["earlier_pct"] > 95 and rep["clock"] == {"best": "a minute earlier", "aligned": False}
    assert "do NOT agree: the quotes fit best a minute earlier" in rep["text"]


def test_without_stored_quotes_the_comparison_is_blocked_and_estimates_nothing(store):
    rep = OC.alignment(store, "SPY", "databento", "tradefeed")
    assert rep["status"] == "blocked" and rep["reason"].startswith("BLOCKED: no bid/ask quote dataset is stored for SPY from databento") and "none are estimated" in rep["reason"]
    ex = OC.execution(strategy(), store, "databento", "tradefeed")
    assert ex["status"] == "blocked" and "none are estimated" in ex["reason"] and "pnl" not in ex


# ----------------------------------------------------------------------------- what the spread costs
def test_the_same_strategy_on_prints_at_the_quote_and_at_the_midpoint(store):
    rep = OC.execution(strategy(), store, "quotefeed", "tradefeed")
    assert rep["status"] == "ran" and rep["sessions_in_both"] == 5 and rep["trades"] == {"trade_prints": 5, "quotes": 5, "in_both": 5, "same_contract": 5, "only_trade_prints": 0, "only_quotes": 0}
    p = rep["pnl"]
    # bought at 10:00 and sold at 15:30: the path loses the same amount either way; crossing a ten-cent spread twice costs ten dollars a contract
    assert p["quotes_at_the_midpoint"] - p["quotes_paying_the_spread"] == pytest.approx(5 * 10.0) and rep["spread_cost"] == pytest.approx(50.0)
    assert p["trade_prints"] == pytest.approx(p["quotes_at_the_midpoint"], abs=5 * 1.0), "prints at the midpoint give the midpoint result (to the cent a minute the path moves)"
    assert abs(rep["entry_print_position_in_half_spreads"]["median"]) < 0.25 and rep["rows"][0]["same_contract"] is True
    r0 = rep["rows"][0]
    assert r0["quotes"]["entry"] == r0["quotes"]["entry_ask"] and r0["quotes"]["exit"] == r0["quotes"]["exit_bid"], "the quote run buys at the ask and sells at the bid"
    assert "crossing the quoted spread cost 50.00" in rep["text"]


def test_a_band_of_quotes_is_a_band_the_outermost_quoted_strike_is_not_the_one_asked_for(store):
    """The quote dataset says it holds a band of strikes (complete_chain False), like a selective download. A strike beyond it
    is refused exactly as on trade data; a quote dataset that does not say so is a whole chain and keeps its old behaviour."""
    from strata.engine.engine import run_backtest
    c = strategy()
    c["strategies"][0]["legs"][0]["strike"] = {"method": "pct_otm", "value": 2.0}  # strike 510: the band stops at 502
    c.update(pricing={"mode": "historical"}, data={"options_provider": "quotefeed"})
    res = run_backtest(c, store=store, record_paths=False)
    assert res.trades == [] and res.meta["contract_selection"]["selective_storage"] is True and len(res.meta["contract_selection"]["beyond_dates"]) == 0  # quote-mode refusals are recorded as no_contract
    assert res.meta["executability"]["not_executed_by_reason"] == {"no_contract": 5}


# ----------------------------------------------------------------------------- the staged provider check
KEY = "db-" + "a1" * 14 + "b"


def provider(calls: list, datasets=("OPRA.PILLAR", "XNAS.ITCH"), csv_header="ts_recv,ts_event,rtype,publisher_id,instrument_id,price,size,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,symbol"):
    def handler(req: httpx.Request) -> httpx.Response:
        path, q = req.url.path.split("/v0/")[-1], dict(req.url.params)
        calls.append((path, q, req.headers.get("authorization", "")[:6]))
        if path == "metadata.list_datasets":
            return httpx.Response(200, json=list(datasets))
        if path == "metadata.get_dataset_range":
            return httpx.Response(200, json={"start": "2013-04-01T00:00:00.000000000Z", "end": "2026-11-26T00:00:00.000000000Z", "schema": {"cbbo-1m": {"start": "2013-04-01T00:00:00.000000000Z", "end": "2026-11-26T00:00:00.000000000Z"}}})
        if path == "metadata.list_unit_prices":
            return httpx.Response(200, json=[{"mode": "historical-streaming", "unit_prices": {"cbbo-1m": 2.0, "trades": 280.0}}, {"mode": "live", "unit_prices": {"cbbo-1m": 2.4}}])
        if path == "metadata.get_cost":
            return httpx.Response(200, json=0.0125 * len(q["symbols"].split(",")))
        if path == "timeseries.get_range":
            d = date.fromisoformat(q["start"][:10])
            lines = [csv_header]
            for raw in q["symbols"].split(","):
                _r, exp, right, k = DB.from_osi(raw)
                for i, t in enumerate(MINUTES[:60]):
                    ts = datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + ".000000000Z"
                    m = mid(k, i)
                    lines.append(f"{ts},{ts},195,30,1,{m:.2f},1,{m - 0.05:.2f},{m + 0.05:.2f},20,20,{raw}" if "bid_px_00" in csv_header else f"{ts},{ts},{m - 0.05:.2f},{m + 0.05:.2f},20,20,{raw}")
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(404, text="unexpected")
    return handler


def alpaca_archive(st: MarketStore):
    """What the sample is chosen from: stored Alpaca trade bars (here the fixture's, under the provider's name) and a daily history."""
    for d in DAYS:
        b = st.read_option_bars("SPY", d, provider="tradefeed")
        idx = st.option_bar_index("SPY", provider="tradefeed").filter(pl.col("date") == d).drop("fetched_at", "window", strict=False)
        st.write_option_bars("SPY", "alpaca", b, idx.to_dicts())
    days = [PREV] + DAYS
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 1.0} for d in days]), provider="yahoo")


def test_without_a_key_the_check_is_blocked_and_says_exactly_how_to_connect(store, monkeypatch):
    monkeypatch.setattr(DB, "_key", lambda: "")
    calls = use(monkeypatch, provider([]))
    alpaca_archive(store)
    rep = DB.access(store, "SPY", sessions=4)
    assert rep["status"] == "blocked" and rep["verified"] is False and rep["reason"].startswith("BLOCKED: no Databento key is stored") and rep["needs"] == ["DATABENTO_API_KEY"]
    assert calls == [], "nothing is requested without a key"
    how = " ".join(rep["how"])
    assert "https://databento.com and choose Sign up" in how and "portal/keys" in how and "starts with db-" in how and "$125" in how and "strata access databento --budget" in how and "portal/billing" in how
    assert "Strata cannot see the credit or the limit" in how, "what the application cannot check is said, not implied"
    assert rep["sample"]["symbols"] > 0 and rep["sample"]["estimate_usd"] < 0.01 and "not a quote" not in rep["sample"]["estimate_note"] and "estimate" in rep["sample"]["estimate_note"]
    text = DB.render(rep)
    assert "BLOCKED" in text and "To connect it:" in text and "never run against the live service" in text and KEY not in text
    assert DB.saved()["status"] == "blocked"


def test_with_a_key_the_free_stages_run_and_it_stops_at_the_price(store, monkeypatch):
    monkeypatch.setattr(DB, "_key", lambda: KEY)
    calls: list = []
    use(monkeypatch, provider(calls))
    alpaca_archive(store)
    rep = DB.access(store, "SPY", sessions=4)
    assert rep["status"] == "quoted" and [s["stage"] for s in rep["stages"]] == ["key", "datasets", "range", "prices", "quote"] and all(s["ok"] for s in rep["stages"])
    assert rep["unit_price_usd_per_gb"] == 2.0 and rep["sample"]["quoted_usd"] == pytest.approx(0.0125 * rep["sample"]["symbols"]) and rep["spent_usd"] == 0.0
    assert "Stopped before the first billable request" in rep["reason"] and "no budget was given" in rep["reason"]
    assert not any(p == "timeseries.get_range" for p, *_ in calls), "the download endpoint was never touched"
    assert all(a == "Basic " for *_x, a in calls) and store.option_dataset("SPY", "quote", "databento") is None
    cost = next(q for p, q, _ in calls if p == "metadata.get_cost")
    assert cost["schema"] == "cbbo-1m" and cost["stype_in"] == "raw_symbol" and cost["dataset"] == "OPRA.PILLAR" and cost["symbols"].split(",")[0] == DB.osi("SPY", DAYS[0], "C", 498.0)
    assert cost["start"].endswith("14:30") and cost["end"].endswith("21:10"), "aligned to ten-minute boundaries, as the provider asks for an exact quote"
    # a budget below the quote changes nothing
    low = DB.access(store, "SPY", sessions=4, budget_usd=0.01)
    assert low["status"] == "quoted" and "the budget is $0.01" in low["reason"] and store.option_dataset("SPY", "quote", "databento") is None


def test_with_a_budget_that_covers_the_quote_the_sample_is_stored_as_a_band_and_set_beside_the_trades(store, monkeypatch):
    monkeypatch.setattr(DB, "_key", lambda: KEY)
    calls: list = []
    use(monkeypatch, provider(calls))
    alpaca_archive(store)
    rep = DB.access(store, "SPY", sessions=4, budget_usd=1.0)
    assert rep["status"] == "sampled" and rep["verified"] is True and rep["spent_usd"] == pytest.approx(rep["sample"]["quoted_usd"]) and rep["spent_usd"] <= 1.0
    key = store.option_dataset("SPY", "quote", "databento")
    assert key is not None and key.res == "1m" and store.descriptor(key)["complete_chain"] is False, "the dataset says it is a band of contracts, not a chain"
    assert len(store.option_days("SPY", "quote", dataset=key)) == len(rep["sample"]["sessions"])
    d0 = date.fromisoformat(rep["sample"]["sessions"][0]["date"])
    df = store.read_option_day("SPY", d0, "quote", dataset=key)
    assert df.height == 5 * 60 and df["bid"].min() > 0 and (df["ask"] >= df["bid"]).all() and df["ts"].min() == datetime.fromtimestamp(et_epoch(d0, "09:30"), tz=timezone.utc)
    al = rep["alignment"]
    assert al["status"] == "ran" and al["inside_pct"] == 100.0 and al["clock"]["aligned"] and al["contracts_both"] > 0
    dl = next(q for p, q, _ in calls if p == "timeseries.get_range")
    assert dl["encoding"] == "csv" and dl["pretty_px"] == "true" and dl["map_symbols"] == "true"
    assert "The sample was downloaded with this key" in DB.render(rep)


def test_an_answer_in_another_shape_stores_nothing_and_shows_the_header(store, monkeypatch):
    monkeypatch.setattr(DB, "_key", lambda: KEY)
    use(monkeypatch, provider([], csv_header="ts_recv,ts_event,bid,ask,bid_size,ask_size,raw_symbol"))
    alpaca_archive(store)
    rep = DB.access(store, "SPY", sessions=2, budget_usd=1.0)
    assert rep["status"] == "failed" and "does not have the columns this adapter was written for" in rep["reason"] and "the header was: ts_recv,ts_event,bid,ask" in rep["reason"].replace(", ", ",")
    assert store.option_dataset("SPY", "quote", "databento") is None
    use(monkeypatch, provider([], datasets=("XNAS.ITCH",)))
    assert DB.access(store, "SPY", sessions=2)["status"] == "not_entitled"


def test_the_raw_symbol_is_the_21_character_osi_form():
    raw = DB.osi("SPY", date(2024, 1, 19), "C", 470.0)
    assert raw == "SPY   240119C00470000" and len(raw) == 21 and DB.from_osi(raw) == ("SPY", date(2024, 1, 19), "C", 470.0)
    assert DB.from_osi("TSLA 230901C00250000") == ("TSLA", date(2023, 9, 1), "C", 250.0), "the single-space form seen in the provider's samples is read too"
    assert DB.osi("SPXW", date(2024, 3, 15), "P", 5012.5) == "SPXW  240315P05012500"
    assert DB._ts("2024-01-19T14:31:00.123456789Z") == datetime(2024, 1, 19, 14, 31, 0, 123456, tzinfo=timezone.utc)
