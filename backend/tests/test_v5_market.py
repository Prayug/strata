"""V5: the provider marketplace, the new sources, the selective cost estimate and the guided
Alpaca connection. Offline: provider responses are recorded shapes or a real trimmed
response; nothing here may claim a live verification."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)

FIX = Path(__file__).parent / "fixtures"


def test_the_marketplace_separates_what_is_advertised_from_what_was_verified(tmp_settings):
    from strata.data.providers import bench as B
    from strata.data.providers import market as MK
    out = MK.listing()
    rows = {r["id"]: r for r in out["sources"]}
    assert len(rows) >= 20 and out["priority_symbols"] == ["SPY", "QQQ", "IWM", "SPX"] and out["checked"]
    for r in rows.values():
        assert set(r["symbols"]) == {"SPY", "QQQ", "IWM", "SPX"} and r["pricing"]["summary"] and r["limits"] and r["url"].startswith("https://") and r["tested"] and r["verification"]["status"]
        assert r["advertised"]["history"] and r["integration"] in ("built-in", "adapter", "import", "none")
    # nothing has been checked on this fresh installation, so nothing may say verified
    assert out["summary"]["verified"] == 0 and all(r["verification"]["status"] != "verified" for r in rows.values())
    assert rows["alpaca"]["verification"]["status"] == "unverified" and "No credentials" in rows["alpaca"]["verification"]["detail"]
    assert rows["massive"]["verification"]["status"] == "unverified" and "never run against the live service" in rows["massive"]["verification"]["detail"]
    assert rows["databento"]["verification"]["status"] == "unverified" and rows["github_eod"]["verification"]["status"] == "not checked yet"
    assert rows["cboe_datashop"]["verification"]["status"] == "import only" and "not tested on a purchased file" in rows["cboe_datashop"]["verification"]["detail"]
    assert rows["tradier"]["verification"]["status"] == "no integration" and "Expired contracts are not kept" in rows["tradier"]["limits"][0]
    # a recorded live result changes the row; a mocked test never does
    B._save("github_eod", {"source": "github_eod", "status": "verified", "verified": True, "measured_at": "2026-11-25T10:00:00+00:00", "steps": [{"name": "repository listing", "ok": True, "dataset": "SPY", "first": "2008", "last": "2025"}], "notes": []})
    again = {r["id"]: r for r in MK.listing()["sources"]}
    assert again["github_eod"]["verification"]["status"] == "verified" and again["github_eod"]["verification"]["checked_at"].startswith("2026-11-25") and MK.listing()["summary"]["verified"] == 1


def test_alpha_vantage_chains_are_parsed_from_a_real_response_and_notices_are_errors():
    from strata.data.providers import sources as S
    payload = json.loads((FIX / "alphavantage_historical_options_IBM_2017-11-15.json").read_text())
    assert "fetched 2026-11-25" in payload["_provenance"]
    df = S.AlphaVantage.parse_chain(payload)
    assert df.height == 40 and set(df["right"].unique()) == {"C", "P"} and df["_d"].unique().to_list() == [date(2017, 11, 15)]
    assert (df["bid"] <= df["ask"]).all() and df["delta"].is_not_null().all() and df["expiration"].min() == date(2017, 11, 17)
    row = df.filter((pl.col("strike") == 140) & (pl.col("right") == "C") & (pl.col("expiration") == date(2017, 11, 17))).row(0, named=True)
    assert row["bid"] == 7.10 and row["ask"] == 7.30 and row["open_interest"] == 226 and abs(row["delta"] - 0.95222) < 1e-9
    with pytest.raises(S.SourceError, match="premium"):
        S.AlphaVantage.parse_chain({"Information": "Thank you for using Alpha Vantage! This is a premium endpoint."})
    assert S._redact("https://x.test/query?function=A&apikey=SECRET123&symbol=SPY") == "https://x.test/query?function=A&apikey=[redacted]&symbol=SPY"


def test_the_massive_adapter_reads_documented_shapes_and_sends_its_key_in_a_header(tmp_settings, monkeypatch):
    from strata.api import settings as ST
    from strata.core import calendar as cal
    from strata.data.providers import sources as S
    assert S.Massive().available()[0] is False
    ST.set_secret("POLYGON_API_KEY", "test-key")
    src = S.Massive()
    d = date(2025, 6, 4)
    calls = []

    class R:
        def __init__(self, data):
            self._d, self.status_code, self.headers, self.content = data, 200, {}, b"{}"

        def json(self):
            return self._d

    def fake(client, url, params=None, headers=None, retries=4):
        calls.append((url, dict(params or {}), dict(headers or {})))
        if "/reference/options/contracts" in url:
            if (params or {}).get("expired") == "false":
                return R({"results": []})
            return R({"results": [{"ticker": "O:SPY250604C00595000", "underlying_ticker": "SPY", "expiration_date": "2025-06-04", "strike_price": 595, "contract_type": "call", "exercise_style": "american", "shares_per_contract": 100},
                                  {"ticker": "O:SPY1250604C00595000", "underlying_ticker": "SPY", "expiration_date": "2025-06-04", "strike_price": 595, "contract_type": "call", "exercise_style": "american", "shares_per_contract": 110}]})
        o = int(cal.session_bounds(d)[0].timestamp() * 1000)
        return R({"results": [{"t": o + 60000 * i, "o": 1.0 + i / 100, "h": 1.1 + i / 100, "l": 0.9 + i / 100, "c": 1.05 + i / 100, "v": 10 + i, "n": 3, "vw": 1.02} for i in range(5)] + [{"t": o + 86400000 * 2, "o": 9, "h": 9, "l": 9, "c": 9, "v": 1}]})

    monkeypatch.setattr(S, "http_get", fake)
    cs = src.list_contracts(None, "SPY", d, d, d, 590, 600)
    assert [c.occ for c in cs] == ["SPY250604C00595000"], "the adjusted (non-standard) contract is left out"
    df, status, pages = src.fetch_batch(None, "SPY", cs, [d])
    assert df.height == 5 and status == {(d, "SPY250604C00595000"): "ok"} and pages == 1, "a bar from another session in the same response is not stored for this one"
    assert all(h.get("Authorization") == "Bearer test-key" for _u, _p, h in calls) and not any("test-key" in u or "test-key" in json.dumps(p) for u, p, _h in calls), "the key is never in a URL"
    assert src.batches({d: cs, d + timedelta(days=1): cs}) == [(cs, [d, d + timedelta(days=1)])], "one request per contract covers all its sessions"
    assert abs(src.rate - 5 / 60) < 1e-9


def test_wrds_optionmetrics_exports_are_recognised_and_strikes_are_rescaled(tmp_path):
    from strata.data import bulk as BK
    df = pl.DataFrame({"secid": [1, 1], "date": ["2019-03-01", "2019-03-01"], "symbol": ["SPY 190315C280000", "SPY 190315P280000"], "exdate": ["2019-03-15", "2019-03-15"], "cp_flag": ["C", "P"], "strike_price": [280000, 280000],
                       "best_bid": [2.10, 1.95], "best_offer": [2.14, 1.99], "volume": [100, 80], "open_interest": [5000, 4000], "impl_volatility": [0.12, 0.13], "delta": [0.52, -0.48], "ticker": ["SPY", "SPY"]})
    assert BK.detect(df.columns, df) == "wrds_optionmetrics"
    out, sym = BK.normalise(df, "wrds_optionmetrics")
    assert sym == "SPY" and out["strike"].to_list() == [280.0, 280.0] and out["bid"].to_list() == [2.10, 1.95] and out["right"].to_list() == ["C", "P"] and str(out["expiration"][0]) == "2019-03-15"


def _strategy_cfg():
    from tests.test_v5_engine import D1, call
    return {"start": D1.isoformat(), "end": D1.isoformat(), "strategies": [call(1)], "pricing": {"mode": "trades"}}


def test_the_cost_estimate_counts_only_what_one_strategy_needs_and_never_invents_a_price(tmp_path):
    from strata.data.providers import market as MK
    from tests.test_v5_engine import setup
    st = setup(tmp_path)
    need = MK.needs_of(_strategy_cfg(), st)
    assert need["signals"] == 1 and need["signal_sessions"] == 1 and need["option_legs"] == 1 and need["contracts_per_signal"] == 13 and need["days"] == ["2024-03-04"] and need["intraday"] is True
    est = MK.estimate(need, today=date(2026, 11, 25))
    rows = {r["id"]: r for r in est["sources"]}
    assert rows["alpaca"]["covered_sessions"] == 1 and rows["alpaca"]["dollars"] == 0.0 and rows["alpaca"]["requests"] == 1 and rows["alpaca"]["ready"] is False
    assert rows["massive"]["covered_sessions"] == 0 and any("None of the signal sessions" in n for n in rows["massive"]["notes"]), "March 2024 is older than the free plan's two years"
    # V6: the provider documents $2 per GB for 1-minute bid/ask, so the figure is an estimate from that rate (labelled as one), and the exact price is still the provider's own quote
    assert rows["databento"]["dollars"] == 0.0 and "Estimate:" in rows["databento"]["dollars_note"] and "exact price" in rows["databento"]["dollars_note"] and rows["databento"]["records"] == 13 * 390
    assert rows["databento"]["within_credit"] is True and rows["databento"]["credit"] == 125.0 and rows["databento"]["gigabytes"] == round(13 * 390 * 80 / 1e9, 4)
    assert rows["alphavantage"]["dollars"] == 49.99 and any("cannot price this strategy's intraday entries" in n for n in rows["alphavantage"]["notes"])
    assert rows["github_eod"]["megabytes"] == 180 and any("never called without your approval" in n for n in est["notes"])
    many = MK.estimate({**need, "days": [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(0, 200, 2)]}, today=date(2026, 11, 25))
    m = {r["id"]: r for r in many["sources"]}["massive"]
    assert m["requests"] == 100 * 13 and m["minutes"] == 260.0 and any("hours" in n for n in m["notes"])


def test_the_guided_alpaca_connection_claims_nothing_without_credentials(tmp_settings):
    from strata.data.providers import bench as B
    from strata.data.providers import market as MK
    out = MK.alpaca_validation()
    assert out["ok"] is False and out["verified"] is False and [s["status"] for s in out["steps"]] == ["missing", "not_run", "not_run", "not_run", "not_run"]
    assert [s["id"] for s in out["steps"]] == ["credentials", "entitlements", "contract", "bars", "backtest"] and "nothing is claimed to work" in out["summary"]
    assert B.saved()["alpaca_validation"]["verified"] is False
    assert {r["id"]: r for r in MK.listing()["sources"]}["alpaca"]["verification"]["status"] == "unverified"


def test_the_guided_alpaca_connection_runs_every_step_on_a_recorded_account(tmp_settings, monkeypatch):
    """With credentials and a provider that answers (recorded response shapes), the workflow
    retrieves an expired contract, stores its trade bars and backtests them. This exercises
    the workflow, not Alpaca: the result is not a live verification and the test says so."""
    from strata.api import settings as ST
    from strata.core import calendar as cal
    from strata.data.providers import bench as B
    from strata.data.providers import market as MK
    from strata.data.providers import sources as S
    from strata.data.store import MarketStore, SymbolMeta
    from tests.helpers import write_minutes
    ST.set_secret("ALPACA_API_KEY", "k")
    ST.set_secret("ALPACA_API_SECRET", "s")
    st = MarketStore()
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    today = B.datetime.now(cal.ET).date()
    exp = [d for d in cal.trading_days(today - timedelta(days=45), today - timedelta(days=25)) if d.weekday() == 4][-1]
    prev = cal.trading_days(exp - timedelta(days=7), exp - timedelta(days=1))[-1]
    write_minutes(st, "SPY", [prev, exp], lambda d, i: 500.0)
    occ = S.occ_symbol("SPY", exp, "C", 500.0)

    class R:
        def __init__(self, data):
            self._d, self.status_code, self.headers, self.content = data, 200, {"x-ratelimit-limit": "200"}, b"{}"

        def json(self):
            return self._d

    def fake(client, url, params=None, headers=None, retries=4):
        assert headers and headers.get("APCA-API-KEY-ID") == "k"
        if "/v2/options/contracts" in url:
            return R({"option_contracts": [{"symbol": occ, "root_symbol": "SPY", "expiration_date": exp.isoformat(), "strike_price": "500", "type": "call", "status": "inactive", "id": "abc", "multiplier": "100", "style": "american"}] if (params or {}).get("status") == "inactive" else []})
        if "/v1beta1/options/bars" in url:
            o = cal.session_bounds(exp)[0]
            bars = [{"t": (o + timedelta(minutes=i)).astimezone(B.timezone.utc).isoformat().replace("+00:00", "Z"), "o": 2.0, "h": 2.1, "l": 1.9, "c": 2.0 + (i % 5) / 100, "v": 25, "n": 4, "vw": 2.0} for i in range(0, 380)]
            return R({"bars": {occ: bars}, "next_page_token": None})
        if "/v1beta1/options/quotes" in url:
            raise S.SourceError("HTTP 403 from data.alpaca.markets: subscription does not permit querying OPRA data")
        if "/v2/stocks/bars" in url:
            if (params or {}).get("feed") == "sip":
                raise S.SourceError("HTTP 403 from data.alpaca.markets: subscription does not permit querying recent SIP data")
            return R({"bars": {"SPY": [{"t": "2016-01-04T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}})
        raise AssertionError(url)

    monkeypatch.setattr(S, "http_get", fake)
    out = MK.alpaca_validation(st)
    by = {s["id"]: s for s in out["steps"]}
    assert [s["status"] for s in out["steps"]] == ["ok"] * 5, [s["detail"] for s in out["steps"] if s["status"] != "ok"]
    ent = {e["what"]: e["granted"] for e in by["entitlements"]["entitlements"]}
    assert ent["option bid/ask quotes, OPRA feed"] is False and ent["stock bars, SIP feed"] is False and ent["stock bars, IEX feed"] is True, "what the account cannot read is reported, not assumed"
    assert by["contract"]["occ"] == occ and by["bars"]["bars"] == 380 and "trade prints" in by["bars"]["detail"]
    assert by["backtest"]["trade"]["exit_reason"] == "time_exit" and "nothing was modelled" in by["backtest"]["detail"]
    assert st.option_bar_index("SPY", provider="alpaca").height == 1


# ----------------------------------------------------------------------------- the collector, unattended
def _clock(monkeypatch, CS, y, m, d, hh=17):
    from datetime import datetime
    monkeypatch.setattr(CS, "datetime", type("dt", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(y, m, d, hh, 0, tzinfo=tz))}))


def test_an_interrupted_cycle_is_detected_and_two_cycles_never_overlap(tmp_settings, monkeypatch):
    from strata.api import db
    from tests.test_v4_execution_data import collector_env
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    _clock(monkeypatch, CS, 2024, 3, 8)
    with db.connect() as con:  # a cycle from another process whose heartbeat is fresh
        con.execute("INSERT INTO collector_runs (started_at, status, trigger, report) VALUES (?,?,?,?)", (db.now(), "running", "schedule", json.dumps({"heartbeat": db.now(), "pid": 1})))
    with pytest.raises(RuntimeError, match="another collector process is running"):
        CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert opt.asked == [], "nothing was requested while another cycle held the collector"
    with db.connect() as con:  # ... and now that process has died: its heartbeat is an hour old
        con.execute("UPDATE collector_runs SET report=? WHERE status='running'", (json.dumps({"heartbeat": "2020-01-01T00:00:00+00:00"}),))
    r = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    hist = CS.history(5)
    assert r["status"] == "done" and [h["status"] for h in hist] == ["done", "interrupted"] and "stopped without finishing" in hist[1]["report"]["notes"][0]


def test_failed_sessions_are_queued_retried_with_backoff_and_given_up_after_the_limit(tmp_settings, monkeypatch):
    from strata.api import notify as N
    from tests.test_v4_execution_data import collector_env
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch, fail_after=0)
    CS.save_config({"symbols": ["SPY"], "max_attempts": 2})
    _clock(monkeypatch, CS, 2024, 3, 8)
    r1 = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    q = CS.queue("pending")
    assert r1["status"] == "partial" and {x["session"] for x in q} == {"2024-03-07", "2024-03-08"} and all(x["attempts"] == 1 and x["not_before"].startswith("2024-03-08T17:30") for x in q), "first retry half an hour later (the test clock reads 17:00)"
    assert CS.due_sessions("SPY") == [], "a queued session is not retried before its delay has passed"
    r2 = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert {x["session"] for x in CS.queue("abandoned")} == {"2024-03-07", "2024-03-08"} and any("given up after 2 failed cycles" in n for n in r2["notes"])
    opt.fail_after = None
    r3 = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert r3["status"] == "done" and CS.queue("pending") == [] and CS.queue("abandoned") == [], "a later success clears the queue"
    notes = N.listing()["notifications"]
    assert [n["level"] for n in notes][:3] == ["info", "warn", "warn"] and "queued and retried" in notes[1]["body"] and N.listing()["unread"] == 3
    assert N.mark_read() == 3 and N.listing(unread_only=True)["notifications"] == []
    assert CS._backoff(1).total_seconds() == 1800 and CS._backoff(3).total_seconds() == 7200 and CS._backoff(20).total_seconds() == 86400


def test_missed_cycles_are_recovered_where_the_provider_still_serves_them_and_the_rest_is_reported(tmp_settings, monkeypatch):
    from strata.api import db
    from tests.test_v4_execution_data import collector_env
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    CS.save_config({"symbols": ["SPY"], "recover_sessions": 10})
    _clock(monkeypatch, CS, 2024, 2, 28)
    assert CS.run_once(store=st, today=date(2024, 2, 28), trigger="test")["status"] == "done"
    with db.connect() as con:  # the machine was then off for a week
        con.execute("UPDATE collector_runs SET started_at=?", ("2024-02-28T22:00:00+00:00",))
    _clock(monkeypatch, CS, 2024, 3, 8)
    opt.bars_window = lambda today=None: (date(2024, 3, 4), date(2030, 1, 1))  # the provider no longer serves anything before 4 March
    r = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    lost = {u["date"] for u in r["unrecoverable"]}
    assert any("reaching back" in n for n in r["notes"]) and lost == {"2024-02-28", "2024-02-29", "2024-03-01"} and all("no longer serves" in u["reason"] for u in r["unrecoverable"])
    assert {x["date"] for x in r["recovered"]} == {"2024-03-04", "2024-03-05", "2024-03-06"} and {d for d, _ in opt.asked} >= {date(2024, 3, 4), date(2024, 3, 8)}
    assert not any(d < date(2024, 3, 4) and d > date(2024, 2, 28) for d, _ in opt.asked), "nothing is requested for sessions the provider cannot serve, and nothing is made up for them"
    state = CS.state(st)
    assert len(state["recoverability"]) >= 8 and any("PERMANENTLY unavailable" in x["after_a_miss"] for x in state["recoverability"]) and "Nothing is collected while the computer is asleep" in state["offline"]


def test_an_unhealthy_provider_costs_one_probe_and_integrity_problems_are_reported(tmp_settings, monkeypatch):
    from tests.test_v4_execution_data import collector_env
    CS, st, opt, yh = collector_env(tmp_settings, monkeypatch)
    CS.save_config({"symbols": ["SPY"]})
    _clock(monkeypatch, CS, 2024, 3, 8)
    opt.health = lambda: (_ for _ in ()).throw(RuntimeError("HTTP 503"))
    r = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert opt.asked == [] and r["status"] == "partial" and r["health"][0]["ok"] is False and "queued for retry" in r["errors"][0] and len(CS.queue("pending")) == 2
    del opt.health
    r = CS.run_once(store=st, today=date(2024, 3, 8), trigger="test")
    assert r["status"] == "done" and r["integrity"][0]["sessions_checked"] == 2 and r["integrity"][0]["problems"] == []
    # the index is then made to disagree with the file: the check says so instead of trusting either
    idx = st.option_bar_index("SPY", provider="yahoo_options")
    ds = st.option_dataset("SPY", "trade", "yahoo_options")
    idx.with_columns(pl.when(pl.col("status") == "ok").then(pl.lit(7, dtype=pl.Int32)).otherwise(pl.col("bars")).alias("bars")).write_parquet(st.dataset_dir(ds) / "index.parquet")
    chk = CS.integrity(st, "SPY", "yahoo_options", [date(2024, 3, 8)])
    assert chk["problems"] and "do not match the index" in chk["problems"][0]
