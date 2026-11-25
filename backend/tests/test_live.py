"""Opt-in tests against the real providers. They are skipped unless STRATA_LIVE=1.

    STRATA_LIVE=1 pytest tests/test_live.py -v

Keyless providers (Yahoo, Cboe, the open option archive, Yahoo option trades) need only
network access. Providers that need an account run only when their credentials are in the
environment (ALPACA_API_KEY + ALPACA_API_SECRET, ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY, OPENROUTER_API_KEY); otherwise each reports itself as skipped with the
reason. Nothing here calls a paid data API, and the model tests send one tiny request each.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

LIVE = os.environ.get("STRATA_LIVE") == "1"
pytestmark = [pytest.mark.skipif(not LIVE, reason="live provider tests are opt-in: set STRATA_LIVE=1"), pytest.mark.timeout(600)]


@pytest.fixture()
def live(tmp_path, monkeypatch):
    """A fresh data directory with downloads enabled (the rest of the suite runs offline)."""
    from strata import config
    from strata.api import db
    from strata.data.store import MarketStore
    monkeypatch.delenv("STRATA_OFFLINE", raising=False)
    before = config.get_settings()
    config.override_settings(data_dir=tmp_path)
    db.init_db()
    yield MarketStore()
    config._settings = before


def need_env(*names: str):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip("credentials not in the environment: " + ", ".join(missing))


# ----------------------------------------------------------------------------- keyless market data
def test_yahoo_daily_and_intraday_bars(live):
    from strata.data import planner as P
    from strata.data.providers.sources import get_source
    out = get_source("yahoo", live).ingest("daily", ["SPY"], None, None)
    assert not out.get("error"), out
    daily = live.read_daily("SPY")
    assert daily.height > 8000 and daily["date"][0] <= date(1993, 2, 1)
    assert daily["date"][-1] >= P.last_completed_session() - timedelta(days=5)  # current, not a stale copy
    assert daily["date"].n_unique() == daily.height
    get_source("yahoo", live).ingest("intraday", ["SPY"], None, None)
    mins = live.read_bars("SPY", "1m")
    assert mins is not None and mins.height > 1000 and mins["ts"].n_unique() == mins.height
    assert not live.is_synthetic("SPY")


def test_cboe_publishes_the_vix_history(live):
    from strata.data.providers.sources import get_source
    get_source("cboe", live).ingest("vix_daily", ["VIX"], None, None)
    vix = live.read_daily("VIX")
    assert vix.height > 9000 and vix["date"][0] <= date(1990, 1, 5) and 5 < float(vix["close"][-1]) < 150


def test_independent_archive_agrees_with_yahoo_closes(live):
    from strata.data.providers.sources import get_source, verify_daily
    get_source("yahoo", live).ingest("daily", ["SPY"], None, None)
    v = verify_daily(live, "SPY")
    assert v["compared"] > 1000 and v["within_0_1pct"] > 0.99, v


def test_yahoo_option_trade_bars_for_a_recent_contract(live):
    """One listed weekly contract near the money, the last completed session: real prints only."""
    import polars as pl
    from strata.core import calendar as cal
    from strata.data import planner as P
    from strata.data.providers import sources as S
    S.get_source("yahoo", live).ingest("daily", ["SPY"], None, None)
    d = P.last_completed_session()
    px = float(live.read_daily("SPY").filter(pl.col("date") <= d)["close"][-1])
    src = next(s for s in S.option_bar_sources(live) if s.name == "yahoo_options")
    with S._client() as client:
        cs = src.list_contracts(client, "SPY", d, d + timedelta(days=2), d + timedelta(days=9), px - 2, px + 2, ("C",))
    assert cs, "no listed contract near the money"
    rep = src.acquire("SPY", {d: cs[:2]})
    assert rep["pairs_failed"] == 0 and rep["pairs_ok"] >= 1, rep
    bars = live.read_option_bars("SPY", d, provider="yahoo_options")
    o, c = (int(x.timestamp()) for x in cal.session_bounds(d))
    ts = bars["ts"].dt.epoch("s")
    assert int(ts.min()) >= o and int(ts.max()) < c
    assert bars.filter(pl.col("volume") <= 0).is_empty() and {"bid", "ask"}.isdisjoint(bars.columns)
    again = src.acquire("SPY", {d: cs[:2]})  # resumable: nothing is requested twice
    assert again["pairs_cached"] == rep["pairs_ok"] + rep["pairs_empty"] and again["requests"] == 0


def test_the_validation_command_passes_end_to_end(tmp_path, monkeypatch):
    from strata import config
    from strata.validate import run_validation
    before = config.get_settings()
    monkeypatch.setenv("STRATA_SECRET_STORE", "file")
    try:
        rep = run_validation(tmp_path / "v", out=lambda *_: None)
    finally:
        config._settings = before
        os.environ["STRATA_OFFLINE"] = "1"
    assert rep["ok"], [(s["n"], s["name"], s["error"]) for s in rep["steps"] if not s["ok"]]
    assert [s["n"] for s in rep["steps"]] == list(range(1, 14))


# ----------------------------------------------------------------------------- Alpaca (free account)
def test_alpaca_connection_stock_bars_and_point_in_time_contracts(live):
    need_env("ALPACA_API_KEY", "ALPACA_API_SECRET")
    import polars as pl
    from strata.core import calendar as cal
    from strata.data import planner as P
    from strata.data.providers import sources as S
    src = S.Alpaca(live)
    chk = src.check()
    assert chk["ok"] and chk["stocks"] and chk["options"], chk
    d = cal.prev_trading_day(P.last_completed_session())
    out = src.stock_bars("SPY", d, d)
    assert not out.get("error"), out
    bars = live.read_bars("SPY", "1m", provider="alpaca")
    assert bars.height >= 380 and bars["ts"].n_unique() == bars.height  # a complete session, paginated to the end
    assert d in live.complete_sessions(live.ensure_dataset("alpaca", "SPY", "underlying", "trade", "1m"))
    px = float(bars["close"][-1])
    with S._client() as client:
        cs = src.list_contracts(client, "SPY", d, d, d + timedelta(days=7), px - 3, px + 3)
    assert cs and all(c.expiration >= d and c.root.startswith("SPY") for c in cs)
    # a contract that expired on `d` is inactive today but was tradable then: it must still be listed
    old = cal.next_trading_day(d, -5)
    with S._client() as client:
        past = src.list_contracts(client, "SPY", old, old, old, px * 0.9, px * 1.1)
    assert past, "contracts that have since expired were not returned for a past session"
    rep = src.acquire("SPY", {d: cs[:3]})
    assert rep["pairs_failed"] == 0, rep
    ob = live.read_option_bars("SPY", d, provider="alpaca")
    if ob is not None and not ob.is_empty():
        assert ob.filter(pl.col("volume") <= 0).is_empty() and ob.select(pl.struct("occ", "ts").n_unique()).item() == ob.height


# ----------------------------------------------------------------------------- language models
@pytest.mark.parametrize("provider,env", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_model_answers_and_reads_a_request_into_the_fixed_tools(live, provider, env):
    need_env(env)
    from strata.ai.agent import llm_interpret
    from strata.ai.llm import build, list_models
    from strata.api import db
    m = build(provider)
    assert m.json("You are a connectivity check.", [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}], max_tokens=200).get("ok") is True
    assert list_models(provider), "the provider returned no models for this account"
    r = llm_interpret(m, "How often does SPY rise 1% after a 3% down day?", None, [], live, lambda: [])
    assert r["action"] == "new"
    d = r["draft"]
    assert d.kind == "study" and d.symbol == "SPY" and d.study and d.study.target and abs(d.study.target.value - 1.0) < 1e-9
    assert any("-3" in c.expr.replace(" ", "") for c in d.conditions), [c.expr for c in d.conditions]
    usage = db.usage_summary(1)
    assert usage["calls"] >= 1 and usage["input_tokens"] > 0  # token usage is recorded per call
    # the key never appears in anything the app would show or log
    key = os.environ[env]
    assert key not in repr(m) and key not in str(usage)


# ----------------------------------------------------------------------------- V4: provider verification
def test_live_free_sources_are_verified_by_real_requests(tmp_path):
    """The live check of the keyless sources: what is recorded as 'verified' was actually fetched."""
    from strata.config import get_settings, override_settings
    from strata.data.providers import bench as PB
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    try:
        y = PB.benchmark_source("yahoo")
        assert y["status"] in ("verified", "partly verified") and y["bars_per_second"] > 0
        depth = {d["dataset"]: d for d in y["historical_depth"]}
        assert depth["daily bars"]["first"] < "2000-01-01" and "1-minute bars" in depth
        assert all(s.get("missing_pct") is not None for s in y["steps"] if s["ok"] and "minute" in s.get("dataset", ""))
        c = PB.benchmark_source("cboe")
        assert c["status"] == "verified" and c["steps"][0]["first"].startswith("1990")
        assert PB.saved()["yahoo"]["measured_at"]
    finally:
        override_settings(data_dir=old)


def test_live_alpaca_entitlements_are_measured_not_assumed(tmp_path):
    """Needs ALPACA_API_KEY / ALPACA_API_SECRET. Records which feeds and datasets this account can
    really read; without credentials the source must be reported as unverified."""
    import os

    from strata.config import get_settings, override_settings
    from strata.data.providers import bench as PB
    have = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET"))
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    try:
        r = PB.benchmark_source("alpaca")
        if not have:
            assert r["status"] == "unverified" and not r["verified"]
            pytest.skip("no Alpaca credentials: the integration stays UNVERIFIED")
        ent = {e["what"]: e["granted"] for e in r["entitlements"]}
        assert r["verified"] and ent["stock bars, IEX feed"] is True, r
        assert "option contract definitions, including expired contracts" in ent and "historical option trade bars (indicative feed)" in ent
        assert any(s["ok"] and s.get("first", "").startswith("2016") for s in r["steps"]) or any("depth" in s["name"] and not s["ok"] for s in r["steps"])
    finally:
        override_settings(data_dir=old)
