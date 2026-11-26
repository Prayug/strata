"""V7: readiness is the strategy's own, not the session's.

A session that has some option data is not a session the strategy can trade. Each case below stores option trade bars
that would have passed the old check ("trade prints exist for the session") and must not pass the new one, and each
verdict is compared with what the engine then does: the precheck may count fewer sessions than a run executes, never more.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data import planner as P
from strata.data import readiness as RD
from strata.data.providers import sources as S
from strata.data.store import MarketStore, SymbolMeta
from strata.engine.engine import run_backtest
from strata.engine.scan import scan_signals
from tests.helpers import et_epoch, write_minutes
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture: a temporary data directory)

DAYS = cal.trading_days(date(2024, 3, 4), date(2024, 3, 25))[:12]
PREV = cal.prev_trading_day(DAYS[0])
READY, PARTIAL, NO_EXP, GAP, STALE, NO_EXIT, NOTHING = DAYS[:7]
M_IN, M_MID, M_GONE = DAYS[8:11]  # a three-session position: entered, held, then no stored observation


def price(d: date, i: int) -> float:
    return 520.0 if d == GAP else 500.0  # GAP opens twenty points above the previous close and stays there


def bars(d: date, exp: date, strike: float, prints: list[tuple], right: str = "C") -> tuple[pl.DataFrame, dict]:
    """prints: (HH:MM bar start, price). Returns the bars and the index row for the contract."""
    occ = S.occ_symbol("SPY", exp, right, strike)
    rows = [{"ts": datetime.fromtimestamp(et_epoch(d, t), tz=timezone.utc), "occ": occ, "expiration": exp, "strike": strike, "right": right, "open": px, "high": px, "low": px, "close": px, "volume": 5} for t, px in prints]
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), {"date": d, "occ": occ, "expiration": exp, "strike": strike, "right": right, "bars": len(rows), "status": "ok", "pages": 1}


def put(st: MarketStore, d: date, contracts: list[tuple]):
    """contracts: (expiration, strike, prints)."""
    frames, index = [], []
    for exp, k, prints in contracts:
        b, row = bars(d, exp, k, prints)
        frames.append(b)
        index.append(row)
    st.write_option_bars("SPY", "testfeed", pl.concat(frames), index)


@pytest.fixture
def store(tmp_settings):
    st = MarketStore(tmp_settings / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", multiplier=100.0))
    write_minutes(st, "SPY", [PREV] + DAYS, price)
    whole = [("09:59", 2.0), ("10:30", 2.2), ("15:28", 2.6)]
    band = lambda d, prints: [(d, float(k), prints) for k in range(495, 506)]  # eleven strikes around the previous close
    put(st, READY, band(READY, whole))
    put(st, PARTIAL, band(PARTIAL, whole))
    put(st, NO_EXP, [(NO_EXP + timedelta(days=7), float(k), whole) for k in range(495, 506)])  # only next week's expiration is stored
    put(st, GAP, band(GAP, whole))  # the underlying trades at 520: the money is fifteen points above the stored band
    put(st, STALE, [(STALE, float(k), [("09:35", 2.0), ("14:00", 1.0)]) for k in range(495, 506)])  # nothing near 10:00
    put(st, NO_EXIT, band(NO_EXIT, [("09:59", 2.0), ("10:10", 2.1)]))  # nothing after 10:10
    key = st.option_dataset("SPY", "trade", "testfeed")
    st.mark_sessions(key, {PARTIAL.isoformat(): {"status": "partial", "reason": "the provider's option history for this session is incomplete: its first trade print of any near-dated contract is at 12:55 New York time"}})
    week = M_IN + timedelta(days=7)
    put(st, M_IN, [(week, 500.0, whole)])
    put(st, M_MID, [(week, 500.0, whole)])  # M_GONE: nothing stored for the contract
    return st


def strategy(dte: int = 0, exits: dict | None = None, **leg) -> dict:
    return {"name": "call at 10:00", "symbol": "SPY",
            "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": dte, **({"dte_mode": "exact"} if dte == 0 else {}), "strike": {"method": "atm_offset", "value": 0, "unit": "points"}, **leg}],
            "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}},
            "exits": exits or {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:30", "dte_lte": None}}}


def config(a: date, b: date, **kw) -> dict:
    return {"start": a.isoformat(), "end": b.isoformat(), "strategies": [strategy(**kw)], "pricing": {"mode": "trades"}, "data": {"options_provider": "testfeed"},
            "execution": {"signal_interval_minutes": 1, "commission_per_contract": 0.0}}


def test_each_signal_is_checked_against_the_contract_it_needs_and_the_run_agrees(store):
    cfg = config(DAYS[0], DAYS[6])
    signals = scan_signals(cfg, store)["signals"]
    assert [s["date"] for s in signals] == DAYS[:7]
    rd = P.signal_readiness(store, cfg, signals, "testfeed")
    got = {r["date"]: r for r in rd["rows"]}
    status = {d: got[d.isoformat()]["status"] for d in DAYS[:7]}
    assert status == {READY: "trades", PARTIAL: "partial", NO_EXP: "expiration", GAP: "contract", STALE: "stale", NO_EXIT: "exit_uncertain", NOTHING: "missing"}
    # every one of the first six sessions HAS option trade data stored: the old check called all of them priceable
    idx_days = set(store.option_bar_index("SPY", provider="testfeed")["date"].to_list())
    assert all(d in idx_days for d in DAYS[:6])
    assert rd["ready"] == 1 and rd["entry_ready"] == 2 and rd["counts"]["trades"] == 1
    # what each verdict says
    ok = got[READY.isoformat()]
    assert ok["entry"] == {"time": "10:00", "contracts": [f"500C {READY.isoformat()}"], "print_age_seconds": 0} and ok["exit"]["planned"] == "15:30"
    assert "12:55" in got[PARTIAL.isoformat()]["detail"]
    assert "no exact 0DTE call expiration is stored for the session (stored: 7DTE)" in got[NO_EXP.isoformat()]["detail"]
    assert "the underlying was at 520.00 at the signal; the strikes stored for the session run from 495 to 505" in got[GAP.isoformat()]["detail"]
    assert "no usable trade print between 10:00 and 10:05" in got[STALE.isoformat()]["detail"] and got[STALE.isoformat()]["window"] == ["10:00", "10:05"]
    assert "has no print within 5 min before the planned exit at 15:30 or after it" in got[NO_EXIT.isoformat()]["detail"] and got[NO_EXIT.isoformat()]["entry"]["time"] == "10:00"
    # the engine, on the same configuration: it executes exactly the sessions whose entry was found, and no other
    res = run_backtest(cfg, store=store, record_paths=False)
    executed = sorted(t["entry_date"] for t in res.trades)
    assert executed == sorted(d.isoformat() for d in (READY, NO_EXIT))
    assert len(executed) == rd["entry_ready"] and len(executed) >= rd["ready"], "the check never promises more than a run executes"
    by = {t["entry_date"]: t for t in res.trades}
    assert by[READY.isoformat()]["exit_reason"] == "time_exit" and "deferred_exit" not in by[READY.isoformat()]["flags"]
    assert "deferred_exit" in by[NO_EXIT.isoformat()]["flags"], "the session flagged for its exit is the one whose exit could not be priced when planned"
    why = res.meta["executability"]["not_executed_by_reason"]
    assert why == {"no_option_data": 2, "contract_not_stored": 2, "no_recent_trade": 1}  # partial + nothing stored; missing expiration + gap; stale


def test_a_stale_print_is_not_an_entry_observation(store):
    """The contract traded at 09:35 and at 14:00. An order at 10:00 waits five minutes and sees no print younger than five
    minutes: the 09:35 print is not a fill, and neither is the 14:00 one."""
    cfg = config(STALE, STALE)
    sig = scan_signals(cfg, store)["signals"]
    assert P.signal_readiness(store, cfg, sig, "testfeed")["rows"][0]["status"] == "stale"
    # with a longer wait the order is still working when nothing has printed: still stale; a print inside the window changes it
    cfg["execution"]["max_fill_delay_minutes"] = 30
    assert P.signal_readiness(store, cfg, sig, "testfeed")["rows"][0]["status"] == "stale"
    put(store, STALE, [(STALE, 500.0, [("10:12", 2.4), ("15:29", 2.0)])])
    r = P.signal_readiness(store, cfg, sig, "testfeed")["rows"][0]
    assert r["status"] == "trades" and r["entry"]["time"] == "10:13", "filled once the 10:12 bar has ended, never before the print existed"
    assert run_backtest(cfg, store=store, record_paths=False).trades[0]["entry_time"][11:16] == "10:13"
    # a print made BEFORE the order could use it does not become a later fill: with latency the 09:59 print of another day is too old
    late = config(READY, READY)
    late["execution"].update(latency_seconds=420, max_fill_delay_minutes=10)
    sg = scan_signals(late, store)["signals"]
    r2 = P.signal_readiness(store, late, sg, "testfeed")["rows"][0]
    assert r2["status"] == "stale", "ready at 10:07; the 09:59 print is eight minutes old by then and nothing printed until 10:30"
    assert run_backtest(late, store=store, record_paths=False).trades == []


def test_a_position_held_over_several_sessions_needs_the_contract_on_each_of_them(store):
    exits = {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": False}, "max_hold": {"enabled": True, "days": 2}}
    cfg = config(M_IN, M_IN, dte=7, exits=exits)
    sig = scan_signals(cfg, store)["signals"]
    assert [s["date"] for s in sig] == [M_IN]
    rd = P.signal_readiness(store, cfg, sig, "testfeed")
    r = rd["rows"][0]
    assert rd["settings"]["hold_sessions"] == 2
    assert r["status"] == "exit_uncertain" and r["missing_later"] == [M_GONE.isoformat()], "stored on the entry session and the next; not on the third"
    assert f"no stored observations for the position on {M_GONE.isoformat()}" in r["detail"] and r["entry"]["contracts"] == [f"500C {(M_IN + timedelta(days=7)).isoformat()}"]
    # stored on every session of the holding period: nothing to flag
    put(store, M_GONE, [(M_IN + timedelta(days=7), 500.0, [("09:59", 2.0), ("15:28", 2.6)])])
    assert P.signal_readiness(store, cfg, sig, "testfeed")["rows"][0]["status"] == "trades"


def test_the_precheck_reports_these_distinctions_and_what_can_be_fetched(store, monkeypatch):
    cfg = config(DAYS[0], DAYS[6])
    pc = RD.precheck(cfg, store)
    assert pc["signal_sessions"] == 7 and pc["priced"] == 1 and pc["potential"] == 2 and pc["fetchable"] == 0
    assert pc["counts"] == {"quotes": 0, "trades": 1, "eod": 0, "exit_uncertain": 1, "stale": 1, "contract": 1, "expiration": 1, "partial": 1, "no_contract": 0, "missing": 1}
    assert [(s["date"], s["status"]) for s in pc["sessions"]] == [(d.isoformat(), x) for d, x in zip(DAYS[:7], ("trades", "partial", "expiration", "contract", "stale", "exit_uncertain", "missing"))]
    t = pc["text"]
    assert "fires on 7 sessions" in t and "1 pass every check on stored option observations" in t.replace("passes", "pass") and "14.3%" in t
    for phrase in ("1 more has an entry that can be priced but no observation near the planned exit", "1 has the contract stored but no trade inside the entry window", "1 needs a strike outside the stored range",
                   "1 needs an expiration that is not stored", "1 is held only in part by the provider", "1 has no stored option prices", "A run confirms each one"):
        assert phrase in t, phrase
    assert pc["untested"] == [d.isoformat() for d in DAYS[1:7]] and pc["executable"] is True
    sub = next(a for a in pc["alternatives"] if a["id"] == "subset")
    assert sub["sessions"] == [READY.isoformat(), NO_EXIT.isoformat()] and "download" not in [a["id"] for a in pc["alternatives"]]
    assert pc["settings"] == {"max_trade_age_seconds": 300, "max_fill_delay_minutes": 5.0, "signal_interval_minutes": 1, "latency_seconds": 0.0, "hold_sessions": 0}

    # a connected free source whose history covers these sessions: what is missing can be fetched; what never traded cannot
    class Feed(S.OptionBarSource):
        name, label, tier = "testfeed", "Test feed", "free_key"

        def available(self):
            return True, "key configured"

        def bars_window(self, today=None):
            return date(2024, 1, 1), date(2026, 12, 31)

    monkeypatch.setattr(S, "option_bar_sources", lambda store=None: [Feed(store)])
    pc2 = RD.precheck(cfg, store)
    fetch = {s["date"]: (s["downloadable"] or {}).get("source") for s in pc2["sessions"]}
    assert fetch == {READY.isoformat(): None, PARTIAL.isoformat(): None, NO_EXP.isoformat(): "testfeed", GAP.isoformat(): "testfeed", STALE.isoformat(): None, NO_EXIT.isoformat(): None, NOTHING.isoformat(): "testfeed"}
    assert pc2["fetchable"] == 3 and "3 of the unpriced sessions can be downloaded from Test feed at no cost" in pc2["text"]
    dl = pc2["alternatives"][0]
    assert dl["id"] == "download" and dl["sessions"] == [NO_EXP.isoformat(), GAP.isoformat(), NOTHING.isoformat()] and "Only the contracts this strategy could select" in dl["detail"]


def test_the_coverage_the_pipeline_uses_is_the_same_determination(store):
    cfg = config(DAYS[0], DAYS[6])
    signals = scan_signals(cfg, store)["signals"]
    req = P.requirements("options", cfg, "trades")
    cov = P.trade_coverage(store, req, signals, "testfeed", cfg=cfg)
    assert cov["executable"] == 2 and cov["fully_covered"] == 1 and cov["exit_uncertain"] == 1 and cov["signals"] == 7
    assert cov["readiness"]["stale"] == 1 and cov["readiness"]["contract"] == 1 and cov["readiness"]["expiration"] == 1 and cov["readiness"]["partial"] == 1
    reasons = {i["reason"]: i["signals"] for i in cov["issues"]}
    assert reasons == {P.STATUS_TEXT[k]: 1 for k in ("partial", "expiration", "contract", "stale", "missing")}
    # the older, coarser check (no configuration) counted the stale session and the uncovered exit as executable
    old = P.trade_coverage(store, req, signals, "testfeed")
    assert old["executable"] >= cov["executable"]


def test_the_provider_is_asked_only_about_the_signals_the_stored_observations_do_not_price(store):
    """A strategy whose observations are all on disk needs nothing from anyone. Before this the contracts of every signal
    were looked up at the provider first: minutes of waiting for a run that takes seconds."""
    cfg = config(DAYS[0], DAYS[6])
    signals = scan_signals(cfg, store)["signals"]
    todo = P.unpriced_signals(store, cfg, signals, "testfeed")
    assert [s["date"] for s in todo] == [PARTIAL, NO_EXP, GAP, STALE, NO_EXIT, NOTHING] and READY not in [s["date"] for s in todo]
    one = [s for s in signals if s["date"] == READY]
    assert P.unpriced_signals(store, cfg, one, "testfeed") == []
