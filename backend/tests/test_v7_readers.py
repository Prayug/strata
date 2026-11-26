"""V7: what closes a position is an exit, and never also a condition of the entry.

Found by typing the first sentence below into the running application: "exit ... when SPY has risen 1%" was read as the
underlying target AND as an entry condition, so the strategy asked for a day that had fallen 1% and risen 1%, and never
traded. The readers now treat a clause inside an exit statement as an exit: a move of the underlying there is the target
or the stop, an event there is an exit condition, and what cannot be represented stays an exit that could not be
represented. None of it is offered to any reader as a condition of the entry."""
from __future__ import annotations

import pytest

from tests.test_v2_research import InlineJobs, tmp_settings  # noqa: F401
from tests.test_v7_experiment import grid_store


def enrich(text):
    from strata.ai import nl_rules as NL
    from strata.ai.nl_stateful import enrich as E
    return E(text, NL.TICKERS)


def reading(d) -> dict:
    return {"entry": d.entry_time, "when": [c.expr for c in d.conditions], "then": [s.expr for s in d.sequence], "exit": d.exit_time, "target": d.underlying_target_pct, "stop": d.underlying_stop_pct,
            "premium_target": d.premium_target_pct, "exit_when": [c.expr for c in d.exit_conditions], "unsupported": d.unsupported}


@pytest.mark.parametrize("text", [
    "Buy ATM SPY 0DTE calls at the open after a 1% down day, exit at 3:55 PM Eastern or when SPY has risen 1%",
    "Buy ATM SPY 0DTE calls at the open after a 1% down day and sell at 3:55 PM Eastern or once SPY is up 1% from the entry",
    "Buy ATM SPY 0DTE calls at the open after a 1% down day; take profit when SPY rises 1%, otherwise sell at 3:55 PM Eastern",
    "Buy SPY 0DTE calls at the open after a 1% down day. Close the trade when SPY has gained 1% or at 15:55 ET.",
    "Buy ATM SPY 0DTE calls at the open after a 1% down day, exit at 3:55 PM Eastern or when SPY has risen 1%, from 2024-01-18 to 2026-11-25",
])
def test_a_move_of_the_underlying_in_an_exit_clause_is_the_target_and_not_an_entry_condition(tmp_settings, text):
    r = reading(enrich(text))
    assert r["when"] == ["previous_session.return_pct <= -1"] and r["then"] == [] and r["entry"] == "open"
    assert r["target"] == 1.0 and r["exit"] == "15:55" and r["exit_when"] == [] and r["unsupported"] == []


def test_the_period_of_the_test_is_not_part_of_the_exit(tmp_settings):
    from datetime import date
    d = enrich("Buy ATM SPY 0DTE calls at the open after a 1% down day, exit at 3:55 PM Eastern or when SPY has risen 1%, from 2024-01-18 to 2026-11-25")
    assert (d.start, d.end) == (date(2024, 1, 18), date(2026, 11, 25)) and d.unsupported == []
    from strata.ai import nl_rules as NL
    t = "exit when SPY has risen 0.75% in 2025"
    assert [t[a:b] for a, b in NL.exit_clauses(t)] == ["when SPY has risen 0.75%"]  # a decimal point does not end the clause, the year does


def test_an_event_in_an_exit_clause_is_an_exit_condition(tmp_settings):
    r = reading(enrich("Buy SPY 0DTE calls at 10:00 AM Eastern and sell when price crosses below VWAP or at the close"))
    assert r["when"] == [] and r["exit_when"] == ["crosses_below(price, vwap)"] and r["exit"] == "close" and r["entry"] == "10:00"
    r = reading(enrich("Buy SPY 0DTE calls at the open, exit when SPY breaks below the low of the first 15 minutes"))
    assert r["when"] == [] and r["exit_when"] == ["price < opening_range_low(15)"] and r["unsupported"] == []
    # the same words before the exit verb are still the entry
    r = reading(enrich("Buy SPY 0DTE calls when SPY breaks above the high of the first 15 minutes, exit at 3 PM Eastern"))
    assert r["when"] == ["price > opening_range_high(15)"] and r["exit_when"] == [] and r["exit"] == "15:00"
    r = reading(enrich("Buy SPY 0DTE calls when SPY has risen 1% from the open, exit at 3:55 PM Eastern"))
    assert r["when"] == ["from_open_pct >= 1"] and r["target"] is None


def test_a_verb_that_opens_the_position_and_a_condition_known_before_the_entry_are_not_exits(tmp_settings):
    from strata.ai import nl_rules as NL
    assert NL.exit_clauses("Sell SPY 0DTE puts 1% out of the money when SPY is up 0.5% from the open; close at 3:30 PM Eastern") == []
    r = reading(enrich("Sell SPY 0DTE puts 1% out of the money when SPY is up 0.5% from the open; close at 3:30 PM Eastern"))
    assert r["when"] == ["from_open_pct >= 0.5"] and r["exit"] == "15:30"
    # "when the previous day fell 2%" cannot close anything: it was known before the entry
    r = reading(enrich("Buy SPY 0DTE calls at the open and exit at the close when the previous day fell 2%"))
    assert r["when"] == ["previous_session.return_pct <= -2"] and r["exit"] == "close" and r["exit_when"] == []
    # "them" makes the sale an exit even though the option is named
    t = "Buy SPY calls at the open. Sell them if SPY drops 0.7% from the entry or gains 1.5%."
    assert [t[a:b] for a, b in NL.exit_clauses(t)] == ["if SPY drops 0.7% from the entry or gains 1.5%"]


def test_an_exit_that_cannot_be_represented_is_said_and_is_never_turned_into_an_entry_rule(tmp_settings):
    d = enrich("Buy SPY 0DTE calls at the open and exit when the moon is full or at the close")
    assert d.conditions == [] and d.exit_conditions == [] and d.exit_time == "close" and d.unsupported == ["when the moon is full or at the close"]
    # two sizes, one of which was not placed: the clause is flagged rather than half-read
    d = enrich("Buy SPY calls at the open. Sell them if SPY drops 0.7% from the entry or gains 1.5%.")
    assert d.conditions == [] and any("gains 1.5%" in u for u in d.unsupported)


def test_through_the_application_the_first_sentence_trades(tmp_settings, monkeypatch):
    """The sentence as typed into the interface: it used to stop with "the entry rules never fired"."""
    from strata.ai.agent import Agent
    from strata.api import db
    st = grid_store(tmp_settings)
    ag, sid = Agent(InlineJobs(), st, llm_factory=lambda: None), db.create_session("t")["id"]
    plan = ag.post(sid, "Buy ATM SPY 0DTE calls at 10:00 AM Eastern, exit at 3:30 PM Eastern or when SPY has risen 1%")[-1]["content"]
    assert plan["draft"]["underlying_target_pct"] == 1.0 and plan["draft"]["conditions"] == [] and plan["state"] == "running" and (plan.get("unsupported") or []) == []


def test_a_partial_exit_a_premium_stop_and_a_clock_exit_in_one_clause_are_all_represented(tmp_settings):
    """The acceptance sentence: the reading was right and the new exit check called it "not represented" until the levels of
    a partial exit counted as exit levels."""
    d = enrich("Buy two ATM QQQ 0DTE calls when price reclaims VWAP after declining 0.5% from the open, sell half when the premium is up 25%, with a 15% premium stop, exit at 3:30 PM Eastern, from 2024-01-18 to 2026-11-25")
    assert [s.expr for s in d.sequence] == ["from_open_pct <= -0.5", "crosses_above(price, vwap)"] and d.quantity == 2
    assert [(p.premium_pct, p.fraction) for p in d.partials] == [(25.0, 0.5)] and d.premium_stop_pct == 15.0 and d.exit_time == "15:30" and d.unsupported == []


def test_an_entry_that_waits_for_an_event_is_not_taken_at_or_after_the_time_everything_is_closed(tmp_settings):
    """Measured on stored data: a VWAP-reclaim strategy with "exit at 3:30 PM" opened 9 of its 263 positions after 3:30 and
    carried two of them into expiration; those 9 lost more than the other 254 made. The reading now says, as an assumption
    that can be edited, that nothing is opened from the exit time on."""
    from datetime import datetime, timezone

    import polars as pl

    from strata.ai.draft import Draft, DraftCondition, compile_draft
    from strata.data.providers import sources as S
    from strata.engine.engine import run_backtest
    from tests.helpers import et_epoch, write_minutes
    from tests.test_v7_experiment import DAYS, PREV
    d = enrich("Buy SPY 0DTE calls when price reclaims VWAP after declining 0.5% from the open, exit at 3:30 PM Eastern")
    c = compile_draft(d)
    kids = c["strategy"]["entry"]["conditions"]["children"]
    assert [(k["type"], k["params"]) for k in kids if k.get("id") == "lw"] == [("time_window", {"start": "09:30", "end": "15:29"})]
    assert any(a["field"] == "last_entry" and "15:30" in a["reason"] for a in c["assumptions"]) and any(i["label"] == "Last entry" and i["status"] == "assumed" for i in c["items"])
    # the five minutes before the bell that a same-day position is closed in by default are covered the same way
    kids = compile_draft(enrich("Buy SPY 0DTE calls when price crosses above VWAP"))["strategy"]["entry"]["conditions"]["children"]
    assert [k["params"]["end"] for k in kids if k.get("id") == "lw"] == ["15:54"]
    # an entry with a clock of its own keeps it: nothing is added
    kids = compile_draft(enrich("Buy SPY 0DTE calls at 10:00 AM Eastern, exit at 3:30 PM Eastern"))["strategy"]["entry"]["conditions"]["children"]
    assert not [k for k in kids if k.get("id") == "lw"]

    # in the engine: the event at 11:00 is traded, the same event at 15:45 is not
    st = grid_store(tmp_settings)
    for n, d_ in enumerate(DAYS[:2]):  # prints late in the session too, so that only the rule decides
        occ = S.occ_symbol("SPY", d_, "C", 502.0)
        rows = [{"ts": datetime.fromtimestamp(et_epoch(d_, t), tz=timezone.utc), "occ": occ, "expiration": d_, "strike": 502.0, "right": "C", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 5} for t in ("10:59", "15:29", "15:44", "15:58")]
        st.write_option_bars("SPY", "testfeed", pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))), [{"date": d_, "occ": occ, "expiration": d_, "strike": 502.0, "right": "C", "bars": 4, "status": "ok", "pages": 1}])
    write_minutes(st, "SPY", [PREV] + DAYS, lambda d_, i: 500.0 + (2.0 if (d_ == DAYS[0] and i >= 90) or (d_ == DAYS[1] and i >= 375) else 0.0))
    draft = Draft(kind="options", symbol="SPY", structure="long_call", dte=0, conditions=[DraftCondition(expr="crosses_above(price, 501)", text="price crosses above 501")], exit_time="15:30", start=DAYS[0], end=DAYS[1])
    strat = compile_draft(draft, st)["strategy"]
    cfg = {"start": DAYS[0].isoformat(), "end": DAYS[1].isoformat(), "pricing": {"mode": "trades"}, "data": {"options_provider": "testfeed"}, "execution": {"signal_interval_minutes": 1}, "strategies": [strat]}
    trades = run_backtest(cfg, store=st, record_paths=False).trades
    assert [(t["entry_date"], t["entry_time"][11:16], t["exit_reason"]) for t in trades] == [(DAYS[0].isoformat(), "11:01", "time_exit")]
    # without the window the second session's event is taken at 15:46, after the exit time, and carried to the bell
    strat["entry"]["conditions"]["children"] = [k for k in strat["entry"]["conditions"]["children"] if k.get("id") != "lw"]
    late = run_backtest({**cfg, "strategies": [strat]}, store=st, record_paths=False).trades
    assert len(late) == 2 and late[1]["entry_date"] == DAYS[1].isoformat() and late[1]["entry_time"][11:13] == "15"
