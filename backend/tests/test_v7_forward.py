"""V7: prospective paper validation is incremental.

V6 sealed each session once, but produced every evaluation by replaying the whole window since activation: positions and
cash were whatever the replay gave that day. Here the account is a stored thing. Each new session is run from the state
stored after the last one and from nothing else; a sealed session is never simulated again to produce the record; and stored
data that is revised later is compared, written down as a revision, and changes no decision.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from strata.core import calendar as cal
from tests.helpers import make_store, put_spread_strategy, write_minutes, write_options
from tests.test_v2_research import tmp_settings  # noqa: F401
from tests.test_v6_forward import D, clock, quotes  # noqa: F401


def experiment(tmp, clock):
    """Sell the 0.20-delta put expiring tomorrow and buy the put 5 lower at 10:00 (as in the V6 test): open overnight."""
    from strata.research import forward as FW
    st = make_store(tmp)
    write_minutes(st, "TST", [date(2024, 3, 1)], lambda d, i: 100.0)
    strat = put_spread_strategy(legs=[{"id": "sp", "side": "sell", "right": "P", "dte": 1, "strike": {"method": "delta", "value": 0.20}},
                                      {"id": "lp", "side": "buy", "right": "P", "dte": 1, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}}])
    clock.set(date(2024, 3, 1))
    ex = FW.create("strategy", "TST", {"start": "2024-01-02", "end": "2024-03-01", "strategies": [strat]}, "overnight spread", assumptions={"execution": {"fill_model": "conservative"}}, store=st)
    return FW, st, ex


def session(st, clock, i: int, options: bool = True):
    write_minutes(st, "TST", [D[i]], lambda d, j: 100.0)
    if options:
        write_options(st, "TST", D[i], quotes(D[i], cal.next_trading_day(D[i])))
    clock.set(D[i])


def spy(monkeypatch, FW) -> list:
    """Every window the engine is asked to run, with the number of open positions it starts from."""
    calls, real = [], FW._strategy_events

    def wrapped(ex, window, store, resume=None, incremental=False):
        calls.append(([d.isoformat() for d in window], None if resume is None else len(resume.get("positions") or []), incremental))
        return real(ex, window, store, resume, incremental)
    monkeypatch.setattr(FW, "_strategy_events", wrapped)
    return calls


def states(eid: str) -> list[dict]:
    from strata.api import db
    with db.connect() as con:
        return [dict(r) for r in con.execute("SELECT session, state, state_hash, engine_version, data_fingerprint, origin FROM forward_states WHERE experiment_id=? ORDER BY session", (eid,))]


# ----------------------------------------------------------------------------- only what is new is run
def test_each_evaluation_runs_only_the_new_session_from_the_stored_account(tmp_settings, clock, monkeypatch):
    FW, st, ex = experiment(tmp_settings, clock)
    calls = spy(monkeypatch, FW)
    session(st, clock, 0)
    r1 = FW.evaluate(ex["id"], st)
    assert calls == [(["2024-03-04"], None, True)], "the first session starts from an empty account"
    assert r1["newly_sealed"] == ["2024-03-04"] and r1["method"].startswith("incremental") and r1["account"]["open_positions"] == 1 and r1["account"]["through"] == "2024-03-04"
    s1 = states(ex["id"])
    assert [(x["session"], x["origin"]) for x in s1] == [("2024-03-04", "sealed")] and s1[0]["data_fingerprint"] and len(s1[0]["state_hash"]) == 64
    held = json.loads(s1[0]["state"])
    assert len(held["positions"]) == 1 and [l["strike"] for l in held["positions"][0]["legs"]] == [95, 90] and held["cash"] == pytest.approx(100_000 + 50 - 1.30)
    # the next session: the engine is given that session and the stored state, and nothing before it
    session(st, clock, 1)
    r2 = FW.evaluate(ex["id"], st)
    assert calls[1:] == [(["2024-03-05"], 1, True)], "one session, starting from the one open position that was stored"
    assert r2["newly_sealed"] == ["2024-03-05"] and r2["open_positions"] == [] and r2["metrics"]["net_pnl"] == pytest.approx(48.70) and r2["account"]["open_positions"] == 0
    assert r2["reconciliation"] == {"checked": 2, "changed": [], "revisions_new": 0}
    # asked again with nothing new: no session is run, the record is read back as it is
    n = len(calls)
    r3 = FW.evaluate(ex["id"], st)
    assert len(calls) == n and r3["newly_sealed"] == [] and r3["metrics"]["net_pnl"] == pytest.approx(48.70) and len(r3["trades"]) == 1
    # every sealed session names the state it started from and the observations it read
    seal = [e for e in FW.ledger(ex["id"]) if e["kind"] == "session"]
    assert all(e["body"]["from_stored_state"] and e["body"]["data_fingerprint"] for e in seal) and [e["session"] for e in seal] == ["2024-03-04", "2024-03-05"]
    assert FW.verify(ex["id"])["ok"]


def test_a_stored_account_state_cannot_be_changed_or_removed(tmp_settings, clock):
    from strata.api import db
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    FW.evaluate(ex["id"], st)
    with db.connect() as con, pytest.raises(sqlite3.DatabaseError, match="cannot be changed"):
        con.execute("UPDATE forward_states SET state='{}' WHERE experiment_id=?", (ex["id"],))
    with db.connect() as con, pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        con.execute("DELETE FROM forward_states WHERE experiment_id=?", (ex["id"],))
    assert len(states(ex["id"])) == 1


def test_a_session_without_option_observations_is_still_sealed_and_the_account_carried(tmp_settings, clock):
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    FW.evaluate(ex["id"], st)
    session(st, clock, 1)
    FW.evaluate(ex["id"], st)
    session(st, clock, 2, options=False)  # prices, no option chain
    r = FW.evaluate(ex["id"], st)
    assert r["newly_sealed"] == ["2024-03-06"] and r["not_executed_by_reason"] == {"no_option_data": 1}, "run on its own, a session with no chain is a session with a signal that could not be priced"
    assert [x["session"] for x in states(ex["id"])] == ["2024-03-04", "2024-03-05", "2024-03-06"] and r["account"]["cash"] == pytest.approx(100_048.70)


# ----------------------------------------------------------------------------- revised data: recorded, never applied
def test_data_revised_after_a_seal_becomes_a_revision_and_changes_no_decision(tmp_settings, clock, monkeypatch):
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    FW.evaluate(ex["id"], st)
    session(st, clock, 1)
    FW.evaluate(ex["id"], st)
    before_ledger = [e for e in FW.ledger(ex["id"]) if e["kind"] in ("fill", "exit", "session")]
    before_states = states(ex["id"])
    assert FW.reconcile(ex["id"], st)["changed"] == [], "nothing was revised: nothing is replayed"
    # the first session's quotes are rewritten: the short put would now have been sold at 2.00, not 1.00
    q = quotes(D[0], D[1])
    for row in q:
        if row["strike"] == 95 and row["expiration"] == D[1]:
            row["bid"], row["ask"] = 2.00, 2.10
    write_options(st, "TST", D[0], q)
    calls = spy(monkeypatch, FW)
    rec = FW.reconcile(ex["id"], st)
    assert rec["changed"] == ["2024-03-04"] and rec["revisions_new"] >= 1 and "The sealed record and every later decision are unchanged" in rec["text"]
    assert calls == [(["2024-03-04"], None, True)], "only the revised session is simulated again, from its own starting state"
    rev = [e for e in FW.ledger(ex["id"]) if e["kind"] == "revision"]
    assert rev and rev[0]["body"]["what"] == "changed" and rev[0]["body"]["entry"] == "fill" and rev[0]["body"]["sealed"]["entry_cash"] == 50 and rev[0]["body"]["replay"]["entry_cash"] == 150
    # what was sealed is exactly what it was, and so is every stored state
    assert [e for e in FW.ledger(ex["id"]) if e["kind"] in ("fill", "exit", "session")] == before_ledger and states(ex["id"]) == before_states
    # a second pass writes nothing new, and the next session continues from the SEALED account (48.70), not the revised one
    assert FW.reconcile(ex["id"], st)["revisions_new"] == 0
    session(st, clock, 2)
    r = FW.evaluate(ex["id"], st)
    assert r["newly_sealed"] == ["2024-03-06"] and r["metrics"]["net_pnl"] == pytest.approx(48.70) and json.loads(states(ex["id"])[-1]["state"])["realised"] == pytest.approx(48.70)
    assert any("revision" in n for n in r["notes"]) and FW.verify(ex["id"])["ok"]


def test_an_evaluation_reports_revised_sessions_by_itself(tmp_settings, clock):
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    FW.evaluate(ex["id"], st)
    q = quotes(D[0], D[1])
    for row in q:
        if row["strike"] == 95 and row["expiration"] == D[1]:
            row["bid"], row["ask"] = 2.00, 2.10
    write_options(st, "TST", D[0], q)
    session(st, clock, 1)
    r = FW.evaluate(ex["id"], st)
    assert r["reconciliation"]["changed"] == ["2024-03-04"] and r["reconciliation"]["revisions_new"] >= 1 and r["revisions_new"] >= 1
    assert r["metrics"]["net_pnl"] == pytest.approx(48.70), "the session that followed was run from the sealed account"


# ----------------------------------------------------------------------------- versions
def test_an_engine_change_is_written_down_before_a_session_is_sealed_under_it(tmp_settings, clock, monkeypatch):
    import strata.config as C
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    FW.evaluate(ex["id"], st)
    frozen = ex["engine_version"]
    monkeypatch.setattr(C, "ENGINE_VERSION", "99.0.0")
    session(st, clock, 1)
    r = FW.evaluate(ex["id"], st)
    led = FW.ledger(ex["id"])
    ch = [e for e in led if e["kind"] == "engine_change"]
    assert len(ch) == 1 and ch[0]["body"]["frozen_under"] == frozen and ch[0]["body"]["now"] == "99.0.0" and ch[0]["body"]["from_session"] == "2024-03-05"
    seals = [e for e in led if e["kind"] == "session"]
    assert ch[0]["seq"] < seals[-1]["seq"] and seals[0]["body"]["engine_version"] == frozen and seals[-1]["body"]["engine_version"] == "99.0.0"
    assert states(ex["id"])[-1]["engine_version"] == "99.0.0" and any("Frozen under engine" in n for n in r["notes"])
    session(st, clock, 2)
    FW.evaluate(ex["id"], st)
    assert len([e for e in FW.ledger(ex["id"]) if e["kind"] == "engine_change"]) == 1, "written once per engine, not once per session"


def test_a_record_sealed_before_states_were_stored_has_its_account_rebuilt_once(tmp_settings, clock, monkeypatch):
    """Version 6 sealed sessions without storing the account. The first V7 evaluation rebuilds it by one run over the sealed
    sessions, says so in the ledger, and continues incrementally."""
    FW, st, ex = experiment(tmp_settings, clock)
    session(st, clock, 0)
    monkeypatch.setattr(FW, "_save_state", lambda *a, **k: "")  # as V6 did: sealed, no state stored
    FW.evaluate(ex["id"], st)
    monkeypatch.undo()
    clock_ = clock
    from strata.research import forward as FW2
    monkeypatch.setattr(FW2, "_now", clock_)
    assert states(ex["id"]) == []
    calls = spy(monkeypatch, FW2)
    session(st, clock_, 1)
    r = FW2.evaluate(ex["id"], st)
    assert calls == [(["2024-03-04"], None, True), (["2024-03-05"], 1, True)], "one run through the sealed session to rebuild the account, then the new session from it"
    assert [(x["session"], x["origin"]) for x in states(ex["id"])] == [("2024-03-04", "rebuilt"), ("2024-03-05", "sealed")]
    rb = [e for e in FW2.ledger(ex["id"]) if e["kind"] == "state_rebuilt"]
    assert len(rb) == 1 and rb[0]["body"]["open_positions"] == 1 and "were not changed" in rb[0]["body"]["note"]
    assert r["newly_sealed"] == ["2024-03-05"] and r["metrics"]["net_pnl"] == pytest.approx(48.70) and any("rebuilt once" in n for n in r["notes"])


# ----------------------------------------------------------------------------- the property the whole thing rests on
def test_a_run_continued_session_by_session_is_the_run_made_in_one_piece(tmp_settings, clock):
    """The engine's saved state is complete: positions, cash, what a leg remembers about how it was priced."""
    from strata.engine.engine import run_backtest
    FW, st, ex = experiment(tmp_settings, clock)
    for i in range(4):
        session(st, clock, i)
    cfg = lambda a, b: {**ex["config"], "start": a.isoformat(), "end": b.isoformat()}
    strip = lambda t: json.dumps({k: v for k, v in t.items() if k != "id"}, sort_keys=True, default=str)
    whole = run_backtest(cfg(D[0], D[3]), store=st, record_paths=False, keep_open=True)
    state, got = None, []
    for d in D:
        res = run_backtest(cfg(d, d), store=st, record_paths=False, keep_open=True, resume=state)
        state = json.loads(json.dumps(res.meta["state"]))  # through JSON, as it is stored
        got += res.trades
    # one position at a time: opened on the first session and expired on the second, again on the third and the fourth
    assert sorted(map(strip, got)) == sorted(map(strip, whole.trades)) and [t["entry_date"] for t in got] == ["2024-03-04", "2024-03-06"]
    assert state["cash"] == pytest.approx(whole.meta["state"]["cash"]) and len(state["positions"]) == len(whole.meta["state"]["positions"]) == 0
    assert state["realised"] == pytest.approx(2 * 48.70) and state["closed"] == 2 and state["cash"] == pytest.approx(100_000 + 2 * 48.70)
    # a position that was open when the state was saved keeps what its legs remember: here, that they were priced from quotes
    mid = run_backtest(cfg(D[0], D[0]), store=st, record_paths=False, keep_open=True)
    held = json.loads(json.dumps(mid.meta["state"]))
    assert len(held["positions"]) == 1 and [l["price_source"] for l in mid.meta["open_positions"][0]["legs"]] == ["quote", "quote"]
    nxt = run_backtest(cfg(D[1], D[1]), store=st, record_paths=False, keep_open=True, resume=held)
    assert [l["price_source"] for l in nxt.trades[0]["legs"]] == ["quote", "quote"] and nxt.trades[0]["pnl"] == pytest.approx(48.70) and nxt.trades[0]["entry_date"] == "2024-03-04"
    with pytest.raises(ValueError, match="format this engine does not read"):
        run_backtest(cfg(D[1], D[1]), store=st, record_paths=False, resume={**state, "version": 2})
