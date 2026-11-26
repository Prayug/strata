"""V6: the integrity of forward validation.

The V5 rule dated an experiment by the last session STORED when it was frozen. With stale
local data that date was too early, and sessions downloaded afterwards counted as forward
evidence although they had already happened. These tests pin the replacement: activation is
an instant on the clock, the first eligible session comes from the exchange calendar, every
session is sealed once in an append-only ledger, and later changes to stored data are
recorded as revisions without altering what was sealed.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from strata.core import calendar as cal
from tests.helpers import chain, make_store, put_spread_strategy, write_minutes, write_options
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v4_research import rich_store


def at(d: date, hhmm: str = "17:00") -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET).astimezone(timezone.utc)


class Clock:
    def __init__(self):
        self.at = datetime.now(timezone.utc)

    def set(self, d: date, hhmm: str = "17:00"):
        self.at = at(d, hhmm)

    def __call__(self):
        return self.at


@pytest.fixture
def clock(monkeypatch):
    from strata.research import forward as FW
    c = Clock()
    monkeypatch.setattr(FW, "_now", c)
    return c


def after(d: date, n: int) -> list[date]:
    out = []
    while len(out) < n:
        d = cal.next_trading_day(d)
        out.append(d)
    return out


def write_days(st, sym: str, bars: list[tuple[date, float, float]]):
    """(date, open, close) -> a daily bar; replaces a bar already stored for that date."""
    st.write_daily(sym, pl.DataFrame({"date": [b[0] for b in bars], "open": [b[1] for b in bars], "high": [max(b[1], b[2]) * 1.001 for b in bars], "low": [min(b[1], b[2]) * 0.999 for b in bars],
                                      "close": [b[2] for b in bars], "volume": [1e6] * len(bars)}), provider="yahoo")
    st.rebuild_manifest(sym)
    from strata.research.features import _CACHE
    _CACHE.clear()


def path(st, sym: str, days: list[date], moves: list[float]) -> list[tuple[date, float, float]]:
    """Sessions that open at the previous close and move by `moves` percent."""
    px = float(st.read_daily(sym)["close"][-1])
    out = []
    for d, m in zip(days, moves):
        out.append((d, px, px * (1 + m / 100)))
        px = out[-1][2]
    return out


Q = "Is TST more likely to close higher when the previous day fell more than 1%?"


def study(st, clock, when: date, hhmm: str = "17:00"):
    from strata.ai import nl_graph as NG
    from strata.research import forward as FW
    from strata.research import graph as G
    g = NG.plan(Q, ["TST"], "TST", st)
    clock.set(when, hhmm)
    return FW.create_from_graph(g, G.run_graph(g, st), store=st)


# ----------------------------------------------------------------------------- activation
def test_the_first_eligible_session_comes_from_the_clock_and_the_exchange_calendar():
    from strata.research import forward as FW
    mon, tue = date(2024, 3, 4), date(2024, 3, 5)
    assert FW.first_session(at(mon, "08:00"))[:2] == (mon, "before_open")
    assert FW.first_session(at(mon, "09:29"))[:2] == (mon, "before_open")
    assert FW.first_session(at(mon, "09:30"))[:2] == (tue, "during_session"), "at the open the session has begun: it cannot count"
    assert FW.first_session(at(mon, "12:00"))[:2] == (tue, "during_session")
    assert FW.first_session(at(mon, "15:59"))[:2] == (tue, "during_session")
    assert FW.first_session(at(mon, "16:00"))[:2] == (tue, "after_close")
    assert FW.first_session(at(date(2024, 3, 8), "18:00"))[:2] == (date(2024, 3, 11), "after_close"), "Friday evening -> Monday"
    assert FW.first_session(at(date(2024, 3, 9), "12:00"))[:2] == (date(2024, 3, 11), "market_closed"), "Saturday"
    assert FW.first_session(at(date(2024, 3, 29), "10:00"))[:2] == (date(2024, 4, 1), "market_closed"), "Good Friday is not a session"
    assert FW.first_session(at(date(2024, 11, 29), "12:30"))[:2] == (date(2024, 12, 2), "during_session"), "the half day after Thanksgiving"
    assert FW.first_session(at(date(2024, 11, 29), "13:30"))[:2] == (date(2024, 12, 2), "after_close"), "it closed at 13:00"
    # the instant is UTC: 01:00 UTC on the 5th is still the evening of the 4th in New York
    assert FW.first_session(datetime(2024, 3, 5, 1, 0, tzinfo=timezone.utc))[:2] == (tue, "after_close")
    assert "does not count" in FW.first_session(at(mon, "12:00"))[2]


def test_stale_local_data_cannot_backdate_an_experiment(tmp_settings, clock):
    """THE V5 BUG. The store ends 12 sessions before the experiment is frozen. Those 12 sessions are downloaded
    afterwards together with 5 that really are new. Only the 5 may count (the old rule counted all 17)."""
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    stored_last = st.read_daily("TST")["date"][-1]
    stale = after(stored_last, 12)  # these sessions have already happened when the experiment is frozen; they are just not stored
    ex = study(st, clock, stale[-1])
    first = cal.next_trading_day(stale[-1])
    assert ex["first_session"] == first.isoformat() and ex["activation_date"] == stale[-1].isoformat() != stored_last.isoformat()
    assert ex["activated_at"] == at(stale[-1]).isoformat(timespec="seconds") and ex["activation"]["state"] == "after_close"
    assert ex["activation"]["stored_through"] == stored_last.isoformat() and ex["activation"]["sessions_not_yet_stored"] == 12
    rep = FW.evaluate(ex["id"], st)
    assert rep["status"] == "waiting" and rep["forward_sessions"] == 0 and "nothing earlier can count" in rep["notes"][0]
    # the backfill: 12 old sessions and 5 new ones arrive in one download
    new = after(stale[-1], 5)
    write_days(st, "TST", path(st, "TST", stale + new, [-2, 1, -2, 1, -2, 1, -2, 1, -2, 1, -2, 1] + [-2, 1, -2, -0.5, 0.5]))
    clock.set(new[-1])
    rep = FW.evaluate(ex["id"], st)
    assert rep["status"] == "evaluated" and rep["forward_sessions"] == 5 and rep["first"] == first.isoformat() == new[0].isoformat() and rep["last"] == new[-1].isoformat()
    assert rep["newly_sealed"] == [d.isoformat() for d in new]
    led = FW.ledger(ex["id"])
    assert min(e["session"] for e in led) == first.isoformat(), "no entry of any kind exists for a session before activation"
    # the condition held on six of the backfilled sessions (after each 2% fall) and they are not evidence
    assert [o["date"] for o in rep["sessions"]] == [new[1].isoformat(), new[3].isoformat()] and rep["signals"] == 2 and rep["hits"] == 1
    assert rep["comparison"]["forward_n"] == 2 and "a record, not a verdict" in rep["notes"][-1]
    # the clock moves on but nothing new is stored: those sessions wait, they are not sealed empty
    later = after(new[-1], 3)
    clock.set(later[-1])
    rep = FW.evaluate(ex["id"], st)
    assert rep["forward_sessions"] == 5 and rep["awaiting_data"] == [d.isoformat() for d in later] and rep["newly_sealed"] == [] and rep["eligible_sessions"] == 8


def test_a_session_already_in_progress_at_activation_never_counts(tmp_settings, clock):
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    d1, d2, d3 = after(st.read_daily("TST")["date"][-1], 3)
    ex = study(st, clock, d1, "12:00")  # frozen at noon on d1
    assert ex["activation"]["state"] == "during_session" and ex["first_session"] == d2.isoformat() and ex["activation_date"] == d1.isoformat()
    write_days(st, "TST", path(st, "TST", [d1, d2, d3], [-2, 1, -2]))
    clock.set(d1)
    assert FW.evaluate(ex["id"], st)["status"] == "waiting", "d1 has closed and is stored, and still cannot count"
    clock.set(d3, "11:00")  # d3 is open now: it has not closed, so only d2 is sealed
    rep = FW.evaluate(ex["id"], st)
    assert rep["newly_sealed"] == [d2.isoformat()] and rep["forward_sessions"] == 1 and [o["date"] for o in rep["sessions"]] == [d2.isoformat()]
    clock.set(d3)
    assert FW.evaluate(ex["id"], st)["newly_sealed"] == [d3.isoformat()]


def test_freezing_before_the_open_counts_that_session(tmp_settings, clock):
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    d1, d2 = after(st.read_daily("TST")["date"][-1], 2)
    write_days(st, "TST", path(st, "TST", [d1], [-2]))
    ex = study(st, clock, d2, "08:15")
    assert ex["activation"]["state"] == "before_open" and ex["first_session"] == d2.isoformat() and ex["activation_date"] == d1.isoformat()
    write_days(st, "TST", path(st, "TST", [d2], [1]))
    clock.set(d2)
    rep = FW.evaluate(ex["id"], st)
    assert rep["newly_sealed"] == [d2.isoformat()] and rep["signals"] == 1 and rep["hits"] == 1


# ----------------------------------------------------------------------------- sealing
def test_sealed_sessions_are_not_rewritten_when_stored_data_changes(tmp_settings, clock):
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    last = st.read_daily("TST")["date"][-1]
    ex = study(st, clock, last)
    f = after(last, 6)
    bars = path(st, "TST", f, [-2, 1, -2, -0.5, 0.2, 0.3])
    write_days(st, "TST", bars)
    clock.set(f[-1])
    rep = FW.evaluate(ex["id"], st)
    assert rep["forward_sessions"] == 6 and rep["signals"] == 2 and rep["hits"] == 1 and rep["probability"] == 0.5
    assert [(o["date"], o["outcome"]) for o in rep["sessions"]] == [(f[1].isoformat(), "target"), (f[3].isoformat(), "neither")]
    before = FW.ledger(ex["id"])
    assert [e["kind"] for e in before].count("session") == 6 and before[0]["kind"] == "activation" and before[-1]["kind"] == "position"
    # evaluating again changes nothing: every session is sealed once
    rep2 = FW.evaluate(ex["id"], st)
    assert rep2["newly_sealed"] == [] and len(FW.ledger(ex["id"])) == len(before) and rep2["revisions"] == []
    # the stored bar of a sealed session is revised so that its outcome flips, and another sealed session now meets the condition
    write_days(st, "TST", [(f[1], bars[1][1], bars[1][1] * 0.99), (f[4], bars[4][1], bars[4][1] * 0.98)])
    rep3 = FW.evaluate(ex["id"], st)
    assert (rep3["signals"], rep3["hits"], rep3["probability"]) == (2, 1, 0.5), "the report is read from the sealed entries"
    assert [(o["date"], o["outcome"]) for o in rep3["sessions"]] == [(f[1].isoformat(), "target"), (f[3].isoformat(), "neither")]
    kinds = {(r["session"], r["what"], r["entry"]) for r in rep3["revisions"]}
    assert (f[1].isoformat(), "changed", "observation") in kinds and (f[5].isoformat(), "appeared", "observation") in kinds
    assert rep3["revisions_new"] == len(rep3["revisions"]) >= 2 and "sealed entries were not altered" in " ".join(rep3["notes"])
    after_ = FW.ledger(ex["id"])
    assert after_[:len(before)] == before, "nothing that was written has changed"
    assert {e["kind"] for e in after_[len(before):]} == {"revision"}
    # the same revision is recorded once
    rep4 = FW.evaluate(ex["id"], st)
    assert rep4["revisions_new"] == 0 and len(FW.ledger(ex["id"])) == len(after_) and rep4["ledger"]["ok"]


def test_an_outcome_measured_over_several_sessions_is_added_once_it_is_known_and_only_for_a_sealed_condition(tmp_settings, clock):
    """The condition is known at the close of a session; the outcome is the close five sessions later."""
    from strata.ai import nl_graph as NG
    from strata.research import forward as FW
    from strata.research import graph as G
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    last = st.read_daily("TST")["date"][-1]
    g = NG.plan("How often is TST higher 5 days later after falling more than 1%?", ["TST"], "TST", st)
    assert g["core"]["outcome"]["params"]["horizon"] == "sessions:5"
    clock.set(last)
    ex = FW.create_from_graph(g, G.run_graph(g, st), store=st)
    f = after(last, 8)
    bars = path(st, "TST", f, [-2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    write_days(st, "TST", bars[:3])
    clock.set(f[2])
    r1 = FW.evaluate(ex["id"], st)
    assert r1["forward_sessions"] == 3 and r1["conditions"] == 1 and r1["signals"] == 0 and r1["pending_outcomes"] == [f[0].isoformat()] and "not known yet" in " ".join(r1["notes"])
    assert [(e["kind"], e["session"]) for e in FW.ledger(ex["id"]) if e["kind"] in ("condition", "observation")] == [("condition", f[0].isoformat())]
    # a bar of a sealed session is revised: that session now meets the condition too, but it was not sealed that way
    write_days(st, "TST", [(f[1], bars[1][1], bars[1][1] * 0.97)] + path(st, "TST", f[2:6], [0.5, 0.5, 0.5, 0.5]))
    clock.set(f[5])
    r2 = FW.evaluate(ex["id"], st)
    assert r2["forward_sessions"] == 6 and r2["conditions"] == 1 and r2["signals"] == 1 and r2["pending_outcomes"] == [], "the outcome of the sealed condition is now known and is added"
    obs = [e for e in FW.ledger(ex["id"]) if e["kind"] == "observation"]
    assert [(e["session"], e["body"]["outcome"]) for e in obs] == [(f[0].isoformat(), "target")] and r2["hits"] == 1 and obs[0]["recorded_at"] == at(f[5]).isoformat(timespec="seconds")
    assert (f[1].isoformat(), "appeared", "condition") in {(r["session"], r["what"], r["entry"]) for r in r2["revisions"]}
    # when the outcome of the session that was edited in becomes known, it is still not evidence
    write_days(st, "TST", path(st, "TST", f[6:], [0.5, 0.5]))
    clock.set(f[7])
    r3 = FW.evaluate(ex["id"], st)
    assert r3["forward_sessions"] == 8 and r3["signals"] == 1 and r3["conditions"] == 1 and [o["date"] for o in r3["sessions"]] == [f[0].isoformat()]
    assert (f[1].isoformat(), "appeared", "observation") in {(r["session"], r["what"], r["entry"]) for r in r3["revisions"]} and r3["ledger"]["ok"]


def test_the_ledger_cannot_be_edited_and_tampering_is_detected(tmp_settings, clock):
    from strata.api import db
    from strata.config import ENGINE_VERSION
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    last = st.read_daily("TST")["date"][-1]
    ex = study(st, clock, last)
    f = after(last, 3)
    write_days(st, "TST", path(st, "TST", f, [-2, 1, 0.1]))
    clock.set(f[-1])
    FW.evaluate(ex["id"], st)
    root = FW.ledger(ex["id"])[0]
    assert root["kind"] == "activation" and root["prev_hash"] == FW.GENESIS and root["body"]["config_hash"] == ex["config_hash"] and root["body"]["engine_version"] == ENGINE_VERSION == ex["engine_version"]
    assert root["body"]["activated_at"] == ex["activated_at"] and root["body"]["first_session"] == ex["first_session"] and "assumptions" in root["body"]
    v = FW.verify(ex["id"])
    assert v["ok"] and v["entries"] == len(FW.ledger(ex["id"])) and v["head"] == FW.ledger(ex["id"])[-1]["hash"]
    with db.connect() as con:
        for sql in ("UPDATE forward_ledger SET body='{}' WHERE experiment_id=?", "DELETE FROM forward_ledger WHERE experiment_id=?", "UPDATE forward_experiments SET config='{}' WHERE id=?",
                    "UPDATE forward_experiments SET first_session='2000-01-03' WHERE id=?", "UPDATE forward_experiments SET activated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
                    "UPDATE forward_experiments SET engine_version='0' WHERE id=?", "UPDATE forward_experiments SET activation_date='2000-01-03' WHERE id=?"):
            with pytest.raises(sqlite3.DatabaseError, match="append-only|cannot be edited"):
                con.execute(sql, (ex["id"],))
        con.execute("UPDATE forward_experiments SET status='stopped' WHERE id=?", (ex["id"],))  # stopping is not an edit of what was frozen
        con.execute("UPDATE forward_experiments SET status='active' WHERE id=?", (ex["id"],))
    assert FW.verify(ex["id"])["ok"]
    # someone with access to the file removes the guard and edits an entry: the chain no longer verifies and nothing is evaluated
    with db.connect() as con:
        con.execute("DROP TRIGGER forward_ledger_no_update")
        con.execute("UPDATE forward_ledger SET body=replace(body, '\"target\"', '\"stop\"') WHERE experiment_id=? AND kind='observation'", (ex["id"],))
    v = FW.verify(ex["id"])
    assert not v["ok"] and v["broken_at"] is not None and "altered" in v["detail"]
    with pytest.raises(ValueError, match="integrity check"):
        FW.evaluate(ex["id"], st)
    assert FW.evaluate_active(st)[0]["status"] == "error"


def test_an_experiment_from_before_version_6_is_dated_from_when_it_was_created(tmp_settings, clock):
    """A V5 row: activation was the last STORED session (stale), and an evaluation already counted backfilled sessions."""
    from strata.ai import nl_graph as NG
    from strata.api import db
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    stored_last = st.read_daily("TST")["date"][-1]
    stale = after(stored_last, 8)
    created = at(stale[-1], "11:00")  # frozen mid-session, with data 8 sessions old
    g = NG.plan(Q, ["TST"], "TST", st)
    cfg = {"graph": g}
    with db.connect() as con:
        con.execute("INSERT INTO forward_experiments (id, name, kind, symbol, config, config_hash, version, benchmark, assumptions, activation_date, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("legacy0001", "old", "study", "TST", json.dumps(cfg), FW._hash(cfg), 1, "{}", "{}", stored_last.isoformat(), "active", created.isoformat(timespec="seconds"), created.isoformat(timespec="seconds")))
        con.execute("INSERT INTO forward_evaluations (experiment_id, evaluated_at, through_date, report) VALUES (?,?,?,?)", ("legacy0001", created.isoformat(), stale[-2].isoformat(), json.dumps({"status": "evaluated", "forward_sessions": 7, "notes": []})))
    ex = FW.get("legacy0001")
    first = cal.next_trading_day(stale[-1])
    assert ex["first_session"] == first.isoformat() and ex["activated_at"] == created.isoformat(timespec="seconds") and ex["activation_date"] == stale[-1].isoformat()
    assert ex["activation"]["legacy_activation_date"] == stored_last.isoformat() and "now dated from the moment it was created" in ex["activation"]["migrated"]
    assert ex["latest"]["superseded"] is True and "superseded" in ex["latest"]["notes"][0] and ex["sealed_sessions"] == 0
    new = after(stale[-1], 2)
    write_days(st, "TST", path(st, "TST", stale + new, [-2, 1] * 4 + [-2, 1]))
    clock.set(new[-1])
    rep = FW.evaluate("legacy0001", st)
    assert rep["forward_sessions"] == 2 and rep["first"] == first.isoformat() and [o["date"] for o in rep["sessions"]] == [new[1].isoformat()]
    assert FW.ledger("legacy0001")[0]["kind"] == "activation" and FW.verify("legacy0001")["ok"]
    with db.connect() as con, pytest.raises(sqlite3.DatabaseError, match="cannot be edited"):
        con.execute("UPDATE forward_experiments SET first_session='2000-01-03' WHERE id='legacy0001'")  # once dated, it is frozen like any other


# ----------------------------------------------------------------------------- a strategy: fills, fees, open positions, missing data
D = [date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6), date(2024, 3, 7)]


def quotes(d: date, nxt: date) -> list[dict]:
    q = []
    for t in ("10:00", "12:00", "15:30"):
        q += chain(t, {(100, "P"): (3.0, 3.2, -0.5), (95, "P"): (1.00, 1.10, -0.20), (90, "P"): (0.40, 0.50, -0.10), (85, "P"): (0.10, 0.20, -0.05)}, nxt)
        q += chain(t, {(100, "P"): (1.0, 1.2, -0.5), (95, "P"): (0.10, 0.20, -0.05), (90, "P"): (0.00, 0.05, -0.02)}, d)
    return q


def test_a_strategy_is_sealed_session_by_session_and_an_open_position_survives(tmp_settings, clock):
    """Sell the 0.20-delta put expiring tomorrow, buy the put 5 lower, at 10:00. Credit (1.00 - 0.50) x 100 = $50,
    two contracts at $0.65 = $1.30 in fees, both expire worthless: +$48.70. The position is open overnight."""
    from strata.research import forward as FW
    st = make_store(tmp_settings)
    write_minutes(st, "TST", [date(2024, 3, 1)], lambda d, i: 100.0)
    strat = put_spread_strategy(legs=[{"id": "sp", "side": "sell", "right": "P", "dte": 1, "strike": {"method": "delta", "value": 0.20}},
                                      {"id": "lp", "side": "buy", "right": "P", "dte": 1, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}}])
    cfg = {"start": "2024-01-02", "end": "2024-03-01", "strategies": [strat]}
    clock.set(date(2024, 3, 1))
    ex = FW.create("strategy", "TST", cfg, "overnight spread", assumptions={"execution": {"fill_model": "conservative"}}, store=st)
    assert ex["first_session"] == "2024-03-04"
    # session 1: the entry
    write_minutes(st, "TST", [D[0]], lambda d, i: 100.0)
    write_options(st, "TST", D[0], quotes(D[0], D[1]))
    clock.set(D[0])
    r1 = FW.evaluate(ex["id"], st)
    assert r1["newly_sealed"] == ["2024-03-04"] and (r1["signals"], r1["executed"], r1["metrics"]["trades"]) == (1, 1, 0) and r1["trades"] == []
    assert len(r1["open_positions"]) == 1 and r1["open_positions"][0]["reproduced"] and "still open" in r1["notes"][0]
    fill = next(e for e in FW.ledger(ex["id"]) if e["kind"] == "fill")
    assert fill["session"] == "2024-03-04" and fill["body"]["entry_cash"] == 50 and fill["body"]["entry_time"].startswith("2024-03-04T10:00")
    assert [(l["right"], l["strike"], l["expiration"], l["side"], l["entry_price"], l["price_source"]) for l in fill["body"]["legs"]] == [("P", 95, "2024-03-05", "sell", 1, "quote"), ("P", 90, "2024-03-05", "buy", 0.5, "quote")]
    snap = [e for e in FW.ledger(ex["id"]) if e["kind"] == "position"][-1]
    assert len(snap["body"]["open"]) == 1 and snap["body"]["open"][0]["key"] == fill["key"]
    # "restart": nothing is held in memory between evaluations; the next one rebuilds the open position from the ledger and the replay
    write_minutes(st, "TST", [D[1]], lambda d, i: 100.0)
    write_options(st, "TST", D[1], quotes(D[1], D[2]))
    clock.set(D[1])
    r2 = FW.evaluate(ex["id"], st)
    assert r2["newly_sealed"] == ["2024-03-05"] and r2["open_positions"] == [] and len(r2["trades"]) == 1
    t = r2["trades"][0]
    assert (t["entry_date"], t["exit_reason"], t["units"]) == ("2024-03-04", "expiration", 1) and t["pnl"] == pytest.approx(48.70) and t["fees"] == pytest.approx(1.30)
    assert r2["metrics"] == pytest.approx({"trades": 1, "net_pnl": 48.70, "win_rate": 1.0, "expectancy": 48.70, "profit_factor": None, "max_drawdown": 0.0, "fees": 1.30})
    ex_ = next(e for e in FW.ledger(ex["id"]) if e["kind"] == "exit")
    assert ex_["key"] == fill["key"] and ex_["session"] == "2024-03-05" and FW.ledger(ex["id"])[1] == fill, "the sealed entry was not rewritten when its exit was added"
    # session 3 has no option chain: the signal is a missing-data event, not a trade and not 'no signal'
    write_minutes(st, "TST", [D[2]], lambda d, i: 100.0)
    clock.set(D[2])
    r3 = FW.evaluate(ex["id"], st)
    assert r3["newly_sealed"] == ["2024-03-06"] and (r3["signals"], r3["executed"], r3["not_executed"]) == (2, 1, 1) and r3["not_executed_by_reason"] == {"no_option_data": 1}
    assert r3["missed_signals"][0]["missing_data"] is True and r3["missing_data"][0]["session"] == "2024-03-06" and r3["data"]["with_option_data"] == 2 and r3["data"]["sessions"] == 3
    assert "1 of the 3 forward sessions have no stored option prices" in " ".join(r3["notes"])
    n = len(FW.ledger(ex["id"]))
    # the chain for session 3 is stored afterwards: the sealed session is not re-traded
    write_options(st, "TST", D[2], quotes(D[2], D[3]))
    r4 = FW.evaluate(ex["id"], st)
    assert r4["newly_sealed"] == [] and (r4["signals"], r4["executed"]) == (2, 1) and len(r4["trades"]) == 1 and r4["metrics"]["net_pnl"] == pytest.approx(48.70)
    assert {(r["session"], r["what"], r["entry"]) for r in r4["revisions"]} == {("2024-03-06", "appeared", "fill"), ("2024-03-06", "vanished", "missing_data")}
    assert len(FW.ledger(ex["id"])) == n + 2 and FW.verify(ex["id"])["ok"]
    assert [e["kind"] for e in FW.ledger(ex["id"])] == ["activation", "fill", "session", "position", "exit", "session", "position", "missing_data", "session", "position", "revision", "revision"]
    # every sealed session names the engine and the datasets it was computed from
    s1 = next(e for e in FW.ledger(ex["id"]) if e["kind"] == "session")
    assert s1["body"]["engine_version"] and s1["body"]["sources"]["pricing"] == "historical" and "TST" in s1["body"]["sources"]["datasets"] and s1["body"]["minutes_after_close"] == 60


def test_only_experiments_with_unsealed_stored_sessions_are_evaluated_after_a_collection(tmp_settings, clock):
    from strata.research import forward as FW
    st = rich_store(tmp_settings, sessions=900, effect=0.0, intraday=0)
    last = st.read_daily("TST")["date"][-1]
    ex = study(st, clock, last)
    assert FW.pending(ex, st) == [] and FW.evaluate_active(st) == []
    f = after(last, 2)
    clock.set(f[-1])
    assert FW.pending(FW.get(ex["id"]), st) == [] and FW.evaluate_active(st) == [], "two sessions have closed but nothing is stored for them"
    write_days(st, "TST", path(st, "TST", f, [-2, 1]))
    assert FW.pending(FW.get(ex["id"]), st) == [d.isoformat() for d in f]
    out = FW.evaluate_active(st)
    assert [(o["id"], o["status"], o["forward_sessions"], o["newly_sealed"]) for o in out] == [(ex["id"], "evaluated", 2, 2)]
    assert FW.evaluate_active(st) == [] and FW.get(ex["id"])["sealed_sessions"] == 2
    assert timedelta(0) <= datetime.now(timezone.utc) - datetime.fromisoformat(FW.get(ex["id"])["updated_at"]) < timedelta(minutes=5)
