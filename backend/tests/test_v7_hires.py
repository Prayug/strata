"""V7: execution on individual trade events.

A trade-bar backtest fills at the last print it has seen, which happened before its decision. Here the same decisions are
filled on the first eligible print AFTER them, from a tape of individual trades. Every tape in this file is written by hand
or served through a mocked transport: what the provider's real tape shows is measured separately (docs/V7_REPORT.md)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import numpy as np
import pytest

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.engine import hires as H
from strata.engine.engine import run_backtest
from strata.engine.models import BacktestConfig, Execution
from tests.helpers import et_epoch
from tests.test_v2_research import tmp_settings  # noqa: F401
from tests.test_v7_expansion import D_BAND, cfg, premium, put_chain, store  # noqa: F401  (a band of 0DTE calls, flat underlying at 500)

OCC = S.occ_symbol("SPY", D_BAND, "C", 500.0)
NS = H.NS


def at(hhmmss: str, ms: int = 0, d: date = D_BAND) -> str:
    """An exchange timestamp on the session, New York clock -> the provider's UTC nanosecond string."""
    h, m, s = map(int, hhmmss.split(":"))
    t = datetime(d.year, d.month, d.day, h, m, s, tzinfo=cal.ET).astimezone(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms * 1_000_000:09d}Z"


def pr(hhmmss: str, price: float, size: int = 5, cond: str = "I", ms: int = 0, exch: str = "C") -> dict:
    return {"t": at(hhmmss, ms), "p": price, "s": size, "c": cond, "x": exch}


def bar_run(store, **over):
    """An at-the-money call bought at 10:00 and sold at 15:30, on minute bars."""
    c = cfg(D_BAND, {"method": "atm_offset", "value": 0, "unit": "points"})
    c["execution"].update(over.pop("execution", {}))
    c["strategies"][0]["exits"].update(over.pop("exits", {}))
    res = run_backtest(c, store=store, record_paths=False)
    assert len(res.trades) == 1
    return res.trades, BacktestConfig.model_validate(c)


ENTRY = [pr("09:59:58", 1.00), pr("10:00:00", 1.02, ms=0), pr("10:00:00", 1.07, ms=412), pr("10:00:03", 1.09)]
EXIT = [pr("15:29:59", 2.00), pr("15:30:02", 1.90, ms=250), pr("15:30:05", 1.85)]


# ----------------------------------------------------------------------------- the first print AFTER the decision
def test_a_fill_is_the_first_eligible_print_after_the_decision_never_one_before_it(store):
    trades, c = bar_run(store)
    bar = trades[0]["legs"][0]
    assert trades[0]["signal_ts"] == et_epoch(D_BAND, "10:00") and (trades[0]["meta"]["entry_print_age_seconds"] or 0) >= 0, "the bar run priced the entry from a print made before 10:00"
    src = H.ListSource({OCC: ENTRY + EXIT})
    rep = H.replay(trades, c, src)
    p = rep["positions"][0]
    en, xi = p["legs"][0]["entry"], p["legs"][0]["exit"]
    # 09:59:58 is before the decision; 10:00:00.000 is AT it, not after it: neither is a fill
    assert en["print"] == 1.07 and en["et"] == "10:00:00.412" and en["seconds_after_decision"] == 0.412 and en["prints"][0]["condition"] == "I"
    assert xi["print"] == 1.90 and xi["et"] == "15:30:02.250" and xi["seconds_after_decision"] == 2.25 and xi["policy"] == "decision_clock"
    assert p["status"] == "repriced" and p["tick"]["pnl"] == round((1.90 - 1.07) * 100 - trades[0]["commissions"], 2)
    assert p["pnl_difference"] == round(p["tick"]["pnl"] - trades[0]["pnl"], 2) and en["bar_price"] == bar["entry_price"]
    # only the windows around the two decisions were asked for, not the session
    assert [(a // NS, b // NS) for _, a, b in src.asked] == [(et_epoch(D_BAND, "10:00"), et_epoch(D_BAND, "10:05")), (et_epoch(D_BAND, "15:30"), et_epoch(D_BAND, "16:00"))]
    s = rep["summary"]
    assert s["repriced"] == 1 and s["entries_not_filled"] == 0 and s["tick_pnl_repriced"] == p["tick"]["pnl"] and s["difference"] == p["pnl_difference"]
    assert s["conditions_of_prints_used"]["I"]["count"] == 2 and "executed electronically" in s["conditions_of_prints_used"]["I"]["meaning"]
    assert "not a bid or an ask" in s["text"]


def test_latency_moves_the_moment_an_order_can_first_be_filled(store):
    trades, c = bar_run(store)
    src = H.ListSource({OCC: ENTRY + EXIT})
    rep = H.replay(trades, c, src, H.HiresSettings(latency_seconds=2.0))
    en, xi = rep["positions"][0]["legs"][0]["entry"], rep["positions"][0]["legs"][0]["exit"]
    assert en["print"] == 1.09 and en["et"] == "10:00:03.000", "eligible at 10:00:02: the 10:00:00.412 print is too early"
    assert xi["print"] == 1.90, "eligible at 15:30:02.000: the print a quarter of a second later fills it"
    # exactly at the eligible instant is not after it: with three seconds the 10:00:03.000 print no longer fills the entry
    late = H.ListSource({OCC: ENTRY + [pr("10:00:04", 1.12)] + EXIT})
    rep = H.replay(trades, c, late, H.HiresSettings(latency_seconds=3.0))
    assert rep["positions"][0]["legs"][0]["entry"]["print"] == 1.12 and rep["positions"][0]["legs"][0]["exit"]["print"] == 1.85
    assert H.replay(trades, c, src, H.HiresSettings(latency_seconds=3.0))["positions"][0]["status"] == "entry_not_filled"


def test_prints_that_say_nothing_about_a_single_order_do_not_fill_one(store):
    trades, c = bar_run(store)
    tape = [pr("10:00:01", 0.80, cond="f"), pr("10:00:02", 0.85, cond="g"), pr("10:00:03", 0.90, cond="B"), pr("10:00:04", 0.95, size=1), pr("10:00:05", 1.10, size=4), pr("10:00:06", 1.20, cond="j")] + EXIT
    one = lambda **kw: H.replay(trades, c, H.ListSource({OCC: tape}), H.HiresSettings(**kw))["positions"][0]["legs"][0]["entry"]
    assert one()["print"] == 0.95, "legs of spreads (f, g) and a late report (B) are passed over; the first single-leg print fills"
    assert one(min_size=3)["print"] == 1.10, "a one-lot is ignored when prints must be at least three contracts"
    assert one(conditions="all_in_sequence")["print"] == 0.80, "asked to accept any in-sequence trade, the spread leg is first"
    assert H.cond_class("f") == "complex" and H.cond_class("j") == "book" and H.cond_class("B") == "late" and H.cond_class("?") == "unknown"
    assert set(H.FILTERS) == {"single_leg", "single_leg_and_book", "all_in_sequence"} and all(k in "ISabcdeJ" for k, v in H.CONDITIONS.items() if v[1] == "single")
    with pytest.raises(ValueError, match="conditions must be one of"):
        H.HiresSettings(conditions="anything").validate()


def test_an_entry_with_no_eligible_print_in_its_working_time_is_not_filled(store):
    trades, c = bar_run(store)
    quiet = [pr("09:59:58", 1.00), pr("10:07:00", 1.30)] + EXIT  # nothing between 10:00 and 10:05
    rep = H.replay(trades, c, H.ListSource({OCC: quiet}))
    p = rep["positions"][0]
    assert p["status"] == "entry_not_filled" and p["tick"] == {"pnl": 0.0, "executed": False}
    assert "no trade of this contract was reported in the window" in p["notes"][0] and "from a print made before its decision" in p["notes"][0]
    s = rep["summary"]
    assert s["entries_not_filled"] == 1 and s["repriced"] == 0 and s["bar_pnl_of_entries_not_filled"] == trades[0]["pnl"] and "was not filled" in s["text"]
    # prints exist but none can fill: the reason says so instead of "no trade"
    rep = H.replay(trades, c, H.ListSource({OCC: [pr("10:00:01", 1.0, cond="f")] + EXIT}))
    assert "1 trade in the window, none of a kind that can fill an order" in rep["positions"][0]["notes"][0]


def test_an_exit_is_never_abandoned_but_says_when_the_tape_did_not_establish_it(store):
    trades, c = bar_run(store)
    rep = H.replay(trades, c, H.ListSource({OCC: ENTRY + [pr("15:29:59", 2.0), pr("15:36:00", 1.50)]}))
    xi = rep["positions"][0]["legs"][0]["exit"]
    assert xi["print"] == 1.50 and xi["late"] is True and xi["seconds_after_decision"] == 360.0 and rep["summary"]["late_exits"] == 1
    rep = H.replay(trades, c, H.ListSource({OCC: ENTRY + [pr("15:29:59", 2.0)]}))
    p = rep["positions"][0]
    assert p["legs"][0]["exit"]["kept_bar_price"] is True and p["legs"][0]["exit"]["price"] == trades[0]["legs"][0]["exit_price"] and p["status"] == "repriced"
    assert "is not established by the tape" in p["notes"][0] and rep["summary"]["exits_not_established"] == 1


def test_a_tape_that_was_not_read_to_the_end_establishes_nothing(store):
    trades, c = bar_run(store)

    class Cut(H.ListSource):
        def fetch(self, occ, t0, t1):
            t = super().fetch(occ, t0, t1)
            t.complete = False
            return t
    rep = H.replay(trades, c, Cut({OCC: ENTRY + EXIT}))
    p = rep["positions"][0]
    assert p["status"] == "entry_unknown" and p["tick"]["pnl"] is None and rep["summary"]["not_established"] == 1 and rep["summary"]["tick_pnl_all_positions"] is None


# ----------------------------------------------------------------------------- size
def test_an_order_larger_than_a_print_walks_the_tape_when_asked_to(store):
    trades, c = bar_run(store, execution={})
    trades[0]["legs"][0]["contracts"] = 10
    tape = [pr("10:00:01", 1.00, size=3), pr("10:00:02", 1.10, size=4), pr("10:00:03", 1.20, size=5), pr("10:00:04", 1.30, size=50)] + [pr("15:30:01", 2.0, size=100)]
    first = H.replay(trades, c, H.ListSource({OCC: tape}))["positions"][0]["legs"][0]["entry"]
    assert first["print"] == 1.00 and first["smaller_than_order"] is True, "first_print: the price of the first eligible print, and a note that it was smaller than the order"
    walk = H.replay(trades, c, H.ListSource({OCC: tape}), H.HiresSettings(fill_model="size_limited"))["positions"][0]["legs"][0]["entry"]
    assert [x["taken"] for x in walk["prints"]] == [3, 4, 3] and walk["print"] == pytest.approx((3 * 1.00 + 4 * 1.10 + 3 * 1.20) / 10) and walk["et"] == "10:00:03.000"
    half = H.replay(trades, c, H.ListSource({OCC: tape}), H.HiresSettings(fill_model="size_limited", participation=0.5))["positions"][0]["legs"][0]["entry"]
    assert [x["taken"] for x in half["prints"]] == [1, 2, 2, 5] and half["et"] == "10:00:04.000", "half of each print: it takes four prints, and a later one"
    thin = H.replay(trades, c, H.ListSource({OCC: tape[:2] + tape[4:]}), H.HiresSettings(fill_model="size_limited"))["positions"][0]
    assert thin["status"] == "entry_partial" and thin["legs"][0]["entry"]["filled"] == 7 and "only 7 of 10 contracts could be filled" in thin["notes"][0]


# ----------------------------------------------------------------------------- the open
def test_an_order_decided_at_the_open_can_be_held_back_and_first_minute_fills_are_reported(store):
    c0 = cfg(D_BAND, {"method": "atm_offset", "value": 0, "unit": "points"})
    c0["strategies"][0]["entry"]["conditions"]["children"][0]["params"]["times"] = "09:30"
    put_chain(store, D_BAND, D_BAND, [500], times=("09:30",))  # a print in the opening minute, so the bar run can enter at 09:31
    res = run_backtest(c0, store=store, record_paths=False)
    c = BacktestConfig.model_validate(c0)
    assert res.trades and res.trades[0]["signal_ts"] == et_epoch(D_BAND, "09:30")
    tape = [pr("09:30:00", 3.00, ms=498), pr("09:30:07", 2.40), pr("09:30:45", 2.10)] + EXIT
    rep = H.replay(res.trades, c, H.ListSource({OCC: tape}))
    en = rep["positions"][0]["legs"][0]["entry"]
    assert en["print"] == 3.00 and en["first_minute"] is True and rep["summary"]["first_minute_entries"]["n"] == 1
    held = H.replay(res.trades, c, H.ListSource({OCC: tape}), H.HiresSettings(open_delay_seconds=30))["positions"][0]["legs"][0]["entry"]
    assert held["print"] == 2.10 and held["et"] == "09:30:45.000", "not eligible before 09:30:30"


# ----------------------------------------------------------------------------- target and stop: which came first
def test_resting_orders_let_the_tape_decide_whether_the_target_or_the_stop_came_first(store):
    trades, c = bar_run(store, exits={"profit_target": {"enabled": True, "mode": "pct", "value": 30}, "stop_loss": {"enabled": True, "mode": "pct", "value": 15}})
    rest = H.HiresSettings(exit_policy="resting_orders")
    fill = pr("10:00:01", 1.00)  # entry at 1.00: the target rests at 1.30, the stop triggers at 0.85
    # the stop is touched at 10:02, the target only at 10:04: the stop came first, and it fills on the NEXT print (a market order has no price of its own)
    tape = [fill, pr("10:01:00", 0.95), pr("10:02:00", 0.84), pr("10:02:01", 0.70), pr("10:04:00", 1.45)] + EXIT
    p = H.replay(trades, c, H.ListSource({OCC: tape}), rest)["positions"][0]
    xi = p["legs"][0]["exit"]
    assert xi["reason"] == "stop_loss" and xi["first"] == "stop" and xi["trigger_print"] == 0.84 and xi["print"] == 0.70 and xi["et"] == "10:02:01.000" and xi["target_touched_later"] is True
    assert xi["limit"] == 1.30 and xi["stop"] == 0.85 and p["tick"]["exit_reason"] == "stop_loss"
    # the other order: a print THROUGH 1.30 fills the limit at 1.30, not at the print
    tape = [fill, pr("10:01:00", 1.30), pr("10:01:30", 1.36), pr("10:03:00", 0.60)] + EXIT
    p = H.replay(trades, c, H.ListSource({OCC: tape}), rest)["positions"][0]
    xi = p["legs"][0]["exit"]
    assert xi["reason"] == "profit_target" and xi["first"] == "target" and xi["price"] == pytest.approx(1.30) and xi["print"] == 1.36 and xi["et"] == "10:01:30.000"
    assert xi["touches_at_the_limit_not_counted"] == 1 and xi["stop_touched_later"] is True, "the print AT 1.30 is a touch: the order's place in the queue is unknown"
    assert p["tick"]["gross_pnl"] == pytest.approx(30.0)
    touch = H.replay(trades, c, H.ListSource({OCC: tape}), H.HiresSettings(exit_policy="resting_orders", limit_needs_trade_through=False))["positions"][0]["legs"][0]["exit"]
    assert touch["et"] == "10:01:00.000", "told that a touch is enough, the limit fills one print earlier"
    # neither is touched: the backtest's own exit decision stands, filled on the first print after it
    calm = [fill, pr("10:30:00", 1.05), pr("14:00:00", 1.10)] + [pr("15:29:59", 1.12), pr("15:30:02", 1.11)]
    bar_calm = [dict(trades[0], exit_reason="time_exit", exit_ts=et_epoch(D_BAND, "15:30"), legs=[dict(trades[0]["legs"][0], exit_reason="time_exit", exit_ts=et_epoch(D_BAND, "15:30"))])]
    p = H.replay(bar_calm, c, H.ListSource({OCC: calm}), rest)["positions"][0]
    assert p["legs"][0]["exit"]["neither_touched"] is True and p["legs"][0]["exit"]["print"] == 1.11 and p["tick"]["exit_reason"] == "time_exit"
    # the summary counts positions that left for a different reason than on the backtest's clock
    tape = [fill, pr("10:01:30", 1.36)] + EXIT
    rep = H.replay(bar_calm, c, H.ListSource({OCC: tape}), rest)
    assert rep["summary"]["exit_reason_changed"] == 1 and rep["summary"]["exit_reason_changes"][0] == {"id": 1, "date": D_BAND.isoformat(), "bar": "time_exit", "tick": "profit_target"}
    assert "left for a different reason" in rep["summary"]["text"] and "the tape shows profit target at 10:01:30.000" in rep["positions"][0]["notes"][0]


# ----------------------------------------------------------------------------- two legs
def test_each_leg_of_a_spread_fills_on_its_own_tape_and_the_gap_between_them_is_reported(store):
    c0 = cfg(D_BAND, {"method": "atm_offset", "value": 0, "unit": "points"})
    c0["strategies"][0]["legs"].append({"id": "sc", "side": "sell", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "leg_offset", "value": 3, "unit": "points", "ref_leg": "lc"}})
    res = run_backtest(c0, store=store, record_paths=False)
    c = BacktestConfig.model_validate(c0)
    short = S.occ_symbol("SPY", D_BAND, "C", 503.0)
    tapes = {OCC: [pr("10:00:01", 1.00)] + EXIT, short: [pr("10:00:09", 0.20), pr("15:30:04", 0.10)]}
    rep = H.replay(res.trades, c, H.ListSource(tapes))
    p = rep["positions"][0]
    assert p["entry_leg_gap_seconds"] == 8.0 and p["status"] == "repriced" and [l["entry"]["print"] for l in p["legs"]] == [1.00, 0.20]
    assert p["tick"]["gross_pnl"] == pytest.approx((1.90 - 1.00) * 100 - (0.10 - 0.20) * 100)
    refused = H.replay(res.trades, c, H.ListSource(tapes), H.HiresSettings(max_leg_gap_seconds=5))["positions"][0]
    assert refused["status"] == "entry_legged" and "8.0 s apart" in refused["notes"][0] and refused["tick"]["executed"] is False
    note = H.replay(res.trades, c, H.ListSource(tapes), H.HiresSettings(exit_policy="resting_orders"))["positions"][0]["notes"]
    assert any("only replayed for a single option leg" in n for n in note)


# ----------------------------------------------------------------------------- the arithmetic and the transport
def test_a_print_is_priced_with_the_engines_own_arithmetic(store):
    ex = {"trade_bar_slippage_pct": 2.0, "slippage_per_contract": 0.01, "extra_slippage_pct": 1.0}
    trades, c = bar_run(store, execution=ex)
    leg = trades[0]["legs"][0]
    o, cl = (int(x.timestamp()) for x in cal.session_bounds(D_BAND))
    assert H.priced(leg["entry_print"], +1, c.execution, 1, trades[0]["entry_ts"], cl, o, cl) == pytest.approx(leg["entry_price"])
    assert H.priced(leg["exit_print"], -1, c.execution, 1, trades[0]["exit_ts"], cl, o, cl) == pytest.approx(leg["exit_price"])
    assert H.priced(1.0, +1, Execution(), 1, o, cl, o, cl) == 1.0 and H.priced(1.0, -1, Execution(trade_bar_slippage_pct=5), 1, o, cl, o, cl) == pytest.approx(0.95)
    # nanoseconds survive: no float is involved
    ns = H.parse_ns("2026-11-24T14:30:00.498817169Z")
    assert ns == 1795530600 * NS + 498817169 and H.iso_ns(ns) == "2026-11-24T14:30:00.498817169Z" and H.et_clock(ns) == "09:30:00.498"


def test_the_provider_is_read_to_the_last_page_and_the_window_is_kept(store, monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("test-key-id", "test-secret"))
    monkeypatch.setattr(S.Alpaca, "rate", 1000.0)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(dict(req.url.params))
        assert req.url.path == "/v1beta1/options/trades" and req.method == "GET"
        if req.url.params.get("page_token") == "p2":
            return httpx.Response(200, json={"trades": {OCC: [pr("10:00:03", 1.09)]}, "next_page_token": None})
        return httpx.Response(200, json={"trades": {OCC: [pr("10:00:00", 1.07, ms=412)]}, "next_page_token": "p2"})
    monkeypatch.setattr(S, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    src = H.AlpacaTrades(store)
    a, b = et_epoch(D_BAND, "10:00") * NS, et_epoch(D_BAND, "10:05") * NS
    t = src.fetch(OCC, a, b)
    assert len(t) == 2 and t.complete and t.pages == 2 and t.price.tolist() == [1.07, 1.09] and calls[0]["sort"] == "asc" and calls[0]["limit"] == "10000"
    assert calls[0]["start"] == "2024-03-04T15:00:00.000000000Z" and calls[0]["end"] == "2024-03-04T15:05:00.000000000Z"
    n = len(calls)
    again = src.fetch(OCC, a + NS, b - NS)  # a window inside the one already read: served from the cache, cut to size
    assert len(calls) == n and again.cached and again.price.tolist() == [1.09] or again.price.tolist() == [1.07, 1.09]
    # without a key nothing is asked and nothing is concluded
    monkeypatch.setattr(S.Alpaca, "_keys", lambda self: ("", ""))
    trades, c = bar_run(store)
    rep = H.replay(trades, c, H.AlpacaTrades(store))
    assert rep["status"] == "blocked" and rep["reason"].startswith("BLOCKED") and rep["positions"] == []
    quotes = BacktestConfig.model_validate({**c.model_dump(mode="json"), "pricing": {"mode": "historical"}})
    assert H.replay(trades, quotes, H.ListSource({}))["status"] == "not_applicable"


def test_a_stored_run_keeps_its_report_and_every_fill_can_be_exported(store, tmp_settings):
    from strata.api.results import save_backtest
    c0 = cfg(D_BAND, {"method": "atm_offset", "value": 0, "unit": "points"})
    res = run_backtest(c0, store=store, record_paths=False)
    save_backtest("t_hires", res)
    rep = H.run_for("t_hires", store, source=H.ListSource({OCC: ENTRY + EXIT}))
    assert rep["run_id"] == "t_hires" and rep["summary"]["repriced"] == 1
    again = H.saved("t_hires")
    assert set(again) == {"decision_clock"} and again["decision_clock"]["summary"]["tick_pnl_repriced"] == rep["summary"]["tick_pnl_repriced"]
    H.run_for("t_hires", store, settings={"exit_policy": "resting_orders"}, source=H.ListSource({OCC: ENTRY + EXIT}))
    assert set(H.saved("t_hires")) == {"decision_clock", "resting_orders"}
    row = H.csv_rows(rep)[0]
    assert row["contract"] == OCC and row["entry_tick_time_utc"] == at("10:00:00", 412) and row["entry_tick_print"] == 1.07 and row["entry_condition"] == "I" and row["entry_exchange"] == "C"
    assert row["exit_tick_print"] == 1.90 and row["pnl_tick"] == rep["positions"][0]["tick"]["pnl"] and row["pnl_bar"] == res.trades[0]["pnl"]
    assert np.isfinite(row["entry_seconds_after_decision"])
