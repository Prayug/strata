"""V5: the strategy IDE's backend - section edits that leave the rest untouched, the
condition debugger, rule previews - and the onboarding state."""
from __future__ import annotations

import copy
from datetime import date

import pytest

from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v5_engine import D1, call, ramp, setup


def strat():
    s = call(2)
    s["exits"] = {"profit_target": {"enabled": True, "mode": "pct", "value": 50}, "stop_loss": {"enabled": True, "mode": "pct", "value": 40}}
    s["entry"]["conditions"]["children"].append({"id": "dip", "kind": "cond", "type": "expr", "params": {"expr": "from_open_pct <= -0.5"}})
    s["entry"]["adds"] = [{"when": "price <= entry.underlying_price * 0.995", "quantity": 1, "max_adds": 1}, {"when": "price <= entry.underlying_price * 0.99", "quantity": 1, "max_adds": 1}]
    return s


def test_the_four_example_edits_change_only_the_selected_section():
    from strata.api import ide as IDE
    base = strat()
    r = IDE.edit_section(base, "exits.stop_loss", "Replace this fixed stop with an ATR-based stop.")
    assert r["applied"] == "atr_stop" and sorted(r["changed"]) == ["exits.stop_loss.enabled", "exits.underlying"]
    assert r["strategy"]["exits"]["stop_loss"]["enabled"] is False and r["strategy"]["exits"]["underlying"] == {"enabled": True, "stop_atr": 1.5, "atr_period": 14}
    assert r["strategy"]["exits"]["profit_target"] == base["exits"]["profit_target"] and r["strategy"]["entry"] == base["entry"] and r["strategy"]["legs"] == base["legs"]
    assert any("1.5 ATR is used" in n for n in r["notes"]) and any("fixed for the life of the position" in n for n in r["notes"])
    assert IDE.edit_section(base, "exits.stop_loss", "replace with a 2x ATR(20) stop")["strategy"]["exits"]["underlying"]["stop_atr"] == 2.0

    r = IDE.edit_section(base, "entry.conditions.children[1]", "Only allow this entry condition between 9:30 and 11:00 Eastern.")
    node = r["strategy"]["entry"]["conditions"]["children"][1]
    assert r["applied"] == "time_window" and r["changed"] == ["entry.conditions.children[1]"] and node["kind"] == "group" and node["children"][0] == base["entry"]["conditions"]["children"][1]
    assert node["children"][1]["params"] == {"start": "09:30", "end": "11:00"} and r["strategy"]["entry"]["conditions"]["children"][0] == base["entry"]["conditions"]["children"][0]

    r = IDE.edit_section(base, "entry", "Add a rule requiring QQQ confirmation.")
    added = r["strategy"]["entry"]["conditions"]["children"][-1]
    assert r["applied"] == "confirmation" and added["params"]["expr"] == "QQQ.from_open_pct > 0" and len(r["strategy"]["entry"]["conditions"]["children"]) == 3 and any("was not specified further" in n for n in r["notes"])
    assert IDE.edit_section(base, "entry", "Add a rule requiring QQQ to be above its VWAP for confirmation")["strategy"]["entry"]["conditions"]["children"][-1]["params"]["expr"] == "QQQ.price > QQQ.vwap"

    r = IDE.edit_section(base, "entry.adds", "Change the second scale-in to occur after a further 0.25% decline.")
    assert r["applied"] == "scale_in" and r["changed"] == ["entry.adds[1].text", "entry.adds[1].when"] and r["strategy"]["entry"]["adds"][1]["when"] == "price <= entry.underlying_price * 0.9925"
    assert r["strategy"]["entry"]["adds"][0] == base["entry"]["adds"][0] and any("measured from the scale-in before it" in n for n in r["notes"])
    assert base == strat(), "the document that was passed in is never modified"


def test_an_instruction_that_cannot_be_expressed_is_refused_not_approximated():
    from strata.api import ide as IDE
    base = strat()
    r = IDE.edit_section(base, "exits", "exit when the moon is full")
    assert r["applied"] is None and r["changed"] == [] and r["strategy"] == base and "Nothing was changed" in r["unsupported"][0]
    odd = copy.deepcopy(base)
    odd["entry"]["adds"][0]["when"] = "price <= vwap"
    with pytest.raises(ValueError, match="no defined starting point"):
        IDE.edit_section(odd, "entry.adds", "Change the second scale-in to occur after a further 0.25% decline.")
    with pytest.raises(ValueError, match="needs a distance"):
        IDE.edit_section(base, "exits", "trail the underlying price")
    r = IDE.edit_section(base, "exits", "take half off at 50% and move the stop to breakeven")
    assert r["strategy"]["exits"]["stages"] == [{"name": "partial at 50%", "profit_pct": 50.0, "close_pct": 50.0, "then_stop": "breakeven"}]


def test_validation_reports_the_path_of_every_problem():
    from strata.api import ide as IDE
    assert IDE.validate(strat())["ok"]
    bad = strat()
    bad["entry"]["conditions"]["children"][1]["params"]["expr"] = "from_open_pct <= "
    bad["exits"]["stages"] = [{"profit_pct": 40, "when": "price > 1"}]
    bad["sizing"] = {"method": "expr", "expr": "position.pnl"}
    v = IDE.validate(bad)
    paths = {e["path"] for e in v["errors"]}
    assert not v["ok"] and "entry.conditions.children[1].params.expr" in paths and any(p.startswith("exits.stages") for p in paths) and any("cannot read entry" in e["message"] for e in v["errors"])
    assert [x["path"] for x in IDE.validate(strat())["expressions"]] == ["entry.conditions.children[1].params.expr", "entry.adds[0].when", "entry.adds[1].when"]


def test_the_debugger_replays_a_session_with_watches_and_a_breakpoint(tmp_path):
    from strata.api import ide as IDE
    st = setup(tmp_path, ramp([(0, 100.0), (25, 99.0), (389, 99.0)]))  # down 1% by 09:55
    s = call(1)
    s["entry"]["conditions"]["children"].append({"id": "dip", "kind": "cond", "type": "expr", "params": {"expr": "from_open_pct <= -0.5"}})
    d = IDE.debug_session(s, D1, st, watch=["from_open_pct", "price"], break_when="from_open_pct <= -0.9")
    assert d["signal"]["time"] == "10:00" and "fired at 10:00" in d["summary"] and d["evaluations"] == 31, "one evaluation a minute from 09:30 until the day's single entry fired"
    by = {t["time"]: t for t in d["timeline"]}
    assert by["09:30"]["watch"]["from_open_pct"] == 0.0 and abs(by["10:00"]["watch"]["from_open_pct"] + 1.0) < 1e-6 and by["10:00"]["passed"] is True and by["09:45"]["passed"] is False
    dip = next(i for i in by["09:45"]["items"] if "from_open_pct" in i["text"])
    assert dip["passed"] is True and "from_open_pct=-0.56" in dip["text"], "each condition's value at the chosen tick is shown"
    bp = d["breakpoint"]
    assert bp["hit"] and bp["time"] == "09:54" and by["09:54"]["break"] is True, "the first tick on which the breakpoint expression holds"
    assert IDE.debug_session(s, D1, st, break_when="from_open_pct <= -9")["breakpoint"]["hit"] is False
    with pytest.raises(ValueError):
        IDE.debug_session(s, D1, st, watch=["nonsense("])
    assert IDE.debug_session(s, date(2024, 3, 9), st)["trading_day"] is False


def test_a_rule_preview_counts_history_and_explains_what_it_cannot_show(tmp_path):
    from strata.api import ide as IDE
    from tests.test_v4_research import rich_store
    st = rich_store(tmp_path, sessions=500, intraday=0)
    p = IDE.preview_rule("TST", "previous_session.return_pct <= -1", "open", st)
    assert p["ok"] and p["kind"] == "condition" and 0 < p["true"] < p["evaluable"] and len(p["recent"]) == 5 and 0 < p["share"] < 0.5
    v = IDE.preview_rule("TST", "atr_pct(14)", "open", st)
    assert v["kind"] == "value" and v["p05"] < v["median"] < v["p95"]
    assert IDE.preview_rule("TST", "price >", "open", st)["ok"] is False
    assert IDE.preview_rule("TST", "position.pnl_pct > 10", "open", st)["position"] is True
    assert "No price history" in IDE.preview_rule("TST", "ZZZQ.change_pct > 0", "open", st)["error"]


def test_onboarding_reflects_what_is_actually_stored(tmp_settings):
    from strata.api.v5_api import onboarding
    import json
    empty = json.loads(onboarding().body)
    assert empty["has_data"] is False and empty["suggestions"] == [] and empty["steps"][0]["done"] is False and "can be downloaded for free" in empty["steps"][0]["detail"] and empty["mode"] == "simple"
    assert {u["id"] for u in empty["unlock"]} == {"github_eod", "alpaca", "massive", "databento"} and all(u["status"] != "verified" for u in empty["unlock"])
    from tests.test_v4_research import rich_store
    rich_store(tmp_settings, sessions=900, intraday=0)
    from strata.api.settings import set_setting
    set_setting("data.default_symbol", "TST")
    full = json.loads(onboarding().body)
    assert full["has_data"] and full["steps"][0]["done"] and full["suggestions"][0]["ready"] and "TST" in full["suggestions"][0]["text"]
    assert not next(s for s in full["suggestions"] if "afternoon" in s["text"])["ready"], "a question that needs intraday bars is not offered as ready when none are stored"
    assert full["steps"][2]["done"] is False and "No option prices are stored" in full["steps"][2]["detail"]


# ----------------------------------------------------------------------------- reading a strategy description
def test_times_written_without_a_zone_share_one_zone(tmp_settings, monkeypatch):
    """"Enter at 10:00, out by 15:30" must never be half Pacific and half Eastern, whatever the default zone is."""
    from strata.ai import nl_rules as NL
    for pref in (0, 60, 180):
        monkeypatch.setattr(NL, "default_tz_offset", lambda pref=pref: pref)
        ts = NL.find_times("Buy at 10:00 and exit by 15:30", None)
        assert [t["et"] for t in ts] == ["10:00", "15:30"] and {t["assumed_tz"] for t in ts} == {"Eastern"}
    monkeypatch.setattr(NL, "default_tz_offset", lambda: 180)
    assert [t["et"] for t in NL.find_times("in at 7:15 and out by 12:45", None)] == ["10:15", "15:45"]  # both fit Pacific, the user's zone
    assert [t["et"] for t in NL.find_times("enter at 10:00", None)] == ["13:00"]  # one time alone: the user's zone
    # a zone written once applies to the other time in the same request
    assert [(t["et"], t["assumed_tz"]) for t in NL.find_times("only between 9:30 and 11:00 Eastern", None)] == [("09:30", None), ("11:00", None)]


def test_a_strategy_description_uses_the_research_reader_for_conditions(tmp_settings):
    from strata.ai import nl_stateful as NS
    d = NS.enrich("Buy a 0DTE at-the-money SPY call when SPY is down more than 0.2% from the open and QQQ is weak too, between 10:00 and 15:00 ET. Take profit at 50%, stop at 30%, out by 15:30 ET.", ["SPY", "QQQ"], "SPY")
    assert [c.expr for c in d.conditions] == ["time >= hhmm(10, 0) and time <= hhmm(15, 0)", "from_open_pct <= -0.2", "QQQ.from_open_pct < 0"]
    assert d.unsupported == [] and d.entry_time is None  # a window and a move "from the open" are not an instruction to enter at 10:00 or at the open
    assert d.premium_target_pct == 50 and d.premium_stop_pct == 30 and d.underlying_target_pct is None  # 0.2% is a condition, not a target
    assert any("QQQ is weak too" in n for n in d.notes)  # the reading of a vague phrase is stated
    d = NS.enrich("Buy a 0DTE ATM SPY call at 10:00 ET when SPY is below its open. Take profit 50%, stop 30%.", ["SPY"], "SPY")
    assert [c.expr for c in d.conditions] == ["from_open_pct < 0"] and d.entry_time == "10:00" and d.unsupported == []


def test_the_rest_of_a_partly_read_clause_is_reported_not_dropped(tmp_settings):
    from strata.ai import nl_stateful as NS
    d = NS.enrich("Buy a 0DTE ATM SPY call at 10:00 ET when SPY gapped down 1% and the moon is full. Take profit 50%, stop 30%.", ["SPY"], "SPY")
    assert [c.expr for c in d.conditions] == ["gap_pct <= -1"] and d.unsupported == ["the moon is full"]
    # a statement about the session's own close cannot be evaluated at an intraday entry: refused, not approximated
    d = NS.enrich("Buy a 0DTE ATM SPY call at 10:00 ET when QQQ closes down 2% on the day. Take profit 50%, stop 30%.", ["SPY", "QQQ"], "SPY")
    assert d.conditions == [] and len(d.unsupported) == 1 and "QQQ" in d.unsupported[0]


def test_entry_conditions_refuse_what_cannot_be_known_at_the_entry(tmp_settings):
    from strata.ai import nl_graph as NG
    ec = lambda p, at: NG.entry_conditions(p, "SPY", ["SPY", "QQQ", "VIX"], None, at)  # noqa: E731
    assert [c["expr"] for c in ec("when QQQ is above its open", "10:30")] == ["QQQ.from_open_pct > 0"]
    assert [c["expr"] for c in ec("if SPY gapped down more than 1%", "open")] == ["gap_pct <= -1"]
    assert [c["expr"] for c in ec("when it is a Monday and VIX rose overnight", "open")] == ["weekday == 0", "VIX.gap_pct > 0"]
    assert ec("when SPY closes down 2%", "15:55") is None           # the close is not known at 15:55
    assert ec("when the first 30 minutes sold off", "11:00") is None  # a stretch that does not end at the entry
    assert ec("when the moon is full", "10:00") is None
    got, left = NG.entry_conditions("when SPY is below its open and the moon is full", "SPY", ["SPY"], None, "10:00", partial=True)
    assert [c["expr"] for c in got] == ["from_open_pct < 0"] and left == ["the moon is full"]


def test_the_compact_form_is_exact_and_keeps_what_the_strategy_does():
    from strata.api import ide as IDE
    from strata.engine.models import StrategyConfig
    from strata.engine.templates import examples
    for cfg in examples("SPY", False):
        full = StrategyConfig.model_validate(cfg).model_dump(mode="json")
        c = IDE.compact(cfg)
        assert IDE._no_rule_ids(StrategyConfig.model_validate(c).model_dump(mode="json")) == IDE._no_rule_ids(full)
        assert len(str(c)) < len(str(full)) * 0.6
        assert all({"side", "quantity"} <= set(leg) and ("dte" in leg or leg.get("kind") == "stock") for leg in c["legs"])  # essentials are shown even at their defaults
        assert all(v.get("enabled") is True for k, v in c["exits"].items() if isinstance(v, dict) and "enabled" in v and full["exits"][k]["enabled"])
        assert c["sizing"]["method"] == full["sizing"]["method"]
    v = IDE.validate(IDE.compact(next(iter(examples("SPY", False)))))
    assert v["ok"] and v["full"]["exits"]["trigger_price"] == "close_price"  # validation hands back the complete document
