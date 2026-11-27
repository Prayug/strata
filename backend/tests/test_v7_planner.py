"""V7: reading a request.

Three readers: the built-in ones (the sentences of the brief are pinned here), a language model filling a form of typed
fields (every value checked against the sentence, every expression written by the application), and the two together (the
built-in readers first, the model only for what they could not place). The models in this file are scripted: what the two
local models installed on the development machine did with sentences they had not seen is measured in docs/V7_REPORT.md."""
from __future__ import annotations

import json

import httpx
import pytest

from tests.test_v2_research import FakeLLM, InlineJobs, tmp_settings  # noqa: F401
from tests.test_v7_experiment import base_draft, grid_store

KNOWN = ["IWM", "QQQ", "SPY"]


def enrich(text):
    from strata.ai import nl_rules as NL
    from strata.ai.nl_stateful import enrich as E
    return E(text, NL.TICKERS)


def compiled(d) -> dict:
    from strata.ai.draft import compile_draft
    c = compile_draft(d)
    assert c["errors"] == [], c["errors"]
    return c["strategy"]


# ----------------------------------------------------------------------------- the built-in readers on the brief's sentences
def test_intraday_events_of_the_brief_are_rules_and_ordered_events_are_sequences():
    d = enrich("Buy QQQ puts when the first 15-minute candle breaks the previous day's low.")
    # the candle has to be complete before it can have broken anything: the rule is looked at from 09:45
    assert (d.symbol, d.structure, d.entry_time, [c.expr for c in d.conditions]) == ("QQQ", "long_put", "09:45", ["opening_range_low(15) < prev_low"])
    d = enrich("Buy SPY calls when price reclaims VWAP after declining 0.5% from the open.")
    assert [(s.name, s.expr) for s in d.sequence] == [("drop", "from_open_pct <= -0.5"), ("reclaim", "crosses_above(price, vwap)")] and d.conditions == [] and d.entry_time is None
    assert [s["when"] for s in compiled(d)["entry"]["sequence"]] == ["from_open_pct <= -0.5", "crosses_above(price, vwap)"]
    d = enrich("Buy 0DTE calls after three consecutive bearish five-minute candles followed by a bullish reversal.")
    assert d.dte == 0 and [s.expr for s in d.sequence] == ['consecutive_bearish_candles("5m") >= 3', 'consecutive_bullish_candles("5m") >= 1']
    d = enrich("Buy SPY 0DTE calls on the first green five-minute candle after SPY has dropped more than 1% from the open.")
    assert [s.expr for s in d.sequence] == ["from_open_pct <= -1", 'consecutive_bullish_candles("5m") >= 1']
    d = enrich("Buy QQQ 0DTE puts when price falls below the low of the first hour.")
    assert [c.expr for c in d.conditions] == ["price < opening_range_low(60)"] and d.symbol == "QQQ"
    d = enrich("Buy SPY 0DTE calls when SPY breaks above the first 30 minutes' high with QQQ confirmation.")
    assert [c.expr for c in d.conditions] == ["price > opening_range_high(30)", "QQQ.price > QQQ.opening_range_high(30)"] and d.unsupported == []


def test_minutes_held_entries_a_day_and_partial_exits_reach_the_engine_configuration():
    d = enrich("Buy 0DTE SPY calls at 10:00 AM Eastern, sell after 45 minutes.")
    assert (d.entry_time, d.hold_minutes, d.exit_time) == ("10:00", 45, None) and compiled(d)["exits"]["max_hold"] == {"enabled": True, "minutes": 45, "days": None}
    d = enrich("Buy SPY 0DTE puts up to three times a day when price crosses below VWAP; exit each after 20 minutes.")
    s = compiled(d)
    assert (d.max_entries, d.hold_minutes) == (3, 20) and s["entry"]["max_entries_per_day"] == 3 and s["entry"]["max_open_positions"] == 3 and [c.expr for c in d.conditions] == ["crosses_below(price, vwap)"]
    d = enrich("Buy one ATM SPY 0DTE call at 9:35 AM Eastern and sell half when the premium is up 25%.")
    st = compiled(d)["exits"]["stages"]
    assert [(p.premium_pct, p.fraction) for p in d.partials] == [(25.0, 0.5)] and d.premium_target_pct is None
    assert len(st) == 1 and st[0]["profit_pct"] == 25.0 and st[0]["close_pct"] == 50.0  # a percent of the premium, not of the underlying


def test_a_target_against_a_stop_changes_the_strategy_and_what_names_no_value_is_asked(tmp_settings):
    from strata.ai.agent import Agent
    ag = Agent(InlineJobs(), grid_store(tmp_settings), llm_factory=lambda: None)
    cur = base_draft()
    r = ag._rules("Test a 30% premium profit target versus a 15% premium stop.", cur, KNOWN, None, "SPY")
    assert r["action"] == "modify" and (r["draft"].premium_target_pct, r["draft"].premium_stop_pct) == (30.0, 15.0) and r["draft"].underlying_target_pct is None
    r = ag._rules("Test multiple entries and partial exits.", cur, KNOWN, None, "SPY")
    d = r.get("draft")
    assert r["action"] in ("unknown", "experiment_needs_base") or (d is not None and (d.unsupported or any(q.get("answered") is None for q in d.questions))), "how many entries and at what level was invented"


# ----------------------------------------------------------------------------- the form
def form(**kw) -> dict:
    from strata.ai import planner_form as PF
    return {**PF.EMPTY, **kw}


def sig(type_, **kw) -> dict:
    from strata.ai.planner_form import _sig
    return _sig(type_, **kw)


def route(text, current=None, tmp=None, **kw):
    from strata.ai import planner_form as PF
    from strata.data.store import MarketStore
    return PF.to_route(form(**kw), text, current, MarketStore(tmp / "market") if tmp else None, KNOWN, "SPY")


def test_the_form_becomes_the_same_draft_a_typed_strategy_becomes_and_the_application_writes_the_expressions(tmp_settings):
    r = route("Sell a 5-point wide SPY put credit spread at 30 delta, 7DTE, on Mondays at 10:15 AM Eastern; hold for 3 trading days or until price crosses below VWAP.", tmp=tmp_settings,
              symbol="SPY", position="put_credit_spread", dte=7, strike_type="delta", strike_value=30, spread_width=5, entry_time="10:15 AM Eastern", hold_days=3, exit_signals=[sig("crosses_below_vwap")])
    d = r["draft"]
    assert r["action"] == "new" and (d.structure, d.dte, d.strike.method, d.strike.value, d.width, d.entry_time, d.hold_sessions, d.weekdays) == ("bull_put_spread", 7, "delta", 0.3, 5.0, "10:15", 3, ["Mon"])
    assert [c.expr for c in d.exit_conditions] == ["crosses_below(price, vwap)"] and d.unsupported == [] and compiled(d)["symbol"] == "SPY"
    r = route("When QQQ falls 1.2% from its open and then crosses back above VWAP, buy weekly calls and sell them after 25 minutes.", tmp=tmp_settings,
              symbol="QQQ", position="buy_call", dte=7, signals=[sig("down_from_open", percent=1.2), sig("crosses_above_vwap")], signals_in_order=True, exit_after_minutes=25)
    d = r["draft"]
    assert [s.expr for s in d.sequence] == ["from_open_pct <= -1.2", "crosses_above(price, vwap)"] and (d.symbol, d.dte, d.hold_minutes) == ("QQQ", 7, 25)


def test_a_value_that_is_not_in_the_sentence_is_removed_and_said(tmp_settings):
    r = route("Buy QQQ 1DTE calls at 12:30 PM Pacific with a 50% premium stop.", tmp=tmp_settings, symbol="QQQ", position="buy_call", dte=1, entry_time="12:30 PM", premium_stop_percent=50,
              premium_target_percent=80, exit_after_minutes=30, contracts=4, signals=[sig("gap_down", percent=0.7)])
    d = r["draft"]
    # the clock is the sentence's, read in the sentence's zone; the stop is there; the target, the hold, the size and the gap were made up
    assert (d.entry_time, d.premium_stop_pct, d.premium_target_pct, d.hold_minutes, d.quantity, d.conditions) == ("15:30", 50.0, None, None, 1, [])
    said = " ".join(d.notes)
    assert "a premium target of 80" in said and "an exit after 30" in said and "a size of 4" in said and "not in your sentence" in said
    assert any("no words for it" in u for u in d.unsupported)  # a gap is not mentioned at all: that signal is refused, not merely stripped of its number
    # the symbol and the kind of position are the words' as well
    d = route("Buy SPY puts at the open", tmp=tmp_settings, symbol="TSLA", position="buy_call", entry_time="the open")["draft"]
    assert (d.symbol, d.structure, d.entry_time) == ("SPY", "long_put", "open") and "the words say long put" in " ".join(d.notes)


def test_the_bell_is_an_entry_time_only_when_the_sentence_says_to_enter_at_it(tmp_settings):
    d = route("Buy SPY calls when price reclaims VWAP after declining 0.5% from the open.", tmp=tmp_settings, symbol="SPY", position="buy_call", entry_time="the open",
              signals=[sig("crosses_above_vwap"), sig("down_from_open", percent=0.5)], signals_in_order=True)["draft"]
    assert d.entry_time is None and 'the entry time "the open"' in " ".join(d.notes)
    # "B after A": the words give the reclaim first, and the decline happens first
    assert [s.expr for s in d.sequence] == ["from_open_pct <= -0.5", "crosses_above(price, vwap)"]
    d = route("Buy SPY calls at the open after a red previous day.", tmp=tmp_settings, symbol="SPY", position="buy_call", entry_time="the open", signals=[sig("previous_day_down")], signals_in_order=True)["draft"]
    assert d.entry_time == "open" and [c.expr for c in d.conditions] == ["previous_session.return_pct < 0"] and d.sequence == []  # one signal is not a sequence


def test_a_signal_needs_its_words_and_a_number_needs_a_signal_that_takes_it(tmp_settings):
    d = route("Buy QQQ 0DTE puts when price falls below the low of the first hour.", tmp=tmp_settings, symbol="QQQ", position="buy_put", dte=0, signals=[sig("first_candle_breaks_previous_low", minutes=60)])["draft"]
    assert d.conditions == [] and any("first candle breaks yesterday's low" in u and "no words for it" in u for u in d.unsupported)  # the sentence never mentions yesterday
    d = route("Buy SPY 0DTE calls on the first green five-minute candle after SPY has dropped more than 1% from the open.", tmp=tmp_settings, symbol="SPY", position="buy_call", dte=0,
              signals=[sig("down_from_open", percent=1, minutes=5)])["draft"]
    assert [c.expr for c in d.conditions] == ["from_open_pct <= -1"] and any("takes no minutes" in u for u in d.unsupported)  # the five minutes belong to a candle the form left out
    # a percentage of the sentence that the form has no place for is a part that was not read
    d = route("Buy SPY calls when price reclaims VWAP after declining 0.5% from the open.", tmp=tmp_settings, symbol="SPY", position="buy_call", signals=[sig("crosses_above_vwap")])["draft"]
    assert any("0.5%" in u for u in d.unsupported)
    # a confirmation on another symbol holds together with the signal it confirms
    d = route("Buy SPY 0DTE calls when SPY breaks above the first 30 minutes' high with QQQ confirmation.", tmp=tmp_settings, symbol="SPY", position="buy_call", dte=0,
              signals=[sig("breaks_above_opening_range", minutes=30, on_symbol="QQQ")], signals_in_order=True)["draft"]
    assert [c.expr for c in d.conditions] == ["price > opening_range_high(30)", "QQQ.price > QQQ.opening_range_high(30)"] and d.sequence == []


def test_lists_and_ranges_are_arithmetic_and_one_number_has_one_role(tmp_settings):
    cur = base_draft()
    # the model wrote the range out wrongly: the values are the arithmetic reader's
    r = route("Compare entry times from 9:30 to 11:00 in 15-minute increments and 0DTE versus 1DTE.", cur, tmp_settings, request="comparison",
              compare=[{"setting": "entry_time", "values": ["9:30 AM", "10:00 AM", "11:00 AM"]}, {"setting": "dte", "values": ["0", "1"]}])
    assert r["action"] == "experiment" and r["base_from"] == "current" and r["configs"] == 14
    assert {d["field"]: d["values"] for d in r["dims"]} == {"entry_time": ["open", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00"], "dte": [0, 1]}
    # a setting the patterns do not know is the model's, and only with values it can read
    r = route("Try stops at a fifth and a third of the premium.", cur, tmp_settings, request="comparison", compare=[{"setting": "premium_stop", "values": ["20", "33"]}])
    assert r["dims"] == [{"field": "premium_stop", "values": [20.0, 33.0], "text": "", "assumed": False}]
    # a target and a stop are two settings: nothing is compared, the strategy is changed
    r = route("Test a 30% premium profit target versus a 15% premium stop.", cur, tmp_settings, request="comparison", premium_target_percent=30, premium_stop_percent=15,
              compare=[{"setting": "premium_target", "values": ["30%", "15%"]}])
    assert r["action"] == "modify" and (r["draft"].premium_target_pct, r["draft"].premium_stop_pct) == (30.0, 15.0) and set(r["changed"]) == {"premium_target_pct", "premium_stop_pct"}
    # a setting that is named without values gets the same assumption the built-in reader makes, and it is said
    r = route("Compare ATM and OTM contracts across multiple entry times.", cur, tmp_settings, request="comparison", compare=[{"setting": "strike", "values": ["atm", "otm"]}])
    assert {d["field"]: (d["values"], d["assumed"]) for d in r["dims"]} == {"strike": ([("atm", 0.0), ("pct_otm", 1.0)], True), "entry_time": (["open", "10:00", "11:00", "13:00"], True)}
    # with nothing to apply it to, a comparison asks
    r = route("Compare exits after 15, 30 and 60 minutes.", None, tmp_settings, request="comparison", compare=[{"setting": "exit_after_minutes", "values": ["15", "30", "60"]}])
    assert r["action"] == "experiment_needs_base" and "Which strategy" in r["question"]


def test_the_model_is_asked_once_under_a_schema_and_what_is_not_a_form_is_an_error(tmp_settings):
    from strata.ai import evalset_v7 as E7
    from strata.ai import planner_form as PF
    from strata.ai.llm import LLMError
    from strata.data.store import MarketStore
    st = MarketStore(tmp_settings / "market")
    seen = {}

    class Local(FakeLLM):
        provider = "ollama"

        def json(self, system, messages, schema=None, max_tokens=4000):
            seen.update(schema=schema, system=system, max_tokens=max_tokens)
            return super().json(system, messages, schema=schema, max_tokens=max_tokens)
    m = Local(objs=[form(symbol="SPY", position="buy_call", dte=0, entry_time="10:00 AM Eastern", exit_after_minutes=45)])
    r = PF.form_interpret(m, "Buy 0DTE SPY calls at 10:00 AM Eastern, sell after 45 minutes.", None, st, KNOWN)
    assert r["action"] == "new" and (r["draft"].entry_time, r["draft"].hold_minutes, r["draft"].dte) == ("10:00", 45, 0) and len(m.calls) == 1 and r["form"]["position"] == "buy_call"
    sch = seen["schema"]
    assert sch["required"] == list(sch["properties"]) and sch["additionalProperties"] is False and sch["properties"]["signals"]["items"]["properties"]["type"]["enum"] == list(PF.SIGNALS)
    # the form has no field that could hold a price, a statistic or a result, and no free expression
    assert not any(k in sch["properties"] for k in ("expr", "price", "result", "win_rate", "pnl", "code")) and "expr" not in sch["properties"]["signals"]["items"]["properties"]
    assert len(seen["system"]) < 12000 and set(PF.TAKES) <= set(PF.SIGNALS) and set(PF.ANCHOR) == set(PF.SIGNALS) == set(PF.SIGNAL_TEXT)
    for ty, (_text, build) in PF.SIGNALS.items():  # every signal writes an expression of the engine's language
        from strata.engine import expr as X
        assert X.validate(build({"percent": 1.0, "minutes": 15, "count": 2, "level": 20.0}, "")) == [], ty
    # the worked examples of the prompt are not sentences of either evaluation set, and the sealed set is as it was sealed
    texts = {c["text"] for c in E7.UNSEEN + E7.DEVELOPMENT}
    assert not any(t in texts for t, _f in PF.EXAMPLES) and E7.sealed_hash() == "7ed431778fa541f062d1881cc913b81f0d072f99420cf0c64f599ecacd858d22"
    assert PF.form_interpret(Local(objs=[form(request="other")]), "what is the weather", None, st, KNOWN)["action"] == "unknown"
    with pytest.raises(LLMError, match="did not return the form"):
        PF.form_interpret(Local(objs=[{"setting": "entry_time", "values": []}, {"oops": 1}]), "Buy SPY calls", None, st, KNOWN)


# ----------------------------------------------------------------------------- the two together
class Local(FakeLLM):
    provider, model, name = "ollama", "small", "ollama:small"


class Reader9(Local):
    """A local model that answers in V9's question language (strata.ai.model_reader). Scripted."""

    def __init__(self, readings):
        super().__init__(objs=list(readings))


def test_a_local_model_is_asked_only_about_what_the_built_in_readers_could_not_place(tmp_settings, monkeypatch):
    """V9 (replacing V7's "assist" mode): with a local model the built-in readers read first, and the model is asked only
    about a message they could not place completely; its reading is used only after every part of it has been checked."""
    from strata.ai.agent import Agent
    st = grid_store(tmp_settings)
    m = Reader9([])
    ag = Agent(InlineJobs(), st, llm_factory=lambda: m)
    r, who, note = ag.interpret(None, "Buy 0DTE SPY calls at 10:00 AM Eastern, sell after 45 minutes.", None, None, None)
    assert who == "built-in rules" and note is None and m.calls == [] and r["draft"].hold_minutes == 45  # read completely: the model was not called
    # the built-in readers place nothing: the model reads, and its checked reading is used, and the reader says so
    text = "Buy SPY 0DTE calls at 10:00 AM Eastern when SPY is 0.4% under its open, sell after 45 minutes."
    monkeypatch.setattr(ag, "_rules", lambda *a, **k: {"action": "unknown", "text": "?"})
    m._objs = [{"act": "strategy", "kind": "strategy", "symbol": {"v": "SPY", "q": "SPY"}, "conds": [{"k": "from_open", "q": "0.4% under its open", "op": "<=", "v": -0.4}],
                "strat": {"struct": "long_call", "q": "Buy SPY 0DTE calls", "dte": {"q": "0DTE"}, "entry": {"q": "at 10:00 AM Eastern"}, "hold": {"q": "after 45 minutes"}}}]
    r, who, note = ag.interpret(None, text, None, None, None)
    assert "ollama:small" in who and len(m.calls) == 1
    assert [c.expr for c in r["draft"].conditions] == ["from_open_pct <= -0.4"] and r["draft"].hold_minutes == 45 and r["draft"].entry_time == "10:00"
    # a model reading that leaves part of the message unplaced is not used: the built-in reply stands
    m._objs = [{"act": "strategy", "kind": "strategy", "symbol": {"v": "SPY", "q": "SPY"}, "strat": {"struct": "long_call", "q": "Buy SPY 0DTE calls", "entry": {"q": "at 10:00 AM Eastern"}}}]
    r, who, note = ag.interpret(None, text, None, None, None)
    assert who == "built-in rules" and r["action"] == "unknown" and len(m.calls) == 2


def test_the_reading_order_is_a_setting_and_a_command_is_never_read_as_a_strategy(tmp_settings):
    """V9: llm.mode = rules (built-in readers only) | model_first | rules_first | auto (model first for a hosted model, the
    built-in readers first for a local one). A command the model names goes to the command, never to a strategy."""
    from strata.ai.agent import Agent
    from strata.api.settings import set_setting
    st = grid_store(tmp_settings)
    m = Reader9([{"act": "command", "cmd": "about_selection", "filter": "losing"}])
    ag = Agent(InlineJobs(), st, llm_factory=lambda: m)
    set_setting("llm.mode", "model_first")
    # a follow-up the deterministic edit layer places is answered there, without the model (it kept every measured context)
    r, who, _ = ag.interpret(None, "show me the losing trades", base_draft(), None, None)
    assert r["action"] == "show_trades" and len(m.calls) == 0
    # one it does not place goes to the model, whose command is followed (and is never read as a strategy)
    r, who, _ = ag.interpret(None, "anything noteworthy among the losers?", base_draft(), None, None)
    assert r["action"] == "about_selection" and len(m.calls) == 1
    set_setting("llm.mode", "rules")
    r, who, _ = ag.interpret(None, "anything noteworthy among the losers?", base_draft(), None, None)
    assert who == "built-in rules" and len(m.calls) == 1, "rules: the model is not called"


def test_a_local_model_is_asked_for_a_bounded_context(tmp_settings, monkeypatch):
    from strata.ai import llm as L
    from strata.api.settings import set_setting
    bodies = []

    class R:
        status_code = 200

        def json(self):
            return {"message": {"content": json.dumps({"ok": True})}, "prompt_eval_count": 10, "eval_count": 3}
    monkeypatch.setattr(httpx, "request", lambda method, url, **kw: (bodies.append((url, kw["json"])), R())[1])
    assert L.Ollama("http://127.0.0.1:11434", "small").json("s", [{"role": "user", "content": "x"}], schema={"type": "object"}) == {"ok": True}
    url, body = bodies[0]
    # left to the server the context is the model's maximum: 128k tokens for a 6 GB model is 13 GB of memory on this machine
    assert url.endswith("/api/chat") and body["options"]["num_ctx"] == 12288 and body["options"]["temperature"] == 0 and body["format"] == {"type": "object"} and body["stream"] is False
    set_setting("llm.ollama.context_tokens", 4096)
    L.Ollama("http://127.0.0.1:11434", "small").text("s", [{"role": "user", "content": "x"}])
    assert bodies[1][1]["options"]["num_ctx"] == 4096 and "format" not in bodies[1][1]


# ----------------------------------------------------------------------------- two readers that have to agree
def test_two_readers_must_agree_or_the_difference_is_asked_before_anything_runs(tmp_settings):
    """V9: the model reads first (model_first); when the deterministic reader also reads every part of the sentence and the
    two readings differ, the user is asked which was meant, both shown; nothing runs on a disagreement."""
    from strata.ai.agent import Agent
    from strata.api.settings import set_setting
    st = grid_store(tmp_settings)
    set_setting("llm.mode", "model_first")
    text = "How often does SPY close above its open after a down day?"
    # "a down day" is read by the deterministic reader as the close against the previous close
    same = {"act": "ask", "kind": "frequency", "symbol": {"v": "SPY", "q": "SPY"}, "conds": [{"k": "prev_return", "q": "a down day", "op": "<", "v": 0}],
            "out": {"k": "close_vs", "q": "close above its open", "ref": "open", "dir": "up"}}
    m = Reader9([same])
    ag = Agent(InlineJobs(), st, llm_factory=lambda: m)
    r, who, note = ag.interpret(None, text, None, None, None)
    assert who == "ollama:small, confirmed by the built-in reader" and r["action"] in ("new", "investigate") and len(m.calls) == 1
    other = {**same, "conds": [{"k": "prev_otc", "q": "a down day", "op": "<", "v": 0}]}  # the same words, as the day's own candle
    m._objs = [other]
    r, who, note = ag.interpret(None, text, None, None, None)
    assert r["action"] == "clarify" and len(r["options"]) == 2 and r.get("pending"), "two complete readings differ: asked, not run"

