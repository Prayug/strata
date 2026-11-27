"""V9: a language model reads first and every part of its reading is checked; answers are recalculated independently;
position sizing in a trader's terms; questions about selected observations; what to investigate next.

The model here is a stand-in that returns fixed readings, so what is tested is the checking around it (grounding,
coverage, the cross-check, edits), not any model's accuracy. Accuracy is measured by strata.ai.eval_v9 on the evaluation
sets, and the sealed V9 set is only checked here for not having changed.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from strata.ai import model_reader as MR
from strata.ai import semantics as SM
from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture)
from tests.test_v7_questions import DAYS, store  # noqa: F401  (fixture: 80 TST sessions)

KNOWN = ["SPY", "QQQ", "IWM", "TST", "VIX"]


class FakeModel:
    """Returns the next reading from a list (one per call). Records what it was shown."""
    provider, model = "fake", "fixed-readings"
    purpose = "read"
    last_usage = None
    cancel = None

    def __init__(self, *readings):
        self.readings = list(readings)
        self.seen = []

    @property
    def name(self):
        return "fake:fixed-readings"

    def json(self, system, messages, schema=None, max_tokens=900):
        self.seen.append(json.loads(messages[-1]["content"]))
        return self.readings.pop(0) if self.readings else {"act": "unclear", "unclear": {"q": "", "ask": "?"}}


# ----------------------------------------------------------------------------- the sealed set
def test_the_sealed_v9_set_is_the_one_that_was_written():
    from strata.ai import evalset_v9 as E
    assert E.sealed_hash() == "868db97d358ffaadcca113cc37d48e37adc318237a87c28a5d0bd2bec389d1e9", "the sealed V9 set must not change after it was written"
    assert len(E.SEALED) == 62 and len(E.SEALED_CONVERSATIONS) == 7 and len(E.SEALED_ANSWERS) == 12


# ----------------------------------------------------------------------------- grounding: the words, not the model's numbers
def test_numbers_come_from_the_quoted_words_and_unquoted_parts_are_dropped():
    MR.reset()
    raw = {"act": "ask", "kind": "frequency", "symbol": {"v": "QQQ", "q": "qqq"},
           "conds": [{"k": "gap", "q": "gaps down 1.5%", "op": "<=", "v": -1.5}, {"k": "vix", "q": "vix above 30", "op": ">", "v": 30}],  # the second is not in the message
           "out": {"k": "close_vs", "q": "close green", "ref": "open", "dir": "up"}, "period": {"q": "since 2020"}}
    text = "how often does qqq close green after it gaps down 1.5% since 2020"
    f, problems, quotes = MR.question_frame(raw, text, KNOWN, "SPY")
    assert not problems
    assert [(c["key"], c["op"], c["value"]) for c in f["conditions"]] == [("gap", "<=", -1.5)]
    assert f["outcome"]["ref"] == "open" and f["outcome"]["direction"] == "up" and f["period"][0] == "2020-01-01"
    assert MR.dropped() == ["vix above 30"], "a part whose words are not in the message is not used"


def test_a_count_is_not_a_percentage_and_a_part_must_name_what_it_is_about():
    MR.reset()
    text = "after 3 red days in a row how often is spy green the next day"
    raw = {"act": "ask", "kind": "frequency", "conds": [{"k": "prev_return", "q": "3 red days in a row", "op": "<=", "v": -3}],
           "out": {"k": "close_vs", "q": "green the next day", "ref": "prev_close", "dir": "up"}}
    f, problems, _ = MR.question_frame(raw, text, KNOWN, "SPY")
    assert problems and not f["conditions"], "three days in a row is a streak, never a 3% decline"
    raw = {"act": "ask", "kind": "frequency", "conds": [{"k": "prev_otc", "q": "vix is under 15", "op": "<=", "v": -15}], "out": {"k": "close_vs", "q": "close higher", "dir": "up"}}
    f, problems, _ = MR.question_frame(raw, "how often does spy close higher when vix is under 15", KNOWN, "SPY")
    assert any("VIX" in p for p in problems)


def test_coverage_finds_a_clause_no_part_holds():
    text = "how often does iwm close above its open on mondays after a 2% down day"
    left = MR.coverage(text, ["iwm", "close above its open", "a 2% down day"], KNOWN)
    assert any("monday" in x.lower() for x in left), left
    assert not MR.coverage(text, ["iwm", "close above its open", "on mondays", "a 2% down day"], KNOWN)


def test_times_are_read_by_the_deterministic_clock_including_zones():
    MR.reset()
    text = "how often does spy hit +0.3% between 6:30 and 7:30 pacific"
    raw = {"act": "ask", "kind": "frequency", "symbol": {"v": "SPY", "q": "spy"}, "out": {"k": "touch", "q": "hit +0.3%", "v": 0.3}, "start": {"q": "6:30"}, "end": {"q": "7:30 pacific"}}
    f, problems, _ = MR.question_frame(raw, text, KNOWN, "SPY")
    assert not problems and f["start"] == "open" and f["end"] == "10:30" and f["outcome"]["value"] == 0.3


# ----------------------------------------------------------------------------- the agent: model first, checked
def test_a_checked_model_reading_runs_and_says_which_reader_read_it(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    m = FakeModel({"act": "ask", "kind": "frequency", "symbol": {"v": "TST", "q": "TST"}, "conds": [{"k": "weekday", "q": "on mondays", "day": "Mon"}],
                   "out": {"k": "close_vs", "q": "close above its open", "ref": "open", "dir": "up"}})
    ag = agent(store, m)
    sid = session()
    out = ag.post(sid, "how often does TST close above its open on mondays")
    plan = next(x for x in out if x["kind"] in ("plan", "run"))
    assert "fake:fixed-readings" in (plan["content"].get("interpreter") or plan["content"].get("reader") or json.dumps(plan["content"]))
    _, v, d = ag._state(sid)
    assert d.weekdays == ["Mon"] and "open" in json.dumps(d.study.model_dump())
    assert m.seen and m.seen[0]["message"] == "how often does TST close above its open on mondays"


def test_two_complete_readings_that_differ_are_asked_about_not_run(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    # "a down day": the deterministic reader reads the close against the previous close; this model, the day's own candle
    m = FakeModel({"act": "ask", "kind": "frequency", "symbol": {"v": "TST", "q": "TST"}, "conds": [{"k": "prev_otc", "q": "a down day", "op": "<", "v": 0}],
                   "out": {"k": "close_vs", "q": "close above its open", "ref": "open", "dir": "up"}})
    ag = agent(store, m)
    sid = session()
    out = ag.post(sid, "how often does TST close above its open after a down day")
    first = next(x for x in out if x["role"] == "assistant")
    assert first["content"].get("route") == "clarify" and len(first["content"].get("options") or []) == 2
    _, v, _ = ag._state(sid)
    assert v is None, "nothing runs on a disagreement"


def test_readings_that_differ_only_in_an_ambiguous_outcome_are_both_measured(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    # "close higher": against the previous close (the deterministic reader) or the open (this model): both are measured, and said
    m = FakeModel({"act": "ask", "kind": "frequency", "symbol": {"v": "TST", "q": "TST"}, "conds": [{"k": "gap", "q": "gaps up 1%", "op": ">=", "v": 1}],
                   "out": {"k": "close_vs", "q": "close higher", "ref": "open", "dir": "up"}})
    ag = agent(store, m)
    sid = session()
    out = ag.post(sid, "how often does TST close higher after it gaps up 1%")
    inv = next(x for x in out if x["kind"] == "investigation")
    assert len(inv["content"]["plan"]["variants"]) == 2 and any("both are measured" in n for n in inv["content"]["plan"].get("interpretation_notes") or [])


def test_an_edit_changes_only_what_the_message_quotes(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    m = FakeModel({"act": "ask", "kind": "frequency", "symbol": {"v": "TST", "q": "TST"}, "conds": [{"k": "gap", "q": "gaps down 1%", "op": "<=", "v": -1}],
                   "out": {"k": "close_vs", "q": "close above its open", "ref": "open", "dir": "up"}},
                  {"act": "edit", "conds": [{"k": "gap", "q": "2%", "op": "<=", "v": -2}]})
    ag = agent(store, m)
    sid = session()
    ag.post(sid, "how often does TST close above its open after it gaps down 1%")
    out = ag.post(sid, "make it gaps down 2%")
    change = next((x["content"]["change"] for x in out if (x.get("content") or {}).get("change")), None)
    assert change and "2%" in change["text"] and "Everything else stays the same" in change["text"]
    _, st = ag._v8_state() if hasattr(ag, "_v8_state") else (None, {})
    f = st.get("frame") or {}
    assert [(c["key"], c["value"]) for c in f.get("conditions") or []] == [("gap", -2.0)] and f["outcome"]["ref"] == "open"


def test_an_unknown_ticker_is_declined_with_the_closest_question_offered(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    m = FakeModel({"act": "strategy", "kind": "strategy", "symbol": {"v": "NVDA", "q": "nvda"}, "strat": {"struct": "long_call", "q": "buy nvda calls", "entry": {"q": "at the open"}}})
    ag = agent(store, m)
    sid = session()
    out = ag.post(sid, "buy nvda calls at the open")
    c = next(x for x in out if x["role"] == "assistant")["content"]
    assert c.get("declined") and "NVDA" in c["text"] and any(o.get("ask") for o in c.get("options") or [])


def test_a_model_cannot_declare_something_unanswerable_without_a_reason_the_code_can_see(tmp_settings, store):  # noqa: F811
    from tests.test_v3_agent import agent, session
    m = FakeModel({"act": "unsupported", "unsupported": {"q": "every morning", "why": "no data on mornings", "closest": "something else"}})
    ag = agent(store, m)
    sid = session()
    out = ag.post(sid, "how often does TST close above its open")
    c = next(x for x in out if x["role"] == "assistant")["content"]
    assert not c.get("declined"), "the built-in readers read it instead"


# ----------------------------------------------------------------------------- independent recalculation
def test_the_reference_agrees_with_the_study_engine_on_the_fixture(tmp_settings, store):  # noqa: F811
    from strata.ai.draft import compile_draft
    from strata.research.reference import Reference
    from strata.research.study import run_study
    R = Reference(tmp_settings / "market" / "datasets")
    for text in ("How often does TST close above its open?", "How often does TST rise 0.2% from the open before 11:00 ET?",
                 "how often does TST close above its open after the first 15 minute candle is green"):
        r = SM.read(text, KNOWN, "TST")
        d = SM.to_route(r)["draft"]
        res = run_study(compile_draft(d, store)["study"], store, max_sessions_out=9000)
        ref = R.check_frame(r.frame)
        assert ref is not None, text
        assert (res["stats"]["qualifying_sessions"], res["stats"]["target_first"]) == (ref["n"], ref["k"]), text


def test_a_session_horizon_ends_at_the_official_close(tmp_settings, store):  # noqa: F811
    """The last minute bar of a session can differ from the official close; the study ends at the official close."""
    import polars as pl
    from strata.ai.draft import compile_draft
    from strata.research.study import run_study
    d0 = DAYS[10]
    daily = store.read_daily("TST")
    row = daily.filter(pl.col("date") == d0)
    o = float(row["open"][0])
    # the official close one cent below the open: the session closed down, whatever the last bar says
    daily = daily.with_columns(pl.when(pl.col("date") == d0).then(pl.lit(o - 0.01)).otherwise(pl.col("close")).alias("close"))
    store.write_daily("TST", daily)
    r = SM.read("How often does TST close above its open?", KNOWN, "TST")
    res = run_study(compile_draft(SM.to_route(r)["draft"], store)["study"], store, max_sessions_out=9000)
    s = next(x for x in res["sessions"] if x["date"] == d0.isoformat())
    assert s["outcome"] != "target"


def test_a_vix_condition_is_the_previous_close_the_reading_names():
    e, label = SM.cond_expr({"key": "vix", "op": ">", "value": 25.0})
    assert e == "vix_prev_close > 25" and "the session before" in label
    e, _ = SM.cond_expr({"key": "vix", "op": ">", "value": 25.0, "when": "now"})
    assert e == "vix > 25"


# ----------------------------------------------------------------------------- position sizing
def test_sizing_in_a_traders_terms_compiles_onto_the_engine():
    from strata.ai.draft import Draft, DraftSizing, compile_draft, sizing_config, sizing_text
    d = Draft(kind="options", symbol="TST", structure="long_call", dte=0, entry_time="10:00", exit_time="15:30", premium_stop_pct=50,
              sizing=DraftSizing(mode="risk_dollars", value=300, capital=10000, daily_loss_stop=200))
    cfg = sizing_config(d)
    assert cfg["method"] == "risk_dollars" and cfg["dollars"] == 300 and cfg["risk_basis"] == "stop" and cfg["daily_loss_stop"] == 200
    assert "the loss if the 50% stop" in sizing_text(d) and "$10,000" in sizing_text(d)
    comp = compile_draft(d)
    assert comp["strategy"]["sizing"]["method"] == "risk_dollars" and all(l["quantity"] == 1 for l in comp["strategy"]["legs"])
    assert any(i["field"] == "sizing" for i in comp["items"])
    d2 = Draft(kind="options", symbol="TST", structure="long_put", dte=0, sizing=DraftSizing(mode="dollars", value=500))
    assert sizing_config(d2)["method"] == "premium_dollars"
    assert sizing_config(Draft(kind="options", symbol="TST", structure="long_put", quantity=3))["method"] == "fixed"


def test_the_engine_sizes_by_premium_and_by_risk_at_the_stop():
    from types import SimpleNamespace
    from strata.engine.models import Execution, Sizing
    from strata.engine.rules import RuleStrategy
    s = RuleStrategy.__new__(RuleStrategy)
    s.allocation, s.idx = 1.0, 0
    acct = SimpleNamespace(equity=lambda: 10_000.0, closed=[])

    class Ch:  # one 0 DTE call priced at 2.00 (bid = ask: a trade print)
        bid = ask = mid = [2.0]
        is_call = [True]
        strike = [100.0]
    sel = [SimpleNamespace(row=0, leg=SimpleNamespace(sign=1, quantity=1), expiration=0)]
    ctx = SimpleNamespace(account=acct, chain=lambda: (Ch(), 0), price=lambda: 100.0, today=date(2024, 3, 4),
                          bt=SimpleNamespace(metas={"TST": SimpleNamespace(multiplier=100.0)}, ex=Execution(), cfg=SimpleNamespace(capital=SimpleNamespace(margin_model="regt", initial_capital=10_000.0))))
    s.symbol = "TST"

    def units(**kw):
        s.cfg = SimpleNamespace(sizing=Sizing(**kw), exits=SimpleNamespace(stop_loss=SimpleNamespace(enabled=True, mode="pct", value=50.0)))
        return s._units(ctx, sel)
    assert units(method="premium_dollars", dollars=500)[0] == 2  # one contract costs 200
    assert units(method="premium_dollars", dollars=150)[0] == 0
    assert units(method="risk_dollars", dollars=300, risk_basis="stop")[0] == 3  # 100 lost at the 50% stop
    assert units(method="risk_dollars", dollars=300, risk_basis="max_loss")[0] == 1  # 200 if the whole premium is lost
    assert units(method="risk_pct", pct=2, risk_basis="stop")[0] == 2  # 2% of 10,000


def test_a_daily_loss_stop_blocks_new_entries_that_day():
    from types import SimpleNamespace
    from strata.core import calendar as cal
    from strata.engine.models import Sizing
    from strata.engine.rules import RuleStrategy
    s = RuleStrategy.__new__(RuleStrategy)
    s.allocation, s.idx = 1.0, 0
    day = date(2024, 3, 4)
    t0 = int(cal.session_bounds(day)[0].timestamp())
    lost = SimpleNamespace(strategy_idx=0, exit_ts=t0 + 600, pnl_close=lambda: -250.0)
    ctx = SimpleNamespace(account=SimpleNamespace(equity=lambda: 10_000.0, closed=[lost]), today=day)
    s.cfg = SimpleNamespace(sizing=Sizing(method="fixed", contracts=1, daily_loss_stop=200), exits=None)
    n, why = s._units(ctx, [])
    assert n == 0 and "daily loss stop" in why


# ----------------------------------------------------------------------------- what sets a group apart
def test_a_contrast_reports_what_random_splits_give_and_how_many_things_were_compared():
    from strata.research.contrast import compare
    g = [{"gap": 1.0 + 0.01 * i, "weekday": "Mon"} for i in range(12)]
    r = [{"gap": 0.0 + 0.01 * i, "weekday": "Tue"} for i in range(15)]
    out = compare(g, r, "trades", "the losing trades", "the others")
    text = " ".join(out["lines"])
    assert "random" in text.lower() and "things known at the entry were compared" in text and "probability that" not in text
    assert out["rows"][0]["p"] < 0.01


# ----------------------------------------------------------------------------- the hosted path, without calling anyone
def test_the_hosted_reader_asks_for_the_schema_caches_the_instructions_and_never_reads_a_refusal(monkeypatch):
    """No paid call: a stand-in for the Anthropic SDK client records what the adapter sends."""
    from types import SimpleNamespace
    import anthropic
    from strata.ai import llm as L
    sent = []

    class Messages:
        def __init__(self, reply):
            self.reply = reply

        def create(self, **kw):
            sent.append(kw)
            return self.reply

    reading = {"act": "ask", "kind": "frequency", "symbol": {"v": "SPY", "q": "SPY"}, "out": {"k": "close_vs", "q": "close green", "dir": "up"}}
    ok = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(reading))], stop_reason="end_turn", usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    class Client:
        def __init__(self, **kw):
            self.beta = SimpleNamespace(messages=Messages(ok))
            self.messages = Messages(ok)
    monkeypatch.setattr(anthropic, "Anthropic", Client)
    m = L.Anthropic("sk-test-not-a-key", "claude-opus-5-5")
    raw = MR.ask_model(m, "how often does SPY close green", MR.context_of(None, None, None, [], None))
    assert raw == reading
    kw = sent[-1]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["format"]["schema"] == MR.SCHEMA and kw["cache_control"] == {"type": "ephemeral"}
    assert kw["fallbacks"] == "default" and "sk-test-not-a-key" not in json.dumps(kw["messages"]) + kw["system"]
    refused = SimpleNamespace(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="other"), usage=SimpleNamespace(input_tokens=1, output_tokens=0))
    m.client.beta.messages.reply = refused
    with pytest.raises(L.LLMError):
        MR.ask_model(m, "how often does SPY close green", MR.context_of(None, None, None, [], None))
