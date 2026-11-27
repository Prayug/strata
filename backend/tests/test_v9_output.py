"""V9, found in the sealed run's output after it was scored (neither changes how a message is read):

- the data preview of a strategy kept the FIRST 4,000 sessions its entry fires on, so a rule firing on more (an entry at
  the open on SPY's daily history since 1993) was previewed on 1993 to 2008 only and said no session could be priced;
- a configured model rephrased the result, and the number check let it attach true numbers to the wrong quantities
  ("closed above its open in 2,741 sessions out of 2,741"); the result's text is now always the calculated one.
"""
from __future__ import annotations

import functools

from tests.test_v2_research import tmp_settings  # noqa: F401  (fixture: a temporary data directory)
from tests.test_v7_readiness import DAYS, M_IN, READY, config, store  # noqa: F401  (fixture: SPY sessions with stored option prints)


def test_the_preview_counts_every_session_the_rule_fires_on_not_the_first_few(store, monkeypatch):  # noqa: F811
    from strata.data import readiness as RD
    from strata.data.providers import market as MK
    cfg = config(DAYS[0], DAYS[-1])
    full = MK.needs_of(cfg, store)
    capped = MK.needs_of(cfg, store, max_signals=3)  # the rule fires on 12 sessions: more than the cap
    assert full["signal_sessions"] == capped["signal_sessions"] == len(DAYS)
    assert capped["first"] == DAYS[0].isoformat() and capped["last"] == DAYS[-1].isoformat() and capped["signals"] == full["signals"]
    assert M_IN.isoformat() in capped["days"], "a session after the cap is still previewed"
    whole = RD.precheck(cfg, store)
    monkeypatch.setattr(MK, "needs_of", functools.partial(MK.needs_of, max_signals=3))
    small = RD.precheck(cfg, store)
    assert small["signal_sessions"] == whole["signal_sessions"] == len(DAYS) and small["counts"] == whole["counts"]
    assert small["priced"] == whole["priced"] >= 1 and small["text"] == whole["text"]
    assert {r["date"]: r["status"] for r in small["sessions"]}[READY.isoformat()] == "trades"


def test_a_configured_model_does_not_rewrite_the_result(tmp_settings):  # noqa: F811
    from strata.ai.agent import Agent
    from strata.api import db
    from strata.api.settings import set_setting
    from tests.test_v2_research import FakeLLM, InlineJobs, study_store
    st = study_store(tmp_settings)
    set_setting("llm.mode", "rules")  # the built-in readers read the message; the model is only offered the result
    # every number below is one of the result's (2 sessions, 100%), attached to the wrong quantity: it passed the number check
    fake = FakeLLM("TST rose 1% from the open on 2 of the 2 sessions, 100.0% of them, which is a strong and reliable pattern.")
    ag = Agent(InlineJobs(), st, llm_factory=lambda: fake)
    sid = db.create_session("t")["id"]
    ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "single" and res["explanation_source"] == "deterministic"
    assert "1 of the 2 sessions" in res["text"] and "2 of the 2" not in res["text"] and "reliable" not in res["text"]
    assert not any("FACTS" in json_of(c) for c in fake.calls), "the result was not sent to the model to rephrase"
    # why: the number check alone accepts that text (each number is one of the result's), so it cannot be the safeguard
    from strata.ai import explain as EX
    from strata.api.results import load_run_result
    run = db.get_run(res["run_id"])
    assert EX.explain_run(run, load_run_result(run), fake)["source"] == "fake:m"


def json_of(call) -> str:
    system, messages = call
    return system + " ".join(str(m.get("content")) for m in messages)
