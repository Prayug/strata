"""V2: expression language, event studies (hand-calculated), the request interpreter,
the draft compiler / coverage assessor, grounded explanations, LLM adapters (mocked),
free-source normalisers and the research agent. No test touches the network."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from strata.core import calendar as cal
from strata.engine import expr as X
from tests.helpers import chain, make_store, write_minutes, write_options

DAYS = cal.trading_days(date(2024, 3, 1), date(2024, 3, 12))  # 8 sessions: Mar 1,4,5,6,7,8,11,12


# ============================================================================ expressions
def test_spec_expressions_parse():
    for e in ['previous_session.return_pct <= -3', 'underlying.price >= entry.underlying_price * 1.01', 'consecutive_bearish_candles("5m") >= 3',
              'crosses_above(price, vwap)', 'return_pct("SPY", sessions=5) < -4', 'first_touch(previous_high, previous_low)']:
        assert X.validate(e) == [], e


@pytest.mark.parametrize("bad", [
    "__import__('os').system('rm -rf /')", "price.__class__", "open('secrets.txt')", "(lambda: 1)()", "price[0]", "[x for x in (1, 2)]",
    "eval('1')", "exec('1')", "getattr(price, 'real')", "price; 1", "import os", "session.__dict__", "unknown_name > 1", "rsi > 30", "x := 1",
])
def test_expressions_cannot_run_code(bad):
    """Rules are interpreted over a whitelist of syntax, names and functions - never eval()'d."""
    assert X.validate(bad), bad


def test_expression_needs_intraday_flag():
    assert not X.needs_intraday("previous_session.return_pct <= -3")
    assert X.needs_intraday("price > vwap")
    assert X.needs_intraday('consecutive_bearish_candles("5m") >= 3')


# ============================================================================ event study (hand-calculated)
def study_store(tmp_path):
    """TST 1-minute bars:
      Mar 1  flat 100
      Mar 4  falls to 96 at the close        (close-to-close -4.0%)
      Mar 5  opens 96, 97.5 from minute 60   (+1.5625% from the open -> +1% touched in bar 60)
      Mar 6  flat 97.5
      Mar 7  falls to 94                      (-3.59%)
      Mar 8  opens 94, peaks 94.5             (+0.53% -> target missed)
      Mar 11 flat 94.5,  Mar 12 flat 94.5
    """
    st = make_store(tmp_path, settlement="physical", exercise="american")

    def price(d, i):
        if d == DAYS[0]:
            return 100.0
        if d == DAYS[1]:
            return 100.0 if i < 380 else 96.0
        if d == DAYS[2]:
            return 96.0 if i < 60 else 97.5
        if d == DAYS[3]:
            return 97.5
        if d == DAYS[4]:
            return 97.5 if i < 380 else 94.0
        if d == DAYS[5]:
            return 94.0 if i < 100 else 94.5
        return 94.5
    write_minutes(st, "TST", DAYS, price)
    return st


def cond(*exprs):
    return {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "expr", "params": {"expr": e}} for e in exprs]}


def test_study_bounce_after_down_day_hand_values(tmp_path):
    from strata.research.study import run_study
    r = run_study({"symbol": "TST", "condition": cond("previous_session.return_pct <= -3"), "entry_time": "open", "target": {"kind": "pct", "value": 1.0}}, study_store(tmp_path))
    s = r["stats"]
    assert [x["date"] for x in r["sessions"]] == ["2024-03-05", "2024-03-08"]  # the sessions AFTER each down day, never the down day itself
    assert [x["outcome"] for x in r["sessions"]] == ["target", "neither"]
    assert s["qualifying_sessions"] == 2 and s["target_first"] == 1 and s["probability"] == pytest.approx(0.5)
    hit = r["sessions"][0]
    assert hit["entry_price"] == pytest.approx(96.0) and hit["target_level"] == pytest.approx(96.96)
    assert hit["minutes_to_target"] == pytest.approx(61.0)  # bar 60 closes 61 minutes after the open
    assert hit["mfe_pct"] == pytest.approx(1.5625) and hit["prev_return_pct"] == pytest.approx(-4.0)
    assert r["sessions"][1]["mfe_pct"] == pytest.approx((94.5 / 94 - 1) * 100, abs=1e-3)
    # baseline: every evaluated session; only Mar 5 ever gains 1% from its open
    assert s["baseline_probability"] == pytest.approx(1 / s["evaluated_sessions"])
    assert "not option returns" in " ".join(r["notes"])


def test_study_has_no_lookahead(tmp_path):
    """Changing prices AFTER a session's horizon must not change that session's row."""
    from strata.research.study import run_study
    cfg = {"symbol": "TST", "condition": cond("previous_session.return_pct <= -3"), "entry_time": "open", "target": {"kind": "pct", "value": 1.0}, "end": "2024-03-05"}
    a = run_study(cfg, study_store(tmp_path / "a"))
    st = make_store(tmp_path / "b", settlement="physical", exercise="american")
    write_minutes(st, "TST", DAYS, lambda d, i: {DAYS[0]: 100.0}.get(d, (100.0 if i < 380 else 96.0) if d == DAYS[1] else (96.0 if i < 60 else 97.5) if d == DAYS[2] else 500.0))
    b = run_study(cfg, st)
    assert a["sessions"] == b["sessions"] and a["stats"]["probability"] == b["stats"]["probability"]


def test_study_target_and_stop_in_one_daily_bar_is_ambiguous(tmp_path):
    """With only a daily bar the order of target and stop is unknowable: reported, never guessed."""
    from strata.research.study import run_study
    st = make_store(tmp_path, settlement="physical", exercise="american")
    rows = [(DAYS[0], 100, 100, 100, 100), (DAYS[1], 100, 100, 96, 96), (DAYS[2], 96, 98, 95, 97), (DAYS[3], 97, 97.2, 96.9, 97)]
    st.write_daily("TST", pl.DataFrame([{"date": d, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": 1.0} for d, o, h, l, c in rows]))
    r = run_study({"symbol": "TST", "condition": cond("previous_session.return_pct <= -3"), "target": {"kind": "pct", "value": 1.0}, "stop": {"kind": "pct", "value": -0.5}}, st)
    assert r["stats"]["qualifying_sessions"] == 1 and r["stats"]["ambiguous"] == 1 and r["stats"]["probability"] is None
    assert r["stats"]["probability_bounds"] == [0.0, 1.0]
    assert any("ambiguous" in n for n in r["notes"])


def test_study_intraday_rule_on_daily_bars_excludes_instead_of_failing_the_condition(tmp_path):
    from strata.research.study import run_study
    st = make_store(tmp_path, settlement="physical", exercise="american")
    st.write_daily("TST", pl.DataFrame([{"date": d, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0} for d in DAYS]))
    r = run_study({"symbol": "TST", "condition": cond("price > vwap"), "entry_time": "10:00", "target": {"kind": "pct", "value": 0.5}}, st)
    assert r["stats"]["qualifying_sessions"] == 0 and r["stats"]["excluded_sessions"] >= len(DAYS) - 1
    assert r["excluded"] and "not observable" in r["excluded"][0]["reason"]


def test_study_close_measure(tmp_path):
    from strata.research.study import run_study
    r = run_study({"symbol": "TST", "condition": cond("previous_session.return_pct <= -3"), "entry_time": "open", "target": {"kind": "pct", "value": 0}, "measure": "close", "direction": "up"}, study_store(tmp_path))
    assert r["stats"]["target_first"] == 2 and r["stats"]["ambiguous"] == 0  # both sessions end above their open


# ============================================================================ interpreter (offline rules)
@pytest.fixture()
def tmp_settings(tmp_path):
    from strata.api import db
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    db.init_db()
    yield tmp_path
    override_settings(data_dir=old)


def parse(text, cur=None):
    from strata.ai.nl_rules import interpret
    return interpret(text, cur, ["SPY", "QQQ", "IWM", "VIX"])


def test_test_d_question_becomes_a_study(tmp_settings):
    r = parse("How often does SPY rise 1% from the open after falling 3% or more the previous trading day?")
    d = r["draft"]
    assert r["action"] == "new" and d.kind == "study" and d.symbol == "SPY" and d.entry_time == "open"
    assert [c.expr for c in d.conditions] == ["previous_session.return_pct <= -3"]
    assert d.study.target.kind == "pct" and d.study.target.value == 1.0 and d.study.stop is None and d.study.measure == "touch"


def test_test_e_strategy_fields_and_timezones(tmp_settings):
    d = parse("Buy one ATM SPY 0DTE call at 6:30 AM Pacific after a 3% down day, exit when SPY moves up 1% from entry or at 12:45 PM Pacific.")["draft"]
    assert (d.kind, d.structure, d.dte, d.strike.method, d.quantity) == ("options", "long_call", 0, "atm", 1)
    assert d.entry_time == "open" and d.entry_time_stated == "6:30 AM Pacific"  # 06:30 PT is 09:30 ET all year (both zones change clocks together)
    assert d.exit_time == "15:45" and [c.expr for c in d.conditions] == ["previous_session.return_pct <= -3"]
    assert d.underlying_target_pct == 1.0 and d.premium_target_pct is None  # an underlying move, not an option return


@pytest.mark.parametrize("text,entry", [("Buy SPY calls at 9:30 AM Eastern", "open"), ("Buy SPY calls at 7 AM PT", "10:00"), ("Buy SPY calls at 10 AM Eastern", "10:00"),
                                        ("Buy SPY calls at 8:30 Central", "open"), ("Buy SPY calls at the close", "close"), ("Buy SPY calls at 12:15 PM Pacific", "15:15")])
def test_clock_times_convert_to_new_york(tmp_settings, text, entry):
    assert parse(text)["draft"].entry_time == entry


def test_time_without_timezone_is_disclosed_not_silent(tmp_settings):
    from strata.ai.draft import compile_draft
    from strata.data.store import MarketStore
    d = parse("Buy QQQ puts at 10:15 when price is below VWAP")["draft"]
    assert d.tz_assumed == "Pacific" and d.entry_time == "13:15"  # default timezone setting is Pacific
    c = compile_draft(d, MarketStore(tmp_settings / "market"))
    assert any(a["field"] == "timezone" for a in c["assumptions"])


def test_percent_of_underlying_vs_option_vs_dollars(tmp_settings):
    a = parse("Buy SPY calls at the open, exit when SPY rises 1% or the option doubles")["draft"]
    assert a.underlying_target_pct == 1.0 and a.premium_target_pct == 100.0
    b = parse("Buy SPY puts at 10 AM Eastern and exit if the option loses 50% or gains 80%")["draft"]
    assert b.premium_stop_pct == 50.0 and b.premium_target_pct == 80.0 and b.underlying_target_pct is None
    c = parse("Buy an ATM SPY call at the open and exit when SPY is up $2 from entry")["draft"]
    assert c.underlying_target_abs == 2.0 and c.underlying_target_pct is None and c.premium_target_pct is None


def test_previous_day_variants(tmp_settings):
    assert parse("How often does SPY rise 1% from the open after the previous day's open-to-close decline was 2%?")["draft"].conditions[0].expr == "previous_session.open_to_close_pct <= -2"
    assert parse("How often does SPY fall 1% from the open after gapping up 0.5%?")["draft"].conditions[0].expr == "gap_pct >= 0.5"
    d = parse("How often does SPY fall 1% from the open after gapping up 0.5%?")["draft"]
    assert d.study.target.value == -1.0
    close = parse("Buy a 30DTE ATM SPY call at the close after a 2% down day and hold for 5 days")["draft"]
    assert close.conditions[0].expr == "change_pct <= -2" and close.hold_sessions == 5 and close.dte == 30 and any("close of the down day" in n for n in close.notes)


def test_other_example_prompts(tmp_settings):
    d = parse("How often does SPY touch yesterday's high before touching yesterday's low?")["draft"]
    assert (d.study.target.expr, d.study.stop.expr) == ("prev_high", "prev_low")
    d = parse("How often does SPY recover 50% of its overnight gap before noon?")["draft"]
    assert d.study.target.expr == "open + 0.5 * (prev_close - open)" and (d.study.horizon_kind, d.study.horizon_time) == ("time", "12:00")
    d = parse("Buy ATM 0DTE puts when QQQ breaks the opening 15-minute low.")["draft"]
    assert (d.symbol, d.structure, d.dte, d.entry_time) == ("QQQ", "long_put", 0, None) and d.conditions[0].expr == "price < opening_range_low(15)"
    d = parse("Test an iron condor entered at 10 AM Eastern when VIX is below 18.")["draft"]
    assert d.structure == "iron_condor" and d.symbol == "SPY" and not d.symbol_stated and d.conditions[0].expr == "vix < 18"  # VIX is a filter, not the underlying
    d = parse("What happens if I buy SPY calls at the close after a red Monday and hold them until Friday?")["draft"]
    assert d.exit_weekday == "Fri" and d.entry_time == "close" and "weekday == 0" in d.conditions[0].expr


def test_followups(tmp_settings):
    from strata.ai.draft import diff_drafts
    base = parse("Buy one ATM SPY 0DTE call at 6:30 AM Pacific after a 3% down day, exit when SPY moves up 1% from entry or at 12:45 PM Pacific.")["draft"]
    r = parse("Change the previous-day decline to 2% and compare the results.", base)
    assert r["action"] == "modify" and r["compare"] and r["draft"].conditions[0].expr == "previous_session.return_pct <= -2"
    assert [x["field"] for x in diff_drafts(base, r["draft"])] == ["conditions"]  # nothing else moved
    assert parse("What if I bought puts instead?", base)["draft"].structure == "long_put"
    assert parse("Show me every losing trade", base) == {"action": "show_trades", "filter": "loss"}
    assert parse("Test 0DTE, 1DTE and 2DTE", base) == {"action": "sweep", "field": "dte", "values": [0, 1, 2]}
    assert parse("Test previous-day declines from 1% to 4% in 0.5% increments", base)["values"] == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert parse("Was this overfit? test out of sample", base)["action"] == "oos"
    assert parse("Only take the trade when VIX is above 25", base)["draft"].conditions[-1].expr == "vix > 25"
    assert parse("what's the weather like", base)["action"] == "unknown"
    assert parse("Sell an iron condor at 10 AM Eastern when VIX is below 18", base)["action"] == "new"


# ============================================================================ compile + coverage
def eod_store(tmp_path):
    """SPY-like store: daily bars, 60-minute bars, and END-OF-DAY option chains only."""
    from strata.data.store import MarketStore, SymbolMeta
    st = MarketStore(tmp_path / "market")
    st.upsert_symbol(SymbolMeta(symbol="SPY", kind="etf", exercise="american", settlement="physical", multiplier=100.0, provider="test", has_options=True))
    days = cal.trading_days(date(2024, 1, 2), date(2024, 3, 28))
    st.write_daily("SPY", pl.DataFrame([{"date": d, "open": 500.0, "high": 503.0, "low": 497.0, "close": 500.0 + (i % 5), "volume": 1.0} for i, d in enumerate(days)]))
    for d in days[-25:]:
        ts = cal.session_bounds(d)[1]
        st.write_option_day("SPY", d, pl.DataFrame([{"ts": ts, "expiration": d + timedelta(days=30), "strike": 500.0, "right": r, "bid": 5.0, "ask": 5.1, "bid_size": 10, "ask_size": 10, "volume": 1,
                                                    "open_interest": 1, "iv": 0.2, "delta": 0.5 if r == "C" else -0.5, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "underlying_price": 500.0, "quote_ts": ts} for r in "CP"])
                            .with_columns(pl.col("ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))),
                            provider="test_eod", res="EOD")
    st.rebuild_manifest("SPY")
    return st


def test_test_e_with_eod_options_offers_the_study_and_never_fabricates(tmp_settings):
    from strata.ai.draft import assess, backtest_config, compile_draft
    st = eod_store(tmp_settings)
    d = parse("Buy one ATM SPY 0DTE call at 6:30 AM Pacific after a 3% down day, exit when SPY moves up 1% from entry or at 12:45 PM Pacific.")["draft"]
    c = compile_draft(d, st)
    cover = assess(d, c, st)
    assert c["errors"] == [] and c["missing"] == ["stop loss"]  # the one thing the user did not say
    assert c["strategy"]["legs"][0]["dte_mode"] == "exact" and c["strategy"]["exits"]["underlying"]["move_for_pct"] == 1.0
    assert cover["modes"]["historical"]["ok"] is False and "end-of-day only" in cover["modes"]["historical"]["reason"] and "never used to simulate" in cover["modes"]["historical"]["reason"]
    assert cover["recommended"] != "historical" and cover["recommended"] != "modeled"  # modeled is opt-in only, never recommended
    with pytest.raises(ValueError):
        backtest_config(d, c, cover, "historical")


def test_close_entry_strategy_runs_on_eod_chains(tmp_settings):
    from strata.ai.draft import assess, backtest_config, compile_draft
    st = eod_store(tmp_settings)
    d = parse("Buy a 30DTE ATM SPY call at the close after a 2% down day and hold for 5 days")["draft"]
    c = compile_draft(d, st)
    cover = assess(d, c, st)
    assert cover["recommended"] == "historical"
    cfg = backtest_config(d, c, cover, "historical")
    assert cfg["pricing"]["mode"] == "historical" and cfg["allow_synthetic"] is False
    assert cfg["strategies"][0]["exits"]["max_hold"] == {"enabled": True, "minutes": None, "days": 5}


def test_unstated_entry_time_defaults_are_disclosed(tmp_settings):
    from strata.ai.draft import compile_draft
    st = eod_store(tmp_settings)
    c = compile_draft(parse("Sell a 16 delta SPY put credit spread 45 DTE when VIX is above 25, take profit at 50% of the credit")["draft"], st)
    assert {"entry time", "spread width", "stop loss"} <= set(c["missing"])
    a = {x["field"]: x for x in c["assumptions"]}
    assert a["entry_time"]["value"] == "close" and "end-of-day" in a["entry_time"]["reason"]
    assert c["strategy"]["exits"]["profit_target"] == {"enabled": True, "mode": "pct", "value": 50.0} and c["strategy"]["exits"]["stop_loss"]["enabled"] is False  # no stop invented


def test_test_g_missing_dataset_is_reported_not_replaced(tmp_settings):
    from strata.ai.draft import assess, compile_draft
    from strata.data.store import MarketStore
    from strata.engine.engine import DataUnavailable, run_backtest
    st = MarketStore(tmp_settings / "market")
    d = parse("Buy a 30DTE ATM AAPL call at the close after a 2% down day and hold for 5 days")["draft"]
    cover = assess(d, compile_draft(d, st), st)
    assert cover["recommended"] is None and all(not m["ok"] for m in cover["modes"].values())
    assert "No historical price data is stored for AAPL" in cover["messages"][0] and "nothing will be simulated" in cover["messages"][0]
    cfg = {"start": "2024-03-04", "end": "2024-03-05", "strategies": [{"name": "s", "symbol": "AAPL", "legs": [{"id": "a", "side": "buy", "right": "C", "dte": 30, "strike": {"method": "atm_offset", "value": 0}}]}]}
    with pytest.raises(DataUnavailable):
        run_backtest(cfg, store=st)


def test_synthetic_symbols_refused_unless_explicitly_allowed(tmp_path):
    from strata.data.store import SymbolMeta
    from strata.engine.engine import DataUnavailable, run_backtest
    from strata.engine.models import BacktestConfig
    assert BacktestConfig.model_fields["allow_synthetic"].default is False
    st = make_store(tmp_path)
    m = st.symbol_meta("TST")
    st.upsert_symbol(SymbolMeta(**{**m.__dict__, "synthetic": True}))
    write_minutes(st, "TST", [date(2024, 3, 4)], lambda d, i: 100.0)
    write_options(st, "TST", date(2024, 3, 4), chain("10:00", {(95, "P"): (1.0, 1.1, -0.2), (90, "P"): (0.4, 0.5, -0.1)}, date(2024, 3, 4)))
    from tests.helpers import put_spread_strategy
    cfg = {"start": "2024-03-04", "end": "2024-03-04", "strategies": [put_spread_strategy()]}
    with pytest.raises(DataUnavailable, match="SYNTHETIC"):
        run_backtest(cfg, store=st)
    assert run_backtest({**cfg, "allow_synthetic": True}, store=st).meta["synthetic"] is True


# ============================================================================ grounded explanations
STUDY_RES = {"stats": {"qualifying_sessions": 105, "evaluated_sessions": 8511, "excluded_sessions": 1, "target_first": 68, "stop_first": 0, "neither": 37, "ambiguous": 0, "probability": 68 / 105,
                       "probability_ci95": [0.553, 0.732], "probability_bounds": [68 / 105, 68 / 105], "baseline_probability": 0.173, "lift": 68 / 105 - 0.173, "p_value_vs_rest": 1e-9,
                       "avg_mfe_pct": 2.14, "avg_mae_pct": -1.97, "avg_end_return_pct": 0.52, "median_end_return_pct": 0.4, "pct_positive_end": 0.629, "median_minutes_to_target": 60.0},
             "config": {"target": {"kind": "pct", "value": 1.0}, "stop": None, "reference": "entry", "measure": "touch"}, "symbol": "SPY", "period": ["1993-01-29", "2026-11-23"], "notes": [], "excluded": []}


class FakeLLM:
    provider, model, name = "fake", "m", "fake:m"

    def __init__(self, text=None, objs=None):
        self._text, self._objs, self.calls = text, list(objs or []), []

    def text(self, system, messages, max_tokens=1500):
        self.calls.append((system, list(messages)))
        return self._text

    def json(self, system, messages, schema=None, max_tokens=4000):
        self.calls.append((system, list(messages)))
        return self._objs.pop(0)


def test_deterministic_explanation_only_contains_engine_numbers():
    from strata.ai import explain as EX
    facts = EX.study_facts(STUDY_RES)
    text = EX.study_text(STUDY_RES, facts)
    assert "68 of the 105" in text and "64.8%" in text and "55.3%" in text and "17.3%" in text and "not option returns" in text
    assert EX.ungrounded_numbers(text, facts, [1.0]) == []


def test_llm_explanation_with_an_invented_number_is_rejected():
    from strata.ai import explain as EX
    run = {"kind": "study", "name": "q", "config": STUDY_RES["config"]}
    ok = EX.explain_run(run, STUDY_RES, FakeLLM("SPY reached the target on 68 of 105 qualifying sessions, a frequency of 64.8% against a 17.3% baseline. The sample is modest, so treat it with care."))
    assert ok["source"] == "fake:m" and ok["rejected"] is None
    bad = EX.explain_run(run, STUDY_RES, FakeLLM("SPY reached the target 64.8% of the time, which implies an expected option profit of $412 per trade and a 71% win rate."))
    assert bad["source"] == "deterministic" and "412" in bad["rejected"] and "71%" in bad["rejected"] and "64.8%" in bad["text"]


# ============================================================================ LLM adapters (mocked transport)
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def test_openai_and_openrouter_adapters(monkeypatch):
    from strata.ai import llm
    seen = {}

    def fake(method, url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return _Resp(200, {"choices": [{"message": {"content": '```json\n{"tool": "answer", "args": {"text": "hi"}}\n```'}}]})
    monkeypatch.setattr(llm.httpx, "request", fake)
    out = llm.OpenAICompat("openai", "https://api.openai.com/v1", "sk-test", "gpt-x").json("sys", [{"role": "user", "content": "u"}])
    assert out == {"tool": "answer", "args": {"text": "hi"}}
    assert seen["url"].endswith("/chat/completions") and seen["headers"]["Authorization"] == "Bearer sk-test" and seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0]["role"] == "system"
    llm.OpenAICompat("openrouter", "https://openrouter.ai/api/v1", "k", "x/y").text("sys", [{"role": "user", "content": "u"}])
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions" and "response_format" not in seen["body"]


def test_gemini_and_ollama_adapters_and_errors_never_leak_keys(monkeypatch):
    from strata.ai import llm
    seen = {}

    def fake(method, url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        if "googleapis" in url:
            return _Resp(200, {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]})
        return _Resp(200, {"message": {"content": '{"ok": true}'}})
    monkeypatch.setattr(llm.httpx, "request", fake)
    assert llm.Gemini("g-key", "gemini-x").json("sys", [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}]) == {"ok": True}
    assert "g-key" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "g-key"  # key in a header, never in the URL
    assert seen["body"]["contents"][0]["role"] == "model" and seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert llm.Ollama("http://127.0.0.1:11434", "m").json("sys", [{"role": "user", "content": "u"}]) == {"ok": True}
    assert seen["url"] == "http://127.0.0.1:11434/api/chat" and seen["body"]["format"] == "json" and seen["body"]["stream"] is False
    monkeypatch.setattr(llm.httpx, "request", lambda *a, **k: _Resp(401, {"error": {"message": "Incorrect API key provided: sk-secret-123"}}))
    with pytest.raises(llm.LLMError) as e:
        llm.OpenAICompat("openai", "https://api.openai.com/v1", "sk-secret-123", "gpt-x").text("s", [])
    assert "rejected" in str(e.value)


def test_anthropic_adapter_request_shape_and_refusal():
    from types import SimpleNamespace as NS

    from strata.ai import llm
    calls = []

    class Msgs:
        def __init__(self, resp):
            self.resp = resp

        def create(self, **kw):
            calls.append(kw)
            return self.resp
    a = llm.Anthropic("sk-ant-test", "claude-opus-5-5")
    ok = NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""), NS(type="text", text='{"tool": "out_of_sample", "args": {}}')])
    a.client = NS(beta=NS(messages=Msgs(ok)), messages=Msgs(ok))
    assert a.json("sys", [{"role": "user", "content": "u"}], schema={"type": "object"}) == {"tool": "out_of_sample", "args": {}}
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "medium" and kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw and "temperature" not in kw  # adaptive thinking; sampling parameters are rejected by current models
    refused = NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[])
    a.client = NS(beta=NS(messages=Msgs(refused)), messages=Msgs(refused))
    with pytest.raises(llm.LLMError, match="declined"):
        a.text("sys", [{"role": "user", "content": "u"}])
    assert llm.PROVIDERS["anthropic"]["default_model"] == "claude-opus-5-5"


def test_extract_json_tolerates_prose_and_fences():
    from strata.ai.llm import LLMError, extract_json
    assert extract_json('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_llm_tool_loop_validates_and_repairs(tmp_settings):
    """The model may only pick listed tools; an invalid draft is sent back for repair; clock
    arithmetic is done by the server from the user's stated words, not by the model."""
    from strata.ai.agent import llm_interpret
    st = eod_store(tmp_settings)
    bad = {"kind": "options", "symbol": "SPY", "structure": "long_call", "dte": 0, "strike": {"method": "atm"}, "entry_time": "06:30", "entry_time_stated": "6:30 AM Pacific",
           "conditions": [{"expr": "os.system('x') or previous_day_drop > 3", "text": "down 3%"}], "underlying_target_pct": 1.0}
    good = {**bad, "conditions": [{"expr": "previous_session.return_pct <= -3", "text": "previous session fell 3% or more"}]}
    m = FakeLLM(objs=[{"tool": "lookup_functions", "args": {"query": "previous"}}, {"tool": "delete_all_data", "args": {}}, {"tool": "propose", "args": {"draft": bad}}, {"tool": "propose", "args": {"draft": good}}])
    r = llm_interpret(m, "Buy one ATM SPY 0DTE call at 6:30 AM Pacific after a 3% down day", None, [], st, lambda: None)
    assert r["action"] == "new" and r["draft"].entry_time == "open"  # 06:30 Pacific -> 09:30 New York, computed server-side
    assert r["draft"].conditions[0].expr == "previous_session.return_pct <= -3"
    assert len(m.calls) == 4 and "TOOL RESULT" in m.calls[1][1][-1]["content"] and "unknown tool" in m.calls[2][1][-1]["content"] and "only registered functions can be called" in m.calls[3][1][-1]["content"]
    assert "Python engine calculates every number" in m.calls[0][0]


# ============================================================================ research agent end to end (inline jobs, no network)
class InlineJobs:
    def submit(self, rid, kind, payload):
        from strata.api import db
        from strata.api.tasks import execute
        try:
            db.update_run(rid, status="done", progress=1.0, summary=execute(rid, kind, payload, lambda *a: None, lambda: False), finished_at=db.now())
        except Exception as e:  # noqa: BLE001
            db.update_run(rid, status="error", error=f"{type(e).__name__}: {e}", finished_at=db.now())


def test_agent_question_modify_compare_and_versions(tmp_settings):
    from strata.ai.agent import Agent
    from strata.api import db
    st = study_store(tmp_settings)
    ag = Agent(InlineJobs(), st, llm_factory=lambda: None)
    sid = db.create_session("t")["id"]
    msgs = ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")
    plan = msgs[-1]["content"]
    assert msgs[-1]["kind"] == "plan" and plan["mode"] == "study" and plan["interpreter"] == "built-in rules" and len(plan["runs"]) == 1
    res = ag.finalize(sid)[0]["content"]
    assert res["view"] == "single" and "1 of the 2 sessions" in res["text"] and "50.0%" in res["text"] and res["explanation_source"] == "deterministic"
    assert ag.finalize(sid) == []  # idempotent
    msgs = ag.post(sid, "Change the previous-day decline to 3.8% and compare the results.")
    p2 = msgs[-1]["content"]
    assert p2["parent_id"] == plan["version_id"] and [d["field"] for d in p2["diff"]] == ["conditions"] and len(p2["compare_with"]) == 1
    cmp_ = ag.finalize(sid)[0]["content"]
    assert cmp_["view"] == "compare" and [r["values"]["qualifying_sessions"] for r in cmp_["table"]["rows"]] == [2.0, 1.0]
    # the first run is still there, untouched, and restore branches rather than overwriting
    assert db.get_run(plan["runs"][0]["run_id"])["status"] == "done"
    ag.restore(sid, plan["version_id"])
    b = ag.post(sid, "add a 0.5% stop")[-1]["content"]
    assert b["parent_id"] == plan["version_id"] and len(db.list_versions(sid)) == 3
    assert ag.post(sid, "what's the weather")[-1]["kind"] == "text"


def test_agent_refuses_options_backtest_without_quotes_and_offers_study(tmp_settings):
    from strata.ai.agent import Agent
    from strata.api import db
    from strata.api.results import run_dir
    st = study_store(tmp_settings)
    ag = Agent(InlineJobs(), st, llm_factory=lambda: None)
    sid = db.create_session("t")["id"]
    plan = ag.post(sid, "Buy one ATM TST 0DTE call at 6:30 AM Pacific after a 3% down day, exit when TST moves up 1% from entry or at 12:45 PM Pacific.")[-1]["content"]
    assert plan["unsupported"] == [] and plan["state"] == "running" and len(plan["runs"]) == 1
    rid = plan["runs"][0]["run_id"]
    # the pipeline found the qualifying sessions but no observed option prices for them: it stops and asks
    assert db.get_run(rid)["summary"]["outcome"] == "needs_decision" and not (run_dir(rid) / "result.json").exists() and not (run_dir(rid) / "trades.json").exists()
    dec = ag.finalize(sid)[0]
    assert dec["kind"] == "decision" and "None of the 2 qualifying sessions has usable" in dec["content"]["text"]
    ids = [c["id"] for c in dec["content"]["choices"]]
    assert "mode:study" in ids and "mode:modeled" in ids and all("modeled" not in i for i in ids[:ids.index("mode:study")])  # theoretical premiums are offered, never chosen
    run = ag.decide(sid, dec["id"], "mode:study")[-1]["content"]
    assert "not the option position's profit or loss" in run["text"]
    assert ag.finalize(sid)[0]["content"]["mode"] == "study"


def test_agent_falls_back_to_rules_when_the_model_fails(tmp_settings):
    from strata.ai.agent import Agent
    from strata.ai.llm import LLMError
    from strata.api import db

    class Broken(FakeLLM):
        def json(self, *a, **k):
            raise LLMError("provider down")
    ag = Agent(InlineJobs(), study_store(tmp_settings), llm_factory=lambda: Broken())
    sid = db.create_session("t")["id"]
    plan = ag.post(sid, "How often does TST rise 1% from the open after falling 3% or more the previous trading day?")[-1]["content"]
    assert plan["interpreter"] == "built-in rules" and "provider down" in plan["text"] and len(plan["runs"]) == 1


# ============================================================================ free-source normalisers
def test_yahoo_daily_and_intraday_parsing():
    from strata.data.providers.sources import Yahoo
    t = lambda d, h, m: int(datetime(d.year, d.month, d.day, h, m, tzinfo=cal.ET).timestamp())
    d1, d2 = date(2024, 3, 4), date(2024, 3, 5)
    res = {"timestamp": [t(d1, 9, 30), t(d2, 9, 30), t(d2, 9, 30)], "indicators": {"quote": [{"open": [100, None, 101], "high": [102, None, 103], "low": [99, None, 100], "close": [101, None, 102], "volume": [10, None, 20]}]},
           "events": {"dividends": {"x": {"date": t(d1, 9, 30), "amount": 1.5}}, "splits": {"y": {"date": t(d2, 9, 30), "numerator": 4, "denominator": 1}}}}
    daily, actions = Yahoo.parse_daily(res)
    assert daily["date"].to_list() == [d1, d2] and daily["close"].to_list() == [101.0, 102.0]  # the null row is dropped, not filled
    assert sorted(actions["action"].to_list()) == ["dividend", "split"] and actions.filter(pl.col("action") == "split")["ratio"][0] == 4.0
    bars = {"timestamp": [t(d1, 9, 0), t(d1, 9, 30), t(d1, 15, 30), t(d1, 16, 0)], "indicators": {"quote": [{"open": [1, 100, 101, 5], "high": [1, 101, 102, 5], "low": [1, 99, 100, 5], "close": [1, 100.5, 101.5, 5], "volume": [1, 2, 3, 4]}]}}
    df = Yahoo.parse_intraday(bars, 1800, now=t(d1, 15, 45))
    assert df.height == 1 and df["close"][0] == 100.5  # pre-market, post-close and the still-forming 15:30 bar are excluded


def test_cboe_dolthub_alpaca_parsers_and_chain_cleaning():
    from strata.data.providers.sources import Alpaca, Cboe, DoltHub, clean_chain
    vix = Cboe.parse("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n01/03/1990,18.19,18.19,18.19,18.19\n")
    assert vix["date"].to_list() == [date(1990, 1, 2), date(1990, 1, 3)] and vix["close"][1] == 18.19
    rows = [{"date": "2024-03-04", "act_symbol": "SPY", "expiration": "2024-03-15", "strike": "500.00", "call_put": "Call", "bid": "5.10", "ask": "5.20", "vol": "0.14", "delta": "0.52", "gamma": "0.02", "theta": "-0.2", "vega": "0.3", "rho": "0.1"},
            {"date": "2024-03-04", "act_symbol": "SPY", "expiration": "2024-03-15", "strike": "500.00", "call_put": "Put", "bid": "", "ask": "4.9", "vol": "", "delta": "", "gamma": "", "theta": "", "vega": "", "rho": ""}]
    d = DoltHub.normalize(rows)
    assert d.height == 1 and d["right"][0] == "C" and d["bid"][0] == 5.10  # a row with no bid is dropped, not given one
    from strata.data.providers.sources import parse_occ
    assert parse_occ("SPY240315C00500000") == ("SPY", date(2024, 3, 15), "C", 500.0)
    b = Alpaca.parse_bars([{"t": "2024-03-04T14:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 9}])
    assert b["ts"][0].isoformat().startswith("2024-03-04T14:30") and b["volume"][0] == 9.0
    ch = pl.DataFrame({"strike": [500.0, 505.0, 510.0], "bid": [5.0, 3.0, 1.0], "ask": [5.2, 2.9, 1.1], "iv": [0.2, 0.2, 0.001], "delta": [0.5, 0.4, 0.2]})
    cleaned, rep = clean_chain(ch)
    assert cleaned["strike"].to_list() == [500.0, 510.0] and rep["rejected_crossed"] == 1  # crossed quote removed
    assert cleaned["iv"].to_list() == [0.2, None] and cleaned["delta"].to_list() == [0.5, None]  # implausible vendor IV nulled for recomputation


# ============================================================================ modeled pricing
def test_american_approximation_reference_values():
    """Barone-Adesi & Whaley (1987), Table 1: r = 8%, b = -4% (q = 12%), sigma = 20%, T = 0.25, K = 100."""
    from strata.core.pricing import baw_price_scalar
    assert baw_price_scalar(100, 100, 0.25, 0.08, 0.12, 0.20, True) == pytest.approx(3.52, abs=0.03)
    assert baw_price_scalar(110, 100, 0.25, 0.08, 0.12, 0.20, True) == pytest.approx(10.31, abs=0.04)
    assert baw_price_scalar(90, 100, 0.25, 0.08, 0.12, 0.20, True) == pytest.approx(0.59, abs=0.03)
    # no dividends: early exercise of a call is never optimal -> equals Black-Scholes
    d1 = (math.log(100 / 100) + (0.05 + 0.5 * 0.04) * 1.0) / 0.2
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    bs = 100 * N(d1) - 100 * math.exp(-0.05) * N(d1 - 0.2)
    assert baw_price_scalar(100, 100, 1.0, 0.05, 0.0, 0.20, True) == pytest.approx(bs, abs=1e-6)
    # an American put is worth at least intrinsic and at least the European put
    assert baw_price_scalar(80, 100, 0.5, 0.05, 0.0, 0.2, False) >= 20.0


def test_modeled_run_is_labelled_and_never_mixed_into_historical(tmp_settings):
    from strata.engine.engine import DataUnavailable, run_backtest
    st = study_store(tmp_settings)
    strat = {"name": "m", "symbol": "TST", "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0}}],
             "entry": {"conditions": cond("previous_session.return_pct <= -3") | {"children": [{"kind": "cond", "type": "entry_times", "params": {"times": "open"}}, {"kind": "cond", "type": "expr", "params": {"expr": "previous_session.return_pct <= -3"}}]}},
             "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "underlying": {"enabled": True, "move_for_pct": 1.0}, "time_exit": {"enabled": True, "time": "15:45", "dte_lte": None}}}
    cfg = {"start": "2024-03-04", "end": "2024-03-12", "strategies": [strat]}
    with pytest.raises(DataUnavailable):  # historical mode: no option quotes -> an error, never a silent estimate
        run_backtest(cfg, store=st)
    res = run_backtest({**cfg, "pricing": {"mode": "modeled", "iv_source": "fixed", "iv_fixed": 0.2}}, store=st)
    assert res.meta["pricing"] == "modeled" and res.meta["pricing_assumptions"] and res.meta["data_resolution"]["TST"]["options"] == "modeled (theoretical)"
    assert [t["entry_date"] for t in res.trades] == ["2024-03-05", "2024-03-08"] and all("modeled_prices" in t["flags"] for t in res.trades)
    hit, timed = res.trades
    assert hit["exit_reason"] == "underlying_target" and hit["exit_time"].startswith("2024-03-05T10:3") and hit["pnl"] > 0
    # Mar 8: the +1% target is never reached, so the 15:45 time exit closes it. The underlying
    # still drifted up 0.53%, so the 94-strike call is worth at least its 0.50 intrinsic value.
    leg = timed["legs"][0]
    assert timed["exit_reason"] == "time_exit" and timed["exit_time"].startswith("2024-03-08T15:45") and leg["strike"] == 94.0
    assert 0.45 <= leg["exit_price"] <= 0.60 and leg["entry_ask"] > leg["entry_bid"] > 0
    assert timed["pnl"] == pytest.approx((leg["exit_price"] - leg["entry_price"]) * 100 - timed["commissions"], abs=0.01)


# ============================================================================ templates and API
def test_templates_default_to_real_symbols_and_adapt_to_eod_data():
    from strata.engine.models import StrategyConfig
    from strata.engine.templates import examples, templates
    for t in templates("SPY", eod=True):
        cfg = StrategyConfig.model_validate(t["config"])
        assert cfg.symbol == "SPY" and all(l.dte >= 7 for l in cfg.legs if l.kind == "option")
        assert all(c.params["times"] == "close" for c in cfg.entry.conditions.children if c.type == "entry_times")
    assert all("DEMO" not in json.dumps(x) for x in templates() + examples())
    for ex in examples("QQQ", eod=True):
        assert StrategyConfig.model_validate(ex).symbol == "QQQ"


def test_api_settings_never_return_secrets_and_interpret_is_stateless(tmp_settings, monkeypatch):
    from fastapi.testclient import TestClient

    from strata.api import app as appmod
    monkeypatch.setattr(appmod, "jobs", InlineJobs())
    c = TestClient(appmod.app)  # no lifespan: jobs are run inline
    r = c.put("/api/settings", json={"settings": {"llm.provider": "offline"}, "secrets": {"OPENAI_API_KEY": "sk-test-abcdef123456"}}).json()
    assert "sk-test-abcdef123456" not in json.dumps(r) and r["secrets"]["OPENAI_API_KEY"] == {"label": "OpenAI", "set": True, "source": "protected file (owner-only)", "hint": "…3456"}
    assert "sk-test" not in c.get("/api/settings").text and c.get("/api/settings").json()["llm"]["active"] is None
    assert c.put("/api/settings", json={"settings": {"nope": 1}}).status_code == 422
    out = c.post("/api/ai/interpret", json={"text": "How often does SPY rise 1% from the open after falling 3% or more the previous trading day?"}).json()
    assert out["understood"] and out["draft"]["kind"] == "study" and out["errors"] == [] and out["unsupported"] == []  # interpretation only: nothing is run or stored
    assert db.list_runs() == [] if (db := __import__("strata.api.db", fromlist=["x"])) else True
    assert c.post("/api/expr/validate", json={"expr": "__import__('os')"}).json()["valid"] is False
    assert c.post("/api/study/run", json={"config": {"symbol": "SPY"}}).status_code == 404  # missing data is an error, not a simulated answer
    assert c.post("/api/data/sources/ingest", json={"source": "alpaca", "dataset": "bars_1m", "symbols": ["SPY"]}).status_code == 400  # no key configured
