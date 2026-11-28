"""V10: a reading runs only when it holds what the message says, with the same meaning.

What is tested here is the machinery (clock times, the typed inventory of a message, the check of a reading against it, the
repairs, the relation questions and their calculation, follow-ups). How well messages nobody has seen are read is not a
unit test: it is measured on the independent development questions and, once, on the held-out set (strata.ai.eval_v10,
docs/v10_evidence/eval).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from strata.ai import intent as I
from strata.ai import nl_rules as NL
from strata.ai import semantics as SM

KNOWN = ["IWM", "QQQ", "SPY", "VIX"]


# ----------------------------------------------------------------------------- a small market: daily bars and one-minute bars
@pytest.fixture
def market(tmp_path):
    from strata.api import db
    from strata.config import get_settings, override_settings
    from strata.core import calendar as cal
    from strata.data.store import MarketStore, SymbolMeta
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    db.init_db()
    st = MarketStore(tmp_path / "market")
    rng = np.random.default_rng(11)
    days = cal.trading_days(date(2024, 1, 2), date(2024, 6, 28))
    data = {}
    for sym, px0 in (("SPY", 480.0), ("QQQ", 410.0), ("IWM", 200.0)):
        rows, daily, prev = [], [], px0
        for d in days:
            o = prev * (1 + rng.normal(0, 0.004))
            steps = rng.normal(0, 0.0006, 390)
            path = o * np.cumprod(1 + steps)
            t0 = int(cal.session_bounds(d)[0].timestamp())
            op = np.r_[o, path[:-1]]
            hi, lo = np.maximum(op, path) * 1.0001, np.minimum(op, path) * 0.9999
            rows += [(t0 + 60 * k, float(op[k]), float(hi[k]), float(lo[k]), float(path[k]), 1e4) for k in range(390)]
            daily.append((d, float(o), float(hi.max()), float(lo.min()), float(path[-1]), 1e7))
            prev = float(path[-1])
        st.upsert_symbol(SymbolMeta(symbol=sym, kind="etf", vol_index="VIX"))
        st.write_daily(sym, pl.DataFrame(daily, schema=["date", "open", "high", "low", "close", "volume"], orient="row"), provider="yahoo")
        st.write_bars(sym, "1m", pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row").with_columns(pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))), provider="alpaca")
        st.rebuild_manifest(sym)
        data[sym] = (daily, rows)
    v = 14 + 8 * (0.5 + 0.5 * np.sin(np.arange(len(days)) / 9))
    st.upsert_symbol(SymbolMeta(symbol="VIX", kind="volatility_index", has_options=False))
    st.write_daily("VIX", pl.DataFrame({"date": days, "open": v, "high": v + 1, "low": v - 1, "close": v, "volume": [0.0] * len(days)}), provider="cboe")
    st.rebuild_manifest("VIX")
    yield st, data, days
    override_settings(data_dir=old)


@pytest.fixture
def ag(market):
    from strata.ai.agent import Agent
    from strata.ai.eval_v8 import NoJobs
    a = Agent(NoJobs(), market[0], llm_factory=lambda: None)
    a._data_check = lambda *x, **k: None  # which sessions have option prices is not what these tests are about
    return a


def session():
    from strata.api import db
    return db.create_session("t")["id"]


def route(ag, text, sid=None):
    sid = sid or session()
    ag._state(sid)
    return ag.interpret(sid, text, None, None, None)[0]


def post(ag, sid, text):
    return ag.post(sid, text)[-1]


def draft_of(ag, sid):
    return ag._state(sid)[2]


# ----------------------------------------------------------------------------- clock times
def test_times_are_read_as_traders_write_them():
    t = lambda s: [(x["et"], x.get("assumed_tz")) for x in NL.find_times(s, None) if x["how"] != "named"]  # noqa: E731
    assert t("out by 1245 PST") == [("15:45", None)], "four digits with a zone are hours and minutes"
    assert t("in at 0935 ET out at 1030 ET") == [("09:35", None), ("10:30", None)]
    assert t("exit at 2:30 CT") == [("15:30", None)], "an hour with a zone and no am/pm is the one inside the session"
    assert t("14:30 london time") == [("09:30", None)]
    assert [x[0] for x in t("from 10 to 2")] == ["10:00", "14:00"], "an hour alone is a time after a word that introduces one"
    for not_a_time in ("buy at 10 delta", "after 3 red days", "when vix is at 12", "risk 1500 per trade", "by 3 points", "since 2020", "spy at 630"):
        assert t(not_a_time) == [], not_a_time
    assert t("Buy 1 SPY 0DTE call at 0635 PT. Take profit at 30%") == [("09:35", None)], "a full stop ends the sentence, not the time"


def test_a_time_with_no_zone_is_new_york_time_unless_it_cannot_be():
    from strata.api import settings as ST
    old = ST.DEFAULTS["ui.timezone_confirmed"]
    ST.DEFAULTS["ui.timezone_confirmed"] = False
    try:
        z = lambda s: [(x["et"], x["assumed_tz"], x["zones"]) for x in NL.find_times(s, None) if x["how"] != "named"]  # noqa: E731
        assert z("buy at 10")[0][:2] == ("10:00", "Eastern")
        assert z("at 7")[0] == ("10:00", "Pacific", ["Pacific"]), "7:00 is inside the session only in Pacific time"
        assert z("buy at 8:45 and out by 2:30") == [("09:45", "Central", ["Central"]), ("15:30", "Central", ["Central"])], "only Central puts both inside the session"
        assert z("at 845")[0][2] == ["Central", "Mountain", "Pacific"], "8:45 cannot be New York time and fits three zones: that is asked"
    finally:
        ST.DEFAULTS["ui.timezone_confirmed"] = old
    assert I.zone_revealed("buy the put right at the 830 open, sell at 10") == 60, "a time called the open says which zone the writer is in"
    assert I.zone_revealed("in at 7, out by the 1pm close") == 180
    assert I.with_zone("sell it at 10 whichever hits first", 60)[0] == "sell it at 10 CT whichever hits first"


# ----------------------------------------------------------------------------- the typed inventory of a message
def said(text, trade=None):
    return [(x["type"], x["value"], x.get("about") or x.get("role")) for x in I.said(text, KNOWN, trade=trade) if x["type"] != "ticker"]


def test_a_number_is_what_the_words_make_it_not_its_digits():
    assert said("Buy ATM SPY 0DTE calls at 7am, out by 1245 PST, sell when they're up 40%") == [("dte", 0.0, None), ("clock", 600, None), ("clock", 945, None), ("pct", 40.0, "option")]
    assert said("buy spy puts at 10am ET after a 1% up day, sell at 3pm ET")[1] == ("pct", 1.0, "underlying"), "the size of a day is a move of the price"
    assert said("exit when spy is up 0.5% or the option loses 25%", True) == [("pct", 0.5, "underlying"), ("pct", 25.0, "option")]
    assert said("buy calls when it drops 1% from the open and sell when its up 30%") == [("pct", 1.0, "underlying"), ("pct", 30.0, "option")], "30% cannot be a move of the underlying in the life of the trade"
    assert said("sell when up 1%", True) == [("pct", 1.0, None)], "nothing says what this is a percentage of"
    assert said("selling it when the option has gained 60%, when it has lost 35%, or at 3:30 pm ET", True)[:2] == [("pct", 60.0, "option"), ("pct", 35.0, "option")]
    assert said("Sell them if SPY drops 0.7% from the entry or gains 1.5%.", True) == [("pct", 0.7, "underlying"), ("pct", 1.5, "underlying")], "the second half of an 'or' has the subject of the first"
    assert ("pct", 2.0, "account") in said("Size each trade at 2% of a $40,000 account and let the account compound.", True)
    assert said("tp 30 sl 15", True) == [("pct", 30.0, "option"), ("pct", 15.0, "option")], "a trader's shorthand for a target and a stop on the premium"
    assert said("spy gaps up at least .3%") == [], "the reader tidies '.3%' to '0.3%' before anything reads it"
    assert said(I.tidy("spy gaps up at least .3%")) == [("pct", 0.3, "underlying")]
    assert ("days", 3.0, "streak") in said("after 3 red days in a row how often is spy green the next day")
    assert said("how many times has spy fallen 3% in one day since 2020") == [("pct", 3.0, "underlying"), ("year", 2020, None)], "'in one day' is not a count of days"
    assert ("dte", 7.0, None) in said("sell three IWM puts at about 25 delta in the expiration closest to seven days away", True)


def test_what_a_message_is_and_what_a_thing_in_it_is_for():
    assert not I.is_trade("When does IWM put in its high?") and I.is_trade("buy the atm put at the open") and I.is_trade("what about puts")
    role = lambda text, short=False: [(x["text"], I.role_of(x, text, short=short)) for x in I.said(text, KNOWN, trade=True) if x["type"] in ("pct", "clock")]  # noqa: E731
    assert role("BTO spy 0dte atm call @ 0635 PT. PT 30% SL 20%, time stop 0730 PT") == [("0635 PT", "entry_time"), ("30%", "premium_target"), ("20%", "premium_stop"), ("0730 PT", "exit_time")]
    assert role("take profit +40% stop -25%") == [("40%", "premium_target"), ("25%", "premium_stop")], "a signed size says which it is"
    assert role("sell puts at 10am ET and buy them back at 2pm ET") == [("10am ET", "entry_time"), ("2pm ET", "exit_time")], "a sale that opens the position is an entry"
    assert role("What if I waited until 7 AM PT?") == [("7 AM PT", "entry_time")]
    assert role("Buy them back once they have lost half their value", short=True) == [("lost half their value", "premium_target")], "a sold option losing value is the gain"


# ----------------------------------------------------------------------------- the check, and what it does to a reading
def test_the_briefs_failures_are_read_as_written(ag):
    r = route(ag, "Buy ATM SPY 0DTE calls at 7am, out by 1245 PST, sell when they're up 40%")
    d = r["draft"]
    assert (d.entry_time, d.exit_time, d.premium_target_pct, d.underlying_target_pct) == ("10:00", "15:45", 40.0, None)
    assert not d.review and not d.unsupported
    r = route(ag, "When does IWM put in its high?")
    assert r["action"] == "profile" and r["frame"]["kind"] == "timing" and r["frame"]["measure"] == "high_time" and r["frame"]["symbol"] == "IWM"
    r = route(ag, "how often does tsla close green after a 3% down day")
    assert r["action"] == "clarify" and r.get("declined") and "TSLA" in r["text"], "a ticker that is not stored is never replaced"
    r = route(ag, "when vix is above 25 how often does qqq rise 1% from the open before noon")
    conds = [(c["key"], c["op"], c["value"]) for c in r["frame"]["conditions"]]
    assert r["frame"]["symbol"] == "QQQ" and ("vix", ">", 25.0) in conds, "the VIX is a condition, never what the question is about"


def test_a_reading_that_does_not_hold_what_was_said_waits_with_the_reason(ag):
    sid = session()
    m = post(ag, sid, "How often does SPY rise 1% from the open when retail sentiment is euphoric?")
    assert m["kind"] == "text" and m["content"]["clarify"] and "retail sentiment is euphoric" in m["content"]["text"] and "nothing was run" in m["content"]["text"]
    assert [o["label"] for o in m["content"]["options"]] == ["Run it as read"]
    ran = post(ag, sid, "Run it as read")
    assert ran["kind"] in ("plan", "investigation", "result"), "the reading runs when the user says so, and only then"
    r = route(ag, "chance qqq takes out yesterday's high in the first hour?")
    assert r["action"] == "profile" and r["frame"]["outcome"]["rel"] == {"a": "high_by:10:30", "op": ">", "b": "prev_high"}


def test_what_the_words_leave_one_way_to_place_is_placed(ag):
    d = route(ag, "bt this: BTO spy 0dte atm call @ 0635 pt only if the first 5 min candle is green. PT 30% SL 20%, time stop 0730 pt. 5 cons")["draft"]
    assert (d.entry_time, d.exit_time, d.premium_target_pct, d.premium_stop_pct, d.quantity) == ("09:35", "10:30", 30.0, 20.0, 5)
    assert [c.expr for c in d.conditions] == ["from_open_pct > 0"] and not d.unsupported and not d.review
    d = route(ag, "Buy QQQ calls at 10am eastern only when it gapped down at least 0.5% and yesterday was a red day, sell at 3pm eastern")["draft"]
    assert sorted(c.expr for c in d.conditions) == ["gap_pct <= -0.5", "previous_session.return_pct < 0"], "one size said is the size of one condition"
    d = route(ag, "risk 3% of a 50k account on spy calls at 9:45 ET with a 50% stop")["draft"]
    assert (d.premium_stop_pct, d.underlying_stop_pct, d.underlying_target_pct) == (50.0, None, None) and (d.sizing.mode, d.sizing.value, d.sizing.capital) == ("risk_pct", 3.0, 50000.0)
    d = route(ag, "short spy 0dte puts at noon eastern if spy is green on the day, 100% stop")["draft"]
    assert [c.expr for c in d.conditions] == ["change_pct > 0"], "'green on the day' is against the previous close, not against the open"
    d = route(ag, "qqq idea.. if the first 15 min is red, buy a 0dte put one strike otm at 8:45 ct, take profit +40% stop -25% and if neither hits im out at noon. $300 per trade")["draft"]
    assert (d.strike.method, d.strike.value, d.premium_target_pct, d.premium_stop_pct, d.exit_time) == ("strikes_otm", 1.0, 40.0, 25.0, "13:00"), "noon is noon in the zone of the message's other times"
    assert (d.sizing.mode, d.sizing.value) == ("dollars", 300.0)


def test_what_is_genuinely_open_is_asked_once(ag):
    from strata.api import settings as ST
    ST.set_setting("ui.timezone_confirmed", False)
    r = route(ag, "what if i just bought a spy 0dte call at the money every day at 845 and sold it by 11. 1 contract")
    assert r["action"] == "clarify" and "845" in r["text"] and [o["label"].split(" time")[0] for o in r["options"]] == ["Central", "Mountain", "Pacific"]
    assert "845 CT" in r["options"][0]["ask"], "each answer is the message with the zone written in"
    r = route(ag, "buy a spy 0dte atm call at the open and sell when it is up 4%")
    assert r["action"] == "clarify" and "percentage of the option's price, or of SPY's price" in r["text"]
    r = route(ag, "buy spy puts at 10am after a 1% up day, sell at 3pm")
    assert r["action"] == "new" and (r["draft"].entry_time, r["draft"].exit_time) == ("10:00", "15:00"), "10am can be New York time, so it is, and that is said"


def test_a_trade_described_in_full_after_a_question_is_its_own_trade(ag):
    sid = session()
    post(ag, sid, "How often does SPY rise 1% from the open after falling 2% or more the previous trading day?")
    post(ag, sid, "Buy ATM SPY 0DTE calls at the open after a 2% down day, exit at 15:55 ET")
    d = draft_of(ag, sid)
    assert d.kind == "options" and d.underlying_target_pct is None, "the question's 1% move is not a target of this trade"
    post(ag, sid, "What if I bought puts instead?")
    assert draft_of(ag, sid).structure == "long_put"


# ----------------------------------------------------------------------------- one price against another level
def test_relation_questions_are_read_as_asked():
    f = lambda t: SM.read(t, KNOWN).frame  # noqa: E731
    fr = f("How often does the market take out yesterday's high?")
    assert fr["outcome"]["rel"] == {"a": "high", "op": ">", "b": "prev_high"} and not fr["conditions"]
    fr = f("how often does spy poke above yestrdays high in the first hour and then close back under it")
    assert fr["conditions"][0]["rel"] == {"a": "high_by:10:30", "op": ">", "b": "prev_high"} and fr["outcome"]["rel"] == {"a": "close", "op": "<", "b": "prev_high"}
    fr = f("How often does SPY take out the prior day's high? And when it does, is it more often the real thing or a head fake, meaning does it close above that high or back under it.")
    assert fr["outcome"]["rel"]["a"] == "high" and fr["then"]["rel"] == {"a": "close", "op": ">", "b": "prev_high"}
    fr = f("Does SPY close above its 7 AM Pacific price on days after it fell at least 1%?")
    assert fr["outcome"]["rel"] == {"a": "close", "op": ">", "b": "at:10:00"} and [(c["key"], c["value"]) for c in fr["conditions"]] == [("prev_return", -1.0)]
    assert SM.describe(fr) == "On SPY, when the previous session fell 1% or more: how often it closes above its price at 10:00 New York time."
    fr = f("spy opens red vs yday's close but is back green by 0700 PT. how often does it finish green?")
    assert [c["rel"] for c in fr["conditions"]] == [{"a": "open", "op": "<", "b": "prev_close"}, {"a": "at:10:00", "op": ">", "b": "prev_close"}] and fr["outcome"]["rel"] == {"a": "close", "op": ">", "b": "prev_close"}
    fr = f("How many times since 2000 has IWM finished a day more than 3% below where it finished the day before? Just the count, and which years they pile up in.")
    assert fr["outcome"]["rel"] == {"a": "close", "op": "<", "b": "prev_close", "pct": -3.0} and fr["period"][0] == "2000-01-01" and not fr.get("unread")


def test_the_relation_count_is_the_one_a_direct_count_gives(market):
    from strata.research import profile as PR
    st, data, days = market
    daily, rows = data["SPY"]
    # counted here from the arrays the store was built from, without the profile code
    took, closed_above = 0, 0
    for i in range(1, len(daily)):
        if daily[i][2] > daily[i - 1][2]:
            took += 1
            closed_above += daily[i][4] > daily[i - 1][2]
    r = PR.relation_frequency(st, "SPY", {"a": "high", "op": ">", "b": "prev_high"}, then={"a": "close", "op": ">", "b": "prev_high"})
    assert (r["k"], r["n"]) == (took, len(daily) - 1) and (r["then"]["k"], r["then"]["n"]) == (closed_above, took) and r["bars"] == "daily bars"
    # the price at 10:00 is the close of the 09:59 bar (the 30th minute of the session)
    hits = n = 0
    for i in range(1, len(daily)):
        if daily[i - 1][4] / daily[i - 2][4] - 1 <= -0.002 if i >= 2 else False:
            n += 1
            hits += daily[i][4] > rows[i * 390 + 29][4]
    r = PR.relation_frequency(st, "SPY", {"a": "close", "op": ">", "b": "at:10:00"}, conditions=[{"key": "prev_return", "op": "<=", "value": -0.2}])
    assert (r["k"], r["n"]) == (hits, n) and "one-minute" in r["bars"]


# ----------------------------------------------------------------------------- follow-ups
def test_follow_ups_change_what_they_name(ag):
    sid = session()
    post(ag, sid, "0dte spy iron condor. shorts at 12 delta, wings 5 wide, on at 0700 pt. take it off at 50% of the credit, stop out if it costs double the credit to buy back, otherwise let it expire. 2 lots")
    d = draft_of(ag, sid)
    assert (d.structure, d.strike.value, d.width, d.quantity, d.entry_time, d.premium_target_pct, d.premium_stop_pct, d.hold_to_expiration) == ("iron_condor", 0.12, 5.0, 2, "10:00", 50.0, 100.0, True)
    post(ag, sid, "same on qqq")
    assert draft_of(ag, sid).symbol == "QQQ"
    post(ag, sid, "kill the stop")
    assert draft_of(ag, sid).premium_stop_pct is None and draft_of(ag, sid).premium_target_pct == 50.0
    post(ag, sid, "move the exit to 1245")
    assert draft_of(ag, sid).exit_time == "15:45", "the zone the first message settled (0700 can only be Pacific) is the conversation's"
    post(ag, sid, "3 lots and add a stop at 150%")
    assert (draft_of(ag, sid).quantity, draft_of(ag, sid).premium_stop_pct) == (3, 150.0)
    m = post(ag, sid, "show the losers")
    assert m["content"]["route"] == "show_trades"


def test_follow_ups_on_a_question_keep_the_question(ag):
    sid = session()
    post(ag, sid, "How often does SPY take out the prior day's high?")
    m = post(ag, sid, "and the low?")
    assert m["content"]["frame"]["outcome"]["rel"] == {"a": "low", "op": "<", "b": "prev_low"}
    m = post(ag, sid, "ok stay with the lows. only the days that open under yesterday's close")
    assert [c["rel"] for c in m["content"]["frame"]["conditions"]] == [{"a": "open", "op": "<", "b": "prev_close"}]
    m = post(ag, sid, "gaps over .4% only")
    assert [(c["key"], c["op"], c["value"]) for c in m["content"]["frame"]["conditions"]] == [("gap", "<=", -0.4)], "the size is new, the direction is the question's own"
    m = post(ag, sid, "same thing for the q's")
    assert m["content"]["frame"]["symbol"] == "QQQ"
    m = post(ag, sid, "could that just be noise? theres not many of them")
    assert m["content"]["route"] == "robustness" and "plausibly between" in m["content"]["text"], "a count from the bars is judged as a count"


# ----------------------------------------------------------------------------- the evaluation data
def test_the_held_out_set_is_the_one_that_was_sealed():
    import json

    from strata.ai import eval_v10 as V
    seal = json.loads((V.EVAL_DIR / "heldout_seal.json").read_text())
    now = V.seal(sorted((V.EVAL_DIR / "heldout").glob("*.jsonl")))
    assert now == seal, "the held-out questions must not change after they were sealed"
    for p in sorted((V.EVAL_DIR / "heldout").glob("*.jsonl")) + sorted((V.EVAL_DIR / "dev").glob("*.jsonl")):
        assert not V.check_authored(p)["problems"], p.name


# ============================================================================================================================
# V10, second part: confirming complex readings, what cannot be answered, the reading as a sentence with parts to change,
# more question concepts, more of a trade's words, independent recalculation of the answers computed from bars
# ============================================================================================================================
def _confirming():
    from strata.api import settings as ST
    ST.set_setting("ai.confirm_complex", True)


def test_a_simple_question_is_answered_and_a_complex_one_shows_its_reading_first(ag):
    _confirming()
    assert route(ag, "how often does qqq close green after a gap down")["action"] in ("new", "investigate", "profile"), "one condition: answered at once"
    sid = session()
    r = route(ag, "when vix is above 15 how often does qqq close green after a gap down since 2024", sid)
    assert r["action"] == "clarify" and r["waiting"] == {"reasons": [], "complex": True} and r["pending"]["kind"] == "confirm_frame", "two conditions: the reading is shown and waits"
    assert "Here is how I read it: On QQQ, when the VIX closed above 15 the session before and it opened below the previous close" in r["text"] and "Nothing was run" in r["text"]
    # the go-ahead runs exactly the reading that was shown; a change to it edits THAT reading and runs
    m = post(ag, sid, "when vix is above 15 how often does qqq close green after a gap down since 2024")
    assert m["content"]["confirm"] == {"reasons": [], "complex": True} and [p["t"] for p in m["content"]["reading_parts"] if p["t"] != "text"] == ["ticker", "condition", "condition", "period"]
    after = ag.post(sid, "use SPY instead")
    # (since the held-out run: a change to a waiting reading shows the changed reading and waits again; tests/test_v10_heldout.py)
    fr = ag._state(sid)[0]["state"]["pending"]["frame"]
    assert fr["symbol"] == "SPY" and [c["key"] for c in fr["conditions"]] == ["vix", "gap"] and fr["period"][0] == "2024-01-01", "the waiting reading was edited, not replaced"
    assert next(m for m in after if m["role"] == "assistant")["content"]["text"].startswith("Changed QQQ to SPY")
    sid2 = session()
    post(ag, sid2, "when vix is above 15 how often does qqq close green after a gap down since 2024")
    ag.post(sid2, "Run it as read")
    assert ag._state(sid2)[0]["state"]["frame"]["symbol"] == "QQQ" and ag._state(sid2)[0]["state"].get("pending") is None
    # a reading that waits because part of the message is not in it still waits after a change, with the part still named
    sid3 = session()
    post(ag, sid3, "How often does SPY rise 1% from the open when retail sentiment is euphoric?")
    again = post(ag, sid3, "use QQQ instead")["content"]
    assert again["confirm"]["reasons"] and "retail sentiment is euphoric" in again["text"] and "On QQQ" in again["reading"] and ag._state(sid3)[0]["state"].get("frame") is None


def test_what_the_stored_data_cannot_answer_is_declined_with_the_reason(ag):
    for text, word in (("fomc days only: buy the 0dte atm spy straddle at 10 ET, sell at noon", "calendar of scheduled events"), ("how does spy do in opex week", "calendar of scheduled events"),
                       ("What has SPY's five-day return been when the 14-day RSI closes below 30?", "technical indicator"), ("how does QQQ do after it closes back above its 50-day", "technical indicator"),
                       ("can you redo it buying at the ask and selling at the bid", "No historical bid and ask quotes"), ("vix is at 14 and spy is up 6 days straight. is tomorrow more likely red or green?", "does not forecast"),
                       ("should i hold my calls overnight", "does not forecast")):
        r = route(ag, text)
        assert r["action"] == "clarify" and r.get("declined") and word in r["text"] and "Nothing was run" in r["text"], (text, r.get("text"))
    r = route(ag, "vix is at 14 and spy is up 6 days straight. is tomorrow more likely red or green?")
    assert r["options"] == [{"label": "How often does SPY close higher the next day after 6 up days in a row?", "ask": "How often does SPY close higher the next day after 6 up days in a row?"}], "the nearest thing it can measure is offered, not run"
    r = route(ag, "fomc days only: buy the 0dte atm spy straddle at 10 ET, sell at noon")
    assert r["options"][0]["ask"] == "Buy the 0dte atm spy straddle at 10 ET, sell at noon"
    # not everything with one of those words is out of scope
    assert not route(ag, "ok forget opex. are mondays any different for spy").get("declined"), "a thing the message sets aside is not asked for"
    assert not route(ag, "after a 2% down day how often is the next day green").get("declined"), "the next day, in the past, is history"
    assert not route(ag, "how often is spy above vwap at noon when it gapped up").get("declined"), "VWAP is the one indicator that is stored"
    assert I.out_of_scope("what usually happens the day after spy drops 2%", KNOWN) is None
    # option prices have a first day: a test of a period before it is declined, one that starts before it says where it starts
    ag._option_range = lambda sym: (date(2024, 3, 1), date(2024, 6, 28))
    r = route(ag, "would buying spy 0dte calls at the open and selling at noon have worked in 2022?")
    assert r["action"] == "clarify" and r.get("declined") and "stored from 2024-03-01" in r["text"]
    r = route(ag, "buy spy 0dte calls at the open, sell at noon ET, since 2020")
    assert r["action"] == "new" and any("Option prices are stored from 2024-03-01" in a for a in r["assumptions"])


def test_the_reading_is_a_sentence_whose_parts_can_be_changed(ag):
    f = SM.read("when vix is above 25 how often does qqq close green after a gap down since 2020", KNOWN, "SPY").frame
    parts = I.reading_parts(f, KNOWN)
    assert "".join(p["s"] for p in parts) == SM.describe(f), "the parts are the sentence, cut up"
    marked = {p["t"]: p for p in parts if p["t"] != "text"}
    assert marked["ticker"]["s"] == "QQQ" and [c["say"] for c in marked["ticker"]["choices"]] == ["use IWM instead", "use SPY instead"]
    assert marked["period"]["s"] == "since 2020-01-01" and marked["period"]["choices"][0]["say"] == "use the whole history"
    says = [p["choices"][0]["say"] for p in parts if p["t"] == "condition"]
    assert says == ["remove the vix condition", "remove the gap condition"]
    # every one of those follow-ups does what its label says, through the same reader as anything typed
    sid = session()
    post(ag, sid, "when vix is above 15 how often does qqq close green after a gap down since 2024")
    fr = lambda: ag._state(sid)[0]["state"]["frame"]  # noqa: E731
    ag.post(sid, "remove the vix condition")
    assert fr()["symbol"] == "QQQ" and [c["key"] for c in fr()["conditions"]] == ["gap"], "removing the VIX condition does not move the question onto VIX"
    ag.post(sid, "use IWM instead")
    ag.post(sid, "use the whole history")
    assert fr()["symbol"] == "IWM" and fr()["period"] == [None, None]
    ag.post(sid, "since 2024")
    ag.post(sid, "remove the gap condition")
    assert fr()["period"][0] == "2024-01-01" and fr()["conditions"] == []
    # a time said in another zone is said back in that zone too
    r = route(ag, "Does SPY close above its 7 AM Pacific price on days after it fell at least 1%?")
    fr2 = r.get("frame") or r["pending"]["frame"]
    assert fr2["zone"] == 180 and "10:00 New York time (7:00 AM Pacific)" in SM.describe(fr2)


def test_more_ways_of_asking_are_read_as_what_they_say():
    rd = lambda t: SM.read(I.tidy(t), KNOWN, "SPY")  # noqa: E731
    f = rd("spy gap up >.5%, how often does it fill same day").frame
    assert f["conditions"][0] | {"text": ""} == {"key": "gap", "op": ">=", "value": 0.5, "text": ""} and (f["outcome"]["key"], f["outcome"]["level"]) == ("touch_level", "prev_close")
    f = rd("Take every QQQ session that opened at least half a percent under the prior close. In what share of them did price get back up to that prior close before the bell? 2016 onward.").frame
    assert (f["conditions"][0]["op"], f["conditions"][0]["value"]) == ("<=", -0.5) and f["outcome"]["level"] == "prev_close" and f["period"] == ["2016-01-01", None]
    f = rd("when IWM opens above yesterday's high, how often does it come back and trade under yesterday's high before the close?").frame
    assert f["outcome"]["rel"] == {"a": "low_by:16:00", "op": "<", "b": "prev_high"}, "trading UNDER the high is the low against it (until V10: read as trading above it)"
    f = rd("Four down closes in a row in IWM, where is it a week later? Go back as far as you've got.").frame
    assert (f["kind"], f["end"], f["outcome"]) == ("movement", "sessions:5", {"key": "end_return", "ref": "prev_close"}) and f["conditions"][0]["value"] == 4 and not rd("go back as far as you've got on that").unread
    assert "over the next 5 sessions (from that session's close to the close 5 sessions later)" in SM.describe(f)
    f = rd("After SPY has closed lower three days in a row, what has the return been over the following five trading days, close to close?").frame
    assert f["conditions"][0] | {"text": ""} == {"key": "down_streak", "op": ">=", "value": 3.0, "text": ""} and f["end"] == "sessions:5"
    f = rd("after spy is down 3 days straight how often does it bounce the next day").frame
    assert f["outcome"]["direction"] == "up", "a bounce after a run of down days is up"
    f = rd("iwm after 2 down days in a row, odds of a green day?").frame
    assert f["outcome"] | {"text": ""} == {"key": "close_vs", "ref": "prev_close", "direction": "up", "text": ""} and "when 2 or more down sessions in a row" in SM.describe(f)
    f = rd("is monday or friday more likely to close red in spy? since 2016").frame
    assert (f["kind"], f["measure"], f["focus"]) == ("breakdown", "close<prev_close", ["Mon", "Fri"]) and "Mondays against Fridays" in SM.describe(f)
    assert rd("spy close to close by day of the week, last 15 years or so").frame["measure"] == "close_to_close"
    assert rd("What percentage of the time has SPY closed above its opening price since the start of 2000? I would also like that broken out by day of the week").frame["measure"] == "close>open"
    f = rd("what's the fattest hour of the day for IWM, range-wise? i.e. which hour has the biggest average high minus low").frame
    assert (f["kind"], f["measure"], f["bucket"]) == ("timing", "range", 60) and not rd("which hour has the biggest average high minus low for IWM").unread
    assert rd("do tuesdays tend to be green for qqq").frame["outcome"]["direction"] == "up" and rd("when yesterday was up, is today usually up too? spy").frame["outcome"]["key"] == "close_vs"
    assert rd("how often does qqq trade 1% below its open at some point").frame["outcome"]["value"] == -1.0
    f = rd("after a 1%+ gap down on SPY, how often is it still below the open at noon ET?").frame
    assert f["outcome"]["rel"] == {"a": "at:12:00", "op": "<", "b": "open"} and f["conditions"][0]["value"] == -1.0
    assert rd("does a strong first hour mean a strong day for qqq?").frame["kind"] == "agreement", "'a strong day for' is not a weekday question"
    assert rd("how often does a 1% gap up on qqq keep going til close").frame["outcome"]["key"] in ("continue", "close_vs"), "never 'the gap fills'"
    for text in ("how often does spy close green since 2016 through 2024", "how often does spy close green from 2016 on.", "how often does spy close green going back to 2010"):
        assert rd(text).frame["period"][0], text
    # the writer's own commentary is set aside, and said
    kept, aside = I.without_commentary("What time of day does QQQ usually put in its low? I've always figured it's either the first half hour or right into the close, but that's just from staring at it.")
    assert kept == "What time of day does QQQ usually put in its low?" and len(aside) == 1
    assert I.without_commentary("Buy SPY calls at the open. I think a 30% stop makes sense.")[1] == [], "a sentence with a number in it is never set aside"


def test_weekday_shares_and_ranges_are_computed(market):
    from strata.research import profile as PR
    st, data, days = market
    daily = data["SPY"][0]
    res = PR.breakdown(st, "SPY", "weekday", "close<prev_close")
    mon = next(g for g in res["groups"] if g["group"] == "Monday")
    idx = [i for i, row in enumerate(daily) if i > 0 and row[0].weekday() == 0]
    assert mon["sessions"] == len(idx) and mon["hits"] == sum(1 for i in idx if daily[i][4] < daily[i - 1][4]) and abs(mon["share"] - mon["hits"] / len(idx) * 100) < 1e-9
    assert res["share_measure"] and res["measure_label"] == "closing below the previous close" and mon["share_ci95"][0] < mon["share"] < mon["share_ci95"][1]
    from strata.ai import answer as ANS
    a = ANS.breakdown(res, {"focus": ["Mon", "Fri"], "by": "weekday"})
    assert "Mondays" in a["headline"] and "Fridays" in a["headline"] and a["viz"]["share"] is True and any("against the day's own open instead" in h for h in a["happened"])
    rng = PR.timing(st, "SPY", "range", 60)
    mv = PR.timing(st, "SPY", "abs_move", 60)
    assert all(r["mean_abs_move_pct"] >= m["mean_abs_move_pct"] - 1e-12 for r, m in zip(rng["buckets"], mv["buckets"])), "a part's high-to-low range is never smaller than its start-to-end move"
    assert rng["value_name"] == "average high-to-low range" and "widest range" in ANS.timing(rng, {})["headline"]


def test_the_words_of_a_trade_are_placed(ag):
    _confirming()
    d = route(ag, "At 10:30 ET, if SPY is above VWAP, sell a put spread that expires that day: short the 20 delta, long the strike 5 points under it. Two contracts, let it expire.")["draft"]
    assert (d.structure, d.dte, d.width, d.quantity, d.strike.method, d.strike.value, d.hold_to_expiration) == ("bull_put_spread", 0, 5.0, 2, "delta", 0.2, True), "sold: a put CREDIT spread, expiring that day"
    assert d.review == ["This has several parts, so check that each one is what you meant."], "complete, and shown before it runs"
    assert NL.find_structure("sto spy ccs exp friday") == "bear_call_spread" and NL.find_structure("buy a qqq put spread") == "bear_put_spread" and NL.find_structure("sell a call vertical on iwm") == "bear_call_spread"
    d = route(ag, "buy spy 0dte atm calls at 10 ET, 30% tp 20% sl, 20 min max hold")["draft"]
    assert (d.premium_target_pct, d.premium_stop_pct, d.hold_minutes) == (30.0, 20.0, 20)
    d = route(ag, "buy one SPY 0DTE call each time it crosses above VWAP, max 3 a day, and stop trading for the day after losing $200")["draft"]
    assert d.max_entries == 3 and d.sizing.daily_loss_stop == 200.0 and [c.expr for c in d.conditions] == ["crosses_above(price, vwap)"], "'after losing $200' is not a down day"
    d = route(ag, "Buy SPY 0DTE at-the-money calls at 9:45 ET after a down day, take profit at 50% of the premium, stop at 30%, exit by noon ET. Risk $200 per trade.")["draft"]
    assert [c.expr for c in d.conditions] == ["previous_session.return_pct < 0"], "one condition said once is held once"
    d = route(ag, "buy spy puts at 11am et if spy gapped down over 1%, 50% take profit, 30% stop loss")["draft"]
    assert (d.premium_target_pct, d.premium_stop_pct) == (50.0, 30.0) and not [x for x in d.review if "several parts" not in x], "a label after the number names it"
    d = route(ag, "qqq 0dte puts on gap ups >1%, 30% tp / 20% sl, in at 9:35 et")["draft"]
    assert [c.expr for c in d.conditions] == ["gap_pct >= 1"] and not [x for x in d.review if "several parts" not in x], "the size of a gap is the underlying's"
    d = route(ag, "Buy SPY 0DTE calls at the open and take profit when the calls are up 50%")["draft"]
    assert d.premium_target_pct == 50.0 and d.unsupported == []
    d = route(ag, "buy spy 0dte calls at the open and sell when SPY is up half a percent")["draft"]
    assert d.underlying_target_pct == 0.5
    assert [x for x in I.said("IWM 7dte puts 1% otm after any 2% up day", KNOWN, trade=True) if x["type"] == "pct"][0]["about"] == "strike"
    # a trade that names no ticker is asked about; one that leaves most of itself open shows its defaults and waits
    r = route(ag, "first 5 min candle is green -> buy a 0dte atm call, sell at +30% or at the close. 1 contract a day")
    assert r["action"] == "clarify" and "Which ticker" in r["text"] and [o["label"] for o in r["options"]] == ["IWM", "QQQ", "SPY"] and r["options"][2]["ask"].endswith("(on SPY)")
    sid = session()
    m = post(ag, sid, "backtest SPY iron condors")
    assert m["content"]["state"] == "awaiting_confirmation" and "Your message leaves" in m["content"]["text"] and m["content"]["runs"] == []
    # a follow-up that resizes a condition changes that condition (and an exit's percentage never does)
    sid = session()
    ag.post(sid, "buy qqq 0dte puts at 10:15 ET when qqq gapped up 1%, out at noon ET")
    ag.post(sid, "what if the gap is only half a percent")
    assert [c.expr for c in draft_of(ag, sid).conditions] == ["gap_pct >= 0.5"]
    ag.post(sid, "exit after a 1% underlying move instead")
    assert [c.expr for c in draft_of(ag, sid).conditions] == ["gap_pct >= 0.5"]


def test_a_trade_is_written_out_in_full():
    from strata.ai.draft import Draft, DraftCondition, DraftSizing, DraftStrike
    d = Draft(symbol="SPY", structure="bull_put_spread", dte=0, strike=DraftStrike(method="delta", value=0.2), width=5, quantity=2, entry_time="10:30", hold_to_expiration=True, premium_target_pct=50,
              conditions=[DraftCondition(expr="price > vwap", text="the price is above VWAP")], weekdays=["Mon"], underlying_stop_pct=0.7, sizing=DraftSizing(mode="risk_pct", value=3, capital=30000, daily_loss_stop=1500))
    s = I.describe_draft(d)
    text = " ".join(s["lines"])
    for part in ("Sell SPY put credit spreads (sell a put, buy a lower one)", "expiring the same day (0DTE)", "short strike near 20 delta", "the other strike 5 points further out", "at 10:30 AM New York time",
                 "Only enter when the price is above VWAP, and only on Mondays", "take profit when the position has gained 50% of the credit received", "exit when SPY has moved 0.7% against the trade from the entry",
                 "hold to expiration", "3% of the account at risk per trade", "starting with $30,000", "sized off the starting account", "no new trade that day after losing $1,500"):
        assert part in text, part
    assert any(x.startswith("Strikes:") for x in s["more"]) and any("0.7% against the trade" in x for x in s["more"]), "what the editable sentence has no slot for is listed"
    # what was not stated is the value the run will use, and is marked as a default
    s2 = I.describe_draft(Draft(symbol="QQQ", structure="short_put", entry_time="open", hold_to_expiration=True), {"dte": 14, "strike": {"method": "delta", "value": 0.2}, "width": None})
    assert "expiring about 14 days out (not stated: the default)" in s2["lines"][0] and "near 20 delta (not stated: the default)" in s2["lines"][0]


def test_answers_computed_from_bars_are_recalculated_independently(market):
    from strata.research import profile as PR
    from strata.research import reference as RF
    from strata.research import verify as VF
    st = market[0]
    RF._REF = None  # the reference of this test's store
    try:
        for text in ("how often does QQQ take out yesterday's high in the first hour", "when IWM opens above yesterday's high, how often does it trade under yesterday's high before the close",
                     "how often does spy close above its 10:00 ET price after a gap down", "how often does spy poke above yesterday's high and then close back under it"):
            f = SM.read(text, KNOWN, "SPY").frame
            res = PR.relation_frequency(st, f["symbol"], f["outcome"]["rel"], f.get("period"), f.get("conditions"), (f.get("then") or {}).get("rel"))
            c = VF.profile(f, res)
            assert c["status"] == "same" and c["level"] == "Fully verified" and res["n"] > 5, (text, c)
        for measure in ("high_time", "low_time", "abs_move", "range"):
            assert VF.profile({"kind": "timing", "symbol": "SPY", "measure": measure, "bucket": 30}, PR.timing(st, "SPY", measure, 30))["status"] == "same", measure
        for measure in ("open_to_close", "close_to_close", "close<prev_close", "close>open"):
            assert VF.profile({"kind": "breakdown", "symbol": "QQQ", "by": "weekday", "measure": measure}, PR.breakdown(st, "QQQ", "weekday", measure))["status"] == "same", measure
        ev = {"key": "gap", "op": ">=", "value": 0.3}
        assert VF.profile({"kind": "frequency", "symbol": "IWM", "outcome": {"key": "event", "event": ev}}, PR.event_frequency(st, "IWM", ev))["status"] == "same"
        # a disagreement is reported as one, with where to look
        f = SM.read("how often does QQQ take out yesterday's high in the first hour", KNOWN, "SPY").frame
        res = PR.relation_frequency(st, "QQQ", f["outcome"]["rel"])
        wrong = VF.profile(f, {**res, "k": res["k"] - 2, "dates": res["dates"][2:]})
        assert wrong["status"] == "differs" and wrong["level"] == "Disagreement" and len(wrong["dates"]) == 2
    finally:
        RF._REF = None


def test_a_proposal_from_a_model_must_keep_the_content():
    from strata.ai import proposer as P
    msg = "bto 3 qqq 0dte 25d puts @ 0940 et, tp 50 sl 30, flat by 1500 et"
    assert P.faithful(msg, "Buy 3 QQQ 0DTE 25-delta puts at 0940 ET. Take profit at 50% of the premium. Stop at 30% of the premium. Exit at 1500 ET.", KNOWN) == []
    assert any(d.startswith("dropped") for d in P.faithful(msg, "Buy 3 QQQ 0DTE 25-delta puts at 0940 ET. Take profit at 50% of the premium. Exit at 1500 ET.", KNOWN)), "a stop that vanished"
    assert any(d.startswith("added") for d in P.faithful("does qqq bounce after a red open", "On QQQ, on days when it opens 1% or more below the previous close, how often does it close above its open?", KNOWN)), "a size nobody said"
    assert P.faithful("spy gap up >.5%, how often does it fill same day", "On SPY, on days when it opens 0.5% or more above the previous close, how often does it trade back to the previous close?", KNOWN) == []


# ----------------------------------------------------------------------------- replies, in a person's own words
def test_a_reply_that_only_agrees_or_refuses_is_about_what_waits(ag):
    for t in ("yes", "yep thats what i meant", "looks right, go ahead", "Run it as read", "ok go ahead and run that", "perfect, run it"):
        assert I.go_ahead(t) == "yes", t
    for t in ("no", "that's not what I meant", "nah don't run it", "never mind", "no thanks"):
        assert I.go_ahead(t) == "no", t
    for t in ("yes but only fridays", "no, 2% not 3%", "run it on QQQ", "is that right?", "no stop", "right after the open", "go long", "thanks"):
        assert I.go_ahead(t) is None, t
    _confirming()
    q = "when vix is above 15 how often does qqq close green after a gap down since 2024"
    sid = session()
    assert post(ag, sid, q)["content"]["confirm"]["complex"] is True
    ag.post(sid, "yep thats what i meant")
    st = ag._state(sid)[0]["state"]
    assert st.get("pending") is None and st["frame"]["symbol"] == "QQQ", "the waiting reading ran on a typed go-ahead"
    sid = session()
    post(ag, sid, q)
    assert post(ag, sid, "no")["content"]["text"].startswith("Not run.") and ag._state(sid)[0]["state"].get("pending") is None
    assert post(ag, sid, "yes")["content"]["text"].startswith("Nothing is waiting to be run")
    # a choice typed by its place stands for the message its button sends
    sid = session()
    m = post(ag, sid, "sell a 5 wide SPY put credit spread 0dte at 10:30 ET, 20 delta, take profit at 50% of the credit, stop at 2x the credit")
    assert m["content"]["clarify"] and [o["label"][:16] for o in m["content"]["options"]] == ["When it costs 2x", "When the loss is"], "2x the credit: the price to buy back at, or the loss?"
    post(ag, sid, "the first one")
    d = draft_of(ag, sid)
    assert (d.structure, d.premium_target_pct, d.premium_stop_pct) == ("bull_put_spread", 50.0, 100.0)
    # the one alternative offered with a refusal is taken by a bare yes
    sid = session()
    m = post(ag, sid, "how often does NVDA close above its open")
    assert m["content"]["declined"] and "Stored: IWM, QQQ, SPY." in m["content"]["text"] and [o["label"] for o in m["content"]["options"]] == ["how often does SPY close above its open"]
    ag.post(sid, "sure")
    assert ag._state(sid)[0]["state"]["frame"]["symbol"] == "SPY"


def test_a_trade_follow_up_changes_the_trade_on_screen(ag):
    assert I.tidy("ok and put a stop on it") == "ok and place a stop on it" and I.tidy("buy a put on SPY") == "buy a put on SPY" and I.tidy("sell put spreads") == "sell put spreads"
    sid = session()
    post(ag, sid, "At 10:30 ET, if SPY is above VWAP, sell a put spread that expires that day: short the 20 delta, long the strike 5 points under it. Two contracts, let it expire.")
    d = draft_of(ag, sid)
    assert (d.structure, d.dte, d.width, d.quantity, d.entry_time) == ("bull_put_spread", 0, 5.0, 2, "10:30")
    m = post(ag, sid, "ok and put a stop on it, the losers must be ugly")
    assert m["content"]["text"].startswith("A stop needs a level") and draft_of(ag, sid).structure == "bull_put_spread", "\"put\" the verb is not the option, and no level is made up"
    post(ag, sid, "0.4% on SPY")
    assert draft_of(ag, sid).underlying_stop_pct == 0.4, "the bare level answers the question that was asked"
    post(ag, sid, "buy back the spread at 50% of the credit")
    assert draft_of(ag, sid).premium_target_pct == 50.0
    post(ag, sid, "kill the stop")
    post(ag, sid, "buy it back if it doubles")
    assert draft_of(ag, sid).premium_stop_pct == 100.0, "a sold position that costs twice as much has lost its credit: the stop, not a target"
    ed = I.edit_trade(draft_of(ag, sid), "run that on the Qs and put it next to the SPY numbers", KNOWN)
    assert ed["draft"].symbol == "QQQ" and ed["compare"] is True and ed["changed"] == ["symbol"]
    post(ag, sid, "run that on the Qs and put it next to the SPY numbers")
    assert draft_of(ag, sid).symbol == "QQQ" and not draft_of(ag, sid).review, "SPY names what it is compared with: nothing is left unused"
    assert I.said("sell same-day QQQ put credit spreads at 10", KNOWN, trade=True)[0]["type"] == "dte"


def test_a_follow_up_that_opens_another_position_is_applied_or_flagged(ag):
    exp = {"buy a put instead maybe, 30 delta": "long_put", "what if I sold a call spread instead": "bear_call_spread", "make it an iron condor": "iron_condor", "sell puts instead of buying calls": "short_put",
           "out if SPY drops 0.4% from where it was when I sold the spread": None, "buy back the short put at 50%": None, "move the short put to 15 delta": None, "widen the put spread to 10": None,
           "No, I'm selling the put spread from before, not buying a put": None, "sell them at noon": None}
    for t, e in exp.items():
        assert I.named_instrument(t) == e, t
    sid = session()
    post(ag, sid, "sell a 5 wide SPY put credit spread 0dte at 10:30 ET, 20 delta, take profit at 50% of the credit, hold to expiration")
    assert draft_of(ag, sid).strike.value == 0.2 and draft_of(ag, sid).entry_time == "10:30", "\"10:30 ET, 20 delta\" is a time and one delta"
    post(ag, sid, "buy a put instead maybe, 30 delta")
    d = draft_of(ag, sid)
    assert (d.structure, d.strike.value, d.width) == ("long_put", 0.3, None) and d.review and "position changed from selling put credit spreads to buying puts" in d.review[0]
    sid = session()
    post(ag, sid, "buy SPY 0dte atm calls at 10 ET, exit at 3 pm ET")
    post(ag, sid, "sell puts instead")
    assert draft_of(ag, sid).structure == "short_put" and not draft_of(ag, sid).review, "one leg for another, no exits to carry over: it runs"
    r = route(ag, "buy SPY 0dte calls at 10:30 ET, compare 20, 30 and 40 delta, exit at 3 pm ET")
    assert r["action"] == "experiment" and [v for _, v in r["dims"][0]["values"]] == [0.2, 0.3, 0.4], "a real list is still a comparison"


def test_several_entry_times_in_one_trade_are_said_not_dropped(ag):
    t = "30k account, take it at 10:30, noon and 1:30 whenever it's above VWAP at that time, risk 3% of the account on each one. keep it off the 30k."
    assert [x["text"] for x in I.entry_time_list(t, KNOWN)] == ["10:30", "noon", "1:30"]
    assert not [x for x in I.said(t, KNOWN, trade=True) if x["type"] == "number"], "\"the 30k\" is thousands of dollars wherever it stands"
    assert I.entry_time_list("buy SPY 0dte atm calls at 10 ET, exit at 12 or at 3 pm", KNOWN) is None and I.entry_time_list("compare entries at 9:45, 10:30 and 11:15", KNOWN) is None
    r = route(ag, "buy SPY 0dte atm calls at 9:45 ET and 11:00 ET, sell at 3 pm ET")
    assert r["action"] == "clarify" and r["declined"] and "9:45 AM and 11:00 AM" in r["text"] and r["options"][0]["ask"] == "buy SPY 0dte atm calls at 9:45 ET, sell at 3 pm ET"


def test_a_part_of_the_day_against_the_day_before_it():
    f = SM.read("Does SPY tend to reverse during lunch?", KNOWN, "SPY")
    assert f.ok and (f.frame["start"], f.frame["end"]) == ("11:30", "13:30") and f.frame["outcome"]["direction"] == "opposite" and not f.unread
    assert SM.describe(f.frame) == "On SPY: how often the move from 11:30 to 13:30 New York time goes the other way from the morning's move (the open to 11:30)."
    assert any("\"lunch\" is read as 11:30 to 13:30 New York time" in a["text"] for a in f.frame["assumptions"]), "what the word stands for is said"
    rt = SM.to_route(f)
    assert rt["action"] == "investigate" and [v["label"] for v in rt["plan"]["variants"]] == ["after an up morning", "after a down morning"]
    g = SM.read("does spy reverse over lunch on fridays", KNOWN, "SPY")
    assert [c["key"] for c in g.frame["conditions"]] == ["weekday", "first_candle"] and "when on Fridays" in SM.describe(g.frame)
    assert SM.read("does SPY reverse", KNOWN, "SPY").clarify["text"].startswith("\"Reverse\" after what?"), "with no part of the day and no event, it is asked"
    h = SM.read("How far does QQQ usually bounce after a gap down?", KNOWN, "SPY")
    assert h.ok and h.frame["kind"] == "movement" and not h.unclaimed and not h.unread, "in a how-far question the bounce is the move after the event"


def test_a_reply_that_says_what_a_vague_word_meant_takes_its_place(ag):
    _confirming()
    sid = session()
    m = post(ag, sid, "How far does QQQ usually bounce after a bad open?")
    assert m["content"]["confirm"]["reasons"][0].startswith("Your message gives no size for that"), "three sizes are tried and said, and nothing runs"
    ag.post(sid, "a gap down of at least half a percent")
    st = ag._state(sid)[0]["state"]
    pf = st["pending"]["frame"]  # (since the held-out run the changed reading is shown and waits again)
    assert st["pending"]["notes"] == [] and [(c["key"], c["value"]) for c in pf["conditions"]] == [("gap", -0.5)] and pf["start"] == "open", "the size replaces the vague word, and its warning is gone"
    ag.post(sid, "yes")
    st = ag._state(sid)[0]["state"]
    assert st.get("pending") is None and [(c["key"], c["value"]) for c in st["frame"]["conditions"]] == [("gap", -0.5)]
    sid = session()
    post(ag, sid, "when vix is above 15 how often does qqq close green after a 3% down day")
    ag.post(sid, "what if the drop is 2% instead")
    fr = ag._state(sid)[0]["state"]["pending"]["frame"]
    assert [(c["key"], c.get("value")) for c in fr["conditions"]] == [("vix", 15.0), ("prev_return", -2.0)], "\"the drop\" names yesterday's fall: it is resized, nothing is removed"


def test_a_refused_new_trade_gives_way_to_the_trade_before_it(ag):
    from strata.api import db
    sid = session()
    post(ag, sid, "sell a 5 wide SPY put credit spread 0dte at 10:30 ET, 20 delta, take profit at 50% of the credit, hold to expiration")
    v1 = ag._state(sid)[1]
    db.set_version_runs(v1["id"], [{"id": "r1", "status": "done"}])  # it has run
    post(ag, sid, "new idea: buy a QQQ put at the open, something like two weeks out, sell when vibes turn")
    v2 = ag._state(sid)[1]
    assert v2["id"] != v1["id"] and v2["parent_id"] is None and ag._plan_message(sid, v2["id"])["content"]["state"] == "awaiting_confirmation"
    post(ag, sid, "No, same trade as before, plus a stop: out if SPY drops 0.4% from where it was when I sold the spread.")
    d = draft_of(ag, sid)
    assert (d.symbol, d.structure, d.premium_target_pct, d.underlying_stop_pct) == ("SPY", "bull_put_spread", 50.0, 0.4), "the correction is read against the trade before the refused reading"
    assert ag._plan_message(sid, v2["id"])["content"]["state"] == "set_aside"


def test_times_that_name_the_open_and_noon_are_times(ag):
    assert I.zone_revealed("How often does SPY's first 9:30 candle predict its overall day?") == 0, "9:30 names the opening candle: New York"
    assert I.zone_revealed("does the first 6:30 candle predict the day") == 180 and I.zone_revealed("is it green at 9:30") is None
    assert [(x["type"], x["value"]) for x in I.said("close it at noon", KNOWN) if x["type"] == "clock"] == [("clock", 720)]
    assert not [x for x in I.said("what happens in the afternoon", KNOWN) if x["type"] == "clock"]
    sid = session()
    post(ag, sid, "buy SPY 0dte atm calls at 10 ET, exit at 3 pm ET, stop 30%")
    post(ag, sid, "actually buy a QQQ 0dte atm straddle at the open and close it at noon, I want to see that instead of the calls thing")
    d = draft_of(ag, sid)
    assert (d.symbol, d.structure, d.entry_time, d.exit_time) == ("QQQ", "long_straddle", "open", "12:00"), "every part the message names is changed, noon included"
    sid = session()
    post(ag, sid, "buy 1 SPY 0dte atm straddle at 9:35 ET, hold to the close")
    post(ag, sid, "Add an early exit: take it off as soon as the straddle is worth 40% more than I paid.")
    d = draft_of(ag, sid)
    assert (d.structure, d.premium_target_pct, d.entry_time) == ("long_straddle", 40.0, "09:35"), "a longer message that speaks of the position on screen changes it"


def test_a_research_plan_shows_its_reading_and_never_starts_on_its_own(ag):
    _confirming()
    sid = session()
    m = ag.post(sid, "Is SPY more likely to close higher when the previous day fell more than 1% and VIX is above 15, checked year by year?")[-1]
    if m["kind"] == "graph":  # the older planner took it: nothing runs until the user says so
        assert m["content"]["state"] == "awaiting" and m["content"]["runs"] == []
    # the plan as its sentence, the reason it waits, and its go-ahead: by button or in the user's own words
    g = ag.v5_graph("Is SPY more likely to close higher when the previous day fell more than 1% and VIX is above 15?", KNOWN, "SPY")
    assert g is not None and not g.get("unsupported")
    sid = session()
    c = ag._graph(sid, g, "test", review={"reasons": [], "complex": True})[0]["content"]
    assert c["state"] == "awaiting" and c["confirm"] == {"reasons": [], "complex": True} and c["runs"] == [] and c["text"].startswith("Here is how I read it: ") and "Nothing has been run." in c["text"]
    assert [x["label"] for x in c["choices"]] == ["Yes, run it", "I'll rephrase"]
    out = ag.post(sid, "yep, go ahead")
    assert any(x["kind"] == "graph" and x["content"]["state"] == "running" for x in out), "a typed go-ahead presses the plan's own button"
    # a streak the plan does not hold is not "held" by some other 2 in it
    r = {"action": "graph", "graph": {"question": "red 2 days in a row", "nodes": [{"op": "condition", "params": {"expr": "previous_session.return_pct < 0", "text": "spy is red 2 days in a row"}}], "core": {"horizon": 2}}}
    issues = I.check(r, "if spy is red 2 days in a row does it bounce", KNOWN, None)["issues"]
    assert [i["kind"] for i in issues] == ["unused"] and "2 days in a row" in issues[0]["text"]
    r["graph"]["nodes"][0]["params"]["expr"] = "down_streak >= 2"
    assert I.check(r, "if spy is red 2 days in a row does it bounce", KNOWN, None)["issues"] == []


def test_more_of_what_a_trade_message_says_is_placed_or_said(ag):
    # a window said with "between", a streak said as closes below the day before, a trader's "north of"
    assert I.tidy("how much does QQQ move between 11:00 and 12:00") == "how much does QQQ move from 11:00 to 12:00"
    assert I.tidy("when VIX is between 15 and 20") == "when VIX is between 15 and 20" and I.tidy("is the 10 to 11 hour bigger") == "is from 10:00 to 11:00 bigger"
    assert I.tidy("it closed below the day before two days in a row") == "it closed down two days in a row" and I.tidy("the VIX is north of 25") == "the VIX is above 25"
    f = SM.read(I.tidy("how does QQQ move between 11:00 and 12:00 ET on days it gaps down"), KNOWN, "SPY").frame
    assert (f["kind"], f["start"], f["end"]) == ("movement", "11:00", "12:00")
    # noon in a zone, and what a message says a time is NOT
    c = [x for x in I.said("0dte 30d calls at 0640 on gap down days, STC at +40% / -25% or noon pt", KNOWN, trade=True) if x["type"] == "clock"]
    assert [(x["value"], x["text"]) for x in c] == [(580, "0640"), (900, "noon")], "noon pt is 15:00 in New York"
    c = I.said("the time exit is noon Pacific, 3:00 PM New York, not 3:55", KNOWN, trade=True)
    assert [x.get("negated", False) for x in c if x["type"] == "clock"] == [False, False, True]
    sid = session()
    post(ag, sid, "qqq 0dte 30d calls at 0640 pt on gap down days, STC at +40% / -25% or noon pt. 3 cons.")
    d = draft_of(ag, sid)
    assert (d.symbol, d.entry_time, d.exit_time, d.premium_target_pct, d.premium_stop_pct, d.quantity) == ("QQQ", "09:40", "15:00", 40.0, 25.0, 3)
    post(ag, sid, "the time exit is noon Pacific, 3:00 PM New York, not 3:55")
    assert draft_of(ag, sid).exit_time == "15:00"
    # a move exit on a position that leans neither way is said as either direction; a subject with "itself" keeps its subject
    sid = session()
    post(ag, sid, "long the atm qqq 0dte straddle right at the open. dump it on a .6% move either way, otherwise out at noon ET. 2 lots")
    assert "exit when QQQ has moved 0.6% in either direction from its price at entry" in " ".join(I.describe_draft(draft_of(ag, sid))["lines"])
    x = next(x for x in I.said("take it off if it can be bought back for half the credit received, or if SPY itself has moved more than 0.7% in either direction from where it stood at entry", KNOWN, trade=True) if x["value"] == 0.7)
    assert x["about"] == "underlying"
    # a covered call said by its parts, with its ticker in the sentence that was the writer's own comment
    r = route(ag, "My brother-in-law swears by covered calls on IWM. So test it for me: own 100 shares, sell the call about 30 days out at roughly a 30 delta, let it go to expiration. How did that do against just sitting in the shares?")
    d = r["draft"]
    assert (d.symbol, d.structure, d.dte, d.strike.value, d.hold_to_expiration) == ("IWM", "covered_call", 30, 0.3, True)
    assert any("the comparison with simply holding the shares is not calculated" in x for x in d.review), "what was asked for and is not produced is said"
    r = route(ag, "buy 100 shares of SPY at the open and sell at the close")
    assert r["declined"] and r["text"].startswith("Strata backtests option trades, not trades in the shares themselves") and r["options"][0]["ask"] == "What does SPY usually do from the open to the close?"


def test_the_question_readers_own_reading_is_shown_when_a_plan_lost_part_of_it(ag):
    r = route(ag, "On SPY from 2020 onward, when it closed below the day before two days in a row and then opens below yesterday's close, how often does it close higher than where it opened that morning versus lower?")
    assert r["action"] == "clarify" and r["pending"]["kind"] == "confirm_frame"
    fr = r["pending"]["frame"]
    assert [c["key"] for c in fr["conditions"]] == ["down_streak", "rel"] and fr["outcome"]["rel"] == {"a": "close", "op": ">", "b": "open"} and fr["period"][0] == "2020-01-01"


def test_a_distance_from_the_money_is_a_strike_and_the_day_before_is_not_a_weekday(ag):
    r = route(ag, "Sell the 0dte QQQ put 10 points below the money at 9:45 eastern, one lot, close it at 20%.")
    d = r["draft"]
    assert (d.symbol, d.structure, d.strike.method, d.strike.value, d.premium_target_pct) == ("QQQ", "short_put", "strikes_otm", 10.0, 20.0)
    d = route(ag, "sell a SPY 0dte put credit spread, short strike 5 points below the money, 5 wide, at 10 ET, hold to expiration")["draft"]
    assert (d.strike.method, d.strike.value, d.width) == ("strikes_otm", 5.0, 5.0), "where the short strike is, and how wide the spread is, are two things"
    d = route(ag, "buy SPY 0dte calls $3 otm at 10 ET, out at noon ET")["draft"]
    assert (d.strike.method, d.strike.value, d.exit_time, d.sizing) == ("strikes_otm", 3.0, "12:00", None), "$3 otm is a distance, not an amount of money"
    t = I.tidy("spy, mondays only, where the friday before was a down day and vix finished that friday above 20. what time of day does the low usually print?")
    f = SM.read(t, KNOWN, "SPY")
    assert f.frame["kind"] == "timing" and f.frame["measure"] == "low_time" and [c["key"] for c in f.frame["conditions"]] == ["weekday", "prev_return", "vix"] and not f.unread and not f.unclaimed
    assert [c["value"] for c in f.frame["conditions"] if c["key"] == "weekday"] == ["Mon"], "the Friday before a Monday is the day before, not a second weekday"
    assert SM.read(I.tidy("is monday or friday more likely to close red"), KNOWN, "SPY").frame["kind"] == "breakdown"
