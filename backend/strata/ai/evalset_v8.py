"""V8 evaluation of understanding: sentences, conversations and answers, written before the V8 reader existed.

THE SEALED PART (SEALED, SEALED_CONVERSATIONS, SEALED_ANSWERS) MUST NOT BE EDITED, read while developing, or used to tune
anything. It is held by hash (`sealed_hash`, checked by a test) and scored once, at the end, and the score is reported as a
first attempt only if nothing was changed in response to it. The development part (DEVELOPMENT, DEV_CONVERSATIONS) is what
the reader is built and tested against; it includes the examples of the V8 brief.

Both parts were written by the same author as the reader. That is disclosed with every score: the sealed sentences were
unseen by the code, not by the person who wrote it.

What an item expects is written in a canonical vocabulary that does not depend on any reader (see eval_v8.facts):

  type        frequency | agreement | movement | timing | breakdown | strategy | explore | compare | explain
  symbol      the underlying; symbols for a question asked of several at once
  conditions  canonical comparisons, matched as a set: "prev_return<=-2", "gap>=1", "weekday=Mon", "vix>25",
              "up_streak>=3", "first_candle:15:down", "above_vwap" ... (> and >= are the same on continuous data;
              a value of * matches any)
  sequence    ordered stages, each a canonical comparison
  outcome     close>open | close<open | close>prev_close | close<prev_close | close>start | close<start
              | touch:+0.5 | touch:-1 | touch:prev_close | touch:prev_low | touch:open | end_return | overnight ...
  start, end  open | close | HH:MM (New York) | sessions:N
  candle      candle lengths in minutes (agreement); scope rest | day | both; predictor first_candle | prev_direction
  by, measure breakdowns and timing profiles: weekday | month | prev_return_decile ; high_time | low_time | abs_move
  structure, dte, strike, quantity, entry, exit, premium_target, premium_stop, underlying_target, underlying_stop, hold
  period      [start, end] (None for an open end)

  A list where one value is expected means "any of these". A key set to None means "must be absent".
  clarify     the correct response is a question, before anything runs
  flag        the correct response says what cannot be done or represented, and runs nothing
  clarify_ok  a question is an acceptable response as well as the reading(s) given
  alternatives  further fact sets, any one of which is correct
"""
from __future__ import annotations

import hashlib
import json


def S(text: str, level: str, category: str, alternatives: list[dict] | None = None, **expect) -> dict:
    return {"text": text, "level": level, "category": category, "expect": expect, "alternatives": alternatives or []}


def T(text: str, alternatives: list[dict] | None = None, **expect) -> dict:
    return {"text": text, "expect": expect, "alternatives": alternatives or []}


# =============================================================================================================== SEALED
SEALED = [
    # ---- questions about how often something happens
    S("how often does SPY close higher than it opened on mondays?", "beginner", "frequency",
      type="frequency", symbol="SPY", conditions=["weekday=Mon"], outcome="close>open"),
    S("what percent of the time does qqq close below its open after a gap up of 1% or more", "beginner", "informal",
      type="frequency", symbol="QQQ", conditions=["gap>=1"], outcome="close<open"),
    S("After IWM drops 2% or more in a day, how often does it close higher the next day?", "intermediate", "frequency",
      type="frequency", symbol="IWM", conditions=["prev_return<=-2"], outcome=["close>prev_close", "close>open"]),
    S("How often does SPY touch +0.5% from the open before 11am ET?", "intermediate", "frequency",
      type="frequency", symbol="SPY", conditions=[], outcome="touch:+0.5", start="open", end="11:00"),
    S("P(SPY trades below the prior day's low within the first 30 minutes | gap down > 0.5%)", "expert", "frequency",
      type="frequency", symbol="SPY", conditions=["gap<=-0.5"], outcome="touch:prev_low", start="open", end="10:00"),
    S("does SPY usually go back up after a really bad day?", "beginner", "ambiguous", clarify_ok=True,
      type="frequency", symbol="SPY", conditions=["prev_return<=*"], outcome=["close>open", "close>prev_close"]),
    S("how often is the first 15 minutes red but spy still ends green?", "beginner", "informal",
      alternatives=[{"type": "agreement", "symbol": "SPY", "candle": [15], "scope": ["day", "both"]}],
      type="frequency", symbol="SPY", conditions=["first_candle:15:down"], outcome="close>open"),
    S("how often does spy fill the gap when it gaps down more than half a percent", "intermediate", "informal",
      type="frequency", symbol="SPY", conditions=["gap<=-0.5"], outcome="touch:prev_close"),
    S("Conditional on VIX closing above 25 the prior day, what's the hit rate for QQQ trading 1% above the open intraday?", "expert", "frequency",
      type="frequency", symbol="QQQ", conditions=["vix>25"], outcome="touch:+1", start="open"),
    S("What happens to SPY after it falls 3% in one day?", "beginner", "movement",
      alternatives=[{"type": "frequency", "symbol": "SPY", "conditions": ["prev_return<=-3"], "outcome": ["close>open", "close>prev_close"]}],
      type="movement", symbol="SPY", conditions=["prev_return<=-3"]),
    S("Show me how QQQ trades in the first hour after a gap up over 1%", "intermediate", "movement",
      type="movement", symbol="QQQ", conditions=["gap>1"], start="open", end="10:30"),
    # ---- does one thing predict another
    S("is the first 30 minutes a good sign of how the day ends for spy?", "beginner", "agreement",
      type="agreement", symbol="SPY", candle=[30]),
    S("Does IWM's opening 5-minute candle predict the direction of the rest of the session?", "intermediate", "agreement",
      type="agreement", symbol="IWM", candle=[5], scope="rest"),
    S("When SPY's first 15m bar is green, what's the probability the 15:59 close is above the 9:45 price?", "expert", "agreement",
      alternatives=[{"type": "agreement", "symbol": "SPY", "candle": [15], "scope": ["rest", "both"]}],
      type="frequency", symbol="SPY", conditions=["first_candle:15:up"], outcome="close>start", start="09:45"),
    S("does yesterdays direction predict todays for qqq", "beginner", "informal",
      type="agreement", symbol="QQQ", predictor="prev_direction"),
    # ---- when during the day
    S("when during the day does spy usually make its high?", "beginner", "timing",
      type="timing", symbol="SPY", measure="high_time"),
    S("Which half hour of the session has the biggest moves in QQQ?", "intermediate", "timing",
      type="timing", symbol="QQQ", measure="abs_move", bucket=30),
    S("intraday volatility profile for IWM, 15-minute buckets, since 2020", "expert", "timing",
      type="timing", symbol="IWM", measure="abs_move", bucket=15, period=["2020-01-01", None]),
    # ---- breakdowns
    S("which day of the week is best for SPY?", "beginner", "breakdown",
      type="breakdown", symbol="SPY", by="weekday"),
    S("Break down QQQ's average open-to-close return by month of the year", "intermediate", "breakdown",
      type="breakdown", symbol="QQQ", by="month", measure="open_to_close"),
    # ---- option strategies
    S("what if i bought spy calls every morning and sold at lunch?", "beginner", "informal",
      type="strategy", symbol="SPY", structure="long_call", conditions=[], entry=["open", "09:30"], exit=["12:00", "15:00"]),
    S("Backtest buying 1 ATM QQQ put expiring same day at 10:15 ET whenever QQQ gapped up 0.75%+, target +40% on the premium, stop -20%", "intermediate", "strategy",
      type="strategy", symbol="QQQ", structure="long_put", dte=0, strike="atm", quantity=1, entry="10:15", conditions=["gap>=0.75"], premium_target=40, premium_stop=20),
    S("Long SPY 7DTE 30-delta calls at 9:45 ET on days the prior session closed down >1%, exit when SPY trades +0.75% from entry or at 15:30 ET, whichever first", "expert", "multi_condition",
      type="strategy", symbol="SPY", structure="long_call", dte=7, strike="delta:30", entry="09:45", conditions=["prev_return<-1"], underlying_target=0.75, exit="15:30"),
    S("if i buy a put on iwm when it opens up a lot, do i make money?", "beginner", "ambiguous", clarify_ok=True,
      type="strategy", symbol="IWM", structure="long_put", conditions=["gap>=*"]),
    S("Buy SPY 0DTE calls at the open and take profit when SPY is up 0.5%", "intermediate", "underlying_vs_premium",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry=["open", "09:30"], underlying_target=0.5, premium_target=None),
    S("Buy SPY 0DTE calls at the open and take profit when the calls are up 50%", "intermediate", "underlying_vs_premium",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry=["open", "09:30"], premium_target=50, underlying_target=None),
    S("Buy SPY 0DTE calls at the open, take profit at 25%", "intermediate", "ambiguous", clarify=True),
    S("Sell a 1-wide SPY put credit spread, 5 points OTM, 0DTE, at 10:00 ET when VIX < 15, hold to expiration", "expert", "multi_condition",
      type="strategy", symbol="SPY", structure="bull_put_spread", dte=0, entry="10:00", conditions=["vix<15"], hold="to_expiration"),
    S("Buy QQQ calls at 10am eastern only when it gapped down at least 0.5% and yesterday was a red day, sell at 3pm eastern", "intermediate", "multi_condition",
      type="strategy", symbol="QQQ", structure="long_call", entry="10:00", exit="15:00", conditions=["gap<=-0.5", "prev_return<0"]),
    S("Enter SPY 0DTE puts after it first rallies 0.3% off the open and then breaks back below the open price; out at 14:00 ET", "expert", "multi_condition",
      type="strategy", symbol="SPY", structure="long_put", dte=0, sequence=["from_open>=0.3", "from_open<0"], exit="14:00"),
    # ---- clock times
    S("buy spy calls at 7am pacific, exit 12:30 pacific", "intermediate", "timezone",
      type="strategy", symbol="SPY", structure="long_call", entry="10:00", exit="15:30"),
    S("Buy IWM puts at 11 central time", "intermediate", "timezone",
      type="strategy", symbol="IWM", structure="long_put", entry="12:00"),
    S("buy qqq calls at 9:45, sell at 3:45", "beginner", "timezone",
      type="strategy", symbol="QQQ", structure="long_call", entry="09:45", exit="15:45"),
    S("Buy SPY calls at 10:30 my time", "beginner", "timezone",
      type="strategy", symbol="SPY", structure="long_call", entry="13:30"),
    # ---- periods
    S("how did buying spy 0dte calls after red days do in 2022?", "beginner", "data",
      type="strategy", symbol="SPY", structure="long_call", dte=0, conditions=["prev_return<0"], period=["2022-01-01", "2022-12-31"]),
    S("Since 2010, how often does SPY close above its open when it gapped down 1% or more?", "intermediate", "frequency",
      type="frequency", symbol="SPY", conditions=["gap<=-1"], outcome="close>open", period=["2010-01-01", None]),
    # ---- things that must be asked about or declined, not answered with something else
    S("how often does IWM bounce back after dropping 1.5% in the morning?", "beginner", "ambiguous", clarify_ok=True,
      type="frequency", symbol="IWM", conditions=["from_open<=-1.5"], outcome=["touch:open", "close>start"]),
    S("what about tesla", "beginner", "ambiguous", flag=True),
    S("will spy go up tomorrow?", "beginner", "ambiguous", flag=True),
    S("why did my trade lose money?", "beginner", "ambiguous", flag=True),
    S("is buying qqq puts at the close every day profitable?", "beginner", "strategy",
      type="strategy", symbol="QQQ", structure="long_put", conditions=[], entry=["close", "15:55", "15:59", "15:50", "16:00"]),
    S("What's the t-stat on SPY's overnight return after a down close?", "expert", "frequency", flag=True,
      alternatives=[{"type": "movement", "symbol": "SPY", "conditions": ["prev_return<0"], "outcome": "overnight"}]),
    S("compare SPY and QQQ: how often each closes above the open after a gap down of 1%", "intermediate", "frequency",
      type="frequency", symbols=["QQQ", "SPY"], conditions=["gap<=-1"], outcome="close>open"),
    S("do 0dte calls print on fomc days?", "intermediate", "data", flag=True),
    S("Bucket SPY sessions by prior-day return deciles and show next-day open-to-close win rate per decile", "expert", "breakdown", flag=True,
      alternatives=[{"type": "breakdown", "symbol": "SPY", "by": "prev_return_decile"}]),
    S("how much does spy usually move in a day?", "beginner", "movement",
      type="movement", symbol="SPY", conditions=[]),
    S("Odds QQQ closes red after 3 green days in a row?", "intermediate", "informal",
      type="frequency", symbol="QQQ", conditions=["up_streak>=3"], outcome=["close<open", "close<prev_close"]),
    S("How often does SPY reclaim VWAP within an hour after losing it in the first 30 minutes?", "expert", "multi_condition", flag=True,
      alternatives=[{"type": "frequency", "symbol": "SPY", "sequence": ["crosses_below:vwap", "crosses_above:vwap"]}]),
    S("spy red open candle -> green close how often", "intermediate", "informal", clarify_ok=True,
      type="frequency", symbol="SPY", conditions=["first_candle:*:down"], outcome="close>open"),
    S("qqq 0dte puts on gap ups >1%, 30% tp / 20% sl, in at 9:35 et", "expert", "informal",
      type="strategy", symbol="QQQ", structure="long_put", dte=0, entry="09:35", conditions=["gap>1"], premium_target=30, premium_stop=20),
    S("how often does iwm close within 0.2% of its open", "intermediate", "frequency", flag=True,
      alternatives=[{"type": "frequency", "symbol": "IWM", "outcome": "abs_close_open<=0.2"}]),
    S("whats the win rate if i buy spy calls when its above vwap at 11am et and hold til close", "intermediate", "informal",
      type="strategy", symbol="SPY", structure="long_call", entry="11:00", conditions=["above_vwap"], exit=["close", "15:55", "15:59", "16:00"]),
]

SEALED_CONVERSATIONS = [
    {"id": "c1", "level": "beginner", "turns": [
        T("how often does iwm close above its open after a down day", type="frequency", symbol="IWM", conditions=["prev_return<0"], outcome="close>open"),
        T("what about 2% down days", type="frequency", symbol="IWM", conditions=["prev_return<=-2"], outcome="close>open"),
        T("and for spy?", type="frequency", symbol="SPY", conditions=["prev_return<=-2"], outcome="close>open"),
        T("only since 2018", type="frequency", symbol="SPY", conditions=["prev_return<=-2"], period=["2018-01-01", None]),
        T("undo", type="frequency", symbol="SPY", conditions=["prev_return<=-2"], period=[None, None]),
    ]},
    {"id": "c2", "level": "intermediate", "turns": [
        T("does the first 10 minute candle tell you anything about the rest of the day in QQQ?", type="agreement", symbol="QQQ", candle=[10], scope="rest"),
        T("try 30 minutes instead", type="agreement", symbol="QQQ", candle=[30], scope="rest"),
        T("spy", type="agreement", symbol="SPY", candle=[30]),
        T("only when it gapped up", type="agreement", symbol="SPY", candle=[30], conditions=["gap>0"]),
        T("ok so buy calls when that candle is green", type="strategy", symbol="SPY", structure="long_call", entry="10:00", conditions=["first_candle:30:up", "gap>0"]),
    ]},
    {"id": "c3", "level": "intermediate", "turns": [
        T("Buy 2 SPY 0DTE calls at 9:40 ET after a 1% down day, sell at 15:00 ET", type="strategy", symbol="SPY", structure="long_call", dte=0, quantity=2, entry="09:40", exit="15:00", conditions=["prev_return<=-1"]),
        T("make it puts", type="strategy", symbol="SPY", structure="long_put", quantity=2, entry="09:40", exit="15:00", conditions=["prev_return<=-1"]),
        T("enter 20 min later", type="strategy", symbol="SPY", structure="long_put", entry="10:00", exit="15:00"),
        T("add a 25% stop on the option", type="strategy", symbol="SPY", structure="long_put", entry="10:00", premium_stop=25),
        T("exit at the close instead", type="strategy", symbol="SPY", structure="long_put", premium_stop=25, exit=["close", "15:55", "15:59", "16:00"]),
        T("compare with the previous version", type="compare"),
    ]},
    {"id": "c4", "level": "beginner", "turns": [
        T("how often does spy drop 1% from the open before noon eastern", type="frequency", symbol="SPY", outcome="touch:-1", start="open", end="12:00"),
        T("no, 0.75", type="frequency", symbol="SPY", outcome="touch:-0.75", end="12:00"),
        T("on fridays", type="frequency", symbol="SPY", outcome="touch:-0.75", conditions=["weekday=Fri"]),
        T("sorry, thursdays", type="frequency", symbol="SPY", outcome="touch:-0.75", conditions=["weekday=Thu"]),
        T("show me the days it happened", type="explore"),
    ]},
    {"id": "c5", "level": "intermediate", "turns": [
        T("buy qqq 0dte puts at 7:15am pt when qqq gapped up 1%, out at 9am pt", type="strategy", symbol="QQQ", structure="long_put", dte=0, entry="10:15", exit="12:00", conditions=["gap>=1"]),
        T("make the exit 30 min later", type="strategy", symbol="QQQ", entry="10:15", exit="12:30"),
        T("and use iwm", type="strategy", symbol="IWM", entry="10:15", exit="12:30", conditions=["gap>=1"]),
        T("what if the gap is only half a percent", type="strategy", symbol="IWM", conditions=["gap>=0.5"]),
    ]},
    {"id": "c6", "level": "intermediate", "turns": [
        T("Buy SPY calls at the open when SPY gapped down 1% and VIX is above 20, sell at 11am eastern", type="strategy", symbol="SPY", structure="long_call", entry=["open", "09:30"], exit="11:00", conditions=["gap<=-1", "vix>20"]),
        T("make it 2", clarify=True),
        T("the gap", type="strategy", symbol="SPY", conditions=["gap<=-2", "vix>20"]),
    ]},
    {"id": "c7", "level": "beginner", "turns": [
        T("what does QQQ do in the first 2 hours after a 2% gap down?", type="movement", symbol="QQQ", conditions=["gap<=-2"], start="open", end="11:30"),
        T("same for gap ups", type="movement", symbol="QQQ", conditions=["gap>=2"]),
        T("show me the worst ones", type="explore"),
    ]},
    {"id": "c8", "level": "intermediate", "turns": [
        T("how often does SPY rise 0.5% from the open on days after a 1.5% drop", type="frequency", symbol="SPY", conditions=["prev_return<=-1.5"], outcome="touch:+0.5"),
        T("what if i bought calls on those days", type="strategy", symbol="SPY", structure="long_call", conditions=["prev_return<=-1.5"]),
        T("use 1 dte", type="strategy", symbol="SPY", structure="long_call", dte=1),
        T("why did the worst trade lose?", type="explain"),
    ]},
]

# Answers: the question is run on the stored history and the reply is checked against a calculation made here, directly
# on the stored bars, with none of the application's code (`reference`), or for statements it must and must not make.
SEALED_ANSWERS = [
    {"id": "a1", "text": "How often did SPY close above its open after a 2% or larger down day, since 1993?", "reference": "a1", "numbers": ["pct", "k"]},
    {"id": "a2", "text": "For QQQ, how often does the rest of the day go the same way as the first 15-minute candle?", "reference": "a2", "numbers": ["pct"], "must": [["baseline", "unrelated", "guess", "chance"]]},
    {"id": "a3", "text": "Which weekday has the highest average open-to-close return for IWM?", "reference": "a3", "numbers": [], "names": ["weekday"]},
    {"id": "a4", "text": "When during the day does SPY most often set its daily high?", "reference": "a4", "numbers": [], "names": ["bucket"]},
    {"id": "a5", "text": "Is buying ATM SPY 0DTE calls at the open after a 1% down day a good strategy or did it just get lucky?", "reference": None, "numbers": [],
     "must": [["luck", "chance", "can't tell", "cannot tell", "not distinguishable", "no evidence", "not enough evidence"]], "consistent": True},
    {"id": "a6", "text": "Backtest buying SPY 0DTE calls every Monday at the open during 2021", "reference": None, "numbers": [],
     "must": [["2024", "January 18"]], "no_trades_in": "2021"},
    {"id": "a7", "text": "How often does IWM gap up more than 1%?", "reference": "a7", "numbers": ["pct"]},
    {"id": "a8", "text": "What's the average move in SPY from the open to the close on days it gaps down at least 1%?", "reference": "a8", "numbers": ["mean"]},
    {"id": "a9", "text": "How likely is QQQ to finish the day above yesterday's close after opening more than 0.5% below it?", "reference": "a9", "numbers": ["pct"]},
    {"id": "a10", "turns": ["Buy ATM QQQ 0DTE calls at the open after a 1% down day, sell at 3:55 pm eastern", "why did the worst trade lose?"], "reference": "worst_trade", "numbers": ["pnl"], "names": ["date"]},
]
# Statements no answer may make, in any item (a p-value is not the probability that a result is due to chance or that a
# strategy is profitable; a backtest proves nothing)
FORBIDDEN = ["probability that the strategy", "probability that this strategy", "chance that this is luck", "probability that the result is due to chance", "proven profitable",
             "guaranteed", "proves that", "will make money", "risk-free"]


# ---------------------------------------------------------------------------------------------- independent references
def _daily(store, sym: str):
    import polars as pl
    d = store.read_daily(sym).sort("date")
    return d.with_columns(prev_close=pl.col("close").shift(1), prev2_close=pl.col("close").shift(2))


def _minutes(store, sym: str):
    """The provider's one-minute bars of the regular session, by New York date and minute of the day."""
    import polars as pl
    m = store.read_minutes(sym, provider="alpaca")
    m = m.with_columns(et=pl.col("ts").dt.convert_time_zone("America/New_York"))
    m = m.with_columns(d=pl.col("et").dt.date(), mod=pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32))
    return m.filter((pl.col("mod") >= 570) & (pl.col("mod") < 960))


def reference(name: str, store) -> dict:
    import polars as pl
    if name == "a1":
        d = _daily(store, "SPY").with_columns(prev_ret=(pl.col("prev_close") / pl.col("prev2_close") - 1) * 100).drop_nulls("prev_ret")
        q = d.filter(pl.col("prev_ret") <= -2)
        k = q.filter(pl.col("close") > pl.col("open")).height
        return {"n": q.height, "k": k, "pct": k / q.height * 100}
    if name == "a2":
        m = _minutes(store, "QQQ")
        g = m.group_by("d").agg(o=pl.col("open").sort_by("mod").first(), first_mod=pl.col("mod").min(), c15=pl.col("close").filter(pl.col("mod") == 584).first(),
                                close=pl.col("close").sort_by("mod").last())
        g = g.filter((pl.col("first_mod") == 570) & pl.col("c15").is_not_null())
        g = g.with_columns(cand=(pl.col("c15") - pl.col("o")).sign(), rest=(pl.col("close") - pl.col("c15")).sign()).filter(pl.col("cand") != 0)
        agree = g.filter(pl.col("cand") == pl.col("rest")).height
        return {"n": g.height, "agree": agree, "pct": agree / g.height * 100, "rest_up_pct": float((g["rest"] > 0).mean() * 100)}
    if name == "a3":
        d = _daily(store, "IWM").with_columns(oc=(pl.col("close") / pl.col("open") - 1) * 100, wd=pl.col("date").dt.weekday())
        w = d.group_by("wd").agg(mean=pl.col("oc").mean(), n=pl.len()).sort("mean", descending=True)
        best = w.row(0, named=True)
        return {"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][best["wd"] - 1], "mean": best["mean"], "n": best["n"]}
    if name == "a4":
        m = _minutes(store, "SPY")
        g = m.sort(["d", "mod"]).group_by("d").agg(first_mod=pl.col("mod").min(), hi_mod=pl.col("mod").sort_by("high", descending=True, maintain_order=True).first())
        g = g.filter(pl.col("first_mod") == 570).with_columns(bucket=(pl.col("hi_mod") - 570) // 30)
        c = g.group_by("bucket").agg(n=pl.len()).sort("n", descending=True)
        b = int(c["bucket"][0])
        s = 570 + 30 * b
        return {"bucket": f"{s // 60:02d}:{s % 60:02d}", "share": float(c["n"][0] / g.height * 100), "sessions": g.height}
    if name == "a7":
        d = _daily(store, "IWM").drop_nulls("prev_close").with_columns(gap=(pl.col("open") / pl.col("prev_close") - 1) * 100)
        k = d.filter(pl.col("gap") > 1).height
        return {"n": d.height, "k": k, "pct": k / d.height * 100}
    if name == "a8":
        d = _daily(store, "SPY").drop_nulls("prev_close").with_columns(gap=(pl.col("open") / pl.col("prev_close") - 1) * 100, oc=(pl.col("close") / pl.col("open") - 1) * 100)
        q = d.filter(pl.col("gap") <= -1)
        return {"n": q.height, "mean": float(q["oc"].mean())}
    if name == "a9":
        d = _daily(store, "QQQ").drop_nulls("prev_close").with_columns(gap=(pl.col("open") / pl.col("prev_close") - 1) * 100)
        q = d.filter(pl.col("gap") < -0.5)
        k = q.filter(pl.col("close") > pl.col("prev_close")).height
        return {"n": q.height, "k": k, "pct": k / q.height * 100}
    raise KeyError(name)


def sealed_payload() -> str:
    return json.dumps({"sealed": SEALED, "conversations": SEALED_CONVERSATIONS, "answers": SEALED_ANSWERS, "forbidden": FORBIDDEN}, sort_keys=True)


def sealed_hash() -> str:
    """The hash of the sealed items. Recorded in docs/v8_evidence/sealed_v8.txt when the set was written; a test holds it."""
    return hashlib.sha256(sealed_payload().encode()).hexdigest()


# ========================================================================================================== DEVELOPMENT
# The examples of the V8 brief, and sentences written for building the reader. Anything here may be used and changed.
DEVELOPMENT = [
    S("How often does SPY reverse after a red opening candle?", "beginner", "brief", clarify_ok=True,
      type="frequency", symbol="SPY", conditions=["first_candle:*:down"], outcome=["close>start", "close>open"]),
    S("What happens if I buy calls every morning?", "beginner", "brief", type="strategy", symbol="SPY", structure="long_call", conditions=[], entry=["open", "09:30"]),
    # V8 policy (changed during development): "recover" is measured in each of its usual readings, side by side, not asked about
    S("Does QQQ usually recover after falling 1%?", "beginner", "brief", type="frequency", symbol="QQQ", conditions=["prev_return<=-1"], outcome="close>prev_close"),
    S("Which weekdays tend to perform better?", "beginner", "brief", type="breakdown", symbol="SPY", by="weekday"),
    S("How often is SPY's first candle right about the rest of the day?", "beginner", "brief", type="agreement", symbol="SPY", scope="rest"),
    S("How often is SPY's overall day, predicted by the first 9:30 open candle?", "beginner", "typed", type="agreement", symbol="SPY"),
    S("How often does SPY's first 9:30ET candle, predict its overall day?", "beginner", "typed", type="agreement", symbol="SPY"),
    S("Is the first five-minute candle a good predictor of the day?", "beginner", "brief", type="agreement", symbol="SPY", candle=[5]),
    S("What happens after SPY drops 2%?", "beginner", "brief", type="movement", symbol="SPY", conditions=["prev_return<=-2"]),
    S("How often does QQQ recover from a bad morning?", "beginner", "brief", clarify_ok=True, type="frequency", symbol="QQQ"),
    S("Is buying calls every morning profitable historically?", "beginner", "brief", type="strategy", symbol="SPY", structure="long_call", entry=["open", "09:30"]),
    S("How differently do 0DTE and 7DTE options behave?", "beginner", "brief", clarify_ok=True, type="compare"),
    S("When during the day does SPY tend to move the most?", "beginner", "brief", type="timing", symbol="SPY", measure="abs_move"),
    S("how often does spy gap down more than 1%", "beginner", "dev", type="frequency", symbol="SPY", outcome="event:gap<=-1"),
    S("what time does qqq usually bottom out", "beginner", "dev", type="timing", symbol="QQQ", measure="low_time"),
    S("average open to close move for spy by weekday", "intermediate", "dev", type="breakdown", symbol="SPY", by="weekday", measure="open_to_close"),
    S("how often does iwm close red after a green first 30 minutes", "intermediate", "dev", type="frequency", symbol="IWM", conditions=["first_candle:30:up"], outcome="close<open"),
    S("does the opening 15 min candle predict the rest of the day for iwm", "intermediate", "dev", type="agreement", symbol="IWM", candle=[15], scope="rest"),
    S("how does spy trade in the first 30 minutes after a 1% gap up", "intermediate", "dev", type="movement", symbol="SPY", conditions=["gap>=1"], end="10:00"),
    S("buy spy 0dte calls at 6:30a pst after a 1% down day, sell at 12:45p pst", "intermediate", "dev", type="strategy", symbol="SPY", structure="long_call", dte=0, entry="09:30", exit="15:45", conditions=["prev_return<=-1"]),
]
# PROBE: written on 2026-11-26, after DEVELOPMENT had been tuned to 20/20, to check mid-development that the layer is not
# fitted to DEVELOPMENT alone. Scored once as written (docs/v8_evidence/probe_v8_first_score.json); it is development data
# from then on, so its later scores are regression lines, not estimates. The sealed set above was not opened to write it.
PROBE = [
    S("how often does iwm close red after gapping up 1% or more", "beginner", "probe", type="frequency", symbol="IWM", conditions=["gap>=1"], outcome=["close<open", "close<prev_close"]),
    S("what % of days does spy trade above yesterday's high", "beginner", "probe", type="frequency", symbol="SPY", conditions=[], outcome="touch:prev_high"),
    S("chances qqq finishes green when it opens down half a percent or more", "beginner", "probe", clarify_ok=True, type="frequency", symbol="QQQ", conditions=["gap<=-0.5"], outcome=["close>open", "close>prev_close"]),
    S("how many times has spy fallen 3% in one day since 2020", "beginner", "probe", type="frequency", symbol="SPY", outcome=["event:day_return<=-3", "event:prev_return<=-3", "event:move<=-3"], period=["2020-01-01", None]),
    S("on fridays how often does qqq close higher than it opened", "beginner", "probe", type="frequency", symbol="QQQ", conditions=["weekday=Fri"], outcome="close>open"),
    S("how often does spy fill the gap when it gaps down more than 0.5%", "intermediate", "probe", type="frequency", symbol="SPY", conditions=["gap<=-0.5"], outcome="touch:prev_close"),
    S("what does spy usually do in the first hour after a 1% gap down", "intermediate", "probe", type="movement", symbol="SPY", conditions=["gap<=-1"], end="10:30"),
    S("is the opening 15 min candle predictive for QQQ", "beginner", "probe", type="agreement", symbol="QQQ", candle=[15]),
    S("does yesterday's direction predict today for IWM", "intermediate", "probe", type="agreement", symbol="IWM", predictor="prev_direction"),
    S("what time of day does qqq usually put in its high", "beginner", "probe", type="timing", symbol="QQQ", measure="high_time"),
    S("when is spy most volatile during the day", "beginner", "probe", type="timing", symbol="SPY", measure="abs_move"),
    S("which month has been best for spy", "beginner", "probe", type="breakdown", symbol="SPY", by="month"),
    S("is monday a bad day for iwm", "beginner", "probe", type="breakdown", symbol="IWM", by="weekday"),
    S("buy spy puts at 10am et if spy is down 0.5% from the open, sell at 3pm et", "intermediate", "probe", type="strategy", symbol="SPY", structure="long_put", entry="10:00", exit="15:00", conditions=["from_open<=-0.5"]),
    S("backtest buying 7dte qqq calls at the open when vix closed above 25 the day before", "experienced", "probe", type="strategy", symbol="QQQ", structure="long_call", dte=7, entry=["open", "09:30"], conditions=["vix>25"]),
    S("how often does spy rise after an fomc announcement", "intermediate", "probe", flag=True),
    S("will spy go up tomorrow", "beginner", "probe", flag=True),
    S("does spy rally when the dollar is weak", "intermediate", "probe", flag=True),
    S("how does TSLA trade after earnings", "beginner", "probe", flag=True),
    S("hw often does spy dump 2% intrday", "beginner", "probe", type="frequency", symbol="SPY", conditions=[], outcome=["touch:-2"]),
    S("qqq green after a red open? how often", "beginner", "probe", clarify_ok=True, type="frequency", symbol="QQQ", conditions=["gap<0"], outcome=["close>open", "close>prev_close"]),
    S("how often does spy hit +0.5% before 7am pacific", "intermediate", "probe", type="frequency", symbol="SPY", outcome="touch:+0.5", end="10:00"),
    # expectation widened after the first score: "up at 10:30" as quoted (against the previous close) is at least as usual a reading as "above the open"
    S("if spy is up at 10:30 CT does it usually close up", "intermediate", "probe", [dict(type="frequency", symbol="SPY", conditions=["first_candle:120:up"], start="11:30", outcome=["close>start", "close>open", "close>prev_close"])],
      clarify_ok=True, type="frequency", symbol="SPY", conditions=["change>0"], start="11:30", outcome=["close>start", "close>open", "close>prev_close"]),
    S("after 3 red days in a row how often is the next day green for spy", "intermediate", "probe", type="frequency", symbol="SPY", conditions=["down_streak>=3"], outcome=["close>open", "close>prev_close"]),
    S("how often does qqq drop 1% from the open and still close green", "intermediate", "probe", type="frequency", symbol="QQQ", conditions=["from_open<=-1"], outcome=["close>open", "close>prev_close"]),
    S("what usually happens to iwm after it falls 2.5% in a day", "beginner", "probe", type="movement", symbol="IWM", conditions=["prev_return<=-2.5"]),
    S("how often is spy higher a week after a 2% down day", "intermediate", "probe", type="frequency", symbol="SPY", conditions=["prev_return<=-2"], outcome=["close>prev_close", "close>open", "close>start"], end="sessions:5"),
    S("does qqq tend to keep falling after a 2% drop", "intermediate", "probe", type="frequency", symbol="QQQ", conditions=["prev_return<=-2"], outcome=["close<prev_close", "close<open"]),
    S("how often does spy close within 0.2% of its open", "experienced", "probe", clarify_ok=True, type="frequency", symbol="SPY"),
    S("sell spy 0dte puts at noon et every day and buy them back at 3:50pm et", "experienced", "probe", type="strategy", symbol="SPY", structure="short_put", dte=0, entry="12:00", exit="15:50", conditions=[]),
]
PROBE_CONVERSATIONS = [
    {"id": "probe-gap", "turns": [
        T("how often does qqq gap up more than 1%?", type="frequency", symbol="QQQ", outcome="event:gap>=1"),
        T("what about iwm", type="frequency", symbol="IWM", outcome="event:gap>=1"),
        T("only since 2022", type="frequency", symbol="IWM", outcome="event:gap>=1", period=["2022-01-01", None]),
        T("show me those days", type="explore"),
    ]},
    {"id": "probe-follow", "turns": [
        T("How often does SPY close above its open after a 1% gap down?", type="frequency", symbol="SPY", conditions=["gap<=-1"], outcome="close>open"),
        T("make it 2%", type="frequency", symbol="SPY", conditions=["gap<=-2"], outcome="close>open"),
        T("and only on mondays", type="frequency", symbol="SPY", conditions=["gap<=-2", "weekday=Mon"], outcome="close>open"),
        T("undo", type="frequency", symbol="SPY", conditions=["gap<=-2"], outcome="close>open"),
        T("try qqq", type="frequency", symbol="QQQ", conditions=["gap<=-2"], outcome="close>open"),
    ]},
]
# PROBE2: written on 2026-11-27, AFTER the sealed set was scored (and its failures read) and after the fixes that followed,
# to measure those fixes on sentences nobody had tuned on. Scored once (docs/v8_evidence/probe2_v8_first_score.json).
# Its sentences avoid the sealed set's wording; they are development data from then on.
PROBE2 = [
    S("buy iwm calls at 10:15 et when it's down 1% from the open, take profit at 40%, stop at 25%", "intermediate", "probe2", type="strategy", symbol="IWM", structure="long_call",
      entry="10:15", conditions=["from_open<=-1"], premium_target=40, premium_stop=25),
    S("puts on spy at the open after a 2% up day, out by 11 am eastern", "intermediate", "probe2", type="strategy", symbol="SPY", structure="long_put", entry=["open", "09:30"],
      conditions=["prev_return>=2"], exit="11:00"),
    S("buy qqq 1dte calls at 6:45am pacific and sell them at 9:30am pacific", "intermediate", "probe2", type="strategy", symbol="QQQ", structure="long_call", dte=1, entry="09:45", exit="12:30"),
    S("spy 0dte calls at the open, sell at noon", "beginner", "probe2", type="strategy", symbol="SPY", structure="long_call", dte=0, entry=["open", "09:30"], exit="12:00"),
    S("calls on qqq every day at 2pm et, hold till the close", "beginner", "probe2", type="strategy", symbol="QQQ", structure="long_call", entry="14:00", exit=["close", "15:55", "15:59", "16:00"]),
    S("sell spy 7dte puts at the open when vix is under 15 the day before", "experienced", "probe2", type="strategy", symbol="SPY", structure="short_put", dte=7, entry=["open", "09:30"], conditions=["vix<15"]),
    S("buy spy puts at 11am et if spy gapped down over 1%, 50% take profit, 30% stop loss", "intermediate", "probe2", type="strategy", symbol="SPY", structure="long_put", entry="11:00",
      conditions=["gap<=-1"], premium_target=50, premium_stop=30),
    S("what if i shorted qqq calls at lunch every day", "beginner", "probe2", clarify_ok=True, type="strategy", symbol="QQQ", structure="short_call", entry="12:00"),
    S("buy iwm puts after it first pops 0.5% off the open and then falls back under the open", "experienced", "probe2", clarify_ok=True, type="strategy", symbol="IWM", structure="long_put",
      sequence=["from_open>=0.5", "from_open<0"]),
    S("how often does iwm end the day higher after gapping up more than 1%", "beginner", "probe2", type="frequency", symbol="IWM", conditions=["gap>=1"], outcome=["close>prev_close", "close>open"]),
    S("odds spy finishes red after 2 green days in a row", "beginner", "probe2", type="frequency", symbol="SPY", conditions=["up_streak>=2"], outcome=["close<open", "close<prev_close"]),
    S("how often does qqq hit -1% before 11am et", "intermediate", "probe2", type="frequency", symbol="QQQ", outcome="touch:-1", end="11:00"),
    S("when qqq is down 0.5% at 10am et how often does it close green", "intermediate", "probe2", [dict(type="frequency", symbol="QQQ", conditions=["first_candle:30:down"], start="10:00",
      outcome=["close>open", "close>prev_close", "close>start"])], clarify_ok=True, type="frequency", symbol="QQQ", conditions=["change<=-0.5"], start="10:00", outcome=["close>open", "close>prev_close", "close>start"]),
    S("what's the probability spy is up 1% within 3 days after a 2% drop", "intermediate", "probe2", type="frequency", symbol="SPY", conditions=["prev_return<=-2"], outcome="touch:+1", end="sessions:3"),
    S("what does qqq usually do after a 3% down day", "beginner", "probe2", type="movement", symbol="QQQ", conditions=["prev_return<=-3"]),
    S("show me how iwm behaves in the first 30 minutes after it gaps down 1%", "intermediate", "probe2", type="movement", symbol="IWM", conditions=["gap<=-1"], end="10:00"),
    S("what time does iwm usually hit its low of the day", "beginner", "probe2", type="timing", symbol="IWM", measure="low_time"),
    S("which month is worst for qqq", "beginner", "probe2", type="breakdown", symbol="QQQ", by="month"),
    S("are fridays good for spy", "beginner", "probe2", type="breakdown", symbol="SPY", by="weekday"),
    S("does the first 30 min candle predict the rest of the day for qqq", "intermediate", "probe2", type="agreement", symbol="QQQ", candle=[30], scope="rest"),
    S("how often does spy dip 0.5% below yesterday's low", "intermediate", "probe2", clarify_ok=True, type="frequency", symbol="SPY"),
    S("how many times did qqq gap down 2% or more since 2022", "beginner", "probe2", type="frequency", symbol="QQQ", outcome="event:gap<=-2", period=["2022-01-01", None]),
    S("will qqq rally next week", "beginner", "probe2", flag=True),
    S("how often does spy go up after a cpi print", "intermediate", "probe2", flag=True),
    S("buy nvda calls at the open", "beginner", "probe2", flag=True),
    S("how does spy trade when the 10 year yield spikes", "intermediate", "probe2", flag=True),
    S("hw often dos qqq close abve its open on mondays", "beginner", "probe2", type="frequency", symbol="QQQ", conditions=["weekday=Mon"], outcome="close>open"),
    S("iwm ripping 1.5% off the open how often", "beginner", "probe2", clarify_ok=True, type="frequency", symbol="IWM", outcome=["touch:+1.5"]),
]
PROBE2_CONVERSATIONS = [
    {"id": "probe2-strat", "turns": [
        T("buy spy 0dte calls at the open after a 1% down day, sell at 3pm et", type="strategy", symbol="SPY", structure="long_call", dte=0, entry=["open", "09:30"], conditions=["prev_return<=-1"], exit="15:00"),
        T("make it puts", type="strategy", symbol="SPY", structure="long_put", conditions=["prev_return<=-1"], exit="15:00"),
        T("exit an hour earlier", type="strategy", symbol="SPY", structure="long_put", exit="14:00"),
        T("compare both", type="compare"),
    ]},
    {"id": "probe2-question", "turns": [
        T("how often does iwm close above its open after a 1% gap up?", type="frequency", symbol="IWM", conditions=["gap>=1"], outcome="close>open"),
        T("what about qqq", type="frequency", symbol="QQQ", conditions=["gap>=1"], outcome="close>open"),
        T("and only since 2023", type="frequency", symbol="QQQ", conditions=["gap>=1"], outcome="close>open", period=["2023-01-01", None]),
        T("undo", type="frequency", symbol="QQQ", conditions=["gap>=1"], outcome="close>open", period=[None, None]),
    ]},
]
DEV_CONVERSATIONS = [
    {"id": "brief", "turns": [
        T("How often is SPY's first candle right about the rest of the day?", type="agreement", symbol="SPY", scope="rest"),
        T("Use 5 min.", type="agreement", symbol="SPY", candle=[5], scope="rest"),
        T("Actually do QQQ.", type="agreement", symbol="QQQ", candle=[5], scope="rest"),
        T("What about when yesterday was red?", type="agreement", symbol="QQQ", candle=[5], conditions=["prev_return<0"]),
        T("Nah, I meant down at least 1%.", type="agreement", symbol="QQQ", candle=[5], conditions=["prev_return<=-1"]),
        T("Show me the losing days.", type="explore"),
        T("What if I bought calls instead?", type="strategy", symbol="QQQ", structure="long_call", conditions=["first_candle:5:up", "prev_return<=-1"]),
        T("Exit an hour later.", clarify_ok=True, type="strategy", symbol="QQQ", structure="long_call"),
        T("Compare both.", type="compare"),
    ]},
    {"id": "typed", "turns": [
        T("How often is SPY's overall day, predicted by the first 9:30 open candle?", type="agreement", symbol="SPY"),
        T("no I meant 6:30a, PST", type="agreement", symbol="SPY"),
    ]},
]
