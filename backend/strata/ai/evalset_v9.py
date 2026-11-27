"""V9 evaluation of understanding: sentences, conversations and numerical answers, written on 2026-11-27 BEFORE any V9 change
to the readers, the model reader, the conversation layer or position sizing.

THE SEALED PART (SEALED, SEALED_CONVERSATIONS, SEALED_ANSWERS) MUST NOT BE EDITED, run while developing, or used to tune
anything. It is held by hash (`sealed_hash`, checked by a test) and scored once, at the end of V9. A development set
(DEVELOPMENT, DEV_CONVERSATIONS: the examples of the V9 brief) is what the work is built against.

Who wrote it: the same author as the code (there was no independent author available). To keep it as independent of the
implementation as that allows, it was written before the V9 code existed, in the phrasing of ordinary traders (slang,
typos, missing time zones, several conditions in one breath), and every item accepts each reading a knowledgeable person
would accept. That is disclosed with every score.

The vocabulary is the V8 one (see evalset_v8: type, symbol, conditions, outcome, start, end, candle, scope, predictor, by,
measure, structure, dte, strike, quantity, entry, exit, premium_target, premium_stop, underlying_target, underlying_stop,
hold, period, clarify, clarify_ok, flag, alternatives), with these additions:

  prev_otc        the previous session's open-to-close move, for "red day" / "green day" read as the candle's colour:
                  "prev_otc<0" (prev_return is the close-to-close move)
  touch:prev_high the price trades at or above the previous session's high (like touch:prev_low)
  sizing          how much each trade buys: "contracts:N" | "dollars:N" (premium spent per trade) | "risk_dollars:N"
                  (dollars lost if the stop is hit, or the whole premium without a stop) | "risk_pct:N" (the same, as a
                  share of the account) | "equity_pct:N" (premium spent as a share of the account)
  capital         the starting account in dollars;  reinvest  profits change the size of later trades (true/false)
  daily_loss_stop no new entries that session once the day's realised loss reaches this many dollars
  entries         entries allowed per session
  type            adds: explore (a list of the observations behind a result), selection (a question about the listed
                  or selected observations), next_steps (what to investigate next), reliability, explain, compare
  offers_alternative   with flag: the reply says why it cannot be answered AND offers the closest question that can be,
                  as something to run (without running a changed question by itself)
  supported       False for a request the application is not expected to answer (flag items); the 90% target of the
                  brief is measured on the supported items whose correct response is a run

Times: New York. A time with no zone is read as a knowledgeable person would: where only one zone puts it inside the
session (7am can only be Pacific), that zone; where two do, either reading is accepted, and so is a question.
"""
from __future__ import annotations

import hashlib
import json


def S(text: str, level: str, category: str, alternatives: list[dict] | None = None, supported: bool = True, **expect) -> dict:
    return {"text": text, "level": level, "category": category, "expect": expect, "alternatives": alternatives or [], "supported": supported and not expect.get("flag")}


def T(text: str, alternatives: list[dict] | None = None, **expect) -> dict:
    return {"text": text, "expect": expect, "alternatives": alternatives or []}


GREEN = ["close>open", "close>prev_close"]
RED = ["close<open", "close<prev_close"]

# =============================================================================================================== SEALED
SEALED = [
    # ---- how often (12)
    S("how many times since 2020 has QQQ opened down more than 1% and still closed green?", "intermediate", "frequency",
      type="frequency", symbol="QQQ", conditions=["gap<=-1"], outcome=GREEN, period=["2020-01-01", None]),
    S("When IWM gaps up half a percent or more, how often does the gap get filled the same day?", "intermediate", "frequency",
      type="frequency", symbol="IWM", conditions=["gap>=0.5"], outcome="touch:prev_close", end="close"),
    S("after 3 red days in a row how often is spy green the next day", "beginner", "informal",
      type="frequency", symbol="SPY", conditions=["down_streak>=3"], outcome=GREEN),
    S("how often does qqq trade 1% below its open at some point in the session", "beginner", "frequency",
      type="frequency", symbol="QQQ", conditions=[], outcome="touch:-1", start="open", end="close"),
    S("is it common for SPY to drop 2% from the open and then finish the day green?", "intermediate", "multi_part",
      type="frequency", symbol="SPY", conditions=["from_open<=-2"], outcome=GREEN),
    S("what % of the time does IWM close higher than it opened when VIX was above 25 the day before", "intermediate", "frequency",
      type="frequency", symbol="IWM", conditions=["vix>25"], outcome="close>open"),
    S("how often does SPY hit +0.3% between 6:30 and 7:30 pacific", "intermediate", "timezone",
      type="frequency", symbol="SPY", conditions=[], outcome="touch:+0.3", start="open", end="10:30"),
    S("chance qqq takes out yesterday's high in the first hour?", "intermediate", "informal",
      type="frequency", symbol="QQQ", conditions=[], outcome="touch:prev_high", start="open", end="10:30"),
    S("after a 1%+ gap down on SPY, how often is it still below the open at noon ET?", "intermediate", "frequency",
      type="frequency", symbol="SPY", conditions=["gap<=-1"], outcome="close<open", end="12:00"),
    S("how often is iwm lower 2 days after a 2% drop", "expert", "ambiguous", clarify_ok=True,
      type="frequency", symbol="IWM", conditions=["prev_return<=-2"], outcome=["close<prev_close", "close<start"], end="sessions:2"),
    S("what fraction of mondays does spy open above friday's close", "intermediate", "frequency",
      type="frequency", symbol="SPY", conditions=["weekday=Mon"], outcome="event:gap>0"),
    S("how often does qqq close in the green after the first 5 minute candle is red", "beginner", "frequency",
      type="frequency", symbol="QQQ", conditions=["first_candle:5:down"], outcome=["close>open", "close>prev_close", "close>start"]),

    # ---- does one thing predict another (5)
    S("does the first hour's direction predict the close for qqq?", "intermediate", "agreement",
      type="agreement", symbol="QQQ", candle=[60], scope=["day", "both"]),
    S("Is a green opening 15-min candle on IWM a good sign for the rest of the session?", "intermediate", "agreement",
      alternatives=[{"type": "frequency", "symbol": "IWM", "conditions": ["first_candle:15:up"], "outcome": "close>start"}],
      type="agreement", symbol="IWM", candle=[15], scope="rest"),
    S("when yesterday was up, is today usually up too? spy", "beginner", "agreement",
      alternatives=[{"type": "frequency", "symbol": "SPY", "conditions": ["prev_return>0"], "outcome": GREEN}],
      type="agreement", symbol="SPY", predictor="prev_direction"),
    S("how reliable is the 30 minute opening range direction at calling the day on SPY", "expert", "agreement",
      type="agreement", symbol="SPY", candle=[30], scope=["day", "both"]),
    S("how often does the first 15 minute candle on qqq point the same way as the close?", "intermediate", "agreement",
      type="agreement", symbol="QQQ", candle=[15], scope=["day", "both"]),

    # ---- what usually happens (5)
    S("what does QQQ typically do in the first hour after a 1.5% gap down", "intermediate", "movement",
      type="movement", symbol="QQQ", conditions=["gap<=-1.5"], start="open", end="10:30"),
    S("show me how spy trades the day after a 3% drop", "beginner", "movement",
      type="movement", symbol="SPY", conditions=["prev_return<=-3"], start="open", end="close"),
    S("average move from 10am to the close on fridays for IWM", "intermediate", "timezone", clarify_ok=True,
      alternatives=[{"type": "movement", "symbol": "IWM", "conditions": ["weekday=Fri"], "start": "13:00", "end": "close"}],
      type="movement", symbol="IWM", conditions=["weekday=Fri"], start="10:00", end="close"),
    S("how far does iwm usually run after gapping up 1%", "beginner", "movement",
      type="movement", symbol="IWM", conditions=["gap>=1"]),
    S("typical spy path on days it opens more than 1% lower", "intermediate", "movement",
      type="movement", symbol="SPY", conditions=["gap<=-1"]),

    # ---- when during the day (4)
    S("what time of day does qqq usually bottom out", "beginner", "timing",
      type="timing", symbol="QQQ", measure="low_time"),
    S("which half hour has the biggest moves in SPY on mondays", "intermediate", "timing",
      type="timing", symbol="SPY", measure="abs_move", bucket=30, conditions=["weekday=Mon"]),
    S("when does IWM usually put in its high on gap up days", "intermediate", "timing",
      type="timing", symbol="IWM", measure="high_time", conditions=["gap>0"]),
    S("is the last hour more volatile than the first for qqq", "intermediate", "timing",
      type="timing", symbol="QQQ", measure="abs_move", bucket=60),

    # ---- broken down (4)
    S("are fridays worse than other days for qqq?", "beginner", "breakdown",
      type="breakdown", symbol="QQQ", by="weekday", focus="Fri"),
    S("IWM returns by month since 2018", "intermediate", "breakdown",
      type="breakdown", symbol="IWM", by="month", period=["2018-01-01", None]),
    S("best day of the week to be long spy at the open and sell at the close", "intermediate", "breakdown",
      type="breakdown", symbol="SPY", by="weekday", measure="open_to_close"),
    S("do tuesdays tend to be green for qqq", "beginner", "breakdown",
      alternatives=[{"type": "frequency", "symbol": "QQQ", "conditions": ["weekday=Tue"], "outcome": GREEN}],
      type="breakdown", symbol="QQQ", by="weekday", focus="Tue"),

    # ---- strategies (10)
    S("buy 1 SPY 0dte ATM call at 9:45 ET every day, take profit at 50%, stop at 25%, flat by 3:30pm ET", "intermediate", "strategy",
      type="strategy", symbol="SPY", structure="long_call", dte=0, strike="atm", quantity=1, entry="09:45", exit="15:30", premium_target=50, premium_stop=25),
    S("backtest selling qqq 0dte puts at 10am pacific when it's red on the day", "expert", "timezone",
      alternatives=[{"type": "strategy", "symbol": "QQQ", "structure": "short_put", "dte": 0, "entry": "13:00", "conditions": ["from_open<0"]}],
      type="strategy", symbol="QQQ", structure="short_put", dte=0, entry="13:00", conditions=["change<0"]),
    S("IWM 7dte puts 1% otm after any 2% up day, sell 2 days later", "intermediate", "strategy",
      type="strategy", symbol="IWM", structure="long_put", dte=7, strike="otm:1", conditions=["prev_return>=2"], hold="sessions:2"),
    S("what if I bought QQQ calls at the open only on gap ups bigger than half a percent and sold at noon ET", "beginner", "strategy",
      type="strategy", symbol="QQQ", structure="long_call", entry="open", conditions=["gap>=0.5"], exit="12:00"),
    S("buy spy puts at 10:30 ET if it gapped up but is now below vwap, 1dte, cut it at a 30% loss", "expert", "multi_part",
      type="strategy", symbol="SPY", structure="long_put", dte=1, entry="10:30", conditions=["gap>0", "below_vwap"], premium_stop=30),
    S("buy spy 0dte calls at the open and sell when SPY is up half a percent", "intermediate", "underlying_vs_premium",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry="open", underlying_target=0.5, premium_target=None),
    S("buy qqq calls at 10 ET and sell when they're up 40%", "intermediate", "underlying_vs_premium",
      type="strategy", symbol="QQQ", structure="long_call", entry="10:00", premium_target=40, underlying_target=None),
    S("buy IWM 0DTE puts at the open, sell when it's up 20% or at 11 ET", "intermediate", "underlying_vs_premium", clarify_ok=True,
      type="strategy", symbol="IWM", structure="long_put", dte=0, entry="open", premium_target=20, exit="11:00"),
    S("sell a qqq 0dte iron condor at 10am ET and hold it to the close", "expert", "strategy",
      alternatives=[{"type": "strategy", "symbol": "QQQ", "structure": "iron_condor", "dte": 0, "entry": "10:00", "hold": "to_expiration"}],
      type="strategy", symbol="QQQ", structure="iron_condor", dte=0, entry="10:00", exit=["close", "15:55", "15:59", "16:00"]),
    S("straddle on SPY at 9:31 ET 0dte, close it after 45 min", "intermediate", "strategy",
      type="strategy", symbol="SPY", structure="long_straddle", dte=0, entry="09:31", hold="minutes:45"),

    # ---- position sizing (6)
    S("buy SPY 0DTE calls at the open after a down day, spend $500 a trade", "beginner", "sizing",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry="open", conditions=["prev_return<0"], sizing="dollars:500"),
    S("QQQ 1DTE calls at 10am ET with a 40% stop, risking 2% of a $25k account each time", "intermediate", "sizing",
      type="strategy", symbol="QQQ", structure="long_call", dte=1, entry="10:00", premium_stop=40, sizing="risk_pct:2", capital=25000),
    S("start with 10k and put 5% of the account into SPY 0DTE puts every time it gaps down 1% or more, reinvesting profits", "intermediate", "sizing",
      type="strategy", symbol="SPY", structure="long_put", dte=0, conditions=["gap<=-1"], capital=10000, sizing="equity_pct:5", reinvest=True),
    S("buy one SPY 0DTE call each time it crosses above VWAP, max 3 a day, and stop trading for the day after losing $200", "expert", "sizing",
      type="strategy", symbol="SPY", structure="long_call", dte=0, quantity=1, conditions=["crosses_above:vwap"], entries=3, daily_loss_stop=200),
    S("how much would I have made risking $300 per trade buying IWM puts at the open after a 1% up day with a 50% stop", "intermediate", "sizing",
      type="strategy", symbol="IWM", structure="long_put", entry="open", conditions=["prev_return>=1"], premium_stop=50, sizing="risk_dollars:300"),
    S("buy 3 contracts of QQQ ATM 0DTE calls at 9:40 ET", "beginner", "sizing",
      type="strategy", symbol="QQQ", structure="long_call", dte=0, strike="atm", entry="09:40", quantity=3),

    # ---- the right response is a question (5)
    S("does spy bounce?", "beginner", "ambiguous", clarify=True),
    S("what's the best time to buy calls", "beginner", "ambiguous", clarify=True),
    S("how often does it go up 1%", "beginner", "ambiguous", clarify_ok=True,
      type="frequency", symbol="SPY", outcome="touch:+1"),
    S("what happens when the market tanks", "beginner", "ambiguous", clarify_ok=True,
      alternatives=[{"type": "movement", "symbol": "SPY", "conditions": ["from_open<=*"]}, {"type": "movement", "symbol": "SPY", "conditions": ["gap<=*"]}],
      type="movement", symbol="SPY", conditions=["prev_return<=*"]),
    S("buy calls when it dips", "beginner", "ambiguous", clarify=True),

    # ---- not answerable: say why and offer the closest question that is (7)
    S("how does SPY do the day after FOMC?", "intermediate", "unsupported", flag=True, offers_alternative=True),
    S("how often does TSLA gap up 2%?", "beginner", "unsupported", flag=True, offers_alternative=True),
    S("will qqq go up tomorrow", "beginner", "unsupported", flag=True, offers_alternative=True),
    S("how often does RSI under 30 lead to a bounce on QQQ", "intermediate", "unsupported", flag=True, offers_alternative=True),
    S("backtest SPY iron condors since 2015", "expert", "data_coverage", flag=True, offers_alternative=True),
    S("would buying spy calls every morning have worked in 2022?", "beginner", "data_coverage", flag=True, offers_alternative=True),
    S("what's SPY's average return during earnings season weeks", "intermediate", "unsupported", flag=True, offers_alternative=True),

    # ---- typed in a hurry (4)
    S("hw often dos qqq close red aftr a gren day lol", "beginner", "informal",
      alternatives=[{"type": "frequency", "symbol": "QQQ", "conditions": ["prev_otc>0"], "outcome": RED}],
      type="frequency", symbol="QQQ", conditions=["prev_return>0"], outcome=RED),
    S("spy 0dte calls @ open, tp 30 sl 15, out by 1245 pst", "intermediate", "informal",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry="open", premium_target=30, premium_stop=15, exit="15:45"),
    S("qqq gap n go - how often does a 1% gap up keep going til close", "intermediate", "informal",
      type="frequency", symbol="QQQ", conditions=["gap>=1"], outcome="close>open"),
    S("iwm after 2 down days in a row, odds of a green day?", "beginner", "informal",
      type="frequency", symbol="IWM", conditions=["down_streak>=2"], outcome=GREEN),
]

SEALED_CONVERSATIONS = [
    {"id": "v9c1", "level": "beginner", "turns": [
        T("how often does SPY close green after a red day",
          alternatives=[{"type": "frequency", "symbol": "SPY", "conditions": ["prev_otc<0"], "outcome": GREEN}],
          type="frequency", symbol="SPY", conditions=["prev_return<0"], outcome=GREEN),
        T("what about QQQ", type="frequency", symbol="QQQ"),
        T("only when the drop was at least 1.5%", type="frequency", symbol="QQQ", conditions=["prev_return<=-1.5"]),
        T("show me the days it didn't work", type="explore"),
        T("undo that last change",
          alternatives=[{"type": "frequency", "symbol": "QQQ", "conditions": ["prev_otc<0"]}],
          type="frequency", symbol="QQQ", conditions=["prev_return<0"]),
    ]},
    {"id": "v9c2", "level": "intermediate", "turns": [
        T("buy QQQ 0DTE calls at 10am ET after a gap up", type="strategy", symbol="QQQ", structure="long_call", dte=0, entry="10:00", conditions=["gap>0"]),
        T("make it 1DTE", type="strategy", symbol="QQQ", structure="long_call", dte=1, entry="10:00", conditions=["gap>0"]),
        T("risk $300 per trade", type="strategy", symbol="QQQ", dte=1, sizing="risk_dollars:300"),
        T("start with a $5,000 account", type="strategy", symbol="QQQ", dte=1, sizing="risk_dollars:300", capital=5000),
        T("why did the worst trade lose?", type="explain"),
        T("compare it with the version before I changed the sizing", type="compare"),
    ]},
    {"id": "v9c3", "level": "beginner", "turns": [
        T("when during the day does IWM move the most?", type="timing", symbol="IWM", measure="abs_move"),
        T("just mondays", type="timing", symbol="IWM", conditions=["weekday=Mon"]),
        T("same for qqq", type="timing", symbol="QQQ", conditions=["weekday=Mon"]),
        T("what would you look at next?", type="next_steps"),
    ]},
    {"id": "v9c4", "level": "intermediate", "turns": [
        T("does the first 30 minutes predict the rest of the day for SPY", type="agreement", symbol="SPY", candle=[30], scope="rest"),
        T("show me the days it was wrong", type="explore"),
        T("anything those days have in common?", type="selection"),
        T("try it with 15 minutes", type="agreement", symbol="SPY", candle=[15], scope="rest"),
    ]},
    {"id": "v9c5", "level": "intermediate", "turns": [
        T("buy spy puts at 10 when it's down 1% from the open", clarify_ok=True,
          alternatives=[{"type": "strategy", "symbol": "SPY", "structure": "long_put", "entry": "13:00", "conditions": ["from_open<=-1"]}],
          type="strategy", symbol="SPY", structure="long_put", entry="10:00", conditions=["from_open<=-1"]),
        T("nah 0.5%", type="strategy", symbol="SPY", structure="long_put", conditions=["from_open<=-0.5"]),
        T("exit at 12", clarify_ok=True, alternatives=[{"type": "strategy", "symbol": "SPY", "exit": "15:00"}],
          type="strategy", symbol="SPY", exit="12:00"),
        T("actually 12 pacific", type="strategy", symbol="SPY", structure="long_put", conditions=["from_open<=-0.5"], exit="15:00"),
        T("use 0dte", type="strategy", symbol="SPY", structure="long_put", dte=0, exit="15:00"),
    ]},
    {"id": "v9c6", "level": "intermediate", "turns": [
        T("how often does qqq fill a gap down", type="frequency", symbol="QQQ", conditions=["gap<0"], outcome="touch:prev_close"),
        T("what about on CPI days?", flag=True, offers_alternative=True),
        T("ok then just on mondays", type="frequency", symbol="QQQ", conditions=["gap<0", "weekday=Mon"], outcome="touch:prev_close"),
    ]},
    {"id": "v9c7", "level": "intermediate", "turns": [
        T("does a strong first hour mean a strong day for qqq?", clarify_ok=True,
          alternatives=[{"type": "frequency", "symbol": "QQQ", "conditions": ["first_candle:60:up"]}],
          type="agreement", symbol="QQQ", candle=[60], scope=["day", "both", "rest"]),
        T("what if yesterday was red too",
          alternatives=[{"type": "agreement", "symbol": "QQQ", "conditions": ["prev_otc<0"]}, {"type": "frequency", "symbol": "QQQ", "conditions": ["first_candle:60:up", "prev_return<0"]},
                        {"type": "frequency", "symbol": "QQQ", "conditions": ["first_candle:60:up", "prev_otc<0"]}],
          type="agreement", symbol="QQQ", conditions=["prev_return<0"]),
        T("show me the worst days", type="explore"),
        T("is this just noise?", type="reliability"),
    ]},
]

# Numbers: run on the stored history; each reply must contain the independently calculated value (strata.research.reference,
# written separately from the engines that produce the answer) within the tolerance, and none of the forbidden statements.
SEALED_ANSWERS = [
    {"id": "v9a1", "text": "How often has SPY closed above its open since 2016?", "reference": "close_above_open", "args": {"symbol": "SPY", "start": "2016-01-01"}, "numbers": ["pct"]},
    {"id": "v9a2", "text": "After QQQ gaps down 1% or more, how often does it fill the gap the same day?", "reference": "gap_fill", "args": {"symbol": "QQQ", "gap": -1.0}, "numbers": ["pct", "k"]},
    {"id": "v9a3", "text": "How often does IWM close below its open the day after a 2% up day?", "reference": "after_prev_return", "args": {"symbol": "IWM", "prev": 2.0, "outcome": "close<open"}, "numbers": ["pct", "k"]},
    {"id": "v9a4", "text": "Does the first 15-minute candle predict the rest of SPY's session?", "reference": "candle_agreement", "args": {"symbol": "SPY", "minutes": 15}, "numbers": ["rest_pct", "day_pct"]},
    {"id": "v9a5", "text": "Which weekday has the best average open-to-close return for QQQ?", "reference": "best_weekday", "args": {"symbol": "QQQ"}, "names": ["weekday"], "numbers": ["mean"]},
    {"id": "v9a6", "text": "On days SPY gaps up 0.5% or more, what is the average move from 10:00 ET to the close?", "reference": "window_mean_after_gap", "args": {"symbol": "SPY", "gap": 0.5, "start": "10:00"}, "numbers": ["mean"]},
    {"id": "v9a7", "text": "In 2025, how many times did IWM gap down more than 1%?", "reference": "gap_count", "args": {"symbol": "IWM", "gap": -1.0, "year": 2025}, "numbers": ["k"]},
    {"id": "v9a8", "text": "When during the day does QQQ usually make its low?", "reference": "extreme_bucket", "args": {"symbol": "QQQ", "which": "low"}, "names": ["bucket"]},
    {"id": "v9a9", "text": "How often does SPY touch +0.5% from the open before 11am ET?", "reference": "touch_before", "args": {"symbol": "SPY", "pct": 0.5, "until": "11:00"}, "numbers": ["pct"]},
    {"id": "v9a10", "text": "How often does QQQ close below the previous close the day after 3 up days in a row?", "reference": "after_streak", "args": {"symbol": "QQQ", "n": 3, "direction": "up", "outcome": "close<prev_close"}, "numbers": ["pct", "k"]},
    {"id": "v9a11", "text": "Buy 1 ATM SPY 0DTE call at 10:00 ET after a 1% down day and sell at 15:30 ET, using option trade prices",
     "reference": "option_backtest", "args": {"symbol": "SPY", "right": "call", "entry": "10:00", "exit": "15:30", "prev": -1.0, "contracts": 1}, "numbers": ["trades", "pnl"]},
    {"id": "v9a12", "text": "Buy ATM QQQ 0DTE calls at 10:00 ET after a 1% down day, sell at 15:30 ET, spending $500 per trade, using option trade prices",
     "reference": "option_backtest", "args": {"symbol": "QQQ", "right": "call", "entry": "10:00", "exit": "15:30", "prev": -1.0, "dollars": 500}, "numbers": ["trades", "pnl"]},
]
FORBIDDEN = ["probability that the strategy", "probability that this strategy", "chance that this is luck", "probability that the result is due to chance", "proven profitable",
             "guaranteed", "proves that", "will make money", "risk-free"]


def sealed_payload() -> str:
    return json.dumps({"sealed": SEALED, "conversations": SEALED_CONVERSATIONS, "answers": SEALED_ANSWERS, "forbidden": FORBIDDEN}, sort_keys=True)


def sealed_hash() -> str:
    """The hash of the sealed items, recorded in docs/v9_evidence/sealed_v9.txt when the set was written; a test holds it."""
    return hashlib.sha256(sealed_payload().encode()).hexdigest()


# ========================================================================================================== DEVELOPMENT
# The examples of the V9 brief, and sentences written while building. Anything here may be used and changed.
DEVELOPMENT = [
    S("How often does SPY fake out its opening move?", "intermediate", "brief", clarify_ok=True,
      alternatives=[{"type": "frequency", "symbol": "SPY", "outcome": ["close>start", "close<start"]}],
      type="agreement", symbol="SPY", scope="rest"),
    S("Does a strong first hour usually mean a strong day?", "beginner", "brief", clarify_ok=True,
      alternatives=[{"type": "frequency", "symbol": "SPY", "conditions": ["first_candle:60:up"]}],
      type="agreement", symbol="SPY", candle=[60], scope=["day", "both", "rest"]),
    S("how often does IWM close below the open after the first 30 minutes are green", "beginner", "dev",
      type="frequency", symbol="IWM", conditions=["first_candle:30:up"], outcome=["close<open", "close<start"]),
    S("buy qqq 0dte puts at 11am ET when it's up 0.5% from the open, 2 contracts, 25% stop", "intermediate", "dev",
      type="strategy", symbol="QQQ", structure="long_put", dte=0, entry="11:00", conditions=["from_open>=0.5"], quantity=2, premium_stop=25),
    S("start with $20k and risk 1% per trade buying SPY 0DTE calls at 10:30 ET after a gap down", "intermediate", "dev",
      type="strategy", symbol="SPY", structure="long_call", dte=0, entry="10:30", conditions=["gap<0"], sizing="risk_pct:1", capital=20000),
    S("how often does qqq hit -1% from the open before 11:30 ET on mondays", "intermediate", "dev",
      type="frequency", symbol="QQQ", conditions=["weekday=Mon"], outcome="touch:-1", end="11:30"),
    S("what's the chance SPY rains on its parade tomorrow", "beginner", "dev", flag=True, offers_alternative=True),
]
DEV_CONVERSATIONS = [
    {"id": "d9c1", "level": "intermediate", "turns": [
        T("How often does SPY close green after a red opening 15 minute candle?", type="frequency", symbol="SPY", conditions=["first_candle:15:down"], outcome=["close>open", "close>start", "close>prev_close"]),
        T("What happens if yesterday was red too?", alternatives=[{"type": "frequency", "symbol": "SPY", "conditions": ["first_candle:15:down", "prev_otc<0"]}],
          type="frequency", symbol="SPY", conditions=["first_candle:15:down", "prev_return<0"]),
        T("Show me where this went wrong.", type="explore"),
    ]},
    {"id": "d9c2", "level": "intermediate", "turns": [
        T("Buy SPY 0DTE calls at 6:45 AM Pacific after a 1% down day, exit at 12:45 PM Pacific", type="strategy", symbol="SPY", structure="long_call", dte=0, entry="09:45", exit="15:45", conditions=["prev_return<=-1"]),
        T("What if I waited until 7 AM?", type="strategy", symbol="SPY", entry="10:00", exit="15:45"),
        T("How much would I have made risking $500 each time?", type="strategy", symbol="SPY", entry="10:00", sizing="risk_dollars:500"),
        T("Is there anything interesting about the losing trades?", type="selection"),
        T("What would you investigate next?", type="next_steps"),
    ]},
]

# Written during V9 development (after the first model-reader measurement), in phrasings neither reader was built on; used to
# choose the reading policy and to fix the readers. Development data: never a held-out estimate.
PROBE = [
    S("on fridays, how often does qqq close above where it opened", "beginner", "probe", type="frequency", symbol="QQQ", conditions=["weekday=Fri"], outcome="close>open"),
    S("chance IWM is above its open at 11am ET after gapping down 0.75%", "intermediate", "probe", type="frequency", symbol="IWM", conditions=["gap<=-0.75"], outcome="close>open", end="11:00"),
    S("what does spy usually do after 4 straight up days", "beginner", "probe", type="movement", symbol="SPY", conditions=["up_streak>=4"]),
    S("is wednesday the best day for iwm", "beginner", "probe", type="breakdown", symbol="IWM", by="weekday", focus="Wed"),
    S("when is qqq calmest during the day", "beginner", "probe", type="timing", symbol="QQQ", measure="abs_move"),
    S("how often does spy's first 5 min bar match the color of the daily candle", "intermediate", "probe", type="agreement", symbol="SPY", candle=[5], scope=["day", "both"]),
    S("if qqq gaps up 2% does it usually fade by the close", "intermediate", "probe", clarify_ok=True, type="frequency", symbol="QQQ", conditions=["gap>=2"], outcome="close<open"),
    S("buy 2 iwm 1dte calls at 10:15 ET on mondays, sell at 3pm ET", "intermediate", "probe",
      type="strategy", symbol="IWM", structure="long_call", dte=1, quantity=2, entry="10:15", exit="15:00", conditions=["weekday=Mon"]),
    S("short spy 0dte puts at noon eastern if spy is green on the day, 100% stop", "expert", "probe",
      type="strategy", symbol="SPY", structure="short_put", dte=0, entry="12:00", conditions=["change>0"], premium_stop=100),
    S("spend $1000 per trade on qqq 0dte puts at the open after a 1.5% up day", "intermediate", "probe",
      type="strategy", symbol="QQQ", structure="long_put", dte=0, entry="open", conditions=["prev_return>=1.5"], sizing="dollars:1000"),
    S("risk 3% of a 50k account on spy calls at 9:45 ET with a 50% stop", "intermediate", "probe",
      type="strategy", symbol="SPY", structure="long_call", entry="09:45", premium_stop=50, sizing="risk_pct:3", capital=50000),
    S("how often does iwm trade above yesterday's high before 10:30 ET", "intermediate", "probe", type="frequency", symbol="IWM", outcome="touch:prev_high", end="10:30"),
    S("median move from 2pm ET to the close for spy", "intermediate", "probe", type="movement", symbol="SPY", start="14:00", end="close"),
    S("what % of qqq gap downs over 1% get filled", "intermediate", "probe", type="frequency", symbol="QQQ", conditions=["gap<=-1"], outcome="touch:prev_close"),
    S("does iwm close higher the day after vix closes above 30", "intermediate", "probe", type="frequency", symbol="IWM", conditions=["vix>30"], outcome=["close>prev_close", "close>open"]),
    S("will spy hit 600 next month", "beginner", "probe", flag=True, offers_alternative=True),
    S("spy calls when RSI is oversold", "intermediate", "probe", flag=True, offers_alternative=True),
    S("how many times did qqq drop 3% in a day in 2022", "intermediate", "probe", type="frequency", symbol="QQQ", outcome="event:day_return<=-3", period=["2022-01-01", "2022-12-31"],
      alternatives=[{"type": "frequency", "symbol": "QQQ", "conditions": ["prev_return<=-3"], "period": ["2022-01-01", "2022-12-31"]}]),
    S("probability spy finishes above its open when the first 15 min are red and it gapped up", "expert", "probe",
      type="frequency", symbol="SPY", conditions=["first_candle:15:down", "gap>0"], outcome="close>open"),
    S("after spy is down 3 days straight how often does it bounce the next day", "beginner", "probe", clarify_ok=True,
      type="frequency", symbol="SPY", conditions=["down_streak>=3"], outcome=["close>prev_close", "close>open"]),
    S("qqq 7dte calls 2% otm at 10 ET when vix is below 20, take profit 100%", "expert", "probe",
      type="strategy", symbol="QQQ", structure="long_call", dte=7, strike="otm:2", entry="10:00", conditions=["vix<20"], premium_target=100),
    S("how often does qqq close red after opening up 0.5% or more", "beginner", "probe", type="frequency", symbol="QQQ", conditions=["gap>=0.5"], outcome=["close<open", "close<prev_close"]),
    S("what usually happens to iwm in the last hour on days it's down 1% at 3pm ET", "expert", "probe",
      type="movement", symbol="IWM", conditions=["change<=-1"], start="15:00", end="close",
      alternatives=[{"type": "movement", "symbol": "IWM", "conditions": ["from_open<=-1"], "start": "15:00", "end": "close"}]),
    S("buy tsla puts at the open", "beginner", "probe", flag=True, offers_alternative=True),
    S("which month is historically worst for qqq", "beginner", "probe", type="breakdown", symbol="QQQ", by="month"),
]
PROBE_CONVERSATIONS = [
    {"id": "p9c1", "level": "beginner", "turns": [
        T("how often does qqq close higher after a 1% gap down", type="frequency", symbol="QQQ", conditions=["gap<=-1"], outcome=["close>open", "close>prev_close"]),
        T("make it 2%", type="frequency", symbol="QQQ", conditions=["gap<=-2"]),
        T("on iwm", type="frequency", symbol="IWM", conditions=["gap<=-2"]),
        T("since 2021 only", type="frequency", symbol="IWM", period=["2021-01-01", None]),
        T("undo", type="frequency", symbol="IWM", period=[None, None]),
    ]},
    {"id": "p9c2", "level": "intermediate", "turns": [
        T("buy spy 0dte puts at 10:30 ET after a gap up, out at 2pm ET", type="strategy", symbol="SPY", structure="long_put", dte=0, entry="10:30", exit="14:00", conditions=["gap>0"]),
        T("use calls", type="strategy", symbol="SPY", structure="long_call", entry="10:30", exit="14:00"),
        T("spend $300 per trade", type="strategy", symbol="SPY", structure="long_call", sizing="dollars:300"),
        T("why did the worst trade lose", type="explain"),
    ]},
    {"id": "p9c3", "level": "beginner", "turns": [
        T("when does spy usually make its high", type="timing", symbol="SPY", measure="high_time"),
        T("on gap down days", type="timing", symbol="SPY", conditions=["gap<0"]),
        T("what would you check next", type="next_steps"),
    ]},
]

# development answers: to check the numerical-answer harness before the sealed answers are run once
DEV_ANSWERS = [
    {"id": "d9a1", "text": "How often has QQQ closed above its open since 2018?", "reference": "close_above_open", "args": {"symbol": "QQQ", "start": "2018-01-01"}, "numbers": ["pct"]},
    {"id": "d9a2", "text": "After SPY gaps down 0.5% or more, how often does it fill the gap the same day?", "reference": "gap_fill", "args": {"symbol": "SPY", "gap": -0.5}, "numbers": ["pct", "k"]},
    {"id": "d9a3", "text": "Does the first 30-minute candle predict the rest of QQQ's session?", "reference": "candle_agreement", "args": {"symbol": "QQQ", "minutes": 30}, "numbers": ["rest_pct", "day_pct"]},
    {"id": "d9a4", "text": "Buy 1 ATM IWM 0DTE put at 10:15 ET after a 1% up day and sell at 15:00 ET, using option trade prices",
     "reference": "option_backtest", "args": {"symbol": "IWM", "right": "put", "entry": "10:15", "exit": "15:00", "prev": 1.0, "contracts": 1}, "numbers": ["trades", "pnl"]},
]
