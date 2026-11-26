"""V7: sentences about intraday strategies and one-sentence experiments, with the meaning each one has.

Two sets, kept apart on purpose.

DEVELOPMENT  the sentences the V7 readers were built and corrected against (the examples of the V7 brief and variants of
             them). Their score says what was fixed.
UNSEEN       written on 2026-11-26 BEFORE any V7 reader existed, in other words and with other numbers, and not looked at
             again while the readers were built. Each reader (the built-in rules; each local or hosted language model) is
             scored on it ONCE, and that first score is recorded in docs/V7_REPORT.md beside the SHA-256 of this set. A
             score taken after a reader was changed in response to these sentences is not an estimate for an unseen
             sentence and is never reported as one.

How a case is scored (strata.ai.eval_v7), mechanically:

  parameters   symbol, structure, expiration, strike, quantity, exits: compared field by field
  the entry    not by reading the expression. The interpretation is compiled and its entry rule is run on stored sessions;
               the reference rule written here is run on the same sessions; the two must fire at the same minutes of the
               same days. Two different spellings of the same rule are the same rule; a rule that drops a condition is not.
  experiments  the base strategy as above, plus the dimensions: which setting is varied and over exactly which values
  flags        a sentence that cannot be represented must be reported as such, not approximated

Clock times are New York time. A reference rule is a list of rule-language conditions (all must hold at the entry tick),
optionally after a `sequence` of stages that must have happened in order earlier in the session, optionally at fixed `times`.
"""
from __future__ import annotations

ATM = ("atm", 0.0)


def S(text, sym="SPY", structure="long_call", dte=0, strike=None, entry=None, when=(), sequence=(), exits=None, quantity=None, max_entries=None, adds=None, partial=None, tag="strategy"):
    """One strategy. `entry` is a clock time for a timed entry (None: the entry is the first tick at which the rule holds)."""
    return {"kind": "strategy", "text": text, "symbol": sym, "structure": structure, "dte": dte, "strike": strike, "entry": entry, "when": list(when), "sequence": [dict(s) for s in sequence], "exits": exits or {},
            "quantity": quantity, "max_entries": max_entries, "adds": adds, "partial": partial, "tag": tag}


def E(text, base, dims, tag="experiment"):
    """One experiment: a base strategy and the settings varied. dims: [(field, [values])]."""
    return {"kind": "experiment", "text": text, "base": base, "dims": [{"field": f, "values": list(v)} for f, v in dims], "tag": tag}


def U(text, tag="must be flagged"):
    return {"kind": "flag", "text": text, "tag": tag}


# ============================================================================ UNSEEN (sealed 2026-11-26, before the V7 readers)
UNSEEN = [
    # ---- intraday entries
    S("Buy an at-the-money QQQ put expiring the same day whenever the low of the first 30 minutes is broken, and get out at 2 PM New York time.",
      sym="QQQ", structure="long_put", dte=0, strike=ATM, when=["price < opening_range_low(30)"], exits={"time": "14:00"}, tag="levels"),
    S("Go long SPY 0DTE calls the first time SPY trades back above VWAP after having been down at least 0.75% from the open.",
      dte=0, sequence=[{"name": "drop", "when": "from_open_pct <= -0.75"}, {"name": "reclaim", "when": "crosses_above(price, vwap)"}], tag="sequence"),
    S("Buy IWM 0DTE calls after two consecutive bearish 15-minute candles followed by a bullish 15-minute candle.",
      sym="IWM", dte=0, sequence=[{"name": "bears", "when": 'consecutive_bearish_candles("15m") >= 2'}, {"name": "bull", "when": 'consecutive_bullish_candles("15m") >= 1'}], tag="candles"),
    S("Buy 1DTE SPY calls 1% out of the money at 10:15 AM Eastern if SPY gapped up more than 0.3% and is still above its opening price; sell after 45 minutes.",
      dte=1, strike=("pct_otm", 1.0), entry="10:15", when=["gap_pct > 0.3", "from_open_pct > 0"], exits={"minutes": 45}, tag="timed"),
    S("Buy 0DTE SPY puts at 11:00 AM Eastern if both SPY and QQQ are trading below their opening prices.",
      structure="long_put", dte=0, entry="11:00", when=["from_open_pct < 0", "QQQ.from_open_pct < 0"], tag="cross-symbol"),
    S("Sell a 0DTE SPY put credit spread 5 points wide at 9:45 AM Eastern when the low of the opening 15 minutes is above yesterday's high.",
      structure="bull_put_spread", dte=0, entry="09:45", when=["opening_range_low(15) > prev_high"], exits={}, tag="levels"),
    S("Buy QQQ 0DTE calls when price breaks above the high of the first hour, but only before 1 PM Eastern.",
      sym="QQQ", dte=0, when=["price > opening_range_high(60)", "time < hhmm(13, 0)"], tag="levels"),
    S("Buy an ATM SPY straddle expiring today at 3 PM Eastern when the day's range so far is under 0.4%, and close it five minutes before the bell.",
      structure="long_straddle", dte=0, strike=ATM, entry="15:00", when=["range_pct < 0.4"], exits={"time": "15:55"}, tag="timed"),
    S("Buy IWM 7DTE puts at the open the day after IWM closed higher for three sessions in a row, and hold for two sessions.",
      sym="IWM", structure="long_put", dte=7, entry="open", when=["consecutive_up_days() >= 3"], exits={"hold_sessions": 2}, tag="daily"),
    S("Buy SPY 0DTE calls on the first bullish five-minute candle after SPY has fallen more than 1% below the previous close.",
      dte=0, sequence=[{"name": "down", "when": "change_pct < -1"}, {"name": "green", "when": 'consecutive_bullish_candles("5m") >= 1'}], tag="candles"),
    S("Buy 30-delta QQQ calls two weeks out when QQQ crosses above yesterday's high, and take the trade off at a 40% gain or a 20% loss on the premium.",
      sym="QQQ", dte=14, strike=("delta", 0.30), when=["crosses_above(price, prev_high)"], exits={"premium_target": 40, "premium_stop": 20}, tag="levels"),
    S("Buy 0DTE SPY calls at noon Eastern on days when SPY has been at least 1% below its open at some point in the morning and is now back above the open.",
      dte=0, entry="12:00", when=["max_decline_from_open_pct <= -1", "from_open_pct > 0"], tag="state"),
    S("Buy ATM 0DTE QQQ puts when QQQ falls back below its opening price after having been up more than 0.6% on the day's open.",
      sym="QQQ", structure="long_put", dte=0, strike=ATM, sequence=[{"name": "rally", "when": "from_open_pct > 0.6"}, {"name": "fail", "when": "crosses_below(price, open)"}], tag="sequence"),
    S("Buy SPY 0DTE calls at 10:30 AM Eastern when SPY is above VWAP and the VIX closed below 18 yesterday; sell at a 1.5% rise in SPY or at 3 PM Eastern.",
      dte=0, entry="10:30", when=["price > vwap", "vix_prev_close < 18"], exits={"underlying_for": 1.5, "time": "15:00"}, tag="timed"),
    # ---- several entries, scaling, partial exits
    S("Enter SPY 0DTE puts up to three times a day, each time price crosses below VWAP, no later than 2:30 PM Eastern, and exit each one after 20 minutes.",
      structure="long_put", dte=0, when=["crosses_below(price, vwap)", "time <= hhmm(14, 30)"], exits={"minutes": 20}, max_entries=3, tag="multiple entries"),
    S("Buy one at-the-money SPY 0DTE call at 9:35 AM Eastern, add one more contract if SPY drops another half percent from the entry, and sell half of the position once the premium is up 25%.",
      dte=0, strike=ATM, entry="09:35", adds={"underlying_move_pct": -0.5, "quantity": 1, "max_adds": 1}, partial={"premium_pct": 25, "fraction": 0.5}, tag="multiple entries"),
    # ---- experiments in one sentence
    E("For SPY at-the-money 0DTE calls bought at the open after a down day of at least 1%, compare selling after 10, 20, 40 and 90 minutes.",
      S("", dte=0, strike=ATM, entry="open", when=["previous_session.return_pct <= -1"]), [("exit_minutes", [10, 20, 40, 90])]),
    E("Buy 0DTE calls when price breaks above the high of the first 30 minutes; run it on SPY, QQQ and IWM side by side.",
      S("", dte=0, when=["price > opening_range_high(30)"]), [("symbol", ["SPY", "QQQ", "IWM"])]),
    E("Buy SPY calls at 10 AM Eastern after a red previous day: test same-day, next-day, 2-day, 5-day and 21-day expirations.",
      S("", dte=0, entry="10:00", when=["previous_session.return_pct < 0"]), [("dte", [0, 1, 2, 5, 21])]),
    E("Buy 0DTE SPY calls at the open after a 2% down day; compare at-the-money with 0.5%, 1% and 2% out of the money.",
      S("", dte=0, entry="open", when=["previous_session.return_pct <= -2"]), [("strike", [("atm", 0.0), ("pct_otm", 0.5), ("pct_otm", 1.0), ("pct_otm", 2.0)])]),
    E("Try entry times every 30 minutes from 9:30 to 12:00 for buying ATM 0DTE QQQ puts after an up day.",
      S("", sym="QQQ", structure="long_put", dte=0, strike=ATM, when=["previous_session.return_pct > 0"]), [("entry_time", ["open", "10:00", "10:30", "11:00", "11:30", "12:00"])]),
    E("Buy SPY 0DTE calls at the open after a decline; sweep the decline from 1% to 4% in half-percent steps and the premium stop over 20%, 35% and 50%.",
      S("", dte=0, entry="open", when=["previous_session.return_pct <= -1"]), [("prev_decline", [1, 1.5, 2, 2.5, 3, 3.5, 4]), ("premium_stop", [20, 35, 50])]),
    E("Compare a 25% premium target with a 50% premium target, each with a 10% and a 20% premium stop, on ATM SPY 0DTE calls bought at 9:45 AM Eastern.",
      S("", dte=0, strike=ATM, entry="09:45"), [("premium_target", [25, 50]), ("premium_stop", [10, 20])]),
    E("Compare entering at the open, at 10:00 and at 11:00, with 0DTE against 1DTE, for SPY calls after a 1.5% down day.",
      S("", when=["previous_session.return_pct <= -1.5"]), [("entry_time", ["open", "10:00", "11:00"]), ("dte", [0, 1])]),
    E("Buy IWM puts expiring the same day at 10 AM Eastern after a gap up of more than 0.5%: compare holding 30 minutes, holding 2 hours and holding to the close.",
      S("", sym="IWM", structure="long_put", dte=0, entry="10:00", when=["gap_pct > 0.5"]), [("exit_minutes", [30, 120, "close"])]),
    E("Test deltas of 20, 30, 40 and 50 for SPY 7DTE calls bought at 10:30 AM Eastern after two down days in a row.",
      S("", dte=7, entry="10:30", when=["consecutive_down_days() >= 2"]), [("strike", [("delta", 0.20), ("delta", 0.30), ("delta", 0.40), ("delta", 0.50)])]),
    E("Same signal on SPY and QQQ: buy ATM 0DTE calls when price crosses above the opening 15-minute high, with exits after 30 and 60 minutes.",
      S("", dte=0, strike=ATM, when=["crosses_above(price, opening_range_high(15))"]), [("symbol", ["SPY", "QQQ"]), ("exit_minutes", [30, 60])]),
    E("Sweep the underlying target from 0.5% to 2% in half-percent steps for SPY 0DTE calls bought at the open after a 3% down day.",
      S("", dte=0, entry="open", when=["previous_session.return_pct <= -3"]), [("underlying_target", [0.5, 1, 1.5, 2])]),
    # ---- what cannot be represented and must be said so
    U("Buy SPY calls when the bid-ask spread on the at-the-money call tightens below 2 cents."),
    U("Buy QQQ puts when unusual options activity shows large call sweeps."),
    U("Buy SPY 0DTE calls when the cumulative NYSE tick index turns positive."),
    U("Sell IWM calls when retail sentiment on Reddit peaks."),
    U("Buy SPY calls when dark pool print volume exceeds 5 million shares in an hour."),
    U("Buy SPY 0DTE puts when the 10-year Treasury yield jumps more than 5 basis points intraday."),
]

# ============================================================================ DEVELOPMENT (what the V7 readers were built against)
BASE = "Buy ATM SPY 0DTE calls at the open after a red previous day."  # the strategy a follow-up below is said about
RED = ["previous_session.return_pct < 0"]


def F(case: dict) -> dict:
    """The case is a follow-up: it is read with BASE as the strategy the conversation is about."""
    return {**case, "follow_up": True}


def M(text, tag="brief", **fields):
    """A follow-up that changes the current strategy: the fields the draft must then have."""
    return {"kind": "modify", "text": text, "fields": fields, "tag": tag, "follow_up": True}


DEVELOPMENT = [
    # the examples of the V7 brief. Several of them name no strategy: they are said about the one under discussion
    S("Buy ATM SPY calls at the open after a red previous day.", strike=ATM, entry="open", when=RED, dte=None, tag="brief"),
    S("Buy QQQ puts when the first 15-minute candle breaks the previous day's low.", sym="QQQ", structure="long_put", dte=None, entry="09:45", when=["opening_range_low(15) < prev_low"], tag="brief"),
    S("Buy SPY calls when price reclaims VWAP after declining 0.5% from the open.", dte=None, sequence=[{"name": "drop", "when": "from_open_pct <= -0.5"}, {"name": "reclaim", "when": "crosses_above(price, vwap)"}], tag="brief"),
    S("Buy 0DTE calls after three consecutive bearish five-minute candles followed by a bullish reversal.", dte=0,
      sequence=[{"name": "bears", "when": 'consecutive_bearish_candles("5m") >= 3'}, {"name": "bull", "when": 'consecutive_bullish_candles("5m") >= 1'}], tag="brief"),
    F(E("Compare ATM and OTM contracts across multiple entry times.", S("", dte=0, when=RED), [("strike", [("atm", 0.0), ("pct_otm", 1.0)]), ("entry_time", ["open", "10:00", "11:00", "13:00"])], tag="brief")),
    M("Test a 30% premium profit target versus a 15% premium stop.", premium_target_pct=30.0, premium_stop_pct=15.0),
    F(E("Compare exits after 15, 30, 60, and 120 minutes.", S("", dte=0, strike=ATM, entry="open", when=RED), [("exit_minutes", [15, 30, 60, 120])], tag="brief")),
    F(E("Compare the same signal across SPY, QQQ, and IWM.", S("", dte=0, strike=ATM, entry="open", when=RED), [("symbol", ["SPY", "QQQ", "IWM"])], tag="brief")),
    F(E("Test 0DTE, 1DTE, 3DTE, 7DTE, and 30DTE contracts.", S("", strike=ATM, entry="open", when=RED), [("dte", [0, 1, 3, 7, 30])], tag="brief")),
    F(U("Test multiple entries and partial exits.", tag="brief")),  # it says neither how many nor at what level: that has to be asked, not invented
    F(E("Compare entry times from 9:30 to 11:00 in 15-minute increments, prior-day declines from 0.5% to 3%, 0DTE versus 1DTE, and several profit targets.", S("", strike=ATM, when=["previous_session.return_pct <= -0.5"]),
        [("entry_time", ["open", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00"]), ("prev_decline", [0.5, 1, 1.5, 2, 2.5, 3]), ("dte", [0, 1]), ("premium_target", [25, 50, 100])], tag="brief")),
    # variants written while the readers were built
    S("Buy 0DTE SPY calls at 10:00 AM Eastern, sell after 45 minutes.", dte=0, entry="10:00", exits={"minutes": 45}, tag="variant"),
    S("Buy SPY 0DTE puts up to three times a day when price crosses below VWAP; exit each after 20 minutes.", structure="long_put", dte=0, when=["crosses_below(price, vwap)"], exits={"minutes": 20}, max_entries=3, tag="variant"),
    S("Buy one ATM SPY 0DTE call at 9:35 AM Eastern and sell half when the premium is up 25%.", dte=0, strike=ATM, entry="09:35", partial={"premium_pct": 25, "fraction": 0.5}, tag="variant"),
    S("Buy SPY 0DTE calls when SPY breaks above the first 30 minutes' high with QQQ confirmation.", dte=0, when=["price > opening_range_high(30)", "QQQ.price > QQQ.opening_range_high(30)"], tag="variant"),
    S("Buy SPY 0DTE calls on the first green five-minute candle after SPY has dropped more than 1% from the open.", dte=0,
      sequence=[{"name": "drop", "when": "from_open_pct <= -1"}, {"name": "green", "when": 'consecutive_bullish_candles("5m") >= 1'}], tag="variant"),
    S("Buy QQQ 0DTE puts when price falls below the low of the first hour.", sym="QQQ", structure="long_put", dte=0, when=["price < opening_range_low(60)"], tag="variant"),
    E("Buy SPY 0DTE calls at the open after a 1% down day and compare 20, 40 and 80 delta strikes with exits after 5, 15 and 45 minutes.", S("", dte=0, entry="open", when=["previous_session.return_pct <= -1"]),
      [("strike", [("delta", 0.2), ("delta", 0.4), ("delta", 0.8)]), ("exit_minutes", [5, 15, 45])], tag="variant"),
    F(E("Hold for 20 minutes versus 90 minutes versus to the close.", S("", dte=0, strike=ATM, entry="open", when=RED), [("exit_minutes", [20, 90, "close"])], tag="variant")),
]


def sealed_hash() -> str:
    """SHA-256 of the unseen set as written: quoted in the report so that a later edit of a sentence or of its meaning shows."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(UNSEEN, sort_keys=True, default=str).encode()).hexdigest()
