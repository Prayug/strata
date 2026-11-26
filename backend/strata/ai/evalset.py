"""How well does Strata understand what it is told? A fixed set of sentences with the meaning each one has.

Every case states the MEANING a careful reader would take from the sentence, written
independently of what any of Strata's readers currently produce. `evaluate` runs each
sentence through the same routing the application uses (`intent.understand`) and compares
field by field. Three outcomes are kept apart, because they are not equally bad:

  correct   every stated element is in the interpretation, with nothing invented
  flagged   something was not understood AND the interpretation says so (the user is told)
  silent    the interpretation is wrong or incomplete and gives no sign of it

The number that matters most is `silent`: a reader that drops a condition without saying
so answers a different question from the one that was asked.

Times are New York time in the expectations: 6:30 AM Pacific is the 09:30 open.
"""
from __future__ import annotations

ORIGINAL = "Buy ATM SPY 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in SPY, only on days when SPY fell at least 3% the previous trading day."


def S(text, cond=(), out=None, sym="SPY", period=None, tag="study", **kw):
    return {"text": text, "kind": "study", "symbol": sym, "conditions": list(cond), "outcome": out, "period": period, "tag": tag, **kw}


def O(text, sym="SPY", cond=(), tag="strategy", **kw):
    return {"text": text, "kind": "strategy", "symbol": sym, "conditions": list(cond), "tag": tag, **kw}


def M(text, tag="follow-up", **kw):
    return {"text": text, "kind": "modify", "follow_up": True, "tag": tag, **kw}


def A(text, action, tag="conversation", follow_up=False, **kw):
    return {"text": text, "kind": "action", "action": action, "tag": tag, "follow_up": follow_up, **kw}


def U(text, tag="must be flagged"):
    return {"text": text, "kind": "flag", "tag": tag}


PREV = "prev_return|day_return"  # "after a 3% down day": the same fact, read at the next open or at that day's close
up = lambda **k: {"direction": "up", **k}
down = lambda **k: {"direction": "down", **k}

CASES = [
    # ------------------------------------------------------------------ questions about the underlying
    S("How often does SPY rise 1% from the open after a 3% down day?", [("prev_return", "le", -3)], up(size=1, measure="touch", **{"from": "open"}, horizon="session")),
    S("After SPY falls 3% in a day, how often does it gain at least 1% intraday the next session?", [("prev_return", "le", -3)], up(size=1, measure="touch", **{"from": "open"}, horizon="session")),
    S("What fraction of the time does SPY trade 1% above its open following a 3% decline?", [(PREV, "le", -3)], up(size=1, measure="touch", **{"from": "open"})),
    S("How often does SPY close higher the day after falling 2% or more?", [(PREV, "le", -2)], up(size=None, measure="close")),
    S("When SPY gaps down more than 1%, how often does it close above the open?", [("gap", "le", -1)], up(size=None, measure="close", **{"from": "open"}, horizon="session")),
    S("How often does SPY close lower after gapping up more than 0.5%?", [("gap", "ge", 0.5)], down(size=None, measure="close", **{"from": "open"})),
    S("How often does QQQ rise 2% within 5 days after falling more than 1%?", [(PREV, "le", -1)], up(size=2, measure="touch", horizon="sessions:5"), sym="QQQ"),
    S("How often does IWM rally 2% within 5 days after dropping 3% in a session?", [(PREV, "le", -3)], up(size=2, measure="touch", horizon="sessions:5"), sym="IWM"),
    S("After a 3% down day, what is the probability SPY is higher 5 days later?", [(PREV, "le", -3)], up(size=None, measure="close", horizon="sessions:5")),
    S("When VIX is above 30, how often does SPY finish the day up more than 1%?", [("vix", "ge", 30)], up(size=1, measure="close")),
    S("If VIX is above 25, how often does SPY fall more than 1% that day?", [("vix", "ge", 25)], down(size=1)),
    S("When the VIX closes above 30, how often does SPY rise over the next week?", [("vix", "ge", 30)], up(size=None, horizon="sessions:5")),
    S("Is SPY more likely to close higher when the previous day fell more than 1% and VIX is above 20?", [("prev_return", "le", -1), ("vix", "ge", 20)], up(size=None, measure="close")),
    S("When VIX is above 30 and SPY gaps down more than 1%, how often does SPY close above its open?", [("vix", "ge", 30), ("gap", "le", -1)], up(size=None, measure="close", **{"from": "open"})),
    S("When VIX is above 30, how often does SPY rise 1% from the open after a 3% down day?", [("vix", "ge", 30), ("prev_return", "le", -3)], up(size=1, measure="touch", **{"from": "open"})),
    S("How often does SPY drop 1% from the open on days after a 2% rally?", [("prev_return", "ge", 2)], down(size=1, measure="touch", **{"from": "open"})),
    S("How often does SPY gain more than 0.5% the day after a 1% up day?", [(PREV, "ge", 1)], up(size=0.5)),
    S("After three consecutive down days, how often does SPY rise the next day?", [("down_streak", "ge", 3)], up(size=None)),
    # "red" is a session that closed below its open (a red candle) or below the previous close: either reading is a streak of two
    S("After two red days in a row, how often is the third day green?", [("down_streak|red_streak", "ge", 2)], up(size=None, measure="close")),
    S("How often does QQQ finish green on Mondays?", [("weekday", "is", "Mon")], up(size=None, measure="close"), sym="QQQ"),
    S("Since 2015, how often does SPY close up after a down day of at least 2%?", [(PREV, "le", -2)], up(size=None, measure="close"), period=("2015", None)),
    S("Between 2018 and 2022, how often did QQQ rise 1% intraday after falling 2% the day before?", [("prev_return", "le", -2)], up(size=1, measure="touch", **{"from": "open"}), sym="QQQ", period=("2018", "2022")),
    S("When SPY is up more than 1% at 10:30, how often does it finish the day higher than that?", [("from_open", "ge", 1)], up(size=None, measure="close", **{"from": "10:30"})),
    S("If SPY drops 1% in the first hour, how often does it end the day lower than where it was at 10:30?", [("from_open", "le", -1)], down(size=None, measure="close", **{"from": "10:30"})),
    S("How often does SPY reach 1% above the open before falling 0.5% below it after a 2% decline the previous day?", [("prev_return", "le", -2)], up(size=1, measure="touch", stop=0.5, **{"from": "open"})),
    S("Does SPY tend to bounce after a 1.5% drop?", [(PREV, "le", -1.5)], up()),
    S("Does SPY tend to recover after falling at least 2% during the previous session?", [("prev_return", "le", -2)], up()),
    S("How often does SPY fall at least 1% the day after the VIX closes below 15?", [("vix", "le", 15)], down(size=1)),
    S("How often does SPY rise 1% from the open after a 4% down day?", [("prev_return", "le", -4)], up(size=1, measure="touch", **{"from": "open"})),
    S("Following a drop of 3 percent or more, how frequently does SPY climb 1 percent from the open?", [(PREV, "le", -3)], up(size=1, measure="touch", **{"from": "open"})),
    S("How often does SPY close at least 1% higher the day after a 2% drop?", [(PREV, "le", -2)], up(size=1, measure="close")),
    S("When SPY opens more than 1% lower, how often does it rally 1% from the open?", [("gap", "le", -1)], up(size=1, measure="touch", **{"from": "open"})),
    # decided at Thursday's close (Thursday, up 1%) or at Friday's open (Friday, previous session up 1%): the same sessions either way
    S("How often does IWM close lower on Fridays after rising 1% on Thursday?", [(PREV, "ge", 1), ("weekday", "is", "Thu|Fri")], down(size=None, measure="close"), sym="IWM"),
    S("After SPY drops 2%, how often does it drop another 1% the next day?", [(PREV, "le", -2)], down(size=1)),
    S("When QQQ falls more than 2% and VIX is above 25, how often does QQQ rise 1% from the open the next day?", [("prev_return", "le", -2), ("vix", "ge", 25)], up(size=1, measure="touch", **{"from": "open"}), sym="QQQ"),
    # ------------------------------------------------------------------ option strategies
    O(ORIGINAL, cond=[("prev_return", "le", -3)], structure="long_call", dte=0, strike=("atm", 0), entry="open", exits={"underlying_for": 1}),
    O("Buy ATM SPY 0DTE calls at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move.", structure="long_call", dte=0, strike=("atm", 0), entry="open", exits={"time": "15:45", "underlying_for": 1}),
    O("Buy SPY 0DTE at-the-money calls at 6:30 AM PT when SPY fell at least 1% the previous day.", cond=[("prev_return", "le", -1)], structure="long_call", dte=0, strike=("atm", 0), entry="open"),
    O("Buy SPY 0DTE at-the-money calls at 6:30 AM PT when SPY fell at least 2% the previous day.", cond=[("prev_return", "le", -2)], structure="long_call", dte=0, entry="open"),
    O("Buy ATM SPY 1DTE calls at the open after SPY closes down 3% or more.", cond=[("prev_return", "le", -3)], structure="long_call", dte=1, strike=("atm", 0), entry="open"),
    O("Buy ATM SPY 7DTE calls at the open after SPY closes down 3% or more.", cond=[("prev_return", "le", -3)], structure="long_call", dte=7, entry="open"),
    O("Sell a 30-delta SPY put with 7 days to expiration every Monday at 10:00 AM Eastern, take profit at 50% and stop at 200%.", cond=[("weekday", "is", "Mon")], structure="short_put", dte=7, strike=("delta", 0.30), entry="10:00",
      exits={"premium_target": 50, "premium_stop": 200}),
    O("Buy a 0DTE SPY put at 9:45 ET when SPY gaps up more than 1%, exit at 3 PM ET.", cond=[("gap", "ge", 1)], structure="long_put", dte=0, entry="09:45", exits={"time": "15:00"}),
    O("Sell an iron condor on SPY with 16-delta shorts and 5-point wings, 30 DTE, close at 50% profit.", structure="iron_condor", dte=30, strike=("delta", 0.16), width=5, exits={"premium_target": 50}),
    O("Buy ATM QQQ calls expiring the same day at 7:00 AM Pacific and sell at 12:00 PM Pacific.", sym="QQQ", structure="long_call", dte=0, strike=("atm", 0), entry="10:00", exits={"time": "15:00"}),
    O("Buy a SPY call spread 5 points wide, 1DTE, at the open after SPY closes down 2% or more; exit at the close.", cond=[("prev_return", "le", -2)], structure="bull_call_spread", dte=1, width=5, entry="open", exits={"time": "close"}),
    O("Buy SPY 0DTE calls 1% out of the money at 8:30 AM Central when the VIX is above 20, with a 100% profit target and a 50% stop.", cond=[("vix", "ge", 20)], structure="long_call", dte=0, strike=("pct_otm", 1), entry="open",
      exits={"premium_target": 100, "premium_stop": 50}),
    O("Short strangle on SPY, 20 delta, 14 DTE, entered Tuesdays at 11:00 AM Eastern, stop at 200% of the credit.", cond=[("weekday", "is", "Tue")], structure="short_strangle", dte=14, strike=("delta", 0.20), entry="11:00", exits={"premium_stop": 200}),
    O("Buy 2 SPY 7DTE calls at the money at 10 AM Eastern after two down days in a row.", cond=[("down_streak", "ge", 2)], structure="long_call", dte=7, strike=("atm", 0), quantity=2, entry="10:00"),
    O("Sell a put credit spread on QQQ, short strike at 25 delta, 10 wide, 45 DTE, take profit at 50%.", sym="QQQ", structure="bull_put_spread", dte=45, strike=("delta", 0.25), width=10, exits={"premium_target": 50}),
    O("Buy 0DTE ATM SPY puts at 11:00 AM ET when SPY is down 1% from the open, exit when SPY falls another 1% or at 3:30 PM ET.", cond=[("from_open", "le", -1)], structure="long_put", dte=0, strike=("atm", 0), entry="11:00",
      exits={"underlying_for": 1, "time": "15:30"}),
    O("Buy SPY 0DTE calls at 6:30 AM Pacific on days when SPY fell at least 3% the previous day and the VIX is below 40.", cond=[("prev_return", "le", -3), ("vix", "le", 40)], structure="long_call", dte=0, entry="open"),
    O("Buy SPY 0DTE calls at 6:30 AM Pacific after a 3% down day, 2 contracts, stop at 40% loss, target SPY up 1.5%.", cond=[("prev_return", "le", -3)], structure="long_call", dte=0, quantity=2, entry="open",
      exits={"premium_stop": 40, "underlying_for": 1.5}),
    O("Buy ATM 0DTE SPY puts at the open on days after SPY rallied 2% or more, exit at noon Eastern.", cond=[("prev_return", "ge", 2)], structure="long_put", dte=0, strike=("atm", 0), entry="open", exits={"time": "12:00"}),
    O("Sell a 0DTE SPY iron butterfly at 1 PM Eastern, 10-point wings, close at 25% profit.", structure="iron_butterfly", dte=0, width=10, entry="13:00", exits={"premium_target": 25}),
    O("Buy 0DTE SPY calls at 10:00 AM New York time if SPY is up at least 0.5% from the open; exit at 50% profit or 25% loss.", cond=[("from_open", "ge", 0.5)], structure="long_call", dte=0, entry="10:00",
      exits={"premium_target": 50, "premium_stop": 25}),
    O("Buy an ATM SPY straddle at 9:35 AM Eastern and close it at 2:30 PM Eastern.", structure="long_straddle", strike=("atm", 0), entry="09:35", exits={"time": "14:30"}),
    O("Sell SPY 0DTE puts at 20 delta at 10:30 AM ET when SPY is above the open, close at 3:45 PM ET.", cond=[("from_open", "ge", 0)], structure="short_put", dte=0, strike=("delta", 0.20), entry="10:30", exits={"time": "15:45"}),
    O("Buy a 30-delta SPY call 21 days out at 12:00 PM Eastern and hold to expiration.", structure="long_call", dte=21, strike=("delta", 0.30), entry="12:00", exits={"to_expiration": True}),
    O("Buy SPY 0DTE calls at the open after a 3% down day and sell at 12:45 PM Pacific.", cond=[("prev_return", "le", -3)], structure="long_call", dte=0, entry="open", exits={"time": "15:45"}),
    O("Buy a bear put spread on IWM, 3 points wide, 14 DTE, at 10 AM ET when IWM gapped down more than 1%.", sym="IWM", cond=[("gap", "le", -1)], structure="bear_put_spread", dte=14, width=3, entry="10:00"),
    O("Sell a call credit spread on SPY, 30 delta short, 5 wide, 7 DTE, at 9:45 AM ET after SPY rose 2% the previous day; take profit at 50%, stop at 100%.", cond=[("prev_return", "ge", 2)], structure="bear_call_spread", dte=7,
      strike=("delta", 0.30), width=5, entry="09:45", exits={"premium_target": 50, "premium_stop": 100}),
    O("Buy SPY 0DTE calls at 6:30 AM Pacific when SPY fell at least 3% the previous day, using 3 contracts and a 60% premium stop.", cond=[("prev_return", "le", -3)], structure="long_call", dte=0, quantity=3, entry="open", exits={"premium_stop": 60}),
    O("Buy QQQ 0DTE ATM puts at 6:45 AM Pacific after QQQ rose more than 2% the previous day; exit at 9:00 AM Pacific.", sym="QQQ", cond=[("prev_return", "ge", 2)], structure="long_put", dte=0, strike=("atm", 0), entry="09:45",
      exits={"time": "12:00"}),
    # time zones: one strategy, eight ways of saying when
    O("Buy ATM SPY 0DTE calls at 6:30 AM PT.", structure="long_call", dte=0, entry="open", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 7 AM PST.", structure="long_call", dte=0, entry="10:00", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 8:45 AM Central.", structure="long_call", dte=0, entry="09:45", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 10:15 AM EST.", structure="long_call", dte=0, entry="10:15", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 12:30 PM Pacific.", structure="long_call", dte=0, entry="15:30", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 1 PM ET.", structure="long_call", dte=0, entry="13:00", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 9:30 AM Eastern.", structure="long_call", dte=0, entry="open", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 7:35 AM Mountain.", structure="long_call", dte=0, entry="09:35", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 6:30 AM Pacific and sell them at 1 PM Eastern.", structure="long_call", dte=0, entry="open", exits={"time": "13:00"}, tag="time zones"),
    # no zone is stated: whichever the installation's default is, the interpretation has to SAY which one it assumed
    O("Buy ATM SPY 0DTE calls at 9:45 AM and sell them at 11:15 AM.", structure="long_call", dte=0, tz_assumed=True, tag="time zones"),
    # ------------------------------------------------------------------ follow-ups on the original strategy
    A("Run the same strategy after 1% and 2% declines.", "sweep", follow_up=True, values=[1, 2], tag="follow-up"),
    A("Compare 0DTE, 1DTE, and 7DTE calls.", "sweep", follow_up=True, field="dte", values=[0, 1, 7], tag="follow-up"),
    M("Enter at 6:30 AM Pacific and exit at 12:45 PM Pacific or after a 1% underlying move.", entry="open", exits={"time": "15:45", "underlying_for": 1}, keep=[("prev_return", "le", -3)]),
    M("Make the target 2% instead.", exits={"underlying_for": 2}, keep=[("prev_return", "le", -3)]),
    M("Use puts instead.", structure="long_put", keep=[("prev_return", "le", -3)]),
    M("Add a 50% stop loss on the premium.", exits={"premium_stop": 50, "underlying_for": 1}, keep=[("prev_return", "le", -3)]),
    M("Exit at 11:30 AM Pacific instead.", exits={"time": "14:30"}, keep=[("prev_return", "le", -3)]),
    M("Use 7DTE instead of 0DTE.", dte=7, keep=[("prev_return", "le", -3)]),
    M("Buy 3 contracts.", quantity=3, keep=[("prev_return", "le", -3)]),
    M("Use the 30-delta call instead of at the money.", strike=("delta", 0.30), keep=[("prev_return", "le", -3)]),
    M("Require the VIX to be above 20 as well.", keep=[("prev_return", "le", -3), ("vix", "ge", 20)]),
    M("Change the decline threshold to 2.5%.", keep=[("prev_return", "le", -2.5)]),
    M("Only trade on Mondays and Fridays.", keep=[("prev_return", "le", -3), ("weekday", "is", ["Mon", "Fri"])]),
    M("Exit at the close instead.", exits={"time": "close"}, keep=[("prev_return", "le", -3)]),
    M("Enter at 10 AM Eastern instead of the open.", entry="10:00", keep=[("prev_return", "le", -3)]),
    A("Include commissions and conservative slippage.", "modify", follow_up=True, changed=["costs"], tag="follow-up"),
    A("Show every actual option fill and its source.", "fills", follow_up=True, tag="follow-up"),
    A("Show exactly which historical sessions could not be tested.", "untested", follow_up=True, tag="follow-up"),
    A("Estimate what it would cost to acquire the missing observations.", "cost", follow_up=True, tag="follow-up"),
    # ------------------------------------------------------------------ conversation
    A("What have we found so far?", "memory"),
    A("Have I already tested this?", "registry"),
    A("Start a frozen forward-validation experiment.", "forward"),
    A("Challenge this result.", "challenge"),
    A("Stress test the execution assumptions.", "stress"),
    A("Which results are actually independent?", "independent"),
    A("Does this still hold after 2020?", "still_hold", year=2020),
    A("Summarize our research.", "memory"),
    A("Is the holdout still untouched?", "registry"),
    A("Play devil's advocate.", "challenge"),
    # ------------------------------------------------------------------ what must be flagged rather than misread
    U("Buy SPY 0DTE calls at the open when Mercury is in retrograde."),
    U("Buy SPY 0DTE calls at 10 AM Eastern when the order book imbalance exceeds 60%."),
    U("Sell SPY puts at the open whenever the Fed chair sounds dovish."),
    U("Buy SPY calls at the open when dealer gamma exposure turns negative."),
    U("How often does SPY rise 1% after the Fed chair smiles?"),
    U("Buy SPY 0DTE calls at the open after a 3% down day, but only if Twitter sentiment is bullish."),
    U("Buy ATM SPY calls at the open when the put/call open interest ratio on the 500 strike is above 2."),
]


# Written AFTER the readers were corrected against CASES, in different words, and scored once before anything was changed
# in response. CASES measures what was fixed; HELDOUT is the closer estimate of how an unseen sentence fares.
HELDOUT = [
    S("What percentage of the time does SPY finish higher on the day following a decline of 2.5% or more?", [(PREV, "le", -2.5)], up(size=None, measure="close")),
    S("After QQQ gaps up at least 1%, how often does it close below its opening price?", [("gap", "ge", 1)], down(size=None, measure="close", **{"from": "open"}), sym="QQQ"),
    S("How frequently does SPY trade 0.75% above the open on days when the VIX closed above 28 the day before?", [("vix", "ge", 28)], up(size=0.75, measure="touch", **{"from": "open"})),
    S("If SPY loses 4% in one session, what are the odds it gains 2% at some point over the following 3 days?", [(PREV, "le", -4)], up(size=2, measure="touch", horizon="sessions:3")),
    S("When IWM is down 1.5% at 11:00, how often does it close above its 11:00 price?", [("from_open", "le", -1.5)], up(size=None, measure="close", **{"from": "11:00"}), sym="IWM"),
    S("How often does SPY end the session lower after opening more than 0.75% higher?", [("gap", "ge", 0.75)], down(size=None, measure="close", **{"from": "open"})),
    S("On Wednesdays after a 1% drop, how often does SPY rise 1% from the open?", [("prev_return", "le", -1), ("weekday", "is", "Wed")], up(size=1, measure="touch", **{"from": "open"})),
    S("How often is SPY up more than 2% five sessions after it falls 5% in a day?", [(PREV, "le", -5)], up(size=2, measure="close", horizon="sessions:5")),
    S("After four straight down days, how often does QQQ close green the next session?", [("down_streak|red_streak", "ge", 4)], up(size=None, measure="close"), sym="QQQ"),
    S("When the VIX is under 14 and SPY gapped up over 0.5%, how often does SPY fall 0.5% from the open?", [("vix", "le", 14), ("gap", "ge", 0.5)], down(size=0.5, measure="touch", **{"from": "open"})),
    O("Purchase at-the-money SPY calls that expire the same day at 9:30 AM Eastern whenever SPY lost 2.5% or more the day before; close the trade once SPY is up 0.75% or at 3:50 PM Eastern.", cond=[("prev_return", "le", -2.5)],
      structure="long_call", dte=0, strike=("atm", 0), entry="open", exits={"underlying_for": 0.75, "time": "15:50"}),
    O("Go long SPY puts, 0DTE, at the money, at 7:15 AM Pacific after SPY gained more than 1.5% the prior session, and get out at 11:00 AM Pacific.", cond=[("prev_return", "ge", 1.5)], structure="long_put", dte=0, strike=("atm", 0),
      entry="10:15", exits={"time": "14:00"}),
    O("Write a 15-delta QQQ put 10 days out every Thursday at 9:40 AM Eastern; take 60% of the credit as profit.", sym="QQQ", cond=[("weekday", "is", "Thu")], structure="short_put", dte=10, strike=("delta", 0.15), entry="09:40",
      exits={"premium_target": 60}),
    O("Open an iron condor on IWM, 45 DTE, shorts at 12 delta, 4 wide, and close at 50% of the credit.", sym="IWM", structure="iron_condor", dte=45, strike=("delta", 0.12), width=4, exits={"premium_target": 50}),
    O("Buy 5 contracts of the SPY 0DTE at-the-money call at 10:45 AM Central if SPY is down more than 0.8% from the open, with a 35% premium stop.", cond=[("from_open", "le", -0.8)], structure="long_call", dte=0, strike=("atm", 0),
      quantity=5, entry="11:45", exits={"premium_stop": 35}),
    O("Sell a SPY call credit spread, 3 points wide, 20 delta, 2 DTE at 1:15 PM Eastern on days the VIX is over 22.", cond=[("vix", "ge", 22)], structure="bear_call_spread", dte=2, strike=("delta", 0.20), width=3, entry="13:15"),
    O("Buy SPY 0DTE calls at the opening bell following a session where SPY dropped at least 3.5%, exiting on a 1.25% rise in SPY.", cond=[("prev_return", "le", -3.5)], structure="long_call", dte=0, entry="open",
      exits={"underlying_for": 1.25}),
    O("Long straddle on QQQ, at the money, 1DTE, entered at 12:30 PM Pacific.", sym="QQQ", structure="long_straddle", dte=1, strike=("atm", 0), entry="15:30"),
    O("Buy a 0DTE SPY call 0.5% out of the money at 9:50 AM Eastern after three consecutive losing days, target 80% profit.", cond=[("down_streak|red_streak", "ge", 3)], structure="long_call", dte=0, strike=("pct_otm", 0.5), entry="09:50",
      exits={"premium_target": 80}),
    O("Short a 10-delta SPY call 5 days to expiry at 2 PM Eastern when SPY gapped up more than 1%.", cond=[("gap", "ge", 1)], structure="short_call", dte=5, strike=("delta", 0.10), entry="14:00"),
    O("Buy ATM SPY 0DTE calls at 11:20 AM Mountain.", structure="long_call", dte=0, entry="13:20", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 3:10 PM ET.", structure="long_call", dte=0, entry="15:10", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at 9:55 AM Central time and sell at 12:10 PM Central time.", structure="long_call", dte=0, entry="10:55", exits={"time": "13:10"}, tag="time zones"),
    O("Buy ATM SPY 0DTE puts at 6:35 AM PDT.", structure="long_put", dte=0, entry="09:35", tag="time zones"),
    O("Buy ATM SPY 0DTE calls at noon Eastern.", structure="long_call", dte=0, entry="12:00", tag="time zones"),
    M("What if the decline has to be 4% instead?", keep=[("prev_return", "le", -4)]),
    M("Switch to 2 contracts and get out at 1 PM Pacific.", quantity=2, exits={"time": "close"}, keep=[("prev_return", "le", -3)]),
    A("How does it do on 1DTE versus 0DTE?", "sweep", follow_up=True, field="dte", values=[0, 1], tag="follow-up"),
    M("Hold until 3 PM Eastern instead.", exits={"time": "15:00"}, keep=[("prev_return", "le", -3)]),
    M("Raise the stop to 75% of the premium.", exits={"premium_stop": 75}, keep=[("prev_return", "le", -3)]),
    A("What did we learn so far?", "memory"),
    A("Which of these sessions were missing data?", "untested", follow_up=True, tag="follow-up"),
    A("Poke holes in this.", "challenge"),
    A("Where do the fill prices come from?", "fills", follow_up=True, tag="follow-up"),
    A("How much would the missing data cost?", "cost", follow_up=True, tag="follow-up"),
    U("Buy SPY 0DTE calls at the open when insiders are buying."),
    U("How often does SPY rise 1% from the open when retail sentiment is euphoric?"),
    U("Sell SPY puts at 10 AM Eastern if the 10-year yield falls 5 basis points."),
    U("Buy SPY calls at the open if the skew index is above 140."),
    U("Buy SPY 0DTE calls at the open after a 3% down day when earnings season has started."),
]


# ----------------------------------------------------------------------------- comparison
def _t(x):
    """Clock times in one spelling: the 09:30 open and the 16:00 close have names."""
    return {"09:30": "open", "9:30": "open", "16:00": "close"}.get(str(x), x)


def _op(o):
    return {"<": "le", "<=": "le", ">": "ge", ">=": "ge", "==": "is", "in": "is", "is": "is"}.get(o, o)


def _match_cond(exp: tuple, c: dict) -> bool:
    keys = set(exp[0].split("|"))
    if c.get("key") not in keys or _op(c.get("op")) != exp[1]:
        return False
    v, want = c.get("value"), exp[2]
    if isinstance(want, (int, float)):
        vals = c.get("values") or ([v] if not isinstance(v, list) else v)
        return any(isinstance(x, (int, float)) and abs(float(x) - float(want)) < 1e-9 for x in vals)
    if isinstance(want, list):
        return sorted(v if isinstance(v, list) else [v]) == sorted(want)
    alts = [a.lower() for a in str(want).split("|")]
    return any(a in [str(x).lower() for x in v] for a in alts) if isinstance(v, list) else any(str(v).lower().startswith(a) for a in alts)


def check(case: dict, it: dict) -> list[str]:
    """What differs between the meaning of the sentence and the interpretation. Empty when they agree."""
    bad: list[str] = []
    kind = case["kind"]
    if kind == "flag":
        flagged = bool(it.get("unsupported")) or it.get("kind") == "unreadable" or bool(it.get("questions")) or any(not c["ok"] for c in it.get("checks") or [])
        return [] if flagged else ["the part that cannot be represented was not flagged"]
    if kind == "action":
        if it.get("kind") == "action" and it.get("action") == case["action"]:
            if "values" in case and sorted(float(abs(x)) for x in (it.get("values") or [])) != sorted(float(x) for x in case["values"]):
                bad.append(f"values {it.get('values')} for {case['values']}")
            if "field" in case and case["field"] not in str(it.get("field") or ""):
                bad.append(f"field {it.get('field')} for {case['field']}")
            if "year" in case and it.get("year") != case["year"]:
                bad.append(f"year {it.get('year')}")
            return bad
        if case["action"] == "modify" and it.get("kind") == "modify" and set(case.get("changed") or []) <= set(it.get("changed") or []):
            return []
        return [f"read as {it.get('kind')}{':' + str(it.get('action')) if it.get('action') else ''}, not as the request '{case['action']}'"]
    want_kind = "modify" if kind == "modify" else kind
    if it.get("kind") != want_kind:
        return [f"read as {it.get('kind')}{':' + str(it.get('action')) if it.get('action') else ''}, not as a {want_kind}"]
    if case.get("symbol") and kind != "modify" and it.get("symbol") != case["symbol"]:
        bad.append(f"symbol {it.get('symbol')} for {case['symbol']}")
    conds = it.get("conditions") or []
    want = case.get("keep") if kind == "modify" else case.get("conditions")
    if want is not None:
        for e in want:
            if not any(_match_cond(e, c) for c in conds):
                bad.append(f"condition {e[0]} {e[1]} {e[2]} is missing")
        if len(conds) > len(want):
            extra = [c for c in conds if not any(_match_cond(e, c) for e in want)]
            if extra:
                bad.append("conditions that were not stated: " + "; ".join(f"{c.get('key')} {c.get('op')} {c.get('value')}" for c in extra))
    o = case.get("outcome")
    if o:
        got = it.get("outcome") or {}
        for k, v in o.items():
            g = got.get(k)
            if k == "from":
                g, v = _t(g), _t(v)
            if k == "size":
                g, v = (float(g) if g else None), (float(v) if v else None)
            if k == "stop":
                g, v = (float(g) if g else None), float(v)
            if g != v:
                bad.append(f"outcome {k}: {g} for {v}")
    for k in ("structure", "dte", "width", "quantity"):
        if k in case and it.get(k) != case[k] and not (k == "width" and it.get(k) is not None and float(it[k]) == float(case[k])):
            bad.append(f"{k}: {it.get(k)} for {case[k]}")
    if "strike" in case:
        s = it.get("strike") or {}
        m, v = case["strike"]
        if s.get("method") != m or abs(float(s.get("value") or 0) - float(v)) > 1e-9:
            bad.append(f"strike: {s.get('method')} {s.get('value')} for {m} {v}")
    if "entry" in case and _t(it.get("entry")) != _t(case["entry"]):
        bad.append(f"entry time: {it.get('entry')} for {case['entry']}")
    for k, v in (case.get("exits") or {}).items():
        g = (it.get("exits") or {}).get(k)
        if k == "time":
            g, v = _t(g), _t(v)
        if isinstance(v, bool):
            g = bool(g)
        if (float(g) if isinstance(g, (int, float)) and not isinstance(g, bool) else g) != (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v):
            bad.append(f"exit {k}: {g} for {v}")
    if case.get("period"):
        got = it.get("period") or [None, None]
        for i, y in enumerate(case["period"]):
            if y and not str(got[i] or "").startswith(str(y)):
                bad.append(f"period {'start' if i == 0 else 'end'}: {got[i]} for {y}")
    if case.get("tz_assumed") and not it.get("tz_assumed"):
        bad.append("no time zone was stated and the interpretation does not say which it assumed")
    return bad


def evaluate(agent, default_symbol: str = "SPY", cases: list[dict] | None = None, llm: bool = False) -> dict:
    """Run every case through the application's own routing and score it. With `llm`, the configured language model reads
    each sentence instead of the built-in readers (it raises when none is configured: that run is then not made, not passed)."""
    from strata.ai import intent as I
    cases = cases if cases is not None else CASES
    if llm and agent.llm()[0] is None:
        # no model, no measurement: a run that could not be made is refused here, not scored as 150 failures (or worse, passes)
        raise RuntimeError("no language model is configured, so its interpretation cannot be measured: add a provider key in Settings, AI assistant. The built-in readers need none (run without --llm).")
    known = sorted(agent.store.stored_symbols())
    base = agent._rules(ORIGINAL, None, known, None, default_symbol).get("draft")
    rows = []
    for i, c in enumerate(cases):
        try:
            it = I.understand(agent, c["text"], default_symbol, current=base if c.get("follow_up") else None, llm=llm)
            bad = check(c, it)
        except Exception as e:  # noqa: BLE001  (a reader that crashes on a sentence has not understood it)
            it, bad = {"kind": "error", "unsupported": [], "checks": []}, [f"the reader failed: {type(e).__name__}: {e}"[:200]]
        said = bool(it.get("unsupported")) or bool(it.get("questions")) or it.get("kind") in ("unreadable", "error") or any(not x["ok"] for x in it.get("checks") or [])
        status = "correct" if not bad else ("flagged" if said and c["kind"] != "flag" else "silent")
        rows.append({"n": i + 1, "tag": c["tag"], "text": c["text"], "status": status, "differences": bad, "read_as": it.get("kind") + (":" + str(it.get("action")) if it.get("action") else ""), "reader": it.get("reader"),
                     "unsupported": [str(u)[:120] for u in (it.get("unsupported") or [])][:4]})
    by: dict[str, dict] = {}
    for r in rows:
        b = by.setdefault(r["tag"], {"cases": 0, "correct": 0, "flagged": 0, "silent": 0})
        b["cases"] += 1
        b[r["status"]] += 1
    n = len(rows)
    tot = {k: sum(1 for r in rows if r["status"] == k) for k in ("correct", "flagged", "silent")}
    return {"cases": n, **tot, "accuracy_pct": round(tot["correct"] / n * 100, 1) if n else None, "silent_pct": round(tot["silent"] / n * 100, 1) if n else None, "by_tag": by,
            "failures": [r for r in rows if r["status"] != "correct"], "rows": rows,
            "interpreter": "language model" if llm else "built-in readers",
            "note": ("Measured with the configured language model. " if llm else "Measured on the built-in readers (no language model). ") + "'Silent' means the interpretation is wrong or incomplete and does not say so."}


HELDOUT_FIRST_RUN = {"cases": 40, "correct": 30, "flagged": 2, "silent": 8, "accuracy_pct": 75.0, "silent_pct": 20.0, "date": "2026-11-25",
                     "note": "The held-out sentences as first scored, before any reader was changed in response. This is the estimate for a sentence nobody has tuned for; the later figure on the same sentences is not."}


def report(agent, default_symbol: str = "SPY", llm: bool = False) -> dict:
    """Both sets, with the held-out set's first-run figures kept beside its current ones."""
    dev = evaluate(agent, default_symbol, CASES, llm)
    held = evaluate(agent, default_symbol, HELDOUT, llm)
    slim = lambda r: {k: r[k] for k in ("cases", "correct", "flagged", "silent", "accuracy_pct", "silent_pct", "by_tag", "failures", "interpreter")}
    return {"development": slim(dev), "held_out": slim(held), "held_out_first_run": HELDOUT_FIRST_RUN, "interpreter": dev["interpreter"],
            "note": "Development sentences were used to correct the readers, so their score says what was fixed. The held-out first run (75%, 8 silent of 40) is the honest estimate for an unseen sentence; "
                    "those sentences have since been used too. " + dev["note"]}
