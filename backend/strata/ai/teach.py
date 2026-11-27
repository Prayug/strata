"""Explanations built from the current result (V8 teaching layer).

"Explain this like I'm new to options", "why did the option lose value if SPY went up?", "what does this chart mean?",
"why is this strategy considered unreliable?", "what does changing the expiration actually do?", "how much money would
this have made?". Each answer is written from the run's own trades, prices and counts: no textbook paragraph and no number
that the engine did not produce. Where a picture helps (calls versus shares), the answer carries an `illustration` that is
labelled as one: it is arithmetic on a made-up price range, not historical data.
"""
from __future__ import annotations

import re

from strata.ai import answer as ANS

TOPICS = [
    ("difference_luck", r"\b(?:the |this |that )?(?:difference|gap|edge|improvement)\b.{0,40}\b(?:luck|lucky|chance|random|real|significant|noise)\b|\b(?:luck|chance|noise)\b.{0,30}\bdifference\b"),
    ("new_to_options", r"\b(?:explain|walk me through|break down)\b.{0,40}\b(?:like i'?m|as if i'?m|for a|to a) (?:new|beginner|five|5|newbie|novice|noob)|\bi'?m new to (?:options|this|trading)\b|\bnew to options\b|\bexplain (?:this|it|that) simply\b|\bin plain english\b|\bfor (?:a )?beginners?\b"),
    ("why_lost_value", r"\bwhy did (?:the |my |this |that )?(?:option|call|put|contract|premium)s? (?:lose|lost|go down|drop|fall)\b|\b(?:option|call|premium) (?:lost|lose|went down|fell)\b.{0,40}\b(?:went up|rose|rallied|was up|moved up)\b|\bwent up but (?:the )?(?:option|call) (?:lost|went down|fell)\b"),
    ("chart", r"\bwhat (?:does|do) (?:this|that|the|these) (?:chart|graph|picture|plot|lines?|bars?|matrix|grid)s? (?:mean|show|say)\b|\bhow (?:do|should) i read (?:this|the|that) (?:chart|graph|matrix|grid)\b|\bexplain (?:this|the) (?:chart|graph)\b"),
    ("unreliable", r"\bwhy (?:is|was) (?:this|it|that|the) (?:strategy |result |finding |backtest )?(?:considered |called |marked |rated )?(?:unreliable|not reliable|weak|not significant|luck|lucky|not trustworthy)\b|\bwhy (?:can'?t|shouldn'?t) i trust (?:this|it|that)\b|\bwhat makes (?:this|it) unreliable\b"),
    ("expiration", r"\bwhat (?:does|would) chang(?:e|ing) the expiration\b|\bwhat does (?:the )?expiration (?:actually )?do\b|\bwhy does (?:the )?expiration matter\b|\b(?:0dte|same[- ]day) (?:vs|versus|or|and) \d+ ?dte\b.{0,30}\b(?:difference|differ|mean)\b"),
    ("difference", r"\bwhat(?:'s| is) the difference between (?:these|the|those) (?:two )?(?:results|runs|versions|strategies)\b|\bhow (?:do|are) (?:these|the two) (?:results|runs|versions) differ"),
    ("outperform", r"\bwhy did (?:the )?.{1,40}\b(?:outperform|beat|do better than|work better than)\b"),
    ("how_much", r"\bhow much (?:money )?(?:would|did|does|will) (?:this|it|that|the strategy|i) (?:strategy )?(?:have )?(?:made|make|earn|earned|lose|lost)\b|\bhow much (?:money )?(?:would i have|did i) (?:made|make|earn|lose|lost)\b"),
]


def topic_of(text: str) -> str | None:
    low = text.lower().replace("’", "'")
    for name, pat in TOPICS:
        if re.search(pat, low):
            return name
    return None


def _strategy_entry(messages: list[dict]):
    """The latest finished option backtest in the conversation: (message, run, trades) or None."""
    from strata.api import db
    from strata.api.results import load_trades
    for m in reversed(messages):
        c = m["content"] if isinstance(m["content"], dict) else {}
        if m["kind"] == "result" and c.get("run_id") and c.get("view") in ("single", "compare", "explain_trade"):
            run = db.get_run(c["run_id"])
            if run and run.get("kind") != "study":
                return m, run, load_trades(run["id"])
    return None


def _latest_result(messages: list[dict]):
    for m in reversed(messages):
        c = m["content"] if isinstance(m["content"], dict) else {}
        if m["kind"] == "result" and c.get("view") not in ("failed", "explain"):
            return m
    return None


def _typical(trades: list[dict]) -> dict | None:
    if not trades:
        return None
    s = sorted(trades, key=lambda t: t.get("pnl") or 0)
    return s[len(s) // 2]


def _leg(t: dict) -> dict | None:
    return next((x for x in t.get("legs") or [] if x.get("kind", "option") == "option"), None)


def new_to_options(messages: list[dict]) -> dict:
    ent = _strategy_entry(messages)
    if ent is None:
        return {"headline": "There is no option backtest in this conversation yet, so there is no real trade to walk through.",
                "happened": ["Ask for one, for example \"Buy SPY calls at the open after a 1% down day\", and then ask again: the explanation uses one of its actual trades."], "limits": []}
    m, run, trades = ent
    t = _typical(trades)
    l0 = _leg(t) if t else None
    if t is None or l0 is None:
        return {"headline": "This backtest made no option trades, so there is nothing to walk through.", "happened": [], "limits": []}
    sym, k, right = t.get("symbol") or "SPY", float(l0["strike"]), l0.get("right") or "C"
    mult = float(l0.get("multiplier") or 100)
    n = int(l0.get("contracts") or t.get("units") or 1)
    p0, p1 = float(l0["entry_price"]), float(l0["exit_price"])
    s0, s1 = t.get("entry_spot"), t.get("exit_spot")
    cost = p0 * mult * n
    fees = (t.get("commissions") or 0) + (t.get("slippage") or 0)
    kind = "call" if right == "C" else "put"
    exp_same = l0.get("expiration") == t.get("entry_date")
    up = right == "C"
    lines = [
        f"On {ANS._date(t['entry_date'])} the rule bought {n} {sym} {k:g} {kind}{'s' if n != 1 else ''} expiring {'the same day' if exp_same else ANS._date(l0.get('expiration'))}. "
        f"A {kind} gains value when {sym} {'rises above' if up else 'falls below'} ${k:g} (the strike) before it expires, and is worth nothing at expiry if it does not.",
        f"It cost ${p0:.2f} per share. One contract covers {mult:g} shares, so it cost ${cost:,.2f}" + (f" plus ${fees:.2f} in fees." if fees else ".") +
        f" That ${cost:,.2f} is the most this trade could lose.",
    ]
    if s0 and s1:
        u = (s1 / s0 - 1) * 100
        pr = (p1 / p0 - 1) * 100 if p0 else 0
        lines.append(f"{sym} went from ${s0:,.2f} to ${s1:,.2f} ({u:+.2f}%) while the {kind} went from ${p0:.2f} to ${p1:.2f} ({pr:+.0f}%). "
                     f"That is the leverage of an option: a {abs(u):.2f}% move in {sym} became a {abs(pr):.0f}% change in what the contract was worth.")
    pnl = t.get("pnl") or 0
    lines.append(f"It closed ({ANS_REASON(t)}) for {'a gain' if pnl > 0 else 'a loss'} of ${abs(pnl):,.2f} after fees. This was a typical trade of this backtest: half of its {len(trades)} trades did better and half worse.")
    illus = None
    if s0:
        shares = cost / s0
        illus = {"kind": "payoff", "illustration": True, "symbol": sym, "right": right, "strike": k, "premium": p0, "multiplier": mult, "contracts": n, "spot": s0, "shares": round(shares, 3),
                 "note": f"Illustration, not historical data: what ${cost:,.0f} in this {kind} and ${cost:,.0f} in {sym} shares ({shares:.2f} shares) would be worth at expiry, for a range of prices."}
        sh_pnl = shares * (s1 - s0) if s1 else None
        if sh_pnl is not None:
            lines.append(f"The same ${cost:,.2f} in {sym} shares would have bought {shares:.2f} shares and {'made' if sh_pnl >= 0 else 'lost'} ${abs(sh_pnl):,.2f} on that move: "
                         f"smaller swings, and no expiry date that can make the whole amount disappear.")
    return {"headline": f"What one of these trades actually was, step by step.", "happened": lines,
            "limits": ["Prices are trade prints, not bid/ask quotes: a real order could have been filled a little worse."],
            "illustration": illus, "trade": {"run_id": run["id"], "trade_id": t["id"]}}


def ANS_REASON(t: dict) -> str:
    from strata.research.explore import REASONS
    return REASONS.get(t.get("exit_reason"), str(t.get("exit_reason") or "exit").replace("_", " "))


def why_lost_value(messages: list[dict]) -> dict:
    ent = _strategy_entry(messages)
    if ent is None:
        return {"headline": "There is no option backtest in this conversation yet, so there is no trade to look at.", "happened": [], "limits": []}
    m, run, trades = ent
    cands = []
    for t in trades:
        l0 = _leg(t)
        if not l0 or not t.get("entry_spot") or not t.get("exit_spot"):
            continue
        right_way = (t["exit_spot"] > t["entry_spot"]) if (l0.get("right") or "C") == "C" else (t["exit_spot"] < t["entry_spot"])
        if right_way and (t.get("pnl") or 0) < 0:
            cands.append(t)
    sym = (trades[0].get("symbol") if trades else None) or "SPY"
    if not cands:
        return {"headline": f"In this backtest no trade lost money while {sym} moved the right way.",
                "happened": [f"Every losing trade of the {len(trades)} lost because {sym} moved against it (or did not move far enough before the exit)."], "limits": []}
    worst = min(cands, key=lambda t: t.get("pnl") or 0)
    story = ANS.why_lost(worst, run["id"])
    head = (f"{len(cands)} of the {len(trades)} trades lost money even though {sym} moved the right way. The clearest case is "
            f"{ANS._date(worst['entry_date'])}:")
    return {"headline": head, "happened": story.get("cause") or [], "limits": ["The split between intrinsic and time value is arithmetic on the trade's own prices; implied volatility changes are part of the time value here."],
            "trade": {"run_id": run["id"], "trade_id": worst["id"]}, "followups": [{"text": "Show me the losing trades", "why": "Every trade that lost money."}]}


def chart(messages: list[dict]) -> dict:
    m = _latest_result(messages)
    if m is None:
        return {"headline": "There is no chart yet: no question has been answered in this conversation.", "happened": [], "limits": []}
    c = m["content"]
    a = c.get("answer") or {}
    k = a.get("kind") or c.get("view")
    viz = (a.get("viz") or {}).get("kind")
    out = []
    if k == "agreement" or viz == "matrix":
        out.append("The grid has four boxes, one for each combination: the first candle went up or down, and the rest of the session went up or down. "
                   "The two boxes where both went the same way are the days the candle was right.")
        out.append("If the candle told you nothing, the right-hand split would look the same in both rows. The answer above compares the share of right days with that no-information baseline, not with 50%.")
    elif k in ("frequency", "frequency_set") or viz in ("counts", "variants"):
        out.append("Each bar splits the days that matched your rules into the ones where the outcome happened and the ones where it did not. The thin mark is the same share on all days, for comparison.")
        out.append("Click a part of a bar to see those exact days.")
    elif k == "movement" or viz == "paths":
        out.append("Every faint line is one day, starting at 0% when the observation begins. The dark line is the median day and the shaded bands hold the middle half and the middle 80% of days.")
        out.append("Where the band is wide, days disagreed; a band centred on 0% means no typical direction.")
    elif k == "timing" or viz == "timing":
        out.append("Each bar is one part of the trading day. Its height is how often that part held the day's extreme (or how big the moves in it were). If every part were equal, the bars would all sit on the dashed line.")
    elif k == "breakdown" or viz == "groups":
        out.append("Each bar is the average move for that group, and the whisker is its 95% range: if two whiskers overlap a lot, the difference between those groups is within what chance produces.")
    elif k == "strategy" or viz == "account":
        out.append("The line is the account: it starts at zero and adds each trade's result after fees, in order. The dots underneath are the individual trades, above the line for gains and below for losses.")
        out.append("A rising line made of a few big jumps depends on those few trades; the answer says what happens without the three best.")
    else:
        out.append("The chart shows the observations behind the answer above; click any part of it to see the days or trades it is made of.")
    return {"headline": "How to read this chart.", "happened": out, "limits": []}


def unreliable(messages: list[dict], trials: int) -> dict:
    m = _latest_result(messages)
    if m is None:
        return {"headline": "There is no result in this conversation yet.", "happened": [], "limits": []}
    a = (m["content"] or {}).get("answer") or {}
    lines = list(a.get("limits") or [])
    nums = {x.get("label"): x for x in a.get("numbers") or []}
    if trials > 1:
        lines.append(f"{trials} variations have been tried in this conversation. When many versions are tried, some look good by luck alone, so the bar for calling one real goes up.")
    if not lines:
        lines.append("Nothing in this result was marked unreliable. Ask \"is this actually good or just luck?\" for the full check.")
    return {"headline": "What limits how much this result can be trusted, from its own numbers:", "happened": lines[:6], "limits": [],
            "followups": [{"text": "Is this actually a good strategy, or did it just get lucky?", "why": "The full set of checks, in plain words."}]}


def expiration(messages: list[dict]) -> dict:
    ent = _strategy_entry(messages)
    lines = []
    if ent is not None:
        m, run, trades = ent
        legs = [(_leg(t), t) for t in trades if _leg(t)]
        if legs:
            avg_p = sum(float(l["entry_price"]) for l, _ in legs) / len(legs)
            mult = float(legs[0][0].get("multiplier") or 100)
            dte = legs[0][1].get("entry_dte")
            th = [abs(float(l["entry_theta"])) for l, _ in legs if l.get("entry_theta") is not None]
            lines.append(f"In this backtest the {'same-day' if dte == 0 else f'{dte}-day'} contracts cost ${avg_p:.2f} per share on average (${avg_p * mult:,.0f} per contract).")
            if th:
                lines.append(f"Their time value was shrinking by about ${sum(th) / len(th):.2f} per share per day at entry (theta). Closer expirations lose time value faster, so the move has to come sooner.")
    lines.append("A later expiration costs more, because it buys more time for the move. It reacts less to each small move in the underlying, and loses its time value more slowly; a same-day contract is cheap, reacts sharply, and is worth only its intrinsic value at the close.")
    return {"headline": "What changing the expiration does, using this backtest's contracts:", "happened": lines, "limits": [],
            "followups": [{"text": "Compare with 7DTE calls", "why": "The same rule with contracts that expire a week later."}]}


def difference_luck(messages: list[dict]) -> dict:
    """"Could the difference just be luck?" about the latest comparison, from its own test."""
    m = next((x for x in reversed(messages) if x["kind"] == "result" and (x["content"] or {}).get("view") in ("compare", "sweep")), None)
    if m is None:
        return {"headline": "There is no comparison in this conversation yet.", "happened": ["Change something (\"exit an hour later\"), then ask to compare both; then ask again."], "limits": []}
    a = (m["content"] or {}).get("answer") or {}
    det = {d.get("label"): d.get("value") for d in a.get("details") or []}
    p = next((v for k, v in det.items() if "p-value" in (k or "")), None)
    try:
        pv = float(p) if p not in (None, "n/a") else None
    except ValueError:
        pv = None
    if pv is None:
        head = "There is no test of this difference to go on."
    elif pv > 0.05:
        head = "It could easily be luck: a difference this size between two versions shows up often even when neither is better."
    else:
        head = "It is larger than single-trade variation usually produces, but both versions were tested on the same days, and choosing between versions after seeing them makes any winner look better than it is."
    lines = [x for x in a.get("happened") or []]
    if pv is not None:
        lines.append(f"In numbers: if the two versions had the same average trade, a gap at least this large would appear about {pv * 100:.0f} times in 100 samples like this one. That is not the chance that one version is better.")
    return {"headline": head, "happened": lines, "limits": a.get("limits") or [], "followups": [{"text": "Is this actually a good strategy, or did it just get lucky?", "why": "The full checks on the current version."}]}


def how_much(messages: list[dict]) -> dict:
    ent = _strategy_entry(messages)
    if ent is None:
        return {"headline": "There is no finished backtest in this conversation yet.", "happened": ["Describe a strategy and it runs on the stored option prices; then ask again."], "limits": []}
    m, run, trades = ent
    a = (m["content"] or {}).get("answer") or {}
    return {"headline": a.get("headline") or (m["content"].get("text") or "").split(". ")[0], "happened": a.get("happened") or [], "limits": a.get("limits") or [],
            "numbers": a.get("numbers") or []}
