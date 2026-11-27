"""Research intents: questions that need a plan rather than a single run.

    "Find out whether SPY tends to reverse after large down days."
    "Compare opening calls, puts, and waiting 30 minutes after a 2% overnight gap."
    "Does the first 15-minute candle predict the rest of the trading session?"
    "Investigate mean reversion after unusually large SPY declines."
    "Show me whether this strategy performs differently on Mondays."
    "Test whether this strategy has any statistically significant edge."

`detect` turns such a message into an action for the agent: an investigation (several
related configurations plus the checks to run on them), a discovery grid, or a request
about the results that already exist. Every plan is made of ordinary Drafts, so each
variant is compiled, covered and executed by exactly the same deterministic path as a
single strategy. The conclusions are assembled from the engine's numbers by
`investigation_report`; no language model writes a result.
"""
from __future__ import annotations

import re

from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, DraftCondition, DraftLevel, DraftStrike, DraftStudy
from strata.ai.nl_stateful import enrich, read_costs, read_mode
from strata.research import stats as ST

N = NL.N


def _study(symbol: str, title: str, conds: list[DraftCondition], entry: str | None = None, target: float | None = 0.0, measure: str = "close", direction: str = "up", **kw) -> Draft:
    st = DraftStudy(target=DraftLevel(kind="pct", value=target) if target is not None else None, measure=measure, direction=direction, **kw)
    return Draft(kind="study", symbol=symbol, symbol_stated=True, title=title, conditions=conds, entry_time=entry, study=st)


def _sym(text: str, known: list[str], default: str) -> str:
    return NL.find_symbol(text, [k for k in known if k not in NL.NOT_UNDERLYING]) or default


# ----------------------------------------------------------------------------- the first candle and the day
CANDLE = r"(?:candle(?:stick)?s?|bars?)"
FIRST = r"(?:first|opening|open|initial|1st)"
PREDICTS = (r"predict\w*|forecast\w*|foreshadow\w*|foretell\w*|indicat\w*|determin\w*|dictat\w*|sets?\s+the\s+tone|carr(?:y|ies)\s+over|agree\w*\s+with|match\w*|"
            r"(?:tells?|says?)\s+(?:us|you|me)?\s*(?:about|how|where|what|which|whether|anything|something)|(?:same|opposite)\s+(?:direction|way|colou?r)")
CLOCK = r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?|[ap]\b)?,?\s*(?:e[sd]?t|eastern|p[sd]?t|pacific|c[sd]?t|central|m[sd]?t|mountain|new york)?(?:\s*time)?"
# candle lengths measured when the question names none: the three a "first candle" usually means. Four studies each (up and
# down, against the whole session and against the rest of it) is the twelve an investigation runs at most
UNSTATED = (5, 15, 30)


def first_candle(text: str) -> dict | None:
    """ "How often does SPY's first 9:30 candle predict its overall day?" -> {"minutes": [...], "stated": bool, "scope": ...}

    The question is whether the direction of the first candle of the session and the direction of the session agree. It
    has two honest answers and both are given unless the sentence asks for one: the WHOLE session, open to close, which
    contains the candle and so agrees with it partly by construction; and the REST of the session, from the candle's close
    on, which is the only part anybody could act on after seeing the candle."""
    low = " ".join(text.lower().split())
    if re.search(r"\b(?:buy|sell|short|long|enter|exit)\b", low):
        return None  # a strategy that mentions the first candle is read by the strategy readers
    bare = re.sub(CLOCK, " ", low)  # "the first 9:30ET candle": the clock names the candle, it is not its length
    num = r"(\d{1,3}|one|two|three|five|ten|fifteen|twenty|thirty|forty[- ]five|sixty)"
    m = (re.search(rf"\b{FIRST}\s+(?:{num}[- ]?(?:minutes?|mins?|m)\b|(half[- ]an?[- ]hour|half[- ]hour|hour))(?:\s+(?:of\s+(?:the\s+)?(?:trading\s+)?(?:day|session)|{CANDLE}))?", bare)
         or re.search(rf"\b{num}[- ]?(?:minutes?|mins?|m)\s+(?:{FIRST}\s+)?(?:opening\s+)?{CANDLE}\b", bare) if re.search(rf"\b{FIRST}\b", bare) else None)
    named = re.search(rf"\b{FIRST}\b[^.?!]{{0,24}}?\b{CANDLE}\b|\b{CANDLE}\b[^.?!]{{0,12}}?\bat\s+the\s+open\b", bare)
    if not (m or named) or not re.search(PREDICTS, bare):
        return None
    mins: list[int] = []
    if m:
        words = {"forty-five": 45, "forty five": 45}
        tok = next((g for g in m.groups() if g), "")
        v = 30 if "half" in tok else 60 if tok == "hour" else words.get(tok) or NL._num(tok)
        if v and 1 <= int(v) <= 180:
            mins = [int(v)]
    rest = bool(re.search(r"\brest of\b|\bremainder\b|\bremaining\b|\bafter (?:it|that|the (?:first|opening))\b|\bfrom (?:there|then) on\b|\bsubsequent\b", bare))
    return {"minutes": mins or list(UNSTATED), "stated": bool(mins), "scope": "rest" if rest else "both"}


def first_candle_plan(text: str, sym: str, fc: dict) -> dict:
    hhmm = lambda mm: f"{(570 + mm) // 60:02d}:{(570 + mm) % 60:02d}"
    variants, pairs = [], []
    for mm in fc["minutes"]:
        t = hhmm(mm)
        name = f"first {mm} min"
        up_c = DraftCondition(expr="from_open_pct > 0", text=f"the first {mm} minute{'s' if mm != 1 else ''} closed above the open")
        dn_c = DraftCondition(expr="from_open_pct < 0", text=f"the first {mm} minute{'s' if mm != 1 else ''} closed below the open")
        for scope in (("day", "rest") if fc["scope"] == "both" else ("rest",)):
            ref = "open" if scope == "day" else "entry"
            what = ("session closes above its open", "session closes below its open") if scope == "day" else (f"close is above the {t} price", f"close is below the {t} price")
            lu, ld = f"{name} up: {what[0]}", f"{name} down: {what[1]}"
            variants.append({"label": lu, "draft": _study(sym, f"{sym}: {lu}", [up_c], t, 0.0, "close", "up", reference=ref)})
            variants.append({"label": ld, "draft": _study(sym, f"{sym}: {ld}", [dn_c], t, 0.0, "close", "down", reference=ref)})
            pairs.append({"minutes": mm, "scope": scope, "from": t, "up": lu, "down": ld})
    one = len(fc["minutes"]) == 1
    which = f"the first {fc['minutes'][0]} minutes" if one else "the first candle"
    notes = [] if fc["stated"] else ["The length of the first candle was not stated, so " + ", ".join(f"{x}-" for x in fc["minutes"][:-1]) + f" and {fc['minutes'][-1]}-minute candles are all measured rather than one being assumed."]
    if fc["scope"] == "both":
        notes.append("\"The day\" is measured twice. Open to close is the day as a whole, and the first candle is part of it: some agreement is built in. From the candle's close to the session's close is what was still unknown when the candle ended.")
    notes.append("A candle's direction is its close against the session's open. Sessions whose candle closed exactly at the open have no direction and are left out; so are sessions without intraday bars.")
    return {"question": text.strip(), "symbol": sym, "kind": "first_candle", "pairs": pairs,
            "hypothesis": f"The direction of {which} of the session and the direction of the session agree more often than unrelated directions would.",
            "method": (f"On every {sym} session with intraday bars: the direction of the first " + (f"{fc['minutes'][0]} minutes" if one else "/".join(str(x) for x in fc["minutes"]) + " minutes") + " (close against the open) is compared with "
                       + ("the direction of the whole session (open to close) and with the direction of the rest of it (the candle's close to the session's close)." if fc["scope"] == "both" else "the direction of the rest of the session (the candle's close to the session's close).")
                       + " Up and down candles are counted separately, each against all sessions."),
            "measure": "this happened", "variants": variants, "checks": ["baseline", "significance", "out_of_sample"], "interpretation_notes": notes}


# ----------------------------------------------------------------------------- fresh research questions
def detect_fresh(text: str, known: list[str], default_symbol: str = "SPY") -> dict | None:
    low = text.lower().strip()
    sym = _sym(text, known, default_symbol)

    # ---- does the first candle of the session predict the day (any phrasing; the older pattern below needs "first N minutes ... predict")
    fc = first_candle(text)
    if fc:
        return {"action": "investigate", "plan": first_candle_plan(text, sym, fc)}

    # ---- exploratory discovery of a hypothesis over a parameter grid
    if re.match(r"(investigate|explore|research|look for|search for|discover|find (?:me )?(?:an? )?(?:edge|pattern|setup)s?|screen for|mine)\b", low) or re.search(r"\bacross (?:multiple|different|reasonable|a range of) (?:parameters|settings|thresholds)\b", low):
        spec = discovery_spec(text, sym)
        if spec:
            return {"action": "discover", "spec": spec}

    # ---- does X tend to reverse / bounce / continue after large moves
    m = re.search(r"(revers\w*|bounc\w*|rebound\w*|recover\w*|mean[- ]revert\w*|snap\w* back|continu\w*|follow[- ]through|keep (?:falling|rising|going))\s+(?:after|following|the day after|on the day after)\s+(?:a\s+|an\s+)?"
                  r"(large|big|sharp|significant|major|heavy|steep|strong)?\s*(down|red|losing|up|green|winning|declin\w*|sell-?off|drop|fall|rally|gain)", low)
    if m and re.search(r"find out|whether|\bdoes\b|\bdo\b|tend|usually|typically|is there|test (?:if|whether)", low):
        down = bool(re.search(r"down|red|losing|declin|sell|drop|fall", m.group(3)))
        cont = bool(re.search(r"continu|follow|keep", m.group(1)))
        up_outcome = (down != cont)  # reversal after a down day = up; continuation after a down day = down
        variants = []
        for x in (1, 2, 3, 4):
            cond = DraftCondition(expr=f"previous_session.return_pct {'<=' if down else '>='} {'-' if down else ''}{x}", text=f"previous session {'fell' if down else 'rose'} {x}% or more (close-to-close)")
            variants.append({"label": f"after a {x}%+ {'down' if down else 'up'} day", "draft": _study(sym, f"{sym} after a {x}%+ {'down' if down else 'up'} day", [cond], "open", 0.0, "close", "up" if up_outcome else "down")})
        what = "reverse" if not cont else "continue"
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym,
            "hypothesis": f"After a large {'down' if down else 'up'} day, {sym} tends to {what}: the next session closes {'above' if up_outcome else 'below'} its open more often than on other days.",
            "method": f"For previous-day {'declines' if down else 'gains'} of at least 1%, 2%, 3% and 4%: observe the next session from its open and record whether it closes {'higher' if up_outcome else 'lower'} than it opened. "
                      "Each is compared with every other session (the baseline).",
            "measure": f"next session closes {'above' if up_outcome else 'below'} its open", "variants": variants, "checks": ["baseline", "significance", "direction", "out_of_sample", "by_year"],
            "interpretation_notes": ["\"Large\" is not a number, so four thresholds are tested rather than one being assumed.", "\"Reverse\" is measured open-to-close on the following session; the overnight gap is not part of it."]}}

    # ---- does the first N-minute candle predict the rest of the session
    m = re.search(r"first (\w+)[- ]?(?:minute|min)s?(?: candle| bar)?[^.?]{0,40}?(predict|forecast|foreshadow|tell|signal|indicat|determin)", low)
    if m and NL._num(m.group(1)):
        mins = int(NL._num(m.group(1)))
        t = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
        up = _study(sym, f"{sym}: first {mins} minutes up", [DraftCondition(expr="from_open_pct > 0", text=f"the first {mins} minutes close above the open")], t, 0.0, "close", "up")
        dn = _study(sym, f"{sym}: first {mins} minutes down", [DraftCondition(expr="from_open_pct < 0", text=f"the first {mins} minutes close below the open")], t, 0.0, "close", "down")
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym, "hypothesis": f"The direction of the first {mins} minutes carries over: the rest of the session moves the same way more often than not.",
            "method": f"Observe each session from {t} ET. When the first {mins} minutes were up, record whether the close is above the {t} price; when they were down, whether it is below. Both are compared with all sessions.",
            "measure": f"close is beyond the {t} price in the same direction as the first {mins} minutes",
            "variants": [{"label": f"first {mins} min up -> rest of session up", "draft": up}, {"label": f"first {mins} min down -> rest of session down", "draft": dn}],
            "checks": ["baseline", "significance", "out_of_sample"],
            "interpretation_notes": [f"Needs intraday bars that can observe {t} ET, so only the sessions with 1-, 5- or 15-minute bars are used.", "\"Rest of the session\" is measured from the end of the first candle to the close."]}}

    # ---- compare option structures and entry timing under one condition
    if re.search(r"\bcompare\b", low) and re.search(r"\bcalls?\b", low) and re.search(r"\bputs?\b", low) and re.search(r"waiting (\d+)\s*(?:minute|min)|(\d+)\s*(?:minute|min)s? (?:later|after the open)", low):
        w = re.search(r"waiting (\d+)\s*(?:minute|min)|(\d+)\s*(?:minute|min)s? (?:later|after the open)", low)
        mins = int(w.group(1) or w.group(2))
        later = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
        base = enrich(re.sub(r"\bcompare\b|\bputs?,?\b|,?\s*and waiting \d+\s*(?:minute|min)s?", " ", text, flags=re.I), known, default_symbol)
        g = re.search(rf"{N}\s*%\s*(?:overnight\s+|opening\s+)?gap", low)
        conds = [c for c in base.conditions]
        notes = []
        if g and not any("gap_pct" in c.expr for c in conds):
            x = float(g.group(1))
            if re.search(r"gap(?:s|ping|ped)?\s*(?:down|lower)|down gap|gap down", low):
                conds.append(DraftCondition(expr=f"gap_pct <= -{x:g}", text=f"opening gap down {x:g}% or more"))
            elif re.search(r"gap(?:s|ping|ped)?\s*(?:up|higher)|up gap|gap up", low):
                conds.append(DraftCondition(expr=f"gap_pct >= {x:g}", text=f"opening gap up {x:g}% or more"))
            else:
                conds.append(DraftCondition(expr=f"abs(gap_pct) >= {x:g}", text=f"opening gap of {x:g}% or more in either direction"))
                notes.append(f"The gap direction was not stated, so gaps of {x:g}% or more in either direction qualify.")
        variants = []
        for label, structure, entry in (("calls at the open", "long_call", "open"), ("puts at the open", "long_put", "open"), (f"calls after waiting {mins} minutes", "long_call", later), (f"puts after waiting {mins} minutes", "long_put", later)):
            d = base.model_copy(deep=True)
            d.kind, d.structure, d.entry_time, d.entry_time_stated, d.conditions, d.study, d.questions = "options", structure, entry, None, list(conds), None, []
            d.title = f"{sym} {label}"
            if d.dte is None:
                d.dte = 0
            if d.strike is None:
                d.strike = DraftStrike(method="atm")
            variants.append({"label": label, "draft": d})
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym, "hypothesis": "After a large overnight gap, the choice between calls and puts, and between entering at the open or waiting, changes the outcome.",
            "method": f"Four positions on the same qualifying sessions: at-the-money calls and puts bought at the open, and the same bought {mins} minutes later. Each is closed before the end of the session.",
            "measure": "profit and loss per trade on observed option prices", "variants": variants, "checks": ["sample", "costs", "paired"],
            "interpretation_notes": notes + ["No expiration was stated: the same-day (0DTE) contract is used because the trade is intraday.", "No exit was stated: each position is closed at 15:55 ET."]}}
    return None


def discovery_spec(text: str, sym: str) -> dict | None:
    """A parameter grid for an exploratory hypothesis. Returns None if the hypothesis is not one of the supported families."""
    from strata.research.discovery import build_grid
    low = text.lower()
    declines = bool(re.search(r"declin|drop|sell-?off|down day|loss|fall|crash|red day|capitulat", low))
    gains = bool(re.search(r"rall|gain|up day|green day|surge|advance|melt-?up", low))
    reversion = bool(re.search(r"mean[- ]revers|revers|bounce|rebound|recover|snap ?back|oversold|fade", low))
    momentum = bool(re.search(r"momentum|continu|follow[- ]through|trend|breakout|persist", low))
    gap = bool(re.search(r"\bgap", low))
    if not (declines or gains or gap):
        return None
    if gap:
        sign = -1 if re.search(r"gap(?:s|ping|ped)?\s*(?:down|lower)|down gap|gap down", low) else 1
        fade = reversion or not momentum
        up = (sign < 0) == fade
        dims = [{"name": "gap", "label": f"Opening gap {'down' if sign < 0 else 'up'} of at least", "values": [0.5, 1.0, 1.5, 2.0], "unit": "%"},
                {"name": "entry", "label": "Observation starts", "values": ["open", "09:45", "10:00", "10:30"]},
                {"name": "target", "label": f"Target ({'up' if up else 'down'} from the entry price)", "values": [0.5, 1.0], "unit": "%"},
                {"name": "regime", "label": "Volatility regime", "values": ["any", "vix20"], "labels": {"any": "any", "vix20": "VIX at least 20"}}]

        def mk(p):
            kids = [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": f"gap_pct {'<=' if sign < 0 else '>='} {'-' if sign < 0 else ''}{p['gap']:g}", "label": f"gap {'down' if sign < 0 else 'up'} {p['gap']:g}%+"}}]
            if p["regime"] == "vix20":
                kids.append({"id": "c1", "kind": "cond", "type": "expr", "params": {"expr": "vix_prev_close >= 20", "label": "VIX closed at 20 or higher the day before"}})
            return ({"symbol": sym, "condition": {"id": "root", "kind": "group", "op": "and", "children": kids}, "entry_time": p["entry"], "reference": "entry",
                     "target": {"kind": "pct", "value": p["target"] if up else -p["target"]}, "horizon": {"kind": "session"}},
                    f"gap {'down' if sign < 0 else 'up'} {p['gap']:g}%+, from {p['entry']}, {'+' if up else '-'}{p['target']:g}% target, {'VIX >= 20' if p['regime'] == 'vix20' else 'any volatility'}")
        hyp = f"After an opening gap {'down' if sign < 0 else 'up'}, {sym} tends to {'fade the gap' if fade else 'keep going'} during the session."
    else:
        down = declines and not (gains and not declines)
        fade = reversion or not momentum
        up = down == fade
        dims = [{"name": "move", "label": f"Previous-day {'decline' if down else 'gain'} of at least", "values": [1, 2, 3, 4], "unit": "%"},
                {"name": "entry", "label": "Observation starts", "values": ["open", "09:45", "10:00", "10:30"]},
                {"name": "target", "label": f"Target ({'up' if up else 'down'} from the entry price)", "values": [0.5, 1.0], "unit": "%"},
                {"name": "regime", "label": "Volatility regime", "values": ["any", "vix20"], "labels": {"any": "any", "vix20": "VIX at least 20"}}]

        def mk(p):
            kids = [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": f"previous_session.return_pct {'<=' if down else '>='} {'-' if down else ''}{p['move']:g}", "label": f"previous day {'fell' if down else 'rose'} {p['move']:g}%+"}}]
            if p["regime"] == "vix20":
                kids.append({"id": "c1", "kind": "cond", "type": "expr", "params": {"expr": "vix_prev_close >= 20", "label": "VIX closed at 20 or higher the day before"}})
            return ({"symbol": sym, "condition": {"id": "root", "kind": "group", "op": "and", "children": kids}, "entry_time": p["entry"], "reference": "entry",
                     "target": {"kind": "pct", "value": p["target"] if up else -p["target"]}, "horizon": {"kind": "session"}},
                    f"{'down' if down else 'up'} {p['move']:g}%+, from {p['entry']}, {'+' if up else '-'}{p['target']:g}% target, {'VIX >= 20' if p['regime'] == 'vix20' else 'any volatility'}")
        hyp = f"After an unusually large {'decline' if down else 'gain'}, {sym} tends to {'reverse' if fade else 'continue'} the next session."
    return {"symbol": sym, "question": text.strip(), "hypothesis": hyp, "direction": "up" if up else "down", "dims": dims, "configs": build_grid(sym, dims, mk),
            "not_explored": ["Calls versus puts and days to expiration are not part of this stage: it measures the underlying's behaviour. Test an option position on any configuration afterwards - observed option prices decide what can be run."]}


# ----------------------------------------------------------------------------- follow-ups about existing work
def detect_followup(text: str, cur: Draft, known: list[str]) -> dict | None:
    low = text.lower().strip()
    if re.search(r"\b(export|download|save|share|print)\b[^.]{0,40}\b(report|results?|research|trade log|trades|configuration|config)\b|\breport\b.*\b(pdf|html|file)\b", low):
        return {"action": "export"}
    # the three questions a result has to be able to answer about itself: what was filled and from what, what could not be tested, what the rest would cost
    if re.search(r"\b(?:every|each|all|the)\b[^.?]{0,40}\bfills?\b|\bfills?\b[^.?]{0,30}\b(?:source|came from|come from|priced)|where (?:did|do) the (?:prices?|fills?|premiums?) come from|source of (?:each|every|the) (?:fill|price|premium)|"
                 r"\b(?:show|see|list|inspect|what were)\b[^.?]{0,20}\b(?:the |actual )?(?:option )?(?:prices|premiums) (?:used|paid|it used|we used|that were used)\b|\bactual option prices\b", low):
        return {"action": "fills"}
    if re.search(r"(?:which|what|exactly which)\b[^.?]{0,50}\b(?:sessions?|days|dates|signals?)\b[^.?]{0,40}\b(?:could not|couldn'?t|cannot|can'?t|were not|weren'?t|was not|not)\s+(?:be\s+)?(?:tested|priced|evaluated|backtested|filled)|"
                 r"\buntested (?:sessions?|days|dates)|(?:sessions?|days|dates) (?:that )?(?:are|were) (?:missing|untested|not tested)|what (?:is|was) missing", low):
        return {"action": "untested"}
    if re.search(r"(?:estimate|how much|what would it cost|what does it cost|cost)\b[^.?]{0,70}\b(?:acquire|obtain|buy|get|download|fill in|purchase)\b[^.?]{0,50}\b(?:missing|data|observations|sessions|prices)|cost (?:of|to get) (?:the )?missing|"
                 r"what would (?:the )?(?:missing )?data cost|how much (?:would|does) (?:the )?(?:missing )?data cost", low):
        return {"action": "cost"}
    m = re.search(r"\b(show|list|display|see|give me|open|view)\b[^.?]{0,40}\b(sessions?|days|dates|trades|observations|occurrences|instances)\b", low)
    if m and not re.search(r"skipped|not taken|unexecuted|missed signal|why", low):
        w = low[m.start():m.end()]
        f = "loss" if re.search(r"losing|loser|unprofitable|missed|failed|unsuccessful", w) else "win" if re.search(r"winning|winner|profitable|successful|hit", w) else "all"
        return {"action": "show_trades", "filter": f}
    if re.search(r"skipped (?:signal|trade|session|entr)|signals? (?:that|which) (?:were|was|got) (?:skipped|not taken|missed|rejected)|why (?:was|were) .{0,30}(?:skipped|not taken|not executed|missed)|(?:unexecuted|unfilled|missed) (?:signal|order|trade)s?", low):
        return {"action": "skipped"}
    if re.search(r"statistic\w* significan|significan\w* edge|is (?:this|it|that) (?:real|significant|just (?:luck|noise|chance))|real edge|any edge|(?:due to|by) chance|p[- ]?value|robust(?:ness)?\b|hold(?:s)? up\b(?!.*out[- ]of[- ]sample)", low):
        return {"action": "robustness"}
    m = re.search(r"\b(mondays?|tuesdays?|wednesdays?|thursdays?|fridays?)\b", low)
    if (m or re.search(r"day of (?:the )?week|weekday", low)) and re.search(r"differ|perform|better|worse|by (?:day|weekday)|break ?down|depend", low):
        return {"action": "breakdown", "by": "weekday", "focus": NL.DAYNAMES.get(m.group(1).rstrip("s")) if m else None}
    if re.search(r"what time|when (?:during|in) the (?:day|session)|time of day|how (?:long|quickly|fast|soon)|how many minutes", low) and re.search(r"target|hit|reach|touch|happen|get there", low):
        return {"action": "timing"}
    if re.search(r"by year|each year|year by year|per year|over the years", low) and re.search(r"how|show|break|perform|result", low):
        return {"action": "breakdown", "by": "year", "focus": None}
    # ---- an option position on the sessions of the current study ("test ATM 0DTE calls on those days")
    structure = NL.find_structure(low)
    if structure and re.search(r"(?:on|for|during) (?:those|these|the same|such) (?:days|sessions|dates)|same (?:days|sessions|conditions|setup|signal)|(?:those|these) (?:days|sessions|signals)|that setup|when that happens", low):
        d = derive_options(text, cur, known, structure)
        return {"action": "derive", "draft": d, "changed": ["position"], "compare": False}
    return None


def derive_options(text: str, cur: Draft, known: list[str], structure: str) -> Draft:
    """An option strategy on the qualifying sessions of the current draft: its symbol, rules,
    entry time and period carry over; the position comes from the new message."""
    fresh = enrich(text, known, cur.symbol)
    d = cur.model_copy(deep=True)
    d.kind, d.structure, d.study, d.title = "options", structure, None, text.strip()[:90]
    d.dte, d.strike, d.width, d.quantity = fresh.dte, fresh.strike, fresh.width, fresh.quantity
    d.mode, d.provider, d.costs = fresh.mode, fresh.provider, fresh.costs or cur.costs
    d.questions, d.unsupported = fresh.questions, fresh.unsupported
    d.premium_target_pct, d.premium_stop_pct = fresh.premium_target_pct, fresh.premium_stop_pct
    d.underlying_target_pct = fresh.underlying_target_pct
    d.underlying_stop_pct = fresh.underlying_stop_pct
    d.exit_time, d.exit_time_stated, d.hold_sessions, d.hold_to_expiration = fresh.exit_time, fresh.exit_time_stated, fresh.hold_sessions, fresh.hold_to_expiration
    if fresh.entry_time and fresh.entry_time_stated:
        d.entry_time, d.entry_time_stated = fresh.entry_time, fresh.entry_time_stated
    elif cur.kind == "study" and not d.entry_time:
        d.entry_time = "open"
    d.notes = [n for n in cur.notes if "Period interpreted" in n]
    d.notes.append("The qualifying sessions are those of the previous question: its symbol, conditions, entry time and period carry over.")
    st = cur.study
    if cur.kind == "study" and st and st.target and st.target.kind == "pct" and st.measure == "touch" and d.underlying_target_pct is None and d.premium_target_pct is None:
        bull = structure in ("long_call", "short_put", "bull_call_spread", "bull_put_spread")
        if (st.target.value > 0) == bull and st.target.value != 0:
            d.underlying_target_pct = abs(st.target.value)
            d.carried["underlying_target_pct"] = "carried over from the question before: its target becomes this position's exit"
            d.notes.append(f"The study's target carries over as the exit: close when {d.symbol} has moved {abs(st.target.value):g}% in the position's favour from the entry (otherwise at the end-of-day exit). Edit it if that is not intended.")
    return d


def followup_edits(text: str, cur: Draft) -> tuple[Draft, list[str]]:
    """Cost assumptions, pricing mode and stateful rules added to the current draft by a follow-up."""
    from strata.ai.nl_stateful import read
    low = text.lower()
    d = cur.model_copy(deep=True)
    changed: list[str] = []
    c = read_costs(low)
    if c is not None and (cur.costs is None or c.model_dump() != cur.costs.model_dump()):
        merged = (cur.costs.model_dump() if cur.costs else {})
        merged.update({k: v for k, v in c.model_dump().items() if v is not None})
        d.costs = type(c).model_validate(merged)
        changed.append("costs")
    mode, prov = read_mode(low)
    if mode and mode != cur.mode and (cur.kind == "options" or mode == "study"):
        d.mode = mode
        changed.append("mode")
    if prov and prov != cur.provider:
        d.provider = prov
        changed.append("provider")
    r = read(text, cur.symbol)
    if r.adds and re.search(r"\badd", low):
        d.adds = r.adds
        changed.append("adds")
    if r.exits and re.search(r"\b(close|exit|sell|get out|flatten)\b", low):
        d.exit_conditions = d.exit_conditions + [x for x in r.exits if x.expr not in {e.expr for e in d.exit_conditions}]
        changed.append("exit_conditions")
    if r.sequence and re.search(r"\b(wait|only enter|only after|after the|then)\b", low):
        d.sequence = r.sequence
        changed.append("sequence")
    for c_ in r.conditions:
        if c_.expr not in {x.expr for x in d.conditions} and re.search(r"\b(only|unless|but|also|and only|require|add (?:a|the) (?:condition|rule|filter))\b", low):
            d.conditions.append(c_)
            changed.append("conditions")
    if r.unsupported:
        d.unsupported = list(dict.fromkeys(d.unsupported + r.unsupported))
        changed.append("unsupported")
    d.notes = d.notes + [n for n in r.notes if n not in d.notes]
    return d, list(dict.fromkeys(changed))


# ----------------------------------------------------------------------------- reports (deterministic)
def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _agreement_report(plan: dict, rows: list[dict], studies: list[dict], trials: int) -> dict:
    """Two directions, and how often they agree: an "up" study and a "down" study over the same sessions, put together.

    agreement   (sessions where the candle was up and the outcome was up + down and down) / sessions with a direction
    if unrelated  the agreement two unrelated directions would show with the same shares of up and down:
                  share(candle up) x share(outcome up, all sessions) + share(candle down) x share(outcome down, all sessions)
    A pair counts as distinguishable only when BOTH of its studies differ from the other sessions after adjusting for
    every comparison of the plan (their own tests resample whole weeks; the interval shown here does not)."""
    from strata.research.robust import wilson
    by = {r["label"]: r for r in rows}
    res_of = {e["label"]: e["result"] for e in studies}
    table, missing = [], []
    for pr in plan["pairs"]:
        u, d = by.get(pr["up"]), by.get(pr["down"])
        if not u or not d or not (u["n"] + d["n"]):
            missing.append(pr)
            continue
        n, k = u["n"] + d["n"], u["hits"] + d["hits"]
        w = u["n"] / n
        chance = None if u["baseline"] is None or d["baseline"] is None else w * u["baseline"] + (1 - w) * d["baseline"]
        both = all(x["adjusted_p_value"] is not None and x["adjusted_p_value"] < 0.05 and (x["lift"] or 0) > 0 for x in (u, d))
        dates = [s_["date"] for lab in (pr["up"], pr["down"]) for s_ in (res_of[lab].get("sessions") or [])]
        table.append({"minutes": pr["minutes"], "scope": pr["scope"], "from": pr["from"], "sessions": n, "agree": k, "agreement": k / n, "ci95": wilson(k, n), "if_unrelated": chance,
                      "up": {"n": u["n"], "probability": u["probability"], "all_sessions": u["baseline"], "adjusted_p_value": u["adjusted_p_value"], "run_id": u["run_id"]},
                      "down": {"n": d["n"], "probability": d["probability"], "all_sessions": d["baseline"], "adjusted_p_value": d["adjusted_p_value"], "run_id": d["run_id"]},
                      "distinguishable": bool(both), "small": n < 60, "first": min(dates) if dates else None, "last": max(dates) if dates else None})
    sym = plan.get("symbol", "")
    day, rest = [t for t in table if t["scope"] == "day"], [t for t in table if t["scope"] == "rest"]

    def span(ts, key="agreement"):
        lo, hi = pct(min(t[key] for t in ts)), pct(max(t[key] for t in ts))
        return lo if lo == hi else f"{lo} to {hi}"

    def un(ts):
        """What unrelated directions would show: one figure when the candle lengths agree on it to within a third of a point."""
        xs = [t["if_unrelated"] for t in ts if t["if_unrelated"] is not None]
        if not xs:
            return "n/a"
        return ("about " if len(xs) > 1 and max(xs) != min(xs) else "") + pct(sum(xs) / len(xs)) if max(xs) - min(xs) < 0.0034 else f"{pct(min(xs))} to {pct(max(xs))}"
    mins = sorted({t["minutes"] for t in table})
    cndl = f"first {mins[0]} minute{'s' if mins[0] != 1 else ''}" if len(mins) == 1 else "first candle"
    findings, limits = [], []
    for mm in mins:
        parts = []
        for t in [x for x in table if x["minutes"] == mm]:
            where = "the whole session (open to close)" if t["scope"] == "day" else f"the rest of the session ({t['from']} to the close)"
            parts.append(f"matched {where} on {pct(t['agreement'])} of {t['sessions']:,} sessions (95% interval {pct(t['ci95'][0])} to {pct(t['ci95'][1])}; {pct(t['if_unrelated'])} if unrelated)")
        findings.append(f"First {mm} minute{'s' if mm != 1 else ''}: " + "; ".join(parts) + ".")
    if not table:
        seen = max((r.get("evaluated") or 0 for r in rows), default=0)
        if seen:  # there are intraday bars, and no candle with a direction: that is a fact about these sessions, not a lack of data
            return {"kind": "study", "rows": rows, "findings": [f"{seen:,} session{'s' if seen != 1 else ''} with intraday bars could be observed; on none of them did the first candle close above or below the session's open."],
                    "verdict": f"On the {seen:,} {sym} session{'s' if seen != 1 else ''} with intraday bars the first candle closed exactly at the session's open every time, so it has no direction to compare with the day's.",
                    "level": "insufficient", "limitations": ["A candle's direction is its close against the session's open; a candle that closes at the open has none. Nothing was put in its place."],
                    "comparisons": len(rows), "trials": trials, "agreement": []}
        return {"kind": "study", "rows": rows, "findings": ["No session with intraday bars could be measured."], "verdict": "There is no stored intraday history to answer this with.", "level": "insufficient",
                "limitations": ["The first candle can only be observed on sessions that have intraday bars."], "comparisons": len(rows), "trials": trials, "agreement": []}
    r_ok = [t for t in rest if t["distinguishable"]]
    tail = ""
    if rest:
        tail = (f"From the end of the candle to the close it matched on {span(rest)}, against {un(rest)} for unrelated directions: " +
                ("that is no more than chance produces, for every candle length measured." if not r_ok and len(rest) > 1 else "that is no more than chance produces." if not r_ok else
                 f"{len(r_ok)} of the {len(rest)} candle lengths differ from chance in both directions after adjusting for the {len(rows)} comparisons." if len(rest) > 1 else
                 f"that differs from chance in both directions after adjusting for the {len(rows)} comparisons."))
    if day:
        longer = len(day) > 2 and all(a_["agreement"] < b_["agreement"] for a_, b_ in zip(day, day[1:]))  # day is in order of candle length
        verdict = (f"The direction of {sym}'s {cndl} matched the direction of the whole session (open to close) on {span(day)} of sessions" + (", more the longer the candle," if longer else "") +
                   f" against {un(day)} if the two were unrelated. Most of that is the candle itself, which is part of the day it is compared with. " + tail)
    else:
        verdict = f"The direction of {sym}'s {cndl} and the direction of the rest of the session: " + tail[0].lower() + tail[1:]
    level = "insufficient" if all(t["small"] for t in (rest or day)) else "none" if rest and not r_ok else "supported" if rest and len(r_ok) == len(rest) else "mixed"
    first, last = min(t["first"] for t in table if t["first"]), max(t["last"] for t in table if t["last"])
    limits.append(f"Only sessions with intraday bars can be measured: {max(t['sessions'] for t in table):,} of them, {first} to {last}.")
    if day:
        limits.append("A match with the whole session is not a forecast: the candle's own move is inside the open-to-close move it is compared with. Only the rest-of-session figures describe what was still unknown when the candle closed.")
    limits.append("Direction only: a session that closes one cent above its open counts the same as one that closes 2% above it. Nothing here says how far the session went.")
    limits.append("Probabilities describe the underlying's behaviour. They are not option returns and say nothing about trading costs.")
    if missing:
        limits.append(f"{len(missing)} of the planned comparisons could not be run and are not in these figures.")
    return {"kind": "study", "rows": rows, "findings": findings, "verdict": verdict, "level": level, "limitations": limits, "comparisons": len(rows), "trials": trials, "agreement": table}


def investigation_report(plan: dict, entries: list[dict], trials: int) -> dict:
    """Findings of an investigation, computed from the variants' stored results.

    entries: [{label, run, result, mode}] in plan order (only finished runs)."""
    rows, findings, limits = [], [], []
    studies = [e for e in entries if e["run"]["kind"] == "study"]
    if studies:
        blocks = [ST.study_block(e["result"]) for e in studies]
        robs = [ST.study_robustness(e["result"], max(trials, len(studies))) for e in studies]
        qs = ST.bh_adjust([b["p_value"] for b in blocks])
        for e, b, q, rb in zip(studies, blocks, qs, robs):
            oos = rb.get("out_of_sample") or {}
            rows.append({"label": e["label"], "run_id": e["run"]["id"], "n": b["n"], "probability": b["probability"], "ci95": b["ci95"], "baseline": b["baseline_probability"], "lift": b["lift"],
                         "p_value": b["p_value"], "adjusted_p_value": q, "avg_end_return_pct": b["avg_end_return_pct"], "other_avg_end_return_pct": b["other_sessions_avg_end_return_pct"],
                         "end_return_p_value": b["end_return_p_value"], "oos_n": oos.get("n"), "oos_probability": oos.get("probability"), "oos_baseline": oos.get("baseline_probability"),
                         "years_above": rb["years_above_baseline"], "level": rb["level"], "period": e["result"].get("period")})
        for r, b in zip(rows, blocks):
            r["hits"], r["evaluated"] = b["hits"], b["baseline_n"]
        if plan.get("pairs"):
            return _agreement_report(plan, rows, studies, trials)
        sig = [r for r in rows if r["adjusted_p_value"] is not None and r["adjusted_p_value"] < 0.05 and r["n"] >= 30]
        small = [r for r in rows if r["n"] < 30]
        measure = plan.get("measure", "the outcome")
        for r in rows:
            if r["n"] == 0:
                findings.append(f"{r['label']}: no qualifying sessions in the stored history.")
                continue
            findings.append(f"{r['label']}: {measure} on {pct(r['probability'])} of {r['n']} sessions (95% interval {pct(r['ci95'][0])} to {pct(r['ci95'][1])}) versus {pct(r['baseline'])} on other sessions"
                            + (f"; adjusted p = {r['adjusted_p_value']:.3f}" if r["adjusted_p_value"] is not None else "") + ".")
        if sig:
            same_oos = [r for r in sig if r["oos_n"] and r["oos_probability"] is not None and r["oos_baseline"] is not None and (r["oos_probability"] > r["oos_baseline"]) == ((r["lift"] or 0) > 0)]
            verdict = (f"{len(sig)} of {len(rows)} variants differ from the baseline after adjusting for the {len(rows)} comparisons"
                       + (f"; {len(same_oos)} of those point the same way in the most recent 30% of history." if sig else "."))
            level = "supported" if len(same_oos) == len(sig) else "mixed"
        elif len(small) == len(rows):
            verdict, level = "Every variant has fewer than 30 qualifying sessions: there is not enough history to answer this either way.", "insufficient"
        else:
            verdict, level = "None of the variants differs from the baseline by more than chance would produce once the number of comparisons is taken into account.", "none"
        if small:
            limits.append(f"{len(small)} variant(s) have fewer than 30 qualifying sessions; their estimates are fragile.")
        limits.append("Probabilities describe the underlying's behaviour. They are not option returns and say nothing about trading costs.")
        res_mix = {k for e in studies for k in (e["result"]["stats"].get("resolution_mix") or {})}
        if res_mix and res_mix != {"1d"}:
            limits.append("Sessions are measured with the finest bars stored for them (" + ", ".join(sorted(res_mix)) + "); intraday history only exists for recent periods.")
        return {"kind": "study", "rows": rows, "findings": findings, "verdict": verdict, "level": level, "limitations": limits, "comparisons": len(rows), "trials": trials}
    # option backtests
    for e in entries:
        res = e["result"]
        m = res.get("metrics") or {}
        cov = res.get("coverage") or {}
        rows.append({"label": e["label"], "run_id": e["run"]["id"], "mode": e["mode"], "trades": m.get("trades"), "net_pnl": m.get("net_pnl"), "win_rate": m.get("win_rate"), "expectancy": m.get("expectancy"),
                     "max_drawdown": m.get("max_drawdown"), "qualifying": (cov.get("signals") or {}).get("qualifying"), "executable": (cov.get("executable") or {}).get("signals"),
                     "pricing": (res.get("meta") or {}).get("pricing"), "period": [res["config"]["start"], res["config"]["end"]], "cost_breakeven_pct": (res.get("cost_sensitivity") or {}).get("breakeven_slippage_pct")})
    done = [r for r in rows if r["trades"]]
    for r in rows:
        if not r["trades"]:
            findings.append(f"{r['label']}: no trades could be executed on observed option prices.")
        else:
            findings.append(f"{r['label']}: {int(r['trades'])} trade{'s' if r['trades'] != 1 else ''} (of {r['qualifying']} qualifying sessions), net {r['net_pnl']:+,.2f}, average {r['expectancy']:+,.2f} per trade, {pct(r['win_rate'])} winners.")
    n_min = min((r["trades"] for r in done), default=0)
    if not done:
        verdict, level = "No variant could be executed on observed option prices, so there is nothing to compare.", "insufficient"
    elif n_min < 30:
        verdict, level = f"The variants ran on {int(n_min)} to {int(max(r['trades'] for r in done))} trades each. That is too few to say one is better than another: differences of this size arise by chance.", "insufficient"
    else:
        verdict, level = "See the per-variant figures; differences should be read together with their sample sizes and cost sensitivity.", "mixed"
    modes = {r["pricing"] for r in done}
    if len(modes) > 1:
        limits.append("The variants were priced differently (" + ", ".join(sorted(m_ or "?" for m_ in modes)) + "); they are not directly comparable.")
    if any((r["qualifying"] or 0) > (r["executable"] or 0) for r in rows):
        limits.append("Only some qualifying sessions have observed option prices; the rest are listed as skipped for lack of data, not counted as losses or wins.")
    limits.append("Past results on the stored data; not a forecast.")
    return {"kind": "backtest", "rows": rows, "findings": findings, "verdict": verdict, "level": level, "limitations": limits, "comparisons": len(rows), "trials": trials}
