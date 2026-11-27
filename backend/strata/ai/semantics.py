"""What a sentence asks, as one question frame, with every meaningful part of the sentence accounted for.

V7 measured the built-in readers on sentences they had never seen: right about half the time, and on a quarter of them
they answered a different question without saying so. The failures had one shape. A reader would recognise part of a
sentence, build something runnable from that part, and drop the rest: "How often does SPY's first candle predict its
overall day?" became a study with no condition and no outcome, and the share of all sessions that closed higher came back
as the answer. This module is the layer in front of those readers:

  normalise   casing, slang ("yday", "tp", "0dte"), number words ("half a percent"), and spelling within one letter of a
              word it knows, keeping a map from every character back to the sentence as typed
  tag         composable pieces, each with the span of the sentence it came from: an instrument, an event that can be a
              condition ("a 2% down day", "gapped up 1%", "the first 15-minute candle was red", "VIX above 20",
              "3 green days in a row", "on Mondays"), an outcome to count ("closes above its open", "touches +0.5% from the
              open", "fills the gap"), when the measurement starts and ends ("before 11am", "the rest of the day", "the
              next day"), a period, the kind of question ("how often", "does X predict Y", "when during the day",
              "which weekday", "what happens after"), option terms, and the words that make a message an edit
  assemble    one frame per question kind (frequency, agreement, movement, timing, breakdown, strategy), with the
              slots that kind requires
  check       clause coverage: every piece that carries meaning must sit in a slot of the frame, or be reported as not
              used; a frame missing a slot its kind requires is not run, and the reply is one focused question

A strategy ("buy calls when ...") is still read by the strategy readers, which V7 measured to be the strongest part of
the old pipeline; this module checks what they produce against the same tags (`strategy_coverage`).

Nothing here computes a result. A frame compiles to the configurations the engine already runs (`to_route`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from strata.ai import nl_rules as NL

STORED_DEFAULT = ("SPY", "QQQ", "IWM")

# ----------------------------------------------------------------------------------------------------------- normalise
WORD_MAP = {
    "yday": "yesterday", "yest": "yesterday", "ystd": "yesterday", "yesterdays": "yesterday", "yesterday's": "yesterday", "ytd": "yesterday",
    "todays": "today", "today's": "today", "tmrw": "tomorrow", "tmr": "tomorrow", "tomorow": "tomorrow",
    "prev": "previous", "prior": "previous", "preceding": "previous",
    "percent": "%", "pct": "%", "perc": "%", "percentage": "%",
    "mins": "min", "minute": "min", "minutes": "min", "minuite": "min", "minues": "min",
    "hrs": "hour", "hr": "hour", "hours": "hour",
    "calls": "call", "puts": "put", "options": "option", "contracts": "contract", "spreads": "spread",
    "tp": "takeprofit", "sl": "stoploss", "eod": "endofday", "w/": "with", "w/o": "without",
    "mondays": "monday", "tuesdays": "tuesday", "wednesdays": "wednesday", "thursdays": "thursday", "fridays": "friday",
    "mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday", "thu": "thursday", "thur": "thursday", "thurs": "thursday", "fri": "friday",
    "candles": "candle", "bars": "bar", "sessions": "session", "days": "day", "weeks": "week", "months": "month", "years": "year",
    "opens": "open", "opened": "open", "opening": "open", "closes": "close", "closed": "close", "closing": "close",
    "gaps": "gap", "gapped": "gap", "gapping": "gap",
    "ends": "end", "ended": "end", "ending": "end", "finishes": "finish", "finished": "finish", "finishing": "finish",
    "drops": "drop", "dropped": "drop", "dropping": "drop", "falls": "fall", "fell": "fall", "falling": "fall", "fallen": "fall", "risen": "rise", "rose": "rise",
    "tanked": "tank", "tanks": "tank", "dumped": "dump", "dumps": "dump", "dumping": "dump", "hw": "how", "hows": "how", "howoften": "how often", "declines": "decline", "declined": "decline", "declining": "decline",
    "rises": "rise", "rose": "rise", "rising": "rise", "rallies": "rally", "rallied": "rally", "rallying": "rally", "gains": "gain", "gained": "gain", "climbs": "climb", "climbed": "climb",
    "recovers": "recover", "recovered": "recover", "recovering": "recover", "bounces": "bounce", "bounced": "bounce", "bouncing": "bounce",
    "reverses": "reverse", "reversed": "reverse", "reversing": "reverse", "reversal": "reverse",
    "touches": "touch", "touched": "touch", "touching": "touch", "hits": "hit", "hitting": "hit", "reaches": "reach", "reached": "reach",
    "trades": "trade", "traded": "trade", "trading": "trade", "predicts": "predict", "predicted": "predict", "predicting": "predict", "predictive": "predict", "predictor": "predict",
    "buying": "buy", "bought": "buy", "buys": "buy", "selling": "sell", "sold": "sell", "sells": "sell",
    "entering": "enter", "entered": "enter", "enters": "enter", "exiting": "exit", "exited": "exit", "exits": "exit",
    "fills": "fill", "filled": "fill", "filling": "fill", "reclaims": "reclaim", "reclaimed": "reclaim",
    "spy's": "spy", "qqq's": "qqq", "iwm's": "iwm", "its": "its", "it's": "it is", "whats": "what is", "what's": "what is",
    "s&p": "spy", "sp500": "spy", "nasdaq": "qqq", "nq": "qqq", "russell": "iwm", "rut": "iwm",
    "greens": "green", "reds": "red", "mornings": "morning", "afternoons": "afternoon",
    "losers": "losing", "winners": "winning", "loser": "losing", "winner": "winning", "loses": "lose", "lost": "lose",
}
NAME_TICKERS = {"tesla": "TSLA", "apple": "AAPL", "nvidia": "NVDA", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
                "facebook": "META", "netflix": "NFLX", "amd": "AMD", "dia": "DIA", "dow": "DIA", "spx": "SPX", "ndx": "NDX", "tsla": "TSLA", "aapl": "AAPL", "nvda": "NVDA",
                "msft": "MSFT", "amzn": "AMZN", "googl": "GOOGL", "nflx": "NFLX"}
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "a couple": 2, "couple": 2, "single": 1}
# words that carry no research meaning on their own (the slots that do are claimed by the recognisers)
FILLER = set("""a an the of in on at for to from by with about into onto over under than then also too as so very really just still even again ever any some
all each every both either one ones this that these those it its it's itself they them their there here what which who whom whose how why when where whether
is are was were be been being am do does did done doing have has had having get gets got getting go goes going went gone make makes made making
i me my we our us you your yours he she his her him please pls plz thanks thank ok okay hey hi hello so um uh lol like
can could would should will shall may might must need needs want wants wanted know think tell show see look looks find check let lets give
usually typically generally normally often tend tends tended historically history historical basically actually really kind sort pretty quite rather
market markets stock stocks etf ticker index price prices day days session sessions time times data number numbers
if and or but nor not no yes after before following whenever once given while until till til since during upon
average mean median typical typically overall whole entire total
eastern pacific central mountain et pt ct mt est edt pst pdt cst cdt ny new york local
what's how's that's there's here's who's i'm i'd i've you're you'd we're they're isn't aren't wasn't weren't doesn't don't didn't won't can't
much many more most less least lot lots bit little
same different other another
work works worked working happen happens happened happening thing things stuff way ways
good bad best worse worst better
""".split())


def _edit1(a: str, b: str) -> bool:
    """a and b differ by one insertion, deletion, substitution or adjacent transposition."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


@dataclass
class Norm:
    text: str        # the sentence as typed
    norm: str        # normalised, lowercase, single-spaced
    omap: list[int]  # norm index -> index in text (len(norm) + 1 entries)

    def span(self, a: int, b: int) -> tuple[int, int]:
        return self.omap[a], self.omap[max(a, b - 1)] + 1 if b > a else self.omap[a]


TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[a-z][a-z'&/]*|[%$+\-<>=/&:]|[^\sa-z\d]", re.I)


def normalise(text: str, vocab: set[str] | None = None) -> Norm:
    """Lowercase tokens with slang and spelling mended, joined by single spaces, with a map back to the sentence."""
    vocab = vocab or set()
    src = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    out: list[str] = []
    omap: list[int] = []
    for m in TOKEN.finditer(src):
        raw = m.group(0)
        w = raw.lower()
        if re.fullmatch(r"\d+,\d{3}", w):
            w = w.replace(",", "")
        elif re.fullmatch(r"\d+,\d{1,2}", w):
            w = w.replace(",", ".")
        if w in WORD_MAP:
            w = WORD_MAP[w]
        elif re.fullmatch(r"[a-z]{5,}", w) and w not in vocab and w not in FILLER:
            # one letter away from a word the recognisers use ("predcit", "recovr", "thrusday"): read as that word
            near = [v for v in vocab if len(v) >= 5 and abs(len(v) - len(w)) <= 1 and _edit1(w, v)]
            if len(near) == 1:
                w = near[0]
        if out:
            out.append(" ")
            omap.append(m.start())
        for k, ch in enumerate(w):
            out.append(ch)
            omap.append(m.start() + min(k, len(raw) - 1))
    norm = "".join(out)
    omap.append(len(src))
    # phrases that name a number
    n = Norm(text, norm, omap)
    for pat, rep in ((r"\bhalf (?:a |of a |of one )?%", "0.5 %"), (r"\ba quarter (?:of a |of one )?%|\bquarter (?:of a )?%", "0.25 %"), (r"\b(?:a|one) %", "1 %"),
                     (r"\bhalf an hour\b|\ba half hour\b|\bhalf hour\b|\bhalf-hour\b", "30 min"), (r"\ban hour\b", "60 min"),
                     (r"\b(\d+(?:\.\d+)?) ?m\b(?! ?%)", r"\1 min"), (r"\b(\d+(?:\.\d+)?) ?h\b", r"\1 hour"), (r"\b(\d+) ?d\b(?!te)", r"\1 day")):
        n = _sub(n, pat, rep)
    for wd, v in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        n = _sub(n, rf"\b{wd}\b(?= (?:- )?(?:%|min|hour|day|dte|delta|green|red|up|down|straight|consecutive|session|week|month|year|contract|call|put|lot|time))", str(v))
    return n


def _sub(n: Norm, pat: str, rep: str) -> Norm:
    """re.sub on the normalised text that keeps the map back to the sentence."""
    out, omap, last = [], [], 0
    for m in re.finditer(pat, n.norm):
        out.append(n.norm[last:m.start()])
        omap.extend(n.omap[last:m.start()])
        r = m.expand(rep)
        out.append(r)
        omap.extend([n.omap[m.start()]] * len(r))
        last = m.end()
    if not out:
        return n
    out.append(n.norm[last:])
    omap.extend(n.omap[last:])
    omap.append(n.omap[-1])
    return Norm(n.text, "".join(out), omap)


# ------------------------------------------------------------------------------------------------------------ mentions
@dataclass
class Mention:
    kind: str          # what was recognised (see the recognisers)
    value: dict        # its meaning
    a: int             # span in the normalised text
    b: int
    text: str          # the words as typed
    role: str = ""     # condition | outcome | start | end | period | symbol | kind | edit | option | explore ...
    used: bool = False


NUM = r"(?P<num>\d+(?:\.\d+)?)"
PCT = r"(?P<sign>[+-] ?)?" + NUM + r" ?%"
ATLEAST = r"(?:(?:at least|or more|\+|or greater|minimum|min) ?)"
MORETHAN = r"(?:more than|over|above|greater than|> ?=?|>=|bigger than|larger than|>)"
UP = r"(?:up|higher|green|rise|rally|gain|climb|bullish|positive|advance|pop)"
DOWN = r"(?:down|lower|red|fall|drop|decline|slide|selloff|sell ?off|tank|dump|bearish|negative|lose|losing|bad)"
SYM_RX = r"(?:spy|qqq|iwm|it|the market|the s&p|the index)"


class Tagger:
    """Finds the pieces of a sentence. Each recogniser claims a span of the normalised text; a span is claimed once."""

    def __init__(self, n: Norm, known: list[str]):
        self.n, self.s = n, n.norm
        self.known = [k.upper() for k in known]
        self.owner: list[Mention | None] = [None] * (len(self.s) + 1)
        self.mentions: list[Mention] = []

    def free(self, a: int, b: int, through: tuple = ()) -> bool:
        """Nothing in [a, b) is claimed, except by mentions of the kinds in `through` (a ticker inside "how does SPY trade")."""
        return all(o is None or o.kind in through for o in self.owner[a:b])

    def add(self, kind: str, value: dict, a: int, b: int, role: str = "", through: tuple = ()) -> Mention | None:
        while b > a and self.s[b - 1] == " ":
            b -= 1
        while a < b and self.s[a] == " ":
            a += 1
        if b <= a or not self.free(a, b, through):
            return None
        oa, ob = self.n.span(a, b)
        m = Mention(kind, value, a, b, self.n.text[oa:ob], role)
        for i in range(a, b):
            if self.owner[i] is None:
                self.owner[i] = m
        self.mentions.append(m)
        return m

    def each(self, pat: str, fn, kind: str, role: str = "", through: tuple = ("symbol",)):
        for m in re.finditer(pat, self.s):
            if not self.free(m.start(), m.end(), through):
                continue
            v = fn(m)
            if v is not None:
                self.add(kind, v, m.start(), m.end(), role, through)


def _f(x) -> float:
    return float(x)


def tag(text: str, known: list[str] | None = None) -> tuple[Norm, list[Mention]]:
    known = list(known or STORED_DEFAULT)
    vocab = set(WORD_MAP.values()) | {"predict", "recover", "bounce", "reverse", "candle", "minute", "thursday", "tuesday", "wednesday", "monday", "friday", "previous", "yesterday",
                                      "percentage", "probability", "likely", "volatility", "breakdown", "weekday", "expiration", "premium", "strangle", "straddle", "condor",
                                      "intraday", "volatile", "session", "afternoon", "morning", "overnight", "consecutive"}
    n = normalise(text, vocab)
    t = Tagger(n, known)
    s = n.norm

    # the first candle of the session ("the first 15-minute candle was red", "a red opening candle", "first 9:30 candle")
    def candle(m, window_aware: bool = False):
        g = m.groupdict()
        mins = g.get("mins")
        mm = int(float(mins)) * (60 if (g.get("unit") or "").startswith("hour") else 1) if mins else (60 if g.get("hour") == "hour" else 30 if g.get("hour") else None)
        d = g.get("d") or g.get("d0")
        if window_aware and not d and re.search(r"\b(?:in|within|during|over|for|through)(?: the)? $", t.s[max(0, m.start() - 12):m.start()]):
            return None  # "in the first 30 minutes": the window that is observed, not a candle that conditions it
        out = {"key": "first_candle", "minutes": mm}
        if d:
            out["direction"] = "up" if re.fullmatch(UP + r"s?", d) else "down"
        return out
    CANDLE = r"(?:candle|bar|candlestick)"
    PRE = r"(?:(?:the|a|an|its) )?"
    DIR = rf"(?P<d0>{UP}|{DOWN})"
    AT930 = r"(?: (?:9 ?: ?30|930)(?: ?(?:et|am|eastern|a))*)?"
    LEN = r"(?:(?P<mins>\d+(?:\.\d+)?) ?-? ?(?P<unit>min|hour)s? )?"
    AFTER = rf"(?: (?:is|was|are|were|closes|close|turns|prints|comes in|end|ends|goes)?(?: )?(?P<d>{UP}|{DOWN}))?"
    t.each(rf"\b{PRE}(?:{DIR} )?(?:first|open|initial|1st){AT930} (?:open )?{LEN}{CANDLE}s?{AFTER}", candle, "event")
    t.each(rf"\b{PRE}(?:{DIR} )?{LEN}(?:open|first) {CANDLE}s?{AFTER}", candle, "event")
    t.each(rf"\b{PRE}(?:{DIR} )?(?:first|open|initial) (?P<mins>\d+(?:\.\d+)?) ?-? ?(?P<unit>min|hour)s?(?: of (?:the )?(?:day|session|trade))?{AFTER}",
           lambda m: candle(m, True), "event")
    t.each(rf"\b{PRE}(?:{DIR} )?(?:first|open|initial) (?P<hour>hour|half ?hour)(?: of (?:the )?(?:day|session|trade))?{AFTER}",
           lambda m: candle(m, True), "event")

    # ---- clock times (the existing reader knows every zone and the one-zone-per-sentence rule)
    msg_tz = next((NL.TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in text.lower()), None)
    for x in NL.find_times(text, msg_tz):
        if x["et"] in ("open", "close", "12:00") and not re.match(r"\d", x["stated"]):
            continue  # "at the open", "noon": the anchors below read these with their context
        stated = x["stated"].split(" (")[0]
        oa = x["pos"]
        # find the normalised span that maps onto this part of the sentence
        a = next((i for i, o in enumerate(n.omap[:-1]) if o >= oa), None)
        if a is None:
            continue
        b = a
        while b < len(s) and n.omap[b] < oa + len(stated):
            b += 1
        t.add("clock", {"et": x["et"], "stated": x["stated"], "assumed_tz": x.get("assumed_tz"), "minutes": x["minutes"]}, a, b)

    # ---- periods
    def period(m):
        a_, b_, note = NL.find_period(m.group(0))
        return None if (a_ is None and b_ is None) else {"start": a_.isoformat() if a_ else None, "end": b_.isoformat() if b_ else None, "note": note}
    t.each(r"\b(?:from|between) (?:19|20)\d\d(?:-\d\d-\d\d)? (?:to|-|through|and|until) (?:19|20)\d\d(?:-\d\d-\d\d)?\b", period, "period", "period")
    t.each(r"\b(?:since|after|from|starting in|starting) (?:19|20)\d\d(?:-\d\d-\d\d)?\b(?! ?%)", period, "period", "period")
    t.each(r"\b(?:in|during|for|throughout|over) (?:the year )?(?:19|20)\d\d\b(?! ?%)", period, "period", "period")
    t.each(r"\b(?:until|before|through) (?:19|20)\d\d\b(?! ?%)", period, "period", "period")
    t.each(r"\b(?:during |in )?(?:the )?(?:covid(?: crash)?|2022 bear market)\b", period, "period", "period")
    t.each(r"\b(?:in |over |for )?(?:the )?(?:last|past) (\d+) (year|month)\b", period, "period", "period")

    # ---- instruments
    for sym in sorted(set(known) | {"SPY", "QQQ", "IWM"}, key=len, reverse=True):
        t.each(rf"\b{re.escape(sym.lower())}\b", lambda m, sym=sym: {"symbol": sym, "stored": sym in known}, "symbol", "symbol")
    t.each(r"\b(" + "|".join(NAME_TICKERS) + r")\b", lambda m: {"symbol": NAME_TICKERS[m.group(1)], "stored": NAME_TICKERS[m.group(1)] in known}, "symbol", "symbol")

    # ---- the kind of question (claimed so that its words are not read as anything else)
    for kind, pat in (
        ("agreement", r"\b(?:predict|(?:a )?(?:good |bad |reliable |useful )?(?:sign|indicator|signal|tell|clue|hint) (?:of|for|about)|tell(?: you| us| me)? (?:anything |something |much )?about|right about|wrong about|say about|foreshadow|match|matches|agree with|same (?:way|direction) as|carry over to|follow through)\b"),
        ("timing", r"\bwhen (?:is|does|do|are) (?:\w+ ){0,3}(?:most volatile|least volatile|calmest|quietest|most active|move the most|moves the most|make (?:its|the) (?:biggest )?moves?)\b|\bwhen during the day\b|\bwhat time(?: of day)?\b|\bwhich (?:half ?hour|hour|time of day|part of the day|30 min|15 min)\b|\btime of day\b|\bintraday (?:volatility )?profile\b|\bvolatility profile\b|\b(?:15|30|60) min buckets?\b"),
        ("breakdown", r"\b(?:good|bad|strong|weak|best|worst|better|worse|green|red) (?:day|days) (?:for|to (?:buy|trade|own|hold))\b|\b(?:is|are) (?:monday|tuesday|wednesday|thursday|friday)s? (?:usually |generally |typically )?(?:good|bad|strong|weak|green|red|up|down|positive|negative)\b|\b(?:which|what) (?:day of the week|weekday|days of the week|month|months)\b|\bby (?:weekday|day of the week|month(?: of the year)?)\b|\bbreak ?down\b|\bbucket\b|\bper (?:weekday|month|decile)\b|\bdeciles?\b|\bwhich weekdays?\b"),
        ("movement", r"\bwhat (?:usually |typically |normally |generally )?(?:happens|happened)\b|\bwhat does \w+ (?:usually |typically |normally |generally )?do\b|\bhow does \w+ (?:usually )?(?:trade|move|behave|do)\b|\bshow me how\b|\bhow much does \w+ (?:usually |typically )?move\b|\btypical (?:move|range|day)\b|\baverage (?:move|range)\b|\bwhere does \w+ go\b|\bhow (?:far|much) does it (?:go|move)\b"),
        ("frequency", r"\b(?:does|do|did|is|are|will|would) (?:\w+ ){0,3}(?:usually|typically|tend to|tends to|generally|normally|often|ever|historically|reliably)\b|^\s*(?:does|do|did) (?:\w+ ){0,2}(?=(?:recover|bounce|rebound|reverse|fill|rise|fall|drop|rally|close|finish|end|gap|keep)\b)|\bhow (?:often|frequently|many times|likely|common)\b|\bis it (?:common|likely|usual)\b|\bwhat (?:is )?(?:the )?(?:%|percent|share|fraction|proportion|odds|probability|chances?|likelihood|hit rate|win rate)\b(?: of the time)?|\b(?:odds|probability|chances?|likelihood|hit rate) (?:of|that|for)?\b|\bp ?\(|\bhow (?:much )?of the time\b|\bwhat % of\b|\b% of the time\b|\bwin rate\b|\bhit rate\b"),
        ("strategy", r"\bbacktest\b|\bif i (?:buy|sell)\b|\bwhat if i (?:buy|sell)\b|\bwould i make money\b|\bdo i make money\b|\bprofitable\b|\bmake money\b"),
        ("explain", r"\bwhy\b|\bexplain\b|\bwhat does (?:this|that|it) mean\b|\bhow come\b"),
        ("reliability", r"\bjust (?:get )?lucky\b|\bluck\b|\bstatistically significant\b|\bsignificant\b|\bis (?:this|it|that) (?:real|reliable|robust)\b|\bgood strategy\b"),
        ("compare", r"\bcompare\b|\bversus\b|\bvs\.?\b|\bcompared (?:to|with)\b|\bside by side\b"),
        ("forecast", r"\b(?:will|is going to|gonna) \w+ (?:go up|go down|rise|fall|drop|rally|crash|close (?:up|down|green|red))\b(?:.*\b(?:tomorrow|today|next week|this week)\b)?"),
    ):
        t.each(pat, lambda m, kind=kind: {"kind": kind}, "kind", "kind")

    # ---- events that are known before or during the session (conditions, and sometimes outcomes)
    def pct(m, neg: bool):
        v = _f(m.group("num"))
        return -v if neg else v

    # a gap at the open
    t.each(rf"\b(?:when |if |after |on days (?:when |that )?)?(?:it |{SYM_RX} )?(?:gap|open) (?:(?P<d>{UP}|{DOWN}|higher|lower|above|below) )?(?:of |by )?(?:{ATLEAST}|{MORETHAN} |more than |at least |over )?(?P<num>\d+(?:\.\d+)?) ?%(?: ?\+| or more| or greater| plus)?(?: (?P<d2>higher|lower|up|down|above|below))?(?: (?:than |from |vs )?(?:the )?(?:previous|yesterday) ?(?:close|'s close|day's close)?)?",
           lambda m: {"key": "gap", "op": ">=" if (m.group("d") or m.group("d2") or "up") in ("up", "higher", "above", "green", "rise", "rally", "gain", "climb", "pop", "bullish", "positive", "advance") else "<=",
                      "value": pct(m, (m.group("d") or m.group("d2") or "up") not in ("up", "higher", "above", "green", "rise", "rally", "gain", "climb", "pop", "bullish", "positive", "advance"))}, "event")
    t.each(rf"\bgap (?P<d>ups|downs)(?: of| by| over| above| more than| bigger than| larger than| greater than| at least| >=?)? ?(?P<num>\d+(?:\.\d+)?) ?%",
           lambda m: {"key": "gap", "op": ">=" if m.group("d") == "ups" else "<=", "value": pct(m, m.group("d") == "downs")}, "event")
    t.each(rf"\b(?:a |an )?(?P<num>\d+(?:\.\d+)?) ?%\+? (?:overnight |opening )?gap (?P<d>up|down|higher|lower)",
           lambda m: {"key": "gap", "op": ">=" if m.group("d") in ("up", "higher") else "<=", "value": pct(m, m.group("d") in ("down", "lower"))}, "event")
    t.each(rf"\b(?:a |an )?(?P<num>\d+(?:\.\d+)?) ?%\+? (?:overnight |opening )?gap (?P<d>{DOWN}|{UP})?",
           lambda m: None, "event")
    t.each(r"\b(?:when |if |after |on days (?:when |that )?)?(?:it |spy |qqq |iwm )?gap (?P<d>up|down|higher|lower)\b(?! ?\d)",
           lambda m: {"key": "gap", "op": ">" if m.group("d") in ("up", "higher") else "<", "value": 0.0, "size": "unstated"}, "event")
    t.each(r"\b(?:a |an )?(?:large |big |huge |massive )?(?:overnight )?gap (?P<d>up|down)s?\b", lambda m: {"key": "gap", "op": ">" if m.group("d") == "up" else "<", "value": 0.0, "size": "unstated"}, "event")
    t.each(r"\bopens? (?:up |down )?(?:a lot|big|sharply|strongly) (?P<d>higher|lower|up|down)?|\bopens? (?P<d2>up|down) (?:a lot|big|sharply|strongly)",
           lambda m: {"key": "gap", "op": "<" if (m.group("d") or m.group("d2") or "") in ("lower", "down") else ">", "value": 0.0, "size": "vague"}, "event")
    t.each(rf"\bopens? (?:more than |at least |over )?(?P<num>\d+(?:\.\d+)?) ?% (?P<d>higher|lower|below|above)(?: (?:it|yesterday|the previous))?(?: ?'?s? ?close)?",
           lambda m: {"key": "gap", "op": ">=" if m.group("d") in ("higher", "above") else "<=", "value": pct(m, m.group("d") in ("lower", "below"))}, "event")

    # the previous session's move named as such, often in an edit: "the previous-day decline to 3.8%", "yesterday's drop of 2%", "prior day's loss at least 4%"
    t.each(rf"\b(?:the )?(?:previous|prior|preceding|yesterday|last)(?:(?: ?- ?| )(?:day|session|trading day))?(?: ?'s)?(?: ?- ?| )?(?:day ?'?s? |session ?'?s? )?(?P<d>decline|drop|fall|loss|selloff|sell off|down move|gain|rise|rally|up move|move)"
           rf"(?: (?:to|of|at|by|be|was|is|has to be|needs to be|should be))*(?: (?:{ATLEAST}|{MORETHAN}))? (?P<num>\d+(?:\.\d+)?) ?%(?: or more)?",
           lambda m: {"key": "prev_return", "op": "<=" if m.group("d") in ("decline", "drop", "fall", "loss", "selloff", "sell off", "down move") else ">=",
                      "value": pct(m, m.group("d") in ("decline", "drop", "fall", "loss", "selloff", "sell off", "down move"))} if m.group("d") != "move" else
                     {"key": "prev_return", "op": "abs>=", "value": _f(m.group("num"))}, "event")
    t.each(rf"\b(?P<num>\d+) (?:straight |consecutive )?(?P<d>{UP}|{DOWN}) (?:day|session)s?(?: in a row| straight| consecutively)?\b",
           lambda m: {"key": "up_streak" if re.fullmatch(UP + r"s?", m.group("d")) else "down_streak", "op": ">=", "value": _f(m.group("num"))}, "event")
    # "the decline has to be 4%": the size of a move the question already has (edits map it onto that condition)
    t.each(rf"\bthe (?P<d>decline|drop|fall|loss|selloff|gain|rise|rally|move)(?: (?:has to be|needs to be|should be|must be|is|of|to|at least|be))+ (?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)?",
           lambda m: {"key": "move", "op": "<=" if m.group("d") in ("decline", "drop", "fall", "loss", "selloff") else ">=", "value": pct(m, m.group("d") in ("decline", "drop", "fall", "loss", "selloff")), "anchor": None}, "event")
    # the previous session's move ("a 2% down day", "yesterday was red", "after IWM drops 2% or more in a day")
    t.each(rf"\b(?:a |an )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: ?\+| or more| or larger| or bigger| or worse| plus)? (?P<d>{DOWN}|{UP}|red|green|losing|winning) (?:day|session)\b",
           lambda m: {"key": "prev_return", "op": "<=" if re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing") else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing")))}, "event")
    t.each(rf"\b(?:a |an )?(?P<d>{DOWN}|{UP}|red|green|losing|winning) (?:day|session) (?:of |by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)?",
           lambda m: {"key": "prev_return", "op": "<=" if re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing") else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing")))}, "event")
    t.each(rf"\b(?:{SYM_RX} )?(?P<d>{DOWN}|{UP})s? (?:by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: ?\+| or more| or larger)? (?:in|on|for|during) (?:a|one|the|a single|1) (?:day|session|trading day)\b",
           lambda m: {"key": "prev_return", "op": "<=" if re.match(DOWN, m.group("d")) else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d"))))}, "event")
    t.each(rf"\b(?:yesterday|the previous (?:day|session)|the day before|last session)(?: was| closed| close| is| end)? (?:(?P<d>{DOWN}|{UP}|red|green))(?: (?:by |more than |at least |over )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)?)?",
           lambda m: {"key": "prev_return", "op": ("<=" if m.group("num") else "<") if (re.match(DOWN, m.group("d")) or m.group("d") == "red") else (">=" if m.group("num") else ">"),
                      "value": pct(m, bool(re.match(DOWN, m.group("d")) or m.group("d") == "red")) if m.group("num") else 0.0}, "event")
    t.each(rf"\b(?:the )?(?:previous|yesterday) (?:day|session)?(?:'s)? ?(?:close|return|move)? (?:was |closed |close |is )?(?:(?P<d>{DOWN}|{UP}) )?(?:more than |at least |over |by )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)?",
           lambda m: {"key": "prev_return", "op": "<=" if (m.group("d") and re.match(DOWN, m.group("d"))) else ">=", "value": pct(m, bool(m.group("d") and re.match(DOWN, m.group("d"))))} if m.group("d") else None, "event")
    t.each(rf"\b(?:a |an )?(?:really |very |big |large |huge |sharp |significant |major |terrible )+(?P<d>{DOWN}|{UP}|red|green|losing|winning) (?:day|session)\b",
           lambda m: {"key": "prev_return", "op": "<" if (re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing")) else ">", "value": 0.0, "size": "vague"}, "event")
    t.each(rf"\b(?:a |an )?(?P<d>{DOWN}|{UP}|red|green|losing|winning) (?:day|session)\b",
           lambda m: {"key": "prev_return", "op": "<" if (re.match(DOWN, m.group("d")) or m.group("d") in ("red", "losing")) else ">", "value": 0.0}, "event")
    # an X% move with no time anchor ("after SPY drops 2%", "after falling 1%", "after a 1.5% drop"): read as a session's move, said
    t.each(rf"\b(?:{SYM_RX} )?(?P<d>{DOWN}|{UP})s? (?:by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: ?\+| or more)?(?! (?:from|off|above|below|of|in the|by \d|at \d|as of \d|by \d))(?<!is down)(?<!is up)",
           lambda m: {"key": "move", "op": "<=" if re.match(DOWN, m.group("d")) else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d")))), "anchor": None}, "event")
    t.each(rf"\b(?:a |an )?(?P<d>drop|fall|decline|selloff|sell off|loss|rise|rally|gain|pop|jump|move down|move up) (?:of |by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more| or larger| or bigger| or worse|\+)?(?! (?:from|off|above|below|in the|by \d))",
           lambda m: {"key": "move", "op": "<=" if m.group("d") in ("drop", "fall", "decline", "selloff", "sell off", "loss", "move down") else ">=",
                      "value": pct(m, m.group("d") in ("drop", "fall", "decline", "selloff", "sell off", "loss", "move down")), "anchor": None}, "event")
    t.each(rf"\b(?:a |an )?(?P<num>\d+(?:\.\d+)?) ?%\+? (?P<d>drop|fall|decline|selloff|sell off|loss|rise|rally|gain|pop|jump)\b",
           lambda m: {"key": "move", "op": "<=" if m.group("d") in ("drop", "fall", "decline", "selloff", "sell off", "loss") else ">=", "value": pct(m, m.group("d") in ("drop", "fall", "decline", "selloff", "sell off", "loss")), "anchor": None}, "event")

    # "SPY is up at 10:30", "if it's down 0.5% by 11": the change on the day so far (against the previous close) at that time
    t.each(rf"\b(?:{SYM_RX} |it )?(?:is|was|s|trades|trading|sits) (?P<d>up|down|green|red|higher|lower|positive|negative)(?: (?:by )?(?P<num>\d+(?:\.\d+)?) ?%)?(?= (?:at|by|as of) \d)",
           lambda m: {"key": "change", "op": (">=" if m.group("num") else ">") if m.group("d") in ("up", "green", "higher", "positive") else ("<=" if m.group("num") else "<"),
                      "value": (pct(m, m.group("d") not in ("up", "green", "higher", "positive")) if m.group("num") else 0.0)}, "event")
    # a move from the session's open during the day ("drops 1.5% from the open", "falls 1% in the morning")
    t.each(rf"\b(?:{SYM_RX} )?(?:is )?(?P<d>{DOWN}|{UP})s? (?:by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)? (?:from|off|below|above|under|over) (?:the |its |today ?)?open",
           lambda m: {"key": "from_open", "op": "<=" if re.match(DOWN, m.group("d")) else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d"))))}, "event")
    t.each(rf"\b(?:{SYM_RX} )?(?:is )?(?P<d>{DOWN}|{UP})s? (?:by )?(?:{ATLEAST}|{MORETHAN} )?(?P<num>\d+(?:\.\d+)?) ?%(?: or more)? (?:in the (?:morning|first hour)|early|in the first (?:\d+ )?(?:min|hour)|intraday|during the (?:day|session)|by (?:lunch|noon|mid ?day))",
           lambda m: {"key": "from_open", "op": "<=" if re.match(DOWN, m.group("d")) else ">=", "value": pct(m, bool(re.match(DOWN, m.group("d")))), "when": "morning"}, "event")
    t.each(r"\b(?:a |an )?(?P<d>bad|rough|weak|red|down|good|strong|green|up) (?:morning|open|first hour)\b",
           lambda m: {"key": "from_open", "op": "<" if m.group("d") in ("bad", "rough", "weak", "red", "down") else ">", "value": 0.0, "size": "vague", "when": "morning"}, "event")

    # VIX, streaks, weekdays, VWAP
    t.each(rf"\bvix (?:close|closing|closed|was|is|at|level|reading)?(?: (?:the )?(?:previous|yesterday)(?: day| session)?)? ?(?P<op>above|over|below|under|>|<|>=|<=|greater than|less than|higher than|lower than) (?P<num>\d+(?:\.\d+)?)(?: (?:the )?(?:previous|yesterday) (?:day|session))?",
           lambda m: {"key": "vix", "op": ">" if m.group("op") in ("above", "over", ">", ">=", "greater than", "higher than") else "<", "value": _f(m.group("num"))}, "event")
    t.each(r"\b(?:on |every |only on |just )?(?P<d>monday|tuesday|wednesday|thursday|friday)s?\b",
           lambda m: {"key": "weekday", "op": "=", "value": m.group("d")[:3].capitalize()}, "event")
    t.each(r"\b(?:is |was |trades |price is |price )?(?P<d>above|below|over|under) (?:the )?vwap\b",
           lambda m: {"key": "vwap", "op": ">" if m.group("d") in ("above", "over") else "<"}, "event")
    t.each(r"\b(?:reclaim|cross(?:es)? (?:back )?above|gets? back above|breaks? (?:back )?above) (?:the )?vwap\b", lambda m: {"key": "vwap_cross", "op": "above"}, "event")
    t.each(r"\b(?:lose|loses|losing|cross(?:es)? below|breaks? below|falls? below|drops? below) (?:the )?vwap\b", lambda m: {"key": "vwap_cross", "op": "below"}, "event")
    t.each(r"\b(?:fomc|fed (?:meeting|day|decision)s?|cpi|nfp|jobs report|earnings|opex|quad witching|rsi|macd|bollinger|fibonacci|moving average|sma|ema|all[- ]time high|52[- ]week)\b",
           lambda m: {"key": "unsupported_event", "what": m.group(0)}, "event")

    # ---- outcomes: what is counted
    t.each(r"\b(?:fill|close) (?:the )?gap\b", lambda m: {"key": "touch_level", "level": "prev_close", "direction": None}, "outcome")
    t.each(r"\b(?:trade|go|break|print|dip|get|fall|drop)s? (?:below|under) (?:the )?(?:previous|yesterday)(?: day)?(?:'s)? ?(?P<w>low|high)\b|\b(?:trade|go|break|print|get|rise|rally)s? (?:above|over) (?:the )?(?:previous|yesterday)(?: day)?(?:'s)? ?(?P<w2>high|low)\b|\bnew low below (?:the )?(?:previous|yesterday)(?: day)?(?:'s)? ?low\b",
           lambda m: {"key": "touch_level", "level": "prev_" + (m.group("w") or m.group("w2") or "low"), "direction": "down" if (m.group("w") or "low") == "low" and not m.group("w2") else "up"}, "outcome")
    t.each(r"\b(?:get|gets|come|comes|go|goes|trade|trades|make it|bounce|recover)s? back (?:to|above|up to) (?:the |its )?(?P<w>open|previous close|yesterday's close)\b",
           lambda m: {"key": "touch_level", "level": "open" if m.group("w") == "open" else "prev_close", "direction": "up"}, "outcome")
    t.each(rf"\b(?:close|finish|end)s? (?:the day |the session )?(?:above|higher than|over) (?:the )?(?:previous|yesterday)(?: day)?(?:'s)? ?close\b|\b(?:close|finish|end)s? (?:the day )?(?:up|higher|green) (?:on the day|for the day) (?:vs|versus|from) (?:the )?(?:previous|yesterday)(?:'s)? close\b|\bfinish (?:the day )?above (?:the )?(?:previous|yesterday)(?:'s)? close\b",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "up"}, "outcome")
    t.each(rf"\b(?:close|finish|end)s? (?P<d>higher|lower|up|down|green|red) (?:on|for) the day\b",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "up" if m.group("d") in ("higher", "up", "green") else "down"}, "outcome")
    t.each(rf"\b(?:close|finish|end)s? (?:the day |the session )?(?:below|lower than|under) (?:the )?(?:previous|yesterday)(?: day)?(?:'s)? ?close\b",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "down"}, "outcome")
    t.each(rf"\b(?:close|finish|end)s? (?:(?:the|that) (?:day|session) )?(?:higher|lower|up|down|green|red) (?:the )?(?:next|following) (?:day|session)\b|\b(?:the )?(?:next|following) (?:day|session) (?:close|closes|finish|end) (?P<d2>higher|lower|up|down|green|red)\b",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "up" if re.search(r"higher|up|green", m.group(0)) else "down", "next_session": True}, "outcome")
    t.each(rf"\b(?:close|finish|end)s? (?:the day |the session )?(?P<d>higher|lower|above|below|up|down|green|red)(?: than)? (?:(?:it|where it) (?:open|was at the open|started)|its open|the open|(?:the )?(?:day|session)'?s? open)\b",
           lambda m: {"key": "close_vs", "ref": "open", "direction": "up" if m.group("d") in ("higher", "above", "up", "green") else "down"}, "outcome")
    t.each(rf"\b(?:close|closes|finish|finishes|end|ends|is|turns|stays) (?:the day |the session )?(?:still )?(?P<d>green|red|positive|negative|up on the day|down on the day)(?: for the day| on the day| at the close)?\b|\b(?:still )?(?:end|finish|close)s? (?:the day )?(?P<d2>green|red)\b",
           lambda m: {"key": "close_vs", "ref": "open", "direction": "up" if (m.group("d") or m.group("d2") or "").startswith(("green", "positive", "up")) else "down"}, "outcome")
    t.each(r"\b(?:close|closes|closed|finish|finishes|finished|end|ends|ended) (?P<d>higher|lower|up|down)\b(?! (?:than|from|vs|versus|on|for) )",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "up" if m.group("d") in ("higher", "up") else "down"}, "outcome")
    t.each(rf"\b(?:is|be|was|are|trades|stays|sits|ends up|end up) (?:{SYM_RX} |it |the price |price )?(?:still |trading )?(?P<d>higher|lower)\b(?! than)",
           lambda m: {"key": "close_vs", "ref": "auto", "direction": "up" if m.group("d") == "higher" else "down"}, "outcome")
    t.each(rf"\b(?:the )?(?P<at>\d\d?:\d\d|close) (?:close |price )?(?:is |was |ends? )?(?P<d>above|below|higher than|lower than) (?:the )?(?P<from>\d\d?:\d\d|open|entry) (?:price|close|level)?",
           lambda m: {"key": "close_vs", "ref": "start" if m.group("from") != "open" else "open", "direction": "up" if m.group("d") in ("above", "higher than") else "down", "from": m.group("from")}, "outcome")
    t.each(rf"\b(?:touch|hit|reach|trade|rise|rally|go up|gain|climb|move up|run|pop|get to) (?:to |up )?(?P<sign>\+ ?)?(?P<num>\d+(?:\.\d+)?) ?%(?: (?:above|over|from|off) (?:the |its )?(?P<ref>open|entry|start|previous close))?(?: (?:intraday|at any point|at some point|during the day))?",
           lambda m: {"key": "touch", "value": _f(m.group("num")), "ref": m.group("ref") or "start"}, "outcome")
    t.each(rf"\b(?:touch|hit|reach|trade|drop|fall|decline|go down|sell off|dip|slide|move down) (?:to |down )?(?P<sign>- ?)?(?P<num>\d+(?:\.\d+)?) ?%(?: (?:below|under|from|off) (?:the |its )?(?P<ref>open|entry|start|previous close))?(?: (?:intraday|at any point|at some point|during the day))?",
           lambda m: {"key": "touch", "value": -_f(m.group("num")), "ref": m.group("ref") or "start"}, "outcome")
    t.each(rf"\b(?P<sign>[+-]) ?(?P<num>\d+(?:\.\d+)?) ?%(?: (?:above|below|from|off) (?:the |its )?(?P<ref>open|entry|start))?",
           lambda m: {"key": "touch", "value": _f(m.group("num")) * (-1 if m.group("sign") == "-" else 1), "ref": m.group("ref") or "start"}, "outcome")
    t.each(r"\b(?:make|makes|set|sets|print|prints|hit|hits|put in|puts in|form|forms) (?:its |the |a )?(?:daily |intraday )?(?P<w>high|low|top|bottom)s?(?: of the day)?\b|\b(?P<w2>bottom|top)s? out\b|\b(?:the )?(?:high|low) of the day\b",
           lambda m: {"key": "extreme_time", "which": "low" if (m.group("w") or m.group("w2") or m.group(0)) in ("low", "bottom") or "low" in m.group(0) else "high"}, "outcome")
    t.each(r"\b(?:biggest|largest|most|strongest) (?:moves?|movement|volatility|swings?)\b|\bmove (?:the )?most\b|\b(?:most|more) volatile\b|\bvolatility\b",
           lambda m: {"key": "abs_move"}, "outcome")
    t.each(r"\b(?:keep|keeps|kept|continue|continues|continued) (?:to )?(?:fall|drop|decline|slide|sell off|selloff|go down|going down|tank|dump|rise|rally|climb|gain|go up|going up|lower|higher)(?:ing)?\b|\bfollow[- ]?through\b|\bextend (?:the|its) (?:drop|decline|fall|losses|gains|rally)\b",
           lambda m: {"key": "continue"}, "outcome")
    t.each(r"\b(?:is |be |was )?(?:the )?(?:next|following) (?:day|session|trading day) (?:is |was |closes |ends |finishes )?(?P<d>green|red|up|down|higher|lower|positive|negative)\b",
           lambda m: {"key": "close_vs", "ref": "prev_close", "direction": "up" if m.group("d") in ("green", "up", "higher", "positive") else "down", "next_session": True}, "outcome")
    t.each(r"\b(?:recover|bounce|bounce back|come back|rebound|snap back)\b", lambda m: {"key": "recover"}, "outcome")
    t.each(r"\b(?:reverse|reversal|turn around|turn)\b", lambda m: {"key": "reverse"}, "outcome")
    t.each(r"\b(?:continue|keep (?:going|falling|rising|dropping)|follow through)\b", lambda m: {"key": "continue"}, "outcome")
    t.each(rf"\b(?:go|goes|move|moves) (?P<d>up|down|higher|lower)\b|\b(?P<d2>rise|fall|drop|rally|decline|climb|go green|go red)\b",
           lambda m: {"key": "direction", "direction": "up" if (m.group("d") or m.group("d2") or "") in ("up", "higher", "rise", "rally", "climb", "go green") else "down"}, "outcome")
    t.each(r"\b(?:the |its )?direction(?: of)?\b|\b(?:the |its )?(?:overall )?(?:day|session)(?:'s)? (?:direction|close|result)\b", lambda m: {"key": "direction_word"}, "outcome")
    t.each(r"\b(?:open[- ]to[- ]close|open to close) (?:return|move|change)?\b", lambda m: {"key": "measure", "measure": "open_to_close"}, "outcome")
    t.each(r"\b(?:close[- ]to[- ]close|daily) (?:return|move|change)\b", lambda m: {"key": "measure", "measure": "close_to_close"}, "outcome")
    t.each(r"\b(?:overnight) (?:return|move|change|gap)?\b", lambda m: {"key": "measure", "measure": "overnight"}, "outcome")
    t.each(r"\b(?:average|mean|median) (?:return|move|change|result)\b|\b(?:return|returns|performance|perform)\b", lambda m: {"key": "measure", "measure": "return"}, "outcome")

    # ---- when the measurement starts and ends
    t.each(r"\b(?:the )?rest of (?:the )?(?:[a-z]{2,5}(?: ?'?s)? )?(?:day|session|trade)\b|\bremainder of (?:the )?(?:[a-z]{2,5}(?: ?'?s)? )?(?:day|session)\b|\bremaining (?:session|day)\b|\bafter (?:the |that )?(?:candle|bar|first \d+ min) (?:close|closes|is done)\b|\bfrom there\b",
           lambda m: {"anchor": "rest"}, "anchor", "scope")
    t.each(r"\b(?:the |its )?(?:whole|entire|full|overall) (?:day|session)\b|\bhow the (?:day|session) (?:ends|end|closes|goes|finishes)\b|\b(?:the )?(?:day|session) (?:as a whole)\b",
           lambda m: {"anchor": "day"}, "anchor", "scope")
    t.each(r"\b(?:in |within |during )?(?:the )?first (?P<num>\d+(?:\.\d+)?) ?(?P<u>min|hour)s?(?! (?:candle|bar))(?: (?:after|of) (?:the )?(?:open|session|day))?",
           lambda m: {"anchor": "end_after_open", "minutes": int(_f(m.group("num")) * (60 if m.group("u") == "hour" else 1))}, "anchor", "end")
    t.each(r"\b(?:in |within |during )(?:the )?first (?P<w>hour|half ?hour)(?! (?:candle|bar))(?: (?:after|of) (?:the )?(?:open|session|day))?",
           lambda m: {"anchor": "end_after_open", "minutes": 60 if m.group("w") == "hour" else 30}, "anchor", "end")
    nsess = lambda g, u: (int(_f(g)) if g and g[0].isdigit() else 1) * (5 if u.startswith("week") else 1)
    t.each(r"\b(?:within|over) (?:the )?(?:next |following )?(?P<num>\d+|a|one) ?(?:trading )?(?P<u>days?|sessions?|weeks?)\b(?: (?:after|later|afterwards?))?|\bin (?:the )?(?:next|following) (?P<num2>\d+|a|one)? ?(?:trading )?(?P<u2>days?|sessions?|weeks?)\b",
           lambda m: {"anchor": "sessions", "n": nsess(m.group("num") or m.group("num2"), m.group("u") or m.group("u2")), "how": "within"}, "anchor", "end")
    t.each(r"\b(?P<num>\d+|a|one) ?(?:trading )?(?P<u>days?|sessions?|weeks?) (?:later|afterwards?|on|out)\b|\b(?P<num3>\d+) ?(?:trading )?(?P<u3>days?|sessions?|weeks?) after\b|\bafter (?P<num2>\d+|a|one) (?:trading )?(?P<u2>days?|sessions?|weeks?)\b|\b(?P<num4>\d+|a|one) (?P<u4>weeks?) after\b",
           lambda m: {"anchor": "sessions", "n": nsess(m.group("num") or m.group("num2") or m.group("num3") or m.group("num4"), m.group("u") or m.group("u2") or m.group("u3") or m.group("u4")), "how": "at"}, "anchor", "end")
    t.each(r"\b(?:within|in|over) (?:the next |an |the )?(?P<num>\d+(?:\.\d+)?)? ?(?P<u>min|hour)s?(?: after| later| of (?:it|that))?\b",
           lambda m: {"anchor": "within", "minutes": int(_f(m.group("num") or 1) * (60 if m.group("u") == "hour" else 1))}, "anchor", "end")
    t.each(r"\b(?:by|before|until|till|til|through) (?:the )?(?P<w>close|endofday|end of (?:the )?day|lunch|noon|midday|mid ?day)\b",
           lambda m: {"anchor": "end_at", "at": "close" if m.group("w") in ("close", "endofday") or m.group("w").startswith("end") else "12:00", "words": m.group("w")}, "anchor", "end")
    t.each(r"\b(?:on )?the (?:day|session|trading day) (?:following|after)\b", lambda m: {"anchor": "next_session"}, "anchor", "end")
    t.each(r"\b(?:on |during |in )?the (?:previous|prior|preceding|last) (?:trade |trading )?(?:day|session)\b(?!'s| (?:close|open|high|low|return|move))", lambda m: {"anchor": "prev_session"}, "anchor", "condition")
    t.each(r"\b(?:the )?(?:next|following) (?:day|session|trading day)\b|\btomorrow\b", lambda m: {"anchor": "next_session"}, "anchor", "end")
    t.each(r"\b(?:at|on|from|after|off|since) (?:the )?(?:market )?open\b|\bat the bell\b|\bopening bell\b|\bevery (?:morning|day at the open)\b|\bin the morning\b|\bevery morning\b",
           lambda m: {"anchor": "open"}, "anchor", "start")
    t.each(r"\b(?:at|into|near|before) (?:the )?(?:close|closing bell|end of (?:the )?day|endofday)\b|\bat endofday\b", lambda m: {"anchor": "close"}, "anchor")
    t.each(r"\bintraday\b|\bduring the (?:day|session)\b|\bat any point\b|\bat some point\b|\bin one day\b|\bin a day\b|\bin a single day\b", lambda m: {"anchor": "intraday"}, "anchor")
    t.each(r"\b(?:yesterday|the day before|the previous (?:day|session))(?: ?'?s)?\b", lambda m: {"anchor": "yesterday"}, "anchor")
    t.each(r"\btoday(?: ?'?s)?\b|\bthe (?:same|current) (?:day|session)\b", lambda m: {"anchor": "today"}, "anchor")

    # ---- option words (a sentence that has them is a strategy, read by the strategy readers)
    t.each(r"\b(?:call|put|option|contract|spread|straddle|strangle|condor|butterfly|premium|0 ?dte|\d+ ?dte|dte|atm|otm|itm|delta|expir\w*|weekl(?:y|ies)|takeprofit|stoploss|profit target|stop loss)\b",
           lambda m: {"word": m.group(0)}, "option", "option")
    t.each(r"\b(?:buy|sell|short|long|enter|exit|hold)\b|\b(?:in|out) (?:at|by)\b", lambda m: {"word": m.group(0).split()[0]}, "action", "option")

    # ---- edits, exploration
    t.each(r"^(?:no+|nah|nope|not quite|wrong|sorry|oops|wait|actually|rather|instead|i meant|i mean|ok so|okay so|so)\b[ ,.!]*(?:i meant |i mean )?", lambda m: {"cue": "correction"}, "edit", "edit")
    t.each(r"\b(?:instead|actually|rather)\b", lambda m: {"cue": "instead"}, "edit", "edit")
    t.each(r"^(?:and |what about |how about |same (?:thing )?(?:for|with) |what if |try |use |make it |change |switch to |only |just |also |add |remove |drop |without |but )", lambda m: {"cue": m.group(0).strip()}, "edit", "edit")
    t.each(r"\b(?:undo|go back|revert|never ?mind|scratch that)\b", lambda m: {"cue": "undo"}, "edit", "edit")
    t.each(r"\b(?:show|list|see|give)(?: me)? (?:all |every |only )?(?:the )?(?P<w>losing|winning|worst|best|biggest|red|green|up|down|failed|successful|missed)? ?(?:day|trade|session|one|date|time|instance|case|example)s?\b(?:(?: it| that| this) (?:happened|worked|failed|did not happen|didn't happen))?|\bthe days (?:it|that) (?:happened|worked|failed)\b|\bwhich (?:days|trades|dates)\b",
           lambda m: {"filter": (m.group("w") or "all")}, "explore", "explore")

    t.mentions.sort(key=lambda x: x.a)
    return n, t.mentions


# ---------------------------------------------------------------------------------------------------------------- frame
KINDS = ("frequency", "agreement", "movement", "timing", "breakdown", "strategy")
TIME_ORDER = {"prev_return": 0, "move": 0, "up_streak": 0, "down_streak": 0, "vix": 0, "weekday": 0, "gap": 1, "first_candle": 2, "from_open": 3, "change": 3, "vwap": 3, "vwap_cross": 3}


def unread_words(n: Norm, mentions: list[Mention]) -> list[dict]:
    """Words of the sentence no recogniser claimed and that are not filler: they may carry meaning nobody read."""
    taken = [False] * (len(n.norm) + 1)
    for m in mentions:
        for i in range(m.a, m.b):
            taken[i] = True
    out = []
    for w in re.finditer(r"[a-z][a-z'&]*|\d+(?:\.\d+)?", n.norm):
        if any(taken[w.start():w.end()]):
            continue
        word = w.group(0)
        if word in FILLER or len(word) <= 2 and not word.isdigit() or word.rstrip("s") in FILLER:
            continue
        oa, ob = n.span(w.start(), w.end())
        while ob < len(n.text) and (n.text[ob].isalnum() or n.text[ob] == "'"):
            ob += 1  # the whole word as typed ("trading", not the part its normal form covers)
        out.append({"text": n.text[oa:ob], "norm": word, "span": [oa, ob]})
    return out


@dataclass
class Reading:
    text: str
    frame: dict
    mentions: list[Mention]
    norm: Norm | None = None
    unread: list[dict] = field(default_factory=list)
    unclaimed: list[Mention] = field(default_factory=list)
    clarify: dict | None = None      # {"text", "options": [{"label", "patch"}], "slot"}
    flag: str | None = None          # a request that is declined, with the reason (nothing runs)

    @property
    def ok(self) -> bool:
        return self.clarify is None and self.flag is None

    @property
    def complete(self) -> bool:
        """Every word that carries meaning is in a slot: nothing was left unread."""
        return self.ok and not self.unread and not self.unclaimed


def _cond(m: Mention) -> dict:
    v = dict(m.value)
    v["text"] = m.text
    return v


def _kind_of(ms: list[Mention], has_option: bool) -> str | None:
    kinds = [m.value["kind"] for m in ms if m.kind == "kind"]
    if has_option or "strategy" in kinds:
        return "strategy"
    for k in ("agreement", "timing", "breakdown", "movement", "frequency"):
        if k in kinds:
            return k
    return None


def read(text: str, known: list[str] | None = None, default_symbol: str = "SPY", context: dict | None = None) -> Reading:
    """The question frame of one sentence on its own (no conversation). See converse.py for messages that edit a frame."""
    known = list(known or STORED_DEFAULT)
    n, ms = tag(text, known)
    used = set()

    def use(m: Mention, role: str):
        m.used, m.role = True, role
        used.add(id(m))

    has_option = any(m.kind in ("option",) for m in ms) or any(m.kind == "action" and m.value["word"] in ("buy", "sell", "short", "long") for m in ms)
    kind = _kind_of(ms, has_option)
    frame: dict = {"kind": kind, "text": text, "assumptions": [], "conditions": [], "symbols": []}
    r = Reading(text, frame, ms, n)

    # ---- instrument(s)
    syms = [m for m in ms if m.kind == "symbol"]
    for m in syms:
        use(m, "symbol")
    # "when VIX is above 30, how often does SPY ...": a symbol inside a condition is that condition's, not the subject
    inside = lambda m: any(e.kind == "event" and e.value.get("key") == "vix" and e.a <= m.a and m.b <= e.b for e in ms)
    syms = [m for m in syms if not inside(m)] or syms
    stored = [m.value["symbol"] for m in syms if m.value["stored"]]
    missing = [m.value["symbol"] for m in syms if not m.value["stored"]]
    frame["symbols"] = list(dict.fromkeys(stored))
    if missing and not stored:
        r.flag = (f"No price history for {', '.join(dict.fromkeys(missing))} is stored, so nothing can be measured for it. Stored: {', '.join(known)}." +
                  (" SPX itself is not stored; SPY tracks the same index." if "SPX" in missing else ""))
    frame["symbol"] = frame["symbols"][0] if frame["symbols"] else default_symbol
    frame["symbol_stated"] = bool(frame["symbols"])
    if not frame["symbols"]:
        frame["assumptions"].append({"slot": "symbol", "text": f"No ticker was named, so {default_symbol} is used."})

    # ---- period
    for m in ms:
        if m.kind == "period":
            use(m, "period")
            frame["period"] = [m.value["start"], m.value["end"]]
            if m.value.get("note"):
                frame["assumptions"].append({"slot": "period", "text": f"Period read as {m.value['note']}."})
    frame.setdefault("period", [None, None])

    for m in ms:
        if m.kind == "kind":
            use(m, "kind")
        if m.kind == "edit" and m.value.get("cue") in ("correction", "instead"):
            use(m, "edit")

    kinds_said = {m.value["kind"] for m in ms if m.kind == "kind"}
    if "forecast" in kinds_said and not kinds_said & {"frequency", "agreement", "movement"}:
        r.flag = ("Strata does not forecast. It can tell you how often something like this happened before, on the stored history, "
                  "for example \"How often does SPY close above its open after a down day?\"")
        return r
    if kind in ("strategy",):
        frame["kind"] = "strategy"
        return r  # the strategy readers take it from here; strategy_coverage checks what they produce
    if "explain" in kinds_said or "reliability" in kinds_said:
        frame["kind"] = "explain" if "explain" in kinds_said else "reliability"
        return r
    if kind is None:
        # a question with no word for what kind it is: an event and an outcome together still make a frequency question
        if any(m.kind == "outcome" and m.value["key"] in ("close_vs", "touch", "touch_level") for m in ms):
            kind = "frequency"
        elif any(m.kind == "explore" for m in ms):
            frame["kind"] = "explore"
            return r
        else:
            frame["kind"] = None
            return r
    frame["kind"] = kind

    events = [m for m in ms if m.kind == "event"]
    outcomes = [m for m in ms if m.kind == "outcome"]
    anchors = [m for m in ms if m.kind == "anchor"]
    clocks = [m for m in ms if m.kind == "clock"]

    for m in events:
        if m.value.get("key") == "unsupported_event":
            use(m, "condition")
            r.flag = (f"\"{m.text}\" cannot be tested: " + ("no calendar of real economic or earnings events is stored (the event dates in the sandbox are simulated)." if re.search(r"fomc|fed|cpi|nfp|jobs|earnings|opex|witching", m.value["what"]) else
                                                             "that indicator is not available in the question reader yet. It can be used in a strategy written in the rule editor."))
            return r

    if kind == "agreement":
        return _agreement(r, events, outcomes, anchors, clocks, use)
    if kind == "timing":
        return _timing(r, events, outcomes, anchors, clocks, use, ms)
    if kind == "breakdown":
        return _breakdown(r, events, outcomes, anchors, clocks, use, ms)
    return _frequency(r, kind, events, outcomes, anchors, clocks, use)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _condition_events(r: Reading, events: list[Mention], use, exclude: set[int] = frozenset()) -> list[dict]:
    conds = []
    for m in events:
        if id(m) in exclude:
            continue
        v = dict(m.value)
        if v.get("key") == "move":
            # an X% move with no time anchor reads as a session's close-to-close move, and the reading says so
            v["key"] = "prev_return"
            r.frame["assumptions"].append({"slot": "condition", "text": f"\"{m.text}\" is read as a session that closed {abs(v['value']):g}% {'down' if v['value'] < 0 else 'up'} on the previous close; the observation starts the next session.",
                                           "alternative": "a move of that size from the open, during the same session"})
        use(m, "condition")
        conds.append({**v, "text": m.text})
    return conds


def _vague_sizes(r: Reading, conds: list[dict]) -> None:
    """A condition with no size ("a really bad day", "opens up a lot") is measured at several sizes, and the reading says so."""
    for c in conds:
        if c.get("size") == "vague" and c["key"] in ("prev_return", "gap", "from_open"):
            sizes = {"prev_return": [1, 2, 3], "gap": [0.5, 1, 2], "from_open": [0.5, 1, 1.5]}[c["key"]]
            c["values"] = [(-x if c["op"] in ("<", "<=") else x) for x in sizes]
            c["op"] = "<=" if c["op"] in ("<", "<=") else ">="
            what = {"prev_return": "the previous session's move", "gap": "the gap at the open", "from_open": "the move from the open"}[c["key"]]
            r.frame["assumptions"].append({"slot": "condition", "text": f"\"{c['text']}\" gives no size, so {what} is measured at {', '.join(f'{x:g}%' for x in sizes)} and each is shown."})


MARKER = re.compile(r"\b(?:after|following|when|whenever|if|once|on days|days (?:when|that|where)|sessions (?:when|that|where)|given|conditional on|in sessions|on sessions|with|where|while)\b(?!.*\b(?:after|when|if|once)\b)")


def _marked(r: Reading, m: Mention) -> bool:
    """Is this event introduced as a condition ("after a 2% drop", "when it gaps down")? Looks back to the clause start."""
    if not r.norm:
        return True
    before = r.norm.norm[:m.a]
    cut = max(before.rfind(","), before.rfind("?"), before.rfind(";"), before.rfind(" how often "), before.rfind(" what "))
    seg = before[cut + 1:] if cut >= 0 else before
    return bool(re.search(r"\b(?:after|following|when|whenever|if|once|on days|days (?:when|that|where)|given|conditional on|in sessions|on sessions|where|while|only)\b", seg))


def _frequency(r: Reading, kind: str, events, outcomes, anchors, clocks, use) -> Reading:
    f = r.frame
    out = None
    # the outcome: what is counted
    primary = [m for m in outcomes if m.value["key"] in ("close_vs", "touch", "touch_level")]
    if not primary and kind == "frequency":
        # "how often does SPY rise 1% (from the open)": a move that is not introduced as a condition is what is counted
        lead = [m for m in events if m.value.get("key") in ("move", "from_open") and m.value.get("value") and not _marked(r, m)]
        if lead:
            m = lead[0]
            use(m, "outcome")
            f["outcome"] = {"key": "touch", "value": float(m.value["value"]), "ref": "open" if m.value.get("key") == "from_open" else "start", "text": m.text}
            events = [e for e in events if e is not m]
            primary = [m]
    vague = [m for m in outcomes if m.value["key"] in ("recover", "reverse", "continue", "direction")]
    measure = [m for m in outcomes if m.value["key"] == "measure"]
    if primary and not f.get("outcome"):
        out = primary[0]
        use(out, "outcome")
        f["outcome"] = {**out.value, "text": out.text}
        for extra in primary[1:]:
            pass  # a second outcome stays unclaimed: reported below
    elif kind == "frequency" and vague and not f.get("outcome"):
        v = vague[0]
        use(v, "outcome")
        if v.value["key"] == "direction":
            f["outcome"] = {"key": "close_vs", "ref": "prev_close", "direction": v.value["direction"], "text": v.text}
            f["assumptions"].append({"slot": "outcome", "text": f"\"{v.text}\" is measured as the close against the previous close.", "alternative": "the close against the session's open"})
        else:
            f["outcome"] = {"key": v.value["key"], "text": v.text}
    elif kind == "frequency" and not f.get("outcome"):
        # "how often does IWM gap up more than 1%": the event itself is what is counted
        ev = [m for m in events if m.value.get("key") in ("gap", "prev_return", "move", "up_streak", "down_streak") and m.value.get("value") is not None and not _marked(r, m)]
        if len(ev) == 1 and not any(m.kind == "anchor" and m.value.get("anchor") == "next_session" for m in anchors):
            m = ev[0]
            use(m, "outcome")
            v = dict(m.value)
            if v.get("key") in ("move", "prev_return"):
                v["key"] = "day_return"  # counted on its own, the session's own move: until V9 a "prev_return" event counted the day after
            f["outcome"] = {"key": "event", "event": v, "text": m.text}
            events = [e for e in events if e is not m]
    if kind == "movement":
        for m in measure:
            use(m, "outcome")
            f["measure"] = m.value["measure"]
        if not f.get("outcome"):
            f["outcome"] = {"key": "end_return"}
        for m in [x for x in outcomes if x.value["key"] == "abs_move"]:
            use(m, "outcome")
            f["measure"] = "abs_move"

    f["conditions"] = _condition_events(r, events, use)
    _vague_sizes(r, f["conditions"])

    # when the observation starts and ends
    start, end = None, None
    for m in anchors:
        a = m.value.get("anchor")
        if a == "open":
            use(m, "start")
            start = "open"
        elif a == "end_after_open":
            use(m, "end")
            end = _hhmm(570 + m.value["minutes"])
        elif a == "within":
            use(m, "end")
            end = f"minutes:{m.value['minutes']}"
        elif a == "end_at":
            use(m, "end")
            end = m.value["at"]
        elif a == "next_session":
            use(m, "end")
            f["next_session"] = True
        elif a == "prev_session" and any(c.get("key") in ("prev_return", "move", "up_streak", "down_streak", "gap") for c in f["conditions"]):
            use(m, "condition")  # "falling 3% the previous trading day": when the condition was met, which it already says
        elif a == "sessions":
            use(m, "end")
            end = f"sessions:{m.value['n']}"
            if m.value.get("how") == "at":
                f["at_end"] = True
        elif a == "intraday":
            use(m, "measure")
        elif a == "close":
            use(m, "end")
            end = "close"
        elif a == "rest":
            use(m, "scope")
            f["scope"] = "rest"
    for c in clocks:
        x = c.value
        # a clock after "before/by/until" ends the observation; otherwise it starts it
        pre = r.norm.norm[max(0, c.a - 14):c.a] if r.norm else ""
        if re.search(r"\b(?:before|by|until|till|til|through|to)\s*$", pre):
            end = x["et"]
        else:
            start = x["et"] if x["et"] != "09:30" else "open"
            f["start_stated"] = x.get("stated")
        use(c, "end" if end == x["et"] else "start")
        if x.get("assumed_tz"):
            f["assumptions"].append({"slot": "time", "text": f"\"{x['stated'].split(' (')[0]}\" had no time zone; it was read as {x['assumed_tz']} time = {x['et']} New York."})
    # the first candle as a condition: the observation starts when the candle has closed
    fc = [c for c in f["conditions"] if c["key"] == "first_candle"]
    for c in fc:
        if not c.get("minutes"):
            c["minutes_options"] = [5, 15, 30]
            f["assumptions"].append({"slot": "candle", "text": "The candle's length was not stated, so 5-, 15- and 30-minute candles are all measured."})
        if c.get("direction") is None:
            # "after the opening candle": the candle's direction is what the outcome is relative to
            c["direction"] = "any"
        if start in (None, "open") and c.get("minutes"):
            start = _hhmm(570 + c["minutes"])
    f["start"] = start or ("signal" if any(c["key"] in ("from_open", "change", "vwap", "vwap_cross") for c in f["conditions"]) else "open")
    f["end"] = end or "close"
    if f.get("next_session") and f.get("outcome", {}).get("key") == "close_vs" and f["outcome"].get("ref") == "open" and f["outcome"].get("next_session") is None:
        pass
    o = f.get("outcome") or {}
    if o.get("key") == "close_vs" and o.get("next_session"):
        f["next_session"] = True
    if o.get("key") == "close_vs" and o.get("ref") == "auto":
        # "is higher 5 days later": than where it was when the condition was met - the close of a day-level condition, else the start
        daily = any(c.get("key") in ("prev_return", "up_streak", "down_streak", "vix") for c in f["conditions"]) or f.get("next_session")
        o["ref"] = "prev_close" if daily else "start"
    if f.get("at_end") and o.get("key") == "touch":
        o["at_end"] = True  # "up more than 2% five sessions after": where it is then, not whether it got there
        if o.get("ref") in (None, "start") and any(c.get("key") in ("prev_return", "up_streak", "down_streak") for c in f["conditions"]):
            o["ref"] = "prev_close"  # ... and against where it was when the day-level condition was met: that day's close

    # what was not said, and is therefore assumed, is said (the classic reader listed these as "assumed" too)
    if o.get("key") == "touch" and not o.get("at_end") and not any(m.used and m.role == "start" for m in anchors + clocks):
        f["assumptions"].append({"slot": "start", "text": f"The {abs(o['value']):g}% is measured from " + ("the open of the session" if (f.get("start") in (None, "open")) else "where the price was when the condition was met") + "."})
    if o and (f.get("end") in (None, "close")) and not any(m.used and m.role == "end" for m in anchors + clocks) and kind == "frequency" and o.get("key") != "event":
        f["assumptions"].append({"slot": "end", "text": "It is counted until that session's close."})
    # required slots
    if kind == "frequency":
        if not o:
            r.clarify = {"slot": "outcome", "text": "What should be counted? I found " + (_describe_conditions(f["conditions"]) or "no event") + ", but nothing to measure after it.",
                         "options": [{"label": "closes above its open", "patch": {"outcome": {"key": "close_vs", "ref": "open", "direction": "up"}}},
                                     {"label": "closes above the previous close", "patch": {"outcome": {"key": "close_vs", "ref": "prev_close", "direction": "up"}}},
                                     {"label": "rises 1% from the open at some point", "patch": {"outcome": {"key": "touch", "value": 1.0, "ref": "start"}}}]}
        elif o.get("key") in ("recover", "reverse", "continue"):
            word = o["text"]
            fall = next((c for c in f["conditions"] if c.get("op") in ("<", "<=") or c.get("direction") == "down"), None)
            if o["key"] == "reverse" and fc:
                # reversing after a candle: the rest of the session goes the other way. One reading, said.
                f["outcome"] = {"key": "close_vs", "ref": "start", "direction": "opposite", "text": word}
                f["assumptions"].append({"slot": "outcome", "text": f"\"{word}\" is read as: from the candle's close to the session's close, the price moves the other way from the candle."})
            elif not f["conditions"]:
                r.clarify = {"slot": "outcome", "text": f"\"{word.capitalize()}\" after what? For example \"after a 1% down day\" or \"after it falls 1% from the open\"."}
            else:
                d = ("up" if fall else "down") if o["key"] != "continue" else ("down" if fall else "up")
                opts = readings_of(f["conditions"], d)
                f["outcome"] = {**opts[0][1], "text": word}
                f["outcome_options"] = [{"label": lab, "outcome": {**oc, "text": word}} for lab, oc in opts]
                f["assumptions"].append({"slot": "outcome", "text": f"\"{word}\" can mean more than one thing, so each is measured: " + "; ".join(lab for lab, _ in opts) + "."})
        # an X% move with no anchor and an outcome that needs the same session: ask which
    if kind == "movement" and not f["conditions"] and not f.get("measure"):
        f["measure"] = "abs_move" if re.search(r"how much|move|range", r.text.lower()) else None
    _claim_leftovers(r)
    return r


def readings_of(conds: list[dict], d: str) -> list[tuple[str, dict]]:
    """What "recover" (d="up") or "keep falling" (d="down") can mean after these conditions, most usual first."""
    keys = {c.get("key") for c in conds}
    above = "above" if d == "up" else "below"
    if "gap" in keys:
        return [("it trades back to the previous close (the gap fills)", {"key": "touch_level", "level": "prev_close", "direction": d}),
                (f"it closes {above} the previous close", {"key": "close_vs", "ref": "prev_close", "direction": d}),
                (f"it closes {above} its open", {"key": "close_vs", "ref": "open", "direction": d})]
    if keys & {"from_open", "change", "first_candle", "vwap", "vwap_cross"}:
        return [("it gets back to the session's open", {"key": "touch_level", "level": "open", "direction": d}),
                (f"it closes {above} where it was when the condition was met", {"key": "close_vs", "ref": "start", "direction": d})]
    return [(f"the next session closes {above} the previous close", {"key": "close_vs", "ref": "prev_close", "direction": d}),
            (f"the next session closes {above} its own open", {"key": "close_vs", "ref": "open", "direction": d})]


def _agreement(r: Reading, events, outcomes, anchors, clocks, use) -> Reading:
    f = r.frame
    preds = [m for m in events if m.value.get("key") == "first_candle"]
    prevdir = [m for m in events if m.value.get("key") == "prev_return" and m.value.get("value") == 0.0 and m.value.get("size") is None]
    if preds:
        p = preds[0]
        use(p, "predictor")
        f["predictor"] = "first_candle"
        mins = p.value.get("minutes")
        f["candle"] = [mins] if mins else [5, 15, 30]
        f["candle_stated"] = bool(mins)
        if not mins:
            f["assumptions"].append({"slot": "candle", "text": "The candle's length was not stated, so 5-, 15- and 30-minute candles are all measured."})
        rest = [m for m in events if m is not p]
    elif re.search(r"\byesterday|previous (?:day|session)|the day before\b", r.text.lower().replace("'", "")) and re.search(r"\b(?:direction|predict|today)", r.text.lower()):
        f["predictor"] = "prev_direction"
        for m in prevdir:
            use(m, "predictor")
        rest = [m for m in events if m not in prevdir]
        # the words "yesterday's direction" are claimed by the predictor
        for m in [x for x in outcomes if x.value["key"] in ("direction_word", "direction")]:
            use(m, "outcome")
    else:
        r.clarify = {"slot": "predictor", "text": "What should do the predicting? For example the first 5-minute candle, or the previous session's direction.",
                     "options": [{"label": "the first 5-minute candle", "patch": {"predictor": "first_candle", "candle": [5]}},
                                 {"label": "the first 15-minute candle", "patch": {"predictor": "first_candle", "candle": [15]}},
                                 {"label": "the previous session's direction", "patch": {"predictor": "prev_direction"}}]}
        rest = events
    scope = None
    for m in anchors:
        if m.value.get("anchor") in ("rest", "day"):
            use(m, "scope")
            scope = "rest" if m.value["anchor"] == "rest" else "day"
        elif m.value.get("anchor") in ("open", "close"):
            use(m, "scope")
    for m in outcomes:
        if m.value["key"] in ("direction_word", "direction", "close_vs") or (m.value["key"] == "measure" and m.value.get("measure") in ("open_to_close", "return")):
            use(m, "outcome")
            if m.value["key"] == "close_vs" and scope is None:
                scope = "day"
    for c in clocks:
        if c.value["et"] == "09:30":
            use(c, "predictor")
    f["scope"] = scope or ("both" if f.get("predictor") == "first_candle" else "day")
    if f.get("predictor") == "first_candle" and scope is None:
        f["assumptions"].append({"slot": "scope", "text": "\"The day\" is measured twice: the whole session (which contains the candle) and the rest of it after the candle closes."})
    f["conditions"] = _condition_events(r, rest, use)
    _vague_sizes(r, f["conditions"])
    _claim_leftovers(r)
    return r


def _timing(r: Reading, events, outcomes, anchors, clocks, use, ms) -> Reading:
    f = r.frame
    meas = None
    for m in outcomes:
        if m.value["key"] == "extreme_time":
            use(m, "measure")
            meas = f"{m.value['which']}_time"
        elif m.value["key"] in ("abs_move", "direction_word"):
            use(m, "measure")
            meas = meas or "abs_move"
        elif m.value["key"] == "measure":
            use(m, "measure")
    f["measure"] = meas or "abs_move"
    if not meas:
        f["assumptions"].append({"slot": "measure", "text": "Measured as the average size of the move in each part of the session."})
    bucket = 30
    for m in anchors:
        if m.value.get("anchor") in ("end_after_open", "within") and m.value.get("minutes") in (5, 10, 15, 30, 60):
            use(m, "bucket")
            bucket = m.value["minutes"]
    low = r.text.lower()
    bm = re.search(r"\b(5|10|15|30|60)[- ]?min(?:ute)?s?\b|\bhalf[- ]?hour\b|\bhour(?:ly)?\b", low)
    if bm:
        bucket = int(bm.group(1)) if bm.group(1) else (30 if "half" in bm.group(0) else 60)
    f["bucket"] = bucket
    f["conditions"] = _condition_events(r, events, use)
    for m in anchors:
        if m.value.get("anchor") in ("intraday", "open", "close"):
            use(m, "measure")
    _claim_leftovers(r, also=("anchor",))
    return r


def _breakdown(r: Reading, events, outcomes, anchors, clocks, use, ms) -> Reading:
    f = r.frame
    low = r.text.lower()
    by = "weekday" if re.search(r"week ?day|day of the week|days of the week|monday|which day|weekdays", low) else "month" if re.search(r"\bmonths?\b", low) else \
        "prev_return_decile" if re.search(r"decile", low) else "weekday"
    f["by"] = by
    meas = None
    for m in outcomes:
        if m.value["key"] == "measure":
            use(m, "measure")
            meas = m.value["measure"]
        elif m.value["key"] == "close_vs":
            use(m, "measure")
            meas = "close>open" if m.value.get("ref") == "open" and m.value.get("direction") == "up" else meas
        elif m.value["key"] in ("direction_word",):
            use(m, "measure")
    # the weekday names are the breakdown, not a condition
    wk = [m for m in events if m.value.get("key") == "weekday"]
    for m in wk:
        use(m, "by")
    if len(wk) == 1:
        f["focus"] = wk[0].value["value"]  # "is Monday a bad day": the answer is about Monday, against the other days
    f["measure"] = meas or "open_to_close"
    if not meas:
        f["assumptions"].append({"slot": "measure", "text": "\"Best\" is measured as the average move from the open to the close."})
    if by == "prev_return_decile":
        r.flag = "Breaking results down by deciles of the previous day's return is not available in the question reader yet; weekday and month breakdowns are."
    f["conditions"] = _condition_events(r, [e for e in events if e not in wk], use)
    for m in anchors:
        use(m, "measure")
    _claim_leftovers(r)
    return r


def _claim_leftovers(r: Reading, also: tuple = ()) -> None:
    """What carried meaning and found no slot. Numbers, times and events that were not used make the reading incomplete."""
    for m in r.mentions:
        if m.used:
            continue
        if m.kind in ("event", "outcome", "clock", "period", "explore") or m.kind in also:
            r.unclaimed.append(m)
        elif m.kind == "anchor" and m.value.get("anchor") in ("end_after_open", "within", "end_at", "next_session"):
            r.unclaimed.append(m)
    r.unread = unread_words(r.norm, r.mentions) if r.norm else []
    r.frame["unread"] = [u["text"] for u in r.unread]
    if r.unclaimed and r.clarify is None and r.flag is None:
        parts = "; ".join(f"\"{m.text}\"" for m in r.unclaimed[:3])
        r.clarify = {"slot": "unclaimed", "text": f"I could not place {parts} in the question, so nothing was run. " + _repair_hint(r),
                     "options": []}


def _repair_hint(r: Reading) -> str:
    k = r.frame.get("kind")
    return {"frequency": "Try saying it as one event and one thing to count, e.g. \"How often does SPY close above its open after a 2% gap down?\"",
            "agreement": "Try e.g. \"Does the first 15-minute candle predict the rest of the day for QQQ?\"",
            "movement": "Try e.g. \"What happens in the first hour after QQQ gaps down 2%?\"",
            "timing": "Try e.g. \"When during the day does SPY usually make its high?\"",
            "breakdown": "Try e.g. \"Which weekday has the best open-to-close return for IWM?\""}.get(k or "", "")


# --------------------------------------------------------------------------------------------------------- descriptions
WEEK = {"Mon": "Mondays", "Tue": "Tuesdays", "Wed": "Wednesdays", "Thu": "Thursdays", "Fri": "Fridays"}


def describe_condition(c: dict) -> str:
    k, op, v = c.get("key"), c.get("op"), c.get("value")
    vals = c.get("values")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    size = (" or ".join(f"{abs(x):g}%" for x in vals if num(x)) if vals else f"{abs(v):g}%" if num(v) else "") if (vals or v) else ""
    if k == "prev_return":
        if not vals and not v:
            return "the previous session closed " + ("down" if op in ("<", "<=") else "up")
        return f"the previous session {'fell' if op in ('<', '<=') else 'rose'} {size} or more"
    if k == "gap":
        if not vals and not v:
            return "it opened " + ("below" if op in ("<", "<=") else "above") + " the previous close"
        return f"it opened {size} or more {'below' if op in ('<', '<=') else 'above'} the previous close"
    if k == "from_open":
        if not vals and not v:
            return "it traded " + ("below" if op in ("<", "<=") else "above") + " its open"
        return f"it {'fell' if op in ('<', '<=') else 'rose'} {size} from its open"
    if k == "change":
        if not v:
            return "it was " + ("down" if op in ("<", "<=") else "up") + " on the day (against the previous close)"
        return f"it was {'down' if op in ('<', '<=') else 'up'} {size} or more on the day (against the previous close)"
    if k == "first_candle":
        mins = c.get("minutes")
        d = c.get("direction")
        return f"the first {mins}-minute candle" + (f" closed {'up' if d == 'up' else 'down'}" if d in ("up", "down") else "") if mins else ("the opening candle" + (f" closed {'up' if d == 'up' else 'down'}" if d in ("up", "down") else ""))
    if k == "weekday":
        return f"on {WEEK.get(v, v)}"
    if k == "vix":
        if c.get("when") == "now":
            return f"the VIX is {'above' if op in ('>', '>=') else 'below'} {v:g} at the time"
        return f"the VIX closed {'above' if op in ('>', '>=') else 'below'} {v:g} the session before"
    if k == "prev_otc":
        if not v:
            return "the previous session closed " + ("below its open (a red candle)" if op in ("<", "<=") else "above its open (a green candle)")
        return f"the previous session {'fell' if op in ('<', '<=') else 'rose'} {abs(v):g}% or more from its open to its close"
    if k in ("up_streak", "down_streak"):
        return f"{v:g} or more {'up' if k == 'up_streak' else 'down'} sessions in a row"
    if k == "vwap":
        return f"the price is {'above' if op == '>' else 'below'} VWAP"
    if k == "vwap_cross":
        return f"the price crosses {'back above' if op == 'above' else 'below'} VWAP"
    return c.get("text") or k or ""


def _describe_conditions(conds: list[dict]) -> str:
    return " and ".join(describe_condition(c) for c in conds)


def describe_outcome(o: dict | None, start: str = "open") -> str:
    if not o:
        return ""
    k = o.get("key")
    if k == "close_vs":
        ref = {"open": "its open", "prev_close": "the previous session's close", "start": f"the price at {start if start not in ('open', 'signal') else 'the start'}"}.get(o.get("ref"), "its open")
        d = o.get("direction")
        if d == "opposite":
            return "the rest of the session moves the other way from the candle"
        return f"it closes {'above' if d == 'up' else 'below'} {ref}"
    if k == "touch":
        ref = "its open" if o.get("ref") == "open" else "the previous session's close" if o.get("ref") == "prev_close" else "where it started" if start not in ("open",) else "its open"
        if o.get("at_end"):
            return f"it is {abs(o['value']):g}% or more {'above' if o['value'] > 0 else 'below'} {ref}"
        return f"it trades {abs(o['value']):g}% {'above' if o['value'] > 0 else 'below'} {ref} at some point"
    if k == "touch_level":
        return {"prev_close": "it trades back to the previous close (the gap fills)", "prev_low": "it trades below the previous session's low", "prev_high": "it trades above the previous session's high",
                "open": "it gets back to the session's open"}.get(o.get("level"), "it reaches the level")
    if k == "event":
        return describe_condition(o.get("event") or {})
    if k == "end_return":
        return "where it ends up"
    return o.get("text") or ""


def describe(frame: dict) -> str:
    """The frame as one plain sentence: what will be measured. Shown above every answer."""
    k = frame.get("kind")
    sym = frame.get("symbol") or "SPY"
    syms = frame.get("symbols") or [sym]
    who = " and ".join(syms) if len(syms) > 1 else sym
    conds = _describe_conditions(frame.get("conditions") or [])
    per = _describe_period(frame.get("period"))
    if k == "frequency":
        o = frame.get("outcome") or {}
        until = frame.get("end")
        tail = "" if until in (None, "close") else f" before {until} New York time" if re.match(r"\d", str(until)) else f" within {str(until).split(':')[1]} minutes" if str(until).startswith("minutes:") else ""
        if str(until).startswith("sessions:"):
            nn = int(str(until).split(":")[1])
            tail = (f" {nn} session{'s' if nn != 1 else ''} later (at the close)" if frame.get("at_end") or (frame.get("outcome") or {}).get("key") == "close_vs" else
                    f" within {nn} session{'s' if nn != 1 else ''}")
        if o.get("key") == "event":
            return f"How often {describe_outcome(o)} on {who}{per}."
        if frame.get("next_session"):
            return f"On {who}, after sessions where {conds or 'nothing in particular'}: how often the next session {describe_outcome(o).removeprefix('it ')}{tail}{per}."
        return f"On {who}{(', when ' + conds) if conds else ''}: how often {describe_outcome(o, frame.get('start') or 'open')}{tail}{per}."
    if k == "agreement":
        if frame.get("predictor") == "prev_direction":
            return f"On {who}: how often the session goes the same way as the previous session{(', when ' + conds) if conds else ''}{per}."
        c = frame.get("candle") or [5, 15, 30]
        cl = (f"the first {c[0]}-minute candle" if len(c) == 1 else "the first candle (" + ", ".join(f"{x}" for x in c) + " minutes)")
        sc = {"rest": "the rest of the session after it", "day": "the whole session", "both": "the whole session and the rest of the session after it"}[frame.get("scope") or "both"]
        return f"On {who}: how often the direction of {cl} matches {sc}{(', when ' + conds) if conds else ''}{per}."
    if k == "movement":
        st, en = frame.get("start") or "open", frame.get("end") or "close"
        span = f"from {'the open' if st == 'open' else st} to {'the close' if en == 'close' else en}"
        return f"On {who}{(', after ' + conds) if conds else ''}: how the price moved {span}{per}."
    if k == "timing":
        what = {"high_time": "when the session's high was made", "low_time": "when the session's low was made", "abs_move": "how much the price moves"}[frame.get("measure") or "abs_move"]
        return f"On {who}: {what}, in {frame.get('bucket') or 30}-minute parts of the session{(', when ' + conds) if conds else ''}{per}."
    if k == "breakdown":
        what = {"open_to_close": "the average move from open to close", "close_to_close": "the average close-to-close move", "close>open": "how often it closes above its open",
                "return": "the average move from open to close"}.get(frame.get("measure") or "open_to_close", "the average move")
        return f"On {who}: {what}, by {'weekday' if frame.get('by') == 'weekday' else 'month'}{(', when ' + conds) if conds else ''}{per}."
    return ""


def _describe_period(p) -> str:
    if not p or (not p[0] and not p[1]):
        return ""
    a, b = p
    if a and b and a[:4] == b[:4] and a.endswith("-01-01") and b.endswith("-12-31"):
        return f" in {a[:4]}"
    if a and not b:
        return f" since {a}"
    if b and not a:
        return f" until {b}"
    return f" from {a} to {b}"


# --------------------------------------------------------------------------------------------------------- to the engine
def cond_expr(c: dict) -> tuple[str, str] | None:
    """A frame condition as a rule-language expression the study engine evaluates, and its label. None: it is not one."""
    k, op, v = c.get("key"), c.get("op"), c.get("value")
    if k == "prev_return":
        return f"previous_session.return_pct {op} {v:g}", describe_condition(c)
    if k == "gap":
        return f"gap_pct {op} {v:g}", describe_condition(c)
    if k == "from_open":
        return f"from_open_pct {op} {v:g}", describe_condition(c)
    if k == "change":
        return f"change_pct {op} {v:g}", describe_condition(c)
    if k == "vix":
        # the previous session's VIX close, which is what the reading says ("the VIX closed above 25 the session before"). Until
        # V9 this compiled to `vix`, the level at the moment of entry (the day's own VIX open at 09:30): an independent
        # recalculation found the reading and the run disagreeing on 50 QQQ sessions. "VIX is above 25 at 10:00" says when.
        return (f"vix {op} {v:g}" if c.get("when") == "now" else f"vix_prev_close {op} {v:g}"), describe_condition(c)
    if k == "prev_otc":
        return f"previous_session.open_to_close_pct {op} {v:g}", describe_condition(c)
    if k == "up_streak":
        return f"consecutive_up_days() >= {int(v)}", describe_condition(c)
    if k == "down_streak":
        return f"consecutive_down_days() >= {int(v)}", describe_condition(c)
    if k == "vwap":
        return f"price {op} vwap", describe_condition(c)
    if k == "vwap_cross":
        return ("crosses_above(price, vwap)" if op == "above" else "crosses_below(price, vwap)"), describe_condition(c)
    if k == "first_candle" and c.get("direction") in ("up", "down") and c.get("minutes"):
        return ("from_open_pct > 0" if c["direction"] == "up" else "from_open_pct < 0"), describe_condition(c)
    return None


def expand(frame: dict) -> list[tuple[str, dict]]:
    """A frame with several values for one slot (sizes of a vague condition, candle lengths, symbols) as plain frames."""
    out = [("", frame)]
    for i, c in enumerate(frame.get("conditions") or []):
        if c.get("values"):
            nxt = []
            for lab, fr in out:
                for v in c["values"]:
                    f2 = {**fr, "conditions": [dict(x) for x in fr["conditions"]]}
                    f2["conditions"][i] = {**{k: x for k, x in c.items() if k != "values"}, "value": v}
                    nxt.append(((lab + ", " if lab else "") + describe_condition(f2["conditions"][i]), f2))
            out = nxt
        if c.get("minutes_options"):
            nxt = []
            for lab, fr in out:
                for mm in c["minutes_options"]:
                    f2 = {**fr, "conditions": [dict(x) for x in fr["conditions"]]}
                    f2["conditions"][i] = {**{k: x for k, x in c.items() if k != "minutes_options"}, "minutes": mm}
                    f2["start"] = _hhmm(570 + mm) if (fr.get("start") in (None, "open")) else fr.get("start")
                    nxt.append(((lab + ", " if lab else "") + f"{mm}-minute candle", f2))
            out = nxt
    if frame.get("outcome_options"):
        out = [((lab + ", " if lab else "") + o["label"], {**fr, "outcome": o["outcome"]}) for lab, fr in out for o in frame["outcome_options"]]
    syms = frame.get("symbols") or []
    if len(syms) > 1:
        out = [((lab + ", " if lab else "") + s, {**fr, "symbol": s, "symbols": [s]}) for lab, fr in out for s in syms]
    return out


def study_draft(frame: dict):
    """A frequency or movement frame (one value in every slot) as the study draft the existing pipeline runs."""
    from strata.ai.draft import Draft, DraftCondition, DraftLevel, DraftStudy
    conds, weekdays, entry = [], None, None
    for c in frame.get("conditions") or []:
        if c.get("key") == "weekday":
            weekdays = sorted(set((weekdays or []) + [c["value"]]), key=["Mon", "Tue", "Wed", "Thu", "Fri"].index)
            continue
        if c.get("key") == "first_candle" and c.get("direction") == "any":
            continue  # the candle's own direction is part of the outcome ("reverses"): handled below
        e = cond_expr(c)
        if e:
            conds.append(DraftCondition(expr=e[0], text=e[1]))
    start = frame.get("start") or "open"
    entry = None if start == "signal" else ("open" if start in ("open", "09:30") else start)
    o = frame.get("outcome") or {}
    st = DraftStudy()
    k = o.get("key")
    if k == "close_vs":
        st.measure = "close"
        st.target = DraftLevel(kind="pct", value=0.0)
        st.direction = "up" if o.get("direction") in ("up", None) else "down"
        st.reference = {"open": "open", "prev_close": "prev_close", "start": "entry"}.get(o.get("ref") or "open", "open")
        if o.get("ref") == "start" and entry == "open":
            st.reference = "entry"
    elif k == "touch":
        st.measure = "close" if o.get("at_end") else "touch"
        st.target = DraftLevel(kind="pct", value=float(o["value"]))
        st.direction = "up" if o["value"] > 0 else "down"
        st.reference = "open" if o.get("ref") == "open" else "prev_close" if o.get("ref") == "prev_close" else "entry"
    elif k == "touch_level":
        st.measure = "touch"
        lvl = {"prev_close": "prev_close", "prev_low": "prev_low", "prev_high": "prev_high", "open": "open"}[o["level"]]
        st.target = DraftLevel(kind="expr", value=0.0, expr=lvl)
        st.direction = o.get("direction")
    elif k == "event":
        # "how often does IWM gap up more than 1%": counted on every session; the event is the outcome
        ev = o.get("event") or {}
        e = cond_expr({**ev, "key": "prev_return" if ev.get("key") == "day_return" else ev.get("key")})
        if ev.get("key") == "day_return":
            st.measure = "close"
            st.reference = "prev_close"
            st.target = DraftLevel(kind="pct", value=float(ev["value"]))
            st.direction = "up" if ev["value"] > 0 else "down"
        elif ev.get("key") == "gap":
            # the gap is known at the open: a level the open itself is beyond. Counted at the opening tick.
            st.measure = "close"
            st.reference = "prev_close"
            st.target = DraftLevel(kind="pct", value=float(ev["value"]))
            st.direction = "up" if ev["value"] > 0 else "down"
            st.horizon_kind, st.horizon_time = "time", "09:31"
        else:
            st.target = None
            if e:
                conds.append(DraftCondition(expr=e[0], text=e[1]))
    else:  # movement: no target, the end return is measured
        st.measure = "close"
        st.target = None
    end = frame.get("end") or "close"
    if frame.get("next_session") or (o.get("next_session")):
        # the condition is about the session before: the observation is the next session, measured over its whole length
        pass
    if isinstance(end, str) and re.match(r"\d\d:\d\d", end):
        st.horizon_kind, st.horizon_time = "time", end
    elif isinstance(end, str) and end.startswith("minutes:"):
        mins = int(end.split(":")[1])
        base = 570 if entry in (None, "open") else int(entry[:2]) * 60 + int(entry[3:])
        st.horizon_kind, st.horizon_time = "time", _hhmm(min(base + mins, 960))
    elif isinstance(end, str) and end.startswith("sessions:"):
        st.horizon_kind, st.horizon_sessions = "sessions", int(end.split(":")[1])
    p = frame.get("period") or [None, None]
    title = describe(frame).rstrip(".")[:120]
    d = Draft(kind="study", title=title, symbol=frame.get("symbol") or "SPY", symbol_stated=bool(frame.get("symbol_stated")), conditions=conds, weekdays=weekdays,
              entry_time=entry, entry_time_stated=frame.get("start_stated") if entry not in (None, "open") or frame.get("start_stated") else None,
              study=st, start=date.fromisoformat(p[0]) if p and p[0] else None, end=date.fromisoformat(p[1]) if p and len(p) > 1 and p[1] else None,
              notes=[a["text"] for a in frame.get("assumptions") or []][:6])
    # "the rest of the session moves the other way from the candle": two studies, one per candle direction (an agreement)
    return d


def agreement_plan(frame: dict) -> dict:
    """An agreement frame as the investigation the existing machinery runs: up and down studies per candle and scope."""
    from strata.ai import research as RS
    extra = [cond_expr(c) for c in frame.get("conditions") or [] if c.get("key") not in ("weekday",)]
    extra = [e for e in extra if e]
    weekdays = [c["value"] for c in frame.get("conditions") or [] if c.get("key") == "weekday"] or None
    sym = frame.get("symbol") or "SPY"
    if frame.get("predictor") == "prev_direction":
        from strata.ai.draft import Draft, DraftCondition, DraftLevel, DraftStudy
        variants, pairs = [], []
        for d_, op in (("up", ">"), ("down", "<")):
            conds = [DraftCondition(expr=f"previous_session.return_pct {op} 0", text=f"the previous session closed {d_}")] + [DraftCondition(expr=e, text=t) for e, t in extra]
            st = DraftStudy(target=DraftLevel(kind="pct", value=0.0), measure="close", direction=d_, reference="open")
            lab = f"previous session {d_}: session closes {'above' if d_ == 'up' else 'below'} its open"
            variants.append({"label": lab, "draft": Draft(kind="study", symbol=sym, symbol_stated=True, title=f"{sym}: {lab}", conditions=conds, entry_time="open", study=st, weekdays=weekdays)})
        pairs.append({"minutes": 0, "scope": "day", "from": "09:30", "up": variants[0]["label"], "down": variants[1]["label"], "predictor": "prev_direction"})
        p = {"question": frame["text"], "symbol": sym, "kind": "agreement", "predictor": "prev_direction", "pairs": pairs, "variants": variants,
             "hypothesis": "The direction of the previous session and the direction of the next session agree more often than unrelated directions would.",
             "method": f"On every {sym} session: whether the session closed above or below its open, against whether the previous session closed up or down.",
             "measure": "this happened", "checks": ["baseline", "significance", "out_of_sample"], "interpretation_notes": [a["text"] for a in frame.get("assumptions") or []]}
    else:
        # "the day" is measured twice - open to close, which contains the candle, and from the candle's close on, which does not -
        # because the first number alone overstates what the candle says about anything still unknown when it closes
        fc = {"minutes": list(frame.get("candle") or [5, 15, 30]), "stated": bool(frame.get("candle_stated")), "scope": "rest" if frame.get("scope") == "rest" else "both"}
        p = RS.first_candle_plan(frame["text"], sym, fc)
        p["kind"] = "first_candle"
        if extra or weekdays:
            from strata.ai.draft import DraftCondition
            for v in p["variants"]:
                d = v["draft"]
                d.conditions = list(d.conditions) + [DraftCondition(expr=e, text=t) for e, t in extra]
                if weekdays:
                    d.weekdays = weekdays
            p["method"] += " Only sessions where " + _describe_conditions(frame.get("conditions") or []) + " are counted."
        own = [a["text"] for a in frame.get("assumptions") or []]
        stated = any("not stated" in x for x in own)
        p["interpretation_notes"] = own + [x for x in p.get("interpretation_notes") or [] if not (stated and "was not stated" in x)]
    p["frame"] = {k: v for k, v in frame.items() if k != "text"}
    p["period"] = frame.get("period")
    if frame.get("period") and any(frame["period"]):
        a, b = frame["period"]
        for v in p["variants"]:
            v["draft"].start = date.fromisoformat(a) if a else None
            v["draft"].end = date.fromisoformat(b) if b else None
    return p


def to_route(r: Reading) -> dict:
    """The action the agent takes for a reading that is complete (r.ok)."""
    f = r.frame
    k = f.get("kind")
    if k == "agreement":
        return {"action": "investigate", "plan": agreement_plan(f), "frame": f}
    if k in ("frequency", "movement"):
        parts = expand(f)
        if f.get("outcome", {}).get("direction") == "opposite":
            # "reverses after a red opening candle": the rest of the session against the candle's direction
            parts2 = []
            for lab, fr in parts:
                c = next((c for c in fr["conditions"] if c.get("key") == "first_candle"), None)
                dirs = [c["direction"]] if c and c.get("direction") in ("up", "down") else ["up", "down"]
                for d_ in dirs:
                    fr2 = {**fr, "conditions": [({**x, "direction": d_} if x is c else x) for x in fr["conditions"]], "outcome": {"key": "close_vs", "ref": "start", "direction": "down" if d_ == "up" else "up"}}
                    parts2.append(((lab + ", " if lab else "") + f"{'up' if d_ == 'up' else 'down'} candle", fr2))
            parts = parts2
        if len(parts) == 1:
            return {"action": "new", "draft": study_draft(parts[0][1]), "frame": f}
        variants = [{"label": lab, "draft": study_draft(fr)} for lab, fr in parts]
        plan = {"question": f["text"], "symbol": f.get("symbol"), "kind": f"{k}_set", "variants": variants, "frame": {kk: v for kk, v in f.items() if kk != "text"},
                "hypothesis": describe(f), "method": "Each variant below is measured the same way on the stored history; " + "; ".join(a["text"] for a in f.get("assumptions") or [])[:400],
                "measure": describe_outcome(f.get("outcome")), "checks": ["baseline", "significance"], "interpretation_notes": [a["text"] for a in f.get("assumptions") or []]}
        return {"action": "investigate", "plan": plan, "frame": f}
    if k == "timing":
        return {"action": "profile", "frame": f}
    if k == "breakdown":
        return {"action": "profile", "frame": f}
    return {"action": "unknown"}


# ----------------------------------------------------------------------------------------------- checking strategies
HOLDS = {"prev_return": ("previous_session", "return_pct", "prev_return", "previous day", "previous session", "prior day", "change_pct", "prev_close", "day_return", "drop", "decline", "fell"),
         "gap": ("gap",), "from_open": ("from_open", "opening", "below its open", "above its open", "from the open"), "vix": ("vix", "volx", "volatility"),
         "weekday": ("weekday", "monday", "tuesday", "wednesday", "thursday", "friday"), "up_streak": ("consecutive", "streak", "in a row"), "down_streak": ("consecutive", "streak", "in a row"),
         "first_candle": ("from_open", "opening_range", "candle", "first_", "first "), "vwap": ("vwap",), "vwap_cross": ("vwap",), "change": ("change_pct", "on the day", "previous close", "prev_close")}
HOLDS["move"] = HOLDS["prev_return"] + HOLDS["from_open"] + ("underlying_target", "underlying_stop", "target", "rise", "rally", "up ")


def holds(serial: str, key: str | None) -> bool:
    """Does a reading (as lower-case text: its expressions and descriptions) hold an event of this kind?"""
    return key != "unsupported_event" and (key not in HOLDS or any(x in serial for x in HOLDS[key]))


def strategy_coverage(text: str, draft, known: list[str]) -> list[str]:
    """What a strategy sentence said that the strategy readers' draft does not hold. Each entry is quoted from the sentence.

    Checked: every event, every other stored symbol, every clock time, every percentage, an exit that was stated, and an
    order of events ("first ... then ..."). Anything listed here stops the run until the person says to go on without it."""
    n, ms = tag(text, known)
    low = n.norm
    missing = []
    conds = " ".join(c.expr for c in draft.conditions) + " " + " ".join(s.expr for s in draft.sequence) + " " + " ".join(c.expr for c in draft.exit_conditions)
    ctexts = " ".join((c.text or "") for c in list(draft.conditions) + list(draft.sequence) + list(draft.exit_conditions))
    serial = (conds + " " + ctexts + (" underlying_target" if draft.underlying_target_pct else "") +
              (" underlying_stop" if draft.underlying_stop_pct else "") + (" weekday " + " ".join(map(str, draft.weekdays)) if draft.weekdays else "")).lower()
    for m in ms:
        if m.kind == "event":
            k = m.value.get("key")
            if not holds(serial, k) and m.text.strip() not in " ".join(draft.unsupported):
                missing.append(m.text)
        elif m.kind == "symbol" and m.value.get("stored") and m.value["symbol"] != draft.symbol and not re.search(rf"\b{m.value['symbol']}\.", conds) and \
                not (m.value["symbol"] == "VIX" and "vix" in conds.lower()):
            missing.append(m.text)
        elif m.kind == "clock":
            et = m.value.get("et")
            placed = {draft.entry_time, draft.exit_time} | set(re.findall(r"\d\d:\d\d", conds + " " + ctexts))
            if draft.entry_time == "open":
                placed.add("09:30")
            if draft.exit_time == "close":
                placed |= {"16:00", "15:59", "15:55"}
            if et and et not in placed:
                missing.append(m.text)
    # every percentage has a place: a target, a stop, a condition, a partial exit
    nums = set()
    for v in (draft.premium_target_pct, draft.premium_stop_pct, draft.underlying_target_pct, draft.underlying_stop_pct, draft.width):
        if v is not None:
            nums.add(round(abs(float(v)), 6))
    sz = getattr(draft, "sizing", None)
    if sz is not None and sz.mode in ("risk_pct", "equity_pct"):
        nums.add(round(abs(float(sz.value)), 6))  # "risk 2% of the account": the sizing's own percentage
    for x in re.findall(r"-?\d+(?:\.\d+)?", conds):
        nums.add(round(abs(float(x)), 6))
    for pt in getattr(draft, "partials", []) or []:
        for v in pt.model_dump().values() if hasattr(pt, "model_dump") else []:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums.add(round(abs(float(v)), 6))
    for mm in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?) ?%", low):
        v = round(float(mm.group(1)), 6)
        if v not in nums and not any(abs(v - x) < 1e-6 for x in nums):
            a, b = n.span(mm.start(), mm.end())
            missing.append(n.text[a:b])
    # an exit that was stated but is not in the draft
    ex = re.search(r"\b(?:sell|sold|selling|exit|exiting|out|get out|close (?:it|them|the position|out)|hold (?:it |them )?(?:til|till|until|to|through)|take profits?|tp|sl|stop)\b[^.;]{0,30}", low)
    has_exit = bool(draft.exit_time or draft.hold_minutes or draft.hold_sessions or draft.premium_target_pct or draft.premium_stop_pct or draft.underlying_target_pct or draft.underlying_stop_pct
                    or draft.exit_conditions or draft.hold_to_expiration or getattr(draft, "partials", None) or draft.underlying_target_abs or draft.underlying_stop_abs)
    if ex and not has_exit:
        a, b = n.span(ex.start(), ex.end())
        missing.append(n.text[a:b].strip())
    # events in an order ("first rallies 0.3% ... then breaks below the open") need an ordered sequence
    if re.search(r"\bthen\b|\bafter it first\b|\bfirst (?:rall|drop|fall|rise|move|pop|dip)", low) and not draft.sequence and len([m for m in ms if m.kind == "event"]) >= 2:
        missing.append("the order of the events (first ..., then ...)")
    return list(dict.fromkeys(missing))
