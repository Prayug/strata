"""Offline interpreter: plain English -> Draft, with no language model.

A deterministic pattern-based parser covering the common ways traders phrase strategies
and market questions. It is the default when no LLM is configured, the fallback when an
LLM returns something invalid, and what the acceptance tests run against. It extracts
only what was said; everything else is left empty for the compiler to flag as missing or
assumed. It never produces financial results.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from strata.ai.draft import WEEKDAYS, Draft, DraftCondition, DraftLevel, DraftStrike, DraftStudy

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "fifteen": 15, "thirty": 30, "sixty": 60}
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "SPX", "NDX", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "DEMO", "DEMOX"]
ALIASES = {"s&p 500": "SPY", "s&p": "SPY", "sp500": "SPY", "nasdaq": "QQQ", "russell": "IWM", "apple": "AAPL", "nvidia": "NVDA", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN"}
TZ_OFFSET = {"pacific": 180, "pt": 180, "pst": 180, "pdt": 180, "mountain": 120, "mt": 120, "mst": 120, "mdt": 120, "central": 60, "ct": 60, "cst": 60, "cdt": 60,
             "eastern": 0, "et": 0, "est": 0, "edt": 0, "new york": 0}
TZ_NAME = {180: "Pacific", 120: "Mountain", 60: "Central", 0: "Eastern"}
DAYNAMES = {"monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu", "friday": "Fri", "mon": "Mon", "tue": "Tue", "tues": "Tue", "wed": "Wed", "thu": "Thu", "thur": "Thu", "thurs": "Thu", "fri": "Fri"}
N = r"(\d+(?:\.\d+)?)"
PREV = r"(?:previous|prior|preceding|yesterday'?s?|the day before|day before|last)"
DOWN = r"(?:fell|fall(?:s|ing|en)?|dropp?(?:ed|s|ing)?|declin\w+|lost|los(?:es|ing)|down|sold off|sell-?off|red|decreas\w+|slid|sank|tumbl\w+|moved down|was down|is down|closed down|closes down)"
UP = r"(?:rose|ris(?:es|ing|en)?|gain(?:ed|s|ing)?|rall(?:y|ied|ies)|up|green|increas\w+|climb\w+|jump\w+|moved up|was up|is up|closed up|closes up)"

EXAMPLES = [
    "How often does SPY rise 1% from the open after falling 3% or more the previous trading day?",
    "Buy SPY calls at 6:30 AM Pacific every day, targeting a 1% underlying price increase, but only when SPY fell at least 3% the previous trading day.",
    "How often does SPY touch yesterday's high before touching yesterday's low?",
    "What happens if I buy SPY calls at the close after a red Monday and hold them until Friday?",
    "Buy a 30DTE ATM SPY call at the close after a 2% down day and hold for 5 days",
    "How often does SPY recover 50% of its overnight gap before noon?",
    "Buy ATM 0DTE puts when QQQ breaks the opening 15-minute low.",
    "Test an iron condor entered at 10 AM Eastern when VIX is below 18.",
]


def _num(tok: str) -> float | None:
    tok = tok.lower().strip()
    if tok in NUM_WORDS:
        return float(NUM_WORDS[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def default_tz_offset() -> int:
    """Offset (minutes behind New York) assumed for clock times given without a timezone."""
    from strata.api.settings import get_setting
    tz = str(get_setting("ui.timezone") or "auto")
    if tz == "auto":
        import time
        tz = time.tzname[0] if time.tzname else "EST"
    t = tz.lower()
    return 180 if ("los_angeles" in t or t in ("pst", "pdt", "pacific")) else 120 if ("denver" in t or t in ("mst", "mdt")) else 60 if ("chicago" in t or t in ("cst", "cdt")) else 0


# "6:30 AM Pacific", "9:30ET", "15:30", and as people type them in a hurry: "6:30a, PST" (a letter for the half of the day,
# attached to the digits; a comma before the zone)
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?|(?<=\d)[ap](?![a-z.]))?(?:\s*,?\s*(pacific|pst|pdt|pt|mountain|mst|mdt|mt|central|cst|cdt|ct|eastern|est|edt|et|new york)(?:\s*time)?)?\b", re.I)
EXIT_CUE = re.compile(r"\b(exit|sell (?:them|it|at|when|by)|close (?:the position|it|them|out)|clos(?:e|ing) (?:at|by)|take profits?|until|till|hold(?:ing)?\b|out by|\bor at\b|get out|"
                      r"buy (?:them|it|the (?:puts?|calls?|options?|contracts?)) back|cover (?:them|it|at|by)|buy to close)", re.I)


EXIT_WHEN = re.compile(r"\b(?:exit(?:s|ing)?|sell(?:s|ing)?|clos(?:e|es|ing)|get(?:ting)?\s+out|take\s+profits?|tak(?:e|ing)\s+(?:it|them)\s+off|cut(?:ting)?\s+(?:it|them|the\s+\w+))\b(?P<mid>(?:[^.;]|\.(?=\d))*?)"
                       r"\b(?P<w>when(?:ever)?|if|once|as\s+soon\s+as)\b(?P<body>(?:[^.;]|\.(?=\d))*)", re.I)  # a decimal point does not end the sentence


def exit_clauses(text: str) -> list[tuple[int, int]]:
    """Character ranges of the conditions under which the position is CLOSED ("exit at 15:55 or when SPY has risen 1%",
    "close the trade if it reclaims VWAP"). What stands there is an exit rule. It is never also an entry condition: read as
    one, "buy after a 1% down day ... exit when SPY has risen 1%" asks for a day that fell 1% and rose 1%.

    Not an exit clause: a verb that opens the position ("sell puts when ..."), and a condition on the previous session or
    the opening gap, which is known before the entry and cannot close anything."""
    out = []
    for m in EXIT_WHEN.finditer(text):
        head = text[m.start():m.start("w")].lower()
        held = r"\b(?:it|them)\b|\b(?:the|my|every|each|all|both)\s+(?:\w+\s+)?(?:puts?|calls?|spreads?|condors?|straddles?|strangles?|contracts?|options?|positions?|trades?|lots?|rest|remainder|other half)\b"
        if re.search(r"\b(?:puts?|calls?|spreads?|condors?|straddles?|strangles?|butterfl\w+|contracts?)\b", head) and not re.search(held, head):
            continue  # "sell puts 1% out of the money when ...": the sale is the entry. "sell them when", "close the calls if" are exits
        if re.search(r"\b(?:buy|buying|enter|entering|open a|go long|go short)\b", head):
            continue  # "... sell at the close; buy when ..." never reaches here (the sentence ended), "sell and buy back when" does
        body = m.group("body")
        if re.search(rf"{PREV}|\bgaps?\b|\bgapp", body.lower()):
            continue
        # the period of the test is not part of the exit ("... when SPY has risen 1%, from 2024-01-18 to 2026-11-25")
        cut = re.search(rf",?\s*\b(?:from|between|since|starting|over|during|in|for)\s+(?:the\s+)?(?:{DATE_TOK}|(?:19|20)\d\d\b|(?:last|past)\s+\w+\s+(?:years?|months?))", body, re.I)
        out.append((m.start("w"), m.start("body") + (cut.start() if cut else len(body))))
    return out


def _session_readings(h: int, mi: int, ap: str | None) -> list[tuple[int, int]]:
    """(offset to New York, hour) readings of a time with no zone that land inside the regular session."""
    cand = []
    for o in dict.fromkeys((0, default_tz_offset(), 180)):
        hh = h if ap else (h + 12 if (h < 7 and o == 0) or (h < 6 and o == 180) else h)
        if 570 <= hh * 60 + mi + o <= 960:
            cand.append((o, hh))
    return cand


def find_times(text: str, msg_offset: int | None) -> list[dict]:
    """Clock times in the text, converted to New York time. Each: {pos, et, stated, assumed_tz}.
    Times written without a zone are all read in ONE zone: the first of (the zone set in Settings, Eastern,
    Pacific) in which every one of them falls inside the regular session. "Enter at 10:00, out by 15:30" is
    therefore never half Pacific and half Eastern."""
    out = []
    low = text.lower()
    found = []
    for m in TIME_RE.finditer(text):
        h, mi, ap, tz = m.group(1), m.group(2), m.group(3), m.group(4)
        if not (mi or ap or tz):
            continue  # a bare number is not a time
        if text[m.end():m.end() + 1] == "%" or low[m.end():m.end() + 4].lstrip().startswith(("dte", "min", "day", "delta")):
            continue
        h, mi = int(h), int(mi or 0)
        if h > 23 or mi > 59:
            continue
        if ap:
            h = (h % 12) + (12 if ap.lower().startswith("p") else 0)
        found.append((m, h, mi, ap, tz))
    pref = default_tz_offset()
    common = None
    named = {TZ_OFFSET[tz.lower()] for _m, _h, _mi, _ap, tz in found if tz and tz.lower() in TZ_OFFSET}
    if msg_offset is None and len(named) == 1:
        msg_offset = named.pop()  # "between 9:30 and 11:00 Eastern": a zone written once applies to the other times too
    if msg_offset is None:
        sets = [{o for o, _ in _session_readings(h, mi, ap)} for _m, h, mi, ap, tz in found if not tz]
        shared = set.intersection(*sets) if sets else set()
        common = next((o for o in dict.fromkeys((pref, 0, 180)) if o in shared), None)
        # 9:30 with no zone is the opening bell: nobody on the west coast calls 12:30 New York time "9:30". It is read as New
        # York time, and so is every other zone-less time in the sentence; the interpretation still says which zone it assumed.
        if 0 in shared and any((h, mi) == (9, 30) and not tz and (ap or "a").lower().startswith("a") for _m, h, mi, ap, tz in found):
            common = 0
    for m, h, mi, ap, tz in found:
        off = TZ_OFFSET.get(tz.lower()) if tz else msg_offset
        assumed = None
        if off is None:
            cand = _session_readings(h, mi, ap)
            want = common if common is not None else pref
            off, h = next((c for c in cand if c[0] == want), cand[0] if cand else (want, h))
            assumed = TZ_NAME.get(off, "Eastern")
        elif not ap:
            if off == 0 and h < 7:
                h += 12
            elif off >= 120 and h < 6:
                h += 12
        et = h * 60 + mi + off
        if not (0 <= et < 24 * 60):
            continue
        stated = m.group(0).strip()
        out.append({"pos": m.start(), "et": f"{et // 60:02d}:{et % 60:02d}", "stated": stated + (f" ({assumed} assumed)" if assumed else ""), "assumed_tz": assumed, "minutes": et})
    for pat, val in ((r"\b(?:at|on|from|after) the (?:market )?open(?:ing bell)?\b|\bat (?:market )?open\b|\bmarket open\b|\bopening bell\b", "open"),
                     (r"\b(?:at|on|by|into|near) the (?:market )?close\b|\bat (?:market )?close\b|\bmarket close\b|\bclosing bell\b|\bend of (?:the )?day\b|\beod\b", "close"),
                     (r"\bnoon\b|\bmidday\b", "12:00")):
        for m in re.finditer(pat, low):
            out.append({"pos": m.start(), "et": val, "stated": m.group(0).strip(), "assumed_tz": None, "minutes": {"open": 570, "close": 960}.get(val, 720)})
    return sorted(out, key=lambda x: x["pos"])


NOT_UNDERLYING = {"VIX", "VXN", "RVX", "DEMOVIX", "ATM", "OTM", "ITM", "DTE", "ET", "PT", "AM", "PM", "RSI", "VWAP", "IV", "EOD"}


def find_symbol(text: str, known: list[str]) -> str | None:
    known = [k for k in known if k not in NOT_UNDERLYING]
    for m in re.finditer(r"\$?\b([A-Z]{2,5})\b", text):
        if m.group(1) in known:
            return m.group(1)
    low = text.lower()
    for k, v in ALIASES.items():
        if k in low:
            return v
    for t in known:
        if re.search(rf"\b{t.lower()}\b", low):
            return t
    return None


def find_structure(low: str) -> str | None:
    short = bool(re.search(r"\b(sell|selling|short|shorting|write|writing|sold)\b", low))
    table = [
        (r"iron condor", "iron_condor"), (r"iron (?:butter)?fl(?:y|ies)", "iron_butterfly"),
        (r"straddle", "short_straddle" if short else "long_straddle"), (r"strangle", "short_strangle" if short else "long_strangle"),
        (r"put credit spread|bull put|short put spread|credit put spread", "bull_put_spread"), (r"call credit spread|bear call|short call spread|credit call spread", "bear_call_spread"),
        (r"call debit spread|bull call|long call spread|debit call spread|call spread", "bull_call_spread"), (r"put debit spread|bear put|long put spread|debit put spread|put spread", "bear_put_spread"),
        (r"covered calls?", "covered_call"), (r"cash[- ]secured puts?|csp\b", "short_put"),
    ]
    for pat, s in table:
        if re.search(pat, low):
            return s
    put = re.search(r"\bputs?\b", low)
    call = re.search(r"\bcalls?\b", low)
    if put and call:
        first = "put" if put.start() < call.start() else "call"
        return {"put": "short_put" if short else "long_put", "call": "short_call" if short else "long_call"}[first]
    # the words between the verb and the option may be hyphenated or numeric ("sell a 30-delta SPY put"); an exit verb ("sell them at noon") is not a short sale
    between = r"(?!(?:them|it|at|when|by|after|before|if|on)\b)(?:[\w.%$-]+\s+){0,6}?"
    if put:
        return "short_put" if short and re.search(rf"\b(?:sell|selling|short|shorting|write|writing)\s+{between}puts?\b", low) else "long_put"
    if call:
        return "short_call" if short and re.search(rf"\b(?:sell|selling|short|shorting|write|writing)\s+{between}calls?\b", low) else "long_call"
    return None


def find_conditions(text: str, spans: list | None = None) -> list[tuple[str, DraftCondition]]:
    """(category, condition) pairs. Category lets a follow-up replace the matching rule.

    `spans` collects the character ranges each rule was read from, so numbers inside them are
    not re-read as targets or stops.
    """
    low = text.lower()
    out: list[tuple[str, DraftCondition]] = []
    _search = re.search

    class _Re:  # records the span of every successful match made while looking for conditions
        @staticmethod
        def search(pat, string, flags=0):
            m = _search(pat, string, flags)
            if m and spans is not None and string is low:
                spans.append(m.span())
            return m

    re_ = _Re

    def add(cat, expr, txt):
        if not any(c == cat for c, _ in out):
            out.append((cat, DraftCondition(expr=expr, text=txt)))

    otc = "open_to_close" if re_.search(r"open[- ]to[- ]close|intraday (?:decline|drop|loss|gain)", low) else None
    field = "previous_session.open_to_close_pct" if otc else "previous_session.return_pct"
    basis = "open-to-close" if otc else "close-to-close"
    # previous-day move with a size
    pats = [
        (rf"{DOWN}\D{{0,30}}?{N}\s*%(?:\s*or more)?[^.;%]{{0,40}}?{PREV}", -1), (rf"{PREV}[^.;%]{{0,50}}?{DOWN}\D{{0,25}}{N}\s*%", -1),
        (rf"(?:after|following|on days after|the day after)\s+an?\s+{N}\s*%\s*(?:or more\s*)?(?:down|red|declin\w*|drop|sell-?off|loss)", -1),
        (rf"{N}\s*%(?:\s*or more|\s*\+)?\s*(?:down|red|declin\w*|drop)\s*(?:day|session)", -1), (rf"{PREV}[- ]day (?:decline|drop|loss|fall)\w*\D{{0,15}}{N}\s*%", -1),
        (rf"{UP}\D{{0,30}}?{N}\s*%(?:\s*or more)?[^.;%]{{0,40}}?{PREV}", 1), (rf"{PREV}[^.;%]{{0,50}}?{UP}\D{{0,25}}{N}\s*%", 1),
        (rf"(?:after|following|on days after|the day after)\s+an?\s+{N}\s*%\s*(?:or more\s*)?(?:up|green|gain|rally)", 1), (rf"{N}\s*%(?:\s*or more|\s*\+)?\s*(?:up|green|gain|rally)\s*(?:day|session)", 1),
    ]
    for pat, sign in pats:
        m = re_.search(pat, low)
        if m:
            x = float(m.group(1))
            # "a 2% down day" names no session explicitly; "the previous day fell 2%" does
            cat = "prev_ret" if re.search(PREV, m.group(0)) else "prev_ret_day"
            add(cat, f"{field} <= -{x:g}" if sign < 0 else f"{field} >= {x:g}", f"previous session {'fell' if sign < 0 else 'rose'} {x:g}% or more ({basis})")
            break
    # red / green weekday ("after a red Monday")
    m = re_.search(r"\b(red|down|green|up)\s+(monday|tuesday|wednesday|thursday|friday)", low)
    if m:
        k = WEEKDAYS.index(DAYNAMES[m.group(2)])
        neg = m.group(1) in ("red", "down")
        add("weekday_color", f"weekday == {k} and change_pct {'<' if neg else '>'} 0", f"{m.group(2).capitalize()} closes {'lower' if neg else 'higher'} than the prior close")
    elif not any(c.startswith("prev_ret") for c, _ in out):
        m = re_.search(rf"(?:after|following|the day after|on days after)\s+an?\s+(red|down|green|up)\s+(?:day|session|close)|{PREV}\s*(?:day|session)?\s*(?:was|is|closed)\s*(red|down|green|up|lower|higher)", low)
        if m:
            w = m.group(1) or m.group(2)
            neg = w in ("red", "down", "lower")
            add("prev_ret" if m.group(2) else "prev_ret_day", f"{field} {'<' if neg else '>'} 0", f"previous session closed {'lower' if neg else 'higher'} ({basis})")
    # streaks
    m = re_.search(r"(\w+)\s+(?:consecutive|straight|successive)?\s*(red|down|bearish|green|up|bullish)\s+(?:(\w+)[- ]?(?:minute|min)\s+)?(candles?|bars?|days?|sessions?)(?:\s+in a row)?", low)
    if m and _num(m.group(1)) and (re_.search(r"consecutive|straight|in a row|successive", low)):
        n, bear = int(_num(m.group(1))), m.group(2) in ("red", "down", "bearish")
        if m.group(4).startswith(("candle", "bar")):
            tf = f"{int(_num(m.group(3)) or 5)}m" if m.group(3) else "5m"
            add("streak", f'consecutive_{"bearish" if bear else "bullish"}_candles("{tf}") >= {n}', f"{n} consecutive {'red' if bear else 'green'} {tf} candles")
        else:
            add("streak", f"consecutive_{'down' if bear else 'up'}_days() >= {n}", f"{n} consecutive {'down' if bear else 'up'} sessions")
    m = re_.search(r"(above|over|below|under|beneath)\s+(?:the\s+)?vwap", low)
    if m:
        up = m.group(1) in ("above", "over")
        add("vwap", f"price {'>' if up else '<'} vwap", f"price {'above' if up else 'below'} session VWAP")
    m = re_.search(rf"gap(?:s|ping|ped)?\s*(down|up|lower|higher)(?:\s*(?:by\s*)?(?:at least|more than|over)?\s*{N}\s*%(?:\s*or more)?)?", low)
    if m:
        dn = m.group(1) in ("down", "lower")
        if m.group(2):
            x = float(m.group(2))
            add("gap", f"gap_pct <= -{x:g}" if dn else f"gap_pct >= {x:g}", f"opening gap {'down' if dn else 'up'} {x:g}% or more")
        else:
            add("gap", "gap_pct < 0" if dn else "gap_pct > 0", f"opens {'below' if dn else 'above'} the previous close")
    if re_.search(rf"reclaims?\s+(?:the\s+)?{PREV}\s*(?:day'?s?)?\s*close", low):
        add("reclaim", "crosses_above(price, prev_close)", "price crosses back above the previous close")
    m = re_.search(r"breaks?\s+(?:(below|above|under|over)\s+)?(?:the\s+)?opening[- ](\d+)[- ]?(?:minute|min)\s+(low|high|range)", low)
    if m:
        mins, side = int(m.group(2)), m.group(3)
        below = side == "low" or m.group(1) in ("below", "under")
        add("orb", f"price < opening_range_low({mins})" if below else f"price > opening_range_high({mins})", f"price breaks {'below' if below else 'above'} the first {mins} minutes' {'low' if below else 'high'}")
    elif re_.search(r"opening[- ]range breakout", low):
        add("orb", "price > opening_range_high(30)", "price breaks above the 30-minute opening range high")
    m = re_.search(rf"vix\s*(?:is|was|stays|trades)?\s*(below|under|less than|<|above|over|greater than|more than|>)\s*{N}", low)
    if m:
        lt = m.group(1) in ("below", "under", "less than", "<")
        add("vix", f"vix {'<' if lt else '>'} {float(m.group(2)):g}", f"VIX {'below' if lt else 'above'} {float(m.group(2)):g}")
    m = re_.search(rf"iv rank\s*(?:is\s*)?(below|under|<|above|over|>)\s*{N}", low)
    if m:
        lt = m.group(1) in ("below", "under", "<")
        add("iv_rank", f"iv_rank {'<' if lt else '>'} {float(m.group(2)):g}", f"IV rank {'below' if lt else 'above'} {float(m.group(2)):g}")
    m = re_.search(rf"rsi(?:\s*\(?\s*(\d+)\s*\)?)?\s*(?:is\s*)?(below|under|<|above|over|>)\s*{N}", low)
    if m:
        lt = m.group(2) in ("below", "under", "<")
        add("rsi", f'rsi({int(m.group(1) or 14)}, "5m") {"<" if lt else ">"} {float(m.group(3)):g}', f"RSI({int(m.group(1) or 14)}, 5m) {'below' if lt else 'above'} {float(m.group(3)):g}")
    m = re_.search(r"(above|below|over|under)\s+(?:the|its)?\s*(\d+)[- ]?(?:day|d)\s*(?:simple\s*)?(?:moving average|sma|ma|ema)", low)
    if m:
        up = m.group(1) in ("above", "over")
        add("trend", f'price {">" if up else "<"} sma({int(m.group(2))}, "1d")', f"price {'above' if up else 'below'} the {m.group(2)}-day moving average")
    return out


def find_weekdays(low: str) -> list[str] | None:
    days = []
    for m in re.finditer(r"(?:\bon|\bevery|\beach|\bentered|\benter(?:ing)?|\btrade(?:d|s)?(?:\s+only)?(?:\s+on)?|\bonly)\s+((?:(?:monday|tuesday|wednesday|thursday|friday)s?(?:\s*(?:,|and|&|or)\s*)?)+)", low):
        for d in re.findall(r"monday|tuesday|wednesday|thursday|friday", m.group(1)):
            if DAYNAMES[d] not in days:
                days.append(DAYNAMES[d])
    return days or None


STOP_WORDS = r"\bstops?\b|\bstopp\w+|against|adverse|\bloss(?:es)?\b|\blos(?:e|es|ing)\b|\bcut\b|falls? another|drops? another|before (?:it )?(?:fall|drop|los|declin)"
GAIN_WORDS = r"target|profit|gain|doubl|take\b|ris(?:e|es|ing)\b|\bup\b|increase|rall"


def _last(pat: str, s: str) -> int:
    ms = list(re.finditer(pat, s))
    return ms[-1].start() if ms else -1


def classify_percents(text: str, is_study: bool, spans: list | None = None) -> dict:
    """Assign each '<n>%' to underlying target/stop or premium target/stop by its context."""
    low = text.lower()
    for a, b in spans or []:
        low = low[:a] + " " * (b - a) + low[b:]
    out: dict = {}
    for m in re.finditer(rf"{N}\s*%", low):
        x = float(m.group(1))
        pre, post = low[max(0, m.start() - 48):m.start()], low[m.end():m.end() + 44]
        # a number belongs to its own clause: the words beyond a semicolon or a full stop say nothing about it
        # ("... 0.5% from the open; exit at 50% profit" made the 50% an underlying move)
        pre, post = re.split(r"[;.]\s", pre)[-1], re.split(r"[;.](?:\s|$)", post)[0]
        ctxw = pre + " % " + post
        if re.search(rf"{PREV}|down day|red day|gap|decline\b(?! in)|vix|iv rank|rsi|otm|out of the money|of its|of the overnight", pre[-34:] + post[:26]) and not re.search(r"target|stop|exit|profit|move[sd]? (?:up|down)|ris|increase", pre[-22:]):
            continue
        premium = re.search(r"premium|option(?:'s)? (?:price|value|return|gain|profit)|on the option|of the option|contract (?:price|value)|credit|debit|return on|of max", ctxw) or (
            re.search(r"profit|gain|return|loss|lose|los(?:s|es)", ctxw) and not re.search(r"underlying|spy|qqq|iwm|stock|share|index|price move|movement|from (?:the )?(?:open|entry)", ctxw))
        # the nearest preceding verb decides: "...doubles or loses 50%" is a stop, "...loses 50% or gains 80%" ends on a target
        ps, pg = _last(STOP_WORDS, pre[-30:]), _last(GAIN_WORDS, pre[-30:])
        stop = (ps > pg) if (ps >= 0 or pg >= 0) else bool(re.search(STOP_WORDS, post[:10]))
        # the word right after the number wins: "50% profit or 25% loss", "a 60% premium stop"
        nxt = re.match(r"\s*(?:of\s+)?(?:the\s+)?(?:premium\s+|option\s+|credit\s+|debit\s+|max\s+)?(profits?|gains?|target|loss(?:es)?|stop(?:[- ]loss)?)\b", post)
        if nxt:
            stop = nxt.group(1).startswith(("loss", "stop"))
        down = re.search(rf"\b(?:{DOWN})\b", pre[-22:] + " " + post[:18]) and not re.search(rf"\b(?:{UP})\b", pre[-12:])
        if premium:
            out.setdefault("premium_stop" if stop else "premium_target", x)
        elif stop:
            out.setdefault("und_stop", x)
        elif re.search(r"target|move|movement|ris|increase|gain|rall|higher|up\b|recover|reach|hit|touch|climb|advance|fall|drop|declin|lower|down\b", ctxw):
            if is_study and down:
                out.setdefault("und_target_down", x)
            else:
                out.setdefault("und_target", x)
    if re.search(r"\bdoubles?\b|\bdoubling\b", low):
        out.setdefault("premium_target", 100.0)
    m = re.search(rf"\bstop(?:[- ]loss)?\s+(?:at|of|is)?\s*{N}\s*(?:x|times)\s+(?:the\s+)?(?:credit|premium|debit)", low)
    if m:
        out.setdefault("premium_stop", float(m.group(1)) * 100)  # "stop at 2x the credit": the loss equals twice the premium received
    return out


def classify_dollars(text: str, spans: list | None = None) -> dict:
    """Dollar / point moves of the UNDERLYING ("exit when SPY is up $2"). Strikes, widths,
    premiums and account sizes are left alone."""
    low = text.lower()
    for a, b in spans or []:
        low = low[:a] + " " * (b - a) + low[b:]
    out: dict = {}
    for m in re.finditer(rf"\$\s*{N}(?!\s*%)|{N}\s*(?:dollars?|bucks|points?|pts)\b", low):
        x = float(m.group(1) or m.group(2))
        pre, post = low[max(0, m.start() - 40):m.start()], low[m.end():m.end() + 36]
        ctx = pre + " $ " + post
        if re.search(r"wide|width|strike|premium|credit|debit|option|contract|account|capital|per trade|risk", pre[-26:] + post[:22]):
            continue
        if not re.search(r"move|rise|ris|up\b|gain|rall|higher|fall|drop|down\b|lower|declin|target|stop|against|from (?:the )?(?:entry|open)", ctx):
            continue
        stop = re.search(r"stop|against|adverse|falls? another|drops? another|loss", pre[-26:] + post[:20])
        out.setdefault("und_stop_abs" if stop else "und_target_abs", x)
    return out


MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# 2024-03-01 | 3/1/2024 | march 1, 2024 | 1 march 2024 | march 2024
DATE_TOK = rf"((?:19|20)\d{{2}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/(?:19|20)\d{{2}}|{MONTH_RE}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_RE}\.?,?\s+(?:19|20)\d{{2}}|{MONTH_RE}\.?\s+(?:19|20)\d{{2}})"


def _date_tok(tok: str) -> tuple[date, date] | None:
    """(first day, last day) a written date covers: one day for a full date, the whole month for "march 2024"."""
    import calendar as _cal
    t = tok.lower().strip()
    try:
        m = re.fullmatch(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})", t)
        if m:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, d
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})", t)  # month/day/year
        if m:
            d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            return d, d
        m = re.fullmatch(rf"({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})", t)
        if m:
            d = date(int(m.group(3)), MONTHS[m.group(1)[:3]], int(m.group(2)))
            return d, d
        m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\.?,?\s+((?:19|20)\d{{2}})", t)
        if m:
            d = date(int(m.group(3)), MONTHS[m.group(2)[:3]], int(m.group(1)))
            return d, d
        m = re.fullmatch(rf"({MONTH_RE})\.?\s+((?:19|20)\d{{2}})", t)
        if m:
            y, mo = int(m.group(2)), MONTHS[m.group(1)[:3]]
            return date(y, mo, 1), date(y, mo, _cal.monthrange(y, mo)[1])
    except ValueError:  # 2024-02-31 and the like: not a date
        return None
    return None


def find_dte(low: str) -> int | None:
    """Days to expiration as written: "7DTE", "expiring in 7 days", "two weeks out", "weeklies".
    A holding period ("hold for 5 days") is not an expiration and is never read as one."""
    m = re.findall(r"\b(\d+)\s*-?\s*dte\b", low)
    if m:
        return int(m[0])
    if re.search(r"zero[- ]dte|same[- ]day expir|expir\w* (?:the )?same day|expir\w* today", low):
        return 0
    unit = {"day": 1, "week": 7, "month": 30}
    W = r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen|thirty|sixty|ninety)"
    for pat in (rf"expir\w*\s+(?:in|within|after)?\s*(?:about|around|roughly|approximately)?\s*{W}\s*-?\s*(day|week|month)s?",
                rf"{W}\s*-?\s*(day|week|month)s?\s+(?:to|until|till|from|before|of time to)\s+(?:the\s+)?expir",
                rf"{W}\s*-?\s*(day|week|month)s?\s+out\b",
                rf"(?:with|at)\s+{W}\s*-?\s*(day|week|month)s?\s+(?:left|remaining|to go)",
                rf"\b{W}\s*-\s*(day|week|month)\s+(?:atm |otm |itm |spy |qqq |iwm )*(?:call|put|option|straddle|strangle|spread|condor)"):
        mm = re.search(pat, low)
        if mm:
            tok = mm.group(1)
            n = 1.0 if tok in ("a", "an") else _num(tok)
            if n is not None:
                return int(round(n * unit[mm.group(2)]))
    if re.search(r"next week'?s? expir|expir\w* next week", low):
        return 7
    if re.search(r"next month'?s? expir|expir\w* next month", low):
        return 30
    if re.search(r"\bweekl(?:y|ies)\b", low):
        return 7
    if re.search(r"\bmonthl(?:y|ies)\b", low):
        return 30
    return None


def find_period(low: str) -> tuple[date | None, date | None, str | None]:
    today = date.today()
    if "2022 bear market" in low or ("bear market" in low and "2022" in low):
        return date(2022, 1, 3), date(2022, 10, 14), "the 2022 bear market (2022-01-03 to 2022-10-14)"
    if "covid" in low:
        return date(2020, 2, 19), date(2020, 4, 30), "the COVID crash (2020-02-19 to 2020-04-30)"
    # explicit dates first: "from 2024-03-01 to 2024-06-30", "between 3/1/2024 and June 2024", "since March 2020"
    m = re.search(rf"(?:from|between)\s+{DATE_TOK}\s*(?:to|-|–|through|thru|and|until|till)\s*{DATE_TOK}", low)
    if m:
        a, b = _date_tok(m.group(1)), _date_tok(m.group(2))
        if a and b and a[0] <= b[1]:
            return a[0], b[1], None
    m = re.search(rf"\bon\s+{DATE_TOK}", low)
    if m and _date_tok(m.group(1)) and re.match(r"\d", m.group(1)):  # a full date, not "on january 2024"
        a = _date_tok(m.group(1))
        return a[0], a[1], None
    m = re.search(rf"(?:since|after|from|starting(?: on| in)?|beginning(?: on| in)?)\s+{DATE_TOK}", low)
    if m and _date_tok(m.group(1)):
        end = re.search(rf"(?:until|till|through|thru|up to|ending(?: on| in)?)\s+{DATE_TOK}", low)
        b = _date_tok(end.group(1)) if end else None
        return _date_tok(m.group(1))[0], (b[1] if b else None), None
    m = re.search(rf"(?:until|till|through|thru|up to|before|ending(?: on| in)?)\s+{DATE_TOK}", low)
    if m and _date_tok(m.group(1)):
        return None, _date_tok(m.group(1))[1], None
    m = re.search(rf"(?:in|during|for)\s+({MONTH_RE}\.?\s+(?:19|20)\d{{2}})\b", low)
    if m and _date_tok(m.group(1)):
        a = _date_tok(m.group(1))
        return a[0], a[1], None
    m = re.search(r"(?:from|between)\s+((?:19|20)\d{2})\s*(?:to|-|through|and|until)\s*((?:19|20)\d{2})(?![-/]\d)", low)
    if m:
        return date(int(m.group(1)), 1, 1), date(int(m.group(2)), 12, 31), None
    m = re.search(r"(?:since|after|from)\s+((?:19|20)\d{2})\b(?![-/]\d)", low)
    if m:
        return date(int(m.group(1)), 1, 1), None, None
    m = re.search(r"(?:in|during|for)\s+((?:19|20)\d{2})\b(?![-/]\d)", low)
    if m:
        return date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31), None
    m = re.search(r"(?:last|past)\s+(\w+)\s+(year|month)s?", low)
    if m and _num(m.group(1)):
        n = int(_num(m.group(1)))
        return today - timedelta(days=int(n * (365.25 if m.group(2) == "year" else 30.5))), None, None
    return None, None, None


def parse_fresh(text: str, known: list[str], default_symbol: str = "SPY") -> Draft:
    low = text.lower()
    sym = find_symbol(text, known)
    structure = find_structure(low)
    question = bool(re.search(r"how often|how frequent|what percent|what fraction|probab|odds of|how many times|how likely|what happens (?:when|if|after)|does .{3,40} (?:usually|tend|typically)", low))
    kind = "options" if structure else "study"
    d = Draft(kind=kind, symbol=sym or default_symbol, symbol_stated=bool(sym), structure=structure, title=text.strip()[:90])
    msg_tz = next((TZ_OFFSET[k] for k in TZ_OFFSET if re.search(rf"\b{k}\b", low) and k not in ("et", "ct", "mt", "pt")), None)
    if msg_tz is None:
        m = re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(pt|et|ct|mt)\b", low)
        msg_tz = TZ_OFFSET[m.group(1)] if m else None
    times = find_times(text, msg_tz)
    cue = EXIT_CUE.search(text)
    cut = cue.start() if cue else len(text) + 1
    before_noon = re.search(r"before (noon|midday|\d)", low)
    entry_t = next((t for t in times if t["pos"] < cut and not (before_noon and abs(t["pos"] - before_noon.start()) < 12)), None)
    exit_t = next((t for t in times if t["pos"] >= cut), None)
    spans: list = []
    closing = exit_clauses(text)
    entry_text = text
    for a_, b_ in closing:  # what closes the position is read below, as an exit; here it is blank
        entry_text = entry_text[:a_] + " " * (b_ - a_) + entry_text[b_:]
    conds = find_conditions(entry_text, spans)
    d.conditions = [c for _, c in conds]
    d.weekdays = find_weekdays(low)
    intraday_signal = any(c in ("reclaim", "orb", "streak", "vwap") and ("candles" in x.expr or c in ("reclaim", "orb", "vwap")) for c, x in conds)
    if entry_t:
        d.entry_time, d.entry_time_stated = ("open" if entry_t["et"] == "09:30" else entry_t["et"]), entry_t["stated"]
    elif any(c == "weekday_color" for c, _ in conds):
        d.entry_time, d.entry_time_stated = "close", None
    if d.entry_time == "close" and kind == "options":
        for i, (cat, c) in enumerate(conds):
            if cat == "prev_ret_day" and "previous_session.return_pct" in c.expr:
                expr = c.expr.replace("previous_session.return_pct", "change_pct")
                fell = "<" in expr
                amt = re.search(r"(\d+(?:\.\d+)?)\s*$", expr)
                size = f"{amt.group(1)}% or more" if amt and float(amt.group(1)) else "at all"
                d.conditions[i] = DraftCondition(expr=expr, text=f"that session {'fell' if fell else 'rose'} {size} (close-to-close), measured at its close")
                d.notes.append("With a close entry, \"after a down day\" is read as buying at the close of the down day itself. Say \"the previous day\" to wait one more session.")
    pc = classify_percents(text, kind == "study", spans)
    s, e, note = find_period(low)
    d.start, d.end = s, e
    if note:
        d.notes.append(f"Period interpreted as {note}.")
    if kind == "options":
        dte = find_dte(low)
        if dte is not None:
            d.dte = dte
        if re.search(r"\batm\b|at[- ]the[- ]money", low):
            d.strike = DraftStrike(method="atm")
        else:
            m2 = re.search(rf"{N}\s*-?\s*delta", low)
            m3 = re.search(rf"{N}\s*%\s*(?:otm|out of the money)", low)
            if m2:
                v = float(m2.group(1))
                d.strike = DraftStrike(method="delta", value=v / 100 if v > 1 else v)
            elif m3:
                d.strike = DraftStrike(method="pct_otm", value=float(m3.group(1)))
        m = re.search(r"\b(?:buy|sell|short|long|trade)\s+(one|two|three|four|five|\d+)\s+(?!dte|day|min|consec|%)", low)
        if m and _num(m.group(1)):
            d.quantity = max(1, int(_num(m.group(1))))
        m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:contracts?|lots?)\b|\b(?:size|quantity)\s+(?:of\s+)?(\d+)\b|\bx\s?(\d+)\b(?!\s*(?:credit|the|premium))", low)
        if m and _num(m.group(1) or m.group(2) or m.group(3)):
            d.quantity = max(1, int(_num(m.group(1) or m.group(2) or m.group(3))))
        m = re.search(rf"{N}[- ]?(?:point|pt|\$|dollar)s?[- ](?:wide|wings?)|width (?:of )?\$?{N}|wings? (?:of )?\$?{N}|\$?{N}\s+(?:points?\s+)?wide\b|\$?{N}[- ](?:point|pt|dollar)s?\s+(?:wide|wings?|spread|width)", low)
        if m:
            d.width = float(next(g for g in m.groups() if g))
        d.underlying_target_pct = pc.get("und_target")
        d.underlying_stop_pct = pc.get("und_stop")
        dc = classify_dollars(text, spans)
        d.underlying_target_abs = dc.get("und_target_abs") if d.underlying_target_pct is None else None
        d.underlying_stop_abs = dc.get("und_stop_abs") if d.underlying_stop_pct is None else None
        d.premium_target_pct = pc.get("premium_target")
        d.premium_stop_pct = pc.get("premium_stop")
        if exit_t:
            d.exit_time, d.exit_time_stated = exit_t["et"], exit_t["stated"]
        # a condition inside an exit clause closes the position. A plain move of the underlying there is the target or the
        # stop that was just read (one number, one role); anything else is an exit condition of its own
        levels = {float(v) for v in pc.values() if isinstance(v, (int, float))}
        for a_, b_ in closing:
            for _cat, c in find_conditions(text[a_:b_], []):
                if {abs(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", c.expr)} & levels or c.expr in {x.expr for x in d.exit_conditions}:
                    continue
                d.exit_conditions.append(c)
        m = re.search(r"hold(?:ing)?\s+(?:them|it|the position|for)?\s*(?:for\s+)?(\w+)\s+(?:trading\s+)?(day|session|week)s?", low)
        if m and _num(m.group(1)):
            d.hold_sessions = int(_num(m.group(1))) * (5 if m.group(2) == "week" else 1)
        elif re.search(r"\bovernight\b", low):
            d.hold_sessions = 1
        m = re.search(r"(?:hold|held|holding)[^.;]{0,24}?(?:until|till|through|to)\s+(?:the\s+)?(monday|tuesday|wednesday|thursday|friday)|(?:sell|exit|close)[^.;]{0,16}?on\s+(monday|tuesday|wednesday|thursday|friday)", low)
        if m:
            d.exit_weekday = DAYNAMES[m.group(1) or m.group(2)]
        if re.search(r"(?:until|till|to|through) expir|hold to expir|let (?:it|them) expire", low):
            d.hold_to_expiration = True
        if intraday_signal and not entry_t:
            d.entry_time = None
    else:
        st = DraftStudy()
        if re.search(r"from the open|from (?:its|the) open(?:ing)?(?: price)?|after (?:the market )?open", low) or not entry_t:
            st.reference = "entry"
        if "und_target" in pc:
            st.target = DraftLevel(kind="pct", value=pc["und_target"])
        elif "und_target_down" in pc:
            st.target = DraftLevel(kind="pct", value=-pc["und_target_down"])
        if "und_stop" in pc:
            sgn = -1 if (st.target is None or st.target.value >= 0) else 1
            st.stop = DraftLevel(kind="pct", value=sgn * pc["und_stop"])
        dc = classify_dollars(text, spans)
        if st.target is None and "und_target_abs" in dc:
            fell = bool(re.search(rf"(?:{DOWN})\s+(?:by\s+)?(?:at least\s+)?\$?\s*{dc['und_target_abs']:g}", low))
            st.target = DraftLevel(kind="abs", value=-dc["und_target_abs"] if fell else dc["und_target_abs"])
        if st.stop is None and "und_stop_abs" in dc:
            st.stop = DraftLevel(kind="abs", value=(-1 if (st.target is None or st.target.value >= 0) else 1) * dc["und_stop_abs"])
        m = re.search(rf"{N}\s*%\s*target[^.;]{{0,30}}?{N}\s*%\s*stop", low)
        if m:
            st.target, st.stop = DraftLevel(kind="pct", value=float(m.group(1))), DraftLevel(kind="pct", value=-float(m.group(2)))
        m = re.search(rf"(?:touch|hit|reach|tag|trade)\w*\s+{PREV}\s*(?:day'?s?)?\s*(high|low)\s+before\s+(?:\w+\s+){{0,3}}{PREV}\s*(?:day'?s?)?\s*(high|low)", low)
        if m:
            st.target = DraftLevel(kind="expr", expr=f"prev_{m.group(1)}")
            st.stop = DraftLevel(kind="expr", expr=f"prev_{m.group(2)}")
        m = re.search(rf"(?:recover|retrace|fill|close)\w*\s+{N}\s*%\s+of\s+(?:its|the)\s+(?:overnight\s+|opening\s+)?gap|fills?\s+(?:its|the)\s+(?:overnight\s+|opening\s+)?gap", low)
        if m:
            frac = float(m.group(1)) / 100 if m.group(1) else 1.0
            st.target = DraftLevel(kind="expr", expr=f"open + {frac:g} * (prev_close - open)")
            if not any(c == "gap" for c, _ in conds):
                d.conditions.append(DraftCondition(expr="abs(gap_pct) >= 0.1", text="session opens with a gap of at least 0.1%", source="assumed"))
        m = re.search(rf"(?:close|closes|closing|finish\w*|end\w*|settle\w*)\s+(?:the (?:day|session|week)\s+)?(higher|lower|up|down|green|red|positive|negative|above|below)(?:\s+(?:by\s+)?(?:at least\s+)?{N}\s*%)?", low)
        if m and not (st.target is not None and st.target.kind == "expr"):
            dn = m.group(1) in ("lower", "down", "red", "negative", "below")
            x = float(m.group(2)) if m.group(2) else (abs(st.target.value) if st.target is not None and st.target.kind == "pct" else 0.0)
            st.target, st.stop = DraftLevel(kind="pct", value=-x if dn else x), None
            st.measure, st.direction = "close", ("down" if dn else "up")
        m = re.search(r"before (noon|midday)", low)
        horizon_t = "12:00" if m else None
        if not horizon_t:
            m = re.search(r"(?:before|by|within the first|until)\s+", low)
            late = [t for t in times if m and t["pos"] > m.start() and t["et"] not in ("open",)] if m else []
            if late and (entry_t is None or late[0] is not entry_t):
                horizon_t = "16:00" if late[0]["et"] == "close" else late[0]["et"]
        if horizon_t and horizon_t != "16:00":
            st.horizon_kind, st.horizon_time = "time", horizon_t
        m = re.search(r"(?:over|within|in|during) the next (\w+) (?:trading )?(day|session|week)s?|(\w+)[- ](?:day|session)s? later", low)
        if m and _num(m.group(1) or m.group(3) or ""):
            st.horizon_kind, st.horizon_sessions = "sessions", int(_num(m.group(1) or m.group(3))) * (5 if (m.group(2) or "") == "week" else 1)
        d.study = st
        if entry_t and entry_t["et"] not in ("open", "09:30"):
            d.entry_time, d.entry_time_stated = entry_t["et"], entry_t["stated"]
        else:
            d.entry_time = "open" if entry_t else None
    for t in (entry_t, exit_t):
        if t and t.get("assumed_tz"):
            d.tz_assumed = t["assumed_tz"]
            d.notes.append(f"\"{t['stated'].split(' (')[0]}\" had no timezone; read as {t['assumed_tz']} time = {t['et']} New York.")
    return d


# ----------------------------------------------------------------------------- follow-ups
FOLLOW_CUES = re.compile(r"^\s*(only|now|also|and |but |what if|how about|change|set|make|use|switch|instead|try|add|remove|drop|without|with |same|compare|test (?:the same|it|this|entry|underlying|previous)|would (?:this|it)|was (?:this|it)|is (?:this|it)|show|run (?:it|this)|how sensitive|restrict|limit|exit|hold|stop)", re.I)
SWEEP_FIELDS = {"prev_ret": "previous-day move threshold (%)", "target": "underlying target (%)", "stop": "underlying stop (%)", "dte": "days to expiration", "entry_time": "entry time (ET)",
                "premium_target": "premium profit target (%)", "premium_stop": "premium stop (%)", "delta": "strike delta", "slippage": "slippage per contract ($)", "gap": "gap threshold (%)", "vix": "VIX threshold", "hold_sessions": "holding period (sessions)"}


def _cat_of(expr: str) -> str:
    for cat, key in (("weekday_color", "weekday =="), ("prev_ret", "previous_session."), ("prev_ret", "change_pct"), ("gap", "gap_pct"), ("vix", "vix"), ("streak", "consecutive_"), ("vwap", "vwap"), ("orb", "opening_range"), ("reclaim", "crosses_above(price, prev_close)"),
                     ("iv_rank", "iv_rank"), ("rsi", "rsi("), ("trend", "sma("), ("weekday_color", "weekday ==")):
        if key in expr:
            return cat
    return expr


def _set_threshold(c: DraftCondition, x: float) -> DraftCondition:
    """Replace the numeric threshold of a simple comparison, keeping its sign convention."""
    m = re.search(r"(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$", c.expr)
    if not m:
        return c
    neg = m.group(2).startswith("-") or (m.group(1) in ("<", "<=") and "return_pct" in c.expr or "gap_pct" in c.expr and m.group(1) in ("<", "<="))
    val = -abs(x) if neg else abs(x)
    op = m.group(1) if m.group(2) not in ("0",) else {"<": "<=", ">": ">="}.get(m.group(1), m.group(1))
    expr = c.expr[:m.start()] + f"{op} {val:g}"
    text = re.sub(r"\d+(?:\.\d+)?%", f"{abs(x):g}%", c.text) if re.search(r"\d%", c.text) else (
        f"previous session {'fell' if val < 0 else 'rose'} {abs(x):g}% or more (close-to-close)" if "previous_session" in expr else
        f"that session {'fell' if val < 0 else 'rose'} {abs(x):g}% or more (close-to-close), measured at its close" if "change_pct" in expr else f"{c.text.split(' (')[0]} (threshold {abs(x):g})")
    return DraftCondition(expr=expr, text=text, source="stated")


def apply_value(d: Draft, field: str, v) -> Draft:
    """Return a copy of the draft with one swept/edited field changed."""
    n = d.model_copy(deep=True)
    if field == "prev_ret":
        idx = next((i for i, c in enumerate(n.conditions) if _cat_of(c.expr) == "prev_ret"), None)
        if idx is None:
            n.conditions.append(DraftCondition(expr=f"previous_session.return_pct <= -{abs(float(v)):g}", text=f"previous session fell {abs(float(v)):g}% or more (close-to-close)"))
        else:
            n.conditions[idx] = _set_threshold(n.conditions[idx], float(v))
    elif field in ("gap", "vix"):
        idx = next((i for i, c in enumerate(n.conditions) if _cat_of(c.expr) == field), None)
        if idx is not None:
            n.conditions[idx] = _set_threshold(n.conditions[idx], float(v))
    elif field == "target":
        if n.kind == "study":
            st = n.study or DraftStudy()
            sign = -1 if (st.target and st.target.kind == "pct" and st.target.value < 0) else 1
            st.target = DraftLevel(kind="pct", value=sign * abs(float(v)))
            n.study = st
        else:
            n.underlying_target_pct = abs(float(v))
    elif field == "stop":
        if n.kind == "study":
            st = n.study or DraftStudy()
            sign = 1 if (st.target and st.target.kind == "pct" and st.target.value < 0) else -1
            st.stop = DraftLevel(kind="pct", value=sign * abs(float(v)))
            n.study = st
        else:
            n.underlying_stop_pct = abs(float(v))
    elif field == "dte":
        n.dte = int(v)
    elif field == "entry_time":
        n.entry_time, n.entry_time_stated = ("open" if v == "09:30" else str(v)), None
    elif field == "premium_target":
        n.premium_target_pct = float(v)
    elif field == "premium_stop":
        n.premium_stop_pct = float(v)
    elif field == "delta":
        n.strike = DraftStrike(method="delta", value=float(v) / 100 if float(v) > 1 else float(v))
    elif field == "hold_sessions":
        n.hold_sessions = int(v)
    return n


def _field_in(low: str, d: Draft | None) -> str | None:
    if re.search(rf"{PREV}|down day|red day|decline|drop\b|fell|fall", low) and not re.search(r"target|stop", low):
        return "prev_ret"
    if re.search(r"entry time|entry at|enter(?:ing)? (?:at|from)|entries? (?:from|at|between)", low):
        return "entry_time"
    if re.search(r"dte|expir", low):
        return "dte"
    if re.search(r"slippage", low):
        return "slippage"
    if re.search(r"stop", low):
        return "premium_stop" if re.search(r"premium|option", low) else "stop"
    if re.search(r"target|take profit|profit", low):
        return "premium_target" if re.search(r"premium|option|profit", low) and not re.search(r"underlying|move", low) else "target"
    if re.search(r"delta", low):
        return "delta"
    if re.search(r"gap", low):
        return "gap"
    if re.search(r"vix", low):
        return "vix"
    if re.search(r"hold", low):
        return "hold_sessions"
    return None


def _range_values(low: str, field: str) -> list | None:
    m = re.search(rf"(?:from|between)\s+{N}\s*%?\s*(?:to|and|-|through)\s*{N}\s*%?(?:[^.;]*?(?:in|by|step(?:s)? of)\s+{N}\s*%?[- ]?(?:increments?|steps?)?)?", low)
    if m and field not in ("entry_time",):
        a, b = float(m.group(1)), float(m.group(2))
        step = float(m.group(3)) if m.group(3) else (0.5 if b - a >= 2 else 0.25 if b - a > 0.5 else 0.1)
        if field in ("dte", "hold_sessions"):
            step = max(1.0, step if m.group(3) else 1.0)
        vals, x = [], a
        while x <= b + 1e-9 and len(vals) < 13:
            vals.append(round(x, 4))
            x += step
        return [int(v) for v in vals] if field in ("dte", "hold_sessions") else vals
    return None


def parse_followup(text: str, cur: Draft, known: list[str], last_field: str | None) -> dict:
    """Interpret a message in the context of the current draft."""
    low = text.lower()
    # ---- non-mutating requests
    m = re.search(r"show (?:me )?(?:every|all|the|each)?\s*(losing|winning|loser|winner|profitable|unprofitable)?\w*\s*(?:trade|session|day)s?|(?:list|see|inspect|open) (?:the |all |every )?(losing|winning)?\s*trades", low)
    if m:
        w = (m.group(1) or m.group(2) or "").lower()
        return {"action": "show_trades", "filter": "loss" if w.startswith(("los", "unprof")) else "win" if w.startswith(("win", "prof")) else "all"}
    if re.search(r"out[- ]of[- ]sample|outside the period|hold-?out|walk[- ]forward|unseen (?:data|period)|overfit", low):
        return {"action": "oos"}
    wants_compare = bool(re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b|side by side", low))
    # ---- sweeps
    field = _field_in(low, cur)
    if re.search(r"how sensitive|sensitivity", low) and field is None:
        field = last_field or ("target" if (cur.underlying_target_pct or (cur.study and cur.study.target)) else "prev_ret")
    dtes = re.findall(r"\b(\d+)\s*-?\s*dte\b", low)
    replace_dte = len(dtes) == 2 and bool(re.search(r"\binstead of\b|\brather than\b|\bnot\b", low)) and cur.dte is not None and int(dtes[1]) == int(cur.dte) and not wants_compare
    if len(dtes) >= 2 and not replace_dte:
        vals = sorted({int(x) for x in dtes})
        if re.search(r"through|thru|to\b", low) and len(vals) == 2:
            vals = [v for v in (0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45) if vals[0] <= v <= vals[1]]
        return {"action": "sweep", "field": "dte", "values": vals[:12]}
    # "enter at 6:30 and exit at 12:45" names one entry time and one exit time: it is an edit, not two entry times to compare
    one_of_each = bool(EXIT_CUE.search(text)) and not re.search(r"entry times?|\bversus\b|\bvs\.?\b|\bcompare\b", low)
    if (field == "entry_time" or re.search(r"entry times?", low)) and not one_of_each:
        msg_tz = next((TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in low), None)
        ts = [t for t in find_times(text, msg_tz) if t["et"] not in ("open", "close", "12:00") or True]
        mins = [t["minutes"] for t in ts]
        if len(mins) >= 2 and re.search(r"from|between|through|to\b", low):
            sm = re.search(r"(\d+)[- ]?(?:minute|min)", low)
            step = int(sm.group(1)) if sm else 15
            a, b = min(mins[0], mins[1]), max(mins[0], mins[1])
            vals = [f"{x // 60:02d}:{x % 60:02d}" for x in range(a, b + 1, step)][:13]
            return {"action": "sweep", "field": "entry_time", "values": vals}
        if len(mins) >= 2:
            return {"action": "sweep", "field": "entry_time", "values": [f"{x // 60:02d}:{x % 60:02d}" for x in mins[:12]]}
    m = re.search(r"(?:versus|vs\.?|instead of|rather than)\s+waiting\s+(\d+)\s*(?:minute|min)", low)
    if m and "open" in low:
        w = 570 + int(m.group(1))
        return {"action": "sweep", "field": "entry_time", "values": ["09:30", f"{w // 60:02d}:{w % 60:02d}"]}
    if field == "slippage":
        return {"action": "sweep", "field": "slippage", "values": [0.0, 0.01, 0.02, 0.05, 0.10]}
    if field and field != "entry_time":
        rv = _range_values(low, field)
        if rv:
            return {"action": "sweep", "field": field, "values": rv}
    nums = [float(x) for x in re.findall(rf"{N}\s*%", low)]
    # "run the same strategy after 1% and 2% declines": the same rules at each of the sizes named
    several = len(nums) >= 2 and re.search(rf"{N}\s*%\s*(?:,|and|or|&)\s*(?:(?:a|an)\s+)?{N}\s*%", low) and re.search(r"\bsame\b|\brun\b|\brerun\b|\brepeat\b|\bagain\b|\balso\b", low) and not one_of_each
    if len(nums) >= 2 and (wants_compare or re.search(r"\btest\b|\btry\b|\bsweep\b", low) or several):
        return {"action": "sweep", "field": field or last_field or "prev_ret", "values": sorted(set(nums))[:12]}
    if re.search(r"how sensitive|sensitivity", low) and field:
        base = {"target": cur.underlying_target_pct or (abs(cur.study.target.value) if cur.study and cur.study.target and cur.study.target.kind == "pct" else 1.0), "stop": cur.underlying_stop_pct or 0.5}.get(field, 2.0)
        return {"action": "sweep", "field": field, "values": [round(base * k, 3) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]}
    # ---- edits
    n = cur.model_copy(deep=True)
    changed: list[str] = []
    new_sym = find_symbol(text, known)
    if new_sym and new_sym != cur.symbol and re.search(rf"\b(on|for|with|use|try|switch to|instead|same)\b[^.]*\b{new_sym}\b|\b{new_sym}\b instead", text, re.I):
        n.symbol, n.symbol_stated = new_sym, True
        changed.append("symbol")
    if re.search(r"\bputs?\b", low) and re.search(r"instead|what if|switch|bought|buy|use|change", low) and cur.structure and "call" in cur.structure:
        n.structure = {"long_call": "long_put", "short_call": "short_put", "bull_call_spread": "bear_put_spread", "bear_call_spread": "bull_put_spread"}.get(cur.structure, cur.structure)
        changed.append("structure")
    elif re.search(r"\bcalls?\b", low) and re.search(r"instead|what if|switch|bought|buy|use|change", low) and cur.structure and "put" in cur.structure:
        n.structure = {"long_put": "long_call", "short_put": "short_call", "bear_put_spread": "bull_call_spread", "bull_put_spread": "bear_call_spread"}.get(cur.structure, cur.structure)
        changed.append("structure")
    elif find_structure(low) and find_structure(low) != cur.structure and re.search(r"instead|switch|use|try|change", low):
        n.structure = find_structure(low)
        changed.append("structure")
    if n.kind == "options":
        m2, m3 = re.search(rf"{N}\s*-?\s*delta", low), re.search(rf"{N}\s*%\s*(?:otm|out of the money)", low)
        if m2:
            v = float(m2.group(1))
            n.strike = DraftStrike(method="delta", value=v / 100 if v > 1 else v)
            changed.append("strike")
        elif m3:
            n.strike = DraftStrike(method="pct_otm", value=float(m3.group(1)))
            changed.append("strike")
        elif re.search(r"\batm\b|at[- ]the[- ]money", low):
            n.strike = DraftStrike(method="atm")
            changed.append("strike")
        mq = re.search(r"\b(one|two|three|four|five|\d+)\s+contracts?\b", low)
        if mq and _num(mq.group(1)):
            n.quantity = max(1, int(_num(mq.group(1))))
            changed.append("quantity")
    m = re.search(rf"(?:change|set|make|move|adjust|lower|raise|increase|decrease)\s+(?:the\s+)?(.{{3,48}}?)\s+(?:to|at)\s+(-?{N[1:-1]})\s*%?", low)
    if m:
        f = _field_in(m.group(1), cur)
        if f and f != "entry_time":  # "change the entry time to 6:45 AM Pacific": the clock reader below takes it, zone and all
            n = apply_value(n, f, float(m.group(2)))
            changed.append(f)
    if not changed or wants_compare:
        for cat, c in find_conditions(text):
            cat = "prev_ret" if cat == "prev_ret_day" else cat
            idx = next((i for i, x in enumerate(n.conditions) if _cat_of(x.expr) == cat), None)
            if idx is not None and cat == "prev_ret" and "change_pct" in n.conditions[idx].expr:
                c = _set_threshold(n.conditions[idx], abs(float(re.search(r"(\d+(?:\.\d+)?)\s*$", c.expr).group(1)))) if re.search(r"(\d+(?:\.\d+)?)\s*$", c.expr) else c
            if idx is None:
                n.conditions.append(c)
            else:
                n.conditions[idx] = c
            if cat not in changed:
                changed.append(cat)
    if "prev_ret" not in changed:
        m = re.search(rf"(?:only|when|if)\s+[^.;]{{0,30}}?{PREV}[^.;%]{{0,40}}?(?:down|red|lower|fell|dropped)\s*(?:by\s*)?(?:more than|at least|over)?\s*{N}\s*%", low)
        if m:
            n = apply_value(n, "prev_ret", float(m.group(1)))
            changed.append("prev_ret")
    msg_tz = next((TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in low), None)
    times = find_times(text, msg_tz)
    cue = EXIT_CUE.search(text)
    cut = cue.start() if cue else len(text) + 1
    t_in, t_out = [t for t in times if t["pos"] < cut], [t for t in times if t["pos"] >= cut]
    hold_until = bool(re.search(r"\b(?:hold|holding)\s+(?:it\s+|them\s+)?(?:until|till|to|through)\b|\bget out\b|\bbe out\b|\bflat by\b", low))
    if hold_until and t_in and not t_out:
        t_out, t_in = t_in, []
    if t_in and re.search(r"enter|entry|buy at|open (?:the trade|at)|at the open|start|observ|measur|watch|look(?:ing)? (?:at|from)", low) and "entry_time" not in changed:
        n.entry_time, n.entry_time_stated = ("open" if t_in[0]["et"] == "09:30" else t_in[0]["et"]), t_in[0]["stated"]
        changed.append("entry_time")
    # both can be stated in one sentence ("enter at 6:30 AM Pacific and exit at 12:45 PM Pacific"): the exit used to be dropped when an entry was read
    if not t_out and len(times) == 1 and re.search(r"\b(?:hold|holding)\s+(?:it\s+|them\s+)?(?:until|till|to|through)\b|\bget out\b|\bbe out\b|\bflat by\b", low):
        t_out, t_in = times, []  # "hold until 3 PM", "get out at 1 PM": the one time named is when the position ends
    if t_out and re.search(r"exit|sell|close (?:out|by|at)|out by|get out|be out|flat by|\bhold", low):
        n.exit_time, n.exit_time_stated = t_out[-1]["et"], t_out[-1]["stated"]
        changed.append("exit_time")
    elif re.search(r"(?:exit|sell|close(?: out)?|get out)\s+(?:it\s+|them\s+)?(?:at|on|by)\s+the\s+close\b", low) and n.kind == "options":
        n.exit_time, n.exit_time_stated = "close", "the close"
        changed.append("exit_time")
    m = re.findall(r"\b(\d+)\s*-?\s*dte\b", low)
    if n.kind == "options" and replace_dte:
        n.dte = int(m[0])
        changed.append("dte")
    elif n.kind == "options" and len(m) <= 1 and find_dte(low) is not None and (m or re.search(r"expir|\bout\b|weekl|monthl|-(?:day|week|month)\b", low)):
        n.dte = find_dte(low)
        changed.append("dte")
    wd = find_weekdays(low)
    if wd and re.search(r"\bonly\b|\bjust\b|\brestrict\w*\b|\blimit\w*\b|\btrade(?:s|d)?\s+on\b", low) and not re.search(r"differ|perform|better|worse|break ?down", low):
        n.weekdays = wd
        changed.append("weekdays")
    # "require the VIX to be above 20 as well": one more entry condition, read by the research reader when the static parser has no rule for it
    if not changed and re.search(r"\b(?:require|requiring|also|as well|too\b|in addition|add (?:a |the )?(?:condition|filter|rule|requirement)|make sure|only (?:if|when))\b", low):
        phrase = re.sub(r"^\s*(?:and\s+|also\s+|please\s+)*(?:require(?:\s+that)?|add (?:a |the )?(?:condition|filter|rule|requirement)(?:\s+(?:that|requiring|of))?|make sure(?:\s+that)?|only (?:if|when)|also)\s+", "", text.strip(" .!?"), flags=re.I)
        phrase = re.sub(r"\s*(?:,\s*)?(?:as well|too|also|in addition)\s*$", "", phrase, flags=re.I)
        phrase = re.sub(r"\bto be\b", "is", phrase)
        try:
            from strata.ai import nl_graph as NG
            got, rest = NG.entry_conditions("when " + phrase, n.symbol, known, None, n.entry_time, partial=True)
        except Exception:  # noqa: BLE001
            got, rest = [], [phrase]
        have = {c.expr.replace(" ", "") for c in n.conditions}
        if got and not [x for x in rest if re.search(r"[a-z]{3,}", x.lower())]:
            for g in got:
                if g["expr"].replace(" ", "") not in have:
                    n.conditions.append(DraftCondition(expr=g["expr"], text=g["text"]))
            changed.append("conditions")
    pc = classify_percents(text, n.kind == "study")
    for key, fld in (("und_target", "target"), ("und_stop", "stop"), ("premium_target", "premium_target"), ("premium_stop", "premium_stop")):
        if key in pc and fld not in changed and "prev_ret" not in changed and re.search(r"target|stop|profit|exit|take", low):
            n = apply_value(n, fld, pc[key])
            changed.append(fld)
    if re.search(r"(?:remove|drop|no|without)\s+(?:the\s+)?stop", low):
        n.underlying_stop_pct = n.premium_stop_pct = None
        if n.study:
            n.study.stop = None
        changed.append("stop")
    m = re.search(r"(?:remove|drop|without)\s+(?:the\s+)?(vix|gap|vwap|previous[- ]day|prior[- ]day|rsi|trend)\s*(?:filter|condition|rule)?", low)
    if m:
        cat = {"previous-day": "prev_ret", "previous day": "prev_ret", "prior-day": "prev_ret", "prior day": "prev_ret"}.get(m.group(1), m.group(1))
        n.conditions = [c for c in n.conditions if _cat_of(c.expr) != cat]
        changed.append(cat)
    m = re.search(r"hold(?:ing)?\s+(?:them|it|for)?\s*(?:for\s+)?(\w+)\s+(?:trading\s+)?(day|session|week)s?", low)
    if m and _num(m.group(1)) and n.kind == "options":
        n.hold_sessions = int(_num(m.group(1))) * (5 if m.group(2) == "week" else 1)
        changed.append("hold_sessions")
    s, e, note = find_period(low)
    if s or e:
        n.start, n.end = s, e
        if note:
            n.notes = [x for x in n.notes if not x.startswith("Period interpreted")] + [f"Period interpreted as {note}."]
        changed.append("period")
    if re.search(r"modeled|theoretical|black[- ]scholes|estimate the option", low):
        n.mode = "modeled"
        changed.append("mode")
    elif re.search(r"market study|underlying only|skip the option|just the underlying|directional study", low):
        n.mode = "study"
        changed.append("mode")
    if not changed:
        return {"action": "unknown"}
    return {"action": "modify", "draft": n, "changed": changed, "compare": wants_compare or "period" in changed}


CORRECTION = re.compile(r"^\s*(?P<cue>(?:(?:no+|nope|not quite|wrong|sorry|oops|wait|actually|rather)\b[\s,.:;!-]*)*"
                        r"(?:(?:i|we)\s+(?:meant|mean)(?:\s+to\s+say)?\b|(?:make|change|set)\s+(?:it|that)(?:\s+to)?\b|it\s+should\s+be\b|should\s+be\b|it'?s\s+supposed\s+to\s+be\b)?)"
                        r"\s*[,:]?\s*(?P<rest>.*)$", re.I | re.S)
FILLER = {"at", "the", "a", "an", "not", "instead", "time", "sharp", "please", "to", "it", "that", "of", "for", "on", "in", "is", "was", "be", "i", "meant", "mean", "said", "and", "new", "york", "day", "days",
          "contract", "contracts", "option", "options", "expiration", "expiry", "exp"}


def correction(text: str) -> str | None:
    """The words after a correcting lead-in ("no I meant 6:30a, PST" -> "6:30a, PST"), or None when the message is not one."""
    m = CORRECTION.match(text or "")
    if not m or not m.group("cue").strip() or not m.group("rest").strip(" .!?"):
        return None
    return m.group("rest").strip()


def _bare(rest: str, *spans: tuple[int, int]) -> bool:
    """Nothing is left of the message once the value at `spans` is taken out: it is a value on its own, not a sentence."""
    left = "".join(ch for i, ch in enumerate(rest) if not any(a <= i < b for a, b in spans))
    return not [w for w in re.findall(r"[a-z]+", left.lower()) if w not in FILLER and w not in TZ_OFFSET]


def bare_correction(rest: str, cur: Draft, known: list[str]) -> dict | None:
    """A correction that is a value on its own: "no I meant 6:30a, PST", "no, 2%", "I meant puts", "sorry, 1DTE".

    The value replaces the one thing in the current reading it can replace. If it could replace two (a rise of 1% after a
    decline of 3%, then "no I meant 2%"), nothing is changed: the reply says how to name the one meant. Words that say what
    they change ("no stop loss", "actually enter at 10:00") are not for this function: None, and the ordinary readers take them."""
    low = rest.lower().strip(" .!?")
    n = cur.model_copy(deep=True)
    msg_tz = next((TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in low), None)
    times = find_times(rest, msg_tz)
    spans = [(m.start(), m.end()) for m in TIME_RE.finditer(rest) if m.group(2) or m.group(3) or m.group(4)]
    if len(times) == 1 and _bare(re.sub(r"\b(?:market|opening|closing)?\s*(?:open|close|bell|noon|midday)\b", " ", rest, flags=re.I), *spans):
        t = times[0]
        stated_exit = cur.kind == "options" and bool(cur.exit_time and cur.exit_time_stated)
        stated_entry = bool(cur.entry_time and cur.entry_time_stated)
        if stated_exit and stated_entry:
            said = t["stated"].split(" (")[0]
            return {"action": "clarify", "text": f"The current reading has an entry time ({cur.entry_time_stated}) and an exit time ({cur.exit_time_stated}), and \"{said}\" could replace either. "
                                                 f"Say which: \"enter at {said}\" or \"exit at {said}\"."}
        if stated_exit and not stated_entry and t["et"] not in ("open", "09:30"):
            n.exit_time, n.exit_time_stated = t["et"], t["stated"]
            return {"action": "modify", "draft": n, "changed": ["exit_time"], "compare": False}
        n.entry_time, n.entry_time_stated = ("open" if t["et"] in ("09:30", "open") else t["et"]), t["stated"]
        return {"action": "modify", "draft": n, "changed": ["entry_time"], "compare": False}
    m = re.fullmatch(rf"(?:an?\s+)?-?{N}\s*(?:%|percent)", low)
    if m:
        x = float(m.group(1))
        cands: list[tuple[str, str]] = []
        for c in cur.conditions:
            cat = _cat_of(c.expr)
            if cat in ("prev_ret", "gap", "vix") and re.search(r"(<=|>=|<|>)\s*-?\d+(?:\.\d+)?\s*$", c.expr):
                cands.append((cat, c.text.split("   [")[0].split(" (")[0]))
        st = cur.study
        if cur.kind == "study" and st is not None:
            if st.target is not None and st.target.kind == "pct" and st.target.value:
                cands.append(("target", f"the {abs(st.target.value):g}% target"))
            if st.stop is not None and st.stop.kind == "pct":
                cands.append(("stop", f"the {abs(st.stop.value):g}% stop"))
        if cur.kind == "options":
            for f, v, name in (("target", cur.underlying_target_pct, "target on the underlying"), ("stop", cur.underlying_stop_pct, "stop on the underlying"),
                               ("premium_target", cur.premium_target_pct, "premium target"), ("premium_stop", cur.premium_stop_pct, "premium stop")):
                if v is not None:
                    cands.append((f, f"the {v:g}% {name}"))
        if len(cands) == 1:
            return {"action": "modify", "draft": apply_value(n, cands[0][0], x), "changed": [cands[0][0]], "compare": False}
        if len(cands) > 1:
            return {"action": "clarify", "text": f"\"{rest.strip(' .!?')}\" could replace more than one thing in the current reading: " + "; ".join(lab for _f, lab in cands) + ". "
                                                 "Nothing was changed. Name the one you mean, for example \"make the target 2%\" or \"change the decline to 2%\" (with your number)."}
    stc = find_structure(low)
    if stc and cur.kind == "options" and _bare(re.sub(r"\b(?:long|short|calls?|puts?|spreads?|straddles?|strangles?|condors?|iron|credit|debit|bull|bear|vertical|butterfl\w+)\b", " ", rest, flags=re.I)):
        n.structure = stc
        return {"action": "modify", "draft": n, "changed": ["structure"], "compare": False}
    if cur.kind == "options" and cur.structure and re.fullmatch(r"(?:the\s+)?(puts?|calls?)", low):
        flip = {"long_call": "long_put", "short_call": "short_put", "bull_call_spread": "bear_put_spread", "bear_call_spread": "bull_put_spread"}
        flip.update({v: k for k, v in flip.items()})
        want_put = low.strip().endswith(("put", "puts"))
        n.structure = flip.get(cur.structure, cur.structure) if want_put != ("put" in cur.structure) else cur.structure
        return {"action": "modify", "draft": n, "changed": ["structure"], "compare": False}
    if cur.kind == "options" and find_dte(low) is not None and _bare(re.sub(r"\d+\s*-?\s*dte|same[- ]day|zero[- ]dte|weekl\w+|monthl\w+|\d+|days?|weeks?|months?|out", " ", rest, flags=re.I)):
        n.dte = find_dte(low)
        return {"action": "modify", "draft": n, "changed": ["dte"], "compare": False}
    sym = find_symbol(rest, known)
    if sym and _bare(re.sub(rf"\b{re.escape(sym)}\b", " ", rest, flags=re.I)):
        n.symbol, n.symbol_stated = sym, True
        return {"action": "modify", "draft": n, "changed": ["symbol"], "compare": False}
    return None


def unplaced_correction(rest: str) -> dict | None:
    """What is said when a correction could not be placed by any reader: only for a few words, never for a whole sentence."""
    if len(rest.split()) > 7:
        return None
    return {"action": "clarify", "text": f"I took \"{rest.strip(' .!?')}\" as a correction of the current reading, but could not tell which part it replaces, so nothing was changed. "
                                         "Name the part, for example \"enter at 10:00 AM Eastern\", \"change the decline to 2%\" or \"use puts instead\"."}


def interpret(text: str, current: Draft | None, known_symbols: list[str], last_field: str | None = None, default_symbol: str = "SPY") -> dict:
    known = list(dict.fromkeys(TICKERS + [s for s in known_symbols if s.isupper()]))
    low = text.lower().strip()
    if current is not None:
        fresh_markers = bool(re.search(r"^(buy|sell|test (?:a|an|buying|selling)|how often|what percent|what happens|backtest)\b", low))
        looks_followup = bool(FOLLOW_CUES.search(low)) or len(low.split()) <= 9
        has_own_subject = bool(find_structure(low)) and bool(re.search(r"\b(buy|sell|short|long|enter|test)\b", low)) and not re.search(r"instead|same|compare|what if", low)
        if looks_followup or not (fresh_markers or has_own_subject):
            r = parse_followup(text, current, known, last_field)
            if r["action"] != "unknown":
                return r
            if looks_followup and not fresh_markers:
                return {"action": "unknown"}
    d = parse_fresh(text, known, default_symbol)
    if d.kind == "study" and not d.conditions and (d.study is None or (d.study.target is None and d.study.stop is None)) and not re.search(r"how often|what happens|how frequent|probab|percent", low):
        return {"action": "unknown"}
    sweep = None
    dtes = sorted({int(x) for x in re.findall(r"\b(\d+)\s*-?\s*dte\b", low)})
    if len(dtes) >= 2 and d.kind == "options":
        d.dte = dtes[0]
        sweep = {"field": "dte", "values": dtes[:12]}
    m = re.search(r"(?:versus|vs\.?|instead of|rather than)\s+waiting\s+(\d+)\s*(?:minute|min)", low)
    if m:
        w = 570 + int(m.group(1))
        d.entry_time = "open"
        sweep = {"field": "entry_time", "values": ["09:30", f"{w // 60:02d}:{w % 60:02d}"]}
    return {"action": "new", "draft": d, "sweep": sweep}
