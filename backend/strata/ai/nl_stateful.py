"""Offline interpreter, part two: temporal and stateful language.

`nl_rules` reads static conditions ("after a 3% down day", "VIX below 18"). This module
reads the parts of a request that describe *order*, *state* and *relationships*:

    wait until SPY drops 0.5% from the open, then buy when it recovers half of that decline
    only enter after the first breakout fails and price reclaims VWAP
    ... unless yesterday was also red
    ... but only if QQQ hasn't broken its own high
    add another contract after a further 0.5% decline
    close both positions when price returns to VWAP

and turns them into explicit engine constructs (event sequences, cross-symbol rules,
scale-in rules, custom exits) rather than approximating them with a static condition.
It also finds what it could NOT represent and reports those words back verbatim, and it
raises a clarifying question when a number's meaning is genuinely ambiguous.
"""
from __future__ import annotations

import re

from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, DraftAdd, DraftCondition, DraftCosts, DraftStage

N = NL.N
FRACTIONS = {"half": 0.5, "a half": 0.5, "one half": 0.5, "a third": 1 / 3, "one third": 1 / 3, "two thirds": 2 / 3, "a quarter": 0.25, "one quarter": 0.25, "three quarters": 0.75, "all": 1.0, "the whole": 1.0}
FRAC = r"(half|a half|one half|a third|one third|two thirds|a quarter|one quarter|three quarters|all|the whole|\d+(?:\.\d+)?\s*%)"
FALL = r"(?:drops?|falls?|declines?|sells? off|is down|dips?|pulls? back|slides?|loses|trades? down)"
RISE = r"(?:rall(?:y|ies)|rises?|gains?|pops?|is up|runs? up|climbs?|jumps?|trades? up)"
LEVELS = {"vwap": "vwap", "the open": "open", "its open": "open", "the opening price": "open", "the previous close": "prev_close", "the prior close": "prev_close", "yesterday's close": "prev_close",
          "yesterdays close": "prev_close", "yesterday's high": "prev_high", "yesterdays high": "prev_high", "the previous high": "prev_high", "yesterday's low": "prev_low", "yesterdays low": "prev_low",
          "the previous low": "prev_low", "the session high": "high", "the day's high": "high", "the high of the day": "high", "the session low": "low", "the day's low": "low", "the low of the day": "low"}
LEVEL_RE = "(" + "|".join(sorted((re.escape(k) for k in LEVELS), key=len, reverse=True)) + ")"
LEVEL_TEXT = {"vwap": "session VWAP", "open": "the session open", "prev_close": "the previous close", "prev_high": "yesterday's high", "prev_low": "yesterday's low", "high": "the session high", "low": "the session low"}
MARKET_WORDS = re.compile(r"breaks?|broke|crosses|cross(?:ed|ing)?|above|below|higher|lower|drops?|falls?|rises?|rall|red|green|candle|vwap|\bhigh\b|\blow\b|gap|volume|rsi|macd|moving average|\bma\b|vix|%|recover|reclaim|fail|touch|bounce|reverse|outperform|underperform|"
                          r"\bweak\w*|\bstrong\w*|\bis (?:up|down|flat|positive|negative)\b|declin\w*|sell-?off|sold off|"
                          r"divergen|trend|momentum|volatil|range|support|resistance|pivot|fib|bollinger|squeeze|imbalance|delta|gamma|open interest|skew|term structure|earnings|fomc|cpi|news", re.I)


def _frac(tok: str) -> float:
    tok = tok.strip().lower()
    if tok in FRACTIONS:
        return FRACTIONS[tok]
    return float(re.sub(r"[^\d.]", "", tok)) / 100


def _frac_text(f: float) -> str:
    return {0.5: "half", 0.25: "a quarter", 1.0: "all", round(1 / 3, 6): "a third", round(2 / 3, 6): "two thirds", 0.75: "three quarters"}.get(round(f, 6), f"{f * 100:g}%")


class Reading:
    """What was read from the text, with the character spans each part came from."""

    def __init__(self, text: str):
        self.text = text
        self.low = text.lower().replace("’", "'")
        self.spans: list[tuple[int, int]] = []
        self.sequence: list[DraftStage] = []
        self.conditions: list[DraftCondition] = []
        self.adds: list[DraftAdd] = []
        self.exits: list[DraftCondition] = []
        self.notes: list[str] = []
        self.unsupported: list[str] = []
        self.consumed_numbers: set[float] = set()
        self.drop_cats: set[str] = set()  # nl_rules condition categories this reading supersedes
        self.direction = 0  # -1 the entry follows a decline, +1 a rally

    def take(self, m: re.Match):
        self.spans.append(m.span())

    def masked(self) -> str:
        """The text with every span read here blanked, for the static-rule parser."""
        s = list(self.text)
        for a, b in self.spans:
            for i in range(a, min(b, len(s))):
                if s[i] not in "\n":
                    s[i] = " "
        # what is being traded stays visible to the static parser even when it sat inside a clause read here
        for m in re.finditer(r"\b(buy|buying|sell|selling|short|long|calls?|puts?|iron condor|iron butterfly|straddle|strangle|spread|covered)\b", self.text, re.I):
            s[m.start():m.end()] = list(self.text[m.start():m.end()])
        return "".join(s)


def read(text: str, symbol: str | None = None) -> Reading:
    r = Reading(text)
    low = r.low
    sym = symbol or "the underlying"

    # ---- 1. decline (or rally) from a reference, then a partial recovery (or fade)
    m = re.search(rf"{FALL}\s+(?:by\s+)?(?:at least\s+)?{N}\s*%\s*(?:or more\s*)?(from (?:the|its) open|below (?:the|its) open|from (?:the |its |yesterday'?s |the previous |the prior )?close)?"
                  rf"[^.;]{{0,70}}?(?:recovers?|retraces?|regains?|bounces? back|wins? back|gets? back|makes? back|recoups?)\s+{FRAC}\s+of\s+(?:that|the|its)\s+(?:decline|drop|loss|fall|move|dip|sell-?off|pullback|losses)", low)
    if m:
        x, ref = float(m.group(1)), ("prev_close" if m.group(2) and "close" in m.group(2) else "open")
        f = _frac(m.group(3))
        base = f"from_open_pct <= -{x:g}" if ref == "open" else f"change_pct <= -{x:g}"
        r.sequence += [DraftStage(name="drop", expr=base, text=f"{sym} is down {x:g}% from {LEVEL_TEXT[ref]}"),
                       DraftStage(name="recover", expr=f'price >= low_since("drop") + {f:g} * ({ref} - low_since("drop"))',
                                  text=f"price recovers {_frac_text(f)} of the decline from {LEVEL_TEXT[ref]} to the low made after the drop")]
        r.take(m)
        r.consumed_numbers.add(x)
        r.direction = -1
        if not m.group(2):
            r.notes.append(f"\"{m.group(0)[:40].strip()}...\": the {x:g}% decline is measured from that session's open (say \"from yesterday's close\" to change it).")
    else:
        m = re.search(rf"{RISE}\s+(?:by\s+)?(?:at least\s+)?{N}\s*%\s*(?:or more\s*)?(from (?:the|its) open|above (?:the|its) open|from (?:the |its |yesterday'?s |the previous |the prior )?close)?"
                      rf"[^.;]{{0,70}}?(?:gives? back|fades?|retraces?|loses|surrenders?|hands? back)\s+{FRAC}\s+of\s+(?:that|the|its)\s+(?:rally|gain|gains|move|advance|pop|rise)", low)
        if m:
            x, ref = float(m.group(1)), ("prev_close" if m.group(2) and "close" in m.group(2) else "open")
            f = _frac(m.group(3))
            base = f"from_open_pct >= {x:g}" if ref == "open" else f"change_pct >= {x:g}"
            r.sequence += [DraftStage(name="rally", expr=base, text=f"{sym} is up {x:g}% from {LEVEL_TEXT[ref]}"),
                           DraftStage(name="fade", expr=f'price <= high_since("rally") - {f:g} * (high_since("rally") - {ref})',
                                      text=f"price gives back {_frac_text(f)} of the rally from {LEVEL_TEXT[ref]} to the high made after it")]
            r.take(m)
            r.consumed_numbers.add(x)
            r.direction = 1

    # ---- 2. a failed breakout (or breakdown), optionally followed by a VWAP reclaim / loss
    m = re.search(r"(?:the\s+)?(?:first\s+)?(breakout|breakdown)\s+(?:attempt\s+)?(?:that\s+)?fail(?:s|ed)?|(?:failed|fake|false)\s+(breakout|breakdown)", low)
    if m:
        down = (m.group(1) or m.group(2)) == "breakdown"
        mm = re.search(r"opening[- ](\d+)[- ]?(?:minute|min)", low)
        mins = int(mm.group(1)) if mm else 30
        lvl = f"opening_range_{'low' if down else 'high'}({mins})"
        r.sequence += [DraftStage(name="breakout", expr=f"price {'<' if down else '>'} {lvl}", text=f"price breaks {'below' if down else 'above'} the first {mins} minutes' {'low' if down else 'high'}"),
                       DraftStage(name="failed", expr=f"price {'>' if down else '<'} {lvl}", text=f"the break fails: price is back {'above' if down else 'below'} that level")]
        r.take(m)
        r.drop_cats.add("orb")
        if not mm:
            r.notes.append(f"\"breakout\" is read as a move {'below' if down else 'above'} the opening range of the first 30 minutes (name another range to change it).")
    m = re.search(r"(?:price\s+)?(?:reclaims?|recaptures?|(?:gets?|moves?|crosses?|closes?|trades?|pops?)\s+back\s+(?:above|over))\s+(?:the\s+)?vwap", low)
    m2 = re.search(r"(?:price\s+)?(?:loses|rejects|(?:gets?|moves?|crosses?|closes?|trades?|falls?|slips?)\s+back\s+(?:below|under))\s+(?:the\s+)?vwap", low) if not m else None
    if (m or m2) and not re.search(r"(?:close|exit|sell|get out)[^.;]{0,40}$", low[:(m or m2).start()][-48:]):
        up = bool(m)
        stage = DraftStage(name="reclaim" if up else "lose_vwap", expr=f"crosses_{'above' if up else 'below'}(price, vwap)", text=f"price crosses back {'above' if up else 'below'} session VWAP")
        if r.sequence:
            r.sequence.append(stage)
        else:
            r.conditions.append(DraftCondition(expr=stage.expr, text=stage.text))
        r.take(m or m2)
        r.drop_cats.add("vwap")

    # ---- 2b. candle streaks phrased with the timeframe first ("three consecutive five-minute red candles")
    m = re.search(r"(\w+)\s+(?:consecutive|straight|successive)\s+(\w+)[- ]?(?:minute|min)\s+(red|down|bearish|green|up|bullish)\s+(?:candles?|bars?)(?:\s+in a row)?", low)
    if m and NL._num(m.group(1)) and NL._num(m.group(2)):
        n, tf, bear = int(NL._num(m.group(1))), f"{int(NL._num(m.group(2)))}m", m.group(3) in ("red", "down", "bearish")
        r.conditions.append(DraftCondition(expr=f'consecutive_{"bearish" if bear else "bullish"}_candles("{tf}") >= {n}', text=f"{n} consecutive {'red' if bear else 'green'} {tf} candles"))
        r.take(m)
        r.drop_cats.add("streak")

    # ---- 3. break of a prior-day level by the traded symbol
    m = re.search(r"(?:breaks?|takes? out|clears?|crosses?|moves?|trades?|pushes?)\s+(above|over|below|under|through)?\s*(?:yesterday'?s|the previous (?:day|session)'?s?|the prior (?:day|session)'?s?)\s+(high|low)", low)
    if m:
        hi = m.group(2) == "high"
        r.conditions.append(DraftCondition(expr=f"crosses_{'above' if hi else 'below'}(price, prev_{m.group(2)})", text=f"price breaks {'above' if hi else 'below'} yesterday's {m.group(2)}"))
        r.take(m)

    # ---- 4. another symbol's state ("but only if QQQ hasn't broken its own high")
    for m in re.finditer(r"\b([A-Z]{2,5})\b\s+(?:has|have|had)?\s*(n't|not|never)?\s*(?:yet\s+)?(?:broken|taken out|exceeded|cleared|made|printed|hit|set)\s+(?:above\s+|below\s+)?(?:its own|its|their own|yesterday'?s|the previous day'?s|a new)?\s*(?:prior\s+|previous\s+)?(high|low)",
                         text.replace("hasn’t", "hasn't").replace("hasnt", "has not")):
        t, neg, lvl = m.group(1), bool(m.group(2)) or "n't" in m.group(0), m.group(3)
        if t in NL.NOT_UNDERLYING:
            continue
        broke = f"{t}.high > {t}.prev_high" if lvl == "high" else f"{t}.low < {t}.prev_low"
        not_broke = f"{t}.high <= {t}.prev_high" if lvl == "high" else f"{t}.low >= {t}.prev_low"
        r.conditions.append(DraftCondition(expr=not_broke if neg else broke, text=f"{t} has {'not ' if neg else ''}traded {'above' if lvl == 'high' else 'below'} its own previous-day {lvl} today"))
        r.take(m)
    for m in re.finditer(r"\b([A-Z]{2,5})\b\s+(?:is|was|stays?|remains?|trades?|is trading)\s+(not\s+)?(up|down|green|red|higher|lower|positive|negative)\b(?:\s+on the (?:day|session))?", text):
        t = m.group(1)
        if t in NL.NOT_UNDERLYING or t == (symbol or ""):
            continue
        if re.match(r"\s+(?:by\s+)?(?:more|less)\s+than\b|\s+(?:a\s+|an\s+)?(?:at least|more than|over)?\s*\d", text[m.end():]):
            continue  # "QQQ is up more than SPY", "QQQ is up 1%": a comparison or a size follows, so this is not the bare "QQQ is up" (the research reader takes the whole clause)
        up = (m.group(3) in ("up", "green", "higher", "positive")) != bool(m.group(2))
        r.conditions.append(DraftCondition(expr=f"{t}.change_pct {'>' if up else '<'} 0", text=f"{t} is {'up' if up else 'down'} versus its previous close"))
        r.take(m)
    for m in re.finditer(r"\b([A-Z]{2,5})\b\s+(?:is|was|stays?|remains?|trades?|is trading|holds?)\s+(above|below|over|under)\s+(?:its\s+|the\s+)?vwap", text):
        t = m.group(1)
        if t in NL.NOT_UNDERLYING or t == (symbol or ""):
            continue
        up = m.group(2) in ("above", "over")
        r.conditions.append(DraftCondition(expr=f"{t}.price {'>' if up else '<'} {t}.vwap", text=f"{t} is {'above' if up else 'below'} its own VWAP"))
        r.take(m)

    # ---- 5. "unless ..." (negated conditions)
    for m in re.finditer(r"\bunless\b\s+([^.;,]+(?:,\s*(?!and\b|but\b|then\b)[^.;,]+)?)", low):
        clause = m.group(1).strip()
        got = []
        mm = re.search(rf"{NL.PREV}\s*(?:day|session)?\s*(?:was|is|closed|finished|ended)\s*(?:also\s+)?(red|green|down|up|lower|higher|negative|positive)", clause)
        if mm:
            w = mm.group(1)
            candle = w in ("red", "green")  # a red day closes below its OPEN; a down day closes below the prior close
            fieldn = "previous_session.open_to_close_pct" if candle else "previous_session.return_pct"
            neg = w in ("red", "down", "lower", "negative")
            got.append(DraftCondition(expr=f"not ({fieldn} {'<' if neg else '>'} 0)", text=f"unless yesterday {'closed below its open (a red day)' if candle and neg else 'closed above its open (a green day)' if candle else 'closed lower' if neg else 'closed higher'}"))
        else:
            for _cat, c in NL.find_conditions(clause):
                got.append(DraftCondition(expr=f"not ({c.expr})", text=f"unless {c.text}"))
        if got:
            r.conditions += got
            r.take(m)
        else:
            r.unsupported.append(f"unless {clause}")
            r.take(m)

    # ---- 6. scale-in ("add another contract after a further 0.5% decline")
    for m in re.finditer(rf"\badd(?:s|ing)?\s+(?:(another|one more|a second|an? additional|(?:\w+)\s+more)\s+)?(?:contracts?|lots?|units?|calls?|puts?|positions?|to the position)?\s*(?:after|when|if|on|once|at)\s+(?:it|price|the price|the underlying|[a-z]{{2,5}})?\s*"
                         rf"(?:{FALL}|{RISE})?\s*(?:an?\s+)?(?:further|additional|another|more|extra)?\s*(?:by\s+)?{N}\s*%\s*(?:further\s+|more\s+|lower|higher)?\s*(decline|drop|fall|dip|down|lower|rally|rise|gain|up|higher)?", low):
        x = float(m.group(2))
        words = m.group(0)
        up = bool(re.search(r"rall|rise|gain|\bup\b|higher|climb|jump|pop", words)) and not re.search(r"declin|drop|fall|dip|down|lower|sell", words)
        qty = 1
        mq = re.match(r"(\w+)\s+more", m.group(1) or "")
        if mq and NL._num(mq.group(1)):
            qty = int(NL._num(mq.group(1)))
        r.adds.append(DraftAdd(expr=f"price {'>=' if up else '<='} entry.underlying_price * (1 {'+' if up else '-'} {x / 100:g})", quantity=qty, max_adds=1,
                               text=f"the underlying is a further {x:g}% {'higher' if up else 'lower'} than at the first entry"))
        r.take(m)
        r.consumed_numbers.add(x)
        r.notes.append(f"\"a further {x:g}%\" is measured from the underlying's price when the first lot was entered.")

    # ---- 7. exits tied to a level ("close both positions when price returns to VWAP")
    for m in re.finditer(rf"(?:close|exit|sell|get out of|flatten|cover)\s+(?:out\s+)?(?:of\s+)?(?:both|all|the|every|my)?\s*(?:positions?|lots?|contracts?|trades?|it|them|everything)?\s*(?:when|once|if|as soon as|after)\s+(?:the\s+)?(?:price|it|the underlying|[a-z]{{2,5}})?\s*"
                         rf"(returns? to|gets? back to|comes? back to|reverts? to|reaches|touches|tags|hits|crosses? back above|crosses? back below|crosses? above|crosses? below|crosses|is back at|is back above|is back below|reclaims|loses|falls? below|drops? below|rises? above|moves? above|breaks? above|breaks? below|trades? above|trades? below)"
                         rf"\s+{LEVEL_RE}", low):
        rel, lvl = m.group(1), LEVELS[m.group(2)]
        if re.search(r"above|reclaims|rises|moves", rel):
            op = ">="
        elif re.search(r"below|loses|falls|drops", rel):
            op = "<="
        else:
            op = ">=" if r.direction <= 0 else "<="  # "returns to": from below after a decline, from above after a rally
            if r.direction == 0:
                r.notes.append(f"\"{rel} {m.group(2)}\": read as price rising back to {LEVEL_TEXT[lvl]} (the entry follows a decline); say \"falls back to\" for the opposite.")
        r.exits.append(DraftCondition(expr=f"price {op} {lvl}", text=f"price {'is back at or above' if op == '>=' else 'is back at or below'} {LEVEL_TEXT[lvl]}"))
        r.take(m)
    return r


def detect_direction(d: Draft) -> int:
    for c in d.conditions:
        if re.search(r"(from_open_pct|change_pct|return_pct)\s*<=?\s*-", c.expr):
            return -1
        if re.search(r"(from_open_pct|change_pct|return_pct)\s*>=?\s*\d", c.expr):
            return 1
    return 0


def read_costs(low: str) -> DraftCosts | None:
    c = DraftCosts()
    hit = False
    if re.search(r"conservative (?:slippage|costs?|fills?|execution|assumptions?)|realistic (?:costs?|slippage|fills?|execution)|pessimistic (?:fills?|slippage|costs?)|worst[- ]case (?:fills?|slippage)", low):
        c.preset, hit = "conservative", True
    elif re.search(r"\b(?:no|without|ignore|ignoring|zero|excluding)\s+(?:commissions?|costs?|fees?|slippage)", low):
        c.preset, hit = "none", True
    m = re.search(rf"\$\s*{N}\s*(?:per contract\s+)?(?:in\s+)?commissions?|commissions?\s+of\s+\$\s*{N}", low)
    if m:
        c.commission_per_contract, hit = float(m.group(1) or m.group(2)), True
    elif re.search(r"\b(?:include|including|add|with|after|net of)\s+(?:the\s+)?(?:commissions?|fees|transaction costs|trading costs)", low) and c.preset is None:
        c.commission_per_contract, hit = 0.65, True
    m = re.search(rf"{N}\s*%\s*(?:of\s+)?slippage|slippage\s+of\s+{N}\s*%", low)
    if m:
        c.slippage_pct, hit = float(m.group(1) or m.group(2)), True
    m = re.search(rf"\$\s*{N}\s*(?:per (?:share|contract)\s+)?(?:of\s+)?slippage|slippage\s+of\s+\$\s*{N}|{N}\s*cents?\s+(?:of\s+)?slippage", low)
    if m:
        v = m.group(1) or m.group(2)
        c.slippage_per_contract, hit = (float(v) if v else float(m.group(3)) / 100), True
    elif re.search(r"\b(?:include|including|add|with)\s+(?:some\s+)?slippage", low) and c.preset is None and c.slippage_pct is None:
        c.preset, hit = "conservative", True
    return c if hit else None


def read_mode(low: str) -> tuple[str | None, str | None]:
    """(pricing mode the user asked for, provider they named)."""
    mode = None
    if re.search(r"trade (?:prices?|prints?|bars?|data)|traded prices?|last[- ]trade|trade[- ]based|trade[- ]price|option trades|actual trades", low):
        mode = "trades"
    elif re.search(r"bid[/ -]?ask|quote(?:s|d)? (?:prices?|data)|using (?:historical )?(?:option )?quotes", low):
        mode = "historical"
    elif re.search(r"modeled|modelled|theoretical|black[- ]scholes|estimate the option|model(?:ed)? premiums?", low):
        mode = "modeled"
    elif re.search(r"market study|underlying only|skip the option|just the underlying|directional study", low):
        mode = "study"
    prov = "alpaca" if re.search(r"\balpaca\b", low) else "yahoo_options" if re.search(r"\byahoo\b", low) and mode == "trades" else None
    return mode, prov


def questions_for(text: str, d: Draft) -> list[dict]:
    """Clarifying questions for details whose meaning is genuinely ambiguous. Each offers the
    readings as one-click answers; the first option is the reading used if nothing is chosen."""
    if d.kind != "options":
        return []
    low = text.lower()
    sym = d.symbol
    out = []
    sl = re.escape(sym.lower())
    # the symbol only qualifies the number when it is tied to it ("1% move in SPY", "SPY rises 1%"), not merely named in the sentence
    und = rf"underlying|stock|share|index|price (?:move|increase|decrease|rise|drop|change)|move in|moves?\b|points?\b|(?:in|of|on) (?:the )?{sl}\b|{sl} (?:rises?|falls?|moves?|is up|is down|gains?|drops?|rall\w*|declin\w*)"
    prem = r"premium|option|contract|on the (?:call|put)s?|of the (?:call|put)s?|\broi\b|return on|credit|debit|of max|doubl"
    for kind, pat in (("target", rf"(?:target(?:ing|s)?|take[- ]profit|profit target|\btp\b)\s*(?:of|at|is|=|:|a|an|the)?\s*(?:a\s+|an\s+)?{N}\s*%|{N}\s*%\s*(?:profit\s+)?target"),
                      ("stop", rf"(?:stop(?:[- ]loss)?|stops?\s+out)\s*(?:of|at|is|=|:|a|an|the)?\s*(?:a\s+|an\s+)?{N}\s*%|{N}\s*%\s*stop")):
        m = re.search(pat, low)
        if not m:
            continue
        x = float(m.group(1) or m.group(2))
        # only the words that belong to this number: stop at the neighbouring "and ... N%" clause
        pre = re.split(r"\band\b|,|;", low[max(0, m.start() - 34):m.start()])[-1]
        post = re.split(r"\band\b|,|;", low[m.end():m.end() + 46])[0]
        window = pre + low[m.start():m.end()] + post
        if re.search(und, window) or re.search(prem, window):
            continue  # the user said which one
        u_field, p_field = ("underlying_target_pct", "premium_target_pct") if kind == "target" else ("underlying_stop_pct", "premium_stop_pct")
        opts = [{"label": f"{x:g}% move in {sym}", "patch": {u_field: x, p_field: None}, "detail": f"exit when {sym} itself has moved {x:g}% {'in favour' if kind == 'target' else 'against the position'} from the entry"},
                {"label": f"{x:g}% {'gain' if kind == 'target' else 'loss'} on the option", "patch": {u_field: None, p_field: x}, "detail": f"exit when the option position is {'up' if kind == 'target' else 'down'} {x:g}% of the premium paid"}]
        if x >= 15:
            opts.reverse()  # the underlying does not move 15%+ in the life of a short-dated trade: the option reading is the plausible one
        out.append({"id": kind, "text": f"You said {kind} {x:g}%. Do you mean a {x:g}% move in {sym} or a {x:g}% {'gain' if kind == 'target' else 'loss'} on the option?", "options": opts, "default": 0,
                    "assumed": opts[0]["label"]})
    return out


EXITISH = re.compile(r"\b(target|stop|profits?|loss|exit|sell|close (?:the|it|them|out|at|by)|out by|get out|from (?:the |my )?entry|hold)\b")
WINDOW = re.compile(r"\b(?:between|from)\s+(?P<a>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?)\s*(?:and|to|until|till|through|-)\s*"
                    r"(?P<b>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?(?:\s*(?:pacific|pst|pdt|pt|mountain|mst|mdt|mt|central|cst|cdt|ct|eastern|est|edt|et|new york)\b(?:\s*time)?)?)", re.I)


def read_window(text: str, d: Draft) -> tuple[int, int] | None:
    """An entry window ("between 10:00 and 15:00", "from 9:45 to 11:30 Eastern") becomes a condition on the clock.
    The static reader takes the first clock time it meets as THE entry time; inside a window that time is only
    where the window starts, so the entry is left to the conditions. Returns the span of text it read."""
    if d.kind != "options":
        return None
    for m in WINDOW.finditer(text):
        seg = m.group(0)
        if ":" not in seg and not re.search(r"[ap]\.?m", seg, re.I):
            continue  # "between 2 and 3" is not a clock window
        if re.search(r"(exit|sell|close|hold|out)\s*$", text[max(0, m.start() - 16):m.start()].lower().strip()):
            continue  # a window for the exit is not a window for the entry
        ts = NL.find_times(seg, None)
        if len(ts) != 2:
            continue
        a, b = sorted(t["minutes"] for t in ts)
        if not (570 <= a < b <= 960):
            continue
        expr = f"time >= hhmm({a // 60}, {a % 60}) and time <= hhmm({b // 60}, {b % 60})"
        if expr not in {c.expr for c in d.conditions}:
            d.conditions.append(DraftCondition(expr=expr, text=f"the time is between {a // 60:02d}:{a % 60:02d} and {b // 60:02d}:{b % 60:02d} New York time"))
        em = 570 if d.entry_time == "open" else 960 if d.entry_time == "close" else (int(d.entry_time[:2]) * 60 + int(d.entry_time[3:]) if d.entry_time and re.fullmatch(r"\d\d:\d\d", d.entry_time) else None)
        at_start = d.entry_time in {t["et"] for t in ts} or (em is not None and not a <= em <= b)  # the window's own start, or an entry time the window rules out
        if at_start and not re.search(rf"\b(?:at|enter at|buy at|entry at)\s+{re.escape(m.group('a').strip())}", text[:m.start()] + text[m.end():], re.I):
            d.entry_time, d.entry_time_stated = None, None
            d.notes.append(f"\"{seg.strip()}\" is read as a window, not an entry time: the position is opened on the first evaluation between {a // 60:02d}:{a % 60:02d} and {b // 60:02d}:{b % 60:02d} New York time at which every condition holds.")
        if any(t.get("assumed_tz") for t in ts):
            d.tz_assumed = ts[0]["assumed_tz"]
        return m.span()
    return None


def find_unsupported(text: str, spans: list[tuple[int, int]], d: Draft) -> list[str]:
    """Clauses that state a market condition but were not turned into any rule."""
    out, partly = [], []
    low = text.lower()
    pos = 0
    for part in re.split(r"(?<=[.;!?])\s+|,\s+(?=(?:and|but|then|unless|only|if|when|while|after|once|provided|as long as|except)\b)|\s+(?=(?:but only if|only if|only when|unless|provided that|as long as|except when|when|whenever|while)\b)", text):
        a = text.find(part, pos)
        if a < 0:
            a = pos
        b = a + len(part)
        pos = b
        pl = part.lower().strip(" ,.;")
        if len(pl) < 8 or not MARKET_WORDS.search(pl):
            continue
        if not re.search(r"\b(if|when|unless|until|after|once|only|while|before|provided|as long as|except|whenever)\b", pl):
            continue
        over = [(max(s, a) - a, min(e, b) - a) for s, e in spans if s < b and e > a]
        if over:
            if d.kind == "options":
                partly += _remainder(part, over, d.symbol)  # a clause that was only partly read: the rest of it is still owed
            continue
        # the tail of an exit statement ("exit | when SPY moves up 1% from entry or at 12:45") belongs to the exit the static parser read
        if re.search(r"(exit|exits|sell|sells|close|closes|out|profit|profits|stop|stops|target|targets)\s*$", text[max(0, a - 18):a].lower().strip()):
            continue
        used = {x for x in (d.underlying_target_pct, d.underlying_stop_pct, d.premium_target_pct, d.premium_stop_pct, d.underlying_target_abs, d.underlying_stop_abs) if x is not None}
        nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", re.sub(r"\d{1,2}:\d{2}", " ", pl))}
        if used and nums and nums <= used | {float(d.dte or 0)} and EXITISH.search(pl):
            continue
        # clauses the static parser understands: an exit / hold / time / size statement
        if re.search(r"\b(target|stop|take profit|hold|expir|exit at|sell at|close at|by the close|at the close|at the open|dte|delta|contracts?\b.*\bbuy)\b", pl) and not re.search(r"\b(unless|only if|provided|as long as|except)\b", pl):
            continue
        out.append(part.strip(" ,.;"))
    # anything the static parser already turned into a rule is not unsupported
    known = " ".join(c.text.lower() for c in list(d.conditions) + list(d.exit_conditions)) + " ".join(s.text.lower() for s in d.sequence)
    res = []
    for u in out:
        if not _covered(u.lower(), low, known, d):
            res.append(u)
        elif d.kind == "options":  # in a market question the rest of the clause IS the question, so only a strategy is checked
            res += _remainder(u, [], d.symbol)
    res += _unread_conditionals(text, spans, d, res + partly)
    return list(dict.fromkeys(res + partly))


# "when ...", "if ...", "after ...", "on days when ...", "only if ...": a clause that makes the entry conditional
COND_CLAUSE = re.compile(r"(?:\b(?:but\s+)?(?:only\s+)?(?:when(?:ever)?|if|after|once|provided(?:\s+that)?|as\s+long\s+as|following)\b|\bon\s+(?:days?|sessions?|mornings?)\s+(?:when|after|that|where|on\s+which|following)\b)"
                         r"\s+(?P<body>.+?)(?=\s*(?:\.(?!\d)|;|,\s*(?:\d+\s+contracts?|using|with\s+a|stop|target|take|exit|exiting|sell|selling|close|closing|and\s+(?:exit|sell|close|take|stop|hold|get\s+out))|$|"
                         r",?\s*(?:but\s+|and\s+)?(?:only\s+)?(?:if|when(?:ever)?|provided(?:\s+that)?|as\s+long\s+as|unless)\b|"
                         r"\s+(?:and\s+)?(?:exit(?:ing)?|sell\s+(?:it|them|at|when|after)|close\s+(?:it|them|at|when|the)|get\s+out|take\s+profits?|with\s+a\s+\d|using\s+\d|stop\s+(?:at|out)|hold\s+(?:to|until|for))\b))", re.I)
TIMEISH = re.compile(r"^\s*(?:the\s+)?(?:market|session|trading|bell|day)?\s*(?:opens?|open(?:ing)?|closes?|close|closing)\s*$|^\s*(?:\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?(?:\s+\w+)?|noon|midday|lunch)\s*$|^\s*(?:filled|assigned|expir\w*|possible|available)\b", re.I)


def _unread_conditionals(text: str, spans: list[tuple[int, int]], d: Draft, already: list[str]) -> list[str]:
    """Every clause that makes the entry conditional must have become a rule. The first pass above lets a whole clause through
    when it contains one phrase the static parser knows ("at the open ... after SPY closes down 3%") or no market vocabulary at all
    ("when Mercury is in retrograde"), so the condition in it could vanish without a word. Here each conditional clause is
    checked by itself: either a rule was read from it, or it is returned as not understood (the research reader is then
    asked, and what it cannot read either stays in front of the user)."""
    out = []
    seen = " ".join(already).lower()
    exprs = {c.expr for c in d.conditions}
    for m in COND_CLAUSE.finditer(text):
        body = m.group("body").strip(" ,.;")
        a, b = m.span("body")
        low = body.lower()
        if len(low) < 4 or TIMEISH.search(low) or low in seen or (seen and any(low in x or x in low for x in (u.lower() for u in already))):
            continue
        # part of an exit statement: "exit when SPY moves up 1%", "sell after 3 days", "close it if ..."
        before = text[max(0, m.start() - 28):m.start()].lower()
        if re.search(r"(?:exit|exits|sell|sells|close|closes|out|cover|profit|profits|stop|stops|target|targets|hold|holding)\s*(?:it|them|the\s+\w+|early|everything|out)?\s*$", before) and not re.search(r"\b(?:buy|enter|open)\b[^.;]*$", before[-12:]):
            continue
        if re.match(r"\d+\s+(?:trading\s+)?(?:days?|sessions?|hours?|minutes?|weeks?)\b", low) or re.match(r"(?:the\s+)?(?:open|close|bell)\b", low):
            continue  # "after 3 days", "after the open": a duration or a clock reference, not a market condition
        if any(s_ < b and e_ > a for s_, e_ in spans):
            continue  # the stateful reader took (at least part of) it: its remainder was checked above
        found = NL.find_conditions(low, [])
        if found and all(c.expr in exprs or NL._cat_of(c.expr) in {NL._cat_of(e) for e in exprs} for _, c in found):
            continue  # the static parser read a rule from this clause
        if d.weekdays and re.search(r"\b(?:mon|tues?|wed(?:nes)?|thur?s?|fri)(?:day)?s?\b", low) and not re.search(r"\d", low):
            continue  # "on Mondays": the weekday filter has it
        out.append(m.group(0).strip(" ,.;"))
    return out


def _remainder(clause: str, read: list[tuple[int, int]], symbol: str = "") -> list[str]:
    """What is left of a clause once the parts that became rules are taken out. "when SPY gaps down 1% and the
    moon is full": the gap is a rule, and "the moon is full" is still owed an answer. Only a sub-phrase with real
    content is returned (two or more words that are not filler, tickers, or exit / time vocabulary)."""
    from strata.ai import nl_graph as NG
    spans: list = []
    NL.find_conditions(clause.lower(), spans)
    s = clause
    for a, b in sorted(list(spans) + list(read), reverse=True):
        s = s[:max(0, a)] + " " * (min(len(s), b) - max(0, a)) + s[min(len(s), b):]
    s = WINDOW.sub(lambda m: " " * len(m.group(0)), s)
    s = NL.TIME_RE.sub(lambda m: " " * len(m.group(0)) if (m.group(2) or m.group(3) or m.group(4)) else m.group(0), s)
    s = re.sub(r"^\s*(?:and\s+|but\s+)?(?:only\s+)?(?:if|when|whenever|while|provided(?:\s+that)?|as long as|after|once)\b", " ", s, flags=re.I)
    tick = {t.lower() for t in NL.TICKERS} | {symbol.lower()}
    out = []
    for bit in NG._clauses_of(s):
        if EXITISH.search(bit.lower()) or re.search(r"\b(?:buy|sell|enter|contracts?|calls?|puts?|dte|delta|expir\w*|strike|how often|how many|what|backtest|test|show|tell)\b", bit.lower()):
            continue
        words = [w for w in re.findall(r"[a-z][a-z'&-]{2,}", bit.lower()) if w not in NG.FILLER and w.rstrip("s") not in NG.FILLER and w not in tick]
        if len(words) >= 2:
            out.append(re.sub(r"\s+", " ", bit).strip(" ,.;"))
    return out


def _covered(clause: str, low: str, known: str, d: Draft) -> bool:
    spans: list = []
    found = NL.find_conditions(clause, spans)
    exprs = {c.expr for c in d.conditions}
    if found and all(c.expr in exprs or NL._cat_of(c.expr) in {NL._cat_of(e) for e in exprs} for _, c in found):
        return True
    return False


def _read_with_research_reader(d: Draft, known: list[str], window: tuple[str, int, int] | None = None, full_text: str | None = None) -> None:
    """A condition the strategy reader did not understand is offered to the research question reader, which knows
    many more (other symbols, relative performance, gaps, volatility, streaks, weekdays, moving averages). A
    sub-phrase is accepted only when every word of it was understood and it can be evaluated at the entry
    instant; each reading is stated in the notes. What is left stays in `unsupported`, in the user's words."""
    if not d.unsupported or d.kind != "options":
        return
    from strata.ai import nl_graph as NG
    try:
        from strata.data.store import MarketStore
        store = MarketStore()
    except Exception:  # noqa: BLE001
        store = None
    left = []
    have = {c.expr.replace(" ", "") for c in d.conditions}
    for u in d.unsupported:
        if re.match(r"\s*(?:unless|except)\b", u.lower()):
            left.append(u)  # a negated clause is not rewritten here
            continue
        phrase = u.replace(window[0], " ") if window and window[0] in u else u  # a clock window in the clause was already read
        try:
            got, rest = NG.entry_conditions(phrase, d.symbol, known, store, d.entry_time, partial=True)
        except Exception:  # noqa: BLE001 - the fallback must never break a request the strategy reader handled
            got, rest = [], [phrase]
        if not got:
            if phrase.strip(" ,.;") and (phrase == u or re.search(r"[a-z]{3,}", re.sub(r"\b(?:and|but|only|when|if|while|then|also|too)\b", " ", phrase.lower()))):
                left.append(u if phrase == u else phrase.strip(" ,.;"))
            continue
        for g in got:
            if g["expr"].replace(" ", "") not in have:
                d.conditions.append(DraftCondition(expr=g["expr"], text=g["text"]))
                have.add(g["expr"].replace(" ", ""))
        # a size that this condition used is not ALSO a target or a stop: "up at least 0.5% from the open" was being read twice
        used = {float(x) for x in re.findall(rf"{N}\s*%", phrase)} & {abs(float(x)) for g in got for x in re.findall(r"-?\d+(?:\.\d+)?", g["expr"])}
        rest_txt = (full_text or "").replace(phrase.strip(" ,.;"), " ")
        for f in ("underlying_target_pct", "underlying_stop_pct", "premium_target_pct", "premium_stop_pct"):
            v = getattr(d, f)
            if v is not None and float(v) in used and not re.search(rf"(?<![\d.]){float(v):g}\s*%", rest_txt):
                setattr(d, f, None)
        if d.entry_time == "open" and re.match(r"\s*(?:from|off|since|versus|vs)\b", (d.entry_time_stated or "").lower()) and any("from_open" in g["expr"] for g in got):
            d.entry_time, d.entry_time_stated = None, None
            d.notes.append("\"from the open\" describes the move, not when to enter (a move from the open is always zero at the open). The position is opened on the first evaluation at which every condition holds.")
        d.notes.append("Read as: " + "; ".join(f"{g['text']} (`{g['expr']}`)" for g in got) + ". Evaluated at the entry time.")
        for a_ in dict.fromkeys(a_ for g in got for a_ in g.get("assumptions", [])):
            d.notes.append(a_)
        # what is left of the clause counts only if it is itself about the market: the order ("buy 0DTE calls at 10:00") is not an unread condition
        left += [x for x in rest if re.search(r"[a-z]{3,}", x.lower()) and not re.search(r"\b(?:buy|sell|short|enter|contracts?|calls?|puts?|spreads?|condors?|straddles?|strangles?|dte|delta|expir\w*|strike)\b", x.lower())
                 and not re.fullmatch(r"\s*(?:on\s+)?(?:days?|sessions?|mornings?)\s*", x.lower())]
    d.unsupported = left
    # the bridge can restate what the static parser already read loosely ("lost 2.5% the day before" as "the previous session fell"):
    # a bare sign next to a stated size on the same quantity adds nothing and is dropped
    sized = {m.group(1) for c in d.conditions for m in [re.fullmatch(r"(previous_session\.return_pct|return_pct\(1\)|from_open_pct|gap_pct)\s*(?:<=|>=)\s*-?\d+(?:\.\d+)?", c.expr.strip())] if m and not re.search(r"(?:<=|>=)\s*-?0(?:\.0+)?$", c.expr.strip())}
    same = {"previous_session.return_pct": {"previous_session.return_pct", "return_pct(1)"}, "return_pct(1)": {"previous_session.return_pct", "return_pct(1)"}}
    def bare(c):
        m = re.fullmatch(r"(previous_session\.return_pct|return_pct\(1\)|from_open_pct|gap_pct)\s*(?:<|>|<=|>=)\s*-?0(?:\.0+)?", c.expr.strip())
        return bool(m and (same.get(m.group(1), {m.group(1)}) & sized))
    d.conditions = [c for c in d.conditions if not bare(c)]


def enrich(text: str, known: list[str], default_symbol: str = "SPY") -> Draft:
    """A fresh draft from the whole request: the stateful parts first, then the static parser
    on what is left, then questions and anything that could not be represented."""
    sym = NL.find_symbol(text, [k for k in known if k not in NL.NOT_UNDERLYING])
    r = read(text, sym or default_symbol)
    d = NL.parse_fresh(r.masked(), known, default_symbol)
    d.title = text.strip()[:90]
    low = r.low
    if r.drop_cats:
        d.conditions = [c for c in d.conditions if NL._cat_of(c.expr) not in r.drop_cats]
    # numbers the stateful reading used are not targets or stops
    for f in ("underlying_target_pct", "underlying_stop_pct", "premium_target_pct", "premium_stop_pct"):
        if getattr(d, f) in r.consumed_numbers:
            setattr(d, f, None)
    intraday_ctx = bool(r.adds or r.sequence or any("vwap" in c.expr for c in r.exits))
    if intraday_ctx:
        # "a 1% decline" beside intraday constructs is an intraday decline, not "the previous session fell 1%"
        for i, c in enumerate(d.conditions):
            m = re.fullmatch(r"previous_session\.return_pct (<=|>=) (-?\d+(?:\.\d+)?)", c.expr)
            if m and not re.search(rf"{NL.PREV}|down day|red day|up day|green day|\d\s*%\s*(?:down|up|red|green)\s*(?:day|session)", low):
                x = abs(float(m.group(2)))
                fell = m.group(1) == "<="
                d.conditions[i] = DraftCondition(expr=f"from_open_pct {'<=' if fell else '>='} {'-' if fell else ''}{x:g}", text=f"{d.symbol} is {'down' if fell else 'up'} {x:g}% from the session open")
                d.notes.append(f"\"a {x:g}% {'decline' if fell else 'rally'}\" is read as a move from that session's open, because the rest of the request is intraday. Say \"the previous day\" for a prior-session move.")
                r.direction = r.direction or (-1 if fell else 1)
    d.sequence = r.sequence
    d.conditions = d.conditions + [c for c in r.conditions if c.expr not in {x.expr for x in d.conditions}]
    d.adds = r.adds
    d.exit_conditions = r.exits
    d.notes += r.notes
    if (r.sequence or any("crosses_" in c.expr for c in r.conditions)) and d.entry_time in ("open", "09:30"):
        d.entry_time = None  # an intraday trigger decides the entry time
    d.costs = read_costs(low)
    mode, prov = read_mode(low)
    if mode and (d.kind == "options" or mode == "study"):
        d.mode = mode
    d.provider = prov
    # a trade was described (enter / add / close ...) but no option position: it is still a strategy, and which position is asked
    trade_words = re.search(r"\b(enter|entering|buy|sell|go long|go short|add another|add \w+ more|contracts?|close (?:both|all|the) positions?|scale in)\b", low)
    asks = re.search(r"how often|how frequent|what percent|what fraction|probab|odds of|how many times|how likely|what happens|\bdoes\b|\bdo\b|is there|tend to", low)
    if d.kind == "study" and d.structure is None and trade_words and not asks and (d.sequence or d.adds or d.exit_conditions or d.conditions):
        d.kind, d.study = "options", None
    d.questions = questions_for(text, d)
    for q in d.questions:  # the draft carries the plausible reading until the user answers
        for k, v in q["options"][q["default"]]["patch"].items():
            setattr(d, k, v)
    # what was not represented is worked out last, from the draft as it now stands: a number is only "already used
    # by a target or a stop" if it still is one after the questions' default readings were applied
    wspan = read_window(text, d)
    d.unsupported = list(dict.fromkeys(r.unsupported + find_unsupported(text, r.spans, d)))
    _read_with_research_reader(d, known, (text[wspan[0]:wspan[1]], *wspan) if wspan else None, text)
    if d.kind == "options" and d.structure is None:
        bias = -1 if r.direction < 0 or detect_direction(d) < 0 else 1
        opts = [{"label": "Calls (long)", "patch": {"structure": "long_call"}, "detail": "buy calls: profits if the underlying rises"},
                {"label": "Puts (long)", "patch": {"structure": "long_put"}, "detail": "buy puts: profits if the underlying falls"}]
        d.questions.append({"id": "structure", "text": "Which option position should this enter? The request describes when to trade but not what.", "options": opts, "default": None, "essential": True,
                            "hint": "The entry follows a decline: calls bet on a rebound, puts on continuation." if bias < 0 else ""})
    return d
