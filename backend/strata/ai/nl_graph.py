"""A market question in plain English -> a research graph.

This is a compositional reader, not a list of question templates. A question is split into
an outcome ("is SPY more likely to rally in the afternoon"), any number of conditions
("the morning sold off", "QQQ outperformed SPY", "VIX increased overnight") and modifiers
("controlling for volatility", "compare with QQQ and IWM", "since 2015"). Each condition
is read as SUBJECT x SPAN x PREDICATE:

    subject     which symbol (the one being studied unless another is named)
    span        overnight | the morning | the first N minutes | the previous session |
                the last N sessions | the session itself | so far today
    predicate   rose / fell (with a size, or "large"), out- or under-performed another
                symbol, is above / below a level, a streak, a weekday, a calendar event...

so "QQQ outperformed SPY in the morning" needs no rule of its own: it is the morning span
applied to two subjects and compared. Every reading becomes typed graph nodes
(`research/graph.py`), the graph validator decides whether the result is a legitimate
analysis, and the same builder serves follow-ups ("apply the same idea to QQQ", "combine
the first study's condition with the second's filter") by editing the parsed core.

What it will not do: guess. Words that are not understood are returned in `unsupported`
and shown to the user; a qualitative size ("large") is turned into a number from the
symbol's own history and reported as an assumption; and when a language model is
configured its graph goes through exactly the same validator.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np

from strata.ai import nl_rules as NL
from strata.engine import expr as X
from strata.research import graph as G
from strata.research.features import FEATURES, feature_def
from strata.research.matching import DEFAULT_CONTROLS

N = NL.N
VOL_WORDS = r"(?:the\s+)?(?:vix|volatility index|vol index|fear index|fear gauge)"
SUBJ_ALIASES = {"nasdaq": "QQQ", "the nasdaq": "QQQ", "tech": "QQQ", "tech stocks": "QQQ", "small caps": "IWM", "small-caps": "IWM", "the russell": "IWM", "russell": "IWM", "the s&p": "SPY", "s&p 500": "SPY",
                "the s&p 500": "SPY", "s&p": "SPY", "the dow": "DIA", "dow": "DIA", "the market": None, "stocks": None, "it": None, "price": None, "the index": None}
STRONG_DOWN = r"sold off|sell-?off|sells off|selloff|tumbl\w+|plung\w+|slump\w+|sank|sinks|crash\w+|collaps\w+|slid\w*|drop(?:s|ped)? sharply|fell sharply"
STRONG_UP = r"surg\w+|soar\w+|spik\w+|jump\w+|rip\w*|rall(?:y|ied|ies) sharply|rose sharply|shot up"
PLAIN_DOWN = r"fell|falls?|fall(?:ing|en)|dropp?(?:ed|s|ing)?|declin\w+|lost|los(?:es|ing)|was down|is down|were down|down|decreas\w+|weak\w*|red|negative|lower|slipp\w+|dipp?\w*|pull(?:s|ed)? back"
PLAIN_UP = r"rose|ris(?:es|ing|en)?|gain(?:ed|s|ing)?|rall(?:y|ied|ies|ying)|was up|is up|were up|up|green|increas\w+|climb\w+|advanc\w+|strong\w*|positive|higher|bounc\w+|recover\w+|rebound\w+"
BIG = r"large|big|sharp|significant|major|heavy|steep|substantial|strong|huge|severe|unusually large|outsized|hard"
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "couple of": 2, "few": 3}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
# stored symbols that are also ordinary words are only read as tickers when written in capitals
ORDINARY = set("all now day low high open key any are for has was one two out big see well next fast cost gain loss true real peak play run fun car cat on it at by be go so up am an as do he if in is me my no of or to us we oil gold cash life love good best ever edge fact form free full hope idea info land line list long main mind move near news nice odd old own part plan plus post pure rare ride safe sign star step sure task team tech test tool turn unit view wave way week work year".split())
INTRO = re.compile(r"\b(?:when|whenever|if|after|following|given(?:\s+that)?|once|provided(?:\s+that)?|on\s+(?:days|sessions)\s+(?:when|where|that|after|on which)|on\s+(?:days|sessions))\b")
LEADS = re.compile(r"\b(?:leads?\s+to|led\s+to|predicts?|is\s+followed\s+by|are\s+followed\s+by|tends?\s+to\s+be\s+followed\s+by|results?\s+in|precedes?|mean\s+for|means?\s+for)\b")
CONTROL = re.compile(r"(?:,?\s*(?:even\s+)?(?:after\s+|when\s+|while\s+)?(?:controlling|adjusting|accounting|correcting|controlled|adjusted)\s+for|,?\s*(?:even\s+)?holding\s+|,?\s*matched\s+on|,?\s*(?:after\s+)?conditioning\s+on|,?\s*regardless\s+of|,?\s*independent(?:ly)?\s+of)\s+"
                     r"(.+?)(?:\s+(?:constant|fixed|equal))?(?=[?.;]|$|,\s*(?:and\s+)?(?:compare|since|from|between|over the))")
COMPARE = re.compile(r"\b(?:compar\w+\s+(?:that|this|it|the same|those|these)?\s*(?:behaviou?r|result|finding|pattern|effect|idea|analysis)?\s*(?:with|to|against|across|for|on)|same\s+(?:for|on|with)|also\s+(?:for|on|check)|(?:what|how)\s+about|versus|vs\.?)\s+(.+?)(?=[?.;]|$)")
QUESTION = re.compile(r"[,;]\s*(?:then\s+)?(?:how\s+(?:often|likely|frequently|many\s+times)|what\s+(?:is|are|happens|was)|what'?s|does|do|did|is|are|will|would|can|the\s+(?:probability|odds|chance|likelihood)|"
                      r"what\s+(?:percentage|fraction|share)|(?:on\s+)?how\s+many)\b")
HYPOTHESIS = re.compile(r"\b(?:tend|tends|tendency|whether|more likely|less likely|is it true|investigate|test(?:ing)?\s+(?:whether|if)|edge|significan\w+|predict\w*|hypothesis|validate|confirm\w*)\b")


@dataclass
class Clause:
    label: str
    span: str  # overnight | intraday | prev | last_n | day | static
    feature: str | None = None
    expr: str | None = None  # numeric (with op / value) or boolean (op None)
    op: str | None = None
    value: float | None = None
    hi: float | None = None
    at: str = "open"
    source: str = ""
    sign: int = 0  # -1 a decline, +1 a rise (lets "reverse" / "continue" resolve a direction)
    measure: str = ""  # wording of the value being compared
    n: int = 0  # sessions for last_n
    subject: str | None = None
    other: str | None = None


@dataclass
class Core:
    symbol: str
    clauses: list[Clause] = field(default_factory=list)
    outcome: dict = field(default_factory=lambda: {"op": "forward_return", "params": {"entry": "open", "horizon": "session", "direction": "up"}})
    anchor: dict | None = None
    controls: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None


# ----------------------------------------------------------------------------- small readers
def _numw(tok: str | None) -> float | None:
    if tok is None:
        return None
    tok = tok.strip().lower()
    if tok in NUMW:
        return float(NUMW[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def _round_thr(x: float) -> float:
    x = abs(float(x))
    if x >= 1:
        return round(x * 4) / 4
    if x >= 0.2:
        return round(x * 20) / 20
    return max(0.01, round(x, 2))


class Reader:
    """State shared while one question is read."""

    def __init__(self, text: str, known: list[str], default_symbol: str, store=None):
        self.text, self.low = text, " " + text.lower().strip() + " "
        self.known = [k.upper() for k in known]
        self.store = store
        self.assumptions: list[str] = []
        self.unsupported: list[str] = []
        self.symbol = (NL.find_symbol(text, self.known) or default_symbol or "SPY").upper()
        self.vol = "VIX"
        if store is not None:
            try:
                self.vol = store.symbol_meta(self.symbol).vol_index or "VIX"
            except Exception:  # noqa: BLE001
                pass
        self._ft = None
        # how the volatility index may be named: the usual words, or the stored index's own symbol
        self.vol_re = VOL_WORDS if self.vol.lower() == "vix" else f"(?:{VOL_WORDS}|\\b{re.escape(self.vol.lower())}\\b)"

    # ----- data-aware helpers
    def ft(self):
        if self._ft is None and self.store is not None:
            from strata.research.features import get_features
            try:
                self._ft = get_features(self.store, self.symbol)
            except LookupError:
                self._ft = False
        return self._ft or None

    def tick(self, candidates: list[str]) -> str:
        """The candidate clock time observable on the most stored sessions (first wins a tie)."""
        ft = self.ft()
        if ft is None:
            return candidates[0]
        best, cnt = candidates[0], -1
        for c in candidates:
            k = int(ft._tick(c)[1].sum())
            if k > cnt * 1.5 and k > cnt:  # a later candidate must be clearly better to displace an earlier one
                best, cnt = c, k
        return best

    def typical(self, feature: str | None, at: str, q: float, fallback: float) -> float:
        ft = self.ft()
        if ft is None or feature is None:
            return fallback
        try:
            x = ft.col(feature, at)
        except (KeyError, ValueError):
            return fallback
        x = np.abs(x[np.isfinite(x)])
        return _round_thr(float(np.quantile(x, q))) if x.size >= 100 else fallback

    def subject(self, s: str | None) -> str | None:
        """A phrase -> ticker (None = the symbol being studied)."""
        if not s:
            return None
        s = s.strip().lower()
        if re.fullmatch(self.vol_re, s):
            return self.vol
        if s in SUBJ_ALIASES:
            return SUBJ_ALIASES[s]
        up = s.upper().lstrip("$")
        if up in self.known or up in NL.TICKERS or (X.is_ticker(up) and up not in NL.NOT_UNDERLYING and len(up) <= 5 and up == s.upper() and s.isupper()):
            return None if up == self.symbol else up
        return None


SUBJ_RE = r"(?:\$?[A-Za-z]{1,5}\b|" + VOL_WORDS + "|" + "|".join(sorted((re.escape(k) for k in SUBJ_ALIASES), key=len, reverse=True)) + ")"


def _find_subject(r: Reader, clause: str) -> tuple[str | None, bool]:
    """(ticker or None for the studied symbol, was a subject named explicitly)."""
    m = re.search(r.vol_re, clause)
    if m:
        return r.vol, True
    for alias, tk in sorted(SUBJ_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if tk and re.search(rf"\b{re.escape(alias)}\b", clause):
            return (None if tk == r.symbol else tk), True
    for tok in re.findall(r"\b[A-Za-z]{2,5}\b", clause):
        up = tok.upper()
        if (up in r.known or up in NL.TICKERS) and up not in NL.NOT_UNDERLYING and (tok.isupper() or up in ("SPY", "QQQ", "IWM", "DIA", "SPX", "NDX") or (up in r.known and len(up) >= 3 and tok.lower() not in ORDINARY)):
            return (None if up == r.symbol else up), True
    return None, False


# ----------------------------------------------------------------------------- measures
def _measure(r: Reader, subj: str | None, span: str, extra: dict, tick: str) -> dict:
    """The value a span refers to, for a subject: a catalog feature when one exists (fast
    path), otherwise a rule-language expression. `at` is where it is read."""
    s = subj
    name = s or r.symbol
    if span == "overnight":
        return {"feature": "gap" if s is None else f"{s}_gap", "at": "open", "label": f"{name}'s opening gap", "unit": "%", "q": ("gap", "open"), "e": "gap_pct" if s is None else f"{s}.gap_pct"}
    if span == "prev":
        return {"feature": "ret_1" if s is None else f"{s}_ret_1", "at": "open", "label": f"{name}'s previous session (close to close)", "unit": "%", "q": ("ret_1", "open"),
                "e": "return_pct(1)" if s is None else f'return_pct("{s}", 1)'}
    if span == "day":
        return {"feature": "ret_1" if s is None else None, "expr": None if s is None else f'return_pct("{s}", 1)', "at": "close", "label": f"{name}'s session (close to close)", "unit": "%", "q": ("ret_1", "close"),
                "e": "return_pct(1)" if s is None else f'return_pct("{s}", 1)'}
    if span == "last_n":
        k = int(extra.get("n") or 5)
        cat = f"ret_{k}" if (s is None and f"ret_{k}" in FEATURES) else None
        return {"feature": cat, "expr": None if cat else (f"return_pct({k})" if s is None else f'return_pct("{s}", {k})'), "at": "open", "label": f"{name}'s return over the last {k} sessions", "unit": "%",
                "q": (cat or "ret_5", "open"), "e": f"return_pct({k})" if s is None else f'return_pct("{s}", {k})', "n": k}
    # intraday spans are read at a clock time
    at = extra["at"] if span == "clock" else tick
    if span == "first_n":
        mins = int(extra.get("minutes") or 60)
        at = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
        lab = f"{name}'s move in the first {mins} minutes"
    else:
        lab = f"{name}'s move since the open, at {at}"
    return {"feature": "from_open" if s is None else None, "expr": None if s is None else f"{s}.from_open_pct", "at": at, "label": lab, "unit": "%", "q": ("from_open", at), "e": "from_open_pct" if s is None else f"{s}.from_open_pct"}


# ----------------------------------------------------------------------------- clause readers
# words that carry no condition of their own: what is left of a phrase after every reader has taken its part is
# compared against this list, and anything else is reported as not understood instead of being dropped silently
FILLER = set("""a an the it its is was are were be been being has have had having does do did will would can could should may might of on in at by to from for with than then that this these those there their and or but also as so
 if when whenever after following once given more less least most over under about around only just very really ever tend tends tendency likely unlikely happen happens happened what how often much many which whether find out show me tell
 please day days session sessions market markets price prices stock stocks time times case cases previous prior preceding yesterday yesterdays overnight morning afternoon today last past week weeks month months first hour hours minute
 minutes open opens opened opening close closes closed closing intraday since before during noon lunch midday same row consecutive straight successive percent pct or moves move moved moving one two three four five six seven eight nine ten
 trading again still already then both either each every any some no not into onto up down higher lower level levels half between above below back out off far away exactly roughly approximately large big sharp significant major heavy
 steep substantial strong huge severe unusually outsized hard small slight slightly little bit kind sort type also too etc""".split())


class _Work:
    """A phrase being consumed: every reader blanks the part it understood."""

    def __init__(self, text: str):
        self.s = " " + text.strip().lower() + " "

    def take(self, m) -> None:
        if m is not None:
            self.s = self.s[:m.start()] + " " * (m.end() - m.start()) + self.s[m.end():]

    def find(self, pat: str):
        return re.search(pat, self.s)

    def left(self, r: "Reader") -> list[str]:
        skip = FILLER | {k.lower() for k in r.known} | {t.lower() for t in NL.TICKERS} | {"vix", "volatility", "index"}
        return [w for w in re.findall(r"[a-z][a-z'&-]{2,}", self.s) if w not in skip and w.rstrip("s") not in skip]


def _span_match(c: str):
    """(span, extra, match) - which stretch of time a condition is about."""
    m = re.search(r"first\s+(\d+)\s*(?:minutes|min|m)\b|opening\s+(\d+)\s*(?:minutes|min)|first\s+(half[- ]hour|hour|two hours)", c)
    if m:
        mins = int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else {"half-hour": 30, "half hour": 30, "hour": 60, "two hours": 120}[m.group(3)]
        return "first_n", {"minutes": mins}, m
    m = re.search(r"\bovernight\b|pre-?market|\bat the open\b|\bgap\w*|\bopen(?:s|ed|ing)?\s+(?:\S+\s+){0,5}?(?:lower|higher|down|up)\b", c)
    if m:
        return "overnight", {}, m
    # an explicit clock time: "is up more than 1% at 10:30", "down 1% by 11:00 am" (New York time)
    m = re.search(r"\b(?:at|by|as of|around)\s+(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?!\s*%)\b|\b(?:at|by)\s+(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", c)
    if m:
        h, mi, ap = int(m.group(1) or m.group(4)), int(m.group(2) or 0), (m.group(3) or m.group(5) or "").replace(".", "")
        if ap == "pm" and h < 12:
            h += 12
        elif not ap and h < 8:
            h += 12  # "at 2:30" in a trading day is the afternoon
        if 9 * 60 + 30 < h * 60 + mi < 16 * 60:
            return "clock", {"at": f"{h:02d}:{mi:02d}"}, m
    m = re.search(r"\bmorning(?:\s+session)?\b|by (?:noon|midday|lunch)|before (?:noon|lunch)|first half of the (?:day|session)", c)
    if m:
        return "morning", {}, m
    m = re.search(r"(?:over|in|during)?\s*(?:the\s+)?(?:last|past|previous|prior|preceding|trailing)\s+(\w+|\d+)\s+(day|session|week|month)s\b", c)
    if m and _numw(m.group(1)):
        return "last_n", {"n": int(_numw(m.group(1))) * {"day": 1, "session": 1, "week": 5, "month": 21}[m.group(2)]}, m
    m = re.search(r"(?:over|in|during)\s+the\s+(?:last|past|previous|prior)\s+(week|month)\b", c)
    if m:
        return "last_n", {"n": 5 if m.group(1) == "week" else 21}, m
    m = re.search(rf"{NL.PREV}\s*(?:day|session|close)?|\byesterday\b|the day before", c)
    if m and not re.search(r"(?:previous|prior|last) (?:week|month)'?s? (?:high|low)", c):
        return "prev", {}, m
    m = re.search(r"so far|since the open|(?:from|off|versus|vs\.?|against|relative to)\s+(?:the|its|their)\s+open(?:ing)?(?:\s+price)?\b|\bintraday\b|\btoday\b|on the day\b|by then", c)
    if m and not (m.group(0).strip() in ("on the day", "today") and re.search(r"\bclos(?:es|ed|ing)\b", c)):  # "closes down 2% on the day" is about the close, not the session so far
        return "sofar", {}, m
    m = re.search(r"\b(?:days?|sessions?)\b|\bclos(?:e|es|ed|ing)\b", c)
    if m:
        return "day", {}, m
    return None, {}, None


def _read_clause(r: Reader, clause: str, prev_span: tuple[str, dict] | None, noon: str) -> list[Clause] | None:
    """Every condition inside one phrase. Returns None when nothing in it was understood;
    words left over next to an understood condition go to `r.unsupported`."""
    w = _Work(clause)
    src = clause.strip()
    out: list[Clause] = []
    # ----- relative performance: A outperformed B (over a span)
    m = w.find(rf"({SUBJ_RE})?\s*(?:has\s+|had\s+|is\s+|was\s+|have\s+)?(out|under)perform\w*\s+({SUBJ_RE})") or \
        w.find(rf"({SUBJ_RE})\s+(?:is|was|has been|were)\s+(stronger|weaker)\s+than\s+({SUBJ_RE})") or w.find(rf"({SUBJ_RE})\s+(beat|trailed|lagged|led)\s+({SUBJ_RE})") or \
        w.find(rf"({SUBJ_RE})\s+(?:is|was|are|were|has been)\s+(up|higher|down|lower)\s+(?:by\s+)?more\s+than\s+({SUBJ_RE})") or \
        w.find(rf"({SUBJ_RE})\s+(?:has\s+|have\s+|had\s+)?(gained|risen|rallied|rose|fallen|dropped|lost|fell)\s+more\s+than\s+({SUBJ_RE})")
    if m:
        a, b = r.subject(m.group(1)), r.subject(m.group(3))
        # "A is down more than B" / "A fell more than B": A did worse than B
        up = m.group(2) in ("out", "stronger", "beat", "led", "up", "higher", "gained", "risen", "rallied", "rose")
        w.take(m)
        span, extra, sm = _span_match(w.s)
        w.take(sm)
        sp, ex = (span, extra) if span not in (None, "day") else (prev_span or ("prev", {}))
        if span is None and prev_span:
            r.assumptions.append(f"\"{src}\" is read over the same span as the condition before it.")
        elif span is None:
            r.assumptions.append(f"\"{src}\" names no time span; the previous session (close to close) is used.")
        ma, mb = _measure(r, a, sp, ex, noon), _measure(r, b, sp, ex, noon)
        mm = w.find(rf"by\s+(?:more than|at least|over)?\s*{N}\s*(?:%|percent|pct|points?)")
        size = float(mm.group(1)) if mm else None
        w.take(mm)
        na, nb = a or r.symbol, b or r.symbol
        out.append(Clause(f"{na} has {'out' if up else 'under'}performed {nb}" + (f" by {size:g} points or more" if size else "") + f" ({_span_words(sp, ex, ma['at'])})", sp, None, f"({ma['e']}) - ({mb['e']})",
                          ">=" if (up and size) else ">" if up else "<=" if size else "<", (size or 0.0) * (1 if up else -1), None, ma["at"], src, 0, f"{na} minus {nb}, {_span_words(sp, ex, ma['at'])}", ex.get("n", 0), a, b))
    # ----- volatility index level
    m = w.find(rf"{r.vol_re}\s*(?:is|was|are|were|stays|closed|closes|trades|trading)?\s*(above|below|over|under|greater than|less than|at least|at most|higher than|lower than|>|<)\s*{N}(?!\s*%)")
    if m:
        lt = m.group(1) in ("below", "under", "less than", "at most", "lower than", "<")
        out.append(Clause(f"the volatility index closed {'below' if lt else 'above'} {float(m.group(2)):g} the session before", "static", "vix_prev", None, "<" if lt else ">", float(m.group(2)), None, "open", src, 0, "the volatility index (previous close)"))
        w.take(m)
    m = w.find(r"(?:(?:realis|realiz)ed\s+)?(?:volatility|vol)\s+(?:is|was|stays|remains)?\s*(high|elevated|low|calm|quiet|subdued|above average|below average)|(high|elevated|low|calm|quiet)[- ](?:volatility|vol)\b")
    if m and not re.search(r.vol_re, m.group(0)):
        hi_ = (m.group(1) or m.group(2)) in ("high", "elevated", "above average")
        r.assumptions.append(f"\"{src}\": {'high' if hi_ else 'low'} volatility is taken as realised volatility in the {'top' if hi_ else 'bottom'} third of its own past year.")
        out.append(Clause(f"realised volatility is in the {'top' if hi_ else 'bottom'} third of its past year", "static", "rv_pctile", None, ">=" if hi_ else "<=", 66.67 if hi_ else 33.33, None, "open", src, 0, "the realised-volatility percentile"))
        w.take(m)
    # ----- streaks
    m = w.find(r"(\w+|\d+)\s+(?:or more\s+)?(?:consecutive\s+|straight\s+|successive\s+|back[- ]to[- ]back\s+)?(down|red|bearish|losing|lower|up|green|bullish|winning|higher)\s+(?:days?|sessions?|closes?)(?:\s+in a row)?")
    if m and _numw(m.group(1)) and _numw(m.group(1)) >= 2 and re.search(r"consecutive|straight|in a row|successive|back[- ]to[- ]back|streak", w.s):
        k = int(_numw(m.group(1)))
        feat = {"down": "down_streak", "losing": "down_streak", "lower": "down_streak", "red": "red_streak", "bearish": "red_streak", "up": "up_streak", "winning": "up_streak", "higher": "up_streak", "green": "green_streak", "bullish": "green_streak"}[m.group(2)]
        out.append(Clause(f"{k} or more sessions in a row closed {'lower' if 'down' in feat else 'higher' if 'up' in feat else 'below their open' if 'red' in feat else 'above their open'}", "day", feat, None, ">=", float(k), None, "close", src,
                          -1 if feat in ("down_streak", "red_streak") else 1, FEATURES[feat].words))
        w.take(m)
    # ----- moving averages
    m = w.find(r"(above|below|over|under|beneath)\s+(?:its|the|their)?\s*(\d+)[- ]?(?:day|session|d)\s*(?:simple\s+|exponential\s+)?(?:moving average|sma|ema|ma|average)")
    if m:
        k, up = int(m.group(2)), m.group(1) in ("above", "over")
        feat = f"sma{k}_dist" if f"sma{k}_dist" in FEATURES else None
        sj, _ = _find_subject(r, w.s)
        if sj:
            feat = None
        e = f"dist_pct(sma(1), sma({k}))" if not sj else f'dist_pct(close("{sj}", 1), sma({k}))'
        if sj:
            r.unsupported.append(f"a moving average of {sj} (only the studied symbol's own averages can be read) in \"{src}\"")
        else:
            out.append(Clause(f"the last close is {'above' if up else 'below'} its {k}-day average", "static", feat, None if feat else e, ">" if up else "<", 0.0, None, "open", src, 0, f"the last close relative to its {k}-day average"))
        w.take(m)
    # ----- distance from the 1-year high
    m = w.find(rf"{N}\s*%\s*(?:or more\s+)?(?:below|off|from|under|down from)\s+(?:its|the|their)?\s*(?:52[- ]week|one[- ]year|1[- ]year|yearly|all[- ]time)?\s*highs?") or w.find(rf"drawdown\s+(?:of\s+)?(?:more than|at least|over)?\s*{N}\s*%")
    if m:
        v = float(m.group(1))
        out.append(Clause(f"the last close is {v:g}% or more below the 1-year high", "static", "dist_high252", None, "<=", -v, None, "open", src, -1, "the drawdown from the 1-year high"))
        w.take(m)
    # ----- weekdays ("on a Monday"; "a red Monday" is left to the move reader below, with the weekday added)
    m = w.find(r"(?:on\s+)?(?:a\s+|an\s+)?\b(monday|tuesday|wednesday|thursday|friday)s?\b")
    if m:
        out.append(Clause(f"it is a {m.group(1).capitalize()}", "static", "weekday", None, "==", float(WEEKDAYS.index(m.group(1))), None, "open", src, 0, "the weekday"))
        w.take(m)
    # ----- scheduled events
    m = w.find(r"(?:on\s+)?(?:an?\s+)?\b(fomc|fed|cpi|inflation report|nfp|jobs report|payrolls|opex)\b(?:\s+(?:day|days|meeting|meetings|decision|release|announcement)s?)?")
    if m:
        ev = {"fomc": "FOMC", "fed": "FOMC", "cpi": "CPI", "inflation report": "CPI", "nfp": "NFP", "jobs report": "NFP", "payrolls": "NFP", "opex": "OPEX"}[m.group(1)]
        out.append(Clause(f"a scheduled {ev} event is on the day's calendar", "static", None, f'event_today("{ev}")', None, None, None, "open", src, 0, ""))
        w.take(m)
    # ----- volume
    m = w.find(rf"(?:(heavy|high|unusually high|elevated|low|light|thin)\s+)?(?:volume|turnover)(?:\s+(?:is|was|were))?(?:\s+(heavy|high|elevated|above average|above normal|low|light|thin|below average|below normal))?(?:\s+(?:of\s+)?(?:at least\s+|more than\s+)?{N}\s*(?:x|times)(?:\s+(?:the\s+)?(?:average|normal|usual))?)?")
    if m:
        word = (m.group(1) or m.group(2) or "heavy")
        hi_ = not re.search(r"low|light|thin|below", word)
        v = float(m.group(3)) if m.group(3) else (1.5 if hi_ else 0.7)
        if not m.group(3):
            r.assumptions.append(f"\"{src}\": {'heavy' if hi_ else 'light'} volume is taken as {v:g} times the 20-session average or {'more' if hi_ else 'less'}.")
        out.append(Clause(f"the session's volume was {v:g}x its 20-session average or {'more' if hi_ else 'less'}", "day", "vol_ratio", None, ">=" if hi_ else "<=", v, None, "close", src, 0, FEATURES["vol_ratio"].words))
        w.take(m)
    # ----- price against the session's own open ("is below its open", "trading above the open")
    m = w.find(r"(?:(?:is|was|are|were|trad\w+|stays?|remains?|holds?|still)\s+)*(above|below|over|under|beneath)\s+(?:its|the|their)\s+(?:own\s+)?(?:session(?:'s)?\s+|day'?s\s+|opening\s+)?open(?:ing)?(?:\s+price)?\b(?!\s*(?:range|gap|candle|bar))")
    if m:
        up = m.group(1) in ("above", "over")
        sj, _ = _find_subject(r, w.s)
        me = _measure(r, sj, "sofar", {}, noon)
        nm = sj or r.symbol
        out.append(Clause(f"{nm} is {'above' if up else 'below'} its session open at {me['at']}", "sofar", me.get("feature"), me.get("expr"), ">" if up else "<", 0.0, None, me["at"], src, 1 if up else -1,
                          f"{nm}'s move since the open, at {me['at']}", 0, sj))
        w.take(m)
    # ----- a move: SUBJECT x SPAN x direction (and size), read from what is left
    dm = w.find(rf"\b(?:{STRONG_DOWN})\b") or w.find(rf"\b(?:{STRONG_UP})\b")
    strong = dm is not None
    if dm is None:
        dn, up = w.find(rf"\b(?:{PLAIN_DOWN})\b"), w.find(rf"\b(?:{PLAIN_UP})\b")
        dm = dn if (dn and (not up or dn.start() < up.start())) else up
    if dm is not None:
        sign = -1 if re.fullmatch(rf"(?:{STRONG_DOWN})|(?:{PLAIN_DOWN})", dm.group(0)) else 1
        subj, named = _find_subject(r, w.s)
        span, extra, sm = _span_match(w.s)
        mm = w.find(rf"(?:by\s+|of\s+)?(?:more than\s+|at least\s+|over\s+|greater than\s+|>=?\s*)?(?:a\s+|an\s+)?{N}\s*(?:%|percent|pct)(?:\s*or more|\s*\+)?")
        size = float(mm.group(1)) if mm else None
        big = bool(w.find(rf"\b(?:{BIG})(?:ly)?\b"))
        for x in (dm, sm, mm):
            w.take(x)
        if span is None:
            span, extra = prev_span or ("day", {})
            if prev_span and named:
                r.assumptions.append(f"\"{src}\" names no time span; it is read over the same span as the condition before it.")
        me = _measure(r, subj, span, extra, noon)
        if size is None:
            if big:
                size = r.typical(*me["q"], 0.95, {"overnight": 1.0, "day": 2.0, "prev": 2.0, "last_n": 4.0}.get(span, 0.75))
                r.assumptions.append(f"\"{src}\": 'large' is taken as {size:g}% or more - about the largest 5% of such moves in {r.symbol}'s stored history. Say a number to change it.")
            elif strong:
                size = r.typical(*me["q"], 0.5, {"overnight": 0.3, "day": 0.5, "prev": 0.5, "last_n": 1.5}.get(span, 0.3))
                r.assumptions.append(f"\"{src}\": the wording implies more than a marginal move, so it is taken as {size:g}% or more (a larger-than-typical move for that span). Say a number to change it.")
        name = subj or r.symbol
        label = f"{name} {'fell' if sign < 0 else 'rose'}" + (f" {size:g}% or more" if size else "") + f" {_span_words(span, extra, me['at'])}"
        out.append(Clause(label, span, me.get("feature"), me.get("expr"), ("<=" if sign < 0 else ">=") if size else ("<" if sign < 0 else ">"), (-(size or 0.0) if sign < 0 else (size or 0.0)), None, me["at"], src, sign, me["label"], extra.get("n", 0), subj))
    # ----- anything the strategy question reader already understands (VWAP, RSI, opening range, IV rank, ...)
    spans: list = []
    low_left = w.s
    for _cat, dc in NL.find_conditions(low_left, spans):
        if any(dc.expr.replace(" ", "") == (c.expr or "").replace(" ", "") for c in out) or re.search(r"previous_session\.|gap_pct|weekday|vix [<>]|sma\(", dc.expr) and out:
            continue
        intr = X.needs_intraday(dc.expr)
        out.append(Clause(dc.text, "sofar" if intr else "static", None, dc.expr, None, None, None, noon if intr else "open", src, 0, ""))
    for a_, b_ in spans:
        w.s = w.s[:a_] + " " * (b_ - a_) + w.s[b_:]
    left = w.left(r)
    if not out:
        return None
    if left:
        r.unsupported.append(f"'{' '.join(left)}' in \"{src}\"")
    return out


def _span_words(span: str, extra: dict, at: str) -> str:
    return {"overnight": "overnight (open against the previous close)", "prev": "in the previous session", "day": "on the session", "last_n": f"over the last {extra.get('n', 5)} sessions",
            "first_n": f"in the first {extra.get('minutes', 60)} minutes", "morning": f"between the open and {at}", "sofar": f"between the open and {at}", "clock": f"between the open and {at}"}.get(span, "")


# ----------------------------------------------------------------------------- outcome
def _read_outcome(r: Reader, part: str, clauses: list[Clause], noon: str, late: str) -> tuple[dict, dict]:
    """(outcome node spec, info) from the outcome part of the question."""
    c = " " + part.lower() + " "
    info = {"entry_hint": None, "horizon_stated": False, "direction_stated": False, "sessions": None}
    cond_sign = sum(cl.sign for cl in clauses)
    direction = None
    if re.search(r"\brevers\w*|mean[- ]?revert\w*|\bfade\w*|snap\w* back|bounce back", c):
        direction = "up" if cond_sign < 0 else "down" if cond_sign > 0 else None
        if direction is None:
            r.unsupported.append("'reverse' has no direction here: no condition states which way the market moved first")
    elif re.search(r"\bcontinu\w*|follow[- ]through|momentum|keep (?:going|falling|rising)|persist\w*|extend\w*", c):
        direction = "down" if cond_sign < 0 else "up" if cond_sign > 0 else None
    if direction is None:
        up = re.search(r"\b(?:rall\w+|rise\w*|rose|go(?:es)? up|higher|recover\w*|rebound\w*|bounc\w+|gain\w*|positive|green|up|upside|advance\w*|strength|bullish|above|climb\w*|jump\w*|increas\w+)\b", c)
        dn = re.search(r"\b(?:fall\w*|fell|drop\w*|declin\w+|lower|sell[- ]?off|sells? off|negative|red|down|downside|weak\w*|lose\w*|bearish|below|sink\w*|sank|slid\w*|decreas\w+)\b", c)
        if up and dn:
            direction = "up" if up.start() < dn.start() else "down"
        elif up or dn:
            direction = "up" if up else "down"
    info["direction_stated"] = direction is not None
    direction = direction or "up"
    # where the observation starts and how long it runs
    entry, horizon = None, None
    if re.search(r"\bafternoon\b|after (?:lunch|noon|midday)|second half of the (?:day|session)", c):
        entry, horizon = noon, "session"
    elif re.search(r"last hour|final hour|power hour|into the close|late in the (?:day|session)", c):
        entry, horizon = late, "session"
    m = re.search(r"(?:within|over|in|during)\s+(?:the\s+)?(?:next|following|subsequent)?\s*(\w+|\d+)\s+(?:trading\s+)?(day|session|week|month)s?\b|(?:next|following)\s+(\w+|\d+)\s+(?:trading\s+)?(day|session|week|month)s\b|(\w+|\d+)\s+(?:trading\s+)?(day|session|week)s?\s+(?:later|after|afterwards?|on)\b", c)
    if m:
        tok = m.group(1) or m.group(3) or m.group(5)
        unit = m.group(2) or m.group(4) or m.group(6)
        k = _numw(tok)
        if k:
            info["sessions"] = int(k) * {"day": 1, "session": 1, "week": 5, "month": 21}[unit]
    if info["sessions"] is None:
        if re.search(r"\b(?:next|following)\s+(?:trading\s+)?(?:day|session)\b|\btomorrow\b|the day after|day[- ]after|overnight and next", c):
            info["sessions"] = 1
        elif re.search(r"\b(?:next|following)\s+week\b|over the week", c):
            info["sessions"] = 5
        elif re.search(r"\b(?:next|following)\s+month\b", c):
            info["sessions"] = 21
    if info["sessions"] is None and entry is None and re.search(r"by the close|rest of the (?:day|session)|that (?:same )?day|same day|same session|intraday|during the (?:day|session)|into the close|close (?:higher|lower|green|red|up|down)", c):
        horizon = "session"
    # a move measured from the open of the session ("rise 1% from the open", "gain 1% intraday"), or over the whole day against the previous close
    if entry is None and re.search(r"\b(?:from|off|above|below|against|versus|vs\.?|relative to)\s+(?:the|its|that|that day'?s|the (?:next|following) day'?s|the day'?s|the session'?s)?\s*open(?:ing)?(?:\s+price)?\b|\bintraday\b|"
                                  r"\bat any (?:point|time|moment)\b|\bduring the (?:next |following |same )?(?:trading )?(?:day|session)\b|\bopen[- ]to[- ]close\b", c):
        entry, horizon = "open", "session"
        if info["sessions"] == 1:
            info["sessions"] = None  # "intraday the next session": the session after the condition is the one that opens, not a second one
    day_move = bool(re.search(r"\b(?:finish\w*|end\w*|clos\w*)\s+the\s+(?:day|session)\b|\bon the (?:day|session)\b|\b(?:up|down|green|red|positive|negative)\s+(?:day|session)\b|\bday(?:'s)? (?:return|change|move)\b", c)) and entry is None and not info["sessions"]
    info["day_move"] = day_move
    info["entry_hint"], info["horizon_stated"] = entry, bool(horizon or info["sessions"] or day_move)
    closes = bool(re.search(r"\b(?:clos\w*|finish\w*|end(?:s|ed|ing)?|settl\w*)\b", c))
    # size: "1% before 0.5%", "reach 1%", "close 1% higher", "rise 1%"
    m = re.search(rf"{N}\s*%\s*(?:target|gain|rally|rise|move|profit|drop|decline|fall)?\s*(?:\w+\s+){{0,3}}?before\s+(?:a\s+|an\s+)?{N}\s*%", c)
    if m:
        return {"op": "target_stop", "params": {"direction": direction, "target": float(m.group(1)), "stop": float(m.group(2))}, "_entry": entry, "_horizon": horizon}, info
    m = re.search(rf"(?:reach\w*|hit\w*|touch\w*|trade\w*)\s+(?:a\s+)?(?:\w+\s+){{0,2}}?(?:\+|-)?{N}\s*%", c)
    if m:
        return {"op": "target_stop", "params": {"direction": direction, "target": float(m.group(1))}, "_entry": entry, "_horizon": horizon}, info
    c = re.sub(rf"{N}\s*(?:percent|pct)\b", r"\1%", c)
    m = re.search(rf"(?:by|of)?\s*(?:more than|at least|over|≥|>=?)\s*{N}\s*%|{N}\s*%\s*(?:or more\s+)?(?:higher|lower|gain|rally|rise|drop|decline|up|down|above|below|move|increase|decrease)|"
                  rf"\b(?:ris\w+|rose|gain\w*|rall\w+|climb\w*|jump\w*|advanc\w+|increas\w+|recover\w*|bounc\w+|rebound\w*|fall\w*|fell|drop\w*|declin\w+|los\w+|sink\w*|sank|slid\w*|up|down|higher|lower)\s+(?:by\s+|of\s+|another\s+|a\s+further\s+)?(?:about\s+|roughly\s+)?{N}\s*%", c)
    thr = float(m.group(1) or m.group(2) or m.group(3)) if m else 0.0
    info["size"] = thr or None
    state = bool(info["sessions"]) and bool(re.search(r"\b(?:is|are|was|were|be|stands?|sits?|trad(?:es|ing))\s+(?:still\s+)?(?:up|down|higher|lower|above|below)\b", c))
    if thr and not closes and not day_move and not state:
        # "rise 1%", "gain 1% intraday", "drop 2% within a week": the level has to be TRADED at some point before the horizon ends, not closed at
        info["touch"] = True
        r.assumptions.append(f"'{m.group(0).strip()}' is read as trading {thr:g}% {'above' if direction == 'up' else 'below'} the starting price at any time before the horizon ends, not as closing there. Say 'close {thr:g}% higher' for the other reading.")
        return {"op": "target_stop", "params": {"direction": direction, "target": thr}, "_entry": entry, "_horizon": horizon}, info
    return {"op": "forward_return", "params": {"direction": direction, **({"threshold": thr} if thr else {})}, "_entry": entry, "_horizon": horizon}, info


# ----------------------------------------------------------------------------- the reader
def _split(text: str) -> tuple[str, str]:
    """(outcome part, condition part)."""
    low = text
    m = INTRO.search(low)
    lead = LEADS.search(low)
    if lead and (not m or lead.start() < m.start()):
        return low[lead.end():], low[:lead.start()]
    if m:
        head, tail = low[:m.start()], low[m.end():]
        if not re.search(r"[a-z]{3}", head):
            # the sentence OPENS with the condition ("after a 3% fall, how often does it ...?"): what follows the comma and asks the question is the outcome
            q = QUESTION.search(tail)
            if q:
                return tail[q.start():], tail[:q.start()]
        return head, tail
    return low, ""


def _clauses_of(part: str) -> list[str]:
    part = re.sub(rf"between\s+{N}\s*%?\s+and\s+{N}", r"between \1 to \2", part)
    bits = re.split(r"\s*;\s*|\s*,\s*(?:and\s+|but\s+)?|\s+and\s+(?:also\s+)?|\s+but\s+|\s+while\s+|\s+as well as\s+|\s+(?:when|whenever|if|after|following|once|given that|provided that)\s+", part)
    return [b.strip(" .?!") for b in bits if b and b.strip(" .?!")]


def parse(text: str, known: list[str], default_symbol: str = "SPY", store=None) -> dict | None:
    """{"core", "analyses", "sweeps", "assumptions", "unsupported", "symbols"} or None when the
    text is not a market question this reader can turn into a graph."""
    r = Reader(text, known, default_symbol, store)
    low = r.low
    # ----- modifiers are lifted out first
    controls_txt = None
    m = CONTROL.search(low)
    if m:
        controls_txt = m.group(1)
        low = low[:m.start()] + " " + low[m.end():]
    others: list[str] = []
    start, end, _lbl = NL.find_period(low)
    m = COMPARE.search(low)
    if m:
        others = [t.upper() for t in re.findall(r"\b[A-Za-z]{2,5}\b", m.group(1)) if t.upper() in set(r.known) | set(NL.TICKERS)]
        low = low[:m.start()] + " " + low[m.end():]
    if start or end:
        low = re.sub(rf"(?:since|after|before|from|between|until|through|during|in)\s+{NL.DATE_TOK}(?:\s*(?:to|and|through|until|-)\s*{NL.DATE_TOK})?", " ", low)
        low = re.sub(r"(?:since|after|before|from|between|until|through|during|in)\s+(?:19|20)\d{2}(?:\s*(?:to|and|through|until|-)\s*(?:19|20)\d{2})?", " ", low)
        low = re.sub(r"(?:in|over|during)?\s*the\s+(?:last|past)\s+\w+\s+(?:years?|months?)", " ", low)
    out_part, cond_part = _split(low)
    noon, late = r.tick(["12:00", "12:30"]), r.tick(["15:00", "14:30", "15:30"])
    # ----- conditions
    clauses: list[Clause] = []
    prev_span = None
    for bit in _clauses_of(cond_part):
        got = _read_clause(r, bit, prev_span, noon)
        if got is None:
            r.unsupported.append(bit)
            continue
        clauses += got
        if got[0].span not in ("static",):
            prev_span = (got[0].span, {"n": got[0].n} if got[0].span == "last_n" else {})
    if not cond_part.strip():
        # no introducer: look for a condition inside the whole sentence ("how often does SPY close higher on Mondays")
        for bit in _clauses_of(out_part):
            got = _read_clause(r, bit, None, noon) if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|fomc|cpi|nfp|vix|volatility index)\b", bit) else None
            if got:
                clauses += [g for g in got if g.span == "static"]
    oc, info = _read_outcome(r, out_part, clauses, noon, late)
    if not clauses and not info["direction_stated"]:
        return None
    if not clauses:
        return None  # an outcome with nothing to condition on is a plain frequency: the study path answers that already
    # ----- resolve the decision instant: the latest moment any condition is known
    intraday = [c for c in clauses if c.at not in ("open", "close")]
    own_day = [c for c in clauses if c.span == "day"]
    entry = oc.pop("_entry")
    horizon = oc.pop("_horizon")
    if entry is None and info.get("day_move") and not intraday and not [c for c in clauses if c.span == "overnight"]:
        # "finish the day up more than 1%": the whole session against the previous close, so the decision instant is that close
        entry, info["sessions"] = "close", 1
        r.assumptions.append("A move 'on the day' is the whole session against the previous close, so the question is decided at that close and every condition is read as it stood then.")
        for c in clauses:
            if c.span == "day":
                c.label = c.label.replace(" on the session", " in the previous session")
    if entry is None:
        if intraday:
            entry = max((c.at for c in intraday), key=G.tick_order)
        elif own_day and not [c for c in clauses if c.span == "overnight"]:
            entry = "close"
        else:
            entry = "open"
    if entry != "close":
        for c in own_day:  # read as of the open: "after a down day" then refers to the session that finished before
            c.at = "open"
            c.label = c.label.replace(" on the session", " in the previous session")
            c.span = "prev"
    for c in clauses:
        if c.span in ("morning", "sofar") and c.at != entry and G.tick_order(entry) > 570 and entry != "close":
            c.at = entry
            c.label = re.sub(r"between the open and \d\d:\d\d", f"between the open and {entry}", c.label)
            c.measure = re.sub(r"at \d\d:\d\d", f"at {entry}", c.measure)
    sweeps = []
    if entry == "close":
        for c in clauses:  # at a close decision, values about completed sessions include the session that just closed
            if c.feature in FEATURES and FEATURES[c.feature].kind == "asof":
                c.at = "close"
        k = info["sessions"] or 1
        horizon = f"sessions:{k}"
        if not info["sessions"]:
            r.assumptions.append("No horizon was stated. The outcome is measured from that session's close to the close 1 session later, and 3 and 5 sessions are evaluated alongside it (all three are counted as trials).")
            sweeps.append({"target": "outcome", "param": "horizon", "values": ["sessions:1", "sessions:3", "sessions:5"]})
    elif info["sessions"]:
        horizon = f"sessions:{info['sessions']}"
    else:
        if not horizon:
            r.assumptions.append(f"No horizon was stated. The outcome is measured from {'the open' if entry == 'open' else entry + ' New York time'} to that session's close.")
        horizon = "session"
    oc["params"].update(entry=entry, horizon=horizon)
    # every size the question states must have landed somewhere: in the outcome, or in a condition. One that did not is reported, never dropped
    used = {float(v) for v in (oc["params"].get("target"), oc["params"].get("stop"), oc["params"].get("threshold")) if v is not None}
    out_part, cond_part = (re.sub(rf"{N}\s*(?:percent|pct)\b", r"\1%", x) for x in (out_part, cond_part))
    for tok in re.findall(rf"{N}\s*%", out_part):
        if float(tok) not in used and not any(re.search(rf"(?<![\d.]){re.escape(tok)}\s*%", c.source or "") for c in clauses):
            r.unsupported.append(f"the size '{tok}%' in \"{out_part.strip(' ?.')}\"")
    for tok in re.findall(rf"{N}\s*%", cond_part):
        if not any(re.search(rf"(?<![\d.]){re.escape(tok)}\s*%", c.source or "") for c in clauses) and float(tok) not in used and not any(tok in u for u in r.unsupported):
            r.unsupported.append(f"the size '{tok}%' in \"{cond_part.strip(' ?.')}\"")
    if not info["direction_stated"]:
        r.assumptions.append("No direction was stated for the outcome; 'finishes higher' is measured.")
    if entry not in ("open", "close"):
        ft = r.ft()
        if ft is not None:
            k = int(ft._tick(entry)[1].sum())
            r.assumptions.append(f"The decision is made at {entry} New York time, the nearest clock time the stored bars can show on the most sessions ({k:,} of {ft.n:,} stored sessions have a bar boundary there).")
    core = Core(r.symbol, clauses, oc, None, [], start.isoformat() if start else None, end.isoformat() if end else None)
    # ----- controls
    if controls_txt:
        used = {c.feature for c in clauses if c.feature}
        for key, feats in sorted(DEFAULT_CONTROLS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(key)}\b", controls_txt):
                for f in feats:
                    if f in used:
                        r.assumptions.append(f"{FEATURES[f].label} defines the condition, so it is not used as a control.")
                    elif f not in core.controls:
                        core.controls.append(f)
                controls_txt = re.sub(rf"\b{re.escape(key)}\b", " ", controls_txt)
        left = re.sub(r"\b(?:and|the|market|level|levels|of|prior|previous|recent|its)\b|[,.]", " ", controls_txt).strip()
        if left and not core.controls:
            r.unsupported.append(f"controlling for '{left}'")
        elif left:
            r.unsupported.append(f"the control '{left}'")
    analyses = ["probability"]
    if core.controls:
        analyses.append("matched")
    full = r.low
    if re.search(r"distribution|price paths?|what happens after|how (?:does|do) (?:it|they|price) behave|paths? (?:after|following)", full):
        analyses.append("distribution")
    if re.search(r"which (?:historical |market |observable )?(?:conditions?|factors?|variables?)|what (?:conditions?|factors?|predicts?)|associated with|what else matters", full):
        analyses.append("scan")
    analyses.append("robustness")
    if HYPOTHESIS.search(full):  # an explicit hypothesis: only then may the final test period be scored (it can be used once)
        analyses.append("validation")
    if others:
        sweeps.append({"target": "data", "param": "symbol", "values": list(dict.fromkeys([r.symbol] + others))})
    return {"core": core, "analyses": list(dict.fromkeys(analyses)), "sweeps": sweeps, "assumptions": r.assumptions, "unsupported": r.unsupported, "symbol": r.symbol}


# ----------------------------------------------------------------------------- building the graph
SCAN = ["rv_pctile", "vix_prev", "sma50_dist", "sma200_dist", "dist_high252", "ret_5", "ret_20", "vol_ratio", "range_1", "gap", "down_streak", "weekday"]


def build(core: Core, analyses: list[str], sweeps: list[dict] | None = None, question: str = "", assumptions: list[str] | None = None, unsupported: list[str] | None = None) -> dict:
    """The graph for a parsed core. Deterministic: the same core always gives the same graph."""
    nodes: list[dict] = [{"id": "data", "op": "data", "params": {"symbol": core.symbol, **({"start": core.start} if core.start else {}), **({"end": core.end} if core.end else {})}}]
    masks = []
    entry = core.outcome["params"].get("entry", "open")
    for i, c in enumerate(core.clauses, start=1):
        if c.op is None:  # a boolean rule
            nodes.append({"id": f"c{i}", "op": "filter", "inputs": {"data": "data"}, "params": {"expr": c.expr, "at": c.at, "label": c.label}})
        else:
            fp = {"at": c.at, **({"name": c.feature} if c.feature else {"expr": c.expr}), **({"label": c.measure} if c.measure and not c.feature else {})}
            nodes.append({"id": f"f{i}", "op": "feature", "inputs": {"data": "data"}, "params": fp})
            nodes.append({"id": f"c{i}", "op": "filter", "inputs": {"x": f"f{i}"}, "params": {"op": c.op, "value": c.value, **({"hi": c.hi} if c.hi is not None else {}), "label": c.label}})
        masks.append(f"c{i}")
    cond = masks[0] if len(masks) == 1 else "when"
    if len(masks) > 1:
        nodes.append({"id": "when", "op": "combine", "inputs": {"masks": masks}, "params": {"how": "and"}})
    op = dict(core.outcome)
    nodes.append({"id": "outcome", "op": op["op"], "inputs": {"data": "data"}, "params": dict(op["params"])})
    used = {c.feature for c in core.clauses if c.feature}
    if "probability" in analyses or "scan" in analyses:
        by = []
        if "scan" in analyses:
            tick = "close" if entry == "close" else "open"
            for f in SCAN:
                if f in used or (f == "gap" and entry == "close"):
                    continue
                nodes.append({"id": f"s_{f}", "op": "feature", "inputs": {"data": "data"}, "params": {"name": f, "at": tick if FEATURES[f].kind == "asof" else "open"}})
                by.append(f"s_{f}")
        nodes.append({"id": "probability", "op": "conditional_probability", "inputs": {"outcome": "outcome", "condition": cond, **({"by": by} if by else {})}})
        if "scan" in analyses and by:
            nodes.append({"id": "drivers", "op": "regression", "inputs": {"y": "outcome", "xs": by, "condition": cond}, "params": {"target": "hit"}})
    if "matched" in analyses and core.controls:
        ids = []
        for f in core.controls:
            if f in used:
                continue
            tick = "close" if (entry == "close" and FEATURES[f].kind == "asof") else "open"
            nid = f"k_{f}"
            if not any(n["id"] == nid for n in nodes):
                nodes.append({"id": nid, "op": "feature", "inputs": {"data": "data"}, "params": {"name": f, "at": tick}})
            ids.append(nid)
        if ids:
            nodes.append({"id": "matched", "op": "matched_comparison", "inputs": {"outcome": "outcome", "condition": cond, "controls": ids}})
    if "distribution" in analyses:
        nodes.append({"id": "distribution", "op": "distribution", "inputs": {"outcome": "outcome", "condition": cond}})
    if "robustness" in analyses:
        nodes.append({"id": "robustness", "op": "robustness", "inputs": {"outcome": "outcome", "condition": cond}})
    if "validation" in analyses:
        nodes.append({"id": "validation", "op": "validation", "inputs": {"outcome": "outcome", "condition": cond}})
    if "strategies" in analyses:
        nodes.append({"id": "strategies", "op": "options_backtest", "inputs": {"outcome": "outcome", "condition": cond}})
    main = "matched" if ("matched" in analyses and any(n["id"] == "matched" for n in nodes) and "probability" not in analyses) else "probability" if any(n["id"] == "probability" for n in nodes) else None
    for j, sw in enumerate(sweeps or [], start=1):
        if main is None:
            break
        tgt = sw["target"]
        if tgt.startswith("clause:"):
            tgt = f"c{int(tgt.split(':')[1])}"
        nodes.append({"id": f"sweep{j}" if len(sweeps or []) > 1 else "sweep", "op": "sweep", "inputs": {"result": "matched" if (sw.get("on") == "matched" and any(n["id"] == "matched" for n in nodes)) else main},
                      "params": {"node": tgt, "param": sw["param"], "values": sw["values"]}})
    return {"version": G.VERSION, "question": question, "nodes": nodes, "assumptions": list(assumptions or []), "unsupported": list(unsupported or []), "core": dump_core(core), "analyses": list(analyses), "sweeps": list(sweeps or [])}


def dump_core(c: Core) -> dict:
    return {"symbol": c.symbol, "start": c.start, "end": c.end, "controls": list(c.controls), "outcome": copy.deepcopy(c.outcome), "anchor": c.anchor,
            "clauses": [{k: getattr(x, k) for k in ("label", "span", "feature", "expr", "op", "value", "hi", "at", "source", "sign", "measure", "n", "subject", "other")} for x in c.clauses]}


def load_core(d: dict) -> Core:
    return Core(d["symbol"], [Clause(**x) for x in d.get("clauses", [])], copy.deepcopy(d.get("outcome") or {}), d.get("anchor"), list(d.get("controls") or []), d.get("start"), d.get("end"))


def plan(text: str, known: list[str], default_symbol: str = "SPY", store=None) -> dict | None:
    """Question -> validated graph (or None). The returned graph carries its own assumptions
    and the words that were not understood."""
    p = parse(text, known, default_symbol, store)
    if p is None:
        return None
    g = build(p["core"], p["analyses"], p["sweeps"], text.strip(), p["assumptions"], p["unsupported"])
    chk = G.validate(g)
    if chk["errors"]:
        g["errors"] = chk["errors"]
    return g


# ----------------------------------------------------------------------------- follow-ups on an existing graph
def retarget(graph: dict, symbol: str) -> dict:
    """The same idea on another symbol. A condition that named the old symbol as its
    subject follows the new one; conditions about other named symbols are kept."""
    core = load_core(graph["core"])
    old = core.symbol
    for c in core.clauses:
        for attr in ("label", "measure"):
            setattr(c, attr, re.sub(rf"\b{re.escape(old)}\b", symbol, getattr(c, attr) or ""))
        if c.expr and symbol in X.symbols_used(c.expr) | {c.subject, c.other}:
            raise ValueError(f"The idea compares {old} with {symbol}; applied to {symbol} it would compare {symbol} with itself.")
    core.symbol = symbol
    return build(core, graph.get("analyses") or ["probability"], [s for s in graph.get("sweeps") or [] if s.get("param") != "symbol"], f"{graph.get('question', '')} (applied to {symbol})",
                 [a for a in graph.get("assumptions") or [] if "stored history" not in a] + [f"Thresholds are carried over from {old} unchanged; they were not re-derived for {symbol}."], graph.get("unsupported"))


def with_analyses(graph: dict, analyses: list[str], sweeps: list[dict] | None = None, question: str | None = None, start: str | None = None, end: str | None = None, controls: list[str] | None = None) -> dict:
    core = load_core(graph["core"])
    if start is not None or end is not None:
        core.start, core.end = start, end
    if controls is not None:
        core.controls = controls
    return build(core, analyses, sweeps if sweeps is not None else [], question or graph.get("question", ""), graph.get("assumptions"), graph.get("unsupported"))


def combine(a: dict, a_idx: list[int], b: dict, b_idx: list[int], question: str = "") -> dict:
    """Clauses `a_idx` of one stored graph with clauses `b_idx` of another, on graph a's
    symbol and outcome. Clause ticks are kept; the validator refuses the result if a clause
    would be read after the outcome starts."""
    ca, cb = load_core(a["core"]), load_core(b["core"])
    if ca.symbol != cb.symbol:
        raise ValueError(f"The two studies are about different symbols ({ca.symbol} and {cb.symbol}).")
    core = Core(ca.symbol, [ca.clauses[i] for i in a_idx] + [cb.clauses[i] for i in b_idx], copy.deepcopy(ca.outcome), None, [], ca.start, ca.end)
    entry = core.outcome["params"].get("entry", "open")
    late = [c for c in core.clauses if G.tick_order(c.at) > G.tick_order(entry)]
    if late:
        entry = max((c.at for c in core.clauses), key=G.tick_order)
        core.outcome["params"]["entry"] = entry
        if entry == "close" and not str(core.outcome["params"].get("horizon", "")).startswith("sessions:"):
            core.outcome["params"]["horizon"] = "sessions:1"
    return build(core, ["probability", "robustness"], [], question, [f"The outcome and symbol come from the first study; the decision is made at {entry}, the latest moment any of the combined conditions is known."], [])


def find_clauses(graph: dict, words: str) -> list[int]:
    """Indices of the clauses of a stored graph that a phrase refers to ("the opening-range
    condition", "the volatility filter")."""
    w = words.lower()
    keys = {"volatil": ["volatil", "vix", "rvol", "rv_"], "vix": ["vix", "volatility index"], "opening range": ["opening_range", "opening range", "first", "or15", "or30", "or60"], "opening-range": ["opening_range", "opening range", "or15", "or30", "or60"],
            "gap": ["gap"], "volume": ["volume", "vol_ratio"], "streak": ["streak", "in a row", "consecutive"], "trend": ["sma", "average"], "moving average": ["sma", "average"], "weekday": ["weekday", "monday", "tuesday", "wednesday", "thursday", "friday"],
            "morning": ["from_open", "since the open"], "down day": ["ret_1", "fell"], "decline": ["ret_1", "fell", "from_open"], "drawdown": ["dist_high", "below the 1-year"], "vwap": ["vwap"], "rsi": ["rsi"], "outperform": ["outperform", "minus"]}
    want = [v for k, v in keys.items() if k in w]
    out = []
    for i, c in enumerate(graph.get("core", {}).get("clauses", [])):
        blob = " ".join(str(c.get(k) or "") for k in ("label", "feature", "expr", "source", "measure")).lower()
        if any(any(t in blob for t in ts) for ts in want):
            out.append(i)
    return out


# ----------------------------------------------------------------------------- the same reader, for a strategy's entry rule
def _clause_rule(c: Clause, symbol: str, clock: str | None) -> tuple[str, str] | None:
    """(rule-language expression, wording) for one condition in an entry rule, or None when it
    cannot be evaluated at the entry instant."""
    if c.span == "day":
        return None  # about the session's own close
    if c.span in ("morning", "first_n") and c.at != clock:
        return None  # a stretch of the session that does not end at the entry
    try:
        base = c.expr or feature_def(c.feature, symbol).expr
    except (KeyError, ValueError):
        return None
    if c.op is None:
        expr = base
    elif c.op == "between" and c.hi is not None:
        expr = f"({base}) >= {c.value:g} and ({base}) <= {c.hi:g}"
    elif c.value is None:
        return None
    elif c.op == "abs>=":
        expr = f"abs({base}) >= {c.value:g}"
    else:
        expr = f"{base} {c.op} {c.value:g}" if re.fullmatch(r"[\w.]+(?:\([^()]*\))?", base) else f"({base}) {c.op} {c.value:g}"
    expr = re.sub(r"([<>]=?) -0\b(?!\.)", r"\1 0", expr)
    if X.validate(expr):
        return None
    label = c.label
    if c.span == "sofar" and c.at != clock:  # read whenever the entry happens, not at the reader's default clock time
        label = re.sub(r"between the open and \d\d:\d\d", "between the open and the entry time", re.sub(r" at \d\d:\d\d$", " at the entry time", label))
    return expr, label


def entry_conditions(phrase: str, symbol: str, known: list[str], store=None, entry: str | None = None, partial: bool = False):
    """A condition phrase written as rule-language expressions for a STRATEGY entry rule: [{"expr", "text", "assumptions"}].

    A condition is accepted only when every word of its sub-phrase was understood AND it can be evaluated at the
    entry instant. A statement about the session's own close, or about a stretch of the session other than "so
    far" (the first 30 minutes, the morning) when the entry is not exactly at its end, is NOT turned into a rule:
    the engine would evaluate it at the entry tick, which is a different question.

    Returns the list, or None unless the whole phrase was understood. With `partial=True` it returns
    (conditions, sub-phrases that were not understood) so the caller can keep what was read and report the rest."""
    lead = re.match(r"^\s*(?:and\s+|but\s+)?(?:only\s+)?(?:on\s+(?:days?|sessions?|mornings?)\s+)?(if|when|whenever|while|provided(?:\s+that)?|as long as|after|following|once|that|where|on which)\s+", phrase.strip(" ,.;"), flags=re.I)
    text = phrase.strip(" ,.;")[lead.end():] if lead else phrase.strip(" ,.;")
    if not text:
        return ([], []) if partial else None
    r = Reader(text, known, symbol, store)
    r.symbol = symbol.upper()  # the strategy's own underlying: another symbol named in the phrase is a subject, not the instrument
    clock = entry if entry and re.fullmatch(r"\d\d:\d\d", entry) else None
    intraday_entry = entry != "close"
    # "after SPY closes down 3%", "on days after a 2% rally": the session spoken of is finished before this entry. And at the open
    # nothing has happened yet today, so a move with no reference can only be the previous session's.
    follows = bool(lead and lead.group(1).lower() in ("after", "following", "once")) or entry in ("open", "09:30")
    default = ("prev", {}) if follows else (("sofar", {}) if intraday_entry else None)
    out, left = [], []
    prev_span = None
    for bit in _clauses_of(text):
        r.unsupported = []
        n0 = len(r.assumptions)
        spanless = _span_match(" " + bit.lower() + " ")[0] is None
        # a present-tense state with no reference ("QQQ is weak") in an intraday entry rule is read as "so far today"
        cs = _read_clause(r, bit, prev_span or default, clock or "12:00")
        if cs and follows:
            for c in cs:
                if c.span == "day":  # the completed session the entry follows
                    c.span, c.at = "prev", "open"
                    c.label = c.label.replace(" on the session", " in the previous session")
        rules = [_clause_rule(c, r.symbol, clock) for c in cs] if cs and not r.unsupported else [None]
        if any(x is None for x in rules):
            left.append(bit)
            del r.assumptions[n0:]
            continue
        notes = list(r.assumptions[n0:])
        if spanless and prev_span is None and any(c.span == "sofar" and c.subject for c in cs):
            notes.append(f"\"{bit}\" names no reference point; it is measured from that session's open, at the entry time. Say \"on the day\" or \"from the previous close\" to mean something else.")
        out += [{"expr": e, "text": t, "assumptions": notes} for e, t in rules]
        if cs[0].span not in ("static", "day"):
            prev_span = (cs[0].span, {"n": cs[0].n} if cs[0].span == "last_n" else {})
    if partial:
        return out, left
    return out if out and not left else None


# ----------------------------------------------------------------------------- reference text for a language-model planner (its graph goes through the same validator)
def llm_instructions(symbol: str) -> str:
    prims = "\n".join(f"- {p.op}: inputs {p.inputs or '{}'} -> {p.output}; params {{{', '.join(k + ('*' if v['required'] else '') for k, v in p.params.items())}}}. {p.semantics}" for p in G.PRIMITIVES.values())
    feats = ", ".join(f"{f.name} ({f.label.lower()})" for f in FEATURES.values())
    names = ", ".join(sorted(k for k, v in X.REGISTRY.items() if v["category"] not in ("Position",)))
    return ('A research graph is {"nodes": [{"id": str, "op": str, "inputs": {port: node id or [node ids]}, "params": {...}}], "assumptions": [str], "unsupported": [str]}.\n'
            "It is structure only: no code, and no result is ever computed or stated by you. Only these primitives exist (a * marks a required parameter):\n" + prims + "\n"
            "Catalog features for the `feature` primitive's `name`: " + feats + ". Cross-symbol catalog features: <SYMBOL>_ret_1, <SYMBOL>_gap, rs5_<SYMBOL>.\n"
            "Anything else is an `expr` in the strategy rule language, read at a tick. Names available: " + names + ". Another symbol's value is written TICKER.field (QQQ.from_open_pct, VIX.gap_pct).\n"
            "Ticks are open, close or HH:MM New York. A condition must be known no later than the outcome's entry: to condition on the morning, start the outcome at that clock time. "
            "An observation that starts at the close needs horizon sessions:N.\n"
            "Put every interpretation you had to choose in `assumptions`, and every part of the question you could not represent in `unsupported` (do not approximate it).\n"
            f"The symbol being studied is {symbol} unless the question names another.")
