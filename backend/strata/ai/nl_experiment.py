"""One sentence, a bounded experiment: a base strategy and the settings to vary.

    Compare entry times from 9:30 to 11:00 in 15-minute increments, prior-day declines from 0.5% to 3%,
    0DTE versus 1DTE, and several profit targets.

is not one strategy and it is not a vague wish. It is a grid: four settings, each over a stated (or, where the sentence
leaves it open, an explicitly assumed) list of values. This reader finds the dimensions, takes them out of the sentence and
hands what is left to the strategy readers as the base. Where the sentence names no strategy, the base is the strategy the
conversation is already about; where there is none either, it asks instead of inventing one.

The result is a plan, not a run:

    {"action": "experiment", "draft": <base Draft>, "dims": [{"field", "values", "text", "assumed"}], "configs": N, ...}

Every value that was not in the sentence is marked `assumed` and listed in `notes`, so that "several profit targets" never
turns into numbers the user did not see. The grid is bounded: more than MAX_CONFIGS combinations is refused with the count,
never trimmed silently.
"""
from __future__ import annotations

import re

from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, DraftCondition, DraftStrike

MAX_CONFIGS = 300
FIELDS = {"symbol": "underlying", "dte": "days to expiration", "entry_time": "entry time (New York)", "exit_minutes": "minutes held after the entry", "strike": "strike", "premium_target": "profit target, % of premium",
          "premium_stop": "stop, % of premium", "underlying_target": "target, % move of the underlying", "prev_decline": "previous-day decline, %"}
NUMS = r"(\d+(?:\.\d+)?)"
SEP = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|versus|vs\.?|against|&)\s+)"  # "15, 30, 60, and 120": a comma, a word, or both
LIST = rf"(\d+(?:\.\d+)?\s*%?(?:{SEP}\d+(?:\.\d+)?\s*%?)+)"
STEP_WORDS = {"half-percent": 0.5, "half percent": 0.5, "half a percent": 0.5, "quarter-percent": 0.25, "quarter percent": 0.25, "one-percent": 1.0, "one percent": 1.0, "whole-percent": 1.0}
CUE = re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b|\bsweep\b|\bside by side\b|\btest\b|\btry\b|\bacross\b|\beach with\b|\bincrements?\b|\bsteps?\b|\bevery \d+ minutes\b|\bagainst\b|\bfor each\b|\bmultiple\b|\bseveral\b|\bdifferent\b|\bvary\b|\bgrid\b", re.I)


def _floats(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]


def _rng(a: float, b: float, step: float, cap: int = 25) -> list[float]:
    out, x = [], a
    while x <= b + 1e-9 and len(out) < cap:
        out.append(round(x, 4))
        x += step
    return out


def _step(low: str, span_end: int, default: float) -> tuple[float, bool, int]:
    """A step that follows a range ("in half-percent steps", "in 0.25% increments"). Returns (step, stated, end index)."""
    tail = low[span_end:span_end + 48]
    m = re.match(r"\s*,?\s*(?:in|by|with|at)\s+(half[- ]percent|half a percent|quarter[- ]percent|one[- ]percent|whole[- ]percent|\d+(?:\.\d+)?\s*%?)[- ]?\s*(?:increments?|steps?|intervals?)", tail)
    if not m:
        m = re.match(r"\s*,?\s*(?:in\s+)?(?:increments?|steps?)\s+of\s+(half a percent|\d+(?:\.\d+)?\s*%?)", tail)
    if not m:
        return default, False, span_end
    tok = m.group(1).strip()
    return (STEP_WORDS.get(tok.replace("  ", " ")) or float(re.sub(r"[^\d.]", "", tok))), True, span_end + m.end()


def _clock(tok: str) -> str | None:
    """'9:30', '11', '10 am', 'noon', 'the open' -> 'HH:MM' New York (or 'open'). Times in an experiment are read as New York
    session times: a bare hour between 1 and 4 is the afternoon."""
    t = tok.strip().lower()
    if t in ("the open", "open", "the opening bell", "the bell"):
        return "open"
    if t in ("noon", "midday"):
        return "12:00"
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", t)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if ap == "pm" and h < 12:
        h += 12
    elif not ap and 1 <= h <= 4:
        h += 12
    if not (9 <= h <= 16) or (h == 9 and mi < 30) or (h == 16 and mi > 0):
        return None
    return "open" if (h, mi) == (9, 30) else f"{h:02d}:{mi:02d}"


TIME = r"(\d{1,2}(?::\d{2})?(?:\s*(?:a\.?m\.?|p\.?m\.?)\b)?|noon|midday|the open)"  # trailing space is only taken with am/pm: a list must be able to go on


class Found:
    def __init__(self, text: str):
        self.text, self.low = text, text.lower().replace("’", "'")
        self.dims: list[dict] = []
        self.spans: list[tuple[int, int]] = []
        self.notes: list[str] = []

    def add(self, field: str, values: list, span: tuple[int, int], assumed: bool = False, note: str = ""):
        if any(d["field"] == field for d in self.dims) or len(values) < 2:
            return
        vals = list(dict.fromkeys(tuple(v) if isinstance(v, list) else v for v in values))
        self.dims.append({"field": field, "values": vals, "text": self.text[span[0]:span[1]].strip(" ,;."), "assumed": bool(assumed)})
        self.spans.append(span)
        if note:
            self.notes.append(note)

    def masked(self) -> str:
        """The sentence without the dimensions: what is left describes the base strategy (or nothing)."""
        out = list(self.text)
        for a, b in self.spans:
            for i in range(a, min(b, len(out))):
                out[i] = " "
        s = "".join(out)
        s = re.sub(r"\b(?:compare|sweep|test|try|vary)\b\s*(?:it\s+)?(?:with|over|across|on|for|against)?\s*(?=[,;:.]|$|\band\b)", " ", s, flags=re.I)
        s = re.sub(r"\s*[,;:]\s*(?:and\s+)?(?=[,;:.]|$)", " ", s)
        s = re.sub(r"\b(?:and|with|each with|against|versus|vs\.?|for|on|over|across)\s*(?=[,;:.]|$)", " ", s, flags=re.I)
        return re.sub(r"\s{2,}", " ", s).strip(" ,;:.")


def sentence_zone(text: str) -> tuple[int | None, str | None]:
    """(minutes behind New York, name) of the ONE zone the sentence's clock times are read in: the zone it names, or the one
    the rule reader assumes for times written without one. Every part of an experiment is read in the same zone."""
    named = {NL.TZ_OFFSET[k] for k in NL.TZ_OFFSET if re.search(rf"\b{k}\b", text.lower()) and k not in ("et", "pt", "mt", "ct")} | {NL.TZ_OFFSET[m.group(1).lower()] for m in re.finditer(r"\b(ET|PT|MT|CT|EST|PST|EDT|PDT)\b", text)}
    if len(named) == 1:
        off = named.pop()
        return off, NL.TZ_NAME.get(off, "Eastern")
    ts = [t for t in NL.find_times(text, None) if t.get("assumed_tz")]
    if ts:
        name = ts[0]["assumed_tz"]
        return {v: k for k, v in NL.TZ_NAME.items()}.get(name, 0), name
    return None, None


def clocks(fragment: str, zone: int | None) -> list[str]:
    """The clock times of a fragment in New York time ('open' for 09:30), read in the sentence's zone."""
    out = []
    for t in NL.find_times(fragment, zone):
        if 570 <= t["minutes"] <= 960:
            out.append("open" if t["minutes"] == 570 else "close" if t["minutes"] == 960 else t["et"])
    if len(out) < 2:  # bare hours ("10 and 11") are not clock times to the general reader; in a list of entry times they are
        alt = [_clock(x) for x in re.findall(TIME, fragment.lower())]
        alt = [a for a in alt if a]
        if len(alt) > len(out):
            return alt
    return out


def find_dims(text: str, known: list[str]) -> Found:
    f = Found(text)
    low = f.low
    zone, zone_name = sentence_zone(text)
    f.zone = zone_name
    # ---- entry times: a range with a step, a list, or "several entry times"
    m = re.search(rf"(?:entry times?|entries|entering|enter(?:ing)? times?)\s+(?:from\s+|between\s+)?{TIME}\s*(?:to|through|until|till|and|-)\s*{TIME}(?:\s*,?\s*(?:in|at|every)\s+(\d+)[- ]?\s*(?:minute|min)s?(?:\s*(?:increments?|steps?|intervals?))?)?", low)
    if not m:
        m2 = re.search(rf"(?:entry times?\s+)?every\s+(\d+)\s*(?:minute|min)s?\s+(?:from|between)\s+{TIME}\s*(?:to|through|until|till|and|-)\s*{TIME}", low)
        if m2:
            ab, step = clocks(text[m2.start():m2.end()], zone), int(m2.group(1))
            if len(ab) >= 2:
                f.add("entry_time", _times(ab[0], ab[-1], step), m2.span())
    else:
        ab = clocks(text[m.start():m.end()], zone)
        if len(ab) >= 2:
            step = int(m.group(3)) if m.group(3) else 30
            f.add("entry_time", _times(ab[0], ab[-1], step), m.span(), assumed=not m.group(3), note="" if m.group(3) else f"No step was given for the entry times: every {step} minutes is assumed.")
    if not any(d["field"] == "entry_time" for d in f.dims):
        m = re.search(rf"(?:entering|entries?|enter|entry|buying|selling|opening)\s+(?:times?\s+)?(?:of\s+)?((?:at\s+)?{TIME}(?:{SEP}(?:at\s+)?{TIME})+)", low)
        if m:
            ts = clocks(text[m.start():m.end()], zone)
            if len(ts) >= 2:
                f.add("entry_time", ts, m.span())
    if not any(d["field"] == "entry_time" for d in f.dims):
        m = re.search(r"(?:across\s+|at\s+|over\s+|with\s+)?(?:multiple|several|different|various|a few)\s+entry times?", low)
        if m:
            f.add("entry_time", ["open", "10:00", "11:00", "13:00"], m.span(), assumed=True, note="\"multiple entry times\" names none: the open, 10:00, 11:00 and 13:00 New York time are assumed.")
    # ---- minutes held after the entry
    m = re.search(rf"(?:exits?|exiting|sell(?:ing)?|clos(?:e|ing)(?:\s+(?:it|them|out))?|get(?:ting)? out|hold(?:ing)?(?:\s+for)?)\s+(?:after|for)?\s*{LIST}\s*(?:minutes?|mins?)\b", low)
    if m:
        f.add("exit_minutes", [int(x) for x in _floats(m.group(1))], m.span())
    else:
        # "holding 30 minutes, holding 2 hours and holding to the close" / "hold for 20 minutes versus 90 minutes versus to the close"
        m = re.search(r"(?:hold(?:ing)?|exit(?:ing)?|sell(?:ing)?|clos(?:e|ing))\b[^.;]*?\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b(?:[^.;]*?(?:\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b|(?:to|until|till|through)\s+the\s+(?:close|closing bell|bell)))+", low)
        if m:
            vals = []
            for h in re.finditer(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b|(?:to|until|till|through)\s+the\s+(close|closing bell|bell)", m.group(0)):
                vals.append("close" if h.group(3) else int(float(h.group(1)) * (60 if h.group(2).startswith("h") else 1)))
            if len(vals) >= 2 and not re.search(r"\bthen\b|\band then\b", m.group(0)):
                f.add("exit_minutes", vals, m.span())
    # ---- expirations
    dtes = list(re.finditer(r"\b(\d+)\s*-?\s*dte\b", low))
    if len(dtes) >= 2:
        f.add("dte", [int(x.group(1)) for x in dtes], (dtes[0].start(), dtes[-1].end()))
    else:
        m = re.search(r"((?:(?:same[- ]day|next[- ]day|\d+[- ]day|weekly|monthly)\s*(?:,|and|or|versus|vs\.?)?\s*){2,})\s*expir\w+", low)
        if m:
            vals = []
            for tok in re.findall(r"same[- ]day|next[- ]day|\d+[- ]day|weekly|monthly", m.group(1)):
                vals.append(0 if tok.startswith("same") else 1 if tok.startswith("next") else 7 if tok == "weekly" else 30 if tok == "monthly" else int(re.match(r"\d+", tok).group(0)))
            f.add("dte", vals, m.span())
    # ---- strikes
    m = re.search(rf"deltas?\s+of\s+{LIST}|{LIST}\s*-?\s*deltas?\b", low)
    if m:
        vals = [("delta", round(x / 100 if x > 1 else x, 4)) for x in _floats(m.group(1) or m.group(2))]
        f.add("strike", vals, m.span())
    else:
        m = re.search(rf"(?:at[- ]the[- ]money|atm)\s+(?:calls?\s+|puts?\s+|contracts?\s+|strikes?\s+)?(?:with|and|versus|vs\.?|against|to)\s+{LIST}\s*%?\s*(?:out[- ]of[- ]the[- ]money|otm)", low)
        if not m:  # one distance: "ATM versus 1% OTM"
            m = re.search(rf"(?:at[- ]the[- ]money|atm)\s+(?:calls?\s+|puts?\s+|contracts?\s+|strikes?\s+)?(?:with|and|versus|vs\.?|against|to|or)\s+({NUMS})\s*%\s*(?:out[- ]of[- ]the[- ]money|otm)(?:\s+(?:contracts?|calls?|puts?|strikes?|options?))?", low)
        if m:
            f.add("strike", [("atm", 0.0)] + [("pct_otm", x) for x in _floats(m.group(1))], m.span())
        else:
            m = re.search(r"\b(?:atm|at[- ]the[- ]money)\s+(?:and|versus|vs\.?|against|with|or)\s+(?:otm|out[- ]of[- ]the[- ]money)(?:\s+(?:contracts?|calls?|puts?|strikes?|options?))?", low)
            if m:
                f.add("strike", [("atm", 0.0), ("pct_otm", 1.0)], m.span(), assumed=True, note="\"OTM\" names no distance: 1% out of the money is assumed (say \"2% OTM\" or \"30 delta\" to change it).")
    # ---- premium targets and stops (always called premium, profit or option here: an underlying target is its own dimension)
    m = re.search(rf"(?:underlying|spy|qqq|iwm)\s+(?:move\s+)?targets?\s+(?:from|between)\s+{NUMS}\s*%?\s*(?:to|and|through|-)\s*{NUMS}\s*%", low)
    if m:
        st, stated, end = _step(low, m.end(), 0.5)
        f.add("underlying_target", _rng(float(m.group(1)), float(m.group(2)), st), (m.start(), end), assumed=not stated, note="" if stated else f"No step was given for the underlying target: {st:g}% is assumed.")
    t_occ = list(re.finditer(rf"{NUMS}\s*%\s*(?:premium\s+|option\s+)?(?:profit\s+)?targets?", low))
    if len(t_occ) >= 2 and not any(d["field"] == "underlying_target" for d in f.dims):
        f.add("premium_target", [float(x.group(1)) for x in t_occ], (t_occ[0].start(), t_occ[-1].end()))
    else:
        m = re.search(rf"(?:premium\s+|profit\s+)targets?\s+(?:of|over|at)\s+{LIST}\s*%?", low)
        if m:
            f.add("premium_target", _floats(m.group(1)), m.span())
        else:
            m = re.search(r"(?:several|multiple|different|various|a few)\s+(?:premium\s+)?(?:profit\s+)?targets", low)
            if m:
                f.add("premium_target", [25.0, 50.0, 100.0], m.span(), assumed=True, note="\"several profit targets\" names none: 25%, 50% and 100% of the premium are assumed.")
    m = re.search(rf"(?:a\s+)?{NUMS}\s*%\s*(?:,|and|or|versus|vs\.?)\s*(?:a\s+)?{NUMS}\s*%(?:\s*(?:,|and|or)\s*(?:a\s+)?{NUMS}\s*%)?\s*(?:premium\s+|option\s+)?stop", low)
    if m:
        f.add("premium_stop", [float(x) for x in m.groups() if x], m.span())
    else:
        m = re.search(rf"(?:premium\s+)?stops?\s+(?:loss\s+)?(?:of|over|at)\s+{LIST}\s*%?", low)
        if m:
            f.add("premium_stop", _floats(m.group(1)), m.span())
    # ---- the previous day's decline
    m = re.search(rf"(?:(?:prior|previous)[- ]day\s+)?(?:declines?|drops?|down days?)\s+(?:of\s+)?(?:from|between)\s+{NUMS}\s*%?\s*(?:to|and|through|-)\s*{NUMS}\s*%", low)
    if m:
        st, stated, end = _step(low, m.end(), 0.5)
        f.add("prev_decline", _rng(float(m.group(1)), float(m.group(2)), st), (m.start(), end), assumed=not stated, note="" if stated else f"No step was given for the decline: {st:g}% is assumed.")
    # ---- the same thing on several underlyings (a list of tickers after a comparison word, never "if SPY and QQQ are both ...")
    tick = "|".join(re.escape(t) for t in sorted(set(NL.TICKERS) | set(known), key=len, reverse=True) if t not in NL.NOT_UNDERLYING)
    m = re.search(rf"(?:across|on|for|between|in)\s+((?:{tick})\b(?:{SEP}(?:{tick})\b)+)", f.text)
    if m and not re.search(r"\b(?:both|if|when|while|unless)\s*$", f.text[:m.start()].rstrip()[-12:], re.I) and not re.match(r"\s*(?:are|is|have|has|were|trade|both)\b", f.text[m.end():], re.I):
        syms = re.findall(rf"\b({tick})\b", m.group(1))
        if len(set(syms)) >= 2 and CUE.search(low):
            f.add("symbol", syms, m.span())
    return f


def _times(a: str, b: str, step: int) -> list[str]:
    to_min = lambda t: 570 if t == "open" else int(t[:2]) * 60 + int(t[3:])
    lo, hi = sorted((to_min(a), to_min(b)))
    return ["open" if x == 570 else f"{x // 60:02d}:{x % 60:02d}" for x in range(lo, hi + 1, max(1, step))][:30]


STRATEGY_WORDS = re.compile(r"\b(buy|sell|long|short|calls?|puts?|straddle|strangle|spread|condor|butterfly|breakout|when|after|if)\b", re.I)


def apply(d: Draft, field: str, v) -> Draft:
    """A copy of the base with one setting changed to one value of a dimension."""
    n = d.model_copy(deep=True)
    if field == "symbol":
        n.symbol, n.symbol_stated = str(v).upper(), True
    elif field == "dte":
        n.dte = int(v)
    elif field == "entry_time":
        n.entry_time, n.entry_time_stated = str(v), None
    elif field == "exit_minutes":
        if v == "close":
            n.hold_minutes, n.exit_time, n.exit_time_stated = None, "close", None
        else:
            n.hold_minutes, n.exit_time, n.exit_time_stated = int(v), None, None
    elif field == "strike":
        m, x = v
        n.strike = DraftStrike(method=m, value=float(x))
    elif field == "premium_target":
        n.premium_target_pct = float(v)
    elif field == "premium_stop":
        n.premium_stop_pct = float(v)
    elif field == "underlying_target":
        n.underlying_target_pct = float(v)
    elif field == "prev_decline":
        x = abs(float(v))
        idx = next((i for i, c in enumerate(n.conditions) if "previous_session.return_pct" in c.expr or c.expr.replace(" ", "").startswith("return_pct(1)")), None)
        new = DraftCondition(expr=f"previous_session.return_pct <= -{x:g}", text=f"previous session fell {x:g}% or more (close-to-close)")
        if idx is None:
            n.conditions.append(new)
        else:
            n.conditions[idx] = new
    else:
        raise ValueError(f"unknown experiment setting {field!r}")
    return n


def label(field: str, v) -> str:
    if field == "strike":
        m, x = v
        return "ATM" if m == "atm" else f"{x:g}% OTM" if m == "pct_otm" else f"{round(x * 100):g} delta"
    if field == "dte":
        return f"{int(v)}DTE"
    if field == "exit_minutes":
        return "to the close" if v == "close" else f"{int(v)} min"
    if field in ("premium_target", "premium_stop", "underlying_target", "prev_decline"):
        return f"{float(v):g}%"
    return "09:30" if v == "open" else str(v)


def configurations(plan: dict) -> list[dict]:
    """Every combination of the plan's dimensions: [{"key": {field: value}, "label": "...", "draft": Draft}]."""
    base: Draft = plan["draft"]
    combos: list[tuple[dict, Draft]] = [({}, base)]
    for dim in plan["dims"]:
        combos = [({**k, dim["field"]: v}, apply(d, dim["field"], v)) for k, d in combos for v in dim["values"]]
    return [{"key": k, "label": ", ".join(label(f, v) for f, v in k.items()) or "as described", "draft": d} for k, d in combos]


def read(text: str, current: Draft | None, known: list[str], default_symbol: str = "SPY") -> dict | None:
    """The experiment a sentence describes, or None when it does not describe one."""
    f = find_dims(text, known)
    if not f.dims:
        return None
    # a list of values for one setting is a comparison whether or not the sentence says "compare"; values that had to be
    # assumed ("several entry times") only count when it does
    if not CUE.search(text) and all(d["assumed"] for d in f.dims):
        return None
    rest = f.masked()
    base, source, notes = None, None, list(f.notes)
    position = re.search(r"\b(calls?|puts?|straddles?|strangles?|spreads?|condors?|butterfl\w+)\b", rest, re.I)
    if position and not re.search(r"\b(buy|sell|short|go long|go short)\b", rest, re.I) and re.search(r"\b(buying|selling|shorting)\b", text, re.I):
        # the verb went with the dimension ("compare buying at 9:45, 10:30 and 11:15 for ATM SPY puts ..."): it is still that trade
        rest = ("Sell " if re.search(r"\b(selling|shorting)\b", text, re.I) else "Buy ") + re.sub(r"^(?:and\s+|for\s+|of\s+|on\s+)+", "", rest, flags=re.I)
    if position and re.search(r"\b(buy|sell|short|go long|go short)\b", rest, re.I):
        from strata.ai import nl_stateful as NLS
        zname = getattr(f, "zone", None)
        if zname and any(d["field"] == "entry_time" for d in f.dims):
            # the times left in the base are read in the same zone as the entry times that were taken out of it
            rest = NL.TIME_RE.sub(lambda m: m.group(0) if (m.group(4) or not (m.group(2) or m.group(3))) else f"{m.group(0).rstrip()} {zname} ", rest)
        base = NLS.enrich(rest, known, default_symbol)
        source = "sentence"
        base.title = rest[:90]
    elif current is not None and current.kind == "options":
        base, source = current.model_copy(deep=True), "current"
        # a condition added in the same breath ("... after a red previous day") still belongs to the base
        if re.search(r"\b(after|when|if|on days)\b", rest, re.I):
            extra = [c for _cat, c in NL.find_conditions(rest)]
            base.conditions += [c for c in extra if c.expr not in {x.expr for x in base.conditions}]
    n = 1
    for d in f.dims:
        n *= len(d["values"])
    if getattr(f, "zone", None) and any(d["field"] == "entry_time" and not d["assumed"] for d in f.dims) and not re.search(r"eastern|pacific|central|mountain|new york|\b[ECMP][SD]?T\b", text, re.I):
        notes.append(f"No time zone was stated: the entry times are read as {f.zone} time and shown in New York time.")
    out = {"action": "experiment", "dims": f.dims, "configs": n, "notes": notes, "remainder": rest, "base_from": source, "fields": {d["field"]: FIELDS[d["field"]] for d in f.dims}}
    if base is None:
        out.update(action="experiment_needs_base", question="Which strategy should these be compared on? Describe the signal and the position (for example \"ATM SPY 0DTE calls at the open after a red previous day\"), and the comparison is applied to it.")
        return out
    # a dimension that varies the previous day's decline needs the condition it varies: the base gets it at the first value
    for d in f.dims:
        if d["field"] == "prev_decline" and not any("previous_session.return_pct" in c.expr for c in base.conditions):
            base = apply(base, "prev_decline", d["values"][0])
    out["draft"] = base
    if n > MAX_CONFIGS:
        out["too_many"] = f"{n:,} combinations ({' x '.join(str(len(d['values'])) for d in f.dims)}) is more than the {MAX_CONFIGS} one experiment may run. Narrow a range or drop a setting; nothing was trimmed for you."
    return out
