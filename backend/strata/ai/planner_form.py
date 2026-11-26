"""A language model as a form-filler: the planner for small local models.

The full planner (`agent.llm_interpret`) hands a model twenty thousand characters of tools, draft fields and an expression
language and takes free expressions back. A hosted model can use that. The 4-8 billion parameter models that run on a laptop
cannot: measured on this machine, the two installed ones picked an unrelated tool or wrote expressions that do not exist.

Here the model does one small thing. It copies what the sentence says into a short form of typed fields, under a JSON schema
the server enforces while the model writes. Everything after that is deterministic, and it is the code a typed-in strategy
goes through:

  * a signal is one of a closed list (SIGNALS). The expression it stands for is written here, never by the model
  * a clock time must be one of the clock times of the sentence; its zone and its conversion are read from the sentence
  * a number must occur in the sentence. A value the model made up is removed and reported, never used
  * what the model could not place goes to `not_expressible`, and from there to the draft's `unsupported` list

The model never states a price, a statistic or a result, and it cannot: the form has no field for one.
"""
from __future__ import annotations

import json
import re

from strata.ai import nl_experiment as NX
from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, diff_drafts
from strata.ai.llm import LLM, LLMError
from strata.engine import expr as X

POSITIONS = {"buy_call": "long_call", "buy_put": "long_put", "sell_put": "short_put", "sell_call": "short_call", "call_debit_spread": "bull_call_spread", "put_debit_spread": "bear_put_spread",
             "put_credit_spread": "bull_put_spread", "call_credit_spread": "bear_call_spread", "iron_condor": "iron_condor", "straddle": "long_straddle", "strangle": "long_strangle"}


def _need(s: dict, key: str, what: str):
    if s.get(key) is None:
        raise ValueError(f"{what} is not stated")
    return s[key]


def _tf(s: dict) -> str:
    m = int(s.get("minutes") or 5)
    return f"{m}m" if m < 60 or m % 60 else f"{m // 60}h"


# type: (what the model is told, the expression it stands for). P is "" or "QQQ." (the same signal read on another symbol)
SIGNALS: dict[str, tuple[str, object]] = {
    "previous_day_down": ("the previous day was red / down / fell; percent = fell AT LEAST this much (null: any decline)", lambda s, P: f"{P}previous_session.return_pct <= -{abs(s['percent']):g}" if s.get("percent") else f"{P}previous_session.return_pct < 0"),
    "previous_day_up": ("the previous day was green / up / rose; percent = rose at least this much (null: any gain)", lambda s, P: f"{P}previous_session.return_pct >= {abs(s['percent']):g}" if s.get("percent") else f"{P}previous_session.return_pct > 0"),
    "gap_down": ("today opened below yesterday's close; percent = by at least this much", lambda s, P: f"{P}gap_pct <= -{abs(s['percent']):g}" if s.get("percent") else f"{P}gap_pct < 0"),
    "gap_up": ("today opened above yesterday's close; percent = by at least this much", lambda s, P: f"{P}gap_pct >= {abs(s['percent']):g}" if s.get("percent") else f"{P}gap_pct > 0"),
    "down_from_open": ("price has fallen `percent` from today's open", lambda s, P: f"{P}from_open_pct <= -{abs(_need(s, 'percent', 'the size of the decline')):g}"),
    "up_from_open": ("price has risen `percent` from today's open", lambda s, P: f"{P}from_open_pct >= {abs(_need(s, 'percent', 'the size of the gain')):g}"),
    "price_above_vwap": ("price is above VWAP", lambda s, P: f"{P}price > {P}vwap"),
    "price_below_vwap": ("price is below VWAP", lambda s, P: f"{P}price < {P}vwap"),
    "crosses_above_vwap": ("price crosses up through VWAP (reclaims VWAP)", lambda s, P: f"crosses_above({P}price, {P}vwap)"),
    "crosses_below_vwap": ("price crosses down through VWAP (loses VWAP)", lambda s, P: f"crosses_below({P}price, {P}vwap)"),
    "breaks_above_opening_range": ("price rises above the HIGH OF THE FIRST `minutes` MINUTES of today (first hour = 60, opening range)", lambda s, P: f"{P}price > {P}opening_range_high({int(_need(s, 'minutes', 'the length of the opening range'))})"),
    "breaks_below_opening_range": ("price falls below the LOW OF THE FIRST `minutes` MINUTES of today (first hour = 60, opening range)", lambda s, P: f"{P}price < {P}opening_range_low({int(_need(s, 'minutes', 'the length of the opening range'))})"),
    "first_candle_breaks_previous_low": ("today's first `minutes`-minute candle trades below YESTERDAY's low (only when the sentence mentions the previous day)", lambda s, P: f"{P}opening_range_low({int(_need(s, 'minutes', 'the length of the first candle'))}) < {P}prev_low"),
    "first_candle_breaks_previous_high": ("today's first `minutes`-minute candle trades above YESTERDAY's high (only when the sentence mentions the previous day)", lambda s, P: f"{P}opening_range_high({int(_need(s, 'minutes', 'the length of the first candle'))}) > {P}prev_high"),
    "breaks_previous_high": ("price crosses above yesterday's high", lambda s, P: f"crosses_above({P}price, {P}prev_high)"),
    "breaks_previous_low": ("price crosses below yesterday's low", lambda s, P: f"crosses_below({P}price, {P}prev_low)"),
    "bearish_candles_in_a_row": ("`count` red candles in a row, each `minutes` minutes long", lambda s, P: f'{P}consecutive_bearish_candles("{_tf(s)}") >= {int(s.get("count") or 1)}'),
    "bullish_candles_in_a_row": ("`count` green candles in a row, each `minutes` minutes long (a green candle / a bullish reversal candle = count 1)", lambda s, P: f'{P}consecutive_bullish_candles("{_tf(s)}") >= {int(s.get("count") or 1)}'),
    "vix_above": ("the VIX is above `level`", lambda s, P: f"vix > {float(_need(s, 'level', 'the VIX level')):g}"),
    "vix_below": ("the VIX is below `level`", lambda s, P: f"vix < {float(_need(s, 'level', 'the VIX level')):g}"),
}
SIGNAL_TEXT = {"previous_day_down": "the previous session fell", "previous_day_up": "the previous session rose", "gap_down": "the session gapped down", "gap_up": "the session gapped up", "down_from_open": "down from the open",
               "up_from_open": "up from the open", "price_above_vwap": "price above VWAP", "price_below_vwap": "price below VWAP", "crosses_above_vwap": "price crosses above VWAP", "crosses_below_vwap": "price crosses below VWAP",
               "breaks_above_opening_range": "price breaks above the opening range", "breaks_below_opening_range": "price breaks below the opening range", "first_candle_breaks_previous_low": "the first candle breaks yesterday's low",
               "first_candle_breaks_previous_high": "the first candle breaks yesterday's high", "breaks_previous_high": "price crosses above yesterday's high", "breaks_previous_low": "price crosses below yesterday's low",
               "bearish_candles_in_a_row": "consecutive bearish candles", "bullish_candles_in_a_row": "consecutive bullish candles", "vix_above": "VIX above a level", "vix_below": "VIX below a level"}
TAKES = {"previous_day_down": ("percent",), "previous_day_up": ("percent",), "gap_down": ("percent",), "gap_up": ("percent",), "down_from_open": ("percent",), "up_from_open": ("percent",),
         "breaks_above_opening_range": ("minutes",), "breaks_below_opening_range": ("minutes",), "first_candle_breaks_previous_low": ("minutes",), "first_candle_breaks_previous_high": ("minutes",),
         "bearish_candles_in_a_row": ("count", "minutes"), "bullish_candles_in_a_row": ("count", "minutes"), "vix_above": ("level",), "vix_below": ("level",)}
# the words a sentence has to contain for a signal to be what it says. A signal the sentence has no word for is not used
PREVIOUS = r"yesterday|previous|prior|preceding|day before|last (?:day|session)|overnight|\bred day|\bgreen day|\bdown day|\bup day"
ANCHOR = {"previous_day_down": PREVIOUS, "previous_day_up": PREVIOUS, "gap_down": r"\bgaps?\b|gapp|opens? (?:lower|below|down)|opening (?:lower|below|down)", "gap_up": r"\bgaps?\b|gapp|opens? (?:higher|above|up)|opening (?:higher|above|up)",
          "down_from_open": r"open", "up_from_open": r"open", "price_above_vwap": r"vwap", "price_below_vwap": r"vwap", "crosses_above_vwap": r"vwap", "crosses_below_vwap": r"vwap",
          "breaks_above_opening_range": r"opening range|first \w+(?:[- ]\w+)?(?: of)?|\borb\b|initial", "breaks_below_opening_range": r"opening range|first \w+(?:[- ]\w+)?(?: of)?|\borb\b|initial",
          "first_candle_breaks_previous_low": PREVIOUS, "first_candle_breaks_previous_high": PREVIOUS, "breaks_previous_high": PREVIOUS, "breaks_previous_low": PREVIOUS,
          "bearish_candles_in_a_row": r"candle|\bbars?\b", "bullish_candles_in_a_row": r"candle|\bbars?\b", "vix_above": r"\bvix\b|volatility index", "vix_below": r"\bvix\b|volatility index"}
SETTINGS = ["symbol", "dte", "entry_time", "exit_after_minutes", "strike", "premium_target", "premium_stop", "underlying_target", "previous_day_decline"]
SETTING_FIELD = {"exit_after_minutes": "exit_minutes", "previous_day_decline": "prev_decline"}


def _opt(t, **kw) -> dict:
    return {"type": [t, "null"], **kw}


def schema(symbols: list[str]) -> dict:
    sig = {"type": "object", "additionalProperties": False, "required": ["type", "percent", "minutes", "count", "level", "on_symbol"],
           "properties": {"type": {"type": "string", "enum": list(SIGNALS)}, "percent": _opt("number"), "minutes": _opt("integer"), "count": _opt("integer"), "level": _opt("number"), "on_symbol": _opt("string")}}
    cmp_ = {"type": "object", "additionalProperties": False, "required": ["setting", "values"], "properties": {"setting": {"type": "string", "enum": SETTINGS}, "values": {"type": "array", "items": {"type": "string"}}}}
    props = {
        "request": {"type": "string", "enum": ["strategy", "comparison", "change", "other"]},
        "symbol": _opt("string"), "position": {"enum": list(POSITIONS) + [None]}, "dte": _opt("integer"),
        "strike_type": {"enum": ["atm", "otm_percent", "itm_percent", "delta", None]}, "strike_value": _opt("number"), "spread_width": _opt("number"), "contracts": _opt("integer"),
        "entry_time": _opt("string"), "signals": {"type": "array", "items": sig}, "signals_in_order": {"type": "boolean"}, "entries_per_day": _opt("integer"),
        "exit_after_minutes": _opt("integer"), "exit_time": _opt("string"), "hold_days": _opt("integer"), "hold_to_expiration": {"type": "boolean"}, "exit_signals": {"type": "array", "items": sig},
        "premium_target_percent": _opt("number"), "premium_stop_percent": _opt("number"), "underlying_target_percent": _opt("number"), "underlying_stop_percent": _opt("number"),
        "partial_exit_fraction": _opt("number"), "partial_exit_premium_percent": _opt("number"),
        "compare": {"type": "array", "items": cmp_}, "not_expressible": {"type": "array", "items": {"type": "string"}},
    }
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


EMPTY = {"request": "strategy", "symbol": None, "position": None, "dte": None, "strike_type": None, "strike_value": None, "spread_width": None, "contracts": None, "entry_time": None, "signals": [], "signals_in_order": False,
         "entries_per_day": None, "exit_after_minutes": None, "exit_time": None, "hold_days": None, "hold_to_expiration": False, "exit_signals": [], "premium_target_percent": None, "premium_stop_percent": None, "underlying_target_percent": None,
         "underlying_stop_percent": None, "partial_exit_fraction": None, "partial_exit_premium_percent": None, "compare": [], "not_expressible": []}


def _sig(type_, **kw) -> dict:
    return {"type": type_, "percent": None, "minutes": None, "count": None, "level": None, "on_symbol": None, **kw}


# worked examples for the prompt. None of them is a sentence of the evaluation sets
EXAMPLES = [
    ("Sell two IWM 2DTE puts 1.5% out of the money at 11:20 AM Central if the previous day gained at least 0.8%; take profit at 40% of the premium, stop at 90%.",
     {**EMPTY, "symbol": "IWM", "position": "sell_put", "dte": 2, "strike_type": "otm_percent", "strike_value": 1.5, "contracts": 2, "entry_time": "11:20 AM Central", "signals": [_sig("previous_day_up", percent=0.8)],
      "premium_target_percent": 40, "premium_stop_percent": 90}),
    ("When QQQ falls 1.2% from its open and then crosses back above VWAP, buy weekly calls and sell them after 25 minutes.",
     {**EMPTY, "symbol": "QQQ", "position": "buy_call", "dte": 7, "signals": [_sig("down_from_open", percent=1.2), _sig("crosses_above_vwap")], "signals_in_order": True, "exit_after_minutes": 25}),
    ("Buy IWM 1DTE puts after two green 15-minute candles in a row followed by a red one.",
     {**EMPTY, "symbol": "IWM", "position": "buy_put", "dte": 1, "signals": [_sig("bullish_candles_in_a_row", count=2, minutes=15), _sig("bearish_candles_in_a_row", count=1, minutes=15)], "signals_in_order": True}),
    ("Try that with 5, 10 and 20 minute holds and with 35 versus 45 delta strikes.",
     {**EMPTY, "request": "comparison", "compare": [{"setting": "exit_after_minutes", "values": ["5", "10", "20"]}, {"setting": "strike", "values": ["35 delta", "45 delta"]}]}),
    ("Buy SPY 0DTE calls at the open when retail sentiment is extremely bearish, out by 12:45 PM.",
     {**EMPTY, "symbol": "SPY", "position": "buy_call", "dte": 0, "entry_time": "the open", "exit_time": "12:45 PM", "not_expressible": ["when retail sentiment is extremely bearish"]}),
]


def prompt(current: Draft | None) -> str:
    sig = "\n".join(f"  {k}: {v[0]}" for k, v in SIGNALS.items())
    ex = "\n".join(f"Sentence: {t}\nForm: {json.dumps(f, separators=(',', ':'))}" for t, f in EXAMPLES)
    cur = ""
    if current is not None:
        cur = ("\nThe user is already discussing this strategy: " + json.dumps({k: v for k, v in current.model_dump(exclude_none=True, include={"symbol", "structure", "dte", "entry_time", "hold_minutes", "premium_target_pct", "premium_stop_pct"}).items()})
               + "\nA sentence that names no position of its own is about it: request is \"change\" (it changes a setting) or \"comparison\" (it compares several values).")
    return f"""You fill in a form about ONE options trading request. A separate engine does every calculation. You only copy what the sentence says into the right field.

Rules
- Copy only what the sentence states. Use null (or an empty list) for everything it does not state. Never guess, never add a default.
- request: "strategy" = it describes a position to open; "comparison" = it asks to compare or test SEVERAL values of a setting; "change" = it changes the strategy under discussion; "other" = a question or anything else.
- position: buy_call, buy_put, sell_put, sell_call or a spread. dte: days to expiration (0DTE = 0, weekly = 7).
- strike: atm (at the money), otm_percent / itm_percent with strike_value = the percent, delta with strike_value = the delta (30 delta = 30).
- entry_time / exit_time: copy the clock exactly as written including AM/PM and time zone ("10:15 AM Eastern"); "the open" only if the sentence says to enter AT the open. null when no clock time is stated.
  "down 0.5% from the open" is a price move, not an entry time: entry_time stays null.
- exit_after_minutes: "sell after 45 minutes". hold_days: "hold for 3 trading days", "sell the next day" = 1. hold_to_expiration: true only if the sentence says to hold to expiration.
- exit_signals: a market condition that CLOSES the position ("sell when price falls back below VWAP"), from the same list as signals.
- entries_per_day: "up to three times a day" = 3. spread_width: the width of a spread in points ("5-point wide" = 5).
- A percent of the OPTION PREMIUM goes in premium_target_percent / premium_stop_percent. A percent move of the STOCK or ETF goes in underlying_target_percent / underlying_stop_percent. Do not mix them.
- partial exit: "sell half when the premium is up 25%" = partial_exit_fraction 0.5, partial_exit_premium_percent 25.
- signals: the market conditions for the ENTRY, each one from the list below. signals_in_order = true when one must happen AFTER another ("after X ... Y", "X followed by Y", "X and then Y");
  then list them in the order they HAPPEN, earliest first: "buy when B after A" = [A, B].
  on_symbol: only when the signal is about ANOTHER ticker than the one traded ("with QQQ confirmation" = the same signal again with on_symbol "QQQ").
- compare: one entry per setting that has several values. Write every value out as text ("0", "1", "7" or "9:30 AM", "10:00 AM" or "atm", "1% otm", "30 delta"). A setting named without values
  ("several profit targets", "multiple entry times") gets an empty values list. [] when nothing is compared. A target and a stop are two different settings, not two values of one.
- not_expressible: the user's own words for any condition or instruction that has no field and no signal here. Never drop it and never replace it with something similar.

Signals
{sig}
{cur}
Examples
{ex}"""


# ----------------------------------------------------------------------------- what the sentence contains
NLS_FRAC = r"\b(?:half|a third|one third|two thirds|a quarter|one quarter|three quarters|\d+(?:\.\d+)?\s*% of (?:the|my) (?:position|contracts))\b"
HOURS = re.compile(r"(\d+(?:\.\d+)?|an?|one|two|three|four|half an?|half|first|the)\s*-?\s*(hours?|hrs?)\b")


def numbers_in(text: str) -> set[float]:
    """Every number the sentence states: digits, number words, fractions, and hours as minutes."""
    low = text.lower().replace("-", " ")
    low = re.sub(r"\b\d{1,2}:\d{2}\b(?:\s*[ap]\.?m\b\.?)?|\b\d{1,2}\s*[ap]\.?m\b\.?", " ", low)  # the digits of a clock time are not a size: "12:30 PM" states no 30
    out = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", low)}
    out |= {float(v) for k, v in NL.NUM_WORDS.items() if len(k) > 2 and re.search(rf"\b{k}\b", low)}
    for w, v in (("half", 0.5), ("third", 1 / 3), ("quarter", 0.25), ("double", 100.0), ("doubles", 100.0), ("triple", 200.0), ("twice", 2.0), ("first", 1.0), ("single", 1.0), ("zero", 0.0), ("same day", 0.0), ("next day", 1.0),
                 ("tomorrow", 1.0), ("weekly", 7.0), ("weeklies", 7.0), ("next week", 7.0), ("monthly", 30.0), ("monthlies", 30.0), ("noon", 12.0)):
        if re.search(rf"\b{w}\b", low):
            out.add(v)
    for m in HOURS.finditer(low):
        tok = m.group(1)
        n = 0.5 if tok.startswith("half") else float(tok) if tok[0].isdigit() else float(NL.NUM_WORDS.get(tok, 1))
        out.add(n * 60)
    return out


def _stated(v, nums: set[float], scale: tuple[float, ...] = (1.0,)) -> bool:
    return any(abs(float(v) * s - n) < 1e-9 for n in nums for s in scale)


def _clock_of(tok: str) -> tuple[int, int] | str | None:
    low = str(tok).lower()
    m = re.search(r"(\d{1,2})(?::(\d{2}))?", low)
    if m and (m.group(2) or re.search(r"[ap]\.?m|eastern|pacific|central|mountain|\b[ecmp][sd]?t\b|o'?clock", low)):
        return int(m.group(1)) % 12, int(m.group(2) or 0)
    if re.search(r"\bopen", low):
        return "open"
    if re.search(r"\bclos|\beod\b|end of", low):
        return "close"
    if re.search(r"noon|midday", low):
        return 0, 0
    return None


def sentence_time(tok: str, text: str, used: set[int]) -> dict | None:
    """The clock time of the SENTENCE the model's copy points at (the conversion to New York time is the sentence reader's)."""
    want = _clock_of(tok)
    if want is None:
        return None
    low = text.lower()
    if want == "open" and not re.search(r"\b(?:at|on|right at|into) the (?:market )?open(?:ing bell|ing)?\b|\bat (?:the )?(?:market )?open\b|\bmarket open\b|\bopening bell\b|\b9:30\b", low):
        return None  # "down 0.5% from the open" names a price, not an entry time
    if want == "close" and not re.search(r"\b(?:at|on|by|into|near|to|until|till|through) the (?:market )?clos(?:e|ing bell)\b|\bat (?:market )?close\b|\bmarket close\b|\bclosing bell\b|\bend of (?:the )?day\b|\beod\b|\b(?:4|16):00\b", low):
        return None
    for t in NL.find_times(text, None):
        if t["pos"] in used:
            continue
        got = t["et"] if t["et"] in ("open", "close") else _clock_of(t["stated"].split(" (")[0])
        bell = {"open": (9, 30), "close": (4, 0)}
        if got == want or (t["et"] == "12:00" and want == (0, 0)) or bell.get(got if isinstance(got, str) else "") == want or bell.get(want if isinstance(want, str) else "") == got:
            used.add(t["pos"])
            return t
    return None


# ----------------------------------------------------------------------------- the form as a draft
def _cross_ok(expr: str) -> bool:
    return not X.validate(expr)


def to_draft(form: dict, text: str, current: Draft | None, known: list[str], default_symbol: str = "SPY") -> tuple[dict, list[str], list[str]]:
    """(raw draft fields the form states, values that were removed because the sentence does not contain them, unsupported parts)."""
    nums = numbers_in(text)
    low = text.lower()
    raw: dict = {}
    dropped: list[str] = []
    unsupported = [str(x).strip() for x in form.get("not_expressible") or [] if str(x).strip() and str(x).strip().lower().strip(" .") in low]
    invented = [str(x).strip() for x in form.get("not_expressible") or [] if str(x).strip() and str(x).strip().lower().strip(" .") not in low]
    if invented:  # words that are not in the sentence are not the user's: they are reported, not attributed to them
        dropped.append("an unplaced part the model described in its own words: " + "; ".join(invented)[:160])

    def num(key: str, label: str, scale=(1.0,), always=()):
        v = form.get(key)
        if v is None:
            return None
        if _stated(v, nums, scale) or v in always:
            return v
        dropped.append(f"{label} {v:g}")
        return None

    sym = str(form.get("symbol") or "").upper().strip()
    if sym:
        found = NL.find_symbol(text, known)
        if found == sym or re.search(rf"\b{re.escape(sym)}\b", text, re.I):
            raw["symbol"], raw["symbol_stated"] = sym, True
        elif found:
            raw["symbol"], raw["symbol_stated"] = found, True
        else:
            dropped.append(f"the symbol {sym}")
    if form.get("position"):
        raw["structure"] = POSITIONS[form["position"]]
        words = NL.find_structure(low)
        if words and words != raw["structure"]:  # "puts" in the sentence and a call in the form: the words are the user's
            dropped.append(f"the position {raw['structure'].replace('_', ' ')} (the words say {words.replace('_', ' ')})")
            raw["structure"] = words
    dte = num("dte", "an expiration of")
    if dte is not None:
        raw["dte"] = int(dte)
    st = form.get("strike_type")
    if st == "atm":
        if re.search(r"\batm\b|at[- ]the[- ]money", low):
            raw["strike"] = {"method": "atm", "value": 0}
    elif st in ("otm_percent", "itm_percent"):
        v = num("strike_value", "a strike distance of")
        if v is not None:
            raw["strike"] = {"method": "pct_otm", "value": abs(float(v)) * (1 if st == "otm_percent" else -1)}
    elif st == "delta":
        v = num("strike_value", "a delta of", scale=(1.0, 100.0))
        if v is not None:
            raw["strike"] = {"method": "delta", "value": float(v) / 100 if float(v) > 1 else float(v)}
    w = num("spread_width", "a spread width of")
    if w is not None and float(w) > 0 and re.search(r"spread|condor|butterfly", raw.get("structure", "")):
        raw["width"] = float(w)
    q = num("contracts", "a size of", always=(1,))
    if q and int(q) > 1:
        raw["quantity"] = int(q)
    used: set[int] = set()
    for key, field in (("entry_time", "entry_time"), ("exit_time", "exit_time")):
        tok = form.get(key)
        if not tok:
            continue
        t = sentence_time(tok, text, used)
        if t is None:
            dropped.append(f"the {'entry' if key == 'entry_time' else 'exit'} time \"{tok}\"")
            continue
        raw[field] = "open" if (t["et"] == "09:30" and field == "entry_time") else t["et"]
        raw[field + "_stated"] = None
        if t.get("assumed_tz"):
            raw["tz_assumed"] = t["assumed_tz"]
    for key, field, label in (("exit_after_minutes", "hold_minutes", "an exit after"), ("entries_per_day", "max_entries", "entries a day of"), ("premium_target_percent", "premium_target_pct", "a premium target of"),
                              ("premium_stop_percent", "premium_stop_pct", "a premium stop of"), ("underlying_target_percent", "underlying_target_pct", "an underlying target of"),
                              ("underlying_stop_percent", "underlying_stop_pct", "an underlying stop of")):
        v = num(key, label)
        if v is not None and float(v) > 0:
            raw[field] = int(v) if field in ("hold_minutes", "max_entries") else abs(float(v))
    if raw.get("max_entries") == 1:
        raw.pop("max_entries")
    hd = num("hold_days", "a holding period of")
    if hd is not None and int(hd) > 0:
        raw["hold_sessions"] = int(hd)
    if form.get("hold_to_expiration") and re.search(r"expir", low):
        raw["hold_to_expiration"] = True
    frac, at = form.get("partial_exit_fraction"), num("partial_exit_premium_percent", "a partial exit at")
    if frac is not None and at is None and raw.get("premium_target_pct") is not None and re.search(NLS_FRAC, low):
        at = raw.pop("premium_target_pct")  # "sell half when the premium is up 25%" states one level, and it is the partial's
    if frac is not None and at is not None:
        f = float(frac) / 100 if float(frac) > 1 else float(frac)
        if _stated(f, nums) or _stated(f * 100, nums):
            raw["partials"] = [{"premium_pct": abs(float(at)), "fraction": f, "text": f"close {f:g} of the position when the premium is up {abs(float(at)):g}%"}]
        else:
            dropped.append(f"a partial exit of {f:g} of the position")
    # ---- signals: the expression is written here
    first_candle = None

    def read_signals(items: list, entry: bool) -> list[tuple[str, str]]:
        nonlocal first_candle
        exprs: list[tuple[str, str]] = []
        for s in items or []:
            one = _one_signal(s, entry, exprs, items)
            if one:
                exprs.append(one)
        return list(dict.fromkeys(exprs))

    def _one_signal(s: dict, entry: bool, exprs: list, items: list):
        nonlocal first_candle
        ty = s.get("type")
        if ty not in SIGNALS:
            return None
        if not re.search(ANCHOR[ty], low):
            unsupported.append(f"the language model read \"{SIGNAL_TEXT[ty]}\" here, but the sentence has no words for it: that part of the rule was not understood")
            return None
        s = dict(s)
        for k, scale in (("percent", (1.0,)), ("minutes", (1.0,)), ("count", (1.0,)), ("level", (1.0,))):
            if s.get(k) is None:
                continue
            if not _stated(s[k], nums, scale):
                if k == "count" and int(s[k]) == 1:
                    continue  # "a green candle"
                dropped.append(f"{SIGNAL_TEXT[ty]}: {k} {float(s[k]):g}")
                s[k] = None
            elif k not in TAKES.get(ty, ()) and not (k == "count" and int(s[k]) == 1):
                # the number is in the sentence, but this signal has no use for it: something the sentence says was not placed
                unsupported.append(f"\"{float(s[k]):g}\" was read as part of \"{SIGNAL_TEXT[ty]}\", which takes no {k}: part of the entry rule may be missing")
        other = str(s.get("on_symbol") or "").upper().strip()
        P = ""
        if other and other != raw.get("symbol", (current.symbol if current else default_symbol)):
            if not re.search(rf"\b{re.escape(other)}\b", text, re.I) and NL.find_symbol(text.replace(raw.get("symbol", ""), ""), known) != other:
                dropped.append(f"{SIGNAL_TEXT[ty]} on {other}")
                return None
            P = other + "."
        try:
            e = SIGNALS[ty][1](s, P)
        except ValueError as err:
            unsupported.append(f"{SIGNAL_TEXT[ty]} ({err})")
            return None
        if X.validate(e):
            unsupported.append(f"{SIGNAL_TEXT[ty]}{' on ' + other if P else ''}")
            return None
        if ty.startswith("first_candle") and not P and entry:
            first_candle = int(s["minutes"])
        if P and re.search(r"\bconfirm", low):
            # "... with QQQ confirmation": the traded symbol shows the move and the other one confirms it
            own = SIGNALS[ty][1](s, "")
            if not any(x[0] == own for x in exprs) and not any((f.get("type") == ty and not f.get("on_symbol")) for f in items or []):
                exprs.append((own, SIGNAL_TEXT[ty]))
        said = ", ".join(f"{'at least ' if k == 'percent' else ''}{float(s[k]):g}{'%' if k == 'percent' else ' min' if k == 'minutes' else ''}" for k in TAKES.get(ty, ()) if s.get(k) is not None)
        return e, SIGNAL_TEXT[ty] + (f": {said}" if said else "") + (f" ({other})" if P else "")

    exprs = read_signals(form.get("signals"), True)
    stated_signals = len([f for f in form.get("signals") or [] if f.get("type") in SIGNALS])
    kinds = [f.get("type") for f in form.get("signals") or [] if f.get("type") in SIGNALS]
    if form.get("signals_in_order") and len(exprs) == 2 and len(kinds) == 2 and kinds[0] != kinds[1]:
        # "buy when B after A": the words give B first and A second, and A happens first. Order in time is what a sequence means
        at = [m.start() if (m := re.search(ANCHOR[k], low)) else -1 for k in kinds]
        after = [m.start() for m in re.finditer(r"\bafter\b|\bonce\b|\bfollowing\b", low)]
        if at[0] >= 0 and at[1] >= 0 and any(at[0] < x < at[1] for x in after):
            exprs.reverse()
    if form.get("signals_in_order") and len(exprs) >= 2 and stated_signals >= 2:  # a signal and its confirmation on another symbol hold together
        raw["sequence"] = [{"name": f"step{i + 1}", "expr": e, "text": t} for i, (e, t) in enumerate(exprs)]
    elif exprs:
        raw["conditions"] = [{"expr": e, "text": t} for e, t in exprs]
    outs = read_signals(form.get("exit_signals"), False)
    if outs:
        raw["exit_conditions"] = [{"expr": e, "text": t} for e, t in outs]
    # dates, weekdays, costs and the kind of price are read from the sentence by the readers every request goes through
    from strata.ai import nl_stateful as NLS
    days = NL.find_weekdays(low)
    if days:
        raw["weekdays"] = days
    a, b, _why = NL.find_period(low)
    if a or b:
        raw["start"], raw["end"] = a, b
    costs, (mode, _note) = NLS.read_costs(low), NLS.read_mode(low)
    if costs is not None:
        raw["costs"] = costs.model_dump()
    if mode:
        raw["mode"] = mode
    if first_candle and "entry_time" not in raw:
        m = 570 + first_candle  # the candle has to be complete before it can have broken anything
        raw["entry_time"], raw["entry_time_stated"] = f"{m // 60:02d}:{m % 60:02d}", None
    return raw, dropped, unsupported


def _values(setting: str, values: list, text: str, known: list[str]) -> tuple[list, list[str]]:
    """Typed values of one compared setting, and the ones that could not be read."""
    field = SETTING_FIELD.get(setting, setting)
    zone, _name = NX.sentence_zone(text)
    out, bad = [], []
    for v in values or []:
        s = str(v).strip()
        low = s.lower()
        try:
            if field == "symbol":
                t = s.upper()
                if t not in set(known) | set(NL.TICKERS):
                    raise ValueError("not a known ticker")
                out.append(t)
            elif field == "dte":
                out.append(0 if re.search(r"same|zero", low) else 1 if re.search(r"next[- ]day|tomorrow", low) else 7 if "week" in low else 30 if "month" in low else int(re.search(r"\d+", low).group(0)))
            elif field == "entry_time":
                if re.search(r"\bopen", low):
                    out.append("open")
                else:
                    ts = NX.clocks(s if re.search(r"[ap]\.?m|:", low) else s + ":00", zone)
                    if not ts:
                        raise ValueError("not a time inside the session")
                    out.append(ts[0])
            elif field == "exit_minutes":
                if re.search(r"clos|eod|end of", low):
                    out.append("close")
                else:
                    n = float(re.search(r"\d+(?:\.\d+)?", low).group(0))
                    out.append(int(n * 60) if re.search(r"hour|hr", low) else int(n))
            elif field == "strike":
                if re.search(r"\batm\b|at[- ]the[- ]money", low):
                    out.append(("atm", 0.0))
                elif "delta" in low:
                    n = float(re.search(r"\d+(?:\.\d+)?", low).group(0))
                    out.append(("delta", n / 100 if n > 1 else n))
                else:
                    n = float(re.search(r"\d+(?:\.\d+)?", low).group(0))
                    out.append(("pct_otm", -n if re.search(r"\bitm\b|in[- ]the[- ]money", low) else n))
            else:
                out.append(abs(float(re.search(r"\d+(?:\.\d+)?", low).group(0))))
        except (AttributeError, ValueError) as e:
            bad.append(f"{s!r} ({e})" if str(e) else repr(s))
    return list(dict.fromkeys(out)), bad


SCALAR_OF = {"premium_target": "premium_target_percent", "premium_stop": "premium_stop_percent", "underlying_target": "underlying_target_percent", "exit_minutes": "exit_after_minutes", "dte": "dte"}
ASSUMED = {"entry_time": (["open", "10:00", "11:00", "13:00"], "No entry times were named: the open, 10:00, 11:00 and 13:00 New York time are assumed."),
           "premium_target": ([25.0, 50.0, 100.0], "No profit targets were named: 25%, 50% and 100% of the premium are assumed.")}


def _number(v) -> float | None:
    if isinstance(v, tuple):
        return float(v[1])
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compared(form: dict, text: str, known: list[str], notes: list[str]) -> tuple[list[dict], list[str]]:
    """The settings to compare and their values.

    Lists, ranges and steps are arithmetic, and the arithmetic reader of the sentence does them (`nl_experiment.find_dims`):
    where it finds a setting, its values are used. The model adds the settings that reader's patterns missed. A number has one
    role: "a 30% target versus a 15% stop" gives a target and a stop, not two values of the target."""
    found = NX.find_dims(text, known)
    asked = bool(form.get("compare")) or form.get("request") == "comparison" or bool(NX.CUE.search(text))
    dims = [dict(d) for d in found.dims if asked or not d["assumed"]]
    if dims:
        notes += [n for n in found.notes if n not in notes]
    bad_values: list[str] = []
    for c in form.get("compare") or []:
        field = SETTING_FIELD.get(c.get("setting"), c.get("setting"))
        if any(d["field"] == field for d in dims):
            continue
        vals, bad = _values(c.get("setting"), c.get("values"), text, known)
        other = {float(form[k]) for f, k in SCALAR_OF.items() if f != field and form.get(k) is not None}
        vals = [v for v in vals if not (_number(v) is not None and _number(v) in other)]
        if not (c.get("values") or []) and field in ASSUMED:
            vals, note = ASSUMED[field]
            dims.append({"field": field, "values": list(vals), "text": "", "assumed": True})
            notes.append(note)
            continue
        if field == "strike" and any(re.fullmatch(r"\s*(?:otm|out[- ]of[- ]the[- ]money)\s*", str(v).lower()) for v in c.get("values") or []):
            vals = list(dict.fromkeys(vals + [("pct_otm", 1.0)]))
            bad = [b for b in bad if not re.search(r"otm|out[- ]of", b.lower())]
            notes.append("\"OTM\" names no distance: 1% out of the money is assumed (say \"2% OTM\" or \"30 delta\" to change it).")
        if bad:
            bad_values.append(f"{NX.FIELDS.get(field, field)}: {', '.join(bad)}")
        if len(vals) >= 2:
            dims.append({"field": field, "values": vals, "text": "", "assumed": False})
    return dims, bad_values


def to_route(form: dict, text: str, current: Draft | None, store, known: list[str], default_symbol: str = "SPY") -> dict:
    """The filled form as the action the built-in readers would return for the same understanding."""
    from strata.ai.agent import draft_problems
    from strata.ai.nl_stateful import questions_for
    req = form.get("request")
    if req == "other":
        return {"action": "unknown", "text": "The language model read this as a question rather than a strategy."}
    raw, dropped, unsupported = to_draft(form, text, current, known, default_symbol)
    notes = [f"The language model also proposed {x}. That is not in your sentence, so it was not used." for x in dropped]
    dims, bad_values = compared(form, text, known, notes)
    own_position = "structure" in raw
    if own_position or current is None or current.kind != "options":
        if not own_position:
            if dims:
                return {"action": "experiment_needs_base", "dims": dims, "configs": 0, "notes": notes, "fields": {d["field"]: NX.FIELDS[d["field"]] for d in dims}, "base_from": None, "remainder": "",
                        "question": "Which strategy should these be compared on? Describe the signal and the position, and the comparison is applied to it."}
            return {"action": "unknown", "text": "The language model found no position (calls, puts or a spread) in this request."}
        if "symbol" not in raw:
            raw["symbol"], raw["symbol_stated"] = default_symbol, False
        d, errs = draft_problems({"kind": "options", **raw, "unsupported": unsupported, "notes": notes}, text.strip())
        src = "sentence"
    else:
        merged = {**current.model_dump(), **{k: v for k, v in raw.items() if k not in ("conditions", "sequence")}}
        if raw.get("conditions"):
            merged["conditions"] = [c.model_dump() for c in current.conditions] + [c for c in raw["conditions"] if c["expr"] not in {x.expr for x in current.conditions}]
        if raw.get("sequence"):
            merged["sequence"] = raw["sequence"]
        merged["unsupported"] = list(dict.fromkeys(list(current.unsupported) + unsupported))
        merged["notes"] = list(current.notes) + notes
        merged["questions"] = []
        d, errs = draft_problems(merged, current.title or text.strip())
        src = "current"
    if errs or d is None:
        raise LLMError("the filled form does not make a valid strategy: " + "; ".join(errs)[:300])
    if bad_values:
        d.unsupported = list(d.unsupported) + [f"values to compare that could not be read ({x})" for x in bad_values]
    if dims:
        base = d
        for dm in dims:
            if dm["field"] == "prev_decline" and not any("previous_session.return_pct" in c.expr for c in base.conditions):
                base = NX.apply(base, "prev_decline", dm["values"][0])
        n = 1
        for dm in dims:
            n *= len(dm["values"])
        out = {"action": "experiment", "draft": base, "dims": dims, "configs": n, "notes": notes, "remainder": "", "base_from": src, "fields": {dm["field"]: NX.FIELDS[dm["field"]] for dm in dims}}
        if n > NX.MAX_CONFIGS:
            out["too_many"] = f"{n:,} combinations is more than the {NX.MAX_CONFIGS} one experiment may run. Narrow a range or drop a setting; nothing was trimmed for you."
        return out
    d.questions = d.questions or questions_for(text, d)
    if src == "sentence":
        # the check every interpretation goes through: a size or a clock time of the sentence that has no place in the reading is
        # said, as a part that could not be represented, and is never passed over
        from strata.ai import intent as I
        it = I.from_route({"action": "new", "draft": d}, text, store, default_symbol)
        lost = [c["detail"] for c in it.get("checks") or [] if not c["ok"] and c["check"] != "Parts that could not be represented"]
        if lost:
            d.unsupported = list(d.unsupported) + [f"{x} (the language model's reading left it out)" for x in lost]
    if src == "current":
        changed = [x["field"] for x in diff_drafts(current, d)]
        if not changed:
            return {"action": "unknown", "text": "The language model found nothing in this request that changes the current strategy."}
        return {"action": "modify", "draft": d, "changed": changed, "compare": False}
    return {"action": "new", "draft": d, "sweep": None}


def form_interpret(llm: LLM, text: str, current: Draft | None, store, known: list[str] | None = None, default_symbol: str = "SPY", cancelled=None) -> dict:
    """One schema-constrained call (one more if the reply is not the form), then the deterministic mapping above."""
    known = known if known is not None else sorted(store.stored_symbols())
    sch = schema(known)
    msgs = [{"role": "user", "content": f"Sentence: {text.strip()}\nForm:"}]
    last = None
    for _ in range(2):
        if cancelled and cancelled():
            raise LLMError("cancelled")
        obj = llm.json(prompt(current), msgs, max_tokens=900, schema=sch)
        if isinstance(obj, dict) and obj.get("request") in ("strategy", "comparison", "change", "other"):
            form = {**EMPTY, **{k: v for k, v in obj.items() if k in EMPTY}}
            route = to_route(form, text, current, store, known, default_symbol)
            route["form"] = form
            return route
        last = obj
        msgs += [{"role": "assistant", "content": json.dumps(obj)[:1500]}, {"role": "user", "content": "That is not the form. Reply with the form as one JSON object with every field."}]
    raise LLMError("the model did not return the form: " + json.dumps(last)[:200])
