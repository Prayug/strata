"""A language model as the first reader of every message, with every part of its reading checked before anything runs.

The model (any provider in llm.py: a hosted model or a local Ollama model) reads the trader's message, the question or
strategy on screen and the last few turns, and returns ONE JSON object in Strata's question language: what kind of
message it is (a new question, a strategy, a change to what is on screen, an action on the result, something that cannot
be answered, something unclear), and the parts of the question, each with the exact words of the message it comes from.

The model never supplies a number that is used. Everything it returns is checked here, deterministically:

  vocabulary   only the condition, outcome and command names below exist (the JSON schema enforces them as well)
  grounding    every part quotes the message; a quote that is not in the message is rejected. Percentages, minutes, days,
               counts and clock times are read again from the quoted words by the deterministic readers (nl_rules,
               semantics): the model only says which words belong to which part. A number the words do not contain is
               rejected.
  coverage     the deterministic tagger lists what in the message carries meaning (instruments, numbers, times, events,
               outcomes, option words); each must lie inside a quoted part or be named as unclear or unsupported. A part
               nobody placed is asked about: nothing runs on a reading that dropped a clause.
  edits        a change may only touch what the new message quotes; everything else is carried over unchanged
  cross-check  the deterministic reader (semantics.py) reads the same message. When it reads it completely and the two
               readings differ, the user is asked which was meant, with both shown; nothing runs on a disagreement.

What is built is the same question frame the deterministic reader builds (semantics.py), so everything after it (the
studies, answers, explore, undo, comparisons) is shared.
"""
from __future__ import annotations

import copy
import json
import re

from strata.ai import nl_rules as NL
from strata.ai import semantics as SM

COND_KEYS = ["prev_return", "prev_otc", "gap", "from_open", "change", "first_candle", "weekday", "vix", "up_streak", "down_streak",
             "above_vwap", "below_vwap", "cross_above_vwap", "cross_below_vwap"]
OUT_KEYS = ["close_vs", "touch", "touch_level", "event", "recover", "reverse", "continue", "move"]
COMMANDS = ["explore", "explain", "compare", "undo", "reliability", "next", "about_selection", "teach", "prices"]
STRUCTS = ["long_call", "long_put", "short_call", "short_put", "long_straddle", "short_straddle", "long_strangle", "short_strangle", "iron_condor",
           "iron_butterfly", "bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread", "covered_call"]

_Q = {"type": "string"}
_N = {"type": ["number", "null"]}
_COND = {"type": "object", "properties": {"k": {"type": "string", "enum": COND_KEYS}, "q": _Q, "op": {"type": ["string", "null"], "enum": [">=", "<=", ">", "<", None]},
                                          "v": _N, "min": {"type": ["integer", "null"]}, "dir": {"type": ["string", "null"], "enum": ["up", "down", None]},
                                          "day": {"type": ["string", "null"], "enum": ["Mon", "Tue", "Wed", "Thu", "Fri", None]}}, "required": ["k", "q"]}
_SPAN = {"type": ["object", "null"], "properties": {"q": _Q}, "required": ["q"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "act": {"type": "string", "enum": ["ask", "strategy", "edit", "command", "unsupported", "unclear"]},
        "kind": {"type": ["string", "null"], "enum": ["frequency", "agreement", "movement", "timing", "breakdown", "strategy", None]},
        "cmd": {"type": ["string", "null"], "enum": COMMANDS + [None]},
        "filter": {"type": ["string", "null"], "enum": ["losing", "winning", "worst", "best", "hit", "miss", "all", None]},
        "symbol": {"type": ["object", "null"], "properties": {"v": _Q, "q": _Q}, "required": ["v", "q"]},
        "conds": {"type": ["array", "null"], "items": _COND},
        "out": {"type": ["object", "null"], "properties": {"k": {"type": "string", "enum": OUT_KEYS}, "q": _Q, "ref": {"type": ["string", "null"], "enum": ["open", "prev_close", "start", None]},
                                                            "dir": {"type": ["string", "null"], "enum": ["up", "down", None]}, "v": _N,
                                                            "lvl": {"type": ["string", "null"], "enum": ["prev_close", "prev_high", "prev_low", "open", None]},
                                                            "ek": {"type": ["string", "null"], "enum": COND_KEYS + [None]}}, "required": ["k", "q"]},
        "start": _SPAN, "end": _SPAN, "period": _SPAN,
        "candle": {"type": ["object", "null"], "properties": {"q": _Q}, "required": ["q"]},
        "scope": {"type": ["string", "null"], "enum": ["rest", "day", "both", None]},
        "pred": {"type": ["string", "null"], "enum": ["first_candle", "prev_direction", None]},
        "measure": {"type": ["string", "null"], "enum": ["high_time", "low_time", "abs_move", "open_to_close", "close_to_close", "close>open", None]},
        "by": {"type": ["string", "null"], "enum": ["weekday", "month", None]},
        "focus": {"type": ["string", "null"], "enum": ["Mon", "Tue", "Wed", "Thu", "Fri", None]},
        "strat": {"type": ["object", "null"], "properties": {
            "struct": {"type": "string", "enum": STRUCTS}, "q": _Q, "dte": {"type": ["object", "null"], "properties": {"q": _Q}, "required": ["q"]},
            "strike": {"type": ["object", "null"], "properties": {"m": {"type": "string", "enum": ["atm", "otm", "itm", "delta"]}, "q": _Q}, "required": ["m", "q"]},
            "entry": _SPAN, "exit": _SPAN, "hold": _SPAN,
            "ptarget": _SPAN, "pstop": _SPAN, "utarget": _SPAN, "ustop": _SPAN, "entries": _SPAN,
            "prices": {"type": ["string", "null"], "enum": ["trades", "quotes", "modeled", None]}}, "required": ["struct", "q"]},
        "size": {"type": ["object", "null"], "properties": {
            "mode": {"type": "string", "enum": ["contracts", "dollars", "risk_dollars", "risk_pct", "equity_pct"]}, "q": _Q,
            "capital": _SPAN, "reinvest": {"type": ["boolean", "null"]}, "daily_stop": _SPAN}, "required": ["mode", "q"]},
        "drop": {"type": ["array", "null"], "items": {"type": "string", "enum": ["conds", "ptarget", "pstop", "utarget", "ustop", "exit", "hold", "period", "size", "daily_stop"]}},
        "unclear": {"type": ["object", "null"], "properties": {"q": _Q, "ask": _Q, "opts": {"type": "array", "items": _Q}}, "required": ["q", "ask"]},
        "unsupported": {"type": ["object", "null"], "properties": {"q": _Q, "why": _Q, "closest": _Q}, "required": ["q", "why"]},
    },
    "required": ["act"],
}

SYSTEM = """You read messages to Strata, a tool that answers questions about the stored price history of SPY, QQQ and IWM (and VIX as a condition), and backtests options strategies on stored option trade prices (from 2024-01-18 on). You return ONE compact JSON object, on one line, describing what the message asks. You never compute or state a result.

act: "ask" a market question; "strategy" an options trade to backtest; "edit" a change to what is CURRENT (only when CURRENT is given); "command" an action on the current result; "unsupported" it cannot be answered from the stored data; "unclear" one essential part is missing or could mean different things.

Questions (act ask), kind:
 frequency: how often something happens ("how often", "odds", "chance", "what % of the time"; "does it usually / tend to" + an outcome such as recover, close higher, fill the gap)
 agreement: does one direction predict another (the first N-minute candle vs the rest of the day / the whole day; yesterday's direction vs today)
 movement: what usually happens / how far it moves / the typical path, after something
 timing: when during the day (the time of the high or low, which part of the day moves most)
 breakdown: grouped by weekday or by month (seasonality, "are fridays worse")
Parts (each with q = the EXACT words of the message, copied):
 symbol {v, q}
 conds: conditions that hold before the outcome is measured. k:
  prev_return the previous session's move, close to previous close ("after a 2% down day" op <= v -2; "yesterday was up" op > v 0); v is a percent, never a count of days
  prev_otc the previous session's candle colour, close vs its own open ("yesterday was red" when meant as the candle)
  gap the open against the previous close ("gaps down 1%" op <= v -1; "opens higher")
  from_open the price against today's open at the start time ("down 1% from the open at 10")
  change the price against the previous close at the start time ("red on the day at 1pm")
  first_candle the first N-minute candle's direction (min N, dir up|down)
  weekday (day Mon..Fri); vix the VIX's previous close (op, v); up_streak / down_streak N sessions in a row (v N)
  above_vwap, below_vwap, cross_above_vwap, cross_below_vwap
 out: what is counted. k:
  close_vs the close (or the price at the end) compared with ref open|prev_close|start, dir up|down ("closes green" ref open up; "closes higher than yesterday" ref prev_close up)
  touch reaches v% from ref open|start|prev_close at any time ("hits +0.5% from the open" v 0.5)
  touch_level trades back to lvl prev_close (a gap fill), prev_high, prev_low, open
  event the condition itself is what is counted, ek its key ("how often does IWM gap up 1%": ek gap)
  recover, reverse, continue: those words, unexplained; move: where the price ends up (movement)
 start / end {q}: the words saying when the measurement starts and ends ("at 10am", "before noon", "in the first hour", "by the close", "the next day", "2 days later")
 candle {q}: the words naming the candle ("first 15 minutes"); scope rest (after the candle) | day (whole day) | both; pred first_candle | prev_direction
 measure: timing high_time|low_time|abs_move; breakdown open_to_close|close_to_close|close>open.  by weekday|month; focus a weekday asked about
 period {q}: years or dates ("since 2020", "in 2022")
Strategies (act strategy, kind strategy): strat {struct, q, dte{q}, strike{m atm|otm|itm|delta, q}, entry{q}, exit{q}, hold{q} ("after 45 min", "2 days later"), ptarget{q} gain of the OPTION premium, pstop{q} loss of the premium, utarget{q} / ustop{q} a move of the UNDERLYING ("when SPY is up 0.5%"), entries{q} times per day, prices trades|quotes|modeled}; conds as above (entry conditions);
 size {mode contracts|dollars|risk_dollars|risk_pct|equity_pct, q, capital{q}, reinvest, daily_stop{q}}: "buy 3 contracts" contracts; "spend $500 a trade" dollars; "risk $300 per trade" risk_dollars; "risk 2% of the account" risk_pct; "put 5% of the account in" equity_pct; "start with 10k" capital; "stop for the day after losing $200" daily_stop.
Edits (act edit): give ONLY the parts that change, same field names. conds = the full new list when conditions change ("what if yesterday was red too" adds one; "only when the drop was 1.5%" changes one). drop = parts removed ("no stop" drop pstop).
Commands (act command), cmd: explore (list the days or trades behind the result; filter losing|winning|worst|best|hit|miss|all), explain (why a trade or day did what it did), compare (versions side by side), undo, reliability (luck, noise, is it real, significance), next (what to investigate next), about_selection (a question about the days or trades listed or selected: "anything in common", "anything interesting about the losers"), teach (explain a concept or the chart), prices (the option prices used).
unsupported {q, why, closest}: forecasts ("will it go up tomorrow"), event dates (FOMC, CPI, earnings, opex), indicators (RSI, MACD, moving averages), symbols not stored, option prices before 2024-01-18; closest = the nearest question Strata CAN answer, as a full sentence.
unclear {q, ask, opts}: ask ONE short question; opts 2-3 full rewritten messages. Only when a knowledgeable trader could not tell what is meant.
Rules: copy quotes exactly from THIS message, never from the examples; give only parts whose words are in the message (symbol only when a ticker is named; CURRENT carries over); a candle, a time or a period the message mentions gets its own part with its words; never invent a number; times without a zone are as written."""

EXAMPLES = [
    ("How often does SPY close green after a 2% gap down?", None,
     {"act": "ask", "kind": "frequency", "symbol": {"v": "SPY", "q": "SPY"}, "conds": [{"k": "gap", "q": "2% gap down", "op": "<=", "v": -2}], "out": {"k": "close_vs", "q": "close green", "ref": "open", "dir": "up"}}),
    ("does the first 15 min candle predict the rest of the day on qqq", None,
     {"act": "ask", "kind": "agreement", "symbol": {"v": "QQQ", "q": "qqq"}, "pred": "first_candle", "candle": {"q": "first 15 min candle"}, "scope": "rest"}),
    ("what does iwm do in the first hour after falling 1.5% the day before", None,
     {"act": "ask", "kind": "movement", "symbol": {"v": "IWM", "q": "iwm"}, "conds": [{"k": "prev_return", "q": "falling 1.5% the day before", "op": "<=", "v": -1.5}], "out": {"k": "move", "q": "what does iwm do"}, "end": {"q": "in the first hour"}}),
    ("when does spy usually make its low", None, {"act": "ask", "kind": "timing", "symbol": {"v": "SPY", "q": "spy"}, "measure": "low_time"}),
    ("odds qqq fills the gap after opening down 0.5%+", None,
     {"act": "ask", "kind": "frequency", "symbol": {"v": "QQQ", "q": "qqq"}, "conds": [{"k": "gap", "q": "opening down 0.5%+", "op": "<=", "v": -0.5}], "out": {"k": "touch_level", "q": "fills the gap", "lvl": "prev_close"}}),
    ("buy spy 0dte calls at 6:45am PT after a red day, take profit 30%, stop 15%, spend $400 a trade", None,
     {"act": "strategy", "kind": "strategy", "symbol": {"v": "SPY", "q": "spy"}, "conds": [{"k": "prev_return", "q": "after a red day", "op": "<", "v": 0}],
      "strat": {"struct": "long_call", "q": "buy spy 0dte calls", "dte": {"q": "0dte"}, "strike": {"m": "atm", "q": "calls"}, "entry": {"q": "at 6:45am PT"}, "ptarget": {"q": "take profit 30%"}, "pstop": {"q": "stop 15%"}},
      "size": {"mode": "dollars", "q": "spend $400 a trade"}}),
    ("use QQQ instead", {"kind": "frequency", "symbol": "SPY"}, {"act": "edit", "symbol": {"v": "QQQ", "q": "QQQ"}}),
    ("what if yesterday was red too", {"kind": "agreement", "symbol": "QQQ", "conds": []}, {"act": "edit", "conds": [{"k": "prev_return", "q": "yesterday was red", "op": "<", "v": 0}]}),
    ("exit an hour later", {"kind": "strategy", "exit": "14:00"}, {"act": "edit", "strat": {"struct": "long_call", "q": "exit an hour later", "exit": {"q": "an hour later"}}}),
    ("show me where this went wrong", {"kind": "strategy"}, {"act": "command", "cmd": "explore", "filter": "losing"}),
    ("anything interesting about the losing trades?", {"kind": "strategy"}, {"act": "command", "cmd": "about_selection", "filter": "losing"}),
    ("what would you look at next", {"kind": "frequency"}, {"act": "command", "cmd": "next"}),
    ("how does qqq trade on cpi days", None, {"act": "unsupported", "unsupported": {"q": "cpi days", "why": "no calendar of economic releases is stored", "closest": "How does QQQ trade the day after a 1% gap down?"}}),
    ("does it bounce", None, {"act": "unclear", "unclear": {"q": "does it bounce", "ask": "Bounce after what, and on which ticker?", "opts": ["How often does SPY close higher the day after a 1% down day?", "How often does SPY get back to its open after falling 1% from the open?"]}}),
]


def invalid(raw: dict) -> str | None:
    """A reply outside the schema (any provider can return one: not every adapter enforces it): what is wrong, or None."""
    if not isinstance(raw, dict):
        return "the reply is not an object"
    allowed = set(SCHEMA["properties"])
    extra = sorted(k for k in raw if k not in allowed)
    if extra:
        return f"the reply has fields Strata does not know ({', '.join(extra[:4])})"
    for c in raw.get("conds") or []:
        if not isinstance(c, dict) or c.get("k") not in COND_KEYS or not isinstance(c.get("q"), str):
            return "a condition is not one Strata knows"
    o = raw.get("out")
    if o is not None and (not isinstance(o, dict) or o.get("k") not in OUT_KEYS):
        return "the outcome is not one Strata knows"
    sym = raw.get("symbol")
    if sym is not None and (not isinstance(sym, dict) or not re.fullmatch(r"[A-Za-z]{1,6}", str(sym.get("v") or ""))):
        return "the ticker is not a ticker"
    st = raw.get("strat")
    if st is not None and (not isinstance(st, dict) or st.get("struct") not in STRUCTS):
        return "the position is not one Strata knows"
    if raw.get("cmd") is not None and raw.get("cmd") not in COMMANDS:
        return "the command is not one Strata knows"
    return None


def _compact(d: dict | None) -> dict | None:
    if not d:
        return None
    keep = {k: v for k, v in d.items() if k in ("kind", "symbol", "conditions", "outcome", "start", "end", "candle", "scope", "predictor", "measure", "bucket", "by", "focus", "period") and v not in (None, [], "")}
    if "conditions" in keep:
        keep["conds"] = [SM.describe_condition(c) for c in keep.pop("conditions")]
    if "outcome" in keep:
        keep["out"] = SM.describe_outcome(keep.pop("outcome"), d.get("start") or "open")
    return keep


def context_of(frame: dict | None, draft, result_kind: str | None, recent: list[dict], selection: dict | None) -> dict:
    """What the model is shown besides the message: the question or strategy on screen, in words; the last turns."""
    cur = None
    if frame and frame.get("kind") in ("frequency", "agreement", "movement", "timing", "breakdown"):
        cur = _compact(frame)
    elif draft is not None and getattr(draft, "kind", None) == "options":
        cur = {"kind": "strategy", "symbol": draft.symbol, "structure": draft.structure, "dte": draft.dte, "entry": draft.entry_time, "exit": draft.exit_time,
               "conds": [c.text or c.expr for c in draft.conditions], "premium_target_pct": draft.premium_target_pct, "premium_stop_pct": draft.premium_stop_pct,
               "underlying_target_pct": draft.underlying_target_pct, "quantity": draft.quantity, "sizing": draft.sizing.model_dump() if getattr(draft, "sizing", None) else None}
        cur = {k: v for k, v in cur.items() if v not in (None, [], "")}
    elif draft is not None:
        cur = {"kind": "study", "title": draft.title}
    return {"current": cur, "result_on_screen": result_kind, "recent": recent[-4:], "selection": selection}


def ask_model(llm, text: str, ctx: dict) -> dict:
    """One call. The reply must be a JSON object; its shape is checked by `build`, not trusted."""
    shots = []
    for msg, cur, out in EXAMPLES:
        shots.append({"role": "user", "content": json.dumps({"message": msg, "current": cur}, separators=(",", ":"))})
        shots.append({"role": "assistant", "content": json.dumps(out, separators=(",", ":"))})
    payload = {"message": text, "current": ctx.get("current"), "result_on_screen": ctx.get("result_on_screen"), "recent": ctx.get("recent") or [], "selection": ctx.get("selection")}
    llm.purpose = "read"
    return llm.json(SYSTEM, shots + [{"role": "user", "content": json.dumps(payload, separators=(",", ":"), default=str)}], schema=SCHEMA, max_tokens=900)


# ---------------------------------------------------------------------------------------------------- grounding helpers
import threading

_TL = threading.local()  # per request: the parts of the reading whose quotes were not in the message


def reset() -> None:
    _TL.dropped = []


def dropped() -> list[str]:
    return list(getattr(_TL, "dropped", None) or [])


def _drop(q: str) -> None:
    if not hasattr(_TL, "dropped"):
        _TL.dropped = []
    _TL.dropped.append(q)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%$.+\-:/' ]", " ", (s or "").lower())).strip()


def find_quote(text: str, q: str) -> tuple[int, int] | None:
    """Where the quoted words are in the message (case, spacing and punctuation ignored; one typo per word tolerated)."""
    if not q or not q.strip():
        return None
    t, qq = text.lower(), q.lower().strip().strip("\"'")
    i = t.find(qq)
    if i >= 0:
        return i, i + len(qq)
    # token-wise: the quote's words in order, each equal or within one edit of a word of the message
    tw = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[\w%$.+\-:']+", t)]
    qw = re.findall(r"[\w%$.+\-:']+", qq)
    if not qw:
        return None
    for s0 in range(len(tw)):
        j, ok = s0, True
        for w in qw:
            if j >= len(tw):
                ok = False
                break
            a = tw[j][2]
            if not (a == w or (len(w) > 3 and SM._edit1(a, w)) or a.rstrip("s") == w.rstrip("s")):
                ok = False
                break
            j += 1
        if ok:
            return tw[s0][0], tw[j - 1][1]
    return None


def _ticker_of(sym: dict) -> str | None:
    """The ticker the quoted words name: the quote itself, or a company name the readers know. A value the words do not
    name (a quote "TST" with a value "SPY") is not a ticker of this message."""
    q = str(sym.get("q") or sym.get("v") or "").strip()
    v = str(sym.get("v") or "").upper().strip()
    words = re.findall(r"[A-Za-z]+", q)
    named = {w.upper() for w in words} | {SM.NAME_TICKERS[w.lower()] for w in words if w.lower() in SM.NAME_TICKERS}
    return v if v in named else (next(iter(named)) if len(named) == 1 and not v else None)


def _numbers(q: str) -> list[float]:
    n = SM.normalise(q)
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", n.norm)]


DOWNW = re.compile(r"\b(?:down|lower|red|fall|fell|falling|drop|dropp|declin|slid|sell|selloff|tank|dump|bearish|negative|loss|losing|lose|bad|below|under|weak)\w*")
UPW = re.compile(r"\b(?:up|higher|green|ris|rose|rally|gain|climb|bullish|positive|advanc|pop|jump|above|over|strong|good)\w*")


def _direction(q: str) -> str | None:
    low = SM.normalise(q).norm
    d, u = bool(DOWNW.search(low)), bool(UPW.search(low))
    return "down" if d and not u else "up" if u and not d else None


def _clock(q: str, msg_tz: str | None) -> dict | None:
    ts = NL.find_times(q, msg_tz)
    return ts[0] if ts else None


# --------------------------------------------------------------------------------------------------------- the frame
class Built:
    """What `build` returns: a route for the agent, or a reason to hand the message to the deterministic readers."""

    def __init__(self, route: dict | None = None, defer: str | None = None, problems: list[str] | None = None, frame: dict | None = None):
        self.route, self.defer, self.problems, self.frame = route, defer, problems or [], frame


def _cond(c: dict, text: str, problems: list[str], msg_tz: str | None) -> dict | None:
    k, q = c.get("k"), (c.get("q") or "").strip()
    if k not in COND_KEYS:
        problems.append(f"unknown condition {k!r}")
        return None
    if not find_quote(text, q):
        _drop(q)  # not the user's words (an example's, or invented): left out; coverage decides whether anything is missing
        return None
    nums = _numbers(q)
    d = _direction(q) or c.get("dir")
    out: dict = {"text": q}
    why = _key_mismatch(k, q)
    if why:
        problems.append(why)
        return None
    if k in ("prev_return", "prev_otc", "gap", "from_open", "change"):
        out["key"] = k
        nq = SM.normalise(q).norm
        if nums and not re.search(r"\d\s*%|\bpercent|\bpct\b", nq):
            nums = []  # "3 red days", "2 green days": a count, not a size
        if nums:
            v = nums[0]
            if c.get("v") not in (None, 0) and abs(abs(float(c["v"])) - v) > 1e-9:
                problems.append(f"the model gave {c['v']} for \"{q}\", whose number is {v:g}")
            down = d == "down" or (d is None and (c.get("op") in ("<", "<=") or (c.get("v") or 0) < 0))
            out.update(op="<=" if down else ">=", value=-v if down else v)
        else:
            if c.get("v") not in (None, 0):
                problems.append(f"\"{q}\" contains no number, but the reading has {c['v']}")
            down = d == "down" or (d is None and c.get("op") in ("<", "<="))
            if d is None and c.get("op") is None:
                problems.append(f"no direction in \"{q}\"")
            out.update(op="<" if down else ">", value=0.0)
            if re.search(r"\b(?:big|large|huge|massive|sharp|really|very|bad|terrible|a lot|significant)\b", q.lower()):
                out["size"] = "vague"
        return out
    if k == "first_candle":
        mins = None
        m = re.search(r"(\d+)\s*-?\s*(?:min|minute|m\b)", q.lower())
        if m:
            mins = int(m.group(1))
        elif re.search(r"\bfirst hour|opening hour|1st hour\b", q.lower()):
            mins = 60
        elif re.search(r"half[- ]?hour", q.lower()):
            mins = 30
        if c.get("min") and mins and int(c["min"]) != mins:
            problems.append(f"the candle in \"{q}\" is {mins} minutes, the reading says {c['min']}")
        return {"key": "first_candle", "minutes": mins, "direction": d, "text": q}
    if k == "weekday":
        m = re.search(r"\b(mon|tue|wed|thu|fri)", q.lower())
        if not m:
            problems.append(f"no weekday in \"{q}\"")
            return None
        return {"key": "weekday", "op": "=", "value": m.group(1).capitalize(), "text": q}
    if k == "vix":
        if not nums:
            problems.append(f"no VIX level in \"{q}\"")
            return None
        below = bool(re.search(r"\b(?:below|under|less than|lower than|<)", q.lower())) or (not re.search(r"\b(?:above|over|greater|higher|>)", q.lower()) and c.get("op") in ("<", "<="))
        return {"key": "vix", "op": "<" if below else ">", "value": nums[-1], "text": q}
    if k in ("up_streak", "down_streak"):
        if not nums:
            problems.append(f"no count in \"{q}\"")
            return None
        key = k if d is None else ("down_streak" if d == "down" else "up_streak")
        return {"key": key, "op": ">=", "value": nums[0], "text": q}
    if k in ("above_vwap", "below_vwap"):
        return {"key": "vwap", "op": ">" if k == "above_vwap" else "<", "text": q}
    if k in ("cross_above_vwap", "cross_below_vwap"):
        return {"key": "vwap_cross", "op": "above" if k == "cross_above_vwap" else "below", "text": q}
    return None


KEY_WORDS = {
    "vix": r"\bvix\b|\bvolatility index\b",
    "gap": r"\bgap|\bopen(?:s|ed|ing)?\b(?:\s+\w+){0,3}\s+(?:up|down|higher|lower|above|below|green|red)|\b(?:up|down|higher|lower)\s+(?:at|on|from)\s+the\s+open",
    "first_candle": r"\bcandle|\bbar\b|\bfirst\b|\bopening\b|\b1st\b|\binitial\b",
    "weekday": r"\b(?:mon|tue|wed|thu|fri)",
    "up_streak": r"in a row|straight|consecutive|streak|\b\d+\s+(?:\w+\s+)?(?:days|sessions)\b",
    "down_streak": r"in a row|straight|consecutive|streak|\b\d+\s+(?:\w+\s+)?(?:days|sessions)\b",
    "above_vwap": r"vwap", "below_vwap": r"vwap", "cross_above_vwap": r"vwap", "cross_below_vwap": r"vwap",
    "from_open": r"\bopen",
}


def _key_mismatch(k: str, q: str) -> str | None:
    """A part filed under a condition its own words do not name ("vix under 15" as a previous-day move)."""
    low = SM.normalise(q).norm
    if k in KEY_WORDS and not re.search(KEY_WORDS[k], low):
        return f"\"{q}\" does not say what a {k} condition is about"
    if k != "vix" and re.search(KEY_WORDS["vix"], low):
        return f"\"{q}\" is about the VIX, not {k}"
    if k in ("prev_return", "prev_otc") and re.search(r"in a row|straight|consecutive|streak", low):
        return f"\"{q}\" is a streak, not a single previous session"
    if k in ("prev_return", "prev_otc") and re.search(r"\bgap|\bopen(?:s|ed|ing)?\b", low) and not re.search(r"yesterday|previous|prior|day before", low):
        return f"\"{q}\" is about the open, not the previous session"
    return None


def _outcome(o: dict, text: str, problems: list[str], conds: list[dict]) -> dict | None:
    k, q = o.get("k"), (o.get("q") or "").strip()
    if k not in OUT_KEYS:
        problems.append(f"unknown outcome {k!r}")
        return None
    if not find_quote(text, q):
        _drop(q)
        return None
    low = q.lower()
    d = _direction(q) or o.get("dir")
    if k == "close_vs":
        ref = o.get("ref") or "open"
        if re.search(r"yesterday|previous|prior|last close|day before", low):
            ref = "prev_close"
        elif re.search(r"\bopen", low):
            ref = "open"
        if d is None:
            problems.append(f"no direction in \"{q}\"")
            d = "up"
        return {"key": "close_vs", "ref": ref, "direction": d, "text": q}
    if k == "touch":
        nums = _numbers(q)
        if not nums:
            problems.append(f"no size in \"{q}\"")
            return None
        v = nums[0]
        neg = bool(re.search(r"(?:^|\s)-\s?\d", q)) or (d == "down")
        ref = "open" if re.search(r"\bopen", low) else "prev_close" if re.search(r"yesterday|previous|prior", low) else (o.get("ref") or "start")
        return {"key": "touch", "value": -v if neg else v, "ref": ref, "text": q}
    if k == "touch_level":
        lvl = o.get("lvl")
        if re.search(r"\bgap\b|\bfill", low):
            lvl = "prev_close"
        elif re.search(r"(?:yesterday|previous|prior)\W*(?:day\W*)?(?:s\W*)?high|\bhigh of (?:yesterday|the previous)", low):
            lvl = "prev_high"
        elif re.search(r"(?:yesterday|previous|prior)\W*(?:day\W*)?(?:s\W*)?low", low):
            lvl = "prev_low"
        if lvl not in ("prev_close", "prev_high", "prev_low", "open"):
            problems.append(f"no level in \"{q}\"")
            return None
        direction = None if lvl == "prev_close" and re.search(r"\bgap\b|\bfill", low) else ("up" if lvl == "prev_high" else "down" if lvl == "prev_low" else d)
        return {"key": "touch_level", "level": lvl, "direction": direction, "text": q}
    if k == "event":
        ek = o.get("ek") or ("gap" if re.search(r"\bgap|\bopen", low) else "prev_return")
        ev = _cond({"k": ek, "q": q, "op": None, "v": None, "dir": o.get("dir")}, text, problems, None)
        if ev is None:
            return None
        if ev["key"] == "prev_return":
            ev["key"] = "day_return"  # counted on its own: the session's own close-to-close move
        return {"key": "event", "event": ev, "text": q}
    if k in ("recover", "reverse", "continue"):
        return {"key": k, "text": q}
    return {"key": "end_return", "text": q}


def _anchor(q: str, text: str, msg_tz: str | None, role: str, problems: list[str]) -> dict:
    """When a measurement starts or ends, from its words: a clock time (any zone), the open or close, the first N minutes,
    the next session, N sessions later."""
    if not find_quote(text, q):
        _drop(q)
        return {}
    low = q.lower()
    t = _clock(q, msg_tz)
    if t is not None and t["et"] not in ("open", "close"):
        mins = int(t["et"][:2]) * 60 + int(t["et"][3:])
        if not (570 <= mins <= 960):
            problems.append(f"\"{q}\" is {t['et']} New York time, outside the session")
        return {"at": "open" if t["et"] == "09:30" else t["et"], "assumed_tz": t.get("assumed_tz"), "stated": t.get("stated")}
    m = re.search(r"first (\d+)\s*(min|minute|hour)", low)
    if m:
        n = int(m.group(1)) * (60 if m.group(2).startswith("hour") else 1)
        return {"at": SM._hhmm(570 + n)}
    if re.search(r"first hour|opening hour", low):
        return {"at": "10:30"}
    if re.search(r"first half ?hour", low):
        return {"at": "10:00"}
    m = re.search(r"(\d+|a|one|two|three|four|five)\s+(?:trading\s+)?(day|session|week)s?\s+(later|after|out|on)", low) or re.search(r"(?:within|in|over)\s+(?:the\s+next\s+)?(\d+|a|one|two|three|four|five)\s+(?:trading\s+)?(day|session|week)s?", low)
    if m:
        wn = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        n = int(m.group(1)) if m.group(1).isdigit() else wn[m.group(1)]
        n *= 5 if m.group(2) == "week" else 1
        return {"at": f"sessions:{n}", "at_end": bool(m.lastindex and m.lastindex >= 3 and m.group(3) in ("later", "after", "out", "on"))}
    if re.search(r"\b(?:next|following) (?:day|session|trading day)|\btomorrow|the day after", low):
        return {"at": "next_session"}
    if re.search(r"\bnoon\b|\blunch|mid ?day", low):
        return {"at": "12:00"}
    if re.search(r"\bclose\b|end of (?:the )?day|eod|\bbell\b(?!.*open)", low):
        return {"at": "close"}
    if re.search(r"\bopen", low):
        return {"at": "open"}
    if re.search(r"rest of (?:the )?(?:day|session)|remainder|after (?:the|that) candle", low):
        return {"scope": "rest"}
    if re.search(r"whole (?:day|session)|entire (?:day|session)|the day\b|the close\b", low):
        return {"scope": "day"}
    problems.append(f"could not tell when \"{q}\" is")
    return {}


def _candle_minutes(q: str) -> int | None:
    low = q.lower()
    m = re.search(r"(\d+)\s*-?\s*(?:min|minute|m\b)", low)
    if m:
        return int(m.group(1))
    if re.search(r"\bhour\b", low) and not re.search(r"half", low):
        return 60
    if re.search(r"half ?hour", low):
        return 30
    return None


def question_frame(raw: dict, text: str, known: list[str], default_symbol: str, base: dict | None = None) -> tuple[dict | None, list[str], list[str]]:
    """A question frame from the model's reading: (frame, problems, quotes). With `base` (an edit), only what the reading
    names changes."""
    problems: list[str] = []
    quotes: list[str] = []
    msg_tz = next((NL.TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in text.lower()), None)
    if msg_tz is None:
        mt = re.search(r"\b(pst|pdt|pt|est|edt|et|cst|cdt|ct|mst|mdt|mt)\b", text.lower())
        if mt:
            msg_tz = NL.TZ_OFFSET.get({"pst": "pacific", "pdt": "pacific", "pt": "pacific", "est": "eastern", "edt": "eastern", "et": "eastern", "cst": "central", "cdt": "central",
                                       "ct": "central", "mst": "mountain", "mdt": "mountain", "mt": "mountain"}[mt.group(1)])
    f = copy.deepcopy(base) if base else {"kind": raw.get("kind"), "text": text, "assumptions": [], "conditions": [], "symbols": [], "period": [None, None]}
    f["text"] = text
    f.setdefault("assumptions", [])
    if raw.get("kind") and raw.get("kind") != "strategy" and (not base or raw.get("kind") != f.get("kind")):
        f["kind"] = raw["kind"]
    sym = raw.get("symbol")
    if sym:
        v = _ticker_of(sym)
        if not v or not find_quote(text, sym.get("q") or v):
            _drop(sym.get("q") or v)
            if not base:
                f["symbol"], f["symbols"], f["symbol_stated"] = default_symbol, [], False
                f["assumptions"].append({"slot": "symbol", "text": f"No ticker was named, so {default_symbol} is used."})
        else:
            quotes.append(sym.get("q") or v)
            stored = [s.upper() for s in known]
            if v not in stored:
                return {"kind": "unsupported", "why": f"No price history for {v} is stored.", "symbol": v}, problems, quotes
            f["symbol"], f["symbols"], f["symbol_stated"] = v, [v], True
    elif not base:
        f["symbol"], f["symbols"], f["symbol_stated"] = default_symbol, [], False
        f["assumptions"].append({"slot": "symbol", "text": f"No ticker was named, so {default_symbol} is used."})
    removal = bool(re.search(r"\b(?:without|remove|drop|no longer|ignore|any day|regardless|get rid of|take out|no (?:condition|filter))\b", text.lower()))
    if raw.get("conds") is not None and base and not raw["conds"] and not removal:
        raw = {**raw, "conds": None}  # an empty list in an edit that removes nothing is not a removal
    if raw.get("conds") is not None:
        conds = []
        for c in raw["conds"]:
            cc = _cond(c, text, problems, msg_tz)
            if cc is not None:
                conds.append(cc)
                quotes.append(cc["text"])
        if base and not removal:
            # a condition of the current question that the message neither restates nor removes is kept
            keys = {c.get("key") for c in conds}
            conds = [c for c in (base.get("conditions") or []) if c.get("key") not in keys] + conds
        seen, uniq = set(), []
        for c in conds:
            key = (c.get("key"), c.get("op"), c.get("value"), c.get("minutes"), c.get("direction"))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        f["conditions"] = uniq
        SM._vague_sizes(SM.Reading(text, f, []), [c for c in uniq if c.get("size") == "vague"])
    if raw.get("out"):
        o = _outcome(raw["out"], text, problems, f.get("conditions") or [])
        if o is not None:
            f["outcome"] = o
            quotes.append(o["text"])
    for role in ("start", "end"):
        sp = raw.get(role)
        if sp and sp.get("q"):
            a = _anchor(sp["q"], text, msg_tz, role, problems)
            quotes.append(sp["q"])
            if a.get("scope"):
                f["scope"] = a["scope"]
            if a.get("at"):
                if role == "start":
                    f["start"] = a["at"]
                    f["start_stated"] = a.get("stated") or sp["q"]
                elif a["at"] == "next_session":
                    f["next_session"] = True
                else:
                    f["end"] = a["at"]
                    if a.get("at_end"):
                        f["at_end"] = True
                if a.get("assumed_tz"):
                    f["assumptions"].append({"slot": "time", "text": f"\"{sp['q']}\" had no time zone; it was read as {a['assumed_tz']} time = {a['at']} New York."})
    if raw.get("period") and raw["period"].get("q"):
        q = raw["period"]["q"]
        if find_quote(text, q):
            a_, b_, note = NL.find_period(q)
            if a_ is None and b_ is None:
                problems.append(f"could not read the period \"{q}\"")
            else:
                f["period"] = [a_.isoformat() if a_ else None, b_.isoformat() if b_ else None]
                quotes.append(q)
        else:
            _drop(q)
    kind = f.get("kind")
    oc = (f.get("outcome") or {}).get("key")
    if kind == "movement" and oc in ("close_vs", "touch", "touch_level", "recover", "reverse", "continue", "event"):
        f["kind"] = kind = "frequency"  # an outcome to count makes it a frequency question ("does it usually recover")
    if kind == "agreement" and oc in ("close_vs", "touch", "touch_level") and any(c.get("key") == "first_candle" and c.get("direction") in ("up", "down") for c in f.get("conditions") or []):
        f["kind"] = kind = "frequency"  # "close red after a green first 30 minutes": one candle colour and an outcome
    if kind == "agreement":
        if raw.get("pred"):
            f["predictor"] = raw["pred"]
        f.setdefault("predictor", "first_candle")
        if raw.get("candle") and raw["candle"].get("q"):
            q = raw["candle"]["q"]
            if find_quote(text, q):
                quotes.append(q)
                mm = _candle_minutes(q)
                f["candle"], f["candle_stated"] = ([mm], True) if mm else ([5, 15, 30], False)
            else:
                _drop(q)
        if f.get("predictor") == "first_candle" and not f.get("candle"):
            f["candle"], f["candle_stated"] = [5, 15, 30], False
            f["assumptions"].append({"slot": "candle", "text": "The candle's length was not stated, so 5-, 15- and 30-minute candles are all measured."})
        if raw.get("scope"):
            f["scope"] = raw["scope"]
        if f.get("predictor") == "first_candle" and not f.get("scope"):
            f["scope"] = "both"
            f["assumptions"].append({"slot": "scope", "text": "\"The day\" is measured twice: the whole session (which contains the candle) and the rest of it after the candle closes."})
        f.pop("outcome", None)
    if kind == "timing":
        f["measure"] = raw.get("measure") if raw.get("measure") in ("high_time", "low_time", "abs_move") else f.get("measure") or "abs_move"
        b = re.search(r"\b(5|10|15|30|60)[- ]?min(?:ute)?s?\b|\bhalf[- ]?hour\b|\b(?:hour|hourly|last hour|first hour)\b", text.lower())
        f["bucket"] = (int(b.group(1)) if b and b.group(1) else 30 if b and "half" in b.group(0) else 60 if b else f.get("bucket") or 30)
        f.pop("outcome", None)
    if kind == "breakdown":
        f["by"] = raw.get("by") or f.get("by") or "weekday"
        f["measure"] = raw.get("measure") if raw.get("measure") in ("open_to_close", "close_to_close", "close>open") else f.get("measure") or "open_to_close"
        if raw.get("focus"):
            f["focus"] = raw["focus"]
        wk = [c for c in f.get("conditions") or [] if c.get("key") == "weekday"]
        if wk and not f.get("focus") and len(wk) == 1:
            f["focus"] = wk[0]["value"]
        f["conditions"] = [c for c in f.get("conditions") or [] if c.get("key") != "weekday"]
        f.pop("outcome", None)
    if kind in ("frequency", "movement"):
        o = f.get("outcome") or {}
        if kind == "movement" and not o:
            f["outcome"] = {"key": "end_return"}
        if kind == "frequency" and not o:
            problems.append("nothing to count")
        # the first candle as a condition: the observation starts when it has closed
        for c in f.get("conditions") or []:
            if c.get("key") == "first_candle":
                if not c.get("minutes"):
                    c["minutes_options"] = [5, 15, 30]
                    f["assumptions"].append({"slot": "candle", "text": "The candle's length was not stated, so 5-, 15- and 30-minute candles are all measured."})
                if c.get("direction") is None:
                    c["direction"] = "any"
                if f.get("start") in (None, "open") and c.get("minutes"):
                    f["start"] = SM._hhmm(570 + c["minutes"])
        o = f.get("outcome") or {}
        if o.get("key") in ("recover", "reverse", "continue"):
            fc = [c for c in f.get("conditions") or [] if c.get("key") == "first_candle"]
            if o["key"] == "reverse" and fc:
                f["outcome"] = {"key": "close_vs", "ref": "start", "direction": "opposite", "text": o["text"]}
                f["assumptions"].append({"slot": "outcome", "text": f"\"{o['text']}\" is read as: from the candle's close to the session's close, the price moves the other way from the candle."})
            elif not f.get("conditions"):
                problems.append(f"\"{o['text']}\" after what?")
            else:
                fall = next((c for c in f["conditions"] if c.get("op") in ("<", "<=") or c.get("direction") == "down"), None)
                dd = ("up" if fall else "down") if o["key"] != "continue" else ("down" if fall else "up")
                opts = SM.readings_of(f["conditions"], dd)
                f["outcome"] = {**opts[0][1], "text": o["text"]}
                f["outcome_options"] = [{"label": lab, "outcome": {**oc, "text": o["text"]}} for lab, oc in opts]
                f["assumptions"].append({"slot": "outcome", "text": f"\"{o['text']}\" can mean more than one thing, so each is measured: " + "; ".join(lab for lab, _ in opts) + "."})
        o = f.get("outcome") or {}
        if o.get("key") == "close_vs" and o.get("ref") == "auto":
            o["ref"] = "prev_close"
        if f.get("at_end") and o.get("key") == "touch":
            o["at_end"] = True
        f.setdefault("start", "signal" if any(c.get("key") in ("from_open", "change", "vwap", "vwap_cross") for c in f.get("conditions") or []) else "open")
        f.setdefault("end", "close")
        if o.get("key") == "touch" and not (raw.get("start") or {}).get("q"):
            f["assumptions"].append({"slot": "start", "text": f"The {abs(o['value']):g}% is measured from " + ("the open of the session" if f.get("start") in (None, "open") else "where the price was when the condition was met") + "."})
    f.setdefault("period", [None, None])
    return f, problems, quotes


# ------------------------------------------------------------------------------------------------------------ coverage
MEANINGFUL = ("symbol", "event", "outcome", "clock", "period", "option")


def coverage(text: str, quotes: list[str], known: list[str], also: list[str] | None = None) -> list[str]:
    """What in the message carries meaning and lies in no quoted part: the clauses the reading did not place."""
    spans = [s for s in (find_quote(text, q) for q in quotes + list(also or [])) if s]
    low = text.lower()
    for q in quotes + list(also or []):  # every occurrence of a quoted phrase (a ticker named twice is one ticker)
        ql = (q or "").lower().strip()
        if ql:
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(ql), low)]
    n, ms = SM.tag(text, known)
    connective = SM.FILLER | {"after", "when", "whenever", "if", "once", "following", "days", "day", "sessions", "session", "that", "its", "it", "does", "do", "did", "is", "was"}

    def inside(oa: int, ob: int) -> bool:
        # every word of the mention that carries meaning lies inside some quoted part (connectives like "after it" need not)
        words = [(m.start() + oa, m.end() + oa, m.group(0).lower()) for m in re.finditer(r"[\w%$.+\-:']+", text[oa:ob])]
        content = [(a, b) for a, b, w in words if w not in connective]
        if not content:
            return True
        return all(any(sa <= a and b <= sb + 1 for sa, sb in spans) for a, b in content)
    left = []
    for m in ms:
        if m.kind not in MEANINGFUL:
            continue
        oa, ob = n.span(m.a, m.b)
        if not inside(oa, ob):
            left.append(m.text)
    for w in SM.unread_words(n, ms):
        oa, ob = w["span"]
        if not inside(oa, ob) and not re.fullmatch(r"\d+(?:\.\d+)?", w["norm"]):
            left.append(w["text"])
    return list(dict.fromkeys(x for x in left if x.strip()))


# ---------------------------------------------------------------------------------------------------------- strategies
def _money(q: str) -> float | None:
    """'$500', '10k', '$25,000', '2.5k', '500 dollars' -> dollars."""
    low = q.lower().replace(",", "")
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(k|thousand|grand)?\b", low)
    if not m:
        return None
    v = float(m.group(1))
    return v * 1000 if m.group(2) else v


def _pct_of(q: str) -> float | None:
    nums = _numbers(q)
    return nums[0] if nums else None


def strategy_draft(raw: dict, text: str, known: list[str], default_symbol: str, base=None):
    """(draft, problems, quotes) from the model's reading of a strategy, or of a change to `base` (a Draft)."""
    from strata.ai.draft import Draft, DraftCondition, DraftSizing, DraftStrike
    problems: list[str] = []
    quotes: list[str] = []
    msg_tz = next((NL.TZ_OFFSET[k] for k in ("pacific", "eastern", "central", "mountain") if k in text.lower()), None)
    d = base.model_copy(deep=True) if base is not None else Draft(kind="options", title=text.strip()[:90], symbol=default_symbol, symbol_stated=False, mode="trades")
    changed: list[str] = []
    sym = raw.get("symbol")
    if sym:
        v = _ticker_of(sym)
        if v and find_quote(text, sym.get("q") or v):
            quotes.append(sym.get("q") or v)
            if v not in [s.upper() for s in known]:
                problems.append(f"no price history for {v} is stored")
            d.symbol, d.symbol_stated = v, True
            changed.append("symbol")
        else:
            _drop(sym.get("q") or v)
    st = raw.get("strat") or {}
    if st:
        q = st.get("q") or ""
        if q and find_quote(text, q):
            quotes.append(q)
        low = (q + " " + text).lower()
        struct = st.get("struct")
        calls, puts = bool(re.search(r"\bcalls?\b", low)), bool(re.search(r"\bputs?\b", low))
        if struct in ("long_call", "short_call") and puts and not calls:
            struct = struct.replace("call", "put")
        if struct in ("long_put", "short_put") and calls and not puts:
            struct = struct.replace("put", "call")
        tl = text.lower()
        sold = re.search(r"\b(?:sell|selling|sold|short|shorting|write|writing)\s+(?:\w+\s+){0,4}(?:calls?|puts?|straddles?|strangles?)\b", tl)
        bought = re.search(r"\b(?:buy|buying|bought|long|get|grab)\s+(?:\w+\s+){0,4}(?:calls?|puts?|straddles?|strangles?)\b", tl)
        if struct in ("short_call", "short_put", "short_straddle", "short_strangle") and not sold:
            struct = struct.replace("short_", "long_")  # options are bought unless the words sell or short them ("puts on spy")
        elif struct in ("long_call", "long_put", "long_straddle", "long_strangle") and sold and not bought:
            struct = struct.replace("long_", "short_")
        if struct and (base is None or struct != d.structure) and (base is None or re.search(r"\b(?:call|put|straddle|strangle|condor|butterfly|spread|sell|short|buy)", text.lower())):
            d.structure = struct
            changed.append("structure")
        for fld, key in (("dte", "dte"),):
            sp = st.get(key)
            if sp and sp.get("q"):
                qq = sp["q"].lower()
                if not find_quote(text, sp["q"]):
                    _drop(sp["q"])
                    continue
                quotes.append(sp["q"])
                m = re.search(r"(\d+)\s*-?\s*dte|(\d+)\s*(?:day|days)\s*(?:to expiration|out|expir)", qq)
                if m:
                    d.dte = int(m.group(1) or m.group(2))
                elif re.search(r"0\s*dte|same[- ]day|expir\w* (?:today|that day)|zero[- ]day", qq):
                    d.dte = 0
                elif re.search(r"weekl", qq):
                    d.dte = 7
                elif re.search(r"tomorrow|next[- ]day", qq):
                    d.dte = 1
                else:
                    problems.append(f"could not read the expiration in \"{sp['q']}\"")
                    continue
                changed.append("dte")
        sk = st.get("strike")
        if sk and sk.get("m"):
            qq = (sk.get("q") or "").lower()
            n = _pct_of(qq) if sk["m"] in ("otm", "itm", "delta") else None
            if sk["m"] == "atm":
                d.strike = DraftStrike(method="atm", value=0.0)
            elif sk["m"] in ("otm", "itm") and n is not None and re.search(r"%|percent|pct", qq):
                d.strike = DraftStrike(method="pct_otm", value=n if sk["m"] == "otm" else -n)
            elif sk["m"] == "delta" and n is not None:
                d.strike = DraftStrike(method="delta", value=n / 100 if n > 1 else n)
            elif sk["m"] != "atm":
                problems.append(f"could not read the strike in \"{sk.get('q')}\"")
            if sk.get("q") and find_quote(text, sk["q"]):
                quotes.append(sk["q"])
            if base is None or sk["m"] != "atm":
                changed.append("strike")
        for fld, key in (("entry_time", "entry"), ("exit_time", "exit")):
            sp = st.get(key)
            if not (sp and sp.get("q")):
                continue
            qq = sp["q"]
            if not find_quote(text, qq):
                _drop(qq)
                continue
            quotes.append(qq)
            t = _clock(qq, msg_tz)
            if t is not None and t["et"] not in ("open", "close"):
                mins = int(t["et"][:2]) * 60 + int(t["et"][3:])
                if not (570 <= mins <= 960):
                    problems.append(f"\"{qq}\" is {t['et']} New York time, outside the session")
                setattr(d, fld, "open" if (t["et"] == "09:30" and fld == "entry_time") else t["et"])
                setattr(d, fld + "_stated", t.get("stated"))
                if t.get("assumed_tz"):
                    d.tz_assumed = t["assumed_tz"]
            elif re.search(r"\bopen|at the bell|opening bell|every morning", qq.lower()):
                setattr(d, fld, "open" if fld == "entry_time" else "09:30")
            elif re.search(r"\bclose\b|end of (?:the )?day|eod", qq.lower()):
                setattr(d, fld, "close")
            elif base is not None:
                from strata.ai import converse as CV
                sh = CV.relative_shift(text, base)
                if sh and sh[1] and sh[0] == fld:
                    setattr(d, fld, sh[1])
                    setattr(d, fld + "_stated", None)
                else:
                    problems.append(f"could not tell the time in \"{qq}\"")
                    continue
            else:
                problems.append(f"could not tell the time in \"{qq}\"")
                continue
            changed.append(fld)
        sp = st.get("hold")
        if sp and sp.get("q") and find_quote(text, sp["q"]):
            quotes.append(sp["q"])
            qq = sp["q"].lower()
            m = re.search(r"(\d+)\s*(min|minute|hour|hr)", qq)
            m2 = re.search(r"(\d+|a|one|two|three|four|five)\s+(?:trading\s+)?(day|session)s?", qq)
            if re.search(r"expir", qq):
                d.hold_to_expiration = True
            elif m:
                d.hold_minutes = int(m.group(1)) * (60 if m.group(2).startswith(("hour", "hr")) else 1)
            elif m2:
                wn = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                d.hold_sessions = int(m2.group(1)) if m2.group(1).isdigit() else wn[m2.group(1)]
            else:
                problems.append(f"could not read how long to hold from \"{sp['q']}\"")
            changed.append("hold")
        for fld, key in (("premium_target_pct", "ptarget"), ("premium_stop_pct", "pstop"), ("underlying_target_pct", "utarget"), ("underlying_stop_pct", "ustop")):
            sp = st.get(key)
            if sp and sp.get("q"):
                if not find_quote(text, sp["q"]):
                    _drop(sp["q"])
                    continue
                n = _pct_of(sp["q"])
                if n is None:
                    problems.append(f"no percentage in \"{sp['q']}\"")
                    continue
                quotes.append(sp["q"])
                setattr(d, fld, abs(n))
                changed.append(fld)
        sp = st.get("entries")
        if sp and sp.get("q") and find_quote(text, sp["q"]):
            n = _numbers(sp["q"])
            if n:
                d.max_entries = int(n[0])
                quotes.append(sp["q"])
                changed.append("max_entries")
        if st.get("prices"):
            d.mode = {"trades": "trades", "quotes": "historical", "modeled": "modeled"}[st["prices"]]
    if raw.get("conds") is not None:
        conds, wk = [], []
        for c in raw["conds"]:
            cc = _cond(c, text, problems, msg_tz)
            if cc is None:
                continue
            quotes.append(cc["text"])
            if cc["key"] == "weekday":
                wk.append(cc["value"])
                continue
            if cc["key"] == "first_candle":
                if not cc.get("minutes") or cc.get("direction") not in ("up", "down"):
                    problems.append(f"the candle in \"{cc['text']}\" needs a length and a colour")
                    continue
                if d.entry_time in (None, "open"):
                    d.entry_time = SM._hhmm(570 + cc["minutes"])
            if cc.get("size") == "vague":
                problems.append(f"\"{cc['text']}\" gives no size")
            e = SM.cond_expr(cc)
            if e is None:
                problems.append(f"\"{cc['text']}\" cannot be used as an entry condition")
                continue
            conds.append(DraftCondition(expr=e[0], text=e[1]))
        d.conditions = conds
        d.weekdays = wk or (None if base is None else d.weekdays)
        changed.append("conditions")
    sz = raw.get("size")
    if sz and sz.get("mode"):
        q = sz.get("q") or ""
        if not find_quote(text, q):
            _drop(q)
        else:
            quotes.append(q)
            mode = sz["mode"]
            prev = d.sizing.model_copy() if d.sizing else DraftSizing(mode="contracts", value=float(d.quantity or 1))
            if mode == "contracts":
                n = _numbers(q) or [1.0]
                d.quantity = int(n[0])
                prev = prev.model_copy(update={"mode": "contracts", "value": float(int(n[0]))})
            elif mode in ("dollars", "risk_dollars"):
                v = _money(q)
                if v is None:
                    problems.append(f"no amount in \"{q}\"")
                else:
                    prev = prev.model_copy(update={"mode": mode, "value": v})
            else:
                v = _pct_of(q)
                if v is None:
                    problems.append(f"no percentage in \"{q}\"")
                else:
                    prev = prev.model_copy(update={"mode": mode, "value": v})
            prev.text = q
            d.sizing = prev
            changed.append("sizing")
    if sz:
        cap = sz.get("capital")
        if cap and cap.get("q") and find_quote(text, cap["q"]):
            v = _money(cap["q"])
            if v:
                d.sizing = (d.sizing or DraftSizing(mode="contracts", value=float(d.quantity or 1))).model_copy(update={"capital": v})
                quotes.append(cap["q"])
                changed.append("sizing")
        if sz.get("reinvest") is not None and re.search(r"reinvest|compound|roll (?:the )?profits|grow the account|re-invest", text.lower()):
            d.sizing = (d.sizing or DraftSizing(mode="contracts", value=float(d.quantity or 1))).model_copy(update={"reinvest": bool(sz["reinvest"])})
            changed.append("sizing")
        ds = sz.get("daily_stop")
        if ds and ds.get("q") and find_quote(text, ds["q"]):
            v = _money(ds["q"])
            if v:
                d.sizing = (d.sizing or DraftSizing(mode="contracts", value=float(d.quantity or 1))).model_copy(update={"daily_loss_stop": v})
                quotes.append(ds["q"])
                changed.append("sizing")
    for x in raw.get("drop") or []:
        fld = {"ptarget": "premium_target_pct", "pstop": "premium_stop_pct", "utarget": "underlying_target_pct", "ustop": "underlying_stop_pct", "exit": "exit_time"}.get(x)
        if fld:
            setattr(d, fld, None)
            changed.append(fld)
        elif x == "hold":
            d.hold_minutes = d.hold_sessions = None
            d.hold_to_expiration = False
            changed.append("hold")
        elif x == "conds":
            d.conditions, d.weekdays = [], None
            changed.append("conditions")
        elif x == "size":
            d.sizing = None
            changed.append("sizing")
        elif x == "daily_stop" and d.sizing:
            d.sizing = d.sizing.model_copy(update={"daily_loss_stop": None})
            changed.append("sizing")
    if base is None and not d.structure:
        problems.append("no option position (calls, puts, a spread ...) was named")
    return d, problems, quotes, list(dict.fromkeys(changed))
