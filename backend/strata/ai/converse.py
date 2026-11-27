"""A conversation about one investigation: each message edits the part of it the message is about.

    "How often is SPY's first candle right about the rest of the day?"   a new question (semantics.read)
    "Use 5 min."                       the candle's length
    "Actually do QQQ."                 the instrument
    "What about when yesterday was red?"   one more condition
    "Nah, I meant down at least 1%."   the condition just added, corrected
    "Show me the losing days."         the evidence behind the answer, filtered (nothing changes)
    "What if I bought calls instead?"  the question becomes an option strategy
    "Exit an hour later."              a strategy setting, moved relative to what it was
    "Compare both."                    this version against the one before
    "undo"                             the frame before the last change

Every edit returns the new frame and a plain account of what changed ("Changed SPY to QQQ. Everything else stays the
same."), so the change can be read, and undone, before anything is trusted. A message that could change two different
things ("make it 2" when a gap of 1% and a VIX of 20 are both in the question) changes neither: it is one question back,
with one button per reading.
"""
from __future__ import annotations

import copy
import re

from strata.ai import semantics as SM

QUESTION_KINDS = ("frequency", "agreement", "movement", "timing", "breakdown")


def facts(frame: dict | None) -> dict:
    """The parts of a frame a person would name when saying what changed."""
    if not frame:
        return {}
    f = frame
    out = {"kind": f.get("kind"), "symbol": f.get("symbol"), "period": tuple(f.get("period") or (None, None))}
    for c in f.get("conditions") or []:
        out[f"cond:{c.get('key')}"] = SM.describe_condition(c)
    if f.get("outcome"):
        out["outcome"] = SM.describe_outcome(f["outcome"], f.get("start") or "open")
    for k in ("candle", "scope", "start", "end", "by", "measure", "bucket", "predictor"):
        if f.get(k) is not None:
            out[k] = tuple(f[k]) if isinstance(f[k], list) else f[k]
    d = f.get("draft")
    if d:
        for k in ("structure", "dte", "quantity", "entry_time", "exit_time", "premium_target_pct", "premium_stop_pct", "underlying_target_pct", "underlying_stop_pct", "hold_minutes", "hold_sessions", "strike", "sizing"):
            if d.get(k) not in (None, [], {}):
                if k == "sizing":
                    try:
                        from strata.ai.draft import Draft, sizing_short
                        out["strategy:sizing"] = sizing_short(Draft.model_validate(d))
                    except Exception:  # noqa: BLE001
                        out["strategy:sizing"] = str(d.get(k))
                    continue
                out[f"strategy:{k}"] = str(d.get(k)) if not isinstance(d.get(k), (str, int, float)) else d.get(k)
        for c in d.get("conditions") or []:
            out[f"rule:{c.get('expr')}"] = c.get("text") or c.get("expr")
    return out


NAMES = {"symbol": "the ticker", "period": "the period", "outcome": "what is counted", "candle": "the candle length", "scope": "the part of the day compared", "start": "when the measurement starts",
         "end": "when the measurement ends", "by": "the breakdown", "measure": "what is measured", "bucket": "the time buckets", "predictor": "what does the predicting",
         "strategy:structure": "the position", "strategy:dte": "the expiration", "strategy:quantity": "the number of contracts", "strategy:entry_time": "the entry time", "strategy:exit_time": "the exit time",
         "strategy:premium_target_pct": "the profit target on the option", "strategy:premium_stop_pct": "the stop on the option", "strategy:underlying_target_pct": "the target on the underlying",
         "strategy:underlying_stop_pct": "the stop on the underlying", "strategy:hold_minutes": "the holding time", "strategy:hold_sessions": "the holding period", "strategy:strike": "the strike",
         "strategy:sizing": "the size of each trade"}
STRUCT = {"long_call": "buying calls", "long_put": "buying puts", "short_call": "selling calls", "short_put": "selling puts"}


def _val(k: str, v) -> str:
    if v is None or v == (None, None):
        return "none" if not k == "period" else "all stored history"
    if k == "period":
        return SM._describe_period(list(v)).strip() or "all stored history"
    if k == "candle":
        return (f"{v[0]} minutes" if len(v) == 1 else ", ".join(str(x) for x in v[:-1]) + f" and {v[-1]} minutes (each measured)")
    if k == "strategy:structure":
        return STRUCT.get(v, str(v).replace("_", " "))
    if k == "strategy:dte":
        return "same-day expiration" if v == 0 else f"{v} days to expiration"
    if k == "strategy:strike":
        import ast
        try:
            x = v if isinstance(v, dict) else ast.literal_eval(str(v))
        except Exception:  # noqa: BLE001
            return str(v)
        m_, val = x.get("method"), x.get("value")
        return {"atm": "at the money", "delta": f"delta {val:g}" if val is not None else "by delta", "pct_otm": f"{val:g}% out of the money" if val is not None else "out of the money"}.get(m_, str(m_))
    if k in ("strategy:premium_target_pct", "strategy:premium_stop_pct", "strategy:underlying_target_pct", "strategy:underlying_stop_pct"):
        return f"{v:g}%"
    if k in ("strategy:entry_time", "strategy:exit_time", "start", "end"):
        return {"open": "the open", "close": "the close", "signal": "when the condition is met"}.get(str(v), f"{v} New York time")
    if k == "scope":
        return {"rest": "the rest of the session", "day": "the whole session", "both": "the whole session and the rest of it"}.get(v, v)
    return str(v)


def summary(old: dict | None, new: dict | None) -> list[str]:
    """What changed between two frames, one plain sentence per change. Empty when nothing did."""
    a, b = facts(old), facts(new)
    out = []
    if a.get("kind") != b.get("kind") and a.get("kind") and b.get("kind"):
        # a question that became a strategy (or the other way) is one change, not a list of every slot that moved; what had
        # to be assumed to make it a strategy is said separately (the message's `assumed`)
        return [{"strategy": "Turned the question into an option strategy, with the same conditions." if a.get("kind") in QUESTION_KINDS else "Changed the strategy."}.get(b["kind"], f"Changed the question to a {b['kind']} question.")]
    # a rule whose number changed is one change, not a removal and an addition
    gone = [k for k in a if k.startswith("rule:") and k not in b]
    came = [k for k in b if k.startswith("rule:") and k not in a]
    shape = lambda k: re.sub(r"-?\d+(?:\.\d+)?", "#", k)
    for k in list(gone):
        twin = next((x for x in came if shape(x) == shape(k)), None)
        if twin:
            out.append(_number_change(a[k], b[twin]))
            gone.remove(k)
            came.remove(twin)
            a = {kk: v for kk, v in a.items() if kk != k}
            b = {kk: v for kk, v in b.items() if kk != twin}
    for k in sorted(set(a) | set(b)):
        if k == "kind" or a.get(k) == b.get(k):
            continue
        if k.startswith("cond:") and k in a and k in b:
            out.append(_number_change(a[k], b[k]))
            continue
        if k.startswith("cond:") or k.startswith("rule:"):
            if k not in a:
                out.append(f"Added: {b[k]}.")
            elif k not in b:
                out.append(f"Removed: {a[k]}.")
            else:
                out.append(f"Changed \"{a[k]}\" to \"{b[k]}\".")
            continue
        if k == "symbol":
            out.append(f"Changed {a.get(k)} to {b.get(k)}.")
            continue
        name = NAMES.get(k, k)
        if k not in a or a.get(k) is None:
            out.append(f"Set {name} to {_val(k, b.get(k))}.")
        elif k not in b or b.get(k) is None:
            out.append(f"Removed {name} ({_val(k, a.get(k))}).")
        else:
            out.append(f"Changed {name} from {_val(k, a.get(k))} to {_val(k, b.get(k))}.")
    return out


def _number_change(old: str, new: str) -> str:
    """"the previous session fell 3% or more" -> "... 3.8% or more": "Now: the previous session fell 3.8% or more (was 3%)."."""
    na, nb = re.findall(r"\d+(?:\.\d+)?%?", old), re.findall(r"\d+(?:\.\d+)?%?", new)
    if len(na) == len(nb) and re.sub(r"\d+(?:\.\d+)?", "#", old) == re.sub(r"\d+(?:\.\d+)?", "#", new):
        diff = [(x, y) for x, y in zip(na, nb) if x != y]
        if len(diff) == 1:
            return f"Now: {new} (was {diff[0][0]})."
    return f"Changed \"{old}\" to \"{new}\"."


def summary_text(lines: list[str]) -> str:
    if not lines:
        return "Nothing changed."
    return " ".join(lines) + (" Everything else stays the same." if len(lines) <= 3 else "")


# ------------------------------------------------------------------------------------------------------------ the cues
UNDO = re.compile(r"^\s*(?:undo|go back|revert|never ?mind|scratch that|back to (?:the )?(?:previous|last) (?:one|version))\b", re.I)
COMPARE = re.compile(r"\bcompare\b.*\b(?:both|previous|last|before|old|original|first)\b|\b(?:both|previous|before)\b.*\bcompare\b|^\s*compare (?:them|the two|these)\b|\bside by side\b", re.I)
CONVERT = re.compile(r"\b(?:what if (?:i|we) |if (?:i|we) |could (?:i|we) |(?:i|we)'?d |let'?s |ok,? so |okay,? so |so |now |then )?(?:buy|bought|buying|trade|traded|trading|sell|sold|selling|use|go long|go short)\b.{0,40}?\b(?:calls?|puts?|options?)\b", re.I)
EXPLORE = re.compile(r"^\s*(?:show|list|see|give|open|which)(?: me)?\b.{0,40}\b(?:days?|dates?|trades?|sessions?|times|ones|examples?|cases?)\b|\bthe days (?:it|that) (?:happened|worked|failed)\b", re.I)
RELATIVE = re.compile(r"\b(?P<what>enter(?:ed|ing)?|entry|buy|bought|buying|get in|got in|exit(?:ed|ing)?|sell|sold|selling|close|closed|get out|got out|out|the exit|the entry)\b.{0,20}?\b(?P<n>\d+(?:\.\d+)?|an?|one|half an?)\s*(?P<u>min(?:ute)?s?|hours?|hrs?|m|h)\b\s*(?P<dir>later|earlier|sooner|before|after)\b|"
                      r"\b(?P<what2>enter|exit|sell|buy|close)\s+(?P<dir2>later|earlier)\s+by\s+(?P<n2>\d+)\s*(?P<u2>min|hour)", re.I)
NUMBER_ONLY = re.compile(r"^\s*(?:no+|nah|nope|sorry|actually|oops|i meant|i mean|make it|change it to|set it to|try|use)?[\s,.:!-]*(?:i meant |i mean |to )?(?P<sign>[+-])?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|pct|min(?:ute)?s?|m)?\s*[.!?]?\s*$", re.I)


def _shift(hhmm: str | None, minutes: int) -> str | None:
    if not hhmm or not re.match(r"\d\d:\d\d", str(hhmm)):
        return None
    t = int(hhmm[:2]) * 60 + int(hhmm[3:5]) + minutes
    return f"{t // 60:02d}:{t % 60:02d}"


def _num_value(n: str) -> float:
    n = n.lower().strip()
    return {"a": 1, "an": 1, "one": 1, "half an": 0.5, "half a": 0.5}.get(n, None) or float(n)


# ------------------------------------------------------------------------------------------------------- frame edits
def numeric_slots(frame: dict) -> list[tuple[str, str, int | None]]:
    """The numbers in a frame a bare number could replace: (slot, plain name, index of the condition)."""
    out = []
    for i, c in enumerate(frame.get("conditions") or []):
        if c.get("key") in ("prev_return", "gap", "from_open", "vix", "up_streak", "down_streak") and (c.get("value") or c.get("values")):
            out.append(("condition", SM.describe_condition(c), i))
    o = frame.get("outcome") or {}
    if o.get("key") == "touch":
        out.append(("outcome", f"the {abs(o['value']):g}% move that is counted", None))
    if frame.get("kind") == "agreement" and frame.get("candle") and len(frame["candle"]) == 1:
        out.append(("candle", f"the {frame['candle'][0]}-minute candle", None))
    d = frame.get("draft") or {}
    for k, name in (("premium_target_pct", "the profit target on the option"), ("premium_stop_pct", "the stop on the option"), ("underlying_target_pct", "the target on the underlying"),
                    ("underlying_stop_pct", "the stop on the underlying")):
        if d.get(k):
            out.append(("draft:" + k, f"{name} ({d[k]:g}%)", None))
    return out


def edit_question(text: str, frame: dict, known: list[str], last_slot: str | None = None) -> dict:
    """A follow-up to a question frame. Returns {action: edit|clarify|none, frame, changes, slot}."""
    f = copy.deepcopy(frame)
    n, ms = SM.tag(text, known)
    low = text.lower()
    changed_slot = None

    # a bare number: replaces the one number it can replace, or asks which
    m = NUMBER_ONLY.match(text)
    if m:
        v = float(m.group("num")) * (-1 if m.group("sign") == "-" else 1)
        unit = (m.group("unit") or "").lower()
        slots = numeric_slots(f)
        if unit.startswith("m") and unit != "%" and f.get("kind") == "agreement":
            slots = [s for s in slots if s[0] == "candle"] or [("candle", "the candle length", None)]
        if last_slot and len(slots) > 1:
            pref = [s for s in slots if s[0] == last_slot.split("#")[0] and (s[2] is None or str(s[2]) == (last_slot.split("#") + [""])[1])]
            slots = pref or slots
        if len(slots) == 1:
            return _apply_number(f, slots[0], v, frame)
        if not slots:
            return {"action": "clarify", "text": f"\"{text.strip()}\" is a number, but the current question has no number it could replace.", "options": []}
        shown = f"{abs(v):g}{'%' if unit in ('%', 'percent', 'pct') or slots[0][0] != 'candle' else ''}"
        return {"action": "clarify", "text": f"\"{shown}\" could replace more than one thing in the current reading: {' or '.join(name for _, name, _ in slots)}. Nothing was changed. Which should be {shown}?",
                "options": [{"label": name, "patch": {"slot": slot, "index": idx, "value": v}} for slot, name, idx in slots]}

    # the instrument
    syms = [x for x in ms if x.kind == "symbol"]
    for x in syms:
        x.used = True
        if x.value["stored"] and x.value["symbol"] != f.get("symbol"):
            f["symbol"], f["symbols"], f["symbol_stated"] = x.value["symbol"], [x.value["symbol"]], True
            changed_slot = changed_slot or "symbol"
        elif not x.value["stored"]:
            return {"action": "clarify", "text": f"No price history for {x.value['symbol']} is stored, so the question cannot be moved to it. Stored: {', '.join(known)}.", "options": []}
    # the period
    for x in ms:
        if x.kind == "period":
            x.used = True
            f["period"] = [x.value["start"], x.value["end"]]
            changed_slot = changed_slot or "period"
    if re.search(r"\b(?:all|whole|full|entire) (?:history|period|data)\b|\bno (?:date|period) (?:limit|filter)\b", low):
        f["period"] = [None, None]
        changed_slot = changed_slot or "period"

    # the candle's length ("use 5 min", "try 30 minutes instead", "15m")
    cm = re.search(r"\b(\d+)\s*-?\s*(?:min(?:ute)?s?|m)\b(?!\s*(?:later|earlier|after|before))", low)
    if cm and f.get("kind") == "agreement" and not re.search(r"\b(?:first|in the|within)\s+\d+\s*min", low.replace("-", " ")) or (cm and f.get("kind") == "agreement" and re.search(r"\b(?:use|try|make it|switch to|instead|candle|bar)\b", low)):
        f["candle"], f["candle_stated"] = [int(cm.group(1))], True
        f["assumptions"] = [a for a in f.get("assumptions") or [] if a.get("slot") != "candle"]
        changed_slot = changed_slot or "candle"
        for x in ms:
            if x.kind in ("anchor", "event") and x.value.get("minutes") == int(cm.group(1)):
                x.used = True
    elif cm and f.get("kind") in ("frequency", "movement") and any(c.get("key") == "first_candle" for c in f.get("conditions") or []) and re.search(r"\b(?:use|try|make it|switch to|instead|candle|bar)\b", low):
        for c in f["conditions"]:
            if c.get("key") == "first_candle":
                c["minutes"] = int(cm.group(1))
                c.pop("minutes_options", None)
                if f.get("start") not in (None, "signal"):
                    f["start"] = SM._hhmm(570 + c["minutes"])
        changed_slot = changed_slot or "candle"

    # scope of an agreement
    for x in ms:
        if x.kind == "anchor" and x.value.get("anchor") in ("rest", "day") and f.get("kind") == "agreement":
            x.used = True
            f["scope"] = "rest" if x.value["anchor"] == "rest" else "day"
            changed_slot = changed_slot or "scope"

    # conditions: added, replaced (same kind), corrected (the one just edited), or removed
    removing = bool(re.search(r"\b(?:remove|drop|without|no longer|ignore|forget)\b", low))
    events = [x for x in ms if x.kind == "event" and x.value.get("key") not in ("unsupported_event",)]
    correction = bool(re.match(r"^\s*(?:no+|nah|nope|sorry|actually|oops|i meant|i mean|not quite|rather|wait)\b", low))
    for x in events:
        x.used = True
        v = dict(x.value)
        if v.get("key") == "first_candle" and f.get("kind") == "agreement":
            if v.get("minutes"):
                f["candle"], f["candle_stated"] = [v["minutes"]], True
                changed_slot = changed_slot or "candle"
            continue
        if v.get("key") == "move":
            # "down at least 1%" after "what about when yesterday was red": the condition being corrected keeps its kind
            target = next((c for c in reversed(f.get("conditions") or []) if c.get("key") in ("prev_return", "gap", "from_open")), None)
            v["key"] = (target or {}).get("key") or "prev_return"
        v["text"] = x.text
        conds = f.setdefault("conditions", [])
        same = [i for i, c in enumerate(conds) if c.get("key") == v["key"]]
        if removing:
            f["conditions"] = [c for c in conds if c.get("key") != v["key"]]
            changed_slot = changed_slot or "condition"
            continue
        if same:
            old = conds[same[-1]]
            restated = (not v.get("value") and old.get("value") and v.get("size") in (None, "unstated") and not correction
                        and (v.get("op") in ("<", "<=")) == (old.get("op") in ("<", "<=")))
            if restated:
                # "what if it gapped down but yesterday was green" after "gaps down more than 1%": the condition is restated, not
                # resized: it keeps its size (until V9 it became "any gap down", said, but not what was meant)
                continue
            conds[same[-1]] = v
            changed_slot = changed_slot or f"condition#{same[-1]}"
        else:
            conds.append(v)
            changed_slot = changed_slot or f"condition#{len(conds) - 1}"
    if removing and not events:
        key = next((k for w, k in (("gap", "gap"), ("yesterday", "prev_return"), ("previous", "prev_return"), ("vix", "vix"), ("vwap", "vwap"), ("monday|tuesday|wednesday|thursday|friday|weekday", "weekday"),
                                    ("candle", "first_candle"), ("streak|in a row", "up_streak")) if re.search(rf"\b(?:{w})", low)), None)
        if key:
            before = len(f.get("conditions") or [])
            f["conditions"] = [c for c in f.get("conditions") or [] if c.get("key") != key and not (key == "up_streak" and c.get("key") == "down_streak")]
            if len(f["conditions"]) != before:
                changed_slot = changed_slot or "condition"

    # what is counted, and when
    outs = [x for x in ms if x.kind == "outcome" and x.value.get("key") in ("close_vs", "touch", "touch_level")]
    if outs and f.get("kind") in ("frequency", "movement"):
        outs[0].used = True
        f["outcome"] = {**outs[0].value, "text": outs[0].text}
        changed_slot = changed_slot or "outcome"
    for x in ms:
        if x.kind == "anchor" and x.value.get("anchor") in ("end_after_open", "end_at", "within") and f.get("kind") in ("frequency", "movement"):
            x.used = True
            f["end"] = SM._hhmm(570 + x.value["minutes"]) if x.value["anchor"] == "end_after_open" else x.value.get("at") if x.value["anchor"] == "end_at" else f"minutes:{x.value['minutes']}"
            changed_slot = changed_slot or "end"
    for x in ms:
        if x.kind == "clock" and f.get("kind") == "agreement":
            x.used = True
            if x.value["et"] == "09:30":
                # "no I meant 6:30a, PST": 6:30 Pacific is the opening bell; the question already starts there
                return {"action": "noop", "text": f"{x.text.strip()} is 09:30 New York time, the opening bell, which is where the candle in this question starts. Nothing changed."}
            return {"action": "clarify", "text": f"{x.text.strip()} is {x.value['et']} New York time. This question is about the first candle of the session, which starts at 09:30; a candle starting later is not available yet.",
                    "options": []}
        if x.kind == "clock" and f.get("kind") in ("frequency", "movement"):
            x.used = True
            pre = n.norm[max(0, x.a - 14):x.a]
            if re.search(r"\b(?:before|by|until|till|til|through|to)\s*$", pre):
                f["end"] = x.value["et"]
                changed_slot = changed_slot or "end"
            else:
                f["start"] = "open" if x.value["et"] == "09:30" else x.value["et"]
                f["start_stated"] = x.value.get("stated")  # the words it was read from stay with the value
                changed_slot = changed_slot or "start"
    if not changed_slot:
        unread = [u["text"] for u in SM.unread_words(n, ms)]
        return {"action": "none", "unread": unread}
    return {"action": "edit", "frame": f, "changes": summary(frame, f), "slot": changed_slot, "correction": correction}


def _apply_number(f: dict, slot: tuple, v: float, old: dict) -> dict:
    kind, _name, idx = slot
    if kind == "condition":
        c = f["conditions"][idx]
        c.pop("values", None)
        c.pop("size", None)
        neg = c.get("op") in ("<", "<=")
        c["value"] = -abs(v) if neg and c.get("key") in ("prev_return", "gap", "from_open") else abs(v)
        if c.get("op") in ("<", ">"):
            c["op"] = c["op"] + "="
        c["text"] = SM.describe_condition(c)
        tag = f"condition#{idx}"
    elif kind == "outcome":
        o = f["outcome"]
        o["value"] = abs(v) if o["value"] > 0 else -abs(v)
        tag = "outcome"
    elif kind == "candle":
        f["candle"], f["candle_stated"] = [int(abs(v))], True
        tag = "candle"
    else:
        k = kind.split(":", 1)[1]
        f["draft"][k] = abs(v)
        tag = kind
    return {"action": "edit", "frame": f, "changes": summary(old, f), "slot": tag}


def answer_clarification(text: str, pending: dict) -> dict | None:
    """The reply to a question Strata asked: a button's label, or words that pick one of the options."""
    low = text.lower().strip(" .!?")
    opts = pending.get("options") or []
    for i, o in enumerate(opts):
        lab = o["label"].lower()
        if low == lab or low == str(i + 1) or (len(low) >= 3 and (low in lab or lab.startswith(low))):
            return {"index": i, "option": o}
    # "the gap": a word of one option's label and of no other
    words = [w for w in re.findall(r"[a-z]+", low) if w not in SM.FILLER]
    hits = [i for i, o in enumerate(opts) if words and all(re.search(rf"\b{re.escape(w)}", o["label"].lower()) for w in words)]
    if len(hits) == 1:
        return {"index": hits[0], "option": opts[hits[0]]}
    return None


# ------------------------------------------------------------------------------------------------- turning into trades
def to_strategy(frame: dict, text: str) -> dict:
    """ "What if I bought calls instead?": the question's condition traded with options. Returns {draft (dict), assumptions}."""
    from strata.ai.draft import Draft, DraftCondition, DraftStrike
    low = text.lower()
    right = "put" if re.search(r"\bputs?\b", low) and not re.search(r"\bcalls?\b", low) else "call"
    structure = "long_put" if right == "put" else "long_call"
    if re.search(r"\b(?:sell|sold|selling|short)\b", low):
        structure = "short_put" if right == "put" else "short_call"
    assumptions = []
    conds, entry, exit_ = [], None, None
    k = frame.get("kind")
    sym = frame.get("symbol") or "SPY"
    if k == "agreement" and frame.get("predictor") == "first_candle":
        mins = (frame.get("candle") or [5])[0]
        if len(frame.get("candle") or []) > 1:
            assumptions.append(f"The question measured several candle lengths; the {mins}-minute candle is used. Say \"use 15 min\" to change it.")
        # calls when the candle closed up, puts when it closed down: the bet the question was about
        want = "down" if right == "put" else "up"
        if re.search(r"\b(?:red|down|bearish)\b", low):
            want = "down"
        elif re.search(r"\b(?:green|up|bullish)\b", low):
            want = "up"
        conds.append(DraftCondition(expr="from_open_pct > 0" if want == "up" else "from_open_pct < 0", text=f"the first {mins}-minute candle closed {want}"))
        entry = SM._hhmm(570 + mins)
        exit_ = "15:55"
        assumptions.append(f"Bought when the first {mins}-minute candle closes ({entry} New York time), held to 15:55.")
    elif k == "agreement":
        want = "down" if right == "put" else "up"
        conds.append(DraftCondition(expr=f"previous_session.return_pct {'>' if want == 'up' else '<'} 0", text=f"the previous session closed {want}"))
        entry, exit_ = "open", "15:55"
        assumptions.append("Bought at the open, held to 15:55.")
    else:
        st = frame.get("start") or "open"
        entry = None if st == "signal" else st
        en = frame.get("end") or "close"
        exit_ = en if re.match(r"\d\d:\d\d", str(en)) else "15:55"
        if en == "close":
            assumptions.append("Held to 15:55, before the same-day contracts expire.")
    for c in frame.get("conditions") or []:
        if c.get("key") == "weekday":
            continue
        if c.get("key") == "first_candle" and k == "agreement":
            continue
        e = SM.cond_expr(c)
        if e:
            conds.append(DraftCondition(expr=e[0], text=e[1]))
    weekdays = [c["value"] for c in frame.get("conditions") or [] if c.get("key") == "weekday"] or None
    o = frame.get("outcome") or {}
    target = None
    if o.get("key") == "touch" and k in ("frequency", "movement"):
        target = abs(o["value"])
        assumptions.append(f"Sold when {sym} moves {target:g}% in the position's favour, the level the question counted; otherwise at the exit time.")
    dte = 0
    m = re.search(r"\b(\d+)\s*-?\s*dte\b", low)
    if m:
        dte = int(m.group(1))
    else:
        assumptions.append("At-the-money contracts expiring the same day (0DTE), one contract: say \"use 7 DTE\" or \"2 contracts\" to change it.")
    p = frame.get("period") or [None, None]
    d = Draft(kind="options", symbol=sym, symbol_stated=True, structure=structure, dte=dte, strike=DraftStrike(method="atm"), quantity=1, conditions=conds, weekdays=weekdays,
              entry_time=entry, exit_time=exit_, underlying_target_pct=target, title=f"{sym}: {STRUCT.get(structure, structure)} " + (SM._describe_conditions(frame.get('conditions') or []) or "")[:80],
              start=p[0], end=p[1] if len(p) > 1 else None, notes=assumptions)
    return {"draft": d, "assumptions": assumptions}


def relative_shift(text: str, draft) -> tuple | None:
    """ "exit an hour later", "enter 15 minutes later", "make the exit 30 min later": (field, new HH:MM, words) or None."""
    m = RELATIVE.search(text)
    if not m:
        return None
    what = (m.group("what") or m.group("what2") or "").lower()
    n_ = m.group("n") or m.group("n2") or "1"
    u = (m.group("u") or m.group("u2") or "min").lower()
    d_ = (m.group("dir") or m.group("dir2") or "later").lower()
    mins = int(round(_num_value(n_) * (60 if u.startswith("h") else 1)))
    sign = 1 if d_ in ("later", "after") else -1
    field = "entry_time" if what.startswith(("enter", "entry", "buy", "bought", "get in", "got in", "the entry")) else "exit_time"
    cur = getattr(draft, field, None)
    if field == "entry_time" and cur in (None, "open"):
        cur = "09:30"
    if field == "exit_time" and cur in (None, "close"):
        cur = "15:55" if cur == "close" or cur is None else cur
    new = _shift(cur, sign * mins)
    return field, new, cur


# ------------------------------------------------------------------------------------------------- position sizing (V9)
_AMT = r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|K|thousand|grand)?"


def _amount(num: str, k: str | None) -> float:
    v = float(num.replace(",", ""))
    return v * 1000 if k else v


def sizing_from_text(text: str, base=None):
    """Position sizing said in a trader's words, read deterministically (the built-in readers' counterpart of the model
    reader's `size` part). Returns (DraftSizing | None, quantity | None, [the words read]) or None when nothing is said.

      "spend $500 a trade", "put 5% of the account into ...", "risk $300 per trade", "risk 2% of a $25k account",
      "start with 10k", "reinvest(ing) profits", "stop (trading) for the day after losing $200", "buy 3 contracts"
    """
    from strata.ai.draft import DraftSizing
    low = text.lower()
    said = []
    prev = base.sizing.model_copy() if (base is not None and getattr(base, "sizing", None)) else None
    mode = value = None
    qty = None
    m = re.search(r"\b(?:spend(?:ing)?|use|using|put(?:ting)?|allocat\w*|invest\w*|budget(?: of)?)\s+(?:up to |about |around )?" + _AMT + r"\s*(?:dollars?\s*)?(?:of premium\s*)?(?:per|a|each|every|on each|for each)\s+(?:trade|time|position|entry)", low)
    if m:
        mode, value = "dollars", _amount(m.group(1), m.group(2))
        said.append(m.group(0))
    m = re.search(r"\b(?:put(?:ting)?|spend(?:ing)?|use|using|allocat\w*|invest\w*)\s+(\d+(?:\.\d+)?)\s*%\s*of\s+(?:the|my|an?)\s+(?:\$?[\d,.]+k?\s+)?(?:account|equity|portfolio|capital|balance)\b", low)
    if m:
        mode, value = "equity_pct", float(m.group(1))
        said.append(m.group(0))
    m = re.search(r"\brisk(?:ing)?\s+(?:up to |about |around )?\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|grand)?|\brisk(?:ing)?\s+(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|grand)?\s+dollars", low)
    if m:
        mode = "risk_dollars"
        value = _amount(m.group(1) or m.group(3), m.group(2) or m.group(4))
        said.append(m.group(0))
    m = re.search(r"\brisk(?:ing)?\s+(?:up to |about |around )?(\d+(?:\.\d+)?)\s*%(?:\s*of\s+(?:the|my|an?)\s+(?:\$?\s*[\d,.]+\s*k?\s+)?(?:account|equity|portfolio|capital|balance))?", low)
    if m:
        mode, value = "risk_pct", float(m.group(1))
        said.append(m.group(0))
    m = re.search(r"\b(?:buy(?:ing)?|get|trade|use)\s+(\d+)\s+contracts?\b|\b(\d+)\s+contracts?\s+(?:per|a|each|every)\s+(?:trade|time)", low)
    if m:
        qty = int(m.group(1) or m.group(2))
        said.append(m.group(0))
    cap = None
    m = re.search(r"\b(?:start(?:ing)?|begin(?:ning)?|begins)\s+with\s+(?:a\s+)?" + _AMT + r"(?:\s+account)?", low) or re.search(r"\b(?:of\s+)?(?:a|an|my)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|grand)?\s+(?:account|portfolio)\b", low)
    if m:
        cap = _amount(m.group(1), m.group(2))
        said.append(m.group(0))
    reinvest = None
    if re.search(r"\b(?:don'?t|do not|without|no)\s+(?:re-?invest\w*|compound\w*)", low):
        reinvest = False
        said.append("no reinvesting")
    elif re.search(r"\bre-?invest\w*|\bcompound\w*", low):
        reinvest = True
        said.append("reinvest")
    stop = None
    m = re.search(r"\b(?:stop|quit|done|no more trades?|halt)\s+(?:trading\s+)?(?:for the day|for that day|that day|for the session)\s+(?:after|once|when|if)\s+(?:i(?:'m| am)?\s+|it(?:'s| is)?\s+|losses?\s+(?:reach|hit)\s+)?(?:los(?:e|ing|t)|down|lost)?\s*" + _AMT, low)
    if m:
        stop = _amount(m.group(1), m.group(2))
        said.append(m.group(0))
    if not said:
        return None
    if mode is None and qty is None and prev is None and not (cap or stop or reinvest is not None):
        return None
    s = prev or DraftSizing(mode="contracts", value=float(qty or (getattr(base, "quantity", 1) if base is not None else 1)))
    upd = {}
    if mode:
        upd.update(mode=mode, value=float(value))
    elif qty:
        upd.update(mode="contracts", value=float(qty))
    if cap:
        upd["capital"] = cap
    if reinvest is not None:
        upd["reinvest"] = reinvest
    if stop:
        upd["daily_loss_stop"] = stop
    s = s.model_copy(update=upd)
    s.text = "; ".join(said)
    if s.mode == "contracts" and not (s.capital or s.daily_loss_stop or s.reinvest):
        return None, int(s.value), said  # a plain number of contracts is the draft's quantity
    return s, (int(s.value) if s.mode == "contracts" else None), said
