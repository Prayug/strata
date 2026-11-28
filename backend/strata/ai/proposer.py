"""V10: a local language model as a proposer, never as a reader.

When the built-in reader stops on a message (it could not place a part, or does not recognise the request at all), a
language model may offer ONE rewrite of the message in the plain form the built-in reader is known to read. The rewrite

  is checked for content   every number, time, ticker and weekday of the message is in the rewrite as the same kind of
                           thing, and the rewrite contains nothing the message does not (`faithful`)
  is read by the same reader  the built-in reader must read the rewrite completely, with nothing left over (`reads`)
  is shown, not run        it is offered as a choice under the reply ("Closest reading: ..."). Choosing it sends it as the
                           user's own message, through the same gate as anything typed.

So the model adds no way of running something the user did not see: a wrong proposal is a wrong suggestion on screen, in
plain English, that the user has to pick. Whether this helps at all is measured, not assumed
(docs/V10_REPORT.md, "Local models"): the approaches and models compared are in `VARIANTS` and `bench`.
"""
from __future__ import annotations

import json
import re
import time

SYSTEM = """You rewrite a trader's message into Strata's plain form. Strata answers questions about the price history of SPY, QQQ and IWM and backtests option trades on them. You do NOT answer the message. Output ONE line: the same request, complete and literal, in the plain form below.

Rules:
- Keep every number, time and ticker exactly as the trader wrote it. Do not convert time zones. Do not add, drop or change a number. Do not invent anything that was not said.
- Leave out the trader's own commentary (what they guess, how long they have traded, who told them). Keep everything that is part of the request.
- If something essential is missing or could mean two different things, write MISSING(<what>) in its place. Do not guess.
- If the message asks for something Strata cannot do (a forecast, a ticker other than SPY/QQQ/IWM, an indicator such as RSI or a moving average, news or event days such as FOMC, CPI, earnings, bid/ask fills), output exactly: CANNOT: <what and why>."""

QUESTION_FORM = """Plain form of a QUESTION:
  On <TICKER>, [on days when <condition> [and <condition>],] <what is asked> [<window>] [<period>]
  what is asked: how often does it <event>? | how far does it move? | at what time of day does it make its high? | at what time of day does it make its low? | how do the weekdays compare? | does <A> predict <B>?
  conditions: the previous day closed up|down [N% or more] ; it opens N% or more above|below the previous close ; it has closed up|down N days in a row ; it is up|down N% or more from the open at <time> ; it is above|below the previous close at <time> ; the first N-minute candle is green|red ; VIX is above|below N ; it is a Monday|...|Friday ; it is above|below VWAP at <time> ; it trades above the previous day's high [by <time>] ; it trades below the previous day's low [by <time>]
  events: close above|below its open ; close above|below the previous close ; close above|below its <time> price ; close above|below the previous day's high|low ; rise|fall N% or more from the open ; trade back to the previous close ; trade above the previous day's high ; trade below the previous day's low
  window: by <time> | from <time> to the close | the next day | N days later
  period: since YYYY | in YYYY | from YYYY to YYYY"""

TRADE_FORM = """Plain form of a TRADE:
  Buy|Sell <N> <TICKER> <expiration> <strike> <structure> at <time> [on days when <condition> [and <condition>]]. [Take profit at N% of the premium.] [Stop at N% of the premium.] [Exit when <TICKER> has moved N% up|down from entry.] [Exit at <time>.] [Hold to expiration.] [Hold N minutes at most.] [Size: $N of premium per trade | risk $N per trade | risk N% of the account.] [Start with $N.] [Reinvest.] [Stop for the day after losing $N.] [<period>.]
  expiration: 0DTE | N-DTE ; strike: at-the-money | N-delta | N strikes out-of-the-money ; structure: calls | puts | straddles | strangles | call credit spreads N wide | put credit spreads N wide | call debit spreads N wide | put debit spreads N wide | iron condors N wide"""

FOLLOW_FORM = """Plain form of a FOLLOW-UP (when CURRENT is given, say only the change):
  Use <TICKER> instead. | Change <part> to <value>. | Add the condition: <condition>. | Remove the <part>. | Only <period>. | Show the losing trades. | Show the winning trades. | Show those days. | Undo. | Compare with the previous version. | Could this be luck? | Why did the worst trade lose?"""

Q_SHOTS = [
    ("does iwm usually finish green if it opened way down, like more than 1 pct", "On IWM, on days when it opens 1% or more below the previous close, how often does it close above its open?"),
    ("spy red 3 days straight.. next day up or down more often? post 2018", "On SPY, on days when it has closed down 3 days in a row, how often does it close above the previous close the next day since 2018?"),
    ("what time does spy usually top out. been watching it for years and i swear its always before lunch", "On SPY, at what time of day does it make its high?"),
    ("qqq down more than 1% by 11 eastern - does it come back to green by the bell or nah", "On QQQ, on days when it is down 1% or more from the previous close at 11 eastern, how often does it close above the previous close?"),
    ("how often does it bounce", "On MISSING(ticker), on days when MISSING(what it bounces after), how often does it MISSING(what counts as a bounce)?"),
    ("will qqq go up tomorrow", "CANNOT: a forecast of tomorrow; Strata measures what happened in the past."),
]
T_SHOTS = [
    ("bto 3 qqq 0dte 25d puts @ 0940 et, tp 50 sl 30, flat by 1500", "Buy 3 QQQ 0DTE 25-delta puts at 0940 ET. Take profit at 50% of the premium. Stop at 30% of the premium. Exit at 1500 ET."),
    ("sell the iwm 16 delta strangle at 10, 7dte, take it off at half the credit or 3x stop, risk 2% of acct, 50k", "Sell 1 IWM 7-DTE 16-delta strangles at 10. Take profit at 50% of the premium. Stop at 300% of the premium. Size: risk 2% of the account. Start with $50k."),
    ("my buddy says just buy spy calls every morning when it opens red and dump them at noon. 2 contracts. is he right", "Buy 2 SPY calls at the open on days when it opens below the previous close. Exit at noon."),
    ("spy calls on fed days", "CANNOT: Fed days need a calendar of scheduled events, which is not stored."),
]
F_SHOTS = [
    ("CURRENT: Buy 1 SPY 0DTE at-the-money calls at 9:45. Stop at 20% of the premium.\nMESSAGE: lose the stop and do 5 cons", "Remove the stop. Change the size to 5 contracts."),
    ("CURRENT: On QQQ, on days when it opens 1% or more below the previous close, how often does it close above its open?\nMESSAGE: same thing but small caps and only after 2020", "Use IWM instead. Only since 2020."),
]

# The approaches compared (bench): how the task is put to the model.
#   plain     one prompt with every form and all examples; free text out
#   focused   only the form and the examples of the kind of message at hand (a question, a trade, a follow-up): a
#             smaller prompt and a narrower task
#   schema    the focused prompt, with the reply constrained to a JSON schema (kind, rewrite, missing, cannot)
VARIANTS = ("plain", "focused", "schema")
SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["question", "trade", "follow_up", "cannot", "missing"]}, "rewrite": {"type": "string"},
                                           "missing": {"type": "array", "items": {"type": "string"}}, "cannot": {"type": "string"}}, "required": ["kind", "rewrite"]}


def _prompt(text: str, current: str | None, variant: str) -> tuple[str, list[dict]]:
    from strata.ai import intent as I
    trade = I.is_trade(text)
    if variant == "plain":
        forms, shots = [QUESTION_FORM, TRADE_FORM, FOLLOW_FORM], Q_SHOTS[:2] + T_SHOTS[:2] + Q_SHOTS[2:3] + Q_SHOTS[5:6] + Q_SHOTS[4:5] + F_SHOTS[:1]
    elif current:
        forms, shots = [FOLLOW_FORM, TRADE_FORM if trade else QUESTION_FORM], F_SHOTS + (T_SHOTS[:1] if trade else Q_SHOTS[:1])
    elif trade:
        forms, shots = [TRADE_FORM], T_SHOTS
    else:
        forms, shots = [QUESTION_FORM], Q_SHOTS
    system = SYSTEM + "\n\n" + "\n\n".join(forms)
    if variant == "schema":
        system += ('\n\nReply with JSON only: {"kind": "question" | "trade" | "follow_up" | "cannot" | "missing", "rewrite": "<the one line>", "missing": ["<what is missing>", ...], "cannot": "<why>"}. '
                   "Use kind \"missing\" when something essential is not said, and \"cannot\" when Strata cannot do it; \"rewrite\" is then an empty string.")
    msgs: list[dict] = []
    for u, a in shots:
        user = u if u.startswith("CURRENT:") else f"MESSAGE: {u}"
        if variant == "schema":
            kind = "cannot" if a.startswith("CANNOT") else "missing" if "MISSING(" in a else "follow_up" if u.startswith("CURRENT:") else "trade" if re.match(r"(?:Buy|Sell)\b", a) else "question"
            a = json.dumps({"kind": kind, "rewrite": "" if kind in ("cannot", "missing") else a, "missing": re.findall(r"MISSING\(([^)]*)\)", a), "cannot": a[8:] if kind == "cannot" else ""})
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": (f"CURRENT: {current}\n" if current else "") + f"MESSAGE: {text}"})
    return system, msgs


def rewrite(llm, text: str, current: str | None = None, variant: str = "focused") -> dict:
    """The model's one-line rewrite of `text`: {"rewrite", "cannot", "missing", "seconds", "raw"}. Raises LLMError."""
    system, msgs = _prompt(text, current, variant)
    t0 = time.time()
    if variant == "schema":
        raw = llm.json(system, msgs, SCHEMA, max_tokens=320)
        raw = raw if isinstance(raw, dict) else {}
        out = {"rewrite": str(raw.get("rewrite") or "").strip() or None, "cannot": (str(raw.get("cannot") or "").strip() or "not something Strata can do") if raw.get("kind") == "cannot" else None,
               "missing": [str(x) for x in (raw.get("missing") or [])] if raw.get("kind") == "missing" or not raw.get("rewrite") else [], "raw": raw}
    else:
        # one line is asked for: generation stops at the end of it (left to run on, a small model writes a paragraph after
        # the line and the call takes several times as long)
        had = getattr(llm, "stop", None)
        llm.stop = ["\n"]
        try:
            reply = str(llm.text(system, msgs, max_tokens=200))
        finally:
            llm.stop = had
        line = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
        line = re.sub(r"^(?:rewrite|plain form|answer)\s*:\s*", "", line, flags=re.I).strip().strip('"')
        out = {"rewrite": None if line.upper().startswith("CANNOT") or "MISSING(" in line or not line else line, "cannot": line[7:].strip(" :") if line.upper().startswith("CANNOT") else None,
               "missing": re.findall(r"MISSING\(([^)]*)\)", line), "raw": line}
    out["seconds"] = round(time.time() - t0, 2)
    return out


def faithful(message: str, proposal: str, known: list[str] | None = None) -> list[str]:
    """What differs between the typed content of a message and of its rewrite: every time, percentage, dollar amount,
    count, expiration, delta, weekday, ticker and year, by kind and value. An empty list means the rewrite says the same
    things; any entry means the model added, dropped or changed something, and the rewrite is not offered."""
    from strata.ai import intent as I

    def bag(t: str) -> dict:
        out: dict = {}
        for x in I.said(I.tidy(t), known, trade=True):
            typ, v = x["type"], x["value"]
            if typ == "pct" and x.get("about") == "frequency":
                continue
            k = (typ if typ not in ("number", "contracts", "days", "strikes", "width", "dte", "delta", "minutes") else "n", str(v) if isinstance(v, str) else f"{float(v):g}")
            if typ == "clock":
                k = ("clock", f"{float(v):g}")
            out[k] = out.get(k, 0) + 1
        return out
    a, b = bag(message), bag(proposal)
    # a count that is 1 by default ("Buy 1 ...") or the regular session's own times may be written out in the rewrite
    lenient = {("n", "1"), ("clock", "570"), ("clock", "960")}
    return [f"dropped: {k[1]}" for k in a if k not in b and k not in lenient] + [f"added: {k[1]}" for k in b if k not in a and k not in lenient]


def reads(agent, sid: str, proposal: str) -> bool:
    """Does the built-in reader read the rewrite completely: a reading that would run, or wait only because it has several
    parts? Nothing is executed or stored: only the interpretation is asked for."""
    try:
        s, version, current = agent._state(sid)
        r, _, _ = agent.interpret(sid, proposal, current, version, s["state"].get("last_field"))
    except Exception:  # noqa: BLE001
        return False
    act = r.get("action")
    if act == "clarify":
        return bool((r.get("waiting") or {}).get("complex"))
    d = r.get("draft")
    if d is not None and (getattr(d, "unsupported", None) or [x for x in (getattr(d, "review", None) or []) if "several parts" not in x] or any(q.get("answered") is None for q in (d.questions or []))):
        return False
    return act in ("new", "modify", "derive", "investigate", "profile")


def propose(agent, llm, sid: str, text: str, current: str | None = None, variant: str = "focused") -> dict:
    """One checked proposal for a message the built-in reader stopped on: {"offer": str | None, "why": str, "seconds"}."""
    from strata.ai.llm import LLMError
    try:
        out = rewrite(llm, text, current, variant)
    except LLMError as e:
        return {"offer": None, "why": f"the model did not answer ({str(e)[:80]})", "seconds": None}
    known = sorted(agent.store.stored_symbols())
    p = out.get("rewrite")
    if not p:
        return {"offer": None, "why": "the model said it cannot be done" if out.get("cannot") else "the model found something missing", "seconds": out["seconds"], "raw": out.get("raw")}
    if len(p) > 420 or p.strip().lower() == text.strip().lower():
        return {"offer": None, "why": "the rewrite is the message itself", "seconds": out["seconds"], "raw": out.get("raw")}
    diff = faithful(text, p, known) if not current else [d for d in faithful(text, p, known) if d.startswith("added")]
    if diff:
        return {"offer": None, "why": "the rewrite changed the content (" + "; ".join(diff[:3]) + ")", "seconds": out["seconds"], "rewrite": p}
    if not reads(agent, sid, p):
        return {"offer": None, "why": "the built-in reader could not read the rewrite either", "seconds": out["seconds"], "rewrite": p}
    return {"offer": p, "why": "kept the content and reads completely", "seconds": out["seconds"]}
