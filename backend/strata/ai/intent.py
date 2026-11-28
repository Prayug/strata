"""One research intent, whichever reader produced it.

Strata grew several ways of understanding a sentence: a strategy-draft reader, a stateful
(sequence) reader, a market-study reader, a compositional question reader that builds
research graphs, and a table of conversational actions. They overlap, and they used to be
tried in a fixed order: the first one that produced anything won, even when it had left a
number out of its reading. This module is the single place where a sentence becomes ONE
canonical intent, so that:

  - every reader's output is expressed in the same schema and can be compared, shown and
    tested the same way
  - nothing stated may vanish: sizes, clock times and conditions in the sentence are
    checked against the intent, and what found no place is listed as not understood
  - the orchestration is the same for every kind of request:

        understand -> validate the interpretation -> check the data -> plan -> execute
        -> explain -> follow up

`understand` performs the first two stages and returns the intent with its `stages`
record. The later stages are added by the agent as they happen (`stage`).

The schema (plain dict, JSON-safe):

  kind        study | strategy | modify | action | unreadable
  reader      which reader produced it (graph, draft, route) - information, not behaviour
  symbol      the underlying
  conditions  [{key, op, value, when, text}]   key is canonical: prev_return, day_return,
              gap, vix, weekday, from_open, streak, ... (see KEYS); `when` is the instant
              the condition is known: previous_session | open | close | HH:MM
  outcome     studies: {direction, size, measure: touch|close, from: open|close|HH:MM,
              horizon: session | sessions:N}
  strategy    structure, dte, strike {method, value}, width, quantity
  entry       open | close | HH:MM (New York), with the user's own words in entry_stated
  exits       {time, underlying_for, underlying_against, premium_target, premium_stop,
              hold_sessions, to_expiration}
  period      [start, end]
  unsupported what was said and could not be represented
  assumptions what was not said and had to be assumed
"""
from __future__ import annotations

import json
import re

STAGES = ["understand", "validate", "data", "plan", "execute", "explain", "follow_up"]
# canonical condition keys <- the names each reader uses
EXPR_KEYS = {"previous_session.return_pct": "prev_return", "session_return": "day_return", "return_pct": "day_return", "gap_pct": "gap", "vix": "vix", "vix_prev_close": "vix", "from_open_pct": "from_open",
             "consecutive_down_days": "down_streak", "consecutive_red_days": "down_streak", "consecutive_up_days": "up_streak", "consecutive_green_days": "up_streak", "rsi": "rsi", "iv_rank": "iv_rank",
             "iv_percentile": "iv_percentile", "dist_from_vwap_pct": "vwap_dist", "rvol": "rvol", "day_volume_ratio": "volume_ratio", "volume_ratio": "volume_ratio", "range_pct": "range", "open_to_close": "open_to_close",
             "max_decline_from_open_pct": "max_decline_from_open", "max_rally_from_open_pct": "max_rally_from_open", "pullback_from_high_pct": "pullback_from_high", "bounce_from_low_pct": "bounce_from_low"}
FEATURE_KEYS = {"gap": "gap", "vix_prev": "vix", "weekday": "weekday", "down_streak": "down_streak", "up_streak": "up_streak", "from_open": "from_open", "ret_5": "ret_5", "ret_20": "ret_20", "rv21": "rv21",
                "rv_pctile": "rv_pctile", "sma50_dist": "sma50_dist", "sma200_dist": "sma200_dist", "dist_high252": "dist_high252", "vol_ratio": "volume_ratio", "range_1": "range"}
KEYS = sorted(set(EXPR_KEYS.values()) | set(FEATURE_KEYS.values()) | {"prev_return", "day_return", "expr"})
CMP = re.compile(r"^\s*\(?\s*([A-Za-z_][\w.]*(?:\([^)]*\))?)\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)\s*\)?\s*$")
NUM = r"(\d+(?:\.\d+)?)"


def _when(at: str | None, span: str | None = None) -> str:
    if span == "prev":
        return "previous_session"
    return {"open": "open", "close": "close"}.get(at or "", at or "open")


def _cond_from_expr(expr: str, text: str = "") -> list[dict]:
    """A draft condition (rule expression) in canonical form. A conjunction of simple comparisons becomes one entry each; anything else is kept as `expr`."""
    parts = [p for p in re.split(r"\s+and\s+|\s*&&\s*", expr.strip()) if p.strip()]
    out = []
    for p in parts:
        m = CMP.match(p)
        if not m:
            out.append({"key": "expr", "op": None, "value": None, "when": None, "text": text or p, "expr": p})
            continue
        name, op, val = m.group(1), m.group(2), float(m.group(3))
        sym = None
        if "." in name and name.split(".")[0].isupper():  # QQQ.from_open_pct: another symbol's value
            sym, name = name.split(".", 1)
        base = re.sub(r"\(.*\)$", "", name)
        key = "prev_return" if name.replace(" ", "") == "return_pct(1)" else (EXPR_KEYS.get(name) or EXPR_KEYS.get(base) or base)
        if key == "weekday" and op == "==" and 0 <= val <= 4:
            out.append({"key": "weekday", "op": "is", "value": ["Mon", "Tue", "Wed", "Thu", "Fri"][int(val)], "when": "open", "text": text or p, "expr": p})
            continue
        when = "previous_session" if name.startswith("previous_session.") or key in ("vix", "down_streak", "up_streak", "prev_return") else "open" if key == "gap" else None
        out.append({"key": key, "op": op, "value": val, "when": when, "text": text or p, "expr": p, **({"subject": sym} if sym else {})})
    return out


def from_graph(g: dict) -> dict:
    core = g["core"]
    conds = []
    for c in core.get("clauses") or []:
        f = c.get("feature")
        if f == "ret_1":
            key = "prev_return" if (c.get("span") == "prev" or c.get("at") == "open") else "day_return"
        elif f:
            key = FEATURE_KEYS.get(f, f)
            if key == "weekday" and isinstance(c.get("value"), (int, float)) and 0 <= c["value"] <= 4:
                c = {**c, "op": "is", "value": ["Mon", "Tue", "Wed", "Thu", "Fri"][int(c["value"])]}
        elif c.get("expr"):
            m = CMP.match(f"{c['expr']} {c.get('op') or '>'} {c.get('value') if c.get('value') is not None else 0}")
            base = re.sub(r"\(.*\)$", "", m.group(1)) if m else ""
            key = {"consecutive_red_days": "red_streak", "consecutive_bearish_candles": "red_streak", "consecutive_green_days": "green_streak"}.get(base) or EXPR_KEYS.get(base) or "expr"
        else:
            key = "expr"
        conds.append({"key": key, "op": c.get("op"), "value": c.get("value"), "when": _when(c.get("at"), c.get("span")), "text": c.get("label") or c.get("source") or "", "source": c.get("source"),
                      **({"expr": c["expr"]} if c.get("expr") else {}), **({"subject": c["subject"]} if c.get("subject") and c.get("subject") != core.get("symbol") else {}), **({"hi": c["hi"]} if c.get("hi") is not None else {})})
    oc = core.get("outcome") or {}
    pa = oc.get("params") or {}
    touch = oc.get("op") == "target_stop"
    size = pa.get("target") if touch else pa.get("threshold")
    outcome = {"direction": pa.get("direction"), "size": float(size) if size else None, "measure": "touch" if touch else "close", "from": pa.get("entry"), "horizon": pa.get("horizon"), **({"stop": float(pa["stop"])} if pa.get("stop") else {})}
    sweeps = g.get("sweeps") or []
    syms = next((s["values"] for s in sweeps if s.get("param") == "symbol"), None)
    return {"kind": "study", "reader": "graph", "symbol": core.get("symbol"), "conditions": conds, "outcome": outcome, "period": [core.get("start"), core.get("end")], "controls": core.get("controls") or [],
            "analyses": g.get("analyses") or [], "compare_symbols": syms, "unsupported": list(g.get("unsupported") or []), "assumptions": [a if isinstance(a, str) else a.get("text", "") for a in (g.get("assumptions") or [])],
            "question": g.get("question")}


def from_hypothesis(plan: dict) -> dict:
    """A bounded research plan (a small grid around what was asked) as one intent: the condition with every threshold the plan tries."""
    cfgs = plan.get("configs") or []
    fam = cfgs[0].get("family") if cfgs else None
    same = [c for c in cfgs if c.get("family") == fam]
    conds = []
    for j, pr in enumerate((same[0].get("preds") or []) if same else []):
        f = pr.get("feature")
        vals = sorted({c["preds"][j]["value"] for c in same if len(c.get("preds") or []) > j and isinstance(c["preds"][j].get("value"), (int, float))})
        conds.append({"key": "prev_return" if f == "ret_1" else FEATURE_KEYS.get(f, f), "op": pr.get("op"), "value": pr.get("value"), "values": vals, "when": "previous_session" if f == "ret_1" else "open", "text": same[0].get("label", "")})
    oc = (same[0].get("outcome") or {}) if same else {}
    outcome = {"direction": oc.get("direction"), "size": float(oc["target"]) if oc.get("target") else None, "measure": oc.get("measure"), "from": oc.get("entry"), "horizon": oc.get("horizon")}
    return {"kind": "study", "reader": "hypothesis", "symbol": plan.get("symbol"), "conditions": conds, "outcome": outcome, "period": [None, None], "grid": True, "configurations": len(cfgs),
            "unsupported": [], "assumptions": list(plan.get("notes") or [])[:6], "question": plan.get("question")}


def from_draft(d, compiled: dict | None = None) -> dict:
    """A strategy or study draft in canonical form. `d` is a strata.ai.draft.Draft."""
    conds = [c for dc in d.conditions for c in _cond_from_expr(dc.expr, dc.text)]
    if d.weekdays:
        conds.append({"key": "weekday", "op": "in", "value": list(d.weekdays), "when": "open", "text": "on " + ", ".join(d.weekdays)})
    base = {"reader": "draft", "symbol": d.symbol, "conditions": conds, "entry": d.entry_time, "entry_stated": d.entry_time_stated, "tz_assumed": d.tz_assumed, "unsupported": list(d.unsupported or []),
            "questions": [q.get("text") or q.get("question") for q in (d.questions or []) if q.get("answered") is None],
            "assumptions": [f"{a['field']}: {a.get('value')}" for a in ((compiled or {}).get("assumptions") or [])], "period": [None, None], "sequence": [{"name": s.name, "expr": s.expr, "within_minutes": s.within_minutes} for s in d.sequence],
            "adds": [{"expr": a.expr, "quantity": a.quantity, "max_adds": a.max_adds} for a in d.adds], "exit_conditions": [c.expr for c in d.exit_conditions]}
    for k in ("start", "end"):
        v = getattr(d, k, None)
        if v:
            base["period"][0 if k == "start" else 1] = str(v)
    if d.kind == "study":
        st = d.study
        outcome = None
        if st is not None:
            hz = "session" if st.horizon_kind == "session" else f"sessions:{st.horizon_sessions}" if st.horizon_kind == "sessions" else f"time:{st.horizon_time}"
            tv = st.target.value if st.target is not None else None
            outcome = {"direction": st.direction or (None if not tv else "up" if tv > 0 else "down"), "size": abs(tv) if tv and st.target.kind == "pct" else None, "measure": st.measure,
                       "from": d.entry_time or "open", "horizon": hz, "reference": st.reference, **({"stop": abs(st.stop.value)} if st.stop else {})}
        return {"kind": "study", **base, "outcome": outcome}
    return {"kind": "strategy", **base, "structure": d.structure, "dte": d.dte, "strike": {"method": d.strike.method, "value": d.strike.value} if d.strike else None, "width": d.width, "quantity": d.quantity,
            "exits": {"time": d.exit_time, "time_stated": d.exit_time_stated, "underlying_for": d.underlying_target_pct, "underlying_against": d.underlying_stop_pct, "premium_target": d.premium_target_pct, "premium_stop": d.premium_stop_pct,
                      "hold_sessions": d.hold_sessions, "minutes": d.hold_minutes, "to_expiration": bool(d.hold_to_expiration), "weekday": d.exit_weekday}, "mode": getattr(d, "mode", None), "provider": d.provider,
            "max_entries": d.max_entries, "partials": [{"premium_pct": p.premium_pct, "underlying_pct": p.underlying_pct, "fraction": p.fraction} for p in d.partials]}


# ----------------------------------------------------------------------------- nothing stated may vanish
CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b|\b(\d{1,2}):(\d{2})\b", re.I)


def _numbers_in(intent: dict) -> set[float]:
    out: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            out.add(abs(float(x)))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for m in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", x):
                out.add(abs(float(m)))
    walk({k: v for k, v in intent.items() if k not in ("question", "stages", "text")})
    return out


def validate(intent: dict, text: str) -> list[dict]:
    """Checks of the interpretation against the sentence itself. Each finding is {check, ok, detail}."""
    low = text.lower()
    found = _numbers_in(intent)
    checks = []
    # 1. every percentage in the sentence has a place in the intent
    lost = [p for p in re.findall(rf"{NUM}\s*(?:%|percent\b|pct\b)", low) if abs(float(p)) not in found]
    listed = " ".join(str(u) for u in intent.get("unsupported") or [])
    lost = [p for p in lost if p not in listed]
    checks.append({"check": "Every size in the sentence is in the interpretation", "ok": not lost, "detail": "" if not lost else "No place was found for: " + ", ".join(f"{p}%" for p in dict.fromkeys(lost))})
    # 2. every clock time in the sentence became a time in the intent (in New York time)
    times = [m.group(0) for m in CLOCK.finditer(low)]
    have = " ".join(str(x) for x in (intent.get("entry"), (intent.get("exits") or {}).get("time"), (intent.get("outcome") or {}).get("from"), (intent.get("outcome") or {}).get("horizon"),
                                     " ".join(str(c.get("when")) + " " + str(c.get("expr", "")) + " " + str(c.get("text", "")) for c in intent.get("conditions") or []),
                                     " ".join(str(s) for s in intent.get("sequence") or []), " ".join(str(s) for s in intent.get("exit_conditions") or [])) if x)
    n_have = len(re.findall(r"\b\d{1,2}:\d{2}\b|hhmm\(|\bopen\b|\bclose\b", have))
    unplaced = max(0, len(times) - n_have) if intent.get("kind") in ("study", "strategy") else 0
    checks.append({"check": "Every clock time in the sentence is in the interpretation", "ok": unplaced == 0, "detail": "" if not unplaced else f"{len(times)} clock time(s) in the sentence, {n_have} in the interpretation"})
    # 3. a time zone that was assumed is said
    if intent.get("tz_assumed"):
        checks.append({"check": "Time zone", "ok": True, "detail": f"No time zone was stated; {intent['tz_assumed']} was assumed.", "assumed": True})
    # 4. the reader's own list of what it could not represent
    if intent.get("unsupported"):
        checks.append({"check": "Parts that could not be represented", "ok": False, "detail": "; ".join(str(u) for u in intent["unsupported"])[:400]})
    return checks


def stage(intent: dict, name: str, status: str = "done", detail: str = "", **data) -> dict:
    """Record one orchestration stage on the intent (understand, validate, data, plan, execute, explain, follow_up)."""
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}")
    rows = intent.setdefault("stages", [])
    row = next((r for r in rows if r["stage"] == name), None)
    if row is None:
        row = {"stage": name}
        rows.append(row)
        rows.sort(key=lambda r: STAGES.index(r["stage"]))
    row.update(status=status, detail=detail, **data)
    return intent


def from_frame(f: dict) -> dict:
    """A V8 question frame (strata.ai.semantics) in the same schema."""
    from strata.ai import semantics as SM
    conds = []
    for c in f.get("conditions") or []:
        if c.get("key") == "weekday":
            conds.append({"key": "weekday", "op": "is", "value": c.get("value"), "when": "open", "text": SM.describe_condition(c)})
            continue
        e = SM.cond_expr(c)
        if e:
            for x in _cond_from_expr(e[0], e[1]):
                if c.get("values"):
                    x["values"] = list(c["values"])
                conds.append(x)
        elif c.get("key") == "first_candle":
            conds.append({"key": "first_candle", "op": None, "value": c.get("minutes"), "when": "open", "text": SM.describe_condition(c)})
    o = f.get("outcome") or {}
    outcome = None
    end = str(f.get("end") or "close")
    hz = end if end.startswith("sessions:") else "session" if end == "close" else f"time:{end}"
    if o.get("key") == "touch":
        outcome = {"direction": "up" if o["value"] > 0 else "down", "size": abs(float(o["value"])), "measure": "close" if o.get("at_end") else "touch", "from": f.get("start") or "open", "horizon": hz}
    elif o.get("key") == "close_vs":
        outcome = {"direction": o.get("direction"), "size": None, "measure": "close", "from": f.get("start") or "open", "horizon": hz if end.startswith("sessions:") else "sessions:1" if f.get("next_session") else hz, "reference": o.get("ref")}
    elif o.get("key") == "touch_level":
        outcome = {"direction": o.get("direction"), "size": None, "measure": "touch", "level": o.get("level"), "from": f.get("start") or "open", "horizon": "session"}
    elif o.get("key") == "event":
        ev = o.get("event") or {}
        outcome = {"event": ev.get("key"), "direction": "up" if (ev.get("value") or 0) > 0 or ev.get("op") in (">", ">=") else "down", "size": abs(float(ev.get("value") or 0)) or None, "measure": "event"}
    return {"kind": "study" if f.get("kind") in ("frequency", "movement", "agreement", "timing", "breakdown") else f.get("kind"), "reader": "frame", "question": f.get("kind"), "symbol": f.get("symbol"),
            "conditions": conds, "outcome": outcome, "period": list(f.get("period") or [None, None]), "unsupported": list(f.get("unread") or []) if False else [],
            "assumptions": [a.get("text") for a in f.get("assumptions") or []], **{k: f[k] for k in ("candle", "scope", "predictor", "by", "measure", "bucket") if f.get(k) is not None}}


def from_route(r: dict, text: str, store, default_symbol: str = "SPY") -> dict:
    """The routing decision of one message as ONE intent, with the first two stages done: understood, and checked against the sentence."""
    from strata.ai.draft import compile_draft
    act = r.get("action")
    if r.get("frame") and (r["frame"].get("kind") in ("frequency", "movement", "agreement", "timing", "breakdown")) and act in ("new", "investigate", "profile"):
        intent = from_frame(r["frame"])
        intent["text"] = text
        stage(intent, "understand", "done", "a question about the underlying", reader="frame")
        intent["checks"] = validate(intent, text)
        bad = [c for c in intent["checks"] if not c["ok"]]
        stage(intent, "validate", "done" if not bad else "attention", "; ".join(c["detail"] for c in bad)[:300] if bad else "every part of the sentence has a place in the interpretation")
        return intent
    if act == "graph" and r.get("graph"):
        intent = from_graph(r["graph"])
    elif act in ("new", "modify", "derive") and r.get("draft") is not None:
        d = r["draft"]
        try:
            compiled = compile_draft(d, store)
        except Exception:  # noqa: BLE001
            compiled = None
        intent = from_draft(d, compiled)
        if act != "new":
            intent["kind"], intent["changed"] = "modify", list(r.get("changed") or [])
            intent["of"] = "study" if d.kind == "study" else "strategy"
    elif act == "hypothesis" and r.get("plan"):
        intent = from_hypothesis(r["plan"])
    elif act and act not in ("unknown", "failed"):
        intent = {"kind": "action", "reader": "route", "action": act, "symbol": default_symbol, "conditions": [], "unsupported": [], "assumptions": [], **{k: v for k, v in r.items() if k in ("field", "values", "by", "year", "symbols", "edit")}}
    else:
        intent = {"kind": "unreadable", "reader": None, "symbol": default_symbol, "conditions": [], "unsupported": [text.strip()], "assumptions": [], **({"action": act} if act else {})}
    if act == "clarify" and r.get("unread"):
        intent["unsupported"] = list(dict.fromkeys(list(intent.get("unsupported") or []) + [str(u) for u in r["unread"]]))  # V10: what the reading did not use stays with it
    intent["text"] = text
    stage(intent, "understand", "done", {"study": "a question about the underlying", "strategy": "an option strategy", "modify": "a change to the current version", "action": f"a request: {intent.get('action')}",
                                         "unreadable": "not understood"}[intent["kind"]], reader=intent.get("reader"))
    checks = validate(intent, text) if intent["kind"] in ("study", "strategy", "modify") else []
    intent["checks"] = checks
    bad = [c for c in checks if not c["ok"]]
    stage(intent, "validate", "done" if not bad else "attention", "; ".join(c["detail"] for c in bad)[:300] if bad else "every size and time in the sentence has a place in the interpretation")
    return intent


def understand(agent, text: str, default_symbol: str = "SPY", current=None, sid: str | None = None, llm: bool = False) -> dict:
    """What the application understands `text` to mean: the same decision the agent takes, expressed as one intent.

    `agent` is a strata.ai.agent.Agent. By default its built-in readers are used (no model, no network). With `llm`, a
    configured language model interprets the sentence instead, exactly as it does in the application; the intent it
    returns goes through the same validation. `current` is the draft a follow-up refers to, if any."""
    known = sorted(agent.store.stored_symbols())
    if llm:
        model, _why = agent.llm()
        if model is None:
            raise RuntimeError("no language model is configured: add a provider key in Settings, AI assistant (the built-in readers need none)")
        r, _interp, _note = agent.interpret(sid, text, current, None, None)
    else:
        r = agent._rules(text, current, known, None, default_symbol)
    it = from_route(r, text, agent.store, default_symbol)
    if it["kind"] == "unreadable" and it.get("action") == "unknown":
        it["kind"] = "action" if False else "unreadable"
    if r.get("action") == "unknown":
        it["action"] = "unknown"
    return it


# =====================================================================================================================
# V10: what the message SAYS, typed, and whether a reading holds it with the same meaning
#
# Until V10 a reading was checked by presence: every number of the sentence had to appear somewhere in the reading. That
# passed "out by 1245 PST" read as a 1,245% move and "sell when they're up 40%" read as a 40% move of SPY, because the
# digits were there. A number is not its digits: it is a clock time, a share of the option's price, a share of the
# underlying's, dollars, contracts, days. `said` types every quantity, time and name written in a message from the words
# around it; `held` types every value a reading contains from the slot it sits in; `check` requires each thing said to be
# held by a slot of the same type (and each value a new reading holds to have been said or declared assumed). What fails
# is listed in plain words and the reading waits; it never runs as if it had been understood.
# =====================================================================================================================
_WORDNUM = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2, "half": 0.5}
_N = r"(\d+(?:\.\d+)?|" + "|".join(k for k in _WORDNUM if k not in ("a", "an", "half", "couple")) + r")"
_UNDER = r"spy|qqq|iwm|the stock|the underlying|underlying|the index|the market|the etf|shares?|spot|the price"
_OPTION = r"premium|options?|contracts?|calls?|puts?|spreads?|condors?|straddles?|strangles?|butterfl\w+|credit|debit|the position|my position|the trade|they|them|'em|they're|theyre"
_ACCOUNT = r"account|equity|capital|portfolio|buying power|bankroll|balance|net liq\w*"
_EXITWORD = r"profits?|gains?|tp|take ?profits?|targets?|stops?|sl|stop ?loss(?:es)?|loss(?:es)?|lose|losing|max (?:profit|loss)|winners?|losers?"
_MOVE_OF_DAY = r"gaps?|gapp\w+|day|days|session|open|opens|opened|close|closes|closed|overnight|yesterday|previous|prior|intraday|morning|afternoon|from the open|on the day|range"
_UPDOWN = r"up|down|red|green|higher|lower|positive|negative|losing|winning|bullish|bearish"
_MOVEN = r"days?|sessions?|gaps?|open|opening|moves?|drops?|declines?|rall(?:y|ies)|sell-?offs?|pullbacks?|bounces?|dips?|pops?|rips?|dumps?|fades?|range|close|candles?|bars?"


def _num(tok: str) -> float | None:
    tok = tok.lower().strip()
    if tok in _WORDNUM:
        return float(_WORDNUM[tok])
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def _clause(low: str, a: int, b: int) -> tuple[str, str]:
    """The words of the same clause before and after [a, b): nothing beyond a comma, a semicolon, a full stop or "then"."""
    left = re.split(r";|,(?!\d)|\.(?!\d)|\bthen\b|\band (?=(?:sell|exit|close|stop|get out|take|buy|enter|cut|hold)\b)", low[:a])[-1]
    right = re.split(r";|,(?!\d)|\.(?!\d)|\bthen\b", low[b:])[0]
    return left[-70:], right[:45]


def about_percent(low: str, a: int, b: int, trade: bool, value: float | None = None, tickers: tuple = ()) -> str | None:
    """What a percentage at [a, b) is a share OF: "option" (its price), "underlying" (its price), "account", or None when
    the words do not say. The noun the number is attached to decides ("40% on the option", "1% in SPY", "2% of the
    account"); then the subject of its clause ("when THEY are up 40%": the options, a plural; "when SPY is up 1%"); then,
    in an options trade, the convention that a profit target or a stop with no subject is on the option's price."""
    left, right = _clause(low, a, b)
    _UNDER = globals()["_UNDER"] + "".join("|" + re.escape(t.lower()) for t in tickers)  # every stored ticker names the underlying
    if re.match(r"\s*(?:of (?:the |all )?(?:time|days|sessions|trades|cases|instances)|chance|odds|probability|win ?rate|hit ?rate|likelihood)\b", right):
        return "frequency"  # "60% of the time": a share of the observations, not a move of anything
    if re.match(r"\s*-?\s*(?:otm|itm|oom|out of the money|in the money|out\b(?! of)|away from the money)\b", right):
        return "strike"  # "puts 1% otm": how far the strike is from the price
    # (a) the noun the number is attached to
    att = re.match(rf"\s*(?:of|on|in|for|from|off|to)?\s*(?:the |its |their |my |our |a |an )?(?:\$?\d[\d,.]*\s?k?\s+)?(?:(?P<acc>{_ACCOUNT})|(?P<opt>{_OPTION}|max (?:profit|loss|gain)|credit received|debit paid)|"
                   rf"(?P<und>{_UNDER}|open|entry(?: price)?|where it was|previous close|prior close|yesterday'?s? close|high|low|vwap))\b", right)
    if att:
        return "account" if att.group("acc") else "option" if att.group("opt") else "underlying"
    if re.search(r"\brisk(?:ing|ed|s)?\s+(?:only\s+|about\s+|just\s+|max\s+|up to\s+)?$", left) or re.match(r"\s*(?:risk\b|per trade\b|of (?:my |the )?(?:stack|roll|acct)\b|of it (?:per|a|each|every) (?:trade|entry|position)\b|(?:per|a|each) (?:trade|entry|position)\b)", right):
        return "account"
    # (b) the number is the size of a day, a gap or a move: "a 1% up day", "a 2% gap", "a 1.5% drop from the open"
    mv = re.match(rf"\s*(?:\+|plus|or (?:more|bigger|larger|worse|better|greater))?\s*(?:(?:{_UPDOWN})\s+)?(?P<w>{_MOVEN}|increases?|decreases?|rises?|falls?|gains?|loss(?:es)?|changes?)\b"
                  rf"(?:\s+(?:in|of|on)\s+(?:the |its |their |my )?(?:(?P<o>{_OPTION})|(?P<u>{_UNDER}))\b)?", right)
    if mv and (mv.group("o") or mv.group("u")):
        return "option" if mv.group("o") else "underlying"
    if mv and re.fullmatch(_MOVEN, mv.group("w")) and not re.fullmatch(r"moves?|drops?|declines?|rall(?:y|ies)|bounces?|dips?|pops?|rips?|dumps?|fades?|pullbacks?|sell-?offs?", mv.group("w")):
        return "underlying"  # the size of a day, a session, a gap, an open, a candle
    if mv and not trade:
        return "underlying"
    if re.search(r"\bgap\w*\s*(?:ups?|downs?|higher|lower)?\s*(?:is|was|are|were|to|be)?\s*(?:only|just)?\s*(?:of|by|at least|more than|over|above|bigger than|larger than|greater than|>=?|<=?|under|below|less than)?\s*$", left):
        return "underlying"  # "puts on gap ups >1%": a gap is the price's
    # (c) the subject the number is said of, right before it: "when THEY are up 40%", "if SPY drops 1%", "the option loses 25%"
    seg = re.split(r"\b(?:when(?:ever)?|if|after|once|until|till|unless|while|as soon as|and|or|but|then|because|since)\b", left)[-1]
    # the last thing named before the number, when at most four words stand between them and no new clause starts there:
    # "they're up 40%", "the option has gained 60%", "if SPY itself has moved more than 0.7%", "when it drops 1%"
    sm = None
    for m_ in re.finditer(rf"\b(?:(?P<o>{_OPTION})|(?P<u>{_UNDER})|(?P<it>it))\b(?:'?re|'?s|'?ve)?", left):
        tail = left[m_.end():]
        if len([w_ for w_ in tail.split() if w_ not in ("itself", "themselves", "actually", "really", "just", "ever")]) <= 4 and not re.search(r"\b(?:when(?:ever)?|if|after|once|until|unless|while|and|or|but|then|at|with|for)\b", tail):
            sm = m_
    if sm is None:
        # "if SPY drops 0.7% from the entry or gains 1.5%": the second half of an "or" has the subject of the first
        halves = re.split(r"\b(?:or|and)\b", left)
        if len(halves) > 1 and not re.search(rf"\b(?:{_OPTION}|{_UNDER}|it)\b", halves[-1]):
            for m_ in re.finditer(rf"\b(?:(?P<o>{_OPTION})|(?P<u>{_UNDER})|(?P<it>it))\b", halves[-2]):
                if len(halves[-2][m_.end():].split()) <= 7:
                    sm = m_
    if sm and sm.group("o"):
        return "option"
    if sm and sm.group("u"):
        return "underlying"
    if re.match(r"\s*moves?\s+(?:either way|in either direction|up or down|in any direction|in one direction)", right):
        return "underlying"  # a move "either way" is the price's: an option's price is said to gain or lose
    if not trade:
        return "underlying"  # a question about prices: its percentages are moves of the price
    # (c2) a condition for getting IN ("after a 1% decline", "when it gaps down 1%"): what the price did before the trade
    mk = [m for m in re.finditer(r"\b(?:when(?:ever)?|if|after|once|following|given|on days|only if|as long as|while)\b", left)]
    exit_verb = r"\b(?:sell\w*|sold|exit\w*|clos(?:e|es|ed|ing)|out|stops?|take ?profits?|tp|sl|cut\w*|dump\w*|bail\w*|trim\w*|scale\w*|cover\w*|flat\w*|get(?:ting)? out|off)\b"
    if mk and not re.search(exit_verb, left[max(0, mk[-1].start() - 44):mk[-1].start()]) and not re.search(exit_verb, left[mk[-1].end():]) and territory(low, a) == "entry":
        return "underlying"
    if value is not None and value >= 10:
        # SPY, QQQ and IWM do not move 10% while an option trade is open, so a size like "up 30%" with no subject (or "it")
        # cannot be about the underlying: it is the position's price
        return "option"
    if sm and sm.group("it"):
        # "it" is one thing. When the contracts were named in the plural ("buy calls"), it is not them: it is the ticker.
        plural = re.search(r"\b(?:calls|puts|options|contracts|spreads|condors|straddles|strangles)\b", low)
        single = re.search(r"\b(?:a|an|the|one|1) (?:\w+ ){0,3}(?:call|put|option|contract|spread|condor|straddle|strangle|butterfly|position)\b", low)
        if plural and not single:
            return "underlying"
    # (d) the words of an open with no subject: "gaps down 1%", "opens 0.5% lower"; a move "on the day"
    if re.search(r"\b(?:gap\w*|opens?|opened)\s+(?:up|down|higher|lower)?\s*(?:by|at least|more than|over)?\s*$", left) or re.match(r"\s*(?:higher|lower|up|down)?\s*(?:on the day|intraday|on the session)\b", right):
        return "underlying"
    # (e) a target or a stop written in the shorthand of an options trader ("tp 30", "sl 15", "pt", "stc at +40%"): of the
    # option's price. A plain "1% target" does not say (the strategy reader asks which is meant).
    if re.search(r"\b(?:tp|sl|pt|stc|btc|take ?profit|stop ?loss)\b", seg[-40:] + " " + right[:24]) and (value is None or value >= 5 or re.search(r"\b(?:tp|sl|pt)\b", seg[-40:])):
        return "option"
    return None


def message_zone(text: str) -> int | None:
    """The one zone a message names, as minutes behind New York (the strategy reader's rule: a zone written once applies
    to the zone-less times of the same sentence)."""
    from strata.ai import nl_rules as NL
    low = text.lower()
    z = next((NL.TZ_OFFSET[k] for k in NL.TZ_OFFSET if re.search(rf"\b{k}\b", low) and k not in ("et", "ct", "mt", "pt", "ny", "uk")), None)
    if z is None:
        m = re.search(r"\b\d{1,4}(?::\d{2})?\s*(?:am|pm|a|p)?\s*,?\s*(pt|et|ct|mt|ny|uk)\b", low)
        z = NL.TZ_OFFSET[m.group(1)] if m else None
    return z


def is_trade(text: str) -> bool:
    """Does the message describe an options trade (as opposed to asking what prices did)? A structure, an expiration or a
    strike word, or a verb of trading with an option word. "Put" and "call" alone are not enough: "when does IWM put in
    its high" and "call it a day" are not trades."""
    low = " " + text.lower() + " "
    if re.search(r"\b(?:straddles?|strangles?|(?:iron )?condors?|butterfl(?:y|ies)|(?:credit|debit|vertical|calendar|put|call|bull|bear) spreads?|covered calls?|\d+ ?dte|0 ?dte|odte|zero dte|dte\b|atm\b|otm\b|itm\b|"
                 r"at[- ]the[- ]money|out[- ]of[- ]the[- ]money|in[- ]the[- ]money|\d+ ?-?delta|premium|contracts?|strikes?|expir\w+|weekl(?:y|ies)|leaps?|theta|bto\b|sto\b|btc\b|stc\b)\b", low):
        return True
    verb = r"buy(?:ing|s)?|bought|sell(?:ing|s)?|sold|short(?:ing|ed|s)?|long|writ(?:e|ing)|wrote|trad(?:e|ing|ed)|open(?:ing|ed)?|enter(?:ing|ed)?|hold(?:ing)?|held|scalp(?:ing|ed)?|play(?:ing)?|grab(?:bing|bed)?|take|took|get|got|load(?:ing)?|swing(?:ing)?|roll(?:ing)?"
    noun = r"calls?|puts?|options?"
    if re.search(rf"\b(?:{verb})\b[^.;?!]{{0,40}}?\b(?:{noun})\b", low) and not re.search(r"\bput (?:in|up|on|together|it|them)\b|\bputs? in (?:its|the|a|an)\b|\bcall (?:it|that|this)\b", low):
        return True
    if re.search(rf"\b(?:{noun})\b[^.;?!]{{0,30}}?\b(?:{verb})\b", low) and re.search(r"\b(?:calls|puts)\b", low):
        return True
    if re.search(r"\b(?:calls|puts)\b", low) and not re.search(r"\bputs? in\b|\bcalls? (?:it|that|for)\b", low):
        return True  # the plural names the contracts: "what about puts", "calls instead"
    return False


def said(text: str, known: list[str] | None = None, trade: bool | None = None) -> list[dict]:
    """Every quantity, time and name written in the message, each with what kind of thing it is.

    type: clock | pct | dollars | width | contracts | days | minutes | dte | delta | weekday | ticker | year | number
    Each item: {type, value, text, pos, end} plus, by type: `about` (pct: option | underlying | account | frequency | None),
    `role` (days: streak | span), `et`/`zones`/`assumed_tz` (clock). `number` is a number none of the rules could type.
    `trade`: whether the message is about an options trade (a follow-up to one is, whatever its own words); None = decide
    from the message."""
    from strata.ai import nl_rules as NL
    known = [k.upper() for k in (known or ["SPY", "QQQ", "IWM"])]
    low = text.lower()
    trade = is_trade(text) if trade is None else trade
    items: list[dict] = []
    taken = [False] * (len(text) + 1)

    def free(a: int, b: int) -> bool:
        return not any(taken[a:b])

    def add(a: int, b: int, typ: str, value, **more):
        if not free(a, b):
            return
        for i in range(a, b):
            taken[i] = True
        items.append({"type": typ, "value": value, "text": text[a:b].strip(), "pos": a, "end": b, **more,
                      **({"negated": True} if re.search(r"(?:\bnot|\bisn'?t|\binstead of|\brather than|\bno longer)\s+(?:at\s+|the\s+|a\s+)?$", low[:a]) else {})})

    # dates and years first, so that "2024-01-18" is not read as numbers and "since 2020" is not a time
    for m in re.finditer(rf"\b{NL.DATE_TOK}", low):
        add(m.start(), m.end(), "year", int(re.search(r"(?:19|20)\d\d", m.group(0)).group(0)), date=True)
    for m in re.finditer(r"\b(?:19|20)\d\d\b(?!\s*(?:%|dte|shares|contracts|points|pts|bucks|dollars))(?<!\$\d\d\d\d)", low):
        if low[max(0, m.start() - 1):m.start()] != "$" and free(m.start(), m.end()):
            add(m.start(), m.end(), "year", int(m.group(0)))
    for m in re.finditer(rf"\b(?:last|past|previous|prior)\s+{_N}\s+(?:years?|yrs?|months?)\b", low):
        add(m.start(), m.end(), "year", "span")  # held by any period the reading has (the start date is computed from today)
    for t in NL.find_times(text, message_zone(text)):
        # a time by its name is left to the readers, except noon: "close it at noon" is a time like any other, and a reading
        # that does not hold it has dropped something
        noon = t.get("how") == "named" and re.fullmatch(r"(?:at |by |around |until |till? )?(?:noon|midday|mid-day)(?:\s+(?:[pecm][sd]?t|pacific|eastern|central|mountain|new york|ny|chicago|la)(?: time)?)?", str(t.get("stated") or "").lower().strip()) is not None
        if (t.get("how") == "named" and not noon) or not free(t["pos"], t.get("end", t["pos"] + 1)):
            continue
        mins, et_ = t["minutes"], t["et"]
        if noon and mins == 720:
            # "noon pt" is noon in that zone; with no zone of its own, noon is in the zone the message's other times are in
            zm = re.match(r"\s*(pt|pst|pdt|pacific|la|mt|mst|mdt|mountain|ct|cst|cdt|central|chicago|et|est|edt|eastern|new york|ny)\b", low[t["end"]:])
            off = ({"p": 180, "l": 180, "m": 120, "c": 60}.get(zm.group(1)[0], 0) if zm else None)
            if off is None:
                off = message_zone(text) or 0
                if not off:
                    only_ = {tuple(x.get("zones") or []) for x in NL.find_times(text, None) if x.get("how") != "named" and x.get("assumed_tz") and len(x.get("zones") or []) == 1}
                    off = {v: k for k, v in ZONE_NAME.items()}.get(next(iter(only_))[0], 0) if len(only_) == 1 else 0
            if 570 <= 720 + off <= 960:
                mins = 720 + off
                et_ = f"{mins // 60:02d}:{mins % 60:02d}"
        add(t["pos"], t["end"], "clock", mins, et=et_, zones=list(t.get("zones") or []), assumed_tz=t.get("assumed_tz"), how=t.get("how"), stated=t["stated"],
            named=mins in (570, 960))  # 9:30 and 16:00 New York are the open and the close, whatever the reading calls them
    if trade:
        # "@ open", "at the open", "out at the close", "ride to the bell": the session's own two times, as the time of a trade
        for m in re.finditer(r"(?:@|\bat|\bon|\bby|\binto|\buntil|\btill?|\bto)\s*(?:the\s+)?(?:market\s+)?(?P<w>open(?:ing)?(?: bell)?|close|closing bell|bell|eod|end of (?:the )?day)\b(?! (?:price|above|below|higher|lower))", low):
            w = m.group("w")
            is_open = w.startswith("open") and not re.search(r"\b(?:to|until|till?)\s*(?:the\s+)?$", low[:m.start("w")])
            if w == "bell":
                is_open = bool(re.search(r"\bopen\w*", low[max(0, m.start() - 20):m.start()])) and not re.search(r"\b(?:to|until|till?|into)\s*(?:the\s+)?$", low[:m.start("w")])
            if re.search(r"\bfrom\s*(?:the\s+)?$", low[:m.start("w")]) or not free(m.start(), m.end()):
                continue
            add(m.start(), m.end(), "clock", 570 if is_open else 960, et="open" if is_open else "close", zones=[], assumed_tz=None, how="named", stated=text[m.start():m.end()].strip(), named=False, anchor=True)
    # percentages
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b|pct\b|per ?cent\b)", low):
        add(m.start(), m.end(), "pct", float(m.group(1)), about=about_percent(low, m.start(), m.end(), trade, float(m.group(1)), tuple(known)))
    for m in re.finditer(r"\b(half|quarter)(?: of)? (?:a |one )?(?:percent|pct|%)|\b(a|one|two|three|four|five) (?:whole |full )?(?:percent|pct)\b", low):
        v = {"half": 0.5, "quarter": 0.25}.get(m.group(1) or "") or _num(m.group(2) or "")
        add(m.start(), m.end(), "pct", float(v), about=about_percent(low, m.start(), m.end(), trade, float(v), tuple(known)))
    # a target or a stop written without the sign, and a multiple of the premium: "tp 30 sl 15", "stop at 2x the credit"
    if trade:
        for m in re.finditer(r"\b(?:tp|sl|take ?profits?|stop ?loss(?:es)?|profit target|pt)\s*(?:at|of|@|=|:|is)?\s*(\d+(?:\.\d+)?)(?!\s*(?:%|\d|dte|delta|points?|pts|bucks|dollars|x\b|:|am|pm))(?![\w.])", low):
            add(m.start(1), m.end(1), "pct", float(m.group(1)), about="option", implied=True)
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*x\b(?=[^.;]{0,24}\b(?:credit|premium|debit|received|collected|paid)\b)|\b(?:stop|sl|target|tp)\s*(?:at|of|@)?\s*(\d+(?:\.\d+)?)\s*x\b", low):
            add(m.start(), m.end(), "pct", float(m.group(1) or m.group(2)) * 100, about="option", multiple=True)
    if trade:
        # sizes of the option's price written in words: "for half the credit", "once they have lost half their value", "if it doubles"
        for m in re.finditer(r"\b(?:for |at |to |of )?half (?:of )?(?:the |its |their |that |my )?(?:credit|premium|value|max(?:imum)? profit|price)(?: received| collected| paid)?\b|\b(?:lost|loses?|losing|down|gained?|gains|up|made|make) half(?: of)?(?: (?:its|their|the) (?:value|price|premium))?\b", low):
            add(m.start(), m.end(), "pct", 50.0, about="option", worded=True)
        for m in re.finditer(r"\b(?:costs?|worth|trading at|priced at|reach(?:es)?|hits?|goes? to|gets? to) (?:double|twice|2x) (?:the |its |their )?(?:credit|premium|what (?:i|we) (?:took in|collected|received|got(?: paid)?|sold (?:it|them) for))\b(?: to (?:buy(?: it| them)? back|close))?|\b(?:double|twice) the (?:credit|premium) to (?:buy(?: it| them)? back|close)\b", low):
            add(m.start(), m.end(), "pct", 100.0, about="option", worded=True, role="premium_stop")  # buying back at twice the credit loses once the credit
        for m in re.finditer(r"\bdoubl(?:e|es|ed|ing)\b(?! (?:the |its |their )?(?:credit|premium) to)|\ba double\b", low):
            add(m.start(), m.end(), "pct", 100.0, about="option", worded=True)
    # the width of a spread
    for m in re.finditer(r"(?<![\w.])\$?(\d+(?:\.\d+)?)\s*-?\s*(?:points?|pts?|dollars?|\$)?\s*-?\s*(?:wide|wings?)\b|\b(?:width|wings?) (?:of )?\$?(\d+(?:\.\d+)?)", low):
        add(m.start(), m.end(), "width", float(m.group(1) or m.group(2)))
    if trade and re.search(r"spread|condor|butterfl|wings?|vertical|\bccs\b|\bpcs\b|\blong (?:the |leg|strike|put|call)|\bshort (?:the |leg|strike|put|call)", low):
        # "long the strike 5 points under it", "long strikes $5 further out": how far the bought strike is from the sold one
        for m in re.finditer(r"(?<![\w.])\$?(\d+(?:\.\d+)?)\s*(?:points?|pts?|dollars?|bucks|strikes? ?\$)?\s*(?:under|below|above|over|further out|farther out|further (?:otm|away)|away|apart|lower|higher|beyond|past)\b(?=[^.;]{0,30}|$)", low):
            if re.match(r"\s+(?:from\s+)?(?:the\s+)?(?:money|atm|at[- ]the[- ]money|spot)\b", low[m.end():]) or re.search(r"\bshort (?:strike|leg|put|call)s?\s*(?:is|at|:|,)?\s*$", low[:m.start()]):
                continue  # "short strike 5 points below the money": where the short strike is, not how wide the spread is
            if re.search(r"\b(?:long|buy|bought|wing|protect\w*|hedg\w*|the strike|strikes?)\b", low[max(0, m.start() - 40):m.start()]) and not re.search(r"%|percent|delta|dte", low[m.start():m.end() + 3]):
                add(m.start(), m.end(), "width", float(m.group(1)))
    if trade:
        for m in re.finditer(r"(?<![\w.])(\d{1,2})\s*d\b(?=\s*(?:calls?|puts?|strangles?|short|long|strikes?|delta))", low):
            add(m.start(), m.end(), "delta", float(m.group(1)))  # "0dte 30d calls": the 30-delta call
        for m in re.finditer(r"\b(?:i have|i've got|i got|i'm starting with|starting with|start with|trading|with)\s+(?:a |an |about |around |roughly )?\$?(\d[\d,]*(?:\.\d+)?)\s*(k|grand|thousand)\b(?!\s*(?:of|in) (?:premium|risk))", low):
            add(m.start(1), m.end(), "dollars", float(m.group(1).replace(",", "")) * 1000, role_hint="capital")
    # dollars
    for m in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|grand|thousand)?\b|(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(k|grand|thousand)?\s*(?:dollars?|bucks|usd)\b|(?<![\w.$])(\d[\d,]*(?:\.\d+)?)\s*(k|grand)\b(?=[^.;]{0,24}\b(?:account|acct|capital|bankroll|portfolio|to start|starting))", low):
        raw = m.group(1) or m.group(3) or m.group(5)
        mult = (m.group(2) or m.group(4) or m.group(6) or "")
        v = float(raw.replace(",", "")) * (1000 if mult in ("k", "grand", "thousand") else 1_000_000 if mult == "m" else 1)
        add(m.start(), m.end(), "dollars", v)
    if trade:
        for m in re.finditer(r"(?<![\w.$])(\d[\d,]*)\s+shares\b", low):
            add(m.start(), m.end(), "shares", float(m.group(1).replace(",", "")))  # "own 100 shares": the stock a covered call is written against
        for m in re.finditer(r"(?<![\w.$])(\d[\d,]*(?:\.\d+)?)k\b", low):
            add(m.start(), m.end(), "dollars", float(m.group(1).replace(",", "")) * 1000)  # "keep it off the 30k": thousands of dollars, wherever it stands
    # expiration, delta, contracts
    for m in re.finditer(r"(?<![\w.])(\d+)\s*-?\s*dte\b|\b(zero|0|o)\s*-?\s*dte\b|(?<![\w.])(\d+)\s*(?:days?|d)\s+(?:to|until|till|from|before)\s+expir\w*", low):
        add(m.start(), m.end(), "dte", float(m.group(1) or m.group(3) or 0))
    for m in re.finditer(rf"\bexpir\w*[^.;,]{{0,24}}?\b{_N}\s*(?:trading\s+)?(days?|weeks?)\s*(?:away|out|later)?\b|\b{_N}\s*(days?|weeks?)\s+(?:out|away|to expir\w*|until expir\w*)\b|\b(?:a|one) (week) out\b", low):
        num = _num(m.group(1) or m.group(3) or "1")
        unit = m.group(2) or m.group(4) or m.group(5) or "day"
        if num is not None and trade:
            a0 = m.start(1) if m.group(1) else m.start(3) if m.group(3) else m.start()
            add(a0, m.end(), "dte", float(num) * (7 if unit.startswith("week") else 1))
    for m in re.finditer(r"\b(?:same[- ]day (?:expir\w+|exp)|expir\w+ (?:the |that |on the |on that )?same day|expir\w+ (?:that|the) (?:same )?day|expir\w+ (?:that|this|the same) (?:afternoon|session)|expires? today|same[- ]day[- ]expir\w+|"
                         r"(?:that|the same)[- ]day(?:'s)? expir\w+|expir\w+ (?:on|at) the (?:day|close) (?:of|it is) (?:entry|opened|traded)|zero days? to expir\w+)\b", low):
        add(m.start(), m.end(), "dte", 0.0, worded=True)
    # "sell same-day QQQ put credit spreads", "same day options": the day names the contract, so it is its expiration
    for m in re.finditer(r"\bsame[- ]day(?= (?:[a-z]{1,5} )?(?:puts?|calls?|options?|contracts?|spreads?|credit|debit|iron|condors?|straddles?|strangles?|butterfl\w+|verticals?)\b)", low):
        add(m.start(), m.end(), "dte", 0.0, worded=True)
    for m in re.finditer(r"(?<![\w.])\.?(\d+(?:\.\d+)?)\s*-?\s*(?:delta|Δ)\b|\bdelta\s+(?:of\s+)?\.?(\d+(?:\.\d+)?)", low):
        v = float(m.group(1) or m.group(2))
        add(m.start(), m.end(), "delta", v if v > 1 else v * 100)
    for m in re.finditer(rf"\b{_N}\s*strikes?\s*(?:otm|out(?: of the money)?|away|above|below|higher|lower|up|down)\b|\b{_N}\s*(?:otm|out of the money)\s*strikes?\b", low):
        v = _num(m.group(1) or m.group(2) or "")
        if v is not None:
            add(m.start(), m.end(), "strikes", v)
    if trade:
        # "the put 10 points below the money", "$5 out of the money": a distance in dollars. SPY, QQQ and IWM list a strike
        # every dollar, so that many listed strikes from the at-the-money one (said when it is placed)
        for m in re.finditer(r"(?<![\w.])\$?(\d+(?:\.\d+)?)\s*(?:points?|pts?|dollars?|bucks)?\s*(?:below|under|above|over|out of|away from|otm from|from)\s+(?:the\s+)?(?:money|at[- ]the[- ]money|atm|spot)\b"
                             r"|(?<![\w.])\$(\d+(?:\.\d+)?)\s*(?:otm|out of the money)\b|(?<![\w.])(\d+(?:\.\d+)?)\s*(?:points?|pts?|dollars?|bucks)\s*(?:otm|out of the money|out)\b", low):
            v = float(m.group(1) or m.group(2) or m.group(3))
            if v == int(v) and 1 <= v <= 60:
                was = next((x for x in items if x["type"] == "dollars" and m.start() <= x["pos"] and x["end"] <= m.end()), None)
                if was is not None:  # "$3 otm" was first taken as an amount of money: it is a distance from the money
                    was.update(type="strikes", value=v, text=text[m.start():m.end()].strip(), pos=m.start(), end=m.end(), points=True)
                    for i_ in range(m.start(), m.end()):
                        taken[i_] = True
                else:
                    add(m.start(), m.end(), "strikes", v, points=True)
    for m in re.finditer(rf"\b{_N}\s*(?:contracts?|lots?|cons)\b|\bx\s?(\d+)\b(?!\s*(?:credit|the|premium))", low):
        v = _num(m.group(1) or m.group(2) or "")
        if v is not None:
            add(m.start(), m.end(), "contracts", v)
    # days and sessions: a run of days (a streak) or a span (a horizon, a holding time)
    DIRW = r"(?:red|green|down|up|losing|winning|lower|higher|negative|positive|bearish|bullish|straight|consecutive|back[- ]to[- ]back)"
    for m in re.finditer(rf"\b{_N}\s*(?:{DIRW}\s+){{0,2}}(?:trading\s+)?(day|session|close|week)s?\b(\s+(?:in a row|straight|running|consecutively|back to back))?", low):
        v = _num(m.group(1))
        if v is None:
            continue
        streak = bool(m.group(3)) or bool(re.search(DIRW, m.group(0)))
        if v == 1 and not streak and re.match(r"(?:a|an|one|single|1)\b", m.group(0)) and not re.match(r"\s*(?:later|after|afterwards?|out|hold)\b", low[m.end():]) \
                and not re.search(r"\b(?:within|over|for|hold\w*|next|following|after|another)\s+$", low[:m.start()]):
            continue  # "in one day", "a day": the same session, not a count of sessions
        add(m.start(), m.end(), "days", v * (5 if m.group(2) == "week" else 1), role="streak" if streak else "span")
    for m in re.finditer(rf"\b{_N}\s*(?:{DIRW}\s+)?(?:in a row|straight)\b|\b{_N}[- ]day (?:losing |winning )?(?:streak|run)\b", low):
        v = _num(m.group(1) or m.group(2) or "")
        if v is not None:
            add(m.start(), m.end(), "days", v, role="streak")
    # minutes and hours
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(?:min(?:ute)?s?|m)\b(?!\w)", low):
        add(m.start(), m.end(), "minutes", float(m.group(1)))
    for m in re.finditer(rf"\b(?:{_N}|an|a|half an?|a half)\s*-?\s*(?:hours?|hrs?)\b", low):
        tok = m.group(1)
        v = (_num(tok) if tok else 0.5 if "half" in m.group(0) else 1.0)
        if v is not None:
            add(m.start(), m.end(), "minutes", v * 60)
    for m in re.finditer(r"\b(?:first|last|final|opening|closing) (?:half[- ]hour|half an hour|30)\b|\bhalf[- ]hour\b", low):
        add(m.start(), m.end(), "minutes", 30.0)
    for m in re.finditer(r"\b(?:first|last|final|opening|closing|power) hour\b", low):
        add(m.start(), m.end(), "minutes", 60.0)
    # weekdays and tickers
    for m in re.finditer(r"\b(mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri)(?:day)?s?\b", low):
        if trade and (re.search(r"\b(?:exp(?:ir\w*)?|expiry)\b[^.;,]{0,12}$", low[:m.start()]) or re.match(r"(?:'s)? (?:exp(?:ir\w*)?|expiry)\b", low[m.end():])):
            continue  # "exp that friday": the day the contracts expire, held as days to expiration (repair)
        if len(m.group(0)) > 3 or m.group(0) in ("mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri"):
            add(m.start(), m.end(), "weekday", m.group(1)[:3].capitalize())
    for sym in sorted(set(known) | {"SPY", "QQQ", "IWM", "VIX"}, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w$])\$?{re.escape(sym.lower())}\b", low):
            add(m.start(), m.end(), "ticker", sym)
    # anything left that is a number
    for m in re.finditer(r"(?<![\w.$])\d+(?:\.\d+)?(?![\w.]*\d)", low):
        if free(m.start(), m.end()) and not re.match(r"(?:st|nd|rd|th)\b", low[m.end():]):
            add(m.start(), m.end(), "number", float(m.group(0)))
    items.sort(key=lambda x: x["pos"])
    return items


# --------------------------------------------------------------------------------------------- what a reading holds
_PCT_VARS = r"return_pct|gap_pct|from_open_pct|change_pct|open_to_close_pct|session_return|range_pct|dist_from_vwap_pct|max_decline_from_open_pct|max_rally_from_open_pct|pullback_from_high_pct|bounce_from_low_pct|_pct\b"


def _hhmm(t) -> int | None:
    if t == "open":
        return 570
    if t == "close":
        return 960
    m = re.fullmatch(r"(\d\d):(\d\d)", str(t or ""))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _held_expr(expr: str, slot: str, label: str, out: list) -> None:
    """The typed values inside one rule expression: "previous_session.return_pct <= -1" holds 1% of the underlying,
    "consecutive_down_days >= 3" three days in a row, "vix > 25" a level, "time >= hhmm('10:30')" a clock time."""
    for m in re.finditer(r"hhmm\(\s*['\"]?(\d\d?):(\d\d)['\"]?\s*\)|['\"](\d\d):(\d\d)['\"]", expr):
        out.append({"type": "clock", "value": int(m.group(1) or m.group(3)) * 60 + int(m.group(2) or m.group(4)), "slot": slot, "label": label})
    for part in re.split(r"\s+and\s+|\s+or\s+|&&|\|\|", re.sub(r"hhmm\([^)]*\)|['\"]\d\d:\d\d['\"]", "", expr)):
        nums = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", re.sub(r"\([^()]*\)", "()", part))]
        inner = [float(x) for x in re.findall(r"\(\s*[\"']?(\d+(?:\.\d+)?)", part)]  # candle length and look-back arguments
        if re.search(r"consecutive_|streak", part):
            typ, about = "days", None
        elif re.search(_PCT_VARS, part) or re.search(r"\*\s*\(?\s*1\s*[-+]\s*0?\.\d+", part):
            typ, about = "pct", "underlying"
        elif re.search(r"\bweekday\b", part):
            typ, about = "weekdaynum", None
        else:
            typ, about = "number", None
        for v in nums:
            if typ == "pct" and v == 0:
                continue
            out.append({"type": typ, "value": abs(v), "slot": slot, "label": label, **({"about": about} if about else {}), **({"role": "streak"} if typ == "days" else {})})
        for v in inner:
            out.append({"type": "number", "value": v, "slot": slot, "label": label})
        for m in re.finditer(r"\*\s*\(?\s*1\s*[-+]\s*(0?\.\d+)", part):  # entry.underlying_price * (1 - 0.005): half a percent
            out.append({"type": "pct", "value": round(float(m.group(1)) * 100, 6), "slot": slot, "label": label, "about": "underlying"})


def held_draft(d) -> list[dict]:
    """Every typed value a strategy or study draft holds, with the slot it sits in (strata.ai.draft.Draft)."""
    H: list[dict] = []

    def h(typ, value, slot, label, **more):
        if value is not None:
            H.append({"type": typ, "value": value, "slot": slot, "label": label, **more})

    h("ticker", d.symbol, "symbol", f"the ticker {d.symbol}", default=not d.symbol_stated)
    for slot, t, lab in (("entry_time", d.entry_time, "the entry time"), ("exit_time", d.exit_time, "the exit time")):
        mm = _hhmm(t)
        if mm is not None:
            h("clock", mm, slot, lab, named=t in ("open", "close"))
    for c in d.conditions:
        _held_expr(c.expr, "conditions", c.text or c.expr, H)
        if c.source == "assumed":
            for x in H:
                if x.get("slot") == "conditions" and x.get("label") == (c.text or c.expr):
                    x["default"] = True
    for s in d.sequence:
        _held_expr(s.expr, "sequence", s.text or s.expr, H)
        h("minutes", float(s.within_minutes) if s.within_minutes else None, "sequence", "the time allowed for that step")
    for a in d.adds:
        _held_expr(a.expr, "adds", a.text or a.expr, H)
    for c in d.exit_conditions:
        _held_expr(c.expr, "exit_conditions", c.text or c.expr, H)
    for w in d.weekdays or []:
        h("weekday", w, "weekdays", f"only on {w}")
    if any("vix" in (c.expr or "").lower() for c in list(d.conditions) + list(d.exit_conditions)) or any("vix" in (s_.expr or "").lower() for s_ in d.sequence):
        h("ticker", "VIX", "conditions", "the level of the VIX")
    if d.kind == "options":
        h("dte", float(d.dte) if d.dte is not None else None, "dte", "days to expiration")
        if d.strike is not None and d.strike.method == "delta":
            h("delta", round(d.strike.value * 100, 6) if d.strike.value <= 1 else d.strike.value, "strike", "the strike's delta")
        if d.strike is not None and d.strike.method == "pct_otm":
            h("pct", d.strike.value, "strike", "how far out of the money the strike is", about="strike")
        if d.strike is not None and d.strike.method == "strikes_otm":
            h("strikes", float(d.strike.value), "strike", "how many strikes out of the money")
        h("width", d.width, "width", "the width of the spread")
        sz = d.sizing
        if sz is None or sz.mode == "contracts":
            h("contracts", float((sz.value if sz is not None else None) or d.quantity or 1), "quantity", "the number of contracts", default=(d.quantity or 1) == 1 and sz is None)
            if d.structure == "covered_call":
                h("shares", 100.0 * float(d.quantity or 1), "structure", "the shares the calls are written against", named=True)
        if sz is not None:
            if sz.mode in ("dollars", "risk_dollars"):
                h("dollars", float(sz.value), "sizing", "dollars of premium per trade" if sz.mode == "dollars" else "dollars at risk per trade")
            elif sz.mode in ("risk_pct", "equity_pct"):
                h("pct", float(sz.value), "sizing", "the share of the account at risk per trade" if sz.mode == "risk_pct" else "the share of the account spent per trade", about="account")
            h("dollars", sz.capital, "sizing.capital", "the starting account")
            h("dollars", sz.daily_loss_stop, "sizing.daily_loss_stop", "the daily loss limit")
        h("pct", d.premium_target_pct, "premium_target_pct", "sell when the option's price is up that much", about="option")
        h("pct", d.premium_stop_pct, "premium_stop_pct", "sell when the option's price is down that much", about="option")
        h("pct", d.underlying_target_pct, "underlying_target_pct", f"sell when {d.symbol} has moved that much in the trade's favour", about="underlying")
        h("pct", d.underlying_stop_pct, "underlying_stop_pct", f"sell when {d.symbol} has moved that much against the trade", about="underlying")
        h("dollars", d.underlying_target_abs, "underlying_target_abs", f"sell when {d.symbol} has moved that many dollars in the trade's favour", about="underlying")
        h("dollars", d.underlying_stop_abs, "underlying_stop_abs", f"sell when {d.symbol} has moved that many dollars against the trade", about="underlying")
        h("days", float(d.hold_sessions) if d.hold_sessions else None, "hold_sessions", "sessions the position is held", role="span")
        if d.hold_to_expiration and (d.dte or 0) == 0:
            h("clock", 960, "hold_to_expiration", "held until the contracts expire at the close", named=True)
        h("minutes", float(d.hold_minutes) if d.hold_minutes else None, "hold_minutes", "minutes the position is held")
        h("weekday", d.exit_weekday, "exit_weekday", "the weekday of the exit")
        h("number", float(d.max_entries) if d.max_entries else None, "max_entries", "entries per session")
        for p in d.partials:
            h("pct", p.premium_pct, "partials", "a partial exit when the option's price is up that much", about="option")
            h("pct", p.underlying_pct, "partials", f"a partial exit when {d.symbol} has moved that much", about="underlying")
            h("pct", round(p.fraction * 100, 6), "partials", "the share of the position closed", about="position")
    st = d.study
    if d.kind == "study" and st is not None:
        for lv, slot, lab in ((st.target, "study.target", "the size of the move that is counted"), (st.stop, "study.stop", "the adverse move that ends the observation")):
            if lv is not None and lv.kind == "pct" and lv.value:
                h("pct", abs(lv.value), slot, lab, about="underlying")
            elif lv is not None and lv.kind == "abs" and lv.value:
                h("dollars", abs(lv.value), slot, lab, about="underlying")
            elif lv is not None and lv.kind == "expr" and lv.expr:
                _held_expr(lv.expr, slot, lab, H)
        if st.horizon_kind == "time":
            h("clock", _hhmm(st.horizon_time), "study.horizon_time", "when the observation ends")
        if st.horizon_kind == "sessions":
            h("days", float(st.horizon_sessions), "study.horizon_sessions", "sessions the observation lasts", role="span")
    for k in ("start", "end"):
        v = getattr(d, k, None)
        if v:
            h("year", int(str(v)[:4]), k, f"the period ({k})")
    return H


def held_frame(f: dict) -> list[dict]:
    """Every typed value a question frame holds (strata.ai.semantics)."""
    H: list[dict] = []

    def h(typ, value, slot, label, **more):
        if value is not None:
            H.append({"type": typ, "value": value, "slot": slot, "label": label, **more})

    from strata.ai import semantics as SM
    h("ticker", f.get("symbol"), "symbol", f"the ticker {f.get('symbol')}", default=not f.get("symbol_stated", True))
    for s_ in f.get("symbols") or []:
        if s_ != f.get("symbol"):
            h("ticker", s_, "symbols", f"the ticker {s_}")

    def cond(c: dict, slot: str):
        k, v = c.get("key"), c.get("value")
        try:
            lab = SM.describe_condition(c)
        except Exception:  # noqa: BLE001
            lab = c.get("text") or str(k)
        if k in ("prev_return", "gap", "from_open", "change", "move", "prev_otc", "day_return"):
            if isinstance(v, (int, float)) and v:
                h("pct", abs(float(v)), slot, lab, about="underlying")
            for x in c.get("values") or []:
                h("pct", abs(float(x)), slot, lab, about="underlying", invented=True)
        elif k in ("up_streak", "down_streak"):
            h("days", float(v), slot, lab, role="streak")
        elif k == "vix":
            h("number", float(v), slot, lab)
            h("ticker", "VIX", slot, lab)
        elif k == "weekday":
            h("weekday", v, slot, lab)
        elif k == "first_candle":
            h("minutes", float(c["minutes"]) if c.get("minutes") else None, slot, lab)
        for tk in ("at", "time"):
            if _hhmm(c.get(tk)) is not None and c.get(tk) not in ("open", "close"):
                h("clock", _hhmm(c[tk]), slot, lab)
        if k == "rel":
            rel_(c.get("rel") or {}, slot, lab)

    def rel_(rel: dict, slot: str, lab: str):
        """A relation between two prices: the times of day it names, and a size when it has one."""
        for side in ("a", "b"):
            x = str(rel.get(side) or "")
            if ":" in x and _hhmm(x.split(":", 1)[1]) is not None:
                h("clock", _hhmm(x.split(":", 1)[1]), slot, lab)
        if rel.get("pct"):
            h("pct", abs(float(rel["pct"])), slot, lab, about="underlying")

    for c in f.get("conditions") or []:
        cond(c, "conditions")
    o = f.get("outcome") or {}
    if o.get("key") == "touch" and o.get("value"):
        h("pct", abs(float(o["value"])), "outcome", "the size of the move that is counted", about="underlying")
    if o.get("key") == "event" and o.get("event"):
        cond(o["event"], "outcome")
    for part in ("outcome", "then"):
        if (f.get(part) or {}).get("key") == "rel":
            rel_(f[part]["rel"], part, SM.describe_rel(f[part]["rel"]))
    for slot in ("start", "end"):
        v = f.get(slot)
        if isinstance(v, str) and v.startswith("sessions:"):
            h("days", float(v.split(":")[1]), slot, "sessions the observation lasts", role="span")
        elif _hhmm(v) is not None and v not in ("open", "close"):
            h("clock", _hhmm(v), slot, "when the measurement starts" if slot == "start" else "when the measurement ends")
    for cm in f.get("candle") or []:
        h("minutes", float(cm), "candle", f"the first {cm}-minute candle")
    p = f.get("period") or [None, None]
    for i, v in enumerate(p):
        if v:
            h("year", int(str(v)[:4]), "period", "the period")
    for w_ in ([f["focus"]] if isinstance(f.get("focus"), str) else list(f.get("focus") or [])):  # one weekday against the rest, or two side by side
        h("weekday", w_, "focus", f"the weekday asked about ({w_})")
    if f.get("kind") == "timing" and f.get("bucket"):
        h("minutes", float(f["bucket"]), "bucket", f"{f['bucket']}-minute parts of the session", default=True)
    return H


def held_of(route: dict) -> tuple[list[dict], bool]:
    """(what the reading of a route holds, whether it could be typed). A draft or a question frame is typed slot by slot;
    a research graph, a hypothesis plan or an experiment is not, and only the presence of each number can be checked."""
    d = route.get("draft")
    fr = route.get("frame")
    if fr and fr.get("kind") in ("frequency", "movement", "agreement", "timing", "breakdown") and (d is None or getattr(d, "kind", None) == "study"):
        return held_frame(fr), True
    if d is not None:
        return held_draft(d), True
    nums = sorted(_numbers_in({k: v for k, v in route.items() if k in ("graph", "plan", "spec", "dims", "frame", "field", "values")}))
    return [{"type": "number", "value": v, "slot": "plan", "label": "the plan"} for v in nums], False


def _operative(route: dict) -> str:
    """What a research plan DOES, as text to search: its steps and settings, without the sentences it quotes from the user
    (the question, a title, the source of a step), which contain the user's own words whatever the plan does."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("question", "text", "title", "source", "src", "hypothesis", "raw", "label", "description", "note", "notes", "assumptions", "unsupported")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip({k: v for k, v in route.items() if k in ("graph", "plan", "spec")}), default=str).lower()


# ------------------------------------------------------------------------------------------- said against held
def _clock12(minutes: float) -> str:
    h, m = int(minutes) // 60, int(minutes) % 60
    return f"{((h + 11) % 12) + 1}:{m:02d} {'AM' if h < 12 else 'PM'}"


def describe_said(s: dict) -> str:
    """One thing the message says, in plain words: what kind of thing it was taken to be."""
    t, v = s["type"], s["value"]
    g = lambda x: f"{float(x):g}"  # noqa: E731
    if t == "clock":
        return f"a time of day, {_clock12(v)} New York time"
    if t == "pct":
        return {"option": f"{g(v)}% of the option's price", "underlying": f"a {g(v)}% move of the price", "account": f"{g(v)}% of the account", "frequency": f"{g(v)}% of the time"}.get(s.get("about"), f"{g(v)}%")
    if t in ("ticker", "weekday", "year"):
        return "a length of time" if v == "span" else str(v)
    words = {"dollars": lambda: f"${float(v):,.0f}", "width": lambda: f"a spread {g(v)} wide", "strikes": lambda: f"{g(v)} strike{'s' if v != 1 else ''} out of the money",
             "contracts": lambda: f"{g(v)} contract{'s' if v != 1 else ''}", "days": lambda: f"{g(v)} sessions" + (" in a row" if s.get("role") == "streak" else ""), "minutes": lambda: f"{g(v)} minutes",
             "dte": lambda: f"{g(v)} days to expiration", "delta": lambda: f"a {g(v)} delta", "number": lambda: g(v)}
    return words[t]() if t in words else str(v)


def _same(a, b, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


WEEKIDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
SM_WEEK = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}


ROLE_OF_SLOT = {"premium_target_pct": "premium_target", "premium_stop_pct": "premium_stop", "underlying_target_pct": "underlying_target", "underlying_stop_pct": "underlying_stop", "entry_time": "entry_time", "exit_time": "exit_time"}
ROLE_WORDS = {"premium_target": "the profit target on the option's price", "premium_stop": "the stop on the option's price", "underlying_target": "a target on the underlying's price",
              "underlying_stop": "a stop on the underlying's price", "entry_time": "the time the trade is entered", "exit_time": "the time the trade is closed"}


def _fits(s: dict, h: dict, prev: list[dict]) -> bool:
    """Is the slot `h` of a reading a place for what `s` says: the same value, and the same kind of thing?"""
    ts, th, v = s["type"], h["type"], s["value"]
    if h.get("slot") == "plan":
        return ts not in ("ticker", "weekday", "clock") and _same(h["value"], v)  # a research plan is not typed: its numbers are searched by value
    if ts == "ticker":
        return th == "ticker" and h["value"] == v
    if ts == "weekday":
        return (th == "weekday" and h["value"] == v) or (th == "weekdaynum" and _same(h["value"], WEEKIDX.get(v, -1)))
    if th in ("ticker", "weekday", "weekdaynum"):
        return False
    if ts == "clock":
        if s.get("anchor") and v == 960:
            return th == "clock" and 950 <= float(h["value"]) <= 960  # "at the close": the reader sells same-day options a few minutes before they stop trading
        return th == "clock" and _same(h["value"], v, 0.5)
    if ts == "pct":
        if th != "pct" or not _same(h["value"], v):
            return False
        a, b = s.get("about"), h.get("about")
        return a is None or b is None or a == b or (a == "underlying" and b == "strike") or (a == "option" and b == "position")
    if ts in ("dollars", "width"):
        return th in ("dollars", "width") and _same(h["value"], v)
    if ts == "minutes":
        if th in ("minutes", "number") and _same(h["value"], v):
            return True
        if th == "clock" and (_same(h["value"], 570 + v) or _same(h["value"], 960 - v)):
            return True  # "the first 30 minutes" ends at 10:00; "the last hour" starts at 15:00
        return th == "clock" and any(p["type"] == "clock" and p.get("slot") == h.get("slot") and _same(abs(h["value"] - p["value"]), v) for p in prev)  # "an hour later"
    if ts == "days":
        return th == "days" and _same(h["value"], v) and (not s.get("role") or not h.get("role") or s["role"] == h["role"])
    if ts == "year" and v == "span":
        return th == "year"  # "the last 15 years": any period the reading holds
    if ts in ("contracts", "dte", "delta", "year", "strikes", "shares"):
        return th == ts and _same(h["value"], v)
    if ts == "number":
        return _same(h["value"], v)
    return False


def check(route: dict, text: str, known: list[str] | None = None, current: dict | None = None, zone_known: bool = False) -> dict:
    """Does the reading in `route` hold what `text` says, with the same meaning?

    `current` is the route of what a follow-up edits (so that only what THIS message changes has to come from it). Returns
    {"said", "held", "typed", "issues"}; an issue is {"kind", "level", "text", ...}:

      unused      something said that the reading holds nowhere                              confirm
      mistyped    said as one kind of thing, held as another (a time as a percentage; a share
                  of the option's price as a move of the underlying)                         confirm
      invented    a size in a new reading that the message does not contain                  confirm
      zone        a time that falls inside the session in more than one time zone            ask
      subject     a percentage, in a trade, with nothing saying what it is a share of        ask
      implausible a move of the underlying no session has seen                               ask
    """
    d = route.get("draft")
    trade = (d is not None and getattr(d, "kind", None) == "options") or bool(current and getattr(current.get("draft"), "kind", None) == "options") or is_trade(text)
    S = [x for x in said(text, known, trade=trade) if not x.get("negated")]  # "not 3:55": what it is not is no part of the request
    if route.get("compare") and current and getattr(current.get("draft"), "symbol", None):
        cm = compare_clause(text.lower(), current["draft"].symbol)  # "next to the SPY numbers" names what it is compared with
        if cm:
            S = [x for x in S if not (cm.start() <= x["pos"] < cm.end())]
    H, typed = held_of(route)
    P = held_of(current)[0] if current else []
    used: set[int] = set()
    issues: list[dict] = []
    key = lambda x: (x["type"], str(x["value"]))  # noqa: E731  (a value carried from what is on screen is not new, whichever slot holds it now)
    prev_keys: dict = {}
    for p in P:
        prev_keys[key(p)] = prev_keys.get(key(p), 0) + 1
    new = []
    for h in H:
        if prev_keys.get(key(h), 0) > 0:
            prev_keys[key(h)] -= 1
        else:
            new.append(h)
    order = sorted(range(len(S)), key=lambda i: (S[i]["type"] == "number", S[i].get("about") is None))  # the best-typed things choose their slot first
    for i in order:
        s = S[i]
        if s["type"] == "pct" and s.get("about") == "frequency":
            continue
        cands = [j for j, h in enumerate(H) if j not in used and _fits(s, h, P)]
        if not cands:
            cands = [j for j, h in enumerate(H) if _fits(s, h, P)]  # one value said twice ("0DTE ... same day") may sit in one slot
        streak_lost = not typed and s["type"] == "days" and s.get("role") == "streak" and not re.search(r"streak|consecutive|in a row|in_a_row|run_of|runs? of", _operative(route))
        if streak_lost:
            cands = []  # a research plan is searched by value only: "2 days in a row" is not held by some other 2 in it
        if cands:
            best = next((j for j in cands if H[j] in new), cands[0])
            best = next((j for j in cands if s.get("about") and H[j].get("about") == s.get("about")), best)
            used.add(best)
            s["slot"] = H[best].get("slot")
            fam = ROLE_OF_SLOT.get(s["slot"])
            role = role_of(s, text, short=bool(d is not None and getattr(d, "structure", None) in SHORT_PREMIUM), bias=bias_of(getattr(d, "structure", None)) if d is not None else None) if (trade and fam and H[best] in new) else None
            if role and role != fam and role.split("_")[0] == fam.split("_")[0] or (role and fam and {role, fam} == {"entry_time", "exit_time"}):
                issues.append({"kind": "misplaced", "level": "confirm", "item": s, "held": H[best], "role": role,
                               "text": f"\"{s['text']}\" reads as {ROLE_WORDS[role]}, but it was taken as {ROLE_WORDS[fam]}."})
            if trade and s["type"] == "pct" and s.get("about") == "underlying" and s["slot"] in ("underlying_target_pct", "underlying_stop_pct") and H[best] in new \
                    and territory(text.lower(), s["pos"]) != "exit" and not re.search(r"\b(?:tp|pt|sl|take ?profits?|profit target|target\w*|aim\w*|stops?|stop ?loss(?:es)?|profits?|loss(?:es)?)\b",
                                                                                     text.lower()[max(0, s["pos"] - 40):s["end"] + 22]):
                # V10: "buy puts after it first pops 0.5% off the open": a move that is part of getting in, held as a way out
                issues.append({"kind": "entry_as_exit", "level": "confirm", "item": s, "held": H[best],
                               "text": f"\"{s['text']}\" is said about getting in, but it was taken as {ROLE_WORDS[fam]}."})
            if s["type"] == "pct" and s.get("about") is None and trade and H[best].get("about") in ("option", "underlying") and H[best] in new:
                other = "underlying" if H[best]["about"] == "option" else "option"
                issues.append({"kind": "subject", "level": "ask", "item": s, "read_as": H[best]["about"], "text": f"\"{s['text']}\" does not say what it is a percentage of. I read it as {s['value']:g}% of "
                               f"{'the option' + chr(39) + 's price' if H[best]['about'] == 'option' else 'the price of the underlying'}; it could also be {s['value']:g}% of {'the option' + chr(39) + 's price' if other == 'option' else 'the price of the underlying'}."})
            continue
        if not typed and s["type"] in ("ticker", "weekday", "year", "minutes", "days") and not streak_lost:
            continue  # a plan's serialisation cannot be searched for these; its own reader's coverage check stands
        digits = re.sub(r"\D", "", s["text"])
        wrong = None
        if s["type"] == "pct":
            wrong = next((h for h in H if h["type"] == "pct" and _same(h["value"], s["value"]) and h.get("about") and s.get("about") and h["about"] != s["about"]), None)
        elif s["type"] == "clock" and digits:
            wrong = next((h for h in H if h["type"] in ("pct", "dollars", "number") and _same(h["value"], float(digits))), None)
        if s.get("named") and wrong is None:
            continue
        if P and wrong is None and re.search(rf"\b{_REMOVE}\b", text.lower()) and any(_fits(s, p_, []) for p_ in P):
            continue  # "remove the vix condition": what a removal names is used by being taken out
        if wrong is not None and typed:
            held_as = (f"{wrong['value']:g}% of the option's price" if wrong.get("about") == "option" else f"a {wrong['value']:g}% move of the underlying" if wrong.get("about") == "underlying" else
                       f"{wrong['value']:g}% of the account" if wrong.get("about") == "account" else wrong.get("label") or "something else") if wrong["type"] == "pct" else (wrong.get("label") or "something else")
            issues.append({"kind": "mistyped", "level": "confirm", "item": s, "held": wrong, "text": f"You wrote \"{s['text']}\", which reads as {describe_said(s)}, but it was taken as {held_as}."})
        else:
            issues.append({"kind": "unused", "level": "confirm", "item": s, "text": f"\"{s['text']}\" ({describe_said(s)}) is not used in this reading."})
    explained = {id(it["held"]) for it in issues if it["kind"] == "mistyped"}  # one mistake is said once
    for h in new:
        if id(h) in {id(H[j]) for j in used} or h.get("default") or h.get("named") or id(h) in explained:
            continue
        if h.get("invented"):
            if not any(i["kind"] == "vague" for i in issues):
                sizes = [x["value"] for x in new if x.get("invented")]
                issues.append({"kind": "vague", "level": "confirm", "held": h, "sizes": sizes,
                               "text": "Your message gives no size for that, so this reading tries " + ", ".join(f"{v:g}%" for v in sizes[:-1]) + f" and {sizes[-1]:g}% and shows each. Say a size if you have one in mind."})
        elif h["type"] == "pct" and h.get("about") in ("option", "underlying") and h["value"] and typed and route.get("action") in ("new", "investigate", "profile", None):
            issues.append({"kind": "invented", "level": "confirm", "held": h, "text": f"The reading contains {h['value']:g}% ({h.get('label')}), which your message does not say."})
    for s in S:
        # a time with no zone is New York time by convention, and said so. It is asked about only when it cannot be New
        # York time and more than one other zone fits ("845": 8:45 Central, Mountain or Pacific)
        if s["type"] == "clock" and len(s.get("zones") or []) > 1 and not zone_known and s.get("slot") and s.get("assumed_tz") not in (None, "Eastern"):
            alts = []
            for z in s["zones"]:
                off = {"Eastern": 0, "Central": 60, "Mountain": 120, "Pacific": 180, "London": -300}[z]
                local = s["value"] - {"Eastern": 0, "Central": 60, "Mountain": 120, "Pacific": 180, "London": -300}[s.get("assumed_tz") or "Eastern"]
                alts.append({"zone": z, "et": local + off})
            issues.append({"kind": "zone", "level": "ask", "item": s, "alternatives": alts,
                           "text": f"\"{s['text']}\" has no time zone and falls inside the session in more than one: " + " or ".join(f"{a['zone']} ({_clock12(a['et'])} New York)" for a in alts) + "."})
    for h in new:
        if id(h) in explained:
            continue
        if h["type"] == "pct" and h.get("about") == "underlying" and float(h["value"]) > 15 and typed:
            issues.append({"kind": "implausible", "level": "ask", "held": h, "text": f"The reading has a {h['value']:g}% move of the underlying ({h.get('label')}). No session of SPY, QQQ or IWM has moved that much, so this is probably not what you meant."})
    if current is None:
        kind_issue = kind_mismatch(route, text, known)
        if kind_issue:
            issues.append({"kind": "kind", "level": "confirm", "text": kind_issue})
    if d is not None and getattr(d, "kind", None) == "options" and d.structure in SHORT_PREMIUM:
        for s_ in S:
            if s_.get("multiple") and s_.get("slot") == "premium_stop_pct" and float(s_["value"]) >= 200:
                n_ = float(s_["value"]) / 100
                ph = re.search(r"\d+(?:\.\d+)?\s*x\b(?:\s+(?:the |my |of the )?(?:credit|premium)(?: received| collected)?)?", text[s_["pos"]:], re.I)
                a_, b_ = (s_["pos"] + ph.start(), s_["pos"] + ph.end()) if ph else (s_["pos"], s_["end"])
                issues.append({"kind": "multiple", "level": "ask", "item": s_,
                               "text": f"\"{text[a_:b_]}\" can mean two stops on a position that was sold: buy it back when it costs {n_:g} times the credit (a loss of {n_ - 1:g} times the credit), "
                                       f"or stop when the loss is {n_:g} times the credit. Which one?",
                               "options": [{"label": f"When it costs {n_:g}x the credit (loss: {n_ - 1:g}x the credit)", "ask": text[:a_] + f"a loss of {(n_ - 1) * 100:g}% of the credit" + text[b_:]},
                                           {"label": f"When the loss is {n_:g}x the credit", "ask": text[:a_] + f"a loss of {n_ * 100:g}% of the credit" + text[b_:]}]})
    # the instrument: a follow-up that says to open another position ("buy a put instead") must end up holding it, and a
    # change between a sold and a bought position carries exits that were written for the other one
    cd = (current or {}).get("draft")
    if d is not None and getattr(d, "kind", None) == "options" and cd is not None and getattr(cd, "kind", None) == "options":
        ni = named_instrument(text)
        if ni and ni != d.structure:
            issues.append({"kind": "instrument", "level": "confirm",
                           "text": f"Your message says to {instrument_words(ni)}, but this reading still has the position as it was ({instrument_words(d.structure, False)})."})
        elif d.structure != cd.structure and (d.structure in SHORT_PREMIUM) != (cd.structure in SHORT_PREMIUM) and (d.premium_target_pct is not None or d.premium_stop_pct is not None):
            issues.append({"kind": "carried_exits", "level": "confirm",
                           "text": f"The position changed from {instrument_words(cd.structure, False)} to {instrument_words(d.structure, False)}. The profit target and stop were written for the old one "
                                   f"(a share of the {'credit received' if cd.structure in SHORT_PREMIUM else 'price paid'}); they are kept as a share of the {'credit received' if d.structure in SHORT_PREMIUM else 'price paid'}."})
    seen, out = set(), []
    for it in issues:
        k = (it["kind"], it["text"])
        if k not in seen:
            seen.add(k)
            out.append(it)
    return {"said": S, "held": H, "typed": typed, "issues": out}


ASKS = {"frequency": "how often something happens", "movement": "how far the price moves", "timing": "when in the day something happens", "breakdown": "how the weekdays or months compare",
        "agreement": "whether one thing goes with another"}
MEASURES = {**ASKS, "movement": "how the price moved afterwards (it counts nothing)", "strategy": "an options trade", "plan": "a research plan with several steps"}


def kind_mismatch(route: dict, text: str, known: list[str] | None = None) -> str | None:
    """A reading that answers another KIND of question than the one asked, in words; None when the kinds agree or the
    message does not say. "Chance QQQ takes out yesterday's high?" asks how often; a reading that measures how the price
    moved after a signal counts nothing, whatever numbers and words it holds."""
    from strata.ai import semantics as SM
    _n, ms = SM.tag(text, known)
    said_kinds = [m.value["kind"] for m in ms if m.kind == "kind" and m.value["kind"] in ASKS]
    if any(m.kind == "outcome" and m.value.get("key") == "extreme_time" for m in ms) and any(m.kind == "when" for m in ms):
        said_kinds.insert(0, "timing")
    if not said_kinds:
        return None
    order = ("agreement", "timing", "breakdown", "movement", "frequency")
    asked = "timing" if said_kinds[0] == "timing" else next(k for k in order if k in said_kinds)
    d, fr, act = route.get("draft"), route.get("frame"), route.get("action")
    if fr and fr.get("kind") in ASKS:
        got = fr["kind"]
    elif act in ("graph", "hypothesis", "discover") or (act == "investigate" and not fr):
        return None  # a plan answers in several ways; its own reader's coverage check stands
    elif d is not None and getattr(d, "kind", None) == "options":
        got = "strategy"
        if is_trade(text):
            return None
    elif d is not None and getattr(d, "kind", None) == "study":
        st = d.study
        counted = st is not None and (st.target is not None or (st.measure == "close" and st.direction is not None))
        got = "frequency" if counted else "movement"
    else:
        return None
    if got == asked or {got, asked} <= {"frequency", "agreement"} or (asked == "movement" and got == "frequency"):
        return None
    if asked == "frequency" and got == "breakdown" and "breakdown" in said_kinds:
        return None
    return f"You asked {ASKS[asked]}; this reading measures {MEASURES[got]}."


# ----------------------------------------------------------------------------------------------- time zones, tidying
ZONE_ABBR = {0: "ET", 60: "CT", 120: "MT", 180: "PT", -300: "London time"}
ZONE_NAME = {0: "Eastern", 60: "Central", 120: "Mountain", 180: "Pacific", -300: "London"}


def tidy(text: str) -> str:
    """The message as the readers get it: typographic quotes and dashes made plain, and a decimal written without its zero
    given one (".3%" is 0.3%, which every number pattern would otherwise read as 3%)."""
    t = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", " - ").replace("\u00a0", " ")
    t = re.sub(r"(?<![\d.\w])\.(\d+)", r"0.\1", t)
    # a fraction of a percent in words is a number every reader must see ("sell when SPY is up half a percent")
    for pat, repl in ((r"\bthree[- ]quarters? of (?:a|one) (?:percent|pct)\b", "0.75%"), (r"\b(?:one|a) and a half (?:percent|pct)\b", "1.5%"), (r"\bhalf (?:of )?(?:a|one) (?:percent|pct)\b|\ba half (?:percent|pct)\b", "0.5%"),
                      (r"\b(?:a|one) quarter (?:of (?:a|one) )?(?:percent|pct)\b|\bquarter (?:of a )?(?:percent|pct)\b", "0.25%"), (r"\b(?:a|one) tenth of (?:a|one) (?:percent|pct)\b", "0.1%")):
        t = re.sub(pat, repl, t, flags=re.I)
    # a trader's "north of 25" is above 25, and "red like 2 days in a row" is 2 days: said so that every reader sees the same
    t = re.sub(r"\bnorth of(?=\s+\$?\d)", "above", t, flags=re.I)
    t = re.sub(r"\bsouth of(?=\s+\$?\d)", "below", t, flags=re.I)
    t = re.sub(r"\b(?:like|say|maybe|roughly|approximately)\s+(?=\d+\s+(?:days?|sessions?|times?|weeks?)\b)", "", t, flags=re.I)
    # a window said with "between": "between 9:30 and 10", "between 10:00 and 11:00 Chicago time", "the 10 to 11 hour". At least
    # one side has to look like a clock (a colon, am/pm, noon, the open or close), so "VIX between 15 and 20" stays as it is
    _T = r"(?:\d{1,2}:\d{2}|\d{1,2})(?:\s*(?:am|pm|a\.m\.|p\.m\.))?|noon|midday|the open|the close"
    def _window(m):
        x, y = m.group("x"), m.group("y")
        clockish = lambda z: bool(re.search(r":|am|pm|a\.m|p\.m|noon|midday|open|close", z, re.I))  # noqa: E731
        return f"from {x} to {y}" if (clockish(x) or clockish(y)) else m.group(0)
    t = re.sub(rf"\bbetween\s+(?:about\s+|around\s+|roughly\s+)?(?P<x>{_T})\s+and\s+(?:about\s+|around\s+|roughly\s+)?(?P<y>{_T})(?![\d%])", _window, t, flags=re.I)
    t = re.sub(r"\b(?:in |during |over )?the (\d{1,2})(?::(\d{2}))? to (\d{1,2})(?::(\d{2}))? (?:o'?clock )?hour\b", lambda m: f"from {m.group(1)}:{m.group(2) or '00'} to {m.group(3)}:{m.group(4) or '00'}", t, flags=re.I)
    # "closed below the day before two days in a row" is two down closes in a row
    for word, way in (("below|under|lower than", "down"), ("above|over|higher than", "up")):
        t = re.sub(rf"\b(clos(?:ed|es|e|ing)|finish(?:ed|es)?|end(?:ed|s)?) (?:{word}) (?:the day before|the prior day|the previous day|the day prior|(?:its|the|their) (?:prior|previous) (?:day'?s? )?close|yesterday'?s? close)"
                   rf"(?=\s+(?:\w+\s+)?(?:days?|sessions?|times?)\s+(?:in a row|straight|running))", rf"\1 {way}", t, flags=re.I)
    # the weekday before the weekday that is studied is "the day before", not a second weekday filter: "mondays only, where
    # the friday before was a down day and vix finished that friday above 20"
    for day, before in (("monday", "friday"), ("tuesday", "monday"), ("wednesday", "tuesday"), ("thursday", "wednesday"), ("friday", "thursday")):
        if re.search(rf"\b{day}s?\b", t, re.I) and re.search(rf"\b{before}\b", t, re.I):
            t = re.sub(rf"\b(?:the )?{before} before\b|\b(?:the )?(?:previous|prior|preceding) {before}\b", "the day before", t, flags=re.I)
            t = re.sub(rf"\b(?:on )?that {before}\b", "the day before", t, flags=re.I)
            t = re.sub(rf"\bafter an? (down|red|up|green|big down|big up) {before}\b", r"after a \1 day", t, flags=re.I)
    t = re.sub(r"\bfinish(es|ed)?\b(?=[^.?!]{0,40}\b(?:above|below|over|under|green|red|higher|lower|up|down)\b)", lambda m: "close" + {"es": "s", "ed": "d"}.get(m.group(1) or "", ""), t, flags=re.I)
    t = re.sub(r"\b(vix (?:close[ds]?|was|is)) (?:the day before|yesterday|the (?:previous|prior) (?:day|session)) (above|below|over|under) (\d+(?:\.\d+)?)", r"\1 \2 \3 the previous day", t, flags=re.I)
    t = re.sub(r"\bwhere the day before was an? (down|red|up|green) day\b", r"after a \1 day", t, flags=re.I)
    # "what time of day does the low usually print?" is "when does it usually make its low?"
    t = re.sub(r"\bdoes the (low|high)(?: of the (?:day|session))? ((?:usually |typically |normally |generally )?)(?:print|come in|get made|get set|get put in|form|land)\b", r"does it \2make its \1", t, flags=re.I)
    # "put" the verb is not the option: "put a stop on it", "put it next to the SPY numbers", "and put in a 30% stop". It is
    # the verb where a sentence or a clause starts with it and a pronoun, an article or a preposition follows; "buy a put on
    # SPY" and "sell put spreads" keep their put.
    t = re.sub(r"(^\s*|[,.;:!?]\s*|\b(?:and|then|to|you|please|let's|lets|i|i'd|i'll|we|would|also|now|just|ok|okay|so|but|can|could)\s+)(put)(?=\s+(?:it|that|this|them|those|these|me|us|both|everything|in|on|up|a|an|the|another|some|my)\b)",
               lambda m: m.group(1) + ("Place" if m.group(2)[0].isupper() else "place"), t, flags=re.I)
    return t


def zone_revealed(text: str) -> int | None:
    """The zone a message gives away without naming it: a clock time it calls the open or the close ("right at the 830
    open", "the 6:30 bell", "the 3pm close") is 9:30 or 16:00 in New York. Minutes behind New York, or None."""
    low = text.lower()
    for m in re.finditer(r"(?<![\d:])(\d{1,2}):?(\d{2})\s*(?:am|a\.m\.|a)?\s*(?:open|opening|bell)\b|\b(?:opens?|opening bell|the bell|market opens?)\s+(?:is\s+)?(?:at|@)\s*(\d{1,2}):?(\d{2})\b", low):
        h, mi = int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4))
        off = 570 - (h * 60 + mi)
        if off in ZONE_ABBR:
            return off
    # "the first 9:30 candle", "the opening 6:30 bar": the time names the session's first candle, which starts at the open
    for m in re.finditer(r"\b(?:first|opening)\s+(\d{1,2}):?(\d{2})\s*(?:am|a\.m\.)?\s*(?:candle|bar|print|tick)s?\b", low):
        off = 570 - (int(m.group(1)) * 60 + int(m.group(2)))
        if off in ZONE_ABBR:
            return off
    fm = re.search(r"\bfirst (?:(\d+) ?-?(?:min(?:ute)?s?)|(hour)|(half[- ]hour|half an hour))", low)
    if fm:
        end = 570 + (int(fm.group(1)) if fm.group(1) else 60 if fm.group(2) else 30)
        if re.search(rf"(?<![\d:]){end // 60}:{end % 60:02d}(?!\d)", low):
            return 0  # "drops 1% in the first hour ... where it was at 10:30": 10:30 is where that hour ends, in New York
    for m in re.finditer(r"(?<![\d:])(\d{1,2})(?::?(\d{2}))?\s*(?:pm|p\.m\.|p)?\s*(?:close|closing bell)\b|\b(?:closes?|closing bell|market closes?)\s+(?:is\s+)?(?:at|@)\s*(\d{1,2})(?::?(\d{2}))?\b", low):
        h, mi = int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4) or 0)
        for hh in (h, h + 12):
            off = 960 - (hh * 60 + mi)
            if off in ZONE_ABBR and off >= 0:
                return off
    return None


def with_zone(text: str, off: int) -> tuple[str, list[str]]:
    """The message with the zone written after every time that has none and could be in that zone. Returns the new text and
    the times it was added to."""
    from strata.ai import nl_rules as NL
    if message_zone(text) is not None:
        return text, []
    ins = []
    for t in NL.find_times(text, None):
        if t.get("how") == "named" or not t.get("assumed_tz"):
            continue
        if ZONE_NAME.get(off) in (t.get("zones") or []) or off == -300:
            ins.append((t["end"], text[t["pos"]:t["end"]]))
    out, done = text, []
    for end, words in sorted(ins, reverse=True):
        out = out[:end] + " " + ZONE_ABBR[off] + out[end:]
        done.append(words.strip())
    return out, list(reversed(done))


# ---------------------------------------------------------------------------------------- roles, and filling what is empty
_EXIT_CUE = r"(?:\bbuy(?:ing)? (?:it |them |the \w+ )?back|\bbought (?:it |them )?back|\bbuy to close|\bout|\bexit\w*|\bsell\w*|\bsold|\bclos(?:e|es|ed|ing)|\bflat(?:ten)?|\boff|\bstc|\bbtc|\btime ?stop|\b(?:hold|held|holding|keep|kept|keeping|stay\w*|rid(?:e|ing)|sit\w*|carry\w*)\b[^.;,]{0,24}?\b(?:until|till?|til)|\bdump\w*|\bcover\w*|\bdone|\bbail\w*|\bcut\w*|\bkill\w*|\bgone|\bhold(?:ing)? (?:it |them )?(?:until|till?|to))"
_ENTRY_CUE = r"(?:\bwait\w*\b[^.;,]{0,14}?\b(?:until|till?|til|for)|\bbuy\w*|\bbought|\bbto|\bsto|\benter\w*|(?<!from )(?<!at )(?<!since )\bentry(?! price)|\bentries|\bopen(?:ing|ed)?|\bin|\bon|\bput (?:it|them) on|\bfire\w*|\bget(?:ting)? in|\bgo(?:ing)? (?:long|short)|\blong|\bshort|\bsell to open|\bwrite|\binitiat\w+)"


_OPENING_SALE = r"\b(?:sell\w*|sold|short\w*|writ\w+|sto)\b(?! (?:it|them|out|half|the rest|everything|all)\b)[^.;,]{0,44}?\b(?:puts?|calls?|spreads?|condors?|straddles?|strangles?|butterfl\w+|premium|ic|ccs|pcs|verticals?)\b"


def territory(low: str, pos: int) -> str:
    """"exit" when the last verb before `pos` in its sentence is one of closing the position, "entry" otherwise. "Buy calls at
    9:45 when ..., selling when the option has gained 60%, when it has lost 35%, or at 3:30 pm": everything after "selling" is
    about getting out, whatever word stands right before each number."""
    left = re.split(r"[;.](?!\d)", low[:pos])[-1]
    left = re.sub(_OPENING_SALE, lambda m: "enter" + " " * (len(m.group(0)) - 5), left)
    ex = [m.end() for m in re.finditer(_EXIT_CUE + r"\b|\btake ?profits?\b|\btp\b|\bsl\b|\bpt\b(?=\s*(?:at|of|@|=|:)?\s*\+?\d)|\bstops?\b|\btake (?:it|them|half) off\b", left)]  # "pt" is Pacific time unless a number follows
    en = [m.end() for m in re.finditer(r"\bbuy\w*|\bbought|\bbto\b|\bsto\b|\benter\w*|\bentries\b|\bgo(?:ing)? (?:long|short)\b", re.sub(r"\bbuy(?:ing)? (?:it |them |the \w+ )?back|\bbought (?:it |them )?back|\bbuy to close", lambda m: " " * len(m.group(0)), left))]
    return "exit" if ex and (not en or max(ex) > max(en)) else "entry"


def bias_of(structure: str | None) -> tuple[int, str | None]:
    """(+1 a position that gains when the underlying rises, -1 when it falls, 0 neither; what ANY move of the underlying is
    for a position that leans neither way: a stop for a sold condor or straddle, a target for a bought straddle)."""
    st = structure or ""
    lean = 1 if st in ("long_call", "short_put", "bull_call_spread", "bull_put_spread", "covered_call") else -1 if st in ("long_put", "short_call", "bear_put_spread", "bear_call_spread") else 0
    return lean, ("stop" if st in ("iron_condor", "iron_butterfly", "short_straddle", "short_strangle") else "target" if st in ("long_straddle", "long_strangle") else None)


def role_of(s: dict, text: str, trade: bool = True, short: bool = False, bias: tuple | None = None) -> str | None:
    """What a thing said in an options trade is FOR, from the words next to it: the entry or exit time, the profit target or
    the stop (on the option's price or on the underlying's), the size. None when the words do not say. `short`: the
    position was sold to open, so its options LOSING value is the gain ("buy them back once they have lost half their
    value" is the target of a short put, and the stop of a long one)."""
    low = text.lower()
    if s.get("role") in ROLE_WORDS:
        return s["role"]
    left, right = low[max(0, s["pos"] - 70):s["pos"]], low[s["end"]:s["end"] + 26]
    left = re.split(r"[;.](?!\d)", left)[-1]
    t = s["type"]
    if t == "clock" and s.get("anchor"):
        # "@ open", "at the close", "to the bell": the word that introduces it is part of it
        lead = low[s["pos"]:s["end"]]
        if re.match(r"(?:by|into|until|till?|to)\b", lead) or territory(low, s["pos"]) == "exit":
            return "exit_time"
        return "entry_time" if s["value"] == 570 or not re.search(_EXIT_CUE + r"\b", left[-30:]) else "exit_time"
    if t == "clock":
        ex = [m.end() for m in re.finditer(_EXIT_CUE + r"\b", left)]
        en = [m.end() for m in re.finditer(_ENTRY_CUE + r"\b|@", re.sub(r"\bbuy(?:ing)? (?:it |them |the \w+ )?back|\bbought (?:it |them )?back|\bbuy to close", lambda m_: " " * len(m_.group(0)), left))]
        # "sell puts at 10", "short the 20 delta strangle at 9:45": a sale that OPENS the position is an entry
        opening = [m.end() for m in re.finditer(r"\b(?:sell\w*|sold|short\w*|writ\w+|sto)\b(?! (?:it|them|out|half|the rest|everything|all)\b)[^.;,]{0,44}?\b(?:puts?|calls?|spreads?|condors?|straddles?|strangles?|butterfl\w+|premium|ic|ccs|pcs)\b", left)]
        if opening and (not ex or max(opening) >= max(ex) - 1):
            ex = [x for x in ex if x > max(opening)]
            en = en + opening
        by = re.search(r"\bby\s*(?:the\s+)?$", left)
        if ex and en and max(en) > max(ex) and territory(low, s["pos"]) == "exit" and re.fullmatch(r"\s*(?:in|on)\b.*", left[max(en) - 2:max(en)] + "x") is None:
            # "sell at +60% or -40% on the option or at 3:55": the "on" of "on the option" is not an entry
            tail_words = left[max(ex):]
            if not re.search(r"\b(?:buy\w*|bought|bto|sto|enter\w*|entries|entry|go(?:ing)? (?:long|short)|open(?:ing|ed)?)\b", tail_words):
                return "exit_time"
        if ex and (not en or max(ex) >= max(en)):
            return "exit_time"
        if by and not en:
            return "exit_time"  # "by 11": a deadline, which in a trade is when it is closed at the latest
        if not ex and not en and territory(low, s["pos"]) == "exit" and re.search(r"(?:\bat|@|\bor)\s*(?:the\s+)?$", left):
            return "exit_time"  # "... sell when it has gained 60%, or at 3:30 pm"
        if en or re.search(r"(?:\bat|@)\s*(?:the\s+)?$", left):
            return "entry_time"
        return None
    if t == "pct" and s.get("about") in ("option", "underlying"):
        sign = re.search(r"([+-])\s*$", left)
        near = lambda pat: max([m.end() for m in re.finditer(pat, left)] or [-1])  # noqa: E731
        tg = near(r"\b(?:tp|pt(?=\s*(?:at|of|@|=|:)?\s*\+?$)|take ?profits?|profits?|targets?|gain\w*|winners?|up|higher|rall\w+|ris\w+|rose|made|make)\b")
        sp = near(r"\b(?:sl|stops?(?: ?loss(?:es)?)?|loss(?:es)?|los\w+|down|lower|drop\w*|fall\w*|fell|against|cut)\b")
        # a label right after the number names it ("50% take profit, 30% stop loss"), unless a label already stands right
        # before it ("PT 30% SL 20%": there the labels come first)
        pre_label = re.search(r"\b(?:tp|pt|take ?profits?|profit target|targets?|sl|stop ?loss(?:es)?|stops?)\s*(?:at|of|@|=|:|is)?\s*[+-]?\s*$", left)
        # which way the message writes its list: "PT 30% SL 20%" (label first) or "30% tp 20% sl" (label after). Whichever
        # form comes first in the message is its style; in the second style a label on the left belongs to the number before.
        LAB = r"(?:tp|pt|sl|take ?profits?|stop ?loss(?:es)?)"
        first_pre, first_post = re.search(rf"\b{LAB}\s*(?:at|of|@|=|:|is)?\s*[+-]?\s*\d", low), re.search(rf"\d\s*%\s*{LAB}\b", low)
        if pre_label and first_post and (first_pre is None or first_post.start() < first_pre.start()):
            pre_label = None
            left = left[:max(0, len(left) - 8)] if re.search(rf"\b{LAB}\s*$", left) else left
        if re.match(r"\s*(?:(?:premium|option|credit|debit|price)\s+)?(?:profits?|gains?|targets?|winners?)\b", right) or (not pre_label and re.match(r"\s*(?:take ?profits?|tp\b)", right)):
            tg = 99
        if re.match(r"\s*(?:(?:premium|option|credit|debit|price)\s+)?(?:loss(?:es)?|stops?|losers?)\b", right) or (not pre_label and re.match(r"\s*(?:stop ?loss(?:es)?|sl\b)", right)):
            sp = 99
        if re.match(r"\s*(?:more|above|over|higher) than (?:i|we) (?:paid|bought)|\s*(?:above|over) (?:my |the |our )?(?:cost|entry|debit|purchase price|fill)\b", right):
            tg = 99
        if re.match(r"\s*(?:less|below|under|lower) than (?:i|we) (?:paid|bought)|\s*(?:below|under) (?:my |the |our )?(?:cost|entry|debit|purchase price|fill)\b", right):
            sp = 99
        if s.get("multiple") and sp < 0 and tg < 0:
            sp = 0  # "2x the credit" with no word: by convention the stop
        if s.get("worded") and sp < 0 and tg < 0 and s["about"] == "option":
            here = low[s["pos"]:s["end"]]
            if short and re.search(r"\bdoubl\w+|\btripl\w+", here):
                sp, tg = 0, -1  # "buy it back if it doubles": a sold position that costs twice as much has lost its credit
            elif re.search(r"\bhalf (?:of )?(?:the |its |their |that |my )?(?:credit|max)", here) or re.search(r"\b(?:bought|buy|buying) (?:it |them )?back\b|\bcaptur\w+|\bkeep\w*|\bcollect\w*", left):
                tg = 0  # closing a credit position for half the credit keeps the other half: the target
            elif re.search(r"\b(?:lost|loses?|losing|down)\b", here):
                sp, tg = (-1, 0) if short else (0, -1)
            elif re.search(r"\b(?:gain\w*|up|made|make|doubl\w+)\b", here):
                sp, tg = (0, -1) if short else (-1, 0)
        if short and s["about"] == "option" and sp < 0 and tg < 0 and abs(float(s["value"]) - 50) < 1e-9 and territory(low, s["pos"]) == "exit" \
                and re.match(r"\s*of (?:the |my |its |that )?(?:credit|premium)", right):
            tg = 0  # bought back for 50% of the credit, or with 50% of it kept: the same level either way
        if sign:
            kind = "target" if sign.group(1) == "+" else "stop"  # a signed size says which it is: "+40% stop -25%" is a target of 40 and a stop of 25
        else:
            kind = "stop" if sp > tg else "target" if tg > sp else None
        if kind and short and s["about"] == "option" and not sign and re.search(r"\b(?:they|them|it|options?|puts?|calls?|contracts?|premium|value)\b[^.;,]{0,26}$", left) \
                and not re.search(r"\b(?:tp|pt|sl|take ?profits?|profits?|targets?|stops?|stop ?loss|loss(?:es)?)\b", left[-30:]):
            # said of the option's own price going up or down, on a position that was sold: down is the gain
            kind = {"stop": "target", "target": "stop"}[kind] if re.search(r"\b(?:lost|loses?|losing|down|lower|drops?|falls?|gain\w*|up|higher|ris\w+)\b", left[-30:]) else kind
        if s["about"] == "underlying" and not sign:
            # a move of the underlying named only by its direction, in the part of the message about getting IN ("after it
            # first pops 0.5% off the open"), is a condition of the entry: it is no way out
            label_ = re.search(r"\b(?:tp|pt|sl|take ?profits?|profit target|target\w*|aim\w*|stops?|stop ?loss(?:es)?|profits?|loss(?:es)?)\b", left[-40:] + " " + right[:22])
            if not label_ and territory(low, s["pos"]) != "exit":
                return None
        if s["about"] == "underlying" and bias is not None and not sign and territory(low, s["pos"]) == "exit":
            # V10: a move of the underlying said by its direction ("get out if SPY rallies 2%") is a stop when that direction
            # is against the position and a target when it is with it. Until V10 a rally was always a target, also on a call
            # credit spread, where a rally is what loses.
            explicit = re.search(r"\b(?:tp|sl|take ?profits?|profit target|targets?|stops?|stop ?loss(?:es)?|loss(?:es)?|profits?|against|favou?r)\b", left[-36:] + " " + right[:20])
            up_ = re.search(r"\b(?:rall\w+|ris\w+|rose|up|higher|climb\w*|gain\w*|runs?|pops?)\b", left[-30:])
            dn_ = re.search(r"\b(?:drop\w*|fall\w*|fell|down|lower|declin\w+|sell\w* ?off|slid\w*|dumps?|tanks?)\b", left[-30:])
            if not explicit and bool(up_) != bool(dn_) and bias[0]:
                kind = "target" if (bool(up_) and bias[0] > 0) or (bool(dn_) and bias[0] < 0) else "stop"
            elif not explicit and not bias[0] and bias[1] and (up_ or dn_ or re.search(r"either (?:way|direction)|up or down|in any direction|moves?\b", left[-30:] + right[:30])):
                kind = bias[1]
        if kind is None:
            return None
        return ("premium_" if s["about"] == "option" else "underlying_") + kind
    return None


def repair(route: dict, text: str, known: list[str] | None = None, current: dict | None = None) -> dict:
    """Fill what the reading of an options trade left EMPTY from things the message says with a clear role, and move a
    percentage that sits in the wrong kind of slot when the right one is empty. Nothing that the reading already holds is
    overwritten, and every change is listed in the draft's notes ("... from \"out by 1245 PST\""). What cannot be placed
    this way stays for `check` to report."""
    d = route.get("draft")
    if d is None or getattr(d, "kind", None) != "options" or route.get("action") not in ("new", "modify", "derive"):
        return route
    res = check(route, text, known, current, zone_known=True)
    todo = [i for i in res["issues"] if i["kind"] in ("unused", "mistyped", "misplaced")]
    open_q = [q for q in (d.questions or []) if q.get("answered") is None and q.get("id") in ("target", "stop")]
    twins = any(getattr(d, a_) is not None and getattr(d, b_) is not None and _same(getattr(d, a_), getattr(d, b_)) for a_, b_ in (("premium_stop_pct", "underlying_stop_pct"), ("premium_target_pct", "underlying_target_pct"))) \
        or any(i["kind"] == "invented" for i in res["issues"])
    if not todo and not open_q and not d.unsupported and not twins and not re.search(r"\bon the day\b|\bfor the day\b", text.lower()):
        return route
    d = d.model_copy(deep=True)
    notes: list[str] = []
    short = bool(re.match(r"short_|bull_put|bear_call|iron_", d.structure or "")) or bool(re.search(r"credit", (d.structure or "")))
    # "Do you mean a 40% move in QQQ or a 40% loss on the option?": not a question when the words say which ("take profit
    # +40%", "sell when they're up 40%"), and what role they give it
    for q in list(d.questions or []):
        if q.get("answered") is not None or q.get("id") not in ("target", "stop"):
            continue
        xs = [abs(float(v)) for o in q.get("options") or [] for v in (o.get("patch") or {}).values() if isinstance(v, (int, float))]
        it = next((x for x in res["said"] if x["type"] == "pct" and xs and _same(x["value"], xs[0]) and x.get("about") in ("option", "underlying")), None)
        if it is None:
            continue
        role = role_of(it, text, short=short, bias=bias_of(d.structure)) or (("premium_" if it["about"] == "option" else "underlying_") + q["id"])
        for f in ("underlying_target_pct", "premium_target_pct", "underlying_stop_pct", "premium_stop_pct"):
            if getattr(d, f) is not None and _same(getattr(d, f), it["value"]):
                setattr(d, f, None)
        if getattr(d, role + "_pct") is None:
            setattr(d, role + "_pct", float(it["value"]))
        d.questions = [x for x in d.questions if x is not q]
        notes.append(f"\"{it['text']}\" is read as {ROLE_WORDS[role]}.")
    if notes:
        res = check({**route, "draft": d}, text, known, current, zone_known=True)
        todo = [i for i in res["issues"] if i["kind"] in ("unused", "mistyped", "misplaced")]
    # values in the slot of another role first: they free the slot something else may belong in
    for it in [i for i in todo if i["kind"] == "misplaced"]:
        s, role, src = it["item"], it["role"], it["held"].get("slot")
        dst = role + ("_pct" if s["type"] == "pct" else "")
        if not hasattr(d, dst) or not hasattr(d, src):
            continue
        cur_dst = getattr(d, dst)
        moved = s["et"] if s["type"] == "clock" else float(s["value"])
        # the value that sits where this one belongs is itself said with the other role: the two were swapped
        other = next((x for x in res["said"] if x is not s and x.get("slot") == dst and role_of(x, text) == ROLE_OF_SLOT.get(src)), None)
        if cur_dst is None or other is not None:
            setattr(d, dst, moved)
            setattr(d, src, (other["et"] if other["type"] == "clock" else float(other["value"])) if other is not None else None)
            if s["type"] == "clock":
                setattr(d, dst + "_stated", s.get("stated") or s["text"])
                setattr(d, src + "_stated", (other.get("stated") or other["text"]) if other is not None else None)
            notes.append(f"\"{s['text']}\" is read as {ROLE_WORDS[role]}.")
    if notes:
        res = check({**route, "draft": d}, text, known, current, zone_known=True)
        todo = [i for i in res["issues"] if i["kind"] in ("unused", "mistyped")]
    for it in [i for i in todo if i["kind"] != "misplaced"]:
        s = it["item"]
        t, v = s["type"], s["value"]
        if it["kind"] == "mistyped" and t == "pct":
            h = it["held"]
            pairs = {"underlying_target_pct": "premium_target_pct", "underlying_stop_pct": "premium_stop_pct", "premium_target_pct": "underlying_target_pct", "premium_stop_pct": "underlying_stop_pct"}
            src = h.get("slot")
            dst = pairs.get(src)
            if dst and getattr(d, dst) is None and ((s.get("about") == "option") == dst.startswith("premium")) and _same(getattr(d, src), v):
                setattr(d, dst, float(v))
                setattr(d, src, None)
                notes.append(f"\"{s['text']}\" is read as {describe_said(s)}.")
            continue
        role = role_of(s, text, short=short, bias=bias_of(d.structure))
        if t == "clock" and role in ("exit_time", "entry_time") and getattr(d, role) is None:
            setattr(d, role, {"09:30": "open", "16:00": "close"}.get(s["et"], s["et"]))
            setattr(d, role + "_stated", s.get("stated") or s["text"])
            notes.append(f"{'Exit' if role == 'exit_time' else 'Entry'} at {_clock12(v)} New York time, from \"{s['text']}\".")
        elif t == "pct" and role and getattr(d, role + "_pct", None) is None and hasattr(d, role + "_pct"):
            setattr(d, role + "_pct", float(v))
            what = {"premium_target": "Sell when the option's price is up", "premium_stop": "Sell when the option's price is down", "underlying_target": f"Sell when {d.symbol} has moved in the trade's favour by",
                    "underlying_stop": f"Sell when {d.symbol} has moved against the trade by"}[role]
            notes.append(f"{what} {float(v):g}%, from \"{s['text']}\".")
        elif t == "contracts" and (d.quantity or 1) == 1 and d.sizing is None and float(v) >= 1:
            d.quantity = int(v)
            notes.append(f"{int(v)} contract{'s' if v != 1 else ''}, from \"{s['text']}\".")
        elif t == "dte" and d.dte is None:
            d.dte = int(v)
        elif t == "width" and d.width is None:
            d.width = float(v)
        elif t == "delta" and d.strike is None:
            from strata.ai.draft import DraftStrike
            d.strike = DraftStrike(method="delta", value=float(v) / 100)
        elif t == "strikes" and (d.strike is None or d.strike.method == "atm" or (s.get("points") and not any(x_["type"] == "delta" for x_ in free))) \
                and (s.get("points") or not re.search(r"\batm\b|at[- ]the[- ]money", text.lower())):
            from strata.ai.draft import DraftStrike
            d.strike = DraftStrike(method="strikes_otm", value=float(v))
            notes.append(f"The strike {int(v)} listed strike{'s' if v != 1 else ''} out of the money, from \"{s['text']}\"" + (f" ({d.symbol} lists a strike every dollar)." if s.get("points") else "."))
    # one size said is one thing: when the reader put the same number in a second slot ("a 50% stop" as a stop on the option
    # AND on the underlying; "risk 3% of the account" as the size AND a 3% target), the slot nothing said is emptied
    res = check({**route, "draft": d}, text, known, current, zone_known=True)
    said_pct = [x for x in res["said"] if x["type"] == "pct"]
    for slot in ROLE_OF_SLOT:
        if not slot.endswith("_pct") or getattr(d, slot) is None:
            continue
        v_ = getattr(d, slot)
        same = [x for x in said_pct if _same(x["value"], v_)]
        if same and not any(x.get("slot") == slot for x in same) and all(x.get("slot") for x in same):
            setattr(d, slot, None)
            notes.append(f"\"{same[0]['text']}\" is read as {describe_said(same[0])} only.")
    low_ = text.lower()
    VARCUE = (("gap_pct", r"gap\w*|open(?:s|ed|ing)?"), ("previous_session.return_pct", r"yesterday\w*|previous|prior|day before|(?:red|green|down|up) day"), ("from_open_pct", r"from (?:the |its )?open"), ("change_pct", r"on the day"))
    for it_ in [x for x in res["said"] if x["type"] == "pct" and x.get("about") == "underlying"]:
        twins_ = [c for c in d.conditions if re.search(rf"(?<![\d.]){re.escape(format(float(it_['value']), 'g'))}(?![\d.])", c.expr)]
        if len(twins_) < 2 or sum(1 for x in res["said"] if x["type"] == "pct" and _same(x["value"], it_["value"])) != 1:
            continue
        near = low_[max(0, it_["pos"] - 26):it_["end"] + 14]
        owner = next((c for var, pat in VARCUE for c in twins_ if var in c.expr and re.search(rf"\b(?:{pat})\b", near)), None)
        if owner is None:
            continue
        for c in twins_:
            if c is owner:
                continue
            c.expr = re.sub(r"(<|>)=?\s*-?\d+(?:\.\d+)?", lambda m_: m_.group(1) + " 0", c.expr)  # the size was said of the other condition: this one is a direction
            c.text = re.sub(r"\s*\d+(?:\.\d+)?%(?: or more)?", "", c.text or "").strip() or c.expr
            notes.append(f"\"{it_['text']}\" is the size of one condition only; the other is read without a size.")
    # V10: one condition said once is held once (the older readers write the previous day's move in two spellings), and a
    # limit on losses in dollars ("stop for the day after losing $200") is not a condition about a down day
    same_as = lambda e: re.sub(r"\s+", "", e).replace("return_pct(1)", "previous_session.return_pct")  # noqa: E731
    seen_, kept_ = set(), []
    for c in d.conditions:
        if same_as(c.expr) in seen_:
            continue
        seen_.add(same_as(c.expr))
        kept_.append(c)
    d.conditions = kept_
    if re.search(r"\b(?:los(?:e|es|ing|t)|down)\s+(?:more than\s+|over\s+)?\$\s?\d|\$\s?\d[\d,.]*k?\s+(?:loss|down|in losses)|daily (?:loss|stop)|loss limit|done for the day|stop(?:s|ping)? (?:trading )?for the day", low_) \
            and not re.search(r"\b(?:yesterday\w*|previous|prior|day before|(?:red|down|green|up|losing|winning) (?:day|session|close)s?|closed? (?:down|lower|red|up|higher|green)|after (?:a|an) |fell|dropped|sold off|rallied|rose)\b", low_):
        d.conditions = [c for c in d.conditions if not re.fullmatch(r"\s*(?:previous_session\.return_pct|return_pct\(1\))\s*[<>]=?\s*-?0(?:\.0+)?\s*", c.expr)]
    # V10: a percentage said about getting OUT ("get out if spy rallies 0.5%") that the older reader made an entry condition
    for it_ in [x for x in res["said"] if x["type"] == "pct" and x.get("about") == "underlying"]:
        if territory(low_, it_["pos"]) != "exit":
            continue
        role_ = role_of(it_, text, short=short, bias=bias_of(d.structure))
        if role_ not in ("underlying_target", "underlying_stop") or getattr(d, role_ + "_pct") is not None:
            continue
        tw = [c for c in d.conditions if re.search(rf"(?:from_open_pct|change_pct|session\.return_pct)\s*[<>]=?\s*-?{re.escape(format(float(it_['value']), 'g'))}\s*$", c.expr)]
        if len(tw) == 1:
            d.conditions = [c for c in d.conditions if c is not tw[0]]
            setattr(d, role_ + "_pct", float(it_["value"]))
            notes.append(f"\"{it_['text']}\" is about getting out: read as {ROLE_WORDS[role_]}, not as a condition for getting in.")
    notes += fill_trade(d, text, known)
    if getattr(d, "kind", None) == "options":
        # V10: a weekday that names the EXPIRATION ("STO the call spread exp that friday") is not a day the trade is entered
        mexp = re.search(r"\b(?:exp(?:ir\w*)?|expiry)\b[^.;,]{0,12}?\b(mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri)(?:day)?s?\b|\b(mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri)(?:day)?(?:'s)? (?:exp(?:ir\w*)?|expiry)\b", low_)
        if mexp:
            wd = (mexp.group(1) or mexp.group(2))[:3].capitalize()
            entry_days = [w for w in (d.weekdays or []) if w != wd]
            if wd in (d.weekdays or []) and entry_days:
                d.weekdays = entry_days
            if len(d.weekdays or []) == 1 and d.dte is None and wd != d.weekdays[0]:
                d.dte = (WEEKIDX[wd] - WEEKIDX[d.weekdays[0]]) % 7
                notes.append(f"Expiring that {SM_WEEK.get(wd, wd)}: {d.dte} days after the {SM_WEEK.get(d.weekdays[0], d.weekdays[0])} entry.")
    if not notes and d.model_dump() == route["draft"].model_dump():
        if d.unsupported:
            # what an earlier layer listed as not represented and the draft does hold ("up 50%", held as the target) is not
            # missing: the list is recomputed against the draft whether or not anything else was repaired
            left = still_unplaced(d, text, known, current)
            if left != list(d.unsupported):
                d.unsupported = left
                return {**route, "draft": d}
        return route
    d.notes = list(dict.fromkeys(list(d.notes) + notes))
    d.unsupported = still_unplaced(d, text, known, current)
    return {**route, "draft": d}


_SIZE_CUES = (("capital", r"accounts?|acct|bankroll|capital|portfolio|balance|stack|to start|start(?:ing|ed)? (?:with|at|out with)|starting|i have|i've got|i got"),
              ("daily_loss_stop", r"done for the day|(?:stop|quit|halt)\w* (?:trading )?(?:for the day|that day)|daily (?:loss )?(?:limit|stop|max)|max (?:daily|day) loss|loss limit|for the day|(?:i'?m|am|once|after|when) down|after losing"),
              ("risk_dollars", r"risk\w*|at risk|max loss per|lose at most"),
              ("dollars", r"(?:per|a|an|each|every) (?:trade|entry|position|time|day)|of premium|in premium|worth of|spend\w*|budget|siz\w+|notional|allocat\w+"))


def fill_trade(d, text: str, known: list[str] | None = None) -> list[str]:
    """What an options trade says in words the strategy reader does not know, placed by what it is: the size of the trade
    (dollars per trade, dollars or a share of the account at risk, the starting account, reinvesting, a daily loss limit),
    holding to expiration, and entry conditions the question reader recognises. Changes `d`; returns what it did."""
    from strata.ai import semantics as SM
    from strata.ai.draft import DraftCondition, DraftSizing
    low = text.lower()
    notes: list[str] = []
    res = check({"action": "new", "draft": d}, text, known, None, zone_known=True)
    free = [i["item"] for i in res["issues"] if i["kind"] == "unused"]
    # ---- how much is traded
    sz = d.sizing.model_copy() if d.sizing is not None else None
    upd: dict = {}
    for it in free:
        if it["type"] == "dollars":
            left, right = _clause(low, it["pos"], it["end"])
            best = None
            for role, pat in _SIZE_CUES:
                for m in re.finditer(rf"\b(?:{pat})\b", left):
                    dist = len(left) - m.end()
                    best = min(best, (dist, role)) if best else (dist, role)
                m = re.search(rf"\b(?:{pat})\b", right[:34])
                if m:
                    best = min(best, (m.start(), role)) if best else (m.start(), role)
            if best is None:
                continue
            role = best[1]
            if role == "dollars" and re.search(r"\brisk\w*\b|\bat risk\b", left[-30:] + " " + right[:16]):
                role = "risk_dollars"
            if role in ("capital", "daily_loss_stop"):
                if (sz is None or getattr(sz, role) is None) and role not in upd:
                    upd[role] = float(it["value"])
                    notes.append(f"{'A starting account of' if role == 'capital' else 'No new trade that day after losing'} ${float(it['value']):,.0f}, from \"{it['text']}\".")
            elif "mode" not in upd and (sz is None or sz.mode == "contracts"):
                upd.update(mode=role, value=float(it["value"]))
                notes.append(f"{'$' + format(float(it['value']), ',.0f') + ' at risk per trade' if role == 'risk_dollars' else '$' + format(float(it['value']), ',.0f') + ' of premium per trade'}, from \"{it['text']}\".")
        elif it["type"] == "pct" and it.get("about") == "account" and "mode" not in upd and (sz is None or sz.mode == "contracts"):
            left, right = _clause(low, it["pos"], it["end"])
            role = "risk_pct" if re.search(r"\brisk\w*|\bmax loss\b", left + " " + right) else "equity_pct"
            upd.update(mode=role, value=float(it["value"]))
            notes.append(f"{float(it['value']):g}% of the account {'at risk' if role == 'risk_pct' else 'in premium'} per trade, from \"{it['text']}\".")
    if re.search(r"\bcompound\w*|\bre-?invest\w*|\broll(?:ing)? (?:the )?(?:gains|profits)\b", low) and not re.search(r"\b(?:don'?t|do not|without|no|not)\s+(?:\w+\s+)?(?:compound\w*|re-?invest\w*)", low):
        if sz is None or not sz.reinvest:
            upd["reinvest"] = True
    if upd:
        base = sz or DraftSizing(mode="contracts", value=float(d.quantity or 1))
        d.sizing = base.model_copy(update=upd)
        d.sizing.text = "; ".join(n.split(", from")[0] for n in notes)
    # ---- held until the contracts expire
    if not d.hold_to_expiration and re.search(r"\blet (?:it|them|the \w+) (?:expire|ride to (?:the )?(?:bell|close|expir\w*))|\bhold\w* (?:it |them |on )?(?:to|until|till|through|into) (?:exp\w*|the bell)|\bheld to (?:exp\w*|the bell)|\bto expir\w+|\bexpire worthless|\bride (?:it |them )?(?:out )?to (?:the )?(?:bell|exp\w*)", low):
        d.hold_to_expiration = True
        notes.append("Held to expiration when no other exit comes first.")
    # ---- V10: a holding time ("20 min max hold", "if neither hits in 90 minutes I'm out"), entries per day ("max 3 a day"),
    # sessions held ("sell 2 days later")
    for it in free:
        left, right = low[max(0, it["pos"] - 56):it["pos"]], low[it["end"]:it["end"] + 30]
        if it["type"] == "minutes" and not d.hold_minutes:
            if re.search(r"(?:max(?:imum)? hold(?:ing)?(?: time)?(?: of| is)?|\bhold\w*(?: (?:it|them|each|every|one|for|up to|at most|no more than|max|only))*|time[- ]?stop(?: of| at| is| after)?|(?:\bout|exit|close|sell|dump|flat|done|gone)\w*(?: (?:it|them|everything|of it))?(?: after| in| within)"
                         r"|neither\b[^.;]{0,44}?\b(?:in|within|after)|give (?:it|them)|\bafter)\s*$", left) or re.match(r"\s*(?:max(?:imum)? hold|hold\b|time[- ]?stop|max\b|tops\b|after (?:entry|i get in|getting in)|in the trade)", right):
                d.hold_minutes = int(it["value"])
                notes.append(f"Closed {int(it['value'])} minutes after the entry at the latest, from \"{it['text']}\".")
        elif it["type"] == "number" and not d.max_entries and float(it["value"]) == int(it["value"]) and 1 <= it["value"] <= 20:
            if (re.search(r"\b(?:max(?:imum)?|at most|up to|no more than|limit(?:ed)? to|cap(?:ped)? at)\s*(?:of\s+)?$", left) and re.match(r"\s*(?:(?:trades?|entries|entry|times?|positions?|shots?)\s*)?(?:a|per|each|every)\s+(?:day|session)\b", right)) \
                    or re.match(r"\s*(?:trades?|entries|times?|positions?)\s+(?:a|per|each)\s+(?:day|session)\s+(?:max|tops|at most|maximum)", right):
                d.max_entries = int(it["value"])
                notes.append(f"At most {int(it['value'])} entries a day, from \"{it['text']}\".")
        elif it["type"] == "days" and it.get("role") == "span" and not d.hold_sessions and not d.hold_minutes and d.exit_time in (None, "close"):
            if re.search(r"\b(?:sell|sold|exit|close|out|hold|held|dump|cover)\w*(?: (?:it|them|everything))?(?: after| for| in)?\s*$", left) and (re.match(r"\s*(?:later|after|out)\b", right) or re.search(r"\b(?:hold|held)\w*(?: (?:it|them))?(?: for)?\s*$", left)):
                d.hold_sessions = int(it["value"])
                notes.append(f"Held {int(it['value'])} session{'s' if it['value'] != 1 else ''}, then sold at that close, from \"{it['text']}\".")
    # ---- "green on the day", "red on the day": the day so far against the PREVIOUS CLOSE (not against the open)
    for m_ in re.finditer(r"\b(?:is|was|still|stays?|trading|sits)?\s*(?P<d>green|red|up|down|positive|negative|higher|lower) (?:on|for) the day\b", low):
        up = m_.group("d") in ("green", "up", "positive", "higher")
        if territory(low, m_.start()) == "exit":
            continue
        want = f"change_pct {'>' if up else '<'} 0"
        wrong = f"from_open_pct {'>' if up else '<'} 0"
        if any(c.expr.replace(" ", "") == want.replace(" ", "") for c in d.conditions):
            continue
        d.conditions = [c for c in d.conditions if c.expr.replace(" ", "") != wrong.replace(" ", "")] + [DraftCondition(expr=want, text=f"it is {'up' if up else 'down'} on the day (against the previous close) at the entry")]
        notes.append(f"\"{m_.group(0).strip()}\" is read as {'above' if up else 'below'} the previous close at the entry.")
    # ---- entry conditions the question reader recognises and the strategy reader left out
    try:
        from strata.ai import nl_rules as NL
        _n, ms = SM.tag(text, known)
        # V10: one price against a level ("on mornings when QQQ has opened above the previous day's high", "only if it is
        # above yesterday's close right then"), in the rule language's own names
        exits_ = NL.exit_clauses(text)
        for m in ms:
            if m.kind != "rel":
                continue
            rel = m.value.get("rel") or {}
            oa, _ob = _n.span(m.a, m.b)
            a_ = {"open": "open", "state": "price"}.get(str(rel.get("a")))
            b_ = {"prev_high": "prev_high", "prev_low": "prev_low", "prev_close": "prev_close", "open": "open"}.get(str(rel.get("b")))
            if a_ is None or b_ is None or a_ == b_ or any(x0 <= oa < x1 for x0, x1 in exits_) or territory(low, oa) == "exit":
                continue
            lvl = f"{b_} * {1 + float(rel['pct']) / 100:g}" if rel.get("pct") else b_
            expr = f"{a_} {rel['op']} {lvl}"
            if any(c.expr.replace(" ", "") == expr.replace(" ", "") for c in d.conditions):
                continue
            words = SM.describe_rel({**rel, "a": "open" if a_ == "open" else "at:entry"}).replace(" at entry New York time", " at the entry").replace("it is ", "the price is ")
            d.conditions.append(DraftCondition(expr=expr, text=words))
            notes.append(f"Only when {words}, from \"{m.text}\".")
        serial = (" ".join(c.expr for c in d.conditions) + " " + " ".join(s_.expr for s_ in d.sequence) + " " + " ".join((c.text or "") for c in d.conditions) + (" weekday " + " ".join(d.weekdays) if d.weekdays else "")).lower()
        exits = NL.exit_clauses(text)
        for m in ms:
            if m.kind != "event" or m.value.get("key") in (None, "unsupported_event", "move"):
                continue
            oa, ob = _n.span(m.a, m.b)
            if any(a_ <= oa < b_ for a_, b_ in exits) or SM.holds(serial, m.value["key"]):
                continue
            if m.value["key"] in ("prev_return", "move") and re.search(r"\blos(?:e|es|ing|t|s|ses)\b", m.text.lower()) and not re.search(r"\b(?:day|session)\b", m.text.lower()):
                continue  # V10: "stop trading for the day after losing $200" is about the account, not about a down day
            if m.value["key"] == "prev_return" and any(re.match(r"\s*(?:previous_session\.return_pct|return_pct\(1\))\s*[<>]", x.expr) for x in d.conditions):
                continue  # the trade already has a previous-day condition: the same words are not added a second time
            c = {**m.value, "text": m.text}
            if c["key"] == "weekday":
                if re.search(rf"\b(?:exp(?:ir\w*)?|expiry)\b[^.;,]{{0,12}}?\b{c['value'].lower()}|\b{c['value'].lower()}\w*(?:'s)? (?:exp(?:ir\w*)?|expiry)\b", low):
                    continue  # the weekday the contracts expire on (repair works out the days to expiration from it)
                if c["value"] not in (d.weekdays or []):
                    d.weekdays = list(d.weekdays or []) + [c["value"]]
                    notes.append(f"Only on {SM.WEEK.get(c['value'], c['value'])}.")
                continue
            if c["key"] == "first_candle":
                if not (c.get("direction") in ("up", "down") and c.get("minutes")):
                    continue
                end = _hhmm_str(570 + int(c["minutes"]))
                if d.entry_time is None:
                    d.entry_time = end
                    notes.append(f"Entered when the first {int(c['minutes'])}-minute candle closes ({end} New York time).")
                elif _hhmm(d.entry_time) != 570 + int(c["minutes"]):
                    continue  # the candle's direction at a later entry is not the price against the open at that time: left for the user to see
            e = SM.cond_expr(c)
            if e and e[0] not in [x.expr for x in d.conditions]:
                d.conditions.append(DraftCondition(expr=e[0], text=e[1]))
                serial += " " + e[0].lower() + " " + e[1].lower()
                notes.append(f"Only when {e[1]}, from \"{m.text}\".")
    except Exception:  # noqa: BLE001  (a condition that cannot be added is reported by the check, not hidden)
        pass
    return notes


def _hhmm_str(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


_FILLER_PHRASE = re.compile(r"\b(?:only|just|but|also|both|that|this|there|still|right|moment|whenever|days?|mornings?|unless|itself|more than|at least|either direction|in either direction|from where it (?:stood|was)|whichever (?:comes?|hits?|happens?|is) (?:first|sooner)|if neither (?:hits?|happens?|triggers?)|if that never happens|otherwise|or else|either way|i'?m|im|is|it|them|then|get out|"
                            r"take it off|selling it|sell it|sell them|buy them back|buy it back|let (?:it|them) expire|hold (?:it|them)? ?to expir\w*|the|a|an|at|by|or|and|if|when|once|on|to|for|of|out|with|so|just|"
                            r"i|my|me|we|per|each|every|a day|day|trade|trades|contract|contracts|position|does that make money|whats the win rate|numbers|backtest(?: this)?(?: for me)?|pls|please)\b", re.I)


def still_unplaced(d, text: str, known: list[str] | None, current: dict | None) -> list[str]:
    """Of the parts the strategy reader listed as not represented, those that still are: a part is placed once every
    quantity, time and condition inside it has a slot in the draft and nothing but connecting words is left."""
    from strata.ai import semantics as SM
    if not d.unsupported:
        return []
    res = check({"action": "new", "draft": d}, text, known, current, zone_known=True)
    unused = [i["item"] for i in res["issues"] if i["kind"] in ("unused", "mistyped", "misplaced", "entry_as_exit")]
    _n, ms = SM.tag(text, known)
    serial = (" ".join(c.expr for c in d.conditions) + " " + " ".join(s_.expr for s_ in d.sequence) + " " + " ".join(c.expr for c in d.exit_conditions) + " " + " ".join((c.text or "") for c in d.conditions)
              + (" weekday " + " ".join(d.weekdays) if d.weekdays else "") + (" underlying_target" if d.underlying_target_pct else "") + (" underlying_stop" if d.underlying_stop_pct else "")).lower()
    low = text.lower()
    keep = []
    for u in d.unsupported:
        if u.startswith("the order of the events") and not d.sequence and not re.search(r"\bfirst\b[^.;]{0,60}\bthen\b|\bthen\b(?! (?:get out|sell|exit|close|i'?m out|out\b|bail|dump|cover|take (?:it|them) off|buy (?:it|them) back|flat))", low):
            continue  # "... rallies 2%, then get out": the "then" introduces the exit, not a second step of the entry
        a = low.find(u.lower().strip())
        if a < 0:
            keep.append(u)
            continue
        b = a + len(u.strip())
        inside = [x for x in res["said"] if a <= x["pos"] < b and x["type"] != "ticker"]
        if any(x in unused or x.get("slot") is None for x in inside):
            keep.append(u)
            continue
        evs = [m for m in ms if m.kind == "event" and a <= _n.span(m.a, m.b)[0] < b and territory(low, _n.span(m.a, m.b)[0]) != "exit"]  # what is said about getting out is an exit, read above
        # a price against a level that fill_trade placed as a condition ("opened above the previous day's high": open > prev_high)
        rels = [m for m in ms if m.kind == "rel" and a <= _n.span(m.a, m.b)[0] < b and re.search(r"\b(?:open|price)\s*[<>]=?\s*(?:prev_high|prev_low|prev_close|open)", serial)]
        if any(not SM.holds(serial, m.value.get("key")) or m.value.get("key") in ("unsupported_event",) for m in evs):
            keep.append(u)
            continue
        rest = low[a:b]
        for x in sorted(inside, key=lambda x: -x["pos"]):
            rest = rest[:x["pos"] - a] + " " + rest[x["end"] - a:]
        for m in evs + rels:
            oa, ob = _n.span(m.a, m.b)
            rest = rest[:max(0, oa - a)] + " " * max(0, min(ob, b) - max(oa, a)) + rest[max(0, ob - a):]
        rest = re.sub(r"[^a-z' ]", " ", _FILLER_PHRASE.sub(" ", rest))
        words = [w.removesuffix("'s") for w in rest.split() if w not in SM.FILLER and len(w) > 2]
        known_words = (r"options?|calls?|puts?|price|premium|credit|value|gained|gains?|lost|loses?|up|down|profit|loss|stop|target|exit|sell|selling|sold|close|closed|bought|back|half|received|hits?|first|"
                       r"comes|time|pacific|eastern|central|has|have|had|when|whichever|neither|never|happens|expiration|expire|hold|them|noon|take|its|their|worth|paid|more|than|soon|mornings?|only|unless|where|was|entry|from|out|get|rall\w+|drops?|falls?|then|exp|"
                       r"can|could|itself|moved?|moves|moving|direction|stood|stands?|off|buy|been|goes|gone|against|costs?|otherwise|let|expires?|reaches|reached|trades?|trading|there|here")
        tickers_ = {k.lower() for k in (known or [])} | {"spy", "qqq", "iwm"}
        trade_words = r"buy|bought|buying|sell|sold|selling|dte|\d*dte|et|pt|ct|est|pst|cst|atm|otm|itm|same|day|each|every|trade|trades|contracts?|lots?|backtest|test|one|way|either|rallies|drops|if|or|and"
        if all(re.fullmatch(known_words, w) or w in tickers_ or re.fullmatch(trade_words, w) for w in words):
            continue  # everything in it has a place
        keep.append(u)
    return keep


def own_terms(text: str, known: list[str] | None = None) -> bool:
    """Does the message state terms of a trade or a question of its own (a time, a size, a condition), as opposed to leaning
    entirely on what is already on screen ("what if I bought calls instead?")?"""
    from strata.ai import semantics as SM
    if any(x["type"] in ("clock", "pct", "dollars", "minutes", "weekday", "year") or (x["type"] == "days") for x in said(text, known)):
        return True
    _n, ms = SM.tag(text, known)
    return any(m.kind == "event" or (m.kind == "anchor" and m.value.get("anchor") in ("open", "close", "end_at", "next_session")) for m in ms)


LEANS = re.compile(r"\b(?:those|these|such|the same|same) (?:days?|sessions?|setups?|conditions?|signals?|mornings?|trades?)\b|\b(?:trade|backtest|test|play|try) (?:it|that|this|them)\b|\bon those\b|\bwhen that happens\b|\binstead\b|\bsame (?:thing|idea|setup)\b", re.I)


# "on those mornings", "when that happens": the days of a question answered earlier in the conversation
# ("same day" alone is an expiration or a session, not a pointer: it took a spread "every morning ... same day" for one)
LEANS_DAYS = re.compile(r"\b(?:those|these|such) (?:days?|sessions?|setups?|conditions?|signals?|mornings?|afternoons?)\b|\b(?:the )?same (?:days|sessions|setups|conditions|signals|mornings)\b|\bon those\b|\bwhen that happens\b"
                        r"|\bthat (?:setup|signal|condition)\b", re.I)


def new_trade(text: str, known: list[str] | None = None) -> bool:
    """A trade described with terms of its own that does not lean on what is on screen ("on those days", "trade it",
    "instead"): it is read as written, not as a change to the question or study before it."""
    return is_trade(text) and own_terms(text, known) and not LEANS.search(text)


# ------------------------------------------------------------------------------------------ commands on what is on screen
_LEAD = r"^(?:(?:ok(?:ay)?|so|now|and|fine|alright|right|cool|nice|hmm+|well|then|wait|quick one)[.,! ]+)*"
_SHOW = r"(?:show|list|see|give|open|display|print|pull up|let me see|lemme see|can i see|what (?:were|are)|which (?:were|are|ones))"
_ROWS = (r"(?:days?|dates?|trades?|sessions?|ones|examples?|cases|losers?|winners?|loss(?:es)?|wins?|stops?|stopped|worst|best|biggest|ugliest|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|instances|occurrences|"
         r"them|those|these|list|rows|history)")


def command_of(text: str) -> dict | None:
    """A request ABOUT the result on screen, as the action the agent already has for it: list the days or trades behind it,
    say what a group of them has in common, say whether it could be chance. None for anything else. Read from what the
    message asks for, not from one sentence shape ("show me the losers", "let me see the ones that got stopped", "just
    luck?", "could that just be noise? theres not many of them", "are the losers mostly on any particular weekday?")."""
    low = re.sub(r"\s+", " ", text.lower().strip())
    body = re.sub(_LEAD, "", low)
    if re.search(r"could ?n[o']t|cannot|can'?t|not (?:be )?(?:tested|priced)|untested|missing|skipped|not taken|unexecuted|\bfills?\b|\bcompare\b|\bversus\b|\bvs\b", body):
        return None  # what could not be tested, how it was filled, a comparison: the agent has other actions for these
    if re.search(r"\bin common\b|\bwhat(?:'s| is| sets| makes)[^.?]{0,24}\b(?:different|apart|special)\b|\bmostly (?:on|in|when)\b|\b(?:any|a) pattern\b|\banything (?:interesting|odd|unusual|special|notable)\b[^.?]{0,40}\b(?:los\w+|win\w+|those|these|them|trades?|days?)\b", body):
        flt = "winning" if re.search(r"\bwin\w*|\bgood\b|\bworked\b", body) else "best" if re.search(r"\bbest\b", body) else "worst" if re.search(r"\bworst\b|\bugliest\b|\bbiggest los", body) else "losing"
        return {"action": "about_selection", "filter": flt, "question": text}
    if re.match(_SHOW + r"\b", body) and re.search(rf"\b{_ROWS}\b", body) and not re.search(r"\bhow often\b|\bwhat happens\b|\bbacktest\b", body):
        flt = ("worst" if re.search(r"\bworst\b|\bugliest\b|\bbiggest los\w+", body) else "best" if re.search(r"\bbest\b|\bbiggest (?:win|gain)\w*", body) else
               "losers" if re.search(r"\blos\w+|\bstopped\b|\bstops?\b|\bfailed\b|\bred\b|\bwrong\b|\bdidn'?t\b", body) else "winners" if re.search(r"\bwin\w*|\bworked\b|\bgreen\b|\bright\b", body) else "all")
        n = re.search(r"\b(\d+|ten|five|three|twenty)\b(?= (?:worst|best|biggest|ugliest|largest))", body)
        return {"action": "show_trades", "filter": {"worst": "loss", "losers": "loss", "best": "win", "winners": "win"}.get(flt, "all"), "explore_filter": flt, **({"limit": int(_num(n.group(1)) or 10)} if n else {})}
    luck = r"\bluck\w*|\brandom\w*|\bnoise\b|\bby chance\b|\bfluke\b|\bcoincidence\b|\b(?:statistically )?significan\w+|\btrust(?:worthy)?\b|\breliab\w+|\bfor real\b|\breal or\b|\btoo (?:thin|small|few)\b|\bnot (?:that )?many\b|\benough (?:trades|days|data|samples?|of them)\b|\bsample (?:size|is)\b|\bhold up\b|\bmeaningful\b"
    if re.search(luck, body) and (len(body.split()) <= 16 or re.search(r"\b(?:that|this|it|those|these|the (?:result|number|edge|difference|gap|thing|pattern))\b", body)) and not re.search(r"\bhow often\b|\bbacktest\b|\bbuy\b|\bsell\b", body):
        return {"action": "robustness"}
    return None


# ------------------------------------------------------------------------------------------- a change to the current trade
_REMOVE = r"(?:kill|remove|drop|lose|ditch|scrap|forget|skip|nix|axe|cut out|take (?:out|off|away)|get rid of|without|no more|no)"
_SLOT_WORDS = (("stop", r"stops?(?: ?loss(?:es)?)?|sl"), ("target", r"(?:profit )?targets?|take ?profits?|tp|pt"), ("exit_time", r"time ?(?:stop|exit)|exit time|timed exit"),
               ("hold_to_expiration", r"hold(?:ing)? to exp\w*"), ("sizing", r"sizing|risk (?:rule|limit)|daily (?:loss )?(?:stop|limit)"), ("conditions", r"conditions?|filters?"))
TICKER_NICK = {"the q's": "QQQ", "the qs": "QQQ", "qs": "QQQ", "q's": "QQQ", "cubes": "QQQ", "the nasdaq": "QQQ", "nasdaq": "QQQ", "nq": "QQQ", "spyder": "SPY", "spiders": "SPY", "spooz": "SPY", "spoos": "SPY", "the s&p": "SPY", "s&p": "SPY",
               "the russell": "IWM", "russell": "IWM", "small caps": "IWM", "smallcaps": "IWM", "rut": "IWM"}


def named_ticker(text: str, known: list[str] | None = None) -> str | None:
    """The one stored ticker a message names, by its symbol or by what traders call it ("the q's", "small caps")."""
    low = " " + text.lower() + " "
    found = [x["value"] for x in said(text, known) if x["type"] == "ticker" and x["value"] != "VIX"]
    for nick, sym in TICKER_NICK.items():
        if re.search(rf"(?<![\w']){re.escape(nick)}(?![\w'])", low):
            found.append(sym)
    found = list(dict.fromkeys(found))
    return found[0] if len(found) == 1 else None


_OPENS = (r"(?:buy|buying|bought|sell|selling|sold|short|shorting|write|writing|long|switch(?:ing)? to|chang\w+ (?:it |that |this )?to|make (?:it|that|this)|go(?:ing)? with|"
          r"what if (?:i|we) (?:bought|sold|did|used|traded|ran|tried)|how about|what about|try|use|do)")
_INSTR = r"(?:calls?|puts?|straddles?|strangles?|(?:iron )?condors?|(?:iron )?butterfl\w+|iron fl(?:y|ies)|spreads?|verticals?)"


def named_instrument(text: str) -> str | None:
    """The position a message says to OPEN ("buy a put instead", "sell an iron condor", "what if I bought calls"), as a
    structure name. None when it names none, or only speaks of the position on screen ("where it was when I sold the
    spread", "buy back the short put at half", "sell them at noon")."""
    from strata.ai import nl_rules as NL
    low = text.lower()
    for m in re.finditer(rf"\b{_OPENS}\b(?! (?:it|them|that|this|out|half|the rest|everything|all|back|to close)\b)(?:\s+[\w.%$'-]+){{0,5}}?\s+{_INSTR}\b", low):
        tail = re.match(r"[^.;,:!?]{0,26}", low[m.end():])  # "put" may go on: "put credit spread", "put spreads"
        seg = low[m.start():m.end() + (tail.end() if tail else 0)]
        before = low[:m.start()]
        if re.search(r"\b(?:when|where|after|since|once|as|that|what) (?:i|we|you) $", before) or re.search(r"\bback\b|\bto close\b|\bthe (?:short|long|other|same) (?:put|call|leg|strike)\b", seg):
            continue  # the trade already on screen, or one leg of it
        if re.match(r"short|long", m.group(0)) and re.search(r"\b(?:the|my|a|an|this|that|each|every|its|your) $", before):
            continue  # "the short put": a leg, not an order to short one
        if re.search(r"(?:\bnot|\bno|n't|\bnever|\bwithout|\binstead of|\brather than|\bstop|\bquit) $", before):
            continue  # "not buying a put": what it is NOT
        if re.search(r"\b(?:the|my|this|that|each|every|same) (?:\w+ )?(?:puts?|calls?|spreads?|condors?|straddles?|strangles?|position|trade)\b", seg) and not re.search(r"\binstead\b", low):
            continue  # "sell the spread at 10:30", "make the put spread 10 wide": the position on screen, changed in a part
        st = NL.find_structure(seg)
        if st:
            return st
    return None


def instrument_words(structure: str | None, verb: bool = True) -> str:
    v, what = _STRUCT_WORDS.get(structure or "", ("trade", (structure or "that position").replace("_", " ")))
    what = what.format(n="", sym="", s="s", ies="ies").replace("  ", " ").strip()
    what = what.split(" (")[0]
    return f"{v.lower()} {what}" if verb else f"{'selling' if v == 'Sell' else 'buying'} {what}"


def entry_time_list(text: str, known: list[str] | None = None) -> list[dict] | None:
    """The clock times of a trade message that are listed together as times to get IN ("take it at 10:30, noon and 1:30"),
    when there are two or more and the message does not ask to compare them. One trade has one entry time a day."""
    low = text.lower()
    if re.search(r"\bcompar\w+|\bversus\b|\bvs\.?\b|\bsweep\b|\beach of\b|\bwhich (?:one|time|entry)\b|\bbetter\b|\bbest\b|\btest (?:each|them|all)\b|\bseparately\b", low):
        return None
    S = [x for x in said(text, known, trade=True) if x["type"] == "clock"]
    for m in re.finditer(r"\b(?:noon|midday)\b", low):  # a time by its name, which the inventory leaves to the readers
        if not any(x["pos"] <= m.start() < x["end"] for x in S):
            S.append({"type": "clock", "value": 720.0, "text": m.group(0), "pos": m.start(), "end": m.end()})
    S.sort(key=lambda x: x["pos"])
    best: list[dict] = []
    run: list[dict] = S[:1]
    for a_, b_ in zip(S, S[1:]):
        if re.fullmatch(r"\s*(?:,|,?\s*and|,?\s*or|&|,?\s*then|,?\s*and then|,?\s*and again)?\s*(?:at\s+|again at\s+|another (?:one )?at\s+|one at\s+)?", low[a_["end"]:b_["pos"]]):
            run.append(b_)
        else:
            best, run = (run if len(run) > len(best) else best), [b_]
    best = run if len(run) > len(best) else best
    if len(best) < 2 or len({x["value"] for x in best}) < 2:
        return None
    if territory(low, best[0]["pos"]) == "exit" or re.search(r"\b(?:between|from)\s+(?:about\s+)?$", low[:best[0]["pos"]]):
        return None  # times to get out, or the two ends of a window
    return best


def part_without_value(text: str, d) -> dict | None:
    """A follow-up on a trade that asks for a part and gives it no value ("put a stop on it", "add a profit target", "use a
    time stop"): a question naming what is missing, with ways to say it. No level is ever filled in for the user."""
    low = text.lower()
    if re.search(r"\d|\bhalf\b|\bdoubl\w+|\btripl\w+|\bquarter\b", low) or re.search(rf"\b{_REMOVE}\b", low):
        return None
    if not re.search(r"\b(?:add|place|put|set|use|with|want|need|include|give it|throw in|how about|what about|try|should have|needs?|let'?s have)\b", low):
        return None
    sold = getattr(d, "structure", None) in SHORT_PREMIUM
    sym = getattr(d, "symbol", None) or "the underlying"
    if re.search(r"\bstops?(?:[- ]loss(?:es)?)?\b", low) and not re.search(r"\btime[- ]stop\b", low):
        ex = (f"\"stop when the loss is 100% of the credit\", \"out if {sym} falls 0.5% from the entry\"" if sold else f"\"stop at 30% of the premium\", \"out if {sym} falls 0.5% from the entry\"")
        return {"action": "clarify", "text": f"A stop needs a level, and your message gives none, so nothing was changed. What should the stop be? For example {ex}.",
                "pending": {"kind": "part_value", "prefix": "stop at"}}
    if re.search(r"\b(?:profit )?targets?\b|\btake[- ]?profits?\b|\btp\b", low):
        ex = "\"take profit at 50% of the credit\"" if sold else "\"take profit at 30% of the premium\""
        return {"action": "clarify", "text": f"A profit target needs a level, and your message gives none, so nothing was changed. What should it be? For example {ex}, or \"out when {sym} is up 0.5% from the entry\".",
                "pending": {"kind": "part_value", "prefix": "take profit at"}}
    if re.search(r"\btime[- ]stop\b|\btime exit\b|\bexit time\b", low):
        return {"action": "clarify", "text": "At what time, or after how long? For example \"out at 2 pm\" or \"out 45 minutes after getting in\". Nothing was changed.",
                "pending": {"kind": "part_value", "prefix": "exit"}}
    return None


_DIM_TYPES = {"entry_time": {"clock"}, "exit_time": {"clock"}, "dte": {"dte", "days", "number"}, "strike": {"delta", "pct", "strikes", "number"}, "premium_target": {"pct", "number"},
              "premium_stop": {"pct", "number"}, "underlying_target": {"pct", "number"}, "underlying_stop": {"pct", "number"}, "exit_minutes": {"minutes", "number", "clock"}, "symbol": {"ticker"}}


def experiment_fits(text: str, ex: dict, known: list[str] | None = None) -> bool:
    """Is every list of values the comparison reader found really a list of that thing? A list that cuts through something
    the message types otherwise (the "30" of "10:30" as a delta) is a misreading: the comparison is then not what was asked."""
    low = text.lower()
    S = said(text, known, trade=True)
    for dm in ex.get("dims") or []:
        t = str(dm.get("text") or "").lower()
        a = low.find(t) if t else -1
        if a < 0:
            continue
        b = a + len(t)
        ok = _DIM_TYPES.get(dm.get("field"))
        for x in S:
            if x["pos"] < b and x["end"] > a and ok is not None and x["type"] not in ok and (x["pos"] < a or x["end"] > b or x["type"] == "clock"):
                return False
    return True


def compare_clause(low: str, symbol: str | None):
    """The part of a follow-up that asks for the result beside the version before ("and put it next to the SPY numbers",
    "compare with the original", "then both results together"), as a match, or None."""
    ref = rf"(?:the\s+)?(?:{re.escape((symbol or 'spy').lower())}|original|first|previous|earlier|last|old|current|one before|before)(?:\s+(?:numbers?|results?|one|version|run|trade|stats?|test|backtest))?"
    return re.search(rf"(?:,?\s*\b(?:and then|and|then)\b)?\s*\b(?:(?:place|put|show|set|line|stack|lay|see)\s+(?:it|that|them|those|both|the (?:two|results?|numbers))\s+(?:up\s+)?(?:next to|beside|alongside|against|side by side with|up against)|"
                     rf"compare(?:\s+(?:it|that|them))?\s+(?:with|to|against)|(?:versus|vs\.?|next to|alongside|side by side with))\s+{ref}(?![\w'])"
                     r"|(?:[,.]?\s*\b(?:and then|and|then)\b)?\s*\b(?:(?:show|see|with|give me)\s+)?both (?:results?|numbers?|of them)?\s*(?:together|side by side)\b|\bside by side\b", low)


def edit_trade(d, text: str, known: list[str] | None = None) -> dict | None:
    """A short message read as a change to the options trade on screen, by what it names: a part to remove ("kill the
    stop"), another ticker ("same on qqq"), a time, a size or a percentage with its role ("move the exit to 1245", "add a
    stop at 30%", "2 lots"), a period ("what about 2024"), calls for puts. Returns {"draft", "changed", "notes"} when every
    quantity in the message found its part, None when the message is not such a change (the other readers then get it)."""
    from strata.ai import nl_rules as NL
    text = text.strip()
    low = text.lower()
    # "... and put it next to the SPY numbers", "compare with the one before", "then both results together": the change is
    # run and shown beside the version it changes. The clause names no part of the trade, so it is taken out before reading.
    cm = compare_clause(low, d.symbol)
    compare = False
    if cm:
        text = (text[:cm.start()] + " " + text[cm.end():]).strip(" ,.;")
        low, compare = text.lower(), True
        if not low:
            return None
    if len(low.split()) > 28:
        return None
    if new_trade(text, known) and len(low.split()) > 12:
        # a longer message with terms of its own is a trade of its own, unless it opens no position and speaks of the one on
        # screen ("take it off as soon as the straddle is worth 40% more than I paid")
        speaks_of_it = re.search(r"\b(?:it|them|the (?:straddle|strangle|spread|condor|butterfly|position|trade|calls?|puts?|contracts?|options?))\b", low)
        if named_instrument(text) is not None or not speaks_of_it or re.search(r"\b(?:buy|buying|bought|sell|selling|sold|short|long)\s+(?:\d+\s+|an?\s+|one\s+|two\s+)?(?:[a-z]{2,5}\s+)?(?:\d+\s*dte\s+|0dte\s+|atm\s+|otm\s+)*(?:calls?|puts?|straddles?|strangles?|spreads?|condors?)\b", low):
            return None
    if re.search(r"\bcompar\w+|\bversus\b|\bvs\.?\b|\bsweep\b|\beach of\b|\bsensitiv\w+|\brange of\b|\bfrom \$?\d[\d.]* ?%? to \$?\d", low):
        return None  # several values to compare are an experiment, not a change
    d2 = d.model_copy(deep=True)
    changed: list[str] = []
    notes: list[str] = []
    short = bool(re.match(r"short_|bull_put|bear_call|iron_", d.structure or ""))
    # ---- a part removed
    rm = re.search(rf"\b{_REMOVE}\b\s+(?:the |that |my |any )?(?P<w>[a-z' ]{{2,26}})", low)
    removed = False
    if rm:
        w = rm.group("w")
        for slot, pat in _SLOT_WORDS:
            if re.match(rf"(?:{pat})\b", w):
                if slot == "stop":
                    if d2.premium_stop_pct is None and d2.underlying_stop_pct is None and d2.underlying_stop_abs is None:
                        return {"draft": d, "changed": [], "notes": ["There is no stop in the current version."], "noop": True}
                    d2.premium_stop_pct = d2.underlying_stop_pct = d2.underlying_stop_abs = None
                    changed += ["premium_stop_pct", "underlying_stop_pct"]
                elif slot == "target":
                    d2.premium_target_pct = d2.underlying_target_pct = d2.underlying_target_abs = None
                    changed += ["premium_target_pct", "underlying_target_pct"]
                elif slot == "exit_time":
                    d2.exit_time = d2.exit_time_stated = None
                    changed.append("exit_time")
                elif slot == "hold_to_expiration":
                    d2.hold_to_expiration = False
                    changed.append("hold_to_expiration")
                elif slot == "sizing":
                    d2.sizing = None
                    changed.append("sizing")
                elif slot == "conditions":
                    d2.conditions, d2.weekdays = [], None
                    changed.append("conditions")
                removed = True
                break
        if not removed:
            key = next((k for k, pat in (("vix", r"vix"), ("gap", r"gap"), ("previous_session", r"(?:previous|prior|yesterday)\w*(?: day)?"), ("vwap", r"vwap"), ("weekday", r"weekday|day of the week|monday|tuesday|wednesday|thursday|friday"))
                        if re.match(rf"(?:{pat})\b", w)), None)
            if key == "weekday" and d2.weekdays:
                d2.weekdays, removed = None, True
                changed.append("weekdays")
            elif key and any(key in c.expr for c in d2.conditions):
                d2.conditions = [c for c in d2.conditions if key not in c.expr]
                removed = True
                changed.append("conditions")
    items = [x for x in said(text, known, trade=True) if not x.get("negated")]
    io = low.find("instead of")
    if io >= 0:
        items = [x for x in items if x["pos"] < io]  # "use 7DTE instead of 0DTE": what follows "instead of" is what goes
    kinds = [x["type"] + ":" + str(x.get("about") or "") for x in items if x["type"] in ("dte", "delta", "contracts", "width", "strikes")]
    if len(kinds) != len(set(kinds)):
        return None  # two values for one part: a list to compare, which the experiment reader takes
    used: set[int] = set()
    name = lambda t_: {"09:30": "open", "16:00": "close"}.get(t_, t_)  # noqa: E731
    # ---- another ticker
    tk = named_ticker(text, known)
    if tk and tk != d2.symbol and not removed:
        d2.symbol, d2.symbol_stated = tk, True
        changed.append("symbol")
    # ---- another position altogether ("buy a put instead", "make it an iron condor"): the parts that still mean something
    # are kept, and the check (intent.check) says what was carried over
    ni = named_instrument(text)
    single = ("long_call", "long_put", "short_call", "short_put")
    if ni and ni != d2.structure and not (ni in single and d2.structure in single and not re.search(r"\b(?:sell|selling|sold|short|shorting|write|writing|buy|buying|bought|long)\b", low)):
        if ni in single or "straddle" in ni or "strangle" in ni:
            d2.width = None  # one leg, or two with no wing: a spread's width means nothing here
        d2.structure = ni
        changed.append("structure")
    # ---- calls for puts and back
    if re.search(r"\b(?:calls?|puts?)\b", low) and d2.structure in ("long_call", "long_put", "short_call", "short_put") and "structure" not in changed and not re.search(r"\bspreads?\b|\bcondor|\bstraddle|\bstrangle", low):
        want = ("short_" if d2.structure.startswith("short") else "long_") + ("put" if re.search(r"\bputs?\b", low) and not re.search(r"\bcalls?\b", low) else "call")
        if re.search(r"\bcalls?\b", low) and re.search(r"\bputs?\b", low):
            want = d2.structure
        if want != d2.structure:
            d2.structure = want
            changed.append("structure")
    # ---- things with a role
    for i, s in enumerate(items):
        t, v = s["type"], s["value"]
        if t == "ticker":
            used.add(i)
            continue
        if t == "clock":
            role = role_of(s, text, short=short)
            if role is None and re.search(r"\b(?:exit|out|close|sell|flat|off|stop)\w*\b", low):
                role = "exit_time"
            if role is None and re.search(r"\b(?:entry|enter|in|buy|open|start)\w*\b", low):
                role = "entry_time"
            if role:
                setattr(d2, role, name(s["et"]))
                setattr(d2, role + "_stated", s.get("stated") or s["text"])
                changed.append(role)
                used.add(i)
        elif t == "pct" and s.get("about") in ("option", "underlying"):
            role = role_of(s, text, short=short, bias=bias_of(d2.structure))
            has_stop, has_tgt = bool(re.search(r"\bstop\w*|\bsl\b", low)), bool(re.search(r"\btarget\w*|\btp\b|\bprofit\w*", low))
            if role is None and has_stop and not has_tgt:
                role = ("premium_" if s["about"] == "option" else "underlying_") + "stop"
            if role is None and has_tgt and not has_stop:
                role = ("premium_" if s["about"] == "option" else "underlying_") + "target"
            if role:
                setattr(d2, role + "_pct", float(v))
                changed.append(role + "_pct")
                used.add(i)
            elif s["about"] == "underlying" and territory(low, s["pos"]) != "exit" and not re.search(r"\b(?:exit|out|sell|close|stop|target|after entry|from entry)\b", low[max(0, s["pos"] - 30):s["end"] + 24]):
                # V10: the size of a condition the trade already has: "what if the gap is only 0.5%", "make the drop 2%"
                near = low[max(0, s["pos"] - 44):s["end"] + 18]
                for var, pat in (("gap_pct", r"gaps?|gapp\w+|open(?:s|ed|ing)?"), ("previous_session.return_pct", r"yesterday\w*|previous|prior|day before|(?:red|green|down|up) day|drop|decline|fall|rally|rise"),
                                 ("from_open_pct", r"from (?:the |its )?open|off the open"), ("change_pct", r"on the day")):
                    idx = [k for k, c in enumerate(d2.conditions) if re.match(rf"^\s*{re.escape(var)}\s*(?:<=|>=|<|>)\s*-?\d", c.expr)]
                    mm = re.match(rf"^\s*{re.escape(var)}\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$", d2.conditions[idx[0]].expr) if len(idx) == 1 else None
                    if mm and re.search(rf"\b(?:{pat})\b", near):
                        c = d2.conditions[idx[0]]
                        neg = float(mm.group(2)) < 0 or mm.group(1) in ("<", "<=")
                        c.expr = f"{var} {'<=' if neg else '>='} {(-abs(v) if neg else abs(v)):g}"
                        c.text = (re.sub(r"\d+(?:\.\d+)?", f"{abs(v):g}", c.text, count=1) if c.text and re.search(r"\d", c.text) else c.text) or c.expr
                        changed.append("conditions")
                        used.add(i)
                        break
        elif t == "contracts":
            d2.quantity = int(v)
            if d2.sizing is not None and d2.sizing.mode == "contracts":
                d2.sizing = d2.sizing.model_copy(update={"value": float(v)})
            changed.append("quantity")
            used.add(i)
        elif t == "dte":
            d2.dte = int(v)
            changed.append("dte")
            used.add(i)
        elif t == "delta":
            from strata.ai.draft import DraftStrike
            d2.strike = DraftStrike(method="delta", value=float(v) / 100)
            changed.append("strike")
            used.add(i)
        elif t == "strikes":
            from strata.ai.draft import DraftStrike
            d2.strike = DraftStrike(method="strikes_otm", value=float(v))
            changed.append("strike")
            used.add(i)
        elif t == "width":
            d2.width = float(v)
            changed.append("width")
            used.add(i)
        elif t == "year":
            s_, e_, _note = NL.find_period(low)
            if not (s_ or e_) and not s.get("date") and v != "span":
                from datetime import date as _date
                s_, e_ = _date(int(v), 1, 1), _date(int(v), 12, 31)  # "what about 2025": that year
            if s_ or e_:
                d2.start, d2.end = s_, e_
                changed.append("period")
                used.add(i)
        elif t == "minutes" and re.search(r"\bhold\w*|\bmax\b|\bafter\b|\blater\b", low):
            if re.search(r"\blater\b|\bearlier\b", low):
                continue  # a time moved against itself: the conversation layer reads it (converse.relative_shift)
            d2.hold_minutes = int(v)
            changed.append("hold_minutes")
            used.add(i)
    before = d2.model_dump()
    notes += fill_trade(d2, text, known)  # sizes of the trade, held to expiration, conditions the question reader knows
    if d2.model_dump() != before:
        changed.append("more")
    left = [s for i, s in enumerate(items) if i not in used and s["type"] not in ("ticker",)]
    if left:
        res = check({"action": "modify", "draft": d2}, text, known, {"draft": d}, zone_known=True)
        if any(i_["kind"] in ("unused", "mistyped", "misplaced") for i_ in res["issues"]):
            return None
    if not changed:
        return None
    return {"draft": d2, "changed": list(dict.fromkeys(c for c in changed if c != "more")) or ["conditions"], "notes": notes, **({"compare": True} if compare else {})}


def unread_phrases(text: str, unread: list[dict]) -> list[str]:
    """The words a reading did not use, as the phrases they were written in: neighbouring unread words are shown with the
    small words between them ("retail sentiment is euphoric", not "retail", "sentiment", "euphoric")."""
    spans = sorted((u["span"][0], u["span"][1]) for u in unread if u.get("span"))
    out: list[list[int]] = []
    for a, b in spans:
        if out and a - out[-1][1] <= 14 and not re.search(r"[,;.?!]", text[out[-1][1]:a]) and len(text[out[-1][1]:a].split()) <= 2:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [text[a:b].strip() for a, b in out] or [u["text"] for u in unread]


# ----------------------------------------------------------------------------------- the reading, with its parts to click
def reading_parts(frame: dict, stored: list[str] | None = None) -> list[dict]:
    """The plain sentence of a question frame (semantics.describe) cut into pieces, the important ones marked so that the
    interface can offer to change them in place: the ticker, the period, each condition. Every piece is
    {"t": "text" | "ticker" | "period" | "condition", "s": the words}; a marked piece also says what can be done with it
    (`say`: the follow-up message that makes the change, which goes through the same reader as anything typed)."""
    from strata.ai import semantics as SM
    sent = SM.describe(frame)
    if not sent:
        return []
    marks: list[tuple[int, int, dict]] = []
    sym = frame.get("symbol") or ""
    i = sent.find(sym) if sym else -1
    others = [x for x in (stored or ["SPY", "QQQ", "IWM"]) if x != sym and x.isupper() and x not in ("VIX",) and not x.startswith("DEMO")]
    if i >= 0 and len(frame.get("symbols") or [sym]) <= 1:
        marks.append((i, i + len(sym), {"t": "ticker", "choices": [{"label": x, "say": f"use {x} instead"} for x in others]}))
    words = {"gap": "gap", "prev_return": "yesterday", "prev_otc": "yesterday", "vix": "vix", "weekday": "weekday", "first_candle": "candle", "up_streak": "streak", "down_streak": "streak"}
    for c in frame.get("conditions") or []:
        txt = SM.describe_condition(c)
        j = sent.find(txt)
        if j >= 0 and txt:
            w = words.get(c.get("key"))
            marks.append((j, j + len(txt), {"t": "condition", **({"choices": [{"label": "Remove this condition", "say": f"remove the {w} condition"}]} if w else {})}))
    per = SM._describe_period(frame.get("period")).strip()
    year = __import__("datetime").date.today().year
    period_choices = [{"label": "The whole stored history", "say": "use the whole history"}] + [{"label": f"Since {y}", "say": f"since {y}"} for y in (2016, 2020, year - 2)] + [{"label": f"Only {year - 1}", "say": f"in {year - 1}"}]
    tail = None
    if per:
        j = sent.rfind(per)
        if j >= 0:
            marks.append((j, j + len(per), {"t": "period", "choices": period_choices}))
    else:
        tail = {"t": "period", "s": "over the whole stored history", "choices": period_choices[1:], "default": True}
    out, pos = [], 0
    for a, b, extra in sorted(marks, key=lambda m: m[0]):
        if a < pos:
            continue
        if a > pos:
            out.append({"t": "text", "s": sent[pos:a]})
        out.append({"s": sent[a:b], **extra})
        pos = b
    rest = sent[pos:]
    if tail is not None:
        out.append({"t": "text", "s": rest.rstrip(".") + ", "})
        out.append(tail)
        out.append({"t": "text", "s": "."})
    elif rest:
        out.append({"t": "text", "s": rest})
    return out


# ------------------------------------------------------------------------- simple or complex: what waits for a go-ahead
def complexity(route: dict) -> list[str]:
    """The parts of a reading that a reader has to get right beyond "what, on which ticker": each condition, a second step,
    a set clock time, every exit after the first, a sizing rule, a multi-leg position. Returned as short names.

    V10 brief, section 5: a simple question is answered at once; a complex one shows its reading and waits. `is_complex`
    draws the line."""
    from strata.ai import semantics as SM
    parts: list[str] = []
    d, f = route.get("draft"), route.get("frame")
    if f and f.get("kind") != "strategy":
        for c in f.get("conditions") or []:
            parts.append(SM.describe_condition(c) or "a condition")
        if f.get("then"):
            parts.append("a second step")
        rel = (f.get("outcome") or {}).get("rel") or {}
        if any(str(rel.get(k) or "").split(":")[0] in ("at", "high_by", "low_by", "high_from", "low_from") and str(rel.get(k)).split(":", 1)[1] not in ("09:30", "16:00") for k in ("a", "b")):
            parts.append("a set time")  # "before the close" and "at the open" are not: every session has those
        if re.fullmatch(r"\d\d:\d\d", str(f.get("start") or "")):
            parts.append("a set time")  # where it starts; a window that only ends early ("the first 2 hours") is not counted
        return parts
    if d is not None and getattr(d, "kind", None) == "options":
        parts += [c.text or c.expr for c in d.conditions]
        parts += [st.name or st.expr for st in d.sequence]
        exits = [k for k in ("premium_target_pct", "premium_stop_pct", "underlying_target_pct", "underlying_stop_pct", "hold_minutes", "hold_sessions") if getattr(d, k, None)]
        if d.exit_time and d.exit_time != "close":
            exits.append("exit_time")
        parts += [f"exit: {k}" for k in exits[1:]]  # one exit is part of any trade
        parts += [f"exit rule: {c.expr}" for c in d.exit_conditions]
        if d.partials:
            parts.append("scaling out")
        if d.adds:
            parts.append("adding to the position")
        if getattr(d, "sizing", None) is not None:
            parts.append("a sizing rule")
        if d.structure not in ("long_call", "long_put", "short_call", "short_put"):
            parts.append("more than one leg")
        return parts
    if d is not None:  # a study draft
        parts += [c.text or c.expr for c in d.conditions]
        parts += [st.name or st.expr for st in d.sequence]
    return parts


COMPLEX_FROM = 2


def is_complex(route: dict) -> bool:
    return len(complexity(route)) >= COMPLEX_FROM


def describe_graph(g: dict) -> str | None:
    """A research plan as one plain sentence (what it conditions on and what it measures). The steps are the technical view;
    the notebook shows this (V10: the reading is never hidden behind a graph)."""
    try:
        it = from_graph(g)
        sym = it.get("symbol") or "the ticker"
        conds = [c["text"] + (" (the session before)" if c.get("when") == "previous_session" and not re.search(r"previous|prior|before|last|yesterday|overnight", c["text"]) else "")
                 for c in it["conditions"] if c.get("text")]
        oc = it.get("outcome") or {}
        frm = oc.get("from") or "open"
        start = {"open": "its open", "close": "that day's close"}.get(frm, f"its price at {frm} New York time")
        hz = str(oc.get("horizon") or "session")
        until = "by the close" if hz == "session" else (f"{hz.split(':')[1]} sessions later" if hz.startswith("sessions:") else f"by {hz.split(':', 1)[1]} New York time" if hz.startswith("time:") else "")
        side = {"up": "above", "down": "below"}.get(oc.get("direction") or "")
        size = oc.get("size")
        if oc.get("measure") == "touch" and size and side:
            what = f"how often it trades {size:g}% {side} {start} at some point {until}".rstrip()
        elif side and size:
            what = f"how often it is {size:g}% or more {side} {start} {until}".rstrip()
        elif side:
            what = f"how often it is {side} {start} {until}, and by how much".replace(" ,", ",")
        else:
            what = f"how it moves from {start} {until}".rstrip()
        names = {a if isinstance(a, str) else (a.get("op") or a.get("kind") or "") for a in (it.get("analyses") or [])}
        tail = " It is also checked year by year and against chance." if any("robust" in n for n in names) else ""
        per = [x for x in (it.get("period") or []) if x]
        when = (f" From {per[0]}" + (f" to {per[1]}." if len(per) > 1 else ".")) if per else ""
        return f"On {sym}" + (", when " + " and ".join(conds) if conds else "") + f": {what}." + tail + when
    except Exception:  # noqa: BLE001  (a sentence that cannot be built leaves the card as it was)
        return None


# -------------------------------------------------------------------------------- what the stored data cannot answer
# The product's limits, in one place (docs/v10_evidence/eval/product_brief.md: "What it cannot do"). A message that
# needs one of these is declined with the reason, whether it is a new request or a change to the one on screen, and the
# nearest request that can be answered is offered. Nothing is run in its place.
_OOS_EVENT = (r"\b(?:fomc|fed(?:eral reserve)?(?: rate)? (?:day|meeting|decision|announcement|week|minute)s?|fed days?|(?:on|around|after|before|into|ahead of) (?:the )?fed\b|powell|cpi|ppi|pce|nfp|non[- ]?farm|payrolls?"
              r"|jobs (?:report|day|number|data|friday)s?|earnings|opex|op-ex|(?:monthly |quarterly |weekly )?(?:options? )?(?:expiration|expiry) (?:week|friday)s?|(?:triple|quad(?:ruple)?) witching|witching|holiday(?: week)?s?"
              r"|half[- ]days?|ex[- ]?div(?:idend)?\w*|dividend (?:day|date)s?|election(?: day| week)?s?|rebalanc\w+|window dressing|news (?:day|event)s?|(?:on|after|before) (?:the )?news\b|economic (?:data|calendar|releases?|events?)"
              r"|macro (?:event|data|day)s?|gdp|retail sales|treasury auctions?)\b")
_OOS_IND = (r"\b(?:rsi|macd|stochastics?|bollinger(?: bands?)?|(?:simple |exponential )?moving averages?|\d{1,3}[- ]?(?:day|d|week|wk|period)[- ]?(?:moving average|ma|sma|ema)\b|\d{1,3}[- ]?(?:dma|sma|ema)\b"
            r"|(?:20|50|100|150|200)[- ]?(?:day|d)\b(?! (?:high|low|range|returns?|period|window|history|lookback|stretch|streak|run|forward|later|ago|after|before))(?<![-\d] \d\d[- ]day)"
            r"|golden cross|death cross|fib(?:onacci|s)?(?: (?:level|retracement)s?)?|ichimoku|adx|atr|average true range|obv|keltner|supertrend|pivot (?:point|level)s?|52[- ]week (?:high|low)s?|all[- ]time highs?|ath)\b")
_OOS_QUOTE = (r"\b(?:(?:at|on|to|hit(?:ting)?|pay(?:ing)?|paid|receiv(?:e|ed|ing)|get(?:ting)?|lift(?:ing)?|cross(?:ing)?|from|buy(?:ing)? (?:at|on)|sell(?:ing)? (?:at|on)|fill(?:ed|s)? (?:at|on)) the (?:bid|ask|offer|mid(?:point)?)\b"
              r"|bid[-/ ](?:and[- ])?ask|bid and (?:the )?ask|nbbo|mid[- ]?(?:point|price)s?|quoted (?:price|spread|market)s?|cross(?:ing)? the spread|(?:bid|ask) (?:price|side)s?)\b")
_OOS_ADVICE = (r"\bshould i\b|\bshall i\b|\bdo you (?:think|recommend|reckon)\b|\bwould you (?:buy|sell|hold|recommend|take)\b|\bis (?:it|this|now) a good (?:time|idea|trade|buy|entry|moment)\b|\bwhat (?:should|would) (?:i|you) (?:do|buy|sell)\b"
               r"|\b(?:worth|safe) (?:buying|selling|holding)\b|\btrying to decide (?:whether|if)\b|\bgood idea to\b|\bwhat(?:'s| is) your (?:view|take|call|opinion)\b")
_OOS_FUTURE = (r"\b(?:will|gonna|going to)\b[^.?!]{0,70}\b(?:tomorrow|tonight|this (?:week|month|afternoon)|next (?:week|month|year|session)|by (?:year|month)[- ]end|eoy)\b"
               r"|\b(?:is|will|does) tomorrow\b|\btomorrow(?:'s)? (?:going to|more likely|likely|gonna|will)\b|\bwhat(?:'s| is| will) (?:going to |gonna )?happen (?:tomorrow|next week|today|tonight)\b|\bprice target\b|\bwhere (?:will|is) \w+ (?:be|going|headed) (?:by|in|next|tomorrow)")
_HISTORICAL = r"\b(?:usually|typically|historically|tend(?:s|ed)? to|how often|on average|in the past|has it|have they|normally|generally|how many times|what (?:share|percent\w*|fraction))\b"


def _nearest_condition(text: str, known: list[str] | None, sym: str) -> str | None:
    """A question Strata can answer, built from the conditions a forecast names ("spy's up six days straight"): how often
    the next session closed higher after them. None when the message names no condition the reader knows."""
    from strata.ai import semantics as SM
    try:
        _, ms = SM.tag(tidy(text), known or list(SM.STORED_DEFAULT))
    except Exception:  # noqa: BLE001
        return None
    parts = []
    for m in ms:
        if m.kind != "event":
            continue
        k, op, v = m.value.get("key"), m.value.get("op"), m.value.get("value")
        if k in ("up_streak", "down_streak") and v:
            parts.append(f"after {int(v)} {'up' if k == 'up_streak' else 'down'} days in a row")
        elif k == "prev_return":
            parts.append(f"after a {abs(v):g}% {'down' if op in ('<', '<=') else 'up'} day" if v else f"after {'a down' if op in ('<', '<=') else 'an up'} day")
        elif k == "vix" and v and op in (">", ">=", "<", "<="):
            parts.append(f"when VIX is {'above' if op in ('>', '>=') else 'below'} {v:g}")
    if not parts:
        return None
    return f"How often does {sym} close higher the next day " + " and ".join(dict.fromkeys(parts)) + "?"


def _without(text: str, a: int, b: int) -> str | None:
    """The message without the clause that holds text[a:b], when what is left still reads as a request."""
    cuts = [0] + [m.end() for m in re.finditer(r"[,;:]\s+|\.\s+|\s+-\s+|\s+\band\b\s+(?=only|just|on\b)|\s+but\s+", text)] + [len(text)]
    lo = max(c for c in cuts if c <= a)
    hi = min(c for c in cuts if c >= b)
    rest = (text[:lo].rstrip(" ,;:-") + " " + text[hi:].lstrip(" ,;:-")).strip() if lo > 0 or hi < len(text) else ""
    rest = re.sub(r"\s+", " ", rest).strip(" ,;:-")
    if len(rest.split()) >= 4 and re.search(r"\b(?:how|what|when|which|does|do|did|is|are|buy|sell|short|long|backtest|test|trade)\b", rest.lower()):
        return rest[0].upper() + rest[1:]
    return None


def out_of_scope(text: str, known: list[str] | None = None, trade_context: bool = False, default_symbol: str = "SPY") -> dict | None:
    """The route that declines `text` when it needs something the stored data does not hold, else None."""
    low = text.lower()
    sym = next((k for k in (known or []) if k != "VIX" and re.search(rf"\b{re.escape(k.lower())}\b", low)), None) or named_ticker(text, known) or default_symbol
    tail = " Nothing was run."
    m = re.search(_OOS_QUOTE, low)
    if m:
        opts = [{"label": "Stress-test the fills with worse execution assumptions", "ask": "Stress-test the execution of this strategy"}] if trade_context else []
        return {"action": "clarify", "declined": True, "unread": [text[m.start():m.end()]], "options": opts,
                "text": ("No historical bid and ask quotes are stored: every option fill is priced from recorded trades. A run that buys at the ask and sells at the bid cannot be done, "
                         "and how far real fills would have been from the recorded trades is not known." + tail + (" What can be done is to re-run the strategy under worse execution assumptions:" if opts else ""))}
    aside = lambda m_: bool(m_ and re.search(r"\b(?:forget|never ?mind|skip|ignore|drop|scrap|without|leave out|leaving out|aside from|not|no|instead of)\b[^.;,?]{0,18}$", low[:m_.start()]))  # noqa: E731
    m = re.search(_OOS_EVENT, low)
    if m and not aside(m):
        what = text[m.start():m.end()]
        alt = _without(text, m.start(), m.end())
        return {"action": "clarify", "declined": True, "unread": [what], "options": [{"label": alt, "ask": alt}] if alt else [],
                "text": (f"\"{what}\" needs a calendar of scheduled events, and none is stored (no Fed, CPI, jobs, earnings, holiday or option-expiration dates; no news), so this cannot be measured or tested."
                         + tail + (" The same request without that part:" if alt else ""))}
    m = re.search(_OOS_IND, low)
    if m and not aside(m) and not re.search(r"\bvwap\b", low[max(0, m.start() - 3):m.end() + 3]):
        what = text[m.start():m.end()]
        return {"action": "clarify", "declined": True, "unread": [what], "options": [],
                "text": (f"\"{what}\" is a technical indicator, and the question box does not compute indicators (VWAP is the one exception). What can be used as a condition: the previous day's move, "
                         "streaks of up or down days, the gap at the open, where the price stands at a time of day, the weekday, the VIX and VWAP." + tail)}
    m = re.search(r"\b(?:buy|buying|bought|sell|selling|short|shorting|long)\s+(?:\d[\d,]*\s+)?(?:shares|stock)\b(?: of \w+)?|\b(?:buy|sell|short)\s+(?:the\s+)?(?:spy|qqq|iwm|etf|stock|shares)\s+(?:itself|outright|directly)\b", low)
    if m and not re.search(r"\bcalls?\b|\bputs?\b|\boptions?\b|\bspreads?\b|\bcovered\b|\bstraddle|\bstrangle|\bcondor", low):
        # a trade in the shares themselves: Strata tests option trades; what the shares did is a question about the price
        near = f"What does {sym} usually do from the open to the close?"
        return {"action": "clarify", "declined": True, "unread": [text[m.start():m.end()]], "options": [{"label": near, "ask": near}],
                "text": ("Strata backtests option trades, not trades in the shares themselves, so this cannot be tested as a trade." + tail +
                         " What the shares did over a stretch of the day or after an event can be measured as a question about the price, for example:")}
    m = re.search(r"\b(?:(?:call|put|long|short|broken[- ]wing|bwb)\s+)?butterfl(?:y|ies)\b|\b(?:call|put)\s+fl(?:y|ies)\b|\bcalendar spreads?\b|\b(?:call|put) calendars?\b|\bdiagonal(?: spread)?s?\b|\bratio (?:spread|backspread)s?\b"
                  r"|\bbackspreads?\b|\bcollars?\b|\bjade lizards?\b|\bbox spreads?\b|\bdouble diagonals?\b|\brisk reversals?\b", low)
    if m and not re.search(r"\biron\s+$", low[:m.start()]) and not re.search(r"\biron (?:butter)?fl", m.group(0)):
        # Found by the V10 held-out run: "a call butterfly with 5-dollar wings" was run as one long call, without a word.
        what = text[m.start():m.end()]
        return {"action": "clarify", "declined": True, "unread": [what], "options": [],
                "text": (f"\"{what}\" is not a position Strata can backtest, so this was not run as something else." + tail +
                         " What it can test: calls and puts bought or sold, vertical spreads (debit and credit), straddles, strangles, iron condors, iron butterflies and covered calls.")}
    adv, fut = re.search(_OOS_ADVICE, low), re.search(_OOS_FUTURE, low)
    if adv or (fut and not re.search(_HISTORICAL, low)):
        near = _nearest_condition(text, known, sym)
        return {"action": "clarify", "declined": True, "unread": [text[(adv or fut).start():(adv or fut).end()]], "options": [{"label": near, "ask": near}] if near else [],
                "text": ("Strata does not forecast or give advice. It measures what happened in the stored history, which says how often something followed in the past, not what will happen next."
                         + tail + (" The nearest thing it can measure:" if near else " For example: \"How often does SPY close higher the day after a 1% down day?\""))}
    return None


# ----------------------------------------------------------------------------------------- an options trade in plain words
_STRUCT_WORDS = {"long_call": ("Buy", "{n} {sym} call{s}"), "long_put": ("Buy", "{n} {sym} put{s}"), "short_call": ("Sell", "{n} {sym} call{s}"), "short_put": ("Sell", "{n} {sym} put{s}"),
                 "bull_call_spread": ("Buy", "{n} {sym} call debit spread{s} (buy a call, sell a higher one)"), "bear_put_spread": ("Buy", "{n} {sym} put debit spread{s} (buy a put, sell a lower one)"),
                 "bull_put_spread": ("Sell", "{n} {sym} put credit spread{s} (sell a put, buy a lower one)"), "bear_call_spread": ("Sell", "{n} {sym} call credit spread{s} (sell a call, buy a higher one)"),
                 "iron_condor": ("Sell", "{n} {sym} iron condor{s} (a put credit spread and a call credit spread)"), "iron_butterfly": ("Sell", "{n} {sym} iron butterfl{ies} (sell the at-the-money call and put, buy a wing on each side)"),
                 "long_straddle": ("Buy", "{n} {sym} straddle{s} (a call and a put at the same strike)"), "short_straddle": ("Sell", "{n} {sym} straddle{s} (a call and a put at the same strike)"),
                 "long_strangle": ("Buy", "{n} {sym} strangle{s} (an out-of-the-money call and put)"), "short_strangle": ("Sell", "{n} {sym} strangle{s} (an out-of-the-money call and put)"),
                 "covered_call": ("Sell", "{n} {sym} covered call{s} (100 shares held, one call sold against them)")}
SHORT_PREMIUM = ("short_put", "short_call", "bull_put_spread", "bear_call_spread", "iron_condor", "iron_butterfly", "short_strangle", "short_straddle", "covered_call")


def _ny(t: str | None, stated: str | None = None) -> str:
    if t in (None, ""):
        return ""
    if t == "open":
        return "the open (9:30 New York time)"
    if t == "close":
        return "the close (16:00 New York time)"
    m = _hhmm(t)
    said_ = f", said as \"{stated}\"" if stated and not re.fullmatch(r"\d\d:\d\d", stated.strip()) and "new york" not in stated.lower() and not re.search(r"\bet\b|eastern", stated.lower()) else ""
    return f"{_clock12(m)} New York time{said_}" if m is not None else str(t)


def describe_draft(d, effective: dict | None = None) -> dict:
    """An options trade as plain sentences, every part of it: what is traded, when it is entered and on which sessions,
    every way out, how much is traded, the period. `d` is a Draft or its dict. Returns {"lines": every sentence, "more":
    the sentences about parts the editable sentence of the interface does not show}. This is what the user is shown as
    the reading of a trade (V10: nothing a trade holds is left out of its reading)."""
    g = (lambda k, default=None: (d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)))
    sub = lambda o, k, default=None: (o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)) if o is not None else default  # noqa: E731
    sym, st = g("symbol") or "SPY", g("structure") or "long_call"
    sz = g("sizing")
    qty = int(g("quantity") or 1)
    by_contracts = sz is None or sub(sz, "mode") == "contracts"
    verb, what = _STRUCT_WORDS.get(st, ("Trade", "{n} {sym} " + st.replace("_", " ")))
    n_txt = (str(qty) if by_contracts else "").strip()
    what = what.format(n=n_txt, sym=sym, s="" if (by_contracts and qty == 1) else "s", ies="y" if (by_contracts and qty == 1) else "ies").replace("  ", " ").strip()
    eff = effective or {}
    dflt = " (not stated: the default)"
    dte = g("dte")
    dte_said = dte is not None
    if dte is None:
        dte = eff.get("dte")
    exp = ("expiring the same day (0DTE)" if dte in (0, None) else f"expiring about {int(dte)} day{'s' if dte != 1 else ''} out") + ("" if dte_said or dte is None and not eff else dflt)
    strike = g("strike")
    k_said = strike is not None
    if strike is None:
        strike = eff.get("strike")
    m_, v_ = sub(strike, "method"), sub(strike, "value", 0.0)
    short = st in SHORT_PREMIUM
    leg = "short strike" if short and st not in ("short_put", "short_call", "covered_call") else "strike"
    k_txt = (f"{leg} at the money" if m_ in (None, "atm") else f"{leg} near {v_ * 100 if v_ <= 1 else v_:g} delta" if m_ == "delta" else f"{leg} about {v_:g}% out of the money" if m_ == "pct_otm"
             else f"{leg} {v_:g} step{'s' if v_ != 1 else ''} out of the money (listed strikes)" if m_ == "strikes_otm" else leg)
    if not k_said and eff:
        k_txt += dflt
    width = g("width")
    w_said = width is not None
    if width is None:
        width = eff.get("width")
    if width and ("spread" in st or st in ("iron_condor", "iron_butterfly")):
        k_txt += f", the other strike {float(width):g} points further out" + ("" if w_said else dflt)
    entry = g("entry_time")
    when = f"at {_ny(entry, g('entry_time_stated'))}" if entry else "whenever the conditions hold"
    lines = [f"{verb} {what}, {exp}, {k_txt}, {when}."]
    more: list[str] = []
    if m_ not in (None, "atm") or (width and ("spread" in st or st in ("iron_condor", "iron_butterfly"))):
        more.append("Strikes: " + k_txt + ".")
    if not dte_said and eff and dte not in (None,):
        more.append(f"Expiration: {'the same day (0DTE)' if dte == 0 else f'about {int(dte)} days out'}{dflt}.")
    if st not in ("long_call", "long_put", "short_call", "short_put"):
        more.append(f"Position: {verb.lower()} {what}.")
    conds = [sub(c, "text") or sub(c, "expr") for c in (g("conditions") or [])]
    days = g("weekdays") or []
    names = {"Mon": "Mondays", "Tue": "Tuesdays", "Wed": "Wednesdays", "Thu": "Thursdays", "Fri": "Fridays"}
    seq = g("sequence") or []
    if seq:
        steps = "; then ".join((sub(x, "text") or sub(x, "name") or sub(x, "expr")) + (f" (within {sub(x, 'within_minutes')} minutes)" if sub(x, "within_minutes") else "") for x in seq)
        lines.append(f"First, in this order: {steps}.")
        more.append(lines[-1])
    on_days = ", ".join(names.get(x, x) for x in days)
    lines.append((("Only enter when " + " and ".join(conds) + (f", and only on {on_days}" if days else "")) if conds else (f"Enter on {on_days} only" if days else "Enter on every session")) + ".")
    if days:
        more.append("Only on " + ", ".join(names.get(x, x) for x in days) + ".")
    if g("max_entries") and int(g("max_entries")) > 1:
        lines.append(f"Up to {int(g('max_entries'))} entries a session.")
        more.append(lines[-1])
    for a in g("adds") or []:
        lines.append(f"Add {sub(a, 'quantity', 1)} more when {sub(a, 'text') or sub(a, 'expr')} (at most {sub(a, 'max_adds', 1)} time{'s' if sub(a, 'max_adds', 1) != 1 else ''}).")
        more.append(lines[-1])
    base = "the credit received" if short else "the price paid"
    exits = []
    if g("premium_target_pct") is not None:
        exits.append(f"take profit when the position has gained {float(g('premium_target_pct')):g}% of {base}")
    if g("premium_stop_pct") is not None:
        exits.append(f"stop when it has lost {float(g('premium_stop_pct')):g}% of {base}")
    ut, us = g("underlying_target_pct"), g("underlying_stop_pct")
    neutral = bias_of(st)[0] == 0  # a straddle, a strangle, a condor: the exit is on a move in EITHER direction (that is how it is run)
    if neutral and (ut is not None or us is not None):
        x_ = float(ut if ut is not None else us)
        exits.append(f"exit when {sym} has moved {x_:g}% in either direction from its price at entry")
        more.append(f"Also exit when {sym} has moved {x_:g}% in either direction from its price at entry.")
        ut = us = None
    if ut is not None:
        exits.append(f"exit when {sym} has moved {float(ut):g}% in the trade's favour from the entry")
        more.append(f"Also exit when {sym} has moved {float(ut):g}% in the trade's favour from the entry.")
    if us is not None:
        exits.append(f"exit when {sym} has moved {float(us):g}% against the trade from the entry")
        more.append(f"Also exit when {sym} has moved {float(us):g}% against the trade from the entry.")
    for k, lab in (("underlying_target_abs", "in the trade's favour"), ("underlying_stop_abs", "against the trade")):
        if g(k) is not None:
            exits.append(f"exit when {sym} has moved {float(g(k)):g} points {lab}")
            more.append(f"Also exit when {sym} has moved {float(g(k)):g} points {lab}.")
    for p_ in g("partials") or []:
        lvl = f"the position is up {sub(p_, 'premium_pct'):g}% of {base}" if sub(p_, "premium_pct") is not None else f"{sym} has moved {sub(p_, 'underlying_pct'):g}% in the trade's favour"
        exits.append(f"close {float(sub(p_, 'fraction', 0.5)) * 100:g}% of the position when {lvl}")
        more.append(f"Close {float(sub(p_, 'fraction', 0.5)) * 100:g}% of the position when {lvl}; the rest stays open.")
    for c in g("exit_conditions") or []:
        exits.append(f"exit when {sub(c, 'text') or sub(c, 'expr')}")
        more.append(f"Also exit when {sub(c, 'text') or sub(c, 'expr')}.")
    last = (f"close what is left at {_ny(g('exit_time'), g('exit_time_stated'))}" if g("exit_time") else f"close {int(g('hold_minutes'))} minutes after the entry at the latest" if g("hold_minutes")
            else f"close at the end of {int(g('hold_sessions'))} session{'s' if g('hold_sessions') != 1 else ''}" if g("hold_sessions")
            else f"close what is left at {_ny(eff['exit_time'])} (same-day contracts are closed before they expire)" if eff.get("exit_time") and dte == 0
            else "hold to expiration" if (g("hold_to_expiration") or dte not in (0, None)) else "close at the end of the session")
    if g("hold_minutes") and g("exit_time"):
        exits.append(f"close {int(g('hold_minutes'))} minutes after the entry at the latest")
        more.append(f"Closed {int(g('hold_minutes'))} minutes after the entry at the latest.")
    if g("hold_to_expiration") and g("exit_time"):
        more.append("Held to expiration when no other exit comes first.")
    exits.append(last)
    lines.append("Ways out, whichever comes first: " + "; ".join(exits) + ".")
    if sz is not None and sub(sz, "mode") != "contracts":
        v = float(sub(sz, "value", 0))
        size = {"dollars": f"about ${v:,.0f} of premium per trade", "risk_dollars": f"${v:,.0f} at risk per trade", "risk_pct": f"{v:g}% of the account at risk per trade", "equity_pct": f"{v:g}% of the account in premium per trade"}[sub(sz, "mode")]
    else:
        size = f"{qty} contract{'s' if qty != 1 else ''} per trade"
    extra = []
    if sz is not None:
        if sub(sz, "capital"):
            extra.append(f"starting with ${float(sub(sz, 'capital')):,.0f}")
        if sub(sz, "mode") in ("risk_pct", "equity_pct") or sub(sz, "reinvest"):
            extra.append("gains reinvested" if sub(sz, "reinvest") else "sized off the starting account, not the running balance")
        if sub(sz, "daily_loss_stop"):
            extra.append(f"no new trade that day after losing ${float(sub(sz, 'daily_loss_stop')):,.0f}")
    lines.append("Size: " + ", ".join([size] + extra) + ".")
    a, b = g("start"), g("end")
    if a or b:
        lines.append("Period: " + (f"from {a}" if a else "from the first stored option session") + (f" to {b}" if b else " to the latest stored session") + ".")
        more.append(lines[-1])
    return {"lines": lines, "more": list(dict.fromkeys(more))}


# ---------------------------------------------------------------------------------- the request, and what is said around it
_COMMENT = (r"^(?:but |and |so |honestly |tbh |fwiw |look,? |i mean,? )*(?:i(?:'ve| have)? always (?:figured|thought|felt|assumed|believed|said)|i (?:figure|think|suspect|bet|swear|reckon|feel like|have a (?:feeling|hunch|theory))|"
            r"my (?:guess|gut|hunch|theory|feeling|sense|read)\b|always felt like|(?:but )?that'?s just|it'?s just (?:a )?(?:feeling|hunch|from)|been (?:watching|trading|doing|staring)|i(?:'ve| have) been (?:watching|trading|doing|fading|staring)|"
            r"my (?:brother|buddy|friend|dad|uncle|wife|husband|boss|coworker|brother-in-law|father-in-law|neighbor|neighbour|mentor)\b|a (?:guy|friend|buddy) (?:i know|at work|of mine)|(?:everyone|people|twitter|reddit|they) (?:say|says|keeps? saying)|"
            r"(?:small caps|the market|it|this thing) keeps? doing this thing|just curious|no rush|thanks in advance|here'?s (?:one|a thing|something) i (?:want|wanted) to kick around)")
_REQUEST_CUE = r"\?|\b(?:backtest|test (?:it|this|that)|show me|how often|how many|what (?:is|are|was|has|does|did|time|share|percent)|does it|do they|can you|buy|sell|short|long)\b"


# ------------------------------------------------------------------------------------------------ a reply that only agrees
# A reading that waits is answered in a person's own words as often as with its button: "yes", "looks right, go ahead",
# "yep thats what i meant", or "no", "not what I meant". Only a message that says NOTHING else counts: "yes but only on
# Fridays" is a change to the reading and goes through the reader like any other message.
_AGREE = (r"run it as read|run it as is|that'?s what i (?:meant|want(?:ed)?|asked(?: for)?|said)|that is what i (?:meant|want(?:ed)?|asked(?: for)?|said)|"
          r"what i (?:meant|wanted|asked(?: for)?|said)|that'?s (?:right|correct|it|fine|good|perfect|exactly it)|that is (?:right|correct|it|fine)|"
          r"looks? (?:good|right|correct|fine|ok|okay|great|perfect)|sounds? (?:good|right|fine|great)|reads? (?:right|correctly|fine)|"
          r"go ahead|go for it|do it|run it|run that|run this|send it|let'?s go|let'?s do it|let'?s see it|let'?s see|show me|works for me|that works|"
          r"all good|as read|as is|as written|of course|uh huh|"
          r"yes|yeah|yea|yep|yup|ya|y|ok|okay|k|kk|sure|correct|right|exactly|precisely|perfect|great|good|fine|cool|"
          r"confirm(?:ed)?|approved?|proceed|continue|go|run|lgtm|absolutely|definitely|affirmative|mhm")
_REFUSE = (r"that'?s not (?:right|correct|it|what i (?:meant|wanted|asked(?: for)?|said))|that is not (?:right|correct|it|what i (?:meant|wanted|asked(?: for)?|said))|"
           r"not what i (?:meant|wanted|asked(?: for)?|said)|that'?s wrong|that is wrong|don'?t run (?:it|that|this)|do not run (?:it|that|this)|"
           r"never ?mind|scrap (?:it|that)|forget (?:it|that)|not (?:right|correct|quite)|"
           r"no|nope|nah|n|cancel|nvm|don'?t|wrong|incorrect|abort")
_REPLY_FILLER = r"(?:and|then|just|now|so|well|hey|hi|um+|uh+|oh|it|that|this|the|a|reading|one|to|me|is|all|way|ok so|i guess|i think|for me|thanks|thank you|thx|ty|please|pls)"


def go_ahead(text: str) -> str | None:
    """"yes" when the whole message only agrees, "no" when it only refuses, None when it says anything else (or both)."""
    low = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower().replace("’", "'"))
    low = re.sub(r"\s+", " ", low).strip()
    if not low or len(low) > 60 or re.search(r"\d", low) or "?" in (text or "") or re.match(r"(?:is|are|was|does|do|did|what|why|how|can|could|should|would|will|which|when|where|who)\b", low):
        return None  # a question about the reading ("is that right?") is not a go-ahead
    left, found = low, set()
    for kind, pat in (("no", _REFUSE), ("yes", _AGREE)):  # refusals first: "not right" must not leave "right" as an agreement
        def take(m, kind=kind):
            found.add(kind)
            return " "
        left = re.sub(rf"(?<![a-z']){'(?:' + pat + ')'}(?![a-z'])", take, left)
    left = re.sub(rf"(?<![a-z'])(?:{_REPLY_FILLER})(?![a-z'])", " ", left).strip()
    if left or len(found) != 1:
        return None
    return found.pop()


def without_commentary(text: str) -> tuple[str, list[str]]:
    """(the request, the sentences left aside). A sentence is left aside only when it opens with the writer's own belief or
    story ("I've always figured ...", "my brother-in-law swears by ..."), asks for nothing, and at least one other sentence
    does. What is left aside is shown to the user (the reply says so), so nothing is dropped without a word."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip()) if p.strip()]
    if len(parts) < 2:
        return text, []
    keep, aside = [], []
    for p_ in parts:
        low = p_.lower().strip()
        if re.match(_COMMENT, low) and not re.search(_REQUEST_CUE, low) and not re.search(r"\d", low):
            aside.append(p_.strip())
        else:
            keep.append(p_.strip())
    if not aside or not keep or not any(re.search(_REQUEST_CUE, k.lower()) for k in keep):
        return text, []
    req = " ".join(keep)
    if named_ticker(req) is None:
        tk = named_ticker(" ".join(aside))  # "My brother-in-law swears by covered calls on IWM. So test it for me: ..."
        if tk:
            req = f"{req.rstrip(' .')} (on {tk})"
    return req, aside
