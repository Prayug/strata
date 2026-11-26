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


def from_route(r: dict, text: str, store, default_symbol: str = "SPY") -> dict:
    """The routing decision of one message as ONE intent, with the first two stages done: understood, and checked against the sentence."""
    from strata.ai.draft import compile_draft
    act = r.get("action")
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
