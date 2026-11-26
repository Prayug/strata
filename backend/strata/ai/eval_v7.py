"""Scoring an interpretation against the meaning of its sentence (the V7 sets in `evalset_v7`).

Nothing here reads an expression to decide whether it "looks right". Parameters are compared field by field, and the entry
rule is compared by what it DOES: the interpretation is compiled, its entry rule is run over stored sessions, the reference
rule of the case is run over the same sessions, and the two must fire at the same minutes of the same days. A reader may
spell a rule any way it likes; it may not drop a condition, read a candle run of three as two, or enter at another time.

Three outcomes, as in the V6 evaluation:

  correct   everything stated is in the interpretation and the entry rule fires where the reference does
  flagged   something is wrong or missing AND the interpretation says so (a question, an unsupported part, a failed check)
  silent    wrong or incomplete, with no sign of it: the reader answered a different question without saying so
"""
from __future__ import annotations

from datetime import date

from strata.ai import evalset_v7 as E7

WINDOW = ("2026-01-02", "2026-11-25")  # 227 stored sessions with one-minute bars for SPY, QQQ and IWM
DUMMY_LEG = [{"id": "x", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}]


def _time(x) -> str | None:
    return {"09:30": "open", "9:30": "open", "16:00": "close"}.get(str(x), None if x is None else str(x))


def reference_config(case: dict, window=WINDOW) -> dict:
    """The case's entry rule as an engine configuration (the position is irrelevant to when the rule fires)."""
    kids = []
    if case.get("entry"):
        kids.append({"kind": "cond", "type": "entry_times", "params": {"times": case["entry"]}})
    kids += [{"kind": "cond", "type": "expr", "params": {"expr": e}} for e in case.get("when") or []]
    entry = {"conditions": {"kind": "group", "op": "and", "children": kids}, "sequence": [{"name": s["name"], "when": s["when"], **({"within_minutes": s["within_minutes"]} if s.get("within_minutes") else {})} for s in case.get("sequence") or []],
             "max_entries_per_day": int(case.get("max_entries") or 1), "max_open_positions": int(case.get("max_entries") or 1)}
    return {"start": window[0], "end": window[1], "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1},
            "strategies": [{"name": "reference", "symbol": case["symbol"], "legs": DUMMY_LEG, "entry": entry, "exits": {"time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}}}]}


def interpreted_config(strategy: dict, window=WINDOW) -> dict:
    """The interpretation's entry rule alone. The window that stops entries at the exit time ("lw") belongs to the exits, which
    are compared field by field: the reference rule has no exit, so it is taken out here and both rules are run to 15:55."""
    cond = strategy["entry"].get("conditions") or {}
    cond = {**cond, "children": [c for c in cond.get("children") or [] if c.get("id") != "lw"]}
    return {"start": window[0], "end": window[1], "pricing": {"mode": "trades"}, "execution": {"signal_interval_minutes": 1},
            "strategies": [{**strategy, "legs": DUMMY_LEG, "exits": {"time_exit": {"enabled": True, "time": "15:55", "dte_lte": None}}, "entry": {**strategy["entry"], "conditions": cond, "adds": []}}]}


_scan_cache: dict[str, list] = {}


def signals(cfg: dict, store) -> list[tuple[str, int]]:
    import hashlib
    import json

    from strata.engine.scan import scan_signals
    k = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    if k not in _scan_cache:
        _scan_cache[k] = sorted((s["date"].isoformat(), int(s["minute"])) for s in scan_signals(cfg, store)["signals"])
    return _scan_cache[k]


def same_entries(case: dict, strategy: dict, store, window=WINDOW) -> list[str]:
    """Differences between where the interpretation's entry rule fires and where the reference rule does."""
    try:
        ref = signals(reference_config(case, window), store)
    except Exception as e:  # noqa: BLE001  (a reference that does not run is a fault of the case, reported as such)
        return [f"the reference rule could not be run ({type(e).__name__}: {str(e)[:120]})"]
    try:
        got = signals(interpreted_config(strategy, window), store)
    except Exception as e:  # noqa: BLE001
        return [f"the interpretation's entry rule could not be run ({type(e).__name__}: {str(e)[:120]})"]
    if ref == got:
        return [] if ref else ["neither rule fired on any of the stored sessions, so the entry rule is not established"]
    a, b = set(ref), set(got)
    days_a, days_b = {d for d, _ in a}, {d for d, _ in b}
    return [f"the entry rule fires differently: {len(b)} entries on {len(days_b)} sessions against the reference's {len(a)} on {len(days_a)} ({len(a & b)} at the same minute; "
            f"{len(days_b - days_a)} sessions only in the interpretation, {len(days_a - days_b)} only in the reference)"]


def _params(case: dict, d, compiled: dict) -> list[str]:
    """Stated parameters of the position and the exits, against the draft as it compiles."""
    bad = []
    eff = compiled.get("effective") or {}
    if d.symbol != case["symbol"]:
        bad.append(f"symbol {d.symbol} for {case['symbol']}")
    if case.get("structure") and d.structure != case["structure"]:
        bad.append(f"structure {d.structure} for {case['structure']}")
    if case.get("dte") is not None and (d.dte if d.dte is not None else eff.get("dte")) != case["dte"]:
        bad.append(f"expiration {d.dte if d.dte is not None else eff.get('dte')}DTE for {case['dte']}DTE")
    if case.get("strike"):
        m, v = case["strike"]
        got = (d.strike.method, float(d.strike.value)) if d.strike is not None else ("atm", 0.0)
        if got[0] != m or abs(got[1] - float(v)) > 1e-9:
            bad.append(f"strike {got[0]} {got[1]:g} for {m} {float(v):g}")
    if case.get("entry") and _time(d.entry_time) != _time(case["entry"]):
        bad.append(f"entry time {d.entry_time} for {case['entry']}")
    if case.get("quantity") and d.quantity != case["quantity"]:
        bad.append(f"quantity {d.quantity} for {case['quantity']}")
    if case.get("max_entries") and int(d.max_entries or 1) != int(case["max_entries"]):
        bad.append(f"entries a session {d.max_entries or 1} for {case['max_entries']}")
    ex = case.get("exits") or {}
    got = {"time": _time(d.exit_time), "minutes": d.hold_minutes, "premium_target": d.premium_target_pct, "premium_stop": d.premium_stop_pct, "underlying_for": d.underlying_target_pct, "hold_sessions": d.hold_sessions}
    for k, v in ex.items():
        g = got.get(k)
        if k == "time":
            v = _time(v)
        if (float(g) if isinstance(g, (int, float)) else g) != (float(v) if isinstance(v, (int, float)) else v):
            bad.append(f"exit {k}: {g} for {v}")
    if case.get("adds"):
        a = case["adds"]
        if not d.adds:
            bad.append("the added contract is missing")
        elif d.adds[0].quantity != a["quantity"] or f"{abs(a['underlying_move_pct']) / 100:g}" not in d.adds[0].expr.replace(" ", ""):
            bad.append(f"the added contract differs: {d.adds[0].expr} x{d.adds[0].quantity}")
    if case.get("partial"):
        p = case["partial"]
        if not d.partials:
            bad.append("the partial exit is missing")
        elif d.partials[0].premium_pct != p["premium_pct"] or abs(d.partials[0].fraction - p["fraction"]) > 1e-9:
            bad.append(f"the partial exit differs: {d.partials[0].fraction:g} of the position at {d.partials[0].premium_pct}")
    return bad


def _norm(field: str, v):
    if field == "strike":
        m, x = v
        return (str(m), round(float(x), 6))
    if field in ("entry_time", "exit_minutes"):
        return _time(v) if field == "entry_time" else (v if v == "close" else int(v))
    if field == "symbol":
        return str(v).upper()
    if field == "dte":
        return int(v)
    return round(float(v), 6)


def apply_ref(case: dict, field: str, v) -> dict:
    """The reference case with one setting of an experiment at one value (only settings that change WHEN the rule fires matter)."""
    c = dict(case)
    if field == "symbol":
        c["symbol"] = str(v)
    elif field == "entry_time":
        c["entry"] = str(v)
    elif field == "prev_decline":
        c["when"] = [w for w in c.get("when") or [] if "previous_session.return_pct" not in w] + [f"previous_session.return_pct <= -{abs(float(v)):g}"]
    return c


def check(case: dict, route: dict, store, window=WINDOW, current=None) -> tuple[list[str], bool]:
    """(differences, did the interpretation say something was open). Empty differences: the sentence was understood."""
    from strata.ai import intent as I
    from strata.ai import nl_experiment as NX
    from strata.ai.draft import compile_draft
    act = route.get("action")
    d = route.get("draft")
    said = act in ("unknown", "failed", "experiment_needs_base") or bool(route.get("too_many"))
    if d is not None:
        said = said or bool(d.unsupported) or any(q.get("answered") is None for q in (d.questions or []))
    if case["kind"] == "flag":
        if d is not None and not said:
            it = I.from_route(route, case["text"], store)
            said = any(not c["ok"] for c in it.get("checks") or [])
        return ([] if said else ["the part that cannot be represented was not flagged"]), said
    if case["kind"] == "modify":
        if act not in ("modify", "derive", "new") or d is None:
            return [f"read as {act}, not as a change to the current strategy"], said
        bad = [f"{k}: {getattr(d, k, None)} for {v}" for k, v in case["fields"].items() if getattr(d, k, None) != v]
        return bad, said
    if case["kind"] == "experiment" and act == "sweep" and len(case["dims"]) == 1 and current is not None:
        # the older comparison of separately stored runs is the same request when one setting is varied on the current strategy
        f = {"prev_ret": "prev_decline", "target": "underlying_target", "delta": "strike"}.get(route.get("field"), route.get("field"))
        vals = [("delta", (float(v) / 100 if float(v) > 1 else float(v))) for v in route.get("values") or []] if route.get("field") == "delta" else list(route.get("values") or [])
        route = {"action": "experiment", "draft": current, "dims": [{"field": f, "values": vals, "assumed": False}]}
        act, d = "experiment", current
    if case["kind"] == "experiment":
        if act != "experiment":
            return [f"read as {act}, not as an experiment"], said
        want = {x["field"]: [_norm(x["field"], v) for v in x["values"]] for x in case["dims"]}
        got = {x["field"]: [_norm(x["field"], tuple(v) if isinstance(v, list) else v) for v in x["values"]] for x in route["dims"]}
        bad = []
        for f, vals in want.items():
            if f not in got:
                bad.append(f"the setting '{f}' is not varied")
            elif sorted(map(str, got[f])) != sorted(map(str, vals)):
                bad.append(f"'{f}' is varied over {got[f]} instead of {vals}")
        bad += [f"a setting that was not asked for is varied: '{f}' over {got[f]}" for f in got if f not in want]
        if any(x.get("assumed") for x in route["dims"] if x["field"] in want):
            said = True  # values the sentence stated may not come back as assumptions
        base_case = dict(case["base"])
        base_draft = d
        # the base is compared at the first value of every setting, applied to both sides
        for x in case["dims"]:
            v0 = x["values"][0]
            base_case = apply_ref(base_case, x["field"], v0)
            if x["field"] in got and base_draft is not None:
                try:
                    base_draft = NX.apply(base_draft, x["field"], tuple(v0) if isinstance(v0, list) else v0)
                except Exception:  # noqa: BLE001
                    pass
        if base_draft is None:
            return bad + ["the experiment has no base strategy"], said
        comp = compile_draft(base_draft, store)
        if comp["errors"] or not comp.get("strategy"):
            return bad + ["the base strategy does not compile: " + "; ".join(comp["errors"])[:160]], said
        skip = {x["field"] for x in case["dims"]}
        pc = {**base_case, "dte": None if "dte" in skip else base_case.get("dte"), "strike": None if "strike" in skip else base_case.get("strike"),
              "exits": {k: v for k, v in (base_case.get("exits") or {}).items() if not (k in ("minutes", "time") and "exit_minutes" in skip) and not (k == "premium_target" and "premium_target" in skip)
                        and not (k == "premium_stop" and "premium_stop" in skip) and not (k == "underlying_for" and "underlying_target" in skip)}}
        bad += _params(pc, base_draft, comp)
        if base_case.get("when") or base_case.get("sequence") or base_case.get("entry"):
            bad += same_entries(base_case, comp["strategy"], store, window)
        return bad, said
    # a strategy
    if act == "experiment":
        return ["read as an experiment, not as one strategy"], said
    if act not in ("new", "modify", "derive") or d is None or d.kind != "options":
        return [f"read as {act}{'' if d is None else ' (' + d.kind + ')'}, not as an option strategy"], said
    comp = compile_draft(d, store)
    if comp["errors"] or not comp.get("strategy"):
        return ["the strategy does not compile: " + "; ".join(comp["errors"])[:160]], True
    bad = _params(case, d, comp)
    if case.get("when") or case.get("sequence") or case.get("entry"):
        bad += same_entries(case, comp["strategy"], store, window)
    return bad, said


def evaluate(agent, cases: list[dict] | None = None, llm: bool = False, window=WINDOW, progress=None, reader=None, label: str | None = None) -> dict:
    """Run every case through one reader and score it: the built-in readers, the application's routing with the configured
    language model (`llm=True`), or `reader(text, current) -> route` for one planner measured on its own."""
    cases = cases if cases is not None else E7.UNSEEN
    known = sorted(agent.store.stored_symbols())
    if llm and reader is None and agent.llm()[0] is None:
        raise RuntimeError("no language model is configured, so its interpretation cannot be measured")
    rows = []
    base = agent._rules(E7.BASE, None, known, None, "SPY").get("draft")  # the strategy a follow-up case is said about
    for i, c in enumerate(cases):
        if progress:
            progress(i / max(1, len(cases)), c["text"][:60])
        who = None
        cur = base.model_copy(deep=True) if (c.get("follow_up") and base is not None) else None
        try:
            if reader is not None:
                route = reader(c["text"], cur)
            elif llm:
                route, who, _note = agent.interpret(None, c["text"], cur, None, None)
            else:
                route = agent._rules(c["text"], cur, known, None, "SPY")
            bad, said = check(c, route, agent.store, window, cur)
        except Exception as e:  # noqa: BLE001  (a reader that fails on a sentence has not understood it, and the failure is shown: it is never silent)
            route, bad, said = {"action": "error"}, [f"the reader failed: {type(e).__name__}: {e}"[:200]], True
        status = "correct" if not bad else ("flagged" if said and (c["kind"] != "flag" or route.get("action") == "error") else "silent")
        rows.append({"n": i + 1, "tag": c["tag"], "kind": c["kind"], "text": c["text"], "status": status, "differences": bad, "read_as": route.get("action"), "reader": who, "asked": bool(said)})
    tot = {k: sum(1 for r in rows if r["status"] == k) for k in ("correct", "flagged", "silent")}
    by: dict[str, dict] = {}
    for r in rows:
        b = by.setdefault(r["kind"], {"cases": 0, "correct": 0, "flagged": 0, "silent": 0})
        b["cases"] += 1
        b[r["status"]] += 1
    n = len(rows)
    tot["correct_but_asked"] = sum(1 for r in rows if r["status"] == "correct" and r["asked"] and r["kind"] != "flag")  # a right reading the user was still asked about
    return {"cases": n, **tot, "accuracy_pct": round(tot["correct"] / n * 100, 1) if n else None, "silent_pct": round(tot["silent"] / n * 100, 1) if n else None, "by_kind": by, "rows": rows,
            "failures": [r for r in rows if r["status"] != "correct"], "interpreter": label or ("language model" if llm else "built-in readers"), "window": list(window), "sealed_sha256": E7.sealed_hash(),
            "run_on": date.today().isoformat()}
