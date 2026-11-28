"""Scoring the V8 understanding sets (evalset_v8): what the application ran, against what the sentence asked.

Facts are read from what was compiled to run (the study, the investigation plan, the profile, the strategy), never from the
reader's own description of it. Each single request is one of:

  correct   it ran what was asked, or it asked a question where a question was the right response
  asked     it asked, or declined, where it could have answered; nothing ran
  silent    it ran something other than what was asked, without saying so

Conversations are scored turn by turn, on the state after each turn; a turn also checks that what earlier turns
established and this turn did not mention is unchanged ("context kept"). Answers are run on the stored history with the
jobs executed in-process, and the reply is checked against an independent calculation (evalset_v8.reference) and for
statements it must and must not make.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from strata.ai import evalset_v8 as E

WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]


class NoJobs:
    """Interpretation only: runs are recorded and never executed."""

    def submit(self, rid, kind, payload):
        return None


class InlineJobs:
    def submit(self, rid, kind, payload):
        from strata.api import db
        from strata.api.tasks import execute
        try:
            db.update_run(rid, status="done", progress=1.0, summary=execute(rid, kind, payload, lambda *a: None, lambda: False), finished_at=db.now())
        except Exception as e:  # noqa: BLE001
            db.update_run(rid, status="error", error=f"{type(e).__name__}: {e}", finished_at=db.now())


# ------------------------------------------------------------------------------------------------- canonical facts
def _num(x) -> str:
    return f"{float(x):g}"


def canon_expr(expr: str, start: str | None) -> list[str]:
    """A rule expression as canonical comparisons (see evalset_v8)."""
    out = []
    for part in re.split(r"\s+and\s+", expr.strip()):
        p = part.strip().strip("()")
        m = re.fullmatch(r"([\w.]+(?:\(\))?)\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)", p)
        if p in ("price > vwap", "price >= vwap"):
            out.append("above_vwap")
            continue
        if p in ("price < vwap", "price <= vwap"):
            out.append("below_vwap")
            continue
        mc = re.fullmatch(r"crosses_(above|below)\(\s*price\s*,\s*vwap\s*\)", p)
        if mc:
            out.append(f"crosses_{mc.group(1)}:vwap")
            continue
        if not m:
            out.append("expr:" + p)
            continue
        name, op, v = m.group(1), m.group(2), float(m.group(3))
        op = {"==": "="}.get(op, op)
        if name in ("previous_session.return_pct", "return_pct(1)"):
            out.append(f"prev_return{op}{_num(v)}")
        elif name in ("gap_pct", "previous_session.gap_pct") and name == "gap_pct":
            out.append(f"gap{op}{_num(v)}")
        elif name == "change_pct":
            out.append(f"change{op}{_num(v)}")
        elif name == "from_open_pct":
            if v == 0 and start and re.match(r"\d\d:\d\d$", start) and start != "09:30":
                mins = int(start[:2]) * 60 + int(start[3:]) - 570
                out.append(f"first_candle:{mins}:{'up' if op in ('>', '>=') else 'down'}")
            else:
                out.append(f"from_open{op}{_num(v)}")
        elif name in ("vix", "vix_prev_close"):
            out.append(f"vix{op}{_num(v)}")
        elif name in ("consecutive_up_days()", "consecutive_green_days()", "consecutive_up_days", "consecutive_green_days"):
            out.append(f"up_streak{op}{_num(v)}")
        elif name in ("consecutive_down_days()", "consecutive_red_days()", "consecutive_down_days", "consecutive_red_days"):
            out.append(f"down_streak{op}{_num(v)}")
        elif name == "weekday" and op == "=":
            out.append(f"weekday={WD[int(v)]}")
        else:
            out.append(f"expr:{p}")
    return out


def _study_facts(cfg: dict, frame_kind: str | None = None) -> dict:
    start = cfg.get("entry_time") or "open"
    conds = []
    for leaf in _leaves(cfg.get("condition") or {}):
        if leaf.get("type") == "expr":
            conds += canon_expr(leaf["params"]["expr"], start)
        elif leaf.get("type") == "day_of_week":
            conds += [f"weekday={d}" for d in leaf["params"]["days"]]
    seq = [c for s in cfg.get("sequence") or [] for c in canon_expr(s.get("when") or "", None)]
    t, ref, meas, dirn = cfg.get("target"), cfg.get("reference"), cfg.get("measure"), cfg.get("direction")
    if t is None:
        outcome = "end_return"
    elif t.get("kind") == "expr":
        outcome = f"touch:{t.get('expr')}"
    elif meas == "close" and float(t.get("value") or 0) == 0:
        side = ">" if dirn in ("up", None) else "<"
        outcome = {"open": f"close{side}open", "prev_close": f"close{side}prev_close", "entry": f"close{side}open" if start == "open" else f"close{side}start"}.get(ref, f"close{side}open")
    elif meas == "close":
        outcome = f"close_vs_{ref}:{_num(t['value'])}"
    else:
        outcome = f"touch:{float(t['value']):+g}"
    hz = cfg.get("horizon") or {}
    end = "close" if hz.get("kind") == "session" else hz.get("time") if hz.get("kind") == "time" else f"sessions:{hz.get('sessions')}"
    if cfg.get("reference") == "prev_close" and hz.get("kind") in ("sessions", "session") and (t is None or meas == "close"):
        # since the V10 held-out audit "N sessions later" is counted from the day the condition was met: the engine is given
        # N - 1 sessions after the observation session and measures from the previous close. The question's N is reported.
        nn_ = (int(hz.get("sessions") or 0) if hz.get("kind") == "sessions" else 0) + 1
        if nn_ > 1 or (frame_kind == "movement" and hz.get("kind") == "session" and cfg.get("target") is None):
            end = f"sessions:{nn_}"
    out = {"type": frame_kind or ("movement" if t is None else "frequency"), "symbol": cfg.get("symbol"), "conditions": conds, "outcome": outcome, "start": "open" if start in ("open", "09:30") else start,
           "end": end, "period": [cfg.get("start"), cfg.get("end")]}
    if seq:
        out["sequence"] = seq
    return out


def _leaves(node: dict) -> list[dict]:
    if node.get("kind") == "group":
        return [x for c in node.get("children") or [] for x in _leaves(c)]
    return [node] if node else []


def _strategy_facts(d, store) -> dict:
    from strata.ai.draft import compile_draft
    comp = compile_draft(d, store)
    strat = comp.get("strategy") or {}
    legs = [l for l in strat.get("legs") or [] if l.get("kind", "option") == "option"]
    entry = d.entry_time or "signal"
    conds = []
    for c in d.conditions:
        conds += canon_expr(c.expr, entry if entry not in ("open", "close") else None)
    if d.weekdays:
        conds += [f"weekday={w}" for w in d.weekdays]
    seq = [c for s in d.sequence for c in canon_expr(s.expr, None)]
    strike = None
    if d.strike:
        strike = "atm" if d.strike.method == "atm" else f"delta:{_num(d.strike.value * 100 if d.strike.value <= 1 else d.strike.value)}" if d.strike.method == "delta" else f"otm:{_num(d.strike.value)}"
    hold = "to_expiration" if d.hold_to_expiration else f"minutes:{d.hold_minutes}" if d.hold_minutes else f"sessions:{d.hold_sessions}" if d.hold_sessions else None
    out = {"type": "strategy", "symbol": d.symbol, "structure": d.structure, "dte": d.dte, "strike": strike, "quantity": d.quantity, "entry": "open" if entry in ("open", "09:30") else entry,
           "exit": d.exit_time, "conditions": conds, "premium_target": d.premium_target_pct, "premium_stop": d.premium_stop_pct, "underlying_target": d.underlying_target_pct,
           "underlying_stop": d.underlying_stop_pct, "hold": hold, "period": [d.start.isoformat() if d.start else None, d.end.isoformat() if d.end else None]}
    if seq:
        out["sequence"] = seq
    if not legs and not d.structure:
        out["structure"] = None
    return out


def facts_of_turn(msgs: list[dict], state: dict, version, store, history: list[dict] | None = None) -> dict:
    """What one turn did: a question asked, a refusal, or the facts of what it set up to run."""
    from strata.ai.draft import Draft
    asst = [m for m in msgs if m.get("role") == "assistant"]
    first = asst[0] if asst else None
    c = (first or {}).get("content") or {}
    route = c.get("route")
    if route == "noop" and history:
        # nothing changed: the facts are those of the last thing set up to run
        prior = [m for m in history if m["role"] == "assistant" and m["kind"] in ("plan", "investigation") or (m["kind"] == "result" and (m.get("content") or {}).get("view") == "profile")]
        if prior:
            return facts_of_turn([prior[-1]], state, version, store)
    if route == "clarify":
        return {"response": "flag" if c.get("declined") else "clarify", "text": c.get("text", "")}
    # a plan that stopped before running - an open question, or a part of the request it could not represent, listed - is a
    # question back, not a silent run. (Rule added after the sealed set was scored on 2026-11-27; that score used the old rule.)
    plan = next((m for m in asst if m["kind"] == "plan"), None)
    if plan is not None:
        pc = plan.get("content") or {}
        if pc.get("state") == "awaiting_answer":
            return {"response": "clarify", "text": pc.get("text", "")}
        if pc.get("state") == "awaiting_confirmation" and (pc.get("unsupported") or (pc.get("draft") or {}).get("unsupported") or (pc.get("draft") or {}).get("review") or "Your message leaves" in str(pc.get("text") or "")):
            return {"response": "flag", "text": pc.get("text", "")}  # V10: a reading that waits with the reasons listed (Draft.review) has not run either
    if route in (None, "unknown") and first and first["kind"] == "text":
        return {"response": "flag", "text": c.get("text", "")}
    if route in ("explore", "show_trades"):
        return {"response": "ran", "type": "explore"}
    if route == "compare_versions" or (first and first["kind"] == "run" and c.get("compare_with")) or c.get("compare_with"):
        return {"response": "ran", "type": "compare"}
    if route == "explain_trade":
        return {"response": "ran", "type": "explain"}
    if route in ("robustness",):
        return {"response": "ran", "type": "reliability"}
    if route == "undo":
        pass
    frame = state.get("frame") or {}
    # what was set up to run
    if first and first["kind"] == "investigation":
        plan = c.get("plan") or {}
        variants = plan.get("variants") or []
        if plan.get("kind") in ("first_candle", "agreement") or plan.get("pairs"):
            pairs = plan.get("pairs") or []
            fr = plan.get("frame") or frame
            extra = []
            for cd in fr.get("conditions") or []:
                from strata.ai import semantics as SM
                e = SM.cond_expr(cd)
                if e:
                    extra += canon_expr(e[0], None)
                elif cd.get("key") == "weekday":
                    extra.append(f"weekday={cd['value']}")
            scopes = {p.get("scope") for p in pairs}
            return {"response": "ran", "type": "agreement", "symbol": plan.get("symbol"), "candle": sorted({int(p["minutes"]) for p in pairs if p.get("minutes")}),
                    "scope": "both" if scopes == {"day", "rest"} else next(iter(scopes), None), "predictor": "prev_direction" if any(p.get("predictor") == "prev_direction" for p in pairs) else "first_candle",
                    "conditions": extra, "period": list(fr.get("period") or [None, None])}
        # a set of variants of one question: the study each run was given, with the differing values joined
        from strata.api import db
        kind = (c.get("frame") or frame).get("kind")
        fs = []
        for r in c.get("runs") or []:
            cfg = (db.get_run(r["run_id"]) or {}).get("config") or {}  # the study configuration the run was given
            req = (cfg.get("_research") or {}).get("requested") or {}
            fs.append(_study_facts({**{k: v for k, v in cfg.items() if k != "_research"}, "start": req.get("start"), "end": req.get("end")}, kind if kind in ("frequency", "movement") else None))
        return {"response": "ran", **_merge(fs)}
    if route == "profile" or (first and first["kind"] == "result" and c.get("view") == "profile"):
        f = c.get("frame") or frame
        if f.get("kind") == "timing":
            return {"response": "ran", "type": "timing", "symbol": f.get("symbol"), "measure": f.get("measure"), "bucket": f.get("bucket"), "conditions": _frame_conds(f), "period": list(f.get("period") or [None, None])}
        if f.get("kind") == "breakdown":
            return {"response": "ran", "type": "breakdown", "symbol": f.get("symbol"), "by": f.get("by"), "measure": f.get("measure"), "conditions": _frame_conds(f), "period": list(f.get("period") or [None, None])}
        if (f.get("outcome") or {}).get("key") == "rel":
            rl = lambda x: f"{x['a']}{x['op']}{x['b']}" + (f"@{x['pct']:g}" if x.get("pct") else "")  # noqa: E731
            o_ = f["outcome"]["rel"]
            a_, b_ = str(o_.get("a")), str(o_.get("b"))
            conds_ = [c for c in _frame_conds(f)] + ["rel:" + rl(c["rel"]) for c in f.get("conditions") or [] if c.get("key") == "rel"]
            if not f.get("then") and not o_.get("pct"):
                # V10 reads these as one price against a level. Where the older sets expect the same question in the older
                # words, it is scored in those words: "above its open at 11:00" is close>open counted until 11:00, and "takes
                # out yesterday's high in the first hour" is a touch of that level from the open until 10:30.
                if a_.startswith("at:") and b_ in ("open", "prev_close"):
                    return {"response": "ran", "type": "frequency", "symbol": f.get("symbol"), "outcome": f"close{o_['op']}{b_}", "start": "open", "end": a_[3:], "conditions": conds_, "period": list(f.get("period") or [None, None])}
                if a_.split(":")[0] in ("high_by", "low_by", "high", "low") and b_ in ("prev_high", "prev_low") and (a_.startswith("high") == (b_ == "prev_high")):
                    return {"response": "ran", "type": "frequency", "symbol": f.get("symbol"), "outcome": f"touch:{b_}", "start": "open", "end": a_.split(":", 1)[1] if ":" in a_ else "close", "conditions": conds_,
                            "period": list(f.get("period") or [None, None])}
            return {"response": "ran", "type": "frequency", "symbol": f.get("symbol"), "outcome": "rel:" + rl(f["outcome"]["rel"]), "then": ("rel:" + rl(f["then"]["rel"])) if f.get("then") else None,
                    "conditions": [c for c in _frame_conds(f)] + ["rel:" + rl(c["rel"]) for c in f.get("conditions") or [] if c.get("key") == "rel"], "period": list(f.get("period") or [None, None])}
        ev = (f.get("outcome") or {}).get("event") or {}
        return {"response": "ran", "type": "frequency", "symbol": f.get("symbol"), "outcome": f"event:{ev.get('key')}{ev.get('op')}{_num(ev.get('value') or 0)}", "conditions": _frame_conds(f),
                "period": list(f.get("period") or [None, None])}
    if version is not None:
        d = Draft.model_validate(version["draft"])
        if d.kind == "options":
            return {"response": "ran", **_strategy_facts(d, store)}
        from strata.ai.draft import compile_draft
        comp = compile_draft(d, store)
        st = comp.get("study") or {}
        kind = frame.get("kind") if frame.get("kind") in ("frequency", "movement") else None
        f = _study_facts({**st, "start": d.start.isoformat() if d.start else None, "end": d.end.isoformat() if d.end else None}, kind)
        return {"response": "ran", **f}
    return {"response": "flag", "text": c.get("text", "")}


def _frame_conds(f: dict) -> list[str]:
    from strata.ai import semantics as SM
    out = []
    for cd in f.get("conditions") or []:
        if cd.get("key") == "weekday":
            out.append(f"weekday={cd['value']}")
            continue
        e = SM.cond_expr(cd)
        if e:
            out += canon_expr(e[0], None)
    return out


def _merge(fs: list[dict]) -> dict:
    if not fs:
        return {"type": "frequency"}
    out = dict(fs[0])
    syms = sorted({f.get("symbol") for f in fs})
    if len(syms) > 1:
        out["symbols"] = syms
    conds = [set(f.get("conditions") or []) for f in fs]
    common = set.intersection(*conds) if conds else set()
    vary = sorted(set.union(*conds) - common) if conds else []
    keys: dict = {}
    for c in vary:
        fc = re.fullmatch(r"first_candle:(\d+):(up|down)", c)
        if fc:
            keys.setdefault(("first_candle", fc.group(2)), []).append(fc.group(1))
            continue
        k = re.match(r"([a-z_:]+?)(<=|>=|<|>|=)(.*)", c)
        keys.setdefault((k.group(1) + k.group(2), None) if k else (c, None), []).append(k.group(3) if k else "")
    merged = []
    for (k, d), xs in keys.items():
        merged.append(f"first_candle:{'|'.join(sorted(set(xs), key=float))}:{d}" if k == "first_candle" else k + "|".join(xs))
    out["conditions"] = sorted(common) + merged
    return out


# ---------------------------------------------------------------------------------------------------------- matching
def _cond_match(exp: str, got: str) -> bool:
    if exp == got:
        return True
    ne = re.fullmatch(r"([a-z_]+(?::\d+|:\*)?(?::(?:up|down))?)(<=|>=|<|>|=)?(.*)", exp)
    ng = re.fullmatch(r"([a-z_]+(?::\d+)?(?::(?:up|down))?)(<=|>=|<|>|=)?(.*)", got)
    if exp.startswith("first_candle:") and got.startswith("first_candle:"):
        a, b = exp.split(":"), got.split(":")
        return (a[1] == "*" or a[1] == b[1] or (a[1] in b[1].split("|") and False)) and a[2] == b[2]
    if not ne or not ng:
        return False
    if ne.group(1) != ng.group(1):
        return False
    oe, og = (ne.group(2) or "").replace("=", "") or "=", (ng.group(2) or "").replace("=", "") or "="
    if oe != og and not (oe in ("<", ">") and og in ("<", ">") and oe == og):
        return False
    ve, vg = ne.group(3), ng.group(3)
    if ve == "*":
        return True
    try:
        return any(abs(float(ve) - float(x)) < 1e-9 for x in vg.split("|"))
    except ValueError:
        return ve == vg


def _value_match(key: str, exp, got) -> bool:
    if isinstance(exp, list) and key not in ("conditions", "sequence", "candle", "symbols", "period"):
        return any(_value_match(key, e, got) for e in exp)
    if exp is None:
        return got in (None, [], "", 0) if key not in ("dte", "quantity") else got is None
    if key in ("conditions", "sequence"):
        got = list(got or [])
        if len(exp) != len(got):
            return False
        used = set()
        for e in exp:
            j = next((i for i, g in enumerate(got) if i not in used and _cond_match(e, g)), None)
            if j is None:
                return False
            used.add(j)
        return True
    if key == "period":
        g = list(got or [None, None])
        return all((e is None and not gg) or (e is not None and gg and str(gg)[:10] == e) for e, gg in zip(exp, g))
    if key in ("candle", "symbols"):
        return sorted(exp) == sorted(got or [])
    if key == "outcome":
        if isinstance(got, (list, tuple)):
            # V10 measures a word with two readings ("close green") both ways and shows each: the expected reading is right
            # when it is one of those measured
            return any(_value_match(key, exp, g_) for g_ in got)
        e, g = str(exp), str(got)
        if e == g:
            return True
        m1, m2 = re.fullmatch(r"touch:([+-]?\d+(?:\.\d+)?)", e), re.fullmatch(r"touch:([+-]?\d+(?:\.\d+)?)", g)
        return bool(m1 and m2 and abs(float(m1.group(1)) - float(m2.group(1))) < 1e-9)
    if key in ("entry", "exit", "start", "end"):
        norm = lambda x: {"09:30": "open", "open": "open"}.get(str(x), str(x))  # noqa: E731
        return norm(exp) == norm(got)
    if isinstance(exp, float) or isinstance(got, float):
        try:
            return abs(float(exp) - float(got)) < 1e-9
        except (TypeError, ValueError):
            return False
    return exp == got


def matches(expect: dict, got: dict) -> tuple[bool, list[str]]:
    bad = []
    for k, v in expect.items():
        if k in ("clarify", "flag", "clarify_ok"):
            continue
        if not _value_match(k, v, got.get(k)):
            bad.append(f"{k}: expected {v!r}, got {got.get(k)!r}")
    return not bad, bad


def score(item: dict, got: dict) -> tuple[str, list[str]]:
    exp = item["expect"]
    resp = got.get("response")
    if exp.get("clarify"):
        return ("correct", []) if resp == "clarify" else ("asked", ["declined instead of asking"]) if resp == "flag" else ("silent", ["ran instead of asking"])
    if exp.get("flag") and not item.get("alternatives"):
        return ("correct", []) if resp in ("flag", "clarify") else ("silent", ["ran something; should have declined or asked"])
    if resp in ("clarify", "flag"):
        if exp.get("clarify_ok") or exp.get("flag"):
            return "correct", []
        return "asked", [f"{resp}: {got.get('text', '')[:160]}"]
    ok, bad = matches({k: v for k, v in exp.items() if k not in ("flag",)}, got) if not exp.get("flag") else (False, ["ran"])
    if ok:
        return "correct", []
    for alt in item.get("alternatives") or []:
        ok2, _ = matches(alt, got)
        if ok2:
            return "correct", []
    return "silent", bad


# ------------------------------------------------------------------------------------------------------------ running
def _fresh(tmp: Path):
    """A scratch database for the evaluation's conversations, reading the SAME stored market data as the application: the
    runs an answer starts (studies, backtests) open the market store through the settings, so the scratch directory links
    to the real one. (Before this was fixed, the sealed answer runs of 2026-11-27 read an empty store: see V8_REPORT.md.)"""
    from strata.api import db
    from strata.config import get_settings, override_settings
    real = get_settings().data_dir / "market"
    link = tmp / "market"
    if real.exists() and not link.exists():
        link.symlink_to(real.resolve(), target_is_directory=True)
    override_settings(data_dir=tmp)
    db.init_db()


def run_singles(items: list[dict], store, agent_factory) -> list[dict]:
    from strata.api import db
    out = []
    for it in items:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        t0 = time.time()
        try:
            msgs = ag.post(sid, it["text"])
            s, v, _ = ag._state(sid)
            got = facts_of_turn(msgs, s["state"], v, store)
        except Exception as e:  # noqa: BLE001
            got = {"response": "error", "error": f"{type(e).__name__}: {e}"}
        status, why = score(it, got) if got.get("response") != "error" else ("silent", [got["error"]])
        out.append({"text": it["text"], "level": it["level"], "category": it["category"], "status": status, "why": why, "got": got, "seconds": round(time.time() - t0, 2)})
    return out


def run_conversations(convs: list[dict], store, agent_factory) -> list[dict]:
    from strata.api import db
    out = []
    for cv in convs:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        turns, prev = [], None
        for t in cv["turns"]:
            try:
                msgs = ag.post(sid, t["text"])
                s, v, _ = ag._state(sid)
                got = facts_of_turn(msgs, s["state"], v, store, db.list_messages(sid))
            except Exception as e:  # noqa: BLE001
                got = {"response": "error", "error": f"{type(e).__name__}: {e}"}
            status, why = score(t, got) if got.get("response") != "error" else ("silent", [got["error"]])
            kept = None
            if prev and got.get("response") == "ran" and prev.get("response") == "ran" and got.get("type") == prev.get("type") and got.get("type") not in ("explore", "compare", "explain"):
                named = set(t["expect"]) | {"response"}
                kept = all(prev.get(k) == got.get(k) for k in prev if k not in named and k not in ("text",) and k in got) if t["expect"].get("type") == prev.get("type") else None
            turns.append({"text": t["text"], "status": status, "why": why, "got": got, "context_kept": kept})
            if got.get("response") == "ran" and got.get("type") not in ("explore", "compare", "explain", "reliability"):
                prev = got
        out.append({"id": cv["id"], "turns": turns})
    return out


def numbers_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?", text or "")]


def run_answers(items: list[dict], store, agent_factory) -> list[dict]:
    from strata.api import db
    from strata.api.results import load_trades
    out = []
    for it in items:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        t0 = time.time()
        texts = []
        try:
            turns = it.get("turns") or [it["text"]]
            for tt in turns:
                ag.post(sid, tt)
                for _ in range(3):
                    ag.finalize(sid)
            msgs = db.list_messages(sid)
            res = [m for m in msgs if m["role"] == "assistant"]
            for m in res:
                c = m["content"] if isinstance(m["content"], dict) else {}
                a = c.get("answer") or {}
                texts.append(" ".join([c.get("text") or ""] + [a.get("headline") or ""] + list(a.get("happened") or []) + list((a.get("story") or {}).get("lines") or []) + list((a.get("story") or {}).get("cause") or [])))
            reply = " ".join(texts)
        except Exception as e:  # noqa: BLE001
            reply = f"ERROR {type(e).__name__}: {e}"
        checks = []
        ref = E.reference(it["reference"], store) if it.get("reference") and it["reference"] != "worst_trade" else None
        nums = numbers_in(reply)
        if it.get("reference") == "worst_trade":
            runs = [r for r in db.list_runs(kind="backtest", limit=50) if True]
            rid = next((m["content"].get("run_id") for m in reversed(db.list_messages(sid)) if m["kind"] == "result" and (m["content"] or {}).get("view") == "single"), None)
            tr = load_trades(rid) if rid else []
            worst = min(tr, key=lambda t: t["pnl"]) if tr else None
            ref = {"pnl": worst["pnl"], "date": worst["entry_date"]} if worst else None
        for key in it.get("numbers") or []:
            if ref is None or key not in ref:
                checks.append((key, False, "no reference"))
                continue
            v = float(ref[key])
            tol = {"pct": 0.15, "k": 1.0, "mean": 0.006, "pnl": 0.01}.get(key, 0.15)
            ok = any(abs(abs(n_) - abs(v)) <= tol for n_ in nums)
            checks.append((key, ok, f"reference {v:.4f}"))
        for key in it.get("names") or []:
            want = str((ref or {}).get(key, ""))
            if key == "date" and want:
                import datetime as _dt
                d = _dt.date.fromisoformat(want)
                ok = want in reply or f"{d.strftime('%B')} {d.day}, {d.year}" in reply
            else:
                ok = bool(want) and want.lower() in reply.lower()
            checks.append((key, ok, f"reference {want}"))
        for group in it.get("must") or []:
            ok = any(g.lower() in reply.lower() for g in group)
            checks.append(("must:" + "/".join(group[:2]), ok, ""))
        bad = [f for f in E.FORBIDDEN if f in reply.lower()]
        checks.append(("forbidden statements", not bad, ", ".join(bad)))
        if it.get("no_trades_in"):
            y = it["no_trades_in"]
            rid = next((m["content"].get("run_id") for m in reversed(db.list_messages(sid)) if m["kind"] == "result" and (m["content"] or {}).get("run_id")), None)
            tr = load_trades(rid) if rid else []
            ok = not any(str(t["entry_date"]).startswith(y) for t in tr)
            checks.append((f"no trades in {y}", ok, f"{len(tr)} trades"))
        out.append({"id": it["id"], "ok": all(c[1] for c in checks), "checks": checks, "reply": reply[:1500], "reference": ref, "seconds": round(time.time() - t0, 1)})
    return out


def summarise(singles: list[dict], convs: list[dict], answers: list[dict] | None = None) -> dict:
    n = len(singles)
    cnt = {k: sum(1 for x in singles if x["status"] == k) for k in ("correct", "asked", "silent")}
    turns = [t for c in convs for t in c["turns"]]
    tc = {k: sum(1 for t in turns if t["status"] == k) for k in ("correct", "asked", "silent")}
    kept = [t["context_kept"] for t in turns if t["context_kept"] is not None]
    out = {"single": {"n": n, **cnt}, "conversation_turns": {"n": len(turns), **tc, "context_kept": f"{sum(kept)} of {len(kept)}"}}
    clar_items = [x for x in singles if x.get("expect_clarify")]
    if answers is not None:
        out["answers"] = {"n": len(answers), "ok": sum(1 for a in answers if a["ok"])}
    return out


def evaluate(which: str = "development", store=None, agent_factory=None, answers: bool = True, tmp: Path | None = None) -> dict:
    """Score a set. `which`: development | sealed. The sealed set is scored once; see the module docstring of evalset_v8."""
    import tempfile
    from strata.ai.agent import Agent
    from strata.data.store import MarketStore
    from strata.config import get_settings, override_settings
    store = store or MarketStore()
    tmp = tmp or Path(tempfile.mkdtemp(prefix="strata-eval-v8-"))
    before = get_settings().data_dir
    _fresh(tmp)
    try:
        return _evaluate(which, store, agent_factory, answers)
    finally:
        override_settings(data_dir=before)  # the conversations were stored in a scratch database; the settings go back


def _evaluate(which: str, store, agent_factory, answers: bool) -> dict:
    from strata.ai.agent import Agent
    singles_set = {"development": E.DEVELOPMENT, "probe": E.PROBE, "probe2": E.PROBE2, "sealed": E.SEALED}[which]
    convs_set = {"development": E.DEV_CONVERSATIONS, "probe": E.PROBE_CONVERSATIONS, "probe2": E.PROBE2_CONVERSATIONS, "sealed": E.SEALED_CONVERSATIONS}[which]
    fac = agent_factory or (lambda: Agent(NoJobs(), store, llm_factory=lambda: None))
    singles = run_singles(singles_set, store, fac)
    convs = run_conversations(convs_set, store, fac)
    ans = None
    if answers and which == "sealed":
        ans = run_answers(E.SEALED_ANSWERS, store, lambda: Agent(InlineJobs(), store, llm_factory=lambda: None))
    return {"which": which, "summary": summarise(singles, convs, ans), "singles": singles, "conversations": convs, "answers": ans, "sealed_hash": E.sealed_hash()}


def dump(res: dict, path: str) -> None:
    Path(path).write_text(json.dumps(res, indent=1, default=str))
