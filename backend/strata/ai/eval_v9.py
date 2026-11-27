"""Scoring the V9 sets (evalset_v9), with the V8 scorer and the V9 vocabulary (sizing, selections, next steps, offered
alternatives), and timing every reply. The reader is whatever the agent factory builds: the built-in readers alone, or a
language model first (model_reader.py). See evalset_v9 for the rules of the sealed part."""
from __future__ import annotations

import json
import time
from pathlib import Path

from strata.ai import eval_v8 as E8
from strata.ai import evalset_v9 as E


_CANON8 = E8.canon_expr


def _canon(expr: str, start):
    out = []
    for c in _CANON8(expr.replace("previous_session.open_to_close_pct", "PREVOTC").replace("vix_prev_close", "vix"), start):
        out.append(c.replace("expr:PREVOTC", "prev_otc").replace("PREVOTC", "prev_otc"))
    fixed = []
    for c in out:
        if c.startswith("prev_otc") and " " in c:
            c = c.replace(" ", "")
        fixed.append(c)
    return fixed


def facts_of_turn(msgs, state, version, store, history=None) -> dict:
    asst = [m for m in msgs if m.get("role") == "assistant"]
    c = (asst[0] if asst else {}).get("content") or {}
    route = c.get("route")
    if route == "clarify":
        opts = c.get("options") or []
        return {"response": "flag" if c.get("declined") else "clarify", "text": c.get("text", ""), "alternative": any(o.get("ask") for o in opts if isinstance(o, dict))}
    if route == "next_steps":
        return {"response": "ran", "type": "next_steps"}
    if route == "about_selection":
        return {"response": "ran", "type": "selection"}
    if route == "fills":
        return {"response": "ran", "type": "explore"}
    E8.canon_expr, keep = _canon, E8.canon_expr
    try:
        got = E8.facts_of_turn(msgs, state, version, store, history)
    finally:
        E8.canon_expr = keep
    if got.get("response") == "ran" and got.get("type") == "breakdown":
        fr = next(((m.get("content") or {}).get("frame") for m in asst if (m.get("content") or {}).get("frame")), None) or state.get("frame") or {}
        got["focus"] = fr.get("focus")  # the weekday a breakdown is asked about ("are fridays worse"); V8's scorer did not read it
    if got.get("response") == "ran" and got.get("type") == "strategy" and version is not None:
        from strata.ai.draft import Draft
        d = Draft.model_validate(version["draft"])
        sz = d.sizing
        if sz is not None:
            got["sizing"] = f"{sz.mode}:{float(sz.value):g}" if sz.mode != "contracts" else None
            got["capital"] = sz.capital
            got["reinvest"] = bool(sz.reinvest)
            got["daily_loss_stop"] = sz.daily_loss_stop
        got["entries"] = d.max_entries
        got["conditions"] = [x.replace(" ", "") for x in got.get("conditions") or []]
    return got


def score(item: dict, got: dict) -> tuple[str, list[str]]:
    exp = item["expect"]
    status, why = E8.score({**item, "expect": {k: v for k, v in exp.items() if k not in ("offers_alternative",)}}, got)
    if status == "correct" and exp.get("offers_alternative") and got.get("response") in ("flag", "clarify") and not got.get("alternative"):
        return "asked", ["declined without offering the closest question that can be answered"]
    return status, why


def run(items: list[dict], convs: list[dict], store, agent_factory) -> dict:
    from strata.api import db
    singles = []
    for it in items:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        t0 = time.time()
        try:
            msgs = ag.post(sid, it["text"])
            s, v, _ = ag._state(sid)
            got = facts_of_turn(msgs, s["state"], v, store)
            reader = next((((m.get("content") or {}).get("reader")) for m in msgs if (m.get("content") or {}).get("reader")), None)
        except Exception as e:  # noqa: BLE001
            got, reader = {"response": "error", "error": f"{type(e).__name__}: {e}"}, None
        st, why = score(it, got) if got.get("response") != "error" else ("silent", [got["error"]])
        singles.append({"text": it["text"], "level": it.get("level"), "category": it.get("category"), "supported": it.get("supported", True), "status": st, "why": why, "got": got,
                        "reader": reader, "seconds": round(time.time() - t0, 2)})
    conversations = []
    for cv in convs:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        turns, prev = [], None
        for t in cv["turns"]:
            t0 = time.time()
            try:
                msgs = ag.post(sid, t["text"])
                s, v, _ = ag._state(sid)
                got = facts_of_turn(msgs, s["state"], v, store, db.list_messages(sid))
            except Exception as e:  # noqa: BLE001
                got = {"response": "error", "error": f"{type(e).__name__}: {e}"}
            st, why = score(t, got) if got.get("response") != "error" else ("silent", [got["error"]])
            kept = None
            if prev and got.get("response") == "ran" and prev.get("response") == "ran" and got.get("type") == prev.get("type") and got.get("type") not in ("explore", "compare", "explain", "selection", "next_steps"):
                named = set(t["expect"]) | {"response"}
                kept = all(prev.get(k) == got.get(k) for k in prev if k not in named and k not in ("text",) and k in got)
            turns.append({"text": t["text"], "status": st, "why": why, "got": got, "context_kept": kept, "seconds": round(time.time() - t0, 2)})
            if got.get("response") == "ran" and got.get("type") not in ("explore", "compare", "explain", "reliability", "selection", "next_steps"):
                prev = got
        conversations.append({"id": cv["id"], "turns": turns})
    return {"singles": singles, "conversations": conversations, "summary": summarise(singles, conversations)}


def summarise(singles, convs) -> dict:
    cnt = lambda xs, k: sum(1 for x in xs if x["status"] == k)  # noqa: E731
    sup = [x for x in singles if x.get("supported") and not x.get("expect_clarify")]
    turns = [t for c in convs for t in c["turns"]]
    kept = [t["context_kept"] for t in turns if t["context_kept"] is not None]
    secs = sorted(x["seconds"] for x in singles + turns)
    return {"single": {"n": len(singles), "correct": cnt(singles, "correct"), "asked": cnt(singles, "asked"), "silent": cnt(singles, "silent")},
            "supported": {"n": len(sup), "correct": cnt(sup, "correct"), "silent": cnt(sup, "silent")},
            "turns": {"n": len(turns), "correct": cnt(turns, "correct"), "asked": cnt(turns, "asked"), "silent": cnt(turns, "silent"), "context_kept": f"{sum(kept)} of {len(kept)}"},
            "seconds": {"median": secs[len(secs) // 2] if secs else None, "p90": secs[int(len(secs) * 0.9)] if secs else None, "max": secs[-1] if secs else None}}


def evaluate(which: str, model: str | None = None, sets=None, policy: str | None = None) -> dict:
    """which: development | sealed (once) | v8dev | v8probe | v8probe2. model: an Ollama model name, or None for the built-in readers."""
    import tempfile
    from strata.ai.agent import Agent
    from strata.ai.evalset_v8 import DEV_CONVERSATIONS as V8DC, DEVELOPMENT as V8D, PROBE, PROBE2, PROBE2_CONVERSATIONS, PROBE_CONVERSATIONS
    from strata.config import get_settings, override_settings
    from strata.data.store import MarketStore
    store = MarketStore()
    pick = {"development": (E.DEVELOPMENT, E.DEV_CONVERSATIONS), "sealed": (E.SEALED, E.SEALED_CONVERSATIONS), "v8dev": (V8D, V8DC),
            "v8probe": (PROBE, PROBE_CONVERSATIONS), "v8probe2": (PROBE2, PROBE2_CONVERSATIONS)}
    items, convs = sets or pick[which]
    llm_factory = (lambda: None)
    if model:
        from strata.ai.llm import Ollama
        llm_factory = lambda: Ollama("http://127.0.0.1:11434", model, context_tokens=8192)  # noqa: E731
    before = get_settings().data_dir
    tmp = Path(tempfile.mkdtemp(prefix="strata-eval-v9-"))
    E8._fresh(tmp)
    from strata.api import settings as ST
    if policy:
        ST.set_setting("llm.mode", policy)
    try:
        res = run(items, convs, store, lambda: Agent(E8.NoJobs(), store, llm_factory=llm_factory))
    finally:
        override_settings(data_dir=before)
    res.update(which=which, model=model or "built-in readers", policy=policy or "auto", sealed_hash=E.sealed_hash())
    return res


def dump(res: dict, path: str) -> None:
    Path(path).write_text(json.dumps(res, indent=1, default=str))


TOL = {"pct": 0.15, "rest_pct": 0.15, "day_pct": 0.15, "k": 0.0, "mean": 0.006, "trades": 0.0, "pnl": 0.011, "share": 0.15}


def run_answers(items: list[dict], store, agent_factory) -> list[dict]:
    """Each question run on the stored history; every number the item names must appear in the reply within its tolerance of
    the independent reference (strata.research.reference.answer_value), and no forbidden statement may. The reply scored
    is the answer's own words (headline, what happened, the story): the independent-check line is left out, so a reply
    cannot pass by quoting the reference it differs from."""
    from strata.api import db
    from strata.research.reference import Reference, answer_value
    R = Reference()
    out = []
    for it in items:
        ag = agent_factory()
        sid = db.create_session("eval")["id"]
        t0 = time.time()
        texts = []
        try:
            for tt in it.get("turns") or [it["text"]]:
                ag.post(sid, tt)
                for _ in range(4):
                    ag.finalize(sid)
            for m in db.list_messages(sid):
                if m["role"] != "assistant":
                    continue
                c = m["content"] if isinstance(m["content"], dict) else {}
                a = c.get("answer") or {}
                texts.append(" ".join([c.get("text") or ""] + [a.get("headline") or ""] + list(a.get("happened") or []) + list((a.get("story") or {}).get("lines") or [])
                                      + [str(n.get("value")) + " " + str(n.get("label")) for n in a.get("numbers") or [] if isinstance(n, dict)]))
            reply = " ".join(texts)
        except Exception as e:  # noqa: BLE001
            reply = f"ERROR {type(e).__name__}: {e}"
        try:
            ref = answer_value(it["reference"], it.get("args") or {}, R)
        except Exception as e:  # noqa: BLE001
            ref = {"error": f"{type(e).__name__}: {e}"}
        nums = E8.numbers_in(reply)
        checks = []
        for key in it.get("numbers") or []:
            v = ref.get(key)
            if v is None:
                checks.append((key, False, "no reference value"))
                continue
            tol = TOL.get(key, 0.15)
            ok = any(abs(abs(n_) - abs(float(v))) <= tol + 1e-9 for n_ in nums)
            checks.append((key, ok, f"reference {float(v):.4f}"))
        for key in it.get("names") or []:
            want = str(ref.get(key) or "")
            ok = bool(want) and (want.lower() in reply.lower() or (key == "bucket" and want in reply))
            checks.append((key, ok, f"reference {want}"))
        bad = [f for f in E.FORBIDDEN if f in reply.lower()]
        checks.append(("forbidden statements", not bad, ", ".join(bad)))
        out.append({"id": it["id"], "text": it.get("text"), "ok": all(c[1] for c in checks), "checks": checks, "reference": {k: v for k, v in ref.items() if k not in ("per_trade", "skipped", "dates", "hits")},
                    "reply": reply[:1500], "seconds": round(time.time() - t0, 1)})
    return out


def evaluate_answers(which: str = "sealed", model: str | None = None, policy: str | None = None) -> dict:
    import tempfile
    from strata.ai.agent import Agent
    from strata.config import get_settings, override_settings
    from strata.data.store import MarketStore
    store = MarketStore()
    items = E.SEALED_ANSWERS if which == "sealed" else getattr(E, "DEV_ANSWERS", [])
    llm_factory = (lambda: None)
    if model:
        from strata.ai.llm import Ollama
        llm_factory = lambda: Ollama("http://127.0.0.1:11434", model, context_tokens=8192)  # noqa: E731
    before = get_settings().data_dir
    tmp = Path(tempfile.mkdtemp(prefix="strata-eval-v9a-"))
    E8._fresh(tmp)
    from strata.api import settings as ST
    if policy:
        ST.set_setting("llm.mode", policy)
    try:
        res = run_answers(items, store, lambda: Agent(E8.InlineJobs(), store, llm_factory=llm_factory))
    finally:
        override_settings(data_dir=before)
    return {"which": which, "model": model or "built-in readers", "policy": policy or "auto", "answers": res, "ok": sum(1 for a in res if a["ok"]), "n": len(res), "sealed_hash": E.sealed_hash()}
