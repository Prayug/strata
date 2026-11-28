"""V10 evaluation: a permanent regression set, independently written development questions, and an untouched held-out set.

Three kinds of data, kept apart (docs/v10_evidence/eval/):

  regression   every sentence of the V8 and V9 sets (their development, probe and, now that they have been scored, sealed
               parts), with the expectations they were written with. These were written by the developer and have all been
               seen, so they measure only that what worked keeps working. They are scored by machine (eval_v8/eval_v9).
  development  questions written for V10 by separate agents that never saw the code (dev/*.jsonl). The developer reads these
               and may build against them.
  held-out     questions by the same authors plus one whose voice is in no development file (heldout/*.jsonl). The developer
               does not read them before the final scoring; `seal` hashes them without printing a word, `observe_file` runs
               them through the application, and separate judge agents give the verdicts (judging_rubric.md).

The authored files carry what the author meant in plain English, not a machine expectation, so `observe` records what the
application did in a form a person (or a judge) can check: whether it answered, waited, asked or declined; its reading of
the request as the user saw it; what was set up to run; the reply.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[3] / "docs" / "v10_evidence" / "eval"


# ------------------------------------------------------------------------------------------------------- regression
def regression_sets() -> dict[str, tuple[list[dict], list[dict]]]:
    from strata.ai import evalset_v8 as E8S
    from strata.ai import evalset_v9 as E9S
    return {
        "v8_development": (E8S.DEVELOPMENT, E8S.DEV_CONVERSATIONS),
        "v8_probe": (E8S.PROBE, E8S.PROBE_CONVERSATIONS),
        "v8_probe2": (E8S.PROBE2, E8S.PROBE2_CONVERSATIONS),
        "v8_sealed": (E8S.SEALED, E8S.SEALED_CONVERSATIONS),
        "v9_development": (E9S.DEVELOPMENT, E9S.DEV_CONVERSATIONS),
        "v9_probe": (E9S.PROBE, E9S.PROBE_CONVERSATIONS),
        "v9_sealed": (E9S.SEALED, E9S.SEALED_CONVERSATIONS),
    }


COMPLEX_LINE = "This has several parts, so check that each one is what you meant."
CONFIRMED: list[str] = []  # messages whose reading waited only because it has several parts, and was then told to run


def _agent_factory(model: str | None, store, fast: bool = False, confirm: bool = False):
    """`fast`: skip the check of which sessions have stored option prices (seconds per strategy; it does not change how a
    message is read). For development runs only; the held-out set is run as the application runs.

    `confirm`: the machine-scored regression sets expect a run. A reading that is complete but waits because it has several
    parts (V10 confirms those before running) is given its go-ahead here, as a user would give it, and counted in
    `CONFIRMED`; a reading that waits because something in it is wrong or missing is never confirmed."""
    from strata.ai import eval_v8 as E8
    from strata.ai.agent import Agent
    llm_factory = (lambda: None)
    if model:
        from strata.ai.llm import Ollama
        llm_factory = lambda: Ollama("http://127.0.0.1:11434", model, context_tokens=8192)  # noqa: E731

    def make(jobs=None):
        ag = Agent(jobs or E8.NoJobs(), store, llm_factory=llm_factory)
        if fast:
            ag._data_check = lambda *a, **k: None
        if confirm:
            from strata.api import db
            post = ag.post

            def post_and_confirm(sid, text):
                msgs = post(sid, text)
                asst = [m for m in msgs if m.get("role") == "assistant"]
                c = (asst[0].get("content") or {}) if asst else {}
                if (c.get("confirm") or {}).get("complex") and asst[0].get("kind") == "graph":
                    CONFIRMED.append(text)  # a research plan that waits only for its go-ahead
                    ag.decide_graph(sid, asst[0]["id"], "run")
                    first_id = min(m["id"] for m in msgs)
                    return [m for m in db.list_messages(sid) if m["id"] >= first_id]
                if (c.get("confirm") or {}).get("complex"):
                    CONFIRMED.append(text)
                    return post(sid, "Run it as read")
                plan = next((m for m in asst if m.get("kind") == "plan"), None)
                pc = (plan or {}).get("content") or {}
                if pc.get("state") == "awaiting_confirmation" and (pc.get("draft") or {}).get("review") == [COMPLEX_LINE] and not (pc.get("draft") or {}).get("unsupported"):
                    CONFIRMED.append(text)
                    ag.confirm(sid, pc["version_id"])
                    first_id = min(m["id"] for m in msgs)
                    return [m for m in db.list_messages(sid) if m["id"] >= first_id]
                return msgs
            ag.post = post_and_confirm
        return ag
    return make


class _Sandbox:
    """A fresh database in a temporary folder; the stored market history is read where it is."""

    def __init__(self, policy: str | None = None, zone_confirmed: bool = False):
        self.policy, self.zone_confirmed = policy, zone_confirmed

    def __enter__(self):
        from strata.ai import eval_v8 as E8
        from strata.api import settings as ST
        from strata.config import get_settings
        self.before = get_settings().data_dir
        E8._fresh(Path(tempfile.mkdtemp(prefix="strata-eval-v10-")))
        if self.policy:
            ST.set_setting("llm.mode", self.policy)
        if self.zone_confirmed:
            ST.set_setting("ui.timezone_confirmed", True)  # the V8 and V9 sets were written for a user whose zone is known
        return self

    def __exit__(self, *exc):
        from strata.config import override_settings
        override_settings(data_dir=self.before)
        return False


def run_regression(model: str | None = None, policy: str | None = None, names: list[str] | None = None) -> dict:
    """Every regression set through the application, scored by the V8/V9 scorer. A reply that asks or declines where a run
    was expected is counted on its own (`asked`), never as correct and never as silent."""
    from strata.ai import eval_v9 as E9
    from strata.data.store import MarketStore
    store = MarketStore()
    out: dict = {"model": model or "built-in readers", "policy": policy or "auto", "sets": {}}
    tot = {"n": 0, "correct": 0, "asked": 0, "silent": 0}
    with _Sandbox(policy, zone_confirmed=True):
        CONFIRMED.clear()
        make = _agent_factory(model, store, fast=True, confirm=True)
        for name, (items, convs) in regression_sets().items():
            if names and name not in names:
                continue
            res = E9.run(items, convs, store, make)
            rows = res["singles"] + [t for c in res["conversations"] for t in c["turns"]]
            s = {k: sum(1 for x in rows if x["status"] == k) for k in ("correct", "asked", "silent")}
            s["n"] = len(rows)
            out["sets"][name] = {"summary": s, "singles": res["singles"], "conversations": res["conversations"]}
            for k in tot:
                tot[k] += s[k]
    tot["confirmed"] = len(CONFIRMED)  # of all turns: readings that were complete, waited for a go-ahead, and got it
    out["total"] = tot
    out["confirmed"] = list(CONFIRMED)
    return out


def regression_failures(res: dict) -> list[dict]:
    rows = []
    for name, s in res["sets"].items():
        for x in s["singles"]:
            if x["status"] != "correct":
                rows.append({"set": name, "text": x["text"], "status": x["status"], "why": x["why"]})
        for c in s["conversations"]:
            for i, t in enumerate(c["turns"]):
                if t["status"] != "correct":
                    rows.append({"set": name, "text": f"[{c['id']} turn {i + 1}] {t['text']}", "status": t["status"], "why": t["why"]})
    return rows


# --------------------------------------------------------------------------------------------------- authored sets
REQUIRED = ("text", "intent", "expected", "if_asked", "also_fine", "complexity")


def load_authored(path: Path) -> list[dict]:
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def check_authored(path: Path) -> dict:
    """The shape of an authored file, without a word of its content: counts and what is malformed, by line number."""
    problems, n_items, n_turns, n_conv, kinds = [], 0, 0, 0, {"answer": 0, "ask": 0, "decline": 0}
    cx = {"simple": 0, "complex": 0}
    ids = set()
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {ln}: not JSON ({e.msg})")
            continue
        n_items += 1
        if not it.get("id") or it["id"] in ids:
            problems.append(f"line {ln}: missing or repeated id")
        ids.add(it.get("id"))
        turns = it.get("turns")
        if not isinstance(turns, list) or not turns:
            problems.append(f"line {ln}: no turns")
            continue
        n_conv += len(turns) > 1
        for i, t in enumerate(turns):
            n_turns += 1
            miss = [k for k in REQUIRED if k not in t]
            if miss:
                problems.append(f"line {ln} turn {i + 1}: missing {', '.join(miss)}")
            if not str(t.get("text") or "").strip() or not str(t.get("intent") or "").strip():
                problems.append(f"line {ln} turn {i + 1}: empty text or intent")
            if t.get("expected") not in kinds:
                problems.append(f"line {ln} turn {i + 1}: expected is not answer|ask|decline")
            else:
                kinds[t["expected"]] += 1
            if t.get("complexity") in cx:
                cx[t["complexity"]] += 1
            else:
                problems.append(f"line {ln} turn {i + 1}: complexity is not simple|complex")
    return {"file": Path(path).name, "items": n_items, "conversations": n_conv, "turns": n_turns, "expected": kinds, "complexity": cx, "problems": problems}


def observe(agent, sid: str, text: str, store) -> dict:
    """One user turn through the application, and what the user would have seen come back."""
    from strata.ai import eval_v9 as E9
    from strata.api import db
    t0 = time.time()
    try:
        msgs = agent.post(sid, text)
        s, v, _ = agent._state(sid)
        got = E9.facts_of_turn(msgs, s["state"], v, store, db.list_messages(sid))
    except Exception as e:  # noqa: BLE001
        return {"disposition": "error", "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 2)}
    asst = [m for m in msgs if m.get("role") == "assistant"]
    first = asst[0] if asst else {}
    c = first.get("content") or {}
    plan = next((m for m in asst if m.get("kind") == "plan"), None)
    pc = (plan or {}).get("content") or {}
    question = None
    if pc.get("state") == "awaiting_answer":
        qs = [q for q in (pc.get("questions") or (pc.get("draft") or {}).get("questions") or []) if q.get("answered") is None]
        question = {"text": (qs[0]["text"] if qs else pc.get("text", "")), "options": [o.get("label") for o in (qs[0].get("options") if qs else [])]}
        disp = "asked"
    elif pc.get("state") == "awaiting_confirmation" or c.get("confirm"):
        disp = "waiting"  # the reading is shown, with the reasons it waits, and a go-ahead runs it
        if c.get("confirm"):
            question = {"text": c.get("text", ""), "options": [(o.get("label") if isinstance(o, dict) else o) for o in (c.get("options") or [])]}
    elif first.get("kind") == "graph":
        disp = {"running": "answered", "blocked": "declined"}.get(c.get("state"), "asked")
    elif got.get("response") == "flag" and c.get("declined"):
        disp = "declined"
    elif got.get("response") in ("clarify", "flag"):
        disp = "asked"
        question = {"text": c.get("text", ""), "options": [(o.get("label") if isinstance(o, dict) else o) for o in (c.get("options") or [])]}
    elif got.get("response") == "ran":
        disp = "answered"
    else:
        disp = "error"
    reading = next((m["content"].get("reading") for m in asst if (m.get("content") or {}).get("reading")), None)
    dr = pc.get("draft") or {}
    draft = {k: v for k, v in dr.items() if k in ("symbol", "structure", "dte", "strike", "width", "quantity", "entry_time", "exit_time", "weekdays", "premium_target_pct", "premium_stop_pct", "underlying_target_pct",
                                                    "underlying_stop_pct", "hold_minutes", "hold_sessions", "hold_to_expiration", "max_entries", "sizing", "start", "end", "review") and v not in (None, [], False, {})}
    if dr.get("conditions"):
        draft["conditions"] = [c.get("expr") for c in dr["conditions"]]
    unsupported = pc.get("unsupported") or (pc.get("draft") or {}).get("unsupported") or []
    return {"disposition": disp, "route": c.get("route"), "kind": first.get("kind"), "reading": reading, "question": question, "not_represented": list(unsupported), "draft": draft,
            "computed": {k: x for k, x in got.items() if k not in ("text",)}, "reply": str(c.get("text") or "")[:700],
            "options": [(o.get("label") if isinstance(o, dict) else o) for o in (c.get("options") or [])],
            "interpreter": c.get("interpreter") or pc.get("interpreter"), "seconds": round(time.time() - t0, 2)}


def observe_items(items: list[dict], model: str | None = None, policy: str | None = None, fast: bool = False) -> list[dict]:
    """Every turn of every authored item, each item in a conversation of its own. Interpretation only: nothing is executed."""
    from strata.api import db
    from strata.data.store import MarketStore
    store = MarketStore()
    out = []
    with _Sandbox(policy):
        make = _agent_factory(model, store, fast=fast)
        for it in items:
            ag = make()
            sid = db.create_session("eval")["id"]
            turns = []
            for t in it["turns"]:
                turns.append({"text": t["text"], **{k: t.get(k) for k in ("intent", "expected", "if_asked", "also_fine", "complexity")}, "saw": observe(ag, sid, t["text"], store)})
            out.append({"id": it.get("id"), "turns": turns})
    return out


def show(observed: list[dict], width: int = 230) -> str:
    """A compact page for reading development results: what was typed, what was meant, what the application did."""
    lines = []
    for it in observed:
        for i, t in enumerate(it["turns"]):
            s = t["saw"]
            lines.append(f"{it['id']}{'.' + str(i + 1) if len(it['turns']) > 1 else ''} [{t.get('expected')}/{t.get('complexity')}] {t['text']}"[:width])
            if t.get("intent"):
                lines.append(f"    meant: {t['intent']}"[:width])
            what = s.get("reading") or (s.get("question") or {}).get("text") or s.get("reply") or s.get("error") or ""
            lines.append(f"    {s['disposition'].upper()} ({s.get('route')}, {s.get('seconds')}s): {what}"[:width])
            comp = {k: v for k, v in (s.get("computed") or {}).items() if k != "response" and v not in (None, [], "", False)}
            if comp:
                lines.append(f"    computed: {json.dumps(comp, default=str)}"[:width])
            if s.get("not_represented"):
                lines.append(f"    not represented: {s['not_represented']}"[:width])
    return "\n".join(lines)


def seal(paths: list[Path]) -> dict:
    """SHA-256 of each held-out file and of them all, in name order. Nothing of their content is printed or returned."""
    files = {}
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        b = Path(p).read_bytes()
        files[p.name] = hashlib.sha256(b).hexdigest()
        h.update(p.name.encode() + b"\0" + b + b"\0")
    return {"files": files, "all": h.hexdigest()}
