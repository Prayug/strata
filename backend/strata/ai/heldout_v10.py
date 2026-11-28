"""The V10 held-out scoring run (docs/v10_evidence/eval/judging_rubric.md).

The developer runs this module and never reads what it handles: every command prints counts, never a question. The
held-out questions go through the application as installed, each item in a notebook of its own, with every answered turn
executed for real on the stored history. Where the application asks or waits, the turn pauses for a SIMULATED USER (a
separate agent holding the author's intent), who answers through this module's `reply` command, at most twice per turn.
`export` then writes, for the judges, what the user saw at each stage.

    python -m strata.ai.heldout_v10 start <config> [--shadow-model M --shadow-variant V] [--source DIR]
    python -m strata.ai.heldout_v10 advance <config>            run every item until it pauses or ends (counts only)
    python -m strata.ai.heldout_v10 pending <config> [--part i/n]   the paused turns, for the simulated users (JSON)
    python -m strata.ai.heldout_v10 reply <config> <key> (--option N | --confirm | --say TEXT)
    python -m strata.ai.heldout_v10 export <config>             judge packets (a file), counts printed
    python -m strata.ai.heldout_v10 status <config>

A run directory is created once; `start` refuses to overwrite one (the set is scored once per configuration).

The second registered configuration (the application plus a local model as a proposer) is measured in the same run, as a
SHADOW: on every turn where the application stops (asks, waits because something is wrong or missing, or does not
understand), the rewrite the proposer would have offered is computed and stored out of sight. The simulated user never
sees it and the application's own behaviour is not changed by it; the judges are later asked whether each stored offer
says what the author meant. That measures what the proposer would add without running every conversation twice.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import sys
import time
from pathlib import Path

from strata.ai import eval_v10 as V

RUN_ROOT = V.EVAL_DIR.parent / "heldout_run"
MAX_REPLIES = 2


@contextlib.contextmanager
def _locked(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load(d: Path) -> dict:
    return json.loads((d / "state.json").read_text())


def _save(d: Path, st: dict) -> None:
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(st, indent=1, default=str))
    tmp.replace(d / "state.json")


@contextlib.contextmanager
def _app(d: Path, st: dict):
    """The application on the run's own database (kept between commands), reading the stored market history."""
    from strata.ai import eval_v8 as E8
    from strata.ai.agent import Agent
    from strata.api import settings as ST
    from strata.config import get_settings, override_settings
    from strata.data.store import MarketStore
    before = get_settings().data_dir
    store = MarketStore()
    sb = d / "sandbox"
    sb.mkdir(exist_ok=True)
    E8._fresh(sb)
    ST._cache.clear()
    llm_factory = (lambda: None)  # the application as installed: no language model reads or proposes
    try:
        yield Agent(E8.InlineJobs(), store, llm_factory=llm_factory), store
    finally:
        override_settings(data_dir=before)


# ------------------------------------------------------------------------------------------------ what the user saw
def view_of(agent, sid: str, since_id: int, store, seconds: float) -> dict:
    """Everything the application showed for the messages from `since_id` on, as the notebook shows it."""
    from strata.ai import eval_v9 as E9
    from strata.api import db
    for _ in range(6):  # finished runs become result messages
        if not agent.finalize(sid):
            break
    msgs = [m for m in db.list_messages(sid) if m["id"] >= since_id]
    asst = [m for m in msgs if m["role"] == "assistant"]
    s, v, _ = agent._state(sid)
    try:
        got = E9.facts_of_turn(msgs, s["state"], v, store, db.list_messages(sid))
    except Exception as e:  # noqa: BLE001
        got = {"response": "error", "text": f"{type(e).__name__}: {e}"}
    first = asst[0] if asst else {}
    c = first.get("content") if isinstance(first.get("content"), dict) else {}
    plan = next((m for m in reversed(asst) if m["kind"] == "plan"), None)
    pc = (plan or {}).get("content") or {}
    results = [m for m in asst if m["kind"] == "result"]
    choices: list[dict] = []
    question = None
    confirm = False
    decision = next((m for m in reversed(asst) if m["kind"] == "decision"), None)
    if decision is not None and not results:
        # the run stopped to ask how to proceed (the stored data does not cover it as asked): a question with choices
        dc = decision["content"] or {}
        disp, question = "asked", dc.get("text")
        choices = [{"index": i, "label": x.get("label"), "how": {"kind": "decide", "message_id": decision["id"], "choice": x.get("id")}} for i, x in enumerate(dc.get("choices") or [])]
    elif results:
        disp = "answered"
    elif pc.get("state") == "awaiting_answer":
        qs = [q for q in (pc.get("questions") or []) if q.get("answered") is None]
        disp = "asked"
        if qs:
            question = qs[0]["text"]
            choices = [{"index": i, "label": o.get("label"), "how": {"kind": "answer", "version_id": pc["version_id"], "question_id": qs[0]["id"], "option": i}} for i, o in enumerate(qs[0].get("options") or [])]
    elif pc.get("state") == "awaiting_confirmation":
        disp, confirm = "waiting", True
        question = pc.get("text")
    elif pc.get("state") in ("running", "ready") and not results:
        disp = "error" if any((m.get("content") or {}).get("view") == "failed" for m in asst) else "answered"  # a run that produced no result message is reported as it is
    elif first.get("kind") == "graph" and c.get("state") == "awaiting" and c.get("confirm"):
        disp, confirm, question = "waiting", True, c.get("text")  # a research plan shown as its sentence, waiting for its go-ahead
    elif first.get("kind") == "graph" and c.get("state") in ("awaiting", "needs_data"):
        disp, question = "asked", c.get("text")
        choices = [{"index": i, "label": x.get("label"), "how": {"kind": "graph", "message_id": first["id"], "choice": x.get("id")}} for i, x in enumerate(c.get("choices") or [])]
    elif c.get("declined"):
        disp = "declined"
    elif c.get("confirm"):
        disp, confirm, question = "waiting", True, c.get("text")
        c = {**c, "options": []}  # its one choice is the go-ahead button
    elif c.get("clarify") or got.get("response") in ("clarify", "flag"):
        disp, question = "asked", c.get("text")
    elif got.get("response") == "ran":
        disp = "answered"
    else:
        disp, question = ("asked", c.get("text")) if c.get("text") else ("error", None)
    if not choices and c.get("options") and disp in ("asked", "waiting", "declined"):
        choices = [{"index": i, "label": o.get("label"), "how": {"kind": "say", "text": o.get("ask") or o.get("label")}} for i, o in enumerate(c["options"])]
    reading = next((m["content"].get("reading") for m in asst if (m.get("content") or {}).get("reading")), None)
    sentence = ((pc.get("sentence") or {}).get("lines")) if pc else None
    answers = []
    for m in results:
        rc = m["content"] or {}
        a = rc.get("answer") or {}
        answers.append({"headline": a.get("headline") or rc.get("text"), "what_happened": a.get("happened"), "why_it_matters": a.get("matters"),
                        "numbers": [{"label": n.get("label"), "value": n.get("value"), "detail": n.get("detail")} for n in (a.get("numbers") or []) if isinstance(n, dict)],
                        "independent_check": ({"status": a["check"].get("status"), "text": a["check"].get("text")} if isinstance(a.get("check"), dict) else None),
                        "what_this_cannot_tell_you": a.get("limits"), "the_money_at_risk": (a.get("sizing") or {}).get("lines"), "view": rc.get("view"), "failed": rc.get("view") == "failed"})
    return {"disposition": disp, "reading_shown": reading or (" ".join(sentence) if sentence else None), "strategy_sentence_more": (pc.get("sentence") or {}).get("more") if pc else None,
            "assumed_shown": next((m["content"].get("assumed") for m in asst if (m.get("content") or {}).get("assumed")), None) or [a.get("value") for a in (pc.get("assumptions") or []) if isinstance(a, dict) and a.get("field") == "reading"] or None,
            "defaults_filled_in": pc.get("missing") or None,
            "left_out_shown": next((m["content"].get("left_out") for m in asst if (m.get("content") or {}).get("left_out")), None), "not_represented_shown": (pc.get("draft") or {}).get("unsupported") or pc.get("unsupported") or None,
            "change_shown": next(((m["content"].get("change") or {}).get("text") for m in asst if (m.get("content") or {}).get("change")), None),
            "question_or_message_shown": question, "choices_shown": [x["label"] for x in choices], "go_ahead_button_shown": confirm,
            "why_it_waits": (pc.get("draft") or {}).get("review") or (c.get("confirm") or {}).get("reasons") or None,
            "answers": answers, "other_text_shown": [str((m.get("content") or {}).get("text") or "")[:900] for m in asst if m["kind"] in ("text", "trades", "decision") and not (m is first and question)][:3] or None,
            "technical_summary_of_what_was_set_up": {k: x for k, x in got.items() if k not in ("text",)}, "seconds": round(seconds, 1),
            "_choices": choices, "_confirm": ({"kind": "confirm", "version_id": pc.get("version_id")} if pc.get("state") == "awaiting_confirmation" else
                                              {"kind": "graph", "message_id": first["id"], "choice": "run"} if (first.get("kind") == "graph" and c.get("confirm")) else
                                              {"kind": "say", "text": "Run it as read"} if c.get("confirm") else None)}


def _post(agent, sid: str, how: dict) -> None:
    if how["kind"] == "say":
        agent.post(sid, how["text"])
    elif how["kind"] == "answer":
        agent.answer(sid, how["version_id"], how["question_id"], how["option"])
    elif how["kind"] == "confirm":
        agent.confirm(sid, how["version_id"])
    elif how["kind"] == "graph":
        agent.decide_graph(sid, how["message_id"], how["choice"])
    elif how["kind"] == "decide":
        agent.decide(sid, how["message_id"], how["choice"])


def _shadow(agent, st: dict, sid: str, text: str, view: dict, first_turn: bool) -> dict | None:
    """The rewrite the proposer (configuration B) would have offered on a turn where the application stopped. Stored with
    the turn, never shown to the simulated user, and without any effect on the application's own reply."""
    sh = st.get("shadow") or {}
    if not sh.get("model"):
        return None
    why = view.get("why_it_waits") or []
    stopped = view.get("disposition") == "asked" or (view.get("disposition") == "waiting" and (view.get("not_represented_shown") or [w for w in why if "several parts" not in w]))
    if not stopped or view.get("_choices") and any(c["how"].get("kind") in ("decide", "answer") for c in view["_choices"]):
        return None  # answered, declined, waiting only for a go-ahead, or asking about data or one detail with its own choices
    try:
        from strata.ai import intent as I
        from strata.ai import proposer as P
        from strata.ai import semantics as SM
        from strata.ai.llm import Ollama
        llm = Ollama("http://127.0.0.1:11434", sh["model"], context_tokens=4096)
        s, _, cur = agent._state(sid)
        fr = s["state"].get("frame")
        current = None if first_turn else (SM.describe(fr) if fr and fr.get("kind") != "strategy" else (" ".join(I.describe_draft(cur)["lines"]) if cur is not None and cur.kind == "options" else None))
        res = P.propose(agent, llm, sid, text, current, sh.get("variant") or "schema")
        return {"offer": res.get("offer"), "why": res.get("why"), "seconds": res.get("seconds")}
    except Exception as e:  # noqa: BLE001
        return {"offer": None, "why": f"{type(e).__name__}: {e}"[:160], "seconds": None}


def _next_id(sid: str) -> int:
    from strata.api import db
    ms = db.list_messages(sid)
    return (max(m["id"] for m in ms) + 1) if ms else 0


def _run_item(agent, store, st: dict, key: str) -> None:
    """One item forward from where it stands, until a turn pauses for the simulated user or the item ends."""
    from strata.api import db
    it = st["items"][key]
    turns = st["source"][key]
    if it.get("sid") is None:
        it["sid"] = db.create_session("heldout")["id"]
    while it["turn"] < len(turns) and not it.get("paused"):
        t = turns[it["turn"]]
        rec = st["records"].setdefault(f"{key}#{it['turn'] + 1}", {"stages": [], "replies": []})
        since = _next_id(it["sid"])
        t0 = time.time()
        try:
            agent.post(it["sid"], t["text"])
            view = view_of(agent, it["sid"], since, store, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            view = {"disposition": "error", "error": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 1), "_choices": [], "_confirm": None}
        sh = _shadow(agent, st, it["sid"], t["text"], view, it["turn"] == 0) if "error" not in view else None
        if sh is not None:
            rec["_shadow"] = sh
        rec["stages"].append(view)
        if view["disposition"] in ("asked", "waiting"):
            it["paused"] = True
        else:
            it["turn"] += 1
    if it["turn"] >= len(turns):
        it["done"] = True


def _screen(v: dict) -> dict:
    """What is on screen, for the simulated user, with an answer's long texts cut: a packet has to fit in one screen of a
    terminal tool (a rehearsal showed that a longer one is cut off, and the reply is then given half blind)."""
    out = {k: x for k, x in v.items() if not k.startswith("_") and x not in (None, [], "")}
    if out.get("answers"):
        out["answers"] = [{"headline": str(a_.get("headline") or "")[:500], "what_happened": [str(x)[:300] for x in (a_.get("what_happened") or [])[:3]],
                           "independent_check": a_.get("independent_check"), "failed": a_.get("failed")} for a_ in out["answers"][-2:]]
    for k in ("question_or_message_shown", "reading_shown"):
        if isinstance(out.get(k), str):
            out[k] = out[k][:1800]
    out.pop("technical_summary_of_what_was_set_up", None)  # the user sees the screen, not the configuration
    return out


def _packet(st: dict, key: str) -> dict:
    """What the simulated user is given for a paused turn: the conversation, what the author meant, what is on screen."""
    it = st["items"][key]
    turns = st["source"][key]
    i = it["turn"]
    rec = st["records"][f"{key}#{i + 1}"]
    last = rec["stages"][-1]
    pub = _screen

    def earlier(j: int) -> dict:
        last_ = st["records"][f"{key}#{j + 1}"]["stages"][-1]
        ans = (last_.get("answers") or [{}])[-1]
        return {"user": turns[j]["text"], "strata": {"disposition": last_.get("disposition"), "reading_shown": str(last_.get("reading_shown") or "")[:600], "headline": str(ans.get("headline") or "")[:300]}}
    return {"key": f"{key}#{i + 1}", "earlier_turns_of_this_conversation": [earlier(j) for j in range(i)],
            "user_message": turns[i]["text"], "author": {k: turns[i].get(k) for k in ("intent", "expected", "if_asked", "also_fine", "complexity")},
            "replies_already_given_this_turn": rec["replies"], "replies_left": MAX_REPLIES - len(rec["replies"]), "on_screen_now": pub(last),
            "how_to_reply": {"choices": [{"option": c["index"], "label": c["label"]} for c in last.get("_choices") or []], "go_ahead_button": bool(last.get("_confirm")), "or": "type a short message"}}


def cmd_start(a) -> None:
    d = RUN_ROOT / a.config
    if (d / "state.json").exists():
        sys.exit(f"{d} exists: the held-out set is scored once per configuration.")
    src = Path(a.source) if a.source else V.EVAL_DIR / "heldout"
    files = sorted(src.glob("*.jsonl"))
    if not a.source:
        seal = V.seal(files)
        want = json.loads((V.EVAL_DIR / "heldout_seal.json").read_text())["all"]
        if seal["all"] != want:
            sys.exit("The held-out files do not match their seal.")
    source, items = {}, {}
    for f in files:
        for it in V.load_authored(f):
            key = f"{f.stem}/{it['id']}"
            source[key] = it["turns"]
            items[key] = {"sid": None, "turn": 0, "paused": False, "done": False}
    with _locked(d):
        _save(d, {"config": a.config, "shadow": {"model": a.shadow_model, "variant": a.shadow_variant or "schema"} if a.shadow_model else None, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "source_dir": str(src), "source": source, "items": items, "records": {}})
    print(f"started {a.config}: {len(items)} items, {sum(len(v) for v in source.values())} turns")


def cmd_advance(a) -> None:
    d = RUN_ROOT / a.config
    with _locked(d):
        st = _load(d)
        t0 = time.time()
        with _app(d, st) as (agent, store):
            todo = [k for k, it in st["items"].items() if not it["done"] and not it["paused"]]
            for n, k in enumerate(todo):
                _run_item(agent, store, st, k)
                if n % 5 == 4:
                    _save(d, st)
                    print(f"  {n + 1}/{len(todo)} items, {time.time() - t0:.0f}s", flush=True)
        _save(d, st)
    cmd_status(a)


def cmd_status(a) -> None:
    st = _load(RUN_ROOT / a.config)
    its = st["items"].values()
    disp: dict = {}
    for r in st["records"].values():
        for i, v in enumerate(r["stages"]):
            disp[(i + 1, v["disposition"])] = disp.get((i + 1, v["disposition"]), 0) + 1
    print(f"{st['config']}: items {len(st['items'])}, done {sum(1 for x in its if x['done'])}, paused {sum(1 for x in its if x['paused'])}, turns recorded {len(st['records'])}")
    print("  (stage, disposition):", ", ".join(f"{k[0]}:{k[1]}={n}" for k, n in sorted(disp.items())))


def cmd_pending(a) -> None:
    """The turns waiting for a simulated user. A part is a FIXED share of the items (by their place among all items, so a
    conversation stays with one user and no turn moves between users while they work). One turn is printed at a time."""
    st = _load(RUN_ROOT / a.config)
    part_of = {k: j for j, k in enumerate(sorted(st["items"]))}
    keys = sorted(k for k, it in st["items"].items() if it["paused"])
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        keys = [k for k in keys if part_of[k] % n == i - 1]
    if a.count:
        print(len(keys))
        return
    if a.all:
        print(json.dumps([_packet(st, k) for k in keys], indent=1, default=str))
        return
    if not keys:
        print(json.dumps({"waiting_for_you": 0, "next": None}))
        return
    print(json.dumps({"waiting_for_you": len(keys), "next": _packet(st, keys[0])}, indent=1, default=str))


def cmd_reply(a) -> None:
    d = RUN_ROOT / a.config
    key, turn = a.key.rsplit("#", 1)
    with _locked(d):
        st = _load(d)
        it = st["items"].get(key)
        if it is None or not it["paused"] or it["turn"] + 1 != int(turn):
            sys.exit("That turn is not waiting for a reply.")
        rec = st["records"][a.key]
        last = rec["stages"][-1]
        if a.confirm:
            how, said = last.get("_confirm"), "(pressed the go-ahead button)"
            if not how:
                sys.exit("No go-ahead button is on screen for this turn.")
        elif a.option is not None:
            ch = next((c for c in last.get("_choices") or [] if c["index"] == a.option), None)
            if ch is None:
                sys.exit("No such choice on screen.")
            how, said = ch["how"], f"(chose: {ch['label']})"
        else:
            how, said = {"kind": "say", "text": a.say}, a.say
        with _app(d, st) as (agent, store):
            since = _next_id(it["sid"])
            t0 = time.time()
            try:
                _post(agent, it["sid"], how)
                view = view_of(agent, it["sid"], since, store, time.time() - t0)
            except Exception as e:  # noqa: BLE001
                view = {"disposition": "error", "error": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 1), "_choices": [], "_confirm": None}
            rec["replies"].append(said)
            rec["stages"].append(view)
            if view["disposition"] in ("asked", "waiting") and len(rec["replies"]) < MAX_REPLIES:
                pass  # still paused: one more reply is allowed
            else:
                it["paused"] = False
                it["turn"] += 1
                _run_item(agent, store, st, key)
        _save(d, st)
        out = {"result_of_your_reply": _screen(view)}
        if it["paused"] and it["turn"] + 1 == int(turn):
            # the same turn still waits: what can be pressed now (what is on screen is the result above)
            out["this_turn_still_waits_for_you"] = {"replies_left": MAX_REPLIES - len(rec["replies"]),
                                                    "how_to_reply": {"choices": [{"option": c["index"], "label": c["label"]} for c in view.get("_choices") or []], "go_ahead_button": bool(view.get("_confirm")), "or": "type a short message"}}
        elif it["paused"]:
            out["note"] = "A later turn of this conversation now waits for you: run `pending` to see it."
        else:
            out["note"] = "This conversation needs nothing more from you." if it["done"] else "This conversation continued without needing you."
        print(json.dumps(out, indent=1, default=str))


def cmd_export(a) -> None:
    d = RUN_ROOT / a.config
    st = _load(d)
    pub = lambda v: {k: x for k, x in v.items() if not k.startswith("_")}  # noqa: E731
    rows = []
    for key, turns in sorted(st["source"].items()):
        for i, t in enumerate(turns):
            rec = st["records"].get(f"{key}#{i + 1}")
            rows.append({"key": f"{key}#{i + 1}", "author_file": key.split("/")[0], "conversation_so_far": [{"user": turns[j]["text"], "strata_final": [pub(v) for v in (st["records"].get(f"{key}#{j + 1}") or {"stages": []})["stages"]][-1:]} for j in range(i)],
                         "user_message": t["text"], "author": {k: t.get(k) for k in ("intent", "expected", "if_asked", "also_fine", "complexity")},
                         "stage_1_what_strata_did": pub(rec["stages"][0]) if rec and rec["stages"] else {"disposition": "not_reached"},
                         "simulated_user_replies": (rec or {}).get("replies") or [], "after_each_reply": [pub(v) for v in ((rec or {}).get("stages") or [])[1:]]})
    out = d / "judge_packets.jsonl"
    out.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    print(f"{len(rows)} turns written to {out}")
    offers = []
    for key, turns in sorted(st["source"].items()):
        for i, t in enumerate(turns):
            rec = st["records"].get(f"{key}#{i + 1}") or {}
            sh = rec.get("_shadow")
            if sh is not None:
                offers.append({"key": f"{key}#{i + 1}", "conversation_so_far": [turns[j]["text"] for j in range(i)], "user_message": t["text"], "author": {k: t.get(k) for k in ("intent", "expected", "if_asked", "also_fine")},
                               "what_strata_showed_instead": pub(rec["stages"][0]).get("question_or_message_shown"), "model_offer": sh.get("offer"), "no_offer_because": None if sh.get("offer") else sh.get("why"), "seconds": sh.get("seconds")})
    (d / "shadow_offers.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in offers) + ("\n" if offers else ""))
    print(f"shadow proposer: {len(offers)} stopped turns, {sum(1 for o in offers if o['model_offer'])} offers written to {d / 'shadow_offers.jsonl'}")
    cmd_status(a)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="heldout_v10")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("config"); s.add_argument("--shadow-model"); s.add_argument("--shadow-variant"); s.add_argument("--source"); s.set_defaults(fn=cmd_start)
    for name, fn in (("advance", cmd_advance), ("status", cmd_status), ("export", cmd_export)):
        s = sub.add_parser(name); s.add_argument("config"); s.set_defaults(fn=fn)
    s = sub.add_parser("pending"); s.add_argument("config"); s.add_argument("--part"); s.add_argument("--count", action="store_true"); s.add_argument("--all", action="store_true"); s.set_defaults(fn=cmd_pending)
    s = sub.add_parser("reply"); s.add_argument("config"); s.add_argument("key"); g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--option", type=int); g.add_argument("--confirm", action="store_true"); g.add_argument("--say"); s.set_defaults(fn=cmd_reply)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
