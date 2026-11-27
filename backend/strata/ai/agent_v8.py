"""V8 conversation behaviour of the research agent: one question frame per notebook, edited by the messages that follow.

The frame lives in the notebook's state (`frame`), with every earlier frame on a stack (`frame_log`) so that "undo" goes
back exactly one change, and a question Strata asked (`pending`) so that the next message can answer it in words
("the gap") as well as with a button. Every reply that changes the frame says what changed (`change` on the message).

Routing order for one message (`_v8_route`, called before any of the older readers):

  an answer to the question Strata just asked
  undo / compare with the previous version / show the observations behind the answer / explain a trade
  an edit of the current question ("use 5 min", "actually do QQQ", "nah, I meant down at least 1%")
  turning the current question into an option strategy ("what if I bought calls instead?")
  a relative change of the current strategy ("exit an hour later") or a bare number ("make it 2")
  a new question (semantics.read), asked about instead of run when a slot it needs is missing
  anything else: the older readers, unchanged, whose strategies are then checked against the sentence
"""
from __future__ import annotations

import copy
import re

from strata.ai import answer as ANS
from strata.ai import converse as CV
from strata.ai import semantics as SM
from strata.ai import teach as TEACH
from strata.api import db

EDIT_LEAD = re.compile(r"^\s*(?:and|but|what about|how about|same (?:thing )?(?:for|with|but)|only|just|also|use|try|make it|change|switch|instead|actually|nah|no+|nope|sorry|i meant|i mean|ok so|okay so|so|now|then|without|remove|drop|add)\b", re.I)
EXPLAIN_TRADE = re.compile(r"\bwhy did (?:the |this |that |my )?(?:worst |biggest |largest |first |last |best )?(?:trade|loss|losing trade|one)\b.*\b(?:lose|lost|fail|go wrong|happen)\b|\bwhy did (?:the )?(?:biggest|worst|largest) loss happen\b|\bexplain (?:the |this |that )?(?:worst |biggest )?(?:trade|loss)\b|\bwhy (?:did|does) (?:it|this|the trade) lose\b", re.I)
WHICH_TRADE = re.compile(r"\b(worst|biggest|largest|best|first|last)\b", re.I)


def _is_new_question(r: SM.Reading, text: str) -> bool:
    kinds = {m.value["kind"] for m in r.mentions if m.kind == "kind"}
    if EDIT_LEAD.match(text) and len(text.split()) <= 9:
        return False
    if kinds & {"frequency", "agreement", "timing", "breakdown", "movement"}:
        return True
    has_event = any(m.kind == "event" for m in r.mentions)
    has_outcome = any(m.kind == "outcome" and m.value.get("key") in ("close_vs", "touch", "touch_level") for m in r.mentions)
    return has_event and has_outcome and len(text.split()) >= 8


_BARE_COMPARE_WORDS = set("compare comparing comparison both them the two these those versions version results result previous last before old original first one ones with to and side by "
                          "please can you could now ok okay so lets let's show me it that this a of vs versus against new latest current".split())


def bare_compare(text: str) -> bool:
    """'Compare both', 'compare them side by side', 'compare with the previous one': nothing but the request to compare.
    'Change the decline to 3.8% and compare the results' is a change that also asks for a comparison."""
    if not CV.COMPARE.search(text):
        return False
    words = re.findall(r"[a-z0-9.%']+", text.lower())
    return bool(words) and all(w.strip(".") in _BARE_COMPARE_WORDS for w in words)


def wants_compare(text: str) -> bool:
    return bool(re.search(r"\b(?:compare|comparing|comparison|side by side|versus|vs\.?)\b", text, re.I))


OLD_FILTER = {"miss": "loss", "losers": "loss", "wrong": "loss", "worst": "loss", "hit": "win", "winners": "win", "right": "win", "best": "win"}


def _explore_filter(text: str, kind: str | None) -> str:
    low = text.lower()
    if re.search(r"\bworst\b", low):
        return "worst"
    if re.search(r"\bbest\b", low):
        return "best"
    if re.search(r"\b(?:losing|lost|losers?|wrong|failed|fail|didn'?t|did not|never|missed|miss|red)\b", low):
        return "losers" if kind == "strategy" else ("wrong" if kind == "agreement" else "miss")
    if re.search(r"\b(?:winning|winners?|won|right|worked|happened|hit|successful|green)\b", low):
        return "winners" if kind == "strategy" else ("right" if kind == "agreement" else "hit")
    return "all"


class ResearchV8:
    # ----------------------------------------------------------------------------------------------------- routing
    def _v8_state(self) -> tuple[str | None, dict]:
        sid = getattr(self._tl, "sid", None)
        s = db.get_session(sid) if sid else None
        st = (s or {}).get("state") if isinstance((s or {}).get("state"), dict) else {}
        return sid, st or {}

    def _v8_route(self, text: str, current, known: list[str], default_symbol: str) -> dict | None:
        self._tl.v8_partial = self._tl.v8_unplaced = None  # what this layer read of the PREVIOUS message never carries over
        sid, st = self._v8_state()
        frame, pending = st.get("frame"), st.get("pending")
        tick = [k for k in known if k.isupper()] or list(SM.STORED_DEFAULT)
        # 0. the answer to a question that was just asked
        if pending:
            a = CV.answer_clarification(text, pending)
            if a is not None:
                return self._v8_apply_option(pending, a["option"], text)
        # 1. conversational actions on what exists
        if CV.UNDO.match(text):
            return {"action": "undo"}
        if (frame or current) and bare_compare(text):
            return {"action": "compare_versions"}
        topic = TEACH.topic_of(text)
        if topic and (frame or current or self._latest_result(sid) if sid else False):
            return {"action": "compare_versions"} if topic in ("difference", "outperform") else {"action": "teach", "topic": topic}
        if (frame or current) and EXPLAIN_TRADE.search(text):
            w = WHICH_TRADE.search(text)
            return {"action": "explain_trade", "which": (w.group(1).lower() if w else "worst")}
        if (frame or current) and CV.EXPLORE.search(text) and not re.search(r"could ?n[o']t|cannot|can'?t|not (?:be )?(?:tested|priced)|untested|missing|skipped|not taken|unexecuted|\bwhy\b|\bfills?\b", text.lower()):
            k = (frame or {}).get("kind") or ("strategy" if current is not None and current.kind == "options" else "frequency")
            f = _explore_filter(text, k)
            return {"action": "show_trades", "filter": OLD_FILTER.get(f, "all"), "explore_filter": f}
        # 2. the current question, edited or traded
        if frame and frame.get("kind") in CV.QUESTION_KINDS:
            if CV.CONVERT.search(text):
                conv = CV.to_strategy(frame, text)
                d = conv["draft"]
                nf = {"kind": "strategy", "symbol": d.symbol, "draft": d.model_dump(mode="json"), "text": text}
                return {"action": "new", "draft": d, "frame": nf, "edit": True, "changes": CV.summary(frame, nf), "assumptions": conv["assumptions"]}
            r_new = SM.read(text, known, frame.get("symbol") or default_symbol)
            if not _is_new_question(r_new, text):
                e = CV.edit_question(text, frame, tick, st.get("last_slot"))
                if e["action"] == "edit":
                    nf = e["frame"]
                    if nf.get("kind") == "movement" and nf.get("outcome", {}).get("key") in ("close_vs", "touch", "touch_level"):
                        nf["kind"] = "frequency"
                    route = SM.to_route(SM.Reading(text=nf.get("text") or text, frame=nf, mentions=[]))
                    if nf.get("kind") == "frequency" and (nf.get("outcome") or {}).get("key") == "event":
                        route = {"action": "profile", "frame": nf}  # counted on every session, as the question was the first time
                    route.update(frame=nf, edit=True, changes=e["changes"], slot=e["slot"])
                    if route.get("action") == "new" and current is not None and getattr(current, "kind", None) == "study" and route.get("draft") is not None:
                        # a study edited into another study is the next version of it: same lineage, a diff, and a comparison when asked
                        from strata.ai.draft import diff_drafts
                        route.update(action="modify", changed=[x["field"] for x in diff_drafts(current, route["draft"])], compare=wants_compare(text))
                    return route
                if e["action"] == "clarify":
                    return {"action": "clarify", "text": e["text"], "options": e.get("options") or [], "pending": {"kind": "frame_number", "options": e.get("options") or [], "frame": frame} if e.get("options") else None}
                if e["action"] == "noop":
                    return {"action": "noop", "text": e["text"]}
                if not (SM.tag(text, known)[1] and any(m_.kind in ("option", "action") for m_ in SM.tag(text, known)[1])) and r_new.frame.get("kind") is None:
                    # a short message that is not a new request and changes nothing this layer can place: the older readers try
                    # (a stop on a study, a cost setting, ...); if they cannot either, this is said instead of guessed
                    unread = e.get("unread") or []
                    self._tl.v8_unplaced = {"action": "clarify", "text": f"I took \"{text.strip()}\" as a change to the current question, but could not tell which part it changes, so nothing was changed. "
                                                        "Name the part, for example \"use the 15-minute candle\", \"do QQQ instead\" or \"only when yesterday was red\"." + (f" (Not understood: {', '.join(unread[:4])}.)" if unread else "")}
                    return None
        # 3. the current strategy: a bare number, or a time moved relative to itself
        if current is not None and current.kind == "options":
            m = CV.NUMBER_ONLY.match(text)
            if m:
                # the V7 correction reader first: it replaces the one value a unit can replace ("no, 2%" -> the 2% decline)
                from strata.ai import nl_rules as NL
                rc = NL.bare_correction(NL.correction(text) or text.strip(), current, list(dict.fromkeys(NL.TICKERS + tick)))
                if rc is not None and rc.get("action") == "modify":
                    return rc
                return self._v8_strategy_number(current, float(m.group("num")), text, (m.group("unit") or "").lower(), rc)
            sh = CV.relative_shift(text, current)
            if sh and sh[1]:
                field, new, old = sh
                d2 = current.model_copy(deep=True)
                setattr(d2, field, new)
                setattr(d2, field.replace("_time", "_time_stated"), None)
                if not (570 <= int(new[:2]) * 60 + int(new[3:]) <= 960):
                    return {"action": "clarify", "text": f"That would put the {'entry' if field == 'entry_time' else 'exit'} at {new} New York time, outside the session (09:30 to 16:00). "
                                                        f"Say the time you want, e.g. \"{'enter' if field == 'entry_time' else 'exit'} at 15:30 ET\"."}
                return {"action": "modify", "draft": d2, "changed": [field], "edit": True}
            if SM.tag(text, known)[1] and any(m_.kind == "kind" and m_.value["kind"] == "reliability" for m_ in SM.tag(text, known)[1]):
                return {"action": "robustness"}
            return None
        # 4. a new question
        r = SM.read(text, known, default_symbol)
        if current is not None and not frame and not _is_new_question(r, text) and r.frame.get("kind") not in ("strategy",):
            return None  # a change to a version this layer did not read: the older readers edit it
        if r.frame.get("kind") == "strategy":
            return None  # the strategy readers take it; agent._readers' result is checked in _v8_check_strategy
        k = r.frame.get("kind")
        if r.flag:
            return {"action": "clarify", "text": r.flag, "declined": True}
        if k in CV.QUESTION_KINDS and (r.unread or r.unclaimed) and r.flag is None:
            # words were left unread: the older readers (research graphs, the strategy readers) may know them. Their reading is
            # then held against what this layer did recognise (_v8_check_reading); if they know nothing either, it is said.
            self._tl.v8_partial = r
            return None
        if k == "agreement" and r.complete and r.frame.get("predictor") == "first_candle" and r.frame.get("scope") == "rest" and r.frame.get("candle_stated") and \
                not re.match(r"(?:so\s+|and\s+|ok(?:ay)?,?\s+)?(?:how (?:often|frequently|reliabl\w+|accurate\w*|many times|much of the time)|what (?:percent\w*|share|fraction|proportion)|in what)\b", text.lower().strip()):
            return None
        if k in CV.QUESTION_KINDS:
            if r.clarify:
                return {"action": "clarify", "text": r.clarify["text"], "options": r.clarify.get("options") or [],
                        "pending": {"kind": "frame_patch", "options": r.clarify.get("options") or [], "frame": r.frame}} if r.clarify.get("options") else \
                       {"action": "clarify", "text": r.clarify["text"], "unread": [m.text for m in r.unclaimed]}
            if r.frame.get("kind") == "frequency" and (r.frame.get("outcome") or {}).get("key") == "event":
                return {"action": "profile", "frame": r.frame}
            route = SM.to_route(r)
            route["frame"] = r.frame
            return route
        if k in ("explain", "reliability") and current is None and not frame:
            return {"action": "clarify", "text": "There is no result in this conversation yet. Ask a question or describe a strategy first, for example \"Buy SPY calls at the open after a 1% down day\", and then ask about it."}
        if k == "explore" and current is None and not frame:
            return {"action": "clarify", "text": "There is nothing to show yet: no question has been answered in this conversation."}
        return None

    def _v8_check_reading(self, text: str, r: dict, known: list[str]) -> dict:
        """Any reading from the older readers, held against the pieces this layer recognised in the sentence.

        A condition the sentence states that the reading does not hold, or a word this layer could not read that the
        reading does not account for either, is listed as not represented: the run then waits for a go-ahead and says
        what it left out, instead of answering a different question. When the older readers found nothing at all, the
        partial reading of this layer is said back, with the words it could not place."""
        partial = getattr(self._tl, "v8_partial", None)
        self._tl.v8_partial = None
        unplaced = getattr(self._tl, "v8_unplaced", None)
        self._tl.v8_unplaced = None
        r = self._v8_check_strategy(text, r, known)
        act = r.get("action")
        if act in ("unknown", None) and unplaced is not None:
            return unplaced
        if act in ("unknown", None) and partial is not None:
            if partial.clarify and partial.clarify.get("slot") != "unclaimed":
                return {"action": "clarify", "text": partial.clarify["text"], "unread": [u["text"] for u in partial.unread]}
            unread = [u["text"] for u in partial.unread] + [m.text for m in partial.unclaimed]
            return {"action": "clarify", "text": (f"I understood part of this ({SM.describe(partial.frame).rstrip('.')}) but not "
                                                  f"{', '.join(repr(u) for u in unread[:4])}, so nothing was run. Rephrase that part, or leave it out if it does not matter."),
                    "unread": unread}
        if act not in ("new", "graph", "hypothesis", "investigate", "discover") or r.get("frame"):
            return r
        serial = _serial(r)
        n, ms = SM.tag(text, known)
        missing = []
        for m in ms:
            if m.kind != "event":
                continue
            k = m.value.get("key")
            if k == "unsupported_event" or not _holds(serial, k):
                missing.append(m.text)
        if partial is not None:
            held = {m.text for m in ms}
            for u in partial.unread:
                w = u["text"]
                if w.lower() in ("close", "higher", "lower") or any(w in h for h in held):
                    continue
                if not _accounts(serial, w):
                    missing.append(w)
        missing = list(dict.fromkeys(missing))
        if not missing:
            return r
        if act == "new" and r.get("draft") is not None:
            d = r["draft"].model_copy(deep=True)
            d.unsupported = list(dict.fromkeys(list(d.unsupported) + missing))
            return {**r, "draft": d}
        if act == "graph" and r.get("graph"):
            g = dict(r["graph"])
            g["unsupported"] = list(dict.fromkeys(list(g.get("unsupported") or []) + missing))
            return {**r, "graph": g}
        return {"action": "clarify", "text": f"I could not place {', '.join(repr(x) for x in missing[:3])} in a reading of this, so nothing was run. Rephrase that part, or leave it out if it does not matter.",
                "unread": missing}

    def _v8_check_strategy(self, text: str, r: dict, known: list[str]) -> dict:
        """A strategy from the older readers, held against the sentence: an entry the sentence placed at the open that the
        reading left open is set (and said); anything else the sentence said that the draft does not hold is listed as not
        represented, and the reading waits for a go-ahead instead of running as if it had been read."""
        d = r.get("draft")
        if d is None or getattr(d, "kind", None) != "options" or r.get("action") not in ("new",):
            return r
        n, ms = SM.tag(text, known)
        d = d.model_copy(deep=True)
        notes = []
        if d.entry_time is None and not d.sequence and not any(c.expr for c in d.conditions if re.search(r"from_open|vwap|crosses|price", c.expr)):
            if any(m.kind == "anchor" and m.value.get("anchor") == "open" for m in ms):
                d.entry_time, d.entry_time_stated = "open", next(m.text for m in ms if m.kind == "anchor" and m.value.get("anchor") == "open")
                notes.append(f"Entry at the open (you said \"{d.entry_time_stated}\").")
        notes += _fill_exits(text, d, n, ms)
        notes += _fix_sizes(d, ms)
        missing = SM.strategy_coverage(text, d, known)
        if missing:
            d.unsupported = list(dict.fromkeys(list(d.unsupported) + [f"{x}" for x in missing]))
        elif d.unsupported and notes:
            # what the older reader could not place and this layer has now placed (an exit, a target) is no longer missing
            filled = r"\b(?:lunch|noon|midday|til close|till close|until close|tp|sl|take ?profit|stop ?loss|out at|sell at|sold at|exit at|close at|every morning|in the morning|at the open|at the bell)\b"
            d.unsupported = [u for u in d.unsupported if not re.search(filled, u.lower())]
        return {**r, "draft": d, **({"assumptions": notes} if notes else {})}

    def _v8_apply_option(self, pending: dict, option: dict, text: str) -> dict:
        patch = option.get("patch") or {}
        fr = copy.deepcopy(pending.get("frame") or {})
        if pending.get("kind") == "frame_number":
            res = CV._apply_number(fr, (patch["slot"], option.get("label"), patch.get("index")), float(patch["value"]), pending.get("frame") or {})
            nf = res["frame"]
            route = SM.to_route(SM.Reading(text=nf.get("text") or text, frame=nf, mentions=[]))
            route.update(frame=nf, edit=True, changes=res["changes"], slot=res["slot"], answered=True)
            return route
        if pending.get("kind") == "strategy_number":
            from strata.ai.draft import Draft
            d = Draft.model_validate(pending["draft"])
            d2 = _set_strategy_number(d, patch)
            return {"action": "modify", "draft": d2, "changed": [patch.get("field") or "conditions"], "edit": True, "answered": True}
        fr.update({k: v for k, v in patch.items()})
        if fr.get("kind") == "agreement" and patch.get("predictor"):
            fr.setdefault("scope", "both")
        r = SM.Reading(text=fr.get("text") or text, frame=fr, mentions=[])
        if fr.get("kind") == "frequency" and (fr.get("outcome") or {}).get("key") == "event":
            return {"action": "profile", "frame": fr, "answered": True}
        route = SM.to_route(r)
        route.update(frame=fr, answered=True)
        return route

    def _v8_strategy_number(self, d, v: float, text: str, unit: str = "", older: dict | None = None) -> dict:
        slots = strategy_number_slots(d)
        if unit in ("%", "percent", "pct"):
            slots = [x for x in slots if x["patch"].get("field") != "quantity"]
        elif unit:
            slots = []
        if len(slots) == 1:
            return {"action": "modify", "draft": _set_strategy_number(d, {**slots[0]["patch"], "value": v}), "changed": [slots[0]["patch"].get("field")], "edit": True}
        if not slots:
            return {"action": "clarify", "text": f"\"{text.strip()}\" is a number, but the current strategy has no number it could replace."}
        opts = [{"label": s["label"], "patch": {**s["patch"], "value": v}} for s in slots]
        lead = (older or {}).get("text") if (older or {}).get("action") == "clarify" else None
        return {"action": "clarify", "text": (lead + " " if lead else "") + f"Which should be {v:g}{'%' if unit in ('%', 'percent', 'pct') else ''}?", "options": opts,
                "pending": {"kind": "strategy_number", "options": opts, "draft": d.model_dump(mode="json")}}

    # ------------------------------------------------------------------------------------------------ new actions
    def _profile(self, sid: str, frame: dict, interp: str) -> list[dict]:
        """Questions answered directly from stored bars: when during the day, by weekday or month, how often an event happens."""
        from strata.research import profile as PR
        f = frame
        try:
            if f.get("kind") == "timing":
                res = PR.timing(self.store, f["symbol"], f.get("measure") or "abs_move", int(f.get("bucket") or 30), f.get("period"), f.get("conditions"))
                ans = ANS.timing(res, f)
                compact = {k: v for k, v in res.items() if k != "per_session"}
            elif f.get("kind") == "breakdown":
                res = PR.breakdown(self.store, f["symbol"], f.get("by") or "weekday", f.get("measure") or "open_to_close", f.get("period"), f.get("conditions"))
                ans = ANS.breakdown(res, f)
                compact = {**res, "groups": [{k: v for k, v in g.items() if k != "dates"} for g in res["groups"]]}
            else:
                ev = (f.get("outcome") or {}).get("event") or {}
                res = PR.event_frequency(self.store, f["symbol"], ev, f.get("period"), f.get("conditions"))
                ans = ANS.event_frequency(res, f)
                compact = {k: v for k, v in res.items() if k != "dates"}
        except (LookupError, ValueError) as e:
            return [self._say(sid, str(e), interpreter=interp)]
        self._trials(sid, 1)
        return [db.add_message(sid, "assistant", "result", {"view": "profile", "frame": {k: v for k, v in f.items() if k != "text"}, "result": compact, "answer": ans, "text": ans["headline"],
                                                            "explanation_source": "deterministic", "interpreter": interp})]

    def _latest_result(self, sid: str) -> dict | None:
        for m in reversed(db.list_messages(sid)):
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] == "result" and c.get("view") in ("single", "investigation", "profile", "compare", "sweep"):
                return m
        return None

    def _explore(self, sid: str, version, r: dict) -> list[dict]:
        """The observations behind the latest answer: the days of a study, the trades of a backtest, the classified days of an
        agreement question, or the sessions of a profile. The message is the "trades" message the workspace already knows
        (run, mode, filter), with an `explore` spec that says exactly which observations and why."""
        f = r.get("explore_filter") or {"loss": "miss", "win": "hit"}.get(r.get("filter") or "all", r.get("filter") or "all")
        m = self._latest_result(sid)
        c = (m or {}).get("content") or {}
        spec, text, run_id, mode = None, None, None, None
        if c.get("view") == "investigation":
            rows = ((c.get("report") or {}).get("agreement") or [])
            rest = [x for x in rows if x["scope"] == "rest"] or rows
            if rest:
                x = sorted(rest, key=lambda z: z["minutes"])[0]
                ff = {"miss": "wrong", "hit": "right", "losers": "wrong", "winners": "right"}.get(f, f)
                spec = {"kind": "classification", "runs": [x["up"]["run_id"], x["down"]["run_id"]], "filter": ff, "minutes": x["minutes"], "scope": x["scope"]}
                n = x["sessions"] - x["agree"] if ff == "wrong" else x["agree"] if ff == "right" else x["sessions"]
                text = {"wrong": f"The {n:,} days the first {x['minutes']}-minute candle and the {'rest of the session' if x['scope'] == 'rest' else 'whole session'} went different ways.",
                        "right": f"The {n:,} days they went the same way."}.get(ff, f"All {n:,} days, each with the candle's direction and the session's.")
                mode = "study"
        elif c.get("view") == "profile":
            spec = {"kind": "profile", "filter": f, "frame": c.get("frame")}
            text, mode = "The sessions behind this answer.", "study"
        if spec is None:
            rid = c.get("run_id") if c.get("view") == "single" else None
            run = db.get_run(rid) if rid else None
            if run is None:
                done = self._done_runs(version) if version else []
                if not done:
                    return [self._say(sid, "No finished run for the current question yet - run it first.")]
                run = db.get_run(done[-1]["run_id"]) or {"id": done[-1]["run_id"], "kind": done[-1]["run"]["kind"]}
            run_id = run["id"]
            sm = run.get("summary") or {}
            if run.get("kind") == "study":
                ff = {"worst": "worst", "best": "best", "miss": "miss", "hit": "hit", "losers": "miss", "winners": "hit", "wrong": "miss", "right": "hit"}.get(f, "all")
                spec, mode = {"kind": "sessions", "run_id": run_id, "filter": ff}, "study"
                q, h = sm.get("qualifying_sessions"), sm.get("target_first")
                cnt = {"miss": (q - h) if q is not None and h is not None else None, "hit": h}.get(ff, q)
                lead = f"The {cnt:,} " if isinstance(cnt, int) else "The "
                text = {"miss": f"{lead}days it did not happen.", "hit": f"{lead}days it happened.", "worst": "The days that ended lowest.", "best": "The days that ended highest."}.get(ff, f"{lead}days that matched your rules.")
            else:
                ff = {"miss": "losers", "hit": "winners", "worst": "losers", "wrong": "losers", "right": "winners", "best": "winners"}.get(f, f)
                spec, mode = {"kind": "trades", "run_id": run_id, "filter": ff}, (run.get("payload") or {}).get("mode") or sm.get("mode")
                text = {"losers": "The trades that lost money.", "winners": "The trades that made money."}.get(ff, "Every trade.")
            text += " Click one to replay that day."
        return [db.add_message(sid, "assistant", "trades", {"text": text, "explore": spec, "run_id": run_id or (spec.get("runs") or [None])[0], "mode": mode,
                                                             "filter": OLD_FILTER.get(spec.get("filter"), "all"), "source_message": (m or {}).get("id")})]

    def _teach(self, sid: str, topic: str, interp: str) -> list[dict]:
        msgs = db.list_messages(sid)
        fn = {"new_to_options": TEACH.new_to_options, "why_lost_value": TEACH.why_lost_value, "chart": TEACH.chart, "expiration": TEACH.expiration, "how_much": TEACH.how_much,
              "difference_luck": TEACH.difference_luck}.get(topic)
        a = TEACH.unreliable(msgs, max(1, self._trials(sid))) if topic == "unreliable" else fn(msgs) if fn else TEACH.chart(msgs)
        _, version, _ = self._state(sid)
        if version and any((db.get_run(r_["run_id"]) or {}).get("status") in ("queued", "running") for r_ in version.get("runs") or []):
            a = {**a, "headline": "The newest version is still running, so this is about the previous result. " + (a.get("headline") or "")}
        a = {"kind": "explain", "topic": topic, "numbers": [], "followups": [], "limits": [], **a}
        text = " ".join([a.get("headline") or ""] + list(a.get("happened") or []))
        return [db.add_message(sid, "assistant", "result", {"view": "explain", "text": text, "answer": a, "explanation_source": "deterministic", "interpreter": interp,
                                                            **({"run_id": a["trade"]["run_id"], "trade_id": a["trade"]["trade_id"]} if a.get("trade") else {})})]

    def _explain_trade(self, sid: str, version, which: str) -> list[dict]:
        from strata.api.results import load_trades
        m = next((x for x in reversed(db.list_messages(sid)) if x["kind"] == "result" and (x["content"] or {}).get("view") in ("single", "compare") and (x["content"] or {}).get("run_id")), None)
        run = db.get_run(m["content"]["run_id"]) if m else None
        if not run or run.get("kind") == "study":
            return [self._say(sid, "There is no finished option backtest in this conversation, so there is no trade to explain. Describe a strategy first, e.g. \"Buy SPY 0DTE calls at the open after a 1% down day\".")]
        trades = load_trades(run["id"])
        if not trades:
            return [self._say(sid, "The backtest made no trades, so there is none to explain.")]
        pick = {"worst": min, "biggest": min, "largest": min, "best": max}.get(which, min)(trades, key=lambda t: t["pnl"]) if which not in ("first", "last") else \
            sorted(trades, key=lambda t: t["entry_ts"])[0 if which == "first" else -1]
        story = ANS.why_lost(pick, run["id"]) if pick["pnl"] < 0 else ANS.trade_story(pick, run["id"])
        amount = f"${abs(pick['pnl']):,.2f}"
        what = {"worst": "biggest loss", "biggest": "biggest loss", "largest": "biggest loss", "best": "best trade"}.get(which, f"{which} trade")
        head = f"The {what}, on {ANS._date(pick['entry_date'])}, {'lost' if pick['pnl'] < 0 else 'made'} {amount} after fees."
        if version and any((db.get_run(r_["run_id"]) or {}).get("status") in ("queued", "running") for r_ in version.get("runs") or []):
            head = "The newest version is still running, so this is about the previous result. " + head
        return [db.add_message(sid, "assistant", "result", {"view": "explain_trade", "run_id": run["id"], "trade_id": pick["id"], "text": head + " " + " ".join(story.get("cause") or []),
                                                            "answer": {"kind": "explain_trade", "headline": head, "story": story, "happened": story.get("cause") or [], "numbers": [], "limits": [],
                                                                       "followups": [{"text": "Show me the losing trades", "why": "Every trade that lost money."}, {"text": "Is this actually good, or just luck?", "why": "What the evidence can say."}]},
                                                            "explanation_source": "deterministic"})]

    def _undo(self, sid: str, interp: str) -> list[dict]:
        s, _, _ = self._state(sid)
        st = s["state"]
        log = list(st.get("frame_log") or [])
        if not log:
            return [self._say(sid, "There is nothing to undo in this conversation.")]
        last = log.pop()
        prev = last.get("frame")
        cur = st.get("frame")
        lines = CV.summary(cur, prev)
        db.update_session(sid, state={**st, "frame_log": log, "frame": prev, "pending": None, "undoing": True})
        if not prev:
            return [self._say(sid, "Went back to before the first question. Ask a new one.")]
        out = self._v8_exec(sid, prev, interp)
        txt = "Undid the last change. " + CV.summary_text(lines)
        if out:
            c = out[0]["content"]
            c["change"] = {"lines": lines, "text": txt, "undo": bool(log), "undone": True}
            db.update_message(out[0]["id"], c)
        s2, _, _ = self._state(sid)
        db.update_session(sid, state={**s2["state"], "frame": prev, "frame_log": log, "undoing": False})
        return out

    def _v8_exec(self, sid: str, frame: dict, interp: str) -> list[dict]:
        from strata.ai.draft import Draft
        if frame.get("kind") == "strategy" and frame.get("draft"):
            return self._propose(sid, Draft.model_validate(frame["draft"]), None, False, interp, None)
        if frame.get("kind") == "frequency" and (frame.get("outcome") or {}).get("key") == "event" or frame.get("kind") in ("timing", "breakdown"):
            return self._profile(sid, frame, interp)
        route = SM.to_route(SM.Reading(text=frame.get("text") or "", frame=frame, mentions=[]))
        if route["action"] == "new":
            return self._propose(sid, route["draft"], None, False, interp, None)
        if route["action"] == "investigate":
            return self._investigate(sid, route["plan"], interp, None)
        return []

    def _compare_versions(self, sid: str, version, interp: str) -> list[dict]:
        if version is None or not version.get("parent_id"):
            return [self._say(sid, "There is only one version so far, so there is nothing to compare it with. Change something first, e.g. \"exit an hour later\", then ask to compare.")]
        parent = db.get_version(version["parent_id"])
        cur_runs = [r for r in version.get("runs") or [] if (db.get_run(r["run_id"]) or {}).get("status") == "done"]
        par_runs = [r for r in (parent or {}).get("runs") or [] if (db.get_run(r["run_id"]) or {}).get("status") == "done"]
        if not cur_runs or not par_runs:
            return [self._say(sid, "One of the two versions has not finished running yet; ask again when both results are in.")]
        a, b = par_runs[-1], cur_runs[-1]
        ph = ANS.draft_phrases(parent.get("draft"), version.get("draft")) if isinstance(parent.get("draft"), dict) else None
        text = (f"Comparing the version with {ph[1]} against the one with {ph[0]}." if ph else f"Comparing v{parent['id']} ({parent['label']}) with v{version['id']} ({version['label']}).")
        return [db.add_message(sid, "assistant", "run", {"text": text, "runs": [{**b, "label": f"v{version['id']} - {version['label']}"}],
                                                         "compare_with": [{**a, "label": f"v{parent['id']} - {parent['label']}"}], "version_id": version["id"], "finalized": False, "mode": b.get("mode")})]

    # --------------------------------------------------------------------------------------------- after each turn
    def _v8_after(self, sid: str, r: dict, out: list[dict], prev_state: dict, text: str) -> None:
        """Record the frame this turn produced, what changed, and any question it asked."""
        try:
            s, v, cur = self._state(sid)
            st = s["state"]
            act = r.get("action")
            first = next((m for m in out if m.get("role") == "assistant"), None)
            upd: dict = {}
            if act == "clarify":
                upd["pending"] = r.get("pending")
                if first is not None and r.get("options"):
                    c = first["content"]
                    c["options"] = [{"label": o["label"]} for o in r["options"]]
                    c["clarify"] = True
                    db.update_message(first["id"], c)
                db.update_session(sid, state={**st, **upd})
                return
            if act in ("undo", "explore", "show_trades", "compare_versions", "explain_trade", "robustness", "teach") or st.get("undoing"):
                if st.get("pending"):
                    db.update_session(sid, state={**st, "pending": None})
                return
            new = r.get("frame")
            if new is None and act in ("new", "modify", "derive") and cur is not None:
                new = {"kind": "strategy" if cur.kind == "options" else "study", "symbol": cur.symbol, "draft": cur.model_dump(mode="json"), "text": text}
            if new is None:
                if st.get("pending"):
                    db.update_session(sid, state={**st, "pending": None})
                return
            if new.get("kind") == "strategy" and cur is not None and cur.kind == "options":
                new = {**new, "draft": cur.model_dump(mode="json"), "symbol": cur.symbol}
            prev = prev_state.get("frame")
            log = list(prev_state.get("frame_log") or [])
            if prev:
                log.append({"frame": prev, "version": prev_state.get("current")})
            upd.update(frame=new, frame_log=log[-30:], pending=None, last_slot=r.get("slot"))
            if v is not None:
                frames = dict(st.get("frames") or {})
                frames[str(v["id"])] = new
                upd["frames"] = dict(list(frames.items())[-60:])
            db.update_session(sid, state={**st, **upd})
            if first is not None:
                c = first["content"]
                c["frame"] = {k: x for k, x in new.items() if k != "draft"}
                c["reading"] = SM.describe(new) if new.get("kind") in CV.QUESTION_KINDS else None
                if prev and (r.get("edit") or act == "modify" or r.get("answered")):
                    lines = list(r.get("changes") or []) or CV.summary(prev, new)
                    c["change"] = {"lines": lines, "text": CV.summary_text(lines), "undo": True}
                if r.get("assumptions"):
                    c["assumed"] = r["assumptions"]
                db.update_message(first["id"], c)
        except Exception:  # noqa: BLE001  (the record of a turn must never be the reason the turn fails)
            return

    # -------------------------------------------------------------------------------------------------- answers
    def _v8_answer(self, m_kind: str, c: dict, content: dict, entries: list[dict]) -> None:
        """Attach the answer (answer.py) to a result about to be stored."""
        try:
            frame = c.get("frame") or {}
            view = content.get("view")
            if view == "single" and entries:
                e = entries[-1]
                if e["run"]["kind"] == "study":
                    content["answer"] = (ANS.movement if frame.get("kind") == "movement" else ANS.frequency)(e["result"], frame or None, e["run"]["id"])
                else:
                    from strata.api.results import load_trades
                    from strata.research import performance as PF
                    trades = load_trades(e["run"]["id"])
                    card = PF.card(e["run"], e["result"], trades, max(1, self._trials_of(c)))
                    content["answer"] = ANS.strategy(e["run"], e["result"], trades, card, {**frame, **({"dte": (frame.get("draft") or {}).get("dte")})})
            elif view == "compare" and len(entries) >= 2:
                content["answer"] = ANS.compare(entries, self._run_descriptions())
            elif view == "sweep" and len(entries) >= 2:
                content["answer"] = ANS.sweep(entries, content.get("field"))
            elif view == "investigation" and content.get("report"):
                rep = content["report"]
                if rep.get("agreement") is not None or (c.get("plan") or {}).get("kind") in ("first_candle", "agreement"):
                    content["answer"] = ANS.agreement(rep, frame or (c.get("plan") or {}).get("frame"))
                else:
                    content["answer"] = frequency_set(rep, frame or (c.get("plan") or {}).get("frame") or {})
        except Exception as ex:  # noqa: BLE001
            content["answer_error"] = f"{type(ex).__name__}: {ex}"[:300]

    def _run_descriptions(self) -> dict:
        """run id -> what that version asked, in words (its question frame, else its draft's title)."""
        sid = getattr(self._tl, "sid", None)
        if not sid:
            return {}
        s = db.get_session(sid) or {}
        frames = (s.get("state") or {}).get("frames") or {} if isinstance(s.get("state"), dict) else {}
        out = {}
        for v in db.list_versions(sid):
            fr = frames.get(str(v["id"]))
            d = SM.describe(fr) if fr and fr.get("kind") in CV.QUESTION_KINDS else ((v.get("draft") or {}).get("title") if isinstance(v.get("draft"), dict) else None) or v.get("label")
            for rr in v.get("runs") or []:
                rid = rr.get("run_id") if isinstance(rr, dict) else rr
                if rid:
                    out[rid] = {"desc": d, "draft": v.get("draft") if isinstance(v.get("draft"), dict) else None}
        return out

    def _trials_of(self, c: dict) -> int:
        sid = getattr(self._tl, "sid", None)
        try:
            return self._trials(sid) if sid else 1
        except Exception:  # noqa: BLE001
            return 1


def _serial(r: dict) -> str:
    """One reading of the older readers as lower-case text: what it holds, in its own words and expressions."""
    import json
    parts = []
    for k in ("draft", "graph", "plan", "spec"):
        v = r.get(k)
        if v is None:
            continue
        if hasattr(v, "model_dump"):
            v = v.model_dump(mode="json")
        try:
            parts.append(json.dumps(v, default=str))
        except Exception:  # noqa: BLE001
            parts.append(str(v))
    return " ".join(parts).lower()


def _holds(serial: str, key: str | None) -> bool:
    return SM.holds(serial, key)


def _accounts(serial: str, word: str) -> bool:
    """A word the semantic layer could not read is accounted for when the reading uses it (or its stem) somewhere."""
    w = word.lower().strip(".,")
    if re.fullmatch(r"[\d.%]+", w):
        return w.rstrip("%") in serial
    stem = re.sub(r"(?:ling|ing|ed|es|s|ly|ity|ities|ation|ations)$", "", w)
    return w in serial or (len(stem) >= 4 and stem in serial)


def frequency_set(rep: dict, frame: dict) -> dict:
    """Several variants of one frequency question (sizes, candle lengths, symbols), answered as a short table."""
    rows = rep.get("rows") or []
    if not rows:
        return {"kind": "frequency_set", "headline": rep.get("verdict") or "", "numbers": [], "followups": []}
    lines = [f"{r['label']}: {ANS._pct((r['probability'] or 0) * 100)} of {ANS._n(r['n'])} days (any day: {ANS._pct((r['baseline'] or 0) * 100)})" for r in rows if r.get("probability") is not None]
    head = "; ".join(lines[:4]) + "."
    sig = [r for r in rows if r.get("adjusted_p_value") is not None and r["adjusted_p_value"] < 0.05 and r["n"] >= 30]
    happened = [f"{len(sig)} of the {len(rows)} variants differ from an ordinary day by more than chance once the {len(rows)} comparisons are counted." if sig else
                f"None of the {len(rows)} variants differs from an ordinary day by more than chance once the {len(rows)} comparisons are counted."]
    return {"kind": "frequency_set", "headline": head, "happened": happened, "limits": rep.get("limitations") or [],
            "numbers": [{"label": r["label"], "value": ANS._pct((r["probability"] or 0) * 100), "detail": f"{ANS._n(r['n'])} days", "explore": {"kind": "sessions", "run_id": r["run_id"], "filter": "all"}} for r in rows[:6]],
            "viz": {"kind": "variants", "rows": [{"label": r["label"], "rate": r["probability"], "ci": r["ci95"], "baseline": r["baseline"], "n": r["n"], "run_id": r["run_id"]} for r in rows]},
            "followups": ANS.followups(frame, "frequency")}


def _fill_exits(text: str, d, n, ms) -> list[str]:
    """Common ways of saying an exit that the older strategy reader can miss, placed in the draft (and said):
    "out at 9am pt", "sold at lunch", "hold til close", "30% tp / 20% sl", "take profit at 25%". Only fills what is empty."""
    low = n.norm
    notes = []
    if d.exit_time is None and not d.hold_minutes and not d.hold_sessions:
        clock = next((m for m in ms if m.kind == "clock" and re.search(r"\b(?:out|sell|sold|exit|close|closing|get out|buy (?:them|it) back|cover)\b(?: (?:it|them|the position))?(?: (?:at|by|around))?\s*$",
                                                                        low[max(0, m.a - 30):m.a])), None)
        if clock is not None and clock.value.get("et") and clock.value["et"] != (d.entry_time or ""):
            d.exit_time, d.exit_time_stated = clock.value["et"], clock.value.get("stated")
            notes.append(f"Exit at {clock.value['et']} New York time (you said \"{clock.text}\").")
        elif re.search(r"\b(?:sell|sold|out|exit|close)\b(?: (?:it|them))? (?:at|by|around) (?:lunch|noon|midday)\b", low):
            d.exit_time, d.exit_time_stated = "12:00", "lunch"
            notes.append("\"At lunch\" is read as an exit at 12:00 New York time.")
        elif re.search(r"\bhold (?:it |them )?(?:til|till|until|to|through|into) (?:the )?(?:close|end of (?:the )?day|endofday)\b|\b(?:sell|exit|out) (?:at|by) (?:the )?close\b", low):
            d.exit_time, d.exit_time_stated = "close", "the close"
            notes.append("Held until the close.")
    if d.premium_target_pct is None:
        mt = re.search(r"(?<![\d.])(\d+(?:\.\d+)?) ?% ?(?:tp|takeprofit|take profit|profit target|target on the (?:option|premium))\b|\b(?:tp|takeprofit|take profit|profit target)(?: (?:at|of|=))? (\d+(?:\.\d+)?) ?%", low)
        if mt:
            d.premium_target_pct = float(mt.group(1) or mt.group(2))
            notes.append(f"Take profit when the option gains {d.premium_target_pct:g}%.")
    if d.premium_stop_pct is None:
        ms_ = re.search(r"(?<![\d.])(\d+(?:\.\d+)?) ?% ?(?:sl|stoploss|stop(?: loss)?|stop on the (?:option|premium))\b|\b(?:sl|stoploss|stop(?: loss)?)(?: (?:at|of|=))? (\d+(?:\.\d+)?) ?%", low)
        if ms_:
            d.premium_stop_pct = float(ms_.group(1) or ms_.group(2))
            notes.append(f"Stop when the option loses {d.premium_stop_pct:g}%.")
    return notes


def _fix_sizes(d, ms) -> list[str]:
    """A condition the older reader kept but sized wrongly ("gap ups >1%" read as any gap up): the size the sentence states
    replaces it, when there is exactly one such condition and one such event (and it is said)."""
    from strata.ai.draft import DraftCondition
    notes = []
    series = {"gap": r"^gap_pct\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)$", "prev_return": r"^previous_session\.return_pct\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)$"}
    for key, pat in series.items():
        evs = [m for m in ms if m.kind == "event" and m.value.get("key") == key and m.value.get("value")]
        idx = [i for i, c in enumerate(d.conditions) if re.match(pat, c.expr.strip())]
        if len(evs) != 1 or len(idx) != 1:
            continue
        ev, i = evs[0], idx[0]
        mm = re.match(pat, d.conditions[i].expr.strip())
        v = float(ev.value["value"])
        if abs(float(mm.group(2)) - v) > 1e-9 and (float(mm.group(2)) == 0 or abs(abs(float(mm.group(2))) - abs(v)) > 1e-9):
            new = SM.cond_expr({"key": key, "op": ev.value.get("op") or (">=" if v > 0 else "<="), "value": v})
            if new:
                d.conditions[i] = DraftCondition(expr=new[0], text=new[1], source="stated")
                notes.append(f"\"{ev.text}\" is read as: {new[1]}.")
    return notes


def strategy_number_slots(d) -> list[dict]:
    out = []
    for i, c in enumerate(d.conditions):
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", c.expr)
        if m and float(m.group(1)) != 0:
            out.append({"label": f"{c.text or c.expr} (now {abs(float(m.group(1))):g})", "patch": {"field": "conditions", "index": i}})
    for k, name in (("premium_target_pct", "the profit target on the option"), ("premium_stop_pct", "the stop on the option"), ("underlying_target_pct", "the target on the underlying"),
                    ("underlying_stop_pct", "the stop on the underlying")):
        if getattr(d, k, None):
            out.append({"label": f"{name} (now {getattr(d, k):g}%)", "patch": {"field": k}})
    out.append({"label": f"the number of contracts (now {d.quantity})", "patch": {"field": "quantity"}})
    return out


def _set_strategy_number(d, patch: dict):
    d2 = d.model_copy(deep=True)
    f, v = patch.get("field"), float(patch.get("value"))
    if f == "conditions":
        c = d2.conditions[patch["index"]]
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", c.expr)
        neg = m and float(m.group(1)) < 0
        newv = -abs(v) if neg else abs(v)
        c.expr = c.expr[:m.start(1)] + f"{newv:g}"
        c.text = re.sub(r"\d+(?:\.\d+)?", f"{abs(v):g}", c.text, count=1) if c.text else c.text
    elif f == "quantity":
        d2.quantity = max(1, int(v))
    else:
        setattr(d2, f, abs(v))
    return d2
