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


def _names_a_kind(text: str) -> bool:
    """Does a follow-up ask for a different kind of answer ("how often", "when during the day", "by weekday", "predict")?
    A conversational lead-in ("what happens if", "what about") does not."""
    low = text.lower()
    return bool(re.search(r"\bhow often\b|\bodds\b|\bchance\b|\bwhat (?:%|percent)|\bwhen during\b|\bwhat time\b|\bwhich (?:half ?hour|hour|weekday|day of the week|month)\b|"
                          r"\bby (?:weekday|month)\b|\bpredict|\btypical (?:path|move)\b|\bhow (?:far|much) does\b|\baverage move\b", low))


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
# V9, for the built-in path (the model reader names these itself)
NEXT_STEPS = re.compile(r"\bwhat (?:would|should|do|could) (?:you|i|we) (?:investigate|look at|look into|test|try|check|do|explore) next\b|\bwhat(?:'s| is)? next\b|\bnext steps?\b|\bwhere (?:would|should|do) (?:you|i|we) (?:look|go) next\b|\bwhat else should i (?:test|check|look at)\b", re.I)
ABOUT_SELECTION = re.compile(r"\b(?:anything|something) (?:interesting|unusual|special|different|notable|odd)\b.*\b(?:trades?|days?|sessions?|losers?|winners?|those|these|them)\b|\b(?:anything|something|what) (?:do )?(?:those|these|the|they) (?:days|trades|sessions)? ?(?:have )?in common\b|\bwhat (?:sets|set|makes) (?:the |those |these )?(?:selected|chosen|losing|winning|losers|winners|bad|good|worst|best)\b|\b(?:selected|chosen) (?:days|trades|sessions)\b", re.I)


def _selection_filter(text: str) -> str:
    low = text.lower()
    return "winning" if re.search(r"\bwin|\bwinners?\b|\bright\b|\bgood\b|\bworked\b", low) else "best" if re.search(r"\bbest\b", low) else \
        "worst" if re.search(r"\bworst\b", low) else "losing"


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

    # ------------------------------------------------------------------------------------------- V9: the model reads first
    def _model_route(self, llm, text: str, current, known: list[str], default_symbol: str) -> tuple[dict | None, str | None, str]:
        """The language model's reading of the message, checked part by part (model_reader.py). Returns (route, note, reader).
        A route of None hands the message to the deterministic readers (the note says why when it matters)."""
        from strata.ai import model_reader as MR
        from strata.ai.llm import LLMError
        sid, st = self._v8_state()
        frame, pending = st.get("frame"), st.get("pending")
        if pending and CV.answer_clarification(text, pending) is not None:
            return None, None, "built-in rules"  # the answer to a question just asked: the buttons' own path
        result_kind = None
        if sid:
            lr = self._latest_result(sid)
            result_kind = ((lr or {}).get("content") or {}).get("view") if lr else None
        ctx = MR.context_of(frame, current, result_kind, [{"role": h["role"], "text": h["content"][:240]} for h in self._history(sid)][-4:] if sid else [], st.get("selection"))
        MR.reset()
        try:
            raw = MR.ask_model(llm, text, ctx)
        except LLMError as e:
            return self._model_failed(llm.name, str(e)[:160], text)
        self._tl.model_raw = raw
        act = raw.get("act") if isinstance(raw, dict) else None
        if act not in ("ask", "strategy", "edit", "command", "unsupported", "unclear"):
            return self._model_failed(llm.name, "the reply is not a reading in Strata's question language", text)
        bad = MR.invalid(raw)
        if bad:
            return self._model_failed(llm.name, bad, text)
        name = llm.name
        has = bool(frame and frame.get("kind") in CV.QUESTION_KINDS) or current is not None
        low = text.lower()
        # what this reading cannot express goes to the readers that can: ordered events, grids of settings, several symbols
        if re.search(r"\bthen\b|\bafter that\b|\bfirst\b.{0,40}\b(?:and|then)\b.{0,30}\b(?:back|falls?|drops?|rises?|reclaims?|fails?)\b", low) and act in ("strategy", "edit"):
            return None, None, "built-in rules"
        if re.search(r"\bcompare\b.{0,60}\b(?:\d+\s*dte|\d+:\d\d|\d+%)\b.{0,20}\b(?:and|vs|versus|or)\b|\bfrom \d+(?:\.\d+)?%? to \d+(?:\.\d+)?%?\b|\bevery \d+ minutes\b|\bsteps?\b", low):
            return None, None, "built-in rules"
        if act == "edit" and not has:
            act = "strategy" if raw.get("strat") else "ask"
        # ---- actions on what is on screen
        if act == "command":
            cmd = raw.get("cmd")
            if not has and cmd not in ("teach",):
                return {"action": "clarify", "text": "There is no result in this conversation yet. Ask a question or describe a strategy first, and then ask about it."}, None, name
            if cmd == "undo":
                return {"action": "undo"}, None, name
            if cmd == "compare":
                return {"action": "compare_versions"}, None, name
            if cmd == "explore":
                k = (frame or {}).get("kind") or ("strategy" if current is not None and current.kind == "options" else "frequency")
                f = raw.get("filter") or _explore_filter(text, k)
                return {"action": "show_trades", "filter": OLD_FILTER.get(f, "all"), "explore_filter": f}, None, name
            if cmd == "explain":
                w = WHICH_TRADE.search(text)
                if current is not None and current.kind == "options":
                    return {"action": "explain_trade", "which": (w.group(1).lower() if w else "worst")}, None, name
                topic = TEACH.topic_of(text) or "chart"
                return {"action": "teach", "topic": topic}, None, name
            if cmd == "teach":
                return {"action": "teach", "topic": TEACH.topic_of(text) or "chart"}, None, name
            if cmd == "reliability":
                return {"action": "robustness"}, None, name
            if cmd == "next":
                return {"action": "next_steps"}, None, name
            if cmd == "about_selection":
                return {"action": "about_selection", "filter": raw.get("filter") or "all", "question": text}, None, name
            if cmd == "prices":
                return {"action": "fills"}, None, name
            return None, None, "built-in rules"
        if act == "unsupported":
            u = raw.get("unsupported") or {}
            det = SM.read(text, known, default_symbol)
            tickers = [t for t in re.findall(r"\b[A-Z]{2,5}\b", text) if t not in ("ET", "PT", "CT", "MT", "EST", "PST", "AM", "PM", "DTE", "ATM", "OTM", "ITM", "VWAP", "RSI", "EOD", "CPI", "FOMC") and t not in known]
            grounds = det.flag or tickers or re.search(r"\b(?:will|gonna|going to)\b.{0,30}\b(?:tomorrow|next week|today)\b|\bfomc|\bfed\b|\bcpi\b|\bnfp\b|\bjobs report|\bearnings|\bopex|\brsi\b|\bmacd\b|moving average|\bsma\b|\bema\b|bollinger|\b(?:19\d\d|20(?:0\d|1\d|2[0-3]))\b", low)
            if not grounds:
                return None, None, "built-in rules"  # the model's "cannot be answered" has no deterministic ground: the readers try
            why = (u.get("why") or "").strip().rstrip(".")
            closest = (u.get("closest") or "").strip()
            ok_closest = closest and not MR.find_quote(closest, text) and len(closest) > 12
            txt = f"That cannot be answered from the stored data: {why}." if why else "That cannot be answered from the stored data."
            det = SM.read(text, known, default_symbol)
            if det.flag:
                txt = det.flag  # the deterministic reason, when there is one, is the one shown
            return {"action": "clarify", "declined": True, "text": txt + (" Nothing was run." if not ok_closest else " Nothing was run. The closest question that can be answered:"),
                    "options": ([{"label": closest, "ask": closest}] if ok_closest else [])}, None, name
        if act == "unclear":
            u = raw.get("unclear") or {}
            det = SM.read(text, known, default_symbol)
            if det.complete and det.frame.get("kind") in CV.QUESTION_KINDS:
                return None, None, "built-in rules"  # the built-in reader read every part of it: no reason to ask
            opts = [o for o in (u.get("opts") or []) if isinstance(o, str) and len(o) > 8][:3]
            return {"action": "clarify", "text": (u.get("ask") or "Which part do you mean?").strip(), "options": [{"label": o, "ask": o} for o in opts]}, None, name
        # ---- a strategy, or a change to the strategy on screen
        strat_edit = act == "edit" and current is not None and current.kind == "options" and not (frame and frame.get("kind") in CV.QUESTION_KINDS)
        if act == "strategy" or strat_edit or (act == "edit" and raw.get("strat") and frame and frame.get("kind") in CV.QUESTION_KINDS):
            base = current if strat_edit else None
            if act == "edit" and not strat_edit and frame:
                # "what if I bought calls instead": the question's condition traded (the V8 conversion), then the model's parts on top
                conv = CV.to_strategy(frame, text)
                base = conv["draft"]
            d, problems, quotes, changed = MR.strategy_draft(raw, text, known, default_symbol, base)
            unknown = next((p_ for p_ in problems if p_.startswith("no price history for")), None)
            if unknown:
                sym_ = unknown.split("for ")[1].split(" ")[0]
                return {"action": "clarify", "declined": True, "text": f"No price history for {sym_} is stored, so it cannot be backtested. Stored: {', '.join(k for k in known if k != 'VIX')}. Nothing was run.",
                        "options": [{"label": re.sub(re.escape(sym_), "SPY", text, flags=re.I), "ask": re.sub(re.escape(sym_), "SPY", text, flags=re.I)}]}, None, name
            left = MR.coverage(text, quotes, known)
            if problems or left:
                return None, (f"The language model's reading ({name}) left part of this unplaced ({'; '.join((problems + [f'not placed: {x}' for x in left])[:3])}), so the built-in reader read it." ), "built-in rules"
            if base is not None and strat_edit:
                from strata.ai.draft import diff_drafts
                ch = [x["field"] for x in diff_drafts(current, d)]
                if not ch:
                    return {"action": "noop", "text": "That does not change anything in the current strategy."}, None, name
                return {"action": "modify", "draft": d, "changed": ch, "compare": wants_compare(text), "edit": True}, None, name
            if base is not None:
                nf = {"kind": "strategy", "symbol": d.symbol, "draft": d.model_dump(mode="json"), "text": text}
                return {"action": "new", "draft": d, "frame": nf, "edit": True, "changes": CV.summary(frame, nf), "assumptions": conv["assumptions"]}, None, name
            return {"action": "new", "draft": d, "sweep": None, "model_checked": True}, None, name
        # ---- a question, or a change to the question on screen
        if act in ("ask", "edit"):
            base = frame if (act == "edit" and frame and frame.get("kind") in CV.QUESTION_KINDS) else None
            f, problems, quotes = MR.question_frame(raw, text, known, frame.get("symbol") if base else default_symbol, base)
            if f and f.get("kind") == "unsupported":
                return {"action": "clarify", "declined": True, "text": f"{f['why']} Stored: {', '.join(known)}. Nothing was run.",
                        "options": [{"label": re.sub(re.escape(f['symbol']), s, text, flags=re.I), "ask": re.sub(re.escape(f['symbol']), s, text, flags=re.I)} for s in ("SPY", "QQQ") if s in known]}, None, name
            also = [((raw.get("candle") or {}).get("q") or "")]
            left = MR.coverage(text, quotes + [x for x in also if x], known)
            if base is not None:
                left = [x for x in left if not re.fullmatch(r"(?i)(?:instead|actually|rather|nah|no|use|try|make it|what about|how about|only|just|also|too|and)", x.strip())]
            if f is None or f.get("kind") not in CV.QUESTION_KINDS or problems or left:
                det = SM.read(text, known, default_symbol) if base is None else None
                why = "; ".join((problems + [f"not placed: \"{x}\"" for x in left])[:3])
                if det is not None and det.complete and det.frame.get("kind") in CV.QUESTION_KINDS:
                    return None, f"The language model's reading ({name}) was incomplete ({why}); the built-in reader read every part, so its reading is used.", "built-in rules"
                return None, f"The language model's reading ({name}) could not be checked ({why}); the built-in reader read this message.", "built-in rules"
            if base is not None:
                ch = CV.summary(base, f)
                if not ch:
                    return {"action": "noop", "text": "That does not change anything in the current question."}, None, name
                route = SM.to_route(SM.Reading(text=f.get("text") or text, frame=f, mentions=[]))
                if f.get("kind") == "frequency" and (f.get("outcome") or {}).get("key") == "event":
                    route = {"action": "profile", "frame": f}
                route.update(frame=f, edit=True, changes=ch, slot=None)
                if route.get("action") == "new" and current is not None and getattr(current, "kind", None) == "study" and route.get("draft") is not None:
                    from strata.ai.draft import diff_drafts
                    route.update(action="modify", changed=[x["field"] for x in diff_drafts(current, route["draft"])], compare=wants_compare(text))
                return route, None, name
            # cross-check: the deterministic reader read the same sentence completely and differently -> ask, nothing runs
            det = SM.read(text, known, default_symbol)
            reader = name
            if det.complete and det.frame.get("kind") in CV.QUESTION_KINDS:
                a, b = CV.facts(f), CV.facts(det.frame)
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in ("period",) or (k == "period" and a.get(k) != b.get(k)))
                if diff and set(diff) <= {"outcome"} and f.get("kind") == det.frame.get("kind") == "frequency" and det.frame.get("outcome") and f.get("outcome"):
                    # the two read the same question and differ only in what the outcome is measured against ("close higher":
                    # than the open, or than the previous close): both are measured and said, as an ambiguous word always is
                    f2 = copy.deepcopy(det.frame)
                    o1, o2 = det.frame["outcome"], f["outcome"]
                    f2["outcome_options"] = [{"label": SM.describe_outcome(o1, f2.get("start") or "open"), "outcome": o1}, {"label": SM.describe_outcome(o2, f2.get("start") or "open"), "outcome": o2}]
                    f2.setdefault("assumptions", []).append({"slot": "outcome", "text": f"\"{o1.get('text') or o2.get('text') or 'the outcome'}\" can be read two ways, so both are measured: " + "; ".join(x["label"] for x in f2["outcome_options"]) + "."})
                    route = SM.to_route(SM.Reading(text=text, frame=f2, mentions=[]))
                    route["frame"] = f2
                    return route, None, f"{name} and the built-in reader (two readings, both measured)"
                if diff:
                    da, db_ = SM.describe(f), SM.describe(det.frame)
                    if da != db_:
                        keep = lambda fr: {k: v for k, v in fr.items() if k not in ("text",)}  # noqa: E731
                        return {"action": "clarify", "text": "This could be read two ways, so nothing was run yet. Which did you mean?",
                                "options": [{"label": da, "patch": keep(f)}, {"label": db_, "patch": keep(det.frame)}],
                                "pending": {"kind": "frame_patch", "options": [{"label": da, "patch": keep(f)}, {"label": db_, "patch": keep(det.frame)}], "frame": f}}, None, name
                reader = f"{name}, confirmed by the built-in reader"
            if f.get("kind") == "frequency" and (f.get("outcome") or {}).get("key") == "event":
                return {"action": "profile", "frame": f}, None, reader
            route = SM.to_route(SM.Reading(text=text, frame=f, mentions=[]))
            route["frame"] = f
            return route, None, reader
        return None, None, "built-in rules"

    def _model_failed(self, name: str, why: str, text: str) -> tuple[dict | None, str | None, str]:
        """The model gave no usable reading. The built-in reader reads the message and the reply says so, or, with the
        fallback turned off in Settings, nothing runs and the reply says why."""
        from strata.api.settings import get_setting
        from strata.ai.agent import is_complex
        if str(get_setting("llm.fallback") or "rules") == "none":
            return {"action": "failed", "text": f"The language model ({name}) could not interpret this request: {why}. Falling back to the built-in interpreter is turned off in Settings, so nothing was run."}, None, name
        return None, (f"The language model ({name}) did not return a usable interpretation ({why}). The built-in interpreter read this request instead"
                      + (" - it understands fewer phrasings, so check the interpretation below before relying on it." if is_complex(text) else ".")), "built-in rules"

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
        if (frame or current) and NEXT_STEPS.search(text):
            return {"action": "next_steps"}
        if (frame or current) and ABOUT_SELECTION.search(text):
            return {"action": "about_selection", "filter": _selection_filter(text), "question": text}
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
                    if nf.get("kind") != frame.get("kind") and frame.get("kind") in CV.QUESTION_KINDS and not _names_a_kind(text):
                        nf = {**nf, "kind": frame["kind"], **({"outcome": frame["outcome"]} if frame.get("outcome") else {})}
                        for k_ in ("measure", "bucket", "by", "focus", "candle", "scope", "predictor"):
                            if frame.get(k_) is not None:
                                nf[k_] = frame[k_]
                        e["changes"] = CV.summary(frame, nf)
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
        # 3. the current strategy: its sizing in a trader's words, a bare number, or a time moved relative to itself
        if current is not None and current.kind == "options":
            sz = CV.sizing_from_text(text, current)
            if sz is not None and not re.search(r"\b(?:calls?|puts?|straddles?|strangles?|spreads?|condors?)\b", text.lower()):
                d2 = current.model_copy(deep=True)
                d2.sizing = sz[0]
                if sz[1]:
                    d2.quantity = sz[1]
                return {"action": "modify", "draft": d2, "changed": ["sizing"], "edit": True}
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
        # V9: every first-candle prediction question is answered by the agreement frame (one question type, one engine). Until
        # V9 "does the first N-minute candle predict the REST of the session" went to the hypothesis lab instead, with a
        # different answer for the same question; the lab keeps explicit "investigate whether ..." hypotheses.
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
        if d.unsupported:
            held = {round(abs(float(v)), 6) for v in (d.premium_target_pct, d.premium_stop_pct, d.underlying_target_pct, d.underlying_stop_pct) if v is not None}
            held |= {round(abs(float(x)), 6) for c in d.conditions for x in re.findall(r"-?\d+(?:\.\d+)?", c.expr)}
            def placed(u: str) -> bool:
                xs = re.findall(r"\d+(?:\.\d+)?", u)
                # only an exit phrase ("100% stop") whose number the draft holds: a condition the reader misread into a target
                # ("dropped at least 3.5%" as a 3.5% target) must stay flagged
                return bool(xs) and all(round(float(x), 6) in held for x in xs) and bool(re.search(r"\bstop|\btarget|\bprofit|\bloss|\btp\b|\bsl\b", u.lower())) \
                    and not re.search(r"\b(?:drop|fell|fall|decline|rall|rose|gap|session|day|yesterday|previous)\w*", u.lower())
            d.unsupported = [u for u in d.unsupported if not placed(u)]
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

    # ------------------------------------------------------------------------------------- V9: exploring a result further
    def _observations_of(self, sid: str, flt: str) -> tuple[str, list[dict], list[dict], str, str] | None:
        """The latest result's observations split into (kind, group, rest, group_name, rest_name): its trades, or its days.
        A selection made in the explore sheet (session state) wins over the words' filter."""
        from strata.research import explore as XP
        m = self._latest_result(sid)
        c = (m or {}).get("content") or {}
        _, st = self._v8_state()
        sl = st.get("selection") or {}
        sel = set(sl.get("ids") or []) if (m is not None and sl.get("result") == m["id"]) else set()  # a selection made on an earlier result does not apply
        view = c.get("view")
        if view == "single" and c.get("run_id"):
            run = db.get_run(c["run_id"]) or {}
            if run.get("kind") == "backtest":
                from strata.api.results import load_trades
                trades = load_trades(run["id"])
                rows = []
                for t in trades:
                    leg = next((x for x in t.get("legs") or [] if x.get("kind") == "option"), {})
                    es, xs = t.get("entry_spot"), t.get("exit_spot")
                    rows.append({"id": t["id"], "date": t.get("entry_date"), "pnl": float(t.get("pnl") or 0), "premium": leg.get("entry_price"), "held": t.get("holding_minutes"),
                                 "entry_minute": t.get("entry_minute"), "exit_reason": str(t.get("exit_reason") or "").replace("_", " "), "vix": t.get("entry_vix"),
                                 "underlying_move": ((xs / es - 1) * 100) if es and xs else None})
                if sel:
                    g = [r for r in rows if str(r["id"]) in sel or r["date"] in sel]
                    names = (f"the {len(g)} selected trades", "the other trades")
                elif flt in ("winning", "best", "hit"):
                    g = [r for r in rows if r["pnl"] > 0] if flt != "best" else sorted(rows, key=lambda r: -r["pnl"])[:max(5, len(rows) // 5)]
                    names = ("the winning trades" if flt != "best" else "the best trades", "the others")
                else:
                    g = [r for r in rows if r["pnl"] <= 0] if flt != "worst" else sorted(rows, key=lambda r: r["pnl"])[:max(5, len(rows) // 5)]
                    names = ("the losing trades" if flt != "worst" else "the worst trades", "the others")
                ids = {r["id"] for r in g}
                return "trades", g, [r for r in rows if r["id"] not in ids], names[0], names[1]
            if run.get("kind") == "study":
                ss = XP.sessions(run["id"], "all").get("rows") or []
                rows = [{"date": r["date"], "hit": r.get("happened") not in ("no", "unclear")} for r in ss]
                return self._split_days(rows, sel, flt, "the days it happened", "the days it did not")
        if view == "investigation":
            rows_ = ((c.get("report") or {}).get("agreement") or [])
            rest = [x for x in rows_ if x["scope"] == "rest"] or rows_
            if rest:
                x = sorted(rest, key=lambda z: z["minutes"])[0]
                cl = XP.classification([x["up"]["run_id"], x["down"]["run_id"]], "all").get("rows") or []
                rows = [{"date": r["date"], "hit": r.get("matched") == "yes"} for r in cl]
                return self._split_days(rows, sel, flt, "the days the candle was right", "the days it was wrong")
        if view == "profile":
            pr = XP.profile(self.store, c.get("frame") or {}, "all").get("rows") or []
            rows = [{"date": r.get("date"), "hit": str(r.get("happened", "yes")) != "no"} for r in pr if r.get("date")]
            return self._split_days(rows, sel, flt, "the days it happened", "the days it did not")
        return None

    def _split_days(self, rows: list[dict], sel: set, flt: str, hit_name: str, miss_name: str):
        if sel:
            g = [r for r in rows if r["date"] in sel]
            return "days", g, [r for r in rows if r["date"] not in sel], f"the {len(g)} selected days", "the other days"
        miss = flt in ("losing", "miss", "worst", "all", None)
        g = [r for r in rows if (not r["hit"]) == miss]
        return "days", g, [r for r in rows if (not r["hit"]) != miss], (miss_name if miss else hit_name), (hit_name if miss else miss_name)

    def _about_selection(self, sid: str, r: dict) -> list[dict]:
        """"Is there anything interesting about the losing trades?": the group against the rest of the same result."""
        from strata.research import contrast as CT
        got = self._observations_of(sid, r.get("filter") or "all")
        if got is None:
            return [self._say(sid, "There is no result in this conversation to look into yet. Ask a question or run a strategy first.")]
        kind, g, rest, gname, rname = got
        m = self._latest_result(sid)
        c = (m or {}).get("content") or {}
        sym = ((c.get("frame") or {}).get("symbol")) or (db.get_run(c.get("run_id")) or {}).get("config", {}).get("symbol") if c.get("run_id") else ((c.get("frame") or {}).get("symbol") or "SPY")
        sym = sym or "SPY"
        try:
            ctx = CT.day_context(self.store, sym, [x["date"] for x in g + rest if x.get("date")])
        except Exception:  # noqa: BLE001
            ctx = {}
        def feat(x):
            d = ctx.get(str(x.get("date"))[:10], {})
            out = {k: v for k, v in x.items() if k not in ("id", "date", "hit", "pnl")}
            out.update({"gap": d.get("gap"), "prev": d.get("prev"), "weekday": d.get("weekday"), "year": d.get("year")})
            if out.get("vix") is None:
                out["vix"] = d.get("vix")
            if kind == "trades":
                out["oc"] = d.get("oc")
            return out
        res = CT.compare([feat(x) for x in g], [feat(x) for x in rest], kind, gname, rname)
        head = res["lines"][0] if res["lines"] else f"Nothing sets {gname} apart."
        ans = {"kind": "contrast", "headline": head, "happened": [l for l in res["lines"][1:] if "compared" not in l and "small sample" not in l], "matters": None,
               "limits": [l for l in res["lines"] if "compared" in l or "small sample" in l],
               "numbers": [], "followups": [{"text": "Show me those " + kind, "why": "The observations behind this comparison."}, {"text": "What would you investigate next?", "why": "Suggestions from this result."}],
               "viz": {"kind": "contrast", "rows": res["rows"][:8], "group": gname, "rest": rname}, "details": {"compared": res["compared"], "group": len(g), "rest": len(rest)}}
        return [db.add_message(sid, "assistant", "result", {"view": "contrast", "answer": ans, "text": head, "frame": c.get("frame")})]

    def _next_steps(self, sid: str, version) -> list[dict]:
        """"What would you investigate next?": suggestions chosen from what this result showed, each with its reason and the
        message that runs it. Deterministic: no number is stated that the result does not hold."""
        m = self._latest_result(sid)
        c = (m or {}).get("content") or {}
        if not c:
            return [self._say(sid, "There is no result in this conversation yet. Ask a question first, and I will suggest where to look next.")]
        a = c.get("answer") or {}
        f = c.get("frame") or {}
        view = c.get("view")
        sugg: list[dict] = []
        asked = {re.sub(r"\W+", " ", ((mm.get("content") or {}).get("text") or "").lower()).strip() for mm in db.list_messages(sid) if mm["role"] == "user"}
        def add(ask, why):
            if ask not in [x["ask"] for x in sugg] and re.sub(r"\W+", " ", ask.lower()).strip() not in asked:  # not what was already asked here
                sugg.append({"ask": ask, "why": why})
        run = db.get_run(c.get("run_id")) if c.get("run_id") else None
        if run and run.get("kind") == "backtest":
            from strata.api.results import load_trades
            tr = load_trades(run["id"])
            n = len(tr)
            losers = sum(1 for t in tr if (t.get("pnl") or 0) <= 0)
            if n < 30:
                add("Try the same strategy on QQQ and IWM", f"{n} trades is too few to tell a real edge from luck; the same rule on other index funds adds observations.")
            add("Is this just luck?", "Checks whether the result stands out from what random entries on the same days would give, accounting for how many versions were tried.")
            if losers:
                add("Is there anything interesting about the losing trades?", f"{losers} of the {n} trades lost money; comparing them with the winners can point to a filter worth testing.")
            d = version.get("draft") if version else {}
            et = (d or {}).get("entry_time")
            if et and et not in ("open", "close"):
                h, mi = int(et[:2]), int(et[3:])
                a1, a2 = f"{(h * 60 + mi - 15) // 60:02d}:{(h * 60 + mi - 15) % 60:02d}", f"{(h * 60 + mi + 15) // 60:02d}:{(h * 60 + mi + 15) % 60:02d}"
                add(f"Compare entries at {a1}, {et} and {a2} ET", "Whether the result depends on the exact entry minute: a real effect should not vanish 15 minutes away.")
            if not (d or {}).get("sizing"):
                add("What if I risked a fixed $500 per trade?", "Sizing by risk instead of a fixed number of contracts changes how much each trade can lose.")
        else:
            st = (a.get("details") or {}) if isinstance(a.get("details"), dict) else {}
            nums = {x.get("key"): x for x in a.get("numbers") or [] if isinstance(x, dict)}
            n = None
            for x in a.get("numbers") or []:
                if isinstance(x, dict) and isinstance(x.get("n"), (int, float)):
                    n = x["n"]
                    break
            text = " ".join([a.get("headline") or ""] + list(a.get("limits") or []))
            if n is not None and n < 40 or re.search(r"aren't enough|not enough|small sample|too few", text):
                others = [s_ for s_ in ("SPY", "QQQ", "IWM") if s_ != (f.get("symbol") or "SPY")]
                add(("Drop the period limit and try " if any(f.get("period") or []) else "Try ") + f"{others[0]} and {others[1]} too", "The sample is small; more sessions make the estimate firmer.")
            weak = bool(re.search(r"within what chance|not distinguishable|little evidence|about what|no clear", text))
            for fu in ANS.followups(f, f.get("kind") or "frequency")[:4]:
                add(fu["text"], fu.get("why") or "")
            if not weak and f.get("kind") in ("frequency", "agreement", "movement"):
                add("Does this still hold since 2022?", "A pattern that is real should also show in the recent years, not only in the early data.")
            if weak:
                add("Is a bigger move any different?", "The effect is weak at this size; a larger move may separate from chance, or show there is nothing there.")
        sugg = sugg[:5]
        if not sugg:
            return [self._say(sid, "I have no specific suggestion for this result.")]
        lines = [f"{i + 1}. {x['ask']} - {x['why']}" for i, x in enumerate(sugg)]
        return [self._say(sid, "Where I would look next, given this result:\n" + "\n".join(lines), "text", options=[{"label": x["ask"], "ask": x["ask"]} for x in sugg], next_steps=sugg)]

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
            if prev.get("kind") in CV.QUESTION_KINDS:
                c["frame"] = prev  # the restored question's own frame: its answer is worded from it (without it, "the outcome happened ...")
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
                    c["options"] = [{"label": o["label"], **({"ask": o["ask"]} if o.get("ask") else {})} for o in r["options"]]
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
                    ov = self._candle_overlay(rep)
                    if ov:
                        content["answer"]["overlay"] = ov
                else:
                    content["answer"] = frequency_set(rep, frame or (c.get("plan") or {}).get("frame") or {})
        except Exception as ex:  # noqa: BLE001
            content["answer_error"] = f"{type(ex).__name__}: {ex}"[:300]
        # V9: every answer carries an independent check of its numbers (strata.research.verify). It never changes the answer.
        try:
            a = content.get("answer")
            if a is None:
                return
            from strata.research import verify as VF
            if view == "single" and entries and entries[-1]["run"]["kind"] == "study":
                a["check"] = VF.study(entries[-1]["result"], frame or None)
            elif view == "single" and entries:
                from strata.api.results import load_trades
                draft = (c.get("draft") if isinstance(c.get("draft"), dict) else None) or (frame.get("draft") if isinstance(frame.get("draft"), dict) else None)
                a["check"] = VF.backtest(entries[-1]["run"], load_trades(entries[-1]["run"]["id"]), draft)
            elif view == "investigation" and content.get("report", {}).get("agreement"):
                a["check"] = VF.agreement(content["report"], frame or (c.get("plan") or {}).get("frame"))
        except Exception as ex:  # noqa: BLE001
            content.setdefault("answer", {})["check"] = {"status": "not_covered", "text": f"The independent check could not run ({type(ex).__name__})."}

    @staticmethod
    def _candle_overlay(rep: dict) -> dict | None:
        """The rest of the session after an up candle and after a down candle, from the candle's close: the median path and the
        middle half of the days, from the two studies the agreement already ran (study path summaries). If the candle
        predicted the rest of the day, the two would pull apart."""
        from strata.api.results import load_run_result
        rows = [x for x in rep.get("agreement") or [] if x.get("scope") == "rest"]
        if not rows:
            return None
        x = sorted(rows, key=lambda z: z["minutes"])[0]
        out = {"minutes_candle": x["minutes"]}
        for side in ("up", "down"):
            run = db.get_run((x.get(side) or {}).get("run_id") or "")
            if not run:
                return None
            ps = (load_run_result(run) or {}).get("path_summary") or {}
            if not ps.get("minutes"):
                return None
            out[side] = {k: ps.get(k) for k in ("minutes", "median", "p25", "p75", "days")}
        return out

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
