"""Research-graph, memory, registry and forward-validation behaviour of the research agent.

Kept apart from agent.py so the conversation loop there stays readable. Everything here
follows the same contract as the rest of the agent: the interpreter (rules or a language
model) only chooses WHAT to compute; the numbers in every reply come from a finished run
or from stored observations, and each message carries the run it refers to.
"""
from __future__ import annotations

import json
import re

from strata.ai import nl_graph as NG
from strata.ai import nl_rules as NL
from strata.api import db
from strata.research import graph as G
from strata.research import ledger as LG
from strata.research import memory as MEM
from strata.research import registry as RG

BATCH_MAX = 12


def _graph_of_run(run_id: str) -> dict | None:
    from strata.api.results import run_dir
    p = run_dir(run_id) / "graph.json"
    return json.loads(p.read_text()) if p.exists() else None


def richer(g: dict) -> bool:
    """Does the graph say more than the older single-question paths can? (Those stay as
    shortcuts for simple questions; everything they cannot express goes to the graph.)"""
    core = g.get("core") or {}
    cl = core.get("clauses") or []
    return bool(core.get("controls") or len(cl) >= 2 or core.get("start") or core.get("end") or any(s.get("param") == "symbol" for s in g.get("sweeps") or [])
                or any(c.get("at") not in ("open", "close") or c.get("subject") or c.get("other") or c.get("feature") not in ("ret_1",) for c in cl)
                or {"scan", "distribution"} & set(g.get("analyses") or []))


def prefer_graph(text: str, graph: dict, fresh: dict) -> bool:
    """Two readers produced a reading of the same question. Neither wins by its place in a list: the reading that accounts
    for the sentence does. The question reader (research graph) is used when the study reader left something unread, or when
    both look complete but disagree about what is being measured (its outcome reader knows 'close', 'within N days', 'on the
    day'). Where the two agree, the older path is kept, with the result views built for it."""
    from strata.ai import intent as I
    d = fresh.get("draft")
    if fresh.get("action") != "new" or d is None or d.kind != "study":
        return False
    gi, di = I.from_graph(graph), I.from_draft(d)
    g_ok = not gi["unsupported"] and all(c["ok"] for c in I.validate(gi, text))
    if not g_ok or not gi["conditions"]:
        return False
    d_ok = not di["unsupported"] and all(c["ok"] for c in I.validate(di, text))
    if not d_ok:
        return True
    go, do = gi.get("outcome") or {}, di.get("outcome") or {}
    if not do.get("size") and do.get("measure") == "touch":
        return True  # "touches" with no level to touch is not a reading of the outcome at all
    size = lambda o: round(float(o.get("size") or 0.0), 6)
    # "the next session", read from its open or from the close before it: where a sentence names no reference the two readers may
    # each pick one, and that is not a disagreement about what was asked
    hz = lambda o: {"sessions:1": "next", "session": "next"}.get(str(o.get("horizon")), str(o.get("horizon")))
    return len(gi["conditions"]) != len(di["conditions"]) or size(go) != size(do) or (go.get("measure") != do.get("measure") and bool(size(go))) or hz(go) != hz(do) \
        or (do.get("direction") is not None and go.get("direction") != do.get("direction"))


def _headline(res: dict) -> str:
    """The answer in a few sentences, leading with the analysis the request was about. The full result of every
    analysis is one click away; this is not where numbers are hidden, only where they are not all repeated."""
    by: dict = {}
    for n in res.get("nodes", []):
        r = n.get("result")
        if r:
            by.setdefault(r["kind"], r)
    first = lambda k, i=0: ((by.get(k) or {}).get("summary") or [None] * (i + 1))[i] if len((by.get(k) or {}).get("summary") or []) > i else None  # noqa: E731
    sw = by.get("sweep")
    if sw and sw.get("param") == "symbol":
        rows = [r for r in sw["rows"] if r.get("status") == "ok" and r.get("probability") is not None]
        bad = [r for r in sw["rows"] if r.get("status") != "ok"]
        parts = ["; ".join(f"{r['value']}: {r['probability'] * 100:.1f}% of {r['n']:,} sessions against {r['baseline_probability'] * 100:.1f}% of the others"
                           + (f" (p {G._fmt_p(r['p_value'])})" if r.get("p_value") is not None else "") + (", too few to judge" if r.get("thin") else "") for r in rows) + "."] if rows else []
        parts += list(sw.get("summary") or [])[1:]
        if bad:
            parts.append("Not computed: " + "; ".join(f"{r['value']} ({str(r.get('error'))[:90]})" for r in bad) + ".")
        return " ".join(parts) or "No symbol could be evaluated."
    if "regression" in by:  # "which conditions go with it": the answer is the scan, the base frequency is context
        tops = [t for t in (by["regression"].get("terms") or []) if t.get("distinguishable") or (t.get("p_adjusted") is not None and t["p_adjusted"] < 0.05)]
        lead = [first("regression")]
        if tops:
            lead.append("Distinguishable after correction: " + "; ".join(str(t.get("label") or t.get("name")) for t in tops[:4]) + ".")
        lead.append("These are associations inside one sample of sessions, not causes, and a split chosen this way needs its own validation.")
        return " ".join(x for x in lead + [first("probability")] if x)
    parts = [first("probability")]
    if "matched" in by:
        parts.append(first("matched"))
    if "distribution" in by and "matched" not in by:
        parts += [first("distribution"), first("distribution", 1)]
    if "scan" in by:
        parts += [first("scan"), first("scan", 1)]
    if "validation" in by:
        parts.append(first("validation"))
    parts = [x for x in parts if x]
    return " ".join(parts) if parts else (" ".join(res.get("summary", [])[:3]) or "No analysis produced a result.")


class ResearchV5:
    """Mixin for strata.ai.agent.Agent."""

    # ------------------------------------------------------------------ routing (built-in interpreter)
    def v5_route(self, sid: str | None, text: str, known: list[str], tick: list[str], default_symbol: str, has_current: bool) -> dict | None:
        low = " " + text.lower().strip() + " "
        graphs = MEM.graphs_of(sid) if sid else []
        if re.search(r"already (?:tested|tried|run|looked|examined)|(?:have|did) (?:i|we) (?:already |ever )?(?:test|tested|tried|try|run)|similar configurations?|on (?:this|the|that) holdout|holdout (?:status|state)|"
                     r"(?:is|was) (?:the|this|my) holdout|how many (?:configurations|things|ideas) have (?:i|we) (?:tested|tried)|experiment registry", low):
            return {"action": "registry"}
        if re.search(r"forward[- ]validat\w*|forward[- ]test\w*|frozen (?:forward|experiment|configuration|version)|paper[- ]trad\w*|freeze (?:this|that|the|it|these)|start (?:a |the )?(?:frozen|forward)", low):
            return {"action": "forward"}
        if re.search(r"what have (?:we|i) (?:found|learned|tested|established|discovered)|summari[sz]e (?:this|our|the|my) (?:research|notebook|findings)|research memory|what(?:'s| is) still open|outstanding questions|"
                     r"open questions|what did (?:we|i) (?:find|learn|discard|rule out)|what have (?:we|i) ruled out", low):
            return {"action": "memory"}
        if re.search(r"(?:which|are)\b.{0,40}\b(?:results|findings|studies)\b.{0,30}\bindependent|actually independent|independent (?:of each other|evidence)", low):
            return {"action": "independent"}
        if re.search(r"why did\b.{0,40}\bfilter\b.{0,60}(?:win rate|hit rate).{0,60}(?:expectancy|average|profit)|(?:improve|raise|increase)d? (?:the )?win rate but (?:reduce|lower|hurt|decrease)", low):
            return {"action": "filter_effect"}
        m = re.search(r"still (?:hold|holds|true|work|works|valid|stand|stands)\b.{0,40}?(?:after|since|from|post[- ]?)\s*((?:19|20)\d{2})|hold(?:s)? up (?:after|since)\s*((?:19|20)\d{2})", low)
        if m:
            return {"action": "still_hold", "year": int(m.group(1) or m.group(2))}
        if re.search(r"\bcombine\b.{0,80}\b(?:condition|filter|rule|signal)\b.{0,80}\bwith\b", low):
            return {"action": "combine", "text": text}
        if re.search(r"something similar to what (?:we|i) (?:discovered|found|saw|learned)|similar to (?:what (?:we|i) found|our finding|yesterday'?s? finding|that finding)|find (?:something|anything) similar", low):
            return {"action": "similar", "text": text}
        if re.search(r"(?:build|create|make|design|draft|turn|convert)\b.{0,40}\bstrateg\w*\b.{0,60}\b(?:finding|pattern|result)s?\b|strateg\w* (?:from|out of|on) (?:the |our |my )?(?:strongest|best|top)", low):
            return {"action": "strategy_from_finding", "anyway": bool(re.search(r"\banyway\b|regardless|even so|still draft", low))}
        if re.search(r"backtest (?:that|this|the|those|these) strateg\w*|price (?:that|this|those|these) strateg\w*|strateg\w* using observed|using observed (?:historical )?options? prices", low) and sid and self._last_strategy_plan(sid):
            return {"action": "backtest_plan"}
        if graphs:
            m = re.search(r"(?:apply|run|try|test|repeat|do)\s+(?:the\s+)?(?:same|that|this|it)\s*(?:idea|analysis|study|test|thing|question)?\s+(?:to|on|for|with)\s+\$?([a-z]{1,5})\b|same (?:idea|analysis|study|thing|test) (?:on|for|with|to) \$?([a-z]{1,5})\b", low)
            if m and (m.group(1) or m.group(2)).upper() in set(tick) | set(known):
                return {"action": "graph_edit", "edit": "retarget", "symbol": (m.group(1) or m.group(2)).upper()}
            m = NG.COMPARE.search(low)
            if m and re.search(r"\b(?:that|this|it|those|the same|these)\b", low[:m.end()]):
                syms = [t.upper() for t in re.findall(r"\b[a-z]{2,5}\b", m.group(1)) if t.upper() in set(tick) | set(known)]
                if syms:
                    return {"action": "graph_edit", "edit": "symbols", "symbols": syms}
            if re.search(r"distribution of (?:the )?(?:price )?paths|price paths?|paths? (?:following|after) (?:those|these|that|the)|show (?:me )?the distribution|what do the paths look like", low):
                return {"action": "graph_edit", "edit": "paths"}
            if re.search(r"(?:which|what) (?:historical |market |observable |other )?(?:conditions?|factors?|variables?)\b.{0,30}\b(?:associated|predict|explain|matter|go with|linked|related)|what else matters|what predicts (?:it|that|a recovery|the recovery)", low) \
                    and not re.search(r"\b(?:when|if)\b.{3,}", low):
                return {"action": "graph_edit", "edit": "scan"}
            m = NG.CONTROL.search(low)
            if m and len(low.split()) <= 12 and not NG.INTRO.search(low[:m.start()]):
                return {"action": "graph_edit", "edit": "controls", "text": m.group(1)}
        return None

    def v5_graph(self, text: str, known: list[str], default_symbol: str) -> dict | None:
        low = text.lower()
        # an order or an option position is a strategy, not a question about the underlying: those keep their own path
        if NL.find_structure(low) or re.search(r"\b(?:buy|sell|short|go long|enter|exit|close out|take profits?|stop[- ]loss|contracts?|\d*\s*dte|strikes?|premiums?|calls?|puts?|spreads?|straddles?|condors?|scale in|add another)\b", low):
            return None
        g = NG.plan(text, known, default_symbol, self.store)
        if g is None or g.get("errors"):
            return None
        if any(str(u).startswith("the size '") for u in g.get("unsupported") or []):
            # a number in the question found no place in this reading: answering a different, smaller question is worse than not answering.
            # The other reader gets the sentence; if it cannot place the number either, the reply says which words were not understood.
            self._dropped_size = [u for u in g["unsupported"] if str(u).startswith("the size '")]
            return None
        return g

    # ------------------------------------------------------------------ launching a graph
    def _graph(self, sid: str, graph: dict, interp: str, note: str | None = None, title: str | None = None, force: bool = False, include_holdout: bool = False, derived_from: list[str] | None = None) -> list[dict]:
        s, _, _ = self._state(sid)
        desc = G.describe(graph, self.store)
        if desc.get("errors"):
            return [self._say(sid, "That analysis cannot be run as stated: " + "; ".join(desc["errors"][:4]) + ".", "error", interpreter=interp)]
        q = title or graph.get("question") or "Research graph"
        if not s["state"].get("current") and not MEM.graphs_of(sid):
            db.update_session(sid, title=q[:70])
        from strata.research.features import FEATURES
        pub = {"question": q, "nodes": desc["nodes"], "assumptions": graph.get("assumptions") or [], "unsupported": graph.get("unsupported") or [], "core": graph.get("core"), "analyses": graph.get("analyses"),
               "controls_labels": [FEATURES[f].label if f in FEATURES else f for f in ((graph.get("core") or {}).get("controls") or [])],
               "sweeps": graph.get("sweeps"), "planner": graph.get("planner") or interp, "include_holdout": include_holdout}
        base = {"graph": pub, "raw": graph, "question": q, "data": desc["required"], "problems": desc.get("problems") or [], "interpreter": interp, "finalized": False, "runs": [], "include_holdout": include_holdout,
                "derived_from": list(derived_from or [])}
        missing = sorted({r["symbol"] for r in desc["required"] if not r.get("ok") and r["what"].startswith("Daily bars")})
        if missing:
            txt = (f"This question reads {', '.join(missing)}, for which no price history is stored yet. Daily history is free; nothing has been run."
                   + (" " + note if note else ""))
            return [db.add_message(sid, "assistant", "graph", {**base, "text": txt, "state": "needs_data", "missing": missing,
                                                              "choices": [{"id": "download", "label": f"Download {', '.join(missing)} (free) and run"}, {"id": "cancel", "label": "Leave it"}]})]
        if desc.get("problems"):
            return [db.add_message(sid, "assistant", "graph", {**base, "text": "The stored data cannot answer this as asked: " + " ".join(desc["problems"]) + " Nothing has been run.", "state": "blocked", "finalized": True})]
        if pub["unsupported"] and not force:
            txt = ("I could read part of this question but not all of it. Not understood: " + "; ".join(f"\"{u}\"" for u in pub["unsupported"][:4])
                   + ". Running the rest would answer a different question, so nothing has been run. You can rephrase, or run it without those parts.")
            return [db.add_message(sid, "assistant", "graph", {**base, "text": txt, "state": "awaiting", "choices": [{"id": "run", "label": "Run it without those parts"}, {"id": "cancel", "label": "I'll rephrase"}]})]
        if include_holdout and not force:
            syms = sorted({n["params"]["symbol"].upper() for n in graph["nodes"] if n["op"] == "data"})
            hs = [h for sym in syms for h in (LG.current(sid, sym), LG.current(sid, sym + "@intraday")) if h and h["status"] in ("sealed", "used")]
            if hs:
                h = hs[0]
                txt = (f"This needs the sessions from {h['start']} to {h['end']}, which this notebook holds back as its final test period ({LG._pub(h)['label']}). Looking at them now will be recorded: the period can no longer "
                       "confirm anything afterwards. Nothing has been run.")
                return [db.add_message(sid, "assistant", "graph", {**base, "text": txt, "state": "awaiting", "choices": [{"id": "run", "label": "Use the final test period (recorded)"}, {"id": "cancel", "label": "Keep it untouched"}]})]
        return [self._submit_graph(sid, base, graph, note, include_holdout)]

    def _submit_graph(self, sid: str, base: dict, graph: dict, note: str | None, include_holdout: bool, message_id: int | None = None) -> dict:
        g = dict(graph)
        if include_holdout:
            g["include_holdout"] = True
        n_an = sum(1 for n in g["nodes"] if G.PRIMITIVES[n["op"]].output == "result")
        rid = db.create_run("graph", ("Graph: " + base["question"])[:90], {"question": base["question"], "session": sid, "nodes": len(g["nodes"]), "symbols": sorted({n["params"]["symbol"].upper() for n in g["nodes"] if n["op"] == "data"})})
        self.jobs.submit(rid, "graph", {"graph": g, "scope": sid})
        trials = LG.count(sid)
        txt = (f"Research graph: {len(g['nodes'])} steps, {n_an} analys{'es' if n_an != 1 else 'is'}. Every condition is checked to be known before the outcome starts, and every analysis is reported, including any that find nothing. "
               f"{trials} configuration{'s have' if trials != 1 else ' has'} been tested in this notebook so far." + (" " + note if note else ""))
        content = {**base, "text": txt, "state": "running", "runs": [{"run_id": rid, "mode": "graph", "label": base["question"][:90]}], "choices": []}
        if message_id is not None:
            db.update_message(message_id, content)
            return db.get_message(message_id)
        return db.add_message(sid, "assistant", "graph", content)

    def decide_graph(self, sid: str, message_id: int, choice: str) -> list[dict]:
        m = db.get_message(message_id)
        if m is None or m["session_id"] != sid or m["kind"] != "graph":
            raise LookupError("graph message not found")
        c = m["content"]
        if c.get("state") not in ("awaiting", "needs_data"):
            return [m]
        if choice == "cancel":
            db.update_message(message_id, {**c, "state": "cancelled", "finalized": True, "choices": [], "text": c["text"] + " (Left as it is.)"})
            return [db.get_message(message_id)]
        if choice == "download":
            rid = db.create_run("quickstart", f"Download {', '.join(c.get('missing') or [])}"[:90], {"symbols": c.get("missing") or [], "session": sid})
            self.jobs.submit(rid, "quickstart", {"symbols": c.get("missing") or [], "intraday": True})
            db.update_message(message_id, {**c, "state": "downloading", "choices": [], "runs": [{"run_id": rid, "mode": "download", "label": "free price history"}], "text": c["text"].replace(" Nothing has been run.", "") + " Downloading it now; the analysis starts when it is stored."})
            return [db.get_message(message_id)]
        if choice != "run":
            raise LookupError("choice not found")
        if c.get("pending"):  # a memory request that needed a yes before it could look at the final test period
            db.update_message(message_id, {**c, "state": "approved", "finalized": True, "choices": []})
            if c["pending"]["action"] == "still_hold":
                return [db.get_message(message_id)] + self._still_hold(sid, int(c["pending"]["year"]), force=True)
            raise LookupError("nothing to run")
        raw = dict(c["raw"])
        if c["graph"].get("unsupported"):
            raw["assumptions"] = list(raw.get("assumptions") or []) + ["Run WITHOUT these parts of the question, which were not understood: " + "; ".join(raw.get("unsupported") or [])]
        base = {k: v for k, v in c.items() if k not in ("text", "state", "choices", "runs")}
        base["raw"] = raw
        return [self._submit_graph(sid, base, raw, None, bool(c.get("include_holdout")), message_id)]

    def _advance_graph_message(self, sid: str, m: dict) -> bool:
        """A graph waiting for a download: start it when the data has arrived. Returns True when handled."""
        c = m["content"]
        if c.get("state") != "downloading":
            return False
        run = db.get_run(c["runs"][0]["run_id"]) if c.get("runs") else None
        if run is None or run["status"] not in ("done", "failed", "cancelled"):
            return True
        if run["status"] != "done":
            db.update_message(m["id"], {**c, "state": "blocked", "finalized": True, "text": "The download did not finish (" + ((run.get("error") or run["status"]).split("\n")[0][:160]) + "), so the analysis was not run."})
            return True
        from strata.research.features import _CACHE
        _CACHE.clear()
        desc = G.describe(c["raw"], self.store)
        if desc.get("problems"):
            db.update_message(m["id"], {**c, "state": "blocked", "finalized": True, "data": desc["required"], "text": "The download finished, but the data still cannot answer this: " + " ".join(desc["problems"])})
            return True
        self._submit_graph(sid, {**{k: v for k, v in c.items() if k not in ("text", "state", "choices", "runs")}, "data": desc["required"], "problems": []}, c["raw"], None, bool(c.get("include_holdout")), m["id"])
        return True

    def _finalize_graph(self, sid: str, c: dict, run: dict, mid: int) -> dict:
        base = {"source_message": mid, "run_id": run["id"], "explanation_source": "deterministic", "failed": []}
        if run["status"] != "done":
            return {**base, "view": "failed", "text": "The research graph did not finish: " + ((run.get("error") or run.get("message") or run["status"]).split("\n")[0][:300])}
        res = _graph_of_run(run["id"])
        if res.get("kind") == "graph_batch":
            return {**base, "view": "graph_batch", "mode": "graph", "text": " ".join(res["summary"]), "caveats": res.get("notes") or [], "label": res.get("title", "")[:80], "rows": res["rows"], "what": res.get("what")}
        found = MEM.findings_from_graph(res)
        MEM.record(sid, run["id"], found)
        stored = [f for f in MEM.list_findings(sid) if f["run_id"] == run["id"]]
        errs = [n for n in res["nodes"] if n["status"] == "error"]
        text = _headline(res)
        if errs:
            text += " Not computed: " + "; ".join(f"{n['title'].lower()} ({n['error']})" for n in errs[:3]) + "."
        kinds = [n["result"]["kind"] for n in res["nodes"] if n.get("result")]
        return {**base, "view": "graph", "mode": "graph", "text": text, "caveats": list(res["limits"]) + [f"Assumed: {a}" for a in res.get("assumptions") or []], "label": (res.get("question") or "Research graph")[:80],
                "question": res.get("question"), "symbols": res["symbols"], "analyses": kinds, "has_sweep": "sweep" in kinds, "trials": res["trials"],
                "findings": [{"id": f["id"], "evidence": f["evidence"], "statement": f["statement"], "kind": f["kind"], "why": f["why"], "convertible": bool((f.get("config") or {}).get("cell"))} for f in stored],
                "ledger": {"contaminated": next((r["contaminated"] for r in res.get("ledger") or [] if r.get("contaminated")), None)}}

    # ------------------------------------------------------------------ follow-ups on the last graph
    def _graph_edit(self, sid: str, r: dict, interp: str, text: str) -> list[dict]:
        graphs = MEM.graphs_of(sid)
        if not graphs:
            return [self._say(sid, "There is no research graph in this notebook yet to build on. Ask a market question first.")]
        last = graphs[-1]["graph"]
        if not last.get("core"):
            return [self._say(sid, "The last analysis was written as a custom graph, which has no single condition and outcome to carry over. Restate the question in full.")]
        edit = r["edit"]
        try:
            if edit == "retarget":
                g = NG.retarget(last, r["symbol"])
                note = f"The same conditions and outcome as before, applied to {r['symbol']}."
            elif edit == "symbols":
                base = last["core"]["symbol"]
                syms = list(dict.fromkeys([base] + [x for x in r["symbols"] if x != base]))
                g = NG.with_analyses(last, ["probability"] + (["matched"] if last["core"].get("controls") else []), [{"target": "data", "param": "symbol", "values": syms, **({"on": "matched"} if last["core"].get("controls") else {})}],
                                     re.sub(r"\s+-\s+compared across .*$", "", last.get("question", "")) + f" - compared across {', '.join(syms)}")
                note = f"The same conditions and outcome on {', '.join(syms)}. Thresholds are the ones used for {base}; they are not re-derived per symbol."
            elif edit == "paths":
                g = NG.with_analyses(last, ["distribution"], [], f"Price paths after: {'; '.join(c['label'] for c in last['core']['clauses'])}")
                note = "The same qualifying sessions as before; this shows what followed them."
            elif edit == "scan":
                g = NG.with_analyses(last, ["probability", "scan"], [], f"Which conditions go with the outcome, among sessions where {'; '.join(c['label'] for c in last['core']['clauses'])}")
                note = ("Among the qualifying sessions only: each candidate value splits them into thirds and the top third is compared with the bottom third. Every value scanned is listed and p-values are corrected for all of them. "
                        "A split found this way is within-sample and needs its own validation.")
            elif edit == "controls":
                from strata.research.matching import DEFAULT_CONTROLS
                feats = [f for k, fs in DEFAULT_CONTROLS.items() if re.search(rf"\b{re.escape(k)}\b", r["text"]) for f in fs]
                if not feats:
                    return [self._say(sid, f"I do not have a control variable for \"{r['text'].strip()}\". Available: " + ", ".join(sorted(DEFAULT_CONTROLS)) + ".")]
                g = NG.with_analyses(last, ["probability", "matched"], [], f"{last.get('question', '')} - controlling for {r['text'].strip()}", controls=list(dict.fromkeys(feats)))
                note = "The same question, now also compared with sessions matched on those variables."
            else:
                raise ValueError(edit)
        except ValueError as e:
            return [self._say(sid, str(e))]
        chk = G.validate(g)
        if chk["errors"]:
            return [self._say(sid, "That follow-up does not make a valid analysis: " + "; ".join(chk["errors"][:3]) + ".", "error")]
        return self._graph(sid, g, interp, note, derived_from=[graphs[-1]["run_id"]])

    # ------------------------------------------------------------------ memory tools
    def _memory(self, sid: str) -> list[dict]:
        mem = MEM.notebook(sid)
        return [db.add_message(sid, "assistant", "memory", {"text": MEM.summary_text(mem), "memory": mem, "view": "memory"})]

    def _independent(self, sid: str) -> list[dict]:
        items = []
        for g in MEM.graphs_of(sid):
            d = MEM.sessions_of(g["result"])
            if d:
                items.append({"label": (g["question"] or g["run_id"])[:90], "dates": set(d), "run_id": g["run_id"]})
        from strata.api.results import load_run_result
        for r in self._session_runs(sid):
            try:
                res = load_run_result(r["run"])
            except Exception:  # noqa: BLE001
                continue
            if r["kind"] == "study":
                d = {x["date"] for x in res.get("sessions", [])}
            else:
                from strata.api.results import load_trades
                d = {str(t["entry_date"])[:10] for t in load_trades(r["run_id"]) if t.get("entry_date")}
            if d:
                items.append({"label": r["label"][:90], "dates": d, "run_id": r["run_id"]})
        if len(items) < 2:
            return [self._say(sid, "Independence is a statement about two or more results, and this notebook has " + ("one result" if items else "no finished result") + " with qualifying sessions so far.")]
        out = MEM.independence(items)
        limits = ["Sharing sessions is the only dependence measured here. Results on different sessions of the same volatile month are also related; the week-resampled tests allow for that, this table does not."]
        an = {"title": "Which results are independent", "answer": out["sentence"], "answerable": True, "limits": limits,
              "facts": [{"label": "Results compared", "value": str(out["results"])}, {"label": "Independent sets of sessions", "value": str(out["independent_groups"])}],
              "tables": [{"title": "Groups that rest on the same sessions", "columns": ["group", "results", "distinct_sessions"], "rows": [{"group": i + 1, "results": "; ".join(g["members"]), "distinct_sessions": g["sessions"]} for i, g in enumerate(out["groups"])]},
                         {"title": "Shared sessions between each pair", "columns": ["a", "b", "shared", "share_of_smaller", "jaccard"], "rows": out["pairs"]}]}
        return [db.add_message(sid, "assistant", "result", {"view": "analysis", "tool": "independence", "text": out["sentence"], "analysis": an, "explanation_source": "deterministic", "caveats": limits, "label": an["title"]})]

    def _filter_effect(self, sid: str) -> list[dict]:
        """The two most recent results that differ by a filter: without it and with it."""
        from strata.api.results import load_run_result, load_trades
        pool = []
        for g in MEM.graphs_of(sid):
            d = MEM.sessions_of(g["result"])
            if d:
                pool.append({"label": g["question"][:80], "values": [v for v in d.values() if v is not None], "unit": "%", "n_clauses": len((g["graph"].get("core") or {}).get("clauses") or [])})
        for r in self._session_runs(sid):
            try:
                if r["kind"] == "study":
                    res = load_run_result(r["run"])
                    pool.append({"label": r["label"][:80], "values": [x["end_return_pct"] for x in res.get("sessions", [])], "unit": "%", "n_clauses": 0})
                else:
                    tr = load_trades(r["run_id"])
                    if tr:
                        pool.append({"label": r["label"][:80], "values": [float(t["pnl"]) for t in tr if t.get("pnl") is not None], "unit": "$", "n_clauses": 0})
            except Exception:  # noqa: BLE001
                continue
        pair = None
        for j in range(len(pool) - 1, 0, -1):
            for i in range(j - 1, -1, -1):
                if pool[i]["unit"] == pool[j]["unit"] and len(pool[i]["values"]) > len(pool[j]["values"]) > 0:
                    pair = (pool[i], pool[j])
                    break
            if pair:
                break
        if pair is None:
            return [self._say(sid, "To explain what a filter did I need two finished results in this notebook: one without the filter and one with it (the filtered one has fewer observations). Run both and ask again.")]
        a, b = pair
        fe = MEM.filter_effect(a["values"], b["values"], a["unit"])
        limits = ["Computed from the stored per-observation results of the two runs; the comparison is descriptive and makes no claim about which version is better out of sample."]
        d = fe.get("decomposition") or {}
        u = (lambda v: f"{v:+.3f}%") if a["unit"] == "%" else (lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.2f}")
        an = {"title": "What the filter changed", "answer": fe["sentence"] + " " + " ".join(fe["facts"]), "answerable": fe["answerable"], "limits": limits,
              "facts": [{"label": "Win rate", "value": f"{fe['before']['win_rate'] * 100:.1f}% -> {fe['after']['win_rate'] * 100:.1f}%"}, {"label": "Expectancy", "value": f"{u(fe['before']['expectancy'])} -> {u(fe['after']['expectancy'])}"}]
                       + ([{"label": "From the win rate", "value": u(d["from_win_rate"])}, {"label": "From win and loss sizes", "value": u(d["from_win_size"] + d["from_loss_size"])}] if d else []) if fe["answerable"] else [],
              "tables": [{"title": "Without and with the filter", "columns": ["version", "n", "win_rate", "avg_win", "avg_loss", "expectancy", "total"],
                          "rows": [{"version": "without: " + a["label"], **fe["before"]}, {"version": "with: " + b["label"], **fe["after"]}]}]}
        return [db.add_message(sid, "assistant", "result", {"view": "analysis", "tool": "filter_effect", "text": fe["sentence"], "analysis": an, "explanation_source": "deterministic", "caveats": limits, "label": an["title"]})]

    def _batch(self, sid: str, title: str, what: str, items: list[dict], note: str, include_holdout: bool = False) -> list[dict]:
        rid = db.create_run("graph", title[:90], {"question": title, "session": sid, "batch": len(items)})
        self.jobs.submit(rid, "graph", {"batch": items, "scope": sid, "title": title, "what": what, "include_holdout": include_holdout})
        return [db.add_message(sid, "assistant", "graph", {"text": note, "question": title, "graph": {"question": title, "nodes": [], "assumptions": [], "unsupported": [], "batch": [i["label"] for i in items]}, "data": [], "problems": [],
                                                          "finalized": False, "state": "running", "runs": [{"run_id": rid, "mode": "graph", "label": title[:90]}], "choices": []})]

    def _still_hold(self, sid: str, year: int, force: bool = False) -> list[dict]:
        fs = [f for f in MEM.list_findings(sid) if f["kind"] == "pattern" and (f.get("config") or {}).get("graph", {}).get("core")]
        if not fs:
            return [self._say(sid, "This notebook has no recorded pattern findings yet, so there is nothing to re-test. (Findings are recorded when a research graph finishes; nothing is assumed from memory.)")]
        seen, items = set(), []
        for f in fs:
            if f["fingerprint"] in seen:
                continue
            seen.add(f["fingerprint"])
            g = NG.with_analyses(f["config"]["graph"], ["probability"], [], f["statement"][:160], start=f"{year}-01-01", end=None)
            items.append({"label": f["config"].get("condition", f["statement"])[:140], "graph": g, "original": f["numbers"], "original_grade": f["evidence"], "finding": f["id"], "original_period": [f.get("period_start"), f.get("period_end")]})
            if len(items) >= BATCH_MAX:
                break
        syms = sorted({i["graph"]["core"]["symbol"] for i in items})
        held = [h for s in syms for h in (LG.current(sid, s), LG.current(sid, s + "@intraday")) if h and h["status"] in ("sealed", "used") and h["end"] >= f"{year}-01-01"]
        if held and not force:
            h = held[0]
            return [db.add_message(sid, "assistant", "graph", {"text": f"Re-testing after {year} needs the sessions from {max(h['start'], f'{year}-01-01')} to {h['end']}, which this notebook holds back as its final test period ({LG._pub(h)['label']}). "
                                                                       "Looking at them will be recorded, and the period can no longer confirm anything afterwards. Nothing has been run.",
                                                              "question": f"Which findings still hold after {year}?", "graph": {"question": f"Which findings still hold after {year}?", "nodes": [], "assumptions": [], "unsupported": []}, "data": [], "problems": [],
                                                              "finalized": False, "state": "awaiting", "runs": [], "pending": {"action": "still_hold", "year": year},
                                                              "choices": [{"id": "run", "label": "Use the final test period (recorded)"}, {"id": "cancel", "label": "Keep it untouched"}]})]
        return self._batch(sid, f"Which findings still hold from {year} on?", "still_hold", items,
                           f"Re-running {len(items)} recorded finding{'s' if len(items) != 1 else ''} on sessions from {year}-01-01 on. The conditions are exactly the stored ones; nothing is re-tuned.", include_holdout=bool(held))

    def _similar(self, sid: str, text: str) -> list[dict]:
        day = MEM.day_of(text)
        fs = [f for f in MEM.list_findings(sid) if f["kind"] == "pattern" and (f.get("config") or {}).get("graph", {}).get("core")]
        pool = [f for f in fs if day is None or f["created_at"][:10] == day.isoformat()]
        if not pool:
            when = f"on {day.isoformat()}" if day else "so far"
            latest = fs[0]["created_at"][:10] if fs else None
            return [self._say(sid, f"No finding was recorded in this notebook {when}." + (f" The most recent ones are from {latest}; say \"similar to our last finding\" to use those." if latest else " Ask a market question first; I will not invent one."))]
        base = sorted(pool, key=lambda f: (-f["rank"], -f["id"]))[0]
        g0 = base["config"]["graph"]
        sym = g0["core"]["symbol"]
        items = []
        for other in [s for s in ("SPY", "QQQ", "IWM", "DIA") if s != sym and s in set(self.store.stored_symbols())][:3]:
            try:
                items.append({"label": f"the same conditions on {other}", "graph": NG.retarget(g0, other)})
            except ValueError:
                continue
        mirror = _mirror(g0)
        if mirror is not None:
            items.append({"label": f"the mirror image on {sym} (opposite direction)", "graph": mirror})
        if not items:
            return [self._say(sid, "I could not form a similar configuration from that finding with the symbols stored here.")]
        return self._batch(sid, f"Similar to: {base['statement'][:120]}", "similar", items,
                           f"Starting from the finding recorded {base['created_at'][:10]} ({base['evidence']}): \"{base['statement'][:200]}\". Testing {len(items)} closely related configurations - the same conditions on other symbols"
                           + (" and the mirror image" if mirror is not None else "") + ". Each one is a new trial and is corrected for as such.")

    def _combine(self, sid: str, text: str) -> list[dict]:
        graphs = [g for g in MEM.graphs_of(sid) if g["graph"].get("core")]
        if len(graphs) < 2:
            return [self._say(sid, "Combining conditions needs at least two finished studies in this notebook; there " + ("is one" if graphs else "are none") + " so far.")]
        m = re.search(r"combine\s+(.+?)\s+with\s+(.+)", text, re.I)
        if not m:
            return [self._say(sid, "Tell me which condition from which study, for example: \"combine the opening-range condition from our first study with the volatility filter from the second\".")]
        a, b = m.group(1), m.group(2)
        ga, gb = MEM.pick_graph(graphs, a if re.search("|".join(MEM.ORD), a.lower()) else "first"), MEM.pick_graph(graphs, b if re.search("|".join(MEM.ORD), b.lower()) else "second")
        if ga is None or gb is None:
            return [self._say(sid, f"This notebook has {len(graphs)} studies; I could not find the ones you mean.")]
        ia, ib = NG.find_clauses(ga["graph"], a), NG.find_clauses(gb["graph"], b)
        if not ia or not ib:
            have = lambda g: "; ".join(c["label"] for c in g["graph"]["core"]["clauses"])  # noqa: E731
            return [self._say(sid, "I could not find " + ("that condition in the first of those studies" if not ia else "that filter in the second of those studies") + f". \"{ga['question'][:80]}\" has: {have(ga)}. \"{gb['question'][:80]}\" has: {have(gb)}.")]
        try:
            g = NG.combine(ga["graph"], ia, gb["graph"], ib, f"Combined: {'; '.join(ga['graph']['core']['clauses'][i]['label'] for i in ia)} + {'; '.join(gb['graph']['core']['clauses'][i]['label'] for i in ib)}")
        except ValueError as e:
            return [self._say(sid, str(e))]
        chk = G.validate(g)
        if chk["errors"]:
            return [self._say(sid, "Those conditions cannot be combined into a valid analysis: " + "; ".join(chk["errors"][:3]) + ".", "error")]
        return self._graph(sid, g, "research memory", f"Conditions taken from \"{ga['question'][:70]}\" and \"{gb['question'][:70]}\". This is a new configuration: it is recorded as a trial, and choosing it after seeing both results is selection.",
                           derived_from=[ga["run_id"], gb["run_id"]])

    # ------------------------------------------------------------------ registry
    def _focus(self, sid: str, version: dict | None) -> dict:
        """What the user is most plausibly asking about: the last graph, else the strategy in focus."""
        graphs = MEM.graphs_of(sid)
        if graphs:
            res = graphs[-1]["result"]
            prob = next((n["result"] for n in res["nodes"] if n.get("result") and n["result"].get("kind") in ("probability", "matched")), None)
            core = graphs[-1]["graph"].get("core") or {}
            if prob is not None:
                sig = {"symbol": prob["symbol"], "features": sorted({c.get("feature") or (c.get("expr") or "").replace(" ", "") for c in core.get("clauses") or []}),
                       "outcome": "|".join(str((core.get("outcome") or {}).get("params", {}).get(k)) for k in ("entry", "horizon", "direction")) + "|" + ("touch" if (core.get("outcome") or {}).get("op") == "target_stop" else "close")}
                return {"symbol": prob["symbol"], "key": prob.get("key"), "signature": sig, "label": prob["condition"], "run_id": graphs[-1]["run_id"]}
        with db.connect() as con:
            row = con.execute("SELECT symbol, key, label, signature FROM trials WHERE scope=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        if row:
            return {"symbol": row["symbol"].split("@")[0], "key": row["key"], "signature": RG._sig(row["signature"]), "label": row["label"]}
        return {"symbol": None, "key": None, "signature": None, "label": None}

    def _registry(self, sid: str, version: dict | None) -> list[dict]:
        f = self._focus(sid, version)
        if not f["symbol"]:
            return [self._say(sid, "Nothing has been tested in this notebook yet, so there is nothing to look up in the experiment registry.")]
        chk = RG.check(sid, f["symbol"], f["key"], f["signature"])
        return [db.add_message(sid, "assistant", "registry", {"text": " ".join(chk["sentences"]), "view": "registry", "check": chk, "about": f["label"], "run_id": f.get("run_id")})]

    # ------------------------------------------------------------------ finding -> strategy
    def _last_strategy_plan(self, sid: str) -> dict | None:
        for m in reversed(db.list_messages(sid)):
            if m["kind"] == "strategy_plan":
                return m
        return None

    def _strategy_from_finding(self, sid: str, anyway: bool = False, finding_id: int | None = None) -> list[dict]:
        from strata.research import hypothesis as HY
        f = MEM.get_finding(finding_id) if finding_id else MEM.strongest(sid)
        if f is None:
            allf = MEM.list_findings(sid)
            if not allf:
                return [self._say(sid, "This notebook has no recorded findings yet. A strategy is built from a finding, so ask a market question first.")]
            best = sorted(allf, key=lambda x: -x["rank"])[0]
            if not anyway:
                return [self._say(sid, f"No finding in this notebook is supported by its evidence. The strongest one on record is graded \"{best['evidence']}\": {best['statement']} "
                                       "Building an option strategy on it would be building on noise. Say \"build it anyway\" to draft structures on it regardless; they will be labelled as resting on an unsupported pattern.")]
            f = best
        cell = (f.get("config") or {}).get("cell")
        if not cell:
            why = "its observation starts at an intraday event, or one of its conditions is read at a different clock time from the entry" if f["kind"] != "study" else "it was stored as a market study without a reusable condition"
            return [self._say(sid, f"The strongest finding ({f['evidence']}) is: {f['statement']} It cannot be written as a strategy entry rule as it stands, because {why}. I will not substitute a loosely related rule.")]
        sym = f["symbol"]
        try:
            has_q = bool(self.store.option_days(sym, "quote"))
        except Exception:  # noqa: BLE001
            has_q = False
        try:
            has_t = not self.store.option_bar_index(sym).is_empty()
        except Exception:  # noqa: BLE001
            has_t = False
        variants, notes = HY.structure_variants(sym, cell["preds"], cell["outcome"], has_q, has_t)
        from strata.ai.draft import compile_draft
        pub = [{"label": v["label"], "items": compile_draft(v["draft"], self.store, self._spot(sym))["items"]} for v in variants]
        cover = self._option_cover(sym, f, sid)
        grade = ""
        if f["evidence"] == "suggestive":  # asked for a "supported" finding: say plainly that none reaches that grade
            grade = (" No finding in this notebook is graded \"supported\" or better; this one is \"suggestive\" (" + "; ".join(f.get("why") or ["it has an open weakness"]) + "). "
                     "The drafts exist so it can be tested, not because it is established.")
        txt = (f"From the strongest finding on record ({f['evidence']}): {f['statement']}{grade} {len(variants)} option structures express it. None has been priced: a pattern in the underlying is not a strategy result until "
               "its trades are evaluated on observed option prices." + (f" {cover['sentence']}" if cover else "") + (" This finding is NOT supported by its evidence; the drafts are exploratory." if f["evidence"] not in MEM.USABLE else ""))
        return [db.add_message(sid, "assistant", "strategy_plan", {"text": txt, "view": "strategy_plan", "finding": {k: f[k] for k in ("id", "evidence", "statement", "kind", "run_id", "why")}, "symbol": sym, "cell": cell,
                                                                  "variants": pub, "notes": notes, "cover": cover, "label": (f["config"].get("condition") or f["statement"])[:90], "launched": False})]

    def _option_cover(self, sym: str, f: dict, sid: str | None = None) -> dict | None:
        """How many of the finding's qualifying sessions have observed option prices stored."""
        res = _graph_of_run(f["run_id"]) if f.get("run_id") else None
        if not res or res.get("kind") != "graph":
            return None
        dates = sorted(MEM.sessions_of(res))
        if not dates:
            return None
        try:
            qd = {d.isoformat() if hasattr(d, "isoformat") else str(d) for d in self.store.option_days(sym, "quote")}
        except Exception:  # noqa: BLE001
            qd = set()
        try:
            idx = self.store.option_bar_index(sym)
            td = {str(x)[:10] for x in idx["date"].to_list()} if not idx.is_empty() and "date" in idx.columns else set()
        except Exception:  # noqa: BLE001
            td = set()
        nq, nt = sum(1 for d in dates if d in qd), sum(1 for d in dates if d in td)
        s = (f"Of the {len(dates)} qualifying sessions behind it ({dates[0]} to {dates[-1]}), {nq} have stored option bid/ask quotes and {nt} have stored option trade prints; the rest cannot be priced with the data stored and will be reported as not testable.")
        have = sorted(qd | td)
        if have and nq + nt == 0:
            # the option data and the sessions the finding was measured on do not overlap: say where a backtest CAN run, and what that period is
            s += f" Option prices are stored for {have[0]} to {have[-1]}, so the backtest runs on the condition's occurrences in that period instead."
            h = LG.current(sid, sym) if sid else None
            if h and h.get("start") and have[-1] >= h["start"]:
                s += (f" Those dates lie in the period this notebook holds back for its final test ({h['start']} onward). Pricing the strategy there is a look at that period: it is recorded, "
                      "and a result on it is a first test of the option strategy, not an independent confirmation of the price pattern.")
        return {"signals": len(dates), "with_quotes": nq, "with_trades": nt, "first": dates[0], "last": dates[-1], "sentence": s}

    def _backtest_plan(self, sid: str) -> list[dict]:
        m = self._last_strategy_plan(sid)
        if m is None:
            return [self._say(sid, "There is no drafted strategy to backtest yet. Ask me to build one from a finding first.")]
        c = m["content"]
        out = self._structures_for(sid, c["symbol"], c["cell"]["preds"], c["cell"]["outcome"], c["label"], c["finding"]["statement"][:200], None)
        db.update_message(m["id"], {**c, "launched": True})
        return out

    # ------------------------------------------------------------------ forward validation
    def _forward(self, sid: str, version: dict | None) -> list[dict]:
        from strata.research import forward as FW
        runs = [r for r in self._session_runs(sid) if r["kind"] == "backtest"]
        why = ""
        if runs:
            # which one: the strategy the notebook last examined closely (the target of the latest execution challenge), else the latest backtest
            r = runs[-1]
            for m in reversed(db.list_messages(sid)):
                c = m["content"] if isinstance(m["content"], dict) else {}
                if m["kind"] == "result" and c.get("view") in ("execution", "challenge") and c.get("target"):
                    tid = c["target"]["id"] if isinstance(c["target"], dict) else c["target"]
                    hit = next((x for x in runs if x["run_id"] == tid), None)
                    if hit:
                        r, why = hit, " It is the one the execution challenge was run on."
                    break
            others = len({x["label"] for x in runs}) - 1
            if others > 0:
                why += f" The other {others} backtested variant{'s' if others != 1 else ''} in this notebook {'were' if others != 1 else 'was'} not frozen; name one to freeze it too."
            try:
                exp = FW.create_from_run(r["run"], session_id=sid, name=r["label"])
            except ValueError as e:
                return [self._say(sid, str(e))]
            what = f"the strategy \"{r['label']}\""
        else:
            graphs = [g for g in MEM.graphs_of(sid) if g["graph"].get("core")]
            if not graphs:
                return [self._say(sid, "A forward-validation experiment freezes a finished configuration - a backtested strategy or a market study - and this notebook has neither yet.")]
            g = graphs[-1]
            exp = FW.create_from_graph(g["graph"], g["result"], session_id=sid, source_run_id=g["run_id"])
            what = f"the market study \"{g['question'][:90]}\""
        txt = (f"Frozen {what} as forward-validation experiment {exp['id']} (version {exp['version']}).{why} Its parameters can no longer be changed: an edit creates a new version and a new experiment. "
               f"{(exp.get('activation') or {}).get('rule', '')} Only sessions from {exp['first_session']} on count as forward evidence, whatever data is stored or downloaded later, and none exists yet - "
               "evaluate it again after the market has traded. Each session is sealed once in an append-only record. Paper only: nothing here can place an order.")
        return [db.add_message(sid, "assistant", "forward", {"text": txt, "view": "forward", "experiment": exp})]


def _mirror(graph: dict) -> dict | None:
    """The opposite-direction version of a single-clause move finding (a decline becomes a
    rise of the same size, an up outcome becomes a down outcome)."""
    core = NG.load_core(graph["core"])
    moves = [c for c in core.clauses if c.sign and c.op in ("<=", "<", ">=", ">") and c.value is not None]
    if not moves:
        return None
    for c in moves:
        c.op = {"<=": ">=", "<": ">", ">=": "<=", ">": "<"}[c.op]
        c.value = -c.value
        c.sign = -c.sign
        c.label = c.label.replace(" fell ", " ~ROSE~ ").replace(" rose ", " fell ").replace(" ~ROSE~ ", " rose ")
    d = core.outcome["params"].get("direction", "up")
    core.outcome["params"]["direction"] = "down" if d == "up" else "up"
    return NG.build(core, ["probability"], [], f"{graph.get('question', '')} (mirror image)", [], [])
