"""The AI research agent: conversation state, controlled tools and orchestration.

Flow for every user message:

    text -> interpreter (offline rules, or an LLM restricted to a fixed tool list)
         -> Draft (validated) -> compile_draft (engine configs + disclosed assumptions)
         -> assess (which research mode the stored data can actually answer)
         -> launch through the normal job system (same path as the REST API)
         -> finalize: facts read from the stored result -> grounded explanation

The language model never produces a result. Its only outputs are a Draft, a choice among
the tools below, or prose that is rejected if it contains a number the engine did not compute.
"""
from __future__ import annotations

import json
import re
from datetime import date

from strata.ai import explain as EX
from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, assess, backtest_config, compile_draft, diff_drafts, window_for
from strata.ai.llm import LLM, LLMError
from strata.api import db
from strata.api.launch import launch_backtest, launch_study
from strata.api.results import load_run_result
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine import expr as X
from strata.engine.models import BacktestConfig

MAX_VARIANTS = 12
MODE_LABEL = {"study": "Historical market study", "historical": "Historical options backtest", "modeled": "Modeled options backtest (estimated premiums)"}
TERMINAL = ("done", "error", "cancelled")

HELP = ("I couldn't turn that into a strategy, a market question or a change to the current one. Try, for example:\n"
        "- \"How often does SPY rise 1% from the open after falling 3% or more the previous trading day?\"\n"
        "- \"Buy a 30DTE ATM SPY call at the close after a 2% down day and hold for 5 days\"\n"
        "- then: \"change the previous-day decline to 2% and compare\", \"what if I bought puts instead?\", \"test 7DTE, 14DTE and 30DTE\", "
        "\"how sensitive is this to the target?\", \"show me every losing trade\", \"only since 2020\", \"test out of sample\".")


# ----------------------------------------------------------------------------- controlled tools
def tool_list_datasets(store: MarketStore) -> list[dict]:
    from strata.data.catalog import catalog
    out = []
    for c in catalog(store):
        d, o = c.get("daily"), c.get("options")
        out.append({"symbol": c["symbol"], "synthetic": bool(c["meta"].get("synthetic")), "daily": [d["first"], d["last"]] if d else None,
                    "intraday": {k: [v["first"], v["last"]] for k, v in (c.get("intraday") or {}).items()},
                    "options": {"first": o["first"], "last": o["last"], "resolution": o["resolution"], "quotes": o["quote_type"]} if o else None})
    return out


def tool_check_coverage(store: MarketStore, symbol: str) -> dict:
    sym = str(symbol).upper()
    return next((x for x in tool_list_datasets(store) if x["symbol"] == sym), {"symbol": sym, "stored": False, "note": "nothing is stored for this symbol; nothing will be simulated in its place"})


def tool_lookup_functions(query: str, limit: int = 14) -> list[dict]:
    q = str(query or "").lower().split()
    hits = [v for k, v in X.REGISTRY.items() if not q or any(w in k.lower() or w in v["description"].lower() or w in v["category"].lower() for w in q)]
    return [{"use": v["signature"], "meaning": v["description"]} for v in hits[:limit]]


def tool_validate_expression(expr: str) -> dict:
    errs = X.validate(str(expr))
    return {"valid": not errs, "errors": errs, "needs_intraday_bars": (not errs) and X.needs_intraday(str(expr))}


def registry_reference() -> str:
    by: dict[str, list[str]] = {}
    for v in X.REGISTRY.values():
        by.setdefault(v["category"], []).append(v["signature"])
    return "\n".join(f"  {cat}: " + ", ".join(sigs) for cat, sigs in by.items())


DRAFT_GUIDE = """Draft fields (omit or null anything the user did not say - the server discloses its own defaults):
  kind: "options" (a position in option contracts) | "study" (a question about how the underlying behaves)
  symbol, symbol_stated (false if the user named no ticker)
  entry_time_stated: the user's own words for the entry clock time incl. timezone, e.g. "6:30 AM Pacific", "at the open", "at the close"
  weekdays: ["Mon",...] only if the user restricted days
  conditions: [{"expr": <expression>, "text": <plain English>}]   (expression reference below)
  structure: long_call|long_put|short_put|short_call|bull_call_spread|bear_put_spread|bull_put_spread|bear_call_spread|iron_condor|iron_butterfly|long_straddle|short_straddle|long_strangle|short_strangle|covered_call
  dte: integer days to expiration (0 = same-day);  strike: {"method": "atm"|"delta"|"pct_otm", "value": number};  width: spread width in points;  quantity
  underlying_target_pct / underlying_stop_pct: exit when THE UNDERLYING moves this % for / against the position from the entry price
  premium_target_pct / premium_stop_pct: exit on a % gain / loss OF THE OPTION PREMIUM (100 = doubles). Never confuse the two.
  exit_time_stated: the user's words for a clock exit; hold_sessions: N trading days; exit_weekday: "Fri"; hold_to_expiration: bool
  study: {"target": {"kind":"pct","value":1}, "stop": {...}|null, "reference": "entry"|"open"|"prev_close", "measure": "touch"|"close", "direction": "up"|"down"|null,
          "horizon_kind": "session"|"time"|"sessions", "horizon_time": "HH:MM" New York, "horizon_sessions": N}
     target/stop value is signed (+1 = 1% above, -1 = 1% below); kind "expr" with "expr": a price level such as prev_high
  start, end: ISO dates only if the user limited the period;  mode: "modeled" ONLY if the user explicitly asked for theoretical/modeled premiums"""


def system_prompt(store: MarketStore, current: Draft | None) -> str:
    data = json.dumps(tool_list_datasets(store), separators=(",", ":"))[:1800]
    return f"""You are the request interpreter inside Strata, an options research platform. You translate what the user wants into ONE JSON action.
You never state results, probabilities, prices or opinions about profitability: a deterministic Python engine calculates every number after you.

Reply with exactly one JSON object: {{"tool": "<name>", "args": {{...}}}}

Read-only tools (the result is returned to you, then you reply again):
  list_datasets {{}} | check_coverage {{"symbol": "SPY"}} | lookup_functions {{"query": "gap"}} | validate_expression {{"expr": "..."}} | get_results {{}}
Final tools (each ends your turn):
  propose {{"draft": <Draft>, "compare": <bool>}}   a new request, or the FULL updated draft when the user changes the current one (compare=true if they asked to compare)
  sweep {{"field": "prev_ret|target|stop|dte|entry_time|premium_target|premium_stop|delta|slippage|gap|vix|hold_sessions", "values": [...]}}   run the current draft for up to {MAX_VARIANTS} values
  show_trades {{"filter": "all"|"win"|"loss"}} | out_of_sample {{}}
  answer {{"text": "..."}}   only for a clarifying question or a reply that needs no numbers

{DRAFT_GUIDE}

Condition expressions (Python-like comparisons joined with and/or/not; only these names and functions exist):
{registry_reference()}
Examples: previous_session.return_pct <= -3 | gap_pct <= -0.5 | price > vwap | consecutive_bearish_candles("5m") >= 3 | crosses_above(price, prev_close) | vix < 18 | return_pct("SPY", sessions=5) < -4
"The previous day fell 3%" means close-to-close (previous_session.return_pct) unless the user says open-to-close (previous_session.open_to_close_pct) or gap (gap_pct).

Stored data: {data}
Current draft: {current.model_dump_json(exclude_none=True) if current else "none"}"""


# ----------------------------------------------------------------------------- interpretation
def _fix_times(d: Draft, raw: dict) -> Draft:
    """Clock arithmetic is done here, not by the model: stated times are converted deterministically."""
    for field, stated_key in (("entry_time", "entry_time_stated"), ("exit_time", "exit_time_stated")):
        stated = getattr(d, stated_key)
        if stated:
            ts = NL.find_times(stated, None)
            if ts:
                val = ts[0]["et"]
                setattr(d, field, "open" if (val == "09:30" and field == "entry_time") else val)
                if ts[0].get("assumed_tz"):
                    d.tz_assumed = ts[0]["assumed_tz"]
                continue
        v = getattr(d, field)
        if v and not (v in ("open", "close") or re.fullmatch(r"\d{2}:\d{2}", v)):
            setattr(d, field, None)
    return d


def llm_interpret(llm: LLM, text: str, current: Draft | None, history: list[dict], store: MarketStore, results_facts) -> dict:
    """Tool loop against a model. Returns the same action shape as nl_rules.interpret (raises LLMError on failure)."""
    msgs = history[-6:] + [{"role": "user", "content": text}]
    system = system_prompt(store, current)
    repairs = 0
    for _ in range(7):
        obj = llm.json(system, msgs, max_tokens=2500)
        tool, args = obj.get("tool"), obj.get("args") if isinstance(obj.get("args"), dict) else {}
        if tool is None and "draft" in obj:  # tolerate a bare draft
            tool, args = "propose", {"draft": obj["draft"], "compare": bool(obj.get("compare"))}

        def back(result):
            msgs.append({"role": "assistant", "content": json.dumps(obj)[:3000]})
            msgs.append({"role": "user", "content": "TOOL RESULT: " + json.dumps(result, default=str)[:3500] + "\nReply with your next JSON action."})

        if tool == "list_datasets":
            back(tool_list_datasets(store))
        elif tool == "check_coverage":
            back(tool_check_coverage(store, args.get("symbol", "")))
        elif tool == "lookup_functions":
            back(tool_lookup_functions(args.get("query", "")))
        elif tool == "validate_expression":
            back(tool_validate_expression(args.get("expr", "")))
        elif tool == "get_results":
            back(results_facts() or {"note": "no finished run yet"})
        elif tool == "propose":
            errs: list[str] = []
            d = None
            try:
                raw = dict(args.get("draft") or {})
                raw.setdefault("title", text.strip()[:90])
                d = _fix_times(Draft.model_validate(raw), raw)
                for c in d.conditions:
                    errs += [f"condition {c.expr!r}: {e}" for e in X.validate(c.expr)]
                if d.kind == "options" and not d.structure:
                    errs.append("kind is 'options' but no structure was given; use kind 'study' for a question about the underlying")
                if d.study:
                    for lv in (d.study.target, d.study.stop):
                        if lv is not None and lv.kind == "expr":
                            errs += [f"level {lv.expr!r}: {e}" for e in X.validate(lv.expr or "")]
            except Exception as e:  # noqa: BLE001  (pydantic validation)
                errs.append(str(e)[:600])
            if errs:
                repairs += 1
                if repairs > 2:
                    raise LLMError("the model kept returning an invalid draft: " + "; ".join(errs)[:300])
                back({"error": "invalid draft - fix and call propose again", "problems": errs})
                continue
            if current is not None and (d.symbol == current.symbol or not d.symbol_stated) and (args.get("compare") or diff_drafts(current, d)) and _is_edit(current, d):
                changed = [x["field"] for x in diff_drafts(current, d)]
                return {"action": "modify", "draft": d, "changed": changed, "compare": bool(args.get("compare"))}
            return {"action": "new", "draft": d, "sweep": None}
        elif tool == "sweep":
            field, values = args.get("field"), args.get("values")
            if field not in NL.SWEEP_FIELDS or not isinstance(values, list) or not values:
                repairs += 1
                if repairs > 2:
                    raise LLMError("the model returned an invalid sweep")
                back({"error": f"field must be one of {sorted(NL.SWEEP_FIELDS)} and values a non-empty list"})
                continue
            return {"action": "sweep", "field": field, "values": values[:MAX_VARIANTS]}
        elif tool == "show_trades":
            return {"action": "show_trades", "filter": args.get("filter") if args.get("filter") in ("all", "win", "loss") else "all"}
        elif tool == "out_of_sample":
            return {"action": "oos"}
        elif tool == "answer" and isinstance(args.get("text"), str) and args["text"].strip():
            return {"action": "answer", "text": args["text"].strip()}
        else:
            repairs += 1
            if repairs > 2:
                raise LLMError("the model did not choose a valid tool")
            back({"error": "unknown tool; reply with one of the listed tools as {\"tool\": ..., \"args\": {...}}"})
    raise LLMError("the model did not reach a decision")


def _is_edit(cur: Draft, new: Draft) -> bool:
    """A proposal counts as an edit of the current draft when it keeps the same kind of question."""
    return cur.kind == new.kind and (cur.kind == "study" or cur.structure == new.structure or bool(set(c.expr for c in cur.conditions) & set(c.expr for c in new.conditions)) or cur.entry_time == new.entry_time)


# ----------------------------------------------------------------------------- the agent
class Agent:
    def __init__(self, jobs, store: MarketStore | None = None, llm_factory=None):
        self.jobs = jobs
        self.store = store or MarketStore()
        if llm_factory is None:
            from strata.ai.llm import active
            llm_factory = active
        self._llm_factory = llm_factory

    # ---- helpers
    def llm(self) -> tuple[LLM | None, str | None]:
        try:
            return self._llm_factory(), None
        except LLMError as e:
            return None, str(e)

    def _spot(self, sym: str) -> float | None:
        try:
            d = self.store.read_daily(sym)
            return float(d["close"][-1]) if not d.is_empty() else None
        except Exception:  # noqa: BLE001
            return None

    def _state(self, sid: str) -> tuple[dict, dict | None, Draft | None]:
        s = db.get_session(sid)
        if s is None:
            raise LookupError("session not found")
        st = s.get("state") if isinstance(s.get("state"), dict) else {}
        s["state"] = st
        v = db.get_version(st["current"]) if st.get("current") else None
        return s, v, (Draft.model_validate(v["draft"]) if v else None)

    def _say(self, sid: str, text: str, kind: str = "text", **extra) -> dict:
        return db.add_message(sid, "assistant", kind, {"text": text, **extra})

    def _history(self, sid: str) -> list[dict]:
        out = []
        for m in db.list_messages(sid)[-8:]:
            c = m["content"] if isinstance(m["content"], dict) else {}
            t = c.get("text") or ""
            if t:
                out.append({"role": m["role"], "content": t[:600]})
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    def _latest_runs(self, version: dict | None, kinds=("study", "historical", "modeled")) -> list[dict]:
        return [r for r in (version or {}).get("runs", []) if r.get("mode") in kinds]

    def _results_facts(self, version: dict | None):
        for r in reversed(self._latest_runs(version)):
            run = db.get_run(r["run_id"])
            if run and run["status"] == "done":
                try:
                    res = load_run_result(run)
                except FileNotFoundError:
                    continue
                facts = EX.study_facts(res) if run["kind"] == "study" else EX.backtest_facts(res)
                return {"run": r.get("label"), "mode": r["mode"], "facts": {f["label"]: f["shown"] for f in facts}}
        return None

    # ---- interpretation
    def interpret(self, sid: str, text: str, current: Draft | None, version: dict | None, last_field: str | None) -> tuple[dict, str, str | None]:
        known = list(self.store.symbols().keys())
        default_symbol = "SPY"
        llm, err = self.llm()
        note = f"Language model unavailable ({err}); used the built-in interpreter." if err else None
        if llm is not None:
            try:
                r = llm_interpret(llm, text, current, self._history(sid)[:-1], self.store, lambda: self._results_facts(version))
                return r, llm.name, None
            except LLMError as e:
                note = f"The language model ({llm.name}) did not return a usable interpretation ({e}); used the built-in interpreter instead."
        return NL.interpret(text, current, known, last_field, default_symbol), "built-in rules", note

    # ---- launching
    def _launch(self, d: Draft, compiled: dict, cover: dict, mode: str, label: str, overrides: dict | None = None) -> dict:
        if mode == "study":
            win = window_for("study", d, cover)
            if compiled.get("study") is None or win is None:
                raise ValueError((cover["modes"]["study"].get("reason")) or "the study is not runnable")
            cfg = {**compiled["study"], "start": win[0], "end": win[1], "label": label}
            out = launch_study(self.jobs, cfg, label)
        else:
            cfg = backtest_config(d, compiled, cover, mode)
            cfg["label"] = label
            for path, v in (overrides or {}).items():
                node = cfg
                ks = path.split(".")
                for k in ks[:-1]:
                    node = node[k]
                node[ks[-1]] = v
            out = launch_backtest(self.jobs, BacktestConfig.model_validate(cfg), label)
        return {"run_id": out["id"], "mode": mode, "label": label, "cached": out["cached"]}

    def _pick_mode(self, d: Draft, cover: dict) -> str | None:
        modes = cover["modes"]
        if d.kind == "study":
            return "study" if modes["study"]["ok"] else None
        if d.mode in ("modeled", "study") and modes[d.mode]["ok"]:
            return d.mode  # explicitly requested by the user
        return "historical" if modes["historical"]["ok"] else None

    def _propose(self, sid: str, d: Draft, parent: dict | None, compare: bool, interp: str, note: str | None, changed: list[str] | None = None, run: bool = True, label: str | None = None) -> list[dict]:
        s, cur_v, cur_d = self._state(sid)
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        cover = assess(d, compiled, self.store)
        diff = diff_drafts(Draft.model_validate(parent["draft"]), d) if parent else []
        vlabel = label or (("Changed " + ", ".join(x["field"].replace("study.", "").replace("_", " ") for x in diff[:3])) if parent and diff else (d.title or "Strategy")[:80])
        v = db.add_version(sid, parent["id"] if parent else None, vlabel, d.model_dump(mode="json"))
        st = {**s["state"], "current": v["id"]}
        db.update_session(sid, title=(d.title[:70] if not s["state"].get("current") and d.title else None), state=st)
        runs, compare_with, lines = [], [], []
        mode = self._pick_mode(d, cover) if not compiled["errors"] else None
        if compiled["errors"]:
            lines.append("I could not build this: " + "; ".join(compiled["errors"]))
        elif mode and run:
            try:
                rl = f"v{v['id']} - {vlabel}"[:90]
                runs.append(self._launch(d, compiled, cover, mode, rl))
                lines.append(f"Running: {MODE_LABEL[mode]}" + (f" ({cover['modes'][mode].get('note')})" if cover["modes"][mode].get("note") else "") + ".")
            except Exception as e:  # noqa: BLE001
                lines.append(f"Could not start the run: {e}")
        elif d.kind == "options":
            lines += cover["messages"] or [cover["modes"]["historical"].get("reason", "The historical options backtest cannot run with the stored data.")]
        else:
            lines += cover["messages"] or [cover["modes"]["study"].get("reason", "This study cannot run with the stored data.")]
        if compare and parent and runs:
            prev = [r for r in parent.get("runs", []) if r["mode"] == runs[0]["mode"]]
            if prev:
                compare_with.append({**prev[-1], "label": f"v{parent['id']} - previous"})
            else:
                try:
                    pd_ = Draft.model_validate(parent["draft"])
                    pc = compile_draft(pd_, self.store, self._spot(pd_.symbol))
                    pr = self._launch(pd_, pc, assess(pd_, pc, self.store), runs[0]["mode"], f"v{parent['id']} - previous")
                    db.set_version_runs(parent["id"], parent.get("runs", []) + [pr])
                    compare_with.append(pr)
                except Exception as e:  # noqa: BLE001
                    lines.append(f"The previous version could not be run in the same mode for comparison ({e}).")
        if runs:
            db.set_version_runs(v["id"], runs)
        if compiled["missing"]:
            lines.append("Not specified, so I assumed defaults you can edit: " + ", ".join(compiled["missing"]) + ".")
        if note:
            lines.append(note)
        head = ("Updated the strategy" if d.kind == "options" else "Updated the study") if parent else ("Here is how I read that strategy" if d.kind == "options" else "Here is how I read that question")
        content = {"text": head + ". " + " ".join(lines), "version_id": v["id"], "parent_id": parent["id"] if parent else None, "interpreter": interp, "kind": d.kind, "title": d.title,
                   "draft": d.model_dump(mode="json"), "items": compiled["items"], "assumptions": compiled["assumptions"], "missing": compiled["missing"], "errors": compiled["errors"],
                   "coverage": {"modes": cover["modes"], "recommended": cover["recommended"], "messages": cover["messages"], "profile": cover["profile"], "symbol": d.symbol},
                   "diff": diff, "changed": changed or [], "runs": runs, "compare_with": compare_with, "mode": mode, "finalized": not runs}
        return [db.add_message(sid, "assistant", "plan", content)]

    def _variants(self, sid: str, d: Draft, version: dict, variants: list[tuple[str, Draft, dict | None]], field: str, what: str, interp: str, note: str | None, mode_hint: str | None = None) -> list[dict]:
        prev_modes = [r["mode"] for r in version.get("runs", [])]
        runs, skipped = [], []
        for label, vd, overrides in variants[:MAX_VARIANTS]:
            compiled = compile_draft(vd, self.store, self._spot(vd.symbol))
            cover = assess(vd, compiled, self.store)
            mode = mode_hint or (prev_modes[-1] if prev_modes and cover["modes"].get(prev_modes[-1], {}).get("ok") else self._pick_mode(vd, cover))
            if compiled["errors"] or not mode or not cover["modes"][mode]["ok"]:
                skipped.append({"label": label, "reason": "; ".join(compiled["errors"]) or (cover["modes"].get(mode or "historical", {}).get("reason") or "not runnable with the stored data")})
                continue
            try:
                r = self._launch(vd, compiled, cover, mode, f"v{version['id']} {label}"[:90], overrides)
                runs.append({**r, "label": label, "draft": vd.model_dump(mode="json")})
            except Exception as e:  # noqa: BLE001
                skipped.append({"label": label, "reason": str(e)})
        st = {**self._state(sid)[0]["state"], "last_field": field}
        db.update_session(sid, state=st)
        if not runs:
            return [self._say(sid, f"None of the {what} variants can run: " + "; ".join(f"{x['label']}: {x['reason']}" for x in skipped[:3]), "error")]
        txt = f"Running {len(runs)} variants of {what}" + (f" as {MODE_LABEL[runs[0]['mode']].lower()}" if runs else "") + "." + (f" Skipped {len(skipped)}: " + "; ".join(f"{x['label']} ({x['reason'][:120]})" for x in skipped[:3]) if skipped else "")
        if note:
            txt += " " + note
        return [db.add_message(sid, "assistant", "sweep", {"text": txt, "version_id": version["id"], "field": field, "what": what, "runs": runs, "skipped": skipped, "interpreter": interp, "finalized": False})]

    def _sweep(self, sid: str, version: dict, d: Draft, field: str, values: list, interp: str, note: str | None) -> list[dict]:
        what = NL.SWEEP_FIELDS.get(field, field)
        variants = []
        for v in values[:MAX_VARIANTS]:
            if field == "slippage":
                variants.append((f"slippage ${float(v):.2f}", d, {"execution.slippage_per_contract": float(v)}))
            else:
                try:
                    lab = f"{v}" if field == "entry_time" else f"{v}DTE" if field == "dte" else f"{float(v):g}{'%' if field not in ('delta', 'vix', 'hold_sessions') else ''}"
                    variants.append((f"{what.split(' (')[0]} {lab}", NL.apply_value(d, field, v), None))
                except (TypeError, ValueError):
                    continue
        if field == "slippage" and d.kind != "options":
            return [self._say(sid, "Slippage only applies to option backtests; the current request is a market study.")]
        return self._variants(sid, d, version, variants, field, what, interp, note)

    def _oos(self, sid: str, version: dict, d: Draft, interp: str) -> list[dict]:
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        cover = assess(d, compiled, self.store)
        modes = [r["mode"] for r in version.get("runs", [])]
        mode = modes[-1] if modes else self._pick_mode(d, cover)
        win = window_for(mode, d, cover) if mode else None
        if not win:
            return [self._say(sid, "There is no runnable mode for this request yet, so there is nothing to validate out of sample.")]
        days = cal.trading_days(date.fromisoformat(win[0]), date.fromisoformat(win[1]))
        if len(days) < 120:
            return [self._say(sid, f"Only {len(days)} sessions are available ({win[0]} to {win[1]}); that is too short to split into in-sample and out-of-sample periods.")]
        cut = days[int(len(days) * 0.7)]
        a, b = d.model_copy(deep=True), d.model_copy(deep=True)
        a.start, a.end = days[0], days[int(len(days) * 0.7) - 1]
        b.start, b.end = cut, days[-1]
        note = (f"Split at {cut.isoformat()}: the first 70% of sessions versus the last 30%. This only tests robustness honestly if the rules were chosen without looking at the later period; "
                "every earlier run in this conversation used the full period, so treat agreement between the two as necessary, not sufficient.")
        return self._variants(sid, d, version, [(f"In-sample {a.start} to {a.end}", a, None), (f"Out-of-sample {b.start} to {b.end}", b, None)], "period", "the test period", interp, note, mode_hint=mode)

    # ---- public API
    def post(self, sid: str, text: str) -> list[dict]:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        s, version, cur = self._state(sid)
        out = [db.add_message(sid, "user", "text", {"text": text})]
        r, interp, note = self.interpret(sid, text, cur, version, s["state"].get("last_field"))
        act = r["action"]
        if act == "new":
            out += self._propose(sid, r["draft"], None, False, interp, note, run=not r.get("sweep"))
            if r.get("sweep"):
                _, v2, d2 = self._state(sid)
                out += self._sweep(sid, v2, d2, r["sweep"]["field"], r["sweep"]["values"], interp, None)
        elif act == "modify":
            if not diff_drafts(cur, r["draft"]):
                out.append(self._say(sid, "That is already how the current version is set up, so nothing changed."))
            else:
                out += self._propose(sid, r["draft"], version, bool(r.get("compare")), interp, note, r.get("changed"))
        elif cur is None and act in ("sweep", "show_trades", "oos"):
            out.append(self._say(sid, "There is no strategy or study in this conversation yet. Describe one first."))
        elif act == "sweep":
            out += self._sweep(sid, version, cur, r["field"], r["values"], interp, note)
        elif act == "oos":
            out += self._oos(sid, version, cur, interp)
        elif act == "show_trades":
            done = [x for x in reversed(self._latest_runs(version)) if (db.get_run(x["run_id"]) or {}).get("status") == "done"]
            if not done:
                out.append(self._say(sid, "No finished run for the current version yet - run it first."))
            else:
                x = done[0]
                noun = "sessions" if x["mode"] == "study" else "trades"
                which = {"loss": "losing " if x["mode"] != "study" else "missed ", "win": "winning " if x["mode"] != "study" else "successful ", "all": ""}[r["filter"]]
                out.append(db.add_message(sid, "assistant", "trades", {"text": f"Showing the {which}{noun} of {x['label']} below. Click a row to replay it.", "run_id": x["run_id"], "mode": x["mode"], "filter": r["filter"]}))
        elif act == "answer":
            facts = self._results_facts(version)
            bad = EX.ungrounded_numbers(r["text"], [{"value": float(n), "unit": "ratio"} for n in re.findall(r"\d+(?:\.\d+)?", json.dumps(facts or {}) + (cur.model_dump_json() if cur else ""))], [])
            if bad:
                out.append(self._say(sid, "I can only quote numbers the engine has calculated, and that reply would have included others. Ask me to run or compare something and I will report the measured result.", interpreter=interp))
            else:
                out.append(self._say(sid, r["text"], interpreter=interp))
        else:
            out.append(self._say(sid, HELP + (f"\n\n{note}" if note else "")))
        return out

    def edit(self, sid: str, draft: dict, run: bool = True) -> list[dict]:
        """Manual edit from the interpretation panel: a new version branched from the current one."""
        s, version, cur = self._state(sid)
        d = Draft.model_validate(draft)
        if cur is not None and not diff_drafts(cur, d):
            return [self._say(sid, "No fields changed.")]
        return self._propose(sid, d, version, False, "manual edit", None, run=run)

    def apply(self, sid: str, field: str, value) -> list[dict]:
        s, version, cur = self._state(sid)
        if cur is None:
            raise ValueError("nothing to modify yet")
        return self._propose(sid, NL.apply_value(cur, field, value), version, True, "sweep selection", None)

    def run(self, sid: str, version_id: int, mode: str) -> list[dict]:
        """Run a version in an explicitly chosen mode (e.g. the user opts in to modeled premiums)."""
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid or mode not in MODE_LABEL:
            raise LookupError("version not found")
        d = Draft.model_validate(v["draft"])
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        cover = assess(d, compiled, self.store)
        if compiled["errors"] or not cover["modes"][mode]["ok"]:
            return [self._say(sid, cover["modes"][mode].get("reason") or "; ".join(compiled["errors"]), "error")]
        r = self._launch(d, compiled, cover, mode, f"v{v['id']} - {v['label']} [{mode}]"[:90])
        db.set_version_runs(v["id"], v.get("runs", []) + [r])
        txt = f"Running: {MODE_LABEL[mode]}. {cover['modes'][mode].get('note') or ''}"
        if mode == "modeled":
            txt += " You opted in to estimated premiums: results will be labelled as modeled and are not observed option prices."
        if mode == "study" and d.kind == "options":
            txt += " This measures the underlying's move only - it is not the option position's profit or loss."
        return [db.add_message(sid, "assistant", "run", {"text": txt, "version_id": v["id"], "runs": [r], "mode": mode, "finalized": False})]

    def restore(self, sid: str, version_id: int) -> list[dict]:
        s, _, _ = self._state(sid)
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid:
            raise LookupError("version not found")
        db.update_session(sid, state={**s["state"], "current": v["id"]})
        return [self._say(sid, f"Restored version {v['id']} ({v['label']}). New changes will branch from it; later versions are kept.", "note", version_id=v["id"])]

    def finalize(self, sid: str) -> list[dict]:
        """Turn finished runs into result messages (explanation + comparison). Idempotent."""
        out = []
        llm, _ = self.llm()
        for m in db.list_messages(sid):
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] not in ("plan", "sweep", "run") or c.get("finalized") or not c.get("runs"):
                continue
            refs = list(c.get("compare_with") or []) + list(c["runs"])
            runs = {r["run_id"]: db.get_run(r["run_id"]) for r in refs}
            if any(x is None or x["status"] not in TERMINAL for x in runs.values()):
                continue
            entries, failed = [], []
            for r in refs:
                run = runs[r["run_id"]]
                if run["status"] != "done":
                    failed.append({"label": r["label"], "status": run["status"], "error": (run.get("error") or run.get("message") or "").split("\n")[0][:400]})
                    continue
                try:
                    entries.append({"label": r["label"], "run": run, "result": load_run_result(run), "mode": r["mode"]})
                except FileNotFoundError:
                    failed.append({"label": r["label"], "status": "error", "error": "result files are missing"})
            content: dict = {"source_message": m["id"], "version_id": c.get("version_id"), "failed": failed}
            main = [e for e in entries if e["run"]["id"] in {r["run_id"] for r in c["runs"]}]
            if m["kind"] in ("plan", "run") and main:
                e = main[-1]
                ex = EX.explain_run(e["run"], e["result"], llm)
                content.update({"view": "single", "run_id": e["run"]["id"], "mode": e["mode"], "label": e["label"], "text": ex["text"], "explanation_source": ex["source"], "explanation_rejected": ex["rejected"],
                                "facts": ex["facts"], "caveats": ex["caveats"]})
                if c.get("compare_with") and len(entries) >= 2:
                    table = EX.compare_table(entries)
                    content.update({"view": "compare", "table": table, "compare_text": EX.compare_text(table)})
            elif m["kind"] == "sweep" and entries:
                table = EX.compare_table(entries)
                content.update({"view": "sweep", "field": c.get("field"), "what": c.get("what"), "table": table, "text": EX.compare_text(table, c.get("what") or ""), "mode": entries[0]["mode"],
                                "explanation_source": "deterministic", "caveats": (EX.study_caveats(entries[0]["result"]) if entries[0]["run"]["kind"] == "study" else EX.backtest_caveats(entries[0]["result"]))[:4]})
            else:
                content.update({"view": "failed", "text": "The run did not finish: " + "; ".join(f"{f['label']}: {f['error'] or f['status']}" for f in failed[:4])})
            if failed and content.get("view") != "failed":
                content["text"] = (content.get("text") or "") + " Not completed: " + "; ".join(f"{f['label']} ({f['error'] or f['status']})" for f in failed[:4]) + "."
            out.append(db.add_message(sid, "assistant", "result", content))
            db.update_message(m["id"], {**c, "finalized": True})
        return out

    def payload(self, sid: str) -> dict:
        s, v, _ = self._state(sid)
        return {"session": s, "messages": db.list_messages(sid), "versions": db.list_versions(sid), "current_version": v["id"] if v else None}
