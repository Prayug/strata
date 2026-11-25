"""The AI research agent: conversation state, controlled tools and orchestration.

Flow for every user message:

    text -> interpreter (offline rules, or an LLM restricted to a fixed, schema-checked tool list)
         -> an action: a Draft, a change to the current draft, a research plan (several
            related drafts plus checks), an exploratory grid, or a request about results
         -> compile_draft (engine configs + disclosed assumptions + unrepresented parts)
         -> ask first if a detail is genuinely ambiguous or part of the request could not be
            represented; in "review" mode also wait for confirmation of the assumptions
         -> the research pipeline (coverage -> free data acquisition -> qualifying sessions
            -> selective option data -> simulation), submitted through the normal job system
         -> finalize: facts read from the stored result -> grounded explanation, or the
            options when the data cannot cover what was asked

The language model never produces a result. Its only outputs are a Draft, a choice among
the tools below (arguments validated against strict schemas), or prose that is rejected if
it contains a number the engine did not compute.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from strata.ai import explain as EX
from strata.ai import nl_rules as NL
from strata.ai import research as RS
from strata.ai.draft import Draft, build_spec, compile_draft, diff_drafts
from strata.ai.llm import LLM, LLMError
from strata.ai.nl_stateful import enrich
from strata.api import db
from strata.api.launch import launch_research
from strata.api.results import load_run_result, load_trades
from strata.data.store import MarketStore
from strata.engine import expr as X
from strata.research import hypothesis as HY
from strata.research import ledger as LG
from strata.research import reasoning as RQ
from strata.research import stats as ST

MAX_VARIANTS = 12
MODE_LABEL = {"study": "Historical market study", "historical": "Historical options backtest (bid/ask quotes)", "trades": "Historical options backtest (option trade prices)",
              "modeled": "Modeled options backtest (estimated premiums)", "auto": "Options backtest"}
TERMINAL = ("done", "error", "cancelled")

HELP = ("I couldn't turn that into a strategy, a market question or a change to the current one. Try, for example:\n"
        "- \"How often does SPY rise 1% from the open after falling 3% or more the previous trading day?\"\n"
        "- \"Wait until SPY drops 0.5% from the open, then buy calls when it recovers half of that decline\"\n"
        "- \"Find out whether SPY tends to reverse after large down days\" or \"Investigate mean reversion after unusually large SPY declines\"\n"
        "- then: \"compare that with 2% and 4% down days\", \"test ATM 0DTE calls on those days using option trade prices\", \"include commissions and conservative slippage\", "
        "\"show every skipped signal\", \"is this statistically significant?\", \"test out of sample\", \"export the report\".")


# ----------------------------------------------------------------------------- controlled tools
def tool_list_datasets(store: MarketStore) -> list[dict]:
    from strata.data.catalog import catalog
    out = []
    for c in catalog(store):
        d, o, t = c.get("daily"), c.get("options"), c.get("option_trades")
        out.append({"symbol": c["symbol"], "synthetic": bool(c["meta"].get("synthetic")), "daily": [d["first"], d["last"]] if d else None,
                    "intraday": {k: [v["first"], v["last"]] for k, v in (c.get("intraday") or {}).items()},
                    "option_quotes": {"first": o["first"], "last": o["last"], "resolution": o["resolution"], "provider": o["provider"]} if o else None,
                    "option_trades": {"first": t["first"], "last": t["last"], "provider": t["provider"], "sessions": t["sessions"], "selective": True} if t else None})
    return out


def tool_check_coverage(store: MarketStore, symbol: str) -> dict:
    sym = str(symbol).upper()
    return next((x for x in tool_list_datasets(store) if x["symbol"] == sym), {"symbol": sym, "stored": False, "note": "nothing is stored for this symbol yet; free daily history is downloaded automatically when a request needs it, and nothing is ever simulated"})


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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Strict):
    pass


class CoverageArgs(_Strict):
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9.]{0,7}$")


class LookupArgs(_Strict):
    query: str = Field("", max_length=80)


class ValidateArgs(_Strict):
    expr: str = Field(max_length=2000)


class ProposeArgs(_Strict):
    draft: dict
    compare: bool = False


class SweepArgs(_Strict):
    field: Literal["prev_ret", "target", "stop", "dte", "entry_time", "premium_target", "premium_stop", "delta", "slippage", "gap", "vix", "hold_sessions"]
    values: list[float | str] = Field(min_length=1, max_length=MAX_VARIANTS)


class ShowTradesArgs(_Strict):
    filter: Literal["all", "win", "loss"] = "all"


class BreakdownArgs(_Strict):
    by: Literal["weekday", "year"] = "weekday"


class VariantArg(_Strict):
    label: str = Field(max_length=80)
    draft: dict


class InvestigateArgs(_Strict):
    question: str = Field(max_length=400)
    hypothesis: str = Field("", max_length=400)
    method: str = Field("", max_length=600)
    measure: str = Field("", max_length=200)
    variants: list[VariantArg] = Field(min_length=2, max_length=8)


class DiscoverArgs(_Strict):
    hypothesis: str = Field(max_length=400)
    symbol: str = Field("SPY", pattern=r"^[A-Za-z][A-Za-z0-9.]{0,7}$")


class AnswerArgs(_Strict):
    text: str = Field(min_length=1, max_length=1500)


# name -> (argument schema, ends the turn)
TOOLS: dict[str, tuple[type[BaseModel], bool]] = {
    "list_datasets": (NoArgs, False), "check_coverage": (CoverageArgs, False), "lookup_functions": (LookupArgs, False), "validate_expression": (ValidateArgs, False), "get_results": (NoArgs, False),
    "propose": (ProposeArgs, True), "sweep": (SweepArgs, True), "investigate": (InvestigateArgs, True), "discover": (DiscoverArgs, True), "show_trades": (ShowTradesArgs, True),
    "out_of_sample": (NoArgs, True), "robustness": (NoArgs, True), "breakdown": (BreakdownArgs, True), "timing": (NoArgs, True), "skipped_signals": (NoArgs, True), "export_report": (NoArgs, True),
    "answer": (AnswerArgs, True),
}

ACTION_SCHEMA = {"type": "object", "properties": {"tool": {"type": "string", "enum": list(TOOLS)}, "args": {"type": "object"}}, "required": ["tool", "args"], "additionalProperties": False}

DRAFT_GUIDE = """Draft fields (omit or null anything the user did not say - the server discloses its own defaults):
  kind: "options" (a position in option contracts) | "study" (a question about how the underlying behaves)
  symbol, symbol_stated (false if the user named no ticker)
  entry_time_stated: the user's own words for the entry clock time incl. timezone, e.g. "6:30 AM Pacific", "at the open", "at the close"
  weekdays: ["Mon",...] only if the user restricted days
  conditions: [{"expr": <expression>, "text": <plain English>}]   static rules that must hold at the entry (expression reference below)
  sequence: [{"name": "drop", "expr": "from_open_pct <= -0.5", "text": "..."}, {"name": "recover", "expr": "price >= low_since(\\"drop\\") + 0.5 * (open - low_since(\\"drop\\"))", "text": "..."}]
     use a sequence whenever the request says things must happen IN ORDER ("wait until X, then Y", "after the first breakout fails and price reclaims VWAP").
     Each step is evaluated only after the previous one happened, on a later tick. Later steps read earlier ones with stage_price("name"), low_since("name"), high_since("name"), minutes_since("name").
     Never approximate an ordered request with a single static condition.
  adds: [{"expr": "price <= entry.underlying_price * 0.995", "text": "...", "quantity": 1, "max_adds": 1}]   scale into the position ("add another contract after a further 0.5% decline")
  exit_conditions: [{"expr": "price >= vwap", "text": "..."}]   close the position (every lot) when any of these holds; may read entry.*, position.*, group.*
  another symbol's values: TICKER.value, e.g. QQQ.high <= QQQ.prev_high ("QQQ has not broken its own high")
  unsupported: ["<the user's own words>"]   every part of the request you could NOT express with the fields and expressions here. Never drop such a part silently and never substitute something loosely related.
  structure: long_call|long_put|short_put|short_call|bull_call_spread|bear_put_spread|bull_put_spread|bear_call_spread|iron_condor|iron_butterfly|long_straddle|short_straddle|long_strangle|short_strangle|covered_call
  dte: integer days to expiration (0 = same-day);  strike: {"method": "atm"|"delta"|"pct_otm", "value": number};  width: spread width in points;  quantity
  underlying_target_pct / underlying_stop_pct: exit when THE UNDERLYING moves this % for / against the position from the entry price
  premium_target_pct / premium_stop_pct: exit on a % gain / loss OF THE OPTION PREMIUM (100 = doubles). Never confuse the two; if the user's words do not say which, leave both null and add nothing - the server asks.
  exit_time_stated: the user's words for a clock exit; hold_sessions: N trading days; exit_weekday: "Fri"; hold_to_expiration: bool
  costs: {"preset": "conservative"|"none"|null, "commission_per_contract": $, "slippage_per_contract": $ per share, "slippage_pct": % of the trade print}   only if the user mentioned costs
  mode: "trades" if the user asked for option TRADE prices, "historical" for bid/ask quotes, "modeled" ONLY if they explicitly asked for theoretical premiums, "study" for the underlying only; else null
  study: {"target": {"kind":"pct","value":1}, "stop": {...}|null, "reference": "entry"|"open"|"prev_close", "measure": "touch"|"close", "direction": "up"|"down"|null,
          "horizon_kind": "session"|"time"|"sessions", "horizon_time": "HH:MM" New York, "horizon_sessions": N}
     target/stop value is signed (+1 = 1% above, -1 = 1% below); kind "expr" with "expr": a price level such as prev_high
  start, end: ISO dates only if the user limited the period"""


def system_prompt(store: MarketStore, current: Draft | None) -> str:
    data = json.dumps(tool_list_datasets(store), separators=(",", ":"))[:2000]
    return f"""You are the request interpreter inside Strata, a quantitative research platform. You translate what the user wants into ONE JSON action.
You never state results, probabilities, prices or opinions about profitability: a deterministic Python engine calculates every number after you.
Missing data is not your concern: the server downloads free data automatically and tells the user what cannot be covered.

Reply with exactly one JSON object: {{"tool": "<name>", "args": {{...}}}}

Read-only tools (the result is returned to you, then you reply again):
  list_datasets {{}} | check_coverage {{"symbol": "SPY"}} | lookup_functions {{"query": "gap"}} | validate_expression {{"expr": "..."}} | get_results {{}}
Final tools (each ends your turn):
  propose {{"draft": <Draft>, "compare": <bool>}}   a new request, or the FULL updated draft when the user changes the current one (compare=true if they asked to compare)
  sweep {{"field": "prev_ret|target|stop|dte|entry_time|premium_target|premium_stop|delta|slippage|gap|vix|hold_sessions", "values": [...]}}   run the current draft for up to {MAX_VARIANTS} values
  investigate {{"question": "...", "hypothesis": "...", "method": "...", "measure": "...", "variants": [{{"label": "...", "draft": <Draft>}}, ...]}}
      a research question that needs several related configurations compared (2-8 variants), e.g. "does it tend to reverse after large down days" -> one study per threshold
  discover {{"hypothesis": "...", "symbol": "SPY"}}   open-ended exploration of a hypothesis over a parameter grid ("investigate mean reversion after large declines")
  show_trades {{"filter": "all"|"win"|"loss"}} | out_of_sample {{}} | robustness {{}} (is the result statistically meaningful?) | breakdown {{"by": "weekday"|"year"}} | timing {{}} (when the target is usually hit)
  skipped_signals {{}} (signals that did not become trades, with reasons) | export_report {{}}
  answer {{"text": "..."}}   only for a clarifying question or a reply that needs no numbers

{DRAFT_GUIDE}

Condition expressions (Python-like comparisons joined with and/or/not; only these names and functions exist):
{registry_reference()}
Examples: previous_session.return_pct <= -3 | gap_pct <= -0.5 | price > vwap | consecutive_bearish_candles("5m") >= 3 | crosses_above(price, prev_high) | vix < 18 | QQQ.high <= QQQ.prev_high
"The previous day fell 3%" means close-to-close (previous_session.return_pct) unless the user says open-to-close (previous_session.open_to_close_pct) or gap (gap_pct).
A "red" day closed below its open (previous_session.open_to_close_pct < 0); a "down" day closed below the prior close (previous_session.return_pct < 0).

Stored data: {data}
Current draft: {current.model_dump_json(exclude_none=True) if current else "none"}"""


# ----------------------------------------------------------------------------- interpretation
def _fix_times(d: Draft) -> Draft:
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


def draft_problems(raw: dict, title: str) -> tuple[Draft | None, list[str]]:
    """Validate a model-proposed draft against the schema and the expression whitelist."""
    errs: list[str] = []
    d = None
    try:
        raw = dict(raw or {})
        raw.setdefault("title", title[:90])
        d = _fix_times(Draft.model_validate(raw))
        for c in list(d.conditions) + list(d.exit_conditions):
            errs += [f"condition {c.expr!r}: {e}" for e in X.validate(c.expr)]
        for s in d.sequence:
            errs += [f"sequence step {s.name!r} ({s.expr!r}): {e}" for e in X.validate(s.expr)]
        for a in d.adds:
            errs += [f"add rule {a.expr!r}: {e}" for e in X.validate(a.expr)]
        if d.kind == "options" and not d.structure:
            errs.append("kind is 'options' but no structure was given; use kind 'study' for a question about the underlying")
        if d.study:
            for lv in (d.study.target, d.study.stop):
                if lv is not None and lv.kind == "expr":
                    errs += [f"level {lv.expr!r}: {e}" for e in X.validate(lv.expr or "")]
        if not errs:
            c = compile_draft(d)
            errs += [e for e in c["errors"] if "sequence" in e or "stage" in e or "ref(" in e]
    except ValidationError as e:
        errs.append("; ".join(f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors()[:6]))
    except Exception as e:  # noqa: BLE001
        errs.append(str(e)[:400])
    return d, errs


def llm_interpret(llm: LLM, text: str, current: Draft | None, history: list[dict], store: MarketStore, results_facts, cancelled=None) -> dict:
    """Tool loop against a model. Every tool call is validated against its schema; a call
    that fails validation is returned to the model with the reason (twice at most). Returns
    the same action shape as the rule interpreter, or raises LLMError."""
    msgs = history[-6:] + [{"role": "user", "content": text}]
    system = system_prompt(store, current)
    repairs = 0
    for _ in range(8):
        if cancelled and cancelled():
            raise LLMError("cancelled")
        obj = llm.json(system, msgs, max_tokens=3000, schema=ACTION_SCHEMA)
        tool, raw_args = obj.get("tool"), obj.get("args") if isinstance(obj.get("args"), dict) else {}
        if tool is None and "draft" in obj:  # tolerate a bare draft
            tool, raw_args = "propose", {"draft": obj["draft"], "compare": bool(obj.get("compare"))}

        def back(result):
            msgs.append({"role": "assistant", "content": json.dumps(obj)[:3500]})
            msgs.append({"role": "user", "content": "TOOL RESULT: " + json.dumps(result, default=str)[:3500] + "\nReply with your next JSON action."})

        def repair(problem) -> None:
            nonlocal repairs
            repairs += 1
            if repairs > 2:
                raise LLMError("the model kept returning an invalid action: " + str(problem)[:300])
            back({"error": "invalid action - fix it and reply again", "problems": problem})

        if tool not in TOOLS:
            repair(f"unknown tool {tool!r}; use one of {', '.join(TOOLS)}")
            continue
        try:
            args = TOOLS[tool][0].model_validate(raw_args)
        except ValidationError as e:
            repair([f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors()[:6]])
            continue
        if tool == "list_datasets":
            back(tool_list_datasets(store))
        elif tool == "check_coverage":
            back(tool_check_coverage(store, args.symbol))
        elif tool == "lookup_functions":
            back(tool_lookup_functions(args.query))
        elif tool == "validate_expression":
            back(tool_validate_expression(args.expr))
        elif tool == "get_results":
            back(results_facts() or {"note": "no finished run yet"})
        elif tool == "propose":
            d, errs = draft_problems(args.draft, text.strip())
            if errs:
                repair(errs)
                continue
            from strata.ai.nl_stateful import questions_for
            d.questions = d.questions or questions_for(text, d)
            if current is not None and (d.symbol == current.symbol or not d.symbol_stated) and (args.compare or diff_drafts(current, d)) and _is_edit(current, d):
                return {"action": "modify", "draft": d, "changed": [x["field"] for x in diff_drafts(current, d)], "compare": bool(args.compare)}
            return {"action": "new", "draft": d, "sweep": None}
        elif tool == "investigate":
            variants, errs = [], []
            for v in args.variants:
                d, e = draft_problems(v.draft, v.label)
                errs += [f"variant {v.label!r}: {x}" for x in e]
                if d is not None:
                    variants.append({"label": v.label, "draft": d})
            if errs:
                repair(errs[:8])
                continue
            return {"action": "investigate", "plan": {"question": args.question, "symbol": variants[0]["draft"].symbol, "hypothesis": args.hypothesis, "method": args.method, "measure": args.measure or "the outcome",
                                                      "variants": variants, "checks": ["baseline", "significance", "out_of_sample"], "interpretation_notes": []}}
        elif tool == "discover":
            spec = RS.discovery_spec(args.hypothesis, args.symbol.upper())
            if spec is None:
                repair("discover supports hypotheses about what follows large declines, large gains or opening gaps; use investigate with explicit variants for anything else")
                continue
            return {"action": "discover", "spec": spec}
        elif tool == "sweep":
            return {"action": "sweep", "field": args.field, "values": list(args.values)[:MAX_VARIANTS]}
        elif tool == "show_trades":
            return {"action": "show_trades", "filter": args.filter}
        elif tool == "out_of_sample":
            return {"action": "oos"}
        elif tool == "robustness":
            return {"action": "robustness"}
        elif tool == "breakdown":
            return {"action": "breakdown", "by": args.by, "focus": None}
        elif tool == "timing":
            return {"action": "timing"}
        elif tool == "skipped_signals":
            return {"action": "skipped"}
        elif tool == "export_report":
            return {"action": "export"}
        elif tool == "answer":
            return {"action": "answer", "text": args.text.strip()}
    raise LLMError("the model did not reach a decision")


def _is_edit(cur: Draft, new: Draft) -> bool:
    """A proposal counts as an edit of the current draft when it keeps the same kind of question."""
    return cur.kind == new.kind and (cur.kind == "study" or cur.structure == new.structure or bool(set(c.expr for c in cur.conditions) & set(c.expr for c in new.conditions)) or cur.entry_time == new.entry_time)


def is_complex(text: str) -> bool:
    """Requests whose meaning depends on order, state or relationships. If a language model
    was expected to read one of these and could not, the user is told the built-in reader
    was used instead (it is never a silent downgrade)."""
    return bool(re.search(r"\bthen\b|\bunless\b|\bonly if\b|\bafter (?:the|a) (?:first|second)\b|\bfurther\b|\badd (?:another|more)\b|\bwait (?:until|for)\b|\bfails?\b|\breclaims?\b|\bhasn'?t\b|\bwhile\b|\bprovided\b", text.lower()))


# ----------------------------------------------------------------------------- the agent
class Agent:
    def __init__(self, jobs, store: MarketStore | None = None, llm_factory=None):
        self.jobs = jobs
        self.store = store or MarketStore()
        if llm_factory is None:
            from strata.ai.llm import active
            llm_factory = active
        self._llm_factory = llm_factory
        import threading
        self._tl = threading.local()  # the research session a call is acting for (trials are recorded against it)

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
        self._tl.sid = sid
        st = s.get("state") if isinstance(s.get("state"), dict) else {}
        s["state"] = st
        v = db.get_version(st["current"]) if st.get("current") else None
        return s, v, (Draft.model_validate(v["draft"]) if v else None)

    def _set(self, sid: str, **kw) -> dict:
        s, _, _ = self._state(sid)
        st = {**s["state"], **kw}
        db.update_session(sid, state=st)
        return st

    def _trials(self, sid: str, add: int = 0) -> int:
        """Configurations tested in this research session (every one is counted, whatever its result)."""
        s, _, _ = self._state(sid)
        n = int(s["state"].get("trials") or 0) + add
        if add:
            db.update_session(sid, state={**s["state"], "trials": n})
        return max(n, LG.count(sid))  # the ledger also holds configurations evaluated inside plans, maps and stress runs

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

    def _done_runs(self, version: dict | None) -> list[dict]:
        out = []
        for r in (version or {}).get("runs", []):
            run = db.get_run(r["run_id"])
            if run and run["status"] == "done" and (run.get("summary") or {}).get("outcome", "completed") == "completed":
                out.append({**r, "run": run})
        return out

    def _latest(self, version: dict | None) -> tuple[dict, dict] | tuple[None, None]:
        for r in reversed(self._done_runs(version)):
            try:
                return r, load_run_result(r["run"])
            except FileNotFoundError:
                continue
        return None, None

    def _results_facts(self, version: dict | None):
        r, res = self._latest(version)
        if r is None:
            return None
        facts = EX.study_facts(res) if r["run"]["kind"] == "study" else EX.backtest_facts(res)
        return {"run": r.get("label"), "mode": r.get("mode"), "facts": {f["label"]: f["shown"] for f in facts}}

    # ---- interpretation
    def interpret(self, sid: str, text: str, current: Draft | None, version: dict | None, last_field: str | None) -> tuple[dict, str, str | None]:
        known = sorted(self.store.stored_symbols())
        from strata.api.settings import get_setting
        default_symbol = str(get_setting("data.default_symbol") or "SPY")
        llm, err = self.llm()
        note = None
        if err:
            note = f"The configured language model is unavailable ({err}), so the built-in interpreter read this request."
        if llm is not None:
            try:
                r = llm_interpret(llm, text, current, self._history(sid)[:-1], self.store, lambda: self._results_facts(version))
                return r, llm.name, None
            except LLMError as e:
                fb = str(get_setting("llm.fallback") or "rules")
                if fb == "none":
                    return {"action": "failed", "text": f"The language model ({llm.name}) could not interpret this request: {e}. Falling back to the built-in interpreter is turned off in Settings, so nothing was run."}, llm.name, None
                note = (f"The language model ({llm.name}) did not return a usable interpretation ({e}). The built-in interpreter read this request instead"
                        + (" - it understands fewer phrasings, so check the interpretation below before relying on it." if is_complex(text) else "."))
        return self._rules(text, current, known, last_field, default_symbol), "built-in rules", note

    def _rules(self, text: str, current: Draft | None, known: list[str], last_field: str | None, default_symbol: str) -> dict:
        tick = list(dict.fromkeys(NL.TICKERS + [s for s in known if s.isupper()]))
        low = text.lower().strip()
        if re.search(r"\bchallenge\b|play devil'?s advocate|what could be wrong|poke holes|how could this be misleading|red[- ]team|be skeptical|critique this", low):
            return {"action": "challenge"}
        if re.search(r"execution (?:uncertainty|stress|risk|assumptions)|stress[- ]test (?:the )?(?:execution|fills)|realistic (?:execution|fills)|worse (?:execution|fills)|missed fills|partial fills|order delay", low):
            return {"action": "stress"}
        if re.search(r"(?:test|try|compare|which|what) .*option (?:structures?|positions?|strateg)|option structures?|calls? (?:versus|vs\.?|or) puts? on (?:this|that|these|those|it)|price (?:this|that|it) with options", low):
            return {"action": "structures"}
        rq = RQ.detect(text)
        if rq and (current is not None or rq["tool"] in ("threshold_difference",)):
            return {"action": "reason", **rq}
        if current is not None:
            r = RS.detect_followup(text, current, tick)
            if r:
                return r
        # broad market questions become a bounded research plan (the Hypothesis Lab) when there is enough history to split
        if re.match(r"(?:please\s+)?(?:(?:investigate|research|study|examine|analy[sz]e|test|check|find out|determine|see)\s+(?:whether|if|how|what|why)|does|do|is|are|what|how|which|compare|hypothesis)\b", low) or "hypothesis" in low:
            sym = RS._sym(text, tick, default_symbol)
            plan = HY.build_plan(text, sym, self.store, known)
            if plan is not None:
                try:
                    enough = self.store.read_daily(sym).height >= 750
                except Exception:  # noqa: BLE001
                    enough = False
                if enough:
                    return {"action": "hypothesis", "plan": plan}
        fresh = RS.detect_fresh(text, tick, default_symbol)
        if fresh:
            return fresh
        if current is not None and not re.search(r"sensitiv|sweep|range of|from \$?\d.*to \$?\d", text.lower()):
            # "include commissions and conservative slippage", "use option trade prices": a change of assumptions, compared with before
            d2, ch = RS.followup_edits(text, current)
            if {"costs", "mode", "provider"} & set(ch):
                return {"action": "modify", "draft": d2, "changed": ch, "compare": "costs" in ch}
        r = NL.interpret(text, current, known, last_field, default_symbol)
        if r["action"] == "new":
            d = enrich(text, tick, default_symbol)
            return {"action": "new", "draft": d, "sweep": r.get("sweep")}
        if current is not None and r["action"] in ("modify", "unknown"):
            base = r["draft"] if r["action"] == "modify" else current
            d2, ch = RS.followup_edits(text, base)
            if ch:
                return {"action": "modify", "draft": d2, "changed": list(dict.fromkeys((r.get("changed") or []) + ch)), "compare": bool(r.get("compare")) or "costs" in ch or "mode" in ch}
        if r["action"] == "sweep" and current is not None and re.search(r"compare (?:that|this|it|these|those|the results?)\b", text.lower()):
            base = _current_value(current, r["field"])
            if base is not None and base not in r["values"]:
                r["values"] = sorted(set(r["values"]) | {base}, key=lambda x: (isinstance(x, str), x))[:MAX_VARIANTS]
        if r["action"] == "unknown" and current is None:
            d = enrich(text, tick, default_symbol)
            if d.sequence or d.adds or d.exit_conditions or d.conditions:
                return {"action": "new", "draft": d, "sweep": None}
        return r

    # ---- launching
    def _launch(self, d: Draft, compiled: dict, mode: str | None, label: str, interp: str = "", approvals: dict | None = None) -> dict:
        spec = build_spec(d, compiled, mode, approvals, label, interp)
        out = launch_research(self.jobs, spec, label)
        sid = getattr(self._tl, "sid", None)
        if sid:
            try:  # every configuration that is run is a trial of this research session, whatever its result turns out to be
                core = spec.get("study") if spec["kind"] == "study" else {"strategies": spec["backtest"].get("strategies"), "execution": spec["backtest"].get("execution"), "pricing": (spec["backtest"].get("pricing") or {}).get("mode")}
                st_ = spec.get("study") or {}
                # a study that can only be observed on intraday bars is tracked against the intraday holdout, everything else against the daily one
                intraday = spec["kind"] == "study" and (st_.get("entry_time", "open") not in ("open", "close") or bool(st_.get("sequence")) or (st_.get("horizon") or {}).get("kind") == "time")
                LG.record(sid, d.symbol + ("@intraday" if intraday else ""), [{"kind": d.carried.get("trial_kind") or ("study" if spec["kind"] == "study" else "backtest"), "key": d.carried.get("ledger_key") or LG.config_key(core),
                                                                                                      "label": label, "run_id": out["id"], "start": d.start, "end": d.end}])
            except Exception:  # noqa: BLE001  (bookkeeping must never block a run)
                pass
        return {"run_id": out["id"], "mode": spec["mode"], "label": label, "cached": out["cached"], "approvals": approvals or {}}

    def _blockers(self, d: Draft, compiled: dict, exec_mode: str) -> tuple[str | None, list[str]]:
        """Why a compiled draft may not run yet: (state, lines for the user)."""
        open_q = [q for q in d.questions if q.get("answered") is None]
        errors = [e for e in compiled["errors"] if not (open_q and "no option position was described" in e)]
        if errors:
            return "blocked", ["I could not build this: " + "; ".join(errors)]
        if open_q:
            return "awaiting_answer", [open_q[0]["text"] if len(open_q) == 1 else f"{len(open_q)} details need a quick answer before this can run."]
        if d.unsupported:
            return "awaiting_confirmation", ["Part of the request could not be represented: " + "; ".join(f"\"{u}\"" for u in d.unsupported) + ". Nothing was substituted for it. You can run the rest as shown, or rephrase that part."]
        if exec_mode == "review":
            n = len(compiled["assumptions"])
            return "awaiting_confirmation", [f"Review mode: check the interpretation" + (f" and the {n} assumption{'s' if n != 1 else ''} I made" if n else "") + ", then run it."]
        return None, []

    def _propose(self, sid: str, d: Draft, parent: dict | None, compare: bool, interp: str, note: str | None, changed: list[str] | None = None, run: bool = True, label: str | None = None, mode: str | None = None) -> list[dict]:
        s, _cur_v, _cur_d = self._state(sid)
        exec_mode = s["state"].get("exec_mode", "quick")
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        diff = diff_drafts(Draft.model_validate(parent["draft"]), d) if parent else []
        vlabel = label or (("Changed " + ", ".join(x["field"].replace("study.", "").replace("_", " ") for x in diff[:3])) if parent and diff else (d.title or "Strategy")[:80])
        v = db.add_version(sid, parent["id"] if parent else None, vlabel, d.model_dump(mode="json"))
        st = {**s["state"], "current": v["id"]}
        db.update_session(sid, title=(d.title[:70] if not s["state"].get("current") and d.title else None), state=st)
        state, lines = self._blockers(d, compiled, exec_mode)
        runs, compare_with = [], []
        if state is None and run:
            try:
                rl = f"v{v['id']} - {vlabel}"[:90]
                runs.append(self._launch(d, compiled, mode, rl, interp))
                self._trials(sid, 1)
                state = "running"
            except Exception as e:  # noqa: BLE001
                state, lines = "blocked", [f"Could not start the run: {e}"]
        elif state is None:
            state = "ready"
        if compare and parent and runs:
            prev = [r for r in parent.get("runs", []) if (db.get_run(r["run_id"]) or {}).get("status") == "done"]
            if prev:
                compare_with.append({**prev[-1], "label": f"v{parent['id']} - previous"})
            else:
                try:
                    pd_ = Draft.model_validate(parent["draft"])
                    pr = self._launch(pd_, compile_draft(pd_, self.store, self._spot(pd_.symbol)), None, f"v{parent['id']} - previous", interp)
                    db.set_version_runs(parent["id"], parent.get("runs", []) + [pr])
                    compare_with.append(pr)
                except Exception as e:  # noqa: BLE001
                    lines.append(f"The previous version could not be run for comparison ({e}).")
        if runs:
            db.set_version_runs(v["id"], runs)
        if compiled["missing"] and state in ("running", "ready"):
            lines.append("Not specified, so I assumed defaults you can edit: " + ", ".join(compiled["missing"]) + ".")
        if note:
            lines.append(note)
        head = ("Updated the strategy" if d.kind == "options" else "Updated the study") if parent else ("Here is how I read that strategy" if d.kind == "options" else "Here is how I read that question")
        content = self._plan_content(d, compiled, v, parent, interp, diff, changed, runs, compare_with, state, head + ". " + " ".join(lines), mode)
        if note:
            content["note"] = note  # shown on the card itself: how this run relates to what came before it
        return [db.add_message(sid, "assistant", "plan", content)]

    def _plan_content(self, d: Draft, compiled: dict, v: dict, parent: dict | None, interp: str, diff: list, changed: list | None, runs: list, compare_with: list, state: str, text: str, mode: str | None) -> dict:
        return {"text": text.strip(), "version_id": v["id"], "parent_id": parent["id"] if parent else None, "interpreter": interp, "kind": d.kind, "title": d.title, "state": state,
                "draft": d.model_dump(mode="json"), "items": compiled["items"], "assumptions": compiled["assumptions"], "missing": compiled["missing"], "errors": compiled["errors"],
                "unsupported": d.unsupported, "questions": d.questions, "notes": d.notes, "config": {"strategy": compiled.get("strategy"), "study": compiled.get("study")},
                "diff": diff, "changed": changed or [], "runs": runs, "compare_with": compare_with, "mode": mode or d.mode or ("study" if d.kind == "study" else None), "finalized": not runs}

    def _plan_message(self, sid: str, version_id: int) -> dict | None:
        for m in reversed(db.list_messages(sid)):
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] == "plan" and c.get("version_id") == version_id:
                return m
        return None

    def _advance(self, sid: str, v: dict, d: Draft, interp: str, confirmed: bool = False, mode: str | None = None) -> list[dict]:
        """Re-evaluate a version after an answer / confirmation and run it if nothing blocks."""
        s, _, _ = self._state(sid)
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        state, lines = self._blockers(d, compiled, "quick" if confirmed else s["state"].get("exec_mode", "quick"))
        if confirmed and state == "awaiting_confirmation":
            state, lines = None, []
        runs = list(v.get("runs", []))
        if state is None:
            try:
                r = self._launch(d, compiled, mode, f"v{v['id']} - {v['label']}"[:90], interp)
                runs.append(r)
                db.set_version_runs(v["id"], runs)
                self._trials(sid, 1)
                state, lines = "running", ["Running."]
            except Exception as e:  # noqa: BLE001
                state, lines = "blocked", [f"Could not start the run: {e}"]
        m = self._plan_message(sid, v["id"])
        if m is None:
            return [self._say(sid, " ".join(lines))]
        c = m["content"]
        new = self._plan_content(d, compiled, v, db.get_version(v["parent_id"]) if v.get("parent_id") else None, c.get("interpreter", interp), c.get("diff") or [], c.get("changed"),
                                 [r for r in runs if r["run_id"] not in {x["run_id"] for x in c.get("resolved_runs", [])}][-1:] if state == "running" else c.get("runs", []),
                                 c.get("compare_with") or [], state, (c.get("text", "").split(" Running.")[0] + (" " + " ".join(lines) if state != "running" else " Running.")).strip(), mode or c.get("mode"))
        new["finalized"] = state != "running"
        if c.get("note"):
            new["note"] = c["note"]
        db.update_message(m["id"], new)
        return [db.get_message(m["id"])]

    def _variants(self, sid: str, version: dict, variants: list[tuple[str, Draft, str | None]], field: str, what: str, interp: str, note: str | None, kind: str = "sweep", extra: dict | None = None) -> list[dict]:
        runs, skipped = [], []
        for label, vd, mode in variants[:MAX_VARIANTS]:
            compiled = compile_draft(vd, self.store, self._spot(vd.symbol))
            if compiled["errors"]:
                skipped.append({"label": label, "reason": "; ".join(compiled["errors"])})
                continue
            try:
                r = self._launch(vd, compiled, mode, f"v{version['id']} {label}"[:90], interp, {"reduced_window": True})
                runs.append({**r, "label": label, "draft": vd.model_dump(mode="json")})
            except Exception as e:  # noqa: BLE001
                skipped.append({"label": label, "reason": str(e)})
        self._set(sid, last_field=field)
        if not runs:
            return [self._say(sid, f"None of the {what} variants can run: " + "; ".join(f"{x['label']}: {x['reason']}" for x in skipped[:3]), "error")]
        self._trials(sid, len(runs))
        txt = f"Running {len(runs)} variants of {what}." + (f" Skipped {len(skipped)}: " + "; ".join(f"{x['label']} ({x['reason'][:120]})" for x in skipped[:3]) if skipped else "")
        if note:
            txt += " " + note
        return [db.add_message(sid, "assistant", kind, {"text": txt, "version_id": version["id"], "field": field, "what": what, "runs": runs, "skipped": skipped, "interpreter": interp, "finalized": False, **(extra or {})})]

    def _mode_of(self, version: dict, d: Draft) -> str | None:
        """Keep variants in the mode the current version actually ran in."""
        for r in reversed(self._done_runs(version)):
            m = (r["run"].get("summary") or {}).get("mode")
            if m:
                return m
        return d.mode

    def _keep_exit(self, base: Draft, new: Draft) -> Draft:
        """When only the expiration changes, keep the exit the base version actually used.

        A same-day exit that was assumed for a 0DTE position would otherwise turn into
        "hold until expiration" for a longer-dated one, and the comparison would mix two
        changes. The kept exit is shown as an assumption."""
        if base.kind != "options" or new.kind != "options" or new.dte == base.dte or new.dte in (None, 0):
            return new
        if new.exit_time or new.hold_sessions or new.exit_weekday or new.hold_to_expiration or new.exit_conditions:
            return new
        eff = next((a for a in compile_draft(base, self.store, self._spot(base.symbol))["assumptions"] if a["field"] == "exit_time"), None)
        if not eff:
            return new
        n = new.model_copy(deep=True)
        n.exit_time, n.exit_time_stated = str(eff["value"]), None
        n.carried = {**n.carried, "exit_time": "kept from the version this is compared with, so the two differ only in expiration (say \"hold to expiration\" to change it)"}
        return n

    def _sweep(self, sid: str, version: dict, d: Draft, field: str, values: list, interp: str, note: str | None) -> list[dict]:
        what = NL.SWEEP_FIELDS.get(field, field)
        mode = self._mode_of(version, d)
        variants = []
        for v in values[:MAX_VARIANTS]:
            if field == "slippage":
                vd = d.model_copy(deep=True)
                from strata.ai.draft import DraftCosts
                c = (vd.costs or DraftCosts()).model_copy()
                c.slippage_per_contract = float(v)
                vd.costs = c
                variants.append((f"slippage ${float(v):.2f}", vd, mode))
            else:
                try:
                    lab = f"{v}" if field == "entry_time" else f"{v}DTE" if field == "dte" else f"{float(v):g}{'%' if field not in ('delta', 'vix', 'hold_sessions') else ''}"
                    vd = NL.apply_value(d, field, v)
                    variants.append((f"{what.split(' (')[0]} {lab}", self._keep_exit(d, vd) if field == "dte" else vd, mode))
                except (TypeError, ValueError):
                    continue
        if field == "slippage" and d.kind != "options":
            return [self._say(sid, "Slippage only applies to option backtests; the current request is a market study.")]
        return self._variants(sid, version, variants, field, what, interp, note)

    def _oos(self, sid: str, version: dict, d: Draft, interp: str) -> list[dict]:
        r, res = self._latest(version)
        if r is None:
            return [self._say(sid, "There is no finished run for this request yet, so there is nothing to validate out of sample.")]
        a, b = (res["period"] if r["run"]["kind"] == "study" else [res["config"]["start"], res["config"]["end"]])
        if r["run"]["kind"] != "study":
            cov = res.get("coverage") or {}
            a, b = (cov.get("requested") or {}).get("start", a), (cov.get("requested") or {}).get("end", b)
        from strata.core import calendar as cal
        days = cal.trading_days(date.fromisoformat(a), date.fromisoformat(b))
        if len(days) < 120:
            return [self._say(sid, f"Only {len(days)} sessions are covered ({a} to {b}); that is too short to split into in-sample and out-of-sample periods.")]
        cut = days[int(len(days) * 0.7)]
        x, y = d.model_copy(deep=True), d.model_copy(deep=True)
        x.start, x.end = days[0], days[int(len(days) * 0.7) - 1]
        y.start, y.end = cut, days[-1]
        note = (f"Split at {cut.isoformat()}: the first 70% of sessions versus the last 30%. This only tests robustness honestly if the rules were chosen without looking at the later period; "
                "every earlier run in this conversation used the full period, so treat agreement between the two as necessary, not sufficient.")
        mode = self._mode_of(version, d)
        return self._variants(sid, version, [(f"In-sample {x.start} to {x.end}", x, mode), (f"Out-of-sample {y.start} to {y.end}", y, mode)], "period", "the test period", interp, note)

    def _investigate(self, sid: str, plan: dict, interp: str, note: str | None) -> list[dict]:
        variants = [(v["label"], v["draft"] if isinstance(v["draft"], Draft) else Draft.model_validate(v["draft"]), None) for v in plan["variants"]]
        first = variants[0][1]
        s, _, _ = self._state(sid)
        v = db.add_version(sid, None, ("Investigation: " + plan["question"])[:80], first.model_dump(mode="json"))
        db.update_session(sid, title=(plan["question"][:70] if not s["state"].get("current") else None), state={**s["state"], "current": v["id"]})
        pub = {k: plan.get(k) for k in ("question", "hypothesis", "method", "measure", "checks", "interpretation_notes", "symbol")}
        pub["variants"] = [{"label": lb, "items": compile_draft(d, self.store, self._spot(d.symbol))["items"], "kind": d.kind} for lb, d, _ in variants]
        out = self._variants(sid, v, variants, "investigation", "the research question", interp, note, kind="investigation", extra={"plan": pub})
        if out and out[0]["kind"] == "investigation":
            db.set_version_runs(v["id"], out[0]["content"]["runs"])
            c = out[0]["content"]
            c["text"] = (f"Research plan: {len(c['runs'])} related configurations will be run and compared with their baselines. " + (plan.get("method") or "") + (" " + note if note else "")).strip()
            db.update_message(out[0]["id"], c)
            out = [db.get_message(out[0]["id"])]
        return out

    def _discover(self, sid: str, spec: dict, interp: str, note: str | None) -> list[dict]:
        s, _, _ = self._state(sid)
        rid = db.create_run("discovery", ("Discovery: " + spec["question"])[:90], {"question": spec["question"], "symbol": spec["symbol"], "dims": spec["dims"], "configs": len(spec["configs"])})
        self.jobs.submit(rid, "discovery", {"spec": spec, "scope": sid})
        if not s["state"].get("current"):
            db.update_session(sid, title=spec["question"][:70])
        self._trials(sid, len(spec["configs"]))
        dims = "; ".join(f"{d['label']}: {', '.join(str(d.get('labels', {}).get(v, v)) + d.get('unit', '') for v in d['values'])}" for d in spec["dims"])
        txt = (f"Exploratory research: {len(spec['configs'])} configurations of this hypothesis will be evaluated ({dims}). History is split into an explore period, a validation period and a final test period "
               "that no selection step looks at. Every configuration is recorded, including the ones that show nothing." + (" " + note if note else ""))
        return [db.add_message(sid, "assistant", "discovery", {"text": txt, "runs": [{"run_id": rid, "mode": "discovery", "label": "discovery"}], "hypothesis": spec["hypothesis"], "question": spec["question"],
                                                               "dims": spec["dims"], "tested": len(spec["configs"]), "not_explored": spec.get("not_explored") or [], "interpreter": interp, "finalized": False})]

    # ---- the Hypothesis Lab
    def _hypothesis(self, sid: str, plan: dict, interp: str, note: str | None) -> list[dict]:
        s, _, _ = self._state(sid)
        data = HY.describe_data(plan, self.store)
        rid = db.create_run("hypothesis", ("Hypothesis: " + plan["question"])[:90], {"question": plan["question"], "symbol": plan["symbol"], "configs": len(plan["configs"]), "session": sid})
        self.jobs.submit(rid, "hypothesis", {"plan": plan, "scope": sid})
        if not s["state"].get("current"):
            db.update_session(sid, title=plan["question"][:70])
        hold = LG.state(sid, plan["symbol"])
        pub = {k: v for k, v in plan.items() if k != "configs"}
        pub["configs"] = [{k: c[k] for k in ("key", "family", "label", "tag", "x")} | {"outcome_text": HY.OutcomeSpec.load(c["outcome"]).text(), "basis": HY.basis_of(c)} for c in plan["configs"]]
        txt = (f"Research plan: {len(plan['configs'])} explicitly listed configurations will be evaluated against their baselines - no more, and all of them will be reported. "
               "Selection uses a training period, a validation period can only veto, and a final test period is scored once for whatever qualifies. "
               f"{hold['configurations']} configuration{'s have' if hold['configurations'] != 1 else ' has'} been tested in this session so far; significance is corrected for all of them plus these." + (" " + note if note else ""))
        return [db.add_message(sid, "assistant", "hypothesis", {"text": txt, "runs": [{"run_id": rid, "mode": "hypothesis", "label": "hypothesis"}], "plan": pub, "data": data, "question": plan["question"],
                                                                "hypothesis": plan.get("hypothesis"), "interpreter": interp, "finalized": False})]

    def select_hypothesis(self, sid: str, run_id: str, key: str, what: str = "study") -> list[dict]:
        """Take one configuration of a finished plan further: as a full market study, or as a
        set of option positions priced on observed premiums."""
        self._state(sid)
        run = db.get_run(run_id)
        if run is None or run["kind"] != "hypothesis" or run["status"] != "done":
            raise LookupError("plan result not found")
        from strata.api.results import run_dir
        res = json.loads((run_dir(run_id) / "hypothesis.json").read_text())
        row = next((r for r in res["configs"] if r["key"] == key), None)
        if row is None:
            raise LookupError("configuration not found in this plan")
        sym = res["symbol"]
        preds = [HY.Pred.load(p) for p in row["preds"]]
        oc = HY.OutcomeSpec.load(row["outcome"])
        if what == "options":
            return self._structures_for(sid, sym, row["preds"], row["outcome"], row["label"], f"{row['condition']} -> {row['outcome_text']}", key)
        d = self._cell_draft(sym, preds, oc, f"{sym}: {row['condition']}", {"ledger_key": key, "from_plan": run_id})
        out = [db.add_message(sid, "user", "text", {"text": f"Investigate further: {row['label']} ({row['outcome_text']})", "from_plan": run_id, "key": key})]
        return out + self._propose(sid, d, None, False, "research plan", "This is the same configuration as in the plan, run session by session over the whole period, so it is not counted as a new hypothesis.")

    @staticmethod
    def _cell_draft(sym: str, preds: list, oc, title: str, carried: dict | None = None, start=None, end=None) -> Draft:
        """A market-study draft that asks the session-by-session engine exactly what a feature-engine cell measured."""
        if oc.direction == "either":
            raise ValueError("a move in either direction has no single study; open its observations instead")
        from strata.ai.draft import DraftCondition, DraftLevel, DraftStudy
        sgn = 1.0 if oc.direction == "up" else -1.0
        hk = "sessions" if oc.horizon.startswith("sessions:") else "time" if oc.horizon.startswith("time:") else "session"
        return Draft(kind="study", symbol=sym, symbol_stated=True, title=title[:90], entry_time=oc.entry, conditions=[DraftCondition(expr=p.expr(sym, oc.entry), text=p.text(sym)) for p in preds],
                     study=DraftStudy(target=None if oc.target is None else DraftLevel(kind="pct", value=sgn * oc.target), stop=None if oc.stop is None else DraftLevel(kind="pct", value=-sgn * oc.stop), measure=oc.measure,
                                      direction=oc.direction, horizon_kind=hk, horizon_sessions=int(oc.horizon.split(":")[1]) if hk == "sessions" else 1, horizon_time=oc.horizon.split(":", 1)[1] if hk == "time" else "12:00"),
                     carried=dict(carried or {}), start=start, end=end)

    def propose_cell(self, sid: str, symbol: str, preds: list[dict], outcome: dict, label: str = "", prior_trials: int = 0, start=None, end=None) -> list[dict]:
        """Open one cell of a sensitivity map or encyclopedia row as a full study in a notebook.
        The cells looked at before choosing this one are carried over as trials: picking the
        most interesting cell of a map is a selection, and the notebook's statistics allow for it."""
        self._state(sid)
        sym = symbol.upper()
        ps = [HY.Pred.load(p) for p in preds]
        oc = HY.OutcomeSpec.load(outcome)
        from strata.research.features import cond_text
        cond = cond_text(ps, sym)
        d = self._cell_draft(sym, ps, oc, f"{sym}: {cond}", {"ledger_key": LG.config_key({"symbol": sym, "preds": [p.dump() for p in ps], "outcome": oc.dump()})}, start, end)
        out = [db.add_message(sid, "user", "text", {"text": label or f"{sym}: when {cond}, how often does this follow: {oc.text()}?"})]
        note = None
        if prior_trials > 1:
            LG.record(sid, sym, [{"kind": "sensitivity", "key": f"explorer-{i}", "label": "a cell viewed in the explorer before this one was chosen"} for i in range(int(min(prior_trials, 500)) - 1)])
            note = f"This was picked from a map of {prior_trials} cells, so {prior_trials} configurations are counted as tried in this notebook."
        return out + self._propose(sid, d, None, False, "explorer", note)

    def _structures_for(self, sid: str, sym: str, preds: list[dict], outcome: dict, label: str, what: str, key: str | None) -> list[dict]:
        try:
            has_q = bool(self.store.option_days(sym, "quote"))
        except Exception:  # noqa: BLE001
            has_q = False
        try:
            has_t = not self.store.option_bar_index(sym).is_empty()
        except Exception:  # noqa: BLE001
            has_t = False
        variants, notes = HY.structure_variants(sym, preds, outcome, has_q, has_t)
        for x in variants:
            x["draft"].carried = {**x["draft"].carried, "trial_kind": "option-structure"}  # the condition is fixed; only the option position varies
        first = variants[0]["draft"]
        s, _, _ = self._state(sid)
        v = db.add_version(sid, None, ("Option structures: " + label)[:80], first.model_dump(mode="json"))
        db.update_session(sid, state={**s["state"], "current": v["id"]})
        plan = {"question": f"Which option position, if any, would have profited from: {what}?", "symbol": sym, "stage": "options",
                "hypothesis": "If the pattern in the underlying is large enough to overcome option premiums, spreads and time decay, at least one of these positions shows a positive average on observed prices.",
                "method": f"{len(variants)} option positions entered on exactly the same qualifying sessions and exited on the same schedule, priced only on observed option prices. Sessions without stored option prices are reported as not testable.",
                "measure": "profit and loss per trade on observed option prices", "checks": ["sample", "costs", "paired"],
                "interpretation_notes": notes + ["This is stage two. Stage one established how the UNDERLYING behaved; a real pattern there can still lose money here."], "from_plan_key": key}
        pub = dict(plan)
        pub["variants"] = [{"label": x["label"], "items": compile_draft(x["draft"], self.store, self._spot(sym))["items"], "kind": "options"} for x in variants]
        out = [db.add_message(sid, "user", "text", {"text": f"Test option structures on: {label}"})]
        res = self._variants(sid, v, [(x["label"], x["draft"], None) for x in variants], "structure", "the option structure", "research plan", None, kind="investigation", extra={"plan": pub})
        if res and res[0]["kind"] == "investigation":
            db.set_version_runs(v["id"], res[0]["content"]["runs"])
            c = res[0]["content"]
            c["text"] = f"Stage two - options profitability: {len(c['runs'])} option positions on the same qualifying sessions, priced on observed premiums only. " + " ".join(notes[:1])
            db.update_message(res[0]["id"], c)
            res = [db.get_message(res[0]["id"])]
        return out + res

    def _structures(self, sid: str, version: dict | None, cur: Draft | None, interp: str) -> list[dict]:
        """Option structures for the study in focus (stage two of a hypothesis)."""
        if cur is None or cur.kind != "study" or cur.study is None:
            return [self._say(sid, "Option structures are tested on a market study: run a study (or pick a configuration of a research plan) first, then ask again.")]
        st = cur.study
        if cur.sequence or any(not re.fullmatch(r"[\w\s().,\"'<>=+\-*/%]+", c.expr) for c in cur.conditions):
            return [self._say(sid, "This study starts from an event sequence, which the option-structure comparison does not carry over yet. Describe the option position you want on it and it will run on the same signals.")]
        tgt = abs(st.target.value) if st.target is not None and st.target.kind == "pct" else None
        stp = abs(st.stop.value) if st.stop is not None and st.stop.kind == "pct" else None
        direction = st.direction or ("down" if st.target is not None and st.target.kind == "pct" and st.target.value < 0 else "up")
        hz = f"sessions:{st.horizon_sessions}" if st.horizon_kind == "sessions" else f"time:{st.horizon_time}" if st.horizon_kind == "time" else "session"
        outcome = {"entry": cur.entry_time or "open", "horizon": hz, "direction": direction, "target": tgt, "stop": stp, "measure": st.measure}
        # the study's own rule-language conditions are reused verbatim
        from strata.research.features import Pred
        variants, notes = HY.structure_variants(cur.symbol, [], outcome, True, True)
        for x in variants:
            x["draft"].conditions = [c.model_copy() for c in cur.conditions]
            x["draft"].weekdays = cur.weekdays
            x["draft"].start, x["draft"].end = cur.start, cur.end
            x["draft"].carried = {**x["draft"].carried, "trial_kind": "option-structure"}
        del Pred
        s, _, _ = self._state(sid)
        v = db.add_version(sid, version["id"] if version else None, ("Option structures: " + (cur.title or cur.symbol))[:80], variants[0]["draft"].model_dump(mode="json"))
        db.update_session(sid, state={**s["state"], "current": v["id"]})
        plan = {"question": f"Which option position, if any, would have profited from \"{cur.title or 'this study'}\"?", "symbol": cur.symbol, "stage": "options",
                "hypothesis": "If the pattern in the underlying is large enough to overcome option premiums, spreads and time decay, at least one of these positions shows a positive average on observed prices.",
                "method": f"{len(variants)} option positions entered on the study's qualifying sessions, priced only on observed option prices.", "measure": "profit and loss per trade on observed option prices",
                "checks": ["sample", "costs", "paired"], "interpretation_notes": notes + ["This is stage two. The study measured the UNDERLYING; a real pattern there can still lose money in options."]}
        pub = dict(plan)
        pub["variants"] = [{"label": x["label"], "items": compile_draft(x["draft"], self.store, self._spot(cur.symbol))["items"], "kind": "options"} for x in variants]
        res = self._variants(sid, v, [(x["label"], x["draft"], None) for x in variants], "structure", "the option structure", interp, None, kind="investigation", extra={"plan": pub})
        if res and res[0]["kind"] == "investigation":
            db.set_version_runs(v["id"], res[0]["content"]["runs"])
        return res

    # ---- analyses that need their own job (stress ladder, adversarial review)
    def _analysis_job(self, sid: str, version: dict | None, kind: str, run_id: str | None = None) -> list[dict]:
        if run_id:
            run = db.get_run(run_id)
            if run is None or run["status"] != "done":
                raise LookupError("run not found")
            target, label = run, run.get("name") or run_id
        else:
            r, _res, err = self._need_result(sid, version)
            if err:
                return err
            target, label = r["run"], r["label"]
        if kind == "stress" and target["kind"] == "study":
            return [self._say(sid, "A market study has no orders or fills, so there is no execution to stress. Test an option position on it first; the stress test then re-simulates that position under worse execution.")]
        trials = max(1, self._trials(sid))
        name = {"stress": "Execution stress: ", "challenge": "Challenge: "}[kind] + re.sub(r"^v[0-9]+ - ", "", label)
        rid = db.create_run(kind, name[:90], {"target": target["id"], "session": sid, "trials": trials})
        self.jobs.submit(rid, kind, {"target": target["id"], "scope": sid, "trials": trials})
        txt = {"stress": "Re-simulating this strategy under an explicit ladder of worse execution assumptions: slippage, order delay, older prices, minimum volume, participation limits and missed fills. Every row decides entries, exits and fills again.",
               "challenge": "Reviewing this result as a sceptic: looking for look-ahead, selection effects, thin samples, clustering, optimistic fills, fragile parameters and a fair benchmark, and running the tests that can settle each one."}[kind]
        return [db.add_message(sid, "assistant", kind, {"text": txt, "runs": [{"run_id": rid, "mode": kind, "label": name[:90]}], "target": target["id"], "target_label": label, "finalized": False})]

    def _session_runs(self, sid: str) -> list[dict]:
        """Every finished study / backtest of the session, oldest first."""
        seen, out = set(), []
        for m in db.list_messages(sid):
            c = m["content"] if isinstance(m["content"], dict) else {}
            for r in (c.get("runs") or []):
                if r["run_id"] in seen:
                    continue
                run = db.get_run(r["run_id"])
                if run and run["status"] == "done" and run["kind"] in ("study", "backtest") and (run.get("summary") or {}).get("outcome", "completed") == "completed":
                    seen.add(r["run_id"])
                    out.append({"run_id": r["run_id"], "label": re.sub(r"^v[0-9]+ - ", "", r.get("label") or run.get("name") or ""), "kind": run["kind"], "run": run})
        return out

    def _stress_of(self, run_id: str) -> dict | None:
        """The latest finished stress result that refers to a run, if any."""
        from strata.api.results import run_dir
        for kind, fn, key in (("stress", "execution.json", None), ("challenge", "challenge.json", "stress")):
            for r in db.runs_targeting(run_id, (kind,)):
                p = run_dir(r["id"]) / fn
                if r["status"] == "done" and p.exists():
                    d = json.loads(p.read_text())
                    d = d.get(key) if key else d
                    if d:
                        return d
        return None

    def _reason(self, sid: str, version: dict | None, r: dict, text: str) -> list[dict]:
        tool = r["tool"]
        runs = self._session_runs(sid)
        cur_r, cur_res = self._latest(version)
        if cur_r is None and runs:
            cur_r = runs[-1]
            cur_res = load_run_result(cur_r["run"])
        if cur_r is None:
            return [self._say(sid, "There is no finished result in this conversation to reason about yet. Run a study or a strategy first.")]
        run = cur_r["run"]
        is_study = run["kind"] == "study"
        trades = None if is_study else load_trades(run["id"])
        label = re.sub(r"^v[0-9]+ - ", "", cur_r.get("label") or "")
        ref = [run["id"]]
        if tool == "subperiod":
            out = RQ.subperiod(cur_res, trades, r.get("split") or "2020")
        elif tool == "threshold_difference":
            nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            studies = [x for x in runs if x["kind"] == "study"]
            pick = []
            for nmb in nums[:2]:
                hit = next((x for x in reversed(studies) if re.search(rf"(?<![\d.]){re.escape(nmb)}(?:\.0)?\s*%", x["label"]) and x not in pick), None)
                if hit:
                    pick.append(hit)
            if len(pick) < 2:
                pick = studies[-2:]
            if len(pick) < 2:
                return [self._say(sid, "That comparison needs two finished market studies in this conversation (for example the 2% and the 3% version). Run both, then ask again.")]
            ref = [pick[0]["run_id"], pick[1]["run_id"]]
            out = RQ.threshold_difference(load_run_result(pick[0]["run"]), load_run_result(pick[1]["run"]), pick[0]["label"], pick[1]["label"])
        elif tool == "cost_survival":
            out = RQ.cost_survival(cur_res, trades, None if is_study else self._stress_of(run["id"]))
        elif tool == "concentration":
            out = RQ.concentration(cur_res, trades)
        elif tool == "volatility_explained":
            out = RQ.volatility_explained(cur_res, trades, self.store)
        elif tool == "assumption_impact":
            out = RQ.assumption_impact(cur_res, trades, None if is_study else self._stress_of(run["id"]), self.store)
            if out.get("variants"):
                self._trials(sid, 0)
                LG.record(sid, cur_res.get("symbol") or "", [{"kind": "sensitivity", "key": LG.config_key({"of": run["id"], "variant": row["change"]}), "label": row["change"], "run_id": run["id"]} for row in out["tables"][0]["rows"]])
            if out.get("needs") == "stress":
                return [self._say(sid, out["answer"])] + self._analysis_job(sid, version, "stress", run["id"])
        elif tool == "suspicious":
            out = RQ.suspicious(cur_res, trades)
        elif tool == "benchmarks":
            out = RQ.benchmarks(cur_res, trades)
        else:  # options_vs_underlying
            st_ = next((x for x in reversed(runs) if x["kind"] == "study"), None)
            bt_ = next((x for x in reversed(runs) if x["kind"] == "backtest"), None)
            if st_ is None or bt_ is None:
                return [self._say(sid, "Explaining why an option position differs from the underlying's behaviour needs both a market study and an option backtest of the same idea in this conversation.")]
            ref = [st_["run_id"], bt_["run_id"]]
            out = RQ.options_vs_underlying(load_run_result(st_["run"]), load_run_result(bt_["run"]), load_trades(bt_["run_id"]))
        return [db.add_message(sid, "assistant", "result", {"view": "analysis", "tool": tool, "run_id": ref[0], "run_ids": ref, "label": label, "version_id": version["id"] if version else None, "mode": "study" if is_study else (run.get("summary") or {}).get("mode"),
                                                            "text": out["answer"], "analysis": out, "explanation_source": "deterministic", "caveats": out.get("limits") or []})]

    # ---- analyses of an existing result (no new runs)
    def _need_result(self, sid: str, version: dict | None):
        r, res = self._latest(version)
        if r is None:
            return None, None, [self._say(sid, "There is no finished run for the current request yet. Run it first, then ask again.")]
        return r, res, None

    def _robustness(self, sid: str, version: dict) -> list[dict]:
        r, res, err = self._need_result(sid, version)
        if err:
            return err
        trials = max(1, self._trials(sid))
        rob = ST.study_robustness(res, trials) if r["run"]["kind"] == "study" else ST.backtest_robustness(res, load_trades(r["run_id"]), trials)
        lines = [rob["verdict"]] + [f"{'Passed' if c['ok'] else 'Not passed'} - {c['name']}: {c['detail']}." for c in rob["checks"]]
        if trials > 1:
            lines.append(f"{trials} configurations have been tested in this session; the significance threshold is adjusted for that.")
        return [db.add_message(sid, "assistant", "result", {"view": "robustness", "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "mode": (r["run"].get("summary") or {}).get("mode"),
                                                            "text": " ".join(lines), "robustness": rob, "explanation_source": "deterministic"})]

    def _breakdown(self, sid: str, version: dict, by: str, focus: str | None) -> list[dict]:
        r, res, err = self._need_result(sid, version)
        if err:
            return err
        study = r["run"]["kind"] == "study"
        if by == "weekday":
            if study:
                rows = [{**x, "hit": 1.0 if x["outcome"] == "target" else 0.0} for x in res.get("sessions", []) if x["outcome"] != "ambiguous"]
                bd = ST.weekday_breakdown(rows, "hit", lambda x: date.fromisoformat(x["date"]).weekday())
                bd["value"], bd["unit"] = "frequency of the outcome", "prob"
            else:
                bd = ST.weekday_breakdown(load_trades(r["run_id"]), "pnl", lambda x: int(x["entry_weekday"]))
                bd["value"], bd["unit"] = "average profit per trade", "usd"
            fmt = (lambda v: "n/a" if v is None else f"{v * 100:.1f}%") if study else (lambda v: "n/a" if v is None else f"{v:+,.2f}")
            sig = [x for x in bd["rows"] if x["adjusted_p_value"] is not None and x["adjusted_p_value"] < 0.05 and x["n"] >= 10]
            parts = ["By day of the week: " + "; ".join(f"{x['day']} {fmt(x['mean'])} ({x['n']})" for x in bd["rows"]) + "."]
            if focus:
                f = next(x for x in bd["rows"] if x["day"] == focus)
                parts.append(f"{focus}: {fmt(f['mean'])} over {f['n']} {'sessions' if study else 'trades'} versus {fmt(f['rest_mean'])} on other days"
                             + (f" (p = {f['p_value']:.3f}, {f['adjusted_p_value']:.3f} after adjusting for five comparisons)." if f["p_value"] is not None else " - too few observations to test."))
            parts.append((f"{', '.join(x['day'] for x in sig)} differ{'s' if len(sig) == 1 else ''} from the other days beyond what chance would produce." if sig
                          else "No day differs from the others by more than chance would produce."))
            return [db.add_message(sid, "assistant", "result", {"view": "breakdown", "by": "weekday", "focus": focus, "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "text": " ".join(parts),
                                                                "breakdown": bd, "explanation_source": "deterministic"})]
        years = res.get("by_year") if study else (res.get("breakdowns") or {}).get("year")
        if not years:
            return [self._say(sid, "This result has no per-year breakdown.")]
        if study:
            txt = "By year (frequency of the outcome, sessions): " + "; ".join("%s %s of %d" % (y["year"], "n/a" if y["probability"] is None else "%.0f%%" % (y["probability"] * 100), y["n"]) for y in years[-14:]) + "."
        else:
            txt = "By year (net P&L, trades): " + "; ".join("{} {:+,.0f} ({})".format(y["key"], y["pnl"], y["trades"]) for y in years[-14:]) + "."
        return [db.add_message(sid, "assistant", "result", {"view": "breakdown", "by": "year", "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "text": txt, "breakdown": {"rows": years}, "explanation_source": "deterministic"})]

    def _timing(self, sid: str, version: dict) -> list[dict]:
        r, res, err = self._need_result(sid, version)
        if err:
            return err
        if r["run"]["kind"] != "study":
            return [self._say(sid, "Time-to-target is measured by market studies of the underlying. For this backtest, open the trade log: every trade shows its entry and exit times.")]
        rows = [x for x in res.get("sessions", []) if x["outcome"] == "target" and x.get("target_minute") is not None]
        hits = sum(1 for x in res.get("sessions", []) if x["outcome"] == "target")
        if not rows:
            mix = res["stats"].get("resolution_mix") or {}
            return [self._say(sid, f"The target was reached on {hits} qualifying sessions, but none of them has intraday bars ({', '.join(f'{v} with {k} bars' for k, v in mix.items())}), so the time of day is unknown. "
                                   "A daily bar shows that a level was touched, not when. Intraday history for those dates is not available from a free source.")]
        mins = sorted(x["target_minute"] for x in rows)
        med = mins[len(mins) // 2]
        buckets = [("09:30-10:00", 570, 600), ("10:00-11:00", 600, 660), ("11:00-12:00", 660, 720), ("12:00-14:00", 720, 840), ("14:00-16:00", 840, 961)]
        dist = [{"window": n, "hits": sum(1 for m in mins if a <= m < b)} for n, a, b in buckets]
        first_hour = sum(1 for m in mins if m < 630)
        fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
        txt = (f"The time is known for {len(rows)} of the {hits} sessions on which the target was reached (the others only have daily bars). On those, the median time was {fmt(med)} ET; "
               f"{first_hour} of {len(rows)} were reached within the first hour. By window: " + "; ".join(f"{d['window']} {d['hits']}" for d in dist) + "."
               + (" With so few timed sessions this is indicative only." if len(rows) < 30 else ""))
        return [db.add_message(sid, "assistant", "result", {"view": "timing", "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "text": txt, "explanation_source": "deterministic",
                                                            "timing": {"timed": len(rows), "hits": hits, "median_minute": med, "distribution": dist, "sessions": [{"date": x["date"], "minute": x["target_minute"], "resolution": x["resolution"]} for x in rows][-400:]}})]

    def _skipped(self, sid: str, version: dict) -> list[dict]:
        r, res, err = self._need_result(sid, version)
        if err:
            return err
        if r["run"]["kind"] == "study":
            ex = res.get("excluded") or []
            n = sum(x["count"] for x in ex)
            txt = (f"A market study has no orders to skip. {n} session{'s were' if n != 1 else ' was'} excluded because they could not be evaluated: " + "; ".join(f"{x['reason']} ({x['count']})" for x in ex[:5]) + "."
                   if ex else "A market study has no orders to skip, and no session was excluded for missing data.")
            return [db.add_message(sid, "assistant", "result", {"view": "skipped", "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "text": txt, "skipped": {"count": n, "rows": [], "excluded": ex}, "explanation_source": "deterministic"})]
        sk = (res.get("coverage") or {}).get("skipped_signals") or {"count": 0, "rows": [], "by_kind": {}}
        names = {"no_option_data": "no observed option prices for the session", "no_recent_trade": "no recent trade print for the contract", "contract_not_stored": "the required contract is not in the stored data",
                 "no_quote": "no option quote within the allowed delay", "no_contract": "no contract matched the selection rules", "capital": "not enough capital", "rejected": "order rejected", "not_taken": "a position was already open"}
        by = sk.get("by_kind") or {}
        ex = (res.get("diagnostics") or {}).get("excluded_sessions") or {}
        n_ex = sum(len(v) for v in ex.values())
        txt = (f"{sk['count']} qualifying session{'s' if sk['count'] != 1 else ''} did not become a trade: " + "; ".join(f"{names.get(k, k)} ({v})" for k, v in sorted(by.items(), key=lambda kv: -kv[1])) + "."
               if sk["count"] else "Every qualifying session became a trade.")
        txt += " These are data and execution limits; none of them is counted as a losing or winning trade."
        if n_ex:
            txt += f" Separately, {n_ex} session{'s' if n_ex != 1 else ''} could not be evaluated at all (missing inputs) and so produced no signal either way."
        return [db.add_message(sid, "assistant", "result", {"view": "skipped", "run_id": r["run_id"], "version_id": version["id"], "label": r["label"], "text": txt, "explanation_source": "deterministic",
                                                            "skipped": {**sk, "names": names, "excluded": [{"reason": k, "count": len(v), "dates": v[-12:]} for k, v in ex.items()]}})]

    def _export(self, sid: str, version: dict | None) -> list[dict]:
        r, _res, err = self._need_result(sid, version)
        if err:
            return err
        rid = r["run_id"]
        links = [{"label": "Research report (HTML, self-contained)", "href": f"/api/ai/sessions/{sid}/report?format=html&run={rid}", "file": f"strata-report-{rid}.html"},
                 {"label": "Research report (Markdown)", "href": f"/api/ai/sessions/{sid}/report?format=md&run={rid}", "file": f"strata-report-{rid}.md"},
                 {"label": "Everything as JSON (results, coverage, configuration)", "href": f"/api/ai/sessions/{sid}/report?format=json&run={rid}", "file": f"strata-report-{rid}.json"},
                 {"label": "Observations (CSV): " + ("qualifying sessions" if r["run"]["kind"] == "study" else "trade log"), "href": f"/api/runs/{rid}/export?what=observations", "file": f"strata-{rid}-observations.csv"},
                 {"label": "Reproducible configuration (JSON)", "href": f"/api/runs/{rid}/export?what=config", "file": f"strata-{rid}-config.json"}]
        if r["run"]["kind"] != "study":
            links.insert(4, {"label": "Skipped signals (CSV)", "href": f"/api/runs/{rid}/export?what=skipped", "file": f"strata-{rid}-skipped.csv"})
        return [db.add_message(sid, "assistant", "result", {"view": "export", "run_id": rid, "version_id": version["id"], "label": r["label"], "links": links, "explanation_source": "deterministic",
                                                            "text": "The report for the current result is ready to download. It contains the question, the interpretation and assumptions, data sources and coverage, every statistic with its observations, the limitations and the configuration needed to reproduce it."})]

    # ---- public API
    def post(self, sid: str, text: str) -> list[dict]:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        s, version, cur = self._state(sid)
        out = [db.add_message(sid, "user", "text", {"text": text})]
        r, interp, note = self.interpret(sid, text, cur, version, s["state"].get("last_field"))
        act = r["action"]
        needs_cur = act in ("sweep", "show_trades", "oos", "robustness", "breakdown", "timing", "skipped", "export", "derive")
        if act == "failed":
            out.append(self._say(sid, r["text"], "error", interpreter=interp))
        elif act == "new":
            out += self._propose(sid, r["draft"], None, False, interp, note, run=not r.get("sweep"))
            if r.get("sweep"):
                _, v2, d2 = self._state(sid)
                out += self._sweep(sid, v2, d2, r["sweep"]["field"], r["sweep"]["values"], interp, None)
        elif act == "hypothesis":
            out += self._hypothesis(sid, r["plan"], interp, note)
        elif act == "reason":
            out += self._reason(sid, version, r, text)
        elif act == "challenge":
            out += self._analysis_job(sid, version, "challenge")
        elif act == "stress":
            out += self._analysis_job(sid, version, "stress")
        elif act == "structures":
            out += self._structures(sid, version, cur, interp)
        elif act == "investigate":
            out += self._investigate(sid, r["plan"], interp, note)
        elif act == "discover":
            out += self._discover(sid, r["spec"], interp, note)
        elif needs_cur and cur is None:
            out.append(self._say(sid, "There is no strategy or study in this conversation yet. Describe one first."))
        elif act in ("modify", "derive"):
            if act == "modify" and not diff_drafts(cur, r["draft"]):
                out.append(self._say(sid, "That is already how the current version is set up, so nothing changed."))
            else:
                nd = self._keep_exit(cur, r["draft"]) if act == "modify" else r["draft"]
                out += self._propose(sid, nd, version, bool(r.get("compare")), interp, note, r.get("changed"))
        elif act == "sweep":
            out += self._sweep(sid, version, cur, r["field"], r["values"], interp, note)
        elif act == "oos":
            out += self._oos(sid, version, cur, interp)
        elif act == "robustness":
            out += self._robustness(sid, version)
        elif act == "breakdown":
            out += self._breakdown(sid, version, r.get("by", "weekday"), r.get("focus"))
        elif act == "timing":
            out += self._timing(sid, version)
        elif act == "skipped":
            out += self._skipped(sid, version)
        elif act == "export":
            out += self._export(sid, version)
        elif act == "show_trades":
            done = self._done_runs(version)
            if not done:
                out.append(self._say(sid, "No finished run for the current version yet - run it first."))
            else:
                x = done[-1]
                study = x["run"]["kind"] == "study"
                noun = "sessions" if study else "trades"
                which = {"loss": "missed " if study else "losing ", "win": "successful " if study else "winning ", "all": "qualifying " if study else ""}[r["filter"]]
                out.append(db.add_message(sid, "assistant", "trades", {"text": f"Showing the {which}{noun} of \"{re.sub(r'^v[0-9]+ - ', '', x['label'])}\" in the workspace. Click a row to replay that day.", "run_id": x["run_id"], "mode": "study" if study else x.get("mode"), "filter": r["filter"]}))
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

    def set_mode(self, sid: str, exec_mode: str) -> dict:
        if exec_mode not in ("quick", "review"):
            raise ValueError("mode must be quick or review")
        return self._set(sid, exec_mode=exec_mode)

    def answer(self, sid: str, version_id: int, question_id: str, option: int) -> list[dict]:
        """The user picked one reading of an ambiguous detail."""
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid:
            raise LookupError("version not found")
        d = Draft.model_validate(v["draft"])
        q = next((x for x in d.questions if x["id"] == question_id), None)
        if q is None or not (0 <= option < len(q["options"])):
            raise LookupError("question not found")
        for k, val in q["options"][option]["patch"].items():
            setattr(d, k, val)
        q["answered"] = option
        d = Draft.model_validate(d.model_dump(mode="json"))
        db.update_version(version_id, draft=d.model_dump(mode="json"))
        v = db.get_version(version_id)
        m = self._plan_message(sid, version_id)
        return self._advance(sid, v, d, (m["content"].get("interpreter") if m else "") or "")

    def confirm(self, sid: str, version_id: int, mode: str | None = None) -> list[dict]:
        """Run a version that was waiting for confirmation (review mode, or an unrepresented part)."""
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid:
            raise LookupError("version not found")
        d = Draft.model_validate(v["draft"])
        m = self._plan_message(sid, version_id)
        return self._advance(sid, v, d, (m["content"].get("interpreter") if m else "") or "", confirmed=True, mode=mode)

    def edit(self, sid: str, draft: dict, run: bool = True) -> list[dict]:
        """Manual edit from the interpretation panel: a new version branched from the current one."""
        s, version, cur = self._state(sid)
        d = Draft.model_validate(draft)
        errs = [e for c in list(d.conditions) + list(d.exit_conditions) for e in X.validate(c.expr)] + [e for st in d.sequence for e in X.validate(st.expr)] + [e for a in d.adds for e in X.validate(a.expr)]
        if errs:
            return [self._say(sid, "That edit contains a rule I cannot evaluate: " + "; ".join(errs[:3]), "error")]
        if cur is not None and not diff_drafts(cur, d):
            return [self._say(sid, "No fields changed.")]
        d.questions = [q for q in d.questions if q.get("answered") is None and not _settled(q, cur, d)]
        return self._propose(sid, d, version, False, "manual edit", None, run=run)

    def apply(self, sid: str, field: str, value) -> list[dict]:
        s, version, cur = self._state(sid)
        if cur is None:
            raise ValueError("nothing to modify yet")
        return self._propose(sid, NL.apply_value(cur, field, value), version, True, "sweep selection", None)

    def run(self, sid: str, version_id: int, mode: str, approvals: dict | None = None) -> list[dict]:
        """Run a version in an explicitly chosen mode (e.g. the user opts in to modeled premiums)."""
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid or mode not in MODE_LABEL:
            raise LookupError("version not found")
        self._tl.sid = sid
        d = Draft.model_validate(v["draft"])
        compiled = compile_draft(d, self.store, self._spot(d.symbol))
        if compiled["errors"]:
            return [self._say(sid, "; ".join(compiled["errors"]), "error")]
        try:
            r = self._launch(d, compiled, mode, f"v{v['id']} - {v['label']} [{mode}]"[:90], "", approvals)
        except Exception as e:  # noqa: BLE001
            return [self._say(sid, f"Could not start the run: {e}", "error")]
        db.set_version_runs(v["id"], v.get("runs", []) + [r])
        self._trials(sid, 1)
        txt = f"Running: {MODE_LABEL[mode]}."
        if mode == "modeled":
            txt += " You opted in to estimated premiums: results will be labelled as modeled and are not observed option prices."
        if mode == "study" and d.kind == "options":
            txt += " This measures the underlying's move only - it is not the option position's profit or loss."
        if mode == "trades":
            txt += " Fills use observed option trade prints (no bid/ask exists in that data); a print is only used while it is recent."
        return [db.add_message(sid, "assistant", "run", {"text": txt, "version_id": v["id"], "runs": [r], "mode": mode, "finalized": False})]

    def decide(self, sid: str, message_id: int, choice: str) -> list[dict]:
        """Act on one of the options offered when the data could not cover a request."""
        m = db.get_message(message_id)
        if m is None or m["session_id"] != sid or m["kind"] != "decision":
            raise LookupError("decision not found")
        c = m["content"]
        ch = next((x for x in c.get("choices", []) if x["id"] == choice), None)
        if ch is None:
            raise LookupError("choice not found")
        prev = dict(c.get("approvals") or {})
        mode = c.get("mode") or "auto"
        out: list[dict] = []
        if choice == "reduced_window":
            out = self.run(sid, c["version_id"], mode, {**prev, "reduced_window": True})
        elif choice == "approve_download":
            out = self.run(sid, c["version_id"], mode, {**prev, "large_download": True})
        elif choice.startswith("mode:"):
            out = self.run(sid, c["version_id"], choice.split(":", 1)[1], {k: v for k, v in prev.items() if k != "reduced_window"})
        elif choice == "connect_alpaca":
            out = [self._say(sid, "A free Alpaca account provides the missing history. Enter its key ID and secret here: they go to this computer's credential store, both APIs are tested, and this run starts again and downloads only what it needs.", "note", action="connect_alpaca",
                             version_id=c["version_id"], mode=mode)]
        else:
            out = [self._say(sid, ch.get("detail") or ch["label"], "note")]
        db.update_message(message_id, {**c, "resolved": choice})
        return out

    def restore(self, sid: str, version_id: int) -> list[dict]:
        s, _, _ = self._state(sid)
        v = db.get_version(version_id)
        if v is None or v["session_id"] != sid:
            raise LookupError("version not found")
        db.update_session(sid, state={**s["state"], "current": v["id"]})
        return [self._say(sid, f"Restored version {v['id']} ({v['label']}). New changes will branch from it; later versions are kept.", "note", version_id=v["id"])]

    def pin(self, sid: str, run_id: str, on: bool = True) -> dict:
        s, _, _ = self._state(sid)
        pins = [x for x in s["state"].get("pinned", []) if x != run_id]
        if on:
            pins.append(run_id)
        return self._set(sid, pinned=pins[-8:])

    def finalize(self, sid: str) -> list[dict]:
        """Turn finished runs into result messages (explanation, comparison, findings, or the
        options when coverage fell short). Idempotent."""
        out = []
        llm, _ = self.llm()
        for m in db.list_messages(sid):
            c = m["content"] if isinstance(m["content"], dict) else {}
            if m["kind"] not in ("plan", "sweep", "run", "investigation", "discovery", "hypothesis", "stress", "challenge") or c.get("finalized") or not c.get("runs"):
                continue
            refs = list(c.get("compare_with") or []) + list(c["runs"])
            runs = {r["run_id"]: db.get_run(r["run_id"]) for r in refs}
            if any(x is None or x["status"] not in TERMINAL for x in runs.values()):
                continue
            if m["kind"] in ("hypothesis", "stress", "challenge"):
                out.append(db.add_message(sid, "assistant", "result", self._finalize_analysis(m["kind"], c, runs[c["runs"][0]["run_id"]], m["id"])))
                db.update_message(m["id"], {**c, "finalized": True})
                continue
            entries, failed, decisions = [], [], []
            for r in refs:
                run = runs[r["run_id"]]
                summ = run.get("summary") or {}
                if run["status"] != "done":
                    failed.append({"label": r["label"], "status": run["status"], "error": (run.get("error") or run.get("message") or "").split("\n")[0][:400]})
                    continue
                if summ.get("outcome") == "needs_decision":
                    decisions.append({"ref": r, "run": run, "decision": summ.get("decision") or {}})
                    continue
                try:
                    entries.append({"label": r["label"], "run": run, "result": load_run_result(run), "mode": summ.get("mode") or r.get("mode")})
                except FileNotFoundError:
                    failed.append({"label": r["label"], "status": "error", "error": "result files are missing"})
            content: dict = {"source_message": m["id"], "version_id": c.get("version_id"), "failed": failed}
            main_ids = {r["run_id"] for r in c["runs"]}
            main = [e for e in entries if e["run"]["id"] in main_ids]
            if m["kind"] == "discovery":
                if main:
                    res = main[0]["result"]
                    content.update(_discovery_view(res, main[0]["run"]["id"]))
                else:
                    content.update({"view": "failed", "text": "The exploration did not finish: " + "; ".join(f"{f['error'] or f['status']}" for f in failed[:3])})
            elif m["kind"] in ("plan", "run"):
                dec = [x for x in decisions if x["run"]["id"] in main_ids]
                if dec:
                    d0 = dec[-1]
                    out.append(db.add_message(sid, "assistant", "decision", {
                        "text": d0["decision"].get("reason", "The stored and freely available data cannot cover this request."), "choices": d0["decision"].get("choices", []), "run_id": d0["run"]["id"],
                        "version_id": c.get("version_id"), "mode": d0["ref"].get("mode"), "approvals": d0["ref"].get("approvals") or {}, "coverage": d0["decision"].get("coverage"),
                        "unavailable": d0["decision"].get("unavailable") or (d0["decision"].get("coverage") or {}).get("unavailable") or [], "source_message": m["id"]}))
                    db.update_message(m["id"], {**c, "finalized": True, "state": "needs_decision" if m["kind"] == "plan" else c.get("state")})
                    continue
                if main:
                    e = main[-1]
                    ex = EX.explain_run(e["run"], e["result"], llm)
                    cov = e["result"].get("coverage") or {}
                    content.update({"view": "single", "run_id": e["run"]["id"], "mode": e["mode"], "label": e["label"], "text": ex["text"], "explanation_source": ex["source"], "explanation_rejected": ex["rejected"],
                                    "facts": ex["facts"], "caveats": ex["caveats"], "coverage": _coverage_brief(cov), "mode_label": cov.get("mode_label"), "mode_reason": cov.get("mode_reason")})
                    if c.get("compare_with") and len(entries) >= 2:
                        table = EX.compare_table(entries)
                        content.update({"view": "compare", "table": table, "compare_text": EX.compare_text(table), "compatibility": compatibility(entries)})
                else:
                    content.update({"view": "failed", "text": "The run did not finish: " + "; ".join(f"{f['label']}: {f['error'] or f['status']}" for f in failed[:4])})
            elif m["kind"] == "investigation":
                for x in decisions:
                    failed.append({"label": x["ref"]["label"], "status": "not covered", "error": (x["decision"].get("reason") or "")[:300]})
                if entries:
                    rep = RS.investigation_report(c.get("plan") or {}, entries, max(1, self._trials(sid)))
                    table = EX.compare_table(entries)
                    content.update({"view": "investigation", "plan": c.get("plan"), "report": rep, "table": table, "text": rep["verdict"] + " " + " ".join(rep["findings"][:6]), "mode": entries[0]["mode"],
                                    "explanation_source": "deterministic", "caveats": rep["limitations"], "compatibility": compatibility(entries)})
                else:
                    content.update({"view": "failed", "text": "None of the planned configurations could be run: " + "; ".join(f"{f['label']}: {f['error'] or f['status']}" for f in failed[:4])})
            elif entries:
                for x in decisions:
                    failed.append({"label": x["ref"]["label"], "status": "not covered", "error": (x["decision"].get("reason") or "")[:300]})
                table = EX.compare_table(entries)
                txt = EX.compare_text(table, c.get("what") or "")
                if len(entries) == 1:
                    e0 = entries[0]
                    txt = f"Only one of the {len(refs)} variants could be run ({e0['label']}), so there is nothing to compare it with. " + EX.explain_run(e0["run"], e0["result"], None)["text"]
                content.update({"view": "sweep", "field": c.get("field"), "what": c.get("what"), "table": table, "text": txt, "mode": entries[0]["mode"],
                                "explanation_source": "deterministic", "compatibility": compatibility(entries),
                                "caveats": (EX.study_caveats(entries[0]["result"]) if entries[0]["run"]["kind"] == "study" else EX.backtest_caveats(entries[0]["result"]))[:4]})
            else:
                for x in decisions:
                    failed.append({"label": x["ref"]["label"], "status": "not covered", "error": (x["decision"].get("reason") or "")[:300]})
                content.update({"view": "failed", "text": "The run did not finish: " + "; ".join(f"{f['label']}: {f['error'] or f['status']}" for f in failed[:4])})
            if failed and content.get("view") != "failed":
                content["text"] = (content.get("text") or "") + " Not completed: " + "; ".join(f"{f['label']} ({f['error'] or f['status']})" for f in failed[:4]) + "."
            out.append(db.add_message(sid, "assistant", "result", content))
            db.update_message(m["id"], {**c, "finalized": True, **({"state": "done"} if m["kind"] == "plan" else {})})
        return out

    def _finalize_analysis(self, kind: str, c: dict, run: dict, mid: int) -> dict:
        from strata.api.results import run_dir
        base = {"source_message": mid, "run_id": run["id"], "explanation_source": "deterministic", "failed": []}
        if run["status"] != "done":
            return {**base, "view": "failed", "text": {"hypothesis": "The research plan", "stress": "The execution stress test", "challenge": "The review"}[kind] + " did not finish: " + ((run.get("error") or run.get("message") or run["status"]).split("\n")[0][:300])}
        fn = {"hypothesis": "hypothesis.json", "stress": "execution.json", "challenge": "challenge.json"}[kind]
        res = json.loads((run_dir(run["id"]) / fn).read_text())
        if kind == "hypothesis":
            ok = [r for r in res["configs"] if r["status"] == "ok"]
            cands = [r for r in ok if r["key"] in res["candidates"]]
            conf = [r for r in ok if r["key"] in res["confirmed"]]
            lines = [f"{res['tested']} configurations were evaluated ({res['evaluable']} had the data they need); {len(cands)} qualified as candidate{'s' if len(cands) != 1 else ''} and {len(conf)} {'was' if len(conf) == 1 else 'were'} confirmed on the final test period."]
            for r in (conf or cands)[:3]:
                tr, te = r["train"], r.get("test") or {}
                lines.append(f"{'Confirmed' if r in conf else r['status_label'].capitalize()}: when {r['condition']}, {HY.OutcomeSpec.load(r['outcome']).text()} happened on {tr['probability'] * 100:.1f}% of {tr['n']} training sessions against "
                             f"{tr['baseline_probability'] * 100:.1f}% on all evaluated sessions" + (f"; {te['probability'] * 100:.1f}% of {te['n']} in the final test period." if te.get("probability") is not None else "."))
            if not cands:
                why: dict[str, int] = {}
                for r in ok:
                    for x in r.get("rejected_because", [])[:1]:
                        k = re.sub(r"\d+", "#", x)
                        why[k] = why.get(k, 0) + 1
                top = sorted(why.items(), key=lambda kv: -kv[1])[:2]
                lines.append("Nothing qualified. Most common reasons: " + "; ".join(f"{k} ({v})" for k, v in top) + "." if top else "Nothing qualified.")
            lines.append("These describe the underlying's behaviour. Whether an option position would have profited is a separate test on observed premiums.")
            if res.get("contaminated"):
                lines.append(LG.holdout_note(res["contaminated"]))
            return {**base, "view": "hypothesis", "mode": "hypothesis", "text": " ".join(lines), "caveats": res["notes"], "question": res["question"], "hypothesis": res.get("hypothesis"), "label": res["question"][:80],
                    "summary": {"tested": res["tested"], "evaluable": res["evaluable"], "candidates": len(cands), "confirmed": len(conf), "session_trials": res["trials"]["session"], "funnel": res["funnel"]},
                    "candidates": [{"key": r["key"], "label": r["label"], "outcome_text": r["outcome_text"], "status_label": r["status_label"], "grade": r.get("grade"), "basis": r["basis"], "options_cover": r.get("options_cover")} for r in cands]}
        if kind == "stress":
            s_ = res["summary"]
            txt = " ".join(res["notes"][:3])
            return {**base, "view": "execution", "mode": "stress", "text": txt, "caveats": res["notes"], "target": c.get("target"), "label": c["runs"][0]["label"], "summary": s_,
                    "rows": [{k: r.get(k) for k in ("id", "group", "label", "status", "trades", "missed", "net_pnl", "expectancy", "note")} for r in res["rows"]]}
        return {**base, "view": "challenge", "mode": "challenge", "text": res["verdict"], "caveats": res["notes"], "target": c.get("target"), "label": c["runs"][0]["label"], "counts": res["counts"],
                "findings": [{k: f[k] for k in ("topic", "name", "severity", "title")} for f in res["findings"]]}

    def payload(self, sid: str) -> dict:
        s, v, _ = self._state(sid)
        syms = sorted({(m["content"].get("plan") or {}).get("symbol") or (m["content"].get("draft") or {}).get("symbol") or "" for m in db.list_messages(sid) if isinstance(m["content"], dict)} - {""})
        return {"session": s, "messages": db.list_messages(sid), "versions": db.list_versions(sid), "current_version": v["id"] if v else None,
                "exec_mode": s["state"].get("exec_mode", "quick"), "trials": max(int(s["state"].get("trials") or 0), LG.count(sid)), "pinned": s["state"].get("pinned", []),
                "ledger": {"summary": LG.summary(sid), "holdouts": [h for sym in syms for h in (LG.state(sid, sym)["holdout"], LG.state(sid, sym + "@intraday")["holdout"]) if h]}}


def _settled(q: dict, old: Draft | None, new: Draft) -> bool:
    """A question is settled once an edit sets exactly one of the readings it offered."""
    for o in q.get("options", []):
        if all(getattr(new, k, None) == v for k, v in o["patch"].items()) and old is not None and any(getattr(old, k, None) != getattr(new, k, None) for k in o["patch"]):
            return True
    return False


def _current_value(d: Draft, field: str):
    """The value the current draft has for a sweepable field (to include it in a comparison)."""
    if field == "prev_ret":
        for c in d.conditions:
            m = re.search(r"(?:previous_session\.\w+|change_pct)\s*[<>]=?\s*-?(\d+(?:\.\d+)?)\s*$", c.expr)
            if m:
                return float(m.group(1))
    if field == "target":
        return d.underlying_target_pct if d.kind == "options" else (abs(d.study.target.value) if d.study and d.study.target and d.study.target.kind == "pct" else None)
    if field == "dte":
        return d.dte
    if field == "hold_sessions":
        return d.hold_sessions
    return None


def _coverage_brief(cov: dict) -> dict | None:
    if not cov:
        return None
    return {"requested": cov.get("requested"), "usable": (cov.get("underlying") or {}).get("usable"), "signals": {k: v for k, v in (cov.get("signals") or {}).items() if k != "dates"},
            "executable": cov.get("executable"), "options": {k: v for k, v in (cov.get("options") or {}).items() if k in ("observation", "dataset", "executable", "signals", "issues")} if cov.get("options") else None,
            "unavailable": cov.get("unavailable") or [], "awaiting_approval": [{k: a.get(k) for k in ("label", "why", "est_bytes", "est_seconds", "requests", "source", "dataset")} for a in (cov.get("awaiting_approval") or [])],
            "downloads": [{k: d.get(k) for k in ("label", "dataset", "symbols", "why", "seconds", "bytes_added", "error")} for d in (cov.get("downloads") or [])], "mode": cov.get("mode"), "mode_label": cov.get("mode_label")}


def compatibility(entries: list[dict]) -> dict:
    """Are these runs comparable? Same kind of pricing, same period, same execution
    assumptions - anything that differs is listed so a comparison is never read as like-for-like."""
    if len(entries) < 2:
        return {"compatible": True, "differences": []}
    diffs = []

    def val(e, f):
        res = e["result"]
        if e["run"]["kind"] == "study":
            return {"pricing": "underlying prices (study)", "period": tuple(res.get("period") or []), "execution": None, "symbol": res.get("symbol")}[f]
        ex = res["config"].get("execution") or {}
        return {"pricing": (res.get("meta") or {}).get("pricing"), "period": (res["config"]["start"], res["config"]["end"]), "symbol": ",".join(sorted({s["symbol"] for s in res["config"]["strategies"]})),
                "execution": tuple((k, ex.get(k)) for k in ("fill_model", "slippage_per_contract", "commission_per_contract", "fee_per_contract", "trade_bar_slippage_pct", "latency_seconds"))}[f]
    names = {"pricing": "Pricing", "period": "Period", "execution": "Execution costs", "symbol": "Symbol"}
    labels = {"historical": "observed bid/ask quotes", "trades": "observed option trade prints", "modeled": "THEORETICAL premiums (model)"}
    for f in ("pricing", "symbol", "period", "execution"):
        vals = [val(e, f) for e in entries]
        if len(set(vals)) > 1:
            if f == "execution":
                keys = sorted({k for v in vals if v for k, _ in v if len({dict(x).get(k) for x in vals if x}) > 1})
                shown = [", ".join(f"{k}={dict(v).get(k)}" for k in keys) if v else "n/a" for v in vals]
            elif f == "period":
                shown = [f"{v[0]} to {v[1]}" if v else "n/a" for v in vals]
            else:
                shown = [labels.get(v, str(v)) for v in vals]
            diffs.append({"field": f, "label": names[f], "values": [{"label": e["label"], "value": s} for e, s in zip(entries, shown)],
                          "severity": "high" if f in ("pricing", "symbol") else "medium"})
    periods = [val(e, "period") for e in entries if val(e, "period")]
    shared = None
    if periods and len(set(periods)) > 1:
        a, b = max(p[0] for p in periods), min(p[1] for p in periods)
        shared = [a, b] if a <= b else None
    warn = None
    pr = {val(e, "pricing") for e in entries}
    if "modeled" in pr and len(pr) > 1:
        warn = "One of these runs uses THEORETICAL option premiums and another uses observed prices. Their profit figures are not comparable."
    elif len(pr) > 1:
        warn = "These runs were priced on different kinds of observation (" + ", ".join(sorted(labels.get(p, str(p)) for p in pr)) + "). Differences between them include the pricing difference."
    return {"compatible": not diffs, "differences": diffs, "shared_window": shared, "warning": warn}


def _discovery_view(res: dict, run_id: str) -> dict:
    ok = [c for c in res["configs"] if c["status"] == "ok"]
    cands = [c for c in ok if c["key"] in res["candidates"]]
    conf = [c for c in ok if c["key"] in res["confirmed"]]
    lines = [f"{res['tested']} configurations were evaluated; {len(cands)} qualified as candidates and {len(conf)} {'was' if len(conf) == 1 else 'were'} confirmed on the untouched final period."]
    for c in conf[:3]:
        e, t = c["explore"], c.get("test") or {}
        lines.append(f"Confirmed: {c['label']} - {e['probability'] * 100:.1f}% of {e['n']} sessions in the explore period versus {e['baseline_probability'] * 100:.1f}% baseline; "
                     f"{t['probability'] * 100:.1f}% of {t['n']} versus {t['baseline_probability'] * 100:.1f}% in the final test period.")
    if cands and not conf:
        lines.append("None of the candidates was confirmed on the final test period, so the hypothesis remains unconfirmed.")
    if not cands:
        why: dict[str, int] = {}
        for c in ok:
            for r in c.get("rejected_because", [])[:1]:
                why[r] = why.get(r, 0) + 1
        top = sorted(why.items(), key=lambda kv: -kv[1])[:2]
        lines.append("No configuration qualified. Most common reasons: " + "; ".join(f"{k} ({v})" for k, v in top) + "." if top else "No configuration qualified.")
    lines.append("Everything else is exploratory: it was found by searching, and searching always finds something.")
    slim = [{k: c.get(k) for k in ("key", "label", "params", "status", "status_label", "q_value", "bonferroni_p", "stability", "candidate", "directional", "rejected_because", "test_note", "error", "excluded_sessions")} |
            {"explore": c.get("explore"), "validate": c.get("validate"), "test": c.get("test")} for c in res["configs"]]
    return {"view": "discovery", "run_id": run_id, "text": " ".join(lines), "explanation_source": "deterministic", "caveats": res["notes"],
            "discovery": {"question": res["question"], "hypothesis": res["hypothesis"], "direction": res["direction"], "dims": res["dims"], "splits": res["splits"], "tested": res["tested"], "evaluable": res["evaluable"],
                          "candidates": res["candidates"], "confirmed": res["confirmed"], "test_period_used": res["test_period_used"], "configs": slim}}
