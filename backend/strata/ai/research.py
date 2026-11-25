"""Research intents: questions that need a plan rather than a single run.

    "Find out whether SPY tends to reverse after large down days."
    "Compare opening calls, puts, and waiting 30 minutes after a 2% overnight gap."
    "Does the first 15-minute candle predict the rest of the trading session?"
    "Investigate mean reversion after unusually large SPY declines."
    "Show me whether this strategy performs differently on Mondays."
    "Test whether this strategy has any statistically significant edge."

`detect` turns such a message into an action for the agent: an investigation (several
related configurations plus the checks to run on them), a discovery grid, or a request
about the results that already exist. Every plan is made of ordinary Drafts, so each
variant is compiled, covered and executed by exactly the same deterministic path as a
single strategy. The conclusions are assembled from the engine's numbers by
`investigation_report`; no language model writes a result.
"""
from __future__ import annotations

import re

from strata.ai import nl_rules as NL
from strata.ai.draft import Draft, DraftCondition, DraftLevel, DraftStrike, DraftStudy
from strata.ai.nl_stateful import enrich, read_costs, read_mode
from strata.research import stats as ST

N = NL.N


def _study(symbol: str, title: str, conds: list[DraftCondition], entry: str | None = None, target: float | None = 0.0, measure: str = "close", direction: str = "up", **kw) -> Draft:
    st = DraftStudy(target=DraftLevel(kind="pct", value=target) if target is not None else None, measure=measure, direction=direction, **kw)
    return Draft(kind="study", symbol=symbol, symbol_stated=True, title=title, conditions=conds, entry_time=entry, study=st)


def _sym(text: str, known: list[str], default: str) -> str:
    return NL.find_symbol(text, [k for k in known if k not in NL.NOT_UNDERLYING]) or default


# ----------------------------------------------------------------------------- fresh research questions
def detect_fresh(text: str, known: list[str], default_symbol: str = "SPY") -> dict | None:
    low = text.lower().strip()
    sym = _sym(text, known, default_symbol)

    # ---- exploratory discovery of a hypothesis over a parameter grid
    if re.match(r"(investigate|explore|research|look for|search for|discover|find (?:me )?(?:an? )?(?:edge|pattern|setup)s?|screen for|mine)\b", low) or re.search(r"\bacross (?:multiple|different|reasonable|a range of) (?:parameters|settings|thresholds)\b", low):
        spec = discovery_spec(text, sym)
        if spec:
            return {"action": "discover", "spec": spec}

    # ---- does X tend to reverse / bounce / continue after large moves
    m = re.search(r"(revers\w*|bounc\w*|rebound\w*|recover\w*|mean[- ]revert\w*|snap\w* back|continu\w*|follow[- ]through|keep (?:falling|rising|going))\s+(?:after|following|the day after|on the day after)\s+(?:a\s+|an\s+)?"
                  r"(large|big|sharp|significant|major|heavy|steep|strong)?\s*(down|red|losing|up|green|winning|declin\w*|sell-?off|drop|fall|rally|gain)", low)
    if m and re.search(r"find out|whether|\bdoes\b|\bdo\b|tend|usually|typically|is there|test (?:if|whether)", low):
        down = bool(re.search(r"down|red|losing|declin|sell|drop|fall", m.group(3)))
        cont = bool(re.search(r"continu|follow|keep", m.group(1)))
        up_outcome = (down != cont)  # reversal after a down day = up; continuation after a down day = down
        variants = []
        for x in (1, 2, 3, 4):
            cond = DraftCondition(expr=f"previous_session.return_pct {'<=' if down else '>='} {'-' if down else ''}{x}", text=f"previous session {'fell' if down else 'rose'} {x}% or more (close-to-close)")
            variants.append({"label": f"after a {x}%+ {'down' if down else 'up'} day", "draft": _study(sym, f"{sym} after a {x}%+ {'down' if down else 'up'} day", [cond], "open", 0.0, "close", "up" if up_outcome else "down")})
        what = "reverse" if not cont else "continue"
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym,
            "hypothesis": f"After a large {'down' if down else 'up'} day, {sym} tends to {what}: the next session closes {'above' if up_outcome else 'below'} its open more often than on other days.",
            "method": f"For previous-day {'declines' if down else 'gains'} of at least 1%, 2%, 3% and 4%: observe the next session from its open and record whether it closes {'higher' if up_outcome else 'lower'} than it opened. "
                      "Each is compared with every other session (the baseline).",
            "measure": f"next session closes {'above' if up_outcome else 'below'} its open", "variants": variants, "checks": ["baseline", "significance", "direction", "out_of_sample", "by_year"],
            "interpretation_notes": ["\"Large\" is not a number, so four thresholds are tested rather than one being assumed.", "\"Reverse\" is measured open-to-close on the following session; the overnight gap is not part of it."]}}

    # ---- does the first N-minute candle predict the rest of the session
    m = re.search(r"first (\w+)[- ]?(?:minute|min)s?(?: candle| bar)?[^.?]{0,40}?(predict|forecast|foreshadow|tell|signal|indicat|determin)", low)
    if m and NL._num(m.group(1)):
        mins = int(NL._num(m.group(1)))
        t = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
        up = _study(sym, f"{sym}: first {mins} minutes up", [DraftCondition(expr="from_open_pct > 0", text=f"the first {mins} minutes close above the open")], t, 0.0, "close", "up")
        dn = _study(sym, f"{sym}: first {mins} minutes down", [DraftCondition(expr="from_open_pct < 0", text=f"the first {mins} minutes close below the open")], t, 0.0, "close", "down")
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym, "hypothesis": f"The direction of the first {mins} minutes carries over: the rest of the session moves the same way more often than not.",
            "method": f"Observe each session from {t} ET. When the first {mins} minutes were up, record whether the close is above the {t} price; when they were down, whether it is below. Both are compared with all sessions.",
            "measure": f"close is beyond the {t} price in the same direction as the first {mins} minutes",
            "variants": [{"label": f"first {mins} min up -> rest of session up", "draft": up}, {"label": f"first {mins} min down -> rest of session down", "draft": dn}],
            "checks": ["baseline", "significance", "out_of_sample"],
            "interpretation_notes": [f"Needs intraday bars that can observe {t} ET, so only the sessions with 1-, 5- or 15-minute bars are used.", "\"Rest of the session\" is measured from the end of the first candle to the close."]}}

    # ---- compare option structures and entry timing under one condition
    if re.search(r"\bcompare\b", low) and re.search(r"\bcalls?\b", low) and re.search(r"\bputs?\b", low) and re.search(r"waiting (\d+)\s*(?:minute|min)|(\d+)\s*(?:minute|min)s? (?:later|after the open)", low):
        w = re.search(r"waiting (\d+)\s*(?:minute|min)|(\d+)\s*(?:minute|min)s? (?:later|after the open)", low)
        mins = int(w.group(1) or w.group(2))
        later = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
        base = enrich(re.sub(r"\bcompare\b|\bputs?,?\b|,?\s*and waiting \d+\s*(?:minute|min)s?", " ", text, flags=re.I), known, default_symbol)
        g = re.search(rf"{N}\s*%\s*(?:overnight\s+|opening\s+)?gap", low)
        conds = [c for c in base.conditions]
        notes = []
        if g and not any("gap_pct" in c.expr for c in conds):
            x = float(g.group(1))
            if re.search(r"gap(?:s|ping|ped)?\s*(?:down|lower)|down gap|gap down", low):
                conds.append(DraftCondition(expr=f"gap_pct <= -{x:g}", text=f"opening gap down {x:g}% or more"))
            elif re.search(r"gap(?:s|ping|ped)?\s*(?:up|higher)|up gap|gap up", low):
                conds.append(DraftCondition(expr=f"gap_pct >= {x:g}", text=f"opening gap up {x:g}% or more"))
            else:
                conds.append(DraftCondition(expr=f"abs(gap_pct) >= {x:g}", text=f"opening gap of {x:g}% or more in either direction"))
                notes.append(f"The gap direction was not stated, so gaps of {x:g}% or more in either direction qualify.")
        variants = []
        for label, structure, entry in (("calls at the open", "long_call", "open"), ("puts at the open", "long_put", "open"), (f"calls after waiting {mins} minutes", "long_call", later), (f"puts after waiting {mins} minutes", "long_put", later)):
            d = base.model_copy(deep=True)
            d.kind, d.structure, d.entry_time, d.entry_time_stated, d.conditions, d.study, d.questions = "options", structure, entry, None, list(conds), None, []
            d.title = f"{sym} {label}"
            if d.dte is None:
                d.dte = 0
            if d.strike is None:
                d.strike = DraftStrike(method="atm")
            variants.append({"label": label, "draft": d})
        return {"action": "investigate", "plan": {
            "question": text.strip(), "symbol": sym, "hypothesis": "After a large overnight gap, the choice between calls and puts, and between entering at the open or waiting, changes the outcome.",
            "method": f"Four positions on the same qualifying sessions: at-the-money calls and puts bought at the open, and the same bought {mins} minutes later. Each is closed before the end of the session.",
            "measure": "profit and loss per trade on observed option prices", "variants": variants, "checks": ["sample", "costs", "paired"],
            "interpretation_notes": notes + ["No expiration was stated: the same-day (0DTE) contract is used because the trade is intraday.", "No exit was stated: each position is closed at 15:55 ET."]}}
    return None


def discovery_spec(text: str, sym: str) -> dict | None:
    """A parameter grid for an exploratory hypothesis. Returns None if the hypothesis is not one of the supported families."""
    from strata.research.discovery import build_grid
    low = text.lower()
    declines = bool(re.search(r"declin|drop|sell-?off|down day|loss|fall|crash|red day|capitulat", low))
    gains = bool(re.search(r"rall|gain|up day|green day|surge|advance|melt-?up", low))
    reversion = bool(re.search(r"mean[- ]revers|revers|bounce|rebound|recover|snap ?back|oversold|fade", low))
    momentum = bool(re.search(r"momentum|continu|follow[- ]through|trend|breakout|persist", low))
    gap = bool(re.search(r"\bgap", low))
    if not (declines or gains or gap):
        return None
    if gap:
        sign = -1 if re.search(r"gap(?:s|ping|ped)?\s*(?:down|lower)|down gap|gap down", low) else 1
        fade = reversion or not momentum
        up = (sign < 0) == fade
        dims = [{"name": "gap", "label": f"Opening gap {'down' if sign < 0 else 'up'} of at least", "values": [0.5, 1.0, 1.5, 2.0], "unit": "%"},
                {"name": "entry", "label": "Observation starts", "values": ["open", "09:45", "10:00", "10:30"]},
                {"name": "target", "label": f"Target ({'up' if up else 'down'} from the entry price)", "values": [0.5, 1.0], "unit": "%"},
                {"name": "regime", "label": "Volatility regime", "values": ["any", "vix20"], "labels": {"any": "any", "vix20": "VIX at least 20"}}]

        def mk(p):
            kids = [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": f"gap_pct {'<=' if sign < 0 else '>='} {'-' if sign < 0 else ''}{p['gap']:g}", "label": f"gap {'down' if sign < 0 else 'up'} {p['gap']:g}%+"}}]
            if p["regime"] == "vix20":
                kids.append({"id": "c1", "kind": "cond", "type": "expr", "params": {"expr": "vix_prev_close >= 20", "label": "VIX closed at 20 or higher the day before"}})
            return ({"symbol": sym, "condition": {"id": "root", "kind": "group", "op": "and", "children": kids}, "entry_time": p["entry"], "reference": "entry",
                     "target": {"kind": "pct", "value": p["target"] if up else -p["target"]}, "horizon": {"kind": "session"}},
                    f"gap {'down' if sign < 0 else 'up'} {p['gap']:g}%+, from {p['entry']}, {'+' if up else '-'}{p['target']:g}% target, {'VIX >= 20' if p['regime'] == 'vix20' else 'any volatility'}")
        hyp = f"After an opening gap {'down' if sign < 0 else 'up'}, {sym} tends to {'fade the gap' if fade else 'keep going'} during the session."
    else:
        down = declines and not (gains and not declines)
        fade = reversion or not momentum
        up = down == fade
        dims = [{"name": "move", "label": f"Previous-day {'decline' if down else 'gain'} of at least", "values": [1, 2, 3, 4], "unit": "%"},
                {"name": "entry", "label": "Observation starts", "values": ["open", "09:45", "10:00", "10:30"]},
                {"name": "target", "label": f"Target ({'up' if up else 'down'} from the entry price)", "values": [0.5, 1.0], "unit": "%"},
                {"name": "regime", "label": "Volatility regime", "values": ["any", "vix20"], "labels": {"any": "any", "vix20": "VIX at least 20"}}]

        def mk(p):
            kids = [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": f"previous_session.return_pct {'<=' if down else '>='} {'-' if down else ''}{p['move']:g}", "label": f"previous day {'fell' if down else 'rose'} {p['move']:g}%+"}}]
            if p["regime"] == "vix20":
                kids.append({"id": "c1", "kind": "cond", "type": "expr", "params": {"expr": "vix_prev_close >= 20", "label": "VIX closed at 20 or higher the day before"}})
            return ({"symbol": sym, "condition": {"id": "root", "kind": "group", "op": "and", "children": kids}, "entry_time": p["entry"], "reference": "entry",
                     "target": {"kind": "pct", "value": p["target"] if up else -p["target"]}, "horizon": {"kind": "session"}},
                    f"{'down' if down else 'up'} {p['move']:g}%+, from {p['entry']}, {'+' if up else '-'}{p['target']:g}% target, {'VIX >= 20' if p['regime'] == 'vix20' else 'any volatility'}")
        hyp = f"After an unusually large {'decline' if down else 'gain'}, {sym} tends to {'reverse' if fade else 'continue'} the next session."
    return {"symbol": sym, "question": text.strip(), "hypothesis": hyp, "direction": "up" if up else "down", "dims": dims, "configs": build_grid(sym, dims, mk),
            "not_explored": ["Calls versus puts and days to expiration are not part of this stage: it measures the underlying's behaviour. Test an option position on any configuration afterwards - observed option prices decide what can be run."]}


# ----------------------------------------------------------------------------- follow-ups about existing work
def detect_followup(text: str, cur: Draft, known: list[str]) -> dict | None:
    low = text.lower().strip()
    if re.search(r"\b(export|download|save|share|print)\b[^.]{0,40}\b(report|results?|research|trade log|trades|configuration|config)\b|\breport\b.*\b(pdf|html|file)\b", low):
        return {"action": "export"}
    m = re.search(r"\b(show|list|display|see|give me|open|view)\b[^.?]{0,40}\b(sessions?|days|dates|trades|observations|occurrences|instances)\b", low)
    if m and not re.search(r"skipped|not taken|unexecuted|missed signal|why", low):
        w = low[m.start():m.end()]
        f = "loss" if re.search(r"losing|loser|unprofitable|missed|failed|unsuccessful", w) else "win" if re.search(r"winning|winner|profitable|successful|hit", w) else "all"
        return {"action": "show_trades", "filter": f}
    if re.search(r"skipped (?:signal|trade|session|entr)|signals? (?:that|which) (?:were|was|got) (?:skipped|not taken|missed|rejected)|why (?:was|were) .{0,30}(?:skipped|not taken|not executed|missed)|(?:unexecuted|unfilled|missed) (?:signal|order|trade)s?", low):
        return {"action": "skipped"}
    if re.search(r"statistic\w* significan|significan\w* edge|is (?:this|it|that) (?:real|significant|just (?:luck|noise|chance))|real edge|any edge|(?:due to|by) chance|p[- ]?value|robust(?:ness)?\b|hold(?:s)? up\b(?!.*out[- ]of[- ]sample)", low):
        return {"action": "robustness"}
    m = re.search(r"\b(mondays?|tuesdays?|wednesdays?|thursdays?|fridays?)\b", low)
    if (m or re.search(r"day of (?:the )?week|weekday", low)) and re.search(r"differ|perform|better|worse|by (?:day|weekday)|break ?down|depend", low):
        return {"action": "breakdown", "by": "weekday", "focus": NL.DAYNAMES.get(m.group(1).rstrip("s")) if m else None}
    if re.search(r"what time|when (?:during|in) the (?:day|session)|time of day|how (?:long|quickly|fast|soon)|how many minutes", low) and re.search(r"target|hit|reach|touch|happen|get there", low):
        return {"action": "timing"}
    if re.search(r"by year|each year|year by year|per year|over the years", low) and re.search(r"how|show|break|perform|result", low):
        return {"action": "breakdown", "by": "year", "focus": None}
    # ---- an option position on the sessions of the current study ("test ATM 0DTE calls on those days")
    structure = NL.find_structure(low)
    if structure and re.search(r"(?:on|for|during) (?:those|these|the same|such) (?:days|sessions|dates)|same (?:days|sessions|conditions|setup|signal)|(?:those|these) (?:days|sessions|signals)|that setup|when that happens", low):
        d = derive_options(text, cur, known, structure)
        return {"action": "derive", "draft": d, "changed": ["position"], "compare": False}
    return None


def derive_options(text: str, cur: Draft, known: list[str], structure: str) -> Draft:
    """An option strategy on the qualifying sessions of the current draft: its symbol, rules,
    entry time and period carry over; the position comes from the new message."""
    fresh = enrich(text, known, cur.symbol)
    d = cur.model_copy(deep=True)
    d.kind, d.structure, d.study, d.title = "options", structure, None, text.strip()[:90]
    d.dte, d.strike, d.width, d.quantity = fresh.dte, fresh.strike, fresh.width, fresh.quantity
    d.mode, d.provider, d.costs = fresh.mode, fresh.provider, fresh.costs or cur.costs
    d.questions, d.unsupported = fresh.questions, fresh.unsupported
    d.premium_target_pct, d.premium_stop_pct = fresh.premium_target_pct, fresh.premium_stop_pct
    d.underlying_target_pct = fresh.underlying_target_pct
    d.underlying_stop_pct = fresh.underlying_stop_pct
    d.exit_time, d.exit_time_stated, d.hold_sessions, d.hold_to_expiration = fresh.exit_time, fresh.exit_time_stated, fresh.hold_sessions, fresh.hold_to_expiration
    if fresh.entry_time and fresh.entry_time_stated:
        d.entry_time, d.entry_time_stated = fresh.entry_time, fresh.entry_time_stated
    elif cur.kind == "study" and not d.entry_time:
        d.entry_time = "open"
    d.notes = [n for n in cur.notes if "Period interpreted" in n]
    d.notes.append("The qualifying sessions are those of the previous question: its symbol, conditions, entry time and period carry over.")
    st = cur.study
    if cur.kind == "study" and st and st.target and st.target.kind == "pct" and st.measure == "touch" and d.underlying_target_pct is None and d.premium_target_pct is None:
        bull = structure in ("long_call", "short_put", "bull_call_spread", "bull_put_spread")
        if (st.target.value > 0) == bull and st.target.value != 0:
            d.underlying_target_pct = abs(st.target.value)
            d.carried["underlying_target_pct"] = "carried over from the question before: its target becomes this position's exit"
            d.notes.append(f"The study's target carries over as the exit: close when {d.symbol} has moved {abs(st.target.value):g}% in the position's favour from the entry (otherwise at the end-of-day exit). Edit it if that is not intended.")
    return d


def followup_edits(text: str, cur: Draft) -> tuple[Draft, list[str]]:
    """Cost assumptions, pricing mode and stateful rules added to the current draft by a follow-up."""
    from strata.ai.nl_stateful import read
    low = text.lower()
    d = cur.model_copy(deep=True)
    changed: list[str] = []
    c = read_costs(low)
    if c is not None and (cur.costs is None or c.model_dump() != cur.costs.model_dump()):
        merged = (cur.costs.model_dump() if cur.costs else {})
        merged.update({k: v for k, v in c.model_dump().items() if v is not None})
        d.costs = type(c).model_validate(merged)
        changed.append("costs")
    mode, prov = read_mode(low)
    if mode and mode != cur.mode and (cur.kind == "options" or mode == "study"):
        d.mode = mode
        changed.append("mode")
    if prov and prov != cur.provider:
        d.provider = prov
        changed.append("provider")
    r = read(text, cur.symbol)
    if r.adds and re.search(r"\badd", low):
        d.adds = r.adds
        changed.append("adds")
    if r.exits and re.search(r"\b(close|exit|sell|get out|flatten)\b", low):
        d.exit_conditions = d.exit_conditions + [x for x in r.exits if x.expr not in {e.expr for e in d.exit_conditions}]
        changed.append("exit_conditions")
    if r.sequence and re.search(r"\b(wait|only enter|only after|after the|then)\b", low):
        d.sequence = r.sequence
        changed.append("sequence")
    for c_ in r.conditions:
        if c_.expr not in {x.expr for x in d.conditions} and re.search(r"\b(only|unless|but|also|and only|require|add (?:a|the) (?:condition|rule|filter))\b", low):
            d.conditions.append(c_)
            changed.append("conditions")
    if r.unsupported:
        d.unsupported = list(dict.fromkeys(d.unsupported + r.unsupported))
        changed.append("unsupported")
    d.notes = d.notes + [n for n in r.notes if n not in d.notes]
    return d, list(dict.fromkeys(changed))


# ----------------------------------------------------------------------------- reports (deterministic)
def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def investigation_report(plan: dict, entries: list[dict], trials: int) -> dict:
    """Findings of an investigation, computed from the variants' stored results.

    entries: [{label, run, result, mode}] in plan order (only finished runs)."""
    rows, findings, limits = [], [], []
    studies = [e for e in entries if e["run"]["kind"] == "study"]
    if studies:
        blocks = [ST.study_block(e["result"]) for e in studies]
        robs = [ST.study_robustness(e["result"], max(trials, len(studies))) for e in studies]
        qs = ST.bh_adjust([b["p_value"] for b in blocks])
        for e, b, q, rb in zip(studies, blocks, qs, robs):
            oos = rb.get("out_of_sample") or {}
            rows.append({"label": e["label"], "run_id": e["run"]["id"], "n": b["n"], "probability": b["probability"], "ci95": b["ci95"], "baseline": b["baseline_probability"], "lift": b["lift"],
                         "p_value": b["p_value"], "adjusted_p_value": q, "avg_end_return_pct": b["avg_end_return_pct"], "other_avg_end_return_pct": b["other_sessions_avg_end_return_pct"],
                         "end_return_p_value": b["end_return_p_value"], "oos_n": oos.get("n"), "oos_probability": oos.get("probability"), "oos_baseline": oos.get("baseline_probability"),
                         "years_above": rb["years_above_baseline"], "level": rb["level"], "period": e["result"].get("period")})
        sig = [r for r in rows if r["adjusted_p_value"] is not None and r["adjusted_p_value"] < 0.05 and r["n"] >= 30]
        small = [r for r in rows if r["n"] < 30]
        measure = plan.get("measure", "the outcome")
        for r in rows:
            if r["n"] == 0:
                findings.append(f"{r['label']}: no qualifying sessions in the stored history.")
                continue
            findings.append(f"{r['label']}: {measure} on {pct(r['probability'])} of {r['n']} sessions (95% interval {pct(r['ci95'][0])} to {pct(r['ci95'][1])}) versus {pct(r['baseline'])} on other sessions"
                            + (f"; adjusted p = {r['adjusted_p_value']:.3f}" if r["adjusted_p_value"] is not None else "") + ".")
        if sig:
            same_oos = [r for r in sig if r["oos_n"] and r["oos_probability"] is not None and r["oos_baseline"] is not None and (r["oos_probability"] > r["oos_baseline"]) == ((r["lift"] or 0) > 0)]
            verdict = (f"{len(sig)} of {len(rows)} variants differ from the baseline after adjusting for the {len(rows)} comparisons"
                       + (f"; {len(same_oos)} of those point the same way in the most recent 30% of history." if sig else "."))
            level = "supported" if len(same_oos) == len(sig) else "mixed"
        elif len(small) == len(rows):
            verdict, level = "Every variant has fewer than 30 qualifying sessions: there is not enough history to answer this either way.", "insufficient"
        else:
            verdict, level = "None of the variants differs from the baseline by more than chance would produce once the number of comparisons is taken into account.", "none"
        if small:
            limits.append(f"{len(small)} variant(s) have fewer than 30 qualifying sessions; their estimates are fragile.")
        limits.append("Probabilities describe the underlying's behaviour. They are not option returns and say nothing about trading costs.")
        res_mix = {k for e in studies for k in (e["result"]["stats"].get("resolution_mix") or {})}
        if res_mix and res_mix != {"1d"}:
            limits.append("Sessions are measured with the finest bars stored for them (" + ", ".join(sorted(res_mix)) + "); intraday history only exists for recent periods.")
        return {"kind": "study", "rows": rows, "findings": findings, "verdict": verdict, "level": level, "limitations": limits, "comparisons": len(rows), "trials": trials}
    # option backtests
    for e in entries:
        res = e["result"]
        m = res.get("metrics") or {}
        cov = res.get("coverage") or {}
        rows.append({"label": e["label"], "run_id": e["run"]["id"], "mode": e["mode"], "trades": m.get("trades"), "net_pnl": m.get("net_pnl"), "win_rate": m.get("win_rate"), "expectancy": m.get("expectancy"),
                     "max_drawdown": m.get("max_drawdown"), "qualifying": (cov.get("signals") or {}).get("qualifying"), "executable": (cov.get("executable") or {}).get("signals"),
                     "pricing": (res.get("meta") or {}).get("pricing"), "period": [res["config"]["start"], res["config"]["end"]], "cost_breakeven_pct": (res.get("cost_sensitivity") or {}).get("breakeven_slippage_pct")})
    done = [r for r in rows if r["trades"]]
    for r in rows:
        if not r["trades"]:
            findings.append(f"{r['label']}: no trades could be executed on observed option prices.")
        else:
            findings.append(f"{r['label']}: {int(r['trades'])} trade{'s' if r['trades'] != 1 else ''} (of {r['qualifying']} qualifying sessions), net {r['net_pnl']:+,.2f}, average {r['expectancy']:+,.2f} per trade, {pct(r['win_rate'])} winners.")
    n_min = min((r["trades"] for r in done), default=0)
    if not done:
        verdict, level = "No variant could be executed on observed option prices, so there is nothing to compare.", "insufficient"
    elif n_min < 30:
        verdict, level = f"The variants ran on {int(n_min)} to {int(max(r['trades'] for r in done))} trades each. That is too few to say one is better than another: differences of this size arise by chance.", "insufficient"
    else:
        verdict, level = "See the per-variant figures; differences should be read together with their sample sizes and cost sensitivity.", "mixed"
    modes = {r["pricing"] for r in done}
    if len(modes) > 1:
        limits.append("The variants were priced differently (" + ", ".join(sorted(m_ or "?" for m_ in modes)) + "); they are not directly comparable.")
    if any((r["qualifying"] or 0) > (r["executable"] or 0) for r in rows):
        limits.append("Only some qualifying sessions have observed option prices; the rest are listed as skipped for lack of data, not counted as losses or wins.")
    limits.append("Past results on the stored data; not a forecast.")
    return {"kind": "backtest", "rows": rows, "findings": findings, "verdict": verdict, "level": level, "limitations": limits, "comparisons": len(rows), "trials": trials}
