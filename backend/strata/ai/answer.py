"""The answer to a question, in the order a person reads it.

  1. the direct answer, in one or two sentences, with the number that answers it
  2. what happened: the counts behind that number, and the comparison that gives it meaning
  3. why it matters, when there is something to say
  4. the limitations that change how far it can be trusted
  5. a few key numbers, each of which opens the observations behind it
  6. one chart chosen for the kind of question
  7. follow-up questions built from this question and the data that is stored

Every number comes from a finished run or from stored bars, and is put into words that say what it is: a frequency is a
count out of a count; an average comes with how far it could be off; a p-value is described as how unusual the result
would be if the condition made no difference, never as the probability that a strategy works or that a result is luck.
Nothing here is written by a language model.
"""
from __future__ import annotations

import math
import re

from strata.ai import semantics as SM

# plain names first; the technical name is kept for the details view
TERMS = {
    "expectancy": ("Average result per trade", "Expectancy", "What one trade made on average, after fees."),
    "drawdown": ("Biggest drop in account value", "Maximum drawdown", "The largest fall from a previous high of the running total."),
    "mfe": ("Furthest the price moved in your favour", "Maximum favourable excursion", "The best point a trade or day reached before it ended."),
    "mae": ("Furthest the price moved against you", "Maximum adverse excursion", "The worst point a trade or day reached before it ended."),
    "execution": ("What happens with worse fills?", "Execution sensitivity", "The same trades with every fill a little worse than the price recorded."),
    "probability": ("How often did this happen?", "Conditional probability", "The share of matching days on which the outcome happened."),
    "qualifying": ("Days that matched your rules", "Qualifying sessions", "The sessions on which every condition held."),
    "robustness": ("How reliable is this result?", "Robustness checks", "Whether the result survives worse fills, later data and the loss of its best trades."),
    "oos": ("Tested on data not used to develop it", "Out-of-sample test", "The later part of the history, measured on its own."),
    "multiple": ("Accounting for how many things were tried", "Multiple-testing correction", "When many variations are tried, the best one looks good partly by luck; the correction allows for that."),
    "p_value": ("How unusual this would be if nothing were going on", "p-value", "How often a difference at least this large would appear if the condition made no difference. It is not the probability that the result is real."),
    "ci": ("The range the true figure is likely in", "95% confidence interval", "Given how much the results vary, the range that would contain the true average 95 times in 100."),
    "baseline": ("What happens on any day", "Baseline", "The same measurement on every day, whether the rule held or not: what the rule has to beat."),
}
TERMS = {k: {"plain": v[0], "technical": v[1], "meaning": v[2]} for k, v in TERMS.items()}  # plain words first; the technical name is in the details


def _pct(x, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}%"


def _sp(x, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:+.{d}f}%"


def _usd(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{'-' if x < 0 else '+'}${abs(x):,.{d}f}"


def _n(x) -> str:
    return f"{int(x):,}"


def _date(s: str | None) -> str:
    if not s:
        return ""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(str(s)[:10])
        return f"{d.strftime('%B')} {d.day}, {d.year}"
    except ValueError:
        return str(s)


def _month(s: str | None) -> str:
    if not s:
        return ""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(str(s)[:10])
        return f"{d.strftime('%b')} {d.year}"
    except ValueError:
        return str(s)


def _subject(frame: dict | None) -> str:
    f = frame or {}
    return f.get("symbol") or "SPY"


def _small(n: int) -> str | None:
    if n < 12:
        return f"Only {n} day{'s' if n != 1 else ''} matched: far too few to establish a pattern. Read this as a description of those days, not as a tendency."
    if n < 30:
        return f"{n} days is a small sample: a few days going the other way would change the picture."
    return None


# ------------------------------------------------------------------------------------------------------ follow-ups
def followups(frame: dict | None, result_kind: str, extra: dict | None = None) -> list[dict]:
    """Questions that follow from this one, each phrased so that the reader takes it as an edit of this question."""
    f = frame or {}
    k = f.get("kind") or result_kind
    sym = f.get("symbol") or "SPY"
    others = [s for s in ("SPY", "QQQ", "IWM") if s != sym]
    conds = f.get("conditions") or []
    out = []

    def add(text, why):
        if text not in [o["text"] for o in out]:
            out.append({"text": text, "why": why})
    pr = next((c for c in conds if c.get("key") == "prev_return" and c.get("value")), None)
    gp = next((c for c in conds if c.get("key") == "gap" and c.get("value")), None)
    if k in ("frequency", "movement"):
        if pr:
            v = abs(pr["value"])
            bigger = 2 * v if v < 2 else v + 1
            add(f"What about {bigger:g}% {'down' if pr['value'] < 0 else 'up'} days?", "Does the size of the move matter?")
        if gp:
            v = abs(gp["value"])
            add(f"What about gaps of {2 * v:g}%?", "Does a bigger gap change it?")
        if not gp and k == "frequency":
            add("Only when it gapped down 1% or more", "Does a large overnight gap change the result?")
        add(f"Is {others[0]} any different?", "The same question on another index fund.")
        if k == "frequency":
            add("Show me the days it didn't happen", "The days behind the other side of the count.")
        else:
            add("Show me the worst days", "The sessions furthest from the average.")
        if (f.get("end") or "close") == "close" and k == "movement":
            add("What happens in the first hour?", "The same days, measured over the first hour only.")
        if (f.get("start") or "open") in ("open", None) and k == "frequency" and (f.get("outcome") or {}).get("key") in ("close_vs", "touch"):
            add("What if I bought calls on those days?", "The same condition traded with options, on real option prices (from January 2024).")
    elif k == "agreement":
        c = f.get("candle") or [5]
        if len(c) == 1:
            add(f"Use {15 if c[0] != 15 else 30} min", "Does a longer candle predict better?")
        add(f"Actually do {others[0]}", "The same question on another index fund.")
        if not any(x.get("key") == "prev_return" for x in conds):
            add("What about when yesterday was red?", "Only the sessions after a down day.")
        add("Show me the days it was wrong", "The days where the candle and the rest of the session went different ways.")
        add("What if I bought calls instead?", "Calls when the candle closes up, on real option prices (from January 2024).")
    elif k == "timing":
        add("What about the low of the day?" if f.get("measure") == "high_time" else "When does it make its high?", "The other end of the day's range.")
        add(f"Is {others[0]} any different?", "The same question on another index fund.")
        add("Only on days after a 1% drop", "Does a bad previous day change the pattern?")
    elif k == "breakdown":
        add("By month instead" if f.get("by") == "weekday" else "By weekday instead", "Another way to cut the same days.")
        add(f"Is {others[0]} any different?", "The same question on another index fund.")
        add("Only since 2020", "Has it held in recent years?")
    elif k == "strategy":
        add("Why did the biggest loss happen?", "The worst trade, explained from its option prices.")
        add("Explain this like I'm new to options", "One real trade from this test, step by step.")
        add("Is this actually good, or just luck?", "What the evidence can and cannot say.")
        add("Show me the losing trades", "Every trade that lost money.")
        if (extra or {}).get("dte") in (0, None):
            add("Try 7 DTE", "Contracts with a week left lose value more slowly.")
        add("Exit an hour earlier", "Does holding less time help?")
    return out[:5]


# --------------------------------------------------------------------------------------------------------- frequency
def frequency(res: dict, frame: dict | None, run_id: str | None = None) -> dict:
    st = res.get("stats") or {}
    n = int(st.get("qualifying_sessions") or 0)
    hits = int(st.get("target_first") or 0)
    amb = int(st.get("ambiguous") or 0)
    p = st.get("probability")
    ci = st.get("probability_ci95") or [None, None]
    pb = st.get("baseline_probability")
    bn = int(st.get("baseline_sessions") or 0)
    pval = st.get("p_value_vs_rest")
    f = frame or {}
    sym = res.get("symbol") or _subject(f)
    period = res.get("period") or [None, None]
    o = f.get("outcome") or {}
    what = SM.describe_outcome(o, f.get("start") or "open") or "the outcome happened"
    what_short = re.sub(r"^it ", "", what)
    conds = SM._describe_conditions(f.get("conditions") or []) if f else ""
    if f.get("next_session") and o.get("key") == "close_vs":
        what_short = "the next session " + what_short
    miss = n - hits - amb
    if n == 0:
        return {"kind": "frequency", "headline": f"No day in the stored {sym} history matched your rules{(': ' + conds) if conds else ''}, so there is nothing to count.", "happened": [], "limits": [],
                "numbers": [], "viz": None, "followups": followups(f, "frequency")}
    head = f"{sym} {what_short} on {_n(hits)} of {_n(n)} days{(' when ' + conds) if conds else ''} ({_pct(p * 100 if p is not None else None)})."
    happened = []
    cmp_ = None
    # "the gap fills" against all days is not a fair comparison: on a day without a gap the price starts at the previous close
    keys = {c.get("key") for c in f.get("conditions") or []}
    mechanical = o.get("key") == "touch_level" and ((o.get("level") == "prev_close" and "gap" in keys) or (o.get("level") == "open" and keys & {"from_open", "change"}))
    if mechanical:
        happened.append("There is no fair \"any day\" comparison for this: on days without the condition the price starts at that level, so reaching it is automatic.")
        pb = None
    if pb is not None and p is not None and conds:
        inside = ci[0] is not None and ci[0] <= pb <= ci[1]
        if inside:
            cmp_ = f"That is about the same as on any day: {_pct(pb * 100)} of all {_n(bn)} days."
        else:
            cmp_ = f"That is {'more' if p > pb else 'less'} often than on a typical day, when it happened {_pct(pb * 100)} of the time ({_n(bn)} days)."
        happened.append(cmp_)
    if ci[0] is not None:
        happened.append(f"Allowing for how few days there are, the true rate is likely between {_pct(ci[0] * 100)} and {_pct(ci[1] * 100)}.")
    if amb:
        happened.append(f"On {_n(amb)} more day{'s' if amb != 1 else ''} the order of events inside one price bar cannot be known; they are counted as neither.")
    matters = None
    if pb is not None and p is not None and conds:
        diff = (p - pb) * 100
        if pval is not None and pval < 0.05 and abs(diff) >= 3:
            matters = f"The condition moves the rate by {abs(diff):.1f} percentage points, and a difference this large would be unusual if the condition made no difference. It is a difference in how often, not a trading result."
        elif pval is not None:
            matters = "The condition makes little difference: days that match it behave much like the rest."
    limits = [x for x in [_small(n)] if x]
    limits.append("This describes the price of " + sym + " itself, not an option position.")
    excl = int(st.get("excluded_sessions") or 0)
    if excl:
        why = " ".join(str(x.get("reason") or "") for x in res.get("excluded") or [])
        if "intraday" in why or "daily bars" in why:
            limits.append(f"{_n(excl)} earlier session{'s' if excl != 1 else ''} {'have' if excl != 1 else 'has'} only a daily price record, so a time of day cannot be checked on {'them' if excl != 1 else 'it'} "
                          "(minute-by-minute prices are stored from 2016); they are not counted either way.")
        else:
            limits.append(f"{_n(excl)} session{'s' if excl != 1 else ''} could not be measured (prices missing for part of the session) and {'are' if excl != 1 else 'is'} not counted either way.")
    res_mix = st.get("resolution_mix") or {}
    if res_mix.get("1d") and len(res_mix) > 1:
        limits.append(f"{_n(res_mix['1d'])} of the days are before minute-by-minute history (2016 for these funds), so only their daily bars are used.")
    numbers = [{"label": "Days that matched your rules", "value": _n(n), "explore": {"run_id": run_id, "kind": "sessions", "filter": "all"}},
               {"label": "Days it happened", "value": _n(hits), "explore": {"run_id": run_id, "kind": "sessions", "filter": "hit"}},
               {"label": "Days it didn't", "value": _n(miss), "explore": {"run_id": run_id, "kind": "sessions", "filter": "miss"}}]
    if pb is not None and conds:
        numbers.append({"label": "On any day", "value": _pct(pb * 100), "detail": f"of {_n(bn)} days"})
    return {"kind": "frequency", "headline": head, "happened": happened, "matters": matters, "limits": limits, "numbers": numbers,
            "viz": {"kind": "frequency", "hits": hits, "misses": miss, "ambiguous": amb, "n": n, "rate": p, "ci": ci, "baseline": pb if conds else None, "baseline_n": bn, "by_year": res.get("by_year") or [],
                    "label_hit": what_short, "run_id": run_id},
            "details": [{"label": "Two-sided test against all other days (p-value)", "value": f"{pval:.4f}" if pval is not None else "n/a"}, {"label": "Period", "value": f"{period[0]} to {period[1]}"},
                        {"label": "95% interval (Wilson)", "value": f"{_pct((ci[0] or 0) * 100)} to {_pct((ci[1] or 0) * 100)}" if ci[0] is not None else "n/a"}],
            "terms": [TERMS[k] for k in ("probability", "qualifying", "baseline", "ci", "p_value")], "followups": followups(f, "frequency")}


def movement(res: dict, frame: dict | None, run_id: str | None = None) -> dict:
    st = res.get("stats") or {}
    f = frame or {}
    sym = res.get("symbol") or _subject(f)
    n = int(st.get("qualifying_sessions") or 0)
    avg, med = st.get("avg_end_return_pct"), st.get("median_end_return_pct")
    ci = st.get("end_return_ci95") or [None, None]
    pos = st.get("pct_positive_end")
    conds = SM._describe_conditions(f.get("conditions") or [])
    st_, en = f.get("start") or "open", f.get("end") or "close"
    span = ("from the open" if st_ in ("open", None) else f"from {st_}") + (" to the close" if en in ("close", None) else f" to {en}" if re.match(r"\d", str(en)) else "")
    if n == 0:
        return {"kind": "movement", "headline": f"No {sym} day matched{(': ' + conds) if conds else ''}.", "numbers": [], "followups": followups(f, "movement")}
    ps = res.get("path_summary") or {}
    q = ps.get("end_quartiles")
    head = (f"After {conds}, {sym} moved {_sp(avg)} on average {span} (median {_sp(med)}), over {_n(n)} days." if conds else
            f"{sym} moved {_sp(avg)} on average {span} (median {_sp(med)}), over {_n(n)} days.")
    happened = []
    if pos is not None:
        happened.append(f"It ended higher than it started on {_pct(pos * 100)} of those days.")
    if q:
        happened.append(f"Half of the days ended between {_sp(q[0])} and {_sp(q[1])}; the furthest it typically ran in its favour was {_sp(st.get('median_mfe_pct'))} and against it {_sp(st.get('median_mae_pct'))} (medians).")
    if ci[0] is not None:
        happened.append(f"The average could be off: the range it is likely in runs from {_sp(ci[0])} to {_sp(ci[1])}" + (", which includes zero." if ci[0] <= 0 <= ci[1] else "."))
    limits = [x for x in [_small(n)] if x] + ["Price of " + sym + " itself, not an option position."]
    numbers = [{"label": "Days that matched", "value": _n(n), "explore": {"run_id": run_id, "kind": "sessions", "filter": "all"}},
               {"label": "Average move", "value": _sp(avg), "detail": span},
               {"label": "Ended higher", "value": _pct(pos * 100) if pos is not None else "n/a", "explore": {"run_id": run_id, "kind": "sessions", "filter": "up"}}]
    return {"kind": "movement", "headline": head, "happened": happened, "limits": limits, "numbers": numbers,
            "viz": {"kind": "paths", "paths": ps, "run_id": run_id, "end_hist": res.get("end_hist")}, "terms": [TERMS[k] for k in ("mfe", "mae", "ci")], "followups": followups(f, "movement"),
            "details": [{"label": "Average, 95% bootstrap interval", "value": f"{_sp(ci[0])} to {_sp(ci[1])}" if ci[0] is not None else "n/a"}]}


# --------------------------------------------------------------------------------------------------------- agreement
def agreement(rep: dict, frame: dict | None) -> dict:
    rows = rep.get("agreement") or []
    f = frame or {}
    sym = f.get("symbol") or ""
    if not rows:
        return {"kind": "agreement", "headline": rep.get("verdict") or "Nothing could be measured.", "happened": rep.get("findings") or [], "limits": rep.get("limitations") or [], "numbers": [],
                "followups": followups(f, "agreement")}
    pred = (rows[0].get("predictor") or f.get("predictor") or "first_candle")
    mins = sorted({r["minutes"] for r in rows})
    day = [r for r in rows if r["scope"] == "day"]
    rest = [r for r in rows if r["scope"] == "rest"]
    lead = min(mins)

    def one(scope_rows, m):
        return next((r for r in scope_rows if r["minutes"] == m), None)
    d0, r0 = one(day, lead), one(rest, lead)
    head_parts, happened = [], []
    name = f"The first {lead}-minute candle" if pred == "first_candle" else "The previous session's direction"
    def _always_up(r):
        # "guess up every day", on the same days that were classified (with any condition applied): from the counts themselves
        if not r:
            return None
        u, d = r["up"], r["down"]
        uh = (u.get("probability") or 0) * u["n"]
        dh = (d.get("probability") or 0) * d["n"]
        tot = u["n"] + d["n"]
        return (uh + (d["n"] - dh)) / tot if tot else None
    always_up = _always_up(r0 or d0)
    if d0 and r0 and r0.get("distinguishable"):
        head_parts.append(f"{name} matched the direction of the whole day {_pct(d0['agreement'] * 100)} of the time, and the direction of the rest of the session, after the candle closed, {_pct(r0['agreement'] * 100)} of the time.")
    elif d0 and r0:
        head_parts.append(f"{name} matched the direction of the whole day {_pct(d0['agreement'] * 100)} of the time. However, it matched the direction of the rest of the session, after the candle closed, only {_pct(r0['agreement'] * 100)} of the time.")
        head_parts.append("The candle looks predictive mainly because its own move is part of the whole day. After it closes, there is little evidence that its direction predicts what happens next."
                          if not r0.get("distinguishable") else "After it closes, its direction still agrees with the rest of the session more often than unrelated directions would.")
    elif r0:
        head_parts.append(f"{name} matched the direction of the rest of the session {_pct(r0['agreement'] * 100)} of the time, " +
                          ("about what two unrelated directions would give." if not r0.get("distinguishable") else "more often than two unrelated directions would."))
    elif d0:
        head_parts.append(f"{name} matched the direction of the {'whole' if pred == 'first_candle' else 'next'} session {_pct(d0['agreement'] * 100)} of the time" +
                          (", about what two unrelated directions would give." if not d0.get("distinguishable") else ", more often than two unrelated directions would."))
    ref = r0 or d0
    if ref and ref.get("if_unrelated") is not None:
        happened.append(f"If the two directions were unrelated they would match about {_pct(ref['if_unrelated'] * 100)} of the time; the measured rate's likely range is {_pct(ref['ci95'][0] * 100)} to {_pct(ref['ci95'][1] * 100)}.")
    if r0 and always_up is not None and pred == "first_candle":
        happened.append(f"Simply guessing \"up\" every day would have matched the rest of the session {_pct(always_up * 100)} of the time" +
                        (f", which is {'better' if always_up > r0['agreement'] else 'worse'} than following the candle." if abs(always_up - r0["agreement"]) > 0.002 else "."))
    if len(mins) > 1:
        happened.append("Longer candles: " + "; ".join(f"{m} minutes: {_pct(one(day, m)['agreement'] * 100)} of the whole day" + (f", {_pct(one(rest, m)['agreement'] * 100)} of the rest" if one(rest, m) else "")
                                                     for m in mins[1:] if one(day, m)) + ".")
    n = (ref or {}).get("sessions") or 0
    matrix = []
    for r in rows:
        u, d = r["up"], r["down"]
        uh = round((u.get("probability") or 0) * u["n"])
        dh = round((d.get("probability") or 0) * d["n"])
        matrix.append({"minutes": r["minutes"], "scope": r["scope"], "from": r.get("from"), "up_up": uh, "up_down": u["n"] - uh, "down_down": dh, "down_up": d["n"] - dh, "agreement": r["agreement"],
                       "if_unrelated": r.get("if_unrelated"), "always_up": _always_up(r), "sessions": r["sessions"], "up_run": u.get("run_id"), "down_run": d.get("run_id"), "distinguishable": r.get("distinguishable")})
    numbers = []
    if d0:
        numbers.append({"label": "Matched the whole day", "value": _pct(d0["agreement"] * 100), "detail": f"{_n(d0['sessions'])} days", "explore": {"kind": "classification", "runs": [d0["up"]["run_id"], d0["down"]["run_id"]], "filter": "all"}})
    if r0:
        numbers.append({"label": "Matched the rest of the day", "value": _pct(r0["agreement"] * 100), "detail": f"{_n(r0['sessions'])} days", "explore": {"kind": "classification", "runs": [r0["up"]["run_id"], r0["down"]["run_id"]], "filter": "all"}})
        numbers.append({"label": "Days the candle was wrong", "value": _n(r0["sessions"] - r0["agree"]), "explore": {"kind": "classification", "runs": [r0["up"]["run_id"], r0["down"]["run_id"]], "filter": "wrong"}})
    if ref and ref.get("if_unrelated") is not None:
        numbers.append({"label": "If unrelated", "value": _pct(ref["if_unrelated"] * 100)})
    limits = [x for x in rep.get("limitations") or [] if "not a forecast" in x or "Direction only" in x or "Only sessions" in x][:3]
    return {"kind": "agreement", "headline": " ".join(head_parts), "happened": happened, "limits": limits, "numbers": numbers,
            "viz": {"kind": "matrix", "rows": matrix, "lead": lead, "predictor": pred}, "terms": [TERMS["baseline"], TERMS["ci"], TERMS["multiple"]], "followups": followups(f, "agreement"),
            "details": [{"label": "Adjusted for comparisons", "value": f"{rep.get('comparisons')} studies"}] + [{"label": l, "value": ""} for l in rep.get("findings") or []]}


# --------------------------------------------------------------------------------------------------- timing, breakdown
def timing(res: dict, frame: dict | None) -> dict:
    f = frame or {}
    sym = res["symbol"]
    b = res["buckets"]
    n = res["sessions"]
    per = f"{_n(n)} sessions from {_month(res['first'])} to {_month(res['last'])}"
    if res["measure"] in ("high_time", "low_time"):
        which = "high" if res["measure"] == "high_time" else "low"
        ranked = sorted(b, key=lambda r: -r["sessions"])
        a, c = ranked[0], ranked[1]
        head = f"{sym} made its {which} of the day between {a['start']} and {a['end']} on {_pct(a['share'])} of sessions, more than any other half hour" if res["bucket_minutes"] == 30 else \
            f"{sym} made its {which} of the day between {a['start']} and {a['end']} on {_pct(a['share'])} of sessions, more than in any other {res['bucket_minutes']}-minute part of the day"
        head += f"; the next most common was {c['start']}–{c['end']} ({_pct(c['share'])})."
        happened = [f"If the {which} could fall anywhere, each part of the day would hold about {_pct(res['even_share'])}." if res.get("even_share") else "",
                    f"Counted on {per} (New York time)."]
        matters = "The open and the close are where the day's extremes cluster: a level set in the middle of the day is usually not the day's high or low." if a["start"] in ("09:30", "15:30") else None
        numbers = [{"label": f"{a['start']}–{a['end']}", "value": _pct(a["share"]), "explore": {"kind": "timing", "filter": a["start"]}},
                   {"label": f"{c['start']}–{c['end']}", "value": _pct(c["share"]), "explore": {"kind": "timing", "filter": c["start"]}},
                   {"label": "Sessions", "value": _n(n)}]
    else:
        ranked = sorted([r for r in b if r["mean_abs_move_pct"] is not None], key=lambda r: -r["mean_abs_move_pct"])
        a, c = ranked[0], ranked[1]
        quiet = min(b, key=lambda r: r["mean_abs_move_pct"] if r["mean_abs_move_pct"] is not None else 99)
        head = (f"{sym} moves most between {a['start']} and {a['end']}: {a['mean_abs_move_pct']:.2f}% on average, against {quiet['mean_abs_move_pct']:.2f}% in the quietest part of the day ({quiet['start']}–{quiet['end']}). "
                f"The next most active is {c['start']}–{c['end']} ({c['mean_abs_move_pct']:.2f}%).")
        happened = [f"That part of the day held the session's largest move on {_pct(a['largest_move_share'])} of sessions.", f"Counted on {per} (New York time)."]
        matters = None
        numbers = [{"label": f"{a['start']}–{a['end']}", "value": f"{a['mean_abs_move_pct']:.2f}%", "detail": "average move"},
                   {"label": f"{quiet['start']}–{quiet['end']}", "value": f"{quiet['mean_abs_move_pct']:.2f}%", "detail": "quietest"}, {"label": "Sessions", "value": _n(n)}]
    return {"kind": "timing", "headline": head, "happened": [h for h in happened if h], "matters": matters, "limits": res.get("notes") or [], "numbers": numbers,
            "viz": {"kind": "timing", "buckets": b, "measure": res["measure"], "even": res.get("even_share")}, "followups": followups(f, "timing"), "terms": []}


def breakdown(res: dict, frame: dict | None) -> dict:
    f = frame or {}
    sym = res["symbol"]
    g = res["groups"]
    best = max(g, key=lambda x: x["mean_pct"])
    worst = min(g, key=lambda x: x["mean_pct"])
    p = res["p_value_groups_differ"]
    unit = "weekday" if res["by"] == "weekday" else "month"
    plural = (best["group"] + "s") if res["by"] == "weekday" else best["group"]
    plural_w = (worst["group"] + "s") if res["by"] == "weekday" else worst["group"]
    head = f"Over {_n(res['sessions'])} sessions, {res['measure_label']} on {sym} averaged most on {plural} ({_sp(best['mean_pct'], 3)}) and least on {plural_w} ({_sp(worst['mean_pct'], 3)}). "
    head += ("These differences are no larger than shuffling the labels produces: no " + unit + " stands out." if p >= 0.05 else
             f"The {unit}s differ by more than shuffling the labels usually produces, but the averages are a few hundredths of a percent against daily moves of around a percent.")
    foc = next((x for x in g if f.get("focus") and str(x["group"])[:3].lower() == str(f["focus"])[:3].lower()), None)
    if foc is not None and res["by"] == "weekday":
        # the question named one day: answer about that day first, against the others
        fn = foc["group"] + "s"
        others = foc.get("others_mean_pct")
        lower = others is not None and foc["mean_pct"] < others
        inside = foc.get("ci95") and others is not None and foc["ci95"][0] <= others <= foc["ci95"][1]
        head = (f"{fn} on {sym} averaged {_sp(foc['mean_pct'], 3)} from the open to the close, against {_sp(others, 3)} on the other days: "
                + ("slightly worse" if lower else "slightly better") + (", but well within what chance produces, so " + fn + " are not reliably " + ("bad." if lower else "good.") if inside or p >= 0.05 else
                                                                         ", and the gap is larger than chance usually produces, though it is small next to a typical day's move.")
                + f" ({_n(foc['sessions'])} {fn}.)")
    happened = [f"On {plural} it closed higher than it opened on {_pct(best['positive_share'])} of days; on {plural_w}, {_pct(worst['positive_share'])}." if res["measure"] in ("open_to_close", "return") else "",
                f"From {_date(res['first'])} to {_date(res['last'])}."]
    numbers = [{"label": f"Best: {best['group']}", "value": _sp(best["mean_pct"], 3), "detail": f"{_n(best['sessions'])} days", "explore": {"kind": "group", "filter": best["group"]}},
               {"label": f"Worst: {worst['group']}", "value": _sp(worst["mean_pct"], 3), "detail": f"{_n(worst['sessions'])} days", "explore": {"kind": "group", "filter": worst["group"]}},
               {"label": "All days", "value": _sp(res["overall_mean_pct"], 3)}]
    return {"kind": "breakdown", "headline": head, "happened": [h for h in happened if h], "limits": res.get("notes") or [], "numbers": numbers,
            "viz": {"kind": "breakdown", "groups": [{k: v for k, v in x.items() if k != "dates"} for x in g], "by": res["by"]}, "followups": followups(f, "breakdown"),
            "details": [{"label": "Groups differ (permutation p-value)", "value": f"{p:.3f}"}], "terms": [TERMS["p_value"]]}


def event_frequency(res: dict, frame: dict | None) -> dict:
    f = frame or {}
    k, n = res["k"], res["n"]
    head = f"{res['symbol']} {res['what']} on {_n(k)} of {_n(n)} sessions ({_pct(res['pct'])}), from {_date(res['first'])} to {_date(res['last'])}."
    numbers = [{"label": "Sessions it happened", "value": _n(k), "explore": {"kind": "dates", "filter": "hit"}}, {"label": "All sessions", "value": _n(n)}, {"label": "Share", "value": _pct(res["pct"])}]
    return {"kind": "frequency", "headline": head, "happened": [f"About once every {n / k:.0f} sessions." if k else "It never happened in the stored history."], "limits": res.get("notes") or [],
            "numbers": numbers, "viz": {"kind": "frequency", "hits": k, "misses": n - k, "ambiguous": 0, "n": n, "rate": k / n if n else None, "by_year": res.get("by_year") or []},
            "followups": followups(f, "frequency")}


# ---------------------------------------------------------------------------------------------------------- strategy
def strategy(run: dict, res: dict, trades: list[dict], card: dict | None, frame: dict | None = None) -> dict:
    """A backtest of an option strategy, answered in money, with what the evidence can and cannot say."""
    m = res.get("metrics") or {}
    cov = res.get("coverage") or {}
    n = len(trades)
    net = sum(float(t["pnl"]) for t in trades) if trades else 0.0
    fees = sum(float(t.get("commissions") or 0) for t in trades)
    sec = {s["id"]: s for s in (card or {}).get("sections") or []}
    ex = (sec.get("expectancy") or {}).get("figures") or {}
    avg = ex.get("mean") if ex.get("mean") is not None else (net / n if n else None)
    ci = ex.get("ci95") or [None, None]
    first = min((t["entry_date"] for t in trades), default=None)
    last = max((t["entry_date"] for t in trades), default=None)
    label = (frame or {}).get("title") or run.get("name") or "this strategy"
    qual = (cov.get("signals") or {}).get("qualifying")
    priced = (cov.get("executable") or {}).get("signals")
    if n == 0:
        span = _option_span(run, res)
        return {"kind": "strategy", "headline": "No trade could be made: " + (f"your rule occurred {qual} times, but option prices were available for none of those days." if qual else "the rule never held on a day with stored option prices."),
                "happened": [span] if span else [], "limits": [coverage_sentence(cov)], "numbers": [], "followups": followups(frame, "strategy")}
    head = f"Over {_n(n)} trade{'s' if n != 1 else ''} from {_month(first)} to {_month(last)}, this made {_usd(net)} after {_usd(fees).lstrip('+')} in fees: {_usd(avg)} per trade on average."
    happened = []
    if ci[0] is not None:
        if ci[0] <= 0 <= ci[1]:
            happened.append(f"That average is not reliably different from zero. Given how much single trades vary, the average per trade is likely between {_usd(ci[0])} and {_usd(ci[1])}: a range that includes losing money.")
        else:
            happened.append(f"Given how much single trades vary, the average per trade is likely between {_usd(ci[0])} and {_usd(ci[1])}. That range is {'above' if ci[0] > 0 else 'below'} zero, but it is one historical sample.")
    wins = [t for t in trades if t["pnl"] > 0]
    happened.append(f"{_n(len(wins))} of {_n(n)} trades made money ({_pct(len(wins) / n * 100)}).")
    streak = cur = 0
    for t in sorted(trades, key=lambda t: t["entry_ts"]):
        cur = cur + 1 if t["pnl"] <= 0 else 0
        streak = max(streak, cur)
    if streak >= 3:
        happened.append(f"The longest run of losing trades was {streak}.")
    exe = (sec.get("execution") or {}).get("figures") or {}
    rows = exe.get("rows") or []
    w2 = next((r for r in rows if abs(r["worse_fills_pct"] - 2) < 1e-9), None)
    be = exe.get("breakeven_pct")
    if w2 is not None:
        happened.append(f"With every fill 2% worse than the recorded price, the result would have been {_usd(w2['net'])}" + (f"; it disappears at about {be:.1f}% worse fills." if be else "."))
    conc = (sec.get("concentration") or {}).get("figures") or {}
    if conc.get("net_without_best_3") is not None and n >= 10:
        happened.append(f"Without its three best trades the result is {_usd(conc['net_without_best_3'])}.")
    matters = None
    if ci[0] is not None and ci[0] <= 0 <= ci[1]:
        matters = ("A positive total is not evidence of an edge: with this many trades and this much variation, a result like this comes up often by chance. Treat it as a description of what happened, not a strategy that has been shown to work."
                   if net > 0 else "The loss is real money in this history, but it is not evidence that the idea reliably loses either: with this much variation between trades, an average this far below zero comes up often by chance.")
    limits = [coverage_sentence(cov)]
    limits.append("Prices are real option trades, not bid/ask quotes: the spread you would have paid is not in these results.")
    if (card or {}).get("evidence", {}).get("level") in ("insufficient",):
        limits.append("Fewer than 30 trades: averages from this few trades move a lot with each new trade.")
    worst = min(trades, key=lambda t: t["pnl"])
    best = max(trades, key=lambda t: t["pnl"])
    numbers = [{"label": "Total after fees", "value": _usd(net), "explore": {"kind": "trades", "run_id": run["id"], "filter": "all"}},
               {"label": "Average result per trade", "value": _usd(avg), "detail": f"{_n(n)} trades"},
               {"label": "Trades that made money", "value": _pct(len(wins) / n * 100), "explore": {"kind": "trades", "run_id": run["id"], "filter": "winners"}},
               {"label": "Biggest loss", "value": _usd(worst["pnl"]), "detail": _date(worst["entry_date"]), "explore": {"kind": "replay", "run_id": run["id"], "trade": worst["id"]}}]
    if qual and priced is not None and priced < qual:
        numbers.append({"label": "Days that could not be tested", "value": _n(qual - priced), "explore": {"kind": "skipped", "run_id": run["id"]}})
    eq, s_ = [], 0.0
    for t in sorted(trades, key=lambda t: t["exit_ts"] or t["entry_ts"]):
        s_ += float(t["pnl"])
        eq.append({"date": t["entry_date"], "pnl": round(float(t["pnl"]), 2), "total": round(s_, 2), "id": t["id"]})
    sz = sizing_story(run, res, trades, frame)
    if sz:
        happened += sz["lines"][:2]
    return {"kind": "strategy", "headline": head, "happened": happened, "matters": matters, "limits": limits, "numbers": numbers, "sizing": sz,
            "viz": {"kind": "strategy", "trades": eq, "run_id": run["id"], "worst": worst["id"], "best": best["id"]}, "example": example_trade(trades, run["id"]),
            "terms": [TERMS[k] for k in ("expectancy", "drawdown", "execution", "ci", "robustness")], "followups": followups(frame, "strategy", {"dte": (frame or {}).get("dte")}),
            "details": [{"label": l, "value": ""} for l in [(card or {}).get("evidence", {}).get("text", "")] if l] + [{"label": (sec.get(k) or {}).get("title", k), "value": (sec.get(k) or {}).get("text", "")} for k in ("expectancy", "drawdown", "execution", "concentration") if sec.get(k)]}


def sizing_story(run: dict, res: dict, trades: list[dict], frame: dict | None) -> dict | None:
    """What the position sizing did, with four amounts kept apart: the premium paid, the most each trade could have lost, the
    loss the stop was meant to allow, and what was actually lost. Every figure is from the run's own trades."""
    cfg = ((res.get("config") or {}).get("strategies") or [{}])[0].get("sizing") or ((run.get("config") or {}).get("strategies") or [{}])[0].get("sizing") or {}
    method = cfg.get("method") or "fixed"
    if not trades:
        return None
    units = [int(t.get("units") or 1) for t in trades]
    debit = [max(0.0, -float(t.get("entry_cash") or 0)) for t in trades]
    maxloss = [float(t["max_loss"]) if t.get("max_loss") is not None else d for t, d in zip(trades, debit)]
    losers = [t for t in trades if t["pnl"] < 0]
    stop = ((res.get("config") or {}).get("strategies") or [{}])[0].get("exits", {}).get("stop_loss") or {}
    stop_pct = float(stop.get("value")) if stop.get("enabled") and stop.get("mode") == "pct" else None
    lines = []
    budget = {"premium_dollars": f"${cfg.get('dollars', 0):,.0f} of premium", "premium_pct": f"{cfg.get('pct', 0):g}% of the account in premium",
              "risk_dollars": f"${cfg.get('dollars', 0):,.0f} of risk", "risk_pct": f"{cfg.get('pct', 0):g}% of the account at risk"}.get(method)
    if budget:
        lines.append(f"Sized at {budget} per trade, positions ranged from {min(units)} to {max(units)} contract{'s' if max(units) != 1 else ''} (median {sorted(units)[len(units) // 2]}).")
    dp = sorted(debit)
    if any(dp):
        lines.append(f"Premium paid per trade: median {_usd(dp[len(dp) // 2]).lstrip('+')}, largest {_usd(dp[-1]).lstrip('+')}. For a bought option that premium is also the most the trade could lose.")
    if stop_pct is not None and any(dp):
        planned = [d * stop_pct / 100 for d in debit]
        over = [t for t, p in zip(trades, planned) if t["pnl"] < 0 and -t["pnl"] > p * 1.05 + float(t.get("commissions") or 0)]
        lines.append(f"The {stop_pct:g}% stop meant to limit a loss to about {_usd(sorted(planned)[len(planned) // 2]).lstrip('+')} per trade (median); the price jumped past it on {len(over)} of {len(losers)} losing trades, which lost more than planned.")
    if losers:
        lw = sorted(-t["pnl"] for t in losers)
        lines.append(f"Realised losses: median {_usd(lw[len(lw) // 2]).lstrip('+')}, worst {_usd(lw[-1]).lstrip('+')}, on {len(losers)} losing trades.")
    skipped = 0
    for s_ in ((res.get("diagnostics") or {}).get("skipped_signals") or res.get("skipped_signals") or []):
        why = str(s_.get("reason") or s_.get("why") or "") if isinstance(s_, dict) else str(s_)
        if "budget" in why or "daily loss stop" in why:
            skipped += 1
    if skipped:
        lines.append(f"{skipped} signal{'s were' if skipped != 1 else ' was'} not traded because of the sizing rules (one contract cost more than the budget, or the daily loss stop was reached).")
    if cfg.get("daily_loss_stop"):
        lines.append(f"No new trade was opened on a day once closed trades had lost ${cfg['daily_loss_stop']:,.0f}.")
    if method == "fixed" and len(lines) <= 2:
        return {"lines": lines, "method": method}
    return {"lines": lines, "method": method, "units": {"min": min(units), "max": max(units)}, "premium": {"median": dp[len(dp) // 2], "max": dp[-1]},
            "max_loss": {"max": max(maxloss)}, "realized_loss": {"median": (sorted(-t["pnl"] for t in losers)[len(losers) // 2] if losers else 0)}}


def _option_span(run: dict, res: dict) -> str | None:
    """When option profits CAN be tested for this symbol, in one sentence ("from January 2024 onward")."""
    try:
        from strata.data.catalog import catalog
        from strata.data.store import MarketStore
        sym = ((res.get("config") or {}).get("strategies") or [{}])[0].get("symbol") or (run.get("config") or {}).get("symbol")
        c = next((x for x in catalog(MarketStore()) if x["symbol"] == sym), None)
        tr = (c or {}).get("option_trades")
        if not tr:
            return None
        return f"Real historical option trades for {sym} are stored from {_month(tr['first'])} to {_month(tr['last'])}, so option profits can be tested only in that span."
    except Exception:  # noqa: BLE001
        return None


def coverage_sentence(cov: dict) -> str:
    """Which days could be tested, in one sentence a person can act on."""
    sig = cov.get("signals") or {}
    exe = cov.get("executable") or {}
    q, p = sig.get("qualifying"), exe.get("signals")
    win = exe.get("window") or [None, None]
    if q is None:
        return "These results use real historical option trades, available from January 18, 2024."
    if p is not None and p < q:
        return f"Your rule occurred on {_n(q)} days, but the option prices needed exist for {_n(p)} of them; the others are not counted as wins or losses."
    return f"Your rule occurred on {_n(q)} days{(' from ' + _date(win[0])) if win and win[0] else ''}, and all {_n(q)} could be tested on real option trades."


def example_trade(trades: list[dict], run_id: str) -> dict | None:
    """One actual trade, told step by step: the typical winner if there is one, else the typical trade."""
    if not trades:
        return None
    wins = sorted([t for t in trades if t["pnl"] > 0], key=lambda t: t["pnl"])
    pick = wins[len(wins) // 2] if wins else sorted(trades, key=lambda t: t["pnl"])[len(trades) // 2]
    return trade_story(pick, run_id)


def trade_story(t: dict, run_id: str | None = None) -> dict:
    """An option trade in plain words, from its own record: contract, why it was chosen, cost, what the underlying and the
    premium did, why it exited, and the money. The contract multiplier is the trade's own."""
    legs = [l for l in t.get("legs") or [] if l.get("kind", "option") == "option"]
    lines = []
    if not legs:
        return {"trade": t.get("id"), "lines": [], "run_id": run_id}
    units = int(t.get("units") or 1)
    for l in legs:
        qty = abs(int(l.get("qty") or l.get("quantity") or 1)) * units
        mult = float(l.get("multiplier") or 100)
        side = "bought" if (l.get("side") or "buy") == "buy" or float(l.get("qty") or 1) > 0 else "sold"
        right = "call" if (l.get("right") or "C") == "C" else "put"
        ep, xp = float(l["entry_price"]), float(l["exit_price"])
        lines.append(f"On {_date(t['entry_date'])} at {str(t.get('entry_time') or '')[11:16]} you {side} {qty} {t['symbol']} {l['strike']:g} {right}{'s' if qty != 1 else ''} expiring {_date(l.get('expiration'))}, "
                     f"for ${ep:.2f} per share. One contract covers {mult:g} shares, so it cost ${ep * mult * qty:,.2f} before fees.")
    spot0, spot1 = t.get("entry_spot"), t.get("exit_spot")
    if spot0 and spot1:
        mv = (spot1 / spot0 - 1) * 100
        lines.append(f"{t['symbol']} was at ${spot0:,.2f} then and ${spot1:,.2f} when the position closed ({mv:+.2f}%).")
    l0 = legs[0]
    ep, xp = float(l0["entry_price"]), float(l0["exit_price"])
    pm = (xp / ep - 1) * 100 if ep else 0.0
    lines.append(f"The option's price went from ${ep:.2f} to ${xp:.2f} ({pm:+.1f}% on the premium).")
    if spot0 and spot1 and abs(mv) > 0.05:
        lines.append(f"Note the scale: a {abs(mv):.2f}% move in {t['symbol']} became a {abs(pm):.0f}% move in the option. The two are not the same percentage, and the option can lose value even when {t['symbol']} moves the right way, because its time value runs out.")
    why = {"time_exit": "the planned exit time arrived", "profit_target": "the profit target was reached", "stop_loss": "the stop was hit", "expiration": "the contract expired",
           "max_hold": "the planned holding time ran out", "end_of_backtest": "the test period ended", "underlying_target": f"{t['symbol']} reached the target move",
           "underlying_stop": f"{t['symbol']} hit the stop level", "custom_exit": "an exit rule fired", "exercised": "it expired in the money and was settled",
           "expired_worthless": "it expired worthless", "trailing_stop": "the trailing stop was hit"}.get(t.get("exit_reason") or "", (t.get("exit_reason") or "").replace("_", " "))
    gross, fees = float(t.get("gross_pnl") or t["pnl"]), float(t.get("commissions") or 0)
    lines.append(f"It closed because {why}. Result: {_usd(gross)} before fees, {_usd(float(t['pnl']))} after ${fees:.2f} in fees.")
    return {"trade": t["id"], "date": t["entry_date"], "pnl": float(t["pnl"]), "lines": lines, "run_id": run_id}


def why_lost(t: dict, run_id: str | None = None) -> dict:
    """Why one trade lost: the underlying's move against the option's, from the trade's own prices. Intrinsic value is what
    the option would be worth if it expired at that moment; the rest of its price is time value, which falls as expiry nears."""
    story = trade_story(t, run_id)
    legs = [l for l in t.get("legs") or [] if l.get("kind", "option") == "option"]
    if not legs:
        return story
    l0 = legs[0]
    k, right = float(l0["strike"]), l0.get("right") or "C"
    s0, s1 = t.get("entry_spot"), t.get("exit_spot")
    ep, xp = float(l0["entry_price"]), float(l0["exit_price"])
    if s0 and s1:
        intr = (lambda s: max(s - k, 0.0)) if right == "C" else (lambda s: max(k - s, 0.0))
        i0, i1 = intr(s0), intr(s1)
        e0, e1 = ep - i0, max(xp - i1, 0.0)
        wrong_way = (s1 < s0) if right == "C" else (s1 > s0)
        cause = []
        kind = "call" if right == "C" else "put"
        mv = (s1 / s0 - 1) * 100
        if wrong_way:
            # the price moved the wrong way: that move is the cause. The fall in time value is part of it (a contract further
            # from its strike, closer to expiry, is worth less as a chance), so the two are not split into separate causes
            cause.append(f"{t['symbol']} moved against the position: {'down' if right == 'C' else 'up'} {abs(mv):.2f}%, from ${s0:,.2f} to ${s1:,.2f}.")
            cause.append(f"At entry the {k:g} {kind} cost ${ep:.2f}" + (f", of which ${e0:.2f} was time value: the price of the chance that {t['symbol']} would be {'above' if right == 'C' else 'below'} ${k:g} before it expired."
                                                                       if i0 < 0.5 * ep else f": ${i0:.2f} intrinsic value and ${e0:.2f} time value."))
            cause.append(f"With {t['symbol']} moving away from the strike and less time left, that chance shrank: at the exit the {kind} was worth ${xp:.2f}"
                         + (f" ({(xp / ep - 1) * 100:+.0f}%)." if ep else "."))
        else:
            cause.append(f"At entry the {k:g} {kind} was worth ${ep:.2f}: ${i0:.2f} of it intrinsic value (what it would be worth if it expired then) and ${e0:.2f} time value. "
                         f"At the exit it was worth ${xp:.2f}: ${i1:.2f} intrinsic and ${e1:.2f} time value.")
            if e0 - e1 > abs(i1 - i0):
                cause.append(f"So {t['symbol']} moved the right way, but not far or fast enough: the option gained ${i1 - i0:.2f} of intrinsic value and lost ${e0 - e1:.2f} of time value as expiry came closer.")
        story["cause"] = cause
    return story


# ------------------------------------------------------------------------------------------------- how reliable is it
def reliability(rob: dict, trials: int, run_id: str | None = None) -> dict:
    """"Is this actually good, or did it just get lucky?" in plain words, from the robustness checks of the result."""
    level = rob.get("level")
    head = {"insufficient": "There are too few trades or days to tell skill from luck.",
            "none": "It may just have been luck: the average result is not reliably different from zero.",
            "mixed": "Partly: the average result is above zero over the whole period, but it does not hold up on every check.",
            "supported": "The result holds up on every check here, which is as much as a backtest can say. It is still past data, not a forecast."}.get(level, rob.get("verdict") or "")
    plain = {"Sample size": "Enough trades", "Average trade different from zero": "Average result reliably above zero", "Holds in later data": "Still works on the later part of the data",
             "Not carried by a few trades": "Not carried by a few lucky trades", "Survives worse fills": "Survives slightly worse prices", "Holds in later sessions": "Still works on later days"}
    happened = []
    for c in rob.get("checks") or []:
        happened.append(f"{'Yes' if c['ok'] else 'No'} - {plain.get(c['name'], c['name'])}: {c['detail']}.")
    limits = []
    if trials > 1:
        limits.append(f"We tried {trials} variations in this conversation. Some were bound to look good even if none had a genuine advantage, so the significance test is adjusted for all of them.")
    limits.append("The p-value in the details is how often a result at least this good would appear if the strategy had no edge. It is not the probability that the strategy works.")
    return {"kind": "reliability", "headline": head, "happened": happened, "limits": limits, "numbers": [], "level": level,
            "followups": [{"text": "Show me the losing trades", "why": "The trades behind the worst part of the result."}, {"text": "What happens with worse fills?", "why": "The same trades at worse prices."}]}


# ------------------------------------------------------------------------------------------------- comparing results
def _streak(trades: list[dict]) -> int:
    best = cur = 0
    for t in sorted(trades, key=lambda t: t["entry_ts"]):
        cur = cur + 1 if t["pnl"] <= 0 else 0
        best = max(best, cur)
    return best


def _short(label: str) -> str:
    import re as _re
    return _re.sub(r"^v\d+ - ", "", label or "").strip()[:60] or "this version"


def _differ(x: str, y: str) -> tuple[str, str]:
    """The clauses of two descriptions that differ: ("when the previous session fell 3.8% or more", "when ... 3% or more")."""
    import re as _re
    ca = [c.strip(" .") for c in _re.split(r"[,:;]", x or "") if c.strip(" .")]
    cb = [c.strip(" .") for c in _re.split(r"[,:;]", y or "") if c.strip(" .")]
    da = [c for c in ca if c not in cb]
    db_ = [c for c in cb if c not in ca]
    if da and db_:
        return " and ".join(da), " and ".join(db_)
    return x, y


def _with(clause: str) -> str:
    """A clause as the start of a sentence: "when ..." and "on SPY" read as they are; anything else gets "with"."""
    c = clause.strip()
    return c if c.lower().startswith(("when ", "on ", "after ", "in ", "from ", "only ")) else "with " + c


def _clock_et(t) -> str:
    if t in (None, ""):
        return "no set time"
    if t in ("open", "close"):
        return f"the {t}"
    h, m = int(str(t)[:2]), int(str(t)[3:5])
    return f"{(h + 11) % 12 + 1}:{m:02d} {'AM' if h < 12 else 'PM'} ET"


def draft_phrases(da: dict, db_: dict) -> tuple[str, str] | None:
    """What tells two strategy versions apart, as two short phrases: ("exit at 3:55 PM ET", "exit at 3:45 PM ET")."""
    if not isinstance(da, dict) or not isinstance(db_, dict):
        return None
    pa, pb = [], []
    show = {"symbol": lambda v: str(v), "structure": lambda v: {"long_call": "calls", "long_put": "puts", "short_call": "short calls", "short_put": "short puts"}.get(v, str(v)),
            "dte": lambda v: f"{v}DTE", "entry_time": lambda v: f"entry at {_clock_et(v)}", "exit_time": lambda v: f"exit at {_clock_et(v)}",
            "premium_target_pct": lambda v: "no profit target" if v is None else f"a {v:g}% profit target", "premium_stop_pct": lambda v: "no stop" if v is None else f"a {v:g}% stop",
            "underlying_target_pct": lambda v: "no underlying target" if v is None else f"exit on a {v:g}% move", "quantity": lambda v: f"{v} contract{'s' if v != 1 else ''}",
            "hold_minutes": lambda v: "no holding limit" if v is None else f"held {v} minutes"}
    for k, f in show.items():
        if da.get(k) != db_.get(k):
            pa.append(f(da.get(k)))
            pb.append(f(db_.get(k)))
    ca = [c.get("text") or c.get("expr") for c in da.get("conditions") or []]
    cb = [c.get("text") or c.get("expr") for c in db_.get("conditions") or []]
    if ca != cb:
        only_a, only_b = [c for c in ca if c not in cb], [c for c in cb if c not in ca]
        pa.append("only when " + " and ".join(only_a) if only_a else "without " + " and ".join(only_b))
        pb.append("only when " + " and ".join(only_b) if only_b else "without " + " and ".join(only_a))
    if not pa:
        return None
    return ", ".join(pa), ", ".join(pb)


def _names(a: dict, b: dict, desc: dict | None = None) -> tuple[str, str]:
    desc = desc or {}
    ra_, rb_ = desc.get(a["run"]["id"]), desc.get(b["run"]["id"])
    if isinstance(ra_, dict) and isinstance(rb_, dict):
        if a["run"]["kind"] != "study" or not (ra_.get("desc") and rb_.get("desc") and ra_["desc"] != rb_["desc"]):
            ph = draft_phrases(ra_.get("draft"), rb_.get("draft"))
            if ph:
                return ph
        ra_, rb_ = ra_.get("desc"), rb_.get("desc")
        desc = {a["run"]["id"]: ra_, b["run"]["id"]: rb_}
    ra, rb = desc.get(a["run"]["id"]) or a["run"].get("name") or a["label"], desc.get(b["run"]["id"]) or b["run"].get("name") or b["label"]
    if ra and rb and ra != rb:
        da, db_ = _differ(ra, rb)
        if len(da) < 80 and len(db_) < 80:
            return da, db_
    return _short(a["label"]), _short(b["label"])


def _days(n: int) -> str:
    return f"{_n(n)} day{'s' if n != 1 else ''}"


def compare(entries: list[dict], desc: dict | None = None) -> dict:
    """Two results side by side, in sentences: what differs, by how much, and whether the difference is beyond chance."""
    from strata.api.results import load_trades
    a, b = entries[-1], entries[0]  # the newest is "this version", the first is what it is compared with
    na, nb = _names(a, b, desc)
    if a["run"]["kind"] == "study" and b["run"]["kind"] == "study":
        sa, sb = a["result"].get("stats") or {}, b["result"].get("stats") or {}
        pa, pb, qa, qb = sa.get("probability"), sb.get("probability"), sa.get("qualifying_sessions") or 0, sb.get("qualifying_sessions") or 0
        if pa is None or pb is None:
            return {"kind": "compare", "headline": "One of the two results has no frequency to compare.", "happened": [], "limits": [], "numbers": []}
        ka, kb = round(pa * qa), round(pb * qb)
        import math
        pool = (ka + kb) / (qa + qb) if qa + qb else 0
        se = math.sqrt(pool * (1 - pool) * (1 / max(qa, 1) + 1 / max(qb, 1))) if 0 < pool < 1 else 0
        z = (pa - pb) / se if se else 0
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z)))) if se else 1.0
        more = na if pa > pb else nb
        hi, lo = (na, nb) if more == na else (nb, na)
        head = (f"{_with(hi)[0].upper() + _with(hi)[1:]}, it happened more often: {_pct((pa if more == na else pb) * 100)} of {_days(qa if more == na else qb)}, "
                f"against {_pct((pb if more == na else pa) * 100)} of {_days(qb if more == na else qa)} {_with(lo)}.") if pa != pb else \
               f"Both happened equally often ({_pct(pa * 100)}): {_days(qa)} {_with(na)}, {_days(qb)} {_with(nb)}."
        happened = []
        happened.append("A difference this size is within what chance produces with these sample sizes." if p > 0.05 else
                        "A difference this size would be unusual by chance alone with these sample sizes, although both are measured on the same history.")
        lim = []
        if min(qa, qb) < 30:
            lim.append(f"{min(qa, qb)} days is a small sample for one of the two.")
        return {"kind": "compare", "headline": head, "happened": happened, "limits": lim, "numbers": [
            {"label": na, "value": _pct(pa * 100), "detail": f"{_n(qa)} days", "explore": {"kind": "sessions", "run_id": a["run"]["id"], "filter": "all"}},
            {"label": nb, "value": _pct(pb * 100), "detail": f"{_n(qb)} days", "explore": {"kind": "sessions", "run_id": b["run"]["id"], "filter": "all"}}],
            "details": [{"label": "p-value of the difference", "value": f"{p:.3f}"}], "followups": []}
    ta, tb = load_trades(a["run"]["id"]), load_trades(b["run"]["id"])
    if not ta or not tb:
        return {"kind": "compare", "headline": "One of the two made no trades, so there is little to compare.", "happened": [], "limits": [], "numbers": []}
    neta, netb = sum(t["pnl"] for t in ta), sum(t["pnl"] for t in tb)
    sa, sb = _streak(ta), _streak(tb)
    first = lambda ts: min(t["entry_date"] for t in ts)
    last = lambda ts: max(t["entry_date"] for t in ts)
    lead, lag, nlead, nlag = (ta, tb, na, nb) if neta >= netb else (tb, ta, nb, na)
    nl, ng = sum(t["pnl"] for t in lead), sum(t["pnl"] for t in lag)
    if abs(nl - ng) < 0.005:
        parts = [f"Both made the same: {_usd(nl)} after fees, {_with(na)} and {_with(nb)}" + (" (the change did not alter any trade's result)." if len(ta) == len(tb) else ".")]
    else:
        parts = [f"{_with(nlead)[0].upper() + _with(nlead)[1:]}, it made more money overall: {_usd(nl)} after fees, against {_usd(ng)} {_with(nlag)}."]
    if len(lag) < len(lead):
        parts.append(f"The other took fewer trades ({_n(len(lag))} against {_n(len(lead))})" + (f" and had a shorter worst losing streak ({_streak(lag)} against {_streak(lead)} losses in a row)." if _streak(lag) < _streak(lead) else "."))
    elif _streak(lag) < _streak(lead):
        parts.append(f"The other had a shorter worst losing streak ({_streak(lag)} against {_streak(lead)} losses in a row).")
    cs = lambda e: next((r.get("net") for r in ((e["result"].get("cost_sensitivity") or {}).get("rows") or []) if abs((r.get("worse_fills_pct") or 0) - 2) < 1e-9), None)
    w_lead, w_lag = cs(a if nlead == na else b), cs(b if nlead == na else a)
    if w_lead is not None and w_lag is not None and w_lead <= w_lag:
        parts.append(f"Its advantage disappears when fills are 2% worse: {_usd(w_lead)} against {_usd(w_lag)}.")
    try:
        from scipy.stats import ttest_ind
        p = float(ttest_ind([t["pnl"] for t in ta], [t["pnl"] for t in tb], equal_var=False).pvalue)
    except Exception:  # noqa: BLE001
        p = None
    happened = []
    if p is not None:
        happened.append(("The average trade of the two is not reliably different: single trades vary too much for this many trades to separate them." if p > 0.05 else
                         "The difference in the average trade is larger than single-trade variation usually produces, on this one history."))
    limits = []
    if (first(ta), last(ta)) != (first(tb), last(tb)):
        limits.append(f"They cover different days ({_month(first(ta))}-{_month(last(ta))} against {_month(first(tb))}-{_month(last(tb))}), so part of the difference is the dates, not the rule.")
    limits.append("Both are one contract per trade, with the same fees per contract.")
    return {"kind": "compare", "headline": " ".join(parts), "happened": happened, "limits": limits,
            "numbers": [{"label": na, "value": _usd(neta), "detail": f"{_n(len(ta))} trades", "explore": {"kind": "trades", "run_id": a["run"]["id"], "filter": "all"}},
                        {"label": nb, "value": _usd(netb), "detail": f"{_n(len(tb))} trades", "explore": {"kind": "trades", "run_id": b["run"]["id"], "filter": "all"}}],
            "viz": {"kind": "compare_strategies", "series": [{"label": na, "run_id": a["run"]["id"], "trades": _curve(ta)}, {"label": nb, "run_id": b["run"]["id"], "trades": _curve(tb)}]},
            "details": [{"label": "p-value of the difference in average trade", "value": "n/a" if p is None else f"{p:.3f}"}], "followups": [{"text": "Is the difference just luck?", "why": "The checks on the newer version."}]}


def _curve(trades: list[dict]) -> list[dict]:
    out, s = [], 0.0
    for t in sorted(trades, key=lambda t: t["exit_ts"] or t["entry_ts"]):
        s += float(t["pnl"])
        out.append({"date": t["entry_date"], "total": round(s, 2), "pnl": round(float(t["pnl"]), 2), "id": t["id"]})
    return out


def sweep(entries: list[dict], field: str | None) -> dict:
    """One setting tried at several values: which did best, and whether nearby values did about as well (a consistent
    region) or one value stands alone (more likely luck)."""
    from strata.api.results import load_trades
    rows = []
    for e in entries:
        if e["run"]["kind"] == "study":
            st = e["result"].get("stats") or {}
            rows.append({"label": _short(e["label"]), "value": st.get("probability"), "n": st.get("qualifying_sessions") or 0, "run_id": e["run"]["id"], "study": True})
        else:
            tr = load_trades(e["run"]["id"])
            rows.append({"label": _short(e["label"]), "value": sum(t["pnl"] for t in tr) if tr else None, "n": len(tr), "run_id": e["run"]["id"], "study": False})
    ok = [r for r in rows if r["value"] is not None]
    if len(ok) < 2:
        return {"kind": "sweep", "headline": "Fewer than two of the settings produced a result, so there is nothing to compare.", "happened": [], "limits": [], "numbers": []}
    study = ok[0]["study"]
    fmt = (lambda v: _pct(v * 100)) if study else _usd
    best_i = max(range(len(ok)), key=lambda i: ok[i]["value"])
    best = ok[best_i]
    nb = [ok[j] for j in (best_i - 1, best_i + 1) if 0 <= j < len(ok)]
    pos = [r for r in ok if (r["value"] > 0 if not study else True)]
    if study:
        spread = max(r["value"] for r in ok) - min(r["value"] for r in ok)
        consistent = all(abs(r["value"] - best["value"]) < 0.05 for r in nb)
    else:
        consistent = all(r["value"] > 0 and r["value"] >= 0.5 * best["value"] for r in nb) if best["value"] > 0 else False
    head = f"{best['label']} did best ({fmt(best['value'])}, {_n(best['n'])} {'days' if study else 'trades'})."
    happened = ["The settings next to it did about as well, so the result is not resting on one lucky value." if consistent else
                "The settings next to it did noticeably worse, so the best result stands alone. A lone peak is more often luck than a real effect."]
    if not study:
        happened.append(f"{len(pos)} of {len(ok)} settings made money.")
    limits = [f"{len(ok)} settings were tried: the best of several always looks better than it would on new data."]
    return {"kind": "sweep", "headline": head, "happened": happened, "limits": limits, "consistent": consistent,
            "numbers": [{"label": r["label"], "value": fmt(r["value"]), "detail": f"{_n(r['n'])} {'days' if study else 'trades'}",
                         "explore": {"kind": "sessions" if study else "trades", "run_id": r["run_id"], "filter": "all"}} for r in ok[:12]],
            "viz": {"kind": "sweep", "field": field, "rows": [{"label": r["label"], "value": r["value"], "n": r["n"], "run_id": r["run_id"]} for r in ok], "study": study, "best": best_i}}
