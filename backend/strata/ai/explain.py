"""Grounded explanations of results.

Every number in an explanation comes from `facts`: values read straight out of a stored
backtest or study result. The default narrative is assembled deterministically from those
facts. When a language model is configured it may rephrase the same facts, but its text is
checked token by token: any number that does not match a fact (or a parameter of the
request) causes the model's text to be discarded in favour of the deterministic one.
"""
from __future__ import annotations

import json
import math
import re

FRACTION_KEYS = {"win_rate", "total_return", "cagr", "max_drawdown", "volatility", "exposure", "benchmark_return", "avg_capital_utilization"}


def _fin(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def show(v, unit: str) -> str:
    if not _fin(v):
        return "n/a"
    if unit == "prob":
        return f"{v * 100:.1f}%"
    if unit == "pct":
        return f"{v:+.2f}%"
    if unit == "usd":
        return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    if unit == "count":
        return f"{int(round(v)):,}"
    if unit == "min":
        return f"{v:.0f} min"
    if unit == "p":
        return "< 0.001" if v < 0.001 else f"{v:.3f}"
    return f"{v:.2f}"


def _fact(out: list, key: str, label: str, value, unit: str):
    if _fin(value):
        out.append({"key": key, "label": label, "value": float(value), "unit": unit, "shown": show(value, unit)})


# ----------------------------------------------------------------------------- facts
def study_facts(res: dict) -> list[dict]:
    s, f = res["stats"], []
    _fact(f, "qualifying_sessions", "sessions meeting the condition", s.get("qualifying_sessions"), "count")
    _fact(f, "evaluated_sessions", "sessions evaluated in total", s.get("evaluated_sessions"), "count")
    _fact(f, "excluded_sessions", "sessions excluded (missing data)", s.get("excluded_sessions"), "count")
    _fact(f, "target_first", "sessions where the target was reached first", s.get("target_first"), "count")
    _fact(f, "stop_first", "sessions where the stop was reached first", s.get("stop_first"), "count")
    _fact(f, "neither", "sessions where neither level was reached", s.get("neither"), "count")
    _fact(f, "ambiguous", "ambiguous sessions (target and stop inside one bar)", s.get("ambiguous"), "count")
    _fact(f, "probability", "historical frequency of the target", s.get("probability"), "prob")
    ci = s.get("probability_ci95") or [None, None]
    _fact(f, "probability_ci_low", "95% confidence interval, low", ci[0], "prob")
    _fact(f, "probability_ci_high", "95% confidence interval, high", ci[1], "prob")
    pb = s.get("probability_bounds") or [None, None]
    if s.get("ambiguous"):
        _fact(f, "probability_bound_low", "frequency if every ambiguous session failed", pb[0], "prob")
        _fact(f, "probability_bound_high", "frequency if every ambiguous session succeeded", pb[1], "prob")
    _fact(f, "baseline_probability", "frequency on all evaluated sessions (baseline)", s.get("baseline_probability"), "prob")
    _fact(f, "lift", "difference versus baseline", s.get("lift"), "prob")
    _fact(f, "p_value", "p-value versus the remaining sessions", s.get("p_value_vs_rest"), "p")
    _fact(f, "avg_mfe_pct", "average best move after entry", s.get("avg_mfe_pct"), "pct")
    _fact(f, "avg_mae_pct", "average worst move after entry", s.get("avg_mae_pct"), "pct")
    _fact(f, "avg_end_return_pct", "average return at the end of the horizon", s.get("avg_end_return_pct"), "pct")
    _fact(f, "median_end_return_pct", "median return at the end of the horizon", s.get("median_end_return_pct"), "pct")
    _fact(f, "pct_positive_end", "share of sessions ending higher", s.get("pct_positive_end"), "prob")
    _fact(f, "median_minutes_to_target", "median time to reach the target", s.get("median_minutes_to_target"), "min")
    return f


def backtest_facts(res: dict) -> list[dict]:
    m, d, f = res.get("metrics") or {}, res.get("diagnostics") or {}, []
    _fact(f, "trades", "trades", m.get("trades"), "count")
    _fact(f, "net_pnl", "net profit after costs", m.get("net_pnl"), "usd")
    _fact(f, "total_return", "total return on starting capital", m.get("total_return"), "prob")
    _fact(f, "win_rate", "share of winning trades", m.get("win_rate"), "prob")
    _fact(f, "avg_winner", "average winning trade", m.get("avg_winner"), "usd")
    _fact(f, "avg_loser", "average losing trade", m.get("avg_loser"), "usd")
    _fact(f, "expectancy", "average profit per trade", m.get("expectancy"), "usd")
    _fact(f, "profit_factor", "profit factor", m.get("profit_factor"), "ratio")
    _fact(f, "largest_winner", "largest winner", m.get("largest_winner"), "usd")
    _fact(f, "largest_loser", "largest loser", m.get("largest_loser"), "usd")
    _fact(f, "max_drawdown", "maximum drawdown", m.get("max_drawdown"), "prob")
    _fact(f, "max_drawdown_dollars", "maximum drawdown in dollars", m.get("max_drawdown_dollars"), "usd")
    _fact(f, "sharpe", "Sharpe ratio", m.get("sharpe"), "ratio")
    _fact(f, "t_stat", "t-statistic of the per-trade profit", m.get("t_stat"), "ratio")
    _fact(f, "total_commissions", "commissions paid", m.get("total_commissions"), "usd")
    _fact(f, "total_slippage", "slippage paid", m.get("total_slippage"), "usd")
    _fact(f, "avg_holding_minutes", "average holding time", m.get("avg_holding_minutes"), "min")
    _fact(f, "sessions", "sessions simulated", m.get("sessions"), "count")
    _fact(f, "expired_orders", "orders that expired without a fresh option quote", len(d.get("expired_orders") or []) if isinstance(d.get("expired_orders"), list) else d.get("expired_orders"), "count")
    ex = d.get("excluded_sessions")
    _fact(f, "excluded_sessions", "sessions excluded because a rule could not be evaluated", (sum(len(v) for v in ex.values()) if isinstance(ex, dict) else len(ex)) if ex else 0, "count")
    _fact(f, "deferred_exits", "exits delayed to the next available quote", d.get("deferred_exits"), "count")
    _fact(f, "stale_marks", "marks using a stale quote", d.get("stale_marks"), "count")
    return f


def fget(facts: list[dict], key: str):
    return next((x for x in facts if x["key"] == key), None)


def fv(facts: list[dict], key: str):
    x = fget(facts, key)
    return x["value"] if x else None


def fs(facts: list[dict], key: str) -> str:
    x = fget(facts, key)
    return x["shown"] if x else "n/a"


# ----------------------------------------------------------------------------- caveats
def study_caveats(res: dict) -> list[str]:
    out = list(res.get("notes") or [])
    for e in (res.get("excluded") or [])[:3]:
        out.append(f"{e['count']} session(s) excluded: {e['reason']}.")
    if res.get("synthetic"):
        out.append("SIMULATED data: this symbol is synthetic sandbox data, not real market history.")
    return out


def backtest_caveats(res: dict) -> list[str]:
    meta, d, m = res.get("meta") or {}, res.get("diagnostics") or {}, res.get("metrics") or {}
    out = []
    if meta.get("pricing") == "modeled":
        out.append("MODELED premiums: option prices are theoretical estimates from a pricing model, not observed market prices. " + " ".join(meta.get("pricing_assumptions") or [])[:400])
    for sym, r in (meta.get("data_resolution") or {}).items():
        mix = ", ".join(f"{n} sessions at {k}" for k, n in (r.get("underlying_sessions_by_resolution") or {}).items())
        out.append(f"{sym}: option data = {r.get('options')}; underlying = {mix}.")
        if r.get("trade_based"):
            out.append("Some fills used last-trade prices (no bid/ask was available); no spread was invented for them.")
    if meta.get("synthetic"):
        out.append("SIMULATED data: at least one symbol is synthetic sandbox data, not real market history.")
    n = m.get("trades") or 0
    if n and n < 30:
        out.append(f"Only {int(n)} trades: too few for the averages to be reliable.")
    ex = d.get("excluded_sessions")
    if ex:
        items = ex.items() if isinstance(ex, dict) else []
        for reason, days in list(items)[:3]:
            out.append(f"{len(days)} session(s) excluded: {reason}.")
    eo = d.get("expired_orders")
    if eo:
        k = len(eo) if isinstance(eo, list) else eo
        out.append(f"{k} entry signal(s) expired because no fresh option quote was available within the allowed delay; they were not filled late.")
    for w in (d.get("warnings") or [])[:3]:
        out.append(str(w))
    return out


# ----------------------------------------------------------------------------- narratives
def _level_words(cfg: dict) -> str:
    t, meas = cfg.get("target"), cfg.get("measure", "touch")
    if not t:
        return "no target was set"
    if t.get("kind") == "expr":
        return f"price reached {t.get('expr')}"
    v, unit = t.get("value", 0), "%" if t.get("kind") == "pct" else " points"
    ref = {"entry": "the starting price", "open": "the open", "prev_close": "the previous close"}[cfg.get("reference", "entry")]
    if meas == "close":
        side = "below" if (cfg.get("direction") == "down" or v < 0) else "above"
        return f"the horizon ended {side} {ref}" if v == 0 else f"the horizon ended at least {abs(v):g}{unit} {side} {ref}"
    return f"price moved {'up' if v >= 0 else 'down'} {abs(v):g}{unit} from {ref}"


def study_text(res: dict, facts: list[dict]) -> str:
    cfg = res["config"]
    n = fv(facts, "qualifying_sessions") or 0
    period = f"{res['period'][0]} to {res['period'][1]}"
    if not n:
        return (f"No session between {period} met the condition, so there is nothing to measure. "
                f"{fs(facts, 'evaluated_sessions')} sessions were evaluated in total and {fs(facts, 'excluded_sessions')} were excluded for missing data.")
    parts = []
    if cfg.get("target"):
        hit = fs(facts, "target_first")
        parts.append(f"On {res['symbol']}, {hit} of the {fs(facts, 'qualifying_sessions')} sessions that met the condition ({period}) saw the outcome - {_level_words(cfg)} - "
                     f"a historical frequency of {fs(facts, 'probability')} (95% confidence interval {fs(facts, 'probability_ci_low')} to {fs(facts, 'probability_ci_high')}).")
        if cfg.get("stop"):
            parts.append(f"The stop was reached first on {fs(facts, 'stop_first')} sessions and neither level on {fs(facts, 'neither')}.")
        if fv(facts, "ambiguous"):
            parts.append(f"{fs(facts, 'ambiguous')} sessions are ambiguous because target and stop fell inside the same price bar; depending on their true order the frequency lies between {fs(facts, 'probability_bound_low')} and {fs(facts, 'probability_bound_high')}.")
        if fget(facts, "baseline_probability"):
            lift = fv(facts, "lift") or 0
            p = fv(facts, "p_value")
            sig = "" if p is None else (f"; the difference is statistically distinguishable from the other sessions (p {fs(facts, 'p_value')})" if p < 0.05 else f"; the difference is not statistically distinguishable from the other sessions (p {fs(facts, 'p_value')})")
            parts.append(f"Across all {fs(facts, 'evaluated_sessions')} evaluated sessions the same outcome occurred {fs(facts, 'baseline_probability')} of the time, so the condition {'raises' if lift >= 0 else 'lowers'} it by {show(abs(lift), 'prob').replace('%', ' percentage points')}{sig}.")
    else:
        parts.append(f"{fs(facts, 'qualifying_sessions')} {res['symbol']} sessions met the condition between {period}.")
    parts.append(f"After the starting point the average best move was {fs(facts, 'avg_mfe_pct')} and the average worst move {fs(facts, 'avg_mae_pct')}; the horizon ended {fs(facts, 'avg_end_return_pct')} on average and higher on {fs(facts, 'pct_positive_end')} of sessions.")
    if fget(facts, "median_minutes_to_target"):
        parts.append(f"Where intraday bars exist, the target was typically reached {fs(facts, 'median_minutes_to_target')} after the start.")
    parts.append("These are underlying price statistics, not option returns, and past frequency is not a forecast.")
    return " ".join(parts)


def backtest_text(res: dict, facts: list[dict]) -> str:
    cfg = res.get("config") or {}
    n = fv(facts, "trades") or 0
    period = f"{cfg.get('start')} to {cfg.get('end')}"
    modeled = (res.get("meta") or {}).get("pricing") == "modeled"
    kind = "Modeled-premium estimate" if modeled else "Historical options backtest"
    if not n:
        ex = fv(facts, "excluded_sessions") or 0
        eo = fv(facts, "expired_orders") or 0
        why = []
        if eo:
            why.append(f"{fs(facts, 'expired_orders')} entry signals expired without a usable option quote")
        if ex:
            why.append(f"{fs(facts, 'excluded_sessions')} sessions were excluded because a rule could not be evaluated")
        return f"{kind}, {period}: no trades were taken across {fs(facts, 'sessions')} sessions" + (" (" + "; ".join(why) + ")" if why else " - the entry conditions never held with tradable quotes") + "."
    pnl = fv(facts, "net_pnl") or 0
    parts = [f"{kind}, {period}: {fs(facts, 'trades')} trades made a net {'profit' if pnl >= 0 else 'loss'} of {show(abs(pnl), 'usd')} after {fs(facts, 'total_commissions')} in commissions"
             + (f" and {fs(facts, 'total_slippage')} of slippage" if (fv(facts, 'total_slippage') or 0) > 0 else "") + f" ({fs(facts, 'total_return')} on starting capital)."]
    parts.append(f"{fs(facts, 'win_rate')} of trades were winners; the average winner was {fs(facts, 'avg_winner')} against an average loser of {fs(facts, 'avg_loser')}, for {fs(facts, 'expectancy')} per trade"
                 + (f" and a profit factor of {fs(facts, 'profit_factor')}" if fget(facts, 'profit_factor') else "") + ".")
    parts.append(f"The largest winner was {fs(facts, 'largest_winner')} and the largest loser {fs(facts, 'largest_loser')}; the deepest drawdown was {fs(facts, 'max_drawdown')} of equity.")
    t = fv(facts, "t_stat")
    if t is not None:
        parts.append(f"The per-trade profit has a t-statistic of {fs(facts, 't_stat')}, which is {'above' if abs(t) >= 2 else 'below'} the conventional threshold of two, so the edge is {'statistically distinguishable from zero' if abs(t) >= 2 else 'not statistically distinguishable from zero'} in this sample.")
    if modeled:
        parts.append("Premiums are theoretical estimates, so treat every dollar figure as approximate.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- grounding check
NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def ungrounded_numbers(text: str, facts: list[dict], allowed: list[float] | None = None) -> list[str]:
    """Numbers in `text` that match no fact (in any of its displayed roundings) and no allowed literal."""
    cands: list[float] = [95.0, 100.0]
    for f in facts:
        v = f["value"]
        cands += [v, abs(v)]
        if f["unit"] == "prob":
            cands += [v * 100, abs(v) * 100]
        if f["unit"] == "min":
            cands += [v / 60]
        if f["unit"] == "p":
            cands += [0.001, 0.05]  # "p < 0.001" and the conventional 5% threshold are wording, not results
    cands += [float(a) for a in (allowed or [])]
    cands += [abs(float(a)) for a in (allowed or [])]
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)  # ISO dates
    t = re.sub(r"\b\d{1,2}:\d{2}\b", " ", t)  # clock times
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)  # years
    bad = []
    for tok in NUM_RE.findall(t):
        raw = tok.replace("$", "").replace(",", "").replace("%", "").lstrip("+")
        try:
            n = abs(float(raw))
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if not any(abs(round(abs(c), dec) - n) <= 10 ** -(dec + 6) + 1e-9 for c in cands):
            bad.append(tok)
    return bad


def _config_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _config_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _config_numbers(v)
    elif _fin(obj):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", obj)[:12]]
    return out


def explain_run(run: dict, result: dict, llm=None, question: str | None = None) -> dict:
    """Explanation for one finished run. `result` is the stored study / backtest payload."""
    is_study = run["kind"] == "study"
    facts = study_facts(result) if is_study else backtest_facts(result)
    caveats = study_caveats(result) if is_study else backtest_caveats(result)
    text = study_text(result, facts) if is_study else backtest_text(result, facts)
    out = {"text": text, "source": "deterministic", "facts": facts, "caveats": caveats, "rejected": None}
    if llm is None:
        return out
    from strata.ai.llm import LLMError
    system = ("You explain quantitative research results to a trader. Use ONLY the numbers listed under FACTS, copied exactly from their 'shown' text. "
              "Never calculate, estimate or round a new number, never give trading advice and never predict the future. Say plainly when the sample is small "
              "or the result is not statistically distinguishable from chance. Mention the most important caveats. Write 4 to 7 sentences of plain prose: no lists, no headings, no markdown.")
    payload = {"result_type": "underlying market study (not option returns)" if is_study else ("MODELED option backtest (theoretical premiums)" if (result.get("meta") or {}).get("pricing") == "modeled" else "historical options backtest"),
               "request": run.get("name"), "period": result.get("period") or [result.get("config", {}).get("start"), result.get("config", {}).get("end")],
               "FACTS": [{"what": f["label"], "shown": f["shown"]} for f in facts], "CAVEATS": caveats}
    if question:
        payload["user_question"] = question
    try:
        t = llm.text(system, [{"role": "user", "content": json.dumps(payload, indent=1)}], max_tokens=700).strip()
    except LLMError as e:
        out["rejected"] = f"model unavailable: {e}"
        return out
    bad = ungrounded_numbers(t, facts, _config_numbers(run.get("config")))
    if bad or len(t) < 40:
        out["rejected"] = "the model's text contained numbers that are not in the results (" + ", ".join(bad[:6]) + "), so the calculated summary is shown instead" if bad else "the model returned no usable text"
        return out
    out.update({"text": t, "source": llm.name})
    return out


# ----------------------------------------------------------------------------- comparisons
STUDY_COLS = [("qualifying_sessions", "Sessions", "count"), ("target_first", "Hits", "count"), ("probability", "Frequency", "prob"), ("probability_ci_low", "CI low", "prob"), ("probability_ci_high", "CI high", "prob"),
              ("baseline_probability", "Baseline", "prob"), ("avg_end_return_pct", "Avg end return", "pct"), ("ambiguous", "Ambiguous", "count")]
BT_COLS = [("trades", "Trades", "count"), ("net_pnl", "Net P&L", "usd"), ("win_rate", "Win rate", "prob"), ("expectancy", "Per trade", "usd"), ("profit_factor", "Profit factor", "ratio"),
           ("max_drawdown", "Max drawdown", "prob"), ("sharpe", "Sharpe", "ratio"), ("t_stat", "t-stat", "ratio")]


def compare_table(entries: list[dict]) -> dict:
    """entries: [{label, run, result}] (finished runs of the same kind) -> columns + rows of shown facts."""
    if not entries:
        return {"columns": [], "rows": []}
    is_study = entries[0]["run"]["kind"] == "study"
    cols = STUDY_COLS if is_study else BT_COLS
    rows = []
    for e in entries:
        facts = study_facts(e["result"]) if is_study else backtest_facts(e["result"])
        rows.append({"label": e["label"], "run_id": e["run"]["id"], "values": {k: fv(facts, k) for k, _, _ in cols}, "shown": {k: fs(facts, k) for k, _, _ in cols}})
    return {"kind": "study" if is_study else "backtest", "columns": [{"key": k, "label": l, "unit": u} for k, l, u in cols], "rows": rows}


def compare_text(table: dict, what: str = "") -> str:
    rows = table.get("rows") or []
    if len(rows) < 2:
        return ""
    study = table.get("kind") == "study"
    key, name, n_key = ("probability", "frequency", "qualifying_sessions") if study else ("net_pnl", "net P&L", "trades")
    parts = []
    if len(rows) == 2:
        a, b = rows
        parts.append(f"{a['label']}: {a['shown'][key]} {name} over {a['shown'][n_key]} {'sessions' if study else 'trades'}. {b['label']}: {b['shown'][key]} over {b['shown'][n_key]}.")
        va, vb = a["values"].get(key), b["values"].get(key)
        if _fin(va) and _fin(vb):
            diff = vb - va
            parts.append(f"The change {'raised' if diff > 0 else 'lowered' if diff < 0 else 'did not change'} the {name}" + (f" by {show(abs(diff), 'prob').replace('%', ' percentage points') if study else show(abs(diff), 'usd')}." if diff else "."))
    else:
        ok = [r for r in rows if _fin(r["values"].get(key))]
        if ok:
            best, worst = max(ok, key=lambda r: r["values"][key]), min(ok, key=lambda r: r["values"][key])
            parts.append(f"Across {len(rows)} variants{(' of ' + what) if what else ''}, the {name} ranged from {worst['shown'][key]} ({worst['label']}, {worst['shown'][n_key]} {'sessions' if study else 'trades'}) to {best['shown'][key]} ({best['label']}, {best['shown'][n_key]}).")
        thin = [r["label"] for r in rows if (r["values"].get(n_key) or 0) < 30]
        if thin:
            parts.append(f"Fewer than 30 observations for: {', '.join(thin[:6])} - treat those as indicative only.")
        parts.append(f"Testing {len(rows)} variants and picking the best one overstates the edge; prefer settings whose neighbours also perform well.")
    return " ".join(parts)
