"""Quantitative reasoning tools: questions ABOUT results, answered from stored observations.

    "Why did the strategy perform worse after 2020?"            -> subperiod
    "Is the 3% threshold genuinely different from the 2% one?"  -> threshold_difference
    "Would these results survive higher transaction costs?"     -> cost_survival
    "Are most profits concentrated in a handful of sessions?"   -> concentration
    "How much of the result is explained by volatility?"        -> volatility_explained
    "Which assumptions have the biggest effect?"                -> assumption_impact
    "Does this look too good?"                                  -> suspicious
    "What should I compare this with?"                          -> benchmarks
    "Why do the options lose when the pattern is real?"         -> options_vs_underlying

Every function is deterministic and returns the same shape:
    {title, answer, facts: [{label, value}], tables: [{title, columns, rows}], limits: [...], answerable}
`answer` is assembled from the numbers in `facts`/`tables`; nothing is written by a
language model. When the stored data cannot answer the question, `answerable` is False and
`answer` says what is missing instead of guessing.
"""
from __future__ import annotations

import math
import re

import numpy as np

from strata.research import robust as R


def _out(title, answer, facts=None, tables=None, limits=None, answerable=True, **extra) -> dict:
    return {"title": title, "answer": answer, "facts": facts or [], "tables": tables or [], "limits": limits or [], "answerable": answerable, **extra}


def _pct(x, d=1):
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 100:.{d}f}%"


def _pts(x, d=1):
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 100:+.{d}f} pts"


def _usd(x):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+,.0f}"


def is_study(res: dict) -> bool:
    return res.get("kind") == "study" or "stats" in res and "sessions" in res


def _study_sets(res: dict):
    """(qualifying {date: hit?}, others {date: hit?}, end returns of both) from a study result."""
    q = {r["date"]: (r["outcome"] == "target") for r in res.get("sessions", []) if r["outcome"] != "ambiguous"}
    qe = {r["date"]: r["end_return_pct"] for r in res.get("sessions", [])}
    b = res.get("baseline_sessions") or {"dates": [], "outcomes": "", "end_returns": []}
    signal = (res.get("config") or {}).get("entry_time") == "signal" or bool((res.get("config") or {}).get("sequence"))
    rest = {d: (o == "t") for d, o in zip(b["dates"], b["outcomes"]) if o != "a" and (signal or d not in qe)}
    return q, rest, qe


def _lift(q: dict, rest: dict) -> dict:
    return R.lift_test([float(v) for v in q.values()], list(q), [float(v) for v in rest.values()], list(rest), "week", 800) if q and rest else {"n": len(q), "p": None, "rest_p": None, "lift": None, "ci": [None, None], "p_value": None}


# ----------------------------------------------------------------------------- sub-periods
def subperiod(res: dict, trades: list[dict] | None, split: str) -> dict:
    """Before versus from `split` (an ISO date or a year)."""
    cut = f"{split}-01-01" if re.fullmatch(r"\d{4}", str(split)) else str(split)[:10]
    if is_study(res):
        q, rest, _ = _study_sets(res)
        a = _lift({d: v for d, v in q.items() if d < cut}, {d: v for d, v in rest.items() if d < cut})
        b = _lift({d: v for d, v in q.items() if d >= cut}, {d: v for d, v in rest.items() if d >= cut})
        if not a["n"] or not b["n"]:
            side = "before" if not a["n"] else "from"
            return _out(f"Before and after {cut}", f"There are no qualifying sessions {side} {cut}, so the two periods cannot be compared.", answerable=False)
        rows = [{"period": f"before {cut}", "sessions": a["n"], "frequency": a["p"], "other_sessions": a["rest_p"], "difference": a["lift"], "p_value": a["p_value"]},
                {"period": f"from {cut}", "sessions": b["n"], "frequency": b["p"], "other_sessions": b["rest_p"], "difference": b["lift"], "p_value": b["p_value"]}]
        d_freq, d_base, d_lift = b["p"] - a["p"], (b["rest_p"] or 0) - (a["rest_p"] or 0), (b["lift"] or 0) - (a["lift"] or 0)
        # is the change in the difference itself distinguishable? (two independent periods: z-test on the bootstrap standard errors)
        p_change = None
        if a.get("se") and b.get("se"):
            from scipy.stats import norm
            p_change = float(2 * (1 - norm.cdf(abs(d_lift) / math.sqrt(a["se"] ** 2 + b["se"] ** 2))))
        vq = [r for r in res.get("sessions", []) if r.get("vix_prev") is not None]
        va, vb = [r["vix_prev"] for r in vq if r["date"] < cut], [r["vix_prev"] for r in vq if r["date"] >= cut]
        ans = (f"Before {cut}: {_pct(a['p'])} of {a['n']} qualifying sessions against {_pct(a['rest_p'])} on other sessions ({_pts(a['lift'])}). "
               f"From {cut}: {_pct(b['p'])} of {b['n']} against {_pct(b['rest_p'])} ({_pts(b['lift'])}). "
               f"The frequency changed by {_pts(d_freq)}: {_pts(d_base)} of that is the baseline moving (all sessions behaved differently), {_pts(d_lift)} is the condition's own effect changing. ")
        if p_change is None:
            ans += "Too few independent weeks in one of the periods to test whether that change is more than chance."
        elif p_change >= 0.05:
            ans += f"The change in the condition's effect is NOT distinguishable from chance (p = {p_change:.2f}): the two periods are compatible with one unchanged effect measured twice."
        else:
            ans += f"The change in the condition's effect is distinguishable from chance (p = {p_change:.3f})."
        if va and vb:
            ans += f" Qualifying sessions had an average volatility-index level of {np.mean(va):.1f} before and {np.mean(vb):.1f} after."
        lim = []
        if min(a["n"], b["n"]) < R.MIN_OBS:
            lim.append(f"One period has only {min(a['n'], b['n'])} qualifying sessions; its frequency is very imprecise.")
        lim.append(f"Splitting at {cut} is itself a choice; a split picked after looking at the results is one more hypothesis tried.")
        return _out(f"Before and after {cut}", ans, [{"label": "Change in frequency", "value": _pts(d_freq)}, {"label": "From the baseline", "value": _pts(d_base)}, {"label": "From the condition", "value": _pts(d_lift)},
                                                      {"label": "p-value of the change", "value": "n/a" if p_change is None else f"{p_change:.3f}"}],
                    [{"title": "By period", "columns": ["period", "sessions", "frequency", "other_sessions", "difference", "p_value"], "rows": rows}], lim)
    tr = trades or []
    a, b = [t for t in tr if t["entry_date"] < cut], [t for t in tr if t["entry_date"] >= cut]
    if not a or not b:
        return _out(f"Before and after {cut}", f"All {len(tr)} trades fall on one side of {cut}, so the two periods cannot be compared. " + (f"The first trade is on {min(t['entry_date'] for t in tr)}." if tr else "There are no trades."), answerable=False)
    md = R.mean_diff_test([t["pnl"] for t in b], [t["entry_date"] for t in b], [t["pnl"] for t in a], [t["entry_date"] for t in a], "week", 800)
    rows = [{"period": f"before {cut}", "trades": len(a), "net_pnl": sum(t["pnl"] for t in a), "per_trade": float(np.mean([t["pnl"] for t in a])), "win_rate": sum(1 for t in a if t["pnl"] > 0) / len(a)},
            {"period": f"from {cut}", "trades": len(b), "net_pnl": sum(t["pnl"] for t in b), "per_trade": float(np.mean([t["pnl"] for t in b])), "win_rate": sum(1 for t in b if t["pnl"] > 0) / len(b)}]
    vx = lambda ts: [t["entry_vix"] for t in ts if t.get("entry_vix") is not None]
    ans = (f"Before {cut}: {len(a)} trades averaging {_usd(rows[0]['per_trade'])} each. From {cut}: {len(b)} trades averaging {_usd(rows[1]['per_trade'])}. The difference is {_usd(md['diff'])} per trade"
           + (f", 95% interval {_usd(md['ci'][0])} to {_usd(md['ci'][1])}, p = {md['p_value']:.2f}: " + ("distinguishable from chance." if md["p_value"] < 0.05 else "NOT distinguishable from chance - trade outcomes vary this much on their own.")
              if md["p_value"] is not None else ". Too few independent weeks to test it."))
    if vx(a) and vx(b):
        ans += f" Average volatility-index level at entry: {np.mean(vx(a)):.1f} before, {np.mean(vx(b)):.1f} after."
    lim = [f"Only {min(len(a), len(b))} trades in the smaller period."] if min(len(a), len(b)) < R.MIN_OBS else []
    lim.append("The option data may not cover both periods equally; check the coverage section before reading this as a change in the strategy.")
    return _out(f"Before and after {cut}", ans, [{"label": "Difference per trade", "value": _usd(md["diff"])}, {"label": "p-value", "value": "n/a" if md["p_value"] is None else f"{md['p_value']:.3f}"}],
                [{"title": "By period", "columns": ["period", "trades", "net_pnl", "per_trade", "win_rate"], "rows": rows}], lim)


# ----------------------------------------------------------------------------- two thresholds
def threshold_difference(res_a: dict, res_b: dict, label_a: str, label_b: str) -> dict:
    """Are two studies (typically nested thresholds of one condition) genuinely different?"""
    if not (is_study(res_a) and is_study(res_b)):
        return _out("Are the two different?", "This comparison needs two market studies of the same symbol.", answerable=False)
    qa, _, _ = _study_sets(res_a)
    qb, _, _ = _study_sets(res_b)
    both = set(qa) & set(qb)
    only_a, only_b = {d: v for d, v in qa.items() if d not in both}, {d: v for d, v in qb.items() if d not in both}
    disagree = [d for d in both if qa[d] != qb[d]]
    big, small, lb, ls = (qa, qb, label_a, label_b) if len(qa) >= len(qb) else (qb, qa, label_b, label_a)
    nested = set(small) <= set(big)
    facts = [{"label": f"Sessions in {label_a}", "value": str(len(qa))}, {"label": f"Sessions in {label_b}", "value": str(len(qb))}, {"label": "In both", "value": str(len(both))}]
    if nested and len(big) > len(small):
        outer = {d: v for d, v in big.items() if d not in small}
        lt = R.lift_test([float(v) for v in small.values()], list(small), [float(v) for v in outer.values()], list(outer), "week", 1000)
        ans = (f"Every session of \"{ls}\" is also a session of \"{lb}\" ({len(small)} of {len(big)}), so their headline numbers overlap and cannot be compared as if they were separate samples. "
               f"The fair comparison is between the sessions ONLY the wider one includes and the sessions of the stricter one: {_pct(lt['rest_p'])} of {lt['rest_n']} versus {_pct(lt['p'])} of {lt['n']}, a difference of {_pts(lt['lift'])}")
        if lt["p_value"] is None:
            ans += ". There are too few independent weeks to test it."
        else:
            ans += f" (95% interval {_pts(lt['ci'][0])} to {_pts(lt['ci'][1])}, p = {lt['p_value']:.2f}). " + ("They are genuinely different." if lt["p_value"] < 0.05 else "That is NOT a distinguishable difference: the stricter threshold has not been shown to behave differently, it only has fewer observations.")
        tab = [{"group": f"only in \"{lb}\"", "sessions": lt["rest_n"], "frequency": lt["rest_p"]}, {"group": f"\"{ls}\"", "sessions": lt["n"], "frequency": lt["p"]}]
        return _out("Are the two thresholds different?", ans, facts + [{"label": "Difference", "value": _pts(lt["lift"])}, {"label": "p-value", "value": "n/a" if lt["p_value"] is None else f"{lt['p_value']:.3f}"}],
                    [{"title": "Non-overlapping groups", "columns": ["group", "sessions", "frequency"], "rows": tab}],
                    ["A threshold chosen because it looked best among several is one more hypothesis tried; the comparison above does not correct for that."])
    if only_a and only_b:
        lt = R.lift_test([float(v) for v in only_a.values()], list(only_a), [float(v) for v in only_b.values()], list(only_b), "week", 1000)
        ans = (f"The two share {len(both)} sessions" + (f" (their outcomes differ on {len(disagree)} of them, so they measure different things)" if disagree else "") + f". On the sessions only one of them includes: {_pct(lt['p'])} of {lt['n']} for \"{label_a}\" "
               f"against {_pct(lt['rest_p'])} of {lt['rest_n']} for \"{label_b}\", {_pts(lt['lift'])}" + (f", p = {lt['p_value']:.2f}." if lt["p_value"] is not None else "; too few independent weeks to test it."))
        return _out("Are the two different?", ans, facts, [], [])
    return _out("Are the two different?", f"The two studies cover the same {len(both)} sessions" + (f" and disagree on the outcome of {len(disagree)} of them." if disagree else ", with the same outcomes: they are the same result."), facts)


# ----------------------------------------------------------------------------- costs
def cost_survival(res: dict, trades: list[dict] | None, stress: dict | None = None) -> dict:
    if is_study(res):
        st = res["stats"]
        edge = None
        b = res.get("baseline_sessions") or {}
        qd = {r["date"] for r in res.get("sessions", [])}
        rest = [e for d, e in zip(b.get("dates", []), b.get("end_returns", [])) if d not in qd]
        if st.get("avg_end_return_pct") is not None and rest:
            edge = st["avg_end_return_pct"] - float(np.mean(rest))
        ans = ("This is a market study: it has no trades, so no transaction costs were applied. "
               + (f"Qualifying sessions end {st['avg_end_return_pct']:+.2f}% from the entry on average, {edge:+.2f} points different from other sessions. " if edge is not None else "")
               + "Trading the underlying itself costs roughly 0.01-0.03% per round trip in a liquid ETF. Options are different: the spread and the premium paid are usually many times the size of an edge like this, "
                 "and only a backtest on observed option prices can say whether anything is left. Ask to test an option position on it.")
        return _out("Would it survive costs?", ans, [{"label": "Average move of qualifying sessions", "value": f"{st['avg_end_return_pct']:+.2f}%" if st.get("avg_end_return_pct") is not None else "n/a"},
                                                    {"label": "Difference from other sessions", "value": f"{edge:+.2f} pts" if edge is not None else "n/a"}], [], ["No option prices are involved in a market study."], answerable=False)
    cs = res.get("cost_sensitivity") or {}
    rows = cs.get("rows") or []
    tables, facts = [], []
    n = len(trades or [])
    base = rows[0]["net_pnl"] if rows else sum(t["pnl"] for t in trades or [])
    ans = f"As simulated the {n} trades net {_usd(base)}. "
    if stress:
        ok = [r for r in stress["rows"] if r["status"] == "ok"]
        tables.append({"title": "Re-simulated under worse execution", "columns": ["label", "trades", "missed", "net_pnl", "expectancy"], "rows": [{k: r.get(k) for k in ("label", "trades", "missed", "net_pnl", "expectancy")} for r in ok]})
        s = stress["summary"]
        be = s.get("breakeven_slippage_pct")
        neg = s.get("turns_negative") or []
        if (s.get("simulated_net_pnl") or 0) <= 0:
            ans += "It is already negative as simulated, so worse execution only deepens the loss"
            if (s.get("signal_net_pnl") or 0) > 0:
                ans += f"; with no costs at all the signal showed {_usd(s['signal_net_pnl'])}, so execution costs are what turn it negative"
            ans += ". "
        elif be is not None:
            ans += f"It reaches zero when every fill is about {be:.1f}% worse than simulated. "
        else:
            ans += "It stays positive across the slippage levels tested (up to 10% worse per fill). "
        if neg and (s.get("simulated_net_pnl") or 0) > 0:
            ans += "It turns negative under: " + "; ".join(neg[:5]) + ". "
        facts += [{"label": "Signal with no costs", "value": _usd(s.get("signal_net_pnl"))}, {"label": "As simulated", "value": _usd(s.get("simulated_net_pnl"))}, {"label": "Adverse combination", "value": _usd(s.get("adverse_net_pnl"))},
                  {"label": "Break-even slippage", "value": "n/a" if be is None else f"{be:.1f}%"}]
    elif rows:
        tables.append({"title": "Each fill repriced x% worse (trades unchanged)", "columns": ["extra_slippage_pct", "net_pnl", "win_rate", "avg_trade"], "rows": rows})
        be = cs.get("breakeven_slippage_pct")
        w = next((r for r in rows if r["extra_slippage_pct"] == 2), None)
        if base <= 0:
            ans += "It is already negative, so higher costs only deepen the loss. "
        elif be is not None:
            ans += f"Repricing every fill worse, it reaches zero at about {be:.1f}% per fill" + (f"; at 2% worse it nets {_usd(w['net_pnl'])}. " if w else ". ")
        ans += "This first-order estimate keeps the same trades; a full execution stress test re-decides entries, exits and missed fills."
        facts.append({"label": "Break-even slippage (first order)", "value": "n/a" if be is None else f"{be:.1f}%"})
    else:
        return _out("Would it survive costs?", "No cost analysis is stored for this run. Run the execution stress test to measure it.", answerable=False)
    lim = [f"Only {n} trades: the break-even level describes these trades and is not a forecast."] if n < R.MIN_OBS else []
    if (res.get("meta") or {}).get("pricing") == "trades" or (res.get("coverage") or {}).get("mode") == "trades":
        lim.append("Priced on trade prints: the real bid/ask spread at these moments was not observed, so the slippage levels are assumptions, not measurements.")
    return _out("Would it survive costs?", ans.strip(), facts, tables, lim)


# ----------------------------------------------------------------------------- concentration
def concentration(res: dict, trades: list[dict] | None) -> dict:
    if is_study(res):
        rows = [r for r in res.get("sessions", []) if r["outcome"] != "ambiguous"]
        if not rows:
            return _out("Concentration", "There are no qualifying sessions.", answerable=False)
        by_m: dict[str, list] = {}
        for r in rows:
            m = by_m.setdefault(r["date"][:7], [0, 0])
            m[0] += 1
            m[1] += r["outcome"] == "target"
        top = sorted(by_m.items(), key=lambda kv: -kv[1][0])[:5]
        n, k = len(rows), sum(1 for r in rows if r["outcome"] == "target")
        share = sum(v[0] for _, v in top[:3]) / n
        weeks = int(np.unique(R.cluster_ids([r["date"] for r in rows], "week")).size)
        years = len({r["date"][:4] for r in rows})
        ans = (f"The {n} qualifying sessions fall in {weeks} separate weeks and {years} calendar years. The three busiest months ({', '.join(m for m, _ in top[:3])}) hold {share * 100:.0f}% of them. "
               + ("That is heavy clustering: the sample behaves like far fewer than " + str(n) + " independent observations, which is why the intervals here resample whole weeks. " if share >= 0.25 or weeks < n * 0.6 else "They are fairly spread out. ")
               + f"{k} of the {n} reached the target.")
        return _out("Are the observations concentrated?", ans, [{"label": "Qualifying sessions", "value": str(n)}, {"label": "Separate weeks", "value": str(weeks)}, {"label": "Share in the 3 busiest months", "value": f"{share * 100:.0f}%"}],
                    [{"title": "Busiest months", "columns": ["month", "sessions", "hits"], "rows": [{"month": m, "sessions": v[0], "hits": v[1]} for m, v in top]}])
    tr = sorted(trades or [], key=lambda t: -t["pnl"])
    n = len(tr)
    if not n:
        return _out("Concentration", "There are no trades.", answerable=False)
    total = sum(t["pnl"] for t in tr)
    wins = [t for t in tr if t["pnl"] > 0]
    gross = sum(t["pnl"] for t in wins)
    k_half, acc = 0, 0.0
    for t in wins:
        acc += t["pnl"]
        k_half += 1
        if acc >= gross / 2:
            break
    without = {k: total - sum(t["pnl"] for t in tr[:k]) for k in (1, 3, 5) if k < n}
    by_day: dict[str, float] = {}
    for t in tr:
        by_day[t["entry_date"]] = by_day.get(t["entry_date"], 0.0) + t["pnl"]
    days = sorted(by_day.items(), key=lambda kv: -kv[1])
    ans = f"{n} trades net {_usd(total)}; {len(wins)} were profitable, earning {_usd(gross)} before the losers. "
    if gross > 0:
        ans += f"Half of all the winnings came from the best {k_half} trade{'s' if k_half != 1 else ''} ({k_half / n * 100:.0f}% of trades). "
    if without:
        ans += "Without the best " + ", ".join(f"{k}: {_usd(v)}" for k, v in without.items()) + ". "
    if total > 0 and 3 in without and without[3] <= 0:
        ans += "The result depends on three trades: remove them and nothing is left. "
    elif total > 0 and 3 in without and without[3] / total < 0.4:
        ans += "More than half of the profit comes from three trades. "
    elif total > 0:
        ans += "The profit is not carried by a handful of trades. "
    ans += f"The worst trade lost {_usd(tr[-1]['pnl'])}."
    lim = [f"With {n} trades, a few of them will always dominate; concentration is expected at this sample size."] if n < R.MIN_OBS else []
    return _out("Are profits concentrated?", ans, [{"label": "Net result", "value": _usd(total)}, {"label": "Trades for half the winnings", "value": str(k_half)}] + [{"label": f"Without the best {k}", "value": _usd(v)} for k, v in without.items()],
                [{"title": "Best and worst sessions", "columns": ["date", "pnl"], "rows": [{"date": d, "pnl": v} for d, v in days[:5] + days[-3:]]}], lim)


# ----------------------------------------------------------------------------- volatility as a confounder
def volatility_explained(res: dict, trades: list[dict] | None, store=None) -> dict:
    """How much of a study's difference from its baseline remains inside volatility bands."""
    bands = [(0, 15, "below 15"), (15, 20, "15 to 20"), (20, 30, "20 to 30"), (30, 1e9, "30 or more")]
    if not is_study(res):
        tr = [t for t in (trades or []) if t.get("entry_vix") is not None]
        if len(tr) < 4:
            return _out("How much is explained by volatility?", "The volatility-index level at entry is not recorded for enough of these trades to split them.", answerable=False)
        rows = []
        for lo, hi, name in bands:
            g = [t["pnl"] for t in tr if lo <= t["entry_vix"] < hi]
            if g:
                rows.append({"band": name, "trades": len(g), "net_pnl": float(sum(g)), "per_trade": float(np.mean(g)), "win_rate": sum(1 for x in g if x > 0) / len(g)})
        tab = R.regime_table([t["pnl"] for t in tr], [next(n for lo, hi, n in bands if lo <= t["entry_vix"] < hi) for t in tr], [t["entry_date"] for t in tr], [b[2] for b in bands])
        best = max(rows, key=lambda r: r["net_pnl"])
        total = sum(r["net_pnl"] for r in rows)
        ans = (f"Split by the volatility index at entry: " + "; ".join(f"{r['band']}: {r['trades']} trades, {_usd(r['per_trade'])} each" for r in rows) + ". "
               + (f"The band \"{best['band']}\" contributes {_usd(best['net_pnl'])} of the {_usd(total)} total. " if total else "")
               + (f"The bands differ more than chance would produce (p = {tab['heterogeneity_p']:.2f})." if tab["heterogeneity_p"] is not None and tab["heterogeneity_p"] < 0.05 else
                  "The differences between bands are within what chance produces at these sample sizes." if tab["heterogeneity_p"] is not None else "There are too few trades per band to test whether they differ."))
        return _out("How much is explained by volatility?", ans, [], [{"title": "By volatility band at entry", "columns": ["band", "trades", "net_pnl", "per_trade", "win_rate"], "rows": rows}],
                    ["Bands with fewer than 10 trades are shown but excluded from the test."])
    q, rest, _ = _study_sets(res)
    vix = {r["date"]: r.get("vix_prev") for r in res.get("sessions", [])}
    sym = res.get("symbol") or (res.get("config") or {}).get("symbol")
    try:
        from strata.research.features import get_features
        ft = get_features(store, sym)
        col = ft.col("vix_prev", "open")
        allv = {d: float(v) for d, v in zip(ft.dates, col) if math.isfinite(v)}
    except Exception:  # noqa: BLE001
        allv = {}
    if not allv:
        return _out("How much is explained by volatility?", "No volatility-index history is stored for this symbol, so the sessions cannot be grouped by volatility.", answerable=False)
    raw = _lift(q, rest)
    rows, wsum, wtot = [], 0.0, 0
    for lo, hi, name in bands:
        qb = {d: v for d, v in q.items() if lo <= (vix.get(d) if vix.get(d) is not None else allv.get(d, -1)) < hi}
        rb = {d: v for d, v in rest.items() if lo <= allv.get(d, -1) < hi}
        if not qb:
            continue
        pq, pr = sum(qb.values()) / len(qb), (sum(rb.values()) / len(rb) if rb else None)
        rows.append({"band": name, "qualifying": len(qb), "frequency": pq, "other_sessions": len(rb), "other_frequency": pr, "difference": None if pr is None else pq - pr})
        if pr is not None and len(rb) >= 10:
            wsum += (pq - pr) * len(qb)
            wtot += len(qb)
    if raw["lift"] is None or not wtot:
        return _out("How much is explained by volatility?", "There are too few comparable sessions inside each volatility band to separate the effect.", [], [{"title": "By volatility band", "columns": ["band", "qualifying", "frequency", "other_sessions", "other_frequency", "difference"], "rows": rows}], answerable=False)
    adj = wsum / wtot
    expl = (1 - adj / raw["lift"]) if raw["lift"] else None
    share_hi = sum(r["qualifying"] for r in rows if r["band"] in ("20 to 30", "30 or more")) / max(1, sum(r["qualifying"] for r in rows))
    rest_hi = sum(1 for d in rest if allv.get(d, 0) >= 20) / max(1, len(rest))
    ans = (f"Qualifying sessions happen in high volatility far more often than other sessions do ({share_hi * 100:.0f}% of them have the volatility index at 20 or more, against {rest_hi * 100:.0f}% of other sessions). "
           if share_hi - rest_hi > 0.15 else f"Qualifying sessions are spread over volatility levels much like other sessions ({share_hi * 100:.0f}% versus {rest_hi * 100:.0f}% with the index at 20 or more). ")
    ans += f"Overall the outcome is {_pts(raw['lift'])} more frequent on qualifying sessions. Comparing only sessions inside the same volatility band, the difference averages {_pts(adj)}. "
    if expl is None:
        pass
    elif expl >= 0.6:
        ans += f"About {expl * 100:.0f}% of the raw difference is therefore explained by WHEN these sessions occur (volatile periods), not by the condition itself."
    elif expl >= 0.25:
        ans += f"Roughly {expl * 100:.0f}% of the raw difference is explained by volatility; the rest remains within bands."
    elif expl <= -0.25:
        ans += "The difference is larger within bands than overall, so volatility was hiding part of it rather than creating it."
    else:
        ans += "Volatility explains little of it: the difference is about the same within bands."
    lim = ["Four fixed bands of the volatility index's previous close are a coarse control; within a band, volatility still varies.",
           "Bands with fewer than 10 other sessions are left out of the within-band average."]
    return _out("How much is explained by volatility?", ans, [{"label": "Raw difference", "value": _pts(raw["lift"])}, {"label": "Within volatility bands", "value": _pts(adj)}, {"label": "Explained by volatility", "value": "n/a" if expl is None else f"{expl * 100:.0f}%"}],
                [{"title": "By volatility band", "columns": ["band", "qualifying", "frequency", "other_sessions", "other_frequency", "difference"], "rows": rows}], lim)


# ----------------------------------------------------------------------------- assumptions
NUM_RE = re.compile(r"(?<![\w.\"])(-?\d+(?:\.\d+)?)(?![\w\"])")


def _perturb_tree(node: dict, path=()):
    """(path, expr) of every numeric threshold comparison in a condition tree."""
    if node.get("kind") == "cond":
        e = (node.get("params") or {}).get("expr")
        if node.get("type") == "expr" and e and re.search(r"(<=|>=|<|>)\s*-?\d", e):
            yield path, e
        return
    for i, c in enumerate(node.get("children") or []):
        yield from _perturb_tree(c, path + (i,))


def _set_expr(tree: dict, path, expr: str) -> dict:
    import copy
    t = copy.deepcopy(tree)
    n = t
    for i in path:
        n = n["children"][i]
    n["params"]["expr"] = expr
    return t


def assumption_impact(res: dict, trades: list[dict] | None, stress: dict | None = None, store=None) -> dict:
    """One assumption changed at a time: which ones move the result most."""
    if is_study(res):
        from strata.research.study import run_study
        cfg = res["config"]
        base_p, base_n = res["stats"].get("probability"), res["stats"]["qualifying_sessions"]
        if base_p is None:
            return _out("Which assumptions matter most?", "The study has no resolved observations to perturb.", answerable=False)
        variants = []
        for path, e in list(_perturb_tree(cfg.get("condition") or {}))[:3]:
            m = list(re.finditer(r"(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)", e))[-1]
            v = float(m.group(2))
            for f in (0.75, 1.25):
                nv = round(v * f, 4) if v else (0.25 if f > 1 else -0.25)
                variants.append((f"threshold {v:g} -> {nv:g} in \"{e}\"", {**cfg, "condition": _set_expr(cfg["condition"], path, e[:m.start(2)] + f"{nv:g}" + e[m.end(2):])}))
        for name in ("target", "stop"):
            lv = cfg.get(name)
            if lv and lv.get("kind") == "pct" and lv.get("value"):
                for f in (0.75, 1.25):
                    variants.append((f"{name} {abs(lv['value']):g}% -> {abs(lv['value'] * f):g}%", {**cfg, name: {**lv, "value": lv["value"] * f}}))
        et = cfg.get("entry_time")
        for alt in [x for x in ("open", "10:30", "close") if x != et][:2]:
            if alt == "close" and (cfg.get("horizon") or {}).get("kind") != "sessions":
                continue
            variants.append((f"entry {et} -> {alt}", {**cfg, "entry_time": alt}))
        hz = cfg.get("horizon") or {}
        if hz.get("kind") == "sessions":
            for k in sorted({max(1, hz["sessions"] - 1), hz["sessions"] + 2} - {hz["sessions"]}):
                variants.append((f"horizon {hz['sessions']} -> {k} sessions", {**cfg, "horizon": {**hz, "sessions": k}}))
        rows = []
        for label, c in variants[:12]:
            try:
                r = run_study(c, store, max_sessions_out=0)["stats"]
            except Exception as ex:  # noqa: BLE001
                rows.append({"change": label, "sessions": None, "frequency": None, "difference_from_baseline": None, "shift": None, "note": str(ex)[:80]})
                continue
            rows.append({"change": label, "sessions": r["qualifying_sessions"], "frequency": r["probability"], "difference_from_baseline": r["lift"],
                         "shift": None if r["probability"] is None else r["probability"] - base_p, "note": "thin" if r["qualifying_sessions"] < R.MIN_OBS else ""})
        ok = sorted([r for r in rows if r["shift"] is not None], key=lambda r: -abs(r["shift"]))
        if not ok:
            return _out("Which assumptions matter most?", "None of the study's assumptions could be varied automatically (its condition has no numeric threshold).", answerable=False)
        flips = [r for r in ok if r["difference_from_baseline"] is not None and res["stats"].get("lift") is not None and (r["difference_from_baseline"] > 0) != (res["stats"]["lift"] > 0)]
        ans = (f"Changing one assumption at a time ({len(ok)} variants; base case {_pct(base_p)} of {base_n} sessions): the largest shift comes from {ok[0]['change']} ({_pts(ok[0]['shift'])}, {ok[0]['sessions']} sessions)"
               + (f", then {ok[1]['change']} ({_pts(ok[1]['shift'])})" if len(ok) > 1 else "") + ". "
               + (f"{len(flips)} variant{'s' if len(flips) != 1 else ''} reverse the sign of the difference from the baseline: the conclusion depends on those choices." if flips else "No variant reverses the sign of the difference from the baseline."))
        return _out("Which assumptions matter most?", ans, [{"label": "Base case", "value": f"{_pct(base_p)} of {base_n}"}, {"label": "Variants tested", "value": str(len(ok))}, {"label": "Sign reversals", "value": str(len(flips))}],
                    [{"title": "One change at a time", "columns": ["change", "sessions", "frequency", "difference_from_baseline", "shift", "note"], "rows": ok}],
                    ["Each variant is one more configuration tried; they are recorded in the research ledger.", "Variants are run on the whole period, including any final test period of this session."], variants=len(ok))
    if not stress:
        return _out("Which assumptions matter most?", "For an option backtest this needs the execution stress test (each assumption re-simulated). Run \"challenge this strategy\" or the execution stress test first.", answerable=False, needs="stress")
    base = stress["summary"].get("simulated_net_pnl")
    groups: dict[str, dict] = {}
    for r in stress["rows"]:
        if r["status"] != "ok" or r["id"] in ("base",) or r.get("delta_net_pnl") is None:
            continue
        g = groups.setdefault(r["group"], {"assumption": r["group"], "worst_case": r["label"], "change_in_net_pnl": r["delta_net_pnl"], "missed": r.get("missed")})
        if abs(r["delta_net_pnl"]) > abs(g["change_in_net_pnl"]):
            g.update(worst_case=r["label"], change_in_net_pnl=r["delta_net_pnl"], missed=r.get("missed"))
    rows = sorted(groups.values(), key=lambda g: -abs(g["change_in_net_pnl"]))
    if not rows:
        return _out("Which assumptions matter most?", "The stress test produced no comparable scenarios.", answerable=False)
    ans = (f"Re-simulating with one execution assumption changed at a time (base case {_usd(base)}): the largest effect is {rows[0]['assumption'].lower()} - \"{rows[0]['worst_case']}\" changes the result by {_usd(rows[0]['change_in_net_pnl'])}"
           + (f"; next is {rows[1]['assumption'].lower()} ({_usd(rows[1]['change_in_net_pnl'])})" if len(rows) > 1 else "") + ". "
           + "Assumptions that change nothing were not binding for these trades (for example a volume limit the contracts already exceeded).")
    return _out("Which assumptions matter most?", ans, [{"label": "Base case", "value": _usd(base)}, {"label": "Most sensitive to", "value": rows[0]["assumption"]}],
                [{"title": "Largest effect per assumption", "columns": ["assumption", "worst_case", "change_in_net_pnl", "missed"], "rows": rows}],
                ["The reference row \"signal only\" removes all costs at once and is the largest by construction."] if rows[0]["assumption"] == "Reference" else [])


# ----------------------------------------------------------------------------- too good to be true
def suspicious(res: dict, trades: list[dict] | None) -> dict:
    flags, facts = [], []
    if is_study(res):
        st = res["stats"]
        p, n = st.get("probability"), st["qualifying_sessions"]
        if p is not None and n >= 20 and (p >= 0.9 or p <= 0.1):
            flags.append(f"The outcome happened on {_pct(p)} of {n} sessions. Frequencies this extreme usually mean the outcome is nearly implied by the condition (for example a target on the same side as a move that has already happened) rather than predicted by it.")
        if st.get("ambiguous") and st["ambiguous"] / max(n, 1) > 0.2:
            flags.append(f"{st['ambiguous']} of {n} observations are ambiguous (target and stop inside one bar): the frequency is computed on the rest, which may not be representative.")
        mix = st.get("resolution_mix") or {}
        if mix.get("1d") and (res.get("config") or {}).get("stop") and (res.get("config") or {}).get("measure") == "touch":
            flags.append(f"{mix['1d']} sessions are resolved from daily bars, where the order of target and stop inside the day is unknown.")
        facts = [{"label": "Frequency", "value": _pct(p)}, {"label": "Sessions", "value": str(n)}]
    else:
        m = res.get("metrics") or {}
        n = int(m.get("trades") or len(trades or []))
        wr, pf, sh = m.get("win_rate"), m.get("profit_factor"), m.get("sharpe")
        if wr is not None and n >= 20 and wr >= 0.85:
            tr = trades or []
            avg_w = np.mean([t["pnl"] for t in tr if t["pnl"] > 0]) if any(t["pnl"] > 0 for t in tr) else 0
            worst = min((t["pnl"] for t in tr), default=0)
            flags.append(f"Win rate {_pct(wr)} over {n} trades. The average win is {_usd(avg_w)} and the worst loss {_usd(worst)}: a high win rate with rare large losses is the signature of selling insurance, "
                         "and a sample without its bad period looks better than the strategy is.")
        if pf is not None and math.isfinite(pf) and pf > 4 and n >= 10:
            flags.append(f"Profit factor {pf:.1f}. Values this high almost always shrink with more data or realistic fills.")
        if sh is not None and math.isfinite(sh) and sh > 3:
            flags.append(f"Sharpe ratio {sh:.1f}. Sustained values above 3 are rare outside capacity-limited strategies; check fills and sample length before believing it.")
        ex = (res.get("meta") or {}).get("execution") or (res.get("config") or {}).get("execution") or {}
        cov = res.get("coverage") or {}
        pricing = (res.get("meta") or {}).get("pricing") or cov.get("mode")
        if pricing == "trades" and not ex.get("trade_bar_slippage_pct") and not ex.get("extra_slippage_pct"):
            flags.append("Fills are at observed trade prints with no slippage: that assumes every order traded at the last print, with no spread.")
        if pricing == "modeled":
            flags.append("Option premiums are MODELLED, not observed. No profitability conclusion can rest on this run.")
        if ex.get("fill_model") == "midpoint":
            flags.append("Fills are at the midpoint of the quote; real orders usually pay part of the spread.")
        ys = {}
        for t in trades or []:
            ys[t["entry_date"][:4]] = ys.get(t["entry_date"][:4], 0.0) + t["pnl"]
        if len(ys) >= 4 and all(v > 0 for v in ys.values()):
            flags.append(f"Every one of the {len(ys)} calendar years is profitable. Possible, but worth checking against the execution stress test.")
        sig, exe = (cov.get("signals"), cov.get("executed")) if cov else (None, None)
        facts = [{"label": "Trades", "value": str(n)}, {"label": "Win rate", "value": _pct(wr)}, {"label": "Profit factor", "value": "n/a" if pf is None or not math.isfinite(pf) else f"{pf:.2f}"}]
        if n < R.MIN_OBS:
            flags.append(f"Only {n} trades. Any headline number from this sample - good or bad - is weak evidence.")
        del sig, exe
    if not flags:
        return _out("Does it look too good?", "Nothing in the headline numbers is implausibly good. That is not a validation: it only means the usual warning signs (extreme win rate, extreme profit factor, costless fills) are absent.", facts)
    return _out("Does it look too good?", " ".join(flags), facts, [], [], flags=flags)


# ----------------------------------------------------------------------------- benchmarks
def benchmarks(res: dict, trades: list[dict] | None) -> dict:
    if is_study(res):
        st = res["stats"]
        recs = [f"All other evaluated sessions over the same period (used here: {_pct(st.get('baseline_probability'))} of {st.get('baseline_sessions')}).",
                "Sessions with a similar volatility level: conditions built on large moves select volatile periods, when almost any target is reached more often. Ask \"how much is explained by volatility?\".",
                "The mirror condition (the same move in the other direction), which shows whether the effect is about direction or only about size."]
        ans = "For a market study the right comparison is the same observation on sessions WITHOUT the condition. " + " ".join(f"({i + 1}) {r}" for i, r in enumerate(recs))
        return _out("What should this be compared with?", ans, [{"label": "Baseline in use", "value": _pct(st.get("baseline_probability"))}], [], ["Buy-and-hold is not a meaningful benchmark for a frequency."])
    cfg = res.get("config") or {}
    legs = (cfg.get("strategies") or [{}])[0].get("legs") or []
    long_only = all(l.get("side") == "buy" for l in legs)
    recs = ["The same option position opened on EVERY session with option data, without the entry condition. It answers whether the condition adds anything beyond owning that structure. (\"Challenge this strategy\" runs it.)",
            "The opposite side (puts instead of calls, or the reverse) on the same sessions: if both lose, the cost of the options is larger than the pattern."]
    if long_only:
        recs.append("The underlying itself over the same holding windows, to see how much of the result is plain market direction.")
    else:
        recs.append("Cash at the risk-free rate, since a premium-selling position is compared with not taking the risk at all.")
    m = res.get("metrics") or {}
    bad = []
    if m.get("benchmark_return") is not None:
        bad.append(f"The report's buy-and-hold figure ({_pct(m['benchmark_return'])}) holds the underlying through the whole period; this strategy was in the market on {len({t['entry_date'] for t in trades or []})} sessions, so the two are not exposed to the same risk.")
    return _out("What should this be compared with?", "Recommended comparisons: " + " ".join(f"({i + 1}) {r}" for i, r in enumerate(recs)) + (" " + bad[0] if bad else ""), [], [], bad)


# ----------------------------------------------------------------------------- options versus the underlying
def options_vs_underlying(study: dict, bt: dict, trades: list[dict]) -> dict:
    """Why an option position's profitability differs from the behaviour of the underlying."""
    q = {r["date"]: r for r in study.get("sessions", [])}
    tr = {}
    for t in trades:
        tr.setdefault(t["entry_date"], []).append(t)
    common = sorted(set(q) & set(tr))
    st = study["stats"]
    if not common:
        return _out("Options versus the underlying", f"The study has {len(q)} qualifying sessions and the backtest {len(trades)} trades, but none on the same dates, so they cannot be lined up. "
                    "The option data probably covers a different part of the period: compare the coverage sections.", answerable=False)
    rows, hit_win, hit_n, miss_win, miss_n = [], 0, 0, 0, 0
    moves, prem = [], []
    for d in common:
        pnl = sum(t["pnl"] for t in tr[d])
        hit = q[d]["outcome"] == "target"
        hit_n += hit
        miss_n += not hit
        hit_win += hit and pnl > 0
        miss_win += (not hit) and pnl > 0
        t0 = tr[d][0]
        basis = abs(t0.get("basis") or 0)
        spot = t0.get("entry_spot") or 0
        um = ((t0.get("exit_spot") or spot) / spot - 1) * 100 if spot else None
        moves.append(um)
        prem.append(basis / (spot * 100) * 100 if spot else None)
        rows.append({"date": d, "underlying_outcome": q[d]["outcome"], "underlying_move_pct": um, "option_pnl": pnl, "premium_pct_of_spot": prem[-1]})
    n = len(common)
    win = sum(1 for r in rows if r["option_pnl"] > 0)
    net = sum(r["option_pnl"] for r in rows)
    pm = [p for p in prem if p]
    ans = (f"On the {n} sessions both cover, the underlying outcome happened on {hit_n} ({hit_n / n * 100:.0f}%) while the option position made money on {win} ({win / n * 100:.0f}%), netting {_usd(net)}. ")
    if hit_n:
        ans += f"When the underlying did what the study measures, the options were profitable {hit_win} of {hit_n} times; "
    if miss_n:
        ans += f"when it did not, {miss_win} of {miss_n}. "
    if pm:
        ans += (f"The premium paid averaged {np.mean(pm):.2f}% of the underlying's price, so for a single long option the underlying has to move by more than that, in the right direction, before expiry or exit, just to break even. "
                "A frequency above 50% says nothing about the SIZE of the moves, and option profit depends on size, time decay and the spread. ")
    if hit_n and hit_win < hit_n:
        ans += f"{hit_n - hit_win} sessions had the right underlying outcome and still lost money in options: the move was smaller than what the premium and costs required."
    lim = [f"Only {n} sessions overlap ({len(q)} qualifying sessions, {len(trades)} trades): the rest of the study has no option prices."] if n < len(q) else []
    if n < R.MIN_OBS:
        lim.append("Too few overlapping sessions for any statistical statement; this is a description of what happened.")
    return _out("Why options differ from the underlying", ans, [{"label": "Sessions in both", "value": str(n)}, {"label": "Underlying outcome", "value": f"{hit_n}/{n}"}, {"label": "Options profitable", "value": f"{win}/{n}"},
                                                                {"label": "Study frequency (all sessions)", "value": _pct(st.get("probability"))}],
                [{"title": "Session by session", "columns": ["date", "underlying_outcome", "underlying_move_pct", "option_pnl", "premium_pct_of_spot"], "rows": rows[-60:]}], lim)


def paired_runs(trades_a: list[dict], trades_b: list[dict], label_a: str, label_b: str) -> dict:
    """Two backtests compared on the sessions both traded."""
    a, b = {}, {}
    for t in trades_a:
        a[t["entry_date"]] = a.get(t["entry_date"], 0.0) + t["pnl"]
    for t in trades_b:
        b[t["entry_date"]] = b.get(t["entry_date"], 0.0) + t["pnl"]
    pc = R.paired_compare(a, b, "week", 1000)
    if not pc["pairs"]:
        return _out("Difference between the two", f"\"{label_a}\" traded {len(a)} sessions and \"{label_b}\" {len(b)}, none in common. Their totals describe different sessions and cannot be compared directly.", answerable=False)
    ans = (f"On the {pc['pairs']} sessions both traded, \"{label_a}\" averaged {_usd(pc['mean_a'])} and \"{label_b}\" {_usd(pc['mean_b'])}: a difference of {_usd(pc['mean_diff'])} per session"
           + (f" (95% interval {_usd(pc['ci'][0])} to {_usd(pc['ci'][1])}" + (f", p = {pc['p_value']:.2f})" if pc["p_value"] is not None else ")") if pc["ci"][0] is not None else "")
           + f". \"{label_a}\" did better on {pc['a_better']} sessions, \"{label_b}\" on {pc['b_better']}. ")
    if pc["only_a"] or pc["only_b"]:
        ans += f"{pc['only_a']} sessions were traded only by the first and {pc['only_b']} only by the second; those are excluded, because a difference in WHICH sessions were traded is not a difference in the strategy. "
    if pc["p_value"] is not None:
        ans += "The difference is distinguishable from chance." if pc["p_value"] < 0.05 else "The difference is NOT distinguishable from chance."
    elif pc["pairs"] < 6:
        ans += "Too few shared sessions to test it."
    return _out("Difference between the two, on identical sessions", ans, [{"label": "Shared sessions", "value": str(pc["pairs"])}, {"label": "Mean difference", "value": _usd(pc["mean_diff"])},
                                                                           {"label": "p-value", "value": "n/a" if pc["p_value"] is None else f"{pc['p_value']:.3f}"}], [],
                [f"Only {pc['pairs']} shared sessions."] if pc["pairs"] < R.MIN_OBS else [])


# ----------------------------------------------------------------------------- routing
PATTERNS = [
    ("subperiod", r"(?:worse|better|differ\w*|chang\w*|perform\w*|hold up|work\w*)\b.*\b(?:after|since|from|before|post|pre)[- ]?\s*((?:19|20)\d{2})|\b(?:after|since|before|post|pre)[- ]?\s*((?:19|20)\d{2})\b.*\b(?:worse|better|differ|chang|why)"),
    ("threshold_difference", r"(genuinely|really|actually|truly|statistically|significantly|meaningfully)\s+(different|better|worse)|different from the|(\d+(?:\.\d+)?)\s*%\s*(?:threshold)?\s*(?:vs\.?|versus|or|compared (?:to|with)|from)\s*(?:the\s*)?(\d+(?:\.\d+)?)\s*%"),
    ("cost_survival", r"surviv\w*.*(cost|slippage|commission|fee|spread)|(higher|more|realistic|worse)\s+(transaction\s+)?(cost|slippage|commission|fee|spread|fill)|after costs|net of costs|cost sensitiv"),
    ("concentration", r"concentrat|handful|a few (?:trades|sessions|days)|depend\w* on (?:a few|one|two|a couple)|outlier|most (?:of the )?profits?|best (?:few )?trades"),
    ("volatility_explained", r"explained by (?:vol|vix)|(?:due|down|thanks) to (?:vol|vix)|just (?:vol|vix)|control(?:ling)? for (?:vol|vix)|confound|because of (?:vol|vix)|volatility (?:effect|regime)"),
    ("assumption_impact", r"which assumptions?|biggest effect|most sensitive|sensitiv\w+ to (?:the )?assumptions?|what matters most|largest (?:effect|impact)|drives? the result"),
    ("suspicious", r"too good|suspicious|unrealistic|believ|sanity|trust (?:this|these|it)|red flags?"),
    ("benchmarks", r"benchmark|compare (?:this|it) (?:to|with|against) what|right comparison|fair comparison|what should (?:i|this|it) (?:be )?compar"),
    ("options_vs_underlying", r"why (?:do|did|does|are|is)\s+(?:the\s+)?(?:options?|calls?|puts?|spreads?).*(?:los\w*|differ\w*|worse|not (?:profit|work))|options? (?:profit\w*|results?) differ|underlying.*(?:but|yet|while).*(?:option|call|put)|pattern is real"),
]


def detect(text: str) -> dict | None:
    low = text.lower()
    for name, pat in PATTERNS:
        m = re.search(pat, low)
        if m:
            out = {"tool": name}
            if name == "subperiod":
                out["split"] = next(g for g in m.groups() if g)
            return out
    return None
