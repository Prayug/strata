"""Research memory: what a notebook has already established, and a knowledge base across notebooks.

Nothing here is written by a language model and nothing is remembered that was not
computed. A FINDING is a row derived from a finished run: a deterministic statement, the
numbers behind it, an evidence grade, the configuration that reproduces it and the run it
came from. The notebook's memory (`notebook`) is assembled on demand from those rows, the
stored messages and the research ledger, so it cannot drift from the results it describes
and it cannot contain a finding that no run produced.

Evidence grades, strongest first
    confirmed            held on a final test period nothing had examined, here or elsewhere
    holds, not independent   held on this notebook's final period, but related configurations
                         elsewhere had already examined those dates
    supported            distinguishable from the baseline, survives the matched comparison
                         when one was run, no robustness flag; not validated on a final period
    suggestive           distinguishable on its own test, with an open weakness
    not confirmed        a candidate that failed on the final test period
    not supported        within what chance produces
    insufficient         too few observations to say anything

The tools below answer the follow-up questions a research assistant should be able to
answer from memory ("which of our findings still hold after 2020?", "which of these
results are independent?", "why did the filter raise the win rate but lower expectancy?").
Each one recomputes from stored observations or re-runs a stored configuration.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np

from strata.api import db

GRADES = ["confirmed", "holds, not independent", "supported", "suggestive", "holds on seen data", "not confirmed", "not supported", "insufficient"]
RANK = {g: len(GRADES) - i for i, g in enumerate(GRADES)}
USABLE = ("confirmed", "holds, not independent", "supported", "suggestive")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------- deriving findings
def grade_graph(res: dict) -> tuple[str, list[str]]:
    """Evidence grade of a graph's main claim, and the reasons for it."""
    by = {n["id"]: n.get("result") for n in res["nodes"] if n.get("result")}
    prob = next((r for r in by.values() if r.get("kind") == "probability"), None)
    matched = next((r for r in by.values() if r.get("kind") == "matched"), None)
    rob = next((r for r in by.values() if r.get("kind") == "robustness"), None)
    val = next((r for r in by.values() if r.get("kind") == "validation"), None)
    e = (prob or matched or {}).get("evidence") or {}
    why = []
    if not e or (e.get("n") or 0) < 30:
        return "insufficient", [f"only {e.get('n', 0)} qualifying sessions"]
    if val is not None:
        st = val.get("status")
        if st == "confirmed":
            return "confirmed", ["held on a final test period that nothing had examined"]
        if st == "confirmed_not_independent":
            return "holds, not independent", ["held on the final test period, but related configurations elsewhere had examined those dates"]
        if st == "holds_on_seen_data":
            return "holds on seen data", ["held on a final test period that had already been looked at"]
        if st == "not_confirmed":
            return "not confirmed", val.get("reasons") or ["the final test period did not confirm it"]
    p = e.get("p_value")
    if p is None:
        return "insufficient", ["too few independent weeks for a dependence-robust test"]
    if p >= 0.05:
        mv = (matched or {}).get("frequency", {}).get("verdict")
        if mv == "persists":
            why.append("the plain comparison is within chance, but the difference is distinguishable against matched sessions")
            if rob and rob.get("flags"):
                return "suggestive", why + rob["flags"][:2]
            return "suggestive", why
        return "not supported", [f"the difference is within what chance produces (p {p:.3f})"]
    weak = []
    if matched is not None and matched["frequency"].get("verdict") == "mixed":
        weak.append("the matched estimates disagree on whether it survives the controls")
    elif matched is not None and matched["frequency"].get("verdict") not in ("persists", "not_estimable"):
        weak.append("it does not survive the matched comparison")
    if rob is not None:
        weak += rob.get("flags") or []
    if val is not None and val.get("status") == "not_a_candidate":
        weak.append("it did not qualify on the training and validation periods: " + "; ".join(val.get("reasons") or []))
    if weak:
        return "suggestive", weak[:3]
    return "supported", ["distinguishable from the baseline" + (", survives the matched comparison" if matched is not None and matched["frequency"].get("verdict") == "persists" else "") + ", no robustness flag; not validated on a final test period"]


def findings_from_graph(res: dict) -> list[dict]:
    out = []
    by = {n["id"]: n.get("result") for n in res["nodes"] if n.get("result")}
    prob = next((r for r in by.values() if r.get("kind") == "probability"), None)
    matched = next((r for r in by.values() if r.get("kind") == "matched"), None)
    main = prob or matched
    g = res.get("graph") or {}
    if main is not None:
        e = main["evidence"]
        grade, why = grade_graph(res)
        stmt = (f"{main['symbol']}: when {main['condition']}, {main['outcome']} on {_pct(e.get('probability'))} of {e.get('n', 0)} sessions against {_pct(e.get('rest_probability'))} of the others"
                + (f" ({main['window'][0]} to {main['window'][1]})" if main.get("window") and main["window"][0] else "") + ".")
        if matched is not None and matched["frequency"].get("verdict") not in (None, "not_estimable"):
            stmt += " " + matched["frequency"]["sentence"]
        out.append({"kind": "pattern", "symbol": main["symbol"], "statement": stmt, "evidence": grade, "why": why, "fingerprint": main.get("key") or (prob or {}).get("key"),
                    "numbers": {"n": e.get("n"), "probability": e.get("probability"), "rest_probability": e.get("rest_probability"), "lift": e.get("lift"), "p_value": e.get("p_value"), "avg_end_return_pct": e.get("avg_end_return_pct"),
                                "rest_avg_end_return_pct": e.get("rest_avg_end_return_pct"), "matched": (matched or {}).get("frequency", {}).get("nearest", {}).get("estimate") if matched else None,
                                "matched_verdict": (matched or {}).get("frequency", {}).get("verdict") if matched else None},
                    "config": {"graph": g, "cell": (prob or {}).get("cell"), "condition": main["condition"], "outcome": main["outcome"]}, "tags": _tags(g), "period": main.get("window") or [None, None]})
    # refinements found by splitting the qualifying sessions on another value
    for b in (prob or {}).get("by") or []:
        if b.get("p_adjusted") is None or b["p_adjusted"] >= 0.05 or not b.get("buckets"):
            continue
        hi, lo = b["buckets"][-1], b["buckets"][0]
        better = hi if (hi["probability"] or 0) >= (lo["probability"] or 0) else lo
        edge = better["range"][0] if better is hi else better["range"][1]
        cell = None
        if (prob or {}).get("cell") and b.get("feature") and not str(b["feature"]).startswith("("):
            cell = {"preds": prob["cell"]["preds"] + [{"feature": b["feature"], "op": ">=" if better is hi else "<", "value": edge}], "outcome": prob["cell"]["outcome"]}
        out.append({"kind": "refinement", "symbol": prob["symbol"], "evidence": "suggestive", "why": ["found by splitting the qualifying sessions after seeing them (within-sample); corrected for the number of values scanned, not validated"],
                    "statement": f"{prob['symbol']}: among sessions where {prob['condition']}, {prob['outcome']} on {_pct(better['probability'])} of {better['n']} sessions when {b['label']} is "
                                 f"{'at least' if better is hi else 'below'} {edge:.4g}{b.get('unit') or ''}, against {_pct((lo if better is hi else hi)['probability'])} at the other end (adjusted p {b['p_adjusted']:.4f}).",
                    "fingerprint": f"{prob.get('key')}|{b['feature']}|{'hi' if better is hi else 'lo'}", "numbers": {"n": better["n"], "probability": better["probability"], "other": (lo if better is hi else hi)["probability"], "p_adjusted": b["p_adjusted"], "difference": b.get("difference")},
                    "config": {"graph": g, "cell": cell, "condition": f"{prob['condition']} and {b['label']} is {'at least' if better is hi else 'below'} {edge:.4g}{b.get('unit') or ''}", "outcome": prob["outcome"],
                               "refine": {"feature": b["feature"], "op": ">=" if better is hi else "<", "value": edge}}, "tags": _tags(g) + [str(b["feature"])], "period": prob.get("window") or [None, None]})
    return out


def _tags(graph: dict) -> list[str]:
    tags = set()
    for n in graph.get("nodes") or []:
        pa = n.get("params") or {}
        if n.get("op") == "feature" and pa.get("name"):
            tags.add(str(pa["name"]))
        for src in (pa.get("expr"),):
            if src:
                tags |= {w for w in re.findall(r"[a-z_]{4,}", str(src))}
        if n.get("op") in ("event", "sequence", "matched_comparison", "validation", "sweep"):
            tags.add(n["op"])
    return sorted(tags)[:24]


def findings_from_run(run: dict, result: dict) -> list[dict]:
    """Findings for the older result kinds (market studies and backtests)."""
    kind = run["kind"]
    if kind == "study":
        st = result.get("stats") or {}
        n, p, pb = st.get("resolved", st.get("sessions")), st.get("probability"), st.get("baseline_probability")
        if not n:
            return []
        sig = st.get("p_value")
        grade = "insufficient" if n < 30 else "suggestive" if (sig is not None and sig < 0.05) else "not supported" if sig is not None else "insufficient"
        cfg = result.get("config") or run.get("config") or {}
        return [{"kind": "study", "symbol": cfg.get("symbol", ""), "evidence": grade, "why": ["a single market study; not validated on a final test period"],
                 "statement": f"{cfg.get('symbol', '')}: {cfg.get('label') or run.get('name') or 'market study'} - the outcome occurred on {_pct(p)} of {n} sessions against {_pct(pb)} on all evaluated sessions.",
                 "fingerprint": None, "numbers": {"n": n, "probability": p, "baseline_probability": pb, "p_value": sig}, "config": {"study": cfg}, "tags": ["study"], "period": [result.get("start"), result.get("end")]}]
    if kind == "backtest":
        m = result.get("metrics") or {}
        n = m.get("trades") or 0
        cfg = run.get("config") or {}
        if not n:
            return []
        pnl = m.get("net_pnl")
        return [{"kind": "strategy", "symbol": (cfg.get("strategies") or [{}])[0].get("symbol") or cfg.get("symbol") or "", "evidence": "insufficient" if n < 30 else "suggestive" if (pnl or 0) > 0 else "not supported",
                 "why": [f"{n} trades on observed option prices; a backtest result, not a validated edge"], "fingerprint": None,
                 "statement": f"{run.get('name') or 'Strategy'}: {n} trades, net {'+' if (pnl or 0) >= 0 else '-'}${abs(pnl or 0):,.0f}, win rate {_pct(m.get('win_rate'))}, expectancy ${m.get('expectancy') or 0:,.0f} per trade.",
                 "numbers": {"trades": n, "net_pnl": pnl, "win_rate": m.get("win_rate"), "expectancy": m.get("expectancy")}, "config": {"backtest": cfg}, "tags": ["strategy", "options"], "period": [cfg.get("start"), cfg.get("end")]}]
    return []


def record(session_id: str | None, run_id: str, findings: list[dict]) -> int:
    """Store the findings of a run once (re-finalising the same run does not duplicate them)."""
    if not findings:
        return 0
    with db.connect() as con:
        if con.execute("SELECT 1 FROM findings WHERE run_id=? LIMIT 1", (run_id,)).fetchone():
            return 0
        now = db.now()
        con.executemany("INSERT INTO findings (session_id, run_id, symbol, kind, statement, evidence, fingerprint, config, numbers, tags, period_start, period_end, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(session_id, run_id, (f.get("symbol") or "").upper(), f["kind"], f["statement"], f["evidence"], f.get("fingerprint"), json.dumps({**f.get("config", {}), "why": f.get("why", [])}, default=str),
                          json.dumps(f.get("numbers") or {}, default=str), json.dumps(f.get("tags") or []), (f.get("period") or [None, None])[0], (f.get("period") or [None, None])[1], now) for f in findings])
    return len(findings)


def _row(r) -> dict:
    d = db._row(r)
    d["rank"] = RANK.get(d["evidence"], 0)
    d["why"] = (d.get("config") or {}).get("why", []) if isinstance(d.get("config"), dict) else []
    return d


def list_findings(session_id: str | None = None, symbol: str | None = None, q: str | None = None, evidence: list[str] | None = None, limit: int = 200) -> list[dict]:
    where, a = [], []
    if session_id:
        where.append("session_id=?")
        a.append(session_id)
    if symbol:
        where.append("symbol=?")
        a.append(symbol.upper())
    if evidence:
        where.append("evidence IN (" + ",".join("?" * len(evidence)) + ")")
        a += list(evidence)
    for w in re.findall(r"[\w%.-]{2,}", q or "")[:6]:
        where.append("(statement LIKE ? OR tags LIKE ? OR symbol LIKE ?)")
        a += [f"%{w}%"] * 3
    with db.connect() as con:
        rows = [_row(r) for r in con.execute("SELECT * FROM findings" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?", a + [int(limit)])]
        titles = {r[0]: r[1] for r in con.execute("SELECT id, title FROM sessions")}
    for r in rows:
        r["notebook"] = titles.get(r["session_id"], "")
    return rows


def get_finding(fid: int) -> dict | None:
    with db.connect() as con:
        r = con.execute("SELECT * FROM findings WHERE id=?", (int(fid),)).fetchone()
    return _row(r) if r else None


def strongest(session_id: str, kinds: tuple[str, ...] = ("pattern", "refinement", "study")) -> dict | None:
    """The best-supported finding of a notebook that can be turned into a strategy."""
    fs = [f for f in list_findings(session_id) if f["kind"] in kinds and f["evidence"] in USABLE]
    if not fs:
        return None
    return sorted(fs, key=lambda f: (-f["rank"], (f.get("numbers") or {}).get("p_value") if (f.get("numbers") or {}).get("p_value") is not None else (f.get("numbers") or {}).get("p_adjusted") or 1.0, -f["id"]))[0]


# ----------------------------------------------------------------------------- the notebook's memory
def notebook(session_id: str) -> dict:
    """Everything the notebook has established so far, grouped the way a researcher would
    ask for it. Every item carries the run or message it comes from."""
    from strata.research import ledger as LG
    msgs = db.list_messages(session_id)
    fs = list_findings(session_id)
    questions, decisions, assumptions, limits, open_q, conditions, comparisons = [], [], [], [], [], {}, []
    for m in msgs:
        c = m["content"] if isinstance(m["content"], dict) else {}
        if m["role"] == "user" and c.get("text"):
            questions.append({"at": m["created_at"], "text": c["text"], "message": m["id"]})
        if m["kind"] in ("graph", "hypothesis", "plan"):
            for a in (c.get("graph") or {}).get("assumptions") or c.get("assumptions") or []:
                assumptions.append({"text": a if isinstance(a, str) else a.get("text", ""), "message": m["id"]})
            for u in (c.get("graph") or {}).get("unsupported") or []:
                open_q.append({"text": f"Not understood, and not part of any analysis: {u}", "message": m["id"]})
            for cl in ((c.get("graph") or {}).get("core") or {}).get("clauses") or []:
                k = cl["label"]
                conditions.setdefault(k, {"condition": k, "times": 0, "first": m["created_at"], "message": m["id"]})["times"] += 1
        if m["kind"] == "result":
            for cv in (c.get("caveats") or [])[:6]:
                if isinstance(cv, str) and re.search(r"could not be observed|not evaluable|only .* stored|no (?:option|price)|not testable|sessions? (?:were|was) excluded|recent", cv):
                    limits.append({"text": cv, "message": m["id"]})
            if c.get("view") in ("sweep", "compare", "investigation") or (c.get("view") == "graph" and c.get("has_sweep")):
                comparisons.append({"text": (c.get("text") or "")[:300], "message": m["id"], "run_id": c.get("run_id")})
        if m["kind"] == "decision" or c.get("decision"):
            decisions.append({"text": (c.get("text") or "")[:240], "message": m["id"], "at": m["created_at"]})
    for f in fs:
        if f["evidence"] in ("suggestive", "supported") and f["kind"] in ("pattern", "refinement"):
            open_q.append({"text": f"Not yet validated on a final test period: {f['statement'][:200]}", "finding": f["id"]})
    seen = set()
    limits = [x for x in limits if not (x["text"] in seen or seen.add(x["text"]))][:12]
    seen = set()
    assumptions = [x for x in assumptions if x["text"] and not (x["text"] in seen or seen.add(x["text"]))][:20]
    led = LG.summary(session_id)
    return {"session": session_id, "hypotheses": questions[-40:], "tested_conditions": sorted(conditions.values(), key=lambda x: -x["times"])[:40],
            "findings": [f for f in fs if f["evidence"] in USABLE or f["evidence"] == "holds on seen data"], "discarded": [f for f in fs if f["evidence"] in ("not supported", "not confirmed", "insufficient")],
            "decisions": decisions[-20:], "assumptions": assumptions, "limitations": limits, "comparisons": comparisons[-12:], "open_questions": open_q[:20],
            "ledger": {"configurations": led["configurations"], "evaluations": led["evaluations"], "by_kind": led["by_kind"]}, "counts": {"findings": len(fs)}}


def summary_text(mem: dict) -> str:
    f, d = mem["findings"], mem["discarded"]
    parts = [f"This notebook has asked {len(mem['hypotheses'])} question{'s' if len(mem['hypotheses']) != 1 else ''} and tested {mem['ledger']['configurations']} distinct configuration{'s' if mem['ledger']['configurations'] != 1 else ''}."]
    if f:
        best = sorted(f, key=lambda x: -x["rank"])[0]
        parts.append(f"{len(f)} finding{'s are' if len(f) != 1 else ' is'} on record. The best supported ({best['evidence']}): {best['statement']}")
    else:
        parts.append("No finding is on record as supported yet.")
    if d:
        parts.append(f"{len(d)} idea{'s were' if len(d) != 1 else ' was'} set aside (within chance, too few observations, or not confirmed).")
    if mem["open_questions"]:
        parts.append(f"{len(mem['open_questions'])} open item{'s' if len(mem['open_questions']) != 1 else ''}, starting with: {mem['open_questions'][0]['text']}")
    return " ".join(parts)


# ----------------------------------------------------------------------------- resolving references
ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "last": -1, "latest": -1, "most recent": -1, "previous": -1, "prior": -2}


def graphs_of(session_id: str) -> list[dict]:
    """Every finished graph of the notebook, oldest first: {"run_id", "graph", "question", "at", "message"}."""
    from strata.api.results import run_dir
    out, seen = [], set()
    for m in db.list_messages(session_id):
        c = m["content"] if isinstance(m["content"], dict) else {}
        if m["kind"] != "graph":
            continue
        for r in c.get("runs") or []:
            if r["run_id"] in seen:
                continue
            p = run_dir(r["run_id"]) / "graph.json"
            if p.exists():
                seen.add(r["run_id"])
                res = json.loads(p.read_text())
                out.append({"run_id": r["run_id"], "graph": res["graph"], "question": res.get("question", ""), "at": m["created_at"], "message": m["id"], "result": res})
    return out


def day_of(text: str, today: date | None = None) -> date | None:
    today = today or datetime.now(timezone.utc).date()
    low = text.lower()
    if "yesterday" in low:
        return today - timedelta(days=1)
    if re.search(r"\btoday\b|this (?:morning|afternoon|session)|earlier", low):
        return today
    m = re.search(r"(\d+)\s+days?\s+ago", low)
    if m:
        return today - timedelta(days=int(m.group(1)))
    if "last week" in low:
        return today - timedelta(days=7)
    return None


def pick_graph(graphs: list[dict], words: str) -> dict | None:
    """The stored graph a phrase refers to: "our first study", "the second", "the last one"."""
    if not graphs:
        return None
    low = words.lower()
    for w, i in sorted(ORD.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(w)}\b", low):
            try:
                return graphs[i]
            except IndexError:
                return None
    return graphs[-1]


# ----------------------------------------------------------------------------- tools
def sessions_of(result: dict) -> dict[str, float]:
    """{date: end-of-horizon return} of the qualifying sessions of a graph or study result."""
    if result.get("kind") == "graph":
        r = next((n["result"] for n in result["nodes"] if n.get("result") and n["result"].get("kind") == "probability"), None)
        return {s["date"]: s.get("end_return_pct") for s in (r or {}).get("sessions", [])}
    return {s["date"]: s.get("end_return_pct") for s in result.get("sessions", [])}


def independence(items: list[dict], threshold: float = 0.5) -> dict:
    """Which results rest on the same observations. `items`: [{"label", "dates": set | list}].
    Two results are tied when the smaller one shares at least `threshold` of its sessions
    with the other; tied results form one group and count as one piece of evidence."""
    sets = [set(i["dates"]) for i in items]
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not sets[i] or not sets[j]:
                continue
            inter = len(sets[i] & sets[j])
            share = inter / min(len(sets[i]), len(sets[j]))
            pairs.append({"a": items[i]["label"], "b": items[j]["label"], "shared": inter, "share_of_smaller": round(share, 3), "jaccard": round(inter / len(sets[i] | sets[j]), 3)})
            if share >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [{"members": [items[i]["label"] for i in g], "sessions": len(set().union(*[sets[i] for i in g]))} for g in groups.values()]
    out.sort(key=lambda g: -len(g["members"]))
    return {"results": n, "independent_groups": len(out), "groups": out, "pairs": sorted(pairs, key=lambda p: -p["share_of_smaller"])[:40], "threshold": threshold,
            "sentence": (f"The {n} results rest on {len(out)} independent set{'s' if len(out) != 1 else ''} of sessions." if n else "There are no results to compare.")
                        + (" Results in the same group share at least half of the smaller one's sessions, so they are the same evidence seen twice, not separate confirmations." if len(out) < n else
                           " No two of them share half of their sessions." if n > 1 else "")}


def filter_effect(before: list[float], after: list[float], unit: str = "%") -> dict:
    """Why a filter can raise the win rate and lower expectancy at once. `before` / `after`
    are the per-observation results (returns or trade P&L) without and with the filter."""
    def stats(x):
        x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
        w, l = x[x > 0], x[x <= 0]
        return {"n": int(x.size), "win_rate": float((x > 0).mean()) if x.size else None, "avg_win": float(w.mean()) if w.size else 0.0, "avg_loss": float(l.mean()) if l.size else 0.0,
                "expectancy": float(x.mean()) if x.size else None, "total": float(x.sum()), "p95": float(np.quantile(x, 0.95)) if x.size >= 20 else None, "p05": float(np.quantile(x, 0.05)) if x.size >= 20 else None,
                "top5_share": float(np.sort(x)[-5:].sum() / x.sum()) if x.size >= 10 and x.sum() > 0 else None}
    a, b = stats(before), stats(after)
    out = {"before": a, "after": b, "unit": unit, "removed": a["n"] - b["n"], "answerable": bool(a["n"] and b["n"]), "facts": [], "sentence": ""}
    if not out["answerable"]:
        out["sentence"] = "Both versions need observations to compare."
        return out
    dp = b["win_rate"] - a["win_rate"]
    # expectancy = p * avg_win + (1 - p) * avg_loss  ->  split the change into its three sources (mid-point weights)
    pm, wm, lm = (a["win_rate"] + b["win_rate"]) / 2, (a["avg_win"] + b["avg_win"]) / 2, (a["avg_loss"] + b["avg_loss"]) / 2
    from_rate = dp * (wm - lm)
    from_wins = pm * (b["avg_win"] - a["avg_win"])
    from_losses = (1 - pm) * (b["avg_loss"] - a["avg_loss"])
    out["decomposition"] = {"from_win_rate": from_rate, "from_win_size": from_wins, "from_loss_size": from_losses, "total": b["expectancy"] - a["expectancy"]}
    f = lambda v: (f"{v:+.3f}%" if unit == "%" else f"{'+' if v >= 0 else '-'}${abs(v):,.2f}")  # noqa: E731
    out["facts"] = [f"Win rate {a['win_rate'] * 100:.1f}% -> {b['win_rate'] * 100:.1f}% ({a['n']} -> {b['n']} observations).",
                    f"Average win {f(a['avg_win'])} -> {f(b['avg_win'])}; average loss {f(a['avg_loss'])} -> {f(b['avg_loss'])}.",
                    f"Expectancy {f(a['expectancy'])} -> {f(b['expectancy'])}: {f(from_rate)} from the win rate, {f(from_wins)} from the size of wins, {f(from_losses)} from the size of losses."]
    de = b["expectancy"] - a["expectancy"]
    if dp > 0 and de < 0:
        cause = "the wins that remain are smaller" if from_wins < from_losses else "the losses that remain are larger"
        out["sentence"] = (f"The filter kept a higher share of winners (+{dp * 100:.1f} points) but {cause}: the change in win rate added {f(from_rate)} per observation while the change in win and loss sizes took away "
                           f"{f(-(from_wins + from_losses))}. A win rate says how often, expectancy says how much, and the filter removed the observations that carried the large gains.")
    elif dp > 0 and de >= 0:
        out["sentence"] = f"The filter raised both the win rate (+{dp * 100:.1f} points) and expectancy ({f(de)} per observation); there is no trade-off to explain here."
    else:
        out["sentence"] = f"With the filter the win rate changed by {dp * 100:+.1f} points and expectancy by {f(de)} per observation."
    return out
