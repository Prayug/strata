"""The research-wide experiment registry.

The ledger (ledger.py) keeps trials and the final test period PER NOTEBOOK, which is what a
multiple-testing correction needs. It cannot see that the same idea was already tried in
another notebook, or that the dates one notebook calls "untouched" were examined a dozen
times elsewhere. This module reads the same `trials` and `holdouts` tables across every
notebook and answers those questions. It records nothing itself and blocks nothing: it is
an audit trail and a set of warnings.

  fingerprint   the configuration hash (`ledger.config_key`): identical rules and
                parameters, whatever they were called and whichever window they ran on
  signature     what a configuration is about - symbol, the values it conditions on, the
                kind of outcome - used to find closely related hypotheses
  phase         exploration (market studies, graphs, sweeps), development (option
                backtests, optimisation, stress), validation (scored on a final test
                period), confirmation (a fresh period after that), forward (frozen and
                evaluated on sessions that did not exist when it was frozen)

"Independently confirmed" is reserved for a configuration whose final test period was
untouched in its own notebook AND was not examined by a related configuration anywhere
else. Anything short of that is labelled for what it is.
"""
from __future__ import annotations

import json

from strata.api import db
from strata.research import ledger as LG


def _base(symbol: str) -> str:
    return symbol.upper().split("@")[0]


def _titles(con, scopes) -> dict[str, str]:
    scopes = [s for s in scopes if s]
    if not scopes:
        return {}
    q = ",".join("?" * len(scopes))
    out = {r[0]: r[1] for r in con.execute(f"SELECT id, title FROM sessions WHERE id IN ({q})", scopes)}
    return {s: out.get(s) or {"sensitivity": "Sensitivity explorer", "encyclopedia": "Market behaviour", "ml": "ML research", "patterns": "Pattern explorer", "ide": "Strategy IDE"}.get(s.split(":")[0], s) for s in scopes}


def _sig(row) -> dict | None:
    s = row["signature"] if isinstance(row, dict) else row
    if not s:
        return None
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return None


def similarity(a: dict | None, b: dict | None) -> float:
    """0..1: how much two signatures are about the same thing (same symbol required)."""
    if not a or not b or _base(str(a.get("symbol") or "")) != _base(str(b.get("symbol") or "")):
        return 0.0
    fa, fb = set(a.get("features") or []), set(b.get("features") or [])
    jac = (len(fa & fb) / len(fa | fb)) if (fa | fb) else 0.0
    oa, ob = str(a.get("outcome") or "").split("|"), str(b.get("outcome") or "").split("|")
    same_out = sum(1 for x, y in zip(oa, ob) if x == y) / max(1, max(len(oa), len(ob)))
    return round(0.7 * jac + 0.3 * same_out, 3)


def exposure(symbol: str, start: str, end: str | None = None, exclude_scope: str | None = None, signature: dict | None = None) -> dict:
    """Configurations in OTHER notebooks that were evaluated on a window reaching into
    [start, end] for this symbol. `related` counts those about the same thing."""
    base = _base(symbol)
    with db.connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT scope, symbol, key, kind, label, signature, start, end, MAX(created_at) AS last FROM trials WHERE (symbol=? OR symbol LIKE ?) AND (end IS NULL OR end >= ?) AND (? IS NULL OR start IS NULL OR start <= ?)"
            + (" AND scope != ?" if exclude_scope else "") + " GROUP BY scope, key", [base, base + "@%", start, end, end] + ([exclude_scope] if exclude_scope else []))]
        titles = _titles(con, {r["scope"] for r in rows})
    by: dict[str, dict] = {}
    related = 0
    for r in rows:
        b = by.setdefault(r["scope"], {"scope": r["scope"], "title": titles.get(r["scope"], r["scope"]), "configurations": 0, "related": 0, "last": r["last"], "kinds": {}})
        b["configurations"] += 1
        b["kinds"][r["kind"]] = b["kinds"].get(r["kind"], 0) + 1
        b["last"] = max(b["last"], r["last"])
        if signature and similarity(signature, _sig(r["signature"])) >= 0.5:
            b["related"] += 1
            related += 1
    return {"symbol": base, "start": start, "end": end, "configurations": len(rows), "related": related, "notebooks": len(by), "by_notebook": sorted(by.values(), key=lambda b: -b["configurations"])[:20]}


def duplicates(key: str, exclude_scope: str | None = None) -> list[dict]:
    """Every place the identical configuration was evaluated."""
    with db.connect() as con:
        rows = [dict(r) for r in con.execute("SELECT scope, symbol, kind, label, run_id, start, end, value, n, p_value, phase, MIN(created_at) AS first, MAX(created_at) AS last, COUNT(*) AS evaluations FROM trials WHERE key=?"
                                             + (" AND scope != ?" if exclude_scope else "") + " GROUP BY scope ORDER BY first", [key] + ([exclude_scope] if exclude_scope else []))]
        titles = _titles(con, {r["scope"] for r in rows})
    for r in rows:
        r["title"] = titles.get(r["scope"], r["scope"])
    return rows


def related(signature: dict, exclude_key: str | None = None, limit: int = 20, min_score: float = 0.5) -> list[dict]:
    """Closely related hypotheses anywhere in Strata, most similar first."""
    base = _base(str(signature.get("symbol") or ""))
    with db.connect() as con:
        rows = [dict(r) for r in con.execute("SELECT scope, symbol, key, kind, label, signature, run_id, start, end, value, n, p_value, phase, MAX(created_at) AS last FROM trials "
                                             "WHERE (symbol=? OR symbol LIKE ?) AND signature IS NOT NULL GROUP BY key, scope ORDER BY last DESC LIMIT 4000", (base, base + "@%"))]
        titles = _titles(con, {r["scope"] for r in rows})
    out = []
    for r in rows:
        if r["key"] == exclude_key:
            continue
        sc = similarity(signature, _sig(r["signature"]))
        if sc >= min_score:
            sg = _sig(r["signature"]) or {}
            out.append({"scope": r["scope"], "title": titles.get(r["scope"], r["scope"]), "key": r["key"], "kind": r["kind"], "label": r["label"], "similarity": sc, "features": sg.get("features"), "outcome": sg.get("outcome"),
                        "run_id": r["run_id"], "start": r["start"], "end": r["end"], "value": r["value"], "n": r["n"], "p_value": r["p_value"], "phase": r["phase"], "last": r["last"]})
    return sorted(out, key=lambda x: (-x["similarity"], x["last"]), reverse=False)[:limit]


def holdouts(symbol: str | None = None) -> list[dict]:
    """Every final test period in every notebook, with how much the rest of Strata looked at it."""
    with db.connect() as con:
        rows = [db._row(r) for r in con.execute("SELECT * FROM holdouts" + (" WHERE symbol=? OR symbol LIKE ?" if symbol else "") + " ORDER BY id DESC LIMIT 500", ([_base(symbol), _base(symbol) + "@%"] if symbol else []))]
        titles = _titles(con, {r["scope"] for r in rows})
    out = []
    for h in rows:
        ex = exposure(h["symbol"], h["start"], h["end"], exclude_scope=h["scope"])
        p = LG._pub(h)
        p.update(title=titles.get(h["scope"], h["scope"]), elsewhere=ex["configurations"], elsewhere_notebooks=ex["notebooks"],
                 independent=bool(p["status"] in ("sealed", "used") and ex["configurations"] == 0))
        out.append(p)
    return out


def overview() -> dict:
    with db.connect() as con:
        tot = con.execute("SELECT COUNT(DISTINCT key), COUNT(*), COUNT(DISTINCT scope) FROM trials").fetchone()
        sym = [dict(r) for r in con.execute("SELECT symbol, COUNT(DISTINCT key) AS configurations, COUNT(*) AS evaluations, COUNT(DISTINCT scope) AS notebooks, MIN(start) AS first, MAX(end) AS last FROM trials GROUP BY symbol ORDER BY configurations DESC")]
        phase = [dict(r) for r in con.execute("SELECT COALESCE(phase, 'exploration') AS phase, COUNT(DISTINCT key) AS configurations FROM trials GROUP BY 1 ORDER BY 2 DESC")]
        kinds = [dict(r) for r in con.execute("SELECT kind, COUNT(DISTINCT key) AS configurations, COUNT(*) AS evaluations FROM trials GROUP BY kind ORDER BY 2 DESC")]
        fam = [dict(r) for r in con.execute("SELECT family, symbol, scope, COUNT(DISTINCT key) AS configurations, MIN(created_at) AS first, MAX(created_at) AS last FROM trials WHERE family != '' GROUP BY family, symbol, scope ORDER BY last DESC LIMIT 200")]
        rep = [dict(r) for r in con.execute("SELECT key, MIN(label) AS label, symbol, COUNT(DISTINCT scope) AS notebooks, COUNT(*) AS evaluations, MAX(created_at) AS last FROM trials GROUP BY key HAVING COUNT(*) > 1 ORDER BY notebooks DESC, evaluations DESC LIMIT 100")]
        feats: dict[str, int] = {}
        for (s,) in con.execute("SELECT DISTINCT signature FROM trials WHERE signature IS NOT NULL LIMIT 20000"):
            for f in (_sig(s) or {}).get("features") or []:
                feats[f] = feats.get(f, 0) + 1
        titles = _titles(con, {r["scope"] for r in fam})
    for r in fam:
        r["title"] = titles.get(r["scope"], r["scope"])
    hs = holdouts()
    return {"configurations": int(tot[0]), "evaluations": int(tot[1]), "notebooks": int(tot[2]), "by_symbol": sym, "by_phase": phase, "by_kind": kinds, "families": fam, "repeated": rep,
            "features": sorted(({"feature": k, "configurations": v} for k, v in feats.items()), key=lambda x: -x["configurations"])[:60], "holdouts": hs,
            "not_independent": sum(1 for h in hs if h["status"] in ("sealed", "used") and not h["independent"])}


def audit(scope: str | None = None, symbol: str | None = None, phase: str | None = None, q: str | None = None, limit: int = 300, offset: int = 0) -> dict:
    where, a = [], []
    if scope:
        where.append("scope=?")
        a.append(scope)
    if symbol:
        where.append("(symbol=? OR symbol LIKE ?)")
        a += [_base(symbol), _base(symbol) + "@%"]
    if phase:
        where.append("COALESCE(phase, 'exploration')=?")
        a.append(phase)
    if q:
        where.append("(label LIKE ? OR family LIKE ? OR key LIKE ?)")
        a += [f"%{q}%"] * 3
    w = (" WHERE " + " AND ".join(where)) if where else ""
    with db.connect() as con:
        total = int(con.execute("SELECT COUNT(*) FROM trials" + w, a).fetchone()[0])
        rows = [db._row(r) for r in con.execute("SELECT * FROM trials" + w + " ORDER BY id DESC LIMIT ? OFFSET ?", a + [int(limit), int(offset)])]
        titles = _titles(con, {r["scope"] for r in rows})
    for r in rows:
        r["title"] = titles.get(r["scope"], r["scope"])
    return {"total": total, "rows": rows}


def check(scope: str | None, symbol: str, key: str | None = None, signature: dict | None = None) -> dict:
    """The answer to "have I already tested this, or something like it, on this holdout?"."""
    base = _base(symbol)
    out = {"symbol": base, "scope": scope, "exact": [], "related": [], "holdouts": [], "sentences": []}
    if key:
        out["exact"] = duplicates(key)
    if signature:
        out["related"] = related(signature, exclude_key=key)
    with db.connect() as con:
        syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM holdouts WHERE scope=? AND (symbol=? OR symbol LIKE ?)", (scope or "", base, base + "@%"))] if scope else []
    for s in syms:
        h = LG.state(scope, s)["holdout"]
        if h is None:
            continue
        ex = exposure(s, h["start"], h["end"], exclude_scope=scope, signature=signature)
        here = LG._exposure(scope, s, h["start"])
        scored = sorted({k for u in h.get("uses") or [] for k in u.get("keys", [])})
        out["holdouts"].append({**h, "basis": "intraday entries" if "@" in s else "daily entries", "here": here, "scored": len(scored), "elsewhere": ex, "this_configuration_scored": bool(key and key in scored)})
    s = out["sentences"]
    if key:
        mine = [d for d in out["exact"] if d["scope"] == scope]
        other = [d for d in out["exact"] if d["scope"] != scope]
        if mine or other:
            s.append("This exact configuration has been evaluated before: " + "; ".join(([f"{sum(d['evaluations'] for d in mine)} time(s) in this notebook"] if mine else [])
                                                                                   + ([f"in {len(other)} other notebook(s) ({', '.join(d['title'] for d in other[:3])})"] if other else [])) + ".")
        else:
            s.append("This exact configuration has not been evaluated before anywhere in Strata.")
    if signature:
        rel = out["related"]
        s.append(f"{len(rel)} closely related configuration{'s have' if len(rel) != 1 else ' has'} been evaluated on {base} (same symbol, overlapping conditions and outcome)." if rel
                 else f"No closely related configuration on {base} is recorded.")
    if not out["holdouts"]:
        s.append(f"This notebook has not reserved a final test period for {base} yet, so nothing has been tested on one.")
    for h in out["holdouts"]:
        t = f"The final test period for {h['basis']} ({h['start']} to {h['end']}) is {h['label']}: {h['scored']} configuration(s) have been scored on it in this notebook"
        t += f", and {h['here']} configuration(s) tested here covered those dates" if h["here"] else ""
        e = h["elsewhere"]
        t += (f". Elsewhere in Strata, {e['configurations']} configuration(s) in {e['notebooks']} notebook(s) were evaluated on those dates" + (f", {e['related']} of them closely related to this one" if signature else "")
              + ": a result on it is not an independent confirmation.") if e["configurations"] else ". No other notebook has evaluated anything on those dates."
        s.append(t)
    return out
