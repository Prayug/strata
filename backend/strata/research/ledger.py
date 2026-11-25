"""The research ledger: every configuration tested, and the state of the final test period.

Two things make a backtest look better than it is without anyone cheating on purpose:
trying many configurations and reporting the best, and going back to tune after seeing the
"untouched" final period. Both are bookkeeping problems, so they are handled by bookkeeping.

Trials
    Every configuration that is evaluated - shown in a report or not, successful or not -
    is recorded against its research session (the `scope`) and symbol. Significance tests
    are corrected for the number of DISTINCT configurations in the ledger, not for the
    number that happen to be displayed. Re-running an identical configuration is not a new
    hypothesis and is counted once.

Holdouts
    The final test period of a session/symbol is a row with a status:
      sealed        reserved, never evaluated, and no earlier run in the session included it
      exposed       reserved, but earlier runs in this session already covered those dates
      used          candidates were scored on it once and the result was shown
      contaminated  new configurations were tested on data including it AFTER its results
                    were shown (i.e. the strategy was tuned with knowledge of the test)
    Only a `sealed` period can produce "confirmed on untouched data". Once a period is
    contaminated (or was exposed) a new holdout is required: sessions that did not exist in
    the session's data when the old one was inspected. If there are not enough of them yet,
    the ledger says how many more are needed instead of pretending.
"""
from __future__ import annotations

import hashlib
import json

from strata.api import db

# Kinds of trial that can tune a hypothesis. "option-structure" is deliberately absent: pricing option positions on a
# condition that was already selected does not re-choose the condition (it is still counted as a trial, because
# picking the best of several structures is itself a selection - of the structure, not of the pattern). The one
# exception is a final period that has not been scored yet: see _contaminate_if_tuned.
TUNING_KINDS = ("study", "backtest", "plan", "sweep", "discovery", "sensitivity", "optimization", "graph", "ml")
# where in the life of an idea a trial sits; the registry reports them separately
PHASES = ("exploration", "development", "validation", "confirmation", "forward")
PHASE_OF = {"study": "exploration", "graph": "exploration", "plan": "exploration", "sweep": "exploration", "discovery": "exploration", "sensitivity": "exploration", "ml": "exploration",
            "backtest": "development", "optimization": "development", "option-structure": "development", "stress": "development", "forward": "forward"}
MIN_HOLDOUT_SESSIONS = 40  # a fresh holdout smaller than this cannot confirm or reject anything
IGNORED_KEYS = {"id", "name", "label", "title", "start", "end", "run_id", "created_at", "notes", "interpreter", "approvals"}


def config_key(cfg) -> str:
    """Stable hash of what a configuration IS (its rules and parameters), ignoring names,
    ids and the date window, so the same idea on a different window is the same trial."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in sorted(o.items()) if k not in IGNORED_KEYS and v is not None}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        if isinstance(o, float):
            return round(o, 10)
        return o
    return hashlib.sha256(json.dumps(strip(cfg), sort_keys=True, default=str).encode()).hexdigest()[:20]


# ----------------------------------------------------------------------------- trials
def record(scope: str, symbol: str, trials: list[dict], family: str = "") -> dict:
    """Append trials. Each: {kind, key, label?, run_id?, start?, end?, metric?, value?, n?, p_value?, shown?}.
    Returns {"added", "new_configurations", "contaminated": holdout row or None}."""
    if not trials:
        return {"added": 0, "new_configurations": 0, "contaminated": None}
    sym = symbol.upper()
    with db.connect() as con:
        seen = {r[0] for r in con.execute("SELECT DISTINCT key FROM trials WHERE scope=? AND symbol=?", (scope, sym))}
        now = db.now()
        con.executemany(
            "INSERT INTO trials (scope, symbol, family, kind, key, label, run_id, start, end, metric, value, n, p_value, shown, created_at, signature, phase) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(scope, sym, t.get("family") or family, t["kind"], t["key"], (t.get("label") or "")[:200], t.get("run_id"), _iso(t.get("start")), _iso(t.get("end")), t.get("metric"),
              _num(t.get("value")), t.get("n"), _num(t.get("p_value")), 0 if t.get("shown") is False else 1, now,
              json.dumps(t["signature"], sort_keys=True) if t.get("signature") else None, t.get("phase") or PHASE_OF.get(t["kind"], "exploration")) for t in trials])
    fresh = [t for t in trials if t["key"] not in seen]
    hit = _contaminate_if_tuned(scope, sym, fresh) if fresh else None
    return {"added": len(trials), "new_configurations": len({t["key"] for t in fresh}), "contaminated": hit}


def _iso(d):
    return None if d is None else str(d)[:10]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def count(scope: str, symbol: str | None = None, family: str | None = None) -> int:
    """Distinct configurations tested in the scope (optionally for one symbol / family)."""
    q, a = "SELECT COUNT(DISTINCT key) FROM trials WHERE scope=?", [scope]
    if symbol:
        q += " AND symbol=?"
        a.append(symbol.upper())
    if family:
        q += " AND family=?"
        a.append(family)
    with db.connect() as con:
        return int(con.execute(q, a).fetchone()[0])


def summary(scope: str, symbol: str | None = None) -> dict:
    q, a = "SELECT kind, COUNT(DISTINCT key) AS n, COUNT(*) AS evaluations, SUM(shown) AS shown FROM trials WHERE scope=?", [scope]
    if symbol:
        q += " AND symbol=?"
        a.append(symbol.upper())
    with db.connect() as con:
        rows = [dict(r) for r in con.execute(q + " GROUP BY kind ORDER BY n DESC", a)]
        fam = [dict(r) for r in con.execute("SELECT family, symbol, COUNT(DISTINCT key) AS n, MIN(created_at) AS first, MAX(created_at) AS last FROM trials WHERE scope=?" + (" AND symbol=?" if symbol else "")
                                            + " GROUP BY family, symbol ORDER BY last DESC", a)]
    return {"scope": scope, "configurations": count(scope, symbol), "evaluations": sum(r["evaluations"] for r in rows), "by_kind": rows, "families": fam}


def list_trials(scope: str, symbol: str | None = None, limit: int = 2000) -> list[dict]:
    q, a = "SELECT * FROM trials WHERE scope=?", [scope]
    if symbol:
        q += " AND symbol=?"
        a.append(symbol.upper())
    with db.connect() as con:
        return [dict(r) for r in con.execute(q + " ORDER BY id DESC LIMIT ?", a + [int(limit)])]


def delete_scope(scope: str) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM trials WHERE scope=?", (scope,))
        con.execute("DELETE FROM holdouts WHERE scope=?", (scope,))


# ----------------------------------------------------------------------------- holdouts
def _holdouts(scope: str, symbol: str) -> list[dict]:
    with db.connect() as con:
        return [db._row(r) for r in con.execute("SELECT * FROM holdouts WHERE scope=? AND symbol=? ORDER BY id", (scope, symbol.upper()))]


def current(scope: str, symbol: str) -> dict | None:
    hs = _holdouts(scope, symbol)
    return hs[-1] if hs else None


def _exposure(scope: str, symbol: str, start: str) -> int:
    """Distinct configurations already evaluated in this scope on a window reaching into [start, ...)."""
    with db.connect() as con:
        return int(con.execute("SELECT COUNT(DISTINCT key) FROM trials WHERE scope=? AND symbol=? AND (end IS NULL OR end >= ?)", (scope, symbol.upper(), start)).fetchone()[0])


def reserve(scope: str, symbol: str, days: list[str], frac: float = 0.2) -> dict:
    """The final test period to use for this session and symbol, given the sessions stored
    (`days`, ISO dates in order). Returns a holdout row plus `usable` (may it confirm
    anything?) and, when it may not, `needed` (how many more sessions a fresh one requires)."""
    sym = symbol.upper()
    if not days:
        raise ValueError("no sessions to reserve a final test period from")
    cur = current(scope, sym)
    if cur is None:
        start = days[min(len(days) - 1, int(len(days) * (1 - frac)))]
        exposed = _exposure(scope, sym, start)
        return _create(scope, sym, start, days[-1], "exposed" if exposed else "sealed",
                       f"{exposed} configuration(s) tested earlier in this session already covered these dates" if exposed else "")
    if cur["status"] in ("sealed", "used"):
        if days[-1] > cur["end"] and cur["status"] == "sealed":
            _update(cur["id"], end=days[-1])  # sessions collected since it was reserved join it (nothing has looked at them)
            cur["end"] = days[-1]
        return _pub(cur)
    # exposed or contaminated: only sessions after it can form a new, untouched holdout
    later = [d for d in days if d > cur["end"]]
    if len(later) >= MIN_HOLDOUT_SESSIONS:
        return _create(scope, sym, later[0], later[-1], "sealed", f"replaces the {cur['status']} period {cur['start']} to {cur['end']}")
    out = _pub(cur)
    out.update(usable=False, needed=MIN_HOLDOUT_SESSIONS - len(later), newer_sessions=len(later))
    return out


def _create(scope, sym, start, end, status, reason) -> dict:
    now = db.now()
    with db.connect() as con:
        cur = con.execute("INSERT INTO holdouts (scope, symbol, start, end, status, uses, reason, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)", (scope, sym, start, end, status, "[]", reason, now, now))
        hid = cur.lastrowid
    return _pub(get(hid))


def get(hid: int) -> dict | None:
    with db.connect() as con:
        return db._row(con.execute("SELECT * FROM holdouts WHERE id=?", (hid,)).fetchone())


def _update(hid: int, **f) -> None:
    f["updated_at"] = db.now()
    if "uses" in f:
        f["uses"] = json.dumps(f["uses"])
    with db.connect() as con:
        con.execute(f"UPDATE holdouts SET {', '.join(k + '=?' for k in f)} WHERE id=?", [*f.values(), hid])


def _pub(h: dict) -> dict:
    h = dict(h)
    h["uses"] = h.get("uses") if isinstance(h.get("uses"), list) else []
    h["usable"] = h["status"] == "sealed"
    h["label"] = {"sealed": "untouched", "exposed": "already seen", "used": "used once", "contaminated": "contaminated"}[h["status"]]
    return h


def use(hid: int, keys: list[str], run_id: str | None = None, what: str = "") -> dict:
    """Record that candidates were scored on the holdout and the result shown. A sealed
    period becomes `used`; scoring a DIFFERENT set of candidates on a period that was
    already used is a second look and contaminates it."""
    h = get(hid)
    if h is None:
        raise LookupError("holdout not found")
    uses = h.get("uses") if isinstance(h.get("uses"), list) else []
    seen = {k for u in uses for k in u.get("keys", [])}
    new = [k for k in keys if k not in seen]
    uses.append({"at": db.now(), "run_id": run_id, "what": what, "keys": list(keys)})
    if h["status"] == "sealed":
        _update(hid, status="used", uses=uses)
        with db.connect() as con:  # the configurations scored on the final period move from exploration to validation
            con.executemany("UPDATE trials SET phase='validation' WHERE scope=? AND symbol=? AND key=?", [(h["scope"], h["symbol"], k) for k in keys])
    elif h["status"] == "used" and new:
        _update(hid, status="contaminated", uses=uses, reason=f"{len(new)} further candidate(s) were scored on the final test period after its first results had been shown")
    else:
        _update(hid, uses=uses)
    return _pub(get(hid))


def _contaminate_if_tuned(scope: str, sym: str, fresh: list[dict]) -> dict | None:
    cur = current(scope, sym)
    if cur is not None and cur["status"] == "sealed":
        covers = lambda t: t.get("end") is None or _iso(t["end"]) >= cur["start"]  # noqa: E731
        seen = [t for t in fresh if t["kind"] in TUNING_KINDS and covers(t)]
        if seen:  # looked at before anything was scored on it: no longer untouched, so it can no longer confirm
            n = len({t["key"] for t in seen})
            _update(cur["id"], status="exposed", reason=f"{n} configuration{'s were' if n != 1 else ' was'} evaluated on data that includes this period ({cur['start']} to {cur['end']}) before anything was scored on it")
            return _pub(get(cur["id"]))
        # Option positions priced on a condition do not re-choose the condition, so once the pattern HAS been scored on its
        # final period they are harmless to it (see TUNING_KINDS). Before that, they are a look: an at-the-money call's profit
        # on those sessions is very nearly the pattern's own outcome there, so the period can no longer test the pattern blind.
        priced = [t for t in fresh if t["kind"] == "option-structure" and covers(t)]
        if priced:
            n = len({t["key"] for t in priced})
            _update(cur["id"], status="exposed", reason=f"{n} option position{'s were' if n != 1 else ' was'} priced on sessions inside this period ({cur['start']} to {cur['end']}) before the price pattern itself was scored on it; "
                                                     "their results show how the pattern fared there")
            return _pub(get(cur["id"]))
        return None
    if cur is None or cur["status"] != "used":
        return None
    scored = {k for u in (cur.get("uses") or []) for k in u.get("keys", [])}
    tuned = [t for t in fresh if t["kind"] in TUNING_KINDS and t["key"] not in scored and (t.get("end") is None or _iso(t["end"]) >= cur["start"])]
    if not tuned:
        return None
    n = len({t["key"] for t in tuned})
    _update(cur["id"], status="contaminated",
            reason=f"{n} new configuration{'s were' if n != 1 else ' was'} tested on data that includes the final test period ({cur['start']} to {cur['end']}) after its results had been shown")
    return _pub(get(cur["id"]))


def state(scope: str, symbol: str) -> dict:
    """What the interface shows about trials and the final test period for a session/symbol."""
    hs = [_pub(h) for h in _holdouts(scope, symbol)]
    return {"scope": scope, "symbol": symbol.upper(), "configurations": count(scope, symbol), "holdout": hs[-1] if hs else None, "history": hs[:-1]}


def holdout_note(h: dict | None) -> str:
    if h is None:
        return "No final test period has been reserved yet."
    span = f"{h['start']} to {h['end']}"
    if h["status"] == "sealed":
        return f"The final test period ({span}) has not been evaluated by anything in this session."
    if h["status"] == "exposed":
        return f"The final test period ({span}) is not untouched: {h.get('reason') or 'earlier runs in this session covered it'}. Results on it are labelled accordingly."
    if h["status"] == "used":
        return f"The final test period ({span}) has been used once. Testing new configurations on data that includes it will contaminate it."
    more = f" {h['needed']} more sessions are needed before a new one can be reserved." if h.get("needed") else ""
    return f"The final test period ({span}) is contaminated: {h.get('reason')}. A new holdout is required before anything can be called confirmed.{more}"
