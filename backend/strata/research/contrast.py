"""What sets a group of observations apart from the rest of the same result: "is there anything interesting about the losing
trades?", "anything those days have in common?".

Every feature is compared between the group and the rest of the SAME result (not all history), and each difference comes
with how often a difference at least that large appears when the same observations are split at random into groups of the
same sizes (a permutation test: 2,000 random splits). That share is reported as what it is, never as the probability that
the difference is real. How many features were compared is said, because with many comparisons a few large-looking
differences are expected by chance; a finding here is a lead to test on its own, not a result.
"""
from __future__ import annotations

import math
import random
from datetime import date

import numpy as np

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _perm(a: list[float], b: list[float], n: int = 2000, seed: int = 7) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    x = np.array(a + b, float)
    na = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(x)
        if abs(x[:na].mean() - x[na:].mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def day_context(store, symbol: str, days: list[str]) -> dict[str, dict]:
    """For each session: the gap at the open, the previous session's move, the session's own open-to-close move, the
    weekday and the VIX close of the day before (the application's own definitions, profile._daily)."""
    import polars as pl
    from strata.research.profile import _daily, _vix
    d = _daily(store, symbol)
    v = _vix(store)
    if v is not None:
        d = d.join(v, on="date", how="left")
    want = {date.fromisoformat(x[:10]) for x in days if x}
    sel = d.filter(pl.col("date").is_in(list(want)))
    out = {}
    for r in sel.iter_rows(named=True):
        out[r["date"].isoformat()] = {"gap": r.get("gap"), "prev": r.get("prev_return"), "oc": r.get("oc"), "weekday": WEEKDAYS[r["date"].weekday()] if r["date"].weekday() < 5 else None,
                                       "vix": r.get("vix_prev"), "year": r["date"].year}
    return out


AFTER_ENTRY = {"oc", "underlying_move", "held", "exit_reason"}  # part of what happened, not known when the trade was entered
NUMERIC = {"gap": ("the gap at the open", "{:+.2f}%"), "prev": ("the previous session's move", "{:+.2f}%"), "vix": ("the VIX the day before", "{:.1f}"),
           "oc": ("the session's open-to-close move", "{:+.2f}%"), "premium": ("the premium paid per share", "${:.2f}"), "underlying_move": ("the underlying's move while the trade was open", "{:+.2f}%"),
           "held": ("minutes held", "{:.0f}"), "entry_minute": ("the entry time", "{}")}


def compare(group: list[dict], rest: list[dict], what: str = "trades", group_name: str = "the losing trades", rest_name: str = "the others") -> dict:
    """group / rest: dicts of features (numeric or categorical). Returns lines in plain words and the table behind them."""
    rows = []
    if not group or not rest:
        return {"lines": [f"There is nothing to compare {group_name} with: {'none' if not group else 'all'} of the {what} are in that group."], "rows": [], "compared": 0}
    keys = [k for k in NUMERIC if any(isinstance(g.get(k), (int, float)) and g.get(k) is not None for g in group)]
    for k in keys:
        a = [float(g[k]) for g in group if isinstance(g.get(k), (int, float)) and g.get(k) is not None and math.isfinite(float(g[k]))]
        b = [float(r[k]) for r in rest if isinstance(r.get(k), (int, float)) and r.get(k) is not None and math.isfinite(float(r[k]))]
        if len(a) < 3 or len(b) < 3:
            continue
        rows.append({"feature": k, "label": NUMERIC[k][0], "group": float(np.mean(a)), "rest": float(np.mean(b)), "n_group": len(a), "n_rest": len(b), "p": _perm(a, b), "kind": "numeric"})
    for k, label in (("weekday", "the weekday"), ("exit_reason", "how the trade ended"), ("year", "the year")):
        cats = sorted({str(g.get(k)) for g in group + rest if g.get(k) is not None})
        if len(cats) < 2:
            continue
        best = None
        for c in cats:
            a = [1.0 if str(g.get(k)) == c else 0.0 for g in group if g.get(k) is not None]
            b = [1.0 if str(r.get(k)) == c else 0.0 for r in rest if r.get(k) is not None]
            if len(a) < 3 or len(b) < 3:
                continue
            diff = np.mean(a) - np.mean(b)
            if best is None or abs(diff) > abs(best[1]):
                best = (c, diff, a, b)
        if best is not None:
            c, diff, a, b = best
            rows.append({"feature": k, "label": label, "category": c, "group": float(np.mean(a)), "rest": float(np.mean(b)), "n_group": len(a), "n_rest": len(b), "p": _perm(a, b), "kind": "share"})
    for r in rows:
        r["known_at_entry"] = r["feature"] not in AFTER_ENTRY
    rows.sort(key=lambda r: (not r["known_at_entry"], r["p"] if math.isfinite(r["p"]) else 9, -abs(r["group"] - r["rest"])))
    n = sum(1 for r in rows if r["known_at_entry"])
    lines = []
    for r in [x for x in rows if x["known_at_entry"]][:3]:
        if r["kind"] == "numeric":
            fmt = NUMERIC[r["feature"]][1]
            gv, rv = (f"{int(r['group']) // 60:02d}:{int(r['group']) % 60:02d}", f"{int(r['rest']) // 60:02d}:{int(r['rest']) % 60:02d}") if r["feature"] == "entry_minute" else (fmt.format(r["group"]), fmt.format(r["rest"]))
            head = f"For {group_name}, {r['label']} averaged {gv}, against {rv} for {rest_name}."
        else:
            head = f"{r['category']} accounts for {r['group'] * 100:.0f}% of {group_name} and {r['rest'] * 100:.0f}% of {rest_name} ({r['label']})."
        if math.isfinite(r["p"]):
            share = r["p"] * 100
            head += (f" Splitting the same {what} into random groups of these sizes gives a difference at least this large {share:.0f}% of the time." if share >= 1 else
                     f" Random splits of the same {what} into groups of these sizes almost never give a difference this large (under 1% of the time).")
        lines.append(head)
    after = [r for r in rows if not r["known_at_entry"] and math.isfinite(r["p"]) and r["p"] < 0.05]
    if after:
        r = after[0]
        fmt = NUMERIC.get(r["feature"], ("", "{:+.2f}"))[1]
        gv, rv = (fmt.format(r["group"]), fmt.format(r["rest"])) if r["kind"] == "numeric" else (f"{r['group'] * 100:.0f}%", f"{r['rest'] * 100:.0f}%")
        lines.append(f"What happened during them: {r['label']} was {gv}, against {rv} for {rest_name}. That is how these {what} won or lost, not something known when they were entered, so it cannot be used as a filter.")
    low = [r for r in rows if r["known_at_entry"] and math.isfinite(r["p"]) and r["p"] < 0.05]
    if n:
        lines.append(f"{n} things known at the entry were compared. " + (f"With that many comparisons, about {max(1, round(0.05 * n))} would look this different by chance alone, so " +
                     ("none of these stands out." if not low else "treat these as leads to test on their own, not as findings.")))
    small = min(len(group), len(rest))
    if small < 20:
        lines.append(f"{small} {what} in the smaller group is a small sample.")
    return {"lines": lines, "rows": rows, "compared": n}
