"""The market behaviour encyclopedia: reproducible, configurable historical studies.

Each entry is a small program over the session-feature table - never a stored number.
Opening an entry computes it on the data stored for the chosen symbol and period, with the
parameters the reader sets, and returns:

    summary   sentences assembled from the computed values
    stats     headline tiles
    tables    rows with samples and dependence-aware intervals
    charts    simple series for the interface
    cells     the exact configuration behind each row, so its sessions can be opened
    hypothesis  a sentence that turns the entry into a research question for the AI Lab

Rows with fewer than 30 observations are flagged, not hidden. Everything describes the
UNDERLYING; none of it is an option return.
"""
from __future__ import annotations

import numpy as np

from strata.data.store import MarketStore
from strata.research import evidence as EV
from strata.research import robust as R
from strata.research.features import A, OutcomeSpec, Pred, T, cond_text, get_features

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _param(name, label, kind, default, **kw):
    return {"name": name, "label": label, "kind": kind, "default": default, **kw}


COMMON = [_param("symbol", "Symbol", "symbol", "SPY"), _param("start", "From", "date", None), _param("end", "To", "date", None)]


def _row(ft, preds: list[Pred], oc: OutcomeSpec, pm: np.ndarray, label: str, reps: int = 400) -> dict:
    e = EV.of(ft, ft.evaluate(preds, oc), pm, oc.direction, reps=reps, return_ci=True)
    row = {"label": label, "n": e["n"], "hits": e["hits"], "frequency": e["probability"], "baseline": e["baseline_probability"], "difference": e["lift"],
           "avg_return_pct": e["avg_end_return_pct"], "median_return_pct": e["median_end_return_pct"], "thin": e["n"] < 30, "ci95": e["ci95"], "p_value": e["p_value"],
           "cell": {"preds": [p.dump() for p in preds], "outcome": oc.dump()}, "condition": cond_text(preds, ft.symbol), "evaluated": e["evaluated"]}
    if e["return_ci95"][0] is not None:
        row["return_ci95"] = e["return_ci95"]
    return row


def _finish(entry: dict, ft, rows: list[dict], summary: list[str], stats: list[dict], charts: list[dict], hypothesis: str, notes: list[str] | None = None, columns=None) -> dict:
    adj = R.adjust([r.get("p_value") for r in rows], "holm")
    for r, a in zip(rows, adj):
        r["p_adjusted"] = a
        r["distinguishable"] = bool(a is not None and a < 0.05 and not r["thin"])
    thin = sum(1 for r in rows if r["thin"])
    base_notes = [f"{len(rows)} rows are compared; p-values resample whole calendar weeks and are corrected for all rows (Holm).", "Underlying price behaviour only - none of this is an option return."]
    if thin:
        base_notes.append(f"{thin} row{'s have' if thin != 1 else ' has'} fewer than 30 observations and should not be relied on.")
    return {"id": entry["id"], "title": entry["title"], "category": entry["category"], "symbol": ft.symbol, "summary": " ".join(summary), "stats": stats,
            "tables": [{"title": entry["title"], "columns": columns or ["label", "n", "frequency", "baseline", "difference", "avg_return_pct", "p_adjusted"], "rows": rows}], "charts": charts,
            "hypothesis": hypothesis, "notes": base_notes + (notes or []), "synthetic": bool(ft.store.is_synthetic(ft.symbol)) if ft.store else False}


def _pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _bar(title, rows, key="frequency", base_key="baseline", pct=True):
    return {"type": "bar", "title": title, "x": [r["label"] for r in rows], "series": [{"name": "condition", "values": [None if r.get(key) is None else (r[key] * 100 if pct else r[key]) for r in rows]}]
            + ([{"name": "all evaluated sessions", "values": [None if r.get(base_key) is None else r[base_key] * 100 for r in rows]}] if base_key else []), "unit": "%" if pct else "", "thin": [r["thin"] for r in rows],
            "ci": [[None if v is None else v * 100 for v in (r.get("ci95") or [None, None])] for r in rows] if pct and key == "frequency" else None}


# ----------------------------------------------------------------------------- entries
def overnight_gaps(ft, pm, p):
    rows = []
    oc_u, oc_d = OutcomeSpec("open", "session", "up", 0.0, None, "close"), OutcomeSpec("open", "session", "down", 0.0, None, "close")
    sizes = sorted({0.25, 0.5, 1.0, float(p["size"])})
    for x in sizes:
        rows.append(_row(ft, [Pred("gap", "<=", -x)], oc_u, pm, f"gap down {x:g}%+ : closes above the open"))
    for x in sizes:
        rows.append(_row(ft, [Pred("gap", ">=", x)], oc_d, pm, f"gap up {x:g}%+ : closes below the open"))
    fill_d = _row(ft, [Pred("gap", "<=", -float(p["size"]))], OutcomeSpec("open", "session", "up", float(p["size"]), None, "touch"), pm, f"gap down {p['size']:g}%+ : fully retraced during the session")
    fill_u = _row(ft, [Pred("gap", ">=", float(p["size"]))], OutcomeSpec("open", "session", "down", float(p["size"]), None, "touch"), pm, f"gap up {p['size']:g}%+ : fully retraced during the session")
    rows += [fill_d, fill_u]
    g = ft.col("gap", "open")[pm]
    g = g[np.isfinite(g)]
    main_d = next(r for r in rows if r["label"].startswith(f"gap down {p['size']:g}%+ : closes"))
    main_u = next(r for r in rows if r["label"].startswith(f"gap up {p['size']:g}%+ : closes"))
    s = [f"{ft.symbol} opened {p['size']:g}% or more below the previous close on {main_d['n']} sessions and closed above its open on {_pct(main_d['frequency'])} of them (all sessions: {_pct(main_d['baseline'])}).",
         f"It opened {p['size']:g}% or more higher on {main_u['n']} sessions and closed below its open on {_pct(main_u['frequency'])} (all sessions: {_pct(main_u['baseline'])}).",
         f"A gap of that size was fully retraced the same day on {_pct(fill_d['frequency'])} of gaps down and {_pct(fill_u['frequency'])} of gaps up."]
    return rows, s, [{"label": "Median |gap|", "value": f"{np.median(np.abs(g)):.2f}%"}, {"label": f"Gaps of {p['size']:g}%+", "value": str(main_d["n"] + main_u["n"])}, {"label": "Sessions", "value": str(g.size)}], \
        [_bar("Fade frequency by gap size", rows[:len(sizes) * 2])], f"Compare mean reversion and momentum after overnight gaps in {ft.symbol}."


def opening_range(ft, pm, p):
    mins = int(p["minutes"])
    feat = {15: "or15_pos", 30: "or30_pos", 60: "or60_pos"}[mins]
    later = {15: ["10:00", "10:30", "11:30"], 30: ["10:30", "11:00", "11:30"], 60: ["11:30", "12:30", "13:30"]}[mins]
    rows = []
    for t in later:
        rows.append(_row(ft, [Pred(feat, ">", 100)], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"above the range at {t} : closes higher"))
        rows.append(_row(ft, [Pred(feat, "<", 0)], OutcomeSpec(t, "session", "down", 0.0, None, "close"), pm, f"below the range at {t} : closes lower"))
        rows.append(_row(ft, [Pred(feat, "between", 0, 100.0001)], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"inside the range at {t} : closes higher"))
    a, b = rows[0], rows[1]
    s = [f"At {later[0]} ET {ft.symbol} was above the high of its first {mins} minutes on {a['n']} sessions and finished higher than that price on {_pct(a['frequency'])} of them; it was below the low on {b['n']} and finished lower on {_pct(b['frequency'])}.",
         f"Only sessions with {mins}-minute or finer bars can be used ({rows[0]['evaluated']} in this period)."]
    return rows, s, [{"label": "Opening range", "value": f"{mins} min"}, {"label": "Sessions with bars fine enough", "value": str(rows[0]["evaluated"])}], [_bar("Continuation after a breakout", rows)], \
        f"How does the effectiveness of opening range breakouts in {ft.symbol} change with volatility?"


def intraday_reversals(ft, pm, p):
    x = float(p["move"])
    rows = []
    for t in ("10:30", "11:30", "12:30", "13:30"):
        rows.append(_row(ft, [Pred("from_open", "<=", -x)], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"down {x:g}%+ from the open at {t} : closes above that price"))
        rows.append(_row(ft, [Pred("from_open", ">=", x)], OutcomeSpec(t, "session", "down", 0.0, None, "close"), pm, f"up {x:g}%+ from the open at {t} : closes below that price"))
    a, b = rows[0], rows[1]
    s = [f"When {ft.symbol} was {x:g}% or more below its open at 10:30 ET ({a['n']} sessions) it closed above the 10:30 price on {_pct(a['frequency'])}; when it was {x:g}% or more above ({b['n']} sessions) it closed below on {_pct(b['frequency'])}.",
         "A frequency near the baseline means the early move neither reverses nor continues reliably."]
    return rows, s, [{"label": "Early move", "value": f"{x:g}%"}, {"label": "Sessions with intraday bars", "value": str(rows[0]["evaluated"])}], [_bar("Reversal frequency by time of day", rows)], \
        f"Investigate whether {ft.symbol} tends to reverse during the session after an early move of {x:g}% from the open."


def decline_recovery(ft, pm, p):
    rows = []
    for x in (1, 2, 3, 4):
        for hz, lbl in (("sessions:1", "next close higher"), ("sessions:5", "higher 5 sessions later"), ("sessions:20", "higher 20 sessions later")):
            rows.append(_row(ft, [Pred("ret_1", "<=", -float(x))], OutcomeSpec("close", hz, "up", 0.0, None, "close"), pm, f"after a {x}%+ decline : {lbl}"))
    x0 = float(p["decline"])
    main = _row(ft, [Pred("ret_1", "<=", -x0)], OutcomeSpec("close", "sessions:1", "up", 0.0, None, "close"), pm, f"after a {x0:g}%+ decline : next close higher")
    if main["label"] not in {r["label"] for r in rows}:
        rows.insert(0, main)
    s = [f"{ft.symbol} fell {x0:g}% or more on {main['n']} sessions. The next close was higher on {_pct(main['frequency'])} of them (all sessions: {_pct(main['baseline'])}), with an average next-session return of "
         f"{main['avg_return_pct']:+.2f}%." if main["n"] else f"{ft.symbol} did not fall {x0:g}% or more in this period.",
         "Large declines arrive in clusters, so these sessions are far fewer independent observations than their count suggests; the intervals resample whole weeks."]
    return rows, s, [{"label": f"Declines of {x0:g}%+", "value": str(main["n"])}, {"label": "Next close higher", "value": _pct(main["frequency"])}, {"label": "All sessions", "value": _pct(main["baseline"])}], \
        [_bar("Recovery after a decline", [r for r in rows if "next close" in r["label"]])], f"Investigate whether {ft.symbol} tends to reverse following large down days."


def streaks(ft, pm, p):
    rows = []
    for k in (1, 2, 3, 4, 5, 6):
        rows.append(_row(ft, [Pred("down_streak", ">=", float(k))], OutcomeSpec("close", "sessions:1", "up", 0.0, None, "close"), pm, f"{k}+ lower closes in a row : next close higher"))
    for k in (1, 2, 3, 4, 5, 6):
        rows.append(_row(ft, [Pred("up_streak", ">=", float(k))], OutcomeSpec("close", "sessions:1", "up", 0.0, None, "close"), pm, f"{k}+ higher closes in a row : next close higher"))
    a, b = rows[2], rows[8]
    s = [f"After three or more lower closes in a row ({a['n']} sessions) the next close was higher on {_pct(a['frequency'])}; after three or more higher closes ({b['n']} sessions), on {_pct(b['frequency'])}. All sessions: {_pct(a['baseline'])}.",
         "Both rows test \"next close higher\", so a value below the baseline means a decline was more likely."]
    return rows, s, [{"label": "After 3+ down", "value": _pct(a["frequency"])}, {"label": "After 3+ up", "value": _pct(b["frequency"])}, {"label": "All sessions", "value": _pct(a["baseline"])}], [_bar("Next close higher, by streak", rows)], \
        f"Investigate whether calls or puts perform differently following consecutive red days in {ft.symbol}."


def vwap(ft, pm, p):
    rows = []
    for t in ("10:00", "11:00", "12:00", "14:00"):
        rows.append(_row(ft, [Pred("vwap_dist", ">", 0)], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"above VWAP at {t} : closes higher than that price"))
        rows.append(_row(ft, [Pred("vwap_dist", "<", 0)], OutcomeSpec(t, "session", "down", 0.0, None, "close"), pm, f"below VWAP at {t} : closes lower than that price"))
    a = rows[0]
    s = [f"VWAP needs 5-minute or finer bars with volume; {a['evaluated']} sessions in this period have them." + (" That is too few for any conclusion: free sources keep only about 60 days of 5-minute history." if a["evaluated"] < 120 else ""),
         f"Above VWAP at 10:00 ET ({a['n']} sessions): finished higher on {_pct(a['frequency'])}." if a["n"] else "No session could be evaluated."]
    return rows, s, [{"label": "Sessions with 5-minute bars", "value": str(a["evaluated"])}], [_bar("Position against VWAP and the rest of the session", rows)], \
        f"Does being above or below VWAP in the morning predict the rest of the session for {ft.symbol}?"


def volatility_regimes(ft, pm, p):
    bands = [("VIX below 15", Pred("vix_prev", "<", 15)), ("VIX 15 to 20", Pred("vix_prev", "between", 15, 20)), ("VIX 20 to 25", Pred("vix_prev", "between", 20, 25)), ("VIX 25 to 30", Pred("vix_prev", "between", 25, 30)),
             ("VIX 30 or more", Pred("vix_prev", ">=", 30))]
    x = float(p["move"])
    rows = [_row(ft, [pr], OutcomeSpec("open", "session", "either", x, None, "touch"), pm, f"{lbl} : moves {x:g}%+ from the open") for lbl, pr in bands]
    rows += [_row(ft, [pr], OutcomeSpec("open", "session", "up", 0.0, None, "close"), pm, f"{lbl} : closes above the open") for lbl, pr in bands]
    lo, hi = rows[0], rows[4]
    s = [f"With the volatility index below 15 the day before, {ft.symbol} moved {x:g}% or more from its open on {_pct(lo['frequency'])} of {lo['n']} sessions; at 30 or more, on {_pct(hi['frequency'])} of {hi['n']}.",
         f"Direction is another matter: the session closed above its open on {_pct(rows[5]['frequency'])} and {_pct(rows[9]['frequency'])} of those sessions respectively.",
         "Volatility strongly predicts the size of moves and hardly their direction - which is why a condition that selects volatile days makes almost any price target look more reachable."]
    return rows, s, [{"label": "Sessions with VIX 30+", "value": str(hi["n"])}, {"label": f"{x:g}%+ move at VIX < 15", "value": _pct(lo["frequency"])}, {"label": f"{x:g}%+ move at VIX 30+", "value": _pct(hi["frequency"])}], \
        [_bar(f"Sessions moving {x:g}%+ from the open", rows[:5]), _bar("Sessions closing above the open", rows[5:])], f"What conditions are associated with large intraday moves in {ft.symbol}?"


def day_of_week(ft, pm, p):
    rows = [_row(ft, [Pred("weekday", "==", float(i))], OutcomeSpec("open", "session", "up", 0.0, None, "close"), pm, f"{WEEKDAYS[i]} : closes above the open") for i in range(5)]
    rows += [_row(ft, [Pred("weekday", "==", float(i))], OutcomeSpec("close", "sessions:1", "up", 0.0, None, "close"), pm, f"close of {WEEKDAYS[i]} : next close higher") for i in range(5)]
    best = max(rows[:5], key=lambda r: r["frequency"] or 0)
    worst = min(rows[:5], key=lambda r: r["frequency"] or 1)
    any_sig = any(r.get("p_value") is not None and r["p_value"] * 10 < 0.05 for r in rows)
    s = [f"{best['label'].split(' :')[0]}s closed above the open most often ({_pct(best['frequency'])} of {best['n']}), {worst['label'].split(' :')[0]}s least ({_pct(worst['frequency'])} of {worst['n']}).",
         "After correcting for the ten comparisons, " + ("at least one weekday differs from the others." if any_sig else "no weekday is distinguishable from the others: differences of this size appear by chance among five days.")]
    return rows, s, [{"label": "Highest", "value": f"{best['label'].split(' :')[0]} {_pct(best['frequency'])}"}, {"label": "Lowest", "value": f"{worst['label'].split(' :')[0]} {_pct(worst['frequency'])}"}], \
        [_bar("Closes above the open, by weekday", rows[:5])], f"Does {ft.symbol} behave differently on particular days of the week?"


def time_of_day(ft, pm, p):
    x = float(p["target"])
    rows = [_row(ft, [], OutcomeSpec(t, "session", "either", x, None, "touch"), pm, f"from {t} : moves {x:g}%+ before the close") for t in ("open", "10:30", "11:30", "12:30", "13:30", "14:30")]
    rows += [_row(ft, [], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"from {t} : closes higher than that price") for t in ("open", "10:30", "11:30", "12:30", "13:30", "14:30")]
    for r in rows:
        r["p_value"] = None  # an unconditional row has no comparison group
    s = [f"From the open, {ft.symbol} moved {x:g}% or more in either direction before the close on {_pct(rows[0]['frequency'])} of {rows[0]['n']} sessions; from 12:30 ET on {_pct(rows[3]['frequency'])} of {rows[3]['n']}; from 14:30 on {_pct(rows[5]['frequency'])}.",
         "The later the start, the less time and movement remain - the reason a fixed target is harder to reach for late entries and same-day options decay so quickly.",
         "Intraday rows use only sessions with hourly or finer bars."]
    return rows, s, [{"label": f"{x:g}%+ from the open", "value": _pct(rows[0]["frequency"])}, {"label": f"{x:g}%+ from 12:30", "value": _pct(rows[3]["frequency"])}], [_bar(f"Sessions moving {x:g}%+ in the time remaining", rows[:6], base_key=None)], \
        f"How does the chance of a {x:g}% move in {ft.symbol} depend on the time of day it is measured from?"


def drawdown_recovery(ft, pm, p):
    rows = []
    for dd in (-5, -10, -15, -20, -30):
        for hz, lbl in (("sessions:20", "higher 20 sessions later"), ("sessions:60", "higher 60 sessions later")):
            rows.append(_row(ft, [Pred("dist_high252", "<=", float(dd))], OutcomeSpec("close", hz, "up", 0.0, None, "close"), pm, f"{-dd}%+ below the 1-year high : {lbl}"))
    a = next(r for r in rows if r["label"].startswith("10%+") and "60" in r["label"])
    s = [f"{ft.symbol} closed 10% or more below its one-year high on {a['n']} sessions; 60 sessions later it was higher on {_pct(a['frequency'])} of them (all sessions: {_pct(a['baseline'])}), with an average change of {a['avg_return_pct']:+.1f}%." if a["n"] else
         f"{ft.symbol} was never 10% below its one-year high in this period.",
         "Sessions inside one drawdown are counted separately but are not independent: one bear market contributes hundreds of them. Week resampling widens the intervals accordingly, and they are still optimistic for horizons this long."]
    return rows, s, [{"label": "Sessions 10%+ below the high", "value": str(a["n"])}, {"label": "Higher after 60 sessions", "value": _pct(a["frequency"])}], [_bar("Higher 60 sessions later, by drawdown depth", [r for r in rows if "60" in r["label"]])], \
        f"Investigate whether {ft.symbol} tends to recover after falling 10% or more below its one-year high."


def cross_market(ft, pm, p):
    x = str(p["other"]).upper()
    if x not in ft.series:
        raise LookupError(f"No price history is stored for {x}.")
    rows = [_row(ft, [Pred(f"{x}_ret_1", "<=", -1.0)], OutcomeSpec("open", "session", "up", 0.0, None, "close"), pm, f"{x} fell 1%+ yesterday : {ft.symbol} closes above its open"),
            _row(ft, [Pred(f"{x}_ret_1", ">=", 1.0)], OutcomeSpec("open", "session", "up", 0.0, None, "close"), pm, f"{x} rose 1%+ yesterday : {ft.symbol} closes above its open"),
            _row(ft, [Pred(f"rs5_{x}", ">=", 2.0)], OutcomeSpec("close", "sessions:5", "up", 0.0, None, "close"), pm, f"{ft.symbol} beat {x} by 2+ pts over 5 sessions : higher 5 sessions later"),
            _row(ft, [Pred(f"rs5_{x}", "<=", -2.0)], OutcomeSpec("close", "sessions:5", "up", 0.0, None, "close"), pm, f"{ft.symbol} lagged {x} by 2+ pts over 5 sessions : higher 5 sessions later"),
            _row(ft, [Pred(f"{x}_gap", "<=", -0.5), Pred("gap", ">", -0.25)], OutcomeSpec("open", "session", "down", 0.0, None, "close"), pm, f"{x} gapped down 0.5%+ while {ft.symbol} did not : {ft.symbol} closes below its open")]
    c, o = ft.ser.d_c, ft.series[x].d_c
    dx = dict(zip(ft.series[x].days.tolist(), range(len(o))))
    idx = [(i, dx[d]) for i, d in enumerate(ft.ser.days.tolist()) if d in dx and i >= 1 and dx[d] >= 1 and pm[i]]
    ra = np.array([c[i] / c[i - 1] - 1 for i, _ in idx])
    rb = np.array([o[j] / o[j - 1] - 1 for _, j in idx])
    corr = float(np.corrcoef(ra, rb)[0, 1]) if len(idx) > 30 else None
    s = [f"Daily returns of {ft.symbol} and {x} have a correlation of {corr:.2f} over {len(idx)} shared sessions." if corr is not None else f"Too few shared sessions with {x}.",
         f"After {x} fell 1% or more ({rows[0]['n']} sessions), {ft.symbol} closed above its open on {_pct(rows[0]['frequency'])} the next day (all sessions: {_pct(rows[0]['baseline'])}).",
         f"When {ft.symbol} had lagged {x} by two points or more over five sessions ({rows[3]['n']} sessions) it was higher five sessions later on {_pct(rows[3]['frequency'])}."]
    return rows, s, [{"label": f"Correlation with {x}", "value": "n/a" if corr is None else f"{corr:.2f}"}, {"label": "Shared sessions", "value": str(len(idx))}], [_bar(f"{ft.symbol} after moves in {x}", rows)], \
        f"Does {x}'s previous-day move predict {ft.symbol}'s next session?"


def first_candle(ft, pm, p):
    rows = []
    for mm, t in ((15, "09:45"), (30, "10:00"), (60, "10:30")):
        rows.append(_row(ft, [Pred("from_open", ">", 0)], OutcomeSpec(t, "session", "up", 0.0, None, "close"), pm, f"first {mm} minutes up : rest of the session up"))
        rows.append(_row(ft, [Pred("from_open", "<", 0)], OutcomeSpec(t, "session", "down", 0.0, None, "close"), pm, f"first {mm} minutes down : rest of the session down"))
    a, b = rows[4], rows[5]
    s = [f"When the first hour was up ({a['n']} sessions) the rest of the session was also up on {_pct(a['frequency'])}; when it was down ({b['n']} sessions), the rest was down on {_pct(b['frequency'])}.",
         f"The 15- and 30-minute versions need finer bars and have {rows[0]['evaluated']} and {rows[2]['evaluated']} usable sessions."]
    return rows, s, [{"label": "First hour up -> rest up", "value": _pct(a["frequency"])}, {"label": "First hour down -> rest down", "value": _pct(b["frequency"])}], [_bar("Does the first candle carry over?", rows)], \
        f"Does the first 15-minute candle predict the rest of the session for {ft.symbol}?"


ENTRIES: list[dict] = [
    {"id": "overnight-gaps", "title": "Overnight gap behaviour", "category": "Overnight", "blurb": "How often a gap is faded or extended during the session, and how often it is fully retraced.", "fn": overnight_gaps,
     "params": [_param("size", "Gap size (%)", "number", 0.5, min=0.1, max=5, step=0.1)]},
    {"id": "opening-range", "title": "Opening range statistics", "category": "Intraday", "blurb": "What follows when price leaves the range of the first 15, 30 or 60 minutes.", "fn": opening_range,
     "params": [_param("minutes", "Opening range (minutes)", "choice", 30, choices=[15, 30, 60])]},
    {"id": "intraday-reversals", "title": "Intraday reversals", "category": "Intraday", "blurb": "Whether an early move away from the open tends to be given back by the close.", "fn": intraday_reversals,
     "params": [_param("move", "Early move from the open (%)", "number", 0.5, min=0.1, max=3, step=0.1)]},
    {"id": "decline-recovery", "title": "Recovery after a large decline", "category": "Previous sessions", "blurb": "What follows a sharp one-day decline over the next 1, 5 and 20 sessions.", "fn": decline_recovery,
     "params": [_param("decline", "Decline (%)", "number", 2, min=0.5, max=8, step=0.5)]},
    {"id": "streaks", "title": "Consecutive bullish and bearish sessions", "category": "Previous sessions", "blurb": "The next session after runs of higher or lower closes.", "fn": streaks, "params": []},
    {"id": "first-candle", "title": "First candle and the rest of the session", "category": "Intraday", "blurb": "Whether the direction of the first 15, 30 or 60 minutes carries over.", "fn": first_candle, "params": []},
    {"id": "vwap", "title": "VWAP interactions", "category": "Intraday", "blurb": "The rest of the session when price is above or below the session's volume-weighted average.", "fn": vwap, "params": []},
    {"id": "volatility-regimes", "title": "Volatility regime behaviour", "category": "Volatility", "blurb": "How the size and the direction of sessions change with the level of the volatility index.", "fn": volatility_regimes,
     "params": [_param("move", "Move from the open (%)", "number", 1, min=0.25, max=4, step=0.25)]},
    {"id": "day-of-week", "title": "Day-of-week differences", "category": "Calendar", "blurb": "Each weekday against the others, corrected for making five comparisons.", "fn": day_of_week, "params": []},
    {"id": "time-of-day", "title": "Time-of-day distributions", "category": "Intraday", "blurb": "How much movement is left in the session from each starting time.", "fn": time_of_day,
     "params": [_param("target", "Move (%)", "number", 0.5, min=0.1, max=3, step=0.1)]},
    {"id": "drawdown-recovery", "title": "Historical drawdown recovery", "category": "Previous sessions", "blurb": "Forward returns from different depths below the one-year high.", "fn": drawdown_recovery, "params": []},
    {"id": "cross-market", "title": "Cross-market relationships", "category": "Cross-symbol", "blurb": "What another index's previous move or relative strength says about the next session.", "fn": cross_market,
     "params": [_param("other", "Other symbol", "symbol", "QQQ")]},
]
BY_ID = {e["id"]: e for e in ENTRIES}


def listing() -> list[dict]:
    return [{k: e[k] for k in ("id", "title", "category", "blurb")} | {"params": COMMON + e["params"]} for e in ENTRIES]


def run_entry(entry_id: str, params: dict | None = None, store: MarketStore | None = None) -> dict:
    e = BY_ID.get(entry_id)
    if e is None:
        raise LookupError("unknown study")
    p = {x["name"]: x["default"] for x in COMMON + e["params"]} | {k: v for k, v in (params or {}).items() if v not in (None, "")}
    for x in e["params"]:
        if x["kind"] == "number":
            p[x["name"]] = min(max(float(p[x["name"]]), x.get("min", -1e9)), x.get("max", 1e9))
        elif x["kind"] == "choice":
            p[x["name"]] = type(x["default"])(p[x["name"]])
            if p[x["name"]] not in x["choices"]:
                raise ValueError(f"{x['label']} must be one of {x['choices']}")
    store = store or MarketStore()
    sym = str(p["symbol"]).upper()
    cross = tuple(c for c in ("QQQ", "IWM", "SPY", str(p.get("other") or "").upper()) if c and c != sym)
    ft = get_features(store, sym, cross)
    pm = ft.window(p.get("start"), p.get("end"))
    rows, summary, stats, charts, hyp = e["fn"](ft, pm, p)
    d = np.array(ft.dates)[pm]
    out = _finish(e, ft, rows, summary, stats, charts, hyp)
    out.update(params=p, window=[str(d[0]), str(d[-1])] if d.size else [None, None], sessions=int(pm.sum()), cross=list(cross))
    return out
