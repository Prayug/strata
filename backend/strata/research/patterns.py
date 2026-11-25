"""The historical market pattern explorer: what happened AFTER an event.

Pick an event ("SPY fell more than 2% in a session", "QQQ gapped down more than 1%",
"the volatility index rose 20% in a day", "three bearish sessions in a row"...) and this
module returns the distribution of the price paths that followed every historical
occurrence: the individual paths, their average and median, percentile bands, how far
price went for and against, forward returns at several horizons with calendar-clustered
intervals, target-before-stop frequencies, how long targets took, and the same split by
volatility and trend regime.

An event is a condition (catalog predicates and/or a rule-language expression) read at one
anchor instant: the open, the close, or a clock time. That makes every event a valid
research-graph condition and a valid strategy entry rule, which is what lets a pattern be
handed to the option backtester (`to_cell`). Everything is measured strictly after the
anchor; the event's own session is never part of its "after".

Nothing here is an option return, and overlapping forward windows are not independent
observations: intervals resample whole calendar weeks for that reason.
"""
from __future__ import annotations

import math

import numpy as np

from strata.data.store import MarketStore
from strata.research import evidence as EV
from strata.research import robust as R
from strata.research.features import A, ExprPred, OutcomeSpec, Pred, S as STOP, T, get_features

HORIZONS = [1, 2, 3, 5, 10, 20]
MAX_PATHS_OUT = 400


def _e(id_, title, group, anchor, params, build, describe, needs="daily"):
    return {"id": id_, "title": title, "group": group, "anchor": anchor, "params": params, "build": build, "describe": describe, "needs": needs}


def _n(name, label, default, lo, hi, step, unit="%"):
    return {"name": name, "label": label, "default": default, "min": lo, "max": hi, "step": step, "unit": unit}


EVENTS = {e["id"]: e for e in [
    _e("down_day", "Large down day", "Session moves", "close", [_n("x", "Decline of at least", 2.0, 0.25, 10, 0.25)],
       lambda p, s: [Pred("ret_1", "<=", -p["x"])], lambda p, s: f"{s} closes {p['x']:g}% or more below the previous close"),
    _e("up_day", "Large up day", "Session moves", "close", [_n("x", "Rise of at least", 2.0, 0.25, 10, 0.25)],
       lambda p, s: [Pred("ret_1", ">=", p["x"])], lambda p, s: f"{s} closes {p['x']:g}% or more above the previous close"),
    _e("gap_down", "Gap down at the open", "Overnight", "open", [_n("x", "Gap of at least", 1.0, 0.1, 8, 0.1)],
       lambda p, s: [Pred("gap", "<=", -p["x"])], lambda p, s: f"{s} opens {p['x']:g}% or more below the previous close"),
    _e("gap_up", "Gap up at the open", "Overnight", "open", [_n("x", "Gap of at least", 1.0, 0.1, 8, 0.1)],
       lambda p, s: [Pred("gap", ">=", p["x"])], lambda p, s: f"{s} opens {p['x']:g}% or more above the previous close"),
    _e("vol_spike", "Volatility index jumps", "Volatility", "close", [_n("x", "One-day rise of at least", 20.0, 5, 100, 5)],
       lambda p, s: [ExprPred(f'return_pct("{{VOL}}", 1) >= {p["x"]:g}', f"the volatility index rose {p['x']:g}% or more in one session")],
       lambda p, s: f"the volatility index rises {p['x']:g}% or more in one session"),
    _e("vol_level", "Volatility index above a level", "Volatility", "open", [_n("x", "Previous close above", 30.0, 10, 80, 1, "")],
       lambda p, s: [Pred("vix_prev", ">=", p["x"])], lambda p, s: f"the volatility index closed at {p['x']:g} or higher the session before"),
    _e("break_week_low", "Close below last week's low", "Levels", "close", [],
       lambda p, s: [ExprPred("sma(1) < prev_week_low() and close(2) >= prev_week_low()", "the close broke below the previous week's low")],
       lambda p, s: f"{s} closes below the previous calendar week's low, having closed above it the session before"),
    _e("break_week_high", "Close above last week's high", "Levels", "close", [],
       lambda p, s: [ExprPred("sma(1) > prev_week_high() and close(2) <= prev_week_high()", "the close broke above the previous week's high")],
       lambda p, s: f"{s} closes above the previous calendar week's high, having closed below it the session before"),
    _e("red_streak", "Consecutive bearish sessions", "Streaks", "close", [_n("n", "Sessions in a row", 3, 2, 10, 1, "")],
       lambda p, s: [Pred("red_streak", "==", float(int(p["n"])))], lambda p, s: f"{int(p['n'])} sessions in a row have closed below their open (exactly {int(p['n'])}, so each streak is one event)"),
    _e("green_streak", "Consecutive bullish sessions", "Streaks", "close", [_n("n", "Sessions in a row", 3, 2, 10, 1, "")],
       lambda p, s: [Pred("green_streak", "==", float(int(p["n"])))], lambda p, s: f"{int(p['n'])} sessions in a row have closed above their open (exactly {int(p['n'])})"),
    _e("drawdown", "Drawdown from the 1-year high", "Levels", "close", [_n("x", "At least", 10.0, 2, 50, 1)],
       lambda p, s: [Pred("dist_high252", "<=", -p["x"]), ExprPred(f"dist_pct(close(2), rolling_high(252)) > -{p['x']:g}", "the previous close was not yet that far below it")],
       lambda p, s: f"{s} closes {p['x']:g}% or more below its 1-year high for the first time"),
    _e("opening_volume", "Unusually heavy opening candle", "Intraday", "tick", [_n("x", "Volume at least", 2.0, 1.2, 6, 0.1, "x"), {"name": "tf", "label": "Candle", "default": "30m", "options": ["5m", "15m", "30m", "60m"]}],
       lambda p, s: [ExprPred(f'volume_ratio("{p["tf"]}", 20) >= {p["x"]:g}', f"the first {p['tf']} candle traded {p['x']:g}x the volume of the previous 20")],
       lambda p, s: f"the first {p['tf']} candle of the session trades {p['x']:g} times the average volume of the previous 20 candles", "intraday"),
    _e("range_day", "Unusually wide session", "Session moves", "close", [_n("x", "High-low range of at least", 3.0, 0.5, 12, 0.25)],
       lambda p, s: [Pred("range_1", ">=", p["x"])], lambda p, s: f"{s}'s session range is {p['x']:g}% of its open or more"),
    _e("custom", "Custom rule", "Custom", "close", [{"name": "expr", "label": "Rule (rule language)", "default": "consecutive_down_days() >= 3", "kind": "expr"}, {"name": "at", "label": "Read at", "default": "close", "kind": "tick"}],
       lambda p, s: [ExprPred(str(p["expr"]), str(p["expr"]))], lambda p, s: str(p["expr"])),
]}


def catalog() -> list[dict]:
    return [{"id": e["id"], "title": e["title"], "group": e["group"], "anchor": e["anchor"], "params": e["params"], "needs": e["needs"]} for e in EVENTS.values()]


def resolve(event: dict, symbol: str, store: MarketStore) -> dict:
    """{"preds", "anchor", "text", "params"} for an event spec {"id", "params": {...}}."""
    e = EVENTS.get(event.get("id"))
    if e is None:
        raise ValueError(f"unknown event {event.get('id')!r}")
    p = {x["name"]: x["default"] for x in e["params"]}
    for k, v in (event.get("params") or {}).items():
        if k in p and v is not None:
            p[k] = v if isinstance(p[k], str) else float(v)
    for x in e["params"]:
        if "min" in x and not x["min"] <= float(p[x["name"]]) <= x["max"]:
            raise ValueError(f"{x['label']}: {p[x['name']]} is outside {x['min']} to {x['max']}")
        if "options" in x and p[x["name"]] not in x["options"]:
            raise ValueError(f"{x['label']}: choose one of {', '.join(x['options'])}")
    anchor = e["anchor"]
    if e["id"] == "custom":
        from strata.engine import expr as X
        bad = X.validate(str(p["expr"]))
        if bad:
            raise ValueError(f"the rule is not valid: {bad[0]}")
        if X.uses_position(str(p["expr"])):
            raise ValueError("the rule reads an open position, which does not exist in a market study")
        anchor = str(p["at"])
        if X.needs_intraday(str(p["expr"])) and anchor in ("open", "close"):
            raise ValueError("this rule reads the session so far, so it needs a clock time (for example 10:30)")
    if anchor == "tick":
        mins = int(str(p["tf"]).rstrip("m"))
        anchor = f"{(570 + mins) // 60:02d}:{(570 + mins) % 60:02d}"
    vol = store.symbol_meta(symbol).vol_index or "VIX"
    preds = [ExprPred(q.src.replace("{VOL}", vol), q.label) if isinstance(q, ExprPred) else q for q in e["build"](p, symbol)]
    return {"id": e["id"], "title": e["title"], "preds": preds, "anchor": anchor, "text": e["describe"](p, symbol), "params": p}


# ----------------------------------------------------------------------------- paths
def _closes(ft) -> np.ndarray:
    s = ft.ser
    return s.c[s.day_end - 1]


def forward_matrix(ft, idx: np.ndarray, entry: str, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """(close path, running high, running low) as % from the entry price, one row per event.
    Column 0 is the entry itself. Cells beyond the stored history are NaN."""
    s = ft.ser
    c, hi, lo = _closes(ft), s.d_h, s.d_l
    oc = ft.outcome(OutcomeSpec(entry, "sessions:1" if entry == "close" else "session", "up", 0.0, None, "close"))
    px = oc.entry_px[idx]
    same = entry != "close"  # the entry session's own close is the first step unless the entry IS that close
    steps = (["entry", "close"] if same else ["event close"]) + [f"+{k}" for k in range(1, H + 1)]
    off = np.arange(0 if same else 1, H + 1)
    cols = idx[:, None] + off[None, :]
    ok = cols < ft.n
    cc = np.where(ok, c[np.minimum(cols, ft.n - 1)], np.nan)
    hh = np.where(ok, hi[np.minimum(cols, ft.n - 1)], np.nan)
    ll = np.where(ok, lo[np.minimum(cols, ft.n - 1)], np.nan)
    if same and entry != "open":
        # an intraday entry: the entry session's high/low before the entry are not part of "after". Its excursions come
        # from the bars after the entry (the outcome engine), so the first step uses those instead of the daily bar
        hh[:, 0] = px * (1 + np.where(np.isfinite(oc.mfe[idx]), oc.mfe[idx], 0) / 100)
        ll[:, 0] = px * (1 + np.where(np.isfinite(oc.mae[idx]), oc.mae[idx], 0) / 100)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (cc / px[:, None] - 1) * 100
        HI = (np.fmax.accumulate(np.where(np.isfinite(hh), hh, -np.inf), axis=1) / px[:, None] - 1) * 100
        LO = (np.fmin.accumulate(np.where(np.isfinite(ll), ll, np.inf), axis=1) / px[:, None] - 1) * 100
    HI[~ok], LO[~ok] = np.nan, np.nan
    z = np.zeros((idx.size, 1))
    return np.hstack([z, P]), np.hstack([z, np.maximum(HI, 0)]), np.hstack([z, np.minimum(LO, 0)]), steps, px


def _band(M: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = np.nanquantile(M, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0) if M.shape[0] else np.full((5, M.shape[1]), np.nan)
            mean = np.nanmean(M, axis=0) if M.shape[0] else np.full(M.shape[1], np.nan)
    f = lambda a: [None if not math.isfinite(x) else round(float(x), 4) for x in a]  # noqa: E731
    return {"mean": f(mean), "p05": f(q[0]), "p25": f(q[1]), "median": f(q[2]), "p75": f(q[3]), "p95": f(q[4]), "n": [int(x) for x in np.isfinite(M).sum(axis=0)]}


def paths(ft, q: np.ndarray, entry: str, H: int = 10, rest: np.ndarray | None = None, reps: int = 400, keep: int = MAX_PATHS_OUT) -> dict:
    """Summary of the paths that followed the qualifying sessions `q` (and, as a reference,
    the sessions in `rest`)."""
    idx = np.flatnonzero(q)
    if idx.size == 0:
        return {"n": 0, "steps": [], "bands": None, "baseline": None, "paths": [], "dates": []}
    P, HI, LO, steps, px = forward_matrix(ft, idx, entry, H)
    out = {"n": int(idx.size), "steps": steps, "bands": _band(P), "mfe": _band(HI), "mae": _band(LO), "baseline": None, "entry": entry}
    if rest is not None and rest.any():
        ridx = np.flatnonzero(rest)
        if ridx.size > 6000:
            ridx = ridx[np.linspace(0, ridx.size - 1, 6000).astype(int)]
        PB = forward_matrix(ft, ridx, entry, H)[0]
        b = _band(PB)
        out["baseline"] = {"mean": b["mean"], "median": b["median"], "p25": b["p25"], "p75": b["p75"], "n": int(ridx.size)}
    sel = idx[-keep:]
    out["paths"] = [[None if not math.isfinite(x) else round(float(x), 3) for x in row] for row in P[-keep:]]
    out["dates"] = [ft.dates[i] for i in sel]
    out["shown"] = int(sel.size)
    return out


def _horizon_rows(ft, q: np.ndarray, rest: np.ndarray, entry: str, horizons: list[int], reps: int) -> list[dict]:
    dates = np.array(ft.dates)
    rows = []
    for h in horizons:
        hz = f"sessions:{h}" if (h > 0 or entry == "close") else "session"
        if entry != "close" and h == 0:
            hz = "session"
        oc = ft.outcome(OutcomeSpec(entry, hz, "up", 0.0, None, "close"))
        a, b = q & oc.valid, rest & oc.valid
        ra, rb = oc.end_ret[a], oc.end_ret[b]
        row = {"horizon": h, "label": ("to the close" if hz == "session" else f"{h} session{'s' if h != 1 else ''} later"), "n": int(a.sum()), "mean": float(ra.mean()) if ra.size else None,
               "median": float(np.median(ra)) if ra.size else None, "positive": float((ra > 0).mean()) if ra.size else None, "positive_ci95": R.wilson(int((ra > 0).sum()), int(ra.size)),
               "p05": float(np.quantile(ra, 0.05)) if ra.size >= 20 else None, "p95": float(np.quantile(ra, 0.95)) if ra.size >= 20 else None,
               "rest_n": int(b.sum()), "rest_mean": float(rb.mean()) if rb.size else None, "rest_positive": float((rb > 0).mean()) if rb.size else None,
               "mean_ci95": [None, None], "difference": None, "p_value": None, "avg_mfe": float(np.nanmean(oc.mfe[a])) if a.any() else None, "avg_mae": float(np.nanmean(oc.mae[a])) if a.any() else None, "thin": int(a.sum()) < 30}
        if ra.size >= 6:
            ci = R.cluster_mean_ci(ra, dates[a], "week", reps)
            row["mean_ci95"] = ci["ci"]
            row["positive_ci95"] = R.cluster_prop_ci((ra > 0).astype(float), dates[a], "week", reps)["ci"]
        if ra.size >= 10 and rb.size >= 10:
            md = R.mean_diff_test(ra, dates[a], rb, dates[b], "week", reps)
            row.update(difference=md.get("diff", (row["mean"] - row["rest_mean"])), p_value=md.get("p_value"))
        rows.append(row)
    for r_, a_ in zip(rows, R.adjust([r_["p_value"] for r_ in rows], "holm")):
        r_["p_adjusted"] = a_
    return rows


def _target(ft, q: np.ndarray, rest: np.ndarray, entry: str, H: int, direction: str, target: float, stop: float | None, reps: int) -> dict:
    hz = f"sessions:{H}"
    oc_spec = OutcomeSpec(entry, hz, direction, float(target), None if not stop else float(stop), "touch")
    oc = ft.outcome(oc_spec)
    e = EV.compare(ft.dates, oc.code, oc.end_ret, q & oc.valid, (q | rest) & oc.valid, direction, reps=reps)
    idx = np.flatnonzero(q & oc.valid)
    out = {"target": target, "stop": stop, "direction": direction, "within_sessions": H, "evidence": e, "outcome_text": oc_spec.text(),
           "stopped": int((oc.code[idx] == STOP).sum()), "ambiguous": int((oc.code[idx] == A).sum()), "time_to_target": None, "outcome": oc_spec.dump()}
    if idx.size:
        P, HI, LO, steps, _ = forward_matrix(ft, idx, entry, H)
        hit = oc.code[idx] == T
        reach = (HI >= target) if direction == "up" else (LO <= -target)
        first = np.where(reach.any(axis=1), reach.argmax(axis=1), -1)
        f = first[hit & (first >= 0)]
        if f.size:
            cnt = np.bincount(f, minlength=len(steps))
            out["time_to_target"] = {"steps": steps, "counts": [int(x) for x in cnt], "median_step": steps[int(np.median(f))], "n": int(f.size),
                                     "note": "The step is the first daily bar whose range reached the target; daily bars cannot place it inside the session."}
    return out


def _regimes(ft, q: np.ndarray, entry: str, h: int, reps: int) -> list[dict]:
    """The same forward return, split by where volatility and trend stood at the anchor."""
    tick = "close" if entry == "close" else "open"
    oc = ft.outcome(OutcomeSpec(entry, f"sessions:{h}" if (h > 0 or entry == "close") else "session", "up", 0.0, None, "close"))
    ok = q & oc.valid
    dates = np.array(ft.dates)
    out = []
    specs = [("Volatility (1-year percentile of realised volatility)", ft.col("rv_pctile", tick), [(0, 33.34, "calm third"), (33.34, 66.67, "middle third"), (66.67, 101, "turbulent third")]),
             ("Trend (close against the 200-day average)", ft.col("sma200_dist", tick), [(-1e9, 0, "below the 200-day average"), (0, 1e9, "above the 200-day average")]),
             ("Volatility index (previous close)", ft.col("vix_prev", "open"), [(0, 15, "below 15"), (15, 20, "15 to 20"), (20, 30, "20 to 30"), (30, 1e9, "30 and above")])]
    for title, x, bins in specs:
        rows = []
        for lo, hi, name in bins:
            m = ok & np.isfinite(x) & (x >= lo) & (x < hi)
            r = oc.end_ret[m]
            row = {"regime": name, "n": int(m.sum()), "mean": float(r.mean()) if r.size else None, "median": float(np.median(r)) if r.size else None, "positive": float((r > 0).mean()) if r.size else None,
                   "mean_ci95": R.cluster_mean_ci(r, dates[m], "week", reps)["ci"] if r.size >= 6 else [None, None], "thin": int(m.sum()) < 30}
            rows.append(row)
        if sum(r["n"] for r in rows):
            out.append({"title": title, "rows": rows, "missing": int((ok & ~np.isfinite(x)).sum())})
    return out


def _event_masks(ft, ev: dict):
    """(qualifying, evaluable) for an event's predicates at its anchor."""
    q = np.ones(ft.n, bool)
    any_false = np.zeros(ft.n, bool)
    all_known = np.ones(ft.n, bool)
    for p in ev["preds"]:
        m, fin = p.truth(ft, ev["anchor"])
        any_false |= fin & ~m
        all_known &= fin
        q &= m
    known = any_false | all_known
    q &= all_known
    q[0] = False
    return q, known


def explore(symbol: str, event: dict, store: MarketStore | None = None, start: str | None = None, end: str | None = None, horizon: int = 10, target: float | None = None, stop: float | None = None,
            direction: str = "up", overlays: list[dict] | None = None, reps: int = 400, main_horizon: int | None = None) -> dict:
    """Everything the pattern explorer shows for one event (and, optionally, overlaid ones)."""
    store = store or MarketStore()
    sym = symbol.upper()
    H = max(1, min(int(horizon), 60))
    ev = resolve(event, sym, store)
    from strata.engine import expr as X
    cross: set[str] = set()
    over = []
    for o in overlays or []:  # an overlay that cannot be resolved is reported in its own row, it does not fail the page
        try:
            over += resolve(o, sym, store)["preds"]
        except (ValueError, LookupError):
            pass
    for p in ev["preds"] + over:
        if isinstance(p, ExprPred):
            cross |= X.symbols_used(p.src)
    ft = get_features(store, sym, tuple(sorted(cross - {sym})))
    if ft.missing_cross:
        raise LookupError(f"No price history is stored for {', '.join(ft.missing_cross)}.")
    pm = ft.window(start, end)
    q, known = _event_masks(ft, ev)
    _, observable = ft._tick(ev["anchor"])
    base = known & pm & observable
    q &= base
    rest = base & ~q
    hs = [h for h in ([0] if ev["anchor"] != "close" else []) + HORIZONS if h <= H]
    mh = main_horizon if main_horizon in hs else (5 if 5 in hs else hs[-1])
    out = {"kind": "pattern", "symbol": sym, "event": {"id": ev["id"], "title": ev["title"], "text": ev["text"], "params": ev["params"], "anchor": ev["anchor"]},
           "window": [ft.dates[int(np.flatnonzero(base)[0])], ft.dates[int(np.flatnonzero(base)[-1])]] if base.any() else [None, None],
           "n": int(q.sum()), "evaluated": int(base.sum()), "not_evaluable": int((pm & ~base).sum()), "frequency": (float(q.sum()) / int(base.sum())) if base.any() else None,
           "dates": [ft.dates[i] for i in np.flatnonzero(q)], "horizon": H, "main_horizon": mh, "notes": [], "overlays": [], "synthetic": store.is_synthetic(sym)}
    if not q.any():
        out["notes"].append("No stored session satisfies this event. Loosen the threshold, or check that the data it reads is stored.")
        return out
    out["paths"] = paths(ft, q, ev["anchor"], H, rest, reps)
    out["horizons"] = _horizon_rows(ft, q, rest, ev["anchor"], hs, reps)
    out["regimes"] = _regimes(ft, q, ev["anchor"], mh, reps)
    if target:
        out["target"] = _target(ft, q, rest, ev["anchor"], H, direction, float(target), stop, reps)
    yrs: dict[str, int] = {}
    for d in out["dates"]:
        yrs[d[:4]] = yrs.get(d[:4], 0) + 1
    out["by_year"] = [{"year": y, "n": n} for y, n in sorted(yrs.items())]
    top = max(yrs.values()) / out["n"]
    if top > 0.35:
        out["notes"].append(f"{max(yrs, key=yrs.get)} alone supplies {top * 100:.0f}% of the occurrences: the picture is dominated by one episode.")
    idx = np.flatnonzero(q)
    if idx.size > 1:
        gaps = np.diff(idx)
        ov = float((gaps <= mh).mean())
        if ov > 0.25:
            out["notes"].append(f"{ov * 100:.0f}% of occurrences fall within {mh} sessions of the previous one, so their forward windows overlap and are not independent observations. Intervals resample whole calendar weeks to allow for it.")
    if out["n"] < 30:
        out["notes"].append(f"Only {out['n']} occurrences: averages and bands are descriptive, not evidence of a tendency.")
    out["notes"].append("Underlying price behaviour only. Whether an option position would have made money on these dates is a separate question, answered on observed option prices.")
    out["cell"] = to_cell(ev, mh, direction, target, stop)
    for o in overlays or []:
        try:
            oe = resolve(o, sym, store)
            oq, okn = _event_masks(ft, oe)
            _, oobs = ft._tick(oe["anchor"])
            ob = okn & pm & oobs
            oq &= ob
            row = {"event": {"id": oe["id"], "title": oe["title"], "text": oe["text"], "params": oe["params"], "anchor": oe["anchor"]}, "n": int(oq.sum()), "evaluated": int(ob.sum()),
                   "overlap_with_main": int((oq & q).sum())}
            if oq.any():
                pp = paths(ft, oq, oe["anchor"], H, None, reps, keep=0)
                row.update(steps=pp["steps"], bands=pp["bands"], horizons=_horizon_rows(ft, oq, ob & ~oq, oe["anchor"], [h for h in hs if h > 0 or oe["anchor"] != "close"], reps))
            out["overlays"].append(row)
        except (ValueError, LookupError) as ex:
            out["overlays"].append({"event": o, "error": str(ex)[:200]})
    return out


def to_cell(ev: dict, horizon: int, direction: str = "up", target: float | None = None, stop: float | None = None) -> dict:
    """The event as a (condition, outcome) pair that hypothesis plans, the research graph and
    the option-structure drafter all accept."""
    hz = f"sessions:{max(1, horizon)}" if (horizon > 0 or ev["anchor"] == "close") else "session"
    oc = OutcomeSpec(ev["anchor"], hz, direction, float(target) if target else 0.0, float(stop) if stop else None, "touch" if target else "close")
    return {"preds": [p.dump() for p in ev["preds"]], "outcome": oc.dump(), "text": ev["text"]}
