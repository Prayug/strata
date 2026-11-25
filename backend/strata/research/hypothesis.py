"""The Hypothesis Lab: broad market questions turned into explicit, bounded research plans.

    "Investigate whether SPY tends to reverse following large down days."
    "Does the first 15-minute candle predict the rest of the session?"
    "What conditions are associated with large intraday moves?"
    "Compare mean reversion and momentum after overnight gaps."
    "How does the effectiveness of opening range breakouts change with volatility?"
    "Investigate whether calls or puts perform differently following consecutive red days."
    "What historical conditions are associated with a 1% rally before a 0.5% decline?"

A plan states, before anything is computed: the hypothesis and its null, the baseline, the
data it needs, every candidate condition and parameter value it will try (a fixed list, at
most `max_configs`), the primary metric, the minimum sample, and the training, validation
and final test periods. `run_plan` then evaluates every configuration with the
deterministic feature engine and reports ALL of them, including the ones that show
nothing. It is not a search for an attractive equity curve:

  * selection uses the training period only; the validation period can only veto
  * significance uses calendar-cluster tests (observations in the same week travel
    together) and is corrected for every configuration tried in the research session
  * a candidate must also agree with its neighbouring parameter values
  * the final test period is managed by the research ledger: it is scored once, for the
    candidates only, and only while it is still untouched
  * this stage measures the UNDERLYING. Whether an option position would have made money
    is a separate question, answered on observed premiums by `structure_variants`.
"""
from __future__ import annotations

import math
import re

import numpy as np

from strata.data.store import MarketStore
from strata.research import evidence as EV
from strata.research import ledger as LG
from strata.research import robust as R
from strata.research.features import A, CODE_NAMES, FEATURES, OutcomeSpec, Pred, SessionFeatures, T, cond_text, feature_def, get_features, to_study

MAX_CONFIGS, MIN_SAMPLE, MIN_VALIDATE, MIN_TEST, MAX_CANDIDATES = 60, 30, 8, 8, 5
PLAN_REPS = 1000
NUM = r"(\d+(?:\.\d+)?)"


# ----------------------------------------------------------------------------- plan construction
def _cfg(family: str, x: float, preds: list[Pred], oc: OutcomeSpec, sym: str, tag: str = "") -> dict:
    return {"family": family, "x": x, "preds": [p.dump() for p in preds], "outcome": oc.dump(), "label": (cond_text(preds, sym) + (f" [{tag}]" if tag else "")), "tag": tag}


def _finish(plan: dict) -> dict:
    """Keys, bounds and the standard fields every plan carries."""
    seen, out = set(), []
    for c in plan["configs"]:
        key = LG.config_key({"symbol": plan["symbol"], "preds": c["preds"], "outcome": c["outcome"]})
        if key in seen:
            continue
        seen.add(key)
        out.append({**c, "key": key})
    plan["dropped_for_bound"] = max(0, len(out) - plan.get("max_configs", MAX_CONFIGS))
    plan["configs"] = out[:plan.get("max_configs", MAX_CONFIGS)]
    plan.setdefault("max_configs", MAX_CONFIGS)
    plan.setdefault("min_sample", MIN_SAMPLE)
    plan.setdefault("primary_metric", "frequency")
    plan.setdefault("stage", "underlying")
    plan.setdefault("controls", [])
    plan.setdefault("notes", [])
    plan["metric_label"] = {"frequency": "how much more often the outcome happens on qualifying sessions than on all other evaluated sessions (percentage points)",
                            "return": "the average return of qualifying sessions minus that of all other evaluated sessions"}[plan["primary_metric"]]
    plan["baseline"] = plan.get("baseline") or "Every other evaluated session in the same period, observed from the same entry over the same horizon."
    fams: dict[str, dict] = {}
    for c in plan["configs"]:
        f = fams.setdefault(c["family"], {"family": c["family"], "configs": 0, "values": []})
        f["configs"] += 1
        f["values"].append(c["x"])
    plan["families"] = list(fams.values())
    return plan


def _thresh_family(sym: str, feat: str, op: str, values: list[float], oc: OutcomeSpec, extra: list[Pred] | None = None, tag: str = "") -> list[dict]:
    fam = f"{feat} {op} | {oc.key()}" + (f" | {tag}" if tag else "")
    return [_cfg(fam, float(v), [Pred(feat, op, float(v))] + list(extra or []), oc, sym, tag) for v in values]


def build_plan(question: str, symbol: str, store: MarketStore | None = None, known: list[str] | None = None) -> dict | None:
    """A bounded research plan for a broad market question, or None when the question is not
    one of the supported hypothesis families (the caller then says so rather than guessing)."""
    low = " ".join(question.lower().split())
    sym = symbol.upper()
    known = [k.upper() for k in (known or [])]
    cross = [x for x in ("QQQ", "IWM", "SPY") if x != sym and (not known or x in known)][:2]
    P: dict = {"question": question.strip(), "symbol": sym, "cross": cross, "configs": [], "notes": []}

    # ---- "a 1% rally before a 0.5% decline": which conditions go with a target-before-stop outcome
    m = re.search(rf"{NUM}\s*%\s*(rally|rise|gain|advance|up ?move|bounce|increase)\s*(?:before|prior to|ahead of)\s*(?:an?\s*)?{NUM}\s*%\s*(decline|drop|fall|loss|pullback|sell-?off|down ?move|decrease)", low) or \
        re.search(rf"{NUM}\s*%\s*(decline|drop|fall|loss|sell-?off|down ?move|decrease)\s*(?:before|prior to|ahead of)\s*(?:an?\s*)?{NUM}\s*%\s*(rally|rise|gain|advance|up ?move|bounce|increase)", low)
    if m:
        up = bool(re.match(r"rally|rise|gain|advance|up|bounce|increase", m.group(2)))
        tgt, stp = float(m.group(1)), float(m.group(3))
        oc = OutcomeSpec("open", "session", "up" if up else "down", tgt, stp, "touch")
        P.update(hypothesis=f"Some conditions known before the open make a {tgt:g}% {'rally' if up else 'decline'} before a {stp:g}% {'decline' if up else 'rally'} (from the open, within the session) more likely than usual for {sym}.",
                 null=f"No condition changes how often {sym} trades {tgt:g}% {'above' if up else 'below'} its open before {stp:g}% the other way.",
                 outcomes=[{"label": oc.text(), **oc.dump()}])
        P["configs"] = _scan(sym, oc, cross)
        P["notes"] += [f"\"Before\" is read within one session, measured from the open. When both levels fall inside the same price bar the order cannot be known and the session is counted as ambiguous, not guessed.",
                       "This is a scan of many candidate conditions, so a strict correction for the number tried is applied; expect most of them to show nothing."]
        P["controls"] = ["Sessions with wider ranges touch both levels more often; a condition only counts if the session also ends further in that direction."]
        return _finish(P)

    # ---- "what conditions are associated with large intraday moves"
    if re.search(r"(large|big|outsized|wide|unusual|extreme)\w*\s+(intraday|intra-day|daily|session)\s+(moves?|ranges?|swings?)|large moves? (?:during|in) the (?:day|session)|high[- ]range (?:days|sessions)", low):
        x = re.search(rf"{NUM}\s*%", low)
        size = float(x.group(1)) if x else 1.0
        oc = OutcomeSpec("open", "session", "either", size, None, "touch")
        P.update(hypothesis=f"Some conditions known before the open are followed by {sym} moving {size:g}% or more away from its open during the session more often than usual.",
                 null=f"No condition changes how often {sym} moves {size:g}% from its open.", outcomes=[{"label": oc.text(), **oc.dump()}])
        P["configs"] = _scan(sym, oc, cross, vol_first=True)
        P["notes"] += [("\"Large\" was not given a number, so " if not x else "") + f"a move of {size:g}% from the open in either direction counts as large.",
                       "This says how MUCH the underlying moved, not which way: it is relevant to positions that profit from movement (straddles) or from its absence (iron condors), not to calls or puts alone."]
        P["controls"] = ["Volatility clusters: large moves follow large moves. Conditions are therefore tested with calendar-cluster statistics rather than as independent draws."]
        return _finish(P)

    # ---- "does the first N-minute candle predict the rest of the session"
    m = re.search(r"first (\w+)[- ]?(?:minute|min)s?(?: candle| bar)?", low)
    if m and re.search(r"predict|forecast|foreshadow|tell|signal|indicat|determin|carry|rest of the (?:day|session)", low):
        from strata.ai.nl_rules import _num
        mins = int(_num(m.group(1)) or 15)
        cfgs = []
        for mm in sorted({mins, 30, 60}):
            t = f"{(570 + mm) // 60:02d}:{(570 + mm) % 60:02d}"
            for d, op in (("up", ">"), ("down", "<")):
                oc = OutcomeSpec(t, "session", d, 0.0, None, "close")
                vals = [0.0, 0.25, 0.5] if d == "up" else [0.0, -0.25, -0.5]
                cfgs += _thresh_family(sym, "from_open", ">=" if (d == "up" and False) else op, vals, oc, tag=f"first {mm} min")
        P.update(hypothesis=f"The direction of the first {mins} minutes carries over: {sym} finishes the session further in the same direction more often than not.",
                 null="The move from the end of the first candle to the close is unrelated to the candle's direction.",
                 outcomes=[{"label": "the close is beyond the price at the end of the first candle, in the candle's direction", "entry": f"after the first {mins} minutes"}], configs=cfgs)
        P["notes"] += [f"The first {mins}-minute candle is compared with 30- and 60-minute versions so one window is not singled out.",
                       "Needs intraday bars that can observe the end of the candle; sessions that only have coarser bars are excluded and counted."]
        return _finish(P)

    # ---- opening range breakouts (and volatility)
    if re.search(r"opening[- ]range|\borb\b", low):
        cfgs = []
        vix_bands = [("VIX below 15", Pred("vix_prev", "<", 15)), ("VIX 15 to 20", Pred("vix_prev", "between", 15, 20)), ("VIX 20 to 30", Pred("vix_prev", "between", 20, 30)), ("VIX 30 or more", Pred("vix_prev", ">=", 30))]
        for feat, entries in (("or30_pos", ("10:30", "11:00")), ("or60_pos", ("11:30", "12:30"))):
            for t in entries:
                for d, pr in (("up", Pred(feat, ">", 100)), ("down", Pred(feat, "<", 0))):
                    oc = OutcomeSpec(t, "session", d, 0.0, None, "close")
                    cfgs.append(_cfg(f"{feat} break {d} | {t}", 0.0, [pr], oc, sym, "any volatility"))
                    for j, (lbl, vp) in enumerate(vix_bands):
                        cfgs.append(_cfg(f"{feat} break {d} | {t} | by VIX", float(j + 1), [pr, vp], oc, sym, lbl))
        P.update(hypothesis=f"When {sym} is trading outside its opening range, it finishes the session further in that direction more often than not, and how often depends on the level of volatility.",
                 null="A session that is outside its opening range finishes beyond the breakout price no more often than any other session finishes beyond the same clock-time price.",
                 outcomes=[{"label": "the close is beyond the price at the observation time, in the direction of the breakout"}], configs=cfgs)
        P["notes"] += ["A breakout is read as: at the observation time the price is above the high (or below the low) of the first 30 or 60 minutes.",
                       "Volatility is the volatility index's previous close, in four bands. The bands are fixed in advance, not fitted.",
                       "The 30-minute range needs 30-minute or finer bars; the 60-minute version is included because more history is stored at hourly resolution."]
        return _finish(P)

    # ---- overnight gaps: mean reversion versus momentum
    if re.search(r"\bgaps?\b|gap(?:s|ped|ping)? (?:up|down)|overnight", low):
        cfgs = []
        for d_gap, op, vals in (("down", "<=", [-0.5, -1.0, -1.5, -2.0]), ("up", ">=", [0.5, 1.0, 1.5, 2.0])):
            fade = "up" if d_gap == "down" else "down"
            cfgs += _thresh_family(sym, "gap", op, vals, OutcomeSpec("open", "session", fade, 0.0, None, "close"), tag=f"gap {d_gap}: fade")
            cfgs += _thresh_family(sym, "gap", op, vals, OutcomeSpec("open", "session", fade, 0.5, 0.5, "touch"), tag=f"gap {d_gap}: fade 0.5% before 0.5% further")
            cfgs += _thresh_family(sym, "gap", op, vals[:3], OutcomeSpec("10:30", "session", fade, 0.0, None, "close"), tag=f"gap {d_gap}: fade after the first hour")
        P.update(hypothesis=f"After an overnight gap, {sym} tends to retrace during the session (mean reversion) rather than keep going (momentum).",
                 null="The session's direction after the open is unrelated to the gap.",
                 outcomes=[{"label": "the session closes on the fade side of its open (above it after a gap down, below it after a gap up)"},
                           {"label": "price trades 0.5% in the fade direction before 0.5% in the gap direction"}, {"label": "the same, observed from 10:30 ET"}], configs=cfgs)
        P["notes"] += ["Each row tests the FADE. A frequency reliably below the baseline is evidence for momentum; one reliably above is evidence for mean reversion; neither is the common result.",
                       "\"Gap\" is the open against the previous close. Four sizes are tested in each direction instead of one being assumed."]
        return _finish(P)

    # ---- consecutive red / down / green / up days (calls versus puts)
    m = re.search(r"consecutive\s+(red|green|down|up|bearish|bullish|losing|winning|lower|higher)|(\w+)\s+(red|green|down|up|losing|winning)\s+days in a row|streak", low)
    if m:
        w = (m.group(1) or m.group(3) or "red") if m.lastindex else "red"
        red = w in ("red", "bearish")
        down = w in ("down", "losing", "lower")
        feat = "red_streak" if red else "down_streak" if down else "green_streak" if w in ("green", "bullish") else "up_streak"
        after_down = feat in ("red_streak", "down_streak")
        cfgs = []
        for oc, tag in ((OutcomeSpec("open", "session", "up", 0.0, None, "close"), "next session closes above its open"),
                        (OutcomeSpec("close", "sessions:1", "up", 0.0, None, "close"), "next close is higher"),
                        (OutcomeSpec("close", "sessions:3", "up", 0.0, None, "close"), "close 3 sessions later is higher"),
                        (OutcomeSpec("close", "sessions:5", "up", 0.0, None, "close"), "close 5 sessions later is higher")):
            cfgs += _thresh_family(sym, feat, ">=", [2, 3, 4, 5], oc, tag=tag)
        P.update(hypothesis=f"After several consecutive {'red' if red else 'down' if down else 'green' if feat == 'green_streak' else 'up'} sessions, {sym}'s next move is not a coin flip: "
                            f"{'a rebound' if after_down else 'a pullback'} or a continuation is more likely than on other days.",
                 null="What follows a streak is no different from what follows any other session.",
                 outcomes=[{"label": "the following session(s) finish higher"}], configs=cfgs)
        P["notes"] += [f"A {'red session closes below its open' if red or feat == 'green_streak' else 'down day closes below the previous close'}; streaks of 2, 3, 4 and 5 or more are tested.",
                       "Rows test \"finishes higher\". A frequency below the baseline means the opposite (lower) happened more often."]
        if re.search(r"\bcalls?\b|\bputs?\b|options?", low):
            P["notes"].append("Calls versus puts is the second stage: once the underlying's behaviour is established, both are run on the same sessions with observed option prices. A pattern in the underlying does not by itself make either profitable.")
            P["options_followup"] = True
        return _finish(P)

    # ---- reversal / continuation after large moves
    if re.search(r"revers|bounc|rebound|recover|mean[- ]rever|snap ?back|continu|follow[- ]through|momentum|keep (?:falling|rising|going)", low) and \
            re.search(r"down|red|declin|sell-?off|drop|fall|loss|up day|green|rall|gain|surge", low):
        down = bool(re.search(r"down|red|declin|sell-?off|drop|fall|loss", low))
        cont = bool(re.search(r"continu|follow[- ]through|momentum|keep (?:falling|rising|going)", low)) and not re.search(r"revers|bounc|rebound|recover|mean[- ]rever|snap ?back", low)
        up_outcome = down != cont
        d = "up" if up_outcome else "down"
        vals = [-1, -1.5, -2, -2.5, -3, -4] if down else [1, 1.5, 2, 2.5, 3, 4]
        op = "<=" if down else ">="
        cfgs = []
        for oc, tag in ((OutcomeSpec("open", "session", d, 0.0, None, "close"), "next session, open to close"),
                        (OutcomeSpec("close", "sessions:1", d, 0.0, None, "close"), "close to next close"),
                        (OutcomeSpec("close", "sessions:3", d, 0.0, None, "close"), "close to close, 3 sessions"),
                        (OutcomeSpec("close", "sessions:5", d, 0.0, None, "close"), "close to close, 5 sessions"),
                        (OutcomeSpec("open", "session", d, 1.0, 1.0, "touch"), "1% before 1% the other way, next session")):
            cfgs += _thresh_family(sym, "ret_1", op, vals, oc, tag=tag)
        for band, vp in (("VIX below 20", Pred("vix_prev", "<", 20)), ("VIX 20 or more", Pred("vix_prev", ">=", 20))):
            cfgs += _thresh_family(sym, "ret_1", op, vals[:5:2], OutcomeSpec("close", "sessions:1", d, 0.0, None, "close"), extra=[vp], tag=f"close to next close, {band}")
        what = "reverse" if not cont else "continue"
        P.update(hypothesis=f"After a large {'down' if down else 'up'} day, {sym} tends to {what}: prices are {'higher' if up_outcome else 'lower'} afterwards more often than after other days.",
                 null=f"What follows a large {'down' if down else 'up'} day is no different from what follows any other day.",
                 outcomes=[{"label": f"the next session closes {'above' if up_outcome else 'below'} its open"}, {"label": f"the close 1, 3 and 5 sessions later is {'higher' if up_outcome else 'lower'} than the close of the {'down' if down else 'up'} day"},
                           {"label": f"1% {'up' if up_outcome else 'down'} before 1% the other way during the next session"}], configs=cfgs)
        P["notes"] += ["\"Large\" is not a number, so six thresholds from 1% to 4% are tested rather than one being assumed.",
                       "Two entries are tested because they answer different questions: from the next open (the overnight gap is not captured) and from the close of the large day itself (it is).",
                       "Large declines cluster in volatile periods, so rows are also split by the volatility index and every test resamples whole weeks."]
        P["controls"] = ["Volatility regime (volatility index, previous close): below 20 versus 20 or more."]
        P["options_followup"] = True
        return _finish(P)
    return None


def _scan(sym: str, oc: OutcomeSpec, cross: list[str], vol_first: bool = False) -> list[dict]:
    """A fixed, pre-declared scan of single conditions (one feature at a time, a few values each)."""
    fams = [("ret_1", "<=", [-1, -2, -3]), ("ret_1", ">=", [1, 2, 3]), ("gap", "<=", [-0.5, -1]), ("gap", ">=", [0.5, 1]), ("down_streak", ">=", [2, 3, 4]), ("up_streak", ">=", [2, 3, 4]),
            ("sma200_dist", "<", [0]), ("sma200_dist", ">=", [0]), ("sma20_dist", "<=", [-3, -5]), ("vix_prev", ">=", [20, 25, 30]), ("vix_prev", "<", [15]), ("rv_pctile", ">=", [80, 90]),
            ("range_1", ">=", [1.5, 2, 3]), ("vol_ratio", ">=", [1.5, 2]), ("dist_high20", ">=", [-0.5]), ("dist_high252", "<=", [-10, -20])]
    if vol_first:
        fams = [f for f in fams if f[0] in ("vix_prev", "rv_pctile", "range_1", "vol_ratio")] + [("gap", "abs>=", [0.5, 1.0]), ("ret_1", "abs>=", [1, 2, 3])] + [f for f in fams if f[0] in ("down_streak", "dist_high252", "sma200_dist")]
    out = []
    for feat, op, vals in fams:
        out += _thresh_family(sym, feat, op, vals, oc)
    for x in cross[:1]:
        out += _thresh_family(sym, f"{x}_ret_1", "<=", [-1, -2], oc)
        out += _thresh_family(sym, f"rs5_{x}", "<=", [-2], oc)
        out += _thresh_family(sym, f"rs5_{x}", ">=", [2], oc)
    for wd in range(5):
        out.append(_cfg(f"weekday | {oc.key()}", float(wd), [Pred("weekday", "==", wd)], oc, sym))
    return out


def validate_plan(plan: dict) -> list[str]:
    """Bounds and references of a plan (also applied to plans proposed by a language model)."""
    errs = []
    cfgs = plan.get("configs") or []
    if not cfgs:
        errs.append("the plan has no configurations")
    if len(cfgs) > min(int(plan.get("max_configs") or MAX_CONFIGS), 200):
        errs.append(f"the plan has {len(cfgs)} configurations; the bound is {plan.get('max_configs') or MAX_CONFIGS}")
    for c in cfgs:
        for p in c.get("preds", []):
            try:
                feature_def(p["feature"], plan.get("symbol", "SPY"))
            except KeyError:
                errs.append(f"unknown feature {p.get('feature')!r}")
            if p.get("op") not in ("<", "<=", ">", ">=", "==", "abs>=", "between"):
                errs.append(f"unknown operator {p.get('op')!r}")
        oc = c.get("outcome") or {}
        if oc.get("measure") not in ("touch", "close") or oc.get("direction") not in ("up", "down", "either"):
            errs.append("an outcome must measure 'touch' or 'close' in direction up, down or either")
    return sorted(set(errs))


def basis_of(c: dict) -> str:
    """daily: answerable from daily bars over the whole history. intraday: needs intraday bars."""
    oc = c["outcome"]
    if oc["entry"] not in ("open", "close") or str(oc.get("horizon", "")).startswith("time:"):
        return "intraday"
    return "intraday" if any(FEATURES.get(p["feature"]) is not None and FEATURES[p["feature"]].kind == "intraday" for p in c["preds"]) else "daily"


def describe_data(plan: dict, store: MarketStore | None = None) -> dict:
    """What the plan needs and what is stored - shown before anything runs."""
    store = store or MarketStore()
    ft = get_features(store, plan["symbol"], tuple(plan.get("cross") or ()))
    bases = sorted({basis_of(c) for c in plan["configs"]})
    intr = np.flatnonzero(ft.res < 86400)
    need = [{"what": "Daily bars", "symbol": plan["symbol"], "sessions": ft.n, "first": ft.dates[0], "last": ft.dates[-1], "ok": ft.n >= 500}]
    if "intraday" in bases:
        mix = {}
        for r in ft.res[intr]:
            k = f"{int(r) // 60}m"
            mix[k] = mix.get(k, 0) + 1
        need.append({"what": "Intraday bars", "symbol": plan["symbol"], "sessions": int(intr.size), "first": ft.dates[int(intr[0])] if intr.size else None, "last": ft.dates[int(intr[-1])] if intr.size else None,
                     "ok": intr.size >= 250, "mix": mix, "note": "Free sources keep only recent intraday history; older sessions cannot be evaluated for these rows and are reported as excluded." if intr.size < 1500 else ""})
    feats = {p["feature"] for c in plan["configs"] for p in c["preds"]}
    if "vix_prev" in feats:
        vs = ft.series.get(ft.vol_symbol) if ft.vol_symbol else None
        need.append({"what": "Volatility index, daily", "symbol": ft.vol_symbol or "VIX", "sessions": int(len(vs.days)) if vs is not None else 0, "ok": vs is not None})
    for x in plan.get("cross") or []:
        if any(x in f for f in feats):
            xs = ft.series.get(x)
            need.append({"what": "Daily bars (cross-symbol)", "symbol": x, "sessions": int(len(xs.days)) if xs is not None else 0, "ok": xs is not None})
    return {"required": need, "bases": bases, "sessions": ft.n, "intraday_sessions": int(intr.size)}


# ----------------------------------------------------------------------------- periods
def _split(days: list[str], holdout: dict | None) -> dict:
    """Train / validation from the sessions before the final test period (75% / 25% of them)."""
    if holdout:
        pre = [d for d in days if d < holdout["start"]]
        test = [holdout["start"], holdout["end"]]
    else:
        cut = days[int(len(days) * 0.8)]
        pre = [d for d in days if d < cut]
        test = [cut, days[-1]]
    if len(pre) < 60:
        raise ValueError(f"Only {len(pre)} sessions are available before the final test period: too few to train and validate on.")
    v0 = pre[int(len(pre) * 0.75)]
    return {"train": [pre[0], R._to_date(v0).fromordinal(R._to_date(v0).toordinal() - 1).isoformat()], "validate": [v0, pre[-1]], "test": test, "sessions": len(days),
            "train_sessions": int(len(pre) * 0.75), "validate_sessions": len(pre) - int(len(pre) * 0.75), "test_sessions": len([d for d in days if test[0] <= d <= test[1]])}


def _block(ft: SessionFeatures, ev: dict, pm: np.ndarray, direction: str, reps: int = PLAN_REPS, robust: bool = True) -> dict:
    """Statistics of one configuration inside one period (the shared evidence function)."""
    return EV.of(ft, ev, pm, direction, reps=reps, robust=robust, mean_diff=True)


def run_plan(plan: dict, store: MarketStore | None = None, scope: str | None = None, progress=None, should_cancel=None, run_id: str | None = None) -> dict:
    """Evaluate every configuration of a plan. `scope` is the research session the ledger
    tracks trials and the final test period under; without one the split is local and the
    result says that its final test period is not protected across runs."""
    errs = validate_plan(plan)
    if errs:
        raise ValueError("; ".join(errs))
    store = store or MarketStore()
    sym = plan["symbol"]
    ft = get_features(store, sym, tuple(plan.get("cross") or ()))
    cfgs = plan["configs"]
    bases = sorted({basis_of(c) for c in cfgs})
    periods, holdouts, pmask = {}, {}, {}
    for b in bases:
        days = ft.dates[1:] if b == "daily" else [ft.dates[i] for i in np.flatnonzero(ft.res < 86400) if i >= 1]
        key = sym if b == "daily" else f"{sym}@intraday"
        if len(days) < 250:
            periods[b] = None
            continue
        h = LG.reserve(scope, key, days) if scope else None
        try:
            periods[b] = _split(days, h)
        except ValueError:
            periods[b] = None
            continue
        holdouts[b] = h
        pmask[b] = {k: ft.window(*periods[b][k]) for k in ("train", "validate", "test")}
    rows = []
    evs: dict[str, dict] = {}
    for i, c in enumerate(cfgs):
        if should_cancel and should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()
        if progress and i % 4 == 0:
            progress(0.85 * i / len(cfgs), f"configuration {i + 1} of {len(cfgs)}: {c['label'][:70]}")
        b = basis_of(c)
        preds = [Pred.load(p) for p in c["preds"]]
        oc = OutcomeSpec.load(c["outcome"])
        row = {"key": c["key"], "family": c["family"], "x": c["x"], "label": c["label"], "tag": c.get("tag", ""), "condition": cond_text(preds, sym), "outcome_text": oc.text(), "outcome": oc.dump(), "preds": c["preds"],
               "basis": b, "status": "ok", "study": to_study(sym, preds, oc, label=c["label"]) if oc.direction != "either" else None}
        if periods.get(b) is None:
            row.update(status="no_data", error="fewer than 250 sessions with the bars this configuration needs are stored" if b == "intraday" else "too little history")
            rows.append(row)
            continue
        missing = [p.feature for p in preds if any(x in p.feature for x in ft.missing_cross)]
        if missing:
            row.update(status="no_data", error=f"no price history is stored for {', '.join(ft.missing_cross)}")
            rows.append(row)
            continue
        try:
            ev = ft.evaluate(preds, oc)
        except (ValueError, KeyError) as e:
            row.update(status="error", error=str(e)[:200])
            rows.append(row)
            continue
        evs[c["key"]] = ev
        pm = pmask[b]
        row["train"] = _block(ft, ev, pm["train"], oc.direction)
        row["validate"] = _block(ft, ev, pm["validate"], oc.direction, robust=False)
        row["excluded_sessions"] = int((~ev["evaluable"] & (pm["train"] | pm["validate"] | pm["test"])).sum())
        row["exclusions"] = {k: int(v) for k, v in list(ev["outcome"].reasons.items())[:3]} | ({"a condition could not be evaluated (missing history)": ev["not_evaluable"]} if ev["not_evaluable"] else {})
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    # ---- multiple hypotheses: every configuration tried in this research session counts, not only this plan's
    plan_keys = {r["key"] for r in rows}
    if scope:
        seen = {t["key"] for t in LG.list_trials(scope, limit=20000)}
        session_total = len(seen | plan_keys)
    else:
        session_total = len(plan_keys)
    primary = plan.get("primary_metric", "frequency")
    pv = [(r["train"]["p_value"] if primary == "frequency" else r["train"]["end_return_p_value"]) for r in ok]
    holm = R.adjust(pv, "holm", m=session_total)
    bh = R.adjust(pv, "bh", m=session_total)
    for r, a, q in zip(ok, holm, bh):
        r["p_primary"], r["p_adjusted"], r["q_value"] = (r["train"]["p_value"] if primary == "frequency" else r["train"]["end_return_p_value"]), a, q
    # ---- stability: neighbouring values of the same parameter must point the same way
    fam: dict[str, list[dict]] = {}
    for r in ok:
        fam.setdefault(r["family"], []).append(r)
    for rs in fam.values():
        rs.sort(key=lambda r: r["x"])
        for i, r in enumerate(rs):
            same = tot = 0
            for j in (i - 1, i + 1):
                if 0 <= j < len(rs):
                    nb = rs[j]
                    if nb["train"]["lift"] is not None and r["train"]["lift"] is not None and nb["train"]["n"] >= 10:
                        tot += 1
                        same += (nb["train"]["lift"] > 0) == (r["train"]["lift"] > 0)
            r["stability"], r["neighbours"] = ((same / tot) if tot else None), tot
    min_n = int(plan.get("min_sample") or MIN_SAMPLE)
    for r in ok:
        tr, va = r["train"], r["validate"]
        why = []
        if tr["n"] < min_n:
            why.append(f"only {tr['n']} observations in the training period (the plan requires {min_n})")
        elif tr["clusters"] < R.MIN_CLUSTERS:
            why.append(f"the {tr['n']} observations fall in only {tr['clusters']} separate weeks")
        if r["p_adjusted"] is None:
            if tr["n"] >= min_n:
                why.append("no dependence-robust test is possible on this sample")
        elif r["p_adjusted"] >= 0.10:
            why.append(f"not distinguishable from the baseline once the {session_total} configurations tried are allowed for")
        if r["stability"] is not None and r["stability"] < 0.5:
            why.append("neighbouring parameter values point the other way")
        if va["n"] < MIN_VALIDATE:
            why.append(f"only {va['n']} observations in the validation period")
        elif va["lift"] is None or tr["lift"] is None or (va["lift"] > 0) != (tr["lift"] > 0):
            why.append("the validation period points the other way")
        oc = r["outcome"]
        r["directional"] = None
        if oc["measure"] == "touch" and oc["direction"] != "either":
            # a level is touched more often simply because ranges are wider; a directional claim also needs
            # the sessions to END further that way than other sessions do
            d_ = tr.get("directional_diff_pct")
            r["directional"] = bool(d_ is not None and d_ * (1 if (tr["lift"] or 0) > 0 else -1) > 0 and tr["end_return_p_value"] is not None and tr["end_return_p_value"] < 0.10)
            if not r["directional"]:
                why.append("the sessions do not end further in that direction than other sessions: the different touch rate reflects wider ranges, not direction")
        r["candidate"] = not why
        r["rejected_because"] = why
        r["effect"] = "more often" if (tr["lift"] or 0) > 0 else "less often"
        r["status_label"] = "exploratory"
    # ---- does picking the best configuration on part of the history pay off on the rest? (purged, embargoed)
    sel = _selection_cv(ft, ok, evs, pmask)
    # ---- final test period: once, for the candidates, only while it is untouched
    cands = sorted([r for r in ok if r["candidate"]], key=lambda r: (r["p_adjusted"], -(r["stability"] or 0)))[:MAX_CANDIDATES]
    test_note: dict[str, str] = {}
    used = False
    for b in bases:
        cb = [r for r in cands if r["basis"] == b]
        h = holdouts.get(b)
        if not cb:
            continue
        if scope and h is not None and not h.get("usable"):
            already = {k for u in (h.get("uses") or []) for k in u.get("keys", [])}
            for r in cb:
                if h["status"] == "used" and r["key"] in already:
                    continue  # scored before on this very period: shown again below, nothing new is revealed
                r["status_label"], r["test_note"] = "candidate - no untouched data", LG.holdout_note(h)
            cb = [r for r in cb if h["status"] == "used" and r["key"] in already]
            test_note[b] = LG.holdout_note(h)
            if not cb:
                continue
        pm = pmask[b]["test"]
        for r in cb:
            te = _block(ft, evs[r["key"]], pm, r["outcome"]["direction"])
            r["test"] = te
            same = te["lift"] is not None and r["train"]["lift"] is not None and (te["lift"] > 0) == (r["train"]["lift"] > 0)
            p1 = (te["p_value"] / 2) if (te["p_value"] is not None and same) else None
            adj = None if p1 is None else min(1.0, p1 * len(cb))
            r["test_p_adjusted"] = adj
            if te["n"] < MIN_TEST:
                r["status_label"], r["test_note"] = "candidate - untested", f"only {te['n']} observations in the final test period: too few to confirm or reject"
            elif adj is not None and adj < 0.05:
                r["status_label"], r["test_note"] = "confirmed on untouched data", f"same direction in the final test period, p = {adj:.4f} (one-sided, calendar-cluster test, adjusted for {len(cb)} candidate{'s' if len(cb) != 1 else ''})"
            else:
                r["status_label"], r["test_note"] = "not confirmed", "the final test period does not confirm it" + (f" (p = {adj:.4f})" if adj is not None else " (opposite direction)" if not same else " (no test possible)")
        used = True
        if scope and h is not None:
            holdouts[b] = LG.use(h["id"], [r["key"] for r in cb], run_id, plan["question"][:120])
        elif not scope:
            for r in cb:
                if r["status_label"] == "confirmed on untouched data":
                    r["status_label"] = "confirmed on the final period (not tracked)"
    for r in ok:
        hs = "confirmed" if r["status_label"].startswith("confirmed") else "untested" if r["status_label"] in ("candidate - untested", "candidate - no untouched data") else "unconfirmed" if r["status_label"] == "not confirmed" else None
        r["grade"] = R.grade(r["train"]["n"], r["train"]["clusters"], r["p_adjusted"], holdout=hs, stable=None if r["stability"] is None else r["stability"] >= 0.5,
                             contaminated=bool(scope and holdouts.get(r["basis"]) and holdouts[r["basis"]]["status"] in ("contaminated", "exposed")))
        if not r["candidate"] and r["grade"]["level"] in ("weak", "moderate", "strong"):
            r["grade"] = {"level": "none", "label": "Did not qualify", "reasons": r["rejected_because"][:2]}
    # ---- option data that exists for a second stage
    cover = _option_cover(store, ft, sym, ok, evs)
    # ---- ledger
    contaminated = None
    if scope:
        for b in bases:
            key = sym if b == "daily" else f"{sym}@intraday"
            tr = [{"kind": "plan", "key": r["key"], "label": r["label"], "run_id": run_id, "start": (periods[b] or {}).get("train", [None])[0], "end": (periods[b] or {}).get("validate", [None, None])[1],
                   "metric": "lift", "value": (r.get("train") or {}).get("lift"), "n": (r.get("train") or {}).get("n"), "p_value": r.get("p_primary")} for r in rows if r["basis"] == b and r["status"] == "ok"]
            out = LG.record(scope, key, tr, family=plan["question"][:120])
            contaminated = contaminated or out.get("contaminated")
        for b in list(holdouts):
            if holdouts[b] is not None:
                holdouts[b] = LG._pub(LG.get(holdouts[b]["id"]))
    confirmed = [r for r in cands if r["status_label"].startswith("confirmed")]
    n_enough = sum(1 for r in ok if r["train"]["n"] >= min_n)
    n_sig = sum(1 for r in ok if r["train"]["n"] >= min_n and r["p_adjusted"] is not None and r["p_adjusted"] < 0.10)
    n_stable = sum(1 for r in ok if r["train"]["n"] >= min_n and r["p_adjusted"] is not None and r["p_adjusted"] < 0.10 and not (r["stability"] is not None and r["stability"] < 0.5))
    funnel = [{"step": "Configurations in the plan", "n": len(rows)}, {"step": "Could be evaluated", "n": len(ok)}, {"step": f"At least {min_n} training observations", "n": n_enough},
              {"step": "Distinguishable from the baseline after correction", "n": n_sig}, {"step": "Agree with neighbouring values", "n": n_stable},
              {"step": "Agree with the validation period (candidates)", "n": len([r for r in ok if r["candidate"]])}, {"step": "Confirmed on the final test period", "n": len(confirmed)}]
    notes = [f"All {len(rows)} configurations are listed, including {len(rows) - len(cands)} that did not qualify.",
             f"Significance is corrected (Holm) for {session_total} configuration{'s' if session_total != 1 else ''}" + (" tried in this research session" if scope else " in this plan") + ", using tests that resample whole calendar weeks.",
             "These are frequencies and returns of the UNDERLYING. They say nothing yet about option premiums, spreads or fills: a real pattern in the underlying can still lose money in options."]
    for b in bases:
        if periods.get(b):
            p = periods[b]
            notes.append(f"{'Daily' if b == 'daily' else 'Intraday'} rows - training {p['train'][0]} to {p['train'][1]}, validation {p['validate'][0]} to {p['validate'][1]}, final test {p['test'][0]} to {p['test'][1]}.")
        else:
            notes.append(f"{'Intraday' if b == 'intraday' else 'Daily'} rows could not be evaluated: too few sessions with the required bars are stored.")
    if used:
        notes.append("The final test period has now been scored for the candidates above. Testing further configurations on data that includes it will mark it as contaminated.")
    elif not cands:
        notes.append("No configuration qualified, so the final test period was not looked at.")
    for b, t in test_note.items():
        notes.append(t)
    if not scope:
        notes.append("This run is not attached to a research session, so its final test period is not protected against repeated use.")
    if plan.get("dropped_for_bound"):
        notes.append(f"{plan['dropped_for_bound']} further configurations were left out to stay within the plan's bound of {plan['max_configs']}.")
    if progress:
        progress(1.0, "done")
    pub_plan = {k: v for k, v in plan.items() if k != "configs"} | {"configs": len(cfgs)}
    return {"kind": "hypothesis", "symbol": sym, "question": plan["question"], "hypothesis": plan.get("hypothesis"), "plan": pub_plan, "periods": periods, "holdouts": holdouts, "configs": rows, "tested": len(rows),
            "evaluable": len(ok), "candidates": [r["key"] for r in cands], "confirmed": [r["key"] for r in confirmed], "test_period_used": used, "funnel": funnel, "selection_cv": sel,
            "trials": {"plan": len(rows), "session": session_total, "correction": "Holm"}, "contaminated": contaminated, "options_cover": cover, "notes": notes, "data": describe_data(plan, store),
            "synthetic": store.is_synthetic(sym), "scope": scope}


def _selection_cv(ft: SessionFeatures, ok: list[dict], evs: dict, pmask: dict) -> dict | None:
    """Purged, embargoed cross-validation of the act of choosing the best configuration."""
    out = {}
    for b in sorted({r["basis"] for r in ok}):
        rs = [r for r in ok if r["basis"] == b and r["train"]["n"] >= 10]
        if len(rs) < 2:
            continue
        pre = pmask[b]["train"] | pmask[b]["validate"]
        idx = np.flatnonzero(pre)
        M = np.full((idx.size, len(rs)), np.nan)
        span = np.zeros(idx.size, np.int64)
        for j, r in enumerate(rs):
            ev = evs[r["key"]]
            o = ev["outcome"]
            base = ev["evaluable"] & pre & (o.code != A)
            rate = float((o.code[base] == T).mean()) if base.any() else 0.0
            q = (ev["qualifying"] & (o.code != A))[idx]
            sgn = 1.0 if (r["train"]["lift"] or 0) >= 0 else -1.0  # a configuration is scored in the direction it was selected for
            M[q, j] = sgn * ((o.code[idx][q] == T).astype(float) - rate)
            span = np.maximum(span, o.span[idx])
        res = R.selection_cv(M, idx, span, k=5, embargo=int((span - idx).max()) if idx.size else 0)
        if res.get("folds"):
            res["configs"] = len(rs)
            res["note"] = ("In each fold the configuration with the largest effect on the training sessions was scored on the held-out block (observations overlapping the block were purged). "
                           "If the chosen one usually ranks in the lower half out of sample, selection is mostly fitting noise.")
            out[b] = res
    return out or None


def _option_cover(store: MarketStore, ft: SessionFeatures, sym: str, ok: list[dict], evs: dict) -> dict:
    """How many qualifying sessions of each configuration have observed option prices stored."""
    try:
        q_days = {d.isoformat() for d in store.option_days(sym, "quote")}
    except Exception:  # noqa: BLE001
        q_days = set()
    try:
        idx = store.option_bar_index(sym)
        t_days = {str(d) for d in idx["date"].unique().to_list()} if not idx.is_empty() else set()
    except Exception:  # noqa: BLE001
        t_days = set()
    dates = np.array(ft.dates)
    per = {}
    for r in ok:
        qd = set(dates[evs[r["key"]]["qualifying"]].tolist())
        eod = r["outcome"]["entry"] == "close"
        per[r["key"]] = {"qualifying": len(qd), "with_quotes": len(qd & q_days) if eod else 0, "with_trades": len(qd & t_days), "quotes_usable": eod}
        r["options_cover"] = per[r["key"]]
    return {"quote_sessions": len(q_days), "trade_sessions": len(t_days), "quote_range": [min(q_days), max(q_days)] if q_days else None, "trade_range": [min(t_days), max(t_days)] if t_days else None,
            "note": "End-of-day option quotes can price positions entered at the close; positions entered at the open or intraday need intraday option prices (trade prints), which free sources only keep for recent sessions."}


def save_plan_result(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "hypothesis.json").write_text(dumps(res))
    spans = [p for p in res["periods"].values() if p]
    return {"hypothesis": True, "tested": res["tested"], "evaluable": res["evaluable"], "candidates": len(res["candidates"]), "confirmed": len(res["confirmed"]), "symbols": [res["symbol"]],
            "start": min(p["train"][0] for p in spans) if spans else None, "end": max(p["test"][1] for p in spans) if spans else None, "synthetic": res.get("synthetic"), "session_trials": res["trials"]["session"]}


def observations(res: dict, key: str, store: MarketStore | None = None, period: str | None = None) -> dict:
    """The sessions behind one configuration of a finished plan (recomputed from the stored
    plan so every number can be traced to its observations)."""
    row = next((r for r in res["configs"] if r["key"] == key), None)
    if row is None:
        raise LookupError("configuration not found in this plan")
    ft = get_features(store or MarketStore(), res["symbol"], tuple((res.get("plan") or {}).get("cross") or ()))
    preds = [Pred.load(p) for p in row["preds"]]
    oc = OutcomeSpec.load(row["outcome"])
    ev = ft.evaluate(preds, oc)
    o = ev["outcome"]
    per = (res["periods"] or {}).get(row["basis"]) or {}
    q = ev["qualifying"].copy()
    if period and per.get(period):
        q &= ft.window(*per[period])
    elif per and not row.get("test"):
        q &= ~ft.window(*per["test"])  # the final test period of a configuration that was not scored on it stays unseen
    cols = {p.name: p.values(ft, oc.entry) for p in preds}

    def which(d):
        return next((k for k in ("train", "validate", "test") if per.get(k) and per[k][0] <= d <= per[k][1]), None)
    rows = [{"date": ft.dates[i], "outcome": CODE_NAMES[o.code[i]], "entry_price": round(float(o.entry_px[i]), 4), "end_return_pct": round(float(o.end_ret[i]), 4), "mfe_pct": round(float(o.mfe[i]), 4),
             "mae_pct": round(float(o.mae[i]), 4), "period": which(ft.dates[i]), **{k: round(float(v[i]), 4) for k, v in cols.items()}} for i in np.flatnonzero(q)]
    return {"key": key, "label": row["label"], "outcome_text": row["outcome_text"], "columns": ["date", "period", "outcome", "entry_price", "end_return_pct", "mfe_pct", "mae_pct"] + list(cols), "rows": rows,
            "hidden_test_period": bool(per and not row.get("test") and not period)}


# ----------------------------------------------------------------------------- stage two: option structures
def structure_variants(symbol: str, preds: list[dict], outcome: dict, has_quotes: bool, has_trades: bool) -> tuple[list[dict], list[str]]:
    """Option positions that express one hypothesis, as ordinary strategy drafts. Returns
    (variants, notes). They run through the same pipeline as any strategy: observed premiums
    only, coverage reported per signal, nothing modelled unless the user opts in."""
    from strata.ai.draft import Draft, DraftCondition, DraftStrike
    sym = symbol.upper()
    oc = OutcomeSpec.load(outcome)
    ps = [Pred.load(p) for p in preds]
    conds = [DraftCondition(expr=p.expr(sym, oc.entry), text=p.text(sym)) for p in ps]
    eod = oc.entry == "close"
    hold = int(oc.horizon.split(":")[1]) if oc.horizon.startswith("sessions:") else None
    notes = []
    if eod:
        dtes = [7, 14, 30]
        notes.append("Entries are at the close, so end-of-day bid/ask quotes can price them where they are stored. Expirations of about 7, 14 and 30 days are compared (the nearest listed one is used).")
    else:
        dtes = [0, 1, 7]
        notes.append("Entries are intraday, so only intraday option prices (trade prints) can price them. Free sources keep those for recent sessions only, so most qualifying sessions will be reported as not testable.")

    def base(title: str, structure: str, dte: int, strike: DraftStrike | None = None, **kw) -> Draft:
        d = Draft(kind="options", title=f"{sym} {title}", symbol=sym, symbol_stated=True, entry_time=oc.entry, conditions=list(conds), structure=structure, dte=dte, strike=strike or DraftStrike(method="atm"), **kw)
        if hold:
            d.hold_sessions = hold
        elif oc.horizon.startswith("time:"):
            d.exit_time = oc.horizon.split(":", 1)[1]
        else:
            d.exit_time = "15:55"
        return d

    d0 = dtes[0]
    either = oc.direction == "either"
    up = oc.direction == "up"
    out: list[dict] = []
    if either:
        out += [{"label": f"long straddle, {d0}DTE (profits from a large move)", "draft": base(f"long straddle {d0}DTE", "long_straddle", d0)},
                {"label": f"iron condor, {d0}DTE (profits when the move stays small)", "draft": base(f"iron condor {d0}DTE", "iron_condor", d0, width=5)},
                {"label": f"long straddle, {dtes[1]}DTE", "draft": base(f"long straddle {dtes[1]}DTE", "long_straddle", dtes[1])},
                {"label": f"long strangle, {d0}DTE", "draft": base(f"long strangle {d0}DTE", "long_strangle", d0)}]
        notes.append("The hypothesis is about the SIZE of the move, so positions that depend on movement (straddle, strangle) are compared with one that depends on its absence (iron condor).")
    else:
        main, opp = ("long_call", "long_put") if up else ("long_put", "long_call")
        debit, credit = ("bull_call_spread", "bull_put_spread") if up else ("bear_put_spread", "bear_call_spread")
        nm = {"long_call": "call", "long_put": "put"}
        out += [{"label": f"at-the-money {nm[main]}, {d0}DTE", "draft": base(f"ATM {nm[main]} {d0}DTE", main, d0)},
                {"label": f"at-the-money {nm[opp]}, {d0}DTE (the opposite side, as a control)", "draft": base(f"ATM {nm[opp]} {d0}DTE", opp, d0)},
                {"label": f"30-delta {nm[main]}, {d0}DTE (out of the money)", "draft": base(f"30-delta {nm[main]} {d0}DTE", main, d0, DraftStrike(method="delta", value=0.30))}]
        for x in dtes[1:]:
            out.append({"label": f"at-the-money {nm[main]}, {x}DTE", "draft": base(f"ATM {nm[main]} {x}DTE", main, x)})
        out += [{"label": f"debit spread ({debit.replace('_', ' ')}), {d0}DTE", "draft": base(f"{debit.replace('_', ' ')} {d0}DTE", debit, d0, width=5)},
                {"label": f"credit spread ({credit.replace('_', ' ')}), {d0}DTE", "draft": base(f"{credit.replace('_', ' ')} {d0}DTE", credit, d0, width=5)},
                {"label": f"long straddle, {d0}DTE (direction-neutral)", "draft": base(f"long straddle {d0}DTE", "long_straddle", d0)},
                {"label": f"iron condor, {d0}DTE (short volatility)", "draft": base(f"iron condor {d0}DTE", "iron_condor", d0, width=5)}]
        prem = base(f"ATM {nm[main]} {d0}DTE, premium exits", main, d0)
        prem.premium_target_pct, prem.premium_stop_pct = 50.0, 50.0
        out.append({"label": f"at-the-money {nm[main]}, {d0}DTE, exit at +50% / -50% of premium", "draft": prem})
        if oc.target or oc.stop:
            und = base(f"ATM {nm[main]} {d0}DTE, underlying exits", main, d0)
            und.underlying_target_pct, und.underlying_stop_pct = (oc.target or None), (oc.stop or None)
            out.append({"label": f"at-the-money {nm[main]}, {d0}DTE, exit on the underlying's {('+' if up else '-') + format(oc.target, 'g') + '% target' if oc.target else ''}{' / ' if oc.target and oc.stop else ''}{format(oc.stop, 'g') + '% stop' if oc.stop else ''}", "draft": und})
        notes.append("The opposite side is included on purpose: if calls and puts both lose on the same sessions, the cost of the options outweighs the pattern in the underlying.")
    if eod and not has_quotes:
        notes.append("No end-of-day option quotes are stored for this symbol, so these positions cannot be priced until some are downloaded or imported.")
    if not eod and not has_trades:
        notes.append("No intraday option prices are stored for this symbol yet.")
    return out[:12], notes
