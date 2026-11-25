"""Exploratory strategy discovery with guard rails against fooling yourself.

A hypothesis ("mean reversion after unusually large declines") is turned into a grid of
market studies over reasonable parameters. Every configuration is evaluated and recorded -
including the ones that show nothing - and the output is NOT a ranking by performance:

  * history is split by date into EXPLORE (first 60%), VALIDATE (next 20%) and a final
    TEST period (last 20%) that no selection step ever looks at
  * significance is judged against the baseline on the explore period, with p-values
    adjusted for the number of configurations tried (Benjamini-Hochberg and Bonferroni)
  * a configuration only becomes a candidate if it has enough observations, survives the
    adjustment, agrees with its neighbouring parameter values (stability) and points the
    same way on the validation period
  * only candidates are then scored on the final test period, once. What passes there is
    reported as "confirmed on untouched data"; everything else stays "exploratory"
"""
from __future__ import annotations

import itertools
import math

from strata.data.store import MarketStore
from strata.research import stats as ST
from strata.research.study import StudyConfig, _Host, run_study

MIN_EXPLORE, MIN_VALIDATE, MIN_TEST = 30, 8, 8
MAX_CANDIDATES = 5


def run_discovery(spec: dict, store: MarketStore | None = None, progress=None, should_cancel=None) -> dict:
    """spec: {symbol, question, hypothesis, direction, dims: [{name, label, values, labels}],
              configs: [{key, params: {dim: value}, label, study: StudyConfig dict}]}"""
    store = store or MarketStore()
    configs = spec["configs"]
    if not configs:
        raise ValueError("no configurations to explore")
    first = StudyConfig.model_validate(configs[0]["study"])
    host = _Host(store, first)
    ser = host.series.get(first.symbol)
    if ser is None or ser.empty():
        raise LookupError(f"No historical price data is stored for {first.symbol}.")
    from strata.engine.market import from_day_num
    days = [from_day_num(x).isoformat() for x in ser.days]
    a = spec.get("start") or days[0]
    b = spec.get("end") or days[-1]
    days = [d for d in days if a <= d <= b]
    if len(days) < 250:
        raise ValueError(f"Only {len(days)} sessions of history are stored: too little to split into explore, validation and test periods.")
    v_start, t_start = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
    splits = {"explore": [days[0], ST._day_before(v_start)], "validate": [v_start, ST._day_before(t_start)], "test": [t_start, days[-1]], "sessions": len(days)}
    rows = []
    sealed: dict[str, dict] = {}
    for i, c in enumerate(configs):
        if should_cancel and should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()
        if progress:
            progress(0.9 * i / len(configs), f"configuration {i + 1} of {len(configs)}: {c['label']}")
        try:
            res = run_study({**c["study"], "start": days[0], "end": days[-1]}, store, host=host)
        except Exception as e:  # noqa: BLE001  (a configuration that cannot be evaluated is recorded, not dropped)
            rows.append({"key": c["key"], "label": c["label"], "params": c["params"], "status": "error", "error": str(e)[:200]})
            continue
        ex = ST.study_block(res, *splits["explore"])
        va = ST.study_block(res, *splits["validate"])
        sealed[c["key"]] = ST.study_block(res, *splits["test"])  # computed, but never used to choose anything
        excl = sum(x["count"] for x in res.get("excluded", []))
        rows.append({"key": c["key"], "label": c["label"], "params": c["params"], "status": "ok", "explore": ex, "validate": va,
                     "excluded_sessions": excl, "resolution_mix": res["stats"].get("resolution_mix"), "study": c["study"]})
    ok = [r for r in rows if r["status"] == "ok"]
    qs = ST.bh_adjust([r["explore"]["p_value"] for r in ok])
    for r, q in zip(ok, qs):
        r["q_value"] = q
        r["bonferroni_p"] = ST.bonferroni(r["explore"]["p_value"], len(ok))
    by_key = {r["key"]: r for r in ok}
    dims = spec["dims"]
    for r in ok:  # stability: do the neighbouring parameter values point the same way?
        same = tot = 0
        for d in dims:
            vals = d["values"]
            i = vals.index(r["params"][d["name"]])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    nb = by_key.get(_key({**r["params"], d["name"]: vals[j]}, dims))
                    if nb and nb["explore"]["lift"] is not None and r["explore"]["lift"] is not None and nb["explore"]["n"] >= 10:
                        tot += 1
                        same += (nb["explore"]["lift"] > 0) == (r["explore"]["lift"] > 0)
        r["stability"] = (same / tot) if tot else None
        r["neighbours"] = tot
    for r in ok:
        ex, va = r["explore"], r["validate"]
        reasons = []
        if ex["n"] < MIN_EXPLORE:
            reasons.append(f"only {ex['n']} observations in the explore period (need {MIN_EXPLORE})")
        if r["q_value"] is None or r["q_value"] >= 0.10:
            reasons.append("not distinguishable from the baseline after adjusting for the number of configurations tried" if r["q_value"] is not None else "no test possible")
        if r["stability"] is not None and r["stability"] < 0.5:
            reasons.append("neighbouring parameter values point the other way (unstable)")
        if va["n"] < MIN_VALIDATE:
            reasons.append(f"only {va['n']} observations in the validation period")
        elif va["lift"] is None or ex["lift"] is None or (va["lift"] > 0) != (ex["lift"] > 0):
            reasons.append("the validation period points the other way")
        # a target is touched more often simply because ranges are wider after large moves; a directional
        # hypothesis also needs the sessions to END further in that direction than other sessions do
        sgn = 1 if spec.get("direction", "up") == "up" else -1
        de, dv = ex.get("end_return_diff_pct"), va.get("end_return_diff_pct")
        r["directional"] = bool(de is not None and de * sgn > 0 and ex.get("end_return_p_value") is not None and ex["end_return_p_value"] < 0.10 and (dv is None or dv * sgn > 0))
        if not r["directional"]:
            reasons.append("the sessions do not end further in that direction than other sessions: the higher touch rate reflects wider ranges, not direction")
        r["candidate"] = not reasons
        r["rejected_because"] = reasons
    cands = sorted([r for r in ok if r["candidate"]], key=lambda r: (r["q_value"], -(r["stability"] or 0)))[:MAX_CANDIDATES]
    for r in ok:
        r["status_label"] = "exploratory"
    for r in cands:  # the final test period is used once, for the candidates only
        te = sealed[r["key"]]
        r["test"] = te
        p1 = te["p_value"] / 2 if te["p_value"] is not None and te["lift"] is not None and r["explore"]["lift"] is not None and (te["lift"] > 0) == (r["explore"]["lift"] > 0) else None
        adj = ST.bonferroni(p1, len(cands))
        r["test_p_adjusted"] = adj
        if te["n"] < MIN_TEST:
            r["status_label"], r["test_note"] = "candidate - untested", f"only {te['n']} observations in the final test period: too few to confirm or reject"
        elif adj is not None and adj < 0.05:
            r["status_label"], r["test_note"] = "confirmed on untouched data", f"same direction on the final test period, p = {adj:.4f} (one-sided, adjusted for {len(cands)} candidates)"
        else:
            r["status_label"], r["test_note"] = "not confirmed", "the final test period does not confirm it" + (f" (p = {adj:.4f})" if adj is not None else " (opposite direction)")
    confirmed = [r for r in cands if r["status_label"] == "confirmed on untouched data"]
    notes = [
        f"{len(rows)} configurations were evaluated and all of them are listed, including {len(rows) - len(cands)} that did not qualify.",
        f"Explore {splits['explore'][0]} to {splits['explore'][1]}, validation {splits['validate'][0]} to {splits['validate'][1]}, final test {splits['test'][0]} to {splits['test'][1]}.",
        "Configurations are not ranked by performance. A candidate needs enough observations, significance after multiple-testing adjustment, agreement with neighbouring parameters and with the validation period.",
        "These are probabilities of the UNDERLYING's behaviour, not option returns; transaction costs and option pricing are not part of this stage.",
    ]
    if cands:
        notes.append(f"The final test period has now been used once for this hypothesis ({len(cands)} candidate{'s' if len(cands) != 1 else ''}). Tuning further against it would turn it into training data.")
    else:
        notes.append("No configuration qualified as a candidate, so the final test period was not used and remains untouched.")
    low_n = sum(1 for r in ok if r["explore"]["n"] < MIN_EXPLORE)
    if low_n:
        notes.append(f"{low_n} configurations have fewer than {MIN_EXPLORE} observations in the explore period - typically intraday entry times, for which only recent history is stored.")
    if progress:
        progress(1.0, "done")
    return {"kind": "discovery", "symbol": first.symbol, "question": spec.get("question", ""), "hypothesis": spec.get("hypothesis", ""), "direction": spec.get("direction", "up"), "dims": dims, "splits": splits,
            "configs": rows, "tested": len(rows), "evaluable": len(ok), "candidates": [r["key"] for r in cands], "confirmed": [r["key"] for r in confirmed], "test_period_used": bool(cands), "notes": notes,
            "synthetic": store.is_synthetic(first.symbol)}


def _key(params: dict, dims: list[dict]) -> str:
    return "|".join(f"{d['name']}={params[d['name']]}" for d in dims)


def build_grid(symbol: str, dims: list[dict], make_study) -> list[dict]:
    """Cartesian product of the dimensions -> configurations. `make_study(params)` returns
    (StudyConfig dict, label)."""
    out = []
    for combo in itertools.product(*[d["values"] for d in dims]):
        params = {d["name"]: v for d, v in zip(dims, combo)}
        study, label = make_study(params)
        out.append({"key": _key(params, dims), "params": params, "label": label, "study": study})
    return out


def save_discovery(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "discovery.json").write_text(dumps(res))
    return {"discovery": True, "tested": res["tested"], "evaluable": res["evaluable"], "candidates": len(res["candidates"]), "confirmed": len(res["confirmed"]), "symbols": [res["symbol"]],
            "start": res["splits"]["explore"][0], "end": res["splits"]["test"][1], "synthetic": res.get("synthetic")}
