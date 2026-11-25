"""The adversarial reviewer: reasons a finished result could be misleading.

`challenge` does the opposite of an optimiser. Given a finished study or backtest it looks
for the ways the result could be wrong, and where a cheap experiment can settle the
question it runs the experiment instead of offering an opinion:

  look-ahead        prices after the decision instant are scrambled and the entry rule
                    re-evaluated; any change in its answer is a look-ahead leak
  benchmark         the same position opened on every session, without the entry rule
  execution         the execution stress ladder (research/execution.py)
  parameters        each numeric choice nudged one step either way
  dependence        intervals recomputed with whole calendar weeks resampled

Every finding carries its evidence: numbers computed from this run's configuration, its
data diagnostics or the experiments above. A topic that cannot be examined with the
stored data is reported as "could not be checked", never as a pass.

Severity:  high = the conclusion probably does not hold;  medium = a real weakness that
limits what can be claimed;  low = worth knowing;  ok = examined, nothing found;
na = does not apply / could not be checked.
"""
from __future__ import annotations

import copy
import math
from datetime import date, timedelta

import numpy as np

from strata.research import reasoning as RS
from strata.research import robust as R

TOPICS = [("lookahead", "Look-ahead bias"), ("selection", "Selection bias"), ("survivorship", "Survivorship bias"), ("overfitting", "Overfitting"), ("sample", "Small sample size"),
          ("coverage", "Incomplete market coverage"), ("leverage", "Excessive leverage"), ("fills", "Optimistic fill assumptions"), ("concentration", "Dependence on a few profitable trades"),
          ("clustering", "Volatility clustering"), ("regime", "Market regime dependence"), ("costs", "Transaction-cost sensitivity"), ("parameters", "Unstable parameter choices"),
          ("benchmark", "Inappropriate benchmark comparisons"), ("intrabar", "Ambiguous intrabar execution"), ("contracts", "Option contract-selection bias")]
ORDER = {"high": 0, "medium": 1, "low": 2, "ok": 3, "na": 4}


def _p(p: float | None) -> str:
    return "n/a" if p is None else "below 0.0001" if p < 1e-4 else f"{p:.4f}"


def _f(topic: str, severity: str, title: str, evidence: str, test: str | None = None, table: dict | None = None) -> dict:
    return {"topic": topic, "name": dict(TOPICS)[topic], "severity": severity, "title": title, "evidence": evidence, "test": test, "table": table}


# ----------------------------------------------------------------------------- look-ahead probe
class ShockedStore:
    """A view of the market store in which every price observed AFTER `t_cut` is scrambled.
    A rule that only reads the past returns the same answer on it."""

    def __init__(self, store, t_cut: int, seed: int = 11):
        self._s, self._t, self._seed = store, int(t_cut), seed

    def __getattr__(self, k):
        return getattr(self._s, k)

    def _factor(self, n: int, salt: int) -> np.ndarray:
        return np.random.default_rng(self._seed + salt).uniform(0.6, 1.4, n)

    def read_best_bars(self, sym, start=None, end=None):
        import polars as pl
        df = self._s.read_best_bars(sym, start, end)
        if df.is_empty():
            return df
        ts = df["ts"].dt.epoch("s").to_numpy()
        en = df["end"].dt.epoch("s").to_numpy() if "end" in df.columns else ts + 60
        f = np.where(en > self._t, self._factor(df.height, len(sym)), 1.0)  # bars not yet complete at the cut
        fo = np.where(ts > self._t, f, 1.0)  # an open strictly after the cut is unknown too
        return df.with_columns((pl.col("open") * pl.Series(fo)).alias("open"), (pl.col("high") * pl.Series(f)).alias("high"), (pl.col("low") * pl.Series(f)).alias("low"),
                               (pl.col("close") * pl.Series(f)).alias("close"), (pl.col("volume") * pl.Series(f)).alias("volume") if "volume" in df.columns else pl.lit(None).alias("_x")).drop("_x", strict=False)

    def official_closes(self, sym, start=None, end=None):
        import polars as pl
        from strata.core import calendar as cal
        df = self._s.official_closes(sym, start, end)
        if df is None or df.is_empty():
            return df
        cut_d = cal.to_et(self._t * 1_000_000_000).date()
        known = int(cal.session_bounds(cut_d)[1].timestamp()) <= self._t  # the cut session's close has printed
        d = df["date"].to_list()
        f = np.array([1.0 if (x < cut_d or (x == cut_d and known)) else 1.3 for x in d])
        return df.with_columns((pl.col("close") * pl.Series(f)).alias("close"))


def lookahead_probe_study(cfg: dict, res: dict, store, k: int = 6) -> dict:
    """Re-run the study on sampled sessions after scrambling everything later than the entry instant."""
    from strata.core import calendar as cal
    from strata.research.study import run_study
    sig = cfg.get("entry_time") == "signal" or bool(cfg.get("sequence"))
    rows = res.get("sessions") or []
    qual = {r["date"]: r for r in rows}
    base = (res.get("baseline_sessions") or {}).get("dates") or []
    rng = np.random.default_rng(5)
    pick_q = list(rng.choice(sorted(qual), min(k, len(qual)), replace=False)) if qual else []
    non = [d for d in base if d not in qual]
    pick_n = list(rng.choice(non, min(k, len(non)), replace=False)) if non else []
    checked = diff = 0
    examples = []
    for d in pick_q + pick_n:
        day = date.fromisoformat(str(d))
        o, c = cal.session_bounds(day)
        if d in qual:
            t = int(o.timestamp()) + (qual[d].get("entry_minute", 570) - 570) * 60
        else:
            et = cfg.get("entry_time")
            t = int(c.timestamp()) if (sig or et == "close") else int(o.timestamp()) if et == "open" else int(o.timestamp()) + (cal.parse_hhmm(et) - 570) * 60
        try:
            r2 = run_study({**cfg, "start": str(d), "end": str(d), "target": None, "stop": None, "measure": "touch", "horizon": {"kind": "sessions", "sessions": 1} if cfg.get("entry_time") == "close" else cfg.get("horizon", {"kind": "session"})},
                           ShockedStore(store, t), max_sessions_out=5)
        except Exception as e:  # noqa: BLE001
            examples.append(f"{d}: could not be re-evaluated ({str(e)[:60]})")
            continue
        was = d in qual
        now = any(x["date"] == str(d) for x in r2.get("sessions", []))
        price_same = (not was) or (not now) or abs(r2["sessions"][0]["entry_price"] - qual[d]["entry_price"]) < 1e-6
        checked += 1
        if was != now or not price_same:
            diff += 1
            examples.append(f"{d}: qualified {'yes' if was else 'no'} originally, {'yes' if now else 'no'} with later prices scrambled")
    return {"checked": checked, "changed": diff, "examples": examples[:4], "sampled_qualifying": len(pick_q), "sampled_other": len(pick_n)}


def lookahead_probe_backtest(cfg: dict, trades: list[dict], store, k: int = 5) -> dict:
    """The same test for a strategy's entry rules: signals of sampled sessions with the future scrambled."""
    from strata.core import calendar as cal
    from strata.engine.scan import scan_signals
    days = sorted({t["entry_date"] for t in trades})
    rng = np.random.default_rng(5)
    pick = list(rng.choice(days, min(k, len(days)), replace=False)) if days else []
    by_day = {}
    for t in trades:
        by_day.setdefault(t["entry_date"], t)
    checked = diff = 0
    examples = []
    for d in pick:
        day = date.fromisoformat(d)
        t_sig = int(by_day[d].get("signal_ts") or by_day[d]["entry_ts"])
        one = {**copy.deepcopy(cfg), "start": d, "end": d}
        try:
            a = scan_signals(one, store)["signals"]
            b = scan_signals(one, ShockedStore(store, t_sig))["signals"]
        except Exception as e:  # noqa: BLE001
            examples.append(f"{d}: could not be re-evaluated ({str(e)[:60]})")
            continue
        fa = next((s for s in a if s["date"] == day), None)
        fb = next((s for s in b if s["date"] == day), None)
        checked += 1
        if fa is None or fb is None or fa["t"] != fb["t"] or abs(fa["price"] - fb["price"]) > 1e-6:
            diff += 1
            examples.append(f"{d}: signal {'at ' + cal.fmt_hhmm(fa['minute']) if fa else 'absent'} originally, {'at ' + cal.fmt_hhmm(fb['minute']) if fb else 'absent'} with later prices scrambled")
    # and one session without a trade, cut at its close: nothing after that day may matter
    return {"checked": checked, "changed": diff, "examples": examples[:4], "sampled_qualifying": len(pick), "sampled_other": 0}


# ----------------------------------------------------------------------------- experiments for backtests
def extra_scenarios(cfg: dict) -> list[dict]:
    """Parameter nudges and the unconditional benchmark, as extra rows for the stress pool."""
    out = []
    s0 = cfg["strategies"][0]
    for key, name in (("profit_target", "profit target"), ("stop_loss", "stop loss")):
        th = (s0.get("exits") or {}).get(key) or {}
        if th.get("enabled") and th.get("value"):
            for f in (0.75, 1.25):
                c = copy.deepcopy(cfg)
                c["strategies"][0]["exits"][key]["value"] = th["value"] * f
                out.append({"id": f"p_{key}_{f}", "group": "Parameters", "label": f"{name} {th['value']:g} -> {th['value'] * f:g}", "assumption": "one parameter nudged", "config": c, "severity": 1})
    for li, leg in enumerate(s0.get("legs", [])):
        if leg.get("kind", "option") != "option":
            continue
        if leg.get("dte", 0) > 0:
            for nd in sorted({max(0, leg["dte"] - 2), leg["dte"] + 3}):
                c = copy.deepcopy(cfg)
                for l2 in c["strategies"][0]["legs"]:
                    if l2.get("kind", "option") == "option" and l2.get("dte") == leg["dte"]:
                        l2["dte"] = nd
                out.append({"id": f"p_dte_{nd}", "group": "Parameters", "label": f"expiration {leg['dte']}DTE -> {nd}DTE", "assumption": "one parameter nudged", "config": c, "severity": 1})
        st = leg.get("strike") or {}
        if st.get("method") == "delta" and st.get("value"):
            for f in (0.8, 1.2):
                c = copy.deepcopy(cfg)
                c["strategies"][0]["legs"][li]["strike"]["value"] = round(st["value"] * f, 3)
                out.append({"id": f"p_delta_{li}_{f}", "group": "Parameters", "label": f"delta {st['value']:g} -> {st['value'] * f:.2g}", "assumption": "one parameter nudged", "config": c, "severity": 1})
        break  # nudging the first option leg is enough to show whether the result is a knife-edge
    cond = (s0.get("entry") or {}).get("conditions") or {}
    kids = [c for c in (cond.get("children") or []) if c.get("type") not in ("entry_times", "weekdays", "time_window")]
    if kids or (s0.get("entry") or {}).get("sequence"):
        c = copy.deepcopy(cfg)
        e = c["strategies"][0]["entry"]
        e["conditions"] = {**cond, "children": [x for x in (cond.get("children") or []) if x.get("type") in ("entry_times", "weekdays", "time_window")]}
        e["sequence"] = []
        if e["conditions"]["children"]:  # an entry time remains: the same structure, every session
            out.append({"id": "bench_uncond", "group": "Benchmark", "label": "Same position on every session (no entry condition)", "assumption": "the entry condition removed; everything else identical", "config": c, "severity": 0})
    return out


# ----------------------------------------------------------------------------- the review
def challenge_study(run: dict, res: dict, store, trials: int = 1, holdout: dict | None = None) -> dict:
    cfg = res["config"]
    st = res["stats"]
    n = st["qualifying_sessions"]
    out = []
    # --- look-ahead
    try:
        pr = lookahead_probe_study(cfg, res, store)
        if pr["checked"] == 0:
            out.append(_f("lookahead", "na", "Could not be tested", "No session could be re-evaluated with later prices scrambled. " + " ".join(pr["examples"])))
        elif pr["changed"]:
            out.append(_f("lookahead", "high", "The entry rule reads prices from after the decision", f"On {pr['changed']} of {pr['checked']} sampled sessions the rule's answer changed when every price after the entry instant was scrambled. " + "; ".join(pr["examples"]),
                          "future-scramble test"))
        else:
            out.append(_f("lookahead", "ok", "No look-ahead found", f"The entry rule was re-evaluated on {pr['checked']} sampled sessions ({pr['sampled_qualifying']} qualifying, {pr['sampled_other']} not) after every price later than the "
                          "entry instant had been scrambled, and gave the same answer and entry price each time. The outcome is measured only on bars after the entry.", "future-scramble test"))
    except Exception as e:  # noqa: BLE001
        out.append(_f("lookahead", "na", "Could not be tested", f"The future-scramble test failed to run: {str(e)[:160]}"))
    # --- dependence-aware statistics
    q, rest, qe = RS._study_sets(res)
    lt = RS._lift(q, rest)
    all_q = [r["date"] for r in res.get("sessions") or []]
    weeks = int(np.unique(R.cluster_ids(all_q, "week")).size) if all_q else 0
    adj = None if lt.get("p_value") is None else min(1.0, lt["p_value"] * max(1, trials))
    # --- selection
    if trials > 1:
        sev = "high" if (lt.get("p_iid") is not None and lt["p_iid"] < 0.05 and (adj is None or adj >= 0.05)) else "medium" if trials >= 10 else "low"
        out.append(_f("selection", sev, f"One of {trials} configurations tried in this session",
                      f"{trials} configurations have been tested in this research session"
                      + (f". Against other sessions the uncorrected p-value is {_p(lt['p_iid'])}" if lt.get("p_iid") is not None else "")
                      + (f"; resampling whole weeks gives {_p(lt['p_value'])}; corrected for {trials} tries it is {_p(adj)}." if adj is not None else ". No dependence-robust test is possible on this sample.")
                      + (" The result is significant only if the other tries are ignored." if sev == "high" else "")))
    else:
        out.append(_f("selection", "low", "A single configuration - as far as the ledger knows", "No other configuration has been recorded in this research session. Thresholds chosen from earlier experience or from charts are tries the ledger cannot see."))
    # --- survivorship
    sym = res.get("symbol")
    idx = sym in ("SPY", "QQQ", "IWM", "DIA", "SPX", "NDX", "RUT")
    out.append(_f("survivorship", "low" if idx else "medium", "Broad index product" if idx else "A single security chosen today",
                  f"{sym} is a broad index product whose membership is maintained by the index; studying it does not select a survivor, though the index itself drops failing companies." if idx else
                  f"{sym} was chosen from today's listings. A security that still exists and is liquid enough to study has, by construction, survived; the same rule on names that were delisted is not in this sample."))
    # --- overfitting
    thr = len(list(RS._perturb_tree(cfg.get("condition") or {})))
    knobs = thr + (1 if cfg.get("target") else 0) + (1 if cfg.get("stop") else 0) + (1 if cfg.get("entry_time") not in ("open", "close") else 0) + len(cfg.get("sequence") or [])
    per = n / max(knobs, 1)
    out.append(_f("overfitting", "high" if per < 10 else "medium" if per < 30 else "low", f"{knobs} chosen value{'s' if knobs != 1 else ''} for {n} observations",
                  f"The study fixes {knobs} value{'s' if knobs != 1 else ''} (thresholds, levels, entry time, sequence steps) and has {n} qualifying sessions: {per:.0f} observations per chosen value. "
                  + ("Fewer than 10 per value means the rule can be shaped around individual sessions." if per < 10 else "Below about 30 per value the estimate is easily bent by the choices." if per < 30 else "That is a reasonable ratio.")))
    # --- sample
    ci = R.cluster_prop_ci([float(v) for v in q.values()], list(q), "week") if q else {"ci": [None, None], "design_effect": None}
    sev = "high" if n < 30 or weeks < R.MIN_CLUSTERS else "medium" if n < 100 else "low"
    out.append(_f("sample", sev, f"{n} qualifying sessions in {weeks} separate weeks",
                  f"{n} qualifying sessions ({len(q)} with a known outcome) in {weeks} separate weeks. " + (f"The frequency is {st['probability'] * 100:.1f}% with a week-resampled 95% interval of {ci['ci'][0] * 100:.1f}% to {ci['ci'][1] * 100:.1f}% "
                  f"(the interval assuming independence is {st['probability_ci95'][0] * 100:.1f}% to {st['probability_ci95'][1] * 100:.1f}%). " if st.get("probability") is not None and ci["ci"][0] is not None else "")
                  + ("No statistical conclusion should be drawn from this sample." if sev == "high" else "")))
    # --- coverage
    ex = st.get("excluded_sessions") or 0
    ev = st.get("evaluated_sessions") or 0
    mix = st.get("resolution_mix") or {}
    share = ex / max(ex + ev, 1)
    out.append(_f("coverage", "medium" if share > 0.2 else "low" if ex else "ok", f"{ex} of {ex + ev} sessions could not be evaluated" if ex else "Every session in the period was evaluated",
                  (f"{ex} sessions ({share * 100:.0f}%) were excluded because the data to evaluate them is not stored: " + "; ".join(f"{e['reason']} ({e['count']})" for e in (res.get('excluded') or [])[:3]) + ". "
                   "If the excluded sessions differ systematically (for example, only recent years have intraday bars) the result describes the covered period only. " if ex else "")
                  + (f"Bar resolution of qualifying sessions: {', '.join(f'{k}: {v}' for k, v in mix.items())}." if mix else "")))
    out.append(_f("leverage", "na", "Not applicable to a market study", "A market study measures the underlying's behaviour and holds no position, so there is no leverage to assess."))
    out.append(_f("fills", "na", "Not applicable to a market study", "No orders are simulated. The entry price is the underlying's observed price at the entry instant; whether an option could have been bought near a fair price there is a separate question."))
    # --- concentration / clustering
    cz = RS.concentration(res, None)
    sh = next((f["value"] for f in cz["facts"] if "busiest" in f["label"]), None)
    de = ci.get("design_effect")
    out.append(_f("concentration", "na", "Not applicable (no profits)", "A study has no profits to concentrate. See volatility clustering for how the observations bunch in time."))
    sev = "high" if (de or 1) >= 2.5 or weeks < n * 0.4 else "medium" if (de or 1) >= 1.5 or weeks < n * 0.6 else "low"
    out.append(_f("clustering", sev, f"Observations bunch in time: {weeks} weeks for {n} sessions",
                  f"The {n} qualifying sessions fall in {weeks} separate weeks" + (f"; the three busiest months hold {sh} of them" if sh else "") + ". "
                  + (f"Resampling whole weeks, the sample is worth about {n / de:.0f} independent observations (design effect {de:.1f}). " if de else "")
                  + ("Tests that treat each session as an independent draw overstate the evidence here; " if sev != "low" else "")
                  + (f"with weeks resampled the p-value against other sessions is {_p(lt['p_value'])} (independence would give {_p(lt['p_iid'])})." if lt.get("p_value") is not None and lt.get("p_iid") is not None else ""),
                  "calendar-cluster bootstrap"))
    # --- regime
    vx = RS.volatility_explained(res, None, store)
    yrs = [y for y in res.get("by_year", []) if y["n"] >= 5 and y.get("probability") is not None]
    pb = st.get("baseline_probability")
    above = sum(1 for y in yrs if pb is not None and y["probability"] > pb)
    expl = next((f["value"] for f in vx["facts"] if f["label"] == "Explained by volatility"), None) if vx["answerable"] else None
    sev = "high" if expl not in (None, "n/a") and float(expl.rstrip("%")) >= 60 else "medium" if (yrs and above / len(yrs) < 0.6) or (expl not in (None, "n/a") and float(expl.rstrip("%")) >= 25) else "low"
    out.append(_f("regime", sev, "Depends on the volatility regime" if sev != "low" else "Similar across regimes",
                  (vx["answer"] + " " if vx["answerable"] else "Volatility could not be controlled for: " + vx["answer"] + " ")
                  + (f"By year: above the overall baseline in {above} of {len(yrs)} years with at least 5 qualifying sessions." if yrs else ""), "within-band comparison", vx["tables"][0] if vx.get("tables") else None))
    # --- costs
    cs = RS.cost_survival(res, None)
    out.append(_f("costs", "medium", "No costs are in this result", cs["answer"]))
    # --- parameters
    ai = RS.assumption_impact(res, None, None, store)
    if ai["answerable"]:
        flips = int(next(f["value"] for f in ai["facts"] if f["label"] == "Sign reversals"))
        tot = ai.get("variants") or 1
        out.append(_f("parameters", "high" if flips / tot > 0.3 else "medium" if flips else "low", f"{flips} of {tot} nearby settings reverse the conclusion" if flips else "Nearby settings agree", ai["answer"], "one-at-a-time perturbation", ai["tables"][0]))
    else:
        out.append(_f("parameters", "na", "Could not be tested", ai["answer"]))
    # --- benchmark
    lift, er = st.get("lift"), st.get("avg_end_return_pct")
    touch = cfg.get("measure", "touch") == "touch" and cfg.get("target")
    rob_dir = None
    if touch:
        b = res.get("baseline_sessions") or {}
        rest_e = [e for d, e in zip(b.get("dates", []), b.get("end_returns", [])) if d not in qe]
        if rest_e and er is not None:
            up = (cfg["target"].get("value") or 0) >= 0
            rob_dir = (er - float(np.mean(rest_e))) * (1 if up else -1)
    if touch and rob_dir is not None and rob_dir <= 0 and (lift or 0) > 0:
        out.append(_f("benchmark", "high", "A higher touch rate without a directional edge",
                      f"The target is touched {lift * 100:+.1f} points more often than on other sessions, but qualifying sessions END {abs(rob_dir):.2f} points {'lower' if (cfg['target'].get('value') or 0) >= 0 else 'higher'} than other sessions on average. "
                      "Ranges are simply wider after the condition, so both targets and stops are reached more often. Compared with sessions of similar range the edge disappears."))
    else:
        out.append(_f("benchmark", "low", "Compared with all other evaluated sessions", RS.benchmarks(res, None)["answer"]))
    # --- intrabar
    amb = st.get("ambiguous") or 0
    daily_n = mix.get("1d", 0)
    both = bool(cfg.get("target") and cfg.get("stop") and cfg.get("measure", "touch") == "touch")
    sev = "high" if n and amb / n > 0.2 else "medium" if amb or (both and daily_n) else "ok"
    out.append(_f("intrabar", sev, f"{amb} observations have an unknowable order" if amb else "No ambiguous observations",
                  (f"On {amb} of {n} qualifying sessions the target and the stop fall inside the same price bar, so which came first cannot be known at the stored resolution. They are excluded from the frequency; "
                   f"counting them all as hits or all as misses gives {st['probability_bounds'][1] * 100:.1f}% and {st['probability_bounds'][0] * 100:.1f}%. " if amb and st.get("probability_bounds", [None])[0] is not None else "")
                  + (f"{daily_n} qualifying sessions are resolved from one daily bar; finer bars would settle the order." if both and daily_n else "Outcomes are " + ("measured on closing prices, which have no intrabar ambiguity." if cfg.get("measure") == "close" else "resolved on the stored bars."))))
    out.append(_f("contracts", "na", "Not applicable to a market study", "No option contracts are selected in a market study."))
    # --- holdout
    extra = []
    if holdout and holdout.get("status") in ("contaminated", "exposed", "used"):
        from strata.research.ledger import holdout_note
        extra.append(holdout_note(holdout))
    return _finish(run, out, extra, "study")


def challenge_backtest(run: dict, res: dict, trades: list[dict], store, stress: dict | None, trials: int = 1, holdout: dict | None = None) -> dict:
    cfg = res.get("config") or {}
    m = res.get("metrics") or {}
    meta = res.get("meta") or {}
    cov = res.get("coverage") or {}
    n = len(trades)
    pnl = [t["pnl"] for t in trades]
    total = float(sum(pnl))
    out = []
    pricing = meta.get("pricing") or (cfg.get("pricing") or {}).get("mode")
    # --- look-ahead
    try:
        pr = lookahead_probe_backtest(cfg, trades, store) if trades else {"checked": 0, "changed": 0, "examples": []}
        if not pr["checked"]:
            out.append(_f("lookahead", "na", "Could not be tested", "There are no trades whose entry signal could be re-evaluated. " + " ".join(pr["examples"])))
        elif pr["changed"]:
            out.append(_f("lookahead", "high", "The entry rule reads prices from after the decision", f"On {pr['changed']} of {pr['checked']} sampled trade days the entry signal changed when every price after the signal instant was scrambled. " + "; ".join(pr["examples"]), "future-scramble test"))
        else:
            out.append(_f("lookahead", "ok", "No look-ahead found in the entry rule", f"The entry signal of {pr['checked']} sampled trade days was recomputed after every underlying price later than the signal instant had been scrambled; time and price of each signal "
                          "were unchanged. Fills use option prices observed at or before the fill time; contract selection uses the chain as it stood at the decision.", "future-scramble test"))
    except Exception as e:  # noqa: BLE001
        out.append(_f("lookahead", "na", "Could not be tested", f"The future-scramble test failed to run: {str(e)[:160]}"))
    # --- selection
    mt = R.hac_mean_test(pnl) if n >= 3 else {"p_value": None}
    weeks = int(np.unique(R.cluster_ids([t["entry_date"] for t in trades], "week")).size) if trades else 0
    cmt = R.cluster_mean_ci(pnl, [t["entry_date"] for t in trades], "week") if n >= 3 else {"ci": [None, None], "design_effect": None, "se": None}
    p_rob = None
    if cmt.get("se"):
        from scipy.stats import t as tdist
        p_rob = float(2 * (1 - tdist.cdf(abs(np.mean(pnl)) / cmt["se"], max(2, weeks - 1))))
    adj = None if p_rob is None else min(1.0, p_rob * max(1, trials))
    if trials > 1:
        sev = "high" if (p_rob is not None and p_rob < 0.05 and adj >= 0.05) else "medium" if trials >= 10 else "low"
        out.append(_f("selection", sev, f"One of {trials} configurations tried in this session", f"{trials} configurations have been tested in this research session. "
                      + (f"The average trade's p-value against zero is {_p(p_rob)} with weeks resampled; corrected for {trials} tries it is {_p(adj)}." if adj is not None else "The sample is too small for any test.")
                      + (" It is significant only if the other tries are ignored." if sev == "high" else "")))
    else:
        out.append(_f("selection", "low", "A single configuration - as far as the ledger knows", "No other configuration is recorded in this research session. Choices made from earlier experience or by looking at charts are tries the ledger cannot see."))
    sym = cfg.get("strategies", [{}])[0].get("symbol")
    idx = sym in ("SPY", "QQQ", "IWM", "DIA", "SPX", "NDX", "RUT")
    out.append(_f("survivorship", "low" if idx else "medium", "Broad index product" if idx else "A single security chosen today",
                  (f"{sym} is a broad index product. " if idx else f"{sym} was chosen from today's listings; names that were delisted are not in this sample. ")
                  + "For the options, only contracts that actually traded (or were quoted) can be selected - see contract-selection bias."))
    # --- overfitting
    s0 = cfg.get("strategies", [{}])[0]
    exits = s0.get("exits") or {}
    knobs = len(list(RS._perturb_tree((s0.get("entry") or {}).get("conditions") or {}))) + len((s0.get("entry") or {}).get("sequence") or []) + sum(1 for k in ("profit_target", "stop_loss", "trailing_stop", "time_exit", "max_hold", "underlying") if (exits.get(k) or {}).get("enabled")) \
        + 2 * len([l for l in s0.get("legs", []) if l.get("kind", "option") == "option"])
    per = n / max(knobs, 1)
    dsr = None
    daily = res.get("daily_returns") or []
    out.append(_f("overfitting", "high" if per < 5 else "medium" if per < 20 else "low", f"{knobs} chosen values for {n} trades",
                  f"The strategy fixes about {knobs} values (entry thresholds, exits, expiration and strike of each leg) and produced {n} trades: {per:.1f} trades per chosen value. "
                  + ("That is far too few to distinguish a rule that works from one that fits these trades." if per < 5 else "Each choice has little data behind it." if per < 20 else "A reasonable ratio.")
                  + (f" {trials} configurations were tried in this session." if trials > 1 else "")))
    del dsr, daily
    # --- sample
    sev = "high" if n < 30 or weeks < R.MIN_CLUSTERS else "medium" if n < 100 else "low"
    out.append(_f("sample", sev, f"{n} trades in {weeks} separate weeks",
                  f"{n} trades in {weeks} separate weeks. " + (f"The average trade is {np.mean(pnl):+,.2f} with a week-resampled 95% interval of {cmt['ci'][0]:+,.2f} to {cmt['ci'][1]:+,.2f}. " if n >= 3 and cmt["ci"][0] is not None else "")
                  + ("With fewer than 30 trades (or a dozen separate weeks) nothing can be concluded about an edge; the result is a description of these trades." if sev == "high" else
                     "Enough to test, not enough for a precise estimate." if sev == "medium" else "")))
    # --- coverage
    exb = meta.get("executability") or {}
    sig_d = cov.get("signals") if isinstance(cov.get("signals"), dict) else {}
    exe_d = cov.get("executable") if isinstance(cov.get("executable"), dict) else {}
    all_sig = sig_d.get("qualifying") or exb.get("signals")  # every session on which the entry rule fired
    exe = exe_d.get("signals") if exe_d.get("signals") is not None else exb.get("executed")  # those with observed option prices
    if all_sig and exe is not None:
        sh = exe / all_sig
        why = (cov.get("options") or {}).get("issues") or cov.get("unavailable") or []
        out.append(_f("coverage", "high" if sh < 0.3 else "medium" if sh < 0.8 else "low", f"{exe} of {all_sig} signals could be priced ({sh * 100:.0f}%)",
                      f"The entry rule fired on {all_sig} sessions; observed option prices existed for {exe} of them. " + (f"The other {all_sig - exe} are not in the result at all" + (": " + "; ".join(str(w.get('reason') or w)[:120] for w in why[:2]) if why else "") + ". " if all_sig > exe else "")
                      + ("The priced sessions are a small and possibly unrepresentative part of the signals (typically the most recent ones), so the result says little about the rule as a whole." if sh < 0.3 else "")))
    else:
        out.append(_f("coverage", "na", "Signal coverage is not recorded for this run", "The run does not record how many signals lacked option data, so coverage cannot be assessed."))
    # --- leverage
    cap = (cfg.get("capital") or {}).get("initial_capital") or 100000
    mx_risk = max((t.get("max_risk") or 0 for t in trades), default=0)
    util = m.get("max_capital_utilization")
    undefined = [t for t in trades if (t.get("max_loss") is None) and any(l["side"] == "sell" for l in t["legs"] if l["kind"] == "option")]
    worst = min(pnl, default=0)
    sev = "high" if undefined or (mx_risk and mx_risk / cap > 0.25) else "medium" if mx_risk and mx_risk / cap > 0.1 else "low"
    out.append(_f("leverage", sev, "Undefined-risk positions" if undefined else f"Largest position risks {mx_risk / cap * 100:.1f}% of capital",
                  f"The largest position's maximum risk is {mx_risk:,.0f} against capital of {cap:,.0f} ({mx_risk / cap * 100:.1f}%)" + (f"; peak buying-power use {util * 100:.0f}%" if util is not None else "") + f". The worst realised trade was {worst:+,.0f}. "
                  + (f"{len(undefined)} trades held short options without a defined maximum loss: the realised losses understate what a gap could do. " if undefined else "")
                  + "Returns are stated on the full capital; stated on the capital actually at risk they are larger in both directions."))
    # --- fills
    ex = meta.get("execution") or cfg.get("execution") or {}
    ages = exb.get("median_entry_print_age_seconds")
    if pricing == "trades":
        sl = (ex.get("trade_bar_slippage_pct") or 0) + (ex.get("extra_slippage_pct") or 0)
        out.append(_f("fills", "high" if not sl else "medium", "Filled at trade prints" + (" with no slippage" if not sl else f" plus {sl:g}%"),
                      "The option data is trade prints: the bid/ask spread at these moments was never observed. Fills assume the order traded at the last print" + (f" {sl:g}% worse" if sl else ", with nothing added")
                      + (f"; the median print was {ages:.0f} seconds old when used (limit {exb.get('max_trade_age_seconds')}s)" if ages is not None else "") + ". A print shows that someone traded at that price, not that this order would have. "
                      + "The execution stress test shows the result under worse fills; no spread is invented."))
    elif pricing == "modeled":
        out.append(_f("fills", "high", "Premiums are modelled, not observed", "Every option price in this run comes from a pricing model with assumed volatility and an assumed spread. No conclusion about profitability can rest on it."))
    else:
        fm = ex.get("fill_model", "conservative")
        out.append(_f("fills", "low" if fm == "conservative" else "medium", {"conservative": "Buys at the ask, sells at the bid", "midpoint": "Fills at the midpoint"}.get(fm, "Custom fill model"),
                      {"conservative": "Fills cross the full quoted spread (buy at the ask, sell at the bid), which is the conservative use of quote data.", "midpoint": "Fills at the midpoint assume half the spread is always saved; real orders usually pay part of it."}.get(fm, "A custom share of the spread is paid.")
                      + (f" Slippage of ${ex.get('slippage_per_contract'):g} per share is added." if ex.get("slippage_per_contract") else " No slippage beyond the spread is added.")
                      + " Quotes are end-of-day snapshots, so displayed size and intraday availability are unknown." * bool((res.get("coverage") or {}).get("mode_label", "").lower().find("end-of-day") >= 0)))
    # --- concentration
    cz = RS.concentration(res, trades)
    srt = sorted(pnl, reverse=True)
    wo3 = total - sum(srt[:3]) if n > 3 else None
    sev = "na" if total <= 0 else "high" if wo3 is not None and wo3 <= 0 else "medium" if wo3 is not None and wo3 / total < 0.4 else "low"
    out.append(_f("concentration", sev, "Not profitable, so no profit to concentrate" if total <= 0 else "The profit is three trades" if sev == "high" else "Most of the profit is three trades" if sev == "medium" else "Profit is spread across trades", cz["answer"],
                  "remove the best trades", cz["tables"][0] if cz.get("tables") else None))
    # --- clustering
    de = cmt.get("design_effect")
    iid = R.bootstrap_ci(pnl, method="iid", reps=1000) if n >= 3 else {"ci": [None, None]}
    by_m: dict[str, int] = {}
    for t in trades:
        by_m[t["entry_date"][:7]] = by_m.get(t["entry_date"][:7], 0) + 1
    top3 = sum(sorted(by_m.values(), reverse=True)[:3]) / max(n, 1)
    sev = "high" if top3 >= 0.6 and n >= 6 else "medium" if (de or 1) >= 1.5 or top3 >= 0.35 else "low"
    out.append(_f("clustering", sev, f"{top3 * 100:.0f}% of trades fall in three months" if n else "No trades",
                  f"The {n} trades fall in {weeks} separate weeks; the three busiest months hold {top3 * 100:.0f}% of them. "
                  + (f"Resampling whole weeks, the 95% interval of the average trade is {cmt['ci'][0]:+,.2f} to {cmt['ci'][1]:+,.2f}; treating trades as independent gives {iid['ci'][0]:+,.2f} to {iid['ci'][1]:+,.2f}. " if n >= 3 and cmt["ci"][0] is not None and iid["ci"][0] is not None else "")
                  + ("Trades taken in the same volatile stretch succeed or fail together, so they are much less than " + str(n) + " independent results." if sev != "low" else ""), "calendar-cluster bootstrap"))
    # --- regime
    vx = RS.volatility_explained(res, trades, store)
    ys: dict[str, list] = {}
    for t in trades:
        ys.setdefault(t["entry_date"][:4], []).append(t["pnl"])
    pos_y = sum(1 for v in ys.values() if sum(v) > 0)
    sev = "medium" if (len(ys) >= 3 and total > 0 and pos_y / len(ys) < 0.6) or len(ys) <= 1 else "low"
    out.append(_f("regime", sev, f"All trades are in {len(ys)} calendar year{'s' if len(ys) != 1 else ''}" if len(ys) <= 1 else f"Profitable in {pos_y} of {len(ys)} years",
                  (f"Every trade is in {', '.join(sorted(ys))}: one market environment. Nothing is known about how the strategy behaves in any other. " if len(ys) <= 1 else
                   "By year: " + "; ".join(f"{y}: {sum(v):+,.0f} ({len(v)})" for y, v in sorted(ys.items())) + ". ") + (vx["answer"] if vx["answerable"] else ""), None, vx["tables"][0] if vx.get("tables") and vx["answerable"] else None))
    # --- costs / parameters / benchmark from the stress run
    rows = {r["id"]: r for r in (stress or {}).get("rows", []) if r.get("status") == "ok"}
    if stress:
        s = stress["summary"]
        be = s.get("breakeven_slippage_pct")
        if (s.get("simulated_net_pnl") or 0) <= 0:
            sig_p = s.get("signal_net_pnl")
            out.append(_f("costs", "high" if (sig_p or 0) > 0 else "medium", "Costs turn a paper profit into a loss" if (sig_p or 0) > 0 else "Negative before and after costs",
                          f"With no costs at all the signal shows {sig_p:+,.0f}; as simulated the result is {s['simulated_net_pnl']:+,.0f}. " + ("The spread and fees are larger than the pattern." if (sig_p or 0) > 0 else "Costs are not what makes it lose."),
                          "execution stress test", _stress_table(stress, ("Reference", "Slippage", "Fill price"))))
        else:
            sev = "high" if be is not None and be <= 2 else "medium" if be is not None and be <= 5 else "low"
            out.append(_f("costs", sev, f"Breaks even at {be:.1f}% worse fills" if be is not None else "Positive at every slippage level tested",
                          f"As simulated {s['simulated_net_pnl']:+,.0f}; with no costs {s.get('signal_net_pnl') or 0:+,.0f}. " + (f"The result reaches zero when every fill is about {be:.1f}% worse" + (" - less than a typical option spread." if be <= 2 else ".") if be is not None else "It stays positive up to 10% worse per fill.")
                          + (" It turns negative under: " + "; ".join(s["turns_negative"][:4]) + "." if s.get("turns_negative") else ""), "execution stress test", _stress_table(stress, ("Reference", "Slippage", "Order delay", "Minimum volume", "Participation", "Multi-leg", "Bid/ask", "Combined"))))
        par = [r for r in rows.values() if r["group"] == "Parameters"]
        base_net = s.get("simulated_net_pnl") or 0
        if par:
            flips = [r for r in par if (r["net_pnl"] or 0) * base_net < 0 or (base_net > 0 and (r["net_pnl"] or 0) <= 0)]
            spread = max(abs((r["net_pnl"] or 0) - base_net) for r in par)
            out.append(_f("parameters", "high" if len(flips) >= max(1, len(par) // 2) else "medium" if flips else "low", f"{len(flips)} of {len(par)} one-step changes reverse the result" if flips else "Nearby settings give the same sign",
                          f"Each numeric choice was moved one step and the backtest re-run ({len(par)} variants; base {base_net:+,.0f}). Results range from {min(r['net_pnl'] or 0 for r in par):+,.0f} to {max(r['net_pnl'] or 0 for r in par):+,.0f}"
                          f" (largest change {spread:,.0f}). " + ("A conclusion that flips when a parameter moves one step is a property of that exact setting, not of the idea." if flips else ""),
                          "one-step parameter changes", {"title": "Parameter nudges", "columns": ["label", "trades", "net_pnl", "expectancy"], "rows": [{k: r.get(k) for k in ("label", "trades", "net_pnl", "expectancy")} for r in par]}))
        else:
            out.append(_f("parameters", "na", "No numeric exit or strike parameters to vary", "The strategy has no enabled premium exits, non-zero expiration or delta-based strike to nudge."))
        bu = rows.get("bench_uncond")
        if bu and bu["trades"]:
            base_exp = (rows.get("base") or {}).get("expectancy") or 0
            better = base_exp > (bu["expectancy"] or 0)
            out.append(_f("benchmark", "low" if better and bu["trades"] >= 30 else "medium" if better else "high", "Compared with the same position on every session",
                          f"The same position opened on every session with option data, without the entry condition, made {bu['expectancy'] or 0:+,.2f} per trade over {bu['trades']} trades ({bu['net_pnl']:+,.0f} in total). "
                          f"With the condition: {base_exp:+,.2f} per trade over {n}. " + ("The condition does not improve on simply holding the structure." if not better else "The condition does better than holding the structure unconditionally"
                          + (", though on few trades." if n < 30 else ".")) + " Buy-and-hold of the underlying is not a like-for-like comparison for an option position.", "unconditional benchmark"))
        else:
            out.append(_f("benchmark", "medium", "No like-for-like benchmark could be run", RS.benchmarks(res, trades)["answer"]))
    else:
        cs = RS.cost_survival(res, trades)
        out.append(_f("costs", "medium" if cs["answerable"] else "na", "First-order cost estimate only", cs["answer"]))
        out.append(_f("parameters", "na", "Not tested", "Parameter nudges are part of the stress run, which was not available."))
        out.append(_f("benchmark", "medium", "No like-for-like benchmark was run", RS.benchmarks(res, trades)["answer"]))
    # --- intrabar
    amb = (res.get("diagnostics") or {}).get("ambiguous_exits") or 0
    dr = meta.get("data_resolution") or {}
    out.append(_f("intrabar", "high" if n and amb / n > 0.2 else "medium" if amb else "ok", f"{amb} exits had target and stop inside one bar" if amb else "No ambiguous exits",
                  (f"On {amb} of {n} trades the profit target and the stop were both reachable inside the same bar; the engine takes the adverse one. " if amb else "No trade had its target and stop inside the same bar. ")
                  + (f"Underlying bar resolution of the sessions used: {dr}." if dr else "")))
    # --- contract selection
    cs_ = (res.get("contract_selection") or cov.get("contract_selection") or {})
    edge, exp_, constructed = cs_.get("at_band_edge"), cs_.get("expansions"), cs_.get("constructed")
    if pricing == "trades":
        sev = "high" if edge else "medium"
        out.append(_f("contracts", sev, f"{edge} selections at the edge of the downloaded strikes" if edge else "Selected from a downloaded band of strikes",
                      "Only a band of strikes around the price at the signal was downloaded, so the engine can only choose among those. "
                      + (f"{edge} selections landed on the outermost stored strike: the contract the rule really wanted may lie outside the band. " if edge else "No selection landed on the outermost stored strike" + (f" after {exp_} adaptive expansions of the band" if exp_ else "") + ". ")
                      + (f"{constructed} contract identifiers were constructed rather than taken from a verified listing. " if constructed else "")
                      + "Delta-based selection uses deltas implied by the prints themselves, which are noisy for contracts that trade rarely; and a contract is only eligible if it printed, which favours liquid strikes."))
    elif pricing == "historical":
        out.append(_f("contracts", "low", "Selected from the complete stored chain", "Every listed contract in the stored chain is eligible, using the chain as it stood at the decision time. Filters on spread, volume or open interest, if any, are part of the stated rules."))
    else:
        out.append(_f("contracts", "na", "Modelled contracts", "Strikes come from a synthetic grid, not from listed contracts."))
    extra = []
    if holdout and holdout.get("status") in ("contaminated", "exposed", "used"):
        from strata.research.ledger import holdout_note
        extra.append(holdout_note(holdout))
    return _finish(run, out, extra, "backtest")


def _stress_table(stress: dict, groups: tuple) -> dict:
    return {"title": "Re-simulated under different execution assumptions", "columns": ["label", "trades", "missed", "net_pnl", "expectancy"],
            "rows": [{k: r.get(k) for k in ("label", "trades", "missed", "net_pnl", "expectancy")} for r in stress["rows"] if r.get("status") == "ok" and r["group"] in groups]}


def _finish(run: dict, findings: list[dict], extra: list[str], kind: str) -> dict:
    have = {f["topic"] for f in findings}
    for t, _name in TOPICS:
        if t not in have:
            findings.append(_f(t, "na", "Not examined", "This topic could not be examined for this run."))
    findings.sort(key=lambda f: (ORDER[f["severity"]], [t for t, _ in TOPICS].index(f["topic"])))
    cnt = {s: sum(1 for f in findings if f["severity"] == s) for s in ORDER}
    hi = [f for f in findings if f["severity"] == "high"]
    if hi:
        verdict = f"{len(hi)} serious problem{'s' if len(hi) != 1 else ''}: " + "; ".join(f["name"].lower() for f in hi[:5]) + ". As it stands the result should not be relied on."
    elif cnt["medium"]:
        verdict = f"No fatal problem was found, but {cnt['medium']} weakness{'es' if cnt['medium'] != 1 else ''} limit what can be claimed: " + "; ".join(f["name"].lower() for f in findings if f["severity"] == "medium") + "."
    else:
        verdict = "None of the checks found a serious weakness. That is the absence of known problems on this data, not proof of an edge."
    return {"kind": "challenge", "run_id": run.get("id"), "subject": kind, "findings": findings, "counts": cnt, "verdict": verdict, "notes": extra + [
        "The reviewer looks for reasons the result could be wrong; it does not try to improve it.", "\"Could not be checked\" is not a pass: the stored data did not allow the test."],
            "tests_run": sorted({f["test"] for f in findings if f.get("test")})}


def run_challenge(run: dict, res: dict, trades: list[dict] | None, store, data_dir: str, trials: int = 1, holdout: dict | None = None, progress=None, should_cancel=None) -> dict:
    """The whole review, including the experiments it needs."""
    if RS.is_study(res):
        if progress:
            progress(0.2, "re-evaluating the rule with later prices scrambled")
        out = challenge_study(run, res, store, trials, holdout)
        if progress:
            progress(1.0, "done")
        return out
    from strata.research import execution as EX
    stress = None
    if trades and res.get("config"):
        ctx = EX.context_for(run, res, store)
        scen = EX.build_scenarios(ctx["config"], ctx["quotes_available"], ctx["eod"]) + extra_scenarios(ctx["config"])
        keep = {"signal", "base", "slip1", "slip2", "slip5", "slip10", "delay60", "vol10", "part10x", "quotes", "adverse", "legs2", "bidask"}
        scen = [s for s in scen if s["id"] in keep or s["group"] in ("Parameters", "Benchmark") or "skipped" in s]
        stress = _run_scenarios(ctx["config"], scen, data_dir, ctx["eod"], (lambda f, m: progress(0.1 + 0.75 * f, m)) if progress else None, should_cancel)
    if progress:
        progress(0.9, "re-evaluating entry signals with later prices scrambled")
    out = challenge_backtest(run, res, trades or [], store, stress, trials, holdout)
    out["stress"] = stress
    if progress:
        progress(1.0, "done")
    return out


def _run_scenarios(cfg: dict, scen: list[dict], data_dir: str, eod: bool, progress, should_cancel) -> dict:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    from strata.research import execution as EX
    todo = [s for s in scen if "config" in s]
    results: dict[str, dict] = {}
    workers = max(1, min(4, os.cpu_count() or 1, len(todo)))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=EX._worker_init, initargs=(data_dir,)) as pool:
        futs = {pool.submit(EX.evaluate_scenario, s["config"]): s for s in todo}
        done = 0
        for f in as_completed(futs):
            if should_cancel and should_cancel():
                pool.shutdown(wait=False, cancel_futures=True)
                from strata.engine.engine import BacktestCancelled
                raise BacktestCancelled()
            try:
                results[futs[f]["id"]] = f.result()
            except Exception as e:  # noqa: BLE001
                results[futs[f]["id"]] = {"ok": False, "error": str(e)[:300]}
            done += 1
            if progress:
                progress(done / len(todo), f"{done} of {len(todo)} stress and benchmark simulations")
    return EX.assemble(cfg, scen, results, eod)


def save_challenge(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "challenge.json").write_text(dumps(res))
    return {"challenge": True, "subject": res["subject"], "high": res["counts"]["high"], "medium": res["counts"]["medium"], "verdict": res["verdict"][:300], "target_run": res.get("run_id")}


_ = (math, timedelta)
