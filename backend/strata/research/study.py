"""Historical market studies (event studies) on real underlying prices.

"On sessions where <condition>, how often does <target> happen before <stop> within
<horizon>?" - answered directly from price history, with no option premiums involved.
Results are probabilities and price excursions of the UNDERLYING; they are never option
returns.

Method
  * the condition (same rule engine and expression language as strategies) is evaluated
    point-in-time at the entry tick of every session
  * each session uses the finest stored bars (1m ... 60m, else the daily bar)
  * target / stop are resolved by walking bars after the entry; when both fall inside the
    same bar (always the case for a daily bar that touches both) the order is unknowable and
    the observation is counted as AMBIGUOUS rather than guessed
  * sessions whose condition cannot be evaluated (missing data) are excluded and reported,
    never counted as "condition false"
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine.conditions import evaluate, extra_symbols, validate_tree
from strata.engine.engine import Context
from strata.engine.expr import Evaluator, ExprError
from strata.engine.market import Events, IvHistory, UnderlyingSeries, day_num, from_day_num
from strata.engine.models import CondNode, RegimeSettings, SeqStage
from strata.engine.sequence import SequenceTracker
from strata.research.robust import iid_mean_ci, wilson


class Level(BaseModel):
    kind: Literal["pct", "abs", "expr"] = "pct"
    value: float = 1.0  # signed: +1 = 1% (or $1) above the reference, -1 = below
    expr: Optional[str] = None  # price level, e.g. "prev_high" or "open + 0.5 * (prev_close - open)"


class Horizon(BaseModel):
    kind: Literal["session", "time", "sessions"] = "session"
    time: str = "12:00"  # for kind=time (New York)
    sessions: int = Field(1, ge=1, le=60)  # for kind=sessions: through the close of the N-th following session


class StudyConfig(BaseModel):
    symbol: str = "SPY"
    start: Optional[date] = None
    end: Optional[date] = None
    condition: CondNode = CondNode()
    sequence: list[SeqStage] = Field(default_factory=list)  # event sequence: the observation starts when its last stage happens
    entry_time: str = "open"  # open | close | HH:MM (New York) | signal (the first tick on which the rule fires)
    reference: Literal["entry", "open", "prev_close"] = "entry"
    target: Optional[Level] = None
    stop: Optional[Level] = None
    # touch: does price trade at the target before the stop?   close: is the last price of the horizon beyond the target?
    measure: Literal["touch", "close"] = "touch"
    direction: Optional[Literal["up", "down"]] = None  # side of the target; inferred from the level when omitted
    horizon: Horizon = Horizon()
    label: str = ""
    regime: RegimeSettings = RegimeSettings()


class _Strat:
    idx = 0

    def __init__(self, symbol, vol_symbol):
        self.symbol, self.vol_symbol = symbol, vol_symbol


class _Host:
    """Just enough of a Backtester for Context to evaluate rules without option chains."""

    def __init__(self, store: MarketStore, cfg: StudyConfig):
        self.store = store
        self.series: dict[str, UnderlyingSeries] = {}
        self.metas = {}
        self.chains: dict = {}
        self.iv_cache: dict = {}
        self.iv_hist: dict = {}
        self.regime_settings = {0: cfg.regime}
        self.events = Events(store.read_events())
        meta = store.symbol_meta(cfg.symbol)
        syms = {cfg.symbol} | extra_symbols(cfg.condition)
        for st_ in cfg.sequence:
            from strata.engine import expr as _X
            for src in [st_.when] + list(st_.capture.values()):
                try:
                    syms |= _X.symbols_used(src)
                except _X.ExprError:
                    pass
        if meta.vol_index:
            syms.add(meta.vol_index)
        for sym in syms:
            self.metas[sym] = store.symbol_meta(sym)
            df = store.read_best_bars(sym, None, cfg.end)
            if not df.is_empty():
                self.series[sym] = UnderlyingSeries(df, official=store.official_closes(sym, None, cfg.end))
        if store.option_days(cfg.symbol):
            self.iv_hist[cfg.symbol] = IvHistory.load(store, cfg.symbol)


def _boot(x: np.ndarray, seed: int = 1) -> list[float | None]:
    return iid_mean_ci(x, 2000, seed)


def _f(x):
    return None if x is None or not math.isfinite(x) else float(x)


def _first_touch(h: np.ndarray, l: np.ndarray, o: np.ndarray, tgt: float | None, stp: float | None, up: bool, stop_up: bool):
    """Indices of the first bar touching target / stop (gaps at a bar's open count)."""
    it = is_ = None
    if tgt is not None:
        m = (np.maximum(h, o) >= tgt) if up else (np.minimum(l, o) <= tgt)
        it = int(np.argmax(m)) if m.any() else None
    if stp is not None:
        m = (np.maximum(h, o) >= stp) if stop_up else (np.minimum(l, o) <= stp)
        is_ = int(np.argmax(m)) if m.any() else None
    return it, is_


def run_study(cfg: StudyConfig | dict, store: MarketStore | None = None, max_sessions_out: int = 4000, host: "_Host | None" = None) -> dict:
    if isinstance(cfg, dict):
        cfg = StudyConfig.model_validate(cfg)
    store = store or MarketStore()
    errs = validate_tree(cfg.condition)
    for st_ in cfg.sequence:
        from strata.engine.expr import validate as _v
        errs += [f"sequence stage {st_.name!r}: {x}" for src in [st_.when] + list(st_.capture.values()) for x in _v(src)]
    for lv in (cfg.target, cfg.stop):
        if lv is not None and lv.kind == "expr":
            from strata.engine.expr import validate
            errs += validate(lv.expr or "")
    if errs:
        raise ValueError("; ".join(errs))
    host = host or _Host(store, cfg)  # a grid of studies on one symbol can share the loaded price series
    ser = host.series.get(cfg.symbol)
    if ser is None or ser.empty():
        raise LookupError(f"No historical price data is stored for {cfg.symbol}. Download it in Data - free daily history is available from Yahoo.")
    meta = store.symbol_meta(cfg.symbol)
    ctx = Context(host)
    ctx.strategy = _Strat(cfg.symbol, meta.vol_index)
    first = from_day_num(ser.days[0])
    start = max(cfg.start or first, first)
    end = min(cfg.end or from_day_num(ser.days[-1]), from_day_num(ser.days[-1]))
    entry_minute = None if cfg.entry_time in ("open", "close", "signal") else cal.parse_hhmm(cfg.entry_time)
    study_extra: dict = {}
    rows: list[dict] = []
    base_first: list[str] = []
    excluded: dict[str, list[str]] = {}
    paths: list[tuple] = []
    res_count: dict[str, int] = {}
    vser = host.series.get(meta.vol_index) if meta.vol_index else None

    a0 = int(np.searchsorted(ser.days, day_num(start)))
    b0 = int(np.searchsorted(ser.days, day_num(end), side="right"))
    signal_mode = cfg.entry_time == "signal" or bool(cfg.sequence)
    tracker = SequenceTracker(cfg.sequence)
    base_dates: list[str] = []
    base_end: list[float] = []
    for di in range(max(a0, 1), b0):
        d = from_day_num(ser.days[di])
        s, e = int(ser.day_start[di]), int(ser.day_end[di])
        res = int(ser.d_res[di])
        t_open, t_close = int(ser.d_open_t[di]), int(ser.d_end[di])
        ctx.today, ctx.today_num = d, int(ser.days[di])
        ctx.close_minute = 570 + (t_close - t_open) // 60

        def at(t: int):
            ctx.t = t
            ctx.prev_t = max(t_open, int(ser.end[max(int(np.searchsorted(ser.end, t, side="left")) - 1, s)])) if t > t_open else t
            ctx.minute = 570 + (t - t_open) // 60
            ctx._feat.clear()

        def measure(t_e: int, qualifies: bool, explain: list, count_base: bool) -> bool:
            """Outcome of an observation starting at t_e. Returns False when it was excluded."""
            at(t_e)
            ev = Evaluator(ctx)
            entry_px = ser.price_at(t_e, d)
            ref = {"entry": entry_px, "open": float(ser.o[s]), "prev_close": float(ser.d_c[di - 1])}[cfg.reference]
            if not (math.isfinite(entry_px) and math.isfinite(ref) and ref > 0):
                excluded.setdefault("no price at the entry time", []).append(d.isoformat())
                return False

            def level(lv: Level | None):
                if lv is None:
                    return None
                if lv.kind == "pct":
                    return ref * (1 + lv.value / 100)
                if lv.kind == "abs":
                    return ref + lv.value
                v = ev.eval(lv.expr or "")
                return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

            try:
                tgt, stp = level(cfg.target), level(cfg.stop)
            except ExprError as ex:
                raise ValueError(str(ex)) from None
            if (cfg.target is not None and tgt is None) or (cfg.stop is not None and stp is None):
                excluded.setdefault("target/stop level not evaluable: " + (", ".join(sorted(ev.missing)) or "missing data"), []).append(d.isoformat())
                return False
            # horizon end
            if cfg.horizon.kind == "session":
                t_h = t_close
            elif cfg.horizon.kind == "time":
                t_h = t_open + (cal.parse_hhmm(cfg.horizon.time) - 570) * 60
                if res >= 86400 or t_h <= t_e:
                    excluded.setdefault(f"horizon 'until {cfg.horizon.time}' needs intraday bars" if res >= 86400 else "horizon ends before the entry", []).append(d.isoformat())
                    return False
            else:
                dj = min(di + cfg.horizon.sessions, len(ser.days) - 1)
                if dj < di + cfg.horizon.sessions:
                    return False  # not enough future sessions yet
                t_h = int(ser.d_end[dj])
            # bars strictly after the entry instant up to the horizon
            i0 = int(np.searchsorted(ser.ts, t_e, side="left"))
            i1 = int(np.searchsorted(ser.end, t_h, side="right"))
            h, l, o, c = ser.h[i0:i1], ser.l[i0:i1], ser.o[i0:i1], ser.c[i0:i1]
            if h.size == 0:
                if cfg.target is None and cfg.stop is None:
                    return False
                excluded.setdefault("no bars between entry and horizon", []).append(d.isoformat())
                return False
            up = (cfg.direction == "up") if cfg.direction else (tgt is None or tgt >= entry_px)
            stop_up = stp is not None and stp > entry_px
            it, is_ = _first_touch(h, l, o, tgt, stp, up, stop_up)
            path_res = int(ser.res[i0:i1].max())
            if cfg.measure == "close":
                # outcome decided by the final price of the horizon alone (no path dependence, so never ambiguous)
                last = float(c[-1])
                strict = cfg.target is not None and cfg.target.kind != "expr" and cfg.target.value == 0
                hit = tgt is not None and ((last > tgt if strict else last >= tgt) if up else (last < tgt if strict else last <= tgt))
                first_, it, is_ = ("target" if hit else "neither"), None, None
            elif it is not None and is_ is not None and it == is_:
                # same bar: unknowable, unless the bar opened beyond exactly one of the levels
                gap_t = (o[it] >= tgt) if up else (o[it] <= tgt)
                gap_s = (o[it] >= stp) if stop_up else (o[it] <= stp)
                first_ = "target" if gap_t and not gap_s else "stop" if gap_s and not gap_t else "ambiguous"
            elif it is not None and (is_ is None or it < is_):
                first_ = "target"
            elif is_ is not None:
                first_ = "stop"
            else:
                first_ = "neither"
            t_tgt = None
            if it is not None and path_res < 86400:
                t_tgt = (int(ser.end[i0 + it]) - t_e) / 60.0 if not ((o[it] >= tgt) if up else (o[it] <= tgt)) else (int(ser.ts[i0 + it]) - t_e) / 60.0
            sign = 1.0 if up else -1.0
            mfe = float(((h.max() if up else -l.min()) - sign * entry_px) / entry_px * 100)
            mae = float(((l.min() if up else -h.max()) - sign * entry_px) / entry_px * 100)
            end_ret = float((c[-1] / entry_px - 1) * 100)
            if count_base:
                base_first.append(first_)
                base_dates.append(d.isoformat())
                base_end.append(round(end_ret, 3))
            if not qualifies:
                return True
            resl = "1d" if path_res >= 86400 else f"{path_res // 60}m"
            res_count[resl] = res_count.get(resl, 0) + 1
            vix = None
            if vser is not None:
                pi = vser.prev_day_index(d)
                vix = float(vser.d_c[pi]) if pi >= 0 else None
            trend, _vol = ctx.regime(cfg.symbol)
            rows.append({
                "date": d.isoformat(), "entry_price": round(entry_px, 4), "reference": round(ref, 4), "target_level": _f(tgt), "stop_level": _f(stp),
                "outcome": first_, "target_touched": it is not None, "stop_touched": is_ is not None, "minutes_to_target": _f(t_tgt),
                "entry_minute": int(570 + (t_e - t_open) // 60), "target_minute": (int(570 + (t_e - t_open) // 60 + round(t_tgt))) if t_tgt is not None else None,
                "mfe_pct": round(mfe, 4), "mae_pct": round(mae, 4), "end_return_pct": round(end_ret, 4), "resolution": resl,
                "prev_return_pct": _f((ser.d_c[di - 1] / ser.d_c[di - 2] - 1) * 100) if di >= 2 else None, "gap_pct": _f((ser.o[s] / ser.d_c[di - 1] - 1) * 100),
                "vix_prev": _f(vix), "trend": trend, "weekday": d.weekday(), "why": [x["text"] for x in explain if not x.get("group")][:6],
            })
            if h.size <= 800:
                paths.append((h / entry_px - 1, l / entry_px - 1, o / entry_px - 1, (ser.end[i0:i1] - t_e) / 60.0, path_res, c / entry_px - 1, d.isoformat()))
            return True

        if signal_mode:
            # the observation starts at the first tick on which the rule (and its event sequence) fires
            if res >= 86400:
                excluded.setdefault("an intraday trigger cannot be observed with daily bars", []).append(d.isoformat())
                continue
            if cfg.sequence and res > 300:
                # the steps would collapse onto a handful of hourly ticks: excluded, never read as "did not happen"
                excluded.setdefault(f"a sequence of intraday events needs 5-minute or finer bars; only {res // 60}-minute bars are stored", []).append(d.isoformat())
                continue
            tracker.reset()
            ctx.seq = tracker
            found, missing = None, set()
            for t in [t_open] + [int(x) for x in ser.end[s:e] if t_open < x <= t_close]:
                at(t)
                done, sx, miss = tracker.advance(ctx)
                missing |= {f"sequence: missing {m}" for m in miss}
                if not done:
                    continue
                okc, expl, na = evaluate(cfg.condition, ctx)
                missing |= {x.split("[not evaluable: ")[-1].rstrip("]") if "[not evaluable: " in x else x for x in na}
                if okc:
                    found = (t, sx + expl)
                    break
            if found is None and missing:
                excluded.setdefault(("rule not evaluable: " + "; ".join(sorted(missing)))[:200], []).append(d.isoformat())
                continue
            if not measure(t_open, False, [], True):  # baseline: every evaluated session, observed from its open
                continue
            if found is not None:
                measure(found[0], True, found[1], False)
            continue
        if cfg.entry_time == "open":
            t_e = t_open
        elif cfg.entry_time == "close":
            t_e = t_close
        else:
            t_e = t_open + (entry_minute - 570) * 60
            if not (t_open <= t_e <= t_close) or (res >= 86400) or (t_e - t_open) % res != 0:
                excluded.setdefault(f"entry time {cfg.entry_time} ET not observable with {'daily' if res >= 86400 else str(res // 60) + '-minute'} bars", []).append(d.isoformat())
                continue
        ctx.t = ctx.prev_t = t_e
        ctx.minute = 570 + (t_e - t_open) // 60
        ctx._feat.clear()
        ok, explain, na = evaluate(cfg.condition, ctx)
        if na:
            key = "rule not evaluable: " + "; ".join(sorted({x.split("[not evaluable: ")[-1].rstrip("]") if "[not evaluable: " in x else x for x in na}))
            excluded.setdefault(key[:200], []).append(d.isoformat())
            continue
        measure(t_e, ok, explain, True)
    # every evaluated session (the baseline), kept compactly so any statistic can be traced back to its observations
    study_extra["baseline_sessions"] = {"dates": base_dates, "outcomes": "".join("tsna"[("target", "stop", "neither", "ambiguous").index(x)] for x in base_first), "end_returns": base_end}
    out = _summarise(cfg, rows, base_first, excluded, paths, res_count, start, end, store.is_synthetic(cfg.symbol), max_sessions_out)
    out.update(study_extra)
    out["excluded_sessions"] = {k: v for k, v in excluded.items()}
    if signal_mode:
        out["notes"].insert(0, "The observation starts at the first tick on which the rule fires. The baseline is every evaluated session observed from its open (the rule has no start time on other sessions).")
    return out


def _summarise(cfg: StudyConfig, rows, base_first, excluded, paths, res_count, start, end, synthetic, max_out) -> dict:
    n = len(rows)
    has_t, has_s = cfg.target is not None, cfg.stop is not None
    cnt = {k: sum(1 for r in rows if r["outcome"] == k) for k in ("target", "stop", "neither", "ambiguous")}
    resolved = n - cnt["ambiguous"]
    bcnt = {k: base_first.count(k) for k in ("target", "stop", "neither", "ambiguous")}
    bn = len(base_first)
    bres = bn - bcnt["ambiguous"]
    p = cnt["target"] / resolved if resolved else None
    pb = bcnt["target"] / bres if bres else None
    pval = None
    if resolved and bres and p is not None and pb is not None:
        # condition sessions vs all other sessions (two-proportion z-test)
        k2, n2 = bcnt["target"] - cnt["target"], bres - resolved
        if n2 > 0:
            pp = (cnt["target"] + k2) / (resolved + n2)
            se = math.sqrt(pp * (1 - pp) * (1 / resolved + 1 / n2)) if 0 < pp < 1 else 0
            if se > 0:
                from scipy.stats import norm
                pval = float(2 * (1 - norm.cdf(abs((p - k2 / n2) / se))))
    mfe = np.array([r["mfe_pct"] for r in rows], float)
    mae = np.array([r["mae_pct"] for r in rows], float)
    endr = np.array([r["end_return_pct"] for r in rows], float)
    ttt = np.array([r["minutes_to_target"] for r in rows if r["outcome"] == "target" and r["minutes_to_target"] is not None], float)
    stats = {
        "qualifying_sessions": n, "evaluated_sessions": bn, "excluded_sessions": sum(len(v) for v in excluded.values()),
        "target_first": cnt["target"], "stop_first": cnt["stop"], "neither": cnt["neither"], "ambiguous": cnt["ambiguous"],
        "probability": p, "probability_ci95": wilson(cnt["target"], resolved) if resolved else [None, None],
        "probability_bounds": [cnt["target"] / n, (cnt["target"] + cnt["ambiguous"]) / n] if n else [None, None],
        "baseline_probability": pb, "baseline_ci95": wilson(bcnt["target"], bres) if bres else [None, None], "baseline_sessions": bn,
        "lift": (p - pb) if p is not None and pb is not None else None, "p_value_vs_rest": pval,
        "avg_mfe_pct": _f(mfe.mean()) if n else None, "median_mfe_pct": _f(np.median(mfe)) if n else None,
        "avg_mae_pct": _f(mae.mean()) if n else None, "median_mae_pct": _f(np.median(mae)) if n else None,
        "avg_end_return_pct": _f(endr.mean()) if n else None, "end_return_ci95": _boot(endr) if n else [None, None], "median_end_return_pct": _f(np.median(endr)) if n else None,
        "pct_positive_end": _f((endr > 0).mean()) if n else None,
        "median_minutes_to_target": _f(np.median(ttt)) if ttt.size else None, "timed_target_hits": int(ttt.size),
        "resolution_mix": res_count,
    }
    by_year: dict[str, dict] = {}
    for r in rows:
        y = by_year.setdefault(r["date"][:4], {"year": r["date"][:4], "n": 0, "target": 0, "stop": 0, "ambiguous": 0, "end": []})
        y["n"] += 1
        y["target"] += r["outcome"] == "target"
        y["stop"] += r["outcome"] == "stop"
        y["ambiguous"] += r["outcome"] == "ambiguous"
        y["end"].append(r["end_return_pct"])
    years = [{"year": y["year"], "n": y["n"], "target": int(y["target"]), "stop": int(y["stop"]), "ambiguous": int(y["ambiguous"]),
              "probability": (y["target"] / (y["n"] - y["ambiguous"])) if y["n"] - y["ambiguous"] else None, "avg_end_return_pct": float(np.mean(y["end"]))} for y in sorted(by_year.values(), key=lambda x: x["year"])]

    def bucket(key_fn, order):
        g: dict[str, list] = {}
        for r in rows:
            g.setdefault(key_fn(r), []).append(r)
        out = []
        for k in order + sorted(set(g) - set(order)):
            if k in g:
                rs = g[k]
                res_n = sum(1 for r in rs if r["outcome"] != "ambiguous")
                out.append({"key": k, "n": len(rs), "probability": (sum(1 for r in rs if r["outcome"] == "target") / res_n) if res_n else None,
                            "ci95": wilson(sum(1 for r in rs if r["outcome"] == "target"), res_n), "avg_end_return_pct": float(np.mean([r["end_return_pct"] for r in rs]))})
        return out

    def vix_b(r):
        v = r["vix_prev"]
        return "n/a" if v is None else "VIX < 15" if v < 15 else "VIX 15-20" if v < 20 else "VIX 20-30" if v < 30 else "VIX 30+"

    regimes = {"vix": bucket(vix_b, ["VIX < 15", "VIX 15-20", "VIX 20-30", "VIX 30+", "n/a"]), "trend": bucket(lambda r: r["trend"], ["bull", "neutral", "bear", "unknown"])}
    # intraday charts (only sessions with intraday paths)
    intr = [pth for pth in paths if pth[4] < 86400]
    cum = tod = None
    if intr and has_t:
        tt = np.array([r["minutes_to_target"] for r in rows if r["resolution"] != "1d" and r["outcome"] == "target" and r["minutes_to_target"] is not None], float)
        n_intr = sum(1 for r in rows if r["resolution"] != "1d")
        horizon = max(float(p_[3][-1]) for p_ in intr)
        grid = np.arange(0, horizon + 1, 5 if horizon <= 400 else 30)
        cum = {"minutes": grid.tolist(), "probability": [float((tt <= g).sum() / n_intr) for g in grid], "sessions": n_intr}
        if tt.size:
            hh, edges = np.histogram(tt, bins=min(26, max(4, int(horizon // 15))), range=(0, horizon))
            tod = {"minutes": ((edges[:-1] + edges[1:]) / 2).tolist(), "hits": hh.tolist()}

    def hist(x, bins=30):
        if x.size == 0:
            return None
        lo, hi = np.quantile(x, [0.005, 0.995])
        if hi <= lo:
            hi = lo + 1e-6
        c, e = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
        return {"x": ((e[:-1] + e[1:]) / 2).tolist(), "counts": c.tolist()}

    # target-vs-stop sensitivity on the stored paths (percent of entry price)
    sens = None
    if paths and (has_t or has_s):
        up = (cfg.target.value >= 0) if (cfg.target is not None and cfg.target.kind != "expr") else True
        scale = float(np.median(np.abs(mfe))) if n else 1.0
        base_t = abs(cfg.target.value) if cfg.target is not None and cfg.target.kind == "pct" else max(0.25, round(scale, 2))
        base_s = abs(cfg.stop.value) if cfg.stop is not None and cfg.stop.kind == "pct" else base_t
        tg = sorted({round(base_t * k, 3) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)})
        sg = sorted({round(base_s * k, 3) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)})
        P = np.zeros((len(sg), len(tg)))
        A = np.zeros((len(sg), len(tg)))
        for si_, sv in enumerate(sg):
            for ti_, tv in enumerate(tg):
                hit = amb = 0
                for (hh_, ll_, oo_, _m, _r, *_rest) in paths:
                    if up:
                        mt, ms = np.maximum(hh_, oo_) >= tv / 100, np.minimum(ll_, oo_) <= -sv / 100
                    else:
                        mt, ms = np.minimum(ll_, oo_) <= -tv / 100, np.maximum(hh_, oo_) >= sv / 100
                    a = int(np.argmax(mt)) if mt.any() else None
                    b = int(np.argmax(ms)) if ms.any() else None
                    if a is not None and b is not None and a == b:
                        amb += 1
                    elif a is not None and (b is None or a < b):
                        hit += 1
                tot = len(paths) - amb
                P[si_, ti_] = hit / tot if tot else np.nan
                A[si_, ti_] = amb / len(paths)
        sens = {"targets_pct": tg, "stops_pct": sg, "probability": [[_f(v) for v in row] for row in P.tolist()], "ambiguous_share": A.tolist(), "direction": "up" if up else "down", "sessions": len(paths)}

    path_summary = _path_summary(paths, endr)
    notes = []
    if n and n < 30:
        notes.append(f"Only {n} qualifying sessions: the confidence interval is wide; treat the estimate as indicative.")
    if cnt["ambiguous"]:
        notes.append(f"{cnt['ambiguous']} observations are ambiguous (target and stop fell inside the same price bar, so the order is unknowable at that resolution). They are excluded from the probability; the bounds show both extremes.")
    if cfg.measure == "close":
        notes.append("Outcome measured on the final price of the horizon (close versus the target level); intraday touches along the way do not count.")
    if res_count.get("1d") and (has_t or has_s) and cfg.measure == "touch":
        notes.append(f"{res_count['1d']} of {n} qualifying sessions only have daily bars: touches are known from the day's high/low but not their timing. Add intraday history (Data) to resolve order and time-to-target.")
    notes.append("Underlying price behaviour only - these are not option returns.")
    return {
        "kind": "study", "config": cfg.model_dump(mode="json"), "symbol": cfg.symbol, "period": [start.isoformat(), end.isoformat()],
        "synthetic": bool(synthetic), "stats": stats, "by_year": years, "regimes": regimes, "cumulative": cum, "time_of_day": tod,
        "mfe_hist": hist(mfe), "mae_hist": hist(mae), "end_hist": hist(endr), "sensitivity": sens, "path_summary": path_summary,
        "excluded": [{"reason": k, "count": len(v), "dates": v[:12]} for k, v in sorted(excluded.items(), key=lambda kv: -len(kv[1]))],
        "sessions": rows[-max_out:], "sessions_truncated": n > max_out, "notes": notes,
    }


def _path_summary(paths: list, endr: np.ndarray) -> dict | None:
    """The intraday paths of the observed days from a common start (0% at the entry): percentile bands on a minute grid, a
    sample of the days themselves, and the days that ended furthest from the middle. Percent of the entry price."""
    intr = [p_ for p_ in paths if p_[4] < 86400 and len(p_) >= 7 and len(p_[3])]
    out: dict = {"end_quartiles": [float(np.quantile(endr, 0.25)), float(np.quantile(endr, 0.75))] if endr.size else None}
    if not intr:
        return out
    horizon = float(np.median([p_[3][-1] for p_ in intr]))
    step = 1 if horizon <= 90 else 5 if horizon <= 420 else 30
    grid = np.arange(0, horizon + step / 2, step)
    mat = np.full((len(intr), grid.size), np.nan)
    for i, p_ in enumerate(intr):
        mins, cl = np.asarray(p_[3], float), np.asarray(p_[5], float) * 100
        j = np.searchsorted(mins, grid, side="right") - 1
        ok = j >= 0
        row = np.full(grid.size, np.nan)
        row[ok] = cl[j[ok]]
        row[0] = 0.0
        mat[i] = row
    with np.errstate(all="ignore"):
        q = np.nanpercentile(mat, [10, 25, 50, 75, 90], axis=0)
        mean = np.nanmean(mat, axis=0)
    ends = np.array([np.asarray(p_[5])[-1] * 100 for p_ in intr])
    order = np.argsort(ends)
    pick = sorted(set(order[:5].tolist() + order[-5:].tolist() + np.linspace(0, len(intr) - 1, min(30, len(intr))).astype(int).tolist()))
    sample = [{"date": intr[i][6], "end": round(float(ends[i]), 3), "values": [None if not np.isfinite(v) else round(float(v), 3) for v in mat[i]], "extreme": bool(i in order[:5] or i in order[-5:])} for i in pick]
    out.update({"minutes": grid.tolist(), "p10": q[0].round(4).tolist(), "p25": q[1].round(4).tolist(), "median": q[2].round(4).tolist(), "p75": q[3].round(4).tolist(),
                "p90": q[4].round(4).tolist(), "mean": np.round(mean, 4).tolist(), "sample": sample, "days": len(intr)})
    return out


def session_chart(store: MarketStore, symbol: str, d: date, context_days: int = 0) -> dict:
    """Bars for replaying one session (finest stored resolution) plus prior-day levels."""
    df = store.read_best_bars(symbol, d - timedelta(days=10 + context_days * 2), d)
    if df.is_empty():
        raise LookupError(f"no price data for {symbol} around {d}")
    ser = UnderlyingSeries(df)
    di = ser.day_index(d)
    if di < 0:
        raise LookupError(f"{symbol} has no bars on {d}")
    lo = max(0, di - context_days)
    s, e = int(ser.day_start[lo]), int(ser.day_end[di])
    res = int(ser.d_res[di])
    if res >= 86400:  # daily only: show surrounding sessions for context
        lo = max(0, di - 30)
        full = UnderlyingSeries(store.read_best_bars(symbol, d - timedelta(days=70), d + timedelta(days=20)))
        bars = [[int(full.d_open_t[i]), float(full.d_o[i]), float(full.d_h[i]), float(full.d_l[i]), float(full.d_c[i])] for i in range(len(full.days))]
        return {"symbol": symbol, "date": d.isoformat(), "resolution": "1d", "bars": bars, "prev_close": float(ser.d_c[di - 1]) if di else None,
                "prev_high": float(ser.d_h[di - 1]) if di else None, "prev_low": float(ser.d_l[di - 1]) if di else None, "open": float(ser.d_o[di]), "session_start": int(ser.d_open_t[di])}
    bars = [[int(ser.ts[i]), float(ser.o[i]), float(ser.h[i]), float(ser.l[i]), float(ser.c[i])] for i in range(s, e)]
    return {"symbol": symbol, "date": d.isoformat(), "resolution": f"{res // 60}m", "bars": bars, "prev_close": float(ser.d_c[di - 1]) if di else None,
            "prev_high": float(ser.d_h[di - 1]) if di else None, "prev_low": float(ser.d_l[di - 1]) if di else None, "open": float(ser.d_o[di]), "session_start": int(ser.d_open_t[di])}
