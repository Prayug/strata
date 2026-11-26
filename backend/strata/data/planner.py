"""Dataset planner: what a piece of research needs, what is stored, and what can be acquired.

Given the engine configuration of a study or an options backtest it works out
  * which symbols, sessions and bar resolutions the rules need (including indicator
    warm-up, the volatility index and any cross-symbol rule),
  * whether option observations are needed, of which kind (bid/ask quotes or trade
    prints), for which sessions, expirations and strikes,
  * what is already stored - validated per session and, for selectively stored option
    trade bars, per contract - rather than by first/last date,
  * which provider can supply each gap, whether that is free, and how large the request is.

Nothing here downloads anything except `execute_action` / `acquire_trades`, which the
research pipeline calls for free sources only. Paid providers are listed, never used.
"""
from __future__ import annotations

import ast
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import INTRADAY_RES, DatasetKey, MarketStore
from strata.engine import expr as X
from strata.engine.conditions import extra_symbols, walk_leaves
from strata.engine.models import BacktestConfig, CondNode, LegSpec, StrategyConfig

# a download above any of these is "unusually large": it is described and needs approval first
LARGE_BYTES = 60_000_000
LARGE_SECONDS = 300
LARGE_REQUESTS = 2500
RES_LABEL = {86400: "daily", 3600: "60-minute", 1800: "30-minute", 900: "15-minute", 300: "5-minute", 60: "1-minute"}
TF_SECONDS = {"1m": 60, "2m": 60, "3m": 60, "5m": 300, "10m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600}
VIX_NAMES = {"vix", "vix_prev_close"}


def res_label(sec: int) -> str:
    return RES_LABEL.get(sec, f"{sec // 60}-minute")


def _today() -> date:
    return datetime.now(cal.ET).date()


def last_completed_session(now: datetime | None = None) -> date:
    now = now or datetime.now(cal.ET)
    d = now.date()
    if cal.is_trading_day(d) and now >= cal.session_bounds(d)[1]:
        return d
    return cal.prev_trading_day(d)


# ----------------------------------------------------------------------------- what the rules need
def _exprs(node: CondNode | None) -> list[str]:
    return [str((l.params or {}).get("expr", "")) for l in walk_leaves(node) if l.type == "expr"]


def _expr_needs(src: str) -> tuple[int, int, bool]:
    """(finest bar seconds needed, daily warm-up sessions, uses the volatility index)."""
    try:
        tree = X.parse(src)
    except X.ExprError:
        return 86400, 2, False
    sec, warm, vix = 86400, 2, False
    names = X.names_used(src)
    if names & X.INTRADAY_NAMES:
        sec = 300
    if names & VIX_NAMES:
        vix = True
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        consts = [a.value for a in list(n.args) + [k.value for k in n.keywords] if isinstance(a, ast.Constant)]
        tfs = [c for c in consts if isinstance(c, str) and c in X.TF]
        nums = [c for c in consts if isinstance(c, (int, float)) and not isinstance(c, bool)]
        f = n.func.id
        daily = "1d" in tfs or (not tfs and f in ("sma", "ema", "atr", "atr_pct", "rvol", "rvol_percentile", "rolling_high", "rolling_low", "return_pct", "session_return", "open_to_close",
                                                    "close", "session_high", "session_low", "day_volume_ratio", "relative_strength", "consecutive_down_days", "consecutive_up_days",
                                                    "consecutive_red_days", "consecutive_green_days", "prev_week_high", "prev_week_low", "prev_month_high", "prev_month_low"))
        for tf in tfs:
            if tf != "1d":
                sec = min(sec, TF_SECONDS.get(tf, 300))
        if not daily and not tfs and f in ("rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_pctb", "volume_ratio", "consecutive_bearish_candles", "consecutive_bullish_candles"):
            sec = min(sec, 300)
        if daily:
            look = int(max(nums)) if nums else 20
            if f == "rvol_percentile":
                look = 252 + 21
            if f in ("prev_month_high", "prev_month_low"):
                look = 45
            if f.startswith("consecutive"):
                look = 15
            warm = max(warm, min(look + 2, 600))
    return sec, warm, vix


def _clock_seconds(minute: int) -> int:
    """Coarsest stored bar length able to observe a clock time."""
    for r in (3600, 1800, 900, 300, 60):
        if (minute - 570) * 60 % r == 0:
            return r
    return 60


def _tree_needs(node: CondNode | None) -> dict:
    sec, warm, vix, minutes = 86400, 2, False, []
    for leaf in walk_leaves(node):
        p = leaf.params or {}
        if leaf.type == "expr":
            s, w, v = _expr_needs(str(p.get("expr", "")))
            sec, warm, vix = min(sec, s), max(warm, w), vix or v
        elif leaf.type == "entry_times":
            for x in str(p.get("times", "")).replace(";", ",").split(","):
                x = x.strip().lower()
                if x and x not in ("open", "close"):
                    m = cal.parse_hhmm(x)
                    minutes.append(m)
                    sec = min(sec, _clock_seconds(m))
        elif leaf.type in ("time_window", "session", "price_move", "opening_range", "day_range", "compare"):
            sec = min(sec, 300)
        elif leaf.type in ("vix", "vix_change"):
            vix = True
        elif leaf.type == "regime":
            warm, vix = max(warm, 202), vix or (p.get("vol", "any") != "any")
        elif leaf.type in ("iv", "iv_rank", "iv_percentile"):
            warm = max(warm, 2)
    return {"seconds": sec, "warmup": warm, "vix": vix, "minutes": minutes}


def leg_days(leg: LegSpec, asof: date | None = None) -> int:
    """Days to the expiration a leg aims at. A leg that names an exact expiration date has no fixed number: it is counted
    from `asof` (the first session of the run), which is the longest it can be."""
    if leg.expiration is not None:
        return max(0, (leg.expiration - asof).days) if asof is not None else 0
    return leg.dte


def hold_sessions(strat: StrategyConfig, asof: date | None = None) -> int:
    """Sessions after the entry session during which the position can still be open."""
    ex = strat.exits
    dte = max((leg_days(l, asof) for l in strat.legs if l.kind == "option"), default=0)
    h = dte  # until expiration unless something closes it earlier
    if ex.time_exit.enabled and ex.time_exit.dte_lte is None:
        h = 0
    if ex.max_hold.enabled and ex.max_hold.days:
        h = min(h, ex.max_hold.days)
    if ex.max_hold.enabled and ex.max_hold.minutes and ex.max_hold.minutes < 390:
        h = 0
    if ex.dte_exit.enabled:
        h = min(h, max(dte - ex.dte_exit.dte, 0))
    return max(0, int(math.ceil(h * 5 / 7)) if h > 5 else h)


def requirements(kind: str, cfg: dict, mode: str) -> dict:
    """What a study ("study") or an options backtest ("options") needs from the data."""
    if kind == "study":
        from strata.research.study import StudyConfig
        c = StudyConfig.model_validate(cfg)
        n = _tree_needs(c.condition)
        for src in [s.when for s in c.sequence] + [v for s in c.sequence for v in s.capture.values()]:
            s_, w_, v_ = _expr_needs(src)
            n["seconds"], n["warmup"], n["vix"] = min(n["seconds"], s_, 300), max(n["warmup"], w_), n["vix"] or v_  # a sequence is evaluated tick by tick
        sec, why = n["seconds"], []
        if c.sequence:
            why.append("the trigger is a sequence of intraday events")
        if c.entry_time not in ("open", "close", "signal"):
            sec = min(sec, _clock_seconds(cal.parse_hhmm(c.entry_time)))
            why.append(f"the observation starts at {c.entry_time} ET")
        if c.horizon.kind == "time":
            sec = min(sec, _clock_seconds(cal.parse_hhmm(c.horizon.time)))
            why.append(f"the horizon ends at {c.horizon.time} ET")
        if n["seconds"] < 86400:
            why.append("a rule reads intraday prices")
        und = {c.symbol: {"role": "primary", "seconds": sec, "why": "; ".join(why) or "daily bars are enough for this question", "warmup": n["warmup"]}}
        syms = extra_symbols(c.condition)
        return {"kind": "study", "mode": "study", "symbol": c.symbol, "underlying": und, "aux": _aux(c.symbol, syms, n["vix"], None), "options": None,
                "prefers_intraday": bool(c.target or c.stop), "entry": c.entry_time}
    c = BacktestConfig.model_validate(cfg)
    st = c.strategies[0]
    n = _tree_needs(st.entry.conditions)
    nx = _tree_needs(st.exits.custom)
    for src in [s.when for s in st.entry.sequence] + [v for s in st.entry.sequence for v in s.capture.values()]:
        s_, w_, v_ = _expr_needs(src)
        n["seconds"], n["warmup"], n["vix"] = min(n["seconds"], s_, 300), max(n["warmup"], w_), n["vix"] or v_  # a sequence is evaluated tick by tick
    for a in st.entry.adds:
        s_, w_, v_ = _expr_needs(a.when)
        nx["seconds"], nx["warmup"], nx["vix"] = min(nx["seconds"], s_, 300), max(nx["warmup"], w_), nx["vix"] or v_
    sec, why = n["seconds"], []
    if n["minutes"]:
        why.append("entry at " + ", ".join(cal.fmt_hhmm(m) for m in n["minutes"]) + " ET")
    if n["seconds"] < 86400 and not n["minutes"]:
        why.append("an entry rule reads intraday prices")
    ex = st.exits
    hold = hold_sessions(st, c.start)
    intraday_exit = bool((ex.time_exit.enabled and cal.parse_hhmm(ex.time_exit.time) < 955) or ex.underlying.enabled or nx["seconds"] < 86400 or (ex.max_hold.enabled and ex.max_hold.minutes))
    entries = [x.strip().lower() for l in walk_leaves(st.entry.conditions) if l.type == "entry_times" for x in str((l.params or {}).get("times", "")).replace(";", ",").split(",") if x.strip()]
    clock = "intraday" if (n["minutes"] or n["seconds"] < 86400) else "close" if entries == ["close"] else "open" if entries in (["open"], []) else "intraday"
    # mode "rules" describes what the entry rules alone need (used to find qualifying sessions);
    # execution on trade prints additionally needs minute-level clocks on the sessions that trade
    if mode == "trades":
        sec = min(sec, 300)
        why.append("trade-bar execution needs minute-level clocks")
    if intraday_exit and mode == "modeled":
        sec = min(sec, 3600)
        why.append("an intraday exit needs intraday prices")
    und = {st.symbol: {"role": "primary", "seconds": sec, "why": "; ".join(why) or "daily bars are enough for the entry rule", "warmup": max(n["warmup"], nx["warmup"])}}
    syms = st.symbols()
    vol = st.vol_symbol or None
    # a cross-symbol rule that reads the other symbol's intraday state needs that symbol's intraday bars too
    cross_intraday = set()
    for src in st.expressions():
        try:
            for nm in X.names_used(src):
                if "." in nm and X.is_ticker(nm.split(".")[0]) and (sec < 86400 or nm.split(".")[1] in X.INTRADAY_NAMES):
                    cross_intraday.add(nm.split(".")[0])
        except X.ExprError:
            pass
    opt = None
    if mode != "study":
        opt = {"obs": {"historical": "quote", "trades": "trade", "modeled": None}.get(mode), "clock": clock, "entry_minutes": n["minutes"], "hold_sessions": hold, "intraday_exit": intraday_exit,
               "legs": [l.model_dump(mode="json") for l in st.legs if l.kind == "option"],
               # a leg with an exact expiration date is 0DTE only on that date itself: over a run it is counted from the last and the first session
               "min_dte": min((leg_days(l, c.end) for l in st.legs if l.kind == "option"), default=None),
               "max_dte": max((max(leg_days(l, c.start), l.dte_max or 0) for l in st.legs if l.kind == "option"), default=None)}
    aux = _aux(st.symbol, syms, n["vix"] or nx["vix"] or mode == "modeled", vol)
    for s_ in cross_intraday & set(aux):
        aux[s_]["seconds"] = min(sec, 300)
    return {"kind": "options", "mode": mode, "symbol": st.symbol, "underlying": und, "aux": aux, "options": opt, "prefers_intraday": intraday_exit, "entry": clock}


def _aux(sym: str, cross: set[str], vix: bool, vol_symbol: str | None) -> dict:
    from strata.data.store import KNOWN_SPECS
    out = {}
    for s in sorted(cross - {sym}):
        out[s] = {"role": "cross-symbol rule", "seconds": 86400, "required": True}
    v = vol_symbol or KNOWN_SPECS.get(sym, {}).get("vol_index") or "VIX"
    if v and v != sym:
        out.setdefault(v, {"role": "volatility index", "seconds": 86400, "required": bool(vix)})
    return out


# ----------------------------------------------------------------------------- session-level coverage
def session_table(store: MarketStore, sym: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
    """One row per stored session: finest bar length, bar count, expected count, provider."""
    schema = {"date": pl.Date, "res": pl.Int32, "bars": pl.Int32, "provider": pl.String, "expected": pl.Int32, "complete": pl.Boolean}
    df = store.read_best_bars(sym, start, end)
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    t = (df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date"))
         .group_by("date").agg(pl.col("res").min().alias("res"), pl.len().alias("bars"), pl.col("provider").first()).sort("date"))
    exp = [1 if r >= 86400 else -(-cal.session_minutes(d) * 60 // r) for d, r in zip(t["date"].to_list(), t["res"].to_list())]
    t = t.with_columns(pl.Series("expected", exp, dtype=pl.Int32), pl.col("bars").cast(pl.Int32), pl.col("res").cast(pl.Int32))
    return t.with_columns((pl.col("bars") >= (pl.col("expected") * 0.9).floor()).alias("complete")).select(list(schema))


def _ranges(days: list[date]) -> list[list[str]]:
    """Consecutive trading sessions compressed to [first, last, count] ranges."""
    out, run = [], []
    for d in sorted(days):
        if run and cal.next_trading_day(run[-1]) != d:
            out.append([run[0].isoformat(), run[-1].isoformat(), len(run)])
            run = []
        run.append(d)
    if run:
        out.append([run[0].isoformat(), run[-1].isoformat(), len(run)])
    return out


def _issue(reason: str, days: list[date], **kw) -> dict:
    days = sorted(days)
    return {"reason": reason, "sessions": len(days), "first": days[0].isoformat(), "last": days[-1].isoformat(), "ranges": _ranges(days)[:40], "sample": [d.isoformat() for d in days[-6:]], **kw}


def underlying_coverage(store: MarketStore, req: dict, start: date, end: date, only: list[date] | None = None) -> dict:
    """Requested versus usable sessions for every symbol a rule reads, session by session.

    A session is usable for the primary symbol only if it is stored, its bars are fine
    enough for the rule, the day's bars are (nearly) complete and enough earlier sessions
    exist for indicator warm-up. Auxiliary symbols (volatility index, cross-symbol rules)
    must have a completed PRIOR session for every session they are needed on."""
    sessions = sorted(d for d in only if start <= d <= end) if only is not None else cal.trading_days(start, end)
    sset = set(sessions)
    out: dict = {"requested": {"start": start.isoformat(), "end": end.isoformat(), "sessions": len(sessions)}, "by_symbol": {}, "excluded": {}}
    excluded: dict[date, list[str]] = {}
    usable = set(sessions)
    for sym, need in {**req["underlying"], **req["aux"]}.items():
        primary = need["role"] == "primary" or int(need["seconds"]) < 86400  # anything read intraday is validated session by session
        tab = session_table(store, sym, None, end)
        have = {r["date"]: r for r in tab.iter_rows(named=True)}
        all_days = sorted(have)
        issues, ok = [], []
        sec = int(need["seconds"])
        if primary:
            missing = [d for d in sessions if d not in have]
            coarse = [d for d in sessions if d in have and have[d]["res"] > sec]
            partial = [d for d in sessions if d in have and have[d]["res"] <= sec and have[d]["res"] < 86400 and not have[d]["complete"]]
            warm_n = int(need.get("warmup") or 0)
            pos = {d: i for i, d in enumerate(all_days)}
            short = [d for d in sessions if d in have and d not in coarse and pos[d] < warm_n]
            if missing:
                issues.append(_issue("no prices stored for this session", missing, kind="missing"))
            if coarse:
                by: dict[int, list[date]] = {}
                for d in coarse:
                    by.setdefault(have[d]["res"], []).append(d)
                for r, ds in sorted(by.items()):
                    issues.append(_issue(f"only {res_label(r)} bars are stored; this rule needs {res_label(sec)} bars or finer", ds, kind="too_coarse", have_seconds=r, need_seconds=sec))
            if partial:
                issues.append(_issue("the session's intraday bars are incomplete", partial, kind="partial"))
            if short:
                issues.append(_issue(f"fewer than {warm_n} earlier sessions are stored for indicator warm-up", short, kind="warmup"))
            bad = set(missing) | set(coarse) | set(partial) | set(short)
            ok = [d for d in sessions if d not in bad]
            for d in bad:
                excluded.setdefault(d, []).append(f"{sym}: " + ("no prices" if d in missing else f"bars too coarse ({res_label(have[d]['res'])})" if d in coarse else "incomplete intraday bars" if d in partial else "not enough history for warm-up"))
            usable &= set(ok)
            half = [d for d in ok if cal.session_minutes(d) < 390]
            out["early_closes"] = len(half)
        else:
            # these values are read as of the previous completed session, so that session must exist
            miss = []
            if all_days:
                arr = np.array([d.toordinal() for d in all_days])
                for d in sessions:
                    i = int(np.searchsorted(arr, d.toordinal())) - 1
                    if i < 0 or cal.trading_days_between(all_days[i], d) > 3:
                        miss.append(d)
            else:
                miss = list(sessions)
            ok = [d for d in sessions if d not in set(miss)]
            if miss:
                issues.append(_issue(f"no {sym} history before this session", miss, kind="missing"))
                if need.get("required"):
                    for d in miss:
                        excluded.setdefault(d, []).append(f"{sym}: no data for the {need['role']}")
                    usable -= set(miss)
        stored = [d for d in all_days if d in sset]
        mix: dict[str, int] = {}
        for d in stored:
            k = f"{res_label(have[d]['res'])} via {have[d]['provider']}"
            mix[k] = mix.get(k, 0) + 1
        out["by_symbol"][sym] = {"role": need["role"], "needs": res_label(sec) + (" bars or finer" if sec < 86400 else " bars"), "why": need.get("why", ""), "required": primary or bool(need.get("required")),
                                 "warmup_sessions": need.get("warmup"), "stored": {"first": all_days[0].isoformat() if all_days else None, "last": all_days[-1].isoformat() if all_days else None, "sessions": len(all_days)},
                                 "usable_sessions": len(ok), "resolution_mix": mix, "issues": issues}
        if need["role"] != "primary" and primary:
            out["by_symbol"][sym]["required"] = True
    ok_days = sorted(usable)
    out["usable"] = {"sessions": len(ok_days), "first": ok_days[0].isoformat() if ok_days else None, "last": ok_days[-1].isoformat() if ok_days else None, "ranges": _ranges(ok_days)[:60]}
    by_reason: dict[str, list[date]] = {}
    for d, rs in excluded.items():
        by_reason.setdefault("; ".join(rs), []).append(d)
    out["excluded"] = [_issue(k, v) for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))][:30]
    out["_usable_days"] = ok_days
    return out


# ----------------------------------------------------------------------------- acquisition options
def _action(source: S.Source, dataset: str, symbols: list[str], start: date | None, end: date | None, why: str, sessions: int, **kw) -> dict:
    ok, status = source.available()
    try:
        est = source.plan(dataset, symbols, start, end)
    except Exception as e:  # noqa: BLE001  (estimating must never block planning)
        est = {"requests": 0, "est_bytes": 0, "est_seconds": 0, "note": f"size unknown: {e}"}
    large = est.get("est_bytes", 0) > LARGE_BYTES or est.get("est_seconds", 0) > LARGE_SECONDS or est.get("requests", 0) > LARGE_REQUESTS
    return {"id": f"{source.name}:{dataset}:{','.join(symbols)}:{start}:{end}", "source": source.name, "label": source.label, "dataset": dataset, "symbols": symbols,
            "start": start.isoformat() if start else None, "end": end.isoformat() if end else None, "tier": source.tier, "available": ok, "status": status,
            "sessions": sessions, "requests": int(est.get("requests", 0)), "est_bytes": int(est.get("est_bytes", 0)), "est_seconds": int(est.get("est_seconds", 0)), "cost": 0.0,
            "large": bool(large), "auto": bool(ok and source.tier in ("free", "free_key") and not large), "why": why, "note": est.get("note", ""), **kw}


def underlying_actions(store: MarketStore, req: dict, cov: dict, today: date | None = None) -> tuple[list[dict], list[dict]]:
    """Free downloads that would close underlying gaps, and the gaps nothing free can close."""
    today = today or _today()
    last_done = last_completed_session()
    actions, unfillable = [], []
    yahoo, cboe, alpaca = S.Yahoo(store), S.Cboe(store), S.Alpaca(store)
    al_ok, _ = alpaca.available()
    end = date.fromisoformat(cov["requested"]["end"])
    for sym, info in cov["by_symbol"].items():
        need = {**req["underlying"], **req["aux"]}[sym]
        sec = int(need["seconds"])
        stored_last = date.fromisoformat(info["stored"]["last"]) if info["stored"]["last"] else None
        stale = stored_last is None or stored_last < min(end, last_done)
        for iss in info["issues"] or ([{"kind": "stale", "sessions": 0, "first": None, "last": None}] if stale else []):
            kind = iss.get("kind")
            days = [date.fromisoformat(x) for r in iss.get("ranges", []) for x in _expand(r)] if iss.get("ranges") else []
            if kind == "warmup" and stored_last is not None and not stale:
                continue  # the full daily history is already stored: the first sessions simply have no earlier history
            if kind in ("missing", "stale", "warmup") and (sec >= 86400 or kind in ("warmup", "stale") or need["role"] != "primary"):
                src = cboe if sym in cboe.FILES else yahoo
                actions.append(_action(src, "vix_daily" if src is cboe else "daily", [sym], None, None,
                                       f"{sym} daily history" + (" is not stored" if stored_last is None else f" ends {stored_last}"), iss["sessions"], closes=kind))
                continue
            if kind in ("missing", "too_coarse", "partial"):
                # which free intraday windows contain these sessions at a fine-enough resolution
                got: set[date] = set()
                for res_name, n in (("1m", 29), ("5m", 59), ("60m", 729)):
                    if INTRADAY_RES[res_name] > sec:
                        continue
                    hit = [d for d in days if d >= today - timedelta(days=n) and d not in got]
                    if hit:
                        got |= set(hit)
                if got:
                    actions.append(_action(yahoo, "intraday", [sym], min(got), max(got), f"{len(got)} session(s) of {sym} need {res_label(sec)} bars; Yahoo serves recent intraday history", len(got), closes=kind))
                rest = sorted(set(days) - got)
                al_days = [d for d in rest if d >= alpaca.STOCKS_FIRST] if sym not in S.INDEX_SPECS else []
                if al_days:
                    a = _action(alpaca, "bars_1m", [sym], min(al_days), max(al_days), f"{len(al_days)} session(s) of {sym} need {res_label(sec)} bars; Alpaca has 1-minute bars since 2016", len(al_days), closes=kind)
                    if al_ok:
                        actions.append(a)
                    else:
                        unfillable.append({"symbol": sym, "sessions": len(al_days), "first": al_days[0].isoformat(), "last": al_days[-1].isoformat(), "reason": iss["reason"],
                                           "fix": "connect_alpaca", "detail": f"A free Alpaca account provides 1-minute {sym} bars since 2016 ({len(al_days)} of these sessions)."})
                    rest = sorted(set(rest) - set(al_days))
                if rest:
                    unfillable.append({"symbol": sym, "sessions": len(rest), "first": rest[0].isoformat(), "last": rest[-1].isoformat(), "reason": iss["reason"], "fix": "paid_or_import",
                                       "detail": (f"No free source has {res_label(sec)} {sym} bars for these sessions" + (" (before 2016)" if sym not in S.INDEX_SPECS else " (index intraday history)") +
                                                  ". A paid provider or an imported file would be needed; neither is used automatically.")})
    seen, uniq = set(), []
    for a in actions:
        if a["id"] not in seen:
            seen.add(a["id"])
            uniq.append(a)
    merged: dict[tuple, dict] = {}
    for u in unfillable:  # one entry per symbol and remedy, however many gaps it covers
        k = (u["symbol"], u["fix"])
        if k in merged:
            m = merged[k]
            m["sessions"] += u["sessions"]
            m["first"], m["last"] = min(m["first"], u["first"]), max(m["last"], u["last"])
        else:
            merged[k] = dict(u)
    for (sym, fix), m in merged.items():
        if fix == "connect_alpaca":
            m["detail"] = f"A free Alpaca account provides 1-minute {sym} bars since 2016, which would cover {m['sessions']:,} of these sessions ({m['first']} to {m['last']})."
    return uniq, list(merged.values())


def _expand(r: list) -> list[str]:
    a, b = date.fromisoformat(r[0]), date.fromisoformat(r[1])
    return [d.isoformat() for d in cal.trading_days(a, b)]


def execute_action(store: MarketStore, action: dict, progress=None, should_cancel=None) -> dict:
    """Run one acquisition. Only free (or free-key) sources are ever executed here."""
    src = S.get_source(action["source"], store)
    import os
    if os.environ.get("STRATA_OFFLINE", "") in ("1", "true", "yes"):
        raise S.SourceError("offline mode: automatic downloads are turned off (STRATA_OFFLINE)")
    if src.tier not in ("free", "free_key"):
        raise S.SourceError(f"{src.label} is a paid provider and is never called automatically")
    ok, msg = src.available()
    if not ok:
        raise S.SourceError(f"{src.label}: {msg}")
    start = date.fromisoformat(action["start"]) if action.get("start") else None
    end = date.fromisoformat(action["end"]) if action.get("end") else None
    return src.ingest(action["dataset"], action["symbols"], start, end, progress=progress, should_cancel=should_cancel)


# ----------------------------------------------------------------------------- option quotes (snapshots)
def expiration_index(store: MarketStore, key: DatasetKey) -> dict[str, list[int]]:
    """Session -> sorted days-to-expiration present in a quote dataset (cached per content version)."""
    import json
    base = store.dataset_dir(key)
    head = store.head(key).get("version") or store.commit(key).get("version")
    cp = base / "derived" / "expirations.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        if c.get("version") == head:
            return c["sessions"]
    out = {}
    for d in store.option_days(key.symbol, key.obs, dataset=key):
        p = base / "data" / f"{d.year}" / f"{d.isoformat()}.parquet"
        exps = pl.read_parquet(p, columns=["expiration"])["expiration"].unique().to_list()
        out[d.isoformat()] = sorted({(e - d).days for e in exps if e is not None})
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"version": head, "sessions": out}))
    return out


def _dte_ok(dtes: list[int], leg: dict, d: date | None = None) -> bool:
    """Is one of the expirations listed on session d (given as days to expiration) one the leg can use?"""
    from strata.engine.selection import _choose_expiration
    if not dtes:
        return False
    on = leg.get("expiration")
    if on:  # an exact date: that expiration or none
        on = date.fromisoformat(on) if isinstance(on, str) else on
        return d is not None and (on - d).days in set(int(x) for x in dtes)
    if d is None:  # only the distances are known: no date, so no calendar to say whether the market was open on it
        return _choose_expiration(np.asarray(dtes, np.int64), 0, LegSpec.model_validate(leg), dated=False) is not None
    from strata.engine.market import day_num
    today = int(day_num(d))
    return _choose_expiration(np.asarray(dtes, np.int64) + today, today, LegSpec.model_validate(leg)) is not None


def quote_coverage(store: MarketStore, req: dict, signals: list[dict], provider: str | None = None) -> dict:
    """For every signal: is there an option QUOTE snapshot on the entry session and on each
    session the position can stay open, and does the chain hold an expiration the legs can use?"""
    sym, opt = req["symbol"], req["options"]
    ds = store.option_dataset(sym, "quote", provider)
    out = {"observation": "bid/ask quotes", "dataset": ds.id if ds else None, "signals": len(signals), "executable": 0, "issues": [], "resolution": ds.res if ds else None}
    if not signals:
        return out
    if ds is None:
        out["issues"].append({"reason": f"no option quotes are stored for {sym}", "signals": len(signals), "kind": "no_dataset", "sample": [s["date"].isoformat() for s in signals[-6:]]})
        out["_missing"] = sorted({s["date"] for s in signals})
        return out
    eod = ds.res == "EOD"
    states = store.session_states(ds)
    days = set(store.option_days(sym, "quote", dataset=ds))
    exps = expiration_index(store, ds)
    hold = int(opt["hold_sessions"])
    bad: dict[str, list[date]] = {}
    need_more: set[date] = set()
    ok = 0
    for s in signals:
        d = s["date"]
        if eod and opt["clock"] != "close":
            bad.setdefault("the entry is not at the close, but only end-of-day quotes are stored (one snapshot at 16:00 ET)", []).append(d)
            continue
        if eod and (opt["min_dte"] == 0):
            bad.setdefault("0DTE contracts cannot be traded on end-of-day quotes (they expire at the snapshot)", []).append(d)
            continue
        if d not in days:
            bad.setdefault("no option quotes stored for the entry session", []).append(d)
            need_more.add(d)
            continue
        if (states.get(d.isoformat()) or {}).get("status") != "complete":
            bad.setdefault("the entry session's option quotes are only partially downloaded", []).append(d)
            need_more.add(d)
            continue
        if not all(_dte_ok(exps.get(d.isoformat(), []), leg, d) for leg in opt["legs"]):
            bad.setdefault("the stored chain has no expiration the legs can use on the entry session", []).append(d)
            continue
        later = [x for x in (cal.next_trading_day(d, k) for k in range(1, hold + 1)) if x <= last_completed_session()]
        gaps = [x for x in later if x not in days]
        if gaps:
            bad.setdefault("option quotes are missing on a session while the position is open", []).append(d)
            need_more |= set(gaps)
            continue
        ok += 1
    out["executable"] = ok
    out["issues"] = [{"reason": k, "signals": len(v), "kind": "quotes", "first": min(v).isoformat(), "last": max(v).isoformat(), "sample": [x.isoformat() for x in sorted(v)[-6:]]} for k, v in bad.items()]
    out["_missing"] = sorted(need_more)
    out["partial_sessions"] = sum(1 for v in states.values() if v.get("status") != "complete")
    return out


def quote_actions(store: MarketStore, req: dict, cov: dict) -> tuple[list[dict], list[dict]]:
    """Free sources that could supply the missing option-quote sessions (end-of-day only)."""
    sym, opt = req["symbol"], req["options"]
    missing: list[date] = cov.get("_missing") or []
    actions, unfillable = [], []
    if not missing:
        return actions, unfillable
    cur = DatasetKey.parse(cov["dataset"]).provider if cov.get("dataset") else None
    eod_ok = opt["clock"] == "close" and (opt["min_dte"] or 0) > 0
    if not eod_ok:
        unfillable.append({"symbol": sym, "sessions": len(missing), "first": missing[0].isoformat(), "last": missing[-1].isoformat(), "fix": "paid_or_trades",
                           "reason": "intraday option quotes are needed",
                           "detail": "Free sources only publish END-OF-DAY option quotes. Intraday bid/ask history needs a paid provider (ThetaData, Databento, Polygon), which is never used automatically. "
                                     "Option TRADE prints are available free and can be used instead (trade-price execution)."})
        return actions, unfillable
    gh, dh = S.GithubEod(store), S.DoltHub(store)
    in_gh = [d for d in missing if sym in gh.SYMBOLS and date(2008, 1, 2) <= d <= date(2025, 12, 12)]
    in_dh = [d for d in missing if d >= dh.FIRST and 10 <= (opt["min_dte"] or 0) and (opt["max_dte"] or 0) <= 60]
    # one option dataset per run: stay with the provider already stored when it can serve the gap
    if cur in (None, "github_eod") and in_gh:
        ys = sorted({d.year for d in in_gh})
        actions.append(_action(gh, "options_eod", [sym], date(ys[0], 1, 1), date(ys[-1], 12, 31), f"{len(in_gh)} session(s) need end-of-day {sym} option quotes; the archive ships whole years ({', '.join(map(str, ys))})", len(in_gh), closes="quotes"))
        rest = sorted(set(missing) - set(in_gh))
    elif cur in (None, "dolthub") and in_dh:
        actions.append(_action(dh, "options_eod", [sym], min(in_dh), max(in_dh), f"{len(in_dh)} session(s) need end-of-day {sym} option quotes (sparse chains: expirations 2-8 weeks out)", len(in_dh), closes="quotes", sessions_list=[d.isoformat() for d in in_dh]))
        rest = sorted(set(missing) - set(in_dh))
    else:
        rest = missing
    if rest:
        unfillable.append({"symbol": sym, "sessions": len(rest), "first": rest[0].isoformat(), "last": rest[-1].isoformat(), "fix": "paid_or_import", "reason": "no free end-of-day option quotes for these sessions",
                           "detail": f"No free source covers {sym} option quotes on these sessions" + (f" in the dataset this run uses ({cur})" if cur else "") + "."})
    return actions, unfillable


# ----------------------------------------------------------------------------- option trade bars (selective)
def _strike_step(strikes: list[float], ref: float, sym: str) -> float:
    near = sorted(k for k in set(strikes) if abs(k / ref - 1) < 0.05)
    if len(near) >= 3:
        return float(np.median(np.diff(near)))
    return S.STRIKE_STEP.get(sym, 1.0)


def leg_targets(legs: list[dict], price: float, vol: float | None, step: float) -> dict[str, tuple[float, int]]:
    """For each option leg: (strike aimed at, number of strikes kept either side).

    Uses only the underlying price at the signal time (and, for delta targets, a volatility
    level known before the session) - information that existed at the decision."""
    from strata.core.pricing import bs_greeks_vec
    out: dict[str, tuple[float, int]] = {}
    order = [l for l in legs if l["strike"]["method"] != "leg_offset"] + [l for l in legs if l["strike"]["method"] == "leg_offset"]
    atm = round(price / step) * step
    for l in order:
        m, v, call = l["strike"]["method"], float(l["strike"]["value"]), l["right"] == "C"
        if m == "atm_offset":
            unit = l["strike"].get("unit", "points")
            k = atm + (v if unit == "points" else v * step if unit == "strikes" else atm * v / 100)
            out[l["id"]] = (k, 3)
        elif m == "pct_otm":
            out[l["id"]] = (price * (1 + v / 100) if call else price * (1 - v / 100), 3)
        elif m == "fixed":
            out[l["id"]] = (v, 0)
        elif m in ("delta", "premium"):
            # where the requested delta or premium should sit, from a volatility level known before the session. It only
            # centres the band of strikes that is downloaded: the engine then picks by the delta or premium of observed
            # prints, and a selection that ends on the edge of the band widens it. The band grows with the distance from
            # the money, because that distance is what a wrong volatility guess (the smile is not flat) gets wrong.
            T = max((_leg_days(l) + 0.3) / 365.0, 0.25 / 365.0)
            sig = vol or 0.2
            span = max(60, int(math.ceil(5 * sig * math.sqrt(T) * price / step)))
            ks = np.arange(max(step, atm - span * step), atm + span * step + step / 2, step)
            g = bs_greeks_vec(np.full(ks.size, price), ks, np.full(ks.size, T), 0.04, 0.0, np.full(ks.size, sig), np.full(ks.size, call))
            if m == "delta":
                tgt = abs(v) if abs(v) <= 1 else abs(v) / 100
                k = float(ks[int(np.nanargmin(np.abs(np.abs(g[:, 1]) - tgt)))])
                base = 6
            else:
                k = float(ks[int(np.nanargmin(np.abs(g[:, 0] - v)))])
                base = 12
            out[l["id"]] = (k, max(base, int(math.ceil(0.5 * abs(k - atm) / step))))
        else:  # leg_offset
            ref = out.get(l["strike"].get("ref_leg") or "", (atm, 3))
            unit = l["strike"].get("unit", "points")
            k = ref[0] + (v if unit == "points" else v * step if unit == "strikes" else ref[0] * v / 100)
            out[l["id"]] = (k, max(ref[1], 3))
    return out


def _leg_days(leg: dict, d: date | None = None) -> int:
    """Days to the expiration a leg aims at: from its exact date when it names one, else its days-to-expiration."""
    e = leg.get("expiration")
    if e and d is not None:
        return max(0, ((date.fromisoformat(e) if isinstance(e, str) else e) - d).days)
    return int(leg.get("dte") or 0)


def _pick_expiration(avail: list[date], d: date, leg: dict) -> date | None:
    from strata.engine.selection import _choose_expiration
    if not avail:
        return None
    on = leg.get("expiration")
    if on:  # an exact date: that expiration or none
        on = date.fromisoformat(on) if isinstance(on, str) else on
        return on if on >= d and on in set(avail) else None
    from strata.engine.market import day_num, from_day_num
    # real dates, not offsets: the engine counts the days to a contract's effective expiration (a date on which the market
    # is closed expires the session before), and what is fetched has to be what the engine will choose
    arr = np.array(sorted({int(day_num(e)) for e in avail if e >= d}), np.int64)
    e = _choose_expiration(arr, int(day_num(d)), LegSpec.model_validate(leg))
    return None if e is None else from_day_num(int(e))


def trade_contracts(source: S.OptionBarSource, client, store: MarketStore, sym: str, legs: list[dict], sig: dict, hold: int, widen: int = 0) -> tuple[dict[date, list[S.Contract]], list[str]]:
    """The (session, contract) pairs one signal needs: the contracts its legs could select at
    the signal, on the entry session and every session the position can stay open.

    Point-in-time: expirations must be on or after the signal session, strikes are chosen
    around the underlying price AT THE SIGNAL, and a contract only counts as existing on a
    session if it traded there (decided by its bars, not by today's listing status)."""
    d, price = sig["date"], float(sig["price"])
    notes: list[str] = []
    vix = None
    vd = store.read_daily("VIX")
    if not vd.is_empty():
        prior = vd.filter(pl.col("date") < d)
        vix = float(prior["close"][-1]) / 100 if prior.height else None
    step0 = S.STRIKE_STEP.get(sym, 1.0)
    # an exact expiration date fixes the time to expiry on each signal's session: delta and premium targets are centred with it
    dated = [{**l, "dte": _leg_days(l, d)} if l.get("expiration") else l for l in legs]
    targets = leg_targets(dated, price, vix, step0)
    wanted: dict[date, list[S.Contract]] = {}
    sessions = [d] + [cal.next_trading_day(d, k) for k in range(1, hold + 1)]
    for leg in legs:
        k0, n = targets[leg["id"]]
        n = n * (1 + int(widen)) if leg["strike"]["method"] != "fixed" else n  # adaptive discovery: a wider band when a selection hit the edge of the last one
        # the provider is asked what is LISTED around the target, and the n listed strikes either side of it are kept: far
        # expirations are listed five or ten points apart, where "n points either side" would hold one strike or none
        reach = max(n * step0, 0.02 * price)
        lo, hi = k0 - reach - 1e-6, k0 + reach + 1e-6
        dte = int(leg["dte"])
        on = leg.get("expiration")
        on = date.fromisoformat(on) if isinstance(on, str) else on
        exact = on is not None or leg.get("dte_mode") == "exact" or dte == 0
        if on is not None:
            if on < d:
                notes.append(f"{d}: the {on} expiration had already expired on this session")
                continue
            e_lo = e_hi = on
        else:
            from strata.engine.selection import dte_tolerance
            tol = dte_tolerance(dte)  # the same distance the engine accepts on selectively stored data: what is fetched is what can be traded
            e_lo, e_hi = (d + timedelta(days=dte), d + timedelta(days=dte)) if exact else (d + timedelta(days=max(0, (leg.get("dte_min") or dte) - tol)), d + timedelta(days=(leg.get("dte_max") or dte) + tol))
        try:
            cs = source.list_contracts(client, sym, d, e_lo, e_hi, lo, hi, (leg["right"],))
        except S.SourceError as e:
            notes.append(f"{d}: {str(e)[:160]}")
            continue
        exp = _pick_expiration(sorted({c.expiration for c in cs}), d, leg)
        if exp is None:
            notes.append(f"{d}: no {leg['right']} contract expiring on {on} is listed for {sym} in the strike range asked for" if on is not None else f"{d}: no {'exact ' if exact else ''}{dte}DTE {leg['right']} expiration is listed for {sym}")
            continue
        chosen = [c for c in cs if c.expiration == exp]
        grid = sorted({c.strike for c in chosen})
        keep = ({min(grid, key=lambda k: abs(k - k0))} if n == 0 and grid else  # a fixed strike: that strike
                set([k for k in grid if k <= k0 + 1e-9][-(n + 1):]) | set([k for k in grid if k > k0 + 1e-9][:n]))
        chosen = [c for c in chosen if c.strike in keep]
        for s_ in sessions:
            if s_ <= exp:
                cur = wanted.setdefault(s_, [])
                cur += [c for c in chosen if c not in cur]
    return wanted, notes


def trade_plan(store: MarketStore, req: dict, signals: list[dict], provider: str | None = None, today: date | None = None) -> dict:
    """Which signals a free option-trade source could cover (by date window), per source."""
    today = today or _today()
    srcs = [s for s in S.option_bar_sources(store) if provider in (None, s.name)]
    cur = store.option_dataset(req["symbol"], "trade", provider)
    out = {"observation": "trade prints", "signals": len(signals), "sources": [], "dataset": cur.id if cur else None}
    for src in srcs:
        ok, status = src.available()
        a, b = src.bars_window(today)
        inside = [s for s in signals if a <= s["date"] <= b]
        out["sources"].append({"source": src.name, "label": src.label, "tier": src.tier, "available": ok, "status": status, "first": a.isoformat(), "last": b.isoformat(),
                               "signals_in_window": len(inside), "note": (src.capabilities(today)[-1].get("note") or "")})
    best = None
    for s_, info in zip(srcs, out["sources"]):
        if info["available"] and info["signals_in_window"] and (best is None or info["signals_in_window"] > best[1]["signals_in_window"]):
            best = (s_, info)
    if cur is not None and best is not None and cur.provider != best[0].name:
        # trade data is already stored from another provider: one dataset per run, so keep to the stored one if it can serve
        keep = next(((s_, i) for s_, i in zip(srcs, out["sources"]) if s_.name == cur.provider and i["available"] and i["signals_in_window"]), None)
        best = keep or best
    out["chosen"] = best[1]["source"] if best else (cur.provider if cur else None)
    return out


def acquire_trades(store: MarketStore, req: dict, signals: list[dict], source_name: str, progress=None, should_cancel=None, approve_large: bool = False, widen: dict | None = None) -> dict:
    """Stage two: download option trade bars only for the sessions that produced a signal
    and only for the contracts those signals could select. Returns the acquisition report
    (or an estimate with `needs_approval` when the request is unusually large)."""
    src = next(s for s in S.option_bar_sources(store) if s.name == source_name)
    sym, opt = req["symbol"], req["options"]
    a, b = src.bars_window()
    inside = [s for s in signals if a <= s["date"] <= b]
    outside = [s for s in signals if not (a <= s["date"] <= b)]
    wanted: dict[date, list[S.Contract]] = {}
    notes: list[str] = []
    with S._client() as client:
        for i, sig in enumerate(inside):
            if should_cancel and should_cancel():
                raise S.Cancelled()
            if progress and i % 5 == 0:
                progress(0.3 * i / max(len(inside), 1), f"Validating contracts for {sig['date']} ({i + 1}/{len(inside)})")
            w, n = trade_contracts(src, client, store, sym, opt["legs"], sig, int(opt["hold_sessions"]), int((widen or {}).get(sig["date"], 0)))
            notes += n
            for d, cs in w.items():
                cur = wanted.setdefault(d, [])
                cur += [c for c in cs if c not in cur]
    est = src.estimate(wanted, sym)
    large = est["est_bytes"] > LARGE_BYTES or est["est_seconds"] > LARGE_SECONDS or est["requests"] > LARGE_REQUESTS
    rep = {"source": src.name, "label": src.label, "signals_in_window": len(inside), "signals_outside_window": len(outside), "window": [a.isoformat(), b.isoformat()],
           "contract_sessions": est["pairs_requested"], "estimate": est, "notes": notes[:12], "large": large}
    if large and not approve_large:
        return {**rep, "needs_approval": True}
    sub = (lambda f, m: progress(0.3 + 0.7 * f, m)) if progress else None
    rep["result"] = src.acquire(sym, wanted, sub, should_cancel) if est["pairs_to_fetch"] else {"pairs_requested": est["pairs_requested"], "pairs_cached": est["pairs_cached"], "pairs_ok": 0, "pairs_empty": 0,
                                                                                               "pairs_unavailable": 0, "pairs_failed": 0, "bars": 0, "requests": 0, "pages": 0, "rejected": {}, "errors": [], "out_of_window": []}
    return rep


# Why a qualifying signal can or cannot be traded on stored TRADE observations, in the order the checks are made.
READY = ("trades",)
STATUS_TEXT = {
    "trades": "the contract is stored, it printed inside the entry window, and the exit window is covered",
    "exit_uncertain": "the entry can be priced, but no observation is stored near the planned exit (or on a later session of the holding period)",
    "stale": "the contract is stored but did not trade inside the entry window: no print recent enough to fill at",
    "contract": "the strike the strategy needs is outside the strikes stored for the session",
    "expiration": "option data is stored for the session, but not an expiration the strategy can use",
    "no_contract": "no contract with the required expiration was listed that day",
    "partial": "the provider holds only part of the session",
    "missing": "no option trade data is stored for the session",
}


def signal_readiness(store: MarketStore, cfg: BacktestConfig | dict, signals: list[dict], provider: str | None = None) -> dict:
    """For every qualifying signal: can THIS strategy be entered and exited on the stored option trade observations?

    A session that merely has some option data is not a session the strategy can trade. Each signal is taken through the
    checks the engine itself makes, with the engine's own functions and the run's own execution settings:

      the session      is it stored at all, and does the provider hold it whole
      the expiration   the one the leg would choose among those stored (never a later one for a 0DTE leg); where an
                       end-of-day chain is stored it also says whether such an expiration was listed at all
      the contract     `select_contracts` on the chain as it stood at each tick of the entry window: only contracts that
                       had printed, only prints younger than `max_trade_age_seconds`, the strike rule of each leg
      the entry        a fill exists within `max_fill_delay_minutes` of the signal, never from a print made before the
                       order could have used it
      the exit         a print of the selected contract near the planned exit, and stored observations for it on every
                       later session of the holding period

    The result is what the stored data supports, not a promise: a run can still take fewer trades (capital, position
    limits). It cannot take more. Nothing is downloaded and nothing is modelled."""
    from strata.engine.market import DayChain, UnderlyingSeries, day_num, expand_trade_bars, from_day_num
    from strata.engine.selection import _choose_expiration, dte_tolerance, select_contracts
    cfg = BacktestConfig.model_validate(cfg) if isinstance(cfg, dict) else cfg
    strat = cfg.strategies[0]
    sym = strat.symbol
    legs = [l for l in strat.legs if l.kind == "option"]
    ex = cfg.execution
    max_age = int(ex.max_trade_age_seconds)
    step = max(60, int(ex.signal_interval_minutes) * 60)
    wait = float(ex.max_fill_delay_minutes) * 60
    hold = hold_sessions(strat, cfg.start)
    ds = store.option_dataset(sym, "trade", provider)
    complete = bool(store.descriptor(ds).get("complete_chain")) if ds is not None else False  # every contract of each session, not a band of them
    counts = {k: 0 for k in STATUS_TEXT}
    out: dict = {"observation": "trade prints", "dataset": ds.id if ds else None, "signals": len(signals), "rows": [], "counts": counts, "ready": 0,
                 "settings": {"max_trade_age_seconds": max_age, "max_fill_delay_minutes": float(ex.max_fill_delay_minutes), "signal_interval_minutes": int(ex.signal_interval_minutes),
                              "latency_seconds": float(ex.latency_seconds), "hold_sessions": hold}}
    if not signals:
        return out
    today = _today()
    sources = []
    for src in S.option_bar_sources(store):
        try:
            if src.available()[0]:
                a, b = src.bars_window(today)
                sources.append((src.name, src.label, a, b))
        except Exception:  # noqa: BLE001
            continue

    def fetchable(d: date) -> dict | None:
        return next(({"source": n, "label": lab} for n, lab, a, b in sources if a <= d <= b), None)

    # which expirations were LISTED is only known where an end-of-day chain is stored
    listed: dict[str, list[int]] = {}
    eod = next((k for k in store.datasets(sym) if k.dtype == "options" and k.obs == "quote" and k.res == "EOD"), None)
    if eod is not None:
        try:
            listed = expiration_index(store, eod)
        except Exception:  # noqa: BLE001
            listed = {}
    leg_dicts = [l.model_dump(mode="json") for l in legs]

    def unlisted(d: date) -> bool:
        x = listed.get(d.isoformat())
        return x is not None and not all(_dte_ok(x, l, d) for l in leg_dicts)

    idx = store.option_bar_index(sym, dataset=ds) if ds is not None else pl.DataFrame()
    by_day: dict[date, pl.DataFrame] = {k[0]: v for k, v in idx.partition_by("date", as_dict=True).items()} if not idx.is_empty() else {}
    part = {k: (v.get("reason") or STATUS_TEXT["partial"]) for k, v in store.session_states(ds).items() if v.get("status") == "partial"} if ds is not None else {}
    live = sorted({s["date"] for s in signals if s["date"] in by_day and s["date"].isoformat() not in part})
    ser = None
    if live:
        b = store.read_best_bars(sym, live[0] - timedelta(days=7), live[-1])
        ser = UnderlyingSeries(b) if not b.is_empty() else None
    hhmm = lambda t: datetime.fromtimestamp(int(t), cal.ET).strftime("%H:%M")
    name = lambda k, r, e: f"{k:g}{r} {e.isoformat()}"

    def row(sig, status, detail, **kw):
        d = sig["date"]
        counts[status] += 1
        r = {"date": d.isoformat(), "time": hhmm(sig["t"]) if sig.get("t") else None, "status": status, "detail": detail,
             "downloadable": fetchable(d) if status in ("missing", "contract", "expiration") or kw.get("missing_later") else None, **kw}
        out["rows"].append(r)

    for sig in signals:
        d = sig["date"]
        iso = d.isoformat()
        day = by_day.get(d)
        if iso in part:
            row(sig, "partial", part[iso])
            continue
        if day is None:
            if unlisted(d):
                row(sig, "no_contract", STATUS_TEXT["no_contract"])
            else:
                row(sig, "missing", STATUS_TEXT["missing"])
            continue
        today_n = int(day_num(d))
        traded = day.filter(pl.col("status") == "ok")
        need_exp, miss, far = set(), None, False
        for leg in legs:
            rr = traded.filter(pl.col("right") == leg.right)
            exps = np.array(sorted({int(day_num(e)) for e in rr["expiration"].to_list()}), np.int64)
            e = _choose_expiration(exps, today_n, leg) if exps.size else None
            if e is not None and leg.expiration is None and leg.dte_mode != "exact" and leg.dte != 0 and abs(int(e) - today_n - leg.dte) > dte_tolerance(leg.dte):
                e, far = None, True  # the nearest STORED expiration is too far from the one asked for to be the nearest LISTED one
            if e is None:
                miss = leg
                break
            need_exp.add(from_day_num(int(e)))
        if miss is not None:
            exact = miss.dte_mode == "exact" or miss.dte == 0
            kind_ = "call" if miss.right == "C" else "put"
            if miss.expiration is not None and miss.expiration < d:
                row(sig, "no_contract", f"the {miss.expiration} {kind_} expiration asked for had already expired on this session")
            elif unlisted(d):
                row(sig, "no_contract", STATUS_TEXT["no_contract"])
            else:
                have_e = sorted(set(traded.filter(pl.col("right") == miss.right)["expiration"].to_list()))
                have = sorted({(e - d).days for e in have_e})
                if miss.expiration is not None:
                    row(sig, "expiration", f"the {miss.expiration} {kind_} expiration asked for is not stored for the session (stored: {', '.join(e.isoformat() for e in have_e[:8]) or 'none'})")
                elif far:
                    row(sig, "expiration", f"no {kind_} expiration within {dte_tolerance(miss.dte)} days of {miss.dte}DTE is stored for the session (stored: {', '.join(str(x) + 'DTE' for x in have[:8]) or 'none'})")
                else:
                    row(sig, "expiration", f"no {'exact ' if exact else ''}{miss.dte}DTE {kind_} expiration is stored for the session (stored: {', '.join(str(x) + 'DTE' for x in have[:8]) or 'none'})")
            continue
        o, c = cal.session_bounds(d)
        t_open, t_close = int(o.timestamp()), int(c.timestamp())
        sig_t = int(sig.get("t") or t_open)
        ready = sig_t + float(ex.latency_seconds)
        ticks, t = [], sig_t
        while t <= t_close:
            ticks.append(t)
            if t - sig_t >= wait:
                break
            t += step
        bars = store.read_option_bars(sym, d, dataset=ds)
        bars = bars.filter(pl.col("expiration").is_in(sorted(need_exp))) if bars is not None else pl.DataFrame()
        entry, why = None, ""
        if ser is not None and not bars.is_empty():
            last_tick = datetime.fromtimestamp(ticks[-1], timezone.utc)
            early = bars.filter(pl.col("ts") < last_tick)
            df, _stats = expand_trade_bars(early, ticks, ser, d) if not early.is_empty() else (pl.DataFrame(), {})
            if not df.is_empty():
                ch = DayChain(df, cfg.capital.risk_free_rate)
                ch.kind, ch.complete = "trades", complete
                for tk in ticks:
                    if tk < ready:
                        continue
                    si = ch.snapshot_at(tk)
                    if si < 0 or tk - int(ch.snap_ts[si]) > max_age:
                        continue
                    spot = ser.price_at(tk, d)
                    sel, why, _det = select_contracts(ch, si, list(strat.legs), float(spot), d, max_age)
                    if sel is not None:
                        entry = (tk, [(x.leg.right, float(x.strike), from_day_num(int(x.expiration)), int(tk - ch.quote_ts[x.row])) for x in sel if x.row >= 0])
                        break
        if entry is None:
            # the engine's reason is read against what is STORED for the session, so that a contract that exists in the store
            # but had not printed yet is not reported as missing from it
            price = float(sig.get("price") or 0.0)
            outside = None
            for leg in legs:
                if leg.strike.method != "atm_offset":
                    continue
                ks = sorted(set(traded.filter((pl.col("right") == leg.right) & pl.col("expiration").is_in(sorted(need_exp)))["strike"].to_list()))
                if ks and price:
                    half = (float(np.median(np.diff(ks))) if len(ks) > 1 else 1.0) / 2 + 1e-9
                    if not (ks[0] - half <= price <= ks[-1] + half):
                        outside = (ks[0], ks[-1], price)
            if outside is not None:
                row(sig, "contract", f"the underlying was at {outside[2]:.2f} at the signal; the strikes stored for the session run from {outside[0]:g} to {outside[1]:g}")
            elif "beyond the stored strike range" in why or "outside the stored strike range" in why:
                # the engine refused to pass the outermost stored strike off as the one asked for: the strike is missing, not stale
                row(sig, "contract", why.split(": ", 1)[-1])
            else:
                row(sig, "stale", (f"no usable trade print between {hhmm(ticks[0])} and {hhmm(ticks[-1])} ({why})" if why else
                                   f"none of the contracts of the required expiration had traded by {hhmm(ticks[-1])}"), window=[hhmm(ticks[0]), hhmm(ticks[-1])])
            continue
        tk, picked = entry
        # ---- the exit: a print of each selected contract near the planned exit, and the contract stored on later sessions
        xe = strat.exits
        planned = None
        if hold == 0:
            if xe.time_exit.enabled:
                planned = min(t_close, t_open + (cal.parse_hhmm(xe.time_exit.time) - 570) * 60)
            elif xe.max_hold.enabled and xe.max_hold.minutes:
                planned = min(t_close, tk + int(xe.max_hold.minutes) * 60)
        problems, missing_later, gap_max = [], [], 0
        for right, k, e, _age in picked:
            cb = bars.filter((pl.col("right") == right) & ((pl.col("strike") - k).abs() < 1e-6) & (pl.col("expiration") == e)).sort("ts")
            ends = (cb["ts"].dt.epoch("s").to_numpy().astype(np.int64) + 60) if not cb.is_empty() else np.array([], np.int64)
            after = ends[ends > tk]
            if planned is not None:
                if not ((ends > planned - max_age) & (ends <= t_close)).any():
                    problems.append(f"the {name(k, right, e)} has no print within {max_age // 60} min before the planned exit at {hhmm(planned)} or after it")
            elif hold == 0 and e > d and not after.size:
                problems.append(f"the {name(k, right, e)} did not trade again that session after the entry")
            horizon = planned if planned is not None else t_close
            seq = np.concatenate([[tk], after[after <= horizon], [horizon]])
            gap_max = max(gap_max, int(np.diff(seq).max() // 60) if seq.size > 1 else 0)
            if hold > 0:
                occ = S.occ_symbol(sym, e, right, k)
                for j in range(1, hold + 1):
                    s2 = cal.next_trading_day(d, j)
                    if s2 > e:
                        break
                    d2 = by_day.get(s2)
                    ok2 = d2 is not None and s2.isoformat() not in part and not d2.filter((pl.col("occ") == occ) & (pl.col("status") == "ok")).is_empty()
                    if not ok2 and s2 <= today:
                        missing_later.append(s2.isoformat())
        if missing_later:
            problems.append(f"no stored observations for the position on {', '.join(sorted(set(missing_later))[:4])}" + (f" and {len(set(missing_later)) - 4} more" if len(set(missing_later)) > 4 else ""))
        info = {"entry": {"time": hhmm(tk), "contracts": [name(k, r, e) for r, k, e, _ in picked], "print_age_seconds": max((a for *_x, a in picked), default=0)},
                "exit": {"planned": hhmm(planned) if planned is not None else ("expiration" if hold == 0 else f"up to {hold} session(s) later"), "longest_gap_minutes": gap_max}}
        if problems:
            row(sig, "exit_uncertain", "; ".join(problems), missing_later=sorted(set(missing_later)) or None, **info)
        else:
            row(sig, "trades", STATUS_TEXT["trades"], **info)
    out["ready"] = counts["trades"]
    out["entry_ready"] = counts["trades"] + counts["exit_uncertain"]
    out["fetchable"] = sum(1 for r in out["rows"] if r.get("downloadable"))
    return out


def unpriced_signals(store: MarketStore, cfg: BacktestConfig | dict, signals: list[dict], provider: str | None = None) -> list[dict]:
    """The signals the stored observations do not fully price: the only ones a provider needs to be asked about.

    A signal counts as priced when the engine's own selection, with the run's execution settings, finds the contract it
    asked for with a print inside the entry window and the exit window covered (`signal_readiness` status "trades").
    Everything else (a contract or an expiration that is not stored, a stale print, an uncovered exit, a session held in
    part or not at all) goes to the acquisition step, which decides what can actually be fetched."""
    rd = signal_readiness(store, cfg, signals, provider)
    ready = {r["date"] for r in rd["rows"] if r["status"] in READY}
    return [s for s in signals if s["date"].isoformat() not in ready]


def trade_coverage(store: MarketStore, req: dict, signals: list[dict], provider: str | None = None, cfg: BacktestConfig | dict | None = None) -> dict:
    """Coverage of stored option trade bars for each signal.

    With the run's configuration (`cfg`) this is `signal_readiness`: the engine's own selection and execution settings
    applied to each signal's entry window and exit. Without one it is the older, coarser check: does the index hold, for
    the entry session, a contract of the right expiration and type with a strike around the price at the signal, and did it trade?"""
    if cfg is not None:
        rd = signal_readiness(store, cfg, signals, provider)
        bad: dict[str, list[date]] = {}
        for r in rd["rows"]:
            if r["status"] not in ("trades", "exit_uncertain"):
                bad.setdefault(STATUS_TEXT[r["status"]], []).append(date.fromisoformat(r["date"]))
        ds0 = store.option_dataset(req["symbol"], "trade", provider)
        return {"observation": "trade prints", "dataset": rd["dataset"], "signals": len(signals), "executable": rd["entry_ready"], "fully_covered": rd["ready"],
                "exit_uncertain": rd["counts"]["exit_uncertain"], "fetchable": rd.get("fetchable", 0), "readiness": rd["counts"],
                "issues": ([{"reason": "no option trade data is stored for " + req["symbol"], "signals": len(signals), "kind": "no_dataset", "sample": [s_["date"].isoformat() for s_ in signals[-6:]]}] if ds0 is None and signals else
                           [{"reason": k, "signals": len(v), "kind": "trades", "first": min(v).isoformat(), "last": max(v).isoformat(), "sample": [x.isoformat() for x in sorted(v)[-6:]]} for k, v in bad.items()]),
                # the sessions a wider strike range could still price: the pipeline fetches that range for them before it reports
                "outside_stored_strikes": [r["date"] for r in rd["rows"] if r["status"] == "contract" and r.get("downloadable")],
                "contract_sessions_stored": store.option_bar_index(req["symbol"], dataset=ds0).height if ds0 is not None else 0}
    sym, opt = req["symbol"], req["options"]
    ds = store.option_dataset(sym, "trade", provider)
    out = {"observation": "trade prints", "dataset": ds.id if ds else None, "signals": len(signals), "executable": 0, "issues": []}
    if not signals:
        return out
    if ds is None:
        out["issues"].append({"reason": f"no option trade data is stored for {sym}", "signals": len(signals), "kind": "no_dataset", "sample": [s["date"].isoformat() for s in signals[-6:]]})
        return out
    idx = store.option_bar_index(sym, dataset=ds)
    by_day: dict[date, pl.DataFrame] = {k[0]: v for k, v in idx.partition_by("date", as_dict=True).items()} if not idx.is_empty() else {}
    step = S.STRIKE_STEP.get(sym, 1.0)
    bad: dict[str, list[date]] = {}
    ok = 0
    part = {k for k, v in store.session_states(ds).items() if v.get("status") == "partial"}
    for s in signals:
        d = s["date"]
        day = by_day.get(d)
        if day is None:
            bad.setdefault("no option trade data stored for the session", []).append(d)
            continue
        if d.isoformat() in part:
            bad.setdefault("the provider's option trade history for the session is incomplete (it starts late, stops early or is absent)", []).append(d)
            continue
        targets = leg_targets(opt["legs"], float(s["price"]), None, step)
        reason = None
        for leg in opt["legs"]:
            rows = day.filter(pl.col("right") == leg["right"])
            exp = _pick_expiration(rows["expiration"].unique().to_list(), d, leg)
            if exp is None:
                reason = f"no {leg['dte']}DTE {'call' if leg['right'] == 'C' else 'put'} contract is stored for the session"
                break
            rows = rows.filter(pl.col("expiration") == exp)
            k0, n = targets[leg["id"]]
            near = rows.filter((pl.col("strike") - k0).abs() <= max(n, 1) * step + 1e-6)
            if near.is_empty():
                reason = "the strikes around the price at the signal are not stored"
                break
            if near.filter(pl.col("status") == "ok").is_empty():
                reason = "the required contracts did not trade on the session"
                break
        if reason:
            bad.setdefault(reason, []).append(d)
        else:
            ok += 1
    out["executable"] = ok
    out["issues"] = [{"reason": k, "signals": len(v), "kind": "trades", "first": min(v).isoformat(), "last": max(v).isoformat(), "sample": [x.isoformat() for x in sorted(v)[-6:]]} for k, v in bad.items()]
    out["contract_sessions_stored"] = idx.height
    return out
