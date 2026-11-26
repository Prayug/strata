"""The original use case, tested on the observations that are actually stored.

    Buy at-the-money SPY 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in SPY,
    only on days when SPY fell at least 3% the previous trading day.

`run` does what a careful person would do by hand, and keeps every step apart:

  1  the qualifying sessions are found twice, by arithmetic on the daily closes here and by
     the engine's own signal scan, and the two lists must be identical
  2  the same is done for declines of 1%, 2%, 3% and 4%
  3  for each session: did a 0DTE contract exist (known only where an end-of-day chain is
     stored), and which intraday option observations are stored for it
  4  the strategy is run by the engine on every session that has them
  5  every one of those sessions is priced a second time by the independent reference
     simulator, from the raw stored observations, and the two results are compared
  6  the configuration written by hand is compared with the one the interpreter reads from
     the sentence, setting by setting
  7  coverage is reported against the whole history AND against the period a free provider
     reaches, so that neither a hopeless denominator nor a flattering one stands alone

Nothing is downloaded and nothing is modelled. A session without stored option prices is
listed as missing with the sources that reach its date; it is never priced another way.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

from strata.core import calendar as cal
from strata.data.store import MarketStore

SENTENCE = "Buy ATM {sym} 0DTE calls at 6:30 AM Pacific, targeting a 1% increase in {sym}, only on days when {sym} fell at least {pct:g}% the previous trading day."
THRESHOLDS = (1.0, 2.0, 3.0, 4.0)
# first date each route reaches. Alpaca's is the first session that returned a bar when every session of January and February
# 2024 was asked with a paper key (2026-11-25); the other two are what the provider publishes and have not been run from here.
REACH = [("alpaca", "Alpaca free account (trade prints)", date(2024, 1, 18), None), ("databento", "Databento free credit (1-minute bid/ask)", date(2013, 4, 1), None),
         ("optionsdx", "OptionsDX files, $3 to $50 a year (bid/ask snapshots)", date(2010, 1, 1), date(2023, 12, 31))]


def manual_config(sym: str, pct: float, time_exit: str = "15:55") -> dict:
    """The strategy written out by hand, setting by setting."""
    return {"execution": {"signal_interval_minutes": 1},
            "strategies": [{"name": f"0DTE call after a {pct:g}% down day (by hand)", "symbol": sym,
                            # "0DTE" means the contract expiring that session and no other: on a day without one, nothing is traded in its place
                            "legs": [{"id": "lc", "side": "buy", "right": "C", "dte": 0, "dte_mode": "exact", "strike": {"method": "atm_offset", "value": 0.0}}],
                            "entry": {"conditions": {"kind": "group", "op": "and", "children": [
                                {"kind": "cond", "type": "entry_times", "params": {"times": "open"}},
                                {"kind": "cond", "type": "expr", "params": {"expr": f"previous_session.return_pct <= -{pct:g}"}}]}},
                            "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": time_exit, "dte_lte": None},
                                      "underlying": {"enabled": True, "move_for_pct": 1.0}}}]}


def ai_config(store: MarketStore, sym: str, pct: float) -> dict:
    """The same strategy as the built-in interpreter reads it from the sentence (deterministic: no model, no network)."""
    from strata.ai.agent import Agent
    from strata.ai.draft import build_spec, compile_draft
    text = SENTENCE.format(sym=sym, pct=pct)
    r = Agent(None, store, llm_factory=lambda: None)._rules(text, None, sorted(store.stored_symbols()), None, sym)
    out = {"sentence": text, "action": r.get("action"), "config": None, "assumptions": [], "open": []}
    d = r.get("draft")
    if r.get("action") != "new" or d is None:
        out["open"] = [f"the sentence was not read as a new strategy ({r.get('action')})"]
        return out
    out["open"] = list(d.unsupported or []) + [q.get("question") or str(q) for q in d.questions if q.get("answered") is None]
    c = compile_draft(d, store)
    if c["errors"] or not c.get("strategy"):
        out["open"] += c["errors"]
        return out
    spec = build_spec(d, c, None, {}, d.title, "built-in rules")
    out["config"] = {k: v for k, v in spec["backtest"].items()}
    out["assumptions"] = [{"field": a["field"], "value": a.get("value"), "why": a.get("why") or a.get("text")} for a in c["assumptions"]]
    return out


def _norm(strategy: dict) -> dict:
    """Everything that affects behaviour, nothing that does not (names, editor ids, the parameters of rules that are switched off)."""
    from strata.engine.models import StrategyConfig
    c = StrategyConfig.model_validate(strategy).model_dump(mode="json")
    c.pop("name", None)

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("id", "label")}
        return [strip(x) for x in node] if isinstance(node, list) else node
    c["entry"]["conditions"] = strip(c["entry"]["conditions"])
    c["exits"] = {k: ({"enabled": False} if isinstance(v, dict) and v.get("enabled") is False else v) for k, v in c["exits"].items()}
    return c


def config_difference(a: dict, b: dict) -> list[str]:
    x, y = _norm(a["strategies"][0]), _norm(b["strategies"][0])
    diff = [k for k in x if json.dumps(x[k], sort_keys=True) != json.dumps(y.get(k), sort_keys=True)]
    ea, eb = a.get("execution") or {}, b.get("execution") or {}
    return diff + [f"execution.{k}" for k in sorted(set(ea) | set(eb)) if ea.get(k) != eb.get(k)]


def down_days(store: MarketStore, sym: str, pct: float) -> list[dict]:
    """Sessions that follow a close-to-close fall of at least `pct` percent: plain arithmetic on the stored daily closes."""
    d = store.read_daily(sym)
    days, closes = d["date"].to_list(), d["close"].to_list()
    out = []
    for i in range(2, len(days)):
        if closes[i - 2] and closes[i - 1] is not None:
            r = (closes[i - 1] / closes[i - 2] - 1) * 100
            if r <= -pct:
                out.append({"date": days[i], "fell_on": days[i - 1], "fall_pct": round(r, 3)})
    return out


def _reach_note() -> str:
    from strata.data.providers import market as MK
    m = MK.measured("alpaca")
    if m:
        return (f"Alpaca's reach was measured with the key stored here on {str(m['run_at'])[:10]} ({m['with_observations']} of {m['contract_sessions']} expired contract-sessions returned trade bars; first session with any: {REACH[0][2]}). "
                "The periods of the other providers are what they publish: those routes have not been run from this installation.")
    return "The periods a provider reaches are what that provider publishes. None of the key-gated routes has been run from this installation."


def _runs(days: list[date], gap: int = 12) -> list[tuple[date, date]]:
    """Qualifying sessions grouped into stretches (a gap of more than `gap` sessions starts a new one), so that the engine is
    not walked through hundreds of sessions on which nothing can trade. A 0DTE position never outlives its session."""
    out: list[list[date]] = []
    for d in sorted(set(days)):
        if out and len(cal.trading_days(out[-1][-1], d)) - 1 <= gap:
            out[-1].append(d)
        else:
            out.append([d])
    return [(r[0], r[-1]) for r in out]


_default_trade: dict[str, str | None] = {}


def rank(provider: str, observation: str) -> tuple:
    """Order in which a session's records are preferred: bid/ask quotes before trade prints, then the provider name."""
    return (0 if observation == "bid/ask quotes" else 1, 0 if provider == _default_trade.get("provider") else 1, provider)


def _stored(store: MarketStore, sym: str) -> dict:
    from strata.data.readiness import stored_option_sessions
    return stored_option_sessions(store, sym)


def _ref_spec(cfg: dict, observation: str):
    from strata.audit import reference as R
    s = cfg["strategies"][0]
    ex = s["exits"]
    return R.Spec(legs=[R.Leg(l["side"], l["right"], int(l.get("dte", 0)), float((l.get("strike") or {}).get("value") or 0.0)) for l in s["legs"]], entry="open",
                  time_exit=(ex.get("time_exit") or {}).get("time") if (ex.get("time_exit") or {}).get("enabled") else None, underlying_for_pct=(ex.get("underlying") or {}).get("move_for_pct"),
                  observation=observation, signal_interval_min=int((cfg.get("execution") or {}).get("signal_interval_minutes", 5)), cash_settled=False)


def _brief(t: dict) -> dict:
    return {"entry_time": str(t["entry_time"])[:19], "exit_time": str(t.get("exit_time"))[:19] if t.get("exit_time") else None, "exit_reason": t["exit_reason"], "pnl": round(float(t["pnl"]), 2),
            "fees": round(float(t.get("commissions", t.get("fees", 0.0))), 2), "entry_spot": round(float(t["entry_spot"]), 2),
            "legs": [{"right": l["right"], "strike": l["strike"], "expiration": l["expiration"], "entry_price": l["entry_price"], "exit_price": l.get("exit_price"), "price_source": l.get("price_source")} for l in t["legs"]]}


def acquire(store: MarketStore | None = None, symbol: str = "SPY", thresholds=THRESHOLDS, dtes=(0, 1, 7), source: str = "alpaca", progress=None) -> dict:
    """Download the option trade bars THIS strategy needs from a keyed free source, and nothing else.

    A bulk download takes strikes around the previous close. This strategy trades after a large fall, and the next session
    often opens far from that close (SPY, 2026-11-23: previous close 689.81, open 724.33): the at-the-money strike at the open is
    then outside any band around the previous close. So for every qualifying session inside the source's history the contracts
    are chosen around the underlying's price AT THE SIGNAL, the way the research pipeline does it. Sessions are found from the
    underlying alone, at the lowest threshold asked for (the others are subsets); 0DTE plus the next-day and one-week variants."""
    from strata.data import planner as P
    from strata.data.providers import sources as S
    from strata.engine.scan import scan_signals
    store = store or MarketStore()
    src = next((x for x in S.option_bar_sources(store) if x.name == source), None)
    if src is None:
        raise LookupError(f"no option trade-bar source is called {source!r}")
    ok, why = src.available()
    out: dict = {"source": source, "symbol": symbol, "status": "blocked" if not ok else "ran", "threshold_pct": min(thresholds), "variants": []}
    if not ok:
        out["reason"] = f"BLOCKED: {src.label}: {why}"
        return out
    first, last = src.bars_window()
    for i, dte in enumerate(dtes):
        cfg = manual_config(symbol, min(thresholds))
        leg = cfg["strategies"][0]["legs"][0]
        leg["dte"] = int(dte)
        if dte:
            leg.pop("dte_mode", None)  # "about a day" / "about a week": the nearest listed expiration, as the engine picks it
        full = {**cfg, "start": first.isoformat(), "end": last.isoformat()}
        signals = scan_signals(full, store)["signals"]
        req = P.requirements("options", full, "trades")
        sub = (lambda f, m, i=i: progress((i + f) / len(dtes), f"{dte}DTE: {m}")) if progress else None
        r = P.acquire_trades(store, req, signals, source, sub, None, True)
        res = r.get("result") or {}
        out["variants"].append({"dte": int(dte), "signals_in_window": r.get("signals_in_window"), "contract_sessions": r.get("contract_sessions"), "already_stored": res.get("pairs_cached"), "downloaded_with_trades": res.get("pairs_ok"),
                                "downloaded_without_trades": res.get("pairs_empty"), "failed": res.get("pairs_failed"), "bars": res.get("bars"), "requests": res.get("requests"), "notes": (r.get("notes") or [])[:6],
                                "partial_sessions": res.get("partial_sessions") or {}})
    out["window"] = [first.isoformat(), last.isoformat()]
    return out


def run(store: MarketStore | None = None, symbol: str = "SPY", thresholds=THRESHOLDS, progress=None, today: date | None = None, save: bool = True) -> dict:
    from strata.audit import reference as R
    from strata.engine.engine import DataUnavailable, run_backtest
    from strata.engine.scan import scan_signals
    store = store or MarketStore()
    p = progress or (lambda f, m: None)
    today = today or datetime.now(cal.ET).date()
    daily = store.read_daily(symbol)
    if daily.is_empty():
        raise LookupError(f"no daily prices are stored for {symbol}")
    first, last = daily["date"][0], daily["date"][-1]
    st = _stored(store, symbol)
    dflt = store.option_dataset(symbol, "trade")
    _default_trade["provider"] = dflt.provider if dflt else None  # among trade-print providers, the dataset the engine uses when none is named
    rep: dict = {"symbol": symbol, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sentence": SENTENCE.format(sym=symbol, pct=3.0), "history": [first.isoformat(), last.isoformat()],
                 "stored_option_data": {"sessions_with_intraday_option_data": len(st["sessions"]), "end_of_day_chains": st["eod_dataset"], "end_of_day_sessions": len(st["eod_expirations"]),
                                        "by_provider": {}}, "thresholds": [], "notes": []}
    for d, items in st["sessions"].items():
        for it in items:
            b = rep["stored_option_data"]["by_provider"].setdefault(it["provider"], {"observation": it["observation"], "resolution": it["resolution"], "sessions": 0, "partial_sessions": 0, "first": d.isoformat(), "last": d.isoformat()})
            b["sessions"] += 1
            b["partial_sessions"] += it["status"] != "complete"
            b["first"], b["last"] = min(b["first"], d.isoformat()), max(b["last"], d.isoformat())
    for ti, pct in enumerate(thresholds):
        p(ti / len(thresholds), f"sessions after a fall of {pct:g}% or more")
        man = manual_config(symbol, pct)
        qual = down_days(store, symbol, pct)
        scan = scan_signals({**man, "start": first.isoformat(), "end": last.isoformat()}, store)
        eng_days = sorted({s["date"] for s in scan["signals"]})
        mine = [q["date"] for q in qual]
        row: dict = {"threshold_pct": pct, "qualifying": len(mine), "engine_scan": len(eng_days), "lists_identical": mine == eng_days,
                     "only_by_arithmetic": [d.isoformat() for d in sorted(set(mine) - set(eng_days))][:12], "only_by_engine": [d.isoformat() for d in sorted(set(eng_days) - set(mine))][:12],
                     "not_evaluable": {k: len(v) for k, v in (scan.get("excluded") or {}).items()}}
        # ---- the sentence, read by the interpreter, against the configuration written by hand
        ai = ai_config(store, symbol, pct)
        if ai["config"] is None:
            row["interpreter"] = {"read": False, "open": ai["open"], "sentence": ai["sentence"]}
        else:
            diff = config_difference(man, ai["config"])
            row["interpreter"] = {"read": True, "sentence": ai["sentence"], "differences": diff, "identical": not diff, "assumed": ai["assumptions"], "open": ai["open"]}
        # ---- each session: contract existence, stored observations, engine, reference
        sessions = []
        engine_by: dict[str, dict[str, dict]] = {}  # provider -> session -> trade
        skips_by: dict[str, dict[str, str]] = {}
        groups: dict[tuple, list[date]] = {}
        for d in mine:
            for it in st["sessions"].get(d, []):
                if it["status"] == "complete":
                    groups.setdefault((it["provider"], it["observation"]), []).append(d)
        ai_by: dict[str, dict[str, dict]] = {}
        for (prov, obs), ds_ in sorted(groups.items(), key=lambda kv: rank(kv[0][0], kv[0][1])):
            p(ti / len(thresholds), f"fall of {pct:g}%: engine on {len(ds_)} session(s) of {prov}")
            base = {**man, "pricing": {"mode": "trades" if obs == "trade prints" else "historical"}, "data": {"options_provider": prov}}
            for a_, b_ in _runs(ds_):
                cfg = {**base, "start": a_.isoformat(), "end": b_.isoformat()}
                try:
                    res = run_backtest(cfg, store=store, record_paths=False)
                except DataUnavailable as e:
                    for d in ds_:
                        if a_ <= d <= b_:
                            skips_by.setdefault(prov, {})[d.isoformat()] = str(e)[:200]
                    continue
                for t in res.trades:
                    engine_by.setdefault(prov, {})[t["entry_date"]] = {**t, "_provider": prov}
                for x in (res.diagnostics or {}).get("skipped_signals", []):
                    skips_by.setdefault(prov, {})[cal.to_et(int(x["signal_t"]) * 1_000_000_000).date().isoformat()] = f"{x['kind']}: {x['reason']}"
                for why, days_ in ((res.diagnostics or {}).get("excluded_sessions") or {}).items():
                    for iso_ in days_:
                        skips_by.setdefault(prov, {}).setdefault(iso_, f"session not simulated: {why}")
                if ai["config"] is not None:
                    try:
                        r2 = run_backtest({**ai["config"], "start": cfg["start"], "end": cfg["end"], "pricing": cfg["pricing"], "data": cfg["data"]}, store=store, record_paths=False)
                        for t in r2.trades:
                            ai_by.setdefault(prov, {})[t["entry_date"]] = t
                    except DataUnavailable:
                        pass
        counts = {"priced_by_engine": 0, "priced_by_reference_only": 0, "agree": 0, "disagree": 0, "no_contract": 0, "missing": 0, "signal_not_filled": 0, "partial_at_provider": 0, "pnl_engine": 0.0, "pnl_reference": 0.0,
                  "two_providers": 0, "two_providers_same_trade": 0, "wins": 0, "pnl_worst_print": 0.0, "pnl_best_print": 0.0, "with_print_range": 0}
        for q in qual:
            d = q["date"]
            iso = d.isoformat()
            exps = st["eod_expirations"].get(iso)
            exists = None if exps is None else (0 in exps)
            items = st["sessions"].get(d, [])
            s_row = {"date": iso, "weekday": d.strftime("%a"), "fell_on": q["fell_on"].isoformat(), "fall_pct": q["fall_pct"], "zero_dte_listed": exists, "stored": [{k: it[k] for k in ("provider", "observation", "resolution", "status", "window")} for it in items],
                     "engine": None, "reference": None, "agree": None, "status": None, "reach": [n for _id, n, a, b in REACH if d >= a and (b is None or d <= b)]}
            # one provider is the session's primary record: bid/ask before trade prints, then the dataset the engine uses by default
            items = sorted(items, key=lambda it: rank(it["provider"], it["observation"]))
            priced = [it["provider"] for it in items if iso in engine_by.get(it["provider"], {})]
            primary = priced[0] if priced else None
            engine_trades = engine_by.get(primary, {}) if primary else {}
            ai_trades = ai_by.get(primary, {}) if primary else {}
            if primary:
                s_row["engine"] = {**_brief(engine_trades[iso]), "provider": primary}
                try:
                    rng = minute_range(store, symbol, primary, s_row["engine"])
                except Exception:  # noqa: BLE001
                    rng = None
                if rng:
                    s_row["engine"]["print_range"] = rng
                # the same session priced from a second provider's records: an independent check of the trade itself
                others = []
                for prov in priced[1:]:
                    o = _brief(engine_by[prov][iso])
                    a = s_row["engine"]
                    same = (o["entry_time"] == a["entry_time"] and o["exit_time"] == a["exit_time"] and o["exit_reason"] == a["exit_reason"] and abs(o["pnl"] - a["pnl"]) <= 0.01)
                    others.append({**o, "provider": prov, "same_trade": same})
                    counts["two_providers"] += 1
                    counts["two_providers_same_trade"] += same
                if others:
                    s_row["engine_other_providers"] = others
            else:
                note = next((skips_by[it["provider"]][iso] for it in items if iso in skips_by.get(it["provider"], {})), None)
                if note:
                    s_row["engine_note"] = note
            # the independent reference, on the same provider's raw observations (or, where the engine ran none, on every stored
            # observation of the session, a partially observed one included)
            for it in ([x for x in items if x["provider"] == primary] if primary else items):
                obs = "trade" if it["observation"] == "trade prints" else "quote"
                part = it["status"] != "complete"
                try:
                    sess = R.load_session(store, symbol, d, obs, it["provider"], underlying_provider=it["provider"] if part and obs == "quote" else None, partial=part)
                except Exception as e:  # noqa: BLE001
                    s_row["reference_note"] = f"the raw observations could not be read: {type(e).__name__}: {e}"[:200]
                    continue
                if sess is None:
                    continue
                out = R.simulate_session(sess, _ref_spec(man, obs))
                if out["trade"]:
                    t = out["trade"]
                    s_row["reference"] = {**_brief(t), "provider": it["provider"], "observation": it["observation"], "session_observed": ("09:30 to " + R._et(sess.close_t)[11:16] + " ET" if obs == "quote" else (it.get("reason") or "part of the session")) if part else "whole session",
                                          "outcome_observed": not t.get("open"), "partial": part}
                    break
                s_row["reference_note"] = f"{it['provider']}: {out['skipped']}"
            e_, r_ = s_row["engine"], s_row["reference"]
            if e_ and r_:
                same = (e_["entry_time"] == r_["entry_time"] and e_["exit_time"] == r_["exit_time"] and e_["exit_reason"] == r_["exit_reason"] and abs(e_["pnl"] - r_["pnl"]) <= 0.01
                        and [(l["strike"], round(l["entry_price"], 4), round(l["exit_price"] or 0, 4)) for l in e_["legs"]] == [(l["strike"], round(l["entry_price"], 4), round(l["exit_price"] or 0, 4)) for l in r_["legs"]])
                s_row["agree"] = same
                counts["agree" if same else "disagree"] += 1
            if e_:
                s_row["status"] = "priced"
                counts["priced_by_engine"] += 1
                counts["pnl_engine"] += e_["pnl"]
                counts["wins"] = counts.get("wins", 0) + (e_["pnl"] > 0)
                if e_.get("print_range"):
                    counts["pnl_worst_print"] = round(counts.get("pnl_worst_print", 0.0) + e_["print_range"]["pnl_worst"], 2)
                    counts["pnl_best_print"] = round(counts.get("pnl_best_print", 0.0) + e_["print_range"]["pnl_best"], 2)
                    counts["with_print_range"] = counts.get("with_print_range", 0) + 1
                else:
                    counts["pnl_worst_print"] = round(counts.get("pnl_worst_print", 0.0) + e_["pnl"], 2)
                    counts["pnl_best_print"] = round(counts.get("pnl_best_print", 0.0) + e_["pnl"], 2)
                a = ai_trades.get(iso)
                s_row["same_from_sentence"] = None if a is None else (abs(a["pnl"] - engine_trades[iso]["pnl"]) < 0.005 and a["entry_time"] == engine_trades[iso]["entry_time"] and a["exit_time"] == engine_trades[iso]["exit_time"])
            elif r_ and r_["outcome_observed"] and not (r_.get("partial") and r_["observation"] == "trade prints"):
                s_row["status"] = "priced (reference only)"
                counts["priced_by_reference_only"] += 1
                counts["pnl_reference"] += r_["pnl"]
            elif any(it["status"] != "complete" and it["observation"] == "trade prints" for it in items) and not any(it["status"] == "complete" for it in items):
                # the provider holds the session only in part (its history starts late, stops early or is absent): an entry at the
                # open cannot be established from it, by the engine or by the reference
                s_row["status"] = "provider's history incomplete for the session"
                s_row["provider_note"] = next((it.get("reason") for it in items if it.get("reason")), None)
                s_row["reference"] = None
                counts["partial_at_provider"] += 1
            elif exists is False:
                s_row["status"] = "no 0DTE contract was listed"
                counts["no_contract"] += 1
            elif items:
                s_row["status"] = "signal not filled"
                counts["signal_not_filled"] += 1
            else:
                s_row["status"] = "no intraday option prices stored"
                counts["missing"] += 1
            sessions.append(s_row)
        row["counts"] = {**counts, "pnl_engine": round(counts["pnl_engine"], 2), "pnl_reference": round(counts["pnl_reference"], 2)}
        row["periods"] = [_period(sessions, "All stored history", first, today)] + [_period(sessions, f"Since {a.isoformat()}: reachable by {n}", a, b or today) for _id, n, a, b in REACH]
        row["sessions"] = sessions if len(sessions) <= 400 else [s for s in sessions if s["status"].startswith("priced") or s["date"] >= "2013-04-01"][-400:]
        row["sessions_listed"] = len(row["sessions"])
        row["from_sentence_identical"] = all(s.get("same_from_sentence") is not False for s in sessions)
        rep["thresholds"].append(row)
    rep["notes"] = ["A session is \"priced\" only when its entry and exit were both taken from stored option observations. Nothing is modelled.",
                    "\"Reference only\" means the independent simulator priced the session from raw observations that the engine does not run: a session a vendor sample covers only in part.",
                    "Whether a 0DTE contract was listed is known only for sessions with a stored end-of-day chain; elsewhere it is not established.",
                    "A session two providers hold is priced from each of them separately, and the two trades are compared: neither is used to fill the other.",
                    _reach_note()]
    if save:
        from strata.config import get_settings
        d = get_settings().data_dir / "reports"
        d.mkdir(parents=True, exist_ok=True)
        (d / "usecase.json").write_text(json.dumps(rep, indent=1, default=str))
        (d / "usecase.md").write_text(render(rep))
    p(1.0, "done")
    return rep


def minute_range(store: MarketStore, sym: str, provider: str, trade: dict) -> dict | None:
    """What a trade priced on trade bars could have been filled at: the range of prints in the minute each fill was taken from.

    Trade bars carry no bid and no ask, so the spread is unknown and none is invented. What IS observed is every price the
    contract traded at in the minute the engine used (it fills at that minute's last print once the minute has ended): the
    opening minute of a 0DTE contract can span a third of its premium. The result is stated with that range: bought at the
    minute's high and sold at the exit minute's low, and the reverse."""
    import polars as pl

    from strata.data.providers import sources as S
    if not trade.get("legs") or len(trade["legs"]) != 1 or not trade.get("exit_time"):
        return None
    leg = trade["legs"][0]
    if leg.get("price_source") != "trade_print" or leg.get("exit_price") is None:
        return None
    d = date.fromisoformat(trade["entry_time"][:10])
    bars = store.read_option_bars(sym, d, provider=provider)
    if bars is None or bars.is_empty():
        return None
    occ = S.occ_symbol(sym, date.fromisoformat(str(leg["expiration"])[:10]), leg["right"], float(leg["strike"]))
    b = bars.filter(pl.col("occ") == occ).with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.strftime("%Y-%m-%dT%H:%M").alias("m"))

    def bar_before(when: str):
        t = datetime.fromisoformat(when[:16])
        hit = b.filter(pl.col("m") < t.strftime("%Y-%m-%dT%H:%M")).sort("ts")
        return hit.row(-1, named=True) if hit.height else None
    e, x = bar_before(trade["entry_time"]), bar_before(trade["exit_time"])
    if e is None:
        return None
    mult, fees = 100.0, float(trade.get("fees") or 0.0)
    settled = trade.get("exit_reason") in ("expired_worthless", "expired", "settled", "exercise", "assignment")
    xl, xh = (float(leg["exit_price"]), float(leg["exit_price"])) if settled or x is None else (float(x["low"]), float(x["high"]))
    return {"entry_minute": e["m"][11:16], "entry_low": float(e["low"]), "entry_high": float(e["high"]), "entry_prints": int(e["trade_count"]) if e.get("trade_count") is not None else None, "entry_volume": int(e["volume"] or 0),
            "exit_minute": None if settled or x is None else x["m"][11:16], "exit_low": xl, "exit_high": xh,
            "pnl_worst": round((xl - float(e["high"])) * mult - fees, 2), "pnl_best": round((xh - float(e["low"])) * mult - fees, 2)}


def _period(sessions: list[dict], label: str, a: date, b: date) -> dict:
    inside = [s for s in sessions if a.isoformat() <= s["date"] <= b.isoformat()]
    priced = [s for s in inside if s["status"].startswith("priced")]
    known_absent = sum(1 for s in inside if s["zero_dte_listed"] is False)
    testable = len(inside) - known_absent
    return {"label": label, "from": a.isoformat(), "to": b.isoformat(), "qualifying": len(inside), "no_0dte_contract": known_absent, "contract_not_established": sum(1 for s in inside if s["zero_dte_listed"] is None),
            "testable_at_most": testable, "priced": len(priced), "priced_pct_of_qualifying": round(len(priced) / len(inside) * 100, 1) if inside else None, "priced_pct_of_testable": round(len(priced) / testable * 100, 1) if testable else None,
            "missing": sum(1 for s in inside if s["status"] == "no intraday option prices stored")}


def saved() -> dict | None:
    from strata.config import get_settings
    f = get_settings().data_dir / "reports" / "usecase.json"
    return json.loads(f.read_text()) if f.exists() else None


def render(rep: dict) -> str:
    L = [f"# The original use case on stored observations ({rep['symbol']}), {rep['run_at']}", "", f"> {rep['sentence']}", "", f"Daily history: {rep['history'][0]} to {rep['history'][1]}.", ""]
    so = rep["stored_option_data"]
    L.append("Intraday option observations stored: " + ("; ".join(f"{k}: {v['sessions']} session(s) of {v['observation']} at {v['resolution']} ({v['first']} to {v['last']}" + (f", {v['partial_sessions']} only partly observed" if v["partial_sessions"] else "") + ")"
                                                                  for k, v in so["by_provider"].items()) or "none") + ".")
    L += ["", "Net P&L is for one contract per session, fees included. \"Worst to best print\" restates it with every fill taken at the worst, then the best, price the contract traded at in the minute the fill came from: trade bars have no bid or ask, so this range is the only observed measure of how far a real fill could have been from the print used.",
          "", "| Fall of at least | Qualifying sessions | Arithmetic = engine scan | Priced by the engine | Net P&L of those | Worst to best print | Winning sessions | Priced by the reference only | Engine and reference agree | Same trade from a second provider | Signal not filled | Provider's history incomplete | No 0DTE listed | No option prices stored | Sentence = hand-written |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rep["thresholds"]:
        c = r["counts"]
        L.append(f"| {r['threshold_pct']:g}% | {r['qualifying']} | {'yes' if r['lists_identical'] else 'NO'} | {c['priced_by_engine']} | {c['pnl_engine']:+,.2f} | "
                 + (f"{c['pnl_worst_print']:+,.2f} to {c['pnl_best_print']:+,.2f}" if c.get("with_print_range") else "") + f" | {c.get('wins', 0)} of {c['priced_by_engine']} | {c['priced_by_reference_only']} | {c['agree']} of {c['agree'] + c['disagree']} | "
                 f"{str(c.get('two_providers_same_trade', 0)) + ' of ' + str(c.get('two_providers', 0)) if c.get('two_providers') else 'none held twice'} | {c['signal_not_filled']} | {c.get('partial_at_provider', 0)} | {c['no_contract']} | {c['missing']} | "
                 f"{'yes' if r['interpreter'].get('identical') else 'NO: ' + ', '.join(r['interpreter'].get('differences') or r['interpreter'].get('open') or [])} |")
    for r in rep["thresholds"]:
        L += ["", f"## After a fall of {r['threshold_pct']:g}% or more", "", "| Period | Qualifying | No 0DTE listed | Contract not established | Priced | Priced, % of qualifying | Missing |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for p_ in r["periods"]:
            L.append(f"| {p_['label']} | {p_['qualifying']} | {p_['no_0dte_contract']} | {p_['contract_not_established']} | {p_['priced']} | {'' if p_['priced_pct_of_qualifying'] is None else str(p_['priced_pct_of_qualifying']) + '%'} | {p_['missing']} |")
        priced = [s for s in r["sessions"] if s["status"].startswith("priced")]
        if priced:
            L += ["", "| Session | Fall the day before | Source | Entry | Exit | Reason | Net P&L | Worst to best print | Second computation |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for s in priced:
                x = s["engine"] or s["reference"]
                leg = x["legs"][0]
                src = f"{x['provider']} ({'trade prints' if leg.get('price_source') == 'trade_print' or (s['reference'] or {}).get('observation') == 'trade prints' else 'bid/ask'})"
                second = ("engine and reference agree" if s["agree"] else "reference simulator only (" + (s["reference"] or {}).get("session_observed", "") + ")" if s["engine"] is None else "DIFFERENT" if s["agree"] is False else "")
                for o in s.get("engine_other_providers") or []:
                    ol = o["legs"][0]
                    second += f"; from {o['provider']}: " + ("the same trade" if o["same_trade"] else f"{o['entry_time'][11:16]} {ol['strike']:g}{ol['right']} at {ol['entry_price']:.2f}, {o['exit_time'][11:16] if o['exit_time'] else ''} at {(ol['exit_price'] or 0):.2f}, {o['pnl']:+,.2f}")
                pr = x.get("print_range")
                L.append(f"| {s['date']} | {s['fall_pct']}% | {src} | {x['entry_time'][11:16]} {leg['strike']:g}{leg['right']} at {leg['entry_price']:.2f}" + (f" (minute {pr['entry_low']:.2f}-{pr['entry_high']:.2f})" if pr else "")
                         + f" | {x['exit_time'][11:16] if x['exit_time'] else ''} at {(leg['exit_price'] or 0):.2f}" + (f" (minute {pr['exit_low']:.2f}-{pr['exit_high']:.2f})" if pr and pr.get("exit_minute") else "")
                         + f" | {x['exit_reason'].replace('_', ' ')} | {x['pnl']:+,.2f} | " + (f"{pr['pnl_worst']:+,.2f} to {pr['pnl_best']:+,.2f}" if pr else "") + " | " + second + " |")
        unfilled = [s for s in r["sessions"] if s["status"] in ("signal not filled", "provider's history incomplete for the session")]
        if unfilled:
            L += ["", "Sessions with stored option data that produced no trade:"] + [f"- {s['date']}: {s['status']}" + (f" ({s.get('engine_note') or s.get('provider_note') or s.get('reference_note')})" if (s.get("engine_note") or s.get("provider_note") or s.get("reference_note")) else "") for s in unfilled[:60]]
    L += [""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"
