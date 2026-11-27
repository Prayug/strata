"""The observations behind a number (V8 "every number is explorable").

Every answer number carries an `explore` spec (strata.ai.answer); `observations(spec)` turns it into the rows behind it:

  {"kind": "sessions", "run_id": ..., "filter": all|hit|miss|worst|best}     the days of a study
  {"kind": "excluded", "run_id": ...}                                        the days a study could not evaluate, and why
  {"kind": "classification", "runs": [up, down], "filter": all|right|wrong}  the days of an agreement question, classified
  {"kind": "trades", "run_id": ..., "filter": all|winners|losers|worst|best}  the trades of an option backtest
  {"kind": "untested", "run_id": ...}                                        the signals of a backtest that could not be traded
  {"kind": "profile", "frame": {...}, "filter": all|<group>}                  the sessions behind a timing / weekday / count answer

The result is one table: columns (key, label, unit), rows, and how a row opens a replay. Nothing here is estimated or
recomputed differently from the result it explains: the rows are the stored observations of that run (or, for a profile,
the same function over the same stored bars).

`describe_days(store, symbol, dates)` answers "what do these days have in common?" for a selection of rows: the same few
facts for the selected days and for every stored session, side by side.
"""
from __future__ import annotations

import math
from datetime import date

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _f(x, nd: int = 4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def _hhmm(minute) -> str | None:
    if minute is None:
        return None
    m = int(minute)
    return f"{m // 60:02d}:{m % 60:02d}"


def _load_study(run_id: str) -> dict:
    from strata.api import db
    from strata.api.results import load_run_result
    run = db.get_run(run_id)
    if run is None:
        raise LookupError(f"run {run_id} not found")
    if run.get("kind") != "study":
        raise ValueError("that run is not a study of the underlying")
    return load_run_result(run)


def _session_row(r: dict, hit_word: str = "yes") -> dict:
    return {"date": r["date"], "weekday": WEEKDAYS[int(r.get("weekday", date.fromisoformat(r["date"]).weekday()))],
            "happened": hit_word if r.get("outcome") == "target" else "no" if r.get("outcome") in ("stop", "neither") else "unclear",
            "end_pct": _f(r.get("end_return_pct"), 3), "best_pct": _f(r.get("mfe_pct"), 3), "worst_pct": _f(r.get("mae_pct"), 3),
            "minutes_to": _f(r.get("minutes_to_target"), 0), "at": _hhmm(r.get("target_minute")), "start": _hhmm(r.get("entry_minute")),
            "prev_day_pct": _f(r.get("prev_return_pct"), 2), "gap_pct": _f(r.get("gap_pct"), 2), "vix": _f(r.get("vix_prev"), 1), "why": r.get("why") or []}


SESSION_COLS = [{"key": "date", "label": "Day"}, {"key": "weekday", "label": ""}, {"key": "happened", "label": "Happened?"},
                {"key": "end_pct", "label": "Close vs start", "unit": "%"}, {"key": "best_pct", "label": "Best point", "unit": "%"}, {"key": "worst_pct", "label": "Worst point", "unit": "%"},
                {"key": "at", "label": "When it happened"}, {"key": "prev_day_pct", "label": "Day before", "unit": "%"}, {"key": "gap_pct", "label": "Gap at open", "unit": "%"}]


def sessions(run_id: str, flt: str = "all", limit: int = 5000) -> dict:
    res = _load_study(run_id)
    rows = [_session_row(r) for r in res.get("sessions") or []]
    st = res.get("stats") or {}
    total = len(rows)
    title = {"hit": "Days it happened", "miss": "Days it did not happen", "worst": "Days that ended lowest", "best": "Days that ended highest"}.get(flt, "Days that matched your rules")
    if flt == "hit":
        rows = [r for r in rows if r["happened"] == "yes"]
    elif flt == "miss":
        rows = [r for r in rows if r["happened"] == "no"]
    elif flt in ("worst", "best"):
        rows = sorted([r for r in rows if r["end_pct"] is not None], key=lambda r: r["end_pct"], reverse=(flt == "best"))[:25]
    elif flt in ("up", "down"):
        rows = [r for r in reversed(rows) if r["end_pct"] is not None and ((r["end_pct"] > 0) if flt == "up" else (r["end_pct"] < 0))]
    else:
        rows = list(reversed(rows))
    title = {"up": "Days it ended higher than it started", "down": "Days it ended lower than it started"}.get(flt, title)
    cap = []
    if res.get("sessions_truncated"):
        cap.append(f"Only the latest {total:,} of {st.get('qualifying_sessions', total):,} days are kept with the result.")
    if flt in ("worst", "best"):
        cap.append("The 25 most extreme days by where the price closed against the start.")
    return {"kind": "sessions", "title": title, "count": len(rows), "total": total, "columns": SESSION_COLS, "rows": rows[:limit],
            "caption": " ".join(cap), "replay": {"type": "study_day", "run_id": run_id, "symbol": res.get("symbol")}, "symbol": res.get("symbol")}


def excluded(run_id: str) -> dict:
    res = _load_study(run_id)
    rows = []
    for e in res.get("excluded") or []:
        for d in e.get("dates") or []:
            rows.append({"date": d, "reason": e["reason"]})
        if e.get("count", 0) > len(e.get("dates") or []):
            rows.append({"date": "...", "reason": f"{e['count'] - len(e.get('dates') or [])} more days for the same reason"})
    n = sum(e.get("count", 0) for e in res.get("excluded") or [])
    return {"kind": "excluded", "title": "Days that could not be tested", "count": n, "total": n, "columns": [{"key": "date", "label": "Day"}, {"key": "reason", "label": "Why it was left out"}],
            "rows": rows, "caption": "These days met the rules but the stored data could not decide the outcome on them, so they are not in the counts.", "symbol": res.get("symbol")}


def classification(runs: list[str], flt: str = "all", limit: int = 5000) -> dict:
    """The days of an agreement question: the candle's direction, the later direction, and whether they matched."""
    out, sym = [], None
    for rid, cdir in zip(runs, ("up", "down")):
        res = _load_study(rid)
        sym = res.get("symbol")
        for r in res.get("sessions") or []:
            if r.get("outcome") == "ambiguous":
                continue
            agree = r.get("outcome") == "target"
            later = cdir if agree else ("down" if cdir == "up" else "up")
            out.append({"date": r["date"], "weekday": WEEKDAYS[int(r.get("weekday", 0))], "candle": cdir, "later": later, "matched": "yes" if agree else "no",
                        "candle_pct": None, "later_pct": _f(r.get("end_return_pct"), 3), "_run": rid})
    out.sort(key=lambda x: x["date"], reverse=True)
    total = len(out)
    if flt == "right":
        out = [x for x in out if x["matched"] == "yes"]
    elif flt == "wrong":
        out = [x for x in out if x["matched"] == "no"]
    cols = [{"key": "date", "label": "Day"}, {"key": "weekday", "label": ""}, {"key": "candle", "label": "First candle"}, {"key": "later", "label": "Then"},
            {"key": "matched", "label": "Same way?"}, {"key": "later_pct", "label": "Move after the candle", "unit": "%"}]
    title = {"right": "Days the candle and the session went the same way", "wrong": "Days they went different ways"}.get(flt, "Every classified day")
    return {"kind": "classification", "title": title, "count": len(out), "total": total, "columns": cols, "rows": out[:limit],
            "caption": "A day counts when its first candle closed above or below the open; days with no direction are left out.", "replay": {"type": "study_day", "symbol": sym}, "symbol": sym}


def _contract(t: dict) -> str:
    legs = [x for x in t.get("legs") or [] if x.get("kind") == "option"]
    if not legs:
        return t.get("symbol") or ""
    parts = []
    for x in legs:
        side = "short " if x.get("side") == "sell" else ""
        parts.append(f"{side}{t.get('symbol', '')} {x.get('strike'):g} {'call' if x.get('right') == 'C' else 'put'} exp {x.get('expiration')}")
    return " + ".join(parts)


TRADE_COLS = [{"key": "date", "label": "Day"}, {"key": "contract", "label": "Contract"}, {"key": "entry", "label": "In"}, {"key": "exit", "label": "Out"},
              {"key": "paid", "label": "Premium in", "unit": "$"}, {"key": "got", "label": "Premium out", "unit": "$"}, {"key": "pnl", "label": "Result after fees", "unit": "$"},
              {"key": "premium_pct", "label": "Option change", "unit": "%"}, {"key": "underlying_pct", "label": "Underlying change", "unit": "%"}, {"key": "reason", "label": "Why it closed"}]
REASONS = {"time_exit": "time exit", "target": "profit target", "stop": "stop", "expiration": "expired", "premium_target": "profit target on the option", "premium_stop": "stop on the option",
           "underlying_target": "target on the underlying", "underlying_stop": "stop on the underlying", "end_of_data": "end of data", "exit_condition": "exit rule"}


def trade_row(t: dict) -> dict:
    legs = [x for x in t.get("legs") or [] if x.get("kind") == "option"]
    l0 = legs[0] if legs else {}
    es, xs = t.get("entry_spot"), t.get("exit_spot")
    return {"id": t["id"], "date": t.get("entry_date"), "contract": _contract(t), "entry": (t.get("entry_time") or "")[11:16], "exit": (t.get("exit_time") or "")[11:16],
            "exit_date": (t.get("exit_time") or "")[:10], "paid": _f(l0.get("entry_price"), 2), "got": _f(l0.get("exit_price"), 2), "pnl": _f(t.get("pnl"), 2), "gross": _f(t.get("gross_pnl"), 2),
            "fees": _f((t.get("commissions") or 0) + (t.get("slippage") or 0), 2), "premium_pct": _f(t.get("pnl_pct_premium"), 1),
            "underlying_pct": _f((xs / es - 1) * 100, 2) if es and xs else None, "reason": REASONS.get(t.get("exit_reason"), str(t.get("exit_reason") or "").replace("_", " ")),
            "multiplier": l0.get("multiplier"), "contracts": l0.get("contracts") or t.get("units"), "source": l0.get("price_source")}


def trades(run_id: str, flt: str = "all", limit: int = 5000) -> dict:
    from strata.api import db
    from strata.api.results import load_trades
    run = db.get_run(run_id)
    if run is None:
        raise LookupError(f"run {run_id} not found")
    rows = [trade_row(t) for t in load_trades(run_id)]
    total = len(rows)
    if flt in ("winners", "win", "hit"):
        rows = [r for r in rows if (r["pnl"] or 0) > 0]
    elif flt in ("losers", "loss", "miss"):
        rows = [r for r in rows if (r["pnl"] or 0) <= 0]
    elif flt in ("worst", "best"):
        rows = sorted(rows, key=lambda r: r["pnl"] or 0, reverse=(flt == "best"))[:25]
    else:
        rows = list(reversed(rows))
    title = {"winners": "Trades that made money", "losers": "Trades that lost money", "worst": "The biggest losses", "best": "The biggest gains"}.get(flt, "Every trade")
    net = round(sum((r["pnl"] or 0) for r in rows), 2)
    return {"kind": "trades", "title": title, "count": len(rows), "total": total, "columns": TRADE_COLS, "rows": rows[:limit], "sum": {"pnl": net},
            "caption": f"These {len(rows):,} trades add up to {'+' if net >= 0 else '-'}${abs(net):,.2f} after fees. Premiums are per share; one contract is {int(rows[0]['multiplier'] or 100) if rows else 100} shares.",
            "replay": {"type": "trade", "run_id": run_id}}


def untested(run_id: str) -> dict:
    from strata.api import db
    from strata.api.results import load_result
    run = db.get_run(run_id)
    if run is None:
        raise LookupError(f"run {run_id} not found")
    cov = (load_result(run_id).get("coverage") or {})
    rows = []
    for r in ((cov.get("skipped_signals") or {}).get("rows") or []):
        rows.append({"date": r.get("date") or r.get("session") or "", "reason": r.get("reason") or r.get("why") or r.get("kind") or ""})
    for i in ((cov.get("options") or {}).get("issues") or []):
        for d in i.get("dates") or ([i.get("date")] if i.get("date") else []):
            rows.append({"date": d, "reason": i.get("reason") or ""})
    q = (cov.get("signals") or {}).get("qualifying")
    ex = (cov.get("executable") or {}).get("signals")
    n = (q - ex) if isinstance(q, int) and isinstance(ex, int) else len(rows)
    return {"kind": "untested", "title": "Signals that could not be traded", "count": n, "total": q, "columns": [{"key": "date", "label": "Day"}, {"key": "reason", "label": "Why"}],
            "rows": rows, "caption": "The rule fired on these days, but no stored option price could fill the trade, so they are not in the results. Nothing was substituted."}


def profile(store, frame: dict, flt: str = "all") -> dict:
    import polars as pl
    from strata.research import profile as PR
    f = frame or {}
    sym = f.get("symbol") or "SPY"
    daily = PR._daily(store, sym)
    by_date = {r["date"].isoformat(): r for r in daily.select("date", "gap", "oc", "cc").iter_rows(named=True)}
    day = lambda d, k: _f((by_date.get(d) or {}).get(k), 3)
    if f.get("kind") == "timing":
        res = PR.timing(store, sym, f.get("measure") or "abs_move", int(f.get("bucket") or 30), f.get("period"), f.get("conditions"))
        meas = res.get("measure")
        what = {"high_time": "High made at", "low_time": "Low made at"}.get(meas, "Biggest move in")
        rows = [{"date": r["date"], "weekday": WEEKDAYS[date.fromisoformat(r["date"]).weekday()], "bucket": r.get("bucket"), "time": r.get("time"),
                 "move_pct": _f(r.get("move_pct"), 3), "day_pct": day(r["date"], "oc")} for r in res.get("per_session") or []]
        rows.reverse()
        if flt not in (None, "all"):
            rows = [r for r in rows if str(r.get("bucket") or "").startswith(str(flt))]
        cols = [{"key": "date", "label": "Day"}, {"key": "weekday", "label": ""}, {"key": "bucket", "label": what}]
        cols += [{"key": "time", "label": "Minute"}] if meas in ("high_time", "low_time") else [{"key": "move_pct", "label": "Size of that move", "unit": "%"}]
        cols += [{"key": "day_pct", "label": "Open to close", "unit": "%"}]
        return {"kind": "profile", "title": f"{sym} sessions behind this answer", "count": len(rows), "total": res.get("sessions"), "columns": cols, "rows": rows,
                "caption": "Whole sessions only.", "replay": {"type": "day", "symbol": sym}, "symbol": sym}
    if f.get("kind") == "breakdown":
        res = PR.breakdown(store, sym, f.get("by") or "weekday", f.get("measure") or "open_to_close", f.get("period"), f.get("conditions"))
        k = "cc" if (f.get("measure") or "open_to_close") == "close_to_close" else "oc"
        rows = []
        for g in res.get("groups") or []:
            if flt not in (None, "all") and str(g.get("group"))[:3].lower() != str(flt)[:3].lower():
                continue
            for d in g.get("dates") or []:
                rows.append({"date": d, "group": g.get("group"), "value_pct": day(d, k)})
        rows.sort(key=lambda r: r["date"], reverse=True)
        return {"kind": "profile", "title": f"{sym} sessions" + (f" on {flt}" if flt not in (None, "all") else f" by {f.get('by') or 'weekday'}"), "count": len(rows), "total": res.get("sessions"),
                "columns": [{"key": "date", "label": "Day"}, {"key": "group", "label": "Group"}, {"key": "value_pct", "label": "Close-to-close" if k == "cc" else "Open to close", "unit": "%"}],
                "rows": rows, "caption": "", "replay": {"type": "day", "symbol": sym}, "symbol": sym}
    ev = (f.get("outcome") or {}).get("event") or {}
    res = PR.event_frequency(store, sym, ev, f.get("period"), f.get("conditions"))
    k = "gap" if ev.get("key") == "gap" else "cc"
    rows = [{"date": d, "weekday": WEEKDAYS[date.fromisoformat(d).weekday()], "value_pct": day(d, k), "day_pct": day(d, "oc")} for d in res.get("dates") or []]
    rows.reverse()
    return {"kind": "profile", "title": f"The days it happened on {sym}", "count": len(rows), "total": res.get("n"),
            "columns": [{"key": "date", "label": "Day"}, {"key": "weekday", "label": ""}, {"key": "value_pct", "label": "Gap at the open" if k == "gap" else "Close-to-close", "unit": "%"},
                        {"key": "day_pct", "label": "Open to close", "unit": "%"}],
            "rows": rows, "caption": res.get("what") and f"Days it {res['what']}.", "replay": {"type": "day", "symbol": sym}, "symbol": sym}


def observations(spec: dict, store=None) -> dict:
    k = (spec or {}).get("kind")
    flt = spec.get("filter") or "all"
    if k == "sessions":
        return sessions(spec["run_id"], flt)
    if k == "excluded":
        return excluded(spec["run_id"])
    if k == "classification":
        return classification(spec["runs"], flt)
    if k == "trades":
        return trades(spec["run_id"], flt)
    if k in ("untested", "skipped"):
        return untested(spec["run_id"])
    if k == "profile":
        if store is None:
            from strata.data.store import MarketStore
            store = MarketStore()
        return profile(store, spec.get("frame") or {}, flt)
    raise ValueError(f"unknown observation kind: {k!r}")


# ------------------------------------------------------------------------------------------------- a selection of days
def describe_days(store, symbol: str, days: list[str]) -> dict:
    """What a set of selected days have in common, against every stored session of the same symbol: the gap at the open,
    the day before, the day itself, the VIX the day before, the weekday mix and the years. Read from stored daily bars."""
    import polars as pl
    from strata.research.profile import _daily, _vix
    d = _daily(store, symbol).drop_nulls(["prev_close"])
    v = _vix(store)
    if v is not None:
        d = d.join(v, on="date", how="left")
    want = {date.fromisoformat(x) for x in days if x and x[:4].isdigit()}
    sel = d.filter(pl.col("date").is_in(list(want)))
    if sel.height == 0:
        raise ValueError("none of the selected days are stored sessions of " + symbol)

    def stats(x: pl.DataFrame) -> dict:
        def m(col):
            return _f(x[col].drop_nulls().mean(), 3) if col in x.columns and x[col].drop_nulls().len() else None
        return {"days": x.height, "gap_pct": m("gap"), "prev_day_pct": m("prev_return"), "day_pct": m("oc"), "close_to_close_pct": m("cc"),
                "green_share": _f((x["oc"] > 0).mean(), 3) if "oc" in x.columns and x.height else None, "vix_prev": m("vix_prev")}
    a, b = stats(sel), stats(d)
    wk = {w: 0 for w in WEEKDAYS[:5]}
    for dt in sel["date"].to_list():
        wk[WEEKDAYS[dt.weekday()]] = wk.get(WEEKDAYS[dt.weekday()], 0) + 1
    yrs: dict = {}
    for dt in sel["date"].to_list():
        yrs[dt.year] = yrs.get(dt.year, 0) + 1
    lines = []
    tmpl = {"gap_pct": "On average they opened {x:+.2f}% from the previous close (all {n:,} days: {y:+.2f}%).",
            "prev_day_pct": "The day before, they had moved {x:+.2f}% on average (all days: {y:+.2f}%).",
            "day_pct": "From open to close they moved {x:+.2f}% on average (all days: {y:+.2f}%).",
            "vix_prev": "The VIX averaged {x:.1f} the day before (all days: {y:.1f})."}
    for key, t in tmpl.items():
        x, y = a.get(key), b.get(key)
        if x is not None and y is not None:
            lines.append(t.format(x=x, y=y, n=b["days"]))
    top_wd = max(wk.items(), key=lambda kv: kv[1])
    full = {"Mon": "Mondays", "Tue": "Tuesdays", "Wed": "Wednesdays", "Thu": "Thursdays", "Fri": "Fridays"}
    if a["days"] >= 5 and top_wd[1] / a["days"] > 0.35:
        lines.append(f"{top_wd[1]} of the {a['days']} days were {full[top_wd[0]]} (about {a['days'] // 5} would be usual).")
    if a["days"] < 20:
        lines.append(f"{a['days']} days is a small sample: differences this size can appear by chance.")
    return {"symbol": symbol, "selected": a, "all": b, "weekdays": wk, "years": dict(sorted(yrs.items())), "lines": lines,
            "missing": sorted(x.isoformat() for x in want - set(sel["date"].to_list()))}


# ------------------------------------------------------------------------------------------------- what can be tested
def _month(iso: str | None) -> str:
    if not iso:
        return "?"
    d = date.fromisoformat(iso[:10])
    return d.strftime("%B %Y")


def scope(store, symbol: str) -> dict:
    """What can be tested for one symbol, in plain sentences, with the four kinds of data kept apart: underlying prices,
    option trades (prices people paid), bid/ask quotes, and theoretical (model) option prices."""
    from strata.data.catalog import catalog
    c = next((x for x in catalog(store) if x["symbol"] == symbol), None)
    if c is None:
        raise LookupError(f"{symbol} is not stored")
    d, intr, o, tr = c.get("daily"), c.get("intraday") or {}, c.get("options"), c.get("option_trades")
    m1 = intr.get("1m")
    items = []
    if d:
        items.append({"kind": "underlying", "label": "Price history of " + symbol, "from": d["first"], "to": d["last"],
                      "text": f"{symbol}'s daily prices go back to {_month(d['first'])}" + (f", and its minute-by-minute prices to {_month(m1['first'])}." if m1 else "."),
                      "why": "Enough for questions about how the price itself moved: how often, how far, at what time of day." + ("" if m1 else " Without minute prices, time-of-day questions cannot be answered.")})
    if tr:
        items.append({"kind": "option_trades", "label": "Historical option trades", "from": tr["first"], "to": tr["last"],
                      "text": f"Real historical option trades are stored from {_month(tr['first'])} to {_month(tr['last'])}: the prices contracts actually traded at, minute by minute.",
                      "why": "Option backtests use these prices. They are trades, not quotes: they show what someone paid, not the bid and ask you would have faced."})
    if o:
        eod = o.get("resolution") == "EOD"
        items.append({"kind": "quotes", "label": "Bid/ask quotes" + (" (end of day)" if eod else ""), "from": o["first"], "to": o["last"],
                      "text": (f"End-of-day bid/ask quotes are stored from {_month(o['first'])} to {_month(o['last'])}: one quote per contract per day, at the close." if eod else
                               f"Bid/ask quotes are stored from {_month(o['first'])} to {_month(o['last'])}."),
                      "why": "Quotes say what buying or selling would have cost. End-of-day quotes cannot price a trade during the day."})
    items.append({"kind": "intraday_quotes", "label": "Intraday bid/ask quotes", "from": None, "to": None,
                  "text": "Intraday bid/ask quotes are not stored. The source for them (Databento) needs an account key, so filling trades at the bid or ask during the day has not been verified.",
                  "why": "Without them, intraday option results use trade prices, which can be better than the price you would actually have been filled at."})
    if c.get("iv_history"):
        iv = c["iv_history"]
        items.append({"kind": "theoretical", "label": "Theoretical option prices", "from": iv.get("first"), "to": iv.get("last"),
                      "text": f"Theoretical option prices can be calculated from {_month(iv.get('first'))} using a pricing model and stored volatility. They are estimates, not observed prices, and are only used when you choose them.",
                      "why": "Useful for rough comparisons further back in time; never mixed into results that say they use observed prices."})
    lines = [it["text"] for it in items]
    if d and tr and d["first"] < tr["first"]:
        lines.append(f"So {symbol}'s movement can be studied back to {d['first'][:4]}, but option profits only from {_month(tr['first'])}" + (f" (or from {_month(o['first'])} for end-of-day strategies)." if o and o["first"] < tr["first"] else "."))
    return {"symbol": symbol, "synthetic": bool(c["meta"].get("synthetic")), "items": items, "lines": lines}
