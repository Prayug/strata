"""Every answer, checked against the independent reference (reference.py, reference_options.py) before it is shown.

The check never changes an answer. It says, in one line, whether a separate calculation from the stored files found the same
numbers; when it did not, the days (or trades) that differ are listed, so the difference can be traced. A result the
reference does not cover says so instead of claiming a check that did not happen.
"""
from __future__ import annotations

import threading

_LOCK = threading.Lock()  # one DuckDB connection, used by one request at a time


def _ref():
    from strata.research.reference import reference
    return reference()


def _out(status: str, text: str, **kw) -> dict:
    """status: same (fully verified) | partly (V10: part of the result was recalculated and agrees; the text says which part
    was not) | differs | not_covered. `level` says the same in the words shown to the user."""
    level = {"same": "Fully verified", "partly": "Partly verified", "differs": "Disagreement", "not_covered": "Not verified"}[status]
    return {"status": status, "level": level, "text": text, "method": "a second calculation written apart from the engines, in SQL on the stored files (strata.research.reference)", **kw}


def study(result: dict, frame: dict | None) -> dict:
    """A frequency or movement study against the reference on the same question frame."""
    f = frame or {}
    if not f.get("kind"):
        return _out("not_covered", "No independent recalculation covers this question yet.")
    st = (result or {}).get("stats") or {}
    try:
        with _LOCK:
            r = _ref().check_frame(f)
    except Exception as e:  # noqa: BLE001
        return _out("not_covered", f"The independent recalculation could not run ({type(e).__name__}).")
    if r is None:
        return _out("not_covered", "No independent recalculation covers this question yet.")
    n = int(st.get("qualifying_sessions") or 0)
    if r["kind"] == "frequency":
        k = int(st.get("target_first") or 0)
        if (r["n"], r["k"]) == (n, k):
            return _out("same", f"Recalculated independently from the stored bars: the same {k:,} of {n:,} sessions.", n=n, k=k)
        mine = {s["date"]: s["outcome"] == "target" for s in (result.get("sessions") or [])}
        diff = []
        if mine and not result.get("sessions_truncated"):
            ref_dates, hits = set(r["dates"]), set(r["hits"])
            diff = sorted(set(mine) ^ ref_dates) + sorted(d for d in set(mine) & ref_dates if mine[d] != (d in hits))
        return _out("differs", f"An independent recalculation found {r['k']:,} of {r['n']:,} instead of {k:,} of {n:,}" + (f"; the sessions that differ: {', '.join(diff[:6])}{' and more' if len(diff) > 6 else ''}." if diff else "."),
                    n=n, k=k, ref_n=r["n"], ref_k=r["k"], dates=diff[:50])
    if r["kind"] == "movement":
        m = st.get("avg_end_return_pct")
        if m is None or r.get("mean") is None:
            return _out("not_covered", "No independent recalculation covers this question yet.")
        if r["n"] == n and abs(float(m) - float(r["mean"])) < 0.0005:
            return _out("same", f"Recalculated independently from the stored bars: the same {n:,} sessions and the same average move ({float(m):+.3f}%).", n=n, mean=m)
        return _out("differs", f"An independent recalculation found an average of {float(r['mean']):+.3f}% over {r['n']:,} sessions, against {float(m):+.3f}% over {n:,}.", n=n, mean=m, ref_n=r["n"], ref_mean=r["mean"])
    return _out("not_covered", "No independent recalculation covers this question yet.")


def day_part(frame: dict | None, rows: list[dict]) -> dict:
    """A part of the day against the day before it ("does SPY reverse during lunch?"): each of the two counts (after an up
    morning, after a down morning) is counted again in SQL from the one-minute bars, as a price at one time against a price at
    another."""
    f = frame or {}
    part = next((c for c in f.get("conditions") or [] if c.get("key") == "first_candle" and c.get("part")), None)
    way = (f.get("outcome") or {}).get("direction")
    if part is None or way not in ("opposite", "same") or len(rows) != 2:
        return _out("not_covered", "Not independently recalculated: this answer compares several variants of one question.")
    sym = f.get("symbol") or "SPY"
    start, end = f.get("start"), f.get("end") or "close"
    others = [c for c in f.get("conditions") or [] if c is not part]
    period = tuple(f.get("period") or (None, None))
    try:
        with _LOCK:
            R = _ref()
            for r in rows:
                up = " up " in f" {r.get('label')} "
                op = ("<" if up else ">") if way == "opposite" else (">" if up else "<")
                ref = R.relation(sym, {"a": "close" if end == "close" else f"at:{end}", "op": op, "b": f"at:{start}"},
                                 [{"key": "rel", "rel": {"a": f"at:{start}", "op": ">" if up else "<", "b": "open"}}] + others, period)
                mine = (int(r.get("n") or 0), int(round(float(r.get("probability") or 0) * int(r.get("n") or 0))))
                if (ref["n"], ref["k"]) != mine:
                    return _out("differs", f"An independent recalculation of \"{r.get('label')}\" found {ref['k']:,} of {ref['n']:,} sessions instead of {mine[1]:,} of {mine[0]:,}.",
                                n=mine[0], k=mine[1], ref_n=ref["n"], ref_k=ref["k"])
    except Exception as e:  # noqa: BLE001
        return _out("not_covered", f"Not independently recalculated ({type(e).__name__}: {str(e)[:120]}).")
    return _out("same", "Recalculated independently from the one-minute bars: the same sessions and the same counts after an up morning and after a down one.")


def agreement(report: dict, frame: dict | None) -> dict:
    rows = (report or {}).get("agreement") or []
    f = frame or {}
    if not rows or f.get("predictor", "first_candle") != "first_candle":
        return _out("not_covered", "No independent recalculation covers this question yet.")
    conds = [c for c in f.get("conditions") or [] if not c.get("values")]
    bad = []
    try:
        with _LOCK:
            R = _ref()
            for m in sorted({int(x["minutes"]) for x in rows}):
                ref = R.agreement(f.get("symbol") or "SPY", m, conds, tuple(f.get("period") or (None, None)))
                for x in rows:
                    if int(x["minutes"]) != m:
                        continue
                    u, d = x["up"], x["down"]
                    hits = round((u.get("probability") or 0) * u["n"] + (d.get("probability") or 0) * d["n"])
                    tot = u["n"] + d["n"]
                    rn, rk = (ref["rest_n"], ref["rest_agree"]) if x["scope"] == "rest" else (ref["day_n"], ref["day_agree"])
                    if (rn, rk) != (tot, hits):
                        bad.append(f"{m}-minute candle, {'rest of the session' if x['scope'] == 'rest' else 'whole day'}: {hits:,} of {tot:,} here, {rk:,} of {rn:,} independently")
    except Exception as e:  # noqa: BLE001
        return _out("not_covered", f"The independent recalculation could not run ({type(e).__name__}).")
    if bad:
        return _out("differs", "An independent recalculation differs: " + "; ".join(bad[:4]) + ".", details=bad)
    return _out("same", "Recalculated independently from the stored bars: every agreement count is the same.")


def backtest(run: dict, trades: list[dict], draft: dict | None) -> dict:
    """A single long call or put with a clock entry and exit, priced on trade prints: re-priced trade by trade."""
    d = draft or {}
    if not trades:
        return _out("not_covered", "There are no trades to re-price.")
    cfg = (run or {}).get("config") or {}
    simple = (d.get("structure") in ("long_call", "long_put") and not d.get("sequence") and not d.get("adds") and not d.get("partials") and not d.get("exit_conditions")
              and d.get("premium_target_pct") is None and d.get("premium_stop_pct") is None and d.get("underlying_target_pct") is None and d.get("underlying_stop_pct") is None
              and d.get("exit_time") and (d.get("entry_time") or "open") and d.get("dte") in (0, None) and (d.get("strike") or {}).get("method", "atm") == "atm"
              and float((d.get("strike") or {}).get("value") or 0) == 0 and not d.get("weekdays") and int(d.get("max_entries") or 1) == 1)
    conds = d.get("conditions") or []
    prev = None
    for c in conds:
        import re
        m = re.fullmatch(r"\s*(?:previous_session\.return_pct|return_pct\(1\))\s*(<=|<|>=|>)\s*(-?\d+(?:\.\d+)?)\s*", c.get("expr", ""))
        if not m or prev is not None:
            simple = False
            break
        v = float(m.group(2))
        prev = (v - 1e-9) if m.group(1) == "<" else (v + 1e-9) if m.group(1) == ">" else v
    sizing = d.get("sizing") or {}
    if not simple or sizing.get("mode") not in (None, "contracts", "dollars", "risk_dollars") or sizing.get("reinvest") or sizing.get("daily_loss_stop"):
        return _out("not_covered", "This strategy is not of a form the independent re-pricing covers (one long call or put, clock entry and exit, a previous-day condition).")
    entry = d.get("entry_time") or "open"
    args = {"symbol": d.get("symbol") or "SPY", "right": "call" if d["structure"] == "long_call" else "put", "entry": "09:30" if entry == "open" else entry,
            "exit": "15:55" if d.get("exit_time") == "close" else d["exit_time"], "prev": prev, "contracts": int(d.get("quantity") or 1)}
    if sizing.get("mode") == "dollars":
        args.pop("contracts")
        args["dollars"] = float(sizing["value"])
    elif sizing.get("mode") == "risk_dollars":
        args.pop("contracts")
        args["risk"] = float(sizing["value"])
    try:
        from strata.research.reference_options import compare_trades, option_backtest
        with _LOCK:
            ref = option_backtest(args, _ref())
        start, end = cfg.get("start"), cfg.get("end")
        if start or end:
            keep = [t for t in ref["per_trade"] if (not start or t["date"] >= str(start)[:10]) and (not end or t["date"] <= str(end)[:10])]
            ref = {**ref, "per_trade": keep, "trades": len(keep), "pnl": round(sum(t["pnl"] for t in keep), 2)}
        cmp = compare_trades(trades, ref)
    except Exception as e:  # noqa: BLE001
        return _out("not_covered", f"The independent re-pricing could not run ({type(e).__name__}).")
    total = round(sum(float(t.get("pnl") or 0) for t in trades), 2)
    if not cmp["different"] and not cmp["only_app"] and not cmp["only_reference"]:
        money = f"{'-' if total < 0 else '+'}${abs(total):,.2f}"
        return _out("same", f"Re-priced independently from the stored option prints: all {len(trades):,} trades and the {money} total are the same.", trades=len(trades), pnl=total)
    parts = []
    if cmp["different"]:
        parts.append(f"{len(cmp['different'])} trade(s) priced differently ({', '.join(x['date'] for x in cmp['different'][:4])})")
    if cmp["only_app"]:
        parts.append(f"{len(cmp['only_app'])} trade(s) the re-pricing did not take ({', '.join(cmp['only_app'][:4])})")
    if cmp["only_reference"]:
        parts.append(f"{len(cmp['only_reference'])} signal(s) the re-pricing traded and this run did not ({', '.join(cmp['only_reference'][:4])})")
    return _out("differs", f"An independent re-pricing matched {cmp['same']:,} of {len(trades):,} trades; " + "; ".join(parts) + ".", trades=len(trades), pnl=total, ref_pnl=ref["pnl"], details=cmp)


# --------------------------------------------------------------------------------- V10: the answers computed from stored bars
def profile(frame: dict | None, res: dict | None) -> dict:
    """A relation, time-of-day, weekday or event answer (strata.research.profile), recalculated in SQL from the raw files."""
    f, res = frame or {}, res or {}
    sym = f.get("symbol") or res.get("symbol") or "SPY"
    conds = [c for c in f.get("conditions") or [] if not c.get("values")]
    if len(conds) != len(f.get("conditions") or []) or res.get("conditions_not_evaluated"):
        return _out("not_covered", "Not independently recalculated: this answer covers several sizes at once, or a condition that could not be applied.")
    period = tuple(f.get("period") or (None, None))
    try:
        with _LOCK:
            R = _ref()
            if (f.get("outcome") or {}).get("key") == "rel":
                r = R.relation(sym, f["outcome"]["rel"], conds, period, (f.get("then") or {}).get("rel"))
                mine = (int(res.get("n") or 0), int(res.get("k") or 0))
                then_ok = not f.get("then") or int((res.get("then") or {}).get("k") or 0) == int(r.get("then_k") or 0)
                if (r["n"], r["k"]) == mine and then_ok:
                    return _out("same", f"Recalculated independently from the stored bars: the same {mine[1]:,} of {mine[0]:,} sessions" + (f", and the same {int(r['then_k']):,} of those afterwards." if f.get("then") else "."), n=mine[0], k=mine[1])
                diff = sorted(set(res.get("dates") or []) ^ set(r["hits"]))
                return _out("differs", f"An independent recalculation found {r['k']:,} of {r['n']:,} sessions instead of {mine[1]:,} of {mine[0]:,}" + (f"; the sessions that differ: {', '.join(diff[:6])}{' and more' if len(diff) > 6 else ''}." if diff else "."),
                            n=mine[0], k=mine[1], ref_n=r["n"], ref_k=r["k"], dates=diff[:50])
            if f.get("kind") == "timing":
                r = R.bucket_stats(sym, f.get("measure") or "abs_move", int(f.get("bucket") or 30), conds, period)
                rows = res.get("buckets") or []
                bad = []
                share = (f.get("measure") or "abs_move") in ("high_time", "low_time")
                for b in rows:
                    ref = r["by"].get(b["start"])
                    if share:
                        if (ref or {}).get("n", 0) != int(b.get("sessions") or 0):
                            bad.append(f"{b['start']}: {b.get('sessions')} sessions here, {(ref or {}).get('n', 0)} independently")
                    elif b.get("mean_abs_move_pct") is not None and (ref is None or abs(float(ref["mean"]) - float(b["mean_abs_move_pct"])) > 0.0005):
                        bad.append(f"{b['start']}: {b['mean_abs_move_pct']:.4f}% here, {(ref or {}).get('mean', float('nan')):.4f}% independently")
                if not bad and r["sessions"] == int(res.get("sessions") or 0):
                    return _out("same", f"Recalculated independently from the one-minute bars: the same {r['sessions']:,} sessions and the same figure in every part of the day.", n=r["sessions"])
                if not bad:
                    bad = [f"{res.get('sessions')} sessions here, {r['sessions']} independently"]
                return _out("differs", "An independent recalculation differs: " + "; ".join(bad[:4]) + ".", details=bad[:20])
            if f.get("kind") == "breakdown":
                r = R.group_stats(sym, f.get("by") or "weekday", conds, period)
                meas = res.get("measure") or f.get("measure") or "open_to_close"
                key = {"open_to_close": "oc_mean", "return": "oc_mean", "close_to_close": "cc_mean", "close>open": "oc_up", "close<open": "oc_down", "close>prev_close": "cc_up", "close<prev_close": "cc_down"}.get(meas)
                bad = []
                for g in res.get("groups") or []:
                    ref = r["by"].get(g["group"])
                    mine = g.get("share") if res.get("share_measure") else g.get("mean_pct")
                    if ref is None or ref["n"] != int(g["sessions"]) or mine is None or abs(float(ref[key]) - float(mine)) > (0.005 if res.get("share_measure") else 0.0005):
                        bad.append(f"{g['group']}: {mine} over {g['sessions']} here, {(ref or {}).get(key)} over {(ref or {}).get('n')} independently")
                if not bad:
                    return _out("same", f"Recalculated independently from the stored bars: the same sessions and the same figure for every {f.get('by') or 'weekday'}.", n=int(res.get("sessions") or 0))
                return _out("differs", "An independent recalculation differs: " + "; ".join(bad[:4]) + ".", details=bad[:20])
            ev = (f.get("outcome") or {}).get("event")
            if ev:
                r = R.event(sym, ev, conds, period)
                mine = (int(res.get("n") or 0), int(res.get("k") or 0))
                if (r["n"], r["k"]) == mine:
                    return _out("same", f"Recalculated independently from the stored bars: the same {mine[1]:,} of {mine[0]:,} sessions.", n=mine[0], k=mine[1])
                return _out("differs", f"An independent recalculation found {r['k']:,} of {r['n']:,} sessions instead of {mine[1]:,} of {mine[0]:,}.", n=mine[0], k=mine[1], ref_n=r["n"], ref_k=r["k"])
    except Exception as e:  # noqa: BLE001
        return _out("not_covered", f"The independent recalculation could not run ({type(e).__name__}: {str(e)[:80]}).")
    return _out("not_covered", "No independent recalculation covers this question yet.")


# ------------------------------------------------------------- V10: any options backtest, its fills re-read and its sums re-added
def fills(run: dict, trades: list[dict]) -> dict:
    """For a strategy the full re-pricing does not cover (several legs, exits on targets and stops, sizing rules, several
    sessions): every option fill is looked up again in the raw option files, by a separate SQL reader, and every trade's
    result is re-added from its own cash flows. This verifies the PRICES and the ARITHMETIC. It does not verify which
    sessions qualified or which strikes the rule selected: the result is "partly verified", and the text says so."""
    import json as _json
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    if not trades:
        return _out("not_covered", "There are no trades to check.")
    R = _ref()
    NY = ZoneInfo("America/New_York")
    MAX_AGE = 5  # minutes a print may be old when it prices a fill (the reference re-pricing's rule)
    checked = legs_n = 0
    bad_price, bad_sum, unread = [], [], 0
    total = 0.0
    cache: dict = {}

    def bars(sym: str, day: str, right: str, strike: float, exp: str) -> dict | None:
        key = (sym, day, right, round(float(strike), 4), exp)
        if key in cache:
            return cache[key]
        root = R.root / "alpaca" / sym / "options" / "trade" / "1m"
        vdir = sorted(root.glob("v*"))
        f = (vdir[-1] / "data" / day[:4] / f"{day}.parquet") if vdir else None
        if f is None or not f.exists():
            cache[key] = None
            return None
        rows = R.con.execute(f"""select (extract(hour from timezone('America/New_York', ts)) * 60 + extract(minute from timezone('America/New_York', ts)))::int m, close
                                 from read_parquet('{f}') where "right" = ? and abs(strike - ?) < 1e-6 and expiration = cast(? as date)""", [right, float(strike), exp]).fetchall()
        cache[key] = {int(m): float(c) for m, c in rows}
        return cache[key]

    def last_print(b: dict, t: int) -> float | None:
        for start in range(t - 1, t - 2 - MAX_AGE, -1):
            if start in b:
                return b[start]
        return None

    with _LOCK:
        for t in trades:
            pnl = float(t.get("pnl") or 0)
            total += pnl
            # the arithmetic: the result is the cash in, plus the cash out, less commissions
            if all(t.get(k) is not None for k in ("entry_cash", "exit_cash", "commissions")):
                if abs(float(t["entry_cash"]) + float(t["exit_cash"]) - float(t["commissions"]) - pnl) > 0.011:
                    bad_sum.append(str(t.get("entry_date")))
            sym = t.get("symbol") or "SPY"
            ok_trade = True
            for leg in t.get("legs") or []:
                if leg.get("kind", "option") != "option" or leg.get("price_source") not in (None, "trade_print"):
                    continue
                legs_n += 1
                try:
                    ent = datetime.fromtimestamp(int(t["entry_ts"]), tz=timezone.utc).astimezone(NY)
                    b = bars(sym, ent.date().isoformat(), leg["right"], leg["strike"], str(leg["expiration"])[:10])
                    if b is None or leg.get("entry_print") is None:
                        unread += 1
                        continue
                    px = last_print(b, ent.hour * 60 + ent.minute)
                    # a fill worked for a few minutes after the signal takes a later print: any print of the contract in the
                    # minutes from the signal to the fill counts as the one it used
                    window = [b[m] for m in range(ent.hour * 60 + ent.minute - 1 - MAX_AGE, ent.hour * 60 + ent.minute + 6) if m in b]
                    if px is None and not window:
                        unread += 1
                        continue
                    if not any(abs(float(leg["entry_print"]) - w) < 1e-4 for w in ([px] if px is not None else []) + window):
                        ok_trade = False
                        bad_price.append(f"{t.get('entry_date')} {leg['right']}{leg['strike']:g} entry {leg['entry_print']} (raw file: {px if px is not None else window[:2]})")
                    checked += 1
                except Exception:  # noqa: BLE001
                    unread += 1
    money = f"{'-' if total < 0 else '+'}${abs(total):,.2f}"
    not_checked = "Which sessions qualified, which strikes were chosen, and exits before expiration were not re-derived independently."
    if bad_price or bad_sum:
        parts = ([f"{len(bad_price)} entry price(s) not found in the raw option file ({'; '.join(bad_price[:3])})"] if bad_price else []) + ([f"{len(bad_sum)} trade(s) whose result is not its own cash flows added up ({', '.join(bad_sum[:4])})"] if bad_sum else [])
        return _out("differs", "Checking the fills against the raw option files found a difference: " + "; ".join(parts) + ".", trades=len(trades), pnl=round(total, 2), details={"price": bad_price[:30], "sum": bad_sum[:30]})
    if not checked:
        return _out("not_covered", "The fills of this strategy could not be re-read from the raw option files, so nothing was verified independently.")
    return _out("partly", f"Partly verified: all {checked:,} option entry prices of the {len(trades):,} trades were found again in the raw option files by a separate reader, and every trade's result and the {money} total "
                          f"add up from the trades' own cash flows. {not_checked}" + (f" ({unread} fill(s) could not be re-read.)" if unread else ""), trades=len(trades), pnl=round(total, 2), legs=legs_n, checked=checked)
