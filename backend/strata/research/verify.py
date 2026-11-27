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
    return {"status": status, "text": text, "method": "a second calculation written apart from the engines, in SQL on the stored files (strata.research.reference)", **kw}


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
