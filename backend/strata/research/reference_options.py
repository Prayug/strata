"""An independent re-pricing of simple option backtests from the stored option trade bars (see reference.py for the idea).

It covers one long call or put per signal, with a fixed number of contracts or a dollar budget per trade, a clock entry and
a clock exit, priced on Alpaca's one-minute option TRADE bars (prints, never quotes). The definitions are the execution
assumptions the application documents (strata.engine.models.ExecutionConfig defaults), restated here:

  price at T          the close of the contract's last one-minute bar that ENDS at or before T and no more than 300 seconds
                      before it (an older print is stale and is not used); no slippage is added to a print by default
  working order       an entry with no usable print keeps working, minute by minute, for up to 5 minutes after the signal
  strike (at the money)  the listed strike nearest the underlying's price at that minute (the lower one on a tie), where
                      "listed" means it has printed at least once before that minute; it must have a usable print, else the
                      order keeps working (another strike is never substituted)
  expiration          0 DTE: the session's own expiration; N DTE: the nearest listed expiration at least N days out
  underlying price    the close of the underlying's last one-minute bar before T (reference.py)
  exit                the same price rule at the exit time, minute by minute until a print is found; a 0 DTE contract with
                      no print left is settled at its intrinsic value on the official close. An exit time after an early
                      close (13:00) is the close: a 0 DTE contract then goes at its intrinsic value against the
                      underlying's last minute price
  sessions            a session the provider holds only in part (sessions.json: "partial") is not used
  commissions         0.65 per contract per side
  contracts           fixed, or floor(budget / (price x 100)), or floor(risk budget / risk of one contract) where that risk is
                      the premium (no stop); a trade whose single contract is over the budget is not taken (and is counted)
"""
from __future__ import annotations

import math
from datetime import date, timedelta

COMMISSION = 0.65
MAX_AGE = 300
WORK_MINUTES = 5


def _m(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def option_backtest(args: dict, R) -> dict:
    sym = args["symbol"].upper()
    right = "C" if args.get("right", "call").lower().startswith("c") else "P"
    entry, exit_ = _m(args["entry"]), _m(args.get("exit", "15:55"))
    dte = int(args.get("dte", 0))
    R.load(sym)
    root = R.root / "alpaca" / sym / "options" / "trade" / "1m"
    vdir = sorted(root.glob("v*"))
    if not vdir:
        raise LookupError(f"no option trade bars stored for {sym}")
    data = vdir[-1] / "data"
    import json
    try:
        states = json.loads((vdir[-1] / "sessions.json").read_text())
        states = states.get("sessions", states) if isinstance(states, dict) else {}
    except (OSError, ValueError):
        states = {}
    partial = {d for d, v in states.items() if (v.get("status") if isinstance(v, dict) else v) == "partial"}
    conds = []
    if args.get("prev") is not None:
        p = float(args["prev"])
        conds.append(f"s.prev_ret {'<=' if p < 0 else '>='} {p}")
    where = " and ".join(conds + ["s.has_minute"]) or "true"
    days = [d for (d,) in R.q(f"select s.date from s_{sym} s where {where} and s.prev2_close is not null order by s.date")]
    trades, skipped = [], {"no option data": [], "partial session": [], "no contract": [], "no usable print": [], "over budget": []}
    for d in days:
        f = data / str(d.year) / f"{d.isoformat()}.parquet"
        if not f.exists():
            skipped["no option data"].append(str(d))
            continue
        if d.isoformat() in partial:
            skipped["partial session"].append(str(d))
            continue
        t = _one(R, sym, d, f, right, entry, exit_, dte, args)
        if isinstance(t, str):
            skipped[t].append(str(d))
        else:
            trades.append(t)
    pnl = sum(t["pnl"] for t in trades)
    return {"trades": len(trades), "pnl": round(pnl, 2), "contracts_total": sum(t["contracts"] for t in trades), "signals": len(days), "per_trade": trades,
            "skipped": {k: v for k, v in skipped.items() if v}}


def _one(R, sym: str, d: date, f, right: str, entry: int, exit_: int, dte: int, args: dict):
    con = R.con
    exp_sql = f"expiration = DATE '{d.isoformat()}'" if dte == 0 else \
        f"expiration = (select min(expiration) from read_parquet('{f}') where expiration >= DATE '{(d + timedelta(days=dte)).isoformat()}')"
    rows = con.execute(f"""select strike, (extract(hour from timezone('America/New_York', ts)) * 60 + extract(minute from timezone('America/New_York', ts)))::int m, close
                           from read_parquet('{f}') where "right" = '{right}' and {exp_sql}""").fetchall()
    if not rows:
        return "no contract"
    bars: dict[float, dict[int, float]] = {}
    for k, m, c in rows:
        bars.setdefault(float(k), {})[int(m)] = float(c)
    strikes = sorted(bars)
    under = dict((int(m), float(c)) for m, c in R.q(f"select m, c from b_{sym} where date = DATE '{d.isoformat()}'"))

    def price(k: float, t: int) -> float | None:
        # the last bar ending at or before t (bar start <= t - 1), ending no more than MAX_AGE seconds before t
        bk = bars.get(k) or {}
        for start in range(t - 1, t - 2 - MAX_AGE // 60, -1):  # a bar ending exactly MAX_AGE seconds before t is still usable
            if start in bk:
                return bk[start]
        return None

    def spot(t: int) -> float | None:
        for start in range(t - 1, 569, -1):
            if start in under:
                return under[start]
        return None

    chosen = None
    for t in range(entry, entry + WORK_MINUTES + 1):
        s = spot(t)
        if s is None:
            continue
        listed = [k for k in strikes if any(m < t for m in bars[k])]
        if not listed:
            continue
        k = min(listed, key=lambda x: (abs(x - s), x))
        px = price(k, t)
        if px is not None:
            chosen = (t, k, px, s)
            break
    if not chosen:
        return "no usable print"
    t_in, k, p_in, s_in = chosen
    if args.get("dollars"):
        q = int(math.floor(float(args["dollars"]) / (p_in * 100)))
        if q < 1:
            return "over budget"
    elif args.get("risk"):
        # risk of one contract: the loss at the stop (premium x stop %) when there is one, else the whole premium
        per = p_in * 100 * (float(args["stop_pct"]) / 100 if args.get("stop_pct") else 1.0)
        q = int(math.floor(float(args["risk"]) / per))
        if q < 1:
            return "over budget"
    else:
        q = int(args.get("contracts", 1))
    p_out, t_out = None, None
    close_m = (max(under) + 1) if under else 960  # 13:00 on an early-close day: a later exit time is the close
    if exit_ > close_m and dte == 0:
        # an exit moved to an early close happens at the expiration instant, where the contract is worth its intrinsic value
        # against the underlying's last price (the application's rule; it differs from the last print by a cent or two)
        last = spot(close_m)
        if last is not None:
            p_out, t_out = max(0.0, (last - k) if right == "C" else (k - last)), close_m
    for t in ([] if p_out is not None else range(min(exit_, close_m), 961)):
        px = price(k, t)
        if px is not None:
            p_out, t_out = px, t
            break
    settled = False
    if p_out is None:
        close = R.q(f"select close from s_{sym} where date = DATE '{d.isoformat()}'")[0][0]
        p_out = max(0.0, (close - k) if right == "C" else (k - close))
        settled, t_out = True, 960
    gross = (p_out - p_in) * 100 * q
    fees = COMMISSION * q * (1 if settled else 2)
    return {"date": d.isoformat(), "strike": k, "entry_minute": t_in, "exit_minute": t_out, "entry_print": p_in, "exit_print": p_out, "spot": s_in, "contracts": q,
            "gross": round(gross, 2), "fees": round(fees, 2), "pnl": round(gross - fees, 2), "settled": settled}


def compare_trades(app_trades: list[dict], ref: dict) -> dict:
    """Trade by trade: the application's log against the re-pricing. Returns the matches and every difference, by date."""
    mine = {t["date"]: t for t in ref["per_trade"]}
    out = {"same": 0, "different": [], "only_app": [], "only_reference": []}
    seen = set()
    for t in app_trades:
        dte = str(t.get("entry_date"))
        seen.add(dte)
        r = mine.get(dte)
        if r is None:
            out["only_app"].append(dte)
            continue
        leg = (t.get("legs") or [{}])[0]
        diffs = []
        if abs(float(leg.get("strike") or 0) - r["strike"]) > 1e-6:
            diffs.append(f"strike {leg.get('strike')} vs {r['strike']}")
        if leg.get("entry_print") is not None and abs(float(leg["entry_print"]) - r["entry_print"]) > 1e-6:
            diffs.append(f"entry print {leg['entry_print']} vs {r['entry_print']}")
        if leg.get("exit_print") is not None and abs(float(leg["exit_print"]) - r["exit_print"]) > 1e-6:
            diffs.append(f"exit print {leg['exit_print']} vs {r['exit_print']}")
        if abs(float(t.get("pnl") or 0) - r["pnl"]) > 0.011:
            diffs.append(f"P&L {t.get('pnl')} vs {r['pnl']}")
        if diffs:
            out["different"].append({"date": dte, "what": diffs})
        else:
            out["same"] += 1
    out["only_reference"] = sorted(set(mine) - seen)
    return out
