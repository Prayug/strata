"""A second simulator, deliberately small, that shares no code with `strata.engine`.

The engine is large: clocks, liquidity, margin, assignment, scaling, rolling, hedging. Tests
that only ask it to agree with itself cannot see a mistake in what it believes a fill is.
This module prices one narrow family of strategies from raw observations with plain loops:

  one position at a time, one entry per session, one or two option legs with the same
  expiration (strike = the listed strike nearest the underlying, plus an offset in points, or
  a fixed strike), entered at a clock time, closed by a premium target, a premium stop, an
  underlying move for or against, a clock time, or cash settlement at expiration

It reads the same stored observations and applies the same STATED execution assumptions,
re-derived here from their description rather than copied:

  clock      decisions happen at the open, at the end of every completed underlying bar on
             the signal grid, at the close, at the entry and time-exit clock times, and at
             every option quote snapshot
  underlying the price known at time T is the close of the last bar that ENDED by T; at the
             open it is the first bar's open
  quotes     an order fills at the first decision time at which a quote snapshot exists that
             is no older than `max_quote_age` and was observed at or after the signal (or
             was still that fresh at the signal). Buy at the ask, sell at the bid
             ("conservative"), or at the midpoint. No fill within `max_fill_delay`: expired
  prints     where only trade prints exist, the price is the close of the contract's last
             bar that ended by T, usable only while it is younger than `max_trade_age`
  exits      checked from the decision time AFTER the entry. Premium rules compare what the
             position would fetch if closed now (under the fill model) with the premium paid
             or received, and need a fresh price for every open leg; time and underlying rules
             do not, but a close does:
             without one it is deferred to the next fresh price. If none comes before the
             contract expires, the position ends by settlement; the rule still names the exit
  fees       commission per contract on every fill; none on settlement at expiration

`simulate` returns plain trade records. `compare` lines them up with the engine's. Where
the two disagree, one of them is wrong, and neither is assumed to be the right one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
PRIORITY = ["stop_loss", "underlying_stop", "time_exit", "underlying_target", "profit_target"]  # which rule names the exit when several fire at once


@dataclass
class Leg:
    side: str  # buy | sell
    right: str  # C | P
    dte: int = 0
    offset: float = 0.0  # points from the at-the-money strike
    strike: float | None = None  # a fixed strike instead of an offset
    quantity: int = 1


@dataclass
class Spec:
    legs: list[Leg]
    entry: str = "open"  # "open" or "HH:MM" New York
    time_exit: str | None = None
    profit_target_pct: float | None = None  # % of the premium paid or received
    stop_loss_pct: float | None = None
    underlying_for_pct: float | None = None  # % move of the underlying in the position's favour, from its price at entry
    underlying_against_pct: float | None = None
    fill: str = "conservative"
    commission: float = 0.65
    max_quote_age: int = 120
    max_trade_age: int = 300
    max_fill_delay_min: float = 5.0
    signal_interval_min: int = 5
    multiplier: float = 100.0
    observation: str = "quote"  # quote | trade
    cash_settled: bool = True


@dataclass
class Session:
    """One session's raw observations. bars: (start, end, open, high, low, close) in epoch seconds, ascending.
    quotes: snapshot time -> {(expiration, strike, right): (bid, ask, quote_time)}. prints: (expiration, strike, right) -> [(bar_end, close)]
    ascending. listed: the contracts known to exist on the session (for prints: everything stored for it, traded yet or not)."""
    day: date
    open_t: int
    close_t: int
    bars: list[tuple]
    quotes: dict[int, dict] = field(default_factory=dict)
    prints: dict[tuple, list] = field(default_factory=dict)
    listed: set = field(default_factory=set)


def _hhmm(d: date, s: str) -> int:
    h, m = map(int, s.split(":"))
    return int(datetime(d.year, d.month, d.day, h, m, tzinfo=NY).timestamp())


def _et(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).astimezone(NY).isoformat()


def price_at(s: Session, t: int) -> float:
    last = None
    for b in s.bars:
        if b[1] <= t:
            last = b
        else:
            break
    if last is not None:
        return float(last[5])
    return float(s.bars[0][2]) if s.bars and s.bars[0][0] <= t else math.nan


def decision_times(s: Session, spec: Spec) -> list[int]:
    step = max(60, spec.signal_interval_min * 60)
    out = {s.open_t}
    for b in s.bars:
        res = b[1] - b[0]
        grid = max(step, res)
        if s.open_t < b[1] <= s.close_t and ((b[1] - s.open_t) % grid == 0 or b[1] == s.close_t):
            out.add(int(b[1]))
    res = (s.bars[0][1] - s.bars[0][0]) if s.bars else 86400
    for clock in (spec.entry, spec.time_exit):
        if clock and clock != "open" and res < 86400:
            t = _hhmm(s.day, clock)
            if s.open_t <= t <= s.close_t and (t - s.open_t) % res == 0:
                out.add(t)
    out |= {t for t in s.quotes if s.open_t <= t <= s.close_t}
    return sorted(out)


def _observe(s: Session, spec: Spec, key: tuple, t: int):
    """(bid, ask, observed_at) of a contract as it can be known at t, or None. For prints bid = ask = the last print."""
    if spec.observation == "trade":
        last = None
        for end, px in s.prints.get(key, ()):
            if end <= t:
                last = (px, px, end)
            else:
                break
        return last
    snaps = [x for x in s.quotes if x <= t]
    if not snaps:
        return None
    q = s.quotes[max(snaps)].get(key)
    return None if q is None else (q[0], q[1], q[2] if len(q) > 2 and q[2] is not None else max(snaps))


def _fresh(spec: Spec, obs, t: int) -> bool:
    return obs is not None and t - obs[2] <= (spec.max_trade_age if spec.observation == "trade" else spec.max_quote_age)


def _fill(spec: Spec, side: str, bid: float, ask: float) -> float:
    if spec.observation == "trade":
        return ask  # a print: one price
    if spec.fill == "midpoint":
        return (bid + ask) / 2
    return ask if side == "buy" else bid


def _listed_strikes(s: Session, spec: Spec, exp: date, right: str, t: int) -> list[float]:
    if spec.observation == "trade":
        return sorted({k for (e, k, r) in s.listed if e == exp and r == right})
    snaps = [x for x in s.quotes if x <= t]
    return sorted({k for (e, k, r) in s.quotes[max(snaps)] if e == exp and r == right}) if snaps else []


def simulate_session(s: Session, spec: Spec) -> dict:
    """One session. Returns {"trade": record | None, "signal": bool, "skipped": reason | None}."""
    times = decision_times(s, spec)
    entry_t = s.open_t if spec.entry == "open" else _hhmm(s.day, spec.entry)
    if entry_t not in times:
        return {"trade": None, "signal": False, "skipped": "the entry time is not a decision time on this session's bars"}
    exit_t = _hhmm(s.day, spec.time_exit) if spec.time_exit else None
    if exit_t is not None:
        exit_t = min(exit_t, s.close_t)  # an early close: a later time exit fires at that session's close
    mult = spec.multiplier
    pos = None
    pending_exit = None
    for i, t in enumerate(times):
        if t < entry_t:
            continue
        spot = price_at(s, t)
        last = i == len(times) - 1
        if pos is None:
            # ---- the order is working from the signal until it fills or runs out of time
            waited = t - entry_t >= spec.max_fill_delay_min * 60 or last
            if spec.observation == "quote":
                snaps = [x for x in s.quotes if x <= t]
                if not s.quotes:
                    return {"trade": None, "signal": True, "skipped": "no_option_data"}
                snap = max(snaps) if snaps else None
                ok = snap is not None and t - snap <= spec.max_quote_age and (snap >= entry_t or entry_t - snap <= spec.max_quote_age)
            else:
                if not s.prints:
                    return {"trade": None, "signal": True, "skipped": "no_option_data"}
                ok = True
            if ok:
                legs, why = [], None
                for lg in spec.legs:
                    exp = s.day + timedelta(days=lg.dte)
                    ks = _listed_strikes(s, spec, exp, lg.right, t)
                    if not ks:
                        why = "no_contract"
                        break
                    if lg.strike is not None:
                        k = min(ks, key=lambda x: abs(x - lg.strike))
                    else:
                        atm = min(ks, key=lambda x: (abs(x - spot), x))
                        k = min(ks, key=lambda x: abs(x - (atm + lg.offset)))
                        if abs(k - (atm + lg.offset)) > 1e-6:
                            why = "no_contract"
                            break
                    obs = _observe(s, spec, (exp, k, lg.right), t)
                    if not _fresh(spec, obs, t) or not all(math.isfinite(x) for x in obs[:2]):
                        why = "no_recent_trade" if spec.observation == "trade" else "no_contract"
                        break
                    legs.append({"side": lg.side, "right": lg.right, "strike": k, "expiration": exp, "qty": lg.quantity, "entry": _fill(spec, lg.side, obs[0], obs[1])})
                if why is None:
                    cash = sum((-1 if x["side"] == "buy" else 1) * x["qty"] * mult * x["entry"] for x in legs)
                    fees = sum(x["qty"] for x in legs) * spec.commission
                    pos = {"entry_t": t, "legs": legs, "entry_cash": cash, "fees": fees, "entry_spot": spot, "signal_t": entry_t}
                    continue
                if spec.observation == "quote" or waited:
                    return {"trade": None, "signal": True, "skipped": why}
                continue  # prints: keep the order working, the contract may print in the next minutes
            if waited:
                return {"trade": None, "signal": True, "skipped": "no_recent_trade" if spec.observation == "trade" else "no_quote"}
            continue
        # ---- an open position: which rule fires at t
        obs = [_observe(s, spec, (x["expiration"], x["strike"], x["right"]), t) for x in pos["legs"]]
        fresh = all(_fresh(spec, o, t) for o in obs)
        hits = []
        if pending_exit:
            hits = [pending_exit]
        else:
            if fresh:
                # what the position would fetch if it were closed now, under the same fill model as a real close
                value = sum((1 if x["side"] == "buy" else -1) * x["qty"] * mult * _fill(spec, "sell" if x["side"] == "buy" else "buy", o[0], o[1]) for x, o in zip(pos["legs"], obs))
                pnl = pos["entry_cash"] + value
                basis = abs(pos["entry_cash"])
                pct = pnl / basis * 100 if basis > 1e-9 else math.nan
                if spec.profit_target_pct is not None and pct >= spec.profit_target_pct:
                    hits.append("profit_target")
                if spec.stop_loss_pct is not None and pct <= -spec.stop_loss_pct:
                    hits.append("stop_loss")
            if exit_t is not None and t >= exit_t:
                hits.append("time_exit")
            # the direction of the position, from what it is: long calls and short puts gain when the underlying rises
            sign = sum((1 if x["side"] == "buy" else -1) * (1 if x["right"] == "C" else -1) * x["qty"] for x in pos["legs"])
            if sign != 0 and math.isfinite(spot):
                mv = (spot / pos["entry_spot"] - 1) * 100 * (1 if sign > 0 else -1)
                if spec.underlying_for_pct is not None and mv >= spec.underlying_for_pct:
                    hits.append("underlying_target")
                if spec.underlying_against_pct is not None and -mv >= spec.underlying_against_pct:
                    hits.append("underlying_stop")
        if hits:
            reason = min(hits, key=lambda h: PRIORITY.index(h))
            tradable = all(t < int(datetime(x["expiration"].year, x["expiration"].month, x["expiration"].day, 16, 0, tzinfo=NY).timestamp()) if x["expiration"] != s.day else t < s.close_t for x in pos["legs"])
            if tradable and fresh:
                return {"trade": _close(pos, s, spec, t, reason, obs), "signal": True, "skipped": None}
            pending_exit = reason  # no fresh price to close at: carried to the next one (or to settlement)
    if pos is None:
        return {"trade": None, "signal": True, "skipped": "no_quote"}
    # ---- still open at the close
    if all(x["expiration"] == s.day for x in pos["legs"]) and spec.cash_settled:
        und = float(s.bars[-1][5])
        value = 0.0
        for x in pos["legs"]:
            intrinsic = max(0.0, und - x["strike"]) if x["right"] == "C" else max(0.0, x["strike"] - und)
            x["exit"] = intrinsic
            value += (1 if x["side"] == "buy" else -1) * x["qty"] * mult * intrinsic
        # a rule that had decided to exit but never got a fresh price names the exit; the position still ended by settlement
        rec = _record(pos, s, s.close_t, pending_exit or "expiration", value, 0.0)
        rec["settled_after_deferred_exit"] = bool(pending_exit)
        return {"trade": rec, "signal": True, "skipped": None}
    return {"trade": {**_record(pos, s, None, "open", 0.0, 0.0), "open": True}, "signal": True, "skipped": None}


def _close(pos: dict, s: Session, spec: Spec, t: int, reason: str, obs: list) -> dict:
    value, fees = 0.0, 0.0
    for x, o in zip(pos["legs"], obs):
        px = _fill(spec, "sell" if x["side"] == "buy" else "buy", o[0], o[1])
        x["exit"] = px
        value += (1 if x["side"] == "buy" else -1) * x["qty"] * spec.multiplier * px
        fees += x["qty"] * spec.commission
    return _record(pos, s, t, reason, value, fees)


def _record(pos: dict, s: Session, t: int | None, reason: str, exit_cash: float, exit_fees: float) -> dict:
    fees = pos["fees"] + exit_fees
    return {"date": s.day.isoformat(), "signal_time": _et(pos["signal_t"]), "entry_time": _et(pos["entry_t"]), "exit_time": _et(t) if t is not None else None, "exit_reason": reason, "entry_spot": pos["entry_spot"],
            "entry_cash": pos["entry_cash"], "exit_cash": exit_cash, "fees": fees, "pnl": pos["entry_cash"] + exit_cash - fees,
            "legs": [{"side": x["side"], "right": x["right"], "strike": x["strike"], "expiration": x["expiration"].isoformat(), "entry_price": x["entry"], "exit_price": x.get("exit")} for x in pos["legs"]]}


def simulate(sessions: list[Session], spec: Spec) -> dict:
    trades, skipped = [], []
    for s in sessions:
        out = simulate_session(s, spec)
        if out["trade"]:
            trades.append(out["trade"])
        elif out["signal"]:
            skipped.append({"date": s.day.isoformat(), "why": out["skipped"]})
    return {"trades": trades, "skipped": skipped}


# ----------------------------------------------------------------------------- reading raw observations from a store
def load_session(store, symbol: str, d: date, observation: str = "quote", provider: str | None = None, underlying_provider: str | None = None, partial: bool = False) -> Session | None:
    """One session's observations read straight from the stored files. With `partial`, a session whose intraday bars
    cover only part of the day (a vendor sample) is loaded as far as it was observed, and `close_t` is the end of the last bar."""
    import polars as pl

    from strata.core import calendar as cal
    o, c = cal.session_bounds(d)
    ot, ct = int(o.timestamp()), int(c.timestamp())
    bars = None
    if underlying_provider or partial:
        for res, sec in (("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800), ("60m", 3600)):
            b = store.read_bars(symbol, res, d, d, with_provider=True)
            if b.is_empty():
                continue
            if underlying_provider:
                b = b.filter(pl.col("provider") == underlying_provider)
            b = b.with_columns(pl.col("ts").dt.epoch("s").alias("_t")).filter((pl.col("_t") >= ot) & (pl.col("_t") < ct)).sort("_t")
            if not b.is_empty():
                bars = [(int(t), min(int(t) + sec, ct), float(a), float(h), float(l), float(cl)) for t, a, h, l, cl in zip(b["_t"], b["open"], b["high"], b["low"], b["close"])]
                break
    if bars is None:
        b = store.read_best_bars(symbol, d, d)
        if b.is_empty():
            return None
        bars = [(int(t.timestamp()), int(e.timestamp()), float(a), float(h), float(l), float(cl)) for t, e, a, h, l, cl in zip(b["ts"], b["end"], b["open"], b["high"], b["low"], b["close"])]
    s = Session(d, ot, ct if not partial else max(x[1] for x in bars), bars)
    if observation == "trade":
        ds = store.option_dataset(symbol, "trade", provider)
        df = store.read_option_bars(symbol, d, dataset=ds) if ds is not None else None
        if df is not None and not df.is_empty():
            df = df.with_columns((pl.col("ts").dt.epoch("s") + 60).alias("_end")).sort(["occ", "_end"])
            for e, k, r, end, px in zip(df["expiration"], df["strike"], df["right"], df["_end"], df["close"]):
                s.prints.setdefault((e, float(k), r), []).append((min(int(end), ct), float(px)))
            s.listed = set(s.prints)
            try:  # contracts that were downloaded for the session but never printed are listed too
                idx = store.option_bar_index(symbol, dataset=ds)
                from strata.data.providers.sources import parse_occ
                for occ in idx.filter(pl.col("date") == d)["occ"].to_list():
                    _root, e, r, k = parse_occ(occ)
                    s.listed.add((e, float(k), r))
            except Exception:  # noqa: BLE001
                pass
    else:
        ds = store.option_dataset(symbol, "quote", provider)
        df = store.read_option_day(symbol, d, "quote", dataset=ds) if ds is not None else None
        if df is not None and not df.is_empty():
            df = df.with_columns(pl.col("ts").dt.epoch("s").alias("_t"), (pl.col("quote_ts").dt.epoch("s") if "quote_ts" in df.columns else pl.col("ts").dt.epoch("s")).alias("_q"))
            for t, e, k, r, bid, ask, q in zip(df["_t"], df["expiration"], df["strike"], df["right"], df["bid"], df["ask"], df["_q"]):
                if bid is None or ask is None:
                    continue
                s.quotes.setdefault(int(t), {})[(e, float(k), r)] = (float(bid), float(ask), int(q) if q is not None else int(t))
    return s


def compare(engine_trades: list[dict], ref_trades: list[dict], tol: float = 0.005) -> dict:
    """Line the two up by entry session. A difference of more than half a cent in a price, or any difference in contract, time or reason, is reported."""
    by = {t["date"]: t for t in ref_trades}
    diffs, matched = [], 0
    seen = set()
    for e in engine_trades:
        d = e["entry_date"]
        seen.add(d)
        r = by.get(d)
        if r is None:
            diffs.append({"date": d, "what": "the engine traded, the reference did not"})
            continue
        problems = []
        if str(e["entry_time"])[:19] != str(r["entry_time"])[:19]:
            problems.append(f"entry time {str(e['entry_time'])[11:19]} vs {str(r['entry_time'])[11:19]}")
        if r.get("exit_time") and str(e["exit_time"])[:19] != str(r["exit_time"])[:19]:
            problems.append(f"exit time {str(e['exit_time'])[11:19]} vs {str(r['exit_time'])[11:19]}")
        if e["exit_reason"] != r["exit_reason"]:
            problems.append(f"exit reason {e['exit_reason']} vs {r['exit_reason']}")
        el = sorted(((l["right"], float(l["strike"]), l["expiration"], l["side"], float(l["entry_price"]), float(l["exit_price"] if l["exit_price"] is not None else math.nan)) for l in e["legs"] if l.get("kind", "option") == "option"))
        rl = sorted(((l["right"], float(l["strike"]), l["expiration"], l["side"], float(l["entry_price"]), float(l["exit_price"] if l["exit_price"] is not None else math.nan)) for l in r["legs"]))
        if [x[:4] for x in el] != [x[:4] for x in rl]:
            problems.append(f"contracts {[x[:3] for x in el]} vs {[x[:3] for x in rl]}")
        else:
            for a, b in zip(el, rl):
                if abs(a[4] - b[4]) > tol:
                    problems.append(f"{a[0]}{a[1]:g} entry price {a[4]:.4f} vs {b[4]:.4f}")
                if math.isfinite(a[5]) and math.isfinite(b[5]) and abs(a[5] - b[5]) > tol:
                    problems.append(f"{a[0]}{a[1]:g} exit price {a[5]:.4f} vs {b[5]:.4f}")
        if abs(float(e["pnl"]) - float(r["pnl"])) > max(tol * 100, 0.01):
            problems.append(f"P&L {e['pnl']:.2f} vs {r['pnl']:.2f}")
        if bool(r.get("settled_after_deferred_exit")) != ("settled_after_deferred_exit" in (e.get("flags") or [])):
            problems.append("only one of the two says the position ended by settlement after an exit that could not be filled")
        if abs(float(e["commissions"]) - float(r["fees"])) > 0.005:
            problems.append(f"fees {e['commissions']:.2f} vs {r['fees']:.2f}")
        if problems:
            diffs.append({"date": d, "what": "; ".join(problems)})
        else:
            matched += 1
    for d in by:
        if d not in seen:
            diffs.append({"date": d, "what": "the reference traded, the engine did not"})
    return {"engine_trades": len(engine_trades), "reference_trades": len(ref_trades), "matched": matched, "differences": diffs, "agree": not diffs}
