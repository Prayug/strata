"""Strategy presets for the visual builder (each is an ordinary StrategyConfig dict).

Presets are built for a real symbol (SPY by default) and adapt to the option data that is
actually stored: with end-of-day chains the entry is the close and same-day (0DTE)
structures become weeklies, because end-of-day quotes cannot fill intraday orders."""
from __future__ import annotations


def leg(id, side, right, dte, method="delta", value=0.3, unit="points", ref=None, qty=1, group=None, kind="option"):
    d = {"id": id, "side": side, "kind": kind, "right": right, "quantity": qty, "dte": dte, "strike": {"method": method, "value": value, "unit": unit, "ref_leg": ref}}
    if group:
        d["group"] = group
    return d


def stock(id="stk", side="buy", shares=100):
    return {"id": id, "side": side, "kind": "stock", "right": "C", "quantity": shares, "dte": 0, "strike": {"method": "fixed", "value": 0}}


def entry_at(times="10:00"):
    return {"conditions": {"id": "root", "kind": "group", "op": "and", "children": [
        {"id": "c1", "kind": "cond", "type": "entry_times", "params": {"times": times}},
    ]}, "max_entries_per_day": 1, "max_open_positions": 1}


def exits(pt=50, sl=200, time_exit=None, dte_exit=None, dte_time="10:00"):
    e = {"profit_target": {"enabled": pt is not None, "mode": "pct", "value": pt or 50}, "stop_loss": {"enabled": sl is not None, "mode": "pct", "value": sl or 200}}
    if time_exit:
        e["time_exit"] = {"enabled": True, "time": time_exit, "dte_lte": 0}
    if dte_exit is not None:
        e["dte_exit"] = {"enabled": True, "dte": dte_exit, "time": dte_time}
    return e


def _adapt(cfg: dict, symbol: str, eod: bool, width_scale: float = 1.0) -> dict:
    """Point a preset at `symbol`; with end-of-day option data move it onto the close."""
    cfg = {**cfg, "symbol": symbol}
    legs = []
    for l in cfg["legs"]:
        l = {**l, "strike": dict(l["strike"])}
        if l["strike"]["method"] in ("leg_offset", "atm_offset") and l["strike"]["value"]:
            l["strike"]["value"] = round(l["strike"]["value"] * width_scale) or (1 if l["strike"]["value"] > 0 else -1)
        if eod and l.get("kind", "option") == "option" and l["dte"] < 7:
            l["dte"] = 7
        if l.get("kind", "option") == "option" and l["dte"] == 0:
            l["dte_mode"] = "exact"
        legs.append(l)
    cfg["legs"] = legs
    if eod:
        ex = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg["exits"].items()}
        if "time_exit" in ex:
            ex.pop("time_exit")
            ex.setdefault("dte_exit", {"enabled": True, "dte": 1, "time": "16:00"})
        if "dte_exit" in ex:
            ex["dte_exit"]["time"] = "16:00"
        cfg["exits"] = ex
        kids = []
        for c in cfg["entry"]["conditions"]["children"]:
            if c.get("type") == "entry_times":
                c = {**c, "params": {"times": "close"}}
            kids.append(c)
        cfg["entry"] = {**cfg["entry"], "conditions": {**cfg["entry"]["conditions"], "children": kids}}
    return cfg


def T(key, label, category, legs, dte_hint, description, ex=None, entry=None):
    return {"key": key, "label": label, "category": category, "description": description,
            "config": {"name": label, "symbol": "SPY", "legs": legs, "entry": entry or entry_at("10:00"), "exits": ex or exits(),
                       "sizing": {"method": "fixed", "contracts": 1}}, "dte_hint": dte_hint}


def templates(symbol: str = "SPY", eod: bool = False, width_scale: float = 1.0) -> list[dict]:
    out = []
    for t in _templates():
        cfg = _adapt(t["config"], symbol, eod, width_scale)
        dte = min((l["dte"] for l in cfg["legs"] if l.get("kind", "option") == "option"), default=t["dte_hint"])
        note = " End-of-day option data: enters at the close, weekly instead of same-day expiry." if eod and t["dte_hint"] < 7 else (" Enters at the close (end-of-day option data)." if eod else "")
        out.append({**t, "config": cfg, "dte_hint": dte, "description": t["description"] + note})
    return out


def _templates() -> list[dict]:
    return [
        T("long_call", "Long call", "Single leg", [leg("lc", "buy", "C", 30, value=0.5)], 30, "Buy a call: bullish, defined risk (debit).", exits(100, 50, dte_exit=7)),
        T("long_put", "Long put", "Single leg", [leg("lp", "buy", "P", 30, value=0.5)], 30, "Buy a put: bearish, defined risk (debit).", exits(100, 50, dte_exit=7)),
        T("short_put", "Short put (naked)", "Single leg", [leg("sp", "sell", "P", 30, value=0.2)], 30, "Sell an OTM put: bullish/neutral, Reg-T naked margin.", exits(50, 200, dte_exit=7)),
        T("short_call", "Short call (naked)", "Single leg", [leg("sc", "sell", "C", 30, value=0.16)], 30, "Sell an OTM call: bearish/neutral, undefined risk.", exits(50, 200, dte_exit=7)),
        T("covered_call", "Covered call", "Income", [stock(), leg("sc", "sell", "C", 30, value=0.3)], 30, "Long 100 shares + short OTM call.", exits(None, None)),
        T("csp", "Cash-secured put", "Income", [leg("sp", "sell", "P", 30, value=0.25)], 30, "Short put fully collateralised (use the cash margin model).", exits(50, None, dte_exit=7)),
        T("bull_call", "Bull call spread", "Vertical", [leg("lc", "buy", "C", 14, value=0.5), leg("sc", "sell", "C", 14, "leg_offset", 5, ref="lc")], 14, "Debit call vertical.", exits(80, 50, dte_exit=1)),
        T("bear_put", "Bear put spread", "Vertical", [leg("lp", "buy", "P", 14, value=0.5), leg("sp", "sell", "P", 14, "leg_offset", -5, ref="lp")], 14, "Debit put vertical.", exits(80, 50, dte_exit=1)),
        T("bull_put", "Bull put credit spread", "Vertical", [leg("sp", "sell", "P", 0, value=0.15), leg("lp", "buy", "P", 0, "leg_offset", -5, ref="sp")], 0, "Credit put vertical (0DTE by default).", exits(50, 200)),
        T("bear_call", "Bear call credit spread", "Vertical", [leg("sc", "sell", "C", 0, value=0.15), leg("lc", "buy", "C", 0, "leg_offset", 5, ref="sc")], 0, "Credit call vertical (0DTE by default).", exits(50, 200)),
        T("iron_condor", "Iron condor", "Neutral", [leg("sp", "sell", "P", 0, value=0.12, group="put_side"), leg("lp", "buy", "P", 0, "leg_offset", -5, ref="sp", group="put_side"),
                                                       leg("sc", "sell", "C", 0, value=0.12, group="call_side"), leg("lc", "buy", "C", 0, "leg_offset", 5, ref="sc", group="call_side")], 0,
          "Short put spread + short call spread.", exits(50, 200)),
        T("iron_fly", "Iron butterfly", "Neutral", [leg("sp", "sell", "P", 0, "atm_offset", 0), leg("sc", "sell", "C", 0, "atm_offset", 0), leg("lp", "buy", "P", 0, "leg_offset", -10, ref="sp"), leg("lc", "buy", "C", 0, "leg_offset", 10, ref="sc")], 0,
          "Short ATM straddle with protective wings.", exits(25, 100)),
        T("long_straddle", "Long straddle", "Volatility", [leg("lc", "buy", "C", 7, "atm_offset", 0), leg("lp", "buy", "P", 7, "atm_offset", 0)], 7, "Buy ATM call + put.", exits(50, 40, dte_exit=1)),
        T("short_straddle", "Short straddle", "Volatility", [leg("sc", "sell", "C", 0, "atm_offset", 0), leg("sp", "sell", "P", 0, "atm_offset", 0)], 0, "Sell ATM call + put (undefined risk).", exits(25, 100)),
        T("long_strangle", "Long strangle", "Volatility", [leg("lc", "buy", "C", 7, value=0.25), leg("lp", "buy", "P", 7, value=0.25)], 7, "Buy OTM call + put.", exits(80, 50, dte_exit=1)),
        T("short_strangle", "Short strangle", "Volatility", [leg("sc", "sell", "C", 30, value=0.16), leg("sp", "sell", "P", 30, value=0.16)], 30, "Sell OTM call + put.", exits(50, 200, dte_exit=7)),
        T("call_calendar", "Call calendar", "Time spread", [leg("sc", "sell", "C", 7, "atm_offset", 0), leg("lc", "buy", "C", 35, "leg_offset", 0, ref="sc")], 7, "Short front ATM call, long back-month call at the same strike.", exits(25, 30, dte_exit=1)),
        T("put_calendar", "Put calendar", "Time spread", [leg("sp", "sell", "P", 7, "atm_offset", 0), leg("lp", "buy", "P", 35, "leg_offset", 0, ref="sp")], 7, "Short front ATM put, long back-month put at the same strike.", exits(25, 30, dte_exit=1)),
        T("diagonal", "Call diagonal", "Time spread", [leg("lc", "buy", "C", 35, value=0.7), leg("sc", "sell", "C", 7, value=0.3)], 7, "Long deep back-month call, short near-term OTM call.", exits(30, 40, dte_exit=1)),
        T("call_butterfly", "Long call butterfly", "Butterfly", [leg("lc1", "buy", "C", 7, "atm_offset", -5), leg("sc", "sell", "C", 7, "atm_offset", 0, qty=2), leg("lc2", "buy", "C", 7, "atm_offset", 5)], 7, "1-2-1 call butterfly centred at the money.", exits(100, 50, dte_exit=0)),
        T("bwb", "Broken-wing put butterfly", "Butterfly", [leg("lp1", "buy", "P", 14, value=0.35), leg("sp", "sell", "P", 14, "leg_offset", -5, ref="lp1", qty=2), leg("lp2", "buy", "P", 14, "leg_offset", -15, ref="lp1")], 14, "Put butterfly with a wider lower wing (often a credit).", exits(50, 100, dte_exit=1)),
        T("put_ratio", "Put ratio spread (1x2)", "Ratio", [leg("lp", "buy", "P", 14, value=0.4), leg("sp", "sell", "P", 14, "leg_offset", -5, ref="lp", qty=2)], 14, "Buy one put, sell two further OTM (one naked put).", exits(50, 150, dte_exit=1)),
        T("custom", "Custom", "Custom", [leg("l1", "sell", "P", 0, value=0.2)], 0, "Start from a single leg and build anything."),
    ]


def _c(type_, **params):
    return {"kind": "cond", "type": type_, "params": params}


def _entry(*conds, op="and", **kw):
    return {"conditions": {"kind": "group", "op": op, "children": list(conds)}, "max_entries_per_day": 1, "max_open_positions": 1, **kw}


def examples(symbol: str = "SPY", eod: bool = True) -> list[dict]:
    """Curated starting points on a real symbol. The first group runs on free end-of-day
    option chains; the second needs intraday option quotes and says so in its name."""
    def expr(e, label):
        return _c("expr", expr=e, label=label)
    daily = [
        {"name": f"{symbol} 30DTE put credit spread 20\u0394", "symbol": symbol,
         "legs": [leg("sp", "sell", "P", 30, value=0.20), leg("lp", "buy", "P", 30, "leg_offset", -5, ref="sp")],
         "entry": _entry(_c("entry_times", times="close"), _c("day_of_week", days=["Mon", "Wed"]), max_open_positions=4),
         "exits": exits(50, 200, dte_exit=7, dte_time="16:00")},
        {"name": f"{symbol} call after a 2% down day (5-session hold)", "symbol": symbol,
         "legs": [{**leg("lc", "buy", "C", 30, "atm_offset", 0)}],
         "entry": _entry(_c("entry_times", times="close"), expr("change_pct <= -2", "session fell 2% or more, measured at its close"), max_open_positions=5),
         "exits": {**exits(None, None), "max_hold": {"enabled": True, "days": 5}}},
        {"name": f"{symbol} 45DTE short strangle 16\u0394", "symbol": symbol,
         "legs": [leg("sp", "sell", "P", 45, value=0.16), leg("sc", "sell", "C", 45, value=0.16)],
         "entry": _entry(_c("entry_times", times="close"), _c("day_of_week", days=["Mon"]), max_open_positions=3),
         "exits": exits(50, 200, dte_exit=21, dte_time="16:00")},
        {"name": f"{symbol} 30DTE iron condor 16\u0394 when VIX > 20", "symbol": symbol,
         "legs": [leg("sp", "sell", "P", 30, value=0.16, group="put_side"), leg("lp", "buy", "P", 30, "leg_offset", -10, ref="sp", group="put_side"),
                  leg("sc", "sell", "C", 30, value=0.16, group="call_side"), leg("lc", "buy", "C", 30, "leg_offset", 10, ref="sc", group="call_side")],
         "entry": _entry(_c("entry_times", times="close"), _c("day_of_week", days=["Fri"]), expr("vix > 20", "VIX above 20"), max_open_positions=3),
         "exits": exits(50, 200, dte_exit=7, dte_time="16:00")},
        {"name": f"{symbol} covered call 30\u0394 (monthly)", "symbol": symbol,
         "legs": [stock(), leg("sc", "sell", "C", 30, value=0.30)],
         "entry": _entry(_c("entry_times", times="close")),
         "exits": exits(80, None, dte_exit=2, dte_time="16:00")},
        {"name": f"{symbol} put after 3 up days above the 50-day average", "symbol": symbol,
         "legs": [leg("lp", "buy", "P", 14, "atm_offset", 0)],
         "entry": _entry(_c("entry_times", times="close"), expr('consecutive_up_days() >= 3 and price > sma(50, "1d")', "three straight up closes while above the 50-day average"), max_open_positions=2),
         "exits": {**exits(100, 50), "max_hold": {"enabled": True, "days": 5}}},
    ]
    intraday = [
        {"name": f"{symbol} 0DTE iron condor 12\u0394 (needs intraday option quotes)", "symbol": symbol,
         "legs": [{**leg("sp", "sell", "P", 0, value=0.12, group="put_side"), "dte_mode": "exact"}, {**leg("lp", "buy", "P", 0, "leg_offset", -5, ref="sp", group="put_side"), "dte_mode": "exact"},
                  {**leg("sc", "sell", "C", 0, value=0.12, group="call_side"), "dte_mode": "exact"}, {**leg("lc", "buy", "C", 0, "leg_offset", 5, ref="sc", group="call_side"), "dte_mode": "exact"}],
         "entry": _entry(_c("entry_times", times="10:00"), _c("vix", cmp="<", value=30), _c("event", types=["FOMC", "CPI"], when="today", mode="exclude")),
         "exits": {**exits(50, 200), "time_exit": {"enabled": True, "time": "15:45", "dte_lte": 0}}},
        {"name": f"{symbol} 0DTE call after a 3% down day, +1% underlying target (needs intraday option quotes)", "symbol": symbol,
         "legs": [{**leg("lc", "buy", "C", 0, "atm_offset", 0), "dte_mode": "exact"}],
         "entry": _entry(_c("entry_times", times="open"), expr("previous_session.return_pct <= -3", "previous session fell 3% or more (close-to-close)")),
         "exits": {**exits(None, None), "underlying": {"enabled": True, "move_for_pct": 1.0}, "time_exit": {"enabled": True, "time": "15:45", "dte_lte": None}}},
    ]
    return daily + intraday
