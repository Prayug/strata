"""Strategy presets for the visual builder (each is an ordinary StrategyConfig dict)."""
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


def exits(pt=50, sl=200, time_exit=None, dte_exit=None):
    e = {"profit_target": {"enabled": pt is not None, "mode": "pct", "value": pt or 50}, "stop_loss": {"enabled": sl is not None, "mode": "pct", "value": sl or 200}}
    if time_exit:
        e["time_exit"] = {"enabled": True, "time": time_exit, "dte_lte": 0}
    if dte_exit is not None:
        e["dte_exit"] = {"enabled": True, "dte": dte_exit, "time": "10:00"}
    return e


def T(key, label, category, legs, dte_hint, description, ex=None, entry=None):
    return {"key": key, "label": label, "category": category, "description": description,
            "config": {"name": label, "symbol": "DEMO", "legs": legs, "entry": entry or entry_at("10:00"), "exits": ex or exits(),
                       "sizing": {"method": "fixed", "contracts": 1}}, "dte_hint": dte_hint}


def templates() -> list[dict]:
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


def examples() -> list[dict]:
    """Curated, ready-to-run strategies on the synthetic demo symbols."""
    return [
        {"name": "0DTE iron condor 12Δ", "symbol": "DEMO",
         "legs": [leg("sp", "sell", "P", 0, value=0.12, group="put_side"), leg("lp", "buy", "P", 0, "leg_offset", -5, ref="sp", group="put_side"),
                  leg("sc", "sell", "C", 0, value=0.12, group="call_side"), leg("lc", "buy", "C", 0, "leg_offset", 5, ref="sc", group="call_side")],
         "entry": _entry(_c("entry_times", times="10:00"), _c("vix", cmp="<", value=30), _c("event", types=["FOMC", "CPI"], when="today", mode="exclude")),
         "exits": {**exits(50, 200), "time_exit": {"enabled": True, "time": "15:45", "dte_lte": 0}}},
        {"name": "30DTE put credit spread 20Δ", "symbol": "DEMO",
         "legs": [leg("sp", "sell", "P", 30, value=0.20), leg("lp", "buy", "P", 30, "leg_offset", -5, ref="sp")],
         "entry": _entry(_c("entry_times", times="10:00"), _c("day_of_week", days=["Mon", "Wed"]), _c("iv_rank", cmp=">", value=20, lookback=252), max_open_positions=4),
         "exits": exits(50, 200, dte_exit=7)},
        {"name": "45DTE short strangle 16Δ", "symbol": "DEMO",
         "legs": [leg("sp", "sell", "P", 45, value=0.16), leg("sc", "sell", "C", 45, value=0.16)],
         "entry": _entry(_c("entry_times", times="10:30"), _c("day_of_week", days=["Mon"]), _c("regime", trend="any", vol="normal"), max_open_positions=3),
         "exits": exits(50, 200, dte_exit=21)},
        {"name": "Covered call 30Δ (monthly)", "symbol": "DEMO",
         "legs": [stock(), leg("sc", "sell", "C", 30, value=0.30)],
         "entry": _entry(_c("entry_times", times="10:00")),
         "exits": exits(80, None, dte_exit=2)},
        {"name": "DEMOX 0DTE iron fly (quiet open)", "symbol": "DEMOX",
         "legs": [leg("sp", "sell", "P", 0, "atm_offset", 0), leg("sc", "sell", "C", 0, "atm_offset", 0), leg("lp", "buy", "P", 0, "leg_offset", -50, ref="sp"), leg("lc", "buy", "C", 0, "leg_offset", 50, ref="sc")],
         "entry": _entry(_c("entry_times", times="10:30"), _c("gap", cmp="<", value=0.5), _c("gap", cmp=">", value=-0.5), _c("day_range", cmp="<", value=0.8)),
         "exits": {**exits(25, 100), "time_exit": {"enabled": True, "time": "15:30", "dte_lte": 0}}},
        {"name": "ATM call calendar 7/35", "symbol": "DEMO",
         "legs": [leg("sc", "sell", "C", 7, "atm_offset", 0), leg("lc", "buy", "C", 35, "leg_offset", 0, ref="sc")],
         "entry": _entry(_c("entry_times", times="11:00"), _c("day_of_week", days=["Mon", "Tue"]), _c("iv", cmp="<", value=22), max_open_positions=2),
         "exits": exits(25, 30, dte_exit=1)},
        {"name": "ORB breakout 0DTE bull call spread", "symbol": "DEMO",
         "legs": [leg("lc", "buy", "C", 0, "atm_offset", 0), leg("sc", "sell", "C", 0, "leg_offset", 2, ref="lc")],
         "entry": _entry(_c("time_window", start="10:00", end="12:00"), _c("opening_range", minutes=30, direction="breakout_up", buffer_pct=0.05),
                         _c("compare", lhs={"kind": "price"}, cmp=">", rhs={"kind": "indicator", "name": "vwap", "params": {}, "tf": "1m"})),
         "exits": {**exits(80, 50), "time_exit": {"enabled": True, "time": "15:30", "dte_lte": 0}}},
    ]
