"""Nested AND/OR rule engine with human-readable explanations.

Every leaf evaluates against a point-in-time Context (see engine.Context) and returns
(passed, explanation). Groups evaluate all enabled children (no short-circuit) so the
trade inspector can show exactly which conditions held and their values.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from strata.core.calendar import fmt_hhmm, parse_hhmm
from strata.engine.models import CondNode

CMP = {
    ">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: abs(a - b) < 1e-9, "!=": lambda a, b: abs(a - b) >= 1e-9,
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SESSIONS = {"open": (570, 600), "morning": (600, 720), "midday": (720, 840), "afternoon": (840, 930), "close": (-30, 0)}

# --- metadata consumed by the UI to render condition editors -------------------------
P = lambda name, type, label, default=None, **kw: {"name": name, "type": type, "label": label, "default": default, **kw}
CMP_OPTS = [">", "<", ">=", "<="]
OPERAND_FEATURES = ["price", "open", "prev_close", "prev_high", "prev_low", "day_high", "day_low", "vix", "iv", "iv_rank", "iv_pct", "gap_pct", "day_move_pct"]
INDICATOR_NAMES = ["rsi", "sma", "ema", "macd", "vwap", "atr", "bbands", "rvol"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "60m", "1d"]

LEAF_TYPES: dict[str, dict] = {
    "entry_times": {"label": "Entry at times", "category": "Time", "params": [P("times", "string", "Times (ET, comma separated)", "10:00")]},
    "time_window": {"label": "Time window", "category": "Time", "params": [P("start", "time", "From", "09:45"), P("end", "time", "Until", "15:30")]},
    "session": {"label": "Trading session", "category": "Time", "params": [P("name", "select", "Session", "morning", options=list(SESSIONS))]},
    "day_of_week": {"label": "Day of week", "category": "Time", "params": [P("days", "multiselect", "Days", ["Mon", "Tue", "Wed", "Thu", "Fri"], options=DOW[:5])]},
    "price_move": {"label": "Underlying move", "category": "Price", "params": [P("ref", "select", "Since", "open", options=["prev_close", "open", "minutes_ago"]), P("minutes", "int", "Minutes (for minutes_ago)", 30), P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "Move %", 0.5), P("absolute", "bool", "Use absolute move", False)]},
    "opening_range": {"label": "Opening range breakout", "category": "Price", "params": [P("minutes", "int", "Range minutes", 30), P("direction", "select", "Signal", "breakout_up", options=["breakout_up", "breakout_down", "either", "inside"]), P("buffer_pct", "number", "Buffer %", 0.0)]},
    "prev_day_level": {"label": "Previous-day level", "category": "Price", "params": [P("level", "select", "Level", "high", options=["high", "low", "close"]), P("relation", "select", "Price is", "above", options=["above", "below"])]},
    "gap": {"label": "Opening gap", "category": "Price", "params": [P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "Gap %", 0.5)]},
    "day_range": {"label": "Intraday range so far", "category": "Price", "params": [P("cmp", "select", "Compare", "<", options=CMP_OPTS), P("value", "number", "Range %", 1.0)]},
    "compare": {"label": "Indicator / feature comparison", "category": "Indicators", "params": [P("lhs", "operand", "Left", {"kind": "indicator", "name": "rsi", "params": {"period": 14}, "tf": "5m"}), P("cmp", "select", "Compare", "<", options=CMP_OPTS + ["crosses_above", "crosses_below"]), P("rhs", "operand", "Right", {"kind": "const", "value": 30})]},
    "vix": {"label": "VIX level", "category": "Volatility", "params": [P("cmp", "select", "Compare", "<", options=CMP_OPTS), P("value", "number", "VIX", 25)]},
    "vix_change": {"label": "VIX change vs prior close", "category": "Volatility", "params": [P("cmp", "select", "Compare", "<", options=CMP_OPTS), P("value", "number", "Change %", 5)]},
    "iv": {"label": "ATM implied vol (30d)", "category": "Volatility", "params": [P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "IV %", 15)]},
    "iv_rank": {"label": "IV rank", "category": "Volatility", "params": [P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "IV rank (0-100)", 30), P("lookback", "int", "Lookback sessions", 252)]},
    "iv_percentile": {"label": "IV percentile", "category": "Volatility", "params": [P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "IV percentile (0-100)", 50), P("lookback", "int", "Lookback sessions", 252)]},
    "regime": {"label": "Market regime", "category": "Regime", "params": [P("trend", "select", "Trend", "any", options=["any", "bull", "bear", "neutral"]), P("vol", "select", "Vol regime", "any", options=["any", "low", "normal", "high"])]},
    "event": {"label": "Scheduled economic event", "category": "Events", "params": [P("types", "multiselect", "Event types", ["FOMC", "CPI", "NFP"], options=["FOMC", "CPI", "NFP", "PPI", "GDP", "OPEX"]), P("when", "select", "When", "today", options=["today", "tomorrow", "within_n"]), P("n", "int", "Sessions ahead (within_n)", 2), P("mode", "select", "Mode", "exclude", options=["exclude", "require"])]},
    "position_pnl_pct": {"label": "Position P&L %", "category": "Position", "exit_only": True, "params": [P("cmp", "select", "Compare", ">=", options=CMP_OPTS), P("value", "number", "% of premium", 50)]},
    "position_dte": {"label": "Position DTE", "category": "Position", "exit_only": True, "params": [P("cmp", "select", "Compare", "<=", options=CMP_OPTS), P("value", "int", "DTE", 1)]},
    "position_delta": {"label": "Position |delta| per unit", "category": "Position", "exit_only": True, "params": [P("cmp", "select", "Compare", ">", options=CMP_OPTS), P("value", "number", "Share-equivalent delta", 30)]},
    "minutes_in_trade": {"label": "Minutes in trade", "category": "Position", "exit_only": True, "params": [P("cmp", "select", "Compare", ">=", options=CMP_OPTS), P("value", "int", "Minutes", 60)]},
}


def _fmt(x: float, nd=2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def _cmp(a, op, b) -> bool:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return False
    return CMP[op](a, b)


def operand_value(ctx, op: dict, prev: bool = False) -> float:
    kind = op.get("kind", "const")
    if kind == "const":
        return float(op.get("value", 0))
    sym = op.get("symbol") or ctx.symbol
    if kind == "price":
        return ctx.price(sym, prev=prev)
    if kind == "indicator":
        return ctx.indicator(sym, op.get("name", "rsi"), op.get("params") or {}, op.get("tf", "5m"), op.get("field"), prev=prev)
    if kind == "feature":
        return ctx.feature(op.get("name", "price"), sym, prev=prev)
    raise ValueError(f"unknown operand kind {kind}")


def operand_label(op: dict) -> str:
    kind = op.get("kind", "const")
    if kind == "const":
        return f"{op.get('value')}"
    if kind == "price":
        return "price"
    if kind == "indicator":
        ps = ",".join(str(v) for v in (op.get("params") or {}).values())
        fld = f".{op['field']}" if op.get("field") else ""
        return f"{op.get('name', '').upper()}({ps}{',' if ps else ''}{op.get('tf', '5m')}){fld}"
    return op.get("name", "?")


def eval_leaf(node: CondNode, ctx, pos=None) -> tuple[bool, str]:
    p = node.params or {}
    t = node.type
    m = ctx.minute
    if t == "entry_times":
        times = [parse_hhmm(x) for x in str(p.get("times", "10:00")).replace(";", ",").split(",") if x.strip()]
        step = max(1, ctx.tick_minutes)
        hit = [x for x in times if x <= m < x + step]
        return bool(hit), f"time {fmt_hhmm(m)} {'matches' if hit else 'not in'} entry times {','.join(fmt_hhmm(x) for x in times)}"
    if t == "time_window":
        a, b = parse_hhmm(p.get("start", "09:30")), parse_hhmm(p.get("end", "16:00"))
        ok = a <= m <= b
        return ok, f"time {fmt_hhmm(m)} {'within' if ok else 'outside'} {fmt_hhmm(a)}-{fmt_hhmm(b)}"
    if t == "session":
        a, b = SESSIONS[p.get("name", "morning")]
        if a < 0:
            a, b = ctx.close_minute + a, ctx.close_minute
        ok = a <= m < b
        return ok, f"{fmt_hhmm(m)} {'in' if ok else 'not in'} {p.get('name')} session ({fmt_hhmm(a)}-{fmt_hhmm(b)})"
    if t == "day_of_week":
        days = p.get("days") or []
        dn = DOW[ctx.today.weekday()]
        return dn in days, f"{dn} {'in' if dn in days else 'not in'} {days}"
    if t == "price_move":
        ref = p.get("ref", "open")
        price = ctx.price(ctx.symbol)
        if ref == "prev_close":
            base = ctx.feature("prev_close", ctx.symbol)
        elif ref == "open":
            base = ctx.feature("open", ctx.symbol)
        else:
            base = ctx.price_minutes_ago(ctx.symbol, int(p.get("minutes", 30)))
        mv = (price / base - 1) * 100 if base and math.isfinite(base) else float("nan")
        val = abs(mv) if p.get("absolute") else mv
        ok = _cmp(val, p.get("cmp", ">"), float(p.get("value", 0)))
        return ok, f"{'|move|' if p.get('absolute') else 'move'} since {ref} = {_fmt(val)}% {p.get('cmp')} {p.get('value')}%"
    if t == "opening_range":
        mins = int(p.get("minutes", 30))
        hi, lo, complete = ctx.opening_range(ctx.symbol, mins)
        if not complete:
            return False, f"opening range ({mins}m) not complete yet"
        price = ctx.price(ctx.symbol)
        buf = float(p.get("buffer_pct", 0)) / 100
        d = p.get("direction", "breakout_up")
        up, dn = price > hi * (1 + buf), price < lo * (1 - buf)
        ok = {"breakout_up": up, "breakout_down": dn, "either": up or dn, "inside": not (up or dn)}[d]
        return ok, f"price {_fmt(price)} vs OR{mins} [{_fmt(lo)}, {_fmt(hi)}] -> {d} {'yes' if ok else 'no'}"
    if t == "prev_day_level":
        lvl = p.get("level", "high")
        v = ctx.feature(f"prev_{lvl}", ctx.symbol)
        price = ctx.price(ctx.symbol)
        ok = price > v if p.get("relation", "above") == "above" else price < v
        if not math.isfinite(v):
            ok = False
        return ok, f"price {_fmt(price)} {p.get('relation')} prev-day {lvl} {_fmt(v)}"
    if t == "gap":
        g = ctx.feature("gap_pct", ctx.symbol)
        ok = _cmp(g, p.get("cmp", ">"), float(p.get("value", 0)))
        return ok, f"gap {_fmt(g)}% {p.get('cmp')} {p.get('value')}%"
    if t == "day_range":
        hi, lo = ctx.feature("day_high", ctx.symbol), ctx.feature("day_low", ctx.symbol)
        o = ctx.feature("open", ctx.symbol)
        rng = (hi - lo) / o * 100 if o else float("nan")
        ok = _cmp(rng, p.get("cmp", "<"), float(p.get("value", 1)))
        return ok, f"day range {_fmt(rng)}% {p.get('cmp')} {p.get('value')}%"
    if t == "compare":
        lhs, rhs = p.get("lhs") or {}, p.get("rhs") or {"kind": "const", "value": 0}
        op = p.get("cmp", ">")
        a, b = operand_value(ctx, lhs), operand_value(ctx, rhs)
        if op in ("crosses_above", "crosses_below"):
            a0, b0 = operand_value(ctx, lhs, prev=True), operand_value(ctx, rhs, prev=True)
            if op == "crosses_above":
                ok = _cmp(a0, "<=", b0) and _cmp(a, ">", b)
            else:
                ok = _cmp(a0, ">=", b0) and _cmp(a, "<", b)
            return ok, f"{operand_label(lhs)} {op.replace('_', ' ')} {operand_label(rhs)} ({_fmt(a0)}->{_fmt(a)} vs {_fmt(b0)}->{_fmt(b)})"
        ok = _cmp(a, op, b)
        return ok, f"{operand_label(lhs)} = {_fmt(a)} {op} {operand_label(rhs)} = {_fmt(b)}"
    if t == "vix":
        v = ctx.feature("vix", ctx.symbol)
        ok = _cmp(v, p.get("cmp", "<"), float(p.get("value", 25)))
        return ok, f"VIX {_fmt(v)} {p.get('cmp')} {p.get('value')}"
    if t == "vix_change":
        v, pc = ctx.feature("vix", ctx.symbol), ctx.feature("vix_prev_close", ctx.symbol)
        ch = (v / pc - 1) * 100 if pc else float("nan")
        ok = _cmp(ch, p.get("cmp", "<"), float(p.get("value", 5)))
        return ok, f"VIX change {_fmt(ch)}% {p.get('cmp')} {p.get('value')}%"
    if t == "iv":
        v = ctx.feature("iv", ctx.symbol)
        ok = _cmp(v, p.get("cmp", ">"), float(p.get("value", 15)))
        return ok, f"ATM IV30 {_fmt(v)}% {p.get('cmp')} {p.get('value')}%"
    if t in ("iv_rank", "iv_percentile"):
        v = ctx.iv_rank(ctx.symbol, int(p.get("lookback", 252)), pct=(t == "iv_percentile"))
        ok = _cmp(v, p.get("cmp", ">"), float(p.get("value", 30)))
        return ok, f"{'IV percentile' if t == 'iv_percentile' else 'IV rank'} {_fmt(v, 1)} {p.get('cmp')} {p.get('value')}"
    if t == "regime":
        trend, vol = ctx.regime(ctx.symbol)
        want_t, want_v = p.get("trend", "any"), p.get("vol", "any")
        ok = (want_t in ("any", trend)) and (want_v in ("any", vol))
        return ok, f"regime trend={trend} vol={vol} (want {want_t}/{want_v})"
    if t == "event":
        types = p.get("types") or None
        when = p.get("when", "today")
        if when == "today":
            evs = ctx.events.on(ctx.today, types)
        elif when == "tomorrow":
            from strata.core.calendar import next_trading_day
            evs = ctx.events.on(next_trading_day(ctx.today), types)
        else:
            evs = ctx.events.within(ctx.today, int(p.get("n", 2)), types)
        has = bool(evs)
        ok = has if p.get("mode", "exclude") == "require" else not has
        names = ",".join(sorted({e["event_type"] for e in evs})) or "none"
        return ok, f"events {when}: {names} ({p.get('mode')})"
    # position-scoped leaves
    if pos is None:
        return False, f"{t} requires an open position"
    if t == "position_pnl_pct":
        v = ctx.position_pnl_pct(pos)
        ok = _cmp(v, p.get("cmp", ">="), float(p.get("value", 50)))
        return ok, f"P&L {_fmt(v, 1)}% {p.get('cmp')} {p.get('value')}%"
    if t == "position_dte":
        v = pos.front_dte(ctx.today_num)
        ok = _cmp(float(v), p.get("cmp", "<="), float(p.get("value", 1)))
        return ok, f"DTE {v} {p.get('cmp')} {p.get('value')}"
    if t == "position_delta":
        v = abs(pos.greeks()[0]) / max(pos.units, 1)
        ok = _cmp(v, p.get("cmp", ">"), float(p.get("value", 30)))
        return ok, f"|delta|/unit {_fmt(v, 1)} {p.get('cmp')} {p.get('value')}"
    if t == "minutes_in_trade":
        v = (ctx.t - pos.entry_ts) / 60
        ok = _cmp(v, p.get("cmp", ">="), float(p.get("value", 60)))
        return ok, f"{v:.0f} min in trade {p.get('cmp')} {p.get('value')}"
    raise ValueError(f"unknown condition type {t!r}")


def evaluate(node: CondNode | None, ctx, pos=None) -> tuple[bool, list[dict]]:
    """Returns (passed, flat list of {id, depth, text, passed, group?})."""
    out: list[dict] = []
    if node is None:
        return True, out

    def rec(n: CondNode, depth: int) -> bool | None:
        if not n.enabled:
            return None
        if n.kind == "group":
            vals = [v for v in (rec(c, depth + 1) for c in n.children) if v is not None]
            if not vals:
                res = True
            else:
                res = all(vals) if n.op == "and" else any(vals)
            if n.negate:
                res = not res
            if depth > 0:
                out.append({"id": n.id, "depth": depth, "text": f"{'NOT ' if n.negate else ''}{n.op.upper()} group", "passed": res, "group": True})
            return res
        try:
            ok, text = eval_leaf(n, ctx, pos)
        except Exception as e:  # malformed params -> condition fails, reason recorded
            ok, text = False, f"{n.type}: error {e}"
        if n.negate:
            ok, text = not ok, f"NOT ({text})"
        out.append({"id": n.id, "depth": depth, "text": text, "passed": ok, "type": n.type})
        return ok

    res = rec(node, 0)
    return (True if res is None else res), out


def required_tick_types(node: CondNode | None) -> set[str]:
    s: set[str] = set()
    if node is None:
        return s

    def rec(n):
        if n.kind == "cond" and n.enabled:
            s.add(n.type)
        for c in n.children:
            rec(c)

    rec(node)
    return s
