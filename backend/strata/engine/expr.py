"""Safe expression language for trading rules.

    previous_session.return_pct <= -3
    price >= entry.underlying_price * 1.01
    consecutive_bearish_candles("5m") >= 3 and price > vwap
    crosses_above(price, vwap)
    return_pct("QQQ", sessions=5) - return_pct("SPY", sessions=5) > 2
    first_touch(prev_high, prev_low) == 1
    price > prev_high and QQQ.high <= QQQ.prev_high          (another symbol's values)
    price >= low_since("drop") + 0.5 * (open - low_since("drop"))   (state of an event sequence)
    price <= entry.underlying_price * 0.995                  (add / exit rules: the position's own state)

Expressions are parsed with Python's `ast` module and evaluated by a whitelist interpreter:
only literals, arithmetic, comparisons, and/or/not and the names/functions registered
below are allowed - no attribute access on arbitrary objects, no subscripts, no lambdas,
no imports. Every accessor is point-in-time (it only sees bars that have completed).

Unknown inputs (missing data) evaluate to None and propagate: a rule that depends on
missing data is "not evaluable" rather than false, so the engine can report the session as
excluded instead of silently treating it as a non-signal.
"""
from __future__ import annotations

import ast
import math
import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from strata.core import calendar as cal

NA = None
TF = ("1m", "2m", "3m", "5m", "10m", "15m", "30m", "60m", "1h", "1d")


class ExprError(ValueError):
    pass


ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
           ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Call, ast.Name, ast.Load, ast.Attribute, ast.Constant, ast.keyword, ast.IfExp)

# ------------------------------------------------------------------ registry (docs for UI + AI)
def _d(sig, desc, cat, needs="daily"):
    return {"signature": sig, "description": desc, "category": cat, "needs": needs}


REGISTRY: dict[str, dict] = {
    # prices & session
    "price": _d("price", "Underlying price now (close of the last completed bar; the session open at the opening tick)", "Price"),
    "open": _d("open", "Today's session open", "Price"),
    "high": _d("high", "Session high so far", "Price", "intraday"),
    "low": _d("low", "Session low so far", "Price", "intraday"),
    "prev_close": _d("prev_close", "Previous session close", "Price"),
    "prev_open": _d("prev_open", "Previous session open", "Price"),
    "prev_high": _d("prev_high", "Previous session high", "Price"),
    "prev_low": _d("prev_low", "Previous session low", "Price"),
    "vwap": _d("vwap", "Session volume-weighted average price so far", "Price", "intraday"),
    "gap_pct": _d("gap_pct", "Overnight gap: today's open vs the previous close, %", "Price"),
    "change_pct": _d("change_pct", "Price vs the previous close, %", "Price"),
    "from_open_pct": _d("from_open_pct", "Price vs today's open, %", "Price"),
    "range_pct": _d("range_pct", "Session high-low range so far as % of the open", "Price", "intraday"),
    "dist_from_vwap_pct": _d("dist_from_vwap_pct", "Price vs session VWAP, %", "Price", "intraday"),
    "volume": _d("volume", "Session volume so far", "Volume", "intraday"),
    "time": _d("time", "Minutes since midnight, New York time (compare with hhmm(10, 30))", "Time"),
    "minutes_since_open": _d("minutes_since_open", "Minutes since the 09:30 ET open", "Time"),
    "weekday": _d("weekday", "0=Mon ... 4=Fri", "Time"),
    "minutes_to_close": _d("minutes_to_close", "Minutes until this session's close (early closes included)", "Time"),
    "is_last_session_of_week": _d("is_last_session_of_week", "True on the final trading session of the calendar week (Friday, or Thursday before a holiday)", "Time"),
    "vix": _d("vix", "Volatility index level now (previous close when no intraday VIX is stored)", "Volatility"),
    "vix_prev_close": _d("vix_prev_close", "Volatility index previous close", "Volatility"),
    "iv": _d("iv", "ATM 30-day implied volatility from the stored chain, %", "Volatility", "options"),
    "iv_rank": _d("iv_rank", "IV rank over 252 sessions (0-100)", "Volatility", "options"),
    "iv_percentile": _d("iv_percentile", "IV percentile over 252 sessions (0-100)", "Volatility", "options"),
    # objects
    "previous_session": _d("previous_session.<field>", "Fields of the previous session: return_pct (close-to-close), open_to_close_pct, gap_pct, range_pct, open, high, low, close, volume", "Sessions"),
    "session": _d("session.<field>", "Today so far: open, high, low, return_pct (vs previous close), from_open_pct, gap_pct, range_pct, volume", "Sessions"),
    "entry": _d("entry.<field>", "Exit / add rules - the open position's entry: underlying_price, premium, time, minutes_ago, signal_price, vwap, session_open", "Position"),
    "position": _d("position.<field>", "Exit / add rules: pnl_pct (of premium), pnl, dte, minutes_held, sessions_held, delta, max_gain_pct, max_loss_pct, underlying_high, underlying_low, underlying_move_pct, lot, adds", "Position"),
    "group": _d("group.<field>", "Exit / add rules - every lot of a scaled position together: pnl, pnl_pct, lots, avg_underlying_entry", "Position"),
    # running session state
    "max_decline_from_open_pct": _d("max_decline_from_open_pct", "Largest fall below today's open so far, % (0 or negative)", "State", "intraday"),
    "max_rally_from_open_pct": _d("max_rally_from_open_pct", "Largest rise above today's open so far, % (0 or positive)", "State", "intraday"),
    "pullback_from_high_pct": _d("pullback_from_high_pct", "Price vs the session high so far, % (0 or negative)", "State", "intraday"),
    "bounce_from_low_pct": _d("bounce_from_low_pct", "Price vs the session low so far, % (0 or positive)", "State", "intraday"),
    # event sequences (stages are declared on the strategy; these read their state)
    "occurred": _d('occurred("stage")', "True once the named stage of the entry sequence has happened today", "Sequences", "intraday"),
    "stage_price": _d('stage_price("stage")', "Underlying price when the named stage happened", "Sequences", "intraday"),
    "stage_time": _d('stage_time("stage")', "Minutes since midnight (New York) when the named stage happened", "Sequences", "intraday"),
    "minutes_since": _d('minutes_since("stage")', "Minutes elapsed since the named stage happened", "Sequences", "intraday"),
    "low_since": _d('low_since("stage")', "Lowest price since the named stage happened (including the price at the stage)", "Sequences", "intraday"),
    "high_since": _d('high_since("stage")', "Highest price since the named stage happened (including the price at the stage)", "Sequences", "intraday"),
    "ref": _d('ref("name")', "A reference price captured when a stage happened (declared on the stage)", "Sequences", "intraday"),
    # history
    "return_pct": _d("return_pct(symbol=None, sessions=1)", "Close-to-close return over the last N completed sessions, %", "Sessions"),
    "session_return": _d("session_return(n=1, symbol=None)", "Close-to-close return of the n-th previous session, % (1 = yesterday)", "Sessions"),
    "open_to_close": _d("open_to_close(n=1, symbol=None)", "Open-to-close return of the n-th previous session, %", "Sessions"),
    "close": _d("close(n=1, symbol=None)", "Close of the n-th previous session", "Sessions"),
    "session_high": _d("session_high(n=1, symbol=None)", "High of the n-th previous session", "Sessions"),
    "session_low": _d("session_low(n=1, symbol=None)", "Low of the n-th previous session", "Sessions"),
    "consecutive_down_days": _d("consecutive_down_days(symbol=None)", "Completed sessions in a row with a lower close", "Sessions"),
    "consecutive_up_days": _d("consecutive_up_days(symbol=None)", "Completed sessions in a row with a higher close", "Sessions"),
    "consecutive_red_days": _d("consecutive_red_days(symbol=None)", "Completed sessions in a row closing below their open", "Sessions"),
    "consecutive_green_days": _d("consecutive_green_days(symbol=None)", "Completed sessions in a row closing above their open", "Sessions"),
    "consecutive_bearish_candles": _d('consecutive_bearish_candles(tf="5m")', "Completed candles in a row closing below their open (today)", "Candles", "intraday"),
    "consecutive_bullish_candles": _d('consecutive_bullish_candles(tf="5m")', "Completed candles in a row closing above their open (today)", "Candles", "intraday"),
    "prev_week_high": _d("prev_week_high()", "High of the previous calendar week", "Levels"),
    "prev_week_low": _d("prev_week_low()", "Low of the previous calendar week", "Levels"),
    "prev_month_high": _d("prev_month_high()", "High of the previous calendar month", "Levels"),
    "prev_month_low": _d("prev_month_low()", "Low of the previous calendar month", "Levels"),
    "rolling_high": _d('rolling_high(n, tf="1d")', "Highest high of the last n completed bars", "Levels"),
    "rolling_low": _d('rolling_low(n, tf="1d")', "Lowest low of the last n completed bars", "Levels"),
    "opening_range_high": _d("opening_range_high(minutes=30)", "High of the first N minutes (unknown until they have elapsed)", "Levels", "intraday"),
    "opening_range_low": _d("opening_range_low(minutes=30)", "Low of the first N minutes", "Levels", "intraday"),
    "gap_fill_pct": _d("gap_fill_pct()", "How much of today's opening gap has been retraced so far, % (100 = touched the previous close)", "Levels", "intraday"),
    "retracement_pct": _d("retracement_pct(a, b)", "Where price sits between levels a (0%) and b (100%)", "Levels"),
    "dist_pct": _d("dist_pct(a, b)", "(a / b - 1) * 100", "Math"),
    "touched": _d("touched(level)", "True if today's range so far includes the level", "Levels", "intraday"),
    "first_touch": _d("first_touch(a, b)", "Which level traded first today: 1 = a, 2 = b, 0 = neither yet; unknown when both fall in the same bar", "Levels", "intraday"),
    "time_above": _d("time_above(level)", "Minutes today spent with bars closing above the level", "Levels", "intraday"),
    "time_below": _d("time_below(level)", "Minutes today spent with bars closing below the level", "Levels", "intraday"),
    "crosses_above": _d("crosses_above(a, b)", "a was <= b at the previous evaluation and is > b now", "Sequences", "intraday"),
    "crosses_below": _d("crosses_below(a, b)", "a was >= b at the previous evaluation and is < b now", "Sequences", "intraday"),
    "prev": _d("prev(x)", "Value of x at the previous evaluation tick", "Sequences", "intraday"),
    # indicators
    "rsi": _d('rsi(n=14, tf="5m")', "Wilder RSI", "Indicators"),
    "sma": _d('sma(n=20, tf="1d")', "Simple moving average of closes", "Indicators"),
    "ema": _d('ema(n=20, tf="1d")', "Exponential moving average of closes", "Indicators"),
    "atr": _d('atr(n=14, tf="1d")', "Average true range", "Indicators"),
    "atr_pct": _d('atr_pct(n=14, tf="1d")', "ATR as % of price", "Indicators"),
    "macd": _d('macd(tf="5m")', "MACD(12,26,9) line", "Indicators"),
    "macd_signal": _d('macd_signal(tf="5m")', "MACD signal line", "Indicators"),
    "macd_hist": _d('macd_hist(tf="5m")', "MACD histogram", "Indicators"),
    "bb_upper": _d('bb_upper(n=20, k=2, tf="5m")', "Upper Bollinger band", "Indicators"),
    "bb_lower": _d('bb_lower(n=20, k=2, tf="5m")', "Lower Bollinger band", "Indicators"),
    "bb_pctb": _d('bb_pctb(n=20, k=2, tf="5m")', "Bollinger %B", "Indicators"),
    "rvol": _d("rvol(n=21)", "Realised volatility of daily closes, annualised %", "Volatility"),
    "rvol_percentile": _d("rvol_percentile(n=21, lookback=252)", "Percentile of realised vol within its own history (0-100)", "Volatility"),
    "volume_ratio": _d('volume_ratio(tf="5m", n=20)', "Last completed bar's volume / average of the previous n bars", "Volume", "intraday"),
    "day_volume_ratio": _d("day_volume_ratio(n=20)", "Last completed session's volume / average of the n sessions before it", "Volume"),
    # cross-symbol & events
    "relative_strength": _d("relative_strength(a, b, sessions=20)", "return_pct(a) - return_pct(b) over N sessions, percentage points", "Cross-symbol"),
    "event_today": _d('event_today("FOMC", ...)', "A scheduled event of one of these types is on today's calendar", "Events"),
    "event_within": _d('event_within(n, "CPI", ...)', "An event of these types falls within the next n sessions", "Events"),
    "hhmm": _d("hhmm(h, m)", "Minutes since midnight for a clock time, e.g. time >= hhmm(10, 30)", "Time"),
    "abs": _d("abs(x)", "Absolute value", "Math"), "min": _d("min(a, b, ...)", "Minimum", "Math"), "max": _d("max(a, b, ...)", "Maximum", "Math"),
}
INTRADAY_NAMES = {k for k, v in REGISTRY.items() if v["needs"] == "intraday"}


# Alternative spellings accepted in rules; rewritten to the canonical name before evaluation.
ALIASES = {"previous_close": "prev_close", "previous_open": "prev_open", "previous_high": "prev_high", "previous_low": "prev_low",
           "yesterday_close": "prev_close", "yesterday_high": "prev_high", "yesterday_low": "prev_low"}
FUNCTIONS = {k for k, v in REGISTRY.items() if "(" in v["signature"]}
# every field that may follow a dot; anything else is rejected when the rule is parsed
FIELDS = {
    "underlying": {"price"},
    "previous_session": {"open", "high", "low", "close", "volume", "open_to_close_pct", "range_pct", "return_pct", "gap_pct"},
    "session": {"open", "high", "low", "return_pct", "from_open_pct", "gap_pct", "range_pct", "volume"},
    "entry": {"underlying_price", "premium", "time", "minutes_ago", "signal_price", "vwap", "session_open"},
    "position": {"pnl_pct", "pnl", "dte", "minutes_held", "sessions_held", "delta", "max_gain_pct", "max_loss_pct", "underlying_high", "underlying_low", "underlying_move_pct", "lot", "adds"},
    "group": {"pnl", "pnl_pct", "lots", "avg_underlying_entry"},
}
POSITION_OBJECTS = ("entry", "position", "group")
STAGE_FUNCS = ("occurred", "stage_price", "stage_time", "minutes_since", "low_since", "high_since")
# plain values that can be read for another symbol as TICKER.value, e.g. QQQ.prev_high
SYMBOL_FIELDS = {"price", "open", "high", "low", "prev_close", "prev_open", "prev_high", "prev_low", "vwap", "gap_pct", "change_pct", "from_open_pct", "range_pct", "volume",
                 "dist_from_vwap_pct", "max_decline_from_open_pct", "max_rally_from_open_pct", "pullback_from_high_pct", "bounce_from_low_pct"}
_TICKER = re.compile(r"[A-Z][A-Z0-9]{0,5}")


def is_ticker(name: str) -> bool:
    return bool(_TICKER.fullmatch(name)) and name not in FIELDS and name not in REGISTRY


class _Canon(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name):
        return ast.copy_location(ast.Name(id=ALIASES[node.id], ctx=node.ctx), node) if node.id in ALIASES else node


@lru_cache(maxsize=4096)
def parse(src: str) -> ast.Expression:
    if len(src) > 2000:
        raise ExprError("expression is too long")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in expression {src!r}: {e.msg}") from None
    tree = ast.fix_missing_locations(_Canon().visit(tree))
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    tickers = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and is_ticker(n.value.id)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) in tickers:
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and id(node.value) in tickers:
            if node.attr not in SYMBOL_FIELDS:
                raise ExprError(f"{node.value.id}.{node.attr}: another symbol's values are limited to {', '.join(sorted(SYMBOL_FIELDS))}")
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in STAGE_FUNCS + ("ref",):
            if len(node.args) != 1 or node.keywords or not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                raise ExprError(f'{node.func.id} takes one quoted name, e.g. {REGISTRY[node.func.id]["signature"]}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in REGISTRY and node.func.id not in FUNCTIONS:
            raise ExprError(f"{node.func.id} is a value, not a function: write {REGISTRY[node.func.id]['signature']}")
        if isinstance(node, ast.Name) and node.id in FUNCTIONS and id(node) not in called:
            raise ExprError(f"{node.id} is a function: write {REGISTRY[node.id]['signature']}")
        if not isinstance(node, ALLOWED):
            raise ExprError(f"{type(node).__name__} is not allowed in expressions")
        if isinstance(node, ast.Name) and node.id not in REGISTRY and node.id not in ("underlying", "True", "False"):
            raise ExprError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in REGISTRY):
            raise ExprError("only registered functions can be called")
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id in FIELDS):
            raise ExprError("a dot is only allowed after previous_session, session, entry, position, group, underlying or a ticker (e.g. QQQ.prev_high)")
        if isinstance(node, ast.Attribute) and node.attr not in FIELDS[node.value.id]:
            raise ExprError(f"{node.value.id} has no field {node.attr!r}; available: {', '.join(sorted(FIELDS[node.value.id]))}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool)):
            raise ExprError("unsupported literal")
    return tree


def validate(src: str) -> list[str]:
    try:
        parse(src)
        return []
    except ExprError as e:
        return [str(e)]


def names_used(src: str) -> set[str]:
    tree = parse(src)
    out = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    out |= {f"{n.value.id}.{n.attr}" for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return out


def stages_used(src: str) -> set[str]:
    """Names of sequence stages an expression reads."""
    return {n.args[0].value for n in ast.walk(parse(src)) if isinstance(n, ast.Call) and n.func.id in STAGE_FUNCS and n.args and isinstance(n.args[0], ast.Constant)}


def refs_used(src: str) -> set[str]:
    return {n.args[0].value for n in ast.walk(parse(src)) if isinstance(n, ast.Call) and n.func.id == "ref" and n.args and isinstance(n.args[0], ast.Constant)}


def uses_position(src: str) -> bool:
    """True when the rule reads an open position (entry.*, position.*, group.*)."""
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in POSITION_OBJECTS for n in ast.walk(parse(src)))


def symbols_used(src: str) -> set[str]:
    """Other symbols referenced, e.g. return_pct("QQQ", sessions=5) or QQQ.prev_high."""
    out = set()
    for n in ast.walk(parse(src)):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and is_ticker(n.value.id):
            out.add(n.value.id)
        if isinstance(n, ast.Call):
            for a in list(n.args) + [k.value for k in n.keywords]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value not in TF and a.value.isupper() and len(a.value) <= 6 and n.func.id not in ("event_today", "event_within"):
                    out.add(a.value)
    return out


def needs_intraday(src: str) -> bool:
    u = names_used(src)
    if u & INTRADAY_NAMES or {x.split(".", 1)[1] for x in u if "." in x} & INTRADAY_NAMES:
        return True
    for n in ast.walk(parse(src)):
        if isinstance(n, ast.Call):
            for a in list(n.args) + [k.value for k in n.keywords]:
                if isinstance(a, ast.Constant) and a.value in TF and a.value != "1d":
                    return True
            if n.func.id in ("rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_pctb") and not any(isinstance(a, ast.Constant) and a.value == "1d" for a in list(n.args) + [k.value for k in n.keywords]):
                return True
    return False


def _num(x) -> float | None:
    if x is None or isinstance(x, str):
        return None
    if isinstance(x, bool):
        return float(x)
    x = float(x)
    return x if math.isfinite(x) else None


class Evaluator:
    """Evaluates parsed expressions against an engine Context (point-in-time)."""

    def __init__(self, ctx, pos=None, symbol: str | None = None):
        self.ctx = ctx
        self.pos = pos
        self.sym = symbol or ctx.symbol
        self.shift = False  # evaluate at the previous tick
        self.missing: set[str] = set()

    # ---- plumbing
    @property
    def t(self) -> int:
        return self.ctx.prev_t if self.shift else self.ctx.t

    def ser(self, sym=None):
        s = self.ctx.series(sym or self.sym)
        if s is None or s.empty():
            self.missing.add(f"{sym or self.sym} prices")
            return None
        return s

    def _na(self, what: str):
        self.missing.add(what)
        return None

    def eval(self, src: str):
        return self._ev(parse(src).body)

    def _ev(self, n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in ("True", "False"):
                return n.id == "True"
            return self.name(n.id)
        if isinstance(n, ast.Attribute):
            if is_ticker(n.value.id):  # another symbol's value, e.g. QQQ.prev_high
                other = Evaluator(self.ctx, self.pos, symbol=n.value.id)
                other.shift = self.shift
                v = other.name(n.attr)
                self.missing |= other.missing
                return v
            return self.attr(n.value.id, n.attr)
        if isinstance(n, ast.UnaryOp):
            v = self._ev(n.operand)
            if isinstance(n.op, ast.Not):
                return None if v is None else (not v)
            v = _num(v)
            return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
        if isinstance(n, ast.BinOp):
            a, b = _num(self._ev(n.left)), _num(self._ev(n.right))
            if a is None or b is None:
                return None
            try:
                if isinstance(n.op, ast.Add):
                    return a + b
                if isinstance(n.op, ast.Sub):
                    return a - b
                if isinstance(n.op, ast.Mult):
                    return a * b
                if isinstance(n.op, ast.Div):
                    return a / b if b != 0 else None
                if isinstance(n.op, ast.Mod):
                    return a % b if b != 0 else None
                return a ** b
            except (OverflowError, ZeroDivisionError, ValueError):
                return None
        if isinstance(n, ast.Compare):
            left = self._ev(n.left)
            res: bool | None = True
            for op, comp in zip(n.ops, n.comparators):
                right = self._ev(comp)
                a, b = _num(left), _num(right)
                if a is None or b is None:
                    return None
                ok = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b, ast.Eq: abs(a - b) < 1e-9, ast.NotEq: abs(a - b) >= 1e-9}[type(op)]
                if not ok:
                    res = False
                left = right
            return res
        if isinstance(n, ast.BoolOp):
            vals = [self._ev(v) for v in n.values]
            vals = [None if v is None else bool(v) for v in vals]
            if isinstance(n.op, ast.And):
                return False if any(v is False for v in vals) else (None if any(v is None for v in vals) else True)
            return True if any(v is True for v in vals) else (None if any(v is None for v in vals) else False)
        if isinstance(n, ast.IfExp):
            c = self._ev(n.test)
            return None if c is None else self._ev(n.body if c else n.orelse)
        if isinstance(n, ast.Call):
            return self.call(n)
        raise ExprError(f"cannot evaluate {type(n).__name__}")

    # ---- names
    def name(self, k: str):
        c = self.ctx
        s = self.ser()
        if k in ("time",):
            return float(c.minute)
        if k == "minutes_since_open":
            return float(c.minute - 570)
        if k == "weekday":
            return float(c.today.weekday())
        if k == "minutes_to_close":
            return float(c.close_minute - c.minute)
        if k == "is_last_session_of_week":
            return cal.next_trading_day(c.today).isocalendar()[1] != c.today.isocalendar()[1]
        if k in ("vix", "vix_prev_close", "iv"):
            v = c.feature(k, self.sym, prev=self.shift)
            return v if math.isfinite(v) else self._na({"iv": "option chain IV"}.get(k, "VIX"))
        if k in ("iv_rank", "iv_percentile"):
            v = c.iv_rank(self.sym, 252, pct=(k == "iv_percentile"))
            return v if math.isfinite(v) else self._na("IV history")
        if s is None:
            return None
        if k == "price":
            return self._f(s.price_at(self.t, c.today), "price")
        if k == "open":
            return self._f(s.session_open(c.today), "session open")
        if k in ("prev_close", "prev_open", "prev_high", "prev_low"):
            i = s.prev_day_index(c.today)
            return self._f({"prev_close": s.d_c, "prev_open": s.d_o, "prev_high": s.d_h, "prev_low": s.d_l}[k][i], "previous session") if i >= 0 else self._na("previous session")
        if k in ("high", "low", "range_pct", "volume"):
            a, b = s.today_slice(c.today, self.t)
            if b <= a:
                o = s.session_open(c.today)
                return {"high": o, "low": o, "range_pct": 0.0, "volume": 0.0}[k] if math.isfinite(o) and c.minute <= 571 else self._na("intraday bars")
            if k == "high":
                return float(s.h[a:b].max())
            if k == "low":
                return float(s.l[a:b].min())
            if k == "volume":
                return float(s.v[a:b].sum())
            return float((s.h[a:b].max() - s.l[a:b].min()) / s.o[a] * 100)
        if k in ("vwap", "dist_from_vwap_pct"):
            if s.resolution(c.today) > 300:
                return self._na("intraday bars (VWAP)")
            v = s.indicator("vwap", {}, "1m", "value", self.t, c.today)
            if not math.isfinite(v):
                return self._na("intraday bars (VWAP)")
            return v if k == "vwap" else (self.name("price") / v - 1) * 100
        if k in ("max_decline_from_open_pct", "max_rally_from_open_pct", "pullback_from_high_pct", "bounce_from_low_pct"):
            o, hi, lo, p = self.name("open"), self.name("high"), self.name("low"), self.name("price")
            if None in (o, hi, lo, p):
                return None
            if k == "max_decline_from_open_pct":
                return min(0.0, (lo / o - 1) * 100)
            if k == "max_rally_from_open_pct":
                return max(0.0, (hi / o - 1) * 100)
            return min(0.0, (p / hi - 1) * 100) if k == "pullback_from_high_pct" else max(0.0, (p / lo - 1) * 100)
        if k == "gap_pct":
            o, pc = self.name("open"), self.name("prev_close")
            return None if o is None or pc is None else (o / pc - 1) * 100
        if k == "change_pct":
            p, pc = self.name("price"), self.name("prev_close")
            return None if p is None or pc is None else (p / pc - 1) * 100
        if k == "from_open_pct":
            p, o = self.name("price"), self.name("open")
            return None if p is None or o is None else (p / o - 1) * 100
        raise ExprError(f"{k} must be used as {REGISTRY[k]['signature']}")

    def _f(self, v, what):
        v = float(v)
        return v if math.isfinite(v) else self._na(what)

    def attr(self, obj: str, a: str):
        c = self.ctx
        if obj == "underlying":
            if a == "price":
                return self.name("price")
            raise ExprError("underlying has only .price")
        if obj == "previous_session":
            s = self.ser()
            if s is None:
                return None
            i = s.prev_day_index(c.today)
            if i < 0:
                return self._na("previous session")
            if a in ("open", "high", "low", "close", "volume"):
                return float({"open": s.d_o, "high": s.d_h, "low": s.d_l, "close": s.d_c, "volume": s.d_v}[a][i])
            if a == "open_to_close_pct":
                return float((s.d_c[i] / s.d_o[i] - 1) * 100)
            if a == "range_pct":
                return float((s.d_h[i] - s.d_l[i]) / s.d_o[i] * 100)
            if i < 1:
                return self._na("two prior sessions")
            if a == "return_pct":
                return float((s.d_c[i] / s.d_c[i - 1] - 1) * 100)
            if a == "gap_pct":
                return float((s.d_o[i] / s.d_c[i - 1] - 1) * 100)
            raise ExprError(f"previous_session has no field {a!r}")
        if obj == "session":
            m = {"open": "open", "high": "high", "low": "low", "return_pct": "change_pct", "from_open_pct": "from_open_pct", "gap_pct": "gap_pct", "range_pct": "range_pct", "volume": "volume"}
            if a not in m:
                raise ExprError(f"session has no field {a!r}")
            return self.name(m[a])
        p = self.pos
        if p is None:
            raise ExprError(f"{obj}.{a} is only available in exit and add rules (it reads an open position)")
        if obj == "group":
            gid = p.meta.get("group") or p.id
            acct = c.bt.account
            lots = [x for x in list(acct.positions) + list(acct.closed) if (x.meta.get("group") or x.id) == gid]
            open_lots = [x for x in lots if x.is_open]
            if a == "lots":
                return float(len(open_lots))
            if a == "avg_underlying_entry":
                w = sum(x.units for x in lots)
                return float(sum(x.entry_spot * x.units for x in lots) / w) if w else None
            pnl = sum(x.gross_pnl(False) for x in open_lots) + sum(x.entry_cash + x.exit_cash for x in lots if not x.is_open)
            if a == "pnl":
                return float(pnl)
            basis = sum(x.basis for x in lots)
            return float(pnl / basis * 100) if basis else None
        if obj == "entry":
            if a == "signal_price":
                return _num(p.meta.get("signal_price")) or float(p.entry_spot)
            if a == "vwap":
                v = _num(p.meta.get("entry_vwap"))
                return v if v is not None else self._na("VWAP at entry")
            if a == "session_open":
                v = _num(p.meta.get("entry_session_open"))
                return v if v is not None else self._na("session open at entry")
            if a == "underlying_price":
                return float(p.entry_spot)
            if a == "premium":
                return float(abs(p.entry_cash) / max(p.units, 1) / max((p.legs[0].multiplier if p.legs else 100), 1))
            if a == "time":
                return float(cal.minute_of_day_et(p.entry_ts * 1_000_000_000))
            if a == "minutes_ago":
                return float((c.t - p.entry_ts) / 60)
            raise ExprError(f"entry has no field {a!r}")
        if a == "pnl_pct":
            return float(c.position_pnl_pct(p))
        if a == "pnl":
            return float(p.gross_pnl(False))
        if a == "dte":
            return float(p.front_dte(c.today_num))
        if a == "minutes_held":
            return float((c.t - p.entry_ts) / 60)
        if a == "sessions_held":
            from strata.engine.market import from_day_num
            return float(cal.trading_days_between(from_day_num(p.entry_day), c.today))
        if a == "delta":
            return float(p.greeks()[0] / max(p.units, 1))
        if a in ("max_gain_pct", "max_loss_pct"):
            return float((p.mfe if a == "max_gain_pct" else p.mae) / p.basis * 100) if p.basis else None
        if a in ("underlying_high", "underlying_low", "underlying_move_pct"):
            s = self.ser(p.symbol)
            if s is None:
                return None
            px = s.price_at(self.t, c.today)
            if a == "underlying_move_pct":
                return float((px / p.entry_spot - 1) * 100) if math.isfinite(px) and p.entry_spot else self._na("price")
            hi, lo = s.range_between(p.entry_ts, self.t)
            if a == "underlying_high":
                return float(max(p.entry_spot, hi)) if math.isfinite(hi) else float(p.entry_spot)
            return float(min(p.entry_spot, lo)) if math.isfinite(lo) else float(p.entry_spot)
        if a == "lot":
            return float(p.meta.get("lot", 1))
        if a == "adds":
            gid = p.meta.get("group") or p.id
            return float(sum(1 for x in list(c.bt.account.positions) + list(c.bt.account.closed) if x.meta.get("group") == gid and x.meta.get("lot", 1) > 1))
        raise ExprError(f"position has no field {a!r}")

    # ---- event-sequence state (today's stages, or the stages recorded on the position at entry)
    def _stage(self, name):
        st = None
        if self.pos is not None:
            st = (self.pos.meta.get("seq") or {}).get(str(name))
        if st is None:
            tr = getattr(self.ctx, "seq", None)
            st = tr.state.get(str(name)) if tr is not None else None
        return st

    def f_occurred(self, name):
        return self._stage(name) is not None

    def f_stage_price(self, name):
        st = self._stage(name)
        return None if st is None else float(st["price"])  # not yet: unknown, but not missing data

    def f_stage_time(self, name):
        st = self._stage(name)
        return None if st is None else float(st["minute"])

    def f_minutes_since(self, name):
        st = self._stage(name)
        return None if st is None else float((self.t - st["t"]) / 60)

    def _extreme_since(self, name, high: bool):
        st = self._stage(name)
        s = self.ser()
        if st is None or s is None:
            return None
        hi, lo = s.range_between(st["t"], self.t)
        base = float(st["price"])
        if high:
            return max(base, hi) if math.isfinite(hi) else base
        return min(base, lo) if math.isfinite(lo) else base

    def f_low_since(self, name):
        return self._extreme_since(name, False)

    def f_high_since(self, name):
        return self._extreme_since(name, True)

    def f_ref(self, name):
        v = None
        if self.pos is not None:
            v = (self.pos.meta.get("refs") or {}).get(str(name))
        if v is None:
            tr = getattr(self.ctx, "seq", None)
            v = tr.refs.get(str(name)) if tr is not None else None
        return None if v is None else float(v)

    # ---- functions
    def call(self, n: ast.Call):
        f = n.func.id
        if f in ("crosses_above", "crosses_below", "prev"):
            if f == "prev":
                return self._shifted(n.args[0])
            a1, b1 = _num(self._ev(n.args[0])), _num(self._ev(n.args[1]))
            a0, b0 = _num(self._shifted(n.args[0])), _num(self._shifted(n.args[1]))
            if None in (a0, b0, a1, b1):
                return None
            return (a0 <= b0 and a1 > b1) if f == "crosses_above" else (a0 >= b0 and a1 < b1)
        args = [self._ev(a) for a in n.args]
        kw = {k.arg: self._ev(k.value) for k in n.keywords}
        fn = getattr(self, f"f_{f}", None)
        if fn is None:
            raise ExprError(f"{f} is a value, not a function")
        try:
            return fn(*args, **kw)
        except TypeError as e:
            raise ExprError(f"bad arguments for {REGISTRY[f]['signature']}: {e}") from None

    def _shifted(self, node):
        old = self.shift
        self.shift = True
        try:
            return self._ev(node)
        finally:
            self.shift = old

    @staticmethod
    def _split(args, default_sym=None):
        """Leading string argument (if any) is a symbol."""
        if args and isinstance(args[0], str) and args[0] not in TF:
            return args[0], list(args[1:])
        return default_sym, list(args)

    def _completed(self, s) -> int:
        return s.completed_day(self.t)

    def f_abs(self, x):
        x = _num(x)
        return None if x is None else abs(x)

    def f_min(self, *xs):
        v = [_num(x) for x in xs]
        return None if any(x is None for x in v) else min(v)

    def f_max(self, *xs):
        v = [_num(x) for x in xs]
        return None if any(x is None for x in v) else max(v)

    def f_hhmm(self, h, m=0):
        return float(int(h) * 60 + int(m))

    def f_dist_pct(self, a, b):
        a, b = _num(a), _num(b)
        return None if a is None or b in (None, 0) else (a / b - 1) * 100

    def f_retracement_pct(self, a, b):
        a, b, p = _num(a), _num(b), self.name("price")
        return None if None in (a, b, p) or a == b else (p - a) / (b - a) * 100

    def f_return_pct(self, *args, symbol=None, sessions=1):
        sym, rest = self._split(args, symbol)
        n = int(rest[0]) if rest else int(sessions)
        s = self.ser(sym)
        if s is None:
            return None
        i = self._completed(s)
        return float((s.d_c[i] / s.d_c[i - n] - 1) * 100) if i - n >= 0 else self._na(f"{n + 1} completed sessions")

    def _nth(self, args, n, symbol):
        sym, rest = self._split(args, symbol)
        n = int(rest[0]) if rest else int(n)
        s = self.ser(sym)
        if s is None:
            return None, 0
        return s, s.prev_day_index(self.ctx.today) - (n - 1)

    def f_session_return(self, *args, n=1, symbol=None):
        s, i = self._nth(args, n, symbol)
        return float((s.d_c[i] / s.d_c[i - 1] - 1) * 100) if s is not None and i >= 1 else self._na("session history")

    def f_open_to_close(self, *args, n=1, symbol=None):
        s, i = self._nth(args, n, symbol)
        return float((s.d_c[i] / s.d_o[i] - 1) * 100) if s is not None and i >= 0 else self._na("session history")

    def f_close(self, *args, n=1, symbol=None):
        s, i = self._nth(args, n, symbol)
        return float(s.d_c[i]) if s is not None and i >= 0 else self._na("session history")

    def f_session_high(self, *args, n=1, symbol=None):
        s, i = self._nth(args, n, symbol)
        return float(s.d_h[i]) if s is not None and i >= 0 else self._na("session history")

    def f_session_low(self, *args, n=1, symbol=None):
        s, i = self._nth(args, n, symbol)
        return float(s.d_l[i]) if s is not None and i >= 0 else self._na("session history")

    def _streak(self, sym, test) -> float | None:
        s = self.ser(sym)
        if s is None:
            return None
        i, k = self._completed(s), 0
        while i - k >= 1 and test(s, i - k):
            k += 1
        return float(k)

    def f_consecutive_down_days(self, symbol=None):
        return self._streak(symbol, lambda s, i: s.d_c[i] < s.d_c[i - 1])

    def f_consecutive_up_days(self, symbol=None):
        return self._streak(symbol, lambda s, i: s.d_c[i] > s.d_c[i - 1])

    def f_consecutive_red_days(self, symbol=None):
        return self._streak(symbol, lambda s, i: s.d_c[i] < s.d_o[i])

    def f_consecutive_green_days(self, symbol=None):
        return self._streak(symbol, lambda s, i: s.d_c[i] > s.d_o[i])

    def _candles(self, tf: str, bearish: bool):
        s = self.ser()
        if s is None:
            return None
        m = s.tf_minutes(tf)
        if s.resolution(self.ctx.today) > m * 60:
            return self._na(f"{tf} bars")
        ts, end, o, h, l, c, v, dd = s._tf(m)
        i = int(np.searchsorted(end, self.t, side="right")) - 1
        k, today = 0, self.ctx.today_num
        while i - k >= 0 and dd[i - k] == today and ((c[i - k] < o[i - k]) if bearish else (c[i - k] > o[i - k])):
            k += 1
        return float(k)

    def f_consecutive_bearish_candles(self, tf="5m"):
        return self._candles(tf, True)

    def f_consecutive_bullish_candles(self, tf="5m"):
        return self._candles(tf, False)

    def _period_level(self, kind: str, high: bool):
        s = self.ser()
        if s is None:
            return None
        key = ("period", kind, high, self.ctx.today_num)
        if key not in s._memo:
            from strata.engine.market import from_day_num
            today = self.ctx.today
            if kind == "week":
                start = today - timedelta(days=today.weekday() + 7)
                stop = start + timedelta(days=6)
            else:
                first = today.replace(day=1)
                stop = first - timedelta(days=1)
                start = stop.replace(day=1)
            a = int(np.searchsorted(s.days, (start - date(1970, 1, 1)).days))
            b = int(np.searchsorted(s.days, (stop - date(1970, 1, 1)).days, side="right"))
            s._memo[key] = (float(s.d_h[a:b].max()) if high else float(s.d_l[a:b].min())) if b > a else None
        v = s._memo[key]
        return v if v is not None else self._na(f"previous {kind}")

    def f_prev_week_high(self):
        return self._period_level("week", True)

    def f_prev_week_low(self):
        return self._period_level("week", False)

    def f_prev_month_high(self):
        return self._period_level("month", True)

    def f_prev_month_low(self):
        return self._period_level("month", False)

    def _rolling(self, n, tf, high):
        s = self.ser()
        if s is None:
            return None
        n = int(n)
        if tf == "1d":
            i = self._completed(s)
            if i + 1 < n:
                return self._na(f"{n} sessions")
            return float(s.d_h[i - n + 1:i + 1].max() if high else s.d_l[i - n + 1:i + 1].min())
        m = s.tf_minutes(tf)
        if s.resolution(self.ctx.today) > m * 60:
            return self._na(f"{tf} bars")
        ts, end, o, h, l, c, v, dd = s._tf(m)
        i = int(np.searchsorted(end, self.t, side="right")) - 1
        if i + 1 < n:
            return self._na(f"{n} {tf} bars")
        return float(h[i - n + 1:i + 1].max() if high else l[i - n + 1:i + 1].min())

    def f_rolling_high(self, n=20, tf="1d"):
        return self._rolling(n, tf, True)

    def f_rolling_low(self, n=20, tf="1d"):
        return self._rolling(n, tf, False)

    def _or(self, minutes, high):
        s = self.ser()
        if s is None:
            return None
        if s.resolution(self.ctx.today) > int(minutes) * 60:
            return self._na("intraday bars (opening range)")
        hi, lo, complete = s.session_range_until(self.ctx.today, self.t, int(minutes))
        return (hi if high else lo) if complete else None  # still forming: unknown yet, but not missing data

    def f_opening_range_high(self, minutes=30):
        return self._or(minutes, True)

    def f_opening_range_low(self, minutes=30):
        return self._or(minutes, False)

    def f_gap_fill_pct(self):
        o, pc, hi, lo = self.name("open"), self.name("prev_close"), self.name("high"), self.name("low")
        if None in (o, pc, hi, lo) or o == pc:
            return None if o != pc else 100.0
        return float(np.clip((hi - o) / (pc - o) if o < pc else (o - lo) / (o - pc), 0, 10) * 100)

    def _today(self):
        s = self.ser()
        if s is None:
            return None, 0, 0
        a, b = s.today_slice(self.ctx.today, self.t)
        return s, a, b

    def f_touched(self, level):
        level = _num(level)
        s, a, b = self._today()
        if s is None or level is None:
            return None
        if b <= a:
            return self._na("intraday bars")
        return bool(s.l[a:b].min() <= level <= s.h[a:b].max())

    def f_first_touch(self, la, lb):
        la, lb = _num(la), _num(lb)
        s, a, b = self._today()
        if s is None or la is None or lb is None:
            return None
        if b <= a:
            return 0.0 if s.resolution(self.ctx.today) <= 3600 else self._na("intraday bars")
        h, l = s.h[a:b], s.l[a:b]
        pc = np.r_[s.o[a], s.c[a:b - 1]]  # gaps between bars count as touches
        ta = np.nonzero((np.minimum(l, pc) <= la) & (la <= np.maximum(h, pc)))[0]
        tb = np.nonzero((np.minimum(l, pc) <= lb) & (lb <= np.maximum(h, pc)))[0]
        ia, ib = (ta[0] if ta.size else 10**9), (tb[0] if tb.size else 10**9)
        if ia == ib:
            return 0.0 if ia == 10**9 else self._na("first-touch order (both levels inside one bar)")
        return 1.0 if ia < ib else 2.0

    def _time_side(self, level, above):
        level = _num(level)
        s, a, b = self._today()
        if s is None or level is None:
            return None
        if s.resolution(self.ctx.today) > 900:
            return self._na("intraday bars")
        m = (s.c[a:b] > level) if above else (s.c[a:b] < level)
        return float(((s.end[a:b] - s.ts[a:b])[m]).sum() / 60)

    def f_time_above(self, level):
        return self._time_side(level, True)

    def f_time_below(self, level):
        return self._time_side(level, False)

    def _ind(self, name, params, tf, field=None):
        s = self.ser()
        if s is None:
            return None
        v = s.indicator(name, params, tf, field, self.t, self.ctx.today)
        return v if math.isfinite(v) else self._na(f"{name}({tf}) history")

    def f_rsi(self, n=14, tf="5m"):
        return self._ind("rsi", {"period": int(n)}, tf)

    def f_sma(self, n=20, tf="1d"):
        return self._ind("sma", {"period": int(n)}, tf)

    def f_ema(self, n=20, tf="1d"):
        return self._ind("ema", {"period": int(n)}, tf)

    def f_atr(self, n=14, tf="1d"):
        return self._ind("atr", {"period": int(n)}, tf, "value")

    def f_atr_pct(self, n=14, tf="1d"):
        return self._ind("atr", {"period": int(n)}, tf, "pct")

    def f_macd(self, tf="5m"):
        return self._ind("macd", {}, tf, "line")

    def f_macd_signal(self, tf="5m"):
        return self._ind("macd", {}, tf, "signal")

    def f_macd_hist(self, tf="5m"):
        return self._ind("macd", {}, tf, "hist")

    def f_bb_upper(self, n=20, k=2.0, tf="5m"):
        return self._ind("bbands", {"period": int(n), "k": float(k)}, tf, "upper")

    def f_bb_lower(self, n=20, k=2.0, tf="5m"):
        return self._ind("bbands", {"period": int(n), "k": float(k)}, tf, "lower")

    def f_bb_pctb(self, n=20, k=2.0, tf="5m"):
        return self._ind("bbands", {"period": int(n), "k": float(k)}, tf, "pctb")

    def _rvol_series(self, s, n):
        key = ("rvol", int(n))
        if key not in s._memo:
            r = np.diff(np.log(s.d_c))
            out = np.full(len(s.d_c), np.nan)
            for i in range(int(n), len(s.d_c)):
                out[i] = r[i - int(n):i].std(ddof=1) * math.sqrt(252) * 100
            s._memo[key] = out
        return s._memo[key]

    def f_rvol(self, n=21):
        s = self.ser()
        if s is None:
            return None
        i = self._completed(s)
        v = self._rvol_series(s, n)[i] if i >= 0 else float("nan")
        return float(v) if math.isfinite(v) else self._na(f"{n} sessions of history")

    def f_rvol_percentile(self, n=21, lookback=252):
        s = self.ser()
        if s is None:
            return None
        i = self._completed(s)
        arr = self._rvol_series(s, n)
        w = arr[max(0, i - int(lookback)):i]
        w = w[np.isfinite(w)]
        if i < 0 or not math.isfinite(arr[i]) or w.size < 20:
            return self._na("volatility history")
        return float((w < arr[i]).mean() * 100)

    def f_volume_ratio(self, tf="5m", n=20):
        s = self.ser()
        if s is None:
            return None
        m = s.tf_minutes(tf)
        if s.resolution(self.ctx.today) > m * 60:
            return self._na(f"{tf} bars")
        ts, end, o, h, l, c, v, dd = s._tf(m)
        i = int(np.searchsorted(end, self.t, side="right")) - 1
        if i < int(n):
            return self._na("volume history")
        base = v[i - int(n):i].mean()
        return float(v[i] / base) if base > 0 else self._na("volume")

    def f_day_volume_ratio(self, n=20):
        s = self.ser()
        if s is None:
            return None
        i = self._completed(s)
        if i < int(n):
            return self._na("volume history")
        base = s.d_v[i - int(n):i].mean()
        return float(s.d_v[i] / base) if base > 0 else self._na("volume")

    def f_relative_strength(self, a, b, sessions=20):
        ra, rb = self.f_return_pct(a, sessions=sessions), self.f_return_pct(b, sessions=sessions)
        return None if ra is None or rb is None else ra - rb

    def f_event_today(self, *types):
        return bool(self.ctx.events.on(self.ctx.today, list(types) or None))

    def f_event_within(self, n, *types):
        return bool(self.ctx.events.within(self.ctx.today, int(n), list(types) or None))

    # values that are also callable with a symbol, e.g. price("QQQ")
    def f_price(self, symbol=None):
        s = self.ser(symbol)
        return None if s is None else self._f(s.price_at(self.t, self.ctx.today), f"{symbol or self.sym} price")

    def f_change_pct(self, symbol=None):
        s = self.ser(symbol)
        if s is None:
            return None
        i = s.prev_day_index(self.ctx.today)
        p = s.price_at(self.t, self.ctx.today)
        return float((p / s.d_c[i] - 1) * 100) if i >= 0 and math.isfinite(p) else self._na(f"{symbol} prices")

    def f_gap_pct(self, symbol=None):
        s = self.ser(symbol)
        if s is None:
            return None
        i = s.prev_day_index(self.ctx.today)
        o = s.session_open(self.ctx.today)
        return float((o / s.d_c[i] - 1) * 100) if i >= 0 and math.isfinite(o) else self._na(f"{symbol} prices")


def explain(src: str, ev: Evaluator) -> tuple[bool | None, str]:
    """Evaluate and describe: shows the numbers behind each comparison."""
    tree = parse(src).body

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, bool):
            return "true" if v else "false"
        return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)

    def side(n) -> str:
        try:
            return str(ast.literal_eval(n))  # plain literal such as 30 or -3
        except (ValueError, TypeError, SyntaxError):
            return f"{ast.unparse(n)}={fmt(ev._ev(n))}"

    parts = []

    def walk(n):
        if isinstance(n, ast.BoolOp):
            for v in n.values:
                walk(v)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1:
            sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[type(n.ops[0])]
            parts.append(f"{side(n.left)} {sym} {side(n.comparators[0])}")
        else:
            parts.append(f"{ast.unparse(n)}={fmt(ev._ev(n))}")

    res = ev._ev(tree)
    walk(tree)
    res = None if res is None else bool(res)
    text = "; ".join(parts)
    if res is None:
        text += f" [not evaluable: missing {', '.join(sorted(ev.missing)) or 'data'}]"
    return res, text
