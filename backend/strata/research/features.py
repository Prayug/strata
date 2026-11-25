"""Per-session feature table: everything a hypothesis can condition on, computed once.

Discovery, sensitivity maps and the market-behaviour studies all ask the same kind of
question thousands of times: "on the sessions where <features satisfy something>, what
happened next?". Evaluating each variant session by session through the rule engine repeats
the same work, so this module computes every feature for every stored session in one
vectorised pass and answers each variant with array masks.

It is NOT a second definition of the rules. Each feature is tied to an expression of the
strategy rule language (`FEATURES[name].expr`) and a regression test evaluates both on real
sessions and requires them to agree, so a configuration found here can be handed to the
session-by-session engine (`to_study`) and reproduces. Features that need intraday bars
are evaluated by the rule engine itself and cached.

Point-in-time: a value used at a session's open only reads sessions that had finished
before that open; a value used at the close may also read the session that just closed.
Outcomes are measured strictly after the entry instant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from strata.core import calendar as cal
from strata.core import indicators as ind
from strata.data.store import MarketStore
from strata.engine.market import UnderlyingSeries, day_num, from_day_num

NAN = float("nan")
T, S, N_, A = 0, 1, 2, 3  # outcome codes: target, stop, neither, ambiguous (-1 = not evaluated)
CODE_NAMES = ("target", "stop", "neither", "ambiguous")


@dataclass(frozen=True)
class FeatureDef:
    name: str
    label: str
    unit: str
    group: str
    expr: str  # rule-language expression with the same value at the tick the feature is read
    kind: str = "asof"  # asof: as of the last completed session | session: today's own | cross | intraday
    words: str = ""  # how a threshold reads in a sentence: "{name} {op} {value}{unit}" when empty
    expr_close: str | None = None  # the expression to use at the closing tick when it differs


def _f(name, label, unit, group, expr, kind="asof", words="", expr_close=None):
    return FeatureDef(name, label, unit, group, expr, kind, words, expr_close)


FEATURES: dict[str, FeatureDef] = {f.name: f for f in [
    _f("ret_1", "Last completed session's return", "%", "Previous sessions", "return_pct(1)", words="the last session's close-to-close return"),
    _f("ret_2", "Return over the last 2 sessions", "%", "Previous sessions", "return_pct(2)", words="the 2-session return"),
    _f("ret_5", "Return over the last 5 sessions", "%", "Previous sessions", "return_pct(5)", words="the 5-session return"),
    _f("ret_20", "Return over the last 20 sessions", "%", "Previous sessions", "return_pct(20)", words="the 20-session return"),
    _f("oc_1", "Last completed session, open to close", "%", "Previous sessions", "open_to_close(1)", words="the last session's open-to-close move", expr_close="dist_pct(sma(1), open)"),
    _f("range_1", "Last completed session's range", "%", "Historical ranges", "previous_session.range_pct", words="the last session's high-low range", expr_close="range_pct"),
    _f("down_streak", "Consecutive lower closes", "days", "Streaks", "consecutive_down_days()", words="consecutive lower closes"),
    _f("up_streak", "Consecutive higher closes", "days", "Streaks", "consecutive_up_days()", words="consecutive higher closes"),
    _f("red_streak", "Consecutive red sessions (close below open)", "days", "Streaks", "consecutive_red_days()", words="consecutive red sessions"),
    _f("green_streak", "Consecutive green sessions (close above open)", "days", "Streaks", "consecutive_green_days()", words="consecutive green sessions"),
    _f("sma20_dist", "Last close vs 20-day average", "%", "Moving averages", "dist_pct(sma(1), sma(20))", words="the last close relative to its 20-day average"),
    _f("sma50_dist", "Last close vs 50-day average", "%", "Moving averages", "dist_pct(sma(1), sma(50))", words="the last close relative to its 50-day average"),
    _f("sma200_dist", "Last close vs 200-day average", "%", "Moving averages", "dist_pct(sma(1), sma(200))", words="the last close relative to its 200-day average"),
    _f("rv21", "Realised volatility (21 sessions, annualised)", "%", "Volatility", "rvol(21)", words="21-session realised volatility"),
    _f("rv_pctile", "Realised-volatility percentile (1 year)", "", "Volatility", "rvol_percentile(21, 252)", words="the realised-volatility percentile"),
    _f("vix_prev", "Volatility index, previous close", "", "Volatility", "vix_prev_close", kind="session", words="the volatility index (previous close)"),
    _f("vol_ratio", "Last session's volume vs 20-day average", "x", "Volume", "day_volume_ratio(20)", words="the last session's volume relative to its 20-day average"),
    _f("dist_high20", "Last close vs 20-day high", "%", "Historical ranges", "dist_pct(sma(1), rolling_high(20))", words="the last close relative to the 20-day high"),
    _f("dist_low20", "Last close vs 20-day low", "%", "Historical ranges", "dist_pct(sma(1), rolling_low(20))", words="the last close relative to the 20-day low"),
    _f("dist_high252", "Drawdown from the 1-year high", "%", "Historical ranges", "dist_pct(sma(1), rolling_high(252))", words="the drawdown from the 1-year high"),
    _f("gap", "Opening gap", "%", "Overnight", "gap_pct", kind="session", words="the opening gap"),
    _f("weekday", "Day of week (0 = Monday)", "", "Calendar", "weekday", kind="session", words="the weekday"),
    # intraday, read at the entry time (evaluated by the rule engine)
    _f("from_open", "Move since the open, at the entry time", "%", "Intraday", "from_open_pct", kind="intraday", words="the move since the open"),
    _f("vwap_dist", "Price vs session VWAP, at the entry time", "%", "Intraday", "dist_from_vwap_pct", kind="intraday", words="the price relative to VWAP"),
    _f("range_so_far", "Session range so far, at the entry time", "%", "Intraday", "range_pct", kind="intraday", words="the session range so far"),
    _f("or15_pos", "Position within the first 15 minutes' range (0 = low, 100 = high)", "", "Intraday", "retracement_pct(opening_range_low(15), opening_range_high(15))", kind="intraday",
       words="the position within the 15-minute opening range"),
    _f("or30_pos", "Position within the first 30 minutes' range (0 = low, 100 = high)", "", "Intraday", "retracement_pct(opening_range_low(30), opening_range_high(30))", kind="intraday",
       words="the position within the 30-minute opening range"),
    _f("or60_pos", "Position within the first 60 minutes' range (0 = low, 100 = high)", "", "Intraday", "retracement_pct(opening_range_low(60), opening_range_high(60))", kind="intraday",
       words="the position within the 60-minute opening range"),
    _f("bar_vol_ratio", "Last 5-minute bar's volume vs the previous 20 bars", "x", "Intraday", 'volume_ratio("5m", 20)', kind="intraday", words="the last 5-minute bar's relative volume"),
]}


def cross_feature(name: str, base: str) -> FeatureDef | None:
    """Cross-symbol features are named <SYMBOL>_ret_1, <SYMBOL>_gap and rs5_<SYMBOL>."""
    if name.startswith("rs5_"):
        x = name[4:]
        return FeatureDef(name, f"{base} minus {x}, 5-session return", "pts", "Cross-symbol", f'relative_strength("{base}", "{x}", 5)', "cross", f"{base}'s 5-session return minus {x}'s")
    if name.endswith("_ret_1"):
        x = name[:-6]
        return FeatureDef(name, f"{x}: last completed session's return", "%", "Cross-symbol", f'return_pct("{x}", 1)', "cross", f"{x}'s last close-to-close return")
    if name.endswith("_gap"):
        x = name[:-4]
        return FeatureDef(name, f"{x}: opening gap", "%", "Cross-symbol", f"{x}.gap_pct", "cross", f"{x}'s opening gap")
    return None


def feature_def(name: str, base: str = "SPY") -> FeatureDef:
    f = FEATURES.get(name) or cross_feature(name, base)
    if f is None:
        raise KeyError(f"unknown feature {name!r}")
    return f


# ----------------------------------------------------------------------------- conditions
OPS = {">=": "at least", "<=": "at most", ">": "above", "<": "below", "==": "equal to", "abs>=": "at least (either direction)", "between": "between"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass(frozen=True)
class Pred:
    feature: str
    op: str
    value: float
    hi: float | None = None  # for "between": value <= x < hi

    def mask(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            if self.op == ">=":
                return x >= self.value
            if self.op == "<=":
                return x <= self.value
            if self.op == ">":
                return x > self.value
            if self.op == "<":
                return x < self.value
            if self.op == "==":
                return x == self.value
            if self.op == "abs>=":
                return np.abs(x) >= self.value
            if self.op == "between":
                return (x >= self.value) & (x < self.hi)
        raise ValueError(f"unknown operator {self.op}")

    def expr(self, base: str, entry: str) -> str:
        f = feature_def(self.feature, base)
        e = f.expr_close if (entry == "close" and f.expr_close) else f.expr
        if self.op == "abs>=":
            return f"abs({e}) >= {self.value:g}"
        if self.op == "between":
            return f"({e} >= {self.value:g}) and ({e} < {self.hi:g})"
        return f"{e} {self.op} {self.value:g}"

    def text(self, base: str = "SPY") -> str:
        f = feature_def(self.feature, base)
        u = f.unit if f.unit in ("%", "x") else (" " + f.unit if f.unit else "")
        if self.feature == "weekday" and self.op == "==":
            return f"it is a {WEEKDAYS[int(self.value)]}"
        w = f.words or f.label.lower()
        if self.op == "between":
            return f"{w} is between {self.value:g}{u} and {self.hi:g}{u}"
        if self.op == "abs>=":
            return f"{w} is {self.value:g}{u} or more in either direction"
        return f"{w} is {OPS[self.op]} {self.value:g}{u}"

    def dump(self) -> dict:
        return {"feature": self.feature, "op": self.op, "value": self.value, **({"hi": self.hi} if self.hi is not None else {})}

    @staticmethod
    def load(d: dict) -> "Pred":
        return Pred(d["feature"], d["op"], float(d["value"]), None if d.get("hi") is None else float(d["hi"]))


def cond_text(preds: list[Pred], base: str = "SPY") -> str:
    return " and ".join(p.text(base) for p in preds) if preds else "every session"


def cond_tree(preds: list[Pred], base: str, entry: str) -> dict:
    """The same condition as a rule-engine tree (what a strategy or study uses)."""
    return {"id": "root", "kind": "group", "op": "and",
            "children": [{"id": f"c{i}", "kind": "cond", "type": "expr", "params": {"expr": p.expr(base, entry), "label": p.text(base)}} for i, p in enumerate(preds)]}


# ----------------------------------------------------------------------------- outcomes
@dataclass(frozen=True)
class OutcomeSpec:
    entry: str = "open"  # open | close | HH:MM (New York)
    horizon: str = "session"  # session | sessions:N (through the close of the N-th following session) | time:HH:MM
    direction: str = "up"
    target: float | None = 0.0  # % from the entry price in `direction`; 0 with measure=close means "finishes beyond the entry"
    stop: float | None = None  # % against
    measure: str = "close"  # close: where the horizon ends | touch: is the target traded before the stop

    def key(self) -> str:
        return f"{self.entry}|{self.horizon}|{self.direction}|{self.target}|{self.stop}|{self.measure}"

    def text(self) -> str:
        if self.direction == "either":
            hz = "during the session" if self.horizon == "session" else (f"within {self.horizon.split(':')[1]} more session(s)" if self.horizon.startswith("sessions:") else f"by {self.horizon.split(':', 1)[1]} ET")
            return (f"finishes {self.target:g}% or more away from the entry price, {hz}" if self.measure == "close" else f"trades {self.target:g}% or more away from the entry price in either direction, {hz}")
        side = "above" if self.direction == "up" else "below"
        hz = {"session": "by the close"}.get(self.horizon) or (f"within {self.horizon.split(':')[1]} more session{'s' if self.horizon.split(':')[1] != '1' else ''}" if self.horizon.startswith("sessions:") else f"by {self.horizon.split(':', 1)[1]} ET")
        if self.measure == "close":
            lvl = "its entry price" if not self.target else f"{self.target:g}% {side} its entry price"
            end = {"session": "the session closes"}.get(self.horizon) or (f"the close {self.horizon.split(':')[1]} session{'s' if self.horizon.split(':')[1] != '1' else ''} later is" if self.horizon.startswith("sessions:") else f"the price at {self.horizon.split(':', 1)[1]} ET is")
            return f"{end} {side} {lvl}" if self.horizon == "session" else f"{end} {side} {lvl}"
        t = f"trades {self.target:g}% {side} the entry price"
        return t + (f" before {self.stop:g}% the other way" if self.stop else "") + f", {hz}"

    def dump(self) -> dict:
        return {"entry": self.entry, "horizon": self.horizon, "direction": self.direction, "target": self.target, "stop": self.stop, "measure": self.measure}

    @staticmethod
    def load(d: dict) -> "OutcomeSpec":
        return OutcomeSpec(d.get("entry", "open"), d.get("horizon", "session"), d.get("direction", "up"), d.get("target", 0.0), d.get("stop"), d.get("measure", "close"))


def to_study(symbol: str, preds: list[Pred], oc: OutcomeSpec, start=None, end=None, label: str = "") -> dict:
    """The StudyConfig that asks the session-by-session engine the same question."""
    sgn = 1.0 if oc.direction == "up" else -1.0
    hz = {"kind": "session"}
    if oc.horizon.startswith("sessions:"):
        hz = {"kind": "sessions", "sessions": int(oc.horizon.split(":")[1])}
    elif oc.horizon.startswith("time:"):
        hz = {"kind": "time", "time": oc.horizon.split(":", 1)[1]}
    cfg = {"symbol": symbol, "condition": cond_tree(preds, symbol, oc.entry), "entry_time": oc.entry, "reference": "entry", "measure": oc.measure, "direction": oc.direction, "horizon": hz,
           "target": None if oc.target is None else {"kind": "pct", "value": sgn * oc.target}, "stop": None if oc.stop is None else {"kind": "pct", "value": -sgn * oc.stop}, "label": label}
    if start:
        cfg["start"] = str(start)
    if end:
        cfg["end"] = str(end)
    return cfg


@dataclass
class Outcome:
    code: np.ndarray  # int8 per session: 0 target, 1 stop, 2 neither, 3 ambiguous, -1 not evaluated
    end_ret: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray
    entry_px: np.ndarray
    span: np.ndarray  # index of the last session the observation runs through
    reasons: dict = field(default_factory=dict)  # why sessions were not evaluated -> count

    @property
    def valid(self) -> np.ndarray:
        return self.code >= 0


# ----------------------------------------------------------------------------- the table
class SessionFeatures:
    """All sessions of one symbol with their features and lazily computed outcomes."""

    def __init__(self, store: MarketStore | None, symbol: str, cross: tuple[str, ...] = (), end: date | None = None, series: dict[str, UnderlyingSeries] | None = None, vol_symbol: str | None = None):
        self.store = store
        self.symbol = symbol.upper()
        self.cross = tuple(x.upper() for x in cross)
        if series is not None:
            self.series = series
            self.vol_symbol = vol_symbol
            self.host = None
        else:
            from strata.research.study import StudyConfig, _Host
            store = store or MarketStore()
            self.store = store
            self.host = _Host(store, StudyConfig(symbol=self.symbol, end=end))
            for x in self.cross:
                if x not in self.host.series:
                    df = store.read_best_bars(x, None, end)
                    if not df.is_empty():
                        self.host.metas[x] = store.symbol_meta(x)
                        self.host.series[x] = UnderlyingSeries(df, official=store.official_closes(x, None, end))
            self.series = self.host.series
            self.vol_symbol = store.symbol_meta(self.symbol).vol_index
        self.ser = self.series.get(self.symbol)
        if self.ser is None or self.ser.empty():
            raise LookupError(f"No historical price data is stored for {self.symbol}.")
        s = self.ser
        self.n = len(s.days)
        self.dates = [from_day_num(int(x)).isoformat() for x in s.days]
        self.t_open = s.d_open_t.astype(np.int64)
        self.t_close = s.d_end.astype(np.int64)
        self.res = s.d_res.astype(np.int64)
        self._asof: dict[str, np.ndarray] = {}
        self._cols: dict[tuple[str, str], np.ndarray] = {}
        self._outcomes: dict[str, Outcome] = {}
        self._ctx = None
        self.missing_cross = [x for x in self.cross if x not in self.series]

    # ----- as-of arrays: value[a] is what is known once session a has completed
    def _asof_array(self, name: str) -> np.ndarray:
        if name in self._asof:
            return self._asof[name]
        s = self.ser
        c, o, h, l, v = s.d_c, s.d_o, s.d_h, s.d_l, s.d_v
        n = self.n
        out = np.full(n, NAN)
        if name.startswith("ret_"):
            k = int(name[4:])
            out[k:] = (c[k:] / c[:-k] - 1) * 100
        elif name == "oc_1":
            out = (c / o - 1) * 100
        elif name == "range_1":
            out = (h - l) / o * 100
        elif name in ("down_streak", "up_streak", "red_streak", "green_streak"):
            if name == "down_streak":
                t = np.r_[False, c[1:] < c[:-1]]
            elif name == "up_streak":
                t = np.r_[False, c[1:] > c[:-1]]
            else:
                t = (c < o) if name == "red_streak" else (c > o)
                t = t.copy()
                t[0] = False  # the engine's streak never counts the first stored session
            idx = np.arange(n)
            last_false = np.maximum.accumulate(np.where(~t, idx, -1))
            out = (idx - last_false).astype(float)
        elif name in ("sma20_dist", "sma50_dist", "sma200_dist"):
            k = int(name[3:].split("_")[0])
            out = (c / ind.sma(c, k) - 1) * 100
        elif name == "rv21":
            out = self._rvol(21)
        elif name == "rv_pctile":
            arr = self._rvol(21)
            for a in range(n):
                if not math.isfinite(arr[a]):
                    continue
                w = arr[max(0, a - 252):a]
                w = w[np.isfinite(w)]
                if w.size >= 20:
                    out[a] = (w < arr[a]).mean() * 100
        elif name == "vol_ratio":
            cs = np.r_[0.0, np.cumsum(v)]
            a = np.arange(20, n)
            base = (cs[a] - cs[a - 20]) / 20
            with np.errstate(divide="ignore", invalid="ignore"):
                out[20:] = np.where(base > 0, v[20:] / base, NAN)
        elif name in ("dist_high20", "dist_low20", "dist_high252"):
            k = 252 if name.endswith("252") else 20
            if n >= k:
                from numpy.lib.stride_tricks import sliding_window_view
                if "high" in name:
                    lvl = sliding_window_view(h, k).max(axis=1)
                else:
                    lvl = sliding_window_view(l, k).min(axis=1)
                out[k - 1:] = (c[k - 1:] / lvl - 1) * 100
        else:
            raise KeyError(name)
        self._asof[name] = out
        return out

    def _rvol(self, k: int) -> np.ndarray:
        key = f"_rvol{k}"
        if key not in self._asof:
            c = self.ser.d_c
            out = np.full(self.n, NAN)
            if self.n > k:
                from numpy.lib.stride_tricks import sliding_window_view
                r = np.diff(np.log(c))
                sd = sliding_window_view(r, k).std(axis=1, ddof=1) * math.sqrt(252) * 100  # sd[j] covers r[j:j+k]
                out[k:] = sd[:self.n - k]
            self._asof[key] = out
        return self._asof[key]

    def _completed(self, ser: UnderlyingSeries, tick: np.ndarray) -> np.ndarray:
        return np.searchsorted(ser.d_end, tick, side="right") - 1

    def _tick(self, entry: str) -> tuple[np.ndarray, np.ndarray]:
        """(tick time per session, observable mask) for an entry kind."""
        if entry == "open":
            return self.t_open, np.ones(self.n, bool)
        if entry == "close":
            return self.t_close, np.ones(self.n, bool)
        m = cal.parse_hhmm(entry)
        t = self.t_open + (m - 570) * 60
        ok = (t >= self.t_open) & (t <= self.t_close) & (self.res < 86400) & ((t - self.t_open) % np.maximum(self.res, 1) == 0)
        return t, ok

    def col(self, name: str, entry: str = "open") -> np.ndarray:
        """Feature value per session as it can be read at the entry tick (NaN = not evaluable)."""
        key = (name, "close" if entry == "close" else "open" if entry == "open" else entry)
        fd = feature_def(name, self.symbol)
        if fd.kind != "intraday" and entry not in ("open", "close"):
            key = (name, "open")  # values about completed sessions are the same at any time before the close
        if key in self._cols:
            return self._cols[key]
        n = self.n
        s = self.ser
        tick, _ = self._tick("close" if key[1] == "close" else "open")
        out = np.full(n, NAN)
        if fd.kind == "asof":
            a = self._completed(s, tick)
            arr = self._asof_array(name)
            ok = a >= 0
            out[ok] = arr[a[ok]]
        elif name == "gap":
            out[1:] = (s.d_o[1:] / s.d_c[:-1] - 1) * 100
        elif name == "weekday":
            out = np.array([date.fromisoformat(d).weekday() for d in self.dates], float)
        elif name == "vix_prev":
            vs = self.series.get(self.vol_symbol) if self.vol_symbol else None
            if vs is not None and not vs.empty():
                j = np.searchsorted(vs.days, s.days) - 1
                ok = j >= 0
                out[ok] = vs.d_c[j[ok]]
        elif fd.kind == "cross":
            out = self._cross(name, tick)
        elif fd.kind == "intraday":
            if entry in ("open", "close"):
                raise ValueError(f"{fd.label} needs an intraday entry time")
            out = self.engine_values(fd.expr, entry)
        else:
            raise KeyError(name)
        self._cols[key] = out
        return out

    def _cross(self, name: str, tick: np.ndarray) -> np.ndarray:
        out = np.full(self.n, NAN)
        s = self.ser
        if name.startswith("rs5_"):
            x = self.series.get(name[4:])
            if x is None:
                return out
            a, b = self._completed(s, tick), self._completed(x, tick)
            ok = (a >= 5) & (b >= 5)
            out[ok] = (s.d_c[a[ok]] / s.d_c[a[ok] - 5] - 1) * 100 - (x.d_c[b[ok]] / x.d_c[b[ok] - 5] - 1) * 100
        elif name.endswith("_ret_1"):
            x = self.series.get(name[:-6])
            if x is None:
                return out
            b = self._completed(x, tick)
            ok = b >= 1
            out[ok] = (x.d_c[b[ok]] / x.d_c[b[ok] - 1] - 1) * 100
        else:  # <X>_gap: X's own open today against its previous close
            x = self.series.get(name[:-4])
            if x is None:
                return out
            j = np.searchsorted(x.days, s.days)
            ok = (j < len(x.days)) & (j >= 1)
            ok[ok] &= x.days[j[ok]] == s.days[ok]
            out[ok] = (x.d_o[j[ok]] / x.d_c[j[ok] - 1] - 1) * 100
        return out

    # ----- the rule engine, for intraday features and for cross-checking
    def _context(self):
        if self._ctx is None:
            from strata.engine.engine import Context
            from strata.research.study import StudyConfig, _Host, _Strat
            host = self.host
            if host is None:  # built from series directly (tests): a minimal host
                host = _Host.__new__(_Host)
                host.store, host.series, host.metas, host.chains, host.iv_cache, host.iv_hist = self.store, self.series, {}, {}, {}, {}
                from strata.engine.market import Events
                from strata.engine.models import RegimeSettings
                import polars as pl
                host.regime_settings = {0: RegimeSettings()}
                host.events = Events(pl.DataFrame())
                if self.vol_symbol:
                    from strata.data.store import SymbolMeta
                    host.metas[self.symbol] = SymbolMeta(symbol=self.symbol, vol_index=self.vol_symbol)
            ctx = Context(host)
            ctx.strategy = _Strat(self.symbol, self.vol_symbol)
            self._ctx = ctx
        return self._ctx

    def engine_values(self, expr: str, entry: str, sessions: np.ndarray | None = None) -> np.ndarray:
        """Evaluate a rule-language expression with the rule engine at the entry tick of each
        session (NaN where it is not evaluable). Slow path: used for intraday features and tests."""
        from strata.engine.expr import Evaluator, ExprError
        ctx = self._context()
        s = self.ser
        tick, ok = self._tick(entry)
        out = np.full(self.n, NAN)
        idx = np.flatnonzero(ok) if sessions is None else np.asarray(sessions)
        for i in idx:
            if i < 1 or not ok[i]:
                continue
            t, t0 = int(tick[i]), int(self.t_open[i])
            ctx.today, ctx.today_num = from_day_num(int(s.days[i])), int(s.days[i])
            ctx.close_minute = 570 + (int(self.t_close[i]) - t0) // 60
            ctx.t = t
            a = int(s.day_start[i])
            ctx.prev_t = max(t0, int(s.end[max(int(np.searchsorted(s.end, t, side="left")) - 1, a)])) if t > t0 else t
            ctx.minute = 570 + (t - t0) // 60
            ctx._feat.clear()
            ev = Evaluator(ctx)
            try:
                v = ev.eval(expr)
            except ExprError:
                continue
            if isinstance(v, bool):
                out[i] = float(v)
            elif isinstance(v, (int, float)) and math.isfinite(v):
                out[i] = float(v)
        return out

    # ----- outcomes
    def outcome(self, oc: OutcomeSpec) -> Outcome:
        k = oc.key()
        if k not in self._outcomes:
            self._outcomes[k] = self._measure(oc)
        return self._outcomes[k]

    def _levels(self, oc: OutcomeSpec, px):
        up = oc.direction == "up"
        sgn = 1.0 if up else -1.0
        tgt = None if oc.target is None else px * (1 + sgn * oc.target / 100)
        stp = None if oc.stop is None else px * (1 - sgn * oc.stop / 100)
        return up, tgt, stp

    def _measure(self, oc: OutcomeSpec) -> Outcome:
        n, s = self.n, self.ser
        code = np.full(n, -1, np.int8)
        end_ret, mfe, mae, epx = (np.full(n, NAN) for _ in range(4))
        span = np.arange(n)
        reasons: dict[str, int] = {}
        tick, obs = self._tick(oc.entry)
        hold = int(oc.horizon.split(":")[1]) if oc.horizon.startswith("sessions:") else 0
        todo = np.ones(n, bool)
        todo[0] = False  # the first stored session has no previous session to condition on
        if not obs.all():
            bad = todo & ~obs
            if bad.any():
                reasons[f"entry time {oc.entry} ET is not observable with the bars stored for the session"] = int(bad.sum())
            todo &= obs
        if oc.horizon.startswith("time:"):
            th = self.t_open + (cal.parse_hhmm(oc.horizon.split(":", 1)[1]) - 570) * 60
            bad = todo & ((self.res >= 86400) | (th <= tick))
            if bad.any():
                reasons["the horizon needs intraday bars, or ends before the entry"] = int(bad.sum())
            todo &= ~bad
        else:
            last = np.minimum(np.arange(n) + hold, n - 1)
            todo &= (np.arange(n) + hold) <= n - 1  # not enough following sessions yet: skipped, as in the engine
            th = self.t_close[last]
            span = last
        if oc.entry == "close" and hold == 0 and not oc.horizon.startswith("time:"):
            reasons["an observation that starts at the close needs a horizon of one or more sessions"] = int(todo.sum())
            return Outcome(code, end_ret, mfe, mae, epx, span, reasons)
        # --- fast path: the whole observation is made of daily bars
        vec = np.zeros(n, bool)
        if oc.entry in ("open", "close") and not oc.horizon.startswith("time:"):
            daily = self.res >= 86400
            cs = np.r_[0, np.cumsum(~daily)]
            first = np.arange(n) + (0 if oc.entry == "open" else 1)
            lastd = np.minimum(np.arange(n) + hold, n - 1)
            vec = todo & ((cs[lastd + 1] - cs[np.minimum(first, n)]) == 0) & (first <= lastd)
            if vec.any():
                self._measure_daily(oc, np.flatnonzero(vec), hold, code, end_ret, mfe, mae, epx)
        # --- general path: walk the stored bars of each remaining session
        for i in np.flatnonzero(todo & ~vec):
            self._measure_one(oc, int(i), int(tick[i]), int(th[i]), code, end_ret, mfe, mae, epx, reasons)
        return Outcome(code, end_ret, mfe, mae, epx, span, reasons)

    def _measure_daily(self, oc: OutcomeSpec, idx: np.ndarray, hold: int, code, end_ret, mfe, mae, epx) -> None:
        s = self.ser
        o, h, l, c = s.d_o, s.d_h, s.d_l, s.c[s.day_end - 1]  # the engine reads the last BAR's close for prices
        off0 = 0 if oc.entry == "open" else 1
        w = hold + 1 - off0
        px = o[idx] if oc.entry == "open" else c[idx]
        cols = idx[:, None] + off0 + np.arange(w)[None, :]
        H, L, O, C = h[cols], l[cols], o[cols], c[cols]
        up, tgt, stp = self._levels(oc, px)
        good = np.isfinite(px) & (px > 0)
        hi, lo = np.maximum(H, O), np.minimum(L, O)
        last = C[:, -1]
        sgn = 1.0 if up else -1.0
        out = np.full(idx.size, N_, np.int8)
        if oc.direction == "either":
            t_ = (oc.target or 0.0) / 100
            far = np.maximum(hi.max(axis=1) / px - 1, 1 - lo.min(axis=1) / px)
            hit = (np.abs(last / px - 1) >= t_) if oc.measure == "close" else (far >= t_)
            out[hit] = T
            ii = idx[good]
            code[ii], epx[ii] = out[good], px[good]
            end_ret[ii] = (last[good] / px[good] - 1) * 100
            mfe[ii] = (far * 100)[good]
            mae[ii] = 0.0
            return
        if oc.measure == "close":
            if tgt is not None:
                strict = oc.target == 0
                hit = ((last > tgt) if strict else (last >= tgt)) if up else ((last < tgt) if strict else (last <= tgt))
                out[hit] = T
        else:
            big = w + 1
            it = np.full(idx.size, big)
            is_ = np.full(idx.size, big)
            if tgt is not None:
                mt = (hi >= tgt[:, None]) if up else (lo <= tgt[:, None])
                it = np.where(mt.any(axis=1), mt.argmax(axis=1), big)
            if stp is not None:
                ms = (lo <= stp[:, None]) if up else (hi >= stp[:, None])
                is_ = np.where(ms.any(axis=1), ms.argmax(axis=1), big)
            out[(it < is_)] = T
            out[(is_ < it)] = S
            same = (it == is_) & (it < big)
            if same.any():
                r = np.flatnonzero(same)
                ob = O[r, it[r]]
                gt = (ob >= tgt[r]) if up else (ob <= tgt[r])
                gs = (ob <= stp[r]) if up else (ob >= stp[r])
                out[r] = np.where(gt & ~gs, T, np.where(gs & ~gt, S, A))
        ok = good
        ii = idx[ok]
        code[ii] = out[ok]
        epx[ii] = px[ok]
        end_ret[ii] = (last[ok] / px[ok] - 1) * 100
        mx, mn = H.max(axis=1), L.min(axis=1)
        mfe[ii] = (((mx if up else -mn) - sgn * px) / px * 100)[ok]
        mae[ii] = (((mn if up else -mx) - sgn * px) / px * 100)[ok]

    def _measure_one(self, oc: OutcomeSpec, i: int, t_e: int, t_h: int, code, end_ret, mfe, mae, epx, reasons) -> None:
        s = self.ser
        d = from_day_num(int(s.days[i]))
        px = s.price_at(t_e, d)
        if not (math.isfinite(px) and px > 0):
            reasons["no price at the entry time"] = reasons.get("no price at the entry time", 0) + 1
            return
        i0 = int(np.searchsorted(s.ts, t_e, side="left"))
        i1 = int(np.searchsorted(s.end, t_h, side="right"))
        h, l, o, c = s.h[i0:i1], s.l[i0:i1], s.o[i0:i1], s.c[i0:i1]
        if h.size == 0:
            reasons["no bars between the entry and the end of the horizon"] = reasons.get("no bars between the entry and the end of the horizon", 0) + 1
            return
        up, tgt, stp = self._levels(oc, px)
        if oc.direction == "either":
            t_ = (oc.target or 0.0) / 100
            far = max(float(max(h.max(), o.max())) / px - 1, 1 - float(min(l.min(), o.min())) / px)
            hit = (abs(float(c[-1]) / px - 1) >= t_) if oc.measure == "close" else (far >= t_)
            code[i], epx[i], end_ret[i], mfe[i], mae[i] = (T if hit else N_), px, (c[-1] / px - 1) * 100, far * 100, 0.0
            return
        if oc.measure == "close":
            last = float(c[-1])
            strict = oc.target == 0
            hit = tgt is not None and (((last > tgt) if strict else (last >= tgt)) if up else ((last < tgt) if strict else (last <= tgt)))
            res = T if hit else N_
        else:
            it = is_ = None
            if tgt is not None:
                m = (np.maximum(h, o) >= tgt) if up else (np.minimum(l, o) <= tgt)
                it = int(np.argmax(m)) if m.any() else None
            if stp is not None:
                m = (np.minimum(l, o) <= stp) if up else (np.maximum(h, o) >= stp)
                is_ = int(np.argmax(m)) if m.any() else None
            if it is not None and is_ is not None and it == is_:
                gt = (o[it] >= tgt) if up else (o[it] <= tgt)
                gs = (o[it] <= stp) if up else (o[it] >= stp)
                res = T if gt and not gs else S if gs and not gt else A
            elif it is not None and (is_ is None or it < is_):
                res = T
            elif is_ is not None:
                res = S
            else:
                res = N_
        sgn = 1.0 if up else -1.0
        code[i] = res
        epx[i] = px
        end_ret[i] = (c[-1] / px - 1) * 100
        mfe[i] = ((h.max() if up else -l.min()) - sgn * px) / px * 100
        mae[i] = ((l.min() if up else -h.max()) - sgn * px) / px * 100

    # ----- evaluation of one configuration
    def evaluate(self, preds: list[Pred], oc: OutcomeSpec) -> dict:
        """Masks for one configuration: which sessions were evaluated (the baseline), which
        qualify, and their outcomes."""
        out = self.outcome(oc)
        qual = np.ones(self.n, bool)
        any_false = np.zeros(self.n, bool)
        all_known = np.ones(self.n, bool)
        for p in preds:
            x = self.col(p.feature, oc.entry)
            fin = np.isfinite(x)
            m = p.mask(x) & fin
            any_false |= fin & ~m
            all_known &= fin
            qual &= m
        # Kleene logic, as in the rule engine: a condition with a missing input is still known to be
        # false when another part of it is false; only otherwise is the session "not evaluable"
        known = any_false | all_known
        evaluable = out.valid & known
        qual &= evaluable
        return {"evaluable": evaluable, "qualifying": qual, "outcome": out, "not_evaluable": int((out.valid & ~known).sum())}

    def window(self, start: str | None, end: str | None) -> np.ndarray:
        d = np.array(self.dates)
        m = np.ones(self.n, bool)
        if start:
            m &= d >= str(start)
        if end:
            m &= d <= str(end)
        return m


# ----------------------------------------------------------------------------- cache
_CACHE: dict[tuple, SessionFeatures] = {}


def get_features(store: MarketStore | None, symbol: str, cross: tuple[str, ...] = ()) -> SessionFeatures:
    """One table per (store, symbol, data version): reused across requests until the data changes."""
    store = store or MarketStore()
    sym = symbol.upper()
    cross = tuple(sorted({x.upper() for x in cross} - {sym}))
    try:
        ver = tuple(store.dataset_version(x) for x in (sym,) + cross)
    except Exception:  # noqa: BLE001
        ver = ()
    key = (str(store.root), sym, cross, ver)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[:2] == key[:2] and k != key]:
            del _CACHE[k]  # an older data version of the same symbol
        _CACHE[key] = SessionFeatures(store, sym, cross)
    return _CACHE[key]


def catalog(symbol: str = "SPY", cross: tuple[str, ...] = ()) -> list[dict]:
    """Feature list for the interface."""
    out = [{"name": f.name, "label": f.label, "unit": f.unit, "group": f.group, "expr": f.expr, "needs_intraday": f.kind == "intraday"} for f in FEATURES.values()]
    for x in cross:
        for nm in (f"{x}_ret_1", f"{x}_gap", f"rs5_{x}"):
            f = cross_feature(nm, symbol)
            out.append({"name": f.name, "label": f.label, "unit": f.unit, "group": f.group, "expr": f.expr, "needs_intraday": False})
    return out
