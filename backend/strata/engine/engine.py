"""Chronological, event-driven options backtesting engine.

Per session:
  1. execute pending stock liquidations (from exercise/assignment) at the session open
  2. apply corporate actions (splits) to open positions
  3. for every option snapshot time t (union across symbols, ascending):
       a. mark every open leg to the latest quote at or before t (stale quotes flagged,
          missing quotes keep the last observed mark - never a theoretical price)
       b. run each strategy's on_snapshot(ctx): exits first, then entries
       c. recompute margin; liquidate on a margin deficit
  4. after the close: settle expiring legs (cash or physical), model early assignment,
     record the daily account state
The engine is deterministic: no randomness, ordered iteration, inputs fingerprinted.
"""
from __future__ import annotations

import math
import re
import time as _time
from array import array
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from strata.config import ENGINE_VERSION
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine import execution as X
from strata.engine.margin import MLeg, max_profit_loss, requirement
from strata.engine.market import (DayChain, Events, IvHistory, UnderlyingSeries, atm_iv_from_chain, day_num,
                                  expand_trade_bars, from_day_num)
from strata.engine.models import BacktestConfig, LegSpec
from strata.engine.portfolio import Account, LegState, Position
from strata.engine.selection import Selected, select_contracts


class BacktestCancelled(Exception):
    pass


class DataUnavailable(Exception):
    pass


class Strategy:
    """Base class for strategies. Subclass and implement on_snapshot (and optionally the
    other hooks). The declarative builder strategies are RuleStrategy instances."""

    name: str = "strategy"
    symbol: str = "SPY"
    vol_symbol: str | None = None
    allocation: float = 1.0
    idx: int = 0

    def on_start(self, ctx: "Context") -> None: ...

    def on_day_start(self, ctx: "Context") -> None: ...

    def on_snapshot(self, ctx: "Context") -> None: ...

    def on_day_end(self, ctx: "Context") -> None: ...

    def on_position_closed(self, ctx: "Context", pos: Position) -> None: ...


@dataclass
class Diagnostics:
    missing_sessions: dict = field(default_factory=dict)
    skipped_entries: dict = field(default_factory=dict)
    signals: int = 0
    entries: int = 0
    stale_marks: int = 0
    missing_marks: int = 0
    deferred_exits: int = 0
    partial_fills: int = 0
    margin_calls: int = 0
    assignments: int = 0
    exercises: int = 0
    ambiguous_exits: int = 0
    expired_orders: int = 0
    skipped_signals: list = field(default_factory=list)  # every signal that did not become a trade, with the reason
    no_option_data: dict = field(default_factory=dict)  # symbol -> sessions with a signal clock but no stored option data
    excluded_sessions: dict = field(default_factory=dict)  # reason -> [dates]
    picks: list = field(default_factory=list)  # contract selections of filled entries: what was aimed at, what was chosen, and whether it sat on the edge of the stored strikes
    warnings: list = field(default_factory=list)

    def exclude(self, d, reason: str):
        """A session that could not be evaluated (missing data) - never counted as 'no signal'."""
        import re
        key = re.sub(r"=-?\d+(\.\d+)?", "", reason)[:200]
        lst = self.excluded_sessions.setdefault(key, [])
        iso = d.isoformat()
        if not lst or lst[-1] != iso:
            lst.append(iso)

    def skip(self, reason: str):
        import re
        key = reason  # aggregate by reason: mask per-trade values (strikes, prices, amounts) but keep DTE / minutes
        for pat, sub in ((r"\(\d[\d.]*\)", "(#)"), (r"nearest \d[\d.]*(?!\d*DTE)", "nearest #"), (r"=\d+(\.\d+)?", "=#"), (r"(credit|debit|need|available|budget|unit) -?[\d,]+(\.\d+)?", r"\1 #"),
                         (r"(?<=points )\(#\); nearest #", "(#); nearest #"), (r"all \d+ candidates", "all candidates")):
            key = re.sub(pat, sub, key)
        key = key[:200]
        self.skipped_entries[key] = self.skipped_entries.get(key, 0) + 1

    def skip_signal(self, t: int, signal_t: int, kind: str, reason: str, strategy: str = ""):
        """Record one signal that could not be executed (kept individually so a report can
        list every skipped signal, not just a count)."""
        if len(self.skipped_signals) < 20000:
            self.skipped_signals.append({"t": int(t), "signal_t": int(signal_t), "kind": kind, "reason": reason[:300], "strategy": strategy})

    def to_dict(self):
        d = self.__dict__.copy()
        d["missing_sessions"] = {k: [x.isoformat() for x in v] for k, v in self.missing_sessions.items()}
        return d


class Context:
    """Point-in-time view handed to strategies and the condition engine."""

    def __init__(self, bt: "Backtester"):
        self.bt = bt
        self.t = 0
        self.prev_t = 0
        self.today: date = date(1970, 1, 1)
        self.today_num = 0
        self.minute = 0
        self.close_minute = 960
        self.tick_minutes = 1
        self.day_minutes: set[int] = set()
        self.is_last_tick = False
        self.strategy: Strategy | None = None
        self.seq = None  # today's event-sequence state of the strategy being evaluated (see engine/sequence.py)
        self._feat: dict = {}
        self.events: Events = bt.events

    # --- identity
    @property
    def symbol(self) -> str:
        return self.strategy.symbol

    @property
    def account(self) -> Account:
        return self.bt.account

    # --- prices / features
    def series(self, sym: str) -> UnderlyingSeries | None:
        return self.bt.series.get(sym)

    @property
    def session_res(self) -> int:
        """Seconds per bar of the finest underlying data stored for today's session."""
        s = self.bt.series.get(self.symbol)
        return s.resolution(self.today) if s is not None else 0

    def price(self, sym: str | None = None, prev: bool = False) -> float:
        sym = sym or self.symbol
        s = self.bt.series.get(sym)
        t = self.prev_t if prev else self.t
        px = s.price_at(t, self.today) if s is not None and not s.empty() else float("nan")
        if not math.isfinite(px):
            ch = self.bt.chains.get(sym)
            if ch is not None and ch.kind == "quotes":
                si = ch.snapshot_at(t)
                if si >= 0:
                    u = ch.under[ch.rows(si)]
                    u = u[np.isfinite(u)]
                    if u.size:
                        return float(u[0])
        return px

    def price_minutes_ago(self, sym: str, n: int) -> float:
        s = self.bt.series[sym]
        return s.price_at(self.t - 60 * n, self.today)

    def opening_range(self, sym: str, minutes: int):
        return self.bt.series[sym].session_range_until(self.today, self.t, minutes)

    def indicator(self, sym, name, params, tf, field=None, prev=False) -> float:
        s = self.bt.series.get(sym)
        if s is None:
            return float("nan")
        t = self.prev_t if prev else self.t
        if prev and tf == "1d":
            return s.indicator(name, params, tf, field, t, self.today, offset=1)
        return s.indicator(name, params, tf, field, t, self.today)

    def vol_symbol(self, sym: str) -> str | None:
        st = self.strategy
        if st is not None and st.vol_symbol:
            return st.vol_symbol
        return self.bt.metas[sym].vol_index if sym in self.bt.metas else None

    def feature(self, name: str, sym: str | None = None, prev: bool = False) -> float:
        sym = sym or self.symbol
        t = self.prev_t if prev else self.t
        key = (name, sym, t)
        if key in self._feat:
            return self._feat[key]
        v = self._feature(name, sym, t)
        self._feat[key] = v
        return v

    def _feature(self, name: str, sym: str, t: int) -> float:
        s = self.bt.series.get(sym)
        nan = float("nan")
        if name == "price":
            return s.price_at(t, self.today) if s else nan
        if name in ("prev_close", "prev_high", "prev_low"):
            if not s:
                return nan
            i = s.prev_day_index(self.today)
            if i < 0:
                return nan
            return float({"prev_close": s.d_c, "prev_high": s.d_h, "prev_low": s.d_l}[name][i])
        if name == "open":
            return s.session_open(self.today) if s else nan
        if name == "gap_pct":
            o, pc = self._feature("open", sym, t), self._feature("prev_close", sym, t)
            return (o / pc - 1) * 100 if pc and math.isfinite(pc) else nan
        if name == "day_move_pct":
            o = self._feature("open", sym, t)
            return (s.price_at(t, self.today) / o - 1) * 100 if o else nan
        if name in ("day_high", "day_low"):
            hi, lo, _ = s.session_range_until(self.today, t)
            if not math.isfinite(hi):
                o = s.session_open(self.today)
                return o
            return hi if name == "day_high" else lo
        if name in ("vix", "vix_prev_close"):
            vs = self.vol_symbol(sym)
            ser = self.bt.series.get(vs) if vs else None
            if ser is None or ser.empty():
                return nan
            if name == "vix_prev_close":
                i = ser.prev_day_index(self.today)
                return float(ser.d_c[i]) if i >= 0 else nan
            if ser.day_index(self.today) >= 0:
                return ser.price_at(t, self.today)
            i = ser.prev_day_index(self.today)
            return float(ser.d_c[i]) if i >= 0 else nan
        if name == "iv":
            ch = self.bt.chains.get(sym)
            if ch is None:
                return nan
            si = ch.snapshot_at(t)
            if si < 0:
                return nan
            ck = ("_iv", sym, si, self.today_num)
            if ck not in self.bt.iv_cache:
                spot = self._feature("price", sym, t)
                v = atm_iv_from_chain(ch, si, spot, 30.0)
                self.bt.iv_cache[ck] = (v * 100) if v is not None else nan
            return self.bt.iv_cache[ck]
        if name in ("iv_rank", "iv_pct"):
            return self.iv_rank(sym, 252, pct=(name == "iv_pct"))
        raise ValueError(f"unknown feature {name}")

    def iv_rank(self, sym: str, lookback: int = 252, pct: bool = False) -> float:
        cur = self.feature("iv", sym)
        hist = self.bt.iv_hist.get(sym)
        if hist is None or not math.isfinite(cur):
            return float("nan")
        w = hist.window(self.today, lookback) * 100
        if w.size < min(20, lookback):
            return float("nan")
        if pct:
            return float((w < cur).mean() * 100)
        lo, hi = w.min(), w.max()
        return float(np.clip((cur - lo) / (hi - lo) * 100, 0, 100)) if hi > lo else 50.0

    def regime(self, sym: str) -> tuple[str, str]:
        st = self.bt.regime_settings.get(self.strategy.idx) if self.strategy else None
        fast, slow = (st.trend_fast, st.trend_slow) if st else (50, 200)
        lo, hi = (st.vix_low, st.vix_high) if st else (15, 25)
        s = self.bt.series[sym]
        pc = self.feature("prev_close", sym)
        f, sl = s.daily_sma(self.today, fast), s.daily_sma(self.today, slow)
        if not (math.isfinite(f) and math.isfinite(sl)):
            trend = "unknown"
        elif pc > sl and f > sl:
            trend = "bull"
        elif pc < sl and f < sl:
            trend = "bear"
        else:
            trend = "neutral"
        v = self.feature("vix", sym)
        vol = "unknown" if not math.isfinite(v) else ("low" if v < lo else "high" if v > hi else "normal")
        return trend, vol

    # --- chain & orders
    def chain(self, sym: str | None = None) -> tuple[DayChain | None, int]:
        ch = self.bt.chains.get(sym or self.symbol)
        if ch is None:
            return None, -1
        return ch, ch.snapshot_at(self.t)

    def positions(self, all_strategies: bool = False) -> list[Position]:
        if all_strategies:
            return list(self.bt.account.positions)
        return [p for p in self.bt.account.positions if p.strategy_idx == self.strategy.idx]

    def select(self, legs: list[LegSpec], sym: str | None = None):
        ch, si = self.chain(sym)
        if ch is None or si < 0:
            return None, "no option chain for this session", {}
        if self.t - int(ch.snap_ts[si]) > self.bt.max_age:
            return None, ("no option trade has printed recently" if self.bt.trades else "latest chain snapshot is stale"), {}
        sel, why, details = select_contracts(ch, si, legs, self.price(sym or self.symbol), self.today, self.bt.max_age)
        if sel is not None and len(self.bt.diag.picks) < 20000:
            for lid, d in details.items():
                if "pick" in d:
                    self.bt.diag.picks.append({"date": self.today.isoformat(), "t": int(self.t), "leg": lid, **d["pick"]})
        return sel, why, details

    def open(self, selected: list[Selected], units: int, explain: list | None = None, tag: str = "", meta: dict | None = None):
        return self.bt.open_position(self.strategy, selected, units, self, explain or [], tag, meta or {})

    def close(self, pos: Position, reason: str, legs: list[int] | None = None, explain: list | None = None) -> bool:
        return self.bt.close_position(pos, reason, self, legs, explain or [])

    def position_pnl_pct(self, pos: Position, use_mid: bool = False) -> float:
        return pos.gross_pnl(use_mid) / pos.basis * 100

    def log(self, kind: str, msg: str, pos: Position | None = None, data: dict | None = None):
        self.bt.log(self.t, kind, msg, pos, data)


class _DayChains(dict):
    """The day's option chains by symbol.

    A chain built from per-contract trade bars costs tens of thousands of rows to expand, and a strategy that fires on a few
    sessions a year needs it on those sessions only: such a chain is registered with `defer` and built the first time
    something asks for it (a rule that fired, an open position being marked). Its snapshot times are the engine's own
    ticks, so the day's clock, which is taken from the chains already built, does not depend on it."""

    def __init__(self):
        super().__init__()
        self._pending: dict = {}

    def defer(self, sym: str, loader) -> None:
        self._pending[sym] = loader

    def _load(self, sym) -> None:
        f = self._pending.pop(sym, None)
        if f is not None:
            super().__setitem__(sym, f())

    def get(self, sym, default=None):
        self._load(sym)
        return super().get(sym, default)

    def __getitem__(self, sym):
        self._load(sym)
        return super().__getitem__(sym)

    def __contains__(self, sym) -> bool:
        return sym in self._pending or super().__contains__(sym)


class Backtester:
    def __init__(self, cfg: BacktestConfig, store: MarketStore | None = None, strategies: list[Strategy] | None = None,
                 progress=None, should_cancel=None, record_paths: bool = True, signals_only: bool = False, extra_symbols: set[str] | None = None,
                 sessions: set | None = None, resume: dict | None = None, keep_open: bool = False):
        from strata.engine.rules import RuleStrategy

        self.cfg = cfg
        # continuing a paper record instead of replaying it: `resume` is the state another run left (cash, open positions,
        # what each strategy remembers) and this run starts from it on its first session. With `keep_open`, positions still
        # open at the end are NOT closed at the last price: they are handed on in the state, as an account would hold them.
        self._resume, self.keep_open = resume, bool(keep_open)
        # the sessions on which an entry rule of this run fires, when a scan of the SAME rules has already found them (an
        # experiment runs many configurations that share their entry rule). Any other session is passed over while nothing is
        # open: no rule can fire on it, so nothing about the trades changes. Left None, every session is evaluated.
        self.only_sessions = set(sessions) if sessions is not None else None
        self.extra_symbols = {x.upper() for x in (extra_symbols or ())}
        self.store = store or MarketStore()
        self.ex = cfg.execution
        self.signals_only = signals_only  # evaluate entry rules on the underlying only; no option data is read
        self.trades = cfg.pricing.mode == "trades" and not signals_only
        # how old the latest observation of a contract may be before it is refused
        self.max_age = cfg.execution.max_trade_age_seconds if self.trades else cfg.execution.max_quote_age_seconds
        self.opt_ds: dict = {}
        self.account = Account(cfg.capital.initial_capital, cfg.capital.margin_model)
        self.strategies: list[Strategy] = strategies or [RuleStrategy(s, i) for i, s in enumerate(cfg.strategies)]
        for i, s in enumerate(self.strategies):
            s.idx = i
        self.regime_settings = {i: s.regime for i, s in enumerate(cfg.strategies)} if not strategies else {}
        self.progress = progress
        self.should_cancel = should_cancel
        self.record_paths = record_paths
        self.diag = Diagnostics()
        self.events_log: list[dict] = []
        self.chains: dict[str, DayChain | None] = _DayChains()
        self.series: dict[str, UnderlyingSeries] = {}
        self.iv_hist: dict[str, IvHistory] = {}
        self.iv_cache: dict = {}
        self.metas = {}
        self.next_pos_id = 1
        self.daily: list[dict] = []
        self.path_cols = {k: array("d") for k in ["pos", "t", "spot", "pnl_mark", "pnl_close", "delta", "gamma", "theta", "vega"]}
        self.leg_cols = {k: array("d") for k in ["pos", "t", "leg", "bid", "ask", "mid", "iv", "delta"]}
        self.intraday_equity = {"t": array("d"), "equity": array("d")}
        self.events = Events(self.store.read_events())
        self.splits: dict[str, dict[date, float]] = {}
        self.dividends: dict[str, dict[date, float]] = {}

    # ------------------------------------------------------------------ setup
    def _load(self):
        cfg = self.cfg
        opt_syms = sorted({s.symbol for s in self.strategies})
        syms = set(opt_syms)
        for s in self.strategies:
            m = self.store.symbol_meta(s.symbol)
            self.metas[s.symbol] = m
            vs = s.vol_symbol or m.vol_index
            if vs:
                syms.add(vs)
        for s in self.strategies:
            c = getattr(s, "cfg", None)
            if c is not None:
                syms |= c.symbols()  # every other symbol a rule reads, e.g. QQQ.prev_high
        syms |= self.extra_symbols  # symbols only an observer reads (a debugger watch), never the rules
        bench = cfg.benchmark or opt_syms[0]
        syms.add(bench)
        self.bench = bench
        warm = cfg.start - timedelta(days=420)
        self.dataset_versions = {}
        self.synthetic = False
        for sym in sorted(syms):
            meta = self.store.symbol_meta(sym)
            self.metas[sym] = meta
            if self.store.is_synthetic(sym):
                self.synthetic = True
                if not cfg.allow_synthetic:
                    raise DataUnavailable(f"{sym} is a SYNTHETIC dataset and allow_synthetic is false")
            df = self.store.read_best_bars(sym, warm, cfg.end)
            if df.is_empty():
                if sym in opt_syms:
                    raise DataUnavailable(f"No historical price data is stored for {sym}. Download it in Data (free daily history is available from Yahoo) - Strata never substitutes simulated prices.")
                self.diag.warnings.append(f"no price data for auxiliary symbol {sym}; rules that need it are reported as not evaluable")
                continue
            ser = UnderlyingSeries(df, official=self.store.official_closes(sym, warm, cfg.end))
            ser.split_adjusted = self.store.price_adjustment(sym) == "split"
            ser.providers = {f"{r['provider']}:{_res_label(int(r['res']))}": int(r["n"]) for r in df.filter(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date() >= cfg.start)
                             .with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).unique(subset=["_d"]).group_by("provider", "res").agg(pl.len().alias("n")).iter_rows(named=True)}
            if ser.split_adjusted:  # vendor prices are split-adjusted to today: express them in start-date share terms
                ser.rescale(self.store.split_factor(sym, cfg.start))
            self.series[sym] = ser
            self.dataset_versions[sym] = self.store.dataset_version(sym)
        self.modeled = cfg.pricing.mode == "modeled" and not self.signals_only
        self.datasets_used: dict = {}
        for sym in opt_syms:
            if self.signals_only:
                self.opt_ds[sym] = None
                continue
            # one explicitly chosen option dataset per symbol and run: quotes and trade prints are never blended
            obs = "trade" if self.trades else "quote"
            ds = None if self.modeled else self.store.option_dataset(sym, obs, cfg.data.options_provider)
            self.opt_ds[sym] = ds
            if not self.modeled:
                what = "option trade prints" if self.trades else "option quotes"
                if ds is None:
                    other = self.store.option_dataset(sym, "quote" if self.trades else "trade")
                    hint = (f" Only {'bid/ask quotes' if self.trades else 'trade prints'} are stored ({other.id}); they are a different kind of observation and are not substituted." if other else "")
                    prov = f" from provider {cfg.data.options_provider!r}" if cfg.data.options_provider else ""
                    raise DataUnavailable(f"No historical {what} are stored for {sym}{prov}.{hint} Strata does not fabricate option prices: acquire option history, "
                                          "run a Market Study on the underlying, or explicitly choose the Modeled mode.")
                desc = self.store.descriptor(ds)
                bars_fmt = desc.get("format") == "bars"
                days = sorted(self.store.option_bar_index(sym, dataset=ds)["date"].unique().to_list()) if bars_fmt else self.store.option_days(sym, obs, dataset=ds)
                # a continuing record is run one session at a time: a session for which no option observation is stored is not a
                # reason to stop. It is evaluated, its signals are recorded as not priced, and open positions are carried
                if not any(cfg.start <= d <= cfg.end for d in days) and not self.keep_open:
                    have = f"{ds.id} covers {days[0]} to {days[-1]}" if days else f"{ds.id} is empty"
                    raise DataUnavailable(f"No historical {what} for {sym} between {cfg.start} and {cfg.end} ({have}). Strata does not fabricate option prices: "
                                          "acquire option history for this period, run a Market Study on the underlying, or explicitly choose the Modeled mode.")
                self.datasets_used[sym] = {"dataset": ds.id, "provider": ds.provider, "observation": "trade prints" if self.trades else "bid/ask quotes", "resolution": ds.res,
                                           "format": desc.get("format") or "snapshots", "license": desc.get("license", ""), "limitations": desc.get("limitations") or [],
                                           "partial_sessions": sum(1 for d, v in self.store.session_states(ds).items() if v.get("status") != "complete" and cfg.start.isoformat() <= d <= cfg.end.isoformat())}
                self._bars_fmt = getattr(self, "_bars_fmt", {})
                self._bars_fmt[sym] = bars_fmt
                if bars_fmt:
                    defs = self.store.read_contracts(ds)
                    self._mults = getattr(self, "_mults", {})
                    if not defs.is_empty():
                        from strata.engine.market import contract_key
                        # a multiplier that is not positive is not a multiplier (one provider lists 0 on old contracts): such a
                        # definition says nothing, and the symbol's own contract size applies as for any contract without one
                        bad_m = defs.filter(pl.col("multiplier").is_not_null() & (pl.col("multiplier") <= 0)).height
                        if bad_m:
                            self.datasets_used[sym]["definitions_without_a_usable_multiplier"] = bad_m
                        dd = defs.filter(pl.col("multiplier").is_not_null() & (pl.col("multiplier") > 0))
                        keys = contract_key(dd["expiration"].cast(pl.Int32).to_numpy(), (dd["right"] == "C").to_numpy(), dd["strike"].to_numpy())
                        self._mults[sym] = dict(zip(keys.tolist(), dd["multiplier"].to_list()))
            self.iv_hist[sym] = IvHistory.load(self.store, sym, None if self.trades else ds)
            ca = self.store.read_corporate_actions(sym)
            for r in ca.iter_rows(named=True):
                if r["action"] == "split" and r["ratio"]:
                    self.splits.setdefault(sym, {})[r["ex_date"]] = float(r["ratio"])
                elif r["action"] == "dividend" and r["amount"]:
                    self.dividends.setdefault(sym, {})[r["ex_date"]] = float(r["amount"])
        self.opt_syms = opt_syms
        for sym in list(self.series):
            for r in self.store.read_corporate_actions(sym).iter_rows(named=True):
                if r["action"] == "split" and r["ratio"]:
                    self.splits.setdefault(sym, {})[r["ex_date"]] = float(r["ratio"])
        first_avail = min((s.days[0] for s in self.series.values() if not s.empty()), default=None)
        if first_avail is not None and from_day_num(first_avail) > cfg.start - timedelta(days=280):
            self.diag.warnings.append("less than ~200 sessions of price history before the start date: long-period daily indicators and regime filters are n/a early in the run")

    # ------------------------------------------------------------------ logging
    def log(self, t: int, kind: str, msg: str, pos: Position | None = None, data: dict | None = None):
        self.events_log.append({"t": t, "kind": kind, "msg": msg, "pos": pos.id if pos else None, **({"data": data} if data else {})})

    # ------------------------------------------------------------------ marks
    def _mark_leg(self, l: LegState, ch: DayChain | None, si: int, t: int, pos: Position):
        if l.kind == "stock":
            s = self.series[pos.symbol].price_at(t, self.ctx.today)
            l.bid = l.ask = l.mid = s
            return
        if t >= l.exp_close:
            # at/after its expiration close a contract no longer trades: its value is its
            # settlement (intrinsic) value; any quote printed now is ignored
            s = self.series[pos.symbol].price_at(t, self.ctx.today)
            intr = max(s - l.strike, 0.0) if l.is_call else max(l.strike - s, 0.0)
            l.bid = l.ask = l.mid = intr
            l.stale = False
            return
        if ch is None or si < 0:
            l.stale = True
            return
        i = ch.find(si, l.key)
        if i < 0:
            l.stale = True
            self.diag.missing_marks += 1
            pos.flags.add("missing_quotes")
            return
        b, a = ch.bid[i], ch.ask[i]
        if not (math.isfinite(b) and math.isfinite(a)) or a < b:
            l.stale = True
            return
        l.bid, l.ask, l.mid = float(b), float(a), float(ch.mid[i])
        l.iv = float(ch.iv[i])
        l.greeks = (float(ch.delta[i]), float(ch.gamma[i]), float(ch.theta[i]), float(ch.vega[i]))
        l.quote_ts = int(ch.quote_ts[i])
        l.bid_size, l.ask_size, l.volume = int(ch.bid_size[i]), int(ch.ask_size[i]), int(ch.volume[i])
        l.trade_based = bool(ch.trade_based[i])
        l.stale = (t - l.quote_ts) > self.max_age or l.quote_ts > t
        if l.stale:
            if l.quote_ts == getattr(l, "_stale_seen", None):
                return  # same old quote seen again between option snapshots: count once
            l._stale_seen = l.quote_ts
            self.diag.stale_marks += 1
            pos.flags.add("stale_quotes")

    def _update_position_values(self, p: Position):
        mv = cv = 0.0
        for l in p.legs:
            if not l.is_open or not math.isfinite(l.mid):
                continue
            if l.kind == "stock":
                mv += l.qty * l.mid
                cv += l.qty * X.stock_fill_price(l.mid, -np.sign(l.qty), self.ex)
                continue
            mv += l.qty * l.multiplier * l.mid
            cv += l.qty * l.multiplier * X.fill_price(l.bid, l.ask, -int(np.sign(l.qty)), self.ex)
        if p.shares:
            s = self.series[p.symbol].price_at(self.ctx.t, self.ctx.today)
            mv += p.shares * s
            cv += p.shares * X.stock_fill_price(s, -np.sign(p.shares), self.ex)
        p.mark_value, p.close_value = mv, cv

    def mark_all(self, t: int):
        for p in self.account.positions:
            ch = self.chains.get(p.symbol)
            si = ch.snapshot_at(t) if ch is not None else -1
            for l in p.legs:
                if l.is_open:
                    self._mark_leg(l, ch, si, t, p)
            p.last_spot = self.series[p.symbol].price_at(t, self.ctx.today)
            self._update_position_values(p)
            pnl = p.pnl_mark()
            p.mfe = max(p.mfe, pnl)
            p.mae = min(p.mae, pnl)
            if self.record_paths:
                d, g, th, v = p.greeks()
                for k, val in zip(self.path_cols, (p.id, t, p.last_spot, pnl, p.pnl_close(), d, g, th, v)):
                    self.path_cols[k].append(float(val))
                for li, l in enumerate(p.legs):
                    if l.is_open:
                        for k, val in zip(self.leg_cols, (p.id, t, li, l.bid, l.ask, l.mid, l.iv, l.greeks[0])):
                            self.leg_cols[k].append(float(val) if val is not None else math.nan)

    # ------------------------------------------------------------------ orders
    def _situational(self, px: float, side: int, t: int, exp_close: int | None = None) -> float:
        """Auction-window and expiration-day surcharges on an option fill (side +1 buy, -1 sell).
        Both are explicit assumptions from the execution settings; with the defaults they do nothing."""
        ex = self.ex
        if not (ex.auction_extra_pct or ex.expiry_day_extra_pct):
            return px
        c = self.ctx
        minute = 570 + (t - self._t_open) // 60 if getattr(self, "_t_open", None) is not None else c.minute
        extra = 0.0
        if ex.auction_extra_pct and ((ex.no_entry_first_minutes and minute < 570 + ex.no_entry_first_minutes) or (ex.no_entry_last_minutes and minute >= c.close_minute - ex.no_entry_last_minutes)):
            extra += ex.auction_extra_pct
        if ex.expiry_day_extra_pct and exp_close is not None and 0 <= exp_close - t <= ex.expiry_day_minutes * 60:
            extra += ex.expiry_day_extra_pct
        return max(0.0, px * (1 + side * extra / 100.0)) if extra else px

    def open_position(self, strat: Strategy, sel: list[Selected], units: int, ctx: Context, explain: list, tag: str, meta: dict):
        ex = self.ex
        ch = self.chains.get(strat.symbol)
        mult = self.metas[strat.symbol].multiplier
        if units < 1:
            return None, "size below one unit"
        if len(self.account.positions) >= self.cfg.capital.max_concurrent_positions:
            return None, "portfolio max concurrent positions reached"
        spot = ctx.price(strat.symbol)
        # liquidity
        max_units = units
        for s in sel:
            if s.row < 0:
                continue
            buy = s.leg.side == "buy"
            cap = X.liquidity_cap(int(ch.ask_size[s.row] if buy else ch.bid_size[s.row]), int(ch.volume[s.row]), ex)
            if math.isfinite(cap):
                max_units = min(max_units, int(cap // s.leg.quantity))
        partial = False
        if max_units < units:
            if not ex.allow_partial_fills or max_units < 1:
                return None, "insufficient displayed liquidity"
            units, partial = max_units, True
        legs: list[LegState] = []
        cash = slip = comm = 0.0
        trade_based = False
        print_ages: list[int] = []
        trade_freq: list[float] = []
        for s in sel:
            sign = s.leg.sign
            if s.row < 0:  # stock
                shares = sign * s.leg.quantity * units
                px = X.stock_fill_price(spot, sign, ex)
                cash -= shares * px
                slip += abs(px - spot) * abs(shares)
                comm += X.stock_commission(shares, ex)
                ls = LegState(s.leg.id, "stock", False, 0.0, -1, -1, sign * s.leg.quantity, 1.0, s.leg.group, qty=shares,
                              entry_price=px, entry_bid=spot, entry_ask=spot, entry_mid=spot, entry_greeks=(1.0, 0.0, 0.0, 0.0))
                ls.bid = ls.ask = ls.mid = spot
                legs.append(ls)
                continue
            i = s.row
            q = sign * s.leg.quantity * units
            lmult = float(ch.multipliers.get(int(ch.key[i]), mult))  # the contract's own multiplier when its definition is stored
            px = X.fill_price(float(ch.bid[i]), float(ch.ask[i]), sign, ex)
            if ex.multi_leg_extra_pct and sum(1 for z in sel if z.row >= 0) >= 2:
                px = max(0.0, px * (1 + sign * ex.multi_leg_extra_pct / 100.0))
            px = self._situational(px, sign, ctx.t, int(cal.session_bounds(from_day_num(int(ch.exp[i])))[1].timestamp()) if ex.expiry_day_extra_pct else None)
            tb = bool(ch.trade_based[i])
            if tb:  # trade print, no quote: the print plus an explicit slippage assumption
                px = max(0.0, px * (1 + sign * ex.trade_bar_slippage_pct / 100.0))
                trade_based = True
                print_ages.append(int(ctx.t - ch.quote_ts[i]))
                st_ = ch.contract_stats.get(int(ch.key[i]))
                if st_:
                    trade_freq.append(st_["bars"] / max(1, (ctx.close_minute - 570)))
            cash -= q * lmult * px
            slip += abs(px - float(ch.mid[i])) * abs(q) * lmult
            comm += X.option_commission(q, ex)
            g = (float(ch.delta[i]), float(ch.gamma[i]), float(ch.theta[i]), float(ch.vega[i]))
            ls = LegState(s.leg.id, "option", bool(ch.is_call[i]), float(ch.strike[i]), int(ch.exp[i]), int(ch.key[i]),
                          sign * s.leg.quantity, lmult, s.leg.group, qty=q, entry_price=px, entry_bid=float(ch.bid[i]),
                          entry_ask=float(ch.ask[i]), entry_mid=float(ch.mid[i]), entry_iv=float(ch.iv[i]), entry_greeks=g)
            ls.bid, ls.ask, ls.mid, ls.iv, ls.greeks, ls.quote_ts = ls.entry_bid, ls.entry_ask, ls.entry_mid, ls.entry_iv, g, int(ch.quote_ts[i])
            ls.trade_based = ls.entry_trade_based = tb
            ls.entry_volume, ls.entry_size = int(ch.volume[i]), int(ch.ask_size[i] if sign > 0 else ch.bid_size[i])
            ls.exp_close = int(cal.session_bounds(from_day_num(ls.exp))[1].timestamp())
            legs.append(ls)
        comm += ex.commission_per_order
        per_unit_credit = cash / (units * mult)
        sc = getattr(strat, "cfg", None)
        if sc is not None:
            if sc.entry.min_credit is not None and per_unit_credit < sc.entry.min_credit:
                return None, f"credit {per_unit_credit:.2f} below minimum {sc.entry.min_credit:.2f}"
            if sc.entry.max_debit is not None and -per_unit_credit > sc.entry.max_debit:
                return None, f"debit {-per_unit_credit:.2f} above maximum {sc.entry.max_debit:.2f}"
        mlegs = [MLeg(l.is_call, l.strike, l.qty, l.multiplier, l.mid) for l in legs if l.kind == "option"]
        shares = sum(l.qty for l in legs if l.kind == "stock")
        R = requirement(mlegs, shares, spot, self.cfg.capital.margin_model)
        if not math.isfinite(R):
            return None, "uncovered short calls are not permitted under the cash margin model"
        bpr = max(0.0, R - cash)
        equity = self.account.equity()
        if self.account.available_bp() + cash - comm - R < 0:
            return None, f"insufficient buying power (need {bpr:,.0f}, available {self.account.available_bp():,.0f})"
        cap_limit = self.cfg.capital.max_bp_usage_pct / 100 * equity
        if self.account.bp_used() + bpr > cap_limit + 1e-6:
            return None, f"max buying-power usage {self.cfg.capital.max_bp_usage_pct:g}% would be exceeded"
        pos = Position(self.next_pos_id, strat.idx, strat.name, strat.symbol, units, legs, ctx.t, ctx.today_num, spot,
                       entry_cash=cash, commissions=comm, slippage=slip, max_risk=bpr, requirement=R, tag=tag)
        self.next_pos_id += 1
        pos.entry_explain = explain
        pos.meta = meta
        pos.entry_vix = ctx.feature("vix", strat.symbol)
        pos.entry_iv = ctx.feature("iv", strat.symbol)
        tr, vr = ctx.regime(strat.symbol)
        pos.entry_regime = f"{tr}/{vr}"
        pos.last_spot = spot
        pos.signal_ts = int(meta.get("signal_ts") or ctx.t)
        if trade_based:
            pos.flags.add("trade_based_fill")
            self._saw_trade_based = True
            pos.meta["entry_print_age_seconds"] = max(print_ages) if print_ages else None
            if trade_freq:
                pos.meta["contract_minutes_traded_pct"] = round(min(trade_freq) * 100, 2)
        if self.modeled:
            pos.flags.add("modeled_prices")
        if partial:
            pos.flags.add("partial_fill")
            self.diag.partial_fills += 1
        self.account.cash += cash - comm
        self.account.commissions += comm
        self.account.slippage += slip
        self.account.positions.append(pos)
        self._update_position_values(pos)
        self.diag.entries += 1
        self.log(ctx.t, "entry", f"opened {units}x {strat.name}: " + ", ".join(f"{'+' if l.qty > 0 else ''}{l.qty} {l.label()} @ {l.entry_price:.2f}" for l in legs) + f" | net {'credit' if cash > 0 else 'debit'} {abs(cash):,.2f}, BPR {bpr:,.0f}", pos)
        return pos, ""

    def _close_leg(self, pos: Position, l: LegState, n: int, t: int, reason: str) -> float:
        """Close n contracts (n>0) of leg at the current quote. Returns cash flow."""
        if l.kind == "stock":
            q = int(np.sign(l.qty)) * n
            px = X.stock_fill_price(l.mid, -int(np.sign(q)), self.ex)
            flow = q * px
            comm = X.stock_commission(q, self.ex)
            slip = abs(px - l.mid) * abs(q)
        else:
            q = int(np.sign(l.qty)) * n
            px = X.fill_price(l.bid, l.ask, -int(np.sign(q)), self.ex)
            if self.ex.multi_leg_extra_pct and sum(1 for z in pos.legs if z.kind == "option") >= 2:
                px = max(0.0, px * (1 - int(np.sign(q)) * self.ex.multi_leg_extra_pct / 100.0))
            px = self._situational(px, -int(np.sign(q)), t, l.exp_close)
            if getattr(l, "trade_based", False):
                px = max(0.0, px * (1 - int(np.sign(q)) * self.ex.trade_bar_slippage_pct / 100.0))
            flow = q * l.multiplier * px
            comm = X.option_commission(q, self.ex)
            slip = abs(px - l.mid) * abs(q) * l.multiplier
        if l.leg_id == "_hedge":
            # the closing trade of the hedge is a hedge trade like the others: without it the position's hedge record showed the
            # opening proceeds as its whole result and a share count that never returned to zero
            h = pos.meta.setdefault("hedge", {"trades": 0, "shares_traded": 0, "cash": 0.0})
            h["trades"] += 1
            h["shares_traded"] += n
            h["cash"] += flow
            h["shares"] = int(l.qty - q)
        prev_closed = l.closed_qty
        l.exit_price = px if prev_closed == 0 else (l.exit_price * prev_closed + px * n) / (prev_closed + n)
        l.exit_bid, l.exit_ask, l.exit_mid, l.exit_greeks = l.bid, l.ask, l.mid, l.greeks
        l.exit_ts, l.exit_reason = t, reason
        l.closed_qty += n
        l.exit_cash += flow
        l.qty -= q
        pos.exit_cash += flow
        pos.commissions += comm
        pos.slippage += slip
        self.account.cash += flow - comm
        self.account.commissions += comm
        self.account.slippage += slip
        return flow

    def close_position(self, pos: Position, reason: str, ctx: Context, leg_idx: list[int] | None = None, explain: list | None = None, ignore_liquidity: bool = False, units: int | None = None) -> bool:
        """Close the whole position (or a subset of legs) at the current snapshot. Returns
        True when everything requested was filled; otherwise the remainder is retried at
        the next snapshot. With `units`, exactly that many units are closed and the rest of
        the position stays open (a staged exit); it either fills in full or not at all."""
        t = ctx.t
        if units is not None:
            return self._scale_out(pos, reason, ctx, int(units), explain)
        # legs at/after their expiration close cannot be traded; they settle after the close
        targets = [i for i, l in enumerate(pos.legs) if l.is_open and (leg_idx is None or i in leg_idx) and t < l.exp_close]
        if not targets:
            return True
        blocked = [pos.legs[i] for i in targets if pos.legs[i].stale or not math.isfinite(pos.legs[i].mid)]
        if blocked:
            first = pos.meta.get("exit_signal_ts") is None
            if leg_idx is None:
                pos.pending_exit = reason
                pos.meta.setdefault("exit_signal_ts", t)
            if first or leg_idx is not None:
                pos.flags.add("deferred_exit")
                self.diag.deferred_exits += 1
                self.log(t, "exit_deferred", f"{reason}: no fresh quote for {', '.join(l.label() for l in blocked)}; filling at the next observable quote", pos)
            return False
        # liquidity (by units, ratio-preserving for full closes)
        caps = []
        for i in targets:
            l = pos.legs[i]
            if l.kind == "stock":
                continue
            size = l.bid_size if l.qty > 0 else l.ask_size
            cap = math.inf if ignore_liquidity else X.liquidity_cap(size, l.volume, self.ex)
            caps.append(cap // abs(l.ratio) if math.isfinite(cap) else math.inf)
        open_units = min((abs(pos.legs[i].qty) // max(abs(pos.legs[i].ratio), 1) for i in targets if pos.legs[i].leg_id != "_hedge"), default=0)
        if open_units == 0 and any(pos.legs[i].leg_id == "_hedge" for i in targets):
            open_units = 1  # only hedge shares are left: close them
        n_units = int(min([open_units] + caps)) if caps else open_units
        if n_units <= 0 and open_units > 0:
            pos.pending_exit = reason if leg_idx is None else pos.pending_exit
            self.diag.deferred_exits += 1
            pos.flags.add("deferred_exit")
            self.log(t, "exit_deferred", f"{reason}: no displayed size to close; retrying", pos)
            return False
        for i in targets:
            l = pos.legs[i]
            n = abs(l.qty) if (n_units >= open_units or l.leg_id == "_hedge") else abs(l.ratio) * n_units
            if n > 0:
                self._close_leg(pos, l, int(n), t, reason)
        if explain:
            pos.exit_explain = explain
        full = n_units >= open_units
        if not full:
            pos.flags.add("partial_exit")
            self.diag.partial_fills += 1
            pos.pending_exit = reason
            pos.units = max(1, open_units - n_units)
        self._update_position_values(pos)
        if leg_idx is not None and full:
            self.log(t, "leg_exit", f"{reason}: closed {', '.join(pos.legs[i].label() for i in targets)}", pos)
        if not pos.is_open:
            self._finalize(pos, t, reason)
        elif full and leg_idx is None:
            pos.pending_exit = None
        return full

    def _scale_out(self, pos: Position, reason: str, ctx: Context, units: int, explain: list | None) -> bool:
        t = ctx.t
        legs = [l for l in pos.legs if l.is_open and l.leg_id != "_hedge" and t < l.exp_close]
        if units < 1 or units >= pos.units or not legs:
            return False
        if any(l.stale or not math.isfinite(l.mid) for l in legs):
            return False
        for l in legs:
            if l.kind == "stock":
                continue
            cap = X.liquidity_cap(l.bid_size if l.qty > 0 else l.ask_size, l.volume, self.ex)
            if math.isfinite(cap) and cap // max(abs(l.ratio), 1) < units:
                return False  # not enough displayed size for the whole stage: nothing is filled, the stage is tried again
        for l in legs:
            self._close_leg(pos, l, int(abs(l.ratio) * units), t, reason)
        pos.units -= units
        pos.flags.add("scaled_out")
        self.diag.scale_outs = getattr(self.diag, "scale_outs", 0) + 1
        self._update_position_values(pos)
        self.log(t, "scale_out", f"{reason}: closed {units} unit{'s' if units != 1 else ''}, {pos.units} left open", pos, {"explain": explain} if explain else None)
        return True

    def _close_orphan_hedge(self, pos: Position, t: int, px: float) -> None:
        """Hedge shares are held AGAINST options. Once no option leg of the position is open (they expired or were
        settled), the hedge is closed at `px` instead of living on as a stock position that the strategy never chose."""
        if any(l.is_open and l.kind == "option" for l in pos.legs) or not math.isfinite(px):
            return
        for l in pos.legs:
            if l.is_open and l.leg_id == "_hedge":
                l.bid = l.ask = l.mid = px
                l.stale = False
                n = int(abs(l.qty))
                self._close_leg(pos, l, n, t, "hedge_closed")
                self.log(t, "hedge", f"closed the hedge ({n} shares at {l.exit_price:.2f}): the options it was held against have expired", pos)

    def hedge_trade(self, pos: Position, d_shares: float, ctx: Context) -> None:
        """Buy (+) or sell (-) shares of the underlying inside a position, as its delta hedge."""
        d = int(round(d_shares))
        if d == 0:
            return
        spot = ctx.price(pos.symbol)
        if not math.isfinite(spot):
            return
        sign = 1 if d > 0 else -1
        px = X.stock_fill_price(spot, sign, self.ex)
        comm = X.stock_commission(d, self.ex)
        leg = next((l for l in pos.legs if l.leg_id == "_hedge"), None)
        if leg is None:
            leg = LegState("_hedge", "stock", False, 0.0, -1, -1, 0, 1.0, "hedge", qty=0, entry_price=px, entry_bid=spot, entry_ask=spot, entry_mid=spot, entry_greeks=(1.0, 0.0, 0.0, 0.0))
            pos.legs.append(leg)
        leg.qty += d
        leg.bid = leg.ask = leg.mid = spot
        leg.stale = False
        flow = -d * px
        pos.exit_cash += flow  # hedge cash flows are part of the position's realised result
        pos.commissions += comm
        pos.slippage += abs(px - spot) * abs(d)
        self.account.cash += flow - comm
        self.account.commissions += comm
        self.account.slippage += abs(px - spot) * abs(d)
        h = pos.meta.setdefault("hedge", {"trades": 0, "shares_traded": 0, "cash": 0.0})
        h["trades"] += 1
        h["shares_traded"] += abs(d)
        h["cash"] += flow
        h["shares"] = leg.qty
        pos.flags.add("delta_hedged")
        self._update_position_values(pos)
        if len(self.events_log) < 20000:
            self.log(ctx.t, "hedge", f"{'bought' if d > 0 else 'sold'} {abs(d)} shares at {px:.2f} (hedge now {leg.qty:+d} shares)", pos)

    def _finalize(self, pos: Position, t: int, reason: str):
        pos.exit_ts = t
        pos.exit_reason = pos.exit_reason or reason
        pos.mark_value = pos.close_value = 0.0
        if pos in self.account.positions:
            self.account.positions.remove(pos)
        self.account.closed.append(pos)
        pnl = pos.entry_cash + pos.exit_cash - pos.commissions
        self.log(t, "exit", f"closed ({pos.exit_reason}) P&L {pnl:+,.2f}", pos)
        for s in self.strategies:
            if s.idx == pos.strategy_idx:
                s.on_position_closed(self.ctx, pos)

    # ------------------------------------------------------------------ settlement
    def _deliver(self, pos: Position, shares: float, price: float):
        """Stock delivered by exercise/assignment at `price`. Offsets opposite stock legs in
        the position first (e.g. a covered call's shares are called away)."""
        self.account.cash -= shares * price
        pos.exit_cash -= shares * price
        for l in pos.legs:
            if abs(shares) < 1e-9:
                break
            if l.kind == "stock" and l.is_open and np.sign(l.qty) == -np.sign(shares):
                sgn = int(np.sign(l.qty))
                n = min(abs(l.qty), abs(shares))
                prev = l.closed_qty
                l.exit_price = price if prev == 0 else (l.exit_price * prev + price * n) / (prev + n)
                l.exit_ts, l.exit_reason = self.ctx.t, "delivered"
                l.closed_qty += n
                l.exit_cash += sgn * n * price
                l.qty -= sgn * n
                shares += sgn * n
                if l.leg_id == "_hedge":  # delivered shares that close hedge shares are the hedge's closing trade, at the strike
                    h = pos.meta.setdefault("hedge", {"trades": 0, "shares_traded": 0, "cash": 0.0})
                    h["trades"] += 1
                    h["shares_traded"] += int(n)
                    h["cash"] += sgn * n * price
                    h["shares"] = int(l.qty)
        pos.shares += shares

    def _settles_am(self, sym: str, exp: date) -> bool:
        st = self.metas[sym].settlement_time
        if st == "AM":
            return True
        if st == "AM_MONTHLY":
            return exp.weekday() == 4 and 15 <= exp.day <= 21
        return False

    def _trade_days(self, sym: str, ds) -> set:
        """Sessions on which at least one stored contract of this trade-bar dataset traded (from its index, read once)."""
        cache = self.__dict__.setdefault("_trade_days_cache", {})
        if sym not in cache:
            idx = self.store.option_bar_index(sym, dataset=ds)
            cache[sym] = set(idx.filter(pl.col("status") == "ok")["date"].unique().to_list()) if not idx.is_empty() else set()
        return cache[sym]

    def _expirations_in_play(self, sym: str, d, ds):
        """The expirations a strategy of this symbol could select today, plus those of its open positions: (dates, [(first,
        last)]). None when the whole stored chain has to be loaded.

        A same-day strategy reads one expiration of the five or six stored for a session; expanding the others into the
        chain costs most of the session's load time and none of it is ever looked at. The window of a leg that takes "the
        nearest expiration" is the distance selection itself accepts on selectively stored data (`dte_tolerance`), so what is
        left out could not have been chosen. Anything that reads the chain as a whole keeps the whole chain: implied
        volatility of the underlying, rolls to another expiration, and a dataset that holds complete chains."""
        from strata.engine.conditions import walk_leaves
        from strata.engine.selection import dte_tolerance
        if self._complete_chain(ds):
            return None
        exact: set = set()
        spans: list[tuple] = []
        for c in self.cfg.strategies:
            if c.symbol != sym:
                continue
            if c.exits.roll.enabled or c.hedge.enabled:
                return None
            if any(re.search(r"\biv\b|iv_rank|iv_percentile|\biv\(", x) for x in c.expressions()):
                return None
            if any(l.type in ("iv", "iv_rank", "iv_percentile") for tree in (c.entry.conditions, c.exits.custom, *[sy.conditions for sy in c.entry.systems]) for l in walk_leaves(tree)):
                return None
            for l in list(c.legs) + [x for sy in c.entry.systems for x in (sy.legs or [])]:
                if l.kind != "option":
                    continue
                if l.expiration is not None:
                    exact.add(l.expiration)
                elif l.dte_mode == "exact" or l.dte == 0:
                    exact.add(d + timedelta(days=int(l.dte)))
                else:
                    tol = dte_tolerance(l.dte)
                    lo = max(0, l.dte - tol) if l.dte_mode != "at_least" else l.dte
                    hi = l.dte + tol if l.dte_mode != "at_most" else l.dte
                    if l.dte_min is not None:
                        lo = max(lo, l.dte_min)
                    if l.dte_max is not None:
                        hi = min(hi, l.dte_max)
                    spans.append((d + timedelta(days=int(lo)), d + timedelta(days=int(hi))))
        for p in self.account.positions:
            if p.symbol == sym:
                exact |= {from_day_num(int(l.exp)) for l in p.legs if l.kind == "option" and l.exp >= 0}
        return exact, spans

    def _trade_chain(self, sym: str, d, ds, ticks: list[int]):
        """The session's chain as it could be observed at each tick, from the stored trade bars (see expand_trade_bars)."""
        cols = ["ts", "occ", "expiration", "strike", "right", "close", "volume"]  # what the chain is built from
        play = self._expirations_in_play(sym, d, ds)
        bars = None
        if play is not None:
            exact, spans = play
            keep = pl.col("expiration").is_in(sorted(exact)) if exact else pl.lit(False)
            for a_, b_ in spans:
                keep = keep | ((pl.col("expiration") >= a_) & (pl.col("expiration") <= b_))
            bars = self.store.read_option_bars(sym, d, dataset=ds, columns=cols, where=keep)  # only those expirations leave the file
            if bars is not None and bars.is_empty():
                bars = None
            # none of them is stored for this session: the whole chain is kept, so that the selection says exactly which
            # expiration was missing instead of this looking like a session with no option data at all
        if bars is None:
            bars = self.store.read_option_bars(sym, d, dataset=ds, columns=cols)
        df, cstats = expand_trade_bars(bars, ticks, self.series.get(sym), d) if bars is not None else (pl.DataFrame(), {})
        if df.is_empty():
            lst = self.diag.no_option_data.setdefault(sym, [])
            if d.isoformat() not in lst:
                lst.append(d.isoformat())
            return None
        ch = DayChain(df, self.cfg.capital.risk_free_rate)
        ch.kind, ch.contract_stats, ch.multipliers = "trades", cstats, getattr(self, "_mults", {}).get(sym, {})
        ch.complete = self._complete_chain(ds)
        return ch

    def _complete_chain(self, ds) -> bool:
        """Does the dataset hold EVERY contract of each session, or a band of them?

        Trade bars are downloaded selectively, so they are a band unless the dataset says it is whole (an imported vendor
        file). Quote chains are whole (archives and vendor files hold the full chain) unless the dataset says it is a band
        (a selective download of quotes for the contracts a strategy needs). Selection on a band must be able to show that
        the stored strikes contain the contract asked for; on a whole chain there is nothing beyond what is stored."""
        cache = self.__dict__.setdefault("_complete_cache", {})
        k = getattr(ds, "id", str(ds))
        if k not in cache:
            try:
                flag = self.store.descriptor(ds).get("complete_chain") if ds is not None else None
            except Exception:  # noqa: BLE001
                flag = None
            cache[k] = bool(flag) if getattr(ds, "obs", "trade") == "trade" else flag is not False
        return cache[k]

    def _partial_trade_sessions(self, sym: str, ds) -> dict:
        """{iso date: reason} for the sessions of an option trade dataset that the provider holds only in part."""
        cache = self.__dict__.setdefault("_partial_cache", {})
        if sym not in cache:
            cache[sym] = {k: (v.get("reason") or "the stored option trade history for this session is incomplete") for k, v in self.store.session_states(ds).items() if v.get("status") == "partial"}
        return cache[sym]

    def settle_expiring(self, d: date, am: bool):
        """Settle what expires on this session: at its open for contracts settled in the morning, at its close for the rest.

        A contract dated on a day the market is closed expires at the close of the last session before that date (see
        `effective_expiration`). A leg whose expiration has already passed without a session to settle it on (the session
        is not stored) is settled at the next open there is, and the run says so: it is never carried on as if alive."""
        from strata.engine.selection import effective_expiration
        dn = day_num(d)
        close_t = int(cal.session_bounds(d)[1].timestamp())
        for pos in list(self.account.positions):
            meta = self.metas[pos.symbol]
            ser = self.series[pos.symbol]
            legs, late = [], False
            for l in pos.legs:
                if not (l.is_open and l.kind == "option"):
                    continue
                eff = effective_expiration(l.exp)
                if eff == dn and (self._settles_am(pos.symbol, d) == am if l.exp == dn else not am):
                    legs.append(l)  # a moved expiration settles at the close, like the session it was moved to
                elif eff < dn and am:
                    legs.append(l)
                    late = True
            if not legs:
                continue
            S = ser.session_open(d) if am else ser.session_close(d)
            if late:
                pos.flags.add("settled_late")
                self.diag.warnings.append(f"{d}: a {pos.symbol} leg had expired on a session that is not stored; it was settled at this session's open ({S:.2f}), which is not the price it expired at")
            if not math.isfinite(S):
                self.diag.warnings.append(f"{d}: no settlement price for {pos.symbol}; legs settled at last stored price")
                S = pos.last_spot
            for l in legs:
                intr = max(S - l.strike, 0.0) if l.is_call else max(l.strike - S, 0.0)
                q = l.qty
                l.exit_bid = l.exit_ask = l.exit_mid = intr
                l.exit_ts = close_t
                prev = l.closed_qty
                l.closed_qty += abs(q)
                l.qty = 0

                def settled_at(px: float, _l=l, _prev=prev, _n=abs(q)) -> float:
                    # contracts already sold earlier (a staged exit) keep the price they were sold at: the leg's exit price is the average over all of them
                    return px if _prev == 0 else (_l.exit_price * _prev + px * _n) / (_prev + _n)
                if intr < 0.01:
                    l.exit_price, l.exit_reason = settled_at(0.0), "expired_worthless"
                    continue
                if meta.settlement == "cash" or self.ex.physical_settlement == "intrinsic":
                    flow = q * l.multiplier * intr
                    self.account.cash += flow
                    pos.exit_cash += flow
                    l.exit_price, l.exit_reason = settled_at(intr), "cash_settled"
                    l.exit_cash += flow
                    if meta.settlement != "cash":
                        pos.flags.add("settled_at_intrinsic")
                else:
                    shares = q * l.multiplier * (1 if l.is_call else -1)
                    self._deliver(pos, shares, l.strike)
                    l.exit_price, l.exit_reason = settled_at(intr), ("exercised" if q > 0 else "assigned")
                    fee = self.ex.exercise_fee * abs(q)
                    pos.commissions += fee
                    self.account.cash -= fee
                    self.account.commissions += fee
                    pos.flags.add("exercised" if q > 0 else "assigned")
                    if q > 0:
                        self.diag.exercises += 1
                    else:
                        self.diag.assignments += 1
                    self.log(close_t, "settlement", f"{l.label()} {'exercised' if q > 0 else 'assigned'} at {S:.2f}: {shares:+.0f} shares @ {l.strike:g}", pos)
            if abs(pos.shares) > 1e-9:
                pos.pending_stock_close = True
            self.ctx.t = close_t
            self._close_orphan_hedge(pos, close_t, S)
            self._update_position_values(pos)
            if not pos.is_open:
                if pos.pending_exit:
                    # a rule had decided to exit but no fresh price ever came to close at: the position ended by settlement, not by a fill.
                    # The reason stays the rule that decided it; the flag (and each leg's own exit reason) says how it actually ended.
                    pos.flags.add("settled_after_deferred_exit")
                self._finalize(pos, close_t, pos.pending_exit or "expiration")

    def early_assignment(self, d: date):
        if not self.ex.early_assignment:
            return
        nxt = cal.next_trading_day(d)
        close_t = int(cal.session_bounds(d)[1].timestamp())
        for pos in list(self.account.positions):
            meta = self.metas[pos.symbol]
            if meta.exercise != "american" or meta.settlement != "physical":
                continue
            S = self.series[pos.symbol].session_close(d)
            div = self.dividends.get(pos.symbol, {}).get(nxt, 0.0)
            for l in pos.legs:
                if not (l.is_open and l.kind == "option" and l.qty < 0) or l.exp <= day_num(d):
                    continue
                intr = max(S - l.strike, 0.0) if l.is_call else max(l.strike - S, 0.0)
                if intr <= 0 or not math.isfinite(l.mid):
                    continue
                extrinsic = l.mid - intr
                trigger = extrinsic < self.ex.assignment_extrinsic_threshold or (l.is_call and div > 0 and extrinsic < div)
                if not trigger:
                    continue
                q = l.qty
                shares = q * l.multiplier * (1 if l.is_call else -1)
                l.exit_bid, l.exit_ask, l.exit_mid = l.bid, l.ask, l.mid
                l.exit_price, l.exit_reason, l.exit_ts = intr, "early_assignment", close_t
                l.closed_qty += abs(q)
                l.qty = 0
                self._deliver(pos, shares, l.strike)
                pos.flags.add("early_assignment")
                pos.pending_stock_close = True
                self.diag.assignments += 1
                self.log(close_t, "assignment", f"early assignment of {l.label()} (extrinsic {extrinsic:.3f}{', dividend %.2f' % div if div else ''}): {shares:+.0f} shares @ {l.strike:g}", pos)
            self._update_position_values(pos)

    def liquidate_stock(self, d: date, t_open: int, at_close: bool = False):
        for pos in list(self.account.positions):
            if not pos.pending_stock_close:
                continue
            ser = self.series[pos.symbol]
            px_raw = ser.session_close(d) if at_close else ser.session_open(d)
            if not math.isfinite(px_raw):
                continue
            sh = pos.shares
            px = X.stock_fill_price(px_raw, -int(np.sign(sh)), self.ex)
            flow = sh * px
            comm = X.stock_commission(sh, self.ex)
            self.account.cash += flow - comm
            self.account.commissions += comm
            pos.exit_cash += flow
            pos.commissions += comm
            pos.slippage += abs(px - px_raw) * abs(sh)
            pos.shares = 0.0
            pos.pending_stock_close = False
            self.log(t_open, "stock_close", f"liquidated {sh:+.0f} delivered shares @ {px:.2f}", pos)
            self.ctx.t = t_open
            self._update_position_values(pos)
            if not pos.is_open:
                reason = "assigned" if "assigned" in pos.flags or "early_assignment" in pos.flags else "exercised"
                self._finalize(pos, t_open, pos.exit_reason or reason)

    def apply_splits(self, d: date):
        for sym, ser in self.series.items():
            r = self.splits.get(sym, {}).get(d)
            if r and getattr(ser, "split_adjusted", False):
                ser.rescale(1.0 / r)  # history is now quoted in post-split share terms
        for pos in self.account.positions:
            r = self.splits.get(pos.symbol, {}).get(d)
            if not r:
                continue
            for l in pos.legs:
                if l.kind == "stock":
                    l.qty = int(round(l.qty * r))
                    l.entry_price /= r
                    continue
                l.strike /= r
                if abs(r - round(r)) < 1e-9:
                    l.qty = int(l.qty * round(r))
                    l.ratio = int(l.ratio * round(r))
                    l.entry_price /= r
                else:
                    l.multiplier *= r
                    l.entry_price /= r
                from strata.engine.market import contract_key
                l.key = int(contract_key(np.array([l.exp]), np.array([l.is_call]), np.array([l.strike]))[0])
            pos.shares *= r
            pos.flags.add("contract_adjusted")
            self.log(int(cal.session_bounds(d)[0].timestamp()), "corporate_action", f"{r:g}:1 split applied to position legs", pos)

    # ------------------------------------------------------------------ margin
    def margin_check(self, t: int):
        for p in self.account.positions:
            p.requirement = requirement(p.margin_legs(), p.stock_qty(), p.last_spot, self.cfg.capital.margin_model)
        if not self.cfg.capital.liquidate_on_margin_call:
            return
        guard = 0
        while self.account.available_bp() < -1e-6 and self.account.positions and guard < 50:
            guard += 1
            worst = max(self.account.positions, key=lambda p: p.requirement)
            self.diag.margin_calls += 1
            worst.flags.add("margin_call")
            self.log(t, "margin_call", f"margin deficit {self.account.available_bp():,.0f}: liquidating position {worst.id}", worst)
            if not self.close_position(worst, "margin_call", self.ctx):
                break

    # ------------------------------------------------------------------ clocks
    def _signal_ticks(self, d: date, t_open: int, t_close: int) -> set[int]:
        """Times at which strategies evaluate their rules, derived from the UNDERLYING's
        bars (independent of when option quotes exist): the open, every
        `signal_interval_minutes` of completed bars, the close, plus any clock times the
        rules mention when the session's bar resolution can observe them."""
        step = max(60, int(self.ex.signal_interval_minutes) * 60)
        ticks: set[int] = set()
        for st in self.strategies:
            ser = self.series.get(st.symbol)
            if ser is None:
                continue
            di = ser.day_index(d)
            if di < 0:
                continue
            a, b = int(ser.day_start[di]), int(ser.day_end[di])
            res = int(ser.d_res[di])
            ticks.add(t_open)
            ends = ser.end[a:b]
            ends = ends[(ends > t_open) & (ends <= t_close)]
            grid = max(step, res) if res < 86400 else 86400
            ticks |= {int(e) for e in ends if (int(e) - t_open) % grid == 0 or int(e) == t_close}
            if res < 86400:
                for m in getattr(st, "explicit_minutes", ()):
                    t = t_open + (m - 570) * 60
                    if t_open <= t <= t_close and (t - t_open) % res == 0:
                        ticks.add(t)
        return ticks

    def _modeled_chain(self, sym: str, d: date, ticks: list[int]):
        from strata.data.modeled import base_vol, modeled_chain
        pr = self.cfg.pricing
        ser = self.series.get(sym)
        if ser is None or ser.day_index(d) < 0 or not ticks:
            return None
        every = max(60, pr.snapshot_minutes * 60)
        t0 = ticks[0]
        use = [t for t in ticks if (t - t0) % every == 0 or t == ticks[-1]]
        vs = self.metas[sym].vol_index
        vser = self.series.get(vs) if vs else None
        spots, vols = [], []
        i = ser.completed_day(t0 - 1)
        rv = float(np.diff(np.log(ser.d_c[max(0, i - 21):i + 1])).std(ddof=1) * math.sqrt(252)) if i >= 5 else float("nan")
        for t in use:
            spots.append(ser.price_at(t, d))
            vix = float("nan")
            if vser is not None and not vser.empty():
                pi = vser.prev_day_index(d)
                vix = vser.price_at(t, d) if vser.day_index(d) >= 0 and (vser.resolution(d) < 86400 or t >= int(vser.d_end[vser.day_index(d)])) else (float(vser.d_c[pi]) if pi >= 0 else float("nan"))
            v, self.model_vol_note = base_vol(pr, vix, rv)
            vols.append(v)
        max_dte = max((max(l.dte, l.dte_max or 0, (l.expiration - d).days if l.expiration else 0) for c in self.cfg.strategies if c.symbol == sym for l in c.legs if l.kind == "option"), default=7)
        df = modeled_chain(d, use, spots, vols, pr, max_dte)
        return DayChain(df, pr.rate) if not df.is_empty() else None

    # ------------------------------------------------------------------ main loop
    def run(self) -> "BacktestResult":
        t0 = _time.time()
        self._load()
        self.ctx = ctx = Context(self)
        days = cal.trading_days(self.cfg.start, self.cfg.end)
        for s in self.strategies:
            ctx.strategy = s
            s.on_start(ctx)
        if self._resume is not None:
            self.restore(self._resume)
        last_t = 0
        for di, d in enumerate(days):
            if self.should_cancel and self.should_cancel():
                raise BacktestCancelled()
            ctx.today, ctx.today_num = d, day_num(d)
            if self.only_sessions is not None and d not in self.only_sessions and not self.account.positions:
                self._day_max_bp, self._day_max_pos = 0.0, 0
                self._day_open_equity = self.account.equity()
                self._peak_equity = max(getattr(self, "_peak_equity", self._day_open_equity), self._day_open_equity)
                self._record_day(d)
                if self.progress and (di % 5 == 0 or di == len(days) - 1):
                    self.progress(di + 1, len(days), d)
                continue
            open_dt, close_dt = cal.session_bounds(d)
            ctx.close_minute = close_dt.hour * 60 + close_dt.minute
            t_open = int(open_dt.timestamp())
            ctx.t = ctx.prev_t = t_open
            ctx._feat.clear()
            self.iv_cache.clear()
            self.liquidate_stock(d, t_open)
            self.apply_splits(d)
            t_close = int(close_dt.timestamp())
            sig_ticks = self._signal_ticks(d, t_open, t_close)
            self.chains = _DayChains()
            for sym in self.opt_syms:
                if self.signals_only:
                    self.chains[sym] = None
                    continue
                if self.modeled:
                    self.chains[sym] = self._modeled_chain(sym, d, sorted(sig_ticks))
                    if self.chains[sym] is None:
                        self.diag.exclude(d, f"{sym}: no underlying prices to model option premiums from")
                    continue
                ds = self.opt_ds[sym]
                if self.trades and self._bars_fmt.get(sym):
                    part = self._partial_trade_sessions(sym, ds).get(d.isoformat())
                    if part is not None:
                        # the provider holds this session only in part: an entry would slide to whatever print it happens to have
                        self.chains[sym] = None
                        self.diag.exclude(d, f"{sym}: {part}")
                        continue
                    if d not in self._trade_days(sym, ds) or self.series.get(sym) is None:
                        # no trade data stored for this session: recorded separately from "no signal"
                        self.chains[sym] = None
                        self.diag.no_option_data.setdefault(sym, []).append(d.isoformat())
                        continue
                    # built on first use: only a session on which a rule fires or a position is open pays for its option data
                    self.chains.defer(sym, lambda sym=sym, d=d, ds=ds, ticks=sorted(sig_ticks): self._trade_chain(sym, d, ds, ticks))
                    continue
                df = self.store.read_option_day(sym, d, "trade" if self.trades else "quote", dataset=ds)
                if df is None or df.is_empty():
                    self.chains[sym] = None
                    self.diag.missing_sessions.setdefault(sym, []).append(d)
                else:
                    self.chains[sym] = DayChain(df, self.cfg.capital.risk_free_rate)
                    if self.trades:
                        self.chains[sym].kind = "trades"
                    self.chains[sym].complete = self._complete_chain(ds)
            self.settle_expiring(d, am=True)
            clock = sorted({int(x) for ch in self.chains.values() if ch is not None for x in ch.snap_ts if t_open <= x <= t_close} | sig_ticks)
            diffs = np.diff(clock) if len(clock) > 1 else np.array([60])
            ctx.tick_minutes = max(1, int(np.median(diffs) // 60))
            ctx.day_minutes = {570 + (t - t_open) // 60 for t in clock}
            self._day_max_bp = self.account.bp_used()
            self._day_max_pos = len(self.account.positions)
            self._t_open = t_open
            self._day_open_equity = self.account.equity()
            self._peak_equity = max(getattr(self, "_peak_equity", self._day_open_equity), self._day_open_equity)
            for s in self.strategies:
                ctx.strategy = s
                s.on_day_start(ctx)
            prev = None
            for ti, t in enumerate(clock):
                ctx.prev_t = prev if prev is not None else t
                prev = t
                ctx.t = t
                ctx.minute = 570 + (t - t_open) // 60
                ctx.is_last_tick = ti == len(clock) - 1
                ctx._feat.clear()
                self.mark_all(t)
                for s in self.strategies:
                    ctx.strategy = s
                    s.on_snapshot(ctx)
                self.margin_check(t)
                self._peak_equity = max(self._peak_equity, self.account.equity())
                self._day_max_bp = max(self._day_max_bp, self.account.bp_used())
                self._day_max_pos = max(self._day_max_pos, len(self.account.positions))
                self.intraday_equity["t"].append(t)
                self.intraday_equity["equity"].append(self.account.equity())
                last_t = t
            for s in self.strategies:
                ctx.strategy = s
                s.on_day_end(ctx)
            ctx.t = int(close_dt.timestamp())
            self.settle_expiring(d, am=False)
            self.early_assignment(d)
            if di == len(days) - 1:
                self.liquidate_stock(d, int(close_dt.timestamp()), at_close=True)
            self._record_day(d)
            if self.progress and (di % 5 == 0 or di == len(days) - 1):
                self.progress(di + 1, len(days), d)
        # close anything still open at the last snapshot, at the last observed quotes (a continuing record keeps it open)
        for pos in ([] if self.keep_open else list(self.account.positions)):
            ctx.t = last_t or ctx.t
            if any(l.stale for l in pos.legs if l.is_open):
                pos.flags.add("closed_at_last_known_quote")
            for l in pos.legs:
                l.stale = not math.isfinite(l.mid)
            if not self.close_position(pos, "end_of_backtest", ctx, ignore_liquidity=True):
                pos.flags.add("unfilled_at_end")
                self._finalize(pos, ctx.t, "end_of_backtest")
        if self.daily:
            self.daily[-1]["equity"] = self.account.equity()
            self.daily[-1]["cash"] = self.account.cash
            self.daily[-1]["open_positions"] = len(self.account.positions)  # everything was just closed: the last row must not still count it as open
            self.daily[-1]["commissions"] = self.account.commissions
        return self._result(_time.time() - t0)

    # ------------------------------------------------------------------ state: saving and continuing
    def state(self) -> dict:
        """Everything a later run needs to continue this account exactly: cash, every open position with its legs and what
        its exit rules remember, the position counter, and what each strategy carries from one session to the next.
        JSON-safe (a number that is not a number is stored as null and restored as such)."""
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple, set, frozenset)):
                return [clean(v) for v in (sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)]
            if isinstance(o, (np.floating, float)):
                return float(o) if math.isfinite(float(o)) else None
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (date, datetime)):
                return o.isoformat()
            return o
        acct = self.account
        return clean({"version": 1, "engine_version": ENGINE_VERSION, "through": self.daily[-1]["date"] if self.daily else None, "initial": acct.initial, "cash": acct.cash, "commissions": acct.commissions,
                      "slippage": acct.slippage, "interest": getattr(acct, "interest", 0.0), "next_pos_id": self.next_pos_id, "peak_equity": getattr(self, "_peak_equity", acct.initial),
                      # every attribute, not only the declared fields: the engine also notes on a leg whether it was priced from a
                      # trade print, the volume it saw at entry, and on a position the instant of its signal
                      "positions": [{**{k: v for k, v in vars(p).items() if k != "legs"}, "legs": [dict(vars(l)) for l in p.legs]} for p in acct.positions],
                      "strategies": [{"idx": st.idx, "name": st.name, "last_exit_t": getattr(st, "last_exit_t", None), "last_exit_reason": getattr(st, "last_exit_reason", None)} for st in self.strategies],
                      "closed": len(acct.closed) + int((self._resume or {}).get("closed") or 0),
                      "realised": sum(p.entry_cash + p.exit_cash - p.commissions for p in acct.closed) + float((self._resume or {}).get("realised") or 0.0)})

    def restore(self, st: dict) -> None:
        """Start from a saved state instead of from an empty account."""
        from dataclasses import fields
        if int(st.get("version") or 0) != 1:
            raise ValueError("this saved state was written in a format this engine does not read")
        nan4 = (math.nan,) * 4

        def build(cls, raw: dict, **over):
            kw = {}
            declared = set()
            for f in fields(cls):
                declared.add(f.name)
                if f.name in over:
                    kw[f.name] = over[f.name]
                    continue
                if f.name not in raw:
                    continue
                v = raw[f.name]
                if f.name in ("entry_greeks", "exit_greeks", "greeks"):
                    v = tuple(math.nan if x is None else float(x) for x in (v or nan4))
                elif v is None and isinstance(f.default, float):
                    v = math.nan if not math.isfinite(f.default) else f.default  # a null where a number lives was "not a number" when it was saved
                elif f.name == "flags":
                    v = set(v or [])
                kw[f.name] = v
            obj = cls(**kw)
            for k, v in raw.items():  # what the engine noted on the object beyond its declared fields
                if k not in declared and k != "legs":
                    setattr(obj, k, v)
            return obj
        acct = self.account
        acct.cash, acct.commissions, acct.slippage = float(st["cash"]), float(st.get("commissions") or 0.0), float(st.get("slippage") or 0.0)
        if st.get("interest"):
            acct.interest = float(st["interest"])
        acct.positions = [build(Position, p, legs=[build(LegState, l) for l in p.get("legs") or []]) for p in st.get("positions") or []]
        self.next_pos_id = int(st.get("next_pos_id") or 1)
        self._peak_equity = float(st.get("peak_equity") or acct.initial)
        by = {x.get("idx"): x for x in st.get("strategies") or []}
        for strat in self.strategies:
            x = by.get(strat.idx)
            if x is not None and hasattr(strat, "last_exit_t"):
                strat.last_exit_t, strat.last_exit_reason = x.get("last_exit_t"), x.get("last_exit_reason")

    def _close_gaps(self) -> list[dict]:
        rows = _close_gap_rows(self.series, self.cfg.start, self.cfg.end)
        if rows:
            w = rows[0]
            self.diag.warnings.append(f"{len(rows)} session(s) have intraday bars that end away from the official close (largest: {w['symbol']} {w['date']}, last bar {w['last_intraday_bar']:.2f} against an official close of "
                                      f"{w['official_close']:.2f}). Day-level rules and expiration settlement use the official close.")
        return rows

    def _record_day(self, d: date):
        acct = self.account
        rate = self.cfg.capital.cash_interest_rate
        if rate:
            prev = self.daily[-1]["date"] if self.daily else d - timedelta(days=1)
            idle = max(acct.cash - acct.total_requirement(), 0.0)
            interest = idle * rate * (d - prev).days / 365.0
            acct.cash += interest
            acct.interest = getattr(acct, "interest", 0.0) + interest
        d_, g, th, v = 0.0, 0.0, 0.0, 0.0
        for p in acct.positions:
            a, b, c, e = p.greeks()
            d_ += a
            g += b
            th += c
            v += e
        bser = self.series.get(self.bench)
        vix_sym = self.metas[self.opt_syms[0]].vol_index
        vser = self.series.get(vix_sym) if vix_sym else None
        per = {}
        for s in self.strategies:
            realized = sum(p.entry_cash + p.exit_cash - p.commissions for p in acct.closed if p.strategy_idx == s.idx)
            unreal = sum(p.pnl_mark() for p in acct.positions if p.strategy_idx == s.idx)
            per[f"s{s.idx}_pnl"] = realized + unreal
            per[f"s{s.idx}_open"] = sum(1 for p in acct.positions if p.strategy_idx == s.idx)
        self.daily.append({
            **per,
            "date": d, "equity": acct.equity(), "cash": acct.cash, "bp_used": acct.bp_used(), "requirement": acct.total_requirement(),
            "open_positions": len(acct.positions), "bp_used_max": max(getattr(self, "_day_max_bp", 0.0), acct.bp_used()),
            "positions_max": max(getattr(self, "_day_max_pos", 0), len(acct.positions)), "delta": d_, "gamma": g, "theta": th, "vega": v,
            "benchmark": bser.session_close(d) if bser else math.nan,
            "vix": vser.session_close(d) if vser else math.nan,
            "commissions": acct.commissions, "interest": getattr(acct, "interest", 0.0),
        })

    def _contract_selection(self) -> dict:
        """How contracts were chosen, and whether the choice was limited by what is stored.

        With selectively downloaded trade data only a band of strikes exists. A selection
        that lands on the outermost stored strike may not be the contract the rule wanted,
        so those are counted and their sessions listed (the research pipeline widens the
        band for them and runs again)."""
        picks = self.diag.picks
        trade_days = {cal.to_et(p.entry_ts * 1_000_000_000).date().isoformat() for p in list(self.account.closed) + list(self.account.positions)}
        used = [p for p in picks if p["date"] in trade_days]
        last: dict[tuple, dict] = {}
        for p in used:
            last[(p["date"], p["leg"])] = p  # the selection that led to the fill is the last one of the session for that leg
        rows = list(last.values())
        complete = True
        constructed = adjusted = 0
        mults: dict[str, int] = {}
        for sym in self.opt_syms:
            ds = (self.datasets_used or {}).get(sym) if isinstance(self.datasets_used, dict) else None
            try:
                from strata.data.store import DatasetKey
                key = DatasetKey.parse(ds["dataset"] if isinstance(ds, dict) else ds) if ds else None
                desc = self.store.descriptor(key) if key else {}
                if (self.trades and not desc.get("complete_chain")) or (not self.trades and desc.get("complete_chain") is False):
                    complete = False  # a band of strikes: trade bars unless the dataset says it is whole, quotes only when it says it is a band
                if key is not None and self.trades:
                    con = self.store.read_contracts(key)
                    if not con.is_empty():
                        constructed += int(con["source"].fill_null("").str.contains("constructed").sum()) if "source" in con.columns else 0
                        adjusted += int(con["adjusted"].fill_null(False).sum()) if "adjusted" in con.columns else 0
                        if "multiplier" in con.columns:
                            for m_, n_ in con.group_by("multiplier").len().iter_rows():
                                mults[f"{m_:g}" if m_ is not None else "unknown"] = int(n_)
            except Exception:  # noqa: BLE001  (diagnostics only)
                pass
        selective = (not complete) and not self.modeled
        edge = [r for r in rows if r.get("edge") and selective and r["method"] in ("delta", "premium", "pct_otm")]
        # signals that were NOT filled because the contract asked for lies beyond the stored strikes: nothing was substituted
        # for them, and a wider download can still price them
        beyond = sorted({cal.to_et(x["signal_t"] * 1_000_000_000).date().isoformat() for x in self.diag.skipped_signals
                         if x.get("kind") == "contract_not_stored" and ("beyond the stored strike range" in x.get("reason", "") or "outside the stored strike range" in x.get("reason", ""))}) if selective else []
        d_err = [abs(r["achieved"] - r["target"]) for r in rows if r["method"] == "delta" and r.get("achieved") is not None]
        p_err = [abs(r["achieved"] - r["target"]) / r["target"] for r in rows if r["method"] == "premium" and r.get("achieved") is not None and r["target"]]
        return {"selections": len(rows), "selective_storage": bool(selective), "complete_chain": bool(not selective), "at_band_edge": len(edge), "edge_dates": sorted({r["date"] for r in edge}),
                "beyond_band": len(beyond), "beyond_dates": beyond,
                "edge_examples": [{k: r.get(k) for k in ("date", "leg", "method", "strike", "stored_low", "stored_high", "edge", "target", "achieved")} for r in edge[:8]],
                "delta_error_median": float(np.median(d_err)) if d_err else None, "delta_error_max": float(max(d_err)) if d_err else None, "delta_misses": int(sum(1 for x in d_err if x > 0.10)),
                "premium_error_median": float(np.median(p_err)) if p_err else None, "premium_misses": int(sum(1 for x in p_err if x > 0.25)),
                "methods": {m: sum(1 for r in rows if r["method"] == m) for m in sorted({r["method"] for r in rows})},
                "contracts_constructed": constructed, "contracts_adjusted": adjusted, "multipliers": mults,
                "note": ("Contracts are chosen among a downloaded band of strikes around the price at each signal. A strike that lies beyond the band is never replaced by the outermost stored one: the signal is left unfilled, "
                         "and the band is widened and the run repeated whenever a selection lands on its edge or beyond it." if selective else
                         "Contracts are chosen among every contract in the stored chain as it stood at the decision time.")}

    def _executability(self) -> dict:
        """How many signals could actually be executed with the observed option prices."""
        closed = self.account.closed
        ages = [p.meta.get("entry_print_age_seconds") for p in closed if p.meta.get("entry_print_age_seconds") is not None]
        freq = [p.meta.get("contract_minutes_traded_pct") for p in closed if p.meta.get("contract_minutes_traded_pct") is not None]
        sig = self.diag.signals
        by: dict[str, int] = {}
        for s in self.diag.skipped_signals:
            by[s["kind"]] = by.get(s["kind"], 0) + 1
        return {"signals": sig, "executed": self.diag.entries, "executed_pct": (self.diag.entries / sig * 100) if sig else None, "not_executed": max(sig - self.diag.entries, 0),
                "not_executed_by_reason": by, "expired_orders": self.diag.expired_orders, "stale_marks": self.diag.stale_marks, "deferred_exits": self.diag.deferred_exits,
                "sessions_without_option_data": {k: len(v) for k, v in self.diag.no_option_data.items()},
                "median_entry_print_age_seconds": float(np.median(ages)) if ages else None, "max_entry_print_age_seconds": max(ages) if ages else None,
                "median_contract_minutes_traded_pct": float(np.median(freq)) if freq else None, "max_trade_age_seconds": self.ex.max_trade_age_seconds if self.trades else None,
                "trade_bar_slippage_pct": self.ex.trade_bar_slippage_pct if self.trades else None}

    def _resolution_summary(self) -> dict:
        out = {}
        for sym in self.opt_syms:
            ser = self.series.get(sym)
            if ser is None:
                continue
            a = int(np.searchsorted(ser.days, day_num(self.cfg.start)))
            b = int(np.searchsorted(ser.days, day_num(self.cfg.end), side="right"))
            vals, counts = np.unique(ser.d_res[a:b], return_counts=True)
            used = self.datasets_used.get(sym)
            out[sym] = {"underlying_sessions_by_resolution": {_res_label(int(v)): int(c) for v, c in zip(vals, counts)},
                        "underlying_sessions_by_provider": getattr(ser, "providers", {}),
                        "options": "modeled (theoretical)" if self.modeled else (f"{used['resolution']} {used['observation']} via {used['provider']}" if used else "stored chains"),
                        "options_dataset": used["dataset"] if used else None,
                        "trade_based": bool(self.trades or getattr(self, "_saw_trade_based", False))}
        return out

    # ------------------------------------------------------------------ results
    def _result(self, runtime: float) -> "BacktestResult":
        trades = [trade_record(p, self) for p in sorted(self.account.closed, key=lambda p: p.id)]
        daily = pl.DataFrame(self.daily) if self.daily else pl.DataFrame()
        paths = pl.DataFrame({k: np.frombuffer(v, dtype=np.float64) for k, v in self.path_cols.items()}) if self.record_paths else None
        leg_paths = pl.DataFrame({k: np.frombuffer(v, dtype=np.float64) for k, v in self.leg_cols.items()}) if self.record_paths else None
        intraday = pl.DataFrame({k: np.frombuffer(v, dtype=np.float64) for k, v in self.intraday_equity.items()})
        meta = {
            "engine_version": ENGINE_VERSION,
            "config_hash": self.cfg.config_hash(),
            "dataset_versions": self.dataset_versions,
            "events_version": self.store.events_version(),
            "synthetic": self.synthetic,
            "pricing": self.cfg.pricing.mode,
            "datasets_used": self.datasets_used,
            "executability": self._executability(),
            "contract_selection": self._contract_selection(),
            "pricing_assumptions": (__import__("strata.data.modeled", fromlist=["x"]).assumptions_text(self.cfg.pricing) + [f"Volatility input used: {getattr(self, 'model_vol_note', 'n/a')}."]) if self.modeled else [],
            "data_resolution": self._resolution_summary(),
            "runtime_seconds": round(runtime, 3),
            **({"state": self.state(), "open_positions": [trade_record(p, self) for p in sorted(self.account.positions, key=lambda p: p.id)]} if self.keep_open else {}),
            "sessions": len(self.daily),
            "symbols": sorted(self.series.keys()),
            "benchmark": self.bench,
            "execution": self.cfg.execution.model_dump(),
            "capital": self.cfg.capital.model_dump(),
            "data_quality": {
                "official_close_gaps": self._close_gaps(),
                "volume_artefacts": {sym: int(ser.volume_artefacts) for sym, ser in self.series.items() if getattr(ser, "volume_artefacts", 0)},
                "missing_sessions": sum(len(v) for v in self.diag.missing_sessions.values()),
                "stale_marks": self.diag.stale_marks,
                "missing_marks": self.diag.missing_marks,
                "computed_greeks": any(ch is not None and ch.greeks_computed for ch in self.chains.values()),
            },
        }
        return BacktestResult(self.cfg, meta, trades, daily, paths, leg_paths, intraday, self.diag.to_dict(), self.events_log)


def _close_gap_rows(series: dict, start: date, end: date) -> list[dict]:
    """Sessions inside the window whose last intraday bar ends away from the official close (the official close is the one used)."""
    from strata.engine.market import from_day_num
    out = []
    for sym, ser in series.items():
        for dn, intraday, official in getattr(ser, "close_gaps", []):
            d = from_day_num(dn)
            if start <= d <= end:
                out.append({"symbol": sym, "date": d.isoformat(), "last_intraday_bar": round(intraday, 4), "official_close": round(official, 4), "difference_pct": round((intraday / official - 1) * 100, 3)})
    return sorted(out, key=lambda r: -abs(r["difference_pct"]))[:50]


def _res_label(sec: int) -> str:
    return "1d" if sec >= 86400 else f"{sec // 60}m"


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def trade_record(p: Position, bt: Backtester) -> dict:
    pnl = p.entry_cash + p.exit_cash - p.commissions
    legs = []
    for l in p.legs:
        legs.append({
            "id": l.leg_id, "kind": l.kind, "right": None if l.kind == "stock" else ("C" if l.is_call else "P"), "strike": l.strike,
            "expiration": from_day_num(l.exp).isoformat() if l.exp >= 0 else None, "ratio": l.ratio, "contracts": l.closed_qty,
            "side": "buy" if l.ratio > 0 else "sell", "group": l.group, "multiplier": l.multiplier,
            "price_source": "stock" if l.kind == "stock" else "model" if bt.modeled else "trade_print" if getattr(l, "entry_trade_based", False) else "quote",
            # trade-print data has no bid/ask: the observed print is reported as such, never as a quote
            "entry_print": _num(l.entry_mid) if getattr(l, "entry_trade_based", False) else None, "exit_print": _num(l.exit_mid) if getattr(l, "entry_trade_based", False) else None,
            "entry_price": _num(l.entry_price), "entry_bid": None if getattr(l, "entry_trade_based", False) else _num(l.entry_bid),
            "entry_ask": None if getattr(l, "entry_trade_based", False) else _num(l.entry_ask), "entry_mid": _num(l.entry_mid),
            "entry_volume": getattr(l, "entry_volume", None) if getattr(l, "entry_volume", -1) is not None and getattr(l, "entry_volume", -1) >= 0 else None,
            "entry_displayed_size": getattr(l, "entry_size", None) if getattr(l, "entry_size", -1) is not None and getattr(l, "entry_size", -1) >= 0 else None,
            "entry_iv": _num(l.entry_iv), "entry_delta": _num(l.entry_greeks[0]), "entry_gamma": _num(l.entry_greeks[1]),
            "entry_theta": _num(l.entry_greeks[2]), "entry_vega": _num(l.entry_greeks[3]),
            "exit_price": _num(l.exit_price), "exit_bid": None if getattr(l, "entry_trade_based", False) else _num(l.exit_bid),
            "exit_ask": None if getattr(l, "entry_trade_based", False) else _num(l.exit_ask), "exit_mid": _num(l.exit_mid),
            "exit_delta": _num(l.exit_greeks[0]), "exit_gamma": _num(l.exit_greeks[1]), "exit_theta": _num(l.exit_greeks[2]), "exit_vega": _num(l.exit_greeks[3]),
            "exit_ts": l.exit_ts or None, "exit_reason": l.exit_reason,
            # hedge shares are traded many times inside one position: their result is the net cash of all of those trades, not one entry and one exit price
            "pnl": _num((p.meta.get("hedge") or {}).get("cash")) if l.leg_id == "_hedge" else (_num(np.sign(l.ratio) * l.closed_qty * l.multiplier * (l.exit_price - l.entry_price)) if l.closed_qty else None),
        })
    opt_legs = [l for l in p.legs if l.kind == "option"]
    entry_dte = min((l.exp for l in opt_legs), default=p.entry_day) - p.entry_day
    short_deltas = [abs(l.entry_greeks[0]) for l in opt_legs if l.ratio < 0 and math.isfinite(l.entry_greeks[0])]
    all_deltas = [abs(l.entry_greeks[0]) for l in opt_legs if math.isfinite(l.entry_greeks[0])]
    mleg = [MLeg(l.is_call, l.strike, l.ratio * p.units, l.multiplier, 0.0) for l in opt_legs]
    single_exp = len({l.exp for l in opt_legs}) <= 1
    mp, ml = max_profit_loss(mleg, sum(l.ratio * p.units for l in p.legs if l.kind == "stock"), p.entry_cash) if single_exp and opt_legs else (None, None)
    ent_dt = cal.to_et(p.entry_ts * 1_000_000_000)
    return {
        "id": p.id, "strategy": p.strategy, "strategy_idx": p.strategy_idx, "symbol": p.symbol, "tag": p.tag,
        "signal_ts": getattr(p, "signal_ts", p.entry_ts), "fill_delay_seconds": p.entry_ts - getattr(p, "signal_ts", p.entry_ts),
        "exit_signal_ts": p.meta.get("exit_signal_ts"), "exit_delay_seconds": (p.exit_ts - p.meta["exit_signal_ts"]) if p.meta.get("exit_signal_ts") and p.exit_ts else 0,
        "entry_ts": p.entry_ts, "exit_ts": p.exit_ts, "entry_time": ent_dt.isoformat(), "exit_time": cal.to_et(p.exit_ts * 1_000_000_000).isoformat() if p.exit_ts else None,
        "entry_date": ent_dt.date().isoformat(), "entry_minute": ent_dt.hour * 60 + ent_dt.minute, "entry_weekday": ent_dt.weekday(),
        "holding_minutes": (p.exit_ts - p.entry_ts) / 60 if p.exit_ts else None, "units": p.units,
        "entry_cash": p.entry_cash, "exit_cash": p.exit_cash, "commissions": p.commissions, "slippage": p.slippage,
        "pnl": pnl, "gross_pnl": pnl + p.commissions, "max_risk": p.max_risk, "return_on_risk": pnl / p.max_risk if p.max_risk > 0 else None,
        "pnl_pct_premium": pnl / p.basis * 100, "credit": p.entry_cash > 0, "basis": p.basis,
        "mfe": p.mfe, "mae": p.mae, "exit_reason": p.exit_reason, "entry_spot": p.entry_spot, "exit_spot": _num(p.last_spot),
        "entry_dte": entry_dte, "short_delta": max(short_deltas) if short_deltas else (max(all_deltas) if all_deltas else None),
        "entry_vix": _num(p.entry_vix), "entry_iv": _num(p.entry_iv), "entry_regime": p.entry_regime,
        "max_profit": _num(mp), "max_loss": _num(ml), "flags": sorted(p.flags), "roll_group": p.roll_group, "roll_count": p.roll_count,
        "group": p.meta.get("group") or (p.id if p.meta.get("lot", 1) == 1 else None), "lot": p.meta.get("lot", 1),
        "system": p.meta.get("system"), "original_units": p.meta.get("original_units", p.units), "stage_exits": p.meta.get("stage_exits") or [], "hedge": p.meta.get("hedge"), "entry_atr": _num(p.meta.get("entry_atr")),
        "legs": legs, "entry_explain": p.entry_explain, "exit_explain": p.exit_explain, "meta": p.meta,
    }


@dataclass
class BacktestResult:
    config: BacktestConfig
    meta: dict
    trades: list[dict]
    daily: pl.DataFrame
    paths: pl.DataFrame | None
    leg_paths: pl.DataFrame | None
    intraday: pl.DataFrame
    diagnostics: dict
    events: list[dict]

    def fingerprint(self) -> str:
        import hashlib
        import json
        h = hashlib.sha256()
        h.update(json.dumps([{k: v for k, v in t.items() if k not in ("entry_explain", "exit_explain")} for t in self.trades], sort_keys=True, default=str).encode())
        if not self.daily.is_empty():
            h.update(self.daily.select(["equity", "cash"]).to_numpy().round(6).tobytes())
        return h.hexdigest()[:16]


def run_backtest(cfg: BacktestConfig | dict, store: MarketStore | None = None, strategies: list[Strategy] | None = None, **kw) -> BacktestResult:
    if isinstance(cfg, dict):
        cfg = BacktestConfig.model_validate(cfg)
    return Backtester(cfg, store, strategies, **kw).run()
