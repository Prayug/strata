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
                                  from_day_num)
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
    excluded_sessions: dict = field(default_factory=dict)  # reason -> [dates]
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
            if ch is not None:
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
        if self.t - int(ch.snap_ts[si]) > self.bt.cfg.execution.max_quote_age_seconds:
            return None, "latest chain snapshot is stale", {}
        return select_contracts(ch, si, legs, self.price(sym or self.symbol), self.today, self.bt.cfg.execution.max_quote_age_seconds)

    def open(self, selected: list[Selected], units: int, explain: list | None = None, tag: str = "", meta: dict | None = None):
        return self.bt.open_position(self.strategy, selected, units, self, explain or [], tag, meta or {})

    def close(self, pos: Position, reason: str, legs: list[int] | None = None, explain: list | None = None) -> bool:
        return self.bt.close_position(pos, reason, self, legs, explain or [])

    def position_pnl_pct(self, pos: Position, use_mid: bool = False) -> float:
        return pos.gross_pnl(use_mid) / pos.basis * 100

    def log(self, kind: str, msg: str, pos: Position | None = None, data: dict | None = None):
        self.bt.log(self.t, kind, msg, pos, data)


class Backtester:
    def __init__(self, cfg: BacktestConfig, store: MarketStore | None = None, strategies: list[Strategy] | None = None,
                 progress=None, should_cancel=None, record_paths: bool = True):
        from strata.engine.rules import RuleStrategy

        self.cfg = cfg
        self.store = store or MarketStore()
        self.ex = cfg.execution
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
        self.chains: dict[str, DayChain | None] = {}
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
        from strata.engine.conditions import extra_symbols
        for s in self.strategies:
            c = getattr(s, "cfg", None)
            if c is not None:
                syms |= extra_symbols(c.entry.conditions) | extra_symbols(c.exits.custom)
        bench = cfg.benchmark or opt_syms[0]
        syms.add(bench)
        self.bench = bench
        warm = cfg.start - timedelta(days=420)
        self.dataset_versions = {}
        self.synthetic = False
        for sym in sorted(syms):
            meta = self.store.symbol_meta(sym)
            self.metas[sym] = meta
            if meta.synthetic:
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
            ser.split_adjusted = (meta.extra or {}).get("price_adjustment") == "split"
            if ser.split_adjusted:  # vendor prices are split-adjusted to today: express them in start-date share terms
                ser.rescale(self.store.split_factor(sym, cfg.start))
            self.series[sym] = ser
            self.dataset_versions[sym] = self.store.dataset_version(sym)
        self.modeled = cfg.pricing.mode == "modeled"
        for sym in opt_syms:
            days = self.store.option_days(sym)
            if not self.modeled and not any(cfg.start <= d <= cfg.end for d in days):
                have = f"stored chains cover {days[0]} to {days[-1]}" if days else "no option data is stored for this symbol"
                raise DataUnavailable(f"No historical option quotes for {sym} between {cfg.start} and {cfg.end} ({have}). Strata does not fabricate option prices: "
                                      "download option history in Data, run a Market Study on the underlying, or explicitly choose the Modeled mode.")
            self.iv_hist[sym] = IvHistory.load(self.store, sym)
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
        l.stale = (t - l.quote_ts) > self.ex.max_quote_age_seconds or l.quote_ts > t
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
            px = X.fill_price(float(ch.bid[i]), float(ch.ask[i]), sign, ex)
            tb = bool(ch.trade_based[i])
            if tb:  # trade print, no quote: the print plus an explicit slippage assumption
                px = max(0.0, px * (1 + sign * ex.trade_bar_slippage_pct / 100.0))
                trade_based = True
            cash -= q * mult * px
            slip += abs(px - float(ch.mid[i])) * abs(q) * mult
            comm += X.option_commission(q, ex)
            g = (float(ch.delta[i]), float(ch.gamma[i]), float(ch.theta[i]), float(ch.vega[i]))
            ls = LegState(s.leg.id, "option", bool(ch.is_call[i]), float(ch.strike[i]), int(ch.exp[i]), int(ch.key[i]),
                          sign * s.leg.quantity, mult, s.leg.group, qty=q, entry_price=px, entry_bid=float(ch.bid[i]),
                          entry_ask=float(ch.ask[i]), entry_mid=float(ch.mid[i]), entry_iv=float(ch.iv[i]), entry_greeks=g)
            ls.bid, ls.ask, ls.mid, ls.iv, ls.greeks, ls.quote_ts = ls.entry_bid, ls.entry_ask, ls.entry_mid, ls.entry_iv, g, int(ch.quote_ts[i])
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
            if getattr(l, "trade_based", False):
                px = max(0.0, px * (1 - int(np.sign(q)) * self.ex.trade_bar_slippage_pct / 100.0))
            flow = q * l.multiplier * px
            comm = X.option_commission(q, self.ex)
            slip = abs(px - l.mid) * abs(q) * l.multiplier
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

    def close_position(self, pos: Position, reason: str, ctx: Context, leg_idx: list[int] | None = None, explain: list | None = None, ignore_liquidity: bool = False) -> bool:
        """Close the whole position (or a subset of legs) at the current snapshot. Returns
        True when everything requested was filled; otherwise the remainder is retried at
        the next snapshot."""
        t = ctx.t
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
        open_units = min((abs(pos.legs[i].qty) // max(abs(pos.legs[i].ratio), 1) for i in targets), default=0)
        n_units = int(min([open_units] + caps)) if caps else open_units
        if n_units <= 0 and open_units > 0:
            pos.pending_exit = reason if leg_idx is None else pos.pending_exit
            self.diag.deferred_exits += 1
            pos.flags.add("deferred_exit")
            self.log(t, "exit_deferred", f"{reason}: no displayed size to close; retrying", pos)
            return False
        for i in targets:
            l = pos.legs[i]
            n = abs(l.qty) if n_units >= open_units else abs(l.ratio) * n_units
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
        pos.shares += shares

    def _settles_am(self, sym: str, exp: date) -> bool:
        st = self.metas[sym].settlement_time
        if st == "AM":
            return True
        if st == "AM_MONTHLY":
            return exp.weekday() == 4 and 15 <= exp.day <= 21
        return False

    def settle_expiring(self, d: date, am: bool):
        dn = day_num(d)
        close_t = int(cal.session_bounds(d)[1].timestamp())
        for pos in list(self.account.positions):
            meta = self.metas[pos.symbol]
            ser = self.series[pos.symbol]
            legs = [l for l in pos.legs if l.is_open and l.kind == "option" and l.exp == dn and self._settles_am(pos.symbol, d) == am]
            if not legs:
                continue
            S = ser.session_open(d) if am else ser.session_close(d)
            if not math.isfinite(S):
                self.diag.warnings.append(f"{d}: no settlement price for {pos.symbol}; legs settled at last stored price")
                S = pos.last_spot
            for l in legs:
                intr = max(S - l.strike, 0.0) if l.is_call else max(l.strike - S, 0.0)
                q = l.qty
                l.exit_bid = l.exit_ask = l.exit_mid = intr
                l.exit_ts = close_t
                l.closed_qty += abs(q)
                l.qty = 0
                if intr < 0.01:
                    l.exit_price, l.exit_reason = 0.0, "expired_worthless"
                    continue
                if meta.settlement == "cash" or self.ex.physical_settlement == "intrinsic":
                    flow = q * l.multiplier * intr
                    self.account.cash += flow
                    pos.exit_cash += flow
                    l.exit_price, l.exit_reason = intr, "cash_settled"
                    l.exit_cash += flow
                    if meta.settlement != "cash":
                        pos.flags.add("settled_at_intrinsic")
                else:
                    shares = q * l.multiplier * (1 if l.is_call else -1)
                    self._deliver(pos, shares, l.strike)
                    l.exit_price, l.exit_reason = intr, ("exercised" if q > 0 else "assigned")
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
            self._update_position_values(pos)
            if not pos.is_open:
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
        max_dte = max((max(l.dte, l.dte_max or 0) for c in self.cfg.strategies if c.symbol == sym for l in c.legs if l.kind == "option"), default=7)
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
        last_t = 0
        for di, d in enumerate(days):
            if self.should_cancel and self.should_cancel():
                raise BacktestCancelled()
            ctx.today, ctx.today_num = d, day_num(d)
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
            self.chains = {}
            for sym in self.opt_syms:
                if self.modeled:
                    self.chains[sym] = self._modeled_chain(sym, d, sorted(sig_ticks))
                    if self.chains[sym] is None:
                        self.diag.exclude(d, f"{sym}: no underlying prices to model option premiums from")
                    continue
                df = self.store.read_option_day(sym, d)
                if df is None or df.is_empty():
                    self.chains[sym] = None
                    self.diag.missing_sessions.setdefault(sym, []).append(d)
                else:
                    self.chains[sym] = DayChain(df, self.cfg.capital.risk_free_rate)
            self.settle_expiring(d, am=True)
            clock = sorted({int(x) for ch in self.chains.values() if ch is not None for x in ch.snap_ts if t_open <= x <= t_close} | sig_ticks)
            diffs = np.diff(clock) if len(clock) > 1 else np.array([60])
            ctx.tick_minutes = max(1, int(np.median(diffs) // 60))
            ctx.day_minutes = {570 + (t - t_open) // 60 for t in clock}
            self._day_max_bp = self.account.bp_used()
            self._day_max_pos = len(self.account.positions)
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
        # close anything still open at the last snapshot, at the last observed quotes
        for pos in list(self.account.positions):
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
        return self._result(_time.time() - t0)

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

    def _resolution_summary(self) -> dict:
        out = {}
        for sym in self.opt_syms:
            ser = self.series.get(sym)
            if ser is None:
                continue
            a = int(np.searchsorted(ser.days, day_num(self.cfg.start)))
            b = int(np.searchsorted(ser.days, day_num(self.cfg.end), side="right"))
            vals, counts = np.unique(ser.d_res[a:b], return_counts=True)
            m = self.metas[sym]
            src = (m.extra or {}).get("sources") or {}
            opt = next((v for k, v in src.items() if k.startswith("options:")), None)
            out[sym] = {"underlying_sessions_by_resolution": {_res_label(int(v)): int(c) for v, c in zip(vals, counts)},
                        "options": "modeled (theoretical)" if self.modeled else (f"{opt['resolution']} {opt.get('quote_type', '')} via {opt['provider']}" if opt else (f"{m.option_interval_minutes}m quotes" if m.option_interval_minutes else "stored chains")),
                        "trade_based": bool(getattr(self, "_saw_trade_based", False))}
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
            "pricing_assumptions": (__import__("strata.data.modeled", fromlist=["x"]).assumptions_text(self.cfg.pricing) + [f"Volatility input used: {getattr(self, 'model_vol_note', 'n/a')}."]) if self.modeled else [],
            "data_resolution": self._resolution_summary(),
            "runtime_seconds": round(runtime, 3),
            "sessions": len(self.daily),
            "symbols": sorted(self.series.keys()),
            "benchmark": self.bench,
            "execution": self.cfg.execution.model_dump(),
            "capital": self.cfg.capital.model_dump(),
            "data_quality": {
                "missing_sessions": sum(len(v) for v in self.diag.missing_sessions.values()),
                "stale_marks": self.diag.stale_marks,
                "missing_marks": self.diag.missing_marks,
                "computed_greeks": any(ch is not None and ch.greeks_computed for ch in self.chains.values()),
            },
        }
        return BacktestResult(self.cfg, meta, trades, daily, paths, leg_paths, intraday, self.diag.to_dict(), self.events_log)


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
            "entry_price": _num(l.entry_price), "entry_bid": _num(l.entry_bid), "entry_ask": _num(l.entry_ask), "entry_mid": _num(l.entry_mid),
            "entry_iv": _num(l.entry_iv), "entry_delta": _num(l.entry_greeks[0]), "entry_gamma": _num(l.entry_greeks[1]),
            "entry_theta": _num(l.entry_greeks[2]), "entry_vega": _num(l.entry_greeks[3]),
            "exit_price": _num(l.exit_price), "exit_bid": _num(l.exit_bid), "exit_ask": _num(l.exit_ask), "exit_mid": _num(l.exit_mid),
            "exit_delta": _num(l.exit_greeks[0]), "exit_gamma": _num(l.exit_greeks[1]), "exit_theta": _num(l.exit_greeks[2]), "exit_vega": _num(l.exit_greeks[3]),
            "exit_ts": l.exit_ts or None, "exit_reason": l.exit_reason,
            "pnl": _num(np.sign(l.ratio) * l.closed_qty * l.multiplier * (l.exit_price - l.entry_price)) if l.closed_qty else None,
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
