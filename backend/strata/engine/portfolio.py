"""Positions, legs and account-level accounting (cash, marks, buying power)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from strata.engine.margin import MLeg, requirement


@dataclass
class LegState:
    leg_id: str
    kind: str  # option | stock
    is_call: bool
    strike: float
    exp: int  # day number (-1 stock)
    key: int
    ratio: int  # signed contracts per unit
    multiplier: float
    group: str | None
    qty: int = 0  # signed open contracts (or shares for stock)
    exp_close: int = 2**62  # epoch seconds of the expiration close (options)
    entry_price: float = 0.0
    entry_bid: float = math.nan
    entry_ask: float = math.nan
    entry_mid: float = math.nan
    entry_iv: float = math.nan
    entry_greeks: tuple = (math.nan,) * 4
    exit_price: float = math.nan
    exit_bid: float = math.nan
    exit_ask: float = math.nan
    exit_mid: float = math.nan
    exit_greeks: tuple = (math.nan,) * 4
    exit_ts: int = 0
    exit_reason: str = ""
    closed_qty: int = 0
    exit_cash: float = 0.0
    # latest observation
    bid: float = math.nan
    ask: float = math.nan
    mid: float = math.nan
    iv: float = math.nan
    greeks: tuple = (math.nan,) * 4
    quote_ts: int = 0
    stale: bool = False
    bid_size: int = -1
    ask_size: int = -1
    volume: int = -1

    @property
    def is_open(self) -> bool:
        return self.qty != 0

    def label(self) -> str:
        if self.kind == "stock":
            return "STOCK"
        from strata.engine.market import from_day_num
        return f"{from_day_num(self.exp).isoformat()} {self.strike:g}{'C' if self.is_call else 'P'}"


@dataclass
class Position:
    id: int
    strategy_idx: int
    strategy: str
    symbol: str
    units: int
    legs: list[LegState]
    entry_ts: int
    entry_day: int
    entry_spot: float
    entry_cash: float = 0.0  # net option/stock cash flow at entry (+ credit)
    exit_cash: float = 0.0  # cash from closes/settlements
    commissions: float = 0.0
    slippage: float = 0.0  # cost vs mid, $
    max_risk: float = 0.0  # buying power reduction at entry
    requirement: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    peak_pnl: float = 0.0
    mark_value: float = 0.0
    close_value: float = 0.0
    shares: float = 0.0  # stock delivered by exercise/assignment awaiting liquidation
    share_cost: float = 0.0
    pending_stock_close: bool = False
    exit_ts: int = 0
    exit_reason: str = ""
    entry_explain: list = field(default_factory=list)
    exit_explain: list = field(default_factory=list)
    flags: set = field(default_factory=set)
    roll_group: int = 0
    roll_count: int = 0
    tag: str = ""
    entry_vix: float = math.nan
    entry_iv: float = math.nan
    entry_regime: str = ""
    last_spot: float = math.nan
    pending_exit: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return any(l.is_open for l in self.legs) or abs(self.shares) > 0

    @property
    def basis(self) -> float:
        b = abs(self.entry_cash)
        return b if b > 1e-9 else max(self.max_risk, 1.0)

    @property
    def is_credit(self) -> bool:
        return self.entry_cash > 0

    def pnl_mark(self) -> float:
        return self.entry_cash + self.exit_cash + self.mark_value - self.commissions

    def pnl_close(self) -> float:
        return self.entry_cash + self.exit_cash + self.close_value - self.commissions

    def gross_pnl(self, use_mid: bool) -> float:
        return self.entry_cash + self.exit_cash + (self.mark_value if use_mid else self.close_value)

    def greeks(self) -> tuple[float, float, float, float]:
        d = g = th = v = 0.0
        for l in self.legs:
            if not l.is_open:
                continue
            if l.kind == "stock":
                d += l.qty
                continue
            gd, gg, gt, gv = l.greeks
            q = l.qty * l.multiplier
            d += q * (gd if math.isfinite(gd) else 0)
            g += q * (gg if math.isfinite(gg) else 0)
            th += q * (gt if math.isfinite(gt) else 0)
            v += q * (gv if math.isfinite(gv) else 0)
        d += self.shares
        return d, g, th, v

    def front_dte(self, today: int) -> int:
        exps = [l.exp for l in self.legs if l.is_open and l.kind == "option"]
        return (min(exps) - today) if exps else 0

    def margin_legs(self) -> list[MLeg]:
        return [MLeg(l.is_call, l.strike, l.qty, l.multiplier, l.mid if math.isfinite(l.mid) else 0.0) for l in self.legs if l.is_open and l.kind == "option"]

    def stock_qty(self) -> float:
        return sum(l.qty for l in self.legs if l.is_open and l.kind == "stock") + self.shares


class Account:
    def __init__(self, initial: float, margin_model: str):
        self.initial = initial
        self.cash = initial
        self.margin_model = margin_model
        self.positions: list[Position] = []  # open
        self.closed: list[Position] = []
        self.commissions = 0.0
        self.slippage = 0.0

    def equity(self) -> float:
        return self.cash + sum(p.mark_value for p in self.positions)

    def liquidation_equity(self) -> float:
        return self.cash + sum(p.close_value for p in self.positions)

    def position_requirement(self, p: Position, spot: float) -> float:
        return requirement(p.margin_legs(), p.stock_qty(), spot, self.margin_model)

    def total_requirement(self) -> float:
        return sum(p.requirement for p in self.positions)

    def bp_used(self) -> float:
        """Buying power currently tied up: requirement minus premium credited, plus debits paid."""
        used = 0.0
        for p in self.positions:
            open_cash = p.entry_cash + p.exit_cash
            used += max(0.0, p.requirement - open_cash) if math.isfinite(p.requirement) else math.inf
        return used

    def available_bp(self) -> float:
        return self.cash - self.total_requirement()
