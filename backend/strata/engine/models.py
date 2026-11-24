"""Declarative configuration schemas shared by the engine, API and UI."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class StrikeSel(BaseModel):
    method: Literal["delta", "premium", "pct_otm", "atm_offset", "fixed", "leg_offset"] = "delta"
    value: float = 0.30
    unit: Literal["points", "strikes", "pct"] = "points"
    ref_leg: Optional[str] = None


class LegSpec(BaseModel):
    id: str
    side: Literal["buy", "sell"]
    kind: Literal["option", "stock"] = "option"
    right: Literal["C", "P"] = "P"
    quantity: int = Field(1, ge=1)
    dte: int = Field(0, ge=0)
    dte_min: Optional[int] = None
    dte_max: Optional[int] = None
    dte_mode: Literal["nearest", "at_least", "at_most"] = "nearest"
    strike: StrikeSel = StrikeSel()
    iv_min: Optional[float] = None  # percent, e.g. 12 = 12%
    iv_max: Optional[float] = None
    max_spread_abs: Optional[float] = None  # $ per share
    max_spread_pct: Optional[float] = None  # % of mid
    min_bid: float = 0.0
    min_volume: Optional[int] = None
    min_open_interest: Optional[int] = None
    group: Optional[str] = None

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1


class CondNode(BaseModel):
    id: str = ""
    kind: Literal["group", "cond"] = "group"
    op: Literal["and", "or"] = "and"
    negate: bool = False
    enabled: bool = True
    children: list["CondNode"] = Field(default_factory=list)
    type: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class EntryRules(BaseModel):
    conditions: CondNode = CondNode()
    max_entries_per_day: int = Field(1, ge=1)
    cooldown_minutes: int = Field(0, ge=0)
    max_open_positions: int = Field(1, ge=1)
    reentry_after: list[Literal["any", "target", "stop", "time", "none"]] = Field(default_factory=lambda: ["any"])
    min_credit: Optional[float] = None  # per unit, $ per share
    max_debit: Optional[float] = None


class Threshold(BaseModel):
    enabled: bool = False
    mode: Literal["pct", "dollars"] = "pct"
    value: float = 50.0


class TrailingStop(BaseModel):
    enabled: bool = False
    mode: Literal["pct", "dollars"] = "pct"
    activate: float = 25.0
    trail: float = 15.0


class TimeExit(BaseModel):
    enabled: bool = False
    time: str = "15:45"
    dte_lte: Optional[int] = 0  # only on sessions where the front leg DTE <= this (None = every day)


class DteExit(BaseModel):
    enabled: bool = False
    dte: int = 1
    time: str = "09:45"


class MaxHold(BaseModel):
    enabled: bool = False
    minutes: Optional[int] = None
    days: Optional[int] = None


class UnderlyingExit(BaseModel):
    enabled: bool = False
    move_against_pct: Optional[float] = None
    move_for_pct: Optional[float] = None
    strike_breach: bool = False
    breach_buffer_pct: float = 0.0


class GreekExit(BaseModel):
    enabled: bool = False
    max_abs_delta: Optional[float] = None  # share-equivalents per unit
    short_leg_delta: Optional[float] = None  # abs delta of any short leg


class LegRule(BaseModel):
    target: str  # leg id or group name
    stop_pct: Optional[float] = None  # short leg: close when price >= entry*(1+x/100); long: <= entry*(1-x/100)
    target_pct: Optional[float] = None


class RollRule(BaseModel):
    enabled: bool = False
    trigger: Literal["dte", "short_delta", "loss_pct", "profit_pct"] = "dte"
    value: float = 1.0
    max_rolls: int = 3


class ExitRules(BaseModel):
    profit_target: Threshold = Threshold(enabled=True, mode="pct", value=50)
    stop_loss: Threshold = Threshold(enabled=True, mode="pct", value=200)
    trailing_stop: TrailingStop = TrailingStop()
    time_exit: TimeExit = TimeExit()
    dte_exit: DteExit = DteExit()
    max_hold: MaxHold = MaxHold()
    underlying: UnderlyingExit = UnderlyingExit()
    greeks: GreekExit = GreekExit()
    leg_rules: list[LegRule] = Field(default_factory=list)
    custom: Optional[CondNode] = None
    roll: RollRule = RollRule()
    trigger_price: Literal["close_price", "mid"] = "close_price"


class Sizing(BaseModel):
    method: Literal["fixed", "risk_pct", "bp_pct"] = "fixed"
    contracts: int = Field(1, ge=1)
    pct: float = 2.0
    max_contracts: int = 100


class RegimeSettings(BaseModel):
    vix_low: float = 15.0
    vix_high: float = 25.0
    trend_fast: int = 50
    trend_slow: int = 200


class StrategyConfig(BaseModel):
    name: str = "Untitled strategy"
    symbol: str = "DEMO"
    legs: list[LegSpec]
    entry: EntryRules = EntryRules()
    exits: ExitRules = ExitRules()
    sizing: Sizing = Sizing()
    allocation: float = Field(1.0, gt=0, le=1.0)
    vol_symbol: Optional[str] = None
    regime: RegimeSettings = RegimeSettings()

    @model_validator(mode="after")
    def _check_legs(self):
        if not self.legs:
            raise ValueError("strategy needs at least one leg")
        ids = [l.id for l in self.legs]
        if len(set(ids)) != len(ids):
            raise ValueError("leg ids must be unique")
        for l in self.legs:
            if l.strike.method == "leg_offset" and l.strike.ref_leg not in ids:
                raise ValueError(f"leg {l.id}: ref_leg {l.strike.ref_leg!r} not found")
        return self


class Execution(BaseModel):
    fill_model: Literal["conservative", "midpoint", "custom"] = "conservative"
    custom_fraction: float = Field(0.5, ge=0.0, le=1.0)
    slippage_per_contract: float = 0.0  # $ per share added against you on every option fill
    commission_per_contract: float = 0.65
    fee_per_contract: float = 0.0
    commission_per_order: float = 0.0
    stock_commission_per_share: float = 0.0
    stock_slippage_bps: float = 2.0
    exercise_fee: float = 0.0
    participation_rate: float = Field(1.0, gt=0, le=1.0)
    max_volume_pct: Optional[float] = None
    allow_partial_fills: bool = True
    max_quote_age_seconds: int = 120
    early_assignment: bool = True
    assignment_extrinsic_threshold: float = 0.05
    physical_settlement: Literal["close_next_open", "intrinsic"] = "close_next_open"


class Capital(BaseModel):
    initial_capital: float = Field(100_000.0, gt=0)
    margin_model: Literal["regt", "cash"] = "regt"
    max_bp_usage_pct: float = Field(100.0, gt=0, le=100.0)
    max_concurrent_positions: int = Field(50, ge=1)
    risk_free_rate: float = 0.0  # used for Sharpe / Sortino excess returns
    cash_interest_rate: float = 0.0  # annual rate credited on idle cash (cash minus margin held)
    cash_benchmark_rate: float = 0.04  # rate of the "cash" benchmark curve
    liquidate_on_margin_call: bool = True


class BacktestConfig(BaseModel):
    strategies: list[StrategyConfig]
    start: date
    end: date
    capital: Capital = Capital()
    execution: Execution = Execution()
    benchmark: Optional[str] = None
    allow_synthetic: bool = True
    label: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.end < self.start:
            raise ValueError("end date is before start date")
        if not self.strategies:
            raise ValueError("at least one strategy is required")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]
