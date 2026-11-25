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
    dte_mode: Literal["nearest", "at_least", "at_most", "exact"] = "nearest"  # dte 0 with "nearest" is always exact
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


class SeqStage(BaseModel):
    """One step of an event sequence. Stages must happen in order, each on a later
    evaluation tick than the one before; the entry signal fires when the last one happens."""
    name: str
    when: str  # rule expression; may read earlier stages: stage_price("a"), low_since("a"), minutes_since("a") ...
    within_minutes: Optional[int] = Field(None, ge=1)  # must happen within N minutes of the previous stage, else the sequence starts over
    capture: dict[str, str] = Field(default_factory=dict)  # reference prices recorded when the stage happens: name -> expression, read back with ref("name")
    text: str = ""


class AddRule(BaseModel):
    """Scale into an open position: when `when` holds for the first lot, open another lot."""
    when: str  # rule expression evaluated against the FIRST lot (entry.*, position.*, group.*, stage state)
    quantity: int = Field(1, ge=1)  # units added each time
    max_adds: int = Field(1, ge=1)
    contract: Literal["same", "reselect"] = "same"  # the same contracts as the first lot, or select again at the add
    text: str = ""


class EntryRules(BaseModel):
    conditions: CondNode = CondNode()
    sequence: list[SeqStage] = Field(default_factory=list)
    adds: list[AddRule] = Field(default_factory=list)
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
    symbol: str = "SPY"
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
        from strata.engine import expr as X
        names = [s.name for s in self.entry.sequence]
        if len(set(names)) != len(names) or any(not n.strip() for n in names):
            raise ValueError("sequence stage names must be unique and non-empty")
        refs = {k for s in self.entry.sequence for k in s.capture}
        for i, st in enumerate(self.entry.sequence):
            for src in [st.when] + list(st.capture.values()):
                errs = X.validate(src)
                if errs:
                    raise ValueError(f"sequence stage {st.name!r}: {errs[0]}")
                if X.uses_position(src):
                    raise ValueError(f"sequence stage {st.name!r}: entry.*, position.* and group.* only exist once a position is open")
                later = X.stages_used(src) - set(names[:i])
                if later:
                    raise ValueError(f"sequence stage {st.name!r} reads stage {sorted(later)[0]!r}, which is not an earlier stage")
        for a in self.entry.adds:
            errs = X.validate(a.when)
            if errs:
                raise ValueError(f"add rule: {errs[0]}")
        for src in self.expressions():
            try:
                bad = X.stages_used(src) - set(names)
                bad_ref = X.refs_used(src) - refs
            except X.ExprError:
                continue  # reported by the rule validators
            if bad:
                raise ValueError(f"rule {src!r} reads stage {sorted(bad)[0]!r}, but the entry sequence has no such stage")
            if bad_ref:
                raise ValueError(f"rule {src!r} reads ref({sorted(bad_ref)[0]!r}), but no stage captures it")
        return self

    def expressions(self) -> list[str]:
        """Every rule expression of the strategy (entry tree, sequence, adds, custom exit)."""
        out: list[str] = []

        def walk(n: Optional[CondNode]):
            if n is None or not n.enabled:
                return
            if n.kind == "cond" and n.type == "expr":
                out.append(str((n.params or {}).get("expr", "")))
            for c in n.children:
                walk(c)
        walk(self.entry.conditions)
        walk(self.exits.custom)
        for st in self.entry.sequence:
            out += [st.when] + list(st.capture.values())
        out += [a.when for a in self.entry.adds]
        return out

    def symbols(self) -> set[str]:
        """Other symbols any rule reads (cross-symbol rules)."""
        from strata.engine import expr as X
        from strata.engine.conditions import extra_symbols
        out = set(extra_symbols(self.entry.conditions)) | set(extra_symbols(self.exits.custom))
        for src in self.expressions():
            try:
                out |= X.symbols_used(src)
            except X.ExprError:
                pass
        return out - {self.symbol}


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
    # --- clocks: signals are evaluated on the underlying's bars; orders fill at the next
    # eligible option quote. Signal time and fill time are recorded separately.
    signal_interval_minutes: int = Field(5, ge=1)
    latency_seconds: float = Field(0.0, ge=0)
    max_fill_delay_minutes: float = Field(5.0, ge=0)
    trade_bar_slippage_pct: float = Field(0.0, ge=0)  # only for trade-print (no-quote) option data: % of the print, against you
    max_trade_age_seconds: int = Field(300, ge=60)  # trade-bar pricing: a print older than this is stale and is not used for a fill
    # --- execution stress assumptions (0 = off). Explicit, never inferred from the data:
    extra_slippage_pct: float = Field(0.0, ge=0)  # every option fill this % worse than the quote/print it was priced from
    multi_leg_extra_pct: float = Field(0.0, ge=0)  # additional % against you on each leg of a position with two or more option legs (legs do not fill together)
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


class Pricing(BaseModel):
    """historical = observed option bid/ask QUOTES. trades = observed option TRADE prints
    (no bid/ask exists; fills use the last print that is still recent, plus an explicit
    slippage assumption). modeled = real underlying prices with THEORETICAL option premiums
    from the assumptions below (results are estimates). The three are never mixed."""
    mode: Literal["historical", "trades", "modeled"] = "historical"
    model: Literal["american_baw", "black_scholes"] = "american_baw"
    iv_source: Literal["vix", "realized", "fixed"] = "vix"
    iv_fixed: float = 0.20
    iv_scale: float = 1.0
    skew_per_10pct: float = 0.03  # vol added per 10% of downside moneyness (and removed on the upside)
    spread_pct: float = 1.0  # assumed bid-ask spread, % of premium
    min_spread: float = 0.01
    rate: float = 0.04
    dividend_yield: float = 0.013
    snapshot_minutes: int = Field(5, ge=1)
    strike_step: Optional[float] = None


class DataSelection(BaseModel):
    """Which stored datasets a run reads. Left empty, the engine resolves one option
    dataset per symbol for the pricing mode and reports its choice in the result."""
    options_provider: Optional[str] = None  # e.g. "github_eod", "alpaca", "yahoo_options"


class BacktestConfig(BaseModel):
    strategies: list[StrategyConfig]
    start: date
    end: date
    capital: Capital = Capital()
    execution: Execution = Execution()
    pricing: Pricing = Pricing()
    data: DataSelection = DataSelection()
    benchmark: Optional[str] = None
    allow_synthetic: bool = False
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
