"""Strategy drafts: the structured interpretation of a plain-English request.

An interpreter (the offline rule parser or an LLM) only fills in a `Draft`. Everything
after that is deterministic Python: the draft is compiled into the engine's own
StrategyConfig / StudyConfig, every default the user did not state is recorded as an
explicit assumption, missing details are listed, and data coverage decides which research
mode can answer the question. No financial number is ever produced by a language model.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine import expr as X
from strata.engine.models import BacktestConfig, StrategyConfig
from strata.research.study import StudyConfig

BULLISH = {"long_call", "short_put", "bull_call_spread", "bull_put_spread", "covered_call"}
BEARISH = {"long_put", "short_call", "bear_put_spread", "bear_call_spread"}
STRUCTURES = {
    "long_call": "Long call", "long_put": "Long put", "short_put": "Short put", "short_call": "Short call",
    "bull_call_spread": "Bull call (debit) spread", "bear_put_spread": "Bear put (debit) spread", "bull_put_spread": "Bull put (credit) spread",
    "bear_call_spread": "Bear call (credit) spread", "iron_condor": "Iron condor", "iron_butterfly": "Iron butterfly", "long_straddle": "Long straddle",
    "short_straddle": "Short straddle", "long_strangle": "Long strangle", "short_strangle": "Short strangle", "covered_call": "Covered call",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


class DraftCondition(BaseModel):
    expr: str
    text: str = ""
    source: Literal["stated", "assumed"] = "stated"


class DraftStrike(BaseModel):
    # strikes_otm (V10): that many listed strikes out of the money ("the call two strikes OTM"), counted from the at-the-money one
    method: Literal["atm", "delta", "pct_otm", "strikes_otm"] = "atm"
    value: float = 0.0


class DraftLevel(BaseModel):
    kind: Literal["pct", "abs", "expr"] = "pct"
    value: float = 0.0
    expr: Optional[str] = None


class DraftStage(BaseModel):
    """One step of an event sequence ("wait until X, then enter when Y")."""
    name: str
    expr: str
    text: str = ""
    within_minutes: Optional[int] = None


class DraftAdd(BaseModel):
    """Scale into the position: open another lot when `expr` holds for the first lot."""
    expr: str
    text: str = ""
    quantity: int = 1
    max_adds: int = 1


class DraftPartial(BaseModel):
    """Close part of the position when a level is reached; the rest stays open under the other exit rules."""
    premium_pct: Optional[float] = None  # the position is up this % of the premium paid (or received)
    underlying_pct: Optional[float] = None  # the underlying has moved this % in the position's favour
    fraction: float = 0.5  # the share of the contracts the position was opened with
    text: str = ""


class DraftCosts(BaseModel):
    """Execution-cost assumptions the user asked for (None = engine defaults)."""
    preset: Optional[Literal["conservative", "none"]] = None
    commission_per_contract: Optional[float] = None
    slippage_per_contract: Optional[float] = None  # $ per share against you on every option fill
    slippage_pct: Optional[float] = None  # % of the observed trade print against you (trade-price execution only)


class DraftSizing(BaseModel):
    """How much each trade buys, in the trader's terms (V9). Compiled onto the existing portfolio engine's sizing.

    contracts     a fixed number of contracts (the default, from `quantity`)
    dollars       spend about this much premium per trade: floor(value / (premium x multiplier)) contracts; a single contract
                  costing more than the budget is not bought
    risk_dollars  lose at most this much if the stop is hit (with a premium stop: premium x stop%), or the whole premium
                  without a stop: contracts = floor(value / risk of one contract)
    risk_pct      the same as a share of the account's equity at the time of the trade
    equity_pct    spend this share of the account's equity on premium
    capital       the starting account;  reinvest: equity (not the starting capital) sizes later trades
    daily_loss_stop  no new entries that session once its realised loss reaches this many dollars
    """
    mode: Literal["contracts", "dollars", "risk_dollars", "risk_pct", "equity_pct"] = "contracts"
    value: float = 1.0
    capital: Optional[float] = None
    reinvest: bool = False
    daily_loss_stop: Optional[float] = None
    text: str = ""


class DraftStudy(BaseModel):
    target: Optional[DraftLevel] = None
    stop: Optional[DraftLevel] = None
    reference: Literal["entry", "open", "prev_close"] = "entry"
    measure: Literal["touch", "close"] = "touch"  # touch = trades at the level; close = final price of the horizon is beyond it
    direction: Optional[Literal["up", "down"]] = None
    horizon_kind: Literal["session", "time", "sessions"] = "session"
    horizon_time: str = "12:00"
    horizon_sessions: int = 1


TICKER = re.compile(r"[A-Z^][A-Z0-9^=_-]{0,15}(?:\.[A-Z0-9]{1,6})?")  # a symbol is a ticker: it also names folders in the data store


class Draft(BaseModel):
    kind: Literal["options", "study"] = "options"
    title: str = ""
    symbol: str = "SPY"
    symbol_stated: bool = True

    @field_validator("symbol")
    @classmethod
    def _symbol_is_a_ticker(cls, v: str) -> str:
        v = str(v).strip().upper()
        if not TICKER.fullmatch(v):
            raise ValueError(f"{v[:40]!r} is not a ticker symbol")
        return v

    entry_time: Optional[str] = None  # New York "HH:MM", or "open" / "close"; None = whenever the conditions hold
    entry_time_stated: Optional[str] = None  # the user's words, e.g. "6:30 AM Pacific"
    tz_assumed: Optional[str] = None  # set when a clock time carried no timezone and one had to be assumed
    weekdays: Optional[list[str]] = None
    conditions: list[DraftCondition] = Field(default_factory=list)
    sequence: list[DraftStage] = Field(default_factory=list)  # events that must happen in order before the entry
    adds: list[DraftAdd] = Field(default_factory=list)  # scale-in rules
    exit_conditions: list[DraftCondition] = Field(default_factory=list)  # close the position when any of these holds
    unsupported: list[str] = Field(default_factory=list)  # parts of the request that could not be represented (never silently dropped)
    questions: list[dict] = Field(default_factory=list)  # open clarifying questions: {id, text, options: [{label, patch}], default}
    # V10: why this reading waits for a go-ahead, in plain words (intent.check): something said that it does not hold, a
    # number held as another kind of thing than it was said as, a size the message does not contain
    review: list[str] = Field(default_factory=list)
    costs: Optional[DraftCosts] = None
    provider: Optional[str] = None  # a specific stored option dataset / provider, if the user named one
    carried: dict[str, str] = Field(default_factory=dict)  # field -> why it was carried over or kept without being stated in this message
    # ---- option position
    structure: Optional[str] = None
    dte: Optional[int] = None
    strike: Optional[DraftStrike] = None
    width: Optional[float] = None
    quantity: int = 1
    # ---- exits
    underlying_target_pct: Optional[float] = None  # favourable move of the underlying from entry, %
    underlying_stop_pct: Optional[float] = None  # adverse move, %
    underlying_target_abs: Optional[float] = None  # favourable move of the underlying in price points ($), not percent
    underlying_stop_abs: Optional[float] = None
    premium_target_pct: Optional[float] = None  # gain as % of premium
    premium_stop_pct: Optional[float] = None  # loss as % of premium
    exit_time: Optional[str] = None  # New York "HH:MM" or "close"
    exit_time_stated: Optional[str] = None
    hold_sessions: Optional[int] = None
    hold_minutes: Optional[int] = None  # close this many minutes after the entry (a clock exit relative to the fill, not to the session)
    exit_weekday: Optional[str] = None
    hold_to_expiration: bool = False
    max_entries: Optional[int] = None  # entries allowed per session (None: one)
    sizing: Optional[DraftSizing] = None  # V9: dollars / risk / share of the account per trade, a starting account, reinvesting, a daily loss stop
    partials: list[DraftPartial] = Field(default_factory=list)  # partial exits, in order
    # ---- market study
    study: Optional[DraftStudy] = None
    # ---- scope
    start: Optional[date] = None
    end: Optional[date] = None
    mode: Optional[Literal["historical", "trades", "study", "modeled"]] = None
    notes: list[str] = Field(default_factory=list)


def direction(d: Draft) -> int:
    return 1 if d.structure in BULLISH else -1 if d.structure in BEARISH else 0


def _cond_tree(d: Draft, for_study: bool = False) -> dict:
    kids = []
    if d.entry_time and not for_study:
        kids.append({"id": "t", "kind": "cond", "type": "entry_times", "params": {"times": d.entry_time}})
    if d.weekdays:
        kids.append({"id": "w", "kind": "cond", "type": "day_of_week", "params": {"days": d.weekdays}})
    for i, c in enumerate(d.conditions):
        kids.append({"id": f"c{i}", "kind": "cond", "type": "expr", "params": {"expr": c.expr, "label": c.text}})
    return {"id": "root", "kind": "group", "op": "and", "children": kids}


def _leg(id, side, right, dte, method, value, ref=None, qty=1, group=None, kind="option"):
    return {"id": id, "side": side, "kind": kind, "right": right, "quantity": qty, "dte": dte, "dte_mode": "exact" if dte == 0 else "nearest",
            "strike": {"method": method, "value": value, "unit": "points", "ref_leg": ref}, **({"group": group} if group else {})}


def build_legs(structure: str, dte: int, strike: DraftStrike, width: float, qty: int) -> list[dict]:
    if strike.method == "strikes_otm" and structure not in ("iron_butterfly", "long_straddle", "short_straddle", "covered_call"):
        legs = build_legs(structure, dte, DraftStrike(method="atm"), width, qty)
        for l in legs:
            st = l["strike"]
            if st["method"] == "atm_offset":  # calls sit above the at-the-money strike when out of the money, puts below
                st.update(value=(1 if l["right"] == "C" else -1) * abs(float(strike.value)), unit="strikes")
        return legs
    m, v = {"atm": ("atm_offset", 0.0), "delta": ("delta", strike.value), "pct_otm": ("pct_otm", strike.value), "strikes_otm": ("atm_offset", 0.0)}[strike.method]
    L = lambda *a, **k: _leg(*a, **{"qty": qty, **k})
    if structure == "long_call":
        return [L("lc", "buy", "C", dte, m, v)]
    if structure == "long_put":
        return [L("lp", "buy", "P", dte, m, v)]
    if structure == "short_put":
        return [L("sp", "sell", "P", dte, m, v)]
    if structure == "short_call":
        return [L("sc", "sell", "C", dte, m, v)]
    if structure == "bull_call_spread":
        return [L("lc", "buy", "C", dte, m, v), L("sc", "sell", "C", dte, "leg_offset", width, "lc")]
    if structure == "bear_put_spread":
        return [L("lp", "buy", "P", dte, m, v), L("sp", "sell", "P", dte, "leg_offset", -width, "lp")]
    if structure == "bull_put_spread":
        return [L("sp", "sell", "P", dte, m, v), L("lp", "buy", "P", dte, "leg_offset", -width, "sp")]
    if structure == "bear_call_spread":
        return [L("sc", "sell", "C", dte, m, v), L("lc", "buy", "C", dte, "leg_offset", width, "sc")]
    if structure == "iron_condor":
        return [L("sp", "sell", "P", dte, m, v, group="put_side"), L("lp", "buy", "P", dte, "leg_offset", -width, "sp", group="put_side"),
                L("sc", "sell", "C", dte, m, v, group="call_side"), L("lc", "buy", "C", dte, "leg_offset", width, "sc", group="call_side")]
    if structure == "iron_butterfly":
        return [L("sp", "sell", "P", dte, "atm_offset", 0), L("sc", "sell", "C", dte, "atm_offset", 0), L("lp", "buy", "P", dte, "leg_offset", -width, "sp"), L("lc", "buy", "C", dte, "leg_offset", width, "sc")]
    if structure in ("long_straddle", "short_straddle"):
        s = "buy" if structure.startswith("long") else "sell"
        return [L("c", s, "C", dte, "atm_offset", 0), L("p", s, "P", dte, "atm_offset", 0)]
    if structure in ("long_strangle", "short_strangle"):
        s = "buy" if structure.startswith("long") else "sell"
        dm, dv = ("delta", v if m == "delta" else 0.2)
        return [L("c", s, "C", dte, dm, dv), L("p", s, "P", dte, dm, dv)]
    if structure == "covered_call":
        return [{"id": "stk", "side": "buy", "kind": "stock", "right": "C", "quantity": 100 * qty, "dte": 0, "strike": {"method": "fixed", "value": 0}}, L("sc", "sell", "C", dte, m if m == "delta" else "delta", v if m == "delta" else 0.3)]
    raise ValueError(f"unsupported structure {structure!r}")


def fmt_time(t: Optional[str], stated: Optional[str] = None) -> str:
    if not t:
        return "whenever the conditions hold"
    base = {"open": "the open (09:30 ET)", "close": "the close (16:00 ET, earlier on half days)"}.get(t, f"{t} ET")
    return f"{base} - you said \"{stated}\"" if stated and stated.lower() not in base.lower() else base


SESSION_SO_FAR = {"time", "minutes_since_open", "from_open_pct", "max_decline_from_open_pct", "max_rally_from_open_pct", "bounce_from_low_pct"}


def _during_session(expr: str) -> bool:
    try:
        u = X.names_used(expr)
    except Exception:  # noqa: BLE001
        return False
    return bool(u & SESSION_SO_FAR or {x.split(".", 1)[1] for x in u if "." in x} & SESSION_SO_FAR)


def compile_draft(d: Draft, store: MarketStore | None = None, spot: float | None = None) -> dict:
    """Draft -> engine configs + an itemised interpretation (stated / assumed / missing)."""
    store = store or MarketStore()
    items: list[dict] = []
    assumptions: list[dict] = []
    missing: list[str] = []
    errors: list[str] = []

    def add(label, value, status="stated", field=None):
        items.append({"label": label, "value": value, "status": status, "field": field})

    def assume(field, value, reason):
        assumptions.append({"field": field, "value": value, "reason": reason})

    for c in list(d.conditions) + list(d.exit_conditions):
        errors += X.validate(c.expr)
    for s_ in d.sequence:
        errors += [f"step '{s_.name}': {e}" for e in X.validate(s_.expr)]
    for a_ in d.adds:
        errors += [f"add rule: {e}" for e in X.validate(a_.expr)]
    add("Underlying", d.symbol, "stated" if d.symbol_stated else "assumed", "symbol")
    if not d.symbol_stated:
        assume("symbol", d.symbol, "no ticker mentioned, so the default symbol is used")
    if d.tz_assumed:
        assume("timezone", d.tz_assumed, f"a clock time was given without a timezone, so it was read as {d.tz_assumed} time (change the default under Settings)")
    if d.weekdays:
        add("Days", ", ".join(d.weekdays), field="weekdays")
    for c in d.conditions:
        add("Condition", f"{c.text or c.expr}   [{c.expr}]", c.source, "conditions")
    for i, s_ in enumerate(d.sequence):
        add(f"Step {i + 1} of {len(d.sequence)}", f"{s_.text or s_.name}   [{s_.expr}]" + (f" - within {s_.within_minutes} min of the previous step" if s_.within_minutes else ""), "stated", "sequence")
    if d.sequence:
        add("Order", "The steps must happen in this order, each later than the one before; the entry is when the last one happens. Reset every session.", "stated", "sequence")
    for u in d.unsupported:
        add("Not represented", u, "unsupported", "unsupported")

    # ------------------------------------------------------------ study config (always derivable)
    st = d.study or DraftStudy()
    dirn = direction(d)
    study_entry = d.entry_time or "open"
    if d.kind == "options":
        sgn = dirn or 1
        tgt = DraftLevel(kind="pct", value=d.underlying_target_pct * sgn) if d.underlying_target_pct else DraftLevel(kind="abs", value=d.underlying_target_abs * sgn) if d.underlying_target_abs else None
        stp = DraftLevel(kind="pct", value=-d.underlying_stop_pct * sgn) if d.underlying_stop_pct else DraftLevel(kind="abs", value=-d.underlying_stop_abs * sgn) if d.underlying_stop_abs else None
        hz = {"kind": "session"}
        if d.exit_time and d.exit_time not in ("close",):
            hz = {"kind": "time", "time": d.exit_time}
        elif d.hold_sessions:
            hz = {"kind": "sessions", "sessions": d.hold_sessions}
        elif d.exit_weekday and study_entry == "close":
            hz = {"kind": "sessions", "sessions": 4}
        st_cfg = {"target": tgt.model_dump() if tgt else None, "stop": stp.model_dump() if stp else None, "reference": "entry", "horizon": hz}
    else:
        hz = {"kind": st.horizon_kind, "time": st.horizon_time, "sessions": st.horizon_sessions}
        st_cfg = {"target": st.target.model_dump() if st.target else None, "stop": st.stop.model_dump() if st.stop else None, "reference": st.reference, "horizon": hz,
                  "measure": st.measure, "direction": st.direction}
    intraday_trigger = any(X.needs_intraday(c.expr) for c in d.conditions)
    if d.sequence or (d.kind == "study" and not d.entry_time and intraday_trigger):
        study_entry = "signal"  # the observation starts when the rule (or its event sequence) fires
    study = {"symbol": d.symbol, "start": d.start.isoformat() if d.start else None, "end": d.end.isoformat() if d.end else None, "condition": _cond_tree(d, for_study=True),
             "sequence": [{"name": s_.name, "when": s_.expr, "text": s_.text, "within_minutes": s_.within_minutes} for s_ in d.sequence],
             "entry_time": study_entry, "label": d.title, **st_cfg}
    try:
        StudyConfig.model_validate(study)
    except Exception as e:  # noqa: BLE001
        errors.append(f"study: {e}")

    strategy = None
    entry_time, dte = d.entry_time, d.dte
    if d.kind == "study":
        add("Question type", "Historical market study (underlying prices; no option premiums)")
        add("Observation starts", "when the rule fires (the first tick on which it holds)" if study_entry == "signal" else fmt_time(study_entry, d.entry_time_stated),
            "stated" if (d.entry_time or study_entry == "signal") else "assumed", "entry_time")
        if not d.entry_time and study_entry != "signal":
            assume("entry_time", "open", "no time given, so each session is observed from its open")
        lv = lambda x: None if x is None else (f"{x.value:+g}% from the {st.reference if st.reference != 'entry' else 'starting price'}" if x.kind == "pct" else f"{x.value:+g} $ from the starting price" if x.kind == "abs" else f"price level: {x.expr}")
        if st.target and st.measure == "close":
            side = "below" if (st.direction == "down" or (st.target.kind == "pct" and st.target.value < 0)) else "above"
            add("Outcome", (f"final price of the horizon is {side} the starting price" if st.target.kind == "pct" and st.target.value == 0 else f"final price of the horizon is at or {side} {lv(st.target)}") + " (a close, not an intraday touch)", field="study.target")
        elif st.target:
            add("Target", lv(st.target) + " - counted when price trades there at any point (a touch)", field="study.target")
        if st.stop:
            add("Stop", lv(st.stop), field="study.stop")
        add("Horizon", {"session": "until that session's close", "time": f"until {st.horizon_time} ET", "sessions": f"through the close {st.horizon_sessions} session(s) later"}[st.horizon_kind], field="study.horizon")
    else:
        if not d.structure:
            missing.append("structure")
            errors.append("no option position was described (calls, puts, a spread ...)")
        else:
            add("Position", f"{d.quantity if (d.sizing is None or d.sizing.mode == 'contracts') else ''}{' x ' if (d.sizing is None or d.sizing.mode == 'contracts') else ''}{STRUCTURES.get(d.structure, d.structure)}", field="structure")
            if d.sizing is not None:
                add("Size", sizing_text(d), "stated", "sizing")
        add("Entry", fmt_time(d.entry_time, d.entry_time_stated) + (" - once per eligible session" if d.entry_time else " - first signal each session"), "stated" if d.entry_time or d.conditions or d.sequence else "assumed", "entry_time")
        # a rule that reads the clock, or a move measured from the session's own open, can only become true during the
        # session: assuming "enter at the open" for it would produce a strategy that can never trade
        intraday_rule = any(X.needs_intraday(c.expr) or _during_session(c.expr) for c in d.conditions) or bool(d.sequence)
        same_day = (bool(d.exit_time) or bool(d.hold_minutes)) and not d.hold_sessions and not d.exit_weekday
        multi_day = bool(d.hold_sessions or d.exit_weekday or d.hold_to_expiration)
        short_premium = d.structure in ("short_put", "short_call", "bull_put_spread", "bear_call_spread", "iron_condor", "iron_butterfly", "short_strangle", "short_straddle", "covered_call")
        # ---- expiration
        dte = d.dte
        if dte is None:
            missing.append("expiration")
            if same_day:
                dte, why = 0, "the exit is on the same day as the entry, so the same-day (0DTE) expiration is used"
            elif multi_day:
                dte = max(7, (d.hold_sessions or 5) + 9)
                why = f"about {dte} days to expiration leaves room for the holding period"
            elif d.entry_time not in (None, "close") or intraday_rule:
                dte, why = 0, "an intraday entry with no holding period reads as a day trade, so the same-day (0DTE) expiration is used"
            else:
                dte = 30 if short_premium else 14
                why = "no expiration or holding period was given; a common " + ("30-day premium-selling" if short_premium else "two-week") + " expiration is used"
            assume("dte", dte, why)
            add("Expiration", f"{dte}DTE", "assumed", "dte")
        else:
            add("Expiration", f"{dte}DTE" + (" (exact - never a later expiry)" if dte == 0 else " (nearest listed)"), field="dte")
        entry_time = d.entry_time
        if not d.entry_time and not intraday_rule:
            from strata.data.catalog import profile
            opt = (profile(store, d.symbol) or {}).get("options") or {}
            if dte > 0 and opt.get("resolution") == "EOD":
                entry_time = "close"
                assume("entry_time", "close", f"no entry time given; entering at the close, which is when the stored {d.symbol} option quotes are observed (end-of-day data)")
            else:
                entry_time = "open"
                assume("entry_time", "open", "no entry time given; entering at the open")
            missing.append("entry time")
            for it in items:
                if it["field"] == "entry_time":
                    it["value"], it["status"] = fmt_time(entry_time) + " - once per eligible session", "assumed"
        # ---- strike
        strike = d.strike
        if strike is None:
            strike = DraftStrike(method="delta", value=0.16 if d.structure == "iron_condor" else 0.20) if short_premium else DraftStrike(method="atm")
            if d.structure not in ("iron_butterfly", "long_straddle", "short_straddle"):
                missing.append("strike")
                assume("strike", "ATM" if strike.method == "atm" else f"{strike.value:g} delta", "at-the-money is the neutral default for a long option" if strike.method == "atm" else "a typical short-premium distance")
                add("Strike", "At the money" if strike.method == "atm" else f"{strike.value:g} delta short strike", "assumed", "strike")
        else:
            add("Strike", {"atm": "At the money", "delta": f"{strike.value:g} delta", "pct_otm": f"{strike.value:g}% out of the money",
                           "strikes_otm": f"{strike.value:g} strike{'s' if strike.value != 1 else ''} out of the money"}[strike.method], field="strike")
        width = d.width
        if d.structure and ("spread" in d.structure or d.structure in ("iron_condor", "iron_butterfly")):
            if width is None:
                width = max(1.0, round((spot or 500) * 0.01)) if d.structure != "iron_butterfly" else max(2.0, round((spot or 500) * 0.02))
                missing.append("spread width")
                assume("width", width, "about 1% of the underlying price")
                add("Width", f"{width:g} points", "assumed", "width")
            else:
                add("Width", f"{width:g} points", field="width")
        # ---- exits
        exits: dict = {"profit_target": {"enabled": False, "mode": "pct", "value": 50}, "stop_loss": {"enabled": False, "mode": "pct", "value": 200}}
        any_exit = False
        if d.underlying_target_pct:
            mv = "rises" if dirn >= 0 else "falls"
            add("Target", f"{d.symbol} {mv} {d.underlying_target_pct:g}% from its price at entry (underlying move, not option return)", field="underlying_target_pct")
            any_exit = True
        if d.underlying_stop_pct:
            add("Stop", f"{d.symbol} moves {d.underlying_stop_pct:g}% against the position from its entry price", field="underlying_stop_pct")
        if d.underlying_target_pct or d.underlying_stop_pct:
            if dirn != 0:
                exits["underlying"] = {"enabled": True, "move_for_pct": d.underlying_target_pct, "move_against_pct": d.underlying_stop_pct}
            else:
                exits["underlying"] = {"enabled": True, "move_against_pct": d.underlying_target_pct or d.underlying_stop_pct}
        custom_kids: list[dict] = []
        if d.underlying_target_abs:
            up = dirn >= 0
            custom_kids.append({"id": "ut", "kind": "cond", "type": "expr", "params": {"expr": f"price {'>=' if up else '<='} entry.underlying_price {'+' if up else '-'} {d.underlying_target_abs:g}", "label": f"{d.symbol} {'up' if up else 'down'} ${d.underlying_target_abs:g} from entry"}})
            add("Target", f"{d.symbol} {'rises' if up else 'falls'} ${d.underlying_target_abs:g} from its price at entry (a dollar move of the underlying, not a percent and not option profit)", field="underlying_target_abs")
            any_exit = True
        if d.underlying_stop_abs:
            up = dirn >= 0
            custom_kids.append({"id": "us", "kind": "cond", "type": "expr", "params": {"expr": f"price {'<=' if up else '>='} entry.underlying_price {'-' if up else '+'} {d.underlying_stop_abs:g}", "label": f"{d.symbol} ${d.underlying_stop_abs:g} against the position"}})
            add("Stop", f"{d.symbol} moves ${d.underlying_stop_abs:g} against the position from its entry price", field="underlying_stop_abs")
        if d.premium_target_pct:
            exits["profit_target"] = {"enabled": True, "mode": "pct", "value": d.premium_target_pct}
            add("Profit target", f"{d.premium_target_pct:g}% of the option premium (option return, not underlying move)", field="premium_target_pct")
            any_exit = True
        if d.premium_stop_pct:
            exits["stop_loss"] = {"enabled": True, "mode": "pct", "value": d.premium_stop_pct}
            add("Stop loss", f"{d.premium_stop_pct:g}% of the option premium", field="premium_stop_pct")
        if not d.underlying_stop_pct and not d.premium_stop_pct and not d.underlying_stop_abs:
            missing.append("stop loss")
            assume("stop", None, "no stop was described, so none is applied (loss is limited to the premium for long options)" if d.structure in ("long_call", "long_put", "long_straddle", "long_strangle", "bull_call_spread", "bear_put_spread") else "no stop was described, so none is applied")
            add("Stop loss", "None", "assumed", "premium_stop_pct")
        exit_time = d.exit_time
        if d.hold_sessions:
            exits["max_hold"] = {"enabled": True, "days": d.hold_sessions}
            add("Holding period", f"{d.hold_sessions} session(s), exiting at the same time of day", field="hold_sessions")
            any_exit = True
        if d.hold_minutes:
            exits["max_hold"] = {"enabled": True, "minutes": int(d.hold_minutes)}
            add("Holding period", f"{int(d.hold_minutes)} minutes after the entry fill (or the close, whichever comes first)", field="hold_minutes")
            any_exit = True
        if d.partials:
            stages = []
            for i, p_ in enumerate(d.partials):
                trig = {"profit_pct": p_.premium_pct} if p_.premium_pct is not None else {"underlying_for_pct": p_.underlying_pct}
                stages.append({"name": f"partial{i + 1}", **trig, "close_pct": round(max(1.0, min(100.0, p_.fraction * 100)), 4)})
                what = f"the premium is up {p_.premium_pct:g}% (option return)" if p_.premium_pct is not None else f"{d.symbol} has moved {p_.underlying_pct:g}% in the position's favour (underlying move)"
                add("Partial exit", f"Close {p_.fraction * 100:g}% of the contracts when {what}; the rest stays open under the other exit rules", field="partials")
            exits["stages"] = stages
            any_exit = True
        if d.exit_weekday:
            k = WEEKDAYS.index(d.exit_weekday)
            last = " or is_last_session_of_week" if k == 4 else ""
            custom_kids.append({"id": "x1", "kind": "cond", "type": "expr", "params": {"expr": f"(weekday == {k}{last}) and minutes_to_close <= 5", "label": f"{d.exit_weekday} close"}})
            add("Exit", f"At the close on {d.exit_weekday}" + (" (or the week's last session)" if k == 4 else ""), field="exit_weekday")
            any_exit = True
        for i, c in enumerate(d.exit_conditions):
            custom_kids.append({"id": f"xc{i}", "kind": "cond", "type": "expr", "params": {"expr": c.expr, "label": c.text}})
            add("Exit rule", f"{c.text or c.expr}   [{c.expr}]" + (" - closes every lot" if d.adds else ""), c.source, "exit_conditions")
            any_exit = True
        for a_ in d.adds:
            add("Scale in", f"{a_.text or a_.expr}   [{a_.expr}] - add {a_.quantity} more, at most {a_.max_adds} time{'s' if a_.max_adds != 1 else ''}; same contract as the first lot", "stated", "adds")
        if custom_kids:
            exits["custom"] = {"id": "x", "kind": "group", "op": "or", "children": custom_kids}
        if exit_time:
            exits["time_exit"] = {"enabled": True, "time": "16:00" if exit_time == "close" else exit_time, "dte_lte": None}
            add("Time exit", fmt_time(exit_time, d.exit_time_stated), field="exit_time")
            any_exit = True
        elif d.hold_to_expiration:
            add("Exit", "Hold until expiration (settled per the contract)", field="hold_to_expiration")
        elif not d.hold_sessions and not d.exit_weekday:
            if dte == 0 and d.hold_minutes:
                exits["time_exit"] = {"enabled": True, "time": "15:55", "dte_lte": None}  # a same-day contract is never carried into its expiration
            elif d.hold_minutes:
                pass
            elif dte == 0:
                exits["time_exit"] = {"enabled": True, "time": "15:55", "dte_lte": None}
                assume("exit_time", "15:55", "0DTE positions are closed five minutes before the bell so they are not exercised")
                add("End-of-day exit", "15:55 ET (before the 0DTE contracts expire)", "assumed", "exit_time")
            else:
                assume("exit", "expiration", "no exit rule was described, so the position is held until expiration")
                add("Exit", "Hold until expiration", "assumed", "hold_to_expiration")
            if not any_exit:
                missing.append("exit rule")
        if d.structure and not errors:
            try:
                by_contract = d.sizing is None or d.sizing.mode == "contracts"
                legs = build_legs(d.structure, dte, strike, width or 5.0, max(1, d.quantity) if by_contract else 1)
                tree = _cond_tree(Draft(**{**d.model_dump(), "entry_time": entry_time}))
                te = exits.get("time_exit") or {}
                if te.get("enabled") and te.get("dte_lte") is None and entry_time is None:
                    # an entry that waits for an event has no clock of its own. Taken at or after the time everything is closed
                    # it would be closed at once, or (after a same-day contract's last exit) carried into expiration: measured on
                    # stored data, 9 of 263 trades of one strategy did exactly that and lost more than the other 254 made
                    m_ = cal.parse_hhmm(te["time"]) - 1
                    tree["children"].append({"id": "lw", "kind": "cond", "type": "time_window", "params": {"start": "09:30", "end": f"{m_ // 60:02d}:{m_ % 60:02d}"}})
                    assume("last_entry", te["time"], f"no position is opened at or after {te['time']}, the time every position is closed")
                    add("Last entry", f"before {te['time']} ET, the time every position is closed", "assumed", "exit_time")
                strategy = StrategyConfig.model_validate({
                    "name": d.title or "AI strategy", "symbol": d.symbol, "legs": legs,
                    "entry": {"conditions": tree, "max_entries_per_day": max(1, int(d.max_entries or 1)), "max_open_positions": max(1, int(d.max_entries or 1)) if same_day or dte == 0 else 5,
                              "sequence": [{"name": s_.name, "when": s_.expr, "text": s_.text, "within_minutes": s_.within_minutes} for s_ in d.sequence],
                              "adds": [{"when": a_.expr, "text": a_.text, "quantity": a_.quantity, "max_adds": a_.max_adds} for a_ in d.adds]},
                    "exits": exits, "sizing": sizing_config(d),
                }).model_dump(mode="json")
            except Exception as e:  # noqa: BLE001
                errors.append(f"strategy: {e}")
    if d.start or d.end:
        add("Period", f"{d.start or 'earliest data'} to {d.end or 'latest data'}", field="period")
    else:
        add("Period", "All available history (no period was stated)", "assumed", "period")
    for f_, why_ in d.carried.items():  # not said in this request: shown as an assumption, with the reason
        for it in items:
            if it["field"] == f_ and it["status"] == "stated":
                it["status"] = "assumed"
                assume(f_, it["value"], why_)
    execution = execution_overrides(d)
    if d.kind == "options":
        if d.costs and (d.costs.preset or d.costs.commission_per_contract is not None or d.costs.slippage_per_contract is not None or d.costs.slippage_pct is not None):
            add("Costs", costs_text(execution), "stated", "costs")
        else:
            add("Costs", "$0.65 commission per contract; quotes fill at the bid/ask; trade prints fill at the print with no extra slippage", "assumed", "costs")
    effective = {"entry_time": entry_time, "dte": dte, "execution": execution} if d.kind == "options" else {"entry_time": study_entry}
    if d.kind == "options":
        # V10: the values the run will actually use where the request left them open, so that the reading shown to the user
        # states them (until V10 the sentence said "0 days to expiration" for a trade compiled with 14)
        _strike, _width = locals().get("strike"), locals().get("width")
        effective.update(strike={"method": _strike.method, "value": _strike.value} if _strike is not None else None, width=_width,
                         exit_time=((locals().get("exits") or {}).get("time_exit") or {}).get("time") if (locals().get("exits") or {}).get("time_exit", {}).get("enabled") else None)
    return {"strategy": strategy, "study": study if not any(e.startswith("study") for e in errors) else None, "items": items, "assumptions": assumptions, "effective": effective,
            "missing": sorted(set(missing), key=missing.index), "errors": errors}


def sizing_config(d: Draft) -> dict:
    """The engine's sizing (strata.engine.models.Sizing) for the draft's sizing in a trader's terms."""
    s = d.sizing
    if s is None or s.mode == "contracts":
        out: dict = {"method": "fixed", "contracts": 1}
    elif s.mode == "dollars":
        out = {"method": "premium_dollars", "dollars": float(s.value)}
    elif s.mode == "equity_pct":
        out = {"method": "premium_pct", "pct": float(s.value), "reinvest": bool(s.reinvest)}
    elif s.mode == "risk_dollars":
        out = {"method": "risk_dollars", "dollars": float(s.value), "risk_basis": "stop" if d.premium_stop_pct else "max_loss"}
    else:
        out = {"method": "risk_pct", "pct": float(s.value), "reinvest": bool(s.reinvest), "risk_basis": "stop" if d.premium_stop_pct else "max_loss"}
    if s is not None and s.daily_loss_stop:
        out["daily_loss_stop"] = float(s.daily_loss_stop)
    if out["method"] != "fixed":
        out["max_contracts"] = 500  # a budget decides the count; the cap only guards against a mispriced print
    return out


def sizing_short(d: Draft) -> str:
    """The sizing in a few words, for a change line ("$400 of premium" to "$250 at risk")."""
    s = d.sizing
    if s is None or s.mode == "contracts":
        base = f"{d.quantity if s is None else int(s.value)} contract{'s' if (d.quantity if s is None else int(s.value)) != 1 else ''}"
    else:
        base = {"dollars": f"${s.value:,.0f} of premium", "risk_dollars": f"${s.value:,.0f} at risk", "risk_pct": f"{s.value:g}% of the account at risk",
                "equity_pct": f"{s.value:g}% of the account in premium"}[s.mode]
    extra = [f"from a ${s.capital:,.0f} account" if s is not None and s.capital else "", ("reinvesting" if s.reinvest else "") if s is not None and s.mode in ("risk_pct", "equity_pct") else "",
             f"stopping for the day after losing ${s.daily_loss_stop:,.0f}" if s is not None and s.daily_loss_stop else ""]
    return ", ".join([base] + [x for x in extra if x]) + " per trade"


def sizing_text(d: Draft) -> str:
    """The sizing in words, with what "risk" means for this position."""
    s = d.sizing
    if s is None or s.mode == "contracts":
        txt = f"{d.quantity} contract{'s' if d.quantity != 1 else ''} per trade"
    elif s.mode == "dollars":
        txt = f"spend up to ${s.value:,.0f} of premium per trade (as many whole contracts as that buys; none when one contract costs more)"
    elif s.mode == "equity_pct":
        txt = f"spend up to {s.value:g}% of the account on premium per trade"
    else:
        amount = f"${s.value:,.0f}" if s.mode == "risk_dollars" else f"{s.value:g}% of the account"
        basis = (f"the loss if the {d.premium_stop_pct:g}% stop on the option is hit (premium x {d.premium_stop_pct:g}%)" if d.premium_stop_pct else
                 "the most the position can lose (for a bought option, the premium paid)")
        txt = f"risk up to {amount} per trade, where risk is {basis}; a stop can be jumped, so a realised loss can be larger"
    if s is not None and s.mode in ("equity_pct", "risk_pct"):
        txt += "; the account grows or shrinks with the results (profits are reinvested)" if s.reinvest else "; always of the starting account (profits are not reinvested)"
    if s is not None and s.capital:
        txt += f"; starting account ${s.capital:,.0f}"
    if s is not None and s.daily_loss_stop:
        txt += f"; no new trade that day once closed trades have lost ${s.daily_loss_stop:,.0f}"
    return txt


def execution_overrides(d: Draft) -> dict:
    """Execution settings implied by the draft's cost assumptions (only the keys that differ from the engine defaults)."""
    c = d.costs
    out: dict = {}
    if c is None:
        return out
    if c.preset == "conservative":
        # worse than the defaults on every axis: a fee on top of the commission, a few cents per share on every
        # fill, a short delay before the order reaches the market, and 3% off every trade print
        out.update(commission_per_contract=0.65, fee_per_contract=0.05, slippage_per_contract=0.03, trade_bar_slippage_pct=3.0, latency_seconds=5, fill_model="conservative")
    elif c.preset == "none":
        out.update(commission_per_contract=0.0, fee_per_contract=0.0, slippage_per_contract=0.0, trade_bar_slippage_pct=0.0)
    if c.commission_per_contract is not None:
        out["commission_per_contract"] = float(c.commission_per_contract)
    if c.slippage_per_contract is not None:
        out["slippage_per_contract"] = float(c.slippage_per_contract)
    if c.slippage_pct is not None:
        out["trade_bar_slippage_pct"] = float(c.slippage_pct)
    return out


def costs_text(ex: dict) -> str:
    parts = [f"${ex.get('commission_per_contract', 0.65) + ex.get('fee_per_contract', 0):.2f} per contract in commissions and fees"]
    if ex.get("slippage_per_contract"):
        parts.append(f"${ex['slippage_per_contract']:.2f} per share of slippage on every option fill")
    if ex.get("trade_bar_slippage_pct"):
        parts.append(f"{ex['trade_bar_slippage_pct']:g}% worse than each observed trade print")
    if ex.get("latency_seconds"):
        parts.append(f"{ex['latency_seconds']:g} s order delay")
    return "; ".join(parts)


def build_spec(d: Draft, compiled: dict, mode: str | None = None, approvals: dict | None = None, label: str = "", interpreter: str = "") -> dict:
    """The research-pipeline request for a compiled draft. `mode` overrides the draft's own
    (None = the draft's stated mode, else chosen by the pipeline from what the data allows)."""
    as_study = d.kind == "study" or (mode or d.mode) == "study"
    if as_study and compiled.get("study") is None:
        raise ValueError("the study is not runnable: " + "; ".join(compiled.get("errors") or ["invalid configuration"]))
    if not as_study and compiled.get("strategy") is None:
        raise ValueError("the strategy is not runnable: " + "; ".join(compiled.get("errors") or ["invalid configuration"]))
    study = {k: v for k, v in (compiled.get("study") or {}).items() if k not in ("start", "end")} if compiled.get("study") else None
    spec: dict = {"kind": "study" if as_study else "options", "mode": "study" if as_study else (mode or d.mode or "auto"), "study": study, "backtest": None, "symbol": d.symbol, "label": label or d.title,
                  "requested": {"start": d.start.isoformat() if d.start else None, "end": d.end.isoformat() if d.end else None, "stated": bool(d.start or d.end)},
                  "approvals": dict(approvals or {}), "provider": d.provider}
    if not as_study:
        intraday = bool(d.sequence or d.adds) or any(X.needs_intraday(c.expr) for c in list(d.conditions) + list(d.exit_conditions)) or (d.entry_time not in (None, "open", "close")) \
            or bool(d.underlying_target_pct or d.underlying_stop_pct or d.underlying_target_abs or d.underlying_stop_abs or d.exit_time)
        ex = {"signal_interval_minutes": 1 if intraday or spec["mode"] in ("trades", "auto") else 5, **(compiled.get("effective") or {}).get("execution", {})}
        spec["backtest"] = {"strategies": [compiled["strategy"]], "execution": ex, "label": label or d.title}
        if d.sizing is not None and d.sizing.capital:
            spec["backtest"]["capital"] = {"initial_capital": float(d.sizing.capital)}
    n = len(d.conditions) + len(d.sequence)
    spec["interpretation"] = {"interpreter": interpreter, "detail": f"{n} entry rule{'s' if n != 1 else ''}" + (f", {len(d.sequence)}-step sequence" if d.sequence else "") +
                              (f"; {len(compiled.get('assumptions') or [])} assumption(s) disclosed" if compiled.get("assumptions") else "") + (f"; read by {interpreter}" if interpreter else "")}
    return spec


# ----------------------------------------------------------------------------- coverage
def _needs(d: Draft, compiled: dict) -> dict:
    entry = (compiled.get("effective") or {}).get("entry_time") or d.entry_time or ("open" if not any(X.needs_intraday(c.expr) for c in d.conditions) else None)
    clock = "close" if entry == "close" else "open" if entry in ("open", "09:30") else "intraday"
    intraday_rules = any(X.needs_intraday(c.expr) for c in d.conditions)
    strat = compiled.get("strategy")
    dtes = [l["dte"] for l in (strat or {}).get("legs", []) if l.get("kind", "option") == "option"]
    same_day_exit = bool(d.exit_time) and not d.hold_sessions and not d.exit_weekday
    intraday_exit = same_day_exit or bool(d.underlying_target_pct or d.underlying_stop_pct or d.underlying_target_abs or d.underlying_stop_abs) and clock != "close"
    return {"clock": clock, "entry": entry, "intraday_rules": intraday_rules, "min_dte": min(dtes) if dtes else None, "same_day_exit": same_day_exit, "intraday_exit": intraday_exit}


def _range(a, b):
    return f"{a} to {b}"


def assess(d: Draft, compiled: dict, store: MarketStore | None = None) -> dict:
    """Which research modes can answer this request with the data that is actually stored."""
    from strata.data.catalog import profile
    store = store or MarketStore()
    prof = profile(store, d.symbol)
    need = _needs(d, compiled)
    out = {"symbol": d.symbol, "needs": need, "modes": {}, "messages": [], "profile": None}
    if prof is None or not (prof.get("daily") or prof.get("intraday")):
        msg = f"No historical price data is stored for {d.symbol}. Download it under Data (daily history is free from Yahoo); nothing will be simulated in its place."
        for m in ("historical", "study", "modeled"):
            out["modes"][m] = {"ok": False, "reason": msg}
        out["recommended"] = None
        out["messages"].append(msg)
        return out
    daily, intr, opt = prof.get("daily"), prof.get("intraday") or {}, prof.get("options")
    out["profile"] = {"daily": daily and [daily["first"], daily["last"]], "intraday": {k: [v["first"], v["last"], v["sessions"]] for k, v in intr.items()},
                      "options": opt and {"first": opt["first"], "last": opt["last"], "resolution": opt["resolution"], "quote_type": opt["quote_type"], "provider": opt["provider"], "sessions": opt["sessions"]},
                      "synthetic": bool(prof["meta"].get("synthetic"))}
    best_intr = next((r for r in ("1m", "5m", "15m", "30m", "60m") if r in intr), None)
    need_intr = need["clock"] == "intraday" or need["intraday_rules"] or (d.kind == "study" and d.study is not None and d.study.horizon_kind == "time")
    entry_min = None if need["entry"] in (None, "open", "close") else cal.parse_hhmm(need["entry"])
    # resolutions able to observe the entry minute
    usable = [r for r in intr if entry_min is None or (entry_min - 570) % int(r[:-1]) == 0]
    fine = next((r for r in ("1m", "5m", "15m", "30m", "60m") if r in usable), None)
    und_desc = (f"daily bars {_range(daily['first'], daily['last'])}" if daily else "no daily bars") + "".join(f"; {r} bars {_range(v['first'], v['last'])}" for r, v in intr.items())

    # ---- study
    if need_intr:
        if fine:
            v = intr[fine]
            note = f"Needs intraday prices: evaluated on the {v['sessions']} sessions with {fine} bars ({_range(v['first'], v['last'])})."
            out["modes"]["study"] = {"ok": True, "window": [v["first"], v["last"]], "resolution": fine, "note": note}
        else:
            why = f"This question needs intraday prices{' at ' + need['entry'] + ' ET' if entry_min else ''}, but {d.symbol} only has {und_desc}."
            out["modes"]["study"] = {"ok": False, "reason": why + " Add intraday history under Data (Yahoo: recent months free; Alpaca free key: 1-minute bars since 2016)."}
    else:
        out["modes"]["study"] = {"ok": bool(daily), "window": [daily["first"], daily["last"]] if daily else None, "resolution": "1d" + (f" (+{best_intr} where stored)" if best_intr else ""),
                                 "note": f"Uses {und_desc}." + (" Target/stop order within a day is only resolved where intraday bars exist." if d.kind == "study" and d.study and d.study.target and d.study.stop else "")}
    # ---- historical options
    if d.kind == "study":
        out["modes"]["historical"] = {"ok": False, "reason": "This is a question about the underlying's behaviour; no option position was described."}
        out["modes"]["modeled"] = {"ok": False, "reason": "No option position was described."}
    elif not opt:
        free = "the free end-of-day archive (SPY/QQQ/IWM, 2008-2025) or DoltHub (2019 onward)" if d.symbol in ("SPY", "QQQ", "IWM") else "DoltHub's free end-of-day chains (2019 onward)"
        out["modes"]["historical"] = {"ok": False, "reason": f"No option quotes are stored for {d.symbol}. You can download {free} under Data, or connect an intraday provider (ThetaData, Databento, Polygon)."}
    else:
        span = _range(opt["first"], opt["last"])
        if opt["resolution"] == "EOD":
            blockers = []
            if need["clock"] != "close":
                blockers.append("an entry at the open (09:30 ET)" if need["entry"] == "open" else f"an entry at {need['entry']} ET" if need["entry"] else "an entry on an intraday signal")
            if need["min_dte"] == 0:
                blockers.append("0DTE contracts")
            if need["intraday_exit"] and need["clock"] == "close":
                blockers.append("an intraday exit")
            if blockers:
                out["modes"]["historical"] = {"ok": False, "reason": f"{d.symbol} option history is end-of-day only (one {opt['quote_type'][:-1]} per contract at the 16:00 ET close, {span}, via {opt['provider']}). "
                                              f"{(lambda t: t[0].upper() + t[1:])(' and '.join(blockers))} need{'s' if len(blockers) == 1 else ''} intraday option quotes, which are not stored. End-of-day data is never used to simulate fills at other times."}
            else:
                out["modes"]["historical"] = {"ok": True, "window": [opt["first"], opt["last"]], "resolution": "EOD", "note": f"End-of-day {opt['quote_type']} {span} via {opt['provider']}: entries and exits fill at the 16:00 ET snapshot."}
        else:
            if need_intr and not fine:
                out["modes"]["historical"] = {"ok": False, "reason": f"Option quotes exist ({opt['resolution']}, {span}) but the underlying has no intraday bars able to evaluate the entry rule."}
            else:
                out["modes"]["historical"] = {"ok": True, "window": [opt["first"], opt["last"]], "resolution": opt["resolution"],
                                              "note": f"{opt['resolution']} option {opt['quote_type']} {span} via {opt['provider']}." + (" Trade prints only - fills use last-trade prices, no bid/ask." if opt["quote_type"] == "trades" else "")}
    # ---- modeled
    if d.kind == "options":
        if need["clock"] == "intraday" or need["intraday_rules"] or need["intraday_exit"]:
            r = fine or best_intr
            if r and (fine or not entry_min):
                v = intr[r]
                out["modes"]["modeled"] = {"ok": True, "window": [v["first"], v["last"]], "resolution": r,
                                           "note": f"Real {d.symbol} {r} prices ({_range(v['first'], v['last'])}) with THEORETICAL option premiums. Estimates, not observed option prices."}
            else:
                out["modes"]["modeled"] = {"ok": False, "reason": f"Modeled premiums still need real intraday {d.symbol} prices for this timing; only {und_desc} is stored."}
        else:
            out["modes"]["modeled"] = {"ok": bool(daily), "window": [daily["first"], daily["last"]] if daily else None, "resolution": "1d",
                                       "note": f"Real {d.symbol} prices with THEORETICAL option premiums (volatility from VIX). Estimates, not observed option prices."}
    h, s = out["modes"]["historical"], out["modes"]["study"]
    out["recommended"] = "historical" if h["ok"] else "study" if s["ok"] else None
    if d.kind == "options" and not h["ok"]:
        tail = "You can run the directional market study on the underlying now" + (", or explicitly opt in to a modeled-premium estimate." if out["modes"]["modeled"]["ok"] else ".")
        out["messages"].append(f"{h['reason']} {tail if s['ok'] else s.get('reason', '')}")
    elif not s["ok"] and d.kind == "study":
        out["messages"].append(s["reason"])
    return out


def window_for(mode: str, d: Draft, cover: dict) -> tuple[str, str] | None:
    m = cover["modes"].get(mode) or {}
    if not m.get("ok") or not m.get("window"):
        return None
    a, b = m["window"]
    if d.start:
        a = max(a, d.start.isoformat())
    if d.end:
        b = min(b, d.end.isoformat())
    return (a, b) if a <= b else None


def backtest_config(d: Draft, compiled: dict, cover: dict, mode: str) -> dict:
    """BacktestConfig for a compiled draft in 'historical' or 'modeled' mode."""
    win = window_for(mode, d, cover)
    if win is None or compiled.get("strategy") is None:
        raise ValueError((cover["modes"].get(mode) or {}).get("reason") or "strategy is not runnable")
    need = cover["needs"]
    fine_signals = need["clock"] == "intraday" or need["intraday_exit"] or need["intraday_rules"]
    cfg = {"strategies": [compiled["strategy"]], "start": win[0], "end": win[1], "label": d.title,
           "execution": {"signal_interval_minutes": 1 if fine_signals else 5},
           "pricing": {"mode": "modeled" if mode == "modeled" else "historical"},
           "allow_synthetic": bool((cover.get("profile") or {}).get("synthetic"))}
    return BacktestConfig.model_validate(cfg).model_dump(mode="json")


def _show(k: str, v):
    if v is None:
        return "none"
    if k == "conditions":
        return "; ".join(c["text"] or c["expr"] for c in v) or "none"
    if k == "strike":
        return {"atm": "ATM", "delta": f"{v['value']:g} delta", "pct_otm": f"{v['value']:g}% OTM", "strikes_otm": f"{v['value']:g} strikes OTM"}[v["method"]]
    if k in ("study.target", "study.stop"):
        return f"{v['value']:+g}%" if v["kind"] == "pct" else f"{v['value']:+g} $" if v["kind"] == "abs" else v["expr"]
    if k == "structure":
        return STRUCTURES.get(v, v)
    if k == "sizing" and isinstance(v, dict):
        return sizing_short(Draft(kind="options", sizing=DraftSizing.model_validate(v)))
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return v


def diff_drafts(a: Draft | None, b: Draft) -> list[dict]:
    """Field-level changes between two versions (for the history panel)."""
    if a is None:
        return []

    def flat(d: Draft) -> dict:
        x = d.model_dump(mode="json")
        st = x.pop("study", None) or {}
        for k in ("title", "notes", "entry_time_stated", "exit_time_stated", "symbol_stated", "tz_assumed", "questions", "carried", "unsupported", "review"):  # review: why a reading waits, not a part of it
            x.pop(k, None)
        if d.kind == "study":
            x.update({f"study.{k}": v for k, v in st.items()})
        return x

    fa, fb = flat(a), flat(b)
    return [{"field": k, "from": _show(k, fa.get(k)), "to": _show(k, fb.get(k))} for k in fb if fa.get(k) != fb.get(k)]
