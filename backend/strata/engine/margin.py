"""Buying-power / margin requirements.

Requirement R is the cash that must be held against a position *in addition to* the
premium already reflected in cash. Buying-power reduction (BPR) = R - entry cash flow.

  * Defined-risk structures (every short option covered by a long option of the same
    type, or calls by long stock): R = worst expiration value of the legs, with every
    leg valued at intrinsic (conservative for longer-dated long legs, which are worth at
    least intrinsic). This covers verticals, condors, butterflies, calendars, diagonals,
    ratio spreads with covered wings and covered calls.
  * Uncovered short options under "regt": CBOE/Reg-T naked formula
      call: max(20% S - OTM, 10% S) + premium ; put: max(20% S - OTM, 10% K) + premium
    per share; when both calls and puts are uncovered (strangle/straddle) the larger
    side plus the premium of the other side.
  * Under "cash": uncovered puts are cash-secured (strike x multiplier), uncovered calls
    are not permitted (infinite requirement).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MLeg:
    is_call: bool
    strike: float
    qty: int  # signed contracts
    multiplier: float
    mid: float  # current premium per share


def _value_at(legs: list[MLeg], shares: float, S: float) -> float:
    v = shares * S
    for l in legs:
        intr = max(S - l.strike, 0.0) if l.is_call else max(l.strike - S, 0.0)
        v += l.qty * l.multiplier * intr
    return v


def defined_risk_requirement(legs: list[MLeg], shares: float) -> float:
    if not legs and shares >= 0:
        return 0.0
    strikes = sorted({l.strike for l in legs})
    pts = [0.0] + strikes + [(strikes[-1] if strikes else 1.0) * 4 + 1.0]
    worst = min(_value_at(legs, shares, S) for S in pts)
    slope = shares + sum(l.qty * l.multiplier for l in legs if l.is_call)
    if slope < -1e-9:
        return math.inf
    return max(0.0, -worst)


def naked_req(l: MLeg, spot: float) -> float:
    if l.is_call:
        otm = max(l.strike - spot, 0.0)
        per = max(0.20 * spot - otm, 0.10 * spot) + l.mid
    else:
        otm = max(spot - l.strike, 0.0)
        per = max(0.20 * spot - otm, 0.10 * l.strike) + l.mid
    return per * l.multiplier


def requirement(legs: list[MLeg], shares: float, spot: float, model: str = "regt") -> float:
    legs = [l for l in legs if l.qty != 0]
    mult = legs[0].multiplier if legs else 100.0
    short_calls = sum(-l.qty for l in legs if l.is_call and l.qty < 0)
    long_calls = sum(l.qty for l in legs if l.is_call and l.qty > 0)
    short_puts = sum(-l.qty for l in legs if not l.is_call and l.qty < 0)
    long_puts = sum(l.qty for l in legs if not l.is_call and l.qty > 0)
    unc_calls = max(0.0, short_calls - long_calls - max(shares, 0.0) / mult)
    unc_puts = max(0.0, short_puts - long_puts)
    if unc_calls <= 1e-9 and unc_puts <= 1e-9:
        return defined_risk_requirement(legs, shares)
    if model == "cash":
        if unc_calls > 1e-9:
            return math.inf
        return defined_risk_requirement(legs, shares)
    # Reg-T: strip the uncovered contracts (riskiest first) and margin them as naked
    rest = [MLeg(l.is_call, l.strike, l.qty, l.multiplier, l.mid) for l in legs]
    naked_c, naked_p = [], []

    def strip(is_call: bool, n: float, out: list):
        shorts = sorted([l for l in rest if l.is_call == is_call and l.qty < 0], key=lambda x: x.strike, reverse=not is_call)
        for l in shorts:
            if n <= 1e-9:
                break
            take = min(-l.qty, n)
            out.append(MLeg(l.is_call, l.strike, -int(take), l.multiplier, l.mid))
            l.qty += int(take)
            n -= take

    strip(True, unc_calls, naked_c)
    strip(False, unc_puts, naked_p)
    r_rest = defined_risk_requirement(rest, shares)
    rc = sum(naked_req(l, spot) * -l.qty for l in naked_c)
    rp = sum(naked_req(l, spot) * -l.qty for l in naked_p)
    if naked_c and naked_p:
        prem_c = sum(l.mid * l.multiplier * -l.qty for l in naked_c)
        prem_p = sum(l.mid * l.multiplier * -l.qty for l in naked_p)
        r_naked = max(rc + prem_p, rp + prem_c)
    else:
        r_naked = rc + rp
    return r_rest + r_naked


def max_profit_loss(legs: list[MLeg], shares: float, entry_cash: float) -> tuple[float, float]:
    """Expiration max profit / max loss (both in $, loss positive) for single-expiration
    structures; inf when unbounded."""
    strikes = sorted({l.strike for l in legs})
    pts = [0.0] + strikes + [(strikes[-1] if strikes else 1.0) * 4 + 1.0]
    vals = [_value_at(legs, shares, S) + entry_cash for S in pts]
    slope = shares + sum(l.qty * l.multiplier for l in legs if l.is_call)
    mp = math.inf if slope > 1e-9 else max(vals)
    ml = math.inf if slope < -1e-9 else -min(vals)
    return mp, max(ml, 0.0)
