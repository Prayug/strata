"""Contract selection: turn declarative leg specs into concrete contracts using only the
chain snapshot observable at the decision time."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from strata.engine.market import DayChain, day_num, exp_close_epoch, from_day_num
from strata.engine.models import LegSpec


@dataclass
class Selected:
    leg: LegSpec
    row: int  # index into DayChain arrays (-1 for stock legs)
    strike: float
    expiration: int  # day number
    dte: int


class SelectionError(Exception):
    pass


def _choose_expiration(exps: np.ndarray, today: int, leg: LegSpec) -> int | None:
    dte = exps - today
    ok = dte >= 0
    if leg.dte_min is not None:
        ok &= dte >= leg.dte_min
    if leg.dte_max is not None:
        ok &= dte <= leg.dte_max
    cand = exps[ok]
    if cand.size == 0:
        return None
    d = cand - today
    if leg.dte_mode == "exact" or (leg.dte_mode == "nearest" and leg.dte == 0):
        m = d == leg.dte  # a 0DTE leg never falls through to a later expiration
        return int(cand[m][0]) if m.any() else None
    if leg.dte_mode == "at_least":
        m = d >= leg.dte
        return int(cand[m][np.argmin(d[m])]) if m.any() else None
    if leg.dte_mode == "at_most":
        m = d <= leg.dte
        return int(cand[m][np.argmax(d[m])]) if m.any() else None
    return int(cand[np.argmin(np.abs(d - leg.dte) * 2 + (d < leg.dte))])  # ties -> longer


def _filters(ch: DayChain, idx: np.ndarray, leg: LegSpec, t: int, max_age: int) -> tuple[np.ndarray, dict]:
    reasons = {}
    keep = np.ones(idx.size, bool)

    def apply(mask, name):
        nonlocal keep
        bad = keep & ~mask
        if bad.any():
            reasons[name] = int(bad.sum())
        keep &= mask

    bid, ask, mid = ch.bid[idx], ch.ask[idx], ch.mid[idx]
    apply(np.isfinite(bid) & np.isfinite(ask) & (ask >= bid) & (ask > 0), "invalid_quote")
    apply(ch.quote_ts[idx] <= t, "future_quote")
    apply((t - ch.quote_ts[idx]) <= max_age, "stale_quote")
    if leg.side == "sell":
        apply(bid > max(leg.min_bid, 0.0), "no_bid")
    elif leg.min_bid > 0:
        apply(bid >= leg.min_bid, "min_bid")
    if leg.iv_min is not None:
        apply(ch.iv[idx] * 100 >= leg.iv_min, "iv_min")
    if leg.iv_max is not None:
        apply(ch.iv[idx] * 100 <= leg.iv_max, "iv_max")
    if leg.max_spread_abs is not None:
        apply((ask - bid) <= leg.max_spread_abs + 1e-9, "spread_abs")
    if leg.max_spread_pct is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            apply(np.where(mid > 0, (ask - bid) / mid * 100, np.inf) <= leg.max_spread_pct, "spread_pct")
    if leg.min_volume is not None:
        apply(ch.volume[idx] >= leg.min_volume, "min_volume")
    if leg.min_open_interest is not None:
        apply(ch.oi[idx] >= leg.min_open_interest, "min_oi")
    return idx[keep], reasons


def select_contracts(ch: DayChain, si: int, legs: list[LegSpec], spot: float, today: date, max_quote_age: int) -> tuple[list[Selected] | None, str, dict]:
    """Returns (selection, failure_reason, details)."""
    t = int(ch.snap_ts[si])
    s = ch.rows(si)
    today_n = day_num(today)
    exps_all = np.unique(ch.exp[s])
    # expirations whose close is still ahead (a 0DTE after the close is not tradable)
    alive = exps_all[exp_close_epoch(exps_all) > t]
    chosen: dict[str, Selected] = {}
    details: dict = {}
    order = [l for l in legs if l.strike.method != "leg_offset"] + [l for l in legs if l.strike.method == "leg_offset"]
    for leg in order:
        if leg.kind == "stock":
            chosen[leg.id] = Selected(leg, -1, 0.0, -1, 0)
            continue
        e = _choose_expiration(alive, today_n, leg)
        if e is None:
            listed = sorted(int(x) for x in (alive - today_n)) if alive.size else []
            near = min(listed, key=lambda x: abs(x - leg.dte)) if listed else None
            exact = leg.dte_mode == "exact" or leg.dte == 0
            return None, (f"leg {leg.id}: no {'exact ' if exact else ''}{leg.dte}DTE expiration in the stored chain" + (f" (nearest listed: {near}DTE)" if near is not None else "")), details
        base = np.arange(s.start, s.stop)
        m = (ch.exp[base] == e) & (ch.is_call[base] == (leg.right == "C"))
        idx_all = base[m]
        if idx_all.size == 0:
            return None, f"leg {leg.id}: no {leg.right} contracts for {from_day_num(e)}", details
        idx, rej = _filters(ch, idx_all, leg, t, max_quote_age)
        if rej:
            details[leg.id] = {"filtered": rej}
        if idx.size == 0:
            return None, f"leg {leg.id}: all {idx_all.size} candidates filtered ({', '.join(f'{k}={v}' for k, v in rej.items())})", details
        K = ch.strike[idx]
        sm = leg.strike
        is_call = leg.right == "C"
        if sm.method == "delta":
            target = abs(sm.value) if abs(sm.value) <= 1 else abs(sm.value) / 100.0
            dl = np.abs(ch.delta[idx])
            ok = np.isfinite(dl)
            if not ok.any():
                return None, f"leg {leg.id}: no deltas available", details
            j = idx[ok][np.argmin(np.abs(dl[ok] - target))]
        elif sm.method == "premium":
            j = idx[np.argmin(np.abs(ch.mid[idx] - sm.value))]
        elif sm.method == "pct_otm":
            target = spot * (1 + sm.value / 100.0) if is_call else spot * (1 - sm.value / 100.0)
            j = idx[np.argmin(np.abs(K - target))]
        elif sm.method == "fixed":
            j = idx[np.argmin(np.abs(K - sm.value))]
        elif sm.method == "atm_offset":
            all_k = np.unique(ch.strike[idx_all])
            if ch.kind == "trades":
                # selectively stored trade data only holds a band of strikes: if the underlying has left it,
                # the at-the-money contract is not stored and a neighbour must not be passed off as ATM
                half = (float(np.median(np.diff(all_k))) if all_k.size > 1 else 1.0) / 2 + 1e-9
                if not (all_k.min() - half <= spot <= all_k.max() + half):
                    return None, f"leg {leg.id}: the underlying ({spot:.2f}) is outside the stored strike range {all_k.min():g}-{all_k.max():g} for this session", details
            atm = all_k[np.argmin(np.abs(all_k - spot))]
            target = _offset(atm, sm.value, sm.unit, all_k)
            j = idx[np.argmin(np.abs(K - target))]
            if sm.unit == "points" and abs(ch.strike[j] - target) > _tol(sm.value):
                return None, f"leg {leg.id}: no tradable strike at ATM {sm.value:+g} ({target:g}); nearest {ch.strike[j]:g}", details
        else:  # leg_offset
            ref = chosen.get(sm.ref_leg)
            if ref is None or ref.row < 0:
                return None, f"leg {leg.id}: reference leg {sm.ref_leg} unresolved", details
            all_k = np.unique(ch.strike[idx_all])
            target = _offset(ref.strike, sm.value, sm.unit, all_k)
            j = idx[np.argmin(np.abs(K - target))]
            if sm.value != 0 and np.sign(ch.strike[j] - ref.strike) != np.sign(sm.value):
                return None, f"leg {leg.id}: no strike {sm.value:+g} {sm.unit} from leg {sm.ref_leg}", details
            if sm.unit != "strikes" and abs(ch.strike[j] - target) > _tol(target - ref.strike):
                return None, f"leg {leg.id}: no tradable strike at {sm.ref_leg} {sm.value:+g} {sm.unit} ({target:g}); nearest {ch.strike[j]:g}", details
        chosen[leg.id] = Selected(leg, int(j), float(ch.strike[j]), int(e), int(e - today_n))
    return [chosen[l.id] for l in legs], "", details


def _tol(offset: float) -> float:
    """Accept a strike within 10% of the requested offset (min $0.50): spreads keep their
    intended width instead of silently collapsing to a narrower one."""
    return max(0.5, 0.1 * abs(offset))


def _offset(base: float, value: float, unit: str, strikes: np.ndarray) -> float:
    if unit == "points":
        return base + value
    if unit == "pct":
        return base * (1 + value / 100.0)
    i = int(np.argmin(np.abs(strikes - base)))
    j = int(np.clip(i + int(round(value)), 0, len(strikes) - 1))
    return float(strikes[j])
