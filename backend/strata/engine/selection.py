"""Contract selection: turn declarative leg specs into concrete contracts using only the
chain snapshot observable at the decision time."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from strata.core import calendar as cal
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


_EFFECTIVE: dict[int, int] = {}
_MOVED: set[int] = set()


def effective_expiration(e: int) -> int:
    """The day (a day number) on which a contract dated `e` stops trading and is settled.

    Its own date when the market is open that day. When the market turned out to be closed on it, the last session before:
    the exchanges move such expirations forward and the contracts keep their symbols (2025-01-09, the national day of
    mourning: contracts dated the 9th expired at the close of the 8th). A leg dated on a closed day and left to expire "on
    its date" never expired at all: found on the stored SPY history, where one such call stayed open for two years, marked
    at intrinsic value, and put $12,823 into a run's net result that none of its trades had made."""
    e = int(e)
    v = _EFFECTIVE.get(e)
    if v is None:
        d = from_day_num(e)
        v = e if cal.is_trading_day(d) else int(day_num(cal.prev_trading_day(d)))
        _EFFECTIVE[e] = v
        if v != e:
            _MOVED.add(e)
    return v


def _effective(exps: np.ndarray) -> np.ndarray:
    """`exps` with every date on which the market is closed replaced by the session before it (the same array when none is)."""
    xs = exps.tolist()
    for x in xs:
        if x not in _EFFECTIVE:
            effective_expiration(x)
    if not _MOVED or _MOVED.isdisjoint(xs):
        return exps
    return np.array([_EFFECTIVE[x] for x in xs], dtype=exps.dtype)


def _choose_expiration(exps: np.ndarray, today: int, leg: LegSpec, dated: bool = True) -> int | None:
    """The expiration among `exps` that the leg asks for. `exps` and `today` are day numbers of real dates, unless `dated`
    is False (distances from a day nobody named: the calendar cannot be asked about those)."""
    if leg.expiration is not None:
        # an exact date is that expiration or nothing: never the nearest one that happens to be listed
        e = day_num(leg.expiration)
        return int(e) if e >= today and bool((exps == e).any()) else None
    dte = (_effective(exps) if dated else exps) - today  # days until the contract really expires, which is what "1DTE" asks about
    ok = dte >= 0
    if leg.dte_min is not None:
        ok &= dte >= leg.dte_min
    if leg.dte_max is not None:
        ok &= dte <= leg.dte_max
    cand = exps[ok]
    if cand.size == 0:
        return None
    d = dte[ok]
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
    # trade data that was downloaded selectively holds a band of strikes and a few expirations, and at any moment only the
    # contracts that have printed. A leg may be filled only with the contract it asked for: when the stored data cannot show
    # that a contract is that one, the signal is left unfilled and says why (never the nearest thing that happens to be stored).
    known = getattr(ch, "complete", None)
    selective = (ch.kind == "trades") if known is None else not known  # trade bars are a band unless the dataset says it is whole; quotes are whole unless it says it is a band
    order = [l for l in legs if l.strike.method != "leg_offset"] + [l for l in legs if l.strike.method == "leg_offset"]
    for leg in order:
        if leg.kind == "stock":
            chosen[leg.id] = Selected(leg, -1, 0.0, -1, 0)
            continue
        e = _choose_expiration(alive, today_n, leg)
        if e is None:
            listed = sorted(int(x) for x in (alive - today_n)) if alive.size else []
            if leg.expiration is not None:
                if day_num(leg.expiration) < today_n:
                    return None, f"leg {leg.id}: no {leg.expiration} expiration in the stored chain (it expired before this session)", details
                near_e = min((int(x) for x in alive), key=lambda x: abs(x - day_num(leg.expiration))) if alive.size else None
                return None, (f"leg {leg.id}: no {leg.expiration} expiration in the stored chain" + (f" (nearest stored: {from_day_num(near_e)})" if near_e is not None else "")), details
            near = min(listed, key=lambda x: abs(x - leg.dte)) if listed else None
            exact = leg.dte_mode == "exact" or leg.dte == 0
            return None, (f"leg {leg.id}: no {'exact ' if exact else ''}{leg.dte}DTE expiration in the stored chain" + (f" (nearest listed: {near}DTE)" if near is not None else "")), details
        days_left = effective_expiration(e) - today_n
        if selective and leg.expiration is None and leg.dte_mode != "exact" and leg.dte != 0 and abs(int(days_left) - leg.dte) > dte_tolerance(leg.dte):
            return None, f"leg {leg.id}: no expiration within {dte_tolerance(leg.dte)} days of {leg.dte}DTE in the stored chain (nearest stored: {int(days_left)}DTE)", details
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
        all_k = np.unique(ch.strike[idx_all])
        half = (float(np.median(np.diff(all_k))) if all_k.size > 1 else 1.0) / 2 + 1e-9
        k_lo, k_hi = float(all_k.min()), float(all_k.max())
        beyond = lambda what: (None, f"leg {leg.id}: {what} lies beyond the stored strike range {k_lo:g}-{k_hi:g} for this session", details)  # noqa: E731
        if sm.method in ("delta", "premium"):
            if sm.method == "delta":
                target = abs(sm.value) if abs(sm.value) <= 1 else abs(sm.value) / 100.0
                val = np.abs(ch.delta[idx])
                label, fmt = f"|delta| {target:g}", (lambda x: f"|delta| {x:.2f}")
            else:
                target = float(sm.value)
                val = ch.mid[idx].astype(float)
                label, fmt = f"a premium near {target:g}", (lambda x: f"a last trade at {x:.2f}")
            ok = np.isfinite(val)
            if not ok.any():
                return None, f"leg {leg.id}: no {'deltas' if sm.method == 'delta' else 'prices'} available", details
            cand, v = idx[ok], val[ok]
            p = int(np.argmin(np.abs(v - target)))
            j = cand[p]
            if selective:
                # the band of strikes that is stored must be able to show that this IS the contract asked for
                err = abs(float(v[p]) - target)
                near = err <= 0.10 if sm.method == "delta" else (target > 0 and err / target <= 0.25)
                above, below = v >= target, v <= target
                grid_best = False
                if above.any() and below.any():
                    # prints on both sides of the target: if no stored strike lies between the two nearest, nothing closer exists
                    ka = float(ch.strike[cand[above][np.argmin(v[above])]])
                    kb = float(ch.strike[cand[below][np.argmax(v[below])]])
                    grid_best = not bool(((all_k > min(ka, kb) + 1e-9) & (all_k < max(ka, kb) - 1e-9)).any())
                if not (near or grid_best):
                    k_sel = float(ch.strike[j])
                    if not (above.any() and below.any()):
                        # every recent print is on one side of the target: the contract asked for is further out of the money
                        # (all too rich) or further in the money (all too cheap). Is a stored strike left in that direction?
                        outward = bool(above.all()) == is_call  # too rich: calls need a higher strike, puts a lower one
                        ks_c = ch.strike[cand]
                        if (outward and float(ks_c.max()) >= k_hi - 1e-9) or (not outward and float(ks_c.min()) <= k_lo + 1e-9):
                            return beyond(f"{label} (the outermost stored strike {float(ks_c.max()) if outward else float(ks_c.min()):g} has {fmt(float(v[np.argmax(ks_c)] if outward else v[np.argmin(ks_c)]))})")
                    return None, f"leg {leg.id}: no contract near {label} has a usable trade print at this time (the nearest with one is the {k_sel:g} strike, {fmt(float(v[p]))})", details
        elif sm.method == "pct_otm":
            target = spot * (1 + sm.value / 100.0) if is_call else spot * (1 - sm.value / 100.0)
            j = idx[np.argmin(np.abs(K - target))]
            if selective:
                if not (k_lo - half <= target <= k_hi + half):
                    return beyond(f"{sm.value:g}% out of the money is strike {target:.2f}, which")
                why = _next_to(all_k, target, float(ch.strike[j]), f"{sm.value:g}% out of the money (strike {target:.2f})")
                if why:
                    return None, f"leg {leg.id}: {why}", details
        elif sm.method == "fixed":
            j = idx[np.argmin(np.abs(K - sm.value))]
            if selective and abs(float(ch.strike[j]) - sm.value) > half:
                if k_lo - half <= sm.value <= k_hi + half:
                    return None, f"leg {leg.id}: strike {sm.value:g} has no usable trade print at this time (nearest printed: {float(ch.strike[j]):g})", details
                return beyond(f"strike {sm.value:g}")
        elif sm.method == "atm_offset":
            if selective:
                # selectively stored data only holds a band of strikes: if the underlying has left it,
                # the at-the-money contract is not stored and a neighbour must not be passed off as ATM
                if not (k_lo - half <= spot <= k_hi + half):
                    return None, f"leg {leg.id}: the underlying ({spot:.2f}) is outside the stored strike range {k_lo:g}-{k_hi:g} for this session", details
            atm = all_k[np.argmin(np.abs(all_k - spot))]
            if selective and sm.unit == "strikes" and not (0 <= int(np.argmin(np.abs(all_k - atm))) + int(round(sm.value)) < all_k.size):
                return beyond(f"the strike {sm.value:+g} strikes from the money")
            target = _offset(atm, sm.value, sm.unit, all_k)
            j = idx[np.argmin(np.abs(K - target))]
            if sm.unit == "points" and abs(ch.strike[j] - target) > _tol(sm.value):
                return None, f"leg {leg.id}: no tradable strike at ATM {sm.value:+g} ({target:g}); nearest {ch.strike[j]:g}", details
            if selective and sm.unit == "strikes" and abs(float(ch.strike[j]) - target) > 1e-6:
                return None, f"leg {leg.id}: strike {target:g} (the money {sm.value:+g} strikes) has no usable trade print at this time (nearest printed: {float(ch.strike[j]):g})", details
            if selective and sm.unit == "pct":
                if not (k_lo - half <= target <= k_hi + half):
                    return beyond(f"the money {sm.value:+g}% is strike {target:.2f}, which")
                why = _next_to(all_k, target, float(ch.strike[j]), f"the money {sm.value:+g}% (strike {target:.2f})")
                if why:
                    return None, f"leg {leg.id}: {why}", details
        else:  # leg_offset
            ref = chosen.get(sm.ref_leg)
            if ref is None or ref.row < 0:
                return None, f"leg {leg.id}: reference leg {sm.ref_leg} unresolved", details
            if selective and sm.unit == "strikes" and not (0 <= int(np.argmin(np.abs(all_k - ref.strike))) + int(round(sm.value)) < all_k.size):
                return beyond(f"the strike {sm.value:+g} strikes from leg {sm.ref_leg}")
            target = _offset(ref.strike, sm.value, sm.unit, all_k)
            j = idx[np.argmin(np.abs(K - target))]
            if sm.value != 0 and np.sign(ch.strike[j] - ref.strike) != np.sign(sm.value):
                return None, f"leg {leg.id}: no strike {sm.value:+g} {sm.unit} from leg {sm.ref_leg}", details
            if sm.unit != "strikes" and abs(ch.strike[j] - target) > _tol(target - ref.strike):
                return None, f"leg {leg.id}: no tradable strike at {sm.ref_leg} {sm.value:+g} {sm.unit} ({target:g}); nearest {ch.strike[j]:g}", details
            if selective and sm.unit == "strikes" and abs(float(ch.strike[j]) - target) > 1e-6:
                return None, f"leg {leg.id}: strike {target:g} ({sm.ref_leg} {sm.value:+g} strikes) has no usable trade print at this time (nearest printed: {float(ch.strike[j]):g})", details
        chosen[leg.id] = Selected(leg, int(j), float(ch.strike[j]), int(e), int(days_left))
        if sm.method != "leg_offset":
            ks = np.unique(ch.strike[idx_all])
            k_sel = float(ch.strike[j])
            pick = {"method": sm.method, "strike": k_sel, "stored_low": float(ks.min()), "stored_high": float(ks.max()), "stored_strikes": int(ks.size), "expiration": int(e),
                    "edge": None if ks.size < 2 or sm.method == "fixed" else "low" if k_sel <= ks.min() else "high" if k_sel >= ks.max() else None}
            if sm.method == "delta":
                pick.update(target=float(abs(sm.value) if abs(sm.value) <= 1 else abs(sm.value) / 100.0), achieved=float(abs(ch.delta[j])) if np.isfinite(ch.delta[j]) else None)
            elif sm.method == "premium":
                pick.update(target=float(sm.value), achieved=float(ch.mid[j]) if np.isfinite(ch.mid[j]) else None)
            elif sm.method == "pct_otm":
                pick.update(target=float(spot * (1 + sm.value / 100.0) if is_call else spot * (1 - sm.value / 100.0)), achieved=k_sel)
            details.setdefault(leg.id, {})["pick"] = pick
    return [chosen[l.id] for l in legs], "", details


def dte_tolerance(dte: int) -> int:
    """On selectively stored data: how far the chosen expiration may be from the days-to-expiration a leg asked for.
    Listed expirations are at most a few days apart near the front and weeks apart further out, so the nearest LISTED one is
    within about a quarter of the request; further away than that, what is stored is simply not the expiration asked for."""
    return max(6, int(round(0.25 * dte)))


def _next_to(strikes: np.ndarray, target: float, chosen_k: float, what: str) -> str:
    """'' when the chosen strike is one of the two stored strikes either side of the target, else why it is not acceptable."""
    below, above = strikes[strikes <= target + 1e-9], strikes[strikes >= target - 1e-9]
    a = float(below.max()) if below.size else float(strikes.min())
    b = float(above.min()) if above.size else float(strikes.max())
    if abs(chosen_k - a) < 1e-9 or abs(chosen_k - b) < 1e-9:
        return ""
    return f"neither stored strike next to {what} ({a:g}{'' if a == b else f' and {b:g}'}) has a usable trade print at this time (the nearest with one is {chosen_k:g})"


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
