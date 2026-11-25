"""Strategy-builder preview: resolve legs against a real stored chain snapshot and return
payoff curves, Greeks, premium and buying-power figures.

The selected contracts, quotes, Greeks and margin are historical/observed. The payoff
curves are by nature theoretical: expiration payoff is exact for legs expiring on the
front date; later-dated legs and the T+0 curve use Black-Scholes with each leg's observed
IV and are labelled as estimates."""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np

from strata.core import calendar as cal
from strata.core.pricing import bs_greeks_vec, bs_price_vec
from strata.data.store import MarketStore
from strata.engine.execution import fill_price
from strata.engine.margin import MLeg, max_profit_loss, requirement
from strata.engine.market import DayChain, UnderlyingSeries, exp_close_epoch, from_day_num
from strata.engine.models import Execution, StrategyConfig
from strata.engine.selection import select_contracts


def preview(strategy: dict, day: date | None, time: str, execution: dict | None) -> dict:
    cfg = StrategyConfig.model_validate(strategy)
    ex = Execution.model_validate(execution or {})
    st = MarketStore()
    days = st.option_days(cfg.symbol)
    if not days:
        return {"error": f"no option data stored for {cfg.symbol}"}
    cands = [d for d in days if not day or d <= day]
    d = cands[-1] if cands else days[0]
    df = st.read_option_day(cfg.symbol, d)
    ch = DayChain(df)
    t = int(datetime.combine(d, datetime.strptime(time, "%H:%M").time(), tzinfo=cal.ET).timestamp())
    si = ch.snapshot_at(t)
    if si < 0:
        si = 0
    t = int(ch.snap_ts[si])
    m = st.read_minutes(cfg.symbol, d, d)
    spot = float("nan")
    if not m.is_empty():
        spot = UnderlyingSeries(m).price_at(t, d)
    if not math.isfinite(spot):
        u = ch.under[ch.rows(si)]
        spot = float(u[np.isfinite(u)][0])
    sel, why, details = select_contracts(ch, si, cfg.legs, spot, d, max_quote_age=10**9)
    fallback_note = None
    if sel is None:
        # the requested snapshot cannot fill every leg (e.g. a quote is missing); show the
        # next snapshot that can, and say so
        for sj in range(si + 1, min(si + 6, len(ch.snap_ts))):
            tj = int(ch.snap_ts[sj])
            sp = UnderlyingSeries(m).price_at(tj, d) if not m.is_empty() else spot
            s2, w2, d2 = select_contracts(ch, sj, cfg.legs, sp, d, max_quote_age=10**9)
            if s2 is not None:
                fallback_note = f"No fill at the requested time ({why}); showing the next snapshot that resolves."
                sel, si, t, spot, details = s2, sj, tj, sp, d2
                break
    meta = st.symbol_meta(cfg.symbol)
    base = {"date": d.isoformat(), "time": cal.fmt_hhmm(cal.minute_of_day_et(t * 1_000_000_000)), "ts": t, "spot": spot, "symbol": cfg.symbol, "fallback": fallback_note,
            "synthetic": st.is_synthetic(cfg.symbol), "available_days": [days[0].isoformat(), days[-1].isoformat()]}
    if sel is None:
        return base | {"error": why, "details": details}
    mult = meta.multiplier
    legs, mlegs = [], []
    cash_c = cash_m = 0.0
    greeks = np.zeros(4)
    shares = 0.0
    for s in sel:
        if s.row < 0:
            q = s.leg.sign * s.leg.quantity
            shares += q
            cash_c -= q * spot
            cash_m -= q * spot
            greeks[0] += q
            legs.append({"id": s.leg.id, "kind": "stock", "side": s.leg.side, "qty": q, "price": spot})
            continue
        r = ch.row_dict(s.row)
        q = s.leg.sign * s.leg.quantity
        pc = fill_price(r["bid"], r["ask"], s.leg.sign, ex)
        cash_c -= q * mult * pc
        cash_m -= q * mult * r["mid"]
        for gi, gk in enumerate(["delta", "gamma", "theta", "vega"]):
            greeks[gi] += q * mult * (r[gk] or 0.0)
        mlegs.append(MLeg(r["right"] == "C", r["strike"], q, mult, r["mid"]))
        legs.append({"id": s.leg.id, "kind": "option", "side": s.leg.side, "qty": q, "dte": s.dte, "fill": pc, **r})
    R = requirement(mlegs, shares, spot, "regt")
    R_cash = requirement(mlegs, shares, spot, "cash")
    exps = sorted({l["expiration"] for l in legs if l["kind"] == "option"})
    front = exps[0] if exps else d.isoformat()
    front_close = int(cal.session_bounds(date.fromisoformat(front))[1].timestamp())
    iv_atm = np.nanmean([l["iv"] for l in legs if l["kind"] == "option" and l.get("iv")]) if legs else 0.2
    T_front = max((front_close - t) / (365 * 86400), 1 / (365 * 24))
    width = max(0.03, 3.5 * (iv_atm if math.isfinite(iv_atm) else 0.2) * math.sqrt(T_front))
    ks = [l["strike"] for l in legs if l["kind"] == "option"]
    if ks:
        width = max(width, max(abs(k / spot - 1) for k in ks) * 1.4)
    S = np.linspace(spot * (1 - width), spot * (1 + width), 241)
    exp_val = np.zeros_like(S) + shares * S
    t0_val = np.zeros_like(S) + shares * S
    estimated = False
    from strata.core.pricing import implied_vol_scalar
    for l in legs:
        if l["kind"] != "option":
            continue
        is_call = l["right"] == "C"
        K = np.full_like(S, l["strike"])
        ec = exp_close_epoch(np.array([(date.fromisoformat(l["expiration"]) - date(1970, 1, 1)).days]))[0]
        T_now = max((ec - t) / (365 * 86400), 0.0)
        # re-imply vol from the leg's own mid with the curve's time convention so the T+0
        # curve passes through the observed mark at the current spot
        iv_cal = implied_vol_scalar(l["mid"], spot, l["strike"], max(T_now, 1e-6), 0.0, 0.0, is_call)
        sig = np.full_like(S, iv_cal if math.isfinite(iv_cal) else (l["iv"] or iv_atm))
        t0_val += l["qty"] * mult * bs_price_vec(S, K, np.full_like(S, T_now), 0.0, 0.0, sig, np.full(S.shape, is_call))
        if l["expiration"] == front:
            exp_val += l["qty"] * mult * (np.maximum(S - l["strike"], 0) if is_call else np.maximum(l["strike"] - S, 0))
        else:
            estimated = True
            T_rem = max((ec - front_close) / (365 * 86400), 0.0)
            exp_val += l["qty"] * mult * bs_price_vec(S, K, np.full_like(S, T_rem), 0.0, 0.0, sig, np.full(S.shape, is_call))
    pnl_exp = exp_val + cash_c
    pnl_t0 = t0_val + cash_c
    be = []
    for i in range(len(S) - 1):
        a, b = pnl_exp[i], pnl_exp[i + 1]
        if a == 0 or a * b < 0:
            be.append(float(S[i] - a * (S[i + 1] - S[i]) / (b - a)) if b != a else float(S[i]))
    single = len(exps) <= 1
    mp, ml = max_profit_loss(mlegs, shares, cash_c) if single else (float(pnl_exp.max()), float(-pnl_exp.min()))
    # probability of profit at the front expiration: lognormal whose total volatility is
    # implied by the ATM straddle (straddle ~ 0.8 * S * sigma*sqrt(T)), an estimate
    sl = ch.rows(si)
    fe = (date.fromisoformat(front) - date(1970, 1, 1)).days
    idx = np.arange(sl.start, sl.stop)[ch.exp[sl] == fe]
    w = None
    if idx.size:
        k_atm = ch.strike[idx][np.argmin(np.abs(ch.strike[idx] - spot))]
        both = idx[ch.strike[idx] == k_atm]
        if both.size == 2:
            w = float(ch.mid[both].sum()) / (0.7979 * spot)
    if not w or not math.isfinite(w) or w <= 0:
        w = (iv_atm if math.isfinite(iv_atm) else 0.2) * math.sqrt(T_front)
    z = np.log(S / spot) / w + 0.5 * w
    from scipy.stats import norm
    cdf = norm.cdf(z)
    dens = np.diff(np.r_[0.0, cdf, 1.0])[:len(S)]
    pop = float(dens[pnl_exp > 0].sum())
    return base | {
        "legs": legs, "credit_conservative": cash_c, "credit_mid": cash_m, "per_unit": cash_c / mult,
        "greeks": {"delta": greeks[0], "gamma": greeks[1], "theta": greeks[2], "vega": greeks[3]},
        "margin": {"regt_requirement": R, "regt_bpr": max(0.0, R - cash_c) if math.isfinite(R) else None,
                   "cash_requirement": R_cash if math.isfinite(R_cash) else None, "cash_bpr": max(0.0, R_cash - cash_c) if math.isfinite(R_cash) else None},
        "max_profit": mp if math.isfinite(mp) else None, "max_loss": ml if math.isfinite(ml) else None, "breakevens": be, "pop_estimate": pop,
        "payoff": {"x": S.tolist(), "expiration": pnl_exp.tolist(), "t0": pnl_t0.tolist(), "front_expiration": front, "estimated": estimated},
        "notes": ([fallback_note] if fallback_note else []) + ["Contracts, quotes, Greeks and margin come from the stored chain snapshot.",
                  "Payoff at expiration is exact for front-expiry legs; T+0 and later-dated legs use Black-Scholes with observed IVs (estimate).",
                  "Probability of profit is a lognormal estimate using the ATM straddle, not a historical frequency."],
        "selection_details": details,
    }
