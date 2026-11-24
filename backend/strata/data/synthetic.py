"""SYNTHETIC demonstration dataset generator.

Everything produced here is simulated. Symbols are suffixed/named so they can never be
confused with real tickers (DEMO, DEMOX, DEMOVIX) and are flagged `synthetic=True` in the
symbol catalog; every backtest that touches them is labelled SYNTHETIC in the API and UI.

Model (deliberately richer than plain GBM so that strategies behave plausibly):
  * daily variance follows a mean-reverting square-root process with a leverage effect
  * overnight gaps carry ~18% of daily variance plus rare jumps; scheduled synthetic
    "CPI/NFP" days get larger gaps and "FOMC" days an afternoon volatility burst
  * intraday 1-minute returns are Student-t with a U-shaped intraday variance profile
  * implied vol = expected variance over the option's life (business time) times a
    variance risk premium, with put skew, a smile term and event variance for
    expirations spanning scheduled events (so IV crush after events is visible)
  * quotes are Black-Scholes mids with spreads that widen in the wings, at the open and
    in high-vol regimes; a small fraction of quotes are stale or missing, a few
    snapshots have feed outages and one session has no option data at all, so the
    engine's missing-data handling is exercised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.core.pricing import bs_greeks_vec
from strata.data.store import MarketStore, SymbolMeta

SYNTH_SOURCE = "synthetic"


@dataclass
class UnderlyingSpec:
    symbol: str
    s0: float
    style: str = "etf"  # etf -> physical/american ; index -> cash/european
    corr_to_first: float = 1.0
    theta_vol: float = 0.17
    kappa: float = 4.0
    xi: float = 0.55
    vrp: float = 0.12
    drift: float = 0.08
    fine_inc: float = 1.0
    fine_band: float = 0.04
    coarse_inc: float = 5.0
    max_dte: int = 100
    weekly_dte: int = 49


@dataclass
class SyntheticConfig:
    start: date = date(2023, 1, 3)
    end: date = date(2024, 12, 31)
    seed: int = 20240101
    option_interval: int = 15
    underlyings: list[UnderlyingSpec] = field(default_factory=lambda: [
        UnderlyingSpec("DEMO", 400.0, "etf", fine_band=0.08),
        UnderlyingSpec("DEMOX", 2000.0, "index", corr_to_first=0.72, theta_vol=0.21, drift=0.10, fine_inc=5.0, fine_band=0.03, coarse_inc=25.0, max_dte=40, weekly_dte=35),
    ])
    vol_symbol: str = "DEMOVIX"


def _event_calendar(days: list[date]) -> dict[date, list[tuple[str, str]]]:
    """Deterministic synthetic schedule loosely modelled on real release cadences."""
    out: dict[date, list[tuple[str, str]]] = {}
    dayset = set(days)
    months = sorted({(d.year, d.month) for d in days})

    def first_trading_on_or_after(d):
        while d not in dayset and d <= days[-1]:
            d += timedelta(days=1)
        return d if d in dayset else None

    fomc_months = {1: 4, 3: 3, 5: 1, 6: 2, 7: 4, 9: 3, 11: 1, 12: 2}  # week-of-month for a Wednesday
    for y, m in months:
        nfp = cal._nth_weekday(y, m, 4, 1)  # first Friday
        if nfp in dayset:
            out.setdefault(nfp, []).append(("NFP", "08:30"))
        cpi = first_trading_on_or_after(date(y, m, 12))
        if cpi:
            out.setdefault(cpi, []).append(("CPI", "08:30"))
        if m in fomc_months:
            f = cal._nth_weekday(y, m, 2, fomc_months[m])
            if f in dayset:
                out.setdefault(f, []).append(("FOMC", "14:00"))
    return out


EVENT_GAP_MULT = {"CPI": 2.2, "NFP": 1.8}
EVENT_VAR = {"CPI": 0.009 ** 2, "NFP": 0.007 ** 2, "FOMC": 0.008 ** 2}


def _intraday_weights(n: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / 390.0
    w = 1.0 + 1.6 * np.exp(-x / 0.04) + 0.7 * np.exp(-(n / 390.0 - x) / 0.06)
    return w / w.sum()


class _Sim:
    def __init__(self, cfg: SyntheticConfig, log=print):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.days = cal.trading_days(cfg.start, cfg.end)
        self.events = _event_calendar(self.days)
        self.log = log

    def simulate_underlyings(self):
        cfg, rng = self.cfg, self.rng
        nU = len(cfg.underlyings)
        res = {u.symbol: {"ts": [], "o": [], "h": [], "l": [], "c": [], "v": [], "var": [], "open_var": []} for u in cfg.underlyings}
        vix = {"ts": [], "o": [], "h": [], "l": [], "c": [], "v": []}
        var = np.array([u.theta_vol ** 2 for u in cfg.underlyings])
        price = np.array([u.s0 for u in cfg.underlyings])
        dt = 1.0 / 252
        corr = np.array([u.corr_to_first for u in cfg.underlyings])
        self.day_var = {u.symbol: {} for u in cfg.underlyings}
        self.vix_open = {}
        for di, d in enumerate(self.days):
            n = cal.session_minutes(d)
            w = _intraday_weights(n)
            ev = [e for e, _ in self.events.get(d, [])]
            gap_mult = max([EVENT_GAP_MULT.get(e, 1.0) for e in ev] + [1.0])
            open_ts = int(cal.session_bounds(d)[0].timestamp())
            ts = open_ts + 60 * np.arange(n)
            z_common = rng.standard_t(5, size=n) / math.sqrt(5 / 3)
            g_common = rng.standard_normal()
            jump = rng.random() < 0.02
            jsize = rng.normal(0, 0.018) if jump else 0.0
            day_ret_shock = np.zeros(nU)
            for k, u in enumerate(cfg.underlyings):
                v = var[k]
                self.day_var[u.symbol][d] = v
                rho = corr[k]
                z_ind = rng.standard_t(5, size=n) / math.sqrt(5 / 3)
                z = rho * z_common + math.sqrt(max(0.0, 1 - rho * rho)) * z_ind
                g = rho * g_common + math.sqrt(max(0.0, 1 - rho * rho)) * rng.standard_normal()
                gap = g * math.sqrt(0.18 * v * dt) * gap_mult + jsize * (1 if k == 0 else rho)
                sig = np.sqrt(0.82 * v * dt * w)
                if "FOMC" in ev:
                    i0 = min(n - 1, 270)
                    sig[i0:i0 + 60] *= 2.6
                drift = (u.drift - 0.5 * v) * dt * 0.82 / n
                r = drift + sig * z
                o0 = price[k] * math.exp(gap)
                closes = o0 * np.exp(np.cumsum(r))
                opens = np.empty(n)
                opens[0] = o0
                opens[1:] = closes[:-1]
                wick = np.abs(rng.standard_normal((2, n))) * sig * 0.35
                highs = np.maximum(opens, closes) * (1 + wick[0])
                lows = np.minimum(opens, closes) * (1 - wick[1])
                vol = (2e5 * w * n * rng.uniform(0.6, 1.4, n) * (1 + 3 * np.abs(r) / (sig + 1e-12) * 0.1)).round()
                R = res[u.symbol]
                R["ts"].append(ts); R["o"].append(opens); R["h"].append(highs); R["l"].append(lows); R["c"].append(closes); R["v"].append(vol)
                day_ret_shock[k] = (math.log(closes[-1] / price[k])) / math.sqrt(v * dt)
                price[k] = closes[-1]
            # vol index (based on first underlying)
            u0 = cfg.underlyings[0]
            v0 = var[0]
            tau = 30 / 365
            ev30 = u0.theta_vol ** 2 + (v0 - u0.theta_vol ** 2) * (1 - math.exp(-u0.kappa * tau)) / (u0.kappa * tau)
            vix_open = 100 * math.sqrt(ev30) * (1 + u0.vrp) * math.exp(rng.normal(0, 0.03))
            self.vix_open[d] = vix_open
            cum = np.log(res[u0.symbol]["c"][-1] / res[u0.symbol]["o"][-1][0])
            vix_c = vix_open * np.exp(-3.2 * cum + np.cumsum(rng.normal(0, 0.0012, n)))
            vix_o = np.r_[vix_open, vix_c[:-1]]
            vix["ts"].append(ts); vix["o"].append(vix_o); vix["c"].append(vix_c)
            vix["h"].append(np.maximum(vix_o, vix_c) * (1 + np.abs(rng.normal(0, 0.001, n))))
            vix["l"].append(np.minimum(vix_o, vix_c) * (1 - np.abs(rng.normal(0, 0.001, n))))
            vix["v"].append(np.zeros(n))
            # variance update with leverage effect
            for k, u in enumerate(cfg.underlyings):
                zv = -0.7 * day_ret_shock[k] + math.sqrt(1 - 0.49) * rng.standard_normal()
                vk = var[k] + u.kappa * (u.theta_vol ** 2 - var[k]) * dt + u.xi * math.sqrt(max(var[k], 1e-6)) * math.sqrt(dt) * zv
                var[k] = min(max(vk, 0.06 ** 2), 0.9 ** 2)
        self.bars = {}
        for sym, R in res.items():
            self.bars[sym] = {k: np.concatenate(R[k]) for k in ["ts", "o", "h", "l", "c", "v"]}
        self.bars[self.cfg.vol_symbol] = {k: np.concatenate(vix[k]) for k in ["ts", "o", "h", "l", "c", "v"]}

    # ---------------- options ----------------
    def expirations_for(self, d: date, u: UnderlyingSpec) -> list[date]:
        exps = set()
        k = 0
        dd = d
        while k < 3:
            if cal.is_trading_day(dd):
                exps.add(dd)
                k += 1
            dd += timedelta(days=1)
        horizon_w = d + timedelta(days=u.weekly_dte)
        f = d + timedelta(days=(4 - d.weekday()) % 7)
        while f <= horizon_w:
            e = f if cal.is_trading_day(f) else cal.prev_trading_day(f)
            if e >= d:
                exps.add(e)
            f += timedelta(days=7)
        horizon_m = d + timedelta(days=u.max_dte)
        y, m = d.year, d.month
        while True:
            tf = cal._nth_weekday(y, m, 4, 3)
            if tf > horizon_m:
                break
            e = tf if cal.is_trading_day(tf) else cal.prev_trading_day(tf)
            if e >= d:
                exps.add(e)
            m += 1
            if m > 12:
                y, m = y + 1, 1
        return sorted(e for e in exps if e <= self.days[-1] + timedelta(days=150))

    def _tau_business(self, d: date, minute_idx: np.ndarray, n_minutes: int, exp: date) -> np.ndarray:
        """Variance-time to the expiration close (years of 252 sessions)."""
        w = _intraday_weights(n_minutes)
        cumw = np.r_[0.0, np.cumsum(w)]
        rem_today = 0.82 * (1.0 - cumw[np.clip(minute_idx, 0, n_minutes)])
        if exp == d:
            return rem_today / 252.0
        nfull = cal.trading_days_between(d, exp)
        return (rem_today + nfull) / 252.0

    def generate_options(self, store: MarketStore, u: UnderlyingSpec, progress=None):
        cfg, rng = self.cfg, self.rng
        B = self.bars[u.symbol]
        V = self.bars[cfg.vol_symbol]
        strike_sets: dict[date, set[float]] = {}
        interval = cfg.option_interval
        outage_days = set(rng.choice(len(self.days), size=3, replace=False).tolist())
        missing_day = int(rng.integers(len(self.days) // 3, 2 * len(self.days) // 3))
        offset = 0
        prev_stale = None
        total_rows = 0
        for di, d in enumerate(self.days):
            n = cal.session_minutes(d)
            sl = slice(offset, offset + n)
            offset += n
            if di == missing_day:
                continue  # deliberately missing option session (underlying still present)
            o, c = B["o"][sl], B["c"][sl]
            vix_c, vix_o = V["c"][sl], V["o"][sl]
            snap_idx = np.arange(0, n + 1, interval)
            if snap_idx[-1] != n:
                snap_idx = np.r_[snap_idx, n]
            S_snap = np.where(snap_idx == 0, o[0], c[np.clip(snap_idx - 1, 0, n - 1)])
            vix_snap = np.where(snap_idx == 0, vix_o[0], vix_c[np.clip(snap_idx - 1, 0, n - 1)])
            open_ts = int(cal.session_bounds(d)[0].timestamp())
            ts_snap = open_ts + 60 * snap_idx
            v_day = self.day_var[u.symbol][d]
            v_t = v_day * (vix_snap / vix_snap[0]) ** 2
            exps = self.expirations_for(d, u)
            s_ref = o[0]
            day_events = sorted([(dd, e) for dd, evs in self.events.items() for e, _ in evs if d <= dd <= d + timedelta(days=u.max_dte + 5)])
            rows = []
            for e in exps:
                tau_open = self._tau_business(d, np.array([0]), n, e)[0]
                atm_guess = math.sqrt(u.theta_vol ** 2)
                band = min(0.12, max(0.035, 2.0 * atm_guess * math.sqrt(max(tau_open, 1 / 252))))
                lo, hi = s_ref * (1 - band), s_ref * (1 + band)
                fine_lo, fine_hi = s_ref * (1 - min(band, u.fine_band)), s_ref * (1 + min(band, u.fine_band))
                ks = set(np.arange(math.ceil(fine_lo / u.fine_inc), math.floor(fine_hi / u.fine_inc) + 1) * u.fine_inc)
                ks |= set(np.arange(math.ceil(lo / u.coarse_inc), math.floor(hi / u.coarse_inc) + 1) * u.coarse_inc)
                st = strike_sets.setdefault(e, set())
                st |= {round(float(k), 2) for k in ks}
                K = np.array(sorted(st))
                tau = self._tau_business(d, snap_idx, n, e)  # per snapshot
                ev_var = np.zeros_like(tau)
                for (dd, ename) in day_events:
                    if dd > e:
                        continue
                    t_ev = 8 * 60 + 30 if ename != "FOMC" else 14 * 60
                    ev_minute_from_open = t_ev - 9 * 60 - 30
                    if dd > d:
                        ev_var += EVENT_VAR.get(ename, 0.0)
                    elif dd == d:
                        ev_var += np.where(snap_idx < ev_minute_from_open, EVENT_VAR.get(ename, 0.0), 0.0)
                rows.append((e, K, tau, ev_var))
            # assemble vectorized arrays: snapshots x contracts
            all_K, all_tau, all_ev, all_exp, all_snap = [], [], [], [], []
            for (e, K, tau, ev_var) in rows:
                nk = K.shape[0]
                for si in range(snap_idx.shape[0]):
                    all_K.append(np.repeat(K, 2))
                    all_tau.append(np.full(2 * nk, tau[si]))
                    all_ev.append(np.full(2 * nk, ev_var[si]))
                    all_exp.append(np.full(2 * nk, (e - date(1970, 1, 1)).days, dtype=np.int32))
                    all_snap.append(np.full(2 * nk, si, dtype=np.int32))
            if not all_K:
                continue
            Kv = np.concatenate(all_K)
            tau = np.concatenate(all_tau)
            evv = np.concatenate(all_ev)
            expd = np.concatenate(all_exp)
            si = np.concatenate(all_snap)
            m = Kv.shape[0]
            is_call = np.tile(np.array([True, False]), m // 2)
            S = S_snap[si]
            vt = v_t[si]
            th = u.theta_vol ** 2
            tau_c = np.maximum(tau, 1e-9)
            ev_var_t = th + (vt - th) * (1 - np.exp(-u.kappa * tau_c)) / (u.kappa * tau_c)
            total_var = ev_var_t * tau + evv
            atm = np.sqrt(np.maximum(total_var, 1e-12) / tau_c) * (1 + u.vrp)
            x = np.clip(np.log(Kv / S) / (atm * np.sqrt(tau_c) + 1e-9), -8.0, 8.0)
            skew = 0.11 + 0.04 / (1 + 50 * tau_c)
            sig = atm * np.clip(1 - skew * x + 0.018 * x * x, 0.55, 3.5)
            G = bs_greeks_vec(S, Kv, tau, 0.0, 0.0, sig, is_call)
            mid = G[:, 0]
            tick = 0.01
            vixf = np.sqrt(vix_snap[si] / 18.0)
            openf = np.where(snap_idx[si] == 0, 2.2, np.where(snap_idx[si] <= 15, 1.35, 1.0))
            spread = (0.01 + 0.006 * mid + 0.002 * np.sqrt(np.maximum(mid, 0))) * (1 + 0.2 * np.abs(x)) * vixf * openf
            spread = np.maximum(tick, np.round(spread / tick) * tick)
            bid = np.floor((mid - spread / 2) / tick + 1e-9) * tick
            ask = np.ceil((mid + spread / 2) / tick - 1e-9) * tick
            bid = np.where(bid < tick, 0.0, bid)
            ask = np.maximum(ask, np.where(mid < 0.005, 0.01, bid + tick))
            ax = np.abs(x)
            dte_days = expd - (d - date(1970, 1, 1)).days
            bsz = (10 + 450 * np.exp(-0.6 * ax) * rng.uniform(0.4, 1.6, m)).astype(np.int32)
            asz = (10 + 450 * np.exp(-0.6 * ax) * rng.uniform(0.4, 1.6, m)).astype(np.int32)
            w = _intraday_weights(n)
            cumw = np.r_[0.0, np.cumsum(w)]
            # per-contract daily volume (deterministic per contract within the day)
            base_vol = 8000 * np.exp(-0.8 * ax) / (1 + dte_days / 4.0)
            vol = (base_vol * cumw[np.clip(snap_idx[si], 0, n)] * rng.uniform(0.9, 1.1, m)).astype(np.int64)
            oi = (25000 * np.exp(-0.45 * ax) * (0.5 + 0.5 * np.minimum(dte_days, 30) / 30)).astype(np.int64)
            ts = ts_snap[si].astype(np.int64)
            quote_ts = ts.copy()
            # stale quotes: copy the previous snapshot's quote for the same contract
            nsnap = snap_idx.shape[0]
            stale = (rng.random(m) < 0.004) & (si > 0)
            # rows are laid out expiration-major, then snapshot, then strike/right: the same
            # contract one snapshot earlier sits exactly 2*n_strikes rows before.
            block_sizes = []
            for (e, K, tau_e, _) in rows:
                block_sizes.append(np.full(2 * K.shape[0] * nsnap, 2 * K.shape[0]))
            stride = np.concatenate(block_sizes)
            idx_prev = np.where(stale, np.arange(m) - stride, np.arange(m))
            bid, ask = bid[idx_prev], ask[idx_prev]
            quote_ts = ts[idx_prev]
            iv = sig[idx_prev]
            delta, gamma, theta, vega = G[idx_prev, 1], G[idx_prev, 2], G[idx_prev, 3] * 365 / 252, G[idx_prev, 4]
            keep = (rng.random(m) >= 0.0015) & (tau > 0)  # no quotes at/after a contract's own expiration close
            if di in outage_days:
                bad = (snap_idx[si] >= 120) & (snap_idx[si] < 120 + 2 * interval)
                keep &= ~bad
            df = pl.DataFrame({
                "ts": ts[keep], "expiration": expd[keep], "strike": Kv[keep],
                "right": np.where(is_call[keep], "C", "P"), "bid": np.round(bid[keep], 2), "ask": np.round(ask[keep], 2),
                "bid_size": bsz[keep], "ask_size": asz[keep], "volume": vol[keep],
                "open_interest": oi[keep], "iv": iv[keep], "delta": delta[keep], "gamma": gamma[keep], "theta": theta[keep],
                "vega": vega[keep], "underlying_price": np.round(S[keep], 4), "quote_ts": quote_ts[keep],
            }).with_columns(
                pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                pl.from_epoch("quote_ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
                pl.col("expiration").cast(pl.Date),
            )
            store.write_option_day(u.symbol, d, df)
            total_rows += df.height
            # drop expired strike sets
            for e in [e for e in strike_sets if e < d]:
                del strike_sets[e]
            if progress and di % 20 == 0:
                progress(u.symbol, di, len(self.days))
        return total_rows


def generate_demo_dataset(store: MarketStore | None = None, cfg: SyntheticConfig | None = None, log=print, progress=None) -> dict:
    from strata.data.derived import derive_iv_history

    store = store or MarketStore()
    cfg = cfg or SyntheticConfig()
    sim = _Sim(cfg, log)
    log(f"[synthetic] simulating {len(sim.days)} sessions for {[u.symbol for u in cfg.underlyings]}")
    sim.simulate_underlyings()
    for sym, B in sim.bars.items():
        store.remove_symbol(sym)
        df = pl.DataFrame({"ts": B["ts"], "open": B["o"], "high": B["h"], "low": B["l"], "close": B["c"], "volume": B["v"]}).with_columns(
            pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
            *[pl.col(c).round(4) for c in ["open", "high", "low", "close"]],
        )
        store.write_minutes(sym, df)
        from strata.data.store import minutes_to_daily

        store.write_daily(sym, minutes_to_daily(df))
    store.upsert_symbol(SymbolMeta(
        symbol=cfg.vol_symbol, name="Synthetic volatility index", kind="volatility_index", exercise="european", settlement="cash",
        provider=SYNTH_SOURCE, synthetic=True, has_options=False,
        description="SYNTHETIC 30-day implied volatility index derived from the DEMO variance process.",
    ))
    rows = {}
    for u in cfg.underlyings:
        is_index = u.style == "index"
        store.upsert_symbol(SymbolMeta(
            symbol=u.symbol,
            name=("Synthetic cash-settled index" if is_index else "Synthetic ETF-like underlying"),
            kind="index" if is_index else "etf",
            exercise="european" if is_index else "american",
            settlement="cash" if is_index else "physical",
            settlement_time="PM", multiplier=100.0, vol_index=cfg.vol_symbol, provider=SYNTH_SOURCE, synthetic=True,
            has_options=True, option_interval_minutes=cfg.option_interval,
            description=("SYNTHETIC data generated by strata.data.synthetic (seed %d). Not real market history." % cfg.seed),
            extra={"seed": cfg.seed, "start": cfg.start.isoformat(), "end": cfg.end.isoformat(), "model": "sqrt-variance + t-shocks + skewed BS quotes"},
        ))
        log(f"[synthetic] generating option chains for {u.symbol}")
        rows[u.symbol] = sim.generate_options(store, u, progress=progress)
        derive_iv_history(store, u.symbol)
    ev_rows = [{"date": d, "time": t, "event_type": e, "description": f"SYNTHETIC {e} release", "source": SYNTH_SOURCE} for d, evs in sim.events.items() for e, t in evs]
    store.write_events(pl.DataFrame(ev_rows), replace_source=SYNTH_SOURCE)
    for sym in list(sim.bars.keys()):
        store.rebuild_manifest(sym)
    return {"sessions": len(sim.days), "option_rows": rows, "events": len(ev_rows)}
