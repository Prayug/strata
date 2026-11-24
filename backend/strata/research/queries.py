"""Market research workspace: answers historical questions directly from stored data.

Every query returns {title, description, sample_size, stats, tables, charts, notes}. Charts
are generic specs rendered by the UI. Confidence intervals: Wilson for frequencies,
bootstrap (percentile, 2000 resamples, fixed seed) for means.
"""
from __future__ import annotations

import math
import threading
from datetime import date, timedelta

import duckdb
import numpy as np
import polars as pl
from scipy import stats as sst

from strata.core import calendar as cal
from strata.data.store import MarketStore


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 1) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n, x.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    lo, hi = boot_ci(x)
    return {"n": int(x.size), "mean": float(x.mean()), "mean_ci95": [lo, hi], "median": float(np.median(x)), "std": float(x.std(ddof=1)) if x.size > 1 else None,
            "p05": float(np.quantile(x, 0.05)), "p25": float(np.quantile(x, 0.25)), "p75": float(np.quantile(x, 0.75)), "p95": float(np.quantile(x, 0.95)),
            "min": float(x.min()), "max": float(x.max())}


def hist_chart(x: np.ndarray, title: str, xlabel: str, bins: int = 40, compare: np.ndarray | None = None, compare_name: str = "all days") -> dict:
    x = x[np.isfinite(x)]
    allx = x if compare is None else np.concatenate([x, compare[np.isfinite(compare)]])
    if allx.size == 0:
        return {"type": "histogram", "title": title, "x": [], "series": []}
    lo, hi = np.quantile(allx, [0.005, 0.995])
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    series = [{"name": "selected", "data": (np.histogram(np.clip(x, lo, hi), edges)[0] / max(x.size, 1)).tolist()}]
    if compare is not None:
        c = compare[np.isfinite(compare)]
        series.append({"name": compare_name, "data": (np.histogram(np.clip(c, lo, hi), edges)[0] / max(c.size, 1)).tolist()})
    return {"type": "histogram", "title": title, "x": ((edges[:-1] + edges[1:]) / 2).tolist(), "series": series, "xLabel": xlabel, "yLabel": "frequency"}


class Research:
    def __init__(self, store: MarketStore | None = None):
        self.store = store or MarketStore()

    # ------------------------------------------------------------------ day table
    def day_table(self, sym: str, start: date | None, end: date | None) -> pl.DataFrame:
        """One row per session with features observable at the open (prev_*, gap, vix_prev)
        and outcomes realised later in the day (labelled as such)."""
        m = self.store.read_minutes(sym, start - timedelta(days=400) if start else None, end)
        if m.is_empty():
            raise ValueError(f"no price data for {sym}")
        m = m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("et")).with_columns(
            pl.col("et").dt.date().alias("date"), (pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32) - 570).alias("mfo"))
        d = m.group_by("date", maintain_order=True).agg(
            pl.col("open").first().alias("open"), pl.col("high").max().alias("high"), pl.col("low").min().alias("low"), pl.col("close").last().alias("close"),
            *[pl.col("close").filter(pl.col("mfo") < h).last().alias(f"px_{h}") for h in (15, 30, 60, 120, 240)],
        ).sort("date")
        d = d.with_columns(
            pl.col("close").shift(1).alias("prev_close"), pl.col("high").shift(1).alias("prev_high"), pl.col("low").shift(1).alias("prev_low"),
        ).with_columns(
            ((pl.col("open") / pl.col("prev_close") - 1) * 100).alias("gap_pct"),
            ((pl.col("close") / pl.col("open") - 1) * 100).alias("oc_ret"),
            ((pl.col("close") / pl.col("prev_close") - 1) * 100).alias("cc_ret"),
            ((pl.col("high") - pl.col("low")) / pl.col("open") * 100).alias("range_pct"),
            ((pl.col("prev_close") / pl.col("close").shift(2) - 1) * 100).alias("prev_ret"),
            ((pl.col("prev_high") - pl.col("prev_low")) / pl.col("prev_close") * 100).alias("prev_range_pct"),
            pl.col("close").rolling_mean(50).shift(1).alias("sma50"), pl.col("close").rolling_mean(200).shift(1).alias("sma200"),
            pl.col("date").dt.weekday().alias("weekday"),
        )
        for h in (15, 30, 60, 120, 240):
            d = d.with_columns(((pl.col(f"px_{h}") / pl.col("open") - 1) * 100).alias(f"move_{h}"))
        meta = self.store.symbol_meta(sym)
        vs = meta.vol_index
        if vs:
            vd = self.store.read_daily(vs)
            if not vd.is_empty():
                d = d.join(vd.select(pl.col("date"), pl.col("close").alias("vix_close"), pl.col("open").alias("vix_open")), on="date", how="left").with_columns(
                    pl.col("vix_close").shift(1).alias("vix_prev"))
        if "vix_prev" not in d.columns:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("vix_prev"), pl.lit(None, dtype=pl.Float64).alias("vix_close"))
        iv = self.store.read_iv_history(sym)
        if not iv.is_empty():
            d = d.join(iv.select("date", "atm_iv_30d", "atm_iv_front"), on="date", how="left").with_columns(
                pl.col("atm_iv_30d").shift(1).alias("iv_prev"), pl.col("atm_iv_front").shift(1).alias("iv_front_prev"))
            r = d["iv_prev"].to_numpy().astype(float)
            rank = np.full(r.size, np.nan)
            for i in range(r.size):
                w = r[max(0, i - 252):i]
                w = w[np.isfinite(w)]
                if w.size >= 20 and np.isfinite(r[i]) and w.max() > w.min():
                    rank[i] = (r[i] - w.min()) / (w.max() - w.min()) * 100
            d = d.with_columns(pl.Series("iv_rank_prev", rank))
        ev = self.store.read_events()
        if not ev.is_empty():
            evd = ev.group_by("date").agg(pl.col("event_type").unique().alias("events"))
            d = d.join(evd, on="date", how="left")
        else:
            d = d.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias("events"))
        d = d.with_columns(
            pl.when((pl.col("prev_close") > pl.col("sma200")) & (pl.col("sma50") > pl.col("sma200"))).then(pl.lit("bull"))
            .when((pl.col("prev_close") < pl.col("sma200")) & (pl.col("sma50") < pl.col("sma200"))).then(pl.lit("bear"))
            .when(pl.col("sma200").is_null()).then(pl.lit("unknown")).otherwise(pl.lit("neutral")).alias("trend"))
        if start:
            d = d.filter(pl.col("date") >= start)
        if end:
            d = d.filter(pl.col("date") <= end)
        return d

    @staticmethod
    def apply_filters(d: pl.DataFrame, filters: list[dict]) -> tuple[pl.DataFrame, list[str]]:
        desc = []
        for f in filters or []:
            fld, op, v = f.get("field"), f.get("op", ">"), f.get("value")
            if fld == "weekday":
                days = [["Mon", "Tue", "Wed", "Thu", "Fri"].index(x) + 1 for x in (v or [])]
                d = d.filter(pl.col("weekday").is_in(days))
                desc.append(f"weekday in {v}")
                continue
            if fld == "event":
                has = pl.col("events").list.set_intersection(pl.lit(v or ["FOMC", "CPI", "NFP"])).list.len() > 0
                d = d.filter(has.fill_null(False) if op == "is" else ~has.fill_null(False))
                desc.append(f"event {'is' if op == 'is' else 'not'} {v}")
                continue
            if fld == "trend":
                d = d.filter(pl.col("trend") == v)
                desc.append(f"trend = {v}")
                continue
            if fld not in d.columns:
                raise ValueError(f"unknown filter field {fld}")
            c = pl.col(fld)
            if op == "between":
                lo, hi = v
                d = d.filter((c >= lo) & (c <= hi))
            else:
                d = d.filter({">": c > v, "<": c < v, ">=": c >= v, "<=": c <= v}[op])
            desc.append(f"{fld} {op} {v}")
        return d, desc

    # ------------------------------------------------------------------ templates
    def intraday_move(self, symbol: str, start=None, end=None, thresholds=(0.25, 0.5, 1.0), filters=None, absolute=True, **_):
        d = self.day_table(symbol, start, end)
        base = d
        d, fdesc = self.apply_filters(d, filters)
        horizons = [15, 30, 60, 120, 240]
        rows = []
        for h in horizons + ["close"]:
            col = "oc_ret" if h == "close" else f"move_{h}"
            x = d[col].to_numpy().astype(float)
            x = x[np.isfinite(x)]
            ax = np.abs(x) if absolute else x
            row = {"horizon": f"{h}m" if h != "close" else "close", "n": int(x.size)}
            for th in thresholds:
                k = int((ax > th).sum())
                lo, hi = wilson(k, x.size)
                row[f"> {th}%"] = {"p": k / x.size if x.size else None, "ci": [lo, hi], "k": k}
            rows.append(row)
        x30 = d["move_30"].to_numpy().astype(float)
        charts = [
            {"type": "bar_ci", "title": "P(|move from open| > X%) by time after open", "x": [r["horizon"] for r in rows],
             "series": [{"name": f"> {th}%", "data": [r[f'> {th}%']['p'] for r in rows], "ci": [r[f'> {th}%']['ci'] for r in rows]} for th in thresholds],
             "yLabel": "frequency", "yFormat": "pct"},
            hist_chart(x30, "Move from open after 30 minutes (%)", "%", compare=base["move_30"].to_numpy().astype(float) if filters else None),
        ]
        return {"title": f"How often does {symbol} move more than X% after the open?", "sample_size": d.height, "filters": fdesc,
                "tables": [{"title": "Frequency (Wilson 95% CI)", "rows": rows}], "charts": charts, "stats": {"move_30m": describe(x30), "open_to_close": describe(d["oc_ret"].to_numpy().astype(float))},
                "notes": ["Moves measured from the session open to the close of the last 1-minute bar completed by each horizon.", "Filters use information known at the open (previous-day values, gap, prior VIX close)."]}

    def premium_decay(self, symbol: str, start=None, end=None, structure: str = "straddle", ref_time: str = "09:45", group_by_vix: bool = True, filters=None, recenter: bool = True, **_):
        root = self.store.root / "options" / symbol
        if not root.exists():
            raise ValueError(f"no option data for {symbol}")
        con = duckdb.connect()
        q = f"""
            SELECT ts, expiration, strike, "right", bid, ask, underlying_price
            FROM read_parquet('{root.as_posix()}/*/*.parquet')
            WHERE expiration = CAST(timezone('America/New_York', ts) AS DATE)
            {f"AND expiration >= DATE '{start}'" if start else ''} {f"AND expiration <= DATE '{end}'" if end else ''}
        """
        df = con.sql(q).pl()
        con.close()
        if df.is_empty():
            raise ValueError("no same-day-expiration (0DTE) quotes in the selected range")
        df = df.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("et")).with_columns(
            (pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32)).alias("minute"), pl.col("et").dt.date().alias("date"), ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
        refm = cal.parse_hhmm(ref_time)
        dt = self.day_table(symbol, start, end)
        dt, fdesc = self.apply_filters(dt, filters)
        keep_days = set(dt["date"].to_list())
        vix_by_day = dict(zip(dt["date"].to_list(), dt["vix_prev"].to_list()))
        curves: dict[str, list[np.ndarray]] = {}
        minutes_axis = None
        for (day,), g in df.group_by(["date"]):
            if day not in keep_days:
                continue
            ref = g.filter(pl.col("minute") == refm)
            if ref.is_empty():
                continue
            spot = ref["underlying_price"][0]
            ks = ref["strike"].unique().to_numpy()
            k = float(ks[np.argmin(np.abs(ks - spot))])
            if recenter:
                # rolling ATM: the strike nearest spot at every snapshot isolates time decay
                gg = g.filter(pl.col("minute") >= refm).with_columns((pl.col("strike") - pl.col("underlying_price")).abs().alias("_dist"))
                katm = gg.group_by("minute").agg(pl.col("strike").sort_by("_dist").first().alias("_k"))
                gg = gg.join(katm, on="minute").filter(pl.col("strike") == pl.col("_k"))
            else:
                gg = g.filter((pl.col("strike") == k) & (pl.col("minute") >= refm))
            piv = gg.pivot(on="right", index="minute", values="mid", aggregate_function="first").sort("minute")
            if structure == "straddle":
                if "C" not in piv.columns or "P" not in piv.columns:
                    continue
                val = (piv["C"] + piv["P"]).to_numpy()
            else:
                col = "C" if structure == "call" else "P"
                if col not in piv.columns:
                    continue
                val = piv[col].to_numpy()
            mins = piv["minute"].to_numpy()
            if not np.isfinite(val[0]) or val[0] <= 0:
                continue
            if minutes_axis is None or len(mins) > len(minutes_axis):
                minutes_axis = mins
            v = vix_by_day.get(day)
            grp = "all" if not group_by_vix else "VIX n/a" if v is None else ("VIX < 15" if v < 15 else "VIX 15-20" if v < 20 else "VIX 20-25" if v < 25 else "VIX 25+")
            series = dict(zip(mins.tolist(), (val / val[0] * 100).tolist()))
            curves.setdefault(grp, []).append(series)
        if minutes_axis is None:
            raise ValueError("no sessions with quotes at the reference time")
        xs = [int(m) for m in minutes_axis]
        charts_series = []
        table = []
        for grp in sorted(curves):
            mat = np.array([[c.get(m, np.nan) for m in xs] for c in curves[grp]], float)
            mean = np.nanmean(mat, axis=0)
            charts_series.append({"name": f"{grp} (n={mat.shape[0]})", "data": mean.tolist(), "p25": np.nanquantile(mat, 0.25, axis=0).tolist(), "p75": np.nanquantile(mat, 0.75, axis=0).tolist()})
            for m_target in (600, 720, 840, 900, 945):
                if m_target in xs:
                    j = xs.index(m_target)
                    col = mat[:, j]
                    table.append({"group": grp, "time": cal.fmt_hhmm(m_target), "mean_pct_of_ref": float(np.nanmean(col)), "median": float(np.nanmedian(col)), "n": int(np.isfinite(col).sum())})
        total = sum(len(v) for v in curves.values())
        return {"title": f"0DTE ATM {structure} premium decay ({symbol})", "sample_size": total, "filters": fdesc,
                "charts": [{"type": "band", "title": f"ATM {structure} mid as % of {ref_time} value", "x": [cal.fmt_hhmm(m) for m in xs], "series": charts_series, "yLabel": "% of reference premium"}],
                "tables": [{"title": "Premium remaining", "rows": table}],
                "notes": [("Rolling ATM: at each snapshot the strike nearest spot is used, isolating time decay from moneyness changes." if recenter else f"Fixed strike: ATM at {ref_time}, held all session (value includes intrinsic gained as the underlying moves)."),
                          "mid = (bid+ask)/2 of observed quotes; VIX bucket uses the previous session's close (known at the open)."]}

    def rv_vs_iv(self, symbol: str, start=None, end=None, window: int = 21, **_):
        d = self.day_table(symbol, start, end)
        if "atm_iv_30d" not in d.columns:
            raise ValueError("no IV history for this symbol (import option chains first)")
        c = d["close"].to_numpy().astype(float)
        lr = np.diff(np.log(c), prepend=np.nan)
        n = len(c)
        fwd = np.full(n, np.nan)
        back = np.full(n, np.nan)
        for i in range(n):
            if i + window < n:
                fwd[i] = np.nanstd(lr[i + 1:i + window + 1], ddof=1) * math.sqrt(252) * 100
            if i >= window:
                back[i] = np.nanstd(lr[i - window + 1:i + 1], ddof=1) * math.sqrt(252) * 100
        iv = d["atm_iv_30d"].to_numpy().astype(float) * 100
        vrp = iv - fwd
        dates = [x.isoformat() for x in d["date"].to_list()]
        ok = np.isfinite(vrp)
        return {"title": f"Implied vs realised volatility ({symbol})", "sample_size": int(ok.sum()),
                "stats": {"iv30": describe(iv), "realised_fwd": describe(fwd), "vrp": describe(vrp), "pct_days_iv_above_rv": float((vrp[ok] > 0).mean()) if ok.any() else None,
                          "corr_iv_fwd_rv": float(np.corrcoef(iv[ok], fwd[ok])[0, 1]) if ok.sum() > 3 else None},
                "charts": [
                    {"type": "line", "title": "ATM IV (30d) vs realised volatility", "x": dates, "series": [{"name": "ATM IV 30d", "data": iv.tolist()}, {"name": f"RV next {window}d", "data": fwd.tolist()}, {"name": f"RV trailing {window}d", "data": back.tolist()}], "yLabel": "vol %"},
                    {"type": "scatter", "title": "IV at t vs realised over the following window", "series": [{"name": "days", "data": [[a, b] for a, b in zip(iv[ok].tolist(), fwd[ok].tolist())]}], "xLabel": "IV %", "yLabel": "forward RV %", "diagonal": True},
                    hist_chart(vrp, "Variance risk premium (IV - forward RV)", "vol points"),
                ],
                "notes": [f"Forward realised vol uses the {window} sessions after each date (an outcome, not a signal).", "IV = constant-maturity 30-day ATM IV derived from stored chains at each session's last snapshot."]}

    def vix_regimes(self, symbol: str, start=None, end=None, edges=(15, 20, 25, 30), **_):
        d = self.day_table(symbol, start, end)
        v = d["vix_prev"].to_numpy().astype(float)
        labels = [f"< {edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
        idx = np.digitize(v, edges)
        rows = []
        cc, oc, rng = d["cc_ret"].to_numpy().astype(float), d["oc_ret"].to_numpy().astype(float), d["range_pct"].to_numpy().astype(float)
        ivf = d["iv_front_prev"].to_numpy().astype(float) * 100 if "iv_front_prev" in d.columns else np.full(len(v), np.nan)
        implied_move = ivf / math.sqrt(252)
        for i, lab in enumerate(labels):
            m = (idx == i) & np.isfinite(v)
            if not m.any():
                continue
            ci = boot_ci(np.abs(cc[m]))
            rows.append({"vix_regime": lab, "sessions": int(m.sum()), "mean_cc_ret": float(np.nanmean(cc[m])), "mean_abs_cc_ret": float(np.nanmean(np.abs(cc[m]))), "abs_ret_ci95": list(ci),
                         "mean_range_pct": float(np.nanmean(rng[m])), "implied_daily_move": float(np.nanmean(implied_move[m])) if np.isfinite(implied_move[m]).any() else None,
                         "realised_to_implied": float(np.nanmean(np.abs(cc[m])) / np.nanmean(implied_move[m])) if np.isfinite(implied_move[m]).any() else None,
                         "p_down_day": float((cc[m] < 0).mean())})
        return {"title": f"{symbol} behaviour across VIX regimes", "sample_size": int(np.isfinite(v).sum()),
                "tables": [{"title": "By prior-day VIX close", "rows": rows}],
                "charts": [{"type": "bar", "title": "Mean |close-to-close return| vs implied daily move", "x": [r["vix_regime"] for r in rows],
                            "series": [{"name": "realised |ret| %", "data": [r["mean_abs_cc_ret"] for r in rows]}, {"name": "implied 1-day move % (front IV/sqrt(252))", "data": [r["implied_daily_move"] for r in rows]}], "yLabel": "%"},
                           {"type": "bar", "title": "Sessions per regime", "x": [r["vix_regime"] for r in rows], "series": [{"name": "sessions", "data": [r["sessions"] for r in rows]}]}],
                "notes": ["Regime = previous session's vol-index close (known before the open).", "Implied 1-day move uses the front-expiry ATM IV at the prior close; E|N(0,s)| = 0.8s so a ratio near 0.8 means fairly priced."]}

    def event_premium(self, symbol: str, start=None, end=None, event_types=("FOMC", "CPI", "NFP"), **_):
        d = self.day_table(symbol, start, end)
        evs = d["events"].to_list()
        is_ev = np.array([bool(e) and bool(set(e) & set(event_types)) for e in evs])
        cc = np.abs(d["cc_ret"].to_numpy().astype(float))
        ivf = d["atm_iv_front"].to_numpy().astype(float) * 100 if "atm_iv_front" in d.columns else np.full(len(cc), np.nan)
        ivp = np.r_[np.nan, ivf[:-1]]
        iv_chg = ivf - ivp
        pre = np.r_[is_ev[1:], False]
        post = np.r_[False, is_ev[:-1]]
        none = ~(is_ev | pre | post)
        rows = []
        for lab, m in [("day before event", pre), ("event day", is_ev), ("day after event", post), ("other days", none)]:
            rows.append({"bucket": lab, "sessions": int(m.sum()), "mean_abs_ret": float(np.nanmean(cc[m])) if m.any() else None, "abs_ret_ci95": list(boot_ci(cc[m])) if m.any() else None,
                         "front_iv_close": float(np.nanmean(ivf[m])) if m.any() else None, "iv_change": float(np.nanmean(iv_chg[m])) if m.any() else None})
        t = sst.ttest_ind(cc[is_ev], cc[none], equal_var=False, nan_policy="omit") if is_ev.sum() > 2 else None
        return {"title": f"Premium and moves around scheduled events ({', '.join(event_types)})", "sample_size": int(is_ev.sum()),
                "tables": [{"title": "Event buckets", "rows": rows}],
                "stats": {"welch_t_abs_ret_event_vs_other": float(t.statistic) if t is not None else None, "p_value": float(t.pvalue) if t is not None else None},
                "charts": [{"type": "bar", "title": "Front ATM IV at close and day-over-day change", "x": [r["bucket"] for r in rows], "series": [{"name": "front IV %", "data": [r["front_iv_close"] for r in rows]}, {"name": "IV change (pts)", "data": [r["iv_change"] for r in rows]}]},
                           {"type": "bar_ci", "title": "Mean |close-to-close return| (%)", "x": [r["bucket"] for r in rows], "series": [{"name": "|ret|", "data": [r["mean_abs_ret"] for r in rows], "ci": [r["abs_ret_ci95"] for r in rows]}]}],
                "notes": ["Event dates come from the events table (synthetic schedule for DEMO data; import a real calendar CSV for real symbols).", "IV crush shows as a negative IV change on the event day for pre-market releases."]}

    def conditional(self, symbol: str, target: str = "oc_ret", start=None, end=None, filters=None, **_):
        d = self.day_table(symbol, start, end)
        if target not in d.columns:
            raise ValueError(f"unknown target {target}")
        allx = d[target].to_numpy().astype(float)
        sel, fdesc = self.apply_filters(d, filters)
        x = sel[target].to_numpy().astype(float)
        t = sst.ttest_ind(x[np.isfinite(x)], allx[np.isfinite(allx)], equal_var=False) if np.isfinite(x).sum() > 2 else None
        ks = sst.ks_2samp(x[np.isfinite(x)], allx[np.isfinite(allx)]) if np.isfinite(x).sum() > 2 else None
        notes = []
        if np.isfinite(x).sum() < 30:
            notes.append(f"Small sample: only {int(np.isfinite(x).sum())} sessions match - interpret with caution.")
        lab = TARGETS.get(target, target)
        return {"title": f"{lab} | {' AND '.join(fdesc) or 'all sessions'}", "sample_size": int(np.isfinite(x).sum()), "filters": fdesc,
                "stats": {"selected": describe(x), "unconditional": describe(allx), "welch_t": float(t.statistic) if t else None, "p_value": float(t.pvalue) if t else None, "ks_p_value": float(ks.pvalue) if ks else None},
                "charts": [hist_chart(x, f"{lab}: conditional vs unconditional", lab, compare=allx)],
                "tables": [], "notes": notes + ["Filters use values known at the open; targets are realised outcomes."]}

    def entry_timing(self, **_):
        raise ValueError("entry_timing runs as a background job (POST /api/research/entry-timing)")


TARGETS = {"oc_ret": "Open-to-close return %", "cc_ret": "Close-to-close return %", "gap_pct": "Opening gap %", "range_pct": "Intraday range %",
           "move_15": "Move first 15m %", "move_30": "Move first 30m %", "move_60": "Move first 60m %", "move_120": "Move first 2h %", "move_240": "Move first 4h %"}
FILTER_FIELDS = {"gap_pct": "Opening gap %", "prev_ret": "Previous day return %", "prev_range_pct": "Previous day range %", "vix_prev": "Prior VIX close",
                 "iv_prev": "Prior ATM IV (decimal)", "iv_rank_prev": "Prior IV rank", "weekday": "Weekday", "event": "Scheduled event", "trend": "Trend regime"}

TEMPLATES = {
    "intraday_move": {"label": "Moves after the open", "question": "How often does the underlying move more than X% after the open?", "params": ["symbol", "start", "end", "thresholds", "filters"]},
    "premium_decay": {"label": "0DTE premium decay", "question": "How do ATM option premiums decay through a 0DTE session?", "params": ["symbol", "start", "end", "structure", "ref_time", "group_by_vix", "recenter", "filters"]},
    "rv_vs_iv": {"label": "Realised vs implied vol", "question": "How does realised volatility compare with implied volatility?", "params": ["symbol", "start", "end", "window"]},
    "vix_regimes": {"label": "VIX regimes", "question": "How does the underlying (and its options) behave across VIX regimes?", "params": ["symbol", "start", "end"]},
    "event_premium": {"label": "Scheduled events", "question": "What happens to premium and moves around scheduled economic releases?", "params": ["symbol", "start", "end", "event_types"]},
    "conditional": {"label": "Custom conditional query", "question": "Distribution of an outcome given conditions known at the open", "params": ["symbol", "start", "end", "target", "filters"]},
    "entry_timing": {"label": "Entry-time sensitivity", "question": "How sensitive are a strategy's results to its entry time?", "params": ["strategy", "times"]},
}


# ------------------------------------------------------------------ SQL console
def sql_query(sql: str, store: MarketStore | None = None, limit: int = 5000, timeout: float = 20.0) -> dict:
    store = store or MarketStore()
    root = store.root
    s = sql.strip().rstrip(";")
    first = s.split(None, 1)[0].lower() if s else ""
    if first not in ("select", "with", "describe", "show", "summarize", "pivot"):
        raise ValueError("only read-only queries (SELECT / WITH / DESCRIBE / SUMMARIZE) are allowed")
    import re
    low = re.sub(r"'[^']*'", "''", s.lower())
    banned = re.search(r"\b(read_\w+|copy|attach|detach|install|load|export|import|pragma|set|create|insert|update|delete|drop|alter|call|glob|httpfs)\b", low)
    if banned:
        raise ValueError(f"'{banned.group(1)}' is not allowed in the research console; query the views bars, daily, options, iv_history, events")
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    try:
        if (root / "underlying").exists():
            con.execute(f"CREATE VIEW bars AS SELECT regexp_extract(filename, 'underlying/([^/]+)/', 1) AS symbol, ts, open, high, low, close, volume FROM read_parquet('{(root / 'underlying').as_posix()}/*/*.parquet', filename=true)")
        if (root / "daily").exists():
            con.execute(f"CREATE VIEW daily AS SELECT regexp_extract(filename, 'daily/([^/.]+)', 1) AS symbol, * EXCLUDE (filename) FROM read_parquet('{(root / 'daily').as_posix()}/*.parquet', filename=true)")
        if (root / "options").exists():
            con.execute(f"CREATE VIEW options AS SELECT regexp_extract(filename, 'options/([^/]+)/', 1) AS symbol, * EXCLUDE (filename) FROM read_parquet('{(root / 'options').as_posix()}/*/*/*.parquet', filename=true, union_by_name=true)")
        if (root / "derived").exists():
            con.execute(f"CREATE VIEW iv_history AS SELECT regexp_extract(filename, 'derived/([^/_]+)_iv', 1) AS symbol, * EXCLUDE (filename) FROM read_parquet('{(root / 'derived').as_posix()}/*_iv.parquet', filename=true)")
        if (root / "events" / "events.parquet").exists():
            con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{(root / 'events' / 'events.parquet').as_posix()}')")
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            rel = con.execute(f"SELECT * FROM ({s}) LIMIT {limit + 1}" if first in ("select", "with") else s)
            cols = [c[0] for c in rel.description]
            rows = rel.fetchall()
        finally:
            timer.cancel()
    finally:
        con.close()
    trunc = len(rows) > limit
    rows = rows[:limit]

    def conv(v):
        if isinstance(v, (date,)):
            return v.isoformat()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v
    return {"columns": cols, "rows": [[conv(v) for v in r] for r in rows], "truncated": trunc, "row_count": len(rows)}
