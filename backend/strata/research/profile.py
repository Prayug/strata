"""Two questions the study engine was not built for, computed directly from the stored bars.

  timing      when during the session something happens: the half hour in which the day's high (or low) was made, or how
              much the price moves in each part of the session
  breakdown   the same daily measurement grouped by weekday or by month, with a test of whether the groups differ by more
              than chance would make them

Both are descriptive. Every figure is counted from stored sessions, the sessions behind every figure are kept so they can be
listed, and nothing is projected forward.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from strata.data.store import MarketStore

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _daily(store: MarketStore, sym: str) -> pl.DataFrame:
    d = store.read_daily(sym).sort("date")
    if d.is_empty():
        raise LookupError(f"No daily history is stored for {sym}.")
    d = d.with_columns(prev_close=pl.col("close").shift(1), prev2_close=pl.col("close").shift(2))
    d = d.with_columns(gap=(pl.col("open") / pl.col("prev_close") - 1) * 100, prev_return=(pl.col("prev_close") / pl.col("prev2_close") - 1) * 100,
                       oc=(pl.col("close") / pl.col("open") - 1) * 100, cc=(pl.col("close") / pl.col("prev_close") - 1) * 100, wd=pl.col("date").dt.weekday() - 1, mo=pl.col("date").dt.month())
    return d


def _vix(store: MarketStore) -> pl.DataFrame | None:
    v = store.read_daily("VIX")
    if v.is_empty():
        return None
    return v.sort("date").select("date", pl.col("close").shift(1).alias("vix_prev"))


def _filter_days(store: MarketStore, d: pl.DataFrame, conditions: list[dict]) -> tuple[pl.DataFrame, list[str]]:
    """Sessions that meet conditions known by the open (previous session's move, the gap, the weekday, the VIX, a streak).
    Returns the filtered sessions and the conditions that cannot be evaluated here (reported, never dropped)."""
    out, cannot = d, []
    for c in conditions or []:
        k, op, v = c.get("key"), c.get("op"), c.get("value")
        if k in ("prev_return", "gap"):
            col = pl.col(k)
            out = out.filter(col <= v if op in ("<=",) else col < v if op == "<" else col >= v if op == ">=" else col > v)
        elif k == "weekday":
            out = out.filter(pl.col("wd") == ["Mon", "Tue", "Wed", "Thu", "Fri"].index(v))
        elif k == "vix":
            vx = _vix(store)
            if vx is None:
                cannot.append("the VIX (no VIX history is stored)")
                continue
            out = out.join(vx, on="date", how="left").filter(pl.col("vix_prev") > v if op in (">", ">=") else pl.col("vix_prev") < v).drop("vix_prev")
        elif k in ("up_streak", "down_streak"):
            up = (pl.col("cc") > 0) if k == "up_streak" else (pl.col("cc") < 0)
            s = out.with_columns(flag=up.shift(1).fill_null(False))
            run, runs = 0, []
            for f in s["flag"].to_list():
                run = run + 1 if f else 0
                runs.append(run)
            out = s.with_columns(streak=pl.Series(runs)).filter(pl.col("streak") >= int(v)).drop("flag", "streak")
        else:
            cannot.append(c.get("text") or k)
    return out, cannot


def _period(d: pl.DataFrame, period) -> pl.DataFrame:
    if period and period[0]:
        d = d.filter(pl.col("date") >= date.fromisoformat(period[0]))
    if period and len(period) > 1 and period[1]:
        d = d.filter(pl.col("date") <= date.fromisoformat(period[1]))
    return d


def timing(store: MarketStore, symbol: str, measure: str = "abs_move", bucket: int = 30, period=None, conditions: list[dict] | None = None) -> dict:
    """When during the session: the bucket of the day's high or low, or the average absolute move per bucket."""
    m = store.read_minutes(symbol, provider="alpaca")
    source = "alpaca 1-minute bars (consolidated tape)"
    if m.is_empty():
        m = store.read_minutes(symbol)
        source = "1-minute bars"
    if m.is_empty():
        raise LookupError(f"No intraday history is stored for {symbol}: when something happens during the day cannot be measured.")
    m = m.with_columns(et=pl.col("ts").dt.convert_time_zone("America/New_York"))
    m = m.with_columns(d=pl.col("et").dt.date(), mod=pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32)).filter((pl.col("mod") >= 570) & (pl.col("mod") < 960))
    days = _period(_daily(store, symbol), period)
    days, cannot = _filter_days(store, days, conditions or [])
    m = m.filter(pl.col("d").is_in(days["date"].implode()))
    # whole sessions only: a session that starts late or ends early would put its high in the wrong bucket
    g = m.group_by("d").agg(first=pl.col("mod").min(), last=pl.col("mod").max(), n=pl.len())
    full = g.filter((pl.col("first") == 570) & (pl.col("last") >= 959))
    early = g.filter((pl.col("first") == 570) & (pl.col("last") < 959) & (pl.col("last") >= 779))  # early closes (13:00) kept: their buckets end sooner
    keep = pl.concat([full, early])["d"]
    m = m.filter(pl.col("d").is_in(keep.implode())).sort(["d", "mod"])
    m = m.with_columns(b=((pl.col("mod") - 570) // bucket).cast(pl.Int32))
    nb = (390 + bucket - 1) // bucket
    labels = [f"{(570 + i * bucket) // 60:02d}:{(570 + i * bucket) % 60:02d}" for i in range(nb)]
    ends = [f"{min(570 + (i + 1) * bucket, 960) // 60:02d}:{min(570 + (i + 1) * bucket, 960) % 60:02d}" for i in range(nb)]
    n_days = int(keep.len())
    if n_days == 0:
        raise LookupError(f"No whole session of intraday bars is stored for {symbol} in that period.")
    per_session = []
    if measure in ("high_time", "low_time"):
        col = "high" if measure == "high_time" else "low"
        ext = m.group_by("d", maintain_order=True).agg(pl.col("b").sort_by(col, descending=(measure == "high_time"), maintain_order=True).first().alias("bucket"),
                                                     pl.col("mod").sort_by(col, descending=(measure == "high_time"), maintain_order=True).first().alias("minute"))
        cnt = ext.group_by("bucket").agg(n=pl.len())
        counts = {int(r["bucket"]): int(r["n"]) for r in cnt.iter_rows(named=True)}
        rows = [{"start": labels[i], "end": ends[i], "sessions": counts.get(i, 0), "share": counts.get(i, 0) / n_days * 100} for i in range(nb)]
        per_session = [{"date": r["d"].isoformat(), "bucket": labels[int(r["bucket"])], "time": f"{int(r['minute']) // 60:02d}:{int(r['minute']) % 60:02d}"} for r in ext.sort("d").iter_rows(named=True)]
        top = max(rows, key=lambda r: r["sessions"])
        even = 100 / nb
        value_name = "share of sessions"
    else:
        # the absolute move of each bucket: |last close / first open - 1|, averaged over sessions
        bb = m.group_by(["d", "b"], maintain_order=True).agg(o=pl.col("open").first(), c=pl.col("close").last())
        bb = bb.with_columns(mv=((pl.col("c") / pl.col("o") - 1).abs() * 100))
        agg = bb.group_by("b").agg(mean=pl.col("mv").mean(), median=pl.col("mv").median(), n=pl.len())
        a = {int(r["b"]): r for r in agg.iter_rows(named=True)}
        # the bucket with the largest move of each session
        big = bb.sort(["d", "mv"], descending=[False, True]).group_by("d", maintain_order=True).first()
        bigc = {int(r["b"]): int(r["n"]) for r in big.group_by("b").agg(n=pl.len()).iter_rows(named=True)}
        rows = [{"start": labels[i], "end": ends[i], "mean_abs_move_pct": float(a[i]["mean"]) if i in a else None, "median_abs_move_pct": float(a[i]["median"]) if i in a else None,
                 "sessions": int(a[i]["n"]) if i in a else 0, "largest_move_share": bigc.get(i, 0) / n_days * 100} for i in range(nb)]
        top = max([r for r in rows if r["mean_abs_move_pct"] is not None], key=lambda r: r["mean_abs_move_pct"])
        even = None
        value_name = "average size of the move"
        per_session = [{"date": r["d"].isoformat(), "bucket": labels[int(r["b"])], "move_pct": round(float(r["mv"]), 4)} for r in big.sort("d").iter_rows(named=True)]
    first, last = keep.min(), keep.max()
    return {"kind": "timing", "symbol": symbol, "measure": measure, "bucket_minutes": bucket, "buckets": rows, "sessions": n_days, "first": first.isoformat(), "last": last.isoformat(),
            "top": top, "even_share": even, "value_name": value_name, "per_session": per_session, "conditions_not_evaluated": cannot, "source": source,
            "notes": ["Only whole sessions of one-minute bars are used; a session the provider holds in part is left out."] + ([f"Not applied (cannot be evaluated here): {', '.join(cannot)}."] if cannot else [])}


def breakdown(store: MarketStore, symbol: str, by: str = "weekday", measure: str = "open_to_close", period=None, conditions: list[dict] | None = None) -> dict:
    """A daily measurement grouped by weekday or month, each group against all the others."""
    d = _period(_daily(store, symbol), period).drop_nulls(["prev_close"])
    d, cannot = _filter_days(store, d, conditions or [])
    col = {"open_to_close": "oc", "return": "oc", "close_to_close": "cc", "close>open": "oc"}.get(measure, "oc")
    key = "wd" if by == "weekday" else "mo"
    names = WEEKDAYS if by == "weekday" else MONTHS
    x = d.select("date", key, col).drop_nulls()
    vals = x[col].to_numpy()
    groups = []
    for gk in sorted(set(x[key].to_list())):
        if by == "weekday" and gk > 4:
            continue
        sub = x.filter(pl.col(key) == gk)
        v = sub[col].to_numpy()
        rest = x.filter(pl.col(key) != gk)[col].to_numpy()
        se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        groups.append({"group": names[gk if by == "weekday" else gk - 1], "sessions": int(len(v)), "mean_pct": float(v.mean()), "median_pct": float(np.median(v)), "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
                       "positive_share": float((v > 0).mean() * 100), "others_mean_pct": float(rest.mean()) if len(rest) else None,
                       "dates": [z.isoformat() for z in sub["date"].to_list()]})
    # do the groups differ by more than chance? A permutation test on the spread of the group means (no normality assumed)
    rng = np.random.default_rng(7)
    labs = x[key].to_numpy()
    uniq = np.unique(labs)

    def spread(lb):
        return float(np.var([vals[lb == u].mean() for u in uniq]))
    obs = spread(labs)
    perm = [spread(rng.permutation(labs)) for _ in range(400)]
    p = (1 + sum(1 for s in perm if s >= obs)) / (1 + len(perm))
    best = max(groups, key=lambda g: g["mean_pct"])
    worst = min(groups, key=lambda g: g["mean_pct"])
    return {"kind": "breakdown", "symbol": symbol, "by": by, "measure": measure, "groups": groups, "sessions": int(len(vals)), "overall_mean_pct": float(vals.mean()) if len(vals) else None,
            "overall_positive_share": float((vals > 0).mean() * 100) if len(vals) else None, "p_value_groups_differ": p, "best": best["group"], "worst": worst["group"],
            "first": x["date"].min().isoformat() if x.height else None, "last": x["date"].max().isoformat() if x.height else None, "conditions_not_evaluated": cannot,
            "measure_label": {"oc": "the move from the open to the close", "cc": "the close-to-close move"}[col],
            "notes": ["Differences between groups are tested by shuffling the group labels 400 times: if the real spread of the group averages is ordinary among the shuffled ones, the groups do not differ by more than chance."]
            + ([f"Not applied (cannot be evaluated here): {', '.join(cannot)}."] if cannot else [])}


def event_frequency(store: MarketStore, symbol: str, event: dict, period=None, conditions: list[dict] | None = None) -> dict:
    """How often an event known from daily bars happens at all: "how often does IWM gap up more than 1%"."""
    d = _period(_daily(store, symbol), period).drop_nulls(["prev_close"])
    d, cannot = _filter_days(store, d, conditions or [])
    k_, op, v = event.get("key"), event.get("op"), event.get("value")
    col = {"gap": "gap", "day_return": "cc", "prev_return": "cc"}.get(k_)
    if col is None:
        raise ValueError(f"how often {k_} happens is not available")
    e = pl.col(col)
    cond = e <= v if op == "<=" else e < v if op == "<" else e >= v if op == ">=" else e > v
    hit = d.filter(cond)
    what = {"gap": f"opened {abs(v):g}% or more {'above' if v > 0 else 'below'} the previous close", "cc": f"closed {abs(v):g}% or more {'above' if v > 0 else 'below'} the previous close"}[col]
    if not v:
        what = {"gap": f"opened {'above' if op in ('>', '>=') else 'below'} the previous close", "cc": f"closed {'up' if op in ('>', '>=') else 'down'} on the day"}[col]
    years = []
    for y in sorted(set(d["date"].dt.year().to_list())):
        dy = d.filter(pl.col("date").dt.year() == y)
        ky = dy.filter(cond).height
        years.append({"year": str(y), "n": dy.height, "target": ky, "probability": ky / dy.height if dy.height else None})
    return {"kind": "event", "symbol": symbol, "what": what, "k": hit.height, "n": d.height, "pct": hit.height / d.height * 100 if d.height else None,
            "first": d["date"].min().isoformat() if d.height else None, "last": d["date"].max().isoformat() if d.height else None,
            "dates": [x.isoformat() for x in hit["date"].to_list()], "by_year": years, "conditions_not_evaluated": cannot,
            "notes": ["Counted from daily bars: the open against the previous session's close."] + ([f"Not applied (cannot be evaluated here): {', '.join(cannot)}."] if cannot else [])}
