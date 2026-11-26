"""Market replay on stored observations.

Everything a replay shows is something that is on disk: the underlying's stored bars, and the one-minute TRADE bars of an
option contract in the minutes in which it printed. Nothing is interpolated to make a chart look continuous:

  * a minute in which a contract did not trade has no bar. The chart has a gap there, and the mark carried across it says
    how old it is
  * a bar is observable when it has ENDED (the engine's rule). A mark at 10:15 comes from a print made before 10:15
  * the levels (VWAP, opening range, the previous session's close, high and low) are computed with the engine's own
    functions from the same stored bars, so a level on the chart is the level a rule was tested against
  * a trade is laid over the observations with its decision time, its fill time and the age of the print it was filled on.
    Its running profit is the stored prints of its legs against its entry prices: an observation of the market, not the
    engine's own bookkeeping read back

Trade bars carry no bid and no ask. A replay of them shows where the contract traded, not where an order could have been
filled.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.core import indicators as IND
from strata.data.providers import sources as S
from strata.data.store import MarketStore

MAX_POINTS = 2400  # candles sent for one chart: beyond it the stored bars are aggregated, and the reply says to what


def _f(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _bucket(span_sessions: int, stored_seconds: int) -> int:
    """Seconds per candle for a span: the stored resolution while it fits, then the next round size that does."""
    per_day = 23400 // max(60, stored_seconds)
    for sec in (stored_seconds, 300, 900, 1800, 3600, 23400):
        if sec >= stored_seconds and span_sessions * max(1, 23400 // sec) <= MAX_POINTS:
            return sec
    return 23400 if per_day else stored_seconds


def underlying(store: MarketStore, symbol: str, d0: date, d1: date | None = None, bucket: int | None = None) -> dict:
    """Stored bars of the underlying for d0..d1 with the engine's levels. `bucket` (seconds) asks for a coarser candle."""
    d1 = d1 or d0
    days = cal.trading_days(d0, d1)
    if not days:
        raise ValueError(f"{d0} is not a trading session")
    res, df = None, pl.DataFrame()
    for r, sec in (("1m", 60), ("5m", 300), ("60m", 3600)):
        df = store.read_bars(symbol, r, d0, d1)
        if df is not None and not df.is_empty():
            res = (r, sec)
            break
    if res is None:
        return {"symbol": symbol, "start": d0.isoformat(), "end": d1.isoformat(), "stored": None, "candles": [], "note": f"No intraday bars are stored for {symbol} on these sessions."}
    df = df.sort("ts").with_columns(pl.col("ts").dt.epoch("s").alias("t"), pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("d"))
    sid = df["d"].cast(pl.Int32).to_numpy().astype(np.int64)
    h, l, c, v = (df[k].to_numpy().astype(float) for k in ("high", "low", "close", "volume"))
    vwap = IND.session_vwap(h, l, c, np.nan_to_num(v), sid)
    ema9, ema21 = IND.ema(c, 9), IND.ema(c, 21)
    df = df.with_columns(pl.Series("vwap", vwap), pl.Series("ema9", ema9), pl.Series("ema21", ema21))
    sec = max(res[1], int(bucket or _bucket(len(days), res[1])))
    if sec > res[1]:
        key = (pl.col("t") // sec * sec) if sec < 23400 else pl.col("d").cast(pl.Int32).cast(pl.Int64)
        g = df.with_columns(key.alias("b")).group_by(["d", "b"], maintain_order=True).agg(pl.col("t").first(), pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
                                                                                           pl.col("vwap").last(), pl.col("ema9").last(), pl.col("ema21").last())
    else:
        g = df
    daily = store.read_daily(symbol)
    sessions = []
    for d in days:
        day = df.filter(pl.col("d") == d)
        if day.is_empty():
            sessions.append({"date": d.isoformat(), "stored": False})
            continue
        o_ts = int(cal.session_bounds(d)[0].timestamp())
        prev = daily.filter(pl.col("date") < d).tail(1) if not daily.is_empty() else pl.DataFrame()
        f = store.split_factor(symbol, d)
        rng = {}
        for n in (5, 15, 30, 60):
            first = day.filter(pl.col("t") < o_ts + n * 60)
            if not first.is_empty() and int(day["t"].max()) + res[1] >= o_ts + n * 60:
                rng[str(n)] = {"high": _f(first["high"].max()), "low": _f(first["low"].min()), "complete_at": o_ts + n * 60}
        sessions.append({"date": d.isoformat(), "stored": True, "bars": day.height, "open_ts": o_ts, "close_ts": int(cal.session_bounds(d)[1].timestamp()), "open": _f(day["open"][0]), "high": _f(day["high"].max()),
                         "low": _f(day["low"].min()), "close": _f(day["close"][-1]), "volume": _f(day["volume"].sum()),
                         "previous": None if prev.is_empty() else {"date": prev["date"][0].isoformat(), "close": _f(prev["close"][0]), "high": _f(prev["high"][0]), "low": _f(prev["low"][0])},
                         "opening_range": rng, **({"as_traded_factor": f, "note": "Prices are split-adjusted; the contracts of this session were struck on prices this many times higher."} if abs(f - 1.0) > 1e-9 else {})})
    return {"symbol": symbol, "start": d0.isoformat(), "end": d1.isoformat(), "stored": res[0], "candle_seconds": sec,
            "aggregated": sec > res[1], "note": (f"Stored {res[0]} bars aggregated to {sec // 60}-minute candles to draw {len(days)} sessions." if sec > res[1] else f"Stored {res[0]} bars as they are."),
            "candles": [[int(t), _f(o), _f(hh), _f(ll), _f(cc), _f(vv)] for t, o, hh, ll, cc, vv in g.select("t", "open", "high", "low", "close", "volume").iter_rows()],
            "vwap": [_f(x) for x in g["vwap"].to_list()], "ema9": [_f(x) for x in g["ema9"].to_list()], "ema21": [_f(x) for x in g["ema21"].to_list()], "sessions": sessions,
            "levels": "VWAP is the session's volume-weighted typical price so far and the moving averages are of the stored closes: the functions the engine tests a rule against."}


def sessions(store: MarketStore, symbol: str, provider: str | None = None) -> dict:
    """The sessions of a symbol that have stored option observations (newest first), with how many contracts each holds."""
    ds = store.option_dataset(symbol, "trade", provider)
    if ds is None:
        return {"symbol": symbol, "provider": None, "sessions": [], "note": f"No option trade observations are stored for {symbol}."}
    idx = store.option_bar_index(symbol, dataset=ds)
    if idx.is_empty():
        return {"symbol": symbol, "provider": ds.provider, "sessions": []}
    g = idx.filter(pl.col("status") == "ok").group_by("date").agg(pl.len().alias("contracts"), pl.col("bars").sum().alias("bars"), pl.col("expiration").n_unique().alias("expirations")).sort("date", descending=True)
    return {"symbol": symbol, "provider": ds.provider, "first": g["date"].min().isoformat(), "last": g["date"].max().isoformat(), "count": g.height,
            "sessions": [{"date": d.isoformat(), "contracts": int(n), "bars": int(b), "expirations": int(e)} for d, n, b, e in g.iter_rows()]}


def contracts(store: MarketStore, symbol: str, d: date, provider: str | None = None) -> dict:
    """The contracts with stored observations on one session, by expiration: what a replay can switch between."""
    ds = store.option_dataset(symbol, "trade", provider)
    idx = store.option_bar_index(symbol, dataset=ds) if ds is not None else pl.DataFrame()
    if idx.is_empty():
        return {"symbol": symbol, "date": d.isoformat(), "expirations": [], "note": f"No option trade observations are stored for {symbol}."}
    x = idx.filter(pl.col("date") == d)
    if x.is_empty():
        return {"symbol": symbol, "date": d.isoformat(), "provider": ds.provider, "expirations": [], "note": f"No contract of {symbol} has stored observations on {d}."}
    daily = store.read_daily(symbol)
    prev = daily.filter(pl.col("date") < d).tail(1) if not daily.is_empty() else pl.DataFrame()
    f = store.split_factor(symbol, d)
    ref = None if prev.is_empty() else float(prev["close"][0]) * f
    out = []
    for (exp,), g in x.sort(["expiration", "strike", "right"]).group_by(["expiration"], maintain_order=True):
        rows = [{"occ": o, "strike": float(k), "right": r, "bars": int(b), "traded": st == "ok"} for o, k, r, b, st in g.select("occ", "strike", "right", "bars", "status").iter_rows()]
        out.append({"expiration": exp.isoformat(), "dte": (exp - d).days, "contracts": len(rows), "traded": sum(1 for r in rows if r["traded"]), "strikes": [min(r["strike"] for r in rows), max(r["strike"] for r in rows)], "rows": rows})
    return {"symbol": symbol, "date": d.isoformat(), "provider": ds.provider, "previous_close_as_traded": ref, "expirations": out,
            "note": "Contracts are listed from the stored index: \"traded\" means the contract printed at least once that session. Bars are read when a contract is opened."}


def contract(store: MarketStore, symbol: str, occ: str, d0: date, d1: date | None = None, provider: str | None = None, bucket: int | None = None) -> dict:
    """The stored one-minute trade bars of one contract over d0..d1. Only minutes in which it printed exist."""
    d1 = d1 or d0
    root, exp, right, strike = S.parse_occ(occ)
    ds = store.option_dataset(symbol, "trade", provider)
    days = [d for d in cal.trading_days(d0, d1) if d <= exp]
    frames, per_day = [], []
    for d in days:
        b = store.read_option_bars(symbol, d, dataset=ds) if ds is not None else None
        if b is None or b.is_empty():
            per_day.append({"date": d.isoformat(), "stored": False, "bars": 0})
            continue
        one = b.filter(pl.col("occ") == occ).sort("ts")
        per_day.append({"date": d.isoformat(), "stored": True, "bars": one.height, "volume": int(one["volume"].fill_null(0).sum()) if one.height else 0,
                        "minutes_with_prints_pct": round(one.height / max(1, cal.session_minutes(d)) * 100, 1)})
        if one.height:
            frames.append(one)
    if not frames:
        return {"symbol": symbol, "occ": occ, "expiration": exp.isoformat(), "strike": strike, "right": right, "bars": [], "sessions": per_day, "provider": ds.provider if ds else None,
                "note": "This contract has no stored print on these sessions. A contract that did not trade has no price here: none is constructed for it."}
    df = pl.concat(frames).with_columns(pl.col("ts").dt.epoch("s").alias("t"))
    sec = 60
    want = int(bucket or 0)
    if not want and df.height > MAX_POINTS:
        want = next(s for s in (300, 900, 1800, 3600, 23400) if df.height / (s / 60) <= MAX_POINTS or s == 23400)
    if want > 60:
        sec = want
        df = df.with_columns((pl.col("t") // sec * sec).alias("b")).group_by("b", maintain_order=True).agg(pl.col("t").first(), pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
                                                                                                         pl.len().alias("n"), pl.col("t").last().alias("last_t"))
    else:
        df = df.with_columns(pl.lit(1).alias("n"), pl.col("t").alias("last_t"))
    close_of = {d: int(cal.session_bounds(d)[1].timestamp()) for d in days}

    def seen(t_last: int) -> int:  # when the last print of the candle became observable: the end of its minute, never after the bell
        d = cal.to_et(t_last * 1_000_000_000).date()
        return min(t_last + 60, close_of.get(d, t_last + 60))
    rows = [[int(t), _f(o), _f(h), _f(l), _f(c), int(v or 0), seen(int(tl)), int(n)] for t, o, h, l, c, v, tl, n in df.select("t", "open", "high", "low", "close", "volume", "last_t", "n").iter_rows()]
    return {"symbol": symbol, "occ": occ, "expiration": exp.isoformat(), "strike": strike, "right": right, "provider": ds.provider, "candle_seconds": sec, "aggregated": sec > 60,
            "columns": ["t", "open", "high", "low", "close", "volume", "observable_at", "minutes_with_prints"], "bars": rows, "sessions": per_day,
            "observation": "trade prints (no bid/ask): each bar is the trades of one minute in which the contract traded",
            "note": ("Stored one-minute trade bars as they are: a missing minute is a minute without a trade." if sec == 60 else f"Stored one-minute trade bars aggregated to {sec // 60}-minute candles; a candle exists only where a print does.")}


# ----------------------------------------------------------------------------- one trade of a run over the observations
def _levels(run_cfg: dict | None, t: dict) -> dict:
    """The exit rules this trade was run with, as prices on the two charts where a rule is a price."""
    from strata.engine.models import BacktestConfig
    out: dict = {"premium": [], "underlying": []}
    try:
        strategies = BacktestConfig.model_validate(run_cfg).strategies  # with the engine's defaults, as the run applied them
    except Exception:  # noqa: BLE001  (a run stored by an older version may not validate today: no level is guessed for it)
        return out
    strat = next((s for s in strategies if s.name == t.get("strategy")), None) or (strategies[int(t.get("strategy_idx") or 0)] if strategies else None)
    if strat is None:
        return out
    ex = strat.exits
    legs = [l for l in t["legs"] if l.get("kind") == "option"]
    single = legs[0] if len(legs) == 1 else None
    for rule, name, sign in ((ex.profit_target, "profit target", 1), (ex.stop_loss, "stop", -1)):
        if not rule.enabled:
            continue
        row = {"rule": name, "value": rule.value, "unit": rule.mode, "text": f"{name}: {rule.value:g}{'% of the premium' if rule.mode == 'pct' else ' dollars'}"}
        if single is not None and rule.mode == "pct":
            long = single["side"] == "buy"
            px = float(single["entry_price"]) * (1 + sign * (1 if long else -1) * rule.value / 100.0)
            if px > 0:
                row["price"] = round(px, 4)
                row["text"] = f"{name}: {rule.value:g}% of the {float(single['entry_price']):.2f} {'paid' if long else 'received'} = {px:.2f}"
        out["premium"].append(row)
    ue, spot = ex.underlying, t.get("entry_spot")
    if ue.enabled and spot:
        bullish = None if single is None else ((single["side"] == "buy") == (single["right"] == "C"))
        for pct, name, sign in ((ue.move_for_pct, "underlying target", 1), (ue.move_against_pct, "underlying stop", -1)):
            if pct is None:
                continue
            row = {"rule": name, "value": pct, "unit": "pct", "text": f"{name}: {pct:g}% {'for' if sign > 0 else 'against'} the position from {float(spot):.2f}"}
            if bullish is not None:
                row["price"] = round(float(spot) * (1 + sign * (1 if bullish else -1) * pct / 100.0), 4)
            out["underlying"].append(row)
    if ex.max_hold.enabled and ex.max_hold.minutes:
        out["time"] = {"rule": "time held", "minutes": int(ex.max_hold.minutes), "at": int(t["entry_ts"]) + 60 * int(ex.max_hold.minutes)}
    if ex.time_exit.enabled:
        out["clock"] = {"rule": "clock exit", "time": ex.time_exit.time}
    for i, st in enumerate(ex.stages):
        if st.profit_pct is not None and single is not None:
            long = single["side"] == "buy"
            out["premium"].append({"rule": f"partial exit {i + 1}", "value": st.profit_pct, "unit": "pct", "price": round(float(single["entry_price"]) * (1 + (1 if long else -1) * st.profit_pct / 100.0), 4),
                                   "text": f"close {st.close_pct:g}% of the position at {st.profit_pct:g}% of the premium"})
    return out


def trade(rid: str, tid: int, store: MarketStore | None = None) -> dict:
    """One trade of a stored run laid over the stored observations of its sessions."""
    from strata.api import db
    from strata.api.results import load_result, load_trades
    store = store or MarketStore()
    t = next((x for x in load_trades(rid) if x["id"] == tid), None)
    if t is None:
        raise KeyError(tid)
    run = db.get_run(rid) or {}
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    # A run made through the lab stores the REQUEST it was made from (no dates, and the research settings beside the
    # strategy); the configuration the engine ran is in its result. Read from the request, every exit level of every such
    # run failed to validate and the replay drew no stop, no target and no clock exit: on real runs only, where no test looked.
    try:
        ran = (load_result(rid) or {}).get("config")
    except Exception:  # noqa: BLE001
        ran = None
    if isinstance(ran, dict) and ran.get("strategies"):
        cfg = ran
    provider = ((cfg.get("data") or {}).get("options_provider")) or None
    sym = t["symbol"]
    d0 = date.fromisoformat(t["entry_date"])
    d1 = cal.to_et(int(t["exit_ts"] or t["entry_ts"]) * 1_000_000_000).date()
    span = cal.trading_days(d0, d1)
    und = underlying(store, sym, d0, d1)
    legs = []
    for l in t["legs"]:
        if l.get("kind") != "option" or not l.get("expiration"):
            continue
        occ = S.occ_symbol(sym, date.fromisoformat(l["expiration"]), l["right"], float(l["strike"]))
        c = contract(store, sym, occ, d0, d1, provider, bucket=und["candle_seconds"] if und.get("aggregated") else None)
        legs.append({"id": l["id"], "side": l["side"], "ratio": l.get("ratio") or 1, "contracts": l.get("contracts") or 1, "multiplier": l.get("multiplier") or 100.0, "occ": occ, "strike": l["strike"], "right": l["right"],
                     "expiration": l["expiration"], "entry_price": l["entry_price"], "exit_price": l.get("exit_price"), "price_source": l.get("price_source"), "entry_volume_so_far": l.get("entry_volume"),
                     "entry_delta": l.get("entry_delta"), "bars": c["bars"], "candle_seconds": c.get("candle_seconds", 60), "sessions": c["sessions"]})
    # the position marked on stored prints: at each minute, every leg's last print that had ended by then
    ent, ext = int(t["entry_ts"]), int(t["exit_ts"] or t["entry_ts"])
    ticks, step = [], (60 if len(span) <= 2 else 300)
    for d in span:
        o, c_ = (int(x.timestamp()) for x in cal.session_bounds(d))
        lo = max(o + 60, ent)
        ticks += list(range(o + -(-(lo - o) // step) * step, min(c_, ext) + 1, step))  # on the session's own clock: 09:35, 09:40, ...
    ticks = sorted(set(ticks) | {ext})  # and the moment it was closed
    marks = {"t": [], "pnl": [], "oldest_print_seconds": [], "legs_observed": []}
    if legs and ticks:
        T = np.asarray(ticks, np.int64)
        total = np.zeros(T.size)
        age = np.zeros(T.size)
        seen_all = np.ones(T.size, bool)
        for lg in legs:
            if not lg["bars"]:
                seen_all[:] = False
                continue
            obs = np.asarray([b[6] for b in lg["bars"]], np.int64)
            px = np.asarray([b[4] for b in lg["bars"]], float)
            i = np.searchsorted(obs, T, side="right") - 1
            ok = i >= 0
            seen_all &= ok
            sign = 1.0 if lg["side"] == "buy" else -1.0
            total += np.where(ok, sign * (px[np.maximum(i, 0)] - float(lg["entry_price"])) * float(lg["contracts"]) * float(lg["multiplier"]), 0.0)
            age = np.maximum(age, np.where(ok, T - obs[np.maximum(i, 0)], 0))
        marks = {"t": [int(x) for x in T], "pnl": [round(float(x), 2) if s else None for x, s in zip(total, seen_all)], "oldest_print_seconds": [int(a) if s else None for a, s in zip(age, seen_all)],
                 "legs_observed": int(seen_all.sum())}
    meta = t.get("meta") or {}
    age_in = meta.get("entry_print_age_seconds")
    markers = [{"kind": "decision", "t": int(t.get("signal_ts") or ent), "text": "the entry rule held", "price": meta.get("signal_price")},
               {"kind": "entry", "t": ent, "text": f"filled {t.get('fill_delay_seconds') or 0}s after the decision" + (f", on a print made {age_in}s before the fill" if age_in is not None else ""), "print_age_seconds": age_in,
                "print_ts": ent - int(age_in) if age_in is not None else None, "underlying": t.get("entry_spot")}]
    for sx in t.get("stage_exits") or []:
        if isinstance(sx, dict) and sx.get("ts"):
            markers.append({"kind": "partial", "t": int(sx["ts"]), "text": str(sx.get("reason") or sx.get("name") or "partial exit")})
    if t.get("exit_ts"):
        if t.get("exit_signal_ts"):
            markers.append({"kind": "exit_decision", "t": int(t["exit_signal_ts"]), "text": "the exit rule held"})
        markers.append({"kind": "exit", "t": ext, "text": str(t.get("exit_reason") or "exit").replace("_", " "), "underlying": t.get("exit_spot")})
    return {"run": rid, "trade": {k: t.get(k) for k in ("id", "strategy", "symbol", "entry_time", "exit_time", "entry_date", "pnl", "gross_pnl", "commissions", "slippage", "exit_reason", "units", "basis", "max_risk", "entry_spot", "exit_spot",
                                                         "holding_minutes", "mfe", "mae", "flags", "entry_dte", "pnl_pct_premium")},
            "underlying": und, "legs": legs, "markers": sorted(markers, key=lambda m: m["t"]), "levels": _levels(cfg, t), "marks": marks,
            "entry_explain": t.get("entry_explain") or [], "exit_explain": t.get("exit_explain") or [],
            "mark_step_seconds": 60 if len(span) <= 2 else 300,
            "reading": ("The running profit is each leg's last stored print (observable once its minute has ended) against the entry price, times the contracts held. "
                        "Where a leg had not printed since the entry there is no mark, and where the print is old the age is shown: nothing is filled in between prints. "
                        "These are trade prints, not quotes: the profit shown is where the contract traded, not what an exit order would have received.")}
