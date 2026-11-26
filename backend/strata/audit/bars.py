"""Are the stored intraday bars of the underlying what they claim to be?

Every intraday dataset of a symbol is compared, session by session, with two independent
references: the exchange calendar (which sessions exist, when each opens and closes) and
the daily record (the official close, the day's range and volume). The checks:

  missing sessions     the exchange was open inside the dataset's range and no bar is stored
  non-sessions         bars on a weekend or an exchange holiday
  partial sessions     the bars start late or stop early (an early close is not "early")
  missing minutes      fewer bars than the session has intervals: gaps inside it
  outside hours        bars before 09:30 or after the close, New York time (extended hours)
  early closes         a 13:00 close must end the bars at 13:00
  daylight saving      the first bar must be at 09:30 New York time on both sides of a change
  official close       the last bar against the daily close: a late-print gap, or a different
                       price scale (an unadjusted split)
  range                intraday high and low against the daily high and low
  volume               the bars' volume as a share of the daily volume: a feed that carries
                       one exchange shows a few percent of the consolidated total
  mixed feeds          two providers at the same resolution on the same sessions

Nothing is repaired here. A finding says what is stored and which reference it disagrees
with, so that incompatible feeds are never combined without the difference being known.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl

from strata.core import calendar as cal
from strata.data.store import MarketStore

SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600}
CLOSE_GAP_PCT = 0.25  # the same threshold the engine reports
SCALE_BAND = 0.2

# Sessions on which a market-wide circuit breaker stopped all trading for 15 minutes (Level 1 halts, March 2020):
# the minutes of the halt have no bar on any feed.
MARKET_WIDE_HALTS = {date(2020, 3, 9): "09:34-09:49", date(2020, 3, 12): "09:35-09:50", date(2020, 3, 16): "09:30-09:45", date(2020, 3, 18): "12:56-13:11"}
FEEDS = {"yahoo": "Yahoo Finance chart endpoint. Its feed is not documented; volumes are close to consolidated totals for liquid ETFs, and the last bar can end before the closing auction.",
         "alpaca": "Alpaca stock bars: \"sip\" is the consolidated tape of every exchange; \"iex\" is one exchange, a few percent of the volume. The dataset records which was requested.",
         "massive": "Massive (Polygon) aggregates: built from the consolidated tape.", "import": "An imported file: the feed is whatever the file's vendor used.",
         "optionsdx": "The underlying price printed in an option-quote file at each snapshot: not trade bars; no volume; high and low between snapshots are not observed."}


def _dst_changes(first: date, last: date) -> list[date]:
    """The first session on or after each change of the New York UTC offset inside the range."""
    out, d = [], first
    prev = datetime.combine(d, time(12), cal.ET).utcoffset()
    while d <= last:
        off = datetime.combine(d, time(12), cal.ET).utcoffset()
        if off != prev:
            x = d
            while not cal.is_trading_day(x):
                x += timedelta(days=1)
            out.append(x)
            prev = off
        d += timedelta(days=1)
    return out


def audit_dataset(store: MarketStore, sym: str, res: str, provider: str, daily: pl.DataFrame | None, today: date | None = None) -> dict:
    sec = SECONDS[res]
    # the dataset's own bars are audited, whether or not a higher-ranked feed is read first for the same sessions
    b = store.read_bars(sym, res, None, None, provider=provider, with_provider=True)
    out: dict = {"provider": provider, "resolution": res, "bars": b.height, "feed": FEEDS.get(provider, "not documented"), "findings": [], "sessions": 0}
    if b.is_empty():
        return out
    used = store.read_bars(sym, res, None, None, with_provider=True)
    mine = set(b.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))["_d"].unique().to_list())
    read_here = set(used.filter(pl.col("provider") == provider).select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))["_d"].unique().to_list()) if not used.is_empty() else set()
    others = sorted(set(used.filter(pl.col("provider") != provider)["provider"].unique().to_list())) if not used.is_empty() else []
    out["shadowed"] = {"sessions": len(mine - read_here), "by": others if mine - read_here else []}
    b = b.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et")).with_columns(pl.col("_et").dt.date().alias("_d"), (pl.col("_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_et").dt.minute().cast(pl.Int32)).alias("_m"))
    first, last = b["_d"].min(), b["_d"].max()
    days = cal.trading_days(first, last)
    have = set(b["_d"].unique().to_list())
    f = out["findings"]

    def add(check: str, n: int, detail: str, sample: list | None = None, level: str = "warn"):
        if n:
            f.append({"check": check, "sessions": int(n), "detail": detail, "sample": [str(x) for x in (sample or [])][:8], "level": level})

    missing = [d for d in days if d not in have]
    add("missing sessions", len(missing), f"the exchange was open on {len(missing)} session(s) between {first} and {last} for which no {res} bar is stored", missing[-8:])
    off_days = sorted(d for d in have if not cal.is_trading_day(d))
    add("bars on a non-session", len(off_days), "bars are stored on a weekend or an exchange holiday", off_days)
    close_min = {d: (cal.session_close_time(d).hour * 60 + cal.session_close_time(d).minute) for d in days}
    cm = pl.DataFrame({"_d": list(close_min), "_cm": list(close_min.values())}, schema={"_d": pl.Date, "_cm": pl.Int32})
    b = b.join(cm, on="_d", how="inner")
    outside = b.filter((pl.col("_m") < 570) | (pl.col("_m") >= pl.col("_cm")))
    add("bars outside regular hours", outside["_d"].n_unique(), f"{outside.height:,} bar(s) start before 09:30 or at or after the session's close, New York time: extended-hours prices are mixed into regular-session data",
        sorted(outside["_d"].unique().to_list())[-8:])
    reg = b.filter((pl.col("_m") >= 570) & (pl.col("_m") < pl.col("_cm")))
    g = reg.group_by("_d").agg(pl.len().alias("n"), pl.col("_m").min().alias("first"), pl.col("_m").max().alias("last"), pl.col("_cm").first(), pl.col("open").first().alias("o"), pl.col("high").max().alias("h"), pl.col("low").min().alias("l"),
                               pl.col("close").last().alias("c"), pl.col("volume").sum().alias("v")).sort("_d")
    g = g.with_columns(((pl.col("_cm") - 570) * 60 / sec).ceil().cast(pl.Int32).alias("expected"))
    out["sessions"] = g.height
    out["first"], out["last"] = str(first), str(last)
    step = sec // 60
    late = g.filter(pl.col("first") > 570 + max(step, 5))
    early = g.filter(pl.col("last") + step < pl.col("_cm") - max(step, 5))
    add("partial sessions", late.height + early.filter(~pl.col("_d").is_in(late["_d"].implode())).height,
        f"{late.height} session(s) start more than {max(step, 5)} minutes after the open and {early.height} stop more than that before the close: a session high, low or close taken from them would be wrong",
        sorted(set(late["_d"].to_list()) | set(early["_d"].to_list()))[-8:])
    whole = g.filter(~pl.col("_d").is_in(late["_d"].implode()) & ~pl.col("_d").is_in(early["_d"].implode()))
    gaps = whole.filter(pl.col("n") < pl.col("expected"))
    miss = int((gaps["expected"] - gaps["n"]).sum()) if gaps.height else 0
    halted = gaps.filter(pl.col("_d").is_in(list(MARKET_WIDE_HALTS)))
    in_halts = int((halted["expected"] - halted["n"]).sum()) if halted.height else 0
    add("missing intervals", gaps.height, f"{miss:,} {res} interval(s) have no bar inside otherwise whole sessions (" + (f"{miss / max(1, int(whole['expected'].sum())) * 100:.2f}% of the intervals" if whole.height else "") + "). "
        + ("A minute with no trade has no bar; for a liquid ETF that should be rare." if res == "1m" else "At this resolution every interval of a liquid ETF trades.")
        + (f" {in_halts:,} of them are on the {halted.height} session(s) with a market-wide trading halt ({', '.join(str(x) for x in halted['_d'].to_list())}), when nothing traded for about 15 minutes." if in_halts else ""),
        gaps.sort(pl.col("expected") - pl.col("n"), descending=True)["_d"].to_list()[:8],
        level="info" if whole.height and miss / max(1, int(whole["expected"].sum())) < 0.002 else "warn")
    out["intervals"] = {"expected": int(whole["expected"].sum()) if whole.height else 0, "missing": miss}
    # bars that carry the day's running total instead of their own volume
    from strata.data.quality import artefact_prone, cumulative_volume_artefacts
    art_bars, art_days = 0, []
    out["volume_rule"] = "applied" if artefact_prone(provider) else "not applied: this feed reports the consolidated tape's own volume, where one minute can hold a large block"
    if sec <= 300 and artefact_prone(provider):
        for (d,), part in reg.sort("ts").partition_by("_d", as_dict=True, maintain_order=True).items():
            bad = cumulative_volume_artefacts(part["volume"].fill_null(0).to_numpy(), sec)
            if bad.any():
                art_bars += int(bad.sum())
                art_days.append(d)
    out["volume_artefacts"] = {"bars": art_bars, "sessions": len(art_days)}
    add("cumulative volume in single bars", len(art_days), f"{art_bars:,} bar(s) on {len(art_days)} session(s) carry the day's running volume total instead of their own volume. Their volume is treated as unknown: they get no weight in VWAP or volume rules. Prices are unaffected.",
        art_days[-8:])
    # early closes: a 13:00 session must end at 13:00
    ec = [d for d in days if cal.session_close_time(d) < time(16, 0) and d in have]
    bad_ec = [d for d in ec if d in set(outside.filter(pl.col("_m") >= pl.col("_cm"))["_d"].to_list())]
    out["early_closes"] = {"sessions": len(ec), "with_bars_after_the_close": len(bad_ec)}
    add("bars after an early close", len(bad_ec), "on a session that closed at 13:00 there are bars after 13:00", bad_ec)
    # daylight saving: 09:30 New York time on the sessions around each change
    wrong = []
    changes = _dst_changes(first, last)
    for ch in changes:
        for d in (cal.prev_trading_day(ch), ch):
            row = g.filter(pl.col("_d") == d)
            if row.height and int(row["first"][0]) != 570 and d not in set(late["_d"].to_list()):
                wrong.append(d)
    out["daylight_saving"] = {"changes_in_range": [str(x) for x in changes], "sessions_not_starting_at_0930": len(wrong)}
    add("daylight saving", len(wrong), "the first bar is not at 09:30 New York time on a session next to a clock change: the timestamps were shifted by an hour", wrong)
    # the daily record
    if daily is not None and not daily.is_empty():
        j = whole.join(daily.rename({"date": "_d"}), on="_d", how="inner").with_columns(((pl.col("c") / pl.col("close") - 1) * 100).alias("dc"), ((pl.col("h") / pl.col("high") - 1) * 100).alias("dh"), ((pl.col("l") / pl.col("low") - 1) * 100).alias("dl"),
                                                                                             (pl.col("v") / pl.col("volume") * 100).alias("vp"))
        scale = j.filter(pl.col("dc").abs() >= SCALE_BAND * 100)
        add("different price scale", scale.height, "the last bar is more than 20% from the daily close: the two datasets are on different price scales (one of them is not split-adjusted)", scale["_d"].to_list())
        j = j.filter(pl.col("dc").abs() < SCALE_BAND * 100)
        gap = j.filter(pl.col("dc").abs() > CLOSE_GAP_PCT).sort(pl.col("dc").abs(), descending=True)
        add("last bar away from the official close", gap.height, f"the last {res} bar ends more than {CLOSE_GAP_PCT}% from the official close on {gap.height} session(s)"
            + (f" (largest: {gap['_d'][0]}, {gap['c'][0]:.2f} against {gap['close'][0]:.2f}, {gap['dc'][0]:+.2f}%)" if gap.height else "") + ". Day-level rules and settlement use the official close.", gap["_d"].to_list())
        rng = j.filter((pl.col("dh") < -CLOSE_GAP_PCT) | (pl.col("dl") > CLOSE_GAP_PCT))
        # where does the outlying daily extreme sit? At about the PREVIOUS close it is a late or reference-priced trade report, not a missing print
        prev = daily.sort("date").with_columns(pl.col("close").shift(1).alias("_pc")).select(pl.col("date").alias("_d"), "_pc")
        rp = rng.join(prev, on="_d", how="left")
        near_prev = rp.filter(((pl.col("dh") < -CLOSE_GAP_PCT) & ((pl.col("high") / pl.col("_pc") - 1).abs() <= 0.001)) | ((pl.col("dl") > CLOSE_GAP_PCT) & ((pl.col("low") / pl.col("_pc") - 1).abs() <= 0.001))).height if rng.height else 0
        out["daily_extreme_outside_bars"] = {"sessions": rng.height, "at_previous_close": near_prev}
        whole_feed = not artefact_prone(provider) and res == "1m"
        add("daily high or low outside the regular-session bars", rng.height,
            f"the daily record's high is above every {res} bar, or its low below every bar, by more than {CLOSE_GAP_PCT}% on {rng.height} session(s); on {near_prev} of them that extreme is within 0.1% of the PREVIOUS close. "
            + ("A daily bar follows the consolidated tape's rule for highs and lows, which counts late and reference-priced trade reports that belong to no minute of the session; these bars come from the same tape, so nothing is missing from them. "
               if whole_feed else "Either prints are missing from these bars, or the daily record counts late and reference-priced trade reports that belong to no bar (the usual case when the extreme sits at the previous close). ")
            + "A rule that reads the daily high or low sees a price nobody could trade at that moment: on sessions with intraday bars the engine takes the session's range from the bars.",
            rng["_d"].to_list()[-8:], level="info" if whole_feed else "warn")
        vol = j.filter(pl.col("vp").is_finite() & (pl.col("volume") > 0))
        if vol.height:
            med = float(vol["vp"].median())
            out["volume_share_of_daily_pct"] = round(med, 1)
            if med > 130:
                add("volume above the daily total", vol.height, f"the bars' volume sums to a median of {med:.0f}% of the daily volume, which is impossible for real per-bar volumes (see the cumulative-volume finding)", [])
            if med < 50:
                add("volume far below the daily total", vol.height, f"the bars carry a median of {med:.1f}% of the daily volume: this is not a consolidated feed (one exchange, or prints are missing)", [])
        out["compared_with_daily"] = {"sessions": j.height, "median_abs_close_difference_pct": round(float(j["dc"].abs().median()), 4) if j.height else None,
                                      "within_0_05_pct": int(j.filter(pl.col("dc").abs() <= 0.05).height), "largest_pct": round(float(j["dc"].abs().max()), 3) if j.height else None}
    out["ok"] = not any(x["level"] == "warn" for x in f)
    return out


def cross_feed(store: MarketStore, sym: str, res: str, providers: list[str]) -> list[dict]:
    """Two feeds that both hold a session are two independent records of the same minutes: how closely do their prices agree?
    Bars are matched by start time; nothing is merged or corrected."""
    out = []
    frames = {p: store.read_bars(sym, res, None, None, provider=p) for p in providers}
    for i, a in enumerate(providers):
        for b in providers[i + 1:]:
            fa, fb = frames[a], frames[b]
            if fa.is_empty() or fb.is_empty():
                continue
            j = fa.join(fb, on="ts", how="inner", suffix="_b")
            if j.is_empty():
                continue
            j = j.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"), ((pl.col("close_b") / pl.col("close") - 1).abs() * 100).alias("dc"),
                               ((pl.col("high_b") / pl.col("high") - 1).abs() * 100).alias("dh"), ((pl.col("low_b") / pl.col("low") - 1).abs() * 100).alias("dl"), ((pl.col("open_b") / pl.col("open") - 1).abs() * 100).alias("do"))
            days = j["_d"].n_unique()
            only_a = fa.join(fb, on="ts", how="anti").with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).filter(pl.col("_d").is_in(j["_d"].unique().implode())).height
            only_b = fb.join(fa, on="ts", how="anti").with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).filter(pl.col("_d").is_in(j["_d"].unique().implode())).height
            out.append({"a": a, "b": b, "sessions": int(days), "bars_compared": j.height, "bars_only_in_a": int(only_a), "bars_only_in_b": int(only_b),
                        "close_within_0_02_pct": round(j.filter(pl.col("dc") <= 0.02).height / j.height * 100, 2), "close_within_0_05_pct": round(j.filter(pl.col("dc") <= 0.05).height / j.height * 100, 2),
                        "median_abs_close_difference_pct": round(float(j["dc"].median()), 4), "largest_close_difference_pct": round(float(j["dc"].max()), 3),
                        "high_within_0_05_pct": round(j.filter(pl.col("dh") <= 0.05).height / j.height * 100, 2), "low_within_0_05_pct": round(j.filter(pl.col("dl") <= 0.05).height / j.height * 100, 2),
                        "open_within_0_05_pct": round(j.filter(pl.col("do") <= 0.05).height / j.height * 100, 2)})
    return out


def audit(store: MarketStore | None = None, sym: str = "SPY", today: date | None = None) -> dict:
    store = store or MarketStore()
    daily = store.read_daily(sym).select("date", "open", "high", "low", "close", "volume") if store.has_daily(sym) else None
    rows = []
    seen: dict[str, list[str]] = {}
    for k in store.datasets(sym, "underlying"):
        if k.res not in SECONDS:
            continue
        r = audit_dataset(store, sym, k.res, k.provider, daily, today)
        r["dataset"] = k.id
        d = store.descriptor(k)
        r["adjustment"] = d.get("adjustment")
        r["source_feed"] = d.get("feed") or (d.get("source") if isinstance(d.get("source"), str) else None)
        rows.append(r)
        seen.setdefault(k.res, []).append(k.provider)
    mixed = [{"resolution": res, "providers": ps, "note": "Two providers store bars at this resolution. A session is read from one of them, never blended; their feeds differ: " + " | ".join(f"{p}: {FEEDS.get(p, 'not documented')}" for p in ps),
              "agreement": cross_feed(store, sym, res, ps)} for res, ps in seen.items() if len(ps) > 1]
    cov = {}
    if daily is not None and not daily.is_empty():
        best = store.read_best_bars(sym)
        if not best.is_empty():
            best = best.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d"))
            per = best.group_by("_d").agg(pl.col("res").min()).group_by("res").agg(pl.len().alias("n"), pl.col("_d").min().alias("first"), pl.col("_d").max().alias("last")).sort("res")
            cov = {("daily only" if int(r["res"]) >= 86400 else f"{int(r['res']) // 60}-minute"): {"sessions": int(r["n"]), "first": str(r["first"]), "last": str(r["last"])} for r in per.iter_rows(named=True)}
    return {"symbol": sym, "daily": None if daily is None else {"sessions": daily.height, "first": str(daily["date"][0]), "last": str(daily["date"][-1])}, "finest_resolution_by_session": cov, "datasets": rows, "mixed_feeds": mixed,
            "warnings": sum(1 for r in rows for x in r["findings"] if x["level"] == "warn"),
            "routes_to_more": ["Alpaca free account: 1-minute bars from the consolidated tape since 2016-01-04, delayed 15 minutes (built in; needs the key): `strata download alpaca bars_1m SYMBOL`, or Data, Download.",
                               "Massive (Polygon) free plan: 2 years of 1-minute aggregates, 5 requests a minute (adapter built; never run with a real key here).",
                               "Yahoo (in use): 1-minute bars for about 30 days, 5-minute for 60 days, hourly for about 2 years. Older sessions only exist if they were collected while available.",
                               "University of Washington: TAQ through the Foster School's WRDS subscription has every trade and quote; access is the school's decision and its use rules do not allow automated downloading.",
                               "An OptionsDX option file carries the underlying price at each snapshot: enough to run that session, not a substitute for trade bars."]}


def render(rep: dict) -> str:
    L = [f"# Intraday bar audit: {rep['symbol']}", ""]
    if rep.get("daily"):
        L.append(f"Daily record: {rep['daily']['sessions']:,} sessions, {rep['daily']['first']} to {rep['daily']['last']}.")
    for k, v in rep["finest_resolution_by_session"].items():
        L.append(f"- finest bars {k}: {v['sessions']:,} sessions ({v['first']} to {v['last']})")
    for r in rep["datasets"]:
        L += ["", f"## {r['dataset']}", f"{r['bars']:,} bars over {r['sessions']:,} sessions" + (f", {r.get('first')} to {r.get('last')}" if r.get("first") else "") + f". Feed: {r['feed']}"]
        sh = r.get("shadowed") or {}
        if sh.get("sessions"):
            L.append(f"{sh['sessions']:,} of these sessions are read from {', '.join(sh['by'])} instead, which ranks higher at this resolution: this dataset is kept but not used for them.")
        c = r.get("compared_with_daily")
        if c:
            L.append(f"Last bar against the official close on {c['sessions']:,} whole sessions: median difference {c['median_abs_close_difference_pct']}%, {c['within_0_05_pct']:,} within 0.05%, largest {c['largest_pct']}%." + (f" Volume: {r['volume_share_of_daily_pct']}% of the daily total (median)." if "volume_share_of_daily_pct" in r else ""))
        if r.get("daylight_saving"):
            L.append(f"Clock changes in range: {len(r['daylight_saving']['changes_in_range'])}; sessions next to one that do not start at 09:30 New York time: {r['daylight_saving']['sessions_not_starting_at_0930']}. Early closes: {r['early_closes']['sessions']}, with bars after the close: {r['early_closes']['with_bars_after_the_close']}.")
        for x in r["findings"]:
            L.append(f"- {'WARNING' if x['level'] == 'warn' else 'note'}: {x['check']}: {x['detail']}" + (f" ({', '.join(x['sample'])})" if x["sample"] else ""))
        if not r["findings"]:
            L.append("- no finding")
    for m in rep["mixed_feeds"]:
        L += ["", f"Mixed feeds at {m['resolution']}: {m['note']}"]
        for g in m.get("agreement") or []:
            L.append(f"- {g['a']} against {g['b']} on the {g['sessions']} session(s) both hold: {g['bars_compared']:,} bars matched by start time; close within 0.02% on {g['close_within_0_02_pct']}% and within 0.05% on {g['close_within_0_05_pct']}% "
                     f"(median difference {g['median_abs_close_difference_pct']}%, largest {g['largest_close_difference_pct']}%); high within 0.05% on {g['high_within_0_05_pct']}%, low on {g['low_within_0_05_pct']}%, open on {g['open_within_0_05_pct']}%; "
                     f"{g['bars_only_in_a']:,} bar(s) only in {g['a']}, {g['bars_only_in_b']:,} only in {g['b']}.")
    L += ["", "Routes to more intraday history:"] + [f"- {x}" for x in rep["routes_to_more"]]
    return "\n".join(L) + "\n"
