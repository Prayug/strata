"""Stored option trade bars, checked three ways. Nothing here repairs or fills anything.

  audit_option_bars      what is stored, session by session: contracts asked for, contracts that traded, bars and the
                         hours they span; sessions the exchange was open on that are absent, empty or thin
  verify_against_trades  a sample of (session, contract) pairs is fetched again from the provider as INDIVIDUAL TRADES
                         and rebuilt into one-minute bars here. A stored bar is confirmed when its open, high, low,
                         close and volume equal the bar rebuilt from those trades: a second record of the same minutes,
                         from a different endpoint, through none of the code that stored the bars
  cross_provider         where two providers hold the same contract on the same session, their minute prices are compared

`verify_against_trades` needs the provider's key and makes read-only requests; without one it is BLOCKED and says so.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timezone

import polars as pl

from strata.core import calendar as cal
from strata.data.store import MarketStore

THIN = 0.35  # a session with fewer bars than this share of the median of its neighbours is reported
NEIGHBOURS = 10


def _et_min(col: str = "ts") -> pl.Expr:
    et = pl.col(col).dt.convert_time_zone("America/New_York")
    return et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)


def audit_option_bars(store: MarketStore | None = None, sym: str = "SPY", provider: str | None = None, progress=None) -> dict:
    """What is stored for one option trade-bar dataset, by session, against the exchange calendar."""
    store = store or MarketStore()
    ds = store.option_dataset(sym, "trade", provider)
    out: dict = {"symbol": sym, "dataset": ds.id if ds else None, "provider": ds.provider if ds else provider, "sessions": 0, "findings": []}
    if ds is None:
        out["note"] = f"no option trade bars are stored for {sym}" + (f" from {provider}" if provider else "")
        return out
    idx = store.option_bar_index(sym, dataset=ds)
    if idx.is_empty():
        out["note"] = "the dataset has no index"
        return out
    per = idx.group_by("date").agg(pl.len().alias("contracts"), (pl.col("status") == "ok").sum().alias("traded"), pl.col("bars").sum().alias("bars"),
                                   (pl.col("expiration") == pl.col("date")).sum().alias("contracts_0dte"), ((pl.col("expiration") == pl.col("date")) & (pl.col("status") == "ok")).sum().alias("traded_0dte"),
                                   pl.col("expiration").n_unique().alias("expirations"), pl.col("strike").min().alias("strike_lo"), pl.col("strike").max().alias("strike_hi")).sort("date")
    first, last = per["date"][0], per["date"][-1]
    days = cal.trading_days(first, last)
    have = set(per["date"].to_list())
    absent = [d for d in days if d not in have]
    empty = per.filter(pl.col("bars") == 0)["date"].to_list()
    # the hours the bars span, from the files themselves (the index does not hold times)
    spans = []
    live = per.filter(pl.col("bars") > 0)["date"].to_list()
    for i, d in enumerate(live):
        if progress and i % 50 == 0:
            progress(i / max(1, len(live)), f"reading {d}")
        b = store.read_option_bars(sym, d, dataset=ds)
        if b is None or b.is_empty():
            spans.append((d, None, None, 0))
            continue
        m = b.select(_et_min().alias("m"))["m"]
        spans.append((d, int(m.min()), int(m.max()), b.height))
    sp = pl.DataFrame(spans, schema={"date": pl.Date, "first_min": pl.Int32, "last_min": pl.Int32, "rows": pl.Int64}, orient="row") if spans else pl.DataFrame(schema={"date": pl.Date, "first_min": pl.Int32, "last_min": pl.Int32, "rows": pl.Int64})
    per = per.join(sp, on="date", how="left")
    close_min = {d: cal.session_close_time(d).hour * 60 + cal.session_close_time(d).minute for d in per["date"].to_list()}
    per = per.with_columns(pl.col("date").replace_strict(close_min, return_dtype=pl.Int32).alias("close_min"))
    mismatch = per.filter((pl.col("bars") > 0) & (pl.col("rows") != pl.col("bars")))
    late = per.filter((pl.col("bars") > 0) & (pl.col("first_min") > 575))
    early = per.filter((pl.col("bars") > 0) & (pl.col("last_min") < pl.col("close_min") - 6))
    # thin sessions: against the median of the sessions around them
    bars = per["bars"].to_list()
    thin = []
    for i, d in enumerate(per["date"].to_list()):
        around = sorted(bars[max(0, i - NEIGHBOURS):i] + bars[i + 1:i + 1 + NEIGHBOURS])
        med = around[len(around) // 2] if around else 0
        if med and 0 < bars[i] < THIN * med:
            thin.append((d, bars[i], med))
    f = out["findings"]

    def add(check, n, detail, sample, level="warn"):
        if n:
            f.append({"check": check, "sessions": int(n), "detail": detail, "sample": [str(x) for x in sample][:12], "level": level})

    add("sessions not downloaded", len(absent), f"the exchange was open on {len(absent)} session(s) between {first} and {last} that have no entry in this dataset: nothing was asked for them", absent[-12:])
    add("sessions with no trade at the provider", len(empty), f"{len(empty)} session(s) were asked for and returned no bar for ANY contract: the provider has no option history for them", empty)
    add("thin sessions", len(thin), f"{len(thin)} session(s) hold less than {THIN:.0%} of the bars of the sessions around them: part of the session is missing at the provider, it was a short session, "
        "or the underlying opened far from the previous close and traded away from the stored strikes (the band is taken around the previous close)",
        [f"{d} ({n:,} bars against a median of {m:,})" for d, n, m in thin])
    add("sessions that start late", late.height, f"on {late.height} session(s) the first stored bar is after 09:35 New York time", late["date"].to_list()[-12:])
    add("sessions that stop early", early.height, f"on {early.height} session(s) the last stored bar is more than 6 minutes before the close", early["date"].to_list()[-12:])
    add("index and files disagree", mismatch.height, f"on {mismatch.height} session(s) the bars counted in the index differ from the rows in the session's file", mismatch["date"].to_list()[-12:])
    half = [d for d in per["date"].to_list() if close_min[d] < 960]
    whole = per.filter(pl.col("bars") > 0)
    out.update(sessions=per.height, first=str(first), last=str(last), calendar_sessions=len(days), sessions_with_trades=int(whole.height),
               contract_sessions=int(per["contracts"].sum()), contract_sessions_with_trades=int(per["traded"].sum()), bars=int(per["bars"].sum()),
               contract_sessions_0dte=int(per["contracts_0dte"].sum()), contract_sessions_0dte_with_trades=int(per["traded_0dte"].sum()),
               sessions_with_a_0dte_contract=int(per.filter(pl.col("contracts_0dte") > 0).height), sessions_without_a_0dte_contract=[str(d) for d in per.filter(pl.col("contracts_0dte") == 0)["date"].to_list()][:40],
               median_bars_per_session=int(whole["bars"].median()) if whole.height else 0, median_contracts_per_session=int(per["contracts"].median()), early_close_sessions=len(half),
               observation="trade bars: a contract has a bar only in the minutes it traded; there is no bid or ask",
               ok=not any(x["level"] == "warn" and x["check"] != "sessions not downloaded" for x in f))
    out["by_year"] = [{"year": int(r["y"]), "sessions": int(r["n"]), "contract_sessions": int(r["c"]), "with_trades": int(r["t"]), "bars": int(r["b"])}
                      for r in per.with_columns(pl.col("date").dt.year().alias("y")).group_by("y").agg(pl.len().alias("n"), pl.col("contracts").sum().alias("c"), pl.col("traded").sum().alias("t"), pl.col("bars").sum().alias("b")).sort("y").iter_rows(named=True)]
    return out


# ----------------------------------------------------------------------------- stored bars against the provider's own trades
def bars_from_trades(trades: list[dict]) -> pl.DataFrame:
    """One-minute bars rebuilt from individual trades: t (time), p (price), s (size). Written here, sharing nothing with the download."""
    if not trades:
        return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64, "n": pl.Int64})
    rows: dict[datetime, list] = {}
    for tr in sorted(trades, key=lambda x: x["t"]):
        t = datetime.fromisoformat(tr["t"].replace("Z", "+00:00")[:19] + "+00:00")  # whole seconds are enough to name the minute
        k = t.replace(second=0, microsecond=0)
        p, sz = float(tr["p"]), int(tr.get("s") or 0)
        r = rows.get(k)
        if r is None:
            rows[k] = [p, p, p, p, sz, 1]
        else:
            r[1], r[2], r[3], r[4], r[5] = max(r[1], p), min(r[2], p), p, r[4] + sz, r[5] + 1
    return pl.DataFrame([(k, *v) for k, v in sorted(rows.items())], schema={"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64, "n": pl.Int64}, orient="row")


def compare_bars(stored: pl.DataFrame, rebuilt: pl.DataFrame) -> dict:
    """Minute by minute: which stored bars equal the bar rebuilt from the trades."""
    a = stored.select("ts", "open", "high", "low", "close", pl.col("volume").cast(pl.Int64))
    j = a.join(rebuilt, on="ts", how="full", suffix="_t", coalesce=True)
    both = j.filter(pl.col("open").is_not_null() & pl.col("open_t").is_not_null())
    eq = lambda c: (pl.col(c) - pl.col(c + "_t")).abs() < 5e-7
    prices = both.filter(eq("open") & eq("high") & eq("low") & eq("close"))
    exact = prices.filter(pl.col("volume") == pl.col("volume_t"))
    return {"stored_bars": a.height, "rebuilt_bars": rebuilt.height, "minutes_in_both": both.height, "only_stored": j.filter(pl.col("open_t").is_null()).height, "only_in_trades": j.filter(pl.col("open").is_null()).height,
            "prices_identical": prices.height, "prices_and_volume_identical": exact.height,
            "close_identical": both.filter(eq("close")).height, "largest_close_difference": round(float((both["close"] - both["close_t"]).abs().max()), 4) if both.height else 0.0}


def _sample(idx: pl.DataFrame, n: int, seed: int) -> list[dict]:
    """Pairs spread over the whole period AND over quiet and busy contracts: every (quarter, activity band) cell in a seeded
    random order, one pair from each until n are found. The busiest band is the at-the-money contracts that trade every minute."""
    ok = idx.filter((pl.col("status") == "ok") & (pl.col("bars") >= 5)).with_columns((pl.col("date").dt.year() * 4 + (pl.col("date").dt.month() - 1) // 3).alias("q"))
    if ok.is_empty():
        return []
    rng = random.Random(seed)
    bands = [(5, 40), (40, 150), (150, 300), (300, 100_000)]
    cells = [(q, b) for q in sorted(ok["q"].unique().to_list()) for b in bands]
    rng.shuffle(cells)
    # the bands in turn, so that a small sample still holds every kind of contract
    cells.sort(key=lambda c: 0)  # stable: keeps the shuffled order
    by_band = {b: [c for c in cells if c[1] == b] for b in bands}
    order = [by_band[b][i] for i in range(max(len(v) for v in by_band.values())) for b in bands if i < len(by_band[b])]
    out, seen = [], set()
    for rounds in range(8):
        for q, (lo, hi) in order:
            if len(out) >= n:
                return out
            cand = ok.filter((pl.col("q") == q) & (pl.col("bars") >= lo) & (pl.col("bars") < hi))
            if cand.is_empty():
                continue
            r = cand.row(rng.randrange(cand.height), named=True)
            if (r["date"], r["occ"]) not in seen:
                seen.add((r["date"], r["occ"]))
                out.append(r)
    return out


def verify_against_trades(store: MarketStore | None = None, sym: str = "SPY", sample: int = 24, seed: int = 20261125, progress=None, max_pages: int = 12, save: bool = True) -> dict:
    """Fetch a sample of stored (session, contract) pairs again as individual trades and compare the rebuilt bars with the stored ones."""
    from strata.data.providers import access as A
    from strata.data.providers import sources as S
    store = store or MarketStore()
    src = S.Alpaca(store)
    rep: dict = {"symbol": sym, "provider": "alpaca", "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "blocked", "pairs": [], "seed": seed}
    ok, why = src.available()
    if not ok:
        rep["reason"] = f"BLOCKED: {why}. The comparison needs the provider's trade records, which need the key."
        return _save(rep, save)
    ds = store.option_dataset(sym, "trade", "alpaca")
    if ds is None:
        rep.update(status="nothing_stored", reason=f"no Alpaca option bars are stored for {sym}")
        return _save(rep, save)
    pairs = _sample(store.option_bar_index(sym, dataset=ds), sample, seed)
    H = src._h()
    th = S.Throttle(src.rate)
    with S._client() as client:
        for i, r in enumerate(pairs):
            if progress:
                progress(i / max(1, len(pairs)), f"{r['date']} {r['occ']}")
            d = r["date"]
            a, b = cal.session_bounds(d)
            trades, token, pages, complete = [], None, 0, True
            while True:
                th.wait()
                x = A.probe(client, f"{src.DATA}/v1beta1/options/trades", {"symbols": r["occ"], "start": A._iso(a), "end": A._iso(b), "limit": 10000, "sort": "asc", **({"page_token": token} if token else {})}, H)
                pages += 1
                if x["status"] != 200:
                    complete = False
                    break
                trades += ((x["_data"] or {}).get("trades") or {}).get(r["occ"], [])
                token = (x["_data"] or {}).get("next_page_token")
                if not token:
                    break
                if pages >= max_pages:
                    complete = False
                    break
            row = {"date": str(d), "contract": r["occ"], "dte": (r["expiration"] - d).days, "bars_in_index": int(r["bars"]), "trades": len(trades), "pages": pages, "complete": complete}
            if not complete:
                row["note"] = "the trades were not read to the end (too many pages, or the request failed): not compared"
                rep["pairs"].append(row)
                continue
            stored = store.read_option_bars(sym, d, dataset=ds)
            stored = stored.filter(pl.col("occ") == r["occ"]) if stored is not None else pl.DataFrame()
            o, c = (x.astimezone(timezone.utc) for x in cal.session_bounds(d))
            rebuilt = bars_from_trades(trades).filter((pl.col("ts") >= o) & (pl.col("ts") < c))
            row.update(compare_bars(stored, rebuilt))
            conds: dict[str, int] = {}
            for tr in trades:
                conds[str(tr.get("c") or "")] = conds.get(str(tr.get("c") or ""), 0) + 1
            row["trade_conditions"] = dict(sorted(conds.items(), key=lambda kv: -kv[1])[:6])
            rep["pairs"].append(row)
    done = [p for p in rep["pairs"] if p.get("complete")]
    tot = lambda k: sum(int(p.get(k) or 0) for p in done)
    rep.update(status="ran" if done else "failed", pairs_compared=len(done), pairs_not_compared=len(rep["pairs"]) - len(done), stored_bars=tot("stored_bars"), rebuilt_bars=tot("rebuilt_bars"), minutes_in_both=tot("minutes_in_both"),
               only_stored=tot("only_stored"), only_in_trades=tot("only_in_trades"), prices_identical=tot("prices_identical"), prices_and_volume_identical=tot("prices_and_volume_identical"), close_identical=tot("close_identical"),
               trades_read=tot("trades"))
    sb = max(1, rep["stored_bars"])
    rep["verified"] = bool(done) and rep["prices_identical"] / sb >= 0.995 and rep["only_stored"] == 0
    rep["summary"] = (f"{len(done)} contract-sessions between {min(p['date'] for p in done)} and {max(p['date'] for p in done)} were fetched again as {rep['trades_read']:,} individual trades and rebuilt into one-minute bars. "
                      f"{rep['prices_identical']:,} of the {rep['stored_bars']:,} stored bars have the same open, high, low and close as the rebuilt bar ({rep['prices_identical'] / sb:.2%}); "
                      f"{rep['prices_and_volume_identical']:,} also have the same volume ({rep['prices_and_volume_identical'] / sb:.2%}). "
                      f"{rep['only_stored']:,} stored bar(s) have no trade behind them; {rep['only_in_trades']:,} minute(s) with trades have no stored bar.") if done else "No pair could be compared."
    return _save(rep, save)


def _save(rep: dict, save: bool) -> dict:
    if save:
        from strata.config import get_settings
        d = get_settings().data_dir / "reports"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"option_bars_verification_{rep['symbol']}.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def saved(sym: str = "SPY") -> dict | None:
    from strata.config import get_settings
    f = get_settings().data_dir / "reports" / f"option_bars_verification_{sym}.json"
    return json.loads(f.read_text()) if f.exists() else None


# ----------------------------------------------------------------------------- two providers, the same contract and minute
def cross_provider(store: MarketStore | None = None, sym: str = "SPY") -> list[dict]:
    """Every pair of option trade-bar datasets of a symbol: on the (session, contract, minute) triples both hold, how their prices compare."""
    store = store or MarketStore()
    keys = [k for k in store.datasets(sym, "options", "trade")]
    out = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            ia, ib = store.option_bar_index(sym, dataset=ka), store.option_bar_index(sym, dataset=kb)
            if ia.is_empty() or ib.is_empty():
                continue
            common = ia.filter(pl.col("status") == "ok").join(ib.filter(pl.col("status") == "ok"), on=["date", "occ"], how="inner")
            if common.is_empty():
                out.append({"a": ka.provider, "b": kb.provider, "sessions": 0, "contract_sessions": 0, "note": "no contract-session is held by both"})
                continue
            tot = {"bars_a": 0, "bars_b": 0, "both": 0, "close_eq": 0, "ohlc_eq": 0, "vol_eq": 0, "only_a": 0, "only_b": 0}
            worst = 0.0
            for (d,), part in common.partition_by("date", as_dict=True).items():
                occs = part["occ"].to_list()
                a = store.read_option_bars(sym, d, dataset=ka)
                b = store.read_option_bars(sym, d, dataset=kb)
                if a is None or b is None:
                    continue
                a = a.filter(pl.col("occ").is_in(occs)).select("occ", "ts", "open", "high", "low", "close", pl.col("volume").cast(pl.Int64))
                b = b.filter(pl.col("occ").is_in(occs)).select("occ", "ts", "open", "high", "low", "close", pl.col("volume").cast(pl.Int64))
                j = a.join(b, on=["occ", "ts"], how="full", suffix="_b", coalesce=True)
                both = j.filter(pl.col("close").is_not_null() & pl.col("close_b").is_not_null())
                eq = lambda c: (pl.col(c) - pl.col(c + "_b")).abs() < 5e-7
                tot["bars_a"] += a.height
                tot["bars_b"] += b.height
                tot["both"] += both.height
                tot["close_eq"] += both.filter(eq("close")).height
                tot["ohlc_eq"] += both.filter(eq("open") & eq("high") & eq("low") & eq("close")).height
                tot["vol_eq"] += both.filter(pl.col("volume") == pl.col("volume_b")).height
                tot["only_a"] += j.filter(pl.col("close_b").is_null()).height
                tot["only_b"] += j.filter(pl.col("close").is_null()).height
                if both.height:
                    worst = max(worst, float((both["close"] - both["close_b"]).abs().max()))
            n = max(1, tot["both"])
            out.append({"a": ka.provider, "b": kb.provider, "sessions": int(common["date"].n_unique()), "first": str(common["date"].min()), "last": str(common["date"].max()), "contract_sessions": common.height,
                        "bars_a": tot["bars_a"], "bars_b": tot["bars_b"], "minutes_in_both": tot["both"], "only_in_a": tot["only_a"], "only_in_b": tot["only_b"],
                        "close_identical_pct": round(tot["close_eq"] / n * 100, 2), "ohlc_identical_pct": round(tot["ohlc_eq"] / n * 100, 2), "volume_identical_pct": round(tot["vol_eq"] / n * 100, 2), "largest_close_difference": round(worst, 4)})
    return out


def render(aud: dict, ver: dict | None = None, cross: list[dict] | None = None) -> str:
    L = [f"# Option trade bars: {aud['symbol']}" + (f" ({aud['dataset']})" if aud.get("dataset") else ""), ""]
    if not aud.get("sessions"):
        L.append(aud.get("note") or "nothing stored")
    else:
        L += [f"{aud['sessions']:,} sessions stored of the {aud['calendar_sessions']:,} the exchange was open on between {aud['first']} and {aud['last']}; {aud['sessions_with_trades']:,} hold at least one bar.",
              f"{aud['contract_sessions']:,} contract-sessions were asked for, {aud['contract_sessions_with_trades']:,} traded: {aud['bars']:,} one-minute bars. Median per session: {aud['median_contracts_per_session']:,} contracts, {aud['median_bars_per_session']:,} bars.",
              f"Same-day expirations: {aud['contract_sessions_0dte']:,} contract-sessions on {aud['sessions_with_a_0dte_contract']:,} sessions, {aud['contract_sessions_0dte_with_trades']:,} traded.",
              f"Observation: {aud['observation']}."]
        for y in aud.get("by_year", []):
            L.append(f"- {y['year']}: {y['sessions']} sessions, {y['with_trades']:,} of {y['contract_sessions']:,} contract-sessions traded, {y['bars']:,} bars")
        for x in aud["findings"]:
            L.append(f"- {'WARNING' if x['level'] == 'warn' else 'note'}: {x['check']}: {x['detail']}" + (f" ({', '.join(x['sample'])})" if x["sample"] else ""))
        if not aud["findings"]:
            L.append("- no finding")
    if ver:
        L += ["", "## Stored bars against the provider's own trades", ver.get("summary") or ver.get("reason") or ver.get("status", "")]
        bad = [p for p in ver.get("pairs", []) if p.get("complete") and p.get("prices_identical") != p.get("stored_bars")]
        for p in bad[:8]:
            L.append(f"- {p['date']} {p['contract']}: {p['prices_identical']} of {p['stored_bars']} bars identical in price, {p['only_stored']} stored without a trade, {p['only_in_trades']} minutes of trades without a bar; trade conditions {p.get('trade_conditions')}")
    for c in cross or []:
        if not c.get("contract_sessions"):
            continue
        L += ["", f"## {c['a']} against {c['b']}", f"{c['contract_sessions']:,} contract-sessions on {c['sessions']} session(s) ({c['first']} to {c['last']}) are held by both: {c['bars_a']:,} bars in {c['a']}, {c['bars_b']:,} in {c['b']}, {c['minutes_in_both']:,} minutes in both. "
              f"Close identical on {c['close_identical_pct']}% of them, all four prices on {c['ohlc_identical_pct']}%, volume on {c['volume_identical_pct']}%; largest close difference {c['largest_close_difference']}. "
              f"{c['only_in_a']:,} minute(s) only in {c['a']}, {c['only_in_b']:,} only in {c['b']}."]
    return "\n".join(L) + "\n"
