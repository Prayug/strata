"""What can be researched with the data that is actually stored.

Five capabilities, each counted from local files and from nothing else (not from what a
provider advertises, not from what an adapter could download):

  daily            daily bars of the underlying
  intraday         a whole session of intraday bars of the underlying
  eod_options      an end-of-day option chain (one snapshot at the close): cannot price an
                   entry before the close or a 0DTE trade
  option_trades    intraday option TRADE prints for the session, with the fine underlying
                   bars their execution needs: prices of trades that happened, no bid/ask
  option_quotes    intraday option bid/ask quotes for the whole session

`precheck` answers the question for one strategy before anything is planned: on which of
the sessions its entry rule fires is there data to price it, with what, and what could be
done about the rest. A strategy that cannot be evaluated is never quietly turned into a
different one: the alternatives are listed for the user to choose.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from strata.data.store import MarketStore

CAPS = [("daily", "Daily underlying research", "How often, after what, by how much: any study of daily bars."),
        ("intraday", "Intraday underlying research", "Studies and rules that read the session as it unfolds (a time of day, a move from the open)."),
        ("eod_options", "End-of-day options research", "Strategies entered and exited at the close, on end-of-day bid/ask. Not 0DTE, not intraday entries."),
        ("option_trades", "Intraday options backtests on trade prices", "Intraday entries and exits priced on observed trade prints. No bid/ask: the spread is not known."),
        ("option_quotes", "Intraday options backtests on bid/ask quotes", "Intraday entries and exits priced on observed bid and ask.")]


def stored_option_sessions(store: MarketStore, sym: str) -> dict:
    """Per session, the intraday option observations that are stored: {"sessions": {date: [...]}, "eod_expirations": {iso: [dte...]}, "eod_dataset": id | None}."""
    by: dict[date, list] = {}
    eod_exp, eod_key = {}, None
    for k in store.datasets(sym):
        if k.dtype != "options":
            continue
        if k.obs == "quote" and k.res == "EOD":
            from strata.data import planner as PL
            eod_exp, eod_key = PL.expiration_index(store, k), k
            continue
        if k.obs == "quote":
            states = store.session_states(k)
            for d in store.option_days(sym, "quote", dataset=k):
                st = states.get(d.isoformat()) or {}
                by.setdefault(d, []).append({"provider": k.provider, "observation": "bid/ask quotes", "resolution": k.res, "status": st.get("status") or "complete",
                                             "window": [st.get("first_snapshot"), st.get("last_snapshot")] if st.get("first_snapshot") else None, "dataset": k.id})
        elif k.obs == "trade":
            idx = store.option_bar_index(sym, dataset=k)
            if not idx.is_empty() and "date" in idx.columns:
                states = store.session_states(k)
                for d in sorted(set(idx["date"].to_list())):
                    st = states.get(d.isoformat()) or {}
                    by.setdefault(d, []).append({"provider": k.provider, "observation": "trade prints", "resolution": k.res, "status": "partial" if st.get("status") == "partial" else "complete",
                                                 "window": [st.get("first_bar"), st.get("last_bar")] if st.get("first_bar") else None, "reason": st.get("reason"), "dataset": k.id})
    return {"sessions": by, "eod_expirations": eod_exp, "eod_dataset": eod_key.id if eod_key else None}


def _span(days: list[date]) -> dict:
    days = sorted(days)
    by: dict[int, int] = {}
    for d in days:
        by[d.year] = by.get(d.year, 0) + 1
    return {"sessions": len(days), "first": days[0].isoformat() if days else None, "last": days[-1].isoformat() if days else None, "by_year": by}


def symbol_readiness(store: MarketStore, sym: str) -> dict:
    daily = store.read_daily(sym)["date"].to_list() if store.has_daily(sym) else []
    fine: dict[date, int] = {}
    if daily:
        b = store.read_best_bars(sym)
        if not b.is_empty():
            g = b.filter(pl.col("res") < 86400).with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_d")).group_by("_d").agg(pl.col("res").min())
            fine = {r["_d"]: int(r["res"]) for r in g.iter_rows(named=True)}
    st = stored_option_sessions(store, sym)
    eod = [date.fromisoformat(d) for d in st["eod_expirations"]]
    # a session a provider holds only in part (its history starts late, stops early or is absent) is stored but is not priced
    trades = [d for d, items in st["sessions"].items() if any(i["observation"] == "trade prints" and i["status"] == "complete" for i in items)]
    part_trades = [d for d, items in st["sessions"].items() if any(i["observation"] == "trade prints" and i["status"] != "complete" for i in items) and d not in trades]
    quotes = [d for d, items in st["sessions"].items() if any(i["observation"] == "bid/ask quotes" and i["status"] == "complete" for i in items)]
    partial = [d for d, items in st["sessions"].items() if any(i["observation"] == "bid/ask quotes" and i["status"] != "complete" for i in items) and d not in quotes]
    usable_trades = [d for d in trades if fine.get(d, 10 ** 9) <= 300]
    rows = {"daily": {**_span(daily), "detail": "from the daily record"},
            "intraday": {**_span(list(fine)), "detail": "sessions with a whole session of intraday bars", "by_resolution": {f"{k // 60}-minute": v for k, v in sorted(_count(fine).items())}},
            "eod_options": {**_span(eod), "detail": "end-of-day chains" + (f" ({st['eod_dataset']})" if st["eod_dataset"] else "")},
            "option_trades": {**_span(usable_trades), "detail": "trade prints with 5-minute or finer underlying bars",
                              **({"unusable": len(trades) - len(usable_trades), "unusable_why": "option prints are stored but the underlying has no 5-minute or finer bars on those sessions"} if len(trades) > len(usable_trades) else {}),
                              **({"partial_sessions": len(part_trades), "partial_note": "the provider holds only part of the session (its option history starts late, stops early or is absent): stored, not priced",
                                  "partial_sample": [d.isoformat() for d in sorted(part_trades)][-8:]} if part_trades else {})},
            "option_quotes": {**_span(quotes), "detail": "whole sessions of intraday bid/ask", **({"partial_sessions": len(partial), "partial_note": "observed for part of the session only (a vendor sample): priced by the independent reference, not run by the engine",
                                                                                                    "partial_sample": [d.isoformat() for d in sorted(partial)][-6:]} if partial else {})}}
    caps = []
    for cid, title, what in CAPS:
        r = rows[cid]
        caps.append({"id": cid, "title": title, "answers": what, "ready": r["sessions"] > 0, **r})
    years = sorted({y for c in caps for y in c["by_year"]})
    return {"symbol": sym, "synthetic": store.is_synthetic(sym), "capabilities": caps, "years": years, "matrix": [{"year": y, **{c["id"]: c["by_year"].get(y, 0) for c in caps}} for y in years],
            "summary": _summary(sym, caps)}


def _count(fine: dict) -> dict:
    out: dict[int, int] = {}
    for v in fine.values():
        out[v] = out.get(v, 0) + 1
    return out


def _summary(sym: str, caps: list[dict]) -> str:
    c = {x["id"]: x for x in caps}
    parts = [f"{sym}: daily research on {c['daily']['sessions']:,} sessions" + (f" ({c['daily']['first'][:4]} to {c['daily']['last'][:4]})" if c["daily"]["sessions"] else "")]
    parts.append(f"intraday research on {c['intraday']['sessions']:,}")
    parts.append(f"end-of-day option chains on {c['eod_options']['sessions']:,}")
    it, iq = c["option_trades"]["sessions"], c["option_quotes"]["sessions"]
    parts.append(f"intraday option backtests on {it:,} session{'s' if it != 1 else ''} of trade prints and {iq:,} of bid/ask" + (f" (plus {c['option_quotes']['partial_sessions']} partly observed)" if c["option_quotes"].get("partial_sessions") else ""))
    return "; ".join(parts) + "."


def readiness(store: MarketStore | None = None, symbols: list[str] | None = None) -> dict:
    store = store or MarketStore()
    syms = symbols or sorted(s for s in store.stored_symbols() if not store.is_synthetic(s) and store.symbol_meta(s).kind not in ("volatility_index",))
    rows = [symbol_readiness(store, s) for s in syms]
    return {"symbols": rows, "capabilities": [{"id": c, "title": t, "answers": w} for c, t, w in CAPS],
            "note": "Counted from the files stored on this computer. A provider's advertised coverage is not counted until its data is here."}


# ----------------------------------------------------------------------------- one strategy, before anything is planned
def precheck(config: dict, store: MarketStore | None = None, max_rows: int = 300) -> dict:
    """Can this strategy be evaluated on stored observations? Signals are found from the underlying alone; each signal
    session is then classified by what is stored for it. Nothing is downloaded and nothing is modelled."""
    from strata.data.providers import market as MK
    store = store or MarketStore()
    need = MK.needs_of(config, store)
    sym = need["symbol"]
    st = stored_option_sessions(store, sym)
    absent = set(need.get("no_contract_days") or [])
    eod_days = set(st["eod_expirations"])
    intraday_needed = bool(need.get("intraday"))
    rows, counts = [], {"quotes": 0, "trades": 0, "partial": 0, "eod": 0, "no_contract": 0, "missing": 0}
    for iso in need["days"]:
        d = date.fromisoformat(iso)
        items = st["sessions"].get(d, [])
        if any(i["observation"] == "bid/ask quotes" and i["status"] == "complete" for i in items):
            how = "quotes"
        elif any(i["observation"] == "trade prints" for i in items):
            how = "trades"
        elif iso in absent:
            how = "no_contract"
        elif any(i["status"] != "complete" for i in items):
            how = "partial"
        elif not intraday_needed and iso in eod_days:
            how = "eod"
        else:
            how = "missing"
        counts[how] += 1
        rows.append({"date": iso, "status": how, "stored": [f"{i['provider']}: {i['observation']} ({i['resolution']}" + (", part of the session" if i["status"] != "complete" else "") + ")" for i in items]})
    n = len(rows)
    priced = counts["quotes"] + counts["trades"] + counts["eod"]
    budget = MK.budget_table(need) if n else None
    free = (budget["budgets"][0]["best"] or {}) if budget else {}
    cheapest = next((b["best"] for b in (budget["budgets"] if budget else []) if b["best"] and b["best"]["cost"] > 0 and b["best"]["sessions"] > max(priced, free.get("sessions", 0))), None)
    alts = []
    if n:
        alts.append({"id": "study", "label": "Run the underlying market study", "detail": f"What {sym} itself did on those {n:,} sessions needs no option prices: how often the move happened, how far, by when.", "available": True})
    if priced:
        ds_ = [r["date"] for r in rows if r["status"] in ("quotes", "trades", "eod")]
        alts.append({"id": "subset", "label": f"Run it on the {priced:,} session{'s' if priced != 1 else ''} that can be priced", "detail": f"{ds_[0]} to {ds_[-1]}. The other {n - priced:,} are reported as not tested, never filled in.", "available": True, "sessions": ds_[-60:]})
    if free.get("sessions", 0) > priced:
        alts.append({"id": "connect", "label": f"Connect a free account: up to {free['sessions']:,} of the {budget.get('reachable_sessions', budget['testable_sessions']):,} sessions any source reaches", "detail": "; ".join(free.get("needs") or []) + ". None of these has been run with a real key on this installation.", "available": False,
                     "sessions_with_bid_ask": free.get("sessions_with_bid_ask"), "sessions_trade_prints_only": free.get("sessions_trade_prints_only")})
    if cheapest and cheapest.get("cost", 0) > 0:
        alts.append({"id": "buy", "label": f"Buy and import: {cheapest['label']}", "detail": f"About ${cheapest['cost']:g} for up to {cheapest['sessions']:,} sessions ({cheapest['sessions_with_bid_ask']:,} with bid/ask). Strata never buys anything; the purchase is yours.", "available": False, "cost": cheapest["cost"]})
    alts.append({"id": "import", "label": "Import licensed files you already have", "detail": "Cboe DataShop, OptionsDX, historicaloptiondata.com and generic CSV or Parquet layouts are recognised (Data, Import).", "available": True})
    alts.append({"id": "modeled", "label": "Use theoretical premiums (opt in)", "detail": "Black-Scholes prices from the underlying. Not observed option prices: the result is a model's, and is labelled so everywhere it appears.", "available": True, "warning": True})
    if not n:
        text = f"The entry rule never fires on the stored {sym} history, so there is nothing to price."
    else:
        bits = []
        if counts["quotes"]:
            bits.append(f"{counts['quotes']:,} with intraday bid/ask")
        if counts["trades"]:
            bits.append(f"{counts['trades']:,} with intraday trade prints")
        if counts["eod"]:
            bits.append(f"{counts['eod']:,} with end-of-day quotes")
        text = (f"The entry rule fires on {n:,} session{'s' if n != 1 else ''} ({need['first']} to {need['last']}). " + (f"{priced:,} can be priced on stored option observations ({', '.join(bits)})" if priced else "None can be priced on stored option observations")
                + f": {priced / n * 100:.1f}% of them."
                + (f" {counts['partial']} more {'are' if counts['partial'] != 1 else 'is'} observed for part of the session only." if counts["partial"] else "")
                + (f" On {counts['no_contract']:,} no contract with the required expiration was listed, so they cannot be tested with any data." if counts["no_contract"] else "")
                + (f" {counts['missing']:,} have no stored option prices." if counts["missing"] else ""))
    return {"symbol": sym, "signal_sessions": n, "priced": priced, "priced_pct": round(priced / n * 100, 1) if n else None, "counts": counts, "first": need.get("first"), "last": need.get("last"), "intraday": intraday_needed,
            "contract_existence": need.get("contract_existence"), "text": text, "alternatives": alts, "sessions": rows[-max_rows:], "sessions_listed": min(n, max_rows),
            "untested": [r["date"] for r in rows if r["status"] in ("missing", "partial", "no_contract")][-max_rows:], "budget": budget, "executable": priced > 0,
            "blocking": priced == 0 and n > 0}
