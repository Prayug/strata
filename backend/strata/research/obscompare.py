"""Trade prints against bid/ask quotes: do the two datasets describe the same market, and what does the difference cost.

Two questions, answered only from observations that are stored:

  alignment   Are the same contracts in both datasets, and do their clocks agree? For every contract-minute both hold, the
              trade bar's prices are compared with the bid and the ask sampled around that minute. Prints of a liquid
              contract sit inside the quoted market almost always; if they sit inside it a minute EARLIER or LATER far more
              often, one of the two clocks is labelled differently, and that is reported before anything is priced.

  execution   The same strategy, three ways on the sessions both datasets cover: filled at trade prints, filled at the
              quote on the side the order pays (buy at the ask, sell at the bid) and filled at the midpoint. The difference
              between the first and the second is what a print-based backtest cannot see.

Nothing here estimates a spread. Where quotes are not stored the comparison does not run and says what is missing.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl

from strata.data.store import DatasetKey, MarketStore

TICK = 0.01  # option prices move in cents: a print one cent outside a sampled quote is inside the market between samples


def _stats(a: np.ndarray) -> dict | None:
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4), "p05": round(float(np.percentile(a, 5)), 4), "p95": round(float(np.percentile(a, 95)), 4)}


def common_sessions(store: MarketStore, sym: str, quote_ds: DatasetKey, trade_ds: DatasetKey) -> list[date]:
    q = set(store.option_days(sym, "quote", dataset=quote_ds, complete_only=True))
    idx = store.option_bar_index(sym, dataset=trade_ds)
    t = set(idx.filter(pl.col("status") == "ok")["date"].unique().to_list()) if not idx.is_empty() else set()
    part = {date.fromisoformat(k) for k, v in store.session_states(trade_ds).items() if v.get("status") == "partial"}
    return sorted((q & t) - part)


def alignment(store: MarketStore, sym: str, quote_provider: str, trade_provider: str, sessions: list[date] | None = None, max_sessions: int = 40) -> dict:
    """Identifier and timestamp alignment of a quote dataset and a trade-bar dataset of the same symbol."""
    qd, td = store.option_dataset(sym, "quote", quote_provider), store.option_dataset(sym, "trade", trade_provider)
    rep: dict = {"symbol": sym, "quotes": qd.id if qd else None, "trades": td.id if td else None, "status": "blocked", "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if qd is None or td is None:
        rep["reason"] = ("BLOCKED: no bid/ask quote dataset is stored for " + sym + (f" from {quote_provider}" if quote_provider else "") + ". The comparison needs observed quotes; none are estimated." if qd is None
                         else f"no option trade bars are stored for {sym} from {trade_provider}")
        return rep
    if qd.res == "EOD":
        rep.update(reason=f"BLOCKED: the stored quotes for {sym} ({qd.id}) are one end-of-day snapshot a session. Minute-by-minute trade bars cannot be aligned with them.")
        return rep
    days = [d for d in (sessions or common_sessions(store, sym, qd, td))][:max_sessions]
    if not days:
        rep.update(status="no_overlap", reason="the two datasets have no complete session in common")
        return rep
    tot = {"contract_minutes": 0, "inside": 0, "close_inside": 0, "inside_if_quotes_a_minute_earlier": 0, "inside_if_quotes_a_minute_later": 0, "contracts_both": 0, "contracts_only_quotes": 0, "contracts_only_trades": 0, "crossed_or_empty_quotes": 0}
    pos, spread, per = [], [], []
    seen: dict[str, int] = {}
    for d in days:
        q = store.read_option_day(sym, d, "quote", dataset=qd)
        b = store.read_option_bars(sym, d, dataset=td)
        if q is None or b is None or q.is_empty() or b.is_empty():
            continue
        key = ["expiration", "strike", "right"]
        q = q.select("ts", *key, "bid", "ask").with_columns(pl.col("ts").dt.truncate("1m"))
        bad = q.filter(pl.col("bid").is_null() | pl.col("ask").is_null() | (pl.col("ask") < pl.col("bid")) | (pl.col("ask") <= 0)).height
        q = q.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") >= pl.col("bid")) & (pl.col("ask") > 0)).unique(subset=["ts", *key], keep="last")
        kq, kb = q.select(key).unique(), b.select(key).unique()
        both = kq.join(kb, on=key, how="inner").height
        tot["contracts_both"] += both
        tot["contracts_only_quotes"] += kq.height - both
        tot["contracts_only_trades"] += kb.height - both
        tot["crossed_or_empty_quotes"] += bad
        bars = b.select(pl.col("ts").alias("t0"), *key, "low", "high", "close")
        one = {"date": d.isoformat(), "contracts_in_both": both}
        for name, shift in (("close_inside", 0), ("inside_if_quotes_a_minute_earlier", -60), ("inside_if_quotes_a_minute_later", 60)):
            # the clock test: a bar that starts at t0 closes on its last print before t0 + 1 minute, so the quote sampled AT
            # t0 + 1 minute is the nearest one to it. The same test with the quote records moved a minute either way says
            # whether they fit better under another label.
            qb = q.with_columns((pl.col("ts") + pl.duration(seconds=shift) - pl.duration(seconds=60)).alias("t0")).select("t0", *key, pl.col("bid").alias("b1"), pl.col("ask").alias("a1"))
            j = bars.join(qb, on=["t0", *key], how="inner")
            if j.is_empty():
                one[name] = None
                continue
            c, b1, a1 = j["close"].to_numpy(), j["b1"].to_numpy(), j["a1"].to_numpy()
            ok = (c >= b1 - TICK - 1e-9) & (c <= a1 + TICK + 1e-9)
            tot[name] += int(ok.sum())
            seen[name] = seen.get(name, 0) + j.height  # each label is judged on the minutes it can be compared on
            one[name] = round(float(ok.mean()), 4)
            if shift == 0:
                tot["contract_minutes"] += j.height
                one["contract_minutes"] = j.height
                half = (a1 - b1) / 2
                m = half > 0
                pos.append((c[m] - (a1[m] + b1[m]) / 2) / half[m])
                spread.append((a1 - b1)[m] / ((a1 + b1) / 2)[m] * 100)
                # every print of the minute against the market at both ends of it
                qa = q.with_columns(pl.col("ts").alias("t0")).select("t0", *key, pl.col("bid").alias("b0"), pl.col("ask").alias("a0"))
                j2 = j.join(qa, on=["t0", *key], how="inner")
                if not j2.is_empty():
                    lo, hi = np.minimum(j2["b0"].to_numpy(), j2["b1"].to_numpy()) - TICK - 1e-9, np.maximum(j2["a0"].to_numpy(), j2["a1"].to_numpy()) + TICK + 1e-9
                    tot["inside"] += int(((j2["low"].to_numpy() >= lo) & (j2["high"].to_numpy() <= hi)).sum())
                    tot["minutes_with_both_ends"] = tot.get("minutes_with_both_ends", 0) + j2.height
        per.append(one)
    n = max(1, tot["contract_minutes"])
    rep.update(status="ran" if tot["contract_minutes"] else "no_overlap", sessions=len(per), per_session=per[:60], **tot,
               inside_pct=round(tot["inside"] / max(1, tot.get("minutes_with_both_ends", 0)) * 100, 2), close_inside_pct=round(tot["close_inside"] / n * 100, 2),
               earlier_pct=round(tot["inside_if_quotes_a_minute_earlier"] / max(1, seen.get("inside_if_quotes_a_minute_earlier", 0)) * 100, 2),
               later_pct=round(tot["inside_if_quotes_a_minute_later"] / max(1, seen.get("inside_if_quotes_a_minute_later", 0)) * 100, 2),
               print_position_in_half_spreads=_stats(np.concatenate(pos) if pos else np.zeros(0)), quoted_spread_pct_of_mid=_stats(np.concatenate(spread) if spread else np.zeros(0)))
    if rep["status"] != "ran":
        rep["reason"] = "no contract-minute is present in both datasets on the sessions they share"
        return rep
    best = max((("as labelled", rep["close_inside_pct"]), ("a minute earlier", rep["earlier_pct"]), ("a minute later", rep["later_pct"])), key=lambda kv: kv[1])
    rep["clock"] = {"best": best[0], "aligned": best[0] == "as labelled" or rep["close_inside_pct"] >= best[1] - 1.0}
    rep["identifiers_aligned"] = tot["contracts_both"] > 0 and tot["contracts_only_trades"] <= 0.02 * max(1, tot["contracts_both"] + tot["contracts_only_trades"])
    rep["text"] = (f"{tot['contract_minutes']:,} contract-minutes on {len(per)} sessions are in both datasets. In {rep['close_inside_pct']:.1f}% of them the minute's last print lies between the bid and the ask sampled at the end of that minute "
                   f"({rep['earlier_pct']:.1f}% if the quote records were a minute earlier, {rep['later_pct']:.1f}% if a minute later), so the two clocks " + ("agree" if rep["clock"]["aligned"] else f"do NOT agree: the quotes fit best {best[0]}")
                   + f"; every print of the minute lies inside the market quoted at its two ends in {rep['inside_pct']:.1f}%"
                   + f". {tot['contracts_both']:,} contract-sessions are in both; {tot['contracts_only_trades']:,} traded without a stored quote and {tot['contracts_only_quotes']:,} are quoted without a stored trade (a contract that did not trade has no bar).")
    return rep


def execution(cfg: dict, store: MarketStore, quote_provider: str, trade_provider: str, should_cancel=None) -> dict:
    """The same strategy filled at trade prints, at the quote on the paying side and at the midpoint, on the sessions both datasets cover."""
    from strata.engine.engine import run_backtest
    from strata.engine.models import BacktestConfig
    base = BacktestConfig.model_validate(cfg)
    sym = base.strategies[0].symbol
    qd, td = store.option_dataset(sym, "quote", quote_provider), store.option_dataset(sym, "trade", trade_provider)
    rep: dict = {"symbol": sym, "quotes": qd.id if qd else None, "trades": td.id if td else None, "status": "blocked", "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if qd is None or td is None or qd.res == "EOD":
        rep["reason"] = ("BLOCKED: no intraday bid/ask quote dataset is stored for " + sym + (f" from {quote_provider}" if quote_provider else "") + ". A quote-based result needs observed quotes; none are estimated." if (qd is None or qd.res == "EOD")
                         else f"no option trade bars are stored for {sym} from {trade_provider}")
        return rep
    days = [d for d in common_sessions(store, sym, qd, td) if base.start <= d <= base.end]
    if not days:
        rep.update(status="no_overlap", reason=f"the two datasets share no complete session between {base.start} and {base.end}")
        return rep
    raw = base.model_dump(mode="json")

    def run(mode: str, provider: str, fill: str) -> dict:
        c = {**raw, "pricing": {**raw["pricing"], "mode": mode}, "data": {**raw["data"], "options_provider": provider}, "execution": {**raw["execution"], "fill_model": fill}}
        res = run_backtest(c, store=store, record_paths=False, should_cancel=should_cancel)
        keep = {d.isoformat() for d in days}
        return {t["entry_date"]: t for t in res.trades if t["entry_date"] in keep}
    tr, qa, qm = run("trades", trade_provider, "conservative"), run("historical", quote_provider, "conservative"), run("historical", quote_provider, "midpoint")
    rows, pos_e, pos_x = [], [], []
    for d in sorted(set(tr) | set(qa)):
        a, b, m = tr.get(d), qa.get(d), qm.get(d)
        row = {"date": d, "trade_prints": _brief(a), "quotes": _brief(b), "midpoint": _brief(m), "same_contract": None}
        if a and b:
            la, lb = a["legs"], b["legs"]
            row["same_contract"] = len(la) == len(lb) and all((x["strike"], x["expiration"], x["right"]) == (y["strike"], y["expiration"], y["right"]) for x, y in zip(la, lb))
            if row["same_contract"]:
                for x, y in zip(la, lb):
                    if y.get("entry_bid") is not None and y.get("entry_ask") is not None and y["entry_ask"] > y["entry_bid"] and x.get("entry_print") is not None:
                        pos_e.append((x["entry_print"] - (y["entry_ask"] + y["entry_bid"]) / 2) / ((y["entry_ask"] - y["entry_bid"]) / 2))
                    if y.get("exit_bid") is not None and y.get("exit_ask") is not None and y["exit_ask"] > y["exit_bid"] and x.get("exit_print") is not None:
                        pos_x.append((x["exit_print"] - (y["exit_ask"] + y["exit_bid"]) / 2) / ((y["exit_ask"] - y["exit_bid"]) / 2))
        rows.append(row)
    both = [r for r in rows if r["trade_prints"] and r["quotes"]]
    same = [r for r in both if r["same_contract"]]
    tot = lambda rs, k: round(sum(float(r[k]["pnl"]) for r in rs if r[k]), 2)
    rep.update(status="ran", sessions_in_both=len(days), first=days[0].isoformat(), last=days[-1].isoformat(), rows=rows[:400],
               trades={"trade_prints": len(tr), "quotes": len(qa), "in_both": len(both), "same_contract": len(same), "only_trade_prints": len([r for r in rows if r["trade_prints"] and not r["quotes"]]),
                       "only_quotes": len([r for r in rows if r["quotes"] and not r["trade_prints"]])},
               pnl={"trade_prints": tot(same, "trade_prints"), "quotes_paying_the_spread": tot(same, "quotes"), "quotes_at_the_midpoint": tot([r for r in same if r["midpoint"]], "midpoint")},
               winners={"trade_prints": sum(1 for r in same if r["trade_prints"]["pnl"] > 0), "quotes_paying_the_spread": sum(1 for r in same if r["quotes"]["pnl"] > 0)},
               entry_print_position_in_half_spreads=_stats(np.asarray(pos_e, float)), exit_print_position_in_half_spreads=_stats(np.asarray(pos_x, float)))
    p = rep["pnl"]
    rep["spread_cost"] = round(p["quotes_at_the_midpoint"] - p["quotes_paying_the_spread"], 2)
    rep["text"] = (f"On the {len(days)} sessions both datasets cover, {len(same)} positions were taken in the same contract by all three fills. They made {p['trade_prints']:+,.2f} filled at trade prints, "
                   f"{p['quotes_paying_the_spread']:+,.2f} buying at the ask and selling at the bid, and {p['quotes_at_the_midpoint']:+,.2f} at the midpoint: crossing the quoted spread cost {rep['spread_cost']:,.2f}. "
                   + (f"{len(both) - len(same)} position(s) chose a different contract under the two datasets and are left out of the totals. " if len(both) != len(same) else "")
                   + (f"{rep['trades']['only_trade_prints']} traded on prints with no quote-based fill and {rep['trades']['only_quotes']} the other way round. " if rep["trades"]["only_trade_prints"] or rep["trades"]["only_quotes"] else ""))
    return rep


def _brief(t: dict | None) -> dict | None:
    if t is None:
        return None
    l = t["legs"][0]
    return {"pnl": round(float(t["pnl"]), 2), "entry": l.get("entry_price"), "exit": l.get("exit_price"), "strike": l.get("strike"), "expiration": l.get("expiration"), "exit_reason": t.get("exit_reason"),
            "entry_bid": l.get("entry_bid"), "entry_ask": l.get("entry_ask"), "exit_bid": l.get("exit_bid"), "exit_ask": l.get("exit_ask"), "entry_time": t.get("entry_time"), "exit_time": t.get("exit_time")}
