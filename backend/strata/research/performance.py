"""What a strategy did, in the order a reader should take it in.

The card puts first what decides whether anything after it means anything: how many trades there actually were, and how many
of the signals the stored observations could price at all. Then the result after costs, the average trade and how uncertain
it is, what the path was like (drawdown, losing streaks), where the result came from (the distribution, the calendar, the
time of day, the few largest trades), how much of it survives worse fills, and what the same position did WITHOUT the
signal. Every figure is read from the run's stored trades; nothing is re-simulated to produce the card.

Two things are kept apart throughout. A historical result is what these trades made on this data. Evidence of an edge is a
statement about trades that have not happened, and a card says so only as far as the sample and the number of
configurations tried allow.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import MarketStore
from strata.research import stats as ST

WORSE_FILLS = (1.0, 2.0, 5.0, 10.0)
SETTLED = ("expired_worthless", "exercised", "assigned", "expiration", "expired")  # the position was not closed by a trade


def _turnover(t: dict) -> float:
    return sum(abs(float(l.get("contracts") or 0)) * float(l.get("multiplier") or 100.0) * (float(l.get("entry_price") or 0.0) + float(l.get("exit_price") or 0.0)) for l in t["legs"] if l.get("kind") == "option")


def _group(trades: list[dict], key) -> list[dict]:
    by: dict[str, list[float]] = {}
    for t in trades:
        by.setdefault(key(t), []).append(float(t["pnl"]))
    return [{"key": k, "trades": len(v), "net": round(sum(v), 2), "average": round(sum(v) / len(v), 2), "win_rate": round(sum(1 for x in v if x > 0) / len(v) * 100, 1)} for k, v in sorted(by.items())]


def _streak(p: np.ndarray, losing: bool) -> int:
    best = cur = 0
    for x in p:
        cur = cur + 1 if ((x <= 0) if losing else (x > 0)) else 0
        best = max(best, cur)
    return best


def card(run: dict, res: dict, trades: list[dict], trials: int = 1, baseline: dict | None = None, hires: dict | None = None) -> dict:
    """The performance card of one stored backtest: an ordered list of sections, each with its figures and one plain sentence."""
    meta = res.get("meta") or {}
    cfg = res.get("config") or {}
    ex = meta.get("executability") or {}
    trades = sorted(trades, key=lambda t: (t.get("exit_ts") or t["entry_ts"], t["id"]))
    p = np.array([float(t["pnl"]) for t in trades], float)
    n = int(p.size)
    start, end = cfg.get("start"), cfg.get("end")
    days = cal.trading_days(date.fromisoformat(start), date.fromisoformat(end)) if start and end else []
    nod = ex.get("sessions_without_option_data")
    no_data = sum((v if isinstance(v, int) else len(v)) for v in nod.values()) if isinstance(nod, dict) else int(nod or 0)
    sections: list[dict] = []
    observation = "trade prints (no bid/ask)" if any("trade_based_fill" in (t.get("flags") or []) for t in trades) or ((cfg.get("pricing") or {}).get("mode") == "trades") else "bid/ask quotes"

    # 1. how many trades there actually were
    sections.append({"id": "trades", "title": "Trades actually taken", "figures": {"trades": n, "winners": int((p > 0).sum()), "losers": int((p < 0).sum()), "flat": int((p == 0).sum()), "sessions_with_a_trade": len({t["entry_date"] for t in trades})},
                     "text": (f"{n} trades on {len({t['entry_date'] for t in trades})} sessions." if n else "No trade was taken.") + ("" if n >= 30 else " Fewer than 30 trades: every average below is fragile, and none of it is evidence of an edge.")})
    # 2. what was asked for against what the stored data could price
    sig, done = ex.get("signals"), ex.get("executed")
    reasons = dict(ex.get("not_executed_by_reason") or {})
    cov = res.get("coverage") or {}
    asked, fired, could = cov.get("requested") or {}, (cov.get("signals") or {}).get("qualifying"), (cov.get("executable") or {})
    wide = fired is not None and could.get("signals") is not None and fired > (sig or 0)
    if wide:
        # the request covered more history than the stored option observations do: the run was made on the part that could
        # be priced, and the card says how small that part is before it says anything else
        for it in (cov.get("options") or {}).get("issues") or []:
            reasons[str(it.get("reason"))] = int(it.get("signals") or 0)
        total, priced = int(fired), int(done if done is not None else could["signals"])
        txt = (f"Over the period asked for ({asked.get('start')} to {asked.get('end')}{'' if asked.get('stated') else ', all stored history: no period was stated'}) the entry rule fired on {total:,} sessions. "
               f"{priced:,} of them ({priced / total * 100:.1f}%) could be priced on stored option observations, all between {(could.get('window') or [start, end])[0]} and {(could.get('window') or [start, end])[1]}. "
               f"Everything below describes those {priced:,} and no others. Not priced, and not losses: " + "; ".join(f"{v:,} {k}" for k, v in reasons.items()) + ".")
        fig = {"requested": [asked.get("start"), asked.get("end")], "period_stated": bool(asked.get("stated")), "window": [start, end], "sessions_in_window": len(days), "signals": total, "priced": priced,
               "priced_pct": round(priced / total * 100, 1) if total else None, "not_priced_by_reason": reasons, "observation": observation}
    else:
        # a rule that fires again in a session after an entry could not be filled is counted each time by the engine, and once
        # (a session) by the check made before the run: both numbers are said, so the two accounts of one run read as one
        on = f" on {int(fired):,} sessions" if fired is not None and sig is not None and int(fired) != int(sig) else ""
        txt = ("The entry rule fired " + (f"{sig} times{on}; {done} could be priced on stored observations" if sig is not None else "an unrecorded number of times")
               + (f" ({done / sig * 100:.0f}%)" if sig else "") + ". "
               + ("The others are not losses and not skipped by choice: " + ", ".join(f"{v} {str(k).replace('_', ' ')}" for k, v in reasons.items()) + ". " if reasons else "")
               + (f"{no_data} of the {len(days)} sessions in the window have no stored option data at all." if no_data else ""))
        fig = {"window": [start, end], "sessions_in_window": len(days), "sessions_without_option_data": no_data, "signals": sig, "sessions_with_a_signal": fired, "priced": done, "priced_pct": round(done / sig * 100, 1) if sig else None,
               "not_priced_by_reason": reasons, "observation": observation}
    sections.append({"id": "coverage", "title": "Requested against available", "figures": fig, "text": txt})
    if n == 0:
        return {"run": run.get("id"), "name": run.get("name"), "observation": observation, "sections": sections, "evidence": {"level": "insufficient", "text": "There are no trades to judge."}}
    comm, slip = sum(float(t.get("commissions") or 0) for t in trades), sum(float(t.get("slippage") or 0) for t in trades)
    gross = float(sum(float(t.get("gross_pnl") if t.get("gross_pnl") is not None else t["pnl"]) for t in trades))
    cap = float(((cfg.get("capital") or {}).get("initial_capital")) or 0) or None
    paid = sum(abs(float(t.get("basis") or 0)) for t in trades)
    # 3. the result after costs
    # The account keeps its own books (cash and marked positions), and its net result has to be what the trades add up to. It
    # once was not: a contract dated on a day the market was closed never expired in the engine and sat in the account for two
    # years, marked. That is fixed where it happened; this line is here so that anything like it is said, not averaged in.
    acct = (res.get("metrics") or {}).get("net_pnl")
    gap = None if acct is None or not n else round(float(acct) - float(p.sum()), 2)
    off = gap is not None and abs(gap) > 0.5
    sections.append({"id": "net", "title": "Net result after costs", "figures": {"net": round(float(p.sum()), 2), "gross": round(gross, 2), "commissions": round(comm, 2), "slippage": round(slip, 2),
                                                                               "pct_of_starting_capital": round(float(p.sum()) / cap * 100, 2) if cap else None, "pct_of_premium_traded": round(float(p.sum()) / paid * 100, 2) if paid else None,
                                                                               "account_net": None if acct is None else round(float(acct), 2), "account_minus_trades": gap},
                     "warning": off,
                     "text": f"{p.sum():+,.2f} after {comm:,.2f} of commissions and {slip:,.2f} of assumed slippage ({gross:+,.2f} before them)."
                             + (f" That is {p.sum() / paid * 100:+.1f}% of the {paid:,.0f} of premium the trades put down." if paid else "")
                             + (f" The account's own books say {float(acct):+,.2f}: {gap:+,.2f} that these trades do not account for. Something was held that never became a trade; do not rely on either figure until that is explained." if off else "")})
    # 4. the average trade
    wins, losses = p[p > 0], p[p < 0]
    sections.append({"id": "average", "title": "Average trade", "figures": {"mean": round(float(p.mean()), 2), "median": round(float(np.median(p)), 2), "win_rate": round(float((p > 0).mean() * 100), 1),
                                                                         "average_win": round(float(wins.mean()), 2) if wins.size else None, "average_loss": round(float(losses.mean()), 2) if losses.size else None,
                                                                         "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.size and wins.size else None},
                     "text": f"{p.mean():+,.2f} a trade (median {np.median(p):+,.2f}); {(p > 0).mean() * 100:.0f}% of trades made money"
                             + (f", averaging {wins.mean():+,.2f} against {losses.mean():+,.2f} on the losers." if wins.size and losses.size else ".")})
    # 5. expected value, and how well it is known
    mt = ST.mean_test(p.tolist())
    adj = ST.bonferroni(mt["p_value"], trials)
    known = mt["ci95"][0] is not None
    sections.append({"id": "expectancy", "title": "Expected value per trade", "figures": {"mean": round(mt["mean"], 2), "ci95": [round(x, 2) for x in mt["ci95"]] if known else None, "p_value": mt["p_value"],
                                                                                        "configurations_tried": trials, "p_adjusted": adj},
                     "text": (f"The average is {mt['mean']:+,.2f}; resampling these trades puts it between {mt['ci95'][0]:+,.2f} and {mt['ci95'][1]:+,.2f} 95 times in 100. " if known else f"The average is {mt['mean']:+,.2f}; there are too few trades for an interval. ")
                             + ("" if mt["p_value"] is None else (f"Against zero that is p = {mt['p_value']:.3f}" + (f", and p = {adj:.3f} once the {trials} configurations tried in this research session are allowed for" if trials > 1 and adj is not None else "") + ". "))
                             + ("The interval includes zero: these trades do not show an average different from none." if known and mt["ci95"][0] <= 0 <= mt["ci95"][1] else "")})
    # 6. and 7. the path: drawdown and losing streaks
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0))
    dd = peak - eq
    j = int(dd.argmax())
    i = int(np.argmax(eq[: j + 1])) if dd[j] > 0 and eq[: j + 1].max() > 0 else -1
    under = int(max((len(r) for r in "".join("U" if x > 0 else "-" for x in dd).split("-")), default=0))
    sections.append({"id": "drawdown", "title": "Maximum drawdown", "figures": {"max_drawdown": round(float(dd[j]), 2), "from": trades[i]["entry_date"] if i >= 0 else trades[0]["entry_date"], "to": trades[j]["entry_date"],
                                                                              "longest_run_below_a_previous_high_trades": under},
                     "text": (f"The largest fall from a previous high of the cumulative result was {dd[j]:,.2f}, reached on {trades[j]['entry_date']}; the longest stretch below a previous high lasted {under} trades."
                              if dd[j] > 0 else "The cumulative result never fell below a previous high.")})
    ml, mw = _streak(p, True), _streak(p, False)
    sections.append({"id": "streaks", "title": "Consecutive losses", "figures": {"max_consecutive_losses": ml, "max_consecutive_wins": mw},
                     "text": f"The longest run of trades that did not make money was {ml}; the longest winning run was {mw}."})
    # 8. the distribution of results
    edges = np.histogram_bin_edges(p, bins=min(24, max(6, int(math.sqrt(n)) * 2)))
    counts, _ = np.histogram(p, bins=edges)
    q = {k: round(float(np.percentile(p, k)), 2) for k in (5, 25, 50, 75, 95)}
    sections.append({"id": "distribution", "title": "Distribution of trade results", "figures": {"percentiles": q, "best": round(float(p.max()), 2), "worst": round(float(p.min()), 2),
                                                                                              "histogram": {"edges": [round(float(x), 2) for x in edges], "counts": [int(x) for x in counts]}},
                     "text": f"Half of the trades fall between {q[25]:+,.2f} and {q[75]:+,.2f}; one in twenty is worse than {q[5]:+,.2f} and one in twenty better than {q[95]:+,.2f}. The best was {p.max():+,.2f}, the worst {p.min():+,.2f}."})
    # 9. the calendar
    years, months = _group(trades, lambda t: t["entry_date"][:4]), _group(trades, lambda t: t["entry_date"][:7])
    pos_m = sum(1 for m in months if m["net"] > 0)
    sections.append({"id": "calendar", "title": "By year and month", "figures": {"years": years, "months": months},
                     "text": f"{pos_m} of the {len(months)} months with a trade ended above zero; " + ", ".join(f"{y['key']}: {y['net']:+,.0f} over {y['trades']}" for y in years) + "."})
    # 10. the time of day
    tod = _group(trades, lambda t: f"{int(t['entry_minute']) // 30 * 30 // 60:02d}:{int(t['entry_minute']) // 30 * 30 % 60:02d}")
    top_slot = max(tod, key=lambda r: r["trades"])
    sections.append({"id": "time_of_day", "title": "By time of entry (New York)", "figures": {"half_hours": tod},
                     "text": (f"All entries fall in the half hour from {tod[0]['key']}." if len(tod) == 1 else f"Entries are spread over {len(tod)} half hours; most ({top_slot['trades']}) fall in the one from {top_slot['key']}, which made {top_slot['net']:+,.2f}.")})
    # 11. how much of it survives worse fills
    turn = np.array([_turnover(t) for t in trades], float)
    rows = [{"worse_fills_pct": x, "net": round(float((p - turn * x / 100).sum()), 2), "win_rate": round(float(((p - turn * x / 100) > 0).mean() * 100), 1)} for x in WORSE_FILLS]
    be = float(p.sum() / turn.sum() * 100) if p.sum() > 0 and turn.sum() > 0 else None
    sens = {"rows": rows, "breakeven_pct": round(be, 2) if be is not None else None, "method": "every option fill repriced x% against the position with the same trades (first order: exits are not re-decided)"}
    txt = f"With every fill 2% worse the result is {rows[1]['net']:+,.2f}, with 5% worse {rows[2]['net']:+,.2f}" + (f"; it reaches zero at about {be:.1f}% worse fills." if be is not None else ".")
    if hires and (hires.get("summary") or {}).get("repriced"):
        hs = hires["summary"]
        sens["individual_trades"] = {**{k: hs.get(k) for k in ("positions", "repriced", "bar_pnl_repriced", "tick_pnl_repriced", "difference", "entries_not_filled")},
                                     "median_entry_seconds_after_decision": (hs.get("entry_seconds_after_decision") or {}).get("median"), "entry_price_worse_pct_mean": (hs.get("entry_price_worse_pct") or {}).get("mean")}
        txt += f" Filled on the first individual print after each decision instead, the {hs['repriced']} trades that could be repriced made {hs['tick_pnl_repriced']:+,.2f} against {hs['bar_pnl_repriced']:+,.2f} on minute bars."
    if observation.startswith("trade prints"):
        txt += " These are trade prints: the bid/ask spread is not in any of these figures."
    sections.append({"id": "execution", "title": "Sensitivity to execution", "figures": sens, "text": txt})
    # 12. how much of it is a few trades
    order = np.sort(p)[::-1]
    total = float(p.sum())
    top5pct = max(1, int(math.ceil(n * 0.05)))
    gp = float(wins.sum()) if wins.size else 0.0
    half = int(np.searchsorted(np.cumsum(order[order > 0]), gp / 2) + 1) if gp > 0 else None
    conc = {"best_trade": round(float(order[0]), 2), "net_without_best": round(total - float(order[0]), 2), "net_without_best_3": round(total - float(order[:3].sum()), 2), "net_without_best_5pct": round(total - float(order[:top5pct].sum()), 2),
            "best_5pct_trades": top5pct, "trades_making_half_of_all_profit": half, "best_3_share_of_net_pct": round(float(order[:3].sum()) / total * 100, 1) if total > 0 else None}
    sections.append({"id": "concentration", "title": "Dependence on the largest trades", "figures": conc,
                     "text": f"Without its best trade ({order[0]:+,.2f}) the result is {conc['net_without_best']:+,.2f}; without its best three, {conc['net_without_best_3']:+,.2f}."
                             + (f" Half of all the profit made by winning trades came from {half} of them." if half else "")
                             + (" The result is positive only because of its largest trades." if total > 0 and conc["net_without_best_3"] <= 0 else "")})
    # 13. against the same position without the signal
    if baseline and baseline.get("trades"):
        b = baseline
        sections.append({"id": "baseline", "title": "Against the same position without the signal", "figures": {k: b.get(k) for k in ("definition", "trades", "net", "average", "win_rate", "difference_in_average", "p_value", "window")},
                         "text": (f"The same position entered {b['definition']} made {b['average']:+,.2f} a trade over {b['trades']} trades. With the signal the average is {p.mean():+,.2f}: a difference of {b['difference_in_average']:+,.2f}"
                                  + (f" (p = {b['p_value']:.3f})." if b.get("p_value") is not None else ".") + (" The signal's sessions are not distinguishable from all sessions." if b.get("p_value") is not None and b["p_value"] >= 0.05 else ""))})
    else:
        sections.append({"id": "baseline", "title": "Against the same position without the signal", "figures": None,
                         "text": "Not run yet. The baseline enters the same position at the same time on every session, whatever the signal says: it shows how much of the result is the signal and how much is the position."})
    rob = ST.backtest_robustness({"cost_sensitivity": {"rows": [{"extra_slippage_pct": r["worse_fills_pct"], "net_pnl": r["net"]} for r in rows], "breakeven_slippage_pct": be}}, trades, trials)
    return {"run": run.get("id"), "name": run.get("name"), "observation": observation, "window": [start, end], "sections": sections,
            "evidence": {"level": rob["level"], "text": rob["verdict"], "checks": rob["checks"], "configurations_tried": trials,
                         "reading": "The sections above are what these trades did on this data. Whether that is evidence of an edge is this line, and it depends on the number of trades and on how many configurations were tried before this one."}}


# ----------------------------------------------------------------------------- every fill, with the stored observation behind it
def fills(rid: str, store: MarketStore | None = None) -> dict:
    """One row per option fill of a run: when it was decided, when and at what price it was filled, and the stored bar the
    price was taken from, read back from the store now. A fill whose stored bar no longer says what the fill used is marked."""
    from strata.api import db
    from strata.api.results import load_trades
    store = store or MarketStore()
    run = db.get_run(rid) or {}
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    provider = (cfg.get("data") or {}).get("options_provider")
    trades = load_trades(rid)
    cache: dict[tuple[str, date], pl.DataFrame | None] = {}
    datasets: dict[str, object] = {}

    def bars(sym: str, d: date):
        if (sym, d) not in cache:
            if sym not in datasets:
                datasets[sym] = store.option_dataset(sym, "trade", provider)
            b = store.read_option_bars(sym, d, dataset=datasets[sym]) if datasets[sym] is not None else None
            cache[(sym, d)] = None if b is None or b.is_empty() else b.with_columns(pl.col("ts").dt.epoch("s").alias("t")).sort(["occ", "t"])
        return cache[(sym, d)]
    rows, checked, matched, settled = [], 0, 0, 0
    for t in trades:
        sym = t["symbol"]
        for l in t["legs"]:
            if l.get("kind") != "option" or not l.get("expiration"):
                continue
            occ = S.occ_symbol(sym, date.fromisoformat(l["expiration"]), l["right"], float(l["strike"]))
            for which in ("entry", "exit"):
                ts = t.get(f"{which}_ts") if which == "entry" else (l.get("exit_ts") or t.get("exit_ts"))
                if ts is None:
                    continue
                price, prt = l.get(f"{which}_price"), l.get(f"{which}_print")
                side = l["side"] if which == "entry" else ("sell" if l["side"] == "buy" else "buy")
                decided = t.get("signal_ts") if which == "entry" else (t.get("exit_signal_ts") or ts)
                why = "entry" if which == "entry" else (l.get("exit_reason") or t.get("exit_reason"))
                if which == "exit" and why in SETTLED:
                    # held to expiration: the contract was not sold, it was settled at its intrinsic value. That is arithmetic on the
                    # underlying's close, not a fill, and there is no print to compare it with
                    rows.append({"trade": t["id"], "leg": l["id"], "action": "settlement", "side": side, "contract": occ, "contracts": l.get("contracts"), "decided_et": cal.to_et(int(ts) * 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S"),
                                 "filled_et": cal.to_et(int(ts) * 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S"), "fill_price": price, "print_used": None, "price_source": "intrinsic value at expiration",
                                 "reason": why, "note": f"settled at expiration ({str(why).replace('_', ' ')}) at its intrinsic value with the underlying at {t.get('exit_spot')}: not a trade"})
                    settled += 1
                    continue
                row = {"trade": t["id"], "leg": l["id"], "action": which, "side": side, "contract": occ, "contracts": l.get("contracts"), "decided_et": cal.to_et(int(decided or ts) * 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S"),
                       "filled_et": cal.to_et(int(ts) * 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S"), "fill_price": price, "print_used": prt, "price_source": l.get("price_source"),
                       "adjustment_against_the_order": round((float(price) - float(prt)) * (1 if side == "buy" else -1), 4) if price is not None and prt is not None else None,
                       "reason": "entry" if which == "entry" else (l.get("exit_reason") or t.get("exit_reason"))}
                d = cal.to_et(int(ts) * 1_000_000_000).date()
                b = bars(sym, d) if l.get("price_source") == "trade_print" else None
                if b is not None:
                    close_t = int(cal.session_bounds(d)[1].timestamp())
                    one = b.filter((pl.col("occ") == occ) & (pl.min_horizontal(pl.col("t") + 60, pl.lit(close_t)) <= int(ts)))
                    if one.height:
                        last = one.tail(1)
                        seen = min(int(last["t"][0]) + 60, close_t)
                        row.update(stored_bar_start_et=cal.to_et(int(last["t"][0]) * 1_000_000_000).strftime("%H:%M:%S"), stored_bar_close=float(last["close"][0]), stored_bar_high=float(last["high"][0]), stored_bar_low=float(last["low"][0]),
                                   stored_bar_volume=int(last["volume"][0] or 0), print_age_seconds=int(ts) - seen)
                        if prt is not None:
                            checked += 1
                            ok = abs(float(last["close"][0]) - float(prt)) < 1e-9
                            matched += int(ok)
                            row["matches_stored_bar"] = ok
                    else:
                        row["matches_stored_bar"] = False
                        row["note"] = "no stored bar of this contract had ended by the fill time"
                        checked += 1
                rows.append(row)
    return {"run": rid, "fills": rows, "checked_against_stored_bars": checked, "matching": matched, "settled_at_expiration": settled,
            "columns": ["trade", "leg", "action", "side", "contract", "contracts", "decided_et", "filled_et", "fill_price", "print_used", "adjustment_against_the_order", "price_source", "stored_bar_start_et", "stored_bar_close",
                        "stored_bar_high", "stored_bar_low", "stored_bar_volume", "print_age_seconds", "matches_stored_bar", "reason", "note"],
            "reading": ("Each fill is the close of the last stored one-minute trade bar of its contract that had ended by the fill time, moved against the order by the run's slippage assumption. "
                        "\"matches\" compares the print the fill used with that bar as it is stored today: a mismatch means the stored data has changed since the run."
                        if checked else "This run was not filled on stored trade bars, so there is no stored bar to compare each fill with.")}


# ----------------------------------------------------------------------------- the same position without the signal
def baseline_config(cfg: dict, trades: list[dict]) -> tuple[dict, str] | None:
    """The run's configuration with the signal taken out: each strategy enters at its clock time on every session. A strategy
    with no clock time enters at the minute its own trades most often entered."""
    from strata.engine.conditions import walk_leaves
    from strata.engine.models import BacktestConfig
    c = BacktestConfig.model_validate(cfg).model_dump(mode="json")
    said = []
    for i, s in enumerate(c["strategies"]):
        model = BacktestConfig.model_validate(cfg).strategies[i]
        clocks = [l for l in walk_leaves(model.entry.conditions) if l.type == "entry_times" and l.enabled]
        times = ",".join(str((l.params or {}).get("times")) for l in clocks) if clocks else None
        if not times:
            mins = [int(t["entry_minute"]) for t in trades if t.get("strategy_idx") == i]
            if not mins:
                return None
            m = Counter(mins).most_common(1)[0][0]
            times = f"{m // 60:02d}:{m % 60:02d}"
            said.append(f"at {times} New York time (the minute at which its own trades most often entered) on every session")
        else:
            said.append("at the open on every session" if times == "open" else f"at {times} New York time on every session")
        s["entry"]["conditions"] = {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": times}}]}
        s["entry"]["sequence"] = []
        s["entry"]["systems"] = []
        s["entry"]["max_entries_per_day"] = 1
        s["name"] = f"{s['name']} (no signal)"
    return c, "; ".join(dict.fromkeys(said))


def run_baseline(rid: str, store: MarketStore | None = None, progress=None, should_cancel=None) -> dict:
    """Run the baseline of a stored backtest and keep it beside the run (baseline.json)."""
    from strata.api import db
    from strata.api.results import dumps, load_trades, run_dir
    from strata.engine.engine import run_backtest
    store = store or MarketStore()
    run = db.get_run(rid)
    if run is None or not isinstance(run.get("config"), dict) or not run["config"].get("strategies"):
        raise ValueError("this run has no strategy configuration to take the signal out of")
    trades = load_trades(rid)
    made = baseline_config(run["config"], trades)
    if made is None:
        raise ValueError("the run has no trades and no clock time, so there is no time at which to enter the baseline")
    cfg, definition = made
    res = run_backtest(cfg, store=store, record_paths=False, progress=(lambda a, b, d: progress(a / max(1, b), str(d))) if progress else None, should_cancel=should_cancel)
    b = [float(t["pnl"]) for t in res.trades]
    a = [float(t["pnl"]) for t in trades]
    dt = ST.diff_test(a, b)
    ex = res.meta.get("executability") or {}
    out = {"definition": definition, "window": [cfg["start"], cfg["end"]], "trades": len(b), "net": round(sum(b), 2), "average": round(sum(b) / len(b), 2) if b else None,
           "win_rate": round(sum(1 for x in b if x > 0) / len(b) * 100, 1) if b else None, "strategy_trades": len(a), "strategy_average": round(sum(a) / len(a), 2) if a else None,
           "difference_in_average": round(sum(a) / len(a) - sum(b) / len(b), 2) if a and b else None, "p_value": dt["p_value"], "signals": ex.get("signals"), "priced": ex.get("executed"),
           "note": "The baseline is a control, not a strategy: it has no free setting, and it is not counted among the configurations tried. It includes the sessions on which the signal fired."}
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "baseline.json").write_text(dumps(out))
    return out


def saved_baseline(rid: str) -> dict | None:
    from strata.api.results import run_dir
    p = run_dir(rid) / "baseline.json"
    return json.loads(p.read_text()) if p.exists() else None
