"""Execution on individual trade events: the same decisions, filled on the first print that could have filled them.

The backtest engine prices an option fill from one-minute TRADE BARS: at a decision it uses the last print it has seen,
which by construction happened BEFORE the decision. That is the usual approximation and it is good when a contract trades
every second; it is optimistic or pessimistic by an unknown amount when it does not, and it cannot say in which order a
target and a stop were touched inside one interval.

This module takes a finished trade-bar backtest and re-executes its fills on the provider's individual trades:

  exact timestamps      every print carries the exchange timestamp (nanoseconds); a fill is one print, or several
  latency               an order becomes eligible `latency_seconds` after its decision, entries and exits alike
  first eligible print  the fill is the FIRST print strictly AFTER the order became eligible. A print made before the
                        decision is never a fill: it is what the decision was made on
  stale rejection       no eligible print inside the working time: the order is not filled (entries) or is filled late
                        and says how late (exits, which are never abandoned)
  conditions and size   prints are filtered by their OPRA condition and by size before they can fill anything: by
                        default only single-leg, in-sequence, regular-hours trades; legs of spreads, stock-tied trades,
                        late reports and cancels are left out
  partial execution     `first_print`: the whole order at the first eligible print's price (it only establishes that the
                        contract traded there). `size_limited`: the order takes at most `participation` of each print's
                        size and walks forward until it is full; what is not filled in time is reported as partial
  multi-leg timing      each leg fills on its own contract's prints; the report gives how far apart the legs filled,
                        and can refuse an entry whose legs are further apart than a limit
  the opening           an order decided at the open can be held back for `open_delay_seconds`; fills in the first
                        minute are reported as their own group
  target and stop       `resting_orders`: the profit target rests as a limit order and the stop as a stop order from the
                        moment of the entry fill, and the tape decides which was touched first. This is a different
                        exit policy from the engine's (which looks at its clock ticks); both results are reported

What it does not do: an individual print is a price at which SOMEBODY traded. It is not a bid or an ask, and it does not
guarantee that this order would have been filled there. No spread is inferred from prints. Decisions that depend on the
underlying (signals, underlying targets) are not re-made: they are the backtest's.
"""
from __future__ import annotations

import calendar as _calendar
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from strata.core import calendar as cal
from strata.engine.models import BacktestConfig, Execution

# OPRA trade conditions exactly as the provider's own table gives them (GET /v1beta1/options/meta/conditions/trade,
# read on 2026-11-26), grouped by whether such a print says anything about where one more single-leg order could trade.
CONDITIONS: dict[str, tuple[str, str]] = {
    "I": ("AUTO - executed electronically", "single"), "S": ("ISOI - intermarket sweep order", "single"),
    "a": ("SLAN - single leg auction, non ISO", "single"), "b": ("SLAI - single leg auction, ISO", "single"),
    "c": ("SLCN - single leg cross, non ISO", "single"), "d": ("SCLI - single leg cross, ISO", "single"), "e": ("SLFT - single leg floor trade", "single"),
    "J": ("REOP - reopening after a halt", "single"),
    "j": ("MESL - multi leg auto-electronic trade against single leg(s)", "book"), "l": ("MASL - multi leg auction against single leg(s)", "book"),
    "m": ("MFSL - multi leg floor trade against single leg(s)", "book"),
    "f": ("MLET - multi leg auto-electronic trade", "complex"), "g": ("MLAT - multi leg auction", "complex"), "h": ("MLCT - multi leg cross", "complex"),
    "i": ("MLFT - multi leg floor trade", "complex"), "t": ("CBMO - multi leg floor trade of proprietary products", "complex"),
    "u": ("MCTP - multilateral compression trade", "complex"),
    "k": ("TLAT - stock options auction", "stock_tied"), "n": ("TLET - stock options auto-electronic trade", "stock_tied"), "o": ("TLCT - stock options cross", "stock_tied"),
    "p": ("TLFT - stock options floor trade", "stock_tied"), "q": ("TESL - stock options auto-electronic trade against single leg(s)", "stock_tied"),
    "r": ("TASL - stock options auction against single leg(s)", "stock_tied"), "s": ("TFSL - stock options floor trade against single leg(s)", "stock_tied"),
    "A": ("CANC - cancel of a trade previously reported", "void"), "C": ("CNCL - cancel of the last trade", "void"), "E": ("CNCO - cancel of the opening trade", "void"),
    "G": ("CNOL - cancel of the only trade", "void"),
    "B": ("OSEQ - reported late and out of sequence", "late"), "D": ("LATE - reported late, in sequence", "late"), "F": ("OPEN - late report of the opening trade, out of sequence", "late"),
    "H": ("OPNL - late report of the opening trade, in sequence", "late"),
    "v": ("EXHT - extended hours trade", "extended"),
}
CLASS_TEXT = {"single": "single-leg trades in sequence", "book": "legs of spreads that traded against single-leg orders", "complex": "legs of spreads priced as a package",
              "stock_tied": "option trades tied to a stock trade", "void": "cancels", "late": "late or out-of-sequence reports", "extended": "extended-hours trades", "unknown": "conditions the table does not list"}
FILTERS = {"single_leg": ("single",), "single_leg_and_book": ("single", "book"), "all_in_sequence": ("single", "book", "complex", "stock_tied")}
# exits the engine makes without trading the option: there is no print to look for
NOT_TRADED = {"expiration", "expired_worthless", "exercised", "assigned", "early_assignment", "end_of_backtest", "margin_call"}


def cond_class(code: str) -> str:
    return CONDITIONS.get(code, ("", "unknown"))[1]


@dataclass
class HiresSettings:
    latency_seconds: float = 0.0  # decision to eligible, entries and exits
    max_fill_delay_seconds: float = 300.0  # how long an ENTRY order works before it is given up
    conditions: str = "single_leg"  # which prints may fill an order: one of FILTERS
    min_size: int = 1  # prints smaller than this are ignored
    fill_model: str = "first_print"  # "first_print" | "size_limited"
    participation: float = 1.0  # size_limited: the share of each print's size the order may take
    open_delay_seconds: float = 0.0  # an order decided in the first seconds of the session is not eligible before this long after the open
    max_leg_gap_seconds: float | None = None  # refuse an entry whose legs fill further apart than this (None: report only)
    exit_policy: str = "decision_clock"  # "decision_clock" (the backtest's exit decisions) | "resting_orders" (target as a limit, stop as a stop, from the fill)
    limit_needs_trade_through: bool = True  # a resting limit is filled only by a print THROUGH its price; a print AT it is a touch (queue position unknown)
    exit_search_seconds: float = 3600.0  # how long after an exit decision a print is looked for before the bar price is kept and flagged

    @classmethod
    def from_execution(cls, ex: Execution, **over) -> "HiresSettings":
        base = {"latency_seconds": float(ex.latency_seconds), "max_fill_delay_seconds": max(60.0, float(ex.max_fill_delay_minutes) * 60.0), "participation": float(ex.participation_rate)}
        return cls(**{**base, **{k: v for k, v in over.items() if v is not None}})

    def validate(self) -> "HiresSettings":
        if self.conditions not in FILTERS:
            raise ValueError(f"conditions must be one of {', '.join(FILTERS)}")
        if self.fill_model not in ("first_print", "size_limited"):
            raise ValueError("fill_model must be first_print or size_limited")
        if self.exit_policy not in ("decision_clock", "resting_orders"):
            raise ValueError("exit_policy must be decision_clock or resting_orders")
        if not (0 < self.participation <= 1):
            raise ValueError("participation must be above 0 and at most 1")
        return self


# ----------------------------------------------------------------------------- the tape
@dataclass
class Tape:
    """Prints of one contract inside [t0, t1] (epoch nanoseconds), in time order. `complete` is False when the provider's
    answer was not read to its last page: nothing may then be concluded from the absence of a print."""
    occ: str
    t0: int
    t1: int
    ts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    price: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    size: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    cond: np.ndarray = field(default_factory=lambda: np.zeros(0, "U1"))
    exch: np.ndarray = field(default_factory=lambda: np.zeros(0, "U1"))
    complete: bool = True
    pages: int = 0
    cached: bool = False

    def __len__(self) -> int:
        return int(self.ts.size)

    def cut(self, t0: int, t1: int) -> "Tape":
        a, b = int(np.searchsorted(self.ts, t0, "left")), int(np.searchsorted(self.ts, t1, "right"))
        return Tape(self.occ, t0, t1, self.ts[a:b], self.price[a:b], self.size[a:b], self.cond[a:b], self.exch[a:b], self.complete, self.pages, self.cached)

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame({"ts": self.ts, "price": self.price, "size": self.size, "cond": self.cond.astype(str).tolist(), "exch": self.exch.astype(str).tolist()},
                            schema={"ts": pl.Int64, "price": pl.Float64, "size": pl.Int64, "cond": pl.String, "exch": pl.String})


def parse_ns(ts: str) -> int:
    """'2026-11-24T14:30:00.498817169Z' -> epoch nanoseconds, exactly (no float)."""
    s = ts.rstrip("Z")
    if "+" in s[10:]:
        s = s[:s.index("+", 10)]
    head, _, frac = s.partition(".")
    sec = _calendar.timegm(datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").timetuple())
    return sec * 1_000_000_000 + int((frac + "000000000")[:9])


def iso_ns(ns: int) -> str:
    sec, rem = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(sec, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{rem:09d}Z"


def et_clock(ns: int) -> str:
    sec, rem = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(sec, cal.ET).strftime("%H:%M:%S") + f".{rem // 1_000_000:03d}"


def tape_from_records(occ: str, t0: int, t1: int, recs: list[dict], complete: bool = True, pages: int = 0) -> Tape:
    if not recs:
        return Tape(occ, t0, t1, complete=complete, pages=pages)
    ts = np.array([parse_ns(r["t"]) for r in recs], np.int64)
    order = np.argsort(ts, kind="stable")
    return Tape(occ, t0, t1, ts[order], np.array([float(r["p"]) for r in recs], np.float64)[order], np.array([int(r.get("s") or 0) for r in recs], np.int64)[order],
                np.array([str(r.get("c") or "")[:1] for r in recs], "U1")[order], np.array([str(r.get("x") or "")[:1] for r in recs], "U1")[order], complete, pages)


class TradeSource:
    """Where individual trades come from. `fetch` returns EVERY print of the contract inside the window, or says it could not."""
    name = "source"
    label = "trade source"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def fetch(self, occ: str, t0: int, t1: int) -> Tape:
        raise NotImplementedError


class ListSource(TradeSource):
    """Prints held in memory (tests, and re-running a stored report's evidence)."""
    name, label = "memory", "prints in memory"

    def __init__(self, tapes: dict[str, list[dict]]):
        self.tapes = {occ: tape_from_records(occ, 0, 2**62, recs) for occ, recs in tapes.items()}
        self.asked: list[tuple[str, int, int]] = []

    def fetch(self, occ, t0, t1):
        self.asked.append((occ, t0, t1))
        full = self.tapes.get(occ)
        return full.cut(t0, t1) if full is not None else Tape(occ, t0, t1)


class AlpacaTrades(TradeSource):
    """Individual option trades from Alpaca (GET /v1beta1/options/trades), read to the last page of each window and kept in
    the provider cache so that a report can be rebuilt without asking again."""
    name, label = "alpaca", "Alpaca individual option trades"
    MAX_PAGES = 60  # 600,000 prints: more than any single contract-session measured (an at-the-money same-day SPY call had about 130,000)

    def __init__(self, store=None):
        from strata.config import get_settings
        from strata.data.providers import sources as S
        self.src = S.Alpaca(store)
        self.dir = get_settings().cache_dir / "alpaca" / "ticks"
        self.requests = 0
        self.pages = 0

    def available(self):
        return self.src.available()

    def _cached(self, occ: str, t0: int, t1: int) -> Tape | None:
        d = self.dir / occ
        if not d.exists():
            return None
        for f in d.glob("*.parquet"):
            try:
                a, b = (int(x) for x in f.stem.split("_"))
            except ValueError:
                continue
            if a <= t0 and b >= t1:
                df = pl.read_parquet(f)
                full = Tape(occ, a, b, df["ts"].to_numpy(), df["price"].to_numpy(), df["size"].to_numpy(), np.array(df["cond"].to_list(), "U1"), np.array(df["exch"].to_list(), "U1"), True, 0, True)
                return full.cut(t0, t1)
        return None

    def fetch(self, occ, t0, t1):
        from strata.data.providers import sources as S
        hit = self._cached(occ, t0, t1)
        if hit is not None:
            return hit
        recs, pages, complete = [], 0, True
        params = {"symbols": occ, "start": iso_ns(t0), "end": iso_ns(t1), "limit": 10000, "sort": "asc"}
        with S._client() as client:
            try:
                for page in self.src.pages(client, f"{self.src.DATA}/v1beta1/options/trades", params, "trades", max_pages=self.MAX_PAGES):
                    pages += 1
                    recs += (page or {}).get(occ, []) if isinstance(page, dict) else []
            except S.SourceError as e:
                if "pagination did not finish" not in str(e):
                    raise
                complete = False
        self.requests += 1
        self.pages += pages
        tape = tape_from_records(occ, t0, t1, recs, complete, pages).cut(t0, t1)
        tape.complete, tape.pages = complete, pages
        if complete:
            p = self.dir / occ / f"{t0}_{t1}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            tape.frame().write_parquet(p)
        return tape


# ----------------------------------------------------------------------------- filling an order on a tape
def eligible_mask(tape: Tape, s: HiresSettings) -> np.ndarray:
    classes = FILTERS[s.conditions]
    ok = np.array([cond_class(c) in classes for c in tape.cond], bool) if len(tape) else np.zeros(0, bool)
    return ok & (tape.size >= max(1, int(s.min_size))) & (tape.price > 0)


def fill_on_tape(tape: Tape, after_ns: int, until_ns: int, qty: int, s: HiresSettings) -> dict:
    """Fill `qty` contracts on prints strictly after `after_ns` and no later than `until_ns`.
    Returns {status: filled|partial|none|unknown, price, ts, prints, filled, ...}."""
    if not tape.complete:
        return {"status": "unknown", "why": "the provider's trades for this window were not read to the end"}
    ok = eligible_mask(tape, s) & (tape.ts > after_ns) & (tape.ts <= until_ns)
    idx = np.flatnonzero(ok)
    seen = int(((tape.ts > after_ns) & (tape.ts <= until_ns)).sum())
    if idx.size == 0:
        return {"status": "none", "prints_in_window": seen, "why": ("no trade of this contract was reported in the window" if seen == 0 else
                                                                 f"{seen} trade{'s' if seen != 1 else ''} in the window, none of a kind that can fill an order ({CLASS_TEXT.get('single')} of at least {s.min_size} contract{'s' if s.min_size != 1 else ''})")}
    if s.fill_model == "first_print":
        i = int(idx[0])
        return {"status": "filled", "price": float(tape.price[i]), "ts": int(tape.ts[i]), "filled": qty, "prints": [_print(tape, i, qty)], "prints_in_window": seen,
                "smaller_than_order": bool(tape.size[i] < qty)}
    need, used, notional = qty, [], 0.0
    for i in idx:
        take = min(need, int(math.floor(int(tape.size[i]) * s.participation)))
        if take <= 0:
            continue
        used.append(_print(tape, int(i), take))
        notional += take * float(tape.price[i])
        need -= take
        if need == 0:
            break
    got = qty - need
    if got == 0:
        return {"status": "none", "prints_in_window": seen, "why": f"no print was large enough to give this order a contract at {s.participation:.0%} participation"}
    return {"status": "filled" if need == 0 else "partial", "price": notional / got, "ts": int(used[-1]["ts"]), "filled": got, "prints": used, "prints_in_window": seen}


def _print(tape: Tape, i: int, take: int) -> dict:
    return {"ts": int(tape.ts[i]), "time": iso_ns(int(tape.ts[i])), "et": et_clock(int(tape.ts[i])), "price": float(tape.price[i]), "size": int(tape.size[i]), "taken": int(take),
            "condition": str(tape.cond[i]), "exchange": str(tape.exch[i])}


def priced(print_px: float, side: int, ex: Execution, option_legs: int, t_s: float, exp_close_s: int | None, t_open_s: int, t_close_s: int) -> float:
    """The engine's own arithmetic from a print to a fill price (side +1 buy, -1 sell): the per-contract slippage, the
    stress surcharges and the trade-print slippage, all against the order. Identical to what the bar run applied, so that
    the two results differ only in WHICH print they were priced from."""
    p = print_px + side * ex.slippage_per_contract
    if ex.extra_slippage_pct:
        p *= 1 + side * ex.extra_slippage_pct / 100.0
    p = max(0.0, p)
    if ex.multi_leg_extra_pct and option_legs >= 2:
        p = max(0.0, p * (1 + side * ex.multi_leg_extra_pct / 100.0))
    extra = 0.0
    minute, close_minute = 570 + (t_s - t_open_s) // 60, 570 + (t_close_s - t_open_s) // 60
    if ex.auction_extra_pct and ((ex.no_entry_first_minutes and minute < 570 + ex.no_entry_first_minutes) or (ex.no_entry_last_minutes and minute >= close_minute - ex.no_entry_last_minutes)):
        extra += ex.auction_extra_pct
    if ex.expiry_day_extra_pct and exp_close_s is not None and 0 <= exp_close_s - t_s <= ex.expiry_day_minutes * 60:
        extra += ex.expiry_day_extra_pct
    if extra:
        p = max(0.0, p * (1 + side * extra / 100.0))
    return max(0.0, p * (1 + side * ex.trade_bar_slippage_pct / 100.0))


# ----------------------------------------------------------------------------- one position
NS = 1_000_000_000


def _occ(sym: str, leg: dict) -> str:
    from strata.data.providers.sources import occ_symbol
    return occ_symbol(sym, date.fromisoformat(leg["expiration"]), leg["right"], float(leg["strike"]))


def _targets(cfg_strategy, entry_px: float, long: bool) -> tuple[float | None, float | None]:
    """(limit price of the profit target, trigger price of the stop) for a single option leg, from the entry fill."""
    ex = cfg_strategy.exits
    tp = ex.profit_target.value if ex.profit_target.enabled and ex.profit_target.mode == "pct" else None
    sl = ex.stop_loss.value if ex.stop_loss.enabled and ex.stop_loss.mode == "pct" else None
    if long:
        return (entry_px * (1 + tp / 100) if tp is not None else None), (entry_px * (1 - sl / 100) if sl is not None else None)
    return (entry_px * (1 - tp / 100) if tp is not None else None), (entry_px * (1 + sl / 100) if sl is not None else None)


def replay_position(t: dict, cfg: BacktestConfig, source: TradeSource, s: HiresSettings) -> dict:
    """One position of a trade-bar backtest, executed again on individual trades."""
    ex = cfg.execution
    strat = cfg.strategies[int(t.get("strategy_idx") or 0)]
    sym = t["symbol"]
    legs = [l for l in t["legs"] if l.get("kind") == "option"]
    out: dict = {"id": t["id"], "date": t["entry_date"], "symbol": sym, "bar": {"pnl": t.get("pnl"), "gross_pnl": t.get("gross_pnl"), "exit_reason": t.get("exit_reason"), "entry_time": t.get("entry_time"), "exit_time": t.get("exit_time")},
                 "legs": [], "status": "repriced", "notes": []}
    if not legs or any(l.get("price_source") != "trade_print" for l in legs) or any(l.get("kind") == "stock" for l in t["legs"]):
        out.update(status="not_applicable", notes=["this position was not priced from option trade prints alone (stock legs, quotes or modelled prices): there is nothing to re-execute on the option tape"])
        return out
    d = date.fromisoformat(t["entry_date"])
    o, c = cal.session_bounds(d)
    t_open, t_close = int(o.timestamp()), int(c.timestamp())
    sig = int(t.get("signal_ts") or t["entry_ts"])
    elig = max(sig + s.latency_seconds, t_open + s.open_delay_seconds) if sig - t_open < 60 else sig + s.latency_seconds
    after, until = int(round(elig * NS)), int(round((sig + s.max_fill_delay_seconds) * NS))
    n_opt = len(legs)
    # ---- entries: each leg on its own contract's prints
    fills = []
    for l in legs:
        occ = _occ(sym, l)
        side = 1 if l["side"] == "buy" else -1
        qty = abs(int(l.get("contracts") or 1))
        tape = source.fetch(occ, after, until)
        f = fill_on_tape(tape, after, until, qty, s)
        exp_close = int(cal.session_bounds(date.fromisoformat(l["expiration"]))[1].timestamp())
        row = {"id": l["id"], "contract": occ, "side": l["side"], "contracts": qty, "multiplier": float(l.get("multiplier") or 100.0), "exp_close": exp_close,
               "entry": {"decision": iso_ns(sig * NS), "eligible": iso_ns(after), "bar_price": l.get("entry_price"), "bar_print": l.get("entry_print"), "bar_time": t.get("entry_time"),
                         "bar_print_age_seconds": (t.get("meta") or {}).get("entry_print_age_seconds"), **{k: v for k, v in f.items() if k not in ("price", "ts")}}}
        if f["status"] in ("filled", "partial"):
            px = priced(f["price"], side, ex, n_opt, f["ts"] / NS, exp_close, t_open, t_close)
            row["entry"].update(print=f["price"], price=px, ts=f["ts"], time=iso_ns(f["ts"]), et=et_clock(f["ts"]), seconds_after_decision=round(f["ts"] / NS - sig, 3),
                                first_minute=bool(f["ts"] / NS - t_open < 60))
        fills.append((l, side, qty, f, row))
        out["legs"].append(row)
    bad = [r for *_x, f, r in fills if f["status"] != "filled"]
    if bad:
        kinds = {f["status"] for *_x, f, r in fills}
        out["status"] = "entry_unknown" if "unknown" in kinds else "entry_partial" if kinds <= {"filled", "partial"} else "entry_not_filled"
        why = bad[0]["entry"].get("why") or f"only {bad[0]['entry'].get('filled')} of {bad[0]['contracts']} contracts could be filled"
        out["notes"].append(f"entry: {why} between {et_clock(after)} and {et_clock(until)} New York time. The bar run filled at {legs[0].get('entry_price')} from a print made before its decision.")
        out["tick"] = {"pnl": None if out["status"] == "entry_unknown" else 0.0, "executed": False}
        return out
    ets = [f["ts"] for *_x, f, r in fills]
    entry_ns = max(ets)
    gap = (max(ets) - min(ets)) / NS
    out["entry_leg_gap_seconds"] = round(gap, 3) if n_opt > 1 else None
    if n_opt > 1 and s.max_leg_gap_seconds is not None and gap > s.max_leg_gap_seconds:
        out.update(status="entry_legged", tick={"pnl": 0.0, "executed": False})
        out["notes"].append(f"entry: the legs' first eligible prints are {gap:.1f} s apart, more than the {s.max_leg_gap_seconds:g} s allowed: the spread could not be entered as one position")
        return out
    # ---- exits
    resting = None
    if s.exit_policy == "resting_orders":
        if n_opt == 1 and not t.get("stage_exits"):
            resting = _resting_exit(t, cfg, strat, source, s, fills[0], entry_ns, t_open, t_close)
        else:
            out["notes"].append("resting orders are only replayed for a single option leg: a spread's limit rests in the complex order book, which single-leg prints do not show. The backtest's own exit decisions were used.")
    gross, exit_ns_all, delays = 0.0, [], []
    for l, side, qty, f, row in fills:
        entry_px = row["entry"]["price"]
        xr = (resting if resting is not None else _clock_exit(t, l, source, s, row, side, qty, ex, n_opt, t_open, t_close, entry_ns))
        row["exit"] = xr
        exit_px = xr.get("price")
        if exit_px is None:
            out["status"] = "exit_unknown"
            out["notes"].append(f"exit: {xr.get('why')}")
            out["tick"] = {"pnl": None, "executed": True}
            return out
        gross += side * qty * row["multiplier"] * (exit_px - entry_px)
        if xr.get("ts"):
            exit_ns_all.append(xr["ts"])
        if xr.get("seconds_after_decision") is not None:
            delays.append(xr["seconds_after_decision"])
        if xr.get("kept_bar_price"):
            out["notes"].append(f"exit: {xr.get('why')}")
    comm = float(t.get("commissions") or 0.0)
    reason = (resting or {}).get("reason") or t.get("exit_reason")
    out["tick"] = {"pnl": round(gross - comm, 2), "gross_pnl": round(gross, 2), "executed": True, "exit_reason": reason,
                   "entry_time": iso_ns(entry_ns), "exit_time": iso_ns(max(exit_ns_all)) if exit_ns_all else None,
                   "entry_seconds_after_decision": round(entry_ns / NS - sig, 3), "exit_seconds_after_decision": max(delays) if delays else None}
    out["pnl_difference"] = round(out["tick"]["pnl"] - float(t.get("pnl") or 0.0), 2)
    if resting is not None and reason != t.get("exit_reason"):
        out["notes"].append(f"exit: with the target and the stop resting from the fill, the tape shows {reason.replace('_', ' ')} at {resting.get('et')}; the backtest's clock found {str(t.get('exit_reason')).replace('_', ' ')} at {str(t.get('exit_time'))[11:19]}")
    return out


def _clock_exit(t, l, source, s, row, side, qty, ex, n_opt, t_open, t_close, entry_ns) -> dict:
    """The backtest's own exit decision for this leg, filled on the first eligible print after it."""
    reason = l.get("exit_reason") or t.get("exit_reason")
    dec = int(l.get("exit_ts") or t.get("exit_ts") or 0)
    dec_sig = int(t.get("exit_signal_ts") or dec)  # a deferred exit was decided earlier than it was priced
    base = {"policy": "decision_clock", "reason": reason, "bar_price": l.get("exit_price"), "bar_print": l.get("exit_print"), "bar_time": datetime.fromtimestamp(dec, cal.ET).isoformat() if dec else None}
    if reason in NOT_TRADED or l.get("exit_price") is None:
        return {**base, "price": l.get("exit_price"), "settled": True, "why": "the contract was not traded out: it expired or was settled, at the same value in both results"}
    d2 = datetime.fromtimestamp(dec_sig, cal.ET).date()
    after = max(int(round((dec_sig + s.latency_seconds) * NS)), entry_ns)
    close2 = int(cal.session_bounds(d2)[1].timestamp())
    until = int(round(min(dec_sig + s.exit_search_seconds, close2) * NS))
    f = fill_on_tape(source.fetch(row["contract"], after, until), after, until, qty, s)
    info = {**base, "decision": iso_ns(dec_sig * NS), "eligible": iso_ns(after), **{k: v for k, v in f.items() if k not in ("price", "ts")}}
    if f["status"] == "filled" or (f["status"] == "partial" and f.get("filled")):
        o2 = int(cal.session_bounds(d2)[0].timestamp())
        px = priced(f["price"], -side, ex, n_opt, f["ts"] / NS, row["exp_close"], o2, close2)
        return {**info, "print": f["price"], "price": px, "ts": f["ts"], "time": iso_ns(f["ts"]), "et": et_clock(f["ts"]), "seconds_after_decision": round(f["ts"] / NS - dec_sig, 3),
                "late": bool(f["ts"] / NS - dec_sig > s.max_fill_delay_seconds)}
    if f["status"] == "unknown":
        return {**info, "price": None}
    return {**info, "price": l.get("exit_price"), "kept_bar_price": True,
            "why": f"no eligible print of {row['contract']} within {s.exit_search_seconds / 60:g} min after the exit decision at {et_clock(dec_sig * NS)[:8]}; the bar run's exit price ({l.get('exit_price')}) is kept and the exit is not established by the tape"}


def _resting_exit(t, cfg, strat, source, s, fill, entry_ns, t_open, t_close) -> dict | None:
    """Single option leg: the profit target rests as a limit and the stop as a stop order from the entry fill. The tape says
    which was touched first. Falls back to the backtest's own exit decision when neither is touched before it."""
    l, side, qty, f, row = fill
    ex = cfg.execution
    entry_px = row["entry"]["price"]
    limit, stop = _targets(strat, entry_px, side > 0)
    if limit is None and stop is None:
        return None
    dec = int(t.get("exit_signal_ts") or l.get("exit_ts") or t.get("exit_ts") or 0)
    horizon_s = dec if dec else t_close
    d2 = datetime.fromtimestamp(horizon_s, cal.ET).date()
    if d2 != date.fromisoformat(t["entry_date"]):
        return None  # held over sessions: the resting replay stays within one session; the clock exit is used
    tape = source.fetch(row["contract"], entry_ns, int(horizon_s * NS))
    if not tape.complete:
        return {"policy": "resting_orders", "price": None, "why": "the contract's trades between the entry and the exit decision were not read to the end, so which order was touched first is not known"}
    ok = eligible_mask(tape, s) & (tape.ts > entry_ns)
    p, ts = tape.price[ok], tape.ts[ok]
    long = side > 0
    hit_t = (p > limit if s.limit_needs_trade_through else p >= limit) if (limit is not None and long) else (p < limit if s.limit_needs_trade_through else p <= limit) if limit is not None else np.zeros(p.size, bool)
    hit_s = (p <= stop if long else p >= stop) if stop is not None else np.zeros(p.size, bool)
    touch = ((p == limit) & ~hit_t) if limit is not None else np.zeros(p.size, bool)
    it = int(np.argmax(hit_t)) if hit_t.any() else None
    is_ = int(np.argmax(hit_s)) if hit_s.any() else None
    base = {"policy": "resting_orders", "limit": round(limit, 4) if limit is not None else None, "stop": round(stop, 4) if stop is not None else None, "prints_scanned": int(p.size),
            "touches_at_the_limit_not_counted": int(touch[: (it if it is not None else p.size)].sum()), "bar_price": l.get("exit_price"), "bar_reason": t.get("exit_reason")}
    if it is None and is_ is None:
        return None if not dec else {**_clock_exit(t, l, source, s, row, side, qty, ex, 1, t_open, t_close, entry_ns), **{k: v for k, v in base.items() if k in ("limit", "stop", "prints_scanned", "touches_at_the_limit_not_counted")}, "policy": "resting_orders",
                                     "neither_touched": True}
    if it is not None and (is_ is None or it < is_):
        # the limit order: filled at its own price by the first print through it
        px = priced(float(limit), -side, ex, 1, ts[it] / NS, row["exp_close"], t_open, t_close)
        return {**base, "reason": "profit_target", "price": px, "print": float(p[it]), "ts": int(ts[it]), "time": iso_ns(int(ts[it])), "et": et_clock(int(ts[it])), "first": "target",
                "seconds_after_decision": 0.0, "stop_touched_later": bool(is_ is not None)}
    # the stop: triggered by the print, filled by the first eligible print after it (a market order has no price of its own)
    trig = int(ts[is_])
    after = int(round(trig + s.latency_seconds * NS))
    later = np.flatnonzero(ts > after)
    if later.size == 0:
        more = source.fetch(row["contract"], after, int(min(trig / NS + s.exit_search_seconds, t_close) * NS))
        f2 = fill_on_tape(more, after, more.t1, qty, s)
        if f2["status"] not in ("filled", "partial"):
            return {**base, "reason": "stop_loss", "price": None, "why": f"the stop was triggered at {et_clock(trig)} by a print at {float(p[is_]):.2f}, and no eligible print followed it before the close: where it would have been filled is not known"}
        fp, fts = f2["price"], f2["ts"]
    else:
        fp, fts = float(p[later[0]]), int(ts[later[0]])
    px = priced(fp, -side, ex, 1, fts / NS, row["exp_close"], t_open, t_close)
    return {**base, "reason": "stop_loss", "price": px, "print": fp, "ts": fts, "time": iso_ns(fts), "et": et_clock(fts), "first": "stop", "trigger_print": float(p[is_]), "trigger_time": iso_ns(trig),
            "seconds_after_decision": round((fts - trig) / NS, 3), "target_touched_later": bool(it is not None)}


# ----------------------------------------------------------------------------- a whole run
def replay(trades: list[dict], cfg: BacktestConfig | dict, source: TradeSource, settings: HiresSettings | None = None, progress: Callable | None = None, should_cancel: Callable | None = None) -> dict:
    """Re-execute every position of a trade-bar backtest on individual trades. Returns the report (JSON-serialisable)."""
    cfg = BacktestConfig.model_validate(cfg) if isinstance(cfg, dict) else cfg
    s = (settings or HiresSettings.from_execution(cfg.execution)).validate()
    rep: dict = {"source": source.name, "source_label": source.label, "settings": asdict(s), "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pricing_mode": cfg.pricing.mode,
                 "positions": [], "status": "ran"}
    ok, why = source.available()
    if cfg.pricing.mode != "trades":
        rep.update(status="not_applicable", reason="Only a backtest priced from option trade prints can be re-executed on individual trades: this one used " + ("bid/ask quotes" if cfg.pricing.mode == "historical" else "modelled premiums") + ".")
        return rep
    if not ok:
        rep.update(status="blocked", reason=f"BLOCKED: {why}. Individual trades come from the provider and need its key.")
        return rep
    for i, t in enumerate(trades):
        if should_cancel and should_cancel():
            rep["status"] = "cancelled"
            break
        if progress:
            progress(i / max(1, len(trades)), f"{t['entry_date']} position {t['id']} ({i + 1}/{len(trades)})")
        try:
            rep["positions"].append(replay_position(t, cfg, source, s))
        except Exception as e:  # noqa: BLE001  (one position that cannot be read must not lose the others)
            rep["positions"].append({"id": t.get("id"), "date": t.get("entry_date"), "status": "error", "notes": [f"{type(e).__name__}: {e}"[:240]], "bar": {"pnl": t.get("pnl")}})
    rep["summary"] = summarise(rep["positions"], s)
    rep["requests"] = getattr(source, "requests", None)
    rep["pages"] = getattr(source, "pages", None)
    if progress:
        progress(1.0, "done")
    return rep


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4), "p05": round(float(np.percentile(a, 5)), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def summarise(pos: list[dict], s: HiresSettings) -> dict:
    by: dict[str, int] = {}
    for p in pos:
        by[p["status"]] = by.get(p["status"], 0) + 1
    both = [p for p in pos if p["status"] == "repriced" and (p.get("tick") or {}).get("pnl") is not None]
    unfilled = [p for p in pos if p["status"] in ("entry_not_filled", "entry_partial", "entry_legged")]
    unknown = [p for p in pos if p["status"] in ("entry_unknown", "exit_unknown", "error")]
    bar_all = sum(float((p.get("bar") or {}).get("pnl") or 0.0) for p in pos if p["status"] != "not_applicable")
    bar_both = sum(float(p["bar"]["pnl"] or 0.0) for p in both)
    tick_both = sum(float(p["tick"]["pnl"]) for p in both)
    e_px, x_px, e_lat, x_lat, conds, first_min, late, kept = [], [], [], [], {}, [], 0, 0
    for p in both:
        for l in p["legs"]:
            en, xi = l["entry"], l.get("exit") or {}
            if en.get("bar_price") and en.get("price") is not None:
                e_px.append((en["price"] / en["bar_price"] - 1) * 100 * (1 if l["side"] == "buy" else -1))  # positive: the tape was worse for the order
                if en.get("first_minute"):
                    first_min.append(e_px[-1])
            if en.get("seconds_after_decision") is not None:
                e_lat.append(en["seconds_after_decision"])
            if xi.get("bar_price") and xi.get("price") is not None and not xi.get("settled") and not xi.get("kept_bar_price"):
                x_px.append((xi["price"] / xi["bar_price"] - 1) * 100 * (-1 if l["side"] == "buy" else 1))
            if xi.get("seconds_after_decision") is not None and xi.get("policy") == "decision_clock" and not xi.get("neither_touched"):
                x_lat.append(xi["seconds_after_decision"])
            late += int(bool(xi.get("late")))
            kept += int(bool(xi.get("kept_bar_price")))
            for pr in (en.get("prints") or []) + (xi.get("prints") or []):
                conds[pr["condition"]] = conds.get(pr["condition"], 0) + 1
    changed = [p for p in both if (p["tick"].get("exit_reason") or "") != (p["bar"].get("exit_reason") or "")]
    wins = lambda key: sum(1 for p in both if float(p[key]["pnl"] or 0.0) > 0)
    out = {"positions": len(pos), "by_status": by, "repriced": len(both), "entries_not_filled": len(unfilled), "not_established": len(unknown),
           "bar_pnl_all_positions": round(bar_all, 2), "bar_pnl_repriced": round(bar_both, 2), "tick_pnl_repriced": round(tick_both, 2), "difference": round(tick_both - bar_both, 2),
           "bar_pnl_of_entries_not_filled": round(sum(float(p["bar"]["pnl"] or 0.0) for p in unfilled), 2),
           "tick_pnl_all_positions": round(tick_both, 2) if not unknown else None,
           "winners": {"bar": wins("bar"), "tick": wins("tick")},
           "per_position_difference": _stats([p["pnl_difference"] for p in both]),
           "entry_price_worse_pct": _stats(e_px), "exit_price_worse_pct": _stats(x_px), "entry_seconds_after_decision": _stats(e_lat), "exit_seconds_after_decision": _stats(x_lat),
           "first_minute_entries": {"n": len(first_min), "price_worse_pct": _stats(first_min)}, "late_exits": late, "exits_not_established": kept,
           "exit_reason_changed": len(changed), "exit_reason_changes": [{"id": p["id"], "date": p["date"], "bar": p["bar"].get("exit_reason"), "tick": p["tick"].get("exit_reason")} for p in changed[:40]],
           "conditions_of_prints_used": {k: {"count": v, "meaning": CONDITIONS.get(k, ("not in the provider's table", ""))[0]} for k, v in sorted(conds.items(), key=lambda kv: -kv[1])}}
    n = len(both)
    out["text"] = (f"{n} of {len(pos)} positions were executed again on individual trades. " if n else f"None of the {len(pos)} positions could be executed again on individual trades. ")
    if n:
        d = out["difference"]
        out["text"] += (f"Their result is {tick_both:+,.2f} on individual trades against {bar_both:+,.2f} on minute bars ({d:+,.2f}). "
                        f"An entry filled {out['entry_seconds_after_decision']['median']:g} s after its decision at the median, at a price {abs(out['entry_price_worse_pct']['mean']):.2f}% "
                        f"{'worse' if out['entry_price_worse_pct']['mean'] > 0 else 'better'} for the order than the bar price on average. " if out["entry_seconds_after_decision"] and out["entry_price_worse_pct"] else "")
    if unfilled:
        out["text"] += f"{len(unfilled)} entr{'ies' if len(unfilled) != 1 else 'y'} found no eligible print inside the working time and {'were' if len(unfilled) != 1 else 'was'} not filled (the bar run made {out['bar_pnl_of_entries_not_filled']:+,.2f} on them). "
    if unknown:
        out["text"] += f"{len(unknown)} could not be established (the provider's trades were not read to the end, or failed). "
    if s.exit_policy == "resting_orders":
        out["text"] += f"With the target and the stop resting from the fill, {len(changed)} position{'s' if len(changed) != 1 else ''} left for a different reason than on the backtest's clock. "
    out["text"] += "A print is a price at which a trade happened, not a bid or an ask: this narrows where an order could have filled, it does not establish it."
    return out


def csv_rows(rep: dict) -> list[dict]:
    """One row per leg fill pair, for export and independent checking."""
    rows = []
    for p in rep.get("positions") or []:
        for l in p.get("legs") or []:
            en, xi = l.get("entry") or {}, l.get("exit") or {}
            rows.append({"position": p["id"], "date": p["date"], "status": p["status"], "contract": l["contract"], "side": l["side"], "contracts": l["contracts"],
                         "entry_decision_utc": en.get("decision"), "entry_bar_price": en.get("bar_price"), "entry_bar_print_age_s": en.get("bar_print_age_seconds"),
                         "entry_tick_time_utc": en.get("time"), "entry_tick_print": en.get("print"), "entry_tick_price": en.get("price"), "entry_seconds_after_decision": en.get("seconds_after_decision"),
                         "entry_condition": ((en.get("prints") or [{}])[0]).get("condition"), "entry_exchange": ((en.get("prints") or [{}])[0]).get("exchange"), "entry_print_size": ((en.get("prints") or [{}])[0]).get("size"),
                         "exit_policy": xi.get("policy"), "exit_reason_bar": (p.get("bar") or {}).get("exit_reason"), "exit_reason_tick": (p.get("tick") or {}).get("exit_reason"),
                         "exit_decision_utc": xi.get("decision") or xi.get("trigger_time"), "exit_bar_price": xi.get("bar_price"), "exit_tick_time_utc": xi.get("time"), "exit_tick_print": xi.get("print"),
                         "exit_tick_price": xi.get("price"), "exit_seconds_after_decision": xi.get("seconds_after_decision"),
                         "pnl_bar": (p.get("bar") or {}).get("pnl"), "pnl_tick": (p.get("tick") or {}).get("pnl"), "pnl_difference": p.get("pnl_difference"), "notes": " | ".join(p.get("notes") or [])})
    return rows


def run_for(rid: str, store=None, settings: dict | None = None, source: TradeSource | None = None, progress=None, should_cancel=None, save: bool = True) -> dict:
    """Re-execute a stored backtest run on individual trades and keep the report beside its result."""
    from strata.api.results import dumps, load_result, load_trades, run_dir
    res = load_result(rid)
    cfg = BacktestConfig.model_validate(res["config"])
    s = HiresSettings.from_execution(cfg.execution, **(settings or {})).validate()
    rep = replay(load_trades(rid), cfg, source or AlpacaTrades(store), s, progress, should_cancel)
    rep["run_id"] = rid
    if save and rep["status"] in ("ran", "cancelled"):
        name = "hires.json" if s.exit_policy == "decision_clock" else "hires_resting.json"
        Path(run_dir(rid) / name).write_text(dumps(rep))
    return rep


def saved(rid: str) -> dict:
    """The stored reports of a run: {"decision_clock": ..., "resting_orders": ...} (absent when not run)."""
    import json
    from strata.api.results import run_dir
    out = {}
    for key, name in (("decision_clock", "hires.json"), ("resting_orders", "hires_resting.json")):
        p = run_dir(rid) / name
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out
