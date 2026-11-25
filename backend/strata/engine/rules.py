"""RuleStrategy: executes a declarative StrategyConfig (what the builder and the AI lab produce).

Signals and fills are separate events:
  * rules are evaluated on the signal clock (the underlying's bars)
  * a passing entry rule creates an ORDER stamped with the signal time
  * the order fills at the first option quote observed at/after signal + latency that is
    still fresh, provided it arrives within `max_fill_delay_minutes`; otherwise the order
    expires and is reported - it is never filled at a later, unrelated quote
  * exits requested while no fresh quote exists are filled at the next observable quote
"""
from __future__ import annotations

import math

from strata.core import calendar as cal
from strata.engine import expr as X
from strata.engine.conditions import evaluate, explicit_minutes, walk_leaves
from strata.engine.engine import Context, Strategy
from strata.engine.sequence import SequenceTracker
from strata.engine.margin import MLeg, max_profit_loss, requirement
from strata.engine.models import StrategyConfig
from strata.engine.portfolio import Position

LOSS_REASONS = {"stop_loss", "trailing_stop", "underlying_stop", "greek_stop", "strike_breach"}
PROFIT_REASONS = {"profit_target", "underlying_target"}
PRIORITY = ["stop_loss", "trailing_stop", "strike_breach", "underlying_stop", "greek_stop", "custom_exit", "roll", "time_exit", "dte_exit", "max_hold", "underlying_target", "profit_target"]
REASON_CLASS = {"profit_target": "target", "underlying_target": "target", "stop_loss": "stop", "trailing_stop": "stop", "underlying_stop": "stop",
                "time_exit": "time", "dte_exit": "time", "max_hold": "time"}


SEQUENCE_MAX_BAR_SECONDS = 300  # the same threshold the data planner uses for step-by-step rules


def _res_label(sec: int) -> str:
    return "daily" if sec >= 86400 else f"{sec // 60}-minute"


class RuleStrategy(Strategy):
    def __init__(self, cfg: StrategyConfig, idx: int = 0):
        self.cfg = cfg
        self.idx = idx
        self.name = cfg.name
        self.symbol = cfg.symbol
        self.vol_symbol = cfg.vol_symbol
        self.allocation = cfg.allocation
        self.regime = cfg.regime
        self.entries_today = 0
        self.last_exit_t = None
        self.last_exit_reason = None
        self.exit_today = False
        self.pending: dict | None = None
        self.na_today: set[str] = set()
        ex = cfg.exits
        self.entry_minutes = set()
        for leaf in walk_leaves(cfg.entry.conditions):
            if leaf.type == "entry_times":  # open / close always have a tick, so only clock times can be unobservable
                self.entry_minutes |= {cal.parse_hhmm(x) for x in str((leaf.params or {}).get("times", "")).replace(";", ",").split(",") if x.strip() and x.strip().lower() not in ("open", "close")}
        self.explicit_minutes = explicit_minutes(cfg.entry.conditions) | explicit_minutes(ex.custom)
        self.tracker = SequenceTracker(cfg.entry.sequence)
        self.add_pending: dict[int, int] = {}  # root position id -> tick at which an add was first attempted (for reporting)
        import ast as _ast
        for src in [s.when for s in cfg.entry.sequence] + [a.when for a in cfg.entry.adds]:
            try:
                for n in _ast.walk(X.parse(src)):
                    if isinstance(n, _ast.Call) and n.func.id == "hhmm" and all(isinstance(a, _ast.Constant) for a in n.args):
                        self.explicit_minutes.add(int(n.args[0].value) * 60 + (int(n.args[1].value) if len(n.args) > 1 else 0))
            except X.ExprError:
                pass
        if ex.time_exit.enabled:
            self.explicit_minutes.add(cal.parse_hhmm(ex.time_exit.time))
        if ex.dte_exit.enabled:
            self.explicit_minutes.add(cal.parse_hhmm(ex.dte_exit.time))

    def on_day_start(self, ctx: Context):
        self.entries_today = 0
        self.exit_today = False
        self.na_today = set()
        self.pending = None
        self.tracker.reset()
        ctx.seq = self.tracker
        # a sequence of intraday events is only meaningful on fine bars: on hourly or daily bars the
        # steps would collapse onto a handful of ticks, so such a session is excluded, never "no signal"
        self.coarse = bool(self.tracker) and ctx.session_res > SEQUENCE_MAX_BAR_SECONDS

    def on_day_end(self, ctx: Context):
        diag = ctx.bt.diag
        if self.pending:
            self._expire(ctx, "the session closed before an eligible option quote arrived")
        if getattr(self, "coarse", False):
            diag.exclude(ctx.today, f"{self.name}: the entry is a sequence of intraday events, which needs 5-minute or finer bars; only {_res_label(ctx.session_res)} bars are stored for the session")
            return
        if self.entries_today:
            return
        if self.na_today:
            for r in sorted(self.na_today):
                diag.exclude(ctx.today, f"{self.name}: rule not evaluable - {r}")
        elif self.entry_minutes and not (self.entry_minutes & ctx.day_minutes):
            times = ", ".join(cal.fmt_hhmm(m) for m in sorted(self.entry_minutes))
            diag.exclude(ctx.today, f"{self.name}: entry time {times} ET cannot be observed with {_res_label(ctx.session_res)} price bars")

    def on_position_closed(self, ctx: Context, pos: Position):
        self.last_exit_t = pos.exit_ts
        self.last_exit_reason = pos.exit_reason
        self.exit_today = True

    # ------------------------------------------------------------------ exits
    def check_exits(self, ctx: Context, pos: Position):
        ex = self.cfg.exits
        if pos.pending_exit:
            ctx.close(pos, pos.pending_exit)
            return
        # price-based rules need fresh option quotes; time / underlying rules do not
        fresh = not any(l.stale for l in pos.legs if l.is_open)
        u = max(pos.units, 1)
        hits: list[tuple[str, str]] = []
        tgt_thr = stop_thr = None
        pnl = pct = float("nan")
        if fresh:
            pnl = pos.gross_pnl(use_mid=ex.trigger_price == "mid")
            pct = pnl / pos.basis * 100
            pos.peak_pnl = max(pos.peak_pnl, pnl)
            if ex.profit_target.enabled:
                thr = ex.profit_target.value if ex.profit_target.mode == "pct" else ex.profit_target.value * u
                val = pct if ex.profit_target.mode == "pct" else pnl
                tgt_thr = thr / 100 * pos.basis if ex.profit_target.mode == "pct" else thr
                if val >= thr:
                    hits.append(("profit_target", f"P&L {pnl:+,.2f} ({pct:+.1f}% of premium) >= target {ex.profit_target.value:g}{'%' if ex.profit_target.mode == 'pct' else ' $/unit'}"))
            if ex.stop_loss.enabled:
                thr = ex.stop_loss.value if ex.stop_loss.mode == "pct" else ex.stop_loss.value * u
                val = pct if ex.stop_loss.mode == "pct" else pnl
                stop_thr = -thr / 100 * pos.basis if ex.stop_loss.mode == "pct" else -thr
                if val <= -thr:
                    hits.append(("stop_loss", f"P&L {pnl:+,.2f} ({pct:+.1f}% of premium) <= stop -{ex.stop_loss.value:g}{'%' if ex.stop_loss.mode == 'pct' else ' $/unit'}"))
            ts = ex.trailing_stop
            if ts.enabled:
                scale = pos.basis / 100 if ts.mode == "pct" else u
                act, trail = ts.activate * scale, ts.trail * scale
                if pos.peak_pnl >= act and pos.peak_pnl - pnl >= trail:
                    hits.append(("trailing_stop", f"peak P&L {pos.peak_pnl:,.2f} gave back {pos.peak_pnl - pnl:,.2f} >= trail {trail:,.2f}"))
        front = pos.front_dte(ctx.today_num)
        # a time exit later than an early close fires at that session's close
        if ex.time_exit.enabled and ctx.minute >= min(cal.parse_hhmm(ex.time_exit.time), ctx.close_minute) and (ex.time_exit.dte_lte is None or front <= ex.time_exit.dte_lte):
            hits.append(("time_exit", f"time {cal.fmt_hhmm(ctx.minute)} >= {ex.time_exit.time} with front DTE {front}"))
        if ex.dte_exit.enabled and front <= ex.dte_exit.dte and ctx.minute >= min(cal.parse_hhmm(ex.dte_exit.time), ctx.close_minute):
            hits.append(("dte_exit", f"front DTE {front} <= {ex.dte_exit.dte}"))
        if ex.max_hold.enabled:
            held_min = (ctx.t - pos.entry_ts) / 60
            if ex.max_hold.minutes and held_min >= ex.max_hold.minutes:
                hits.append(("max_hold", f"held {held_min:.0f} min >= {ex.max_hold.minutes}"))
            if ex.max_hold.days:
                from strata.engine.market import from_day_num
                sessions = cal.trading_days_between(from_day_num(pos.entry_day), ctx.today)
                entry_min = cal.minute_of_day_et(pos.entry_ts * 1_000_000_000)
                # N sessions later at the same time of day (or that session's close if it ends earlier)
                if sessions > ex.max_hold.days or (sessions == ex.max_hold.days and ctx.minute >= min(entry_min, ctx.close_minute)):
                    hits.append(("max_hold", f"held {sessions} sessions >= {ex.max_hold.days}"))
        ue = ex.underlying
        spot = ctx.price(pos.symbol)
        if ue.enabled and math.isfinite(spot):
            d0 = sum(l.ratio * (l.entry_greeks[0] if math.isfinite(l.entry_greeks[0]) else 0) * (l.multiplier if l.kind == "option" else 1) for l in pos.legs)
            mv = (spot / pos.entry_spot - 1) * 100
            neutral = abs(d0) < 5
            directional = mv * (1 if d0 > 0 else -1)
            if ue.move_against_pct is not None:
                against = abs(mv) if neutral else -directional
                if against >= ue.move_against_pct:
                    hits.append(("underlying_stop", f"underlying {spot:.2f} moved {mv:+.2f}% from entry {pos.entry_spot:.2f} ({'abs' if neutral else 'against'} {against:.2f}% >= {ue.move_against_pct:g}%)"))
            if ue.move_for_pct is not None and not neutral and directional >= ue.move_for_pct:
                hits.append(("underlying_target", f"underlying {spot:.2f} moved {mv:+.2f}% from entry {pos.entry_spot:.2f} in favour >= {ue.move_for_pct:g}%"))
            if ue.strike_breach:
                b = ue.breach_buffer_pct / 100
                for l in pos.legs:
                    if l.is_open and l.kind == "option" and l.qty < 0:
                        if (l.is_call and spot > l.strike * (1 + b)) or (not l.is_call and spot < l.strike * (1 - b)):
                            hits.append(("strike_breach", f"underlying {spot:.2f} breached short {l.label()}"))
                            break
        ge = ex.greeks
        if ge.enabled and fresh:
            dlt = pos.greeks()[0] / u
            if ge.max_abs_delta is not None and abs(dlt) > ge.max_abs_delta:
                hits.append(("greek_stop", f"|delta|/unit {abs(dlt):.1f} > {ge.max_abs_delta}"))
            if ge.short_leg_delta is not None:
                for l in pos.legs:
                    if l.is_open and l.kind == "option" and l.qty < 0 and math.isfinite(l.greeks[0]) and abs(l.greeks[0]) > ge.short_leg_delta:
                        hits.append(("greek_stop", f"short {l.label()} |delta| {abs(l.greeks[0]):.2f} > {ge.short_leg_delta}"))
                        break
        if ex.custom is not None and ex.custom.children:
            ok, expl, _ = evaluate(ex.custom, ctx, pos)
            if ok:
                hits.append(("custom_exit", "; ".join(e["text"] for e in expl if e.get("passed") and not e.get("group"))))
        rr = ex.roll
        if rr.enabled and pos.roll_count < rr.max_rolls and fresh:
            trig = False
            if rr.trigger == "dte" and front <= rr.value:
                trig = True
            elif rr.trigger == "short_delta":
                trig = any(l.is_open and l.qty < 0 and math.isfinite(l.greeks[0]) and abs(l.greeks[0]) >= rr.value for l in pos.legs)
            elif rr.trigger == "loss_pct" and pct <= -rr.value:
                trig = True
            elif rr.trigger == "profit_pct" and pct >= rr.value:
                trig = True
            if trig:
                hits.append(("roll", f"roll trigger {rr.trigger} {rr.value:g} hit (roll {pos.roll_count + 1}/{rr.max_rolls})"))

        # intrabar ambiguity: could the underlying's path between ticks have touched both the
        # stop and the target? (delta-gamma estimate from completed bars; flag only)
        if fresh and tgt_thr is not None and stop_thr is not None and ctx.prev_t < ctx.t:
            ser = ctx.series(pos.symbol)
            hi, lo = ser.range_between(ctx.prev_t, ctx.t)
            if math.isfinite(hi):
                d, g, _, _ = pos.greeks()
                est = [pnl + d * (x - spot) + 0.5 * g * (x - spot) ** 2 for x in (hi, lo)]
                if max(est) >= tgt_thr and min(est) <= stop_thr:
                    pos.flags.add("intrabar_ambiguous")

        if hits:
            reasons = {r for r, _ in hits}
            if reasons & LOSS_REASONS and reasons & PROFIT_REASONS:
                pos.flags.add("ambiguous_exit")
                ctx.bt.diag.ambiguous_exits += 1
            chosen = next(r for r in PRIORITY if r in reasons)
            explain = [{"text": txt, "passed": True, "reason": r} for r, txt in hits]
            if chosen == "roll":
                self._roll(ctx, pos, explain)
            else:
                pos.exit_explain = explain
                ctx.close(pos, chosen, explain=explain)
            return
        if not fresh:
            return
        for rule in ex.leg_rules:
            idxs = [i for i, l in enumerate(pos.legs) if l.is_open and (l.leg_id == rule.target or l.group == rule.target)]
            for i in list(idxs):
                l = pos.legs[i]
                px = l.mid if ex.trigger_price == "mid" else (l.ask if l.qty < 0 else l.bid)
                if not math.isfinite(px) or l.entry_price <= 0:
                    continue
                short = l.qty < 0
                chg = (px / l.entry_price - 1) * 100
                if rule.stop_pct is not None and ((short and chg >= rule.stop_pct) or (not short and -chg >= rule.stop_pct)):
                    ctx.close(pos, "leg_stop", legs=idxs, explain=[{"text": f"{l.label()} price {px:.2f} vs entry {l.entry_price:.2f} ({chg:+.0f}%) hit leg stop {rule.stop_pct:g}% on {rule.target}", "passed": True}])
                    break
                if rule.target_pct is not None and ((short and -chg >= rule.target_pct) or (not short and chg >= rule.target_pct)):
                    ctx.close(pos, "leg_target", legs=idxs, explain=[{"text": f"{l.label()} price {px:.2f} vs entry {l.entry_price:.2f} ({chg:+.0f}%) hit leg target {rule.target_pct:g}% on {rule.target}", "passed": True}])
                    break
            if not pos.is_open:
                return

    def _roll(self, ctx: Context, pos: Position, explain):
        if not ctx.close(pos, "roll", explain=explain):
            return
        sel, why, _ = ctx.select(self.cfg.legs)
        if sel is None:
            ctx.log("roll_failed", f"roll could not reselect contracts: {why}", pos)
            return
        new, why = ctx.open(sel, pos.units, explain=[{"text": f"rolled from position {pos.id}", "passed": True}], tag="roll")
        if new is not None:
            new.roll_group = pos.roll_group or pos.id
            new.roll_count = pos.roll_count + 1
            self.entries_today += 1
        else:
            ctx.log("roll_failed", f"roll re-entry rejected: {why}", pos)

    # ------------------------------------------------------------------ entries
    def _units(self, ctx: Context, sel) -> tuple[int, str]:
        s = self.cfg.sizing
        if s.method == "fixed":
            return min(s.contracts, s.max_contracts), ""
        ch, _ = ctx.chain()
        mult = ctx.bt.metas[self.symbol].multiplier
        spot = ctx.price()
        cash = 0.0
        legs = []
        shares = 0.0
        from strata.engine.execution import fill_price
        for x in sel:
            if x.row < 0:
                shares += x.leg.sign * x.leg.quantity
                cash -= x.leg.sign * x.leg.quantity * spot
                continue
            px = fill_price(float(ch.bid[x.row]), float(ch.ask[x.row]), x.leg.sign, ctx.bt.ex)
            q = x.leg.sign * x.leg.quantity
            cash -= q * mult * px
            legs.append(MLeg(bool(ch.is_call[x.row]), float(ch.strike[x.row]), q, mult, float(ch.mid[x.row])))
        R = requirement(legs, shares, spot, ctx.bt.cfg.capital.margin_model)
        bpr = max(R - cash, 1e-9)
        if s.method == "risk_pct":
            single = len({x.expiration for x in sel if x.row >= 0}) <= 1
            _, ml = max_profit_loss(legs, shares, cash) if single else (None, math.inf)
            risk = ml if ml is not None and math.isfinite(ml) and ml > 0 else bpr
        else:
            risk = bpr
        budget = s.pct / 100 * ctx.account.equity() * self.allocation
        units = int(budget // risk) if math.isfinite(risk) and risk > 0 else 0
        units = min(units, s.max_contracts)
        if units < 1:
            return 0, f"sizing: {s.pct:g}% budget {budget:,.0f} < risk per unit {risk:,.0f}"
        return units, ""

    def _expire(self, ctx: Context, why: str, kind: str = "no_price"):
        pd = self.pending
        self.pending = None
        ctx.bt.diag.expired_orders += 1
        ctx.bt.diag.skip(f"order expired unfilled: {why}")
        ctx.bt.diag.skip_signal(ctx.t, pd["signal_t"], kind, why, self.name)
        ctx.log("order_expired", f"entry signal at {cal.fmt_hhmm(pd['minute'])} ET not filled: {why}")

    def _reject(self, ctx: Context, pd: dict, kind: str, why: str, log_kind: str = "entry_skipped"):
        ctx.bt.diag.skip(why if kind != "capital" else "insufficient capital for one unit")
        ctx.bt.diag.skip_signal(ctx.t, pd["signal_t"], kind, why, self.name)
        if len(ctx.bt.events_log) < 20000:
            ctx.log(log_kind, why)

    def _try_fill(self, ctx: Context):
        pd = self.pending
        ex = ctx.bt.ex
        ready_t = pd["signal_t"] + ex.latency_seconds
        ch, si = ctx.chain()
        trades = ctx.bt.trades
        max_age = ctx.bt.max_age
        waited_out = ctx.t - pd["signal_t"] >= ex.max_fill_delay_minutes * 60 or ctx.is_last_tick
        eligible = False
        if ch is not None and si >= 0 and ctx.t >= ready_t:
            snap_t = int(ch.snap_ts[si])
            fresh_now = ctx.t - snap_t <= max_age
            # quotes: the quote must be observed at/after the order reaches the market (or, with
            # no latency, be a still-fresh quote at the signal time).
            # trades: only prints that already exist at the execution time can be used, and
            # only while they are recent (max_trade_age_seconds) - never a later print.
            after_signal = trades or snap_t >= ready_t or (ex.latency_seconds == 0 and pd["signal_t"] - snap_t <= max_age)
            eligible = fresh_now and after_signal
        if not eligible:
            if ch is None:
                self._expire(ctx, f"no option {'trade data' if trades else 'chain'} is stored for {self.symbol} on this session", "no_option_data")
            elif waited_out:
                if trades:
                    self._expire(ctx, f"no option trade printed within {max_age // 60:g} min of the execution time during the {ex.max_fill_delay_minutes:g} min after the signal", "no_recent_trade")
                else:
                    nxt = [int(x) for x in ch.snap_ts if x > ctx.t]
                    detail = f"next option quote is {(nxt[0] - pd['signal_t']) / 60:.0f} min after the signal" if nxt else "no later option quote this session"
                    self._expire(ctx, f"no option quote within {ex.max_fill_delay_minutes:g} min of the signal ({detail})", "no_quote")
            return
        sel, why, _details = ctx.select(self.cfg.legs)
        if sel is None:
            if trades and not waited_out:
                pd["last_reason"] = why  # keep the order working: the contract may print in the next minutes
                return
            self.pending = None
            if trades:
                ctx.bt.diag.expired_orders += 1
                # distinguish "the contract is not in the stored data" from "it is stored but did not print recently"
                absent = "expiration in the stored chain" in why or "contracts for" in why or "stored strike range" in why
                kind = "contract_not_stored" if absent else "no_recent_trade"
                tail = "the required contract is not in the stored trade data for this session" if absent else f"no usable trade print for the required contract within {ex.max_fill_delay_minutes:g} min of the signal"
                self._reject(ctx, pd, kind, f"{why} - {tail}", "order_expired")
            else:
                self._reject(ctx, pd, "no_contract", why)
            return
        self.pending = None
        units, why = self._units(ctx, sel)
        if units < 1:
            self._reject(ctx, pd, "capital", why)
            return
        sel_expl = []
        for x in sel:
            if x.row < 0:
                sel_expl.append({"text": f"{x.leg.id}: {x.leg.side} {x.leg.quantity} shares", "passed": True, "selection": True})
                continue
            sm = x.leg.strike
            target = {"delta": f"|delta| ~ {sm.value:g}", "premium": f"premium ~ {sm.value:g}", "pct_otm": f"{sm.value:g}% OTM",
                      "atm_offset": f"ATM {sm.value:+g} {sm.unit}", "fixed": f"strike {sm.value:g}", "leg_offset": f"{sm.ref_leg} {sm.value:+g} {sm.unit}"}[sm.method]
            sel_expl.append({"text": f"{x.leg.id}: {x.leg.side} {x.leg.right} {x.leg.dte}DTE target {target} -> {x.strike:g}{x.leg.right} exp {x.expiration - ctx.today_num}d, delta {ch.delta[x.row]:+.3f}, " + (f"last trade {ch.mid[x.row]:.2f} printed {int(ctx.t - ch.quote_ts[x.row])}s earlier (no bid/ask exists in trade data)" if ch.trade_based[x.row] else f"bid/ask {ch.bid[x.row]:.2f}/{ch.ask[x.row]:.2f}"), "passed": True, "selection": True})
        delay = ctx.t - pd["signal_t"]
        timing = [{"text": f"signal {cal.fmt_hhmm(pd['minute'])} ET -> filled {cal.fmt_hhmm(ctx.minute)} ET ({delay / 60:.0f} min later, first eligible option quote)", "passed": True, "selection": True}] if delay else []
        pos, why = ctx.open(sel, units, explain=pd["explain"] + timing + sel_expl, meta={"signal_ts": pd["signal_t"], "signal_price": pd.get("signal_price"), "lot": 1, **self._entry_state(ctx), **(pd.get("state") or {})})
        if pos is None:
            self._reject(ctx, pd, "rejected", why, "entry_rejected")
            return
        self.entries_today += 1

    def check_entry(self, ctx: Context):
        if self.pending:
            self._try_fill(ctx)
            return
        e = self.cfg.entry
        if self.entries_today >= e.max_entries_per_day:
            return
        if len([p for p in ctx.positions() if p.meta.get("lot", 1) == 1]) >= e.max_open_positions:
            return  # lots added to an existing position are part of it, not a new position
        if self.last_exit_t is not None and (ctx.t - self.last_exit_t) < e.cooldown_minutes * 60:
            return
        if self.exit_today and "any" not in e.reentry_after:
            cls = REASON_CLASS.get(self.last_exit_reason or "", "other")
            if "none" in e.reentry_after or cls not in e.reentry_after:
                return
        ok, explain = self.entry_signal(ctx)
        if not ok:
            return
        ctx.bt.diag.signals += 1
        self.pending = {"signal_t": ctx.t, "minute": ctx.minute, "explain": explain, "signal_price": ctx.price(), "state": self.tracker.completed if self.tracker else None}
        self._try_fill(ctx)

    def entry_signal(self, ctx: Context) -> tuple[bool, list[dict]]:
        """Does the entry rule fire at this tick? The event sequence (if any) must complete
        at this tick AND the condition tree must hold at this tick. Shared by the backtest
        and the signal scan, so both see exactly the same signals."""
        e = self.cfg.entry
        ctx.seq = self.tracker
        if getattr(self, "coarse", False):
            return False, []
        seq_expl: list[dict] = []
        if self.tracker:
            done, seq_expl, missing = self.tracker.advance(ctx)
            if missing:
                self.na_today |= {f"sequence: missing {m}" for m in missing}
            if not done:
                return False, seq_expl
        ok, explain, na = evaluate(e.conditions, ctx)
        if na:
            self.na_today |= {x.split("[not evaluable: ")[-1].rstrip("]") if "[not evaluable: " in x else x for x in na}
        return ok, seq_expl + explain

    def _entry_state(self, ctx: Context) -> dict:
        """Facts recorded on the position at entry so exit and add rules can refer to them later."""
        ev = X.Evaluator(ctx)
        vwap = ev.name("vwap") if ctx.session_res and ctx.session_res <= 300 else None
        o = ev.name("open")
        return {"entry_vwap": vwap, "entry_session_open": o}

    # ------------------------------------------------------------------ scaling in
    def _same_contracts(self, ctx: Context, root: Position):
        """The first lot's contracts at the current snapshot (None if one has no usable price now)."""
        from strata.engine.selection import Selected
        ch, si = ctx.chain()
        if ch is None or si < 0:
            return None, "no option data at this time"
        specs = {l.id: l for l in self.cfg.legs}
        out = []
        for l in root.legs:
            spec = specs.get(l.leg_id)
            if spec is None:
                return None, f"leg {l.leg_id} is not part of the strategy definition"
            if l.kind == "stock":
                out.append(Selected(spec, -1, 0.0, -1, 0))
                continue
            i = ch.find(si, l.key)
            if i < 0:
                return None, f"{l.label()} has no {'trade print' if ctx.bt.trades else 'quote'} at this time"
            if ctx.t - int(ch.quote_ts[i]) > ctx.bt.max_age or not (math.isfinite(ch.bid[i]) and math.isfinite(ch.ask[i])):
                return None, f"{l.label()} has no recent {'trade print' if ctx.bt.trades else 'quote'}"
            out.append(Selected(spec, int(i), float(ch.strike[i]), int(l.exp), int(l.exp - ctx.today_num)))
        return out, ""

    def check_adds(self, ctx: Context):
        """Scale-in rules: evaluated against the first lot of every open position group."""
        rules = self.cfg.entry.adds
        if not rules:
            return
        acct = ctx.bt.account
        for root in [p for p in ctx.positions() if p.meta.get("lot", 1) == 1 and p.is_open and p.entry_ts != ctx.t]:
            lots = [x for x in list(acct.positions) + list(acct.closed) if x.meta.get("group") == root.id]
            for ri, rule in enumerate(rules):
                made = sum(1 for x in lots if x.meta.get("add_rule") == ri)
                if made >= rule.max_adds:
                    continue
                ev = X.Evaluator(ctx, root)
                res, text = X.explain(rule.when, ev)
                if res is not True:
                    continue
                if rule.contract == "same":
                    sel, why = self._same_contracts(ctx, root)
                else:
                    sel, why, _ = ctx.select(self.cfg.legs)
                key = root.id * 100 + ri
                if sel is None:
                    if key not in self.add_pending:  # report once; keep trying while the rule holds
                        self.add_pending[key] = ctx.t
                        ctx.log("add_deferred", f"add to position {root.id} wanted ({rule.text or rule.when}) but {why}; retrying", root)
                    continue
                lot = len(lots) + 2
                expl = [{"text": f"add #{made + 1} to position {root.id}: {rule.text + ': ' if rule.text else ''}{text}", "passed": True, "add": True}]
                pos, why = ctx.open(sel, rule.quantity, explain=expl, tag="add",
                                    meta={"signal_ts": self.add_pending.pop(key, ctx.t), "signal_price": ctx.price(), "group": root.id, "lot": lot, "add_rule": ri,
                                          "seq": root.meta.get("seq"), "refs": root.meta.get("refs"), **self._entry_state(ctx)})
                if pos is None:
                    ctx.bt.diag.skip(f"add rejected: {why}")
                    ctx.log("add_rejected", f"add to position {root.id} rejected: {why}", root)
                    continue
                root.meta.setdefault("group", root.id)
                lots.append(pos)

    def on_snapshot(self, ctx: Context):
        ctx.seq = self.tracker
        for pos in list(ctx.positions()):
            if pos.entry_ts == ctx.t:
                continue
            self.check_exits(ctx, pos)
        self.check_adds(ctx)
        self.check_entry(ctx)
