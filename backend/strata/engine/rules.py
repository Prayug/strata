"""RuleStrategy: executes a declarative StrategyConfig (what the visual builder produces)."""
from __future__ import annotations

import math

import numpy as np

from strata.core import calendar as cal
from strata.engine.conditions import evaluate
from strata.engine.engine import Context, Strategy
from strata.engine.margin import MLeg, max_profit_loss, requirement
from strata.engine.models import StrategyConfig
from strata.engine.portfolio import Position

LOSS_REASONS = {"stop_loss", "trailing_stop", "underlying_stop", "greek_stop", "strike_breach"}
PROFIT_REASONS = {"profit_target"}
PRIORITY = ["stop_loss", "trailing_stop", "strike_breach", "underlying_stop", "greek_stop", "custom_exit", "roll", "time_exit", "dte_exit", "max_hold", "underlying_target", "profit_target"]
REASON_CLASS = {"profit_target": "target", "stop_loss": "stop", "trailing_stop": "stop", "time_exit": "time", "dte_exit": "time", "max_hold": "time"}


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

    def on_day_start(self, ctx: Context):
        self.entries_today = 0
        self.exit_today = False

    def on_position_closed(self, ctx: Context, pos: Position):
        self.last_exit_t = pos.exit_ts
        self.last_exit_reason = pos.exit_reason
        self.exit_today = True

    # ------------------------------------------------------------------ exits
    def _pnl(self, ctx: Context, pos: Position) -> float:
        return pos.gross_pnl(use_mid=self.cfg.exits.trigger_price == "mid")

    def check_exits(self, ctx: Context, pos: Position):
        ex = self.cfg.exits
        if pos.pending_exit:
            ctx.close(pos, pos.pending_exit)
            return
        if any(l.stale for l in pos.legs if l.is_open):
            return  # cannot evaluate price-based rules on stale marks; retried next snapshot
        pnl = self._pnl(ctx, pos)
        pct = pnl / pos.basis * 100
        u = max(pos.units, 1)
        pos.peak_pnl = max(pos.peak_pnl, pnl)
        hits: list[tuple[str, str]] = []
        tgt_thr = stop_thr = None
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
        if ex.time_exit.enabled and ctx.minute >= cal.parse_hhmm(ex.time_exit.time) and (ex.time_exit.dte_lte is None or front <= ex.time_exit.dte_lte):
            hits.append(("time_exit", f"time {cal.fmt_hhmm(ctx.minute)} >= {ex.time_exit.time} with front DTE {front}"))
        if ex.dte_exit.enabled and front <= ex.dte_exit.dte and ctx.minute >= cal.parse_hhmm(ex.dte_exit.time):
            hits.append(("dte_exit", f"front DTE {front} <= {ex.dte_exit.dte}"))
        if ex.max_hold.enabled:
            held_min = (ctx.t - pos.entry_ts) / 60
            if ex.max_hold.minutes and held_min >= ex.max_hold.minutes:
                hits.append(("max_hold", f"held {held_min:.0f} min >= {ex.max_hold.minutes}"))
            if ex.max_hold.days:
                from strata.engine.market import from_day_num
                sessions = cal.trading_days_between(from_day_num(pos.entry_day), ctx.today)
                if sessions >= ex.max_hold.days:
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
                    hits.append(("underlying_stop", f"underlying moved {mv:+.2f}% ({'abs' if neutral else 'against'} {against:.2f}% >= {ue.move_against_pct}%)"))
            if ue.move_for_pct is not None and not neutral and directional >= ue.move_for_pct:
                hits.append(("underlying_target", f"underlying moved {mv:+.2f}% in favour >= {ue.move_for_pct}%"))
            if ue.strike_breach:
                b = ue.breach_buffer_pct / 100
                for l in pos.legs:
                    if l.is_open and l.kind == "option" and l.qty < 0:
                        if (l.is_call and spot > l.strike * (1 + b)) or (not l.is_call and spot < l.strike * (1 - b)):
                            hits.append(("strike_breach", f"underlying {spot:.2f} breached short {l.label()}"))
                            break
        ge = ex.greeks
        if ge.enabled:
            dlt = pos.greeks()[0] / u
            if ge.max_abs_delta is not None and abs(dlt) > ge.max_abs_delta:
                hits.append(("greek_stop", f"|delta|/unit {abs(dlt):.1f} > {ge.max_abs_delta}"))
            if ge.short_leg_delta is not None:
                for l in pos.legs:
                    if l.is_open and l.kind == "option" and l.qty < 0 and math.isfinite(l.greeks[0]) and abs(l.greeks[0]) > ge.short_leg_delta:
                        hits.append(("greek_stop", f"short {l.label()} |delta| {abs(l.greeks[0]):.2f} > {ge.short_leg_delta}"))
                        break
        if ex.custom is not None and ex.custom.children:
            ok, expl = evaluate(ex.custom, ctx, pos)
            if ok:
                hits.append(("custom_exit", "; ".join(e["text"] for e in expl if e.get("passed") and not e.get("group"))))
        rr = ex.roll
        if rr.enabled and pos.roll_count < rr.max_rolls:
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

        # intrabar ambiguity: could the underlying's 1-minute path between snapshots have
        # touched both the stop and the target? (delta-gamma estimate; flag only)
        if tgt_thr is not None and stop_thr is not None and ctx.prev_t < ctx.t:
            ser = ctx.series(pos.symbol)
            hi, lo = ser.range_between(ctx.prev_t, ctx.t)
            s_prev = ser.price_at(ctx.prev_t, ctx.today)
            if math.isfinite(hi) and math.isfinite(s_prev):
                d, g, _, _ = pos.greeks()
                prev_pnl = pnl  # approximation anchor
                est = [prev_pnl + d * (x - spot) + 0.5 * g * (x - spot) ** 2 for x in (hi, lo)]
                if max(est) >= tgt_thr and min(est) <= stop_thr:
                    pos.flags.add("intrabar_ambiguous")

        # leg-level rules (independent of the full-position decision)
        full = [h for h in hits]
        if full:
            reasons = {r for r, _ in full}
            chosen = None
            if reasons & LOSS_REASONS and reasons & PROFIT_REASONS:
                pos.flags.add("ambiguous_exit")
                ctx.bt.diag.ambiguous_exits += 1
            for r in PRIORITY:
                if r in reasons:
                    chosen = r
                    break
            explain = [{"text": txt, "passed": True, "reason": r} for r, txt in full]
            if chosen == "roll":
                self._roll(ctx, pos, explain)
            else:
                ctx.close(pos, chosen, explain=explain)
            return
        for rule in ex.leg_rules:
            idxs = [i for i, l in enumerate(pos.legs) if l.is_open and (l.leg_id == rule.target or l.group == rule.target)]
            if not idxs:
                continue
            for i in list(idxs):
                l = pos.legs[i]
                px = l.mid if self.cfg.exits.trigger_price == "mid" else (l.ask if l.qty < 0 else l.bid)
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

    def check_entry(self, ctx: Context):
        e = self.cfg.entry
        if self.entries_today >= e.max_entries_per_day:
            return
        if len(ctx.positions()) >= e.max_open_positions:
            return
        if self.last_exit_t is not None and (ctx.t - self.last_exit_t) < e.cooldown_minutes * 60:
            return
        if self.exit_today and "any" not in e.reentry_after:
            cls = REASON_CLASS.get(self.last_exit_reason or "", "other")
            if "none" in e.reentry_after or cls not in e.reentry_after:
                return
        ok, explain = evaluate(e.conditions, ctx)
        if not ok:
            return
        ctx.bt.diag.signals += 1
        sel, why, details = ctx.select(self.cfg.legs)
        if sel is None:
            ctx.bt.diag.skip(why)
            if len(ctx.bt.events_log) < 20000:
                ctx.log("entry_skipped", why)
            return
        units, why = self._units(ctx, sel)
        if units < 1:
            ctx.bt.diag.skip("insufficient capital for one unit")
            ctx.log("entry_skipped", why)
            return
        ch, _ = ctx.chain()
        sel_expl = []
        for x in sel:
            if x.row < 0:
                sel_expl.append({"text": f"{x.leg.id}: {x.leg.side} {x.leg.quantity} shares", "passed": True, "selection": True})
                continue
            sm = x.leg.strike
            target = {"delta": f"|delta| ~ {sm.value:g}", "premium": f"premium ~ {sm.value:g}", "pct_otm": f"{sm.value:g}% OTM",
                      "atm_offset": f"ATM {sm.value:+g} {sm.unit}", "fixed": f"strike {sm.value:g}", "leg_offset": f"{sm.ref_leg} {sm.value:+g} {sm.unit}"}[sm.method]
            sel_expl.append({"text": f"{x.leg.id}: {x.leg.side} {x.leg.right} {x.leg.dte}DTE target {target} -> {x.strike:g}{x.leg.right} exp {x.expiration - ctx.today_num}d, delta {ch.delta[x.row]:+.3f}, bid/ask {ch.bid[x.row]:.2f}/{ch.ask[x.row]:.2f}", "passed": True, "selection": True})
        pos, why = ctx.open(sel, units, explain=explain + sel_expl)
        if pos is None:
            ctx.bt.diag.skip(why)
            ctx.log("entry_rejected", why)
            return
        self.entries_today += 1

    def on_snapshot(self, ctx: Context):
        for pos in list(ctx.positions()):
            if pos.entry_ts == ctx.t:
                continue
            self.check_exits(ctx, pos)
        self.check_entry(ctx)
