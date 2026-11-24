"""Example code strategy on real data. Run:

    strata fetch SPY --options            # free end-of-day SPY chains (once)
    strata backtest examples/python_strategy.py --save

Sells a 30-DTE, 20-delta SPY put credit spread at Monday's close when VIX is below 25; takes
profit at 50% of the credit and closes a week before expiration. It trades at the close because
the free option data is end-of-day: quotes exist at 16:00 ET only.
"""
from strata.sdk import LegSpec, Strategy, StrikeSel


class MondayPutSpread(Strategy):
    name = "SPY Monday 30DTE put spread (python)"
    symbol = "SPY"

    LEGS = [
        LegSpec(id="sp", side="sell", right="P", dte=30, strike=StrikeSel(method="delta", value=0.20)),
        LegSpec(id="lp", side="buy", right="P", dte=30, strike=StrikeSel(method="leg_offset", value=-5, ref_leg="sp")),
    ]

    def on_snapshot(self, ctx):
        if not ctx.is_last_tick:  # act on the closing snapshot only
            return
        for pos in ctx.positions():
            if pos.entry_ts == ctx.t:
                continue
            pnl = ctx.position_pnl_pct(pos)
            if pnl >= 50:
                ctx.close(pos, "profit_target", explain=[{"text": f"P&L {pnl:.0f}% of credit >= 50%", "passed": True}])
            elif pos.front_dte(ctx.today_num) <= 7:
                ctx.close(pos, "dte_exit", explain=[{"text": "7 days to expiration", "passed": True}])
        if ctx.today.weekday() != 0 or len(ctx.positions()) >= 4:
            return
        vix = ctx.feature("vix")
        if vix is None or not (vix < 25):  # no VIX reading: skip rather than assume
            return
        sel, why, _ = ctx.select(self.LEGS)
        if sel:
            ctx.open(sel, units=1, explain=[{"text": f"Monday close, VIX {vix:.1f} < 25", "passed": True}])


STRATEGIES = [MondayPutSpread()]
CONFIG = {"start": "2023-01-03", "end": "2024-12-31",
          "strategies": [{"name": MondayPutSpread.name, "symbol": "SPY", "legs": [l.model_dump() for l in MondayPutSpread.LEGS]}]}
