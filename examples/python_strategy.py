"""Example code strategy. Run:  strata backtest examples/python_strategy.py --save"""
from strata.sdk import LegSpec, Strategy, StrikeSel


class RsiPutSpread(Strategy):
    """Sell a 0DTE 15-delta put spread at 10:30 when 5-minute RSI is above 50 and the
    session's VIX is below 25; take profit at 50% of credit, stop at 2x credit, flat by 15:45."""

    name = "RSI-filtered 0DTE put spread (python)"
    symbol = "DEMO"

    LEGS = [
        LegSpec(id="sp", side="sell", right="P", dte=0, strike=StrikeSel(method="delta", value=0.15)),
        LegSpec(id="lp", side="buy", right="P", dte=0, strike=StrikeSel(method="leg_offset", value=-5, ref_leg="sp")),
    ]

    def on_snapshot(self, ctx):
        for pos in ctx.positions():
            if pos.entry_ts == ctx.t:
                continue
            pnl = ctx.position_pnl_pct(pos)
            if pnl >= 50:
                ctx.close(pos, "profit_target", explain=[{"text": f"P&L {pnl:.0f}% >= 50%", "passed": True}])
            elif pnl <= -200:
                ctx.close(pos, "stop_loss", explain=[{"text": f"P&L {pnl:.0f}% <= -200%", "passed": True}])
            elif ctx.minute >= 15 * 60 + 45:
                ctx.close(pos, "time_exit")
        if ctx.minute != 10 * 60 + 30 or ctx.positions():
            return
        rsi = ctx.indicator(self.symbol, "rsi", {"period": 14}, "5m")
        vix = ctx.feature("vix")
        if not (rsi > 50 and vix < 25):
            return
        sel, why, _ = ctx.select(self.LEGS)
        if sel:
            ctx.open(sel, units=1, explain=[{"text": f"RSI(14,5m)={rsi:.1f} > 50, VIX={vix:.1f} < 25", "passed": True}])


STRATEGIES = [RsiPutSpread()]
CONFIG = {"start": "2023-03-01", "end": "2024-12-31", "strategies": [{"name": RsiPutSpread.name, "symbol": "DEMO", "legs": [l.model_dump() for l in RsiPutSpread.LEGS]}]}
