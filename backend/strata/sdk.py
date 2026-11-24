"""Python strategy interface for users who prefer code over the visual builder.

    from strata.sdk import Strategy, LegSpec, StrikeSel, run_backtest

    class MyStrategy(Strategy):
        name, symbol = "my put spread", "SPY"

        def on_snapshot(self, ctx):
            for pos in ctx.positions():                      # manage open positions
                if ctx.position_pnl_pct(pos) >= 50:
                    ctx.close(pos, "profit_target")
            if ctx.minute == 10 * 60 and not ctx.positions():
                sel, why, _ = ctx.select([
                    LegSpec(id="sp", side="sell", right="P", dte=0, strike=StrikeSel(method="delta", value=0.15)),
                    LegSpec(id="lp", side="buy", right="P", dte=0, strike=StrikeSel(method="leg_offset", value=-5, ref_leg="sp")),
                ])
                if sel:
                    ctx.open(sel, units=1, explain=[{"text": "10:00 entry", "passed": True}])

Everything the context exposes is point-in-time: ctx.price(), ctx.indicator(), ctx.feature(),
ctx.chain() and ctx.iv_rank() only see data observable at ctx.t. Orders go through the
same execution, margin and settlement models as the visual builder's strategies.
"""
from strata.engine.engine import BacktestResult, Context, Strategy, run_backtest
from strata.engine.models import BacktestConfig, Capital, CondNode, Execution, LegSpec, StrategyConfig, StrikeSel

__all__ = ["Strategy", "Context", "run_backtest", "BacktestResult", "BacktestConfig", "Capital", "Execution", "LegSpec", "StrikeSel", "StrategyConfig", "CondNode"]
