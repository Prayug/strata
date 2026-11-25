"""Stage one of a two-stage options study: find the sessions where a strategy's entry rules
fire, using the underlying's prices only.

No option data is read. The scan runs the same engine loop, clocks and rule evaluation as a
backtest, so a signal found here is exactly the signal the backtest will see. Its output
tells the dataset planner which sessions (and which underlying price at the decision time)
option observations are actually needed for - nothing is downloaded for other sessions.
"""
from __future__ import annotations

from datetime import date

from strata.data.store import MarketStore
from strata.engine.engine import Backtester, Context
from strata.engine.models import BacktestConfig
from strata.engine.rules import RuleStrategy


class SignalScanner(RuleStrategy):
    """A RuleStrategy that records entry signals instead of placing orders."""

    def on_start(self, ctx: Context):
        self.signals: list[dict] = []
        self._last_signal_t = None

    def on_snapshot(self, ctx: Context):
        e = self.cfg.entry
        if self.entries_today >= e.max_entries_per_day:
            return
        if self._last_signal_t is not None and (ctx.t - self._last_signal_t) < e.cooldown_minutes * 60:
            return
        ok, explain = self.entry_signal(ctx)  # the same rule evaluation as a backtest, sequences included
        if not ok:
            return
        ctx.bt.diag.signals += 1
        self.entries_today += 1
        self._last_signal_t = ctx.t
        self.signals.append({"date": ctx.today, "t": int(ctx.t), "minute": int(ctx.minute), "price": float(ctx.price()), "strategy": self.idx, "symbol": self.symbol,
                             "resolution": int(ctx.session_res), "why": [x["text"] for x in explain if not x.get("group")][:6]})


def scan_signals(cfg: BacktestConfig | dict, store: MarketStore | None = None, progress=None, should_cancel=None) -> dict:
    """Entry signals of every strategy in `cfg` over its window.

    Returns signals (date, time, underlying price at the signal - all point-in-time), the
    sessions that could not be evaluated (with the reason; never counted as "no signal") and
    the number of sessions evaluated. Position limits are not simulated: a backtest may take
    fewer trades than there are signals, never more."""
    if isinstance(cfg, dict):
        cfg = BacktestConfig.model_validate(cfg)
    strategies = [SignalScanner(s, i) for i, s in enumerate(cfg.strategies)]
    bt = Backtester(cfg, store, strategies, progress=progress, should_cancel=should_cancel, record_paths=False, signals_only=True)
    bt.run()
    signals = sorted((s for st in strategies for s in st.signals), key=lambda s: (s["t"], s["strategy"]))
    evaluated = [r["date"] for r in bt.daily]
    excluded = bt.diag.excluded_sessions
    ex_dates = {d for v in excluded.values() for d in v}
    by_res: dict[str, int] = {}
    for sym in bt.opt_syms:
        for k, v in getattr(bt.series.get(sym), "providers", {}).items():
            by_res[k] = by_res.get(k, 0) + v
    return {"signals": signals, "sessions": len(evaluated), "sessions_evaluable": len([d for d in evaluated if d.isoformat() not in ex_dates]),
            "first": evaluated[0] if evaluated else None, "last": evaluated[-1] if evaluated else None,
            "excluded": excluded, "warnings": bt.diag.warnings, "underlying_sessions": by_res, "dataset_versions": bt.dataset_versions}


def signal_dates(scan: dict) -> list[date]:
    return sorted({s["date"] for s in scan["signals"]})
