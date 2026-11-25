"""Event sequences: entry rules that depend on things happening in order.

    "Wait until SPY drops 0.5% from the open, then buy when it recovers half of that decline"

    stages:  drop     from_open_pct <= -0.5
             recover  price >= low_since("drop") + 0.5 * (open - low_since("drop"))

The tracker holds today's state: which stage is next, and for each stage that has happened
the time and underlying price at which it happened plus any reference prices it captured.
Stages are evaluated point-in-time on the engine's signal clock, one stage per tick, each
on a strictly later tick than the previous one - so "then" always means "afterwards". The
state resets at the start of every session. When the last stage happens the sequence is
complete (the entry signal) and it starts over, so a later re-entry needs the whole
sequence again.
"""
from __future__ import annotations

from strata.core.calendar import fmt_hhmm
from strata.engine.expr import Evaluator, explain
from strata.engine.models import SeqStage


class SequenceTracker:
    def __init__(self, stages: list[SeqStage]):
        self.stages = list(stages)
        self.reset()

    def reset(self):
        self.i = 0
        self.state: dict[str, dict] = {}
        self.refs: dict[str, float] = {}
        self.last_t: int | None = None
        self.completed: dict | None = None  # snapshot of the state at the last completion

    def __bool__(self) -> bool:
        return bool(self.stages)

    def snapshot(self) -> dict:
        return {"seq": {k: dict(v) for k, v in self.state.items()}, "refs": dict(self.refs)}

    def advance(self, ctx) -> tuple[bool, list[dict], set[str]]:
        """Evaluate the next stage at this tick.
        Returns (sequence completed at this tick, explanations, inputs that were missing)."""
        if not self.stages:
            return True, [], set()
        out: list[dict] = []
        missing: set[str] = set()
        ctx.seq = self
        st = self.stages[self.i]
        if self.i > 0 and st.within_minutes and self.last_t is not None and ctx.t - self.last_t > st.within_minutes * 60:
            prev = self.stages[self.i - 1].name
            out.append({"text": f"sequence restarted: '{st.name}' did not happen within {st.within_minutes} min of '{prev}'", "passed": False, "stage": st.name})
            self.i, self.state, self.refs, self.last_t = 0, {}, {}, None
            st = self.stages[0]
        if self.last_t is not None and ctx.t <= self.last_t:
            return False, out, missing  # the next stage can only happen on a later tick
        ev = Evaluator(ctx)
        res, text = explain(st.when, ev)
        if res is None and ev.missing:
            missing |= ev.missing
        label = st.text or st.name
        if res is not True:
            done = ", ".join(f"{n} at {fmt_hhmm(v['minute'])}" for n, v in self.state.items())
            out.append({"text": f"waiting for stage {self.i + 1}/{len(self.stages)} '{label}': {text}" + (f" (so far: {done})" if done else ""), "passed": False, "stage": st.name})
            return False, out, missing
        price = ctx.price()
        if self.i == 0:
            self.state, self.refs = {}, {}  # a new sequence begins: forget the previous one
        self.state[st.name] = {"t": int(ctx.t), "minute": int(ctx.minute), "price": float(price)}
        for name, src in st.capture.items():
            v = ev.eval(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.refs[name] = float(v)
        self.last_t = int(ctx.t)
        self.i += 1
        out.append({"text": f"stage {self.i}/{len(self.stages)} '{label}' happened at {fmt_hhmm(ctx.minute)} ET (price {price:.2f}): {text}", "passed": True, "stage": st.name})
        if self.i < len(self.stages):
            return False, out, missing
        steps = " -> ".join("%s %s @ %.2f" % (n, fmt_hhmm(v["minute"]), v["price"]) for n, v in self.state.items())
        hist = [{"text": f"sequence: {steps}", "passed": True, "sequence": True}]
        self.completed = self.snapshot()
        self.i, self.last_t = 0, None  # a further entry needs the whole sequence again; the state stays readable until the next first stage
        return True, hist + out, missing
