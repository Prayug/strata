"""Why did the strategy enter on this day - or not?

Replays one historical session through the same rule evaluation the engine uses and
records, tick by tick, what every condition (and every step of an event sequence)
evaluated to and with which numbers. Used by the interpretation preview ("how will each
condition be evaluated against an actual session?") and by the calendar's day view.

Nothing here simulates a trade: fills, exits and prices shown next to it come from the
stored result of the run being inspected.
"""
from __future__ import annotations

from datetime import date, timedelta

from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine.conditions import evaluate, walk_leaves
from strata.engine.engine import Backtester, Context, DataUnavailable
from strata.engine.models import BacktestConfig
from strata.engine.rules import RuleStrategy


class _Recorder(RuleStrategy):
    """Evaluates the entry rule at every tick and keeps the explanations."""

    def on_start(self, ctx: Context):
        self.ticks: list[dict] = []
        self.signal: dict | None = None

    def on_snapshot(self, ctx: Context):
        ctx.seq = self.tracker
        if self.signal is not None and self.entries_today >= self.cfg.entry.max_entries_per_day:
            return
        seq_before = self.tracker.i if self.tracker else 0
        ok, explain = self.entry_signal(ctx)
        # the condition tree is evaluated even while a sequence is still waiting, so the view can show every rule's state
        tree_ok, tree_expl, na = (ok, [x for x in explain if not x.get("stage") and not x.get("sequence")], []) if (ok or not self.tracker) else evaluate(self.cfg.entry.conditions, ctx)
        items = [{"id": x.get("id"), "text": x["text"], "passed": x.get("passed"), "kind": "stage" if x.get("stage") else "sequence" if x.get("sequence") else ("group" if x.get("group") else "condition")}
                 for x in ([e for e in explain if e.get("stage") or e.get("sequence")] + tree_expl)]
        self.ticks.append({"t": int(ctx.t), "minute": int(ctx.minute), "price": float(ctx.price()), "passed": bool(ok), "tree": bool(tree_ok), "na": bool(na), "items": items,
                           "stage": self.tracker.i if self.tracker else None, "stage_advanced": bool(self.tracker and (self.tracker.i != seq_before or ok))})
        if ok and self.signal is None:
            self.signal = {"t": int(ctx.t), "minute": int(ctx.minute), "price": float(ctx.price())}
            self.entries_today += 1


def explain_day(cfg: dict | BacktestConfig, d: date, store: MarketStore | None = None) -> dict:
    """Tick-by-tick evaluation of a strategy's entry rule on session `d`."""
    store = store or MarketStore()
    if isinstance(cfg, dict):
        cfg = BacktestConfig.model_validate({**cfg, "start": d.isoformat(), "end": d.isoformat()})
    else:
        cfg = cfg.model_copy(update={"start": d, "end": d})
    if not cal.is_trading_day(d):
        return {"date": d.isoformat(), "trading_day": False, "summary": f"{d} is not a trading session.", "ticks": [], "rules": []}
    rec = _Recorder(cfg.strategies[0], 0)
    bt = Backtester(cfg, store, [rec], record_paths=False, signals_only=True)
    try:
        bt.run()
    except DataUnavailable as e:
        return {"date": d.isoformat(), "trading_day": True, "summary": str(e), "ticks": [], "rules": [], "error": str(e)}
    ticks = rec.ticks
    excluded = [k for k, v in bt.diag.excluded_sessions.items() if d.isoformat() in v]
    ser = bt.series.get(cfg.strategies[0].symbol)
    res = ser.resolution(d) if ser is not None else 0
    if not ticks:
        return {"date": d.isoformat(), "trading_day": True, "summary": f"No prices are stored for {cfg.strategies[0].symbol} on {d}, so the rule could not be evaluated. That is missing data, not a failed condition.",
                "ticks": [], "rules": [], "excluded": excluded, "resolution": None}
    # one row per rule: when it first held, how often, and its value at the decisive tick
    order: list[str] = []
    by: dict[str, dict] = {}
    key_tick = next((t for t in ticks if t["passed"]), None)
    for t in ticks:
        for it in t["items"]:
            if it["kind"] == "group":
                continue
            k = (it.get("id") or "") + "|" + _stem(it["text"]) if it["kind"] == "condition" else it["kind"] + "|" + _stem(it["text"])
            if k not in by:
                by[k] = {"kind": it["kind"], "label": _stem(it["text"]), "true_ticks": 0, "ticks": 0, "first_true": None, "last_text": it["text"], "na_ticks": 0, "at_signal": None, "states": []}
                order.append(k)
            b = by[k]
            b["ticks"] += 1
            b["last_text"] = it["text"]
            if it["passed"] is True:
                b["true_ticks"] += 1
                if b["first_true"] is None:
                    b["first_true"] = t["minute"]
            elif it["passed"] is None:
                b["na_ticks"] += 1
            if not b["states"] or b["states"][-1][1] != it["passed"]:
                b["states"].append([t["minute"], it["passed"], it["text"]])
            if key_tick is not None and t is key_tick:
                b["at_signal"] = {"passed": it["passed"], "text": it["text"]}
    rules = []
    for k in order:
        b = by[k]
        rules.append({"kind": b["kind"], "label": b["label"], "held_ticks": b["true_ticks"], "ticks": b["ticks"], "first_true": cal.fmt_hhmm(b["first_true"]) if b["first_true"] is not None else None,
                      "not_evaluable_ticks": b["na_ticks"], "at_signal": b["at_signal"], "last": b["last_text"], "changes": [{"time": cal.fmt_hhmm(m), "passed": p, "text": txt} for m, p, txt in b["states"][:24]]})
    sig = rec.signal
    sym = cfg.strategies[0].symbol
    if sig:
        summary = f"The entry rule fired at {cal.fmt_hhmm(sig['minute'])} ET with {sym} at {sig['price']:.2f}."
    elif excluded:
        summary = f"The rule could not be evaluated on this session ({excluded[0]}). It is excluded, not counted as a session without a signal."
    else:
        never = [r for r in rules if r["kind"] == "condition" and r["held_ticks"] == 0]
        waiting = [r for r in rules if r["kind"] == "stage" and r["held_ticks"] == 0]
        if waiting:
            summary = f"No entry: the sequence never got past \"{waiting[0]['label']}\"."
        elif never:
            summary = "No entry: " + "; ".join(f"\"{r['label']}\" was never true" for r in never[:3]) + "."
        else:
            summary = "No entry: the rules were each true at some point, but never all at the same time."
    slim = []
    prev = None
    for t in ticks:  # keep the ticks where something changed, plus the first, the signal and the last
        state = (t["passed"], t["tree"], t["stage"], tuple(i["passed"] for i in t["items"]))
        if state != prev or t is ticks[-1] or (sig and t["t"] == sig["t"]):
            slim.append({"time": cal.fmt_hhmm(t["minute"]), "t": t["t"], "price": round(t["price"], 4), "passed": t["passed"], "items": t["items"]})
        prev = state
    return {"date": d.isoformat(), "trading_day": True, "symbol": sym, "resolution": "daily" if res >= 86400 else f"{res // 60}-minute", "evaluations": len(ticks), "signal": sig and {**sig, "time": cal.fmt_hhmm(sig["minute"])},
            "summary": summary, "rules": rules, "ticks": slim[:120], "excluded": excluded, "sequence": [s.name for s in cfg.strategies[0].entry.sequence],
            "stages": {k: {**v, "time": cal.fmt_hhmm(v["minute"])} for k, v in ((rec.tracker.completed or {}).get("seq") or rec.tracker.state).items()} if rec.tracker else {}}


def _stem(text: str) -> str:
    """The part of an explanation that names the rule, without the values."""
    import re
    t = re.sub(r"=\s*-?[\d.]+(?:e-?\d+)?", "", text)
    t = re.sub(r"\(not yet determined\)|\[not evaluable.*?\]", "", t)
    t = re.sub(r"waiting for stage \d+/\d+ '([^']*)'.*", r"\1", t)
    t = re.sub(r"stage \d+/\d+ '([^']*)' happened.*", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip(" :;")
    return t[:140]


def study_as_strategy(study: dict) -> dict:
    """A market study's condition as an entry rule, so its sessions can be explained the same way."""
    kids = list((study.get("condition") or {}).get("children") or [])
    et = study.get("entry_time", "open")
    if et not in ("signal",):
        kids = [{"id": "t", "kind": "cond", "type": "entry_times", "params": {"times": et}}] + kids
    return {"strategies": [{"name": study.get("label") or "study", "symbol": study["symbol"],
                            "legs": [{"id": "x", "side": "buy", "right": "C", "dte": 0, "strike": {"method": "atm_offset", "value": 0, "unit": "points"}}],
                            "entry": {"conditions": {"id": "root", "kind": "group", "op": "and", "children": kids}, "sequence": study.get("sequence") or []}}],
            "execution": {"signal_interval_minutes": 1}}


def sample_dates(store: MarketStore, symbol: str, signals: list[str], n: int = 3) -> dict:
    """A recent qualifying session and a recent non-qualifying one to preview a rule on."""
    daily = store.read_daily(symbol)
    days = [x.isoformat() for x in daily["date"].to_list()[-260:]] if not daily.is_empty() else []
    sig = set(signals)
    return {"qualifying": [d for d in reversed(days) if d in sig][:n] or sorted(sig)[-n:][::-1], "not_qualifying": [d for d in reversed(days) if d not in sig][:n]}


def nearest_session(d: date) -> date:
    while not cal.is_trading_day(d):
        d -= timedelta(days=1)
    return d
