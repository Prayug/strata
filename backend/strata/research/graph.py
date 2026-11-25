"""Composable research graphs.

A research question is represented as a small directed acyclic graph of typed primitives,
not as one of a fixed list of question templates and never as generated code:

    data -> feature -> filter -> combine ----------.
    data -> forward_return ------------------------+-> conditional_probability
                                                   +-> matched_comparison
                                                   +-> robustness / validation / sweep ...

Each primitive declares its inputs, its output type, its parameters, the data it needs and
what evaluating it means (`PRIMITIVES`). A graph is plain JSON, so a language model, the
deterministic question parser, a person editing JSON, and a saved template all produce the
same thing, and `validate` decides whether it may run:

  * structure    known primitives, existing inputs, no cycles, bounded size
  * types        every port receives the type it declares
  * parameters   checked against the primitive's schema; expressions are parsed by the
                 strategy rule language, so nothing outside that language can be evaluated
  * point in time  every series, condition and control carries the instant it becomes
                 known ("open", a clock time, "close"). A condition that is known later
                 than the outcome starts is refused: it would use the future

Execution is deterministic and reuses the engines that already exist: the vectorised
session-feature table for values read at a fixed tick, the rule engine for free-form
expressions, the event-study engine for intraday triggers and sequences, the shared
evidence function for every frequency and interval, and the research ledger for trials and
the final test period. This module adds no second way to compute any of them.
"""
from __future__ import annotations

import copy
import re
import math
from dataclasses import dataclass, field

import numpy as np

from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine import expr as X
from strata.research import evidence as EV
from strata.research import ledger as LG
from strata.research import matching as MT
from strata.research import robust as R
from strata.research.features import A, CODE_NAMES, FEATURES, ExprPred, OutcomeSpec, Pred, SessionFeatures, T, feature_def, get_features, to_study

VERSION = 1
MAX_NODES, MAX_SWEEP, MAX_SESSIONS_OUT = 48, 24, 3000
OPS_CMP = (">=", "<=", ">", "<", "==", "abs>=", "between")


class GraphError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# ----------------------------------------------------------------------------- primitives
def _p(type_, doc, required=False, default=None, **kw):
    return {"type": type_, "doc": doc, "required": required, "default": default, **kw}


@dataclass(frozen=True)
class Prim:
    op: str
    title: str
    category: str
    inputs: dict  # port -> type; "name?" optional, "name*" a list
    output: str
    params: dict
    semantics: str
    requires: str


PRIMITIVES: dict[str, Prim] = {p.op: p for p in [
    Prim("data", "Historical data", "Data", {}, "dataset",
         {"symbol": _p("symbol", "Underlying to study", True), "start": _p("date", "First session (default: everything stored)"), "end": _p("date", "Last session"),
          "cross": _p("symbols", "Other symbols whose values conditions may read; found automatically from the expressions")},
         "Loads every stored session of the symbol with the finest bars available per session. Nothing is downloaded here.",
         "Daily bars for the symbol; intraday bars for any node that reads a clock time."),
    Prim("feature", "Feature", "Features", {"data": "dataset"}, "series",
         {"name": _p("feature", "A catalog feature (vectorised)"), "expr": _p("expr", "Or any numeric rule-language expression"), "at": _p("tick", "When it is read: open, close or HH:MM New York", default="open"),
          "label": _p("string", "How to name it in the report")},
         "One value per session, as it could be read at the tick. Catalog features about completed sessions are known from the open.",
         "Daily bars; intraday bars at the tick for expressions that read the session so far."),
    Prim("filter", "Condition", "Conditions", {"x?": "series", "data?": "dataset"}, "mask",
         {"op": _p("enum", "Comparison for a series input", enum=list(OPS_CMP)), "value": _p("number", "Threshold"), "hi": _p("number", "Upper bound for between"),
          "expr": _p("expr", "Or a boolean rule-language expression read at `at` (needs the data input)"), "at": _p("tick", "Tick for the expression form", default="open"), "label": _p("string", "Wording")},
         "True, false or unknown per session. Unknown (missing input) is never treated as false.",
         "Whatever its series or expression needs."),
    Prim("combine", "Combine conditions", "Conditions", {"masks*": "mask"}, "mask",
         {"how": _p("enum", "and | or | not", default="and", enum=["and", "or", "not"])},
         "Three-valued logic: a combination is unknown only when no known part already decides it.", "None beyond its inputs."),
    Prim("event", "Intraday event", "Events", {"data": "dataset"}, "anchor",
         {"expr": _p("expr", "Boolean rule; the event is the first tick on which it is true", True), "label": _p("string", "Wording")},
         "Scans each session tick by tick and anchors the observation at the first tick on which the rule holds.",
         "Intraday bars (daily bars cannot place an event inside the session)."),
    Prim("sequence", "Temporal sequence", "Events", {"data": "dataset"}, "anchor",
         {"steps": _p("steps", "Ordered stages: [{name, when, within_minutes?}]", True), "label": _p("string", "Wording")},
         "Stages must happen in order on successive ticks; the observation is anchored when the last one happens.",
         "5-minute or finer bars."),
    Prim("forward_return", "Forward return", "Outcomes", {"data": "dataset", "anchor?": "anchor"}, "outcome",
         {"entry": _p("tick", "Where the observation starts: open, close, HH:MM (ignored with an anchor)", default="open"), "horizon": _p("horizon", "session | sessions:N | time:HH:MM", default="session"),
          "direction": _p("enum", "up | down", default="up", enum=["up", "down"]), "threshold": _p("number", "Counts as a hit when the horizon ends this many % beyond the entry price", default=0.0)},
         "Return from the entry instant to the end of the horizon. A hit is a finish beyond the entry price (plus the threshold) in the stated direction.",
         "Bars from the entry to the horizon end."),
    Prim("target_stop", "Target before stop", "Outcomes", {"data": "dataset", "anchor?": "anchor"}, "outcome",
         {"entry": _p("tick", "Where the observation starts", default="open"), "horizon": _p("horizon", "session | sessions:N | time:HH:MM", default="session"),
          "direction": _p("enum", "up | down | either", default="up", enum=["up", "down", "either"]), "target": _p("number", "% move that counts as the target", True), "stop": _p("number", "% adverse move that ends the observation")},
         "Walks the bars after the entry. When target and stop fall inside one bar the order is unknowable and the observation is counted as ambiguous, not guessed.",
         "Bars from the entry to the horizon end; finer bars leave fewer ambiguous observations."),
    Prim("conditional_probability", "Conditional probability", "Analysis", {"outcome": "outcome", "condition?": "mask", "by*": "series"}, "result",
         {"min_sample": _p("number", "Observations needed before anything is concluded", default=30)},
         "How often the outcome happens when the condition holds, against every other evaluated session, with week-resampled intervals. `by` splits the qualifying sessions into terciles of each series.",
         "None beyond its inputs."),
    Prim("matched_comparison", "Matched comparison", "Analysis", {"outcome": "outcome", "condition": "mask", "controls*": "series"}, "result",
         {"k": _p("number", "Comparable sessions per qualifying session", default=5), "caliper": _p("number", "Largest allowed distance, in standard deviations per control", default=0.5)},
         "Compares qualifying sessions with non-qualifying sessions that looked the same on the controls (stratified, nearest-neighbour and regression estimates). An association, not a causal effect.",
         "Every control must be known at the decision tick."),
    Prim("distribution", "Distribution", "Analysis", {"outcome": "outcome", "condition?": "mask"}, "result",
         {"paths": _p("bool", "Also summarise the price paths that follow", default=True), "horizon_sessions": _p("number", "Sessions of path to follow for daily paths", default=10)},
         "Quantiles and histogram of end-of-horizon returns and excursions for qualifying sessions against the rest, and the paths that followed.",
         "None beyond its inputs."),
    Prim("correlation", "Correlation", "Analysis", {"x": "series", "y": "series|outcome", "condition?": "mask"}, "result",
         {"method": _p("enum", "pearson | spearman", default="spearman", enum=["pearson", "spearman"])},
         "Association between two per-session values with a block-bootstrap interval. With an outcome as y, x must be known before the outcome starts.",
         "None beyond its inputs."),
    Prim("regression", "Regression", "Analysis", {"y": "outcome|series", "xs*": "series", "condition?": "mask"}, "result",
         {"target": _p("enum", "hit (0/1) or return (%)", default="return", enum=["hit", "return"])},
         "Least squares with calendar-week clustered errors; regressors are standardised so coefficients are comparable. A linear probability model when the target is the hit flag.",
         "Every regressor must be known before the outcome starts."),
    Prim("sweep", "Parameter sweep", "Analysis", {"result": "result"}, "result",
         {"node": _p("node", "The node whose parameter varies", True), "param": _p("string", "Parameter name (e.g. value, entry, symbol)", True), "values": _p("list", "Values to try", True)},
         "Re-evaluates the result once per value. Every value is a recorded trial and p-values are corrected for all of them.",
         "The same as the result it repeats, for every value."),
    Prim("robustness", "Robustness", "Validation", {"outcome": "outcome", "condition": "mask"}, "result",
         {}, "Sub-periods, calendar years, each clause removed in turn, and neighbouring thresholds. Reports where the finding breaks rather than a single score.",
         "None beyond its inputs."),
    Prim("validation", "Statistical validation", "Validation", {"outcome": "outcome", "condition": "mask"}, "result",
         {"alpha": _p("number", "Significance level after correction", default=0.05)},
         "Train and validation periods decide whether the idea is a candidate; only then is the final test period scored, once. The ledger records that it was used.",
         "A final test period the research session has not looked at."),
    Prim("options_backtest", "Option strategies", "Strategy", {"outcome": "outcome", "condition": "mask"}, "result",
         {"max_variants": _p("number", "How many compatible structures to draft", default=6)},
         "Drafts option positions that express the finding and reports which signals have observed option prices. The backtests themselves run in the strategy engine on observed premiums.",
         "Observed option quotes or trade prints at the entry instant; nothing is modelled."),
]}
CATEGORIES = ["Data", "Features", "Conditions", "Events", "Outcomes", "Analysis", "Validation", "Strategy"]


def primitives() -> list[dict]:
    """The primitive catalog for the interface and for the language model's instructions."""
    return [{"op": p.op, "title": p.title, "category": p.category, "inputs": p.inputs, "output": p.output, "params": p.params, "semantics": p.semantics, "requires": p.requires} for p in PRIMITIVES.values()]


# ----------------------------------------------------------------------------- ticks
def tick_order(at: str) -> float:
    """When a value becomes known inside its session (minutes since midnight, New York)."""
    if at == "open":
        return 570.0
    if at == "close":
        return 961.0
    if at == "event":
        return 960.5  # an intraday trigger: somewhere inside the session
    return float(cal.parse_hhmm(at))


def _tick_ok(at) -> bool:
    if at in ("open", "close"):
        return True
    try:
        m = cal.parse_hhmm(str(at))
    except Exception:  # noqa: BLE001
        return False
    return 570 < m < 960


def _horizon_ok(h) -> bool:
    if h == "session":
        return True
    if isinstance(h, str) and h.startswith("sessions:"):
        return h.split(":")[1].isdigit() and 1 <= int(h.split(":")[1]) <= 60
    if isinstance(h, str) and h.startswith("time:"):
        return _tick_ok(h.split(":", 1)[1])
    return False


def _later(a: str, b: str) -> str:
    return a if tick_order(a) >= tick_order(b) else b


# ----------------------------------------------------------------------------- validation
def _ports(prim: Prim) -> dict[str, tuple[str, bool, bool]]:
    out = {}
    for k, t in prim.inputs.items():
        out[k.rstrip("?*")] = (t, k.endswith("?") or k.endswith("*"), k.endswith("*"))
    return out


def _order(nodes: dict[str, dict]) -> list[str]:
    """Topological order; raises GraphError on a cycle."""
    seen: dict[str, int] = {}
    out: list[str] = []

    def visit(i: str, path: tuple):
        if seen.get(i) == 2:
            return
        if seen.get(i) == 1:
            raise GraphError([f"the graph has a cycle through {' -> '.join(path + (i,))}"])
        seen[i] = 1
        for ref in _refs(nodes[i]):
            if ref in nodes:
                visit(ref, path + (i,))
        seen[i] = 2
        out.append(i)

    for i in nodes:
        visit(i, ())
    return out


def _refs(node: dict) -> list[str]:
    out = []
    for v in (node.get("inputs") or {}).values():
        out += v if isinstance(v, list) else [v]
    if node.get("op") == "sweep" and (node.get("params") or {}).get("node"):
        out.append(node["params"]["node"])
    return [x for x in out if isinstance(x, str)]


def _check_param(name: str, spec: dict, v, errs: list[str], where: str) -> None:
    t = spec["type"]
    if t == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errs.append(f"{where}: {name} must be a number")
    elif t == "enum":
        if v not in spec["enum"]:
            errs.append(f"{where}: {name} must be one of {', '.join(spec['enum'])}")
    elif t == "tick":
        if not _tick_ok(v):
            errs.append(f"{where}: {name} must be open, close or a New York time between 09:31 and 15:59 (got {v!r})")
    elif t == "horizon":
        if not _horizon_ok(v):
            errs.append(f"{where}: {name} must be session, sessions:N (1-60) or time:HH:MM (got {v!r})")
    elif t == "expr":
        if not isinstance(v, str) or not v.strip():
            errs.append(f"{where}: {name} must be a rule-language expression")
        else:
            bad = X.validate(v)
            if bad:
                errs.append(f"{where}: {name} is not a valid expression: {bad[0]}")
            elif X.uses_position(v):
                errs.append(f"{where}: {name} reads an open position (entry.*, position.*), which does not exist in a market study")
    elif t == "feature":
        try:
            feature_def(str(v), "SPY")
        except KeyError:
            errs.append(f"{where}: unknown feature {v!r}")
    elif t == "symbol":
        if not isinstance(v, str) or not X.is_ticker(v.upper()):
            errs.append(f"{where}: {name} must be a ticker symbol")
    elif t == "symbols":
        if not isinstance(v, list) or not all(isinstance(x, str) and X.is_ticker(x.upper()) for x in v):
            errs.append(f"{where}: {name} must be a list of ticker symbols")
    elif t == "list":
        if not isinstance(v, list) or not v:
            errs.append(f"{where}: {name} must be a non-empty list")
        elif len(v) > MAX_SWEEP:
            errs.append(f"{where}: {name} has {len(v)} values; the limit is {MAX_SWEEP}")
    elif t == "bool":
        if not isinstance(v, bool):
            errs.append(f"{where}: {name} must be true or false")
    elif t == "date":
        try:
            from datetime import date
            date.fromisoformat(str(v))
        except ValueError:
            errs.append(f"{where}: {name} must be a date (YYYY-MM-DD)")
    elif t == "steps":
        if not isinstance(v, list) or not 1 <= len(v) <= 6:
            errs.append(f"{where}: {name} must list 1 to 6 stages")
        else:
            for i, st in enumerate(v):
                if not isinstance(st, dict) or not st.get("when") or not st.get("name"):
                    errs.append(f"{where}: stage {i + 1} needs a name and a `when` expression")
                    continue
                bad = X.validate(str(st["when"]))
                if bad:
                    errs.append(f"{where}: stage {st.get('name')!r} is not a valid expression: {bad[0]}")


def validate(graph: dict) -> dict:
    """Static checks. Returns {"errors": [...], "order": [...], "types": {id: type}, "known_at": {id: tick}}."""
    errs: list[str] = []
    raw = graph.get("nodes")
    if not isinstance(raw, list) or not raw:
        return {"errors": ["the graph has no nodes"], "order": [], "types": {}, "known_at": {}}
    if len(raw) > MAX_NODES:
        errs.append(f"the graph has {len(raw)} nodes; the limit is {MAX_NODES}")
    nodes: dict[str, dict] = {}
    for n in raw:
        if not isinstance(n, dict) or not isinstance(n.get("id"), str) or not n["id"]:
            errs.append("every node needs an id")
            continue
        if n["id"] in nodes:
            errs.append(f"duplicate node id {n['id']!r}")
            continue
        if n.get("op") not in PRIMITIVES:
            errs.append(f"{n['id']}: unknown primitive {n.get('op')!r} (available: {', '.join(PRIMITIVES)})")
            continue
        nodes[n["id"]] = n
    if errs:
        return {"errors": errs, "order": [], "types": {}, "known_at": {}}
    for i, n in nodes.items():
        prim = PRIMITIVES[n["op"]]
        ports = _ports(prim)
        ins = n.get("inputs") or {}
        for k in ins:
            if k not in ports:
                errs.append(f"{i}: {prim.op} has no input named {k!r} (it takes {', '.join(ports) or 'none'})")
        for k, (t, optional, many) in ports.items():
            v = ins.get(k)
            if v is None or v == []:
                if not optional:
                    errs.append(f"{i}: {prim.op} needs its {k!r} input")
                continue
            if many != isinstance(v, list):
                errs.append(f"{i}: input {k!r} must be {'a list of node ids' if many else 'one node id'}")
                continue
            for ref in (v if many else [v]):
                if ref not in nodes:
                    errs.append(f"{i}: input {k!r} refers to {ref!r}, which is not a node")
                elif PRIMITIVES[nodes[ref]["op"]].output not in t.split("|"):
                    errs.append(f"{i}: input {k!r} needs a {t.replace('|', ' or ')}, but {ref!r} produces a {PRIMITIVES[nodes[ref]['op']].output}")
        params = n.get("params") or {}
        for k in params:
            if k not in prim.params:
                errs.append(f"{i}: {prim.op} has no parameter {k!r} (it takes {', '.join(prim.params) or 'none'})")
        for k, spec in prim.params.items():
            if params.get(k) is None:
                if spec["required"]:
                    errs.append(f"{i}: {prim.op} needs the parameter {k!r}")
                continue
            _check_param(k, spec, params[k], errs, i)
    if errs:
        return {"errors": errs, "order": [], "types": {}, "known_at": {}}
    try:
        order = _order(nodes)
    except GraphError as e:
        return {"errors": e.errors, "order": [], "types": {}, "known_at": {}}
    types = {i: PRIMITIVES[nodes[i]["op"]].output for i in order}
    known: dict[str, str] = {}
    for i in order:
        n, pa, ins = nodes[i], nodes[i].get("params") or {}, nodes[i].get("inputs") or {}
        op = n["op"]
        if op == "feature":
            if bool(pa.get("name")) == bool(pa.get("expr")):
                errs.append(f"{i}: a feature needs exactly one of `name` (a catalog feature) or `expr`")
                continue
            at = pa.get("at") or "open"
            if pa.get("name"):
                fd = feature_def(pa["name"], "SPY")
                if fd.kind == "intraday" and at in ("open", "close"):
                    errs.append(f"{i}: {fd.label.lower()} is read during the session, so it needs a clock time in `at`")
                known[i] = at if (fd.kind == "intraday" or at == "close") else "open"
            else:
                known[i] = at
                if X.needs_intraday(pa["expr"]) and at in ("open", "close"):
                    errs.append(f"{i}: this expression reads the session so far, so it needs a clock time in `at`")
        elif op == "filter":
            has_x, has_e = bool(ins.get("x")), bool(pa.get("expr"))
            if has_x == has_e:
                errs.append(f"{i}: a condition takes either a series input with op/value, or a boolean `expr` with the data input")
                continue
            if has_x:
                if pa.get("op") is None or pa.get("value") is None:
                    errs.append(f"{i}: a condition on a series needs `op` and `value`")
                if pa.get("op") == "between" and pa.get("hi") is None:
                    errs.append(f"{i}: between needs `hi`")
                known[i] = known.get(ins["x"], "open")
            else:
                if not ins.get("data"):
                    errs.append(f"{i}: an expression condition needs the data input")
                at = pa.get("at") or "open"
                known[i] = at
                if X.needs_intraday(pa["expr"]) and at in ("open", "close"):
                    errs.append(f"{i}: this expression reads the session so far, so it needs a clock time in `at`")
        elif op == "combine":
            ms = ins.get("masks") or []
            if (pa.get("how") or "and") == "not" and len(ms) != 1:
                errs.append(f"{i}: not takes exactly one condition")
            at = "open"
            for m in ms:
                at = _later(at, known.get(m, "open"))
            known[i] = at
        elif op in ("event", "sequence"):
            known[i] = "event"
        elif op in ("forward_return", "target_stop"):
            entry = "event" if ins.get("anchor") else (pa.get("entry") or "open")
            hz = pa.get("horizon") or "session"
            if entry == "close" and not hz.startswith("sessions:"):
                errs.append(f"{i}: an observation that starts at the close needs a horizon of one or more sessions (sessions:N)")
            if hz.startswith("time:") and entry not in ("event",) and tick_order(hz.split(":", 1)[1]) <= tick_order(entry):
                errs.append(f"{i}: the horizon ends at {hz.split(':', 1)[1]}, which is not after the entry at {entry}")
            known[i] = entry
        elif op == "sweep":
            tgt = pa.get("node")
            if tgt not in nodes:
                errs.append(f"{i}: sweep refers to {tgt!r}, which is not a node")
            elif pa.get("param") not in PRIMITIVES[nodes[tgt]["op"]].params:
                errs.append(f"{i}: {nodes[tgt]['op']} has no parameter {pa.get('param')!r} to vary")
            else:
                spec = PRIMITIVES[nodes[tgt]["op"]].params[pa["param"]]
                for v in pa.get("values") or []:
                    _check_param(pa["param"], spec, v, errs, f"{i} (value {v!r})")
    # ----- point in time: nothing that conditions an outcome may be known after the outcome starts
    for i in order:
        n, ins = nodes[i], nodes[i].get("inputs") or {}
        if n["op"] not in ("conditional_probability", "matched_comparison", "distribution", "correlation", "regression", "robustness", "validation", "options_backtest"):
            continue
        oc = ins.get("outcome") or (ins.get("y") if types.get(ins.get("y")) == "outcome" else None)
        if not oc or oc not in known:
            continue
        start = known[oc]
        for port in ("condition", "controls", "by", "xs", "x"):
            v = ins.get(port)
            for ref in (v if isinstance(v, list) else [v] if v else []):
                at = known.get(ref)
                if at is None:
                    continue
                if start == "event":
                    if at not in ("open",):
                        errs.append(f"{i}: {ref!r} is read at {at}, but the observation starts at an intraday event whose time varies; only values known from the open can be combined with it")
                elif tick_order(at) > tick_order(start):
                    errs.append(f"{i}: {ref!r} is only known at {at}, but the outcome {oc!r} is measured from {start}. Conditioning on it would use information from the future; move the entry to {at} or read the value earlier")
        if n["op"] == "matched_comparison":
            cond_feats = _clause_names(nodes, ins.get("condition"))
            for ref in ins.get("controls") or []:
                key = _series_key(nodes.get(ref) or {})
                if key and key in cond_feats:
                    errs.append(f"{i}: {ref!r} ({key}) defines the condition, so it cannot also be a control: the comparison sessions would be excluded by construction")
    outs = graph.get("outputs")
    if outs is not None:
        for o in outs if isinstance(outs, list) else []:
            if o not in nodes:
                errs.append(f"outputs refers to {o!r}, which is not a node")
    return {"errors": errs, "order": order, "types": types, "known_at": known}


def _series_key(node: dict) -> str | None:
    pa = node.get("params") or {}
    return pa.get("name") or (pa.get("expr") or "").replace(" ", "") or None


def _clause_names(nodes: dict, ref) -> set[str]:
    """Feature names / expressions that define a condition (walks combine and filter nodes)."""
    out: set[str] = set()
    stack = [ref] if ref else []
    while stack:
        n = nodes.get(stack.pop()) or {}
        ins = n.get("inputs") or {}
        if n.get("op") == "combine":
            stack += list(ins.get("masks") or [])
        elif n.get("op") == "filter" and ins.get("x"):
            k = _series_key(nodes.get(ins["x"]) or {})
            if k:
                out.add(k)
    return out


# ----------------------------------------------------------------------------- runtime values
@dataclass
class Dataset:
    symbol: str
    ft: SessionFeatures
    pm: np.ndarray  # sessions inside the requested period and outside the reserved final test period
    full: np.ndarray  # the requested period, including the final test period
    holdout: dict | None = None
    note: str = ""


@dataclass
class Series:
    ds: Dataset
    x: np.ndarray
    at: str
    label: str
    unit: str = ""
    key: str = ""


@dataclass
class Mask:
    ds: Dataset
    true: np.ndarray
    known: np.ndarray
    at: str
    label: str
    parts: list = field(default_factory=list)  # the AND-ed clauses, for ablation
    src: dict | None = None  # {"series": Series, "op", "value", "hi"} or {"expr", "at"} for a single clause
    pred: object = None  # the same clause as a Pred / ExprPred when it reads at the outcome's entry tick


@dataclass
class Out:
    ds: Dataset
    spec: OutcomeSpec | None
    code: np.ndarray
    end_ret: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray
    label: str
    entry: str
    reasons: dict = field(default_factory=dict)
    event: np.ndarray | None = None  # anchored outcomes: sessions on which the event happened
    base_code: np.ndarray | None = None  # anchored outcomes: every evaluated session, observed from its open
    base_ret: np.ndarray | None = None
    study: dict | None = None

    @property
    def valid(self) -> np.ndarray:
        return self.code >= 0


def _fmt_p(p) -> str:
    """Printed to the precision it has. These p-values come from a few hundred resamples, so their resolution is a few
    thousandths and, around 0.1, they move by a hundredth or two when one session is added: four decimals would claim more."""
    if p is None:
        return "not available"
    if p < 0.005:
        return "below 0.005"
    return f"{p:.2f}" if p >= 0.095 else f"{p:.3f}"


def _pct(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------------- execution
class Run:
    def __init__(self, graph: dict, store: MarketStore | None = None, scope: str | None = None, run_id: str | None = None, include_holdout: bool = False, reps: int = 400):
        self.graph = graph
        self.store = store or MarketStore()
        self.scope, self.run_id, self.reps = scope, run_id, reps
        self.include_holdout = include_holdout or bool(graph.get("include_holdout"))
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.values: dict[str, object] = {}
        self.trials: list[dict] = []
        self.cross = self._cross()
        self.wants_validation = any(n["op"] == "validation" for n in graph["nodes"])

    def _cross(self) -> tuple[str, ...]:
        syms: set[str] = set()
        for n in self.graph["nodes"]:
            pa = n.get("params") or {}
            for src in [pa.get("expr")] + [s.get("when") for s in (pa.get("steps") or []) if isinstance(s, dict)]:
                if src:
                    try:
                        syms |= X.symbols_used(src)
                    except X.ExprError:
                        pass
            if pa.get("name"):
                nm = str(pa["name"])
                for pre, suf in (("rs5_", ""), ("", "_ret_1"), ("", "_gap")):
                    if (pre and nm.startswith(pre)) or (suf and nm.endswith(suf)):
                        syms.add(nm[len(pre):len(nm) - len(suf)] if suf else nm[len(pre):])
            syms |= {str(x).upper() for x in (pa.get("cross") or [])}
        return tuple(sorted(syms))

    def get(self, i: str):
        if i not in self.values:
            n = self.nodes[i]
            self.values[i] = getattr(self, "_" + n["op"])(n, n.get("params") or {}, n.get("inputs") or {})
        return self.values[i]

    # ----- data
    def _data(self, n, pa, ins) -> Dataset:
        sym = str(pa["symbol"]).upper()
        ft = get_features(self.store, sym, tuple(x for x in self.cross if x != sym))
        if ft.missing_cross:
            raise LookupError(f"No price history is stored for {', '.join(ft.missing_cross)}, which this question reads. Download it under Data (daily history is free).")
        full = ft.window(pa.get("start"), pa.get("end"))
        pm, hold, note = full.copy(), None, ""
        key = self._holdout_symbol(sym)
        obs = self._observable(ft)  # a final test period is cut from the sessions this graph can observe, not from the calendar
        if self.scope and not self.include_holdout:
            hold = LG.reserve(self.scope, key, [d for d, m in zip(ft.dates, full & obs) if m]) if (self.wants_validation and (full & obs).any()) else LG.current(self.scope, key)
            if hold is not None and hold.get("status") in ("sealed", "used"):
                pm &= np.array(ft.dates) < hold["start"]
                note = f"The final test period ({hold['start']} to {hold['end']}) is held back: everything here is measured on sessions before it."
        elif not self.scope and self.wants_validation:
            days = [d for d, m in zip(ft.dates, full & obs) if m]
            if len(days) >= 250:
                cut = days[int(len(days) * 0.8)]
                hold = {"start": cut, "end": days[-1], "status": "local", "usable": True, "label": "untouched", "id": None}
                pm &= np.array(ft.dates) < cut
                note = f"The final test period ({cut} to {days[-1]}) is held back (not recorded in a research ledger: this run has no notebook)."
        return Dataset(sym, ft, pm, full, hold, note)

    def _observable(self, ft: SessionFeatures) -> np.ndarray:
        """Sessions on which every clock time the graph reads has a bar boundary."""
        ok = np.ones(ft.n, bool)
        for n in self.graph["nodes"]:
            pa = n.get("params") or {}
            for k in ("at", "entry"):
                if pa.get(k) not in (None, "open", "close") and _tick_ok(pa.get(k)):
                    ok &= ft._tick(str(pa[k]))[1]
            hz = pa.get("horizon")
            if isinstance(hz, str) and hz.startswith("time:"):
                ok &= ft.res < 86400
            if n["op"] == "event":
                ok &= ft.res < 86400
            if n["op"] == "sequence":
                ok &= ft.res <= 300
        return ok

    def _holdout_symbol(self, sym: str) -> str:
        intraday = any((n.get("params") or {}).get("entry") not in (None, "open", "close") or n["op"] in ("event", "sequence") for n in self.graph["nodes"] if n["op"] in ("forward_return", "target_stop", "event", "sequence"))
        return f"{sym}@intraday" if intraday else sym

    # ----- features and conditions
    def _feature(self, n, pa, ins) -> Series:
        ds: Dataset = self.get(ins["data"])
        at = pa.get("at") or "open"
        if pa.get("name"):
            fd = feature_def(pa["name"], ds.symbol)
            x = ds.ft.col(pa["name"], at)
            known = at if (fd.kind == "intraday" or at == "close") else "open"
            return Series(ds, x, known, pa.get("label") or fd.words or fd.label.lower(), fd.unit, pa["name"])
        x = ds.ft.engine_values(pa["expr"], at, cache=True)
        return Series(ds, x, at, pa.get("label") or pa["expr"], "", pa["expr"].replace(" ", ""))

    def _filter(self, n, pa, ins) -> Mask:
        if ins.get("x"):
            s: Series = self.get(ins["x"])
            p = Pred(s.key if s.key in FEATURES or _is_cross(s.key) else "_", pa["op"], float(pa["value"]), None if pa.get("hi") is None else float(pa["hi"]))
            fin = np.isfinite(s.x)
            label = pa.get("label") or _clause_text(s, pa)
            xn = self.nodes[ins["x"]].get("params") or {}
            pred = Pred(xn["name"], pa["op"], float(pa["value"]), None if pa.get("hi") is None else float(pa["hi"])) if xn.get("name") else ExprPred(_cmp_expr(xn.get("expr", ""), pa), label)
            m = Mask(s.ds, p.mask(s.x) & fin, fin, s.at, label, src={"series": s, "op": pa["op"], "value": float(pa["value"]), "hi": pa.get("hi"), "tick": xn.get("at") or "open"}, pred=pred)
        else:
            ds: Dataset = self.get(ins["data"])
            at = pa.get("at") or "open"
            x = ds.ft.engine_values(pa["expr"], at, cache=True)
            fin = np.isfinite(x)
            label = pa.get("label") or pa["expr"]
            m = Mask(ds, (x > 0.5) & fin, fin, at, label, src={"expr": pa["expr"], "tick": at}, pred=ExprPred(pa["expr"], label))
        m.parts = [m]
        return m

    def _combine(self, n, pa, ins) -> Mask:
        ms: list[Mask] = [self.get(i) for i in ins["masks"]]
        how = pa.get("how") or "and"
        ds = ms[0].ds
        if any(m.ds is not ds for m in ms):
            raise ValueError(f"{n['id']}: the conditions come from different datasets and cannot be combined")
        at = "open"
        for m in ms:
            at = _later(at, m.at)
        if how == "not":
            m = ms[0]
            return Mask(ds, ~m.true & m.known, m.known.copy(), at, f"not ({m.label})", src=None, parts=[], pred=None)
        if how == "and":
            true = np.ones(ds.ft.n, bool)
            any_false = np.zeros(ds.ft.n, bool)
            all_known = np.ones(ds.ft.n, bool)
            for m in ms:
                true &= m.true
                any_false |= m.known & ~m.true
                all_known &= m.known
            out = Mask(ds, true & all_known, any_false | all_known, at, " and ".join(m.label for m in ms))
            out.parts = [p for m in ms for p in (m.parts or [m])]
            return out
        true = np.zeros(ds.ft.n, bool)
        all_known = np.ones(ds.ft.n, bool)
        for m in ms:
            true |= m.true
            all_known &= m.known
        return Mask(ds, true, true | all_known, at, "(" + " or ".join(m.label for m in ms) + ")")

    # ----- events
    def _event(self, n, pa, ins) -> dict:
        ds: Dataset = self.get(ins["data"])
        tree = {"id": "root", "kind": "group", "op": "and", "children": [{"id": "c0", "kind": "cond", "type": "expr", "params": {"expr": pa["expr"], "label": pa.get("label") or pa["expr"]}}]}
        return {"ds": ds, "condition": tree, "sequence": [], "label": pa.get("label") or f"the first time {pa['expr']}"}

    def _sequence(self, n, pa, ins) -> dict:
        ds: Dataset = self.get(ins["data"])
        steps = [{"name": str(s["name"]), "when": str(s["when"]), **({"within_minutes": int(s["within_minutes"])} if s.get("within_minutes") else {}), "text": s.get("text") or str(s["when"])} for s in pa["steps"]]
        return {"ds": ds, "condition": {"id": "root", "kind": "group", "op": "and", "children": []}, "sequence": steps, "label": pa.get("label") or ", then ".join(s["text"] for s in steps)}

    # ----- outcomes
    def _outcome(self, n, pa, ins, spec: OutcomeSpec) -> Out:
        ds: Dataset = self.get(ins["data"])
        if ins.get("anchor"):
            return self._anchored(ds, self.get(ins["anchor"]), spec)
        o = ds.ft.outcome(spec)
        return Out(ds, spec, o.code, o.end_ret, o.mfe, o.mae, spec.text(), spec.entry, dict(o.reasons))

    def _forward_return(self, n, pa, ins) -> Out:
        return self._outcome(n, pa, ins, OutcomeSpec(pa.get("entry") or "open", pa.get("horizon") or "session", pa.get("direction") or "up", float(pa.get("threshold") or 0.0), None, "close"))

    def _target_stop(self, n, pa, ins) -> Out:
        return self._outcome(n, pa, ins, OutcomeSpec(pa.get("entry") or "open", pa.get("horizon") or "session", pa.get("direction") or "up", float(pa["target"]), None if pa.get("stop") is None else float(pa["stop"]), "touch"))

    def _anchored(self, ds: Dataset, anchor: dict, spec: OutcomeSpec) -> Out:
        """Outcome measured from an intraday event, by the event-study engine."""
        from strata.research.study import run_study
        if anchor["ds"] is not ds:
            raise ValueError("the event and the outcome come from different datasets")
        cfg = to_study(ds.symbol, [], spec)
        cfg.update(condition=anchor["condition"], sequence=anchor["sequence"], entry_time="signal", label=anchor["label"])
        res = run_study(cfg, self.store, max_sessions_out=100000)
        n = ds.ft.n
        idx = {d: i for i, d in enumerate(ds.ft.dates)}
        code = np.full(n, -1, np.int8)
        end_ret, mfe, mae = (np.full(n, np.nan) for _ in range(3))
        event = np.zeros(n, bool)
        for r in res.get("sessions", []):
            i = idx.get(r["date"])
            if i is None:
                continue
            event[i] = True
            code[i] = CODE_NAMES.index(r["outcome"])
            end_ret[i], mfe[i], mae[i] = r["end_return_pct"], r["mfe_pct"], r["mae_pct"]
        b = res.get("baseline_sessions") or {"dates": [], "outcomes": "", "end_returns": []}
        base_code = np.full(n, -1, np.int8)
        base_ret = np.full(n, np.nan)
        for d, o, e in zip(b["dates"], b["outcomes"], b.get("end_returns") or [np.nan] * len(b["dates"])):
            i = idx.get(d)
            if i is not None:
                base_code[i] = "tsna".index(o)
                base_ret[i] = e
        reasons = {k: len(v) for k, v in (res.get("excluded_sessions") or {}).items()}
        return Out(ds, spec, code, end_ret, mfe, mae, f"{spec.text()}, observed from the moment {anchor['label']}", "event", reasons, event, base_code, base_ret, cfg)

    # ----- shared pieces of the analyses
    def _frame(self, o: Out, m: Mask | None):
        """(qualifying, baseline, code, end_ret) for a condition against an outcome."""
        ds = o.ds
        if o.event is not None:
            # anchored: qualifying = event sessions (optionally where a daily condition also held); baseline = every evaluated session from its open
            ok = (o.base_code >= 0) & ds.pm
            q = o.event & ok & (m.true if m is not None else True)
            code = np.where(q, o.code, o.base_code)
            ret = np.where(q, o.end_ret, o.base_ret)
            base = ok if m is None else ok & m.known
            return q, base, code, ret
        if m is None:
            raise ValueError("a condition is needed unless the outcome starts at an intraday event")
        if m.ds is not ds:
            raise ValueError("the condition and the outcome come from different datasets")
        base = o.valid & m.known & ds.pm
        return m.true & base, base, o.code, o.end_ret

    def _evidence(self, o: Out, m: Mask | None, **kw) -> dict:
        q, base, code, ret = self._frame(o, m)
        return EV.compare(o.ds.ft.dates, code, ret, q, base, (o.spec.direction if o.spec else "up"), reps=self.reps, return_ci=True, mean_diff=True,
                          mae=o.mae if o.event is None else None, mfe=o.mfe if o.event is None else None, **kw)

    def _trial(self, kind: str, o: Out, m: Mask | None, e: dict, label: str, extra: dict | None = None) -> str:
        ds = o.ds
        sig = {"symbol": ds.symbol, "clauses": sorted(self._clause_sigs(m)), "outcome": o.spec.dump() if o.spec else {}, "anchor": (o.study or {}).get("label"), **(extra or {})}
        key = LG.config_key(sig)
        _, base, _, _ = self._frame(o, m)
        dates = np.array(ds.ft.dates)[base]
        self.trials.append({"kind": kind, "key": key, "label": label[:200], "run_id": self.run_id, "start": str(dates[0]) if dates.size else None, "end": str(dates[-1]) if dates.size else None,
                            "metric": "lift", "value": e.get("lift"), "n": e.get("n"), "p_value": e.get("p_value"), "symbol": self._holdout_symbol(ds.symbol), "signature": signature(sig)})
        return key

    def _clause_sigs(self, m: Mask | None) -> list[str]:
        if m is None:
            return []
        out = []
        for p in (m.parts or [m]):
            s = p.src or {}
            if "series" in s:
                out.append(f"{s['series'].key}@{s.get('tick')}{s['op']}{s['value']:g}" + (f"..{s['hi']:g}" if s.get("hi") is not None else ""))
            elif "expr" in s:
                out.append(f"{s['expr'].replace(' ', '')}@{s.get('tick')}")
            else:
                out.append(p.label)
        return out

    def _texts(self, o: Out, m: Mask | None) -> dict:
        return {"condition": m.label if m is not None else (o.study or {}).get("label", ""), "outcome": o.label, "symbol": o.ds.symbol,
                "decision_tick": o.entry, "known_at": m.at if m is not None else "event"}

    def _sessions(self, o: Out, q: np.ndarray, code, ret) -> list[dict]:
        idx = np.flatnonzero(q)[-MAX_SESSIONS_OUT:]
        return [{"date": o.ds.ft.dates[i], "outcome": CODE_NAMES[int(code[i])] if code[i] >= 0 else None, "end_return_pct": None if not math.isfinite(ret[i]) else round(float(ret[i]), 4)} for i in idx]

    # ----- analyses
    def _conditional_probability(self, n, pa, ins) -> dict:
        o: Out = self.get(ins["outcome"])
        m: Mask | None = self.get(ins["condition"]) if ins.get("condition") else None
        e = self._evidence(o, m)
        q, base, code, ret = self._frame(o, m)
        tx = self._texts(o, m)
        key = self._trial("graph", o, m, e, f"{tx['condition']} -> {tx['outcome']}")
        need = int(pa.get("min_sample") or 30)
        out = {"kind": "probability", **tx, "evidence": e, "key": key, "thin": e["n"] < need, "min_sample": need, "sessions": self._sessions(o, q, code, ret),
               "not_evaluable": dict(o.reasons), "unknown_condition": int((o.valid & o.ds.pm & ~(m.known if m is not None else True)).sum()) if o.event is None else 0,
               "window": _window(o.ds, base), "study": self._study_of(o, m), "cell": self._cell_of(o, m)}
        out["by"] = [self._by(o, q, base, code, self.get(i)) for i in ins.get("by") or []]
        if out["by"]:
            adj = R.adjust([b.get("p_value") for b in out["by"]], "holm")
            for b, a in zip(out["by"], adj):
                b["p_adjusted"] = a
        out["summary"] = _prob_sentence(out)
        return out

    def _by(self, o: Out, q: np.ndarray, base: np.ndarray, code, s: Series) -> dict:
        """Among the qualifying sessions: does the outcome depend on this value? Terciles, top against bottom."""
        if tick_order(s.at) > tick_order(o.entry if o.entry != "event" else "open") and o.entry != "event":
            raise ValueError(f"{s.label} is known at {s.at}, after the outcome starts")
        ok = q & np.isfinite(s.x) & (code != A)
        x = s.x[ok]
        row = {"feature": s.key, "label": s.label, "unit": s.unit, "n": int(ok.sum()), "buckets": [], "p_value": None, "difference": None}
        if x.size < 30:
            row["note"] = "fewer than 30 qualifying sessions have this value"
            return row
        uniq = np.unique(x)
        edges = uniq[1:] if uniq.size <= 3 else np.unique(np.quantile(x, [1 / 3, 2 / 3]))
        b = np.searchsorted(edges, s.x, side="right")
        dates = np.array(o.ds.ft.dates)
        names = ["low", "middle", "high"] if edges.size == 2 else [f"group {i + 1}" for i in range(edges.size + 1)]
        for j in range(edges.size + 1):
            mj = ok & (b == j)
            nj = int(mj.sum())
            kj = int((code[mj] == T).sum())
            row["buckets"].append({"name": names[j], "range": [None if j == 0 else float(edges[j - 1]), None if j == edges.size else float(edges[j])], "n": nj, "probability": (kj / nj) if nj else None,
                                   "ci95": R.wilson(kj, nj), "avg_end_return_pct": float(np.nanmean(o.end_ret[mj])) if nj else None})
        lo, hi = ok & (b == 0), ok & (b == edges.size)
        if lo.sum() >= 10 and hi.sum() >= 10:
            lt = R.lift_test((code[hi] == T).astype(float), dates[hi], (code[lo] == T).astype(float), dates[lo], "week", self.reps)
            row.update(p_value=lt["p_value"] if lt["p_value"] is not None else lt["p_iid"], difference=lt["lift"], ci95=lt["ci"], test="week-resampled" if lt["p_value"] is not None else "independent-sample (too few weeks)")
        return row

    def _matched_comparison(self, n, pa, ins) -> dict:
        o: Out = self.get(ins["outcome"])
        m: Mask = self.get(ins["condition"])
        q, base, code, ret = self._frame(o, m)
        res_q, pool = q & (code != A), base & ~q & (code != A)
        controls = []
        for i in ins.get("controls") or []:
            s: Series = self.get(i)
            controls.append({"name": s.key or s.label, "label": s.label, "values": s.x, "unit": s.unit})
        tx = self._texts(o, m)
        hit = MT.matched_comparison(o.ds.ft.dates, (code == T).astype(float), res_q, pool, controls, int(pa.get("k") or 5), float(pa.get("caliper") or 0.5), reps=self.reps, binary=True)
        rt = MT.matched_comparison(o.ds.ft.dates, ret, q, base & ~q, controls, int(pa.get("k") or 5), float(pa.get("caliper") or 0.5), reps=self.reps, binary=False)
        e = self._evidence(o, m)
        self._trial("graph", o, m, e, f"matched: {tx['condition']} -> {tx['outcome']}", {"controls": sorted(c["name"] for c in controls)})
        return {"kind": "matched", **tx, "evidence": e, "frequency": hit, "returns": rt, "window": _window(o.ds, base),
                "summary": [hit["sentence"]] + ([f"On end-of-horizon returns: {rt['sentence']}"] if rt.get("sentence") and rt["verdict"] != "not_estimable" else []), "notes": hit["notes"]}

    def _distribution(self, n, pa, ins) -> dict:
        from strata.research import patterns as PT
        o: Out = self.get(ins["outcome"])
        m: Mask | None = self.get(ins["condition"]) if ins.get("condition") else None
        q, base, code, ret = self._frame(o, m)
        a, b = ret[q], ret[base & ~q]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        tx = self._texts(o, m)
        out = {"kind": "distribution", **tx, "n": int(a.size), "rest_n": int(b.size), "window": _window(o.ds, base),
               "quantiles": {"condition": _quant(a), "rest": _quant(b)}, "histogram": _hist(a, b), "excursions": None, "paths": None}
        if o.event is None:
            out["excursions"] = {"mfe": _quant(o.mfe[q][np.isfinite(o.mfe[q])]), "mae": _quant(o.mae[q][np.isfinite(o.mae[q])])}
            if pa.get("paths", True):
                out["paths"] = PT.paths(o.ds.ft, q, o.entry, int(pa.get("horizon_sessions") or 10), base & ~q, reps=self.reps)
        s = [f"{a.size} qualifying sessions: median end-of-horizon return {np.median(a):+.2f}%, middle half {np.quantile(a, 0.25):+.2f}% to {np.quantile(a, 0.75):+.2f}%"
             + (f" (other sessions: median {np.median(b):+.2f}%)." if b.size else ".")] if a.size >= 4 else [f"Only {a.size} qualifying sessions: too few to describe a distribution."]
        out["summary"] = s
        return out

    def _correlation(self, n, pa, ins) -> dict:
        x: Series = self.get(ins["x"])
        yv = self.get(ins["y"])
        ds = x.ds
        if isinstance(yv, Out):
            y, ylabel = yv.end_ret, f"the return that follows ({yv.label})"
            ok = yv.valid & ds.pm
        else:
            y, ylabel = yv.x, yv.label
            ok = ds.pm.copy()
        if ins.get("condition"):
            cm: Mask = self.get(ins["condition"])
            ok &= cm.true
        ok &= np.isfinite(x.x) & np.isfinite(y)
        a, b = x.x[ok], y[ok]
        method = pa.get("method") or "spearman"
        out = {"kind": "correlation", "x": x.label, "y": ylabel, "n": int(a.size), "method": method, "value": None, "ci95": [None, None], "p_value": None, "symbol": ds.symbol, "window": _window(ds, ok),
               "predictive": isinstance(yv, Out)}
        if a.size >= 20 and a.std() > 0 and b.std() > 0:
            from scipy import stats as sst
            if method == "spearman":
                a_, b_ = sst.rankdata(a), sst.rankdata(b)
            else:
                a_, b_ = a, b
            r = float(np.corrcoef(a_, b_)[0, 1])
            # block bootstrap over time: neighbouring sessions share volatility, so pairs are resampled in blocks
            blk = max(R.block_length(a_ * b_), 5)
            rng = np.random.default_rng(7)
            idx = R.moving_block_indices(a.size, blk, self.reps, rng)
            A_, B_ = a_[idx], b_[idx]
            A_ = A_ - A_.mean(axis=1, keepdims=True)
            B_ = B_ - B_.mean(axis=1, keepdims=True)
            den = np.sqrt((A_ ** 2).sum(axis=1) * (B_ ** 2).sum(axis=1))
            draws = (A_ * B_).sum(axis=1)[den > 0] / den[den > 0]
            se = float(draws.std(ddof=1)) if draws.size > 10 else None
            out.update(value=r, ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))] if draws.size > 10 else [None, None], block=int(blk),
                       p_value=R._p_from_draws(draws - draws.mean() + r, r, se, max(2, a.size / blk - 1)) if se else None)
        v = out["value"]
        out["summary"] = [f"{method.capitalize()} correlation between {x.label} and {ylabel}: " + (f"{v:+.3f} over {a.size} sessions (95% interval {out['ci95'][0]:+.3f} to {out['ci95'][1]:+.3f})."
                                                                                              if v is not None and out["ci95"][0] is not None else f"not estimable ({a.size} usable sessions).")
                          + ("" if isinstance(yv, Out) else " Both values belong to the same session, so this describes co-movement, not prediction.")]
        return out

    def _regression(self, n, pa, ins) -> dict:
        yv = self.get(ins["y"])
        xs: list[Series] = [self.get(i) for i in ins.get("xs") or []]
        if not xs:
            raise ValueError(f"{n['id']}: a regression needs at least one regressor")
        ds = xs[0].ds
        if isinstance(yv, Out):
            target = pa.get("target") or "return"
            y = (yv.code == T).astype(float) if target == "hit" else yv.end_ret
            ok = yv.valid & ds.pm & ((yv.code != A) if target == "hit" else True)
            ylabel = ("whether " + yv.label) if target == "hit" else f"the return that follows, % ({yv.label})"
        else:
            y, ok, ylabel, target = yv.x, ds.pm.copy(), yv.label, "value"
        if ins.get("condition"):
            ok &= self.get(ins["condition"]).true
        Xm = np.column_stack([s.x for s in xs])
        ok &= np.isfinite(Xm).all(axis=1) & np.isfinite(y)
        out = {"kind": "regression", "y": ylabel, "target": target, "n": int(ok.sum()), "terms": [], "r2": None, "symbol": ds.symbol, "window": _window(ds, ok), "predictive": isinstance(yv, Out)}
        if ok.sum() < 30 + 5 * len(xs):
            out["summary"] = [f"Only {int(ok.sum())} sessions have every regressor: too few for a regression with {len(xs)} term{'s' if len(xs) != 1 else ''}."]
            return out
        Z = Xm[ok]
        mu, sd = Z.mean(axis=0), Z.std(axis=0, ddof=1)
        keep = sd > 0
        Z = (Z[:, keep] - mu[keep]) / sd[keep]
        yy = y[ok]
        D = np.column_stack([np.ones(Z.shape[0]), Z])
        inv = np.linalg.pinv(D.T @ D)
        beta = inv @ D.T @ yy
        resid = yy - D @ beta
        cl = R.cluster_ids(np.array(ds.ft.dates)[ok], "week")
        uniq, ix = np.unique(cl, return_inverse=True)
        G = uniq.size
        S = np.zeros((G, D.shape[1]))
        np.add.at(S, ix, D * resid[:, None])
        V = inv @ (S.T @ S) @ inv * (G / max(1, G - 1))
        from scipy.stats import t as tdist
        tq = float(tdist.ppf(0.975, max(2, G - 1)))
        kept = [s for s, k_ in zip(xs, keep) if k_]
        ps = []
        for j, s in enumerate(kept, start=1):
            se = math.sqrt(max(V[j, j], 0.0))
            p = float(2 * (1 - tdist.cdf(abs(beta[j]) / se, max(2, G - 1)))) if se > 0 else None
            ps.append(p)
            out["terms"].append({"name": s.key, "label": s.label, "coef_per_sd": float(beta[j]), "se": se, "ci95": [float(beta[j] - tq * se), float(beta[j] + tq * se)], "p_value": p, "sd": float(sd[keep][j - 1]), "mean": float(mu[keep][j - 1])})
        for t_, a_ in zip(out["terms"], R.adjust(ps, "holm")):
            t_["p_adjusted"] = a_
        tot = float(((yy - yy.mean()) ** 2).sum())
        out.update(r2=float(1 - (resid ** 2).sum() / tot) if tot > 0 else None, intercept=float(beta[0]), clusters=int(G))
        strong = [t_ for t_ in out["terms"] if t_["p_adjusted"] is not None and t_["p_adjusted"] < 0.05]
        unit = " points of probability" if target == "hit" else "%"
        scale = 100 if target == "hit" else 1
        out["summary"] = [f"Regression of {ylabel} on {len(kept)} value{'s' if len(kept) != 1 else ''} over {out['n']} sessions (R² {out['r2']:.3f}): "
                          + ("; ".join(f"one standard deviation more of {t_['label']} goes with {t_['coef_per_sd'] * scale:+.2f}{unit}" for t_ in strong[:4]) + " (significant after correcting for all terms)."
                             if strong else "no term is distinguishable from zero after correcting for the number of terms.")]
        return out

    def _sweep(self, n, pa, ins) -> dict:
        tgt, param, values = pa["node"], pa["param"], pa["values"]
        rows = []
        # Across symbols the comparison is only fair on one period, and it must not look into the sessions this notebook
        # holds back: other symbols' sessions on those dates are the same market days. So every symbol is evaluated from
        # the first date all of them have, to the end of the base symbol's own explored window.
        by_symbol = param == "symbol" and len(values) > 1
        base_sym = str((self.nodes[tgt].get("params") or {}).get("symbol") or "").upper()
        common_start = cap = None
        if by_symbol:
            firsts = []
            for v in values:
                try:
                    firsts.append(str(self.store.read_daily(str(v).upper())["date"][0]))
                except Exception:  # noqa: BLE001 - a symbol with no history is reported in its own row
                    pass
            common_start = max(firsts) if firsts else None
            values = sorted(values, key=lambda v: str(v).upper() != base_sym)  # the base symbol first: its window sets the end
        for v in values:
            g = copy.deepcopy(self.graph)
            for gn in g["nodes"]:
                if gn["id"] == tgt:
                    gp = gn.setdefault("params", {})
                    gp[param] = v
                    if by_symbol:
                        if common_start and str(gp.get("start") or "") < common_start:
                            gp["start"] = common_start
                        if cap and (not gp.get("end") or str(gp["end"]) > cap):
                            gp["end"] = cap
                elif by_symbol and base_sym and str(v).upper() != base_sym:
                    gp = gn.get("params") or {}
                    for k in ("label", "measure"):  # "SPY fell 2% or more" is about QQQ when the data is QQQ's
                        if isinstance(gp.get(k), str):
                            gp[k] = re.sub(rf"\b{re.escape(base_sym)}\b", str(v).upper(), gp[k])
            g["nodes"] = [gn for gn in g["nodes"] if gn["op"] != "sweep"]
            chk = validate(g)
            if chk["errors"]:
                rows.append({"value": v, "status": "error", "error": chk["errors"][0]})
                continue
            sub = Run(g, self.store, self.scope, self.run_id, self.include_holdout, self.reps)
            sub.wants_validation = self.wants_validation
            try:
                r = sub.get(ins["result"])
            except (LookupError, ValueError, KeyError) as ex:
                rows.append({"value": v, "status": "error", "error": str(ex)[:200]})
                continue
            for t in sub.trials:
                t["kind"] = "sweep"
            self.trials += sub.trials
            e = r.get("evidence") or {}
            row = {"value": v, "status": "ok", "n": e.get("n"), "probability": e.get("probability"), "baseline_probability": e.get("baseline_probability"), "lift": e.get("lift"), "ci95": e.get("ci95"),
                   "p_value": e.get("p_value"), "avg_end_return_pct": e.get("avg_end_return_pct"), "thin": (e.get("n") or 0) < 30, "symbol": r.get("symbol"), "condition": r.get("condition"), "window": r.get("window")}
            if r.get("kind") == "matched":
                f = r["frequency"]
                row.update(matched=(f.get("nearest") or {}).get("estimate"), matched_ci95=(f.get("nearest") or {}).get("ci95"), verdict=f.get("verdict"))
            if r.get("kind") == "correlation":
                row.update(value_stat=r.get("value"), ci95=r.get("ci95"), p_value=r.get("p_value"), n=r.get("n"))
            rows.append(row)
            if by_symbol and cap is None and str(v).upper() == base_sym and r.get("window"):
                cap = str(r["window"][1])
        ok = [r for r in rows if r["status"] == "ok"]
        for r, a in zip(ok, R.adjust([r.get("p_value") for r in ok], "holm")):
            r["p_adjusted"] = a
            r["distinguishable"] = bool(a is not None and a < 0.05 and not r.get("thin"))
        signs = [r["lift"] > 0 for r in ok if r.get("lift") is not None and not r.get("thin")]
        agree = (sum(signs) == len(signs) or sum(signs) == 0) if signs else None
        what = "the symbol" if param == "symbol" else f"the {PRIMITIVES[self.nodes[tgt]['op']].title.lower()} {param}"
        s = [(f"The same analysis was run on {len(rows)} symbols" if param == "symbol" else f"{len(rows)} values of {what} were evaluated") + "; all are listed and p-values are corrected for all of them (Holm)."]
        if ok:
            keep = [r for r in ok if r.get("distinguishable")]
            s.append(f"{len(keep)} of {len(ok)} remain distinguishable from the baseline after correction" + (", and every usable value points the same way." if agree else ", and the direction changes across values: the result depends on this choice." if agree is False else "."))
        period = None
        if by_symbol and ok:
            a_, b_ = max(str(r["window"][0]) for r in ok if r.get("window")), min(str(r["window"][1]) for r in ok if r.get("window"))
            period = [a_, b_]
            s.append(f"Every symbol is evaluated on the same period, {a_} to {b_}: the history all of them have" + (", ending before the sessions this notebook holds back for a final test." if cap else "."))
        return {"kind": "sweep", "node": tgt, "param": param, "what": what, "rows": rows, "summary": s, "same_direction": agree, "period": period}

    def _robustness(self, n, pa, ins) -> dict:
        o: Out = self.get(ins["outcome"])
        m: Mask = self.get(ins["condition"])
        q, base, code, ret = self._frame(o, m)
        dates = np.array(o.ds.ft.dates)
        d = o.spec.direction if o.spec else "up"
        full = EV.compare(dates, code, ret, q, base, d, reps=self.reps)
        idx = np.flatnonzero(base)
        out = {"kind": "robustness", **self._texts(o, m), "overall": full, "periods": [], "years": [], "ablation": [], "neighbours": [], "window": _window(o.ds, base), "flags": []}
        if idx.size >= 120:
            for k, name in ((2, "half"), (3, "third")):
                for j, part in enumerate(np.array_split(idx, k)):
                    pm = np.zeros(base.size, bool)
                    pm[part] = True
                    e = EV.compare(dates, code, ret, q & pm, base & pm, d, robust=False)
                    out["periods"].append({"label": f"{['first', 'second', 'third'][j]} {name}", "start": str(dates[part[0]]), "end": str(dates[part[-1]]), "n": e["n"], "probability": e["probability"],
                                           "baseline_probability": e["baseline_probability"], "lift": e["lift"], "p_iid": e["p_iid"]})
        yrs = np.array([x[:4] for x in dates])
        for y in sorted(set(yrs[q])):
            pm = yrs == y
            e = EV.compare(dates, code, ret, q & pm, base & pm, d, robust=False)
            out["years"].append({"year": y, "n": e["n"], "probability": e["probability"], "baseline_probability": e["baseline_probability"], "lift": e["lift"]})
        usable = [r for r in out["years"] if r["n"] >= 5 and r["lift"] is not None]
        if usable and full["lift"] is not None:
            same = sum(1 for r in usable if (r["lift"] > 0) == (full["lift"] > 0))
            out["years_agree"] = [same, len(usable)]
            if full["n"]:
                top = max(usable, key=lambda r: r["n"])
                out["largest_year_share"] = top["n"] / full["n"]
                if top["n"] / full["n"] > 0.4:
                    out["flags"].append(f"{top['year']} alone supplies {top['n'] / full['n'] * 100:.0f}% of the qualifying sessions.")
        # ablation: every clause removed in turn (is each one doing any work?)
        parts = m.parts or [m]
        if len(parts) > 1 and o.event is None:
            for j, pj in enumerate(parts):
                rest = [p for i_, p in enumerate(parts) if i_ != j]
                true = np.ones(base.size, bool)
                known = np.ones(base.size, bool)
                for p_ in rest:
                    true &= p_.true
                    known &= p_.known
                b2 = o.valid & known & o.ds.pm
                e = EV.compare(dates, code, ret, true & b2, b2, d, reps=self.reps)
                alone = EV.compare(dates, code, ret, pj.true & o.valid & pj.known & o.ds.pm, o.valid & pj.known & o.ds.pm, d, robust=False)
                out["ablation"].append({"removed": pj.label, "n_without": e["n"], "probability_without": e["probability"], "lift_without": e["lift"], "p_value_without": e["p_value"],
                                        "n_alone": alone["n"], "lift_alone": alone["lift"],
                                        "contributes": None if e["lift"] is None or full["lift"] is None else bool(abs(full["lift"]) > abs(e["lift"]) + 1e-9)})
            idle = [a["removed"] for a in out["ablation"] if a["contributes"] is False]
            if idle:
                out["flags"].append("Removing " + "; ".join(f"'{x}'" for x in idle) + " does not weaken the result: " + ("that clause is" if len(idle) == 1 else "those clauses are") + " not adding evidence, only removing observations.")
        # neighbouring thresholds
        for pj in parts:
            s = pj.src or {}
            if "series" not in s or s["op"] == "==" or o.event is not None:
                continue
            ser: Series = s["series"]
            others = [p for p in parts if p is not pj]
            for mult in (0.75, 0.9, 1.1, 1.25):
                v = s["value"] * mult if s["value"] != 0 else (mult - 1)
                pr = Pred("_", s["op"], float(v), s.get("hi"))
                fin = np.isfinite(ser.x)
                true, known = pr.mask(ser.x) & fin, fin.copy()
                for p_ in others:
                    true &= p_.true
                    known &= p_.known
                b2 = o.valid & known & o.ds.pm
                e = EV.compare(dates, code, ret, true & b2, b2, d, robust=False)
                out["neighbours"].append({"clause": pj.label, "value": float(v), "n": e["n"], "probability": e["probability"], "lift": e["lift"]})
        nb = [r for r in out["neighbours"] if r["n"] >= 20 and r["lift"] is not None]
        if nb and full["lift"] is not None:
            same = sum(1 for r in nb if (r["lift"] > 0) == (full["lift"] > 0))
            out["neighbours_agree"] = [same, len(nb)]
            if same / len(nb) < 0.75:
                out["flags"].append("Neighbouring thresholds do not all point the same way: the exact cut-off matters.")
        halves = [r for r in out["periods"] if r["label"].endswith("half") and r["lift"] is not None]
        if len(halves) == 2 and (halves[0]["lift"] > 0) != (halves[1]["lift"] > 0):
            out["flags"].append("The two halves of the history point in opposite directions.")
        if full["n"] < 30:
            out["flags"].append(f"Only {full['n']} qualifying sessions.")
        ya = out.get("years_agree")
        s = [f"Across calendar years with five or more qualifying sessions, {ya[0]} of {ya[1]} point the same way as the whole period." if ya else "Too few qualifying sessions per year to compare calendar years."]
        s.append("Nothing here singles out a weakness." if not out["flags"] else "Weak points: " + " ".join(out["flags"]))
        out["summary"] = s
        return out

    def _validation(self, n, pa, ins) -> dict:
        o: Out = self.get(ins["outcome"])
        m: Mask = self.get(ins["condition"])
        ds = o.ds
        alpha = float(pa.get("alpha") or 0.05)
        q0, base0, code, ret = self._frame(o, m)  # already restricted to the pre-holdout sessions
        dates = np.array(ds.ft.dates)
        d = o.spec.direction if o.spec else "up"
        pre = np.flatnonzero(base0)
        tx = self._texts(o, m)
        out = {"kind": "validation", **tx, "alpha": alpha, "holdout": ds.holdout, "train": None, "validate": None, "test": None, "status": "not_estimable", "trials": 1, "p_adjusted": None, "reasons": [], "notes": []}
        if pre.size < 120:
            out["reasons"].append(f"Only {pre.size} evaluated sessions lie before the final test period: too few to split into training and validation.")
            out["summary"] = out["reasons"]
            return out
        cut = pre[int(pre.size * 0.75)]
        tr, va = np.zeros(base0.size, bool), np.zeros(base0.size, bool)
        tr[pre[pre < cut]] = True
        va[pre[pre >= cut]] = True
        e_tr = EV.compare(dates, code, ret, q0 & tr, base0 & tr, d, reps=max(self.reps, 1000), mean_diff=True)
        e_va = EV.compare(dates, code, ret, q0 & va, base0 & va, d, robust=False)
        key = self._trial("graph", o, m, e_tr, f"validation: {tx['condition']} -> {tx['outcome']}")
        trials = 1
        if self.scope:
            trials = max(1, LG.count(self.scope, self._holdout_symbol(ds.symbol)) + len({t["key"] for t in self.trials if t["symbol"] == self._holdout_symbol(ds.symbol)}))
        p = e_tr["p_value"] if e_tr["p_value"] is not None else None
        p_adj = None if p is None else min(1.0, p * trials)
        out.update(train={**e_tr, "start": str(dates[pre[0]]), "end": str(dates[cut - 1])}, validate={**e_va, "start": str(dates[cut]), "end": str(dates[pre[-1]])}, trials=trials, p_adjusted=p_adj, key=key)
        why = []
        if e_tr["n"] < 30:
            why.append(f"only {e_tr['n']} qualifying sessions in the training period (30 are needed)")
        if p is None:
            why.append("too few independent weeks in the training period for a dependence-robust test")
        elif p_adj >= alpha:
            why.append(f"not distinguishable from the baseline on the training period after correcting for the {trials} configuration{'s' if trials != 1 else ''} tried in this notebook (adjusted p {_fmt_p(p_adj)})")
        if e_va["n"] < 8:
            why.append(f"only {e_va['n']} qualifying sessions in the validation period")
        elif e_va["lift"] is None or e_tr["lift"] is None or (e_va["lift"] > 0) != (e_tr["lift"] > 0):
            why.append("the validation period points the other way")
        h = ds.holdout
        if why:
            out.update(status="not_a_candidate", reasons=why)
            out["summary"] = ["Not a candidate: " + "; ".join(why) + "." + (f" The final test period ({h['start']} to {h['end']}) was not examined and stays {h.get('label', 'untouched')}." if h else "")]
            return out
        if h is None:
            out.update(status="candidate_untested", reasons=["no final test period could be reserved (too little history)"])
            out["summary"] = ["A candidate on the training and validation periods, but there is no final test period to confirm it on."]
            return out
        te = (dates >= h["start"]) & (dates <= h["end"]) & ds.full
        if o.event is not None:
            ok_t = (o.base_code >= 0) & te
            q_t = o.event & ok_t & m.true if m is not None else o.event & ok_t
            b_t = ok_t if m is None else ok_t & m.known
        else:
            b_t = o.valid & m.known & te
            q_t = m.true & b_t
        if h.get("status") == "contaminated" or h.get("usable") is False and h.get("status") not in ("exposed", "sealed", "local"):
            out.update(status="candidate_untested", reasons=[LG.holdout_note(h)])
            out["summary"] = ["A candidate on the training and validation periods. " + LG.holdout_note(h)]
            return out
        e_te = EV.compare(dates, code, ret, q_t, b_t, d, reps=max(self.reps, 1000), mean_diff=True)
        same = e_te["lift"] is not None and e_tr["lift"] is not None and (e_te["lift"] > 0) == (e_tr["lift"] > 0)
        p1 = (e_te["p_value"] / 2) if (e_te["p_value"] is not None and same) else None
        out["test"] = {**e_te, "start": h["start"], "end": h["end"], "p_one_sided": p1}
        used = h
        if self.scope and h.get("id"):
            used = LG.use(h["id"], [key], self.run_id, f"graph validation: {tx['condition']}"[:160])
            for t in self.trials:
                if t["key"] == key:
                    t["end"], t["phase"] = h["end"], "validation"  # scored on the final test period: no longer exploration
        untouched = h.get("status") in ("sealed", "local")
        sig = next((t.get("signature") for t in self.trials if t["key"] == key), None)
        glob = registry_exposure(ds.symbol, h["start"], h["end"], self.scope, sig) if self.scope else {"configurations": 0, "related": 0, "notebooks": 0}
        out["holdout"], out["global_exposure"] = used, glob
        if e_te["n"] < 8:
            out.update(status="candidate_untested", reasons=[f"only {e_te['n']} qualifying sessions in the final test period: too few to confirm or reject"])
        elif p1 is not None and p1 < alpha:
            if not untouched:
                out["status"] = "holds_on_seen_data"
            elif glob["related"]:
                out["status"] = "confirmed_not_independent"
            else:
                out["status"] = "confirmed"
                if glob["configurations"]:
                    out["notes"].append(f"{glob['configurations']} unrelated configuration{'s' if glob['configurations'] != 1 else ''} in {glob['notebooks']} other notebook{'s' if glob['notebooks'] != 1 else ''} were evaluated on the same dates. "
                                        "They test different things, but the dates are not unseen by you.")
        else:
            out.update(status="not_confirmed", reasons=["the final test period does not confirm it" + (f" (one-sided p {_fmt_p(p1)})" if p1 is not None else " (it points the other way)" if not same else "")])
        label = {"confirmed": "Confirmed on the final test period, which nothing in this notebook or elsewhere in Strata had examined.",
                 "confirmed_not_independent": f"Holds on this notebook's final test period, but it is NOT an independent confirmation: {glob['related']} closely related configuration{'s' if glob['related'] != 1 else ''} "
                                              f"elsewhere in Strata already examined those dates ({glob['configurations']} configuration{'s' if glob['configurations'] != 1 else ''} in {glob['notebooks']} other notebook{'s' if glob['notebooks'] != 1 else ''} in all).",
                 "holds_on_seen_data": f"Holds on the final test period, but that period was {h.get('label', 'already seen')} before this test, so this is not a confirmation.",
                 "not_confirmed": "Not confirmed on the final test period.", "candidate_untested": "A candidate that could not be tested on the final period."}[out["status"]]
        out["summary"] = [label + f" Training: {_pct(e_tr['probability'])} of {e_tr['n']} against {_pct(e_tr['baseline_probability'])} (adjusted p {_fmt_p(p_adj)}). "
                          f"Final test ({h['start']} to {h['end']}): {_pct(e_te['probability'])} of {e_te['n']} against {_pct(e_te['baseline_probability'])}."]
        if self.scope and h.get("id"):
            out["notes"].append("The final test period has now been used once in this notebook. Testing further configurations on data that includes it will be recorded as contamination.")
        return out

    def _options_backtest(self, n, pa, ins) -> dict:
        from strata.research.hypothesis import structure_variants
        o: Out = self.get(ins["outcome"])
        m: Mask = self.get(ins["condition"])
        tx = self._texts(o, m)
        cell = self._cell_of(o, m)
        if cell is None:
            return {"kind": "strategies", **tx, "variants": [], "convertible": False,
                    "summary": ["This finding cannot be written as a strategy rule as it stands: " + self._unconvertible(o, m) + " No approximation is substituted."], "notes": []}
        sym = o.ds.symbol
        has_q, has_t = bool(self.store.option_days(sym)), bool(self.store.option_bar_days(sym)) if hasattr(self.store, "option_bar_days") else False
        variants, notes = structure_variants(sym, cell["preds"], cell["outcome"], has_q, has_t)
        variants = variants[:int(pa.get("max_variants") or 6)]
        q, base, code, ret = self._frame(o, m)
        return {"kind": "strategies", **tx, "variants": variants, "convertible": True, "cell": cell, "signals": int(q.sum()), "notes": notes, "window": _window(o.ds, base),
                "summary": [f"{len(variants)} option structures express this finding. None has been priced yet: a market pattern is not a strategy result until its trades are evaluated on observed option prices."]}

    # ----- converting a (condition, outcome) pair back into things other engines accept
    def _cell_of(self, o: Out, m: Mask | None) -> dict | None:
        """{"preds", "outcome"} when every clause can be read at the outcome's entry tick."""
        if m is None or o.event is not None or o.spec is None:
            return None
        preds = []
        for p in (m.parts or [m]):
            s = p.src or {}
            if p.pred is None:
                return None
            tick = s.get("tick") or "open"
            fixed = isinstance(p.pred, Pred) and feature_def(p.pred.feature, o.ds.symbol).kind != "intraday" and tick != "close"
            if not fixed and tick != o.entry:
                return None  # read at another clock time than the entry: the rule language cannot express that at the entry tick
            preds.append(p.pred.dump())
        return {"preds": preds, "outcome": o.spec.dump()}

    def _unconvertible(self, o: Out, m: Mask | None) -> str:
        if o.event is not None:
            return "its observation starts at an intraday event."
        return "one of its conditions is read at a different clock time from the entry, or is a negation or alternative that has no single rule form."

    def _study_of(self, o: Out, m: Mask | None) -> dict | None:
        if o.study is not None:
            return o.study
        c = self._cell_of(o, m)
        if c is None or o.spec.direction == "either":
            return None
        return to_study(o.ds.symbol, [Pred.load(p) for p in c["preds"]], o.spec)


def _is_cross(name: str) -> bool:
    return name.startswith("rs5_") or name.endswith("_ret_1") or name.endswith("_gap")


def _cmp_expr(e: str, pa: dict) -> str:
    if pa["op"] == "abs>=":
        return f"abs({e}) >= {float(pa['value']):g}"
    if pa["op"] == "between":
        return f"(({e}) >= {float(pa['value']):g}) and (({e}) < {float(pa['hi']):g})"
    return f"({e}) {pa['op']} {float(pa['value']):g}"


def _clause_text(s: Series, pa: dict) -> str:
    u = s.unit if s.unit in ("%", "x") else (" " + s.unit if s.unit else "")
    if pa["op"] == "between":
        return f"{s.label} is between {float(pa['value']):g}{u} and {float(pa['hi']):g}{u}"
    if pa["op"] == "abs>=":
        return f"{s.label} is {float(pa['value']):g}{u} or more in either direction"
    words = {">=": "at least", "<=": "at most", ">": "above", "<": "below", "==": "equal to"}[pa["op"]]
    return f"{s.label} is {words} {float(pa['value']):g}{u}"


def _window(ds: Dataset, mask: np.ndarray | None = None) -> list:
    """First and last session of a mask (default: the dataset's period)."""
    d = np.array(ds.ft.dates)[ds.pm if mask is None else mask]
    return [str(d[0]), str(d[-1])] if d.size else [None, None]


def _quant(x: np.ndarray) -> dict | None:
    if x.size < 4:
        return None
    return {"n": int(x.size), "mean": float(x.mean()), "p05": float(np.quantile(x, 0.05)), "p25": float(np.quantile(x, 0.25)), "p50": float(np.median(x)), "p75": float(np.quantile(x, 0.75)),
            "p95": float(np.quantile(x, 0.95)), "min": float(x.min()), "max": float(x.max()), "std": float(x.std(ddof=1))}


def _hist(a: np.ndarray, b: np.ndarray, bins: int = 31) -> dict | None:
    if a.size < 4:
        return None
    allv = np.r_[a, b] if b.size else a
    lo, hi = np.quantile(allv, [0.005, 0.995])
    if hi <= lo:
        hi = lo + 1e-6
    ha, edges = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    hb, _ = np.histogram(np.clip(b, lo, hi), bins=bins, range=(lo, hi)) if b.size else (np.zeros(bins), None)
    return {"edges": [float(x) for x in edges], "condition": [float(x) for x in ha / max(1, a.size)], "rest": [float(x) for x in hb / max(1, b.size)]}


def _prob_sentence(r: dict) -> list[str]:
    e = r["evidence"]
    if not e["n"]:
        return [f"No evaluated session ({r['window'][0]} to {r['window'][1]}, {e['evaluated']:,} sessions) satisfied: {r['condition']}."]
    s = (f"{r['symbol']}, sessions where {r['condition']}: the outcome ({r['outcome']}) occurred on {e['hits']} of {e['n']} ({_pct(e['probability'])}, 95% interval {_pct(e['ci95'][0])} to {_pct(e['ci95'][1])}), "
         f"against {_pct(e['rest_probability'])} of the other {e['rest_n']:,} evaluated sessions.")
    if r["thin"]:
        s += f" With fewer than {r['min_sample']} observations this cannot support a conclusion."
    elif e["p_value"] is None:
        s += " Too few independent weeks contain qualifying sessions for a dependence-robust test."
    elif e["p_value"] < 0.05:
        s += f" The difference is unlikely to be chance on its own (p {_fmt_p(e['p_value'])}, weeks resampled), before any correction for other configurations tried."
    else:
        s += f" That difference is within what chance produces (p {_fmt_p(e['p_value'])}, weeks resampled)."
    out = [s]
    if e.get("avg_end_return_pct") is not None and e.get("rest_avg_end_return_pct") is not None:
        out.append(f"Average return to the end of the horizon: {e['avg_end_return_pct']:+.2f}% on qualifying sessions, {e['rest_avg_end_return_pct']:+.2f}% on the others.")
    return out


# ----------------------------------------------------------------------------- registry hooks (kept here so graph trials carry a signature)
def signature(sig: dict) -> dict:
    """What a configuration is ABOUT, loosely: used to find closely related hypotheses."""
    feats = sorted({c.split("@")[0].split("(")[0] if "(" not in c.split("@")[0] else c.split("@")[0] for c in sig.get("clauses", [])})
    oc = sig.get("outcome") or {}
    return {"symbol": sig.get("symbol"), "features": feats, "outcome": f"{oc.get('entry')}|{oc.get('horizon')}|{oc.get('direction')}|{oc.get('measure')}", "controls": sig.get("controls") or []}


def registry_exposure(symbol: str, start: str, end: str, scope: str | None, sig: dict | None = None) -> dict:
    from strata.research import registry as RG
    return RG.exposure(symbol, start, end, exclude_scope=scope, signature=sig)


# ----------------------------------------------------------------------------- public entry points
def describe(graph: dict, store: MarketStore | None = None) -> dict:
    """What the graph will need and whether it is stored (shown before anything runs)."""
    store = store or MarketStore()
    chk = validate(graph)
    if chk["errors"]:
        return {"ok": False, "errors": chk["errors"]}
    run = Run(graph, store)
    need, problems = [], []
    for n in graph["nodes"]:
        if n["op"] != "data":
            continue
        sym = str(n["params"]["symbol"]).upper()
        try:
            ft = get_features(store, sym, tuple(x for x in run.cross if x != sym))
        except LookupError as e:
            problems.append(str(e))
            need.append({"what": "Daily bars", "symbol": sym, "ok": False, "sessions": 0})
            continue
        intr = ft.res[ft.res < 86400]
        need.append({"what": "Daily bars", "symbol": sym, "ok": True, "sessions": ft.n, "first": ft.dates[0], "last": ft.dates[-1]})
        for x in run.cross:
            if x == sym:
                continue
            xs = ft.series.get(x)
            need.append({"what": "Daily bars (read by a condition)", "symbol": x, "ok": xs is not None, "sessions": int(len(xs.days)) if xs is not None else 0})
            if xs is None:
                problems.append(f"No price history is stored for {x}.")
        ticks = sorted({str((m.get("params") or {}).get(k)) for m in graph["nodes"] for k in ("at", "entry") if (m.get("params") or {}).get(k) not in (None, "open", "close")})
        events = any(m["op"] in ("event", "sequence") for m in graph["nodes"])
        for t in ticks:
            _, ok = ft._tick(t)
            cnt = int(ok.sum())
            need.append({"what": f"Intraday bars that include {t} ET", "symbol": sym, "ok": cnt > 0, "sessions": cnt, "first": ft.dates[int(np.flatnonzero(ok)[0])] if cnt else None})
            if cnt == 0:
                problems.append(f"No stored session of {sym} has a bar boundary at {t} ET.")
        if events:
            fine = int((ft.res <= 300).sum())
            need.append({"what": "Intraday bars for event detection (5-minute or finer for sequences)", "symbol": sym, "ok": intr.size > 0, "sessions": int(intr.size), "fine_sessions": fine})
    return {"ok": not problems, "errors": [], "problems": problems, "required": need, "order": chk["order"], "known_at": chk["known_at"], "cross": list(run.cross),
            "nodes": [{"id": n["id"], "op": n["op"], "title": PRIMITIVES[n["op"]].title, "category": PRIMITIVES[n["op"]].category, "output": PRIMITIVES[n["op"]].output, "known_at": chk["known_at"].get(n["id"]),
                       "params": n.get("params") or {}, "inputs": n.get("inputs") or {}} for n in graph["nodes"]]}


def run_graph(graph: dict, store: MarketStore | None = None, scope: str | None = None, run_id: str | None = None, progress=None, should_cancel=None, reps: int = 400) -> dict:
    """Validate and execute. Every analysis node is reported, including the ones that found nothing."""
    chk = validate(graph)
    if chk["errors"]:
        raise GraphError(chk["errors"])
    store = store or MarketStore()
    run = Run(graph, store, scope, run_id, reps=reps)
    outs = [i for i in chk["order"] if chk["types"][i] == "result"]
    nodes_out, summary, limits = [], [], []
    for k, i in enumerate(chk["order"]):
        if should_cancel and should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()
        n = run.nodes[i]
        prim = PRIMITIVES[n["op"]]
        if progress:
            progress(0.95 * k / len(chk["order"]), f"{prim.title.lower()} ({i})")
        row = {"id": i, "op": n["op"], "title": prim.title, "category": prim.category, "output": prim.output, "known_at": chk["known_at"].get(i), "params": n.get("params") or {}, "inputs": n.get("inputs") or {},
               "status": "ok", "result": None, "info": None}
        try:
            v = run.get(i)
        except (LookupError, ValueError, KeyError, X.ExprError) as e:
            row.update(status="error", error=str(e)[:300])
            nodes_out.append(row)
            continue
        if isinstance(v, Dataset):
            intr = int((v.ft.res < 86400).sum())
            row["info"] = {"symbol": v.symbol, "sessions": int(v.pm.sum()), "first": _window(v)[0], "last": _window(v)[1], "stored_sessions": v.ft.n, "intraday_sessions": intr, "holdout": v.holdout, "note": v.note}
            if v.note:
                limits.append(v.note)
        elif isinstance(v, Series):
            x = v.x[v.ds.pm]
            fin = np.isfinite(x)
            row["info"] = {"label": v.label, "unit": v.unit, "available": int(fin.sum()), "missing": int((~fin).sum()), "median": float(np.median(x[fin])) if fin.any() else None,
                           "p05": float(np.quantile(x[fin], 0.05)) if fin.sum() >= 20 else None, "p95": float(np.quantile(x[fin], 0.95)) if fin.sum() >= 20 else None}
        elif isinstance(v, Mask):
            pm = v.ds.pm
            row["info"] = {"label": v.label, "true": int((v.true & pm).sum()), "known": int((v.known & pm).sum()), "unknown": int((~v.known & pm).sum()), "clauses": len(v.parts or [v])}
        elif isinstance(v, Out):
            pm = v.ds.pm
            row["info"] = {"label": v.label, "evaluated": int((v.valid & pm).sum()) if v.event is None else int(((v.base_code >= 0) & pm).sum()), "events": None if v.event is None else int((v.event & pm).sum()),
                           "not_evaluated": {k_: int(c) for k_, c in v.reasons.items()}}
            for why, c in v.reasons.items():
                if c:
                    limits.append(f"{c:,} session{'s' if c != 1 else ''} could not be observed: {why}.")
        elif isinstance(v, dict) and n["op"] in ("event", "sequence"):
            row["info"] = {"label": v["label"]}
        else:
            row["result"] = v
            summary += [s for s in (v.get("summary") or [])]
        nodes_out.append(row)
    failed = [r for r in nodes_out if r["status"] == "error"]
    recorded = None
    if scope and run.trials:
        by_sym: dict[str, list] = {}
        for t in run.trials:
            by_sym.setdefault(t.pop("symbol"), []).append(t)
        recorded = [LG.record(scope, s, ts, family=(graph.get("question") or "")[:120]) for s, ts in by_sym.items()]
    syms = sorted({str(n["params"]["symbol"]).upper() for n in graph["nodes"] if n["op"] == "data"})
    if progress:
        progress(1.0, "done")
    limits.append("These are frequencies and returns of the underlying. None of it is an option return; a pattern is not a strategy result until trades are evaluated on observed option prices.")
    return {"kind": "graph", "version": VERSION, "question": graph.get("question", ""), "title": graph.get("title", ""), "symbols": syms, "graph": graph, "nodes": nodes_out, "outputs": outs,
            "summary": summary, "limits": list(dict.fromkeys(limits)), "assumptions": graph.get("assumptions") or [], "unsupported": graph.get("unsupported") or [], "failed": len(failed),
            "trials": len({t["key"] for t in run.trials}), "ledger": recorded, "synthetic": any(store.is_synthetic(s) for s in syms)}


def run_batch(items: list[dict], store: MarketStore | None = None, scope: str | None = None, run_id: str | None = None, progress=None, should_cancel=None, title: str = "", what: str = "",
              include_holdout: bool = False) -> dict:
    """Several stored graphs evaluated side by side (a finding re-tested on a later period,
    or the same idea on other symbols). Every item is a recorded trial; p-values are
    corrected for all of them."""
    store = store or MarketStore()
    rows = []
    for k, it in enumerate(items):
        if should_cancel and should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()
        if progress:
            progress(0.95 * k / max(1, len(items)), it["label"][:80])
        g = dict(it["graph"])
        if include_holdout:
            g["include_holdout"] = True
        row = {"label": it["label"], "original": it.get("original"), "original_grade": it.get("original_grade"), "original_period": it.get("original_period"), "finding": it.get("finding"), "status": "ok"}
        try:
            res = run_graph(g, store, scope, run_id)
        except (GraphError, LookupError, ValueError) as e:
            row.update(status="error", error=str(e)[:240])
            rows.append(row)
            continue
        pr = next((n["result"] for n in res["nodes"] if n.get("result") and n["result"].get("kind") == "probability"), None)
        if pr is None:
            err = next((n.get("error") for n in res["nodes"] if n["status"] == "error"), "no result")
            row.update(status="error", error=str(err)[:240])
        else:
            e = pr["evidence"]
            row.update(symbol=pr["symbol"], condition=pr["condition"], outcome=pr["outcome"], window=pr["window"], n=e["n"], probability=e["probability"], rest_probability=e["rest_probability"], lift=e["lift_vs_rest"],
                       p_value=e["p_value"], ci95=e["ci95"], avg_end_return_pct=e["avg_end_return_pct"], rest_avg_end_return_pct=e["rest_avg_end_return_pct"], thin=e["n"] < 30)
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    for r, a in zip(ok, R.adjust([r.get("p_value") for r in ok], "holm")):
        r["p_adjusted"] = a
        o = r.get("original") or {}
        o_lift = None if o.get("probability") is None or o.get("rest_probability") is None else o["probability"] - o["rest_probability"]
        if r["thin"]:
            r["verdict"] = "too few sessions"
        elif what == "still_hold" and o_lift is not None and r["lift"] is not None:
            same = (r["lift"] > 0) == (o_lift > 0)
            r["verdict"] = ("holds" if same and r["p_value"] is not None and r["p_value"] < 0.05 else "same direction, not distinguishable" if same else
                            "reversed" if r["p_value"] is not None and r["p_value"] < 0.05 else "points the other way, not distinguishable")
        else:
            r["verdict"] = "distinguishable" if a is not None and a < 0.05 else "within chance"
    if what == "still_hold":
        h = sum(1 for r in ok if r.get("verdict") == "holds")
        summary = [f"{len(rows)} recorded finding{'s were' if len(rows) != 1 else ' was'} re-tested on the later period with unchanged conditions: {h} still hold{'s' if h == 1 else ''} on their own test, "
                   f"{sum(1 for r in ok if r.get('verdict', '').startswith('same direction'))} point the same way but are not distinguishable, {sum(1 for r in ok if 'other way' in r.get('verdict', '') or r.get('verdict') == 'reversed')} point the other way, "
                   f"and {sum(1 for r in ok if r.get('verdict') == 'too few sessions')} have too few sessions since then."]
    else:
        d = sum(1 for r in ok if r.get("verdict") == "distinguishable")
        summary = [f"{len(rows)} related configuration{'s were' if len(rows) != 1 else ' was'} evaluated; {d} {'is' if d == 1 else 'are'} distinguishable from the baseline after correcting for all of them."]
    notes = ["Each row is a new trial in this notebook's ledger.", "Underlying price behaviour only; none of this is an option return."]
    if include_holdout:
        notes.insert(0, "This used sessions the notebook was holding back as its final test period; the ledger records that they have now been examined.")
    if progress:
        progress(1.0, "done")
    return {"kind": "graph_batch", "title": title, "what": what, "rows": rows, "summary": summary, "notes": notes, "symbols": sorted({r["symbol"] for r in ok if r.get("symbol")}), "graph": {"question": title, "nodes": []},
            "nodes": [], "outputs": [], "limits": notes, "failed": len(rows) - len(ok), "trials": len(ok), "synthetic": False}


def save_graph_result(rid: str, res: dict) -> dict:
    if res.get("kind") == "graph_batch":
        from strata.api.results import dumps, run_dir
        d = run_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.json").write_text(dumps(res))
        return {"graph": True, "batch": len(res["rows"]), "failed": res["failed"], "symbols": res["symbols"], "headline": res["summary"][0][:300]}
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.json").write_text(dumps(res))
    first = next((n["info"] for n in res["nodes"] if n["op"] == "data" and n.get("info")), {}) or {}
    return {"graph": True, "nodes": len(res["nodes"]), "analyses": len(res["outputs"]), "failed": res["failed"], "symbols": res["symbols"], "start": first.get("first"), "end": first.get("last"),
            "synthetic": res.get("synthetic"), "headline": (res["summary"] or [""])[0][:300]}
