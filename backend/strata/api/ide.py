"""Backend of the Strategy IDE: the same strategy seen as rules, as structured text and as a
replayed session, kept consistent because all three are the one validated StrategyConfig.

  validate        schema and rule-language errors, each with the path of the field it is about
  edit_section    "replace this fixed stop with an ATR-based stop": an instruction applied to
                  ONE selected section. The rest of the strategy is returned byte-for-byte
                  unchanged, and the reply lists exactly which paths changed and what was
                  assumed. An instruction that cannot be expressed is refused, not approximated
  debug_session   the entry rule replayed tick by tick on one historical session: every
                  condition's state and values at every evaluation, optional watch
                  expressions, and a breakpoint (the first tick on which an expression holds)
  preview_rule    how often a rule was true historically, next to the rule as it is typed
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date

from strata.ai import nl_rules as NL
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.engine import expr as X
from strata.engine.models import BacktestConfig, StrategyConfig

MAX_TICKS = 420


# ----------------------------------------------------------------------------- validation
def validate(strategy: dict) -> dict:
    """{"ok", "errors": [{"path", "message"}], "expressions": [{"path", "expr", "ok", "error"}]}"""
    from pydantic import ValidationError
    errors, exprs = [], []
    for path, src in expressions_of(strategy):
        bad = X.validate(src) if isinstance(src, str) and src.strip() else ["empty expression"]
        exprs.append({"path": path, "expr": src, "ok": not bad, "error": bad[0] if bad else None, "needs_intraday": (not bad) and X.needs_intraday(src)})
        if bad:
            errors.append({"path": path, "message": bad[0]})
    sz = strategy.get("sizing") or {}
    if sz.get("method") == "expr" and isinstance(sz.get("expr"), str) and sz["expr"].strip() and not X.validate(sz["expr"]) and X.uses_position(sz["expr"]):
        errors.append({"path": "sizing.expr", "message": "the sizing expression is evaluated before the position exists, so it cannot read entry.*, position.* or group.*"})
    try:
        StrategyConfig.model_validate(strategy)
    except ValidationError as e:
        for er in e.errors()[:20]:
            p = ".".join(str(x) for x in er["loc"])
            msg = er["msg"].removeprefix("Value error, ")
            if not any(x["message"] == msg for x in errors):
                errors.append({"path": p, "message": msg})
    except Exception as e:  # noqa: BLE001
        errors.append({"path": "", "message": str(e)[:300]})
    full = None
    if not errors:
        try:
            full = StrategyConfig.model_validate(strategy).model_dump(mode="json")
        except Exception:  # noqa: BLE001
            full = None
    return {"ok": not errors, "errors": errors, "expressions": exprs, "full": full}


def expressions_of(strategy: dict) -> list[tuple[str, str]]:
    """(path, expression) for every rule expression in a strategy document."""
    out: list[tuple[str, str]] = []

    def tree(node, path):
        if not isinstance(node, dict):
            return
        if node.get("kind") == "cond" and node.get("type") == "expr":
            out.append((f"{path}.params.expr", str((node.get("params") or {}).get("expr", ""))))
        for i, c in enumerate(node.get("children") or []):
            tree(c, f"{path}.children[{i}]")

    e = strategy.get("entry") or {}
    tree(e.get("conditions"), "entry.conditions")
    for i, st in enumerate(e.get("sequence") or []):
        out.append((f"entry.sequence[{i}].when", str(st.get("when", ""))))
    for i, a in enumerate(e.get("adds") or []):
        out.append((f"entry.adds[{i}].when", str(a.get("when", ""))))
    for i, sy in enumerate(e.get("systems") or []):
        tree(sy.get("conditions"), f"entry.systems[{i}].conditions")
    x = strategy.get("exits") or {}
    tree(x.get("custom"), "exits.custom")
    for i, g in enumerate(x.get("stages") or []):
        if g.get("when"):
            out.append((f"exits.stages[{i}].when", str(g["when"])))
    if (strategy.get("sizing") or {}).get("method") == "expr":
        out.append(("sizing.expr", str((strategy.get("sizing") or {}).get("expr") or "")))
    return out


# ----------------------------------------------------------------------------- paths into the document
def _tokens(path: str) -> list:
    out = []
    for part in [p for p in path.split(".") if p]:
        m = re.fullmatch(r"([^\[\]]+)((?:\[\d+\])*)", part)
        if not m:
            raise ValueError(f"not a valid path: {path!r}")
        out.append(m.group(1))
        out += [int(i) for i in re.findall(r"\[(\d+)\]", m.group(2))]
    return out


def get_at(doc, path: str):
    cur = doc
    for t in _tokens(path):
        cur = cur[t] if isinstance(t, int) else (cur or {}).get(t)
        if cur is None:
            return None
    return cur


def set_at(doc, path: str, value) -> None:
    toks = _tokens(path)
    cur = doc
    for t, nxt in zip(toks[:-1], toks[1:]):
        if isinstance(t, int):
            cur = cur[t]
        else:
            if cur.get(t) is None:
                cur[t] = [] if isinstance(nxt, int) else {}
            cur = cur[t]
    last = toks[-1]
    if isinstance(last, int):
        cur[last] = value
    else:
        cur[last] = value


def diff_paths(a, b, base: str = "") -> list[str]:
    """Paths whose values differ between two documents."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("kind") != b.get("kind") and ("kind" in a or "kind" in b):
            return [base]  # a rule replaced by a different kind of node is one change, not one per field
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f"{base}.{k}" if base else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_paths(x, y, f"{base}[{i}]")
        return out
    return [] if a == b else [base]


# ----------------------------------------------------------------------------- editing one section
ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}


def _expr_leaf(expr: str, label: str = "") -> dict:
    return {"id": "ide" + str(abs(hash(expr)) % 10 ** 8), "kind": "cond", "type": "expr", "params": {"expr": expr, **({"label": label} if label else {})}}


def _num(text: str, pat: str, default: float | None = None) -> float | None:
    m = re.search(pat, text)
    return float(m.group(1)) if m else default


def edit_section(strategy: dict, path: str, instruction: str, tz_offset: int | None = None) -> dict:
    """Apply an instruction to the selected section only. Returns {"strategy", "changed",
    "notes", "unsupported", "applied"}; nothing outside the paths in `changed` differs."""
    s = copy.deepcopy(strategy)
    low = instruction.lower().strip()
    notes: list[str] = []
    sel = get_at(s, path) if path else s
    if path and sel is None and not path.startswith(("exits", "entry", "sizing", "hedge")):
        raise ValueError(f"nothing is selected at {path!r}")
    N = r"(\d+(?:\.\d+)?)"
    applied = None

    # ----- an ATR-based stop / target instead of (or next to) the selected premium rule
    if re.search(r"\batr\b", low) and re.search(r"\bstop\b|\bstop[- ]loss\b", low) and path.startswith("exits"):
        k = _num(low, rf"{N}\s*(?:x|times|atrs?)\b") or _num(low, rf"(?:x|times)\s*{N}")
        per = int(_num(low, rf"{N}[- ]?(?:day|period|session)") or _num(low, rf"atr\s*\(?\s*{N}") or 14)
        if k is None:
            k = 1.5
            notes.append("No multiple was given, so 1.5 ATR is used. Edit `exits.underlying.stop_atr` to change it.")
        ex = s.setdefault("exits", {})
        u = ex.setdefault("underlying", {})
        u.update(enabled=True, stop_atr=float(k), atr_period=per)
        if re.search(r"replace|instead|swap|rather than|convert|turn", low) or path.endswith("stop_loss"):
            ex.setdefault("stop_loss", {}).update(enabled=False)
            notes.append("The premium stop is switched off, as asked. The position now exits when the underlying moves that many ATRs against it.")
        notes.append(f"The stop is {k:g} x the {per}-day average true range as it stood at the entry (fixed for the life of the position), measured on the underlying. An entry whose ATR cannot be computed is skipped and reported.")
        applied = "atr_stop"
    elif re.search(r"\batr\b", low) and re.search(r"\btarget\b|take[- ]profit|profit[- ]target", low) and path.startswith("exits"):
        k = _num(low, rf"{N}\s*(?:x|times|atrs?)\b") or 2.0
        per = int(_num(low, rf"{N}[- ]?(?:day|period|session)") or 14)
        ex = s.setdefault("exits", {})
        ex.setdefault("underlying", {}).update(enabled=True, target_atr=float(k), atr_period=per)
        if re.search(r"replace|instead|swap|rather than|convert|turn", low) or path.endswith("profit_target"):
            ex.setdefault("profit_target", {}).update(enabled=False)
            notes.append("The premium target is switched off, as asked.")
        notes.append(f"The target is {k:g} x ATR({per}) in the position's favour, measured on the underlying and fixed at the entry.")
        applied = "atr_target"
    elif re.search(r"trail", low) and path.startswith("exits") and re.search(r"underlying|price|spot|stock|index", low):
        pct = _num(low, rf"{N}\s*%")
        k = _num(low, rf"{N}\s*(?:x|times|atrs?)\b") if pct is None else None
        if pct is None and k is None:
            raise ValueError("A trailing threshold needs a distance: say a percentage (\"trail the underlying by 1%\") or a number of ATRs.")
        u = s.setdefault("exits", {}).setdefault("underlying", {})
        u.update(enabled=True, **({"trail_pct": pct} if pct is not None else {"trail_atr": k}))
        notes.append("The exit follows the best underlying price since entry, so it is also an initial stop at the same distance. It does not apply to a position that is delta-neutral at entry.")
        applied = "trail"
    # ----- partial profit: "take half off at 50%"
    elif path.startswith("exits") and re.search(r"(?:take|close|sell|scale out)\b.*\b(half|a third|a quarter|\d+\s*%)\b.*\bat\b.*?\d", low):
        frac = {"half": 50.0, "a third": 33.34, "a quarter": 25.0}.get(re.search(r"half|a third|a quarter", low).group(0)) if re.search(r"half|a third|a quarter", low) else _num(low, rf"{N}\s*%")
        tgt = _num(low, rf"\bat\b\D{{0,12}}{N}\s*%")
        if frac is None or tgt is None:
            raise ValueError("Say how much to close and at what profit, for example \"take half off at 50% and move the stop to breakeven\".")
        stage = {"name": f"partial at {tgt:g}%", "profit_pct": tgt, "close_pct": frac, **({"then_stop": "breakeven"} if re.search(r"break[- ]?even", low) else {})}
        s.setdefault("exits", {}).setdefault("stages", []).append(stage)
        notes.append(f"Closes {frac:g}% of the contracts the position was opened with once its total P&L reaches {tgt:g}% of the premium" + ("; the rest then stops out if the position's total P&L returns to zero." if stage.get("then_stop") else "."))
        applied = "stage"
    # ----- restrict a condition (or the entry) to a time window
    elif path.startswith("entry") and re.search(r"between|from\b.*\b(?:to|until)\b|only (?:allow|enter|trade)|window", low) and len(NL.find_times(instruction, _zone(low, tz_offset))) >= 2:
        ts = NL.find_times(instruction, _zone(low, tz_offset))[:2]  # a zone named once ("between 9:30 and 11:00 Eastern") applies to both times
        a, b = sorted([ts[0]["minutes"], ts[1]["minutes"]])
        win = {"id": f"win{a}{b}", "kind": "cond", "type": "time_window", "params": {"start": cal.fmt_hhmm(a), "end": cal.fmt_hhmm(b)}}
        node = sel if isinstance(sel, dict) and sel.get("kind") in ("cond", "group") else None
        if node is not None and path != "entry.conditions":
            set_at(s, path, {"id": "g" + win["id"], "kind": "group", "op": "and", "children": [node, win]})
            notes.append(f"Only the selected condition is restricted: it now counts between {cal.fmt_hhmm(a)} and {cal.fmt_hhmm(b)} New York time. Other entry rules are unchanged.")
        else:
            root = s.setdefault("entry", {}).setdefault("conditions", {"kind": "group", "op": "and", "children": []})
            root.setdefault("children", []).append(win)
            notes.append(f"Entries are only allowed between {cal.fmt_hhmm(a)} and {cal.fmt_hhmm(b)} New York time.")
        if any(t.get("assumed_tz") for t in ts):
            notes.append("No time zone was stated; the times were read in the zone set in Settings and converted to New York time.")
        applied = "time_window"
    # ----- confirmation from another symbol
    elif path.startswith("entry") and re.search(r"confirm\w*|requir\w+|only (?:if|when)|add a (?:rule|condition|filter)", low) and _other_symbol(instruction, strategy.get("symbol", "")):
        sym = _other_symbol(instruction, strategy.get("symbol", ""))
        expr, label, assumed = _confirmation(low, sym)
        if X.validate(expr):
            raise ValueError(f"I could not write that confirmation as a rule: {X.validate(expr)[0]}")
        root = s.setdefault("entry", {}).setdefault("conditions", {"kind": "group", "op": "and", "children": []})
        target = sel if (isinstance(sel, dict) and sel.get("kind") == "group" and path != "entry") else root
        target.setdefault("children", []).append(_expr_leaf(expr, label))
        if assumed:
            notes.append(f"\"{sym} confirmation\" was not specified further, so it is taken as: {label}. Edit the expression `{expr}` to mean something else.")
        applied = "confirmation"
    # ----- move a scale-in
    elif path.startswith("entry") and re.search(r"scale[- ]?in|\badd\b|\blot\b", low) and re.search(rf"{N}\s*%", low) and (s.get("entry") or {}).get("adds") is not None:
        adds = s["entry"]["adds"]
        k = next((i for w, i in ORD.items() if re.search(rf"\b{w}\b", low)), None)
        m_idx = re.search(r"adds\[(\d+)\]", path)
        k = int(m_idx.group(1)) if m_idx else k
        if k is None or k >= len(adds):
            raise ValueError(f"The strategy has {len(adds)} scale-in rule{'s' if len(adds) != 1 else ''}; say which one (first, second, ...) or select it.")
        pct = _num(low, rf"{N}\s*%")
        down = not re.search(r"\b(?:rise|rall\w+|gain|higher|increase|up)\b", low)
        ref = 1.0
        if re.search(r"further|more|another|additional|beyond", low) and k > 0:
            m = re.fullmatch(r"\s*price\s*(<=|>=)\s*entry\.underlying_price\s*\*\s*([\d.]+)\s*", adds[k - 1].get("when", ""))
            if not m:
                raise ValueError("\"A further\" decline is measured from the scale-in before it, and that rule is not a simple percentage of the entry price, so there is no defined starting point. Write the expression directly.")
            ref = float(m.group(2))
            notes.append(f"\"Further\" is measured from the scale-in before it, which triggers at {abs(1 - ref) * 100:g}% {'below' if ref < 1 else 'above'} the first lot's entry price.")
        lvl = ref - pct / 100 if down else ref + pct / 100
        adds[k]["when"] = f"price {'<=' if down else '>='} entry.underlying_price * {lvl:.6g}"
        adds[k]["text"] = f"add after the underlying is {abs(1 - lvl) * 100:g}% {'below' if lvl < 1 else 'above'} the first lot's entry price"
        notes.append(f"Scale-in {k + 1} now triggers when the underlying is {abs(1 - lvl) * 100:g}% {'below' if lvl < 1 else 'above'} the first lot's entry price.")
        applied = "scale_in"
    # ----- size by ATR risk
    elif path.startswith("sizing") and re.search(r"\batr\b|volatil", low):
        pct = _num(low, rf"{N}\s*%") or 1.0
        s["sizing"] = {**(s.get("sizing") or {}), "method": "atr_risk", "pct": pct, "atr_multiple": _num(low, rf"{N}\s*(?:x|times|atrs?)\b") or 1.0}
        notes.append(f"Each position risks {pct:g}% of equity for an adverse move of {s['sizing']['atr_multiple']:g} ATR, using the position's delta at entry. An entry whose delta or ATR is unknown is skipped.")
        applied = "atr_sizing"
    elif path.startswith(("hedge", "exits", "sizing")) is False and re.search(r"delta[- ]hedg", low) or (path.startswith("hedge") and re.search(r"hedg", low)):
        mins = int(_num(low, rf"every\s+{N}\s*min") or 30)
        s["hedge"] = {"enabled": True, "rebalance_minutes": mins, "band": _num(low, rf"band\D{{0,10}}{N}") or 5.0}
        notes.append(f"Shares of the underlying are held against the position's option delta and rebalanced every {mins} minutes. Share trades pay the stock slippage and commission in the execution settings.")
        applied = "hedge"
    if applied is None:
        return {"strategy": strategy, "changed": [], "notes": [], "applied": None,
                "unsupported": [f"I could not apply \"{instruction.strip()}\" to {path or 'the strategy'} with the built-in editor. Nothing was changed. Supported here: an ATR-based stop or target, a trailing threshold on the underlying, "
                                "a partial exit, a time window on a condition, confirmation from another symbol, moving a scale-in, ATR-based sizing and a delta hedge. Anything else can be typed directly in the structured view."]}
    v = validate(s)
    if not v["ok"]:
        raise ValueError("That edit would make the strategy invalid: " + "; ".join(f"{e['path']}: {e['message']}" for e in v["errors"][:3]))
    return {"strategy": s, "changed": diff_paths(strategy, s), "notes": notes, "unsupported": [], "applied": applied}


def _zone(low: str, default: int | None) -> int | None:
    """Minutes to add to reach New York time for a zone named anywhere in the instruction."""
    m = re.search(r"\b(eastern|new york|est|edt|et|central|cst|cdt|ct|mountain|mst|mdt|mt|pacific|pst|pdt|pt)\b", low)
    return NL.TZ_OFFSET.get(m.group(1), 0 if m.group(1) == "new york" else default) if m else default


def _other_symbol(text: str, own: str) -> str | None:
    for tok in re.findall(r"\b[A-Z]{2,5}\b", text):
        if tok != own.upper() and tok not in NL.NOT_UNDERLYING and tok not in ("ATR", "AND", "ONLY", "ADD", "THE", "NOT", "ET", "PT") and (tok in NL.TICKERS or X.is_ticker(tok)):
            return tok
    return None


def _confirmation(low: str, sym: str) -> tuple[str, str, bool]:
    """(expression, wording, was the meaning assumed)."""
    if re.search(r"above (?:its |the )?vwap", low):
        return f"{sym}.price > {sym}.vwap", f"{sym} is above its session VWAP", False
    if re.search(r"below (?:its |the )?vwap", low):
        return f"{sym}.price < {sym}.vwap", f"{sym} is below its session VWAP", False
    if re.search(r"(?:up|green|higher|positive) on the day|above (?:its |the )?(?:previous|prior) close", low):
        return f"{sym}.change_pct > 0", f"{sym} is above its previous close", False
    if re.search(r"(?:down|red|lower|negative) on the day|below (?:its |the )?(?:previous|prior) close", low):
        return f"{sym}.change_pct < 0", f"{sym} is below its previous close", False
    if re.search(r"not (?:broken|made) (?:a |its )?(?:new )?(?:session )?high|below (?:its |the )?(?:previous|prior)[- ]day high", low):
        return f"{sym}.high <= {sym}.prev_high", f"{sym} has not traded above its previous session's high", False
    if re.search(r"below (?:its |the )?open|negative|bearish|weak", low):
        return f"{sym}.from_open_pct < 0", f"{sym} is below its own open", not re.search(r"below (?:its |the )?open", low)
    return f"{sym}.from_open_pct > 0", f"{sym} is above its own open", not re.search(r"above (?:its |the )?open", low)


# ----------------------------------------------------------------------------- the condition debugger
def debug_session(strategy: dict, day: date, store: MarketStore | None = None, watch: list[str] | None = None, break_when: str | None = None, execution: dict | None = None) -> dict:
    """Replay the entry rule on one session, tick by tick."""
    from strata.engine.engine import Backtester, DataUnavailable
    from strata.research.explain_day import _Recorder, explain_day
    store = store or MarketStore()
    watch = [w for w in (watch or []) if isinstance(w, str) and w.strip()][:8]
    errs = {w: X.validate(w) for w in watch + ([break_when] if break_when else [])}
    bad = {w: e[0] for w, e in errs.items() if e}
    if bad:
        raise ValueError("; ".join(f"{w}: {e}" for w, e in bad.items()))
    cfg = BacktestConfig.model_validate({"strategies": [strategy], "start": day.isoformat(), "end": day.isoformat(), "execution": {"signal_interval_minutes": 1, **(execution or {})}})
    base = explain_day(cfg, day, store)
    if not base.get("trading_day") or base.get("error") or not base.get("ticks"):
        return {**base, "timeline": [], "watch": [], "breakpoint": None}

    class _Dbg(_Recorder):
        def on_start(self, ctx):
            super().on_start(ctx)
            self.rows: list[dict] = []

        def on_snapshot(self, ctx):
            n = len(self.ticks)
            super().on_snapshot(ctx)
            if len(self.ticks) == n:
                return
            ev = X.Evaluator(ctx)
            vals = {}
            for w in watch:
                try:
                    v = ev.eval(w)
                    vals[w] = None if v is None else (bool(v) if isinstance(v, bool) else round(float(v), 6))
                except X.ExprError:
                    vals[w] = None
            hit = None
            if break_when:
                try:
                    hit = ev.eval(break_when) is True
                except X.ExprError:
                    hit = None
            self.rows.append({"watch": vals, "break": hit})

    rec = _Dbg(cfg.strategies[0], 0)
    seen: set[str] = set()
    for w in watch + ([break_when] if break_when else []):
        seen |= X.symbols_used(w)
    bt = Backtester(cfg, store, [rec], record_paths=False, signals_only=True, extra_symbols=seen)
    try:
        bt.run()
    except DataUnavailable as e:
        return {**base, "timeline": [], "watch": [], "breakpoint": None, "error": str(e)}
    step = max(1, len(rec.ticks) // MAX_TICKS + (1 if len(rec.ticks) % MAX_TICKS and len(rec.ticks) > MAX_TICKS else 0))
    first_break = next((i for i, r in enumerate(rec.rows) if r["break"]), None)
    keep = sorted(set(range(0, len(rec.ticks), step)) | ({first_break} if first_break is not None else set()) | {i for i, t in enumerate(rec.ticks) if t["passed"]})
    timeline = [{"i": i, "time": cal.fmt_hhmm(rec.ticks[i]["minute"]), "t": rec.ticks[i]["t"], "price": round(rec.ticks[i]["price"], 4), "passed": rec.ticks[i]["passed"], "na": rec.ticks[i]["na"],
                 "items": [{"text": it["text"], "passed": it["passed"], "kind": it["kind"]} for it in rec.ticks[i]["items"] if it["kind"] != "group"], "watch": rec.rows[i]["watch"], "break": rec.rows[i]["break"]} for i in keep]
    bp = None
    if break_when:
        bp = {"expr": break_when, "hit": first_break is not None, "time": cal.fmt_hhmm(rec.ticks[first_break]["minute"]) if first_break is not None else None, "index": first_break,
              "note": None if first_break is not None else "The breakpoint expression was never true on this session (or could not be evaluated)."}
    return {**{k: base[k] for k in ("date", "trading_day", "symbol", "resolution", "evaluations", "signal", "summary", "rules", "excluded", "sequence", "stages") if k in base},
            "timeline": timeline, "watch": watch, "breakpoint": bp, "sampled_every": step}


def preview_rule(symbol: str, expr: str, at: str = "open", store: MarketStore | None = None) -> dict:
    """How often a rule was true on the stored history, read at one tick."""
    import numpy as np

    from strata.research.features import get_features
    bad = X.validate(expr)
    if bad:
        return {"ok": False, "error": bad[0]}
    if X.uses_position(expr):
        return {"ok": True, "position": True, "note": "This rule reads an open position, so it has no value until a trade exists. Use the debugger on a session with a trade to see it."}
    intr = X.needs_intraday(expr)
    tick = at if (not intr or at not in ("open", "close")) else "10:30"
    store = store or MarketStore()
    try:
        ft = get_features(store, symbol.upper(), tuple(sorted(X.symbols_used(expr) - {symbol.upper()})))
    except LookupError as e:
        return {"ok": False, "error": str(e)}
    if ft.missing_cross:
        return {"ok": False, "error": f"No price history is stored for {', '.join(ft.missing_cross)}."}
    x = ft.engine_values(expr, tick, cache=True)
    fin = np.isfinite(x)
    out = {"ok": True, "at": tick, "evaluable": int(fin.sum()), "sessions": int(ft.n), "needs_intraday": intr, "moved_tick": tick != at}
    vals = x[fin]
    if vals.size == 0:
        return {**out, "true": 0, "share": None, "note": "The rule could not be evaluated on any stored session at this time" + (" (it needs intraday bars)." if intr else ".")}
    boolean = set(np.unique(vals)) <= {0.0, 1.0}
    if boolean:
        t = fin & (x > 0.5)
        dates = [ft.dates[i] for i in np.flatnonzero(t)[-5:]][::-1]
        return {**out, "kind": "condition", "true": int(t.sum()), "share": float(t.sum() / fin.sum()), "recent": dates, "first": ft.dates[int(np.flatnonzero(fin)[0])], "last": ft.dates[int(np.flatnonzero(fin)[-1])]}
    return {**out, "kind": "value", "median": float(np.median(vals)), "p05": float(np.quantile(vals, 0.05)), "p95": float(np.quantile(vals, 0.95)), "latest": float(vals[-1]), "latest_date": ft.dates[int(np.flatnonzero(fin)[-1])]}


def compact(strategy: dict) -> dict:
    """The form the structured view shows: fields at their defaults are left out, EXCEPT the ones
    that say what the strategy does (the legs' side / right / expiration / strike, every exit that
    is switched on, the sizing rule, how often it may enter). Validating it again reproduces the
    full configuration exactly, and the function checks that before returning it."""
    full = StrategyConfig.model_validate(strategy)
    f = full.model_dump(mode="json")
    c = json.loads(full.model_dump_json(exclude_defaults=True))

    def pick(cn: dict, fn: dict, keep=()) -> dict:  # the model's own field order; a key is shown if it differs from its default or is essential
        return {k: (cn[k] if k in cn else fn[k]) for k in fn if k in cn or k in keep}

    def tree(cn, fn):
        if not isinstance(fn, dict):
            return cn
        cn = cn or {}
        group = fn.get("kind") == "group"
        out = pick(cn, fn, ("kind",) + (("op",) if group and len(fn.get("children") or []) > 1 else ()) + (() if group else ("type", "params")))
        out.pop("children", None)
        out.pop("id", None)  # a rule's id only labels it in explanations; it is not part of what the strategy does
        if fn.get("children"):
            kids = cn.get("children") or [{} for _ in fn["children"]]
            out["children"] = [tree(x, y) for x, y in zip(kids, fn["children"])]
        return out

    out = pick(c, f, ("name", "symbol", "legs", "entry", "exits", "sizing"))
    out["legs"] = []
    for cl, fl in zip(c.get("legs") or [{} for _ in f["legs"]], f["legs"]):
        leg = pick(cl, fl, ("id", "side", "right", "quantity", "dte", "strike") if fl.get("kind", "option") == "option" else ("id", "side", "kind", "quantity"))
        if "strike" in leg:
            leg["strike"] = pick(cl.get("strike") or {}, fl["strike"], ("method", "value"))
        out["legs"].append(leg)
    ce, fe = c.get("entry") or {}, f["entry"]
    ent = pick(ce, fe, ("conditions", "max_entries_per_day", "max_open_positions"))
    ent["conditions"] = tree(ce.get("conditions"), fe["conditions"])
    if fe.get("systems"):
        ent["systems"] = [{**pick(x, y, ("name",)), "conditions": tree(x.get("conditions"), y["conditions"])} for x, y in zip(ce.get("systems") or [{} for _ in fe["systems"]], fe["systems"])]
    out["entry"] = ent
    cx, fx = c.get("exits") or {}, f["exits"]
    ex = {}
    for k, fb in fx.items():
        if isinstance(fb, dict) and "enabled" in fb:
            if fb["enabled"]:  # an exit that is switched on is always shown in full, default or not
                ex[k] = {kk: vv for kk, vv in fb.items() if vv is not None or kk in (cx.get(k) or {})}
            elif k in cx:
                ex[k] = cx[k]
        elif k == "custom" and fb:
            ex[k] = tree(cx.get(k), fb)
        elif k in cx:
            ex[k] = cx[k]
    out["exits"] = ex
    fs, cs = f["sizing"], c.get("sizing") or {}
    out["sizing"] = pick(cs, fs, ("method",) + {"fixed": ("contracts",), "pct_equity": ("pct",), "pct_risk": ("pct",), "atr_risk": ("pct", "atr_multiple", "atr_period"), "expr": ("expr",)}.get(fs["method"], ("contracts", "pct")))
    if _no_rule_ids(StrategyConfig.model_validate(out).model_dump(mode="json")) != _no_rule_ids(f):
        return f  # never show a shortened form that does not mean the same strategy
    return out


def _no_rule_ids(doc: dict) -> dict:
    d = copy.deepcopy(doc)

    def strip(n):
        if isinstance(n, dict):
            n.pop("id", None)
            for c in n.get("children") or []:
                strip(c)

    strip((d.get("entry") or {}).get("conditions"))
    for sy in (d.get("entry") or {}).get("systems") or []:
        strip(sy.get("conditions"))
    strip((d.get("exits") or {}).get("custom"))
    return d


def expand(strategy: dict) -> dict:
    return StrategyConfig.model_validate(strategy).model_dump(mode="json")


def from_interpretation(r: dict, store: MarketStore | None = None) -> dict:
    """The notebook interpreter's reading of a plain-English description -> a strategy document, with what was
    stated, assumed, missing and not understood. Nothing is run."""
    from strata.ai.draft import compile_draft
    if r.get("action") != "new" or r.get("draft") is None:
        return {"understood": False, "note": "That reads as a research question, not a strategy. Describe the position to open and when, for example: "
                                             "buy a 0DTE at-the-money SPY call at 10:00 when SPY is below its open; exit at +50% or -30%."}
    d = r["draft"]
    c = compile_draft(d, store or MarketStore())
    if not c.get("strategy"):
        return {"understood": False, "note": "; ".join(c.get("errors") or c.get("missing") or ["the description has no option position to open"]), "missing": c.get("missing") or []}
    return {"understood": True, "strategy": compact(c["strategy"]), "items": c["items"], "assumptions": c["assumptions"], "missing": c["missing"], "errors": c["errors"], "unsupported": list(d.unsupported or []), "notes": list(d.notes or [])}


def to_text(strategy: dict) -> str:
    """The structured view: the strategy as stable, readable JSON."""
    return json.dumps(strategy, indent=2, sort_keys=False)
