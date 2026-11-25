"""Forward validation: freeze a configuration, then judge it only on sessions that did not
exist when it was frozen.

A backtest can always be made to look good, because its author has seen the data. The one
test that cannot be tuned is the future. An EXPERIMENT here is

  frozen      the full configuration is stored with its hash at activation. Nothing can
              edit it. A change is a NEW experiment (version + 1) with its own activation
              date; the old one keeps running as it was
  dated       the activation date is the last session stored when it was frozen. Only
              sessions AFTER that date count as forward evidence, so nothing can be
              backdated into it
  paper only  it is evaluated by the same deterministic engines on observed prices as
              they are collected. There is no order entry anywhere in Strata

`evaluate` runs the frozen configuration over the forward sessions stored so far and
appends a report (the history of reports is kept): forward signals, simulated fills,
forward P&L, signals that could not be filled and why, how much of the forward window has
the data the configuration needs, and how the forward numbers compare with the historical
benchmark recorded at activation - with the plain statement that a handful of forward
observations cannot confirm or reject anything yet.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date

from strata.api import db
from strata.data.store import MarketStore
from strata.research import robust as R

MIN_FORWARD = 10  # below this many forward observations no comparison is drawn


def _canon(x):
    """Numbers as JSON means them: 2.0 and 2 are the same value. A configuration that went through a browser
    (where every number is a double and 2.0 is written "2") must hash exactly as it did when it was frozen."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return int(x) if x == int(x) and abs(x) < 2 ** 53 else x
    return x


def _hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(_canon(cfg), sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:16]


def _last_session(store: MarketStore, symbol: str) -> str:
    try:
        d = store.read_daily(symbol)
        if not d.is_empty():
            return str(d["date"][-1])
    except Exception:  # noqa: BLE001
        pass
    return date.today().isoformat()


def _pub(r: dict | None) -> dict | None:
    if r is None:
        return None
    with db.connect() as con:
        ev = con.execute("SELECT id, evaluated_at, through_date, report FROM forward_evaluations WHERE experiment_id=? ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
        n = con.execute("SELECT COUNT(*) FROM forward_evaluations WHERE experiment_id=?", (r["id"],)).fetchone()[0]
    out = dict(r)
    out["evaluations"] = int(n)
    out["latest"] = (json.loads(ev["report"]) | {"evaluated_at": ev["evaluated_at"]}) if ev and ev["report"] else None
    return out


def create(kind: str, symbol: str, config: dict, name: str, benchmark: dict | None = None, assumptions: dict | None = None, session_id: str | None = None, source_run_id: str | None = None,
           parent_id: str | None = None, store: MarketStore | None = None) -> dict:
    if kind not in ("strategy", "study"):
        raise ValueError("an experiment freezes a strategy or a market study")
    store = store or MarketStore()
    eid = uuid.uuid4().hex[:10]
    version = 1
    if parent_id:
        p = get(parent_id)
        if p is None:
            raise LookupError("the experiment this is a new version of was not found")
        version = int(p["version"]) + 1
    now = db.now()
    act = _last_session(store, symbol)
    with db.connect() as con:
        con.execute("INSERT INTO forward_experiments (id, name, kind, symbol, config, config_hash, version, parent_id, session_id, source_run_id, benchmark, assumptions, activation_date, status, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (eid, name[:120], kind, symbol.upper(), json.dumps(config, default=str), _hash(config), version, parent_id, session_id, source_run_id,
                                                               json.dumps(benchmark or {}, default=str), json.dumps(assumptions or {}, default=str), act, "active", now, now))
    return get(eid)


def get(eid: str) -> dict | None:
    with db.connect() as con:
        return _pub(db._row(con.execute("SELECT * FROM forward_experiments WHERE id=?", (eid,)).fetchone()))


def listing() -> list[dict]:
    with db.connect() as con:
        rows = [db._row(r) for r in con.execute("SELECT * FROM forward_experiments ORDER BY created_at DESC")]
    return [_pub(r) for r in rows]


def stop(eid: str) -> dict:
    with db.connect() as con:
        con.execute("UPDATE forward_experiments SET status='stopped', updated_at=? WHERE id=?", (db.now(), eid))
    return get(eid)


def new_version(eid: str, config: dict, name: str | None = None, store: MarketStore | None = None) -> dict:
    """A changed configuration is a new experiment. The previous version is left untouched and active."""
    old = get(eid)
    if old is None:
        raise LookupError("experiment not found")
    if _hash(config) == old["config_hash"]:
        raise ValueError("this configuration is identical to the frozen one, so there is nothing to version")
    return create(old["kind"], old["symbol"], config, name or old["name"], {"note": f"version {old['version'] + 1}: no historical benchmark was recorded for the changed configuration", "previous_version": eid},
                  old.get("assumptions") or {}, old.get("session_id"), old.get("source_run_id"), parent_id=eid, store=store)


def create_from_run(run: dict, session_id: str | None = None, name: str | None = None, store: MarketStore | None = None) -> dict:
    """Freeze a finished option backtest."""
    from strata.api.results import load_run_result
    if run["kind"] != "backtest" or run["status"] != "done":
        raise ValueError("only a finished option backtest can be frozen this way")
    res = load_run_result(run)
    cfg = res.get("config")
    if not cfg:
        raise ValueError("this run has no stored configuration to freeze")
    m = res.get("metrics") or {}
    exe = (res.get("meta") or {}).get("executability") or {}
    try:  # trading sessions in the backtest's own period (the rate of signals is compared with the forward rate later)
        from strata.core import calendar as cal
        sessions = max(1, len(cal.trading_days(date.fromisoformat(str(cfg["start"])[:10]), date.fromisoformat(str(cfg["end"])[:10]))))
    except Exception:  # noqa: BLE001
        sessions = None
    bench = {"period": [cfg.get("start"), cfg.get("end")], "trades": m.get("trades"), "win_rate": m.get("win_rate"), "expectancy": m.get("expectancy"), "net_pnl": m.get("net_pnl"), "profit_factor": m.get("profit_factor"),
             "signals": exe.get("signals"), "executed": exe.get("executed"), "sessions": sessions, "signals_per_100_sessions": ((exe.get("signals") or 0) / sessions * 100) if sessions else None, "pricing": (cfg.get("pricing") or {}).get("mode"),
             "source": "the backtest this experiment was frozen from"}
    sym = cfg["strategies"][0]["symbol"]
    return create("strategy", sym, cfg, name or run.get("name") or "Strategy", bench, {"execution": cfg.get("execution") or {}, "pricing": cfg.get("pricing") or {}}, session_id, run["id"], store=store)


def create_from_graph(graph: dict, result: dict, session_id: str | None = None, source_run_id: str | None = None, name: str | None = None, store: MarketStore | None = None) -> dict:
    """Freeze a market study (a research graph with one condition and outcome)."""
    pr = next((n["result"] for n in result.get("nodes", []) if n.get("result") and n["result"].get("kind") == "probability"), None)
    if pr is None or not graph.get("core"):
        raise ValueError("this graph has no single condition and outcome to follow forward")
    e = pr["evidence"]
    bench = {"period": pr.get("window"), "n": e["n"], "probability": e["probability"], "rest_probability": e["rest_probability"], "p_value": e["p_value"], "condition": pr["condition"], "outcome": pr["outcome"],
             "sessions": e["evaluated"], "signals_per_100_sessions": e["n"] / max(1, e["evaluated"]) * 100, "source": "the research graph this experiment was frozen from"}
    return create("study", graph["core"]["symbol"], {"graph": graph}, name or graph.get("question") or "Market study", bench, {"note": "underlying price behaviour only; no fills are simulated"}, session_id, source_run_id, store=store)


# ----------------------------------------------------------------------------- evaluation
def _forward_days(store: MarketStore, symbol: str, after: str) -> list[str]:
    try:
        d = store.read_daily(symbol)
    except Exception:  # noqa: BLE001
        return []
    return [str(x) for x in d["date"].to_list() if str(x) > after]


def evaluate(eid: str, store: MarketStore | None = None) -> dict:
    """Run the frozen configuration on the sessions after its activation date and store the report."""
    store = store or MarketStore()
    ex = get(eid)
    if ex is None:
        raise LookupError("experiment not found")
    if _hash(ex["config"]) != ex["config_hash"]:
        raise ValueError("the stored configuration no longer matches the hash recorded at activation: the experiment is not evaluated")
    days = _forward_days(store, ex["symbol"], ex["activation_date"])
    rep = {"experiment": eid, "version": ex["version"], "activation_date": ex["activation_date"], "forward_sessions": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
           "benchmark": ex.get("benchmark") or {}, "notes": [], "kind": ex["kind"], "config_hash": ex["config_hash"]}
    if not days:
        rep["status"] = "waiting"
        rep["notes"].append(f"No session after {ex['activation_date']} is stored yet. Forward evidence starts with the first session collected after the activation date; nothing earlier can count.")
        return _store(eid, rep)
    if ex["kind"] == "strategy":
        rep.update(_eval_strategy(ex, days, store))
    else:
        rep.update(_eval_study(ex, days, store))
    return _store(eid, rep)


def _store(eid: str, rep: dict) -> dict:
    from strata.api.results import dumps
    with db.connect() as con:
        con.execute("INSERT INTO forward_evaluations (experiment_id, evaluated_at, through_date, report) VALUES (?,?,?,?)", (eid, db.now(), rep.get("last"), dumps(rep)))
        con.execute("UPDATE forward_experiments SET updated_at=? WHERE id=?", (db.now(), eid))
    return json.loads(dumps(rep))


def _eval_strategy(ex: dict, days: list[str], store: MarketStore) -> dict:
    from strata.analytics.metrics import compute_metrics
    from strata.engine.engine import DataUnavailable, run_backtest
    from strata.engine.models import BacktestConfig
    cfg = {**ex["config"], "start": days[0], "end": days[-1]}
    sym = ex["symbol"]
    mode = (cfg.get("pricing") or {}).get("mode", "historical")
    try:
        have = {str(d) for d in (store.option_days(sym, "quote") if mode == "historical" else [])}
    except Exception:  # noqa: BLE001
        have = set()
    if mode == "trades":
        try:
            idx = store.option_bar_index(sym)
            have = {str(x)[:10] for x in idx["date"].to_list()} if not idx.is_empty() and "date" in idx.columns else set()
        except Exception:  # noqa: BLE001
            have = set()
    covered = [d for d in days if d in have] if mode != "modeled" else list(days)
    out = {"data": {"sessions": len(days), "with_option_data": len(covered), "pricing": mode, "missing": [d for d in days if d not in set(covered)][:60]}, "assumptions": ex.get("assumptions") or {}}
    try:
        res = run_backtest(BacktestConfig.model_validate(cfg), store=store, record_paths=False)
    except DataUnavailable as e:
        out.update(status="no_data", signals=None, trades=[], notes=[f"The forward sessions cannot be evaluated yet: {str(e)[:240]}"])
        return out
    m = compute_metrics(res.trades, res.daily, cfg.get("capital", {}).get("initial_capital", 100000.0), cfg.get("capital", {}).get("risk_free_rate", 0.0))
    exe = res.meta.get("executability") or {}
    trades = [{"entry_date": t["entry_date"], "entry_time": t["entry_time"], "exit_time": t["exit_time"], "exit_reason": t["exit_reason"], "pnl": t["pnl"], "units": t["units"], "flags": t["flags"],
               "fill_delay_seconds": t.get("fill_delay_seconds"), "legs": [{"right": l["right"], "strike": l["strike"], "expiration": l["expiration"], "entry_price": l["entry_price"], "exit_price": l["exit_price"], "price_source": l["price_source"]} for l in t["legs"]]}
              for t in res.trades]
    missed = [{"signal_time": x.get("signal_time") or x.get("signal_t"), "kind": x.get("kind"), "reason": x.get("reason")} for x in (res.diagnostics or {}).get("skipped_signals", [])][:200]
    out.update(status="evaluated", signals=exe.get("signals"), executed=exe.get("executed"), not_executed=exe.get("not_executed"), not_executed_by_reason=exe.get("not_executed_by_reason") or {}, trades=trades, missed_signals=missed,
               metrics={k: m.get(k) for k in ("trades", "net_pnl", "win_rate", "expectancy", "profit_factor", "max_drawdown")}, excluded_sessions=len((res.diagnostics or {}).get("excluded_sessions") or []) if isinstance((res.diagnostics or {}).get("excluded_sessions"), list) else None)
    out["comparison"], notes = _compare_strategy(ex.get("benchmark") or {}, out, len(days))
    out["notes"] = notes
    return out


def _compare_strategy(b: dict, f: dict, n_days: int) -> tuple[dict, list[str]]:
    m = f["metrics"]
    n = int(m.get("trades") or 0)
    notes = []
    cmp_ = {"forward_trades": n, "backtest_trades": b.get("trades"), "forward_win_rate": m.get("win_rate"), "backtest_win_rate": b.get("win_rate"), "forward_expectancy": m.get("expectancy"), "backtest_expectancy": b.get("expectancy"),
            "forward_signals_per_100_sessions": (f.get("signals") or 0) / max(1, n_days) * 100, "backtest_signals_per_100_sessions": b.get("signals_per_100_sessions"), "enough": n >= MIN_FORWARD, "win_rate_ci95": [None, None],
            "consistent": None}
    if f["data"]["with_option_data"] < f["data"]["sessions"]:
        notes.append(f"{f['data']['sessions'] - f['data']['with_option_data']} of the {f['data']['sessions']} forward sessions have no stored option prices of the kind this strategy is priced on. Signals on those sessions appear as missed, not as trades. "
                     "Option prices for a past session can often no longer be collected (free sources keep intraday option bars only for contracts still listed), so run the collector while the experiment is active.")
    if (f.get("not_executed") or 0) > 0:
        notes.append(f"{f['not_executed']} forward signal{'s' if f['not_executed'] != 1 else ''} could not be filled: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(f["not_executed_by_reason"].items(), key=lambda kv: -kv[1])[:3]) + ".")
    if n < MIN_FORWARD:
        notes.append(f"{n} forward trade{'s' if n != 1 else ''} so far. Fewer than {MIN_FORWARD} cannot confirm or contradict the backtest; the figures below are a record, not a verdict.")
        return cmp_, notes
    wins = round((m.get("win_rate") or 0) * n)
    ci = R.wilson(int(wins), n)
    cmp_["win_rate_ci95"] = ci
    if b.get("win_rate") is not None and ci[0] is not None:
        cmp_["consistent"] = bool(ci[0] <= b["win_rate"] <= ci[1])
        notes.append(f"Forward win rate {m['win_rate'] * 100:.0f}% over {n} trades (95% interval {ci[0] * 100:.0f}% to {ci[1] * 100:.0f}%); the backtest's was {b['win_rate'] * 100:.0f}%, which is "
                     + ("inside that interval: the forward record does not contradict it." if cmp_["consistent"] else "outside that interval: the forward record differs from the backtest."))
    if b.get("expectancy") is not None and m.get("expectancy") is not None:
        notes.append(f"Forward expectancy ${m['expectancy']:,.0f} per trade against ${b['expectancy']:,.0f} in the backtest.")
    return cmp_, notes


def _eval_study(ex: dict, days: list[str], store: MarketStore) -> dict:
    from strata.ai import nl_graph as NG
    from strata.research import graph as G
    g = NG.with_analyses(ex["config"]["graph"], ["probability"], [], start=days[0], end=days[-1])
    g["include_holdout"] = True  # these sessions did not exist at activation: nothing is being held back from them
    b = ex.get("benchmark") or {}
    try:
        res = G.run_graph(g, store, scope=None, reps=400)
    except (G.GraphError, LookupError, ValueError) as e:
        return {"status": "no_data", "notes": [f"The forward sessions cannot be evaluated yet: {str(e)[:240]}"]}
    pr = next((n["result"] for n in res["nodes"] if n.get("result") and n["result"].get("kind") == "probability"), None)
    if pr is None:
        err = next((n.get("error") for n in res["nodes"] if n["status"] == "error"), "no result")
        return {"status": "no_data", "notes": [f"The forward sessions cannot be evaluated yet: {err}"]}
    e = pr["evidence"]
    ci = R.wilson(e["hits"], e["n"])
    out = {"status": "evaluated", "signals": e["n"], "hits": e["hits"], "probability": e["probability"], "evaluated_sessions": e["evaluated"], "sessions": pr["sessions"], "probability_ci95": ci,
           "data": {"sessions": len(days), "evaluated": e["evaluated"], "not_evaluable": len(days) - e["evaluated"]},
           "comparison": {"forward_n": e["n"], "backtest_n": b.get("n"), "forward_probability": e["probability"], "backtest_probability": b.get("probability"), "enough": e["n"] >= MIN_FORWARD, "consistent": None,
                          "forward_signals_per_100_sessions": e["n"] / max(1, e["evaluated"]) * 100 if e["evaluated"] else None, "backtest_signals_per_100_sessions": b.get("signals_per_100_sessions")}}
    notes = []
    if e["evaluated"] < len(days):
        notes.append(f"{len(days) - e['evaluated']} of the {len(days)} forward sessions could not be evaluated with the bars stored (usually an intraday decision time for which no intraday bars were collected).")
    if e["n"] < MIN_FORWARD:
        notes.append(f"The condition has occurred {e['n']} time{'s' if e['n'] != 1 else ''} since activation. Fewer than {MIN_FORWARD} occurrences cannot confirm or contradict the historical frequency; this is a record, not a verdict.")
    elif b.get("probability") is not None and ci[0] is not None:
        out["comparison"]["consistent"] = bool(ci[0] <= b["probability"] <= ci[1])
        notes.append(f"Forward frequency {e['probability'] * 100:.0f}% of {e['n']} (95% interval {ci[0] * 100:.0f}% to {ci[1] * 100:.0f}%); the historical frequency was {b['probability'] * 100:.0f}%, which is "
                     + ("inside that interval." if out["comparison"]["consistent"] else "outside that interval: the forward record differs from the history."))
    out["notes"] = notes
    return out


def evaluate_active(store: MarketStore | None = None) -> list[dict]:
    """Evaluate every active experiment that has forward sessions it has not seen yet
    (called after a collection cycle). Failures are reported per experiment."""
    store = store or MarketStore()
    out = []
    for ex in listing():
        if ex["status"] != "active":
            continue
        days = _forward_days(store, ex["symbol"], ex["activation_date"])
        if not days or (ex.get("latest") or {}).get("last") == days[-1]:
            continue
        try:
            rep = evaluate(ex["id"], store)
            out.append({"id": ex["id"], "name": ex["name"], "status": rep.get("status"), "forward_sessions": rep.get("forward_sessions"), "last": rep.get("last")})
        except Exception as e:  # noqa: BLE001
            out.append({"id": ex["id"], "name": ex["name"], "status": "error", "error": f"{type(e).__name__}: {e}"[:200]})
    return out
