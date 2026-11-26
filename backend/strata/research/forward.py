"""Forward validation: freeze a configuration, then judge it only on sessions that had not
begun when it was frozen.

A backtest can always be made to look good, because its author has seen the data. The one
test that cannot be tuned is the future. An EXPERIMENT here is

  frozen      the full configuration, its hash, the engine version and the execution
              assumptions are stored at activation. Nothing can edit them (the database
              refuses the update). A change is a NEW experiment (version + 1) with its own
              activation; the old one keeps running as it was
  dated       activation is an immutable UTC instant taken from the clock, never from the
              data. The first session that can count is the first one that had not yet
              opened at that instant, by the exchange calendar: a session already in
              progress does not count, and no session before it ever can, whenever its
              data is stored. Stale local data cannot move the date, and filling in old
              sessions later cannot add evidence
  sealed      each session after activation is evaluated once, after its close, with the
              data stored at that moment, and written to an append-only ledger: signal
              decisions, simulated fills, contracts, fees, P&L, positions left open,
              observations that were missing, and the sources read. Entries are chained by
              hash; the database refuses to change or delete them. If stored data is later
              revised or filled in and the same session would now come out differently,
              the difference is recorded as a revision and the sealed record stands
  paper only  it is evaluated by the same deterministic engines on observed prices as
              they are collected. There is no order entry anywhere in Strata

`evaluate` is INCREMENTAL for a strategy (since V7). The account is a stored thing: after every
sealed session its state is written to an append-only table (cash, every open position with
its legs and what its exit rules remember, what the strategy carries between sessions). The
next evaluation loads that state and runs the engine over the sessions that are new, one at
a time, and over nothing else. A sealed session is never simulated again to produce the
record: positions, cash and earlier decisions are what was stored, not what a replay would
now give. A restart changes nothing, because nothing lived in memory.

`reconcile` is the other half. Stored data can be revised or filled in after a session was
sealed. Each seal records a fingerprint of the observations it read; when a fingerprint no
longer matches, that one session is simulated again FROM ITS OWN STORED STARTING STATE with
the data as it is now, and any difference is written as a `revision` entry. The sealed
entries, the stored states and every later decision stay exactly as they were.

The engine that sealed a session is recorded with it. When the engine has changed since the
experiment was frozen, an `engine_change` entry says from which session on, before any
session is sealed under the new one.

A market study has no account; its sessions are evaluated from the data once and sealed.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timezone

from strata.api import db
from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.research import robust as R

MIN_FORWARD = 10  # below this many forward observations no comparison is drawn
MISSING_KINDS = {"no_option_data", "no_recent_trade", "no_quote", "no_price", "no_contract", "no_atr"}  # a signal that was not filled because an observation was missing
LATE_OK = {"observation"}  # a study's outcome can resolve sessions after its condition: it is appended once, when it is known, and only for a condition that was itself sealed
REPLAYED = {"fill", "exit", "signal", "missing_data", "condition", "observation"}  # entry kinds the replay reproduces (and that are therefore checked against it)
GENESIS = "0" * 64


def _now() -> datetime:
    """The clock. Activation and sealing times come from here and from nowhere else."""
    return datetime.now(timezone.utc)


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


def _r(x):
    """Stored figures are rounded to 6 decimals, so that replaying a session reproduces its entries exactly."""
    if isinstance(x, dict):
        return {str(k): _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if isinstance(x, float):
        return None if x != x or x in (float("inf"), float("-inf")) else round(x, 6)
    return x


def _text(body) -> str:
    return json.dumps(_canon(_r(body)), sort_keys=True, default=str, separators=(",", ":"))


def first_session(at: datetime) -> tuple[date, str, str]:
    """The first session that had not yet opened at `at`, with the state of the market at that instant.

    A session that is already in progress has begun to be observed, so it is excluded and the
    next one is the first that counts. This reads the exchange calendar only: what happens to
    be stored locally plays no part."""
    et = at.astimezone(cal.ET)
    d = et.date()
    if cal.is_trading_day(d):
        o, c = cal.session_bounds(d)
        if et < o:
            return d, "before_open", f"Activated before the {d} session opened, so that session is the first that counts."
        nxt = cal.next_trading_day(d)
        if et < c:
            return nxt, "during_session", f"Activated while the {d} session was in progress. That session had already begun, so it does not count; the first that does is {nxt}."
        return nxt, "after_close", f"Activated after the {d} session closed. The first session that counts is {nxt}."
    nxt = cal.next_trading_day(d)
    return nxt, "market_closed", f"Activated while the market was closed. The first session that counts is {nxt}."


def _stored_through(store: MarketStore, symbol: str) -> str | None:
    try:
        d = store.read_daily(symbol)
        return str(d["date"][-1]) if not d.is_empty() else None
    except Exception:  # noqa: BLE001
        return None


def _upgrade(r: dict) -> dict:
    """An experiment made before V6 was dated by the last session STORED when it was frozen. With stale local data
    that date was too early, and sessions downloaded afterwards counted as forward evidence although they had
    already happened. Such an experiment is re-dated once, from the instant it was created."""
    if r.get("activated_at") and r.get("first_session"):
        return r
    at = datetime.fromisoformat(r["created_at"])
    first, state, rule = first_session(at)
    act = {"state": state, "rule": rule, "legacy_activation_date": r["activation_date"],
           "migrated": f"Created before version 6, when activation was the last stored session ({r['activation_date']}). It is now dated from the moment it was created; evaluations made under the old rule are superseded."}
    from strata.config import ENGINE_VERSION
    with db.connect() as con:
        con.execute("UPDATE forward_experiments SET activated_at=?, first_session=?, activation=?, activation_date=?, engine_version=COALESCE(engine_version, ?) WHERE id=? AND activated_at IS NULL",
                    (at.astimezone(timezone.utc).isoformat(timespec="seconds"), first.isoformat(), json.dumps(act), cal.prev_trading_day(first).isoformat(), "unrecorded (before version 6; evaluated with " + ENGINE_VERSION + ")", r["id"]))
        return db._row(con.execute("SELECT * FROM forward_experiments WHERE id=?", (r["id"],)).fetchone())


def _pub(r: dict | None) -> dict | None:
    if r is None:
        return None
    r = _upgrade(r)
    with db.connect() as con:
        ev = con.execute("SELECT id, evaluated_at, through_date, report FROM forward_evaluations WHERE experiment_id=? ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
        n = con.execute("SELECT COUNT(*) FROM forward_evaluations WHERE experiment_id=?", (r["id"],)).fetchone()[0]
        led = con.execute("SELECT COUNT(*), SUM(kind='session') FROM forward_ledger WHERE experiment_id=?", (r["id"],)).fetchone()
    out = dict(r)
    out["evaluations"] = int(n)
    out["ledger_entries"], out["sealed_sessions"] = int(led[0] or 0), int(led[1] or 0)
    latest = (json.loads(ev["report"]) | {"evaluated_at": ev["evaluated_at"]}) if ev and ev["report"] else None
    if latest is not None and "ledger" not in latest:
        latest = {**latest, "superseded": True, "notes": ["This evaluation was made under the previous activation rule (the last stored session) and is superseded. Evaluate again: only sessions from "
                                                          f"{out['first_session']} on can count."] + list(latest.get("notes") or [])}
    out["latest"] = latest
    return out


# ----------------------------------------------------------------------------- the ledger
def _append(con, eid: str, session: str, kind: str, key: str, body: dict) -> dict:
    last = con.execute("SELECT seq, hash FROM forward_ledger WHERE experiment_id=? ORDER BY seq DESC LIMIT 1", (eid,)).fetchone()
    seq, prev = (int(last["seq"]) + 1, last["hash"]) if last else (0, GENESIS)
    text, at = _text(body), _now().isoformat(timespec="seconds")
    h = hashlib.sha256(f"{prev}|{eid}|{seq}|{session}|{kind}|{key}|{text}|{at}".encode()).hexdigest()
    con.execute("INSERT INTO forward_ledger (experiment_id, seq, session, kind, key, body, recorded_at, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?)", (eid, seq, session, kind, key, text, at, prev, h))
    return {"seq": seq, "session": session, "kind": kind, "key": key, "body": json.loads(text), "recorded_at": at, "hash": h}


def ledger(eid: str, kinds: set[str] | None = None, limit: int | None = None) -> list[dict]:
    """The experiment's entries in the order they were written."""
    with db.connect() as con:
        rows = con.execute("SELECT seq, session, kind, key, body, recorded_at, prev_hash, hash FROM forward_ledger WHERE experiment_id=? ORDER BY seq", (eid,)).fetchall()
    out = [{**dict(r), "body": json.loads(r["body"])} for r in rows if kinds is None or r["kind"] in kinds]
    return out[-limit:] if limit else out


def verify(eid: str) -> dict:
    """Recompute the hash chain. Any entry that was altered, removed or inserted breaks it from that point on."""
    ex = get(eid)
    with db.connect() as con:
        rows = con.execute("SELECT seq, session, kind, key, body, recorded_at, prev_hash, hash FROM forward_ledger WHERE experiment_id=? ORDER BY seq", (eid,)).fetchall()
    prev, n = GENESIS, 0
    for i, r in enumerate(rows):
        h = hashlib.sha256(f"{prev}|{eid}|{r['seq']}|{r['session']}|{r['kind']}|{r['key']}|{r['body']}|{r['recorded_at']}".encode()).hexdigest()
        if r["seq"] != i or r["prev_hash"] != prev or r["hash"] != h:
            return {"ok": False, "entries": len(rows), "broken_at": i, "detail": f"entry {i} ({r['kind']} for {r['session']}) does not match the chain: the ledger was altered after it was written"}
        prev, n = h, n + 1
    if rows:
        root = json.loads(rows[0]["body"])
        if rows[0]["kind"] != "activation" or (ex is not None and root.get("config_hash") != ex["config_hash"]):
            return {"ok": False, "entries": len(rows), "broken_at": 0, "detail": "the ledger does not start from this experiment's activation"}
    return {"ok": True, "entries": n, "head": prev if rows else None, "broken_at": None, "detail": f"{n} entries, each chained to the one before it" if rows else "no entry yet"}


def _root(con, ex: dict) -> None:
    """The first entry: what was frozen. Everything after it is chained to it."""
    if con.execute("SELECT 1 FROM forward_ledger WHERE experiment_id=? LIMIT 1", (ex["id"],)).fetchone():
        return
    _append(con, ex["id"], ex["first_session"], "activation", ex["id"], {"config_hash": ex["config_hash"], "activated_at": ex["activated_at"], "first_session": ex["first_session"], "engine_version": ex.get("engine_version"),
                                                                        "kind": ex["kind"], "symbol": ex["symbol"], "version": ex["version"], "assumptions": ex.get("assumptions") or {}, "activation": ex.get("activation") or {}})


def create(kind: str, symbol: str, config: dict, name: str, benchmark: dict | None = None, assumptions: dict | None = None, session_id: str | None = None, source_run_id: str | None = None,
           parent_id: str | None = None, store: MarketStore | None = None) -> dict:
    if kind not in ("strategy", "study"):
        raise ValueError("an experiment freezes a strategy or a market study")
    from strata.config import ENGINE_VERSION
    store = store or MarketStore()
    eid = uuid.uuid4().hex[:10]
    version = 1
    if parent_id:
        p = get(parent_id)
        if p is None:
            raise LookupError("the experiment this is a new version of was not found")
        version = int(p["version"]) + 1
    at = _now()
    first, state, rule = first_session(at)
    stored = _stored_through(store, symbol)
    behind = len(cal.trading_days(date.fromisoformat(stored), first)) - 1 if stored and date.fromisoformat(stored) < first else 0
    act = {"state": state, "rule": rule, "stored_through": stored, "sessions_not_yet_stored": max(0, behind - 1),
           "note": "The activation date comes from the clock and the exchange calendar. What is stored locally does not affect it."}
    now = at.isoformat(timespec="seconds")
    with db.connect() as con:
        con.execute("INSERT INTO forward_experiments (id, name, kind, symbol, config, config_hash, version, parent_id, session_id, source_run_id, benchmark, assumptions, activation_date, status, created_at, updated_at, "
                    "activated_at, first_session, engine_version, activation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (eid, name[:120], kind, symbol.upper(), json.dumps(config, default=str), _hash(config), version, parent_id, session_id, source_run_id, json.dumps(benchmark or {}, default=str), json.dumps(assumptions or {}, default=str),
                     cal.prev_trading_day(first).isoformat(), "active", now, now, now, first.isoformat(), ENGINE_VERSION, json.dumps(act)))
        _root(con, db._row(con.execute("SELECT * FROM forward_experiments WHERE id=?", (eid,)).fetchone()))
    return get(eid)


def get(eid: str) -> dict | None:
    with db.connect() as con:
        r = db._row(con.execute("SELECT * FROM forward_experiments WHERE id=?", (eid,)).fetchone())
    return _pub(r)


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
def _eligible(ex: dict, now: datetime | None = None) -> list[date]:
    """Sessions since activation that have closed, by the exchange calendar (whether or not their data is stored)."""
    now = now or _now()
    first, today = date.fromisoformat(ex["first_session"]), now.astimezone(cal.ET).date()
    return [d for d in cal.trading_days(first, today) if cal.session_bounds(d)[1] <= now] if first <= today else []


def _stored_days(store: MarketStore, symbol: str) -> set[date]:
    try:
        return set(store.read_daily(symbol)["date"].to_list())
    except Exception:  # noqa: BLE001
        return set()


def pending(ex: dict, store: MarketStore) -> list[str]:
    """Closed sessions since activation whose prices are stored and that are not sealed yet."""
    stored = _stored_days(store, ex["symbol"])
    have = [d for d in _eligible(ex) if d in stored]
    if not have:
        return []
    with db.connect() as con:
        sealed = {r[0] for r in con.execute("SELECT session FROM forward_ledger WHERE experiment_id=? AND kind='session'", (ex["id"],))}
    return [d.isoformat() for d in _eligible(ex) if d <= have[-1] and d.isoformat() not in sealed]


# ----------------------------------------------------------------------------- the stored account
def _states(eid: str) -> list[dict]:
    with db.connect() as con:
        return [dict(r) for r in con.execute("SELECT session, state, state_hash, engine_version, data_fingerprint, origin, created_at FROM forward_states WHERE experiment_id=? ORDER BY session", (eid,))]


def _save_state(con, eid: str, session: str, state: dict, fingerprint: str | None, origin: str) -> str:
    from strata.config import ENGINE_VERSION
    text = json.dumps(state, sort_keys=True, default=str, separators=(",", ":"))
    h = hashlib.sha256(text.encode()).hexdigest()
    con.execute("INSERT INTO forward_states (experiment_id, session, state, state_hash, engine_version, data_fingerprint, origin, created_at) VALUES (?,?,?,?,?,?,?,?)", (eid, session, text, h, ENGINE_VERSION, fingerprint, origin, db.now()))
    return h


def fingerprint(ex: dict, d: date, store: MarketStore) -> str:
    """What was stored for this session when it was read: the underlying's bars and, for an options strategy, the option
    observations of the dataset the run uses. Counts and sums, not file dates: copying or re-indexing a store does not
    change it, a revised or filled-in observation does."""
    import polars as pl
    sym = ex["symbol"]
    parts: list = [sym, d.isoformat()]
    try:
        b = store.read_best_bars(sym, d, d)
        parts += [b.height, round(float(b["close"].sum()), 4), round(float(b["volume"].sum()), 0)] if not b.is_empty() else [0]
        dd = store.read_daily(sym).filter(pl.col("date") == d)
        parts += [round(float(dd["close"][0]), 4)] if dd.height else [None]
    except Exception:  # noqa: BLE001
        parts.append("underlying unreadable")
    if ex["kind"] == "strategy":
        mode = (ex["config"].get("pricing") or {}).get("mode", "historical")
        prov = (ex["config"].get("data") or {}).get("options_provider")
        try:
            if mode == "trades":
                idx = store.option_bar_index(sym, provider=prov)
                day = idx.filter(pl.col("date") == d) if not idx.is_empty() else idx
                parts += ["trades", day.height, int(day["bars"].sum()) if day.height else 0, int(day.filter(pl.col("status") == "ok").height) if day.height else 0]
            elif mode == "historical":
                q = store.read_option_day(sym, d, "quote", provider=prov)
                parts += ["quotes", 0 if q is None else q.height, None if q is None or q.is_empty() else round(float((q["bid"].fill_null(0) + q["ask"].fill_null(0)).sum()), 2)]
        except Exception:  # noqa: BLE001
            parts.append("options unreadable")
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:20]


def evaluate(eid: str, store: MarketStore | None = None, refresh: bool = False) -> dict:
    """Seal the sessions that are new and store the report.

    A strategy is evaluated INCREMENTALLY: the account state stored after the last sealed session is loaded, and the engine
    runs over each new session in turn and over nothing else. With `refresh`, what the configuration needs for the unsealed
    sessions is first fetched from the free sources (never offline, never a paid provider), because a session is sealed with
    the data stored at that moment. Sessions already sealed are not simulated again here; `reconcile` does that, for the
    sessions whose stored data has changed, without touching what was sealed."""
    from strata.config import ENGINE_VERSION
    store = store or MarketStore()
    ex = get(eid)
    if ex is None:
        raise LookupError("experiment not found")
    if _hash(ex["config"]) != ex["config_hash"]:
        raise ValueError("the stored configuration no longer matches the hash recorded at activation: the experiment is not evaluated")
    chain = verify(eid)
    if not chain["ok"]:
        raise ValueError(f"the forward ledger failed its integrity check ({chain['detail']}): the experiment is not evaluated")
    now = _now()
    days = _eligible(ex, now)
    rep = {"experiment": eid, "version": ex["version"], "kind": ex["kind"], "config_hash": ex["config_hash"], "activation_date": ex["activation_date"], "activated_at": ex["activated_at"], "first_session": ex["first_session"],
           "activation": ex.get("activation") or {}, "engine_version": ex.get("engine_version"), "engine_version_now": ENGINE_VERSION, "benchmark": ex.get("benchmark") or {}, "notes": [], "as_of": now.isoformat(timespec="seconds"),
           "eligible_sessions": len(days), "forward_sessions": 0, "first": None, "last": None, "awaiting_data": [], "newly_sealed": [], "revisions": [], "ledger": chain,
           "method": "incremental: each new session is run from the stored account state" if ex["kind"] == "strategy" else "a market study has no account: its sessions are evaluated from the data and sealed once"}
    if not days:
        rep["status"] = "waiting"
        rep["notes"].append(f"No session since activation has closed yet. The first session that can count is {ex['first_session']}; nothing earlier can count, whenever its data is stored.")
        return _store(eid, rep)
    if refresh:
        rep["refresh"] = _refresh(ex, days, store)
    stored = _stored_days(store, ex["symbol"])
    have = [d for d in days if d in stored]
    if not have:
        rep["status"] = "awaiting_data"
        rep["awaiting_data"] = [d.isoformat() for d in days][-60:]
        rep["notes"].append(f"{len(days)} session{'s have' if len(days) != 1 else ' has'} closed since activation ({days[0]} to {days[-1]}), but no {ex['symbol']} price is stored for {'any of them' if len(days) != 1 else 'it'} yet. "
                            f"Refresh the data or run the collector, then evaluate again. Sessions before {ex['first_session']} are never used.")
        return _store(eid, rep)
    window = [d for d in days if d <= have[-1]]
    rep["awaiting_data"] = [d.isoformat() for d in days if d > have[-1]][-60:]
    if ex["kind"] != "strategy":
        events, extra = _study_events(ex, window, store)
        if extra.get("failed"):
            rep["status"] = "awaiting_data"
            rep["notes"].append(extra["failed"])
            rep["awaiting_data"] = [d.isoformat() for d in days][-60:]
            return _store(eid, rep)
        new, revs = _reconcile(ex, window, stored, events, extra)
        rep["newly_sealed"], rep["revisions_new"] = new, len(revs)
        return _store(eid, _report(ex, rep, window, extra))
    # ---- a strategy: from the stored account, the new sessions only
    with db.connect() as con:
        sealed = [r[0] for r in con.execute("SELECT session FROM forward_ledger WHERE experiment_id=? AND kind='session' ORDER BY seq", (eid,))]
    states = _states(eid)
    state = json.loads(states[-1]["state"]) if states else None
    extra: dict = {"engine_version": ENGINE_VERSION, "open": []}
    if sealed and (not states or states[-1]["session"] < sealed[-1]):
        # sealed before account states were stored (version 6 replayed the window at every evaluation): the account is
        # rebuilt ONCE, by one run through the last sealed session with the data stored now, and that is written down
        prior = [d for d in window if d.isoformat() <= sealed[-1]]
        _ev, ex0 = _strategy_events(ex, prior, store, None, incremental=True)
        if ex0.get("failed"):
            rep["status"] = "awaiting_data"
            rep["notes"].append(ex0["failed"])
            return _store(eid, rep)
        state = ex0["state"]
        with db.connect() as con:
            _save_state(con, eid, sealed[-1], state, fingerprint(ex, prior[-1], store), "rebuilt")
            _append(con, eid, sealed[-1], "state_rebuilt", sealed[-1], {"through": sealed[-1], "note": "These sessions were sealed before account states were stored. The account as of this session was rebuilt by one run over the "
                                                                    "sealed sessions with the data stored now; the sealed entries were not changed, and every later session continues from this state.",
                                                                    "open_positions": len(state.get("positions") or []), "cash": state.get("cash")})
        rep["notes"].append(f"The account state through {sealed[-1]} was rebuilt once from the sealed sessions; from now on each session continues from the stored state.")
    new_days = [d for d in window if d.isoformat() not in set(sealed)]
    pinned = str(ex.get("engine_version") or "")
    if new_days and pinned and not pinned.startswith("unrecorded") and pinned != ENGINE_VERSION:
        with db.connect() as con:
            if not con.execute("SELECT 1 FROM forward_ledger WHERE experiment_id=? AND kind='engine_change' AND key=?", (eid, ENGINE_VERSION)).fetchone():
                _root(con, ex)
                _append(con, eid, new_days[0].isoformat(), "engine_change", ENGINE_VERSION, {"frozen_under": pinned, "now": ENGINE_VERSION, "from_session": new_days[0].isoformat(),
                                                                                           "note": f"The experiment was frozen under engine {pinned}. Sessions from {new_days[0]} on are evaluated by engine {ENGINE_VERSION}; "
                                                                                                   "sessions sealed earlier keep the result of the engine that sealed them."})
    newly: list[str] = []
    for d in new_days:
        events, ex1 = _strategy_events(ex, [d], store, state, incremental=True)
        if ex1.get("failed"):
            rep["notes"].append(ex1["failed"])
            rep["awaiting_data"] = [x.isoformat() for x in days if x >= d][-60:]
            break
        fp = fingerprint(ex, d, store)
        ex1["fingerprint"] = fp
        new, _revs = _reconcile(ex, [d], stored, events, ex1)
        state = ex1["state"]
        with db.connect() as con:
            _save_state(con, eid, d.isoformat(), state, fp, "sealed")
        newly += new
        extra = ex1
    rep["newly_sealed"] = newly
    extra["data"] = _data_summary(ex, window, store)
    if "sources" not in extra:
        # nothing was run in this evaluation: the positions still open are the ones the last seal recorded
        last_pos = ledger(eid, {"position"}, 1)
        extra["open"] = (last_pos[-1]["body"].get("open") or []) if last_pos else []
    rep["account"] = None if state is None else {"through": (state.get("through") or "")[:10], "cash": state.get("cash"), "open_positions": len(state.get("positions") or []), "realised": state.get("realised"),
                                                  "states_stored": len(_states(eid))}
    # stored data that changed after a session was sealed: recorded, never applied
    rec = reconcile(eid, store, _evaluating=True)
    rep["revisions_new"] = rec["revisions_new"]
    rep["reconciliation"] = {k: rec[k] for k in ("checked", "changed", "revisions_new")}
    return _store(eid, _report(ex, rep, window, extra))


def reconcile(eid: str, store: MarketStore | None = None, force: bool = False, _evaluating: bool = False) -> dict:
    """Has the data of a sealed session changed since it was sealed, and would the session now come out differently?

    For every sealed session whose stored observations no longer match the fingerprint taken when it was sealed (every
    session with `force`), that ONE session is simulated again from its own stored starting state, with the data as it is
    now, and compared with its sealed entries. A difference is appended to the ledger as a `revision`. Nothing else is
    written: the sealed entries, the stored account states and every later session's decisions stay as they were."""
    store = store or MarketStore()
    ex = get(eid)
    if ex is None:
        raise LookupError("experiment not found")
    out = {"experiment": eid, "checked": 0, "changed": [], "revisions_new": 0, "revisions": [], "forced": bool(force)}
    if ex["kind"] != "strategy":
        out["note"] = "A market study is compared with its sealed entries at every evaluation."
        return out
    states = _states(eid)
    by = {r["session"]: r for r in states}
    order = [r["session"] for r in states]
    stored = _stored_days(store, ex["symbol"])
    for i, sess in enumerate(order):
        row = by[sess]
        if row["origin"] != "sealed":
            continue  # a rebuilt state has no starting state of its own to replay from
        d = date.fromisoformat(sess)
        out["checked"] += 1
        fp = fingerprint(ex, d, store)
        if not force and fp == row["data_fingerprint"]:
            continue
        out["changed"].append(sess)
        before = json.loads(by[order[i - 1]]["state"]) if i > 0 else None
        events, ex1 = _strategy_events(ex, [d], store, before, incremental=True)
        if ex1.get("failed"):
            continue
        ex1["fingerprint"] = fp
        _new, revs = _reconcile(ex, [d], stored, events, ex1, compare_only=True)
        out["revisions"] += revs
    out["revisions_new"] = len(out["revisions"])
    if not _evaluating and out["changed"]:
        with db.connect() as con:
            con.execute("UPDATE forward_experiments SET updated_at=? WHERE id=?", (db.now(), eid))
    out["text"] = (f"{out['checked']} sealed session(s) checked against the data stored now. " + (f"{len(out['changed'])} read different observations than when they were sealed ({', '.join(out['changed'][:6])}"
                   + (f" and {len(out['changed']) - 6} more" if len(out['changed']) > 6 else "") + f"); replaying them from their own starting state gave {out['revisions_new']} difference(s), recorded as revisions. " if out["changed"] else
                   "None of them reads different observations than when it was sealed. ") + "The sealed record and every later decision are unchanged.")
    return out


def _store(eid: str, rep: dict) -> dict:
    from strata.api.results import dumps
    with db.connect() as con:
        con.execute("INSERT INTO forward_evaluations (experiment_id, evaluated_at, through_date, report) VALUES (?,?,?,?)", (eid, db.now(), rep.get("last"), dumps(rep)))
        con.execute("UPDATE forward_experiments SET updated_at=? WHERE id=?", (db.now(), eid))
    return json.loads(dumps(rep))


REVISION = {"changed": "Replaying this session now gives a different result from the one that was sealed (the stored data was revised or filled in afterwards, or the engine changed). The sealed entry stands.",
            "appeared": "Replaying this sealed session now produces an entry that did not exist when it was sealed (data for it was stored afterwards). It is not added to the record.",
            "vanished": "A sealed entry can no longer be reproduced from the stored data. It stays in the record as it was sealed.",
            "orphan": "The replay closes a position that the sealed record never opened. It is not added to the record."}


def _reconcile(ex: dict, window: list[date], stored: set[date], events: list[dict], extra: dict, compare_only: bool = False) -> tuple[list[str], list[dict]]:
    """Append the entries of every session that is new, in order, and seal it. Sessions already sealed are only
    compared with the replay: a difference becomes a `revision` entry and changes nothing that was sealed.
    With `compare_only` nothing but revisions is ever written (the reconciliation pass)."""
    eid = ex["id"]
    win = {d.isoformat() for d in window}
    new: list[str] = []
    revs: list[dict] = []
    with db.connect() as con:
        _root(con, ex)
        rows = con.execute("SELECT session, kind, key, body FROM forward_ledger WHERE experiment_id=? ORDER BY seq", (eid,)).fetchall()
        have = {(r["kind"], r["key"]): r["body"] for r in rows}
        sealed = {r["session"] for r in rows if r["kind"] == "session"}
        by_day: dict[str, list] = {}
        for ev in events:
            by_day.setdefault(ev["session"], []).append(ev)

        def revise(session, what, kind, key, was, now_):
            rk = hashlib.sha256(f"{what}|{kind}|{key}|{_text(now_) if now_ is not None else ''}".encode()).hexdigest()[:16]
            if ("revision", rk) in have:
                return
            body = {"what": what, "entry": kind, "of": key, "sealed": was, "replay": now_, "note": REVISION[what]}
            _append(con, eid, session, "revision", rk, body)
            have[("revision", rk)] = ""
            revs.append({"session": session, **body})

        for d in window:
            iso = d.isoformat()
            evs = by_day.get(iso, [])
            was_sealed = iso in sealed
            if compare_only and not was_sealed:
                continue
            for ev in evs:
                k, text = (ev["kind"], ev["key"]), _text(ev["body"])
                if k in have:
                    if have[k] != text:
                        revise(iso, "changed", ev["kind"], ev["key"], json.loads(have[k]), _r(ev["body"]))
                elif was_sealed and not (ev["kind"] in LATE_OK and ("condition", ev["key"]) in have):
                    revise(iso, "appeared", ev["kind"], ev["key"], None, _r(ev["body"]))
                elif ev["kind"] == "exit" and ("fill", ev["key"]) not in have:
                    revise(iso, "orphan", ev["kind"], ev["key"], None, _r(ev["body"]))
                else:
                    _append(con, eid, iso, ev["kind"], ev["key"], ev["body"])
                    have[k] = text
            if not was_sealed:
                n = {k: sum(1 for e in evs if e["kind"] == k) for k in ("fill", "exit", "signal", "missing_data", "condition", "observation")}
                _append(con, eid, iso, "session", iso, {"date": iso, "underlying_stored": d in stored, "entries": n, "realised_pnl": sum(float(e["body"].get("pnl") or 0) for e in evs if e["kind"] == "exit"),
                                                        "sources": extra.get("sources") or {}, "engine_version": extra.get("engine_version"), "window_end": window[-1].isoformat(), "minutes_after_close": int((_now() - cal.session_bounds(d)[1]).total_seconds() // 60),
                                                        **({"data_fingerprint": extra["fingerprint"], "from_stored_state": True} if extra.get("fingerprint") else {})})
                new.append(iso)
        replayed = {(ev["kind"], ev["key"]) for ev in events}
        for r in rows:
            if r["kind"] in REPLAYED and r["session"] in win and (r["kind"], r["key"]) not in replayed:
                revise(r["session"], "vanished", r["kind"], r["key"], json.loads(r["body"]), None)
        if new and not compare_only:
            _append(con, eid, new[-1], "position", new[-1], {"through": new[-1], "open": extra.get("open") or []})
    return new, revs


def _fill_key(t: dict) -> str:
    legs = ",".join(sorted(f"{l.get('right') or 'S'}{l.get('strike')}@{l.get('expiration')}x{l.get('ratio')}" for l in t["legs"]))
    return f"{t.get('strategy_idx', 0)}|{t['entry_ts']}|{t.get('lot', 1)}|{legs}"


def _data_summary(ex: dict, window: list[date], store: MarketStore) -> dict:
    """How many of the forward sessions have the option observations the strategy's pricing mode reads."""
    sym, mode = ex["symbol"], (ex["config"].get("pricing") or {}).get("mode", "historical")
    have: set[str] = set()
    try:
        if mode == "trades":
            idx = store.option_bar_index(sym)
            have = {str(x)[:10] for x in idx["date"].to_list()} if not idx.is_empty() and "date" in idx.columns else set()
        elif mode == "historical":
            have = {str(d) for d in store.option_days(sym, "quote")}
    except Exception:  # noqa: BLE001
        have = set()
    days = [d.isoformat() for d in window]
    covered = [d for d in days if d in have] if mode != "modeled" else list(days)
    return {"sessions": len(days), "with_option_data": len(covered), "pricing": mode, "missing": [d for d in days if d not in set(covered)][:60]}


def _strategy_events(ex: dict, window: list[date], store: MarketStore, resume: dict | None = None, incremental: bool = False) -> tuple[list[dict], dict]:
    """Run the frozen strategy over `window` and turn the result into dated entries. With `incremental` the run starts from
    the stored account state `resume` (None for the first session) and leaves its positions open: the state it ends in is
    returned in extra["state"]."""
    from strata.config import ENGINE_VERSION
    from strata.engine.engine import DataUnavailable, run_backtest
    from strata.engine.models import BacktestConfig
    cfg = {**ex["config"], "start": window[0].isoformat(), "end": window[-1].isoformat()}
    sym, mode = ex["symbol"], (ex["config"].get("pricing") or {}).get("mode", "historical")
    have: set[str] = set()
    try:
        if mode == "trades":
            idx = store.option_bar_index(sym)
            have = {str(x)[:10] for x in idx["date"].to_list()} if not idx.is_empty() and "date" in idx.columns else set()
        elif mode == "historical":
            have = {str(d) for d in store.option_days(sym, "quote")}
    except Exception:  # noqa: BLE001
        have = set()
    days = [d.isoformat() for d in window]
    covered = [d for d in days if d in have] if mode != "modeled" else list(days)
    extra: dict = {"engine_version": ENGINE_VERSION, "data": {"sessions": len(days), "with_option_data": len(covered), "pricing": mode, "missing": [d for d in days if d not in set(covered)][:60]}, "open": []}
    try:
        res = run_backtest(BacktestConfig.model_validate(cfg), store=store, record_paths=False, **({"resume": resume, "keep_open": True} if incremental else {}))
    except DataUnavailable as e:
        extra["failed"] = f"The sessions since activation cannot be evaluated yet: {str(e)[:240]}"
        return [], extra
    if incremental:
        extra["state"] = res.meta["state"]
    extra["sources"] = {"datasets": res.meta.get("dataset_versions") or {}, "options": (res.meta.get("data") or {}).get("options_dataset") or res.meta.get("options_dataset"), "pricing": mode}
    events: list[dict] = []
    win = set(days)
    for t in list(res.trades) + (list(res.meta.get("open_positions") or []) if incremental else []):
        key = _fill_key(t)
        legs = [{"right": l["right"], "strike": l["strike"], "expiration": l["expiration"], "ratio": l["ratio"], "side": l["side"], "multiplier": l["multiplier"], "price_source": l["price_source"],
                 "entry_price": l["entry_price"], "entry_bid": l["entry_bid"], "entry_ask": l["entry_ask"], "entry_print": l["entry_print"]} for l in t["legs"]]
        if t["entry_date"] in win:
            events.append({"session": t["entry_date"], "kind": "fill", "key": key, "body": {"strategy": t["strategy"], "signal_time": cal.to_et(int(t["signal_ts"]) * 1_000_000_000).isoformat(), "entry_time": t["entry_time"],
                                                                                           "fill_delay_seconds": t["fill_delay_seconds"], "units": t["original_units"], "entry_cash": t["entry_cash"] if not t.get("stage_exits") else None,
                                                                                           "entry_spot": t["entry_spot"], "legs": legs}})
        still_open = t["exit_reason"] == "end_of_backtest" or not t.get("exit_time") or not t.get("exit_reason")
        if still_open:
            extra["open"].append({"key": key, "strategy": t["strategy"], "entry_time": t["entry_time"], "units": t["units"], "legs": [{k: l[k] for k in ("right", "strike", "expiration", "ratio", "entry_price")} for l in legs],
                                  "marked_pnl": t["pnl"], "marked_at": t.get("exit_time"), "note": "marked at the last stored price; the position is still open"})
            continue
        xd = str(t["exit_time"])[:10]
        if xd in win:
            events.append({"session": xd, "kind": "exit", "key": key, "body": {"exit_time": t["exit_time"], "exit_reason": t["exit_reason"], "units": t["units"], "pnl": t["pnl"], "gross_pnl": t["gross_pnl"], "fees": t["commissions"],
                                                                               "slippage": t["slippage"], "flags": t["flags"], "stage_exits": t.get("stage_exits") or [],
                                                                               "legs": [{"right": l["right"], "strike": l["strike"], "expiration": l["expiration"], "exit_price": l["exit_price"], "exit_bid": l["exit_bid"], "exit_ask": l["exit_ask"],
                                                                                         "exit_print": l["exit_print"], "exit_reason": l["exit_reason"]} for l in t["legs"]]}})
    for x in (res.diagnostics or {}).get("skipped_signals", []):
        st = cal.to_et(int(x["signal_t"]) * 1_000_000_000)
        d = st.date().isoformat()
        if d in win:
            kind = "missing_data" if x["kind"] in MISSING_KINDS else "signal"
            events.append({"session": d, "kind": kind, "key": f"{x.get('strategy', '')}|{x['signal_t']}|{x['kind']}", "body": {"signal_time": st.isoformat(), "filled": False, "why": x["kind"], "reason": x["reason"], "strategy": x.get("strategy", "")}})
    for reason, dates in ((res.diagnostics or {}).get("excluded_sessions") or {}).items():
        for d in dates:
            if d in win:
                events.append({"session": d, "kind": "missing_data", "key": f"session|{d}|{reason[:60]}", "body": {"filled": False, "why": "session_not_evaluable", "reason": reason}})
    stored = _stored_days(store, sym)
    for d in window:
        if d not in stored:
            events.append({"session": d.isoformat(), "kind": "missing_data", "key": f"session|{d}|no_bar", "body": {"filled": False, "why": "no_underlying_bar", "reason": f"the exchange was open but no {sym} bar is stored for this session"}})
    events.sort(key=lambda e: (e["session"], {"fill": 0, "signal": 1, "missing_data": 2, "exit": 3}.get(e["kind"], 9), e["key"]))
    return events, extra


def _study_events(ex: dict, window: list[date], store: MarketStore) -> tuple[list[dict], dict]:
    """Replay the frozen market study over the forward window. Each session on which the condition held gets a
    `condition` entry when it is sealed, and an `observation` entry once its outcome is known (the same session
    for a same-day outcome, later for an outcome measured over several sessions)."""
    import numpy as np

    from strata.ai import nl_graph as NG
    from strata.config import ENGINE_VERSION
    from strata.research import graph as G
    g = NG.with_analyses(ex["config"]["graph"], ["probability"], [], start=window[0].isoformat(), end=window[-1].isoformat())
    g["include_holdout"] = True  # these sessions had not begun at activation: nothing is being held back from them
    extra: dict = {"engine_version": ENGINE_VERSION, "open": [], "sources": {"pricing": "underlying prices only"}}
    try:
        res = G.run_graph(g, store, scope=None, reps=400)
        row = next((n for n in res["nodes"] if n.get("result") and n["result"].get("kind") == "probability"), None)
        if row is None:
            raise ValueError(next((n.get("error") for n in res["nodes"] if n["status"] == "error"), None) or "no result")
        run = G.Run(g, store, None, None, reps=1)
        o = run.get(row["inputs"]["outcome"])
        m = run.get(row["inputs"]["condition"]) if row["inputs"].get("condition") else None
    except (G.GraphError, LookupError, ValueError, KeyError) as e:
        extra["failed"] = f"The sessions since activation cannot be evaluated yet: {str(e)[:240]}"
        return [], extra
    pr = row["result"]
    e = pr["evidence"]
    held = (m.true if m is not None else np.ones(len(o.ds.ft.dates), dtype=bool)) & o.ds.pm
    if o.event is not None:
        held = held & o.event  # an outcome anchored on an intraday event: the event is part of the condition
    win = {d.isoformat() for d in window}
    events = [{"session": str(d), "kind": "condition", "key": str(d), "body": {"date": str(d), "condition": pr.get("condition")}} for d, h in zip(o.ds.ft.dates, held) if h and str(d) in win]
    events += [{"session": str(x["date"]), "kind": "observation", "key": str(x["date"]), "body": {"date": str(x["date"]), "outcome": x["outcome"], "end_return_pct": x.get("end_return_pct")}}
               for x in pr["sessions"] if str(x["date"]) in win and x["outcome"] is not None]
    events.sort(key=lambda v: (v["session"], v["kind"]))  # the condition entry is written before its outcome
    extra.update(evaluated=e["evaluated"], not_evaluable=pr.get("not_evaluable") or {}, outcome=pr.get("outcome"), condition=pr.get("condition"))
    return events, extra


def _report(ex: dict, rep: dict, window: list[date], extra: dict) -> dict:
    """The report, read from the ledger (not from the replay)."""
    led = ledger(ex["id"])
    sess = [e for e in led if e["kind"] == "session"]
    revs = [e for e in led if e["kind"] == "revision"]
    rep.update(status="evaluated", forward_sessions=len(sess), first=sess[0]["session"] if sess else None, last=sess[-1]["session"] if sess else None, sealed=[{"session": e["session"], "sealed_at": e["recorded_at"], **e["body"]["entries"]} for e in sess][-250:],
               revisions=[{"session": e["session"], "recorded_at": e["recorded_at"], **{k: e["body"].get(k) for k in ("what", "entry", "of", "note")}} for e in revs][-100:], ledger=verify(ex["id"]))
    if rep.get("engine_version") and rep["engine_version"] != rep["engine_version_now"] and not str(rep["engine_version"]).startswith("unrecorded"):
        rep["notes"].append(f"Frozen under engine {rep['engine_version']}; the engine is now {rep['engine_version_now']}. Sessions already sealed keep the result computed when they were sealed.")
    if revs:
        rep["notes"].append(f"{len(revs)} revision{'s' if len(revs) != 1 else ''} recorded: stored data for a sealed session changed afterwards. The sealed entries were not altered; the revisions are listed below.")
    if rep["awaiting_data"]:
        rep["notes"].append(f"{len(rep['awaiting_data'])} later session{'s have' if len(rep['awaiting_data']) != 1 else ' has'} closed but {'have' if len(rep['awaiting_data']) != 1 else 'has'} no stored {ex['symbol']} price yet, so {'they are' if len(rep['awaiting_data']) != 1 else 'it is'} not sealed.")
    if ex["kind"] == "strategy":
        fills = {e["key"]: e for e in led if e["kind"] == "fill"}
        exits = {e["key"]: e for e in led if e["kind"] == "exit"}
        missed = [e for e in led if e["kind"] in ("signal", "missing_data") and "signal_time" in e["body"]]
        trades = []
        for k, x in exits.items():
            f = fills.get(k)
            if f is None:
                continue
            xl = {(l["right"], l["strike"], l["expiration"]): l for l in x["body"]["legs"]}
            trades.append({"entry_date": f["session"], "entry_time": f["body"]["entry_time"], "exit_time": x["body"]["exit_time"], "exit_reason": x["body"]["exit_reason"], "pnl": x["body"]["pnl"], "fees": x["body"].get("fees"), "units": f["body"]["units"],
                           "flags": x["body"].get("flags") or [], "fill_delay_seconds": f["body"].get("fill_delay_seconds"), "sealed_at": x["recorded_at"],
                           "legs": [{"right": l["right"], "strike": l["strike"], "expiration": l["expiration"], "entry_price": l["entry_price"], "exit_price": (xl.get((l["right"], l["strike"], l["expiration"])) or {}).get("exit_price"), "price_source": l["price_source"]}
                                    for l in f["body"]["legs"]]})
        trades.sort(key=lambda t: t["exit_time"])
        marks = {o["key"]: o for o in extra.get("open") or []}
        open_ = [{"key": k, "entry_date": f["session"], "entry_time": f["body"]["entry_time"], "units": f["body"]["units"], "legs": [{kk: l[kk] for kk in ("right", "strike", "expiration", "entry_price", "price_source")} for l in f["body"]["legs"]],
                  "marked_pnl": (marks.get(k) or {}).get("marked_pnl"), "reproduced": k in marks} for k, f in fills.items() if k not in exits]
        pnl = [float(t["pnl"] or 0) for t in trades]
        wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
        peak = run = dd = 0.0
        for p in pnl:
            run += p
            peak = max(peak, run)
            dd = min(dd, run - peak)
        by: dict[str, int] = {}
        for e in missed:
            by[e["body"]["why"]] = by.get(e["body"]["why"], 0) + 1
        data = dict(extra.get("data") or {})
        rep.update(data=data, assumptions=ex.get("assumptions") or {}, signals=len(fills) + len(missed), executed=len(fills), not_executed=len(missed), not_executed_by_reason=by, trades=trades[-500:], open_positions=open_,
                   missed_signals=[{"signal_time": e["body"]["signal_time"], "kind": e["body"]["why"], "reason": e["body"]["reason"], "missing_data": e["kind"] == "missing_data"} for e in missed][-200:],
                   missing_data=[{"session": e["session"], "why": e["body"]["why"], "reason": e["body"]["reason"]} for e in led if e["kind"] == "missing_data"][-200:],
                   metrics={"trades": len(pnl), "net_pnl": sum(pnl), "win_rate": (len(wins) / len(pnl)) if pnl else None, "expectancy": (sum(pnl) / len(pnl)) if pnl else None,
                            "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None, "max_drawdown": dd, "fees": sum(float(t.get("fees") or 0) for t in trades)})
        rep["comparison"], notes = _compare_strategy(ex.get("benchmark") or {}, rep, len(sess))
        if open_:
            notes.insert(0, f"{len(open_)} position{'s are' if len(open_) != 1 else ' is'} still open. {'They are' if len(open_) != 1 else 'It is'} not in the P&L until {'they close' if len(open_) != 1 else 'it closes'}; the entry is already sealed.")
        rep["notes"] += notes
        return rep
    b = ex.get("benchmark") or {}
    obs = [e for e in led if e["kind"] == "observation"]
    conds = [e for e in led if e["kind"] == "condition"]
    known = {e["key"] for e in obs}
    rep.update(conditions=len(conds), pending_outcomes=[e["session"] for e in conds if e["key"] not in known][-60:])
    if rep["pending_outcomes"]:
        rep["notes"].append(f"The condition held on {len(rep['pending_outcomes'])} session{'s' if len(rep['pending_outcomes']) != 1 else ''} whose outcome is not known yet. {'They are' if len(rep['pending_outcomes']) != 1 else 'It is'} counted once it is.")
    hit_names = _hit_names(obs, extra)
    n, hits = len(obs), sum(1 for e in obs if e["body"]["outcome"] in hit_names)
    ci = R.wilson(hits, n) if n else [None, None]
    evaluated = int(extra.get("evaluated") or 0)
    rep.update(signals=n, hits=hits, probability=(hits / n) if n else None, evaluated_sessions=evaluated, probability_ci95=ci, sessions=[{"date": e["session"], **e["body"], "sealed_at": e["recorded_at"]} for e in obs][-400:],
               data={"sessions": len(window), "evaluated": evaluated, "not_evaluable": max(0, len(window) - evaluated)},
               comparison={"forward_n": n, "backtest_n": b.get("n"), "forward_probability": (hits / n) if n else None, "backtest_probability": b.get("probability"), "enough": n >= MIN_FORWARD, "consistent": None,
                           "forward_signals_per_100_sessions": n / max(1, evaluated) * 100 if evaluated else None, "backtest_signals_per_100_sessions": b.get("signals_per_100_sessions")})
    if evaluated < len(window):
        rep["notes"].append(f"{len(window) - evaluated} of the {len(window)} forward sessions could not be evaluated with the bars stored (usually an intraday decision time for which no intraday bars were collected, or an outcome that is not known yet).")
    if n < MIN_FORWARD:
        rep["notes"].append(f"The condition has occurred {n} time{'s' if n != 1 else ''} since activation. Fewer than {MIN_FORWARD} occurrences cannot confirm or contradict the historical frequency; this is a record, not a verdict.")
    elif b.get("probability") is not None and ci[0] is not None:
        rep["comparison"]["consistent"] = bool(ci[0] <= b["probability"] <= ci[1])
        rep["notes"].append(f"Forward frequency {hits / n * 100:.0f}% of {n} (95% interval {ci[0] * 100:.0f}% to {ci[1] * 100:.0f}%); the historical frequency was {b['probability'] * 100:.0f}%, which is "
                            + ("inside that interval." if rep["comparison"]["consistent"] else "outside that interval: the forward record differs from the history."))
    return rep


def _hit_names(obs: list[dict], extra: dict) -> set[str]:
    """The outcome label that means the outcome followed (the first of the engine's outcome codes)."""
    from strata.research.features import CODE_NAMES
    return {CODE_NAMES[0]}


def _compare_strategy(b: dict, f: dict, n_days: int) -> tuple[dict, list[str]]:
    m = f["metrics"]
    n = int(m.get("trades") or 0)
    notes = []
    cmp_ = {"forward_trades": n, "backtest_trades": b.get("trades"), "forward_win_rate": m.get("win_rate"), "backtest_win_rate": b.get("win_rate"), "forward_expectancy": m.get("expectancy"), "backtest_expectancy": b.get("expectancy"),
            "forward_signals_per_100_sessions": (f.get("signals") or 0) / max(1, n_days) * 100, "backtest_signals_per_100_sessions": b.get("signals_per_100_sessions"), "enough": n >= MIN_FORWARD, "win_rate_ci95": [None, None],
            "consistent": None}
    if f["data"]["with_option_data"] < f["data"]["sessions"]:
        notes.append(f"{f['data']['sessions'] - f['data']['with_option_data']} of the {f['data']['sessions']} forward sessions have no stored option prices of the kind this strategy is priced on. Signals on those sessions appear as missed, not as trades. "
                     "Option prices for a past session can often no longer be collected (free sources keep intraday option bars only for contracts still listed), so run the collector while the experiment is active. "
                     "A session is sealed with the prices stored when it is first evaluated; prices that arrive later do not change it.")
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


# ----------------------------------------------------------------------------- data for the sessions about to be sealed
def _refresh(ex: dict, days: list[date], store: MarketStore) -> dict:
    """Fetch what the frozen configuration needs for the sessions since activation, from free sources only: the
    underlying's bars and, for a strategy, the option observations its signals select. Nothing is fetched in offline
    mode and nothing from a paid provider. A failure is reported and never stops the evaluation."""
    from strata.data import planner as P
    from strata.data.providers import sources as S
    from strata.research import pipeline as PL
    out: dict = {"ran": False, "downloads": [], "errors": []}
    if PL.offline():
        out["why"] = "offline mode: nothing is downloaded, the evaluation uses what is stored"
        return out
    out["ran"] = True
    a, b = days[0], days[-1]

    def run(action: dict):
        try:
            P.execute_action(store, action)
            out["downloads"].append({"source": action["source"], "dataset": action["dataset"], "symbols": action["symbols"], "why": action.get("why")})
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"{action.get('label') or action['source']} {action['dataset']}: {str(e)[:160]}")

    try:
        req = None
        syms = {ex["symbol"]: {}}
        if ex["kind"] == "strategy":
            cfg = {**ex["config"], "start": a.isoformat(), "end": b.isoformat()}
            mode = (cfg.get("pricing") or {}).get("mode", "historical")
            req = P.requirements("options", cfg, mode if mode in ("historical", "trades", "modeled") else "historical")
            syms = {**req["underlying"], **req["aux"]}
        else:
            import re
            known = set(store.list_symbols()) if hasattr(store, "list_symbols") else set()
            for s_ in set(re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", json.dumps(ex["config"]))) & known:
                syms.setdefault(s_, {})
        for sym in syms:
            if store.is_synthetic(sym):
                continue
            d = store.read_daily(sym) if store.has_daily(sym) else None
            last = d["date"][-1] if d is not None and not d.is_empty() else None
            if last is not None and last >= b:
                continue
            src = S.Cboe(store) if sym in S.Cboe.FILES else S.Yahoo(store)
            run(P._action(src, "vix_daily" if isinstance(src, S.Cboe) else "daily", [sym], None, None, f"{sym} daily history ends {last}", 0))
        if req is not None:
            cov = P.underlying_coverage(store, req, a, b)
            cov.pop("_usable_days", None)
            acts, _ = P.underlying_actions(store, req, cov)
            for act in acts:
                if act["available"] and not act["large"] and act["dataset"] not in ("daily", "vix_daily"):
                    run(act)
            mode = (ex["config"].get("pricing") or {}).get("mode", "historical")
            provider = (ex["config"].get("data") or {}).get("options_provider")
            if mode in ("historical", "trades"):
                from strata.engine.scan import scan_signals
                signals = scan_signals({**ex["config"], "start": a.isoformat(), "end": b.isoformat(), "pricing": {**(ex["config"].get("pricing") or {}), "mode": "historical"}}, store)["signals"]
                out["signals_seen"] = len(signals)
                if signals and mode == "trades":
                    plan = P.trade_plan(store, req, signals, provider)
                    info = next((x for x in plan["sources"] if x["source"] == plan.get("chosen")), None)
                    if info and info["available"] and info["signals_in_window"]:
                        r = P.acquire_trades(store, req, signals, plan["chosen"], None, None, False)
                        out["downloads"].append({"source": plan["chosen"], "dataset": "option_trades", "symbols": [req["symbol"]], "why": f"option trade prints for {info['signals_in_window']} signal session(s)",
                                                 "result": {k: (r.get("result") or {}).get(k) for k in ("pairs_requested", "pairs_ok", "pairs_empty", "pairs_unavailable", "bars")}, "needs_approval": bool(r.get("needs_approval"))})
                    elif info is not None and not info["available"]:
                        out["errors"].append(f"{info['label']}: not connected, so the option trade prints for {len(signals)} signal session(s) were not fetched")
                elif signals and mode == "historical":
                    qc = P.quote_coverage(store, req, signals, provider)
                    acts, _ = P.quote_actions(store, req, qc)
                    for act in acts:
                        if act["available"] and not act["large"]:
                            run(act)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}"[:200])
    try:
        from strata.data.catalog import invalidate
        from strata.research.features import _CACHE
        invalidate()
        _CACHE.clear()
    except Exception:  # noqa: BLE001
        pass
    return out


def evaluate_active(store: MarketStore | None = None) -> list[dict]:
    """Seal the new sessions of every active experiment (called after a collection cycle, which has just stored what
    the free sources still serve). Failures are reported per experiment."""
    store = store or MarketStore()
    out = []
    for ex in listing():
        if ex["status"] != "active":
            continue
        try:
            if _hash(ex["config"]) == ex["config_hash"] and verify(ex["id"])["ok"] and not pending(ex, store):
                continue  # intact and nothing new to seal; a tampered configuration or ledger falls through and is reported
            rep = evaluate(ex["id"], store)
            out.append({"id": ex["id"], "name": ex["name"], "status": rep.get("status"), "forward_sessions": rep.get("forward_sessions"), "last": rep.get("last"), "newly_sealed": len(rep.get("newly_sealed") or [])})
        except Exception as e:  # noqa: BLE001
            out.append({"id": ex["id"], "name": ex["name"], "status": "error", "error": f"{type(e).__name__}: {e}"[:200]})
    return out
