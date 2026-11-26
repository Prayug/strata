"""The persistent market-data collector.

Free sources serve recent intraday history and then drop it: Yahoo keeps about 30 days of
1-minute bars, and an option contract's minute bars only while it is listed and for about a
day after it expires. The only way to own that history is to collect it while it exists.

The collector is OPTIONAL and off by default. When enabled it runs a cycle after each
session's close (and on demand): for every watch-list symbol it refreshes the underlying's
bars at the configured resolutions and downloads observed option TRADE bars for the
contracts around the money at the configured expirations (0, 1, 3, 7, 14 and 30 days by
default). Everything goes through the same sources, validation and provider-partitioned
storage as the rest of Strata, so a collected session is indistinguishable from one the
research agent downloaded - and it is resumable: pairs already stored are skipped.

It is a data reader only. It issues GET requests for market data, respects each source's
rate limit and a per-cycle request budget, and it never places, changes or reads orders:
there is no code path here, or in the sources it uses, that can reach a trading endpoint.

Honest bookkeeping: a session the collector did not reach while the data was still
available is recorded as MISSED, with the reason; it is never filled in afterwards.

Unattended operation
  restarts      a cycle that was cut off (power loss, container restart) is found by its
                stale heartbeat and marked interrupted; the next cycle carries on, skipping
                every contract-session already stored
  recovery      when cycles were missed, the next one reaches back over the sessions since
                the last successful cycle and collects whatever the provider still serves;
                what it no longer serves is reported as unrecoverable, with the reason
  retry queue   a symbol-session that failed is queued in the database and retried with
                increasing delays; after `max_attempts` it is given up and reported
  health        each provider is probed with one small request before it is used, so an
                outage costs one request instead of a budget
  integrity     every file a cycle wrote is read back and checked against its index
  notifications each cycle leaves a note in the interface (Strata sends nothing outward)
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

import polars as pl

from strata.api import db
from strata.core import calendar as cal
from strata.data.store import MarketStore

DEFAULT = {
    "enabled": False,
    "symbols": ["SPY", "QQQ"],
    "bar_resolutions": ["1m", "5m", "60m"],
    "options": {"enabled": True, "dte": [0, 1, 3, 7, 14, 30], "strike_pct": 2.0, "max_strikes_each_side": 12, "rights": ["C", "P"], "provider": "yahoo_options"},
    "run_after": "16:20",  # New York time, on trading days
    "frequency_hours": 24,
    "lookback_sessions": 3,
    "recover_sessions": 10,  # after missed cycles, how far back the next one reaches (bounded by what the provider still serves)
    "max_attempts": 5,  # a symbol-session that keeps failing is given up after this many cycles
    "max_requests_per_cycle": 1500,
    "rate_per_second": {"yahoo": 2.0, "yahoo_options": 4.0, "alpaca": 3.0, "massive": 0.083},
    "storage_limit_gb": 20.0,
    "cleanup": {"enabled": False, "keep_underlying_1m_days": 0, "keep_option_trade_days": 0},
}
LICENSE_NOTES = {
    "yahoo": "Yahoo Finance data is for personal, non-commercial research under Yahoo's terms of service. It may not be redistributed.",
    "yahoo_options": "Yahoo Finance data is for personal, non-commercial research under Yahoo's terms of service. It may not be redistributed.",
    "alpaca": "Alpaca market data is licensed to the account holder under Alpaca's market data agreement and may not be redistributed.",
}
_lock = threading.Lock()


# ----------------------------------------------------------------------------- configuration
def load_config() -> dict:
    from strata.api.settings import get_setting
    raw = get_setting("collector.config")
    cur = json.loads(raw) if isinstance(raw, str) and raw else (raw if isinstance(raw, dict) else {})
    cfg = json.loads(json.dumps(DEFAULT))
    for k, v in (cur or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def save_config(new: dict) -> dict:
    from strata.api.settings import set_setting
    from strata.data.providers.sources import STRIKE_STEP
    cfg = load_config()
    for k, v in (new or {}).items():
        if k not in DEFAULT:
            raise ValueError(f"unknown collector setting {k!r}")
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            bad = set(v) - set(DEFAULT[k])
            if bad:
                raise ValueError(f"unknown collector setting {k}.{sorted(bad)[0]}")
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["symbols"] = list(dict.fromkeys(str(s).upper().strip() for s in cfg["symbols"] if str(s).strip()))[:20]
    if not all(r in ("1m", "5m", "60m") for r in cfg["bar_resolutions"]):
        raise ValueError("bar resolutions must be chosen from 1m, 5m and 60m")
    o = cfg["options"]
    o["dte"] = sorted({int(x) for x in o["dte"] if 0 <= int(x) <= 60})[:8]
    o["strike_pct"] = min(max(float(o["strike_pct"]), 0.25), 10.0)
    o["max_strikes_each_side"] = min(max(int(o["max_strikes_each_side"]), 1), 40)
    o["rights"] = [r for r in ("C", "P") if r in [str(x).upper()[:1] for x in o["rights"]]] or ["C", "P"]
    if o["provider"] not in ("yahoo_options", "alpaca", "massive"):
        raise ValueError("the option provider must be yahoo_options, alpaca or massive")
    cfg["max_requests_per_cycle"] = min(max(int(cfg["max_requests_per_cycle"]), 50), 20000)
    cfg["storage_limit_gb"] = max(float(cfg["storage_limit_gb"]), 0.5)
    cfg["lookback_sessions"] = min(max(int(cfg["lookback_sessions"]), 1), 20)
    cfg["recover_sessions"] = min(max(int(cfg["recover_sessions"]), 0), 60)
    cfg["max_attempts"] = min(max(int(cfg["max_attempts"]), 1), 20)
    cfg["frequency_hours"] = min(max(float(cfg["frequency_hours"]), 1), 168)
    cal.parse_hhmm(cfg["run_after"])
    for k in cfg["rate_per_second"]:
        cfg["rate_per_second"][k] = min(max(float(cfg["rate_per_second"][k]), 0.05), 10.0)
    set_setting("collector.config", json.dumps(cfg))
    from strata.data.providers.sources import SOURCES
    lists = bool(getattr(SOURCES.get(o["provider"]), "keeps_expired", False))  # a provider with a contract listing needs no strike grid
    no_opts = [] if lists else [s for s in cfg["symbols"] if s not in STRIKE_STEP]
    return {"config": cfg, "notes": [f"Option contracts are only collected for {', '.join(STRIKE_STEP)} from this source: their strike grids are configured. {', '.join(no_opts)} will have underlying bars collected only."] if no_opts and o["enabled"] else []}


# ----------------------------------------------------------------------------- one cycle
def wanted_contracts(store: MarketStore, sym: str, sessions: list[date], cfg: dict, today: date | None = None) -> tuple[dict[date, list], dict]:
    """The (session, contract) pairs one cycle asks for, and how the request was bounded.

    Strikes are chosen around the PREVIOUS close of each session (information that existed
    before it), nearest the money first; expirations are the first trading day at or after
    session + N days for each configured N."""
    from strata.data.providers import sources as S
    o = cfg["options"]
    step = S.STRIKE_STEP.get(sym)
    info = {"symbol": sym, "sessions": [d.isoformat() for d in sessions], "contracts": 0, "trimmed": False, "skipped": []}
    if step is None:
        info["skipped"].append(f"{sym}: no strike grid is configured, so option contracts cannot be constructed")
        return {}, info
    today = today or datetime.now(cal.ET).date()
    out: dict[date, list] = {}
    for d in sessions:
        ref = S.previous_close(store, sym, d)
        if ref is None:
            info["skipped"].append(f"{d}: no close before the session is stored, so the strike range cannot be chosen")
            continue
        atm = round(ref / step) * step
        n = min(int(o["max_strikes_each_side"]), int(ref * o["strike_pct"] / 100 / step))
        ks = sorted((atm + i * step for i in range(-n, n + 1)), key=lambda k: abs(k - atm))
        exps = sorted({next((e for e in cal.trading_days(d + timedelta(days=int(b)), d + timedelta(days=int(b) + 6))), None) for b in o["dte"]} - {None})
        exps = [e for e in exps if e >= today - timedelta(days=1)]  # older ones were already purged by a source that only serves listed contracts
        cs = [S.contract_from_occ(S.occ_symbol(sym, e, r, round(k, 3)), source="constructed from the OCC symbol; existence verified by the fetch")
              for k in ks for e in exps for r in o["rights"]]  # nearest the money first across every expiration, so a budget trims the wings
        out[d] = cs
        info["contracts"] += len(cs)
    return out, info


def keeps_history(src) -> bool:
    """A provider that lists its contracts and still serves them after they expire (Alpaca, measured): a session that was
    missed can be collected later, so nothing has to be given up as unrecoverable."""
    return bool(getattr(src, "keeps_expired", False))


def listing_universe(cfg: dict) -> tuple[float, int]:
    """The strike band (% of the previous close) and the furthest expiration (days) a listing provider is asked for."""
    o = cfg["options"]
    return float(o["strike_pct"]), max([int(x) for x in o["dte"]] or [0])


def listed_contracts(store: MarketStore, src, sym: str, sessions: list[date], cfg: dict) -> tuple[dict[date, list], dict]:
    """For a provider with a contract listing: every LISTED contract within `strike_pct` of the previous close that expires
    within the largest configured number of days. It is the same selection as `strata download ... option_trades --band N
    --max-dte M` (one function serves both), so a collected session is indistinguishable from a downloaded one.

    The strike range comes from the close of the session BEFORE each one. A session whose previous close is not stored is
    not guessed at: it is reported in `failed` and retried."""
    from strata.data.providers import sources as S
    band, max_dte = listing_universe(cfg)
    rights = set(cfg["options"]["rights"])
    info = {"symbol": sym, "sessions": [d.isoformat() for d in sessions], "contracts": 0, "trimmed": False, "skipped": [], "failed": {},
            "universe": f"every listed contract within {band:g}% of the previous close that expires within {max_dte} days"}
    out: dict[date, list] = {}
    daily = store.read_daily(sym)
    known = set(daily["date"].to_list()) if not daily.is_empty() else set()
    failed = info["failed"]  # keyed by the session's ISO date: the report is stored as JSON
    with S._client() as client:
        for d in sessions:
            prev = cal.next_trading_day(d, -1)
            if prev not in known:
                failed[d.isoformat()] = f"the close of {prev} is not stored, so the strike range of {d} cannot be chosen without guessing"
                continue
            try:
                cs = [c for c in src.bulk_contracts(client, sym, d, band, max_dte) if c.right in rights]
            except S.SourceError as e:
                failed[d.isoformat()] = f"the contract listing failed ({str(e)[:140]})"
                continue
            if not cs:
                failed[d.isoformat()] = "the contract listing returned nothing for this session"
                continue
            out[d] = cs
            info["contracts"] += len(cs)
    info["skipped"] = [f"{d}: {why}" for d, why in failed.items()]
    return out, info


def sessions_behind(store: MarketStore, sym: str, provider: str, last_done: date, cap: int = 60) -> list[date]:
    """The last session stored for this provider's option data and every completed session after it. The last stored one is
    included because a cycle that was cut off may have left it unfinished; contract-sessions already stored are skipped."""
    idx = store.option_bar_index(sym, provider=provider)
    if idx.is_empty():
        return []
    last = idx["date"].max()
    return [d for d in cal.trading_days(last, last_done)][:cap]


STALE_SECONDS = 900  # a running cycle with no heartbeat for this long was cut off


def recover_interrupted() -> list[int]:
    """Mark cycles that were cut off (no heartbeat for a while) as interrupted so a new one can start."""
    now = datetime.now(timezone.utc)
    out = []
    with db.connect() as con:
        for r in con.execute("SELECT id, started_at, report FROM collector_runs WHERE status='running'").fetchall():
            try:
                rep = json.loads(r["report"] or "{}")
            except json.JSONDecodeError:
                rep = {}
            beat = rep.get("heartbeat") or r["started_at"]
            try:
                age = (now - datetime.fromisoformat(beat)).total_seconds()
            except ValueError:
                age = STALE_SECONDS + 1
            if age > STALE_SECONDS:
                rep.setdefault("notes", []).append("This cycle stopped without finishing (the process was restarted or lost power). What it had stored is kept; the next cycle continues from there.")
                con.execute("UPDATE collector_runs SET status='interrupted', finished_at=?, report=? WHERE id=?", (db.now(), json.dumps(rep, default=str), r["id"]))
                out.append(int(r["id"]))
    return out


def _other_running(own: int | None = None) -> dict | None:
    with db.connect() as con:
        r = con.execute("SELECT id, started_at, report FROM collector_runs WHERE status='running'" + (" AND id != ?" if own else "") + " ORDER BY id DESC LIMIT 1", ([own] if own else [])).fetchone()
    return dict(r) if r else None


def _beat(run_id: int, rep: dict) -> None:
    rep["heartbeat"] = db.now()
    with db.connect() as con:
        con.execute("UPDATE collector_runs SET report=? WHERE id=?", (json.dumps({"heartbeat": rep["heartbeat"], "pid": os.getpid(), "requests": rep.get("requests", 0)}), run_id))


# ----------------------------------------------------------------------------- the retry queue
def _backoff(attempts: int) -> timedelta:
    return timedelta(minutes=min(24 * 60, 30 * 2 ** max(0, attempts - 1)))  # 30 min, 1 h, 2 h, ... capped at a day


def enqueue(kind: str, symbol: str, session: date, provider: str, error: str, max_attempts: int = 5) -> dict:
    """Record a symbol-session that failed; it is retried with increasing delays."""
    now = datetime.now(timezone.utc)
    with db.connect() as con:
        row = con.execute("SELECT id, attempts FROM collector_queue WHERE kind=? AND symbol=? AND session=? AND item=''", (kind, symbol, session.isoformat())).fetchone()
        n = (row["attempts"] if row else 0) + 1
        status = "abandoned" if n >= max_attempts else "pending"
        nb = (now + _backoff(n)).isoformat(timespec="seconds")
        if row:
            con.execute("UPDATE collector_queue SET attempts=?, status=?, last_error=?, not_before=?, provider=?, updated_at=? WHERE id=?", (n, status, error[:300], nb, provider, db.now(), row["id"]))
        else:
            con.execute("INSERT INTO collector_queue (kind, symbol, session, item, provider, status, attempts, last_error, not_before, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (kind, symbol, session.isoformat(), "", provider, status, n, error[:300], nb, db.now(), db.now()))
    return {"attempts": n, "status": status, "not_before": nb}


def resolve(kind: str, symbol: str, session: date, status: str = "done", note: str = "") -> None:
    with db.connect() as con:
        con.execute("UPDATE collector_queue SET status=?, last_error=?, updated_at=? WHERE kind=? AND symbol=? AND session=? AND status IN ('pending', 'abandoned')", (status, note[:300], db.now(), kind, symbol, session.isoformat()))


def queue(status: str | None = None, limit: int = 200) -> list[dict]:
    with db.connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM collector_queue" + (" WHERE status=?" if status else "") + " ORDER BY id DESC LIMIT ?", ([status] if status else []) + [int(limit)])]


def due_sessions(symbol: str, now: datetime | None = None) -> list[date]:
    now = now or datetime.now(timezone.utc)
    with db.connect() as con:
        rows = con.execute("SELECT session, not_before FROM collector_queue WHERE kind='options' AND symbol=? AND status='pending'", (symbol,)).fetchall()
    return sorted(date.fromisoformat(r["session"]) for r in rows if not r["not_before"] or datetime.fromisoformat(r["not_before"]) <= now)


# ----------------------------------------------------------------------------- health and integrity
def health(src) -> dict:
    """One small request to a provider before it is used. A source without a probe is assumed reachable."""
    fn = getattr(src, "health", None)
    if fn is None:
        return {"source": getattr(src, "name", "?"), "ok": True, "detail": "no probe"}
    t0 = time.time()
    try:
        info = fn() or {}
        return {"source": src.name, "ok": True, "seconds": round(time.time() - t0, 2), "detail": info.get("detail", "reachable")}
    except Exception as e:  # noqa: BLE001
        return {"source": src.name, "ok": False, "seconds": round(time.time() - t0, 2), "detail": f"{type(e).__name__}: {e}"[:200]}


def integrity(store: MarketStore, sym: str, provider: str, sessions: list[date]) -> dict:
    """Read back what is stored for these sessions and check it against the index:
    readable files, one row per (time, contract), bar counts that match, times inside the
    session, positive prices."""
    out = {"symbol": sym, "sessions_checked": 0, "problems": []}
    try:
        idx = store.option_bar_index(sym, provider=provider)
    except Exception as e:  # noqa: BLE001
        return {**out, "problems": [f"the index could not be read: {e}"[:200]]}
    if idx is None or idx.is_empty():
        return out
    for d in sessions:
        want = idx.filter((pl.col("date") == d) & (pl.col("status") == "ok"))
        if want.is_empty():
            continue
        out["sessions_checked"] += 1
        try:
            df = store.read_option_bars(sym, d, provider=provider)
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"{d}: the stored file cannot be read ({type(e).__name__})")
            continue
        if df is None or df.is_empty():
            out["problems"].append(f"{d}: the index lists {want.height} contracts with trades but no bars are stored")
            continue
        dup = df.height - df.unique(subset=["ts", "occ"]).height
        if dup:
            out["problems"].append(f"{d}: {dup} duplicate bars")
        a, b = (x.astimezone(timezone.utc) for x in cal.session_bounds(d))
        outside = df.filter((pl.col("ts") < a) | (pl.col("ts") > b)).height
        if outside:
            out["problems"].append(f"{d}: {outside} bars lie outside the session")
        if df.filter((pl.col("close") <= 0) | (pl.col("high") < pl.col("low"))).height:
            out["problems"].append(f"{d}: bars with a non-positive price or high below low")
        got = {r["occ"]: r["n"] for r in df.group_by("occ").agg(pl.len().alias("n")).to_dicts()}
        bad = [r["occ"] for r in want.to_dicts() if got.get(r["occ"], 0) != r["bars"]]
        if bad:
            out["problems"].append(f"{d}: {len(bad)} contract{'s' if len(bad) != 1 else ''} whose stored bars do not match the index (first: {bad[0]})")
    return out


RECOVERABILITY = [
    {"data": "Daily bars (underlying, VIX)", "source": "Yahoo, Cboe", "window": "the whole history, at any time", "after_a_miss": "Fully recoverable."},
    {"data": "60-minute bars (underlying)", "source": "Yahoo", "window": "about 730 days back", "after_a_miss": "Recoverable for about two years."},
    {"data": "5-minute bars (underlying)", "source": "Yahoo", "window": "about 60 days back", "after_a_miss": "Recoverable for about two months, then gone from this source."},
    {"data": "1-minute bars (underlying)", "source": "Yahoo", "window": "about 30 days back, 7 days per request", "after_a_miss": "Recoverable for about a month, then gone from this source. A free Alpaca key reaches 2016."},
    {"data": "Option 1-minute trade bars", "source": "Yahoo", "window": "contracts still listed, or expired within about a day; the last ~7 days of bars",
     "after_a_miss": "A contract that has expired more than about a day ago is PERMANENTLY unavailable from this source. Same-day (0DTE) contracts must be collected on the day itself or the next morning."},
    {"data": "Option 1-minute trade bars", "source": "Alpaca (free key)", "window": "any contract, listed or expired, back to 2024-01-18 (measured)", "after_a_miss": "Recoverable at any time: the next cycle collects every session since the last stored one."},
    {"data": "Option 1-minute trade aggregates", "source": "Massive (free key)", "window": "any contract, about two years back", "after_a_miss": "Recoverable for about two years."},
    {"data": "Option bid/ask quotes during the session", "source": "no free source", "window": "not available", "after_a_miss": "Cannot be collected by Strata at all without a paid feed; it is never estimated."},
    {"data": "End-of-day option quotes", "source": "public archive (to December 2025), DoltHub (2019 onward)", "window": "static archive; DoltHub updates daily", "after_a_miss": "Recoverable."},
]


def run_once(progress=None, should_cancel=None, trigger: str = "manual", store: MarketStore | None = None, today: date | None = None) -> dict:
    """One collection cycle. Returns the report that is also stored in the collection history."""
    from strata.data.catalog import invalidate
    from strata.data.providers import sources as S
    if not _lock.acquire(blocking=False):
        raise RuntimeError("a collection cycle is already running in this process")
    try:
        recover_interrupted()
        store = store or MarketStore()
        cfg = load_config()
        started = db.now()
        with db.connect() as con:
            # the check and the claim are one step: the application and a collector beside it share this database, and
            # two processes that both looked first and wrote second would both have found nothing running
            con.execute("BEGIN IMMEDIATE")
            other = con.execute("SELECT id, started_at FROM collector_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
            if other is not None:
                con.rollback()
                raise RuntimeError(f"another collector process is running (cycle {other['id']}, started {other['started_at']}); two cycles never run at once")
            run_id = con.execute("INSERT INTO collector_runs (started_at, status, trigger, report) VALUES (?,?,?,?)", (started, "running", trigger, json.dumps({"heartbeat": started, "pid": os.getpid()}))).lastrowid
    except BaseException:
        _lock.release()
        raise
    rep = {"trigger": trigger, "started_at": started, "symbols": {}, "requests": 0, "bars": 0, "errors": [], "notes": [], "bytes_before": store.storage_bytes(), "health": [], "integrity": [], "recovered": [], "unrecoverable": []}
    status = "done"
    t0 = time.time()
    try:
        today = today or datetime.now(cal.ET).date()
        limit = cfg["storage_limit_gb"] * 1e9
        if rep["bytes_before"] > limit:
            if cfg["cleanup"]["enabled"]:
                rep["cleanup"] = cleanup(store, dry_run=False)
            if store.storage_bytes() > limit:
                rep["notes"].append(f"Storage is at {store.storage_bytes() / 1e9:.1f} GB, above the {cfg['storage_limit_gb']:g} GB limit: nothing was collected. Raise the limit or enable clean-up; data is never deleted unless clean-up is switched on.")
                status = "skipped"
                return rep
        last_done = today if cal.is_trading_day(today) and datetime.now(cal.ET) >= cal.session_bounds(today)[1] else cal.next_trading_day(today, -1)
        sessions = [d for d in cal.trading_days(last_done - timedelta(days=cfg["lookback_sessions"] * 2 + 6), last_done)][-cfg["lookback_sessions"]:]
        # cycles were missed? reach back over the sessions since the last one that finished (bounded), and collect what is still served
        prev_ok = next((r for r in history(20) if r["status"] in ("done", "partial") and r["id"] != run_id), None)
        gap: list[date] = []
        if prev_ok is not None and cfg["recover_sessions"]:
            since = datetime.fromisoformat(prev_ok["started_at"]).astimezone(cal.ET).date()
            gap = [d for d in cal.trading_days(since, last_done) if d < sessions[0]][-int(cfg["recover_sessions"]):]
            if gap:
                rep["notes"].append(f"No cycle finished between {since} and now: reaching back over {len(gap)} earlier session{'s' if len(gap) != 1 else ''} ({gap[0]} to {gap[-1]}) to collect what the provider still serves.")
        syms = cfg["symbols"]
        ysrc = S.get_source("yahoo", store)
        opt_src = next((s for s in S.option_bar_sources(store) if s.name == cfg["options"]["provider"]), None)
        budget = int(cfg["max_requests_per_cycle"])
        hy = health(ysrc)
        ho = health(opt_src) if opt_src is not None and cfg["options"]["enabled"] and opt_src.available()[0] else None
        rep["health"] = [h for h in (hy, ho) if h and h.get("detail") != "no probe"]
        for i, sym in enumerate(syms):
            if should_cancel and should_cancel():
                raise S.Cancelled()
            one = {"underlying": {}, "options": None}
            rep["symbols"][sym] = one
            base = i / len(syms)
            if progress:
                progress(base, f"{sym}: price bars")
            _beat(run_id, rep)
            ysrc.rate = cfg["rate_per_second"].get("yahoo", 2.0)
            if not hy["ok"]:
                rep["errors"].append(f"{sym} price bars: the provider did not answer its health probe ({hy['detail']}); skipped this cycle")
            for ds in ("daily", "intraday") if hy["ok"] else ():
                try:
                    r = ysrc.ingest(ds, [sym], should_cancel=should_cancel)
                    one["underlying"][ds] = {k: v for k, v in (r.get("symbols", {}).get(sym) or r).items() if isinstance(v, (int, float, str, list)) and k != "symbol"} if isinstance(r, dict) else {}
                    rep["requests"] += 1 if ds == "daily" else 3
                except S.SourceError as e:
                    rep["errors"].append(f"{sym} {ds} bars: {str(e)[:160]}")
            asrc = S.get_source("alpaca", store)
            if "1m" in cfg["bar_resolutions"] and getattr(asrc, "name", "") == "alpaca" and asrc.available()[0]:
                # the consolidated tape's minute bars, continued from the last stored session (the free feed above keeps a month)
                try:
                    have_ = store.datasets(sym, "underlying", res="1m", provider="alpaca")
                    done_ = store.complete_sessions(have_[0]) if have_ else set()
                    start_ = (max(done_) + timedelta(days=1)) if done_ else last_done - timedelta(days=45)  # a first cycle takes the recent weeks; the full history is `strata download alpaca bars_1m`
                    asrc.rate = cfg["rate_per_second"].get("alpaca", asrc.rate)
                    r = asrc.stock_bars(sym, start_, last_done, None, should_cancel) if start_ <= last_done else {"bars": 0, "sessions": 0, "sessions_cached": 0, "pages": 0, "feed": "sip"}
                    one["underlying"]["alpaca_1m"] = {**{k: r.get(k) for k in ("bars", "sessions", "sessions_cached", "pages", "feed")}, "from": start_.isoformat(), "through": last_done.isoformat()}
                    rep["requests"] += int(r.get("pages") or 0)
                    rep["underlying_bars"] = rep.get("underlying_bars", 0) + int(r.get("bars") or 0)
                except S.SourceError as e:
                    rep["errors"].append(f"{sym} 1-minute bars (alpaca): {str(e)[:160]}")
            if not cfg["options"]["enabled"]:
                continue
            if opt_src is None or not opt_src.available()[0]:
                one["options"] = {"skipped": f"{cfg['options']['provider']} is not available: {opt_src.available()[1] if opt_src else 'unknown source'}"}
                continue
            if ho is not None and not ho["ok"]:
                one["options"] = {"skipped": f"{opt_src.name} did not answer its health probe ({ho['detail']}); this cycle's sessions are queued for retry"}
                rep["errors"].append(f"{sym} options: provider unreachable; queued for retry")
                for d in sessions:
                    enqueue("options", sym, d, opt_src.name, ho["detail"], cfg["max_attempts"])
                continue
            a, b = opt_src.bars_window(today)
            retry = [d for d in due_sessions(sym) if d not in sessions]
            target = sorted(set(sessions) | set(gap) | set(retry))
            listing = keeps_history(opt_src)
            if listing:
                # nothing expires out of reach here: every session since the last stored one is collected, however long the collector was off
                behind = sessions_behind(store, sym, opt_src.name, last_done)
                extra = [d for d in behind if d not in target]
                if extra:
                    rep["notes"].append(f"{sym}: {len(extra)} earlier session{'s' if len(extra) != 1 else ''} since the last stored one ({extra[0]} to {extra[-1]}) {'are' if len(extra) != 1 else 'is'} collected too; {opt_src.name} still serves expired contracts.")
                target = sorted(set(target) | set(behind))
                wanted, info = listed_contracts(store, opt_src, sym, [d for d in target if a <= d <= b], cfg)
                rep["requests"] += len(wanted) + len(info["failed"])
                for d, why in info["failed"].items():
                    rep["errors"].append(f"{sym} {d}: {why}; queued for retry")
                    enqueue("options", sym, date.fromisoformat(d), opt_src.name, why, cfg["max_attempts"])
            else:
                wanted, info = wanted_contracts(store, sym, [d for d in target if a <= d <= b], cfg, today)
            lost = [d for d in target if not (a <= d <= b) or (d in wanted and not wanted[d])]
            for d in lost:
                wanted.pop(d, None)
                why = f"{opt_src.name} has no option history outside {a} to {b}" if listing else f"{opt_src.name} no longer serves the contracts of this session (its window is {a} to {b}, and expired contracts drop out)"
                rep["unrecoverable"].append({"symbol": sym, "date": d.isoformat(), "reason": why})
                resolve("options", sym, d, "unrecoverable", "the provider does not serve this session's contracts")
            est = opt_src.estimate(wanted, sym) if wanted else {"requests": 0, "pairs_to_fetch": 0, "pairs_cached": 0, "pairs_requested": 0}
            share = max(50, (budget - rep["requests"]) // max(1, len(syms) - i))
            if est["requests"] > share and wanted and listing:
                # a session is collected whole or not at all, oldest first: what does not fit waits for the next cycle and is still served then
                kept: dict[date, list] = {}
                used, fetching = 0, False
                for d in sorted(wanted):
                    n = opt_src.estimate({d: wanted[d]}, sym)["requests"]
                    if fetching and used + n > share:
                        break  # at least one session with something left to fetch is always taken, so a cycle always moves forward
                    kept[d], used, fetching = wanted[d], used + n, fetching or n > 0
                left = [d for d in sorted(wanted) if d not in kept]
                wanted = kept
                if left:
                    info["deferred"] = [d.isoformat() for d in left]
                    rep["notes"].append(f"{sym}: {len(left)} session{'s' if len(left) != 1 else ''} ({left[0]} to {left[-1]}) did not fit in {budget} requests per cycle and {'are' if len(left) != 1 else 'is'} left for the next cycle; nothing is lost, the provider still serves them.")
            elif est["requests"] > share and wanted:
                # stay inside the request budget by keeping the strikes nearest the money (they come first in each list)
                keep = max(0.1, share / est["requests"])
                wanted = {d: cs[:max(2, int(len(cs) * keep))] for d, cs in wanted.items()}
                info["trimmed"] = True
                rep["notes"].append(f"{sym}: the request was cut to the strikes nearest the money to stay within {budget} requests per cycle.")
            opt_src.rate = cfg["rate_per_second"].get(opt_src.name, opt_src.rate)
            try:
                sub = (lambda f, m, base=base: progress(base + (0.1 + 0.9 * f) / len(syms), f"{sym}: {m}")) if progress else None
                r = opt_src.acquire(sym, wanted, sub, should_cancel) if wanted else {}
                one["options"] = {**info, **{k: r.get(k) for k in ("pairs_requested", "pairs_cached", "pairs_ok", "pairs_empty", "pairs_unavailable", "pairs_failed", "bars", "requests", "rejected")}, "errors": (r.get("errors") or [])[:5]}
                rep["requests"] += int(r.get("requests") or 0)
                rep["bars"] += int(r.get("bars") or 0)
                if r.get("pairs_failed"):
                    rep["errors"].append(f"{sym}: {r['pairs_failed']} contract-sessions failed and will be retried next cycle")
                    for d in wanted:
                        q_ = enqueue("options", sym, d, opt_src.name, "; ".join((r.get("errors") or ["request failed"])[:2]), cfg["max_attempts"])
                        if q_["status"] == "abandoned":
                            rep["notes"].append(f"{sym} {d}: given up after {q_['attempts']} failed cycles.")
                else:
                    for d in wanted:
                        resolve("options", sym, d)
                    rec = [d for d in wanted if d in gap or d in retry or (listing and d < sessions[0])]
                    if rec and (r.get("pairs_ok") or r.get("pairs_cached")):
                        rep["recovered"] += [{"symbol": sym, "date": d.isoformat()} for d in rec]
                chk = integrity(store, sym, opt_src.name, sorted(wanted)) if wanted and hasattr(store, "read_option_bars") else None
                if chk is not None and (chk["sessions_checked"] or chk["problems"]):
                    rep["integrity"].append(chk)
                    if chk["problems"]:
                        rep["errors"].append(f"{sym}: integrity check found {len(chk['problems'])} problem(s): {chk['problems'][0]}")
            except S.Cancelled:
                raise
            except S.SourceError as e:
                one["options"] = {**info, "error": str(e)[:200]}
                rep["errors"].append(f"{sym} options: {str(e)[:160]}")
            try:
                store.rebuild_manifest(sym)
            except Exception:  # noqa: BLE001
                pass
        invalidate(store)
        if rep["errors"]:
            status = "partial"
        return rep
    except Exception as e:  # noqa: BLE001
        status = "cancelled" if type(e).__name__ in ("Cancelled", "BacktestCancelled") else "error"
        rep["errors"].append(f"{type(e).__name__}: {e}"[:300])
        if status == "error":
            raise
        return rep
    finally:
        # whatever happens while the cycle is being written up, the lock is released and the run is closed: a cycle that
        # stayed "running" would block every later one until its heartbeat went stale
        try:
            rep["bytes_after"] = store.storage_bytes()
            rep["seconds"] = round(time.time() - t0, 1)
            rep["status"] = status
            if status in ("done", "partial"):
                try:  # sessions that have just arrived are the forward evidence of any frozen experiment
                    from strata.research import forward as FW
                    rep["forward"] = FW.evaluate_active(store)
                except Exception as e:  # noqa: BLE001
                    rep["forward"] = [{"status": "error", "error": f"{type(e).__name__}: {e}"[:200]}]
            try:
                blob = json.dumps(rep, default=str)
            except (TypeError, ValueError) as e:
                blob = json.dumps({"status": status, "requests": rep.get("requests", 0), "bars": rep.get("bars", 0), "errors": [str(x) for x in rep.get("errors", [])] + [f"the cycle's full report could not be stored ({type(e).__name__}: {e})"[:200]]})
            with db.connect() as con:
                con.execute("UPDATE collector_runs SET finished_at=?, status=?, report=? WHERE id=?", (db.now(), status, blob, run_id))
            _notify(rep, status)
        finally:
            _lock.release()
            if progress:
                progress(1.0, "done")


def _notify(rep: dict, status: str) -> None:
    """One note per cycle in the interface; a second one when something needs attention."""
    try:
        from strata.api import notify as N
        syms = ", ".join(rep.get("symbols") or []) or "nothing"
        if status == "done":
            N.add("collector", f"Collected {rep.get('bars', 0):,} option bars ({syms})", f"{rep.get('requests', 0)} requests in {rep.get('seconds', 0)} s." + (f" Recovered {len(rep['recovered'])} earlier session(s)." if rep.get("recovered") else ""), "info", "/data?tab=collector")
        elif status == "partial":
            N.add("collector", f"Collection finished with {len(rep.get('errors') or [])} problem(s)", "; ".join((rep.get("errors") or [])[:3]) + ". Failed sessions are queued and retried.", "warn", "/data?tab=collector")
        elif status in ("error", "skipped"):
            N.add("collector", "Collection did not run" if status == "skipped" else "Collection failed", "; ".join((rep.get("notes") or [])[:1] + (rep.get("errors") or [])[:2]), "error", "/data?tab=collector")
        if rep.get("unrecoverable"):
            u = rep["unrecoverable"]
            N.add("collector", f"{len(u)} session(s) can no longer be collected", f"{u[0]['symbol']} {u[0]['date']}" + (f" and {len(u) - 1} more" if len(u) > 1 else "") + f": {u[0]['reason']}. They are recorded as missed; nothing is filled in.", "warn", "/data?tab=collector")
        for f in rep.get("forward") or []:
            if f.get("status") == "evaluated":
                N.add("forward", f"Forward experiment evaluated: {f.get('name', '')[:80]}", f"{f.get('forward_sessions')} forward session(s) through {f.get('last')}.", "info", f"/forward/{f.get('id')}")
    except Exception:  # noqa: BLE001  (a note that cannot be written must not fail the cycle)
        pass


# ----------------------------------------------------------------------------- clean-up
def cleanup(store: MarketStore, dry_run: bool = True) -> dict:
    """Apply the retention policy. Only the collector's rolling intraday data is ever
    removed (1-minute underlying bars and option trade bars older than the configured
    number of days); daily history and imported or purchased data are never touched."""
    cfg = load_config()["cleanup"]
    today = datetime.now(cal.ET).date()
    removed, freed = [], 0
    for key in store.datasets():
        keep = cfg["keep_underlying_1m_days"] if (key.dtype == "underlying" and key.res == "1m") else cfg["keep_option_trade_days"] if (key.dtype == "options" and key.obs == "trade") else 0
        if not keep or key.provider not in ("yahoo", "yahoo_options"):
            continue
        cut = today - timedelta(days=int(keep))
        base = store.dataset_dir(key) / "data"
        for f in sorted(base.glob("**/*.parquet")) if base.exists() else []:
            try:
                d = date.fromisoformat(f.stem[:10])
            except ValueError:
                continue
            if d < cut:
                freed += f.stat().st_size
                removed.append({"dataset": key.id, "file": f.name})
                if not dry_run:
                    f.unlink()
        if not dry_run and removed:
            try:
                store.commit(key)
            except Exception:  # noqa: BLE001
                pass
    return {"dry_run": dry_run, "files": len(removed), "bytes": freed, "policy": cfg, "sample": removed[:8],
            "note": "Nothing is deleted while clean-up is disabled." if not cfg["enabled"] and not dry_run else "Only rolling 1-minute bars and option trade bars from the free sources are eligible."}


# ----------------------------------------------------------------------------- reporting
def history(limit: int = 30) -> list[dict]:
    with db.connect() as con:
        rows = [db._row(r) for r in con.execute("SELECT * FROM collector_runs ORDER BY id DESC LIMIT ?", (int(limit),))]
    for r in rows:
        if not isinstance(r.get("report"), dict):
            r["report"] = {}
    return rows


def running() -> bool:
    return bool([r for r in db.list_runs(kind="collector", limit=5) if r["status"] in ("running", "queued")])


def coverage(store: MarketStore, cfg: dict, sessions: int = 45, today: date | None = None) -> dict:
    """Per session and symbol: the finest underlying bars stored, and option contracts with
    trades per expiration bucket. A recent session with no option data whose contracts are
    no longer served is MISSED."""
    from strata.data.providers import sources as S
    today = today or datetime.now(cal.ET).date()
    days = [d for d in cal.trading_days(today - timedelta(days=int(sessions * 1.6) + 10), today)][-sessions:]
    buckets = [int(x) for x in cfg["options"]["dte"]]
    src_cls = S.SOURCES.get(cfg["options"]["provider"])
    recoverable = bool(getattr(src_cls, "keeps_expired", False))  # a session this provider has not stored yet can still be fetched
    out = {"sessions": [d.isoformat() for d in days], "buckets": buckets, "symbols": {}, "missed": [], "waiting": [], "recoverable": recoverable}
    for sym in cfg["symbols"]:
        res_by_day: dict[date, str] = {}
        for res in ("60m", "5m", "1m"):
            try:
                df = store.read_bars(sym, res, days[0], days[-1])
            except Exception:  # noqa: BLE001
                df = pl.DataFrame()
            if df is not None and not df.is_empty():
                for d in df["ts"].dt.convert_time_zone("America/New_York").dt.date().unique().to_list():
                    res_by_day[d] = res
        try:
            idx = store.option_bar_index(sym, provider=cfg["options"]["provider"])
        except Exception:  # noqa: BLE001
            idx = pl.DataFrame()
        per: dict[date, dict] = {}
        if idx is not None and not idx.is_empty():
            x = idx.filter(pl.col("date").is_in(pl.Series(days).implode())).with_columns((pl.col("expiration") - pl.col("date")).dt.total_days().alias("dte"))
            for r in x.group_by(["date", "dte", "status"]).agg(pl.len().alias("n")).to_dicts():
                b = min(buckets, key=lambda k: abs(k - r["dte"])) if buckets else 0
                slot = per.setdefault(r["date"], {}).setdefault(str(b), {"ok": 0, "empty": 0})
                slot["ok" if r["status"] == "ok" else "empty"] += r["n"]
        rows = []
        can_opt = (sym in S.STRIKE_STEP or recoverable) and cfg["options"]["enabled"]
        for d in days:
            o = per.get(d, {})
            traded = sum(v["ok"] for v in o.values())
            age = len([x for x in days if x > d])
            state = "collected" if traded else ("not applicable" if not can_opt else "pending" if age <= 1 else "waiting" if recoverable else "missed")
            rows.append({"date": d.isoformat(), "underlying": res_by_day.get(d), "contracts_with_trades": traded, "contracts_without_trades": sum(v["empty"] for v in o.values()), "by_bucket": o, "state": state})
            if state == "missed":
                out["missed"].append({"symbol": sym, "date": d.isoformat(), "reason": "no option trade bars were collected while the contracts were still being served" + (" (the collector was not enabled)" if not cfg["enabled"] else "")})
            elif state == "waiting":
                out["waiting"].append({"symbol": sym, "date": d.isoformat()})
        out["symbols"][sym] = rows
    return out


def quality(store: MarketStore, cfg: dict) -> list[dict]:
    """Data-quality diagnostics of what the collector has stored."""
    out = []
    for sym in cfg["symbols"]:
        try:
            idx = store.option_bar_index(sym, provider=cfg["options"]["provider"])
        except Exception:  # noqa: BLE001
            idx = pl.DataFrame()
        if idx is None or idx.is_empty():
            out.append({"symbol": sym, "contract_sessions": 0, "note": "no option trade data stored"})
            continue
        n = idx.height
        ok = idx.filter(pl.col("status") == "ok")
        thin = ok.filter(pl.col("bars") < 10).height if "bars" in ok.columns else None
        out.append({"symbol": sym, "contract_sessions": n, "with_trades": ok.height, "without_trades": n - ok.height, "without_trades_pct": round((n - ok.height) / n * 100, 1),
                    "thinly_traded": thin, "thinly_traded_pct": round(thin / max(ok.height, 1) * 100, 1) if thin is not None else None, "sessions": int(idx["date"].n_unique()),
                    "first": str(idx["date"].min()), "last": str(idx["date"].max()), "median_bars_per_contract": float(ok["bars"].median()) if "bars" in ok.columns and ok.height else None,
                    "note": "A contract without trades did not print in the session; a contract with few bars traded in only a few minutes, so its last price is often stale."})
    return out


def next_due(cfg: dict, now: datetime | None = None) -> datetime | None:
    if not cfg["enabled"]:
        return None
    now = now or datetime.now(cal.ET)
    last = next((r for r in history(5) if r["status"] in ("done", "partial")), None)
    last_t = datetime.fromisoformat(last["started_at"]).astimezone(cal.ET) if last else None
    if last_t is not None and cfg["frequency_hours"] < 24 and cal.is_trading_day(now.date()):
        # more than one cycle a day: also collect every N hours while the session is open
        o, c = cal.session_bounds(now.date())
        nxt = last_t + timedelta(hours=float(cfg["frequency_hours"]))
        if o <= max(nxt, now) <= c + timedelta(hours=1):
            return max(nxt, now)
    m = cal.parse_hhmm(cfg["run_after"])
    d = now.date()
    # a completed session that no cycle has covered yet (the computer was off, or the collector has just been switched on) is
    # due at once: the free source drops a 0DTE contract about a day after it expires, so waiting for the next close can lose it
    at = lambda day: datetime.combine(day, datetime.min.time(), cal.ET) + timedelta(minutes=m)  # noqa: E731
    prev = d if cal.is_trading_day(d) and at(d) <= now else cal.next_trading_day(d, -1)
    if last_t is None or last_t < at(prev):
        return now
    for _ in range(10):
        if cal.is_trading_day(d):
            t = datetime.combine(d, datetime.min.time(), cal.ET) + timedelta(minutes=m)
            if last_t is None or last_t < t:
                return t if t > now else now
        d += timedelta(days=1)
    return None


def stored_universe(store: MarketStore, provider: str = "alpaca", recent_days: int = 45) -> dict:
    """What the option data already stored from a provider covers, measured from its index over the recent sessions: which
    symbols, how far from the previous close the strikes reach and how far out the expirations go. Read from what is on
    disk, so a collector that continues the archive asks for the same contracts the archive already holds."""
    out = {"provider": provider, "symbols": {}, "band_pct": None, "max_dte": None}
    for key in store.datasets(dtype="options", obs="trade", provider=provider):
        idx = store.option_bar_index(key.symbol, provider=provider)
        if idx.is_empty():
            continue
        last = idx["date"].max()
        rec = idx.filter(pl.col("date") >= last - timedelta(days=recent_days))
        daily = store.read_daily(key.symbol)
        if daily.is_empty():
            continue
        prev = daily.select("date", pl.col("close").shift(1).alias("_prev")).drop_nulls()
        x = rec.join(prev, on="date", how="inner").with_columns(((pl.col("strike") / pl.col("_prev") - 1) * 100).alias("_off"), (pl.col("expiration") - pl.col("date")).dt.total_days().alias("_dte"))
        if x.is_empty():
            continue
        g = x.group_by("date").agg((-pl.col("_off").min()).alias("lo"), pl.col("_off").max().alias("hi"), pl.col("_dte").max().alias("dte"), pl.len().alias("n"))
        band = float(pl.concat([g["lo"], g["hi"]]).median())
        out["symbols"][key.symbol] = {"sessions": int(idx["date"].n_unique()), "first": str(idx["date"].min()), "last": str(last), "band_pct": round(band, 2),
                                      "max_dte": int(g["dte"].median()), "contracts_per_session": int(g["n"].median()), "rights": sorted(rec["right"].unique().to_list())}
    if out["symbols"]:
        vals = list(out["symbols"].values())
        out["band_pct"] = max(0.25, round(min(v["band_pct"] for v in vals) * 2) / 2)  # the listing rounds the band outward to whole strikes, so the measured reach sits just above the setting
        out["max_dte"] = max(v["max_dte"] for v in vals)
    return out


def continuing_config(store: MarketStore, provider: str = "alpaca") -> dict | None:
    """The collector settings that carry the stored archive forward unchanged: the same symbols, strike band and expirations."""
    u = stored_universe(store, provider)
    if not u["symbols"]:
        return None
    rights = sorted({r for v in u["symbols"].values() for r in v["rights"]})
    return {"symbols": sorted(u["symbols"]), "options": {"enabled": True, "provider": provider, "strike_pct": u["band_pct"], "dte": sorted({b for b in (0, 1, 3, 7, 14, 30) if b < u["max_dte"]} | {u["max_dte"]}), "rights": rights},
            "measured": u}


def option_sources(store: MarketStore) -> list[dict]:
    """The option sources the collector can use, whether each is usable now and what happens after a missed session."""
    from strata.data.providers import sources as S
    out = []
    for src in S.option_bar_sources(store):
        try:
            ok, msg = src.available()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}"
        keeps = keeps_history(src)
        out.append({"name": src.name, "label": getattr(src, "label", src.name), "available": bool(ok), "detail": msg, "keeps_expired": keeps,
                    "after_a_miss": "A missed session is collected by the next cycle: expired contracts stay available." if keeps else "A missed session is lost once its contracts drop out of the source (about a day after they expire)."})
    return out


def state(store: MarketStore | None = None) -> dict:
    store = store or MarketStore()
    cfg = load_config()
    hist = history(30)
    by = {}
    for key in store.datasets():
        if key.symbol in cfg["symbols"]:
            by.setdefault(key.symbol, {})[f"{key.provider} {key.dtype} {key.obs} {key.res}"] = store.storage_bytes(key=key)
    nd = next_due(cfg)
    prov = cfg["options"]["provider"]
    srcs = option_sources(store)
    cur = next((x for x in srcs if x["name"] == prov), None)
    keeps = bool(cur and cur["keeps_expired"])
    band, max_dte = listing_universe(cfg)
    cont = continuing_config(store) if any(x["name"] == "alpaca" and x["available"] for x in srcs) else None
    same = bool(cont) and prov == "alpaca" and set(cont["symbols"]) <= set(cfg["symbols"]) and abs(band - cont["options"]["strike_pct"]) < 1e-9 and max_dte == max(cont["options"]["dte"]) and set(cont["options"]["rights"]) <= set(cfg["options"]["rights"])
    return {"config": cfg, "defaults": DEFAULT, "running": running(), "history": hist, "next_due": nd.isoformat() if nd else None, "last": hist[0] if hist else None, "coverage": coverage(store, cfg), "quality": quality(store, cfg),
            "storage": {"total_bytes": store.storage_bytes(), "limit_bytes": int(cfg["storage_limit_gb"] * 1e9), "by_symbol": by},
            "license": [LICENSE_NOTES["yahoo"]] + ([LICENSE_NOTES[prov]] if prov != "yahoo_options" else []),
            "queue": {"pending": queue("pending", 100), "abandoned": queue("abandoned", 50), "unrecoverable": queue("unrecoverable", 50)}, "recoverability": RECOVERABILITY,
            "sources": srcs, "source": cur, "listing": keeps,
            "universe": (f"Every listed contract within {band:g}% of the previous close that expires within {max_dte} days, calls and puts as chosen." if keeps
                         else "Contracts constructed around the previous close at the chosen expirations, nearest the money first."),
            "continue": None if cont is None else {**cont, "in_effect": same},
            "offline": ("Nothing is collected while the computer is asleep or off, or while Strata is not running. With this source that costs nothing but time: the next cycle collects every session since the last stored one." if keeps
                        else "Nothing is collected while the computer is asleep or off, or while Strata is not running. Run the collector on a machine that stays on (see README: unattended collection) if gaps matter."),
            "safety": "The collector only reads market data (GET requests). It cannot place, change or read orders: no trading endpoint is reachable from this code.",
            "how_to_schedule": "While the Strata server is running the collector starts by itself after each close. To collect without the app open, schedule `strata collect --once` (cron or launchd) on a machine that stays awake."}


# ----------------------------------------------------------------------------- scheduler
class Scheduler(threading.Thread):
    """Submits a collection job when one is due. Lives in the API process; one cycle at a time."""

    def __init__(self, jobs, interval: float = 60.0):
        super().__init__(daemon=True, name="strata-collector")
        self.jobs, self.interval = jobs, interval
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def tick(self) -> str | None:
        cfg = load_config()
        if not cfg["enabled"] or running():
            return None
        recover_interrupted()
        if _other_running() is not None:
            return None  # another process (for example the stand-alone collector container) is mid-cycle
        nd = next_due(cfg)
        if nd is None or nd > datetime.now(cal.ET):
            return None
        last = history(1)
        if last and last[0]["status"] in ("error", "skipped", "cancelled") and (datetime.now(cal.ET) - datetime.fromisoformat(last[0]["started_at"]).astimezone(cal.ET)) < timedelta(hours=1):
            return None  # do not hammer a failing source, or restart a cycle someone has just stopped: wait an hour before trying again
        rid = db.create_run("collector", "Data collection cycle", {"trigger": "schedule"})
        self.jobs.submit(rid, "collector", {"trigger": "schedule"})
        return rid

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001  (the scheduler must survive a bad cycle)
                pass
