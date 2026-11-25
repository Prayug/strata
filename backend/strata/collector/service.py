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
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta

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
    "max_requests_per_cycle": 1500,
    "rate_per_second": {"yahoo": 2.0, "yahoo_options": 4.0, "alpaca": 3.0},
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
    if o["provider"] not in ("yahoo_options", "alpaca"):
        raise ValueError("the option provider must be yahoo_options or alpaca")
    cfg["max_requests_per_cycle"] = min(max(int(cfg["max_requests_per_cycle"]), 50), 20000)
    cfg["storage_limit_gb"] = max(float(cfg["storage_limit_gb"]), 0.5)
    cfg["lookback_sessions"] = min(max(int(cfg["lookback_sessions"]), 1), 20)
    cfg["frequency_hours"] = min(max(float(cfg["frequency_hours"]), 1), 168)
    cal.parse_hhmm(cfg["run_after"])
    for k in cfg["rate_per_second"]:
        cfg["rate_per_second"][k] = min(max(float(cfg["rate_per_second"][k]), 0.2), 10.0)
    set_setting("collector.config", json.dumps(cfg))
    no_opts = [s for s in cfg["symbols"] if s not in STRIKE_STEP]
    return {"config": cfg, "notes": [f"Option contracts are only collected for {', '.join(STRIKE_STEP)}: their strike grids are configured. {', '.join(no_opts)} will have underlying bars collected only."] if no_opts and o["enabled"] else []}


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


def run_once(progress=None, should_cancel=None, trigger: str = "manual", store: MarketStore | None = None, today: date | None = None) -> dict:
    """One collection cycle. Returns the report that is also stored in the collection history."""
    from strata.data.catalog import invalidate
    from strata.data.providers import sources as S
    if not _lock.acquire(blocking=False):
        raise RuntimeError("a collection cycle is already running in this process")
    store = store or MarketStore()
    cfg = load_config()
    started = db.now()
    with db.connect() as con:
        run_id = con.execute("INSERT INTO collector_runs (started_at, status, trigger, report) VALUES (?,?,?,?)", (started, "running", trigger, "{}")).lastrowid
    rep = {"trigger": trigger, "started_at": started, "symbols": {}, "requests": 0, "bars": 0, "errors": [], "notes": [], "bytes_before": store.storage_bytes()}
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
        syms = cfg["symbols"]
        ysrc = S.get_source("yahoo", store)
        opt_src = next((s for s in S.option_bar_sources(store) if s.name == cfg["options"]["provider"]), None)
        budget = int(cfg["max_requests_per_cycle"])
        for i, sym in enumerate(syms):
            if should_cancel and should_cancel():
                raise S.Cancelled()
            one = {"underlying": {}, "options": None}
            rep["symbols"][sym] = one
            base = i / len(syms)
            if progress:
                progress(base, f"{sym}: price bars")
            ysrc.rate = cfg["rate_per_second"].get("yahoo", 2.0)
            for ds in ("daily", "intraday"):
                try:
                    r = ysrc.ingest(ds, [sym], should_cancel=should_cancel)
                    one["underlying"][ds] = {k: v for k, v in (r.get("symbols", {}).get(sym) or r).items() if isinstance(v, (int, float, str, list)) and k != "symbol"} if isinstance(r, dict) else {}
                    rep["requests"] += 1 if ds == "daily" else 3
                except S.SourceError as e:
                    rep["errors"].append(f"{sym} {ds} bars: {str(e)[:160]}")
            if not cfg["options"]["enabled"]:
                continue
            if opt_src is None or not opt_src.available()[0]:
                one["options"] = {"skipped": f"{cfg['options']['provider']} is not available: {opt_src.available()[1] if opt_src else 'unknown source'}"}
                continue
            a, b = opt_src.bars_window(today)
            wanted, info = wanted_contracts(store, sym, [d for d in sessions if a <= d <= b], cfg, today)
            est = opt_src.estimate(wanted, sym) if wanted else {"requests": 0, "pairs_to_fetch": 0, "pairs_cached": 0, "pairs_requested": 0}
            share = max(50, (budget - rep["requests"]) // max(1, len(syms) - i))
            if est["requests"] > share and wanted:
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
        rep["bytes_after"] = store.storage_bytes()
        rep["seconds"] = round(time.time() - t0, 1)
        rep["status"] = status
        with db.connect() as con:
            con.execute("UPDATE collector_runs SET finished_at=?, status=?, report=? WHERE id=?", (db.now(), status, json.dumps(rep, default=str), run_id))
        _lock.release()
        if progress:
            progress(1.0, "done")


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
    out = {"sessions": [d.isoformat() for d in days], "buckets": buckets, "symbols": {}, "missed": []}
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
        can_opt = sym in S.STRIKE_STEP and cfg["options"]["enabled"]
        for d in days:
            o = per.get(d, {})
            traded = sum(v["ok"] for v in o.values())
            age = len([x for x in days if x > d])
            state = "collected" if traded else ("not applicable" if not can_opt else "pending" if age <= 1 else "missed")
            rows.append({"date": d.isoformat(), "underlying": res_by_day.get(d), "contracts_with_trades": traded, "contracts_without_trades": sum(v["empty"] for v in o.values()), "by_bucket": o, "state": state})
            if state == "missed":
                out["missed"].append({"symbol": sym, "date": d.isoformat(), "reason": "no option trade bars were collected while the contracts were still being served" + (" (the collector was not enabled)" if not cfg["enabled"] else "")})
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
    for _ in range(10):
        if cal.is_trading_day(d):
            t = datetime.combine(d, datetime.min.time(), cal.ET) + timedelta(minutes=m)
            if last_t is None or last_t < t:
                return t if t > now else now
        d += timedelta(days=1)
    return None


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
    return {"config": cfg, "defaults": DEFAULT, "running": running(), "history": hist, "next_due": nd.isoformat() if nd else None, "last": hist[0] if hist else None, "coverage": coverage(store, cfg), "quality": quality(store, cfg),
            "storage": {"total_bytes": store.storage_bytes(), "limit_bytes": int(cfg["storage_limit_gb"] * 1e9), "by_symbol": by},
            "license": [LICENSE_NOTES["yahoo"]] + ([LICENSE_NOTES[prov]] if prov != "yahoo_options" else []),
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
        nd = next_due(cfg)
        if nd is None or nd > datetime.now(cal.ET):
            return None
        last = history(1)
        if last and last[0]["status"] in ("error", "skipped") and (datetime.now(cal.ET) - datetime.fromisoformat(last[0]["started_at"]).astimezone(cal.ET)) < timedelta(hours=1):
            return None  # do not hammer a failing source: wait an hour before trying again
        rid = db.create_run("collector", "Data collection cycle", {"trigger": "schedule"})
        self.jobs.submit(rid, "collector", {"trigger": "schedule"})
        return rid

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001  (the scheduler must survive a bad cycle)
                pass
