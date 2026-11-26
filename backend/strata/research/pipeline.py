"""The research pipeline: from an engine configuration to a result, acquiring data on the way.

One job, seven visible stages:

  interpret   Interpreting strategy            (done by the agent before the job starts)
  coverage    Checking historical coverage     requirements, per-session validation of the
                                               underlying, free downloads of missing prices
  signals     Identifying qualifying sessions  stage one: entry rules on the underlying only
  acquire     Downloading missing data         stage two: option data for those sessions only
  contracts   Validating contracts             per-signal, contract-level coverage
  simulate    Running simulation               the deterministic engine
  report      Generating research report       result files + the coverage record

Rules it enforces
  * only free sources are ever called; a paid provider is described, never used
  * an unusually large download is estimated and needs approval first
  * a period the user stated is never silently shortened: if it cannot be covered the job
    stops with the options (expand coverage, accept the reduced window, another mode)
  * option prices are never invented and never switched between quotes, trade prints and
    theoretical values without the mode saying so
  * sessions excluded for missing data are reported separately from sessions with no signal
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timezone

from strata.api.results import dumps, run_dir, save_backtest, save_study
from strata.config import ENGINE_VERSION
from strata.core import calendar as cal
from strata.data import planner as P
from strata.data.providers import sources as S
from strata.data.store import MarketStore
from strata.engine.models import BacktestConfig

STAGES = [("interpret", "Interpreting strategy"), ("coverage", "Checking historical coverage"), ("signals", "Identifying qualifying sessions"),
          ("acquire", "Downloading missing data"), ("contracts", "Validating contracts"), ("simulate", "Running simulation"), ("report", "Generating research report")]
MODE_LABEL = {"study": "Historical market study", "historical": "Historical options backtest (bid/ask quotes)", "trades": "Historical options backtest (trade prices)",
              "modeled": "Modeled options backtest (theoretical premiums)"}
SHORTFALL_TOLERANCE = 0.02


MAX_EXPANSIONS = 3  # adaptive contract discovery: how many times a strike range may be widened for one run


def offline() -> bool:
    """STRATA_OFFLINE=1 turns every automatic download off (used by tests and air-gapped installs)."""
    import os
    return os.environ.get("STRATA_OFFLINE", "") in ("1", "true", "yes")


class NeedsDecision(Exception):
    def __init__(self, reason: str, choices: list[dict], **data):
        super().__init__(reason)
        self.reason, self.choices, self.data = reason, choices, data


def fingerprint(spec: dict, store: MarketStore | None = None) -> str:
    """Identity of a research request given the data that is stored right now. Two launches
    reuse one result only if the request, every dataset it can read, the event calendar,
    the engine version and the last completed session are all the same."""
    store = store or MarketStore()
    syms = _symbols(spec)
    vers = {s: store.dataset_version(s) for s in sorted(syms)}
    body = json.dumps({k: spec.get(k) for k in ("kind", "mode", "study", "backtest", "requested", "approvals", "provider")}, sort_keys=True, default=str)
    return hashlib.sha256((body + json.dumps(vers, sort_keys=True) + store.events_version() + ENGINE_VERSION + P.last_completed_session().isoformat()).encode()).hexdigest()[:24]


def _symbols(spec: dict) -> set[str]:
    req = _req(spec, spec.get("mode") if spec.get("mode") not in (None, "auto") else ("study" if spec["kind"] == "study" else "historical"))
    return set(req["underlying"]) | set(req["aux"])


def _bt_cfg(spec: dict, start: date, end: date, mode: str) -> dict:
    cfg = json.loads(json.dumps(spec["backtest"]))
    cfg["start"], cfg["end"] = start.isoformat(), end.isoformat()
    cfg.setdefault("pricing", {})["mode"] = mode if mode in ("historical", "trades", "modeled") else "historical"
    if spec.get("provider"):
        cfg.setdefault("data", {})["options_provider"] = spec["provider"]
    return cfg


def _req(spec: dict, mode: str) -> dict:
    if mode == "study" or spec["kind"] == "study":
        return P.requirements("study", spec["study"], "study")
    return P.requirements("options", _bt_cfg(spec, date(2000, 1, 3), date(2000, 1, 4), mode), mode)


class Pipeline:
    def __init__(self, rid: str, spec: dict, progress=None, should_cancel=None, store: MarketStore | None = None):
        self.rid, self.spec = rid, spec
        self.progress, self.should_cancel = progress, should_cancel
        self.store = store or MarketStore()
        self.dir = run_dir(rid)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stages = [{"id": i, "label": l, "status": "pending", "detail": "", "data": {}} for i, l in STAGES]
        self.downloads: list[dict] = []
        self.t0 = time.time()

    # ---- progress
    def _write(self, outcome: str | None = None, decision: dict | None = None):
        # written to a temporary file and renamed, so a reader polling progress never sees half a file
        tmp = self.dir / "pipeline.json.tmp"
        tmp.write_text(dumps({"run_id": self.rid, "stages": self.stages, "outcome": outcome, "decision": decision, "downloads": self.downloads,
                              "elapsed_seconds": round(time.time() - self.t0, 2), "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        os.replace(tmp, self.dir / "pipeline.json")

    def stage(self, sid: str, status: str, detail: str | None = None, **data):
        i = next(k for k, s in enumerate(self.stages) if s["id"] == sid)
        s = self.stages[i]
        if status == "running" and s["status"] != "running":
            s["started"] = round(time.time() - self.t0, 2)
        if status in ("done", "skipped", "blocked", "error"):
            s["finished"] = round(time.time() - self.t0, 2)
        s["status"] = status
        if detail is not None:
            s["detail"] = detail
        s["data"].update(data)
        self._write()
        if self.progress:
            self.progress(min(0.98, (i + (1.0 if status in ("done", "skipped") else 0.4)) / len(self.stages)), f"{s['label']}{': ' + s['detail'] if s['detail'] else ''}"[:240])

    def sub(self, sid: str):
        i = next(k for k, s in enumerate(self.stages) if s["id"] == sid)
        s = self.stages[i]

        def cb(frac, msg=""):
            if msg:
                s["detail"] = str(msg)[:240]
                self._write()
            if self.progress:
                self.progress(min(0.98, (i + 0.9 * max(0.0, min(1.0, float(frac or 0)))) / len(self.stages)), f"{s['label']}: {msg}"[:240])
        return cb

    def cancelled(self):
        if self.should_cancel and self.should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()

    # ---- helpers
    def _download(self, action: dict, sid: str) -> dict:
        t = time.time()
        before = self.store.storage_bytes(action["symbols"][0]) if action["symbols"] else 0
        out = P.execute_action(self.store, action, self.sub(sid), self.should_cancel)
        rec = {"source": action["source"], "label": action["label"], "dataset": action["dataset"], "symbols": action["symbols"], "why": action["why"], "tier": action["tier"], "cost": 0.0,
               "seconds": round(time.time() - t, 1), "bytes_added": max(0, (self.store.storage_bytes(action["symbols"][0]) if action["symbols"] else 0) - before), "result": _brief(out)}
        self.downloads.append(rec)
        return rec

    def _ensure_daily(self, req: dict):
        """Daily history is tiny and free: fetch it for any symbol that has none or is stale."""
        last = P.last_completed_session()
        for sym, need in {**req["underlying"], **req["aux"]}.items():
            if self.store.is_synthetic(sym):
                continue
            d = self.store.read_daily(sym) if self.store.has_daily(sym) else None
            have_last = d["date"][-1] if d is not None and not d.is_empty() else None
            if have_last is not None and have_last >= last:
                continue
            src = S.Cboe(self.store) if sym in S.Cboe.FILES else S.Yahoo(self.store)
            action = P._action(src, "vix_daily" if isinstance(src, S.Cboe) else "daily", [sym], None, None, f"{sym} daily history " + ("is not stored" if have_last is None else f"ends {have_last}"), 0)
            try:
                self._download(action, "coverage")
            except (S.SourceError, Exception) as e:  # noqa: BLE001
                if have_last is None and (need["role"] == "primary" or need.get("required")):
                    raise LookupError(f"No historical price data is stored for {sym} and the free download failed ({str(e)[:160]}). Nothing is simulated in its place.") from None
                self.downloads.append({"source": action["source"], "label": action["label"], "dataset": action["dataset"], "symbols": [sym], "why": action["why"], "error": str(e)[:200]})

    # ---- the run
    def run(self) -> dict:
        spec = self.spec
        try:
            return self._run()
        except NeedsDecision as nd:
            blocked = next((s for s in self.stages if s["status"] == "running"), None)
            if blocked:
                blocked["status"], blocked["detail"] = "blocked", nd.reason[:240]
            dec = {"reason": nd.reason, "choices": nd.choices, **nd.data}
            self._write("needs_decision", dec)
            return {"outcome": "needs_decision", "decision": dec, "mode": nd.data.get("mode"), "research": True, "symbols": [spec.get("symbol")], "downloads": len(self.downloads)}
        except Exception as e:
            cur = next((s for s in self.stages if s["status"] == "running"), None)
            if cur:
                cur["status"], cur["detail"] = "error", f"{type(e).__name__}: {e}"[:240]
            self._write("error")
            raise

    def _run(self) -> dict:
        spec, store = self.spec, self.store
        kind = spec["kind"]
        mode = spec.get("mode") or "auto"
        if kind == "study":
            mode = "study"
        req_mode = "study" if mode == "study" else "modeled" if mode == "modeled" else "rules"  # what the RULES need; execution needs are checked per signal
        approvals = dict(spec.get("approvals") or {})
        if offline():
            approvals["no_download"] = True  # offline mode: work with what is stored, download nothing
        self.spec = spec = {**spec, "approvals": approvals}
        requested = spec.get("requested") or {}
        stated = bool(requested.get("stated"))
        interp = spec.get("interpretation") or {}
        self.stage("interpret", "done", interp.get("detail") or "configuration validated", **{k: v for k, v in interp.items() if k != "detail"})

        # ------------------------------------------------------------ coverage of the underlying
        self.stage("coverage", "running", "working out which prices this needs")
        req = _req(spec, req_mode)
        sym = req["symbol"]
        if not approvals.get("no_download"):
            self._ensure_daily(req)
        tab = P.session_table(store, sym)
        if tab.is_empty():
            raise LookupError(f"No historical price data is stored for {sym}. Nothing is simulated in its place.")
        first_stored, last_stored = tab["date"][0], tab["date"][-1]
        r_start = date.fromisoformat(requested["start"]) if requested.get("start") else first_stored
        r_end = date.fromisoformat(requested["end"]) if requested.get("end") else last_stored
        r_end = min(r_end, P.last_completed_session())
        if r_end < r_start:
            raise ValueError(f"The requested period ({r_start} to {r_end}) contains no completed trading session.")
        cov = P.underlying_coverage(store, req, r_start, r_end)
        actions, unfillable = P.underlying_actions(store, req, cov)
        pending_large = []
        for a in actions:
            self.cancelled()
            if a["dataset"] in ("daily", "vix_daily") and any(d.get("dataset") == a["dataset"] and d.get("symbols") == a["symbols"] and "error" not in d for d in self.downloads):
                continue
            if a["large"] and not approvals.get("large_download"):
                pending_large.append(a)
                continue
            if a["available"] and not approvals.get("no_download"):
                self.stage("coverage", "running", f"downloading {a['label']} {a['dataset']} for {', '.join(a['symbols'])}")
                try:
                    self._download(a, "coverage")
                except S.SourceError as e:
                    self.downloads.append({"source": a["source"], "label": a["label"], "dataset": a["dataset"], "symbols": a["symbols"], "why": a["why"], "error": str(e)[:200]})
        if self.downloads:
            cov = P.underlying_coverage(store, req, r_start, r_end)
            _, unfillable = P.underlying_actions(store, req, cov)
        usable = cov.pop("_usable_days")
        n_req, n_ok = cov["requested"]["sessions"], len(usable)
        short = n_req - n_ok
        self.stage("coverage", "running", f"{n_ok:,} of {n_req:,} requested sessions have the prices this needs", requested=cov["requested"], usable=cov["usable"])
        if not usable:
            why = "; ".join(f"{x['reason']} ({x['sessions']} sessions)" for x in cov["excluded"][:3]) or "no session has the required prices"
            raise NeedsDecision(f"None of the {n_req:,} requested sessions can be evaluated: {why}.", self._choices(spec, unfillable, pending_large, None, kind, mode), coverage=_pub(cov), unavailable=unfillable, mode=mode)
        if stated and short > max(2, SHORTFALL_TOLERANCE * n_req) and not approvals.get("reduced_window"):
            why = "; ".join(f"{x['reason']} ({x['sessions']:,} sessions)" for x in cov["excluded"][:3])
            raise NeedsDecision(f"You asked for {cov['requested']['start']} to {cov['requested']['end']} ({n_req:,} sessions), but only {n_ok:,} sessions "
                                f"({cov['usable']['first']} to {cov['usable']['last']}) have the prices this needs. {why}.",
                                self._choices(spec, unfillable, pending_large, cov["usable"], kind, mode), coverage=_pub(cov), unavailable=unfillable, mode=mode)
        self.stage("coverage", "done", f"{n_ok:,} of {n_req:,} requested sessions usable" + (f"; {len(self.downloads)} free download(s)" if self.downloads else "; nothing to download"),
                   downloads=len(self.downloads), unavailable=unfillable[:6], excluded=cov["excluded"][:8])
        record: dict = {"mode": mode, "requested": {**cov["requested"], "stated": stated}, "underlying": _pub(cov), "unavailable": unfillable, "awaiting_approval": pending_large}

        # ------------------------------------------------------------ market study: no option data involved
        if mode == "study":
            from strata.research.study import run_study
            self.stage("signals", "running", "evaluating the condition on every session")
            cfg = {**spec["study"], "start": r_start.isoformat(), "end": r_end.isoformat()}
            self.stage("acquire", "skipped", "a market study reads underlying prices only: no option data is needed")
            self.stage("contracts", "skipped", "no option contracts are involved")
            self.stage("simulate", "running", f"{sym}: {r_start} to {r_end}")
            res = run_study(cfg, store)
            st = res["stats"]
            self.stage("signals", "done", f"{st['qualifying_sessions']:,} qualifying sessions out of {st['evaluated_sessions']:,} evaluated; {st['excluded_sessions']:,} could not be evaluated",
                       qualifying=st["qualifying_sessions"], evaluated=st["evaluated_sessions"], excluded=st["excluded_sessions"])
            self.stage("simulate", "done", f"{st['qualifying_sessions']:,} observations")
            self.stage("report", "running", "assembling the report")
            record.update({"signals": {"qualifying": st["qualifying_sessions"], "evaluated_sessions": st["evaluated_sessions"], "not_evaluable_sessions": st["excluded_sessions"], "excluded": res.get("excluded", [])[:12]},
                           "executable": {"signals": st["qualifying_sessions"], "of": st["qualifying_sessions"], "window": res["period"]}, "options": None, "downloads": self.downloads,
                           "mode_label": MODE_LABEL["study"], "mode_reason": "A question about the underlying's behaviour: answered from price history, no option premiums involved."})
            res["coverage"] = record
            res["pipeline"] = self.stages
            summary = save_study(self.rid, res)
            self.stage("report", "done", "ready")
            self._write("completed")
            return {**summary, "outcome": "completed", "mode": "study", "research": True, "downloads": len(self.downloads)}

        # ------------------------------------------------------------ stage one: qualifying sessions
        from strata.engine.scan import scan_signals
        self.stage("signals", "running", f"evaluating the entry rules on {sym} prices, {r_start} to {r_end}")
        scan = scan_signals(_bt_cfg(spec, r_start, r_end, "historical"), store, should_cancel=self.should_cancel)
        signals = scan["signals"]
        n_ex = sum(len(v) for v in scan["excluded"].values())
        n_ex = scan["sessions"] - scan["sessions_evaluable"]  # sessions, not reasons: one session can be excluded for two reasons
        n_ev = scan["sessions_evaluable"]
        self.stage("signals", "done", f"{len(signals):,} qualifying sessions in {n_ev:,} evaluated" + (f"; {n_ex:,} could not be evaluated (kept apart, not counted as 'no signal')" if n_ex else ""),
                   qualifying=len(signals), evaluated=n_ev, excluded=n_ex, last=[s["date"].isoformat() for s in signals[-8:]])
        record["signals"] = {"qualifying": len(signals), "evaluated_sessions": n_ev, "not_evaluable_sessions": n_ex,
                             "excluded": [{"reason": k, "count": len(v), "dates": v[-12:]} for k, v in sorted(scan["excluded"].items(), key=lambda kv: -len(kv[1]))][:12],
                             "dates": [s["date"].isoformat() for s in signals][-4000:]}
        if not signals:
            raise NeedsDecision(f"The entry rules never fired between {r_start} and {r_end} ({n_ev:,} sessions evaluated" + (f", {n_ex:,} not evaluable" if n_ex else "") + "), so there is nothing to backtest.",
                                [{"id": "edit", "label": "Loosen the entry rule", "detail": "Edit a condition in the interpretation panel and run again."}], coverage=record, mode=mode)

        # ------------------------------------------------------------ which kind of option observation can answer this
        opt_cov, acq, mode, reason = self._options(self.spec, req, signals, mode, approvals, record, unfillable, pending_large)
        record.update({"mode": mode, "mode_label": MODE_LABEL[mode], "mode_reason": reason, "options": opt_cov, "acquisition": acq, "downloads": self.downloads})
        k_ok, k = opt_cov["executable"], len(signals)
        if mode != "modeled":
            if k_ok == 0:
                why = "; ".join(f"{x['reason']} ({x['signals']} signal{'s' if x['signals'] != 1 else ''})" for x in opt_cov["issues"][:3]) or "no observed option prices for any qualifying session"
                raise NeedsDecision(f"None of the {k:,} qualifying sessions has usable {opt_cov['observation']}: {why}.",
                                    self._choices(spec, record["unavailable"], pending_large, None, kind, mode, tried=mode), coverage=record, mode=mode)
            if stated and k_ok < k and not approvals.get("reduced_window"):
                why = "; ".join(f"{x['reason']} ({x['signals']})" for x in opt_cov["issues"][:3])
                raise NeedsDecision(f"Only {k_ok:,} of the {k:,} qualifying sessions in the period you asked for have usable {opt_cov['observation']}. {why}.",
                                    self._choices(spec, record["unavailable"], pending_large, {"sessions": k_ok, "what": "qualifying sessions"}, kind, mode, tried=mode), coverage=record, mode=mode)

        # ------------------------------------------------------------ simulate
        a, b = self._window(req, mode, r_start, r_end, signals)
        record["executable"] = {"signals": k_ok if mode != "modeled" else k, "of": k, "window": [a.isoformat(), b.isoformat()]}
        self.stage("simulate", "running", f"{MODE_LABEL[mode]}: {a} to {b}")
        from strata.engine.engine import run_backtest
        cfg = BacktestConfig.model_validate(_bt_cfg(spec, a, b, mode))
        prog = (lambda i, n, d: self.sub("simulate")(i / max(n, 1), f"session {d} ({i}/{n})"))
        res = run_backtest(cfg, store, progress=prog, should_cancel=self.should_cancel)
        adaptive = {"rounds": 0, "widened_sessions": [], "resolved": 0, "remaining": 0, "stopped": None}
        if mode == "trades":
            # adaptive contract discovery: if a filled selection sits on the outermost stored strike, the contract the
            # rule wanted may be outside the downloaded band - widen it for those sessions and simulate again
            src_name = (acq or {}).get("ran", {}).get("source") if isinstance((acq or {}).get("ran"), dict) else None
            src_name = src_name or (opt_cov.get("dataset") or "").split("/")[0] or None
            widen: dict = {}
            first_edge = None
            while adaptive["rounds"] < MAX_EXPANSIONS:
                cs = res.meta.get("contract_selection") or {}
                edge = [date.fromisoformat(x) for x in cs.get("edge_dates") or []]
                if first_edge is None:
                    first_edge = len(edge)
                if not edge:
                    break
                if approvals.get("no_download") or not src_name:
                    adaptive["stopped"] = "downloads are off, so the strike range could not be widened"
                    break
                for d_ in edge:
                    widen[d_] = widen.get(d_, 0) + 1
                self.stage("simulate", "running", f"{len(edge)} selection(s) sit on the edge of the downloaded strikes: widening the range and simulating again")
                try:
                    r2 = P.acquire_trades(store, req, [s for s in signals if s["date"] in set(edge)], src_name, None, self.should_cancel, True, widen)
                except Exception as e:  # noqa: BLE001
                    adaptive["stopped"] = f"the wider range could not be downloaded: {str(e)[:160]}"
                    break
                adaptive["rounds"] += 1
                adaptive["widened_sessions"] = sorted({x.isoformat() for x in widen})
                got = (r2.get("result") or {})
                self.downloads.append({"source": src_name, "label": src_name, "dataset": "option_trades", "symbols": [req["symbol"]], "tier": "free", "cost": 0.0,
                                       "why": f"adaptive strike range: {len(edge)} session(s) widened (round {adaptive['rounds']})", "result": {k: got.get(k) for k in ("pairs_requested", "pairs_cached", "pairs_ok", "pairs_empty", "bars", "requests")}})
                if not (got.get("pairs_ok") or got.get("pairs_empty")):
                    adaptive["stopped"] = "no further strikes are available from the source"
                    break
                res = run_backtest(cfg, store, progress=prog, should_cancel=self.should_cancel)
            left = len((res.meta.get("contract_selection") or {}).get("edge_dates") or [])
            adaptive.update(remaining=left, resolved=max(0, (first_edge or 0) - left), first_edge=first_edge or 0)
            if left and not adaptive["stopped"]:
                adaptive["stopped"] = f"still on the edge after {MAX_EXPANSIONS} widenings"
        record["contract_selection"] = {**(res.meta.get("contract_selection") or {}), "adaptive": adaptive}
        self.stage("simulate", "done", f"{len(res.trades):,} trades from {res.diagnostics['signals']:,} signals in the simulated window"
                   + (f"; strike range widened for {len(adaptive['widened_sessions'])} session(s)" if adaptive["rounds"] else ""))
        self.stage("report", "running", "computing analytics")
        record["simulated"] = {"window": [a.isoformat(), b.isoformat()], "signals": res.diagnostics["signals"], "trades": len(res.trades), "executability": res.meta.get("executability")}
        record["skipped_signals"] = _skipped(signals, res, a, b, opt_cov, mode)
        summary = save_backtest(self.rid, res, extra={"coverage": record, "pipeline": self.stages, "cost_sensitivity": cost_sensitivity(res), "contract_selection": record["contract_selection"]})
        self.stage("report", "done", "ready")
        self._write("completed")
        return {**summary, "outcome": "completed", "mode": mode, "research": True, "downloads": len(self.downloads), "qualifying": k, "executable": record["executable"]["signals"]}

    # ---- option data: choose the observation type, acquire selectively, validate per signal
    def _options(self, spec, req0, signals, mode, approvals, record, unfillable, pending_large):
        store = self.store
        provider = spec.get("provider")
        if mode == "modeled":
            self.stage("acquire", "skipped", "modeled premiums use underlying prices only")
            self.stage("contracts", "skipped", "theoretical prices: no observed contracts")
            return ({"observation": "theoretical premiums", "signals": len(signals), "executable": len(signals), "issues": [], "dataset": None}, None, "modeled",
                    "You opted in to theoretical premiums. Nothing here is an observed option price.")
        tried: dict[str, dict] = {}

        def quotes():
            req = _req(spec, "historical")
            self.stage("acquire", "running", "checking stored option quotes for the qualifying sessions")
            cov = P.quote_coverage(store, req, signals, provider)
            acts, unf = P.quote_actions(store, req, cov)
            acq = {"actions": acts, "ran": []}
            for a in acts:
                self.cancelled()
                if a["large"] and not approvals.get("large_download"):
                    pending_large.append(a)
                    continue
                if not a["available"] or approvals.get("no_download"):
                    continue
                self.stage("acquire", "running", f"downloading {a['label']}: {a['why']}")
                try:
                    acq["ran"].append(self._download(a, "acquire"))
                except S.SourceError as e:
                    self.downloads.append({"source": a["source"], "label": a["label"], "dataset": a["dataset"], "symbols": a["symbols"], "why": a["why"], "error": str(e)[:200]})
            if acq["ran"]:
                cov = P.quote_coverage(store, req, signals, provider)
            cov.pop("_missing", None)
            return cov, acq, unf

        def trades():
            req = _req(spec, "trades")
            plan = P.trade_plan(store, req, signals, provider)
            acq = {"plan": plan, "ran": None}
            src = plan.get("chosen")
            info = next((x for x in plan["sources"] if x["source"] == src), None)
            # trade-bar execution needs minute-level underlying bars on the signal sessions
            need_days = sorted({s["date"] for s in signals})
            tab = {r["date"]: r["res"] for r in P.session_table(store, req["symbol"], need_days[0], need_days[-1]).iter_rows(named=True)}
            coarse = [d for d in need_days if tab.get(d, 10**9) > 300]
            if coarse and not approvals.get("no_download"):
                ucov = P.underlying_coverage(store, req, need_days[0], need_days[-1], only=need_days)
                acts, _ = P.underlying_actions(store, req, ucov)
                for a in acts:
                    if a["available"] and (not a["large"] or approvals.get("large_download")) and a["dataset"] in ("intraday", "bars_1m"):
                        self.stage("acquire", "running", f"downloading {a['label']} intraday bars for the qualifying sessions")
                        try:
                            self._download(a, "acquire")
                        except S.SourceError as e:
                            self.downloads.append({"source": a["source"], "label": a["label"], "dataset": a["dataset"], "symbols": a["symbols"], "why": a["why"], "error": str(e)[:200]})
                    elif a["large"]:
                        pending_large.append(a)
            if info and info["available"] and info["signals_in_window"] and not approvals.get("no_download"):
                self.stage("acquire", "running", f"{info['label']}: option trade prints for {info['signals_in_window']} qualifying session(s) only")
                try:
                    r = P.acquire_trades(store, req, signals, src, self.sub("acquire"), self.should_cancel, bool(approvals.get("large_download")))
                except S.SourceError as e:
                    r = {"source": src, "error": str(e)[:240]}
                acq["ran"] = r
                if r.get("needs_approval"):
                    pending_large.append({"id": f"{src}:option_trades", "source": src, "label": info["label"], "dataset": "option_trades", "symbols": [req["symbol"]], "tier": info["tier"], "cost": 0.0,
                                          "est_bytes": r["estimate"]["est_bytes"], "est_seconds": r["estimate"]["est_seconds"], "requests": r["estimate"]["requests"], "large": True,
                                          "why": f"option trade prints for {r['signals_in_window']} qualifying sessions ({r['contract_sessions']:,} contract-sessions)"})
                elif r.get("result") and r["result"].get("requests"):
                    x = r["result"]
                    self.downloads.append({"source": src, "label": info["label"], "dataset": "option_trades", "symbols": [req["symbol"]], "tier": info["tier"], "cost": 0.0,
                                           "why": f"option trade prints for {r['signals_in_window']} qualifying session(s): {r['contract_sessions']:,} contract-sessions requested",
                                           "result": {k: x.get(k) for k in ("pairs_requested", "pairs_cached", "pairs_ok", "pairs_empty", "pairs_unavailable", "pairs_failed", "bars", "requests", "rejected")}})
            cov = P.trade_coverage(store, req, signals, provider)
            # sessions whose underlying bars are too coarse cannot be executed on trade bars either
            tab = {r["date"]: r["res"] for r in P.session_table(store, req["symbol"], need_days[0], need_days[-1]).iter_rows(named=True)}
            still = {d for d in need_days if tab.get(d, 10**9) > 300}
            unf = []
            for x in plan["sources"]:
                if not x["available"] and x["source"] == "alpaca":
                    n = sum(1 for s in signals if s["date"] >= S.ALPACA_OPTIONS_FIRST)
                    if n:
                        unf.append({"symbol": req["symbol"], "sessions": n, "fix": "connect_alpaca", "reason": "option trade history needs an Alpaca key",
                                    "detail": f"A free Alpaca account provides 1-minute option trade bars since {S.ALPACA_OPTIONS_FIRST}: {n} of the {len(signals)} qualifying sessions fall in that range."})
            old = sum(1 for s in signals if s["date"] < S.ALPACA_OPTIONS_FIRST)
            if old:
                unf.append({"symbol": req["symbol"], "sessions": old, "fix": "paid_or_import", "reason": f"no free option trade history before {S.ALPACA_OPTIONS_FIRST}",
                            "detail": f"{old} qualifying sessions are before {S.ALPACA_OPTIONS_FIRST}, where no free provider has intraday option prices. A paid provider (never used automatically) or imported files would be needed."})
            if still and cov["executable"]:
                cov["issues"].append({"reason": "the underlying has no 5-minute or finer bars on the session, which trade-bar execution needs", "signals": len(still), "kind": "underlying",
                                      "sample": [d.isoformat() for d in sorted(still)[-6:]]})
            return cov, acq, unf

        order = [mode] if mode in ("historical", "trades") else ["historical", "trades"]
        best = None
        for m in order:
            cov, acq, unf = quotes() if m == "historical" else trades()
            tried[m] = cov
            for u in unf:
                if u not in unfillable:
                    unfillable.append(u)
            if best is None or cov["executable"] > best[0]["executable"]:
                best = (cov, acq, m)
            if cov["executable"] > 0:
                break
        cov, acq, chosen = best
        record["unavailable"] = unfillable
        record["awaiting_approval"] = pending_large
        record["modes_tried"] = {m: {"executable": c["executable"], "signals": c["signals"], "issues": c["issues"][:4], "observation": c["observation"]} for m, c in tried.items()}
        n_dl = len([d for d in self.downloads if d.get("dataset") in ("option_trades", "options_eod") and "error" not in d])
        ran = (acq or {}).get("ran")
        cached = (ran.get("result") or {}).get("pairs_cached") if isinstance(ran, dict) else None
        self.stage("acquire", "done" if n_dl else "skipped",
                   (f"option data fetched for the qualifying sessions only ({n_dl} download{'s' if n_dl != 1 else ''})" if n_dl else
                    "nothing downloaded: " + ((f"the {cached:,} contract-sessions this needs are already stored" if cached else "the required option data is already stored") if cov["executable"]
                                              else "no free source can supply the missing option data")),
                   awaiting_approval=[{k: a.get(k) for k in ("label", "why", "est_bytes", "est_seconds", "requests")} for a in pending_large][:4])
        self.stage("contracts", "running", "checking each qualifying session against the stored contracts")
        self.stage("contracts", "done", f"{cov['executable']:,} of {cov['signals']:,} qualifying sessions have usable {cov['observation']}", issues=cov["issues"][:6], dataset=cov.get("dataset"))
        if mode == "auto":
            other = [m for m in tried if m != chosen]
            reason = (f"Chosen automatically: {cov['observation']} cover {cov['executable']} of {cov['signals']} qualifying sessions" +
                      (f"; {tried[other[0]]['observation']} cover {tried[other[0]]['executable']}" if other else "") + ". Theoretical premiums are never chosen automatically.")
        else:
            reason = f"You asked for {MODE_LABEL[chosen].lower()}."
        return cov, acq, chosen, reason

    def _window(self, req, mode, r_start, r_end, signals) -> tuple[date, date]:
        """Sessions to simulate: the requested window, trimmed to where option data exists so
        the engine does not walk years that cannot produce a trade (they are reported, not run)."""
        if mode == "modeled":
            return r_start, r_end
        sym, hold = req["symbol"], int((req["options"] or {}).get("hold_sessions") or 0)
        if mode == "trades":
            ds = self.store.option_dataset(sym, "trade", self.spec.get("provider"))
            idx = self.store.option_bar_index(sym, dataset=ds) if ds else None
            days = sorted(idx["date"].unique().to_list()) if idx is not None and not idx.is_empty() else self.store.option_days(sym, "trade", dataset=ds)
        else:
            ds = self.store.option_dataset(sym, "quote", self.spec.get("provider"))
            days = self.store.option_days(sym, "quote", dataset=ds)
        sig_days = sorted({s["date"] for s in signals})
        have = [d for d in sig_days if d in set(days)]
        if not have:
            return r_start, r_end
        a = max(r_start, have[0])
        b = min(r_end, max(cal.next_trading_day(have[-1], hold) if hold else have[-1], have[-1]))
        return a, max(a, b)

    def _choices(self, spec, unfillable, pending_large, reduced, kind, mode, tried: str | None = None) -> list[dict]:
        out = []
        for a in pending_large[:3]:
            out.append({"id": "approve_download", "label": f"Download {_size(a.get('est_bytes', 0))} from {a['label']}", "kind": "expand",
                        "detail": f"{a['why']}. About {_size(a.get('est_bytes', 0))}, {max(1, a.get('est_seconds', 0) // 60)} min, {a.get('requests', 0):,} requests. Free ($0.00)."})
        if any(u.get("fix") == "connect_alpaca" for u in unfillable):
            u = next(u for u in unfillable if u.get("fix") == "connect_alpaca")
            out.append({"id": "connect_alpaca", "label": "Connect a free Alpaca account", "kind": "expand", "detail": u["detail"]})
        if reduced and reduced.get("sessions"):
            what = reduced.get("what") or "sessions"
            span = f" ({reduced['first']} to {reduced['last']})" if reduced.get("first") else ""
            out.append({"id": "reduced_window", "label": f"Run on the {reduced['sessions']:,} {what} that are covered", "kind": "reduce", "detail": f"Accept the reduced coverage{span}. The report states what was left out."})
        if kind == "options":
            if tried == "historical":
                out.append({"id": "mode:trades", "label": "Use option trade prices instead", "kind": "mode", "detail": "Execute on observed option trade prints (no bid/ask). Free sources: Yahoo (last ~30 days), Alpaca (since 2024-01-18, free key)."})
            out.append({"id": "mode:study", "label": "Run the market study on the underlying", "kind": "mode", "detail": "Measures how the underlying behaved on the qualifying sessions. It is not the option position's profit or loss."})
            out.append({"id": "mode:modeled", "label": "Estimate with modeled premiums (opt-in)", "kind": "mode", "detail": "Real underlying prices with THEORETICAL option premiums. Estimates, never shown as observed prices."})
        if any(u.get("fix") == "paid_or_import" for u in unfillable):
            out.append({"id": "info:paid", "label": "Paid data or imported files", "kind": "info",
                        "detail": "Intraday option history further back needs a paid provider (ThetaData, Databento, Polygon) or files you import under Data. Strata never calls a paid API on its own."})
        return out


def _size(n: float) -> str:
    n = float(n or 0)
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{max(n / 1e3, 1):.0f} KB"


def _brief(out) -> dict:
    """A small, serialisable view of an ingest report."""
    if not isinstance(out, dict):
        return {}
    syms = out.get("symbols") or {}
    return {"symbols": {k: ({kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str)) or kk in ("years",)} if isinstance(v, dict) else v) for k, v in list(syms.items())[:6]}}


def _pub(cov: dict) -> dict:
    return {k: v for k, v in cov.items() if not k.startswith("_")}


def _skipped(signals: list[dict], res, a: date, b: date, opt_cov: dict, mode: str) -> dict:
    """Every qualifying session that did not become a trade, with the reason.

    Signals inside the simulated window come from the engine's own record; signals outside
    it were not simulated because no option data exists there, which is a data limitation
    and is listed as such - never as a rule that failed."""
    rows = []
    traded = {t["entry_date"] for t in res.trades}
    eng: dict[str, dict] = {}
    for s in res.diagnostics.get("skipped_signals", []):
        d = cal.to_et(s["signal_t"] * 1_000_000_000).date().isoformat()
        eng.setdefault(d, s)
    reason_out = (opt_cov["issues"][0]["reason"] if opt_cov.get("issues") else "no observed option prices are stored or freely available for this session") if mode != "modeled" else ""
    for s in signals:
        d = s["date"].isoformat()
        if d in traded:
            continue
        if a <= s["date"] <= b:
            e = eng.get(d)
            rows.append({"date": d, "minute": s["minute"], "price": s["price"], "kind": e["kind"] if e else "not_taken",
                         "reason": e["reason"] if e else "a position was already open or the entry limit for the session was reached", "class": "execution" if e else "position_limit"})
        else:
            rows.append({"date": d, "minute": s["minute"], "price": s["price"], "kind": "no_option_data", "reason": reason_out, "class": "data"})
    by: dict[str, int] = {}
    for r in rows:
        by[r["kind"]] = by.get(r["kind"], 0) + 1
    return {"count": len(rows), "by_kind": by, "rows": rows[-3000:], "truncated": len(rows) > 3000}


def cost_sensitivity(res, levels=(0, 1, 2, 5, 10)) -> dict:
    """First-order sensitivity of the result to worse fills: every option fill is repriced
    x% against the trader with the same trades (entries and exits are not re-decided)."""
    rows = []
    for x in levels:
        pnls = []
        for t in res.trades:
            drag = sum(abs(l["contracts"] or 0) * (l["multiplier"] or 100) * ((l["entry_price"] or 0) + (l["exit_price"] or 0)) * x / 100 for l in t["legs"] if l["kind"] == "option")
            pnls.append(t["pnl"] - drag)
        n = len(pnls)
        rows.append({"extra_slippage_pct": x, "net_pnl": float(sum(pnls)) if n else 0.0, "win_rate": (sum(1 for p in pnls if p > 0) / n) if n else None, "avg_trade": (sum(pnls) / n) if n else None})
    base = rows[0]["net_pnl"]
    breakeven = None
    if res.trades and base > 0:
        notional = sum(abs(l["contracts"] or 0) * (l["multiplier"] or 100) * ((l["entry_price"] or 0) + (l["exit_price"] or 0)) for t in res.trades for l in t["legs"] if l["kind"] == "option")
        breakeven = base / notional * 100 if notional > 0 else None
    return {"rows": rows, "breakeven_slippage_pct": breakeven, "method": "Each option fill repriced x% against the position with trades unchanged (a first-order estimate; exits are not re-decided)."}


def run_pipeline(rid: str, spec: dict, progress=None, should_cancel=None, store: MarketStore | None = None) -> dict:
    out = Pipeline(rid, spec, progress, should_cancel, store).run()
    if out.get("outcome") == "completed":
        try:
            from strata.api import db
            db.update_run(rid, cache_key=fingerprint(spec, store), dataset_versions={s: (store or MarketStore()).dataset_version(s) for s in sorted(_symbols(spec))})
        except Exception:  # noqa: BLE001  (a cache key is an optimisation, never a reason to fail a finished run)
            pass
    return out


def load_pipeline(rid: str) -> dict | None:
    p = run_dir(rid) / "pipeline.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        return None
