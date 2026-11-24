"""Strategy optimisation lab: grid / random / Bayesian (GP-EI) search, run in parallel
processes, with hold-out and walk-forward validation plus overfitting diagnostics."""
from __future__ import annotations

import copy
import itertools
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import date, timedelta
from multiprocessing import get_context
from typing import Any, Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from strata.core import calendar as cal
from strata.engine.models import BacktestConfig
from strata.optimize.overfitting import deflated_sharpe, pareto_front, pbo_cscv, stability_scores

OBJECTIVES = ["sharpe", "sortino", "calmar", "cagr", "net_pnl", "profit_factor", "expectancy", "avg_return_on_risk"]


class ParamSpec(BaseModel):
    path: str
    label: str = ""
    kind: Literal["float", "int", "choice", "time"] = "float"
    values: Optional[list[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    def grid(self) -> list:
        if self.values:
            return list(self.values)
        if self.kind == "time":
            a, b, s = int(self.min), int(self.max), int(self.step or 15)
            return [cal.fmt_hhmm(m) for m in range(a, b + 1, s)]
        if self.min is None or self.max is None:
            raise ValueError(f"param {self.path}: provide values or min/max")
        step = self.step or (self.max - self.min) / 10 or 1
        n = int(math.floor((self.max - self.min) / step + 1e-9)) + 1
        vals = [self.min + i * step for i in range(n)]
        return [int(round(v)) for v in vals] if self.kind == "int" else [round(v, 10) for v in vals]


class OptimizationConfig(BaseModel):
    base: BacktestConfig
    params: list[ParamSpec]
    method: Literal["grid", "random", "bayes"] = "grid"
    objective: str = "sharpe"
    max_evals: int = Field(200, ge=1, le=5000)
    seed: int = 42
    min_trades: int = 30
    validation: Literal["none", "holdout", "walk_forward"] = "holdout"
    holdout_pct: float = Field(30.0, gt=0, lt=90)
    wf_folds: int = Field(4, ge=2, le=12)
    wf_anchored: bool = False
    workers: Optional[int] = None
    prior_trials: int = 0  # configurations already tested on this strategy (multiple-testing)


# ------------------------------------------------------------------ config paths
def set_path(obj: dict, path: str, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur.setdefault(p, {})
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def get_path(obj: dict, path: str):
    cur = obj
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def apply_params(base: dict, params: dict) -> dict:
    cfg = copy.deepcopy(base)
    for path, v in params.items():
        set_path(cfg, path, v)
    return cfg


def optimizable_params(strategy: dict, prefix: str = "strategies.0") -> list[dict]:
    """Suggest parameters the user can sweep, with their current values."""
    out = []
    for i, leg in enumerate(strategy.get("legs", [])):
        if leg.get("kind", "option") != "option":
            continue
        m = leg.get("strike", {}).get("method", "delta")
        lid = leg.get("id", i)
        out.append({"path": f"{prefix}.legs.{i}.dte", "label": f"{lid} DTE", "kind": "int", "current": leg.get("dte", 0)})
        lab = {"delta": "delta", "premium": "premium $", "pct_otm": "% OTM", "atm_offset": "ATM offset", "fixed": "strike", "leg_offset": "width"}[m]
        out.append({"path": f"{prefix}.legs.{i}.strike.value", "label": f"{lid} {lab}", "kind": "float" if m != "fixed" else "float", "current": leg.get("strike", {}).get("value")})
    ex = strategy.get("exits", {})
    for k, lab in [("profit_target", "Profit target %"), ("stop_loss", "Stop loss %")]:
        if ex.get(k, {}).get("enabled", k in ("profit_target", "stop_loss")):
            out.append({"path": f"{prefix}.exits.{k}.value", "label": lab, "kind": "float", "current": ex.get(k, {}).get("value", 50 if k == "profit_target" else 200)})
    if ex.get("time_exit", {}).get("enabled"):
        out.append({"path": f"{prefix}.exits.time_exit.time", "label": "Time exit", "kind": "time", "current": ex["time_exit"].get("time")})

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if node.get("kind") == "cond":
            t, p = node.get("type"), node.get("params", {})
            if t == "entry_times":
                out.append({"path": f"{path}.params.times", "label": "Entry time", "kind": "time", "current": p.get("times")})
            elif t in ("vix", "iv", "iv_rank", "iv_percentile", "gap", "price_move", "vix_change", "day_range"):
                out.append({"path": f"{path}.params.value", "label": f"{t} threshold", "kind": "float", "current": p.get("value")})
            elif t == "compare":
                lhs, rhs = p.get("lhs", {}), p.get("rhs", {})
                if lhs.get("kind") == "indicator":
                    for k, v in (lhs.get("params") or {}).items():
                        out.append({"path": f"{path}.params.lhs.params.{k}", "label": f"{lhs.get('name')} {k}", "kind": "int" if isinstance(v, int) else "float", "current": v})
                if rhs.get("kind") == "const":
                    out.append({"path": f"{path}.params.rhs.value", "label": f"{lhs.get('name', 'lhs')} threshold", "kind": "float", "current": rhs.get("value")})
            elif t == "opening_range":
                out.append({"path": f"{path}.params.minutes", "label": "ORB minutes", "kind": "int", "current": p.get("minutes")})
        for j, c in enumerate(node.get("children", []) or []):
            walk(c, f"{path}.children.{j}")

    walk(strategy.get("entry", {}).get("conditions", {}), f"{prefix}.entry.conditions")
    return out


# ------------------------------------------------------------------ evaluation
def _worker_init(data_dir: str):
    from strata.config import override_settings
    override_settings(data_dir=data_dir)


def evaluate_config(cfg_dict: dict) -> dict:
    from strata.analytics.metrics import compute_metrics
    from strata.engine.engine import run_backtest
    t0 = time.time()
    try:
        cfg = BacktestConfig.model_validate(cfg_dict)
        res = run_backtest(cfg, record_paths=False)
        m = compute_metrics(res.trades, res.daily, cfg.capital.initial_capital, cfg.capital.risk_free_rate)
        eq = res.daily["equity"].to_numpy() if not res.daily.is_empty() else np.array([])
        rets = (eq / np.r_[cfg.capital.initial_capital, eq[:-1]] - 1).tolist() if eq.size else []
        keep = ["net_pnl", "total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor", "expectancy",
                "trades", "avg_return_on_risk", "max_capital_utilization", "avg_capital_utilization", "cvar_95_daily", "t_stat", "exposure"]
        return {"ok": True, "metrics": {k: m.get(k) for k in keep}, "dates": [d.isoformat() for d in res.daily["date"].to_list()] if eq.size else [],
                "returns": rets, "runtime": time.time() - t0, "synthetic": res.meta["synthetic"]}
    except Exception as e:  # a bad combination must not kill the whole study
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def objective_value(r: dict, objective: str, min_trades: int) -> float:
    if not r.get("ok"):
        return -math.inf
    v = r["metrics"].get(objective)
    if v is None or not math.isfinite(v):
        return -math.inf
    return float(v)


class Evaluator:
    """Runs batches of configs in a process pool with progress + cancellation."""

    def __init__(self, workers: int | None, data_dir: str, progress: Callable | None, should_cancel: Callable | None):
        from strata.config import get_settings
        self.workers = max(1, min(workers or get_settings().max_workers, os.cpu_count() or 1))
        self.data_dir = data_dir
        self.progress = progress
        self.should_cancel = should_cancel
        self.pool = None
        self.done = 0
        self.total = 0

    def __enter__(self):
        if self.workers > 1:
            # one compute thread per worker process: N processes x (polars + numba pools)
            # would otherwise oversubscribe the CPU many times over
            for k in ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
                os.environ[k] = "1"
            self.pool = ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"), initializer=_worker_init, initargs=(self.data_dir,))
        return self

    def __exit__(self, *a):
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def run(self, cfgs: list[dict], label: str = "") -> list[dict]:
        from strata.engine.engine import BacktestCancelled
        results: list[dict | None] = [None] * len(cfgs)
        if self.pool is None:
            for i, c in enumerate(cfgs):
                if self.should_cancel and self.should_cancel():
                    raise BacktestCancelled()
                results[i] = evaluate_config(c)
                self.done += 1
                if self.progress:
                    self.progress(self.done, self.total, label)
            return results  # type: ignore
        futs = {}
        it = iter(enumerate(cfgs))
        for _ in range(self.workers * 2):
            nxt = next(it, None)
            if nxt is None:
                break
            futs[self.pool.submit(evaluate_config, nxt[1])] = nxt[0]
        while futs:
            if self.should_cancel and self.should_cancel():
                for f in futs:
                    f.cancel()
                raise BacktestCancelled()
            done, _ = wait(list(futs), timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                i = futs.pop(f)
                results[i] = f.result()
                self.done += 1
                if self.progress:
                    self.progress(self.done, self.total, label)
                nxt = next(it, None)
                if nxt is not None:
                    futs[self.pool.submit(evaluate_config, nxt[1])] = nxt[0]
        return results  # type: ignore


# ------------------------------------------------------------------ search
def _encode(space: list[ParamSpec], params: dict) -> np.ndarray:
    out = []
    for p in space:
        g = p.grid()
        out.append(g.index(params[p.path]) / max(len(g) - 1, 1))
    return np.array(out)


def _gp_ei(X: np.ndarray, y: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Gaussian-process expected improvement (Matern 5/2, length scale by marginal likelihood)."""
    yv = y.copy()
    finite = np.isfinite(yv)
    if finite.sum() < 2:
        return np.random.default_rng(0).random(len(cand))
    lo = np.min(yv[finite])
    yv[~finite] = lo - abs(lo) * 0.5 - 1
    mu, sd = yv.mean(), yv.std() or 1.0
    yn = (yv - mu) / sd

    def k(A, B, ls):
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)) / ls
        return (1 + math.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-math.sqrt(5) * d)

    best_ll, best = -np.inf, None
    for ls in (0.1, 0.2, 0.35, 0.6, 1.0):
        K = k(X, X, ls) + 1e-4 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        ll = -0.5 * yn @ a - np.log(np.diag(L)).sum()
        if ll > best_ll:
            best_ll, best = ll, (ls, L, a)
    if best is None:
        return np.random.default_rng(0).random(len(cand))
    ls, L, a = best
    Ks = k(cand, X, ls)
    m = Ks @ a
    v = np.linalg.solve(L, Ks.T)
    var = np.maximum(1 - (v ** 2).sum(0), 1e-12)
    s = np.sqrt(var)
    from scipy.stats import norm
    imp = m - yn.max() - 0.01
    z = imp / s
    return imp * norm.cdf(z) + s * norm.pdf(z)


def search(space: list[ParamSpec], method: str, max_evals: int, seed: int, evaluate: Callable[[list[dict]], list[float]], batch: int) -> tuple[list[dict], list[float]]:
    grids = [p.grid() for p in space]
    total = int(np.prod([len(g) for g in grids]))
    rng = np.random.default_rng(seed)
    if method == "grid" or total <= max_evals and method != "bayes":
        combos = list(itertools.product(*grids))
        if len(combos) > max_evals:
            sel = rng.choice(len(combos), size=max_evals, replace=False)
            combos = [combos[i] for i in sorted(sel)]
        params = [{p.path: v for p, v in zip(space, c)} for c in combos]
        return params, evaluate(params)
    if method == "random":
        seen, params = set(), []
        while len(params) < min(max_evals, total):
            c = tuple(g[rng.integers(len(g))] for g in grids)
            if c not in seen:
                seen.add(c)
                params.append({p.path: v for p, v in zip(space, c)})
        return params, evaluate(params)
    # bayesian: random initial design then GP-EI batches
    n_init = min(max(batch, 2 * len(space) + 2), max_evals, total)
    seen, params = set(), []
    while len(params) < n_init:
        c = tuple(g[rng.integers(len(g))] for g in grids)
        if c not in seen:
            seen.add(c)
            params.append({p.path: v for p, v in zip(space, c)})
    scores = list(evaluate(params))
    while len(params) < min(max_evals, total):
        X = np.array([_encode(space, pr) for pr in params])
        cands = []
        for _ in range(min(3000, total * 2)):
            c = tuple(g[rng.integers(len(g))] for g in grids)
            if c not in seen:
                cands.append(c)
        cands = list(dict.fromkeys(cands))
        if not cands:
            break
        C = np.array([[g.index(v) / max(len(g) - 1, 1) for g, v in zip(grids, c)] for c in cands])
        ei = _gp_ei(X, np.array(scores, float), C)
        order = np.argsort(-ei)
        pick = []
        for i in order:
            if len(pick) >= min(batch, max_evals - len(params)):
                break
            if all(np.abs(C[i] - C[j]).sum() > 1e-9 for j in pick):
                pick.append(i)
        new = [{p.path: v for p, v in zip(space, cands[i])} for i in pick]
        for i in pick:
            seen.add(cands[i])
        params += new
        scores += list(evaluate(new))
    return params, scores


# ------------------------------------------------------------------ orchestration
def _split_dates(start: date, end: date, frac_is: float) -> tuple[date, date]:
    days = cal.trading_days(start, end)
    k = max(1, int(len(days) * frac_is))
    return days[k - 1], days[min(k, len(days) - 1)]


def _with_dates(cfg: dict, s: date, e: date) -> dict:
    c = copy.deepcopy(cfg)
    c["start"], c["end"] = s.isoformat(), e.isoformat()
    return c


def run_optimization(oc: OptimizationConfig, data_dir: str, progress=None, should_cancel=None) -> dict:
    base = oc.base.model_dump(mode="json")
    space = oc.params
    if not space:
        raise ValueError("select at least one parameter to optimise")
    t0 = time.time()
    is_start, is_end = oc.base.start, oc.base.end
    oos_start = None
    if oc.validation == "holdout":
        is_end, oos_start = _split_dates(oc.base.start, oc.base.end, 1 - oc.holdout_pct / 100)
    all_rows: list[dict] = []
    with Evaluator(oc.workers, data_dir, progress, should_cancel) as ev:
        grid_total = int(np.prod([len(p.grid()) for p in space]))
        n_search = min(oc.max_evals, grid_total)
        folds = []
        if oc.validation == "walk_forward":
            days = cal.trading_days(oc.base.start, oc.base.end)
            chunks = np.array_split(np.arange(len(days)), oc.wf_folds + 1)
            for i in range(1, oc.wf_folds + 1):
                tr = np.concatenate(chunks[:i]) if oc.wf_anchored else chunks[i - 1]
                te = chunks[i]
                folds.append((days[tr[0]], days[tr[-1]], days[te[0]], days[te[-1]]))
        ev.total = n_search + (min(10, n_search) if oos_start else 0) + len(folds) * (n_search + 1)

        def make_eval(s, e, label):
            def f(param_list):
                cfgs = [_with_dates(apply_params(base, p), s, e) for p in param_list]
                res = ev.run(cfgs, label)
                for p, r in zip(param_list, res):
                    all_rows.append({"params": p, "result": r, "phase": label})
                return [objective_value(r, oc.objective, oc.min_trades) for r in res]
            return f

        params, scores = search(space, oc.method, n_search, oc.seed, make_eval(is_start, is_end, "in_sample"), ev.workers)
        main_rows = [r for r in all_rows if r["phase"] == "in_sample"]
        results = [r["result"] for r in main_rows]
        scores_arr = np.array([objective_value(r, oc.objective, oc.min_trades) for r in results])
        trades_arr = np.array([(r.get("metrics") or {}).get("trades") or 0 for r in results])
        eligible = np.where(trades_arr >= oc.min_trades, scores_arr, -np.inf)
        stab = stability_scores([r["params"] for r in main_rows], scores_arr, [{"path": p.path, "values": p.grid()} for p in space])
        order = np.argsort(-np.where(np.isfinite(eligible), eligible, -1e18))
        best_i = int(order[0]) if len(order) else None
        # hold-out evaluation of the top configurations
        oos = []
        if oos_start is not None and best_i is not None:
            top = [int(i) for i in order[:min(10, len(order))] if np.isfinite(scores_arr[i])]
            res = ev.run([_with_dates(apply_params(base, main_rows[i]["params"]), oos_start, oc.base.end) for i in top], "out_of_sample")
            for i, r in zip(top, res):
                oos.append({"index": i, "params": main_rows[i]["params"], "is": main_rows[i]["result"].get("metrics"), "oos": r.get("metrics"), "oos_ok": r.get("ok"),
                            "oos_dates": r.get("dates"), "oos_returns": r.get("returns")})
        # walk-forward
        wf = []
        for fi, (trs, tre, tes, tee) in enumerate(folds):
            fold_rows_start = len(all_rows)
            p2, s2 = search(space, oc.method, n_search, oc.seed + fi + 1, make_eval(trs, tre, f"wf_train_{fi}"), ev.workers)
            fr = all_rows[fold_rows_start:]
            tr_scores = np.array([objective_value(r["result"], oc.objective, oc.min_trades) if ((r["result"].get("metrics") or {}).get("trades") or 0) >= oc.min_trades else -np.inf for r in fr])
            if not np.isfinite(tr_scores).any():
                tr_scores = np.array([objective_value(r["result"], oc.objective, 0) for r in fr])
            bi = int(np.argmax(tr_scores))
            test = ev.run([_with_dates(apply_params(base, fr[bi]["params"]), tes, tee)], f"wf_test_{fi}")[0]
            wf.append({"fold": fi, "train": [trs.isoformat(), tre.isoformat()], "test": [tes.isoformat(), tee.isoformat()], "best_params": fr[bi]["params"],
                       "train_metrics": fr[bi]["result"].get("metrics"), "test_metrics": test.get("metrics"), "test_dates": test.get("dates"), "test_returns": test.get("returns")})
    return analyse(oc, space, main_rows, scores_arr, stab, best_i, oos, wf, time.time() - t0)


def analyse(oc, space, rows, scores, stab, best_i, oos, wf, runtime) -> dict:
    n = len(rows)
    configs = []
    for i, r in enumerate(rows):
        res = r["result"]
        configs.append({"i": i, "params": r["params"], "ok": res.get("ok"), "error": res.get("error"), "metrics": res.get("metrics"),
                        "objective": scores[i] if np.isfinite(scores[i]) else None, "stability": stab[i] if np.isfinite(stab[i]) else None})
    out: dict = {"objective": oc.objective, "method": oc.method, "validation": oc.validation, "n_configs": n,
                 "total_trials": n + oc.prior_trials, "runtime_seconds": round(runtime, 2),
                 "params": [p.model_dump() | {"grid": p.grid()} for p in space], "configs": configs, "best_index": best_i, "warnings": []}
    W = out["warnings"]
    ok_rows = [i for i, r in enumerate(rows) if r["result"].get("ok") and r["result"].get("returns")]
    if any(r["result"].get("synthetic") for r in rows):
        out["synthetic"] = True
    if best_i is not None and np.isfinite(scores[best_i]) and scores[best_i] <= 0:
        W.append({"level": "error", "text": f"The best configuration's {oc.objective} is {scores[best_i]:.2f}: no tested configuration is profitable on this objective."})
    if best_i is None or not ok_rows:
        W.append({"level": "error", "text": "No configuration produced results."})
        return out
    best = rows[best_i]
    bm = best["result"]["metrics"]
    # multiple testing
    rets_best = np.array(best["result"]["returns"], float)
    trial_srs = []
    for i in ok_rows:
        r = np.array(rows[i]["result"]["returns"], float)
        sd = r.std(ddof=1)
        trial_srs.append(r.mean() / sd if sd > 0 else np.nan)
    out["deflated_sharpe"] = deflated_sharpe(rets_best, n + oc.prior_trials, np.array(trial_srs))
    # PBO on aligned return matrix
    L = min(len(rows[i]["result"]["returns"]) for i in ok_rows)
    if len(ok_rows) >= 4 and L >= 50:
        M = np.array([rows[i]["result"]["returns"][:L] for i in ok_rows]).T
        p = pbo_cscv(M, 10)
        out["pbo"] = {"pbo": p.get("pbo"), "combinations": p.get("combinations"), "is_vs_oos": p.get("is_vs_oos"), "logits": p.get("logits"), "reason": p.get("reason")}
    # pareto (return vs drawdown)
    cg = np.array([(r["result"].get("metrics") or {}).get("cagr") or np.nan for r in rows], float)
    dd = np.array([(r["result"].get("metrics") or {}).get("max_drawdown") or np.nan for r in rows], float)
    out["pareto"] = pareto_front(cg, dd)
    # sensitivity: marginal objective per parameter value
    sens = []
    for p in space:
        vals = p.grid()
        rowsv = []
        for v in vals:
            s = [scores[i] for i, r in enumerate(rows) if r["params"].get(p.path) == v and np.isfinite(scores[i])]
            one = None
            if best_i is not None:
                for i, r in enumerate(rows):
                    if r["params"].get(p.path) == v and all(r["params"].get(q.path) == best["params"].get(q.path) for q in space if q.path != p.path):
                        one = scores[i] if np.isfinite(scores[i]) else None
            rowsv.append({"value": v, "mean": float(np.mean(s)) if s else None, "max": float(np.max(s)) if s else None, "n": len(s), "at_best": one})
        sens.append({"path": p.path, "label": p.label or p.path, "values": rowsv})
    out["sensitivity"] = sens
    # warnings
    if (bm.get("trades") or 0) < oc.min_trades:
        W.append({"level": "error", "text": f"Best configuration has only {bm.get('trades')} trades (minimum {oc.min_trades}); no configuration met the sample-size requirement."})
    for p in space:
        g = p.grid()
        if len(g) > 2 and best["params"][p.path] in (g[0], g[-1]):
            W.append({"level": "warn", "text": f"Best {p.label or p.path} = {best['params'][p.path]} is at the edge of the tested range; the optimum may lie outside it."})
    sb = stab[best_i]
    if np.isfinite(sb) and scores[best_i] > 0 and sb < 0.6:
        W.append({"level": "warn", "text": f"Unstable optimum: neighbouring parameter values average only {sb:.0%} of the best objective (isolated spike)."})
    ds = out["deflated_sharpe"].get("dsr")
    if ds is not None and ds < 0.95:
        W.append({"level": "warn", "text": f"Deflated Sharpe ratio {ds:.2f} after {n + oc.prior_trials} trials: the best Sharpe is not significant once multiple testing is accounted for."})
    pb = (out.get("pbo") or {}).get("pbo")
    if pb is not None and pb > 0.4:
        W.append({"level": "warn", "text": f"Probability of backtest overfitting {pb:.0%} (CSCV): the in-sample winner frequently underperforms the median out of sample."})
    if (bm.get("max_capital_utilization") or 0) > 0.8:
        W.append({"level": "warn", "text": f"Best configuration used up to {bm['max_capital_utilization']:.0%} of equity as buying power (excessive leverage)."})
    out["oos"] = oos
    if oos:
        o0 = oos[0]
        is_v, oos_v = (o0["is"] or {}).get(oc.objective), (o0["oos"] or {}).get(oc.objective)
        if is_v is not None and oos_v is not None and is_v > 0:
            deg = 1 - oos_v / is_v
            out["oos_degradation"] = deg
            if deg > 0.5:
                W.append({"level": "warn", "text": f"Out-of-sample {oc.objective} fell {deg:.0%} versus in-sample ({is_v:.2f} -> {oos_v:.2f})."})
        rank_corr = None
        a = [(x["is"] or {}).get(oc.objective) for x in oos]
        b = [(x["oos"] or {}).get(oc.objective) for x in oos]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) >= 4:
            from scipy.stats import spearmanr
            rank_corr = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation)
        out["is_oos_rank_corr"] = rank_corr
    out["walk_forward"] = wf
    if wf:
        rets, dates = [], []
        for f in wf:
            rets += f["test_returns"] or []
            dates += f["test_dates"] or []
        eq = list(np.cumprod(1 + np.array(rets)) * oc.base.capital.initial_capital) if rets else []
        out["wf_equity"] = {"dates": dates, "equity": [float(x) for x in eq]}
        tr = [((f["train_metrics"] or {}).get("cagr")) for f in wf]
        te = [((f["test_metrics"] or {}).get("cagr")) for f in wf]
        if all(x is not None for x in tr + te) and np.mean(tr) > 0:
            out["wf_efficiency"] = float(np.mean(te) / np.mean(tr))
            if out["wf_efficiency"] < 0.5:
                W.append({"level": "warn", "text": f"Walk-forward efficiency {out['wf_efficiency']:.2f}: out-of-sample returns are less than half of in-sample."})
    W.append({"level": "info", "text": f"{n} configurations evaluated in this study ({n + oc.prior_trials} including earlier studies of this strategy). Rankings use in-sample data only."})
    return out
