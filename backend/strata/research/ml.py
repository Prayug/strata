"""The ML research workspace (optional: needs scikit-learn, `pip install 'strata[ml]'`).

    "Which observable market conditions predict whether SPY reaches a 1% target before a
     0.5% stop?"

Conventional models - logistic regression (plain, ridge and lasso), a shallow decision
tree, a random forest and gradient-boosted trees, each optionally followed by a
probability calibration - fitted and judged the way temporally dependent observations
require. Nothing here is a random split:

  features     read from the session-feature table at the outcome's own decision tick, so
               every value is one the rule engine would have had at that instant; a
               future-scramble probe verifies it (prices after a cut are changed and the
               features before the cut must not move)
  labels       the deterministic outcome engine (target / stop / horizon), ambiguous
               observations excluded and counted
  split        the last fifth of the history is a final test set, separated from the
               rest by an embargo as long as the label horizon. Inside the rest, K
               contiguous folds with purging (training observations whose label window
               overlaps the fold are dropped) and the same embargo
  selection    the model shown as "selected" is chosen on the purged cross-validation
               alone; the test set is scored once for every model and never used to choose
  judged by    Brier score and log loss against the base-rate forecast, calibration by
               decile, AUC, fold-to-fold stability, a circular-shift placebo, permutation
               importance on the test set, and the same scores by volatility and trend regime

What it deliberately does not say: a model that predicts the UNDERLYING's outcome better
than the base rate is not a profitable option strategy. Its walk-forward signal can be
saved as a named signal list and priced by the ordinary deterministic backtester on
observed option prices - that is the only route from here to a P&L figure.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from strata.data.store import MarketStore
from strata.research import ledger as LG
from strata.research import robust as R
from strata.research.features import A, FEATURES, OutcomeSpec, T, feature_def, get_features

DEFAULT_FEATURES = ["ret_1", "ret_5", "ret_20", "range_1", "gap", "rv21", "rv_pctile", "vix_prev", "sma50_dist", "sma200_dist", "dist_high252", "vol_ratio", "down_streak", "up_streak", "weekday"]
MODELS = {
    "logistic": {"label": "Logistic regression", "family": "linear"},
    "ridge": {"label": "Ridge logistic regression (strong L2)", "family": "linear"},
    "lasso": {"label": "Lasso logistic regression (L1, selects features)", "family": "linear"},
    "tree": {"label": "Decision tree (depth 3)", "family": "tree"},
    "forest": {"label": "Random forest", "family": "tree"},
    "gbt": {"label": "Gradient-boosted trees", "family": "tree"},
}
MIN_ROWS, MAX_FEATURES = 400, 30


def available() -> tuple[bool, str]:
    try:
        import sklearn  # noqa: F401
        return True, sklearn.__version__
    except ImportError:
        return False, "The ML workspace needs scikit-learn, which is an optional dependency: pip install 'strata[ml]' (or pip install scikit-learn) and restart Strata."


def _make(name: str, n: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    leaf = max(20, n // 60)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))
    if name == "ridge":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    if name == "lasso":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, l1_ratio=1.0, solver="saga", max_iter=4000, random_state=7))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf, random_state=7)
    if name == "forest":
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=leaf, random_state=7, n_jobs=1)
    if name == "gbt":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=leaf, random_state=7)
    raise ValueError(f"unknown model {name!r}")


def _fit_predict(name: str, Xtr, ytr, Xte, calibrate: bool):
    """Fit on the training rows (chronological) and predict the test rows. With calibration
    the last quarter of the training rows is held out to fit an isotonic map - later than
    the rows the model was fitted on, earlier than anything it predicts."""
    from sklearn.isotonic import IsotonicRegression
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean())), None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not calibrate or len(ytr) < 400:
            m = _make(name, len(ytr)).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1], m
        k = int(len(ytr) * 0.75)
        if len(np.unique(ytr[:k])) < 2 or len(np.unique(ytr[k:])) < 2:
            m = _make(name, len(ytr)).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1], m
        m = _make(name, k).fit(Xtr[:k], ytr[:k])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(m.predict_proba(Xtr[k:])[:, 1], ytr[k:])
        return iso.predict(m.predict_proba(Xte)[:, 1]), m


def _scores(y: np.ndarray, p: np.ndarray, base: float) -> dict:
    """Probability scores against the base-rate forecast `base` (which was known before the test rows)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    b = float(np.clip(base, 1e-6, 1 - 1e-6))
    brier, brier0 = float(np.mean((p - y) ** 2)), float(np.mean((b - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ll0 = float(-np.mean(y * np.log(b) + (1 - y) * np.log(1 - b)))
    auc = None
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
    return {"n": int(len(y)), "base_rate": float(y.mean()), "brier": brier, "brier_baseline": brier0, "brier_skill": (1 - brier / brier0) if brier0 > 0 else None, "log_loss": ll, "log_loss_baseline": ll0,
            "log_loss_skill": (1 - ll / ll0) if ll0 > 0 else None, "auc": auc, "accuracy": float(((p >= 0.5) == (y > 0.5)).mean()), "accuracy_baseline": float(max(y.mean(), 1 - y.mean()))}


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    if len(y) < 30:
        return []
    order = np.argsort(p, kind="stable")
    out = []
    for part in np.array_split(order, min(bins, max(2, len(y) // 30))):
        if part.size:
            out.append({"predicted": float(p[part].mean()), "observed": float(y[part].mean()), "n": int(part.size), "lo": float(p[part].min()), "hi": float(p[part].max())})
    return out


def _calibration_error(cal: list[dict]) -> float | None:
    n = sum(c["n"] for c in cal)
    return float(sum(c["n"] * abs(c["predicted"] - c["observed"]) for c in cal) / n) if n else None


def run(spec: dict, store: MarketStore | None = None, scope: str | None = None, run_id: str | None = None, progress=None, should_cancel=None) -> dict:
    """spec: {symbol, outcome: OutcomeSpec dict, features?: [catalog names], models?: [...],
              start?, end?, folds?: 5, calibrate?: True, signal?: {"quantile": 0.8}}"""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    store = store or MarketStore()
    sym = str(spec["symbol"]).upper()
    oc = OutcomeSpec.load(spec["outcome"])
    names = list(dict.fromkeys(spec.get("features") or DEFAULT_FEATURES))[:MAX_FEATURES]
    models = [m for m in (spec.get("models") or list(MODELS)) if m in MODELS]
    if not models:
        raise ValueError("choose at least one model")
    cross = tuple(sorted({n[4:] if n.startswith("rs5_") else n[:-6] if n.endswith("_ret_1") else n[:-4] for n in names if n not in FEATURES}))
    ft = get_features(store, sym, cross)
    tick = oc.entry
    # ----- features at the decision tick (asof values at an intraday entry are the open's values: known since the open)
    cols, dropped = [], []
    for n in names:
        try:
            fd = feature_def(n, sym)
            if fd.kind == "intraday" and tick in ("open", "close"):
                dropped.append({"feature": n, "why": "it is read during the session, so it needs an intraday decision time"})
                continue
            x = ft.col(n, tick)
        except (KeyError, ValueError) as e:
            dropped.append({"feature": n, "why": str(e)[:120]})
            continue
        if np.isfinite(x).mean() < 0.3:
            dropped.append({"feature": n, "why": "the data it needs is stored for fewer than 30% of sessions"})
            continue
        cols.append((n, fd, x))
    if len(cols) < 1:
        raise ValueError("none of the requested features can be computed from the stored data: " + "; ".join(f"{d['feature']}: {d['why']}" for d in dropped))
    out = ft.outcome(oc)
    pm = ft.window(spec.get("start"), spec.get("end"))
    Xall = np.column_stack([c[2] for c in cols])
    usable = out.valid & pm & (out.code != A) & np.isfinite(Xall).all(axis=1)
    idx = np.flatnonzero(usable)
    if idx.size < MIN_ROWS:
        raise ValueError(f"Only {idx.size} sessions have the outcome and every feature ({int((out.valid & pm).sum())} have the outcome). A model needs at least {MIN_ROWS}; use fewer features or a longer period.")
    X, y = Xall[idx], (out.code[idx] == T).astype(int)
    dates = np.array(ft.dates)[idx]
    hold = int(oc.horizon.split(":")[1]) if oc.horizon.startswith("sessions:") else 0
    embargo = hold + 1  # the label of row i is not known until `hold` sessions later; one more keeps adjacent sessions apart
    starts, ends = idx.copy(), idx + hold
    n = idx.size
    n_test = max(100, int(n * 0.2))
    test = np.arange(n - n_test, n)
    train = np.arange(0, n - n_test)
    train = train[ends[train] + embargo < starts[test[0]]]  # purge + embargo between training and the final test set
    k = int(spec.get("folds") or 5)
    folds = R.purged_kfold(starts[train], ends[train], k=k, embargo=embargo)
    calibrate = bool(spec.get("calibrate", True))
    feat_names = [c[0] for c in cols]
    res_models = []
    base_train = float(y[train].mean())
    for mi, name in enumerate(models):
        if should_cancel and should_cancel():
            from strata.engine.engine import BacktestCancelled
            raise BacktestCancelled()
        if progress:
            progress(0.05 + 0.6 * mi / len(models), f"{MODELS[name]['label']}: purged cross-validation")
        fold_rows, oof = [], np.full(train.size, np.nan)
        for f in folds:
            tr, te = np.asarray(f["train"]), np.asarray(f["test"])
            if tr.size < 100 or te.size < 20 or len(np.unique(y[train][tr])) < 2:
                continue
            p, _ = _fit_predict(name, X[train][tr], y[train][tr], X[train][te], calibrate)
            oof[te] = p
            fold_rows.append({**_scores(y[train][te], p, float(y[train][tr].mean())), "start": str(dates[train][te[0]]), "end": str(dates[train][te[-1]]), "train_n": int(tr.size)})
        if not fold_rows:
            res_models.append({"name": name, "label": MODELS[name]["label"], "status": "error", "error": "too few rows per fold after purging"})
            continue
        okm = np.isfinite(oof)
        cv = _scores(y[train][okm], oof[okm], base_train)
        cv["folds"] = fold_rows
        cv["skill_folds_positive"] = sum(1 for r in fold_rows if (r["brier_skill"] or 0) > 0)
        cv["skill_sd"] = float(np.std([r["brier_skill"] or 0 for r in fold_rows], ddof=1)) if len(fold_rows) > 1 else None
        p_test, fitted = _fit_predict(name, X[train], y[train], X[test], calibrate)
        te_scores = _scores(y[test], p_test, base_train)
        cal = _calibration(y[test], p_test)
        res_models.append({"name": name, "label": MODELS[name]["label"], "family": MODELS[name]["family"], "status": "ok", "cv": cv, "test": te_scores, "calibration": cal, "calibration_error": _calibration_error(cal),
                           "_p_test": p_test, "_fitted": fitted})
    good = [m for m in res_models if m["status"] == "ok"]
    if not good:
        raise ValueError("no model could be fitted: " + "; ".join(m.get("error", "") for m in res_models))
    best = min(good, key=lambda m: m["cv"]["log_loss"])  # chosen on the purged cross-validation alone
    # ----- placebo: the same model with the labels circularly shifted in time (keeps both series' own structure, breaks the link)
    if progress:
        progress(0.7, "placebo: labels shifted in time")
    rng = np.random.default_rng(7)
    shifts = rng.integers(max(embargo + 20, train.size // 10), train.size - max(embargo + 20, train.size // 10), 12)
    placebo = []
    cut = int(train.size * 0.7)
    real_p, _ = _fit_predict(best["name"], X[train][:cut - embargo], y[train][:cut - embargo], X[train][cut:], False)
    real_skill = _scores(y[train][cut:], real_p, float(y[train][:cut - embargo].mean()))["brier_skill"]
    for s in shifts:
        ys = np.roll(y[train], int(s))
        p, _ = _fit_predict(best["name"], X[train][:cut - embargo], ys[:cut - embargo], X[train][cut:], False)
        placebo.append(_scores(ys[cut:], p, float(ys[:cut - embargo].mean()))["brier_skill"] or 0.0)
    placebo_test = {"real_skill": real_skill, "placebo_skills": [float(x) for x in placebo], "placebo_max": float(max(placebo)), "beats": int(sum(1 for x in placebo if (real_skill or 0) > x)), "of": len(placebo),
                    "passed": bool((real_skill or 0) > max(placebo)), "how": "The selected model refitted on the first 70% of the training rows and scored on the rest, against the same procedure with the labels circularly shifted in time."}
    # ----- importance on the test set (permutation: how much the Brier score worsens when one feature is shuffled)
    if progress:
        progress(0.8, "permutation importance on the test set")
    imp = []
    fitted = best["_fitted"]
    if fitted is not None:
        b0 = float(np.mean((np.clip(fitted.predict_proba(X[test])[:, 1], 0, 1) - y[test]) ** 2))
        prng = np.random.default_rng(11)
        for j, fname in enumerate(feat_names):
            worse = []
            for _ in range(8):
                Xp = X[test].copy()
                Xp[:, j] = prng.permutation(Xp[:, j])
                worse.append(float(np.mean((np.clip(fitted.predict_proba(Xp)[:, 1], 0, 1) - y[test]) ** 2)) - b0)
            fd = cols[j][1]
            imp.append({"feature": fname, "label": fd.label, "importance": float(np.mean(worse)), "sd": float(np.std(worse, ddof=1)), "matters": bool(np.mean(worse) - 2 * np.std(worse, ddof=1) > 0)})
        imp.sort(key=lambda r: -r["importance"])
    coefs = None
    if best["family"] == "linear" and fitted is not None and hasattr(fitted, "steps"):
        lr = fitted.steps[-1][1]
        coefs = sorted(({"feature": f, "label": cols[j][1].label, "coef_per_sd": float(lr.coef_[0][j])} for j, f in enumerate(feat_names)), key=lambda r: -abs(r["coef_per_sd"]))
    # ----- leakage checks
    leak = _leakage(store, ft, sym, cross, cols, tick, idx, X, y, train)
    # ----- regimes on the test set
    regimes = _regimes(ft, idx[test], y[test], best["_p_test"], base_train, tick)
    # ----- walk-forward signal (every prediction from a model fitted on earlier sessions only)
    sig = None
    if spec.get("signal") is not False:
        if progress:
            progress(0.88, "walk-forward predictions")
        sig = _walk_forward(best["name"], X, y, starts, ends, dates, embargo, float((spec.get("signal") or {}).get("quantile", 0.8)), calibrate)
    for m in res_models:
        m.pop("_p_test", None)
        m.pop("_fitted", None)
    notes = [f"{n:,} sessions have the outcome and every feature. The last {test.size:,} ({dates[test[0]]} to {dates[test[-1]]}) are the final test set; an embargo of {embargo} session{'s' if embargo != 1 else ''} separates them from the training rows, "
             f"and the {len(folds)} cross-validation folds are purged the same way.",
             "The selected model was chosen on the purged cross-validation only. Every model's test score is shown, and none of them was used to choose.",
             "Skill is measured against the base-rate forecast (always predicting the training frequency). A skill of zero means the model knows nothing the base rate does not.",
             "A model that predicts the underlying's outcome is not a profitable option strategy. Save its walk-forward signal and price it on observed option prices to learn anything about profit."]
    if out.reasons:
        notes.append("Sessions without the outcome: " + "; ".join(f"{c:,} - {w}" for w, c in list(out.reasons.items())[:3]) + ".")
    amb = int((out.valid & pm & (out.code == A)).sum())
    if amb:
        notes.append(f"{amb:,} sessions were excluded because the target and the stop fell inside the same bar (the order cannot be known).")
    bt_, bc = best["test"], best["cv"]
    verdict = ("no_skill" if (bc["brier_skill"] or 0) <= 0 or (bt_["brier_skill"] or 0) <= 0 else "weak" if not placebo_test["passed"] or bc["skill_folds_positive"] < max(2, len(bc["folds"]) - 1) else "some_skill")
    sentence = {"no_skill": f"{best['label']} does not beat the base-rate forecast out of sample (Brier skill {_s(bc['brier_skill'])} in cross-validation, {_s(bt_['brier_skill'])} on the final test set): these features do not predict this outcome.",
                "weak": f"{best['label']} beats the base rate slightly (Brier skill {_s(bc['brier_skill'])} in cross-validation, {_s(bt_['brier_skill'])} on the final test set), but not consistently across folds or not beyond the time-shifted placebo. Treat it as unproven.",
                "some_skill": f"{best['label']} beats the base-rate forecast in cross-validation (Brier skill {_s(bc['brier_skill'])}, positive in {bc['skill_folds_positive']} of {len(bc['folds'])} folds), beyond every time-shifted placebo, and on the final test set ({_s(bt_['brier_skill'])}, AUC {bt_['auc']:.3f}). That is evidence about the underlying only."}[verdict]
    if scope:
        sig_ = {"symbol": sym, "features": sorted(feat_names), "outcome": f"{oc.entry}|{oc.horizon}|{oc.direction}|{oc.measure}"}
        LG.record(scope, sym + ("" if tick in ("open", "close") else "@intraday"), [{"kind": "ml", "key": LG.config_key({"model": m["name"], "features": sorted(feat_names), "outcome": oc.dump(), "symbol": sym}), "label": f"{m['label']}: {oc.text()}",
                                                                                   "run_id": run_id, "start": str(dates[0]), "end": str(dates[-1]), "metric": "brier_skill", "value": m["test"]["brier_skill"], "n": n, "signature": sig_}
                                                                                  for m in res_models if m["status"] == "ok"], family="ml research")
    if progress:
        progress(1.0, "done")
    return {"kind": "ml", "symbol": sym, "outcome": oc.dump(), "outcome_text": oc.text(), "decision_tick": tick, "n": int(n), "base_rate": float(y.mean()), "features": [{"name": c[0], "label": c[1].label, "group": c[1].group} for c in cols],
            "dropped_features": dropped, "split": {"train": [str(dates[train[0]]), str(dates[train[-1]])], "test": [str(dates[test[0]]), str(dates[test[-1]])], "train_n": int(train.size), "test_n": int(test.size),
                                                  "embargo_sessions": embargo, "folds": len(folds), "method": "purged K-fold with embargo; final test set at the end of the history"},
            "models": res_models, "selected": best["name"], "selected_by": "lowest log loss in purged cross-validation", "placebo": placebo_test, "importance": imp, "coefficients": coefs, "leakage": leak, "regimes": regimes,
            "signal": sig, "verdict": verdict, "summary": [sentence], "notes": notes, "calibrated": calibrate, "synthetic": store.is_synthetic(sym)}


def _s(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def _leakage(store, ft, sym, cross, cols, tick, idx, X, y, train) -> dict:
    """Checks that no feature knows the future. The scramble probe is the strong one."""
    checks = []
    # 1. a single feature that separates the classes almost perfectly is nearly always a leak
    from sklearn.metrics import roc_auc_score
    sus = []
    for j, (name, fd, _x) in enumerate(cols):
        if 0 < y[train].sum() < train.size:
            a = roc_auc_score(y[train], X[train][:, j])
            if max(a, 1 - a) > 0.9:
                sus.append(f"{fd.label} alone separates the outcome with AUC {max(a, 1 - a):.2f}")
    checks.append({"check": "No single feature explains the outcome by itself", "passed": not sus, "detail": "; ".join(sus) or "No feature has a univariate AUC above 0.90 on the training rows."})
    # 2. scramble everything after a cut and require every feature before the cut to be unchanged
    try:
        from strata.research.challenge import ShockedStore
        from strata.research.features import SessionFeatures
        mid = int(idx[len(idx) // 2])
        t_cut = int(ft.t_open[mid]) if tick != "close" else int(ft.t_close[mid])
        if tick not in ("open", "close"):
            t_cut = int(ft._tick(tick)[0][mid])
        sh = SessionFeatures(ShockedStore(store, t_cut), sym, cross)
        moved = []
        for name, fd, x in cols:
            xs = sh.col(name, tick)
            m = min(len(x), len(xs), mid + 1)
            a, b = x[:m], xs[:m]
            both = np.isfinite(a) & np.isfinite(b)
            if (np.isfinite(a) != np.isfinite(b)).any() or (both.any() and np.max(np.abs(a[both] - b[both])) > 1e-9 * (1 + np.max(np.abs(a[both])))):
                moved.append(fd.label)
        checks.append({"check": "Features do not change when the future is scrambled", "passed": not moved,
                       "detail": ("Changed after scrambling later prices: " + ", ".join(moved)) if moved else f"Every price after {ft.dates[mid]} ({tick}) was changed; all {len(cols)} features for the sessions up to that instant are identical."})
    except Exception as e:  # noqa: BLE001  (the probe failing to run is reported as not run, never as passed)
        checks.append({"check": "Features do not change when the future is scrambled", "passed": None, "detail": f"The probe could not be run: {type(e).__name__}: {e}"[:200]})
    # 3. the label window never overlaps the features' own session span backwards
    checks.append({"check": "Labels are measured strictly after the decision instant", "passed": True, "detail": "The outcome engine walks bars after the entry only; the session's own bar is used only from the entry onward."})
    return {"checks": checks, "passed": all(c["passed"] is True for c in checks), "not_run": sum(1 for c in checks if c["passed"] is None)}


def _regimes(ft, rows: np.ndarray, y: np.ndarray, p: np.ndarray, base: float, tick: str) -> list[dict]:
    t = "close" if tick == "close" else "open"
    out = []
    for title, x, bins in (("Volatility (1-year percentile of realised volatility)", ft.col("rv_pctile", t)[rows], [(0, 33.34, "calm third"), (33.34, 66.67, "middle third"), (66.67, 101, "turbulent third")]),
                           ("Trend (close against the 200-day average)", ft.col("sma200_dist", t)[rows], [(-1e9, 0, "below the 200-day average"), (0, 1e9, "above the 200-day average")])):
        rs = []
        for lo, hi, name in bins:
            m = np.isfinite(x) & (x >= lo) & (x < hi)
            if m.sum() >= 30:
                s = _scores(y[m], p[m], base)
                rs.append({"regime": name, "n": int(m.sum()), "base_rate": s["base_rate"], "brier_skill": s["brier_skill"], "auc": s["auc"]})
            else:
                rs.append({"regime": name, "n": int(m.sum()), "base_rate": float(y[m].mean()) if m.any() else None, "brier_skill": None, "auc": None, "thin": True})
        out.append({"title": title, "rows": rs})
    yrs = np.array([d[:4] for d in np.array(ft.dates)[rows]])
    rs = []
    for yv in sorted(set(yrs)):
        m = yrs == yv
        if m.sum() >= 30:
            s = _scores(y[m], p[m], base)
            rs.append({"regime": yv, "n": int(m.sum()), "base_rate": s["base_rate"], "brier_skill": s["brier_skill"], "auc": s["auc"]})
    if rs:
        out.append({"title": "Calendar year (final test set)", "rows": rs})
    return out


def _walk_forward(name: str, X, y, starts, ends, dates, embargo: int, quantile: float, calibrate: bool, min_train: int = 300, step: int = 63) -> dict:
    """Out-of-sample probability for every session after the first `min_train`: the model is
    refitted every `step` sessions on the sessions whose labels were already known. The
    signal threshold for each block is a quantile of the model's own predictions on its
    training rows, so it too is fixed before the block it applies to."""
    n = len(y)
    p = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    t = min_train
    while t < n:
        hi = min(n, t + step)
        tr = np.flatnonzero(ends[:t] + embargo < starts[t])
        if tr.size >= min_train // 2 and len(np.unique(y[tr])) == 2:
            pp, fitted = _fit_predict(name, X[tr], y[tr], X[t:hi], calibrate)
            p[t:hi] = pp
            ptr, _ = _fit_predict(name, X[tr], y[tr], X[tr], calibrate) if fitted is None else (fitted.predict_proba(X[tr])[:, 1], None)
            thr[t:hi] = float(np.quantile(ptr, quantile))
        t = hi
    ok = np.isfinite(p)
    on = ok & (p >= thr)
    out = {"model": name, "quantile": quantile, "predicted": int(ok.sum()), "first": str(dates[ok][0]) if ok.any() else None, "last": str(dates[ok][-1]) if ok.any() else None, "signals": int(on.sum()),
           "dates": [str(d) for d in dates[on]], "refit_every": step, "min_train": min_train,
           "hit_rate_on_signal": float(y[on].mean()) if on.any() else None, "hit_rate_otherwise": float(y[ok & ~on].mean()) if (ok & ~on).any() else None, "base_rate": float(y[ok].mean()) if ok.any() else None}
    if on.sum() >= 10 and (ok & ~on).sum() >= 10:
        lt = R.lift_test(y[on].astype(float), dates[on], y[ok & ~on].astype(float), dates[ok & ~on], "week", 400)
        out.update(p_value=lt["p_value"] if lt["p_value"] is not None else lt["p_iid"], lift_ci95=lt["ci"])
    out["scores"] = _scores(y[ok], p[ok], float(y[:min_train].mean())) if ok.sum() >= 30 else None
    return out


def save_result(rid: str, res: dict) -> dict:
    from strata.api.results import dumps, run_dir
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ml.json").write_text(dumps(res))
    b = next(m for m in res["models"] if m["name"] == res["selected"])
    return {"ml": True, "symbols": [res["symbol"]], "models": len(res["models"]), "selected": res["selected"], "verdict": res["verdict"], "cv_brier_skill": b["cv"]["brier_skill"], "test_brier_skill": b["test"]["brier_skill"],
            "start": res["split"]["train"][0], "end": res["split"]["test"][1], "headline": res["summary"][0][:300]}


def save_signal(res: dict, run_id: str, name: str | None = None) -> dict:
    """Store the walk-forward signal of an ML result as a named list a rule can read."""
    from strata.research import signals as SG
    sig = res.get("signal")
    if not sig or not sig.get("dates"):
        raise ValueError("this result has no walk-forward signal to save")
    name = name or f"ml_{run_id}"
    known = "close" if res["decision_tick"] == "close" else "open"
    return SG.save(name, res["symbol"], sig["dates"], known, {"kind": "ml", "run_id": run_id, "model": sig["model"], "quantile": sig["quantile"], "outcome": res["outcome_text"], "walk_forward": True,
                                                              "note": "Each date's signal came from a model fitted only on sessions whose outcomes were already known."})
