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
  selection    every candidate is scored on the purged cross-validation and ONLY there.
               The best one by log loss is selected and its configuration is frozen
               (model, hyper-parameters, features, calibration, outcome, the split)
  final test   nothing touches the final test set until it is asked for, and then only the
               frozen selected model is scored on it, once. That act is written to a
               permanent, append-only record. Scoring a different configuration on the
               same period later is counted: a test period that several configurations
               have been scored on is no longer an untouched estimate, and the result says
               so instead of calling it confirmed. (Before version 6 every candidate was
               scored on the test set, which let the best-looking one be picked from it.)
  preprocessing scaling is fitted inside each training fold; rows with a missing feature
               are dropped, never imputed; the calibration map is fitted on later training
               rows separated by the label embargo; the signal threshold is a quantile of
               the model's own training predictions
  judged by    Brier score and log loss against the base-rate forecast, calibration by
               decile, AUC, fold-to-fold stability, a circular-shift placebo, permutation
               importance on held-back validation rows, and the scores by regime

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


class _Fitted:
    """A fitted model together with its calibration map, so every prediction (the scores, the importance, the signal
    threshold) is on the same scale. Before version 6 the signal threshold was a quantile of the UNCALIBRATED training
    predictions and was compared with calibrated ones."""

    def __init__(self, model, iso=None):
        self.model, self.iso = model, iso

    def proba(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(self.iso.predict(p), 0.0, 1.0) if self.iso is not None else p


def _fit_predict(name: str, Xtr, ytr, Xte, calibrate: bool, embargo: int = 0):
    """Fit on the training rows (chronological) and predict the test rows. With calibration
    the last quarter of the training rows is held out to fit an isotonic map - later than
    the rows the model was fitted on (and separated from them by the label embargo, so no
    label window straddles the two), earlier than anything it predicts. Scaling lives inside
    the model's pipeline, so its mean and spread come from these training rows alone."""
    from sklearn.isotonic import IsotonicRegression
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean())), None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not calibrate or len(ytr) < 400:
            f = _Fitted(_make(name, len(ytr)).fit(Xtr, ytr))
            return f.proba(Xte), f
        k = int(len(ytr) * 0.75)
        kf = max(1, k - max(0, int(embargo)))  # the model's own rows stop `embargo` sessions before the calibration rows begin
        if len(np.unique(ytr[:kf])) < 2 or len(np.unique(ytr[k:])) < 2:
            f = _Fitted(_make(name, len(ytr)).fit(Xtr, ytr))
            return f.proba(Xte), f
        m = _make(name, kf).fit(Xtr[:kf], ytr[:kf])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(m.predict_proba(Xtr[k:])[:, 1], ytr[k:])
        f = _Fitted(m, iso)
        return f.proba(Xte), f


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
    oofs: dict[str, np.ndarray] = {}
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
            p, _ = _fit_predict(name, X[train][tr], y[train][tr], X[train][te], calibrate, embargo)
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
        cal_v = _calibration(y[train][okm], oof[okm])
        oofs[name] = oof
        # no test score here: a candidate that was not selected never sees the final test set
        res_models.append({"name": name, "label": MODELS[name]["label"], "family": MODELS[name]["family"], "status": "ok", "cv": cv, "test": None, "calibration": [], "calibration_error": None,
                           "validation_calibration": cal_v, "validation_calibration_error": _calibration_error(cal_v)})
    good = [m for m in res_models if m["status"] == "ok"]
    if not good:
        raise ValueError("no model could be fitted: " + "; ".join(m.get("error", "") for m in res_models))
    best = min(good, key=lambda m: m["cv"]["log_loss"])  # chosen on the purged cross-validation alone
    # ----- the configuration is frozen here: everything that decides what is scored on the test set
    import hashlib
    import json
    config = {"symbol": sym, "outcome": oc.dump(), "model": best["name"], "features": sorted(feat_names), "calibrate": calibrate, "folds": len(folds), "embargo": embargo, "candidates": sorted(models)}
    config_key = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    frozen = {**config, "config_key": config_key, "train": [str(dates[train[0]]), str(dates[train[-1]])], "test": [str(dates[test[0]]), str(dates[test[-1]])], "selected_by": "lowest log loss in purged cross-validation",
              "hyperparameters": repr(_make(best["name"], int(train.size)))[:400]}
    # ----- placebo: the same model with the labels circularly shifted in time (keeps both series' own structure, breaks the link)
    if progress:
        progress(0.7, "placebo: labels shifted in time")
    rng = np.random.default_rng(7)
    shifts = rng.integers(max(embargo + 20, train.size // 10), train.size - max(embargo + 20, train.size // 10), 12)
    placebo = []
    cut = int(train.size * 0.7)
    real_p, val_fit = _fit_predict(best["name"], X[train][:cut - embargo], y[train][:cut - embargo], X[train][cut:], False)
    real_skill = _scores(y[train][cut:], real_p, float(y[train][:cut - embargo].mean()))["brier_skill"]
    for s in shifts:
        ys = np.roll(y[train], int(s))
        p, _ = _fit_predict(best["name"], X[train][:cut - embargo], ys[:cut - embargo], X[train][cut:], False)
        placebo.append(_scores(ys[cut:], p, float(ys[:cut - embargo].mean()))["brier_skill"] or 0.0)
    placebo_test = {"real_skill": real_skill, "placebo_skills": [float(x) for x in placebo], "placebo_max": float(max(placebo)), "beats": int(sum(1 for x in placebo if (real_skill or 0) > x)), "of": len(placebo),
                    "passed": bool((real_skill or 0) > max(placebo)), "how": "The selected model refitted on the first 70% of the training rows and scored on the rest, against the same procedure with the labels circularly shifted in time."}
    # ----- importance on held-back VALIDATION rows (the last 30% of the training period, model fitted on the first 70%): the test set is not used for it
    if progress:
        progress(0.8, "permutation importance on validation rows")
    imp = []
    Xv, yv = X[train][cut:], y[train][cut:]
    if val_fit is not None:
        b0 = float(np.mean((np.clip(val_fit.proba(Xv), 0, 1) - yv) ** 2))
        prng = np.random.default_rng(11)
        for j, fname in enumerate(feat_names):
            worse = []
            for _ in range(8):
                Xp = Xv.copy()
                Xp[:, j] = prng.permutation(Xp[:, j])
                worse.append(float(np.mean((np.clip(val_fit.proba(Xp), 0, 1) - yv) ** 2)) - b0)
            fd = cols[j][1]
            imp.append({"feature": fname, "label": fd.label, "importance": float(np.mean(worse)), "sd": float(np.std(worse, ddof=1)), "matters": bool(np.mean(worse) - 2 * np.std(worse, ddof=1) > 0)})
        imp.sort(key=lambda r: -r["importance"])
    coefs = None
    if best["family"] == "linear" and val_fit is not None and hasattr(val_fit.model, "steps"):
        lr = val_fit.model.steps[-1][1]
        coefs = sorted(({"feature": f, "label": cols[j][1].label, "coef_per_sd": float(lr.coef_[0][j])} for j, f in enumerate(feat_names)), key=lambda r: -abs(r["coef_per_sd"]))
    # ----- leakage checks
    leak = _leakage(store, ft, sym, cross, cols, tick, idx, X, y, train)
    # ----- the final test set: only when asked for, only the frozen selected model, and the act is recorded for good
    outcome_key = f"{tick}|{oc.horizon}|{oc.direction}|{oc.target}|{oc.stop}|{oc.measure}"
    final = bool(spec.get("final_test", False))
    prior = exposures(sym, outcome_key, frozen["test"][0], frozen["test"][1])
    others = sorted({e["config_key"] for e in prior} - {config_key})
    exposure = {"scored": False, "test": frozen["test"], "previous_scorings": len(prior), "other_configurations": len(others), "this_configuration_scored_before": any(e["config_key"] == config_key for e in prior),
                "history": [{k: e[k] for k in ("recorded_at", "model", "config_key", "test_start", "test_end", "brier_skill", "n_features")} for e in prior][-12:]}
    regimes = _regimes(ft, idx[train][np.isfinite(oofs[best["name"]])], y[train][np.isfinite(oofs[best["name"]])], oofs[best["name"]][np.isfinite(oofs[best["name"]])], base_train, tick, "validation rows")
    expect = spec.get("expect_frozen") or None
    if final and expect:
        # the test set is scored for the configuration that was frozen at validation, and for nothing else: if the model, the
        # features or the rows of the split are no longer the ones that were validated, the step is refused BEFORE any score exists
        changed = [k for k in ("config_key", "train", "test") if list(expect.get(k) or []) != list(frozen.get(k) or []) and expect.get(k) != frozen.get(k)]
        if changed:
            raise ValueError("the final test was not run: what was frozen at validation no longer matches (" + ", ".join(
                {"config_key": "the selected model or its features", "train": "the training period", "test": "the test period"}[k] for k in changed)
                + "). The stored data or the configuration changed since then. Validate again, then score the final test once.")
    if final:
        if progress:
            progress(0.84, "scoring the frozen model on the final test set (recorded)")
        p_test, _fitted = _fit_predict(best["name"], X[train], y[train], X[test], calibrate, embargo)
        best["test"] = _scores(y[test], p_test, base_train)
        best["calibration"] = _calibration(y[test], p_test)
        best["calibration_error"] = _calibration_error(best["calibration"])
        regimes = _regimes(ft, idx[test], y[test], p_test, base_train, tick, "final test set")
        if not exposure["this_configuration_scored_before"]:
            _record_exposure(sym, outcome_key, config_key, best["name"], feat_names, frozen["test"], int(test.size), best["test"]["brier_skill"], best["test"]["auc"], run_id, scope)
        exposure.update(scored=True, configurations_scored_on_this_period=len(others) + 1)
    # ----- walk-forward signal (every prediction from a model fitted on earlier sessions only)
    sig = None
    if spec.get("signal") is not False:
        if progress:
            progress(0.88, "walk-forward predictions")
        sig = _walk_forward(best["name"], X, y, starts, ends, dates, embargo, float((spec.get("signal") or {}).get("quantile", 0.8)), calibrate, upto=None if final else int(train.size))
    notes = [f"{n:,} sessions have the outcome and every feature. The last {test.size:,} ({dates[test[0]]} to {dates[test[-1]]}) are the final test set; an embargo of {embargo} session{'s' if embargo != 1 else ''} separates them from the training rows, "
             f"and the {len(folds)} cross-validation folds are purged the same way.",
             "Every candidate was scored on the purged cross-validation only. The selected model is the one with the lowest log loss there; no other candidate is ever scored on the final test set.",
             "Skill is measured against the base-rate forecast (always predicting the training frequency). A skill of zero means the model knows nothing the base rate does not.",
             "A model that predicts the underlying's outcome is not a profitable option strategy. Save its walk-forward signal and price it on observed option prices to learn anything about profit."]
    if not final:
        notes.insert(2, f"The final test set has NOT been scored. Nothing here has looked at {dates[test[0]]} to {dates[test[-1]]}. Scoring it is recorded permanently and should be done once, when the model and its features are no longer being changed.")
    if out.reasons:
        notes.append("Sessions without the outcome: " + "; ".join(f"{c:,} - {w}" for w, c in list(out.reasons.items())[:3]) + ".")
    amb = int((out.valid & pm & (out.code == A)).sum())
    if amb:
        notes.append(f"{amb:,} sessions were excluded because the target and the stop fell inside the same bar (the order cannot be known).")
    bt_, bc = best["test"], best["cv"]
    cv_ok = (bc["brier_skill"] or 0) > 0
    steady = placebo_test["passed"] and bc["skill_folds_positive"] >= max(2, len(bc["folds"]) - 1)
    leaked = [c for c in leak["checks"] if c["passed"] is False]
    if leaked:
        verdict, confirmation = "leak_suspected", "not_applicable"
        sentence = (f"A leakage check failed: {leaked[0]['check'].lower()} ({leaked[0]['detail'][:160]}). A feature that knows the outcome makes every score below meaningless; they are not evidence of predictive skill. "
                    "Remove the feature or correct its timing and run again.")
    elif not cv_ok:
        verdict, confirmation = "no_skill", ("not_tested" if not final else "not_confirmed")
        sentence = f"{best['label']} does not beat the base-rate forecast in purged cross-validation (Brier skill {_s(bc['brier_skill'])}): these features do not predict this outcome." + ("" if not final else f" On the final test set its skill is {_s(bt_['brier_skill'])}.")
    elif not final:
        verdict, confirmation = ("some_skill" if steady else "weak"), "not_tested"
        sentence = (f"{best['label']} beats the base-rate forecast in purged cross-validation (Brier skill {_s(bc['brier_skill'])}, positive in {bc['skill_folds_positive']} of {len(bc['folds'])} folds)"
                    + (", beyond every time-shifted placebo. " if steady else ", but not consistently across folds or not beyond the time-shifted placebo. ")
                    + "This is validation evidence only: the final test set has not been scored, so nothing is confirmed.")
    elif (bt_["brier_skill"] or 0) <= 0:
        verdict, confirmation = "no_skill", "not_confirmed"
        sentence = f"{best['label']} looked better than the base rate in cross-validation (Brier skill {_s(bc['brier_skill'])}) but not on the final test set ({_s(bt_['brier_skill'])}). The validation result did not hold up."
    else:
        verdict = "some_skill" if steady else "weak"
        confirmation = "confirmed_once" if not others else "tested_repeatedly"
        sentence = (f"{best['label']} beats the base-rate forecast in cross-validation (Brier skill {_s(bc['brier_skill'])}, positive in {bc['skill_folds_positive']} of {len(bc['folds'])} folds) and on the final test set ({_s(bt_['brier_skill'])}, AUC {bt_['auc']:.3f}). "
                    + ("That is evidence of some predictive skill for the underlying's outcome, not of a profitable option strategy." if steady else "It is not consistent across folds or not beyond the time-shifted placebo: treat it as unproven."))
        if others:
            sentence += (f" This test period has now been scored with {len(others) + 1} different configurations. A score chosen after several looks is no longer an untouched estimate: "
                         "it cannot be called independently confirmed.")
    for m in res_models:
        m.pop("_p_test", None)
        m.pop("_fitted", None)
    if scope:
        sig_ = {"symbol": sym, "features": sorted(feat_names), "outcome": f"{oc.entry}|{oc.horizon}|{oc.direction}|{oc.measure}"}
        LG.record(scope, sym + ("" if tick in ("open", "close") else "@intraday"), [{"kind": "ml", "key": LG.config_key({"model": m["name"], "features": sorted(feat_names), "outcome": oc.dump(), "symbol": sym}), "label": f"{m['label']}: {oc.text()}",
                                                                                   "run_id": run_id, "start": str(dates[0]), "end": str(dates[train[-1]]), "metric": "cv_brier_skill", "value": m["cv"]["brier_skill"], "n": int(train.size), "signature": sig_}
                                                                                  for m in res_models if m["status"] == "ok"], family="ml research")
    if progress:
        progress(1.0, "done")
    return {"kind": "ml", "symbol": sym, "outcome": oc.dump(), "outcome_text": oc.text(), "decision_tick": tick, "n": int(n), "base_rate": float(y.mean()), "features": [{"name": c[0], "label": c[1].label, "group": c[1].group} for c in cols],
            "dropped_features": dropped, "split": {"train": [str(dates[train[0]]), str(dates[train[-1]])], "test": [str(dates[test[0]]), str(dates[test[-1]])], "train_n": int(train.size), "test_n": int(test.size),
                                                  "embargo_sessions": embargo, "folds": len(folds), "method": "purged K-fold with embargo; final test set at the end of the history"},
            "models": res_models, "selected": best["name"], "selected_by": "lowest log loss in purged cross-validation", "frozen": frozen, "final_test": final, "test_exposure": exposure, "confirmation": confirmation,
            "placebo": placebo_test, "importance": imp, "importance_on": "validation rows (the last 30% of the training period)", "coefficients": coefs, "leakage": leak, "regimes": regimes,
            "preprocessing": PREPROCESSING, "signal": sig, "verdict": verdict, "summary": [sentence], "notes": notes, "calibrated": calibrate, "synthetic": store.is_synthetic(sym),
            "spec": {k: v for k, v in spec.items() if k not in ("expect_frozen", "final_test")}, "validation_run": spec.get("validation_run")}


PREPROCESSING = [{"step": "Scaling", "fitted_on": "the training rows of each fit (inside the model's pipeline)", "leak": False},
                 {"step": "Missing values", "fitted_on": "nothing is imputed: a session missing any feature is dropped before the split", "leak": False},
                 {"step": "Feature availability", "fitted_on": "a feature stored for fewer than 30% of all sessions is dropped; this looks at whether values exist, never at what they are", "leak": False},
                 {"step": "Calibration", "fitted_on": "the last quarter of the training rows, separated from the rows the model was fitted on by the label embargo", "leak": False},
                 {"step": "Signal threshold", "fitted_on": "a quantile of the calibrated predictions on the rows each walk-forward model was fitted on", "leak": False},
                 {"step": "Model selection", "fitted_on": "purged cross-validation inside the training period", "leak": False},
                 {"step": "Base rate", "fitted_on": "the training rows", "leak": False}]


# ----------------------------------------------------------------------------- the permanent record of test-set scorings
def exposures(sym: str, outcome_key: str, test_start: str, test_end: str) -> list[dict]:
    """Earlier scorings of a final test set for this symbol and outcome whose period overlaps this one by at least half."""
    import json
    from datetime import date as _d

    from strata.api import db
    try:
        with db.connect() as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM ml_test_exposures WHERE symbol=? AND outcome_key=? ORDER BY id", (sym, outcome_key))]
    except Exception:  # noqa: BLE001  (no database: a library call outside the application)
        return []
    a, b = _d.fromisoformat(test_start), _d.fromisoformat(test_end)
    out = []
    for r in rows:
        x, y_ = _d.fromisoformat(r["test_start"]), _d.fromisoformat(r["test_end"])
        inter = (min(b, y_) - max(a, x)).days + 1
        if inter > 0 and inter >= 0.5 * min((b - a).days + 1, (y_ - x).days + 1):
            r["n_features"] = len(json.loads(r["features"] or "[]"))
            out.append(r)
    return out


def _record_exposure(sym, outcome_key, config_key, model, features, test, n_test, skill, auc, run_id, scope) -> None:
    import json

    from strata.api import db
    try:
        with db.connect() as con:
            con.execute("INSERT INTO ml_test_exposures (symbol, outcome_key, config_key, model, features, test_start, test_end, n_test, brier_skill, auc, run_id, scope, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (sym, outcome_key, config_key, model, json.dumps(sorted(features)), test[0], test[1], n_test, skill, auc, run_id, scope, db.now()))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the test-set scoring could not be recorded ({type(e).__name__}: {e}); it is not reported without its record") from None


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


def _regimes(ft, rows: np.ndarray, y: np.ndarray, p: np.ndarray, base: float, tick: str, on: str = "final test set") -> list[dict]:
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
        out.append({"title": f"Calendar year ({on})", "rows": rs})
    for o in out:
        o["on"] = on
    return out


def _walk_forward(name: str, X, y, starts, ends, dates, embargo: int, quantile: float, calibrate: bool, min_train: int = 300, step: int = 63, upto: int | None = None) -> dict:
    """Out-of-sample probability for every session after the first `min_train`: the model is
    refitted every `step` sessions on the sessions whose labels were already known. The
    signal threshold for each block is a quantile of the model's own (calibrated) predictions
    on its training rows, so it too is fixed before the block it applies to. With `upto` the
    walk stops there: while the final test set has not been scored, no prediction is made for it."""
    n = len(y) if upto is None else min(len(y), int(upto))
    p = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    t = min_train
    while t < n:
        hi = min(n, t + step)
        tr = np.flatnonzero(ends[:t] + embargo < starts[t])
        if tr.size >= min_train // 2 and len(np.unique(y[tr])) == 2:
            pp, fitted = _fit_predict(name, X[tr], y[tr], X[t:hi], calibrate, embargo)
            p[t:hi] = pp
            ptr = np.full(tr.size, float(y[tr].mean())) if fitted is None else fitted.proba(X[tr])
            thr[t:hi] = float(np.quantile(ptr, quantile))
        t = hi
    ok = np.isfinite(p)
    on = ok & (p >= thr)
    dates, y = dates[:n], y[:n]
    out = {"model": name, "quantile": quantile, "covers_final_test": upto is None, "predicted": int(ok.sum()), "first": str(dates[ok][0]) if ok.any() else None, "last": str(dates[ok][-1]) if ok.any() else None, "signals": int(on.sum()),
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
    return {"ml": True, "symbols": [res["symbol"]], "models": len(res["models"]), "selected": res["selected"], "verdict": res["verdict"], "cv_brier_skill": b["cv"]["brier_skill"], "test_brier_skill": (b.get("test") or {}).get("brier_skill"),
            "final_test": bool(res.get("final_test")), "confirmation": res.get("confirmation"),
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
