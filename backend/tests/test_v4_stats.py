"""V4 statistics: intervals and tests that stay calibrated when observations are dependent,
multiple-hypothesis corrections, purged cross-validation and overlapping holding periods."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from strata.research import robust as R


def weekdays(n: int) -> list[date]:
    out, d = [], date(2000, 1, 3)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ----------------------------------------------------------------------------- calibration
def test_cluster_interval_keeps_its_coverage_when_hits_arrive_in_runs_and_wilson_does_not():
    rng = np.random.default_rng(1)
    days = weekdays(600)
    w = c = 0
    T = 120
    for k in range(T):
        x = R.simulate_clustered_binary(600, 0.3, 0.6, rng)
        r = R.cluster_prop_ci(x, days, "week", reps=400, seed=k)
        w += r["wilson"][0] <= 0.3 <= r["wilson"][1]
        c += r["ci"][0] <= 0.3 <= r["ci"][1]
    assert w / T < 0.80, "the independence interval should visibly under-cover on clustered data"
    assert c / T >= 0.87, f"cluster interval coverage {c / T:.2f} is too far below 95%"
    assert r["design_effect"] > 1.5 and r["n_eff"] < 600


def test_on_independent_data_the_cluster_interval_is_not_narrower_than_wilson():
    rng = np.random.default_rng(2)
    x = (rng.random(500) < 0.4).astype(float)
    r = R.cluster_prop_ci(x, weekdays(500), "week", reps=400)
    assert r["ci"][0] <= r["wilson"][0] + 1e-12 and r["ci"][1] >= r["wilson"][1] - 1e-12


def test_block_bootstrap_covers_the_mean_of_a_serially_correlated_series():
    rng = np.random.default_rng(3)
    iid = blk = 0
    T = 120
    for k in range(T):
        e = rng.standard_normal(500)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.5 * x[i - 1] + e[i]
        r = R.bootstrap_ci(x, reps=400, seed=k)
        iid += r["iid_ci"][0] <= 0 <= r["iid_ci"][1]
        blk += r["ci"][0] <= 0 <= r["ci"][1]
    assert iid / T < 0.82 and blk / T >= 0.85
    assert r["block"] > 1 and r["n_eff"] < 400


def test_moving_and_stationary_blocks_keep_neighbours_together():
    rng = np.random.default_rng(4)
    mb = R.moving_block_indices(100, 10, 5, rng)
    assert mb.shape == (5, 100) and ((np.diff(mb, axis=1) % 100) == 1).mean() > 0.85
    sb = R.stationary_indices(1000, 20, 5, rng)
    runs = ((np.diff(sb, axis=1) % 1000) == 1).mean()
    assert 0.90 < runs < 0.99  # mean block length about 20
    assert R.block_length(rng.standard_normal(500)) == 1  # nothing to preserve in white noise


def test_a_clustered_null_is_not_called_significant_more_often_than_it_should_be():
    rng = np.random.default_rng(5)
    days = weekdays(1500)
    fp_iid = fp_cl = 0
    T = 100
    for k in range(T):
        y = R.simulate_clustered_binary(1500, 0.3, 0.5, rng)
        m = R.simulate_clustered_binary(1500, 0.1, 0.5, rng).astype(bool)
        r = R.lift_test(y[m], [d for d, a in zip(days, m) if a], y[~m], [d for d, a in zip(days, m) if not a], reps=400, seed=k)
        fp_iid += (r["p_iid"] if r["p_iid"] is not None else 1) < 0.05
        fp_cl += (r["p_value"] if r["p_value"] is not None else 1) < 0.05
    assert fp_cl / T <= 0.12, f"cluster test rejects a true null {fp_cl / T:.0%} of the time"
    assert fp_cl <= fp_iid


def test_a_real_effect_is_found_and_small_p_values_keep_their_resolution():
    rng = np.random.default_rng(6)
    days = weekdays(3000)
    m = rng.random(3000) < 0.15
    y = (rng.random(3000) < np.where(m, 0.75, 0.40)).astype(float)
    r = R.lift_test(y[m], [d for d, a in zip(days, m) if a], y[~m], [d for d, a in zip(days, m) if not a], reps=500)
    assert 0.25 < r["lift"] < 0.45 and r["ci"][0] > 0.2
    assert r["p_value"] < 1 / 501, "a resampling p-value must not be floored at 1/(draws+1) when many hypotheses are corrected for"
    tiny = R.lift_test([1.0] * 4, days[:4], y[~m][:200], days[10:210])
    assert tiny["p_value"] is None, "four observations cannot support a dependence-robust p-value"


def test_shift_test_separates_aligned_from_independent_series():
    rng = np.random.default_rng(7)
    m = rng.random(3000) < 0.1
    assert R.shift_test(m, rng.standard_normal(3000))["p_value"] > 0.05
    y = rng.standard_normal(3000) + np.where(m, 0.8, 0.0)
    assert R.shift_test(m, y)["p_value"] < 0.01


# ----------------------------------------------------------------------------- multiple hypotheses
def test_corrections_are_ordered_and_count_hypotheses_that_are_not_in_the_list():
    p = [0.001, 0.02, 0.04, None, 0.5]
    holm, bh, by, bon = (R.adjust(p, m) for m in ("holm", "bh", "by", "bonferroni"))
    assert holm[3] is None and all(a >= b for a, b in zip([x for x in holm if x is not None], [x for x in p if x is not None]))
    assert all(b1 <= h for b1, h in zip([x for x in bh if x is not None], [x for x in holm if x is not None]))
    assert all(y >= b1 for y, b1 in zip([x for x in by if x is not None], [x for x in bh if x is not None]))
    assert bon[0] == 0.004 and holm[0] == 0.004
    more = R.adjust(p, "holm", m=50)  # 50 configurations were tried in the session, only four are shown
    assert more[0] == 0.05 and all(a >= b for a, b in zip([x for x in more if x is not None], [x for x in holm if x is not None]))


def test_evidence_grade_never_gets_past_insufficient_on_a_thin_sample():
    assert R.grade(12, 10, 1e-9)["level"] == "insufficient"
    assert R.grade(200, 6, 1e-9)["level"] == "insufficient"  # 200 observations in six weeks
    assert R.grade(200, 80, 0.2)["level"] == "none"
    assert R.grade(200, 80, 0.001)["level"] == "moderate"  # significant, but nothing confirmed it
    assert R.grade(200, 80, 0.001, holdout="confirmed")["level"] == "strong"
    assert R.grade(200, 80, 0.001, holdout="confirmed", contaminated=True)["level"] == "moderate"
    assert R.grade(60, 40, 0.001, holdout="confirmed")["level"] == "moderate"  # fewer than 100 observations is never "strong"


# ----------------------------------------------------------------------------- chronological validation
def test_purged_kfold_never_trains_on_an_observation_that_overlaps_the_test_block():
    rng = np.random.default_rng(8)
    s = np.sort(rng.choice(2000, 300, replace=False))
    e = s + rng.integers(0, 15, 300)
    folds = R.purged_kfold(s, e, k=5, embargo=10)
    assert len(folds) == 5 and sum(len(f["test"]) for f in folds) == 300
    for f in folds:
        t0, t1 = f["span"]
        tr = f["train"]
        assert not ((s[tr] <= t1) & (e[tr] >= t0)).any(), "a training observation overlaps the test span"
        assert not ((s[tr] > t1) & (s[tr] <= t1 + 10)).any(), "the embargo was not applied"
        assert set(tr).isdisjoint(f["test"])
    assert any(f["purged"] for f in folds) and any(f["embargoed"] for f in folds)


def test_walk_forward_windows_do_not_touch_and_leave_the_embargo():
    w = R.walk_forward(300, 120, 30, embargo=5)
    assert w[0] == {"train": [0, 119], "test": [125, 154]} and all(x["test"][0] - x["train"][1] == 6 for x in w)
    assert R.walk_forward(300, 120, 30, anchored=True)[2]["train"][0] == 0


def test_overlapping_holding_periods_count_for_less_than_their_number():
    assert np.allclose(R.uniqueness([0, 10, 20], [4, 14, 24]), 1.0)
    u = R.uniqueness([0, 0, 0, 0], [9, 9, 9, 9])
    assert np.allclose(u, 0.25)
    o = R.overlap_summary([0, 1, 2, 3, 50], [9, 10, 11, 12, 59])
    assert o["overlapping"] == 4 and o["max_concurrent"] == 4 and 2.0 < o["effective_n"] < 3.2
    # overlapping 10-session returns are serially correlated: the HAC standard error is larger than the naive one
    rng = np.random.default_rng(9)
    r = rng.standard_normal(1010)
    x = np.array([r[i:i + 10].sum() for i in range(1000)])
    assert R.hac_se(x) > 1.8 * x.std(ddof=1) / np.sqrt(1000) and R.effective_n(x) < 400


def test_selection_cv_shows_that_picking_the_best_of_noise_does_not_carry_over():
    rng = np.random.default_rng(10)
    n, c = 600, 25
    M = rng.standard_normal((n, c))
    out = R.selection_cv(M, np.arange(n), np.arange(n), k=5)
    assert out["k"] == 5 and out["mean_train"] > out["mean_test"] + 0.05, "the chosen configuration's training mean should not survive out of sample"
    M[:, 3] += 0.6  # one configuration with a real effect is chosen every time and holds
    good = R.selection_cv(M, np.arange(n), np.arange(n), k=5)
    assert good["distinct_choices"] == 1 and good["share_below_median"] == 0 and good["mean_test"] > 0.4


# ----------------------------------------------------------------------------- paired comparisons and regimes
def test_paired_comparison_uses_only_shared_sessions():
    days = [d.isoformat() for d in weekdays(80)]
    rng = np.random.default_rng(11)
    base = rng.standard_normal(80) * 50
    a = {d: float(v + 20) for d, v in zip(days[:60], base[:60])}
    b = {d: float(v) for d, v in zip(days[20:], base[20:])}
    r = R.paired_compare(a, b)
    assert r["pairs"] == 40 and r["only_a"] == 20 and r["only_b"] == 20
    assert abs(r["mean_diff"] - 20) < 1e-9 and r["a_better"] == 40 and r["sign_p"] < 1e-6 and r["ci"][0] > 0
    assert R.paired_compare({"2024-01-02": 1.0}, {"2024-01-03": 2.0})["pairs"] == 0
    m = R.mcnemar({d: True for d in days[:10]}, {d: i < 2 for i, d in enumerate(days[:10])})
    assert m["a_only"] == 8 and m["b_only"] == 0 and m["p_value"] < 0.01


def test_regime_table_flags_thin_regimes_and_tests_heterogeneity():
    rng = np.random.default_rng(12)
    days = weekdays(900)
    lab = ["calm"] * 600 + ["stressed"] * 290 + ["panic"] * 10
    y = (rng.random(900) < np.where(np.arange(900) < 600, 0.5, 0.7)).astype(float)
    t = R.regime_table(y, lab, days, ["calm", "stressed", "panic"], binary=True)
    rows = {r["regime"]: r for r in t["rows"]}
    assert rows["panic"]["thin"] and not rows["calm"]["thin"] and t["heterogeneity_p"] < 0.01
    assert rows["stressed"]["estimate"] > rows["calm"]["estimate"] and rows["calm"]["ci"][0] < rows["calm"]["estimate"] < rows["calm"]["ci"][1]
