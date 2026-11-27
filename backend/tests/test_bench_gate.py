"""The benchmark gate (`strata bench --check`): a workload is measured against the rest of its own run, the whole
suite against the calibration loop."""
from strata import bench

# benchmarks/baseline.json as committed in V5 (Darwin arm64): name -> (seconds, normalised); calibration 0.0707 s
BASE_CAL = 0.0707
BASE = {
    "cold_ingest": (1.6813, 23.781), "cached_load": (0.0091, 0.129), "single_backtest": (0.3075, 4.349),
    "sweep_12": (3.2052, 45.336), "feature_table_build": (0.027, 0.382), "hypothesis_plan": (0.8363, 11.829),
    "condition_scan": (0.3973, 5.62), "study_engine_one": (0.1261, 1.784), "vectorised_one": (0.0001, 0.001),
    "multi_symbol_study": (0.2614, 3.697), "bulk_import": (0.0934, 1.321), "graph_plan": (0.0007, 0.01),
    "graph_run": (0.0839, 1.187), "pattern_explore": (0.0336, 0.475), "ide_debug_session": (0.0054, 0.076),
    "ml_study": (0.5356, 7.576), "concurrent_jobs_3": (1.0402, 14.713),
}
# GitHub's runner on the V9 push (91e254e, run 46066170926): it ran the calibration loop in 0.061 s, where the other
# runs took 0.118 to 0.177 s, while the workloads ran at about their usual speed for such a runner
CI_CAL = 0.061
CI_SECONDS = {
    "cold_ingest": 4.232, "cached_load": 0.020, "single_backtest": 0.806, "sweep_12": 6.500, "feature_table_build": 0.043,
    "hypothesis_plan": 0.905, "condition_scan": 0.632, "study_engine_one": 0.320, "vectorised_one": 0.0001,
    "multi_symbol_study": 0.555, "bulk_import": 0.198, "graph_plan": 0.001, "graph_run": 0.133, "pattern_explore": 0.059,
    "ide_debug_session": 0.011, "ml_study": 0.598, "concurrent_jobs_3": 1.889,
}


def baseline(names=None) -> dict:
    return {"calibration_seconds": BASE_CAL, "workloads": [{"name": n, "seconds": s, "normalised": z} for n, (s, z) in BASE.items() if names is None or n in names]}


def run(seconds: dict, cal: float) -> dict:
    return {"calibration_seconds": cal, "workloads": [{"name": n, "seconds": s, "normalised": round(s / cal, 3)} for n, s in seconds.items()]}


def test_a_runner_whose_calibration_loop_is_unusually_fast_is_not_a_regression():
    """The V9 CI failure: against the calibration loop alone single_backtest was 3.04x its baseline, over CI's tolerance
    of 3.0, although the same commit timed on the baseline's machine was as fast as V8. Against the rest of its own run
    it is 1.36x: the runner, not the backtest."""
    a = bench.assess(run(CI_SECONDS, CI_CAL), baseline(), tolerance=3.0)
    by = {w["name"]: w for w in a["workloads"]}
    assert by["single_backtest"]["ratio"] > 3.0  # what the old gate failed on
    assert 1.3 < by["single_backtest"]["relative"] < 1.4
    assert not [w["name"] for w in a["workloads"] if w["regressed"]]
    assert a["basis"] == "the run's median" and 1.8 < a["machine"] < 2.1
    assert 2.0 < a["suite_ratio"] < 3.0 and not a["suite_regressed"]
    assert bench.compare(run(CI_SECONDS, CI_CAL), baseline(), tolerance=3.0) == a["workloads"]


def test_one_workload_slowing_down_more_than_the_rest_of_its_run_is_a_regression():
    slow = dict(CI_SECONDS, single_backtest=CI_SECONDS["single_backtest"] * 2.5)
    a = bench.assess(run(slow, CI_CAL), baseline(), tolerance=3.0)
    assert [w["name"] for w in a["workloads"] if w["regressed"]] == ["single_backtest"]
    assert next(w for w in a["workloads"] if w["name"] == "single_backtest")["relative"] > 3.0


def test_a_workload_below_the_noise_floor_is_never_gated():
    noisy = dict(CI_SECONDS, ide_debug_session=0.04)  # 7x its 0.0054 s baseline, but both under 0.05 s
    a = bench.assess(run(noisy, CI_CAL), baseline(), tolerance=3.0)
    w = next(w for w in a["workloads"] if w["name"] == "ide_debug_session")
    assert w["relative"] > 3.0 and not w["gated"] and not w["regressed"]


def test_the_whole_suite_slowing_down_together_is_caught_by_the_calibration_loop():
    """Every workload twice as slow on the baseline's own machine: no workload stands out from the run, but the suite
    is 2x its baseline after the calibration loop, over the local tolerance of 1.75."""
    a = bench.assess(run({n: s * 2 for n, (s, _) in BASE.items()}, BASE_CAL), baseline(), tolerance=1.75)
    assert not [w["name"] for w in a["workloads"] if w["regressed"]]
    assert a["suite_ratio"] == 2.0 and a["suite_regressed"]


def test_with_too_few_workloads_to_measure_the_machine_the_calibration_loop_decides_alone():
    names = ("single_backtest", "sweep_12", "condition_scan")
    seconds = {n: BASE[n][0] for n in names} | {"single_backtest": BASE["single_backtest"][0] * 2}
    a = bench.assess(run(seconds, BASE_CAL), baseline(names), tolerance=1.75)
    assert a["basis"] == "the calibration loop" and a["machine"] is None and not a["suite_regressed"]
    by = {w["name"]: w for w in a["workloads"]}
    assert by["single_backtest"]["regressed"] and by["single_backtest"]["relative"] == by["single_backtest"]["ratio"] == 2.0
    assert not by["sweep_12"]["regressed"]
