import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _call_partial(df: pd.DataFrame, method: str, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="kaplan_meier")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="kaplan_meier",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _classical_km(times: list, events: list, query_times: list) -> list:
    """Independent product-limit KM reference (exact-time, not fixed-width
    binning) used to cross-check the algorithm's discrete life-table method.
    Survival at a query time reflects events strictly before it, matching
    the convention `central()` uses (survival[i] excludes deaths occurring
    exactly at time_steps[i], which only show up in survival[i+1]).
    """
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    event_times = sorted(set(times_arr[events_arr == 1].tolist()))

    survival = 1.0
    var_sum = 0.0
    idx = 0
    results = []
    for q in query_times:
        while idx < len(event_times) and event_times[idx] < q:
            t = event_times[idx]
            n = int((times_arr >= t).sum())
            d = int(((times_arr == t) & (events_arr == 1)).sum())
            if n > 0:
                survival *= 1 - d / n
            if n > d > 0:
                var_sum += d / (n * (n - d))
            idx += 1
        se = survival * np.sqrt(var_sum)
        results.append(
            {
                "survival": survival,
                "ci_lower": max(survival - 1.96 * se, 0.0),
                "ci_upper": min(survival + 1.96 * se, 1.0),
            }
        )
    return results


# ── get_time_range ────────────────────────────────────────────────────────────


def test_get_time_range_basic() -> None:
    df = pd.DataFrame({"Survival.time": [10, 20, 30], "deadstatus.event": [1, 0, 1]})
    result = _call_partial(df, "get_time_range", time_col="Survival.time", event_col="deadstatus.event")
    assert result == {"n": 3, "max_time": 30.0}


def test_get_time_range_drops_nan_rows() -> None:
    df = pd.DataFrame(
        {"Survival.time": [10, None, 30, 40], "deadstatus.event": [1, 1, None, 0]}
    )
    result = _call_partial(df, "get_time_range", time_col="Survival.time", event_col="deadstatus.event")
    # Rows 2 and 3 each have a NaN in one required column; only rows 1 and 4 survive.
    assert result == {"n": 2, "max_time": 40.0}


def test_get_time_range_all_nan_returns_zero() -> None:
    df = pd.DataFrame({"Survival.time": [None, None], "deadstatus.event": [None, None]})
    result = _call_partial(df, "get_time_range", time_col="Survival.time", event_col="deadstatus.event")
    assert result == {"n": 0, "max_time": 0.0}


# ── compute_events ─────────────────────────────────────────────────────────────


def test_compute_events_basic_counts() -> None:
    df = pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80],
            "deadstatus.event": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    )
    result = _call_partial(
        df,
        "compute_events",
        time_col="Survival.time",
        event_col="deadstatus.event",
        time_steps=[0, 30, 60],
    )
    assert result == {"n_risk": [8, 6, 3], "n_events": [2, 1, 2]}


def test_compute_events_all_censored_reports_zero_events() -> None:
    df = pd.DataFrame({"Survival.time": [10, 20, 30], "deadstatus.event": [0, 0, 0]})
    result = _call_partial(
        df, "compute_events", time_col="Survival.time", event_col="deadstatus.event", time_steps=[0, 15]
    )
    assert result["n_events"] == [0, 0]
    assert result["n_risk"] == [3, 2]  # bucket 2 is [15, 30): times 20 and 30 are >=15


def test_compute_events_last_bucket_extends_by_step_size() -> None:
    # Only one explicit time_step is given; the last bucket must still span
    # [t, t + step_size) using time_steps[0] as the implied step size.
    df = pd.DataFrame({"Survival.time": [5, 25], "deadstatus.event": [1, 1]})
    result = _call_partial(
        df, "compute_events", time_col="Survival.time", event_col="deadstatus.event", time_steps=[10]
    )
    # bucket is [10, 20): neither event (5, 25) falls inside it.
    assert result == {"n_risk": [1], "n_events": [0]}


# ── central: pipeline-level behavior ──────────────────────────────────────────


def test_central_returns_error_when_no_usable_data() -> None:
    df = pd.DataFrame({"Survival.time": [None], "deadstatus.event": [None]})
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event")
    assert result == {"error": "No usable data across any organization"}


def test_central_basic_shape_and_aggregation() -> None:
    df = pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80],
            "deadstatus.event": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    )
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=30)

    assert result["n_patients"] == 8
    assert result["n_events"] == 5
    assert result["noise_type"] == "none"
    # time_steps = range(0, global_max + step_days, step_days) = [0, 30, 60, 90]
    assert len(result["curve"]) == 4
    for point in result["curve"]:
        assert set(point) == {"time", "n_risk", "n_events", "survival", "ci_lower", "ci_upper"}


def test_central_single_organization_still_works() -> None:
    df = pd.DataFrame({"Survival.time": [10, 20, 30], "deadstatus.event": [1, 0, 1]})
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=10)
    assert result["n_patients"] == 3
    assert "error" not in result


def test_central_federated_result_matches_single_pooled_run() -> None:
    # The property the multi-node protocol actually depends on: splitting the
    # same rows across several organizations and summing local at-risk/event
    # counts must reproduce exactly the result of running on the pooled data
    # as a single organization (deterministic, no approximation involved).
    rng = np.random.default_rng(0)
    n = 30
    pooled = pd.DataFrame(
        {
            "Survival.time": rng.integers(1, 400, size=n),
            "deadstatus.event": rng.integers(0, 2, size=n),
        }
    )
    shuffled = pooled.sample(frac=1, random_state=1).reset_index(drop=True)
    boundaries = [0, 10, 20, n]
    split = [shuffled.iloc[boundaries[i]:boundaries[i + 1]] for i in range(3)]

    pooled_result = _run_central([pooled], time_col="Survival.time", event_col="deadstatus.event", step_days=20)
    split_result = _run_central(list(split), time_col="Survival.time", event_col="deadstatus.event", step_days=20)

    assert split_result["n_patients"] == pooled_result["n_patients"]
    assert split_result["n_events"] == pooled_result["n_events"]
    assert split_result["curve"] == pooled_result["curve"]


def test_central_survival_matches_classical_km_reference() -> None:
    # With step_days=1 and distinct-per-bin integer times, each bucket
    # contains exactly one observed time value, so the discrete life-table
    # recursion central() uses coincides exactly with the classical
    # product-limit estimator (recomputing the risk set at every observed
    # time) — this cross-checks the actual survival/Greenwood formulas
    # against an independently written reference, not just against
    # themselves.
    times =  [3, 5, 5, 8, 8, 8, 12, 15, 15, 20, 24, 30]
    events = [1, 1, 0, 1, 1, 0, 1,  1,  0,  1,  0,  1]
    df = pd.DataFrame({"Survival.time": times, "deadstatus.event": events})

    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=1)
    query_times = [point["time"] for point in result["curve"]]
    reference = _classical_km(times, events, query_times)

    for point, expected in zip(result["curve"], reference):
        assert np.isclose(point["survival"], expected["survival"], atol=1e-9)
        assert np.isclose(point["ci_lower"], expected["ci_lower"], atol=1e-9)
        assert np.isclose(point["ci_upper"], expected["ci_upper"], atol=1e-9)


def test_central_survival_is_monotonically_non_increasing() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "Survival.time": rng.integers(1, 200, size=40),
            "deadstatus.event": rng.integers(0, 2, size=40),
        }
    )
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=10)
    survival = [point["survival"] for point in result["curve"]]
    assert all(a >= b for a, b in zip(survival, survival[1:]))


def test_central_at_risk_counts_are_non_increasing() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "Survival.time": rng.integers(1, 200, size=40),
            "deadstatus.event": rng.integers(0, 2, size=40),
        }
    )
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=10)
    n_risk = [point["n_risk"] for point in result["curve"]]
    assert all(a >= b for a, b in zip(n_risk, n_risk[1:]))


def test_central_step_days_changes_granularity_not_totals() -> None:
    df = pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80],
            "deadstatus.event": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    )
    coarse = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=60)
    fine = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=10)

    assert len(fine["curve"]) > len(coarse["curve"])
    assert fine["n_patients"] == coarse["n_patients"] == 8
    assert fine["n_events"] == coarse["n_events"] == 5


def test_central_ci_bounds_are_clamped_to_valid_probability_range() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "Survival.time": rng.integers(1, 100, size=60),
            "deadstatus.event": rng.integers(0, 2, size=60),
        }
    )
    result = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=5)
    for point in result["curve"]:
        assert 0.0 <= point["ci_lower"] <= point["survival"] <= point["ci_upper"] <= 1.0
