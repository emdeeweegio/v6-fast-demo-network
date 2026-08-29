import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _call_partial(df: pd.DataFrame, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="average")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "partial", "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="average",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


# ── partial ────────────────────────────────────────────────────────────────────


def test_partial_basic_sum_and_count() -> None:
    df = pd.DataFrame({"age": [10, 20, 30]})
    result = _call_partial(df, column="age")
    assert result == {"sum": 60.0, "count": 3}


def test_partial_skips_nan_values() -> None:
    df = pd.DataFrame({"age": [10, None, 30, None]})
    result = _call_partial(df, column="age")
    assert result == {"sum": 40.0, "count": 2}


def test_partial_all_nan_returns_zero_sum_and_count() -> None:
    df = pd.DataFrame({"age": [None, None]})
    result = _call_partial(df, column="age")
    assert result == {"sum": 0.0, "count": 0}


def test_partial_empty_dataframe() -> None:
    df = pd.DataFrame({"age": []})
    result = _call_partial(df, column="age")
    assert result == {"sum": 0.0, "count": 0}


# ── central ────────────────────────────────────────────────────────────────────


def test_central_returns_none_when_total_count_is_zero() -> None:
    df = pd.DataFrame({"age": [None, None]})
    result = _run_central([df], column="age")
    assert result == {"average": None, "variable": "age", "n": 0}


def test_central_basic_shape() -> None:
    df = pd.DataFrame({"age": [10, 20, 30, 40]})
    result = _run_central([df], column="age")
    assert result == {"average": 25.0, "variable": "age", "n": 4}


def test_central_weights_by_sample_count_not_naive_node_average() -> None:
    # The property that actually matters: averaging the per-node averages
    # directly (naive average-of-averages) would be wrong when node sizes
    # differ. A large node of all 100s and a tiny node of all 0s must pull
    # the global average toward 100, not sit at the midpoint (50).
    big_node = pd.DataFrame({"age": [100] * 99})
    small_node = pd.DataFrame({"age": [0]})

    result = _run_central([big_node, small_node], column="age")

    naive_average_of_node_means = (100 + 0) / 2  # what a (wrong) unweighted average would give
    assert result["average"] != naive_average_of_node_means
    assert np.isclose(result["average"], 99.0)  # (99*100 + 1*0) / 100
    assert result["n"] == 100


def test_central_matches_single_pooled_run() -> None:
    # Splitting the same rows across several organizations and summing local
    # sum/count must reproduce exactly the result of running on the pooled
    # data as a single organization (deterministic, no approximation).
    rng = np.random.default_rng(0)
    pooled = pd.DataFrame({"age": rng.normal(60, 15, size=37)})
    shuffled = pooled.sample(frac=1, random_state=1).reset_index(drop=True)
    boundaries = [0, 10, 24, 37]
    split = [shuffled.iloc[boundaries[i] : boundaries[i + 1]] for i in range(3)]

    pooled_result = _run_central([pooled], column="age")
    split_result = _run_central(split, column="age")

    assert split_result["n"] == pooled_result["n"]
    assert np.isclose(split_result["average"], pooled_result["average"])


def test_central_handles_mix_of_usable_and_all_nan_nodes() -> None:
    usable = pd.DataFrame({"age": [10, 20, 30]})
    all_nan = pd.DataFrame({"age": [None, None]})

    result = _run_central([usable, all_nan], column="age")

    assert result["n"] == 3  # the all-NaN node contributes nothing, not an error
    assert np.isclose(result["average"], 20.0)


def test_central_default_column_matches_lung1_data() -> None:
    # Regression guard: AVERAGE_VAR's default ("age") must exist and be
    # numeric in the default LUNG1 node data this algorithm ships against.
    from pathlib import Path

    lung1_dir = Path(__file__).resolve().parents[2] / "data" / "lung1"
    df = pd.read_csv(lung1_dir / "alpha.csv")
    result = _run_central([df], column="age")
    assert result["n"] > 0
    assert result["average"] is not None
