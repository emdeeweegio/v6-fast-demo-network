import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from fed_statistics import compute_local_stats_frame


def _build_datasets() -> list:
    return [
        pd.DataFrame(
            {
                "age": [40, 50, 60, 70, 80, 90],
                "sex": [0, 1, 1, 0, 1, 0],
                "das28": [2.1, 3.8, 4.2, 5.5, 1.5, 3.0],
                "rf_positivity": [1, 1, 0, 1, 1, 0],
            }
        ),
        pd.DataFrame(
            {
                "age": [30, 35, 45, 55, 65, 75],
                "sex": [1, 1, 0, 0, 1, 1],
                "das28": [2.6, 2.9, 4.8, 5.2, 3.3, 4.1],
                "rf_positivity": [1, 0, 0, 1, 1, 1],
            }
        ),
    ]


def _run_central(statistics, filters=None, options=None) -> dict:
    datasets = _build_datasets()
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="fed_statistics",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {"statistics": statistics, "filters": filters, "options": options},
        },
        organizations=org_ids,
    )
    return client.wait_for_results(task.get("id"))[0]


def test_mean_std_minmax_nrows_match_pooled_computation() -> None:
    result = _run_central(
        {"age": ["mean", "std", "minmax", "nrows"]},
        options={"suppress_threshold": 0},
    )
    columns = result["columns"]

    pooled = pd.concat(_build_datasets(), ignore_index=True)
    assert columns["age"]["nrows"] == len(pooled)
    assert np.isclose(columns["age"]["mean"], pooled["age"].mean())
    assert np.isclose(columns["age"]["std"], pooled["age"].std(ddof=1))
    assert columns["age"]["minmax"] == {"min": float(pooled["age"].min()), "max": float(pooled["age"].max())}


def test_counts_match_pooled_value_counts() -> None:
    result = _run_central({"sex": ["counts"]}, options={"suppress_threshold": 0})
    counts = result["columns"]["sex"]["counts"]

    pooled = pd.concat(_build_datasets(), ignore_index=True)
    expected = pooled["sex"].value_counts()
    assert counts["0"]["lower"] == counts["0"]["upper"] == int(expected[0])
    assert counts["1"]["lower"] == counts["1"]["upper"] == int(expected[1])


def test_filters_and_binned_counts() -> None:
    result = _run_central(
        {"das28": {"stats": ["binned_counts"], "bins": [0, 3.2, 5.1, 10]}},
        filters={"rf_positivity": 1},
        options={"suppress_threshold": 0},
    )
    bins = result["columns"]["das28"]["binned_counts"]
    assert bins["0-3.2"]["lower"] >= 1
    assert bins["3.2-5.1"]["upper"] >= 1


def test_quantile_bootstrap_warning_present() -> None:
    result = _run_central(
        {"age": ["quantiles"]},
        options={"suppress_threshold": 0, "quantile_bootstrap_iterations": 10},
    )
    assert any("Quantile estimation" in warning for warning in result["warnings"])


def test_secondary_suppression_shifts_exactly_one_unsuppressed_category() -> None:
    # A rare category (count=1) triggers suppression; with only one other
    # category left unsuppressed, its true count would be re-identifiable
    # by subtraction from the (public) total unless also perturbed.
    df = pd.DataFrame({"group": ["a"] * 8 + ["b"] * 1})
    options = {"suppress_threshold": 5, "suppress_secondary": True, "suppress_key": "test-secret"}
    real_counts = compute_local_stats_frame(
        df, {"group": {"stats": ["counts"], "bins": None, "min_cap": None, "max_cap": None}}, {}, options
    )["group"]["counts"]

    assert real_counts["b"]["lower"] == 0  # rare category suppressed to the threshold band
    assert real_counts["b"]["upper"] == 5
    # "a" was the only unsuppressed (exact) category, which would let "b"'s
    # exact count be back-calculated from a known total. Secondary
    # suppression must turn "a" into a masked range too, not just a
    # different exact number.
    assert real_counts["a"]["lower"] != real_counts["a"]["upper"]
    assert real_counts["a"]["lower"] != 8
