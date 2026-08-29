import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from kaplan_meier import add_noise_to_event_times


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80],
            "deadstatus.event": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    )


def _call_partial(method: str, df: pd.DataFrame, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="kaplan_meier")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def test_none_noise_type_returns_frame_unchanged() -> None:
    df = _sample_df()
    result = add_noise_to_event_times(df, "Survival.time", noise_type="none")
    pd.testing.assert_frame_equal(result, df)


def test_default_run_is_unaffected_by_noise_plumbing() -> None:
    # Backward-compat: calling get_time_range/compute_events with no noise
    # kwargs at all must reproduce the pre-noise-injection behavior exactly.
    df = _sample_df()
    range_result = _call_partial(
        "get_time_range", df, time_col="Survival.time", event_col="deadstatus.event"
    )
    assert range_result == {"n": 8, "max_time": 80.0}

    events_result = _call_partial(
        "compute_events",
        df,
        time_col="Survival.time",
        event_col="deadstatus.event",
        time_steps=[0, 30, 60],
    )
    assert events_result == {"n_risk": [8, 6, 3], "n_events": [2, 1, 2]}


def test_gaussian_noise_without_snr_raises() -> None:
    df = _sample_df()
    with pytest.raises(ValueError, match="snr"):
        add_noise_to_event_times(df, "Survival.time", noise_type="gaussian", snr=None)


def test_unknown_noise_type_raises() -> None:
    df = _sample_df()
    with pytest.raises(ValueError, match="Unknown noise type"):
        add_noise_to_event_times(df, "Survival.time", noise_type="bogus")


def test_gaussian_noise_changes_values_and_stays_non_negative() -> None:
    df = _sample_df()
    noised = add_noise_to_event_times(df, "Survival.time", noise_type="gaussian", snr=2.0, random_seed=1)
    assert not noised["Survival.time"].equals(df["Survival.time"])
    assert (noised["Survival.time"] >= 0).all()


def test_poisson_noise_keeps_zeros_at_zero() -> None:
    df = pd.DataFrame({"Survival.time": [0, 0, 50, 100]})
    noised = add_noise_to_event_times(df, "Survival.time", noise_type="poisson", random_seed=1)
    assert noised["Survival.time"].iloc[0] == 0
    assert noised["Survival.time"].iloc[1] == 0


def test_same_seed_produces_identical_noise_across_independent_calls() -> None:
    # The property the two-phase protocol actually depends on: get_time_range
    # and compute_events are separate task invocations that must apply
    # identical noise to the same node's data, or phase 1's computed time
    # range won't bound the noised times used in phase 2.
    df = _sample_df()

    first = add_noise_to_event_times(df, "Survival.time", noise_type="gaussian", snr=1.5, random_seed=42)
    second = add_noise_to_event_times(df, "Survival.time", noise_type="gaussian", snr=1.5, random_seed=42)
    pd.testing.assert_frame_equal(first, second)

    first_poisson = add_noise_to_event_times(df, "Survival.time", noise_type="poisson", random_seed=7)
    second_poisson = add_noise_to_event_times(df, "Survival.time", noise_type="poisson", random_seed=7)
    pd.testing.assert_frame_equal(first_poisson, second_poisson)


def test_compute_events_with_noise_stays_consistent_with_get_time_range() -> None:
    # End-to-end version of the same property, through the actual @data
    # entry points: the max_time phase 1 reports must still bound the noised
    # times phase 2 computes on, given the same random_seed.
    df = _sample_df()
    noise_kwargs = {"noise_type": "gaussian", "snr": 3.0, "random_seed": 99}

    range_result = _call_partial(
        "get_time_range", df, time_col="Survival.time", event_col="deadstatus.event", **noise_kwargs
    )
    time_steps = list(range(0, int(range_result["max_time"]) + 30, 30))

    events_result = _call_partial(
        "compute_events",
        df,
        time_col="Survival.time",
        event_col="deadstatus.event",
        time_steps=time_steps,
        **noise_kwargs,
    )
    assert sum(events_result["n_events"]) <= range_result["n"]
    assert events_result["n_risk"][0] == range_result["n"]
