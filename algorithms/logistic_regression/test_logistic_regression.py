from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

import logistic_regression
from logistic_regression import (
    BATCH_RATIO,
    FEATURE_COLS,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_TEST_RATIO,
    LogisticRegressionModel,
    _sd_from_list,
    _sd_to_list,
)

LUNG1_DIR = Path(__file__).resolve().parents[2] / "data" / "lung1"


def _call_partial(df: pd.DataFrame, method: str, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="logistic_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _separable_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    return pd.DataFrame({"x": x, "y": y})


def test_model_init_is_reproducible_given_the_module_seed() -> None:
    # Regression test for the missing torch.manual_seed() bug: central() seeds
    # once before constructing the initial global model, so two models built
    # the same way after seeding must start with identical weights.
    torch.manual_seed(RANDOM_SEED)
    first = LogisticRegressionModel(n_features=4)

    torch.manual_seed(RANDOM_SEED)
    second = LogisticRegressionModel(n_features=4)

    assert torch.equal(first.linear.weight, second.linear.weight)
    assert torch.equal(first.linear.bias, second.linear.bias)


def test_model_init_without_seeding_is_not_reproducible() -> None:
    # Sanity check that the test above is actually exercising something:
    # without calling torch.manual_seed(), two models should (with
    # overwhelming probability) NOT match, confirming the seed call is what
    # makes the previous test pass rather than coincidence.
    first = LogisticRegressionModel(n_features=4)
    second = LogisticRegressionModel(n_features=4)
    assert not torch.equal(first.linear.weight, second.linear.weight)


def test_default_feature_cols_do_not_contain_the_known_leaky_column() -> None:
    # survival_1y was confirmed to be a deterministic function of
    # Survival.time (>= 365 days) and 99.4% correlated with the target
    # deadstatus.event in the LUNG1 data -- including it let the model reach
    # near-perfect accuracy without learning anything federated.
    assert "survival_1y" not in FEATURE_COLS


def test_no_default_feature_is_near_perfectly_correlated_with_the_target() -> None:
    # Broader guard-rail than the exact-name check above: if a future edit
    # swaps in some other column that's nearly a proxy for the target, this
    # should catch it too, using the real LUNG1 data all four nodes serve.
    df = pd.concat(
        [pd.read_csv(LUNG1_DIR / f"{name}.csv") for name in ("alpha", "beta", "gamma", "theta")],
        ignore_index=True,
    )
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    for col in FEATURE_COLS:
        # Point-biserial correlation between a (numeric) feature and the
        # binary target; a leaky proxy shows up as |corr| very close to 1.
        corr = df[col].astype(float).corr(df[TARGET_COL].astype(float))
        assert abs(corr) < 0.9, f"Feature '{col}' is suspiciously correlated with the target (corr={corr:.3f})"


# ── compute_stats ──────────────────────────────────────────────────────────────


def test_compute_stats_basic_sum_and_count() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0, 1, 0, 1, 0]})
    result = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=1.0, seed=1)
    assert result["n"] == 5
    assert result["sum"] == pytest.approx([15.0])
    assert result["sum_sq"] == pytest.approx([55.0])


def test_compute_stats_drops_rows_with_missing_feature_or_target() -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, None]})
    result = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=1.0, seed=1)
    assert result["n"] == 1
    assert result["sum"] == pytest.approx([1.0])


def test_compute_stats_all_nan_returns_empty() -> None:
    df = pd.DataFrame({"x": [None, None], "y": [None, None]})
    result = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y")
    assert result == {"n": 0, "sum": [0.0], "sum_sq": [0.0]}


def test_compute_stats_train_ratio_is_seeded_deterministically() -> None:
    df = pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10})
    first = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=0.5, seed=7)
    second = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=0.5, seed=7)
    assert first == second


# ── partial (local training) ──────────────────────────────────────────────────


def _zeroed_state(n_features: int) -> dict:
    torch.manual_seed(0)
    model = LogisticRegressionModel(n_features)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    return _sd_to_list(model.state_dict())


def test_partial_returns_input_state_unchanged_when_no_usable_rows() -> None:
    df = pd.DataFrame({"x": [None, None], "y": [None, None]})
    state = _zeroed_state(1)
    result = _call_partial(
        df,
        "partial",
        feature_cols=["x"],
        target_col="y",
        state_dict=state,
        global_mean=[0.0],
        global_std=[1.0],
    )
    assert result == {"state_dict": state, "n": 0, "loss": 0.0}


def test_partial_zero_learning_rate_leaves_weights_unchanged() -> None:
    # A zero-length optimizer step should be a no-op: whatever comes in as
    # state_dict must come back out identical (this exercises the
    # load_state_dict -> train -> state_dict round trip without needing to
    # hand-verify the gradient math itself).
    df = _separable_df(30, seed=1)
    state = _zeroed_state(1)
    result = _call_partial(
        df,
        "partial",
        feature_cols=["x"],
        target_col="y",
        state_dict=state,
        global_mean=[0.0],
        global_std=[1.0],
        learning_rate=0.0,
        local_epochs=3,
        batch_ratio=1.0,
    )
    assert np.allclose(result["state_dict"]["linear.weight"], state["linear.weight"])
    assert np.allclose(result["state_dict"]["linear.bias"], state["linear.bias"])


def test_partial_training_reduces_loss_on_separable_data() -> None:
    # Sanity check that gradient descent is actually happening: on cleanly
    # separable data, enough local epochs at a reasonable learning rate must
    # end with a lower loss than starting from zero-initialized weights.
    df = _separable_df(200, seed=2)
    zero_state = _zeroed_state(1)

    result = _call_partial(
        df,
        "partial",
        feature_cols=["x"],
        target_col="y",
        state_dict=zero_state,
        global_mean=[float(df["x"].mean())],
        global_std=[float(df["x"].std())],
        learning_rate=0.5,
        local_epochs=50,
        batch_ratio=1.0,
    )

    # Loss at zero-initialized weights is exactly ln(2) (BCE of p=0.5 on any label).
    assert result["loss"] < 0.6


def test_partial_batch_size_reflects_batch_ratio() -> None:
    df = _separable_df(100, seed=3)
    result = _call_partial(
        df,
        "partial",
        feature_cols=["x"],
        target_col="y",
        state_dict=_zeroed_state(1),
        global_mean=[0.0],
        global_std=[1.0],
        train_ratio=1.0,
        batch_ratio=1.0,
    )
    assert result["n"] == 100


# ── evaluate ───────────────────────────────────────────────────────────────────


def test_evaluate_empty_test_set() -> None:
    df = pd.DataFrame({"x": [None], "y": [None]})
    result = _call_partial(
        df, "evaluate", feature_cols=["x"], target_col="y", state_dict=_zeroed_state(1), global_mean=[0.0], global_std=[1.0]
    )
    assert result == {"n": 0, "correct": 0}


def test_evaluate_perfect_accuracy_with_a_hand_built_decision_boundary() -> None:
    # Build a state_dict implementing a known decision rule (predict 1 iff
    # x > 0) and confirm evaluate() reports 100% accuracy on data that
    # exactly matches that rule.
    torch.manual_seed(0)
    model = LogisticRegressionModel(1)
    with torch.no_grad():
        model.linear.weight[:] = 10.0  # steep boundary at x=0 after normalization
        model.linear.bias[:] = 0.0
    state = _sd_to_list(model.state_dict())

    df = _separable_df(50, seed=4)  # y = (x > 0), exactly what the hand-built model predicts
    result = _call_partial(
        df, "evaluate", feature_cols=["x"], target_col="y", state_dict=state, global_mean=[0.0], global_std=[1.0], train_ratio=0.0
    )
    assert result["n"] == 50
    assert result["correct"] == 50


def test_evaluate_test_split_is_disjoint_from_compute_stats_train_split() -> None:
    # evaluate() and compute_stats() must derive the same train/test
    # partition from (train_ratio, seed) so the reported test accuracy is
    # genuinely held-out, not data the normalization stats were fit on.
    df = pd.DataFrame({"x": list(range(50)), "y": [0, 1] * 25})
    seed, ratio = 11, 0.7

    df_clean = df.dropna(subset=["x", "y"])
    train_idx = set(df_clean.sample(frac=ratio, random_state=seed).index)
    test_idx = set(df_clean.index) - train_idx

    stats = _call_partial(df, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=ratio, seed=seed)
    result = _call_partial(
        df, "evaluate", feature_cols=["x"], target_col="y", state_dict=_zeroed_state(1),
        global_mean=[0.0], global_std=[1.0], train_ratio=ratio, seed=seed,
    )

    assert stats["n"] == len(train_idx)
    assert result["n"] == len(test_idx)


# ── central: full pipeline ──────────────────────────────────────────────────────


def test_central_returns_error_when_no_usable_data() -> None:
    df = pd.DataFrame({"x": [None], "y": [None]})
    result = _run_central([df], feature_cols=["x"], target_col="y")
    assert result == {"error": "No usable data across any organization"}


def test_central_basic_shape() -> None:
    df = _separable_df(60, seed=5)
    result = _run_central([df], feature_cols=["x"], target_col="y")
    assert set(result) == {
        "state_dict", "feature_cols", "global_mean", "global_std",
        "n_train", "n_test", "accuracy", "per_node",
    }
    assert result["n_train"] > 0
    assert result["n_test"] > 0
    assert 0.0 <= result["accuracy"] <= 1.0


def test_central_global_normalization_matches_pooled_computation() -> None:
    # Phase 1 (global_mean/global_std) is pure sum/count aggregation, not
    # iterative optimization -- unlike the trained weights, it must match
    # pooling all nodes' training rows exactly, the same property tested for
    # average.py and coxph's summed statistics.
    node_a = _separable_df(70, seed=6)
    node_b = _separable_df(30, seed=7)
    result = _run_central([node_a, node_b], feature_cols=["x"], target_col="y")

    train_a = node_a.sample(frac=TRAIN_TEST_RATIO, random_state=RANDOM_SEED)
    train_b = node_b.sample(frac=TRAIN_TEST_RATIO, random_state=RANDOM_SEED)
    pooled_train = pd.concat([train_a, train_b])

    assert result["global_mean"] == pytest.approx([pooled_train["x"].mean()], rel=1e-5)
    variance = (pooled_train["x"] ** 2).mean() - pooled_train["x"].mean() ** 2
    assert result["global_std"] == pytest.approx([np.sqrt(max(variance, 1e-8))], rel=1e-5)


def test_central_learns_a_clearly_separable_pattern(monkeypatch) -> None:
    # End-to-end learning check: on cleanly separable data split across
    # nodes, after the full federated training loop the model should do
    # meaningfully better than chance (0.5), not just run without error.
    monkeypatch.setattr(logistic_regression, "N_ROUNDS", 10)
    node_a = _separable_df(80, seed=8)
    node_b = _separable_df(80, seed=9)
    result = _run_central([node_a, node_b], feature_cols=["x"], target_col="y")
    assert result["accuracy"] > 0.8


def test_central_fedavg_weights_by_node_sample_count_not_naively(monkeypatch) -> None:
    # The property confirmed by hand-reading central()'s aggregation loop:
    # reproduce round 1's inputs independently, call partial() directly per
    # node to get real (not hand-derived) per-node updates, then confirm
    # central()'s actual aggregated result matches the n-weighted average of
    # those two real updates and NOT their naive unweighted average.
    monkeypatch.setattr(logistic_regression, "N_ROUNDS", 1)
    monkeypatch.setattr(logistic_regression, "BATCH_RATIO", 1.0)

    node_a = _separable_df(90, seed=10)  # large node
    node_b = _separable_df(10, seed=11)  # small node

    torch.manual_seed(RANDOM_SEED)
    initial_state = _sd_to_list(LogisticRegressionModel(1).state_dict())

    stats_a = _call_partial(node_a, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=TRAIN_TEST_RATIO, seed=RANDOM_SEED)
    stats_b = _call_partial(node_b, "compute_stats", feature_cols=["x"], target_col="y", train_ratio=TRAIN_TEST_RATIO, seed=RANDOM_SEED)
    total_n = stats_a["n"] + stats_b["n"]
    global_mean = [(stats_a["sum"][0] + stats_b["sum"][0]) / total_n]
    variance = (stats_a["sum_sq"][0] + stats_b["sum_sq"][0]) / total_n - global_mean[0] ** 2
    global_std = [float(np.sqrt(max(variance, 1e-8)))]

    common_kwargs = dict(
        feature_cols=["x"], target_col="y", state_dict=initial_state,
        global_mean=global_mean, global_std=global_std,
        learning_rate=LEARNING_RATE, local_epochs=LOCAL_EPOCHS,
        train_ratio=TRAIN_TEST_RATIO, batch_ratio=1.0, seed=RANDOM_SEED,
    )
    update_a = _call_partial(node_a, "partial", **common_kwargs)
    update_b = _call_partial(node_b, "partial", **common_kwargs)

    n_a, n_b = update_a["n"], update_b["n"]
    frac_a, frac_b = n_a / (n_a + n_b), n_b / (n_a + n_b)
    weighted_weight = frac_a * np.array(update_a["state_dict"]["linear.weight"]) + frac_b * np.array(update_b["state_dict"]["linear.weight"])
    naive_weight = 0.5 * np.array(update_a["state_dict"]["linear.weight"]) + 0.5 * np.array(update_b["state_dict"]["linear.weight"])

    central_result = _run_central([node_a, node_b], feature_cols=["x"], target_col="y")
    actual_weight = np.array(central_result["state_dict"]["linear.weight"])

    assert actual_weight == pytest.approx(weighted_weight, abs=1e-5)
    if not np.allclose(weighted_weight, naive_weight, atol=1e-6):
        assert not np.allclose(actual_weight, naive_weight, atol=1e-6)
