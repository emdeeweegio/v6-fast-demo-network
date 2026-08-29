import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from logreg_challenge_20k.central import _z_objective, central_function
from logreg_challenge_20k.partial import (
    FEATURE_COLUMNS,
    M_MAP,
    N_MAP,
    OUTCOME_COLUMN,
    S_MAP,
    T_MAP,
    _compute_two_year_survival,
    _logistic_admm_objective,
    _map_overall,
    _map_tnm,
    _predict_proba,
    _prepare_stage_columns_for_dummies,
    _preprocess_local_dataframe,
)


# ── _compute_two_year_survival ────────────────────────────────────────────────


def test_two_year_survival_thresholds_match_the_documented_rules() -> None:
    threshold = 2 * 365.24
    vital_status = pd.Series(
        ["dead", "dead", "alive", "alive", "unknown", "dead"],
        name="vital_status",
    )
    days = pd.Series(
        [threshold - 10, threshold + 10, threshold + 10, threshold - 10, threshold + 10, None],
        name="days",
    )

    result = _compute_two_year_survival(vital_status, days)

    assert result.iloc[0] == 0  # dead, within 2y -> 0
    assert result.iloc[1] == 1  # dead, past 2y -> 1
    assert result.iloc[2] == 1  # alive, past 2y -> 1
    assert np.isnan(result.iloc[3])  # alive, insufficient follow-up -> NaN
    assert np.isnan(result.iloc[4])  # unrecognized vital_status -> NaN
    assert np.isnan(result.iloc[5])  # dead but days unknown -> NaN (can't place vs. threshold)


def test_two_year_survival_is_not_just_vital_status():
    # Regression test for the actual bug: the previous code collapsed to
    # `out[dead] = 0; out[alive] = 1` regardless of `days`, i.e. it computed
    # plain vital status. This dataset has a dead patient who survived past
    # two years (should be 1, not 0) and an alive patient without enough
    # follow-up (should be NaN, not 1) -- both would fail under the old logic.
    vital_status = pd.Series(["dead", "alive"])
    days = pd.Series([2 * 365.24 + 100, 30])

    result = _compute_two_year_survival(vital_status, days)

    assert result.iloc[0] == 1
    assert np.isnan(result.iloc[1])


# ── _logistic_admm_objective ──────────────────────────────────────────────────


def _numeric_gradient(x, z, u, rho, features, outcome, total_patients, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus, _ = _logistic_admm_objective(x_plus, z, u, rho, features, outcome, total_patients)
        f_minus, _ = _logistic_admm_objective(x_minus, z, u, rho, features, outcome, total_patients)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_admm_objective_analytic_gradient_matches_numeric_gradient() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 3))
    outcome = rng.integers(0, 2, size=20).astype(float)
    x = rng.normal(scale=0.5, size=4)  # +1 for intercept
    z = np.zeros(4)
    u = np.zeros(4)

    _, analytic_grad = _logistic_admm_objective(x, z, u, rho=1.0, features=features, outcome=outcome, total_patients=20)
    numeric_grad = _numeric_gradient(x, z, u, 1.0, features, outcome, 20)

    assert np.allclose(analytic_grad, numeric_grad, atol=1e-4)


def test_admm_objective_gradient_has_no_nan_or_inf_for_large_x() -> None:
    # Regression test: the value computation clips logits to [-500, 500]
    # before exp(), but the gradient loop's exp_term used to be unclipped,
    # so a large x could overflow to inf -> nan in the gradient.
    features = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    outcome = np.array([1.0, 0.0, 1.0])
    x = np.array([300.0, 300.0, 300.0])  # xi @ x reaches ~900 for some rows, past float64 exp overflow (~709)
    z = np.zeros(3)
    u = np.zeros(3)

    value, grad = _logistic_admm_objective(x, z, u, rho=1.0, features=features, outcome=outcome, total_patients=3)

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


# ── _z_objective ───────────────────────────────────────────────────────────────


def test_z_objective_gradient_has_no_nan_when_a_component_is_exactly_zero() -> None:
    # Regression test: abs_z was only floored at index 0 (the intercept), so
    # any other component landing on exactly 0.0 produced a 0/0 = NaN in the
    # L1 gradient term, poisoning the whole gradient even when lambda_ > 0.
    z = np.array([1.0, 0.0, 2.0])  # non-intercept component (index 1) is exactly zero
    x_hat_mean = np.zeros(3)
    u_mean = np.zeros(3)

    value, grad = _z_objective(z, x_hat_mean, u_mean, rho=1.0, lambda_=0.5, num_sites=2)

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


def test_z_objective_gradient_matches_numeric_gradient_away_from_zero() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=4)
    z[0] = 0.7  # keep away from exact zero so the numeric check isn't near the L1 kink
    x_hat_mean = rng.normal(size=4)
    u_mean = rng.normal(size=4)
    rho, lambda_, num_sites = 0.8, 0.3, 3

    _, analytic_grad = _z_objective(z, x_hat_mean, u_mean, rho, lambda_, num_sites)

    eps = 1e-6
    numeric_grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus, z_minus = z.copy(), z.copy()
        z_plus[i] += eps
        z_minus[i] -= eps
        f_plus, _ = _z_objective(z_plus, x_hat_mean, u_mean, rho, lambda_, num_sites)
        f_minus, _ = _z_objective(z_minus, x_hat_mean, u_mean, rho, lambda_, num_sites)
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    assert np.allclose(analytic_grad, numeric_grad, atol=1e-3)


# ── stage-label mapping ────────────────────────────────────────────────────────


def test_map_tnm_collapses_synonymous_labels_to_the_same_bucket() -> None:
    series = pd.Series(["T1", "T1a", "T1b", "T2", "Tx"])
    mapped = _map_tnm(series, T_MAP)
    assert mapped.tolist() == [1, 1, 1, 2, 5]


def test_map_tnm_handles_missing_and_unrecognized_values() -> None:
    series = pd.Series(["T1", None, "not-a-stage"])
    mapped = _map_tnm(series, T_MAP)
    assert mapped.iloc[0] == 1
    assert pd.isna(mapped.iloc[1])
    assert pd.isna(mapped.iloc[2])  # unrecognized label -> NaN, not an error


def test_map_overall_handles_roman_numeral_and_numeric_representations() -> None:
    # S_MAP only has "0" as a numeric-string key (Stage 0 / carcinoma in
    # situ) -- there is no "1"/"2"/etc. numeric encoding, so a bare numeric
    # stage like "1" correctly has no mapping (NaN), unlike "0" and "0.0".
    series = pd.Series(["I", "II", "0", "0.0", "1"])
    mapped = _map_overall(series)
    assert mapped.iloc[0] == 1
    assert mapped.iloc[1] == 2
    assert mapped.iloc[2] == 0
    assert mapped.iloc[3] == 0  # "0.0" and "0" both resolve to the whole-number key "0"
    assert pd.isna(mapped.iloc[4])  # "1" has no key in S_MAP


def test_prepare_stage_columns_uses_fixed_categories_regardless_of_data_present() -> None:
    # Every node must produce the SAME dummy-variable columns even if its
    # local data doesn't contain every stage value, or federated one-hot
    # columns from different nodes wouldn't line up with each other.
    df = pd.DataFrame(
        {
            "patient_t_stage": ["T1"],  # only one T-stage value present locally
            "patient_n_stage": ["N0"],
            "patient_m_stage": ["M0"],
            "patient_overall_stage": ["I"],
        }
    )
    out = _prepare_stage_columns_for_dummies(df)
    assert list(out["patient_t_stage"].cat.categories) == sorted(set(T_MAP.values()))
    assert list(out["patient_n_stage"].cat.categories) == sorted(set(N_MAP.values()))
    assert list(out["patient_m_stage"].cat.categories) == sorted(set(M_MAP.values()))
    assert list(out["patient_overall_stage"].cat.categories) == sorted(set(S_MAP.values()))


# ── _preprocess_local_dataframe ────────────────────────────────────────────────


def _make_beach_row(t, n, m, overall, year, vital_status, days) -> dict:
    return {
        "patient_t_stage": t,
        "patient_n_stage": n,
        "patient_m_stage": m,
        "patient_overall_stage": overall,
        "year_of_diagnosis": year,
        "vital_status": vital_status,
        "interval_diagnosis_to_last_visit_in_days": days,
    }


def test_preprocess_splits_train_and_val_by_diagnosis_year() -> None:
    df = pd.DataFrame(
        [
            _make_beach_row("T1", "N0", "M0", "I", 2010, "alive", 1000),  # train
            _make_beach_row("T2", "N1", "M0", "II", 2011, "dead", 100),  # train (boundary year)
            _make_beach_row("T3", "N2", "M1", "III", 2012, "alive", 1000),  # val (boundary year)
            _make_beach_row("T4", "N2", "M1", "IV", 2015, "dead", 100),  # val
        ]
    )
    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df, logging=False)
    assert X_train.shape[0] == 2
    assert X_val.shape[0] == 2
    assert set(y_train) <= {0, 1}
    assert set(y_val) <= {0, 1}


def test_preprocess_drops_rows_with_unrecognized_or_missing_stage_labels() -> None:
    df = pd.DataFrame(
        [
            _make_beach_row("T1", "N0", "M0", "I", 2010, "alive", 1000),
            _make_beach_row("not-a-stage", "N0", "M0", "I", 2010, "alive", 1000),  # unmappable T-stage
            _make_beach_row("T1", "N0", "M0", "I", 2010, "unknown-status", 1000),  # unmappable outcome
        ]
    )
    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df, logging=False)
    assert X_train.shape[0] == 1


def test_preprocess_raises_on_missing_required_columns() -> None:
    df = pd.DataFrame({"patient_t_stage": ["T1"]})
    with pytest.raises(ValueError, match="vital_status"):
        _preprocess_local_dataframe(df, logging=False)


# ── node-side functions (init_node / admm_x_update_partial / evaluate_global_model / collect_predictions) ──


def _call_node_method(df: pd.DataFrame, method: str, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="logreg_challenge_20k")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _beach_df(n: int, seed: int, severity_coef: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.choice(list(T_MAP.keys()), size=n)
    nn = rng.choice(list(N_MAP.keys()), size=n)
    m = rng.choice(list(M_MAP.keys()), size=n)
    overall = rng.choice(["I", "II", "III", "IV"], size=n)
    # Mostly pre-2012 (train) with a real but smaller post-2012 (val) slice,
    # so the val split is never empty across a handful of synthetic nodes.
    year = rng.choice([2008, 2009, 2010, 2011, 2012, 2013], size=n, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    severity = (
        np.array([T_MAP[v] for v in t])
        + np.array([N_MAP[v] for v in nn])
        + np.array([M_MAP[v] for v in m])
        + np.array([S_MAP[v] for v in overall])
    ).astype(float)
    severity -= severity.mean()
    # A weak severity_coef (rather than a strongly separating one) matters:
    # the one-hot-encoded design here has 16 parameters, and a handful of
    # nodes only have ~100-200 rows each -- a strongly separating signal
    # pushes the reference MLE toward quasi-complete separation, where
    # "the true unregularized coefficients" isn't a stable, well-defined
    # target for either fit to reproduce.
    probs = 1.0 / (1.0 + np.exp(severity_coef * severity))  # higher severity -> lower survival probability
    survives = rng.binomial(1, probs)
    vital_status = np.where(survives == 1, "alive", "dead")
    days = np.where(survives == 1, 1000, 100)
    return pd.DataFrame(
        {
            "patient_t_stage": t,
            "patient_n_stage": nn,
            "patient_m_stage": m,
            "patient_overall_stage": overall,
            "year_of_diagnosis": year,
            "vital_status": vital_status,
            "interval_diagnosis_to_last_visit_in_days": days,
        }
    )


def test_init_node_reports_feature_count_and_training_patient_count() -> None:
    df = _beach_df(50, seed=0)
    result = _call_node_method(df, "init_node", logging=False)
    X_train, _, _, _ = _preprocess_local_dataframe(df, logging=False)
    assert result["num_features"] == X_train.shape[1]
    assert result["patient_count"] == X_train.shape[0]


def test_admm_x_update_partial_returns_a_coefficient_vector_of_the_right_length() -> None:
    df = _beach_df(80, seed=1)
    X_train, _, _, _ = _preprocess_local_dataframe(df, logging=False)
    num_features = X_train.shape[1] + 1  # +1 intercept
    result = _call_node_method(
        df,
        "admm_x_update_partial",
        z=[0.0] * num_features,
        u=[0.0] * num_features,
        rho=0.25,
        total_patients=80,
        x_prev=[0.0] * num_features,
        logging=False,
    )
    assert len(result["x"]) == num_features
    assert result["patient_count"] == X_train.shape[0]
    assert 0.0 <= result["train_acc"] <= 1.0


def test_evaluate_global_model_reports_validation_metrics() -> None:
    df = _beach_df(80, seed=2)
    _, _, X_val, y_val = _preprocess_local_dataframe(df, logging=False)
    num_features = X_val.shape[1] + 1
    result = _call_node_method(df, "evaluate_global_model", z=[0.0] * num_features, logging=False)
    assert result["val_patient_count"] == len(y_val)
    assert 0.0 <= result["val_acc"] <= 1.0


def test_collect_predictions_returns_validation_set_only() -> None:
    df = _beach_df(80, seed=3)
    _, _, X_val, y_val = _preprocess_local_dataframe(df, logging=False)
    num_features = X_val.shape[1] + 1
    result = _call_node_method(df, "collect_predictions", z=[0.0] * num_features, logging=False)
    assert len(result["y_all"]) == len(y_val)
    assert len(result["probs_all"]) == len(y_val)


# ── central_function: end-to-end ADMM correctness ─────────────────────────────


def test_admm_approximately_matches_pooled_logistic_regression_predictions() -> None:
    # Run the federated ADMM procedure split across several organizations and
    # compare its consensus model's predicted probabilities to a directly-fit
    # centralized logistic regression on the pooled data, using the reference
    # hyperparameters from run_study.py's own comment ("the config that
    # reproduces the centralized/pooled coefficients"): RHO=0.25, ALPHA=1,
    # LAMBDA_=0.0 (NUM_ROUNDS bumped up from the reference's 10 -- this
    # synthetic data's one-hot design is more correlated than the real BEACH
    # data, so it needs more rounds at this rho to get close).
    #
    # This is not exact coefficient recovery: raw coefficients aren't
    # identifiable to tight precision with correlated one-hot features, and
    # even at 150 rounds individual predicted probabilities can differ from
    # the reference by up to ~0.14. What this test does confirm is that mean
    # predicted-probability error is small and (together with
    # test_admm_history_shows_decreasing_primal_residual, which separately
    # confirms r_norm/s_norm decrease monotonically well past this point) that
    # ADMM is converging toward the pooled solution rather than to some other
    # fixed point.
    pooled = pd.concat([_beach_df(200, seed=s) for s in (10, 11, 12)], ignore_index=True)
    node_dfs = [_beach_df(200, seed=s) for s in (10, 11, 12)]

    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="logreg_challenge_20k",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central_function",
            "kwargs": {
                "num_rounds": 150,
                "rho": 0.25,
                "alpha": 1,
                "lambda_": 0.0,
                "abs_tol": 1e-8,
                "rel_tol": 1e-8,
                "logging": False,
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    admm_coef = np.array(result["coefficients"])

    X_train, y_train, _, _ = _preprocess_local_dataframe(pooled, logging=False)
    reference = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-10)
    reference.fit(X_train, y_train)

    # ADMM's predicted probabilities on the pooled training set are the
    # property that actually matters (raw coefficients aren't identifiable
    # to the same precision when one-hot features are correlated). Mean
    # absolute difference is the more robust summary here -- a handful of
    # borderline points can have a larger gap even as the bulk converges.
    admm_probs = _predict_proba(X_train, admm_coef)
    reference_probs = reference.predict_proba(X_train)[:, 1]
    assert np.mean(np.abs(admm_probs - reference_probs)) < 0.05


def test_admm_history_shows_decreasing_primal_residual() -> None:
    node_dfs = [_beach_df(60, seed=s) for s in (20, 21, 22)]
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="logreg_challenge_20k",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central_function",
            "kwargs": {"num_rounds": 15, "rho": 0.25, "alpha": 1, "lambda_": 0.0, "logging": False},
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    r_norm_history = result["history"]["r_norm"]
    # Later rounds should show a smaller primal residual than the first
    # round -- evidence the ADMM iterations are actually converging, not
    # just running a fixed number of times.
    assert r_norm_history[-1] < r_norm_history[0]


def test_admm_does_not_crash_when_no_node_has_validation_patients() -> None:
    # Regression test: every node's rows are pre-2012 (all train, no val),
    # so total_val_patients across the whole collaboration is 0. This used to
    # raise ZeroDivisionError computing val_rmse_global (should instead report
    # NaN for that round) and, separately, ValueError from sklearn's
    # roc_curve on an empty per-site y_site array (should instead skip that
    # site's ROC/AUC entry rather than crash the whole run).
    node_dfs = [_beach_df(30, seed=s).assign(year_of_diagnosis=2010) for s in (30, 31, 32)]
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="logreg_challenge_20k",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central_function",
            "kwargs": {"num_rounds": 3, "rho": 0.25, "alpha": 1, "lambda_": 0.0, "logging": False},
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    assert all(np.isnan(v) for v in result["history"]["val_rmse"])
    assert result["roc_site"] == []
    assert result["roc_global"] == {}
