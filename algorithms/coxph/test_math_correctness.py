from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

CURRENT_PATH = Path(__file__).parent
PREDICTORS = ["clin_n_1", "index_tumour_location_oropharynx"]
TIME_COL = "overall_survival_in_days"
OUTCOME_COL = "event_overall_survival"


def _cox_partial_loglik(beta: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    event_mask = event == 1
    z_sum = X[event_mask].sum(axis=0)
    unique_times, counts = np.unique(time[event_mask], return_counts=True)
    exp_xb = np.exp(X @ beta)

    risk_term = 0.0
    for t, d in zip(unique_times, counts):
        denom = exp_xb[time >= t].sum()
        risk_term += float(d) * np.log(float(denom))
    return float(np.dot(z_sum, beta) - risk_term)


def _reference_fit(df: pd.DataFrame) -> np.ndarray:
    working = df[[TIME_COL, OUTCOME_COL, *PREDICTORS]].dropna(how="any").copy()
    X = working[PREDICTORS].astype(float).to_numpy()
    time = working[TIME_COL].astype(float).to_numpy()
    event = working[OUTCOME_COL].astype(int).to_numpy()

    def objective(beta: np.ndarray) -> float:
        return -_cox_partial_loglik(beta, X, time, event)

    result = minimize(
        objective,
        x0=np.zeros(X.shape[1], dtype=float),
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-10},
    )
    # BFGS may report "precision loss" instead of success once the gradient is
    # near machine precision (scipy >=1.15) — the fit itself is still valid,
    # so check the gradient norm directly rather than trusting result.success.
    grad_norm = float(np.linalg.norm(result.jac))
    if not result.success and grad_norm > 1e-4:
        raise RuntimeError(f"reference optimizer failed: {result.message}")
    return result.x


def _run_federated_fit() -> dict:
    dataset = {
        "database": CURRENT_PATH / "HEAD-NECK-RADIOMICS-HN1.csv",
        "db_type": "csv",
        "input_data": {},
    }
    client = MockAlgorithmClient(
        datasets=[[dataset], [dataset]],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": TIME_COL,
                "outcome_col": OUTCOME_COL,
                "expl_vars": PREDICTORS,
                "max_iterations": 25,
                "tolerance": 1e-8,
            },
        },
        organizations=[org_ids[0]],
    )
    return client.wait_for_results(task.get("id"))[0]


def test_coefficients_match_reference_optimizer() -> None:
    df = pd.read_csv(CURRENT_PATH / "HEAD-NECK-RADIOMICS-HN1.csv")
    reference_beta = _reference_fit(df)

    result = _run_federated_fit()
    model_df = pd.read_json(StringIO(result["model"]))
    federated_beta = model_df["Coef"].astype(float).to_numpy()

    assert np.allclose(federated_beta, reference_beta, atol=1e-2, rtol=1e-2)


def test_reported_p_values_match_beta_over_se() -> None:
    result = _run_federated_fit()
    model_df = pd.read_json(StringIO(result["model"]))
    beta = model_df["Coef"].astype(float).to_numpy()
    se = model_df["SE"].astype(float).to_numpy()

    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    expected_p = 2.0 * norm.sf(np.abs(z))
    reported_p = model_df["p-value"].astype(float).to_numpy()

    assert np.allclose(reported_p, expected_p, atol=1e-4, rtol=1e-4)
