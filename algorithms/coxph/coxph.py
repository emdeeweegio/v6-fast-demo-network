import math

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import chi2, norm
from vantage6.algorithm.tools.util import info, warn
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

TIME_COL = "Survival.time"
OUTCOME_COL = "deadstatus.event"
EXPL_VARS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
MAX_ITERATIONS = 10
TOLERANCE = 1e-6


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warn("Matrix inversion failed; using pseudo-inverse")
        return np.linalg.pinv(matrix)


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series,
) -> tuple:
    z_sum_vec = z_sum.to_numpy(dtype=float)
    n_covariates = z_sum_vec.shape[0]
    tot_p1 = np.zeros(n_covariates, dtype=float)
    tot_p2 = np.zeros((n_covariates, n_covariates), dtype=float)

    for index, row in aggregated_time_events.iterrows():
        denom = float(summed_agg1[index])
        if denom <= 0.0:
            continue
        freq = float(row["freq"])
        s1 = freq * (summed_agg2[index] / denom)
        first_part = summed_agg3[index] / denom
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        second_part = numerator / (denom * denom)
        s2 = freq * (first_part - second_part)
        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum_vec - tot_p1
    secondary_derivative = -tot_p2
    return primary_derivative, secondary_derivative


def _compute_log_likelihood(
    z_sum: pd.Series,
    beta: np.ndarray,
    summed_agg1: np.ndarray,
    aggregated_time_events: pd.DataFrame,
) -> float:
    linear_part = float(np.dot(z_sum.to_numpy(dtype=float), beta))
    risk_set_part = 0.0
    for i, row in aggregated_time_events.iterrows():
        if i >= len(summed_agg1):
            break
        denom = float(summed_agg1[i])
        if denom <= 0.0:
            warn(f"Risk set denominator non-positive at index {i}: {denom}")
            continue
        risk_set_part += float(row["freq"]) * math.log(denom)
    return linear_part - risk_set_part


def _build_results_table(beta: np.ndarray, s_errors: np.ndarray, expl_vars: list) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.divide(
            beta,
            s_errors,
            out=np.zeros_like(beta, dtype=float),
            where=s_errors > 0,
        )
    pvalues = 2.0 * norm.sf(np.abs(zvalues))

    results = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "Exp(coef)": np.around(np.exp(beta), 5),
            "SE": np.around(s_errors, 5),
            "Var": expl_vars,
            "Z": zvalues,
            "p-value": pvalues,
        }
    )
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    return results.set_index("Var")


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str = TIME_COL,
    outcome_col: str = OUTCOME_COL,
    expl_vars: list = None,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> dict:
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    n_covs = len(expl_vars)
    max_iterations = max(1, int(max_iterations))
    tolerance = float(tolerance)

    orgs = client.organization.list()
    ids = [org["id"] for org in orgs]

    info("=" * 60)
    info("FEDERATED COX PROPORTIONAL HAZARDS REGRESSION")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Time column   : {time_col}")
    info(f"Outcome column: {outcome_col}")
    info(f"Covariates    : {expl_vars}")

    # ── Phase 1: unique event times ────────────────────────────────────────
    info("")
    info("── PHASE 1: Determine unique event times ────────────────────")
    times_task = client.task.create(
        input_={
            "method": "get_unique_event_times",
            "kwargs": {"time_col": time_col, "outcome_col": outcome_col},
        },
        organizations=ids,
        name="coxph_unique_event_times",
        description="Get unique event times and frequencies",
    )
    times_results = client.wait_for_results(times_task["id"])

    excluded_ids = []
    unique_frames = []
    for output in times_results:
        not_met = output.get("n_threshold_not_met")
        if not_met is not None:
            warn(f"Insufficient events for organization {not_met}; excluding from analysis.")
            excluded_ids.append(not_met)
            continue
        times = output.get("times")
        if times:
            unique_frames.append(pd.DataFrame.from_dict(times))

    ids = [org_id for org_id in ids if org_id not in excluded_ids]
    if not ids:
        return {"error": "No organizations met the minimum event threshold"}
    if not unique_frames:
        return {"error": "No usable event-time data across any organization"}

    aggregated_time_events = pd.concat(unique_frames).groupby(time_col, as_index=False).sum()
    unique_time_events = aggregated_time_events[time_col].tolist()
    info(f"Included organizations: {ids} (excluded: {excluded_ids})")
    info(f"Unique event times    : {len(unique_time_events)}")

    # ── Phase 2: summed Z statistic ─────────────────────────────────────────
    info("")
    info("── PHASE 2: Compute summed Z statistic ──────────────────────")
    z_task = client.task.create(
        input_={
            "method": "compute_summed_z",
            "kwargs": {"outcome_col": outcome_col, "expl_vars": expl_vars},
        },
        organizations=ids,
        name="coxph_summed_z",
        description="Compute summed z statistic",
    )
    z_results = client.wait_for_results(z_task["id"])

    z_sum = pd.Series(0.0, index=expl_vars)
    for output in z_results:
        z_sum += pd.Series(output["sum"], index=expl_vars, dtype=float).fillna(0.0)

    # ── Phase 3: Newton-Raphson iterations ───────────────────────────────────
    info("")
    info("── PHASE 3: Newton-Raphson iterations ───────────────────────")
    beta = np.zeros(n_covs, dtype=float)
    secondary_derivative = -np.eye(n_covs, dtype=float)
    summed_agg1 = np.zeros(len(unique_time_events), dtype=float)

    for iteration in range(max_iterations):
        iteration_task = client.task.create(
            input_={
                "method": "perform_iteration",
                "kwargs": {
                    "time_col": time_col,
                    "expl_vars": expl_vars,
                    "beta": beta.tolist(),
                    "unique_time_events": unique_time_events,
                },
            },
            organizations=ids,
            name=f"coxph_iteration_{iteration + 1}",
            description="Iterating to find the optimal beta",
        )
        results = client.wait_for_results(iteration_task["id"])

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times, dtype=float)
        summed_agg2 = np.zeros((n_times, n_covs), dtype=float)
        summed_agg3 = np.zeros((n_times, n_covs, n_covs), dtype=float)

        for output in results:
            summed_agg1 += np.asarray(output["agg1"], dtype=float)
            agg2_df = pd.DataFrame.from_dict(output["agg2"]).reindex(columns=expl_vars)
            summed_agg2 += agg2_df.to_numpy(dtype=float)
            summed_agg3 += np.asarray(output["agg3"], dtype=float)

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1=summed_agg1,
            summed_agg2=summed_agg2,
            summed_agg3=summed_agg3,
            aggregated_time_events=aggregated_time_events,
            z_sum=z_sum,
        )

        beta_old = beta.copy()
        try:
            beta = beta_old - solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError:
            warn("Hessian is singular; falling back to pseudo-inverse update")
            beta = beta_old - _safe_inverse(secondary_derivative).dot(primary_derivative)

        delta = float(np.max(np.abs(beta - beta_old)))
        info(f"Iteration {iteration + 1}: max |Δbeta| = {delta:.3e}")
        if math.isnan(delta):
            warn("Optimization update produced NaN delta; stopping iterations")
            break
        if delta <= tolerance:
            info("Betas have settled; optimization converged")
            break

    # ── Phase 4: inference ───────────────────────────────────────────────────
    fisher = _safe_inverse(-secondary_derivative)
    s_errors = np.sqrt(np.clip(np.diag(fisher), a_min=0.0, a_max=None))

    information = -secondary_derivative
    wald_statistic = float(beta @ information @ beta)
    overall_p_value = float(chi2.sf(wald_statistic, len(beta)))

    try:
        log_likelihood = _compute_log_likelihood(
            z_sum=z_sum,
            beta=beta,
            summed_agg1=summed_agg1,
            aggregated_time_events=aggregated_time_events,
        )
        if math.isnan(log_likelihood) or math.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        aic = float(-2.0 * log_likelihood + 2.0 * len(beta))
    except (ValueError, FloatingPointError) as exc:
        warn(f"Could not compute AIC due to numerical/data issue: {exc}")
        aic = float("nan")

    results = _build_results_table(beta=beta, s_errors=s_errors, expl_vars=expl_vars)

    model_warnings = []
    for covariate, row in results.iterrows():
        coef = float(row["Coef"])
        se = float(row["SE"])
        if abs(coef) > 10.0 or np.isinf(coef) or np.isnan(coef) or abs(se) > 10.0 or np.isinf(se) or np.isnan(se):
            msg = (
                f"Warning: Covariate '{covariate}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            model_warnings.append(msg)

    info(f"Final coefficients: {beta.tolist()}")
    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results.to_json(),
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": int(len(beta)),
        "warnings": model_warnings,
    }


@data(1)
@algorithm_client
def get_unique_event_times(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    outcome_col: str = OUTCOME_COL,
    minimum_events: int = 10,
) -> dict:
    info("Computing unique event times")
    if int(df[outcome_col].notnull().sum()) <= int(minimum_events):
        org_id = getattr(client, "organization_id", -1)
        warn("Sub-task skipped because the number of samples is too small")
        return {"n_threshold_not_met": int(org_id)}

    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times["freq"] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    info(f"Local unique event times: {len(times)}")
    return {"times": times.to_dict()}


@data(1)
def compute_summed_z(
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    expl_vars: list = None,
) -> dict:
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    info("Computing summed z statistics")
    z_sum = df[df[outcome_col] == 1][expl_vars].sum().astype(float).to_dict()
    return {"sum": z_sum}


@data(1)
def perform_iteration(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    expl_vars: list = None,
    beta: list = None,
    unique_time_events: list = None,
) -> dict:
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    beta_arr = np.asarray(beta, dtype=float)
    info("Computing aggregates for the derivation of the partial likelihood")

    working = df[[time_col, *expl_vars]].dropna(how="any")
    X = working[expl_vars].to_numpy(dtype=float)
    times = working[time_col].to_numpy(dtype=float)

    if X.shape[0] == 0:
        zeros = np.zeros((len(unique_time_events), len(expl_vars)))
        return {
            "agg1": [0.0] * len(unique_time_events),
            "agg2": pd.DataFrame(zeros, columns=expl_vars).to_dict(),
            "agg3": [np.zeros((len(expl_vars), len(expl_vars))).tolist() for _ in unique_time_events],
        }

    exp_xb = np.exp(X @ beta_arr)
    agg1 = []
    agg2_rows = []
    agg3 = []
    n_covariates = len(expl_vars)

    for unique_time in unique_time_events:
        mask = times >= float(unique_time)
        if not np.any(mask):
            agg1.append(0.0)
            agg2_rows.append(np.zeros(n_covariates, dtype=float))
            agg3.append(np.zeros((n_covariates, n_covariates), dtype=float))
            continue

        Xi = X[mask]
        exp_i = exp_xb[mask]
        weighted = Xi * exp_i[:, None]

        agg1.append(float(exp_i.sum()))
        agg2_rows.append(weighted.sum(axis=0))
        agg3.append(Xi.T @ weighted)

    agg2_df = pd.DataFrame(agg2_rows, columns=expl_vars)
    return {
        "agg1": agg1,
        "agg2": agg2_df.to_dict(),
        "agg3": [matrix.tolist() for matrix in agg3],
    }
