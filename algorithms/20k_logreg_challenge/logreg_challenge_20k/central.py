"""
Central (orchestrator) functions for the ADMM logistic regression algorithm.

The central function is executed on a vantage6 node just like any other method,
but it coordinates tasks across all participating organizations:

1. Each node preprocesses its local CSV into the exact feature representation
   used in your standalone script.
2. The central function runs ADMM rounds by:
   - requesting local X-updates from each node;
   - updating the global consensus vector ``z`` and the dual variables ``u``;
   - checking convergence;
   - evaluating the global model on each node's validation set.

The logic below closely mirrors the ``run_local_simulation`` routine from your
single-file simulation, but translated into the vantage6 task model.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.util import info, warn
np.random.seed(67)


def _z_objective(
    z: np.ndarray,
    x_hat_mean: np.ndarray,
    u_mean: np.ndarray,
    rho: float,
    lambda_: float,
    num_sites: int,
) -> Tuple[float, np.ndarray]:
    """
    Exact Python implementation of the Z-update objective from your script.
    """
    z_reg = z.copy()
    z_reg[0] = 0.0  # Skip intercept penalty

    # Value
    l1_val = lambda_ * np.linalg.norm(z_reg, 1)
    diff = z - x_hat_mean - u_mean
    quad_val = (num_sites * rho / 2.0) * np.dot(diff, diff)
    value = l1_val + quad_val

    # Gradient. abs_z is floored at eps for every component (not just the
    # intercept at index 0) — any z_reg entry can land on exactly 0.0 during
    # optimization, and an unguarded 0/0 there produces NaN that poisons the
    # gradient regardless of lambda_ (0.0 * NaN is NaN, not 0, under IEEE 754).
    eps = 1e-64
    abs_z = np.abs(z_reg)
    abs_z = np.where(abs_z < eps, eps, abs_z)

    grad_l1 = lambda_ * (z_reg / abs_z)
    # grad_l1[0] == 0.0 already, since z_reg[0] == 0.0.

    grad_quad = num_sites * rho * diff
    grad = grad_l1 + grad_quad
    return value, grad


def _check_convergence(
    x_mat: np.ndarray,
    z_vec: np.ndarray,
    z_old_vec: np.ndarray,
    u_mat: np.ndarray,
    rho: float,
    abs_tol: float,
    rel_tol: float,
    logging: bool,
) -> Tuple[float, float, float, float]:
    """
    Compute primal and dual residuals and their tolerances, mirroring the
    ``check_convergence`` function from your standalone code.
    """
    num_features, num_sites = x_mat.shape

    z_mat = np.tile(z_vec.reshape(-1, 1), (1, num_sites))
    z_old_mat = np.tile(z_old_vec.reshape(-1, 1), (1, num_sites))

    r_norm = np.linalg.norm(x_mat - z_mat, ord="fro")
    s_norm = np.linalg.norm(rho * (z_mat - z_old_mat), ord="fro")

    eps_abs = np.sqrt(num_sites * num_features) * abs_tol
    eps_rel = rel_tol * max(
        np.linalg.norm(x_mat, "fro"),
        np.linalg.norm(z_mat, "fro"),
    )
    eps_pri = eps_abs + eps_rel

    eps_dual_base = np.sqrt(num_sites * num_features) * abs_tol
    eps_dual_rel = rel_tol * np.linalg.norm(rho * u_mat, "fro")
    eps_dual = eps_dual_base + eps_dual_rel

    # mu = 10
    # tau_incr = 2
    # tau_decr = 2

    # if r_norm > mu * s_norm:
    #     rho *= tau_incr
    # elif s_norm > mu * r_norm:
    #     rho /= tau_decr
    # else:
    #     rho = rho

    if logging == True:
        info(
        "Convergence info: "
        f"r_norm - eps_pri = {r_norm - eps_pri:.4e}, "
        f"s_norm - eps_dual = {s_norm - eps_dual:.4e}"
    )
    return r_norm, s_norm, eps_pri, eps_dual


def _record_training_metrics(
    history: Dict[str, Any],
    site_obj: List[float],
    reg_obj: float,
    site_sse: List[float],
    z: np.ndarray,
    rho: float,
    patient_counts: List[int],
) -> None:
    """
    Mirror of recordAdmmVariables.m: logs objective, SSE, 'RMSE' and z-coefficients.
    """
    total_patients = int(sum(patient_counts))

    obj_loss = float(np.sum(site_obj))
    obj_reg = float(reg_obj)
    obj_total = obj_loss + obj_reg

    history.setdefault("obj_loss_log", []).append(obj_loss)
    history.setdefault("obj_reg_log", []).append(obj_reg)
    history.setdefault("obj_log", []).append(obj_total)

    sum_square_error = float(np.sum(site_sse))
    history.setdefault("sum_square_error_log", []).append(sum_square_error)

    root_mean_square_error = sum_square_error / total_patients
    history.setdefault("root_mean_square_error_log", []).append(root_mean_square_error)

    history.setdefault("z_log", []).append(z.tolist())
    history.setdefault("rho_log", []).append(float(rho))


def _compute_roc_and_calibration(
    y_all: np.ndarray,
    probs_all: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute global ROC/AUC and calibration metrics, analogous to the standalone
    evaluate_final_model() from the local simulation.
    """
    eps = 1e-15
    probs_clipped = np.clip(probs_all, eps, 1.0 - eps)

    # Global ROC / AUC
    fpr_g, tpr_g, _ = roc_curve(y_all, probs_clipped)
    auc_g = roc_auc_score(y_all, probs_clipped)

    # Calibration: logit(p) = a + b * LP
    lp = np.log(probs_clipped / (1.0 - probs_clipped))
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(lp.reshape(-1, 1), y_all)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])

    # Quantile-based calibration curve
    n_quantiles = 10
    quantile_edges = np.quantile(probs_clipped, np.linspace(0, 1, n_quantiles + 1))

    mean_pred: List[float] = []
    mean_obs: List[float] = []
    bin_centers: List[float] = []

    for q in range(n_quantiles):
        lo, hi = quantile_edges[q], quantile_edges[q + 1]
        if q == n_quantiles - 1:
            mask = (probs_clipped >= lo) & (probs_clipped <= hi)
        else:
            mask = (probs_clipped >= lo) & (probs_clipped < hi)
        if not np.any(mask):
            continue

        probs_q = probs_clipped[mask]
        y_q = y_all[mask]

        mean_pred.append(float(np.mean(probs_q)))
        mean_obs.append(float(np.mean(y_q)))
        bin_centers.append(float(np.mean([lo, hi])))

    return {
        "roc_global": {
            "fpr": fpr_g.tolist(),
            "tpr": tpr_g.tolist(),
            "auc": float(auc_g),
        },
        "calibration": {
            "intercept": intercept,
            "slope": slope,
            "bin_centers": bin_centers,
            "mean_pred": mean_pred,
            "mean_obs": mean_obs,
        },
    }


@algorithm_client
def central_function(
    client: AlgorithmClient,
    num_rounds: int = 100,
    rho: float = 3,
    alpha: float = 1.4,
    lambda_: float = 0.0,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    logging: bool = True
) -> Dict[str, Any]:
    """
    Central ADMM logistic regression.

    This function orchestrates the full ADMM procedure across all nodes and
    returns the final global coefficients plus a small convergence history.
    """
    # ------------------------------------------------------------------
    # 1. Determine participating organizations
    # ------------------------------------------------------------------
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]
    num_sites = len(org_ids)

    if num_sites == 0:
        warn("No organizations found in collaboration - nothing to do.")
        return {"status": "no_organizations"}
    if logging == True:
        info(f"Starting ADMM logistic regression over {num_sites} sites")

    # ------------------------------------------------------------------
    # 2. Initialization: determine feature dimension and patient counts
    # ------------------------------------------------------------------
    if logging == True:
        info("Requesting local initialization from all sites")
    init_task = client.task.create(
        input_={
            "method": "init_node",
            "kwargs": {
                "logging" : bool(logging),
            },
        },
        organizations=org_ids,
        name="ADMM initialization",
        description="Determine local feature dimensions and patient counts",
    )
    init_results = client.wait_for_results(task_id=init_task.get("id"))

    num_features_list: List[int] = []
    patient_counts: List[int] = []
    for res in init_results:
        num_features_list.append(int(res["num_features"]))
        patient_counts.append(int(res["patient_count"]))

    if len(set(num_features_list)) != 1:
        warn(
            "Not all sites report the same number of features. "
            f"Reported: {num_features_list}"
        )
        # Use the minimum to keep dimension consistent
        num_features = min(num_features_list)
    else:
        num_features = num_features_list[0]

    total_features = num_features + 1  # +1 for intercept
    total_patients = int(sum(patient_counts))

    if logging == True:
        info(
        f"ADMM state: num_features={num_features}, total_features={total_features}, "
        f"patient_counts={patient_counts}, total_patients={total_patients}"
    )

    # ------------------------------------------------------------------
    # 3. Initialize ADMM variables
    # ------------------------------------------------------------------
    z = np.ones(total_features)
    # z = np.zeros(total_features)
    u_mat = np.zeros((total_features, num_sites))
    x_mat = np.zeros((total_features, num_sites))

    history: Dict[str, List[Any]] = {
        "round": [],
        "r_norm": [],
        "s_norm": [],
        "eps_pri": [],
        "eps_dual": [],
        "val_acc_mean": [],
        "val_acc_per_node": [],
        "val_rmse": [],  # global validation RMSE per round
    }

    # ------------------------------------------------------------------
    # 4. ADMM main loop
    # ------------------------------------------------------------------
    if logging == True:
        info(f"Running up to {num_rounds} ADMM rounds")

    for r in range(1, num_rounds + 1):
        if logging == True:
            info(f"--- ADMM round {r} ---")

        # --------------------------------------------------------------
        # 4A. Local X-updates at each site
        # --------------------------------------------------------------
        tasks = []
        for site_idx, org_id in enumerate(org_ids):
            input_ = {
                "method": "admm_x_update_partial",
                "kwargs": {
                    "z": z.tolist(),
                    "u": u_mat[:, site_idx].tolist(),
                    "x_prev": x_mat[:, site_idx].tolist(),
                    "rho": float(rho),
                    "total_patients": int(total_patients),
                    "logging": bool(logging)
                },
            }
            task = client.task.create(
                input_=input_,
                organizations=[org_id],
                name=f"ADMM X-update round {r}",
                description="Local logistic regression X-update",
            )
            tasks.append(task)

        # Wait for all sites to finish their X-updates
        local_results: List[Dict[str, Any]] = []
        for task in tasks:
            # Each task is sent to a single organization, so result list has len==1
            res_list = client.wait_for_results(task_id=task.get("id"))
            local_results.append(res_list[0])

        # Update X matrix and collect per-site SSE and objective
        site_sse: List[float] = []
        site_obj: List[float] = []
        for site_idx, res in enumerate(local_results):
            x_site = np.asarray(res["x"], dtype=float).reshape(-1)
            if x_site.size != total_features:
                warn(
                    f"Site {site_idx} returned {x_site.size} coefficients, "
                    f"expected {total_features}; truncating/padding as needed."
                )
                if x_site.size > total_features:
                    x_site = x_site[:total_features]
                else:
                    tmp = np.zeros(total_features)
                    tmp[: x_site.size] = x_site
                    x_site = tmp
            x_mat[:, site_idx] = x_site
            site_sse.append(float(res.get("sum_square_error", 0.0)))
            site_obj.append(float(res.get("obj", 0.0)))

        # --------------------------------------------------------------
        # 4B. Global Z- and U-updates (central)
        # --------------------------------------------------------------
        z_old = z.copy()

        z_mat = np.tile(z.reshape(-1, 1), (1, num_sites))
        x_hat = alpha * x_mat + (1.0 - alpha) * z_mat

        x_hat_mean = np.mean(x_hat, axis=1)
        u_mean = np.mean(u_mat, axis=1)
        # if lambda_ == 0:
        #     z = x_hat_mean + u_mean
        res_z = minimize(
            fun=_z_objective,
            x0=z,
            args=(x_hat_mean, u_mean, float(rho), float(lambda_), num_sites),
            jac=True,
            method="BFGS",
            options={"disp": False},
        )
        z = res_z.x

        # Global regularization objective component (lambda * ||z||_1 without intercept)
        z_reg = z.copy()
        z_reg[0] = 0.0
        reg_obj = float(lambda_ * np.linalg.norm(z_reg, 1))

        # Update dual variables
        for site_idx in range(num_sites):
            u_mat[:, site_idx] = u_mat[:, site_idx] + x_hat[:, site_idx] - z

        # --------------------------------------------------------------
        # 4C. Convergence check
        # --------------------------------------------------------------
        r_norm, s_norm, eps_pri, eps_dual = _check_convergence(
            x_mat=x_mat,
            z_vec=z,
            z_old_vec=z_old,
            u_mat=u_mat,
            rho=float(rho),
            abs_tol=float(abs_tol),
            rel_tol=float(rel_tol),
            logging=logging
        )

        # --------------------------------------------------------------
        # 4D. Validation of global model on each node
        # --------------------------------------------------------------
        eval_tasks = []
        for site_idx, org_id in enumerate(org_ids):
            input_ = {
                "method": "evaluate_global_model",
                "kwargs": {
                    "z": z.tolist(),
                    "logging": logging,
                },
            }
            task = client.task.create(
                input_=input_,
                organizations=[org_id],
                name=f"ADMM evaluation round {r}",
                description="Evaluate global model on local validation data",
            )
            eval_tasks.append(task)

        val_acc_per_node: List[float] = []
        val_sse_per_node: List[float] = []
        val_patients_per_node: List[int] = []
        for task in eval_tasks:
            res_list = client.wait_for_results(task_id=task.get("id"))
            res_eval = res_list[0]
            val_acc = float(res_eval["val_acc"])
            val_sse = float(res_eval["val_sum_square_error"])
            val_n = int(res_eval["val_patient_count"])

            val_acc_per_node.append(val_acc)
            val_sse_per_node.append(val_sse)
            val_patients_per_node.append(val_n)

        mean_val_acc = float(np.mean(val_acc_per_node))

        total_val_sse = float(np.sum(val_sse_per_node))
        total_val_patients = int(np.sum(val_patients_per_node))
        val_rmse_global = (
            float(np.sqrt(total_val_sse / total_val_patients))
            if total_val_patients > 0
            else float("nan")
        )

        # Store history
        history["round"].append(r)
        history["r_norm"].append(float(r_norm))
        history["s_norm"].append(float(s_norm))
        history["eps_pri"].append(float(eps_pri))
        history["eps_dual"].append(float(eps_dual))
        history["val_acc_per_node"].append(val_acc_per_node)
        history["val_acc_mean"].append(mean_val_acc)
        history["val_rmse"].append(val_rmse_global)

        # Training progress logs: objective, SSE, 'RMSE' and coefficients
        _record_training_metrics(
            history=history,
            site_obj=site_obj,
            reg_obj=reg_obj,
            site_sse=site_sse,
            z=z,
            rho=rho,
            patient_counts=patient_counts,
        )

        info(
            f"Round {r:03d}: mean val acc={mean_val_acc:.4f}, "
            f"val RMSE={val_rmse_global:.4f} | "
            f"r_norm={r_norm:.4e} (eps={eps_pri:.4e}) | "
            f"s_norm={s_norm:.4e} (eps={eps_dual:.4e})"
        )

        if r_norm < eps_pri and s_norm < eps_dual and r > 0:
            info(f"*** ADMM converged at round {r} ***")
            break

    # ------------------------------------------------------------------
    # 5. Final ROC/AUC and calibration metrics (across all patients)
    # ------------------------------------------------------------------
    if logging == True:
        info("Collecting final predictions for ROC/AUC and calibration")
    pred_tasks = []
    for site_idx, org_id in enumerate(org_ids):
        input_ = {
            "method": "collect_predictions",
            "kwargs": {
                "z": z.tolist(),
                "logging": logging,
            },
        }
        task = client.task.create(
            input_=input_,
            organizations=[org_id],
            name="Collect predictions for ROC/calibration",
            description="Return y and predicted probabilities (val only)",
        )
        pred_tasks.append(task)

    y_all_list: List[np.ndarray] = []
    probs_all_list: List[np.ndarray] = []
    roc_site: List[Dict[str, Any]] = []

    for site_idx, task in enumerate(pred_tasks):
        res_list = client.wait_for_results(task_id=task.get("id"))
        res_pred = res_list[0]
        y_site = np.asarray(res_pred["y_all"], dtype=float)
        probs_site = np.asarray(res_pred["probs_all"], dtype=float)

        y_all_list.append(y_site)
        probs_all_list.append(probs_site)

        # Per-site ROC / AUC (skipped for sites with no validation patients,
        # e.g. all local rows fall on the training side of the year cutoff)
        if y_site.size == 0:
            continue
        eps = 1e-15
        probs_clipped = np.clip(probs_site, eps, 1.0 - eps)
        fpr_s, tpr_s, _ = roc_curve(y_site, probs_clipped)
        auc_s = roc_auc_score(y_site, probs_clipped)
        roc_site.append(
            {
                "site": site_idx,
                "fpr": fpr_s.tolist(),
                "tpr": tpr_s.tolist(),
                "auc": float(auc_s),
            }
        )

    y_all = np.concatenate(y_all_list) if y_all_list else np.array([])
    probs_all = np.concatenate(probs_all_list) if probs_all_list else np.array([])

    roc_global: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    if y_all.size > 0:
        metrics = _compute_roc_and_calibration(y_all=y_all, probs_all=probs_all)
        roc_global = metrics["roc_global"]
        calibration = metrics["calibration"]

    info("ADMM finished; returning final global coefficients, history and metrics")
    return {
        "coefficients": z.tolist(),
        "history": history,
        "rho": float(rho),
        "lambda_": float(lambda_),
        "alpha": float(alpha),
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
        "patient_counts": patient_counts,
        "roc_site": roc_site,
        "roc_global": roc_global,
        "calibration": calibration,
    }

