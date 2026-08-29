import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "coxph:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

TIME_COL = "Survival.time"
OUTCOME_COL = "deadstatus.event"
EXPL_VARS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
MAX_ITERATIONS = 10
TOLERANCE = 1e-6


def print_model_table(model_json: str) -> None:
    rows = json.loads(model_json)
    coef = rows["Coef"]
    exp_coef = rows["Exp(coef)"]
    se = rows["SE"]
    z = rows["Z"]
    pval = rows["p-value"]
    lower_ci = rows["lower_CI"]
    upper_ci = rows["upper_CI"]

    print(f"\n{'Variable':<25}{'Coef':>10}{'Exp(coef)':>12}{'SE':>10}{'Z':>10}{'p-value':>12}{'95% CI':>22}")
    print("-" * 101)
    for var in coef:
        ci = f"[{lower_ci[var]:.4f}, {upper_ci[var]:.4f}]"
        print(
            f"{var:<25}{coef[var]:>10.4f}{exp_coef[var]:>12.4f}{se[var]:>10.4f}"
            f"{z[var]:>10.4f}{pval[var]:>12.4g}{ci:>22}"
        )


def main() -> None:
    client = Client(SERVER_URL, SERVER_PORT, API_PATH)

    try:
        client.authenticate(USERNAME, PASSWORD)
    except Exception as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        sys.exit(1)

    client.setup_encryption(None)

    collaborations = client.collaboration.list()
    collaboration = next(
        (c for c in collaborations["data"] if c["name"] == COLLABORATION_NAME), None
    )
    if collaboration is None:
        print(f"Collaboration '{COLLABORATION_NAME}' not found.", file=sys.stderr)
        sys.exit(1)
    collaboration_id = collaboration["id"]

    organizations = client.organization.list()
    initiating_org = next(
        (o for o in organizations["data"] if o["name"] == INITIATING_ORG), None
    )
    if initiating_org is None:
        print(f"Organization '{INITIATING_ORG}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Collaboration : {collaboration['name']} (id={collaboration_id})")
    print(f"Initiating org: {INITIATING_ORG} (id={initiating_org['id']})")
    print(f"Algorithm image: {ALGORITHM_IMAGE}")
    print(f"Time column   : {TIME_COL}")
    print(f"Outcome column: {OUTCOME_COL}")
    print(f"Covariates    : {EXPL_VARS}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated Cox Proportional Hazards",
        description="Compute federated CoxPH hazard ratios across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {
                "time_col": TIME_COL,
                "outcome_col": OUTCOME_COL,
                "expl_vars": EXPL_VARS,
                "max_iterations": MAX_ITERATIONS,
                "tolerance": TOLERANCE,
            },
        },
        databases=[{"label": "default"}],
    )
    task_id = task["id"]
    print(f"Task created (id={task_id}), waiting for results...")

    results = client.wait_for_results(task_id)

    data_items = results.get("data", []) if isinstance(results, dict) else results
    if not data_items:
        print("No results returned.", file=sys.stderr)
        sys.exit(1)

    raw = data_items[0]
    result_str = raw.get("result") if isinstance(raw, dict) else raw
    if not result_str:
        print("Task completed but result was empty (algorithm may have failed).", file=sys.stderr)
        sys.exit(1)

    result = json.loads(result_str) if isinstance(result_str, str) else result_str

    if "error" in result:
        print(f"Algorithm error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\nIncluded organizations: {result['included_organizations']}")
    if result["excluded_organizations"]:
        print(f"Excluded organizations: {result['excluded_organizations']}")

    print_model_table(result["model"])

    print(f"\nOverall p-value    : {result['overall_p_value']:.4g}")
    print(f"AIC                : {result['aic']:.4f}")
    print(f"Degrees of freedom : {result['degrees_of_freedom']}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
