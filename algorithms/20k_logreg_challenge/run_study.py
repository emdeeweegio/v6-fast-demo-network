import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "20klogregchallenge:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

# note: this algorithm expects BEACH-schema node data (patient_t_stage,
# patient_n_stage, patient_m_stage, patient_overall_stage, year_of_diagnosis,
# vital_status, interval_diagnosis_to_last_visit_in_days) - i.e. the nodes
# must have been started with `./infra.sh --nodes beach up` (nodes.beach.env),
# not the default `./infra.sh up` (nodes.lung1.env, LUNG1 schema). It will
# fail on nodes serving LUNG1 data.

# Values below match the reference run in
# 20kChallengeVantage6/my-fl-project/20kLogRegChallenge/run_on_v6_network.py
# (the config that reproduces the centralized/pooled coefficients).
NUM_ROUNDS = 10
RHO = 0.25
ALPHA = 1
LAMBDA_ = 0.0
ABS_TOL = 1e-3
REL_TOL = 1e-3
LOGGING = False


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
    print(f"num_rounds={NUM_ROUNDS}  rho={RHO}  alpha={ALPHA}  lambda_={LAMBDA_}")
    print(f"abs_tol={ABS_TOL}  rel_tol={REL_TOL}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="ADMM Logistic Regression (20kChallenge)",
        description="Federated ADMM logistic regression across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central_function",
            "kwargs": {
                "num_rounds": NUM_ROUNDS,
                "rho": RHO,
                "alpha": ALPHA,
                "lambda_": LAMBDA_,
                "abs_tol": ABS_TOL,
                "rel_tol": REL_TOL,
                "logging": LOGGING,
            },
        },
        databases=[{"label": "default"}],
    )
    task_id = task["id"]
    print(f"Task created (id={task_id}), waiting for results (this may take a while)...")

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

    print("\n=== ADMM Results ===")
    print(f"  Coefficients: {result.get('coefficients')}")
    print(f"  Patient counts: {result.get('patient_counts')}")

    history = result.get("history", {})
    if history:
        print(f"  Rounds completed: {len(history.get('round', []))}")
        accs = history.get("val_acc_mean", [])
        if accs:
            print(f"  Final mean val accuracy: {accs[-1]:.4f}")
        rmses = history.get("val_rmse", [])
        if rmses:
            print(f"  Final val RMSE: {rmses[-1]:.4f}")

    roc = result.get("roc_global", {})
    if roc:
        print(f"  Global AUC: {roc.get('auc', 'N/A')}")

    cal = result.get("calibration", {})
    if cal:
        print(f"  Calibration intercept: {cal.get('intercept', 'N/A')}")
        print(f"  Calibration slope: {cal.get('slope', 'N/A')}")


if __name__ == "__main__":
    main()
