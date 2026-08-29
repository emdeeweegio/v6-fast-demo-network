import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "logistic_regression:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"
FEATURE_COLS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
TARGET_COL = "deadstatus.event"


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
    print(f"Features      : {FEATURE_COLS}")
    print(f"Target        : {TARGET_COL}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated Logistic Regression",
        description="Train and evaluate a logistic regression model across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL},
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

    accuracy = result.get("accuracy")
    n_train = result.get("n_train", 0)
    n_test = result.get("n_test", 0)
    features = result.get("feature_cols", [])
    global_mean = result.get("global_mean", [])
    global_std = result.get("global_std", [])

    print(f"Features       : {features}")
    print(f"Training samples (federated total): {n_train}")
    print(f"Test samples    (federated total): {n_test}")
    print()
    print("Normalization parameters (computed from federated training data):")
    for feat, mu, sigma in zip(features, global_mean, global_std):
        print(f"  {feat}: mean={mu:.4f}, std={sigma:.4f}")
    print()
    per_node = result.get("per_node", [])
    if per_node:
        print("Per-hospital test results:")
        for node in per_node:
            acc = node.get("accuracy")
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"  {node['org_name']:<12} n_test={node['n']}, correct={node['correct']}, accuracy={acc_str}")
        print()

    if accuracy is None:
        print("Global test accuracy: no test data available")
    else:
        total_correct = sum(node["correct"] for node in per_node) if per_node else int(round(accuracy * n_test))
        print(f"Global test accuracy: {accuracy:.4f}  ({total_correct}/{n_test} correct)")


if __name__ == "__main__":
    main()
