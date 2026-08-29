import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "datavalgen-validate:local"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"
MODEL = "example"


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
    print(f"Model         : {MODEL}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated datavalgen validate",
        description=f"Validate every node's local CSV against datavalgen model '{MODEL}'",
        image=ALGORITHM_IMAGE,
        input_={"method": "central", "kwargs": {"model": MODEL}},
        databases=[{"label": "default"}],
    )
    task_id = task["id"]
    print(f"Task created (id={task_id}), waiting for results...")

    results = client.wait_for_results(task_id)

    # wait_for_results returns {'data': [...], 'links': {...}}; extract the items list
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
    all_ok = result.get("all_ok")
    nodes = result.get("nodes", {})

    print()
    for name, node_result in nodes.items():
        if node_result["ok"]:
            print(f"  {name}: OK")
        elif node_result.get("column_errors"):
            print(f"  {name}: FAILED - {'; '.join(node_result['column_errors'])}")
        else:
            print(f"  {name}: FAILED - {node_result['num_errors']} invalid cell(s)")

    print()
    if all_ok:
        print(f"All nodes have data matching the '{result.get('model', MODEL)}' schema.")
    else:
        print(f"One or more nodes do NOT match the '{result.get('model', MODEL)}' schema.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
