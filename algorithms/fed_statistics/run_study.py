import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "fed_statistics:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

STATISTICS = {
    "age": ["mean", "std", "minmax", "quantiles"],
    "gender": ["counts"],
    "Overall.Stage": ["counts"],
    "Survival.time": ["mean", "std", "minmax"],
}
FILTERS = {}
OPTIONS = {}


def _is_nested_bounds_dict(value) -> bool:
    return isinstance(value, dict) and bool(value) and isinstance(next(iter(value.values())), dict)


def print_columns(columns: dict) -> None:
    for column, stats in columns.items():
        print(f"\n{column}")
        print("-" * len(column))
        for statistic, value in stats.items():
            if value is None:
                print(f"  {statistic:<12}: (suppressed or unavailable)")
            elif _is_nested_bounds_dict(value):
                # counts / binned_counts: {label: {"lower": ..., "upper": ...}}
                for label, bounds in value.items():
                    if bounds["lower"] == bounds["upper"]:
                        print(f"  {statistic:<12}: {label} = {bounds['lower']}")
                    else:
                        print(f"  {statistic:<12}: {label} = [{bounds['lower']}, {bounds['upper']}]")
            elif isinstance(value, dict):
                # minmax / quantiles: flat {field: scalar}
                formatted = ", ".join(
                    f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}"
                    for key, val in value.items()
                )
                print(f"  {statistic:<12}: {formatted}")
            elif isinstance(value, float):
                print(f"  {statistic:<12}: {value:.4f}")
            else:
                print(f"  {statistic:<12}: {value}")


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
    print(f"Statistics    : {STATISTICS}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated descriptive statistics",
        description="Compute federated descriptive statistics across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {
                "statistics": STATISTICS,
                "filters": FILTERS,
                "options": OPTIONS,
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

    print_columns(result["columns"])

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
