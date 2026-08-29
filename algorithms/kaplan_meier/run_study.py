import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "kaplan_meier:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

TIME_COL = "Survival.time"
EVENT_COL = "deadstatus.event"
STEP_DAYS = 30

# Optional privacy noise injection on event times before aggregation.
# NOISE_TYPE: "none" (default), "gaussian" (needs SNR > 0), or "poisson".
# RANDOM_SEED: leave None to let central() generate a fresh one per run
# (it's echoed back in the result for reproducibility); set it to repeat
# a specific run's noise exactly.
NOISE_TYPE = "none"
SNR = None
RANDOM_SEED = None


def plot_km_curve(curve: list, n_patients: int, n_events: int, step_days: int, output_path: str = "km_curve.png") -> None:
    times = [p["time"] for p in curve]
    survival = [p["survival"] for p in curve]
    ci_lower = [p["ci_lower"] for p in curve]
    ci_upper = [p["ci_upper"] for p in curve]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(times, survival, where="post", color="#2171b5", linewidth=2, label="KM estimate")
    ax.fill_between(times, ci_lower, ci_upper, step="post", alpha=0.2, color="#2171b5", label="95% CI")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Time (days)", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_title(
        f"Federated Kaplan-Meier Survival Curve\n"
        f"n={n_patients} patients, {n_events} events, {step_days}-day steps",
        fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    # Annotate median survival
    median_time = None
    for i, s in enumerate(survival):
        if s <= 0.5:
            median_time = times[i]
            break
    if median_time is not None:
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.axvline(median_time, color="gray", linestyle=":", linewidth=1)
        ax.annotate(
            f"Median: {median_time} days",
            xy=(median_time, 0.5),
            xytext=(median_time + max(times) * 0.03, 0.55),
            fontsize=10,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


def print_km_table(curve: list) -> None:
    print(f"\n{'Time':>8}  {'At risk':>8}  {'Events':>8}  {'Survival':>10}  {'95% CI':>22}")
    print("-" * 64)
    for row in curve:
        ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        print(f"{row['time']:>8}  {row['n_risk']:>8}  {row['n_events']:>8}  {row['survival']:>10.4f}  {ci:>22}")


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
    print(f"Event column  : {EVENT_COL}")
    print(f"Step days     : {STEP_DAYS}")
    print(f"Noise type    : {NOISE_TYPE}")
    if NOISE_TYPE != "none":
        print(f"SNR           : {SNR}")
        print(f"Random seed   : {RANDOM_SEED}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated Kaplan-Meier",
        description="Compute federated KM survival curve across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {
                "time_col": TIME_COL,
                "event_col": EVENT_COL,
                "step_days": STEP_DAYS,
                "noise_type": NOISE_TYPE,
                "snr": SNR,
                "random_seed": RANDOM_SEED,
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

    n_patients = result["n_patients"]
    n_events = result["n_events"]
    step_days = result["step_days"]
    curve = result["curve"]

    print(f"\nResults: {n_patients} patients, {n_events} events, {step_days}-day steps")
    if result.get("noise_type", "none") != "none":
        print(f"Noise applied: {result['noise_type']} (snr={result.get('snr')}, random_seed={result.get('random_seed')})")
    print_km_table(curve)
    plot_km_curve(curve, n_patients, n_events, step_days)


if __name__ == "__main__":
    main()
