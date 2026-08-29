import numpy as np
import pandas as pd
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

TIME_COL = "Survival.time"
EVENT_COL = "deadstatus.event"
STEP_DAYS = 30

NOISE_TYPES = {"none", "gaussian", "poisson"}
NOISE_TYPE = "none"
SNR = None
RANDOM_SEED = None


def _apply_gaussian_noise(df: pd.DataFrame, time_col: str, snr: float) -> pd.DataFrame:
    if snr is None or snr <= 0:
        raise ValueError("For Gaussian noise, 'snr' must be provided and > 0.")
    variance = np.var(df[time_col])
    std_dev = np.sqrt(variance / snr)
    noise = np.random.normal(0, std_dev, size=len(df))
    info(f"Applying Gaussian noise with std dev {std_dev:.4f}.")
    df[time_col] = (df[time_col] + np.round(noise)).clip(lower=0.0)
    return df


def _apply_poisson_noise(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    info("Applying Poisson noise.")
    df[time_col] = df[time_col].apply(lambda x: np.random.poisson(lam=x) if x > 0 else 0)
    return df


def add_noise_to_event_times(
    df: pd.DataFrame,
    time_col: str,
    noise_type: str = None,
    snr: float = None,
    random_seed: int = None,
) -> pd.DataFrame:
    noise_type = noise_type or "none"
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type!r}; expected one of {sorted(NOISE_TYPES)}")
    if noise_type == "none":
        return df

    df = df.copy()
    if random_seed is not None:
        np.random.seed(random_seed)

    if noise_type == "gaussian":
        return _apply_gaussian_noise(df, time_col, snr)
    return _apply_poisson_noise(df, time_col)


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    step_days: int = STEP_DAYS,
    noise_type: str = NOISE_TYPE,
    snr: float = SNR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    noise_type = noise_type or "none"
    if noise_type not in NOISE_TYPES:
        return {"error": f"Unknown noise type: {noise_type!r}; expected one of {sorted(NOISE_TYPES)}"}
    if noise_type == "gaussian" and (snr is None or snr <= 0):
        return {"error": "For Gaussian noise, 'snr' must be provided and > 0."}

    # Both phases must apply identical noise to the same node's data, or the
    # time range computed in phase 1 won't bound the noised times used in
    # phase 2. Resolving one seed here (instead of requiring the caller to
    # pass one) keeps noise fresh per run while guaranteeing the two phases
    # agree within this run.
    if noise_type != "none" and random_seed is None:
        random_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        info(f"No random_seed provided for noise injection; generated one: {random_seed}")
    noise_kwargs = {"noise_type": noise_type, "snr": snr, "random_seed": random_seed}

    orgs = client.organization.list()
    org_ids = [org["id"] for org in orgs]

    info("=" * 60)
    info("FEDERATED KAPLAN-MEIER SURVIVAL ANALYSIS")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Time column   : {time_col}")
    info(f"Event column  : {event_col}")
    info(f"Step days     : {step_days}")
    info(f"Noise type    : {noise_type}")
    if noise_type != "none":
        info(f"SNR           : {snr}")
        info(f"Random seed   : {random_seed}")

    # ── Phase 1: determine global time range ─────────────────────────────────
    info("")
    info("── PHASE 1: Determine time range ──────────────────────────")
    range_task = client.task.create(
        input_={
            "method": "get_time_range",
            "kwargs": {"time_col": time_col, "event_col": event_col, **noise_kwargs},
        },
        organizations=org_ids,
        name="km_time_range",
        description="Get local patient count and maximum survival time",
    )
    range_results = client.wait_for_results(range_task["id"])

    total_n = sum(r["n"] for r in range_results)
    global_max = max(r["max_time"] for r in range_results)
    info(f"Total patients : {total_n}")
    info(f"Global max time: {global_max:.1f} days")

    if total_n == 0:
        return {"error": "No usable data across any organization"}

    time_steps = list(range(0, int(global_max) + step_days, step_days))
    info(f"Time steps     : {len(time_steps)} steps (0 – {time_steps[-1]} days, every {step_days} days)")

    # ── Phase 2: collect event table from every node ──────────────────────────
    info("")
    info("── PHASE 2: Collect event tables ──────────────────────────")
    events_task = client.task.create(
        input_={
            "method": "compute_events",
            "kwargs": {
                "time_col": time_col,
                "event_col": event_col,
                "time_steps": time_steps,
                **noise_kwargs,
            },
        },
        organizations=org_ids,
        name="km_events",
        description="Compute at-risk counts and event counts at each time step",
    )
    events_results = client.wait_for_results(events_task["id"])

    n_steps = len(time_steps)
    global_n_risk = np.zeros(n_steps, dtype=int)
    global_n_events = np.zeros(n_steps, dtype=int)

    for r in events_results:
        global_n_risk += np.array(r["n_risk"], dtype=int)
        global_n_events += np.array(r["n_events"], dtype=int)

    total_events = int(global_n_events.sum())
    info(f"Total events across all nodes: {total_events}")

    # ── Phase 3: compute KM survival curve ───────────────────────────────────
    info("")
    info("── PHASE 3: Compute KM curve ───────────────────────────────")

    # survival[i] = P(T > time_steps[i]), computed from events in prior interval.
    # survival[0] = 1.0 by definition (all patients alive at t=0).
    # survival[i] = survival[i-1] * (1 - n_events[i-1] / n_risk[i-1])
    survival = np.ones(n_steps)
    for i in range(1, n_steps):
        n = global_n_risk[i - 1]
        d = global_n_events[i - 1]
        survival[i] = survival[i - 1] * (1 - d / n) if n > 0 else survival[i - 1]

    # Greenwood's formula for 95% CI
    greenwood = np.zeros(n_steps)
    cumsum = 0.0
    for i in range(1, n_steps):
        n = global_n_risk[i - 1]
        d = global_n_events[i - 1]
        if n > d > 0:
            cumsum += d / (n * (n - d))
        greenwood[i] = cumsum

    se = survival * np.sqrt(greenwood)
    ci_lower = np.maximum(survival - 1.96 * se, 0.0)
    ci_upper = np.minimum(survival + 1.96 * se, 1.0)

    info(f"Survival at final step ({time_steps[-1]} days): {survival[-1]:.4f}")

    curve = [
        {
            "time": int(time_steps[i]),
            "n_risk": int(global_n_risk[i]),
            "n_events": int(global_n_events[i]),
            "survival": round(float(survival[i]), 6),
            "ci_lower": round(float(ci_lower[i]), 6),
            "ci_upper": round(float(ci_upper[i]), 6),
        }
        for i in range(n_steps)
    ]

    return {
        "time_col": time_col,
        "event_col": event_col,
        "step_days": step_days,
        "n_patients": int(total_n),
        "n_events": total_events,
        "noise_type": noise_type,
        "snr": snr,
        "random_seed": random_seed,
        "curve": curve,
    }


@data(1)
def get_time_range(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    noise_type: str = NOISE_TYPE,
    snr: float = SNR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    df_clean = df.dropna(subset=[time_col, event_col])
    info(f"Dataset: {len(df)} rows — {len(df_clean)} usable after dropna")
    if len(df_clean) == 0:
        return {"n": 0, "max_time": 0.0}
    # Noise is applied after dropna so both partial calls (this one and
    # compute_events) noise the identical set of rows in the identical
    # order, which is required for the two phases to stay consistent.
    df_clean = add_noise_to_event_times(df_clean, time_col, noise_type, snr, random_seed)
    max_t = float(df_clean[time_col].max())
    n_events = int((df_clean[event_col] == 1).sum())
    info(f"Local max survival time: {max_t:.1f} days — {n_events} events in {len(df_clean)} patients")
    return {"n": len(df_clean), "max_time": max_t}


@data(1)
def compute_events(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    time_steps: list = None,
    noise_type: str = NOISE_TYPE,
    snr: float = SNR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    df_clean = df.dropna(subset=[time_col, event_col])
    info(f"Computing event table: {len(df_clean)} patients, {len(time_steps)} time steps")
    df_clean = add_noise_to_event_times(df_clean, time_col, noise_type, snr, random_seed)

    times = df_clean[time_col].to_numpy(dtype=float)
    events = df_clean[event_col].to_numpy(dtype=int)

    n_steps = len(time_steps)
    step_size = time_steps[1] - time_steps[0] if n_steps > 1 else time_steps[0]

    n_risk = []
    n_events_list = []

    for i, t in enumerate(time_steps):
        t_next = time_steps[i + 1] if i + 1 < n_steps else t + step_size
        at_risk = int((times >= t).sum())
        died = int(((times >= t) & (times < t_next) & (events == 1)).sum())
        n_risk.append(at_risk)
        n_events_list.append(died)

    info(f"Local total events: {sum(n_events_list)} — patients at risk at t=0: {n_risk[0] if n_risk else 0}")
    return {"n_risk": n_risk, "n_events": n_events_list}
