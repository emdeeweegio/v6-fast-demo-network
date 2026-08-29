import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

FEATURE_COLS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
TARGET_COL = "deadstatus.event"      
N_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.1
TRAIN_TEST_RATIO = 0.8   # fraction of each node's data used for training
BATCH_RATIO = 0.3        # fraction of the training set sampled per local update
RANDOM_SEED = 42


class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def _sd_to_list(state_dict: dict) -> dict:
    return {k: v.tolist() for k, v in state_dict.items()}


def _sd_from_list(serial: dict, n_features: int) -> dict:
    return {
        "linear.weight": torch.tensor(serial["linear.weight"], dtype=torch.float32).reshape(1, n_features),
        "linear.bias": torch.tensor(serial["linear.bias"], dtype=torch.float32).reshape(1),
    }


@algorithm_client
def central(
    client: AlgorithmClient,
    feature_cols: list = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> dict:
    t_total = time.time()
    orgs = client.organization.list()
    org_ids = [org["id"] for org in orgs]
    org_names = {org["id"]: org["name"] for org in orgs}
    n_features = len(feature_cols)

    info("=" * 60)
    info("FEDERATED LOGISTIC REGRESSION")
    info("=" * 60)
    info(f"Organizations : {[org_names[oid] for oid in org_ids]}")
    info(f"Features      : {feature_cols}")
    info(f"Target        : {target_col}")
    info(f"Rounds        : {N_ROUNDS}  |  Local epochs : {LOCAL_EPOCHS}")
    info(f"Learning rate : {LEARNING_RATE}  |  Train ratio  : {TRAIN_TEST_RATIO}")
    info(f"Batch ratio   : {BATCH_RATIO}  |  Seed         : {RANDOM_SEED}")

    # ── Phase 1: normalization statistics ────────────────────────────────────
    info("")
    info("── PHASE 1: Normalization statistics ──────────────────────")
    t0 = time.time()
    stats_task = client.task.create(
        input_={
            "method": "compute_stats",
            "kwargs": {
                "feature_cols": feature_cols,
                "target_col": target_col,
                "train_ratio": TRAIN_TEST_RATIO,
                "seed": RANDOM_SEED,
            },
        },
        organizations=org_ids,
        name="logreg_stats",
        description="Compute local feature sums for global normalization",
    )
    info(f"Stats task created (id={stats_task['id']}), waiting for {len(org_ids)} node(s)...")
    stats_results = client.wait_for_results(stats_task["id"])

    total_n = sum(r["n"] for r in stats_results)
    info(f"Received {len(stats_results)} result(s) — total training rows: {total_n}")
    if total_n == 0:
        return {"error": "No usable data across any organization"}

    global_sum = np.array([r["sum"] for r in stats_results]).sum(axis=0)
    global_sum_sq = np.array([r["sum_sq"] for r in stats_results]).sum(axis=0)
    global_mean = global_sum / total_n
    global_var = global_sum_sq / total_n - global_mean ** 2
    global_std = np.sqrt(np.maximum(global_var, 1e-8))
    info("Global normalization parameters:")
    for feat, mu, sigma in zip(feature_cols, global_mean, global_std):
        info(f"  {feat}: mean={mu:.4f}, std={sigma:.4f}")
    info(f"Phase 1 completed in {time.time() - t0:.1f}s")

    # ── Phase 2: federated training rounds ───────────────────────────────────
    info("")
    info("── PHASE 2: Federated training ─────────────────────────────")
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    global_model = LogisticRegressionModel(n_features)
    global_state = _sd_to_list(global_model.state_dict())

    for round_num in range(1, N_ROUNDS + 1):
        t_round = time.time()
        info(f"Round {round_num}/{N_ROUNDS}: submitting to {len(org_ids)} node(s)...")
        task = client.task.create(
            input_={
                "method": "partial",
                "kwargs": {
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                    "state_dict": global_state,
                    "learning_rate": LEARNING_RATE,
                    "local_epochs": LOCAL_EPOCHS,
                    "train_ratio": TRAIN_TEST_RATIO,
                    "batch_ratio": BATCH_RATIO,
                    "global_mean": global_mean.tolist(),
                    "global_std": global_std.tolist(),
                    "seed": RANDOM_SEED,
                },
            },
            organizations=org_ids,
            name=f"logreg_partial_round_{round_num}",
            description=f"Local logistic regression update, round {round_num}",
        )
        partials = client.wait_for_results(task["id"])

        train_n = sum(r["n"] for r in partials)
        if train_n == 0:
            info("No data returned from any node, stopping early")
            break

        info(f"  Node results (batch_n / final loss):")
        for r in partials:
            info(f"    batch_n={r['n']}, loss={r['loss']:.4f}")

        avg_weight = torch.zeros((1, n_features), dtype=torch.float32)
        avg_bias = torch.zeros((1,), dtype=torch.float32)
        for r in partials:
            frac = r["n"] / train_n
            local_sd = _sd_from_list(r["state_dict"], n_features)
            avg_weight += frac * local_sd["linear.weight"]
            avg_bias += frac * local_sd["linear.bias"]

        global_model.load_state_dict({"linear.weight": avg_weight, "linear.bias": avg_bias})
        global_state = _sd_to_list(global_model.state_dict())
        info(f"  Aggregated — bias={avg_bias.item():.4f}, round took {time.time() - t_round:.1f}s")

    info(f"Phase 2 completed in {time.time() - t0:.1f}s")

    # ── Phase 3: per-node evaluation ─────────────────────────────────────────
    info("")
    info("── PHASE 3: Per-node evaluation ────────────────────────────")
    t0 = time.time()
    node_eval_results = []
    total_test_n = 0
    total_correct = 0

    for oid in org_ids:
        org_name = org_names[oid]
        info(f"Evaluating {org_name} (id={oid})...")
        eval_task = client.task.create(
            input_={
                "method": "evaluate",
                "kwargs": {
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                    "state_dict": global_state,
                    "global_mean": global_mean.tolist(),
                    "global_std": global_std.tolist(),
                    "train_ratio": TRAIN_TEST_RATIO,
                    "seed": RANDOM_SEED,
                },
            },
            organizations=[oid],
            name=f"logreg_evaluate_{org_name}",
            description=f"Evaluate global model on {org_name} test set",
        )
        result = client.wait_for_results(eval_task["id"])[0]
        n, correct = result["n"], result["correct"]
        node_acc = correct / n if n > 0 else None
        acc_str = f"{node_acc:.4f}" if node_acc is not None else "N/A"
        info(f"  {org_name}: n_test={n}, correct={correct}, accuracy={acc_str}")
        node_eval_results.append({
            "org_id": oid,
            "org_name": org_name,
            "n": n,
            "correct": correct,
            "accuracy": node_acc,
        })
        total_test_n += n
        total_correct += correct

    global_accuracy = total_correct / total_test_n if total_test_n > 0 else None
    acc_str = f"{global_accuracy:.4f}" if global_accuracy is not None else "N/A"
    info(f"Global test accuracy: {acc_str} ({total_correct}/{total_test_n})")
    info(f"Phase 3 completed in {time.time() - t0:.1f}s")
    info(f"Total runtime: {time.time() - t_total:.1f}s")

    return {
        "state_dict": global_state,
        "feature_cols": feature_cols,
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
        "n_train": total_n,
        "n_test": total_test_n,
        "accuracy": global_accuracy,
        "per_node": node_eval_results,
    }


@data(1)
def compute_stats(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_ratio: float = TRAIN_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict:
    empty = {"n": 0, "sum": [0.0] * len(feature_cols), "sum_sq": [0.0] * len(feature_cols)}
    info(f"Dataset: {len(df)} total rows")
    df_clean = df.dropna(subset=feature_cols + [target_col])
    info(f"After dropna: {len(df_clean)} usable rows ({len(df) - len(df_clean)} dropped)")
    if len(df_clean) == 0:
        return empty

    train_df = df_clean.sample(frac=train_ratio, random_state=seed)
    test_n = len(df_clean) - len(train_df)
    info(f"Split: {len(train_df)} train rows, {test_n} test rows (held out)")
    for col in feature_cols:
        info(f"  {col}: min={train_df[col].min():.3f}, max={train_df[col].max():.3f}, mean={train_df[col].mean():.3f}")

    n = len(train_df)
    feat_sum = train_df[feature_cols].sum().tolist()
    feat_sum_sq = (train_df[feature_cols] ** 2).sum().tolist()
    return {"n": n, "sum": feat_sum, "sum_sq": feat_sum_sq}


@data(1)
def partial(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    state_dict: dict,
    global_mean: list,
    global_std: list,
    learning_rate: float = LEARNING_RATE,
    local_epochs: int = LOCAL_EPOCHS,
    train_ratio: float = TRAIN_TEST_RATIO,
    batch_ratio: float = BATCH_RATIO,
    seed: int = RANDOM_SEED,
) -> dict:
    df_clean = df.dropna(subset=feature_cols + [target_col])
    if len(df_clean) == 0:
        info("No usable rows after dropna, skipping update")
        return {"state_dict": state_dict, "n": 0, "loss": 0.0}

    train_df = df_clean.sample(frac=train_ratio, random_state=seed)
    batch_df = train_df.sample(frac=batch_ratio)
    n = len(batch_df)
    if n == 0:
        info("Batch is empty after sampling, skipping update")
        return {"state_dict": state_dict, "n": 0, "loss": 0.0}

    pos = int(batch_df[target_col].sum())
    info(f"Batch: {n} rows — {pos} positive, {n - pos} negative (from {len(train_df)} train rows)")

    n_features = len(feature_cols)
    mean = torch.tensor(global_mean, dtype=torch.float32)
    std = torch.tensor(global_std, dtype=torch.float32)

    X_raw = torch.tensor(batch_df[feature_cols].to_numpy(), dtype=torch.float32)
    X = (X_raw - mean) / std
    y = torch.tensor(batch_df[target_col].to_numpy(), dtype=torch.float32).unsqueeze(1)

    model = LogisticRegressionModel(n_features)
    model.load_state_dict(_sd_from_list(state_dict, n_features))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    model.train()
    loss = torch.tensor(0.0)
    for epoch in range(local_epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        info(f"  Epoch {epoch + 1}/{local_epochs}: loss={loss.item():.4f}")

    return {"state_dict": _sd_to_list(model.state_dict()), "n": n, "loss": loss.item()}


@data(1)
def evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    state_dict: dict,
    global_mean: list,
    global_std: list,
    train_ratio: float = TRAIN_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict:
    info(f"Dataset: {len(df)} total rows")
    df_clean = df.dropna(subset=feature_cols + [target_col])
    info(f"After dropna: {len(df_clean)} usable rows ({len(df) - len(df_clean)} dropped)")
    if len(df_clean) == 0:
        return {"n": 0, "correct": 0}

    train_idx = df_clean.sample(frac=train_ratio, random_state=seed).index
    test_df = df_clean.drop(train_idx)
    n = len(test_df)
    info(f"Test set: {n} rows (held out from {len(df_clean)} clean rows)")
    if n == 0:
        info("Test set is empty")
        return {"n": 0, "correct": 0}

    n_features = len(feature_cols)
    mean = torch.tensor(global_mean, dtype=torch.float32)
    std = torch.tensor(global_std, dtype=torch.float32)

    X_raw = torch.tensor(test_df[feature_cols].to_numpy(), dtype=torch.float32)
    X = (X_raw - mean) / std
    y_true = test_df[target_col].to_numpy()

    model = LogisticRegressionModel(n_features)
    model.load_state_dict(_sd_from_list(state_dict, n_features))
    model.eval()

    with torch.no_grad():
        preds = np.atleast_1d(model(X).squeeze().numpy())

    predicted = (preds >= 0.5).astype(int)
    correct = int((predicted == y_true).sum())
    pos_true = int(y_true.sum())
    pos_pred = int(predicted.sum())
    info(f"True labels : {pos_true} positive, {n - pos_true} negative")
    info(f"Predictions : {pos_pred} positive, {n - pos_pred} negative")
    info(f"Correct: {correct}/{n} = {correct / n:.4f}")

    return {"n": n, "correct": correct}
