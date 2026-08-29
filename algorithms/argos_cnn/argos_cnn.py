"""Federated training driver for the ModResNet CT segmentation architecture
(see model.py), wired into vantage6's v4 algorithm-tools SDK — same
central/partial + @data(1) DataFrame pattern as logistic_regression.py and
average.py in this folder.

Expected per-node database (the DataFrame @data(1) injects into `partial`) is
a per-slice manifest, not raw NIfTI folders — one row per axial CT slice:

    patient_id   : groups slices belonging to one scan
    slice_index  : int, position of the slice within the patient's scan
    ct_path      : path (inside the node container) to that slice's CT NIfTI
    gt_path      : path to the matching GTV (tumor) ground-truth NIfTI
    has_tumor    : 0/1, whether gt_path's slice contains any positive voxels

This mirrors sort_slices()/get_batch_full() from the original argosfeddeep
repo, reshaped from a per-patient JSON dict into a flat table so it fits
vantage6's @data(1) contract. Building this manifest is out of scope here.

Known limitation: unlike logistic_regression.py's 4-float state dict,
ModResNet has ~12-13M parameters. Its weights are gzip+base64 encoded before
going into the vantage6 task result/input, but that can still be tens of MB
per node per round — verify your vantage6 deployment's result-size limits
before running this for real. The original repo avoided this entirely by
shipping weights through a separate Flask upload/download service
(master_api.py) outside vantage6's task payloads.

Known, deliberate differences from the original training recipe (not bugs —
left as-is on purpose):
  - No data augmentation. The original's get_batch_full() applies random
    flips/rotation/blur/gamma/shear (data_augmentation.py); _sample_batch()
    here does not.
  - FedAvg aggregation is weighted by each node's dataset size (see
    central()); the original's average.py does a plain unweighted np.mean
    across nodes regardless of how much data each one has.
"""
import base64
import gzip
import io
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

from model import ModResNet

PATIENT_COL = "patient_id"
SLICE_COL = "slice_index"
CT_COL = "ct_path"
GT_COL = "gt_path"
LABEL_COL = "has_tumor"
REQUIRED_COLS = {PATIENT_COL, SLICE_COL, CT_COL, GT_COL, LABEL_COL}

NUM_CLASSES = 2
NUM_CHANNELS = 3   # 2.5D: stack of 3 adjacent axial slices, matches params.json's patch_shape
PATCH_SIZE = 512
MIN_BOUND = -800   # HU windowing bounds, matches the original repo's params.json
MAX_BOUND = 200
L2_LOSS = 1e-4     # matches params.json's l2_loss (selective kernel L2 penalty, see model.py)

N_ROUNDS = 2       # smoke-test value: just enough to exercise round-to-round aggregation once
LOCAL_STEPS = 3    # smoke-test value: enough to run the train loop, not to actually learn anything
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
POSITIVE_SLICE_BIAS = 1 / 3   # matches original get_batch_full: ~1-in-3 forced tumor-containing slice
RANDOM_SEED = 42


def _state_dict_to_str(state_dict: dict) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(gzip.compress(buffer.getvalue())).decode("ascii")


def _state_dict_from_str(serial: str) -> dict:
    compressed = base64.b64decode(serial.encode("ascii"))
    return torch.load(io.BytesIO(gzip.decompress(compressed)), map_location="cpu")


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, ignore_background: bool = False, eps: float = 1e-7) -> torch.Tensor:
    # ignore_background=False matches run_online.py's dice_loss2 (the loss actually
    # used for training). average.py has a same-named dice_loss2 with
    # ignore_background=True, but that copy is only used to instantiate a model
    # for weight-averaging (.fit() is never called there) — it's not real training
    # behavior, so it's not what this should match.
    if ignore_background:
        y_true = y_true[:, 1:, :, :]
        y_pred = y_pred[:, 1:, :, :]
    dims = (0, 2, 3)
    numerator = 2 * torch.sum(y_true * y_pred, dim=dims) + eps
    denominator = torch.sum(y_true, dim=dims) + torch.sum(y_pred, dim=dims) + eps
    return 1 - torch.mean(numerator / denominator)


def dice_bce_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # ModResNet.forward already applies softmax, so BCE takes probabilities directly
    # (not logits) — do not swap this for nn.CrossEntropyLoss.
    return dice_loss(y_true, y_pred) + F.binary_cross_entropy(y_pred, y_true)


def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor, ignore_background: bool = True, eps: float = 1e-7) -> float:
    with torch.no_grad():
        if ignore_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        y_pred_bin = (y_pred > 0.15).float()
        dims = (0, 2, 3)
        numerator = 2 * torch.sum(y_true * y_pred_bin, dim=dims) + eps
        denominator = torch.sum(y_true, dim=dims) + torch.sum(y_pred_bin, dim=dims) + eps
        return torch.mean(numerator / denominator).item()


def _normalize_ct(arr: np.ndarray) -> np.ndarray:
    """HU windowing: clip to [MIN_BOUND, MAX_BOUND] and scale to [0, 1]."""
    return np.clip((arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.0, 1.0)


def _sample_batch(df: pd.DataFrame, batch_size: int, num_channels: int, patch_size: int) -> tuple:
    half = num_channels // 2
    ct_batch = np.zeros((batch_size, num_channels, patch_size, patch_size), dtype=np.float32)
    gt_batch = np.zeros((batch_size, patch_size, patch_size), dtype=np.int64)
    patient_ids = df[PATIENT_COL].unique()

    for i in range(batch_size):
        patient_df = df[df[PATIENT_COL] == random.choice(patient_ids)].sort_values(SLICE_COL).reset_index(drop=True)
        n_slices = len(patient_df)

        positive_rows = patient_df[patient_df[LABEL_COL] == 1]
        if random.random() < POSITIVE_SLICE_BIAS and len(positive_rows) > 0:
            # random.choice (not pandas' .sample(), which draws from numpy's
            # global RNG) so this stays governed by the random.seed() call
            # above like every other draw in this function.
            center_idx = random.choice(positive_rows.index.tolist())
        else:
            center_idx = random.randrange(n_slices)

        # Clip neighbor indices at the scan boundary (repeats the edge slice)
        # rather than reproducing the original's occasional out-of-range read.
        for c in range(num_channels):
            neighbor_idx = int(np.clip(center_idx - half + c, 0, n_slices - 1))
            ct_batch[i, c] = _normalize_ct(nib.load(patient_df.loc[neighbor_idx, CT_COL]).get_fdata())

        gt_batch[i] = nib.load(patient_df.loc[center_idx, GT_COL]).get_fdata()

    return torch.from_numpy(ct_batch), torch.from_numpy(gt_batch)


@algorithm_client
def central(
    client: AlgorithmClient,
    n_rounds: int = N_ROUNDS,
    local_steps: int = LOCAL_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    orgs = client.organization.list()
    org_ids = [org["id"] for org in orgs]
    org_names = {org["id"]: org["name"] for org in orgs}

    info("=" * 60)
    info("FEDERATED MODRESNET CT SEGMENTATION")
    info("=" * 60)
    info(f"Organizations : {[org_names[oid] for oid in org_ids]}")
    info(f"Rounds        : {n_rounds}  |  Local steps : {local_steps}")
    info(f"Batch size    : {batch_size}  |  LR          : {learning_rate}")

    torch.manual_seed(RANDOM_SEED)
    global_model = ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    global_state = _state_dict_to_str(global_model.state_dict())

    round_metrics = []
    for round_num in range(1, n_rounds + 1):
        info(f"Round {round_num}/{n_rounds}: submitting to {len(org_ids)} node(s)...")
        task = client.task.create(
            input_={
                "method": "partial",
                "kwargs": {
                    "state_dict": global_state,
                    "local_steps": local_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "round_num": round_num,
                },
            },
            organizations=org_ids,
            name=f"argos_cnn_partial_round_{round_num}",
            description=f"Local ModResNet update, round {round_num}",
        )
        partials = client.wait_for_results(task["id"])

        total_n = sum(r["n"] for r in partials)
        if total_n == 0:
            info("No data returned from any node, stopping early")
            break

        for r in partials:
            info(f"  n={r['n']}, loss={r['loss']:.4f}, dice={r['dice']:.4f}")

        # Weighted FedAvg: average every floating-point tensor by each node's
        # dataset size (r["n"] = len(df) on the node, not samples trained on).
        # Non-floating buffers (e.g. BatchNorm's
        # num_batches_tracked) can't be weighted-averaged without changing
        # dtype, so they're just carried over from the first node.
        averaged = None
        for r in partials:
            local_sd = _state_dict_from_str(r["state_dict"])
            frac = r["n"] / total_n
            if averaged is None:
                averaged = {k: (frac * v.clone() if v.is_floating_point() else v.clone()) for k, v in local_sd.items()}
            else:
                for k, v in local_sd.items():
                    if v.is_floating_point():
                        averaged[k] += frac * v

        global_model.load_state_dict(averaged)
        global_state = _state_dict_to_str(global_model.state_dict())

        round_dice = sum(r["dice"] * r["n"] for r in partials) / total_n
        round_metrics.append({"round": round_num, "n": total_n, "dice": round_dice})
        info(f"  Aggregated — weighted dice={round_dice:.4f}")

    info("Training complete")
    return {
        "state_dict": global_state,
        "n_rounds_completed": len(round_metrics),
        "round_metrics": round_metrics,
    }


@data(1)
def partial(
    df: pd.DataFrame,
    state_dict: str,
    local_steps: int = LOCAL_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    round_num: int = 1,
) -> dict:
    missing = REQUIRED_COLS - set(df.columns)
    if missing or len(df) == 0:
        info(f"Dataset unusable (missing columns: {missing}, rows: {len(df)}), skipping local training")
        return {"state_dict": state_dict, "n": 0, "loss": 0.0, "dice": 0.0}

    n_rows = len(df)

    # Vary the batch-sampling seed by round: reseeding with a constant every
    # call would replay the exact same sequence of sampled batches every
    # round for a given node, starving later rounds of new training signal.
    random.seed(seed + round_num)

    model = ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(_state_dict_from_str(state_dict))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_loss, total_dice, n_samples = 0.0, 0.0, 0
    for step in range(local_steps):
        ct_batch, gt_batch = _sample_batch(df, batch_size, NUM_CHANNELS, PATCH_SIZE)
        gt_onehot = F.one_hot(gt_batch, NUM_CLASSES).permute(0, 3, 1, 2).float()

        optimizer.zero_grad()
        preds = model(ct_batch)
        # Matches the original's total_loss = regularization_loss + loss_value.
        loss = model.l2_regularization_loss(L2_LOSS) + dice_bce_loss(gt_onehot, preds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_dice += dice_score(gt_onehot, preds) * batch_size
        n_samples += batch_size

        if (step + 1) % max(1, local_steps // 5) == 0:
            info(f"  step {step + 1}/{local_steps}: loss={loss.item():.4f}")

    avg_loss = total_loss / n_samples
    avg_dice = total_dice / n_samples
    info(f"Local training done: {n_samples} samples seen, avg_loss={avg_loss:.4f}, avg_dice={avg_dice:.4f}")

    return {
        "state_dict": _state_dict_to_str(model.state_dict()),
        # dataset size (not samples trained on) — this is what central() uses
        # to weight FedAvg, so it must reflect each node's actual data volume.
        "n": n_rows,
        "loss": avg_loss,
        "dice": avg_dice,
    }
