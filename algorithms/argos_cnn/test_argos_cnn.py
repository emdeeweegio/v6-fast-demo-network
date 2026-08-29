import random

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

import argos_cnn
from argos_cnn import (
    CT_COL,
    GT_COL,
    LABEL_COL,
    NUM_CLASSES,
    PATIENT_COL,
    RANDOM_SEED,
    SLICE_COL,
    _normalize_ct,
    _sample_batch,
    _state_dict_to_str,
    dice_bce_loss,
    dice_loss,
    dice_score,
)
from model import ConvResBlock, IdentityBlock, ModResNet

PATCH_SIZE = 4
NUM_CHANNELS = 3


def _make_patient_manifest(tmp_path, n_slices: int = 6, positive_slices=(1, 3, 5)) -> pd.DataFrame:
    rows = []
    for i in range(n_slices):
        ct_path = tmp_path / f"ct_{i}.nii.gz"
        gt_path = tmp_path / f"gt_{i}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((PATCH_SIZE, PATCH_SIZE), i, dtype=np.float32), np.eye(4)), str(ct_path))
        mask = np.full((PATCH_SIZE, PATCH_SIZE), 1 if i in positive_slices else 0, dtype=np.int32)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(gt_path))
        rows.append(
            {
                PATIENT_COL: "p1",
                SLICE_COL: i,
                CT_COL: str(ct_path),
                GT_COL: str(gt_path),
                LABEL_COL: 1 if i in positive_slices else 0,
            }
        )
    return pd.DataFrame(rows)


def test_positive_bias_branch_is_reproducible_given_the_same_seed(tmp_path, monkeypatch) -> None:
    # Regression test for the bug where positive_rows.sample(1) drew from
    # numpy's global RNG instead of the already-seeded `random` module, so
    # random.seed(...) had no effect on ~1/3 of batch draws (whenever the
    # positive-slice-bias branch fired). Force that branch to always fire by
    # patching random.random() to 0.0 (always < POSITIVE_SLICE_BIAS), then
    # confirm two seeded runs pick the identical tumor-containing slice.
    df = _make_patient_manifest(tmp_path)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    random.seed(123)
    ct_first, gt_first = _sample_batch(df, batch_size=1, num_channels=NUM_CHANNELS, patch_size=PATCH_SIZE)

    random.seed(123)
    ct_second, gt_second = _sample_batch(df, batch_size=1, num_channels=NUM_CHANNELS, patch_size=PATCH_SIZE)

    assert torch.equal(ct_first, ct_second)
    assert torch.equal(gt_first, gt_second)


def test_positive_bias_branch_only_selects_slices_with_a_tumor(tmp_path, monkeypatch) -> None:
    df = _make_patient_manifest(tmp_path, n_slices=6, positive_slices=(1, 3, 5))
    monkeypatch.setattr(random, "random", lambda: 0.0)  # always take the positive-bias branch

    random.seed(0)
    _, gt_batch = _sample_batch(df, batch_size=1, num_channels=NUM_CHANNELS, patch_size=PATCH_SIZE)

    # The center slice must be one that was tagged has_tumor=1 (mask value 1).
    assert gt_batch[0].unique().tolist() == [1]


def test_different_seeds_can_select_different_slices(tmp_path, monkeypatch) -> None:
    # Confirms random.seed() is actually driving the selection at all (not
    # just trivially deterministic because there's only one valid choice).
    # Every slice is tagged has_tumor=1 (all eligible for the positive-bias
    # branch) and each mask encodes its own slice index, so the chosen
    # center_idx can be read directly back out of gt_batch's pixel value.
    n_slices = 10
    rows = []
    for i in range(n_slices):
        ct_path = tmp_path / f"ct_{i}.nii.gz"
        gt_path = tmp_path / f"gt_{i}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32), np.eye(4)), str(ct_path))
        nib.save(nib.Nifti1Image(np.full((PATCH_SIZE, PATCH_SIZE), i, dtype=np.int32), np.eye(4)), str(gt_path))
        rows.append({PATIENT_COL: "p1", SLICE_COL: i, CT_COL: str(ct_path), GT_COL: str(gt_path), LABEL_COL: 1})
    df = pd.DataFrame(rows)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    seen_centers = set()
    for seed in range(10):
        random.seed(seed)
        _, gt_batch = _sample_batch(df, batch_size=1, num_channels=NUM_CHANNELS, patch_size=PATCH_SIZE)
        seen_centers.add(gt_batch[0, 0, 0].item())
    assert len(seen_centers) > 1


# ── model.py: architecture ────────────────────────────────────────────────────


def test_model_output_shape_matches_input_spatial_size() -> None:
    torch.manual_seed(0)
    for size in (32, 64):
        model = ModResNet(in_channels=3, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES, size, size)


def test_model_output_is_a_valid_softmax_distribution() -> None:
    torch.manual_seed(0)
    model = ModResNet(in_channels=3, num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert torch.all(out >= 0)
    assert torch.allclose(out.sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)


def test_model_handles_batch_size_greater_than_one() -> None:
    torch.manual_seed(0)
    model = ModResNet(in_channels=3, num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, NUM_CLASSES, 32, 32)


def test_l2_regularization_only_penalizes_identity_and_convres_block_convs() -> None:
    torch.manual_seed(0)
    model = ModResNet(in_channels=3, num_classes=NUM_CLASSES)
    with torch.no_grad():
        for module in model.modules():
            for param in module.parameters(recurse=False):
                param.zero_()

    # Zeroed everywhere: no penalty regardless of L2 lambda.
    assert model.l2_regularization_loss(1.0).item() == 0.0

    # Perturb a conv NOT covered by l2_regularization_loss (the stem) -- must
    # still be exactly zero.
    with torch.no_grad():
        model.stem_conv.weight.fill_(5.0)
    assert model.l2_regularization_loss(1.0).item() == 0.0

    # Perturb a conv that IS covered (an IdentityBlock's conv1) -- must
    # become nonzero and match the direct sum-of-squares computation.
    with torch.no_grad():
        model.stage1[0].conv1.weight.fill_(2.0)
    expected = float(torch.sum(model.stage1[0].conv1.weight.detach() ** 2))
    assert model.l2_regularization_loss(1.0).item() == pytest.approx(expected, rel=1e-5)


def test_identity_block_preserves_spatial_shape() -> None:
    torch.manual_seed(0)
    block = IdentityBlock(channels=16)
    block.eval()
    with torch.no_grad():
        out = block(torch.randn(1, 16, 20, 20))
    assert out.shape == (1, 16, 20, 20)


def test_convres_block_downsamples_and_changes_channels() -> None:
    torch.manual_seed(0)
    block = ConvResBlock(in_channels=16, out_channels=32, stride=2)
    block.eval()
    with torch.no_grad():
        out = block(torch.randn(1, 16, 20, 20))
    assert out.shape == (1, 32, 10, 10)


# ── loss functions ─────────────────────────────────────────────────────────────


def test_dice_loss_is_zero_for_a_perfect_match() -> None:
    y = torch.zeros(1, 2, 4, 4)
    y[:, 1] = 1.0  # all foreground
    assert dice_loss(y, y.clone()).item() < 1e-4


def test_dice_loss_is_near_one_for_complete_mismatch() -> None:
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 1] = 1.0
    y_pred = torch.zeros(1, 2, 4, 4)
    y_pred[:, 0] = 1.0  # entirely disagrees with y_true
    assert dice_loss(y_true, y_pred).item() > 0.99


def test_dice_loss_ignore_background_drops_the_background_channel() -> None:
    # With ignore_background=True, only channel 1+ contributes -- a
    # completely wrong background channel (0) must not affect the score if
    # the foreground channel is a perfect match.
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 1] = 1.0
    y_pred = torch.zeros(1, 2, 4, 4)
    y_pred[:, 1] = 1.0
    y_pred[:, 0] = 0.7  # wrong, but background is ignored
    assert dice_loss(y_true, y_pred, ignore_background=True).item() < 1e-4


def test_dice_bce_loss_combines_dice_and_bce() -> None:
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 1] = 1.0
    y_pred = y_true.clone()
    # Perfect match: dice component ~0, BCE component ~0 too (log(1)=0).
    assert dice_bce_loss(y_true, y_pred).item() < 1e-3


def test_dice_score_thresholds_predictions_before_scoring() -> None:
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 1] = 1.0
    y_pred = torch.zeros(1, 2, 4, 4)
    y_pred[:, 1] = 0.2  # below the 0.15 threshold's complement is irrelevant; this is ABOVE 0.15
    y_pred[:, 0] = 0.8
    score = dice_score(y_true, y_pred, ignore_background=True)
    assert score == pytest.approx(1.0)  # 0.2 > 0.15 threshold -> binarized to 1, matches y_true


def test_dice_score_zero_for_no_overlap() -> None:
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 1] = 1.0
    y_pred = torch.zeros(1, 2, 4, 4)
    y_pred[:, 1] = 0.05  # below threshold -> binarized to 0, no overlap with y_true's foreground
    score = dice_score(y_true, y_pred, ignore_background=True)
    assert score < 0.01


# ── _normalize_ct ──────────────────────────────────────────────────────────────


def test_normalize_ct_clips_and_scales_to_unit_range() -> None:
    arr = np.array([-1000.0, -800.0, -300.0, 200.0, 500.0])
    normalized = _normalize_ct(arr)
    assert normalized[0] == 0.0  # below MIN_BOUND, clipped
    assert normalized[1] == 0.0  # exactly MIN_BOUND
    assert normalized[3] == 1.0  # exactly MAX_BOUND
    assert normalized[4] == 1.0  # above MAX_BOUND, clipped
    assert 0.0 < normalized[2] < 1.0


# ── partial / central integration (small patch size for speed) ────────────────


def _make_manifest_with_shape(tmp_path, prefix: str, n_slices: int, patch_size: int) -> pd.DataFrame:
    rows = []
    for i in range(n_slices):
        ct_path = tmp_path / f"{prefix}_ct_{i}.nii.gz"
        gt_path = tmp_path / f"{prefix}_gt_{i}.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(patch_size, patch_size).astype(np.float32) * 100 - 300, np.eye(4)), str(ct_path))
        mask = np.zeros((patch_size, patch_size), dtype=np.int32)
        has_tumor = i % 3 == 0
        if has_tumor:
            mask[: patch_size // 2, : patch_size // 2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(gt_path))
        rows.append(
            {
                PATIENT_COL: f"{prefix}_patient",
                SLICE_COL: i,
                CT_COL: str(ct_path),
                GT_COL: str(gt_path),
                LABEL_COL: int(has_tumor),
            }
        )
    return pd.DataFrame(rows)


def test_partial_reports_unusable_dataset_when_required_columns_missing(monkeypatch) -> None:
    monkeypatch.setattr(argos_cnn, "PATCH_SIZE", PATCH_SIZE)
    df = pd.DataFrame({"some_other_column": [1, 2, 3]})
    torch.manual_seed(0)
    state = _state_dict_to_str(ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).state_dict())

    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="argos_cnn")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={"method": "partial", "kwargs": {"state_dict": state, "local_steps": 1, "batch_size": 1}},
        organizations=org_ids,
    )
    result = client.wait_for_results(task.get("id"))[0]
    assert result["n"] == 0
    assert result["state_dict"] == state  # echoed back unchanged, not a fresh/corrupted model


def test_partial_trains_and_returns_finite_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(argos_cnn, "PATCH_SIZE", 64)
    df = _make_manifest_with_shape(tmp_path, "a", n_slices=6, patch_size=64)
    torch.manual_seed(0)
    state = _state_dict_to_str(ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).state_dict())

    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="argos_cnn")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "partial",
            "kwargs": {"state_dict": state, "local_steps": 2, "batch_size": 1, "learning_rate": 1e-4, "round_num": 1},
        },
        organizations=org_ids,
    )
    result = client.wait_for_results(task.get("id"))[0]

    assert result["n"] == 6
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["dice"])
    assert 0.0 <= result["dice"] <= 1.0


def test_central_model_init_is_reproducible_given_the_module_seed(tmp_path, monkeypatch) -> None:
    # Regression test for the missing torch.manual_seed() bug (the same
    # class of bug fixed in logistic_regression.py): the initial global
    # model must start from the same weights given the same RANDOM_SEED.
    torch.manual_seed(RANDOM_SEED)
    first = ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    torch.manual_seed(RANDOM_SEED)
    second = ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    assert torch.equal(first.stem_conv.weight, second.stem_conv.weight)


def test_central_fedavg_weights_by_node_dataset_size_not_naively(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(argos_cnn, "PATCH_SIZE", 64)

    node_a = _make_manifest_with_shape(tmp_path, "a", n_slices=9, patch_size=64)  # large node
    node_b = _make_manifest_with_shape(tmp_path, "b", n_slices=3, patch_size=64)  # small node

    torch.manual_seed(RANDOM_SEED)
    initial_state = _state_dict_to_str(ModResNet(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES).state_dict())
    common_kwargs = {
        "state_dict": initial_state,
        "local_steps": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "round_num": 1,
    }

    def _call(df):
        client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="argos_cnn")
        org_ids = [organization["id"] for organization in client.organization.list()]
        task = client.task.create(input_={"method": "partial", "kwargs": common_kwargs}, organizations=org_ids)
        return client.wait_for_results(task.get("id"))[0]

    update_a = _call(node_a)
    update_b = _call(node_b)
    n_a, n_b = update_a["n"], update_b["n"]

    from argos_cnn import _state_dict_from_str

    sd_a = _state_dict_from_str(update_a["state_dict"])
    sd_b = _state_dict_from_str(update_b["state_dict"])
    frac_a, frac_b = n_a / (n_a + n_b), n_b / (n_a + n_b)
    weighted_stem = frac_a * sd_a["stem_conv.weight"] + frac_b * sd_b["stem_conv.weight"]
    naive_stem = 0.5 * sd_a["stem_conv.weight"] + 0.5 * sd_b["stem_conv.weight"]

    monkeypatch.setattr(argos_cnn, "RANDOM_SEED", RANDOM_SEED)
    client = MockAlgorithmClient(
        datasets=[[{"database": node_a, "input_data": {}}], [{"database": node_b, "input_data": {}}]],
        module="argos_cnn",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={"method": "central", "kwargs": {"n_rounds": 1, "local_steps": 1, "batch_size": 1, "learning_rate": 1e-4}},
        organizations=[org_ids[0]],
    )
    central_result = client.wait_for_results(task.get("id"))[0]

    from argos_cnn import _state_dict_from_str as _from_str

    actual_stem = _from_str(central_result["state_dict"])["stem_conv.weight"]
    assert torch.allclose(actual_stem, weighted_stem, atol=1e-5)
    if not torch.allclose(weighted_stem, naive_stem, atol=1e-6):
        assert not torch.allclose(actual_stem, naive_stem, atol=1e-6)


def test_central_stops_early_when_no_organization_has_usable_data() -> None:
    df = pd.DataFrame({"some_other_column": [1, 2, 3]})  # missing REQUIRED_COLS
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="argos_cnn")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={"method": "central", "kwargs": {"n_rounds": 2, "local_steps": 1, "batch_size": 1}},
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    assert result["n_rounds_completed"] == 0
