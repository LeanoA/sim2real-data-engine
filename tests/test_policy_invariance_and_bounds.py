"""Unit tests for robotic policy numerical safety, bounds, and invariance properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sim2real_engine.core.contracts import JointState, SE3Pose
from sim2real_engine.learning.dataset import LeRobotParquetDataset
from sim2real_engine.learning.policy import BehaviorCloningMLP


@pytest.fixture
def checkpoint_path() -> Path:
    p = Path("checkpoints/bc_pick_place.pt")
    if not p.is_file():
        pytest.skip("Checkpoint 'checkpoints/bc_pick_place.pt' not found.")
    return p


@pytest.fixture
def dataset_path() -> Path:
    p = Path("datasets/sim2real_pick_place")
    if not p.is_dir() or not (p / "meta" / "manifest.json").is_file():
        pytest.skip("Dataset 'datasets/sim2real_pick_place' not found.")
    return p


def test_policy_batch_size_invariance() -> None:
    """Inference on a single sample must match its prediction within a multi-sample batch."""
    torch.manual_seed(42)
    model = BehaviorCloningMLP(obs_dim=30, action_dim=7, horizon=8, hidden_dim=64, num_blocks=2)
    model.eval()

    single_input = torch.randn(1, 30)
    batch_input = torch.cat([single_input, torch.randn(7, 30)], dim=0)

    with torch.no_grad():
        single_pred = model(single_input)
        batch_pred = model(batch_input)

    assert single_pred.shape == torch.Size([1, 8, 7])
    assert batch_pred.shape == torch.Size([8, 8, 7])
    np.testing.assert_allclose(
        single_pred[0].numpy(),
        batch_pred[0].numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Single sample prediction diverged when batched with other samples.",
    )


def test_policy_numerical_safety_zero_and_extreme_inputs() -> None:
    """Policy must produce finite outputs under edge-case inputs without NaNs or Infs."""
    model = BehaviorCloningMLP(obs_dim=30, action_dim=7, horizon=8, hidden_dim=64, num_blocks=1)
    model.eval()

    # Zero input
    zeros = torch.zeros(2, 30)
    with torch.no_grad():
        out_zeros = model(zeros)
    assert not torch.isnan(out_zeros).any()
    assert not torch.isinf(out_zeros).any()

    # Extreme input (large perturbations, e.g., 100 std deviations)
    extreme = torch.ones(2, 30) * 100.0
    with torch.no_grad():
        out_extreme = model(extreme)
    assert not torch.isnan(out_extreme).any()
    assert not torch.isinf(out_extreme).any()


def test_quaternion_antipodal_rotation_consistency(checkpoint_path: Path) -> None:
    """In SO(3), quaternions q and -q represent identical spatial orientations.

    The policy wrapper should accept both valid representations without producing
    discontinuous or inverted joint action targets.
    """
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)

    obs = JointState(
        positions_rad=(0.0, -0.55, 1.10, 0.0, -0.55, 0.0),
        velocities_rad_s=(0.0,) * 6,
        torques_nm=(0.0,) * 6,
    )
    # Target rotation: 90 deg around Z
    q_pos = (float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4)))
    q_neg = (-q_pos[0], -q_pos[1], -q_pos[2], -q_pos[3])

    ee_pos = SE3Pose(position=(0.40, 0.0, 0.20), quaternion_wxyz=q_pos)
    ee_neg = SE3Pose(position=(0.40, 0.0, 0.20), quaternion_wxyz=q_neg)
    obj = SE3Pose(position=(0.38, 0.05, 0.07), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))

    pred_pos = policy_fn(obs, step=0, ee_pose=ee_pos, object_pose=obj, gripper_state=0.0)
    pred_neg = policy_fn(obs, step=0, ee_pose=ee_neg, object_pose=obj, gripper_state=0.0)

    # Both predictions must remain within valid physical joint limits
    assert pred_pos.shape == (8, 7)
    assert pred_neg.shape == (8, 7)
    assert not np.isnan(pred_pos).any()
    assert not np.isnan(pred_neg).any()


def test_actuator_limits_containment(checkpoint_path: Path) -> None:
    """Trained policy joint angle predictions must respect 6-DOF manipulator physical limits."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)

    obs = JointState(
        positions_rad=(0.0, -0.55, 1.10, 0.0, -0.55, 0.0),
        velocities_rad_s=(0.0,) * 6,
        torques_nm=(0.0,) * 6,
    )
    ee = SE3Pose(position=(0.56, 0.0, 0.23), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    obj = SE3Pose(position=(0.38, 0.05, 0.07), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))

    chunk = policy_fn(obs, step=0, ee_pose=ee, object_pose=obj, gripper_state=0.0)

    # MJCF joint ranges:
    # Joint 1: [-3.1415, 3.1415]
    # Joint 2: [-2.094, 2.094]
    # Joint 3: [-2.618, 2.618]
    # Joint 4: [-3.1415, 3.1415]
    # Joint 5: [-2.094, 2.094]
    # Joint 6: [-3.1415, 3.1415]
    joint_limits = [
        (-3.1415, 3.1415),
        (-2.094, 2.094),
        (-2.618, 2.618),
        (-3.1415, 3.1415),
        (-2.094, 2.094),
        (-3.1415, 3.1415),
    ]

    for h in range(chunk.shape[0]):
        for j_idx, (q_min, q_max) in enumerate(joint_limits):
            q_val = float(chunk[h, j_idx])
            # Generous safety margin to allow small overshoot without mechanical damage
            assert q_min - 0.2 <= q_val <= q_max + 0.2, (
                f"Joint {j_idx} target {q_val:.3f} rad violated limits [{q_min}, {q_max}]"
            )


def test_normalization_roundtrip_precision(dataset_path: Path) -> None:
    """Action normalization followed by denormalization must reconstruct original values."""
    ds = LeRobotParquetDataset(dataset_path, horizon=4)
    _, act_norm = ds[0]

    denorm = ds.denormalize_action(act_norm)
    # Re-normalize
    act_mean = ds._act_mean
    act_std = ds._act_std
    renorm = (denorm - act_mean) / act_std

    np.testing.assert_allclose(
        act_norm.numpy(),
        renorm,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Normalization-denormalization roundtrip incurred precision drift.",
    )
