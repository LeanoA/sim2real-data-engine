"""Unit and CLI verification tests for Phase 1 core contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import (
    DEFAULT_BACKENDS,
    DomainRandomizationConfig,
    EpisodeStepRecord,
    JointState,
    SE3Pose,
)


def test_se3_pose_default_and_homogeneous_matrix() -> None:
    pose = SE3Pose(position=(0.4, -0.1, 0.25), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    transform = pose.to_homogeneous_matrix()
    assert transform.shape == (4, 4)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], [0.4, -0.1, 0.25], atol=1e-12)


def test_se3_pose_rejects_unnormalized_or_zero_quaternion() -> None:
    with pytest.raises(ValidationError, match="unit L2 norm"):
        SE3Pose(position=(0.0, 0.0, 0.0), quaternion_wxyz=(2.0, 0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="degenerate zero-norm"):
        SE3Pose.from_unnormalized((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))

    normalized_pose = SE3Pose.from_unnormalized((0.1, 0.2, 0.3), (0.0, 0.0, 3.0, 4.0))
    np.testing.assert_allclose(normalized_pose.quaternion_wxyz, (0.0, 0.0, 0.6, 0.8), atol=1e-12)


def test_joint_state_and_episode_step_dimension_validation() -> None:
    joint_state = JointState(
        positions_rad=(0.1, -0.2, 0.3),
        velocities_rad_s=(0.0, 0.0, 0.0),
        torques_nm=(1.2, -0.8, 0.4),
    )
    assert joint_state.dof == 3

    with pytest.raises(ValidationError, match="identical DOF length"):
        JointState(
            positions_rad=(0.1, -0.2),
            velocities_rad_s=(0.0, 0.0, 0.0),
            torques_nm=(1.2, -0.8, 0.4),
        )

    record = EpisodeStepRecord(
        episode_id=0,
        step_index=1,
        timestamp_s=0.02,
        joint_state=joint_state,
        end_effector_pose=SE3Pose(),
        privileged_object_pose=SE3Pose(position=(0.5, 0.0, 0.05)),
        action_joint_targets_rad=(0.15, -0.15, 0.35),
    )
    assert record.joint_state.dof == 3
    assert record.gripper_action is None
    assert record.action == (0.15, -0.15, 0.35)

    # 7D reactive action (dof + 1)
    record_7d = EpisodeStepRecord(
        episode_id=0,
        step_index=1,
        timestamp_s=0.02,
        joint_state=joint_state,
        end_effector_pose=SE3Pose(),
        privileged_object_pose=SE3Pose(position=(0.5, 0.0, 0.05)),
        action_joint_targets_rad=(0.15, -0.15, 0.35, 1.0),
    )
    assert record_7d.gripper_action == 1.0
    assert len(record_7d.action) == 4

    with pytest.raises(ValidationError, match="Action target length"):
        EpisodeStepRecord(
            episode_id=0,
            step_index=1,
            timestamp_s=0.02,
            joint_state=joint_state,
            end_effector_pose=SE3Pose(),
            privileged_object_pose=SE3Pose(),
            action_joint_targets_rad=(0.15, -0.15),
        )

    with pytest.raises(ValidationError, match="Action target length"):
        EpisodeStepRecord(
            episode_id=0,
            step_index=1,
            timestamp_s=0.02,
            joint_state=joint_state,
            end_effector_pose=SE3Pose(),
            privileged_object_pose=SE3Pose(),
            action_joint_targets_rad=(0.15, -0.15, 0.35, 1.0, 2.0),
        )


def test_domain_randomization_config_bounds() -> None:
    cfg = DomainRandomizationConfig(seed=123, mass_scale_range=(0.8, 1.2))
    assert cfg.seed == 123

    with pytest.raises(ValidationError, match="lower bound"):
        DomainRandomizationConfig(mass_scale_range=(1.5, 0.8))

    with pytest.raises(ValidationError, match="strictly positive"):
        DomainRandomizationConfig(friction_scale_range=(0.0, 1.2))


def test_cli_backends_and_validate_config_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["backends", "--json"])
    assert exit_code == 0
    backends_out = json.loads(capsys.readouterr().out)
    names = [item["name"] for item in backends_out["backends"]]
    assert names == ["mujoco", "mujoco_native", "genesis", "isaac_lab"]
    assert DEFAULT_BACKENDS["mujoco"].requires_nvidia_gpu is False
    assert DEFAULT_BACKENDS["isaac_lab"].requires_nvidia_gpu is True

    valid_exit = main(
        [
            "validate-config",
            "--seed",
            "99",
            "--mass-scale-min",
            "0.75",
            "--mass-scale-max",
            "1.25",
            "--json",
        ]
    )
    assert valid_exit == 0
    valid_out = json.loads(capsys.readouterr().out)
    assert valid_out["valid"] is True
    assert valid_out["config"]["seed"] == 99

    invalid_exit = main(
        [
            "validate-config",
            "--mass-scale-min",
            "1.5",
            "--mass-scale-max",
            "0.8",
            "--json",
        ]
    )
    assert invalid_exit == 1
    invalid_out = json.loads(capsys.readouterr().out)
    assert invalid_out["valid"] is False
