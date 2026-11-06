"""Verification suite for Phase 2: SO(3)/SE(3) kinematics, DLS IK, and MuJoCo rollouts."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import SE3Pose
from sim2real_engine.core.kinematics import (
    DampedLeastSquaresIK,
    compute_se3_error,
    quat_slerp,
    so3_log_map,
)
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert, SimulatorBackend
from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter


def test_so3_log_map_small_and_large_angles() -> None:
    identity = np.eye(3, dtype=np.float64)
    np.testing.assert_allclose(so3_log_map(identity), np.zeros(3), atol=1e-12)

    angle = 0.45
    rot_z = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    omega = so3_log_map(rot_z)
    np.testing.assert_allclose(omega, [0.0, 0.0, angle], atol=1e-9)

    rot_pi = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    omega_pi = so3_log_map(rot_pi)
    assert math.isclose(float(np.linalg.norm(omega_pi)), math.pi, rel_tol=1e-6)


def test_quat_slerp_antipodal_and_unit_norm() -> None:
    q0 = (1.0, 0.0, 0.0, 0.0)
    q1_antipodal = (-1.0, 0.0, 0.0, 0.0)
    mid = quat_slerp(q0, q1_antipodal, 0.5)
    np.testing.assert_allclose(mid, (1.0, 0.0, 0.0, 0.0), atol=1e-9)

    q_90z = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    half = quat_slerp(q0, q_90z, 0.5)
    expected_45z = (math.cos(math.pi / 8.0), 0.0, 0.0, math.sin(math.pi / 8.0))
    np.testing.assert_allclose(half, expected_45z, atol=1e-7)
    assert math.isclose(float(np.linalg.norm(half)), 1.0, abs_tol=1e-12)


def test_se3_error_and_dls_ik_singularity_robustness() -> None:
    cur = SE3Pose(position=(0.3, 0.0, 0.2), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    des = SE3Pose(position=(0.4, -0.05, 0.3), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    err = compute_se3_error(cur, des)
    np.testing.assert_allclose(err[:3], [0.1, -0.05, 0.1], atol=1e-12)
    np.testing.assert_allclose(err[3:], [0.0, 0.0, 0.0], atol=1e-12)

    ik = DampedLeastSquaresIK(dof=6, lambda_min=1e-3, lambda_max=1e-1, sigma_thresh=5e-2)
    nearly_singular_jacobian = np.zeros((6, 6), dtype=np.float64)
    nearly_singular_jacobian[0, 0] = 1e-8
    damping = ik.compute_adaptive_damping(nearly_singular_jacobian)
    assert damping == pytest.approx(1e-1, rel=1e-4)

    q_cur = np.zeros(6, dtype=np.float64)
    q_home = np.array([0.1, 0.2, -0.3, 0.0, 0.1, 0.0], dtype=np.float64)
    delta_q = ik.step(
        q_current=q_cur,
        q_home=q_home,
        current_pose=cur,
        target_pose=des,
        jacobian=nearly_singular_jacobian,
        max_step_rad=0.08,
    )
    assert np.all(np.isfinite(delta_q))
    assert float(np.max(np.abs(delta_q))) <= 0.08 + 1e-12


def test_mujoco_adapter_and_privileged_expert_convergence() -> None:
    adapter = MuJoCoKinematicAdapter()
    assert isinstance(adapter, SimulatorBackend)

    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    for episode_seed in (10, 20, 30, 40, 50):
        records = expert.run_episode(backend=adapter, seed=episode_seed, max_steps=120)
        assert len(records) > 5
        assert records[-1].is_terminal is True
        assert expert.stage == "COMPLETED"
        ee_final = np.asarray(records[-1].end_effector_pose.position, dtype=np.float64)
        obj_final = np.asarray(records[-1].privileged_object_pose.position, dtype=np.float64)
        np.testing.assert_allclose(ee_final, obj_final, atol=1e-9)


def test_cli_rollout_subcommand_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["rollout", "--backend", "mujoco", "--episodes", "2", "--seed", "42", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "mujoco"
    assert payload["num_episodes"] == 2
    assert payload["success_count"] == 2
    assert payload["success_rate"] == 1.0
    assert payload["mean_final_ee_error_m"] <= 0.015
