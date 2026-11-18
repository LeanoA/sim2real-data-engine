"""Tests for true SE(3) ego-centric equivariance, SO(3) yaw randomization, and rot noise."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim2real_engine.benchmarks import PerceptionNoiseConfig, evaluate_ood_robustness
from sim2real_engine.cli import main
from sim2real_engine.core.contracts import DomainRandomizationConfig, JointState, SE3Pose
from sim2real_engine.core.kinematics import (
    DampedLeastSquaresIK,
    quat_canonicalize_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    so3_exp_map,
    so3_log_map,
)
from sim2real_engine.learning import (
    BehaviorCloningMLP,
    LeRobotParquetDataset,
    assemble_policy_observation,
    compute_relative_quat_tcp,
    quat_to_rot_matrix,
)
from sim2real_engine.randomization.samplers import DomainRandomizer
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert
from sim2real_engine.simulators.mujoco_native import MuJoCoPhysicsAdapter


def test_so3_exp_log_roundtrip_and_quaternion_group_ops() -> None:
    """so3_exp_map and so3_log_map form an inverse pair; quaternion ops satisfy S^3 group laws."""
    rng = np.random.default_rng(123)
    for _ in range(10):
        omega = rng.uniform(-0.8, 0.8, size=3)
        q = so3_exp_map(omega)
        r = quat_to_rot_matrix(np.asarray(q, dtype=np.float32)).astype(np.float64)
        omega_rec = so3_log_map(r)
        np.testing.assert_allclose(omega_rec, omega, atol=1e-5)

        q_inv = quat_conjugate_wxyz(q)
        q_prod = quat_multiply_wxyz(q, q_inv)
        np.testing.assert_allclose(q_prod, (1.0, 0.0, 0.0, 0.0), atol=1e-7)

        q_neg = (-q[0], -q[1], -q[2], -q[3])
        assert quat_canonicalize_wxyz(q) == pytest.approx(quat_canonicalize_wxyz(q_neg), abs=1e-9)


def test_ego_centric_observation_is_strictly_se3_invariant() -> None:
    """20D ego-centric observation and policy remain invariant under global SE(3) transforms."""
    q_norm = np.zeros(6, dtype=np.float32)
    v_norm = np.zeros(6, dtype=np.float32)
    grip = np.array([0.0], dtype=np.float32)

    ee_pos = np.array([0.38, 0.05, 0.20], dtype=np.float32)
    ee_quat = np.asarray(so3_exp_map(np.array([0.1, -0.2, 0.4])), dtype=np.float32)
    obj_pos = np.array([0.41, 0.02, 0.07], dtype=np.float32)
    obj_quat = np.asarray(so3_exp_map(np.array([0.0, 0.0, -0.35])), dtype=np.float32)

    obs_base = assemble_policy_observation(
        q_norm,
        v_norm,
        ego_centric=True,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        obj_pos=obj_pos,
        obj_quat=obj_quat,
        gripper_state=grip,
    )
    assert obs_base.shape == (20,)

    # Apply an arbitrary global SE(3) rigid transform g = (R_g, p_g) to BOTH EE and Object
    q_g = np.asarray(so3_exp_map(np.array([0.5, -0.7, 1.1])), dtype=np.float32)
    r_g = quat_to_rot_matrix(q_g)
    p_g = np.array([4.5, -2.2, 1.8], dtype=np.float32)

    ee_pos_g = (r_g @ ee_pos) + p_g
    obj_pos_g = (r_g @ obj_pos) + p_g
    ee_quat_g = np.asarray(
        quat_multiply_wxyz(tuple(float(v) for v in q_g), tuple(float(v) for v in ee_quat)),
        dtype=np.float32,
    )
    obj_quat_g = np.asarray(
        quat_multiply_wxyz(tuple(float(v) for v in q_g), tuple(float(v) for v in obj_quat)),
        dtype=np.float32,
    )

    obs_transformed = assemble_policy_observation(
        q_norm,
        v_norm,
        ego_centric=True,
        ee_pos=ee_pos_g,
        ee_quat=ee_quat_g,
        obj_pos=obj_pos_g,
        obj_quat=obj_quat_g,
        gripper_state=grip,
    )
    np.testing.assert_allclose(obs_transformed, obs_base, atol=1e-5)

    # Also invariant under antipodal quaternion negation (q -> -q)
    rel_pos_orig = obs_base[12:15]
    rel_quat_neg = compute_relative_quat_tcp(-obj_quat, ee_quat)
    np.testing.assert_allclose(obs_base[15:19], rel_quat_neg, atol=1e-6)
    assert rel_pos_orig.shape == (3,)

    # Policy wrapper invariance check
    model = BehaviorCloningMLP(obs_dim=20, action_dim=7, horizon=4, hidden_dim=32, num_blocks=1)
    stats = {
        "q_mean": [0.0] * 6,
        "q_std": [1.0] * 6,
        "v_mean": [0.0] * 6,
        "v_std": [1.0] * 6,
        "act_mean": [0.0] * 7,
        "act_std": [1.0] * 7,
        "ego_centric": True,
    }
    fn = model.to_policy_fn(stats, device="cpu")
    js = JointState(
        positions_rad=(0.0,) * 6,
        velocities_rad_s=(0.0,) * 6,
        torques_nm=(0.0,) * 6,
    )
    act_a = fn(
        js,
        0,
        ee_pose=SE3Pose.from_unnormalized(tuple(ee_pos.tolist()), tuple(ee_quat.tolist())),
        object_pose=SE3Pose.from_unnormalized(tuple(obj_pos.tolist()), tuple(obj_quat.tolist())),
        gripper_state=0.0,
    )
    act_b = fn(
        js,
        0,
        ee_pose=SE3Pose.from_unnormalized(tuple(ee_pos_g.tolist()), tuple(ee_quat_g.tolist())),
        object_pose=SE3Pose.from_unnormalized(
            tuple(obj_pos_g.tolist()), tuple(obj_quat_g.tolist())
        ),
        gripper_state=0.0,
    )
    np.testing.assert_allclose(act_a, act_b, atol=1e-5)


def test_ego_centric_dataset_and_cli_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dataset ego_centric=True exposes 20D observations and CLI --ego-centric persists flag."""
    ds_dir = Path("datasets/sim2real_pick_place")
    if not ds_dir.is_dir():
        pytest.skip("Dataset 'datasets/sim2real_pick_place' not found.")

    ds = LeRobotParquetDataset(ds_dir, horizon=2, ego_centric=True)
    assert ds.obs_dim == 20
    assert ds.get_stats_dict()["ego_centric"] is True
    obs, _ = ds[0]
    assert obs.shape == (20,)

    ckpt_path = tmp_path / "ego_model.pt"
    rc = main([
        "train-policy",
        "--dataset-dir", str(ds_dir),
        "--epochs", "1",
        "--batch-size", "128",
        "--horizon", "2",
        "--ego-centric",
        "--output-model", str(ckpt_path),
        "--device", "cpu",
        "--json",
    ])
    assert rc == 0
    capsys.readouterr()

    loaded, meta = BehaviorCloningMLP.load_checkpoint(ckpt_path, device="cpu")
    assert loaded.obs_dim == 20
    assert meta.get("ego_centric") is True


def test_object_yaw_randomization_and_expert_grasp() -> None:
    """SO(3) object yaw perturbation rotates payload and privileged IK expert still succeeds."""
    cfg_yaw = DomainRandomizationConfig(seed=42, object_yaw_range_rad=(-0.45, 0.45))
    pert = DomainRandomizer(cfg_yaw).sample_episode(0)
    assert -0.45 <= pert.object_yaw_rad <= 0.45

    # Fixed 0.4 rad yaw test on MuJoCoPhysicsAdapter with PrivilegedPickPlaceExpert
    fixed_cfg = DomainRandomizationConfig(seed=7, object_yaw_range_rad=(0.4, 0.4))
    pert_fixed = DomainRandomizer(fixed_cfg).sample_episode(0)
    assert pert_fixed.object_yaw_rad == pytest.approx(0.4, abs=1e-9)

    sim = MuJoCoPhysicsAdapter()
    _, _, obj_pose = sim.reset(seed=42, perturbation=pert_fixed)
    expected_qw = math.cos(0.2)
    expected_qz = math.sin(0.2)
    np.testing.assert_allclose(
        obj_pose.quaternion_wxyz, (expected_qw, 0.0, 0.0, expected_qz), atol=1e-5
    )

    expert = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=sim.dof))
    records = expert.run_episode(backend=sim, seed=42, max_steps=120, perturbation=pert_fixed)
    assert records[-1].is_terminal is True


def test_rotational_perception_noise_injection(capsys: pytest.CaptureFixture[str]) -> None:
    """SO(3) rotational perception noise executes in eval loop and CLI."""
    report = evaluate_ood_robustness(
        model_path="checkpoints/bc_pick_place.pt",
        backend="mujoco_native",
        num_episodes=2,
        base_seed=42,
        max_steps=10,
        perception_noise=PerceptionNoiseConfig(rot_noise_std_rad=0.35, noise_mode="per_episode"),
    )
    assert report.nominal.num_episodes == 2

    rc = main([
        "eval-policy",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--perception-rot-noise-deg", "15.0",
        "--episodes", "1",
        "--max-steps", "8",
        "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_episodes"] == 1
