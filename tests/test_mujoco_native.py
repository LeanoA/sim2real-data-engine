"""Verification suite for native DeepMind MuJoCo 4.x physics and MJCF contact dynamics."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.models import get_robot_mjcf_path, load_robot_mjcf
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert, SimulatorBackend
from sim2real_engine.simulators.mujoco_native import MuJoCoPhysicsAdapter


def test_mjcf_model_compilation() -> None:
    path = get_robot_mjcf_path()
    assert path.is_file()

    model, data = load_robot_mjcf()
    assert model.nq == 15
    assert model.nv == 14
    assert model.nu == 7
    assert model.nsensor == 2


def test_mujoco_physics_adapter_protocol_conformance() -> None:
    adapter = MuJoCoPhysicsAdapter()
    assert isinstance(adapter, SimulatorBackend)
    assert adapter.dof == 6
    assert len(adapter.home_configuration) == 6


def test_mujoco_physics_adapter_reset_and_kinematics() -> None:
    adapter = MuJoCoPhysicsAdapter()
    joint_state, ee_pose, obj_pose = adapter.reset(seed=42)

    assert isinstance(joint_state, JointState)
    assert isinstance(ee_pose, SE3Pose)
    assert isinstance(obj_pose, SE3Pose)

    # Robot must initialize at home posture
    np.testing.assert_allclose(
        joint_state.positions_rad, adapter.home_configuration, atol=1e-5
    )

    # Object must initialize on workstation table surface (tabletop z=0.05, box center z=0.07)
    ox, oy, oz = obj_pose.position
    assert 0.30 <= ox <= 0.45
    assert -0.05 <= oy <= 0.15
    assert pytest.approx(oz, abs=0.01) == 0.07


def test_mujoco_physics_adapter_jacobian_dimensions_and_rank() -> None:
    adapter = MuJoCoPhysicsAdapter()
    adapter.reset(seed=42)
    jac = adapter.get_jacobian()

    assert jac.shape == (6, 6)
    # The home posture has high manipulability, full rank 6
    rank = int(np.linalg.matrix_rank(jac))
    assert rank == 6


def test_mujoco_physics_adapter_step_advances_simulation() -> None:
    adapter = MuJoCoPhysicsAdapter(control_dt=0.02, physics_substeps=10)
    adapter.reset(seed=42)

    target_q = tuple(float(v + 0.05) for v in adapter.home_configuration)
    record = adapter.step(joint_targets_rad=target_q)

    assert isinstance(record, EpisodeStepRecord)
    assert record.step_index == 1
    assert pytest.approx(record.timestamp_s, abs=1e-4) == 0.02
    assert len(record.joint_state.torques_nm) == 6
    assert record.gripper_action is None

    # Move payload site near TCP to test physical parallel-jaw contact clamping
    tcp_pos = adapter.data.site("ee_tcp").xpos.copy()
    adapter.data.joint("payload_freejoint").qpos[:3] = tcp_pos
    mujoco.mj_forward(adapter.model, adapter.data)
    assert adapter.tcp_to_payload_distance() <= 0.04

    # Test 7D action: close gripper physically onto payload cube (g = 1.0)
    target_7d_close = (*target_q, 1.0)
    for _ in range(15):
        rec_close = adapter.step(joint_targets_rad=target_7d_close)
    assert rec_close.gripper_action == 1.0
    assert rec_close.gripper_state is not None
    assert 0.20 < rec_close.gripper_state < 0.70
    assert adapter.gripper_contact_force > 1.0
    q_left = float(adapter.data.joint("finger_left_joint").qpos[0])
    q_right = float(adapter.data.joint("finger_right_joint").qpos[0])
    assert pytest.approx(q_left, abs=5e-4) == q_right

    # Test 7D action: open gripper and release contact (g = 0.0)
    target_7d_open = (*target_q, 0.0)
    for _ in range(15):
        rec_open = adapter.step(joint_targets_rad=target_7d_open)
    assert rec_open.gripper_action == 0.0
    assert rec_open.gripper_state is not None
    assert rec_open.gripper_state < 0.10
    assert adapter.gripper_contact_force == 0.0

    # Invalid dimension
    with pytest.raises(ValueError, match="action targets"):
        adapter.step(joint_targets_rad=(*target_q, 1.0, 2.0))


def test_privileged_expert_rollout_with_native_mujoco() -> None:
    adapter = MuJoCoPhysicsAdapter()
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    records = expert.run_episode(backend=adapter, seed=42, max_steps=20)
    assert len(records) > 0
    assert records[0].step_index == 0


def test_mujoco_elbow_up_floor_clearance() -> None:
    adapter = MuJoCoPhysicsAdapter()
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    records = expert.run_episode(backend=adapter, seed=42, max_steps=50)
    assert len(records) > 0
    # Link 3 (elbow) must maintain safe industrial clearance (z > 0.30m) at all times
    elbow_z = float(adapter.data.body("link_3").xpos[2])
    assert elbow_z > 0.30, f"Elbow dipped too low: {elbow_z}m"


def test_cli_rollout_with_native_mujoco(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["rollout", "--backend", "mujoco_native", "--episodes", "2", "--seed", "42"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Rollout complete (mujoco_native)" in captured.out


def test_mujoco_native_pick_and_place_full_rollout() -> None:
    adapter = MuJoCoPhysicsAdapter()
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    records = expert.run_episode(backend=adapter, seed=42, max_steps=120)
    assert len(records) > 10
    assert records[-1].is_terminal is True
    assert expert.stage == "COMPLETED"

    obj_init_z = records[0].privileged_object_pose.position[2]
    obj_final_y = records[-1].privileged_object_pose.position[1]
    obj_final_z = records[-1].privileged_object_pose.position[2]
    max_z = max(r.privileged_object_pose.position[2] for r in records)

    # Payload must lift significantly during transit
    assert max_z > obj_init_z + 0.08, f"Object did not lift: max_z={max_z}"
    # Payload must arrive at dropoff zone near y = -0.10m
    assert pytest.approx(obj_final_y, abs=0.03) == -0.10
    # Payload must rest stably on the table surface (z ~ 0.07m)
    assert pytest.approx(obj_final_z, abs=0.02) == 0.07


def test_mujoco_physics_adapter_domain_perturbation() -> None:
    from sim2real_engine.randomization.samplers import EpisodePerturbation

    adapter = MuJoCoPhysicsAdapter()
    nominal_mass = float(adapter.model.body("payload_object").mass[0])
    nominal_fric = float(adapter.model.geom("payload_geom").friction[0])
    nominal_finger_fric = float(adapter.model.geom("finger_left").friction[0])

    pert = EpisodePerturbation(
        episode_id=1,
        derived_seed=999,
        mass_scale=1.5,
        friction_scale=2.0,
        damping_scale=1.2,
        lighting_intensity=1.0,
        camera_pose_offset=SE3Pose(position=(0.0, 0.0, 0.0)),
    )

    adapter.reset(seed=10, perturbation=pert)
    new_mass = float(adapter.model.body("payload_object").mass[0])
    new_fric = float(adapter.model.geom("payload_geom").friction[0])
    new_finger_fric = float(adapter.model.geom("finger_left").friction[0])
    assert pytest.approx(new_mass) == nominal_mass * 1.5
    assert pytest.approx(new_fric) == nominal_fric * 2.0
    assert pytest.approx(new_finger_fric) == nominal_finger_fric * 2.0

    # Resetting without perturbation restores nominal values
    adapter.reset(seed=11)
    assert pytest.approx(float(adapter.model.body("payload_object").mass[0])) == nominal_mass
    assert pytest.approx(float(adapter.model.geom("payload_geom").friction[0])) == nominal_fric
    assert (
        pytest.approx(float(adapter.model.geom("finger_left").friction[0])) == nominal_finger_fric
    )


def test_mujoco_physics_adapter_empty_close_no_teleport() -> None:
    adapter = MuJoCoPhysicsAdapter()
    adapter.reset(seed=42)

    # Robot at home is far from payload (> 100 mm)
    assert adapter.tcp_to_payload_distance() > 0.10

    # Closing gripper in empty air drives fingers fully shut without moving payload
    for _ in range(15):
        rec = adapter.step((*adapter.home_configuration, 1.0))

    assert rec.gripper_state is not None
    assert adapter.gripper_state > 0.85
    assert adapter.gripper_contact_force == 0.0
    assert adapter.tcp_to_payload_distance() > 0.10
    assert pytest.approx(rec.privileged_object_pose.position[2], abs=0.005) == 0.07


def test_mujoco_native_pick_and_place_dr_coulomb_hold() -> None:
    from sim2real_engine.randomization.samplers import EpisodePerturbation

    pert = EpisodePerturbation(
        episode_id=1,
        derived_seed=99,
        mass_scale=1.5,
        friction_scale=0.6,
        damping_scale=1.2,
        lighting_intensity=1.0,
        camera_pose_offset=SE3Pose(position=(0.0, 0.0, 0.0)),
    )
    adapter = MuJoCoPhysicsAdapter()
    expert = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=adapter.dof))
    records = expert.run_episode(backend=adapter, seed=42, max_steps=120, perturbation=pert)
    assert expert.stage == "COMPLETED"
    max_z = max(r.privileged_object_pose.position[2] for r in records)
    assert max_z > records[0].privileged_object_pose.position[2] + 0.08
    assert pytest.approx(records[-1].privileged_object_pose.position[1], abs=0.03) == -0.10
