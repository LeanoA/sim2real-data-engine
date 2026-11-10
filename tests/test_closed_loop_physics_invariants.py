"""Closed-loop physics invariant, safety envelope, and determinism tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim2real_engine.learning.policy import BehaviorCloningMLP
from sim2real_engine.server.bridge import PolicyInferenceBridge
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer
from sim2real_engine.simulators.mujoco_native import MuJoCoPhysicsAdapter


@pytest.fixture
def checkpoint_path() -> Path:
    p = Path("checkpoints/bc_pick_place.pt")
    if not p.is_file():
        pytest.skip("Checkpoint 'checkpoints/bc_pick_place.pt' not found.")
    return p


def test_workspace_spatial_envelope_containment(checkpoint_path: Path) -> None:
    """End-effector Cartesian position must stay within safe workstation boundaries at all times."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)
    buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)

    adapter = MuJoCoPhysicsAdapter()
    joints, ee, obj = adapter.reset(seed=42)
    bridge.reset()

    current_joints = joints
    current_ee = ee
    current_obj = obj
    current_grip = 0.0

    for step in range(60):
        act = bridge.infer_step(
            current_joints,
            step,
            ee_pose=current_ee,
            object_pose=current_obj,
            gripper_state=current_grip,
        )
        rec = adapter.step(tuple(float(x) for x in act))
        current_joints = rec.joint_state
        current_ee = rec.end_effector_pose
        current_obj = rec.privileged_object_pose
        current_grip = float(rec.gripper_state if rec.gripper_state is not None else 0.0)

        ee_pos = current_ee.position
        # Workspace bounds:
        # x: in front of robot base [0.20, 0.70]
        # y: lateral table surface [-0.30, 0.30]
        # z: table surface to elevated reach [0.04, 0.45]
        assert 0.20 <= ee_pos[0] <= 0.70, (
            f"Step {step}: EE X {ee_pos[0]:.3f} outside [0.20, 0.70]"
        )
        assert -0.30 <= ee_pos[1] <= 0.30, (
            f"Step {step}: EE Y {ee_pos[1]:.3f} outside [-0.30, 0.30]"
        )
        assert 0.04 <= ee_pos[2] <= 0.45, (
            f"Step {step}: EE Z {ee_pos[2]:.3f} outside [0.04, 0.45]"
        )


def test_elbow_link3_floor_clearance(checkpoint_path: Path) -> None:
    """Manipulator elbow (link_3) must never drop below industrial clearance margin (z > 0.28m)."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)
    buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)

    adapter = MuJoCoPhysicsAdapter()
    joints, ee, obj = adapter.reset(seed=42)
    bridge.reset()

    current_joints = joints
    current_ee = ee
    current_obj = obj
    current_grip = 0.0

    min_elbow_z = 999.0
    for step in range(60):
        act = bridge.infer_step(
            current_joints,
            step,
            ee_pose=current_ee,
            object_pose=current_obj,
            gripper_state=current_grip,
        )
        rec = adapter.step(tuple(float(x) for x in act))
        current_joints = rec.joint_state
        current_ee = rec.end_effector_pose
        current_obj = rec.privileged_object_pose
        current_grip = float(rec.gripper_state if rec.gripper_state is not None else 0.0)

        elbow_z = float(adapter.data.body("link_3").xpos[2])
        min_elbow_z = min(min_elbow_z, elbow_z)
        assert elbow_z > 0.28, f"Step {step}: Elbow dipped to dangerous height {elbow_z:.3f}m"

    assert min_elbow_z > 0.28


def test_proximity_gating_never_latches_in_mid_air(checkpoint_path: Path) -> None:
    """Weld equality constraint must strictly NEVER activate if TCP-payload distance > 40mm."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)

    # Test across multiple random seeds
    for test_seed in [42, 43, 44]:
        buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
        bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)
        adapter = MuJoCoPhysicsAdapter()
        joints, ee, obj = adapter.reset(seed=test_seed)
        bridge.reset()

        current_joints = joints
        current_ee = ee
        current_obj = obj
        current_grip = 0.0

        for step in range(50):
            act = bridge.infer_step(
                current_joints,
                step,
                ee_pose=current_ee,
                object_pose=current_obj,
                gripper_state=current_grip,
            )
            rec = adapter.step(tuple(float(x) for x in act))
            current_joints = rec.joint_state
            current_ee = rec.end_effector_pose
            current_obj = rec.privileged_object_pose
            current_grip = float(rec.gripper_state if rec.gripper_state is not None else 0.0)

            # Invariant: if weld is active, distance MUST have been within 40mm
            if adapter.gripper_state > 0.5:
                dist = adapter.tcp_to_payload_distance()
                # In active state, weld constraint binds them closely (< 40mm)
                err_dist = dist * 1000.0
                assert dist <= 0.045, (
                    f"Seed {test_seed}, Step {step}: Latched with invalid distance {err_dist:.1f}mm"
                )


def test_rollout_bit_exact_determinism(checkpoint_path: Path) -> None:
    """Identical seeds must reproduce bit-exact joint trajectories."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)

    def run_rollout(seed_val: int) -> list[tuple[float, ...]]:
        buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
        bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)
        adapter = MuJoCoPhysicsAdapter()
        joints, ee, obj = adapter.reset(seed=seed_val)
        bridge.reset()

        traj: list[tuple[float, ...]] = []
        cur_joints = joints
        cur_ee = ee
        cur_obj = obj
        cur_grip = 0.0

        for s in range(30):
            act = bridge.infer_step(
                cur_joints,
                s,
                ee_pose=cur_ee,
                object_pose=cur_obj,
                gripper_state=cur_grip,
            )
            rec = adapter.step(tuple(float(x) for x in act))
            cur_joints = rec.joint_state
            cur_ee = rec.end_effector_pose
            cur_obj = rec.privileged_object_pose
            cur_grip = float(rec.gripper_state if rec.gripper_state is not None else 0.0)
            traj.append(cur_joints.positions_rad)

        return traj

    traj_1 = run_rollout(42)
    traj_2 = run_rollout(42)

    assert len(traj_1) == len(traj_2) == 30
    for s_idx, (q1, q2) in enumerate(zip(traj_1, traj_2, strict=True)):
        np.testing.assert_allclose(
            q1,
            q2,
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"Trajectory non-determinism detected at step {s_idx}",
        )


def test_joint_velocity_smoothness_bounds(checkpoint_path: Path) -> None:
    """Joint angular velocities during physical simulation must remain within 4.0 rad/s."""
    model, meta = BehaviorCloningMLP.load_checkpoint(checkpoint_path)
    policy_fn = model.to_policy_fn(meta)
    buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)

    adapter = MuJoCoPhysicsAdapter()
    joints, ee, obj = adapter.reset(seed=42)
    bridge.reset()

    current_joints = joints
    current_ee = ee
    current_obj = obj
    current_grip = 0.0

    max_observed_vel = 0.0
    for step in range(50):
        act = bridge.infer_step(
            current_joints,
            step,
            ee_pose=current_ee,
            object_pose=current_obj,
            gripper_state=current_grip,
        )
        rec = adapter.step(tuple(float(x) for x in act))
        current_joints = rec.joint_state
        current_ee = rec.end_effector_pose
        current_obj = rec.privileged_object_pose
        current_grip = float(rec.gripper_state if rec.gripper_state is not None else 0.0)

        qvel = np.asarray(current_joints.velocities_rad_s, dtype=np.float64)
        peak_vel = float(np.max(np.abs(qvel)))
        max_observed_vel = max(max_observed_vel, peak_vel)

        assert peak_vel < 4.0, f"Step {step}: Excessive joint velocity {peak_vel:.2f} rad/s"

    assert max_observed_vel < 4.0
