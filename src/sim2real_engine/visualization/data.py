"""Telemetry and kinematic data collectors for 3D trajectory and Sim-to-Sim visualization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.benchmarks.sim2sim_eval import Sim2SimMetrics, evaluate_sim2sim_gap
from sim2real_engine.core.contracts import BackendName
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert
from sim2real_engine.simulators.cloud_adapters import GenesisAdapter, IsaacLabAdapter
from sim2real_engine.simulators.mujoco_adapter import (
    MuJoCoKinematicAdapter,
    _rot_x,
    _rot_y,
    _rot_z,
)


class ArmLinkPoints(BaseModel):
    """3D Cartesian joint coordinates for an articulated 6-DOF manipulator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base: tuple[float, float, float] = Field(..., description="Base pedestal coordinates (m).")
    shoulder: tuple[float, float, float] = Field(..., description="Shoulder joint coordinates (m).")
    elbow: tuple[float, float, float] = Field(..., description="Elbow joint coordinates (m).")
    wrist: tuple[float, float, float] = Field(..., description="Wrist joint coordinates (m).")
    end_effector: tuple[float, float, float] = Field(
        ...,
        description="End-effector tip coordinates (m).",
    )

    def to_coordinates_tuple(self) -> tuple[list[float], list[float], list[float]]:
        """Return (xs, ys, zs) lists suitable for 3D line plotting."""
        xs = [self.base[0], self.shoulder[0], self.elbow[0], self.wrist[0], self.end_effector[0]]
        ys = [self.base[1], self.shoulder[1], self.elbow[1], self.wrist[1], self.end_effector[1]]
        zs = [self.base[2], self.shoulder[2], self.elbow[2], self.wrist[2], self.end_effector[2]]
        return xs, ys, zs


def compute_arm_links(q: Sequence[float] | NDArray[np.float64]) -> ArmLinkPoints:
    """Compute 3D coordinates of all articulated arm segments from 6-DOF joint angles."""
    if len(q) != 6:
        msg = f"Expected 6 joint angles; got {len(q)}"
        raise ValueError(msg)

    q_arr = np.asarray(q, dtype=np.float64)
    base_height = 0.22
    l1, l2, l3 = 0.28, 0.26, 0.10

    p_base = (0.0, 0.0, 0.0)
    p_shoulder = (0.0, 0.0, base_height)

    r01 = _rot_z(float(q_arr[0]))
    r12 = r01 @ _rot_y(float(q_arr[1]))
    p_elbow_arr = np.array(p_shoulder, dtype=np.float64) + r12 @ np.array(
        [l1, 0.0, 0.0], dtype=np.float64
    )

    r23 = r12 @ _rot_y(float(q_arr[2]))
    p_wrist_arr = p_elbow_arr + r23 @ np.array([l2, 0.0, 0.0], dtype=np.float64)

    r34 = r23 @ _rot_x(float(q_arr[3]))
    r45 = r34 @ _rot_y(float(q_arr[4]))
    r_ee = r45 @ _rot_z(float(q_arr[5]))
    p_ee_arr = p_wrist_arr + r_ee @ np.array([l3, 0.0, 0.0], dtype=np.float64)

    return ArmLinkPoints(
        base=p_base,
        shoulder=p_shoulder,
        elbow=(float(p_elbow_arr[0]), float(p_elbow_arr[1]), float(p_elbow_arr[2])),
        wrist=(float(p_wrist_arr[0]), float(p_wrist_arr[1]), float(p_wrist_arr[2])),
        end_effector=(float(p_ee_arr[0]), float(p_ee_arr[1]), float(p_ee_arr[2])),
    )


class EpisodeVisualizationData(BaseModel):
    """Complete 3D trajectory and kinematic state sequence for a pick-and-place episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int
    timesteps_s: list[float]
    ee_trajectory: list[tuple[float, float, float]]
    object_initial_pose: tuple[float, float, float]
    object_final_pose: tuple[float, float, float]
    waypoints: list[tuple[str, tuple[float, float, float]]]
    key_arm_poses: dict[str, ArmLinkPoints]


def collect_trajectory_data(seed: int = 42, max_steps: int = 120) -> EpisodeVisualizationData:
    """Execute a deterministic privileged expert rollout and capture spatial telemetry."""
    adapter = MuJoCoKinematicAdapter()
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    records = expert.run_episode(backend=adapter, seed=seed, max_steps=max_steps)
    if not records:
        msg = "No episode records produced."
        raise RuntimeError(msg)

    timesteps = [r.timestamp_s for r in records]
    ee_traj = [r.end_effector_pose.position for r in records]
    obj_init = records[0].privileged_object_pose.position
    obj_final = records[-1].privileged_object_pose.position

    # Extract distinct phases for key arm skeleton poses
    key_poses: dict[str, ArmLinkPoints] = {
        "Initial Home": compute_arm_links(records[0].joint_state.positions_rad),
    }

    # Find grasp index (first time object is marked grasped)
    grasp_idx = len(records) // 3
    for idx, rec in enumerate(records):
        dist = np.linalg.norm(
            np.asarray(rec.end_effector_pose.position) - np.asarray(obj_init)
        )
        if dist < 0.025:
            grasp_idx = idx
            break

    key_poses["Grasp Approach"] = compute_arm_links(records[grasp_idx].joint_state.positions_rad)
    key_poses["Final Place"] = compute_arm_links(records[-1].joint_state.positions_rad)

    # Waypoints defined in the privileged expert
    waypoints: list[tuple[str, tuple[float, float, float]]] = [
        ("Home Pose", records[0].end_effector_pose.position),
        ("Object Target", obj_init),
        ("Place Target", (0.35, -0.15, 0.15)),
    ]

    return EpisodeVisualizationData(
        episode_id=seed,
        timesteps_s=timesteps,
        ee_trajectory=ee_traj,
        object_initial_pose=obj_init,
        object_final_pose=obj_final,
        waypoints=waypoints,
        key_arm_poses=key_poses,
    )


class Sim2SimComparisonData(BaseModel):
    """Synchronized cross-engine rollout comparison data and divergence curves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    timesteps_s: list[float]
    ee_trajectories: dict[BackendName, list[tuple[float, float, float]]]
    cartesian_errors_m: dict[str, list[float]]
    joint_rmses_rad: dict[str, list[float]]
    metrics: dict[str, Sim2SimMetrics]


def collect_sim2sim_comparison_data(seed: int = 42, max_steps: int = 120) -> Sim2SimComparisonData:
    """Execute synchronized rollouts across MuJoCo, Genesis, and Isaac Lab and quantify drift."""
    mujoco_sim = MuJoCoKinematicAdapter()
    genesis_sim = GenesisAdapter()
    isaac_sim = IsaacLabAdapter()

    expert_mujoco = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=mujoco_sim.dof))
    expert_genesis = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=genesis_sim.dof))
    expert_isaac = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=isaac_sim.dof))

    rec_mujoco = expert_mujoco.run_episode(backend=mujoco_sim, seed=seed, max_steps=max_steps)
    rec_genesis = expert_genesis.run_episode(backend=genesis_sim, seed=seed, max_steps=max_steps)
    rec_isaac = expert_isaac.run_episode(backend=isaac_sim, seed=seed, max_steps=max_steps)

    horizon = min(len(rec_mujoco), len(rec_genesis), len(rec_isaac))
    timesteps = [rec_mujoco[i].timestamp_s for i in range(horizon)]

    ee_trajs: dict[BackendName, list[tuple[float, float, float]]] = {
        "mujoco": [rec_mujoco[i].end_effector_pose.position for i in range(horizon)],
        "genesis": [rec_genesis[i].end_effector_pose.position for i in range(horizon)],
        "isaac_lab": [rec_isaac[i].end_effector_pose.position for i in range(horizon)],
    }

    cartesian_errors: dict[str, list[float]] = {"genesis": [], "isaac_lab": []}
    joint_rmses: dict[str, list[float]] = {"genesis": [], "isaac_lab": []}

    for i in range(horizon):
        p_ref = np.asarray(ee_trajs["mujoco"][i])
        p_gen = np.asarray(ee_trajs["genesis"][i])
        p_isa = np.asarray(ee_trajs["isaac_lab"][i])

        cartesian_errors["genesis"].append(float(np.linalg.norm(p_gen - p_ref)))
        cartesian_errors["isaac_lab"].append(float(np.linalg.norm(p_isa - p_ref)))

        q_ref = np.asarray(rec_mujoco[i].joint_state.positions_rad)
        q_gen = np.asarray(rec_genesis[i].joint_state.positions_rad)
        q_isa = np.asarray(rec_isaac[i].joint_state.positions_rad)

        joint_rmses["genesis"].append(float(np.sqrt(np.mean((q_gen - q_ref) ** 2))))
        joint_rmses["isaac_lab"].append(float(np.sqrt(np.mean((q_isa - q_ref) ** 2))))

    # Compute overall benchmark summary metrics
    metric_gen = evaluate_sim2sim_gap("mujoco", "genesis", num_episodes=5, seed=seed)
    metric_isa = evaluate_sim2sim_gap("mujoco", "isaac_lab", num_episodes=5, seed=seed)

    return Sim2SimComparisonData(
        seed=seed,
        timesteps_s=timesteps,
        ee_trajectories=ee_trajs,
        cartesian_errors_m=cartesian_errors,
        joint_rmses_rad=joint_rmses,
        metrics={"genesis": metric_gen, "isaac_lab": metric_isa},
    )
