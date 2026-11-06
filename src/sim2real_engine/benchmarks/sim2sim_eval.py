"""Cross-engine physics divergence and Sim-to-Sim gap benchmarking."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.core.contracts import BackendName
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert, SimulatorBackend
from sim2real_engine.simulators.cloud_adapters import GenesisAdapter, IsaacLabAdapter
from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter


class Sim2SimMetrics(BaseModel):
    """Metrics quantifying physical and kinematic divergence between simulation engines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_backend: BackendName = Field(..., description="Baseline reference simulator.")
    target_backend: BackendName = Field(..., description="Target comparison simulator.")
    num_episodes: int = Field(..., ge=1, description="Number of evaluated comparative rollouts.")
    mean_trajectory_rmse: float = Field(
        ...,
        ge=0.0,
        description="Root mean square error across joint trajectory time-series in radians.",
    )
    mean_ee_divergence_m: float = Field(
        ...,
        ge=0.0,
        description="Mean final Cartesian end-effector position error in meters.",
    )
    success_rate_reference: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Task success rate in reference engine.",
    )
    success_rate_target: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Task success rate in target engine.",
    )


def create_simulator_backend(name: BackendName) -> SimulatorBackend:
    """Instantiate a simulator adapter conforming to SimulatorBackend."""
    if name == "mujoco":
        return MuJoCoKinematicAdapter()
    if name == "genesis":
        return GenesisAdapter()
    if name == "isaac_lab":
        return IsaacLabAdapter()
    msg = f"Unknown simulator backend: {name}"
    raise ValueError(msg)


def evaluate_sim2sim_gap(
    reference: BackendName,
    target: BackendName,
    num_episodes: int = 10,
    seed: int = 42,
) -> Sim2SimMetrics:
    """Execute identical seeded expert trajectories across two backends and quantify drift."""
    if num_episodes < 1:
        msg = f"num_episodes must be >= 1; got {num_episodes}"
        raise ValueError(msg)

    ref_sim = create_simulator_backend(reference)
    tgt_sim = create_simulator_backend(target)

    ref_expert = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=ref_sim.dof))
    tgt_expert = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=tgt_sim.dof))

    episode_rmses: list[float] = []
    episode_ee_divs: list[float] = []
    ref_successes = 0
    tgt_successes = 0

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx

        # Parallel rollouts under identical initial conditions
        ref_records = ref_expert.run_episode(backend=ref_sim, seed=ep_seed, max_steps=120)
        tgt_records = tgt_expert.run_episode(backend=tgt_sim, seed=ep_seed, max_steps=120)

        if ref_records and ref_records[-1].is_terminal:
            ref_successes += 1
        if tgt_records and tgt_records[-1].is_terminal:
            tgt_successes += 1

        # Compare trajectory steps across overlapping horizon
        min_steps = min(len(ref_records), len(tgt_records))
        if min_steps > 0:
            step_sq_errors: list[float] = []
            for t in range(min_steps):
                q_ref = np.asarray(ref_records[t].joint_state.positions_rad, dtype=np.float64)
                q_tgt = np.asarray(tgt_records[t].joint_state.positions_rad, dtype=np.float64)
                diff = q_ref - q_tgt
                step_sq_errors.append(float(np.sum(diff * diff)))

            rmse = math.sqrt(float(np.mean(step_sq_errors)))
            episode_rmses.append(rmse)

            # Final end-effector Cartesian divergence
            ee_ref = np.asarray(ref_records[-1].end_effector_pose.position, dtype=np.float64)
            ee_tgt = np.asarray(tgt_records[-1].end_effector_pose.position, dtype=np.float64)
            ee_div = float(np.linalg.norm(ee_ref - ee_tgt))
            episode_ee_divs.append(ee_div)
        else:
            episode_rmses.append(0.0)
            episode_ee_divs.append(0.0)

    return Sim2SimMetrics(
        reference_backend=reference,
        target_backend=target,
        num_episodes=num_episodes,
        mean_trajectory_rmse=round(float(np.mean(episode_rmses)), 6),
        mean_ee_divergence_m=round(float(np.mean(episode_ee_divs)), 6),
        success_rate_reference=round(ref_successes / num_episodes, 4),
        success_rate_target=round(tgt_successes / num_episodes, 4),
    )
