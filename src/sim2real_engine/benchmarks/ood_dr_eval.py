"""Out-of-Distribution (OOD) Domain Randomization robustness and temporal ensembling benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.benchmarks.control_diagnostics import (
    ControlDiagnosticsSummary,
    RegimeControlDiagnostics,
    aggregate_control_diagnostics,
    compute_episode_control_diagnostics,
)
from sim2real_engine.benchmarks.sim2sim_eval import create_simulator_backend
from sim2real_engine.benchmarks.statistics import (
    BinomialCI,
    DistributionStats,
    PsychometricFit,
    fit_psychometric_logistic,
    summarize_distribution,
    wilson_score_interval,
)
from sim2real_engine.core.contracts import BackendName, DomainRandomizationConfig, SE3Pose
from sim2real_engine.core.kinematics import quat_multiply_wxyz, so3_exp_map
from sim2real_engine.learning.policy import BehaviorCloningMLP
from sim2real_engine.randomization.samplers import DomainRandomizer, EpisodePerturbation
from sim2real_engine.server.bridge import PolicyInferenceBridge
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer


def _compute_quaternion_tilt_deg(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    """Compute tilt angle from vertical Z-axis in degrees from unit quaternion."""
    _, x, y, _ = quaternion_wxyz
    # Z component of transformed vertical unit vector: R_33 = 1 - 2*(x^2 + y^2)
    r33 = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    tilt_rad = float(np.arccos(r33))
    return float(np.degrees(tilt_rad))


class StageSuccessRates(BaseModel):
    """Fine-grained breakdown of chained multi-stage manipulation progression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reach_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of successful TCP approach within 25mm of payload before grasp.",
    )
    grasp_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of stable payload lift (>5cm above table for >=10 steps).",
    )
    transport_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of successful cross-table lateral payload transport.",
    )
    place_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of final delivery within placement tolerance.",
    )
    chained_task_success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall chained completion rate (Reach * Grasp * Transport * Place).",
    )
    reach_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for reach success rate."
    )
    grasp_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for grasp success rate."
    )
    transport_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for transport success rate."
    )
    place_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for placement success rate."
    )
    chained_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for chained task success rate."
    )


class SurgicalMetrics(BaseModel):
    """High-precision robotics metrics isolating kinematics, perception, and grasp dynamics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_min_reach_error_mm: float = Field(
        ..., ge=0.0, description="Mean closest proximity of TCP to payload during approach (mm)."
    )
    mean_placement_error_mm: float = Field(
        ..., ge=0.0, description="Mean final distance of payload to target delivery position (mm)."
    )
    mean_execution_time_s: float = Field(
        ..., ge=0.0, description="Mean physical execution duration in seconds."
    )
    mean_hunting_oscillations: float = Field(
        ..., ge=0.0, description="Mean TCP Cartesian velocity reversals (hunting counter)."
    )
    stages: StageSuccessRates = Field(..., description="Chained manipulation stage success rates.")
    min_reach_error_stats: DistributionStats | None = Field(
        None,
        description="Distribution summary (mean, std, median, IQR, 95% CI) for reach error (mm).",
    )
    placement_error_stats: DistributionStats | None = Field(
        None, description="Distribution summary for final placement error (mm)."
    )
    execution_time_stats: DistributionStats | None = Field(
        None, description="Distribution summary for execution duration (s)."
    )
    hunting_oscillations_stats: DistributionStats | None = Field(
        None, description="Distribution summary for TCP hunting reversals."
    )


class OODRegimeMetrics(BaseModel):
    """Evaluation metrics for a specific simulation regime (nominal vs OOD DR)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regime: str = Field(..., description="Name of evaluation regime.")
    num_episodes: int = Field(..., ge=1, description="Number of evaluated episodes.")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Task completion success rate.")
    mean_steps: float = Field(..., ge=0.0, description="Mean steps executed per episode.")
    mean_ee_displacement_m: float = Field(
        ..., ge=0.0, description="Mean end-effector displacement in meters."
    )
    mean_payload_displacement_m: float = Field(
        ..., ge=0.0, description="Mean payload displacement in meters."
    )
    mean_action_jerk: float = Field(
        ...,
        ge=0.0,
        description="Mean 1st-order action step delta L2 norm ||a_t - a_{t-1}||_2.",
    )
    inference_latency_p50_ms: float = Field(
        ..., ge=0.0, description="Median inference latency in milliseconds."
    )
    inference_latency_p99_ms: float = Field(
        ..., ge=0.0, description="99th percentile inference latency in milliseconds."
    )
    surgical: SurgicalMetrics = Field(
        ..., description="Decoupled surgical manipulation and stage telemetry."
    )
    steps_stats: DistributionStats | None = Field(
        None, description="Distribution summary for executed episode step counts."
    )
    action_delta_stats: DistributionStats | None = Field(
        None, description="Distribution summary for per-episode mean action step deltas."
    )
    control_diagnostics: RegimeControlDiagnostics | None = Field(
        default=None,
        description="Physical joint jerk, FFT spectral chatter, energetics, contact, and slip.",
    )


class OODBenchmarkReport(BaseModel):
    """Comparative report evaluating policy robustness under ID vs True OOD perturbations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(..., description="Evaluated policy checkpoint.")
    backend: BackendName = Field(..., description="Simulator backend.")
    num_episodes: int = Field(..., ge=1, description="Number of episodes per regime.")
    nominal: OODRegimeMetrics = Field(..., description="Baseline nominal dynamics performance.")
    id_dr_ensembled: OODRegimeMetrics | None = Field(
        default=None,
        description="In-distribution held-out DR performance (training bounds, unseen seed).",
    )
    ood_dr_ensembled: OODRegimeMetrics = Field(
        ...,
        description="True OOD extrapolated physical + spatial performance (ensembled).",
    )
    ood_dr_no_ensemble: OODRegimeMetrics | None = Field(
        None,
        description="True OOD perturbation performance without temporal ensembling.",
    )
    action_jerk_reduction_pct: float | None = Field(
        None,
        description="Percentage reduction in control action step delta achieved by ensembling.",
    )
    physical_jerk_reduction_pct: float | None = Field(
        default=None,
        description="Percentage reduction in true physical joint jerk (rad/s^3) via ensembling.",
    )
    hf_power_reduction_pct: float | None = Field(
        default=None,
        description="Percentage reduction in >=5 Hz FFT action power ratio via ensembling.",
    )
    id_to_ood_gap_pp: float | None = Field(
        default=None,
        description="Chained task success drop in percentage points from ID-DR to True OOD.",
    )
    ood_config: DomainRandomizationConfig | None = Field(
        default=None,
        description="Extrapolated domain randomization bounds used for True OOD evaluation.",
    )


class PerceptionNoiseConfig(BaseModel):
    """Configuration for synthetic perception noise injected into object observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pos_noise_std_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Gaussian standard deviation for Cartesian object position noise (m).",
    )
    rot_noise_std_rad: float = Field(
        default=0.0,
        ge=0.0,
        description="Gaussian standard deviation for object orientation noise (rad).",
    )
    noise_mode: Literal["per_episode", "per_step"] = Field(
        default="per_episode",
        description="Temporal noise mode: static offset per episode vs per-step frame jitter.",
    )


class PerceptionNoisePoint(BaseModel):
    """Evaluation metrics at a specific perception noise standard deviation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sigma_mm: float = Field(..., ge=0.0, description="Injected position noise std in mm.")
    sigma_m: float = Field(..., ge=0.0, description="Injected position noise std in meters.")
    num_episodes: int = Field(..., ge=1, description="Number of evaluated episodes.")
    grasp_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Grasp latch success rate."
    )
    placement_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Placement delivery rate."
    )
    chained_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Chained task success rate."
    )
    mean_min_reach_error_mm: float = Field(
        ..., ge=0.0, description="Mean closest reach proximity (mm)."
    )
    mean_placement_error_mm: float = Field(
        ..., ge=0.0, description="Mean final placement error (mm)."
    )
    mean_hunting_oscillations: float = Field(
        ..., ge=0.0, description="Mean velocity sign reversals."
    )
    grasp_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for grasp success rate."
    )
    placement_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for placement success rate."
    )
    chained_ci: BinomialCI | None = Field(
        None, description="95% Wilson score CI for chained task success rate."
    )
    min_reach_error_stats: DistributionStats | None = Field(
        None, description="Distribution summary for closest reach proximity (mm)."
    )


class PerceptionSensitivityCurve(BaseModel):
    """Empirical sensitivity curve mapping task and grasp success across perception noise levels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(..., description="Evaluated policy checkpoint.")
    backend: BackendName = Field(..., description="Simulator backend.")
    noise_mode: str = Field(..., description="Temporal noise mode (per_episode or per_step).")
    num_episodes_per_point: int = Field(
        ..., ge=1, description="Episodes evaluated per noise level."
    )
    noise_points: list[PerceptionNoisePoint] = Field(..., description="Sampled sweep points.")
    breakdown_threshold_mm: float | None = Field(
        None,
        description="First noise sigma (mm) where grasp success drops strictly below 50%.",
    )
    zero_noise_grasp_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Grasp rate at sigma=0 mm."
    )
    mechanical_clearance_mm: float = Field(
        default=7.0,
        description="Physical gripper-payload clearance tolerance (mm) before finger collision.",
    )
    psychometric_fit: PsychometricFit | None = Field(
        None,
        description="Binomial MLE logistic fit estimating sigma_50 and degradation slope.",
    )


class EpisodeOutcome(BaseModel):
    """Per-episode binary stage outcomes and continuous kinematic error telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_seed: int
    reach_ok: bool
    grasp_ok: bool
    transport_ok: bool
    place_ok: bool
    chained_ok: bool
    min_reach_error_mm: float
    placement_error_mm: float
    execution_time_s: float
    hunting_oscillations: int


def build_in_distribution_dr_config(seed: int = 1042) -> DomainRandomizationConfig:
    """Return held-out DR config matching the exact training distribution bounds."""
    return DomainRandomizationConfig(seed=seed)


def build_default_ood_config(seed: int = 2042) -> DomainRandomizationConfig:
    """Return extrapolated True OOD physics + spatial config outside training bounds."""
    return DomainRandomizationConfig(
        seed=seed,
        mass_scale_range=(1.35, 2.25),
        friction_scale_range=(0.30, 0.55),
        damping_scale_range=(1.20, 1.60),
        workspace_x_shift_range=(-0.05, 0.05),
        workspace_y_shift_range=(0.04, 0.09),
    )


def run_regime_episodes(
    model: BehaviorCloningMLP,
    metadata: dict[str, object],
    backend_name: BackendName,
    num_episodes: int,
    base_seed: int,
    max_steps: int,
    perturbations: list[EpisodePerturbation] | None,
    enable_ensembling: bool,
    regime_name: str,
    placement_tolerance_mm: float = 20.0,
    max_tilt_deg: float = 15.0,
    perception_noise: PerceptionNoiseConfig | None = None,
    episode_seeds: Sequence[int] | None = None,
) -> tuple[OODRegimeMetrics, list[EpisodeOutcome]]:
    """Run evaluation rollouts and return regime metrics plus per-episode outcomes."""
    adapter = create_simulator_backend(backend_name)
    buffer = TemporalEnsembleBuffer(
        action_dim=model.action_dim,
        horizon=model.horizon,
        enabled=enable_ensembling,
    )
    policy_fn = model.to_policy_fn(metadata)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)

    displacements: list[float] = []
    payload_displacements: list[float] = []
    step_counts: list[int] = []
    all_jerks: list[float] = []

    # Decoupled surgical metrics tracking
    ep_reaches: list[bool] = []
    ep_grasps: list[bool] = []
    ep_transports: list[bool] = []
    ep_places: list[bool] = []
    ep_chained: list[bool] = []

    all_min_reach_mm: list[float] = []
    all_place_err_mm: list[float] = []
    all_exec_times_s: list[float] = []
    all_hunting_counts: list[int] = []
    outcomes: list[EpisodeOutcome] = []
    ep_diagnostics: list[ControlDiagnosticsSummary] = []

    z_table = 0.05

    for ep in range(num_episodes):
        ep_seed = int(episode_seeds[ep]) if episode_seeds is not None else base_seed + ep
        pert = perturbations[ep] if perturbations is not None else None
        joints, initial_ee, initial_obj = adapter.reset(seed=ep_seed, perturbation=pert)

        buffer.reset()
        bridge.reset()
        current_joints = joints
        current_ee = initial_ee
        current_obj = initial_obj
        current_gripper = 0.0

        # Seeded perception noise generator (isolated from physics perturbation seed)
        noise_rng = np.random.default_rng(ep_seed + 10007)
        ep_pos_offset = np.zeros(3, dtype=np.float64)
        ep_rot_offset = np.zeros(3, dtype=np.float64)
        if perception_noise is not None and perception_noise.noise_mode == "per_episode":
            if perception_noise.pos_noise_std_m > 0.0:
                ep_pos_offset = noise_rng.normal(0.0, perception_noise.pos_noise_std_m, size=3)
            if perception_noise.rot_noise_std_rad > 0.0:
                ep_rot_offset = noise_rng.normal(0.0, perception_noise.rot_noise_std_rad, size=3)

        initial_pos = np.asarray(initial_ee.position, dtype=np.float64)
        initial_obj_pos = np.asarray(initial_obj.position, dtype=np.float64)
        target_place = np.array([0.38, -0.10, initial_obj_pos[2]], dtype=np.float64)

        prev_act: np.ndarray | None = None
        episode_jerks: list[float] = []
        ep_actions: list[np.ndarray] = []
        ep_qvels: list[tuple[float, ...]] = []
        ep_torques: list[tuple[float, ...]] = []
        ep_ee_poses: list[SE3Pose] = []
        ep_obj_poses: list[SE3Pose] = []
        ep_grips: list[float] = []
        ep_contact_forces: list[float | None] = []

        min_reach_dist = 999.0
        consecutive_lift_steps = 0
        max_consecutive_lift = 0
        min_obj_y = 999.0

        hunting_reversals = 0
        prev_ee_pos = initial_pos
        prev_ee_vel: np.ndarray | None = None

        step_idx = 0
        for step_idx in range(max_steps):
            has_pos_noise = (
                perception_noise is not None and perception_noise.pos_noise_std_m > 0.0
            )
            has_rot_noise = (
                perception_noise is not None and perception_noise.rot_noise_std_rad > 0.0
            )
            if perception_noise is not None and (has_pos_noise or has_rot_noise):
                if perception_noise.noise_mode == "per_episode":
                    pos_off = ep_pos_offset
                    rot_off = ep_rot_offset
                else:
                    pos_off = (
                        noise_rng.normal(0.0, perception_noise.pos_noise_std_m, size=3)
                        if has_pos_noise
                        else np.zeros(3, dtype=np.float64)
                    )
                    rot_off = (
                        noise_rng.normal(0.0, perception_noise.rot_noise_std_rad, size=3)
                        if has_rot_noise
                        else np.zeros(3, dtype=np.float64)
                    )
                noisy_pos = (
                    float(current_obj.position[0] + pos_off[0]),
                    float(current_obj.position[1] + pos_off[1]),
                    float(current_obj.position[2] + pos_off[2]),
                )
                if has_rot_noise:
                    delta_q = so3_exp_map(rot_off)
                    noisy_quat = quat_multiply_wxyz(delta_q, current_obj.quaternion_wxyz)
                else:
                    noisy_quat = current_obj.quaternion_wxyz
                perceived_obj = SE3Pose.from_unnormalized(
                    position=noisy_pos,
                    quaternion_wxyz=noisy_quat,
                )
            else:
                perceived_obj = current_obj

            act_chunk = bridge.infer_step(
                current_joints,
                step_idx,
                ee_pose=current_ee,
                object_pose=perceived_obj,
                gripper_state=current_gripper,
            )
            act_step = tuple(float(x) for x in act_chunk)

            # Measure control chatter (action jerk) across first 6 manipulator joints
            act_vec = np.asarray(act_step[:6], dtype=np.float64)
            if prev_act is not None:
                jerk = float(np.linalg.norm(act_vec - prev_act))
                episode_jerks.append(jerk)
            prev_act = act_vec

            step_record = adapter.step(act_step)
            current_joints = step_record.joint_state
            current_ee = step_record.end_effector_pose
            current_obj = step_record.privileged_object_pose
            if step_record.gripper_state is not None:
                current_gripper = float(step_record.gripper_state)
            else:
                has_grip = bool(step_record.gripper_action and step_record.gripper_action > 0.5)
                current_gripper = 1.0 if has_grip else 0.0

            ep_actions.append(act_vec)
            ep_qvels.append(current_joints.velocities_rad_s)
            ep_torques.append(current_joints.torques_nm)
            ep_ee_poses.append(current_ee)
            ep_obj_poses.append(current_obj)
            ep_grips.append(current_gripper)
            ep_contact_forces.append(step_record.contact_force_n)

            # 1. Proximity for Reach stage
            tcp_pos = np.asarray(current_ee.position, dtype=np.float64)
            obj_pos = np.asarray(current_obj.position, dtype=np.float64)
            dist_tcp_obj = float(np.linalg.norm(tcp_pos - obj_pos))
            min_reach_dist = min(min_reach_dist, dist_tcp_obj)

            # 2. Lift elevation for Grasp stage (>5cm above table, held for >=10 steps)
            if current_obj.position[2] > (z_table + 0.05):
                consecutive_lift_steps += 1
                max_consecutive_lift = max(max_consecutive_lift, consecutive_lift_steps)
            else:
                consecutive_lift_steps = 0

            # 3. Lateral progress for Transport stage
            min_obj_y = min(min_obj_y, current_obj.position[1])

            # 4. End-effector velocity reversals (Hunting oscillation counter)
            ee_vel = tcp_pos - prev_ee_pos
            if prev_ee_vel is not None:
                for k in range(3):
                    if ee_vel[k] * prev_ee_vel[k] < -1e-8 and abs(ee_vel[k]) > 0.0005:
                        hunting_reversals += 1
            prev_ee_vel = ee_vel
            prev_ee_pos = tcp_pos

            if step_record.is_terminal:
                break

        final_pos = np.asarray(step_record.end_effector_pose.position, dtype=np.float64)
        final_obj_pos = np.asarray(step_record.privileged_object_pose.position, dtype=np.float64)
        disp = float(np.linalg.norm(final_pos - initial_pos))
        obj_disp = float(np.linalg.norm(final_obj_pos - initial_obj_pos))

        displacements.append(disp)
        payload_displacements.append(obj_disp)
        step_counts.append(step_idx + 1)
        if episode_jerks:
            all_jerks.append(float(np.mean(episode_jerks)))

        # Evaluate decoupled stages
        placement_err_m = float(np.linalg.norm(final_obj_pos[:2] - target_place[:2]))
        tilt_deg = _compute_quaternion_tilt_deg(step_record.privileged_object_pose.quaternion_wxyz)

        reach_ok = bool(min_reach_dist <= 0.025)
        grasp_ok = bool(max_consecutive_lift >= 10)
        transport_ok = bool(grasp_ok and (min_obj_y <= -0.05))
        place_ok = bool(
            transport_ok
            and (placement_err_m <= (placement_tolerance_mm / 1000.0))
            and (tilt_deg <= max_tilt_deg)
        )
        chained_ok = bool(reach_ok and grasp_ok and transport_ok and place_ok)

        ep_reaches.append(reach_ok)
        ep_grasps.append(grasp_ok)
        ep_transports.append(transport_ok)
        ep_places.append(place_ok)
        ep_chained.append(chained_ok)

        all_min_reach_mm.append(min_reach_dist * 1000.0)
        all_place_err_mm.append(placement_err_m * 1000.0)
        all_exec_times_s.append((step_idx + 1) * 0.02)
        all_hunting_counts.append(hunting_reversals)
        ep_diagnostics.append(
            compute_episode_control_diagnostics(
                actions=ep_actions,
                joint_velocities=ep_qvels,
                joint_torques=ep_torques,
                ee_poses=ep_ee_poses,
                obj_poses=ep_obj_poses,
                gripper_states=ep_grips,
                contact_forces=ep_contact_forces,
                dt=0.02,
            )
        )
        outcomes.append(
            EpisodeOutcome(
                episode_seed=ep_seed,
                reach_ok=reach_ok,
                grasp_ok=grasp_ok,
                transport_ok=transport_ok,
                place_ok=place_ok,
                chained_ok=chained_ok,
                min_reach_error_mm=min_reach_dist * 1000.0,
                placement_error_mm=placement_err_m * 1000.0,
                execution_time_s=(step_idx + 1) * 0.02,
                hunting_oscillations=hunting_reversals,
            )
        )

    latency_metrics = bridge.get_latency_metrics()
    mean_jerk = float(np.mean(all_jerks)) if all_jerks else 0.0

    # Aggregate surgical metrics with Wilson CIs and distribution summaries
    num_eps = max(1, num_episodes)
    reach_count = sum(1 for r in ep_reaches if r)
    grasp_count = sum(1 for g in ep_grasps if g)
    transport_count = sum(1 for t in ep_transports if t)
    place_count = sum(1 for p in ep_places if p)
    chained_count = sum(1 for c in ep_chained if c)

    stages = StageSuccessRates(
        reach_rate=round(reach_count / num_eps, 3),
        grasp_rate=round(grasp_count / num_eps, 3),
        transport_rate=round(transport_count / num_eps, 3),
        place_rate=round(place_count / num_eps, 3),
        chained_task_success_rate=round(chained_count / num_eps, 3),
        reach_ci=wilson_score_interval(reach_count, num_eps),
        grasp_ci=wilson_score_interval(grasp_count, num_eps),
        transport_ci=wilson_score_interval(transport_count, num_eps),
        place_ci=wilson_score_interval(place_count, num_eps),
        chained_ci=wilson_score_interval(chained_count, num_eps),
    )

    reach_stats = summarize_distribution(all_min_reach_mm, seed=base_seed, decimals=2)
    place_stats = summarize_distribution(all_place_err_mm, seed=base_seed, decimals=2)
    time_stats = summarize_distribution(all_exec_times_s, seed=base_seed, decimals=3)
    hunting_stats = summarize_distribution(all_hunting_counts, seed=base_seed, decimals=2)
    steps_stats = summarize_distribution(step_counts, seed=base_seed, decimals=2)
    action_delta_stats = (
        summarize_distribution(all_jerks, seed=base_seed, decimals=4) if all_jerks else None
    )

    surgical = SurgicalMetrics(
        mean_min_reach_error_mm=reach_stats.mean,
        mean_placement_error_mm=place_stats.mean,
        mean_execution_time_s=time_stats.mean,
        mean_hunting_oscillations=round(hunting_stats.mean, 1),
        stages=stages,
        min_reach_error_stats=reach_stats,
        placement_error_stats=place_stats,
        execution_time_stats=time_stats,
        hunting_oscillations_stats=hunting_stats,
    )

    regime_metrics = OODRegimeMetrics(
        regime=regime_name,
        num_episodes=num_episodes,
        success_rate=stages.chained_task_success_rate,
        mean_steps=round(steps_stats.mean, 1),
        mean_ee_displacement_m=round(float(np.mean(displacements)), 4),
        mean_payload_displacement_m=round(float(np.mean(payload_displacements)), 4),
        mean_action_jerk=round(mean_jerk, 4),
        inference_latency_p50_ms=latency_metrics.p50_ms,
        inference_latency_p99_ms=latency_metrics.p99_ms,
        surgical=surgical,
        steps_stats=steps_stats,
        action_delta_stats=action_delta_stats,
        control_diagnostics=aggregate_control_diagnostics(ep_diagnostics, seed=base_seed),
    )
    return regime_metrics, outcomes


def _evaluate_regime(
    model: BehaviorCloningMLP,
    metadata: dict[str, object],
    backend_name: BackendName,
    num_episodes: int,
    base_seed: int,
    max_steps: int,
    perturbations: list[EpisodePerturbation] | None,
    enable_ensembling: bool,
    regime_name: str,
    placement_tolerance_mm: float = 20.0,
    max_tilt_deg: float = 15.0,
    perception_noise: PerceptionNoiseConfig | None = None,
) -> OODRegimeMetrics:
    """Run evaluation rollouts under a specific dynamics and ensembling regime."""
    metrics, _ = run_regime_episodes(
        model=model,
        metadata=metadata,
        backend_name=backend_name,
        num_episodes=num_episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=perturbations,
        enable_ensembling=enable_ensembling,
        regime_name=regime_name,
        placement_tolerance_mm=placement_tolerance_mm,
        max_tilt_deg=max_tilt_deg,
        perception_noise=perception_noise,
    )
    return metrics


def evaluate_ood_robustness(
    model_path: str | Path,
    backend: BackendName = "mujoco_native",
    num_episodes: int = 5,
    base_seed: int = 42,
    max_steps: int = 120,
    compare_ensembling: bool = False,
    dr_config: DomainRandomizationConfig | None = None,
    placement_tolerance_mm: float = 20.0,
    perception_noise: PerceptionNoiseConfig | None = None,
    id_dr_config: DomainRandomizationConfig | None = None,
) -> OODBenchmarkReport:
    """Benchmark policy robustness across nominal, in-distribution DR, and True OOD regimes."""
    ckpt_path = Path(model_path)
    if not ckpt_path.is_file():
        msg = f"Policy checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)

    model, metadata = BehaviorCloningMLP.load_checkpoint(ckpt_path)

    # 1. Evaluate baseline nominal dynamics
    nominal_metrics = _evaluate_regime(
        model=model,
        metadata=metadata,
        backend_name=backend,
        num_episodes=num_episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=None,
        enable_ensembling=True,
        regime_name="nominal",
        placement_tolerance_mm=placement_tolerance_mm,
        perception_noise=perception_noise,
    )

    # 2. Evaluate held-out In-Distribution DR (training bounds, unseen seed)
    effective_id_cfg = (
        id_dr_config
        if id_dr_config is not None
        else build_in_distribution_dr_config(seed=base_seed + 1000)
    )
    id_perturbations = DomainRandomizer(config=effective_id_cfg).sample_batch(num_episodes)
    id_dr_metrics = _evaluate_regime(
        model=model,
        metadata=metadata,
        backend_name=backend,
        num_episodes=num_episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=id_perturbations,
        enable_ensembling=True,
        regime_name="id_dr_ensembled",
        placement_tolerance_mm=placement_tolerance_mm,
        perception_noise=perception_noise,
    )

    # 3. Evaluate True OOD perturbations (extrapolated physics + spatial shift)
    effective_ood_cfg = (
        dr_config if dr_config is not None else build_default_ood_config(seed=base_seed + 2000)
    )
    ood_perturbations = DomainRandomizer(config=effective_ood_cfg).sample_batch(num_episodes)

    ood_ens_metrics = _evaluate_regime(
        model=model,
        metadata=metadata,
        backend_name=backend,
        num_episodes=num_episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=ood_perturbations,
        enable_ensembling=True,
        regime_name="ood_dr_ensembled",
        placement_tolerance_mm=placement_tolerance_mm,
        perception_noise=perception_noise,
    )

    # 4. Optionally evaluate under identical True OOD perturbations without temporal ensembling
    ood_no_ens_metrics: OODRegimeMetrics | None = None
    jerk_reduction: float | None = None
    phys_jerk_reduction: float | None = None
    hf_power_reduction: float | None = None

    if compare_ensembling:
        ood_no_ens_metrics = _evaluate_regime(
            model=model,
            metadata=metadata,
            backend_name=backend,
            num_episodes=num_episodes,
            base_seed=base_seed,
            max_steps=max_steps,
            perturbations=ood_perturbations,
            enable_ensembling=False,
            regime_name="ood_dr_no_ensemble",
            placement_tolerance_mm=placement_tolerance_mm,
            perception_noise=perception_noise,
        )
        if ood_no_ens_metrics.mean_action_jerk > 0:
            diff = ood_no_ens_metrics.mean_action_jerk - ood_ens_metrics.mean_action_jerk
            jerk_reduction = round(100.0 * diff / ood_no_ens_metrics.mean_action_jerk, 2)
        d_ens = ood_ens_metrics.control_diagnostics
        d_no = ood_no_ens_metrics.control_diagnostics
        if d_ens is not None and d_no is not None:
            if d_no.mean_physical_joint_jerk_rad_s3 > 0.0:
                pdiff = (
                    d_no.mean_physical_joint_jerk_rad_s3 - d_ens.mean_physical_joint_jerk_rad_s3
                )
                phys_jerk_reduction = round(
                    100.0 * pdiff / d_no.mean_physical_joint_jerk_rad_s3, 2
                )
            if d_no.mean_hf_power_ratio > 0.0:
                hdiff = d_no.mean_hf_power_ratio - d_ens.mean_hf_power_ratio
                hf_power_reduction = round(100.0 * hdiff / d_no.mean_hf_power_ratio, 2)

    id_to_ood_gap = round((id_dr_metrics.success_rate - ood_ens_metrics.success_rate) * 100.0, 2)

    return OODBenchmarkReport(
        model_path=str(ckpt_path),
        backend=backend,
        num_episodes=num_episodes,
        nominal=nominal_metrics,
        id_dr_ensembled=id_dr_metrics,
        ood_dr_ensembled=ood_ens_metrics,
        ood_dr_no_ensemble=ood_no_ens_metrics,
        action_jerk_reduction_pct=jerk_reduction,
        physical_jerk_reduction_pct=phys_jerk_reduction,
        hf_power_reduction_pct=hf_power_reduction,
        id_to_ood_gap_pp=id_to_ood_gap,
        ood_config=effective_ood_cfg,
    )


def run_perception_sensitivity_sweep(
    model_path: str | Path,
    backend_name: BackendName = "mujoco_native",
    sigma_min_mm: float = 0.0,
    sigma_max_mm: float = 20.0,
    num_points: int = 9,
    episodes_per_point: int = 10,
    base_seed: int = 42,
    max_steps: int = 120,
    noise_mode: Literal["per_episode", "per_step"] = "per_episode",
    enable_ensembling: bool = True,
    placement_tolerance_mm: float = 20.0,
) -> PerceptionSensitivityCurve:
    """Evaluate policy across perception noise levels to map empirical sensitivity and breakdown."""
    ckpt = Path(model_path)
    if not ckpt.is_file():
        msg = f"Policy checkpoint not found: {ckpt}"
        raise FileNotFoundError(msg)

    model, metadata = BehaviorCloningMLP.load_checkpoint(ckpt)

    sigmas_mm = np.linspace(sigma_min_mm, sigma_max_mm, num_points).tolist()
    points: list[PerceptionNoisePoint] = []
    zero_grasp_rate: float = 0.0
    breakdown_thresh: float | None = None

    for idx, sigma_mm in enumerate(sigmas_mm):
        sigma_m = float(sigma_mm / 1000.0)
        p_cfg = PerceptionNoiseConfig(
            pos_noise_std_m=sigma_m,
            noise_mode=noise_mode,
        )
        sweep_seed = base_seed + (idx * 500)
        regime = _evaluate_regime(
            model=model,
            metadata=metadata,
            backend_name=backend_name,
            num_episodes=episodes_per_point,
            base_seed=sweep_seed,
            max_steps=max_steps,
            perturbations=None,
            enable_ensembling=enable_ensembling,
            regime_name=f"perception_sigma_{sigma_mm:.1f}mm",
            placement_tolerance_mm=placement_tolerance_mm,
            perception_noise=p_cfg,
        )

        pt = PerceptionNoisePoint(
            sigma_mm=round(float(sigma_mm), 2),
            sigma_m=round(sigma_m, 5),
            num_episodes=episodes_per_point,
            grasp_success_rate=regime.surgical.stages.grasp_rate,
            placement_success_rate=regime.surgical.stages.place_rate,
            chained_success_rate=regime.surgical.stages.chained_task_success_rate,
            mean_min_reach_error_mm=regime.surgical.mean_min_reach_error_mm,
            mean_placement_error_mm=regime.surgical.mean_placement_error_mm,
            mean_hunting_oscillations=regime.surgical.mean_hunting_oscillations,
            grasp_ci=regime.surgical.stages.grasp_ci,
            placement_ci=regime.surgical.stages.place_ci,
            chained_ci=regime.surgical.stages.chained_ci,
            min_reach_error_stats=regime.surgical.min_reach_error_stats,
        )
        points.append(pt)

        if idx == 0:
            zero_grasp_rate = pt.grasp_success_rate

        # Grasp breakdown occurs when grasp success drops strictly below 50%
        if breakdown_thresh is None and pt.grasp_success_rate < 0.50:
            breakdown_thresh = round(float(sigma_mm), 2)

    psychometric = fit_psychometric_logistic(
        sigmas_mm=[pt.sigma_mm for pt in points],
        successes=[
            pt.grasp_ci.successes
            if pt.grasp_ci is not None
            else int(round(pt.grasp_success_rate * pt.num_episodes))
            for pt in points
        ],
        trials=[pt.num_episodes for pt in points],
    )

    return PerceptionSensitivityCurve(
        model_path=str(ckpt),
        backend=backend_name,
        noise_mode=noise_mode,
        num_episodes_per_point=episodes_per_point,
        noise_points=points,
        breakdown_threshold_mm=breakdown_thresh,
        zero_noise_grasp_rate=zero_grasp_rate,
        mechanical_clearance_mm=7.0,
        psychometric_fit=psychometric,
    )
