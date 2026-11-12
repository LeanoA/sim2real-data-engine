"""Out-of-Distribution (OOD) Domain Randomization robustness and temporal ensembling benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.benchmarks.sim2sim_eval import create_simulator_backend
from sim2real_engine.core.contracts import BackendName, DomainRandomizationConfig
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
        ..., ge=0.0, description="Mean action delta L2 norm (control smoothness metric)."
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


class OODBenchmarkReport(BaseModel):
    """Comparative report evaluating policy robustness under OOD physical perturbations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(..., description="Evaluated policy checkpoint.")
    backend: BackendName = Field(..., description="Simulator backend.")
    num_episodes: int = Field(..., ge=1, description="Number of episodes per regime.")
    nominal: OODRegimeMetrics = Field(..., description="Baseline nominal dynamics performance.")
    ood_dr_ensembled: OODRegimeMetrics = Field(
        ..., description="Performance under OOD physical perturbations with temporal ensembling."
    )
    ood_dr_no_ensemble: OODRegimeMetrics | None = Field(
        None,
        description="Performance under identical OOD perturbations without temporal ensembling.",
    )
    action_jerk_reduction_pct: float | None = Field(
        None,
        description="Percentage reduction in control action chatter achieved by ensembling.",
    )


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
) -> OODRegimeMetrics:
    """Run evaluation rollouts under a specific dynamics and ensembling regime."""
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

    z_table = 0.05

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        pert = perturbations[ep] if perturbations is not None else None
        joints, initial_ee, initial_obj = adapter.reset(seed=ep_seed, perturbation=pert)

        buffer.reset()
        bridge.reset()
        current_joints = joints
        current_ee = initial_ee
        current_obj = initial_obj
        current_gripper = 0.0

        initial_pos = np.asarray(initial_ee.position, dtype=np.float64)
        initial_obj_pos = np.asarray(initial_obj.position, dtype=np.float64)
        target_place = np.array([0.38, -0.10, initial_obj_pos[2]], dtype=np.float64)

        prev_act: np.ndarray | None = None
        episode_jerks: list[float] = []

        min_reach_dist = 999.0
        consecutive_lift_steps = 0
        max_consecutive_lift = 0
        min_obj_y = 999.0

        hunting_reversals = 0
        prev_ee_pos = initial_pos
        prev_ee_vel: np.ndarray | None = None

        step_idx = 0
        for step_idx in range(max_steps):
            act_chunk = bridge.infer_step(
                current_joints,
                step_idx,
                ee_pose=current_ee,
                object_pose=current_obj,
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

    latency_metrics = bridge.get_latency_metrics()
    mean_jerk = float(np.mean(all_jerks)) if all_jerks else 0.0

    # Aggregate surgical metrics
    num_eps = max(1, num_episodes)
    stages = StageSuccessRates(
        reach_rate=round(sum(1.0 for r in ep_reaches if r) / num_eps, 3),
        grasp_rate=round(sum(1.0 for g in ep_grasps if g) / num_eps, 3),
        transport_rate=round(sum(1.0 for t in ep_transports if t) / num_eps, 3),
        place_rate=round(sum(1.0 for p in ep_places if p) / num_eps, 3),
        chained_task_success_rate=round(sum(1.0 for c in ep_chained if c) / num_eps, 3),
    )
    surgical = SurgicalMetrics(
        mean_min_reach_error_mm=round(float(np.mean(all_min_reach_mm)), 2),
        mean_placement_error_mm=round(float(np.mean(all_place_err_mm)), 2),
        mean_execution_time_s=round(float(np.mean(all_exec_times_s)), 3),
        mean_hunting_oscillations=round(float(np.mean(all_hunting_counts)), 1),
        stages=stages,
    )

    return OODRegimeMetrics(
        regime=regime_name,
        num_episodes=num_episodes,
        success_rate=stages.chained_task_success_rate,
        mean_steps=round(float(np.mean(step_counts)), 1),
        mean_ee_displacement_m=round(float(np.mean(displacements)), 4),
        mean_payload_displacement_m=round(float(np.mean(payload_displacements)), 4),
        mean_action_jerk=round(mean_jerk, 4),
        inference_latency_p50_ms=latency_metrics.p50_ms,
        inference_latency_p99_ms=latency_metrics.p99_ms,
        surgical=surgical,
    )


def evaluate_ood_robustness(
    model_path: str | Path,
    backend: BackendName = "mujoco_native",
    num_episodes: int = 5,
    base_seed: int = 42,
    max_steps: int = 120,
    compare_ensembling: bool = False,
    dr_config: DomainRandomizationConfig | None = None,
    placement_tolerance_mm: float = 20.0,
) -> OODBenchmarkReport:
    """Benchmark policy robustness across nominal vs. OOD domain randomized dynamics."""
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
    )

    # Sample deterministic OOD domain perturbations
    effective_config = (
        dr_config if dr_config is not None else DomainRandomizationConfig(seed=base_seed + 1000)
    )
    randomizer = DomainRandomizer(config=effective_config)
    perturbations = randomizer.sample_batch(num_episodes)

    # 2. Evaluate under OOD perturbations with temporal ensembling
    ood_ens_metrics = _evaluate_regime(
        model=model,
        metadata=metadata,
        backend_name=backend,
        num_episodes=num_episodes,
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=perturbations,
        enable_ensembling=True,
        regime_name="ood_dr_ensembled",
        placement_tolerance_mm=placement_tolerance_mm,
    )

    # 3. Optionally evaluate under identical perturbations without temporal ensembling
    ood_no_ens_metrics: OODRegimeMetrics | None = None
    jerk_reduction: float | None = None

    if compare_ensembling:
        ood_no_ens_metrics = _evaluate_regime(
            model=model,
            metadata=metadata,
            backend_name=backend,
            num_episodes=num_episodes,
            base_seed=base_seed,
            max_steps=max_steps,
            perturbations=perturbations,
            enable_ensembling=False,
            regime_name="ood_dr_no_ensemble",
            placement_tolerance_mm=placement_tolerance_mm,
        )
        if ood_no_ens_metrics.mean_action_jerk > 0:
            diff = ood_no_ens_metrics.mean_action_jerk - ood_ens_metrics.mean_action_jerk
            jerk_reduction = round(100.0 * diff / ood_no_ens_metrics.mean_action_jerk, 2)

    return OODBenchmarkReport(
        model_path=str(ckpt_path),
        backend=backend,
        num_episodes=num_episodes,
        nominal=nominal_metrics,
        ood_dr_ensembled=ood_ens_metrics,
        ood_dr_no_ensemble=ood_no_ens_metrics,
        action_jerk_reduction_pct=jerk_reduction,
    )
