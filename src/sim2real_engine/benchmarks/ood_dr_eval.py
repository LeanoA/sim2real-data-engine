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
    successes = 0
    all_jerks: list[float] = []

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
        prev_act: np.ndarray | None = None
        episode_jerks: list[float] = []

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

        # Success criterion: payload was successfully transported to placement zone
        is_success = bool(obj_disp > 0.15)
        if is_success:
            successes += 1

    latency_metrics = bridge.get_latency_metrics()
    mean_jerk = float(np.mean(all_jerks)) if all_jerks else 0.0

    return OODRegimeMetrics(
        regime=regime_name,
        num_episodes=num_episodes,
        success_rate=round(successes / max(1, num_episodes), 3),
        mean_steps=round(float(np.mean(step_counts)), 1),
        mean_ee_displacement_m=round(float(np.mean(displacements)), 4),
        mean_payload_displacement_m=round(float(np.mean(payload_displacements)), 4),
        mean_action_jerk=round(mean_jerk, 4),
        inference_latency_p50_ms=latency_metrics.p50_ms,
        inference_latency_p99_ms=latency_metrics.p99_ms,
    )


def evaluate_ood_robustness(
    model_path: str | Path,
    backend: BackendName = "mujoco_native",
    num_episodes: int = 5,
    base_seed: int = 42,
    max_steps: int = 120,
    compare_ensembling: bool = False,
    dr_config: DomainRandomizationConfig | None = None,
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
