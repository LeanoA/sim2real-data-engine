"""Command-line interface for the Sim2Real Data Engine."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from sim2real_engine import __version__
from sim2real_engine.benchmarks.sim2sim_eval import (
    create_simulator_backend,
    evaluate_sim2sim_gap,
)
from sim2real_engine.core.contracts import (
    DEFAULT_BACKENDS,
    BackendName,
    DomainRandomizationConfig,
    JointState,
)
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.dataset.exporter import LeRobotEpisodeExporter
from sim2real_engine.randomization.samplers import DomainRandomizer
from sim2real_engine.server.bridge import PolicyInferenceBridge
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert, RolloutSummary
from sim2real_engine.visualization.dashboard import generate_html_dashboard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim2real",
        description=(
            "Modular Synthetic Data Generation, Multi-Engine Physics Benchmarking "
            "& Visuomotor Distillation CLI."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Display framework version and architecture.")
    info_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    backends_parser = subparsers.add_parser(
        "backends",
        help="List registered physics simulator backends and compute requirements.",
    )
    backends_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a domain randomization configuration schedule.",
    )
    validate_parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    validate_parser.add_argument(
        "--mass-scale-min",
        type=float,
        default=0.85,
        help="Minimum payload mass scale factor.",
    )
    validate_parser.add_argument(
        "--mass-scale-max",
        type=float,
        default=1.15,
        help="Maximum payload mass scale factor.",
    )
    validate_parser.add_argument(
        "--friction-scale-min",
        type=float,
        default=0.70,
        help="Minimum Coulomb friction scale factor.",
    )
    validate_parser.add_argument(
        "--friction-scale-max",
        type=float,
        default=1.30,
        help="Maximum Coulomb friction scale factor.",
    )
    validate_parser.add_argument(
        "--camera-jitter-m",
        type=float,
        default=0.015,
        help="Maximum camera position jitter in meters.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    rollout_parser = subparsers.add_parser(
        "rollout",
        help="Execute closed-loop privileged IK expert rollouts on a simulator backend.",
    )
    rollout_parser.add_argument(
        "--backend",
        choices=list(DEFAULT_BACKENDS.keys()),
        default="mujoco",
        help="Target physics simulator backend.",
    )
    rollout_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes to execute.",
    )
    rollout_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base deterministic seed.",
    )
    rollout_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    sample_dr_parser = subparsers.add_parser(
        "sample-dr",
        help="Sample deterministic domain randomization perturbations across episodes.",
    )
    sample_dr_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic base RNG seed.",
    )
    sample_dr_parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to sample.",
    )
    sample_dr_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    export_parser = subparsers.add_parser(
        "export-dataset",
        help="Generate synthetic episode rollouts and export as LeRobotDataset v2 Parquet.",
    )
    export_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes to record.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/sim2real_pick_place",
        help="Destination directory for exported dataset and manifest.",
    )
    export_parser.add_argument(
        "--backend",
        choices=list(DEFAULT_BACKENDS.keys()),
        default="mujoco",
        help="Physics simulator backend.",
    )
    export_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base deterministic RNG seed.",
    )
    export_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    benchmark_parser = subparsers.add_parser(
        "benchmark-bridge",
        help="Benchmark decoupled policy inference bridge with receding-horizon action chunking.",
    )
    benchmark_parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of control loop steps to benchmark.",
    )
    benchmark_parser.add_argument(
        "--chunk-horizon",
        type=int,
        default=16,
        help="Action chunk prediction horizon H.",
    )
    benchmark_parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.05,
        help="Temporal ensemble exponential decay rate m.",
    )
    benchmark_parser.add_argument(
        "--simulated-latency-ms",
        type=float,
        default=8.0,
        help="Simulated neural network inference forward pass latency in ms.",
    )
    benchmark_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    sim2sim_parser = subparsers.add_parser(
        "sim2sim-eval",
        help="Evaluate physics divergence and Sim-to-Sim gap between simulator backends.",
    )
    sim2sim_parser.add_argument(
        "--ref",
        choices=list(DEFAULT_BACKENDS.keys()),
        default="mujoco",
        help="Baseline reference simulator backend.",
    )
    sim2sim_parser.add_argument(
        "--target",
        choices=list(DEFAULT_BACKENDS.keys()),
        default="isaac_lab",
        help="Target simulator backend to evaluate against reference.",
    )
    sim2sim_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of comparative evaluation episodes.",
    )
    sim2sim_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base deterministic RNG seed.",
    )
    sim2sim_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    vis_parser = subparsers.add_parser(
        "visualize",
        help="Generate interactive 3D kinematics and Sim-to-Sim HTML visualization dashboard.",
    )
    vis_parser.add_argument(
        "--backend",
        type=str,
        choices=["mujoco_native", "mujoco"],
        default="mujoco_native",
        help="Simulator backend for trajectory animation (default: mujoco_native).",
    )
    vis_parser.add_argument(
        "--output",
        type=str,
        default="sim2real_visualizer.html",
        help="Path for generated HTML dashboard file.",
    )
    vis_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional trained policy checkpoint (.pt) to compare against expert in 3D.",
    )
    vis_parser.add_argument(
        "--compare-model-path",
        type=str,
        default=None,
        help="Optional secondary policy checkpoint (.pt) for side-by-side multi-model comparison.",
    )
    vis_parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Automatically bundle available canonical and cloud checkpoints into dashboard.",
    )
    vis_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic rollout RNG seed.",
    )
    vis_parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in default web browser upon creation.",
    )
    vis_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    view_phys_parser = subparsers.add_parser(
        "view-physics",
        help="Launch official native MuJoCo 4.x 3D desktop viewer with real-time control.",
    )
    view_phys_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic rollout RNG seed.",
    )
    view_phys_parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of pick-and-place cycles to execute before closing.",
    )
    view_phys_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional trained policy checkpoint (.pt) to execute in real-time 3D.",
    )

    train_parser = subparsers.add_parser(
        "train-policy",
        help="Train a Behavioral Cloning policy network on demonstration Parquet datasets.",
    )
    train_parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./datasets/sim2real_pick_place",
        help="Path to dataset directory containing data/ and meta/manifest.json.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of complete training epochs (default: 15).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial AdamW learning rate (default: 1e-3).",
    )
    train_parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Action chunk prediction horizon H (default: 8).",
    )
    train_parser.add_argument(
        "--include-ee-pose",
        dest="include_ee_pose",
        action="store_true",
        default=True,
        help="Include Cartesian EE pose in observations (default: True, 30D state).",
    )
    train_parser.add_argument(
        "--no-ee-pose",
        dest="include_ee_pose",
        action="store_false",
        help="Exclude Cartesian EE pose from observations (20D state).",
    )
    train_parser.add_argument(
        "--tcp-frame",
        action="store_true",
        default=False,
        help="Transform relative object observations into local TCP frame (R_ee^T (p_obj - p_ee)).",
    )
    train_parser.add_argument(
        "--output-model",
        type=str,
        default="./checkpoints/bc_policy.pt",
        help="Destination checkpoint file path.",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute execution device (default: auto).",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic training RNG seed.",
    )
    train_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    eval_parser = subparsers.add_parser(
        "eval-policy",
        help="Evaluate a trained Behavioral Cloning policy in closed-loop physics simulation.",
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/bc_policy.pt",
        help="Path to saved policy checkpoint file (.pt).",
    )
    eval_parser.add_argument(
        "--backend",
        choices=list(DEFAULT_BACKENDS.keys()),
        default="mujoco_native",
        help="Simulator backend for evaluation rollouts.",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to execute.",
    )
    eval_parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Maximum steps allowed per episode.",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base deterministic evaluation RNG seed.",
    )
    eval_parser.add_argument(
        "--dr",
        action="store_true",
        help="Run out-of-distribution domain randomization robustness benchmark.",
    )
    eval_parser.add_argument(
        "--compare-ensembling",
        action="store_true",
        help="Compare performance with vs without temporal ensembling under identical OOD physics.",
    )
    eval_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def _run_rollout(backend_name: str, episodes: int, base_seed: int, emit_json: bool) -> int:
    if episodes < 1:
        print("Number of episodes must be >= 1.", file=sys.stderr)
        return 1

    adapter = create_simulator_backend(cast(BackendName, backend_name))
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    successes = 0
    step_counts: list[int] = []
    final_errors: list[float] = []

    for idx in range(episodes):
        ep_seed = base_seed + idx
        records = expert.run_episode(backend=adapter, seed=ep_seed, max_steps=120)
        if records and records[-1].is_terminal:
            successes += 1
        step_counts.append(len(records))
        final_wp = expert.current_waypoint()
        ee_end = np.asarray(records[-1].end_effector_pose.position, dtype=np.float64)
        wp_end = np.asarray(final_wp.position, dtype=np.float64)
        final_errors.append(float(np.linalg.norm(ee_end - wp_end)))

    summary = RolloutSummary(
        backend=backend_name,
        num_episodes=episodes,
        success_count=successes,
        success_rate=round(successes / episodes, 4),
        mean_steps=round(float(np.mean(step_counts)), 2),
        mean_final_ee_error_m=round(float(np.mean(final_errors)), 6),
    )
    if emit_json:
        print(json.dumps(summary.model_dump(), indent=2))
    else:
        print(
            f"Rollout complete ({summary.backend}): "
            f"{summary.success_count}/{summary.num_episodes} succeeded "
            f"(mean_steps={summary.mean_steps}, mean_error={summary.mean_final_ee_error_m} m)"
        )
    return 0


def _run_sample_dr(seed: int, episodes: int, emit_json: bool) -> int:
    if episodes < 1:
        print("Number of episodes must be >= 1.", file=sys.stderr)
        return 1

    config = DomainRandomizationConfig(seed=seed)
    randomizer = DomainRandomizer(config=config)
    perturbations = randomizer.sample_batch(episodes)

    if emit_json:
        payload = [p.model_dump() for p in perturbations]
        print(json.dumps(payload, indent=2))
    else:
        print(f"Sampled {len(perturbations)} perturbations (base_seed={seed}):")
        for p in perturbations:
            pos_norm = float(np.linalg.norm(p.camera_pose_offset.position))
            print(
                f"- Ep {p.episode_id}: mass={p.mass_scale:.3f}, "
                f"friction={p.friction_scale:.3f}, "
                f"damping={p.damping_scale:.3f}, "
                f"lighting={p.lighting_intensity:.3f}, "
                f"cam_offset_norm={pos_norm:.4f}m"
            )
    return 0


def _run_export_dataset(
    backend_name: str,
    episodes: int,
    output_dir: str,
    base_seed: int,
    emit_json: bool,
) -> int:
    if episodes < 1:
        print("Number of episodes must be >= 1.", file=sys.stderr)
        return 1

    out_path = Path(output_dir)
    exporter = LeRobotEpisodeExporter(root_dir=out_path)
    adapter = create_simulator_backend(cast(BackendName, backend_name))
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)
    config = DomainRandomizationConfig(seed=base_seed)
    randomizer = DomainRandomizer(config=config)

    for ep_id in range(episodes):
        ep_seed = base_seed + ep_id
        # Sample domain perturbation to drive deterministic variation
        perturbation = randomizer.sample_episode(ep_id)
        records = expert.run_episode(
            backend=adapter, seed=ep_seed, max_steps=120, perturbation=perturbation
        )
        exporter.write_episode(episode_id=ep_id, steps=records)

    manifest = exporter.finalize_manifest()
    integrity_ok = exporter.verify_integrity()

    if emit_json:
        payload = {
            "status": "success",
            "manifest": manifest.model_dump(),
            "integrity_verified": integrity_ok,
            "output_directory": str(out_path.resolve()),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Successfully exported {manifest.num_episodes} episodes "
            f"({manifest.total_steps} total steps) to {out_path}."
        )
        print(f"SHA-256 envelope integrity verified: {integrity_ok}")
    return 0 if integrity_ok else 1


def _run_benchmark_bridge(
    steps: int,
    horizon: int,
    decay_rate: float,
    simulated_latency_ms: float,
    emit_json: bool,
) -> int:
    if steps < 1 or horizon < 1:
        print("Steps and horizon must be >= 1.", file=sys.stderr)
        return 1

    dof = 6
    buffer = TemporalEnsembleBuffer(action_dim=dof, horizon=horizon, decay_rate=decay_rate)
    bridge = PolicyInferenceBridge(buffer=buffer, simulated_latency_ms=simulated_latency_ms)

    curr_q = (0.0,) * dof
    dummy_vel = (0.0,) * dof
    dummy_tau = (0.0,) * dof

    for step_idx in range(steps):
        obs = JointState(
            positions_rad=curr_q,
            velocities_rad_s=dummy_vel,
            torques_nm=dummy_tau,
        )
        action = bridge.infer_step(observation=obs, step=step_idx)
        curr_q = tuple(float(x) for x in action)

    metrics = bridge.get_latency_metrics()

    if emit_json:
        payload = {
            "status": "success",
            "steps": steps,
            "chunk_horizon": horizon,
            "decay_rate": decay_rate,
            "metrics": metrics.model_dump(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Policy Bridge Benchmark ({steps} steps, horizon={horizon}, decay={decay_rate}):"
        )
        print(
            f"- Latency: Mean={metrics.mean_ms:.2f}ms | P50={metrics.p50_ms:.2f}ms | "
            f"P95={metrics.p95_ms:.2f}ms | P99={metrics.p99_ms:.2f}ms | Max={metrics.max_ms:.2f}ms"
        )
    return 0


def _run_sim2sim_eval(
    ref_name: str,
    target_name: str,
    episodes: int,
    base_seed: int,
    emit_json: bool,
) -> int:
    if episodes < 1:
        print("Number of episodes must be >= 1.", file=sys.stderr)
        return 1

    metrics = evaluate_sim2sim_gap(
        reference=cast(BackendName, ref_name),
        target=cast(BackendName, target_name),
        num_episodes=episodes,
        seed=base_seed,
    )

    if emit_json:
        print(json.dumps(metrics.model_dump(), indent=2))
    else:
        print(
            f"Sim-to-Sim Evaluation ({metrics.reference_backend} vs {metrics.target_backend}, "
            f"{metrics.num_episodes} episodes):"
        )
        print(f"- Mean Trajectory RMSE: {metrics.mean_trajectory_rmse:.6f} rad")
        print(f"- Mean Final EE Divergence: {metrics.mean_ee_divergence_m:.6f} m")
        pct_ref = metrics.success_rate_reference * 100
        pct_tgt = metrics.success_rate_target * 100
        print(f"- Success Rate ({metrics.reference_backend}): {pct_ref:.1f}%")
        print(f"- Success Rate ({metrics.target_backend}): {pct_tgt:.1f}%")
    return 0


def _run_visualize(
    output_path: str,
    seed: int,
    backend: BackendName,
    open_browser: bool,
    emit_json: bool,
    model_path: str | None = None,
    compare_model_path: str | None = None,
    multi_model: bool = False,
) -> int:
    if multi_model or compare_model_path is not None:
        from sim2real_engine.visualization.dashboard import generate_policy_comparison_html
        from sim2real_engine.visualization.data import (
            collect_multi_model_comparison,
            collect_policy_vs_expert_comparison,
        )

        canonical_path = Path("checkpoints/bc_pick_place.pt")
        cloud_path = Path("results/bc_cloud_policy.pt")

        models_to_load: dict[str, Path | str] = {}
        primary_p = Path(model_path) if model_path else canonical_path
        secondary_p = Path(compare_model_path) if compare_model_path else cloud_path

        if primary_p.is_file():
            models_to_load["canonical_45ep"] = primary_p
        if secondary_p.is_file():
            models_to_load["cloud_15ep"] = secondary_p

        if len(models_to_load) > 1:
            comps = collect_multi_model_comparison(
                model_paths=models_to_load, seed=seed, backend=backend
            )
            prim_comp = comps["canonical_45ep"]
            add_comps = {k: v for k, v in comps.items() if k != "canonical_45ep"}
            file_path = generate_policy_comparison_html(
                comparison_data=prim_comp,
                output_path=output_path,
                additional_models=add_comps,
            )
        elif models_to_load:
            single_key = next(iter(models_to_load))
            single_comp = collect_policy_vs_expert_comparison(
                model_path=models_to_load[single_key], seed=seed, backend=backend
            )
            file_path = generate_policy_comparison_html(
                comparison_data=single_comp,
                output_path=output_path,
            )
        else:
            comp_data = collect_policy_vs_expert_comparison(
                model_path=model_path or str(canonical_path), seed=seed, backend=backend
            )
            file_path = generate_policy_comparison_html(
                comparison_data=comp_data,
                output_path=output_path,
            )
    elif model_path is not None:
        from sim2real_engine.visualization.dashboard import generate_policy_comparison_html
        from sim2real_engine.visualization.data import collect_policy_vs_expert_comparison

        comp_data = collect_policy_vs_expert_comparison(
            model_path=model_path,
            seed=seed,
            backend=backend,
        )
        file_path = generate_policy_comparison_html(
            comparison_data=comp_data,
            output_path=output_path,
        )
    else:
        file_path = generate_html_dashboard(seed=seed, output_path=output_path, backend=backend)
    if open_browser:
        webbrowser.open(file_path.as_uri())

    if emit_json:
        payload = {
            "status": "success",
            "file": str(file_path),
            "backend": backend,
            "seed": seed,
            "opened": open_browser,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Interactive visualizer generated ({backend}): {file_path}")
        if open_browser:
            print("Opened in default web browser.")
    return 0


def _run_view_physics(
    seed: int,
    episodes: int,
    model_path: str | None = None,
) -> int:
    from sim2real_engine.visualization.mujoco_gui import run_interactive_mujoco_viewer

    run_interactive_mujoco_viewer(seed=seed, episodes=episodes, model_path=model_path)
    return 0


class PolicyEvalSummary(BaseModel):
    """Telemetry metrics summary for closed-loop policy evaluation rollouts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str
    backend: str
    num_episodes: int
    mean_steps: float
    mean_ee_displacement_m: float
    mean_payload_displacement_m: float = 0.0
    inference_latency_p50_ms: float
    inference_latency_p99_ms: float


def _run_train_policy(
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    horizon: int,
    include_ee_pose: bool,
    output_model: str,
    device: str,
    seed: int,
    emit_json: bool,
    use_tcp_frame: bool = False,
) -> int:
    from typing import Any

    from sim2real_engine.learning import (
        LeRobotParquetDataset,
        TrainerConfig,
        train_policy,
    )

    ds_path = Path(dataset_dir)
    if not ds_path.is_dir():
        print(f"Dataset directory not found: {ds_path}", file=sys.stderr)
        return 1

    try:
        dataset = LeRobotParquetDataset(
            root_dir=ds_path,
            horizon=horizon,
            include_ee_pose=include_ee_pose,
            use_tcp_frame=use_tcp_frame,
        )
    except Exception as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=cast(Any, device),
        seed=seed,
    )

    model, summary = train_policy(dataset, config, checkpoint_path=output_model)

    if emit_json:
        payload = {
            "status": "success",
            "model_path": str(Path(output_model).resolve()),
            "summary": summary.model_dump(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Policy training complete: {summary.total_epochs} epochs on "
            f"{summary.total_samples} transitions."
        )
        print(
            f"Initial Loss: {summary.initial_loss:.4f} -> Final Val Loss: "
            f"{summary.final_val_loss:.4f} (-{summary.loss_reduction_pct:.1f}%)"
        )
        print(
            f"Throughput: {summary.throughput_samples_per_s:.1f} samples/s "
            f"({summary.training_time_s:.2f}s total on {summary.device_used})"
        )
        print(f"Checkpoint saved to: {output_model}")
    return 0


def _run_eval_policy(
    model_path: str,
    backend_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    emit_json: bool,
    dr: bool = False,
    compare_ensembling: bool = False,
) -> int:
    from sim2real_engine.learning import BehaviorCloningMLP

    ckpt_path = Path(model_path)
    if not ckpt_path.is_file():
        print(f"Model checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    if dr:
        from sim2real_engine.benchmarks import evaluate_ood_robustness

        report = evaluate_ood_robustness(
            model_path=ckpt_path,
            backend=cast(BackendName, backend_name),
            num_episodes=episodes,
            base_seed=seed,
            max_steps=max_steps,
            compare_ensembling=compare_ensembling,
        )
        if emit_json:
            print(json.dumps(report.model_dump(), indent=2))
        else:
            print(f"OOD Robustness Benchmark ({report.backend}):")
            print(f"Model: {report.model_path}")
            print(f"Episodes: {report.num_episodes}")

            for name, r in [
                ("Nominal", report.nominal),
                ("OOD DR (Ensembled)", report.ood_dr_ensembled),
            ]:
                s = r.surgical
                st = s.stages
                print(
                    f"  {name}: Chained Task Success={st.chained_task_success_rate * 100.0:.1f}%\n"
                    f"    Stages: Reach={st.reach_rate * 100.0:.0f}% | "
                    f"Grasp={st.grasp_rate * 100.0:.0f}% | "
                    f"Transport={st.transport_rate * 100.0:.0f}% | "
                    f"Place={st.place_rate * 100.0:.0f}%\n"
                    f"    Surgical: Proximity={s.mean_min_reach_error_mm:.1f} mm | "
                    f"Place Err={s.mean_placement_error_mm:.1f} mm | "
                    f"Duration={s.mean_execution_time_s:.2f}s | "
                    f"Hunting={s.mean_hunting_oscillations:.1f} rev\n"
                    f"    Dynamics: Payload Disp={r.mean_payload_displacement_m * 1000.0:.1f} mm | "
                    f"Jerk={r.mean_action_jerk:.4f}"
                )

            if report.ood_dr_no_ensemble is not None:
                r_no = report.ood_dr_no_ensemble
                s_no = r_no.surgical
                st_no = s_no.stages
                print(
                    f"  OOD DR (No Ensemble): Chained Success="
                    f"{st_no.chained_task_success_rate * 100.0:.1f}%\n"
                    f"    Stages: Reach={st_no.reach_rate * 100.0:.0f}% | "
                    f"Grasp={st_no.grasp_rate * 100.0:.0f}% | "
                    f"Transport={st_no.transport_rate * 100.0:.0f}% | "
                    f"Place={st_no.place_rate * 100.0:.0f}%\n"
                    f"    Surgical: Proximity={s_no.mean_min_reach_error_mm:.1f} mm | "
                    f"Place Err={s_no.mean_placement_error_mm:.1f} mm | "
                    f"Hunting={s_no.mean_hunting_oscillations:.1f} rev\n"
                    f"    Dynamics: Payload Disp="
                    f"{r_no.mean_payload_displacement_m * 1000.0:.1f} mm | "
                    f"Jerk={r_no.mean_action_jerk:.4f}"
                )
                success_diff = (
                    report.ood_dr_ensembled.success_rate
                    - report.ood_dr_no_ensemble.success_rate
                ) * 100.0
                print(
                    f"Ensembling Robustness Benefit: {success_diff:+.1f}% success rate retention "
                    f"under OOD physical variations"
                )
        return 0

    try:
        model, metadata = BehaviorCloningMLP.load_checkpoint(ckpt_path)
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}", file=sys.stderr)
        return 1

    adapter = create_simulator_backend(cast(BackendName, backend_name))
    buffer = TemporalEnsembleBuffer(action_dim=model.action_dim, horizon=model.horizon)
    policy_fn = model.to_policy_fn(metadata)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)

    displacements: list[float] = []
    payload_displacements: list[float] = []
    step_counts: list[int] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        joints, initial_ee, initial_obj = adapter.reset(seed=ep_seed)
        bridge.reset()
        current_joints = joints
        current_ee = initial_ee
        current_obj = initial_obj
        current_gripper = 0.0
        initial_pos = np.asarray(initial_ee.position, dtype=np.float64)
        initial_obj_pos = np.asarray(initial_obj.position, dtype=np.float64)

        for step_idx in range(max_steps):
            act_chunk = bridge.infer_step(
                current_joints,
                step_idx,
                ee_pose=current_ee,
                object_pose=current_obj,
                gripper_state=current_gripper,
            )
            act_step = tuple(float(x) for x in act_chunk)
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

    latency_metrics = bridge.get_latency_metrics()
    summary = PolicyEvalSummary(
        model_path=str(ckpt_path),
        backend=backend_name,
        num_episodes=episodes,
        mean_steps=round(float(np.mean(step_counts)), 1),
        mean_ee_displacement_m=round(float(np.mean(displacements)), 4),
        mean_payload_displacement_m=round(float(np.mean(payload_displacements)), 4),
        inference_latency_p50_ms=latency_metrics.p50_ms,
        inference_latency_p99_ms=latency_metrics.p99_ms,
    )

    if emit_json:
        print(json.dumps(summary.model_dump(), indent=2))
    else:
        print(f"Policy evaluation complete ({summary.backend}):")
        print(f"Episodes: {summary.num_episodes}, Mean Steps: {summary.mean_steps}")
        print(f"Mean EE Displacement: {summary.mean_ee_displacement_m * 1000.0:.1f} mm")
        print(f"Mean Payload Displacement: {summary.mean_payload_displacement_m * 1000.0:.1f} mm")
        print(
            f"Inference Latency: P50={summary.inference_latency_p50_ms} ms, "
            f"P99={summary.inference_latency_p99_ms} ms"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the `sim2real` CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        payload = {
            "package": "sim2real-engine",
            "version": __version__,
            "supported_backends": list(DEFAULT_BACKENDS.keys()),
            "dataset_format": "LeRobotDataset v2 (Parquet + RGB-D)",
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"sim2real-engine v{__version__}")
            print(f"Supported Backends: {', '.join(DEFAULT_BACKENDS.keys())}")
            print(f"Dataset Format: {payload['dataset_format']}")
        return 0

    if args.command == "backends":
        serialized = [spec.model_dump() for spec in DEFAULT_BACKENDS.values()]
        if args.json:
            print(json.dumps({"backends": serialized}, indent=2))
        else:
            for spec in DEFAULT_BACKENDS.values():
                gpu_tag = "NVIDIA GPU" if spec.requires_nvidia_gpu else "Local CPU/ARM64"
                print(
                    f"- {spec.name} ({spec.execution_tier}, "
                    f"{spec.nominal_physics_hz} Hz, {gpu_tag})"
                )
        return 0

    if args.command == "validate-config":
        try:
            config = DomainRandomizationConfig(
                seed=args.seed,
                mass_scale_range=(args.mass_scale_min, args.mass_scale_max),
                friction_scale_range=(args.friction_scale_min, args.friction_scale_max),
                camera_position_jitter_m=args.camera_jitter_m,
            )
        except ValidationError as exc:
            error_payload = {
                "valid": False,
                "errors": exc.errors(include_url=False, include_context=False),
            }
            if args.json:
                print(json.dumps(error_payload, indent=2))
            else:
                print(f"Configuration invalid: {exc}", file=sys.stderr)
            return 1

        result_payload = {"valid": True, "config": config.model_dump()}
        if args.json:
            print(json.dumps(result_payload, indent=2))
        else:
            print(f"Configuration valid (seed={config.seed}).")
        return 0

    if args.command == "rollout":
        return _run_rollout(
            backend_name=str(args.backend),
            episodes=int(args.episodes),
            base_seed=int(args.seed),
            emit_json=bool(args.json),
        )

    if args.command == "sample-dr":
        return _run_sample_dr(
            seed=int(args.seed),
            episodes=int(args.episodes),
            emit_json=bool(args.json),
        )

    if args.command == "export-dataset":
        return _run_export_dataset(
            backend_name=str(args.backend),
            episodes=int(args.episodes),
            output_dir=str(args.output_dir),
            base_seed=int(args.seed),
            emit_json=bool(args.json),
        )

    if args.command == "benchmark-bridge":
        return _run_benchmark_bridge(
            steps=int(args.steps),
            horizon=int(args.chunk_horizon),
            decay_rate=float(args.decay_rate),
            simulated_latency_ms=float(args.simulated_latency_ms),
            emit_json=bool(args.json),
        )

    if args.command == "sim2sim-eval":
        return _run_sim2sim_eval(
            ref_name=str(args.ref),
            target_name=str(args.target),
            episodes=int(args.episodes),
            base_seed=int(args.seed),
            emit_json=bool(args.json),
        )

    if args.command == "visualize":
        return _run_visualize(
            output_path=str(args.output),
            seed=int(args.seed),
            backend=cast(BackendName, args.backend),
            open_browser=bool(args.open),
            emit_json=bool(args.json),
            model_path=str(args.model_path) if args.model_path else None,
            compare_model_path=str(args.compare_model_path) if args.compare_model_path else None,
            multi_model=bool(args.multi_model),
        )

    if args.command == "view-physics":
        return _run_view_physics(
            seed=int(args.seed),
            episodes=int(args.episodes),
            model_path=str(args.model_path) if args.model_path else None,
        )

    if args.command == "train-policy":
        return _run_train_policy(
            dataset_dir=str(args.dataset_dir),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.lr),
            horizon=int(args.horizon),
            include_ee_pose=bool(args.include_ee_pose),
            output_model=str(args.output_model),
            device=str(args.device),
            seed=int(args.seed),
            emit_json=bool(args.json),
            use_tcp_frame=bool(args.tcp_frame),
        )

    if args.command == "eval-policy":
        return _run_eval_policy(
            model_path=str(args.model_path),
            backend_name=str(args.backend),
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            emit_json=bool(args.json),
            dr=bool(args.dr),
            compare_ensembling=bool(args.compare_ensembling),
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
