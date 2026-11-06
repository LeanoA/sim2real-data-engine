"""Command-line interface for the Sim2Real Data Engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np
from pydantic import ValidationError

from sim2real_engine import __version__
from sim2real_engine.core.contracts import (
    DEFAULT_BACKENDS,
    DomainRandomizationConfig,
)
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.randomization.samplers import DomainRandomizer
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert, RolloutSummary
from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter


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

    return parser


def _run_rollout(backend_name: str, episodes: int, base_seed: int, emit_json: bool) -> int:
    if episodes < 1:
        print("Number of episodes must be >= 1.", file=sys.stderr)
        return 1

    adapter = MuJoCoKinematicAdapter()
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
