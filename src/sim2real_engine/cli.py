"""Command-line interface for the Sim2Real Data Engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from sim2real_engine import __version__
from sim2real_engine.core.contracts import (
    DEFAULT_BACKENDS,
    DomainRandomizationConfig,
)


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

    return parser


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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
