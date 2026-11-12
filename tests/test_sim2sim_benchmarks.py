"""Verification suite for Phase 6: Cloud simulator adapters & Sim-to-Sim gap benchmarks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim2real_engine.benchmarks import (
    PerceptionNoiseConfig,
    PerceptionNoisePoint,
    PerceptionSensitivityCurve,
    run_perception_sensitivity_sweep,
)
from sim2real_engine.benchmarks.sim2sim_eval import (
    Sim2SimMetrics,
    create_simulator_backend,
    evaluate_sim2sim_gap,
)
from sim2real_engine.cli import main
from sim2real_engine.simulators.base import SimulatorBackend
from sim2real_engine.simulators.cloud_adapters import GenesisAdapter, IsaacLabAdapter


def test_cloud_adapters_protocol_conformance() -> None:
    """GenesisAdapter and IsaacLabAdapter must fully conform to SimulatorBackend protocol."""
    genesis = GenesisAdapter()
    isaac = IsaacLabAdapter()

    assert isinstance(genesis, SimulatorBackend)
    assert isinstance(isaac, SimulatorBackend)
    assert genesis.dof == 6
    assert isaac.dof == 6

    # Test reset and step
    j_gen, ee_gen, obj_gen = genesis.reset(seed=42)
    assert len(j_gen.positions_rad) == 6
    assert math.isclose(float(np.linalg.norm(ee_gen.quaternion_wxyz)), 1.0, abs_tol=1e-5)

    rec = genesis.step((0.0, 0.4, -0.8, 0.0, 0.4, 0.0))
    assert rec.joint_state.dof == 6
    assert not rec.is_terminal


def test_evaluate_sim2sim_gap_identical_backend() -> None:
    """Comparing identical backends must produce exactly zero divergence."""
    metrics = evaluate_sim2sim_gap(
        reference="mujoco",
        target="mujoco",
        num_episodes=2,
        seed=100,
    )
    assert isinstance(metrics, Sim2SimMetrics)
    assert metrics.num_episodes == 2
    assert metrics.mean_trajectory_rmse == 0.0
    assert metrics.mean_ee_divergence_m == 0.0
    assert metrics.success_rate_reference == 1.0
    assert metrics.success_rate_target == 1.0


def test_evaluate_sim2sim_gap_cross_engine_divergence() -> None:
    """Cross-engine comparison (MuJoCo vs Isaac Lab) quantifies realistic non-zero drift."""
    metrics = evaluate_sim2sim_gap(
        reference="mujoco",
        target="isaac_lab",
        num_episodes=2,
        seed=42,
    )
    assert metrics.reference_backend == "mujoco"
    assert metrics.target_backend == "isaac_lab"
    assert metrics.mean_trajectory_rmse > 0.0
    assert metrics.mean_ee_divergence_m >= 0.0
    assert metrics.success_rate_reference == 1.0
    assert metrics.success_rate_target == 1.0


def test_create_simulator_backend_unknown_raises() -> None:
    """create_simulator_backend rejects unregistered backend names."""
    with pytest.raises(ValueError, match="Unknown simulator backend"):
        create_simulator_backend("unreal_engine")  # type: ignore[arg-type]


def test_cli_sim2sim_eval_json(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI sim2sim-eval outputs valid JSON metrics."""
    ret = main([
        "sim2sim-eval",
        "--ref", "mujoco",
        "--target", "isaac_lab",
        "--episodes", "2",
        "--json",
    ])
    assert ret == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["reference_backend"] == "mujoco"
    assert payload["target_backend"] == "isaac_lab"
    assert payload["num_episodes"] == 2
    assert "mean_trajectory_rmse" in payload


def test_cli_sim2sim_eval_plain_and_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI sim2sim-eval outputs formatted text summary and rejects invalid episode counts."""
    ret = main([
        "sim2sim-eval",
        "--ref", "mujoco",
        "--target", "genesis",
        "--episodes", "1",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Sim-to-Sim Evaluation (mujoco vs genesis" in captured.out
    assert "Mean Trajectory RMSE:" in captured.out

    ret_err = main(["sim2sim-eval", "--episodes", "0"])
    assert ret_err == 1
    captured_err = capsys.readouterr()
    assert "Number of episodes must be >= 1" in captured_err.err


def test_evaluate_ood_robustness_missing_checkpoint() -> None:
    """Missing checkpoint raises FileNotFoundError."""
    from sim2real_engine.benchmarks.ood_dr_eval import evaluate_ood_robustness

    with pytest.raises(FileNotFoundError, match="Policy checkpoint not found"):
        evaluate_ood_robustness("non_existent_model.pt")


def test_evaluate_ood_robustness_execution() -> None:
    """OOD robustness benchmark executes across nominal and OOD perturbed regimes."""
    from sim2real_engine.benchmarks.ood_dr_eval import (
        OODBenchmarkReport,
        evaluate_ood_robustness,
    )

    report = evaluate_ood_robustness(
        model_path="checkpoints/bc_pick_place.pt",
        backend="mujoco_native",
        num_episodes=2,
        base_seed=42,
        max_steps=10,
        compare_ensembling=True,
    )
    assert isinstance(report, OODBenchmarkReport)
    assert report.num_episodes == 2
    assert report.nominal.num_episodes == 2
    assert report.ood_dr_ensembled.num_episodes == 2
    assert report.ood_dr_no_ensemble is not None
    assert report.ood_dr_no_ensemble.num_episodes == 2
    assert report.action_jerk_reduction_pct is not None

    # Verify decoupled surgical robotics telemetry
    surg = report.nominal.surgical
    assert surg.mean_min_reach_error_mm >= 0.0
    assert surg.mean_placement_error_mm >= 0.0
    assert surg.mean_execution_time_s > 0.0
    assert surg.mean_hunting_oscillations >= 0.0
    assert 0.0 <= surg.stages.reach_rate <= 1.0
    assert 0.0 <= surg.stages.grasp_rate <= 1.0
    assert 0.0 <= surg.stages.transport_rate <= 1.0
    assert 0.0 <= surg.stages.place_rate <= 1.0
    assert 0.0 <= surg.stages.chained_task_success_rate <= 1.0


def test_cli_eval_policy_dr_and_compare_ensembling(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real eval-policy --dr --compare-ensembling` emits formatted summary and JSON."""
    ret = main([
        "eval-policy",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--dr",
        "--compare-ensembling",
        "--episodes", "1",
        "--max-steps", "10",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["backend"] == "mujoco_native"
    assert "nominal" in payload
    assert "ood_dr_ensembled" in payload
    assert payload["ood_dr_no_ensemble"] is not None

    # Plain text run
    ret_plain = main([
        "eval-policy",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--dr",
        "--compare-ensembling",
        "--episodes", "1",
        "--max-steps", "10",
    ])
    assert ret_plain == 0
    captured_plain = capsys.readouterr()
    assert "OOD Robustness Benchmark (mujoco_native)" in captured_plain.out
    assert "Ensembling Robustness Benefit:" in captured_plain.out


def test_perception_noise_config_and_curve() -> None:
    """PerceptionNoiseConfig and PerceptionSensitivityCurve validate mechanical clearance."""
    cfg = PerceptionNoiseConfig(pos_noise_std_m=0.005, noise_mode="per_episode")
    assert cfg.pos_noise_std_m == 0.005
    assert cfg.noise_mode == "per_episode"

    pt = PerceptionNoisePoint(
        sigma_mm=3.0,
        sigma_m=0.003,
        num_episodes=5,
        grasp_success_rate=0.4,
        placement_success_rate=0.2,
        chained_success_rate=0.08,
        mean_min_reach_error_mm=12.5,
        mean_placement_error_mm=25.0,
        mean_hunting_oscillations=1.2,
    )
    assert pt.sigma_mm == 3.0

    curve = PerceptionSensitivityCurve(
        model_path="checkpoints/bc_pick_place.pt",
        backend="mujoco_native",
        noise_mode="per_episode",
        num_episodes_per_point=5,
        noise_points=[pt],
        breakdown_threshold_mm=3.0,
        zero_noise_grasp_rate=1.0,
        mechanical_clearance_mm=3.0,
    )
    assert curve.mechanical_clearance_mm == 3.0
    assert curve.breakdown_threshold_mm == 3.0
    assert len(curve.noise_points) == 1


def test_run_perception_sensitivity_sweep_execution() -> None:
    """`run_perception_sensitivity_sweep` executes over discrete sigmas and computes breakdown."""
    curve = run_perception_sensitivity_sweep(
        model_path="checkpoints/bc_pick_place.pt",
        backend_name="mujoco_native",
        sigma_min_mm=0.0,
        sigma_max_mm=4.0,
        num_points=2,
        episodes_per_point=1,
        max_steps=10,
        noise_mode="per_episode",
    )
    assert isinstance(curve, PerceptionSensitivityCurve)
    assert len(curve.noise_points) == 2
    assert curve.noise_points[0].sigma_mm == 0.0
    assert curve.noise_points[1].sigma_mm == 4.0
    assert 0.0 <= curve.noise_points[0].grasp_success_rate <= 1.0


def test_cli_eval_policy_with_perception_noise(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real eval-policy --perception-noise-mm` injects noise into rollout evaluation."""
    ret = main([
        "eval-policy",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--perception-noise-mm", "2.5",
        "--perception-noise-mode", "per_episode",
        "--episodes", "1",
        "--max-steps", "10",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["backend"] == "mujoco_native"
    assert "nominal" in payload


def test_cli_sweep_perception_noise(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real sweep-perception-noise` outputs ASCII table and JSON format."""
    # JSON output check
    ret_json = main([
        "sweep-perception-noise",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--sigma-min-mm", "0.0",
        "--sigma-max-mm", "3.0",
        "--num-points", "2",
        "--episodes", "1",
        "--max-steps", "10",
        "--json",
    ])
    assert ret_json == 0
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert "mechanical_clearance_mm" in payload
    assert payload["mechanical_clearance_mm"] == 3.0
    assert len(payload["noise_points"]) == 2

    # Plain text ASCII table output check
    ret_plain = main([
        "sweep-perception-noise",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--sigma-min-mm", "0.0",
        "--sigma-max-mm", "3.0",
        "--num-points", "2",
        "--episodes", "1",
        "--max-steps", "10",
    ])
    assert ret_plain == 0
    captured_plain = capsys.readouterr()
    assert "Perception Noise Sensitivity Sweep" in captured_plain.out
    assert "Clearance: 3.0 mm" in captured_plain.out
    assert "Empirical Breakdown Threshold" in captured_plain.out

