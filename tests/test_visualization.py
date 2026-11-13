"""Verification suite for 3D trajectory and Sim-to-Sim visualization tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real_engine.cli import main
from sim2real_engine.visualization.dashboard import generate_html_dashboard
from sim2real_engine.visualization.data import (
    ArmLinkPoints,
    collect_sim2sim_comparison_data,
    collect_trajectory_data,
    compute_arm_links,
)


def test_compute_arm_links_structure_and_dimensions() -> None:
    q_home = [0.0, -0.55, 1.10, 0.0, -0.55, 0.0]
    links = compute_arm_links(q_home)

    assert isinstance(links, ArmLinkPoints)
    assert links.base == (0.0, 0.0, 0.0)
    assert links.shoulder == (0.0, 0.0, 0.22)
    assert len(links.elbow) == 3
    assert links.elbow[2] > 0.30  # Industrial Elbow-Up configuration ensures floor clearance
    assert len(links.wrist) == 3
    assert len(links.end_effector) == 3

    # Check helper tuple formatting for 3D plotting
    xs, ys, zs = links.to_coordinates_tuple()
    assert len(xs) == 5
    assert len(ys) == 5
    assert len(zs) == 5
    assert xs[0] == 0.0 and ys[0] == 0.0 and zs[0] == 0.0


def test_compute_arm_links_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="Expected 6 joint angles"):
        compute_arm_links([0.0, 0.1, 0.2])


def test_collect_trajectory_data_deterministic() -> None:
    traj_data = collect_trajectory_data(seed=42, max_steps=50)

    assert traj_data.episode_id == 42
    assert len(traj_data.timesteps_s) > 0
    assert len(traj_data.ee_trajectory) == len(traj_data.timesteps_s)
    assert len(traj_data.arm_trajectory) == len(traj_data.timesteps_s)
    assert len(traj_data.object_trajectory) == len(traj_data.timesteps_s)
    assert len(traj_data.joint_trajectory_deg) == len(traj_data.timesteps_s)
    assert len(traj_data.joint_trajectory_deg[0]) == 6
    assert "Initial Home" in traj_data.key_arm_poses
    assert "Final Place" in traj_data.key_arm_poses
    assert len(traj_data.waypoints) == 3


def test_collect_sim2sim_comparison_data() -> None:
    sim_data = collect_sim2sim_comparison_data(seed=42, max_steps=30)

    assert sim_data.seed == 42
    assert "mujoco" in sim_data.ee_trajectories
    assert "genesis" in sim_data.ee_trajectories
    assert "isaac_lab" in sim_data.ee_trajectories

    assert len(sim_data.cartesian_errors_m["isaac_lab"]) == len(sim_data.timesteps_s)
    assert len(sim_data.joint_rmses_rad["isaac_lab"]) == len(sim_data.timesteps_s)
    assert "isaac_lab" in sim_data.metrics
    assert "genesis" in sim_data.metrics


def test_generate_html_dashboard(tmp_path: Path) -> None:
    out_file = tmp_path / "test_visualizer.html"
    result_path = generate_html_dashboard(seed=42, output_path=out_file)

    assert result_path.is_file()
    content = result_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "plot-3d-arm" in content
    assert "plot-3d-sim2sim" in content
    assert "plot-drift-cartesian" in content
    assert "plot-drift-joints" in content
    assert "Plotly.newPlot" in content


def test_cli_visualize_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "cli_vis.html"
    code = main(["visualize", "--seed", "42", "--output", str(out_file), "--json"])

    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["backend"] == "mujoco_native"
    assert payload["seed"] == 42
    assert Path(payload["file"]).is_file()

    # Also test explicit kinematic backend
    out_file_kin = tmp_path / "cli_vis_kin.html"
    code_kin = main([
        "visualize",
        "--backend",
        "mujoco",
        "--seed",
        "42",
        "--output",
        str(out_file_kin),
        "--json",
    ])
    assert code_kin == 0
    payload_kin = json.loads(capsys.readouterr().out)
    assert payload_kin["backend"] == "mujoco"
    assert Path(payload_kin["file"]).is_file()


def test_cli_view_physics_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    called_with: dict[str, object] = {}

    def mock_viewer(
        seed: int = 42,
        episodes: int = 3,
        model_path: str | Path | None = None,
    ) -> None:
        called_with["seed"] = seed
        called_with["episodes"] = episodes
        called_with["model_path"] = model_path

    monkeypatch.setattr(
        "sim2real_engine.visualization.mujoco_gui.run_interactive_mujoco_viewer",
        mock_viewer,
    )

    code = main(["view-physics", "--seed", "123", "--episodes", "5"])
    assert code == 0
    assert called_with == {"seed": 123, "episodes": 5, "model_path": None}

    code_policy = main([
        "view-physics",
        "--seed",
        "42",
        "--episodes",
        "2",
        "--model-path",
        "checkpoints/bc_pick_place.pt",
    ])
    assert code_policy == 0
    assert called_with == {
        "seed": 42,
        "episodes": 2,
        "model_path": "checkpoints/bc_pick_place.pt",
    }


def test_policy_vs_expert_visualization_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt_path = Path("checkpoints/bc_pick_place.pt")
    if not ckpt_path.is_file():
        pytest.skip("Checkpoint 'checkpoints/bc_pick_place.pt' not found.")

    out_file = tmp_path / "policy_vis.html"
    code = main([
        "visualize",
        "--model-path",
        str(ckpt_path),
        "--backend",
        "mujoco_native",
        "--seed",
        "42",
        "--output",
        str(out_file),
        "--json",
    ])
    assert code == 0
    assert out_file.is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    html_text = out_file.read_text(encoding="utf-8")
    assert "Policy vs. Privileged Expert 3D Evaluation" in html_text
    assert "plot-2d-telemetry" in html_text
    assert "LATCHED (weld)" in html_text
    assert "Cross-Table Payload Transport" in html_text
    assert "https://github.com/LeanoA/sim2real-data-engine" in html_text


def test_multi_model_visualization_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    c1 = Path("checkpoints/bc_pick_place.pt")
    c2 = Path("results/bc_cloud_policy.pt")
    if not c1.is_file() or not c2.is_file():
        pytest.skip("Both checkpoints are required for multi-model verification.")

    out_file = tmp_path / "multi_vis.html"
    code = main([
        "visualize",
        "--model-path",
        str(c1),
        "--compare-model-path",
        str(c2),
        "--backend",
        "mujoco_native",
        "--seed",
        "42",
        "--output",
        str(out_file),
        "--json",
    ])
    assert code == 0
    assert out_file.is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    html_text = out_file.read_text(encoding="utf-8")
    assert "model-switcher-bar" in html_text
    assert "tab-canonical" in html_text
    assert "tab-cloud" in html_text
    assert "tab-compare" in html_text
    assert "canonical_45ep" in html_text
    assert "cloud_45ep" in html_text
    assert "Client-Side Static Store" in html_text



