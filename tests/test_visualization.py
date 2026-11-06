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
    q_home = [0.0, 0.55, -1.10, 0.0, 0.55, 0.0]
    links = compute_arm_links(q_home)

    assert isinstance(links, ArmLinkPoints)
    assert links.base == (0.0, 0.0, 0.0)
    assert links.shoulder == (0.0, 0.0, 0.22)
    assert len(links.elbow) == 3
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
    assert payload["seed"] == 42
    assert Path(payload["file"]).is_file()
