"""Visualization module for 3D kinematics trajectories and Sim-to-Sim benchmarking."""

from __future__ import annotations

from sim2real_engine.visualization.dashboard import generate_html_dashboard
from sim2real_engine.visualization.data import (
    ArmLinkPoints,
    EpisodeVisualizationData,
    Sim2SimComparisonData,
    collect_sim2sim_comparison_data,
    collect_trajectory_data,
    compute_arm_links,
)
from sim2real_engine.visualization.mujoco_gui import run_interactive_mujoco_viewer

__all__ = [
    "ArmLinkPoints",
    "EpisodeVisualizationData",
    "Sim2SimComparisonData",
    "collect_sim2sim_comparison_data",
    "collect_trajectory_data",
    "compute_arm_links",
    "generate_html_dashboard",
    "run_interactive_mujoco_viewer",
]
