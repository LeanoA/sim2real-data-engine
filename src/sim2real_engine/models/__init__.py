"""Declarative MJCF model definitions and loaders for MuJoCo physics backends."""

from __future__ import annotations

from pathlib import Path

import mujoco

_MODEL_PATH = Path(__file__).parent / "manipulator_6dof.xml"


def get_robot_mjcf_path() -> Path:
    """Return absolute filesystem path to the 6-DOF manipulator MJCF model."""
    return _MODEL_PATH


def load_robot_mjcf() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load and compile the 6-DOF manipulator MJCF XML model into MuJoCo C structures."""
    xml_str = _MODEL_PATH.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    return model, data
