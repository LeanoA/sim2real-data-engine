"""Pluggable physics simulation backend adapters and privileged trajectory experts."""

from __future__ import annotations

from sim2real_engine.simulators.base import (
    ExpertStage,
    PrivilegedPickPlaceExpert,
    RolloutSummary,
    SimulatorBackend,
)
from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter

__all__ = [
    "ExpertStage",
    "MuJoCoKinematicAdapter",
    "PrivilegedPickPlaceExpert",
    "RolloutSummary",
    "SimulatorBackend",
]
