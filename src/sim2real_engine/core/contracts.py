"""Strictly typed data contracts for SE(3) kinematics, randomization, and telemetry."""

from __future__ import annotations

import math
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

BackendName = Literal["mujoco", "genesis", "isaac_lab"]


class SE3Pose(BaseModel):
    """Rigid-body pose in SE(3) parameterized by position (m) and unit quaternion (w, x, y, z)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Cartesian coordinates (x, y, z) in meters.",
    )
    quaternion_wxyz: tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="Unit quaternion (qw, qx, qy, qz) with scalar-first convention.",
    )

    @model_validator(mode="after")
    def _validate_unit_quaternion(self) -> Self:
        qw, qx, qy, qz = self.quaternion_wxyz
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
            msg = f"Quaternion must have unit L2 norm (1.0 +/- 1e-4); got {norm:.6f}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_unnormalized(
        cls,
        position: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> SE3Pose:
        """Normalize a non-zero quaternion and construct a valid SE3Pose."""
        qw, qx, qy, qz = quaternion_wxyz
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm < 1e-12 or not math.isfinite(norm):
            msg = "Cannot normalize degenerate zero-norm quaternion."
            raise ValueError(msg)
        normalized = (qw / norm, qx / norm, qy / norm, qz / norm)
        return cls(position=position, quaternion_wxyz=normalized)

    def to_homogeneous_matrix(self) -> NDArray[np.float64]:
        """Convert SE(3) pose to a 4x4 homogeneous transformation matrix."""
        qw, qx, qy, qz = self.quaternion_wxyz
        x, y, z = self.position
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        matrix[0, 1] = 2.0 * (qx * qy - qz * qw)
        matrix[0, 2] = 2.0 * (qx * qz + qy * qw)
        matrix[1, 0] = 2.0 * (qx * qy + qz * qw)
        matrix[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        matrix[1, 2] = 2.0 * (qy * qz - qx * qw)
        matrix[2, 0] = 2.0 * (qx * qz - qy * qw)
        matrix[2, 1] = 2.0 * (qy * qz + qx * qw)
        matrix[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return matrix


class JointState(BaseModel):
    """Articulated manipulator joint state snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions_rad: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Joint angles in radians.",
    )
    velocities_rad_s: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Joint angular velocities in radians/second.",
    )
    torques_nm: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Applied or measured joint torques in Newton-meters.",
    )

    @model_validator(mode="after")
    def _validate_dof_dimensions(self) -> Self:
        dof = len(self.positions_rad)
        if len(self.velocities_rad_s) != dof or len(self.torques_nm) != dof:
            msg = (
                "JointState vectors must share identical DOF length; "
                f"got positions={dof}, velocities={len(self.velocities_rad_s)}, "
                f"torques={len(self.torques_nm)}"
            )
            raise ValueError(msg)
        return self

    @property
    def dof(self) -> int:
        """Return number of articulated degrees of freedom."""
        return len(self.positions_rad)


class DomainRandomizationConfig(BaseModel):
    """Seeded visual and physical perturbation bounds for synthetic rollout generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=42, ge=0, description="Deterministic RNG seed.")
    mass_scale_range: tuple[float, float] = Field(
        default=(0.85, 1.15),
        description="Multiplicative payload/link mass scaling bounds (min, max).",
    )
    friction_scale_range: tuple[float, float] = Field(
        default=(0.70, 1.30),
        description="Multiplicative Coulomb surface friction scaling bounds (min, max).",
    )
    damping_scale_range: tuple[float, float] = Field(
        default=(0.90, 1.10),
        description="Multiplicative joint damping scaling bounds (min, max).",
    )
    camera_position_jitter_m: float = Field(
        default=0.015,
        ge=0.0,
        le=0.25,
        description="Maximum uniform L-infinity camera position perturbation in meters.",
    )
    lighting_intensity_range: tuple[float, float] = Field(
        default=(0.60, 1.40),
        description="Relative scene illumination multiplier bounds (min, max).",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        ranges = {
            "mass_scale_range": self.mass_scale_range,
            "friction_scale_range": self.friction_scale_range,
            "damping_scale_range": self.damping_scale_range,
            "lighting_intensity_range": self.lighting_intensity_range,
        }
        for field_name, (low, high) in ranges.items():
            if low <= 0.0 or high <= 0.0:
                msg = f"{field_name} bounds must be strictly positive; got ({low}, {high})"
                raise ValueError(msg)
            if low > high:
                msg = f"{field_name} lower bound {low} exceeds upper bound {high}"
                raise ValueError(msg)
        return self


class EpisodeStepRecord(BaseModel):
    """Single synchronized timestep record for LeRobotDataset v2 export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int = Field(..., ge=0)
    step_index: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0)
    joint_state: JointState
    end_effector_pose: SE3Pose
    privileged_object_pose: SE3Pose
    action_joint_targets_rad: tuple[float, ...] = Field(..., min_length=1)
    is_terminal: bool = False

    @model_validator(mode="after")
    def _validate_action_dimension(self) -> Self:
        if len(self.action_joint_targets_rad) != self.joint_state.dof:
            msg = (
                f"Action target length ({len(self.action_joint_targets_rad)}) "
                f"must match JointState DOF ({self.joint_state.dof})"
            )
            raise ValueError(msg)
        return self


class SimulatorBackendSpec(BaseModel):
    """Capability descriptor for a registered physics simulation backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: BackendName
    execution_tier: Literal["local_cpu_arm64", "cloud_gpu_parallel", "cloud_gpu_rtx"]
    nominal_physics_hz: int = Field(..., gt=0)
    supports_ray_traced_rgbd: bool
    requires_nvidia_gpu: bool
    description: str


DEFAULT_BACKENDS: dict[BackendName, SimulatorBackendSpec] = {
    "mujoco": SimulatorBackendSpec(
        name="mujoco",
        execution_tier="local_cpu_arm64",
        nominal_physics_hz=1000,
        supports_ray_traced_rgbd=False,
        requires_nvidia_gpu=False,
        description=(
            "Native Apple Silicon ARM64 & x86 CPU physics engine for deterministic "
            "kinematics, privileged IK validation, and zero-cost CI testing."
        ),
    ),
    "genesis": SimulatorBackendSpec(
        name="genesis",
        execution_tier="cloud_gpu_parallel",
        nominal_physics_hz=1000,
        supports_ray_traced_rgbd=True,
        requires_nvidia_gpu=True,
        description=(
            "High-throughput parallel GPU multi-physics backend for large-scale "
            "contact dynamics sweeps and domain randomization benchmarks."
        ),
    ),
    "isaac_lab": SimulatorBackendSpec(
        name="isaac_lab",
        execution_tier="cloud_gpu_rtx",
        nominal_physics_hz=500,
        supports_ray_traced_rgbd=True,
        requires_nvidia_gpu=True,
        description=(
            "Headless NVIDIA Omniverse / Isaac Lab RTX backend for photorealistic "
            "RGB-D synthetic dataset generation and Sim-to-Real transfer."
        ),
    ),
}
